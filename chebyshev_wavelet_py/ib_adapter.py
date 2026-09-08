"""Async Interactive Brokers adapter for Paper Trading workflows.

``ib_insync`` is intentionally an optional dependency.  The numerical package
remains NumPy-only until an application constructs :class:`IBPaperAdapter`.
The adapter defaults to TWS Paper Trading port 7497 and rejects non-paper
ports unless the caller explicitly opts out of the guard.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

try:
    from ib_insync import IB, LimitOrder
except ImportError:  # pragma: no cover - exercised only in deployments without ib_insync.
    IB = None
    LimitOrder = None


LOGGER = logging.getLogger(__name__)
PAPER_PORTS = frozenset({7497, 4002})


class IBDependencyError(RuntimeError):
    """Raised when the optional IB client library is unavailable."""


class IBConnectionError(RuntimeError):
    """Raised when an adapter operation requires an active IB connection."""


@dataclass(frozen=True)
class HistoricalBars:
    """Columnar bar data returned by :meth:`IBPaperAdapter.historical_bars`."""

    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


@dataclass(frozen=True)
class PositionSnapshot:
    """Minimal account position representation independent of ib_insync types."""

    contract: Any
    quantity: float
    average_cost: float


@dataclass(frozen=True)
class OrderSubmission:
    """Auditable acknowledgement of a limit-order submission."""

    order_id: int
    status: str
    side: str
    quantity: float
    limit_price: float


class IBPaperAdapter:
    """Small asynchronous facade over IB Gateway/TWS Paper Trading.

    The adapter does not manufacture contracts.  Applications should construct
    and retain explicit ``ib_insync.Contract`` instances so exchange, currency,
    and primary-exchange decisions remain visible at the call site.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 71,
        account: str | None = None,
        paper_only: bool = True,
        ib_client: Any | None = None,
    ) -> None:
        if paper_only and port not in PAPER_PORTS:
            raise ValueError(
                f"port {port} is not a recognized Paper Trading port; "
                "set paper_only=False only after an explicit production review"
            )
        if ib_client is None and IB is None:
            raise IBDependencyError(
                "Interactive Brokers support requires the optional 'ib_insync' package."
            )
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account = account
        self.paper_only = paper_only
        self._ib = ib_client if ib_client is not None else IB()
        self._order_lock = asyncio.Lock()
        self._subscriptions: dict[int, Any] = {}

    @property
    def is_connected(self) -> bool:
        """Return whether the underlying IB client reports an active session."""
        return bool(self._ib.isConnected())

    async def connect(self, timeout: float = 10.0) -> None:
        """Connect asynchronously to TWS/IB Gateway."""
        if self.is_connected:
            return
        try:
            await asyncio.wait_for(
                self._ib.connectAsync(self.host, self.port, clientId=self.client_id),
                timeout=timeout,
            )
        except Exception as exc:
            raise IBConnectionError(
                f"unable to connect to IB at {self.host}:{self.port} with client id {self.client_id}"
            ) from exc
        LOGGER.info("connected to IB Paper endpoint %s:%s", self.host, self.port)

    async def disconnect(self) -> None:
        """Cancel active real-time subscriptions and disconnect cleanly."""
        for subscription in tuple(self._subscriptions.values()):
            self._ib.cancelRealTimeBars(subscription)
        self._subscriptions.clear()
        if self.is_connected:
            self._ib.disconnect()
            LOGGER.info("disconnected from IB")

    async def historical_bars(
        self,
        contract: Any,
        duration: str = "90 D",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> HistoricalBars:
        """Fetch historical bars and return contiguous NumPy columns.

        ``endDateTime`` is intentionally blank so IB uses the time at request
        submission.  The method never requests future timestamps.
        """
        self._require_connection()
        try:
            bars = await self._ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=2,
                keepUpToDate=False,
            )
        except Exception:
            LOGGER.exception("historical-bar request failed for %s", contract)
            raise
        if not bars:
            raise RuntimeError(f"IB returned no historical bars for {contract}")
        return HistoricalBars(
            timestamp=_timestamp_array([bar.date for bar in bars]),
            open=np.asarray([bar.open for bar in bars], dtype=float),
            high=np.asarray([bar.high for bar in bars], dtype=float),
            low=np.asarray([bar.low for bar in bars], dtype=float),
            close=np.asarray([bar.close for bar in bars], dtype=float),
            volume=np.asarray([bar.volume for bar in bars], dtype=float),
        )

    def subscribe_realtime_bars(
        self,
        contract: Any,
        callback: Callable[[Any, bool], None] | None = None,
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> int:
        """Subscribe to IB's five-second bars and return a subscription id.

        IB's real-time-bar API supports five-second bars.  The optional callback
        receives the IB bar list and its ``has_new_bar`` flag; it should be fast
        and should delegate expensive work to an asyncio task.
        """
        self._require_connection()
        bars = self._ib.reqRealTimeBars(contract, 5, what_to_show, use_rth)
        if callback is not None:
            bars.updateEvent += callback
        subscription_id = id(bars)
        self._subscriptions[subscription_id] = bars
        return subscription_id

    def unsubscribe_realtime_bars(self, subscription_id: int) -> None:
        """Cancel a previously registered real-time-bar subscription."""
        bars = self._subscriptions.pop(subscription_id, None)
        if bars is not None:
            self._ib.cancelRealTimeBars(bars)

    async def net_liquidation(self) -> float:
        """Return base-currency NetLiquidation for the configured account."""
        self._require_connection()
        values = await self._ib.accountSummaryAsync(account=self.account or "")
        candidates = [
            value
            for value in values
            if value.tag == "NetLiquidation" and (not self.account or value.account == self.account)
        ]
        if not candidates:
            raise RuntimeError("IB account summary did not contain NetLiquidation")
        base_value = next((value for value in candidates if value.currency == "BASE"), candidates[0])
        return float(base_value.value)

    async def positions(self) -> list[PositionSnapshot]:
        """Return current positions for the configured account."""
        self._require_connection()
        positions = await self._ib.positionsAsync(account=self.account or "")
        return [
            PositionSnapshot(position.contract, float(position.position), float(position.avgCost))
            for position in positions
            if not self.account or position.account == self.account
        ]

    async def place_limit_order(
        self,
        contract: Any,
        side: str,
        quantity: float,
        reference_price: float,
        slippage_bps: float = 0.0,
        tif: str = "DAY",
    ) -> OrderSubmission:
        """Submit a marketable-within-cap limit order with an explicit slippage cap.

        A buy limit is ``reference * (1 + bps / 10000)`` and a sell limit is
        ``reference * (1 - bps / 10000)``.  The cap is recorded in the returned
        acknowledgement and never replaced by an unbounded market order.
        """
        self._require_connection()
        normalized_side = side.upper()
        if normalized_side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        if quantity <= 0.0 or not np.isfinite(quantity):
            raise ValueError("quantity must be finite and positive")
        if reference_price <= 0.0 or not np.isfinite(reference_price):
            raise ValueError("reference_price must be finite and positive")
        if slippage_bps < 0.0 or not np.isfinite(slippage_bps):
            raise ValueError("slippage_bps must be finite and non-negative")

        multiplier = 1.0 + slippage_bps / 10_000.0
        limit_price = reference_price * (multiplier if normalized_side == "BUY" else 2.0 - multiplier)
        async with self._order_lock:
            qualified = await self._ib.qualifyContractsAsync(contract)
            if not qualified:
                raise RuntimeError(f"unable to qualify contract for order: {contract}")
            order = LimitOrder(normalized_side, quantity, round(limit_price, 8), tif=tif)
            trade = self._ib.placeOrder(qualified[0], order)
        status = str(trade.orderStatus.status)
        LOGGER.warning(
            "submitted %s limit order: contract=%s quantity=%s limit=%s status=%s",
            normalized_side,
            qualified[0],
            quantity,
            limit_price,
            status,
        )
        return OrderSubmission(int(order.orderId), status, normalized_side, float(quantity), float(limit_price))

    async def cancel_order(self, order_id: int) -> None:
        """Cancel an open order by IB order id."""
        self._require_connection()
        for trade in self._ib.openTrades():
            if trade.order.orderId == order_id:
                self._ib.cancelOrder(trade.order)
                LOGGER.info("cancelled IB order %s", order_id)
                return
        raise KeyError(f"open IB order {order_id} was not found")

    def _require_connection(self) -> None:
        if not self.is_connected:
            raise IBConnectionError("IB operation requested without an active connection")


def _timestamp_array(values: Iterable[Any]) -> np.ndarray:
    """Convert IB date objects while retaining opaque values if parsing fails."""
    try:
        return np.asarray(list(values), dtype="datetime64[ns]")
    except (TypeError, ValueError):
        return np.asarray(list(values), dtype=object)
