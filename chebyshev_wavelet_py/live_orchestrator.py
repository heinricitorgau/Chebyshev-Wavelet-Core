"""Causal, cost-gated asynchronous orchestration for IB Paper Trading."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .execution_guard import PreTradeCostGuard
from .ib_adapter import IBPaperAdapter
from .signal_generator import CausalWaveletSignalGenerator


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstrumentSpec:
    """Trading metadata for one signal column and one explicit IB contract."""

    name: str
    contract: Any
    target_quantity: float

    def __post_init__(self) -> None:
        if self.target_quantity <= 0.0:
            raise ValueError("target_quantity must be positive")


@dataclass(frozen=True)
class OrchestratorConfig:
    """Scheduling, market-data, and execution settings for the live loop."""

    interval_seconds: float = 300.0
    history_duration: str = "180 D"
    bar_size: str = "1 day"
    use_rth: bool = True
    order_slippage_bps: float = 2.9
    submit_orders: bool = False
    edge_bps_per_zscore: float = 2.9

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0.0:
            raise ValueError("interval_seconds must be positive")
        if self.order_slippage_bps < 0.0:
            raise ValueError("order_slippage_bps cannot be negative")
        if self.edge_bps_per_zscore <= 0.0:
            raise ValueError("edge_bps_per_zscore must be positive")


class LiveOrchestrator:
    """Fetch, score, guard, and optionally submit Paper Trading orders.

    The loop is deliberately conservative.  It fetches a complete trailing bar
    history at each decision point, derives causal features from returns ending
    at the last completed bar, and never converts a rejected signal into a
    position change.  Order submission is disabled by default.
    """

    def __init__(
        self,
        adapter: IBPaperAdapter,
        signal_generator: CausalWaveletSignalGenerator,
        cost_guard: PreTradeCostGuard,
        instruments: Sequence[InstrumentSpec],
        config: OrchestratorConfig | None = None,
        expected_edge_model: Callable[[np.ndarray, dict[str, np.ndarray]], np.ndarray] | None = None,
    ) -> None:
        if not instruments:
            raise ValueError("at least one instrument is required")
        self.adapter = adapter
        self.signal_generator = signal_generator
        self.cost_guard = cost_guard
        self.instruments = tuple(instruments)
        self.config = config or OrchestratorConfig()
        self.expected_edge_model = expected_edge_model or self._default_edge_model

    async def run_forever(self, stop_event: asyncio.Event) -> None:
        """Run until ``stop_event`` is set, logging and surviving iteration failures."""
        await self.adapter.connect()
        try:
            while not stop_event.is_set():
                try:
                    await self.run_once()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    LOGGER.exception("live orchestration iteration failed; positions left unchanged")
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self.config.interval_seconds)
                except asyncio.TimeoutError:
                    pass
        finally:
            await self.adapter.disconnect()

    async def run_once(self) -> list[dict[str, Any]]:
        """Execute one causal decision cycle and return its auditable decisions."""
        bars = await asyncio.gather(
            *(
                self.adapter.historical_bars(
                    instrument.contract,
                    duration=self.config.history_duration,
                    bar_size=self.config.bar_size,
                    use_rth=self.config.use_rth,
                )
                for instrument in self.instruments
            )
        )
        close_matrix = self._aligned_close_matrix(bars)
        returns = np.diff(np.log(close_matrix), axis=0)
        features = self.signal_generator.transform(returns)
        latest_signal = features["signal"][-1]
        latest_score = features["scores"][-1]
        if not np.all(np.isfinite(latest_score)):
            LOGGER.info("latest window is incomplete; no orders submitted")
            return []

        expected_edge = self.expected_edge_model(latest_score, features)
        latest_price = close_matrix[-1]
        net_liquidation, positions = await asyncio.gather(
            self.adapter.net_liquidation(), self.adapter.positions()
        )
        if net_liquidation <= 0.0:
            raise RuntimeError("NetLiquidation must be positive before sizing trades")
        current_quantities = self._current_quantities(positions)
        target_quantities = latest_signal * np.asarray(
            [instrument.target_quantity for instrument in self.instruments], dtype=float
        )
        deltas = target_quantities - current_quantities
        projected_turnover = np.abs(deltas) * latest_price / net_liquidation
        guard = self.cost_guard.evaluate(latest_signal, expected_edge, projected_turnover)

        decisions: list[dict[str, Any]] = []
        for index, instrument in enumerate(self.instruments):
            decision = {
                "instrument": instrument.name,
                "signal": float(latest_signal[index]),
                "score": float(latest_score[index]),
                "expected_edge_bps": float(guard["expected_edge_bps"][index]),
                "required_edge_bps": float(guard["required_edge_bps"][index]),
                "projected_turnover": float(projected_turnover[index]),
                "approved": bool(guard["approved"][index]),
                "submitted": False,
            }
            if not guard["approved"][index] or np.isclose(deltas[index], 0.0):
                LOGGER.info("signal intercepted: %s", decision)
                decisions.append(decision)
                continue

            side = "BUY" if deltas[index] > 0.0 else "SELL"
            quantity = abs(float(deltas[index]))
            if self.config.submit_orders:
                acknowledgement = await self.adapter.place_limit_order(
                    instrument.contract,
                    side,
                    quantity,
                    float(latest_price[index]),
                    slippage_bps=self.config.order_slippage_bps,
                )
                decision["submitted"] = True
                decision["order_id"] = acknowledgement.order_id
                decision["limit_price"] = acknowledgement.limit_price
                LOGGER.warning("paper order submitted: %s", decision)
            else:
                LOGGER.warning("paper order simulation only (submit_orders=False): %s", decision)
            decisions.append(decision)
        return decisions

    @staticmethod
    def _aligned_close_matrix(bars: Sequence[Any]) -> np.ndarray:
        """Right-align close histories; no value is forward-filled across assets."""
        common_length = min(bar.close.size for bar in bars)
        if common_length < 2:
            raise RuntimeError("insufficient completed bars to construct returns")
        matrix = np.column_stack([bar.close[-common_length:] for bar in bars])
        if not np.all(np.isfinite(matrix)) or np.any(matrix <= 0.0):
            raise RuntimeError("market data contain invalid close prices")
        return matrix

    def _current_quantities(self, positions: Sequence[Any]) -> np.ndarray:
        """Match positions by conId when available, otherwise by symbol and currency."""
        quantities = np.zeros(len(self.instruments), dtype=float)
        for index, instrument in enumerate(self.instruments):
            for position in positions:
                if self._same_contract(instrument.contract, position.contract):
                    quantities[index] += position.quantity
        return quantities

    @staticmethod
    def _same_contract(left: Any, right: Any) -> bool:
        left_id = getattr(left, "conId", 0)
        right_id = getattr(right, "conId", 0)
        if left_id and right_id:
            return left_id == right_id
        return (
            getattr(left, "symbol", None) == getattr(right, "symbol", None)
            and getattr(left, "currency", None) == getattr(right, "currency", None)
            and getattr(left, "secType", None) == getattr(right, "secType", None)
        )

    def _default_edge_model(
        self, scores: np.ndarray, _: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Map standardized scores to a transparent, conservative bps forecast."""
        return np.abs(np.asarray(scores, dtype=float)) * self.config.edge_bps_per_zscore
