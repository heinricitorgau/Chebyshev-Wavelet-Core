"""Volatility, drawdown, position-scale, and cost-aware risk controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .execution_guard import PreTradeCostGuard


Array = np.ndarray

# Relative slack used when comparing a loss fraction against a stop threshold.
# Sized to absorb double-precision representation error only; it is far below
# any exchange tick and therefore cannot fire a stop the market has not reached.
_STOP_LOSS_TOLERANCE = 1e-12

# Relative slack for the settlement comparison.  At 1e-9 of the available
# balance this is a fraction of the smallest currency unit even on a very large
# account, so it cannot approve a genuine overdraft; it exists because a budget
# derived as ``cash * weight / gross`` can land a few units in the last place
# above the cash it was derived from.
_SETTLEMENT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class RiskConfig:
    """Risk limits expressed in portfolio rather than instrument units."""

    target_annual_volatility: float = 0.10
    volatility_lookback: int = 21
    periods_per_year: int = 252
    max_leverage: float = 2.0
    maximum_drawdown: float = 0.15
    minimum_stop_drawdown: float = 0.05
    drawdown_recovery_scale: float = 0.20

    def __post_init__(self) -> None:
        if self.target_annual_volatility <= 0.0 or self.volatility_lookback < 2:
            raise ValueError("target volatility must be positive and lookback must be at least two")
        if self.periods_per_year <= 0 or self.max_leverage <= 0.0:
            raise ValueError("periods_per_year and max_leverage must be positive")
        if not 0.0 < self.minimum_stop_drawdown <= self.maximum_drawdown < 1.0:
            raise ValueError("drawdown limits must satisfy 0 < minimum <= maximum < 1")
        if not 0.0 <= self.drawdown_recovery_scale <= 1.0:
            raise ValueError("drawdown_recovery_scale must lie in [0, 1]")


@dataclass(frozen=True)
class RiskDecision:
    """Complete risk decision, including each scaling layer and its rationale."""

    target_weights: Array
    guarded_weights: Array
    risk_adjusted_weights: Array
    realized_annual_volatility: float
    volatility_scale: float
    drawdown: float
    dynamic_stop_drawdown: float
    drawdown_scale: float
    stopped_out: bool
    cost_approved: Array


@dataclass(frozen=True)
class SettlementCheck:
    """T+2 settlement result; unsettled sale proceeds are excluded from buying power."""

    settled_cash: float
    unsettled_sale_proceeds: float
    pending_buy_notional: float
    requested_notional: float
    available_settled_cash: float
    approved: bool


@dataclass(frozen=True)
class ShareQuantity:
    """Board-lot and odd-lot decomposition for a cash-constrained order."""

    board_lots: int
    board_lot_shares: int
    odd_lot_shares: int
    total_shares: int
    notional: float


class RiskManager:
    """Apply independent risk layers after signal generation and cost gating.

    ``assess`` is intentionally stateless.  The caller supplies the current
    equity curve and return history, making a replay deterministic and avoiding
    hidden process state in a live trading recovery.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def assess(
        self,
        target_weights: Array,
        portfolio_returns: Array,
        equity_curve: Array,
        expected_edge_bps: Array,
        projected_turnover: Array,
        cost_guard: PreTradeCostGuard,
    ) -> RiskDecision:
        """Return cost-gated and risk-scaled target weights.

        The cost guard is applied before leverage scaling.  This prevents a
        volatility target from reviving a trade that failed the economic hurdle.
        A dynamic drawdown limit tightens when realized volatility exceeds the
        portfolio target; breaching that limit liquidates the proposed book.
        """
        weights = np.asarray(target_weights, dtype=float).reshape(-1)
        edge = np.broadcast_to(np.asarray(expected_edge_bps, dtype=float), weights.shape)
        turnover = np.broadcast_to(np.asarray(projected_turnover, dtype=float), weights.shape)
        if not np.all(np.isfinite(weights)):
            raise ValueError("target_weights must be finite")
        if weights.size == 0:
            raise ValueError("target_weights cannot be empty")

        realized_volatility = self._realized_volatility(portfolio_returns)
        volatility_scale = self._volatility_scale(realized_volatility)
        drawdown = self._drawdown(equity_curve)
        dynamic_stop = self._dynamic_stop_drawdown(realized_volatility)
        stopped_out = drawdown <= -dynamic_stop
        drawdown_scale = 0.0 if stopped_out else self._drawdown_scale(drawdown, dynamic_stop)

        guard = cost_guard.evaluate(weights, edge, turnover)
        guarded_weights = np.asarray(guard["signal"], dtype=float)
        risk_adjusted = guarded_weights * volatility_scale * drawdown_scale
        if stopped_out:
            risk_adjusted = np.zeros_like(risk_adjusted)
        return RiskDecision(
            target_weights=weights,
            guarded_weights=guarded_weights,
            risk_adjusted_weights=risk_adjusted,
            realized_annual_volatility=realized_volatility,
            volatility_scale=volatility_scale,
            drawdown=drawdown,
            dynamic_stop_drawdown=dynamic_stop,
            drawdown_scale=drawdown_scale,
            stopped_out=stopped_out,
            cost_approved=np.asarray(guard["approved"], dtype=bool),
        )

    @staticmethod
    def check_t_plus_two_settlement(
        settled_cash: float,
        requested_notional: float,
        pending_buy_notional: float = 0.0,
        unsettled_sale_proceeds: float = 0.0,
    ) -> SettlementCheck:
        """Approve a cash purchase only from settled funds net of pending buys.

        Sale proceeds still inside their T+2 settlement window are preserved in
        the returned record for audit but are never counted as available cash.

        The approval comparison carries a relative tolerance.  A request built
        as a share of the very cash being checked -- ``cash * weight / gross`` --
        can exceed it by a few units in the last place, which rejected a fully
        funded order for an overdraft of 5e-10 TWD before this was added.
        """
        values = np.asarray(
            [settled_cash, requested_notional, pending_buy_notional, unsettled_sale_proceeds], dtype=float
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("settlement values must be finite and non-negative")
        available = max(0.0, settled_cash - pending_buy_notional)
        tolerance = _SETTLEMENT_TOLERANCE * max(1.0, available)
        return SettlementCheck(
            settled_cash=float(settled_cash),
            unsettled_sale_proceeds=float(unsettled_sale_proceeds),
            pending_buy_notional=float(pending_buy_notional),
            requested_notional=float(requested_notional),
            available_settled_cash=float(available),
            approved=bool(requested_notional <= available + tolerance),
        )

    @staticmethod
    def calculate_share_quantity(
        available_cash: float,
        reference_price: float,
        unit_shares: int = 1000,
        allow_odd_lots: bool = True,
    ) -> ShareQuantity:
        """Convert available cash into board lots and optional odd lots."""
        if not np.isfinite(available_cash) or available_cash < 0.0:
            raise ValueError("available_cash must be finite and non-negative")
        if not np.isfinite(reference_price) or reference_price <= 0.0:
            raise ValueError("reference_price must be finite and positive")
        if not isinstance(unit_shares, int) or unit_shares < 1:
            raise ValueError("unit_shares must be a positive integer")
        affordable = int(np.floor(available_cash / reference_price))
        board_lots = affordable // unit_shares
        board_shares = board_lots * unit_shares
        odd_shares = affordable - board_shares if allow_odd_lots else 0
        total = board_shares + odd_shares
        return ShareQuantity(
            board_lots=board_lots,
            board_lot_shares=board_shares,
            odd_lot_shares=odd_shares,
            total_shares=total,
            notional=float(total * reference_price),
        )

    @staticmethod
    def breach_stop_loss(entry_price: float, current_price: float, stop_loss_pct: float = 0.07) -> bool:
        """Return whether a long position has breached its percentage stop-loss.

        The comparison is made on the realized loss fraction rather than on a
        reconstructed trigger price.  Writing the threshold as
        ``entry_price * (1 - stop_loss_pct)`` is subtly wrong at the boundary:
        ``1.0 - 0.07`` is ``0.9299999999999999`` in binary, one unit in the last
        place below the correctly rounded ``0.93``, so the reconstructed trigger
        sits just under the price an exchange actually quotes.  A 7% stop on a
        NT$10 entry then failed to fire at NT$9.30 -- the quoted stop price
        itself -- and likewise at 128/119.04 and 4321/4018.53.

        ``_STOP_LOSS_TOLERANCE`` absorbs that representation error.  At 1e-12 of
        the entry price it is many orders of magnitude finer than any tick size,
        so it cannot trigger a stop that market data would not, while making the
        quoted boundary price behave as the caller intends.
        """
        if entry_price <= 0.0 or current_price <= 0.0:
            raise ValueError("entry_price and current_price must be positive")
        if not 0.0 < stop_loss_pct < 1.0:
            raise ValueError("stop_loss_pct must lie between zero and one")
        loss_fraction = (entry_price - current_price) / entry_price
        return bool(loss_fraction >= stop_loss_pct - _STOP_LOSS_TOLERANCE)

    def _realized_volatility(self, portfolio_returns: Array) -> float:
        returns = np.asarray(portfolio_returns, dtype=float).reshape(-1)
        returns = returns[np.isfinite(returns)]
        if returns.size < self.config.volatility_lookback:
            return np.nan
        trailing = returns[-self.config.volatility_lookback :]
        return float(np.std(trailing, ddof=1) * np.sqrt(self.config.periods_per_year))

    def _volatility_scale(self, realized_volatility: float) -> float:
        if not np.isfinite(realized_volatility) or realized_volatility <= 0.0:
            return 0.0
        return float(min(self.config.max_leverage, self.config.target_annual_volatility / realized_volatility))

    @staticmethod
    def _drawdown(equity_curve: Array) -> float:
        equity = np.asarray(equity_curve, dtype=float).reshape(-1)
        equity = equity[np.isfinite(equity)]
        if equity.size == 0 or np.any(equity <= 0.0):
            raise ValueError("equity_curve must contain positive finite values")
        return float(equity[-1] / np.max(equity) - 1.0)

    def _dynamic_stop_drawdown(self, realized_volatility: float) -> float:
        """Tighten the drawdown stop when realized risk exceeds the target."""
        if not np.isfinite(realized_volatility) or realized_volatility <= 0.0:
            return self.config.minimum_stop_drawdown
        risk_ratio = self.config.target_annual_volatility / realized_volatility
        return float(
            np.clip(
                self.config.maximum_drawdown * risk_ratio,
                self.config.minimum_stop_drawdown,
                self.config.maximum_drawdown,
            )
        )

    def _drawdown_scale(self, drawdown: float, stop_drawdown: float) -> float:
        if drawdown >= 0.0:
            return 1.0
        remaining_capacity = max(0.0, 1.0 - abs(drawdown) / stop_drawdown)
        return float(self.config.drawdown_recovery_scale + (1.0 - self.config.drawdown_recovery_scale) * remaining_capacity)
