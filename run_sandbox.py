"""Self-contained paper-trading decision sandbox.

The sandbox exercises the full decision path -- configuration, causal signals,
risk layers, Taiwan board-lot sizing, T+2 settlement, and durable logging --
against deterministic synthetic market data.  It never opens a broker
connection and never submits an order, so it can run unattended in CI.

Two invariants are enforced before any work begins and are not configurable
from the command line: ``paper_only`` must be true and ``submit_orders`` must
be false.  A configuration that violates either aborts the run.

Example::

    python run_sandbox.py --cycles 30 --symbols 2330,2317,2454,2412

Exit codes: ``0`` success, ``1`` unexpected failure, ``2`` invalid input or
refused configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from chebyshev_wavelet_py.config_loader import ConfigLoader, ConfigurationError, SystemConfig
from chebyshev_wavelet_py.data_store import DataStore
from chebyshev_wavelet_py.execution_guard import CostGuardConfig, PreTradeCostGuard
from chebyshev_wavelet_py.extended_signals import ExtendedSignalAnalyzer
from chebyshev_wavelet_py.risk_manager import RiskConfig, RiskManager


LOGGER = logging.getLogger("chebyshev_wavelet_py.sandbox")

# Opening reference prices, in TWD, for the default Taiwan-listed symbols.
_DEFAULT_PRICES: dict[str, float] = {
    "2330": 1085.0, "2317": 214.5, "2454": 1420.0, "2412": 128.5, "2881": 92.3,
}
_FALLBACK_PRICE = 100.0


@dataclass(frozen=True)
class SandboxAccount:
    """Simulated cash state, including funds still inside T+2 settlement."""

    settled_cash: float
    unsettled_sale_proceeds: float
    pending_buy_notional: float


@dataclass
class SandboxState:
    """Walk-forward state carried between cycles.

    ``strategy_equity`` is the sandbox's own realized equity, not the market's.
    Feeding a buy-and-hold market index to ``RiskManager.assess`` as if it were
    the strategy's equity curve is a subtle and expensive mistake: a random walk
    at 14% annual volatility spends much of its life more than 10% below its
    running maximum, so the drawdown stop fires permanently and the system takes
    no position at all.  ``assess`` separates ``portfolio_returns`` (volatility
    sizing) from ``equity_curve`` (drawdown) precisely so these can differ.
    """

    strategy_equity: list[float]
    previous_weights: np.ndarray
    entry_prices: dict[str, float]
    held: np.ndarray


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse non-execution parameters only; order submission is not settable."""
    parser = argparse.ArgumentParser(description="Run a simulated decision loop.")
    parser.add_argument("--config", default="config.json", help="Path to the JSON configuration.")
    parser.add_argument("--symbols", default="2330,2317,2454,2412", help="Comma-separated symbols.")
    parser.add_argument("--cycles", type=int, default=20, help="Number of decision cycles.")
    parser.add_argument("--history", type=int, default=180, help="Warm-up bars before cycle one.")
    parser.add_argument("--seed", type=int, default=20260908, help="Seed for reproducible data.")
    parser.add_argument("--settled-cash", type=float, default=3_000_000.0, help="Opening TWD cash.")
    parser.add_argument("--log-path", default=None, help="Override the decision log path.")
    parser.add_argument("--interval", type=float, default=0.0, help="Seconds to wait per cycle.")
    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def load_and_verify_config(path: str | Path) -> SystemConfig:
    """Load configuration and refuse anything that could reach a live venue."""
    config = ConfigLoader(path).load()
    if not config.system.paper_only:
        raise ConfigurationError("sandbox refuses to run with paper_only disabled")
    if config.system.submit_orders:
        raise ConfigurationError("sandbox refuses to run with submit_orders enabled")
    return config


class SyntheticMarket:
    """Deterministic geometric-random-walk prices with a mild common factor.

    A shared market component is included so the cross-sectional layers see
    correlated instruments rather than independent noise, which is the case
    that actually exercises the z-scoring.
    """

    def __init__(self, symbols: Sequence[str], history: int, seed: int) -> None:
        if history < 2:
            raise ValueError("history must be at least two bars")
        self.symbols = tuple(symbols)
        self._rng = np.random.default_rng(seed)
        opening = np.array(
            [_DEFAULT_PRICES.get(symbol, _FALLBACK_PRICE) for symbol in self.symbols]
        )
        shocks = self._draw(history - 1)
        path = np.vstack([np.zeros((1, len(self.symbols))), np.cumsum(shocks, axis=0)])
        self.prices = opening * np.exp(path)

    def _draw(self, rows: int) -> np.ndarray:
        market = self._rng.normal(0.0, 0.009, size=(rows, 1))
        idiosyncratic = self._rng.normal(0.0, 0.011, size=(rows, len(self.symbols)))
        return market + idiosyncratic

    def advance(self) -> np.ndarray:
        """Append one new bar and return the complete history."""
        self.prices = np.vstack([self.prices, self.prices[-1] * np.exp(self._draw(1))[0]])
        return self.prices


def simple_returns(prices: np.ndarray) -> np.ndarray:
    """Bar-over-bar simple returns, with a leading zero row for alignment."""
    return np.vstack([np.zeros((1, prices.shape[1])), prices[1:] / prices[:-1] - 1.0])


async def run_cycle(
    cycle: int,
    market: SyntheticMarket,
    analyzer: ExtendedSignalAnalyzer,
    risk_manager: RiskManager,
    cost_guard: PreTradeCostGuard,
    store: DataStore,
    config: SystemConfig,
    account: SandboxAccount,
    state: SandboxState,
) -> list[dict[str, Any]]:
    """Run one full decision cycle and persist every instrument's outcome."""
    prices = market.advance()
    latest = prices[-1]
    returns = simple_returns(prices)
    entry_prices = state.entry_prices

    # Mark the book held since the previous cycle against this bar's move.
    latest_asset_returns = returns[-1]
    realized = float(np.nansum(state.previous_weights * latest_asset_returns))
    state.strategy_equity.append(state.strategy_equity[-1] * (1.0 + realized))

    snapshot = analyzer.analyze(prices)
    signals = snapshot.signal[-1]

    # A cross or breakout is a momentary event, not a position.  Hold what was
    # opened until either a bearish signal or the per-instrument stop closes it,
    # so the book persists across bars and the stop-loss branch is reachable.
    desired = state.held.copy()
    stop_flags: dict[str, bool] = {}
    for index, symbol in enumerate(market.symbols):
        price = float(latest[index])
        entry = entry_prices.get(symbol)
        breached = (
            entry is not None
            and RiskManager.breach_stop_loss(entry, price, config.trading_rules.stop_loss_pct)
        )
        stop_flags[symbol] = breached
        if breached or signals[index] < 0.0:
            desired[index] = 0.0
            entry_prices.pop(symbol, None)
        elif signals[index] > 0.0 and entry is None:
            desired[index] = 1.0
            entry_prices[symbol] = price
    state.held = desired.copy()

    # Volatility sizing uses the tradable universe; the drawdown stop uses the
    # sandbox's own realized equity.  These are deliberately different series.
    portfolio_returns = np.nanmean(returns[1:], axis=1)
    equity_curve = np.asarray(state.strategy_equity, dtype=float)

    rules = config.trading_rules
    expected_edge_bps = np.abs(desired) * rules.max_break_even_bps * 1.5
    projected_turnover = np.full(desired.shape, rules.reference_turnover)

    decision = risk_manager.assess(
        target_weights=desired,
        portfolio_returns=portfolio_returns,
        equity_curve=equity_curve,
        expected_edge_bps=expected_edge_bps,
        projected_turnover=projected_turnover,
        cost_guard=cost_guard,
    )

    records: list[dict[str, Any]] = []
    gross = float(np.sum(np.abs(decision.risk_adjusted_weights))) or 1.0
    for index, symbol in enumerate(market.symbols):
        weight = float(decision.risk_adjusted_weights[index])
        price = float(latest[index])
        budget = account.settled_cash * abs(weight) / gross

        settlement = RiskManager.check_t_plus_two_settlement(
            settled_cash=account.settled_cash,
            requested_notional=budget,
            pending_buy_notional=account.pending_buy_notional,
            unsettled_sale_proceeds=account.unsettled_sale_proceeds,
        )
        # Size only what passed both gates.  Reporting a share count beside
        # ``approved: false`` invites a downstream consumer to act on a decision
        # the guards rejected, so an unapproved decision is sized at zero.
        cleared = bool(decision.cost_approved[index]) and settlement.approved
        quantity = RiskManager.calculate_share_quantity(
            available_cash=min(budget, settlement.available_settled_cash) if cleared else 0.0,
            reference_price=price,
            unit_shares=rules.unit_shares,
            allow_odd_lots=rules.allow_odd_lots,
        )
        stopped = stop_flags[symbol]

        record = {
            "cycle": cycle,
            "instrument": symbol,
            "price": round(price, 4),
            "signal": float(signals[index]),
            "desired_position": float(desired[index]),
            "score": weight,
            "expected_edge_bps": float(expected_edge_bps[index]),
            "required_edge_bps": rules.max_break_even_bps,
            "projected_turnover": rules.reference_turnover,
            "approved": cleared,
            "submitted": False,  # invariant: the sandbox never submits
            "board_lots": quantity.board_lots,
            "odd_lot_shares": quantity.odd_lot_shares,
            "total_shares": quantity.total_shares,
            "notional": round(quantity.notional, 2),
            "settlement_approved": settlement.approved,
            "available_settled_cash": round(settlement.available_settled_cash, 2),
            "stop_loss_breached": stopped,
            "realized_annual_volatility": _finite(decision.realized_annual_volatility),
            "volatility_scale": _finite(decision.volatility_scale),
            "drawdown": _finite(decision.drawdown),
            "stopped_out": decision.stopped_out,
            "strategy_equity": round(state.strategy_equity[-1], 6),
        }
        records.append(record)
        await store.append_decision_async(record)

    state.previous_weights = decision.risk_adjusted_weights.copy()

    active = sum(1 for record in records if record["total_shares"] > 0)
    LOGGER.info(
        "cycle %02d: vol=%s scale=%.3f equity=%.4f dd=%+.4f stopped=%s active=%d/%d",
        cycle,
        _format(decision.realized_annual_volatility),
        decision.volatility_scale,
        state.strategy_equity[-1],
        decision.drawdown,
        decision.stopped_out,
        active,
        len(records),
    )
    return records


async def async_main(arguments: argparse.Namespace) -> int:
    """Run the configured number of cycles and verify the log replays."""
    try:
        config = load_and_verify_config(arguments.config)
    except ConfigurationError as exc:
        LOGGER.error("configuration refused: %s", exc)
        return 2

    symbols = [symbol.strip() for symbol in arguments.symbols.split(",") if symbol.strip()]
    if len(symbols) < 2:
        LOGGER.error("provide at least two symbols so cross-sectional scores are meaningful")
        return 2
    if arguments.cycles < 1:
        LOGGER.error("--cycles must be at least one")
        return 2
    if arguments.settled_cash <= 0.0:
        LOGGER.error("--settled-cash must be positive")
        return 2

    indicators = config.indicators
    log_path = arguments.log_path or config.data_store.get("path", "logs/decisions.jsonl")

    LOGGER.info(
        "sandbox starting: paper_only=%s submit_orders=%s symbols=%s cycles=%d",
        config.system.paper_only, config.system.submit_orders, ",".join(symbols), arguments.cycles,
    )
    LOGGER.info("decision log: %s", log_path)

    market = SyntheticMarket(symbols, arguments.history, arguments.seed)
    analyzer = ExtendedSignalAnalyzer(
        fast_window=indicators.sma_fast_window,
        slow_window=indicators.sma_slow_window,
        support_resistance_window=indicators.support_resistance_window,
    )
    risk_manager = RiskManager(RiskConfig(**dict(config.risk)))
    cost_guard = PreTradeCostGuard(
        CostGuardConfig(
            break_even_spread_bps=config.trading_rules.max_break_even_bps,
            reference_turnover=config.trading_rules.reference_turnover,
        )
    )
    store = DataStore(log_path)
    account = SandboxAccount(
        settled_cash=arguments.settled_cash,
        unsettled_sale_proceeds=arguments.settled_cash * 0.25,
        pending_buy_notional=0.0,
    )
    state = SandboxState(
        strategy_equity=[1.0],
        previous_weights=np.zeros(len(symbols)),
        entry_prices={},
        held=np.zeros(len(symbols)),
    )

    written = 0
    for cycle in range(1, arguments.cycles + 1):
        records = await run_cycle(
            cycle, market, analyzer, risk_manager, cost_guard, store, config, account, state
        )
        written += len(records)
        if arguments.interval > 0.0:
            await asyncio.sleep(arguments.interval)

    replayed = list(store.replay())
    from_this_run = [record for record in replayed if "cycle" in record]
    LOGGER.info("wrote %d decisions; replayed %d records from %s", written, len(replayed), log_path)
    if len(from_this_run) < written:
        LOGGER.error(
            "replay lost records: wrote %d but only %d replayable", written, len(from_this_run)
        )
        return 1
    if any(record.get("submitted") for record in from_this_run):
        LOGGER.error("invariant violated: a decision was marked submitted")
        return 1

    approved = sum(1 for record in from_this_run[-written:] if record["approved"])
    traded = sum(1 for record in from_this_run[-written:] if record["total_shares"] > 0)
    stops = sum(1 for record in from_this_run[-written:] if record["stop_loss_breached"])
    LOGGER.info(
        "summary: %d decisions, %d cost-and-settlement approved, %d sized, %d stop-loss breaches",
        written, approved, traded, stops,
    )
    LOGGER.info("sandbox complete; no order was submitted at any point")
    return 0


def _finite(value: float) -> float | None:
    """Represent a non-finite diagnostic as JSON ``null`` rather than ``NaN``."""
    return float(value) if np.isfinite(value) else None


def _format(value: float) -> str:
    return f"{value:.4f}" if np.isfinite(value) else "n/a"


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    try:
        return asyncio.run(async_main(parse_arguments(argv)))
    except KeyboardInterrupt:
        LOGGER.info("sandbox interrupted; no order submission was enabled")
        return 130
    except Exception:
        LOGGER.exception("sandbox failed; no order submission was enabled")
        return 1


if __name__ == "__main__":
    sys.exit(main())
