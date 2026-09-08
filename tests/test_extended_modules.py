"""Unit tests for the extended trading modules.

Run with::

    python -m pytest tests/test_extended_modules.py -v

The emphasis is on the properties that silently break a trading system:
causality (no look-ahead), integer share arithmetic at lot boundaries,
floating-point behaviour exactly at a stop-loss threshold, and durability of
the decision log.  Tests that merely re-state an implementation are avoided.
"""

from __future__ import annotations

import asyncio
import json
import math
import warnings
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

import run_sandbox

from chebyshev_wavelet_py.config_loader import (
    ConfigLoader,
    ConfigurationError,
    load_config,
)
from chebyshev_wavelet_py.data_store import DataStore, DecisionRecord, DecisionStore
from chebyshev_wavelet_py.execution_guard import PreTradeCostGuard
from chebyshev_wavelet_py.extended_signals import (
    ExtendedSignalAnalyzer,
    FactorSpec,
    causal_rolling_volatility,
    combine_factors,
    cross_sectional_zscore,
    nonlinear_momentum_filter,
    volatility_adjusted_signal,
)
from chebyshev_wavelet_py.risk_manager import RiskConfig, RiskManager


REPO_ROOT = Path(__file__).resolve().parents[1]


# =====================================================================
# extended_signals: SMA crosses and causality
# =====================================================================


class TestCausalSMA:
    def test_sma_values_match_hand_calculation(self) -> None:
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        analyzer = ExtendedSignalAnalyzer(2, 3, support_resistance_window=2)
        snapshot = analyzer.analyze(prices)
        # fast (window 2) at t=1 is mean(1,2); at t=5 is mean(5,6)
        assert math.isnan(snapshot.fast_sma[0, 0])
        assert snapshot.fast_sma[1, 0] == pytest.approx(1.5)
        assert snapshot.fast_sma[5, 0] == pytest.approx(5.5)
        # slow (window 3) is undefined before t=2
        assert math.isnan(snapshot.slow_sma[1, 0])
        assert snapshot.slow_sma[2, 0] == pytest.approx(2.0)

    def test_golden_cross_fires_once_on_the_crossing_bar(self) -> None:
        # Falling then rising: the fast average crosses above the slow one once.
        prices = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 7.0, 9.0, 12.0, 16.0, 21.0])
        analyzer = ExtendedSignalAnalyzer(2, 4, support_resistance_window=3)
        snapshot = analyzer.analyze(prices)
        golden = snapshot.golden_cross[:, 0]
        fast = snapshot.fast_sma[:, 0]
        slow = snapshot.slow_sma[:, 0]
        fired = np.flatnonzero(golden)
        assert fired.size == 1, f"expected exactly one golden cross, got {fired}"
        index = int(fired[0])
        # A cross means the ordering strictly flipped between the two bars.
        assert fast[index] > slow[index]
        assert fast[index - 1] <= slow[index - 1]

    def test_death_cross_is_the_mirror_of_a_golden_cross(self) -> None:
        # The series must first establish fast-above-slow, then fall through it.
        # A monotonically falling series never crosses: it begins already below.
        prices = np.array([6.0, 7.0, 9.0, 12.0, 16.0, 21.0, 16.0, 12.0, 9.0, 7.0, 6.0, 5.0])
        analyzer = ExtendedSignalAnalyzer(2, 4, support_resistance_window=3)
        snapshot = analyzer.analyze(prices)
        death = snapshot.death_cross[:, 0]
        assert death.any(), "a rise followed by a fall must produce a death cross"
        index = int(np.flatnonzero(death)[0])
        assert snapshot.fast_sma[index, 0] < snapshot.slow_sma[index, 0]
        assert snapshot.fast_sma[index - 1, 0] >= snapshot.slow_sma[index - 1, 0]

    def test_a_monotonic_decline_never_crosses(self) -> None:
        """Regression guard: 'always below' is not the same as 'crossed below'."""
        prices = np.array([21.0, 16.0, 12.0, 9.0, 7.0, 6.0, 5.0, 4.0])
        snapshot = ExtendedSignalAnalyzer(2, 4, support_resistance_window=3).analyze(prices)
        assert not snapshot.death_cross.any()
        assert not snapshot.golden_cross.any()

    def test_cross_flags_are_mutually_exclusive(self) -> None:
        rng = np.random.default_rng(20260908)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=(200, 4)), axis=0))
        analyzer = ExtendedSignalAnalyzer(5, 20, support_resistance_window=10)
        snapshot = analyzer.analyze(prices)
        assert not np.any(snapshot.golden_cross & snapshot.death_cross)


class TestNoLookAheadBias:
    """Perturbing the future must not change a single past value."""

    @staticmethod
    def _prices(seed: int = 7) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, size=(120, 3)), axis=0))

    def test_analyze_outputs_are_bit_identical_on_past_rows(self) -> None:
        prices = self._prices()
        split = 80
        analyzer = ExtendedSignalAnalyzer(10, 30, support_resistance_window=15)

        base = analyzer.analyze(prices)
        perturbed_prices = prices.copy()
        rng = np.random.default_rng(99)
        # A violent, obviously detectable shock to every future bar.
        perturbed_prices[split:] *= np.exp(rng.normal(0.0, 0.5, size=(prices.shape[0] - split, 3)))
        perturbed = analyzer.analyze(perturbed_prices)

        for field in (
            "fast_sma", "slow_sma", "support", "resistance",
            "golden_cross", "death_cross", "breakout", "breakdown", "signal",
        ):
            a = getattr(base, field)[:split]
            b = getattr(perturbed, field)[:split]
            assert np.array_equal(a, b, equal_nan=True), f"look-ahead leak in '{field}'"

    def test_truncating_the_series_reproduces_the_prefix(self) -> None:
        """A causal indicator cannot depend on the series length."""
        prices = self._prices(seed=11)
        split = 70
        analyzer = ExtendedSignalAnalyzer(10, 30, support_resistance_window=15)
        full = analyzer.analyze(prices)
        prefix = analyzer.analyze(prices[:split])
        for field in ("fast_sma", "slow_sma", "support", "resistance", "signal"):
            assert np.array_equal(
                getattr(full, field)[:split], getattr(prefix, field), equal_nan=True
            ), f"'{field}' depends on total series length"

    def test_support_resistance_excludes_the_current_bar(self) -> None:
        # A new all-time high must be a breakout, which is only possible if the
        # resistance level was computed without today's price.
        prices = np.array([10.0, 11.0, 12.0, 13.0, 50.0])[:, None]
        analyzer = ExtendedSignalAnalyzer(1, 2, support_resistance_window=3)
        snapshot = analyzer.analyze(prices)
        assert snapshot.resistance[4, 0] == pytest.approx(13.0)
        assert bool(snapshot.breakout[4, 0])

    def test_rolling_volatility_is_causal(self) -> None:
        rng = np.random.default_rng(3)
        returns = rng.normal(0.0, 0.01, size=(90, 2))
        split = 60
        base = causal_rolling_volatility(returns, lookback=21)
        shocked = returns.copy()
        shocked[split:] += 5.0
        after = causal_rolling_volatility(shocked, lookback=21)
        assert np.array_equal(base[:split], after[:split], equal_nan=True)

    def test_momentum_filter_is_causal(self) -> None:
        rng = np.random.default_rng(5)
        returns = rng.normal(0.0, 0.01, size=(80, 4))
        split = 50
        base = nonlinear_momentum_filter(returns, lookback=21)
        shocked = returns.copy()
        shocked[split:] -= 3.0
        after = nonlinear_momentum_filter(shocked, lookback=21)
        assert np.array_equal(base[:split], after[:split], equal_nan=True)


class TestSignalHelpers:
    def test_zscore_rows_are_standardized(self) -> None:
        values = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        z = cross_sectional_zscore(values)
        assert z.mean(axis=1) == pytest.approx([0.0, 0.0])
        assert np.allclose(z[0], z[1]), "z-scores must be scale invariant"

    def test_zscore_of_a_constant_row_is_zero_not_nan(self) -> None:
        z = cross_sectional_zscore(np.array([[5.0, 5.0, 5.0]]))
        assert np.all(z == 0.0)

    def test_combine_factors_respects_signed_weights(self) -> None:
        factor = np.array([[1.0, 2.0, 3.0]])
        combined = combine_factors(
            {"a": factor, "b": factor},
            (FactorSpec("a", 1.0), FactorSpec("b", -1.0)),
        )
        assert np.allclose(combined, 0.0), "opposite weights must cancel"

    def test_combine_factors_rejects_a_missing_factor(self) -> None:
        with pytest.raises(KeyError):
            combine_factors({"a": np.ones((2, 2))}, (FactorSpec("missing", 1.0),))

    def test_volatility_adjusted_signal_reduces_exposure_when_vol_is_high(self) -> None:
        quiet = np.full((40, 1), 0.001)
        loud = np.full((40, 1), 0.05)
        signal = np.ones((40, 1))
        quiet_scaled = volatility_adjusted_signal(signal, quiet, lookback=21)
        loud_scaled = volatility_adjusted_signal(signal, loud, lookback=21)
        assert quiet_scaled[-1, 0] >= loud_scaled[-1, 0]


# =====================================================================
# risk_manager: settlement, share arithmetic, stop-loss
# =====================================================================


class TestSettlement:
    def test_unsettled_sale_proceeds_are_not_spendable(self) -> None:
        check = RiskManager.check_t_plus_two_settlement(
            settled_cash=100_000.0,
            requested_notional=150_000.0,
            unsettled_sale_proceeds=80_000.0,
        )
        assert check.available_settled_cash == pytest.approx(100_000.0)
        assert not check.approved, "T+2 proceeds must not fund a purchase today"
        assert check.unsettled_sale_proceeds == pytest.approx(80_000.0)

    def test_pending_buys_reduce_available_cash(self) -> None:
        check = RiskManager.check_t_plus_two_settlement(
            settled_cash=100_000.0, requested_notional=60_000.0, pending_buy_notional=50_000.0
        )
        assert check.available_settled_cash == pytest.approx(50_000.0)
        assert not check.approved

    def test_request_exactly_equal_to_available_cash_is_approved(self) -> None:
        check = RiskManager.check_t_plus_two_settlement(
            settled_cash=50_000.0, requested_notional=50_000.0
        )
        assert check.approved

    def test_available_cash_never_goes_negative(self) -> None:
        check = RiskManager.check_t_plus_two_settlement(
            settled_cash=10_000.0, requested_notional=1.0, pending_buy_notional=99_000.0
        )
        assert check.available_settled_cash == 0.0
        assert not check.approved

    def test_zero_notional_request_is_approved(self) -> None:
        assert RiskManager.check_t_plus_two_settlement(0.0, 0.0).approved

    @pytest.mark.parametrize("cash", [3_000_000.0, 1_234_567.89, 250_000.0, 99.5])
    def test_a_budget_derived_from_the_cash_itself_is_approved(self, cash: float) -> None:
        """``cash * weight / gross`` can land a few ulp above ``cash``.

        The sandbox allocates a share of the very balance being checked.  With a
        single active instrument that share is the whole balance, and the round
        trip through the weight left the request 5e-10 TWD too large -- enough to
        reject a fully funded order before the tolerance was added.
        """
        weight = 0.7207122118880053
        budget = cash * weight / weight
        assert RiskManager.check_t_plus_two_settlement(cash, budget).approved

    def test_the_tolerance_does_not_approve_a_real_overdraft(self) -> None:
        cash = 3_000_000.0
        # One cent over is a real overdraft and must still be refused.
        assert not RiskManager.check_t_plus_two_settlement(cash, cash + 0.01).approved

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"settled_cash": -1.0, "requested_notional": 0.0},
            {"settled_cash": 1.0, "requested_notional": float("nan")},
            {"settled_cash": 1.0, "requested_notional": 1.0, "pending_buy_notional": -5.0},
        ],
    )
    def test_invalid_settlement_inputs_are_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            RiskManager.check_t_plus_two_settlement(**kwargs)


class TestShareQuantity:
    def test_exactly_one_board_lot(self) -> None:
        quantity = RiskManager.calculate_share_quantity(
            available_cash=50_000.0, reference_price=50.0, unit_shares=1000
        )
        assert quantity.board_lots == 1
        assert quantity.board_lot_shares == 1000
        assert quantity.odd_lot_shares == 0
        assert quantity.total_shares == 1000

    def test_board_lots_plus_odd_lot(self) -> None:
        quantity = RiskManager.calculate_share_quantity(
            available_cash=75_500.0, reference_price=50.0, unit_shares=1000
        )
        assert quantity.board_lots == 1
        assert quantity.odd_lot_shares == 510
        assert quantity.total_shares == 1510
        assert quantity.notional == pytest.approx(75_500.0)

    def test_odd_lots_disabled_truncates_to_whole_board_lots(self) -> None:
        quantity = RiskManager.calculate_share_quantity(
            available_cash=75_500.0, reference_price=50.0, unit_shares=1000, allow_odd_lots=False
        )
        assert quantity.total_shares == 1000
        assert quantity.odd_lot_shares == 0
        assert quantity.notional == pytest.approx(50_000.0)

    def test_cash_below_one_share_yields_nothing(self) -> None:
        quantity = RiskManager.calculate_share_quantity(10.0, 50.0, unit_shares=1000)
        assert quantity.total_shares == 0
        assert quantity.notional == 0.0

    def test_odd_lot_only_when_cash_is_under_one_board_lot(self) -> None:
        quantity = RiskManager.calculate_share_quantity(9_999.0, 50.0, unit_shares=1000)
        assert quantity.board_lots == 0
        assert quantity.odd_lot_shares == 199
        assert quantity.total_shares == 199

    @pytest.mark.parametrize("price", [33.33, 0.1, 7.7, 123.45, 1234.5])
    def test_lot_boundary_is_exact_despite_float_division(self, price: float) -> None:
        """Cash for exactly N lots must buy N lots, not N lots minus one share.

        ``floor(cash / price)`` is the classic place where binary floating point
        turns 1000.0 into 999.9999999999999.
        """
        unit = 1000
        for lots in (1, 2, 7):
            cash = price * unit * lots
            quantity = RiskManager.calculate_share_quantity(cash, price, unit_shares=unit)
            assert quantity.total_shares == unit * lots, (
                f"price={price} lots={lots}: got {quantity.total_shares}"
            )
            assert quantity.board_lots == lots
            assert quantity.odd_lot_shares == 0

    def test_notional_never_exceeds_available_cash(self) -> None:
        rng = np.random.default_rng(1234)
        for _ in range(300):
            cash = float(rng.uniform(0.0, 5_000_000.0))
            price = float(rng.uniform(0.05, 2_000.0))
            quantity = RiskManager.calculate_share_quantity(cash, price, unit_shares=1000)
            assert quantity.notional <= cash + 1e-6, (
                f"overspend: cash={cash} price={price} notional={quantity.notional}"
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"available_cash": -1.0, "reference_price": 10.0},
            {"available_cash": 10.0, "reference_price": 0.0},
            {"available_cash": 10.0, "reference_price": 10.0, "unit_shares": 0},
        ],
    )
    def test_invalid_share_inputs_are_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            RiskManager.calculate_share_quantity(**kwargs)


class TestStopLoss:
    def test_price_exactly_at_the_seven_percent_stop_triggers(self) -> None:
        assert RiskManager.breach_stop_loss(100.0, 93.0, 0.07)

    def test_price_one_tick_above_the_stop_does_not_trigger(self) -> None:
        assert not RiskManager.breach_stop_loss(100.0, 93.01, 0.07)

    def test_price_below_the_stop_triggers(self) -> None:
        assert RiskManager.breach_stop_loss(100.0, 92.99, 0.07)

    @pytest.mark.parametrize(
        "entry", ["10", "33.33", "87.5", "128", "1005", "4321", "2.5", "999.99"]
    )
    def test_quoted_stop_price_triggers_at_every_entry(self, entry: str) -> None:
        """The decimal price a venue would quote as the 7% stop must fire.

        The threshold is computed in decimal, as a human or an exchange states
        it, rather than by reconstructing ``entry * (1 - pct)`` in binary. Five
        of these eight cases failed before ``breach_stop_loss`` was changed to
        compare loss fractions: a NT$10 entry did not stop out at NT$9.30.
        """
        entry_price = float(entry)
        quoted_stop = float(Decimal(entry) * Decimal("0.93"))
        assert RiskManager.breach_stop_loss(entry_price, quoted_stop, 0.07)

    @pytest.mark.parametrize("entry", ["10", "33.33", "128", "4321"])
    def test_price_meaningfully_above_the_stop_does_not_trigger(self, entry: str) -> None:
        """The tolerance absorbs representation error only, not real distance."""
        entry_price = float(entry)
        quoted_stop = float(Decimal(entry) * Decimal("0.93"))
        # 1e-9 relative is still far finer than any tick, yet a thousand times
        # coarser than the tolerance, so it must remain outside the stop.
        assert not RiskManager.breach_stop_loss(
            entry_price, quoted_stop * (1.0 + 1e-9), 0.07
        )

    def test_a_profitable_position_is_never_stopped_out(self) -> None:
        assert not RiskManager.breach_stop_loss(100.0, 150.0, 0.07)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"entry_price": 0.0, "current_price": 10.0},
            {"entry_price": 10.0, "current_price": -1.0},
            {"entry_price": 10.0, "current_price": 9.0, "stop_loss_pct": 0.0},
            {"entry_price": 10.0, "current_price": 9.0, "stop_loss_pct": 1.0},
        ],
    )
    def test_invalid_stop_inputs_are_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            RiskManager.breach_stop_loss(**kwargs)


class TestRiskAssessment:
    """End-to-end behaviour of the layered risk decision used by the sandbox."""

    @staticmethod
    def _inputs() -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(1)
        returns = rng.normal(0.0, 0.008, 60)
        return returns, 100.0 * np.cumprod(1.0 + returns)

    def test_normal_conditions_produce_a_scaled_book(self) -> None:
        returns, equity = self._inputs()
        decision = RiskManager().assess(
            np.array([0.5, -0.5, 0.0]), returns, equity, 10.0, 1.0, PreTradeCostGuard()
        )
        assert not decision.stopped_out
        assert np.any(decision.risk_adjusted_weights != 0.0)
        assert decision.volatility_scale > 0.0

    def test_insufficient_history_refuses_to_trade(self) -> None:
        """Fail safe: an unknown volatility must not be treated as low risk."""
        returns, equity = self._inputs()
        decision = RiskManager().assess(
            np.array([0.5, -0.5]), returns[:5], equity, 10.0, 1.0, PreTradeCostGuard()
        )
        assert math.isnan(decision.realized_annual_volatility)
        assert decision.volatility_scale == 0.0
        assert np.all(decision.risk_adjusted_weights == 0.0)

    def test_breaching_the_drawdown_stop_liquidates_the_book(self) -> None:
        returns, _ = self._inputs()
        equity = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
        decision = RiskManager().assess(
            np.array([0.5, -0.5]), returns, equity, 10.0, 1.0, PreTradeCostGuard()
        )
        assert decision.stopped_out
        assert np.all(decision.risk_adjusted_weights == 0.0)

    def test_cost_guard_rejection_cannot_be_revived_by_leverage(self) -> None:
        """The economic hurdle is applied before volatility scaling, not after."""
        returns, equity = self._inputs()
        decision = RiskManager().assess(
            np.array([1.0, 1.0]), returns, equity, 0.1, 1.0, PreTradeCostGuard()
        )
        assert not np.any(decision.cost_approved)
        assert np.all(decision.guarded_weights == 0.0)
        assert np.all(decision.risk_adjusted_weights == 0.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"target_weights": np.array([])},
            {"target_weights": np.array([0.5, np.nan])},
            {"projected_turnover": -1.0},
            {"equity_curve": np.array([100.0, 0.0, 50.0])},
        ],
    )
    def test_invalid_assessment_inputs_are_rejected(self, kwargs: dict) -> None:
        returns, equity = self._inputs()
        arguments = dict(
            target_weights=np.array([0.5, -0.5]),
            portfolio_returns=returns,
            equity_curve=equity,
            expected_edge_bps=10.0,
            projected_turnover=1.0,
            cost_guard=PreTradeCostGuard(),
        )
        arguments.update(kwargs)
        with pytest.raises(ValueError):
            RiskManager().assess(**arguments)


class TestNoSpuriousWarnings:
    """Missing market data must not emit warnings the operator cannot act on."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda: cross_sectional_zscore(np.full((2, 3), np.nan)),
            lambda: causal_rolling_volatility(np.full((30, 2), np.nan), 21),
            lambda: nonlinear_momentum_filter(np.full((30, 2), np.nan), 21),
            lambda: combine_factors({"a": np.full((5, 3), np.nan)}, (FactorSpec("a", 1.0),)),
        ],
    )
    def test_all_nan_input_is_silent(self, call) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            call()

    def test_analyzer_is_silent_on_a_halted_instrument(self) -> None:
        prices = np.full((40, 2), 100.0)
        prices[:, 1] = np.nan  # a symbol with no data at all
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            ExtendedSignalAnalyzer(5, 10, support_resistance_window=5).analyze(prices)


class TestRiskConfig:
    def test_defaults_are_self_consistent(self) -> None:
        RiskConfig()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"target_annual_volatility": 0.0},
            {"volatility_lookback": 1},
            {"max_leverage": 0.0},
            {"minimum_stop_drawdown": 0.30, "maximum_drawdown": 0.15},
            {"drawdown_recovery_scale": 1.5},
        ],
    )
    def test_invalid_risk_configuration_is_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            RiskConfig(**kwargs)


# =====================================================================
# data_store: JSONL durability and replay
# =====================================================================


class TestDataStoreJsonl:
    def test_append_and_replay_round_trip(self, tmp_path: Path) -> None:
        store = DataStore(tmp_path / "decisions.jsonl")
        store.append_decision({"instrument": "2330", "signal": 1.0, "approved": True})
        store.append_decision({"instrument": "2317", "signal": -1.0, "approved": False})
        records = list(store.replay())
        assert [r["instrument"] for r in records] == ["2330", "2317"]
        assert records[0]["signal"] == 1.0
        assert records[1]["approved"] is False
        assert all("timestamp" in r for r in records)

    def test_replay_of_a_missing_log_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert list(DataStore(tmp_path / "absent.jsonl").replay()) == []

    def test_parent_directories_are_created(self, tmp_path: Path) -> None:
        store = DataStore(tmp_path / "deep" / "nested" / "decisions.jsonl")
        store.append_decision({"instrument": "2330"})
        assert store.log_path.exists()

    def test_every_line_is_independently_valid_json(self, tmp_path: Path) -> None:
        """One record per line is what makes an interrupted run recoverable."""
        store = DataStore(tmp_path / "decisions.jsonl")
        for index in range(5):
            store.append_decision({"instrument": f"S{index}", "score": index * 1.5})
        lines = store.log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5
        for line in lines:
            assert isinstance(json.loads(line), dict)

    def test_async_append_writes_all_records_in_order(self, tmp_path: Path) -> None:
        store = DataStore(tmp_path / "decisions.jsonl")

        async def scenario() -> None:
            for index in range(20):
                await store.append_decision_async({"instrument": f"S{index}", "seq": index})

        asyncio.run(scenario())
        records = list(store.replay())
        assert [r["seq"] for r in records] == list(range(20))

    def test_concurrent_async_appends_do_not_interleave_or_drop(self, tmp_path: Path) -> None:
        store = DataStore(tmp_path / "decisions.jsonl")

        async def scenario() -> None:
            await asyncio.gather(
                *(store.append_decision_async({"seq": index}) for index in range(60))
            )

        asyncio.run(scenario())
        records = list(store.replay())
        assert len(records) == 60
        assert sorted(r["seq"] for r in records) == list(range(60))

    def test_malformed_line_is_reported_with_its_location(self, tmp_path: Path) -> None:
        path = tmp_path / "decisions.jsonl"
        path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
        with pytest.raises(RuntimeError, match="malformed JSONL"):
            list(DataStore(path).replay())

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "decisions.jsonl"
        path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
        assert len(list(DataStore(path).replay())) == 2

    def test_replay_to_handler_counts_records(self, tmp_path: Path) -> None:
        store = DataStore(tmp_path / "decisions.jsonl")
        store.append_decision({"a": 1})
        store.append_decision({"a": 2})
        seen: list[dict] = []
        assert store.replay_to(seen.append) == 2
        assert len(seen) == 2

    def test_non_serializable_values_do_not_break_the_log(self, tmp_path: Path) -> None:
        store = DataStore(tmp_path / "decisions.jsonl")
        store.append_decision({"instrument": "2330", "signal": np.float64(1.5)})
        records = list(store.replay())
        assert len(records) == 1


class TestDecisionStoreCsv:
    def test_csv_round_trip_preserves_types(self, tmp_path: Path) -> None:
        store = DecisionStore(tmp_path / "decisions.csv", "csv")
        record = DecisionRecord(
            timestamp="2026-09-08T00:00:00+00:00",
            run_id="run-1",
            instrument="2330",
            signal=1.0,
            score=0.5,
            expected_edge_bps=3.0,
            required_edge_bps=2.9,
            projected_turnover=1.0,
            approved=True,
            submitted=False,
            metadata={"note": "unit"},
        )
        store.append([record])
        restored = list(store.iter_records())
        assert len(restored) == 1
        assert restored[0].approved is True
        assert restored[0].submitted is False
        assert restored[0].metadata == {"note": "unit"}

    def test_appending_an_empty_batch_creates_no_file(self, tmp_path: Path) -> None:
        store = DecisionStore(tmp_path / "decisions.csv", "csv")
        store.append([])
        assert not store.path.exists()

    def test_invalid_storage_format_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            DecisionStore(tmp_path / "x", "jsonl")  # type: ignore[arg-type]


# =====================================================================
# config_loader
# =====================================================================


class TestConfigLoader:
    def test_repository_config_loads_and_validates(self) -> None:
        config = load_config(REPO_ROOT / "config.json")
        assert config.schema_version >= 1
        assert config.system.paper_only is True
        assert config.system.submit_orders is False
        assert config.trading_rules.unit_shares == 1000
        assert config.trading_rules.stop_loss_pct == pytest.approx(0.07)
        assert config.indicators.sma_fast_window < config.indicators.sma_slow_window

    def test_dotted_path_access(self) -> None:
        loader = ConfigLoader(REPO_ROOT / "config.json")
        assert loader.get("system.port") == 7497
        assert loader.get("indicators.sma_fast_window") == 20

    def test_missing_key_uses_the_supplied_default(self) -> None:
        loader = ConfigLoader(REPO_ROOT / "config.json")
        assert loader.get("system.does_not_exist", "fallback") == "fallback"
        assert loader.get("no.such.section", 42) == 42
        assert loader.get("system.nested.deeper", None) is None

    def test_missing_key_without_a_default_is_an_error(self) -> None:
        loader = ConfigLoader(REPO_ROOT / "config.json")
        with pytest.raises(ConfigurationError, match="missing required configuration key"):
            loader.get("system.does_not_exist")

    def test_absent_file_is_reported_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            ConfigLoader(tmp_path / "nope.json")

    def test_malformed_json_is_reported_clearly(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="invalid JSON"):
            ConfigLoader(path)

    def test_live_submission_is_refused(self, tmp_path: Path) -> None:
        raw = json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))
        raw["system"]["paper_only"] = False
        raw["system"]["submit_orders"] = True
        raw["system"]["port"] = 7496
        path = tmp_path / "live.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ConfigurationError):
            ConfigLoader(path).load()

    def test_paper_only_requires_a_paper_port(self, tmp_path: Path) -> None:
        raw = json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))
        raw["system"]["port"] = 7496
        path = tmp_path / "wrongport.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Paper Trading port"):
            ConfigLoader(path).load()

    def test_unknown_setting_in_a_section_is_rejected(self, tmp_path: Path) -> None:
        raw = json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))
        raw["system"]["surprise"] = 1
        path = tmp_path / "extra.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ConfigurationError, match="unknown settings"):
            ConfigLoader(path).load()

    def test_risk_section_feeds_risk_config(self) -> None:
        """The config's risk block must be directly usable by RiskManager."""
        config = load_config(REPO_ROOT / "config.json")
        risk_config = RiskConfig(**dict(config.risk))
        assert risk_config.target_annual_volatility > 0.0


# =====================================================================
# run_sandbox: end-to-end integration
# =====================================================================


class TestSandboxIntegration:
    """The sandbox must complete, log durably, and never mark an order sent."""

    @staticmethod
    def _run(tmp_path: Path, *extra: str) -> tuple[int, list[dict]]:
        log_path = tmp_path / "decisions.jsonl"
        code = run_sandbox.main(
            ["--cycles", "10", "--log-path", str(log_path), "--config", str(REPO_ROOT / "config.json"), *extra]
        )
        records = list(DataStore(log_path).replay()) if log_path.exists() else []
        return code, records

    def test_a_full_run_succeeds_and_persists_every_decision(self, tmp_path: Path) -> None:
        code, records = self._run(tmp_path)
        assert code == 0
        assert len(records) == 10 * 4, "one record per instrument per cycle"
        assert {record["cycle"] for record in records} == set(range(1, 11))

    def test_no_decision_is_ever_marked_submitted(self, tmp_path: Path) -> None:
        _, records = self._run(tmp_path)
        assert records and all(record["submitted"] is False for record in records)

    def test_an_unapproved_decision_is_never_sized(self, tmp_path: Path) -> None:
        """A share count beside ``approved: false`` would invite a bad fill."""
        _, records = self._run(tmp_path)
        assert not [r for r in records if r["total_shares"] > 0 and not r["approved"]]

    def test_notional_never_exceeds_available_settled_cash(self, tmp_path: Path) -> None:
        _, records = self._run(tmp_path)
        for record in records:
            assert record["notional"] <= record["available_settled_cash"] + 1e-6

    def test_board_and_odd_lots_reconcile(self, tmp_path: Path) -> None:
        _, records = self._run(tmp_path)
        for record in records:
            assert (
                record["board_lots"] * 1000 + record["odd_lot_shares"] == record["total_shares"]
            )
            assert 0 <= record["odd_lot_shares"] < 1000

    def test_the_run_is_deterministic_for_a_fixed_seed(self, tmp_path: Path) -> None:
        _, first = self._run(tmp_path / "a", "--seed", "4242")
        _, second = self._run(tmp_path / "b", "--seed", "4242")
        strip = lambda rows: [  # noqa: E731 - timestamps differ between runs
            {k: v for k, v in row.items() if k != "timestamp"} for row in rows
        ]
        assert strip(first) == strip(second)

    def test_a_configuration_enabling_live_orders_is_refused(self, tmp_path: Path) -> None:
        raw = json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))
        raw["system"]["submit_orders"] = True
        path = tmp_path / "live.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        assert run_sandbox.main(["--config", str(path), "--cycles", "1"]) == 2

    @pytest.mark.parametrize("extra", [["--symbols", "2330"], ["--cycles", "0"]])
    def test_invalid_arguments_exit_with_code_two(self, tmp_path: Path, extra: list[str]) -> None:
        code = run_sandbox.main(
            ["--config", str(REPO_ROOT / "config.json"),
             "--log-path", str(tmp_path / "d.jsonl"), *extra]
        )
        assert code == 2
