"""Composable, causal signal templates for multi-factor research.

The functions in this module operate on arrays shaped ``(time, assets)`` and
produce values aligned to the final observation used in their calculation.
Callers must therefore execute a resulting signal no earlier than the next
bar.  No function forward-fills, centers a window, or reads future rows.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping

import numpy as np


Array = np.ndarray


@contextmanager
def _expecting_empty_slices() -> Iterator[None]:
    """Silence NumPy's all-NaN-slice warnings where they are already handled.

    ``nanmean``/``nanstd`` warn when a slice contains no finite value.  Every
    call site below masks that case explicitly straight afterwards, so the
    warning carries no information a caller could act on.  Left unsuppressed it
    fires on every bar with a halted or newly listed instrument, which in a live
    run either floods the log or teaches the operator to ignore warnings.  The
    suppression is scoped to the individual reduction rather than the module.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0", category=RuntimeWarning)
        warnings.filterwarnings(
            "ignore", message="invalid value encountered", category=RuntimeWarning
        )
        yield


@dataclass(frozen=True)
class FactorSpec:
    """A named factor and its signed portfolio-combination weight."""

    name: str
    weight: float


@dataclass(frozen=True)
class CandleAnatomy:
    """Per-bar candle geometry, each part expressed as a fraction of the range.

    Requires open, high and low.  ``body`` keeps its sign, so a red bar is
    negative; the two shadows are non-negative by construction.  Bars whose high
    equals their low carry ``NaN`` rather than a fabricated zero, because a
    zero-range bar has no geometry to report.
    """

    body: Array
    upper_shadow: Array
    lower_shadow: Array
    range_fraction: Array


@dataclass(frozen=True)
class TechnicalSignalSnapshot:
    """Causal SMA and support/resistance state for each time and asset.

    ``golden_cross`` and ``death_cross`` record the raw crossing events: a
    crossing either happened on that bar or it did not.  ``signal`` is the
    actionable output and is where the trend filter applies, so the two can be
    inspected separately when a cross is present but was not acted on.
    """

    fast_sma: Array
    slow_sma: Array
    support: Array
    resistance: Array
    golden_cross: Array
    death_cross: Array
    breakout: Array
    breakdown: Array
    signal: Array
    slow_trend: Array
    suppressed_golden: Array
    suppressed_death: Array
    candles: CandleAnatomy | None = None


class ExtendedSignalAnalyzer:
    """Causal technical filters that can be layered over wavelet signals.

    A bullish signal requires either a golden cross or an upward resistance
    breakout; a bearish signal requires either a death cross or a support
    breakdown.  The caller can use the resulting signal as a standalone
    template or as a gate on a Chebyshev-wavelet score.
    """

    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 60,
        support_resistance_window: int = 20,
        trend_filter: bool = True,
    ) -> None:
        if not 1 <= fast_window < slow_window:
            raise ValueError("SMA windows must satisfy 1 <= fast_window < slow_window")
        if support_resistance_window < 2:
            raise ValueError("support_resistance_window must be at least two")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.support_resistance_window = support_resistance_window
        self.trend_filter = trend_filter

    def analyze(
        self,
        close_prices: Array,
        high_prices: Array | None = None,
        low_prices: Array | None = None,
        open_prices: Array | None = None,
    ) -> TechnicalSignalSnapshot:
        """Return indicators derived only from the current and preceding bars.

        Only ``close_prices`` is required.  Supplying highs and lows moves
        support and resistance onto the extremes the market actually traded
        through, which is what a level drawn on a chart represents; closes alone
        systematically understate both.  Adding opens additionally yields the
        candle geometry in :class:`CandleAnatomy`.
        """
        prices = _as_matrix(close_prices)
        if np.any(np.isfinite(prices) & (prices <= 0.0)):
            raise ValueError("finite close prices must be strictly positive")
        highs = _aligned_or_none(high_prices, prices, "high_prices")
        lows = _aligned_or_none(low_prices, prices, "low_prices")
        opens = _aligned_or_none(open_prices, prices, "open_prices")
        if (highs is None) != (lows is None):
            raise ValueError("high_prices and low_prices must be supplied together")
        if highs is not None and lows is not None:
            if np.any(np.isfinite(highs) & np.isfinite(lows) & (highs < lows)):
                raise ValueError("high_prices cannot be below low_prices")

        fast = _causal_sma(prices, self.fast_window)
        slow = _causal_sma(prices, self.slow_window)
        support, resistance = _prior_support_resistance(
            prices, self.support_resistance_window, highs=highs, lows=lows
        )

        prior_fast = np.vstack([np.full((1, prices.shape[1]), np.nan), fast[:-1]])
        prior_slow = np.vstack([np.full((1, prices.shape[1]), np.nan), slow[:-1]])
        golden = (fast > slow) & (prior_fast <= prior_slow)
        death = (fast < slow) & (prior_fast >= prior_slow)
        breakout = prices > resistance
        breakdown = prices < support

        # Direction of the long-term average, from the current and previous bar
        # only.  A cross against the prevailing trend is the classic whipsaw:
        # buying a golden cross while the slow average still falls, or selling a
        # death cross while it still rises.
        #
        # The filter is scoped to crosses.  A support or resistance break is a
        # separate piece of evidence and is left alone, so a bar can still turn
        # bullish through a breakout on the same day its cross was suppressed.
        # Widening the filter to cover breaks would be a judgement about levels
        # that the moving-average rule does not make.
        slow_step = slow - prior_slow
        slow_trend = np.where(
            np.isfinite(slow_step), np.sign(slow_step), np.nan
        )
        trend_falling = slow_step < 0.0
        trend_rising = slow_step > 0.0
        suppressed_golden = golden & trend_falling if self.trend_filter else np.zeros_like(golden)
        suppressed_death = death & trend_rising if self.trend_filter else np.zeros_like(death)

        actionable_golden = golden & ~suppressed_golden
        actionable_death = death & ~suppressed_death
        bullish = actionable_golden | breakout
        bearish = actionable_death | breakdown
        signal = np.where(bullish & ~bearish, 1.0, np.where(bearish & ~bullish, -1.0, 0.0))
        signal[~np.isfinite(prices)] = 0.0
        candles = _candle_anatomy(opens, highs, lows, prices)
        return TechnicalSignalSnapshot(
            fast_sma=fast,
            slow_sma=slow,
            support=support,
            resistance=resistance,
            golden_cross=golden,
            death_cross=death,
            breakout=breakout,
            breakdown=breakdown,
            signal=signal,
            slow_trend=slow_trend,
            suppressed_golden=suppressed_golden,
            suppressed_death=suppressed_death,
            candles=candles,
        )


def cross_sectional_zscore(values: Array) -> Array:
    """Standardize each row across assets without imputing unavailable values."""
    array = _as_matrix(values)
    with _expecting_empty_slices():
        row_mean = np.nanmean(array, axis=1, keepdims=True)
        row_std = np.nanstd(array, axis=1, keepdims=True)
        normalized = (array - row_mean) / np.where(row_std > 0.0, row_std, np.nan)
    return np.where(np.isfinite(normalized), normalized, 0.0)


def combine_factors(
    factor_values: Mapping[str, Array],
    factor_specs: tuple[FactorSpec, ...],
    standardize: bool = True,
) -> Array:
    """Create a weighted factor composite with explicit, auditable weights.

    All factors must have identical ``(time, assets)`` shape.  By default each
    factor is cross-sectionally standardized before combination, which prevents
    a factor's numerical scale from silently becoming its economic weight.
    """
    if not factor_specs:
        raise ValueError("at least one FactorSpec is required")
    composite: Array | None = None
    for spec in factor_specs:
        if spec.name not in factor_values:
            raise KeyError(f"missing factor '{spec.name}'")
        factor = _as_matrix(factor_values[spec.name])
        factor = cross_sectional_zscore(factor) if standardize else factor
        if composite is None:
            composite = np.zeros_like(factor, dtype=float)
        if factor.shape != composite.shape:
            raise ValueError("all factors must share the same (time, assets) shape")
        composite += spec.weight * factor
    assert composite is not None
    return cross_sectional_zscore(composite) if standardize else composite


def causal_rolling_volatility(
    returns: Array,
    lookback: int = 21,
    periods_per_year: int = 252,
    minimum_observations: int | None = None,
) -> Array:
    """Estimate annualized trailing volatility using only completed returns.

    A value at ``t`` uses the interval ``[t-lookback+1, t]``.  Insufficient or
    non-finite windows remain ``NaN`` instead of being repaired with future
    observations.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least two")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    minimum = lookback if minimum_observations is None else minimum_observations
    if not 2 <= minimum <= lookback:
        raise ValueError("minimum_observations must be between 2 and lookback")

    matrix = _as_matrix(returns)
    output = np.full_like(matrix, np.nan, dtype=float)
    if matrix.shape[0] < lookback:
        return output
    windows = _rolling_windows(matrix, lookback)
    valid_count = np.sum(np.isfinite(windows), axis=1)
    with _expecting_empty_slices():
        volatility = np.nanstd(windows, axis=1, ddof=1) * np.sqrt(periods_per_year)
    volatility[valid_count < minimum] = np.nan
    output[lookback - 1 :] = volatility
    return output


def volatility_adjusted_signal(
    signal: Array,
    returns: Array,
    lookback: int = 21,
    target_annual_volatility: float = 0.10,
    max_leverage: float = 2.0,
    periods_per_year: int = 252,
) -> Array:
    """Scale a causal signal inversely with trailing realized volatility."""
    if target_annual_volatility <= 0.0 or max_leverage <= 0.0:
        raise ValueError("target_annual_volatility and max_leverage must be positive")
    signal_matrix = _as_matrix(signal)
    volatility = causal_rolling_volatility(returns, lookback, periods_per_year)
    if signal_matrix.shape != volatility.shape:
        raise ValueError("signal and returns must share the same shape")
    scale = target_annual_volatility / volatility
    scale = np.clip(scale, 0.0, max_leverage)
    return np.where(np.isfinite(scale), signal_matrix * scale, 0.0)


def nonlinear_momentum_filter(
    returns: Array,
    lookback: int = 21,
    saturation: float = 2.0,
    threshold: float = 0.0,
) -> Array:
    """Create a bounded nonlinear momentum score from trailing log returns.

    The score uses ``tanh(z / saturation)``.  This preserves rank information
    around zero while preventing isolated return shocks from dominating the
    cross-sectional portfolio.  A nonzero threshold can suppress weak scores.
    """
    if lookback < 1 or saturation <= 0.0 or threshold < 0.0:
        raise ValueError("lookback and saturation must be positive; threshold cannot be negative")
    matrix = _as_matrix(returns)
    output = np.full_like(matrix, np.nan, dtype=float)
    if matrix.shape[0] < lookback:
        return output
    windows = _rolling_windows(matrix, lookback)
    valid = np.all(np.isfinite(windows), axis=1)
    momentum = np.sum(windows, axis=1)
    standardized = cross_sectional_zscore(momentum)
    score = np.tanh(standardized / saturation)
    score = np.where(np.abs(score) >= threshold, score, 0.0)
    output[lookback - 1 :] = np.where(valid, score, np.nan)
    return output


def _as_matrix(values: Array) -> Array:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("values must be a one- or two-dimensional array")
    return matrix


def _rolling_windows(values: Array, lookback: int) -> Array:
    """Return rolling windows with shape ``(windows, lookback, assets)``."""
    windows = np.lib.stride_tricks.sliding_window_view(values, lookback, axis=0)
    return np.moveaxis(windows, -1, 1)


def _causal_sma(prices: Array, window: int) -> Array:
    output = np.full_like(prices, np.nan, dtype=float)
    if prices.shape[0] < window:
        return output
    windows = _rolling_windows(prices, window)
    valid = np.all(np.isfinite(windows), axis=1)
    means = np.mean(windows, axis=1)
    output[window - 1 :] = np.where(valid, means, np.nan)
    return output


def _aligned_or_none(values: Array | None, reference: Array, name: str) -> Array | None:
    """Coerce an optional price series and require it to match the closes."""
    if values is None:
        return None
    matrix = _as_matrix(values)
    if matrix.shape != reference.shape:
        raise ValueError(f"{name} must have the same (time, assets) shape as close_prices")
    if np.any(np.isfinite(matrix) & (matrix <= 0.0)):
        raise ValueError(f"finite {name} must be strictly positive")
    return matrix


def _candle_anatomy(
    opens: Array | None, highs: Array | None, lows: Array | None, closes: Array
) -> CandleAnatomy | None:
    """Decompose each bar into body and shadows as fractions of its range."""
    if opens is None or highs is None or lows is None:
        return None
    span = highs - lows
    # A zero-range bar has no geometry; report it as unavailable rather than
    # dividing by zero and calling the result a doji.
    usable = np.isfinite(span) & (span > 0.0)
    safe_span = np.where(usable, span, np.nan)
    upper = highs - np.maximum(opens, closes)
    lower = np.minimum(opens, closes) - lows
    return CandleAnatomy(
        body=(closes - opens) / safe_span,
        upper_shadow=upper / safe_span,
        lower_shadow=lower / safe_span,
        range_fraction=np.where(usable, span / closes, np.nan),
    )


def _prior_support_resistance(
    prices: Array,
    window: int,
    highs: Array | None = None,
    lows: Array | None = None,
) -> tuple[Array, Array]:
    """Compute levels from prior bars only, excluding today's breakout bar.

    When highs and lows are available the levels come from the extremes the
    market actually traded through, which is what a line drawn on a chart marks.
    Closes alone place resistance below every intraday high it was built from,
    so a breakout registers earlier than it should.
    """
    support = np.full_like(prices, np.nan, dtype=float)
    resistance = np.full_like(prices, np.nan, dtype=float)
    if prices.shape[0] <= window:
        return support, resistance
    low_source = prices if lows is None else lows
    high_source = prices if highs is None else highs
    low_windows = _rolling_windows(low_source[:-1], window)
    high_windows = _rolling_windows(high_source[:-1], window)
    low_valid = np.all(np.isfinite(low_windows), axis=1)
    high_valid = np.all(np.isfinite(high_windows), axis=1)
    support[window:] = np.where(low_valid, np.min(low_windows, axis=1), np.nan)
    resistance[window:] = np.where(high_valid, np.max(high_windows, axis=1), np.nan)
    return support, resistance
