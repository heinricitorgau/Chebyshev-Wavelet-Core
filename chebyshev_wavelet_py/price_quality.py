"""Guards against price series that carry a hidden cross-sectional bias.

A dividend paid on an unadjusted series shows up as a price drop the issuer
never inflicted on the holder.  Because payout rates differ persistently
between instruments, that drop does not average out across a panel: it becomes
a stable ranking signal pointing the wrong way, and every cross-sectional model
downstream will happily trade it.

The MATLAB side of this project measured the magnitude on its own 31-ETF
universe: implied annualized yields spanning 1.08% to 4.25%, a 3.2 percentage
point spread, which is larger than any edge the project has ever found.  Its
loader therefore defaults to dividend-adjusted prices.  This module gives the
Python side the same protection, since Interactive Brokers returns *unadjusted*
prices under its default ``TRADES`` setting.
"""

from __future__ import annotations

import numpy as np


Array = np.ndarray

# IB ``whatToShow`` values that already fold distributions into the price.
# ADJUSTED_LAST is IB's dividend-adjusted series; TRADES adjusts for splits
# only, which is the trap: it looks adjusted because the split gaps are gone.
_TOTAL_RETURN_SETTINGS = frozenset({"ADJUSTED_LAST"})
_KNOWN_SETTINGS = frozenset(
    {"TRADES", "ADJUSTED_LAST", "MIDPOINT", "BID", "ASK", "BID_ASK", "HISTORICAL_VOLATILITY"}
)


class PriceQualityError(ValueError):
    """Raised when a price series is unsuitable for cross-sectional work."""


def is_total_return(what_to_show: str) -> bool:
    """Return whether an IB ``whatToShow`` setting yields dividend-adjusted prices."""
    return what_to_show.strip().upper() in _TOTAL_RETURN_SETTINGS


def require_total_return(what_to_show: str, allow_unadjusted: bool = False) -> str:
    """Validate a price setting intended for cross-sectional ranking.

    Passing ``allow_unadjusted=True`` is a deliberate, documented choice -- for
    a single-instrument study, or when comparing against an unadjusted
    benchmark -- and never a default.
    """
    setting = what_to_show.strip().upper()
    if setting not in _KNOWN_SETTINGS:
        raise PriceQualityError(
            f"unrecognized whatToShow setting {what_to_show!r}; "
            f"expected one of {sorted(_KNOWN_SETTINGS)}"
        )
    if not is_total_return(setting) and not allow_unadjusted:
        raise PriceQualityError(
            f"{setting} prices are not dividend-adjusted and must not be ranked "
            "cross-sectionally: persistent payout differences between instruments "
            "become a stable false signal. This project measured a 3.2 percentage "
            "point spread in implied yield across a 31-ETF universe, larger than "
            "any edge it has found. Use ADJUSTED_LAST, or pass "
            "allow_unadjusted=True to record an explicit decision."
        )
    return setting


def implied_dividend_yield(
    adjusted_prices: Array,
    unadjusted_prices: Array,
    periods_per_year: int = 252,
) -> Array:
    """Estimate each instrument's annualized payout from the two price series.

    The adjusted series compounds distributions back in, so the gap between the
    two total drifts over the same window is the payout.  Returns one value per
    instrument; the spread across that vector is the size of the false ranking
    signal an unadjusted panel would carry.
    """
    adjusted = np.asarray(adjusted_prices, dtype=float)
    unadjusted = np.asarray(unadjusted_prices, dtype=float)
    if adjusted.ndim == 1:
        adjusted = adjusted[:, None]
    if unadjusted.ndim == 1:
        unadjusted = unadjusted[:, None]
    if adjusted.shape != unadjusted.shape:
        raise PriceQualityError("the two price series must share the same shape")
    if adjusted.shape[0] < 2:
        raise PriceQualityError("at least two observations are required")
    if np.any(adjusted <= 0.0) or np.any(unadjusted <= 0.0):
        raise PriceQualityError("prices must be strictly positive")

    periods = adjusted.shape[0] - 1
    years = periods / float(periods_per_year)
    if years <= 0.0:
        raise PriceQualityError("periods_per_year is too large for this sample")
    adjusted_growth = np.log(adjusted[-1] / adjusted[0])
    unadjusted_growth = np.log(unadjusted[-1] / unadjusted[0])
    return (adjusted_growth - unadjusted_growth) / years


def dividend_drag_spread(
    adjusted_prices: Array, unadjusted_prices: Array, periods_per_year: int = 252
) -> float:
    """Return the spread in implied annualized yield across the panel.

    This is the quantity that matters for a ranked portfolio: a uniform yield
    shifts every instrument equally and cancels in the cross-section, whereas
    the spread does not.
    """
    yields = implied_dividend_yield(adjusted_prices, unadjusted_prices, periods_per_year)
    finite = yields[np.isfinite(yields)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite) - np.min(finite))
