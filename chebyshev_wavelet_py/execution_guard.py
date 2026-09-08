"""Pre-trade cost guardrails informed by empirical break-even thresholds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostGuardConfig:
    """Parameters for the pre-trade economic-efficiency filter.

    The defaults use the daily strategy's 2.9 bps break-even spread and 1.989
    mean daily turnover.  Scaling the threshold by realized projected turnover
    avoids approving a high-turnover trade merely because it shares the same
    raw signal score as a low-turnover trade.
    """

    break_even_spread_bps: float = 2.9
    reference_turnover: float = 1.989
    required_edge_multiple: float = 1.0

    def __post_init__(self) -> None:
        if self.break_even_spread_bps <= 0.0:
            raise ValueError("break_even_spread_bps must be positive")
        if self.reference_turnover <= 0.0:
            raise ValueError("reference_turnover must be positive")
        if self.required_edge_multiple <= 0.0:
            raise ValueError("required_edge_multiple must be positive")


class PreTradeCostGuard:
    """Approve only trades whose expected edge clears a turnover-scaled hurdle."""

    def __init__(self, config: CostGuardConfig | None = None) -> None:
        self.config = config or CostGuardConfig()

    def evaluate(
        self,
        signal: np.ndarray | float,
        expected_edge_bps: np.ndarray | float,
        projected_turnover: np.ndarray | float,
    ) -> dict[str, np.ndarray]:
        """Filter signals whose expected edge cannot clear the cost hurdle.

        ``expected_edge_bps`` is the model's conditional expected gross edge,
        expressed in basis points over the intended holding period.  The guard
        is intentionally deterministic and exposes every intermediate value so
        that rejected trades can be audited rather than silently discarded.
        """
        signal, expected_edge, turnover = np.broadcast_arrays(
            np.asarray(signal, dtype=float),
            np.asarray(expected_edge_bps, dtype=float),
            np.asarray(projected_turnover, dtype=float),
        )
        if np.any(turnover < 0.0):
            raise ValueError("projected_turnover cannot be negative")

        break_even = self.config.break_even_spread_bps * turnover / self.config.reference_turnover
        required_edge = self.config.required_edge_multiple * break_even
        finite_input = np.isfinite(signal) & np.isfinite(expected_edge) & np.isfinite(turnover)
        approved = finite_input & (signal != 0.0) & (np.abs(expected_edge) >= required_edge)
        guarded_signal = np.where(approved, signal, 0.0)
        efficiency = np.divide(
            np.abs(expected_edge),
            required_edge,
            out=np.full_like(expected_edge, np.inf, dtype=float),
            where=required_edge > 0.0,
        )
        return {
            "signal": guarded_signal,
            "approved": approved,
            "break_even_bps": break_even,
            "required_edge_bps": required_edge,
            "expected_edge_bps": expected_edge,
            "efficiency_ratio": efficiency,
        }
