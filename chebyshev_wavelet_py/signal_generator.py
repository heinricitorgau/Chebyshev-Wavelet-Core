"""Causal rolling Chebyshev-wavelet feature projection for time-series data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chebyshev_core import chebyshev_u


@dataclass(frozen=True)
class CausalWaveletSignalGenerator:
    """Project trailing return windows onto a second-kind Chebyshev-U basis.

    ``transform`` returns signals aligned to the final observation in each
    window.  A signal at index ``t`` depends only on observations through
    ``t`` and is therefore available for execution no earlier than ``t + 1``.
    The caller remains responsible for enforcing that execution convention.
    """

    window_size: int
    mode_count: int
    signal_mode: int = 1
    threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least two")
        if self.mode_count < 1:
            raise ValueError("mode_count must be positive")
        if not 0 <= self.signal_mode < self.mode_count:
            raise ValueError("signal_mode must index a retained mode")
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")

    def transform(self, returns: np.ndarray) -> dict[str, np.ndarray]:
        """Return causal coefficients, cross-sectional scores, and signals.

        ``returns`` must be shaped ``(time, assets)``; a one-dimensional input
        is treated as one asset.  Each trailing window is standardized within
        asset before projection so the output captures shape rather than level.
        Windows containing non-finite values remain unavailable rather than
        being silently imputed from later observations.
        """
        data = np.asarray(returns, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2:
            raise ValueError("returns must be a one- or two-dimensional array")

        time_count, asset_count = data.shape
        coefficients = np.full((time_count, self.mode_count, asset_count), np.nan)
        scores = np.full((time_count, asset_count), np.nan)
        signals = np.zeros((time_count, asset_count), dtype=float)
        if time_count < self.window_size:
            return {"coefficients": coefficients, "scores": scores, "signal": signals}

        windows = np.lib.stride_tricks.sliding_window_view(data, self.window_size, axis=0)
        # NumPy places the rolled axis last for axis=0: (windows, assets, time).
        windows = np.moveaxis(windows, -1, 1)
        finite_window = np.all(np.isfinite(windows), axis=1)
        means = np.mean(windows, axis=1, keepdims=True)
        standard_deviations = np.std(windows, axis=1, keepdims=True)
        standardized = (windows - means) / np.where(standard_deviations > 0.0, standard_deviations, np.nan)

        projection = self._projection_matrix()
        window_coefficients = np.einsum("mw,twa->tma", projection, standardized)
        window_coefficients = np.where(finite_window[:, None, :], window_coefficients, np.nan)
        first_valid = self.window_size - 1
        coefficients[first_valid:] = window_coefficients

        raw_scores = window_coefficients[:, self.signal_mode, :]
        score_mean = np.nanmean(raw_scores, axis=1, keepdims=True)
        score_std = np.nanstd(raw_scores, axis=1, keepdims=True)
        window_scores = (raw_scores - score_mean) / np.where(score_std > 0.0, score_std, np.nan)
        scores[first_valid:] = window_scores
        active = np.where(np.abs(window_scores) >= self.threshold, np.sign(window_scores), 0.0)
        signals[first_valid:] = np.where(np.isfinite(active), active, 0.0)
        return {"coefficients": coefficients, "scores": scores, "signal": signals}

    def _projection_matrix(self) -> np.ndarray:
        """Return a reusable least-squares projector from samples to U modes."""
        nodes = np.linspace(-1.0, 1.0, self.window_size)
        basis = np.column_stack([chebyshev_u(mode, nodes) for mode in range(self.mode_count)])
        return np.linalg.pinv(basis)
