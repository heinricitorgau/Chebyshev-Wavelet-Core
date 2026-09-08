"""Second-kind Chebyshev polynomials and normalized dyadic wavelets.

All cell indices are zero based.  With ``J = k - 1``, cell ``n`` is
``[n / 2**J, (n + 1) / 2**J)`` and is mapped to ``[-1, 1)`` by
``u = 2**(J + 1) * t - 2*n - 1``.  This is the convention formalized in
MyMathLib and used by the MATLAB numerical core.
"""

from __future__ import annotations

import numpy as np


def chebyshev_u(degree: int, x: np.ndarray | float) -> np.ndarray:
    """Evaluate the second-kind Chebyshev polynomial ``U_degree(x)``.

    The recurrence ``U_0 = 1``, ``U_1 = 2x``, and
    ``U_{n+1} = 2x U_n - U_{n-1}`` is evaluated on the complete NumPy array
    at once.  No polynomial toolbox or scalar callback is required.
    """
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("degree must be a non-negative integer")

    values = np.asarray(x, dtype=float)
    if degree == 0:
        return np.ones_like(values)
    previous = np.ones_like(values)
    current = 2.0 * values
    for _ in range(1, degree):
        previous, current = current, 2.0 * values * current - previous
    return current


def cell_bounds(level_j: int, cell_index: int) -> tuple[float, float]:
    """Return the left and right endpoints of a zero-based dyadic cell."""
    _validate_cell(level_j, cell_index)
    width = 2.0 ** (-level_j)
    left = cell_index * width
    return left, left + width


def cell_coordinate(level_j: int, cell_index: int, t: np.ndarray | float) -> np.ndarray:
    """Map points in cell ``cell_index`` from ``[0, 1]`` to ``[-1, 1]``."""
    _validate_cell(level_j, cell_index)
    return (2.0 ** (level_j + 1)) * np.asarray(t, dtype=float) - 2.0 * cell_index - 1.0


def wavelet_scale(level_j: int) -> float:
    """Return ``sqrt(2**(J + 2) / pi)``, the MATLAB-aligned normalization."""
    if not isinstance(level_j, (int, np.integer)) or level_j < 0:
        raise ValueError("level_j must be a non-negative integer")
    return float(np.sqrt((2.0 ** (level_j + 2)) / np.pi))


def wavelet(
    level_j: int,
    cell_index: int,
    degree: int,
    t: np.ndarray | float,
) -> np.ndarray:
    """Evaluate a normalized Chebyshev-U wavelet, zero outside its cell.

    The right endpoint is included only for the final dyadic cell.  This has
    no effect on integrals and avoids an artificial zero at ``t == 1`` in
    finite sampled series.
    """
    _validate_cell(level_j, cell_index)
    points = np.asarray(t, dtype=float)
    left, right = cell_bounds(level_j, cell_index)
    is_last_cell = cell_index == (2**level_j - 1)
    in_cell = (points >= left) & ((points < right) | (is_last_cell & (points <= right)))
    coordinate = cell_coordinate(level_j, cell_index, points)
    values = wavelet_scale(level_j) * chebyshev_u(degree, coordinate)
    return np.where(in_cell, values, 0.0)


def cell_weight(level_j: int, cell_index: int, t: np.ndarray | float) -> np.ndarray:
    """Return the non-negative second-kind weight on one dyadic cell.

    The weight is ``sqrt(1-u(t)^2)`` on the cell and zero outside.  Clipping
    handles round-off at the endpoints without changing the mathematical
    definition.
    """
    points = np.asarray(t, dtype=float)
    left, right = cell_bounds(level_j, cell_index)
    is_last_cell = cell_index == (2**level_j - 1)
    in_cell = (points >= left) & ((points < right) | (is_last_cell & (points <= right)))
    coordinate = cell_coordinate(level_j, cell_index, points)
    return np.where(in_cell, np.sqrt(np.maximum(0.0, 1.0 - coordinate**2)), 0.0)


def weighted_inner_product(
    left_values: np.ndarray,
    right_values: np.ndarray,
    weights: np.ndarray,
    quadrature_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a weighted inner product along axis zero.

    Inputs may have trailing broadcastable dimensions.  ``quadrature_weights``
    is optional so callers can use exact second-kind Gauss quadrature weights
    or a discretization-specific integration rule without changing the core.
    """
    left_values, right_values, weights = np.broadcast_arrays(
        np.asarray(left_values, dtype=float),
        np.asarray(right_values, dtype=float),
        np.asarray(weights, dtype=float),
    )
    integrand = left_values * right_values * weights
    if quadrature_weights is None:
        return np.sum(integrand, axis=0)
    quadrature_weights = np.asarray(quadrature_weights, dtype=float)
    return np.sum(integrand * quadrature_weights, axis=0)


def second_kind_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes and weights for ``sqrt(1-x**2)`` Gauss-Chebyshev quadrature."""
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")
    index = np.arange(1, order + 1, dtype=float)
    angles = index * np.pi / (order + 1.0)
    return np.cos(angles), (np.pi / (order + 1.0)) * np.sin(angles) ** 2


def _validate_cell(level_j: int, cell_index: int) -> None:
    if not isinstance(level_j, (int, np.integer)) or level_j < 0:
        raise ValueError("level_j must be a non-negative integer")
    if not isinstance(cell_index, (int, np.integer)) or not 0 <= cell_index < 2**level_j:
        raise ValueError("cell_index must be in [0, 2**level_j)")
