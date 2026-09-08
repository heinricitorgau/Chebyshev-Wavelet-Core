"""Closed-form OMI and POM construction for the Chebyshev-wavelet basis."""

from __future__ import annotations

import numpy as np

from .chebyshev_core import wavelet_scale


def integration_blocks(level_j: int, mode_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct the ``Nblk`` and truncated ``Mblk`` OMI blocks.

    Rows index the wavelet being integrated and columns index the wavelet in
    the expansion of that integral.  The closed forms are

    ``Nblk[m, 0] = 1 / (2**J * (m + 1))`` for even ``m`` and zero otherwise,

    while ``Mblk`` has the coefficients on modes ``m+1``, ``m-1``, and zero
    derived from ``2*T_(m+1) = U_(m+1) - U_(m-1)``.  The missing ``m+1`` term
    on the last row is intentionally the retained-basis truncation.
    """
    _validate_dimensions(level_j, mode_count)
    modes = np.arange(mode_count)
    n_block = np.zeros((mode_count, mode_count), dtype=float)
    even_modes = (modes % 2) == 0
    n_block[even_modes, 0] = 1.0 / (2.0**level_j * (modes[even_modes] + 1.0))

    m_block = np.zeros((mode_count, mode_count), dtype=float)
    denominator = 2.0 ** (level_j + 2) * (modes + 1.0)
    m_block[modes, 0] += ((-1.0) ** modes) / (2.0 ** (level_j + 1) * (modes + 1.0))
    lower = modes[1:]
    m_block[lower, lower - 1] -= 1.0 / denominator[lower]
    upper = modes[:-1]
    m_block[upper, upper + 1] += 1.0 / denominator[upper]
    return n_block, m_block


def operational_matrix_of_integration(level_j: int, mode_count: int) -> np.ndarray:
    """Return the full OMI for all ``2**J`` cells.

    Cell-major ordering is used: flatten index ``cell * mode_count + mode``.
    The full operator is ``I kron Mblk + U kron Nblk``, where ``U`` is strictly
    upper triangular because an integral of a wavelet becomes a constant on
    every later cell.
    """
    n_block, m_block = integration_blocks(level_j, mode_count)
    cell_count = 2**level_j
    later_cells = np.triu(np.ones((cell_count, cell_count), dtype=float), k=1)
    return np.kron(np.eye(cell_count), m_block) + np.kron(later_cells, n_block)


def chebyshev_u_linearization(mode_count: int) -> np.ndarray:
    """Return ``Lambda[l, j, i]`` for ``U_l * U_j = sum_i Lambda[l,j,i] U_i``.

    Only retained degrees are represented.  The Boolean construction is fully
    vectorized and encodes the standard range ``|l-j| <= i <= l+j`` with the
    required parity restriction.
    """
    if not isinstance(mode_count, (int, np.integer)) or mode_count < 1:
        raise ValueError("mode_count must be a positive integer")
    degree_l, degree_j, degree_i = np.indices((mode_count, mode_count, mode_count))
    retained_term = (
        (degree_i >= np.abs(degree_l - degree_j))
        & (degree_i <= degree_l + degree_j)
        & (((degree_l + degree_j - degree_i) % 2) == 0)
    )
    return retained_term.astype(float)


def product_operational_matrix(level_j: int, coefficients: np.ndarray) -> np.ndarray:
    """Build the block-diagonal POM for a wavelet expansion.

    ``coefficients`` is shaped ``(2**J, M)`` and represents
    ``f = sum_{n,l} coefficients[n,l] * psi_(n,l)``.  The returned matrix has
    entry ``[a, b] = <f * psi_b, psi_a>`` in the retained weighted basis.
    Cross-cell blocks vanish by disjoint support; same-cell blocks use the
    proved U-polynomial linearization tensor.
    """
    coefficient_array = np.asarray(coefficients, dtype=float)
    if coefficient_array.ndim != 2:
        raise ValueError("coefficients must have shape (cell_count, mode_count)")
    cell_count, mode_count = coefficient_array.shape
    expected_cells = 2**level_j
    if cell_count != expected_cells:
        raise ValueError(f"expected {expected_cells} cells at level_j={level_j}, got {cell_count}")
    _validate_dimensions(level_j, mode_count)

    linearization = chebyshev_u_linearization(mode_count)
    block_size = mode_count
    pom = np.zeros((cell_count * block_size, cell_count * block_size), dtype=float)
    scale = wavelet_scale(level_j)
    for cell_index in range(cell_count):
        # einsum produces [output_mode, input_mode] from Lambda[l,input,output].
        block = scale * np.einsum("l,lji->ij", coefficient_array[cell_index], linearization)
        start = cell_index * block_size
        pom[start : start + block_size, start : start + block_size] = block
    return pom


def _validate_dimensions(level_j: int, mode_count: int) -> None:
    if not isinstance(level_j, (int, np.integer)) or level_j < 0:
        raise ValueError("level_j must be a non-negative integer")
    if not isinstance(mode_count, (int, np.integer)) or mode_count < 1:
        raise ValueError("mode_count must be a positive integer")
