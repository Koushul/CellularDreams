"""Distance-weighted ligand exposure and receptor gates."""

from __future__ import annotations

import numpy as np


def pairwise_exposure(
    positions: np.ndarray,
    sender_level: np.ndarray,
    length_scale: float,
) -> np.ndarray:
    """``L_i = sum_j x_j exp(-d_ij / lambda)``, excluding self.

    ``positions`` is ``(n, 2)`` in micrometres. ``sender_level`` is ``(n,)``.
    """
    delta = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    kernel = np.exp(-distance / length_scale)
    np.fill_diagonal(kernel, 0.0)
    return kernel @ sender_level


def receptor_gate(subunits: np.ndarray) -> np.ndarray:
    """Geometric mean of receptor subunits. A zero subunit closes the gate.

    ``subunits`` has shape ``(n_cells, n_subunits)`` and is non-negative.
    """
    clipped = np.clip(subunits, 0.0, None)
    if np.any(clipped <= 0.0):
        closed = np.any(clipped <= 0.0, axis=1)
    else:
        closed = np.zeros(clipped.shape[0], dtype=bool)
    with np.errstate(divide="ignore"):
        logs = np.log(np.where(clipped > 0.0, clipped, 1.0))
    gate = np.exp(logs.mean(axis=1))
    gate[closed] = 0.0
    return gate
