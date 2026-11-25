"""Platform observation models applied after the tissue simulation."""

from __future__ import annotations

import numpy as np


def restrict_panel(counts: np.ndarray, gene_index: np.ndarray) -> np.ndarray:
    return counts[:, gene_index]


def spillover(counts: np.ndarray, positions: np.ndarray, fraction: float) -> np.ndarray:
    """Move ``fraction`` of each cell's transcripts to its nearest neighbor.

    Total counts are unchanged.
    """
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("spillover fraction must be in [0, 1]")
    n = counts.shape[0]
    if n < 2 or fraction == 0.0:
        return counts.copy()
    delta = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argmin(distance, axis=1)
    out = counts * (1.0 - fraction)
    for i, j in enumerate(nearest):
        out[j] += counts[i] * fraction
    return out


def _sum_within(
    counts: np.ndarray, positions: np.ndarray, centers: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each cell to the nearest center within ``radius``.

    Returns aggregated counts and the number of cells per center. A cell in
    range of several centers is counted once, at the nearest center, so the
    sum of aggregated counts equals the sum of captured cell counts.
    """
    delta = positions[:, None, :] - centers[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    nearest = np.argmin(distance, axis=1)
    nearest_distance = distance[np.arange(len(positions)), nearest]
    captured = nearest_distance <= radius
    agg = np.zeros((len(centers), counts.shape[1]), dtype=float)
    occupancy = np.zeros(len(centers), dtype=int)
    for i in np.flatnonzero(captured):
        center = nearest[i]
        agg[center] += counts[i]
        occupancy[center] += 1
    return agg, occupancy


def hex_centers(pitch: float, width: float) -> np.ndarray:
    dy = pitch * np.sqrt(3.0) / 2.0
    centers = []
    row = 0
    y = 0.0
    while y <= width + 1e-9:
        x = (pitch / 2.0) if row % 2 else 0.0
        while x <= width + 1e-9:
            centers.append((x, y))
            x += pitch
        y += dy
        row += 1
    return np.asarray(centers, dtype=float)


def visium_spots(
    counts: np.ndarray, positions: np.ndarray, width: float, pitch: float = 100.0, radius: float = 55.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = hex_centers(pitch, width)
    agg, occupancy = _sum_within(counts, positions, centers, radius)
    return agg, centers, occupancy


def slideseq_beads(
    counts: np.ndarray, positions: np.ndarray, width: float, pitch: float = 10.0, radius: float = 10.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.arange(0.0, width + 1e-9, pitch)
    grid = np.stack(np.meshgrid(xs, xs, indexing="xy"), axis=-1).reshape(-1, 2)
    agg, occupancy = _sum_within(counts, positions, grid, radius)
    return agg, grid, occupancy


def batch_effect(counts: np.ndarray, factor: float) -> np.ndarray:
    """Multiply every count by one library factor."""
    return counts * factor
