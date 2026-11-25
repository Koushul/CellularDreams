"""Realism statistics from S3. Real matrices are optional."""

from __future__ import annotations

import numpy as np


RANGES = {
    "log10_library_cv": (0.15, 0.60),
    "zero_fraction": (0.70, 0.95),
    "median_abs_correlation": (0.02, 0.35),
    "morans_i": (0.15, 0.85),
    "displacement_to_noise": (0.30, 3.00),
    "cycle_division_spearman": (0.50, 1.01),
}


def log10_library_cv(counts: np.ndarray) -> float:
    library = counts.sum(axis=1)
    logged = np.log10(library + 1.0)
    return float(logged.std() / max(logged.mean(), 1e-9))


def zero_fraction(counts: np.ndarray) -> float:
    return float(np.mean(counts == 0))


def median_abs_correlation(counts: np.ndarray) -> float:
    if counts.shape[1] < 3:
        return 0.0
    variance = counts.var(axis=0)
    varying = np.flatnonzero(variance > 0)
    if len(varying) < 2:
        return 0.0
    keep = varying[np.argsort(variance[varying])[-min(50, len(varying)):]]
    subset = counts[:, keep]
    corr = np.corrcoef(subset, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    return float(np.median(np.abs(corr[mask])))


def morans_i(values: np.ndarray, positions: np.ndarray, length_scale: float = 30.0) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    delta = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    weights = np.exp(-distance / length_scale)
    np.fill_diagonal(weights, 0.0)
    z = values - values.mean()
    denominator = weights.sum() * np.dot(z, z)
    if denominator == 0:
        return 0.0
    numerator = n * float(z @ weights @ z)
    return numerator / denominator


def displacement_to_noise(start: np.ndarray, end: np.ndarray, expression: np.ndarray) -> float:
    """Median displacement (μm) divided by median pairwise log-expression distance."""
    movement = np.linalg.norm(end - start, axis=1)
    logged = np.log1p(np.clip(expression, 0, None))
    diff = logged[:, None, :] - logged[None, :, :]
    distance = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(len(expression), k=1)
    scale = float(np.median(distance[iu])) if len(iu[0]) else 1.0
    return float(np.median(movement) / max(scale, 1e-9))


def in_range(name: str, value: float) -> bool:
    low, high = RANGES[name]
    return low <= value <= high
