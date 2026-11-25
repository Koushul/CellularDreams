"""Kinetic updates following scMultiSim's Beta-Poisson and telegraph model."""

from __future__ import annotations

import numpy as np


def beta_poisson_mean(kon: float, koff: float, s: float) -> float:
    """Expected count when intrinsic noise is 1: ``s * kon / (kon + koff)``."""
    return s * kon / (kon + koff)


def beta_poisson_draw(
    rng: np.random.Generator, kon: float, koff: float, s: float, n: int
) -> np.ndarray:
    """One Beta-Poisson count per cell, matching ``.betaPoisson`` with noise 1."""
    activity = rng.beta(kon, koff, size=n)
    return rng.poisson(activity * s).astype(float)


def telegraph_step(
    unspliced: float,
    spliced: float,
    promoter_on: bool,
    s_rate: float,
    beta: float,
    degradation: float,
    dt: float,
) -> tuple[float, float]:
    """One scMultiSim telegraph step.

    The spliced update uses the unspliced level from before this step, as
    ``gen_1branch`` does (``y`` uses ``x[t - 1]``).
    """
    previous = unspliced
    if promoter_on:
        unspliced = unspliced + s_rate * dt - beta * unspliced * dt
    else:
        unspliced = unspliced - beta * unspliced * dt
    unspliced = max(unspliced, 0.0)
    spliced = spliced + beta * previous * dt - degradation * spliced * dt
    spliced = max(spliced, 0.0)
    return unspliced, spliced


def piecewise_linear(knots: list[tuple[float, float]], time: float) -> float:
    """Interpolate a coefficient. Knots are ``(time, value)`` in hours."""
    ordered = sorted(knots, key=lambda item: item[0])
    if time <= ordered[0][0]:
        return float(ordered[0][1])
    if time >= ordered[-1][0]:
        return float(ordered[-1][1])
    for (t0, v0), (t1, v1) in zip(ordered, ordered[1:]):
        if t0 <= time <= t1:
            if t1 == t0:
                return float(v1)
            weight = (time - t0) / (t1 - t0)
            return float(v0 * (1.0 - weight) + v1 * weight)
    return float(ordered[-1][1])
