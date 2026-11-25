"""Grid layouts in the spirit of scMultiSim's spatial placements.

Coordinates are micrometres on a square domain. Neighbors are the occupied
orthogonal grid cells, the same four directions the R grid uses.
"""

from __future__ import annotations

from collections import deque

import numpy as np

_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def grid_size(n_cells: int) -> int:
    return max(8, int(np.ceil(np.sqrt(n_cells) * 3)))


def place(layout: str, n_cells: int, n_types: int, rng: np.random.Generator, domain_um: float = 200.0):
    """Return positions, cell types, and neighbor index lists."""
    if n_types < 1:
        raise ValueError("n_types must be positive")
    if n_cells < n_types:
        raise ValueError("n_cells must be at least n_types")
    size = grid_size(n_cells)
    if layout == "basic":
        occupancy = _basic(n_cells, n_types, size, rng)
    elif layout == "enhanced":
        occupancy = _enhanced(n_cells, n_types, size, rng)
    elif layout == "layers":
        occupancy = _layers(n_cells, n_types, size)
    elif layout == "islands":
        occupancy = _islands(n_cells, n_types, size)
    else:
        raise ValueError(f"unknown layout {layout}")
    return _pack(occupancy, size, domain_um)


def _basic(n_cells: int, n_types: int, size: int, rng: np.random.Generator) -> list[tuple[int, int, int]]:
    slots = [(x, y) for y in range(size) for x in range(size)]
    chosen = rng.choice(len(slots), size=n_cells, replace=False)
    return [(slots[int(i)][0], slots[int(i)][1], k % n_types) for k, i in enumerate(chosen)]


def _enhanced(n_cells: int, n_types: int, size: int, rng: np.random.Generator) -> list[tuple[int, int, int]]:
    occupied = -np.ones((size, size), dtype=int)
    placed: list[tuple[int, int, int]] = []
    center = size // 2
    _put(occupied, placed, center, center, 0)
    for index in range(1, n_cells):
        cell_type = index % n_types
        same: list[tuple[int, int]] = []
        other: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for y, x in zip(*np.where(occupied >= 0)):
            neighbor_type = placed[int(occupied[y, x])][2]
            for nx, ny in _empty_neighbors(int(x), int(y), occupied):
                if (nx, ny) in seen:
                    continue
                seen.add((nx, ny))
                if neighbor_type == cell_type:
                    same.append((nx, ny))
                else:
                    other.append((nx, ny))
        if same and (not other or rng.random() < 0.8):
            pool = same
        else:
            pool = other or same
        if not pool:
            empty = np.argwhere(occupied < 0)
            pick = empty[int(rng.integers(len(empty)))]
            nx, ny = int(pick[1]), int(pick[0])
        else:
            nx, ny = pool[int(rng.integers(len(pool)))]
        _put(occupied, placed, nx, ny, cell_type)
    return placed


def _layers(n_cells: int, n_types: int, size: int) -> list[tuple[int, int, int]]:
    quotas = _quotas(n_cells, n_types)
    bands = np.array_split(np.arange(size), n_types)
    occupied = -np.ones((size, size), dtype=int)
    placed: list[tuple[int, int, int]] = []
    for cell_type, quota in enumerate(quotas):
        filled = 0
        for y in bands[cell_type]:
            for x in range(size):
                if filled == quota:
                    break
                _put(occupied, placed, int(x), int(y), cell_type)
                filled += 1
            if filled == quota:
                break
        if filled != quota:
            raise RuntimeError("layer band ran out of squares")
    return placed


def _islands(n_cells: int, n_types: int, size: int) -> list[tuple[int, int, int]]:
    quotas = _quotas(n_cells, n_types)
    occupied = -np.ones((size, size), dtype=int)
    placed: list[tuple[int, int, int]] = []
    frontiers: list[deque[tuple[int, int]]] = []
    remaining = quotas[:]
    for cell_type in range(n_types):
        x = int((cell_type + 1) * size / (n_types + 1))
        y = size // 2
        frontiers.append(deque())
        if remaining[cell_type] == 0:
            continue
        _put(occupied, placed, x, y, cell_type)
        remaining[cell_type] -= 1
        frontiers[cell_type].extend(_empty_neighbors(x, y, occupied))
    guard = 0
    while any(count > 0 for count in remaining):
        guard += 1
        if guard > n_cells * size * size:
            raise RuntimeError("island placement did not finish")
        grew = False
        for cell_type in range(n_types):
            if remaining[cell_type] == 0:
                continue
            spot = _next_empty(frontiers[cell_type], occupied)
            if spot is None:
                center_x = int((cell_type + 1) * size / (n_types + 1))
                spot = _nearest_empty(center_x, size // 2, occupied)
            if spot is None:
                continue
            nx, ny = spot
            _put(occupied, placed, nx, ny, cell_type)
            remaining[cell_type] -= 1
            frontiers[cell_type].extend(_empty_neighbors(nx, ny, occupied))
            grew = True
        if not grew:
            raise RuntimeError("no empty square left for an island")
    return placed


def _quotas(n_cells: int, n_types: int) -> list[int]:
    quotas = [n_cells // n_types] * n_types
    for index in range(n_cells % n_types):
        quotas[index] += 1
    return quotas


def _put(occupied: np.ndarray, placed: list[tuple[int, int, int]], x: int, y: int, cell_type: int) -> None:
    occupied[y, x] = len(placed)
    placed.append((x, y, cell_type))


def _empty_neighbors(x: int, y: int, occupied: np.ndarray) -> list[tuple[int, int]]:
    size = occupied.shape[0]
    found = []
    for dx, dy in _STEPS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size and occupied[ny, nx] < 0:
            found.append((nx, ny))
    return found


def _next_empty(frontier: deque[tuple[int, int]], occupied: np.ndarray) -> tuple[int, int] | None:
    while frontier:
        x, y = frontier.popleft()
        if occupied[y, x] < 0:
            return x, y
    return None


def _nearest_empty(x: int, y: int, occupied: np.ndarray) -> tuple[int, int] | None:
    empty = np.argwhere(occupied < 0)
    if len(empty) == 0:
        return None
    distance = (empty[:, 1] - x) ** 2 + (empty[:, 0] - y) ** 2
    pick = empty[int(np.argmin(distance))]
    return int(pick[1]), int(pick[0])


def _pack(placed: list[tuple[int, int, int]], size: int, domain_um: float):
    pitch = domain_um / max(size - 1, 1)
    order = np.arange(len(placed))
    positions = np.zeros((len(placed), 2), dtype=float)
    cell_type = np.zeros(len(placed), dtype=int)
    lookup = {}
    for index, (x, y, kind) in enumerate(placed):
        positions[index] = (x * pitch, y * pitch)
        cell_type[index] = kind
        lookup[(x, y)] = index
    neighbors: list[np.ndarray] = []
    edges = []
    for index, (x, y, _) in enumerate(placed):
        found = []
        for dx, dy in _STEPS:
            other = lookup.get((x + dx, y + dy))
            if other is None:
                continue
            found.append(other)
            if index < other:
                edges.append((index, other))
        neighbors.append(np.asarray(found, dtype=int))
    return {
        "positions": positions,
        "cell_type": cell_type,
        "neighbors": neighbors,
        "edges": np.asarray(edges, dtype=int).reshape(-1, 2),
        "pitch_um": float(pitch),
        "grid_size": size,
        "order": order,
    }
