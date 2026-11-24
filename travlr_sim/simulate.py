"""Joint tissue kernel and kinetic expression loop."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .exposure import pairwise_exposure, receptor_gate
from .kinetics import piecewise_linear, telegraph_step


@dataclass
class Edge:
    """Regulatory edge. ``kind`` is ``tf`` or ``ligand``."""

    source: int
    target: int
    knots: list[tuple[float, float]]
    kind: str
    length_scale: float = 40.0


@dataclass
class Cell:
    id: int
    parent: int | None
    birth: float
    death: float | None
    pos: np.ndarray
    heading: float
    speed: float
    unspliced: np.ndarray
    spliced: np.ndarray
    promoter_on: np.ndarray
    receptor_subunits: np.ndarray
    division_rate: float
    moves: bool
    chemotaxis: bool
    absorb: bool = False
    alive: bool = True
    samples: list[float] = field(default_factory=list)


@dataclass
class Scene:
    """One precommitted scene. Gene order is fixed and documented in ``gene_names``."""

    name: str
    scene_id: int
    gene_names: list[str]
    edges: list[Edge]
    cells: list[Cell]
    width: float = 200.0
    dt: float = 0.5
    duration: float = 12.0
    contact_scale: float = 10.0
    secreted_scale: float = 40.0
    sample_times: list[float] = field(default_factory=list)
    sample_fraction: float = 0.0
    spatial_times: list[float] | None = None
    ligand_gene: int = 0
    receptor_genes: list[int] = field(default_factory=lambda: [1])
    target_gene: int = 2
    cycle_gene: int = 3
    decoy_gene: int | None = None
    null_gene: int | None = None
    stationary: bool = False
    zonation: bool = False
    signal_on_time: float | None = None
    knockout_time: float | None = None
    knockout_gene: int | None = None
    knockout_cells: list[int] | None = None


@dataclass
class History:
    time: list[float]
    cell_ids: list[np.ndarray]
    positions: list[np.ndarray]
    spliced: list[np.ndarray]
    unspliced: list[np.ndarray]
    velocity: list[np.ndarray]
    exposure: list[np.ndarray]
    impact: list[np.ndarray]
    coefficients: list[np.ndarray]
    division_rate: list[np.ndarray]
    cycle_score: list[np.ndarray]
    samples: list[tuple[float, np.ndarray]]


def _living(cells: list[Cell]) -> list[Cell]:
    return [cell for cell in cells if cell.alive]


def simulate(scene: Scene, rng: np.random.Generator) -> tuple[list[Cell], History]:
    """Run the joint kernel and the kinetic engine from ``t = 0`` through ``duration``."""
    history = History([], [], [], [], [], [], [], [], [], [], [], [])
    n_steps = int(round(scene.duration / scene.dt))
    next_id = max(cell.id for cell in scene.cells) + 1
    for step in range(n_steps + 1):
        time = step * scene.dt
        _apply_perturbation(scene, time)
        living = _living(scene.cells)
        exposure, impact, coefficients = _express(scene, living, time)
        _record(history, scene, living, time, exposure, impact, coefficients)
        _maybe_sample(scene, living, time, rng)
        if step == n_steps:
            break
        next_id = _tissue_step(scene, living, time, rng, next_id)
    return scene.cells, history


def _apply_perturbation(scene: Scene, time: float) -> None:
    if scene.knockout_time is None or time + 1e-9 < scene.knockout_time:
        return
    chosen = set(scene.knockout_cells or [])
    for cell in _living(scene.cells):
        if chosen and cell.id not in chosen:
            continue
        gene = scene.knockout_gene
        if gene is None:
            continue
        cell.spliced[gene] = 0.0
        cell.unspliced[gene] = 0.0
        if gene in scene.receptor_genes:
            subunit = scene.receptor_genes.index(gene)
            cell.receptor_subunits[subunit] = 0.0


def _express(
    scene: Scene, living: list[Cell], time: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(living)
    n_genes = len(scene.gene_names)
    positions = np.stack([cell.pos for cell in living]) if n else np.zeros((0, 2))
    spliced = np.stack([cell.spliced for cell in living]) if n else np.zeros((0, n_genes))
    ligand_edges = [edge for edge in scene.edges if edge.kind == "ligand"]
    exposure = np.zeros((n, max(len(ligand_edges), 1)))
    impact = np.zeros((n, n_genes))
    coefficients = np.zeros(len(scene.edges))
    for index, edge in enumerate(ligand_edges):
        scale = edge.length_scale
        level = spliced[:, edge.source] if n else np.zeros(0)
        if scene.signal_on_time is not None and time + 1e-9 < scene.signal_on_time:
            level = np.zeros_like(level)
        exposure[:, index] = pairwise_exposure(positions, level, scale) if n else exposure[:, index]
    subunits = (
        np.stack([cell.receptor_subunits for cell in living]) if n else np.zeros((0, 1))
    )
    gate = receptor_gate(subunits) if n else np.zeros(0)
    for index, edge in enumerate(scene.edges):
        coef = piecewise_linear(edge.knots, time)
        coefficients[index] = coef
        if edge.kind == "tf":
            driver = spliced[:, edge.source] if n else 0.0
            impact[:, edge.target] += coef * driver
        else:
            ligand_index = ligand_edges.index(edge)
            impact[:, edge.target] += coef * gate * exposure[:, ligand_index]
    if scene.zonation and n:
        morphogen = positions[:, 0] / scene.width
        impact[:, scene.target_gene] += morphogen
    for i, cell in enumerate(living):
        for gene in range(n_genes):
            if scene.gene_names[gene].startswith("receptor"):
                continue
            knocked_out = _gene_knocked_out(scene, cell, gene, time)
            rate = 0.0 if knocked_out else _basal_rate(scene, gene)
            rate = rate * max(0.0, 1.0 + impact[i, gene])
            on = bool(cell.promoter_on[gene])
            beta = 0.4
            degradation = 1.0
            if knocked_out:
                cell.unspliced[gene] = 0.0
                cell.spliced[gene] = 0.0
            else:
                cell.unspliced[gene], cell.spliced[gene] = telegraph_step(
                    cell.unspliced[gene],
                    cell.spliced[gene],
                    on,
                    rate,
                    beta,
                    degradation,
                    scene.dt if time > 0 else 0.0,
                )
        if scene.cycle_gene is not None:
            cell.spliced[scene.cycle_gene] = cell.division_rate * 50.0
        for subunit, gene in enumerate(scene.receptor_genes):
            if _gene_knocked_out(scene, cell, gene, time):
                cell.receptor_subunits[subunit] = 0.0
            cell.spliced[gene] = cell.receptor_subunits[subunit]
            cell.unspliced[gene] = cell.receptor_subunits[subunit]
    return exposure, impact, coefficients


def _basal_rate(scene: Scene, gene: int) -> float:
    name = scene.gene_names[gene]
    if name in {"null", "decoy"}:
        return 8.0
    if name == "ligand":
        return 12.0
    if name.startswith("receptor"):
        return 6.0
    if name == "target":
        return 4.0
    if name == "tf":
        return 5.0
    if name == "cycle":
        return 1.0
    return 3.0


def _gene_knocked_out(scene: Scene, cell: Cell, gene: int, time: float) -> bool:
    if scene.knockout_time is None or scene.knockout_gene != gene:
        return False
    if time + 1e-9 < scene.knockout_time:
        return False
    chosen = scene.knockout_cells
    return chosen is None or cell.id in chosen


def _record(history, scene, living, time, exposure, impact, coefficients) -> None:
    if not living:
        return
    spliced = np.stack([cell.spliced.copy() for cell in living])
    unspliced = np.stack([cell.unspliced.copy() for cell in living])
    history.time.append(time)
    history.cell_ids.append(np.array([cell.id for cell in living], dtype=int))
    history.positions.append(np.stack([cell.pos.copy() for cell in living]))
    history.spliced.append(spliced)
    history.unspliced.append(unspliced)
    history.velocity.append(0.4 * unspliced - 1.0 * spliced)
    history.exposure.append(exposure.copy())
    history.impact.append(impact.copy())
    history.coefficients.append(coefficients.copy())
    history.division_rate.append(np.array([cell.division_rate for cell in living]))
    history.cycle_score.append(spliced[:, scene.cycle_gene].copy())


def _maybe_sample(scene, living, time, rng) -> None:
    if scene.sample_fraction <= 0.0 or not any(abs(time - t) < 1e-8 for t in scene.sample_times):
        return
    eligible = [cell for cell in living if not cell.samples]
    k = int(round(scene.sample_fraction * len(eligible)))
    if k <= 0 or not eligible:
        return
    chosen = rng.choice(len(eligible), size=min(k, len(eligible)), replace=False)
    for index in np.atleast_1d(chosen):
        eligible[int(index)].samples.append(time)


def _tissue_step(scene, living, time, rng, next_id: int) -> int:
    born: list[Cell] = []
    for cell in living:
        if scene.stationary or not cell.moves:
            continue
        if rng.random() > 0.7:
            continue
        turn = rng.normal(0.0, 0.15)
        direction = np.array([np.cos(cell.heading), np.sin(cell.heading)])
        if cell.chemotaxis and float(np.min(cell.receptor_subunits)) > 0.0:
            direction = _chemotaxis_direction(scene, cell, direction)
        cell.heading = float(np.arctan2(direction[1], direction[0])) + turn
        step = np.array([np.cos(cell.heading), np.sin(cell.heading)]) * cell.speed * scene.dt
        proposed = cell.pos + step
        if cell.absorb and (np.any(proposed <= 0.0) or np.any(proposed >= scene.width)):
            cell.alive = False
            cell.death = time + scene.dt
            continue
        cell.pos = np.clip(proposed, 0.0, scene.width)
        if cell.pos[0] >= scene.width - 1.0 and cell.speed > 0 and scene.name == "rim":
            cell.heading = float(np.pi / 2)
    for cell in living:
        probability = 1.0 - np.exp(-cell.division_rate * scene.dt)
        if rng.random() < probability:
            daughter, next_id = _divide(cell, time + scene.dt, rng, next_id, scene)
            born.append(daughter)
    scene.cells.extend(born)
    return next_id


def _chemotaxis_direction(scene: Scene, cell: Cell, direction: np.ndarray) -> np.ndarray:
    others = [other for other in _living(scene.cells) if other.id != cell.id]
    if not others:
        return direction
    gradient = np.zeros(2)
    for other in others:
        delta = other.pos - cell.pos
        distance = float(np.linalg.norm(delta)) + 1e-6
        weight = other.spliced[scene.ligand_gene] * np.exp(-distance / scene.secreted_scale)
        gradient += weight * delta / distance
    if np.linalg.norm(gradient) == 0.0:
        return direction
    gradient = gradient / np.linalg.norm(gradient)
    mixed = 0.5 * direction + 0.5 * gradient
    return mixed / np.linalg.norm(mixed)


def _divide(parent: Cell, time: float, rng: np.random.Generator, next_id: int, scene: Scene) -> tuple[Cell, int]:
    axis = rng.normal(size=2)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    offset = 3.0 * axis
    parent.pos = np.clip(parent.pos - offset, 0.0, scene.width)
    noise = rng.normal(0.0, 0.05, size=parent.spliced.shape)
    daughter = Cell(
        id=next_id,
        parent=parent.id,
        birth=time,
        death=None,
        pos=np.clip(parent.pos + 2 * offset, 0.0, scene.width),
        heading=parent.heading + np.pi,
        speed=parent.speed,
        unspliced=np.clip(parent.unspliced * (1.0 + noise), 0.0, None),
        spliced=np.clip(parent.spliced * (1.0 + noise), 0.0, None),
        promoter_on=parent.promoter_on.copy(),
        receptor_subunits=parent.receptor_subunits.copy(),
        division_rate=parent.division_rate,
        moves=parent.moves,
        chemotaxis=parent.chemotaxis,
        absorb=parent.absorb,
    )
    return daughter, next_id + 1
