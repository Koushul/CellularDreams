"""Precommitted scenes from PRECOMMIT_TRAVLR_SIM.md (S4, S5)."""

from __future__ import annotations

import numpy as np

from .simulate import Cell, Edge, Scene

GENES = ["ligand", "decoy", "null", "receptor", "receptor_b", "target", "tf", "cycle"]
MASTER_SEED = 20241124


def scene_seed(scene_id: int) -> int:
    return MASTER_SEED + 1000 * scene_id


def _cell(
    index: int,
    pos: np.ndarray,
    *,
    speed: float,
    division_rate: float,
    moves: bool,
    chemotaxis: bool,
    receptor: float,
    receptor_b: float = 1.0,
    ligand: float = 4.0,
    heading: float = 0.0,
) -> Cell:
    spliced = np.array([ligand, 3.0, 3.0, receptor, receptor_b, 2.0, 2.0, division_rate * 50.0])
    return Cell(
        id=index,
        parent=None,
        birth=0.0,
        death=None,
        pos=np.asarray(pos, dtype=float),
        heading=heading,
        speed=speed,
        unspliced=spliced.copy(),
        spliced=spliced,
        promoter_on=np.ones(len(GENES), dtype=bool),
        receptor_subunits=np.array([receptor, receptor_b], dtype=float),
        division_rate=division_rate,
        moves=moves,
        chemotaxis=chemotaxis,
    )


def _population(rng: np.random.Generator, n: int, **kwargs) -> list[Cell]:
    positions = rng.uniform(20.0, 180.0, size=(n, 2))
    rates = rng.uniform(0.0, 0.15, size=n)
    cells = []
    for i in range(n):
        cells.append(
            _cell(
                i,
                positions[i],
                division_rate=float(rates[i]),
                receptor=float(kwargs.get("receptor", 2.0)),
                **{k: v for k, v in kwargs.items() if k != "receptor"},
            )
        )
    return cells


def _ligand_edge(value: float, source: int = 0, target: int = 5, scale: float = 40.0) -> Edge:
    return Edge(source, target, [(0.0, value), (12.0, value)], "ligand", scale)


def build_scene(name: str) -> tuple[Scene, np.random.Generator]:
    """Return the scene and the seeded generator specified in S5."""
    if name not in _CATALOG:
        raise KeyError(name)
    scene_id, builder = _CATALOG[name]
    rng = np.random.default_rng(scene_seed(scene_id))
    scene = builder(rng)
    scene.name = name
    scene.scene_id = scene_id
    return scene, rng


def _base(rng, scene_id: int, name: str, **overrides) -> Scene:
    cells = overrides.pop("cells", None)
    if cells is None:
        cells = _population(rng, 40, speed=10.0, moves=True, chemotaxis=False)
    edges = overrides.pop("edges", [_ligand_edge(0.4), Edge(6, 5, [(0.0, 0.2), (12.0, 0.2)], "tf")])
    scene = Scene(
        name=name,
        scene_id=scene_id,
        gene_names=list(GENES),
        edges=edges,
        cells=cells,
        ligand_gene=0,
        receptor_genes=[3, 4],
        target_gene=5,
        cycle_gene=7,
        decoy_gene=1,
        null_gene=2,
    )
    for key, value in overrides.items():
        setattr(scene, key, value)
    return scene


def _coupled(rng):
    return _base(rng, 1, "coupled")


def _fork(rng):
    left = [_cell(i, np.array([40.0, 40.0 + 3 * i]), speed=12.0, division_rate=0.02, moves=True, chemotaxis=False, receptor=2.0, heading=0.0) for i in range(20)]
    right = [_cell(20 + i, np.array([160.0, 40.0 + 3 * i]), speed=12.0, division_rate=0.02, moves=True, chemotaxis=False, receptor=2.0, heading=np.pi) for i in range(20)]
    return _base(rng, 2, "fork", cells=left + right)


def _crossing(rng):
    going = [_cell(i, np.array([30.0, 100.0 + i]), speed=15.0, division_rate=0.0, moves=True, chemotaxis=False, receptor=2.0, heading=0.0) for i in range(20)]
    back = [_cell(20 + i, np.array([170.0, 90.0 + i]), speed=15.0, division_rate=0.0, moves=True, chemotaxis=False, receptor=2.0, heading=np.pi) for i in range(20)]
    return _base(rng, 3, "crossing", cells=going + back)


def _radial(rng):
    cells = []
    for i in range(40):
        angle = 2 * np.pi * i / 40
        cells.append(
            _cell(i, np.array([100.0, 100.0]) + 15 * np.array([np.cos(angle), np.sin(angle)]), speed=8.0, division_rate=0.01, moves=True, chemotaxis=False, receptor=2.0, heading=angle)
        )
    return _base(rng, 4, "radial", cells=cells)


def _rim(rng):
    cells = []
    for i in range(40):
        cells.append(_cell(i, np.array([10.0 + 4 * i, 10.0]), speed=20.0, division_rate=0.0, moves=True, chemotaxis=False, receptor=1.0, heading=0.0))
    return _base(rng, 5, "rim", cells=cells)


def _zonation(rng):
    cells = []
    for i in range(40):
        cells.append(
            _cell(i, np.array([5.0 + i * 4.5, 100.0]), speed=0.0, division_rate=0.0, moves=False, chemotaxis=False, receptor=2.0)
        )
    return _base(rng, 6, "zonation", cells=cells, stationary=True, zonation=True, edges=[])


def _signal(rng):
    cells = _population(rng, 40, speed=0.0, moves=False, chemotaxis=False, receptor=2.0)
    return _base(rng, 7, "signal_in_place", cells=cells, stationary=True, signal_on_time=6.0)


def _decoy(rng):
    edges = [_ligand_edge(0.5, 0, 5), _ligand_edge(0.0, 1, 5)]
    return _base(rng, 8, "decoy", edges=edges)


def _il21(rng):
    cells = []
    for i in range(40):
        positive = i % 2 == 0
        cells.append(
            _cell(
                i,
                np.array([30.0 + (i % 8) * 20, 30.0 + (i // 8) * 30]),
                speed=5.0,
                division_rate=0.03,
                moves=True,
                chemotaxis=False,
                receptor=3.0 if positive else 0.0,
                receptor_b=3.0 if positive else 0.0,
                ligand=6.0,
            )
        )
    edges = [_ligand_edge(0.8, 0, 5), _ligand_edge(0.0, 1, 5), _ligand_edge(0.0, 2, 5)]
    return _base(rng, 9, "il21_like", cells=cells, edges=edges)


def _time_varying(rng):
    edges = [
        Edge(0, 5, [(0.0, 0.4), (6.0, 0.4), (6.0, 0.8), (12.0, 0.8)], "ligand", 40.0),
        Edge(6, 5, [(0.0, 0.2), (6.0, 0.2), (6.0, 0.4), (12.0, 0.4)], "tf"),
    ]
    return _base(
        rng,
        10,
        "time_varying",
        edges=edges,
        sample_times=[4.0, 8.0, 12.0],
        sample_fraction=0.3,
    )


def _chemotaxis(rng):
    source = [_cell(i, np.array([180.0, 80.0 + 4 * i]), speed=0.0, division_rate=0.0, moves=False, chemotaxis=False, receptor=0.0, ligand=30.0) for i in range(5)]
    movers = []
    for i in range(20):
        positive = i < 10
        movers.append(
            _cell(
                5 + i,
                np.array([40.0, 70.0 + 3 * i]),
                speed=20.0,
                division_rate=0.0,
                moves=True,
                chemotaxis=True,
                receptor=4.0 if positive else 0.0,
                receptor_b=4.0 if positive else 0.0,
                ligand=0.0,
                heading=np.pi / 2,
            )
        )
    return _base(rng, 11, "chemotaxis", cells=source + movers, edges=[_ligand_edge(0.3)])


_CATALOG = {
    "coupled": (1, _coupled),
    "fork": (2, _fork),
    "crossing": (3, _crossing),
    "radial": (4, _radial),
    "rim": (5, _rim),
    "zonation": (6, _zonation),
    "signal_in_place": (7, _signal),
    "decoy": (8, _decoy),
    "il21_like": (9, _il21),
    "time_varying": (10, _time_varying),
    "chemotaxis": (11, _chemotaxis),
}
