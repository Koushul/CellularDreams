"""Spatial transcriptomics datasets on top of the existing simulators.

Grid layouts follow scMultiSim's placements (basic, enhanced, layers, islands).
Expression is either a small cell-type program, one of the travlr scenes, or a
Beta-Poisson matrix from the Rust simulator. Neighbor ligand exposure is added
afterwards and kept as ground truth. Visium, Slide-seq, spillover, and panel
files use the observation models already in this package.
"""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from . import __version__
from .layouts import place
from .observe import batch_effect, restrict_panel, slideseq_beads, spillover, visium_spots
from .scenes import build_scene
from .simulate import simulate

PROGRAM_GENES = ["ligand", "receptor", "target", "marker_0", "marker_1", "marker_2", "bg_0", "bg_1"]
LAYOUTS = ("basic", "enhanced", "layers", "islands")


def read_rust_counts(path: Path) -> dict:
    """Read a file written by ``scmultisim-export``."""
    blob = Path(path).read_bytes()
    magic = b"SMSCOUNTS1\n"
    if not blob.startswith(magic):
        raise ValueError(f"{path} is not an scmultisim-export file")
    cursor = len(magic)
    n_genes, n_cells, n_regions = struct.unpack_from("<III", blob, cursor)
    cursor += 12
    count_n = n_genes * n_cells
    atac_n = n_regions * n_cells
    flat = np.frombuffer(blob, dtype="<f8", count=count_n, offset=cursor).copy()
    counts = flat.reshape(n_cells, n_genes)
    cursor += count_n * 8
    atac_flat = np.frombuffer(blob, dtype="<f8", count=atac_n, offset=cursor).copy()
    atac = np.zeros((n_cells, n_regions), dtype=float)
    for cell in range(n_cells):
        start = cell * n_regions
        atac[cell] = atac_flat[start : start + n_regions]
    cursor += atac_n * 8
    depth = np.frombuffer(blob, dtype="<f8", count=n_cells, offset=cursor).copy()
    cursor += n_cells * 8
    labels = []
    for _ in range(n_cells):
        (nbytes,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        labels.append(blob[cursor : cursor + nbytes].decode())
        cursor += nbytes
    return {
        "counts": counts,
        "atac": atac,
        "depth": depth,
        "cell_pop": labels,
        "genes": [f"gene_{i}" for i in range(n_genes)],
    }


def export_rust_counts(path: Path, n_cells: int, seed: int, binary: Path | None = None) -> dict:
    """Run the Rust Beta-Poisson exporter and read it back."""
    binary = binary or _default_binary()
    if not binary.is_file():
        raise FileNotFoundError(
            f"missing {binary}. Build it with `cargo build --release --bin scmultisim-export`."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(binary), "--cells", str(n_cells), "--seed", str(seed), "--out", str(path)],
        check=True,
    )
    return read_rust_counts(path)


def neighbor_exposure(values: np.ndarray, neighbors: list[np.ndarray]) -> np.ndarray:
    """Sum ``values`` over occupied orthogonal neighbors. Self is excluded."""
    exposure = np.zeros(len(neighbors), dtype=float)
    for index, cells in enumerate(neighbors):
        if len(cells):
            exposure[index] = float(values[cells].sum())
    return exposure


def generate(
    out_dir: Path,
    *,
    layout: str = "layers",
    expression: str = "program",
    n_cells: int = 48,
    n_types: int = 3,
    seed: int = 1,
    domain_um: float = 200.0,
    cci_coef: float = 0.15,
    binary: Path | None = None,
) -> ad.AnnData:
    """Write a spatial dataset and return the single-cell AnnData.

    ``expression`` is ``program`` (cell-type markers plus a ligand), ``travlr``
    (an existing scene; ``layout`` is the scene name), or ``rust`` (Beta-Poisson
    counts from ``scmultisim-export``, then a neighbor ligand effect on top).
    """
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    if expression == "travlr":
        table = _from_travlr(layout, seed)
    elif expression == "program":
        table = _from_program(layout, n_cells, n_types, rng, domain_um, cci_coef)
    elif expression == "rust":
        table = _from_rust(out_dir / "counts.sms", layout, n_cells, n_types, seed, domain_um, cci_coef, binary)
    else:
        raise ValueError("expression must be program, travlr, or rust")
    observed = _anndata(table)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "params.json").write_text(json.dumps(_params(table, seed, cci_coef), indent=2))
    observed.write_h5ad(out_dir / "observed.h5ad")
    _write_platforms(out_dir / "platforms", table, rng)
    return observed


def _from_program(layout, n_cells, n_types, rng, domain_um, cci_coef) -> dict:
    spatial = place(layout, n_cells, n_types, rng, domain_um)
    cell_type = spatial["cell_type"]
    basal = np.zeros((n_cells, len(PROGRAM_GENES)))
    basal[:, 0] = np.where(cell_type == 0, 30.0, 0.2)
    basal[:, 1] = 5.0
    basal[:, 2] = 2.0
    for kind in range(min(n_types, 3)):
        basal[:, 3 + kind] = np.where(cell_type == kind, 25.0, 0.3)
    basal[:, 6:] = 4.0
    before = rng.poisson(basal).astype(float)
    exposure = neighbor_exposure(before[:, 0], spatial["neighbors"])
    impact = cci_coef * exposure
    counts = before.copy()
    counts[:, 2] += impact
    return _table(
        counts,
        PROGRAM_GENES,
        spatial,
        layout,
        "program",
        before=before,
        exposure=exposure,
        impact=impact,
        cell_pop=np.array([f"type_{t}" for t in cell_type]),
    )


def _from_rust(path, layout, n_cells, n_types, seed, domain_um, cci_coef, binary) -> dict:
    exported = export_rust_counts(path, n_cells, seed, binary)
    rng = np.random.default_rng(seed + 17)
    spatial = place(layout, n_cells, n_types, rng, domain_um)
    before = exported["counts"]
    ligand = before[:, 0]
    exposure = neighbor_exposure(ligand, spatial["neighbors"])
    impact = cci_coef * exposure
    counts = before.copy()
    if counts.shape[1] > 1:
        counts[:, 1] = counts[:, 1] + impact
    return _table(
        counts,
        exported["genes"],
        spatial,
        layout,
        "rust",
        before=before,
        exposure=exposure,
        impact=impact,
        cell_pop=np.array(exported["cell_pop"]),
        atac=exported["atac"],
        depth=exported["depth"],
    )


def _from_travlr(scene_name: str, seed: int) -> dict:
    scene, rng = build_scene(scene_name)
    _cells, history = simulate(scene, rng)
    positions = history.positions[-1]
    counts = history.spliced[-1].copy()
    n = len(history.cell_ids[-1])
    neighbors = [np.array([], dtype=int) for _ in range(n)]
    exposure = history.exposure[-1].sum(axis=1) if history.exposure[-1].size else np.zeros(n)
    impact = history.impact[-1][:, scene.target_gene] if history.impact[-1].size else np.zeros(n)
    spatial = {
        "positions": positions,
        "cell_type": np.zeros(n, dtype=int),
        "neighbors": neighbors,
        "edges": np.zeros((0, 2), dtype=int),
        "pitch_um": None,
        "grid_size": None,
    }
    return _table(
        counts,
        list(scene.gene_names),
        spatial,
        scene_name,
        "travlr",
        before=counts.copy(),
        exposure=exposure,
        impact=impact,
        cell_pop=np.array([scene_name] * n),
        cell_ids=history.cell_ids[-1],
    )


def _table(counts, genes, spatial, layout, expression, before, exposure, impact, cell_pop, atac=None, depth=None, cell_ids=None):
    return {
        "counts": counts,
        "genes": list(genes),
        "positions": spatial["positions"],
        "cell_type": spatial["cell_type"],
        "neighbors": spatial["neighbors"],
        "edges": spatial["edges"],
        "pitch_um": spatial["pitch_um"],
        "grid_size": spatial["grid_size"],
        "layout": layout,
        "expression": expression,
        "before": before,
        "exposure": exposure,
        "impact": impact,
        "cell_pop": cell_pop,
        "atac": atac,
        "depth": depth,
        "cell_ids": cell_ids if cell_ids is not None else np.arange(len(counts)),
    }


def _anndata(table: dict) -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "cell_id": table["cell_ids"],
            "cell_type": table["cell_type"],
            "cell_pop": table["cell_pop"],
            "ligand_exposure": table["exposure"],
            "cci_impact": table["impact"],
            "layout": table["layout"],
            "expression": table["expression"],
        }
    )
    data = ad.AnnData(X=table["counts"], obs=obs, var=pd.DataFrame(index=table["genes"]))
    data.obsm["spatial"] = np.asarray(table["positions"], dtype=float)
    data.layers["before_cci"] = table["before"]
    data.uns["neighbor_edges"] = table["edges"]
    data.uns["layout"] = table["layout"]
    data.uns["expression"] = table["expression"]
    data.uns["pitch_um"] = -1.0 if table["pitch_um"] is None else float(table["pitch_um"])
    if table["atac"] is not None:
        data.obsm["atac"] = table["atac"]
    if table["depth"] is not None:
        data.obs["cell_depth"] = table["depth"]
    return data


def _params(table: dict, seed: int, cci_coef: float) -> dict:
    return {
        "package": "travlr-sim",
        "version": __version__,
        "workflow": "spatial-transcriptomics",
        "layout": table["layout"],
        "expression": table["expression"],
        "seed": seed,
        "n_cells": int(len(table["counts"])),
        "genes": table["genes"],
        "pitch_um": table["pitch_um"],
        "grid_size": table["grid_size"],
        "cci_coef": cci_coef,
        "domain_um": 200.0,
        "platforms": {
            "visium_radius_um": 55.0,
            "visium_pitch_um": 100.0,
            "slideseq_radius_um": 10.0,
            "spillover_fraction": 0.1,
        },
    }


def _write_platforms(directory: Path, table: dict, rng: np.random.Generator) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    counts = table["counts"]
    positions = table["positions"]
    width = float(max(positions.max(axis=0).max(), 1.0))
    genes = table["genes"]
    panel_n = min(200, counts.shape[1])
    pieces = {
        "identity": (counts, positions, genes, None),
        "panel": (restrict_panel(counts, np.arange(panel_n)), positions, genes[:panel_n], None),
        "spillover": (spillover(counts, positions, 0.1), positions, genes, None),
        "batch": (batch_effect(counts, float(rng.uniform(0.8, 1.2))), positions, genes, None),
    }
    visium, centers, occupancy = visium_spots(counts, positions, width)
    beads, bead_centers, bead_occupancy = slideseq_beads(counts, positions, width)
    pieces["visium"] = (visium, centers, genes, occupancy)
    pieces["slideseq"] = (beads, bead_centers, genes, bead_occupancy)
    for name, (matrix, spatial, var_names, occupancy) in pieces.items():
        if occupancy is None:
            obs = pd.DataFrame({"cell_id": table["cell_ids"], "platform": name})
        else:
            obs = pd.DataFrame(
                {"spot_id": np.arange(len(spatial)), "n_cells": occupancy, "platform": name}
            )
        data = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=var_names))
        data.obsm["spatial"] = np.asarray(spatial, dtype=float)
        data.write_h5ad(directory / f"{name}.h5ad")


def _default_binary() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "target" / "release" / "scmultisim-export"
