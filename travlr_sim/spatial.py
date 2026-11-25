"""Spatial transcriptomics datasets on top of the existing simulators.

Grid layouts follow scMultiSim's placements (basic, enhanced, layers, islands).
Expression is a small cell-type program, a travlr scene, or a Rust scMultiSim
run. The Rust run keeps its own GEX, ATAC, GRN, kinetic rates, and identity
factors; the layout only adds coordinates. Visium, Slide-seq, spillover, and
panel files use the observation models already in this package.
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
    magic = b"SMSCOUNTS2\n"
    if not blob.startswith(magic):
        raise ValueError(f"{path} is not an scmultisim-export v2 file")
    cursor = len(magic)
    (n_matrices,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    matrices = {}
    for _ in range(n_matrices):
        (name_len,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        name = blob[cursor : cursor + name_len].decode()
        cursor += name_len
        nrow, ncol = struct.unpack_from("<II", blob, cursor)
        cursor += 8
        count = nrow * ncol
        flat = np.frombuffer(blob, dtype="<f8", count=count, offset=cursor).copy()
        cursor += count * 8
        matrices[name] = _from_column_major(flat, nrow, ncol)
    (n_labels,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    labels = []
    for _ in range(n_labels):
        (nbytes,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        labels.append(blob[cursor : cursor + nbytes].decode())
        cursor += nbytes
    (n_edges,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    edges = []
    for _ in range(n_edges):
        target, regulator = struct.unpack_from("<II", blob, cursor)
        cursor += 8
        (effect,) = struct.unpack_from("<d", blob, cursor)
        cursor += 8
        edges.append((target, regulator, effect))
    (n_ids,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    gene_ids = np.frombuffer(blob, dtype="<i4", count=n_ids, offset=cursor).copy()
    cursor += n_ids * 4
    regulators = _read_index(blob, cursor)
    cursor = regulators[1]
    targets = _read_index(blob, cursor)
    counts = matrices["counts"]
    return {
        "counts": counts,
        "matrices": matrices,
        "cell_pop": labels,
        "grn_edges": edges,
        "gene_ids": gene_ids,
        "regulators": regulators[0],
        "targets": targets[0],
        "genes": [f"gene_{i}" for i in range(counts.shape[1])],
    }


def _from_column_major(flat: np.ndarray, nrow: int, ncol: int) -> np.ndarray:
    """Return a C-order array of shape ``(ncol, nrow)`` when the matrix is gene × cell.

    Column ``c`` of the Rust matrix becomes row ``c`` here, so a gene × cell
    matrix becomes cells × genes and a region × cell matrix becomes cells × regions.
    A gene × factor matrix is transposed back by the caller when rows must stay genes.
    """
    if nrow == 0 or ncol == 0:
        return np.zeros((ncol, nrow))
    return flat.reshape(ncol, nrow, order="C")


def _read_index(blob: bytes, cursor: int) -> tuple[np.ndarray, int]:
    (n,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    values = np.frombuffer(blob, dtype="<u4", count=n, offset=cursor).copy()
    return values, cursor + n * 4


def export_rust_counts(
    path: Path, n_cells: int, seed: int, binary: Path | None = None, velocity: bool = False
) -> dict:
    """Run the Rust simulator and read its counts plus ground truth."""
    binary = binary or _default_binary()
    if not binary.is_file():
        raise FileNotFoundError(
            f"missing {binary}. Build it with `cargo build --release --bin scmultisim-export`."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(binary), "--cells", str(n_cells), "--seed", str(seed), "--out", str(path)]
    if velocity:
        command.append("--velocity")
    subprocess.run(command, check=True)
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
    velocity: bool = False,
) -> ad.AnnData:
    """Write a spatial dataset and return the single-cell AnnData.

    ``expression="rust"`` keeps the simulator's GEX, ATAC, GRN, kinetic rates,
    and cell factors. The layout adds coordinates and neighbor edges and does
    not change those counts. ``program`` is a small spatial ligand model.
    ``travlr`` uses an existing scene; ``layout`` is then the scene name.
    """
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    if expression == "travlr":
        table = _from_travlr(layout, seed)
    elif expression == "program":
        table = _from_program(layout, n_cells, n_types, rng, domain_um, cci_coef)
    elif expression == "rust":
        table = _from_rust(
            out_dir / "counts.sms", layout, n_cells, n_types, seed, domain_um, binary, velocity
        )
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


def _cells_by_features(matrix: np.ndarray) -> np.ndarray:
    """``_from_column_major`` already put original columns on axis 0."""
    return np.asarray(matrix, dtype=float)


def _from_rust(path, layout, n_cells, n_types, seed, domain_um, binary, velocity: bool) -> dict:
    exported = export_rust_counts(path, n_cells, seed, binary, velocity)
    rng = np.random.default_rng(seed + 17)
    spatial = place(layout, n_cells, n_types, rng, domain_um)
    matrices = exported["matrices"]
    counts = _cells_by_features(matrices["counts"])
    zeros = np.zeros(n_cells)
    table = _table(
        counts,
        exported["genes"],
        spatial,
        layout,
        "rust",
        before=counts.copy(),
        exposure=zeros,
        impact=zeros,
        cell_pop=np.array(exported["cell_pop"]),
        atac=_cells_by_features(matrices["atac_counts"]),
        depth=_cells_by_features(matrices["cell_depth"]).ravel(),
    )
    ground = {
        "kon": _cells_by_features(matrices["kon"]),
        "koff": _cells_by_features(matrices["koff"]),
        "s": _cells_by_features(matrices["s"]),
        "atac_clean": _cells_by_features(matrices["atacseq"]),
        "giv_kon": matrices["giv_kon"].T,
        "giv_koff": matrices["giv_koff"].T,
        "giv_s": matrices["giv_s"].T,
        "cif_kon": matrices["cif_kon"].T,
        "cif_koff": matrices["cif_koff"].T,
        "cif_s": matrices["cif_s"].T,
        "cif_atac": matrices["cif_atac"].T,
        "riv": matrices["riv"].T,
        "region_to_gene": matrices["region_to_gene"].T,
        "geff": matrices["geff"].T,
        "neutral": matrices["neutral"].T,
        "tree": matrices["tree"].T,
        "grn_edges": exported["grn_edges"],
        "gene_ids": exported["gene_ids"],
        "regulators": exported["regulators"],
        "targets": exported["targets"],
    }
    if "unspliced" in matrices:
        ground["unspliced"] = _cells_by_features(matrices["unspliced"])
        ground["velocity"] = _cells_by_features(matrices["velocity"])
        ground["beta"] = _cells_by_features(matrices["beta"]).ravel()
        ground["degradation"] = _cells_by_features(matrices["degradation"]).ravel()
    table["ground"] = ground
    return table


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
        "ground": None,
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
    ground = table.get("ground")
    if ground is not None:
        _attach_ground_truth(data, ground)
    return data


def _attach_ground_truth(data: ad.AnnData, ground: dict) -> None:
    """Store the mechanistic quantities that produced ``X``."""
    for key in ("kon", "koff", "s", "unspliced", "velocity"):
        if key in ground:
            data.layers[key] = ground[key]
    data.obsm["atac_clean"] = ground["atac_clean"]
    for key in ("cif_kon", "cif_koff", "cif_s", "cif_atac"):
        data.obsm[key] = ground[key]
    for key in ("giv_kon", "giv_koff", "giv_s"):
        data.varm[key] = ground[key]
    data.uns["region_to_gene"] = ground["region_to_gene"]
    data.uns["riv"] = ground["riv"]
    data.uns["geff"] = ground["geff"]
    data.uns["neutral"] = ground["neutral"]
    data.uns["tree"] = ground["tree"]
    edges = ground["grn_edges"]
    data.uns["grn_target"] = np.array([edge[0] for edge in edges], dtype=np.int32)
    data.uns["grn_regulator"] = np.array([edge[1] for edge in edges], dtype=np.int32)
    data.uns["grn_effect"] = np.array([edge[2] for edge in edges], dtype=float)
    data.uns["regulators"] = np.asarray(ground["regulators"], dtype=np.int32)
    data.uns["targets"] = np.asarray(ground["targets"], dtype=np.int32)
    data.var["gene_id"] = np.asarray(ground["gene_ids"])
    if "beta" in ground:
        data.var["beta"] = ground["beta"]
        data.var["degradation"] = ground["degradation"]
    data.uns["ground_truth"] = np.array(
        [
            "X is simulated GEX",
            "layers kon, koff, s are the kinetic rates",
            "obsm atac is chromatin accessibility after technical noise",
            "obsm atac_clean is accessibility before that noise",
            "uns grn_* is the gene regulatory network",
            "varm giv_* and obsm cif_* are the identity factors",
            "uns region_to_gene maps regions to genes",
            "obsm spatial is the layout and does not alter X",
        ]
    )


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
