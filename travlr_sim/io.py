"""AnnData export of observations and ground truth."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from . import __version__
from .observe import batch_effect, restrict_panel, slideseq_beads, spillover, visium_spots
from .simulate import History, Scene


def write_scene(directory: Path, scene: Scene, history: History, rng: np.random.Generator) -> None:
    """Write ``params.json``, ``observed.h5ad``, and ``ground_truth.h5ad``."""
    directory.mkdir(parents=True, exist_ok=True)
    params = {
        "package": "travlr-sim",
        "version": __version__,
        "scene": scene.name,
        "scene_id": scene.scene_id,
        "seed": 20241124 + 1000 * scene.scene_id,
        "dt_hours": scene.dt,
        "duration_hours": scene.duration,
        "genes": scene.gene_names,
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "kind": edge.kind,
                "length_scale_um": edge.length_scale,
                "knots": edge.knots,
            }
            for edge in scene.edges
        ],
    }
    (directory / "params.json").write_text(json.dumps(params, indent=2))
    observed = _observed(scene, history, rng)
    observed.write_h5ad(directory / "observed.h5ad")
    _write_platforms(directory / "platforms", scene, history, observed, rng)
    truth = _truth(scene, history)
    truth.write_h5ad(directory / "ground_truth.h5ad")


def _final_frame(history: History):
    return (
        history.cell_ids[-1],
        history.positions[-1],
        history.spliced[-1],
        history.time[-1],
    )


def _observed(scene: Scene, history: History, rng: np.random.Generator) -> ad.AnnData:
    ids, positions, spliced, time = _final_frame(history)
    counts = rng.poisson(np.clip(spliced, 0, None) * 0.25)
    obs = pd.DataFrame(
        {
            "cell_id": ids,
            "time": time,
            "sample_id": scene.name,
            "platform": "identity",
        }
    )
    data = ad.AnnData(X=counts.astype(float), obs=obs, var=pd.DataFrame(index=scene.gene_names))
    data.obsm["spatial"] = positions
    data.uns["platforms"] = {
        "panel_genes": scene.gene_names[: min(4, len(scene.gene_names))],
        "visium_radius_um": 55.0,
        "visium_pitch_um": 100.0,
        "slideseq_radius_um": 10.0,
        "spillover_fraction": 0.1,
    }
    return data


def _write_platforms(
    directory: Path, scene: Scene, history: History, observed: ad.AnnData, rng: np.random.Generator
) -> None:
    """Write one AnnData file per observation model, using the stored identity counts."""
    directory.mkdir(parents=True, exist_ok=True)
    ids, positions, _, time = _final_frame(history)
    counts = np.asarray(observed.X, dtype=float)
    rendered = apply_platforms(counts, positions, scene.width, scene.gene_names, rng)
    panel_genes = rendered.pop("panel_genes")
    for name, matrix in rendered.items():
        if name == "panel":
            var_names = panel_genes
            spatial = positions
            obs = pd.DataFrame({"cell_id": ids, "time": time, "platform": name})
        elif name in {"visium", "slideseq"}:
            if name == "visium":
                matrix, centers, occupancy = visium_spots(counts, positions, scene.width)
            else:
                matrix, centers, occupancy = slideseq_beads(counts, positions, scene.width)
            var_names = scene.gene_names
            spatial = centers
            obs = pd.DataFrame({"spot_id": np.arange(len(centers)), "time": time, "n_cells": occupancy, "platform": name})
        else:
            var_names = scene.gene_names
            spatial = positions
            obs = pd.DataFrame({"cell_id": ids, "time": time, "platform": name})
        data = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=var_names))
        data.obsm["spatial"] = spatial
        data.write_h5ad(directory / f"{name}.h5ad")


def apply_platforms(counts: np.ndarray, positions: np.ndarray, width: float, genes: list[str], rng: np.random.Generator):
    """Return the five observation matrices named in S4."""
    panel_index = np.arange(min(200, counts.shape[1]))
    spilled = spillover(counts, positions, 0.1)
    visium, _, _ = visium_spots(counts, positions, width)
    beads, _, _ = slideseq_beads(counts, positions, width)
    batched = batch_effect(counts, float(rng.uniform(0.8, 1.2)))
    return {
        "identity": counts,
        "panel": restrict_panel(counts, panel_index),
        "visium": visium,
        "slideseq": beads,
        "spillover_batch": batch_effect(spilled, 1.0),
        "batch": batched,
        "panel_genes": [genes[i] for i in panel_index],
    }


def _pad(frames: list[np.ndarray], fill: float) -> np.ndarray:
    width = max(frame.shape[0] for frame in frames)
    padded = []
    for frame in frames:
        if frame.shape[0] == width:
            padded.append(frame)
            continue
        extra_shape = (width - frame.shape[0],) + frame.shape[1:]
        padded.append(np.concatenate([frame, np.full(extra_shape, fill)], axis=0))
    return np.stack(padded)


def _truth(scene: Scene, history: History) -> ad.AnnData:
    """Pack the history into ``uns``.

    Per-step arrays are padded with NaN (or -1 for ids) so a step where cells
    divided still fits in one tensor. ``lineage_*`` columns describe every cell
    that ever lived. ``lineage_samples`` is a comma-separated list of sample times.
    """
    data = ad.AnnData(X=history.spliced[-1])
    data.uns["time"] = np.asarray(history.time)
    data.uns["cell_ids"] = _pad(history.cell_ids, -1)
    data.uns["positions"] = _pad(history.positions, np.nan)
    data.uns["spliced"] = _pad(history.spliced, np.nan)
    data.uns["unspliced"] = _pad(history.unspliced, np.nan)
    data.uns["velocity"] = _pad(history.velocity, np.nan)
    data.uns["exposure"] = _pad(history.exposure, np.nan)
    data.uns["impact"] = _pad(history.impact, np.nan)
    data.uns["coefficients"] = np.stack(history.coefficients)
    data.uns["division_rate"] = _pad([rate[:, None] for rate in history.division_rate], np.nan)[:, :, 0]
    data.uns["cycle_score"] = _pad([score[:, None] for score in history.cycle_score], np.nan)[:, :, 0]
    data.uns["lineage_id"] = np.array([cell.id for cell in scene.cells], dtype=int)
    data.uns["lineage_parent"] = np.array(
        [-1 if cell.parent is None else cell.parent for cell in scene.cells], dtype=int
    )
    data.uns["lineage_birth"] = np.array([cell.birth for cell in scene.cells])
    data.uns["lineage_death"] = np.array(
        [-1.0 if cell.death is None else cell.death for cell in scene.cells]
    )
    data.uns["lineage_samples"] = np.array(
        [",".join(str(t) for t in cell.samples) for cell in scene.cells]
    )
    data.uns["genes"] = np.array(scene.gene_names)
    data.uns["scene"] = scene.name
    data.uns["labels"] = {
        "migrates": scene.name in {"coupled", "fork", "crossing", "radial", "rim", "chemotaxis", "time_varying", "decoy", "il21_like"},
        "stationary": scene.stationary,
        "zonation": scene.zonation,
        "decoy_effect": 0.0,
        "null_effect": 0.0,
    }
    return data
