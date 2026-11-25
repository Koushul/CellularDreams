# travlr-sim

Python simulator for the tissue and expression tests that scMultiSim does not cover. The tissue loop owns positions, motility, division, and time. The expression engine owns the telegraph kinetics, time-varying coefficients, and distance-weighted ligand exposure.

The architecture rule and the numeric gates are in `PRECOMMIT_TRAVLR_SIM.md`. The scMultiSim reading that forced a reimplementation is in `SCMULTISIM_AUDIT.md`.

```python
from travlr_sim.scenes import build_scene
from travlr_sim.simulate import simulate

scene, rng = build_scene("zonation")
cells, history = simulate(scene, rng)
```

`travlr_sim.io.write_scene` writes `params.json`, an AnnData observation file, and a ground-truth AnnData file whose `uns` holds positions, spliced and unspliced levels, velocity, coefficients, exposure, impact, division rate, and lineage.

## Spatial transcriptomics workflow

`python -m travlr_sim` places cells and writes a spatial dataset under `--out`:

- `--layout` is `basic`, `enhanced`, `layers`, or `islands` (grid neighborhoods, as in the R simulator), or a travlr scene name when `--expression travlr`.
- `--expression program` draws cell-type markers and adds a neighbor ligand effect on `target`.
- `--expression rust` runs `scmultisim-export`. `X` is that simulator's gene expression and is not edited after placement. The file also holds ATAC (`obsm['atac']` and `obsm['atac_clean']`), the GRN (`uns['grn_target']`, `uns['grn_regulator']`, `uns['grn_effect']`), kinetic rates (`layers['kon']`, `layers['koff']`, `layers['s']`), gene and cell identity factors, the region-to-gene map, and the differentiation tree. `--velocity` adds unspliced counts and velocity. Build the exporter with `cargo build --release --bin scmultisim-export`.
- `--expression travlr` uses an existing scene, including its positions and kinetic ligand effect.

Each run writes `params.json`, `observed.h5ad` (coordinates in `obsm['spatial']`, counts before the neighbor effect in `layers['before_cci']`), and `platforms/` for identity, panel, Visium, Slide-seq, spillover, and batch.
