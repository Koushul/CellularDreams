# Precommitments for travlr-sim

These rules were written before the simulator implementation. Later parts follow them. They are not results.

## S1. Architecture decision rule

Choose **fork** only when every item below is true:

1. One cell's kinetic state (promoter state, unspliced level, spliced level) can be advanced by one caller-chosen time step through the existing `gen_1branch` path, passing only the previous state and the current kinetic rates.
2. Ligand input for that step can be supplied as a numeric vector from an external neighbor list, without `CreateSpatialGrid` allocating positions.
3. Meeting (1) and (2) does not require replacing the control flow of `.rnaSeqSpatial` or `SampleSubtree`.

If any item is false, choose **reimplement**: a Python kinetic engine that uses the formulas in `SCMULTISIM_AUDIT.md`, with the tissue simulator owning positions, division, and time. Equivalence is then the tests in S2, not a line-by-line port of the grid loop.

The tissue simulator owns space and time. The expression engine owns kinetics, regulation, and interaction effects. That split does not depend on which architecture wins.

## S2. Equivalence tests against scMultiSim

New features (movement, user-specified trajectories, destructive sampling, exponential exposure, perturbations, platform models) are off.

1. **Beta-Poisson mean.** For fixed `kon`, `koff`, `s` and `intrinsic.noise = 1`, the mean of 2,000 draws is within 5% of `s * kon / (kon + koff)`.
2. **Static gene means.** Given one scMultiSim `kon`, `koff`, and `s` matrix, the Python engine's per-gene mean counts, over at least 200 cells, have relative error below 15% against `s * kon / (kon + koff)` on at least 90% of genes. This compares both engines to the same analytic mean, which is what scMultiSim's Beta-Poisson targets.
3. **Velocity equilibrium.** With the promoter held on, splicing `beta` and degradation `d`, after a long deterministic run the spliced level is within 10% of `s / d` and the unspliced level is within 10% of `s / beta`.
4. **Interaction off.** With an empty ligand table, expression does not depend on positions. Two runs that differ only in coordinates match exactly.
5. **Distribution, not bits.** A full scMultiSim count matrix and a travlr-sim matrix are not required to be equal entrywise. Gene-mean Spearman correlation against a saved scMultiSim summary is reported and is not a pass/fail gate, because the generators do not share one random stream. The pass/fail gates are (1)–(4).

## S3. Real datasets and acceptance ranges

No new experiment. Calibration uses only public data. Statistics are computed on the simulated scene `zonation` after the 10x-like observation model, and on the real objects if a local download succeeds. If a download fails, the real column is reported as not measured. Ranges were fixed before looking at travlr-sim output.

| Statistic | Dataset | Acceptance range | Source of the range |
|---|---|---|---|
| Cells per time point (context, not a sim size target) | Di Bella et al. 2021, GEO GSE153162 inside GSE153164 | 4,000–10,000 in the real series | GEO summary for GSE153162 |
| Sequencing depth per cell | same paper, methods | 40,000–70,000 reads/cell in the real libraries | Di Bella et al., Nature 2021 |
| Coefficient of variation of log10 library size | simulated 10x-like sample; shape compared with the Di Bella per-cell count spread | 0.15–0.60 | Precommitted shape band for a 10x library, not a fitted constant |
| Fraction of zero entries after depth sampling | simulated panel of 2,000 genes | 0.70–0.95 | Typical sparse 10x range, precommitted |
| Median absolute Pearson correlation among the 50 most variable genes, within one cell type | simulated; Di Bella excitatory lineage if the matrix is available | 0.02–0.35 | Precommitted. The real value is reported beside it |
| Moran's I of one zonated gene | simulated coordinates; Slide-seq E15 cortex (Stickels / Di Bella spatial time course, public with GSE153164) if the puck is available | 0.15–0.85 | Precommitted. The real Moran's I is reported beside it |
| Displacement-to-noise ratio | LARRY in vitro, GEO GSE140802 (Weinreb et al., Science 2020; days 2, 4, 6). Ratio is median displacement in the simulator divided by the median within-type expression distance. Real clonal data are used only to confirm that sisters at later days exist, not to copy their coordinates | 0.30–3.00 for the simulated ratio | Precommitted band. The plan's contrast for cell cycle is separate (S2 does not cover it; the cell-cycle test is in the implementation tests) |

Cell-cycle acceptance, also fixed here: on the division scene, Spearman correlation between the proliferation-program score and the true division rate must be at least 0.5. The contrast cited in the plan is a correlation of −0.018 from a generator that does not couple the program to division. That number is the failure mode, not a target.

## S4. Scenario library

Master seed `20241124`. Scene seed is `20241124 + 1000 * scene_id`. Domain is a 200 μm square. Step is 0.5 h. Duration is 12 h (24 steps) unless the scene says otherwise. Initial population is 40 cells. Persistent speed is 10 μm/h. Baseline division probability is 0.02 per step. Contact length scale is 10 μm. Secreted length scale is 40 μm. One ligand, one receptor, one target, unless the scene adds features.

| scene_id | Name | Mechanism label that must be exported |
|---|---|---|
| 1 | coupled | Cells change expression state in proportion to distance traveled |
| 2 | fork | Two initial groups, two destinations |
| 3 | crossing | Two groups exchange sides |
| 4 | radial | Outward movement from the center |
| 5 | rim | Movement confined to the boundary |
| 6 | zonation | Positions fixed; expression depends on the x coordinate through a local signal |
| 7 | signal_in_place | Positions fixed; a ligand turns on at 6 h |
| 8 | decoy | A second ligand is spatially correlated with the true ligand and has effect 0 |
| 9 | il21_like | Receptor-positive cells respond; receptor-negative cells do not; one null ligand and one decoy ligand are present |
| 10 | time_varying | A TF–target coefficient doubles at 6 h; 30% of living cells are sequenced at 4 h, 8 h, and 12 h and removed |
| 11 | chemotaxis | Receptor-positive cells bias their heading up the ligand gradient; receptor-negative cells do not |

Each scene is also passed through five observation models: identity, a 200-gene panel, Visium (55 μm spots, 100 μm hexagonal pitch), Slide-seq (10 μm beads), and segmentation spillover of 10% plus a per-time-point batch factor.

## S5. Seeds and output formats

- Master seed `20241124` (integer). Child streams are NumPy `Generator(PCG64(scene_seed))`.
- One directory per scene: `travlr_sim/examples/<scene>/`.
- `params.json` records package version, scene id, seed, step, duration, and every coefficient trajectory.
- `observed.h5ad` is an AnnData object. `X` is the sampled count matrix. `obs` has cell id, time, sample id, and platform. `obsm['spatial']` is positions in micrometres at sampling.
- `ground_truth.h5ad` holds the full simulated history in `uns`: positions, spliced, unspliced, velocity, coefficients, exposure, per-ligand impact, lineage (parent, birth, death), growth state, and scenario labels. Arrays are documented in `travlr_sim` docstrings.
- Regenerating a scene with the same seed and the same code version reproduces the files byte for byte for the numeric arrays.
