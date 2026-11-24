# scMultiSim audit (package 1.1.10)

Audit of the sources in this repository (`DESCRIPTION` version 1.1.10, license Artistic-2.0). The four vignettes were read in full as specifications (`vignettes/workflow.Rmd`, `vignettes/basics.Rmd`, `vignettes/spatialCCI.Rmd`, `vignettes/options.Rmd`). They were not knitted to HTML in this run: rendering needs BiocStyle, a full package install, and the plotting stack, and the numerical claims below come from the R sources those vignettes call. A separate seeded run of `sim_true_counts` (100 cells, 20 CIFs, Phyla5, GRN_params_100) was executed while porting the continuous model and is the numerical reference for the kinetic and Beta-Poisson paths.

## License

Artistic-2.0. Modification and redistribution are allowed. A derivative must not be presented as the original package, and changes to a distributed copy of the R sources must be described. A separate Python engine that cites the model, rather than copying the R sources, is compatible with the license. Forking the R package is also allowed.

## Kinetic parameterization

True expression is driven by three kinetic parameters per gene per cell: `kon`, `koff`, and `s` (transcription rate while the promoter is on). With `do.velocity = FALSE`, each count is one Beta-Poisson draw (`.betaPoisson` in `R/2_sim.R`):

- promoter activity `y ~ Beta(kon, koff)`
- unspliced-like count `x ~ Poisson(y * s)`
- reported count `intrinsic.noise * x + (1 - intrinsic.noise) * (kon / (kon + koff)) * s`

With `do.velocity = TRUE`, `gen_1branch` integrates a two-state telegraph model. Unspliced `x` and spliced `y` step at `stepsize = cycle_length / npart`, with `cycle_length = 1/kon + 1/koff`. The promoter flips with probabilities derived from `npart`. Splicing rate `beta` (default 0.4) and degradation `d` (default 1) are drawn once per gene as `rnorm(mean, 0.1)`. The returned counts are `as.integer` of the trajectory point closest to a uniform random time inside the last cycle. The function accepts `start_u`, `start_s`, and `start_state`, so a later call can resume a cell. The step is in cycle units, not hours, and the number of cycles is an internal option (`num.cycles`, 15 on the first cell of a root edge, otherwise 3).

Where the parameters come from:

- Cell identity factors (CIFs) are continuous Brownian paths on the population tree (`SampleSubtree` / `SampleEdge` in `R/1.1_cif.R`), plus non-differentiating Gaussian columns and, for regulators, extra Gaussian columns. Discrete mode uses `vcv.phylo` and multivariate normals instead.
- Gene identity vectors (GIVs) are sparse Gaussian matrices (`giv.prob`, `giv.mean`, `giv.sd`). Regulator rows of the `s` GIV are rewritten from the GRN: two random differentiating CIFs get effect 2, targets inherit a scaled combination of their regulators' rows, and entries with absolute value below 0.2 are zeroed.
- `params = 10^(matched rank of CIF %*% t(GIV))`, after an optional ATAC adjustment of `kon` and a bimodality shift.
- Chromatin: region identity vectors, a region-to-gene map (`region.distrib`), and a density match that zeros the lowest `atac.p_zero` fraction (default 0.8). Intrinsic ATAC noise is `rnorm(0, mean(atac) * 1.5)` clipped at 0.

## Cell-cell interactions and neighbors

Enabled only when `cci` is a list (`R/3.1_spatial.R`, `.rnaSeqSpatial` in `R/2_sim.R`). Ligand–target edges are a second GRN-like table. Each cell has at most `max.neighbors` interacting neighbors (1 to 4). Neighbors are cells within `radius` on the integer grid, or, if `radius` is `gaussian:sigma`, a distance-weighted sample. The chosen neighbors are not a continuous kernel: each neighbor slot is concatenated into the CIF as extra regulator columns, and the effect matrix is `num_genes` by `n_ligands * max.neighbors`. Cell-type factors (`cell.type.interaction`) multiply ligand effects. One ligand–receptor pair can be dropped at random (`del.lr.pair`, default true). There is no micrometre length scale, no exponential kernel, no receptor subunit, and no rule that several ligands compete for one receptor.

## Spatial layout and placement order

`CreateSpatialGrid` fills a square lattice. Layouts are `enhanced`, `basic`, `layers`, `islands:type,...`, or a user function returning an `n_cell × 2` coordinate matrix in grid units. Continuous populations are cut into pseudo-types along the tree by `step.size`, then placed by the layout. The simulation walks layers. `start.layer = 1` begins with one cell and adds one cell per layer. `start.layer = num.cells` places everyone and then runs a few static layers. Cells do not move after placement. There is no physical speed, no division, and no death.

## Time-varying networks

`dynamic.GRN` (`R/3.2_dyngrn.R`) is the only time-varying option. Each epoch deletes and creates a random set of edges (`num.changing.edges`), draws new weights from a normal, and linearly interpolates those weights over `num.steps`, with `cell.per.step` cells sharing each intermediate matrix. The user does not supply a trajectory. Ligand–receptor effects are not on this schedule. The mechanism is a random rewrite plus a straight ramp, not a piecewise or spline function of simulated time.

## Technical noise

`add_expr_noise` (`R/6_technoise.R`) is a separate call after `sim_true_counts`. It subsamples transcripts (alpha), applies gene-length fragment rates, two PCR stages (`nPCR1`, `nPCR2`), and a sequencing depth (`depth_mean`, `depth_sd`). ATAC observation is a detection probability plus Gaussian noise (`atac.obs.prob`, `atac.sd.frac`). `divide_batches` multiplies genes by batch factors of a chosen effect size. `intrinsic.noise` is the Beta-Poisson weight above, not platform noise. Nothing in this stack is a Visium spot, a Slide-seq bead, a gene panel, or segmentation spillover.

## Feature matrix

| ID | Feature | Status | Evidence |
|---|---|---|---|
| F1 | Movement and division in continuous space | Missing | Grid placement only (`CreateSpatialGrid`). No speed, no post-placement move, no division. |
| F2 | Time-varying regulation | Partial | `dynamic.GRN` ramps a random edge rewrite. No user trajectory, and ligand effects are excluded. |
| F3 | Destructive sampling with lineage | Missing | One cell set is sequenced. The tree is a population phylogeny (`Phyla5`), not parent–child lineage. |
| F4 | SpaceTravLR-style exposure | Partial | Neighbor ligands exist, including a Gaussian radius. No exponential μm kernel, no subunit gate, no shared-receptor aggregation. |
| F5 | Chemotaxis, zonation, decoys | Missing | Layouts can cluster types (`islands`, `layers`). Nothing moves up a gradient, and nothing is labeled null or decoy. |
| F6 | Controlled receptor variation | Missing | No within-type coefficient of variation for a receptor. |
| F7 | Mid-run perturbations | Missing | Parameters are fixed before `.rnaSeq` / `.rnaSeqSpatial`. |
| F8 | Platform observation models | Partial | Capture, PCR, depth, and library batches only. |
| F9 | Ground truth over time | Partial | Counts, velocity, CIF, GIV, grid, and CCI parameters for the final snapshot. No per-step positions, exposures, impacts, or growth. |
| F10 | Realism calibration | Missing | No comparison to an external dataset. |
| F11 | Cell-cycle genes tied to division | Missing | Division is not simulated, so no proliferation program can set a division rate. |
| F12 | Python interoperability | Missing | The distributed simulator is R. Artistic-2.0 allows a separate engine. |

## What can be reused without a core rewrite

- The Beta-Poisson and telegraph formulas, including the resume arguments of `gen_1branch`.
- The CIF / GIV / GRN construction for a static snapshot.
- The idea of ligand genes as an additive input to the transcription rate.
- `add_expr_noise` and `divide_batches` as a description of capture and batch noise, not as the platform models in F8.

What cannot be reused as the tissue loop: neighbor identity is an index into a fixed grid CIF, positions are owned by `CreateSpatialGrid`, and one call of the spatial simulator both places cells and integrates expression. Passing external positions each hour would mean replacing `.rnaSeqSpatial`, not calling it.
