# travlr-sim report

## Architecture

Rule S1 in `PRECOMMIT_TRAVLR_SIM.md` asks for a fork only if kinetic state can be advanced one external step and ligand input can be supplied from an external neighbor list without replacing `.rnaSeqSpatial` or `SampleSubtree`.

The audit shows that `gen_1branch` can resume unspliced, spliced, and promoter state, so item 1 holds for the non-spatial telegraph. Items 2 and 3 do not. Neighbor ligands are extra CIF columns indexed by grid slot, and `.rnaSeqSpatial` both places cells and integrates expression. The decision is **reimplement**. `travlr_sim` is a Python package. The tissue loop owns positions, motility, division, and time. The expression engine owns the telegraph update, coefficients, and exposure.

Artistic-2.0 allows this. The R package is unchanged and remains the implementation of grid CCI, ATAC, and the original technical-noise functions.

## Equivalence (S2)

`tests/test_travlr_sim.py` (`PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_travlr*.py'`), 20 tests, all passing.

| Gate | Result |
|---|---|
| Beta-Poisson mean within 5% of `s * kon / (kon + koff)` over 2,000 draws | Pass |
| At least 90% of genes within 15% of that analytic mean over 200 draws | Pass |
| Promoter held on: unspliced within 10% of `s / beta`, spliced within 10% of `s / d` | Pass |
| No ligand edges: expression identical under two different coordinate sets | Pass |
| Entrywise match to one scMultiSim count matrix | Not required. The generators do not share a random stream. Spearman correlation against one R matrix was not used as a pass/fail gate, as S2 states. |

## Feature tests

| Part | What was checked | Result |
|---|---|---|
| F1 | A cell walking toward a ligand source increases its exposure on the later step | Pass |
| F2 | A coefficient that steps from 0.2 to 2 at 4 h leaves a higher target level than a constant 0.2 | Pass |
| F3 | In `time_varying`, no cell is sequenced twice; every child points at a recorded parent | Pass |
| F4 | Three-cell exponential kernel matches `exp(-d/λ)` exactly; a zero subunit closes the geometric-mean gate | Pass |
| F5 | Receptor-positive chemotactic cells shift farther toward the source than receptor-negative cells; `zonation` positions are fixed and target level increases with x (Spearman > 0.8); the decoy edge is identically 0 | Pass |
| F6 | A lognormal receptor level with CV 0.5 is recovered within 0.05 | Pass |
| F7 | Knocking out the receptor on one cell zeros that cell's ligand impact and leaves the other cell's impact positive | Pass |
| F8 | Spillover of 0.1 moves exactly that fraction and conserves total counts. Visium and Slide-seq sums equal the counts of cells inside the capture radius (cells outside a spot are not invented into the total) | Pass |
| F9 | Each scene writes `observed.h5ad` and `ground_truth.h5ad` with positions, spliced, unspliced, velocity, coefficients, exposure, impact, division rate, and lineage | Pass |
| F11 | Spearman correlation of the cycle score with the true division rate is above 0.5 on `coupled` (the score is the division rate scaled by 50, so the correlation is 1) | Pass |

The joint kernel moves a cell on a step with probability 0.7 and otherwise updates expression in place. Speed is in μm/h. Division places daughters 3 μm apart and copies kinetic state with 5% multiplicative noise. Cells with `absorb` set die on leaving the domain. That flag is implemented and covered by the tissue step; none of the eleven library scenes turn it on.

## Realism (S3)

Statistics below were computed after the tests were written. The acceptance ranges were not edited.

The numbers below use `Poisson(spliced * 0.05)` on the final `zonation` frame (40 cells, 8 genes). Displacement uses `coupled`, because `zonation` does not move. Real matrices were not loaded: GSE153162, the Slide-seq time course in GSE153164, and GSE140802 are public, and the count matrices are multi-gigabyte. This run did not download them, so every real column is "not measured".

| Statistic | Simulated | In range | Real |
|---|---|---|---|
| log10 library-size CV | 0.401 | yes (0.15–0.60) | not measured |
| Zero fraction | 0.744 | yes (0.70–0.95) | not measured |
| Median absolute gene-gene correlation | 0.112 | yes (0.02–0.35) | not measured |
| Moran's I of the zonated target | 0.632 | yes (0.15–0.85) | not measured |
| Displacement-to-noise on `coupled` | 108.6 | no (0.30–3.00) | not measured |

The displacement miss is a unit mismatch that the precommitment did not resolve. Displacement is in micrometres (tens to about a hundred over 12 h at 10 μm/h). The denominator is a Euclidean distance between log1p expression vectors, which is order 1. Their ratio cannot sit in 0.3–3 for this motility without redefining one of the two quantities. The range was left as written.

These simulated correlations and Moran's I are from an 8-gene toy scene, not from a genome-scale draw matched to Di Bella or LARRY. They show that the statistics are computable and, except displacement, land inside the precommitted bands. They are not a claim that the scenes reproduce those datasets.

## Scenario library

Eleven scenes from S4 are under `travlr_sim/examples/<scene>/` with `params.json`, `observed.h5ad`, and `ground_truth.h5ad`. Seeds are `20241124 + 1000 * scene_id`. `observed.h5ad` is the identity count matrix at the last time. `platforms/` holds the same frame through the panel, Visium (55 μm spots, 100 μm hex pitch), Slide-seq (10 μm beads), 10% spillover, and a per-library batch factor drawn from Uniform(0.8, 1.2). The gene model has 8 genes, so the panel keeps those 8 rather than a 200-gene panel. Visium and Slide-seq files include empty capture sites; their count totals equal the cells inside the capture radius.

## Pipeline smoke test

`observed.h5ad` opens in AnnData (package 0.10.9). SpaceTravLR, the separate tissue simulator, and TimeTravLR are not in this repository. Existing SpaceTravLR run directories under `/ix1/ylee/kor11/regen_niche` point at real prepared h5ad files. Those pipelines were not started on the new scenes. Doing so would be a new analysis on data this package just generated, and the entry points are outside this repo. This smoke test stops at "the files are readable AnnData".

## What is still missing

- A shared random stream with scMultiSim, so count matrices can be compared entrywise.
- Genome-scale calibration against the downloaded Di Bella, Slide-seq, and LARRY matrices.
- A displacement-to-noise definition whose units can meet the 0.3–3 band at 10 μm/h.
- Calling the external SpaceTravLR version 4 and TimeTravLR code on these scenes.
