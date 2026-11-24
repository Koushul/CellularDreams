# scMultiSim

**Table of contents**

- [Tutorials and documentation](#tutorials-and-documentation)
- [Installation](#installation)
- [Shiny App](#shiny-app)
- [FAQ](#faq)
- [Contact and reference](#contact)

scMultiSim is an in silico simulator that generates multi-modality data of single-cells, including gene expression, chromatin accessibility, RNA velocity, and spatial location of cells. It takes a cell differential tree and a gene regulatory network (GRN) as input, and simulates spliced and unspliced counts while accounting for the relationships between modalities. The output single cell gene expression data is determined by three factors: cell-cell interactions, within-cell GRNs and chromatin accessibility. Users can tune the effect of each factor on the output data and set various parameters for the underlying model. Furthermore, the GRN can be set in a time-varying mode where the network's structure changes temporally to reflect the dynamic nature of biological networks. We also provide options to simulate technical variations such as batch effects. scMultiSim can be used to benchmark challenging computational tasks on single-cell multi-omics data, including the inference of GRNs, estimation of RNA velocity, integration of single-cell datasets from multiple batches and modalities, and analysis of cell-cell interaction using the cell spatial location data.

![Overview](https://github.com/ZhangLabGT/scMultiSim/raw/img/img/scMultisim.png)

The following figure briefly shows results from the same cell differential tree:

1. Connected scATAC-seq and scRNA-seq, in continuous or discrete mode. Visualized by t-SNE.
2. GRN correlation heatmap, where genes regulated by the same regulator have similar correlations with others.
3. Unspliced counts and RNA velocity ground truth visualized by t-SNE.
4. Spatial cell locations and cell-cell interaction ground truth.
5. Discrete cell population with added batch effects.

![Results](https://github.com/ZhangLabGT/scMultiSim/raw/img/img/results.png)

## Native Rust simulator

The computational core is also implemented as a native Rust crate (`scmultisim`, no R runtime). It follows R 4.5's default generator (Mersenne-Twister, inversion normals, rejection sampling) and R's long-double `cumsum` / `mean`. A seeded Beta-Poisson simulation matches `set.seed(0); Phyla5(); sim_true_counts(...)`. Kinetic mode gives each gene its own generator, seeded from the master stream, and integrates those genes in parallel. The telegraph and the `which.min` sampling rule match R; the promoter-switch uniforms do not, so spliced totals are reproducible for a seed and correlate with R gene by gene, without being the same matrix. OpenBLAS `dgemm` stays on one thread because a reassociated product can change density ranks and then every later draw.

```rust
let mut rng = scmultisim::Rng::seed(0);
let tree = scmultisim::phyla5(&mut rng);
let opt = scmultisim::Options::new(scmultisim::grn_params_100(), 100, 20, tree);
let result = scmultisim::sim_true_counts(&opt, &mut rng).unwrap();
```

`cargo test` checks the generator, OpenBLAS `dgemm`, and a 110-gene × 100-cell Beta-Poisson run (counts, ATAC, and the neutral trajectory) against fixtures exported from the R sources.

Still R-only: discrete populations, spatial cell-cell interaction, dynamic GRNs, PCR / batch technical noise, and the plotting and Shiny helpers.

## Tutorials and documentation

Please check out the [tutorials](https://zhanglabgt.github.io/scMultiSim/articles)
for detailed instructions on how to use scMultiSim.

## Installation

scMultiSim can be installed from BioConductor using the following command:

```R
if (!require("BiocManager")) {
  install.packages("BiocManager")
}

BiocManager::install("scMultiSim")
```

## Shiny App

A Shiny app is provided to help users visualize the effect of each parameter and adjust the simulation options.
To run the app, simply call `run_shiny()`.

<img src="https://github.com/ZhangLabGT/scMultiSim/raw/img/img/shiny_app_sc.png" height="400">

## FAQ

### Running Speed

Simulations should finish in a reasonable time in most cases. On a machine with an i7-12700K CPU and 64GB RAM, using 1000 cells, 100 genes and 50 CIFs, the simulation took under 1 mimute to generate both scRNA-seq and scATAC-seq data. If also generating unspliced and spliced counts, or enabling cell-cell interactions, the running time is longer (~3 minutes when RNA velocity is enabled, and 30 minutes for 500 cells with spatial cell-cell interaction enabled).

## Contact and reference

GitHub issues are welcomed.
It is also possible to send email to the main author
`Hechen Li (hli691 at gatech.edu)`.

### Please cite

Hechen Li, Ziqi Zhang, Michael Squires, Xi Chen, and Xiuwei Zhang. 2023. “scMultiSim: Simulation of Multi-Modality Single Cell Data Guided by Cell-Cell Interactions and Gene Regulatory Networks.” bioRxiv.
