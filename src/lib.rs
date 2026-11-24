//! Native Rust port of scMultiSim.
//!
//! Counts are generated from an R-compatible random stream ([`Rng`]). Seeding
//! with [`Rng::seed`]`(0)`, building a tree with [`phyla5`], then calling
//! [`sim_true_counts`] follows the same stream as `set.seed(0); Phyla5(); sim_true_counts(...)`.
//!
//! Spatial cell-cell interaction, dynamic GRNs, discrete populations, technical
//! PCR noise, and the plotting/Shiny helpers remain in the R sources.

mod matrix;
mod rng;
mod simulate;

pub use matrix::{matmul_bt, Mat};
pub use rng::Rng;
pub use simulate::{
    grn_params_100, phyla1, phyla3, phyla5, sim_true_counts, GrnEdge, Options, SimError, SimResult,
    Tree, Vary,
};
