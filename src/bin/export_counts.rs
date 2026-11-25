//! Write one Beta-Poisson simulation for the spatial workflow.
//!
//! Little-endian layout:
//! magic `SMSCOUNTS1\n`, then `u32` gene count, cell count, region count,
//! column-major `f64` counts (gene + cell * n_genes), column-major ATAC,
//! `f64` cell depth, then each population label as `u32` byte length plus UTF-8.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::ExitCode;

use scmultisim::{grn_params_100, phyla5, sim_true_counts, Options, Rng};

fn arg(flag: &str, default: &str) -> String {
    let mut args = env::args().skip(1);
    while let Some(item) = args.next() {
        if item == flag {
            return args.next().unwrap_or_else(|| default.to_string());
        }
    }
    default.to_string()
}

fn main() -> ExitCode {
    let cells: usize = arg("--cells", "40").parse().unwrap_or(40);
    let seed: u32 = arg("--seed", "0").parse().unwrap_or(0);
    let path = arg("--out", "counts.sms");
    let mut rng = Rng::seed(seed);
    let tree = phyla5(&mut rng);
    let mut options = Options::new(grn_params_100(), cells, 20, tree);
    options.diff_cif_fraction = 0.8;
    options.do_velocity = false;
    let result = match sim_true_counts(&options, &mut rng) {
        Ok(result) => result,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let file = match File::create(&path) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("cannot write {path}: {err}");
            return ExitCode::from(1);
        }
    };
    let mut out = BufWriter::new(file);
    if write_result(&mut out, &result).is_err() {
        eprintln!("failed while writing {path}");
        return ExitCode::from(1);
    }
    println!(
        "genes {} cells {} regions {} -> {path}",
        result.counts.nrow, result.counts.ncol, result.atac_counts.nrow
    );
    ExitCode::SUCCESS
}

fn write_result(out: &mut impl Write, result: &scmultisim::SimResult) -> std::io::Result<()> {
    out.write_all(b"SMSCOUNTS1\n")?;
    out.write_all(&(result.counts.nrow as u32).to_le_bytes())?;
    out.write_all(&(result.counts.ncol as u32).to_le_bytes())?;
    out.write_all(&(result.atac_counts.nrow as u32).to_le_bytes())?;
    for value in &result.counts.data {
        out.write_all(&value.to_le_bytes())?;
    }
    for value in &result.atac_counts.data {
        out.write_all(&value.to_le_bytes())?;
    }
    for value in &result.cell_depth {
        out.write_all(&value.to_le_bytes())?;
    }
    for label in &result.cell_pop {
        let bytes = label.as_bytes();
        out.write_all(&(bytes.len() as u32).to_le_bytes())?;
        out.write_all(bytes)?;
    }
    out.flush()
}
