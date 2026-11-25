//! Write one scMultiSim run for the spatial workflow.
//!
//! `SMSCOUNTS2` is a list of named column-major `f64` matrices, then population
//! labels, GRN edges, gene ids, regulator indexes, and target indexes.
//! A gene × cell matrix is stored as `index = gene + cell * n_genes`.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::ExitCode;

use scmultisim::{grn_params_100, phyla5, sim_true_counts, Mat, Options, Rng, SimResult, Tree};

fn arg(flag: &str, default: &str) -> String {
    let mut args = env::args().skip(1);
    while let Some(item) = args.next() {
        if item == flag {
            return args.next().unwrap_or_else(|| default.to_string());
        }
    }
    default.to_string()
}

fn flag(name: &str) -> bool {
    env::args().skip(1).any(|item| item == name)
}

fn main() -> ExitCode {
    let cells: usize = arg("--cells", "40").parse().unwrap_or(40);
    let seed: u32 = arg("--seed", "0").parse().unwrap_or(0);
    let path = arg("--out", "counts.sms");
    let velocity = flag("--velocity");
    let mut rng = Rng::seed(seed);
    let tree = phyla5(&mut rng);
    let mut options = Options::new(grn_params_100(), cells, 20, tree.clone());
    options.diff_cif_fraction = 0.8;
    options.do_velocity = velocity;
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
    if write_result(&mut out, &result, &tree).is_err() {
        eprintln!("failed while writing {path}");
        return ExitCode::from(1);
    }
    println!(
        "genes {} cells {} regions {} edges {} velocity {velocity} -> {path}",
        result.counts.nrow,
        result.counts.ncol,
        result.atac_counts.nrow,
        result.grn_edges.len()
    );
    ExitCode::SUCCESS
}

fn write_result(out: &mut impl Write, result: &SimResult, tree: &Tree) -> std::io::Result<()> {
    let mut matrices: Vec<(&str, Mat)> = vec![
        ("counts", result.counts.clone()),
        ("atac_counts", result.atac_counts.clone()),
        ("atacseq", result.atacseq_data.clone()),
        ("kon", result.kon.clone()),
        ("koff", result.koff.clone()),
        ("s", result.s.clone()),
        ("giv_kon", result.giv_kon.clone()),
        ("giv_koff", result.giv_koff.clone()),
        ("giv_s", result.giv_s.clone()),
        ("cif_kon", result.cif_kon.clone()),
        ("cif_koff", result.cif_koff.clone()),
        ("cif_s", result.cif_s.clone()),
        ("cif_atac", result.cif_atac.clone()),
        ("riv", result.riv.clone()),
        ("region_to_gene", result.region_to_gene.clone()),
        ("geff", result.geff.clone()),
        ("neutral", result.neutral.clone()),
        ("cell_depth", column(&result.cell_depth)),
    ];
    if let Some(unspliced) = &result.unspliced_counts {
        matrices.push(("unspliced", unspliced.clone()));
    }
    if let Some(velocity) = &result.velocity {
        matrices.push(("velocity", velocity.clone()));
    }
    if !result.beta.is_empty() {
        matrices.push(("beta", column(&result.beta)));
        matrices.push(("degradation", column(&result.degradation)));
    }
    let mut from = Vec::with_capacity(tree.edges.len());
    let mut to = Vec::with_capacity(tree.edges.len());
    let mut len = Vec::with_capacity(tree.edges.len());
    for edge in &tree.edges {
        from.push(edge.from as f64);
        to.push(edge.to as f64);
        len.push(edge.len);
    }
    let mut tree_data = from;
    tree_data.extend(to);
    tree_data.extend(len);
    matrices.push(("tree", Mat::from_col_major(tree.edges.len(), 3, tree_data)));

    out.write_all(b"SMSCOUNTS2\n")?;
    out.write_all(&(matrices.len() as u32).to_le_bytes())?;
    for (name, matrix) in &matrices {
        write_matrix(out, name, matrix)?;
    }
    out.write_all(&(result.cell_pop.len() as u32).to_le_bytes())?;
    for label in &result.cell_pop {
        let bytes = label.as_bytes();
        out.write_all(&(bytes.len() as u32).to_le_bytes())?;
        out.write_all(bytes)?;
    }
    out.write_all(&(result.grn_edges.len() as u32).to_le_bytes())?;
    for (target, regulator, effect) in &result.grn_edges {
        out.write_all(&(*target as u32).to_le_bytes())?;
        out.write_all(&(*regulator as u32).to_le_bytes())?;
        out.write_all(&effect.to_le_bytes())?;
    }
    out.write_all(&(result.gene_ids.len() as u32).to_le_bytes())?;
    for id in &result.gene_ids {
        out.write_all(&id.to_le_bytes())?;
    }
    write_index(out, &result.regulators)?;
    write_index(out, &result.targets)?;
    out.flush()
}

fn column(values: &[f64]) -> Mat {
    Mat::from_col_major(values.len(), 1, values.to_vec())
}

fn write_matrix(out: &mut impl Write, name: &str, matrix: &Mat) -> std::io::Result<()> {
    let bytes = name.as_bytes();
    out.write_all(&(bytes.len() as u32).to_le_bytes())?;
    out.write_all(bytes)?;
    out.write_all(&(matrix.nrow as u32).to_le_bytes())?;
    out.write_all(&(matrix.ncol as u32).to_le_bytes())?;
    for value in &matrix.data {
        out.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_index(out: &mut impl Write, values: &[usize]) -> std::io::Result<()> {
    out.write_all(&(values.len() as u32).to_le_bytes())?;
    for value in values {
        out.write_all(&(*value as u32).to_le_bytes())?;
    }
    Ok(())
}
