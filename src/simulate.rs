//! Continuous (and kinetic) scMultiSim model.
//!
//! Random draws follow the live R implementation: `rand.seed` is not reapplied
//! inside the run, and `Phyla5()` / `Phyla3()` consume the same uniforms as
//! `ape::rtree(2)` before the topology (which is deterministic) is returned.

use crate::matrix::{blas_single_thread, matmul, matmul_bt, matmul_bt_into, Mat};
use crate::rng::{r_index, rank_average, seq_by, Rng};

#[derive(Clone, Debug)]
pub struct GrnEdge {
    pub target: i32,
    pub regulator: i32,
    pub effect: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vary {
    All,
    Kon,
    Koff,
    S,
    ExceptKon,
    ExceptKoff,
    ExceptS,
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub from: i32,
    pub to: i32,
    pub len: f64,
}

#[derive(Clone, Debug)]
pub struct Tree {
    pub n_tip: usize,
    pub edges: Vec<Edge>,
}

impl Tree {
    pub fn root(&self) -> i32 {
        self.edges
            .iter()
            .map(|e| e.from)
            .find(|p| self.edges.iter().all(|e| e.to != *p))
            .expect("tree has a root")
    }

    fn parent(&self, node: i32) -> Option<i32> {
        self.edges.iter().find(|e| e.to == node).map(|e| e.from)
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub grn: Vec<GrnEdge>,
    pub num_cells: usize,
    pub num_cifs: usize,
    pub tree: Tree,
    pub diff_cif_fraction: f64,
    pub do_velocity: bool,
    pub vary: Vary,
    pub intrinsic_noise: f64,
    pub bimod: f64,
    pub atac_effect: f64,
    pub atac_p_zero: f64,
    pub region_distrib: [f64; 3],
    pub giv_prob: f64,
    pub giv_mean: f64,
    pub giv_sd: f64,
    pub riv_prob: f64,
    pub riv_mean: f64,
    pub riv_sd: f64,
    pub cif_center: f64,
    pub cif_sigma: f64,
    pub grn_effect: f64,
    pub unregulated_gene_ratio: f64,
    pub beta: f64,
    pub degradation: f64,
    pub num_cycles: usize,
    pub cycle_len: f64,
    pub scale_s: f64,
    pub discrete_cif: bool,
}

impl Options {
    pub fn new(grn: Vec<GrnEdge>, num_cells: usize, num_cifs: usize, tree: Tree) -> Self {
        Self {
            grn,
            num_cells,
            num_cifs,
            tree,
            diff_cif_fraction: 0.9,
            do_velocity: false,
            vary: Vary::S,
            intrinsic_noise: 1.0,
            bimod: 0.0,
            atac_effect: 0.5,
            atac_p_zero: 0.8,
            region_distrib: [0.1, 0.5, 0.4],
            giv_prob: 0.3,
            giv_mean: 0.0,
            giv_sd: 1.0,
            riv_prob: 0.3,
            riv_mean: 0.0,
            riv_sd: 1.0,
            cif_center: 1.0,
            cif_sigma: 0.1,
            grn_effect: 1.0,
            unregulated_gene_ratio: 0.1,
            beta: 0.4,
            degradation: 1.0,
            num_cycles: 3,
            cycle_len: 1.0,
            scale_s: 1.0,
            discrete_cif: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SimResult {
    /// Gene × cell expression counts.
    pub counts: Mat,
    /// Region × cell ATAC counts after intrinsic noise.
    pub atac_counts: Mat,
    /// Region × cell ATAC signal before intrinsic noise.
    pub atacseq_data: Mat,
    pub unspliced_counts: Option<Mat>,
    pub velocity: Option<Mat>,
    pub cell_pop: Vec<String>,
    pub cell_depth: Vec<f64>,
    /// Rows are `(from, to, time, state)` along the sampled trajectory.
    pub neutral: Mat,
    /// Kinetic parameters, gene × cell, the rates that produced `counts`.
    pub kon: Mat,
    pub koff: Mat,
    pub s: Mat,
    /// Gene identity vectors. Rows are genes.
    pub giv_kon: Mat,
    pub giv_koff: Mat,
    pub giv_s: Mat,
    /// Cell identity factors. Rows are cells.
    pub cif_kon: Mat,
    pub cif_koff: Mat,
    pub cif_s: Mat,
    pub cif_atac: Mat,
    /// Region identity vectors, regions × factors.
    pub riv: Mat,
    /// Region × gene indicator of the chromatin-to-gene map.
    pub region_to_gene: Mat,
    /// Gene × regulator effects used inside the simulation.
    pub geff: Mat,
    pub regulators: Vec<usize>,
    pub targets: Vec<usize>,
    /// Edges as `(target, regulator, effect)` with 0-based gene indexes.
    pub grn_edges: Vec<(usize, usize, f64)>,
    /// Original GRN gene id for each column, or -1 for an unregulated extra gene.
    pub gene_ids: Vec<i32>,
    pub beta: Vec<f64>,
    pub degradation: Vec<f64>,
}

#[derive(Debug)]
pub enum SimError {
    Unsupported(&'static str),
}

impl std::fmt::Display for SimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimError::Unsupported(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for SimError {}

struct Density {
    x: Vec<f64>,
    y: Vec<f64>,
}

fn load_f64(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn load_i32(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn density(x: &[u8], y: &[u8]) -> Density {
    Density {
        x: load_f64(x),
        y: load_f64(y),
    }
}

struct Builtins {
    kon: Density,
    koff: Density,
    s: Density,
    atac: Density,
}

fn builtins() -> Builtins {
    Builtins {
        kon: density(
            include_bytes!("../fixtures/den_kon_x.bin"),
            include_bytes!("../fixtures/den_kon_y.bin"),
        ),
        koff: density(
            include_bytes!("../fixtures/den_koff_x.bin"),
            include_bytes!("../fixtures/den_koff_y.bin"),
        ),
        s: density(
            include_bytes!("../fixtures/den_s_x.bin"),
            include_bytes!("../fixtures/den_s_y.bin"),
        ),
        atac: density(
            include_bytes!("../fixtures/den_atac_x.bin"),
            include_bytes!("../fixtures/den_atac_y.bin"),
        ),
    }
}

/// The bundled 100-gene GRN (`GRN_params_100`).
pub fn grn_params_100() -> Vec<GrnEdge> {
    let n = i32::from_le_bytes(
        include_bytes!("../fixtures/grn100_n.bin")[..4]
            .try_into()
            .unwrap(),
    ) as usize;
    let tgt = load_i32(include_bytes!("../fixtures/grn100_tgt.bin"));
    let reg = load_i32(include_bytes!("../fixtures/grn100_reg.bin"));
    let eff = load_f64(include_bytes!("../fixtures/grn100_eff.bin"));
    (0..n)
        .map(|i| GrnEdge {
            target: tgt[i],
            regulator: reg[i],
            effect: eff[i],
        })
        .collect()
}

fn consume_rtree2(rng: &mut Rng) {
    let _ = rng.sample_uniform(2, 2, false);
    let _ = rng.unif();
    let _ = rng.unif();
}

pub fn phyla5(rng: &mut Rng) -> Tree {
    consume_rtree2(rng);
    Tree {
        n_tip: 5,
        edges: vec![
            Edge {
                from: 6,
                to: 7,
                len: 1.0,
            },
            Edge {
                from: 7,
                to: 8,
                len: 1.0,
            },
            Edge {
                from: 8,
                to: 2,
                len: 1.0,
            },
            Edge {
                from: 8,
                to: 3,
                len: 1.0,
            },
            Edge {
                from: 7,
                to: 9,
                len: 1.0,
            },
            Edge {
                from: 9,
                to: 4,
                len: 0.2,
            },
            Edge {
                from: 9,
                to: 5,
                len: 0.2,
            },
            Edge {
                from: 6,
                to: 1,
                len: 3.0,
            },
        ],
    }
}

pub fn phyla3(rng: &mut Rng) -> Tree {
    consume_rtree2(rng);
    Tree {
        n_tip: 3,
        edges: vec![
            Edge {
                from: 4,
                to: 5,
                len: 1.0,
            },
            Edge {
                from: 5,
                to: 2,
                len: 1.0,
            },
            Edge {
                from: 5,
                to: 3,
                len: 1.0,
            },
            Edge {
                from: 4,
                to: 1,
                len: 2.0,
            },
        ],
    }
}

pub fn phyla1(len: f64) -> Tree {
    Tree {
        n_tip: 1,
        edges: vec![Edge {
            from: 2,
            to: 1,
            len,
        }],
    }
}

struct PreparedGrn {
    regulators: Vec<usize>,
    targets: Vec<usize>,
    geff: Mat,
    n_gene: usize,
    edges: Vec<(usize, usize, f64)>,
    gene_ids: Vec<i32>,
}

fn prepare_grn(edges: &[GrnEdge], ratio: f64) -> PreparedGrn {
    let mut ids: Vec<i32> = edges.iter().flat_map(|e| [e.target, e.regulator]).collect();
    ids.sort_unstable();
    ids.dedup();
    let map = |id: i32| ids.iter().position(|&v| v == id).unwrap();
    let mapped: Vec<(usize, usize, f64)> = edges
        .iter()
        .map(|e| (map(e.target), map(e.regulator), e.effect))
        .collect();
    let n_grn = ids.len();
    let n = n_grn as f64;
    let n_gene = (((n + n * ratio) / 10.0).ceil() * 10.0) as usize;
    let mut regulators: Vec<usize> = mapped.iter().map(|e| e.1).collect();
    regulators.sort_unstable();
    regulators.dedup();
    let mut targets: Vec<usize> = mapped.iter().map(|e| e.0).collect();
    targets.sort_unstable();
    targets.dedup();
    let mut geff = Mat::zeros(n_gene, regulators.len());
    let edges = mapped.clone();
    for (t, r, effect) in mapped {
        let col = regulators.iter().position(|&g| g == r).unwrap();
        geff.set(t, col, effect);
    }
    let mut gene_ids = vec![-1i32; n_gene];
    for (index, id) in ids.iter().enumerate() {
        gene_ids[index] = *id;
    }
    PreparedGrn {
        regulators,
        targets,
        geff,
        n_gene,
        edges,
        gene_ids,
    }
}

fn rows_to_mat(rows: &[[f64; 4]]) -> Mat {
    let mut m = Mat::zeros(rows.len(), 4);
    for (r, row) in rows.iter().enumerate() {
        for c in 0..4 {
            m.set(r, c, row[c]);
        }
    }
    m
}

fn sample_edge(
    rng: &mut Rng,
    tree: &Tree,
    edge: &Edge,
    depth: f64,
    anc_state: f64,
    ncells: usize,
    provided: Option<&[f64]>,
) -> Vec<[f64; 4]> {
    let t_sample = if let Some(ts) = provided {
        let mut t = Vec::with_capacity(ts.len() + 1);
        t.push(0.0);
        t.extend(ts.iter().map(|x| x - depth));
        t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        t
    } else {
        let total: f64 = tree.edges.iter().map(|e| e.len).sum();
        let ncell_branch = (edge.len * ncells as f64 / total).ceil() as i32 - 1;
        if ncell_branch < 0 {
            panic!("the total number of cells is too few");
        }
        let mut t = vec![0.0];
        if ncell_branch == 0 {
            t.push(0.0);
            t.push(edge.len);
        } else {
            let step = edge.len / ncell_branch as f64;
            t.extend(seq_by(0.0, edge.len, step));
        }
        t.push(edge.len);
        t
    };
    let mut changes = Vec::with_capacity(t_sample.len().saturating_sub(1));
    for i in 1..t_sample.len() {
        let sig = t_sample[i] - t_sample[i - 1];
        changes.push(rng.rnorm(0.0, sig.sqrt()));
    }
    let summed = cumsum_r(&changes);
    let mut rows = Vec::with_capacity(summed.len());
    for (i, acc) in summed.iter().enumerate() {
        rows.push([
            edge.from as f64,
            edge.to as f64,
            depth + t_sample[i + 1],
            anc_state + acc,
        ]);
    }
    rows
}

unsafe extern "C" {
    fn r_cumsum(x: *const f64, y: *mut f64, n: usize);
    fn r_mean(x: *const f64, n: usize) -> f64;
}

fn cumsum_r(x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    if !x.is_empty() {
        unsafe { r_cumsum(x.as_ptr(), y.as_mut_ptr(), x.len()) }
    }
    y
}

fn sample_subtree(
    rng: &mut Rng,
    tree: &Tree,
    par: i32,
    depth: f64,
    anc_state: f64,
    ncells: usize,
    neutral: Option<&Mat>,
) -> Vec<[f64; 4]> {
    let children: Vec<i32> = tree
        .edges
        .iter()
        .filter(|e| e.from == par)
        .map(|e| e.to)
        .collect();
    let mut rows = Vec::new();
    for child in children {
        let edge = tree
            .edges
            .iter()
            .find(|e| e.from == par && e.to == child)
            .unwrap();
        let owned;
        let provided = if let Some(n) = neutral {
            owned = (0..n.nrow)
                .filter(|&i| n.get(i, 0) as i32 == edge.from && n.get(i, 1) as i32 == edge.to)
                .map(|i| n.get(i, 2))
                .collect::<Vec<_>>();
            Some(owned.as_slice())
        } else {
            owned = Vec::new();
            let _ = &owned;
            None
        };
        let part = sample_edge(rng, tree, edge, depth, anc_state, ncells, provided);
        let anc_next = part.last().map(|r| r[3]).unwrap_or(anc_state);
        let leaf = tree.edges.iter().all(|e| e.from != child);
        rows.extend(part);
        if !leaf {
            rows.extend(sample_subtree(
                rng,
                tree,
                child,
                depth + edge.len,
                anc_next,
                ncells,
                neutral,
            ));
        }
    }
    rows
}

fn matrix_from_cols(nrow: usize, cols: &[Vec<f64>]) -> Mat {
    let mut m = Mat::zeros(nrow, cols.len());
    for (c, col) in cols.iter().enumerate() {
        m.set_col(c, col);
    }
    m
}

/// Returns `(kon, koff, s, neutral)`.
fn continuous_cif(
    rng: &mut Rng,
    tree: &Tree,
    ncells: usize,
    n_nd: [usize; 3],
    n_diff: [usize; 3],
    n_reg: usize,
    center: f64,
    sigma: f64,
) -> (Mat, Mat, Mat, Mat) {
    let neutral_rows = sample_subtree(rng, tree, tree.root(), 0.0, center, ncells, None);
    let neutral = rows_to_mat(&neutral_rows);
    let mut out = Vec::with_capacity(3);
    for i in 0..3 {
        let mut cols = Vec::new();
        for _ in 0..n_nd[i] {
            cols.push((0..ncells).map(|_| rng.rnorm(center, sigma)).collect());
        }
        if n_diff[i] > 0 {
            for _ in 0..n_diff[i] {
                let sampled =
                    sample_subtree(rng, tree, tree.root(), 0.0, center, ncells, Some(&neutral));
                let col = (0..ncells).map(|r| sampled[r][3]).collect();
                cols.push(col);
            }
        }
        if i < 2 && n_reg > 0 {
            for _ in 0..n_reg {
                cols.push((0..ncells).map(|_| rng.rnorm(center, sigma)).collect());
            }
        }
        out.push(matrix_from_cols(ncells, &cols));
    }
    (out.remove(0), out.remove(0), out.remove(0), neutral)
}

fn identity_vectors(
    rng: &mut Rng,
    size: usize,
    n_cif: usize,
    prob: f64,
    mean: f64,
    sd: f64,
) -> Mat {
    let mut m = Mat::zeros(size, n_cif);
    for i in 0..size {
        let draws = rng.sample_prob(&[1.0 - prob, prob], n_cif, true);
        let mut n1 = 0;
        let mut row = vec![0.0; n_cif];
        for (j, &d) in draws.iter().enumerate() {
            if d == 2 {
                row[j] = 1.0;
                n1 += 1;
            }
        }
        if n1 > 0 {
            for v in &mut row {
                if *v != 0.0 {
                    *v = rng.rnorm(mean, sd);
                }
            }
        }
        m.set_row(i, &row);
    }
    m
}

fn sample_den(rng: &mut Rng, nsample: usize, den: &Density) -> Vec<f64> {
    if nsample == 0 {
        return Vec::new();
    }
    let sumy: f64 = den.y.iter().sum();
    let probs: Vec<f64> = den.y.iter().map(|v| v / sumy).collect();
    let bw = den.x[1] - den.x[0];
    let bins = rng.sample_prob(&probs, nsample, true);
    let mut counts = vec![0usize; probs.len()];
    for id in &bins {
        counts[(id - 1) as usize] += 1;
    }
    let mut samples = vec![0.0; nsample];
    let mut cursor = 0;
    for (j, &c) in counts.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let min = den.x[j] - 0.5 * bw;
        let max = den.x[j] + 0.5 * bw;
        for t in 0..c {
            samples[cursor + t] = rng.runif(min, max);
        }
        cursor += c;
    }
    samples
}

fn match_density(rng: &mut Rng, values_x: &[f64], pool: &mut Vec<f64>, den: &Density) -> Vec<f64> {
    let mut values = values_x.to_vec();
    values.extend_from_slice(pool);
    let ranks = rank_average(&values);
    let nsample = ranks
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .trunc()
        .max(0.0) as usize;
    let mut sorted = sample_den(rng, nsample, den);
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let out = ranks[..values_x.len()]
        .iter()
        .map(|&rk| sorted[r_index(rk)])
        .collect();
    if pool.len() < 20_000 {
        *pool = values;
    }
    out
}

fn match_matrix(rng: &mut Rng, x: &Mat, pool: &mut Vec<f64>, den: &Density) -> Mat {
    let matched = match_density(rng, &x.data, pool, den);
    Mat::from_col_major(x.nrow, x.ncol, matched)
}

fn region_to_gene(rng: &mut Rng, n_region: usize, n_gene: usize, distrib: [f64; 3]) -> Mat {
    let idx = rng.sample_prob(&distrib, n_gene, true);
    let regu_by: Vec<i32> = idx.into_iter().map(|i| i - 1).collect();
    let by1: Vec<usize> = regu_by
        .iter()
        .enumerate()
        .filter(|(_, v)| **v == 1)
        .map(|(i, _)| i)
        .collect();
    let by2: Vec<usize> = regu_by
        .iter()
        .enumerate()
        .filter(|(_, v)| **v == 2)
        .map(|(i, _)| i)
        .collect();
    let mut res = Mat::zeros(n_region, n_gene);
    if !by1.is_empty() {
        let regs = rng.sample_uniform(n_region as i32, by1.len() as i32, true);
        for (k, &g) in by1.iter().enumerate() {
            res.set((regs[k] - 1) as usize, g, 1.0);
        }
    }
    if !by2.is_empty() {
        let regs = rng.sample_uniform((n_region - 1) as i32, by2.len() as i32, true);
        for (k, &g) in by2.iter().enumerate() {
            let r = (regs[k] - 1) as usize;
            res.set(r, g, 1.0);
            res.set(r + 1, g, 1.0);
        }
    }
    res
}

fn build_giv(
    rng: &mut Rng,
    opt: &Options,
    grn: &PreparedGrn,
    n_diff_s: usize,
    n_nd_s: usize,
) -> [Mat; 3] {
    let n_cif = opt.num_cifs;
    let mut giv = [
        identity_vectors(
            rng,
            grn.n_gene,
            n_cif,
            opt.giv_prob,
            opt.giv_mean,
            opt.giv_sd,
        ),
        identity_vectors(
            rng,
            grn.n_gene,
            n_cif,
            opt.giv_prob,
            opt.giv_mean,
            opt.giv_sd,
        ),
        identity_vectors(
            rng,
            grn.n_gene,
            n_cif,
            opt.giv_prob,
            opt.giv_mean,
            opt.giv_sd,
        ),
    ];
    let n_reg = grn.regulators.len();
    giv[0] = giv[0].hcat(&Mat::zeros(grn.n_gene, n_reg));
    giv[1] = giv[1].hcat(&Mat::zeros(grn.n_gene, n_reg));

    let mut in_grn = vec![false; grn.n_gene];
    for &g in grn.regulators.iter().chain(grn.targets.iter()) {
        in_grn[g] = true;
    }
    for g in 0..grn.n_gene {
        if in_grn[g] {
            for c in 0..giv[2].ncol {
                giv[2].set(g, c, 0.0);
            }
        }
    }
    let mut regu_diff = Mat::zeros(n_reg, n_diff_s);
    for row in 0..n_reg {
        let draw = rng.sample_uniform(n_diff_s as i32, 2, false);
        regu_diff.set(row, (draw[0] - 1) as usize, 2.0);
        regu_diff.set(row, (draw[1] - 1) as usize, 2.0);
    }
    for (ri, &g) in grn.regulators.iter().enumerate() {
        for c in 0..n_nd_s {
            giv[2].set(g, c, 0.0);
        }
        for c in 0..n_diff_s {
            giv[2].set(g, n_nd_s + c, regu_diff.get(ri, c));
        }
    }
    let reg_giv = giv[2].select_rows(&grn.regulators);
    let prod = matmul(&grn.geff, &reg_giv);
    let mut eff_rows: Vec<(usize, Vec<f64>)> = Vec::new();
    for g in 0..grn.n_gene {
        let n = (0..n_reg).filter(|&c| grn.geff.get(g, c) != 0.0).count();
        if n == 0 {
            continue;
        }
        let row = (0..n_cif)
            .map(|c| prod.get(g, c) / n as f64 / 2.0)
            .collect::<Vec<_>>();
        eff_rows.push((g, row));
    }
    for (g, mut row) in eff_rows {
        let sum: f64 = (0..n_cif).map(|c| giv[2].get(g, c)).sum();
        if sum > 0.0 {
            for c in 0..n_cif {
                row[c] = row[c] * 0.5 + giv[2].get(g, c) * 0.5;
            }
        }
        for c in 0..n_cif {
            giv[2].set(g, c, row[c]);
        }
    }
    for &g in &grn.targets {
        for c in 0..giv[2].ncol {
            if giv[2].get(g, c).abs() < 0.2 {
                giv[2].set(g, c, 0.0);
            }
        }
    }
    for g in 0..grn.n_gene {
        if !in_grn[g] {
            for c in 0..giv[2].ncol {
                giv[2].set(g, c, giv[2].get(g, c) * 3.0);
            }
        }
    }
    giv
}

fn atac_adjust_kon(
    rng: &mut Rng,
    kon: &mut Mat,
    atac: &Mat,
    region: &Mat,
    effect: f64,
    den: &Density,
) {
    let mut region = region.clone();
    let support = region.col_sums();
    for (g, &sg) in support.iter().enumerate() {
        if sg > 1.0 {
            for r in 0..region.nrow {
                region.set(r, g, region.get(r, g) / sg);
            }
        }
    }
    let atac_kon = matmul(atac, &region);
    let affected: Vec<usize> = support
        .iter()
        .enumerate()
        .filter(|(_, s)| **s > 0.0)
        .map(|(i, _)| i)
        .collect();
    let kon_sub = kon.select_cols(&affected);
    let mut atac_sub = atac_kon.select_cols(&affected);
    let mut holes = Vec::new();
    for i in 0..atac_sub.data.len() {
        if atac_sub.data[i] == 0.0 {
            holes.push(i);
        }
    }
    if !holes.is_empty() {
        let mut filler: Vec<f64> = holes.iter().map(|&i| kon_sub.data[i]).collect();
        let min_f = filler.iter().copied().fold(f64::INFINITY, f64::min);
        for v in &mut filler {
            *v -= min_f;
        }
        let max_f = filler.iter().copied().fold(0.0_f64, f64::max);
        let min_pos = atac_kon
            .data
            .iter()
            .copied()
            .filter(|v| *v > 0.0)
            .fold(f64::INFINITY, f64::min);
        let scale = min_pos / 2.0 / max_f;
        for v in &mut filler {
            *v *= scale;
        }
        for (k, &i) in holes.iter().enumerate() {
            atac_sub.data[i] = filler[k];
        }
    }
    let r_atac = rank_average(&atac_sub.data);
    let r_kon = rank_average(&kon_sub.data);
    let ranks: Vec<f64> = r_atac
        .iter()
        .zip(r_kon.iter())
        .map(|(a, b)| effect * a + (1.0 - effect) * b)
        .collect();
    let nsample = ranks
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .trunc()
        .max(0.0) as usize;
    let mut sorted = sample_den(rng, nsample, den);
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    for (i, &rk) in ranks.iter().enumerate() {
        atac_sub.data[i] = sorted[r_index(rk)];
    }
    for (j, &g) in affected.iter().enumerate() {
        for cell in 0..kon.nrow {
            kon.set(cell, g, atac_sub.get(cell, j));
        }
    }
    let _ = atac_kon;
}

fn pow10_gene_cell(cell_gene: &Mat, bimod: &[f64]) -> Mat {
    let ncells = cell_gene.nrow;
    let ngenes = cell_gene.ncol;
    let mut out = Mat::zeros(ngenes, ncells);
    for g in 0..ngenes {
        for cell in 0..ncells {
            out.set(g, cell, 10f64.powf(cell_gene.get(cell, g) - bimod[g]));
        }
    }
    out
}

fn beta_poisson(rng: &mut Rng, kon: f64, koff: f64, s: f64, noise: f64) -> f64 {
    let y_mean = kon / (kon + koff);
    let x_mean = y_mean * s;
    let y = rng.rbeta(kon, koff);
    let x = rng.rpois(y * s);
    noise * x + (1.0 - noise) * x_mean
}

fn telegraph_step(kon: f64, koff: f64, cycle_factor: f64) -> (f64, f64, f64) {
    let cycle_length = 1.0 / kon + 1.0 / koff;
    let min_w = (1.0 / kon).min(1.0 / koff);
    let npart = ((cycle_length / min_w).ceil() * cycle_factor).max(cycle_length * cycle_factor);
    let stepsize = cycle_length / npart;
    let p_off = 1.0 / (npart * ((1.0 / koff) / cycle_length));
    let p_on = 1.0 / (npart * ((1.0 / kon) / cycle_length));
    (stepsize, p_off, p_on)
}

/// Telegraph trajectory without storing every step.
///
/// The sampled point is the same one R's `which.min` keeps: the first minimum
/// of `|time within the last cycle - target|`. `next_unif` supplies the promoter
/// switches. Callers may pass the master stream or a per-gene stream.
fn integrate_gene(
    next_unif: &mut impl FnMut() -> f64,
    target: f64,
    kon: f64,
    koff: f64,
    s: f64,
    beta: f64,
    d: f64,
    start_state: f64,
    start_u: Option<f64>,
    start_s: Option<f64>,
    ncells: usize,
    cycle_factor: f64,
) -> (f64, f64, f64) {
    let (mut stepsize, mut p_off, mut p_on) = telegraph_step(kon, koff, cycle_factor);
    let mut x = start_u.unwrap_or(s * kon / (kon + koff) / beta);
    let mut y = start_s.unwrap_or(s * kon / (kon + koff) / d);
    let mut which = 1.0;
    let mut time = 0.0;
    let mut state = start_state;
    let mut best_abs = f64::MAX;
    let mut best = (x, y, state);
    if which == ncells as f64 {
        best_abs = ((time - (ncells as f64 - 1.0)) - target).abs();
    }
    loop {
        let new_time = time + stepsize;
        time = new_time;
        if new_time - which > 0.0 {
            let nc = which + 1.0;
            if nc > ncells as f64 {
                break;
            }
            which = nc;
            let stepped = telegraph_step(kon, koff, cycle_factor);
            stepsize = stepped.0;
            p_off = stepped.1;
            p_on = stepped.2;
        }
        let u = next_unif();
        let threshold = if state == 1.0 { 1.0 - p_off } else { p_on };
        let new_state = if u > threshold { 2.0 } else { 1.0 };
        let xprev = x;
        let yprev = y;
        let mut xn = if new_state == 1.0 {
            xprev + s * stepsize - beta * xprev * stepsize
        } else {
            xprev - beta * xprev * stepsize
        };
        if xn < 0.0 {
            xn = 0.0;
        }
        let mut yn = yprev + beta * xprev * stepsize - d * yprev * stepsize;
        if yn < 0.0 {
            yn = 0.0;
        }
        x = xn;
        y = yn;
        state = new_state;
        if which == ncells as f64 {
            let val = ((time - (ncells as f64 - 1.0)) - target).abs();
            if val < best_abs {
                best_abs = val;
                best = (x, y, state);
            }
        }
    }
    (best.0.trunc(), best.1.trunc(), best.2)
}

fn par_map<R, F>(n: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, n.max(1));
    if workers == 1 || n < 8 {
        return (0..n).map(&f).collect();
    }
    let chunk = n.div_ceil(workers);
    let f = &f;
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for t in 0..workers {
            let start = t * chunk;
            if start >= n {
                break;
            }
            let end = (start + chunk).min(n);
            handles.push(scope.spawn(move || (start..end).map(|i| (i, f(i))).collect::<Vec<_>>()));
        }
        let mut out: Vec<Option<R>> = Vec::with_capacity(n);
        out.resize_with(n, || None);
        for handle in handles {
            for (i, value) in handle.join().unwrap() {
                out[i] = Some(value);
            }
        }
        out.into_iter().map(|value| value.unwrap()).collect()
    })
}

struct KineticGene {
    seed: u32,
    kon: f64,
    koff: f64,
    s: f64,
    beta: f64,
    d: f64,
    start_state: f64,
    start_u: Option<f64>,
    start_s: Option<f64>,
}

pub fn sim_true_counts(opt: &Options, rng: &mut Rng) -> Result<SimResult, SimError> {
    blas_single_thread();
    if opt.discrete_cif {
        return Err(SimError::Unsupported(
            "discrete cell populations are not in this Rust port yet",
        ));
    }
    // `.opt_list()` evaluates `tree = .default(Phyla5())` before the supplied
    // tree is kept, so `rtree(2)` always runs once even when the caller already
    // built the tree.
    consume_rtree2(rng);
    let _ = rng.sample_uniform(100_000, 9, false);
    let grn = prepare_grn(&opt.grn, opt.unregulated_gene_ratio);
    let n_gene = grn.n_gene;
    let n_cell = opt.num_cells;
    let n_reg = grn.regulators.len();
    let mask = match opt.vary {
        Vary::All => [true, true, true],
        Vary::Kon => [true, false, false],
        Vary::Koff => [false, true, false],
        Vary::S => [false, false, true],
        Vary::ExceptKon => [false, true, true],
        Vary::ExceptKoff => [true, false, true],
        Vary::ExceptS => [true, true, false],
    };
    let n_diff_cif = (opt.num_cifs as f64 * opt.diff_cif_fraction).ceil() as usize;
    let n_diff = [
        if mask[0] { n_diff_cif } else { 0 },
        if mask[1] { n_diff_cif } else { 0 },
        if mask[2] { n_diff_cif } else { 0 },
    ];
    let n_nd = [
        opt.num_cifs - n_diff[0],
        opt.num_cifs - n_diff[1],
        opt.num_cifs - n_diff[2],
    ];
    let (cif_kon, cif_koff, cif_s, neutral) = continuous_cif(
        rng,
        &opt.tree,
        n_cell,
        n_nd,
        n_diff,
        n_reg,
        opt.cif_center,
        opt.cif_sigma,
    );
    let n_region = opt.region_distrib.len() * n_gene;
    let riv = identity_vectors(
        rng,
        n_region,
        opt.num_cifs,
        opt.riv_prob,
        opt.riv_mean,
        opt.riv_sd,
    );
    let giv = build_giv(rng, opt, &grn, n_diff[2], n_nd[2]);
    let region = region_to_gene(rng, n_region, n_gene, opt.region_distrib);
    let (_a_kon, _a_koff, cif_atac, _) = continuous_cif(
        rng,
        &opt.tree,
        n_cell,
        n_nd,
        n_diff,
        n_reg,
        opt.cif_center,
        opt.cif_sigma,
    );
    let dens = builtins();
    let params_region = matmul_bt(&cif_atac, &riv);
    let n_val = params_region.data.len();
    let ranks = rank_average(&params_region.data);
    let n_zero = (n_val as f64 * opt.atac_p_zero).floor() as usize;
    let n_nonzero = n_val - n_zero;
    let mut sampled = sample_den(rng, n_nonzero, &dens.atac);
    sampled.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut atac_raw = vec![0.0; n_val];
    for i in 0..n_val {
        if ranks[i] > n_zero as f64 {
            let idx = ranks[i] - n_zero as f64;
            atac_raw[i] = 2f64.powf(sampled[r_index(idx)]) - 1.0;
        }
    }
    let atac = Mat::from_col_major(params_region.nrow, params_region.ncol, atac_raw);

    let mut pool_kon = Vec::new();
    let mut pool_koff = Vec::new();
    let mut pool_s = Vec::new();
    let mut kon = match_matrix(rng, &matmul_bt(&cif_kon, &giv[0]), &mut pool_kon, &dens.kon);
    let koff = match_matrix(
        rng,
        &matmul_bt(&cif_koff, &giv[1]),
        &mut pool_koff,
        &dens.koff,
    );
    atac_adjust_kon(rng, &mut kon, &atac, &region, opt.atac_effect, &dens.kon);
    let n_bimod = (n_gene as f64 * 0.5).ceil() as i32;
    let chosen = rng.sample_uniform(n_gene as i32, n_bimod, false);
    let mut bimod = vec![0.0; n_gene];
    for g in chosen {
        bimod[(g - 1) as usize] = opt.bimod;
    }
    let kon = pow10_gene_cell(&kon, &bimod);
    let koff = pow10_gene_cell(&koff, &bimod);

    let s_base = matmul_bt(&cif_s, &giv[2]);
    let mut s_param = Mat::zeros(n_gene, n_cell);
    let root = opt.tree.root();
    let mut cif_regu = vec![vec![0.0; n_reg]; n_cell];
    let mut counts_cg = Mat::zeros(n_cell, n_gene);
    let mut unspliced_cg = Mat::zeros(n_cell, n_gene);
    let mut velocity_cg = Mat::zeros(n_cell, n_gene);
    let mut state = vec![vec![0.0; n_gene]; n_cell];
    let mut cell_time = vec![0.0; n_cell];
    let root_state: Vec<f64> = if opt.do_velocity {
        rng.sample_uniform(2, n_gene as i32, true)
            .into_iter()
            .map(|v| v as f64)
            .collect()
    } else {
        Vec::new()
    };
    let d_genes: Vec<f64> = if opt.do_velocity {
        (0..n_gene)
            .map(|_| rng.rnorm(opt.degradation, 0.1))
            .collect()
    } else {
        Vec::new()
    };
    let beta_genes: Vec<f64> = if opt.do_velocity {
        (0..n_gene).map(|_| rng.rnorm(opt.beta, 0.1)).collect()
    } else {
        Vec::new()
    };

    for edge in &opt.tree.edges {
        let cells: Vec<usize> = (0..n_cell)
            .filter(|&i| {
                neutral.get(i, 0) as i32 == edge.from && neutral.get(i, 1) as i32 == edge.to
            })
            .collect();
        let mut curr = if edge.from == root {
            (0..n_reg)
                .map(|_| rng.rnorm(opt.cif_center, opt.cif_sigma))
                .collect::<Vec<_>>()
        } else {
            let gp = opt.tree.parent(edge.from).expect("parent");
            let last = (0..n_cell)
                .filter(|&i| {
                    neutral.get(i, 0) as i32 == gp && neutral.get(i, 1) as i32 == edge.from
                })
                .max()
                .expect("parent edge produced no cells");
            cif_regu[last].clone()
        };
        let mut linear = vec![0.0; n_gene];
        let mut curr_row = Mat::zeros(1, n_reg);
        let mut extra = Mat::zeros(1, n_gene);
        for (n, &i_cell) in cells.iter().enumerate() {
            s_base.copy_row(i_cell, &mut linear);
            if n_reg > 0 {
                curr_row.data.copy_from_slice(&curr);
                matmul_bt_into(&curr_row, &grn.geff, &mut extra);
                for g in 0..n_gene {
                    linear[g] += extra.get(0, g) * opt.grn_effect;
                }
            }
            let matched = match_density(rng, &linear, &mut pool_s, &dens.s);
            let s_cell: Vec<f64> = matched
                .iter()
                .enumerate()
                .map(|(_, v)| 10f64.powf(*v) * opt.scale_s)
                .collect();
            for (g, value) in s_cell.iter().enumerate() {
                s_param.set(g, i_cell, *value);
            }
            if opt.do_velocity {
                let last_parent = if n == 0 {
                    if edge.from == root {
                        None
                    } else {
                        let gp = opt.tree.parent(edge.from).unwrap();
                        (0..n_cell)
                            .filter(|&i| {
                                neutral.get(i, 0) as i32 == gp
                                    && neutral.get(i, 1) as i32 == edge.from
                            })
                            .max()
                    }
                } else {
                    Some(cells[n - 1])
                };
                let fst = last_parent.is_none();
                let cycles = if fst { 15 } else { opt.num_cycles };
                // One master uniform seeds each gene's own generator, then the
                // shared cycle phase is drawn. The telegraph steps, which are
                // the long loop, run in parallel and do not take turns on the
                // master stream. The numbers differ from R's single stream; the
                // rates and the which.min rule do not.
                let parent = last_parent.unwrap_or(0);
                let mut jobs = Vec::with_capacity(n_gene);
                for g in 0..n_gene {
                    let seed = (rng.unif() * 4294967296.0) as u32;
                    jobs.push(KineticGene {
                        seed,
                        kon: kon.get(g, i_cell),
                        koff: koff.get(g, i_cell),
                        s: s_cell[g],
                        beta: beta_genes[g],
                        d: d_genes[g],
                        start_state: if fst { root_state[g] } else { state[parent][g] },
                        start_u: if fst {
                            None
                        } else {
                            Some(unspliced_cg.get(parent, g))
                        },
                        start_s: if fst {
                            None
                        } else {
                            Some(counts_cg.get(parent, g))
                        },
                    });
                }
                let randpoints: Vec<f64> = (0..cycles).map(|_| rng.unif()).collect();
                let target = randpoints[cycles - 1];
                let solved = par_map(n_gene, |g| {
                    let job = &jobs[g];
                    let mut child = Rng::seed(job.seed);
                    integrate_gene(
                        &mut || child.unif(),
                        target,
                        job.kon,
                        job.koff,
                        job.s,
                        job.beta,
                        job.d,
                        job.start_state,
                        job.start_u,
                        job.start_s,
                        cycles,
                        opt.cycle_len,
                    )
                });
                let mut u_out = vec![0.0; n_gene];
                let mut s_out = vec![0.0; n_gene];
                let mut st_out = vec![0.0; n_gene];
                for (g, (u, s, st)) in solved.into_iter().enumerate() {
                    u_out[g] = u;
                    s_out[g] = s;
                    st_out[g] = st;
                }
                state[i_cell] = st_out;
                let start_t = if fst {
                    0.0
                } else {
                    cell_time[last_parent.unwrap()]
                };
                cell_time[i_cell] = randpoints[cycles - 1] + (cycles - 1) as f64 + start_t;
                for g in 0..n_gene {
                    unspliced_cg.set(i_cell, g, u_out[g]);
                    counts_cg.set(i_cell, g, s_out[g]);
                    velocity_cg.set(i_cell, g, beta_genes[g] * u_out[g] - d_genes[g] * s_out[g]);
                }
                let mean = s_out.iter().sum::<f64>() / n_gene as f64;
                for (k, &gene) in grn.regulators.iter().enumerate() {
                    curr[k] = s_out[gene] / (s_out[gene] + mean);
                }
            } else {
                let mut counts = vec![0.0; n_gene];
                for g in 0..n_gene {
                    counts[g] = beta_poisson(
                        rng,
                        kon.get(g, i_cell),
                        koff.get(g, i_cell),
                        s_cell[g],
                        opt.intrinsic_noise,
                    );
                }
                for g in 0..n_gene {
                    counts_cg.set(i_cell, g, counts[g]);
                }
                let mean = counts.iter().sum::<f64>() / n_gene as f64;
                for (k, &gene) in grn.regulators.iter().enumerate() {
                    curr[k] = counts[gene] / (counts[gene] + mean);
                }
            }
            cif_regu[i_cell] = curr.clone();
        }
    }

    let atac_clean = atac.transpose();
    let mean_atac = unsafe { r_mean(atac_clean.data.as_ptr(), atac_clean.data.len()) };
    let mut atac_counts = atac_clean.clone();
    for v in &mut atac_counts.data {
        *v += rng.rnorm(0.0, mean_atac * 1.5);
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    let mut cell_pop = Vec::with_capacity(n_cell);
    let mut cell_depth = Vec::with_capacity(n_cell);
    for i in 0..n_cell {
        cell_pop.push(format!(
            "{}_{}",
            neutral.get(i, 0) as i32,
            neutral.get(i, 1) as i32
        ));
        cell_depth.push(neutral.get(i, 2));
    }
    Ok(SimResult {
        counts: counts_cg.transpose(),
        atac_counts,
        atacseq_data: atac_clean,
        unspliced_counts: opt.do_velocity.then_some(unspliced_cg.transpose()),
        velocity: opt.do_velocity.then_some(velocity_cg.transpose()),
        cell_pop,
        cell_depth,
        neutral,
        kon,
        koff,
        s: s_param,
        giv_kon: giv[0].clone(),
        giv_koff: giv[1].clone(),
        giv_s: giv[2].clone(),
        cif_kon,
        cif_koff,
        cif_s,
        cif_atac,
        riv,
        region_to_gene: region,
        geff: grn.geff,
        regulators: grn.regulators,
        targets: grn.targets,
        grn_edges: grn.edges,
        gene_ids: grn.gene_ids,
        beta: beta_genes,
        degradation: d_genes,
    })
}
