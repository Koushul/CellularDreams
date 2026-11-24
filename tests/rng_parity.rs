use scmultisim::Rng;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join(name)
}

fn read_f64(name: &str) -> Vec<f64> {
    let bytes = std::fs::read(fixture(name)).unwrap();
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_i32(name: &str) -> Vec<i32> {
    let bytes = std::fs::read(fixture(name)).unwrap();
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn unif_matches_r() {
    let expect = read_f64("rng_unif.bin");
    let mut rng = Rng::seed(0);
    let got: Vec<f64> = (0..expect.len()).map(|_| rng.unif()).collect();
    assert_eq!(got, expect);
}

#[test]
fn norm_matches_r() {
    let expect = read_f64("rng_norm.bin");
    let mut rng = Rng::seed(0);
    let got: Vec<f64> = (0..expect.len()).map(|_| rng.rnorm(0.0, 1.0)).collect();
    assert_eq!(got, expect);
}

#[test]
fn exp_matches_r() {
    let expect = read_f64("rng_exp.bin");
    let mut rng = Rng::seed(0);
    let got: Vec<f64> = (0..expect.len()).map(|_| rng.exp_rand()).collect();
    assert_eq!(got, expect);
}

#[test]
fn pois_matches_r() {
    let expect = read_f64("rng_pois.bin");
    let lambdas = [0.3, 2.5, 12.0, 40.0];
    let mut rng = Rng::seed(0);
    let mut got = Vec::new();
    for &lam in &lambdas {
        for _ in 0..12 {
            got.push(rng.rpois(lam));
        }
    }
    assert_eq!(got, expect);
}

#[test]
fn beta_matches_r() {
    let expect = read_f64("rng_beta.bin");
    let a = [0.2, 0.7, 1.5, 4.0];
    let b = [0.4, 1.1, 2.0, 3.0];
    let mut rng = Rng::seed(0);
    let mut got = Vec::new();
    for i in 0..4 {
        for _ in 0..8 {
            got.push(rng.rbeta(a[i], b[i]));
        }
    }
    assert_eq!(got, expect);
}

#[test]
fn sample_uniform_matches_r() {
    let expect = read_i32("rng_sample9.bin");
    let mut rng = Rng::seed(0);
    assert_eq!(rng.sample_uniform(100_000, 9, false), expect);
    let expect = read_i32("rng_sample_half.bin");
    let mut rng = Rng::seed(0);
    assert_eq!(rng.sample_uniform(110, 55, false), expect);
}

#[test]
fn sample_prob_matches_r() {
    let expect = read_i32("rng_sample01.bin");
    let mut rng = Rng::seed(0);
    let idx = rng.sample_prob(&[0.7, 0.3], 40, true);
    let got: Vec<i32> = idx.iter().map(|&i| i - 1).collect();
    assert_eq!(got, expect);

    let expect = read_i32("rng_sample012.bin");
    let mut rng = Rng::seed(0);
    let idx = rng.sample_prob(&[0.1, 0.5, 0.4], 30, true);
    let got: Vec<i32> = idx.iter().map(|&i| [0, 1, 2][(i - 1) as usize]).collect();
    assert_eq!(got, expect);
}

#[test]
fn walker_matches_r() {
    let probs = read_f64("rng_walker_p.bin");
    let expect = read_i32("rng_walker.bin");
    let mut rng = Rng::seed(2);
    assert_eq!(rng.sample_prob(&probs, 40, true), expect);
}

#[test]
fn matmul_matches_r_openblas() {
    let a = read_f64("mm_A.bin");
    let b = read_f64("mm_B.bin");
    let c = read_f64("mm_C.bin");
    let a = scmultisim::Mat::from_col_major(8, 6, a);
    let b = scmultisim::Mat::from_col_major(11, 6, b);
    let got = scmultisim::matmul_bt(&a, &b);
    assert_eq!(got.data, c);
}
