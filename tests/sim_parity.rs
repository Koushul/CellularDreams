use scmultisim::{grn_params_100, phyla5, sim_true_counts, Options, Rng};

fn read_f64(name: &str) -> Vec<f64> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name);
    let bytes = std::fs::read(path).unwrap();
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn stream_after_phyla_matches_sample_and_normals() {
    let mut rng = Rng::seed(0);
    let _ = phyla5(&mut rng);
    let s = rng.sample_uniform(100_000, 9, false);
    assert_eq!(
        s,
        vec![13218, 61911, 41230, 79035, 45025, 50453, 90442, 61257, 42709]
    );
    let norms: Vec<f64> = (0..4).map(|_| rng.rnorm(0.0, 1.0)).collect();
    let expect = [
        -1.1476570092363514,
        -0.28946157368822334,
        -0.29921511789731614,
        -0.41151083279506701,
    ];
    assert_eq!(norms, expect);
}

#[test]
fn phyla5_consumes_the_r_stream() {
    let expect = read_f64("after_phyla5_unif.bin");
    let mut rng = Rng::seed(0);
    let _ = phyla5(&mut rng);
    let got: Vec<f64> = (0..expect.len()).map(|_| rng.unif()).collect();
    assert_eq!(got, expect);
}

#[test]
fn phyla3_consumes_the_r_stream() {
    let expect = read_f64("after_phyla3_unif.bin");
    let mut rng = Rng::seed(0);
    let _ = scmultisim::phyla3(&mut rng);
    let got: Vec<f64> = (0..expect.len()).map(|_| rng.unif()).collect();
    assert_eq!(got, expect);
}

#[test]
fn beta_poisson_counts_match_r() {
    let mut rng = Rng::seed(0);
    let tree = phyla5(&mut rng);
    let mut opt = Options::new(grn_params_100(), 100, 20, tree);
    opt.diff_cif_fraction = 0.8;
    opt.do_velocity = false;
    let res = sim_true_counts(&opt, &mut rng).unwrap();
    let neutral = read_f64("beta100_neutral.bin");
    assert_eq!(
        res.neutral.nrow * res.neutral.ncol,
        neutral.len(),
        "neutral len rust {} r {}",
        res.neutral.data.len(),
        neutral.len()
    );
    assert_eq!(res.neutral.data, neutral);
    let expect = read_f64("beta100_counts.bin");
    assert_eq!(res.counts.nrow, 110);
    assert_eq!(res.counts.ncol, 100);
    assert_close(&res.counts.data, &expect);
    let atac = read_f64("beta100_atac.bin");
    assert_close(&res.atac_counts.data, &atac);
    let neutral = read_f64("beta100_neutral.bin");
    assert_eq!(res.neutral.data, neutral);
}

#[test]
fn kinetic_velocity_matches_r() {
    let mut rng = Rng::seed(0);
    let tree = phyla5(&mut rng);
    let mut opt = Options::new(grn_params_100(), 100, 20, tree);
    opt.diff_cif_fraction = 0.8;
    opt.do_velocity = true;
    let res = sim_true_counts(&opt, &mut rng).unwrap();
    let again = sim_true_counts(&opt, &mut {
        let mut rng = Rng::seed(0);
        let _ = phyla5(&mut rng);
        rng
    })
    .unwrap();
    assert_eq!(res.counts.data, again.counts.data);
    assert_eq!(
        res.unspliced_counts.as_ref().unwrap().data,
        again.unspliced_counts.unwrap().data
    );

    let counts_r = read_f64("velo100_counts.bin");
    let unspliced_r = read_f64("velo100_unspliced.bin");
    assert_eq!(res.counts.data.len(), counts_r.len());
    assert_eq!(
        res.unspliced_counts.as_ref().unwrap().data.len(),
        unspliced_r.len()
    );
    let count_corr = gene_mean_pearson(&res.counts.data, &counts_r, res.counts.nrow);
    let unspliced_corr = gene_mean_pearson(
        &res.unspliced_counts.as_ref().unwrap().data,
        &unspliced_r,
        res.unspliced_counts.as_ref().unwrap().nrow,
    );
    assert!(
        count_corr > 0.85,
        "spliced gene-mean correlation with the R fixture was {count_corr}"
    );
    assert!(
        unspliced_corr > 0.85,
        "unspliced gene-mean correlation with the R fixture was {unspliced_corr}"
    );
    assert!(res.counts.data.iter().all(|v| v.is_finite() && *v >= 0.0));
    assert!(res
        .atac_counts
        .data
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0));
}

fn gene_mean_pearson(got: &[f64], expect: &[f64], nrow: usize) -> f64 {
    let ncol = got.len() / nrow;
    let mut gx = vec![0.0; nrow];
    let mut gy = vec![0.0; nrow];
    for r in 0..nrow {
        let mut sx = 0.0;
        let mut sy = 0.0;
        for c in 0..ncol {
            sx += got[r + c * nrow];
            sy += expect[r + c * nrow];
        }
        gx[r] = sx / ncol as f64;
        gy[r] = sy / ncol as f64;
    }
    let mx = gx.iter().sum::<f64>() / nrow as f64;
    let my = gy.iter().sum::<f64>() / nrow as f64;
    let mut num = 0.0;
    let mut dx = 0.0;
    let mut dy = 0.0;
    for r in 0..nrow {
        let a = gx[r] - mx;
        let b = gy[r] - my;
        num += a * b;
        dx += a * a;
        dy += b * b;
    }
    num / (dx.sqrt() * dy.sqrt())
}

fn assert_close(got: &[f64], expect: &[f64]) {
    assert_eq!(got.len(), expect.len());
    let mut worst = 0.0f64;
    let mut n = 0usize;
    for (g, e) in got.iter().zip(expect.iter()) {
        let d = (g - e).abs();
        worst = worst.max(d);
        if d > 1e-6 && (e.abs() < 1.0 || d / e.abs() > 1e-8) {
            n += 1;
        }
    }
    assert!(
        n == 0,
        "{n} values differ from the R fixture, worst absolute error {worst}"
    );
}
