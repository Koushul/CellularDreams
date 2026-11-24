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
    assert_eq!(s, vec![13218, 61911, 41230, 79035, 45025, 50453, 90442, 61257, 42709]);
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
    assert_eq!(res.neutral.nrow * res.neutral.ncol, neutral.len(), "neutral len rust {} r {}", res.neutral.data.len(), neutral.len());
    assert_eq!(res.neutral.data, neutral);
    let expect = read_f64("beta100_counts.bin");
    assert_eq!(res.counts.nrow, 110);
    assert_eq!(res.counts.ncol, 100);
    if res.counts.data != expect {
        let mut shown = 0;
        for (i, (g, e)) in res.counts.data.iter().zip(expect.iter()).enumerate() {
            if g != e {
                eprintln!("mismatch at {i}: rust={g} r={e}");
                shown += 1;
                if shown == 8 {
                    break;
                }
            }
        }
        panic!("counts diverged from R");
    }
    let atac = read_f64("beta100_atac.bin");
    assert_eq!(res.atac_counts.data, atac);
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
    assert_eq!(res.counts.data, read_f64("velo100_counts.bin"));
    assert_eq!(
        res.unspliced_counts.unwrap().data,
        read_f64("velo100_unspliced.bin")
    );
    assert_eq!(res.velocity.unwrap().data, read_f64("velo100_velocity.bin"));
    assert_eq!(res.atac_counts.data, read_f64("velo100_atac.bin"));
}
