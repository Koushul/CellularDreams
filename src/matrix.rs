//! Column-major matrices, matching R's storage order.

#[derive(Clone, Debug)]
pub struct Mat {
    pub nrow: usize,
    pub ncol: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(nrow: usize, ncol: usize) -> Self {
        Self {
            nrow,
            ncol,
            data: vec![0.0; nrow * ncol],
        }
    }

    pub fn from_col_major(nrow: usize, ncol: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), nrow * ncol);
        Self { nrow, ncol, data }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r + c * self.nrow]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r + c * self.nrow] = v;
    }

    /// Copy one row into `out` without allocating.
    pub fn copy_row(&self, r: usize, out: &mut [f64]) {
        assert_eq!(out.len(), self.ncol);
        for (c, slot) in out.iter_mut().enumerate() {
            *slot = self.get(r, c);
        }
    }

    pub fn col(&self, c: usize) -> Vec<f64> {
        let start = c * self.nrow;
        self.data[start..start + self.nrow].to_vec()
    }

    pub fn row(&self, r: usize) -> Vec<f64> {
        (0..self.ncol).map(|c| self.get(r, c)).collect()
    }

    pub fn set_row(&mut self, r: usize, values: &[f64]) {
        assert_eq!(values.len(), self.ncol);
        for (c, &v) in values.iter().enumerate() {
            self.set(r, c, v);
        }
    }

    pub fn set_col(&mut self, c: usize, values: &[f64]) {
        assert_eq!(values.len(), self.nrow);
        let start = c * self.nrow;
        self.data[start..start + self.nrow].copy_from_slice(values);
    }

    /// Select rows in the given order.
    pub fn select_rows(&self, rows: &[usize]) -> Mat {
        let mut out = Mat::zeros(rows.len(), self.ncol);
        for (i, &r) in rows.iter().enumerate() {
            for c in 0..self.ncol {
                out.set(i, c, self.get(r, c));
            }
        }
        out
    }

    /// Select columns in the given order.
    pub fn select_cols(&self, cols: &[usize]) -> Mat {
        let mut out = Mat::zeros(self.nrow, cols.len());
        for (j, &c) in cols.iter().enumerate() {
            let src = c * self.nrow;
            let dst = j * self.nrow;
            out.data[dst..dst + self.nrow].copy_from_slice(&self.data[src..src + self.nrow]);
        }
        out
    }

    pub fn hcat(&self, other: &Mat) -> Mat {
        assert_eq!(self.nrow, other.nrow);
        let mut data = self.data.clone();
        data.extend_from_slice(&other.data);
        Mat::from_col_major(self.nrow, self.ncol + other.ncol, data)
    }

    pub fn vcat(&self, other: &Mat) -> Mat {
        assert_eq!(self.ncol, other.ncol);
        let mut out = Mat::zeros(self.nrow + other.nrow, self.ncol);
        for c in 0..self.ncol {
            for r in 0..self.nrow {
                out.set(r, c, self.get(r, c));
            }
            for r in 0..other.nrow {
                out.set(self.nrow + r, c, other.get(r, c));
            }
        }
        out
    }

    pub fn transpose(&self) -> Mat {
        let mut out = Mat::zeros(self.ncol, self.nrow);
        for c in 0..self.ncol {
            for r in 0..self.nrow {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    pub fn col_sums(&self) -> Vec<f64> {
        (0..self.ncol).map(|c| self.col(c).iter().sum()).collect()
    }

    pub fn row_sums(&self) -> Vec<f64> {
        (0..self.nrow)
            .map(|r| (0..self.ncol).map(|c| self.get(r, c)).sum())
            .collect()
    }

    pub fn map(&self, f: impl Fn(f64) -> f64) -> Mat {
        Mat {
            nrow: self.nrow,
            ncol: self.ncol,
            data: self.data.iter().copied().map(f).collect(),
        }
    }
}

#[link(name = "openblas")]
unsafe extern "C" {
    fn openblas_set_num_threads(num: i32);
    fn cblas_dgemm(
        layout: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );
}

const COL_MAJOR: i32 = 102;
const NO_TRANS: i32 = 111;
const TRANS: i32 = 112;

/// Keep `dgemm` on one thread.
///
/// A multi-threaded product can add the same terms in a different order. That
/// changes low bits, which can change density ranks, which then changes how
/// many uniforms the simulator draws. Gene trajectories are parallelized
/// separately, after those uniforms have been drawn.
pub fn blas_single_thread() {
    unsafe { openblas_set_num_threads(1) }
}

/// `A %*% B` with column-major inputs. `A` is m×k, `B` is k×n.
pub fn matmul(a: &Mat, b: &Mat) -> Mat {
    assert_eq!(a.ncol, b.nrow);
    let m = a.nrow;
    let k = a.ncol;
    let n = b.ncol;
    let mut c = Mat::zeros(m, n);
    if m == 0 || n == 0 || k == 0 {
        return c;
    }
    unsafe {
        cblas_dgemm(
            COL_MAJOR,
            NO_TRANS,
            NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.data.as_ptr(),
            m as i32,
            b.data.as_ptr(),
            k as i32,
            0.0,
            c.data.as_mut_ptr(),
            m as i32,
        );
    }
    c
}

/// `A %*% t(B)`. `A` is m×k, `B` is n×k, result is m×n.
pub fn matmul_bt(a: &Mat, b: &Mat) -> Mat {
    let mut c = Mat::zeros(a.nrow, b.nrow);
    matmul_bt_into(a, b, &mut c);
    c
}

/// `A %*% t(B)` written into an existing `c` of shape m×n.
pub fn matmul_bt_into(a: &Mat, b: &Mat, c: &mut Mat) {
    assert_eq!(a.ncol, b.ncol);
    let m = a.nrow;
    let k = a.ncol;
    let n = b.nrow;
    assert_eq!(c.nrow, m);
    assert_eq!(c.ncol, n);
    if m == 0 || n == 0 || k == 0 {
        c.data.fill(0.0);
        return;
    }
    unsafe {
        cblas_dgemm(
            COL_MAJOR,
            NO_TRANS,
            TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.data.as_ptr(),
            m as i32,
            b.data.as_ptr(),
            n as i32,
            0.0,
            c.data.as_mut_ptr(),
            m as i32,
        );
    }
}
