//! R 4.x default generator: Mersenne-Twister, inversion normals, rejection sampling.
//!
//! Distribution samplers follow the same control flow and persistent Poisson table
//! as R's Mathlib so a simulation consumes uniforms in the same order.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UPPER_MASK: u32 = 0x80000000;
const LOWER_MASK: u32 = 0x7fffffff;
const I2_32M1: f64 = 2.3283064365386963e-10;
const M_1_SQRT_2PI: f64 = 0.398942280401432677939946059934;

#[derive(Clone)]
struct PoisCache {
    l: i32,
    m: i32,
    b1: f64,
    b2: f64,
    c: f64,
    c0: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    pp: [f64; 36],
    p0: f64,
    p: f64,
    q: f64,
    s: f64,
    d: f64,
    omega: f64,
    big_l: f64,
    muprev: f64,
    muprev2: f64,
}

impl Default for PoisCache {
    fn default() -> Self {
        Self {
            l: 0,
            m: 0,
            b1: 0.0,
            b2: 0.0,
            c: 0.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
            pp: [0.0; 36],
            p0: 0.0,
            p: 0.0,
            q: 0.0,
            s: 0.0,
            d: 0.0,
            omega: 0.0,
            big_l: 0.0,
            muprev: 0.0,
            muprev2: 0.0,
        }
    }
}

#[derive(Clone)]
pub struct Rng {
    mt: [u32; N],
    mti: usize,
    pois: PoisCache,
}

impl Rng {
    pub fn seed(seed: u32) -> Self {
        let mut s = seed;
        for _ in 0..50 {
            s = s.wrapping_mul(69069).wrapping_add(1);
        }
        let mut i_seed = [0u32; 625];
        for j in 0..625 {
            s = s.wrapping_mul(69069).wrapping_add(1);
            i_seed[j] = s;
        }
        i_seed[0] = 624;
        let mut mt = [0u32; N];
        mt.copy_from_slice(&i_seed[1..625]);
        Self {
            mt,
            mti: 624,
            pois: PoisCache::default(),
        }
    }

    pub fn unif(&mut self) -> f64 {
        let x = self.mt_genrand();
        self.fixup(x)
    }

    fn fixup(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.5 * I2_32M1
        } else if (1.0 - x) <= 0.0 {
            1.0 - 0.5 * I2_32M1
        } else {
            x
        }
    }

    fn mt_genrand(&mut self) -> f64 {
        if self.mti >= N {
            for kk in 0..(N - M) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ ((y & 1) * MATRIX_A);
            }
            for kk in (N - M)..(N - 1) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk - (N - M)] ^ (y >> 1) ^ ((y & 1) * MATRIX_A);
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ ((y & 1) * MATRIX_A);
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        (y as f64) * I2_32M1
    }

    fn rbits(&mut self, bits: i32) -> f64 {
        let mut v: u64 = 0;
        let mut n = 0;
        while n <= bits {
            let v1 = (self.unif() * 65536.0).floor() as u64;
            v = 65536 * v + v1;
            n += 16;
        }
        let mask = if bits >= 63 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        (v & mask) as f64
    }

    fn ceil_log2(dn: f64) -> i32 {
        if dn <= 1.0 {
            return 0;
        }
        let n = dn as u64;
        if n > 0 && (n as f64) == dn {
            return (64 - (n - 1).leading_zeros()) as i32;
        }
        dn.log2().ceil() as i32
    }

    /// Uniform integer in `0..dn` using R's rejection sampler (`sample.kind = "Rejection"`).
    pub fn unif_index(&mut self, dn: f64) -> f64 {
        if dn <= 0.0 {
            return 0.0;
        }
        let bits = Self::ceil_log2(dn);
        loop {
            let dv = self.rbits(bits);
            if dn > dv {
                return dv;
            }
        }
    }

    pub fn runif(&mut self, min: f64, max: f64) -> f64 {
        if min == max {
            return min;
        }
        min + (max - min) * self.unif()
    }

    /// Standard normal via inversion. R builds a 27-bit uniform from two draws.
    pub fn norm_rand(&mut self) -> f64 {
        const BIG: f64 = 134217728.0; // 2^27
        let u1 = self.unif();
        let u = (BIG * u1) as i32 as f64 + self.unif();
        qnorm(u / BIG)
    }

    pub fn rnorm(&mut self, mu: f64, sigma: f64) -> f64 {
        if mu.is_nan() || !sigma.is_finite() || sigma < 0.0 {
            return f64::NAN;
        }
        if sigma == 0.0 || !mu.is_finite() {
            return mu;
        }
        mu + sigma * self.norm_rand()
    }

    pub fn exp_rand(&mut self) -> f64 {
        const Q: [f64; 16] = [
            0.6931471805599453,
            0.9333736875190459,
            0.9888777961838675,
            0.9984959252914960,
            0.9998292811061389,
            0.9999833164100727,
            0.9999985691438767,
            0.9999998906925558,
            0.9999999924734159,
            0.9999999995283275,
            0.9999999999728814,
            0.9999999999985598,
            0.9999999999999289,
            0.9999999999999968,
            0.9999999999999999,
            1.0000000000000000,
        ];
        let mut a = 0.0;
        let mut u = self.unif();
        while u <= 0.0 || u >= 1.0 {
            u = self.unif();
        }
        loop {
            u += u;
            if u > 1.0 {
                break;
            }
            a += Q[0];
        }
        u -= 1.0;
        if u <= Q[0] {
            return a + u;
        }
        let mut i = 0;
        let mut ustar = self.unif();
        let mut umin = ustar;
        loop {
            ustar = self.unif();
            if umin > ustar {
                umin = ustar;
            }
            i += 1;
            if u <= Q[i] {
                break;
            }
        }
        a + umin * Q[0]
    }

    pub fn rpois(&mut self, mu: f64) -> f64 {
        const FACT: [f64; 10] = [1., 1., 2., 6., 24., 120., 720., 5040., 40320., 362880.];
        const A0: f64 = -0.5;
        const A1: f64 = 0.3333333;
        const A2: f64 = -0.2500068;
        const A3: f64 = 0.2000118;
        const A4: f64 = -0.1661269;
        const A5: f64 = 0.1421878;
        const A6: f64 = -0.1384794;
        const A7: f64 = 0.1250060;
        const ONE_7: f64 = 0.1428571428571428571;
        const ONE_12: f64 = 0.0833333333333333333;
        const ONE_24: f64 = 0.0416666666666666667;

        if !mu.is_finite() || mu < 0.0 {
            return f64::NAN;
        }
        if mu <= 0.0 {
            return 0.0;
        }

        let big_mu = mu >= 10.0;
        let mut new_big_mu = false;
        if !(big_mu && mu == self.pois.muprev) {
            if big_mu {
                new_big_mu = true;
                self.pois.muprev = mu;
                self.pois.s = mu.sqrt();
                self.pois.d = 6.0 * mu * mu;
                self.pois.big_l = (mu - 1.1484).floor();
            } else {
                if mu != self.pois.muprev {
                    self.pois.muprev = mu;
                    self.pois.m = 1.max(mu as i32);
                    self.pois.l = 0;
                    let e = (-mu).exp();
                    self.pois.q = e;
                    self.pois.p0 = e;
                    self.pois.p = e;
                }
                loop {
                    let u = self.unif();
                    if u <= self.pois.p0 {
                        return 0.0;
                    }
                    if self.pois.l != 0 {
                        let start = if u <= 0.458 {
                            1
                        } else {
                            self.pois.l.min(self.pois.m)
                        };
                        let mut k = start;
                        while k <= self.pois.l {
                            if u <= self.pois.pp[k as usize] {
                                return k as f64;
                            }
                            k += 1;
                        }
                        if self.pois.l == 35 {
                            continue;
                        }
                    }
                    self.pois.l += 1;
                    let mut k = self.pois.l;
                    while k <= 35 {
                        self.pois.p *= mu / k as f64;
                        self.pois.q += self.pois.p;
                        self.pois.pp[k as usize] = self.pois.q;
                        if u <= self.pois.q {
                            self.pois.l = k;
                            return k as f64;
                        }
                        k += 1;
                    }
                    self.pois.l = 35;
                }
            }
        }

        let s = self.pois.s;
        let d = self.pois.d;
        let big_l = self.pois.big_l;
        let g = mu + s * self.norm_rand();
        let mut pois = -1.0;
        let mut fk;
        let mut difmuk;
        let mut u = 0.0;
        if g >= 0.0 {
            pois = g.floor();
            if pois >= big_l {
                return pois;
            }
            fk = pois;
            difmuk = mu - fk;
            u = self.unif();
            if d * u >= difmuk * difmuk * difmuk {
                return pois;
            }
        } else {
            fk = 0.0;
            difmuk = 0.0;
        }

        if new_big_mu || mu != self.pois.muprev2 {
            self.pois.muprev2 = mu;
            self.pois.omega = M_1_SQRT_2PI / s;
            self.pois.b1 = ONE_24 / mu;
            self.pois.b2 = 0.3 * self.pois.b1 * self.pois.b1;
            self.pois.c3 = ONE_7 * self.pois.b1 * self.pois.b2;
            self.pois.c2 = self.pois.b2 - 15.0 * self.pois.c3;
            self.pois.c1 = self.pois.b1 - 6.0 * self.pois.b2 + 45.0 * self.pois.c3;
            self.pois.c0 = 1.0 - self.pois.b1 + 3.0 * self.pois.b2 - 15.0 * self.pois.c3;
            self.pois.c = 0.1069 / mu;
        }
        let omega = self.pois.omega;
        let c3 = self.pois.c3;
        let c2 = self.pois.c2;
        let c1 = self.pois.c1;
        let c0 = self.pois.c0;
        let c = self.pois.c;

        let mut kflag = if g >= 0.0 { 0 } else { -1 };
        loop {
            if kflag < 0 {
                let e = self.exp_rand();
                u = 2.0 * self.unif() - 1.0;
                let t = 1.8 + fsign(e, u);
                if t <= -0.6744 {
                    continue;
                }
                pois = (mu + s * t).floor();
                fk = pois;
                difmuk = mu - fk;
                kflag = 1;
                if !step_f(
                    self, mu, pois, fk, difmuk, s, omega, c3, c2, c1, c0, c, u, e, kflag, &FACT,
                    A0, A1, A2, A3, A4, A5, A6, A7, ONE_12,
                ) {
                    kflag = -1;
                    continue;
                }
                return pois;
            } else {
                let e = 0.0;
                if step_f(
                    self, mu, pois, fk, difmuk, s, omega, c3, c2, c1, c0, c, u, e, kflag, &FACT,
                    A0, A1, A2, A3, A4, A5, A6, A7, ONE_12,
                ) {
                    return pois;
                }
                kflag = -1;
            }
        }
    }

    pub fn rbeta(&mut self, aa: f64, bb: f64) -> f64 {
        if aa.is_nan() || bb.is_nan() || aa < 0.0 || bb < 0.0 {
            return f64::NAN;
        }
        if !aa.is_finite() && !bb.is_finite() {
            return 0.5;
        }
        if aa == 0.0 && bb == 0.0 {
            return if self.unif() < 0.5 { 0.0 } else { 1.0 };
        }
        if !aa.is_finite() || bb == 0.0 {
            return 1.0;
        }
        if !bb.is_finite() || aa == 0.0 {
            return 0.0;
        }
        let expmax = 1024.0 * std::f64::consts::LN_2;
        let a = aa.min(bb);
        let b = aa.max(bb);
        let alpha = a + b;
        let vw = |u1: f64, beta: f64, scale: f64| -> (f64, f64) {
            let v = beta * (u1 / (1.0 - u1)).ln();
            let w = if v <= expmax {
                let w = scale * v.exp();
                if w.is_finite() {
                    w
                } else {
                    f64::MAX
                }
            } else {
                f64::MAX
            };
            (v, w)
        };
        if a <= 1.0 {
            let beta = 1.0 / a;
            let delta = 1.0 + b - a;
            let k1 = delta * (0.0138889 + 0.0416667 * a) / (b * beta - 0.777778);
            let k2 = 0.25 + (0.5 + 0.25 / delta) * a;
            let w = loop {
                let u1 = self.unif();
                let u2 = self.unif();
                if u1 < 0.5 {
                    let y = u1 * u2;
                    let z = u1 * y;
                    if 0.25 * u2 + z - y >= k1 {
                        continue;
                    }
                    let (v, w) = vw(u1, beta, b);
                    if alpha * ((alpha / (a + w)).ln() + v) - 1.3862944 >= z.ln() {
                        break w;
                    }
                } else {
                    let z = u1 * u1 * u2;
                    if z <= 0.25 {
                        break vw(u1, beta, b).1;
                    }
                    if z >= k2 {
                        continue;
                    }
                    let (v, w) = vw(u1, beta, b);
                    if alpha * ((alpha / (a + w)).ln() + v) - 1.3862944 >= z.ln() {
                        break w;
                    }
                }
            };
            if aa == a {
                a / (a + w)
            } else {
                w / (a + w)
            }
        } else {
            let beta = ((alpha - 2.0) / (2.0 * a * b - alpha)).sqrt();
            let gamma = a + 1.0 / beta;
            let w = loop {
                let u1 = self.unif();
                let u2 = self.unif();
                let (v, w) = vw(u1, beta, a);
                let z = u1 * u1 * u2;
                let r = gamma * v - 1.3862944;
                let s = a + r - w;
                if s + 2.609438 >= 5.0 * z {
                    break w;
                }
                let t = z.ln();
                if s > t {
                    break w;
                }
                if r + alpha * (alpha / (b + w)).ln() >= t {
                    break w;
                }
            };
            if aa != a {
                b / (b + w)
            } else {
                w / (b + w)
            }
        }
    }

    /// `sample.int(n, k, replace)` with equal probabilities. Results are 1-based.
    pub fn sample_uniform(&mut self, n: i32, k: i32, replace: bool) -> Vec<i32> {
        if n < 0 || k < 0 || (!replace && k > n) {
            panic!("invalid sample");
        }
        if replace || k < 2 {
            return (0..k)
                .map(|_| self.unif_index(n as f64) as i32 + 1)
                .collect();
        }
        let mut x: Vec<i32> = (0..n).collect();
        let mut nn = n;
        let mut out = Vec::with_capacity(k as usize);
        for _ in 0..k {
            let j = self.unif_index(nn as f64) as usize;
            out.push(x[j] + 1);
            nn -= 1;
            x[j] = x[nn as usize];
        }
        out
    }

    /// Unequal-probability sample. `probs` need not be normalized. Results are 1-based.
    pub fn sample_prob(&mut self, probs: &[f64], k: usize, replace: bool) -> Vec<i32> {
        let n = probs.len();
        if n == 0 || (!replace && k > n) {
            panic!("invalid probability sample");
        }
        let mut p = probs.to_vec();
        let mut sum = 0.0;
        let mut npos = 0;
        for &v in &p {
            if !v.is_finite() || v < 0.0 {
                panic!("invalid probability");
            }
            if v > 0.0 {
                npos += 1;
                sum += v;
            }
        }
        if npos == 0 || (!replace && k > npos) {
            panic!("too few positive probabilities");
        }
        for v in &mut p {
            *v /= sum;
        }
        if replace {
            let mut nc = 0;
            for &v in &p {
                if n as f64 * v > 0.1 {
                    nc += 1;
                }
            }
            if nc > 200 {
                self.walker_sample(&p, k)
            } else {
                self.prob_replace(&mut p, k)
            }
        } else {
            self.prob_noreplace(&mut p, k)
        }
    }

    fn prob_replace(&mut self, p: &mut [f64], k: usize) -> Vec<i32> {
        let n = p.len();
        let mut perm: Vec<i32> = (1..=n as i32).collect();
        revsort(p, &mut perm);
        for i in 1..n {
            p[i] += p[i - 1];
        }
        let mut ans = vec![0; k];
        let nm1 = n - 1;
        for i in 0..k {
            let ru = self.unif();
            let mut j = 0;
            while j < nm1 {
                if ru <= p[j] {
                    break;
                }
                j += 1;
            }
            ans[i] = perm[j];
        }
        ans
    }

    fn prob_noreplace(&mut self, p: &mut [f64], k: usize) -> Vec<i32> {
        let n = p.len();
        let mut perm: Vec<i32> = (1..=n as i32).collect();
        revsort(p, &mut perm);
        let mut total = 1.0;
        let mut ans = vec![0; k];
        let mut n1 = n - 1;
        for i in 0..k {
            let rt = total * self.unif();
            let mut mass = 0.0;
            let mut j = 0;
            while j < n1 {
                mass += p[j];
                if rt <= mass {
                    break;
                }
                j += 1;
            }
            if j == n1 {
                j = n1 - 1;
            }
            ans[i] = perm[j];
            total -= p[j];
            let mut t = j;
            while t < n1 {
                p[t] = p[t + 1];
                perm[t] = perm[t + 1];
                t += 1;
            }
            n1 -= 1;
        }
        ans
    }

    fn walker_sample(&mut self, p: &[f64], k: usize) -> Vec<i32> {
        let n = p.len();
        let mut q = vec![0.0; n];
        let mut hl = vec![0i32; n];
        let mut a = vec![0i32; n];
        let mut h: i32 = 0;
        let mut l: i32 = n as i32;
        for i in 0..n {
            q[i] = p[i] * n as f64;
            if q[i] < 1.0 {
                hl[h as usize] = i as i32;
                h += 1;
            } else {
                l -= 1;
                hl[l as usize] = i as i32;
            }
        }
        if h > 0 && l < n as i32 {
            for kk in 0..(n - 1) {
                let i = hl[kk] as usize;
                let j = hl[l as usize] as usize;
                a[i] = j as i32;
                q[j] += q[i] - 1.0;
                if q[j] < 1.0 {
                    l += 1;
                }
                if l >= n as i32 {
                    break;
                }
            }
        }
        for i in 0..n {
            q[i] += i as f64;
        }
        let mut ans = vec![0; k];
        for i in 0..k {
            let ki = self.unif_index(n as f64) as usize;
            let ru = ki as f64 + self.unif();
            ans[i] = if ru < q[ki] { ki as i32 + 1 } else { a[ki] + 1 };
        }
        ans
    }
}

fn fsign(x: f64, y: f64) -> f64 {
    if y >= 0.0 {
        x.abs()
    } else {
        -x.abs()
    }
}

fn step_f(
    _rng: &mut Rng,
    mu: f64,
    pois: f64,
    fk: f64,
    difmuk: f64,
    s: f64,
    omega: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
    c: f64,
    u: f64,
    e: f64,
    kflag: i32,
    fact: &[f64; 10],
    a0: f64,
    a1: f64,
    a2: f64,
    a3: f64,
    a4: f64,
    a5: f64,
    a6: f64,
    a7: f64,
    one_12: f64,
) -> bool {
    let (px, py) = if pois < 10.0 {
        let px = -mu;
        let py = mu.powf(pois) / fact[pois as usize];
        (px, py)
    } else {
        let mut del = one_12 / fk;
        del *= 1.0 - 4.8 * del * del;
        let v = difmuk / fk;
        let px = if v.abs() <= 0.25 {
            fk * v
                * v
                * (((((((a7 * v + a6) * v + a5) * v + a4) * v + a3) * v + a2) * v + a1) * v + a0)
                - del
        } else {
            fk * (1.0 + v).ln() - difmuk - del
        };
        let py = M_1_SQRT_2PI / fk.sqrt();
        (px, py)
    };
    let mut x = (0.5 - difmuk) / s;
    x *= x;
    let fx = -0.5 * x;
    let fy = omega * (((c3 * x + c2) * x + c1) * x + c0);
    if kflag > 0 {
        c * u.abs() <= py * (px + e).exp() - fy * (fx + e).exp()
    } else {
        fy - u * fy <= py * (px - fx).exp()
    }
}

/// Heapsort `a` into descending order, permuting `ib` the same way (R `revsort`).
fn revsort(a: &mut [f64], ib: &mut [i32]) {
    let n = a.len() as i32;
    if n <= 1 {
        return;
    }
    let mut l = (n >> 1) + 1;
    let mut ir = n;
    loop {
        let (ra, ii) = if l > 1 {
            l -= 1;
            (a[(l - 1) as usize], ib[(l - 1) as usize])
        } else {
            let ra = a[(ir - 1) as usize];
            let ii = ib[(ir - 1) as usize];
            a[(ir - 1) as usize] = a[0];
            ib[(ir - 1) as usize] = ib[0];
            ir -= 1;
            if ir == 1 {
                a[0] = ra;
                ib[0] = ii;
                return;
            }
            (ra, ii)
        };
        let mut i = l;
        let mut j = l << 1;
        while j <= ir {
            if j < ir && a[(j - 1) as usize] > a[j as usize] {
                j += 1;
            }
            if ra > a[(j - 1) as usize] {
                a[(i - 1) as usize] = a[(j - 1) as usize];
                ib[(i - 1) as usize] = ib[(j - 1) as usize];
                i = j;
                j += i;
            } else {
                j = ir + 1;
            }
        }
        a[(i - 1) as usize] = ra;
        ib[(i - 1) as usize] = ii;
    }
}

pub fn qnorm(p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return q
            * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r
                + 67265.770927008700853)
                * r
                + 45921.953931549871457)
                * r
                + 13731.693765509461125)
                * r
                + 1971.5909503065514427)
                * r
                + 133.14166789178437745)
                * r
                + 3.387132872796366608)
            / (((((((r * 5226.495278852854561 + 28729.085735721942674) * r
                + 39307.89580009271061)
                * r
                + 21213.794301586595867)
                * r
                + 5394.1960214247511077)
                * r
                + 687.1870074920579083)
                * r
                + 42.313330701600911252)
                * r
                + 1.0);
    }
    let lp = if q > 0.0 { (1.0 - p).ln() } else { p.ln() };
    let mut r = (-lp).sqrt();
    let mut val = if r <= 5.0 {
        r += -1.6;
        (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r
            + 0.24178072517745061177)
            * r
            + 1.27045825245236838258)
            * r
            + 3.64784832476320460504)
            * r
            + 5.7694972214606914055)
            * r
            + 4.6303378461565452959)
            * r
            + 1.42343711074968357734)
            / (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r
                + 0.0151986665636164571966)
                * r
                + 0.14810397642748007459)
                * r
                + 0.68976733498510000455)
                * r
                + 1.6763848301838038494)
                * r
                + 2.05319162663775882187)
                * r
                + 1.0)
    } else if r <= 27.0 {
        r += -5.0;
        (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r
            + 0.0012426609473880784386)
            * r
            + 0.026532189526576123093)
            * r
            + 0.29656057182850489123)
            * r
            + 1.7848265399172913358)
            * r
            + 5.4637849111641143699)
            * r
            + 6.6579046435011037772)
            / (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r
                + 1.8463183175100546818e-5)
                * r
                + 7.868691311456132591e-4)
                * r
                + 0.0148753612908506148525)
                * r
                + 0.13692988092273580531)
                * r
                + 0.59983220655588793769)
                * r
                + 1.0)
    } else {
        r * std::f64::consts::SQRT_2
    };
    if q < 0.0 {
        val = -val;
    }
    val
}

pub fn seq_by(from: f64, to: f64, by: f64) -> Vec<f64> {
    let del = to - from;
    if del == 0.0 && to == 0.0 {
        return vec![to];
    }
    let n = del / by;
    let nn = (n + 1e-10) as i32;
    let mut ra = Vec::with_capacity((nn + 1) as usize);
    for i in 0..=nn {
        ra.push(from + (i as f64) * by);
    }
    if nn > 0 {
        let last = *ra.last().unwrap();
        if (by > 0.0 && last > to) || (by < 0.0 && last < to) {
            *ra.last_mut().unwrap() = to;
        }
    }
    ra
}

/// Average-tie ranks, 1-based, matching `rank(x, ties.method = "average")`.
pub fn rank_average(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_unstable_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1) + (j + 1)) as f64 / 2.0;
        for k in i..=j {
            ranks[idx[k]] = avg;
        }
        i = j + 1;
    }
    ranks
}

pub fn r_index(rank: f64) -> usize {
    let i = rank.trunc() as isize;
    if i < 1 {
        0
    } else {
        (i as usize) - 1
    }
}
