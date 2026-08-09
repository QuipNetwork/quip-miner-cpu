//! Site tensors and gates for the BP-TNS kernel.
//!
//! One tensor per node with a physical index of size 2 (index 0 is spin +1,
//! index 1 is spin -1) and one bond index per incident coupling. The layout
//! is row-major with the physical index slowest and bonds in adjacency
//! order. Each bond carries a Schmidt-weight vector `lambda`, and truncation
//! after a two-site gate uses those weights as the environment, which is the
//! simple-update scheme; Tindall and Fishman showed it equals gauging the
//! network with belief propagation, which is why this file has no separate
//! BP pass of its own.

use crate::mps::svd::{jacobi_svd, truncated_svd, JACOBI_MAX_SWEEPS};

/// Division guard for un-absorbing Schmidt weights. A weight this small
/// marks a dead subspace; dividing by the floor instead of the true value
/// leaves the entry finite and negligible.
const LAMBDA_FLOOR: f64 = 1e-12;

/// Relative floor below which a singular value is not kept.
const KEEP_CUTOFF: f64 = 1e-14;

/// One site tensor.
#[derive(Debug, Clone)]
pub(crate) struct SiteTensor {
    /// Physical dimension: 2, or 1 after a sampling projection.
    pub(crate) phys: usize,
    /// Bond dimensions in adjacency order.
    pub(crate) dims: Vec<usize>,
    /// `phys * prod(dims)` entries, physical index slowest, bonds row-major.
    pub(crate) data: Vec<f64>,
}

impl SiteTensor {
    /// The `|+>` state with every bond at dimension 1.
    pub(crate) fn plus(degree: usize) -> Self {
        let amp = std::f64::consts::FRAC_1_SQRT_2;
        Self {
            phys: 2,
            dims: vec![1; degree],
            data: vec![amp, amp],
        }
    }

    /// Product of the bond dimensions.
    pub(crate) fn bond_total(&self) -> usize {
        self.dims.iter().product()
    }

    /// Stride of bond `b` in the flat bond index.
    fn stride(&self, b: usize) -> usize {
        self.dims[b + 1..].iter().product()
    }

    /// Multiply the amplitude of each physical value by a per-value weight.
    /// `w` has one entry per physical index.
    pub(crate) fn scale_phys(&mut self, w: &[f64]) {
        let bt = self.bond_total();
        for (s, &ws) in w.iter().enumerate().take(self.phys) {
            for x in &mut self.data[s * bt..(s + 1) * bt] {
                *x *= ws;
            }
        }
    }

    /// Apply the 2x2 matrix `g` to the physical index: `out[s] = sum_t
    /// g[s][t] in[t]`. Only meaningful while `phys == 2`.
    pub(crate) fn apply_phys_matrix(&mut self, g: &[[f64; 2]; 2]) {
        let bt = self.bond_total();
        for k in 0..bt {
            let a = self.data[k];
            let b = self.data[bt + k];
            self.data[k] = g[0][0] * a + g[0][1] * b;
            self.data[bt + k] = g[1][0] * a + g[1][1] * b;
        }
    }

    /// Multiply along bond `b` by a per-coordinate weight.
    pub(crate) fn scale_bond(&mut self, b: usize, w: &[f64]) {
        let k = self.dims[b];
        let stride = self.stride(b);
        let bt = self.bond_total();
        for s in 0..self.phys {
            for (flat, x) in self.data[s * bt..(s + 1) * bt].iter_mut().enumerate() {
                *x *= w[(flat / stride) % k];
            }
        }
    }

    /// Reshape into a row-major matrix with bond `b` as the trailing axis.
    /// Rows enumerate `(phys, other bonds)`, columns enumerate bond `b`.
    pub(crate) fn matricize(&self, b: usize) -> (Vec<f64>, usize, usize) {
        let k = self.dims[b];
        let stride = self.stride(b);
        let bt = self.bond_total();
        let rest = bt / k;
        let rows = self.phys * rest;
        let mut m = vec![0.0; rows * k];
        for s in 0..self.phys {
            for flat in 0..bt {
                let c = (flat / stride) % k;
                let r = (flat / (stride * k)) * stride + flat % stride;
                m[(s * rest + r) * k + c] = self.data[s * bt + flat];
            }
        }
        (m, rows, k)
    }

    /// Inverse of [`Self::matricize`]: install `m` as the new data with bond
    /// `b` resized to `k_new`.
    pub(crate) fn dematricize(&mut self, b: usize, m: &[f64], k_new: usize) {
        self.dims[b] = k_new;
        let stride = self.stride(b);
        let bt = self.bond_total();
        let rest = bt / k_new;
        let mut data = vec![0.0; self.phys * bt];
        for s in 0..self.phys {
            for flat in 0..bt {
                let c = (flat / stride) % k_new;
                let r = (flat / (stride * k_new)) * stride + flat % stride;
                data[s * bt + flat] = m[(s * rest + r) * k_new + c];
            }
        }
        self.data = data;
    }

    /// Rescale so the largest magnitude is 1, and scrub non-finite entries.
    /// A tensor that lost every entry resets to a uniform state rather than
    /// propagating zeros: the kernel rejects nothing the harness accepted.
    pub(crate) fn renormalize(&mut self) {
        for x in &mut self.data {
            if !x.is_finite() {
                *x = 0.0;
            }
        }
        let max = self.data.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
        if max > 0.0 {
            for x in &mut self.data {
                *x /= max;
            }
        } else {
            self.data.fill(1.0);
        }
    }
}

/// Apply the imaginary-time ZZ gate `exp(-eta j z_u z_v)` (up to scale) on
/// the bond `e` joining sites `u` and `v`, truncating the bond to at most
/// `chi` with the simple-update environment.
///
/// `bu` and `bv` are the positions of bond `e` in each site's adjacency
/// list, `t = tanh(eta * j)` is the gate parameter, `lam` is the bond's
/// Schmidt weights, and `env_u`/`env_v` give the weights of every *other*
/// bond of each site, in adjacency order, with the entry for `e` empty.
#[expect(
    clippy::too_many_arguments,
    reason = "the gate touches two sites, their shared bond, and both environments; bundling them into a struct would be built for one caller"
)]
pub(crate) fn apply_zz(
    tu: &mut SiteTensor,
    tv: &mut SiteTensor,
    bu: usize,
    bv: usize,
    t: f64,
    lam: &mut Vec<f64>,
    env_u: &[&[f64]],
    env_v: &[&[f64]],
    chi: usize,
) {
    // Absorb the environment weights so the SVD sees the BP gauge.
    for (b, w) in env_u.iter().enumerate() {
        if b != bu && !w.is_empty() {
            tu.scale_bond(b, w);
        }
    }
    for (b, w) in env_v.iter().enumerate() {
        if b != bv && !w.is_empty() {
            tv.scale_bond(b, w);
        }
    }

    let (mu, rows_u, k) = tu.matricize(bu);
    let (mv, rows_v, _) = tv.matricize(bv);
    let ru = rows_u / tu.phys;
    let rv = rows_v / tv.phys;

    // theta[(su, ru), (sv, rv)] = sum_c mu[.., c] lam[c] mv[.., c] g(su, sv)
    // with g(z, z') = 1 - t z z': the gate is diagonal in the physical basis.
    let mut theta = vec![0.0; rows_u * rows_v];
    for su in 0..tu.phys {
        for iu in 0..ru {
            let row = su * ru + iu;
            for sv in 0..tv.phys {
                let zz = if su == sv { 1.0 - t } else { 1.0 + t };
                for iv in 0..rv {
                    let col = sv * rv + iv;
                    let mut acc = 0.0;
                    for (c, &l) in lam.iter().enumerate().take(k) {
                        acc += mu[row * k + c] * l * mv[col * k + c];
                    }
                    theta[row * rows_v + col] = acc * zz;
                }
            }
        }
    }

    let cap = chi.max(1);
    let (new_u, new_lam, new_v, keep) =
        match jacobi_svd(&theta, rows_u, rows_v, JACOBI_MAX_SWEEPS) {
            Some(svd) => {
                let s0 = svd.s.first().copied().unwrap_or(0.0);
                let floor = if s0 > 0.0 { KEEP_CUTOFF * s0 } else { 0.0 };
                let counted = svd.s.iter().filter(|&&x| x > floor).count();
                let keep = counted.clamp(1, svd.k.min(cap));
                let mut u = vec![0.0; rows_u * keep];
                for r in 0..rows_u {
                    u[r * keep..(r + 1) * keep]
                        .copy_from_slice(&svd.u[r * svd.k..r * svd.k + keep]);
                }
                let mut v = vec![0.0; rows_v * keep];
                for c in 0..rows_v {
                    for b in 0..keep {
                        v[c * keep + b] = svd.vt[b * rows_v + c];
                    }
                }
                let lam: Vec<f64> = if s0 > 0.0 {
                    svd.s[..keep].iter().map(|&x| x / s0).collect()
                } else {
                    vec![1.0; keep]
                };
                (u, lam, v, keep)
            }
            None => {
                // Jacobi did not converge: fall back to the crate's QR path,
                // which keeps an isometry on the `u` side and folds the
                // weights into `v`. The bond weights become uniform.
                let f = truncated_svd(&theta, rows_u, rows_v, cap);
                let keep = f.k.max(1);
                let mut v = vec![0.0; rows_v * keep];
                for b in 0..f.k {
                    for c in 0..rows_v {
                        v[c * keep + b] = f.carry[b * rows_v + c];
                    }
                }
                (f.q, vec![1.0; keep], v, keep)
            }
        };

    tu.dematricize(bu, &new_u, keep);
    tv.dematricize(bv, &new_v, keep);
    *lam = new_lam;

    // Un-absorb the environment weights and keep the numbers bounded.
    for (b, w) in env_u.iter().enumerate() {
        if b != bu && !w.is_empty() {
            let inv: Vec<f64> = w.iter().map(|&x| 1.0 / x.max(LAMBDA_FLOOR)).collect();
            tu.scale_bond(b, &inv);
        }
    }
    for (b, w) in env_v.iter().enumerate() {
        if b != bv && !w.is_empty() {
            let inv: Vec<f64> = w.iter().map(|&x| 1.0 / x.max(LAMBDA_FLOOR)).collect();
            tv.scale_bond(b, &inv);
        }
    }
    tu.renormalize();
    tv.renormalize();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matricize_round_trips_for_every_bond() {
        let t = SiteTensor {
            phys: 2,
            dims: vec![2, 3, 2],
            data: (0..24).map(f64::from).collect(),
        };
        for b in 0..3 {
            let mut probe = t.clone();
            let (m, rows, k) = probe.matricize(b);
            assert_eq!(rows * k, 24);
            probe.dematricize(b, &m, k);
            assert_eq!(probe.data, t.data, "bond {b}");
            assert_eq!(probe.dims, t.dims);
        }
    }

    #[test]
    fn scale_bond_multiplies_exactly_the_matching_coordinate() {
        let mut t = SiteTensor {
            phys: 1,
            dims: vec![2, 2],
            data: vec![1.0, 1.0, 1.0, 1.0],
        };
        t.scale_bond(0, &[2.0, 3.0]);
        assert_eq!(t.data, vec![2.0, 2.0, 3.0, 3.0]);
        t.scale_bond(1, &[5.0, 7.0]);
        assert_eq!(t.data, vec![10.0, 14.0, 15.0, 21.0]);
    }

    #[test]
    fn renormalize_scrubs_non_finite_and_never_leaves_all_zero() {
        let mut t = SiteTensor {
            phys: 2,
            dims: vec![],
            data: vec![f64::NAN, 4.0],
        };
        t.renormalize();
        assert_eq!(t.data, vec![0.0, 1.0]);
        let mut dead = SiteTensor {
            phys: 2,
            dims: vec![],
            data: vec![0.0, f64::INFINITY],
        };
        dead.renormalize();
        assert_eq!(dead.data, vec![1.0, 1.0]);
    }

    /// One ZZ gate on a two-site network must reproduce exact two-spin
    /// imaginary-time arithmetic: amplitudes proportional to the gate weight
    /// applied to `|+>|+>`.
    #[test]
    fn a_single_gate_on_two_sites_matches_exact_two_spin_evolution() {
        let mut tu = SiteTensor::plus(1);
        let mut tv = SiteTensor::plus(1);
        let mut lam = vec![1.0];
        let t = 0.6f64;
        apply_zz(&mut tu, &mut tv, 0, 0, t, &mut lam, &[&[]], &[&[]], 4);

        // Reconstruct amplitude(su, sv) = sum_c tu[su, c] lam[c] tv[sv, c].
        let k = lam.len();
        let mut amp = [[0.0f64; 2]; 2];
        for (su, row) in amp.iter_mut().enumerate() {
            for (sv, a) in row.iter_mut().enumerate() {
                for (c, &l) in lam.iter().enumerate() {
                    *a += tu.data[su * k + c] * l * tv.data[sv * k + c];
                }
            }
        }
        // Exact: amplitude proportional to g(z, z') = 1 -+ t.
        let same = amp[0][0];
        let diff = amp[0][1];
        assert!((amp[1][1] - same).abs() < 1e-12);
        assert!((amp[1][0] - diff).abs() < 1e-12);
        assert!(
            (diff / same - (1.0 + t) / (1.0 - t)).abs() < 1e-10,
            "ratio {} want {}",
            diff / same,
            (1.0 + t) / (1.0 - t)
        );
    }

    #[test]
    fn the_gate_grows_the_bond_no_further_than_chi() {
        let mut tu = SiteTensor::plus(2);
        let mut tv = SiteTensor::plus(2);
        let mut lam = vec![1.0];
        apply_zz(
            &mut tu,
            &mut tv,
            0,
            0,
            0.3,
            &mut lam,
            &[&[], &[1.0]],
            &[&[], &[1.0]],
            1,
        );
        assert_eq!(lam.len(), 1);
        assert_eq!(tu.dims[0], 1);
        assert_eq!(tv.dims[0], 1);
    }

    #[test]
    fn field_and_transverse_gates_act_on_the_physical_index() {
        let mut t = SiteTensor::plus(0);
        t.scale_phys(&[2.0, 0.5]);
        let ratio = t.data[0] / t.data[1];
        assert!((ratio - 4.0).abs() < 1e-12);
        // The transverse mixer with t = 1 symmetrizes any state.
        t.apply_phys_matrix(&[[1.0, 1.0], [1.0, 1.0]]);
        assert!((t.data[0] - t.data[1]).abs() < 1e-12);
    }
}
