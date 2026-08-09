//! The matrix product state, its gates, its canonical form, and its sampler.
//!
//! Layout: site tensor `k` has shape `bond[k] x 2 x bond[k+1]`, flattened
//! row-major, so element `(l, p, r)` is at `(l * 2 + p) * bond[k+1] + r`.
//! Physical index `0` is spin `+1` and index `1` is spin `-1`, which makes the
//! `Z` eigenvalues `[1.0, -1.0]`.
//!
//! Read as a matrix, site `k` is `bond[k]` rows by `2 * bond[k+1]` columns with
//! column index `p * bond[k+1] + r`; the flattening above is exactly that
//! row-major matrix, which is what lets the gates hand a site straight to the
//! factorization without a copy.

use super::svd::{transpose, truncated_svd};
use rand::rngs::SmallRng;
use rand::Rng;

/// Matrix product state in the site-major layout.
pub(crate) struct Mps {
    /// Flattened tensors, one per site.
    pub(crate) site: Vec<Vec<f64>>,
    /// Bond dimensions, length `n + 1`, with `bond[0] == bond[n] == 1`.
    pub(crate) bond: Vec<usize>,
}

/// Scale `v` to unit Frobenius norm. A zero or non-finite norm leaves it alone,
/// which keeps a state that a gate annihilated finite instead of full of NaN.
pub(crate) fn normalize_vec(v: &mut [f64]) {
    let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if nrm > 0.0 && nrm.is_finite() {
        for x in v {
            *x /= nrm;
        }
    }
}

impl Mps {
    /// `|+>^n`, the `Gamma -> infinity` ground state. Bond dimension 1
    /// everywhere, so it costs `2n` numbers.
    pub(crate) fn plus_state(n: usize) -> Self {
        Self {
            site: vec![vec![std::f64::consts::FRAC_1_SQRT_2; 2]; n],
            bond: vec![1; n + 1],
        }
    }

    /// Chain length.
    pub(crate) fn num_sites(&self) -> usize {
        self.site.len()
    }

    /// Largest bond dimension currently in the chain.
    pub(crate) fn max_bond(&self) -> usize {
        self.bond.iter().copied().max().unwrap_or(1)
    }

    /// Expand to the full `2^n` amplitude vector, site `0` most significant.
    ///
    /// Test oracle only: the cost is exponential in `n`, so nothing in the
    /// production path may call it, and it is compiled out of a release build.
    #[cfg(test)]
    pub(crate) fn to_dense(&self) -> Vec<f64> {
        let mut acc = vec![1.0f64];
        for k in 0..self.num_sites() {
            let dl = self.bond[k];
            let dr = self.bond[k + 1];
            if dl == 0 || acc.len() % dl != 0 {
                return acc;
            }
            let configs = acc.len() / dl;
            let mut next = vec![0.0; configs * 2 * dr];
            for c in 0..configs {
                for l in 0..dl {
                    let av = acc[c * dl + l];
                    if av == 0.0 {
                        continue;
                    }
                    for p in 0..2 {
                        for r in 0..dr {
                            next[(c * 2 + p) * dr + r] +=
                                av * self.site[k][(l * 2 + p) * dr + r];
                        }
                    }
                }
            }
            acc = next;
        }
        acc
    }

    /// Scale site `k` to unit Frobenius norm.
    fn normalize_site(&mut self, k: usize) {
        normalize_vec(&mut self.site[k]);
    }

    /// `(I + tau X)` on every site, the normalized form of
    /// `exp(+eta Gamma X / 2)` with `tau = tanh(eta Gamma / 2)`.
    ///
    /// Single-site, so no bond grows and the cost is `O(n chi^2 d)`, which is
    /// free next to the coupling layer.
    pub(crate) fn apply_transverse(&mut self, tau: f64) {
        if tau == 0.0 || !tau.is_finite() {
            return;
        }
        for k in 0..self.num_sites() {
            let dl = self.bond[k];
            let dr = self.bond[k + 1];
            {
                let a = &mut self.site[k];
                for l in 0..dl {
                    for r in 0..dr {
                        let i0 = (l * 2) * dr + r;
                        let i1 = (l * 2 + 1) * dr + r;
                        let (x0, x1) = (a[i0], a[i1]);
                        a[i0] = x0 + tau * x1;
                        a[i1] = x1 + tau * x0;
                    }
                }
            }
            self.normalize_site(k);
        }
    }

    /// `exp(-eta h Z)` on site `k` in its normalized form `diag(g, 1/g)` with
    /// `g = exp(-eta h)`.
    ///
    /// The exponent is clamped to `+-40`: raw imaginary-time factors overflow
    /// f64 on production-scale energies, and only the relative amplitudes
    /// matter, so the clamp costs nothing that the state uses.
    pub(crate) fn apply_field(&mut self, k: usize, eta_h: f64) {
        if eta_h == 0.0 || !eta_h.is_finite() || k >= self.num_sites() {
            return;
        }
        let g = (-eta_h.clamp(-40.0, 40.0)).exp();
        let dl = self.bond[k];
        let dr = self.bond[k + 1];
        {
            let a = &mut self.site[k];
            for l in 0..dl {
                for r in 0..dr {
                    a[(l * 2) * dr + r] *= g;
                    a[(l * 2 + 1) * dr + r] /= g;
                }
            }
        }
        self.normalize_site(k);
    }

    /// Sweep right to left until every tensor `k > 0` is a right isometry, then
    /// normalize site 0.
    ///
    /// Exact chain-rule sampling needs this form: every tensor to the right of
    /// site `k` then contracts to the identity, so the partial norm at site `k`
    /// is already the correct marginal. Without it the same loop needs a full
    /// right-environment sweep per site and costs `O(n^2)`.
    ///
    /// `chi` caps the bond on the way through. After a gate layer the bonds are
    /// already at or below `chi`, so the cap is normally a no-op; it is applied
    /// anyway so no path can leave an oversized bond behind.
    pub(crate) fn right_canonicalize(&mut self, chi: usize) {
        let n = self.num_sites();
        if n == 0 {
            return;
        }
        for k in (1..n).rev() {
            let dl = self.bond[k];
            let dr = self.bond[k + 1];
            let cols = 2 * dr;
            // Factor B^T = Q C, so B = C^T Q^T and Q^T is the right isometry.
            let bt = transpose(&self.site[k], dl, cols);
            let f = truncated_svd(&bt, cols, dl, chi);
            let kk = f.k;
            self.site[k] = transpose(&f.q, cols, kk);
            self.bond[k] = kk;
            let carry = transpose(&f.carry, kk, dl);
            let dll = self.bond[k - 1];
            let mut next = vec![0.0; dll * 2 * kk];
            {
                let a = &self.site[k - 1];
                for l in 0..dll {
                    for p in 0..2 {
                        for m in 0..dl {
                            let av = a[(l * 2 + p) * dl + m];
                            if av == 0.0 {
                                continue;
                            }
                            for b in 0..kk {
                                next[(l * 2 + p) * kk + b] += av * carry[m * kk + b];
                            }
                        }
                    }
                }
            }
            self.site[k - 1] = next;
            normalize_vec(&mut self.site[k - 1]);
        }
        normalize_vec(&mut self.site[0]);
    }

    /// Apply `I(x)I - t Z(x)Z` across sites `u..=v` as a bond-2 MPO zip-up.
    ///
    /// The coupling factor is exactly rank 2, so a long-range coupling is an
    /// MPO of bond dimension 2 rather than one channel per pending spin. Cost
    /// is `O(span * chi^3)`, and sites outside `[u, v]` are untouched, so the
    /// price follows the span and never the graph bandwidth.
    ///
    /// Every intermediate carry is normalized. The Python prototype normalized
    /// only the final tensor and hit non-converging SVDs on expander instances,
    /// because the carried amplitudes overflowed across a long span;
    /// normalizing each carry fixed four of the five failing cases.
    pub(crate) fn apply_zz(&mut self, u: usize, v: usize, t: f64, chi: usize) {
        const Z: [f64; 2] = [1.0, -1.0];
        let (u, v) = if u <= v { (u, v) } else { (v, u) };
        if u == v || v >= self.num_sites() || t == 0.0 || !t.is_finite() {
            return;
        }
        if chi <= 1 && self.max_bond() == 1 {
            self.apply_zz_product(u, v, t);
            return;
        }
        let chi = chi.max(1);
        let dl = self.bond[u];
        let mut mid = self.bond[u + 1];

        // Site u: open the two MPO channels, 0 = I and 1 = -t Z.
        let mut mat = vec![0.0; dl * 2 * mid * 2];
        {
            let a = &self.site[u];
            for l in 0..dl {
                for (p, &zp) in Z.iter().enumerate() {
                    let row = l * 2 + p;
                    for r in 0..mid {
                        let av = a[row * mid + r];
                        mat[row * (mid * 2) + r * 2] = av;
                        mat[row * (mid * 2) + r * 2 + 1] = -t * zp * av;
                    }
                }
            }
        }
        let f = truncated_svd(&mat, dl * 2, mid * 2, chi);
        self.site[u] = f.q;
        self.bond[u + 1] = f.k;
        let mut nb = f.k;
        let mut carry = f.carry;
        normalize_vec(&mut carry);

        // Sites strictly between u and v: the MPO is the identity on both
        // channels, so the carry simply moves right.
        for k in (u + 1)..v {
            let dl2 = mid;
            let dr2 = self.bond[k + 1];
            let mut tmp = vec![0.0; nb * 2 * dr2 * 2];
            {
                let a = &self.site[k];
                for b in 0..nb {
                    for l2 in 0..dl2 {
                        for c in 0..2 {
                            let cv = carry[(b * dl2 + l2) * 2 + c];
                            if cv == 0.0 {
                                continue;
                            }
                            for p in 0..2 {
                                for r2 in 0..dr2 {
                                    tmp[((b * 2 + p) * dr2 + r2) * 2 + c] +=
                                        cv * a[(l2 * 2 + p) * dr2 + r2];
                                }
                            }
                        }
                    }
                }
            }
            let f = truncated_svd(&tmp, nb * 2, dr2 * 2, chi);
            self.site[k] = f.q;
            self.bond[k + 1] = f.k;
            mid = dr2;
            nb = f.k;
            carry = f.carry;
            normalize_vec(&mut carry);
        }

        // Site v: close the channels against (I, Z).
        let dl3 = mid;
        let dr3 = self.bond[v + 1];
        let mut closed = vec![0.0; nb * 2 * dr3];
        {
            let a = &self.site[v];
            for b in 0..nb {
                for l3 in 0..dl3 {
                    let c0 = carry[(b * dl3 + l3) * 2];
                    let c1 = carry[(b * dl3 + l3) * 2 + 1];
                    for p in 0..2 {
                        let w = c0 + Z[p] * c1;
                        if w == 0.0 {
                            continue;
                        }
                        for r3 in 0..dr3 {
                            closed[(b * 2 + p) * dr3 + r3] += w * a[(l3 * 2 + p) * dr3 + r3];
                        }
                    }
                }
            }
        }
        normalize_vec(&mut closed);
        self.site[v] = closed;
    }

    /// Bond-1 fast path: at bond dimension 1 the state is a product state, so
    /// the coupling gate reduces to a rank-1 truncation of one 2 x 2 matrix on
    /// sites `u` and `v`. The span drops out and the cost is `O(1)` per edge.
    ///
    /// This is the path that runs on the production topology, where the budget
    /// selects bond dimension 1, and it is the whole of `quip-cpu-mfa`. It is
    /// also a better approximation at bond 1 than the generic zip-up would be:
    /// the zip-up would truncate the MPO channel at every intermediate site,
    /// while this keeps both channels exactly for the pair.
    pub(crate) fn apply_zz_product(&mut self, u: usize, v: usize, t: f64) {
        const Z: [f64; 2] = [1.0, -1.0];
        let (u, v) = if u <= v { (u, v) } else { (v, u) };
        if u == v || v >= self.num_sites() || t == 0.0 || !t.is_finite() {
            return;
        }
        if self.site[u].len() != 2 || self.site[v].len() != 2 {
            // Not a product state on this pair; the generic zip-up owns it.
            return;
        }
        let mut m = [0.0f64; 4];
        {
            let a = &self.site[u];
            let b = &self.site[v];
            for p in 0..2 {
                for q in 0..2 {
                    m[p * 2 + q] = a[p] * b[q] * (1.0 - t * Z[p] * Z[q]);
                }
            }
        }
        let f = truncated_svd(&m, 2, 2, 1);
        if f.k != 1 || f.q.len() != 2 || f.carry.len() != 2 {
            return;
        }
        let mut left = f.q;
        let mut right = f.carry;
        normalize_vec(&mut left);
        normalize_vec(&mut right);
        self.site[u] = left;
        self.site[v] = right;
    }

    /// One exact sample by the Bayesian chain rule, in chain order.
    ///
    /// Requires right-canonical form: every tensor to the right of site `k`
    /// then contracts to the identity, so the partial norm at site `k` is
    /// already the correct marginal and one left-to-right pass is exact. Costs
    /// `O(n chi^2 d)` per sample, which is negligible next to the solve.
    ///
    /// A site whose two branches both carry zero weight cannot happen for a
    /// normalized state, but a truncation can produce one; it falls back to a
    /// fair coin so the draw completes instead of dividing by zero.
    pub(crate) fn sample_one(&self, rng: &mut SmallRng) -> Vec<i8> {
        let n = self.num_sites();
        let mut spins = vec![1i8; n];
        let mut msg = vec![1.0f64];
        for (k, spin) in spins.iter_mut().enumerate() {
            let dl = self.bond[k].min(msg.len());
            let dr = self.bond[k + 1];
            let mut w0 = vec![0.0; dr];
            let mut w1 = vec![0.0; dr];
            {
                let a = &self.site[k];
                for l in 0..dl {
                    let mv = msg[l];
                    if mv == 0.0 {
                        continue;
                    }
                    for r in 0..dr {
                        w0[r] += mv * a[(l * 2) * dr + r];
                        w1[r] += mv * a[(l * 2 + 1) * dr + r];
                    }
                }
            }
            let p0: f64 = w0.iter().map(|x| x * x).sum();
            let p1: f64 = w1.iter().map(|x| x * x).sum();
            let total = p0 + p1;
            let pick_zero = if total > 0.0 && total.is_finite() {
                rng.gen::<f64>() < p0 / total
            } else {
                rng.gen::<bool>()
            };
            *spin = if pick_zero { 1 } else { -1 };
            msg = if pick_zero { w0 } else { w1 };
            normalize_vec(&mut msg);
        }
        spins
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn plus_state_has_unit_bonds_and_uniform_amplitudes() {
        let mps = Mps::plus_state(5);
        assert_eq!(mps.num_sites(), 5);
        assert_eq!(mps.bond, vec![1; 6]);
        assert_eq!(mps.max_bond(), 1);
        for k in 0..5 {
            assert_eq!(mps.site[k].len(), 2);
            assert!((mps.site[k][0] - mps.site[k][1]).abs() < 1e-15);
        }
        let dense = mps.to_dense();
        assert_eq!(dense.len(), 32);
        let want = (0.5f64).powf(2.5);
        for (i, &x) in dense.iter().enumerate() {
            assert!((x - want).abs() <= 1e-15, "amplitude {i} = {x}, want {want}");
        }
        let norm: f64 = dense.iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() <= 1e-14, "norm = {norm}");
    }

    #[test]
    fn plus_state_of_zero_sites_is_empty() {
        let mps = Mps::plus_state(0);
        assert_eq!(mps.num_sites(), 0);
        assert_eq!(mps.bond, vec![1]);
        assert_eq!(mps.to_dense(), vec![1.0]);
    }

    #[test]
    fn to_dense_orders_site_zero_as_the_most_significant_bit() {
        // Two sites, hand-built: site 0 = [2, 3], site 1 = [5, 7].
        // Amplitude(p0, p1) = site0[p0] * site1[p1], index = p0 * 2 + p1.
        let mps = Mps {
            site: vec![vec![2.0, 3.0], vec![5.0, 7.0]],
            bond: vec![1, 1, 1],
        };
        assert_eq!(mps.to_dense(), vec![10.0, 14.0, 15.0, 21.0]);
    }

    #[test]
    fn to_dense_contracts_a_bond_of_dimension_two() {
        // site 0: 1 x 2 x 2 = [[a00, a01], [a10, a11]] over (p, r).
        // site 1: 2 x 2 x 1 = [[b00, b01], [b10, b11]] over (l, p).
        let mps = Mps {
            site: vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
            bond: vec![1, 2, 1],
        };
        // amp(p0, p1) = sum_r site0[(0*2+p0)*2 + r] * site1[(r*2+p1)*1 + 0]
        assert_eq!(
            mps.to_dense(),
            vec![
                1.0 * 5.0 + 2.0 * 7.0,
                1.0 * 6.0 + 2.0 * 8.0,
                3.0 * 5.0 + 4.0 * 7.0,
                3.0 * 6.0 + 4.0 * 8.0,
            ]
        );
    }

    #[test]
    fn normalize_vec_leaves_a_zero_vector_alone() {
        let mut v = vec![0.0, 0.0, 0.0];
        normalize_vec(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);

        let mut w = vec![3.0, 4.0];
        normalize_vec(&mut w);
        assert!((w[0] - 0.6).abs() <= 1e-15);
        assert!((w[1] - 0.8).abs() <= 1e-15);
    }
    // ---- dense reference implementation, the oracle for every gate test ----

    fn dense_plus(n: usize) -> Vec<f64> {
        vec![(0.5f64).powf(n as f64 / 2.0); 1 << n]
    }

    fn dense_normalize(v: &mut [f64]) {
        let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm > 0.0 {
            for x in v {
                *x /= nrm;
            }
        }
    }

    /// Physical index of site `k` in configuration `idx`, site `0` most
    /// significant.
    fn bit(idx: usize, k: usize, n: usize) -> usize {
        (idx >> (n - 1 - k)) & 1
    }

    fn dense_transverse(v: &mut [f64], n: usize, tau: f64) {
        for k in 0..n {
            let stride = 1usize << (n - 1 - k);
            let src = v.to_vec();
            for idx in 0..v.len() {
                v[idx] = src[idx] + tau * src[idx ^ stride];
            }
        }
    }

    fn dense_field(v: &mut [f64], n: usize, k: usize, eta_h: f64) {
        let g = (-eta_h.clamp(-40.0, 40.0)).exp();
        for (idx, x) in v.iter_mut().enumerate() {
            *x *= if bit(idx, k, n) == 0 { g } else { 1.0 / g };
        }
    }

    fn dense_zz(v: &mut [f64], n: usize, u: usize, w: usize, t: f64) {
        let z = [1.0f64, -1.0];
        for idx in 0..v.len() {
            v[idx] *= 1.0 - t * z[bit(idx, u, n)] * z[bit(idx, w, n)];
        }
    }

    /// Compare two dense vectors up to a global scale.
    fn assert_dense_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let mut g = got.to_vec();
        let mut w = want.to_vec();
        dense_normalize(&mut g);
        dense_normalize(&mut w);
        for i in 0..g.len() {
            assert!(
                (g[i] - w[i]).abs() <= tol,
                "{what}: amplitude {i} got {} want {}",
                g[i],
                w[i]
            );
        }
    }

    // ---- gate tests ----

    /// The load-bearing algebraic claim of the whole design: the imaginary-time
    /// coupling factor is exactly rank 2, so a long-range coupling costs its
    /// chain span and never the graph bandwidth.
    #[test]
    fn rank_two_coupling_gate_equals_the_dense_exponential() {
        let z = [1.0f64, -1.0];
        for step in 0..=40 {
            let theta = -10.0 + 0.5 * f64::from(step);
            let t = theta.tanh();
            // Dense 4x4 of exp(-theta Z(x)Z) / cosh(theta); it is diagonal.
            for p in 0..2 {
                for q in 0..2 {
                    let dense = (-theta * z[p] * z[q]).exp() / theta.cosh();
                    let rank2 = 1.0 - t * z[p] * z[q];
                    assert!(
                        (dense - rank2).abs() <= 1e-14 * dense.abs().max(1.0),
                        "theta {theta}, entry ({p},{q}): dense {dense} rank2 {rank2}"
                    );
                }
            }
        }
    }

    #[test]
    fn transverse_gate_matches_the_dense_reference() {
        let n = 4;
        let mut mps = Mps::plus_state(n);
        let mut dense = dense_plus(n);
        for &tau in &[0.3f64, -0.15, 0.9] {
            mps.apply_transverse(tau);
            dense_transverse(&mut dense, n, tau);
            assert_dense_close(&mps.to_dense(), &dense, 1e-12, "transverse");
        }
        assert_eq!(mps.max_bond(), 1, "a single-site gate must not grow a bond");
    }

    #[test]
    fn field_gate_matches_the_dense_reference_and_clamps_the_exponent() {
        let n = 3;
        let mut mps = Mps::plus_state(n);
        let mut dense = dense_plus(n);
        for (site, eta_h) in [(0usize, 0.4f64), (2, -1.3), (1, 60.0), (0, -60.0)] {
            mps.apply_field(site, eta_h);
            dense_field(&mut dense, n, site, eta_h);
            assert_dense_close(&mps.to_dense(), &dense, 1e-12, "field");
        }
        assert!(mps.to_dense().iter().all(|x| x.is_finite()));
        assert_eq!(mps.max_bond(), 1);
    }

    #[test]
    fn gates_with_no_effect_are_skipped() {
        let mut mps = Mps::plus_state(3);
        let before = mps.to_dense();
        mps.apply_transverse(0.0);
        mps.apply_transverse(f64::NAN);
        mps.apply_field(1, 0.0);
        mps.apply_field(1, f64::INFINITY);
        assert_eq!(mps.to_dense(), before);
    }

    // ---- Task 6: right canonicalization ----

    /// Every tensor right of site 0 must satisfy `sum_p B_p B_p^T = I`.
    fn assert_right_canonical(mps: &Mps, tol: f64) {
        for k in 1..mps.num_sites() {
            let dl = mps.bond[k];
            let dr = mps.bond[k + 1];
            for b in 0..dl {
                for c in 0..dl {
                    let mut dot = 0.0;
                    for p in 0..2 {
                        for r in 0..dr {
                            dot += mps.site[k][(b * 2 + p) * dr + r]
                                * mps.site[k][(c * 2 + p) * dr + r];
                        }
                    }
                    let want = if b == c { 1.0 } else { 0.0 };
                    assert!(
                        (dot - want).abs() <= tol,
                        "site {k}: row {b} dot row {c} = {dot}, want {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn right_canonicalize_makes_every_tensor_a_right_isometry() {
        let n = 6;
        let mut mps = Mps::plus_state(n);
        mps.apply_transverse(0.4);
        mps.apply_field(0, 0.3);
        mps.apply_field(3, -0.7);
        mps.apply_zz(0, 4, 0.5, 8);
        mps.apply_zz(1, 5, -0.35, 8);
        mps.apply_zz(2, 3, 0.9, 8);
        mps.right_canonicalize(8);
        assert_right_canonical(&mps, 1e-12);

        let dense = mps.to_dense();
        let norm: f64 = dense.iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() <= 1e-12, "state norm = {norm}");
    }

    #[test]
    fn right_canonicalize_preserves_the_state_up_to_normalization() {
        let n = 5;
        let mut mps = Mps::plus_state(n);
        mps.apply_transverse(0.25);
        mps.apply_field(2, 0.8);
        mps.apply_zz(1, 4, 0.6, 16);
        let before = mps.to_dense();
        mps.right_canonicalize(16);
        assert_dense_close(&mps.to_dense(), &before, 1e-12, "canonicalization");
    }

    #[test]
    fn right_canonicalize_is_safe_on_an_empty_and_a_single_site_chain() {
        let mut empty = Mps::plus_state(0);
        empty.right_canonicalize(4);
        assert_eq!(empty.num_sites(), 0);

        let mut one = Mps::plus_state(1);
        one.apply_field(0, 1.5);
        one.right_canonicalize(4);
        let dense = one.to_dense();
        let norm: f64 = dense.iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() <= 1e-12, "norm = {norm}");
        // exp(-eta h Z) with a positive exponent favours spin -1 (index 1).
        assert!(dense[1].abs() > dense[0].abs());
    }

    // ---- Task 7: the coupling zip-up ----

    #[test]
    fn a_single_coupling_gate_matches_the_dense_reference() {
        for &(n, u, w) in &[
            (2usize, 0usize, 1usize),
            (4, 0, 3),
            (5, 1, 3),
            (6, 0, 5),
            (6, 4, 1),
        ] {
            for &t in &[0.35f64, -0.8, 0.99] {
                let chi = 1usize << n.div_ceil(2);
                let mut mps = Mps::plus_state(n);
                let mut dense = dense_plus(n);
                mps.apply_transverse(0.3);
                dense_transverse(&mut dense, n, 0.3);
                mps.apply_zz(u, w, t, chi);
                dense_zz(&mut dense, n, u, w, t);
                assert_dense_close(
                    &mps.to_dense(),
                    &dense,
                    1e-11,
                    &format!("n={n} u={u} w={w} t={t}"),
                );
            }
        }
    }

    #[test]
    fn stacked_coupling_gates_match_the_dense_reference() {
        let n = 7usize;
        let chi = 1usize << n.div_ceil(2);
        let mut mps = Mps::plus_state(n);
        let mut dense = dense_plus(n);
        for &(u, w, t) in &[
            (0usize, 1usize, 0.4f64),
            (0, 6, -0.55),
            (2, 5, 0.7),
            (3, 4, -0.2),
            (1, 4, 0.65),
        ] {
            mps.apply_zz(u, w, t, chi);
            dense_zz(&mut dense, n, u, w, t);
        }
        assert_dense_close(&mps.to_dense(), &dense, 1e-11, "stacked couplings");
        assert!(mps.max_bond() <= chi, "bond {} exceeds chi", mps.max_bond());
    }

    #[test]
    fn the_zip_up_normalizes_every_intermediate_carry() {
        // A long span with a near-hard-constraint coupling: the first prototype
        // normalized only the final tensor and overflowed the carried
        // amplitudes across a span like this.
        let n = 12;
        let mut mps = Mps::plus_state(n);
        mps.apply_transverse(0.5);
        for _ in 0..12 {
            mps.apply_zz(0, 11, 0.999_999, 8);
            mps.right_canonicalize(8);
        }
        let dense = mps.to_dense();
        assert!(dense.iter().all(|x| x.is_finite()), "carry overflowed");
        let norm: f64 = dense.iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() <= 1e-10, "norm = {norm}");
    }

    #[test]
    fn coupling_gates_that_do_nothing_are_skipped() {
        let mut mps = Mps::plus_state(4);
        mps.apply_transverse(0.4);
        let before = mps.to_dense();
        mps.apply_zz(2, 2, 0.5, 8); // self-loop
        mps.apply_zz(1, 3, 0.0, 8); // zero coupling
        mps.apply_zz(1, 3, f64::NAN, 8); // non-finite coupling
        mps.apply_zz(0, 9, 0.5, 8); // out of range
        assert_eq!(mps.to_dense(), before);
    }

    #[test]
    fn coupling_gate_argument_order_does_not_matter() {
        let n = 5;
        let chi = 8;
        let mut forward = Mps::plus_state(n);
        forward.apply_transverse(0.3);
        forward.apply_zz(1, 4, 0.6, chi);
        let mut backward = Mps::plus_state(n);
        backward.apply_transverse(0.3);
        backward.apply_zz(4, 1, 0.6, chi);
        assert_dense_close(&forward.to_dense(), &backward.to_dense(), 1e-12, "order");
    }

    // ---- Task 9: the bond-1 product fast path ----

    #[test]
    fn the_product_fast_path_is_the_best_rank_one_approximation() {
        // Sites 0 and 3 of a five-site product state, coupling t = 0.5.
        let mut mps = Mps::plus_state(5);
        mps.apply_field(0, 0.4);
        mps.apply_field(3, -0.25);
        let a = mps.site[0].clone();
        let b = mps.site[3].clone();
        mps.apply_zz_product(0, 3, 0.5);

        // Hand-built 2x2 of the gated pair, then its best rank-1 factor.
        let z = [1.0f64, -1.0];
        let mut m = [0.0f64; 4];
        for p in 0..2 {
            for q in 0..2 {
                m[p * 2 + q] = a[p] * b[q] * (1.0 - 0.5 * z[p] * z[q]);
            }
        }
        let got_left = &mps.site[0];
        let got_right = &mps.site[3];
        assert_eq!(got_left.len(), 2);
        assert_eq!(got_right.len(), 2);
        assert_eq!(mps.max_bond(), 1, "the fast path must not grow a bond");

        // The outer product of the two returned vectors, rescaled to match the
        // leading singular value, must be the best rank-1 approximation, so its
        // residual equals the smaller singular value.
        let mut outer = [0.0f64; 4];
        for p in 0..2 {
            for q in 0..2 {
                outer[p * 2 + q] = got_left[p] * got_right[q];
            }
        }
        let scale: f64 = (0..4).map(|i| m[i] * outer[i]).sum();
        let resid: f64 = (0..4)
            .map(|i| (m[i] - scale * outer[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        // For a 2x2 the best possible rank-1 residual is exactly the second
        // singular value, so that is the bound to assert. A fixed fraction of
        // the Frobenius norm is not: this matrix has s2/s1 = 0.22, so any
        // threshold below s2 demands better than optimal and fails a correct
        // implementation.
        let svd = super::super::svd::jacobi_svd(&m, 2, 2, super::super::svd::JACOBI_MAX_SWEEPS)
            .expect("a 2x2 always converges");
        assert!(
            resid <= svd.s[1] + 1e-12,
            "residual {resid} exceeds the optimal rank-1 residual {}",
            svd.s[1]
        );
    }

    #[test]
    fn apply_zz_dispatches_to_the_product_path_at_bond_one() {
        let build = || {
            let mut mps = Mps::plus_state(6);
            mps.apply_field(1, 0.5);
            mps.apply_field(4, -0.3);
            mps
        };
        let mut dispatched = build();
        dispatched.apply_zz(1, 4, 0.45, 1);
        let mut direct = build();
        direct.apply_zz_product(1, 4, 0.45);
        assert_eq!(
            dispatched.bond, direct.bond,
            "the dispatch must take the product path"
        );
        for k in 0..6 {
            assert_eq!(
                dispatched.site[k], direct.site[k],
                "site {k} differs between the dispatch and the direct call"
            );
        }
    }

    #[test]
    fn the_product_path_leaves_untouched_sites_untouched() {
        let mut mps = Mps::plus_state(6);
        mps.apply_field(2, 0.9);
        let middle = mps.site[2].clone();
        let far = mps.site[5].clone();
        mps.apply_zz_product(0, 3, 0.7);
        assert_eq!(mps.site[2], middle, "a site inside the span must not move");
        assert_eq!(mps.site[5], far, "a site outside the span must not move");
        assert_eq!(mps.max_bond(), 1);
    }

    #[test]
    fn a_chain_of_product_gates_stays_finite_and_normalized() {
        let mut mps = Mps::plus_state(64);
        mps.apply_transverse(0.6);
        for k in 0..63 {
            mps.apply_zz(k, k + 1, 0.95, 1);
        }
        mps.right_canonicalize(1);
        assert_eq!(mps.max_bond(), 1);
        for k in 0..64 {
            let nrm: f64 = mps.site[k].iter().map(|x| x * x).sum();
            assert!(nrm.is_finite() && nrm > 0.0, "site {k} norm = {nrm}");
        }
    }

    // ---- Task 8: untruncated evolution against dense imaginary time ----

    /// With `chi_max = 2^ceil(n/2)` no truncation can occur, because the
    /// MPO-augmented bond during the zip-up is bounded by
    /// `min(2^(k+1), 2^(n-k))`, which never exceeds `2^ceil(n/2)`. The MPS
    /// evolution must therefore reproduce dense imaginary-time evolution of the
    /// same Trotter product exactly.
    #[test]
    fn untruncated_tebd_matches_dense_imaginary_time_evolution() {
        const N: usize = 8;
        let chi = 1usize << N.div_ceil(2);
        let gates: [(usize, usize, f64); 5] = [
            (0, 1, 0.8),
            (1, 4, -0.6),
            (2, 3, 0.45),
            (0, 7, 0.3),
            (5, 6, -0.9),
        ];
        let fields = [0.3f64, -0.2, 0.0, 0.7, -0.5, 0.1, 0.0, -0.4];
        let etas = [0.05f64, 0.12, 0.3, 0.7];
        let gammas = [2.0f64, 1.3, 0.6, 0.0];

        let mut mps = Mps::plus_state(N);
        let mut dense = dense_plus(N);
        for step in 0..etas.len() {
            let eta = etas[step];
            let tau = (eta * gammas[step] / 2.0).tanh();

            mps.apply_transverse(tau);
            dense_transverse(&mut dense, N, tau);

            for (i, &hi) in fields.iter().enumerate() {
                mps.apply_field(i, eta * hi);
                dense_field(&mut dense, N, i, eta * hi);
            }

            for &(u, w, j) in &gates {
                let t = (eta * j).tanh();
                mps.apply_zz(u, w, t, chi);
                dense_zz(&mut dense, N, u, w, t);
            }

            mps.apply_transverse(tau);
            dense_transverse(&mut dense, N, tau);

            mps.right_canonicalize(chi);
            dense_normalize(&mut dense);

            assert_dense_close(
                &mps.to_dense(),
                &dense,
                1e-10,
                &format!("untruncated TEBD after step {step}"),
            );
        }
        assert_right_canonical(&mps, 1e-11);
    }

    /// The same evolution at a bond cap that does truncate must stay a valid,
    /// normalized state and must stay closer to the exact answer than a
    /// product state is.
    #[test]
    fn truncated_tebd_stays_a_valid_state_and_beats_the_product_state() {
        const N: usize = 8;
        let exact_chi = 1usize << N.div_ceil(2);
        let gates: [(usize, usize, f64); 5] = [
            (0, 1, 0.8),
            (1, 4, -0.6),
            (2, 3, 0.45),
            (0, 7, 0.3),
            (5, 6, -0.9),
        ];
        let etas = [0.2f64, 0.5, 1.0];

        let run = |chi: usize| {
            let mut mps = Mps::plus_state(N);
            for &eta in &etas {
                mps.apply_transverse((eta * 0.5).tanh());
                for &(u, w, j) in &gates {
                    mps.apply_zz(u, w, (eta * j).tanh(), chi);
                }
                mps.apply_transverse((eta * 0.5).tanh());
                mps.right_canonicalize(chi);
            }
            let mut d = mps.to_dense();
            dense_normalize(&mut d);
            d
        };

        let exact = run(exact_chi);
        let overlap = |v: &[f64]| -> f64 {
            v.iter().zip(&exact).map(|(a, b)| a * b).sum::<f64>().abs()
        };
        let at_four = run(4);
        let at_one = run(1);
        assert!(
            at_four.iter().all(|x| x.is_finite()),
            "truncated state must stay finite"
        );
        assert!(
            overlap(&at_four) >= overlap(&at_one),
            "chi=4 overlap {} must be at least the chi=1 overlap {}",
            overlap(&at_four),
            overlap(&at_one)
        );
        assert!(
            overlap(&exact) > 0.999_999,
            "self-overlap {} must be 1",
            overlap(&exact)
        );
    }

    // ---- Task 10: exact chain-rule sampling ----

    #[test]
    fn sampling_returns_one_plus_or_minus_one_per_site() {
        let mut mps = Mps::plus_state(7);
        mps.apply_transverse(0.4);
        mps.right_canonicalize(4);
        let mut rng = SmallRng::seed_from_u64(11);
        for _ in 0..32 {
            let spins = mps.sample_one(&mut rng);
            assert_eq!(spins.len(), 7);
            assert!(spins.iter().all(|&s| s == 1 || s == -1), "{spins:?}");
        }
    }

    #[test]
    fn sampling_a_deterministic_state_always_returns_the_same_configuration() {
        // Strong fields with no couplings: the state collapses onto one
        // configuration, so every draw must return it.
        let n = 6;
        let fields = [3.0f64, -3.0, 3.0, 3.0, -3.0, -3.0];
        let mut mps = Mps::plus_state(n);
        for (i, &h) in fields.iter().enumerate() {
            mps.apply_field(i, 8.0 * h);
        }
        mps.right_canonicalize(4);
        let want: Vec<i8> = fields
            .iter()
            .map(|&h| if h > 0.0 { -1i8 } else { 1i8 })
            .collect();
        let mut rng = SmallRng::seed_from_u64(5);
        for draw in 0..64 {
            assert_eq!(mps.sample_one(&mut rng), want, "draw {draw}");
        }
    }

    #[test]
    fn sampling_is_deterministic_for_a_given_seed() {
        let mut mps = Mps::plus_state(8);
        mps.apply_transverse(0.35);
        mps.apply_zz(0, 5, 0.4, 8);
        mps.right_canonicalize(8);
        let draw = |seed: u64| {
            let mut rng = SmallRng::seed_from_u64(seed);
            (0..16).map(|_| mps.sample_one(&mut rng)).collect::<Vec<_>>()
        };
        assert_eq!(draw(99), draw(99));
    }

    /// Chi-squared against the exact Born distribution. Every outcome of this
    /// state is well populated, so no pooling is needed and the statistic has
    /// 63 degrees of freedom. The 1 - 1e-4 quantile there is near 113.7 by the
    /// Wilson-Hilferty approximation, so a threshold of 120 fails fewer than
    /// 1 run in 10^4 when the sampler is correct.
    #[test]
    fn sampling_matches_the_exact_born_distribution() {
        const N: usize = 6;
        const DRAWS: usize = 1_000_000;
        let mut mps = Mps::plus_state(N);
        mps.apply_transverse(0.2);
        mps.apply_field(1, 0.15);
        mps.apply_field(4, -0.1);
        mps.apply_zz(0, 3, 0.3, 8);
        mps.apply_zz(2, 5, -0.25, 8);
        mps.right_canonicalize(8);

        let dense = mps.to_dense();
        let norm: f64 = dense.iter().map(|x| x * x).sum();
        let expected: Vec<f64> = dense.iter().map(|x| x * x / norm * DRAWS as f64).collect();
        let min_expected = expected.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            min_expected >= 100.0,
            "test premise broken: least populated outcome expects {min_expected} draws"
        );

        let mut counts = vec![0u64; 1 << N];
        let mut rng = SmallRng::seed_from_u64(20_260_807);
        for _ in 0..DRAWS {
            let spins = mps.sample_one(&mut rng);
            let mut idx = 0usize;
            for &s in &spins {
                idx = (idx << 1) | usize::from(s < 0);
            }
            counts[idx] += 1;
        }
        let chi2: f64 = counts
            .iter()
            .zip(&expected)
            .map(|(&c, &e)| {
                let d = c as f64 - e;
                d * d / e
            })
            .sum();
        assert!(
            chi2 < 120.0,
            "chi-squared {chi2} exceeds the 1-in-10^4 threshold for 63 degrees of freedom"
        );
    }

}
