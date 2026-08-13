//! One-sided Jacobi SVD for the small matrices the MPS zip-up factorizes.
//!
//! Every matrix here is at most `2 chi_max x 2 chi_max`, which is 64 x 64 at the
//! default `chi_max = 32`, so a rotation sweep is cheap and a linear-algebra
//! dependency is not justified. One-sided Jacobi never forms `A^T A`, so it
//! keeps high relative accuracy on the small singular values that the
//! truncation then discards.
//!
//! Three rules keep the factorization deterministic, and the whole miner
//! depends on them:
//! 1. Fixed cyclic-by-column-pair sweep order and a fixed threshold.
//! 2. Singular values sort descending, ties broken by original index.
//! 3. The largest-magnitude entry of each left singular vector is positive;
//!    on a tie the lowest row index decides.

use std::sync::atomic::{AtomicU64, Ordering};

/// Sweep cap. One-sided Jacobi converges quadratically, so 60 sweeps is far
/// past what a 64 x 64 matrix needs. Reaching it means the input is
/// pathological, and the caller falls back rather than returning noise.
pub(crate) const JACOBI_MAX_SWEEPS: usize = 60;

/// A column pair is orthogonal enough when `|a.b| <= TOL * sqrt(|a|^2 |b|^2)`.
const JACOBI_TOL: f64 = 1e-14;

/// Relative floor below which a singular value is dropped. Kept far below the
/// truncation that `chi` performs, so the untruncated-evolution test stays
/// exact to the tolerance it asserts.
const SINGULAR_CUTOFF: f64 = 1e-14;

/// How many factorizations fell back to the QR path. Monotonic for the life of
/// the process; the tests compare a before-and-after delta.
static SVD_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Thin SVD `A = U diag(s) V^T` of a row-major `m x n` matrix.
pub(crate) struct Svd {
    /// `m x k` row-major. Columns are the left singular vectors.
    pub(crate) u: Vec<f64>,
    /// Singular values, descending, length `k`.
    pub(crate) s: Vec<f64>,
    /// `k x n` row-major. Rows are the right singular vectors.
    pub(crate) vt: Vec<f64>,
    /// `min(m, n)`.
    pub(crate) k: usize,
}

/// Transpose a row-major `m x n` matrix into a row-major `n x m` matrix.
pub(crate) fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j];
        }
    }
    out
}

/// Write into column `col` of the row-major `m x k` matrix `q` a unit vector
/// orthogonal to every column flagged in `set`. Deterministic: it tries the
/// standard basis vectors in index order and takes the first that survives two
/// passes of Gram-Schmidt. Used for the null directions of a rank-deficient
/// input, so that `U` stays orthonormal instead of carrying zero columns.
fn orthonormal_fill(q: &mut [f64], m: usize, k: usize, col: usize, set: &[bool]) {
    for e in 0..m {
        let mut work = vec![0.0f64; m];
        work[e] = 1.0;
        for _ in 0..2 {
            for c in 0..k {
                if c == col || !set[c] {
                    continue;
                }
                let mut dot = 0.0;
                for i in 0..m {
                    dot += q[i * k + c] * work[i];
                }
                for i in 0..m {
                    work[i] -= dot * q[i * k + c];
                }
            }
        }
        let nrm = work.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm > 0.5 {
            for i in 0..m {
                q[i * k + col] = work[i] / nrm;
            }
            return;
        }
    }
    // `col >= m`: no orthogonal direction exists, so the column stays zero.
}

/// Rotate the columns of a tall matrix (`m >= n`) until they are pairwise
/// orthogonal. Returns the rotated matrix and the accumulated `n x n` rotation,
/// or `None` when the sweep cap is reached.
fn jacobi_columns(
    a: &[f64],
    m: usize,
    n: usize,
    max_sweeps: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut w = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..max_sweeps {
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for i in 0..m {
                    let wp = w[i * n + p];
                    let wq = w[i * n + q];
                    alpha += wp * wp;
                    beta += wq * wq;
                    gamma += wp * wq;
                }
                if gamma.abs() <= JACOBI_TOL * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let sign = if zeta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let wp = w[i * n + p];
                    let wq = w[i * n + q];
                    w[i * n + p] = c * wp - s * wq;
                    w[i * n + q] = s * wp + c * wq;
                }
                for i in 0..n {
                    let vp = v[i * n + p];
                    let vq = v[i * n + q];
                    v[i * n + p] = c * vp - s * vq;
                    v[i * n + q] = s * vp + c * vq;
                }
            }
        }
        if !rotated {
            return Some((w, v));
        }
    }
    None
}

/// Split the rotated matrix into unit columns and their norms, completing the
/// basis for any column whose norm is zero.
fn split_norms(w: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut u = vec![0.0; rows * cols];
    let mut s = vec![0.0; cols];
    let mut set = vec![false; cols];
    for j in 0..cols {
        let mut acc = 0.0;
        for i in 0..rows {
            let x = w[i * cols + j];
            acc += x * x;
        }
        let sigma = acc.sqrt();
        s[j] = sigma;
        if sigma > 0.0 {
            for i in 0..rows {
                u[i * cols + j] = w[i * cols + j] / sigma;
            }
            set[j] = true;
        }
    }
    for j in 0..cols {
        if !set[j] {
            orthonormal_fill(&mut u, rows, cols, j, &set);
            set[j] = true;
        }
    }
    (u, s)
}

/// Thin SVD of a row-major `m x n` matrix, or `None` when the Jacobi sweep cap
/// is reached.
pub(crate) fn jacobi_svd(a: &[f64], m: usize, n: usize, max_sweeps: usize) -> Option<Svd> {
    let k = m.min(n);
    if k == 0 {
        return Some(Svd {
            u: Vec::new(),
            s: Vec::new(),
            vt: Vec::new(),
            k: 0,
        });
    }
    // The kernel needs at least as many rows as columns. When it does not have
    // them, factorize the transpose and swap the two vector sets back.
    let (u_raw, s_raw, v_raw) = if m >= n {
        let (w, rot) = jacobi_columns(a, m, n, max_sweeps)?;
        let (u, s) = split_norms(&w, m, n);
        (u, s, rot)
    } else {
        let at = transpose(a, m, n);
        let (w, rot) = jacobi_columns(&at, n, m, max_sweeps)?;
        let (v, s) = split_norms(&w, n, m);
        (rot, s, v)
    };

    // Descending by value, ties broken by original index: a stable, total order
    // that does not depend on the sort implementation.
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_by(|&x, &y| {
        s_raw[y]
            .partial_cmp(&s_raw[x])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.cmp(&y))
    });

    let mut u = vec![0.0; m * k];
    let mut s = vec![0.0; k];
    let mut v = vec![0.0; n * k];
    for (b, &src) in idx.iter().enumerate() {
        s[b] = s_raw[src];
        for i in 0..m {
            u[i * k + b] = u_raw[i * k + src];
        }
        for i in 0..n {
            v[i * k + b] = v_raw[i * k + src];
        }
    }

    // Sign convention: the largest-magnitude entry of each left vector is made
    // positive, with the lowest row index winning a tie.
    for b in 0..k {
        let mut best = 0.0f64;
        let mut best_val = 0.0f64;
        for i in 0..m {
            let x = u[i * k + b];
            if x.abs() > best {
                best = x.abs();
                best_val = x;
            }
        }
        if best_val < 0.0 {
            for i in 0..m {
                u[i * k + b] = -u[i * k + b];
            }
            for i in 0..n {
                v[i * k + b] = -v[i * k + b];
            }
        }
    }

    let vt = transpose(&v, n, k);
    Some(Svd { u, s, vt, k })
}

/// Fallback-path counter, for the tests that pin the non-convergence
/// behaviour. Production never reads it.
#[cfg(test)]
pub(crate) fn svd_fallback_count() -> u64 {
    SVD_FALLBACKS.load(Ordering::Relaxed)
}

/// `A ~ q * carry` with `q` an `m x k` isometry and `carry` a `k x n` matrix.
pub(crate) struct Factor {
    /// `m x k` row-major, orthonormal columns.
    pub(crate) q: Vec<f64>,
    /// `k x n` row-major.
    pub(crate) carry: Vec<f64>,
    /// Retained bond dimension, at least 1 and at most `chi`.
    pub(crate) k: usize,
}

/// Modified Gram-Schmidt QR of the columns of `a`, truncated to the leading
/// `chi` columns of `Q` and the leading `chi` rows of `R`.
///
/// This is the SVD non-convergence fallback. It keeps the isometry property
/// that the zip-up and the canonical form depend on, so the job completes; it
/// loses the optimality of the SVD truncation, so quality drops on that bond.
fn qr_truncate(a: &[f64], m: usize, n: usize, chi: usize) -> Factor {
    let full = m.min(n);
    let keep = full.min(chi.max(1));
    let mut q = vec![0.0; m * full];
    let mut r = vec![0.0; full * n];
    let mut set = vec![false; full];
    let mut work = vec![0.0; m];
    for j in 0..n {
        for i in 0..m {
            work[i] = a[i * n + j];
        }
        for c in 0..full {
            if !set[c] {
                continue;
            }
            let mut dot = 0.0;
            for i in 0..m {
                dot += q[i * full + c] * work[i];
            }
            r[c * n + j] = dot;
            for i in 0..m {
                work[i] -= dot * q[i * full + c];
            }
        }
        let slot = j.min(full);
        if slot < full && !set[slot] {
            let nrm = work.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nrm > 0.0 {
                for i in 0..m {
                    q[i * full + slot] = work[i] / nrm;
                }
                r[slot * n + j] = nrm;
                set[slot] = true;
            }
        }
    }
    for c in 0..full {
        if !set[c] {
            orthonormal_fill(&mut q, m, full, c, &set);
            set[c] = true;
        }
    }
    let mut qk = vec![0.0; m * keep];
    for i in 0..m {
        for c in 0..keep {
            qk[i * keep + c] = q[i * full + c];
        }
    }
    let rk = r[..keep * n].to_vec();
    Factor {
        q: qk,
        carry: rk,
        k: keep,
    }
}

/// Factorize `a` into an isometry and a carry, capping the bond at `chi`.
pub(crate) fn truncated_svd(a: &[f64], m: usize, n: usize, chi: usize) -> Factor {
    truncated_svd_sweeps(a, m, n, chi, JACOBI_MAX_SWEEPS)
}

/// `truncated_svd` with an explicit Jacobi sweep budget. A budget of `0` forces
/// the QR fallback, which is how the fallback path is tested.
pub(crate) fn truncated_svd_sweeps(
    a: &[f64],
    m: usize,
    n: usize,
    chi: usize,
    max_sweeps: usize,
) -> Factor {
    let chi = chi.max(1);
    let full = m.min(n);
    if full == 0 {
        return Factor {
            q: Vec::new(),
            carry: Vec::new(),
            k: 0,
        };
    }
    let Some(svd) = jacobi_svd(a, m, n, max_sweeps) else {
        SVD_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        return qr_truncate(a, m, n, chi);
    };
    let s0 = svd.s.first().copied().unwrap_or(0.0);
    let floor = if s0 > 0.0 { SINGULAR_CUTOFF * s0 } else { 0.0 };
    let counted = svd.s.iter().filter(|&&x| x > floor).count();
    // Never zero: a zero-rank bond would disconnect the chain.
    let keep = counted.clamp(1, full.min(chi));
    let mut q = vec![0.0; m * keep];
    for i in 0..m {
        for b in 0..keep {
            q[i * keep + b] = svd.u[i * svd.k + b];
        }
    }
    let mut carry = vec![0.0; keep * n];
    for b in 0..keep {
        let sb = svd.s[b];
        for j in 0..n {
            carry[b * n + j] = sb * svd.vt[b * n + j];
        }
    }
    Factor { q, carry, k: keep }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use std::sync::{Mutex, MutexGuard};

    /// `SVD_FALLBACKS` is process-global, so a test that measures a delta across
    /// it cannot run beside another test that increments it. Every test that
    /// forces the fallback takes this lock for the duration of the call.
    static FALLBACK_PATH: Mutex<()> = Mutex::new(());

    /// A panic inside one fallback test poisons the lock. Recover the guard, so
    /// the sibling tests report their own results instead of the poisoning.
    fn lock_fallback_path() -> MutexGuard<'static, ()> {
        FALLBACK_PATH.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn random_matrix(m: usize, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..m * n).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect()
    }

    fn reconstruct(svd: &Svd, m: usize, n: usize) -> Vec<f64> {
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for b in 0..svd.k {
                let uv = svd.u[i * svd.k + b] * svd.s[b];
                if uv == 0.0 {
                    continue;
                }
                for j in 0..n {
                    out[i * n + j] += uv * svd.vt[b * n + j];
                }
            }
        }
        out
    }

    fn frob(a: &[f64]) -> f64 {
        a.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Columns of the `rows x cols` row-major matrix are orthonormal.
    fn assert_columns_orthonormal(a: &[f64], rows: usize, cols: usize, what: &str) {
        for p in 0..cols {
            for q in 0..cols {
                let mut dot = 0.0;
                for i in 0..rows {
                    dot += a[i * cols + p] * a[i * cols + q];
                }
                let want = if p == q { 1.0 } else { 0.0 };
                assert!(
                    (dot - want).abs() <= 1e-12,
                    "{what}: column {p} dot column {q} = {dot}, want {want}"
                );
            }
        }
    }

    #[test]
    fn jacobi_svd_reconstructs_and_is_orthonormal() {
        for &(m, n) in &[
            (1, 1),
            (1, 4),
            (4, 1),
            (2, 3),
            (3, 2),
            (5, 5),
            (8, 6),
            (6, 8),
            (16, 32),
            (32, 16),
        ] {
            let a = random_matrix(m, n, 1000 + (m * 97 + n) as u64);
            let svd = jacobi_svd(&a, m, n, JACOBI_MAX_SWEEPS).expect("Jacobi must converge");
            assert_eq!(svd.k, m.min(n));
            assert_eq!(svd.u.len(), m * svd.k);
            assert_eq!(svd.vt.len(), svd.k * n);

            let back = reconstruct(&svd, m, n);
            let scale = frob(&a).max(1.0);
            for i in 0..m * n {
                assert!(
                    (back[i] - a[i]).abs() <= 1e-12 * scale,
                    "{m}x{n} entry {i}: got {} want {}",
                    back[i],
                    a[i]
                );
            }

            assert_columns_orthonormal(&svd.u, m, svd.k, "U");
            let v = transpose(&svd.vt, svd.k, n);
            assert_columns_orthonormal(&v, n, svd.k, "V");
        }
    }

    #[test]
    fn singular_values_are_non_negative_and_sorted_descending() {
        let a = random_matrix(7, 5, 4242);
        let svd = jacobi_svd(&a, 7, 5, JACOBI_MAX_SWEEPS).expect("converge");
        for b in 0..svd.k {
            assert!(svd.s[b] >= 0.0, "s[{b}] = {} is negative", svd.s[b]);
        }
        for w in svd.s.windows(2) {
            assert!(w[0] >= w[1], "singular values not descending: {:?}", svd.s);
        }
    }

    #[test]
    fn sign_convention_puts_the_largest_left_component_positive() {
        let a = random_matrix(6, 4, 777);
        let svd = jacobi_svd(&a, 6, 4, JACOBI_MAX_SWEEPS).expect("converge");
        for b in 0..svd.k {
            let mut best = 0.0f64;
            let mut best_val = 0.0f64;
            for i in 0..6 {
                let x = svd.u[i * svd.k + b];
                if x.abs() > best {
                    best = x.abs();
                    best_val = x;
                }
            }
            assert!(
                best_val > 0.0,
                "column {b}: largest-magnitude component {best_val} must be positive"
            );
        }
    }

    #[test]
    fn zero_matrix_gives_zero_singular_values_and_an_orthonormal_basis() {
        let a = vec![0.0; 12];
        let svd = jacobi_svd(&a, 4, 3, JACOBI_MAX_SWEEPS).expect("converge");
        assert_eq!(svd.k, 3);
        assert!(svd.s.iter().all(|&x| x == 0.0), "{:?}", svd.s);
        assert_columns_orthonormal(&svd.u, 4, 3, "U of the zero matrix");
        let v = transpose(&svd.vt, 3, 3);
        assert_columns_orthonormal(&v, 3, 3, "V of the zero matrix");
    }

    #[test]
    fn rank_deficient_input_puts_the_deficiency_in_the_trailing_values() {
        // A = outer(x, y) has rank 1 exactly.
        let x = [0.3f64, -0.7, 1.1, 0.5];
        let y = [1.0f64, -0.25, 0.6];
        let mut a = vec![0.0; 12];
        for i in 0..4 {
            for j in 0..3 {
                a[i * 3 + j] = x[i] * y[j];
            }
        }
        let svd = jacobi_svd(&a, 4, 3, JACOBI_MAX_SWEEPS).expect("converge");
        assert!(svd.s[0] > 0.5, "leading value {} too small", svd.s[0]);
        assert!(svd.s[1] <= 1e-12 * svd.s[0], "s[1] = {}", svd.s[1]);
        assert!(svd.s[2] <= 1e-12 * svd.s[0], "s[2] = {}", svd.s[2]);
        let back = reconstruct(&svd, 4, 3);
        for i in 0..12 {
            assert!((back[i] - a[i]).abs() <= 1e-12, "entry {i}");
        }
        assert_columns_orthonormal(&svd.u, 4, 3, "U of a rank-1 matrix");
    }

    #[test]
    fn repeated_singular_values_stay_exact() {
        // 2 * I_4: every singular value is 2, which is the hardest tie case for
        // a rotation-based method.
        let mut a = vec![0.0; 16];
        for i in 0..4 {
            a[i * 4 + i] = 2.0;
        }
        let svd = jacobi_svd(&a, 4, 4, JACOBI_MAX_SWEEPS).expect("converge");
        for b in 0..4 {
            assert!((svd.s[b] - 2.0).abs() <= 1e-13, "s[{b}] = {}", svd.s[b]);
        }
        let back = reconstruct(&svd, 4, 4);
        for i in 0..16 {
            assert!((back[i] - a[i]).abs() <= 1e-13, "entry {i}");
        }
        assert_columns_orthonormal(&svd.u, 4, 4, "U");
    }

    #[test]
    fn transpose_round_trips() {
        let a = random_matrix(3, 5, 9);
        let t = transpose(&a, 3, 5);
        assert_eq!(t.len(), 15);
        for i in 0..3 {
            for j in 0..5 {
                assert_eq!(t[j * 3 + i], a[i * 5 + j]);
            }
        }
        assert_eq!(transpose(&t, 5, 3), a);
    }

    #[test]
    fn a_capped_sweep_budget_reports_non_convergence_instead_of_guessing() {
        let a = random_matrix(6, 4, 31337);
        assert!(jacobi_svd(&a, 6, 4, 0).is_none());
    }
    fn product(q: &[f64], carry: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for b in 0..k {
                let qv = q[i * k + b];
                if qv == 0.0 {
                    continue;
                }
                for j in 0..n {
                    out[i * n + j] += qv * carry[b * n + j];
                }
            }
        }
        out
    }

    #[test]
    fn truncated_svd_at_full_rank_reconstructs_exactly() {
        let a = random_matrix(6, 4, 5150);
        let f = truncated_svd(&a, 6, 4, 4);
        assert_eq!(f.k, 4);
        assert_columns_orthonormal(&f.q, 6, 4, "Q");
        let back = product(&f.q, &f.carry, 6, 4, 4);
        for i in 0..24 {
            assert!((back[i] - a[i]).abs() <= 1e-12, "entry {i}");
        }
    }

    #[test]
    fn truncated_svd_caps_the_bond_and_keeps_the_isometry() {
        let a = random_matrix(8, 8, 606);
        let f = truncated_svd(&a, 8, 8, 3);
        assert_eq!(f.k, 3);
        assert_eq!(f.q.len(), 24);
        assert_eq!(f.carry.len(), 24);
        assert_columns_orthonormal(&f.q, 8, 3, "Q");
        // The rank-3 truncation is optimal, so its residual must not exceed the
        // residual of any other rank-3 product. Check it against the discarded
        // singular values, which is exactly that bound.
        let svd = jacobi_svd(&a, 8, 8, JACOBI_MAX_SWEEPS).expect("converge");
        let tail: f64 = svd.s[3..].iter().map(|x| x * x).sum::<f64>().sqrt();
        let back = product(&f.q, &f.carry, 8, 3, 8);
        let mut resid = 0.0;
        for i in 0..64 {
            let d = back[i] - a[i];
            resid += d * d;
        }
        assert!(
            resid.sqrt() <= tail + 1e-10,
            "residual {} exceeds the optimal {tail}",
            resid.sqrt()
        );
    }

    #[test]
    fn truncated_svd_never_returns_a_zero_rank_bond() {
        let a = vec![0.0; 12];
        let f = truncated_svd(&a, 4, 3, 8);
        assert_eq!(f.k, 1);
        assert_columns_orthonormal(&f.q, 4, 1, "Q of the zero matrix");
        assert!(f.carry.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn non_convergence_falls_back_to_qr_and_counts_the_event() {
        let a = random_matrix(6, 4, 99);
        let _guard = lock_fallback_path();
        let before = svd_fallback_count();
        // A zero sweep budget forces the fallback deterministically; a
        // pathological matrix would be a flaky way to reach the same path.
        let f = truncated_svd_sweeps(&a, 6, 4, 4, 0);
        assert_eq!(svd_fallback_count(), before + 1);
        assert_eq!(f.k, 4);
        assert_columns_orthonormal(&f.q, 6, 4, "Q of the QR fallback");
        // At full rank the QR fallback is exact, so quality only drops when the
        // bond is also capped.
        let back = product(&f.q, &f.carry, 6, 4, 4);
        for i in 0..24 {
            assert!((back[i] - a[i]).abs() <= 1e-12, "entry {i}");
        }
    }

    #[test]
    fn the_qr_fallback_still_produces_an_isometry_when_the_bond_is_capped() {
        let a = random_matrix(8, 6, 1234);
        let _guard = lock_fallback_path();
        let f = truncated_svd_sweeps(&a, 8, 6, 2, 0);
        assert_eq!(f.k, 2);
        assert_columns_orthonormal(&f.q, 8, 2, "Q of the capped QR fallback");
        let back = product(&f.q, &f.carry, 8, 2, 6);
        assert!(back.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn the_qr_fallback_handles_a_linearly_dependent_column() {
        // Column 2 is column 0 twice over, so its residual is zero and the
        // fallback must fill that slot from the basis instead of leaving it.
        let mut a = vec![0.0; 12];
        let col0 = [1.0f64, 0.0, 0.0, 0.0];
        let col1 = [0.0f64, 2.0, 0.0, 0.0];
        for i in 0..4 {
            a[i * 3] = col0[i];
            a[i * 3 + 1] = col1[i];
            a[i * 3 + 2] = 2.0 * col0[i];
        }
        let _guard = lock_fallback_path();
        let f = truncated_svd_sweeps(&a, 4, 3, 3, 0);
        assert_eq!(f.k, 3);
        assert_columns_orthonormal(&f.q, 4, 3, "Q with a dependent column");
        let back = product(&f.q, &f.carry, 4, 3, 3);
        for i in 0..12 {
            assert!(
                (back[i] - a[i]).abs() <= 1e-12,
                "entry {i}: {} vs {}",
                back[i],
                a[i]
            );
        }
    }
}
