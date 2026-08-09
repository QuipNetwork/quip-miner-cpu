//! Annealing schedule: how `num_sweeps`, `sweeps_per_beta`, and `beta_range`
//! become imaginary-time increments and a transverse-field ramp.
//!
//! The beta ladder is the same `quip_miner_core::beta` ladder the SA kernel
//! uses, so `beta_range` means the same thing for every CPU backend. Imaginary
//! time steps are the increments of that ladder, `eta_k = beta_k - beta_(k-1)`
//! with `beta_0 = 0`, so the total imaginary time equals the final beta.
//!
//! With `sweeps_per_beta > 1` the ladder is held for that many Trotter steps,
//! so the held steps carry `eta = 0`. The anneal loop skips them: a zero
//! increment makes every gate the identity, so running one would cost time and
//! change nothing.

use quip_miner_core::beta::{default_ising_beta_range, geometric_beta_schedule};
use quip_miner_core::{IsingGraph, SampleParams};

/// Trotter-step cap. Imaginary-time evolution reaches the ground state of a
/// 30-spin transverse-field model in tens of steps, so spending thousands would
/// force the bond dimension down to stay inside the flop budget and trade a
/// useful parameter for a useless one.
pub(crate) const MAX_TROTTER_STEPS: usize = 256;

/// Polish floor. In the prototype the greedy descent closed more of the gap to
/// the optimum than raising the bond dimension from 1 to 16 did, so it runs
/// even when `num_sweeps` leaves no remainder after the Trotter split.
pub(crate) const MIN_POLISH_SWEEPS: usize = 8;

/// Split `num_sweeps` into `(trotter_steps, polish_sweeps)`.
pub(crate) fn split_sweeps(num_sweeps: usize) -> (usize, usize) {
    let steps = num_sweeps.clamp(1, MAX_TROTTER_STEPS);
    let polish = num_sweeps.saturating_sub(steps).max(MIN_POLISH_SWEEPS);
    (steps, polish)
}

/// `Gamma_0 = 2 max_i (|h_i| + sum_j |J_ij|)`.
///
/// This is twice the per-variable effective-field magnitude that
/// `default_ising_beta_range` already maximizes over, so the transverse ramp
/// and the beta ladder stay on the same scale. It skips the same terms the
/// kernel skips.
pub(crate) fn gamma_zero(graph: &IsingGraph) -> f64 {
    let n = graph.h.len();
    if n == 0 {
        return 0.0;
    }
    let mut acc = vec![0.0f64; n];
    for (i, &hi) in graph.h.iter().enumerate() {
        if hi.is_finite() {
            acc[i] += hi.abs();
        }
    }
    for (k, &(u, v)) in graph.edges.iter().enumerate() {
        if u >= n || v >= n || u == v {
            continue;
        }
        let j = graph.j.get(k).copied().unwrap_or(0.0);
        if !j.is_finite() {
            continue;
        }
        acc[u] += j.abs();
        acc[v] += j.abs();
    }
    2.0 * acc.iter().copied().fold(0.0f64, f64::max)
}

/// Per-step imaginary time and transverse field strength.
pub(crate) struct Schedule {
    /// Imaginary-time increment per Trotter step.
    pub(crate) eta: Vec<f64>,
    /// Transverse field strength per Trotter step, ending at zero.
    pub(crate) gamma: Vec<f64>,
}

/// Build the per-step schedule for `steps` Trotter steps.
pub(crate) fn build(graph: &IsingGraph, params: &SampleParams, steps: usize) -> Schedule {
    let steps = steps.max(1);
    let per = params.sweeps_per_beta.max(1);
    let num_betas = (steps / per).max(1);
    let (hot, cold) = params
        .beta_range
        .unwrap_or_else(|| default_ising_beta_range(graph));
    let ladder = geometric_beta_schedule(hot, cold, num_betas);
    let g0 = gamma_zero(graph);
    let denom = (steps - 1).max(1) as f64;

    let mut eta = Vec::with_capacity(steps);
    let mut gamma = Vec::with_capacity(steps);
    let mut prev = 0.0f64;
    for k in 0..steps {
        let rung = (k / per).min(num_betas - 1);
        let beta = ladder.get(rung).copied().unwrap_or(0.0);
        eta.push((beta - prev).max(0.0));
        prev = beta;
        gamma.push(if steps <= 1 {
            0.0
        } else {
            g0 * (1.0 - k as f64 / denom)
        });
    }
    Schedule { eta, gamma }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain4() -> IsingGraph {
        IsingGraph::new(
            vec![0.5, -0.25, 0.0, 1.0],
            vec![-1.0, 0.5, 2.0],
            vec![(0, 1), (1, 2), (2, 3)],
        )
    }

    #[test]
    fn split_sweeps_caps_the_trotter_phase_and_always_polishes() {
        assert_eq!(split_sweeps(0), (1, MIN_POLISH_SWEEPS));
        assert_eq!(split_sweeps(1), (1, MIN_POLISH_SWEEPS));
        assert_eq!(split_sweeps(64), (64, MIN_POLISH_SWEEPS));
        assert_eq!(split_sweeps(256), (256, MIN_POLISH_SWEEPS));
        assert_eq!(split_sweeps(1024), (256, 768));
        assert_eq!(split_sweeps(4096), (256, 3840));
    }

    #[test]
    fn gamma_zero_is_twice_the_largest_per_variable_field_magnitude() {
        let g = chain4();
        // node 0: |0.5| + |-1.0| = 1.5
        // node 1: |-0.25| + |-1.0| + |0.5| = 1.75
        // node 2: |0.0| + |0.5| + |2.0| = 2.5
        // node 3: |1.0| + |2.0| = 3.0  -> largest
        assert!((gamma_zero(&g) - 6.0).abs() < 1e-12);
    }

    #[test]
    fn gamma_zero_skips_the_terms_the_kernel_skips() {
        let g = IsingGraph::new(
            vec![1.0, f64::NAN],
            vec![2.0, 5.0, f64::INFINITY],
            vec![(0, 0), (0, 9), (0, 1)],
        );
        // Only edge (0,1) counts, and it carries the third coupling, which is
        // non-finite and therefore skipped. Node 0 keeps |h| = 1.
        assert!((gamma_zero(&g) - 2.0).abs() < 1e-12);
        assert_eq!(gamma_zero(&IsingGraph::new(vec![], vec![], vec![])), 0.0);
    }

    #[test]
    fn imaginary_time_increments_sum_to_the_final_beta() {
        let g = chain4();
        let params = SampleParams {
            num_sweeps: 32,
            sweeps_per_beta: 1,
            beta_range: Some((0.1, 4.0)),
            ..Default::default()
        };
        let s = build(&g, &params, 32);
        assert_eq!(s.eta.len(), 32);
        assert_eq!(s.gamma.len(), 32);
        let total: f64 = s.eta.iter().sum();
        assert!(
            (total - 4.0).abs() < 1e-12,
            "total imaginary time {total} must equal the final beta"
        );
        assert!(s.eta.iter().all(|&e| e >= 0.0));
    }

    #[test]
    fn the_transverse_field_ramps_linearly_to_zero() {
        let g = chain4();
        let params = SampleParams {
            num_sweeps: 8,
            sweeps_per_beta: 1,
            ..Default::default()
        };
        let s = build(&g, &params, 8);
        let g0 = gamma_zero(&g);
        assert!((s.gamma[0] - g0).abs() < 1e-12);
        assert!(s.gamma[7].abs() < 1e-12, "field must reach zero");
        for w in s.gamma.windows(2) {
            assert!(w[0] > w[1], "field must decrease: {:?}", s.gamma);
        }
        let mid = g0 * (1.0 - 4.0 / 7.0);
        assert!((s.gamma[4] - mid).abs() < 1e-12);
    }

    #[test]
    fn one_step_carries_no_transverse_field() {
        let g = chain4();
        let params = SampleParams {
            num_sweeps: 1,
            ..Default::default()
        };
        let s = build(&g, &params, 1);
        assert_eq!(s.eta.len(), 1);
        assert_eq!(s.gamma, vec![0.0]);
    }

    #[test]
    fn sweeps_per_beta_holds_the_ladder_for_that_many_steps() {
        let g = chain4();
        let params = SampleParams {
            num_sweeps: 12,
            sweeps_per_beta: 4,
            beta_range: Some((0.5, 4.0)),
            ..Default::default()
        };
        let s = build(&g, &params, 12);
        assert_eq!(s.eta.len(), 12);
        // Three rungs of four steps: the first step of each rung carries the
        // whole increment and the held steps carry zero.
        for rung in 0..3 {
            assert!(s.eta[rung * 4] > 0.0, "rung {rung} must advance beta");
            for held in 1..4 {
                assert!(
                    s.eta[rung * 4 + held] == 0.0,
                    "held step {held} of rung {rung} must not advance beta"
                );
            }
        }
        let total: f64 = s.eta.iter().sum();
        assert!((total - 4.0).abs() < 1e-12);
    }

    #[test]
    fn an_auto_beta_range_produces_a_positive_increasing_ladder() {
        let g = chain4();
        let params = SampleParams {
            num_sweeps: 16,
            ..Default::default()
        };
        let s = build(&g, &params, 16);
        let total: f64 = s.eta.iter().sum();
        assert!(total > 0.0 && total.is_finite(), "total = {total}");
        assert!(s.eta.iter().all(|&e| e.is_finite() && e >= 0.0));
    }
}
