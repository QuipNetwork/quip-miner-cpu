//! Simulated Bifurcation kernels: dSB, bSB, HdSB and HbSB over one integrator.
//!
//! The kernel evolves positions `x_i` and momenta `y_i` with symplectic Euler
//! steps. The momentum updates first, from the old positions. The position then
//! updates from the new momentum. The inelastic wall clip runs after the
//! position update. When the variant carries a heating rate, that term is added
//! last and reads the momentum captured before the step, which is the
//! integration order Kanao and Goto tuned.
//!
//! Sign convention: the SB papers minimize `-(1/2) Σ_ij J_ij s_i s_j` with both
//! index orders summed, while quip minimizes `Σ_i h_i s_i + Σ_k j_k s_u s_v`
//! with each edge listed once. Every entry of the SB matrix is therefore the
//! negated quip value: `J_uv = -j_k` and `J_{i,N} = -h_i` on the ancilla
//! column. The ancilla lives at index `N` in the state arrays only; the
//! returned spin vector has length `N`, so
//! [`quip_protocol::scoring::energy_milli`] scores the original graph unchanged.
//!
//! Parameter mapping:
//! - `num_sweeps` is the time-step count, mapped 1:1, and also sets the pump
//!   schedule `a(t_k) = (k + 1) / num_sweeps`. The step count and the schedule
//!   cannot be tuned independently: the slow ramp is the mechanism.
//! - `num_reads` is the number of independent trajectories.
//! - `seed` is the base seed; the per-read derivation matches `sampler_core`.
//! - `sweeps_per_beta` is **ignored**. SB has no temperature.
//! - `beta_range` is **ignored**, for the same reason. The control parameter is
//!   the pump `a(t)`, whose schedule is fixed by `a0 = 1` and the step count.
//!
//! Randomness enters only at initialization. After `x` and `y` are drawn the
//! dynamics are deterministic for every variant, including the heated ones, so
//! read diversity comes entirely from the initial condition.

/// Internal state precision. One line changes the whole kernel to f64 for a
/// benchmark build; the reported energy never depends on it, because the
/// sampler returns `sign(x)` as `i8` and `energy_milli` rescores in f64.
type Real = f32;

/// Time step. The 2021 paper searched `{0.25, 0.5, 0.75, 1.0, 1.25}` and used
/// 1.0 or 1.25 for the ballistic and discrete variants.
const DT: Real = 1.0;

/// Detuning `a0`, set to 1 in every source. With `a0 = 1` the position update
/// reduces to `x += y * dt`.
const A0: Real = 1.0;

/// Initial draw is `x, y ~ U(-INIT_RANGE, INIT_RANGE)` per particle per read.
/// The papers say only "randomly set around zero"; 0.1 keeps particles away
/// from the walls during the early low-pump phase.
const INIT_RANGE: Real = 0.1;

/// Form of the coupling term in the SB force.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coupling {
    /// `g(x) = sgn(x)`. Used by dSB and HdSB.
    Discrete,
    /// `g(x) = x`. Used by bSB and HbSB.
    Continuous,
}

/// One SB variant: the coupling form and the heating rate `γ`.
///
/// `γ = 0` means no heating. The nonzero values are the K2000-tuned constants
/// from Kanao and Goto 2022, fixed per variant before any tuning starts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SbVariant {
    /// Coupling form for the force term.
    pub coupling: Coupling,
    /// Heating rate. Zero disables the heating step entirely.
    pub gamma: f32,
}

/// Discrete SB. The production-track variant, shipped as `quip-cpu-sb`.
pub const DSB: SbVariant = SbVariant {
    coupling: Coupling::Discrete,
    gamma: 0.0,
};

/// Ballistic SB.
pub const BSB: SbVariant = SbVariant {
    coupling: Coupling::Continuous,
    gamma: 0.0,
};

/// Heated discrete SB.
pub const HDSB: SbVariant = SbVariant {
    coupling: Coupling::Discrete,
    gamma: 0.06,
};

/// Heated ballistic SB.
pub const HBSB: SbVariant = SbVariant {
    coupling: Coupling::Continuous,
    gamma: 0.5,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Hypothesis: the four shipped variants are exactly the coupling form and
    /// heating rate the design table names. A later plan builds three binaries
    /// straight off these constants, so a silent edit here would mislabel a
    /// backend on the wire.
    #[test]
    fn variant_constants_match_the_published_table() {
        assert_eq!(
            DSB,
            SbVariant {
                coupling: Coupling::Discrete,
                gamma: 0.0
            }
        );
        assert_eq!(
            BSB,
            SbVariant {
                coupling: Coupling::Continuous,
                gamma: 0.0
            }
        );
        assert_eq!(
            HDSB,
            SbVariant {
                coupling: Coupling::Discrete,
                gamma: 0.06
            }
        );
        assert_eq!(
            HBSB,
            SbVariant {
                coupling: Coupling::Continuous,
                gamma: 0.5
            }
        );
    }

    /// Hypothesis: the integrator constants are the paper values. The
    /// determinism fixture and the stuck-at-wall guard both depend on them.
    #[test]
    fn integrator_constants_are_the_paper_values() {
        assert_eq!(DT, 1.0);
        assert_eq!(A0, 1.0);
        assert_eq!(INIT_RANGE, 0.1);
    }
}
