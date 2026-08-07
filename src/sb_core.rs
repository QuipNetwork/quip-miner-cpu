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

use quip_miner_core::IsingGraph;

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

/// Per-particle neighbor lists in CSR layout, in `Real` precision.
///
/// Built from the base [`IsingGraph`] with the same defensive posture as
/// `sampler_core::CpuGraph`: edges out of range for `h.len()` are skipped,
/// self-loops `(u, u)` are skipped, and couplings shorter than the edge list
/// read 0.
///
/// This deliberately does not reuse `quip_miner_core::CsrGraph`, whose
/// `from_base` keeps self-loops. A self-loop in an SB neighbor row would inject
/// a spurious self-force `-c0 j_uu sgn(x_u)` into that node and shift its
/// bifurcation, while `energy_milli` scores the loop as an unoptimizable
/// constant. `sb_graph_matches_cpu_graph_adjacency` guards the duplication.
///
/// Biases live outside the CSR rows because they reach the force through the
/// ancilla column rather than through a neighbor row.
struct SbGraph {
    /// Linear biases, one per node, narrowed for the hot loop.
    h: Vec<Real>,
    /// CSR row offsets, length `n + 1`.
    nbr_start: Vec<u32>,
    /// Flattened neighbor node ids.
    nbr_node: Vec<u32>,
    /// Flattened couplings, parallel to `nbr_node`.
    nbr_coup: Vec<Real>,
    /// True when any `h_i != 0`. The ancilla exists only then.
    has_bias: bool,
}

impl SbGraph {
    fn from_base(g: &IsingGraph) -> Self {
        let n = g.h.len();
        let mut deg = vec![0u32; n];
        for &(u, v) in &g.edges {
            if u >= n || v >= n || u == v {
                continue;
            }
            deg[u] += 1;
            deg[v] += 1;
        }
        let mut nbr_start = vec![0u32; n + 1];
        for i in 0..n {
            nbr_start[i + 1] = nbr_start[i] + deg[i];
        }
        let total = nbr_start[n] as usize;
        let mut nbr_node = vec![0u32; total];
        let mut nbr_coup: Vec<Real> = vec![0.0; total];
        let mut cursor: Vec<u32> = nbr_start[..n].to_vec();
        for (k, &(u, v)) in g.edges.iter().enumerate() {
            if u >= n || v >= n || u == v {
                continue;
            }
            let coup = g.j.get(k).copied().unwrap_or(0.0) as Real;
            let pu = cursor[u] as usize;
            nbr_node[pu] = v as u32;
            nbr_coup[pu] = coup;
            cursor[u] += 1;
            let pv = cursor[v] as usize;
            nbr_node[pv] = u as u32;
            nbr_coup[pv] = coup;
            cursor[v] += 1;
        }
        Self {
            h: g.h.iter().map(|&b| b as Real).collect(),
            nbr_start,
            nbr_node,
            nbr_coup,
            has_bias: g.h.iter().any(|&b| b != 0.0),
        }
    }

    fn num_nodes(&self) -> usize {
        self.h.len()
    }

    /// `(neighbor_ids, couplings)` slices for `var`.
    #[inline]
    fn neighbors(&self, var: usize) -> (&[u32], &[Real]) {
        let s = self.nbr_start[var] as usize;
        let e = self.nbr_start[var + 1] as usize;
        (&self.nbr_node[s..e], &self.nbr_coup[s..e])
    }
}

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

    /// Self-loop, out-of-range edge, and a `j` vector shorter than `edges`. All
    /// three defensive cases in one graph.
    fn adversarial() -> IsingGraph {
        IsingGraph::new(
            vec![0.5, -0.25, 0.0],
            vec![2.0, -1.0, 0.75],
            vec![(0, 0), (0, 1), (9, 2), (1, 2)],
        )
    }

    /// Hypothesis: `SbGraph` repeats `CpuGraph`'s CSR construction rather than
    /// reusing `quip_miner_core::CsrGraph`, whose `from_base` keeps self-loops.
    /// A self-loop in an SB neighbor row injects a spurious self-force
    /// `-c0 j_uu sgn(x_u)` that biases that node's bifurcation, while
    /// `energy_milli` scores the loop as an unoptimizable constant. This test
    /// replaces hoisting a shared builder: it pins the duplicate against the
    /// original.
    ///
    /// The fixture's couplings are dyadic, so narrowing them to f32 and widening
    /// back to f64 is exact and the comparison is meaningful.
    #[test]
    fn sb_graph_matches_cpu_graph_adjacency() {
        let g = adversarial();
        let sb = SbGraph::from_base(&g);
        let cpu = crate::sampler_core::CpuGraph::from_base(&g);
        assert_eq!(sb.num_nodes(), cpu.num_nodes());
        for v in 0..sb.num_nodes() {
            let (sb_nodes, sb_coups) = sb.neighbors(v);
            let (cpu_nodes, cpu_coups) = cpu.neighbors(v);
            assert_eq!(sb_nodes, cpu_nodes, "neighbor ids differ at node {v}");
            let widened: Vec<f64> = sb_coups.iter().map(|&c| f64::from(c)).collect();
            assert_eq!(widened, cpu_coups, "couplings differ at node {v}");
        }
    }

    /// Hypothesis: the CSR drops a self-loop and an out-of-range edge outright,
    /// and a missing coupling reads 0.0 rather than shifting the edge list.
    #[test]
    fn sb_graph_drops_self_loops_and_out_of_range_edges() {
        let sb = SbGraph::from_base(&adversarial());
        assert_eq!(sb.neighbors(0).0, &[1]);
        assert_eq!(sb.neighbors(0).1, &[-1.0]);
        assert_eq!(sb.neighbors(1).0, &[0, 2]);
        assert_eq!(sb.neighbors(1).1, &[-1.0, 0.0]);
        assert_eq!(sb.neighbors(2).0, &[1]);
        assert_eq!(sb.neighbors(2).1, &[0.0]);
        assert!(sb.has_bias, "h = [0.5, -0.25, 0.0] carries a bias");
    }

    /// Hypothesis: an all-zero bias vector means no ancilla.
    #[test]
    fn zero_biases_mean_no_ancilla() {
        let g = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
        assert!(!SbGraph::from_base(&g).has_bias);
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
