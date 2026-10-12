//! Lane-batched float64 multi-spin annealing (`cpu-msa-f64`).
//!
//! The research kernel for problems whose couplings and fields are not
//! `±1` and whole numbers. It keeps the packed spin layout of
//! [`crate::sa_msc`], one `u64` per node holding the same spin across 64
//! replicas, and gives up the bit-sliced satisfied-bond counter, which has no
//! form for a float sum. Every node visit instead materialises 64 float64
//! local fields, one per lane, from the packed neighbour words, and runs the
//! Metropolis test per lane. Supplied coefficients are used as they are, with
//! no quantisation, and reported energies are recomputed in float64 from the
//! original model.
//!
//! # The update
//!
//! For `H = Σ h_i s_i + Σ J_uv s_u s_v`, a flip of `s_i` changes the energy by
//! `Δ = -2 s_i (h_i + Σ_j J_ij s_j)`. The neighbour sum is accumulated per
//! lane by adding `J_ij` with its sign bit flipped where the neighbour's lane
//! bit is set (bit set means `s = -1`). Multiplying by `±1` is exact, so this
//! matches a from-scratch [`crate::sampler_core::effective_field`]
//! recomputation for the same CSR neighbour order. The scalar hot loop instead
//! reads an incrementally updated field cache, which can drift from that
//! recomputation due to IEEE rounding.
//! A lane accepts when `Δ ≤ 0`, or when `ln u < -β Δ` for one uniform `u`
//! drawn per uphill candidate. Accepted lanes come back as a mask and the flip is
//! one XOR into the node's word.
//!
//! # Randomness: `per-lane-draw-v1`
//!
//! Every replica word owns one `SmallRng` stream, seeded from the job seed and
//! the word index. At each node visit, each lane with `Δ > 0` draws its own
//! open-interval uniform from that stream, in lane order. A lane with
//! `Δ ≤ 0` accepts without a draw. Two lanes that reach the same spins see
//! different draws, so they separate again, which a shared draw cannot do
//! (QUI-1596). Read `r` depends only on word `r / 64`, so a longer run's
//! leading reads equal a shorter run's reads. Cold lanes start from the same
//! per-word stream as the unit kernel's.
//!
//! # Limits
//!
//! The kernel validates before it allocates and refuses rather than
//! substituting another sampler. Node and edge counts are the crate's
//! protocol limits; reads are `1..=MAX_READS`; every coefficient is finite
//! and, when nonzero, a normal float64; `Σ|h| + Σ|J| ≤ MAX_COEFFICIENT_SUM`;
//! every beta, including a seeded start, is finite, positive and at most
//! `MAX_BETA`. Under those bounds no local field or acceptance product can
//! leave the finite range, so the hot loop carries no finiteness check. The
//! edge list must have no self-loop, no endpoint outside the problem and no
//! repeated undirected pair. Workspace, counted with checked arithmetic
//! before allocation, is capped at `WORKSPACE_CAP_BYTES`.

use quip_solver_core::beta::default_ising_beta_range;
use quip_solver_core::{CancelToken, IsingGraph, SampleParams};
use rand::distributions::Open01;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::mem::size_of;

use crate::coloring::Coloring;
use crate::sa_msc::{MscState, LANES};
use crate::sa_sampler::{read_rng, SeededStart};
use crate::sampler_core::{
    build_beta_schedule, build_seeded_beta_schedule, effective_sweeps_per_beta, CpuGraph,
};
use crate::{DEFAULT_MAX_EDGES, DEFAULT_MAX_NODES};

/// Most reads one call may ask for.
pub(crate) const MAX_READS: usize = 4096;

/// Largest `Σ|h| + Σ|J|` accepted. With [`MAX_BETA`] it keeps every
/// `β Δ` finite.
pub(crate) const MAX_COEFFICIENT_SUM: f64 = 1e100;

/// Largest inverse temperature accepted, at either end or as a seeded start.
pub(crate) const MAX_BETA: f64 = 1e100;

/// Cap on the kernel's own allocations, excluding the caller's arrays. A
/// 2M-node model with 20M edges and 64 reads needs about 0.6 GiB.
pub(crate) const WORKSPACE_CAP_BYTES: usize = 8 << 30;

/// Name of the randomness scheme this kernel implements.
pub(crate) const RNG_SCHEME: &str = "per-lane-draw-v1";

/// Salt for the per-word acceptance streams.
const LANE_SALT: u64 = 0x4D53_415F_4C41_4E45;

/// The acceptance stream of replica word `word`.
fn lane_stream(seed: u64, word: usize) -> SmallRng {
    let mut x = (word as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    SmallRng::seed_from_u64(seed ^ LANE_SALT ^ x ^ (x >> 31))
}

/// Why a request was refused or stopped.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum FloatMsaError {
    /// The request violates a documented bound; the message names which.
    InvalidInput(String),
    /// The kernel's workspace would exceed [`WORKSPACE_CAP_BYTES`].
    /// `needed` is `usize::MAX` when the count itself overflowed.
    MemoryLimit { needed: usize, cap: usize },
    /// The cancel token fired before the run finished.
    Cancelled,
}

impl std::fmt::Display for FloatMsaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(msg) => f.write_str(msg),
            Self::MemoryLimit { needed, cap } => {
                write!(
                    f,
                    "the run needs {needed} bytes of workspace; the cap is {cap}"
                )
            }
            Self::Cancelled => f.write_str("the run was cancelled"),
        }
    }
}

impl std::error::Error for FloatMsaError {}

/// One call's output, in read order.
#[derive(Debug)]
pub(crate) struct FloatMsaSamples {
    /// `num_reads` states of `±1` spins.
    pub(crate) spins: Vec<Vec<i8>>,
    /// Float64 energy of each state under the original model.
    pub(crate) energies: Vec<f64>,
    /// How many leading reads started from a supplied state.
    pub(crate) seeded_reads: usize,
    /// Estimated persistent workspace counted against [`WORKSPACE_CAP_BYTES`].
    /// It excludes caller-owned arrays, validation-pair vectors, the beta
    /// ladder, CSR/coloring construction buffers, seeded lane-reference
    /// vectors, and vector headers for states and output rows.
    pub(crate) workspace_bytes: usize,
    /// Exact beta values, in order, that this run annealed with.
    pub(crate) beta_ladder: Vec<f64>,
}

/// `Σ h_i s_i + Σ_k J_k s_u s_v` in float64, in node then edge order.
///
/// Expects a graph whose edges index into `spins`; the research entry points
/// validate that before calling.
pub(crate) fn energy_f64(spins: &[i8], graph: &IsingGraph) -> f64 {
    let mut energy = 0.0;
    for (&s, &h) in spins.iter().zip(&graph.h) {
        energy += h * f64::from(s);
    }
    for (k, &(u, v)) in graph.edges.iter().enumerate() {
        energy += graph.j[k] * f64::from(spins[u]) * f64::from(spins[v]);
    }
    energy
}

/// A validated request: the ladder to run and the workspace it costs.
struct Plan {
    betas: Vec<f64>,
    words: usize,
    workspace_bytes: usize,
}

fn invalid<T>(msg: String) -> Result<T, FloatMsaError> {
    Err(FloatMsaError::InvalidInput(msg))
}

pub(crate) fn check_size(nodes: usize, edges: usize) -> Result<(), FloatMsaError> {
    if nodes == 0 {
        return invalid("the problem has no nodes".to_owned());
    }
    if nodes > DEFAULT_MAX_NODES as usize {
        return invalid(format!(
            "{nodes} nodes exceed the limit of {DEFAULT_MAX_NODES}"
        ));
    }
    if edges > DEFAULT_MAX_EDGES as usize {
        return invalid(format!(
            "{edges} edges exceed the limit of {DEFAULT_MAX_EDGES}"
        ));
    }
    Ok(())
}

fn check_coefficient(name: &str, index: usize, value: f64) -> Result<(), FloatMsaError> {
    if !value.is_finite() {
        return invalid(format!("{name}[{index}] is not finite"));
    }
    if value != 0.0 && !value.is_normal() {
        return invalid(format!("{name}[{index}] is subnormal"));
    }
    Ok(())
}

pub(crate) fn check_beta(name: &str, beta: f64) -> Result<(), FloatMsaError> {
    if !beta.is_finite() || beta <= 0.0 {
        return invalid(format!(
            "{name} beta must be finite and above zero; got {beta}"
        ));
    }
    if beta > MAX_BETA {
        return invalid(format!("{name} beta {beta} exceeds the cap of {MAX_BETA}"));
    }
    Ok(())
}

/// Bytes the run allocates: CSR adjacency, packed words, output rows and
/// energies, the colouring, per-word acceptance streams, and the per-visit
/// field scratch.
fn workspace_bytes(nodes: usize, edges: usize, reads: usize, words: usize) -> Option<usize> {
    let csr = nodes
        .checked_mul(12)?
        .checked_add(4)?
        .checked_add(edges.checked_mul(24)?)?;
    let packed = words.checked_mul(nodes)?.checked_mul(8)?;
    let output = reads
        .checked_mul(nodes)?
        .checked_add(reads.checked_mul(8)?)?;
    let coloring = nodes.checked_mul(8)?;
    let streams = words.checked_mul(size_of::<SmallRng>())?;
    csr.checked_add(packed)?
        .checked_add(output)?
        .checked_add(coloring)?
        .checked_add(streams)?
        .checked_add(LANES * 8)
}

pub(crate) fn check_workspace(
    nodes: usize,
    edges: usize,
    reads: usize,
) -> Result<usize, FloatMsaError> {
    let needed = workspace_bytes(nodes, edges, reads, reads.div_ceil(LANES)).unwrap_or(usize::MAX);
    if needed > WORKSPACE_CAP_BYTES {
        return Err(FloatMsaError::MemoryLimit {
            needed,
            cap: WORKSPACE_CAP_BYTES,
        });
    }
    Ok(needed)
}

fn validate(
    graph: &IsingGraph,
    params: &SampleParams,
    start: Option<SeededStart<'_>>,
) -> Result<Plan, FloatMsaError> {
    let nodes = graph.h.len();
    let edges = graph.edges.len();
    check_size(nodes, edges)?;
    if graph.j.len() != edges {
        return invalid(format!("{} couplings for {edges} edges", graph.j.len()));
    }
    let reads = params.num_reads;
    if !(1..=MAX_READS).contains(&reads) {
        return invalid(format!("num_reads must be in 1..={MAX_READS}; got {reads}"));
    }
    let needed = check_workspace(nodes, edges, reads)?;

    let mut sum = 0.0f64;
    for (i, &h) in graph.h.iter().enumerate() {
        check_coefficient("h", i, h)?;
        sum += h.abs();
    }
    for (k, &j) in graph.j.iter().enumerate() {
        check_coefficient("J", k, j)?;
        sum += j.abs();
    }
    if sum > MAX_COEFFICIENT_SUM {
        return invalid(format!(
            "sum |h| + sum |J| = {sum} exceeds the cap of {MAX_COEFFICIENT_SUM}"
        ));
    }

    let mut pairs = Vec::with_capacity(edges);
    for (k, &(u, v)) in graph.edges.iter().enumerate() {
        if u >= nodes || v >= nodes {
            return invalid(format!("edge {k} has an endpoint outside 0..{nodes}"));
        }
        if u == v {
            return invalid(format!("edge {k} is a self-loop on node {u}"));
        }
        pairs.push((u.min(v), u.max(v)));
    }
    pairs.sort_unstable();
    for w in pairs.windows(2) {
        if w[0] == w[1] {
            let (u, v) = w[0];
            return invalid(format!("edge ({u}, {v}) is repeated"));
        }
    }

    let (hot, cold) = params
        .beta_range
        .unwrap_or_else(|| default_ising_beta_range(graph));
    check_beta("hot", hot)?;
    check_beta("cold", cold)?;
    if hot > cold {
        return invalid(format!(
            "beta_range runs backwards: hot {hot} is above cold {cold}"
        ));
    }
    if let Some(s) = start {
        if let Some(beta) = s.start_beta {
            check_beta("start", beta)?;
        }
        if s.spins.len() > reads {
            return invalid(format!("{} seeds for {reads} reads", s.spins.len()));
        }
        for (i, state) in s.spins.iter().enumerate() {
            if state.len() != nodes {
                return invalid(format!(
                    "seed {i} has {} spins; the problem has {nodes} nodes",
                    state.len()
                ));
            }
            if state.iter().any(|&v| v != 1 && v != -1) {
                return invalid(format!("seed {i} holds a value other than -1 or +1"));
            }
        }
    }

    let words = reads.div_ceil(LANES);
    // The validated endpoints are the ones the ladder is built from.
    let resolved = SampleParams {
        beta_range: Some((hot, cold)),
        ..params.clone()
    };
    let betas = match start {
        Some(s) => build_seeded_beta_schedule(graph, &resolved, s.start_beta),
        None => build_beta_schedule(graph, &resolved),
    };
    Ok(Plan {
        betas,
        words,
        workspace_bytes: needed,
    })
}

fn check_cancel(cancel: Option<(&CancelToken, Option<u64>)>) -> Result<(), FloatMsaError> {
    if let Some((guard, watermark)) = cancel {
        if guard.is_cancelled(watermark) {
            return Err(FloatMsaError::Cancelled);
        }
    }
    Ok(())
}

/// Anneal `params.num_reads` replicas of `graph`, cold or from `start`.
///
/// Read `r` is lane `r % 64` of word `r / 64`. With `start`, read `r` begins
/// from `start.spins[r]` and later reads begin where a cold run's would. Zero
/// sweeps return the start states unchanged.
///
/// # Errors
///
/// [`FloatMsaError::InvalidInput`] and [`FloatMsaError::MemoryLimit`] before
/// any allocation; [`FloatMsaError::Cancelled`] when the token fires before
/// allocation, at a sweep boundary, or while unpacking.
pub(crate) fn sample_float_msa(
    graph: &IsingGraph,
    params: &SampleParams,
    start: Option<SeededStart<'_>>,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<FloatMsaSamples, FloatMsaError> {
    let plan = validate(graph, params, start)?;
    check_cancel(cancel)?;

    let nodes = graph.h.len();
    let reads = params.num_reads;
    let cpu = CpuGraph::from_base(graph);
    let colors = Coloring::new(&cpu);
    let seeds: &[Vec<i8>] = start.map_or(&[], |s| s.spins);

    let mut states = Vec::with_capacity(plan.words);
    for read in (0..reads).step_by(LANES) {
        let mut rng = read_rng(params.seed, read);
        let word_seeds: Vec<&[i8]> = seeds
            .iter()
            .skip(read)
            .take(LANES)
            .map(Vec::as_slice)
            .collect();
        states.push(if word_seeds.is_empty() {
            MscState::random(nodes, &mut rng)
        } else {
            MscState::seeded(nodes, &word_seeds, &mut rng)
        });
    }

    if params.num_sweeps > 0 {
        anneal(
            &cpu,
            &colors,
            &plan.betas,
            params,
            &mut states,
            cancel,
            || {},
        )?;
    }

    let mut spins = Vec::with_capacity(reads);
    let mut energies = Vec::with_capacity(reads);
    for (word, state) in states.iter().enumerate() {
        check_cancel(cancel)?;
        for lane in 0..LANES.min(reads - word * LANES) {
            let s = state.lane(lane);
            energies.push(energy_f64(&s, graph));
            spins.push(s);
        }
    }
    Ok(FloatMsaSamples {
        spins,
        energies,
        seeded_reads: seeds.len(),
        workspace_bytes: plan.workspace_bytes,
        beta_ladder: plan.betas,
    })
}

/// Run the ladder over every replica word, colour class by colour class.
fn anneal(
    cpu: &CpuGraph,
    colors: &Coloring,
    betas: &[f64],
    params: &SampleParams,
    states: &mut [MscState],
    cancel: Option<(&CancelToken, Option<u64>)>,
    mut after_sweep: impl FnMut(),
) -> Result<(), FloatMsaError> {
    let sweeps_per = effective_sweeps_per_beta(params.num_sweeps, params.sweeps_per_beta);
    let mut streams: Vec<SmallRng> = (0..states.len())
        .map(|w| lane_stream(params.seed, w))
        .collect();
    for &beta in betas {
        for _ in 0..sweeps_per {
            check_cancel(cancel)?;
            for class in colors.classes() {
                for &var in class {
                    let var = var as usize;
                    let (nbrs, coups) = cpu.neighbors(var);
                    let bias = cpu.bias(var);
                    for (state, stream) in states.iter_mut().zip(&mut streams) {
                        let words = state.spins_mut();
                        let own = words[var];
                        let mask = flip_mask(own, words, nbrs, coups, bias, beta, stream);
                        words[var] ^= mask;
                    }
                }
            }
            after_sweep();
        }
    }
    Ok(())
}

/// `h + Σ_j J_ij s_j` for every lane, from the packed neighbour words.
///
/// The neighbour's lane bit, set for `s = -1`, is shifted into the coupling's
/// sign bit. That is exactly `J * s` for `s = ±1`, accumulated in CSR order
/// from the bias, as in `effective_field`'s from-scratch sum.
#[inline]
fn local_fields(spins: &[u64], nbrs: &[u32], coups: &[f64], bias: f64) -> [f64; LANES] {
    let mut local = [bias; LANES];
    for (&node, &coupling) in nbrs.iter().zip(coups) {
        let word = spins[node as usize];
        let bits = coupling.to_bits();
        for (lane, field) in local.iter_mut().enumerate() {
            *field += f64::from_bits(bits ^ (((word >> lane) & 1) << 63));
        }
    }
    local
}

/// Mask of lanes that flip: `Δ ≤ 0`, or `ln u < -β Δ`, per lane, with
/// `Δ = -2 s_i · field` and `s_i` read from `own`.
#[inline]
fn flip_mask(
    own: u64,
    spins: &[u64],
    nbrs: &[u32],
    coups: &[f64],
    bias: f64,
    beta: f64,
    rng: &mut SmallRng,
) -> u64 {
    let local = local_fields(spins, nbrs, coups, bias);
    let mut mask = 0u64;
    for (lane, &field) in local.iter().enumerate() {
        // Spin +1 gives -2·field; the set bit of spin -1 flips that sign.
        let delta = f64::from_bits((-2.0 * field).to_bits() ^ (((own >> lane) & 1) << 63));
        let accept = delta <= 0.0 || {
            let u: f64 = rng.sample(Open01);
            u.ln() < -beta * delta
        };
        if accept {
            mask |= 1 << lane;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::{sample_research, ResearchKernel};
    use crate::sa_sampler::SeededStart;
    use crate::sampler_core::{effective_field, CpuGraph};
    use quip_solver_core::{CancelToken, IsingGraph, SampleParams};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn params(num_reads: usize, num_sweeps: usize, seed: u64) -> SampleParams {
        SampleParams {
            num_reads,
            num_sweeps,
            seed,
            beta_range: Some((0.1, 10.0)),
            ..Default::default()
        }
    }

    /// Every pair coupled with probability `p`, fractional mixed-sign `J`
    /// and fractional `h`.
    fn weighted_graph(n: usize, p: f64, seed: u64) -> IsingGraph {
        let mut rng = SmallRng::seed_from_u64(seed);
        let h: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut edges = Vec::new();
        let mut j = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                if rng.gen::<f64>() < p {
                    edges.push((u, v));
                    j.push(rng.gen_range(-1.0..1.0));
                }
            }
        }
        IsingGraph::new(h, j, edges)
    }

    /// Every state of an `n`-node problem, state `k` in lane `k`.
    fn all_states_packed(n: usize) -> Vec<u64> {
        (0..n)
            .map(|i| {
                let mut w = 0u64;
                for state in 0..(1u64 << n) {
                    if (state >> i) & 1 == 1 {
                        w |= 1 << state;
                    }
                }
                w
            })
            .collect()
    }

    fn lane_spins(words: &[u64], lane: usize) -> Vec<i8> {
        words
            .iter()
            .map(|w| if (w >> lane) & 1 == 0 { 1 } else { -1 })
            .collect()
    }

    fn best(energies: &[f64]) -> f64 {
        energies.iter().copied().fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn flip_deltas_match_direct_energy_differences_on_every_state() {
        let star_edges: Vec<(usize, usize)> = (1..6).map(|v| (0, v)).collect();
        let graphs = [
            IsingGraph::new(vec![0.3, -0.7], vec![-0.375], vec![(0, 1)]),
            IsingGraph::new(
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.5, -0.25],
                vec![(0, 1), (1, 2), (0, 2)],
            ),
            weighted_graph(4, 0.8, 1),
            weighted_graph(5, 0.6, 2),
            IsingGraph::new(
                vec![0.125, -1.5, 0.0, 2.0, -0.01, 0.75],
                vec![1.0, -0.5, 0.3, -0.3, 0.9],
                star_edges,
            ),
            weighted_graph(6, 0.5, 3),
        ];
        for (gi, graph) in graphs.iter().enumerate() {
            let n = graph.h.len();
            let cpu = CpuGraph::from_base(graph);
            let words = all_states_packed(n);
            for var in 0..n {
                let (nbrs, coups) = cpu.neighbors(var);
                let fields = local_fields(&words, nbrs, coups, cpu.bias(var));
                for (state, &field) in fields.iter().enumerate().take(1 << n) {
                    let spins = lane_spins(&words, state);
                    let mut flipped = spins.clone();
                    flipped[var] = -flipped[var];
                    let direct = energy_f64(&flipped, graph) - energy_f64(&spins, graph);
                    let delta = -2.0 * f64::from(spins[var]) * field;
                    assert!(
                        (direct - delta).abs() < 1e-9,
                        "graph {gi} var {var} state {state}: direct {direct} delta {delta}"
                    );
                }
            }
        }
    }

    #[test]
    fn adjacent_float64_couplings_give_distinct_fields() {
        let a = 0.1f64;
        let b = f64::from_bits(a.to_bits() + 1);
        let fa = local_fields(&[0, 0], &[1], &[a], 0.0);
        let fb = local_fields(&[0, 0], &[1], &[b], 0.0);
        assert_eq!(fa[0], a);
        assert_eq!(fb[0], b);
        assert_ne!(fa[0], fb[0]);
    }

    #[test]
    fn a_positive_field_flips_a_positive_spin_downhill() {
        // Plan Task 4 step 7, on the CSR-slice signature. Every lane of a
        // zero word is spin +1, so the downhill flip covers the whole word;
        // in the second call only lane 0 is spin -1 and uphill.
        let mut rng = SmallRng::seed_from_u64(1);
        assert_eq!(flip_mask(0, &[0], &[], &[], 0.25, 1.0, &mut rng), u64::MAX);
        assert_eq!(
            flip_mask(1, &[1], &[], &[], 0.25, MAX_BETA, &mut rng),
            !1u64
        );
    }

    #[test]
    fn masks_match_a_scalar_oracle_given_the_same_stream() {
        let n = 40;
        let graph = weighted_graph(n, 0.3, 7);
        let cpu = CpuGraph::from_base(&graph);
        let mut rng = SmallRng::seed_from_u64(11);
        let words: Vec<u64> = (0..n).map(|_| rng.gen()).collect();
        for _ in 0..200 {
            let var = rng.gen_range(0..n);
            let beta = rng.gen_range(0.01..5.0);
            let mut stream = SmallRng::seed_from_u64(rng.gen());
            let mut expected_stream = stream.clone();
            let (nbrs, coups) = cpu.neighbors(var);
            let mask = flip_mask(
                words[var],
                &words,
                nbrs,
                coups,
                cpu.bias(var),
                beta,
                &mut stream,
            );
            for lane in 0..LANES {
                let spins = lane_spins(&words, lane);
                let field = effective_field(var, &spins, &cpu);
                let delta = -2.0 * f64::from(spins[var]) * field;
                let accept = delta <= 0.0 || {
                    let u: f64 = expected_stream.sample(Open01);
                    u.ln() < -beta * delta
                };
                assert_eq!((mask >> lane) & 1 == 1, accept, "var {var} lane {lane}");
            }
            assert_eq!(stream.gen::<u64>(), expected_stream.gen::<u64>());
        }
    }

    #[test]
    fn each_lane_draws_from_its_words_stream() {
        let graph = IsingGraph::new(
            vec![0.25, -0.375, 0.5],
            vec![0.75, -0.625, 0.5],
            vec![(0, 1), (0, 2), (1, 2)],
        );
        let params = SampleParams {
            sweeps_per_beta: 1,
            beta_range: Some((0.05, 3.0)),
            ..params(128, 3, 0x5A17)
        };
        let cpu = CpuGraph::from_base(&graph);
        let colors = Coloring::new(&cpu);
        let betas = build_beta_schedule(&graph, &params);
        let sweeps_per_beta = effective_sweeps_per_beta(params.num_sweeps, params.sweeps_per_beta);
        let mut expected: Vec<Vec<Vec<i8>>> = (0..params.num_reads)
            .step_by(LANES)
            .map(|read| {
                let mut rng = read_rng(params.seed, read);
                let state = MscState::random(graph.h.len(), &mut rng);
                (0..LANES).map(|lane| state.lane(lane)).collect()
            })
            .collect();
        for (word, lanes) in expected.iter_mut().enumerate() {
            let mut stream = lane_stream(params.seed, word);
            for &beta in &betas {
                for _ in 0..sweeps_per_beta {
                    for class in colors.classes() {
                        for &var in class {
                            let var = var as usize;
                            for spins in lanes.iter_mut() {
                                let field = effective_field(var, spins, &cpu);
                                let delta = -2.0 * f64::from(spins[var]) * field;
                                let accept = delta <= 0.0 || {
                                    let u: f64 = stream.sample(Open01);
                                    u.ln() < -beta * delta
                                };
                                if accept {
                                    spins[var] = -spins[var];
                                }
                            }
                        }
                    }
                }
            }
        }
        let expected: Vec<Vec<i8>> = expected.into_iter().flatten().collect();
        let actual = sample_float_msa(&graph, &params, None, None).expect("valid request");
        for (read, (got, want)) in actual.spins.iter().zip(&expected).enumerate() {
            assert_eq!(got, want, "read {read}");
        }
    }

    #[test]
    fn identical_lanes_separate() {
        let graph = weighted_graph(28, 1.0, 9);
        let seed_state: Vec<i8> = (0..28).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
        let seeds = vec![seed_state; 64];
        let params = SampleParams {
            beta_range: Some((0.05, 0.5)),
            ..params(64, 8, 3)
        };
        let start = SeededStart {
            spins: &seeds,
            start_beta: None,
        };
        let out = sample_float_msa(&graph, &params, Some(start), None).expect("valid request");
        let distinct: std::collections::BTreeSet<&Vec<i8>> = out.spins.iter().collect();
        assert!(
            distinct.len() >= 32,
            "{} distinct rows of 64",
            distinct.len()
        );
    }

    #[test]
    fn dense_models_keep_distinct_reads() {
        // QUI-1596 regression: a complete N=28 fractional model at 64 reads.
        let graph = weighted_graph(28, 1.0, 21);
        for num_sweeps in [50, 1000] {
            let params = SampleParams {
                beta_range: None,
                ..params(64, num_sweeps, 5)
            };
            let msa = sample_float_msa(&graph, &params, None, None).expect("valid request");
            let sa = sample_research(&graph, &params, ResearchKernel::ScalarSa, None, None)
                .expect("valid scalar request");
            let msa_distinct = msa
                .spins
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            let sa_distinct = sa
                .spins
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            assert!(
                msa_distinct > 1 && msa_distinct * 2 >= sa_distinct,
                "sweeps {num_sweeps}: msa {msa_distinct}, sa {sa_distinct}"
            );
        }
    }

    #[test]
    fn read_prefixes_do_not_depend_on_the_read_count() {
        let graph = weighted_graph(12, 0.5, 5);
        let long = sample_float_msa(&graph, &params(129, 20, 4), None, None).unwrap();
        for reads in [1_usize, 63, 64, 65] {
            let short = sample_float_msa(&graph, &params(reads, 20, 4), None, None).unwrap();
            assert_eq!(short.spins[..], long.spins[..reads], "reads {reads}");
        }
    }

    #[test]
    fn every_listed_read_count_returns_that_many_rows_in_seed_order() {
        let graph = weighted_graph(12, 0.5, 5);
        for reads in [1usize, 63, 64, 65, 129] {
            let mut rng = SmallRng::seed_from_u64(reads as u64);
            let seeds: Vec<Vec<i8>> = (0..reads)
                .map(|_| {
                    (0..12)
                        .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
                        .collect()
                })
                .collect();
            let start = SeededStart {
                spins: &seeds,
                start_beta: None,
            };
            let warm = sample_float_msa(&graph, &params(reads, 0, 1), Some(start), None)
                .expect("valid request");
            assert_eq!(warm.energies.len(), reads);
            assert_eq!(
                warm.spins, seeds,
                "{reads} reads: zero sweeps keep the seeds"
            );
            assert_eq!(warm.seeded_reads, reads);
            assert!(warm.workspace_bytes > 0);

            let cold =
                sample_float_msa(&graph, &params(reads, 32, 1), None, None).expect("valid request");
            assert_eq!(cold.spins.len(), reads);
            assert_eq!(cold.energies.len(), reads);
            assert!(cold
                .spins
                .iter()
                .all(|s| s.len() == 12 && s.iter().all(|&v| v == 1 || v == -1)));
            assert_eq!(cold.seeded_reads, 0);
        }
    }

    #[test]
    fn partial_seeds_take_the_first_lanes_and_the_rest_start_where_a_cold_run_starts() {
        let graph = weighted_graph(16, 0.4, 9);
        let seeds: Vec<Vec<i8>> = (0..3)
            .map(|r| {
                (0..16)
                    .map(|i| if (i + r) % 3 == 0 { -1 } else { 1 })
                    .collect()
            })
            .collect();
        let start = SeededStart {
            spins: &seeds,
            start_beta: None,
        };
        let warm =
            sample_float_msa(&graph, &params(70, 0, 4), Some(start), None).expect("valid request");
        let cold = sample_float_msa(&graph, &params(70, 0, 4), None, None).expect("valid request");
        assert_eq!(&warm.spins[..3], &seeds[..]);
        assert_eq!(&warm.spins[3..], &cold.spins[3..]);
        assert_eq!(warm.seeded_reads, 3);
    }

    #[test]
    fn the_same_seed_reproduces_and_a_different_seed_differs() {
        let graph = weighted_graph(30, 0.3, 2);
        let a = sample_float_msa(&graph, &params(8, 64, 21), None, None).expect("valid request");
        let b = sample_float_msa(&graph, &params(8, 64, 21), None, None).expect("valid request");
        let c = sample_float_msa(&graph, &params(8, 64, 22), None, None).expect("valid request");
        assert_eq!(a.spins, b.spins);
        assert_eq!(a.energies, b.energies);
        assert_ne!(a.spins, c.spins);
    }

    #[test]
    fn energies_are_the_float64_energy_of_the_original_model() {
        let graph = weighted_graph(24, 0.3, 3);
        let out = sample_float_msa(&graph, &params(70, 32, 6), None, None).expect("valid request");
        for (spins, &e) in out.spins.iter().zip(&out.energies) {
            assert_eq!(e, energy_f64(spins, &graph));
        }
        let pair = IsingGraph::new(vec![0.125, -0.2], vec![-0.375], vec![(0, 1)]);
        assert_eq!(energy_f64(&[1, 1], &pair), 0.125 - 0.2 - 0.375);
        assert_eq!(energy_f64(&[1, -1], &pair), 0.125 + 0.2 + 0.375);
    }

    #[test]
    fn a_weighted_ferromagnetic_ring_anneals_to_its_ground_state() {
        let n = 32;
        let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
        let j: Vec<f64> = (0..n).map(|i| -(0.25 + 0.05 * i as f64)).collect();
        let optimum: f64 = j.iter().sum();
        let graph = IsingGraph::new(vec![0.0; n], j, edges);
        let out = sample_float_msa(&graph, &params(64, 256, 8), None, None).expect("valid request");
        let got = best(&out.energies);
        assert!((got - optimum).abs() < 1e-9, "best {got} optimum {optimum}");
    }

    #[test]
    fn a_fields_only_problem_lands_on_the_exact_optimum() {
        let mut rng = SmallRng::seed_from_u64(6);
        let h: Vec<f64> = (0..40)
            .map(|_| {
                let magnitude = rng.gen_range(0.25..1.0);
                if rng.gen::<bool>() {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect();
        let optimum: f64 = -h.iter().map(|v| v.abs()).sum::<f64>();
        let graph = IsingGraph::new(h, vec![], vec![]);
        let out = sample_float_msa(&graph, &params(64, 128, 4), None, None).expect("valid request");
        let got = best(&out.energies);
        assert!((got - optimum).abs() < 1e-9, "best {got} optimum {optimum}");
    }

    #[test]
    fn a_cancelled_generation_returns_cancelled_and_a_newer_one_runs() {
        let graph = weighted_graph(8, 0.5, 1);
        let token = CancelToken::default();
        token.cancel_through(7);
        let err = sample_float_msa(&graph, &params(4, 16, 1), None, Some((&token, Some(7))))
            .expect_err("cancelled");
        assert_eq!(err, FloatMsaError::Cancelled);
        assert!(sample_float_msa(&graph, &params(4, 16, 1), None, Some((&token, Some(8)))).is_ok());
    }

    #[test]
    fn cancellation_after_allocation_stops_at_a_later_sweep_checkpoint() {
        let graph = weighted_graph(8, 0.5, 1);
        let cpu = CpuGraph::from_base(&graph);
        let colors = Coloring::new(&cpu);
        let params = SampleParams {
            sweeps_per_beta: 1,
            ..params(64, 2, 1)
        };
        let mut rng = read_rng(params.seed, 0);
        let mut states = vec![MscState::random(graph.h.len(), &mut rng)];
        let token = CancelToken::default();
        let mut completed_sweeps = 0;

        let result = anneal(
            &cpu,
            &colors,
            &[0.1, 1.0],
            &params,
            &mut states,
            Some((&token, Some(7))),
            || {
                completed_sweeps += 1;
                if completed_sweeps == 1 {
                    token.cancel_through(7);
                }
            },
        );

        assert_eq!(result, Err(FloatMsaError::Cancelled));
        assert_eq!(completed_sweeps, 1);
    }

    #[test]
    fn a_seeded_local_minimum_survives_a_sweep_at_a_cold_start_beta() {
        let graph = IsingGraph::new(vec![0.2, 0.2], vec![-1.0], vec![(0, 1)]);
        let seeds = vec![vec![1, 1]];
        let params = SampleParams {
            num_reads: 1,
            num_sweeps: 1,
            sweeps_per_beta: 1,
            seed: 19,
            beta_range: Some((0.1, 10.0)),
            schedule: quip_solver_core::BetaSchedule::Geometric,
        };
        let start = SeededStart {
            spins: &seeds,
            start_beta: Some(MAX_BETA),
        };

        let out = sample_float_msa(&graph, &params, Some(start), None).expect("valid request");
        assert_eq!(out.spins, seeds);
        assert_eq!(out.seeded_reads, 1);
        let ground_state = [-1, -1];
        assert!(out.energies[0] > energy_f64(&ground_state, &graph));
    }

    #[test]
    fn a_large_model_with_many_reads_hits_the_workspace_cap() {
        let n = 2_000_000;
        let graph = IsingGraph::new(vec![0.0; n], vec![], vec![]);
        let err = sample_float_msa(&graph, &params(MAX_READS, 1, 1), None, None)
            .expect_err("4096 reads of 2M nodes exceed the cap");
        assert!(
            matches!(err, FloatMsaError::MemoryLimit { needed, cap } if cap == WORKSPACE_CAP_BYTES && needed > cap),
            "{err:?}"
        );
    }

    #[test]
    fn workspace_estimate_counts_per_word_acceptance_streams() {
        let base = 16 + 8 + 9 + 8 + LANES * 8;
        assert_eq!(
            workspace_bytes(1, 0, 1, 1),
            Some(base + size_of::<SmallRng>())
        );
        assert_eq!(
            workspace_bytes(1, 0, 1, 2),
            Some(base + 8 + 2 * size_of::<SmallRng>())
        );
    }

    #[test]
    fn edge_heavy_workspace_is_refused_from_counts() {
        assert!(check_workspace(2_000_000, 0, 3136).is_ok());
        assert!(matches!(
            check_workspace(2_000_000, 67_108_864, 3136),
            Err(FloatMsaError::MemoryLimit { needed, cap })
                if cap == WORKSPACE_CAP_BYTES && needed > cap
        ));
    }

    #[test]
    fn size_check_accepts_node_cap_and_rejects_the_next_node() {
        assert_eq!(check_size(4_194_304, 0), Ok(()));
        assert_eq!(
            check_size(4_194_305, 0),
            Err(FloatMsaError::InvalidInput(
                "4194305 nodes exceed the limit of 4194304".to_owned()
            ))
        );
    }

    #[test]
    fn size_check_accepts_edge_cap_and_rejects_the_next_edge() {
        assert_eq!(check_size(1, 67_108_864), Ok(()));
        assert_eq!(
            check_size(1, 67_108_865),
            Err(FloatMsaError::InvalidInput(
                "67108865 edges exceed the limit of 67108864".to_owned()
            ))
        );
    }

    #[test]
    fn invalid_inputs_are_rejected_with_a_named_reason() {
        let chain =
            |h: Vec<f64>, j: Vec<f64>, edges: Vec<(usize, usize)>| IsingGraph::new(h, j, edges);
        let with = |beta_range: Option<(f64, f64)>| SampleParams {
            beta_range,
            ..params(4, 8, 1)
        };
        let two = vec![(0, 1), (1, 2)];
        let cases: Vec<(&str, IsingGraph, SampleParams)> = vec![
            (
                "nan field",
                chain(vec![f64::NAN, 0.0, 0.0], vec![0.5, -0.25], two.clone()),
                params(4, 8, 1),
            ),
            (
                "infinite coupling",
                chain(vec![0.0; 3], vec![f64::INFINITY, -0.25], two.clone()),
                params(4, 8, 1),
            ),
            (
                "subnormal coupling",
                chain(vec![0.0; 3], vec![5e-324, -0.25], two.clone()),
                params(4, 8, 1),
            ),
            (
                "subnormal field",
                chain(vec![-1e-310, 0.0, 0.0], vec![0.5, -0.25], two.clone()),
                params(4, 8, 1),
            ),
            (
                "duplicate edge",
                chain(vec![0.0; 3], vec![0.5, -0.25], vec![(0, 1), (1, 0)]),
                params(4, 8, 1),
            ),
            (
                "self-loop",
                chain(vec![0.0; 3], vec![0.5, -0.25], vec![(0, 1), (1, 1)]),
                params(4, 8, 1),
            ),
            (
                "endpoint out of range",
                chain(vec![0.0; 3], vec![0.5, -0.25], vec![(0, 1), (1, 3)]),
                params(4, 8, 1),
            ),
            (
                "fewer couplings than edges",
                chain(vec![0.0; 3], vec![0.5], two.clone()),
                params(4, 8, 1),
            ),
            ("no nodes", chain(vec![], vec![], vec![]), params(4, 8, 1)),
            (
                "too many nodes",
                chain(vec![0.0; 4_194_305], vec![], vec![]),
                params(4, 8, 1),
            ),
            (
                "coefficient sum above the cap",
                chain(vec![1e100, 1e100, 0.0], vec![0.5, -0.25], two.clone()),
                params(4, 8, 1),
            ),
            (
                "zero reads",
                chain(vec![0.0; 3], vec![0.5, -0.25], two.clone()),
                params(0, 8, 1),
            ),
            (
                "too many reads",
                chain(vec![0.0; 3], vec![0.5, -0.25], two.clone()),
                params(MAX_READS + 1, 8, 1),
            ),
            (
                "non-positive beta",
                chain(vec![0.0; 3], vec![0.5, -0.25], two.clone()),
                with(Some((0.0, 1.0))),
            ),
            (
                "beta above the cap",
                chain(vec![0.0; 3], vec![0.5, -0.25], two.clone()),
                with(Some((0.1, 1e101))),
            ),
            (
                "non-finite beta",
                chain(vec![0.0; 3], vec![0.5, -0.25], two.clone()),
                with(Some((0.1, f64::INFINITY))),
            ),
            (
                "beta range backwards",
                chain(vec![0.0; 3], vec![0.5, -0.25], two.clone()),
                with(Some((5.0, 0.1))),
            ),
        ];
        for (name, graph, params) in cases {
            let err = sample_float_msa(&graph, &params, None, None).expect_err(name);
            let named = if let FloatMsaError::InvalidInput(msg) = &err {
                !msg.is_empty()
            } else {
                false
            };
            assert!(named, "{name}: {err:?}");
        }

        let graph = chain(vec![0.1, -0.2, 0.3], vec![0.5, -0.25], two);
        type SeedCase<'a> = (&'a str, Vec<Vec<i8>>, Option<f64>, usize);
        let seed_cases: Vec<SeedCase<'_>> = vec![
            ("seed of the wrong length", vec![vec![1, 1]], None, 4),
            (
                "seed with a value other than ±1",
                vec![vec![1, 0, -1]],
                None,
                4,
            ),
            (
                "more seeds than reads",
                vec![vec![1, 1, 1], vec![-1, -1, -1]],
                None,
                1,
            ),
            (
                "non-finite start beta",
                vec![vec![1, 1, 1]],
                Some(f64::NAN),
                4,
            ),
            ("non-positive start beta", vec![vec![1, 1, 1]], Some(0.0), 4),
            (
                "start beta above the cap",
                vec![vec![1, 1, 1]],
                Some(1e101),
                4,
            ),
        ];
        for (name, seeds, start_beta, reads) in seed_cases {
            let start = SeededStart {
                spins: &seeds,
                start_beta,
            };
            let err =
                sample_float_msa(&graph, &params(reads, 8, 1), Some(start), None).expect_err(name);
            let named = if let FloatMsaError::InvalidInput(msg) = &err {
                !msg.is_empty()
            } else {
                false
            };
            assert!(named, "{name}: {err:?}");
        }
    }
}
