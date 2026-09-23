//! Strict research sampling: an explicit kernel, an observed identity, and
//! float64 energies.
//!
//! The protocol samplers in this crate substitute a kernel when a problem
//! does not qualify: `quip-cpu-msa` answers a fractional problem with the
//! scalar `cpu-sa` kernel and says nothing. That is right for mining and
//! wrong for a benchmark that compares kernels. [`sample_research`] takes the
//! kernel as an argument, refuses any request the named kernel cannot run,
//! and reports the identity it ran under, so requested and observed kernel
//! are equal by construction. The one named exception is `cpu-msa`, which
//! routes to the unit or the float arm by the model and reports the arm as
//! the observed kernel. Energies come back in float64 from the original
//! model rather than as truncated milli-units.

use quip_solver_core::{CancelToken, IsingGraph, SampleParams, SamplerResult};

use crate::sa_int::IntGraph;
use crate::sa_msc::bond_counts;
use crate::sa_msc_f64::{check_beta, energy_f64, sample_float_msa, FloatMsaError};
use crate::sa_sampler::{check_seeds, sample_unit_packed, SeededStart};
use crate::sampler_core::{sample_sa_scalar, ScalarArithmetic};
use crate::{DEFAULT_MAX_EDGES, DEFAULT_MAX_NODES};
use quip_solver_core::beta::default_ising_beta_range;

/// Most reads one research call may ask for, on any kernel.
pub const MAX_READS: usize = crate::sa_msc_f64::MAX_READS;

/// The kernel a research call names. There is no default and no fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResearchKernel {
    /// The scalar `cpu-sa` kernel, in whichever arithmetic its dispatch picks.
    ScalarSa,
    /// The bit-sliced unit-weight kernel of `quip-cpu-msa`, packed arm only.
    UnitMsa,
    /// The lane-batched float64 kernel of [`crate::sa_msc_f64`].
    FloatMsa,
    /// `cpu-msa`: [`Self::UnitMsa`] when the model fits it, [`Self::FloatMsa`]
    /// otherwise. `observed_kernel` names the arm that ran.
    Msa,
}

impl ResearchKernel {
    /// Every kernel, in the order the names are documented.
    pub const ALL: [Self; 4] = [Self::ScalarSa, Self::UnitMsa, Self::FloatMsa, Self::Msa];

    /// The kernel named `name`, or `None` for any other string.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kernel| kernel.name() == name)
    }

    /// The identity string: `cpu-sa`, `cpu-msa-unit`, `cpu-msa-f64` or `cpu-msa`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::ScalarSa => "cpu-sa",
            Self::UnitMsa => "cpu-msa-unit",
            Self::FloatMsa => "cpu-msa-f64",
            Self::Msa => "cpu-msa",
        }
    }
}

/// What ran, as the report needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResearchMetadata {
    /// The kernel the caller named.
    pub requested_kernel: &'static str,
    /// The kernel that produced the samples. Equal to the request: a request
    /// the kernel cannot run is refused, never rerouted.
    pub observed_kernel: &'static str,
    /// `scalar-int`, `scalar-f64`, `unit-bit-sliced` or
    /// `packed-spins-f64-fields`.
    pub representation: &'static str,
    /// How acceptance randomness is shared between reads.
    pub rng_scheme: &'static str,
    /// Leading reads that started from a supplied state.
    pub seeded_reads: usize,
    /// Workspace the kernel counted against its cap. Only the float kernel
    /// counts; the others report `None`, not zero.
    pub workspace_bytes: Option<usize>,
}

/// Samples in read order with their float64 energies.
#[derive(Debug, Clone, PartialEq)]
pub struct ResearchSamples {
    /// `num_reads` states of `±1` spins.
    pub spins: Vec<Vec<i8>>,
    /// `Σ h s + Σ J s s` of each state in float64, from the original model.
    pub energies: Vec<f64>,
    /// Identity and provenance of the run.
    pub metadata: ResearchMetadata,
}

/// Why a research call did not return samples.
#[derive(Debug, Clone, PartialEq)]
pub enum ResearchError {
    /// The model, parameters or seeds violate a documented bound.
    InvalidInput(String),
    /// The unit kernel cannot represent this model. The message says why.
    UnsupportedUnitModel(String),
    /// The scalar kernel has no seeded form.
    SeededScalarUnsupported,
    /// The cancel token fired.
    Cancelled,
    /// The float kernel's workspace would exceed its cap.
    MemoryLimit {
        /// Bytes the run would allocate, `usize::MAX` if the count overflowed.
        needed: usize,
        /// The cap.
        cap: usize,
    },
}

impl std::fmt::Display for ResearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(msg) | Self::UnsupportedUnitModel(msg) => f.write_str(msg),
            Self::SeededScalarUnsupported => {
                f.write_str("cpu-sa has no seeded form; seeds need an MSA kernel")
            }
            Self::Cancelled => f.write_str("the run was cancelled"),
            Self::MemoryLimit { needed, cap } => {
                write!(
                    f,
                    "the run needs {needed} bytes of workspace; the cap is {cap}"
                )
            }
        }
    }
}

impl std::error::Error for ResearchError {}

fn invalid<T>(msg: String) -> Result<T, ResearchError> {
    Err(ResearchError::InvalidInput(msg))
}

/// The checks every kernel shares: a model the scorer can index, finite
/// coefficients, and a schedule every kernel honours the same way. Bounds on
/// coefficient values and memory live with the float kernel.
fn validate_common(
    graph: &IsingGraph,
    params: &SampleParams,
    start: Option<SeededStart<'_>>,
) -> Result<(), ResearchError> {
    let nodes = graph.h.len();
    let edges = graph.edges.len();
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
    if graph.j.len() != edges {
        return invalid(format!("{} couplings for {edges} edges", graph.j.len()));
    }
    if !(1..=MAX_READS).contains(&params.num_reads) {
        return invalid(format!(
            "num_reads must be in 1..={MAX_READS}; got {}",
            params.num_reads
        ));
    }
    // The protocol samplers round zero sweeps up to one rung; the float
    // kernel returns the start states. A comparison needs one meaning.
    if params.num_sweeps == 0 {
        return invalid("num_sweeps must be at least 1; got 0".to_owned());
    }
    let (label, (hot, cold)) = match params.beta_range {
        Some(range) => ("", range),
        None => ("default ", default_ising_beta_range(graph)),
    };
    let beta = |name: &str, value: f64| {
        check_beta(&format!("{label}{name}"), value)
            .map_err(|e| ResearchError::InvalidInput(e.to_string()))
    };
    beta("hot", hot)?;
    beta("cold", cold)?;
    if hot > cold {
        return invalid(format!(
            "{label}beta_range runs backwards: hot {hot} is above cold {cold}"
        ));
    }
    if let Some(start_beta) = start.and_then(|s| s.start_beta) {
        beta("start", start_beta)?;
    }
    if let Some(i) = graph.h.iter().position(|v| !v.is_finite()) {
        return invalid(format!("h[{i}] is not finite"));
    }
    if let Some(k) = graph.j.iter().position(|v| !v.is_finite()) {
        return invalid(format!("J[{k}] is not finite"));
    }
    if let Some(k) = graph
        .edges
        .iter()
        .position(|&(u, v)| u >= nodes || v >= nodes)
    {
        return invalid(format!("edge {k} has an endpoint outside 0..{nodes}"));
    }
    Ok(())
}

/// Seeds fit the model and do not outnumber the reads. Returns the count.
fn validate_seeds(
    start: SeededStart<'_>,
    nodes: usize,
    reads: usize,
) -> Result<usize, ResearchError> {
    check_seeds(start, nodes).map_err(|e| ResearchError::InvalidInput(e.to_string()))?;
    if start.spins.len() > reads {
        return invalid(format!("{} seeds for {reads} reads", start.spins.len()));
    }
    Ok(start.spins.len())
}

/// Anneal `graph` on exactly `kernel`.
///
/// # Errors
///
/// [`ResearchError::InvalidInput`] for a model or request outside the shared
/// or kernel-specific bounds, including zero sweeps and a beta range or
/// start beta that is not finite, not positive, above the cap, or reversed; [`ResearchError::UnsupportedUnitModel`] when
/// the unit kernel cannot represent the model;
/// [`ResearchError::SeededScalarUnsupported`] for a seeded scalar request;
/// [`ResearchError::MemoryLimit`] from the float kernel's cap;
/// [`ResearchError::Cancelled`] when `cancel` fires.
pub fn sample_research(
    graph: &IsingGraph,
    params: &SampleParams,
    kernel: ResearchKernel,
    start: Option<SeededStart<'_>>,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<ResearchSamples, ResearchError> {
    // A seed set with no states is a cold run, as it is for `Msa.sample`.
    let start = start.filter(|s| !s.spins.is_empty());
    validate_common(graph, params, start)?;
    match kernel {
        ResearchKernel::ScalarSa => sample_scalar_research(graph, params, start, cancel),
        ResearchKernel::UnitMsa => sample_unit_research(graph, params, start, cancel),
        ResearchKernel::FloatMsa => sample_float_research(graph, params, start, cancel),
        ResearchKernel::Msa => sample_merged_msa(graph, params, start, cancel),
    }
}

/// The unit arm when the packed integer graph and its bond bound accept the
/// model, the float arm otherwise. The arm is a property of the model, so a
/// caller who names `cpu-msa` reads the arm from `observed_kernel`.
fn sample_merged_msa(
    graph: &IsingGraph,
    params: &SampleParams,
    start: Option<SeededStart<'_>>,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<ResearchSamples, ResearchError> {
    let fits_unit = IntGraph::from_base(graph).is_some_and(|int| bond_counts(&int).is_some());
    let mut out = if fits_unit {
        sample_unit_research(graph, params, start, cancel)?
    } else {
        sample_float_research(graph, params, start, cancel)?
    };
    out.metadata.requested_kernel = ResearchKernel::Msa.name();
    Ok(out)
}

fn finish(
    kernel: ResearchKernel,
    results: Vec<SamplerResult>,
    graph: &IsingGraph,
    representation: &'static str,
    rng_scheme: &'static str,
    seeded_reads: usize,
) -> ResearchSamples {
    let spins: Vec<Vec<i8>> = results.into_iter().map(|r| r.spins).collect();
    let energies = spins.iter().map(|s| energy_f64(s, graph)).collect();
    ResearchSamples {
        spins,
        energies,
        metadata: ResearchMetadata {
            requested_kernel: kernel.name(),
            observed_kernel: kernel.name(),
            representation,
            rng_scheme,
            seeded_reads,
            workspace_bytes: None,
        },
    }
}

fn sample_scalar_research(
    graph: &IsingGraph,
    params: &SampleParams,
    start: Option<SeededStart<'_>>,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<ResearchSamples, ResearchError> {
    if start.is_some() {
        return Err(ResearchError::SeededScalarUnsupported);
    }
    let (results, arithmetic) =
        sample_sa_scalar(graph, params, cancel).map_err(|_| ResearchError::Cancelled)?;
    let representation = match arithmetic {
        ScalarArithmetic::Int => "scalar-int",
        ScalarArithmetic::F64 => "scalar-f64",
    };
    Ok(finish(
        ResearchKernel::ScalarSa,
        results,
        graph,
        representation,
        "per-attempt-uniform-v1",
        0,
    ))
}

fn sample_unit_research(
    graph: &IsingGraph,
    params: &SampleParams,
    start: Option<SeededStart<'_>>,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<ResearchSamples, ResearchError> {
    let Some(int) = IntGraph::from_base(graph) else {
        return Err(ResearchError::UnsupportedUnitModel(
            "the unit kernel takes couplings in {-1, 0, +1} and whole-number fields \
             within its bound; this model has another value"
                .to_owned(),
        ));
    };
    let Some(counts) = bond_counts(&int) else {
        return Err(ResearchError::UnsupportedUnitModel(
            "the unit kernel folds a field in {-1, 0, +1} into one bond and takes at \
             most 63 bonds per node; this model exceeds that"
                .to_owned(),
        ));
    };
    let seeded_reads = match start {
        Some(s) => validate_seeds(s, graph.h.len(), params.num_reads)?,
        None => 0,
    };
    let results = sample_unit_packed(graph, params, &int, &counts, start, cancel)
        .map_err(|_| ResearchError::Cancelled)?;
    Ok(finish(
        ResearchKernel::UnitMsa,
        results,
        graph,
        "unit-bit-sliced",
        "shared-threshold-row-v1",
        seeded_reads,
    ))
}

fn sample_float_research(
    graph: &IsingGraph,
    params: &SampleParams,
    start: Option<SeededStart<'_>>,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<ResearchSamples, ResearchError> {
    let out = sample_float_msa(graph, params, start, cancel).map_err(|e| match e {
        FloatMsaError::InvalidInput(msg) => ResearchError::InvalidInput(msg),
        FloatMsaError::MemoryLimit { needed, cap } => ResearchError::MemoryLimit { needed, cap },
        FloatMsaError::Cancelled => ResearchError::Cancelled,
    })?;
    Ok(ResearchSamples {
        spins: out.spins,
        energies: out.energies,
        metadata: ResearchMetadata {
            requested_kernel: ResearchKernel::FloatMsa.name(),
            observed_kernel: ResearchKernel::FloatMsa.name(),
            representation: "packed-spins-f64-fields",
            rng_scheme: crate::sa_msc_f64::RNG_SCHEME,
            seeded_reads: out.seeded_reads,
            workspace_bytes: Some(out.workspace_bytes),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sa_msc_f64::energy_f64;
    use crate::sa_sampler::{SaSampler, SaVariant, SeededStart};
    use crate::sampler_core::sample_ising;
    use quip_solver_core::{Algorithm, CancelToken, IsingGraph, SampleParams, Sampler};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn params(num_reads: usize, num_sweeps: usize, seed: u64) -> SampleParams {
        SampleParams {
            num_reads,
            num_sweeps,
            seed,
            beta_range: Some((0.1, 5.0)),
            ..Default::default()
        }
    }

    /// `±1` couplings and `{-1, 0, 1}` fields: every kernel takes it.
    fn unit_graph(n: usize, seed: u64) -> IsingGraph {
        let mut rng = SmallRng::seed_from_u64(seed);
        let h: Vec<f64> = (0..n).map(|_| f64::from(rng.gen_range(-1i8..=1))).collect();
        let mut edges = Vec::new();
        let mut j = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                if rng.gen::<f64>() < 0.3 {
                    edges.push((u, v));
                    j.push(if rng.gen::<bool>() { 1.0 } else { -1.0 });
                }
            }
        }
        IsingGraph::new(h, j, edges)
    }

    fn fractional_pair() -> IsingGraph {
        IsingGraph::new(vec![0.125, -0.2], vec![-0.375], vec![(0, 1)])
    }

    fn is_unsupported_unit(err: &ResearchError) -> bool {
        if let ResearchError::UnsupportedUnitModel(msg) = err {
            msg.contains("unit")
        } else {
            false
        }
    }

    fn is_invalid(err: &ResearchError) -> bool {
        if let ResearchError::InvalidInput(msg) = err {
            !msg.is_empty()
        } else {
            false
        }
    }

    #[test]
    fn kernel_names_round_trip_and_unknown_names_do_not_parse() {
        for kernel in ResearchKernel::ALL {
            assert_eq!(ResearchKernel::parse(kernel.name()), Some(kernel));
        }
        assert_eq!(ResearchKernel::parse("msa"), None);
        assert_eq!(ResearchKernel::parse("cuda-msa"), None);
        assert_eq!(ResearchKernel::parse(""), None);
        assert_eq!(ResearchKernel::ScalarSa.name(), "cpu-sa");
        assert_eq!(ResearchKernel::UnitMsa.name(), "cpu-msa-unit");
        assert_eq!(ResearchKernel::FloatMsa.name(), "cpu-msa-f64");
        assert_eq!(ResearchKernel::Msa.name(), "cpu-msa");
    }

    #[test]
    fn fractional_models_cannot_masquerade_as_unit_msa() {
        let graph = fractional_pair();
        let err = sample_research(
            &graph,
            &params(4, 8, 1),
            ResearchKernel::UnitMsa,
            None,
            None,
        )
        .expect_err("a fractional coupling is not a unit model");
        assert!(is_unsupported_unit(&err), "{err:?}");

        let out = sample_research(
            &graph,
            &params(4, 8, 1),
            ResearchKernel::FloatMsa,
            None,
            None,
        )
        .expect("the float kernel takes it");
        assert_eq!(out.metadata.requested_kernel, "cpu-msa-f64");
        assert_eq!(out.metadata.observed_kernel, "cpu-msa-f64");
        assert_eq!(out.metadata.representation, "packed-spins-f64-fields");
        assert_eq!(out.metadata.rng_scheme, "shared-node-threshold-v1");
        assert!(out.metadata.workspace_bytes.is_some());
        for (spins, &e) in out.spins.iter().zip(&out.energies) {
            let expected = f64::from(spins[0]) * 0.125
                + f64::from(spins[1]) * -0.2
                + -0.375 * f64::from(spins[0]) * f64::from(spins[1]);
            assert!((e - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn the_merged_msa_kernel_routes_by_model_and_reports_the_arm() {
        let unit = unit_graph(20, 3);
        let merged = sample_research(&unit, &params(3, 16, 2), ResearchKernel::Msa, None, None)
            .expect("a unit model runs on the merged kernel");
        let explicit = sample_research(
            &unit,
            &params(3, 16, 2),
            ResearchKernel::UnitMsa,
            None,
            None,
        )
        .expect("a unit model runs on the unit kernel");
        assert_eq!(merged.metadata.requested_kernel, "cpu-msa");
        assert_eq!(merged.metadata.observed_kernel, "cpu-msa-unit");
        assert_eq!(merged.metadata.representation, "unit-bit-sliced");
        assert_eq!(merged.spins, explicit.spins);
        assert_eq!(merged.energies, explicit.energies);

        let frac = fractional_pair();
        let merged = sample_research(&frac, &params(4, 8, 1), ResearchKernel::Msa, None, None)
            .expect("a fractional model runs on the merged kernel");
        let explicit = sample_research(
            &frac,
            &params(4, 8, 1),
            ResearchKernel::FloatMsa,
            None,
            None,
        )
        .expect("a fractional model runs on the float kernel");
        assert_eq!(merged.metadata.requested_kernel, "cpu-msa");
        assert_eq!(merged.metadata.observed_kernel, "cpu-msa-f64");
        assert_eq!(merged.metadata.representation, "packed-spins-f64-fields");
        assert_eq!(merged.spins, explicit.spins);
        assert!(merged.metadata.workspace_bytes.is_some());

        // Unit weights past the 63-bond bound also go to the float arm.
        let n = 65;
        let edges: Vec<(usize, usize)> = (1..n).map(|v| (0, v)).collect();
        let star = IsingGraph::new(vec![0.0; n], vec![1.0; n - 1], edges);
        let merged = sample_research(&star, &params(2, 8, 1), ResearchKernel::Msa, None, None)
            .expect("a wide star runs on the merged kernel");
        assert_eq!(merged.metadata.observed_kernel, "cpu-msa-f64");
        assert_eq!(merged.spins.len(), 2);
    }

    #[test]
    fn every_kernel_reports_its_identity_and_representation() {
        let graph = unit_graph(20, 3);
        let expect = [
            (ResearchKernel::ScalarSa, "scalar-int"),
            (ResearchKernel::UnitMsa, "unit-bit-sliced"),
            (ResearchKernel::FloatMsa, "packed-spins-f64-fields"),
        ];
        for (kernel, representation) in expect {
            let out = sample_research(&graph, &params(3, 16, 2), kernel, None, None)
                .expect("a unit model runs on every kernel");
            assert_eq!(out.metadata.requested_kernel, kernel.name());
            assert_eq!(out.metadata.observed_kernel, kernel.name());
            assert_eq!(out.metadata.representation, representation, "{kernel:?}");
            assert_eq!(out.metadata.seeded_reads, 0);
            assert_eq!(out.spins.len(), 3);
            assert_eq!(out.energies.len(), 3);
        }
        // Scalar SA on a fractional model records the float arithmetic it ran.
        let out = sample_research(
            &fractional_pair(),
            &params(3, 16, 2),
            ResearchKernel::ScalarSa,
            None,
            None,
        )
        .expect("scalar SA takes any model");
        assert_eq!(out.metadata.observed_kernel, "cpu-sa");
        assert_eq!(out.metadata.representation, "scalar-f64");
        assert_eq!(out.metadata.workspace_bytes, None);
    }

    #[test]
    fn energies_are_float64_rescoring_of_the_original_model_for_every_kernel() {
        let graph = unit_graph(24, 5);
        for kernel in ResearchKernel::ALL {
            let out = sample_research(&graph, &params(70, 16, 4), kernel, None, None)
                .expect("a unit model runs on every kernel");
            for (spins, &e) in out.spins.iter().zip(&out.energies) {
                assert_eq!(e, energy_f64(spins, &graph), "{kernel:?}");
            }
        }
    }

    #[test]
    fn the_scalar_and_unit_kernels_are_the_existing_samplers_not_substitutes() {
        let graph = unit_graph(24, 7);
        let p = params(70, 16, 9);
        let scalar =
            sample_research(&graph, &p, ResearchKernel::ScalarSa, None, None).expect("scalar");
        let reference: Vec<Vec<i8>> = sample_ising(&graph, &p, Algorithm::Sa)
            .into_iter()
            .map(|r| r.spins)
            .collect();
        assert_eq!(scalar.spins, reference);

        let unit = sample_research(&graph, &p, ResearchKernel::UnitMsa, None, None).expect("unit");
        let reference: Vec<Vec<i8>> = SaSampler::new(SaVariant::MultiSpin)
            .sample(&graph, &p)
            .expect("msa")
            .into_iter()
            .map(|r| r.spins)
            .collect();
        assert_eq!(unit.spins, reference);
    }

    #[test]
    fn the_unit_kernel_refuses_instead_of_falling_back() {
        // |h| = 2 cannot be one ghost bond: today's sampler would take the
        // tabulated arm; the research entry must refuse.
        let wide_field =
            IsingGraph::new(vec![2.0, 0.0, -2.0], vec![1.0, -1.0], vec![(0, 1), (1, 2)]);
        let err = sample_research(
            &wide_field,
            &params(4, 8, 1),
            ResearchKernel::UnitMsa,
            None,
            None,
        )
        .expect_err("refused");
        assert!(is_unsupported_unit(&err), "{err:?}");

        // Degree 64 exceeds the plane budget.
        let edges: Vec<(usize, usize)> = (1..=64).map(|v| (0, v)).collect();
        let star = IsingGraph::new(vec![0.0; 65], vec![1.0; 64], edges);
        let err = sample_research(&star, &params(4, 8, 1), ResearchKernel::UnitMsa, None, None)
            .expect_err("refused");
        assert!(is_unsupported_unit(&err), "{err:?}");
        // The float kernel has no degree limit.
        assert!(sample_research(
            &star,
            &params(4, 8, 1),
            ResearchKernel::FloatMsa,
            None,
            None
        )
        .is_ok());
    }

    #[test]
    fn a_seeded_scalar_request_is_refused() {
        let graph = unit_graph(8, 1);
        let seeds = vec![vec![1i8; 8]];
        let start = SeededStart {
            spins: &seeds,
            start_beta: None,
        };
        let err = sample_research(
            &graph,
            &params(4, 8, 1),
            ResearchKernel::ScalarSa,
            Some(start),
            None,
        )
        .expect_err("refused");
        assert_eq!(err, ResearchError::SeededScalarUnsupported);
    }

    /// A ring whose planted state satisfies every bond, so `planted` and
    /// `-planted` are ground states no sweep at the beta cap can leave.
    fn planted_ring(n: usize) -> (IsingGraph, Vec<i8>) {
        let planted: Vec<i8> = (0..n)
            .map(|i| {
                if ((i * 2_654_435_761) >> 7) & 1 == 1 {
                    1
                } else {
                    -1
                }
            })
            .collect();
        let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
        let j: Vec<f64> = edges
            .iter()
            .map(|&(u, v)| -f64::from(planted[u]) * f64::from(planted[v]))
            .collect();
        (IsingGraph::new(vec![0.0; n], j, edges), planted)
    }

    #[test]
    fn seeded_msa_runs_place_seeds_across_word_boundaries() {
        let (graph, planted) = planted_ring(12);
        let flipped: Vec<i8> = planted.iter().map(|s| -s).collect();
        for reads in [63usize, 64, 65] {
            let seeds: Vec<Vec<i8>> = (0..reads)
                .map(|r| {
                    if r % 2 == 0 {
                        planted.clone()
                    } else {
                        flipped.clone()
                    }
                })
                .collect();
            // One sweep at the beta cap accepts no uphill move, and a ground
            // state has none downhill, so every seeded lane must come back.
            let start = SeededStart {
                spins: &seeds,
                start_beta: Some(1e100),
            };
            for kernel in [ResearchKernel::FloatMsa, ResearchKernel::UnitMsa] {
                let out = sample_research(&graph, &params(reads, 1, 1), kernel, Some(start), None)
                    .expect("seeded run");
                assert_eq!(out.spins, seeds, "{kernel:?} {reads} reads");
                assert_eq!(out.metadata.seeded_reads, reads);
            }
        }
    }

    #[test]
    fn beta_parameters_are_validated_for_every_kernel() {
        let graph = unit_graph(8, 1);
        let seeds = vec![vec![1i8; 8]];
        for kernel in ResearchKernel::ALL {
            for (name, range) in [
                ("nan hot", (f64::NAN, 1.0)),
                ("negative hot", (-1.0, 1.0)),
                ("zero hot", (0.0, 1.0)),
                ("infinite cold", (0.1, f64::INFINITY)),
                ("cold above the cap", (0.1, 1e101)),
                ("reversed range", (10.0, 0.1)),
            ] {
                let p = SampleParams {
                    beta_range: Some(range),
                    ..params(4, 8, 1)
                };
                let err = sample_research(&graph, &p, kernel, None, None).expect_err(name);
                assert!(is_invalid(&err), "{kernel:?} {name}: {err:?}");
            }
        }
        for kernel in [ResearchKernel::UnitMsa, ResearchKernel::FloatMsa] {
            for (name, start_beta) in [
                ("nan", f64::NAN),
                ("zero", 0.0),
                ("negative", -1.0),
                ("infinite", f64::INFINITY),
                ("above the cap", 1e101),
            ] {
                let start = SeededStart {
                    spins: &seeds,
                    start_beta: Some(start_beta),
                };
                let err = sample_research(&graph, &params(4, 8, 1), kernel, Some(start), None)
                    .expect_err(name);
                assert!(is_invalid(&err), "{kernel:?} start beta {name}: {err:?}");
            }
        }
    }

    #[test]
    fn zero_sweeps_are_refused_on_every_kernel() {
        let graph = unit_graph(8, 1);
        for kernel in ResearchKernel::ALL {
            let err = sample_research(&graph, &params(4, 0, 1), kernel, None, None)
                .expect_err("zero sweeps");
            assert!(is_invalid(&err), "{kernel:?}: {err:?}");
        }
    }

    #[test]
    fn an_empty_seed_set_runs_cold_on_every_kernel() {
        let graph = unit_graph(10, 3);
        let none: Vec<Vec<i8>> = Vec::new();
        for kernel in ResearchKernel::ALL {
            let cold = sample_research(&graph, &params(4, 8, 2), kernel, None, None).expect("cold");
            let start = SeededStart {
                spins: &none,
                start_beta: None,
            };
            let empty = sample_research(&graph, &params(4, 8, 2), kernel, Some(start), None)
                .expect("no seeds is a cold run");
            assert_eq!(empty.spins, cold.spins, "{kernel:?}");
            assert_eq!(empty.metadata.seeded_reads, 0);
        }
    }

    #[test]
    fn invalid_inputs_are_refused_by_every_kernel() {
        let nan = IsingGraph::new(vec![f64::NAN, 0.0], vec![1.0], vec![(0, 1)]);
        let out_of_range = IsingGraph::new(vec![0.0, 0.0], vec![1.0], vec![(0, 2)]);
        let short_j = IsingGraph::new(vec![0.0, 0.0], vec![], vec![(0, 1)]);
        let empty = IsingGraph::new(vec![], vec![], vec![]);
        let good = unit_graph(6, 1);
        for kernel in ResearchKernel::ALL {
            for (name, graph) in [
                ("nan", &nan),
                ("endpoint out of range", &out_of_range),
                ("fewer couplings than edges", &short_j),
                ("no nodes", &empty),
            ] {
                let err =
                    sample_research(graph, &params(4, 8, 1), kernel, None, None).expect_err(name);
                assert!(is_invalid(&err), "{kernel:?} {name}: {err:?}");
            }
            let err = sample_research(&good, &params(0, 8, 1), kernel, None, None)
                .expect_err("zero reads");
            assert!(is_invalid(&err), "{kernel:?} zero reads: {err:?}");
            let err = sample_research(&good, &params(MAX_READS + 1, 8, 1), kernel, None, None)
                .expect_err("too many reads");
            assert!(is_invalid(&err), "{kernel:?} too many reads: {err:?}");
        }
        // Seeds: wrong shape, bad value, more seeds than reads, on both MSA kernels.
        for kernel in [ResearchKernel::UnitMsa, ResearchKernel::FloatMsa] {
            let cases: Vec<(&str, Vec<Vec<i8>>, usize)> = vec![
                ("wrong length", vec![vec![1; 5]], 4),
                ("bad value", vec![vec![1, 1, 1, 0, 1, 1]], 4),
                ("more seeds than reads", vec![vec![1; 6], vec![-1; 6]], 1),
            ];
            for (name, seeds, reads) in cases {
                let start = SeededStart {
                    spins: &seeds,
                    start_beta: None,
                };
                let err = sample_research(&good, &params(reads, 8, 1), kernel, Some(start), None)
                    .expect_err(name);
                assert!(is_invalid(&err), "{kernel:?} {name}: {err:?}");
            }
        }
    }

    #[test]
    fn cancellation_is_reported_by_every_kernel() {
        let graph = unit_graph(8, 1);
        let token = CancelToken::default();
        token.cancel_through(3);
        for kernel in ResearchKernel::ALL {
            let err = sample_research(
                &graph,
                &params(4, 8, 1),
                kernel,
                None,
                Some((&token, Some(3))),
            )
            .expect_err("cancelled");
            assert_eq!(err, ResearchError::Cancelled, "{kernel:?}");
        }
    }
}
