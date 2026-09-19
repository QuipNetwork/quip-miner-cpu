//! Python binding to the multi-spin coded annealing kernel.
//!
//! One class, [`Msa`], with one method. A call builds the graph, runs the
//! anneal with the GIL released, and hands back numpy arrays. The kernel runs
//! one job on one core, so a Python caller gets parallelism by calling from
//! several threads.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use quip_miner_cpu::{IsingGraph, SaSampler, SaVariant, SampleParams, Sampler, SeededStart};

/// A multi-spin annealing sampler. It caches the graph colouring between
/// calls, so keep one instance per topology and share it across threads.
#[pyclass(frozen)]
struct Msa {
    sampler: SaSampler,
}

/// `(spins, energy_milli)`: int8 `(reads, nodes)` and int64 `(reads,)`.
type SampleOutput<'py> = (Bound<'py, PyArray2<i8>>, Bound<'py, PyArray1<i64>>);

/// `(h, j)`, both float64, in energy units.
type IsingDraw<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// Everything a run needs, copied out of numpy so the GIL can be released.
struct Run {
    graph: IsingGraph,
    params: SampleParams,
    seeds: Vec<Vec<i8>>,
    start_beta: Option<f64>,
}

fn contiguous<'a, T: numpy::Element>(
    name: &str,
    array: &'a PyReadonlyArray1<'_, T>,
) -> PyResult<&'a [T]> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a contiguous 1-D array")))
}

fn build_graph(
    h: &PyReadonlyArray1<'_, f64>,
    edges: &PyReadonlyArray2<'_, i64>,
    j: &PyReadonlyArray1<'_, f64>,
) -> PyResult<IsingGraph> {
    let h = contiguous("h", h)?;
    let j = contiguous("j", j)?;
    let edges = edges.as_array();
    if edges.ncols() != 2 {
        return Err(PyValueError::new_err("edges must have shape (m, 2)"));
    }
    if edges.nrows() != j.len() {
        return Err(PyValueError::new_err(format!(
            "{} couplings for {} edges",
            j.len(),
            edges.nrows()
        )));
    }
    // The kernel indexes by these without a bounds check of its own, so an
    // index outside the problem has to stop here, as a Python error.
    let node = |raw: i64| -> PyResult<usize> {
        usize::try_from(raw)
            .ok()
            .filter(|&v| v < h.len())
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "edge endpoint {raw} is outside 0..{}; edges index into h",
                    h.len()
                ))
            })
    };
    let mut pairs = Vec::with_capacity(edges.nrows());
    for row in edges.rows() {
        pairs.push((node(row[0])?, node(row[1])?));
    }
    Ok(IsingGraph::new(h.to_vec(), j.to_vec(), pairs))
}

#[pymethods]
impl Msa {
    #[new]
    fn new() -> Self {
        Self {
            sampler: SaSampler::new(SaVariant::MultiSpin),
        }
    }

    /// Anneal one Ising problem `sum(h s) + sum(J s s)`.
    ///
    /// `edges` holds dense indices into `h`. With `initial_spins`, read `r`
    /// starts from row `r`, reads past the last row start cold, and the
    /// anneal starts at `start_beta` (the geometric midpoint of the beta
    /// range when it is `None`). Returns `(spins, energy_milli)`: int8 of
    /// shape `(num_reads, len(h))` and int64 of shape `(num_reads,)`.
    #[pyo3(signature = (h, edges, j, *, num_sweeps, num_reads=64, seed=0,
                        beta_range=None, initial_spins=None, start_beta=None))]
    #[expect(
        clippy::too_many_arguments,
        reason = "one Python keyword per SampleParams field reads better than a dict"
    )]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        h: PyReadonlyArray1<'py, f64>,
        edges: PyReadonlyArray2<'py, i64>,
        j: PyReadonlyArray1<'py, f64>,
        num_sweeps: usize,
        num_reads: usize,
        seed: u64,
        beta_range: Option<(f64, f64)>,
        initial_spins: Option<PyReadonlyArray2<'py, i8>>,
        start_beta: Option<f64>,
    ) -> PyResult<SampleOutput<'py>> {
        if start_beta.is_some() && initial_spins.is_none() {
            return Err(PyValueError::new_err(
                "start_beta applies to a seeded run; pass initial_spins too",
            ));
        }
        if let Some(beta) = start_beta {
            if !beta.is_finite() || beta <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "start_beta must be a finite inverse temperature above zero; got {beta}"
                )));
            }
        }
        if num_reads < 1 {
            return Err(PyValueError::new_err(format!(
                "num_reads must be at least 1; got {num_reads}"
            )));
        }
        let graph = build_graph(&h, &edges, &j)?;
        let nodes = graph.h.len();
        let seeds = initial_spins.map_or_else(Vec::new, |states| {
            states
                .as_array()
                .rows()
                .into_iter()
                .map(|r| r.to_vec())
                .collect()
        });
        let run = Run {
            graph,
            params: SampleParams {
                num_reads,
                num_sweeps,
                seed,
                beta_range,
                ..Default::default()
            },
            seeds,
            start_beta,
        };

        let sampler = &self.sampler;
        let results = py
            .detach(move || {
                if run.seeds.is_empty() {
                    sampler
                        .sample(&run.graph, &run.params)
                        .map_err(|e| format!("{e:?}"))
                } else {
                    let start = SeededStart {
                        spins: &run.seeds,
                        start_beta: run.start_beta,
                    };
                    sampler
                        .sample_seeded(&run.graph, &run.params, start)
                        .map_err(|e| e.to_string())
                }
            })
            .map_err(PyValueError::new_err)?;

        let mut flat = Vec::with_capacity(results.len() * nodes);
        let mut energies = Vec::with_capacity(results.len());
        for result in &results {
            flat.extend_from_slice(&result.spins);
            energies.push(result.energy_milli);
        }
        let spins = Array2::from_shape_vec((results.len(), nodes), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((spins.into_pyarray(py), energies.into_pyarray(py)))
    }
}

/// Draw the Ising model of a nonce, as the protocol does: one h value per
/// node, then one J value per edge, in edge order. Values come back in energy
/// units (milli / 1000) so they feed `Msa.sample` directly.
#[pyfunction]
fn draw_ising<'py>(
    py: Python<'py>,
    nonce: &[u8],
    n_nodes: usize,
    n_edges: usize,
    allowed_h_milli: Vec<i32>,
    allowed_j_milli: Vec<i32>,
) -> PyResult<IsingDraw<'py>> {
    let nonce: [u8; 32] = nonce
        .try_into()
        .map_err(|_| PyValueError::new_err("nonce must be 32 bytes"))?;
    let (h_milli, j_milli) = quip_protocol::chacha8::draw_ising_milli(
        nonce,
        n_nodes,
        n_edges,
        &allowed_h_milli,
        &allowed_j_milli,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let to_units = |v: Vec<i32>| {
        v.into_iter()
            .map(|m| f64::from(m) / 1000.0)
            .collect::<Vec<f64>>()
    };
    Ok((
        PyArray1::from_vec(py, to_units(h_milli)),
        PyArray1::from_vec(py, to_units(j_milli)),
    ))
}

/// The (hot, cold) beta range `Msa.sample` uses when `beta_range` is `None`.
#[pyfunction]
fn default_beta_range(
    h: PyReadonlyArray1<'_, f64>,
    edges: PyReadonlyArray2<'_, i64>,
    j: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64)> {
    let graph = build_graph(&h, &edges, &j)?;
    Ok(quip_solver_core::beta::default_ising_beta_range(&graph))
}

/// Python module `quip_msa`.
#[pymodule]
fn quip_msa(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Msa>()?;
    m.add_function(wrap_pyfunction!(draw_ising, m)?)?;
    m.add_function(wrap_pyfunction!(default_beta_range, m)?)?;
    Ok(())
}
