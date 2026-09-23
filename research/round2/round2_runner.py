"""The controlled CPU comparison of Round 2: identity, host contamination,
and per-job execution shared by ``scripts/round2_cpu.py``.

One run compares ``cpu-sa``, ``cpu-msa-f64``, and ``cpu-msa-unit`` on the
IDENTICAL logical model -- the canonical ``(h, edges, j)`` a Task 2 bundle
carries, never a kernel-specific repeated-unit reconstruction (Task 8 brief,
step 2, and the design's "Weighted MSA contract": recompute output energy
from the original model in float64). ``cpu-msa`` (the auto-selecting arm)
never appears here: the controlled comparison always names its kernel.

The unit kernel's own eligibility is decided empirically, by calling it and
catching the ``ValueError`` it raises on a model it cannot take (the task
brief: "The unit kernel raises an error on models it cannot take. Treat that
as an explicit 'unsupported' record and never substitute SA."). For every
cell but cubic-dimer-pm1, the bundle's canonical ``(h, edges, j)`` is fed to
the kernel unchanged. cubic-dimer-pm1 needs one exception: its canonical
edges are not unit-valued for the unit kernel (a z-bond splits into two
logical half-unit edges at export time), so :func:`kernel_input_for`
reconstructs the repeated-unit input round2_export.py's own
``unit_kernel_input_hash`` already verifies, and :func:`execute_cpu_job`
checks the reconstruction's hash against the bundle's recorded one before
using it. See the task-8 report's Assumptions section for the evidence this
is based on.

Every measured run is meant to happen in its own fresh subprocess (a fresh
``Msa()``, no coloring-cache carryover, an isolated peak RSS): this module
holds the logic that subprocess runs (:func:`execute_cpu_job`), plus the
identity, ordering, host-contamination, and QPU-deadline-record-shape
primitives ``scripts/round2_cpu.py`` orchestrates around it.
"""

from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from quip_miner_dwave import regimes, round2_io

PathLike = os.PathLike

# ---------------------------------------------------------------- constants

#: The five sweep depths of the complete CPU comparison (controller ruling 1).
SWEEP_DEPTHS: Tuple[int, ...] = (512, 2048, 8192, 32768, 131072)

#: The pilot's sweep depths -- the first two rungs of the full ladder (controller ruling 1).
PILOT_SWEEP_DEPTHS: Tuple[int, ...] = (512, 2048)

#: Kernels of the controlled comparison. ``cpu-msa`` (auto-select) never appears here.
CONTROLLED_KERNELS: Tuple[str, ...] = ("cpu-sa", "cpu-msa-f64", "cpu-msa-unit")

#: The kernel whose eligibility is a per-model, empirically-checked outcome.
UNIT_KERNEL = "cpu-msa-unit"

PILOT_MODELS_PER_CELL = 5
PILOT_READS = 64
PILOT_TIMING_REPS = 3
CAMPAIGN_MODELS_PER_CELL = 100
CAMPAIGN_READS = 64

#: Task 8 brief step 9: the seeded/cold weighted-MSA comparison at one deep rung.
SEEDED_SWEEPS = 32768
SEED_LANES = 32
COLD_LANES = 32
SEEDED_SWEEP_READS = SEED_LANES + COLD_LANES
#: Round 1's own MSA-lite convention (``scripts/seeded_sweep.py``'s ``--lite-sweeps``
#: default), reused here as the CPU-lite seed source rather than reinvented.
CPU_LITE_SWEEPS = 512
SEED_SOURCES = ("qpu", "cpu-lite")

#: Seeded shuffles that fix "randomized solver order" (spec, initial campaign
#: proposal) without leaving it to whatever order Python happens to iterate in.
PILOT_ORDER_SEED = 20260922
CAMPAIGN_ORDER_SEED = 20260923

REPETITION_TIMING = "timing"
REPETITION_QUALITY = "quality"

# ------------------------------------------------------------- the one rule


def deadline_status(elapsed_s: float, deadline_s: float, exit_ok: bool) -> str:
    """"failed" / "timeout" / "completed" -- the rule the task brief pins down verbatim.

    A good, late answer is a timeout, not a win: never reclassify it (the
    design's portfolio-replication contract says exactly this about the
    historical 10-second deadline). A bad exit is "failed" even if it would
    have been on time -- lateness is not the only way a run can fail.
    """
    if not exit_ok:
        return "failed"
    if elapsed_s > deadline_s:
        return "timeout"
    return "completed"


class RunnerError(RuntimeError):
    """A controlled-comparison invariant broke: wrong observed kernel, nonfinite score."""


# --------------------------------------------------------- host contamination

#: A run is contaminated if the 1-minute loadavg, before or after, implies more
#: than half of the 16 physical cores carry other runnable work. This machine
#: is "shared with other work" (controller ruling 5); half the physical cores
#: is a documented, not measured, choice -- it is generous enough not to flag
#: ordinary background load, and tight enough to catch a second heavy job.
PHYSICAL_CORES = 16
LOADAVG_1M_CONTAMINATION_THRESHOLD = PHYSICAL_CORES / 2.0

#: A run is contaminated if its pinned core's SMT sibling was busy more than
#: this fraction of the run's wall time. 20% is a documented, not measured,
#: choice: enough slack for a brief kernel interrupt on the sibling, tight
#: enough to catch another process actually sharing the physical core.
SIBLING_BUSY_CONTAMINATION_THRESHOLD = 0.20

_PROC_STAT = Path("/proc/stat")


def cpu_sibling(cpu: int) -> Optional[int]:
    """The other logical CPU on ``cpu``'s physical core, or None with no SMT sibling."""
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    if not path.exists():
        return None
    ids = [int(x) for x in path.read_text(encoding="utf-8").strip().split(",")]
    others = [i for i in ids if i != cpu]
    return others[0] if others else None


def cpufreq_governor(cpu: int) -> Optional[str]:
    """The cpufreq scaling governor of ``cpu``, or None where cpufreq is not exposed."""
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def read_cpu_times(cpu: int) -> Tuple[float, float]:
    """``(busy, total)`` jiffies of one logical CPU from ``/proc/stat``.

    ``busy`` excludes idle and iowait, matching the usual definition of
    "doing work" for a contamination check on a single sibling thread.
    """
    prefix = f"cpu{cpu} "
    for line in _PROC_STAT.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            fields = [float(x) for x in line[len(prefix):].split()]
            idle = fields[3] + (fields[4] if len(fields) > 4 else 0.0)
            total = sum(fields)
            return total - idle, total
    raise ValueError(f"{_PROC_STAT} has no line for cpu{cpu}")


def busy_fraction(before: Tuple[float, float], after: Tuple[float, float]) -> Optional[float]:
    """Share of elapsed jiffies spent busy between two :func:`read_cpu_times` samples.

    None when the two samples cover zero elapsed jiffies (a run too short for
    the jiffy clock to tick), rather than a division that would otherwise
    silently read as "not busy."
    """
    busy_before, total_before = before
    busy_after, total_after = after
    elapsed_total = total_after - total_before
    if elapsed_total <= 0:
        return None
    return (busy_after - busy_before) / elapsed_total


@dataclass(frozen=True)
class HostSample:
    """One point-in-time snapshot of the host, taken around a measured run."""

    loadavg_1m: float
    loadavg_5m: float
    loadavg_15m: float
    governor: Optional[str]
    sibling_cpu_times: Optional[Tuple[float, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loadavg_1m": self.loadavg_1m,
            "loadavg_5m": self.loadavg_5m,
            "loadavg_15m": self.loadavg_15m,
            "governor": self.governor,
        }


def sample_host(cpu: int, sibling: Optional[int]) -> HostSample:
    load1, load5, load15 = os.getloadavg()
    sibling_times = read_cpu_times(sibling) if sibling is not None else None
    return HostSample(load1, load5, load15, cpufreq_governor(cpu), sibling_times)


@dataclass(frozen=True)
class ContaminationVerdict:
    contaminated: bool
    reasons: Tuple[str, ...]
    sibling_busy_fraction: Optional[float]
    thresholds: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contaminated": self.contaminated,
            "reasons": list(self.reasons),
            "sibling_busy_fraction": self.sibling_busy_fraction,
            "thresholds": self.thresholds,
        }


def check_contamination(before: HostSample, after: HostSample) -> ContaminationVerdict:
    """Whether a run between ``before`` and ``after`` is contaminated, and why.

    Never drops a contaminated run: this only labels it, so the caller can
    keep the record and repeat the run under the same manifest (task brief,
    step 4).
    """
    reasons: List[str] = []
    if before.loadavg_1m > LOADAVG_1M_CONTAMINATION_THRESHOLD:
        reasons.append(
            f"loadavg_1m before ({before.loadavg_1m}) exceeds {LOADAVG_1M_CONTAMINATION_THRESHOLD}"
        )
    if after.loadavg_1m > LOADAVG_1M_CONTAMINATION_THRESHOLD:
        reasons.append(
            f"loadavg_1m after ({after.loadavg_1m}) exceeds {LOADAVG_1M_CONTAMINATION_THRESHOLD}"
        )
    sibling_busy: Optional[float] = None
    if before.sibling_cpu_times is not None and after.sibling_cpu_times is not None:
        sibling_busy = busy_fraction(before.sibling_cpu_times, after.sibling_cpu_times)
        if sibling_busy is not None and sibling_busy > SIBLING_BUSY_CONTAMINATION_THRESHOLD:
            reasons.append(
                f"sibling busy fraction ({sibling_busy:.3f}) exceeds "
                f"{SIBLING_BUSY_CONTAMINATION_THRESHOLD}"
            )
    return ContaminationVerdict(
        contaminated=bool(reasons),
        reasons=tuple(reasons),
        sibling_busy_fraction=sibling_busy,
        thresholds={
            "loadavg_1m_max": LOADAVG_1M_CONTAMINATION_THRESHOLD,
            "sibling_busy_max": SIBLING_BUSY_CONTAMINATION_THRESHOLD,
        },
    )


# ------------------------------------------------------------------ identity


def seed_for(model_hash: str, kernel: str, sweeps: int, reads: int, variant: str) -> Tuple[int, str]:
    """``(seed, seed_input_hash)`` for one arm.

    Deterministic in ``(model_hash, kernel, sweeps, reads, variant)`` only --
    NOT in the repetition id, so timing repetitions of the same arm reuse the
    same seed (task brief, step 1) while a distinct ``variant`` (a quality
    repetition's index, or "cold"/a seed-lane index for step 9) draws an
    independently seeded run.
    """
    seed_input_hash = round2_io.run_id(
        {"model_hash": model_hash, "kernel": kernel, "sweeps": sweeps, "reads": reads, "variant": variant}
    )
    seed = int(seed_input_hash[:15], 16) % (1 << 63)
    return seed, seed_input_hash


@dataclass(frozen=True)
class CpuJob:
    """One controlled-comparison job: everything :func:`execute_cpu_job` needs, plus its identity."""

    cell: str
    nonce: str
    kernel: str
    sweeps: int
    reads: int
    repetition_id: int
    repetition_kind: str
    variant: str
    seed: int
    seed_input_hash: str

    def run_key(self, bundles_root: PathLike) -> str:
        """The resume identity of this job: change any field here and it is a different run."""
        return round2_io.run_id(
            {
                "bundles_root": str(bundles_root),
                "cell": self.cell,
                "nonce": self.nonce,
                "kernel": self.kernel,
                "sweeps": self.sweeps,
                "reads": self.reads,
                "repetition_id": self.repetition_id,
                "repetition_kind": self.repetition_kind,
                "variant": self.variant,
                "seed": self.seed,
                "seed_input_hash": self.seed_input_hash,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "CpuJob":
        return CpuJob(**{field: payload[field] for field in CpuJob.__dataclass_fields__})


def _seeded_shuffle(items: Sequence[Any], seed: int) -> List[Any]:
    order = list(items)
    import random

    random.Random(seed).shuffle(order)
    return order


def _sorted_bundle_rows(index: Dict[str, Any], cell: str) -> List[Dict[str, Any]]:
    rows = [row for row in index["rows"] if row["cell"] == cell]
    rows.sort(key=lambda row: row["nonce"])
    return rows


def build_pilot_jobs(index: Dict[str, Any], cells: Sequence[str]) -> List[CpuJob]:
    """The five-model pilot: the first :data:`PILOT_MODELS_PER_CELL` sorted nonces per
    cell, :data:`PILOT_SWEEP_DEPTHS`, every controlled kernel, three timing
    repetitions with a reused seed, in a seeded ("randomized solver order") order.
    """
    jobs: List[CpuJob] = []
    for cell in cells:
        rows = _sorted_bundle_rows(index, cell)[:PILOT_MODELS_PER_CELL]
        for row in rows:
            for sweeps in PILOT_SWEEP_DEPTHS:
                for kernel in CONTROLLED_KERNELS:
                    variant = "timing"
                    seed, seed_input_hash = seed_for(row["model_hash"], kernel, sweeps, PILOT_READS, variant)
                    for rep in range(PILOT_TIMING_REPS):
                        jobs.append(
                            CpuJob(
                                cell=cell, nonce=row["nonce"], kernel=kernel, sweeps=sweeps,
                                reads=PILOT_READS, repetition_id=rep, repetition_kind=REPETITION_TIMING,
                                variant=variant, seed=seed, seed_input_hash=seed_input_hash,
                            )
                        )
    return _seeded_shuffle(jobs, PILOT_ORDER_SEED)


def build_campaign_jobs(index: Dict[str, Any], cells: Sequence[str]) -> List[CpuJob]:
    """The complete CPU comparison: :data:`CAMPAIGN_MODELS_PER_CELL` models per cell,
    :data:`SWEEP_DEPTHS`, every controlled kernel, one timing repetition each.
    """
    jobs: List[CpuJob] = []
    for cell in cells:
        rows = _sorted_bundle_rows(index, cell)[:CAMPAIGN_MODELS_PER_CELL]
        for row in rows:
            for sweeps in SWEEP_DEPTHS:
                for kernel in CONTROLLED_KERNELS:
                    variant = "timing"
                    seed, seed_input_hash = seed_for(row["model_hash"], kernel, sweeps, CAMPAIGN_READS, variant)
                    jobs.append(
                        CpuJob(
                            cell=cell, nonce=row["nonce"], kernel=kernel, sweeps=sweeps,
                            reads=CAMPAIGN_READS, repetition_id=0, repetition_kind=REPETITION_TIMING,
                            variant=variant, seed=seed, seed_input_hash=seed_input_hash,
                        )
                    )
    return _seeded_shuffle(jobs, CAMPAIGN_ORDER_SEED)


# -------------------------------------------------------------- job execution


def _msa():
    import quip_msa  # pyright: ignore[reportMissingModuleSource]

    return quip_msa


#: cubic-dimer-pm1's unit bond, in energy units (UNIT_MILLI["cubic-dimer-pm1"] = 500 milli
#: in quip_miner_dwave.regimes, divided by the milli-to-energy-units factor of 1000).
CUBIC_DIMER_UNIT_ENERGY = 0.5


def _array_input_hash(h: np.ndarray, edges: np.ndarray, j: np.ndarray) -> str:
    """The identity of one kernel input, in the same shape ``round2_export.py``'s
    ``unit_kernel_input_hash`` uses, so a reconstruction here can be checked against it.
    """
    return round2_io.run_id(
        {
            "h": round2_io.array_hash(np.asarray(h, dtype=np.float64)),
            "edges": round2_io.array_hash(np.asarray(edges, dtype=np.int64)),
            "j": round2_io.array_hash(np.asarray(j, dtype=np.float64)),
        }
    )


def kernel_input_for(
    cell: str, kernel: str, h: np.ndarray, edges: np.ndarray, j: np.ndarray, beta_range: Tuple[float, float],
) -> Dict[str, Any]:
    """The exact ``(h, edges, j, beta_range)`` ``kernel`` takes for ``cell``, plus the
    energy scale needed to convert its reported energies back to the canonical model.

    Every cell but cubic-dimer-pm1 empirically accepts its bundle's canonical
    ``(h, edges, j)`` directly for every controlled kernel, including
    ``cpu-msa-unit`` (verified against every cell's frozen Round 2 bundle;
    see the task-8 report's Assumptions). cubic-dimer-pm1's canonical edges
    are NOT unit-valued for the unit kernel: a z-bond splits into two
    logical half-unit edges at export time (``round2_export.py``'s
    docstring), so each edge must be repeated ``|j| / unit`` times at
    coupling +-1 first -- reconstructed here to match
    ``round2_export.py``'s own ``unit_kernel_input_hash`` exactly, never
    invented fresh (:func:`execute_cpu_job` checks the match when the bundle
    recorded one).

    Repeating an edge at coupling +-1 instead of +-``unit`` means the unit
    kernel's reported energy is the canonical energy divided by ``unit``:
    the returned ``energy_scale`` corrects for exactly that, and the
    returned ``beta_range`` is scaled by ``unit`` too, so the annealing
    schedule stays physically equivalent to the canonical (energy_scale=1)
    case rather than running at ``unit`` times the intended temperature.
    """
    if not (cell == "cubic-dimer-pm1" and kernel == UNIT_KERNEL):
        return {
            "h": h, "edges": edges, "j": j, "beta_range": beta_range,
            "energy_scale": 1.0, "input_hash": _array_input_hash(h, edges, j),
        }
    unit = CUBIC_DIMER_UNIT_ENERGY
    ratios = np.abs(j) / unit
    rounded = np.rint(ratios).astype(np.int64)
    if not np.allclose(ratios, rounded, rtol=0.0, atol=1e-9):
        raise RunnerError(
            f"cubic-dimer-pm1: found a coupling that is not a multiple of the unit {unit}; "
            "cannot build a repeated-unit input for cpu-msa-unit"
        )
    kernel_edges = np.repeat(edges, rounded, axis=0)
    kernel_j = np.repeat(np.sign(j), rounded).astype(np.float64)
    kernel_h = np.zeros_like(np.asarray(h, dtype=np.float64))
    kernel_beta = (float(beta_range[0]) * unit, float(beta_range[1]) * unit)
    return {
        "h": kernel_h, "edges": kernel_edges, "j": kernel_j, "beta_range": kernel_beta,
        "energy_scale": unit, "input_hash": _array_input_hash(kernel_h, kernel_edges, kernel_j),
    }


def execute_cpu_job(
    job: CpuJob, bundles_root: PathLike
) -> Tuple[Dict[str, Any], Optional[Dict[str, np.ndarray]]]:
    """Run one :class:`CpuJob` in the current process; return ``(record, samples)``.

    ``record`` is JSON-safe. ``samples`` is ``{"spins": ..., "energies": ...}``
    on a completed run, and ``None`` for an unsupported arm or a failure --
    there is nothing to save in either case.

    Meant to run inside a fresh subprocess (``scripts/round2_cpu.py run-one``):
    a fresh ``Msa()`` here, and :func:`_peak_rss_kb` read at the very end, are
    both only meaningful when this process has done nothing else.

    Never substitutes ``cpu-sa`` for an ineligible ``cpu-msa-unit``: an
    ineligible unit kernel comes back with ``unsupported=True`` and no
    samples, not a different kernel's answer.
    """
    quip_msa = _msa()
    t_start = time.perf_counter()

    manifest, arrays = round2_io.read_bundle(Path(bundles_root) / job.cell / job.nonce)
    h, edges, j = arrays["h"], arrays["edges"], arrays["j"]

    t_setup_start = time.perf_counter()
    sampler = quip_msa.Msa()
    beta_range = quip_msa.default_beta_range(h, edges, j)
    t_setup_end = time.perf_counter()

    kernel_input = kernel_input_for(job.cell, job.kernel, h, edges, j, beta_range)
    bundle_unit_hash = manifest.get("unit_kernel_input_hash")
    if job.kernel == UNIT_KERNEL and bundle_unit_hash is not None and bundle_unit_hash != kernel_input["input_hash"]:
        raise RunnerError(
            f"{job.cell}/{job.nonce}: the reconstructed repeated-unit input hashes to "
            f"{kernel_input['input_hash']!r}, and the bundle records {bundle_unit_hash!r}; refusing "
            "to submit an input that does not match round2_export.py's own reconstruction"
        )

    record: Dict[str, Any] = {
        "schema": "round2-cpu-run-v1",
        "run_key": job.run_key(bundles_root),
        "cell": job.cell,
        "nonce": job.nonce,
        "model_hash": manifest["hash"],
        "requested_kernel": job.kernel,
        "reads": job.reads,
        "sweeps": job.sweeps,
        "beta_range": [float(beta_range[0]), float(beta_range[1])],
        "submitted_beta_range": [float(kernel_input["beta_range"][0]), float(kernel_input["beta_range"][1])],
        "kernel_energy_scale": kernel_input["energy_scale"],
        "kernel_input_hash": kernel_input["input_hash"],
        "seed": job.seed,
        "seed_input_hash": job.seed_input_hash,
        "repetition_id": job.repetition_id,
        "repetition_kind": job.repetition_kind,
        "variant": job.variant,
        "setup_s": t_setup_end - t_setup_start,
        "graph_setup_s": None,  # not exposed by quip_msa's API; see module docstring
    }

    try:
        t_sample_start = time.perf_counter()
        spins, energies, meta = sampler.sample_research(
            kernel_input["h"], kernel_input["edges"], kernel_input["j"], kernel=job.kernel,
            num_sweeps=job.sweeps, num_reads=job.reads, seed=job.seed, beta_range=kernel_input["beta_range"],
        )
        t_sample_end = time.perf_counter()
    except ValueError as exc:
        if job.kernel == UNIT_KERNEL:
            record.update(
                unsupported=True, unsupported_reason=str(exc), exit_ok=True, error=None,
                observed_kernel=None, representation=None, workspace_bytes=None,
                elapsed_sampling_s=None, best_energy=None, mean_energy=None, unique_reads=None,
            )
            record["wall_s"] = time.perf_counter() - t_start
            record["peak_rss_kb"] = _peak_rss_kb()
            return record, None
        record.update(
            unsupported=False, unsupported_reason=None, exit_ok=False, error=f"{type(exc).__name__}: {exc}",
            observed_kernel=None, representation=None, workspace_bytes=None,
            elapsed_sampling_s=None, best_energy=None, mean_energy=None, unique_reads=None,
        )
        record["wall_s"] = time.perf_counter() - t_start
        record["peak_rss_kb"] = _peak_rss_kb()
        return record, None

    observed_kernel = meta["observed_kernel"]
    if observed_kernel != job.kernel:
        raise RunnerError(
            f"{job.cell}/{job.nonce}: requested kernel {job.kernel!r} but observed "
            f"{observed_kernel!r}; the controlled comparison never accepts a substitution"
        )

    # Always rescore from the CANONICAL (h, edges, j) -- never the possibly-rescaled
    # kernel input -- per the Weighted MSA contract: "Recompute output energy from the
    # original model in float64." The kernel's own reported energies are in its own
    # (possibly rescaled) units; kernel_energy_scale converts them back for comparison.
    rescored = regimes.energy(spins, h, edges, j)
    kernel_reported_logical = np.asarray(energies, dtype=np.float64) * kernel_input["energy_scale"]
    if not np.isfinite(rescored).all() or not np.isfinite(kernel_reported_logical).all():
        raise RunnerError(
            f"{job.cell}/{job.nonce}/{job.kernel}: a nonfinite score came back "
            "(kernel-reported or independently rescored); refusing to record it as a result"
        )
    if not np.allclose(rescored, kernel_reported_logical, rtol=1e-9, atol=1e-6):
        raise RunnerError(
            f"{job.cell}/{job.nonce}/{job.kernel}: the kernel's reported energies do not match "
            "an independent float64 rescoring of the original model"
        )

    record.update(
        unsupported=False, unsupported_reason=None, exit_ok=True, error=None,
        observed_kernel=observed_kernel, representation=meta["representation"],
        workspace_bytes=meta["workspace_bytes"],
        elapsed_sampling_s=t_sample_end - t_sample_start,
        best_energy=float(rescored.min()), mean_energy=float(rescored.mean()),
        unique_reads=int(len(np.unique(spins, axis=0))),
    )
    record["wall_s"] = time.perf_counter() - t_start
    record["peak_rss_kb"] = _peak_rss_kb()
    return record, {"spins": spins, "energies": rescored}


def _peak_rss_kb() -> int:
    """This process's own peak resident set size, in KiB (Linux ``ru_maxrss`` units).

    Explicitly NOT kernel memory (task brief, step 3): it is everything this
    Python process has ever touched, interpreter and libraries included.
    Only meaningful read once, at the end of a fresh, single-job subprocess.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


_RECORD_FIELDS = (
    "unsupported", "unsupported_reason", "exit_ok", "error", "observed_kernel", "representation",
    "workspace_bytes", "elapsed_sampling_s", "best_energy", "mean_energy", "unique_reads",
    "setup_s", "graph_setup_s", "beta_range", "wall_s", "peak_rss_kb",
)


def failure_record(job: CpuJob, bundles_root: PathLike, exc: BaseException) -> Dict[str, Any]:
    """A well-formed, fully-shaped run record for a job that raised before it could produce one.

    Used for anything :func:`execute_cpu_job` did not itself catch: a
    :class:`RunnerError` invariant violation, a bundle read failure, or any
    other unexpected exception. Never drops the job silently -- the record
    always exists, with every field :func:`execute_cpu_job` would have set,
    so downstream analysis never has to special-case a missing key.
    """
    record: Dict[str, Any] = {
        "schema": "round2-cpu-run-v1",
        "run_key": job.run_key(bundles_root),
        "cell": job.cell,
        "nonce": job.nonce,
        "model_hash": None,
        "requested_kernel": job.kernel,
        "reads": job.reads,
        "sweeps": job.sweeps,
        "seed": job.seed,
        "seed_input_hash": job.seed_input_hash,
        "repetition_id": job.repetition_id,
        "repetition_kind": job.repetition_kind,
        "variant": job.variant,
    }
    for field in _RECORD_FIELDS:
        record[field] = None
    record["unsupported"] = False
    record["exit_ok"] = False
    record["error"] = f"{type(exc).__name__}: {exc}"
    return record


# ------------------------------------------------------- subprocess orchestration

#: Crash protection only (task brief, step 5: "a subprocess timeout and
#: cleanup"). This is NOT the application/comparison deadline a caller
#: classifies with :func:`deadline_status` -- see :func:`run_one_subprocess`.
DEFAULT_HARD_DEADLINE_S = 600.0


def run_one_subprocess(
    job: CpuJob,
    *,
    bundles_root: PathLike,
    out_dir: PathLike,
    cpu: int,
    script_path: PathLike,
    python_exe: Optional[str] = None,
    hard_deadline_s: float = DEFAULT_HARD_DEADLINE_S,
) -> Tuple[bool, float]:
    """Run one job in a fresh ``run-one`` subprocess, pinned to ``cpu``. Returns ``(exit_ok, wall_s)``.

    ``hard_deadline_s`` only protects the orchestrator from a runaway or
    hung subprocess: a process that blows through it is killed (its whole
    process group, not just the direct child) and the run comes back
    ``exit_ok=False``. It never means "timeout" in :func:`deadline_status`'s
    sense of a late-but-good answer -- an application deadline like the
    portfolio historical arm's 10 seconds is classified separately, from
    the subprocess's own reported elapsed time, only once it has actually
    returned an answer (``exit_ok=True``). One-process-at-a-time timing
    (task brief, step 4) is the caller's responsibility: this starts exactly
    one subprocess and waits for it before returning.
    """
    python_exe = python_exe or sys.executable
    cmd = [
        python_exe, str(script_path), "run-one",
        "--job-json", json.dumps(job.to_dict()),
        "--bundles-root", str(bundles_root),
        "--out-dir", str(out_dir),
        "--cpu", str(cpu),
    ]
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        returncode = proc.wait(timeout=hard_deadline_s)
        return returncode == 0, time.perf_counter() - start
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return False, time.perf_counter() - start


# ------------------------------------------------------- QPU deadline record


def build_qpu_deadline_record(
    *,
    elapsed_s: Optional[float],
    deadline_s: float,
    exit_ok: bool,
    charge_known: bool,
    access_us: Optional[int] = None,
    reconciled_at: Optional[str] = None,
) -> Dict[str, Any]:
    """The record shape of a QPU application-deadline comparison (task brief, step 6).

    Task 9 fills in the real numbers; this only defines and validates the
    shape, offline. ``on_time`` and ``charge`` are separate sections on
    purpose: an on-time answer never implies the charge is known, and a
    timeout never implies the charge is zero -- "Do not use a wall timeout as
    proof that QPU spend is zero" (task brief, step 6). ``elapsed_s`` is None
    for a run that never returned an answer at all (a hard kill before any
    wall-clock reading was possible); ``exit_ok`` must be False in that case.
    """
    if elapsed_s is None and exit_ok:
        raise ValueError("exit_ok cannot be True with no elapsed_s: no answer came back to time")
    status = deadline_status(elapsed_s if elapsed_s is not None else float("inf"), deadline_s, exit_ok)
    if not charge_known and access_us is not None:
        raise ValueError("access_us must be None when charge_known is False; a timeout proves nothing about spend")
    if charge_known and access_us is None:
        raise ValueError("access_us is required once charge_known is True")
    if charge_known and reconciled_at is None:
        raise ValueError("reconciled_at is required once charge_known is True")
    return {
        "schema": "round2-qpu-deadline-v1",
        "on_time": {
            "elapsed_s": elapsed_s,
            "deadline_s": deadline_s,
            "exit_ok": exit_ok,
            "status": status,
        },
        "charge": {
            "known": charge_known,
            "access_us": access_us,
            "reconciled_at": reconciled_at,
            "note": (
                "a wall timeout is not proof of zero QPU spend; charge stays unknown until "
                "reconciled against the provider, however long that takes"
            ),
        },
    }


# ------------------------------------------------------------ campaign sizing


def estimate_campaign(
    pilot_wall_s: Dict[Tuple[str, int], float],
    *,
    cells: int = 5,
    models_per_cell: int = CAMPAIGN_MODELS_PER_CELL,
    depths: Sequence[int] = SWEEP_DEPTHS,
    kernels: Sequence[str] = CONTROLLED_KERNELS,
    unsupported_kernel_cells: int = 0,
    workers: int = 12,
) -> Dict[str, Any]:
    """Linear-in-sweeps extrapolation of the pilot's median wall time to the full campaign.

    ``pilot_wall_s`` maps ``(kernel, sweeps)`` to a representative (median)
    per-job wall time observed in the pilot, for :data:`PILOT_SWEEP_DEPTHS`.
    Every other depth is estimated by scaling the nearest measured depth's
    wall time linearly in the sweep count -- sampling cost is dominated by a
    fixed number of per-sweep updates, so this is a first-order model, not a
    measurement; :func:`execute_cpu_job` records the real wall time of every
    campaign job, superseding this estimate once observed.

    ``unsupported_kernel_cells`` scales down ``cpu-msa-unit``'s job count for
    campaign cells known (from the pilot) never to run it -- an unsupported
    arm cost is negligible (no sampling occurs) but is not exactly zero, so it
    is dropped from the core-second total, not double-counted as an SA run.
    """
    total_core_s = 0.0
    per_kernel: Dict[str, float] = {}
    for kernel in kernels:
        kernel_cells = cells - unsupported_kernel_cells if kernel == UNIT_KERNEL else cells
        kernel_core_s = 0.0
        for sweeps in depths:
            if (kernel, sweeps) in pilot_wall_s:
                per_job = pilot_wall_s[(kernel, sweeps)]
            else:
                nearest = min(
                    (d for (k, d) in pilot_wall_s if k == kernel), key=lambda d: abs(d - sweeps), default=None,
                )
                if nearest is None:
                    raise ValueError(f"no pilot timing for kernel {kernel!r} to extrapolate from")
                per_job = pilot_wall_s[(kernel, nearest)] * (sweeps / nearest)
            kernel_core_s += per_job * kernel_cells * models_per_cell
        per_kernel[kernel] = kernel_core_s
        total_core_s += kernel_core_s
    return {
        "total_core_s": total_core_s,
        "per_kernel_core_s": per_kernel,
        "workers": workers,
        "estimated_wall_s": total_core_s / workers,
        "estimated_wall_hours": total_core_s / workers / 3600.0,
    }


# ------------------------------------------------------- portfolio deadline arm

#: The historical portfolio SA deadline (design's "Portfolio replication contract").
PORTFOLIO_DEADLINE_S = 10.0
#: The reference provider's own settings (design: "500 reads and 500 sweeps, as specified
#: by the reference provider").
PORTFOLIO_NEAL_READS = 500
PORTFOLIO_NEAL_SWEEPS = 500
#: The 18- and 28-asset baskets the design's initial campaign proposal and Task 3's
#: fixtures already share (``scripts/portfolio_replication.py``'s ``FIXTURES``).
PORTFOLIO_DEADLINE_BASKETS: Tuple[Tuple[int, int], ...] = ((18, 6), (28, 9))
PORTFOLIO_BETA_LABELS: Tuple[str, ...] = ("beta-zero", "beta-nonzero")


def build_portfolio_deadline_problem(n: int, k: int, beta_label: str) -> Any:
    """One in-memory ``qpo.problem.PortfolioProblem`` for the deadline arm.

    Requires P's pinned environment (``qpo``) on the path; import
    :mod:`quip_miner_dwave.portfolio_replication` lazily, and run this only
    under P's interpreter (see the module docstring / task-8 report for the
    exact invocation). Never persisted as a Round 2 bundle: Task 3's fixture
    contract has no "beta zero" mode at 18 or 28 assets (both are above
    ``qpo.config.CARDINALITY_BETA_N_MIN``, so its own ramps are never zero
    there), and this arm does not need a persisted bundle -- it scores
    directly against a freshly-built ``PortfolioProblem``, the same
    deterministic, seeded synthetic instance Task 3's own fixture script
    builds (``synthetic_one_factor_problem(n, k, seed=n)``).
    """
    from quip_miner_dwave import portfolio_replication as pr

    if beta_label not in PORTFOLIO_BETA_LABELS:
        raise ValueError(f"beta_label must be one of {PORTFOLIO_BETA_LABELS}, got {beta_label!r}")
    problem = pr.synthetic_one_factor_problem(n, k, seed=n)
    problem.frustration_beta = 0.0 if beta_label == "beta-zero" else pr.repository_default_beta(problem)
    problem.invalidate_cache()
    return problem


def run_portfolio_deadline_arm(n: int, k: int, beta_label: str, seed: int) -> Dict[str, Any]:
    """The historical-settings neal arm (500 reads / 500 sweeps) under the 10 s application deadline.

    dwave-neal (``dwave.samplers.SimulatedAnnealingSampler``) has no internal
    deadline concept at all -- "the reference SA provider itself does not
    enforce it" (design's portfolio-replication contract) -- so this is the
    orchestration that contract calls for: run to completion, never killed
    at 10 seconds, classify with :func:`deadline_status` from the measured
    elapsed time, and KEEP a late-but-good record for diagnosis rather than
    reclassify it as an on-time win. Requires P's pinned environment.
    """
    import dimod
    from dwave.samplers import SimulatedAnnealingSampler

    from quip_miner_dwave import portfolio_replication as pr

    problem = build_portfolio_deadline_problem(n, k, beta_label)
    qubo, _h, _edges, _j, _offset = pr.encode_to_ising(problem)
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo.to_dict())
    sampler = SimulatedAnnealingSampler()

    # dwave-neal's seed must fit a SIGNED 32-bit int, 0 to 2**31 - 1: verified directly
    # (its own error message claims "0 and 2^32 - 1", which is wrong -- 2**31 itself,
    # and every value up to 2**32 - 1, is rejected). quip_msa's own seeds (what
    # seed_for() produces) can be up to 63 bits. The value actually used is what gets
    # recorded, so the record always reflects what really seeded the reference sampler.
    neal_seed = seed % (1 << 31)

    t_start = time.perf_counter()
    exit_ok = True
    error: Optional[str] = None
    response = None
    try:
        response = sampler.sample(
            bqm, num_reads=PORTFOLIO_NEAL_READS, num_sweeps=PORTFOLIO_NEAL_SWEEPS, seed=neal_seed
        )
    except Exception as exc:  # a real, observed failure of the reference sampler
        exit_ok = False
        error = f"{type(exc).__name__}: {exc}"
    elapsed_s = time.perf_counter() - t_start

    record: Dict[str, Any] = {
        "schema": "round2-portfolio-deadline-v1",
        "n_assets": n, "cardinality_k": k, "beta_label": beta_label,
        "frustration_beta": float(problem.frustration_beta),
        "reads": PORTFOLIO_NEAL_READS, "sweeps": PORTFOLIO_NEAL_SWEEPS, "seed": neal_seed,
        "deadline_s": PORTFOLIO_DEADLINE_S,
        "elapsed_s": elapsed_s,
        "exit_ok": exit_ok,
        "error": error,
        "status": deadline_status(elapsed_s, PORTFOLIO_DEADLINE_S, exit_ok),
    }
    if response is not None:
        column = {var: idx for idx, var in enumerate(response.variables)}
        order = [column[i] for i in range(qubo.n)]
        bits = np.asarray(response.record.sample, dtype=np.float64)[:, order]
        spins = bits * 2.0 - 1.0
        scored = pr.score_reads(problem, qubo, spins)
        record["objective"] = float(scored["objective"])
        record["feasible"] = bool(scored["feasible"])
        record["raw_feasible_count"] = scored["raw_feasible_count"]
        record["returned_reads"] = int(scored["returned_reads"])
    return record


# ---------------------------------------------------- seeded/cold weighted MSA (step 9)


#: Cells whose Round 1 saved captures score in whole milli units (``round2_export.py``'s
#: ``MILLI_CELLS``): the bundle's canonical (h, edges, j) is in energy units (milli / 1000),
#: so a rescore against the saved captures must scale back up by 1000 before comparing.
MILLI_CELLS = frozenset({"native-pm1", "native-125", "diamond-pm1", "cubic-dimer-pm1"})


def qpu_seed_lanes(
    cell: str, nonce: str, h: np.ndarray, edges: np.ndarray, j: np.ndarray,
    round1_root: PathLike, anneal_dir: str = "qpu-80", lanes: int = SEED_LANES,
) -> Dict[str, Any]:
    """The ``lanes`` lowest-energy Round 1 QPU reads for this model, validated against the bundle.

    Raises :class:`RunnerError` if the saved reads do not rescore against the
    bundle's own ``(h, edges, j)``: a mismatch here means the saved spins are
    in a different variable order than the bundle expects (or belong to a
    different model entirely), and seeding from them would silently corrupt
    the run rather than merely underperform. Duplicate lanes are reported,
    never silently collapsed (task brief, step 9).
    """
    path = Path(round1_root) / cell / anneal_dir / f"{nonce}.npz"
    with np.load(path) as cap:
        spins = np.asarray(cap["spins"])
        saved_energies = np.asarray(cap["energies"], dtype=np.float64)
    rescored = regimes.energy(spins, h, edges, j)
    # round2_export.py's own rescore_check applies this same x1000 scale for milli cells
    # before comparing a canonical (energy-unit) rescoring against Round 1's saved
    # integer-milli captures; this mirrors it rather than re-deriving it.
    comparable = rescored * 1000.0 if cell in MILLI_CELLS else rescored
    if not np.allclose(comparable, saved_energies, rtol=0.0, atol=1e-6):
        raise RunnerError(
            f"{cell}/{nonce}: round1 {anneal_dir} spins do not rescore against the bundle (max "
            f"abs error {float(np.max(np.abs(comparable - saved_energies)))}); refusing to use them "
            "as seed lanes with an unvalidated variable order"
        )
    order = np.argsort(rescored, kind="stable")[:lanes]
    lane_spins = spins[order]
    unique_lanes = len(np.unique(lane_spins, axis=0))
    return {
        "spins": lane_spins, "energies": rescored[order], "lanes": lanes,
        "unique_lanes": unique_lanes, "duplicate_lanes": lanes - unique_lanes,
    }


def cpu_lite_seed_lanes(
    h: np.ndarray, edges: np.ndarray, j: np.ndarray, seed: int,
    lite_sweeps: int = CPU_LITE_SWEEPS, lanes: int = SEED_LANES,
) -> Dict[str, Any]:
    """The ``lanes`` lowest-energy reads of a shallow (``lite_sweeps``), cold ``cpu-msa-f64`` run.

    The CPU-native "second, separate control" for the seeded/cold comparison
    (task brief, step 9) -- Round 1's own MSA-lite convention
    (``scripts/seeded_sweep.py``), reused here rather than reinvented.
    """
    quip_msa = _msa()
    sampler = quip_msa.Msa()
    beta_range = quip_msa.default_beta_range(h, edges, j)
    spins, energies, meta = sampler.sample_research(
        h, edges, j, kernel="cpu-msa-f64", num_sweeps=lite_sweeps, num_reads=64, seed=seed, beta_range=beta_range,
    )
    order = np.argsort(energies, kind="stable")[:lanes]
    lane_spins = spins[order]
    unique_lanes = len(np.unique(lane_spins, axis=0))
    return {
        "spins": lane_spins, "energies": energies[order], "lanes": lanes,
        "unique_lanes": unique_lanes, "duplicate_lanes": lanes - unique_lanes,
        "observed_kernel": meta["observed_kernel"],
    }


def execute_seeded_sweep_job(
    cell: str, nonce: str, bundles_root: PathLike, round1_root: PathLike, seed_source: str,
    sweeps: int = SEEDED_SWEEPS,
) -> Tuple[Dict[str, Any], Optional[Dict[str, np.ndarray]]]:
    """One seeded/cold comparison at ``sweeps``: :data:`SEED_LANES` lanes from ``seed_source``
    ("qpu" or "cpu-lite"), plus :data:`COLD_LANES` lanes left cold, on ``cpu-msa-f64``.

    ``initial_spins`` carries only the seed lanes: per ``Msa.sample_research``'s
    own contract ("row r seeds read r, later reads start cold"), the reads
    past the seed rows start cold without any extra bookkeeping here.
    """
    if seed_source not in SEED_SOURCES:
        raise ValueError(f"seed_source must be one of {SEED_SOURCES}, got {seed_source!r}")

    manifest, arrays = round2_io.read_bundle(Path(bundles_root) / cell / nonce)
    h, edges, j = arrays["h"], arrays["edges"], arrays["j"]
    reads = SEEDED_SWEEP_READS

    if seed_source == "qpu":
        lite_seed = None
        lanes = qpu_seed_lanes(cell, nonce, h, edges, j, round1_root)
    else:
        lite_seed, _ = seed_for(manifest["hash"], "cpu-msa-f64", CPU_LITE_SWEEPS, 64, "cpu-lite-source")
        lanes = cpu_lite_seed_lanes(h, edges, j, lite_seed)

    seed, seed_input_hash = seed_for(manifest["hash"], "cpu-msa-f64", sweeps, reads, f"{seed_source}-seeded")

    quip_msa = _msa()
    sampler = quip_msa.Msa()
    beta_range = quip_msa.default_beta_range(h, edges, j)
    record: Dict[str, Any] = {
        "schema": "round2-seeded-sweep-v1",
        "cell": cell, "nonce": nonce, "model_hash": manifest["hash"], "seed_source": seed_source,
        "sweeps": sweeps, "reads": reads, "seed_lanes": SEED_LANES, "cold_lanes": COLD_LANES,
        "seed": seed, "seed_input_hash": seed_input_hash, "lite_source_seed": lite_seed,
        "unique_seed_lanes": lanes["unique_lanes"], "duplicate_seed_lanes": lanes["duplicate_lanes"],
        "beta_range": [float(beta_range[0]), float(beta_range[1])],
    }

    try:
        spins, energies, meta = sampler.sample_research(
            h, edges, j, kernel="cpu-msa-f64", num_sweeps=sweeps, num_reads=reads,
            seed=seed, beta_range=beta_range, initial_spins=lanes["spins"],
        )
    except ValueError as exc:
        record.update(unsupported=True, unsupported_reason=str(exc), exit_ok=True, error=None)
        return record, None

    observed_kernel = meta["observed_kernel"]
    if observed_kernel != "cpu-msa-f64":
        raise RunnerError(f"{cell}/{nonce}: seeded sweep observed {observed_kernel!r}, not cpu-msa-f64")
    rescored = regimes.energy(spins, h, edges, j)
    if not np.isfinite(rescored).all():
        raise RunnerError(f"{cell}/{nonce}: seeded sweep produced a nonfinite score")

    seeded_energies = rescored[:SEED_LANES]
    cold_energies = rescored[SEED_LANES:]
    record.update(
        unsupported=False, unsupported_reason=None, exit_ok=True, error=None,
        observed_kernel=observed_kernel, representation=meta["representation"],
        best_seeded_energy=float(seeded_energies.min()), best_cold_energy=float(cold_energies.min()),
        unique_reads=int(len(np.unique(spins, axis=0))),
    )
    return record, {"spins": spins, "energies": rescored}
