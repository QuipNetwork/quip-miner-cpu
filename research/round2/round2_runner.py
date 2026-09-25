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
import math
import os
import resource
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from quip_miner_dwave import regimes, round2_io

PathLike = os.PathLike

# ---------------------------------------------------------------- constants

#: The five sweep depths of the complete CPU comparison (controller ruling 1).
SWEEP_DEPTHS: Tuple[int, ...] = (512, 2048, 8192, 32768, 131072)

#: The pilot's sweep depths -- the first two rungs of the full ladder (controller ruling 1).
PILOT_SWEEP_DEPTHS: Tuple[int, ...] = (512, 2048)

#: Kernels of the controlled comparison. ``cpu-msa`` (auto-select) never appears here.
#: Final review ruling: ``cpu-msa`` auto-routes in Round 2 (unit-eligible -> the unit
#: kernel, otherwise the float kernel). Round 1's own records use the SAME bare name
#: to mean the unit kernel specifically. A consumer that filters `requested_kernel ==
#: "cpu-msa"` across the two rounds will silently mix a Round-1 unit-only arm with a
#: Round-2 auto-routed one -- never pool the two rounds by that bare kernel name.
CONTROLLED_KERNELS: Tuple[str, ...] = ("cpu-sa", "cpu-msa-f64", "cpu-msa-unit")

#: The reference simulated annealing kernel used only by the captured-model set.
NEAL_KERNEL = "dwave-neal"
CAPTURED_KERNELS: Tuple[str, ...] = (*CONTROLLED_KERNELS, NEAL_KERNEL)

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
#: The clean serial timing subset's own repetition kind (change: parallel workers,
#: requirement 3) -- distinct from REPETITION_TIMING so its output never collides
#: with, or is mistaken for, a quality-campaign record of the same arm.
REPETITION_TIMING_SUBSET = "timing-subset"

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


class KernelIneligible(Exception):
    """A model fails an explicit, pre-checked eligibility rule for the requested kernel.

    Distinct from :class:`RunnerError`: this is never a defect, only an anticipated
    "this model cannot use this kernel" outcome, decided BEFORE the kernel is ever
    called (review finding 8: "record 'unsupported' only for the explicit,
    pre-checked eligibility reasons... any other exception is a 'failed' record").
    """


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


def _parse_sibling_list(text: str, cpu: int) -> List[int]:
    """Every OTHER logical CPU named in one sysfs ``..._siblings_list`` value.

    Handles both the comma form (``"4,20"``, this host's shape) and the range
    form (``"4-7"``, and mixtures of the two, ``"0-1,16-17"``) -- sysfs uses
    range notation whenever three or more consecutive IDs share a group, which
    a plain ``split(",")`` misparses as one huge integer (review, minor 4).
    """
    ids: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return [i for i in ids if i != cpu]


def cpu_siblings(cpu: int) -> List[int]:
    """Every OTHER logical CPU on ``cpu``'s physical core (its full thread-sibling
    set, not just the first one -- review, minor 4: more than two threads per
    core needs the full set to group physical cores correctly). Empty with no
    SMT sibling.
    """
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    if not path.exists():
        return []
    return _parse_sibling_list(path.read_text(encoding="utf-8").strip(), cpu)


def cpu_sibling(cpu: int) -> Optional[int]:
    """The other logical CPU on ``cpu``'s physical core (the first, if there is more
    than one), or None with no SMT sibling. See :func:`cpu_siblings` for the full
    set -- this single-value form stays the contamination sampler's own
    interface, which only ever measures one representative sibling thread.
    """
    siblings = cpu_siblings(cpu)
    return siblings[0] if siblings else None


def select_worker_cpus(
    n: int, *, cpus: Optional[Sequence[int]] = None, allowed: Optional[Sequence[int]] = None,
    siblings_of: Callable[[int], Sequence[int]] = cpu_siblings, avoid_cpu0: bool = True,
) -> List[int]:
    """``n`` logical CPUs, one per physical core, with no SMT sibling among them.

    If ``cpus`` is given explicitly, validates every entry is in the allowed
    affinity set (review, minor 6: a bad value would otherwise only fail once
    a job's own ``sched_setaffinity`` call does, one failure record at a
    time), rejects an empty list (review, minor 5), and validates it names no
    two logical CPUs on the same physical core (task brief: "never pin two
    workers to the same physical core"). Its length then decides the worker
    count, and it is returned as given -- including physical core 0 if named
    explicitly; ``avoid_cpu0`` only affects the AUTOMATIC choice below.

    Otherwise chooses ``n`` cores automatically from ``allowed`` (this
    process's own allowed affinity set, ``os.sched_getaffinity(0)``, by
    default), taking cores in ascending order and skipping any logical CPU
    whose SMT sibling(s) were already chosen. With ``avoid_cpu0`` true (the
    default), physical core 0 is never chosen automatically: it is where
    interrupt handling lands, and may already be pinned to other work the
    contamination check cannot see (review, must-fix 1) -- pass
    ``avoid_cpu0=False``, or name core 0 via ``--cpus``, to use it anyway.
    Raises if fewer than ``n`` distinct (eligible) physical cores are
    available. ``siblings_of`` is :func:`cpu_siblings` by default; injectable
    for testing against a synthetic topology.
    """
    if n < 1:
        raise ValueError(f"need at least 1 worker, got {n}")

    def _physical(cpu: int) -> "frozenset[int]":
        return frozenset({cpu, *siblings_of(cpu)})

    if cpus is not None:
        cpus = list(cpus)
        if not cpus:
            raise ValueError("--cpus must not be empty")
        allowed_set = set(allowed) if allowed is not None else set(os.sched_getaffinity(0))
        outside = [cpu for cpu in cpus if cpu not in allowed_set]
        if outside:
            raise ValueError(f"--cpus names {outside}, which is not in the allowed affinity set {sorted(allowed_set)}")
        seen: set = set()
        for cpu in cpus:
            physical = _physical(cpu)
            if physical in seen:
                raise ValueError(f"--cpus names two logical CPUs on the same physical core: {cpus}")
            seen.add(physical)
        return cpus

    pool = sorted(allowed) if allowed is not None else sorted(os.sched_getaffinity(0))
    excluded = _physical(0) if avoid_cpu0 else frozenset()
    chosen: List[int] = []
    used: set = set()
    for cpu in pool:
        physical = _physical(cpu)
        if physical in used or physical == excluded:
            continue
        chosen.append(cpu)
        used.add(physical)
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise ValueError(
            f"only {len(chosen)} distinct physical core(s) available in the allowed affinity set "
            f"{pool}{' (core 0 excluded by default)' if avoid_cpu0 else ''}; need {n}"
        )
    return chosen


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
    loadavg_1m_raw_before: float
    loadavg_1m_raw_after: float
    loadavg_1m_adjusted_before: float
    loadavg_1m_adjusted_after: float
    concurrent_workers: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contaminated": self.contaminated,
            "reasons": list(self.reasons),
            "sibling_busy_fraction": self.sibling_busy_fraction,
            "thresholds": self.thresholds,
            "loadavg_1m_raw_before": self.loadavg_1m_raw_before,
            "loadavg_1m_raw_after": self.loadavg_1m_raw_after,
            "loadavg_1m_adjusted_before": self.loadavg_1m_adjusted_before,
            "loadavg_1m_adjusted_after": self.loadavg_1m_adjusted_after,
            "concurrent_workers": self.concurrent_workers,
        }


def check_contamination(before: HostSample, after: HostSample, *, concurrent_workers: int) -> ContaminationVerdict:
    """Whether a run between ``before`` and ``after`` is contaminated, and why.

    Never drops a contaminated run: this only labels it, so the caller can
    keep the record and repeat the run under the same manifest (task brief,
    step 4).

    ``concurrent_workers`` is subtracted from the raw loadavg before comparing
    to the threshold (controller ruling, workers-change fix round 1): the
    run's OWN workers are not competing load, and without this a parallel run
    of ``N`` workers would see its raw loadavg rise by roughly ``N`` from its
    own presence alone, mislabeling most of a large run as contaminated. This
    is ONE formula for every ``timing_mode``, required (not defaulted) so no
    caller can forget to pass it: a serial run (``concurrent_workers=1``) now
    also subtracts 1, a small, deliberate departure from the exact-raw
    comparison used before this change -- not an attempt to reproduce that
    exact historical number, but the same rule applied uniformly. Both the
    raw and the adjusted loadavg are recorded, alongside ``concurrent_workers``
    and the threshold, so a record never hides which one drove the label.
    """
    adjusted_before = before.loadavg_1m - concurrent_workers
    adjusted_after = after.loadavg_1m - concurrent_workers
    reasons: List[str] = []
    if adjusted_before > LOADAVG_1M_CONTAMINATION_THRESHOLD:
        reasons.append(
            f"adjusted loadavg_1m before ({adjusted_before:.3f}, raw {before.loadavg_1m} minus "
            f"{concurrent_workers} worker(s)) exceeds {LOADAVG_1M_CONTAMINATION_THRESHOLD}"
        )
    if adjusted_after > LOADAVG_1M_CONTAMINATION_THRESHOLD:
        reasons.append(
            f"adjusted loadavg_1m after ({adjusted_after:.3f}, raw {after.loadavg_1m} minus "
            f"{concurrent_workers} worker(s)) exceeds {LOADAVG_1M_CONTAMINATION_THRESHOLD}"
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
        loadavg_1m_raw_before=before.loadavg_1m,
        loadavg_1m_raw_after=after.loadavg_1m,
        loadavg_1m_adjusted_before=adjusted_before,
        loadavg_1m_adjusted_after=adjusted_after,
        concurrent_workers=concurrent_workers,
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

    def run_key(self, bundles_root: PathLike, solver_identity: Mapping[str, Any]) -> str:
        """The resume identity of this job: change any field here, OR the solver's own
        build identity, and it is a different run (review finding 5: if ``quip_msa`` is
        rebuilt partway through a campaign, resume must not silently mix results from
        two different builds).
        """
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
                "solver_identity": dict(solver_identity),
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


def build_timing_subset_jobs(index: Dict[str, Any], cells: Sequence[str]) -> List[CpuJob]:
    """A clean serial timing subset: one model per cell (the first by sorted nonce),
    every sweep depth, every controlled kernel (change: parallel workers,
    requirement 3).

    Reuses the campaign's own seed for each arm (``variant="timing"``, the same as
    :func:`build_campaign_jobs`): this measures the exact same arm the campaign
    does, under controlled serial/pinned conditions, not a different one.
    ``repetition_kind`` is :data:`REPETITION_TIMING_SUBSET`, distinct from the
    campaign's own :data:`REPETITION_TIMING`, so the two never collide on the same
    output path even before the orchestrator's separate output directory is
    considered. Not shuffled: the timing subset is small enough, and run serially
    enough, that job order carries no confound worth randomizing away.
    """
    jobs: List[CpuJob] = []
    for cell in cells:
        rows = _sorted_bundle_rows(index, cell)[:1]
        for row in rows:
            for sweeps in SWEEP_DEPTHS:
                for kernel in CONTROLLED_KERNELS:
                    variant = "timing"
                    seed, seed_input_hash = seed_for(row["model_hash"], kernel, sweeps, CAMPAIGN_READS, variant)
                    jobs.append(
                        CpuJob(
                            cell=cell, nonce=row["nonce"], kernel=kernel, sweeps=sweeps,
                            reads=CAMPAIGN_READS, repetition_id=0, repetition_kind=REPETITION_TIMING_SUBSET,
                            variant=variant, seed=seed, seed_input_hash=seed_input_hash,
                        )
                    )
    return jobs


def build_captured_jobs(index: Dict[str, Any], capture_manifest: Dict[str, Any]) -> List[CpuJob]:
    """Build the four-arm comparison for each model with a physical-pilot capture."""
    indexed_rows = {(row["cell"], row["nonce"]): row for row in index["rows"]}
    captured_models: Dict[Tuple[str, str], str] = {}
    for capture in capture_manifest["jobs"]:
        key = (capture["cell"], capture["nonce"])
        model_hash = capture["model_hash"]
        prior_hash = captured_models.setdefault(key, model_hash)
        if prior_hash != model_hash:
            raise RunnerError(f"{key[0]}/{key[1]} has conflicting captured model hashes")

    jobs: List[CpuJob] = []
    for (cell, nonce), model_hash in captured_models.items():
        row = indexed_rows.get((cell, nonce))
        if row is None:
            raise RunnerError(f"{cell}/{nonce} from the capture manifest is missing from the bundle index")
        if row["model_hash"] != model_hash:
            raise RunnerError(f"{cell}/{nonce} capture model hash disagrees with the bundle index")
        for sweeps in SWEEP_DEPTHS:
            for kernel in CAPTURED_KERNELS:
                variant = "timing"
                seed, seed_input_hash = seed_for(model_hash, kernel, sweeps, CAMPAIGN_READS, variant)
                jobs.append(
                    CpuJob(
                        cell=cell, nonce=nonce, kernel=kernel, sweeps=sweeps,
                        reads=CAMPAIGN_READS, repetition_id=0, repetition_kind=REPETITION_TIMING,
                        variant=variant, seed=seed, seed_input_hash=seed_input_hash,
                    )
                )
    return jobs


# -------------------------------------------------------------- job execution


def _msa():
    import quip_msa  # pyright: ignore[reportMissingModuleSource]

    return quip_msa


def solver_identity() -> Dict[str, Any]:
    """``quip_msa``'s package version and a content hash of its loaded compiled extension.

    Recorded in every run record and folded into the run key (review finding 5), so a
    resume never silently mixes results from two different ``quip_msa`` builds, and
    every record on disk names exactly which build produced it. Degrades to an
    ``"error"``-carrying shape rather than raising: identity-gathering must never crash
    an otherwise-successful, or otherwise-failing, run.
    """
    import hashlib
    import importlib.metadata

    try:
        quip_msa = _msa()
        version = importlib.metadata.version("quip_msa")
        module_file = Path(quip_msa.__file__).resolve()
        binaries = sorted(module_file.parent.glob("*.so"))
        binary_file: Optional[str] = None
        binary_sha256: Optional[str] = None
        if len(binaries) == 1:
            binary_file = str(binaries[0])
            digest = hashlib.sha256()
            with open(binaries[0], "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            binary_sha256 = digest.hexdigest()
        return {
            "package": "quip_msa", "version": version, "module_file": str(module_file),
            "binary_file": binary_file, "binary_sha256": binary_sha256, "error": None,
        }
    except Exception as exc:
        return {
            "package": "quip_msa", "version": None, "module_file": None,
            "binary_file": None, "binary_sha256": None, "error": f"{type(exc).__name__}: {exc}",
        }


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
        raise KernelIneligible(
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


def unit_kernel_ineligibility_reason(h: np.ndarray, j: np.ndarray) -> Optional[str]:
    """Why ``cpu-msa-unit`` cannot take this canonical model, checked BEFORE the kernel
    is ever called -- so an eligibility gap is recorded as "unsupported", and any OTHER
    ``ValueError`` the kernel itself raises is a real "failed" record, never mistaken
    for a known ineligibility (review finding 8).

    Mirrors the kernel's own stated bound (its error message): couplings in
    ``{-1, 0, +1}``, and whole-number fields. Returns None when the model looks
    eligible by this check; the kernel call afterward is the final authority (a
    cubic-dimer-pm1 model instead goes through :func:`kernel_input_for`'s own
    pre-check, which covers its repeated-unit reconstruction).
    """
    j = np.asarray(j, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    rounded_j = np.round(j)
    if not (np.isin(rounded_j, (-1.0, 0.0, 1.0)).all() and np.allclose(j, rounded_j, atol=1e-9)):
        return "the unit kernel takes couplings in {-1, 0, +1}; this model has another value"
    if not np.allclose(h, np.round(h), atol=1e-9):
        return "the unit kernel takes whole-number fields; this model has a fractional value"
    return None


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
    identity = solver_identity()
    t_start = time.perf_counter()

    manifest, arrays = round2_io.read_bundle(Path(bundles_root) / job.cell / job.nonce)
    h, edges, j = arrays["h"], arrays["edges"], arrays["j"]

    t_setup_start = time.perf_counter()
    sampler: Any
    if job.kernel == NEAL_KERNEL:
        from dwave.samplers import SimulatedAnnealingSampler

        sampler = SimulatedAnnealingSampler()
        beta_range = None
    else:
        quip_msa = _msa()
        sampler = quip_msa.Msa()
        beta_range = quip_msa.default_beta_range(h, edges, j)
    effective_seed = job.seed % (1 << 31) if job.kernel == NEAL_KERNEL else None
    t_setup_end = time.perf_counter()

    record: Dict[str, Any] = {
        "schema": "round2-cpu-run-v1",
        "run_key": job.run_key(bundles_root, identity),
        "cell": job.cell,
        "nonce": job.nonce,
        "model_hash": manifest["hash"],
        "requested_kernel": job.kernel,
        "reads": job.reads,
        "sweeps": job.sweeps,
        "beta_range": None if beta_range is None else [float(beta_range[0]), float(beta_range[1])],
        "seed": job.seed,
        "effective_seed": effective_seed,
        "seed_input_hash": job.seed_input_hash,
        "repetition_id": job.repetition_id,
        "repetition_kind": job.repetition_kind,
        "variant": job.variant,
        "setup_s": t_setup_end - t_setup_start,
        "graph_setup_s": None,  # not exposed by quip_msa's API; see module docstring
        "solver_identity": identity,
    }

    def _unsupported(reason: str) -> Tuple[Dict[str, Any], None]:
        record.update(
            unsupported=True, unsupported_reason=reason, exit_ok=True, error=None,
            observed_kernel=None, representation=None, workspace_bytes=None, rng_scheme=None,
            elapsed_sampling_s=None, best_energy=None, mean_energy=None, unique_reads=None,
            submitted_beta_range=None, kernel_energy_scale=None, kernel_input_hash=None,
        )
        record["wall_s"] = time.perf_counter() - t_start
        record["peak_rss_kb"] = _peak_rss_kb()
        return record, None

    def _failed(reason: str) -> Tuple[Dict[str, Any], None]:
        record.update(
            unsupported=False, unsupported_reason=None, exit_ok=False, error=reason,
            observed_kernel=None, representation=None, workspace_bytes=None, rng_scheme=None,
            elapsed_sampling_s=None, best_energy=None, mean_energy=None, unique_reads=None,
            submitted_beta_range=None, kernel_energy_scale=None, kernel_input_hash=None,
        )
        record["wall_s"] = time.perf_counter() - t_start
        record["peak_rss_kb"] = _peak_rss_kb()
        return record, None

    # Eligibility for cpu-msa-unit is decided BEFORE the kernel is ever called, by an
    # explicit, pre-checked rule -- never by reacting to whatever ValueError the kernel
    # itself happens to raise (review finding 8). cubic-dimer-pm1's own pre-check lives
    # inside kernel_input_for (its repeated-unit reconstruction); every other cell is
    # checked here directly against the kernel's stated bound.
    if job.kernel == UNIT_KERNEL and job.cell != "cubic-dimer-pm1":
        reason = unit_kernel_ineligibility_reason(h, j)
        if reason is not None:
            return _unsupported(reason)

    if job.kernel == NEAL_KERNEL:
        kernel_input = {
            "h": h, "edges": edges, "j": j, "beta_range": None,
            "energy_scale": 1.0, "input_hash": _array_input_hash(h, edges, j),
        }
    else:
        assert beta_range is not None
        try:
            kernel_input = kernel_input_for(job.cell, job.kernel, h, edges, j, beta_range)
        except KernelIneligible as exc:
            return _unsupported(str(exc))

    bundle_unit_hash = manifest.get("unit_kernel_input_hash")
    if job.kernel == UNIT_KERNEL and bundle_unit_hash is not None and bundle_unit_hash != kernel_input["input_hash"]:
        raise RunnerError(
            f"{job.cell}/{job.nonce}: the reconstructed repeated-unit input hashes to "
            f"{kernel_input['input_hash']!r}, and the bundle records {bundle_unit_hash!r}; refusing "
            "to submit an input that does not match round2_export.py's own reconstruction"
        )
    record["submitted_beta_range"] = (
        None if kernel_input["beta_range"] is None else
        [float(kernel_input["beta_range"][0]), float(kernel_input["beta_range"][1])]
    )
    record["kernel_energy_scale"] = kernel_input["energy_scale"]
    record["kernel_input_hash"] = kernel_input["input_hash"]

    try:
        if job.kernel == NEAL_KERNEL:
            assert effective_seed is not None
            quadratic: Dict[Tuple[Any, Any], float] = {}
            for edge, coupling in zip(edges, j):
                pair = (int(edge[0]), int(edge[1]))
                quadratic[pair] = quadratic.get(pair, 0.0) + float(coupling)
            ising_h = {i: float(bias) for i, bias in enumerate(h)}

            t_sample_start = time.perf_counter()
            response = sampler.sample_ising(
                ising_h, quadratic, num_sweeps=job.sweeps, num_reads=job.reads, seed=effective_seed,
            )
            t_sample_end = time.perf_counter()
            column = {variable: i for i, variable in enumerate(response.variables)}
            spins = np.asarray(response.record.sample, dtype=np.int8)[:, [column[i] for i in range(len(h))]]
            energies = np.asarray(response.record.energy, dtype=np.float64)
            meta = {
                "observed_kernel": NEAL_KERNEL, "representation": "spin",
                "workspace_bytes": None, "rng_scheme": None,
            }
            beta_range = response.info["beta_range"]
            record["beta_range"] = [float(beta_range[0]), float(beta_range[1])]
        else:
            t_sample_start = time.perf_counter()
            spins, energies, meta = sampler.sample_research(
                kernel_input["h"], kernel_input["edges"], kernel_input["j"], kernel=job.kernel,
                num_sweeps=job.sweeps, num_reads=job.reads, seed=job.seed,
                beta_range=kernel_input["beta_range"],
            )
            t_sample_end = time.perf_counter()
    except ValueError as exc:
        # Past the pre-check, ANY ValueError here is an unanticipated defect -- a
        # "failed" record, never "unsupported" (review finding 8).
        return _failed(f"{type(exc).__name__}: {exc}")

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
        workspace_bytes=meta["workspace_bytes"], rng_scheme=meta.get("rng_scheme"),
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
    "workspace_bytes", "rng_scheme", "elapsed_sampling_s", "best_energy", "mean_energy", "unique_reads",
    "setup_s", "graph_setup_s", "beta_range", "submitted_beta_range", "kernel_energy_scale",
    "kernel_input_hash", "wall_s", "peak_rss_kb", "effective_seed",
)


def failure_record(
    job: CpuJob, bundles_root: PathLike, exc: BaseException,
    *, identity: Optional[Mapping[str, Any]] = None, model_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """A well-formed, fully-shaped run record for a job that raised before it could produce one.

    Used for anything :func:`execute_cpu_job` did not itself catch: a
    :class:`RunnerError` invariant violation, a bundle read failure, or any
    other unexpected exception. Never drops the job silently -- the record
    always exists, with every field :func:`execute_cpu_job` would have set,
    so downstream analysis never has to special-case a missing key.

    ``identity`` (the solver's build identity, see :func:`solver_identity`) defaults to
    a freshly computed one when not given (the caller may already have one and want to
    reuse it, e.g. across many failures in a loop). ``model_hash`` is carried through
    when the caller already knows it (the bundle loaded fine and something else failed
    afterward); otherwise it stays None.
    """
    identity = dict(identity) if identity is not None else solver_identity()
    record: Dict[str, Any] = {
        "schema": "round2-cpu-run-v1",
        "run_key": job.run_key(bundles_root, identity),
        "cell": job.cell,
        "nonce": job.nonce,
        "model_hash": model_hash,
        "requested_kernel": job.kernel,
        "reads": job.reads,
        "sweeps": job.sweeps,
        "seed": job.seed,
        "seed_input_hash": job.seed_input_hash,
        "repetition_id": job.repetition_id,
        "repetition_kind": job.repetition_kind,
        "variant": job.variant,
        "solver_identity": identity,
    }
    for field in _RECORD_FIELDS:
        record[field] = None
    record["effective_seed"] = job.seed % (1 << 31) if job.kernel == NEAL_KERNEL else None
    record["unsupported"] = False
    record["exit_ok"] = False
    record["error"] = f"{type(exc).__name__}: {exc}"
    return record


# ------------------------------------------------------- subprocess orchestration

#: A generous floor under :func:`estimate_hard_deadline_s`, and the fallback for
#: callers that have no per-job size estimate at all. Crash protection only (task
#: brief, step 5: "a subprocess timeout and cleanup") -- NOT the application/
#: comparison deadline a caller classifies with :func:`deadline_status` -- see
#: :func:`run_one_subprocess`.
DEFAULT_HARD_DEADLINE_S = 600.0

#: Calibrated from the five-model pilot's own slowest observed run (native-125,
#: cpu-sa, 2,048 sweeps, 64 reads, n=4,575 spins, under heavy host contention:
#: 43.57 s). ``HARD_DEADLINE_SAFETY_FACTOR`` on top of that gives generous headroom,
#: so :func:`estimate_hard_deadline_s` only ever kills a genuinely hung process, never
#: a slow-but-progressing one -- even at the deepest campaign rung on the largest cell.
COST_PER_SWEEP_SPIN_READ_S = 43.57 / (2048 * 4575 * 64)
HARD_DEADLINE_FLOOR_S = 120.0
HARD_DEADLINE_SAFETY_FACTOR = 5.0


def estimate_hard_deadline_s(sweeps: int, n_spins: int, reads: int) -> float:
    """A generous, sweep/size/read-scaled hard-kill deadline for one measured run.

    Not a timing estimate -- a crash-protection ceiling (task brief, step 1's "the
    hard deadline kills the deepest rungs" defect this fixes): a single fixed value
    across every sweep depth and model size either kills the deepest, largest-model
    rungs or is uselessly loose on the shallowest, smallest ones. A CLI
    ``--hard-deadline-s`` overrides this outright when given.
    """
    return max(
        HARD_DEADLINE_FLOOR_S,
        HARD_DEADLINE_SAFETY_FACTOR * COST_PER_SWEEP_SPIN_READ_S * sweeps * n_spins * reads,
    )


class Cancelled(RuntimeError):
    """The parent process received SIGTERM or SIGINT (e.g. taskd cancelling a run)
    while waiting on a measured subprocess. The child's whole process group is killed
    before this is raised; the caller must let it propagate, writing no record for
    the interrupted job (review finding 9) -- a cancelled run is not a "completed" one.
    """


def _kill_process_group(proc: "subprocess.Popen[bytes]") -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _spawn(cmd: Sequence[str], env: Optional[Mapping[str, str]]) -> "subprocess.Popen[bytes]":
    return subprocess.Popen(cmd, start_new_session=True, env=dict(env) if env is not None else None)


def _wait_with_timeout(proc: "subprocess.Popen[bytes]", hard_deadline_s: float) -> Tuple[bool, float]:
    """Wait for ``proc``, killing its whole process group if ``hard_deadline_s`` elapses first."""
    start = time.perf_counter()
    try:
        returncode = proc.wait(timeout=hard_deadline_s)
        return returncode == 0, time.perf_counter() - start
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        proc.wait()
        return False, time.perf_counter() - start


def run_subprocess_with_hard_deadline(
    cmd: Sequence[str], hard_deadline_s: float,
    *, env: Optional[Mapping[str, str]] = None, on_spawn: Optional[Any] = None,
) -> Tuple[bool, float]:
    """Run ``cmd`` to completion, or kill its whole process group after ``hard_deadline_s``.

    Crash protection only, shared by the CPU-job and portfolio-deadline-arm
    subprocess wrappers (task brief, step 5, and step 6's "a real subprocess
    deadline, with timeout and cleanup, like the CPU arms"). Returns
    ``(exit_ok, wall_s)`` -- ``exit_ok`` is only ever the process's own exit
    code, never an application-level judgment about the answer it produced.

    Also handles the parent itself being cancelled (SIGTERM/SIGINT, exactly what
    taskd sends on cancel, review finding 9): installs handlers for the duration of
    the wait that kill the child's process group and raise :class:`Cancelled`,
    which the caller must not catch -- letting it propagate means no record is ever
    written for a job the parent was told to abandon mid-run.

    Serial use only: ``signal.signal`` only works on the main thread, so this must
    never be called from a worker thread of a parallel run -- see
    :func:`run_subprocess_tracked` for that case.

    ``env``, if given, replaces the child's environment outright (the caller is
    responsible for including anything the child needs, e.g. this parent's own
    ``PYTHONPATH`` when the child runs under a different interpreter, such as P's
    pinned venv, that would not otherwise see it). ``on_spawn``, if given, is
    called once with the child's pid right after it starts -- a testing hook,
    never used by production callers.
    """
    proc = _spawn(cmd, env)
    if on_spawn is not None:
        on_spawn(proc.pid)

    def _cancel(signum: int, _frame: Any) -> None:
        raise Cancelled(f"the parent received signal {signum} while waiting on pid {proc.pid}")

    previous_term = signal.signal(signal.SIGTERM, _cancel)
    previous_int = signal.signal(signal.SIGINT, _cancel)
    try:
        try:
            return _wait_with_timeout(proc, hard_deadline_s)
        except Cancelled:
            _kill_process_group(proc)
            proc.wait()
            raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


class ActiveProcesses:
    """A thread-safe registry of currently-running child processes.

    Exists because ``signal.signal()`` only works on the main thread: a parallel
    run cannot have each worker thread install its own SIGTERM/SIGINT handler the
    way :func:`run_subprocess_with_hard_deadline` does for a single-threaded
    caller. Instead the orchestrator installs ONE handler, in the main thread,
    that calls :meth:`kill_all` here to reach every worker's child at once.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._procs: "set[subprocess.Popen[bytes]]" = set()

    def add(self, proc: "subprocess.Popen[bytes]") -> None:
        with self._lock:
            self._procs.add(proc)

    def discard(self, proc: "subprocess.Popen[bytes]") -> None:
        with self._lock:
            self._procs.discard(proc)

    def is_empty(self) -> bool:
        with self._lock:
            return not self._procs

    def kill_all(self) -> None:
        with self._lock:
            procs = list(self._procs)
        for proc in procs:
            # A process with a known returncode has already been reaped by wait();
            # its pid could since have been recycled by the OS. Never signal it
            # (review, minor 5): the whole point of a pid-based kill is that the
            # pid still names the process we think it does.
            if proc.returncode is None:
                _kill_process_group(proc)


def run_subprocess_tracked(
    cmd: Sequence[str], hard_deadline_s: float, *, registry: ActiveProcesses, cancelled_event: threading.Event,
    env: Optional[Mapping[str, str]] = None, on_spawn: Optional[Any] = None,
) -> Tuple[bool, float]:
    """Like :func:`run_subprocess_with_hard_deadline`, but safe to call from a
    worker thread of a parallel run.

    Installs no signal handler -- instead registers the child with ``registry``
    so the orchestrator's single, main-thread handler can kill it alongside
    every other worker's child, and checks ``cancelled_event`` before spawning
    and again once the wait ends, raising :class:`Cancelled` either way. The
    caller must not catch it: a job cancelled this way gets no record, exactly
    as the serial path never records a job the parent was told to abandon.
    Re-checking immediately after registering (not just before spawning) closes
    the gap where a cancel arrives between the two: a process that was about to
    be missed by a ``kill_all()`` sweep is killed directly instead.
    """
    if cancelled_event.is_set():
        raise Cancelled("cancelled before this job's subprocess was spawned")
    proc = _spawn(cmd, env)
    if on_spawn is not None:
        on_spawn(proc.pid)
    registry.add(proc)
    if cancelled_event.is_set():
        _kill_process_group(proc)
    try:
        exit_ok, wall_s = _wait_with_timeout(proc, hard_deadline_s)
    finally:
        registry.discard(proc)
    if cancelled_event.is_set():
        raise Cancelled("cancelled while this job's subprocess was running")
    return exit_ok, wall_s


def _run_one_cmd(
    job: CpuJob, *, bundles_root: PathLike, out_dir: PathLike, cpu: Optional[int], script_path: PathLike,
    attempt: int, python_exe: Optional[str], hard_deadline_s_override: Optional[float],
    timing_mode: str, concurrent_workers: int,
) -> List[str]:
    python_exe = python_exe or sys.executable
    cmd = [
        python_exe, str(script_path), "run-one",
        "--job-json", json.dumps(job.to_dict()),
        "--bundles-root", str(bundles_root),
        "--attempt", str(attempt),
        "--out-dir", str(out_dir),
        "--timing-mode", timing_mode,
        "--concurrent-workers", str(concurrent_workers),
    ]
    if cpu is not None:
        cmd += ["--cpu", str(cpu)]
    if hard_deadline_s_override is not None:
        cmd += ["--hard-deadline-s", str(hard_deadline_s_override)]
    return cmd


def run_one_subprocess(
    job: CpuJob,
    *,
    bundles_root: PathLike,
    out_dir: PathLike,
    cpu: Optional[int],
    script_path: PathLike,
    attempt: int,
    python_exe: Optional[str] = None,
    hard_deadline_s: float = DEFAULT_HARD_DEADLINE_S,
    hard_deadline_s_override: Optional[float] = None,
    timing_mode: str = "serial",
    concurrent_workers: int = 1,
) -> Tuple[bool, float]:
    """Run one job in a fresh ``run-one`` subprocess, pinned to ``cpu``. Returns ``(exit_ok, wall_s)``.

    Serial use only (a single worker) -- see :func:`run_one_subprocess_tracked` for
    a parallel run's worker threads, which cannot each install their own signal
    handler.

    ``attempt`` is the EXACT attempt number the orchestrator already decided on (via
    its own resolve/repeat-contaminated logic) and is always passed down explicitly:
    the child must write to (or confirm as already-done) that exact attempt, never
    re-derive its own -- a self-resolving child would silently disagree with a
    parent running in ``--repeat-contaminated`` mode, which picks an attempt for a
    reason (contamination) plain resolution has no way to see. ``hard_deadline_s``
    is what THIS process enforces via :func:`run_subprocess_with_hard_deadline`;
    ``hard_deadline_s_override``, when given, is also passed to the child via
    ``--hard-deadline-s`` so both sides use the identical explicit value rather
    than each independently recomputing the size-scaled default. One-process-at-a-
    time timing (task brief, step 4) is the caller's responsibility: this starts
    exactly one subprocess and waits for it before returning. See
    :func:`run_subprocess_with_hard_deadline` for the crash-protection and
    cancellation semantics.
    """
    cmd = _run_one_cmd(
        job, bundles_root=bundles_root, out_dir=out_dir, cpu=cpu, script_path=script_path, attempt=attempt,
        python_exe=python_exe, hard_deadline_s_override=hard_deadline_s_override,
        timing_mode=timing_mode, concurrent_workers=concurrent_workers,
    )
    return run_subprocess_with_hard_deadline(cmd, hard_deadline_s)


def run_one_subprocess_tracked(
    job: CpuJob,
    *,
    bundles_root: PathLike,
    out_dir: PathLike,
    cpu: Optional[int],
    script_path: PathLike,
    attempt: int,
    registry: ActiveProcesses,
    cancelled_event: threading.Event,
    concurrent_workers: int,
    python_exe: Optional[str] = None,
    hard_deadline_s: float = DEFAULT_HARD_DEADLINE_S,
    hard_deadline_s_override: Optional[float] = None,
) -> Tuple[bool, float]:
    """Like :func:`run_one_subprocess`, but for one worker thread of a parallel run
    (``timing_mode`` is always ``"parallel"`` here -- a single-worker run uses
    :func:`run_one_subprocess` instead). See :func:`run_subprocess_tracked` for the
    cross-thread cancellation semantics.

    ``concurrent_workers`` is required (review, minor 7: a default here was an
    arbitrary number with no real meaning): every caller must say explicitly
    how many workers this run has, since it is recorded on every job's record
    and used to adjust the contamination check. It is the size of the worker
    POOL this run started with, not a live count of jobs actually in flight at
    any instant -- the tail of a run, or a resume with fewer remaining jobs
    than workers, still reports the full pool size (review, minor 8).
    """
    cmd = _run_one_cmd(
        job, bundles_root=bundles_root, out_dir=out_dir, cpu=cpu, script_path=script_path, attempt=attempt,
        python_exe=python_exe, hard_deadline_s_override=hard_deadline_s_override,
        timing_mode="parallel", concurrent_workers=concurrent_workers,
    )
    return run_subprocess_tracked(cmd, hard_deadline_s, registry=registry, cancelled_event=cancelled_event)


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
#: This arm's instance source: a synthetic, deterministically-seeded portfolio, never
#: a captured historical or production one (review finding I5: the report must never
#: call this arm "historical").
PORTFOLIO_PROVENANCE = "synthetic-test"


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


def _portfolio_deadline_provenance() -> Dict[str, Any]:
    """P's HEAD commit and the versions of the packages this arm imports.

    Degrades to ``None``s rather than raising, like :func:`solver_identity`:
    provenance-gathering must never crash an otherwise-successful, or
    otherwise-failing, run. Not gated on any tree being clean -- unlike a
    fixture bundle's own manifest (``portfolio_replication.export_provenance``),
    this arm builds nothing that must be reproduced bit-for-bit from a
    committed checkout; it only needs to name, on every record, exactly what
    ran it (review finding I5).
    """
    import importlib.metadata

    p_head: Optional[str] = None
    try:
        import qpo  # pyright: ignore[reportMissingImports]

        if qpo.__file__ is not None:
            p_root = Path(qpo.__file__).resolve().parents[2]
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=p_root, capture_output=True, text=True)
            if head.returncode == 0:
                p_head = head.stdout.strip()
    except Exception:
        p_head = None
    versions: Dict[str, Optional[str]] = {}
    for name in ("qpo", "dimod", "dwave-samplers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"p_head": p_head, "package_versions": versions}


def run_portfolio_deadline_arm(n: int, k: int, beta_label: str, seed: int) -> Dict[str, Any]:
    """The historical-settings neal arm (500 reads / 500 sweeps) under the 10 s application deadline.

    dwave-neal (``dwave.samplers.SimulatedAnnealingSampler``) has no internal
    deadline concept at all -- "the reference SA provider itself does not
    enforce it" (design's portfolio-replication contract) -- so this is the
    orchestration that contract calls for: run to completion, never killed
    at 10 seconds, classify with :func:`deadline_status` from the measured
    END-TO-END time (sampling plus P's own repair and weighting, review
    finding I5 -- a slow repair phase must be able to push a run late even
    when sampling alone was fast), and KEEP a late-but-good record for
    diagnosis rather than reclassify it as an on-time win. The instance is
    always synthetic (``PORTFOLIO_PROVENANCE``), never a captured historical
    one; every record says so. Requires P's pinned environment.
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
    sampling_s = time.perf_counter() - t_start

    repair_s: Optional[float] = None
    extra: Dict[str, Any] = {}
    if response is not None:
        t_repair_start = time.perf_counter()
        column = {var: idx for idx, var in enumerate(response.variables)}
        order = [column[i] for i in range(qubo.n)]
        bits = np.asarray(response.record.sample, dtype=np.float64)[:, order]
        spins = bits * 2.0 - 1.0
        scored = pr.score_reads(problem, qubo, spins)
        extra["objective"] = float(scored["objective"])
        extra["feasible"] = bool(scored["feasible"])
        extra["raw_feasible_count"] = scored["raw_feasible_count"]
        extra["returned_reads"] = int(scored["returned_reads"])

        # "feasible" above comes entirely from P's own repair and weighting; a raw read
        # rarely satisfies the cardinality constraint on its own (raw_feasible_count is
        # commonly 0). Report the WINNING read's own weighting status too, so a reader
        # never mistakes a repaired-feasible answer for a raw one (review finding 6).
        # weighting_failed is P's own tri-state: None means "not observed to fail,"
        # never "succeeded" -- an unknown weighting result is recorded as unknown.
        diagnostics = pr.per_read_diagnostics(problem, spins)
        selected_bits = np.asarray(scored["selected_bits"], dtype=np.int8)
        match = np.all(bits.astype(np.int8) == selected_bits, axis=1)
        if match.any():
            selected_record = diagnostics["records"][int(np.argmax(match))]
            extra["weighting_failed"] = selected_record["weighting_failed"]
            extra["selected_raw_cardinality"] = selected_record["raw_cardinality"]
        else:
            # Should not happen (selected_bits always comes from one of the input
            # reads); recorded as unknown rather than silently assumed one way or
            # the other if it ever does.
            extra["weighting_failed"] = None
            extra["selected_raw_cardinality"] = None
        repair_s = time.perf_counter() - t_repair_start

    end_to_end_s = time.perf_counter() - t_start
    provenance = _portfolio_deadline_provenance()  # bookkeeping only; excluded from the timed budget above
    record: Dict[str, Any] = {
        "schema": "round2-portfolio-deadline-v1",
        "n_assets": n, "cardinality_k": k, "beta_label": beta_label,
        "frustration_beta": float(problem.frustration_beta),
        "reads": PORTFOLIO_NEAL_READS, "sweeps": PORTFOLIO_NEAL_SWEEPS, "seed": neal_seed,
        "deadline_s": PORTFOLIO_DEADLINE_S,
        "provenance": PORTFOLIO_PROVENANCE,
        "elapsed_s": sampling_s,
        "repair_s": repair_s,
        "end_to_end_s": end_to_end_s,
        "exit_ok": exit_ok,
        "error": error,
        "p_head": provenance["p_head"],
        "package_versions": provenance["package_versions"],
        "status": deadline_status(end_to_end_s, PORTFOLIO_DEADLINE_S, exit_ok),
    }
    record.update(extra)
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
    observed_kernel = meta["observed_kernel"]
    if observed_kernel != "cpu-msa-f64":
        # review finding M8: this was captured into the return value but never
        # checked -- a silent kernel substitution here would have gone unnoticed.
        raise RunnerError(f"cpu_lite_seed_lanes observed {observed_kernel!r}, not cpu-msa-f64")
    order = np.argsort(energies, kind="stable")[:lanes]
    lane_spins = spins[order]
    unique_lanes = len(np.unique(lane_spins, axis=0))
    return {
        "spins": lane_spins, "energies": energies[order], "lanes": lanes,
        "unique_lanes": unique_lanes, "duplicate_lanes": lanes - unique_lanes,
        "observed_kernel": observed_kernel,
    }


def seeded_sweep_run_key(
    cell: str, nonce: str, bundles_root: PathLike, seed_source: str, sweeps: int, identity: Mapping[str, Any],
) -> str:
    """The resume identity of one seeded-sweep job: the same "solver build matters"
    guarantee :meth:`CpuJob.run_key` gives the campaign (review finding 5), applied
    to the seeded/cold arm (review finding I4: "check the key on resume, refusing a
    mismatch as the campaign does").
    """
    return round2_io.run_id(
        {
            "bundles_root": str(bundles_root), "cell": cell, "nonce": nonce,
            "seed_source": seed_source, "sweeps": sweeps, "solver_identity": dict(identity),
        }
    )


def execute_seeded_sweep_job(
    cell: str, nonce: str, bundles_root: PathLike, round1_root: PathLike, seed_source: str,
    sweeps: int = SEEDED_SWEEPS,
) -> Tuple[Dict[str, Any], Optional[Dict[str, np.ndarray]]]:
    """One seeded/cold comparison at ``sweeps``: :data:`SEED_LANES` lanes from ``seed_source``
    ("qpu" or "cpu-lite"), plus :data:`COLD_LANES` lanes on a genuinely separate cold
    anneal, both on ``cpu-msa-f64``.

    Two independent ``sample_research`` calls, not one (review finding I4): packing
    both lane groups into a single call with ``initial_spins`` covering only the
    seed rows makes the KERNEL apply the seeded run's shortened, midpoint-start beta
    schedule to the whole call -- the nominally "cold" reads never see the hot end of
    the ladder. The seeded call keeps that midpoint start (the geometric mean of the
    beta range, Round 1's own ``seeded_sweep.py`` convention); the cold call passes no
    ``initial_spins`` and no ``start_beta`` at all (illegal together per
    ``Msa.sample_research``'s own contract), so it anneals the full ladder from the
    hot end, unseeded. Both share the identical ``beta_range``, so the only
    difference between them is exactly the one this comparison is meant to measure.
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

    seed, seed_input_hash = seed_for(manifest["hash"], "cpu-msa-f64", sweeps, SEED_LANES, f"{seed_source}-seeded")
    cold_seed, cold_seed_input_hash = seed_for(manifest["hash"], "cpu-msa-f64", sweeps, COLD_LANES, "cold")

    identity = solver_identity()
    run_key = seeded_sweep_run_key(cell, nonce, bundles_root, seed_source, sweeps, identity)

    quip_msa = _msa()
    sampler = quip_msa.Msa()
    beta_range = quip_msa.default_beta_range(h, edges, j)
    hot, cold = beta_range
    seeded_start_beta = math.sqrt(hot * cold)
    cold_start_beta = hot  # the unseeded call's own default: the full ladder starts hot.
    seeded_beta_ladder = np.geomspace(seeded_start_beta, cold, sweeps)
    cold_beta_ladder = np.geomspace(cold_start_beta, cold, sweeps)
    record: Dict[str, Any] = {
        "schema": "round2-seeded-sweep-v2",
        "cell": cell, "nonce": nonce, "model_hash": manifest["hash"], "seed_source": seed_source,
        "sweeps": sweeps, "reads": reads, "seed_lanes": SEED_LANES, "cold_lanes": COLD_LANES,
        "seed": seed, "seed_input_hash": seed_input_hash, "lite_source_seed": lite_seed,
        "cold_seed": cold_seed, "cold_seed_input_hash": cold_seed_input_hash,
        "unique_seed_lanes": lanes["unique_lanes"], "duplicate_seed_lanes": lanes["duplicate_lanes"],
        "beta_range": [float(hot), float(cold)],
        "seeded_start_beta": float(seeded_start_beta), "cold_start_beta": float(cold_start_beta),
        "seeded_beta_ladder": seeded_beta_ladder.tolist(), "cold_beta_ladder": cold_beta_ladder.tolist(),
        "solver_identity": identity, "run_key": run_key,
    }

    try:
        seeded_spins, _seeded_energies, seeded_meta = sampler.sample_research(
            h, edges, j, kernel="cpu-msa-f64", num_sweeps=sweeps, num_reads=SEED_LANES,
            seed=seed, beta_range=beta_range, initial_spins=lanes["spins"], start_beta=seeded_start_beta,
        )
        cold_spins, _cold_energies, cold_meta = sampler.sample_research(
            h, edges, j, kernel="cpu-msa-f64", num_sweeps=sweeps, num_reads=COLD_LANES,
            seed=cold_seed, beta_range=beta_range,
        )
    except ValueError as exc:
        # cpu-msa-f64 has no eligibility limit at all (unlike cpu-msa-unit): any
        # ValueError here is an unanticipated defect, never a known ineligibility
        # (review findings 7/8 -- "unsupported" is reserved for pre-checked reasons).
        record.update(unsupported=False, unsupported_reason=None, exit_ok=False, error=f"{type(exc).__name__}: {exc}")
        return record, None

    for observed_kernel in (seeded_meta["observed_kernel"], cold_meta["observed_kernel"]):
        if observed_kernel != "cpu-msa-f64":
            raise RunnerError(f"{cell}/{nonce}: seeded sweep observed {observed_kernel!r}, not cpu-msa-f64")

    spins = np.concatenate([seeded_spins, cold_spins], axis=0)
    rescored = regimes.energy(spins, h, edges, j)
    if not np.isfinite(rescored).all():
        raise RunnerError(f"{cell}/{nonce}: seeded sweep produced a nonfinite score")

    seeded_energies = rescored[:SEED_LANES]
    cold_energies = rescored[SEED_LANES:]
    record.update(
        unsupported=False, unsupported_reason=None, exit_ok=True, error=None,
        observed_kernel="cpu-msa-f64", representation=seeded_meta["representation"],
        best_seeded_energy=float(seeded_energies.min()), best_cold_energy=float(cold_energies.min()),
        unique_reads=int(len(np.unique(spins, axis=0))),
    )
    return record, {"spins": spins, "energies": rescored}
