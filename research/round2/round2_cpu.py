#!/usr/bin/env python3
"""Round 2's controlled CPU comparison: pilot, campaign, and cost estimate.

Task 8 of docs/superpowers/plans/2026-09-22-regime-search-round2.md. Every
measured arm compares ``cpu-sa``, ``cpu-msa-f64``, and ``cpu-msa-unit`` on
the identical logical model from a Task 2/5 bundle (never ``cpu-msa``, the
auto-selecting arm). See ``round2_runner.py`` for the identity, host-
contamination, and per-job execution this script orchestrates.

Subcommands:

``run-one``   Runs exactly one job, in the current process, and writes its
              record (and samples, if it completed) atomically. This is what
              ``pilot``/``campaign`` spawn as a fresh subprocess per
              measured run; it can also be invoked directly for one job.
``pilot``     The five-model pilot: the first five sorted nonces per cell,
              512/2,048 sweeps, three timing repetitions, one pinned core
              at a time.
``campaign``  The complete 100-model, five-depth comparison. Resumable: a
              job whose latest attempt already succeeded (or was explicitly
              unsupported) is skipped, so the exact command line below is
              safe to hand to taskd and re-run after any interruption.
              ``--repeat-contaminated`` runs one more attempt for exactly
              the jobs whose latest attempt was contaminated, leaving every
              earlier attempt on disk untouched.
``estimate``  Prints a campaign cost estimate from a pilot's recorded wall
              times.
``seeded-sweep``
              Step 9: the seeded/cold weighted-MSA comparison at 32,768
              sweeps, 32 seed lanes plus 32 cold lanes, against the saved
              Round 1 QPU seeds and CPU-lite seeds.

No QPU calls happen anywhere in this script.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import sys
import threading
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

from quip_miner_dwave import regime_io, round2_io

import round2_runner as runner

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_BUNDLES_ROOT = Path("/home/carback1/quip-data/regimes/round2/bundles")
DEFAULT_CPU_ROOT = Path("/home/carback1/quip-data/regimes/round2/cpu")
DEFAULT_CAPTURED_CPU_ROOT = Path("/home/carback1/quip-data/regimes/round3/cpu")
DEFAULT_CAPTURE_MANIFEST = Path("/home/carback1/quip-data/regimes/round2/physical-capture-manifest.json")
ROUND1 = Path("/home/carback1/quip-data/regimes/round1")


DEFAULT_CELLS = ["clique-portfolio", "cubic-dimer-pm1", "diamond-pm1", "native-125", "native-pm1"]

def _load_index(bundles_root: Path) -> Dict[str, Any]:
    return json.loads((bundles_root / "index.json").read_text(encoding="utf-8"))


def _parse_cpu_list(value: str) -> List[int]:
    return [int(x) for x in value.split(",") if x.strip()]


# ------------------------------------------------------------- attempt-based paths


def _base_name(job: "runner.CpuJob") -> str:
    return f"{job.nonce}__{job.kernel}__{job.sweeps}__{job.repetition_kind}-{job.repetition_id}__{job.variant}"


def _attempt_paths(out_dir: Path, job: "runner.CpuJob", attempt: int) -> Tuple[Path, Path]:
    name = f"{_base_name(job)}__attempt{attempt}"
    cell_dir = out_dir / job.cell
    return cell_dir / f"{name}.json", cell_dir / f"{name}.npz"


def _existing_attempts(out_dir: Path, job: "runner.CpuJob") -> List[int]:
    """Every attempt number already on disk for ``job``, ascending. Never overwritten
    or deleted (review finding 3): a new attempt always gets the next number.
    """
    cell_dir = out_dir / job.cell
    prefix = _base_name(job) + "__attempt"
    if not cell_dir.exists():
        return []
    attempts = []
    for path in cell_dir.glob(f"{prefix}*.json"):
        suffix = path.stem[len(prefix):]
        if suffix.isdigit():
            attempts.append(int(suffix))
    return sorted(attempts)


def _read_validated_record(
    record_path: Path, job: "runner.CpuJob", bundles_root: Path, identity: Dict[str, Any],
) -> Dict[str, Any]:
    """Read a record file, refusing (loudly) if its run key does not match this exact
    job under the CURRENT solver identity -- e.g. quip_msa was rebuilt since this
    attempt was written (review finding 5): resume must never silently mix builds.
    """
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = job.run_key(bundles_root, identity)
    if record.get("run_key") != expected:
        raise SystemExit(
            f"{record_path} holds run key {record.get('run_key')!r}, and this job's run key "
            f"(under the current solver build) is {expected!r}; refusing to resume from a record "
            "that is not this exact job under this exact build. Move or remove the stale file, or "
            "start a new output root, before running again."
        )
    return record


def _is_terminal(
    record: Dict[str, Any], out_dir: Path, job: "runner.CpuJob", attempt: int, *, hard_deadline_s: float,
) -> bool:
    """Whether the latest attempt on disk means no further attempt is needed by default.

    A failed record is never terminal (always retried) UNLESS it is a hard-deadline
    kill recorded under the SAME hard deadline this run would use -- retrying that
    would just repeat the identical timeout (review finding 1: "do not rerun a job
    that already has a timeout record for the same deadline"). Raising the hard
    deadline (a CLI override, or a size-scaled default that changed) makes a stale
    kill record non-terminal again, so a wider deadline gets its own real attempt. A
    completed (non-unsupported) record additionally needs its samples file to
    actually exist on disk to count as done (review finding 2): the record is
    written last, so its mere presence already implies the samples exist, but this
    is checked directly anyway rather than assumed.
    """
    if not record.get("exit_ok"):
        if record.get("killed_reason") == "hard_deadline":
            return record.get("hard_deadline_s") == hard_deadline_s
        return False
    if not record.get("unsupported"):
        _, samples_path = _attempt_paths(out_dir, job, attempt)
        if not samples_path.exists():
            return False
    return True


def resolve_run(
    out_dir: Path, job: "runner.CpuJob", bundles_root: Path, identity: Dict[str, Any], hard_deadline_s: float,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """``(existing_terminal_record_or_None, attempt_number_to_use)`` for ``job``, given
    the hard deadline THIS run would use for it (see :func:`_is_terminal`).
    """
    attempts = _existing_attempts(out_dir, job)
    if not attempts:
        return None, 0
    latest = attempts[-1]
    record_path, _ = _attempt_paths(out_dir, job, latest)
    record = _read_validated_record(record_path, job, bundles_root, identity)
    if _is_terminal(record, out_dir, job, latest, hard_deadline_s=hard_deadline_s):
        return record, latest
    return None, latest + 1


def _is_contaminated_and_terminal(record: Dict[str, Any]) -> bool:
    return bool(record.get("exit_ok")) and bool((record.get("host") or {}).get("contaminated"))


def _default_hard_deadline(bundles_root: Path, job: "runner.CpuJob") -> float:
    _manifest, arrays = round2_io.read_bundle(bundles_root / job.cell / job.nonce)
    return runner.estimate_hard_deadline_s(job.sweeps, len(arrays["h"]), job.reads)


# ---------------------------------------------------------------------- run-one


def cmd_run_one(args: argparse.Namespace) -> int:
    job = runner.CpuJob.from_dict(json.loads(args.job_json))
    bundles_root = Path(args.bundles_root)
    out_dir = Path(args.out_dir)
    cpu = args.cpu

    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    sibling = runner.cpu_sibling(cpu) if cpu is not None else None
    host_before = runner.sample_host(cpu, sibling) if cpu is not None else None

    identity = runner.solver_identity()
    hard_deadline_s = args.hard_deadline_s
    if hard_deadline_s is None:
        hard_deadline_s = _default_hard_deadline(bundles_root, job)

    if args.attempt is not None:
        # An orchestrator already decided this exact attempt (plain resolution, or
        # --repeat-contaminated, which plain resolution has no way to reproduce on
        # its own): use it as given, never re-derive a different one.
        attempt = args.attempt
        record_path, _ = _attempt_paths(out_dir, job, attempt)
        existing = None
        if record_path.exists():
            candidate = _read_validated_record(record_path, job, bundles_root, identity)
            if _is_terminal(candidate, out_dir, job, attempt, hard_deadline_s=hard_deadline_s):
                existing = candidate
    else:
        # Standalone use (no orchestrator): resolve our own attempt.
        existing, attempt = resolve_run(out_dir, job, bundles_root, identity, hard_deadline_s)

    if existing is not None:
        print(f"skip (resumed, attempt {attempt}): {_attempt_paths(out_dir, job, attempt)[0]}")
        return 0 if existing.get("exit_ok") else 1

    try:
        record, samples = runner.execute_cpu_job(job, bundles_root)
    except Exception as exc:  # never drop a job's outcome; see failure_record's docstring
        # Best-effort only (minor 16): the bundle may itself be why this failed, so a
        # second read failing too must never mask the original exception.
        model_hash = None
        try:
            bundle_manifest, _ = round2_io.read_bundle(bundles_root / job.cell / job.nonce)
            model_hash = bundle_manifest.get("hash")
        except Exception:
            pass
        record, samples = runner.failure_record(
            job, bundles_root, exc, identity=identity, model_hash=model_hash,
        ), None

    host_after = runner.sample_host(cpu, sibling) if cpu is not None else None
    if host_before is not None and host_after is not None:
        contamination = runner.check_contamination(
            host_before, host_after, concurrent_workers=args.concurrent_workers,
        )
        record["host"] = {
            "cpu": cpu,
            "sibling_cpu": sibling,
            "affinity": sorted(os.sched_getaffinity(0)),
            "before": host_before.to_dict(),
            "after": host_after.to_dict(),
            **contamination.to_dict(),
        }
    else:
        record["host"] = None
    record["attempt"] = attempt
    record["hard_deadline_s"] = hard_deadline_s
    # Labeled honestly (change: parallel workers, requirement 2): timing analysis
    # must be able to tell a serially-pinned run apart from one that shared the
    # host with other concurrent workers, without guessing from context.
    # timing_mode describes only THIS run's own worker count, never the whole
    # host: a "serial" run launched while an unrelated parallel campaign is
    # also running on the same machine is still labeled "serial" (review,
    # workers fix round 1, item 2) -- it says nothing about what else the host
    # was doing at the time. Contamination sampling is the mechanism that
    # covers that, not this label.
    record["timing_mode"] = args.timing_mode
    record["concurrent_workers"] = args.concurrent_workers

    record_path, samples_path = _attempt_paths(out_dir, job, attempt)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    # Samples first, record last (review finding 2): an interruption between the two
    # never leaves a record on disk claiming a completed run with no samples to show.
    if samples is not None:
        regime_io.atomic_savez(samples_path, spins=samples["spins"], energies=samples["energies"])
    regime_io.atomic_write_json(record_path, record)
    return 0 if record.get("exit_ok") else 1


# ------------------------------------------------------------------ orchestration


def _run_one_job(
    job: "runner.CpuJob", bundles_root: Path, out_dir: Path, cpu: Optional[int], attempt: int,
    identity: Dict[str, Any], hard_deadline_s_override: Optional[float],
    *, concurrent_workers: int = 1,
    registry: Optional["runner.ActiveProcesses"] = None,
    cancelled_event: Optional["threading.Event"] = None,
) -> Dict[str, Any]:
    """Spawn exactly one fresh, pinned ``run-one`` subprocess for ``job``'s given
    ``attempt``, and return its record -- persisting a fallback record if the
    subprocess was killed before it could write anything at all (review finding 1).

    ``registry``/``cancelled_event`` given together means this is one worker
    thread of a parallel run (change: parallel workers): the underlying spawn is
    then :func:`runner.run_one_subprocess_tracked`, which raises
    :class:`runner.Cancelled` instead of returning if the parent was told to
    cancel -- propagated here uncaught, so a cancelled job gets no record at all,
    real or fallback (requirement 4: "writes no false completed record").
    """
    hard_deadline_s = hard_deadline_s_override
    if hard_deadline_s is None:
        hard_deadline_s = _default_hard_deadline(bundles_root, job)

    sibling = runner.cpu_sibling(cpu) if cpu is not None else None
    host_before = runner.sample_host(cpu, sibling) if cpu is not None else None

    if registry is not None:
        assert cancelled_event is not None
        exit_ok, wall_s = runner.run_one_subprocess_tracked(
            job, bundles_root=bundles_root, out_dir=out_dir, cpu=cpu, script_path=SCRIPT_PATH, attempt=attempt,
            registry=registry, cancelled_event=cancelled_event,
            hard_deadline_s=hard_deadline_s, hard_deadline_s_override=hard_deadline_s_override,
            concurrent_workers=concurrent_workers,
        )
        timing_mode = "parallel"
    else:
        exit_ok, wall_s = runner.run_one_subprocess(
            job, bundles_root=bundles_root, out_dir=out_dir, cpu=cpu, script_path=SCRIPT_PATH, attempt=attempt,
            hard_deadline_s=hard_deadline_s, hard_deadline_s_override=hard_deadline_s_override,
            timing_mode="serial" if concurrent_workers == 1 else "parallel", concurrent_workers=concurrent_workers,
        )
        timing_mode = "serial" if concurrent_workers == 1 else "parallel"

    record_path, _samples_path = _attempt_paths(out_dir, job, attempt)
    if record_path.exists():
        return json.loads(record_path.read_text(encoding="utf-8"))

    # The subprocess was killed by the hard deadline before it wrote anything at all
    # (or exited nonzero without writing, e.g. crashed before reaching its own
    # try/except): persist a fallback record atomically, with host set, so this
    # attempt is never silently lost (review finding 1).
    host_after = runner.sample_host(cpu, sibling) if cpu is not None else None
    fallback = runner.failure_record(
        job, bundles_root,
        RuntimeError(f"hard subprocess deadline exceeded ({hard_deadline_s:.1f}s); exit_ok={exit_ok}"),
        identity=identity,
    )
    fallback["wall_s"] = wall_s
    fallback["attempt"] = attempt
    fallback["hard_deadline_s"] = hard_deadline_s
    if wall_s >= hard_deadline_s:
        # The parent's own TimeoutExpired path fired (review finding M1): it waited
        # the full hard deadline before killing the child, so this really is the
        # kill _is_terminal must never retry under the same deadline.
        fallback["killed_reason"] = "hard_deadline"
    elif exit_ok:
        fallback["killed_reason"] = "no_record_written"
    else:
        # The child exited (crashed, or exited nonzero before ever reaching its own
        # write) well before the parent's hard deadline ever fired: a plain crash,
        # not a deadline kill, and must stay retryable (_is_terminal treats anything
        # other than "hard_deadline" as non-terminal already).
        fallback["killed_reason"] = "crashed_no_record"
    fallback["timing_mode"] = timing_mode
    fallback["concurrent_workers"] = concurrent_workers
    if host_before is not None and host_after is not None:
        contamination = runner.check_contamination(host_before, host_after, concurrent_workers=concurrent_workers)
        fallback["host"] = {
            "cpu": cpu, "sibling_cpu": sibling, "affinity": [cpu],
            "before": host_before.to_dict(), "after": host_after.to_dict(),
            **contamination.to_dict(),
        }
    else:
        fallback["host"] = None
    record_path.parent.mkdir(parents=True, exist_ok=True)
    regime_io.atomic_write_json(record_path, fallback)
    return fallback


def _jobs_needing_a_run(
    jobs: List["runner.CpuJob"], bundles_root: Path, out_dir: Path, identity: Dict[str, Any],
    *, hard_deadline_s: Optional[float], repeat_contaminated: bool,
) -> Tuple[List[Tuple["runner.CpuJob", int]], List[Dict[str, Any]]]:
    """``(to_run, already_done)``: which ``(job, attempt)`` pairs still need a fresh
    subprocess this pass, and the already-terminal records to report as-is.

    Resolved entirely up front, in the (single) calling thread, before any worker
    (serial or parallel) starts: safe by construction for parallel dispatch,
    since each job then appears in ``to_run`` at most once and is handed to
    exactly one worker (task brief change: "no two workers may ever take the
    same job").

    ``repeat_contaminated``: only jobs whose latest attempt is a completed-but-
    contaminated record get a new attempt; every other job (never run, or
    already clean) is left untouched. Earlier attempts are never overwritten or
    deleted (review finding 3).
    """
    to_run: List[Tuple["runner.CpuJob", int]] = []
    already_done: List[Dict[str, Any]] = []
    for job in jobs:
        job_hard_deadline_s = hard_deadline_s if hard_deadline_s is not None else _default_hard_deadline(
            bundles_root, job,
        )
        if repeat_contaminated:
            attempts = _existing_attempts(out_dir, job)
            if not attempts:
                continue
            record_path, _ = _attempt_paths(out_dir, job, attempts[-1])
            latest_record = _read_validated_record(record_path, job, bundles_root, identity)
            if not _is_contaminated_and_terminal(latest_record):
                continue
            to_run.append((job, attempts[-1] + 1))
        else:
            existing, attempt = resolve_run(out_dir, job, bundles_root, identity, job_hard_deadline_s)
            if existing is not None:
                already_done.append(existing)
            else:
                to_run.append((job, attempt))
    return to_run, already_done


def _run_jobs_serial(
    to_run: List[Tuple["runner.CpuJob", int]], bundles_root: Path, out_dir: Path, cpu: Optional[int],
    identity: Dict[str, Any], hard_deadline_s: Optional[float], *, concurrent_workers: int = 1,
) -> List[Dict[str, Any]]:
    """Run every ``(job, attempt)`` pair one at a time, each in its own fresh,
    pinned subprocess. This is the path both a plain ``--workers 1`` run and
    the timing subset take (review, minor 9 -- an earlier docstring here
    wrongly claimed the timing subset's ``concurrent_workers`` reflects the
    campaign's own ``--workers``; it does not): every caller of this function
    passes exactly one CPU, so ``concurrent_workers`` defaults to, and stays,
    1, an honest label for a genuinely single-worker run either way.
    """
    records: List[Dict[str, Any]] = []
    for i, (job, attempt) in enumerate(to_run, 1):
        record = _run_one_job(
            job, bundles_root, out_dir, cpu, attempt, identity, hard_deadline_s,
            concurrent_workers=concurrent_workers,
        )
        records.append(record)
        print(
            f"[{i}/{len(to_run)}] {job.cell}/{job.nonce} {job.kernel} sweeps={job.sweeps} "
            f"attempt={attempt}: exit_ok={record.get('exit_ok')} wall_s={record.get('wall_s')}"
        )
    return records


def _run_jobs_parallel(
    to_run: List[Tuple["runner.CpuJob", int]], bundles_root: Path, out_dir: Path, cpus: Sequence[int],
    identity: Dict[str, Any], hard_deadline_s: Optional[float],
) -> List[Dict[str, Any]]:
    """Run every ``(job, attempt)`` pair through a pool of ``len(cpus)`` worker
    threads, one physical core each, one job at a time per worker.

    A shared ``queue.Queue`` hands out jobs one at a time, so no two workers can
    ever take the same job (task brief change, requirement 4). SIGTERM/SIGINT
    install ONE handler here, in the main thread (``signal.signal`` cannot be
    called from a worker thread), that kills every worker's currently-running
    child via a shared :class:`runner.ActiveProcesses` registry and sets a
    ``threading.Event`` every worker checks; a job whose subprocess was killed
    this way raises :class:`runner.Cancelled` in its own worker thread and gets
    no record (requirement 4: "writes no false completed record"). Once every
    worker has stopped, a cancelled run re-raises :class:`runner.Cancelled` in
    the main thread too, so the caller's own exit reflects the cancellation.
    """
    workers = len(cpus)
    job_queue: "queue.Queue[Tuple[runner.CpuJob, int]]" = queue.Queue()
    for item in to_run:
        job_queue.put(item)
    total = job_queue.qsize()

    records: List[Dict[str, Any]] = []
    records_lock = threading.Lock()
    worker_errors: List[BaseException] = []
    registry = runner.ActiveProcesses()
    cancelled = threading.Event()

    def handle_cancel(signum: int, _frame: Any) -> None:
        cancelled.set()
        registry.kill_all()

    previous_term = signal.signal(signal.SIGTERM, handle_cancel)
    previous_int = signal.signal(signal.SIGINT, handle_cancel)

    def worker(cpu: int) -> None:
        while not cancelled.is_set():
            try:
                job, attempt = job_queue.get_nowait()
            except queue.Empty:
                return
            try:
                record = _run_one_job(
                    job, bundles_root, out_dir, cpu, attempt, identity, hard_deadline_s,
                    concurrent_workers=workers, registry=registry, cancelled_event=cancelled,
                )
            except runner.Cancelled:
                return
            except BaseException as exc:  # never let one worker's bug hang the others silently
                # Logged immediately (review, minor 7): only the FIRST error is
                # re-raised after every thread joins, which can be a long time on
                # a real campaign; printing here means every failure is visible
                # right away, not just the one that eventually propagates.
                print(
                    f"worker on cpu {cpu} failed on {job.cell}/{job.nonce} {job.kernel} "
                    f"sweeps={job.sweeps} attempt={attempt}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                with records_lock:
                    worker_errors.append(exc)
                return
            with records_lock:
                records.append(record)
                count = len(records)
            print(
                f"[{count}/{total}] {job.cell}/{job.nonce} {job.kernel} sweeps={job.sweeps} "
                f"attempt={attempt} cpu={cpu}: exit_ok={record.get('exit_ok')} wall_s={record.get('wall_s')}"
            )

    threads = [threading.Thread(target=worker, args=(cpu,), name=f"round2-worker-cpu{cpu}") for cpu in cpus]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)

    if cancelled.is_set():
        raise runner.Cancelled("the parent received a cancel signal during a parallel run")
    if worker_errors:
        raise worker_errors[0]
    return records


def _run_jobs(
    jobs: List["runner.CpuJob"], bundles_root: Path, out_dir: Path, cpus: Sequence[Optional[int]],
    *, hard_deadline_s: Optional[float] = None, repeat_contaminated: bool = False,
) -> List[Dict[str, Any]]:
    """Run every job in ``jobs``, each in its own fresh, pinned subprocess, over
    ``len(cpus)`` workers (1 for a serial run, more for a parallel one).
    """
    identity = runner.solver_identity()
    to_run, records = _jobs_needing_a_run(
        jobs, bundles_root, out_dir, identity, hard_deadline_s=hard_deadline_s,
        repeat_contaminated=repeat_contaminated,
    )
    if len(cpus) <= 1:
        cpu = cpus[0] if cpus else None
        records.extend(_run_jobs_serial(to_run, bundles_root, out_dir, cpu, identity, hard_deadline_s))
    else:
        concrete_cpus = [cpu for cpu in cpus if cpu is not None]
        if len(concrete_cpus) != len(cpus):
            raise ValueError("a parallel run (workers > 1) needs a real CPU for every worker, not None")
        records.extend(_run_jobs_parallel(to_run, bundles_root, out_dir, concrete_cpus, identity, hard_deadline_s))
    return records


def _summarize(records: List[Dict[str, Any]]) -> None:
    by_cell: Dict[str, List[float]] = {}
    contaminated = 0
    unsupported: List[Tuple[str, str, str]] = []
    for record in records:
        if record.get("wall_s") is not None:
            by_cell.setdefault(record["cell"], []).append(record["wall_s"])
        host = record.get("host")
        if host and host.get("contaminated"):
            contaminated += 1
        if record.get("unsupported"):
            unsupported.append((record["cell"], record["requested_kernel"], record.get("unsupported_reason", "")))
    print("\n--- per-cell wall time (s) ---")
    for cell, times in sorted(by_cell.items()):
        print(f"{cell}: n={len(times)} total={sum(times):.2f} median={median(times):.4f}")
    print(f"\ncontaminated runs: {contaminated}")
    print(f"\nunsupported arms ({len(unsupported)}):")
    for cell, kernel, reason in sorted(set(unsupported)):
        print(f"  {cell}/{kernel}: {reason}")


def cmd_pilot(args: argparse.Namespace) -> int:
    bundles_root = Path(args.bundles_root)
    out_dir = Path(args.out_root) / "pilot"
    index = _load_index(bundles_root)
    jobs = runner.build_pilot_jobs(index, args.cells)
    cpus = runner.select_worker_cpus(args.workers, cpus=args.cpus)
    records = _run_jobs(
        jobs, bundles_root, out_dir, cpus,
        hard_deadline_s=args.hard_deadline_s, repeat_contaminated=args.repeat_contaminated,
    )
    _summarize(records)
    return 0


def cmd_campaign(args: argparse.Namespace) -> int:
    bundles_root = Path(args.bundles_root)
    index = _load_index(bundles_root)
    if args.timing_subset:
        # Its own output directory, never the quality campaign's own attempt
        # stream. It runs on a loaded host and takes the fastest run as the run
        # speed (maintainer, 2026-09-24), so it uses every requested worker.
        out_dir = Path(args.out_root) / "timing-subset"
        jobs = runner.build_timing_subset_jobs(index, args.cells)
    else:
        out_dir = Path(args.out_root) / "campaign"
        jobs = runner.build_campaign_jobs(index, args.cells)
    cpus = runner.select_worker_cpus(args.workers, cpus=args.cpus)
    records = _run_jobs(
        jobs, bundles_root, out_dir, cpus,
        hard_deadline_s=args.hard_deadline_s, repeat_contaminated=args.repeat_contaminated,
    )
    _summarize(records)
    return 0


def cmd_captured(args: argparse.Namespace) -> int:
    bundles_root = Path(args.bundles_root)
    index = _load_index(bundles_root)
    capture_manifest = json.loads(Path(args.capture_manifest).read_text(encoding="utf-8"))
    jobs = runner.build_captured_jobs(index, capture_manifest)
    out_dir = Path(args.out_root) / "captured"
    cpus = runner.select_worker_cpus(args.workers, cpus=args.cpus)
    records = _run_jobs(
        jobs, bundles_root, out_dir, cpus,
        hard_deadline_s=args.hard_deadline_s, repeat_contaminated=args.repeat_contaminated,
    )
    _summarize(records)
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    pilot_root = Path(args.pilot_root)
    pilot_wall_s: Dict[Tuple[str, int], List[float]] = {}
    for record_path in sorted(pilot_root.glob("*/*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("wall_s") is None or record.get("unsupported"):
            continue
        key = (record["requested_kernel"], record["sweeps"])
        pilot_wall_s.setdefault(key, []).append(record["wall_s"])
    medians = {key: median(values) for key, values in pilot_wall_s.items()}
    if not medians:
        print("no completed pilot records found; run `pilot` first", file=sys.stderr)
        return 1
    unsupported_cells = {
        record["cell"]
        for record_path in sorted(pilot_root.glob("*/*.json"))
        for record in [json.loads(record_path.read_text(encoding="utf-8"))]
        if record.get("unsupported") and record.get("requested_kernel") == runner.UNIT_KERNEL
    }
    estimate = runner.estimate_campaign(
        medians, unsupported_kernel_cells=len(unsupported_cells), workers=args.workers,
    )
    print(json.dumps(estimate, indent=2))
    return 0


def cmd_seeded_sweep(args: argparse.Namespace) -> int:
    """Task brief step 9: seeded/cold weighted MSA at 32,768 sweeps, 32 seed + 32 cold lanes.

    Writes to ``seeded-sweep-v2``, not the original ``seeded-sweep`` (review finding
    I4): the cold lanes are now a genuinely separate full-ladder anneal rather than
    the tail of the seeded call, which makes every prior seeded-sweep output
    superseded. The old directory is never read or deleted here.
    """
    bundles_root = Path(args.bundles_root)
    round1_root = Path(args.round1_root)
    out_dir = Path(args.out_root) / "seeded-sweep-v2"
    index = _load_index(bundles_root)
    identity = runner.solver_identity()

    results = []
    for cell in args.cells:
        rows = sorted((row for row in index["rows"] if row["cell"] == cell), key=lambda row: row["nonce"])
        for row in rows[: args.models]:
            for seed_source in runner.SEED_SOURCES:
                record_path = out_dir / cell / f"{row['nonce']}__{seed_source}.json"
                samples_path = out_dir / cell / f"{row['nonce']}__{seed_source}.npz"
                expected_key = runner.seeded_sweep_run_key(
                    cell, row["nonce"], bundles_root, seed_source, runner.SEEDED_SWEEPS, identity,
                )
                if record_path.exists():
                    existing = json.loads(record_path.read_text(encoding="utf-8"))
                    # Refuse a mismatch as the campaign does (review finding I4): e.g.
                    # quip_msa was rebuilt since this attempt was written, and resuming
                    # from it would silently mix results from two different builds.
                    if existing.get("run_key") != expected_key:
                        raise SystemExit(
                            f"{record_path} holds run key {existing.get('run_key')!r}, and this job's "
                            f"run key (under the current solver build) is {expected_key!r}; refusing to "
                            "resume from a record that is not this exact job under this exact build. "
                            "Move or remove the stale file, or start a new output root, before running "
                            "again."
                        )
                    # Only a genuinely completed (or explicitly unsupported) prior run,
                    # with its samples actually on disk, is "done": a failed record is
                    # retried, never silently treated as final -- the same failure this
                    # hit once (a fixable rescoring bug) must not be able to hide behind
                    # an old file forever. A completed record with no samples file means
                    # an interruption between the two writes; also retried (finding 2).
                    if existing.get("exit_ok") and (existing.get("unsupported") or samples_path.exists()):
                        print(f"skip (resumed): {record_path}")
                        results.append(existing)
                        continue
                    print(f"retrying an incomplete or previously failed run: {record_path}")
                try:
                    record, samples = runner.execute_seeded_sweep_job(
                        cell, row["nonce"], bundles_root, round1_root, seed_source,
                    )
                except Exception as exc:  # never let one bad job end the whole loop
                    record, samples = {
                        "schema": "round2-seeded-sweep-v2", "cell": cell, "nonce": row["nonce"],
                        "seed_source": seed_source, "exit_ok": False, "unsupported": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "solver_identity": identity, "run_key": expected_key,
                    }, None
                record_path.parent.mkdir(parents=True, exist_ok=True)
                # Samples first, record last (finding 2): an interruption between the two
                # never leaves a record on disk claiming success with no samples to show.
                if samples is not None:
                    regime_io.atomic_savez(samples_path, spins=samples["spins"], energies=samples["energies"])
                regime_io.atomic_write_json(record_path, record)
                results.append(record)
                print(
                    f"{cell}/{row['nonce'][:8]} {seed_source}: exit_ok={record.get('exit_ok')} "
                    f"unsupported={record.get('unsupported')} "
                    f"duplicate_seed_lanes={record.get('duplicate_seed_lanes')}"
                )
    ok = sum(1 for r in results if r.get("exit_ok"))
    print(f"\n{ok}/{len(results)} seeded-sweep jobs completed")
    return 0




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run_one = sub.add_parser("run-one", help="Run exactly one job in this process.")
    run_one.add_argument("--job-json", required=True, help="A CpuJob, as JSON.")
    run_one.add_argument("--bundles-root", required=True)
    run_one.add_argument("--out-dir", required=True)
    run_one.add_argument("--cpu", type=int, default=None)
    run_one.add_argument(
        "--attempt", type=int, default=None,
        help="Use exactly this attempt number (an orchestrator already decided it). "
        "Omit for standalone use, which resolves its own attempt.",
    )
    run_one.add_argument(
        "--hard-deadline-s", type=float, default=None,
        help="Override the size-scaled default (crash protection only; enforced by the parent, not here).",
    )
    run_one.add_argument(
        "--timing-mode", choices=("serial", "parallel"), default="serial",
        help="This run's own worker count only -- never a claim about anything else on the host.",
    )
    run_one.add_argument("--concurrent-workers", type=int, default=1)
    run_one.set_defaults(func=cmd_run_one)

    pilot = sub.add_parser("pilot", help="The five-model pilot.")
    pilot.add_argument("--bundles-root", default=str(DEFAULT_BUNDLES_ROOT))
    pilot.add_argument("--out-root", default=str(DEFAULT_CPU_ROOT))
    pilot.add_argument("--cells", nargs="+", default=list(DEFAULT_CELLS))
    pilot.add_argument("--workers", type=int, default=1, help="Concurrent workers, one physical core each.")
    pilot.add_argument(
        "--cpus", type=_parse_cpu_list, default=None,
        help="Comma-separated logical CPUs, one per physical core (overrides --workers's count).",
    )
    pilot.add_argument("--hard-deadline-s", type=float, default=None, help="Override the size-scaled default.")
    pilot.add_argument(
        "--repeat-contaminated", action="store_true",
        help="Only add one new attempt for jobs whose latest attempt was contaminated.",
    )
    pilot.set_defaults(func=cmd_pilot)

    campaign = sub.add_parser("campaign", help="The complete 100-model, five-depth comparison.")
    campaign.add_argument("--bundles-root", default=str(DEFAULT_BUNDLES_ROOT))
    campaign.add_argument("--out-root", default=str(DEFAULT_CPU_ROOT))
    campaign.add_argument("--cells", nargs="+", default=list(DEFAULT_CELLS))
    campaign.add_argument("--workers", type=int, default=1, help="Concurrent workers, one physical core each.")
    campaign.add_argument(
        "--cpus", type=_parse_cpu_list, default=None,
        help="Comma-separated logical CPUs, one per physical core (overrides --workers's count).",
    )
    campaign.add_argument("--hard-deadline-s", type=float, default=None, help="Override the size-scaled default.")
    campaign.add_argument(
        "--repeat-contaminated", action="store_true",
        help="Only add one new attempt for jobs whose latest attempt was contaminated.",
    )
    campaign.add_argument(
        "--timing-subset", action="store_true",
        help="Run the timing subset instead of the full campaign: one model per cell, every "
        "depth, every eligible kernel, in its own output directory.",
    )
    campaign.set_defaults(func=cmd_campaign)

    captured = sub.add_parser("captured", help="The captured-model four-kernel comparison.")
    captured.add_argument("--bundles-root", default=str(DEFAULT_BUNDLES_ROOT))
    captured.add_argument("--capture-manifest", default=str(DEFAULT_CAPTURE_MANIFEST))
    captured.add_argument("--out-root", default=str(DEFAULT_CAPTURED_CPU_ROOT))
    captured.add_argument("--workers", type=int, default=1, help="Concurrent workers, one physical core each.")
    captured.add_argument(
        "--cpus", type=_parse_cpu_list, default=None,
        help="Comma-separated logical CPUs, one per physical core (overrides --workers's count).",
    )
    captured.add_argument("--hard-deadline-s", type=float, default=None, help="Override the size-scaled default.")
    captured.add_argument(
        "--repeat-contaminated", action="store_true",
        help="Only add one new attempt for jobs whose latest attempt was contaminated.",
    )
    captured.set_defaults(func=cmd_captured)

    estimate = sub.add_parser("estimate", help="Print a campaign cost estimate from pilot records.")
    estimate.add_argument("--pilot-root", default=str(DEFAULT_CPU_ROOT / "pilot"))
    estimate.add_argument("--workers", type=int, default=12)
    estimate.set_defaults(func=cmd_estimate)

    seeded = sub.add_parser(
        "seeded-sweep", help="Step 9: seeded/cold weighted MSA at 32,768 sweeps, 32+32 lanes.",
    )
    seeded.add_argument("--bundles-root", default=str(DEFAULT_BUNDLES_ROOT))
    seeded.add_argument("--round1-root", default=str(ROUND1))
    seeded.add_argument("--out-root", default=str(DEFAULT_CPU_ROOT))
    seeded.add_argument("--cells", nargs="+", default=list(DEFAULT_CELLS))
    seeded.add_argument("--models", type=int, default=5)
    seeded.set_defaults(func=cmd_seeded_sweep)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
