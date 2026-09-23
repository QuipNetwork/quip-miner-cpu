#!/usr/bin/env python3
"""Round 2's controlled CPU comparison: pilot, campaign, and cost estimate.

Task 8 of docs/superpowers/plans/2026-09-22-regime-search-round2.md. Every
measured arm compares ``cpu-sa``, ``cpu-msa-f64``, and ``cpu-msa-unit`` on
the identical logical model from a Task 2/5 bundle (never ``cpu-msa``, the
auto-selecting arm). See ``quip_miner_dwave/round2_runner.py`` for the
identity, host-contamination, and per-job execution this script orchestrates.

Five subcommands:

``run-one``   Runs exactly one job, in the current process, and writes its
              record (and samples, if it completed) atomically. This is what
              ``pilot``/``campaign`` spawn as a fresh subprocess per
              measured run; it can also be invoked directly for one job.
``pilot``     The five-model pilot: the first five sorted nonces per cell,
              512/2,048 sweeps, three timing repetitions, one pinned core
              at a time.
``campaign``  The complete 100-model, five-depth comparison. Resumable: a
              record whose run key already exists on disk is skipped, so
              the exact command line below is safe to hand to taskd and
              re-run after any interruption.
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
import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from quip_miner_dwave import regime_io

import round2_runner as runner

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_BUNDLES_ROOT = Path("/home/carback1/quip-data/regimes/round2/bundles")
DEFAULT_CPU_ROOT = Path("/home/carback1/quip-data/regimes/round2/cpu")
ROUND1 = Path("/home/carback1/quip-data/regimes/round1")

#: Which physical core this runner pins every measured subprocess to.
#: Core 4 (and its SMT sibling) is an arbitrary but fixed and documented
#: choice, away from core 0 where most interrupt handling lands.
DEFAULT_CPU = 4


def _load_index(bundles_root: Path) -> Dict[str, Any]:
    return json.loads((bundles_root / "index.json").read_text(encoding="utf-8"))


def _paths(out_dir: Path, job: "runner.CpuJob") -> Tuple[Path, Path]:
    name = f"{job.nonce}__{job.kernel}__{job.sweeps}__{job.repetition_kind}-{job.repetition_id}__{job.variant}"
    cell_dir = out_dir / job.cell
    return cell_dir / f"{name}.json", cell_dir / f"{name}.npz"


def _load_existing(record_path: Path, job: "runner.CpuJob", bundles_root: Path) -> Optional[Dict[str, Any]]:
    """The record already on disk for ``job``, or None if there is none to resume from.

    Refuses to silently reuse a file whose run key does not match: that file
    was written by a different job spec (a changed sweep count, kernel, or
    bundle root under the same output path), and resuming from it would
    silently mix two runs.
    """
    if not record_path.exists():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = job.run_key(bundles_root)
    if record.get("run_key") != expected:
        raise SystemExit(
            f"{record_path} holds run key {record.get('run_key')!r}, and this job's run key is "
            f"{expected!r}; refusing to resume from a record that is not this exact job. Move or "
            "remove the stale file before running again."
        )
    if record.get("host") is None:
        # A record without host information was interrupted between run-one's
        # write and completion of host sampling (or predates that field);
        # never trust it as "done" -- resume by rerunning it.
        return None
    return record


def cmd_run_one(args: argparse.Namespace) -> int:
    job = runner.CpuJob.from_dict(json.loads(args.job_json))
    bundles_root = Path(args.bundles_root)
    out_dir = Path(args.out_dir)
    cpu = args.cpu

    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    sibling = runner.cpu_sibling(cpu) if cpu is not None else None
    host_before = runner.sample_host(cpu, sibling) if cpu is not None else None

    record_path, samples_path = _paths(out_dir, job)
    existing = _load_existing(record_path, job, bundles_root)
    if existing is not None:
        print(f"skip (resumed): {record_path}")
        return 0 if existing.get("exit_ok") else 1

    try:
        record, samples = runner.execute_cpu_job(job, bundles_root)
    except Exception as exc:  # never drop a job's outcome; see failure_record's docstring
        record, samples = runner.failure_record(job, bundles_root, exc), None

    host_after = runner.sample_host(cpu, sibling) if cpu is not None else None
    if host_before is not None and host_after is not None:
        contamination = runner.check_contamination(host_before, host_after)
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

    record_path.parent.mkdir(parents=True, exist_ok=True)
    regime_io.atomic_write_json(record_path, record)
    if samples is not None:
        regime_io.atomic_savez(samples_path, spins=samples["spins"], energies=samples["energies"])
    return 0 if record.get("exit_ok") else 1


# ------------------------------------------------------------------ orchestration


def _run_jobs(jobs: List["runner.CpuJob"], bundles_root: Path, out_dir: Path, cpu: int) -> List[Dict[str, Any]]:
    """Run every job in ``jobs`` one at a time, each in its own fresh, pinned subprocess.

    Resumable: a job whose record already exists (and matches its run key,
    with host information present) is skipped without spawning a subprocess.
    """
    records: List[Dict[str, Any]] = []
    for i, job in enumerate(jobs, 1):
        record_path, _samples_path = _paths(out_dir, job)
        existing = _load_existing(record_path, job, bundles_root)
        if existing is not None:
            records.append(existing)
            continue
        exit_ok, wall_s = runner.run_one_subprocess(
            job, bundles_root=bundles_root, out_dir=out_dir, cpu=cpu, script_path=SCRIPT_PATH,
        )
        if record_path.exists():
            records.append(json.loads(record_path.read_text(encoding="utf-8")))
        else:
            # The subprocess was killed by the hard deadline before it could write
            # anything at all (task brief step 5's "subprocess timeout and cleanup").
            fallback = runner.failure_record(job, bundles_root, RuntimeError("hard subprocess deadline exceeded"))
            fallback["wall_s"] = wall_s
            fallback["host"] = None
            records.append(fallback)
        print(f"[{i}/{len(jobs)}] {job.cell}/{job.nonce} {job.kernel} sweeps={job.sweeps}: "
              f"exit_ok={records[-1].get('exit_ok')} wall_s={records[-1].get('wall_s')}")
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
    records = _run_jobs(jobs, bundles_root, out_dir, args.cpu)
    _summarize(records)
    return 0


def cmd_campaign(args: argparse.Namespace) -> int:
    bundles_root = Path(args.bundles_root)
    out_dir = Path(args.out_root) / "campaign"
    index = _load_index(bundles_root)
    jobs = runner.build_campaign_jobs(index, args.cells)
    records = _run_jobs(jobs, bundles_root, out_dir, args.cpu)
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
    """Task brief step 9: seeded/cold weighted MSA at 32,768 sweeps, 32 seed + 32 cold lanes."""
    bundles_root = Path(args.bundles_root)
    round1_root = Path(args.round1_root)
    out_dir = Path(args.out_root) / "seeded-sweep"
    index = _load_index(bundles_root)

    results = []
    for cell in args.cells:
        rows = sorted((row for row in index["rows"] if row["cell"] == cell), key=lambda row: row["nonce"])
        for row in rows[: args.models]:
            for seed_source in runner.SEED_SOURCES:
                record_path = out_dir / cell / f"{row['nonce']}__{seed_source}.json"
                samples_path = out_dir / cell / f"{row['nonce']}__{seed_source}.npz"
                if record_path.exists():
                    existing = json.loads(record_path.read_text(encoding="utf-8"))
                    # Only a genuinely completed (or explicitly unsupported) prior run is
                    # "done": a failed record is retried, never silently treated as final --
                    # the same failure this hit once (a fixable rescoring bug) must not be
                    # able to hide behind an old file forever.
                    if existing.get("exit_ok"):
                        print(f"skip (resumed): {record_path}")
                        results.append(existing)
                        continue
                    print(f"retrying a previously failed run: {record_path}")
                try:
                    record, samples = runner.execute_seeded_sweep_job(
                        cell, row["nonce"], bundles_root, round1_root, seed_source,
                    )
                except runner.RunnerError as exc:
                    record, samples = {
                        "schema": "round2-seeded-sweep-v1", "cell": cell, "nonce": row["nonce"],
                        "seed_source": seed_source, "exit_ok": False, "unsupported": False,
                        "error": str(exc),
                    }, None
                record_path.parent.mkdir(parents=True, exist_ok=True)
                regime_io.atomic_write_json(record_path, record)
                if samples is not None:
                    regime_io.atomic_savez(samples_path, spins=samples["spins"], energies=samples["energies"])
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
    run_one.set_defaults(func=cmd_run_one)

    pilot = sub.add_parser("pilot", help="The five-model pilot.")
    pilot.add_argument("--bundles-root", default=str(DEFAULT_BUNDLES_ROOT))
    pilot.add_argument("--out-root", default=str(DEFAULT_CPU_ROOT))
    pilot.add_argument("--cells", nargs="+", default=[
        "clique-portfolio", "cubic-dimer-pm1", "diamond-pm1", "native-125", "native-pm1",
    ])
    pilot.add_argument("--cpu", type=int, default=DEFAULT_CPU)
    pilot.set_defaults(func=cmd_pilot)

    campaign = sub.add_parser("campaign", help="The complete 100-model, five-depth comparison.")
    campaign.add_argument("--bundles-root", default=str(DEFAULT_BUNDLES_ROOT))
    campaign.add_argument("--out-root", default=str(DEFAULT_CPU_ROOT))
    campaign.add_argument("--cells", nargs="+", default=[
        "clique-portfolio", "cubic-dimer-pm1", "diamond-pm1", "native-125", "native-pm1",
    ])
    campaign.add_argument("--cpu", type=int, default=DEFAULT_CPU)
    campaign.set_defaults(func=cmd_campaign)

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
    seeded.add_argument("--cells", nargs="+", default=[
        "clique-portfolio", "cubic-dimer-pm1", "diamond-pm1", "native-125", "native-pm1",
    ])
    seeded.add_argument("--models", type=int, default=5)
    seeded.set_defaults(func=cmd_seeded_sweep)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
