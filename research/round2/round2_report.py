#!/usr/bin/env python3
"""Round 2 regime search report: per-regime tables, the portfolio pipeline's
own tables, and next-test decisions, built from CPU pilot/campaign records,
the seeded/cold weighted-MSA control, and the portfolio deadline arm's
results (synthetic fixtures, historical settings -- see the ``PORTFOLIO``
naming rule below). See ``round2_metrics.py`` for the pure comparison,
feasibility, and bootstrap primitives this module assembles into a document.

Task 10 of D's ``docs/superpowers/plans/2026-09-22-regime-search-round2.md``
is the brief this implements. D's own ``scripts/regime_report.py``
feasibility-rounding fix is out of scope here (D is read-only): this report
carries its own count-based feasibility formatter
(:func:`round2_metrics.format_feasibility`).

No QPU submission happens here. Every QPU-paired table and status sentence is
built from data actually loaded, never from a fixed claim: when no capture is
given (``--qpu-root`` and ``--physical-capture-manifest`` omitted), every
QPU-paired table states plainly that it has no comparable pairs, rather than
asserting a specific reason (such as "pending approval") this module cannot
verify on its own. When real physical-pilot captures ARE given, they are
checked against the capture manifest (:func:`load_qpu_captures`) before
anything is paired or plotted. Figures render as small, dependency-free SVG
documents: no plotting library is installed in D's pinned venv (numpy and
scipy are; matplotlib is not), and this report must run entirely under that
venv.

``cpu-msa`` naming: the report derives this routed arm from the two measured
Round 2 arms, choosing unit when supported and successful, otherwise f64.
The campaign runner still never requests the bare name. Round 1's caveat
remains: its bare ``cpu-msa`` name means the unit kernel specifically, while
the general binding routes between unit and f64. The report never pools the
two rounds by that bare kernel name.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from quip_miner_dwave import regime_io, regimes

import round2_metrics as metrics
import round2_runner as runner

DEFAULT_KERNELS: Tuple[str, ...] = ("cpu-sa", "cpu-msa-f64", "cpu-msa-unit", "cpu-msa")
FULL_DEPTHS: Tuple[int, ...] = (512, 2048, 8192, 32768, 131072)
PILOT_DEPTHS: Tuple[int, ...] = (512, 2048)


def expected_per_arm_for(run: str) -> int:
    """The job count one (cell, kernel, sweeps) arm should have, from the SAME
    constants ``round2_cpu.py`` uses to build the job list (review, Important
    item 2): the pilot's 5 models x 3 timing repetitions, or the campaign's
    100 models x 1 repetition. Read from ``round2_runner`` rather than
    duplicated here, so the two can never silently drift apart.
    """
    if run == "pilot":
        return runner.PILOT_MODELS_PER_CELL * runner.PILOT_TIMING_REPS
    return runner.CAMPAIGN_MODELS_PER_CELL

_ATTEMPT_RE = re.compile(r"^(?P<base>.+)__attempt(?P<attempt>\d+)\.json$")


# ------------------------------------------------------------------- loading


def load_cpu_records(run_dir: Path, cell: str) -> List[Dict[str, Any]]:
    """Every job's record for one cell of a CPU run directory: the latest
    ``exit_ok`` attempt for quality, with timing fields from the successful
    attempt with the smallest finite ``wall_s`` substituted in when that differs.

    Only files named ``*__attempt<N>.json`` count: a bare ``<base>.json`` with
    no attempt suffix predates the attempt-numbering fix and is fully
    contaminated (controller ruling on this task), so it is ignored outright
    rather than silently mixed in.

    A job with no successful attempt at all reports its latest attempt as-is
    (a failed record), unchanged from before. A job WITH a successful attempt
    never reports a later crash's ``None`` energy in its place: taking simply
    the highest attempt number, as this function used to, let a
    ``--repeat-contaminated`` retry that crashed outright silently discard an
    earlier, valid (if contaminated) quality result. Independently, timing
    tracks the successful attempt with the smallest finite ``wall_s``,
    regardless of host contamination or timing mode.
    """
    cell_dir = Path(run_dir) / cell
    if not cell_dir.exists():
        return []
    attempts_by_base: Dict[str, Dict[int, Tuple[Dict[str, Any], Path]]] = {}
    for path in cell_dir.glob("*__attempt*.json"):
        match = _ATTEMPT_RE.match(path.name)
        if match is None:
            continue
        base = match.group("base")
        attempt = int(match.group("attempt"))
        attempts_by_base.setdefault(base, {})[attempt] = (json.loads(path.read_text(encoding="utf-8")), path)

    records: List[Dict[str, Any]] = []
    for by_attempt in attempts_by_base.values():
        ok_attempts = {n: pair for n, pair in by_attempt.items() if pair[0].get("exit_ok")}
        if not ok_attempts:
            records.append(by_attempt[max(by_attempt)][0])
            continue
        quality_record, quality_path = ok_attempts[max(ok_attempts)]
        timed_attempts = {
            n: pair for n, pair in ok_attempts.items()
            if pair[0].get("wall_s") is not None
            and math.isfinite(float(pair[0]["wall_s"]))
        }
        record = dict(quality_record)
        record["_samples_path"] = str(quality_path.with_suffix(".npz"))
        if timed_attempts:
            fastest_record = min(timed_attempts.values(), key=lambda pair: float(pair[0]["wall_s"]))[0]
            if fastest_record is not quality_record:
                for field in (
                    "wall_s", "elapsed_sampling_s", "setup_s", "graph_setup_s",
                    "host", "timing_mode", "concurrent_workers",
                ):
                    record[field] = fastest_record.get(field)
        records.append(record)
    return records


def derive_cpu_msa_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build the report's routed ``cpu-msa`` arm from measured unit and f64 rows."""
    by_job: Dict[Tuple[Any, Any], Dict[str, Dict[str, Any]]] = {}
    for record in records:
        kernel = record.get("requested_kernel")
        if kernel in ("cpu-msa-unit", "cpu-msa-f64") and record.get("model_hash"):
            key = (record.get("model_hash"), record.get("sweeps"))
            by_job.setdefault(key, {})[kernel] = record

    routed: List[Dict[str, Any]] = []
    for sources in by_job.values():
        unit = sources.get("cpu-msa-unit")
        if unit is not None and unit.get("exit_ok") and not unit.get("unsupported"):
            source = unit
        else:
            source = sources.get("cpu-msa-f64")
        if source is None:
            continue
        copy = dict(source)
        copy["requested_kernel"] = "cpu-msa"
        copy["routed_kernel"] = source["requested_kernel"]
        routed.append(copy)
    return routed


def load_seeded_sweep_records(seeded_root: Path, cell: str) -> List[Dict[str, Any]]:
    """Every seeded/cold weighted-MSA record for one cell (Task 8 brief, step 9)."""
    cell_dir = Path(seeded_root) / cell
    if not cell_dir.exists():
        return []
    records = []
    for path in sorted(cell_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_samples_path"] = str(path.with_suffix(".npz"))
        records.append(record)
    return records


def load_portfolio_deadline(path: Path) -> List[Dict[str, Any]]:
    """The deadline arm's (synthetic fixtures, historical settings) own ``results.json``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload.get("records", []))


def load_capture_proposal(path: Path) -> Dict[str, Any]:
    """The physical-range/portfolio capture proposal, whatever its approval status."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_physical_capture_manifest(path: Path) -> Dict[str, Any]:
    """The physical-range capture manifest: the job list a real capture is checked
    against (review, C1)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_qpu_captures(qpu_root: Path, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every real QPU physical-pilot capture under ``qpu_root``, verified against
    ``manifest`` (review, Critical item C1).

    Requires ``mock == False``: a mock capture must never enter a report that
    claims to compare real QPU energies against CPU ones. Requires the
    capture's own ``capture_key`` to name a job in the manifest, and that
    job's ``model_hash`` to match the capture's -- the same "never trust a
    lone file" posture the CPU runner takes toward its own resume records.
    Pairing downstream is by ``model_hash`` alone, per the review's own
    instruction, never by nonce or file position.
    """
    jobs_by_key = {job["capture_key"]: job for job in manifest.get("jobs", [])}
    captures: List[Dict[str, Any]] = []
    for path in sorted(Path(qpu_root).glob("*/scale-*/qpu-*/*.npz")):
        with np.load(path) as data:
            if bool(data["mock"]):
                raise ValueError(f"{path}: a mock capture is not allowed in the physical-pilot report")
            capture_key = str(data["capture_key"])
            model_hash = str(data["model_hash"])
            job = jobs_by_key.get(capture_key)
            if job is None:
                raise ValueError(f"{path}: capture_key {capture_key!r} is not in the manifest")
            if job["model_hash"] != model_hash:
                raise ValueError(
                    f"{path}: model_hash {model_hash!r} does not match the manifest's "
                    f"{job['model_hash']!r} for capture_key {capture_key!r}"
                )
            energies = np.asarray(data["energies"], dtype=np.float64)
            captures.append({
                "path": str(path),
                "cell": job["cell"],
                "nonce": job["nonce"],
                "model_hash": model_hash,
                "requested_scale": float(job["requested_scale"]),
                "anneal_us": int(data["anneal_us"]),
                "best_energy": float(energies.min()),
                "access_us": int(data["access_us"]),
                "end_to_end_s": float(data["end_to_end_s"]),
            })
    return captures


def group_captures_by_arm(captures: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, float, int], List[Dict[str, Any]]]:
    """Captures grouped by ``(cell, requested_scale, anneal_us)`` -- one physical
    QPU arm (review, C1: "pair each capture with CPU records by model_hash,
    per physical scale, anneal time and CPU depth")."""
    grouped: Dict[Tuple[str, float, int], List[Dict[str, Any]]] = {}
    for capture in captures:
        key = (capture["cell"], capture["requested_scale"], capture["anneal_us"])
        grouped.setdefault(key, []).append(capture)
    return grouped


def pair_qpu_cpu_by_model(
    captures: Sequence[Dict[str, Any]], cpu_records: Sequence[Dict[str, Any]],
) -> List[Tuple[str, float, float]]:
    """``(model_hash, qpu_best, cpu_best)`` for every model with both a real QPU
    capture and a completed, finite CPU record (review, C1: pair by
    ``model_hash``, never by nonce or position). A failed, unsupported, or
    nonfinite CPU record excludes that model from the comparable set, the
    same rule :func:`round2_metrics.summarize_arm` applies elsewhere.
    """
    cpu_best: Dict[str, float] = {}
    for record in cpu_records:
        if not record.get("exit_ok") or record.get("unsupported"):
            continue
        best = _finite_float(record.get("best_energy"))
        model_hash = record.get("model_hash")
        if best is not None and model_hash:
            cpu_best[model_hash] = best
    pairs: List[Tuple[str, float, float]] = []
    for capture in captures:
        model_hash = capture["model_hash"]
        if model_hash in cpu_best:
            pairs.append((model_hash, capture["best_energy"], cpu_best[model_hash]))
    return pairs


def matched_runtime_comparison(
    captures: Sequence[Dict[str, Any]], cpu_records: Sequence[Dict[str, Any]],
    kernels: Sequence[str] = DEFAULT_KERNELS,
) -> Dict[Tuple[str, float, int, str], Dict[str, Any]]:
    """Compare each captured model with each kernel at equal access time and energy.

    CPU records are paired by model hash across every available depth. The
    result is keyed by ``(cell, scale, anneal_us, kernel)`` for table rendering
    and direct testing without file access.
    """
    results: Dict[Tuple[str, float, int, str], Dict[str, Any]] = {}
    for arm, arm_captures in group_captures_by_arm(captures).items():
        cell, scale, anneal = arm
        captures_by_model = {capture["model_hash"]: capture for capture in arm_captures}
        for kernel in kernels:
            cpu_by_model: Dict[str, List[Tuple[int, float, float]]] = {}
            routes_by_model: Dict[str, set[str]] = {}
            for record in cpu_records:
                if record.get("cell") != cell or record.get("requested_kernel") != kernel:
                    continue
                if not record.get("exit_ok") or record.get("unsupported"):
                    continue
                energy = _finite_float(record.get("best_energy"))
                sampling_s = _finite_float(record.get("elapsed_sampling_s"))
                model_hash = record.get("model_hash")
                sweeps = record.get("sweeps")
                if energy is None or sampling_s is None or sampling_s < 0 or not model_hash:
                    continue
                if not isinstance(sweeps, int):
                    continue
                if model_hash in captures_by_model:
                    cpu_by_model.setdefault(model_hash, []).append((sweeps, sampling_s, energy))
                    route = record.get("routed_kernel")
                    if isinstance(route, str):
                        routes_by_model.setdefault(model_hash, set()).add(route)

            matched_models = sorted(set(captures_by_model) & set(cpu_by_model))
            strict_outcomes: List[str] = []
            numeric_outcomes: List[str] = []
            material_outcomes: List[str] = []
            over_budget = 0
            time_to_energy: List[float] = []
            time_ratios: List[float] = []
            routed_kernels: set[str] = set()
            for model_hash in matched_models:
                capture = captures_by_model[model_hash]
                qpu_energy = _finite_float(capture.get("best_energy"))
                access_us = _finite_float(capture.get("access_us"))
                if qpu_energy is None or access_us is None or access_us <= 0:
                    continue
                model_records = cpu_by_model[model_hash]
                routed_kernels.update(routes_by_model.get(model_hash, set()))
                budget_s = access_us / 1_000_000
                affordable = [row for row in model_records if row[1] <= budget_s]
                shallowest = min(model_records, key=lambda row: row[0])
                if shallowest[1] > budget_s:
                    over_budget += 1
                else:
                    _depth, _sampling_s, cpu_energy = max(affordable, key=lambda row: row[0])
                    columns = metrics.quality_columns(qpu_energy, cpu_energy)
                    strict_outcomes.append(columns["strict"])
                    numeric_outcomes.append(columns["numeric_tolerance"])
                    material_outcomes.append(columns["material"])

                qualifying_times = [
                    sampling_s for _depth, sampling_s, cpu_energy in model_records
                    if metrics.quality_outcome(qpu_energy, cpu_energy, metrics.NUMERIC_TOLERANCE) != "qpu"
                ]
                if qualifying_times:
                    reached_s = min(qualifying_times)
                    time_to_energy.append(reached_s)
                    time_ratios.append(reached_s / budget_s)

            matched_captures = [captures_by_model[model_hash] for model_hash in matched_models]
            access_seconds = [
                value / 1_000_000 for capture in matched_captures
                if (value := _finite_float(capture.get("access_us"))) is not None
            ]
            end_to_end_seconds = [
                value for capture in matched_captures
                if (value := _finite_float(capture.get("end_to_end_s"))) is not None
            ]
            results[(cell, scale, anneal, kernel)] = {
                "models": len(matched_models),
                "routed_kernels": sorted(routed_kernels),
                "median_qpu_access_s": float(np.median(access_seconds)) if access_seconds else None,
                "median_qpu_end_to_end_s": float(np.median(end_to_end_seconds)) if end_to_end_seconds else None,
                "equal_budget_counts": {
                    "strict": metrics.tally_quality_outcomes(strict_outcomes),
                    "numeric_tolerance": metrics.tally_quality_outcomes(numeric_outcomes),
                    "material": metrics.tally_quality_outcomes(material_outcomes),
                },
                "over_budget": over_budget,
                "reached_qpu_energy": len(time_to_energy),
                "not_reached": len(matched_models) - len(time_to_energy),
                "median_cpu_s_to_qpu_energy": float(np.median(time_to_energy)) if time_to_energy else None,
                "median_ratio_to_qpu_access": float(np.median(time_ratios)) if time_ratios else None,
            }
    return results


def reconcile_spend(qpu_root: Path, captures: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Submits, charges, open jobs, and the charged total against the sum of
    every capture's own ``access_us`` (review, C1: "reconcile spend from
    spend.jsonl ... and the total compared with the sum of access_us").
    """
    path = Path(qpu_root) / regime_io.SPEND_JOURNAL
    submits = charges = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)["event"]
            if event == regime_io.SUBMIT:
                submits += 1
            elif event == regime_io.CHARGE:
                charges += 1
    charged_us, open_jobs = regime_io.read_spend(qpu_root)
    total_access_us = sum(capture["access_us"] for capture in captures)
    return {
        "submits": submits,
        "charges": charges,
        "open_jobs": open_jobs,
        "charged_us": charged_us,
        "total_access_us": total_access_us,
        "reconciled": charged_us == total_access_us and open_jobs == 0,
    }


# ----------------------------------------------------------------- formatting


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}g}"


# -------------------------------------------------------------------- tables


def model_identity_table(records_by_cell: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    lines = [
        "## Model identity", "",
        "| Cell | Distinct models observed | Distinct model hashes |",
        "| -- | -- | -- |",
    ]
    for cell, records in records_by_cell.items():
        nonces = {r.get("nonce") for r in records if r.get("nonce")}
        hashes = {r.get("model_hash") for r in records if r.get("model_hash")}
        lines.append(f"| `{cell}` | {len(nonces)} | {len(hashes)} |")
    return lines


def solver_settings_table(records_by_cell: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    lines = [
        "## Solver settings", "",
        "| Cell | Kernels observed | Sweep depths observed | Reads |",
        "| -- | -- | -- | -- |",
    ]
    for cell, records in records_by_cell.items():
        kernels = sorted({
            str(k) for r in records
            if (k := r.get("requested_kernel")) is not None and k != "cpu-msa"
        })
        depths = sorted({int(d) for r in records if (d := r.get("sweeps")) is not None})
        reads = sorted({int(n) for r in records if (n := r.get("reads")) is not None})
        lines.append(
            f"| `{cell}` | {', '.join(f'`{k}`' for k in kernels) or 'n/a'} | "
            f"{', '.join(str(d) for d in depths) or 'n/a'} | {', '.join(str(r) for r in reads) or 'n/a'} |"
        )
    return lines


def _timing_cell(summary: Dict[str, Any], label: str) -> str:
    """``"median (n)"`` for one timing label, or ``"n/a (0)"`` with none observed
    (review, Important item 1: every timing label prints its own count, so a
    zero-sample label can never be mistaken for a real zero-second median).
    """
    info = summary["timing_by_label"].get(label)
    if not info or info["n"] == 0:
        return "n/a (0)"
    return f"{fmt(info['median_wall_s'])} ({info['n']})"


def depth_quality_time_table(
    records_by_cell: Dict[str, List[Dict[str, Any]]],
    *, kernels: Sequence[str], depths: Sequence[int], expected_per_arm: Optional[int] = None,
    timing_records_by_cell: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    timing_source: Optional[str] = None,
) -> List[str]:
    timing_label = timing_source or "quality run"
    quality_source_note = (
        "Best energy remains from campaign records. "
        if timing_source == "timing-subset" else "Best energy remains from the selected run. "
    )
    time_records_by_cell = timing_records_by_cell if timing_records_by_cell is not None else records_by_cell
    lines = [
        "## Depth, quality, and time", "",
        "Every row is one (cell, kernel, sweep-depth) arm. Energy is the primary comparison. "
        f"Timing uses `{timing_label}` records from a loaded host with parallel workers, one per physical core. "
        "For each job, its fastest successful run is its run speed. The timing columns show the median "
        "sampling and wall times across successful, supported records, regardless of host contamination "
        "or timing mode. The contaminated and parallel wall-time columns remain separate diagnostics. "
        f"Best energy uses every successful record. {quality_source_note}"
        "Sampling time is `elapsed_sampling_s`. Wall time is end-to-end.", "",
        "| Cell | Kernel | Sweeps | Observed | Missing | Completed | Failed | Unsupported | Nonfinite "
        f"| Median best energy | Sampling s, fastest run from {timing_label} (n) "
        f"| Wall s, fastest run from {timing_label} (n) | Wall s, contaminated (n) | Wall s, parallel (n) |",
        "| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |",
    ]
    for cell, records in records_by_cell.items():
        for kernel in kernels:
            for sweeps in depths:
                arm = [r for r in records if r.get("requested_kernel") == kernel and r.get("sweeps") == sweeps]
                summary = metrics.summarize_arm(arm, expected=expected_per_arm)
                source_records = time_records_by_cell.get(cell, [])
                timing_arm = [
                    r for r in source_records
                    if r.get("requested_kernel") == kernel and r.get("sweeps") == sweeps
                ]
                successful_timing = [
                    r for r in timing_arm
                    if r.get("exit_ok") and not r.get("unsupported")
                ]
                sampling_values = [
                    float(r["elapsed_sampling_s"])
                    for r in successful_timing
                    if r.get("elapsed_sampling_s") is not None
                    and math.isfinite(float(r["elapsed_sampling_s"]))
                ]
                wall_values = [
                    float(r["wall_s"])
                    for r in successful_timing
                    if r.get("wall_s") is not None and math.isfinite(float(r["wall_s"]))
                ]
                missing_reason = (
                    f"no matching {timing_label} record"
                    if timing_source == "timing-subset" and not successful_timing else None
                )
                sampling_cell = (
                    f"{fmt(float(np.median(sampling_values)))} ({len(sampling_values)})" if sampling_values
                    else f"n/a (0; {missing_reason or 'no finite sampling-time observations'})"
                )
                wall_cell = (
                    f"{fmt(float(np.median(wall_values)))} ({len(wall_values)})" if wall_values
                    else f"n/a (0; {missing_reason or 'no finite wall-time observations'})"
                )
                lines.append(
                    f"| `{cell}` | `{kernel}` | {sweeps} | {summary['observed']} | {fmt(summary['missing'])} | "
                    f"{summary['completed']} | {summary['failed']} | {summary['unsupported']} | "
                    f"{summary['nonfinite']} | {fmt(summary['median_best_energy'])} | "
                    f"{sampling_cell} | {wall_cell} | "
                    f"{_timing_cell(summary, metrics.CONTAMINATED)} | "
                    f"{_timing_cell(summary, metrics.PARALLEL)} |"
                )
    return lines


def lane_diversity_table(records_by_cell: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """Summarize saved-read diversity per arm, keeping each record's pairs within
    its own model and separating the seeded-sweep-v2 lane groups."""
    grouped: Dict[Tuple[str, str, int, str, str], List[Dict[str, Any]]] = {}
    for cell, records in records_by_cell.items():
        for record in records:
            if not record.get("exit_ok") or record.get("unsupported"):
                continue
            if str(record.get("schema", "")).startswith("round2-seeded-sweep-") and record.get("schema") != "round2-seeded-sweep-v2":
                continue
            samples_path = record.get("_samples_path")
            if not samples_path or not Path(samples_path).is_file():
                continue
            with np.load(samples_path) as data:
                spins = np.asarray(data["spins"])
            if record.get("schema") == "round2-seeded-sweep-v2":
                kernel = str(record.get("observed_kernel", "cpu-msa-f64"))
                seed_source = str(record.get("seed_source", "unknown"))
                seed_count = int(record["seed_lanes"])
                cold_count = int(record["cold_lanes"])
                batches = (
                    ("seeded", spins[:seed_count]),
                    ("cold", spins[seed_count:seed_count + cold_count]),
                )
            else:
                kernel = str(record.get("requested_kernel", "unknown"))
                seed_source = ""
                batches = (("all", spins),)
            for lane_group, batch in batches:
                key = (cell, kernel, int(record["sweeps"]), lane_group, seed_source)
                grouped.setdefault(key, []).append(metrics.lane_diversity(batch))

    lines = [
        "## Lane diversity by arm", "",
        "Unique states and pairwise Hamming distances come from each saved `.npz` spin batch. Per-arm "
        "unique counts and mean distances are averages across records. The minimum is the lowest within-record "
        "pairwise distance. Seeded-sweep-v2 separates the saved seeded and cold lane batches.", "",
        "| Cell | Kernel / seed source | Sweeps | Lane group | Records | Mean unique states / record | "
        "Mean pairwise Hamming / record | Minimum pairwise Hamming |",
        "| -- | -- | -- | -- | -- | -- | -- | -- |",
    ]
    for (cell, kernel, sweeps, lane_group, seed_source), values in sorted(grouped.items()):
        pair_means = [value["mean_pairwise_hamming"] for value in values if value["mean_pairwise_hamming"] is not None]
        pair_mins = [value["minimum_pairwise_hamming"] for value in values if value["minimum_pairwise_hamming"] is not None]
        kernel_label = f"{kernel} ({seed_source})" if seed_source else kernel
        lines.append(
            f"| `{cell}` | `{kernel_label}` | {sweeps} | {lane_group} | {len(values)} | "
            f"{fmt(float(np.mean([value['unique_states'] for value in values])))} | "
            f"{fmt(float(np.mean(pair_means))) if pair_means else 'n/a'} | "
            f"{min(pair_mins) if pair_mins else 'n/a'} |"
        )
    if not grouped:
        lines.append("| n/a | n/a | n/a | n/a | 0 | n/a | n/a | n/a |")
    return lines


def cpu_kernel_gap_table(
    records_by_cell: Dict[str, List[Dict[str, Any]]],
    *, kernels: Sequence[str], depths: Sequence[int], base_kernel: str = "cpu-sa",
) -> List[str]:
    """One paired, per-model gap between each kernel and ``base_kernel`` (review,
    M2): "Median best energy" in the depth table above compares medians over
    whatever model set each arm happened to complete, which silently drifts
    apart whenever either arm has a missing or failed model. This table pairs
    by `model_hash` instead, so every comparison is over the SAME models on
    both sides, with the shared count always shown.
    """
    lines = [
        "## Paired CPU kernel gaps", "",
        f"Each row pairs one kernel against `{base_kernel}` on the same models (matched by `model_hash`), "
        "never by comparing medians over two arms' completed sets, which can differ. Compared counts only "
        "models where both kernels completed with a finite energy at that depth.", "",
        "| Cell | Kernels | Sweeps | Compared | Strict other/base/tie |",
        "| -- | -- | -- | -- | -- |",
    ]
    any_row = False
    for cell, records in records_by_cell.items():
        for kernel in kernels:
            if kernel == base_kernel:
                continue
            for sweeps in depths:
                base_best: Dict[str, float] = {}
                other_best: Dict[str, float] = {}
                for record in records:
                    if record.get("sweeps") != sweeps or not record.get("exit_ok") or record.get("unsupported"):
                        continue
                    best = _finite_float(record.get("best_energy"))
                    model_hash = record.get("model_hash")
                    if best is None or not model_hash:
                        continue
                    requested = record.get("requested_kernel")
                    if requested == base_kernel:
                        base_best[model_hash] = best
                    elif requested == kernel:
                        other_best[model_hash] = best
                shared = sorted(set(base_best) & set(other_best))
                if not shared:
                    continue
                any_row = True
                outcomes = [
                    metrics.quality_outcome(other_best[m], base_best[m], metrics.STRICT_TOLERANCE) for m in shared
                ]
                tally = metrics.tally_quality_outcomes(outcomes)
                lines.append(
                    f"| `{cell}` | `{kernel}` vs `{base_kernel}` | {sweeps} | {len(shared)} | "
                    f"{_outcome_triple(tally)} |"
                )
    if not any_row:
        lines.append("")
        lines.append(f"No comparable pairs against `{base_kernel}` for any kernel or depth given.")
    return lines


def _cell_arms(
    cell: str, captures_by_arm: Dict[Tuple[str, float, int], List[Dict[str, Any]]],
) -> List[Tuple[str, float, int]]:
    return sorted((key for key in captures_by_arm if key[0] == cell), key=lambda key: (key[1], key[2]))


def qpu_outcome_table(
    cells: Sequence[str],
    captures_by_arm: Dict[Tuple[str, float, int], List[Dict[str, Any]]],
    records_by_cell: Dict[str, List[Dict[str, Any]]],
    kernels: Sequence[str], depths: Sequence[int],
) -> List[str]:
    lines = [
        "## Quantum processing unit wins, ties, and losses", "",
        "Every row pairs one physical-scale QPU arm (cell, scale, anneal time) against one CPU arm (kernel, "
        "sweep depth), matched by `model_hash` (review, C1). Compared counts only models with both a real, "
        "manifest-verified capture and a completed, finite CPU record.", "",
        "| Cell | Scale | Anneal us | Kernel | Sweeps | Compared | Strict qpu/cpu/tie | Numeric qpu/cpu/tie "
        "| Material qpu/cpu/tie |",
        "| -- | -- | -- | -- | -- | -- | -- | -- | -- |",
    ]
    any_row = False
    for cell in cells:
        records = records_by_cell.get(cell, [])
        for cell_name, scale, anneal in _cell_arms(cell, captures_by_arm):
            captures = captures_by_arm[(cell_name, scale, anneal)]
            for kernel in kernels:
                for sweeps in depths:
                    cpu_records = [
                        r for r in records if r.get("requested_kernel") == kernel and r.get("sweeps") == sweeps
                    ]
                    pairs = pair_qpu_cpu_by_model(captures, cpu_records)
                    if not pairs:
                        continue
                    any_row = True
                    columns = [metrics.quality_columns(qpu, cpu) for _, qpu, cpu in pairs]
                    strict = metrics.tally_quality_outcomes([c["strict"] for c in columns])
                    numeric = metrics.tally_quality_outcomes([c["numeric_tolerance"] for c in columns])
                    material = metrics.tally_quality_outcomes([c["material"] for c in columns])
                    kernel_label = _kernel_label(kernel, pairs, cpu_records)
                    lines.append(
                        f"| `{cell}` | {scale:g} | {anneal} | `{kernel_label}` | {sweeps} | {len(pairs)} | "
                        f"{_outcome_triple(strict)} | {_outcome_triple(numeric)} | {_outcome_triple(material)} |"
                    )
    if not any_row:
        lines.append("")
        lines.append(
            "No comparable pairs exist because no cell has both a physical-pilot QPU capture and a matching CPU arm yet."
        )
    return lines


def paired_gaps_table(
    cells: Sequence[str],
    captures_by_arm: Dict[Tuple[str, float, int], List[Dict[str, Any]]],
    records_by_cell: Dict[str, List[Dict[str, Any]]],
    kernels: Sequence[str], depths: Sequence[int],
) -> List[str]:
    lines = [
        "## Paired gaps", "",
        "Bootstrap paired-gap intervals (`round2_metrics.bootstrap_paired_gap`: 10,000 resamples, the fixed "
        "recorded seed, grouped by model, since every physical-pilot model is an independent draw). The gap "
        "is the same signed relative gap `quality_outcome` itself computes: negative means the QPU energy is "
        "lower (better).", "",
        "| Cell | Scale | Anneal us | Kernel | Sweeps | Groups | Point | 95% low | 95% high | Descriptive |",
        "| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |",
    ]
    any_row = False
    for cell in cells:
        records = records_by_cell.get(cell, [])
        for cell_name, scale, anneal in _cell_arms(cell, captures_by_arm):
            captures = captures_by_arm[(cell_name, scale, anneal)]
            for kernel in kernels:
                for sweeps in depths:
                    cpu_records = [
                        r for r in records if r.get("requested_kernel") == kernel and r.get("sweeps") == sweeps
                    ]
                    pairs = pair_qpu_cpu_by_model(captures, cpu_records)
                    if not pairs:
                        continue
                    any_row = True
                    gaps = [
                        (qpu - cpu) / max(abs(qpu), abs(cpu), 1e-12) for _, qpu, cpu in pairs
                    ]
                    groups = [model_hash for model_hash, _, _ in pairs]
                    result = metrics.bootstrap_paired_gap(gaps, groups)
                    kernel_label = _kernel_label(kernel, pairs, cpu_records)
                    lines.append(
                        f"| `{cell}` | {scale:g} | {anneal} | `{kernel_label}` | {sweeps} | {result['group_count']} | "
                        f"{fmt(result['point'], 4)} | {fmt(result['low'], 4)} | {fmt(result['high'], 4)} | "
                        f"{'yes' if result['descriptive'] else 'no'} |"
                    )
    if not any_row:
        lines.append("")
        lines.append(
            "No comparable pairs exist because no cell has both a physical-pilot QPU capture and a matching CPU arm yet."
        )
    return lines


def _finite_float(value: Any) -> Optional[float]:
    """``value`` as a finite float, or ``None`` when it is not one -- a narrowing
    helper so a caller can branch on ``is None`` and have the non-None case
    typed as ``float``, not ``Any | None``.
    """
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _outcome_triple(tally: Dict[str, int]) -> str:
    return f"{tally['qpu']}/{tally['cpu']}/{tally['tie']}"


def _kernel_label(
    kernel: str, pairs: Sequence[Tuple[str, float, float]], cpu_records: Sequence[Dict[str, Any]],
) -> str:
    if kernel != "cpu-msa":
        return kernel
    paired_hashes = {model_hash for model_hash, _qpu, _cpu in pairs}
    routed = _routed_kernels(cpu_records, paired_hashes)
    return f"cpu-msa ({_route_label(routed)})"


def _routed_kernels(cpu_records: Sequence[Dict[str, Any]], model_hashes: Iterable[Any]) -> set[str]:
    hashes = set(model_hashes)
    routed: set[str] = set()
    for record in cpu_records:
        route = record.get("routed_kernel")
        if record.get("model_hash") in hashes and isinstance(route, str):
            routed.add(route)
    return routed


def _route_label(routed: Iterable[str]) -> str:
    routed_kernels = set(routed)
    if len(routed_kernels) > 1:
        return "mixed"
    if routed_kernels == {"cpu-msa-unit"}:
        return "unit"
    if routed_kernels == {"cpu-msa-f64"}:
        return "f64"
    else:
        return "unknown"


def matched_runtime_table(
    captures: Sequence[Dict[str, Any]], cpu_records: Sequence[Dict[str, Any]], kernels: Sequence[str],
    *, source_description: Optional[str] = None,
) -> List[str]:
    results = matched_runtime_comparison(captures, cpu_records, kernels)
    lead = source_description or (
        "QPU time is charged access time for 64 reads, with end-to-end time also shown. CPU time is sampling "
        "time for 64 reads from the campaign, using the fastest successful attempt on a loaded host with "
        "parallel workers. Energy outcomes use the same strict, numeric-tolerance, and material rules as "
        "the equal-sweep tables. Equal budget uses the deepest CPU depth completed within that capture's "
        "access-time budget. Time to QPU energy uses the quickest CPU depth that reached the capture's best energy."
    )
    lines = [
        "## QPU against CPU at matched run time", "",
        lead, "",
        "| Cell | Scale | Anneal us | Kernel | Models | Median QPU access s | Median QPU end-to-end s | "
        "Equal budget: strict / numeric / material qpu/cpu/tie | Over budget | Reached QPU energy (n of Models) | "
        "Not reached | Median CPU s to QPU energy | Median ratio to QPU access |",
        "| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |",
    ]
    for (cell, scale, anneal, kernel), row in sorted(results.items()):
        counts = row["equal_budget_counts"]
        equal_budget = " / ".join(
            _outcome_triple(counts[name]) for name in ("strict", "numeric_tolerance", "material")
        )
        kernel_label = (
            f"cpu-msa ({_route_label(row['routed_kernels'])})"
            if kernel == "cpu-msa" else kernel
        )
        lines.append(
            f"| `{cell}` | {scale:g} | {anneal} | `{kernel_label}` | {row['models']} | "
            f"{fmt(row['median_qpu_access_s'])} | {fmt(row['median_qpu_end_to_end_s'])} | {equal_budget} | "
            f"{row['over_budget']} | {row['reached_qpu_energy']} of {row['models']} | {row['not_reached']} | "
            f"{fmt(row['median_cpu_s_to_qpu_energy'])} | {fmt(row['median_ratio_to_qpu_access'])} |"
        )
    if not results:
        lines.append("| n/a | n/a | n/a | n/a | 0 | n/a | n/a | n/a | 0 | 0 of 0 | 0 | n/a | n/a |")
        lines += ["", "No captured QPU arms are available for a matched-run-time comparison."]
    return lines


def seed_lane_table(records_by_cell: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    lines = [
        "## Seed lanes", "",
        "Seed lanes from qpu/cpu-lite against cold lanes, both on cpu-msa-f64 at the same sweep depth (Task 8 "
        "brief, step 9). This compares seeding strategies on the CPU kernel, not QPU quality. Every comparison "
        "uses the three separate quality columns (strict, numeric-tolerance, and 0.5% material -- task brief, "
        "step 2), never a bare zero-tolerance count. Failed, unsupported, or nonfinite seeded or cold records "
        "are excluded from Compared and counted under Excluded (task brief, step 5).", "",
        "| Cell | Seed source | Compared | Excluded | Strict seeded/cold/tie | Numeric seeded/cold/tie "
        "| Material seeded/cold/tie |",
        "| -- | -- | -- | -- | -- | -- | -- |",
    ]
    for cell, records in records_by_cell.items():
        by_source: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            by_source.setdefault(record["seed_source"], []).append(record)
        for source, rows in sorted(by_source.items()):
            compared: List[Dict[str, str]] = []
            excluded = 0
            for row in rows:
                seeded = _finite_float(row.get("best_seeded_energy"))
                cold = _finite_float(row.get("best_cold_energy"))
                if not row.get("exit_ok") or row.get("unsupported") or seeded is None or cold is None:
                    excluded += 1
                    continue
                compared.append(metrics.quality_columns(seeded, cold))
            strict = metrics.tally_quality_outcomes([c["strict"] for c in compared])
            numeric = metrics.tally_quality_outcomes([c["numeric_tolerance"] for c in compared])
            material = metrics.tally_quality_outcomes([c["material"] for c in compared])
            lines.append(
                f"| `{cell}` | `{source}` | {len(compared)} of {len(rows)} | {excluded} | "
                f"{_outcome_triple(strict)} | {_outcome_triple(numeric)} | {_outcome_triple(material)} |"
            )
    return lines


def _capitalize_sentences(text: str, separator: str = "; ") -> str:
    """Rejoin ``text``'s ``separator``-joined fragments as separate sentences, each
    capitalized (house style forbids semicolons in prose, STE Rule 8.1). Used
    only to transcribe a quoted upstream field into prose; the source file
    itself (e.g. ``capture-proposal.json``) is never rewritten.
    """
    parts = text.split(separator)
    sentences = [parts[0]] + [part[:1].upper() + part[1:] if part else part for part in parts[1:]]
    return ". ".join(sentences)


def physical_scale_table(
    capture_proposal: Optional[Dict[str, Any]], cells: Sequence[str],
    captured_regimes: Optional[Iterable[str]] = None,
) -> List[str]:
    """The physical-scale pilot's plan, and its status, for every cell.

    ``captured_regimes`` names cells with real capture output on disk (none,
    today: no Round 2 physical capture has run). Approval is not capture
    (review, Important item 8): a cell is never reported "captured" from
    ``capture_proposal["approval"]`` alone, only from ``captured_regimes``
    actually naming it.
    """
    captured = set(captured_regimes or ())
    lines = [
        "## Physical scale", "",
        "The physical-range pilot (12 sorted nonces, 3 scales, 2 anneal times -- the design doc's initial "
        "campaign proposal) prices captures before they run. A cell with no arm here has no physical-scale "
        "plan at all, not merely no result yet.", "",
    ]
    approval = capture_proposal.get("approval") if capture_proposal else None
    scales_note = capture_proposal.get("physical_scales") if capture_proposal else None
    if capture_proposal is not None:
        scales_text = _capitalize_sentences(scales_note or "n/a")
        lines.append(f"Capture proposal approval status: `{approval}`. Planned scales: {scales_text}")
        lines.append("")
    lines += [
        "| Cell | Anneal times planned (\u00b5s) | Captures planned | Reads per capture | Status |",
        "| -- | -- | -- | -- | -- |",
    ]
    arms_by_cell: Dict[str, List[Dict[str, Any]]] = {}
    for arm in (capture_proposal or {}).get("arms", []):
        arms_by_cell.setdefault(arm["regime"], []).append(arm)
    for cell in cells:
        arms = arms_by_cell.get(cell)
        if not arms:
            lines.append(f"| `{cell}` | n/a | n/a | n/a | unavailable -- no physical-scale plan for this cell |")
            continue
        anneals = ", ".join(str(arm["anneal_us"]) for arm in sorted(arms, key=lambda a: a["anneal_us"]))
        captures = sum(arm["captures"] for arm in arms)
        reads = arms[0].get("reads_per_capture", "n/a")
        if cell in captured:
            status = "captured"
        elif approval == "approved":
            status = "approved, not yet captured"
        else:
            status = f"planned, not yet captured (approval: {approval})"
        lines.append(f"| `{cell}` | {anneals} | {captures} | {reads} | {status} |")
    return lines


def feasibility_table(portfolio_records: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "## Portfolio feasibility", "",
        "Feasibility here is the share of raw returned reads that already meet the cardinality constraint "
        "before repair. P's own repair and weighting happen afterward. See the portfolio pipeline table for "
        "the repaired, selected answer's own feasibility. `weighting_failed` is a tri-state value. An unknown "
        "result never counts as success. P's silent equal-weight fallback exposes no flag when it fires. "
        "Provenance names the market-instance source of every row (review, I5): this report never calls a "
        "synthetic instance historical.", "",
        "| Assets | K | Beta label | Provenance | Status | Raw feasible reads | Weighting |",
        "| -- | -- | -- | -- | -- | -- | -- |",
    ]
    for record in portfolio_records:
        count = record.get("raw_feasible_count")
        total = record.get("returned_reads")
        feasibility_text = "n/a" if count is None or not total else metrics.format_feasibility(count, total)
        weighting_failed = record.get("weighting_failed")
        if weighting_failed is True:
            weighting_text = "failed"
        elif weighting_failed is None:
            weighting_text = "unknown (not observed to fail)"
        else:
            weighting_text = "not observed to fail"
        lines.append(
            f"| {record.get('n_assets')} | {record.get('cardinality_k')} | `{record.get('beta_label')}` | "
            f"{record.get('provenance', 'n/a')} | {record.get('status')} | {feasibility_text} | {weighting_text} |"
        )
    return lines


def portfolio_pipeline_table(portfolio_records: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "## Portfolio pipeline", "",
        "The deadline arm, synthetic fixtures, historical settings (dwave-neal, 500 reads / 500 sweeps, the "
        "design's portfolio-replication contract) ran to completion for every basket and beta label.", "",
        "Sampling s covers only the reference sampler's own call. Repair s covers P's separate repair and "
        "weighting step. End-to-end s is their sum, and the Status column classifies on end-to-end time, per "
        "review finding I5, not on sampling time alone. This report has no 10-second kill switch. A slow "
        "repair phase can still turn a fast sample into a late run.", "",
        "Repaired feasible and Selected raw cardinality name the outcome of the one selected, winning read "
        "after P's own repair and weighting. The feasibility table's own raw feasible reads count something "
        "different: every returned read, before repair. Round 2 has not captured a paired QPU portfolio "
        "result. The strict-win, material-win, speed-only, and joint quality/time columns stay unavailable "
        "until it does.", "",
        "| Assets | K | Beta label | Provenance | Status | Sampling s | Repair s | End-to-end s | Deadline s "
        "| Objective | Repaired feasible | Selected raw cardinality | Strict/material win | Speed-only outcome "
        "| Joint quality/time |",
        "| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |",
    ]
    for record in portfolio_records:
        feasible = record.get("feasible")
        feasible_text = "yes" if feasible is True else "no" if feasible is False else "n/a"
        lines.append(
            f"| {record.get('n_assets')} | {record.get('cardinality_k')} | `{record.get('beta_label')}` | "
            f"{record.get('provenance', 'n/a')} | {record.get('status')} | {fmt(record.get('elapsed_s'))} | "
            f"{fmt(record.get('repair_s'))} | {fmt(record.get('end_to_end_s'))} | {fmt(record.get('deadline_s'))} | "
            f"{fmt(record.get('objective'), 6)} | {feasible_text} | {fmt(record.get('selected_raw_cardinality'))} | "
            "unavailable | unavailable | unavailable |"
        )
    return lines


def historical_context_section() -> List[str]:
    return [
        "## Historical context (cited, not recomputed here)", "",
        "The current portfolio manuscript reports 1,930 comparable races, with 833 QPU quality wins and "
        "1,097 ties at 0.5% materiality. At zero tolerance it reports 970 QPU wins and 960 ties, and neither "
        "table contains an SA win. The manuscript reports about 70% faster device access but supplies no joint "
        "quality/time counts, so these figures do not establish an 80% strict-quality win rate (design doc, "
        "`docs/superpowers/specs/2026-09-22-regime-search-round2-design.md`).",
        "The older 2,088-race window overlaps the manuscript's 1,930-race window. This report keeps the two "
        "separate and never pools them, because a union of overlapping windows would double count shared "
        "races.",
        "The 80% portfolio claim stays unresolved. No source used in this report defines that metric together "
        "with the denominator or joint counts a claim at that scale would need.",
    ]


def next_test_section(
    cells: Sequence[str], records_by_cell: Dict[str, List[Dict[str, Any]]],
    capture_proposal: Optional[Dict[str, Any]], run: str,
    captured_regimes: Optional[Iterable[str]] = None,
) -> List[str]:
    """Task brief, step 10: for each regime, what the measurements establish, what
    stays unresolved, and the next control to run. A table, not prose paragraphs,
    because the same three questions repeat for every regime and a table states
    the answers without forcing five near-identical paragraphs on the reader.

    ``run`` names the CPU data source actually loaded (``pilot`` or
    ``campaign``), never a fixed "from the pilot" claim regardless of what ran
    (review, C1). ``captured_regimes`` names cells with a REAL, manifest-
    verified physical-pilot capture on disk; a cell there is never described
    as still awaiting approval.
    """
    captured = set(captured_regimes or ())
    lines = [
        "## What each regime establishes, and what runs next", "",
        "| Regime | Established | Unresolved | Next control |",
        "| -- | -- | -- | -- |",
    ]
    arms_by_cell: Dict[str, List[Dict[str, Any]]] = {}
    for arm in (capture_proposal or {}).get("arms", []):
        arms_by_cell.setdefault(arm["regime"], []).append(arm)
    for cell in cells:
        records = records_by_cell.get(cell, [])
        measured_records = [r for r in records if r.get("requested_kernel") != "cpu-msa"]
        kernels = sorted({
            str(k) for r in measured_records
            if r.get("exit_ok") and not r.get("unsupported")
            and (k := r.get("requested_kernel")) is not None
        })
        completed = sum(1 for r in measured_records if r.get("exit_ok") and not r.get("unsupported"))
        established = (
            f"{completed} completed CPU timing/quality records across {len(kernels)} kernels "
            f"({', '.join(f'`{k}`' for k in kernels) or 'none yet'}) from the {run} run."
        )
        if cell in captured:
            established += " Real, manifest-verified physical-scale QPU captures also exist for this cell."
            if run == "campaign":
                unresolved = (
                    "The 12-model physical-scale pilot pairs against this cell's loaded CPU campaign models "
                    "(see the QPU wins/ties/losses and paired-gaps tables). It earns no regime verdict on its own."
                )
                next_control = (
                    "Review the paired results against the loaded CPU campaign, then decide whether this "
                    "regime's next round needs a wider physical-scale capture."
                )
            else:
                unresolved = (
                    "The 12-model physical-scale pilot pairs against only this cell's own CPU pilot models so "
                    "far (see the QPU wins/ties/losses and paired-gaps tables). It earns no regime verdict on "
                    "its own."
                )
                next_control = (
                    "Extend the paired QPU/CPU comparison to the full CPU campaign once it finishes, and decide "
                    "whether this regime's next round needs a wider physical-scale capture."
                )
        elif cell in arms_by_cell:
            unresolved = (
                "No Round 2 QPU capture exists yet for this cell. The physical-scale pilot (12 sorted "
                "nonces, 3 scales) is planned but not yet captured, so quality and time outcomes against "
                "the QPU stay undefined."
            )
            next_control = (
                "Capture the planned physical-scale pilot, then compare feasibility and quality against "
                "this cell's classical timings. A 12-model pilot never earns a regime verdict on its own."
            )
        else:
            unresolved = (
                "No Round 2 QPU capture exists yet, and this cell has no physical-scale plan at all. The "
                "CPU comparison alone cannot answer the regime question."
            )
            next_control = (
                "The CPU campaign for this cell is complete. Decide whether a QPU arm belongs in this "
                "regime's next round."
            ) if run == "campaign" else (
                "Extend the campaign to the full 100-model comparison for this cell, then decide whether a "
                "QPU arm belongs in this regime's next round."
            )
        lines.append(f"| `{cell}` | {established} | {unresolved} | {next_control} |")
    lines.append(
        "| Portfolio pipeline | The deadline arm (synthetic fixtures, historical settings) ran to "
        "completion for both baskets and both beta labels, producing repaired objectives and raw "
        "feasibility counts. | No paired QPU portfolio result exists, so strict and material wins, "
        "speed-only outcomes, and joint quality/time counts stay unavailable. The 80% portfolio claim "
        "stays unresolved. | Capture the portfolio pilot (12 frozen market instances, beta-zero and "
        "positive-beta controls) once its provenance and anneal-setting classification are explicit, per "
        "the design doc. |"
    )
    return lines


# -------------------------------------------------------------------- figures


def _svg_document(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">'
        f'<rect width="100%" height="100%" fill="white"/>{body}</svg>'
    )


_PALETTE = ("#217a69", "#b8792a", "#445a86", "#a33a3a")


def _text_width_estimate(text: str, font_size: int = 12) -> float:
    """A rough monospace-ish width estimate, just enough to size an SVG canvas so a
    label never runs off the right edge (task brief, step 9: inspect the rendered
    charts for labels, scale consistency, and collisions -- sized here by
    construction, not merely inspected after the fact).
    """
    return len(text) * font_size * 0.6


def _wrap_lines(text: str, max_chars: int) -> List[str]:
    """Word-wrap ``text`` to at most ``max_chars`` per line (task brief, step 9:
    "the caption wraps inside the canvas" -- a review-found defect where one
    long caption line ran past the canvas's right edge).
    """
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _pad_range(low: float, high: float, fraction: float = 0.12) -> Tuple[float, float]:
    """Expand ``[low, high]`` by ``fraction`` on each side so an extreme point
    never sits exactly on an axis line, including the zero-span case of a
    single distinct value (task brief, step 9: "points sit clear of the axis
    lines" -- a review-found defect where a data extreme landed at ``cx ==
    margin``, the y axis itself).
    """
    span = high - low
    pad = span * fraction if span > 0 else max(abs(high), 1.0) * fraction
    return low - pad, high + pad


def _axis_ticks(low: float, high: float, count: int = 5) -> List[float]:
    """``count`` evenly spaced values from ``low`` to ``high``, inclusive of both
    endpoints (fix round 2: "the y axis shows only two end ticks" -- a chart
    with only its extremes labeled cannot show a reader anything about the
    values between them).
    """
    if count < 2 or high <= low:
        return [low, high]
    step = (high - low) / (count - 1)
    return [low + i * step for i in range(count)]


def _tick_text(value: float) -> str:
    """An axis tick's own label: more precision than a table cell's :func:`fmt`,
    so adjacent ticks read as distinct numbers rather than both rounding to
    the same 3-significant-figure text (fix round 2: "-1.43e+04 and
    -1.44e+04" read as barely different at the old precision).
    """
    return f"{value:.6g}"


def _chart_header(title: str, subtitle: str, width: int, left_margin: int = 10) -> Tuple[str, int]:
    """The title and word-wrapped subtitle of a bar chart, plus the y coordinate
    the chart's own content should start at.

    A bar chart's canvas width comes from its bars and value labels, never
    from its subtitle's length, so an unwrapped subtitle can run well past
    the right edge (fix round 2: "the subtitle is clipped at the right
    edge"). Wrapping it against the ALREADY-DECIDED canvas width, and
    returning where the subtitle ends, lets every bar-chart function grow
    its own height to fit rather than guessing a fixed offset.
    """
    parts = [f'<text x="{left_margin}" y="20" font-size="16" font-weight="bold">{title}</text>']
    max_chars = max(20, int((width - 2 * left_margin) / (12 * 0.6)))
    y = 38
    for line in _wrap_lines(subtitle, max_chars):
        parts.append(f'<text x="{left_margin}" y="{y}" font-size="12">{line}</text>')
        y += 15
    return "".join(parts), y + 12


def quality_time_figure(
    cell: str, records: Sequence[Dict[str, Any]], kernels: Sequence[str], depths: Sequence[int],
    *, timing_records: Optional[Sequence[Dict[str, Any]]] = None, timing_source: Optional[str] = None,
) -> str:
    """One quality/time panel per regime: fastest-run median sampling time (x,
    log-scaled) against median best energy (y) for every (kernel, depth) arm.

    Readable as a static image, not only via hover tooltips (review, Important
    item 7): the caption word-wraps inside the canvas, both axes carry visible
    tick values and a title, each point's sample count is printed next to it
    (not only in its ``<title>``), and the data range is padded so no point
    sits exactly on an axis line.

    The plot box's own y range is ``[plot_top, height - margin]``. The title,
    caption, incomplete-arm warning, and legend all sit strictly above
    ``plot_top``, so none of them can ever land on top of a plotted point.
    """
    width, height = 720, 460
    margin = 90
    max_chars = max(20, int((width - 2 * margin) / (12 * 0.6)))

    points: List[Tuple[str, int, float, float, float, int, int]] = []
    incomplete: List[str] = []
    for kernel in kernels:
        for sweeps in depths:
            arm = [r for r in records if r.get("requested_kernel") == kernel and r.get("sweeps") == sweeps]
            summary = metrics.summarize_arm(arm)
            time_records = timing_records if timing_records is not None else records
            time_arm = [r for r in time_records if r.get("requested_kernel") == kernel and r.get("sweeps") == sweeps]
            successful_time_arm = [
                r for r in time_arm
                if r.get("exit_ok") and not r.get("unsupported")
            ]
            sampling_values = [
                float(r["elapsed_sampling_s"])
                for r in successful_time_arm
                if r.get("elapsed_sampling_s") is not None and math.isfinite(float(r["elapsed_sampling_s"]))
            ]
            wall_values = [
                float(r["wall_s"])
                for r in successful_time_arm
                if r.get("wall_s") is not None and math.isfinite(float(r["wall_s"]))
            ]
            wall = float(np.median(wall_values)) if wall_values else None
            if summary["median_best_energy"] is None or not sampling_values:
                reason = (
                    "No timing-subset record"
                    if timing_source == "timing-subset" and not successful_time_arm
                    else "No finite sampling time"
                )
                incomplete.append(f"{kernel}@{sweeps} ({reason})")
                continue
            sampling = float(np.median(sampling_values))
            points.append((
                kernel, sweeps, sampling, float(wall) if wall is not None else float("nan"),
                summary["median_best_energy"], summary["completed"], len(sampling_values),
            ))

    caption = (
        "Energy units: canonical, rescored from the original model, lower is better. The x axis uses "
        f"the median fastest-run elapsed_sampling_s from {timing_source or 'the quality run'}; each job "
        "contributes its fastest successful run. These timings come from a loaded host with parallel workers, "
        "one per physical core. End-to-end wall time remains separate in the table and point details."
    )
    body_parts: List[str] = [f'<text x="{margin}" y="24" font-size="16" font-weight="bold">{cell}: quality vs. time (CPU)</text>']
    y_cursor = 42
    for line in _wrap_lines(caption, max_chars):
        body_parts.append(f'<text x="{margin}" y="{y_cursor}" font-size="12">{line}</text>')
        y_cursor += 15

    plot_top = y_cursor + 2
    if incomplete:
        for line in _wrap_lines(f"Incomplete arms (no data): {', '.join(incomplete)}", max_chars):
            body_parts.append(f'<text x="{margin}" y="{plot_top}" font-size="12" fill="#a33a3a">{line}</text>')
            plot_top += 15

    if not points:
        body_parts.append(
            f'<text x="{margin}" y="{(plot_top + height - margin) / 2:.0f}" font-size="14">No completed arms to plot.</text>'
        )
    else:
        colors = {kernel: _PALETTE[i % len(_PALETTE)] for i, kernel in enumerate(sorted({p[0] for p in points}))}
        # The legend states each kernel's own sample count(s) directly (review,
        # Important item 7: "the sample count is printed visibly" -- not only in
        # a <title> tooltip, which a static render never shows).
        legend_x = margin
        legend_y = plot_top + 4
        for kernel, color in colors.items():
            n_values = sorted({p[5] for p in points if p[0] == kernel})
            label = f"{kernel} (n={','.join(str(n) for n in n_values)})"
            body_parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="12" height="12" fill="{color}"/>')
            body_parts.append(f'<text x="{legend_x + 16}" y="{legend_y + 11}" font-size="12">{label}</text>')
            legend_x += 20 + int(_text_width_estimate(label)) + 16
        plot_top = legend_y + 26

        raw_wall = [p[2] for p in points]
        raw_energy = [p[4] for p in points]
        log_wall = [math.log10(max(w, 1e-9)) for w in raw_wall]
        x_min, x_max = _pad_range(min(log_wall), max(log_wall))
        y_min, y_max = _pad_range(min(raw_energy), max(raw_energy))
        x_span = x_max - x_min
        y_span = y_max - y_min
        plot_height = height - margin - plot_top

        for kernel, sweeps, sampling_s, wall_s, energy, n, timing_n in points:
            x = margin + (math.log10(max(sampling_s, 1e-9)) - x_min) / x_span * (width - 2 * margin)
            y = height - margin - (energy - y_min) / y_span * plot_height
            body_parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{colors[kernel]}" fill-opacity="0.85">'
                f'<title>{kernel} @ {sweeps} sweeps: median sampling {sampling_s:.3g} s, '
                f'median wall {wall_s:.3g} s, median best energy {energy:.6g}, '
                f'quality n={n}, timing n={timing_n}</title></circle>'
                # The sweep count is part of the visible label, not only the kernel's
                # count (fix round 2: "nothing says which depth is which" when a
                # kernel has one point per depth and both only ever said "n=15").
                f'<text x="{x + 9:.1f}" y="{y - 8:.1f}" font-size="9" fill="#333333">{sweeps} sw, n={n}</text>'
            )

        # At least 5 tick values on each axis, not only the two extremes (fix
        # round 2: "the y axis shows only two end ticks ... show at least 3-5
        # ticks, at a precision that separates them"; "the x axis shows only
        # its end values"), plus the axis titles.
        for tick in _axis_ticks(x_min, x_max, count=5):
            tick_x = margin + (tick - x_min) / x_span * (width - 2 * margin)
            body_parts.append(
                f'<text x="{tick_x:.1f}" y="{height - margin + 16}" font-size="10" text-anchor="middle">'
                f'{_tick_text(10 ** tick)}</text>'
            )
        body_parts.append(
            f'<text x="{(margin + width - margin) / 2:.0f}" y="{height - margin + 32}" font-size="11" '
            'text-anchor="middle">Median sampling time (s), log scale</text>'
        )
        for tick in _axis_ticks(y_min, y_max, count=5):
            tick_y = height - margin - (tick - y_min) / y_span * plot_height
            body_parts.append(
                f'<text x="{margin - 6}" y="{tick_y + 3:.1f}" font-size="10" text-anchor="end">{_tick_text(tick)}</text>'
            )
        y_title_pos = (plot_top + height - margin) / 2
        body_parts.append(
            f'<text x="{margin - 55}" y="{y_title_pos:.0f}" font-size="11" text-anchor="middle" '
            f'transform="rotate(-90 {margin - 55} {y_title_pos:.0f})">Median best energy (canonical units)</text>'
        )
    body_parts.append(f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>')
    body_parts.append(f'<line x1="{margin}" y1="{plot_top}" x2="{margin}" y2="{height - margin}" stroke="black"/>')
    return _svg_document(width, height, "".join(body_parts))


def _bar_chart(title: str, subtitle: str, bars: Sequence[Tuple[str, float]], value_label: str) -> str:
    bar_area = 260
    if not bars:
        width = 640
        header, top = _chart_header(title, subtitle, width)
        height = top + 30
        return _svg_document(width, height, header + f'<text x="10" y="{top + 10}" font-size="14">No data available.</text>')
    value_texts = [f"{value:.4g} {value_label}" for _, value in bars]
    # Wide enough that even a zero-width bar's value label -- which starts right
    # after the bar, at ``margin_left`` -- can never land on top of the longest
    # category label (fix round 2: a real render showed "feasible)0 raw
    # feasible reads" fused together when a long category label met a
    # zero-width bar).
    category_width = int(max(_text_width_estimate(label) for label, _ in bars))
    margin_left = max(240, 30 + category_width)
    margin_right = int(max(_text_width_estimate(text) for text in value_texts)) + 20
    width = margin_left + bar_area + margin_right
    header, top = _chart_header(title, subtitle, width)
    height = top + 34 * len(bars) + 20
    max_value = max(abs(value) for _, value in bars) or 1.0
    body_parts = []
    for i, ((label, value), value_text) in enumerate(zip(bars, value_texts)):
        y = top + i * 34
        bar_width = abs(value) / max_value * bar_area
        body_parts.append(f'<text x="10" y="{y + 15}" font-size="12">{label}</text>')
        body_parts.append(f'<rect x="{margin_left}" y="{y}" width="{bar_width:.1f}" height="20" fill="#445a86"/>')
        body_parts.append(f'<text x="{margin_left + bar_width + 6:.1f}" y="{y + 15}" font-size="12">{value_text}</text>')
    return _svg_document(width, height, header + "".join(body_parts))


#: The signed-bar colors (task brief, step 9's graphical-integrity requirement;
#: review, Important item 6: a negative objective must never draw like a
#: positive one just because a magnitude-only bar chart lost its sign).
_POSITIVE_COLOR = "#217a69"
_NEGATIVE_COLOR = "#a33a3a"


def _signed_bar_chart(title: str, subtitle: str, bars: Sequence[Tuple[str, float]], value_label: str) -> str:
    """A bar chart around a zero baseline: a negative value draws to the LEFT of
    the baseline in :data:`_NEGATIVE_COLOR`, a positive one to the right in
    :data:`_POSITIVE_COLOR`, and the printed value always carries an explicit
    sign (``+`` or ``-``). Never draws ``abs(value)`` as an unsigned magnitude.

    ``margin_left`` is sized from the WIDEST category label plus the WIDEST
    negative value label, not a fixed constant: a negative bar's own value
    label sits at its tip, to the left of the zero baseline, and at a large
    enough bar width that label's left edge can reach back past a fixed
    margin into the category column (fix round 2: "a negative-value label
    collides with the category label"). Sizing the margin from both widths
    together guarantees the two can never overlap, at any bar width.
    """
    half_width = 200
    if not bars:
        width = 260 + 2 * half_width + 100
        header, top = _chart_header(title, subtitle, width)
        height = top + 30
        return _svg_document(width, height, header + f'<text x="10" y="{top + 10}" font-size="14">No data available.</text>')
    value_texts = [f"{value:+.4g} {value_label}" for _, value in bars]
    category_width = int(max(_text_width_estimate(label) for label, _ in bars))
    negative_value_width = int(max(
        (_text_width_estimate(text) for (_, value), text in zip(bars, value_texts) if value < 0), default=0,
    ))
    margin_left = max(120, 40 + category_width + negative_value_width)
    positive_value_width = int(max(
        (_text_width_estimate(text) for (_, value), text in zip(bars, value_texts) if value >= 0), default=0,
    ))
    label_margin = positive_value_width + 20
    width = margin_left + 2 * half_width + label_margin
    header, top = _chart_header(title, subtitle, width)
    height = top + 34 * len(bars) + 20
    max_abs = max(abs(value) for _, value in bars) or 1.0
    baseline_x = margin_left + half_width
    body_parts = [
        f'<line x1="{baseline_x}" y1="{top - 10}" x2="{baseline_x}" y2="{top + 34 * len(bars) - 4}" '
        'stroke="#999999" stroke-dasharray="3,3"/>',
        f'<text x="{baseline_x}" y="{top - 14}" font-size="10" text-anchor="middle">0</text>',
    ]
    for i, ((label, value), value_text) in enumerate(zip(bars, value_texts)):
        y = top + i * 34
        bar_width = abs(value) / max_abs * half_width
        color = _POSITIVE_COLOR if value >= 0 else _NEGATIVE_COLOR
        x = baseline_x if value >= 0 else baseline_x - bar_width
        text_x = baseline_x + bar_width + 6 if value >= 0 else baseline_x - bar_width - 6
        anchor = "start" if value >= 0 else "end"
        body_parts.append(f'<text x="10" y="{y + 15}" font-size="12">{label}</text>')
        body_parts.append(f'<rect x="{x:.1f}" y="{y}" width="{bar_width:.1f}" height="20" fill="{color}"/>')
        body_parts.append(
            f'<text x="{text_x:.1f}" y="{y + 15}" font-size="12" text-anchor="{anchor}">{value_text}</text>'
        )
    return _svg_document(width, height, header + "".join(body_parts))


def physical_scale_figure(
    capture_proposal: Optional[Dict[str, Any]], captured_regimes: Optional[Iterable[str]] = None,
) -> str:
    """The physical-scale PLAN panel for diamond and clique (task brief, step 7):
    the planned capture counts. See :func:`physical_scale_effect_figure` for
    what was actually measured once a capture exists.

    ``captured_regimes`` names cells with a real, manifest-verified capture on
    disk. The subtitle's own capture-status sentence is never a fixed claim
    (review, C1, and a render-inspection defect from this final fix pass:
    the old fixed "No physical capture has run yet" stayed in the SVG even
    after real captures existed for every planned cell).
    """
    arms = (capture_proposal or {}).get("arms", [])
    bars = [(f"{arm['regime']} @ {arm['anneal_us']} us", float(arm["captures"])) for arm in arms]
    approval = (capture_proposal or {}).get("approval", "no plan on file")
    planned_cells = sorted({arm["regime"] for arm in arms})
    captured = set(captured_regimes or ())
    if not planned_cells:
        capture_status = "No physical-scale plan is on file."
    elif captured.issuperset(planned_cells):
        capture_status = "Every planned cell has a real, manifest-verified capture."
    elif captured:
        capture_status = f"Captured so far: {', '.join(sorted(captured))}."
    else:
        capture_status = "No physical capture has run yet."
    subtitle = (
        f"Planned captures per arm (diamond and clique only). Approval status: {approval}. {capture_status}"
    )
    return _bar_chart("Physical-scale pilot plan (diamond, clique)", subtitle, bars, "captures planned")


def physical_scale_effect_figure(cell: str, captures: Sequence[Dict[str, Any]]) -> str:
    """The physical-scale EFFECT panel (task brief, step 7; review, C1): each
    captured model's own best energy at each requested scale, colored by
    anneal time. Per-model points, exactly the 12-model pilot's own data --
    this never averages across models into a single line, and it assigns no
    regime verdict (the design doc: "do not extrapolate 12 models into a
    regime verdict").
    """
    width, height = 640, 420
    margin = 90
    title = f'<text x="{margin}" y="24" font-size="16" font-weight="bold">{cell}: physical-scale effect (QPU)</text>'
    caption = (
        "Energy units: canonical. Each point is one captured model at one requested scale and anneal time. "
        "The 12-model pilot supports no regime verdict on its own."
    )
    max_chars = max(20, int((width - 2 * margin) / (12 * 0.6)))
    body_parts = [title]
    y_cursor = 42
    for line in _wrap_lines(caption, max_chars):
        body_parts.append(f'<text x="{margin}" y="{y_cursor}" font-size="12">{line}</text>')
        y_cursor += 15
    plot_top = y_cursor + 6

    if not captures:
        body_parts.append(
            f'<text x="{margin}" y="{(plot_top + height - margin) / 2:.0f}" font-size="14">'
            "No physical-scale captures for this cell.</text>"
        )
        return _svg_document(width, height, "".join(body_parts))

    anneals = sorted({c["anneal_us"] for c in captures})
    colors = {anneal: _PALETTE[i % len(_PALETTE)] for i, anneal in enumerate(anneals)}
    legend_x = margin
    legend_y = plot_top
    for anneal, color in colors.items():
        label = f"{anneal} us"
        body_parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="12" height="12" fill="{color}"/>')
        body_parts.append(f'<text x="{legend_x + 16}" y="{legend_y + 11}" font-size="12">{label}</text>')
        legend_x += 20 + int(_text_width_estimate(label)) + 16
    plot_top = legend_y + 26

    scales = [c["requested_scale"] for c in captures]
    energies = [c["best_energy"] for c in captures]
    x_min, x_max = _pad_range(min(scales), max(scales))
    y_min, y_max = _pad_range(min(energies), max(energies))
    x_span = x_max - x_min
    y_span = y_max - y_min
    plot_height = height - margin - plot_top
    for capture in captures:
        x = margin + (capture["requested_scale"] - x_min) / x_span * (width - 2 * margin)
        y = height - margin - (capture["best_energy"] - y_min) / y_span * plot_height
        color = colors[capture["anneal_us"]]
        body_parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" fill-opacity="0.85">'
            f'<title>model {capture["model_hash"][:8]}, scale {capture["requested_scale"]:g}, '
            f'{capture["anneal_us"]} us: best energy {capture["best_energy"]:.6g}</title></circle>'
        )
    for tick in _axis_ticks(x_min, x_max, count=5):
        tick_x = margin + (tick - x_min) / x_span * (width - 2 * margin)
        body_parts.append(
            f'<text x="{tick_x:.1f}" y="{height - margin + 16}" font-size="10" text-anchor="middle">'
            f'{_tick_text(tick)}</text>'
        )
    body_parts.append(
        f'<text x="{(margin + width - margin) / 2:.0f}" y="{height - margin + 32}" font-size="11" '
        'text-anchor="middle">Requested scale (share of the audited legal ceiling)</text>'
    )
    for tick in _axis_ticks(y_min, y_max, count=5):
        tick_y = height - margin - (tick - y_min) / y_span * plot_height
        body_parts.append(
            f'<text x="{margin - 6}" y="{tick_y + 3:.1f}" font-size="10" text-anchor="end">{_tick_text(tick)}</text>'
        )
    y_title_pos = (plot_top + height - margin) / 2
    body_parts.append(
        f'<text x="{margin - 55}" y="{y_title_pos:.0f}" font-size="11" text-anchor="middle" '
        f'transform="rotate(-90 {margin - 55} {y_title_pos:.0f})">Best energy (canonical units)</text>'
    )
    body_parts.append(f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>')
    body_parts.append(f'<line x1="{margin}" y1="{plot_top}" x2="{margin}" y2="{height - margin}" stroke="black"/>')
    return _svg_document(width, height, "".join(body_parts))


def portfolio_figure(portfolio_records: Sequence[Dict[str, Any]]) -> str:
    """The portfolio quality panel (task brief, step 7): the repaired final
    objective per basket x beta-label arm, in the portfolio pipeline's own final
    objective units -- not spin-model energy units. Signed around zero (review,
    Important item 6): a negative objective never draws as if it were positive.
    """
    bars = [
        (f"n={record.get('n_assets')} k={record.get('cardinality_k')} {record.get('beta_label')}", float(record["objective"]))
        for record in portfolio_records if record.get("objective") is not None
    ]
    subtitle = (
        "Deadline arm, synthetic fixtures, historical settings (dwave-neal, 500 reads / 500 sweeps), repaired "
        "final objective, original "
        "units. No paired QPU portfolio result yet."
    )
    return _signed_bar_chart("Portfolio pipeline: repaired objective per basket", subtitle, bars, "objective")


def portfolio_repair_figure(portfolio_records: Sequence[Dict[str, Any]]) -> str:
    """The portfolio repair panel (task brief, step 7): raw feasible reads (exact
    cardinality, before repair) per basket, labeled with whether P's repaired,
    selected answer ended feasible (review, Important item 6: "add a repair
    panel that shows raw feasible against repaired counts").
    """
    bars = []
    for record in portfolio_records:
        count = record.get("raw_feasible_count")
        total = record.get("returned_reads")
        if count is None or not total:
            continue
        repaired = "repaired: feasible" if record.get("feasible") else "repaired: infeasible"
        label = f"n={record.get('n_assets')} k={record.get('cardinality_k')} {record.get('beta_label')} ({repaired})"
        bars.append((label, float(count)))
    subtitle = (
        "Raw feasible reads (exact cardinality, before repair) per basket, out of returned_reads (see the "
        "feasibility table for the denominator). The label states the repaired, selected answer's own "
        "feasibility."
    )
    return _bar_chart("Portfolio repair: raw feasible reads vs. the repaired answer", subtitle, bars, "raw feasible reads")


def write_figures(
    out_dir: Path, records_by_cell: Dict[str, List[Dict[str, Any]]], kernels: Sequence[str], depths: Sequence[int],
    capture_proposal: Optional[Dict[str, Any]], portfolio_records: Sequence[Dict[str, Any]],
    captures_by_arm: Optional[Dict[Tuple[str, float, int], List[Dict[str, Any]]]] = None,
    timing_records_by_cell: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    timing_source: Optional[str] = None,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for cell, records in records_by_cell.items():
        path = out_dir / f"quality-time-{cell}.svg"
        path.write_text(
            quality_time_figure(
                cell, records, kernels, depths,
                timing_records=(timing_records_by_cell or {}).get(cell), timing_source=timing_source,
            ),
            encoding="utf-8",
        )
        paths.append(path)
    captured_regimes = {cell for cell, _scale, _anneal in (captures_by_arm or {})}
    physical_path = out_dir / "physical-scale-plan.svg"
    physical_path.write_text(physical_scale_figure(capture_proposal, captured_regimes), encoding="utf-8")
    paths.append(physical_path)
    # One physical-scale EFFECT panel per cell that has a real capture (review,
    # C1): the planned-count panel above states what was proposed; this one
    # shows what was actually measured, per model, never averaged into a verdict.
    captures_by_cell: Dict[str, List[Dict[str, Any]]] = {}
    for (cell, _scale, _anneal), captures in (captures_by_arm or {}).items():
        captures_by_cell.setdefault(cell, []).extend(captures)
    for cell, captures in captures_by_cell.items():
        effect_path = out_dir / f"physical-scale-effect-{cell}.svg"
        effect_path.write_text(physical_scale_effect_figure(cell, captures), encoding="utf-8")
        paths.append(effect_path)
    portfolio_path = out_dir / "portfolio-objective.svg"
    portfolio_path.write_text(portfolio_figure(portfolio_records), encoding="utf-8")
    paths.append(portfolio_path)
    repair_path = out_dir / "portfolio-repair.svg"
    repair_path.write_text(portfolio_repair_figure(portfolio_records), encoding="utf-8")
    paths.append(repair_path)
    return paths


# ---------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--cpu-root", required=True, help="Root of the CPU comparison output (contains pilot/, campaign/).")
    parser.add_argument("--run", default="pilot", choices=("pilot", "campaign"))
    parser.add_argument(
        "--captured-root", default=None,
        help="Root containing captured/ CPU records for the matched-runtime and captured Neal comparisons.",
    )
    parser.add_argument(
        "--timing-subset-root", default=None,
        help="Timing source for campaign quality reports; defaults to cpu-root/timing-subset.",
    )
    parser.add_argument("--cells", nargs="+", default=list(regimes.CELL_NAMES))
    parser.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS))
    parser.add_argument(
        "--depths", type=int, nargs="+", default=None,
        help="Defaults to the pilot depths for --run pilot, the full ladder otherwise.",
    )
    parser.add_argument("--seeded-root", default=None, help="cpu-root's seeded-sweep-v2 directory; omit to skip the seed-lane table.")
    parser.add_argument("--portfolio-results", default=None, help="portfolio-deadline/results.json; omit to skip the portfolio tables.")
    parser.add_argument("--capture-proposal", default=None, help="capture-proposal.json; omit to mark physical scale unavailable.")
    parser.add_argument(
        "--qpu-root", default=None,
        help="qpu-physical-pilot/ (real captures); requires --physical-capture-manifest too.",
    )
    parser.add_argument(
        "--physical-capture-manifest", default=None,
        help="physical-capture-manifest.json, checked against every --qpu-root capture.",
    )
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.qpu_root is not None) != (args.physical_capture_manifest is not None):
        parser.error("--qpu-root and --physical-capture-manifest must be supplied together")
    cpu_root = Path(args.cpu_root)
    run_dir = cpu_root / args.run
    depths = args.depths or (list(PILOT_DEPTHS) if args.run == "pilot" else list(FULL_DEPTHS))

    records_by_cell = {cell: load_cpu_records(run_dir, cell) for cell in args.cells}
    for cell, records in records_by_cell.items():
        records.extend(derive_cpu_msa_records(records))
    timing_source = "timing-subset" if args.run == "campaign" else None
    timing_records_by_cell = (
        {
            cell: load_cpu_records(
                Path(args.timing_subset_root) if args.timing_subset_root else cpu_root / "timing-subset", cell,
            )
            for cell in args.cells
        }
        if timing_source else None
    )
    if timing_records_by_cell is not None:
        for cell, records in timing_records_by_cell.items():
            records.extend(derive_cpu_msa_records(records))
    seeded_by_cell = (
        {cell: load_seeded_sweep_records(Path(args.seeded_root), cell) for cell in args.cells}
        if args.seeded_root else {}
    )
    portfolio_records = load_portfolio_deadline(Path(args.portfolio_results)) if args.portfolio_results else []
    capture_proposal = load_capture_proposal(Path(args.capture_proposal)) if args.capture_proposal else None
    expected_per_arm = expected_per_arm_for(args.run)

    captured_records_by_cell = (
        {
            cell: load_cpu_records(Path(args.captured_root) / "captured", cell)
            for cell in args.cells
        }
        if args.captured_root else {}
    )
    for cell, records in captured_records_by_cell.items():
        records.extend(derive_cpu_msa_records(records))
    qpu_records_by_cell = {
        cell: [
            record for record in records
            if not captured_records_by_cell or record.get("requested_kernel") != runner.NEAL_KERNEL
        ]
        for cell, records in records_by_cell.items()
    }
    qpu_kernels = list(args.kernels)
    if captured_records_by_cell:
        if runner.NEAL_KERNEL not in qpu_kernels:
            qpu_kernels.append(runner.NEAL_KERNEL)
        for cell, records in captured_records_by_cell.items():
            qpu_records_by_cell[cell].extend(
                record for record in records if record.get("requested_kernel") == runner.NEAL_KERNEL
            )
    matched_kernels = list(args.kernels)
    if captured_records_by_cell and runner.NEAL_KERNEL not in matched_kernels:
        matched_kernels.append(runner.NEAL_KERNEL)

    captures: List[Dict[str, Any]] = []
    captures_by_arm: Dict[Tuple[str, float, int], List[Dict[str, Any]]] = {}
    spend_summary: Optional[Dict[str, Any]] = None
    if args.qpu_root and args.physical_capture_manifest:
        manifest = load_physical_capture_manifest(Path(args.physical_capture_manifest))
        captures = load_qpu_captures(Path(args.qpu_root), manifest)
        captures_by_arm = group_captures_by_arm(captures)
        spend_summary = reconcile_spend(Path(args.qpu_root), captures)
    captured_regimes = {cell for cell, _scale, _anneal in captures_by_arm}

    if captures:
        qpu_status = (
            f"Round 2's physical-scale QPU pilot has {len(captures)} real, manifest-verified captures across "
            f"{len(captured_regimes)} cells: {', '.join(sorted(captured_regimes))}. See the QPU wins/ties/"
            "losses, paired-gaps, and physical-scale tables below."
        )
    else:
        qpu_status = (
            "No Round 2 QPU capture is loaded for this report run (pass --qpu-root and "
            "--physical-capture-manifest to load one)."
        )

    lines = [
        "# Round 2 regime search report", "",
        f"Status: draft, {args.run} data, incomplete. This report draws on `{args.run}` CPU records under "
        f"`{cpu_root}`. {qpu_status} This draft states no regime verdict.",
        "",
    ]
    lines += model_identity_table(records_by_cell) + [""]
    lines += solver_settings_table(records_by_cell) + [""]
    lines += depth_quality_time_table(
        records_by_cell, kernels=args.kernels, depths=depths, expected_per_arm=expected_per_arm,
        timing_records_by_cell=timing_records_by_cell, timing_source=timing_source,
    ) + [""]
    diversity_records_by_cell = {
        cell: records + seeded_by_cell.get(cell, []) for cell, records in records_by_cell.items()
    }
    lines += lane_diversity_table(diversity_records_by_cell) + [""]
    lines += cpu_kernel_gap_table(records_by_cell, kernels=args.kernels, depths=depths) + [""]
    lines += qpu_outcome_table(args.cells, captures_by_arm, qpu_records_by_cell, qpu_kernels, depths) + [""]
    lines += paired_gaps_table(args.cells, captures_by_arm, qpu_records_by_cell, qpu_kernels, depths) + [""]
    matched_records_by_cell = captured_records_by_cell if captured_records_by_cell else records_by_cell
    captured_records = [record for records in captured_records_by_cell.values() for record in records]
    captured_model_count = len({
        record["model_hash"] for record in captured_records if isinstance(record.get("model_hash"), str)
    })
    captured_worker_counts = sorted({
        record["concurrent_workers"] for record in captured_records
        if isinstance(record.get("concurrent_workers"), int)
    })
    if len(captured_worker_counts) == 1:
        captured_worker_label = f"{captured_worker_counts[0]} worker"
    elif captured_worker_counts:
        captured_worker_label = f"{captured_worker_counts[0]} to {captured_worker_counts[-1]} workers"
    else:
        captured_worker_label = "worker count unavailable"
    matched_description = (
        "QPU time is charged access time for 64 reads, with end-to-end time also shown. CPU energy and time "
        f"come from the captured run: {captured_model_count} distinct model"
        f"{'s' if captured_model_count != 1 else ''}, {captured_worker_label}. Energy "
        "outcomes use the same strict, numeric-tolerance, and material rules as the equal-sweep tables. Equal "
        "budget uses the deepest CPU depth completed within that capture's access-time budget. Time to QPU "
        "energy uses the quickest CPU depth that reached the capture's best energy."
        if captured_records_by_cell else None
    )
    lines += matched_runtime_table(
        captures, [record for records in matched_records_by_cell.values() for record in records], matched_kernels,
        source_description=matched_description,
    ) + [""]
    if seeded_by_cell:
        lines += seed_lane_table(seeded_by_cell) + [""]
    lines += physical_scale_table(capture_proposal, args.cells, captured_regimes=captured_regimes) + [""]
    if spend_summary is not None:
        lines += [
            "## Physical-pilot spend reconciliation", "",
            f"Submits: {spend_summary['submits']}. Charges: {spend_summary['charges']}. Open jobs (submitted, "
            f"never charged): {spend_summary['open_jobs']}. Charged total: {spend_summary['charged_us']} us. "
            f"Sum of every capture's own access_us: {spend_summary['total_access_us']} us. The totals "
            f"{'reconcile' if spend_summary['reconciled'] else 'do not reconcile'}.",
            "",
        ]
    if portfolio_records:
        lines += feasibility_table(portfolio_records) + [""]
        lines += portfolio_pipeline_table(portfolio_records) + [""]
    lines += historical_context_section() + [""]
    lines += next_test_section(args.cells, records_by_cell, capture_proposal, args.run, captured_regimes) + [""]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    write_figures(
        out_dir, records_by_cell, args.kernels, depths, capture_proposal, portfolio_records, captures_by_arm,
        timing_records_by_cell=timing_records_by_cell, timing_source=timing_source,
    )
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
