#!/usr/bin/env python3
"""Round 2 regime search report: per-regime tables, the portfolio pipeline's
own tables, and next-test decisions, built from CPU pilot/campaign records,
the seeded/cold weighted-MSA control, and the portfolio historical-deadline
arm's results. See ``round2_metrics.py`` for the pure comparison, feasibility,
and bootstrap primitives this module assembles into a document.

Task 10 of D's ``docs/superpowers/plans/2026-09-22-regime-search-round2.md``
is the brief this implements. D's own ``scripts/regime_report.py``
feasibility-rounding fix is out of scope here (D is read-only): this report
carries its own count-based feasibility formatter
(:func:`round2_metrics.format_feasibility`).

No QPU submission happens here, and as of this report's own generation none
of Round 2's own QPU captures have run yet -- the physical-range pilot (12
sorted diamond/clique nonces, 3 scales, 2 anneal times) and the portfolio
pilot both still show ``"approval": "pending"`` in
``/home/carback1/quip-data/regimes/round2/capture-proposal.json``. Every
QPU-paired table below is therefore explicitly labeled unavailable rather
than guessed at or silently left blank. Figures render as small,
dependency-free SVG documents: no plotting library is installed in D's
pinned venv (numpy and scipy are; matplotlib is not), and this report must
run entirely under that venv.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from quip_miner_dwave import regimes

import round2_metrics as metrics
import round2_runner as runner

DEFAULT_KERNELS: Tuple[str, ...] = ("cpu-sa", "cpu-msa-f64", "cpu-msa-unit")
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
    """Every job's LATEST attempt record for one cell of a CPU run directory.

    Only files named ``*__attempt<N>.json`` count: a bare ``<base>.json`` with
    no attempt suffix predates the attempt-numbering fix and is fully
    contaminated (controller ruling on this task), so it is ignored outright
    rather than silently mixed in.
    """
    cell_dir = Path(run_dir) / cell
    if not cell_dir.exists():
        return []
    latest: Dict[str, Tuple[int, Path]] = {}
    for path in cell_dir.glob("*__attempt*.json"):
        match = _ATTEMPT_RE.match(path.name)
        if match is None:
            continue
        base = match.group("base")
        attempt = int(match.group("attempt"))
        if base not in latest or attempt > latest[base][0]:
            latest[base] = (attempt, path)
    return [json.loads(path.read_text(encoding="utf-8")) for _, path in latest.values()]


def load_seeded_sweep_records(seeded_root: Path, cell: str) -> List[Dict[str, Any]]:
    """Every seeded/cold weighted-MSA record for one cell (Task 8 brief, step 9)."""
    cell_dir = Path(seeded_root) / cell
    if not cell_dir.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(cell_dir.glob("*.json"))]


def load_portfolio_deadline(path: Path) -> List[Dict[str, Any]]:
    """The historical-deadline arm's own ``results.json``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload.get("records", []))


def load_capture_proposal(path: Path) -> Dict[str, Any]:
    """The physical-range/portfolio capture proposal, whatever its approval status."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
        kernels = sorted({str(k) for r in records if (k := r.get("requested_kernel")) is not None})
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
) -> List[str]:
    lines = [
        "## Depth, quality, and time", "",
        "Every row is one (cell, kernel, sweep depth) arm. Missing, failed, unsupported, and nonfinite "
        "observations stay outside the comparable denominator (task brief, step 5). Only the completed, "
        "finite subset feeds Median best energy, which counts every `exit_ok` record no matter the host "
        "condition. A completed energy result is equally valid on a clean host or a busy one. Wall time "
        "gets separate treatment. A contaminated run or a parallel-worker run measures a different host "
        "condition than a clean serial run, so this table always reports their medians apart, as "
        "`median (n)` for each label, and never pools them into one number (review, Important item 1).", "",
        "| Cell | Kernel | Sweeps | Observed | Missing | Completed | Failed | Unsupported | Nonfinite "
        "| Median best energy | Wall s, clean serial (n) | Wall s, contaminated (n) | Wall s, parallel (n) |",
        "| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |",
    ]
    for cell, records in records_by_cell.items():
        for kernel in kernels:
            for sweeps in depths:
                arm = [r for r in records if r.get("requested_kernel") == kernel and r.get("sweeps") == sweeps]
                summary = metrics.summarize_arm(arm, expected=expected_per_arm)
                lines.append(
                    f"| `{cell}` | `{kernel}` | {sweeps} | {summary['observed']} | {fmt(summary['missing'])} | "
                    f"{summary['completed']} | {summary['failed']} | {summary['unsupported']} | "
                    f"{summary['nonfinite']} | {fmt(summary['median_best_energy'])} | "
                    f"{_timing_cell(summary, metrics.CLEAN_SERIAL)} | "
                    f"{_timing_cell(summary, metrics.CONTAMINATED)} | "
                    f"{_timing_cell(summary, metrics.PARALLEL)} |"
                )
    return lines


def qpu_outcome_table(cells: Sequence[str]) -> List[str]:
    lines = [
        "## Quantum processing unit wins, ties, and losses", "",
        "No Round 2 QPU capture has run yet: the physical-range and portfolio pilots are still pending "
        "approval (see the physical scale table). Every cell is unavailable until a capture exists to pair "
        "against these CPU arms.", "",
        "| Cell | Status |", "| -- | -- |",
    ]
    for cell in cells:
        lines.append(f"| `{cell}` | unavailable -- no Round 2 QPU capture yet |")
    return lines


def paired_gaps_table(cells: Sequence[str]) -> List[str]:
    lines = [
        "## Paired gaps", "",
        "Bootstrap paired-gap intervals (round2_metrics.bootstrap_paired_gap: 10,000 resamples, a fixed "
        "recorded seed, grouped by model or by market snapshot) need a QPU result to pair against a CPU one. "
        "None exists yet for Round 2.", "",
        "| Cell | Status |", "| -- | -- |",
    ]
    for cell in cells:
        lines.append(f"| `{cell}` | unavailable -- no paired QPU/CPU comparison yet |")
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
        "result never counts as success. P's silent equal-weight fallback exposes no flag when it fires.", "",
        "| Assets | K | Beta label | Status | Raw feasible reads | Weighting |",
        "| -- | -- | -- | -- | -- | -- |",
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
            f"{record.get('status')} | {feasibility_text} | {weighting_text} |"
        )
    return lines


def portfolio_pipeline_table(portfolio_records: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "## Portfolio pipeline", "",
        "The historical-deadline arm (dwave-neal, 500 reads / 500 sweeps, the design's portfolio-replication "
        "contract) ran to completion. This report classifies each run from its measured elapsed time. It "
        "never kills a run at 10 seconds, so a late-but-good answer shows as a timeout, not a win. "
        "Repaired feasible and Selected raw cardinality report P's own repair and weighting outcome for the "
        "one selected, winning read. That differs from the feasibility table's raw feasible reads, which "
        "count over every returned read. The strict-win, material-win, speed-only, and joint quality/time "
        "columns each need a paired QPU portfolio result, which Round 2 has not captured yet.", "",
        "| Assets | K | Beta label | Status | Elapsed s | Deadline s | Objective | Repaired feasible "
        "| Selected raw cardinality | Strict/material win | Speed-only outcome | Joint quality/time |",
        "| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |",
    ]
    for record in portfolio_records:
        feasible = record.get("feasible")
        feasible_text = "yes" if feasible is True else "no" if feasible is False else "n/a"
        lines.append(
            f"| {record.get('n_assets')} | {record.get('cardinality_k')} | `{record.get('beta_label')}` | "
            f"{record.get('status')} | {fmt(record.get('elapsed_s'))} | {fmt(record.get('deadline_s'))} | "
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
    capture_proposal: Optional[Dict[str, Any]],
) -> List[str]:
    """Task brief, step 10: for each regime, what the measurements establish, what
    stays unresolved, and the next control to run. A table, not prose paragraphs,
    because the same three questions repeat for every regime and a table states
    the answers without forcing five near-identical paragraphs on the reader.
    """
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
        kernels = sorted({str(k) for r in records if (k := r.get("requested_kernel")) is not None})
        completed = sum(1 for r in records if r.get("exit_ok") and not r.get("unsupported"))
        established = (
            f"{completed} completed CPU timing/quality records across {len(kernels)} kernels "
            f"({', '.join(f'`{k}`' for k in kernels) or 'none yet'}) from the pilot."
        )
        if cell in arms_by_cell:
            unresolved = (
                "No Round 2 QPU capture exists yet. The physical-scale pilot (12 sorted nonces, 3 scales) "
                "awaits approval, so quality and time outcomes against the QPU stay undefined."
            )
            next_control = (
                "Run the approved physical-scale capture, then compare feasibility and quality against this "
                "pilot's classical timings. A 12-model pilot never earns a regime verdict on its own."
            )
        else:
            unresolved = (
                "No Round 2 QPU capture exists yet, and this cell has no physical-scale plan at all. The CPU "
                "comparison alone cannot answer the regime question."
            )
            next_control = (
                "Extend the campaign to the full 100-model comparison for this cell, then decide whether a "
                "QPU arm belongs in this regime's next round."
            )
        lines.append(f"| `{cell}` | {established} | {unresolved} | {next_control} |")
    lines.append(
        "| Portfolio pipeline | The historical-deadline arm ran to completion for both baskets and both beta "
        "labels, producing repaired objectives and raw feasibility counts. | No paired QPU portfolio result "
        "exists, so strict and material wins, speed-only outcomes, and joint quality/time counts stay "
        "unavailable. The 80% portfolio claim stays unresolved. | Capture the portfolio pilot (12 frozen "
        "market instances, beta-zero and positive-beta controls) once its provenance and anneal-setting "
        "classification are explicit, per the design doc. |"
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


def quality_time_figure(
    cell: str, records: Sequence[Dict[str, Any]], kernels: Sequence[str], depths: Sequence[int],
) -> str:
    """One quality/time panel per regime (task brief, step 7): median wall time
    (x, log-scaled) against median best energy (y) for every (kernel, depth) arm.

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

    points: List[Tuple[str, int, float, float, int]] = []
    incomplete: List[str] = []
    for kernel in kernels:
        for sweeps in depths:
            arm = [r for r in records if r.get("requested_kernel") == kernel and r.get("sweeps") == sweeps]
            summary = metrics.summarize_arm(arm)
            if summary["median_best_energy"] is None or summary["median_wall_s"] is None:
                incomplete.append(f"{kernel}@{sweeps}")
                continue
            points.append((kernel, sweeps, summary["median_wall_s"], summary["median_best_energy"], summary["completed"]))

    caption = (
        "Energy units: canonical, rescored from the original model, lower is better. Wall time is the "
        "clean-serial median only. Contaminated and parallel-worker runs are reported separately in the "
        "depth table, never pooled here. No application deadline applies to this arm."
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
            n_values = sorted({p[4] for p in points if p[0] == kernel})
            label = f"{kernel} (n={','.join(str(n) for n in n_values)})"
            body_parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="12" height="12" fill="{color}"/>')
            body_parts.append(f'<text x="{legend_x + 16}" y="{legend_y + 11}" font-size="12">{label}</text>')
            legend_x += 20 + int(_text_width_estimate(label)) + 16
        plot_top = legend_y + 26

        raw_wall = [p[2] for p in points]
        raw_energy = [p[3] for p in points]
        log_wall = [math.log10(max(w, 1e-9)) for w in raw_wall]
        x_min, x_max = _pad_range(min(log_wall), max(log_wall))
        y_min, y_max = _pad_range(min(raw_energy), max(raw_energy))
        x_span = x_max - x_min
        y_span = y_max - y_min
        plot_height = height - margin - plot_top

        for kernel, sweeps, wall_s, energy, n in points:
            x = margin + (math.log10(max(wall_s, 1e-9)) - x_min) / x_span * (width - 2 * margin)
            y = height - margin - (energy - y_min) / y_span * plot_height
            body_parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{colors[kernel]}" fill-opacity="0.85">'
                f'<title>{kernel} @ {sweeps} sweeps: median wall {wall_s:.3g} s, '
                f'median best energy {energy:.6g}, n={n}</title></circle>'
                f'<text x="{x + 9:.1f}" y="{y - 8:.1f}" font-size="9" fill="#333333">n={n}</text>'
            )

        # Tick values (the raw wall-time and energy extremes) and axis titles,
        # both visible directly on the image, not only in the caption's prose.
        body_parts.append(f'<text x="{margin}" y="{height - margin + 16}" font-size="10">{fmt(min(raw_wall))}</text>')
        body_parts.append(
            f'<text x="{width - margin}" y="{height - margin + 16}" font-size="10" text-anchor="end">'
            f'{fmt(max(raw_wall))}</text>'
        )
        body_parts.append(
            f'<text x="{(margin + width - margin) / 2:.0f}" y="{height - margin + 32}" font-size="11" '
            'text-anchor="middle">Median wall time (s), log scale</text>'
        )
        body_parts.append(
            f'<text x="{margin - 6}" y="{plot_top + 4}" font-size="10" text-anchor="end">{fmt(max(raw_energy))}</text>'
        )
        body_parts.append(
            f'<text x="{margin - 6}" y="{height - margin}" font-size="10" text-anchor="end">{fmt(min(raw_energy))}</text>'
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
    margin_left = 240
    top = 70
    height = 90 + 34 * max(len(bars), 1)
    header = (
        f'<text x="10" y="20" font-size="16" font-weight="bold">{title}</text>'
        f'<text x="10" y="40" font-size="12">{subtitle}</text>'
    )
    if not bars:
        width = 640
        return _svg_document(width, height, header + f'<text x="10" y="{top + 20}" font-size="14">No data available.</text>')
    value_texts = [f"{value:.4g} {value_label}" for _, value in bars]
    margin_right = int(max(_text_width_estimate(text) for text in value_texts)) + 20
    width = margin_left + bar_area + margin_right
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
    """
    half_width = 200
    margin_left = 260
    top = 70
    height = 90 + 34 * max(len(bars), 1)
    header = (
        f'<text x="10" y="20" font-size="16" font-weight="bold">{title}</text>'
        f'<text x="10" y="40" font-size="12">{subtitle}</text>'
    )
    if not bars:
        width = margin_left + 2 * half_width + 100
        return _svg_document(width, height, header + f'<text x="10" y="{top + 20}" font-size="14">No data available.</text>')
    value_texts = [f"{value:+.4g} {value_label}" for _, value in bars]
    label_margin = int(max(_text_width_estimate(text) for text in value_texts)) + 20
    width = margin_left + 2 * half_width + label_margin
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


def physical_scale_figure(capture_proposal: Optional[Dict[str, Any]]) -> str:
    """The physical-scale panel for diamond and clique (task brief, step 7): the
    PLANNED capture counts, since no physical capture has run yet. Never plots
    a result that does not exist.
    """
    arms = (capture_proposal or {}).get("arms", [])
    bars = [(f"{arm['regime']} @ {arm['anneal_us']} us", float(arm["captures"])) for arm in arms]
    approval = (capture_proposal or {}).get("approval", "no plan on file")
    subtitle = (
        f"Planned captures per arm (diamond and clique only). Approval status: {approval}. "
        "No physical capture has run yet."
    )
    return _bar_chart("Physical-scale pilot plan (diamond, clique)", subtitle, bars, "captures planned")


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
        "Historical-deadline arm (dwave-neal, 500 reads / 500 sweeps), repaired final objective, original "
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
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for cell, records in records_by_cell.items():
        path = out_dir / f"quality-time-{cell}.svg"
        path.write_text(quality_time_figure(cell, records, kernels, depths), encoding="utf-8")
        paths.append(path)
    physical_path = out_dir / "physical-scale-plan.svg"
    physical_path.write_text(physical_scale_figure(capture_proposal), encoding="utf-8")
    paths.append(physical_path)
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
    parser.add_argument("--cells", nargs="+", default=list(regimes.CELL_NAMES))
    parser.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS))
    parser.add_argument(
        "--depths", type=int, nargs="+", default=None,
        help="Defaults to the pilot depths for --run pilot, the full ladder otherwise.",
    )
    parser.add_argument("--seeded-root", default=None, help="cpu-root's seeded-sweep directory; omit to skip the seed-lane table.")
    parser.add_argument("--portfolio-results", default=None, help="portfolio-deadline/results.json; omit to skip the portfolio tables.")
    parser.add_argument("--capture-proposal", default=None, help="capture-proposal.json; omit to mark physical scale unavailable.")
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cpu_root = Path(args.cpu_root)
    run_dir = cpu_root / args.run
    depths = args.depths or (list(PILOT_DEPTHS) if args.run == "pilot" else list(FULL_DEPTHS))

    records_by_cell = {cell: load_cpu_records(run_dir, cell) for cell in args.cells}
    seeded_by_cell = (
        {cell: load_seeded_sweep_records(Path(args.seeded_root), cell) for cell in args.cells}
        if args.seeded_root else {}
    )
    portfolio_records = load_portfolio_deadline(Path(args.portfolio_results)) if args.portfolio_results else []
    capture_proposal = load_capture_proposal(Path(args.capture_proposal)) if args.capture_proposal else None
    expected_per_arm = expected_per_arm_for(args.run)

    lines = [
        "# Round 2 regime search report", "",
        f"Status: draft, {args.run} data, incomplete. This report draws on `{args.run}` CPU records under "
        f"`{cpu_root}`. No Round 2 quantum processing unit (QPU) capture has run yet: the physical-scale and "
        "portfolio pilots are still pending approval. This draft proves the reporting pipeline end to end. "
        "It states no regime verdict.",
        "",
    ]
    lines += model_identity_table(records_by_cell) + [""]
    lines += solver_settings_table(records_by_cell) + [""]
    lines += depth_quality_time_table(
        records_by_cell, kernels=args.kernels, depths=depths, expected_per_arm=expected_per_arm,
    ) + [""]
    lines += qpu_outcome_table(args.cells) + [""]
    lines += paired_gaps_table(args.cells) + [""]
    if seeded_by_cell:
        lines += seed_lane_table(seeded_by_cell) + [""]
    lines += physical_scale_table(capture_proposal, args.cells) + [""]
    if portfolio_records:
        lines += feasibility_table(portfolio_records) + [""]
        lines += portfolio_pipeline_table(portfolio_records) + [""]
    lines += historical_context_section() + [""]
    lines += next_test_section(args.cells, records_by_cell, capture_proposal) + [""]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    write_figures(out_dir, records_by_cell, args.kernels, depths, capture_proposal, portfolio_records)
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
