"""Pure metrics for the Round 2 report: pairwise quality outcomes, feasibility
counts, arm summaries, and bootstrap paired gaps.

Every function here takes already-loaded records or plain numbers and returns
plain data; see ``round2_report.py`` for the loading, table assembly, and
figures this module feeds. Task 10 of
``docs/superpowers/plans/2026-09-22-regime-search-round2.md`` (D's plan repo)
is the brief this implements.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------- quality outcome

#: Zero materiality: used only for the explicitly labeled strict-quality column
#: (task brief, step 2). Exact software equality lives at this same tolerance.
STRICT_TOLERANCE = 0.0

#: Absorbs float64 rescoring noise without treating it as a material result.
#: Distinct from experimental materiality (task brief, step 2): this tolerance
#: answers "is this gap numerical noise," not "is this gap worth acting on."
NUMERIC_TOLERANCE = 1e-9

#: The design doc's own manuscript materiality bound (0.5%), reused here rather
#: than reinvented (docs/superpowers/specs/2026-09-22-regime-search-round2-design.md).
MATERIALITY_TOLERANCE = 0.005

QUALITY_OUTCOMES = ("qpu", "tie", "cpu")


def quality_outcome(qpu: float, cpu: float, tolerance: float) -> str:
    """The explicit pairwise quality rule (task brief, step 1), verbatim."""
    if not all(math.isfinite(x) for x in (qpu, cpu, tolerance)) or tolerance < 0:
        raise ValueError("finite scores and nonnegative tolerance required")
    gap = abs(qpu - cpu) / max(abs(qpu), abs(cpu), 1e-12)
    if gap <= tolerance:
        return "tie"
    return "qpu" if qpu < cpu else "cpu"


def quality_columns(qpu: float, cpu: float) -> Dict[str, str]:
    """The three distinct quality columns of one paired comparison (task brief, step 2):
    exact software equality, numeric-tolerance sensitivity, and 0.5% materiality.
    Zero materiality is used only for the ``strict`` column here, never elsewhere.
    """
    return {
        "strict": quality_outcome(qpu, cpu, STRICT_TOLERANCE),
        "numeric_tolerance": quality_outcome(qpu, cpu, NUMERIC_TOLERANCE),
        "material": quality_outcome(qpu, cpu, MATERIALITY_TOLERANCE),
    }


def tally_quality_outcomes(outcomes: Sequence[str]) -> Dict[str, int]:
    """Win/tie/loss counts from a sequence of :func:`quality_outcome` results.

    Always returns every key, even at zero, so a caller never has to guess
    whether a missing key means zero or means the count was never computed
    (task brief, step 5: "every percentage prints its denominator").
    """
    counts = {name: 0 for name in QUALITY_OUTCOMES}
    for outcome in outcomes:
        if outcome not in counts:
            raise ValueError(f"unknown quality outcome {outcome!r}")
        counts[outcome] += 1
    return counts


# ------------------------------------------------------------------ feasibility


def format_feasibility(feasible_count: int, returned_reads: int) -> str:
    """``"31993 / 32000 (99.978125%)"`` (task brief, step 3, regression case).

    Six decimal places distinguish a near-one share from a perfect one: a
    share short by even a single read out of tens of thousands never rounds
    to ``100%`` here, because the fraction still shows in the sixth decimal.
    """
    if returned_reads <= 0:
        raise ValueError("returned_reads must be positive")
    if not 0 <= feasible_count <= returned_reads:
        raise ValueError("feasible_count must be between 0 and returned_reads")
    percent = feasible_count / returned_reads * 100
    percent_text = f"{percent:.6f}".rstrip("0").rstrip(".")
    if feasible_count < returned_reads and percent_text == "100":
        # Six decimals run out of resolution once returned_reads is large enough
        # (one bad read in 2e8 rounds to 99.9999995, which .6f rounds again to
        # 100.000000): an explicit less-than marker holds at any scale, rather
        # than chasing ever more decimal places.
        percent_text = "<100"
    return f"{feasible_count} / {returned_reads} ({percent_text}%)"


def reconstructed_feasible_range(k_share: float, total: int, decimals: int = 4) -> Tuple[int, int]:
    """The integer numerators out of ``total`` consistent with a share already
    rounded to ``decimals`` places (task brief, step 3: old CSV-only data records
    only a rounded share, never a raw count). A single value means the rounding
    leaves only one possible numerator; a range means it does not, and no
    numerator should be invented from inside that range.
    """
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0.0 <= k_share <= 1.0:
        raise ValueError("k_share must be a share between 0 and 1")
    tolerance = 0.5 * 10 ** (-decimals)
    low = max(0, math.ceil((k_share - tolerance) * total - 1e-9))
    high = min(total, math.floor((k_share + tolerance) * total + 1e-9))
    return low, high


def format_reconstructed_feasibility(k_share: float, total: int, decimals: int = 4) -> str:
    """Old CSV-only feasibility, labeled as reconstructed and never given an
    invented exact numerator when the rounding leaves more than one possibility.
    """
    low, high = reconstructed_feasible_range(k_share, total, decimals)
    percent_text = f"{k_share * 100:.{decimals}f}".rstrip("0").rstrip(".")
    count_text = str(low) if low == high else f"{low}-{high}"
    ambiguity = "" if low == high else "; numerator ambiguous"
    return f"{count_text} / {total} ({percent_text}%) [reconstructed from a rounded share{ambiguity}]"


# -------------------------------------------------------------------- arm summary


#: A completed record with no ``timing_mode`` and no contaminated ``host`` is
#: "clean serial" by definition -- a hand-built fixture with neither field
#: (or a real run with no ``--cpu``, so no host sample exists) is treated the
#: same way, never silently folded into "contaminated" or "parallel".
CLEAN_SERIAL = "clean serial"
CONTAMINATED = "contaminated"
PARALLEL = "parallel"


def _timing_label(record: Mapping[str, Any]) -> str:
    """Which of :data:`CLEAN_SERIAL`, :data:`CONTAMINATED`, or :data:`PARALLEL`
    one completed record's wall time belongs to (review, Important item 1:
    "never pool serial and parallel timing without a label").
    """
    if record.get("timing_mode") == "parallel":
        return PARALLEL
    if bool((record.get("host") or {}).get("contaminated")):
        return CONTAMINATED
    return CLEAN_SERIAL


def summarize_arm(records: Sequence[Mapping[str, Any]], expected: Optional[int] = None) -> Dict[str, Any]:
    """The depth/quality/time summary of one (cell, kernel, sweeps) arm.

    Every record lands in exactly one bucket -- completed, unsupported, failed,
    or nonfinite -- and only the completed, finite subset feeds
    ``median_best_energy`` (task brief, step 5: missing, failed, and nonfinite
    observations stay outside the comparable denominator). Quality uses every
    completed record regardless of host contamination or timing mode: an
    ``exit_ok`` energy is exactly as valid whether the run was clean, shared
    with another job, or run in parallel (review, Important item 1: "Quality
    results may use every exit_ok record").

    Wall time is different: a contaminated or parallel-mode run measures a
    busier or differently-loaded host, and pooling it with a clean serial run
    silently corrupts the timing comparison. ``median_wall_s`` and ``clean_n``
    cover the clean-serial subset ONLY; ``timing_by_label`` reports the
    ``{n, median_wall_s}`` of every label (:data:`CLEAN_SERIAL`,
    :data:`CONTAMINATED`, :data:`PARALLEL`) that has at least one completed
    record, so a contaminated or parallel timing is always visible, always
    labeled, and never silently merged into the clean figure.

    ``expected`` is the job count this arm should have if every job already
    has a record; the gap becomes ``missing`` when given, and stays ``None``
    (not zero) when the caller does not know the intended count -- an empty
    arm (``records=()``) is handled the same way as any other, with every
    count at zero rather than a crash.
    """
    completed: List[Mapping[str, Any]] = []
    unsupported = 0
    failed = 0
    nonfinite = 0
    for record in records:
        if record.get("unsupported"):
            unsupported += 1
            continue
        if not record.get("exit_ok"):
            failed += 1
            continue
        best = record.get("best_energy")
        if best is None or not math.isfinite(best):
            nonfinite += 1
            continue
        completed.append(record)
    observed = len(records)
    missing = max(0, expected - observed) if expected is not None else None
    best_energies = [float(r["best_energy"]) for r in completed]

    wall_by_label: Dict[str, List[float]] = {}
    for record in completed:
        wall = record.get("wall_s")
        if wall is not None:
            wall_by_label.setdefault(_timing_label(record), []).append(float(wall))
    timing_by_label = {
        label: {"n": len(values), "median_wall_s": float(np.median(values)) if values else None}
        for label, values in sorted(wall_by_label.items())
    }
    clean_values = wall_by_label.get(CLEAN_SERIAL, [])

    return {
        "observed": observed,
        "expected": expected,
        "missing": missing,
        "completed": len(completed),
        "unsupported": unsupported,
        "failed": failed,
        "nonfinite": nonfinite,
        "median_best_energy": float(np.median(best_energies)) if best_energies else None,
        "median_wall_s": float(np.median(clean_values)) if clean_values else None,
        "clean_n": len(clean_values),
        "timing_by_label": timing_by_label,
    }


def denominator_text(count: int, total: int) -> str:
    """``"N of M"``, so a percentage never appears without stating what it is a share of."""
    return f"{count} of {total}"


# --------------------------------------------------------------- bootstrap gaps

#: Below this many independent groups, an interval is descriptive only (task
#: brief, step 6): too few groups to say a resample distribution approximates
#: sampling variability. A documented, not measured, choice -- the same kind
#: of round-number floor ``regime_metrics.py``'s own screening thresholds use.
MIN_INDEPENDENT_GROUPS = 5

#: The task brief's own fixed resample count (step 6).
BOOTSTRAP_RESAMPLES = 10_000

#: The task brief's "fixed recorded seed" (step 6). Named and recorded in
#: every result (review, Also-fix item 1) rather than left to each caller to
#: pick and remember its own value.
BOOTSTRAP_SEED = 20260922


def bootstrap_paired_gap(
    gaps: Sequence[float], groups: Sequence[str],
    *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """The bootstrap interval of the median paired gap, resampled by GROUP, not
    by point (task brief, step 6: "group correlated portfolio inputs by
    snapshot"). Two gaps that share a group (e.g. two beta labels drawn from
    the same market snapshot) are always resampled together, never treated as
    two independent draws. Reports the group count and marks the interval
    ``descriptive`` when there are too few independent groups
    (:data:`MIN_INDEPENDENT_GROUPS`) for a resample distribution to mean
    anything close to a real confidence bound. ``seed`` and ``resamples``
    travel with the result on every return path, so a reader never has to
    trust an out-of-band claim about what generated it.
    """
    if len(gaps) != len(groups):
        raise ValueError("gaps and groups must be the same length")
    if not gaps:
        return {
            "point": float("nan"), "low": float("nan"), "high": float("nan"),
            "group_count": 0, "descriptive": True, "seed": seed, "resamples": resamples,
        }
    by_group: Dict[str, List[float]] = {}
    for gap, group in zip(gaps, groups):
        by_group.setdefault(group, []).append(float(gap))
    unique_groups = sorted(by_group)
    group_count = len(unique_groups)
    point = float(np.median(list(gaps)))
    if group_count < 2:
        return {
            "point": point, "low": float("nan"), "high": float("nan"),
            "group_count": group_count, "descriptive": True, "seed": seed, "resamples": resamples,
        }
    rng = np.random.default_rng(seed)
    resampled_medians = np.empty(resamples, dtype=np.float64)
    group_index = np.arange(group_count)
    for i in range(resamples):
        picked = rng.choice(group_index, size=group_count, replace=True)
        pooled = [value for g in picked for value in by_group[unique_groups[g]]]
        resampled_medians[i] = np.median(pooled)
    low, high = np.percentile(resampled_medians, [2.5, 97.5])
    return {
        "point": point,
        "low": float(low),
        "high": float(high),
        "group_count": group_count,
        "descriptive": group_count < MIN_INDEPENDENT_GROUPS,
        "seed": seed,
        "resamples": resamples,
    }
