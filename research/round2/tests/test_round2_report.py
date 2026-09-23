"""Tests for round2_metrics.py and round2_report.py: the Round 2 report's
pairwise-quality, feasibility, and bootstrap primitives, and the tables and
figures ``round2_report.py`` builds from them.

Task 10 of docs/superpowers/plans/2026-09-22-regime-search-round2.md (D's
plan repo). D's own ``scripts/regime_report.py`` feasibility-rounding fix is
out of scope here (D is read-only); this report carries its own count-based
feasibility formatter (:func:`round2_metrics.format_feasibility`), and the
brief's regression case is tested directly against it below.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import round2_metrics as metrics
import round2_report as report


# ============================================================ round2_metrics


def test_strict_and_material_quality_are_different():
    # The task brief's own step-1 test, verbatim.
    assert metrics.quality_outcome(-1.001, -1.0, 0.0) == "qpu"
    assert metrics.quality_outcome(-1.001, -1.0, 0.005) == "tie"


def test_quality_outcome_rejects_nonfinite_or_negative_tolerance():
    with pytest.raises(ValueError):
        metrics.quality_outcome(float("nan"), -1.0, 0.0)
    with pytest.raises(ValueError):
        metrics.quality_outcome(-1.0, -1.0, -0.1)


def test_quality_outcome_picks_the_lower_energy_as_the_winner():
    assert metrics.quality_outcome(-2.0, -1.0, 0.0) == "qpu"
    assert metrics.quality_outcome(-1.0, -2.0, 0.0) == "cpu"


def test_quality_columns_keep_strict_numeric_and_material_separate():
    # A gap of 1e-10 is numeric noise: a tie under the numeric tolerance and under
    # material, but a real (if tiny) loss under strict zero-tolerance equality.
    columns = metrics.quality_columns(-1.0000000001, -1.0)
    assert columns == {"strict": "qpu", "numeric_tolerance": "tie", "material": "tie"}
    # A gap of 0.2% is real but immaterial: numeric tolerance sees a difference,
    # 0.5% materiality calls it a tie. Zero materiality is never used outside "strict".
    columns = metrics.quality_columns(-1.002, -1.0)
    assert columns["strict"] == "qpu"
    assert columns["numeric_tolerance"] == "qpu"
    assert columns["material"] == "tie"


def test_tally_quality_outcomes_reports_every_key_even_at_zero():
    assert metrics.tally_quality_outcomes(["qpu", "qpu", "tie"]) == {"qpu": 2, "tie": 1, "cpu": 0}


def test_tally_quality_outcomes_all_ties():
    assert metrics.tally_quality_outcomes(["tie"] * 5) == {"qpu": 0, "tie": 5, "cpu": 0}


def test_tally_quality_outcomes_rejects_an_unknown_outcome():
    with pytest.raises(ValueError):
        metrics.tally_quality_outcomes(["qpu", "bogus"])


def test_format_feasibility_never_rounds_a_near_one_share_to_perfect():
    # The task brief's own regression case: seven infeasible reads out of 32,000.
    assert metrics.format_feasibility(31993, 32000) == "31993 / 32000 (99.978125%)"


def test_format_feasibility_shows_a_true_perfect_share_as_perfect():
    assert metrics.format_feasibility(32000, 32000) == "32000 / 32000 (100%)"


def test_format_feasibility_shows_a_zero_share():
    assert metrics.format_feasibility(0, 500) == "0 / 500 (0%)"


def test_format_feasibility_rejects_an_impossible_count():
    with pytest.raises(ValueError):
        metrics.format_feasibility(33, 32)
    with pytest.raises(ValueError):
        metrics.format_feasibility(0, 0)


def test_reconstructed_feasible_range_is_exact_when_only_one_numerator_fits():
    # 0.7500 at 4 reads: only 3/4 rounds to 0.7500 at 4 decimals.
    assert metrics.reconstructed_feasible_range(0.75, 4) == (3, 3)


def test_reconstructed_feasible_range_is_ambiguous_when_rounding_allows_more_than_one():
    # A share of 0.5 rounded to only 1 decimal, out of 200 reads: every numerator
    # from 90 to 110 (share 0.45 to 0.55) rounds to the same recorded 0.5.
    low, high = metrics.reconstructed_feasible_range(0.5, 200, decimals=1)
    assert (low, high) == (90, 110)


def test_format_reconstructed_feasibility_labels_an_ambiguous_numerator():
    text = metrics.format_reconstructed_feasibility(0.5, 200, decimals=1)
    assert "reconstructed" in text
    assert "ambiguous" in text
    assert "/ 200 " in text


def test_format_reconstructed_feasibility_does_not_call_an_ambiguous_share_exact():
    text = metrics.format_reconstructed_feasibility(0.75, 4)
    assert "ambiguous" not in text
    assert text.startswith("3 / 4 (75%)")


# ----------------------------------------------------------------- arm summary


def test_summarize_arm_handles_an_empty_arm_without_crashing():
    summary = metrics.summarize_arm([])
    assert summary["observed"] == 0
    assert summary["completed"] == 0
    assert summary["missing"] is None
    assert summary["median_best_energy"] is None
    assert summary["median_wall_s"] is None


def test_summarize_arm_reports_a_missing_depth_against_the_expected_count():
    records = [{"exit_ok": True, "unsupported": False, "best_energy": -1.0, "wall_s": 0.1}] * 4
    summary = metrics.summarize_arm(records, expected=5)
    assert summary["observed"] == 4
    assert summary["missing"] == 1
    assert summary["completed"] == 4


def test_summarize_arm_keeps_failed_unsupported_and_nonfinite_out_of_the_median():
    records = [
        {"exit_ok": True, "unsupported": False, "best_energy": -3.0, "wall_s": 1.0},
        {"exit_ok": True, "unsupported": False, "best_energy": -1.0, "wall_s": 3.0},
        {"exit_ok": False, "unsupported": False, "best_energy": None, "wall_s": None},
        {"exit_ok": True, "unsupported": True, "best_energy": None, "wall_s": None},
        {"exit_ok": True, "unsupported": False, "best_energy": float("nan"), "wall_s": 9.0},
    ]
    summary = metrics.summarize_arm(records)
    assert summary["observed"] == 5
    assert summary["completed"] == 2
    assert summary["failed"] == 1
    assert summary["unsupported"] == 1
    assert summary["nonfinite"] == 1
    assert summary["median_best_energy"] == -2.0
    assert summary["median_wall_s"] == 2.0


def test_denominator_text_always_states_what_a_share_is_of():
    assert metrics.denominator_text(3, 4) == "3 of 4"


# --------------------------------------------------------------- bootstrap gaps


def test_bootstrap_paired_gap_is_descriptive_with_too_few_groups():
    # Two portfolio baskets (n=18, n=28) each contribute two beta-label gaps: four
    # gaps, but only two independent snapshots (task brief, step 6).
    gaps = [0.01, 0.012, 0.03, 0.028]
    groups = ["n18", "n18", "n28", "n28"]
    result = metrics.bootstrap_paired_gap(gaps, groups, seed=0, resamples=200)
    assert result["group_count"] == 2
    assert result["descriptive"] is True


def test_bootstrap_paired_gap_never_treats_a_shared_snapshot_as_two_independent_points():
    # All four gaps share one snapshot: resampling must draw the whole group at
    # once, so the group count is 1, not 4, and no real interval is possible.
    gaps = [0.01, 0.012, 0.03, 0.028]
    groups = ["only-snapshot"] * 4
    result = metrics.bootstrap_paired_gap(gaps, groups, seed=0, resamples=200)
    assert result["group_count"] == 1
    assert result["descriptive"] is True
    assert math.isnan(result["low"]) and math.isnan(result["high"])


def test_bootstrap_paired_gap_is_reproducible_under_a_fixed_seed():
    rng = np.random.default_rng(1)
    gaps = list(rng.normal(0.01, 0.002, size=20))
    groups = [f"model-{i}" for i in range(20)]
    first = metrics.bootstrap_paired_gap(gaps, groups, seed=42, resamples=500)
    second = metrics.bootstrap_paired_gap(gaps, groups, seed=42, resamples=500)
    assert first == second
    assert first["group_count"] == 20
    assert first["descriptive"] is False


def test_bootstrap_paired_gap_handles_no_data():
    result = metrics.bootstrap_paired_gap([], [], seed=0)
    assert result["group_count"] == 0
    assert result["descriptive"] is True


def test_bootstrap_paired_gap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        metrics.bootstrap_paired_gap([1.0], ["a", "b"], seed=0)


# ============================================================= round2_report


def _cpu_record(cell, nonce, kernel, sweeps, *, best_energy=-1.0, wall_s=1.0, exit_ok=True, unsupported=False):
    return {
        "schema": "round2-cpu-run-v1", "cell": cell, "nonce": nonce, "requested_kernel": kernel,
        "sweeps": sweeps, "reads": 64, "exit_ok": exit_ok, "unsupported": unsupported,
        "best_energy": best_energy, "wall_s": wall_s, "repetition_kind": "timing", "repetition_id": 0,
        "variant": "timing",
    }


def _write_attempt(cell_dir: Path, nonce, kernel, sweeps, attempt, record):
    cell_dir.mkdir(parents=True, exist_ok=True)
    name = f"{nonce}__{kernel}__{sweeps}__timing-0__timing__attempt{attempt}.json"
    (cell_dir / name).write_text(json.dumps(record), encoding="utf-8")


def test_load_cpu_records_ignores_pre_attempt_files_and_takes_the_latest_attempt(tmp_path):
    cell_dir = tmp_path / "native-pm1"
    cell_dir.mkdir()
    # A stale, pre-fix file with no __attempt suffix: fully contaminated, must be ignored.
    (cell_dir / "aa__cpu-sa__512__timing-0__timing.json").write_text(
        json.dumps(_cpu_record("native-pm1", "aa", "cpu-sa", 512, best_energy=-999.0)), encoding="utf-8",
    )
    _write_attempt(cell_dir, "aa", "cpu-sa", 512, 0, _cpu_record("native-pm1", "aa", "cpu-sa", 512, best_energy=-1.0))
    _write_attempt(cell_dir, "aa", "cpu-sa", 512, 1, _cpu_record("native-pm1", "aa", "cpu-sa", 512, best_energy=-2.0))
    records = report.load_cpu_records(tmp_path, "native-pm1")
    assert len(records) == 1
    assert records[0]["best_energy"] == -2.0  # the latest attempt wins


def test_depth_quality_time_table_handles_an_empty_arm(tmp_path):
    records_by_cell = {"native-pm1": []}
    lines = report.depth_quality_time_table(records_by_cell, kernels=("cpu-sa",), depths=(512,))
    text = "\n".join(lines)
    assert "native-pm1" in text
    assert "n/a" in text


def test_depth_quality_time_table_shows_a_missing_depth(tmp_path):
    records_by_cell = {
        "native-pm1": [_cpu_record("native-pm1", "aa", "cpu-sa", 512)],
    }
    lines = report.depth_quality_time_table(
        records_by_cell, kernels=("cpu-sa",), depths=(512, 2048), expected_per_arm=1,
    )
    row_2048 = next(line for line in lines if "| `native-pm1` | `cpu-sa` | 2048 |" in line)
    assert "missing 1" in row_2048 or row_2048.split("|")[-2].strip() != "0"


def test_feasibility_table_formats_the_regression_case_and_flags_unknown_weighting():
    portfolio_records = [
        {
            "n_assets": 28, "cardinality_k": 9, "beta_label": "beta-zero", "status": "completed",
            "raw_feasible_count": 31993, "returned_reads": 32000, "weighting_failed": None,
        },
    ]
    lines = report.feasibility_table(portfolio_records)
    text = "\n".join(lines)
    assert "31993 / 32000 (99.978125%)" in text
    assert "unknown" in text  # weighting_failed=None must read as unknown, never success


def test_feasibility_table_reports_a_known_weighting_failure():
    portfolio_records = [
        {
            "n_assets": 18, "cardinality_k": 6, "beta_label": "beta-nonzero", "status": "completed",
            "raw_feasible_count": 0, "returned_reads": 500, "weighting_failed": True,
        },
    ]
    lines = report.feasibility_table(portfolio_records)
    text = "\n".join(lines)
    assert "0 / 500 (0%)" in text
    assert "failed" in text


def test_seed_lane_table_all_ties():
    records = [
        {"cell": "native-pm1", "nonce": "aa", "seed_source": "qpu", "best_seeded_energy": -1.0, "best_cold_energy": -1.0},
        {"cell": "native-pm1", "nonce": "bb", "seed_source": "qpu", "best_seeded_energy": -2.0, "best_cold_energy": -2.0},
    ]
    lines = report.seed_lane_table({"native-pm1": records})
    text = "\n".join(lines)
    assert "| `native-pm1` | `qpu` | 2 | 0 | 0 | 2 |" in text


def test_physical_scale_table_labels_unavailable_metadata():
    lines = report.physical_scale_table(None, ["diamond-pm1", "clique-portfolio"])
    text = "\n".join(lines)
    assert "unavailable" in text
    assert "diamond-pm1" in text and "clique-portfolio" in text


def test_physical_scale_table_reads_the_capture_proposal_when_present(tmp_path):
    proposal = {
        "arms": [
            {"regime": "diamond-pm1", "anneal_us": 80, "captures": 36, "reads_per_capture": 64},
            {"regime": "diamond-pm1", "anneal_us": 400, "captures": 36, "reads_per_capture": 64},
        ],
        "physical_scales": "50%,75%,100% of audited legal scale ceiling; logical coefficients preserved",
        "approval": "pending",
    }
    lines = report.physical_scale_table(proposal, ["diamond-pm1", "clique-portfolio"])
    text = "\n".join(lines)
    assert "diamond-pm1" in text and "80" in text and "400" in text
    assert "clique-portfolio" in text and "unavailable" in text
    assert "pending" in text


def test_qpu_outcome_table_and_paired_gaps_table_are_explicitly_unavailable():
    cells = ["native-pm1", "diamond-pm1"]
    outcome_text = "\n".join(report.qpu_outcome_table(cells))
    gaps_text = "\n".join(report.paired_gaps_table(cells))
    assert "unavailable" in outcome_text and "native-pm1" in outcome_text
    assert "unavailable" in gaps_text and "diamond-pm1" in gaps_text


def test_main_writes_a_draft_report_and_figures_from_a_small_fixture(tmp_path, monkeypatch):
    cpu_root = tmp_path / "cpu"
    for cell in ("native-pm1",):
        for kernel in ("cpu-sa", "cpu-msa-f64"):
            cell_dir = cpu_root / "pilot" / cell
            for i, sweeps in enumerate((512, 2048)):
                _write_attempt(
                    cell_dir, f"model{i}", kernel, sweeps, 0,
                    _cpu_record(cell, f"model{i}", kernel, sweeps, best_energy=-10.0 - i, wall_s=0.1 * sweeps),
                )
    portfolio_root = tmp_path / "portfolio-deadline"
    portfolio_root.mkdir(parents=True)
    (portfolio_root / "results.json").write_text(json.dumps({"records": [
        {
            "n_assets": 18, "cardinality_k": 6, "beta_label": "beta-zero", "status": "completed",
            "raw_feasible_count": 0, "returned_reads": 500, "weighting_failed": None, "objective": 0.01,
            "elapsed_s": 0.1, "deadline_s": 10.0,
        },
    ]}), encoding="utf-8")
    out_dir = tmp_path / "report"

    monkeypatch.setattr(
        "sys.argv",
        [
            "round2_report.py",
            "--cpu-root", str(cpu_root), "--run", "pilot",
            "--portfolio-results", str(portfolio_root / "results.json"),
            "--out-dir", str(out_dir),
            "--cells", "native-pm1",
        ],
    )
    assert report.main() == 0
    draft = (out_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "pilot data, incomplete" in draft
    assert "native-pm1" in draft
    figures = list(out_dir.glob("*.svg"))
    assert figures
