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
import re
from pathlib import Path
from typing import Optional

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


def test_format_feasibility_never_prints_100_percent_when_a_read_is_infeasible_at_huge_scale():
    # 1 bad read in 2e8: six decimal places round 99.9999995% up to "100.000000",
    # which would falsely claim a perfect share (review, Important/Also-fix item).
    text = metrics.format_feasibility(199_999_999, 200_000_000)
    assert text == "199999999 / 200000000 (<100%)"


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


def test_lane_diversity_reports_unique_states_and_pairwise_hamming_distances():
    spins = np.asarray([[1, 1, 1], [1, 1, -1], [-1, -1, -1]], dtype=np.int8)
    assert metrics.lane_diversity(spins) == {
        "unique_states": 3,
        "mean_pairwise_hamming": 2.0,
        "minimum_pairwise_hamming": 1,
    }


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


def test_summarize_arm_uses_every_exit_ok_record_for_quality_but_splits_wall_time_by_label():
    clean = {"exit_ok": True, "unsupported": False, "best_energy": -1.0, "wall_s": 1.0}
    contaminated = {
        "exit_ok": True, "unsupported": False, "best_energy": -2.0, "wall_s": 100.0,
        "host": {"contaminated": True},
    }
    parallel = {
        "exit_ok": True, "unsupported": False, "best_energy": -3.0, "wall_s": 0.5,
        "timing_mode": "parallel",
    }
    summary = metrics.summarize_arm([clean, contaminated, parallel])
    # Quality uses every exit_ok record, contaminated and parallel included (task
    # brief's own rule, review Important item 1).
    assert summary["completed"] == 3
    assert summary["median_best_energy"] == -2.0
    # Wall time never pools a contaminated or parallel record with clean serial.
    assert summary["median_wall_s"] == 1.0  # clean-serial only
    assert summary["clean_n"] == 1
    assert summary["timing_by_label"] == {
        "clean serial": {"n": 1, "median_wall_s": 1.0},
        "contaminated": {"n": 1, "median_wall_s": 100.0},
        "parallel": {"n": 1, "median_wall_s": 0.5},
    }


def test_summarize_arm_treats_a_missing_host_as_clean_serial():
    # No host field at all (e.g. a hand-built fixture, or a run with no --cpu):
    # never silently mislabeled contaminated or parallel.
    record = {"exit_ok": True, "unsupported": False, "best_energy": -1.0, "wall_s": 2.0}
    summary = metrics.summarize_arm([record])
    assert summary["clean_n"] == 1
    assert summary["timing_by_label"] == {"clean serial": {"n": 1, "median_wall_s": 2.0}}


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


def test_bootstrap_paired_gap_records_its_own_seed_and_resamples_on_every_path():
    # A "fixed recorded seed" (task brief, step 6) means the seed and resample
    # count travel with the result, on every return path -- including the two
    # early-return paths (no data, too few groups) the review found missing it.
    empty = metrics.bootstrap_paired_gap([], [], seed=7, resamples=123)
    assert empty["seed"] == 7 and empty["resamples"] == 123
    one_group = metrics.bootstrap_paired_gap([1.0, 2.0], ["only", "only"], seed=7, resamples=123)
    assert one_group["seed"] == 7 and one_group["resamples"] == 123
    real = metrics.bootstrap_paired_gap([1.0, 2.0], ["a", "b"], seed=7, resamples=123)
    assert real["seed"] == 7 and real["resamples"] == 123


def test_bootstrap_paired_gap_defaults_to_the_named_fixed_seed_constant():
    result = metrics.bootstrap_paired_gap([1.0, 2.0], ["a", "b"], resamples=50)
    assert result["seed"] == metrics.BOOTSTRAP_SEED


# ============================================================= round2_report


def _cpu_record(
    cell, nonce, kernel, sweeps, *,
    best_energy: Optional[float] = -1.0, wall_s: Optional[float] = 1.0, exit_ok=True, unsupported=False,
):
    return {
        "schema": "round2-cpu-run-v1", "cell": cell, "nonce": nonce, "requested_kernel": kernel,
        "sweeps": sweeps, "reads": 64, "exit_ok": exit_ok, "unsupported": unsupported,
        "best_energy": best_energy, "wall_s": wall_s, "elapsed_sampling_s": wall_s,
        "repetition_kind": "timing", "repetition_id": 0,
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


def test_load_cpu_records_uses_the_latest_exit_ok_attempt_for_quality(tmp_path):
    # A --repeat-contaminated retry (review, M3): attempt0 succeeded, attempt1
    # crashed outright. The old "just take the highest attempt number" rule
    # would report attempt1's None energy, silently discarding a valid result.
    cell_dir = tmp_path / "native-pm1"
    cell_dir.mkdir()
    _write_attempt(
        cell_dir, "aa", "cpu-sa", 512, 0,
        _cpu_record("native-pm1", "aa", "cpu-sa", 512, best_energy=-5.0, wall_s=1.0),
    )
    _write_attempt(
        cell_dir, "aa", "cpu-sa", 512, 1,
        _cpu_record("native-pm1", "aa", "cpu-sa", 512, exit_ok=False, best_energy=None, wall_s=None),
    )
    records = report.load_cpu_records(tmp_path, "native-pm1")
    assert len(records) == 1
    assert records[0]["best_energy"] == -5.0
    assert records[0]["wall_s"] == 1.0


def test_load_cpu_records_uses_fastest_successful_timing_when_that_attempt_is_contaminated(tmp_path):
    # Quality follows the latest success, while timing follows the successful
    # attempt with the lowest finite wall time, including contaminated runs.
    cell_dir = tmp_path / "native-pm1"
    cell_dir.mkdir()
    clean = _cpu_record("native-pm1", "aa", "cpu-sa", 512, best_energy=-5.0, wall_s=1.0)
    clean["host"] = {"contaminated": False}  # a real record always carries this field, never omits it
    _write_attempt(cell_dir, "aa", "cpu-sa", 512, 0, clean)
    contaminated = _cpu_record("native-pm1", "aa", "cpu-sa", 512, best_energy=-5.5, wall_s=0.5)
    contaminated.update(
        host={"contaminated": True}, elapsed_sampling_s=0.25, timing_mode="parallel",
        setup_s=0.1, graph_setup_s=0.2, concurrent_workers=14,
    )
    _write_attempt(cell_dir, "aa", "cpu-sa", 512, 1, contaminated)
    records = report.load_cpu_records(tmp_path, "native-pm1")
    assert len(records) == 1
    assert records[0]["best_energy"] == -5.5  # quality: the latest successful attempt
    assert records[0]["wall_s"] == 0.5
    assert records[0]["elapsed_sampling_s"] == 0.25
    assert records[0]["setup_s"] == 0.1
    assert records[0]["graph_setup_s"] == 0.2
    assert records[0]["host"]["contaminated"] is True
    assert records[0]["timing_mode"] == "parallel"
    assert records[0]["concurrent_workers"] == 14


def test_load_cpu_records_copies_sampling_time_from_the_fastest_attempt(tmp_path):
    cell_dir = tmp_path / "native-pm1"
    clean = _cpu_record("native-pm1", "aa", "cpu-sa", 512, best_energy=-5.0, wall_s=1.0)
    clean.update(host={"contaminated": False}, elapsed_sampling_s=1.0)
    _write_attempt(cell_dir, "aa", "cpu-sa", 512, 0, clean)
    contaminated = _cpu_record("native-pm1", "aa", "cpu-sa", 512, best_energy=-5.5, wall_s=20.0)
    contaminated.update(host={"contaminated": True}, elapsed_sampling_s=20.0)
    _write_attempt(cell_dir, "aa", "cpu-sa", 512, 1, contaminated)

    records = report.load_cpu_records(tmp_path, "native-pm1")

    assert records[0]["best_energy"] == -5.5
    assert records[0]["wall_s"] == 1.0
    assert records[0]["elapsed_sampling_s"] == 1.0


@pytest.mark.parametrize("option", ["--qpu-root", "--physical-capture-manifest"])
def test_main_rejects_an_incomplete_qpu_capture_option_pair(tmp_path, monkeypatch, option):
    cpu_root = tmp_path / "cpu"
    out_dir = tmp_path / "report"
    argv = ["round2_report.py", "--cpu-root", str(cpu_root), "--out-dir", str(out_dir)]
    argv.extend([option, str(tmp_path / "capture-input")])
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit) as exc:
        report.main()

    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("run", "expected_text", "unexpected_text"),
    [
        ("pilot", "CPU pilot models", "full CPU campaign once it finishes"),
        ("campaign", "CPU campaign models", "full CPU campaign once it finishes"),
    ],
)
def test_captured_cell_status_matches_the_loaded_run(run, expected_text, unexpected_text):
    lines = report.next_test_section(
        ["native-pm1"], {"native-pm1": []}, None, run, captured_regimes={"native-pm1"},
    )
    status = "\n".join(lines)

    assert expected_text in status
    if run == "campaign":
        assert unexpected_text not in status


def test_next_test_section_counts_only_supported_kernels():
    # native-125 and clique-portfolio report cpu-msa-unit for every model, but
    # every one of those records is unsupported. The established text must name
    # only kernels that actually produced a supported, completed CPU record.
    records = {
        "native-125": [
            _cpu_record("native-125", "aa", "cpu-msa-f64", 512),
            _cpu_record("native-125", "bb", "cpu-sa", 512),
            _cpu_record("native-125", "cc", "cpu-msa-unit", 512, exit_ok=True, unsupported=True),
        ],
    }
    lines = report.next_test_section(["native-125"], records, None, "pilot")
    row = next(line for line in lines if line.startswith("| `native-125` |"))

    assert "across 2 kernels (`cpu-msa-f64`, `cpu-sa`)" in row
    assert "cpu-msa-unit" not in row


def test_next_test_section_campaign_cell_with_no_capture_says_the_campaign_is_done():
    # When the loaded run is the campaign, the cell with no capture and no
    # physical-scale plan has already run its full comparison. Telling the reader
    # to extend it would ask for work that is complete.
    lines = report.next_test_section(["native-125"], {"native-125": []}, None, "campaign")
    row = next(line for line in lines if line.startswith("| `native-125` |"))

    assert "The CPU campaign for this cell is complete." in row
    assert "Extend the campaign to the full 100-model comparison" not in row


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
    cells_2048 = [c.strip() for c in row_2048.split("|")]
    assert cells_2048[5] == "1"  # Missing column: 1 expected, 0 observed at this depth
    row_512 = next(line for line in lines if "| `native-pm1` | `cpu-sa` | 512 |" in line)
    cells_512 = [c.strip() for c in row_512.split("|")]
    assert cells_512[5] == "0"  # Missing column: 1 expected, 1 observed at this depth


def test_depth_quality_time_table_uses_fastest_run_and_keeps_other_timing_labels():
    records_by_cell = {
        "native-pm1": [
            _cpu_record("native-pm1", "aa", "cpu-sa", 512, wall_s=1.0),
            {**_cpu_record("native-pm1", "bb", "cpu-sa", 512, wall_s=100.0), "host": {"contaminated": True}},
            {**_cpu_record("native-pm1", "cc", "cpu-sa", 512, wall_s=0.5), "timing_mode": "parallel"},
        ],
    }
    lines = report.depth_quality_time_table(records_by_cell, kernels=("cpu-sa",), depths=(512,))
    row = next(line for line in lines if "| `native-pm1` | `cpu-sa` | 512 |" in line)
    # Quality (Completed, Median best energy) counts all three exit_ok records.
    cells = [c.strip() for c in row.split("|")]
    assert cells[6] == "3"  # Completed
    # Fastest-run medians pool all three records. Diagnostic columns retain
    # their contaminated and parallel labels.
    assert "Sampling s, fastest run from quality run (n)" in lines[4]
    assert "Wall s, fastest run from quality run (n)" in lines[4]
    assert cells[11] == "1 (3)"
    assert cells[12] == "1 (3)"
    assert cells[13] == "100 (1)"
    assert cells[14] == "0.5 (1)"


def test_campaign_quality_uses_timing_subset_for_sampling_and_wall_time(tmp_path, monkeypatch):
    cpu_root = tmp_path / "cpu"
    campaign_dir = cpu_root / "campaign" / "native-pm1"
    campaign_record = _cpu_record("native-pm1", "campaign-model", "cpu-sa", 512, best_energy=-7.0, wall_s=99.0)
    campaign_record.update(timing_mode="parallel", elapsed_sampling_s=90.0)
    _write_attempt(campaign_dir, "campaign-model", "cpu-sa", 512, 0, campaign_record)

    timing_dir = cpu_root / "timing-subset" / "native-pm1"
    timing_record = _cpu_record("native-pm1", "timing-model", "cpu-sa", 512, best_energy=-6.0, wall_s=2.0)
    timing_record.update(
        timing_mode="parallel", elapsed_sampling_s=1.25, host={"contaminated": True},
    )
    _write_attempt(timing_dir, "timing-model", "cpu-sa", 512, 0, timing_record)
    out_dir = tmp_path / "report"
    monkeypatch.setattr("sys.argv", [
        "round2_report.py", "--cpu-root", str(cpu_root), "--run", "campaign",
        "--out-dir", str(out_dir), "--cells", "native-pm1", "--kernels", "cpu-sa", "--depths", "512",
    ])

    assert report.main() == 0
    draft = (out_dir / "REPORT.md").read_text(encoding="utf-8")
    depth_section = draft.split("## Depth, quality, and time")[1].split("## ")[0]
    row = next(line for line in depth_section.splitlines() if line.startswith("| `native-pm1` | `cpu-sa` | 512 |"))
    assert "timing-subset" in depth_section
    assert "1.25 (1)" in row  # fastest-run sampling seconds include parallel records
    assert "2 (1)" in row  # end-to-end wall seconds come from the matched subset
    assert [cell.strip() for cell in row.split("|")][10] == "-7"  # quality remains from campaign records
    figure = (out_dir / "quality-time-native-pm1.svg").read_text(encoding="utf-8")
    assert "median sampling 1.25 s" in figure
    assert "median wall 2 s" in figure


def test_campaign_without_timing_subset_reports_missing_timing_reason(tmp_path, monkeypatch):
    cpu_root = tmp_path / "cpu"
    campaign_dir = cpu_root / "campaign" / "native-pm1"
    campaign_record = _cpu_record("native-pm1", "campaign-model", "cpu-sa", 512, best_energy=-7.0, wall_s=0.25)
    campaign_record.update(timing_mode="parallel", elapsed_sampling_s=0.1)
    _write_attempt(campaign_dir, "campaign-model", "cpu-sa", 512, 0, campaign_record)
    out_dir = tmp_path / "report"
    monkeypatch.setattr("sys.argv", [
        "round2_report.py", "--cpu-root", str(cpu_root), "--run", "campaign",
        "--out-dir", str(out_dir), "--cells", "native-pm1", "--kernels", "cpu-sa", "--depths", "512",
    ])

    assert report.main() == 0
    draft = (out_dir / "REPORT.md").read_text(encoding="utf-8")
    depth_section = draft.split("## Depth, quality, and time")[1].split("## ")[0]
    row = next(line for line in depth_section.splitlines() if line.startswith("| `native-pm1` | `cpu-sa` | 512 |"))
    assert "timing-subset" in depth_section
    cells = [cell.strip() for cell in row.split("|")]
    assert "no matching timing-subset record" in cells[11]
    assert "no matching timing-subset record" in cells[12]
    assert cells[14] == "0.25 (1)"  # parallel label remains visible
    figure = (out_dir / "quality-time-native-pm1.svg").read_text(encoding="utf-8")
    assert "No timing-subset record" in figure
    assert "<circle" not in figure


def test_lane_diversity_table_separates_seeded_and_cold_lanes_from_v2_samples(tmp_path):
    spins = np.asarray([
        [1, 1, 1], [1, 1, -1],  # seeded lanes: distance 1
        [-1, -1, -1], [-1, 1, -1],  # cold lanes: distance 1
    ], dtype=np.int8)
    samples_path = tmp_path / "seeded-v2.npz"
    np.savez_compressed(samples_path, spins=spins)
    record = {
        "schema": "round2-seeded-sweep-v2", "cell": "native-pm1", "nonce": "model-a",
        "seed_source": "qpu", "sweeps": 32768, "seed_lanes": 2, "cold_lanes": 2,
        "exit_ok": True, "unsupported": False, "_samples_path": str(samples_path),
    }

    lines = report.lane_diversity_table({"native-pm1": [record]})
    seeded_row = next(line for line in lines if "| seeded |" in line)
    cold_row = next(line for line in lines if "| cold |" in line)
    assert "| 1 | 2 | 1 | 1 |" in seeded_row
    assert "| 1 | 2 | 1 | 1 |" in cold_row


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


def test_feasibility_table_prints_provenance_so_synthetic_is_never_mistaken_for_historical():
    portfolio_records = [
        {
            "n_assets": 28, "cardinality_k": 9, "beta_label": "beta-zero", "status": "completed",
            "raw_feasible_count": 0, "returned_reads": 500, "weighting_failed": None,
            "provenance": "synthetic-test",
        },
    ]
    lines = report.feasibility_table(portfolio_records)
    text = "\n".join(lines)
    assert "synthetic-test" in text
    assert "historical-deadline" not in text.lower()


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


def _seeded_record(
    cell, nonce, source, *,
    seeded: Optional[float] = -1.0, cold: Optional[float] = -1.0, exit_ok=True, unsupported=False,
):
    return {
        "cell": cell, "nonce": nonce, "seed_source": source, "exit_ok": exit_ok, "unsupported": unsupported,
        "best_seeded_energy": seeded, "best_cold_energy": cold,
    }


def test_seed_lane_table_all_ties():
    records = [
        _seeded_record("native-pm1", "aa", "qpu", seeded=-1.0, cold=-1.0),
        _seeded_record("native-pm1", "bb", "qpu", seeded=-2.0, cold=-2.0),
    ]
    lines = report.seed_lane_table({"native-pm1": records})
    text = "\n".join(lines)
    # Strict, numeric, and material all agree here (an exact tie ties at every
    # tolerance), each shown as its own labeled seeded/cold/tie triple (task
    # brief, step 2: never a bare, unlabeled zero-tolerance count).
    assert "| `native-pm1` | `qpu` | 2 of 2 | 0 | 0/0/2 | 0/0/2 | 0/0/2 |" in text


def test_seed_lane_table_labels_strict_numeric_and_material_separately():
    # A 0.2% gap: a real strict/numeric win for the seeded lane, but a material tie.
    records = [_seeded_record("native-pm1", "aa", "qpu", seeded=-1.002, cold=-1.0)]
    lines = report.seed_lane_table({"native-pm1": records})
    text = "\n".join(lines)
    assert "| `native-pm1` | `qpu` | 1 of 1 | 0 | 1/0/0 | 1/0/0 | 0/0/1 |" in text


def test_seed_lane_table_excludes_failed_and_nonfinite_records_and_counts_them():
    records = [
        _seeded_record("native-pm1", "aa", "qpu", seeded=-2.0, cold=-1.0),
        _seeded_record("native-pm1", "bb", "qpu", exit_ok=False, seeded=None, cold=None),
        _seeded_record("native-pm1", "cc", "qpu", seeded=float("nan"), cold=-1.0),
        _seeded_record("native-pm1", "dd", "qpu", unsupported=True, seeded=None, cold=None),
    ]
    lines = report.seed_lane_table({"native-pm1": records})
    text = "\n".join(lines)
    assert "| `native-pm1` | `qpu` | 1 of 4 | 3 |" in text


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


def test_physical_scale_table_never_calls_approval_alone_captured():
    # Approval is not capture (review, Important item 8): until real capture
    # output exists, the status must say so explicitly, never "captured".
    proposal = {
        "arms": [{"regime": "diamond-pm1", "anneal_us": 80, "captures": 36, "reads_per_capture": 64}],
        "approval": "approved",
    }
    lines = report.physical_scale_table(proposal, ["diamond-pm1"])
    text = "\n".join(lines)
    assert "approved, not yet captured" in text
    assert "| captured |" not in text


def test_physical_scale_table_shows_captured_only_when_capture_output_is_named():
    proposal = {
        "arms": [{"regime": "diamond-pm1", "anneal_us": 80, "captures": 36, "reads_per_capture": 64}],
        "approval": "approved",
    }
    lines = report.physical_scale_table(proposal, ["diamond-pm1"], captured_regimes={"diamond-pm1"})
    text = "\n".join(lines)
    assert "| captured |" in text


def test_portfolio_pipeline_table_shows_the_repaired_feasibility_the_feasibility_table_cites():
    # The feasibility table's own text points here for the repaired, selected
    # answer's feasibility (review, Important item 5): both fields must
    # actually appear.
    portfolio_records = [
        {
            "n_assets": 18, "cardinality_k": 6, "beta_label": "beta-zero", "status": "completed",
            "elapsed_s": 0.06, "deadline_s": 10.0, "objective": 0.0001,
            "feasible": True, "selected_raw_cardinality": 2,
        },
        {
            "n_assets": 28, "cardinality_k": 9, "beta_label": "beta-nonzero", "status": "completed",
            "elapsed_s": 0.1, "deadline_s": 10.0, "objective": -0.0002,
            "feasible": False, "selected_raw_cardinality": None,
        },
    ]
    lines = report.portfolio_pipeline_table(portfolio_records)
    text = "\n".join(lines)
    assert "| yes | 2 |" in text
    assert "| no | n/a |" in text


def test_portfolio_pipeline_table_shows_provenance_and_classifies_on_end_to_end_time():
    # elapsed_s is sampling time only; status is decided by end_to_end_s, which
    # includes repair/weighting time (review, I5). Both must be visible, along
    # with the arm's own provenance, never silently defaulted to "historical".
    portfolio_records = [
        {
            "n_assets": 18, "cardinality_k": 6, "beta_label": "beta-zero", "status": "completed",
            "provenance": "synthetic-test", "elapsed_s": 0.06, "repair_s": 0.02, "end_to_end_s": 0.08,
            "deadline_s": 10.0, "objective": 0.0001, "feasible": True, "selected_raw_cardinality": 2,
        },
    ]
    lines = report.portfolio_pipeline_table(portfolio_records)
    text = "\n".join(lines)
    assert "synthetic-test" in text
    assert "0.08" in text  # end_to_end_s, the time the status is classified on
    assert "0.02" in text  # repair_s
    assert "historical-deadline" not in text.lower()  # never call the arm itself historical


def test_qpu_outcome_table_and_paired_gaps_table_state_no_comparable_pairs_with_no_captures():
    cells = ["native-pm1", "diamond-pm1"]
    outcome_text = "\n".join(report.qpu_outcome_table(cells, {}, {}, ["cpu-sa"], [512]))
    gaps_text = "\n".join(report.paired_gaps_table(cells, {}, {}, ["cpu-sa"], [512]))
    assert "No comparable pairs" in outcome_text
    assert "No comparable pairs" in gaps_text


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


def test_main_labels_the_header_with_the_run_actually_loaded(tmp_path, monkeypatch):
    cpu_root = tmp_path / "cpu"
    (cpu_root / "campaign" / "native-pm1").mkdir(parents=True)
    out_dir = tmp_path / "report"
    monkeypatch.setattr(
        "sys.argv",
        [
            "round2_report.py", "--cpu-root", str(cpu_root), "--run", "campaign",
            "--out-dir", str(out_dir), "--cells", "native-pm1",
        ],
    )
    assert report.main() == 0
    draft = (out_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "draft, campaign data, incomplete" in draft
    assert "pilot data" not in draft


def test_main_computes_a_real_missing_count_from_the_run_type(tmp_path, monkeypatch):
    cpu_root = tmp_path / "cpu"
    cell_dir = cpu_root / "pilot" / "native-pm1"
    # Only one record for this arm; the pilot expects 15 (5 models x 3 timing repetitions).
    _write_attempt(cell_dir, "model0", "cpu-sa", 512, 0, _cpu_record("native-pm1", "model0", "cpu-sa", 512))
    out_dir = tmp_path / "report"
    monkeypatch.setattr(
        "sys.argv",
        [
            "round2_report.py", "--cpu-root", str(cpu_root), "--run", "pilot", "--out-dir", str(out_dir),
            "--cells", "native-pm1", "--kernels", "cpu-sa", "--depths", "512",
        ],
    )
    assert report.main() == 0
    draft = (out_dir / "REPORT.md").read_text(encoding="utf-8")
    # The Solver settings table has a coincidentally similar-looking row (Kernels
    # observed | Sweep depths observed | Reads); search only the depth table's
    # own section so the two are never confused.
    depth_section = draft.split("## Depth, quality, and time")[1]
    row = next(line for line in depth_section.splitlines() if line.startswith("| `native-pm1` | `cpu-sa` | 512 |"))
    cells = [c.strip() for c in row.split("|")]
    assert cells[5] == "14"  # Missing: 15 expected, 1 observed


# ==================================================================== figures


def _svg_text_elements(svg: str):
    """Every visible ``<text>`` element as ``(x, y, content)``, excluding anything
    inside a ``<title>`` tooltip (which is invisible on a rendered image).
    """
    return [
        (float(m.group(1)), float(m.group(2)), m.group(3))
        for m in re.finditer(r'<text x="([\-\d.]+)" y="([\-\d.]+)"[^>]*>([^<]*)</text>', svg)
    ]


def _svg_circles(svg: str):
    return [
        (float(m.group(1)), float(m.group(2)))
        for m in re.finditer(r'<circle cx="([\-\d.]+)" cy="([\-\d.]+)"', svg)
    ]


def _canvas_size(svg: str):
    width_match = re.search(r'width="(\d+)"', svg)
    height_match = re.search(r'height="(\d+)"', svg)
    assert width_match is not None and height_match is not None
    return int(width_match.group(1)), int(height_match.group(1))


def test_portfolio_figure_keeps_the_sign_of_a_negative_objective():
    records = [{"n_assets": 28, "cardinality_k": 9, "beta_label": "beta-zero", "objective": -0.0002212}]
    svg = report.portfolio_figure(records)
    texts = " ".join(t for _, _, t in _svg_text_elements(svg))
    assert "-0.0002212" in texts  # the signed value, never abs()


def test_portfolio_figure_draws_negative_and_positive_bars_in_different_colors():
    records = [
        {"n_assets": 18, "cardinality_k": 6, "beta_label": "beta-zero", "objective": 0.0002},
        {"n_assets": 28, "cardinality_k": 9, "beta_label": "beta-zero", "objective": -0.0002},
    ]
    svg = report.portfolio_figure(records)
    assert report._POSITIVE_COLOR in svg
    assert report._NEGATIVE_COLOR in svg


def test_portfolio_repair_figure_shows_raw_feasible_counts_and_the_repair_outcome():
    records = [
        {
            "n_assets": 18, "cardinality_k": 6, "beta_label": "beta-zero",
            "raw_feasible_count": 3, "returned_reads": 500, "feasible": True,
        },
        {
            "n_assets": 28, "cardinality_k": 9, "beta_label": "beta-nonzero",
            "raw_feasible_count": 0, "returned_reads": 500, "feasible": False,
        },
    ]
    svg = report.portfolio_repair_figure(records)
    texts = " ".join(t for _, _, t in _svg_text_elements(svg))
    assert "repaired: feasible" in texts
    assert "repaired: infeasible" in texts
    assert "3 raw feasible reads" in texts


def test_portfolio_repair_figure_handles_no_data():
    svg = report.portfolio_repair_figure([])
    assert "No data available" in svg


def _energy_points(cell, kernel, sweeps_energy_wall):
    return [
        _cpu_record(cell, f"m{i}", kernel, sweeps, best_energy=energy, wall_s=wall)
        for i, (sweeps, energy, wall) in enumerate(sweeps_energy_wall)
    ]


def test_quality_time_figure_prints_visible_axis_titles_and_tick_values():
    records = _energy_points("native-pm1", "cpu-sa", [(512, -100.0, 1.0), (2048, -110.0, 4.0)])
    svg = report.quality_time_figure("native-pm1", records, ["cpu-sa"], [512, 2048])
    texts = [t for _, _, t in _svg_text_elements(svg)]
    joined = " ".join(texts)
    assert "wall time" in joined.lower()
    assert "best energy" in joined.lower()
    # The (padded) tick values are visible, printed at axis precision, not just
    # implied by point position (see the dedicated intermediate-tick tests for
    # the exact set fix round 2 requires).
    y_low, y_high = report._pad_range(-110.0, -100.0)
    assert any(report._tick_text(t) in texts for t in report._axis_ticks(y_low, y_high, count=5))


def test_quality_time_figure_prints_the_sample_count_outside_any_tooltip():
    records = _energy_points("native-pm1", "cpu-sa", [(512, -100.0, 1.0)]) * 15  # n=15, same arm
    svg = report.quality_time_figure("native-pm1", records, ["cpu-sa"], [512])
    texts = [t for _, _, t in _svg_text_elements(svg)]
    assert any("n=15" in t for t in texts)


def test_quality_time_figure_keeps_points_clear_of_the_axis_lines():
    records = _energy_points(
        "native-pm1", "cpu-sa", [(512, -100.0, 1.0), (2048, -100.0, 1.0)],
    )  # two points with the SAME energy and wall time: a zero-span edge case
    svg = report.quality_time_figure("native-pm1", records, ["cpu-sa"], [512, 2048])
    width, height = _canvas_size(svg)
    circles = _svg_circles(svg)
    assert circles
    for cx, cy in circles:
        assert 0 < cx < width
        assert 0 < cy < height
    # Not sitting exactly on the left axis line (a real defect the review found:
    # a point at the data extreme landing exactly at x == margin).
    x_values = sorted({cx for cx, _ in circles})
    assert x_values[0] > 70 + 1  # margin is 70; padding must move it clear


def test_quality_time_figure_wraps_a_long_caption_within_the_canvas():
    svg = report.quality_time_figure("native-pm1", [], ["cpu-sa"], [512])
    width, _ = _canvas_size(svg)
    for x, _, text in _svg_text_elements(svg):
        # A rough character-width estimate (matching round2_report's own
        # _text_width_estimate ratio) must keep every left-anchored text
        # element from running past the right edge of the canvas.
        if not text:
            continue
        estimated_end = x + len(text) * 12 * 0.6
        assert estimated_end <= width, f"text {text!r} at x={x} overflows a {width}px canvas"


def test_quality_time_figure_handles_no_completed_points():
    svg = report.quality_time_figure("native-pm1", [], ["cpu-sa"], [512])
    assert "No completed arms to plot" in svg


# ============================================== fix round 2: figure defects


def test_quality_time_figure_labels_each_point_with_its_sweep_count():
    records = _energy_points("native-pm1", "cpu-sa", [(512, -100.0, 1.0), (2048, -110.0, 4.0)])
    svg = report.quality_time_figure("native-pm1", records, ["cpu-sa"], [512, 2048])
    texts = [t for _, _, t in _svg_text_elements(svg)]
    assert any("512 sw" in t for t in texts)
    assert any("2048 sw" in t for t in texts)


def test_quality_time_figure_prints_intermediate_y_axis_ticks():
    records = _energy_points("native-pm1", "cpu-sa", [(512, -14270.0, 0.136), (2048, -14364.0, 7.05)])
    svg = report.quality_time_figure("native-pm1", records, ["cpu-sa"], [512, 2048])
    texts = [t for _, _, t in _svg_text_elements(svg)]
    y_low, y_high = report._pad_range(-14364.0, -14270.0)
    y_ticks = report._axis_ticks(y_low, y_high, count=5)
    y_texts = {report._tick_text(t) for t in y_ticks}
    assert len(y_texts) >= 4  # distinct at this precision, not just the two extremes
    for text in y_texts:
        assert text in texts


def test_quality_time_figure_prints_intermediate_x_axis_ticks():
    records = _energy_points("native-pm1", "cpu-sa", [(512, -100.0, 0.1), (2048, -110.0, 10.0)])
    svg = report.quality_time_figure("native-pm1", records, ["cpu-sa"], [512, 2048])
    texts = [t for _, _, t in _svg_text_elements(svg)]
    log_low, log_high = report._pad_range(math.log10(0.1), math.log10(10.0))
    x_ticks = [10 ** t for t in report._axis_ticks(log_low, log_high, count=5)]
    x_texts = {report._tick_text(t) for t in x_ticks}
    assert len(x_texts) >= 4
    for text in x_texts:
        assert text in texts


def test_axis_ticks_always_includes_both_endpoints():
    ticks = report._axis_ticks(0.0, 8.0, count=5)
    assert ticks[0] == 0.0
    assert ticks[-1] == 8.0
    assert len(ticks) == 5


def test_portfolio_figure_negative_label_never_overlaps_the_category_column():
    # The controller's rendered defect: "beta-zero-0.0002212 objective" reads as
    # one run of text because the negative label's anchor="end" left edge, at a
    # near-max bar width, extended back into the category column.
    records = [
        {"n_assets": 18, "cardinality_k": 6, "beta_label": "beta-zero", "objective": 0.0001127},
        {"n_assets": 28, "cardinality_k": 9, "beta_label": "beta-zero", "objective": -0.0002212},
    ]
    svg = report.portfolio_figure(records)
    elements = _svg_text_elements(svg)
    category_x, _, category_text = next(e for e in elements if e[2] == "n=28 k=9 beta-zero")
    neg_x, _, neg_text = next(e for e in elements if e[2].startswith("-0.0002212"))
    category_right_edge = category_x + report._text_width_estimate(category_text)
    negative_left_edge = neg_x - report._text_width_estimate(neg_text)  # anchor="end": x is the right edge
    assert negative_left_edge > category_right_edge


def test_physical_scale_figure_subtitle_wraps_within_the_canvas():
    proposal = {
        "arms": [{"regime": "diamond-pm1", "anneal_us": 80, "captures": 36, "reads_per_capture": 64}],
        "approval": "pending",
        "physical_scales": "50%,75%,100% of audited legal scale ceiling; logical coefficients preserved",
    }
    svg = report.physical_scale_figure(proposal)
    width, _ = _canvas_size(svg)
    for x, _, text in _svg_text_elements(svg):
        if not text:
            continue
        assert x + len(text) * 12 * 0.6 <= width, f"text {text!r} at x={x} overflows a {width}px canvas"


def test_physical_scale_figure_never_claims_no_capture_has_run_once_one_has(tmp_path):
    # A real render-inspection defect (final fix pass): once real captures
    # exist for a cell, this figure's own fixed subtitle text was still false.
    proposal = {
        "arms": [{"regime": "diamond-pm1", "anneal_us": 80, "captures": 36, "reads_per_capture": 64}],
        "approval": "approved",
    }
    svg = report.physical_scale_figure(proposal, captured_regimes={"diamond-pm1"})
    assert "No physical capture has run yet" not in svg


def test_portfolio_repair_figure_subtitle_wraps_within_the_canvas():
    records = [
        {
            "n_assets": 18, "cardinality_k": 6, "beta_label": "beta-zero",
            "raw_feasible_count": 3, "returned_reads": 500, "feasible": True,
        },
    ]
    svg = report.portfolio_repair_figure(records)
    width, _ = _canvas_size(svg)
    for x, _, text in _svg_text_elements(svg):
        if not text:
            continue
        assert x + len(text) * 12 * 0.6 <= width, f"text {text!r} at x={x} overflows a {width}px canvas"


def test_portfolio_figure_header_wraps_within_the_canvas():
    records = [{"n_assets": 18, "cardinality_k": 6, "beta_label": "beta-zero", "objective": 0.0001}]
    svg = report.portfolio_figure(records)
    width, _ = _canvas_size(svg)
    for x, _, text in _svg_text_elements(svg):
        if x != 10.0 or not text:
            continue
        assert x + len(text) * 12 * 0.6 <= width, f"text {text!r} at x={x} overflows a {width}px canvas"


def test_bar_chart_value_label_never_overlaps_a_long_category_label():
    # A real defect found by rendering portfolio-repair.svg: a long category
    # label (the repair-outcome suffix) plus a zero-width bar (raw_feasible_count
    # 0) put the value label's start right on top of the category text's tail.
    records = [
        {
            "n_assets": 18, "cardinality_k": 6, "beta_label": "beta-zero",
            "raw_feasible_count": 0, "returned_reads": 500, "feasible": True,
        },
    ]
    svg = report.portfolio_repair_figure(records)
    elements = _svg_text_elements(svg)
    category_x, _, category_text = next(e for e in elements if e[2].startswith("n=18 k=6 beta-zero"))
    value_x, _, value_text = next(e for e in elements if e[2] == "0 raw feasible reads")
    category_right_edge = category_x + report._text_width_estimate(category_text)
    assert value_x >= category_right_edge


# ==================================================== final fix pass: C1 QPU captures


def _write_qpu_capture(
    path: Path, *, cell, nonce, model_hash, capture_key,
    requested_scale, anneal_us, energies, mock=False, access_us=50_000,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    energies_arr = np.asarray(energies, dtype=np.float64)
    spins = np.ones((len(energies_arr), 4), dtype=np.int8)
    np.savez_compressed(
        path,
        spins=spins, energies=energies_arr,
        mock=np.bool_(mock), anneal_us=np.int64(anneal_us), requested_scale=np.float64(requested_scale),
        access_us=np.int64(access_us), capture_key=capture_key, model_hash=model_hash,
    )


def _manifest_job(cell, nonce, model_hash, capture_key, requested_scale, anneal_us):
    return {
        "cell": cell, "nonce": nonce, "model_hash": model_hash, "capture_key": capture_key,
        "requested_scale": requested_scale, "anneal_us": anneal_us,
    }


def test_load_qpu_captures_verifies_against_the_manifest(tmp_path):
    qpu_root = tmp_path / "qpu-physical-pilot"
    _write_qpu_capture(
        qpu_root / "diamond-pm1" / "scale-050" / "qpu-80" / "aa.npz",
        cell="diamond-pm1", nonce="aa", model_hash="mh-aa", capture_key="key-aa",
        requested_scale=0.5, anneal_us=80, energies=[-10.0, -9.0],
    )
    manifest = {"jobs": [_manifest_job("diamond-pm1", "aa", "mh-aa", "key-aa", 0.5, 80)]}
    captures = report.load_qpu_captures(qpu_root, manifest)
    assert len(captures) == 1
    cap = captures[0]
    assert cap["cell"] == "diamond-pm1"
    assert cap["model_hash"] == "mh-aa"
    assert cap["requested_scale"] == 0.5
    assert cap["anneal_us"] == 80
    assert cap["best_energy"] == -10.0
    assert cap["access_us"] == 50_000


def test_load_qpu_captures_rejects_a_mock_capture(tmp_path):
    qpu_root = tmp_path / "qpu-physical-pilot"
    _write_qpu_capture(
        qpu_root / "diamond-pm1" / "scale-050" / "qpu-80" / "aa.npz",
        cell="diamond-pm1", nonce="aa", model_hash="mh-aa", capture_key="key-aa",
        requested_scale=0.5, anneal_us=80, energies=[-10.0], mock=True,
    )
    manifest = {"jobs": [_manifest_job("diamond-pm1", "aa", "mh-aa", "key-aa", 0.5, 80)]}
    with pytest.raises(ValueError, match="mock"):
        report.load_qpu_captures(qpu_root, manifest)


def test_load_qpu_captures_rejects_a_capture_key_not_in_the_manifest(tmp_path):
    qpu_root = tmp_path / "qpu-physical-pilot"
    _write_qpu_capture(
        qpu_root / "diamond-pm1" / "scale-050" / "qpu-80" / "aa.npz",
        cell="diamond-pm1", nonce="aa", model_hash="mh-aa", capture_key="unknown-key",
        requested_scale=0.5, anneal_us=80, energies=[-10.0],
    )
    with pytest.raises(ValueError, match="capture_key"):
        report.load_qpu_captures(qpu_root, {"jobs": []})


def test_load_qpu_captures_rejects_a_model_hash_mismatch(tmp_path):
    qpu_root = tmp_path / "qpu-physical-pilot"
    _write_qpu_capture(
        qpu_root / "diamond-pm1" / "scale-050" / "qpu-80" / "aa.npz",
        cell="diamond-pm1", nonce="aa", model_hash="wrong-hash", capture_key="key-aa",
        requested_scale=0.5, anneal_us=80, energies=[-10.0],
    )
    manifest = {"jobs": [_manifest_job("diamond-pm1", "aa", "mh-aa", "key-aa", 0.5, 80)]}
    with pytest.raises(ValueError, match="model_hash"):
        report.load_qpu_captures(qpu_root, manifest)


def test_pair_qpu_cpu_by_model_matches_on_model_hash_and_excludes_failed_cpu_records():
    captures = [
        {"model_hash": "mh-a", "best_energy": -10.0},
        {"model_hash": "mh-b", "best_energy": -8.0},
    ]
    cpu_ok = _cpu_record("diamond-pm1", "aa", "cpu-sa", 512, best_energy=-9.5)
    cpu_ok["model_hash"] = "mh-a"
    cpu_failed = _cpu_record("diamond-pm1", "bb", "cpu-sa", 512, exit_ok=False, best_energy=None)
    cpu_failed["model_hash"] = "mh-b"
    pairs = report.pair_qpu_cpu_by_model(captures, [cpu_ok, cpu_failed])
    assert pairs == [("mh-a", -10.0, -9.5)]


def test_qpu_outcome_table_shows_real_pairs_when_captures_exist():
    cpu_a = _cpu_record("diamond-pm1", "aa", "cpu-sa", 512, best_energy=-9.9)
    cpu_a["model_hash"] = "mh-a"
    cpu_b = _cpu_record("diamond-pm1", "bb", "cpu-sa", 512, best_energy=-7.0)
    cpu_b["model_hash"] = "mh-b"
    captures_by_arm = {
        ("diamond-pm1", 0.5, 80): [
            {"model_hash": "mh-a", "best_energy": -10.0},
            {"model_hash": "mh-b", "best_energy": -7.0},
        ],
    }
    records_by_cell = {"diamond-pm1": [cpu_a, cpu_b]}
    lines = report.qpu_outcome_table(["diamond-pm1"], captures_by_arm, records_by_cell, ["cpu-sa"], [512])
    text = "\n".join(lines)
    assert "| `diamond-pm1` | 0.5 | 80 | `cpu-sa` | 512 | 2 |" in text
    assert "No Round 2 QPU capture" not in text
    assert "pending approval" not in text


def test_qpu_outcome_table_states_no_comparable_pairs_when_none_exist():
    lines = report.qpu_outcome_table(["native-pm1"], {}, {"native-pm1": []}, ["cpu-sa"], [512])
    assert "No comparable pairs" in "\n".join(lines)


def test_paired_gaps_table_computes_a_real_bootstrap_interval_when_pairs_exist():
    captures_by_arm = {
        ("diamond-pm1", 0.5, 80): [{"model_hash": f"mh-{i}", "best_energy": -10.0 - i} for i in range(6)],
    }
    records = []
    for i in range(6):
        record = _cpu_record("diamond-pm1", f"n{i}", "cpu-sa", 512, best_energy=-9.5 - i)
        record["model_hash"] = f"mh-{i}"
        records.append(record)
    records_by_cell = {"diamond-pm1": records}
    lines = report.paired_gaps_table(["diamond-pm1"], captures_by_arm, records_by_cell, ["cpu-sa"], [512])
    row = next(line for line in lines if line.startswith("| `diamond-pm1` | 0.5 | 80 | `cpu-sa` | 512 |"))
    cells = [c.strip() for c in row.split("|")]
    assert cells[6] == "6"  # Groups: every model is an independent draw
    assert cells[10] == "no"  # 6 groups clears MIN_INDEPENDENT_GROUPS (5): a real interval, not descriptive


def test_paired_gaps_table_states_no_comparable_pairs_when_none_exist():
    lines = report.paired_gaps_table(["native-pm1"], {}, {"native-pm1": []}, ["cpu-sa"], [512])
    assert "No comparable pairs" in "\n".join(lines)


def test_reconcile_spend_compares_the_charged_total_against_capture_access_us(tmp_path):
    from quip_miner_dwave import regime_io

    qpu_root = tmp_path / "qpu-physical-pilot"
    regime_io.append_spend(qpu_root, regime_io.SUBMIT, "diamond-pm1", "aa", 80, 50_000)
    regime_io.append_spend(qpu_root, regime_io.CHARGE, "diamond-pm1", "aa", 80, 48_000)
    result = report.reconcile_spend(qpu_root, [{"access_us": 48_000}])
    assert result["submits"] == 1
    assert result["charges"] == 1
    assert result["open_jobs"] == 0
    assert result["charged_us"] == 48_000
    assert result["total_access_us"] == 48_000
    assert result["reconciled"] is True


def test_reconcile_spend_flags_a_mismatch_between_charges_and_captures(tmp_path):
    from quip_miner_dwave import regime_io

    qpu_root = tmp_path / "qpu-physical-pilot"
    regime_io.append_spend(qpu_root, regime_io.SUBMIT, "diamond-pm1", "aa", 80, 50_000)
    regime_io.append_spend(qpu_root, regime_io.CHARGE, "diamond-pm1", "aa", 80, 48_000)
    # A capture whose own access_us disagrees with the journaled charge.
    result = report.reconcile_spend(qpu_root, [{"access_us": 47_000}])
    assert result["reconciled"] is False


def test_physical_scale_effect_figure_plots_per_model_points():
    captures = [
        {"model_hash": "mh-a", "requested_scale": 0.5, "anneal_us": 80, "best_energy": -10.0},
        {"model_hash": "mh-a", "requested_scale": 1.0, "anneal_us": 80, "best_energy": -12.0},
        {"model_hash": "mh-b", "requested_scale": 0.5, "anneal_us": 400, "best_energy": -9.0},
    ]
    svg = report.physical_scale_effect_figure("diamond-pm1", captures)
    assert svg.count("<circle") == 3
    assert "no regime verdict" in svg.lower()


def test_physical_scale_effect_figure_handles_no_captures():
    svg = report.physical_scale_effect_figure("native-pm1", [])
    assert "No physical-scale capture" in svg


# ==================================================== final fix pass: M2 kernel gaps


def test_cpu_kernel_gap_table_pairs_by_model_not_by_median():
    # Two models: msa-f64 wins on one, loses on the other. A median-of-medians
    # comparison (the old "Depth, quality, and time" table) cannot see this;
    # a per-model pairing can (review, M2).
    records = [
        _cpu_record("diamond-pm1", "aa", "cpu-sa", 512, best_energy=-10.0),
        _cpu_record("diamond-pm1", "aa", "cpu-msa-f64", 512, best_energy=-10.5),
        _cpu_record("diamond-pm1", "bb", "cpu-sa", 512, best_energy=-8.0),
        _cpu_record("diamond-pm1", "bb", "cpu-msa-f64", 512, best_energy=-7.5),
    ]
    for record, model_hash in zip(records, ["mh-a", "mh-a", "mh-b", "mh-b"]):
        record["model_hash"] = model_hash
    lines = report.cpu_kernel_gap_table({"diamond-pm1": records}, kernels=("cpu-msa-f64",), depths=(512,))
    text = "\n".join(lines)
    assert "| `diamond-pm1` | `cpu-msa-f64` vs `cpu-sa` | 512 | 2 | 1/1/0 |" in text


def test_cpu_kernel_gap_table_excludes_models_missing_from_either_kernel():
    records = [
        _cpu_record("diamond-pm1", "aa", "cpu-sa", 512, best_energy=-10.0),
        _cpu_record("diamond-pm1", "aa", "cpu-msa-f64", 512, best_energy=-10.5),
        _cpu_record("diamond-pm1", "bb", "cpu-sa", 512, best_energy=-8.0),
        # "bb" has no cpu-msa-f64 record at all.
    ]
    records[0]["model_hash"] = "mh-a"
    records[1]["model_hash"] = "mh-a"
    records[2]["model_hash"] = "mh-b"
    lines = report.cpu_kernel_gap_table({"diamond-pm1": records}, kernels=("cpu-msa-f64",), depths=(512,))
    text = "\n".join(lines)
    assert "| `diamond-pm1` | `cpu-msa-f64` vs `cpu-sa` | 512 | 1 |" in text


def test_cpu_kernel_gap_table_states_no_comparable_pairs_when_none_exist():
    lines = report.cpu_kernel_gap_table({"native-pm1": []}, kernels=("cpu-msa-f64",), depths=(512,))
    assert "No comparable pairs" in "\n".join(lines)
