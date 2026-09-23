"""Tests for the Round 2 controlled CPU comparison runner.

Task 8 of docs/superpowers/plans/2026-09-22-regime-search-round2.md: the
controlled CPU comparison runner, the portfolio deadline arm, and the
five-model CPU pilot. See ``quip_miner_dwave.round2_runner`` for the module
under test. Tests that need a solver double never call the real ``quip_msa``
kernel: they exercise the runner's own invariant checks (wrong observed
kernel, nonfinite score, unsupported unit arm) against a scripted fake, which
is the only way to force those specific failure paths deterministically.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from quip_miner_dwave import regimes, round2_io

import round2_runner as runner

BUNDLE_EDGES = np.array([[0, 1], [1, 2]], dtype=np.int64)


def _nonce(i: int) -> str:
    return format(i, "064x")


def _write_bundle(bundles_root: Path, cell: str, nonce: str, h=None, j=None) -> str:
    h = np.zeros(3) if h is None else h
    j = np.array([1.0, -1.0]) if j is None else j
    manifest = {
        "cell": cell,
        "nonce": nonce,
        "identity": {"topology_hash": "topo-fixture", "model_order_hash": "order-fixture"},
        "offset": 0.0,
    }
    return round2_io.write_bundle(bundles_root / cell / nonce, manifest, {"h": h, "edges": BUNDLE_EDGES, "j": j})


def _index(bundles_root: Path, rows) -> dict:
    return {"bundles_root": str(bundles_root), "models": len(rows), "rows": rows}


# --------------------------------------------------------------- deadline_status


def test_late_good_answers_are_timeouts():
    assert runner.deadline_status(10.001, 10.0, True) == "timeout"
    assert runner.deadline_status(9.999, 10.0, True) == "completed"


def test_a_failed_run_is_failed_even_if_it_would_have_been_on_time():
    assert runner.deadline_status(1.0, 10.0, False) == "failed"


def test_exactly_on_the_deadline_is_completed():
    assert runner.deadline_status(10.0, 10.0, True) == "completed"


# ----------------------------------------------------------------- host contamination


def test_contamination_flags_high_loadavg_but_never_drops_the_sample():
    before = runner.HostSample(loadavg_1m=99.0, loadavg_5m=1.0, loadavg_15m=1.0, governor="performance",
                                sibling_cpu_times=(0.0, 100.0))
    after = runner.HostSample(loadavg_1m=1.0, loadavg_5m=1.0, loadavg_15m=1.0, governor="performance",
                               sibling_cpu_times=(1.0, 200.0))
    verdict = runner.check_contamination(before, after)
    assert verdict.contaminated
    assert any("loadavg_1m before" in reason for reason in verdict.reasons)
    # the thresholds used are recorded, not just implied
    assert verdict.thresholds["loadavg_1m_max"] == runner.LOADAVG_1M_CONTAMINATION_THRESHOLD


def test_contamination_flags_a_busy_sibling():
    before = runner.HostSample(0.1, 0.1, 0.1, "performance", sibling_cpu_times=(0.0, 100.0))
    after = runner.HostSample(0.1, 0.1, 0.1, "performance", sibling_cpu_times=(60.0, 200.0))
    verdict = runner.check_contamination(before, after)
    assert verdict.contaminated
    assert verdict.sibling_busy_fraction == pytest.approx(0.6)


def test_a_quiet_host_is_not_contaminated():
    before = runner.HostSample(0.1, 0.1, 0.1, "performance", sibling_cpu_times=(0.0, 100.0))
    after = runner.HostSample(0.2, 0.1, 0.1, "performance", sibling_cpu_times=(1.0, 200.0))
    verdict = runner.check_contamination(before, after)
    assert not verdict.contaminated
    assert verdict.reasons == ()


def test_busy_fraction_is_none_for_a_run_too_short_to_tick_the_jiffy_clock():
    assert runner.busy_fraction((0.0, 100.0), (0.0, 100.0)) is None


# --------------------------------------------------------------------- identity


def test_timing_repetitions_of_the_same_arm_reuse_the_seed():
    seed_a, hash_a = runner.seed_for("model-hash", "cpu-sa", 512, 64, "timing")
    seed_b, hash_b = runner.seed_for("model-hash", "cpu-sa", 512, 64, "timing")
    assert (seed_a, hash_a) == (seed_b, hash_b)


def test_a_different_variant_draws_a_different_seed():
    seed_timing, _ = runner.seed_for("model-hash", "cpu-sa", 512, 64, "timing")
    seed_quality, _ = runner.seed_for("model-hash", "cpu-sa", 512, 64, "quality-0")
    assert seed_timing != seed_quality


def test_run_key_changes_with_every_identity_field(tmp_path):
    base = runner.CpuJob(
        cell="native-pm1", nonce=_nonce(1), kernel="cpu-sa", sweeps=512, reads=64,
        repetition_id=0, repetition_kind="timing", variant="timing", seed=1, seed_input_hash="h",
    )
    changed = runner.CpuJob(
        cell="native-pm1", nonce=_nonce(1), kernel="cpu-msa-f64", sweeps=512, reads=64,
        repetition_id=0, repetition_kind="timing", variant="timing", seed=1, seed_input_hash="h",
    )
    assert base.run_key(tmp_path) != changed.run_key(tmp_path)
    assert base.run_key(tmp_path) == base.run_key(tmp_path)


def test_run_key_is_sensitive_to_the_bundles_root(tmp_path):
    job = runner.CpuJob(
        cell="native-pm1", nonce=_nonce(1), kernel="cpu-sa", sweeps=512, reads=64,
        repetition_id=0, repetition_kind="timing", variant="timing", seed=1, seed_input_hash="h",
    )
    assert job.run_key(tmp_path / "a") != job.run_key(tmp_path / "b")


def test_job_round_trips_through_to_dict_and_from_dict():
    job = runner.CpuJob(
        cell="native-pm1", nonce=_nonce(1), kernel="cpu-sa", sweeps=512, reads=64,
        repetition_id=1, repetition_kind="quality", variant="quality-1", seed=7, seed_input_hash="h",
    )
    assert runner.CpuJob.from_dict(job.to_dict()) == job


# ------------------------------------------------------------------ ordering


def _five_model_index(bundles_root, cell, count=5):
    rows = []
    for i in range(count):
        nonce = _nonce(i)
        model_hash = _write_bundle(bundles_root, cell, nonce)
        rows.append({"cell": cell, "nonce": nonce, "model_hash": model_hash, "unit_kernel_input_hash": None})
    return rows


def test_pilot_job_order_is_deterministic_across_builds(tmp_path):
    rows = _five_model_index(tmp_path, "native-pm1")
    index = _index(tmp_path, rows)
    first = runner.build_pilot_jobs(index, ["native-pm1"])
    second = runner.build_pilot_jobs(index, ["native-pm1"])
    assert [job.run_key(tmp_path) for job in first] == [job.run_key(tmp_path) for job in second]


def test_pilot_job_order_is_not_the_trivial_nested_loop_order(tmp_path):
    rows = _five_model_index(tmp_path, "native-pm1")
    index = _index(tmp_path, rows)
    jobs = runner.build_pilot_jobs(index, ["native-pm1"])
    trivial = sorted(jobs, key=lambda j: (j.nonce, j.sweeps, j.kernel, j.repetition_id))
    assert [j.nonce for j in jobs] != [j.nonce for j in trivial] or (
        [j.kernel for j in jobs] != [j.kernel for j in trivial]
    )


def test_pilot_covers_every_model_sweep_kernel_and_repetition(tmp_path):
    rows = _five_model_index(tmp_path, "native-pm1")
    index = _index(tmp_path, rows)
    jobs = runner.build_pilot_jobs(index, ["native-pm1"])
    assert len(jobs) == (
        runner.PILOT_MODELS_PER_CELL * len(runner.PILOT_SWEEP_DEPTHS)
        * len(runner.CONTROLLED_KERNELS) * runner.PILOT_TIMING_REPS
    )
    nonces = {job.nonce for job in jobs}
    assert nonces == {row["nonce"] for row in rows}


def test_pilot_takes_only_the_first_sorted_nonces_per_cell(tmp_path):
    rows = _five_model_index(tmp_path, "native-pm1", count=7)
    index = _index(tmp_path, rows)
    jobs = runner.build_pilot_jobs(index, ["native-pm1"])
    expected = sorted(row["nonce"] for row in rows)[: runner.PILOT_MODELS_PER_CELL]
    assert {job.nonce for job in jobs} == set(expected)


def test_campaign_job_count(tmp_path):
    rows = _five_model_index(tmp_path, "native-pm1", count=3)
    index = _index(tmp_path, rows)
    jobs = runner.build_campaign_jobs(index, ["native-pm1"])
    assert len(jobs) == 3 * len(runner.SWEEP_DEPTHS) * len(runner.CONTROLLED_KERNELS)


# ---------------------------------------------------------------- execute_cpu_job


class _FakeSampler:
    def __init__(self, *, observed_kernel=None, energies=None, raise_error=None, workspace_bytes=None):
        self._observed_kernel = observed_kernel
        self._energies = energies
        self._raise_error = raise_error
        self._workspace_bytes = workspace_bytes

    def sample_research(self, h, edges, j, *, kernel, num_sweeps, num_reads, seed, beta_range):
        if self._raise_error is not None:
            raise self._raise_error
        spins = np.ones((num_reads, len(h)), dtype=np.int8)
        energies = self._energies if self._energies is not None else regimes.energy(spins, h, edges, j)
        meta = {
            "requested_kernel": kernel,
            "observed_kernel": self._observed_kernel or kernel,
            "representation": "fake-representation",
            "rng_scheme": "fake-rng",
            "seeded_reads": 0,
            "workspace_bytes": self._workspace_bytes,
        }
        return spins, np.asarray(energies, dtype=np.float64), meta


def _fake_msa_module(sampler: _FakeSampler):
    module = types.SimpleNamespace()
    module.Msa = lambda: sampler
    module.default_beta_range = lambda h, edges, j: (0.1, 5.0)
    return module


def _job(cell, nonce, kernel="cpu-sa", sweeps=8, reads=4):
    seed, seed_hash = runner.seed_for("m", kernel, sweeps, reads, "timing")
    return runner.CpuJob(
        cell=cell, nonce=nonce, kernel=kernel, sweeps=sweeps, reads=reads,
        repetition_id=0, repetition_kind="timing", variant="timing", seed=seed, seed_input_hash=seed_hash,
    )


def test_a_completed_run_is_rescored_independently(tmp_path, monkeypatch):
    _write_bundle(tmp_path, "native-pm1", _nonce(0))
    monkeypatch.setattr(runner, "_msa", lambda: _fake_msa_module(_FakeSampler()))
    record, samples = runner.execute_cpu_job(_job("native-pm1", _nonce(0)), tmp_path)
    assert record["exit_ok"] is True
    assert record["unsupported"] is False
    assert record["observed_kernel"] == "cpu-sa"
    assert samples is not None
    assert samples["spins"].shape == (4, 3)


def test_wrong_observed_kernel_is_never_recorded_as_a_completed_run(tmp_path, monkeypatch):
    _write_bundle(tmp_path, "native-pm1", _nonce(0))
    monkeypatch.setattr(
        runner, "_msa", lambda: _fake_msa_module(_FakeSampler(observed_kernel="cpu-sa"))
    )
    with pytest.raises(runner.RunnerError, match="observed"):
        runner.execute_cpu_job(_job("native-pm1", _nonce(0), kernel="cpu-msa-f64"), tmp_path)


def test_nonfinite_score_is_rejected(tmp_path, monkeypatch):
    _write_bundle(tmp_path, "native-pm1", _nonce(0))
    monkeypatch.setattr(
        runner, "_msa",
        lambda: _fake_msa_module(_FakeSampler(energies=np.array([float("nan"), 1.0, 2.0, 3.0]))),
    )
    with pytest.raises(runner.RunnerError, match="nonfinite"):
        runner.execute_cpu_job(_job("native-pm1", _nonce(0)), tmp_path)


def test_unsupported_unit_kernel_is_skipped_not_substituted_with_sa(tmp_path, monkeypatch):
    _write_bundle(tmp_path, "native-125", _nonce(0))
    monkeypatch.setattr(
        runner, "_msa",
        lambda: _fake_msa_module(_FakeSampler(raise_error=ValueError("the unit kernel takes couplings in {-1,0,1}"))),
    )
    record, samples = runner.execute_cpu_job(_job("native-125", _nonce(0), kernel="cpu-msa-unit"), tmp_path)
    assert record["unsupported"] is True
    assert record["exit_ok"] is True
    assert "unit kernel" in record["unsupported_reason"]
    assert samples is None


def test_a_real_value_error_on_a_non_unit_kernel_is_a_failure_not_unsupported(tmp_path, monkeypatch):
    _write_bundle(tmp_path, "native-pm1", _nonce(0))
    monkeypatch.setattr(
        runner, "_msa", lambda: _fake_msa_module(_FakeSampler(raise_error=ValueError("boom")))
    )
    record, samples = runner.execute_cpu_job(_job("native-pm1", _nonce(0), kernel="cpu-sa"), tmp_path)
    assert record["exit_ok"] is False
    assert record["unsupported"] is False
    assert samples is None


# ------------------------------------------------- cubic-dimer-pm1 unit-kernel reconstruction


def test_kernel_input_for_is_identity_off_cubic_dimer():
    h = np.zeros(3)
    edges = np.array([[0, 1], [1, 2]])
    j = np.array([-0.5, 1.0])
    result = runner.kernel_input_for("native-pm1", "cpu-msa-unit", h, edges, j, (0.2, 4.0))
    assert result["energy_scale"] == 1.0
    assert result["beta_range"] == (0.2, 4.0)
    np.testing.assert_array_equal(result["edges"], edges)
    np.testing.assert_array_equal(result["j"], j)


def test_kernel_input_for_is_identity_on_cubic_dimer_for_non_unit_kernels():
    h = np.zeros(3)
    edges = np.array([[0, 1], [1, 2]])
    j = np.array([-0.5, 1.0])
    result = runner.kernel_input_for("cubic-dimer-pm1", "cpu-msa-f64", h, edges, j, (0.2, 4.0))
    assert result["energy_scale"] == 1.0
    np.testing.assert_array_equal(result["j"], j)


def test_kernel_input_for_reconstructs_cubic_dimer_repeated_unit_bonds():
    h = np.zeros(3)
    edges = np.array([[0, 1], [1, 2]])
    j = np.array([-0.5, 1.0])
    result = runner.kernel_input_for("cubic-dimer-pm1", "cpu-msa-unit", h, edges, j, (0.2, 4.0))
    assert result["energy_scale"] == 0.5
    assert result["beta_range"] == (0.1, 2.0)
    np.testing.assert_array_equal(result["edges"], [[0, 1], [1, 2], [1, 2]])
    np.testing.assert_array_equal(result["j"], [-1.0, 1.0, 1.0])
    np.testing.assert_array_equal(result["h"], np.zeros(3))


def test_kernel_input_for_rejects_a_non_half_unit_coupling():
    h = np.zeros(2)
    edges = np.array([[0, 1]])
    j = np.array([0.37])
    with pytest.raises(runner.RunnerError, match="not a multiple"):
        runner.kernel_input_for("cubic-dimer-pm1", "cpu-msa-unit", h, edges, j, (0.2, 4.0))


def _cubic_dimer_bundle(tmp_path, nonce, *, unit_hash: object = "unset"):
    h = np.zeros(3)
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    j = np.array([-0.5, 1.0])
    manifest = {
        "cell": "cubic-dimer-pm1", "nonce": nonce,
        "identity": {"topology_hash": "t", "model_order_hash": "o"}, "offset": 0.0,
    }
    if unit_hash != "unset":
        manifest["unit_kernel_input_hash"] = unit_hash
    round2_io.write_bundle(tmp_path / "cubic-dimer-pm1" / nonce, manifest, {"h": h, "edges": edges, "j": j})
    return h, edges, j


def test_execute_cpu_job_succeeds_on_cubic_dimer_unit_kernel_via_reconstruction(tmp_path, monkeypatch):
    _cubic_dimer_bundle(tmp_path, _nonce(0))
    monkeypatch.setattr(runner, "_msa", lambda: _fake_msa_module(_FakeSampler()))
    job = _job("cubic-dimer-pm1", _nonce(0), kernel="cpu-msa-unit")
    record, samples = runner.execute_cpu_job(job, tmp_path)
    assert record["exit_ok"] is True
    assert record["unsupported"] is False
    assert record["kernel_energy_scale"] == 0.5
    assert samples is not None


def test_execute_cpu_job_rejects_a_reconstruction_that_disagrees_with_the_bundle(tmp_path, monkeypatch):
    _cubic_dimer_bundle(tmp_path, _nonce(0), unit_hash="not-the-real-hash")
    monkeypatch.setattr(runner, "_msa", lambda: _fake_msa_module(_FakeSampler()))
    job = _job("cubic-dimer-pm1", _nonce(0), kernel="cpu-msa-unit")
    with pytest.raises(runner.RunnerError, match="unit_kernel_input_hash|does not match"):
        runner.execute_cpu_job(job, tmp_path)


def test_execute_cpu_job_accepts_a_reconstruction_that_matches_the_bundle(tmp_path, monkeypatch):
    # the hash a real Task 5 export would have recorded for this exact model, computed
    # independently of execute_cpu_job's own call to kernel_input_for
    h = np.zeros(3)
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    j = np.array([-0.5, 1.0])
    correct_hash = runner.kernel_input_for("cubic-dimer-pm1", "cpu-msa-unit", h, edges, j, (1.0, 1.0))[
        "input_hash"
    ]
    _cubic_dimer_bundle(tmp_path, _nonce(0), unit_hash=correct_hash)
    monkeypatch.setattr(runner, "_msa", lambda: _fake_msa_module(_FakeSampler()))
    job = _job("cubic-dimer-pm1", _nonce(0), kernel="cpu-msa-unit")
    record, samples = runner.execute_cpu_job(job, tmp_path)
    assert record["exit_ok"] is True
    assert record["unsupported"] is False


# ------------------------------------------------------------- QPU deadline record


def test_qpu_deadline_record_separates_on_time_from_charge():
    record = runner.build_qpu_deadline_record(
        elapsed_s=12.0, deadline_s=10.0, exit_ok=True, charge_known=False,
    )
    assert record["on_time"]["status"] == "timeout"
    assert record["charge"]["known"] is False
    assert record["charge"]["access_us"] is None


def test_qpu_deadline_record_rejects_an_access_time_with_no_known_charge():
    with pytest.raises(ValueError, match="charge_known"):
        runner.build_qpu_deadline_record(
            elapsed_s=1.0, deadline_s=10.0, exit_ok=True, charge_known=False, access_us=0,
        )


def test_qpu_deadline_record_requires_reconciliation_time_once_charged():
    with pytest.raises(ValueError, match="reconciled_at"):
        runner.build_qpu_deadline_record(
            elapsed_s=1.0, deadline_s=10.0, exit_ok=True, charge_known=True, access_us=500,
        )


def test_qpu_deadline_record_a_timeout_never_implies_zero_spend():
    # exactly the scenario the brief calls out: a wall timeout with the charge
    # settled later at a nonzero value must be representable and consistent.
    record = runner.build_qpu_deadline_record(
        elapsed_s=15.0, deadline_s=10.0, exit_ok=True, charge_known=True,
        access_us=80000, reconciled_at="2026-09-24T00:00:00+00:00",
    )
    assert record["on_time"]["status"] == "timeout"
    assert record["charge"]["access_us"] == 80000


def test_qpu_deadline_record_a_hard_kill_before_any_answer_cannot_be_marked_ok():
    with pytest.raises(ValueError, match="exit_ok"):
        runner.build_qpu_deadline_record(elapsed_s=None, deadline_s=10.0, exit_ok=True, charge_known=False)


def test_qpu_deadline_record_a_hard_kill_with_no_elapsed_time_is_failed():
    record = runner.build_qpu_deadline_record(elapsed_s=None, deadline_s=10.0, exit_ok=False, charge_known=False)
    assert record["on_time"]["status"] == "failed"


# ----------------------------------------------------------------- campaign estimate


def test_campaign_estimate_scales_linearly_from_the_nearest_measured_depth():
    pilot = {("cpu-sa", 512): 1.0, ("cpu-sa", 2048): 4.0}
    estimate = runner.estimate_campaign(
        pilot, cells=1, models_per_cell=1, depths=(512, 2048, 8192), kernels=("cpu-sa",),
    )
    # 8192 is nearest to 2048 among measured depths; scaled linearly: 4.0 * (8192/2048) = 16.0
    assert estimate["per_kernel_core_s"]["cpu-sa"] == pytest.approx(1.0 + 4.0 + 16.0)


def test_campaign_estimate_drops_unsupported_unit_cells_from_the_total():
    pilot = {(runner.UNIT_KERNEL, 512): 1.0}
    with_all = runner.estimate_campaign(
        pilot, cells=5, models_per_cell=1, depths=(512,), kernels=(runner.UNIT_KERNEL,),
        unsupported_kernel_cells=0,
    )
    with_one_unsupported = runner.estimate_campaign(
        pilot, cells=5, models_per_cell=1, depths=(512,), kernels=(runner.UNIT_KERNEL,),
        unsupported_kernel_cells=1,
    )
    assert with_one_unsupported["total_core_s"] < with_all["total_core_s"]


# ---------------------------------------------------------------- qpu_seed_lanes


def test_qpu_seed_lanes_scales_milli_cells_before_comparing_to_the_saved_capture(tmp_path):
    # A milli cell's bundle stores (h, edges, j) in energy units (milli / 1000); Round 1's
    # saved capture stores whole-milli integers. A regression once compared them directly
    # (a spurious ~1000x "does not rescore" failure on every milli cell); this locks in the
    # x1000 scale-back round2_export.py's own rescore_check applies.
    h = np.zeros(2)
    edges = np.array([[0, 1]], dtype=np.int64)
    j = np.array([1.0])  # energy units; milli equivalent is 1000
    spins = np.array([[1, 1], [1, -1]], dtype=np.int8)
    energies_milli = regimes.energy(spins, h, edges, j) * 1000.0  # what Round 1 saved
    capture_dir = tmp_path / "native-pm1" / "qpu-80"
    capture_dir.mkdir(parents=True)
    np.savez(capture_dir / f"{_nonce(0)}.npz", spins=spins, energies=energies_milli)

    lanes = runner.qpu_seed_lanes("native-pm1", _nonce(0), h, edges, j, tmp_path, lanes=2)
    assert lanes["unique_lanes"] == 2
    assert lanes["duplicate_lanes"] == 0


def test_qpu_seed_lanes_rejects_a_genuine_mismatch(tmp_path):
    h = np.zeros(2)
    edges = np.array([[0, 1]], dtype=np.int64)
    j = np.array([1.0])
    spins = np.array([[1, 1], [1, -1]], dtype=np.int8)
    capture_dir = tmp_path / "native-pm1" / "qpu-80"
    capture_dir.mkdir(parents=True)
    # a value with no consistent scale relationship to the real rescoring
    np.savez(capture_dir / f"{_nonce(0)}.npz", spins=spins, energies=np.array([12345.0, 999.0]))

    with pytest.raises(runner.RunnerError, match="do not rescore"):
        runner.qpu_seed_lanes("native-pm1", _nonce(0), h, edges, j, tmp_path, lanes=2)


# ------------------------------------------------------------ portfolio deadline arm
#
# These need P's pinned environment (qpo, dimod, dwave.samplers) on the path, unlike
# every other test in this file: they importorskip per-test, not at module level, so
# the rest of the suite still runs (and does) under D's own venv, which lacks qpo.


def test_portfolio_deadline_problem_beta_zero_and_nonzero_are_distinct():
    pytest.importorskip("qpo")
    zero = runner.build_portfolio_deadline_problem(18, 6, "beta-zero")
    nonzero = runner.build_portfolio_deadline_problem(18, 6, "beta-nonzero")
    assert zero.frustration_beta == 0.0
    assert nonzero.frustration_beta > 0.0


def test_portfolio_deadline_arm_masks_the_seed_to_32_bits(monkeypatch):
    pytest.importorskip("qpo")
    pytest.importorskip("dwave.samplers")
    from quip_miner_dwave import portfolio_replication as pr

    problem = runner.build_portfolio_deadline_problem(4, 2, "beta-zero")
    n_vars = pr.encode_to_ising(problem)[0].n
    captured = {}

    class FakeResponse:
        variables = list(range(n_vars))
        record = types.SimpleNamespace(sample=np.ones((1, n_vars), dtype=np.int8))

    class FakeSampler:
        def sample(self, bqm, *, num_reads, num_sweeps, seed):
            captured["seed"] = seed
            captured["num_reads"] = num_reads
            captured["num_sweeps"] = num_sweeps
            return FakeResponse()

    monkeypatch.setattr(
        "dwave.samplers.SimulatedAnnealingSampler", lambda: FakeSampler(), raising=False,
    )
    huge_seed = 796362028114473239  # > 2**32 - 1; exactly what tripped this once
    record = runner.run_portfolio_deadline_arm(4, 2, "beta-zero", huge_seed)
    assert record["seed"] == huge_seed % (1 << 31)
    assert 0 <= captured["seed"] < (1 << 31)
    assert captured["num_reads"] == runner.PORTFOLIO_NEAL_READS
    assert captured["num_sweeps"] == runner.PORTFOLIO_NEAL_SWEEPS
