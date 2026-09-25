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

import os
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quip_miner_dwave import regimes, round2_io

import round2_runner as runner

BUNDLE_EDGES = np.array([[0, 1], [1, 2]], dtype=np.int64)

FAKE_IDENTITY = {"package": "quip_msa", "version": "0.1.0", "binary_sha256": "abc123", "error": None}


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
    verdict = runner.check_contamination(before, after, concurrent_workers=1)
    assert verdict.contaminated
    assert any("loadavg_1m before" in reason for reason in verdict.reasons)
    # the thresholds used are recorded, not just implied
    assert verdict.thresholds["loadavg_1m_max"] == runner.LOADAVG_1M_CONTAMINATION_THRESHOLD


def test_contamination_flags_a_busy_sibling():
    before = runner.HostSample(0.1, 0.1, 0.1, "performance", sibling_cpu_times=(0.0, 100.0))
    after = runner.HostSample(0.1, 0.1, 0.1, "performance", sibling_cpu_times=(60.0, 200.0))
    verdict = runner.check_contamination(before, after, concurrent_workers=1)
    assert verdict.contaminated
    assert verdict.sibling_busy_fraction == pytest.approx(0.6)


def test_a_quiet_host_is_not_contaminated():
    before = runner.HostSample(0.1, 0.1, 0.1, "performance", sibling_cpu_times=(0.0, 100.0))
    after = runner.HostSample(0.2, 0.1, 0.1, "performance", sibling_cpu_times=(1.0, 200.0))
    verdict = runner.check_contamination(before, after, concurrent_workers=1)
    assert not verdict.contaminated
    assert verdict.reasons == ()


# ------------------------------------------ contamination under parallel workers
#
# Controller ruling (workers change, fix round 1, item 3): the loadavg is compared
# to the threshold after subtracting the run's own concurrent_workers -- its own
# workers are not competing load. This is one uniform formula for every
# timing_mode: a serial run (concurrent_workers=1) now subtracts 1 too, a small,
# documented departure from the exact-raw comparison the original pilot used,
# not an attempt to reproduce that number bit for bit.


def test_check_contamination_records_raw_and_adjusted_loadavg():
    before = runner.HostSample(10.0, 1.0, 1.0, "performance", sibling_cpu_times=(0.0, 100.0))
    after = runner.HostSample(10.0, 1.0, 1.0, "performance", sibling_cpu_times=(1.0, 200.0))
    verdict = runner.check_contamination(before, after, concurrent_workers=8)
    data = verdict.to_dict()
    assert data["loadavg_1m_raw_before"] == 10.0
    assert data["loadavg_1m_raw_after"] == 10.0
    assert data["loadavg_1m_adjusted_before"] == pytest.approx(2.0)
    assert data["loadavg_1m_adjusted_after"] == pytest.approx(2.0)
    assert data["concurrent_workers"] == 8
    assert data["thresholds"]["loadavg_1m_max"] == runner.LOADAVG_1M_CONTAMINATION_THRESHOLD


def test_eight_workers_on_an_otherwise_idle_host_are_not_contaminated():
    # raw loadavg ~8 comes entirely from this run's own 8 workers.
    before = runner.HostSample(8.0, 1.0, 1.0, "performance", sibling_cpu_times=(0.0, 100.0))
    after = runner.HostSample(8.2, 1.0, 1.0, "performance", sibling_cpu_times=(1.0, 200.0))
    verdict = runner.check_contamination(before, after, concurrent_workers=8)
    assert not verdict.contaminated


def test_heavy_external_load_still_contaminates_an_eight_worker_run():
    # raw loadavg is 8 (this run's own workers) plus 10 of real external load.
    before = runner.HostSample(18.0, 1.0, 1.0, "performance", sibling_cpu_times=(0.0, 100.0))
    after = runner.HostSample(18.0, 1.0, 1.0, "performance", sibling_cpu_times=(1.0, 200.0))
    verdict = runner.check_contamination(before, after, concurrent_workers=8)
    assert verdict.contaminated
    assert any("adjusted loadavg_1m" in reason for reason in verdict.reasons)


def test_a_serial_run_now_subtracts_one_worker_from_the_raw_loadavg():
    # documents the chosen behavior explicitly: serial is not byte-identical to
    # fix round 1 (raw, no subtraction) -- it subtracts concurrent_workers=1, the
    # same uniform formula parallel runs use.
    before = runner.HostSample(8.5, 1.0, 1.0, "performance", sibling_cpu_times=(0.0, 100.0))
    after = runner.HostSample(8.5, 1.0, 1.0, "performance", sibling_cpu_times=(1.0, 200.0))
    verdict = runner.check_contamination(before, after, concurrent_workers=1)
    # raw (8.5) would exceed the 8.0 threshold; adjusted (8.5 - 1 = 7.5) does not.
    assert not verdict.contaminated
    assert verdict.to_dict()["loadavg_1m_adjusted_before"] == pytest.approx(7.5)


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
    assert base.run_key(tmp_path, FAKE_IDENTITY) != changed.run_key(tmp_path, FAKE_IDENTITY)
    assert base.run_key(tmp_path, FAKE_IDENTITY) == base.run_key(tmp_path, FAKE_IDENTITY)


def test_run_key_is_sensitive_to_the_bundles_root(tmp_path):
    job = runner.CpuJob(
        cell="native-pm1", nonce=_nonce(1), kernel="cpu-sa", sweeps=512, reads=64,
        repetition_id=0, repetition_kind="timing", variant="timing", seed=1, seed_input_hash="h",
    )
    assert job.run_key(tmp_path / "a", FAKE_IDENTITY) != job.run_key(tmp_path / "b", FAKE_IDENTITY)


def test_run_key_is_sensitive_to_the_solvers_build_identity(tmp_path):
    # review finding 5: a quip_msa rebuild partway through a campaign must not let
    # resume silently mix results from two different builds.
    job = runner.CpuJob(
        cell="native-pm1", nonce=_nonce(1), kernel="cpu-sa", sweeps=512, reads=64,
        repetition_id=0, repetition_kind="timing", variant="timing", seed=1, seed_input_hash="h",
    )
    other_build = {**FAKE_IDENTITY, "binary_sha256": "def456"}
    assert job.run_key(tmp_path, FAKE_IDENTITY) != job.run_key(tmp_path, other_build)


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
    assert [job.run_key(tmp_path, FAKE_IDENTITY) for job in first] == [
        job.run_key(tmp_path, FAKE_IDENTITY) for job in second
    ]


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


def test_timing_subset_selects_one_model_per_cell(tmp_path):
    rows = _five_model_index(tmp_path, "native-pm1", count=5)
    index = _index(tmp_path, rows)
    jobs = runner.build_timing_subset_jobs(index, ["native-pm1"])
    nonces = {job.nonce for job in jobs}
    assert nonces == {sorted(row["nonce"] for row in rows)[0]}
    assert len(jobs) == len(runner.SWEEP_DEPTHS) * len(runner.CONTROLLED_KERNELS)


def test_timing_subset_uses_a_distinct_repetition_kind_from_the_campaign(tmp_path):
    rows = _five_model_index(tmp_path, "native-pm1", count=1)
    index = _index(tmp_path, rows)
    subset_jobs = runner.build_timing_subset_jobs(index, ["native-pm1"])
    campaign_jobs = runner.build_campaign_jobs(index, ["native-pm1"])
    assert {job.repetition_kind for job in subset_jobs} != {job.repetition_kind for job in campaign_jobs}


def test_timing_subset_reuses_the_campaigns_seed_for_the_same_arm(tmp_path):
    # the timing subset measures the exact same arm the campaign does, just under
    # controlled serial conditions -- same seed, same deterministic answer.
    rows = _five_model_index(tmp_path, "native-pm1", count=1)
    index = _index(tmp_path, rows)
    subset_job = runner.build_timing_subset_jobs(index, ["native-pm1"])[0]
    matching_campaign_job = next(
        job for job in runner.build_campaign_jobs(index, ["native-pm1"])
        if job.kernel == subset_job.kernel and job.sweeps == subset_job.sweeps
    )
    assert subset_job.seed == matching_campaign_job.seed


def test_captured_jobs_cover_unique_manifest_models_with_all_captured_arms():
    rows = []
    captured = []
    for cell, count in (
        ("clique-portfolio", 12), ("native-pm1", 4), ("diamond-pm1", 4), ("native-125", 4),
    ):
        for i in range(count):
            row = {"cell": cell, "nonce": _nonce(i), "model_hash": f"hash-{cell}-{i}"}
            rows.append(row)
            captured.extend([dict(row), dict(row)])
    rows.append({"cell": "native-pm1", "nonce": _nonce(99), "model_hash": "not-captured"})

    jobs = runner.build_captured_jobs({"rows": rows}, {"jobs": captured})

    assert len(jobs) == 480
    assert {job.kernel for job in jobs} == {
        "cpu-sa", "cpu-msa-f64", "cpu-msa-unit", "dwave-neal",
    }
    assert {job.sweeps for job in jobs} == set(runner.SWEEP_DEPTHS)
    assert {job.reads for job in jobs} == {64}
    assert {job.repetition_id for job in jobs} == {0}
    assert {job.repetition_kind for job in jobs} == {runner.REPETITION_TIMING}
    assert {job.variant for job in jobs} == {"timing"}
    assert len({(job.cell, job.nonce) for job in jobs}) == 24
    assert len({job.run_key(Path("/bundles"), FAKE_IDENTITY) for job in jobs}) == 480
    assert len({job.seed for job in jobs}) == 480
    assert sum(job.kernel == runner.UNIT_KERNEL for job in jobs if job.cell == "clique-portfolio") == 60
    assert all(job.nonce != _nonce(99) for job in jobs)
    neal_job = next(job for job in jobs if job.kernel == "dwave-neal")
    assert (neal_job.seed, neal_job.seed_input_hash) == runner.seed_for(
        f"hash-{neal_job.cell}-{int(neal_job.nonce, 16)}", "dwave-neal", neal_job.sweeps, 64, "timing",
    )


def test_captured_jobs_reject_a_manifest_model_hash_that_disagrees_with_the_bundle_index():
    row = {"cell": "native-pm1", "nonce": _nonce(0), "model_hash": "bundle-hash"}

    with pytest.raises(runner.RunnerError, match="model hash"):
        runner.build_captured_jobs({"rows": [row]}, {"jobs": [{**row, "model_hash": "capture-hash"}]})


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


def _fake_msa_module(sampler: Any):
    module = types.SimpleNamespace()
    module.Msa = lambda: sampler
    module.default_beta_range = lambda h, edges, j: (0.1, 5.0)
    module.__file__ = "/fake/quip_msa/__init__.py"  # solver_identity() degrades cleanly on this
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


def test_dwave_neal_uses_its_default_schedule_and_records_its_actual_beta_range(tmp_path, monkeypatch):
    _write_bundle(tmp_path, "native-pm1", _nonce(0))
    monkeypatch.setattr(runner, "_msa", lambda: _fake_msa_module(_FakeSampler()))
    captured = {}
    spins = np.array([[1, 1, 1], [-1, -1, -1]], dtype=np.int8)
    energies = runner.regimes.energy(spins, np.zeros(3), BUNDLE_EDGES, np.array([1.0, -1.0]))
    response = types.SimpleNamespace(
        variables=[0, 1, 2],
        record=types.SimpleNamespace(sample=spins, energy=energies),
        info={"beta_range": (0.123, 3.456)},
    )

    class FakeNealSampler:
        def sample_ising(self, h, j, **kwargs):
            captured.update(h=h, j=j, kwargs=kwargs)
            return response

    monkeypatch.setattr("dwave.samplers.SimulatedAnnealingSampler", FakeNealSampler)
    job = _job("native-pm1", _nonce(0), kernel="dwave-neal", sweeps=8, reads=2)
    job = runner.CpuJob(**{**job.to_dict(), "seed": (1 << 40) + 17})

    record, samples = runner.execute_cpu_job(job, tmp_path)

    assert captured["kwargs"] == {
        "num_reads": 2, "num_sweeps": 8, "seed": 17,
    }
    assert record["requested_kernel"] == record["observed_kernel"] == "dwave-neal"
    assert record["seed"] == (1 << 40) + 17
    assert record["effective_seed"] == captured["kwargs"]["seed"]
    assert record["beta_range"] == [0.123, 3.456]
    assert record["submitted_beta_range"] is None
    assert record["elapsed_sampling_s"] >= 0.0
    assert samples is not None
    np.testing.assert_array_equal(samples["spins"], spins)
    np.testing.assert_array_equal(samples["energies"], energies)


@pytest.mark.parametrize(
    ("cell", "h", "edges", "j"),
    [
        (
            "cubic-dimer-pm1", np.zeros(3),
            np.array([[0, 1], [1, 2]], dtype=np.int64), np.array([-0.5, 1.0]),
        ),
        (
            "native-pm1", np.zeros(3), BUNDLE_EDGES, np.array([0.001, -0.001]),
        ),
    ],
)
def test_dwave_neal_rescores_canonical_cubic_and_milli_models(
    tmp_path, monkeypatch, cell, h, edges, j,
):
    manifest = {
        "cell": cell, "nonce": _nonce(0),
        "identity": {"topology_hash": "t", "model_order_hash": "o"}, "offset": 0.0,
    }
    round2_io.write_bundle(tmp_path / cell / _nonce(0), manifest, {"h": h, "edges": edges, "j": j})
    spins = np.array([[1, 1, 1], [1, -1, 1]], dtype=np.int8)
    energies = regimes.energy(spins, h, edges, j)
    response = types.SimpleNamespace(
        variables=[0, 1, 2], record=types.SimpleNamespace(sample=spins, energy=energies),
        info={"beta_range": (0.1, 2.0)},
    )

    class FakeNealSampler:
        def sample_ising(self, *args, **kwargs):
            return response

    monkeypatch.setattr("dwave.samplers.SimulatedAnnealingSampler", FakeNealSampler)
    record, samples = runner.execute_cpu_job(
        _job(cell, _nonce(0), kernel="dwave-neal", reads=2), tmp_path,
    )

    assert record["exit_ok"] is True
    assert samples is not None
    np.testing.assert_allclose(samples["energies"], regimes.energy(spins, h, edges, j))


def test_dwave_neal_sampling_time_excludes_input_construction(tmp_path, monkeypatch):
    _write_bundle(tmp_path, "native-pm1", _nonce(0))

    class FakeClock:
        value = 0.0

        def advance(self, seconds):
            self.value += seconds

        def perf_counter(self):
            return self.value

    clock = FakeClock()

    class SlowArray(np.ndarray):
        def __iter__(self):
            clock.advance(2.0)
            return super().__iter__()

    read_bundle = round2_io.read_bundle

    def read_slow_bundle(path):
        manifest, arrays = read_bundle(path)
        return manifest, {name: np.asarray(value).view(SlowArray) for name, value in arrays.items()}

    monkeypatch.setattr(round2_io, "read_bundle", read_slow_bundle)
    monkeypatch.setattr(runner.time, "perf_counter", clock.perf_counter)

    spins = np.ones((2, 3), dtype=np.int8)
    energies = np.zeros(2)
    response = types.SimpleNamespace(
        variables=[0, 1, 2], record=types.SimpleNamespace(sample=spins, energy=energies),
        info={"beta_range": (0.1, 2.0)},
    )

    class TimedNealSampler:
        def sample_ising(self, *args, **kwargs):
            clock.advance(3.0)
            return response

    monkeypatch.setattr("dwave.samplers.SimulatedAnnealingSampler", TimedNealSampler)

    record, _ = runner.execute_cpu_job(
        _job("native-pm1", _nonce(0), kernel="dwave-neal", reads=2), tmp_path,
    )

    assert record["elapsed_sampling_s"] == pytest.approx(3.0)


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


def test_unsupported_unit_kernel_is_decided_before_ever_calling_the_kernel(tmp_path, monkeypatch):
    # native-125's alphabet is not unit-representable: the runner must catch this with
    # its own explicit pre-check, never by calling the kernel and reacting to its
    # ValueError (review finding 8: "record unsupported only for explicit, pre-checked
    # eligibility reasons").
    _write_bundle(tmp_path, "native-125", _nonce(0), j=np.array([0.5, -0.5]))
    calls = []

    class _NeverCalledSampler(_FakeSampler):
        def sample_research(self, *args, **kwargs):
            calls.append(1)
            return super().sample_research(*args, **kwargs)

    monkeypatch.setattr(runner, "_msa", lambda: _fake_msa_module(_NeverCalledSampler()))
    record, samples = runner.execute_cpu_job(_job("native-125", _nonce(0), kernel="cpu-msa-unit"), tmp_path)
    assert record["unsupported"] is True
    assert record["exit_ok"] is True
    assert "unit kernel" in record["unsupported_reason"]
    assert samples is None
    assert calls == []  # the kernel was never invoked


def test_a_value_error_from_the_unit_kernel_after_a_clean_precheck_is_a_failure(tmp_path, monkeypatch):
    # j=[1.0, -1.0] passes the eligibility pre-check; a ValueError the kernel still
    # raises here is an unanticipated defect (e.g. a bad reconstruction), not a known
    # ineligibility, and must never be hidden as "unsupported".
    _write_bundle(tmp_path, "native-pm1", _nonce(0), j=np.array([1.0, -1.0]))
    monkeypatch.setattr(
        runner, "_msa", lambda: _fake_msa_module(_FakeSampler(raise_error=ValueError("unexpected defect")))
    )
    record, samples = runner.execute_cpu_job(_job("native-pm1", _nonce(0), kernel="cpu-msa-unit"), tmp_path)
    assert record["exit_ok"] is False
    assert record["unsupported"] is False
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


def test_unit_kernel_ineligibility_reason_flags_non_unit_couplings():
    assert runner.unit_kernel_ineligibility_reason(np.zeros(2), np.array([0.5, -1.0])) is not None
    assert runner.unit_kernel_ineligibility_reason(np.zeros(2), np.array([1.0, -1.0])) is None


def test_unit_kernel_ineligibility_reason_flags_fractional_fields():
    assert runner.unit_kernel_ineligibility_reason(np.array([0.5, 0.0]), np.array([1.0])) is not None
    assert runner.unit_kernel_ineligibility_reason(np.array([1.0, 0.0]), np.array([1.0])) is None


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
    # A pre-checked eligibility gap, not a defect: KernelIneligible, never RunnerError
    # (review finding 8 -- only explicit, pre-checked reasons become "unsupported").
    h = np.zeros(2)
    edges = np.array([[0, 1]])
    j = np.array([0.37])
    with pytest.raises(runner.KernelIneligible, match="not a multiple"):
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


# ------------------------------------------------------------- execute_seeded_sweep_job


class _SeededSweepFakeSampler:
    """cpu-msa-f64 has no eligibility limit at all: any ValueError it raises is an
    unanticipated defect, never a known ineligibility (review findings 7/8).

    Records every call's ``num_reads``, ``initial_spins`` (presence only), and
    ``start_beta`` in ``self.calls``, so a test can check the seeded and cold
    lanes really come from two independent calls (final-review finding I4:
    the cold lanes must be a genuine full-ladder anneal, not the tail of one
    call the seeded schedule already shortened).
    """

    def __init__(self, *, fail_on_main: bool):
        self._fail_on_main = fail_on_main
        self.calls: list = []

    def sample_research(
        self, h, edges, j, *, kernel, num_sweeps, num_reads, seed, beta_range,
        initial_spins=None, start_beta=None,
    ):
        self.calls.append(
            {"num_reads": num_reads, "seeded": initial_spins is not None, "start_beta": start_beta}
        )
        if self._fail_on_main and num_sweeps == runner.SEEDED_SWEEPS:
            raise ValueError("unexpected defect")
        spins = np.ones((num_reads, len(h)), dtype=np.int8)
        energies = regimes.energy(spins, h, edges, j)
        meta = {
            "observed_kernel": "cpu-msa-f64", "representation": "fake", "rng_scheme": "fake",
            "seeded_reads": 0, "workspace_bytes": None,
        }
        return spins, energies, meta


def test_seeded_sweep_value_error_is_a_failure_not_unsupported(tmp_path, monkeypatch):
    _write_bundle(tmp_path, "native-pm1", _nonce(0))
    monkeypatch.setattr(runner, "_msa", lambda: _fake_msa_module(_SeededSweepFakeSampler(fail_on_main=True)))
    record, samples = runner.execute_seeded_sweep_job("native-pm1", _nonce(0), tmp_path, tmp_path, "cpu-lite")
    assert record["exit_ok"] is False
    assert record["unsupported"] is False
    assert samples is None


def test_seeded_sweep_completes_when_the_kernel_behaves(tmp_path, monkeypatch):
    _write_bundle(tmp_path, "native-pm1", _nonce(0))
    monkeypatch.setattr(runner, "_msa", lambda: _fake_msa_module(_SeededSweepFakeSampler(fail_on_main=False)))
    record, samples = runner.execute_seeded_sweep_job("native-pm1", _nonce(0), tmp_path, tmp_path, "cpu-lite")
    assert record["exit_ok"] is True
    assert record["unsupported"] is False
    assert samples is not None


def test_seeded_and_cold_lanes_come_from_two_independent_calls(tmp_path, monkeypatch):
    # review finding I4: the previous single call let the kernel apply the seeded
    # (midpoint-start) beta schedule to the "cold" reads too. The cold lanes must
    # instead come from their own call, with no initial_spins at all -- a genuine
    # full-ladder anneal, not the tail of the seeded call.
    _write_bundle(tmp_path, "native-pm1", _nonce(0))
    sampler = _SeededSweepFakeSampler(fail_on_main=False)
    monkeypatch.setattr(runner, "_msa", lambda: _fake_msa_module(sampler))
    runner.execute_seeded_sweep_job("native-pm1", _nonce(0), tmp_path, tmp_path, "cpu-lite")

    # one lite-seed call (64 reads, cold) plus the seeded call and the cold call.
    main_calls = [c for c in sampler.calls if c["num_reads"] in (runner.SEED_LANES, runner.COLD_LANES)]
    assert len(main_calls) == 2
    seeded_call = next(c for c in main_calls if c["seeded"])
    cold_call = next(c for c in main_calls if not c["seeded"])
    assert seeded_call["num_reads"] == runner.SEED_LANES
    assert cold_call["num_reads"] == runner.COLD_LANES
    assert seeded_call["start_beta"] is not None  # explicit, not left to the kernel's own default
    assert cold_call["start_beta"] is None  # start_beta is illegal on an unseeded call


def test_seeded_sweep_record_carries_start_betas_solver_identity_and_run_key(tmp_path, monkeypatch):
    _write_bundle(tmp_path, "native-pm1", _nonce(0))
    monkeypatch.setattr(runner, "_msa", lambda: _fake_msa_module(_SeededSweepFakeSampler(fail_on_main=False)))
    record, _samples = runner.execute_seeded_sweep_job("native-pm1", _nonce(0), tmp_path, tmp_path, "cpu-lite")
    hot, cold = record["beta_range"]
    assert record["seeded_start_beta"] == pytest.approx((hot * cold) ** 0.5)
    assert record["cold_start_beta"] == pytest.approx(hot)
    np.testing.assert_allclose(
        record["seeded_beta_ladder"], np.geomspace((hot * cold) ** 0.5, cold, runner.SEEDED_SWEEPS),
    )
    np.testing.assert_allclose(record["cold_beta_ladder"], np.geomspace(hot, cold, runner.SEEDED_SWEEPS))
    assert record["solver_identity"]["package"] == "quip_msa"
    assert record["cold_seed"] != record["seed"]
    assert isinstance(record["run_key"], str) and record["run_key"]


def test_seeded_sweep_schema_is_versioned_v2(tmp_path, monkeypatch):
    _write_bundle(tmp_path, "native-pm1", _nonce(0))
    monkeypatch.setattr(runner, "_msa", lambda: _fake_msa_module(_SeededSweepFakeSampler(fail_on_main=False)))
    record, _samples = runner.execute_seeded_sweep_job("native-pm1", _nonce(0), tmp_path, tmp_path, "cpu-lite")
    assert record["schema"] == "round2-seeded-sweep-v2"


# ---------------------------------------------------------------- cpu_lite_seed_lanes


class _WrongKernelSampler:
    def sample_research(self, h, edges, j, *, kernel, num_sweeps, num_reads, seed, beta_range, initial_spins=None):
        spins = np.ones((num_reads, len(h)), dtype=np.int8)
        energies = regimes.energy(spins, h, edges, j)
        meta = {
            "observed_kernel": "cpu-sa", "representation": "fake", "rng_scheme": "fake",
            "seeded_reads": 0, "workspace_bytes": None,
        }
        return spins, energies, meta


def test_cpu_lite_seed_lanes_rejects_the_wrong_observed_kernel(monkeypatch):
    # review finding M8: cpu_lite_seed_lanes captured meta["observed_kernel"] into its
    # return value but never checked it -- a silent kernel substitution would have gone
    # unnoticed.
    h = np.zeros(3)
    edges = BUNDLE_EDGES
    j = np.array([1.0, -1.0])
    monkeypatch.setattr(runner, "_msa", lambda: _fake_msa_module(_WrongKernelSampler()))
    with pytest.raises(runner.RunnerError, match="observed"):
        runner.cpu_lite_seed_lanes(h, edges, j, seed=0)


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


def test_portfolio_deadline_arm_labels_its_provenance_as_synthetic_test(monkeypatch):
    # review finding I5: the instance is a synthetic, deterministically-seeded
    # portfolio, never a captured historical one -- every record must say so.
    pytest.importorskip("qpo")
    pytest.importorskip("dwave.samplers")
    seed, _ = runner.seed_for("4-2-beta-zero", "neal-500-500", 500, 500, "portfolio-deadline")
    record = runner.run_portfolio_deadline_arm(4, 2, "beta-zero", seed)
    assert record["provenance"] == "synthetic-test"


def test_portfolio_deadline_arm_records_p_head_and_package_versions(monkeypatch):
    pytest.importorskip("qpo")
    pytest.importorskip("dwave.samplers")
    seed, _ = runner.seed_for("4-2-beta-zero", "neal-500-500", 500, 500, "portfolio-deadline")
    record = runner.run_portfolio_deadline_arm(4, 2, "beta-zero", seed)
    assert record["p_head"] is None or isinstance(record["p_head"], str)
    assert record["package_versions"]["dwave-samplers"]
    assert record["package_versions"]["dimod"]


def test_portfolio_deadline_arm_times_repair_separately_from_sampling(monkeypatch):
    pytest.importorskip("qpo")
    pytest.importorskip("dwave.samplers")
    from quip_miner_dwave import portfolio_replication as pr

    problem = runner.build_portfolio_deadline_problem(4, 2, "beta-zero")
    n_vars = pr.encode_to_ising(problem)[0].n

    class FakeResponse:
        variables = list(range(n_vars))
        record = types.SimpleNamespace(sample=np.ones((1, n_vars), dtype=np.int8))

    class FakeSampler:
        def sample(self, bqm, *, num_reads, num_sweeps, seed):
            return FakeResponse()

    monkeypatch.setattr("dwave.samplers.SimulatedAnnealingSampler", lambda: FakeSampler(), raising=False)
    seed, _ = runner.seed_for("4-2-beta-zero", "neal-500-500", 500, 500, "portfolio-deadline")
    record = runner.run_portfolio_deadline_arm(4, 2, "beta-zero", seed)
    assert record["repair_s"] is not None
    assert record["repair_s"] >= 0.0
    assert record["end_to_end_s"] >= record["elapsed_s"] + record["repair_s"]


def test_portfolio_deadline_arm_status_comes_from_end_to_end_time_not_sampling_alone(monkeypatch):
    # review finding I5: a slow repair/weighting phase must be able to push a run past
    # the deadline even when sampling itself was fast -- status must reflect that, not
    # silently call it a win because sampling alone was on time.
    pytest.importorskip("qpo")
    pytest.importorskip("dwave.samplers")
    from quip_miner_dwave import portfolio_replication as pr

    problem = runner.build_portfolio_deadline_problem(4, 2, "beta-zero")
    n_vars = pr.encode_to_ising(problem)[0].n

    class FakeResponse:
        variables = list(range(n_vars))
        record = types.SimpleNamespace(sample=np.ones((1, n_vars), dtype=np.int8))

    class FakeSampler:
        def sample(self, bqm, *, num_reads, num_sweeps, seed):
            return FakeResponse()  # fast: sampling alone is nowhere near the deadline

    def slow_score_reads(problem, qubo, spins):
        time.sleep(runner.PORTFOLIO_DEADLINE_S + 0.05)
        return real_score_reads(problem, qubo, spins)

    real_score_reads = pr.score_reads
    monkeypatch.setattr("dwave.samplers.SimulatedAnnealingSampler", lambda: FakeSampler(), raising=False)
    monkeypatch.setattr(pr, "score_reads", slow_score_reads)
    seed, _ = runner.seed_for("4-2-beta-zero", "neal-500-500", 500, 500, "portfolio-deadline")
    record = runner.run_portfolio_deadline_arm(4, 2, "beta-zero", seed)
    assert record["elapsed_s"] < runner.PORTFOLIO_DEADLINE_S
    assert record["end_to_end_s"] > runner.PORTFOLIO_DEADLINE_S
    assert record["status"] == "timeout"


# ------------------------------------------------------------ hard deadline scaling


def test_estimate_hard_deadline_scales_with_sweeps_and_size():
    shallow = runner.estimate_hard_deadline_s(sweeps=2048, n_spins=4575, reads=64)
    deep = runner.estimate_hard_deadline_s(sweeps=131072, n_spins=4575, reads=64)
    assert deep > shallow
    # deep is 64x the sweeps of shallow, so (above the floor) it must scale ~64x too
    assert deep == pytest.approx(shallow * 64, rel=0.05)


def test_estimate_hard_deadline_never_drops_below_the_floor():
    tiny = runner.estimate_hard_deadline_s(sweeps=1, n_spins=1, reads=1)
    assert tiny == runner.HARD_DEADLINE_FLOOR_S


def test_estimate_hard_deadline_covers_the_pilots_worst_observed_case_with_margin():
    # the exact case the review flagged: native-125 cpu-sa, 2,048 sweeps, 64 reads,
    # n=4,575, observed at 43.57 s under heavy contention.
    deadline = runner.estimate_hard_deadline_s(sweeps=2048, n_spins=4575, reads=64)
    assert deadline > 43.57 * 2  # comfortable margin, not just barely above


# --------------------------------------------------- run_subprocess_with_hard_deadline


def test_run_subprocess_with_hard_deadline_lets_a_quick_process_finish():
    exit_ok, wall_s = runner.run_subprocess_with_hard_deadline(
        [sys.executable, "-c", "pass"], hard_deadline_s=10.0,
    )
    assert exit_ok is True
    assert wall_s < 10.0


def test_run_subprocess_with_hard_deadline_kills_a_hung_process():
    pids = []
    start = time.perf_counter()
    exit_ok, wall_s = runner.run_subprocess_with_hard_deadline(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        hard_deadline_s=0.2,
        on_spawn=pids.append,
    )
    elapsed = time.perf_counter() - start
    assert exit_ok is False
    assert elapsed < 10.0  # killed promptly, never waited out the 30 s sleep
    assert len(pids) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)  # the child is actually gone, not orphaned


def test_run_subprocess_with_hard_deadline_kills_the_child_and_raises_on_sigterm():
    pids: list = []
    timer = threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        with pytest.raises(runner.Cancelled):
            runner.run_subprocess_with_hard_deadline(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                hard_deadline_s=30.0,
                on_spawn=pids.append,
            )
    finally:
        timer.cancel()
    assert len(pids) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)


def test_run_subprocess_with_hard_deadline_restores_signal_handlers_afterward():
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)
    runner.run_subprocess_with_hard_deadline([sys.executable, "-c", "pass"], hard_deadline_s=10.0)
    assert signal.getsignal(signal.SIGTERM) == previous_term
    assert signal.getsignal(signal.SIGINT) == previous_int
def test_portfolio_deadline_arm_records_weighting_status_and_raw_feasible_count():
    pytest.importorskip("qpo")
    pytest.importorskip("dwave.samplers")
    # a real, small basket, run against the real reference sampler (fast: <1s), to
    # verify the actual weighting-status wiring against P's real implementation
    # rather than a fake.
    seed, _ = runner.seed_for("4-2-beta-zero", "neal-500-500", 500, 500, "portfolio-deadline")
    record = runner.run_portfolio_deadline_arm(4, 2, "beta-zero", seed)
    assert record["exit_ok"] is True
    assert "weighting_failed" in record
    assert record["weighting_failed"] in (None, True, False)
    assert isinstance(record["raw_feasible_count"], int)


def test_run_subprocess_with_hard_deadline_passes_a_custom_env(tmp_path):
    marker = tmp_path / "seen.txt"
    cmd = [
        sys.executable, "-c",
        f"import os; open({str(marker)!r}, 'w').write(os.environ.get('MARKER_VAR', ''))",
    ]
    custom_env = {**os.environ, "MARKER_VAR": "custom-value"}
    exit_ok, _wall_s = runner.run_subprocess_with_hard_deadline(cmd, hard_deadline_s=10.0, env=custom_env)
    assert exit_ok is True
    assert marker.read_text() == "custom-value"


def test_child_subprocesses_receive_single_threaded_blas_environment(monkeypatch):
    spawned_envs = []
    popen = subprocess.Popen

    def capture_spawn_env(*args, **kwargs):
        spawned_envs.append(kwargs["env"])
        return popen(*args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", capture_spawn_env)

    exit_ok, _wall_s = runner.run_subprocess_with_hard_deadline(
        [sys.executable, "-c", "pass"], hard_deadline_s=10.0,
        env={"PYTHONPATH": "/caller/python/path"},
    )

    assert exit_ok is True
    assert len(spawned_envs) == 1
    assert {
        name: spawned_envs[0][name]
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    } == {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    assert spawned_envs[0]["PATH"] == os.environ["PATH"]
    assert spawned_envs[0]["PYTHONPATH"] == "/caller/python/path"


# --------------------------------------------------------- select_worker_cpus


def _fake_siblings(pairs):
    """A siblings_of function for a synthetic topology: pairs like {0: [16], 16: [0]}."""
    return lambda cpu: pairs.get(cpu, [])


def _pair_siblings(*pairs):
    """Build a full pairs dict from (a, b) pairs, each other's sole sibling."""
    mapping: dict = {}
    for a, b in pairs:
        mapping[a] = [b]
        mapping[b] = [a]
    return mapping


def test_select_worker_cpus_skips_smt_siblings():
    # host shape from the change brief: cpu n and n+16 are SMT siblings
    siblings = _fake_siblings(_pair_siblings(*((i, i + 16) for i in range(16))))
    chosen = runner.select_worker_cpus(4, allowed=range(32), siblings_of=siblings, avoid_cpu0=False)
    assert chosen == [0, 1, 2, 3]  # ascending, never touching 16-31 until 0-15 exhausted
    for cpu in chosen:
        assert cpu + 16 not in chosen
        assert cpu - 16 not in chosen


def test_select_worker_cpus_never_doubles_up_a_physical_core():
    siblings = _fake_siblings(_pair_siblings((0, 16), (1, 17)))
    # An allowed set that lists both members of each physical core: still only one
    # logical CPU per physical core comes back.
    chosen = runner.select_worker_cpus(2, allowed=[0, 16, 1, 17], siblings_of=siblings, avoid_cpu0=False)
    assert len(chosen) == 2
    physical_cores = {frozenset({cpu, *siblings(cpu)}) for cpu in chosen}
    assert len(physical_cores) == 2  # two DISTINCT physical cores, not one core twice


def test_select_worker_cpus_raises_when_not_enough_physical_cores():
    siblings = _fake_siblings(_pair_siblings((0, 16)))
    with pytest.raises(ValueError, match="need 3"):
        runner.select_worker_cpus(3, allowed=[0, 16], siblings_of=siblings, avoid_cpu0=False)


def test_select_worker_cpus_with_explicit_cpus_validates_distinct_physical_cores():
    siblings = _fake_siblings(_pair_siblings((0, 16), (1, 17)))
    with pytest.raises(ValueError, match="same physical core"):
        runner.select_worker_cpus(2, cpus=[0, 16], allowed=[0, 16, 1, 17], siblings_of=siblings)


def test_select_worker_cpus_with_explicit_cpus_accepts_distinct_physical_cores():
    siblings = _fake_siblings(_pair_siblings((0, 16), (1, 17)))
    chosen = runner.select_worker_cpus(2, cpus=[0, 17], allowed=[0, 16, 1, 17], siblings_of=siblings)
    assert chosen == [0, 17]


def test_select_worker_cpus_handles_more_than_two_threads_per_core():
    # a hypothetical 4-way-SMT core: 0, 8, 16, 24 all share one physical core
    siblings = _fake_siblings({0: [8, 16, 24], 8: [0, 16, 24], 16: [0, 8, 24], 24: [0, 8, 16]})
    chosen = runner.select_worker_cpus(1, allowed=[0, 8, 16, 24], siblings_of=siblings, avoid_cpu0=False)
    assert chosen == [0]


def test_select_worker_cpus_automatic_choice_avoids_physical_core_0_by_default():
    # review finding 1: core 0 handles interrupts and may be pinned to other work;
    # the automatic choice must not land there unless explicitly asked to.
    siblings = _fake_siblings(_pair_siblings(*((i, i + 16) for i in range(16))))
    chosen = runner.select_worker_cpus(4, allowed=range(32), siblings_of=siblings)
    assert 0 not in chosen
    assert 16 not in chosen
    assert chosen == [1, 2, 3, 4]


def test_select_worker_cpus_explicit_cpus_may_still_name_core_0():
    # avoid_cpu0 only changes the AUTOMATIC choice; an explicit --cpus can still ask
    # for core 0 outright.
    siblings = _fake_siblings(_pair_siblings((0, 16)))
    chosen = runner.select_worker_cpus(1, cpus=[0], allowed=[0, 16], siblings_of=siblings)
    assert chosen == [0]


def test_select_worker_cpus_rejects_workers_below_one():
    siblings = _fake_siblings(_pair_siblings((0, 16)))
    with pytest.raises(ValueError, match="at least 1"):
        runner.select_worker_cpus(0, allowed=[0, 16], siblings_of=siblings)


def test_select_worker_cpus_rejects_an_empty_explicit_cpus_list():
    siblings = _fake_siblings(_pair_siblings((0, 16)))
    with pytest.raises(ValueError, match="empty"):
        runner.select_worker_cpus(1, cpus=[], allowed=[0, 16], siblings_of=siblings)


def test_select_worker_cpus_rejects_an_explicit_cpu_outside_the_allowed_set():
    siblings = _fake_siblings(_pair_siblings((0, 16), (1, 17)))
    with pytest.raises(ValueError, match="not in the allowed"):
        runner.select_worker_cpus(1, cpus=[9], allowed=[0, 16, 1, 17], siblings_of=siblings)


# ------------------------------------------------------------------- cpu_siblings


def test_parse_sibling_list_handles_a_comma_list():
    assert runner._parse_sibling_list("4,20", 4) == [20]


def test_parse_sibling_list_handles_a_range():
    assert runner._parse_sibling_list("4-7", 4) == [5, 6, 7]


def test_parse_sibling_list_handles_a_mix_of_ranges_and_commas():
    assert sorted(runner._parse_sibling_list("0-1,16-17", 0)) == [1, 16, 17]


# ------------------------------------------------------ parallel subprocess tracking


def test_active_processes_kill_all_kills_every_registered_process():
    registry = runner.ActiveProcesses()
    procs = [
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        for _ in range(3)
    ]
    for proc in procs:
        registry.add(proc)
    registry.kill_all()
    for proc in procs:
        proc.wait(timeout=5)
        with pytest.raises(ProcessLookupError):
            os.kill(proc.pid, 0)


def test_kill_all_skips_a_process_whose_returncode_is_already_set(monkeypatch):
    # review, minor 5: killing by pid after the child is reaped risks hitting a
    # reused PID. A process with a known returncode has already been reaped
    # (wait() sets it), so kill_all must not touch it at all.
    registry = runner.ActiveProcesses()
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait()
    assert finished.returncode is not None
    running = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    registry.add(finished)
    registry.add(running)

    killed = []
    real_kill = runner._kill_process_group

    def tracking_kill(proc):
        killed.append(proc)
        real_kill(proc)

    monkeypatch.setattr(runner, "_kill_process_group", tracking_kill)
    registry.kill_all()

    assert killed == [running]
    running.wait(timeout=5)
    with pytest.raises(ProcessLookupError):
        os.kill(running.pid, 0)


def test_run_subprocess_tracked_raises_cancelled_without_spawning_when_already_cancelled():
    registry = runner.ActiveProcesses()
    cancelled = threading.Event()
    cancelled.set()
    pids = []
    with pytest.raises(runner.Cancelled):
        runner.run_subprocess_tracked(
            [sys.executable, "-c", "import time; time.sleep(30)"], hard_deadline_s=30.0,
            registry=registry, cancelled_event=cancelled, on_spawn=pids.append,
        )
    assert pids == []  # never spawned a subprocess once already cancelled
    assert registry.is_empty()


def test_run_subprocess_tracked_kills_the_child_if_cancelled_races_the_spawn():
    # Simulate the narrow window between _spawn() and registry.add(): the event
    # becomes set only once a process already exists (via a real thread racing
    # in), which the post-registration re-check must still catch.
    registry = runner.ActiveProcesses()
    cancelled = threading.Event()
    pids = []

    def set_event_on_spawn(pid):
        pids.append(pid)
        cancelled.set()

    with pytest.raises(runner.Cancelled):
        runner.run_subprocess_tracked(
            [sys.executable, "-c", "import time; time.sleep(30)"], hard_deadline_s=30.0,
            registry=registry, cancelled_event=cancelled, on_spawn=set_event_on_spawn,
        )
    assert len(pids) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)


def test_run_subprocess_tracked_returns_normally_when_not_cancelled():
    registry = runner.ActiveProcesses()
    cancelled = threading.Event()
    exit_ok, wall_s = runner.run_subprocess_tracked(
        [sys.executable, "-c", "pass"], hard_deadline_s=10.0, registry=registry, cancelled_event=cancelled,
    )
    assert exit_ok is True
    assert wall_s < 10.0


def test_run_subprocess_tracked_deregisters_the_process_when_done():
    registry = runner.ActiveProcesses()
    cancelled = threading.Event()
    runner.run_subprocess_tracked(
        [sys.executable, "-c", "pass"], hard_deadline_s=10.0, registry=registry, cancelled_event=cancelled,
    )
    assert registry.is_empty()


def test_run_one_subprocess_tracked_requires_concurrent_workers(tmp_path):
    # review, minor 7: concurrent_workers=2 was an arbitrary default; every real
    # caller must say explicitly how many workers this run has.
    job = runner.CpuJob(
        cell="native-pm1", nonce=_nonce(0), kernel="cpu-sa", sweeps=8, reads=4,
        repetition_id=0, repetition_kind="timing", variant="timing", seed=1, seed_input_hash="h",
    )
    with pytest.raises(TypeError):
        runner.run_one_subprocess_tracked(  # type: ignore[call-arg]
            job, bundles_root=tmp_path, out_dir=tmp_path, cpu=None, script_path=tmp_path, attempt=0,
            registry=runner.ActiveProcesses(), cancelled_event=threading.Event(),
        )
