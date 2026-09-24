"""Tests for the Round 2 CPU orchestration script: attempts, resume, and the
subprocess-per-job pipeline. See ``round2_cpu.py`` for the module under test.

Review fix round 1, finding 4: these were the missing step-8 tests -- resume
refusal and resume identity, an interrupted write between the samples file
and the record, and a repeat of a contaminated run.
"""

from __future__ import annotations

import json
import os
import signal
import threading
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from quip_miner_dwave import round2_io

import round2_cpu as cpu
import round2_runner as runner

BUNDLE_EDGES = np.array([[0, 1], [1, 2]], dtype=np.int64)
IDENTITY = {"package": "quip_msa", "version": "0.1.0", "binary_sha256": "abc123", "error": None}


def _nonce(i: int) -> str:
    return format(i, "064x")


def _write_bundle(bundles_root: Path, cell: str, nonce: str) -> str:
    manifest = {
        "cell": cell, "nonce": nonce,
        "identity": {"topology_hash": "t", "model_order_hash": "o"}, "offset": 0.0,
    }
    h = np.zeros(3)
    j = np.array([1.0, -1.0])
    return round2_io.write_bundle(bundles_root / cell / nonce, manifest, {"h": h, "edges": BUNDLE_EDGES, "j": j})


def _job(cell: str = "native-pm1", nonce: str | None = None, **overrides: Any) -> runner.CpuJob:
    nonce = nonce or _nonce(0)
    seed, seed_hash = runner.seed_for("m", "cpu-sa", 512, 64, "timing")
    fields: Dict[str, Any] = dict(
        cell=cell, nonce=nonce, kernel="cpu-sa", sweeps=512, reads=64,
        repetition_id=0, repetition_kind="timing", variant="timing", seed=seed, seed_input_hash=seed_hash,
    )
    fields.update(overrides)
    return runner.CpuJob(**fields)


def _write_record(out_dir: Path, job: runner.CpuJob, attempt: int, bundles_root: Path, **fields) -> Path:
    record_path, _samples_path = cpu._attempt_paths(out_dir, job, attempt)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_key": job.run_key(bundles_root, IDENTITY),
        "cell": job.cell, "nonce": job.nonce, "attempt": attempt,
        "exit_ok": True, "unsupported": False, "host": None,
    }
    record.update(fields)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    return record_path


# --------------------------------------------------------------------- attempts


def test_resolve_run_starts_at_attempt_zero_with_no_prior_attempts(tmp_path):
    _write_bundle(tmp_path / "bundles", "native-pm1", _nonce(0))
    job = _job()
    out_dir = tmp_path / "out"
    existing, attempt = cpu.resolve_run(out_dir, job, tmp_path / "bundles", IDENTITY, 120.0)
    assert existing is None
    assert attempt == 0


def test_resolve_run_skips_a_terminal_completed_attempt(tmp_path):
    bundles_root = tmp_path / "bundles"
    _write_bundle(bundles_root, "native-pm1", _nonce(0))
    job = _job()
    out_dir = tmp_path / "out"
    _write_record(out_dir, job, 0, bundles_root, exit_ok=True, unsupported=False)
    _, samples_path = cpu._attempt_paths(out_dir, job, 0)
    np.savez(samples_path, spins=np.ones((1, 3)), energies=np.zeros(1))

    existing, attempt = cpu.resolve_run(out_dir, job, bundles_root, IDENTITY, 120.0)
    assert existing is not None
    assert attempt == 0


# --------------------------------------------- finding 2: interrupted write regression


def test_resolve_run_retries_a_completed_record_with_no_samples_file(tmp_path):
    # The record is written LAST (finding 2): if a kill lands between the samples
    # write and the record write, only the record could ever exist without its
    # samples -- but this test locks in the *opposite*, impossible-by-design case
    # too: even if a record somehow exists with unsupported=False and no .npz, it
    # must never be trusted as done.
    bundles_root = tmp_path / "bundles"
    _write_bundle(bundles_root, "native-pm1", _nonce(0))
    job = _job()
    out_dir = tmp_path / "out"
    _write_record(out_dir, job, 0, bundles_root, exit_ok=True, unsupported=False)
    # deliberately no .npz written

    existing, attempt = cpu.resolve_run(out_dir, job, bundles_root, IDENTITY, 120.0)
    assert existing is None
    assert attempt == 1  # a new attempt, not attempt 0 clobbered


def test_resolve_run_never_requires_samples_for_an_unsupported_record(tmp_path):
    bundles_root = tmp_path / "bundles"
    _write_bundle(bundles_root, "native-pm1", _nonce(0))
    job = _job()
    out_dir = tmp_path / "out"
    _write_record(out_dir, job, 0, bundles_root, exit_ok=True, unsupported=True)

    existing, attempt = cpu.resolve_run(out_dir, job, bundles_root, IDENTITY, 120.0)
    assert existing is not None
    assert attempt == 0


# ------------------------------------------------------- finding 3: resume / retry


def test_resolve_run_retries_a_plain_failure(tmp_path):
    bundles_root = tmp_path / "bundles"
    _write_bundle(bundles_root, "native-pm1", _nonce(0))
    job = _job()
    out_dir = tmp_path / "out"
    _write_record(out_dir, job, 0, bundles_root, exit_ok=False, unsupported=False, error="boom")

    existing, attempt = cpu.resolve_run(out_dir, job, bundles_root, IDENTITY, 120.0)
    assert existing is None
    assert attempt == 1


def test_resolve_run_does_not_retry_a_hard_deadline_kill_under_the_same_deadline(tmp_path):
    bundles_root = tmp_path / "bundles"
    _write_bundle(bundles_root, "native-pm1", _nonce(0))
    job = _job()
    out_dir = tmp_path / "out"
    _write_record(
        out_dir, job, 0, bundles_root, exit_ok=False, unsupported=False,
        killed_reason="hard_deadline", hard_deadline_s=120.0,
    )

    existing, attempt = cpu.resolve_run(out_dir, job, bundles_root, IDENTITY, 120.0)
    assert existing is not None
    assert attempt == 0


def test_resolve_run_retries_a_hard_deadline_kill_once_the_deadline_changes(tmp_path):
    bundles_root = tmp_path / "bundles"
    _write_bundle(bundles_root, "native-pm1", _nonce(0))
    job = _job()
    out_dir = tmp_path / "out"
    _write_record(
        out_dir, job, 0, bundles_root, exit_ok=False, unsupported=False,
        killed_reason="hard_deadline", hard_deadline_s=120.0,
    )

    existing, attempt = cpu.resolve_run(out_dir, job, bundles_root, IDENTITY, 600.0)
    assert existing is None
    assert attempt == 1


# ---------------------------------- final review, M1: a crash is not a hard-deadline kill


def test_fallback_record_is_hard_deadline_when_the_parent_actually_enforced_it(tmp_path, monkeypatch):
    # wall_s >= hard_deadline_s means run_subprocess_with_hard_deadline's own
    # TimeoutExpired path fired: the parent genuinely killed the child at the deadline.
    bundles_root = tmp_path / "bundles"
    _write_bundle(bundles_root, "native-pm1", _nonce(0))
    job = _job()
    out_dir = tmp_path / "out"
    monkeypatch.setattr(runner, "run_one_subprocess", lambda *a, **k: (False, 5.0))
    record = cpu._run_one_job(job, bundles_root, out_dir, None, 0, IDENTITY, 5.0)
    assert record["killed_reason"] == "hard_deadline"


def test_fallback_record_is_crashed_no_record_when_the_child_exits_before_the_deadline(tmp_path, monkeypatch):
    # review finding M1: the child exited (crashed, or exited nonzero before ever
    # reaching its own write) well before the hard deadline ever fired -- this is a
    # plain crash, not a deadline kill, and must stay retryable.
    bundles_root = tmp_path / "bundles"
    _write_bundle(bundles_root, "native-pm1", _nonce(0))
    job = _job()
    out_dir = tmp_path / "out"
    monkeypatch.setattr(runner, "run_one_subprocess", lambda *a, **k: (False, 0.01))
    record = cpu._run_one_job(job, bundles_root, out_dir, None, 0, IDENTITY, 5.0)
    assert record["killed_reason"] == "crashed_no_record"


def test_crashed_no_record_fallback_is_never_terminal(tmp_path):
    bundles_root = tmp_path / "bundles"
    _write_bundle(bundles_root, "native-pm1", _nonce(0))
    job = _job()
    out_dir = tmp_path / "out"
    record = {"exit_ok": False, "killed_reason": "crashed_no_record"}
    assert cpu._is_terminal(record, out_dir, job, 0, hard_deadline_s=5.0) is False


def test_earlier_attempts_are_never_overwritten_or_deleted(tmp_path):
    bundles_root = tmp_path / "bundles"
    _write_bundle(bundles_root, "native-pm1", _nonce(0))
    job = _job()
    out_dir = tmp_path / "out"
    path0 = _write_record(out_dir, job, 0, bundles_root, exit_ok=False, error="first failure")
    path1 = _write_record(out_dir, job, 1, bundles_root, exit_ok=False, error="second failure")

    existing, attempt = cpu.resolve_run(out_dir, job, bundles_root, IDENTITY, 120.0)
    assert existing is None
    assert attempt == 2
    assert path0.exists()  # untouched
    assert path1.exists()  # untouched
    assert json.loads(path0.read_text())["error"] == "first failure"
    assert json.loads(path1.read_text())["error"] == "second failure"


def test_repeat_contaminated_only_touches_contaminated_jobs(tmp_path, monkeypatch):
    bundles_root = tmp_path / "bundles"
    _write_bundle(bundles_root, "native-pm1", _nonce(0))
    _write_bundle(bundles_root, "native-pm1", _nonce(1))
    clean_job = _job(nonce=_nonce(0))
    contaminated_job = _job(nonce=_nonce(1))
    out_dir = tmp_path / "out"
    _write_record(
        out_dir, clean_job, 0, bundles_root, exit_ok=True, unsupported=True,
        host={"contaminated": False, "reasons": []},
    )
    _write_record(
        out_dir, contaminated_job, 0, bundles_root, exit_ok=True, unsupported=True,
        host={"contaminated": True, "reasons": ["loadavg"]},
    )
    never_run_job = _job(nonce=_nonce(0), kernel="cpu-msa-f64")

    spawned = []

    def fake_run_one_job(job, bundles_root, out_dir, cpu_, attempt, identity, hard_deadline_s_override, **_kwargs):
        spawned.append((job.nonce, job.kernel, attempt))
        return {"exit_ok": True, "unsupported": True, "cell": job.cell, "host": None}

    monkeypatch.setattr(cpu, "_run_one_job", fake_run_one_job)
    monkeypatch.setattr(runner, "solver_identity", lambda: IDENTITY)

    cpu._run_jobs(
        [clean_job, contaminated_job, never_run_job], bundles_root, out_dir, cpus=[4], repeat_contaminated=True,
    )

    assert spawned == [(contaminated_job.nonce, contaminated_job.kernel, 1)]


# ------------------------------------------------------------------- resume refusal


def test_resolve_run_refuses_a_stale_run_key(tmp_path):
    bundles_root = tmp_path / "bundles"
    _write_bundle(bundles_root, "native-pm1", _nonce(0))
    job = _job()
    out_dir = tmp_path / "out"
    record_path, _ = cpu._attempt_paths(out_dir, job, 0)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps({"run_key": "not-this-jobs-key", "exit_ok": True}), encoding="utf-8")

    with pytest.raises(SystemExit, match="refusing to resume"):
        cpu.resolve_run(out_dir, job, bundles_root, IDENTITY, 120.0)


def test_resolve_run_refuses_a_record_from_a_different_solver_build(tmp_path):
    bundles_root = tmp_path / "bundles"
    _write_bundle(bundles_root, "native-pm1", _nonce(0))
    job = _job()
    out_dir = tmp_path / "out"
    _write_record(out_dir, job, 0, bundles_root, exit_ok=True, unsupported=True)

    other_build = {**IDENTITY, "binary_sha256": "different-build"}
    with pytest.raises(SystemExit, match="refusing to resume"):
        cpu.resolve_run(out_dir, job, bundles_root, other_build, 120.0)


# ------------------------------------------------- end-to-end: samples-before-record


def test_cmd_run_one_writes_samples_before_the_record(tmp_path, monkeypatch):
    bundles_root = tmp_path / "bundles"
    _write_bundle(bundles_root, "native-pm1", _nonce(0))
    job = _job()
    out_dir = tmp_path / "out"

    written_order = []

    import quip_miner_dwave.regime_io as regime_io_module

    original_savez = regime_io_module.atomic_savez
    original_write_json = regime_io_module.atomic_write_json

    def tracking_savez(path, **arrays):
        written_order.append(("npz", path))
        return original_savez(path, **arrays)

    def tracking_write_json(path, payload):
        written_order.append(("json", path))
        return original_write_json(path, payload)

    monkeypatch.setattr(cpu.regime_io, "atomic_savez", tracking_savez)
    monkeypatch.setattr(cpu.regime_io, "atomic_write_json", tracking_write_json)

    import argparse

    args = argparse.Namespace(
        job_json=json.dumps(job.to_dict()), bundles_root=str(bundles_root),
        out_dir=str(out_dir), cpu=None, attempt=None, hard_deadline_s=None,
        timing_mode="serial", concurrent_workers=1,
    )
    cpu.cmd_run_one(args)

    kinds = [kind for kind, _path in written_order]
    assert kinds == ["npz", "json"], kinds


# --------------------------------------------- real subprocess: repeat-contaminated
#
# This is a real, end-to-end check (an actual quip_msa subprocess, not a mocked
# _run_one_job): a mocked spawn cannot catch a mismatch between the attempt number
# the parent decides on and the attempt number the child actually writes to, which
# is exactly the bug an earlier version of this fix had (the child self-resolved
# its own attempt, silently disagreeing with --repeat-contaminated's choice, and
# the parent then wrote a false "hard deadline exceeded" record over a real, clean
# run the child had actually completed under a different attempt number).


def test_repeat_contaminated_end_to_end_with_a_real_subprocess(tmp_path):
    bundles_root = tmp_path / "bundles"
    _write_bundle(bundles_root, "native-pm1", _nonce(0))
    job = _job(kernel="cpu-msa-unit", sweeps=512)
    out_dir = tmp_path / "out"

    # A prior, completed-but-contaminated attempt 0. Its run_key must match what the
    # real subprocess will compute (the real solver identity), not the fake IDENTITY
    # the other tests in this file use for pure-Python resolve_run checks.
    real_identity = runner.solver_identity()
    record_path, _ = cpu._attempt_paths(out_dir, job, 0)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps({
        "run_key": job.run_key(bundles_root, real_identity),
        "cell": job.cell, "nonce": job.nonce, "attempt": 0,
        "exit_ok": True, "unsupported": True,
        "host": {"contaminated": True, "reasons": ["loadavg_1m before (99) exceeds 8.0"]},
    }), encoding="utf-8")

    records = cpu._run_jobs([job], bundles_root, out_dir, cpus=[None], repeat_contaminated=True)

    assert len(records) == 1
    record = records[0]
    assert record["exit_ok"] is True
    assert record.get("attempt") == 1
    assert record.get("killed_reason") is None  # a real run, not a spurious kill fallback

    attempt0_path, _ = cpu._attempt_paths(out_dir, job, 0)
    attempt1_path, _ = cpu._attempt_paths(out_dir, job, 1)
    assert attempt0_path.exists()  # untouched
    assert attempt1_path.exists()
    assert json.loads(attempt1_path.read_text())["attempt"] == 1


# ------------------------------------------------- change: parallel workers


def test_cmd_run_one_records_timing_mode_and_concurrent_workers(tmp_path):
    import argparse

    bundles_root = tmp_path / "bundles"
    _write_bundle(bundles_root, "native-pm1", _nonce(0))
    job = _job()
    out_dir = tmp_path / "out"

    args = argparse.Namespace(
        job_json=json.dumps(job.to_dict()), bundles_root=str(bundles_root),
        out_dir=str(out_dir), cpu=None, attempt=None, hard_deadline_s=None,
        timing_mode="parallel", concurrent_workers=4,
    )
    cpu.cmd_run_one(args)

    record_path, _ = cpu._attempt_paths(out_dir, job, 0)
    record = json.loads(record_path.read_text())
    assert record["timing_mode"] == "parallel"
    assert record["concurrent_workers"] == 4


def test_cmd_run_one_defaults_to_serial_with_one_worker(tmp_path):
    import argparse

    bundles_root = tmp_path / "bundles"
    _write_bundle(bundles_root, "native-pm1", _nonce(0))
    job = _job()
    out_dir = tmp_path / "out"

    args = argparse.Namespace(
        job_json=json.dumps(job.to_dict()), bundles_root=str(bundles_root),
        out_dir=str(out_dir), cpu=None, attempt=None, hard_deadline_s=None,
        timing_mode="serial", concurrent_workers=1,
    )
    cpu.cmd_run_one(args)

    record_path, _ = cpu._attempt_paths(out_dir, job, 0)
    record = json.loads(record_path.read_text())
    assert record["timing_mode"] == "serial"
    assert record["concurrent_workers"] == 1


# ------------------------------------------------- change: parallel workers, real runs


def test_parallel_run_end_to_end_with_real_subprocesses(tmp_path):
    bundles_root = tmp_path / "bundles"
    jobs = []
    for i in range(4):
        nonce = _nonce(i)
        _write_bundle(bundles_root, "native-pm1", nonce)
        jobs.append(_job(nonce=nonce, kernel="cpu-msa-unit", sweeps=512))
    out_dir = tmp_path / "out"

    records = cpu._run_jobs(jobs, bundles_root, out_dir, cpus=[10, 11])

    assert len(records) == 4
    assert all(r["exit_ok"] for r in records)
    assert all(r["timing_mode"] == "parallel" for r in records)
    assert all(r["concurrent_workers"] == 2 for r in records)
    # every job actually ran exactly once, at attempt 0
    for job in jobs:
        record_path, _ = cpu._attempt_paths(out_dir, job, 0)
        assert record_path.exists()
        assert not cpu._attempt_paths(out_dir, job, 1)[0].exists()


def test_parallel_run_is_resumable_like_the_serial_path(tmp_path):
    bundles_root = tmp_path / "bundles"
    jobs = []
    for i in range(2):
        nonce = _nonce(i)
        _write_bundle(bundles_root, "native-pm1", nonce)
        jobs.append(_job(nonce=nonce, kernel="cpu-msa-unit", sweeps=512))
    out_dir = tmp_path / "out"

    cpu._run_jobs(jobs, bundles_root, out_dir, cpus=[10, 11])
    # a second parallel pass over the same jobs must resume, not re-run
    second = cpu._run_jobs(jobs, bundles_root, out_dir, cpus=[10, 11])
    assert len(second) == 2
    for job in jobs:
        assert not cpu._attempt_paths(out_dir, job, 1)[0].exists()


def test_parallel_run_killed_by_sigterm_leaves_no_false_completed_record(tmp_path):
    bundles_root = tmp_path / "bundles"
    jobs = []
    for i in range(16):
        nonce = _nonce(i)
        _write_bundle(bundles_root, "native-pm1", nonce)
        jobs.append(_job(nonce=nonce, kernel="cpu-msa-unit", sweeps=512))
    out_dir = tmp_path / "out"

    timer = threading.Timer(0.05, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        with pytest.raises(runner.Cancelled):
            cpu._run_jobs(jobs, bundles_root, out_dir, cpus=[10, 11])
    finally:
        timer.cancel()

    completed = []
    for job in jobs:
        record_path, _ = cpu._attempt_paths(out_dir, job, 0)
        if record_path.exists():
            record = json.loads(record_path.read_text())
            # every record that exists is a real, honest completion -- never a
            # false "completed" fabricated for a job the parent abandoned.
            assert record["exit_ok"] is True
            assert record.get("killed_reason") is None
            completed.append(record)
    # the cancel must have actually interrupted something, or this test proves
    # nothing about the cancellation path
    assert len(completed) < len(jobs)


def test_worker_exception_is_logged_with_job_and_cpu(tmp_path, monkeypatch, capsys):
    # review, minor 7: a worker's real bug (not a job failure -- those already
    # produce a normal failed record) must be visible immediately, with enough
    # context to find it, even though only the first is re-raised after join.
    bundles_root = tmp_path / "bundles"
    _write_bundle(bundles_root, "native-pm1", _nonce(0))
    job = _job()
    out_dir = tmp_path / "out"

    def fake_run_one_job(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cpu, "_run_one_job", fake_run_one_job)
    with pytest.raises(RuntimeError, match="boom"):
        cpu._run_jobs([job], bundles_root, out_dir, cpus=[10, 11])

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert job.nonce in combined
    assert job.cell in combined
    assert "boom" in combined


# ------------------------------------------------- final review, I4: cmd_seeded_sweep


def _seeded_sweep_index(bundles_root: Path, cell: str = "native-pm1") -> str:
    model_hash = _write_bundle(bundles_root, cell, _nonce(0))
    (bundles_root / "index.json").write_text(
        json.dumps({"rows": [{"cell": cell, "nonce": _nonce(0), "model_hash": model_hash}]}),
        encoding="utf-8",
    )
    return model_hash


def _seeded_sweep_args(tmp_path: Path, **overrides: Any):
    import argparse

    fields: Dict[str, Any] = dict(
        bundles_root=str(tmp_path / "bundles"), round1_root=str(tmp_path / "round1"),
        out_root=str(tmp_path / "cpu"), cells=["native-pm1"], models=1,
    )
    fields.update(overrides)
    return argparse.Namespace(**fields)


def _fake_seeded_sweep_job(spins_value=1):
    def fake(cell, nonce, bundles_root, round1_root, seed_source):
        record = {
            "schema": "round2-seeded-sweep-v2", "cell": cell, "nonce": nonce,
            "seed_source": seed_source, "exit_ok": True, "unsupported": False,
        }
        samples = {"spins": np.full((1, 3), spins_value, dtype=np.int8), "energies": np.zeros(1)}
        return record, samples

    return fake


def test_seeded_sweep_writes_to_a_versioned_v2_output_directory_not_the_old_one(tmp_path, monkeypatch):
    # review finding I4: existing seeded-sweep outputs are superseded -- new runs go to
    # a new directory name, and the old one is never touched.
    bundles_root = tmp_path / "bundles"
    _seeded_sweep_index(bundles_root)
    monkeypatch.setattr(runner, "solver_identity", lambda: IDENTITY)
    monkeypatch.setattr(runner, "execute_seeded_sweep_job", _fake_seeded_sweep_job())
    args = _seeded_sweep_args(tmp_path)
    assert cpu.cmd_seeded_sweep(args) == 0
    assert (Path(args.out_root) / "seeded-sweep-v2" / "native-pm1").exists()
    assert not (Path(args.out_root) / "seeded-sweep").exists()


def test_seeded_sweep_resume_skips_a_run_key_matching_completed_record(tmp_path, monkeypatch):
    bundles_root = tmp_path / "bundles"
    _seeded_sweep_index(bundles_root)
    monkeypatch.setattr(runner, "solver_identity", lambda: IDENTITY)
    args = _seeded_sweep_args(tmp_path)
    out_dir = Path(args.out_root) / "seeded-sweep-v2"
    out_dir.mkdir(parents=True)
    for seed_source in runner.SEED_SOURCES:
        expected_key = runner.seeded_sweep_run_key(
            "native-pm1", _nonce(0), bundles_root, seed_source, runner.SEEDED_SWEEPS, IDENTITY,
        )
        cell_dir = out_dir / "native-pm1"
        cell_dir.mkdir(exist_ok=True)
        record_path = cell_dir / f"{_nonce(0)}__{seed_source}.json"
        samples_path = cell_dir / f"{_nonce(0)}__{seed_source}.npz"
        record_path.write_text(json.dumps({
            "schema": "round2-seeded-sweep-v2", "exit_ok": True, "unsupported": False, "run_key": expected_key,
        }), encoding="utf-8")
        np.savez(samples_path, spins=np.ones((1, 3)), energies=np.zeros(1))

    def never_called(*args, **kwargs):
        raise AssertionError("a matching, completed record must not be re-run")

    monkeypatch.setattr(runner, "execute_seeded_sweep_job", never_called)
    assert cpu.cmd_seeded_sweep(args) == 0


def test_seeded_sweep_resume_refuses_a_run_key_mismatch(tmp_path, monkeypatch):
    # review finding I4: "check the key on resume, refusing a mismatch as the campaign
    # does" -- e.g. quip_msa was rebuilt since this attempt was written.
    bundles_root = tmp_path / "bundles"
    _seeded_sweep_index(bundles_root)
    monkeypatch.setattr(runner, "solver_identity", lambda: IDENTITY)
    args = _seeded_sweep_args(tmp_path)
    out_dir = Path(args.out_root) / "seeded-sweep-v2" / "native-pm1"
    out_dir.mkdir(parents=True)
    stale_identity = {**IDENTITY, "binary_sha256": "a-different-build"}
    stale_key = runner.seeded_sweep_run_key(
        "native-pm1", _nonce(0), bundles_root, "qpu", runner.SEEDED_SWEEPS, stale_identity,
    )
    (out_dir / f"{_nonce(0)}__qpu.json").write_text(json.dumps({
        "schema": "round2-seeded-sweep-v2", "exit_ok": True, "unsupported": False, "run_key": stale_key,
    }), encoding="utf-8")

    with pytest.raises(SystemExit, match="run key"):
        cpu.cmd_seeded_sweep(args)


# --------------------------------------- final review, I5: portfolio-deadline hard kill


def test_portfolio_deadline_hard_kill_in_the_parent_is_a_timeout_not_a_failure(tmp_path, monkeypatch):
    # review finding I5: "a hard kill in the parent is recorded as timeout, not failed" --
    # the hard deadline (300 s) always exceeds the application deadline (10 s), so by the
    # time the parent gives up, the run was certainly already late.
    monkeypatch.setattr(
        cpu.runner, "run_subprocess_with_hard_deadline",
        lambda cmd, hard_deadline_s, **kwargs: (False, hard_deadline_s),
    )
    import argparse

    args = argparse.Namespace(
        out_root=str(tmp_path / "cpu"), p_python="fake-python", pythonpath="", hard_deadline_s=0.01,
    )
    assert cpu.cmd_portfolio_deadline(args) == 0
    record_paths = sorted((Path(args.out_root) / "portfolio-deadline").glob("n*-k*-*.json"))
    assert record_paths
    for record_path in record_paths:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["status"] == "timeout"
        assert record["exit_ok"] is False
