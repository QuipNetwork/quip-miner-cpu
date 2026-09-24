"""Task 11 step 3: an offline, end-to-end pass through the whole Round 2 pipeline.

One small fixture bundle (``native-pm1``, whole-number ``h`` and unit-valued
``j`` so every controlled kernel is eligible -- see
``round2_runner.unit_kernel_ineligibility_reason``) runs through:

1. Every eligible CPU arm, via ``round2_runner.execute_cpu_job`` -- the
   runner's own per-job execution, called in-process against the REAL
   ``quip_msa`` kernel (never a scripted double: that is what makes this an
   integration test rather than a repeat of ``test_round2_runner.py``'s own
   unit tests, which fake the kernel on purpose). This is the same
   execution ``round2_cpu.py`` spawns as a fresh subprocess per job in
   production; the subprocess/hard-deadline isolation is orthogonal to what
   this test checks and is skipped here.
2. D's audited QPU capture runner (D's own ``scripts/round2_capture.py``,
   read-only, loaded exactly as D's own ``tests/test_round2_capture.py``
   loads it) in ``--mock`` mode: manifest, dry-run, approve, capture, all
   against a scripted double (``_FakeSampler``, modeled on D's own
   ``_Sampler`` test double) that a ``connect`` guard below refuses to run
   in any mode but mock.
3. ``round2_metrics.summarize_arm``, scoring the CPU records.
4. ``round2_report.main``, building a draft report and its figures from the
   CPU records and a small capture-proposal fixture.

Asserts the task-11 brief's three checks: zero real submissions, complete
identity fields on every record, and reconciled (zero) fake QPU spend.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from quip_miner_dwave import regime_io, round2_io
from quip_miner_dwave.ocean import SampleResult

import round2_metrics as metrics
import round2_report as report
import round2_runner as runner

D_ROOT = Path("/home/carback1/quip-miner-dwave")
CAPTURE_SCRIPT = D_ROOT / "scripts" / "round2_capture.py"

CELL = "native-pm1"
NONCE = "aa"
VARIABLE_LABELS = [100, 101, 102]
BUNDLE_EDGES = np.array([[0, 1], [1, 2]], dtype=np.int64)
#: Whole-number h, unit-valued j: eligible for every controlled kernel,
#: including cpu-msa-unit.
BUNDLE_H = np.zeros(3)
BUNDLE_J = np.array([1.0, -1.0])

#: D's solver-properties fixture (``tests/test_round2_capture.py``'s
#: ``PROPERTIES``), copied verbatim: a real D-Wave Advantage timing model,
#: used offline only -- no network call ever reads this from a live chip.
PROPERTIES = {
    "chip_id": "mock-chip",
    "problem_timing_data": {
        "version": "1.0.0",
        "typical_programming_time": 33631.59,
        "reverse_annealing_with_reinit_prog_time_delta": 0.0,
        "reverse_annealing_without_reinit_prog_time_delta": 31.33,
        "default_programming_thermalization": 1000.0,
        "default_annealing_time": 20.0,
        "readout_time_model": "pwl_log_log",
        "readout_time_model_parameters": [
            0.0, 0.7118156236873552, 1.622919864592155, 2.5460233863267376, 3.6610550848533783,
            1.3562242666114566, 1.6328679675785214, 1.7857781469507206, 1.99055000334358, 1.9905608299940198,
        ],
        "qpu_delay_time_per_sample": 60.57,
        "reverse_annealing_with_reinit_delay_time_delta": -44.5,
        "reverse_annealing_without_reinit_delay_time_delta": -41.5,
        "default_readout_thermalization": 0.0,
        "decorrelation_max_nominal_anneal_time": 2000.0,
        "decorrelation_time_range": [0.0, 0.0],
    },
}


def _load_round2_capture():
    """Load D's capture runner exactly as D's own tests do (read-only)."""
    spec = importlib.util.spec_from_file_location("round2_capture", CAPTURE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeSampler:
    """A double of D's ``OceanSampler``, modeled on D's own test double
    (``tests/test_round2_capture.py::_Sampler``): answers with +1 spins at a
    fixed, invented access time. Never touches a network or a credential.
    """

    def __init__(self, live_nodes=VARIABLE_LABELS, live_edges=((100, 101), (101, 102)), access_us=40_000):
        self._live_nodes = list(live_nodes)
        self._live_edges = list(live_edges)
        self.access_us = access_us
        self.calls: List[Dict[str, Any]] = []
        self.closed = False

    @property
    def live_nodes(self):
        return list(self._live_nodes)

    @property
    def live_edges(self):
        return list(self._live_edges)

    def ensure_connected(self):
        pass

    def set_session_topology(self, nodes, edges):
        pass

    def sample(self, nodes, h, edges, j, *, num_reads, anneal_time_us, nonce_seed, label, auto_scale=None):
        assert auto_scale is False, "Round 2 must always submit with auto_scale=False"
        self.calls.append({"nodes": list(nodes), "nonce_seed": nonce_seed.hex(), "anneal_time_us": anneal_time_us})
        return SampleResult(
            spins=np.ones((num_reads, len(nodes)), dtype=np.int8),
            variables=[int(q) for q in nodes],
            energies=[0.0] * num_reads,
            device_access_time_us=self.access_us,
            num_reads=num_reads,
            submitted_on_s=None,
            solved_on_s=None,
            extra={},
        )

    def close(self):
        self.closed = True


def _write_bundle(bundles_root: Path) -> str:
    manifest = {
        "cell": CELL, "nonce": NONCE,
        "identity": {"topology_hash": "topo-fixture", "model_order_hash": "order-fixture"},
        "variable_labels": list(VARIABLE_LABELS), "offset": 0.0,
    }
    return round2_io.write_bundle(
        bundles_root / CELL / NONCE, manifest, {"h": BUNDLE_H, "edges": BUNDLE_EDGES, "j": BUNDLE_J}
    )


def _write_range_audit(range_audit_dir: Path) -> None:
    """A range-audit entry at scale 1.0 (submitted unchanged): a fixture-scale
    analogue of D's own ``tests/test_round2_capture.py::_write_range_audit``.
    """
    model_dir = range_audit_dir / CELL / NONCE
    model_dir.mkdir(parents=True, exist_ok=True)
    ra_manifest = {
        "cell": CELL, "nonce": NONCE, "status": "ok",
        "j_range_used": "j_range", "ceiling": 2.0, "all_zero": False,
        "limiting_constraint": {"kind": "j", "label": 0, "value": 1.0, "bound": "upper"},
        "coefficient_stats": {}, "group_sums": {},
        "chain_strength": None, "chain_strength_ratio_to_logical": None,
        "scales": [{"requested": 1.0, "submitted_scale": 1.0}],
    }
    regime_io.atomic_write_json(model_dir / "manifest.json", ra_manifest)
    regime_io.atomic_savez(model_dir / "arrays.npz", h_100=BUNDLE_H, j_100=BUNDLE_J)


def _job_spec(tmp_path: Path) -> Dict[str, Any]:
    return {
        "cells": [CELL], "nonces": {CELL: [NONCE]}, "requested_scales": [1.0],
        "anneal_us": [80], "reads": 8, "order_seed": 1,
        "bundles_root": str(tmp_path / "bundles"), "range_audit_dir": str(tmp_path / "range-audit"),
        "properties_path": str(tmp_path / "properties.json"),
        "spec_path": None, "embeddings_dir": None, "reservation_margin": 1.5,
    }


def _run_cpu_arms(bundles_root: Path) -> Dict[str, Dict[str, Any]]:
    """Every controlled kernel, run for real (the actual ``quip_msa`` kernel, never
    a double) through ``round2_runner.execute_cpu_job``.
    """
    records: Dict[str, Dict[str, Any]] = {}
    for kernel in runner.CONTROLLED_KERNELS:
        seed, seed_input_hash = runner.seed_for("integration-fixture", kernel, 8, 4, "timing")
        job = runner.CpuJob(
            cell=CELL, nonce=NONCE, kernel=kernel, sweeps=8, reads=4,
            repetition_id=0, repetition_kind=runner.REPETITION_TIMING, variant="timing",
            seed=seed, seed_input_hash=seed_input_hash,
        )
        record, _samples = runner.execute_cpu_job(job, bundles_root)
        records[kernel] = record
    return records


def test_offline_pipeline_end_to_end(tmp_path, monkeypatch):
    # ---- 1. CPU arms: every controlled kernel, in-process, on one small bundle,
    # against the real quip_msa kernel.
    bundles_root = tmp_path / "bundles"
    model_hash = _write_bundle(bundles_root)

    cpu_records = _run_cpu_arms(bundles_root)
    assert set(cpu_records) == set(runner.CONTROLLED_KERNELS)
    for kernel, record in cpu_records.items():
        assert record["unsupported"] is False, f"{kernel} unexpectedly ineligible: {record['unsupported_reason']}"
        assert record["exit_ok"] is True, f"{kernel} failed: {record['error']}"
        assert record["model_hash"] == model_hash

        # identity fields complete on every CPU record
        assert record["requested_kernel"] == kernel
        assert record["seed_input_hash"]
        assert isinstance(record["seed"], int)
        assert record["kernel_input_hash"]
        identity = record["solver_identity"]
        assert identity["package"] == "quip_msa"
        assert identity["error"] is None
        assert identity["version"]
        assert identity["binary_sha256"]

        assert math.isfinite(record["best_energy"])

    arm_summary = metrics.summarize_arm(list(cpu_records.values()), expected=len(cpu_records))
    assert arm_summary["completed"] == len(cpu_records)
    assert arm_summary["missing"] == 0
    assert arm_summary["median_best_energy"] is not None
    assert math.isfinite(arm_summary["median_best_energy"])

    # ---- 2. Fake QPU capture: D's audited runner, --mock only, never a live sampler.
    round2_capture = _load_round2_capture()
    connect_calls: List[bool] = []

    def guarded_connect(mock):
        connect_calls.append(mock)
        assert mock is True, "this test must never construct a real sampler"
        return _FakeSampler()

    monkeypatch.setattr(round2_capture, "connect", guarded_connect)

    _write_range_audit(tmp_path / "range-audit")
    (tmp_path / "properties.json").write_text(json.dumps(PROPERTIES), encoding="utf-8")
    job_spec_path = tmp_path / "job-spec.json"
    job_spec_path.write_text(json.dumps(_job_spec(tmp_path)), encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    assert round2_capture.main(["manifest", "--job-spec", str(job_spec_path), "--out", str(manifest_path)]) == 0
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    round_dir = tmp_path / "round"
    assert round2_capture.main(["dry-run", "--manifest", str(manifest_path), "--round-dir", str(round_dir)]) == 0
    assert regime_io.read_spend(round_dir) == (0, 0)  # nothing charged before any capture runs

    approval_path = tmp_path / "approval.json"
    assert round2_capture.main([
        "approve", "--manifest", str(manifest_path), "--manifest-hash", manifest_hash,
        "--cap-seconds", "10.0", "--approver", "task-11-integration-test", "--out", str(approval_path),
    ]) == 0

    assert round2_capture.main([
        "capture", "--manifest", str(manifest_path), "--approval", str(approval_path),
        "--round-dir", str(round_dir), "--mock",
    ]) == 0

    # zero real submissions: exactly one connect call, and it was mock
    assert connect_calls == [True]

    # reconciled fake charges: a mock capture spends nothing, and the round's own
    # ledger agrees
    assert regime_io.read_spend(round_dir) == (0, 0)

    # The assertion above only proves a MOCK capture journals nothing, which
    # holds even if read_spend's own reconciliation arithmetic were broken
    # (review, M7). Exercise that arithmetic directly with a fake journal: one
    # submit that gets charged, and one still-open submit with no charge yet.
    fake_round_dir = tmp_path / "fake-spend-round"
    regime_io.append_spend(fake_round_dir, regime_io.SUBMIT, CELL, "nonce-a", 80, 50_000)
    regime_io.append_spend(fake_round_dir, regime_io.CHARGE, CELL, "nonce-a", 80, 48_500)
    regime_io.append_spend(fake_round_dir, regime_io.SUBMIT, CELL, "nonce-b", 80, 60_000)
    charged_us, open_jobs = regime_io.read_spend(fake_round_dir)
    assert charged_us == 48_500 + 60_000  # the real charge, plus the still-open submit's own estimate
    assert open_jobs == 1  # nonce-b has no charge yet

    [capture_path] = list((round_dir / CELL).glob("scale-*/qpu-*/*.npz"))
    with np.load(capture_path) as capture:
        # identity fields complete on the capture record
        assert bool(capture["mock"]) is True
        assert str(capture["coefficient_hash"])
        assert str(capture["request_bytes_hash"])
        assert float(capture["requested_scale"]) == 1.0
        assert float(capture["submitted_scale"]) == 1.0

    # ---- 3 & 4. Scoring already checked above (summarize_arm); now report
    # generation from the CPU records plus a small capture-proposal fixture.
    cpu_root = tmp_path / "cpu"
    cell_dir = cpu_root / "pilot" / CELL
    cell_dir.mkdir(parents=True)
    for kernel, record in cpu_records.items():
        name = f"{NONCE}__{kernel}__8__timing-0__timing__attempt0.json"
        (cell_dir / name).write_text(json.dumps(record), encoding="utf-8")

    capture_proposal_path = tmp_path / "capture-proposal.json"
    capture_proposal_path.write_text(json.dumps({
        "arms": [{"regime": CELL, "anneal_us": 80, "captures": 1, "reads_per_capture": 8}],
        "physical_scales": "100% (fixture, no scaling needed)",
        "approval": "approved",
    }), encoding="utf-8")

    out_dir = tmp_path / "report"
    assert report.main([
        "--cpu-root", str(cpu_root), "--run", "pilot",
        "--kernels", *runner.CONTROLLED_KERNELS, "--depths", "8",
        "--capture-proposal", str(capture_proposal_path),
        "--out-dir", str(out_dir), "--cells", CELL,
    ]) == 0

    draft = (out_dir / "REPORT.md").read_text(encoding="utf-8")
    assert CELL in draft
    assert list(out_dir.glob("*.svg"))
