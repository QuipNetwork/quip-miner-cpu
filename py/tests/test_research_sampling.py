"""The strict research surface: named kernels, observed identity, float64 energies."""

from __future__ import annotations

import threading

import numpy as np
import pytest

import quip_msa

KERNELS = ("cpu-sa", "cpu-msa-unit", "cpu-msa-f64")


def ring(n: int):
    """A unit-weight ring every kernel accepts; ``planted`` satisfies every bond."""

    def spin(i: int) -> int:
        return 1 if ((i * 2_654_435_761) >> 7) & 1 else -1

    planted = np.array([spin(i) for i in range(n)], dtype=np.int8)
    edges = np.array([(i, (i + 1) % n) for i in range(n)], dtype=np.int64)
    j = np.array([-float(planted[u]) * float(planted[v]) for u, v in edges])
    return np.zeros(n), edges, j, planted


def fractional_pair():
    h = np.array([0.125, -0.2], dtype=np.float64)
    edges = np.array([[0, 1]], dtype=np.int64)
    j = np.array([-0.375], dtype=np.float64)
    return h, edges, j


def rescore(spins, h, edges, j):
    return spins @ h + (spins[:, edges[:, 0]] * spins[:, edges[:, 1]] * j).sum(axis=1)


def test_fractional_models_cannot_masquerade_as_unit_msa():
    h = np.array([0.125, -0.2], dtype=np.float64)
    edges = np.array([[0, 1]], dtype=np.int64)
    j = np.array([-0.375], dtype=np.float64)
    with pytest.raises(ValueError, match="unit"):
        quip_msa.Msa().sample_research(
            h, edges, j, kernel="cpu-msa-unit", num_sweeps=8,
        )
    spins, energies, meta = quip_msa.Msa().sample_research(
        h, edges, j, kernel="cpu-msa-f64", num_sweeps=8,
    )
    assert meta["observed_kernel"] == "cpu-msa-f64"
    assert energies.dtype == np.float64
    expected = spins @ h + j[0] * spins[:, 0] * spins[:, 1]
    np.testing.assert_allclose(energies, expected, rtol=1e-12, atol=1e-12)


def test_every_accepted_kernel_reports_its_own_identity():
    h, edges, j, _ = ring(32)
    expected_representation = {
        "cpu-sa": "scalar-int",
        "cpu-msa-unit": "unit-bit-sliced",
        "cpu-msa-f64": "packed-spins-f64-fields",
    }
    for kernel in KERNELS:
        spins, energies, meta = quip_msa.Msa().sample_research(
            h, edges, j, kernel=kernel, num_sweeps=16, num_reads=3, seed=2,
        )
        assert spins.shape == (3, 32) and spins.dtype == np.int8
        assert energies.shape == (3,) and energies.dtype == np.float64
        assert meta["requested_kernel"] == kernel
        assert meta["observed_kernel"] == kernel
        assert meta["representation"] == expected_representation[kernel]
        assert isinstance(meta["rng_scheme"], str) and meta["rng_scheme"]
        assert meta["seeded_reads"] == 0
        if kernel == "cpu-msa-f64":
            assert isinstance(meta["workspace_bytes"], int) and meta["workspace_bytes"] > 0
        else:
            assert meta["workspace_bytes"] is None

    h, edges, j = fractional_pair()
    _, _, meta = quip_msa.Msa().sample_research(
        h, edges, j, kernel="cpu-sa", num_sweeps=16, num_reads=3,
    )
    assert meta["observed_kernel"] == "cpu-sa"
    assert meta["representation"] == "scalar-f64"


def test_energies_are_float64_rescoring_of_the_original_model():
    h, edges, j, _ = ring(24)
    for kernel in KERNELS:
        spins, energies, _ = quip_msa.Msa().sample_research(
            h, edges, j, kernel=kernel, num_sweeps=16, num_reads=70, seed=4,
        )
        np.testing.assert_array_equal(energies, rescore(spins, h, edges, j))


def test_seeded_msa_runs_keep_ground_state_seeds_across_word_boundaries():
    h, edges, j, planted = ring(12)
    for reads in (63, 64, 65):
        seeds = np.stack([planted if r % 2 == 0 else -planted for r in range(reads)])
        # One sweep at the beta cap accepts no uphill move, and a ground state
        # has none downhill, so every seeded lane must come back unchanged.
        spins, _, meta = quip_msa.Msa().sample_research(
            h, edges, j, kernel="cpu-msa-f64", num_sweeps=1, num_reads=reads,
            initial_spins=seeds, start_beta=1e100,
        )
        np.testing.assert_array_equal(spins, seeds)
        assert meta["seeded_reads"] == reads

        spins, _, meta = quip_msa.Msa().sample_research(
            h, edges, j, kernel="cpu-msa-unit", num_sweeps=1, num_reads=reads,
            initial_spins=seeds[:3], start_beta=1e100,
        )
        np.testing.assert_array_equal(spins[:3], seeds[:3])
        assert spins.shape == (reads, 12)
        assert meta["seeded_reads"] == 3


def test_the_merged_msa_kernel_routes_by_model_and_reports_the_arm():
    h, edges, j, _ = ring(32)
    merged = quip_msa.Msa().sample_research(
        h, edges, j, kernel="cpu-msa", num_sweeps=16, num_reads=3, seed=2,
    )
    unit = quip_msa.Msa().sample_research(
        h, edges, j, kernel="cpu-msa-unit", num_sweeps=16, num_reads=3, seed=2,
    )
    assert merged[2]["requested_kernel"] == "cpu-msa"
    assert merged[2]["observed_kernel"] == "cpu-msa-unit"
    np.testing.assert_array_equal(merged[0], unit[0])

    h, edges, j = fractional_pair()
    merged = quip_msa.Msa().sample_research(
        h, edges, j, kernel="cpu-msa", num_sweeps=8, num_reads=4, seed=1,
    )
    f64 = quip_msa.Msa().sample_research(
        h, edges, j, kernel="cpu-msa-f64", num_sweeps=8, num_reads=4, seed=1,
    )
    assert merged[2]["requested_kernel"] == "cpu-msa"
    assert merged[2]["observed_kernel"] == "cpu-msa-f64"
    assert merged[2]["representation"] == "packed-spins-f64-fields"
    np.testing.assert_array_equal(merged[0], f64[0])


def test_zero_sweeps_are_refused_on_every_kernel():
    h, edges, j, _ = ring(8)
    for kernel in KERNELS:
        with pytest.raises(ValueError, match="num_sweeps"):
            quip_msa.Msa().sample_research(h, edges, j, kernel=kernel, num_sweeps=0)


def test_beta_parameters_are_validated_for_every_kernel():
    h, edges, j, planted = ring(8)
    bad_ranges = [(np.nan, 1.0), (-1.0, 1.0), (0.0, 1.0), (0.1, np.inf), (0.1, 1e101), (10.0, 0.1)]
    for kernel in KERNELS:
        for beta_range in bad_ranges:
            with pytest.raises(ValueError, match="beta"):
                quip_msa.Msa().sample_research(
                    h, edges, j, kernel=kernel, num_sweeps=8, beta_range=beta_range,
                )
    for kernel in ("cpu-msa-unit", "cpu-msa-f64"):
        for start_beta in (np.nan, 0.0, -1.0, np.inf, 1e101):
            with pytest.raises(ValueError, match="beta"):
                quip_msa.Msa().sample_research(
                    h, edges, j, kernel=kernel, num_sweeps=8,
                    initial_spins=planted[None, :], start_beta=start_beta,
                )


def test_zero_row_initial_spins_run_cold_and_still_check_the_width():
    h, edges, j, _ = ring(12)
    msa = quip_msa.Msa()
    for kernel in KERNELS:
        cold = msa.sample_research(h, edges, j, kernel=kernel, num_sweeps=8, num_reads=4, seed=3)
        empty = msa.sample_research(
            h, edges, j, kernel=kernel, num_sweeps=8, num_reads=4, seed=3,
            initial_spins=np.zeros((0, 12), dtype=np.int8),
        )
        np.testing.assert_array_equal(empty[0], cold[0])
        assert empty[2]["seeded_reads"] == 0
        with pytest.raises(ValueError, match="initial_spins"):
            msa.sample_research(
                h, edges, j, kernel=kernel, num_sweeps=8,
                initial_spins=np.zeros((0, 5), dtype=np.int8),
            )


def test_a_seeded_scalar_request_is_refused():
    h, edges, j, planted = ring(16)
    with pytest.raises(ValueError, match="seed"):
        quip_msa.Msa().sample_research(
            h, edges, j, kernel="cpu-sa", num_sweeps=8, num_reads=1,
            initial_spins=planted[None, :],
        )


def test_unknown_kernel_names_are_rejected():
    h, edges, j, _ = ring(8)
    for name in ("msa", "cuda-msa", "", "CPU-SA"):
        with pytest.raises(ValueError, match="kernel"):
            quip_msa.Msa().sample_research(h, edges, j, kernel=name, num_sweeps=8)


def test_nan_and_infinity_are_rejected_by_every_kernel():
    h, edges, j, _ = ring(8)
    bad_h = h.copy()
    bad_h[3] = np.nan
    bad_j = j.copy()
    bad_j[0] = np.inf
    for kernel in KERNELS:
        with pytest.raises(ValueError):
            quip_msa.Msa().sample_research(bad_h, edges, j, kernel=kernel, num_sweeps=8)
        with pytest.raises(ValueError):
            quip_msa.Msa().sample_research(h, edges, bad_j, kernel=kernel, num_sweeps=8)


def test_degree_above_63_is_refused_by_the_unit_kernel_only():
    n = 65
    h = np.zeros(n)
    edges = np.array([(0, v) for v in range(1, n)], dtype=np.int64)
    j = np.ones(n - 1)
    with pytest.raises(ValueError, match="unit"):
        quip_msa.Msa().sample_research(h, edges, j, kernel="cpu-msa-unit", num_sweeps=8)
    spins, _, meta = quip_msa.Msa().sample_research(
        h, edges, j, kernel="cpu-msa-f64", num_sweeps=8, num_reads=2,
    )
    assert spins.shape == (2, n)
    assert meta["observed_kernel"] == "cpu-msa-f64"


def test_invalid_seeds_are_rejected():
    h, edges, j, planted = ring(16)
    msa = quip_msa.Msa()
    for kernel in ("cpu-msa-unit", "cpu-msa-f64"):
        with pytest.raises(ValueError):
            msa.sample_research(
                h, edges, j, kernel=kernel, num_sweeps=8,
                initial_spins=np.ones((1, 17), dtype=np.int8),
            )
        zeroed = planted.copy()
        zeroed[2] = 0
        with pytest.raises(ValueError):
            msa.sample_research(
                h, edges, j, kernel=kernel, num_sweeps=8, initial_spins=zeroed[None, :],
            )
        with pytest.raises(ValueError, match="seeds"):
            msa.sample_research(
                h, edges, j, kernel=kernel, num_sweeps=8, num_reads=1,
                initial_spins=np.stack([planted, -planted]),
            )
        with pytest.raises(ValueError, match="start_beta"):
            msa.sample_research(h, edges, j, kernel=kernel, num_sweeps=8, start_beta=2.0)


def test_empty_arrays_are_rejected():
    empty = np.zeros(0)
    no_edges = np.zeros((0, 2), dtype=np.int64)
    for kernel in KERNELS:
        with pytest.raises(ValueError):
            quip_msa.Msa().sample_research(
                empty, no_edges, empty, kernel=kernel, num_sweeps=8,
            )


def test_concurrent_calls_match_serial_results():
    h, edges, j, _ = ring(64)
    msa = quip_msa.Msa()
    serial = {
        kernel: msa.sample_research(
            h, edges, j, kernel=kernel, num_sweeps=32, num_reads=8, seed=9,
        )
        for kernel in KERNELS
    }
    results: dict[tuple[str, int], tuple] = {}
    lock = threading.Lock()

    def work(kernel: str, index: int) -> None:
        out = msa.sample_research(
            h, edges, j, kernel=kernel, num_sweeps=32, num_reads=8, seed=9,
        )
        with lock:
            results[(kernel, index)] = out

    threads = [
        threading.Thread(target=work, args=(kernel, i))
        for kernel in KERNELS
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for (kernel, _), (spins, energies, meta) in results.items():
        np.testing.assert_array_equal(spins, serial[kernel][0])
        np.testing.assert_array_equal(energies, serial[kernel][1])
        assert meta == serial[kernel][2]
