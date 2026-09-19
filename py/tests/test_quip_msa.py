"""The Python surface of the MSA kernel: shapes, seeding, errors, and the GIL."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import quip_msa


def ring(n: int):
    """The conformance driver's proof ring: ``planted`` satisfies every bond."""

    def spin(i: int) -> int:
        return 1 if ((i * 2_654_435_761) >> 7) & 1 else -1

    planted = np.array([spin(i) for i in range(n)], dtype=np.int8)
    edges = np.array([(i, (i + 1) % n) for i in range(n)], dtype=np.int64)
    j = np.array([-float(planted[u]) * float(planted[v]) for u, v in edges])
    return np.zeros(n), edges, j, planted


def test_a_cold_run_returns_one_row_per_read_and_consensus_energies():
    h, edges, j, _ = ring(64)
    spins, energy = quip_msa.Msa().sample(h, edges, j, num_sweeps=32, num_reads=5, seed=3)

    assert spins.shape == (5, 64) and spins.dtype == np.int8
    assert energy.shape == (5,) and energy.dtype == np.int64
    assert set(np.unique(spins)) <= {-1, 1}
    # energy_milli is 1000 * (sum(h s) + sum(J s_u s_v)).
    rescored = 1000 * (spins[:, edges[:, 0]] * spins[:, edges[:, 1]] * j).sum(axis=1)
    assert energy.tolist() == rescored.astype(np.int64).tolist()


def test_the_same_seed_gives_the_same_reads():
    h, edges, j, _ = ring(64)
    msa = quip_msa.Msa()
    first = msa.sample(h, edges, j, num_sweeps=32, num_reads=4, seed=7)
    again = msa.sample(h, edges, j, num_sweeps=32, num_reads=4, seed=7)

    assert np.array_equal(first[0], again[0])


def test_a_seeded_run_keeps_a_ground_state_that_a_cold_run_does_not_reach():
    h, edges, j, planted = ring(512)
    msa = quip_msa.Msa()
    _, cold = msa.sample(h, edges, j, num_sweeps=64, num_reads=1, seed=5)
    spins, warm = msa.sample(
        h, edges, j, num_sweeps=64, num_reads=1, seed=5,
        initial_spins=planted[None, :], start_beta=10.0,
    )

    assert cold[0] > -512_000
    assert warm[0] == -512_000
    assert np.array_equal(spins[0], planted)


def test_reads_past_the_last_seed_start_cold():
    h, edges, j, planted = ring(512)
    _, energy = quip_msa.Msa().sample(
        h, edges, j, num_sweeps=64, num_reads=3, seed=5,
        initial_spins=planted[None, :], start_beta=10.0,
    )

    assert energy[0] == -512_000
    assert energy[1] > -512_000 and energy[2] > -512_000


def test_a_state_of_the_wrong_width_is_a_value_error():
    h, edges, j, planted = ring(64)
    with pytest.raises(ValueError, match="63 spins"):
        quip_msa.Msa().sample(
            h, edges, j, num_sweeps=8, num_reads=1, initial_spins=planted[None, :63]
        )


def test_an_edge_outside_the_problem_is_a_value_error_not_a_crash():
    h, edges, j, _ = ring(8)
    edges = edges.copy()
    edges[0, 1] = 99
    with pytest.raises(ValueError, match="outside 0..8"):
        quip_msa.Msa().sample(h, edges, j, num_sweeps=8, num_reads=1)


def test_a_start_beta_without_states_is_refused():
    h, edges, j, _ = ring(8)
    with pytest.raises(ValueError, match="initial_spins"):
        quip_msa.Msa().sample(h, edges, j, num_sweeps=8, num_reads=1, start_beta=2.0)


def test_a_start_beta_that_is_not_finite_is_refused():
    h, edges, j, planted = ring(8)
    with pytest.raises(ValueError, match="start_beta"):
        quip_msa.Msa().sample(
            h, edges, j, num_sweeps=8, num_reads=1,
            initial_spins=planted[None, :], start_beta=float("nan"),
        )


def test_a_start_beta_that_is_not_positive_is_refused():
    h, edges, j, planted = ring(8)
    with pytest.raises(ValueError, match="start_beta"):
        quip_msa.Msa().sample(
            h, edges, j, num_sweeps=8, num_reads=1,
            initial_spins=planted[None, :], start_beta=0.0,
        )
    with pytest.raises(ValueError, match="start_beta"):
        quip_msa.Msa().sample(
            h, edges, j, num_sweeps=8, num_reads=1,
            initial_spins=planted[None, :], start_beta=-5.0,
        )


def test_a_num_reads_of_zero_is_refused():
    h, edges, j, _ = ring(8)
    with pytest.raises(ValueError, match="num_reads"):
        quip_msa.Msa().sample(h, edges, j, num_sweeps=8, num_reads=0)


def test_the_anneal_runs_with_the_gil_released():
    # While a worker thread anneals, this thread must keep running Python.
    # Counting loop passes does not show that: with the GIL held, this thread
    # still spins for one 5 ms switch interval before the worker takes the GIL.
    # What differs is how much of the call this thread stays responsive for.
    # A gap of 20 ms or more between two passes is time spent parked, so it
    # does not count.
    h, edges, j, _ = ring(4096)
    msa = quip_msa.Msa()

    def anneal():
        msa.sample(h, edges, j, num_sweeps=4096, num_reads=64, seed=1)

    t0 = time.perf_counter()
    anneal()
    alone = time.perf_counter() - t0

    worker = threading.Thread(target=anneal)
    worker.start()
    responsive, prev = 0.0, time.perf_counter()
    while worker.is_alive():
        now = time.perf_counter()
        if now - prev < 0.02:
            responsive += now - prev
        prev = now
    worker.join()

    assert responsive > 0.5 * alone, (
        f"responsive for {responsive * 1000:.0f} ms of a {alone * 1000:.0f} ms anneal"
    )


NONCE = bytes(range(32))


def test_draw_ising_returns_one_value_per_node_and_per_edge():
    h, j = quip_msa.draw_ising(NONCE, 7, 11, [0], [-1000, 1000])
    assert h.shape == (7,) and h.dtype == np.float64
    assert j.shape == (11,) and j.dtype == np.float64


def test_draw_ising_only_draws_allowed_values_in_energy_units():
    h, j = quip_msa.draw_ising(NONCE, 50, 400, [0], [-1000, 1000])
    assert set(np.unique(h)) == {0.0}
    assert set(np.unique(j)) == {-1.0, 1.0}


def test_draw_ising_is_a_pure_function_of_the_nonce():
    first = quip_msa.draw_ising(NONCE, 50, 400, [0], [-1000, 1000])
    again = quip_msa.draw_ising(NONCE, 50, 400, [0], [-1000, 1000])
    other = quip_msa.draw_ising(bytes(32), 50, 400, [0], [-1000, 1000])
    assert np.array_equal(first[1], again[1])
    assert not np.array_equal(first[1], other[1])


def test_draw_ising_spends_one_draw_per_node_before_the_couplings():
    # The protocol draws h first, even from a one-value set. A draw that
    # skipped it would make these two coupling vectors equal.
    _, j_after_5 = quip_msa.draw_ising(NONCE, 5, 64, [0], [-1000, 1000])
    _, j_after_6 = quip_msa.draw_ising(NONCE, 6, 64, [0], [-1000, 1000])
    assert np.array_equal(j_after_5[1:], j_after_6[:-1])


def test_draw_ising_rejects_a_nonce_that_is_not_32_bytes():
    with pytest.raises(ValueError, match="32 bytes"):
        quip_msa.draw_ising(b"short", 5, 5, [0], [-1000, 1000])


def test_draw_ising_rejects_an_empty_allowed_set():
    with pytest.raises(ValueError):
        quip_msa.draw_ising(NONCE, 5, 5, [0], [])


def test_default_beta_range_is_the_range_a_default_run_uses():
    h = np.zeros(4)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    j = np.array([1.0, -1.0, 1.0, 1.0])
    hot, cold = quip_msa.default_beta_range(h, edges, j)
    assert 0.0 < hot < cold
    kernel = quip_msa.Msa()
    default = kernel.sample(h, edges, j, num_sweeps=64, num_reads=8, seed=3)
    explicit = kernel.sample(
        h, edges, j, num_sweeps=64, num_reads=8, seed=3, beta_range=(hot, cold)
    )
    assert np.array_equal(default[0], explicit[0])
