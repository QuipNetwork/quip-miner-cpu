//! Sense-reversing spin barrier shared by the parallel kernels.
//!
//! Both chromatic Gibbs and Simulated Bifurcation synchronise many times per
//! sample: once per colour class, and twice per integration step. A mutex and
//! condvar barrier costs more than the work it separates at that rate, so this
//! spins for a bounded time and then yields.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Sense-reversing barrier.
///
/// A class update is one or two microseconds of work per worker, and a sweep
/// crosses one barrier per class. A mutex-and-condvar barrier costs more than
/// the work it separates at that granularity, so this spins instead.
pub(crate) struct SpinBarrier {
    waiting: AtomicUsize,
    sense: AtomicBool,
    workers: usize,
}

impl SpinBarrier {
    pub(crate) fn new(workers: usize) -> Self {
        Self {
            waiting: AtomicUsize::new(0),
            sense: AtomicBool::new(false),
            workers,
        }
    }

    /// `local` carries this worker's expected sense and flips on every call.
    pub(crate) fn wait(&self, local: &mut bool) {
        *local = !*local;
        if self.waiting.fetch_add(1, Ordering::AcqRel) + 1 == self.workers {
            self.waiting.store(0, Ordering::Release);
            self.sense.store(*local, Ordering::Release);
        } else {
            // Spin briefly, then yield. A pure spin collapses under
            // oversubscription: spinning workers hold cores that runnable
            // workers need, and the barrier never completes on time. Measured
            // at 16 workers on a 12-core host, a pure spin ran roughly 90 times
            // slower than one worker.
            let mut spins = 0u32;
            while self.sense.load(Ordering::Acquire) != *local {
                if spins < 512 {
                    std::hint::spin_loop();
                    spins += 1;
                } else {
                    std::thread::yield_now();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hypothesis: every worker leaves the barrier only after all have entered.
    /// A barrier that releases early would let one kernel worker read state
    /// another is still writing, which no downstream test would catch
    /// deterministically.
    #[test]
    fn no_worker_passes_before_all_arrive() {
        let workers = 4;
        let barrier = SpinBarrier::new(workers);
        let stage = AtomicUsize::new(0);
        let seen_early = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for wid in 0..workers {
                let (barrier, stage, seen_early) = (&barrier, &stage, &seen_early);
                s.spawn(move || {
                    let mut sense = false;
                    for round in 1..=50 {
                        if wid == 0 {
                            std::thread::yield_now();
                        }
                        stage.fetch_add(1, Ordering::AcqRel);
                        barrier.wait(&mut sense);
                        if stage.load(Ordering::Acquire) < round * workers {
                            seen_early.fetch_add(1, Ordering::AcqRel);
                        }
                        barrier.wait(&mut sense);
                    }
                });
            }
        });
        assert_eq!(seen_early.load(Ordering::Acquire), 0);
    }
}
