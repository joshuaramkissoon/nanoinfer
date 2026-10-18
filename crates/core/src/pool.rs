//! A tiny persistent thread pool that *spins* instead of sleeping (stage 6).
//!
//! A decode token is ~49 matmuls, each only 40–150 µs when split across cores.
//! rayon parks idle workers, and waking a parked thread costs tens of µs on
//! macOS. We measured the 2.4 MB attention projection taking *longer* on 4
//! threads than on 1 (62 vs 43 µs). This pool keeps its workers hot:
//!
//! - **dispatch** = bump an epoch counter. Workers busy-wait on it
//!   (`spin_loop`), so they see new work within ~100 ns;
//! - **dynamic chunks**: workers (and the calling thread) claim chunk indices from
//!   one atomic counter, so fast P-cores simply take more chunks than slow E-cores;
//! - **join** = spin until the completed-chunk count reaches the total;
//! - idle workers spin [`SPIN_LIMIT`] times, then `yield_now()` up to
//!   [`YIELD_LIMIT`] times (on a loaded machine that hands the core to whoever
//!   needs it, possibly one of *our* threads that got preempted holding a chunk),
//!   then fall back to a condvar sleep, so an idle process doesn't burn 10 cores.
//!
//! Only exists with the `threads` feature (native builds); wasm stays single-threaded.

use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Condvar, Mutex, OnceLock};

/// Idle `spin_loop` iterations before a waiting thread starts yielding.
pub const SPIN_LIMIT: u32 = 1 << 12;
/// Idle `yield_now` calls (after spinning) before a worker goes to sleep.
pub const YIELD_LIMIT: u32 = 1 << 14;

/// The job a dispatch publishes: lives on the dispatching thread's stack until
/// every chunk has completed.
struct Job<'a> {
    f: &'a (dyn Fn(usize) + Sync),
}

struct Shared {
    /// `epoch << 32 | n_chunks << 16 | next_chunk`. Claiming a chunk is a CAS on
    /// this one word, so a worker can never claim a chunk of a job other than the
    /// one whose epoch it saw, nor one past that job's end.
    state: AtomicU64,
    completed: AtomicUsize,
    job: AtomicPtr<Job<'static>>,
    sleeping: AtomicUsize,
    lock: Mutex<()>,
    wake: Condvar,
    /// Held by the thread currently dispatching. The protocol supports one
    /// dispatcher at a time; a concurrent caller (e.g. two sessions on two
    /// threads, or parallel unit tests) runs its job serially instead.
    dispatch: Mutex<()>,
}

pub struct Pool {
    shared: &'static Shared,
    /// Total threads including the caller.
    threads: usize,
}

static POOL: OnceLock<Pool> = OnceLock::new();

impl Shared {
    /// Claim and run chunks of the job published at `epoch` until none are left.
    fn work(&self, epoch: u64) {
        let job = self.job.load(Acquire);
        let mut s = self.state.load(Acquire);
        loop {
            if s >> 32 != epoch || (s & 0xffff) >= (s >> 16) & 0xffff {
                return;
            }
            match self.state.compare_exchange_weak(s, s + 1, AcqRel, Acquire) {
                Ok(_) => {
                    // Safety: the CAS succeeded for `epoch`, so that job is still in
                    // progress (its chunk isn't completed yet), so the dispatcher
                    // is still blocked and `job` (read after seeing `epoch`, before
                    // this job could finish) points to its live stack frame.
                    unsafe { ((*job).f)((s & 0xffff) as usize) };
                    self.completed.fetch_add(1, Release);
                    s = self.state.load(Acquire);
                }
                Err(cur) => s = cur,
            }
        }
    }

    fn worker_loop(&self) {
        let mut seen = self.state.load(Acquire) >> 32;
        loop {
            let mut spins = 0u32;
            let epoch = loop {
                let e = self.state.load(Acquire) >> 32;
                if e != seen {
                    break e;
                }
                spins += 1;
                if spins < SPIN_LIMIT {
                    std::hint::spin_loop();
                    continue;
                }
                if spins < SPIN_LIMIT + YIELD_LIMIT {
                    std::thread::yield_now();
                    continue;
                }
                // Idle for a while: sleep until the next dispatch.
                let g = self.lock.lock().unwrap();
                self.sleeping.fetch_add(1, SeqCst);
                let g = if self.state.load(SeqCst) >> 32 == seen { self.wake.wait(g).unwrap() } else { g };
                self.sleeping.fetch_sub(1, SeqCst);
                drop(g);
                spins = 0;
            };
            seen = epoch;
            self.work(epoch);
        }
    }
}

impl Pool {
    /// Total threads (workers + the calling thread).
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Run `f(0..n_chunks)` across the pool and the calling thread; returns when
    /// every call has finished. Not re-entrant (don't call `run` from inside `f`).
    /// Concurrent dispatchers are safe: all but one run their job serially.
    pub fn run(&self, n_chunks: usize, f: &(dyn Fn(usize) + Sync)) {
        if self.threads == 1 || n_chunks <= 1 {
            (0..n_chunks).for_each(f);
            return;
        }
        assert!(n_chunks < 1 << 16, "too many chunks");
        let sh = self.shared;
        let Ok(_dispatching) = sh.dispatch.try_lock() else {
            (0..n_chunks).for_each(f);
            return;
        };
        let job = Job { f };
        sh.completed.store(0, Relaxed);
        // Safety (lifetime erasure): workers only dereference `job` while this job
        // has unfinished chunks, and we don't return until all have completed.
        sh.job.store(&job as *const Job as *mut Job<'static>, Release);
        let epoch = ((sh.state.load(Relaxed) >> 32) + 1) & 0xffff_ffff;
        sh.state.store(epoch << 32 | (n_chunks as u64) << 16, SeqCst); // publish: new epoch, next chunk 0
        if sh.sleeping.load(SeqCst) > 0 {
            let _g = sh.lock.lock().unwrap();
            sh.wake.notify_all();
        }
        sh.work(epoch);
        let mut spins = 0u32;
        while sh.completed.load(Acquire) < n_chunks {
            if spins < SPIN_LIMIT {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

/// Create the global pool with `threads` total threads (0 = one per core).
/// `on_start` runs first on every worker thread (e.g. to set a QoS class). Only
/// the first call has an effect; returns the pool's thread count.
pub fn init(threads: usize, on_start: fn()) -> usize {
    POOL.get_or_init(|| {
        let threads = if threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { threads };
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            state: AtomicU64::new(0),
            completed: AtomicUsize::new(0),
            job: AtomicPtr::new(std::ptr::null_mut()),
            sleeping: AtomicUsize::new(0),
            lock: Mutex::new(()),
            wake: Condvar::new(),
            dispatch: Mutex::new(()),
        }));
        for i in 1..threads {
            std::thread::Builder::new()
                .name(format!("nanoinfer-spin-{i}"))
                .spawn(move || {
                    on_start();
                    shared.worker_loop()
                })
                .expect("spawn pool worker");
        }
        Pool { shared, threads }
    })
    .threads
}

/// The global pool, created with one thread per core if [`init`] wasn't called.
pub fn global() -> &'static Pool {
    init(0, || {});
    POOL.get().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_every_chunk_exactly_once_many_times() {
        let pool = global();
        for round in 0..2000usize {
            let n = 1 + round % 37;
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            pool.run(n, &|i| {
                hits[i].fetch_add(1, Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Relaxed) == 1), "round {round}");
        }
    }

    #[test]
    fn concurrent_dispatchers_are_safe() {
        std::thread::scope(|sc| {
            for t in 0..4 {
                sc.spawn(move || {
                    let pool = global();
                    for round in 0..500usize {
                        let n = 1 + (round + t) % 23;
                        let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                        pool.run(n, &|i| {
                            hits[i].fetch_add(1, Relaxed);
                        });
                        assert!(hits.iter().all(|h| h.load(Relaxed) == 1));
                    }
                });
            }
        });
    }
}
