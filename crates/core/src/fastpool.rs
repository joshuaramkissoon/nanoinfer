//! A lower-latency spin pool (stage 11). Same shape as [`crate::pool`] (stage 6),
//! which stays untouched so stage 6 keeps its measured behaviour, with the two
//! costs our microbenchmark found removed:
//!
//! - **Claiming a chunk is one `fetch_add`**, not a CAS loop. Under contention a
//!   CAS retries; an add always succeeds first time. The claim returns the whole
//!   packed word (`epoch << 32 | n << 16 | next`), so a thread always knows which
//!   job (and whether a real chunk) it got. A claim past the end is harmless
//!   (`next` just counts past `n` until the next publish resets it).
//! - **Completions are counted per thread, not per chunk**: each thread adds its
//!   total once, when it runs out of chunks. On the M4 every atomic on a shared
//!   line can bounce between the P and E clusters; this halves them and takes
//!   the per-chunk one off the critical path.
//!
//! Why a claimed chunk's job is always alive: the job can't finish until that
//! chunk is counted complete, and the dispatcher doesn't publish the next job
//! (or free this one's closure) until every chunk is counted. A thread loads the
//! job pointer *after* its successful claim, so it reads this job's pointer.

use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Condvar, Mutex, OnceLock};

pub const SPIN_LIMIT: u32 = 1 << 12;
pub const YIELD_LIMIT: u32 = 1 << 14;

struct Job<'a> {
    f: &'a (dyn Fn(usize) + Sync),
}

#[repr(align(128))]
struct Padded<T>(T);

struct Shared {
    state: Padded<AtomicU64>,
    completed: Padded<AtomicUsize>,
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
    threads: usize,
}

static POOL: OnceLock<Pool> = OnceLock::new();

impl Shared {
    /// Claim and run chunks until the claimed index is past the end. Returns the
    /// epoch of the last claim and how many chunks this thread ran.
    fn work(&self) -> (u64, usize) {
        let mut done = 0;
        loop {
            let s = self.state.0.fetch_add(1, AcqRel);
            let (idx, n) = (s & 0xffff, (s >> 16) & 0xffff);
            if idx >= n {
                return (s >> 32, done);
            }
            let job = self.job.load(Acquire);
            // Safety: see module docs. This chunk is claimed and not yet counted,
            // so its job (and the closure `job` points to) is alive.
            unsafe { ((*job).f)(idx as usize) };
            done += 1;
        }
    }

    fn worker_loop(&self) {
        let mut seen = self.state.0.load(Acquire) >> 32;
        loop {
            let mut spins = 0u32;
            loop {
                if self.state.0.load(Acquire) >> 32 != seen {
                    break;
                }
                spins += 1;
                if spins < SPIN_LIMIT {
                    std::hint::spin_loop();
                } else if spins < SPIN_LIMIT + YIELD_LIMIT {
                    std::thread::yield_now();
                } else {
                    let g = self.lock.lock().unwrap();
                    self.sleeping.fetch_add(1, SeqCst);
                    let g = if self.state.0.load(SeqCst) >> 32 == seen { self.wake.wait(g).unwrap() } else { g };
                    self.sleeping.fetch_sub(1, SeqCst);
                    drop(g);
                    spins = 0;
                }
            }
            let (epoch, done) = self.work();
            if done > 0 {
                self.completed.0.fetch_add(done, Release);
            }
            seen = epoch;
        }
    }
}

impl Pool {
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Run `f(0..n_chunks)` on the pool and the calling thread. One dispatcher at
    /// a time; not re-entrant.
    pub fn run(&self, n_chunks: usize, f: &(dyn Fn(usize) + Sync)) {
        if self.threads == 1 || n_chunks <= 1 {
            (0..n_chunks).for_each(f);
            return;
        }
        assert!(n_chunks < 1 << 15, "too many chunks");
        let sh = self.shared;
        let Ok(_dispatching) = sh.dispatch.try_lock() else {
            (0..n_chunks).for_each(f);
            return;
        };
        let job = Job { f };
        sh.completed.0.store(0, Relaxed);
        // Safety (lifetime erasure): workers only dereference `job` for claimed,
        // uncounted chunks, and we return only once all are counted.
        sh.job.store(&job as *const Job as *mut Job<'static>, Release);
        let epoch = ((sh.state.0.load(Relaxed) >> 32) + 1) & 0xffff_ffff;
        sh.state.0.store(epoch << 32 | (n_chunks as u64) << 16, SeqCst);
        if sh.sleeping.load(SeqCst) > 0 {
            let _g = sh.lock.lock().unwrap();
            sh.wake.notify_all();
        }
        let (_, mine) = sh.work();
        let mut spins = 0u32;
        while sh.completed.0.load(Acquire) + mine < n_chunks {
            if spins < SPIN_LIMIT {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

/// Create the global pool (`threads` total incl. the caller; 0 = one per core).
pub fn init(threads: usize, on_start: fn()) -> usize {
    POOL.get_or_init(|| {
        let threads = if threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { threads };
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            state: Padded(AtomicU64::new(0)),
            completed: Padded(AtomicUsize::new(0)),
            job: AtomicPtr::new(std::ptr::null_mut()),
            sleeping: AtomicUsize::new(0),
            lock: Mutex::new(()),
            wake: Condvar::new(),
            dispatch: Mutex::new(()),
        }));
        for i in 1..threads {
            std::thread::Builder::new()
                .name(format!("nanoinfer-fast-{i}"))
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
        for round in 0..5000usize {
            let n = 1 + round % 41;
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
