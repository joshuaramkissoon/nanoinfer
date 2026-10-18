//! Thread pool setup for the multithreaded stages.
//!
//! macOS doesn't let you pin threads to cores. What it does honour is a thread's
//! quality-of-service class: `user-interactive` threads go to performance cores
//! when they're free, `background` threads are kept on efficiency cores. That's
//! our only handle on "P vs E cores", so it's a flag.

use clap::ValueEnum;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Qos {
    /// QOS_CLASS_USER_INTERACTIVE: prefer performance cores.
    Interactive,
    /// QOS_CLASS_UTILITY.
    Utility,
    /// QOS_CLASS_BACKGROUND: efficiency cores only.
    Background,
}

impl Qos {
    pub fn name(self) -> &'static str {
        match self {
            Qos::Interactive => "interactive",
            Qos::Utility => "utility",
            Qos::Background => "background",
        }
    }
}

#[cfg(target_os = "macos")]
pub fn set_qos(q: Qos) {
    let class = match q {
        Qos::Interactive => 0x21,
        Qos::Utility => 0x11,
        Qos::Background => 0x09,
    };
    extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    // Safety: plain libSystem call affecting only the current thread.
    unsafe {
        pthread_set_qos_class_self_np(class, 0);
    }
}

#[cfg(not(target_os = "macos"))]
pub fn set_qos(_q: Qos) {}

static QOS: std::sync::Mutex<Qos> = std::sync::Mutex::new(Qos::Interactive);

fn set_global_qos() {
    set_qos(*QOS.lock().unwrap());
}

/// Size both thread pools (rayon for stage 5, the core spin pool for stage 6+)
/// to `threads` (0 = one per core), every thread at QoS `q`. Call once, before
/// any forward. The spin pool's workers start immediately; rayon's lazily.
pub fn init(threads: usize, q: Qos) -> usize {
    *QOS.lock().unwrap() = q;
    set_qos(q);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .start_handler(move |_| set_qos(q))
        .build_global()
        .expect("rayon pool already initialised");
    let n = rayon::current_num_threads();
    nanoinfer_core::pool::init(n, set_global_qos);
    nanoinfer_core::fastpool::init(n, set_global_qos);
    n
}
