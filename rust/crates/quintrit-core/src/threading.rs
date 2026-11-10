//! Persistent thread pool management using Rayon.
//!
//! Provides zero-overhead access to pre-spawned worker thread pools
//! to eliminate OS thread creation and destruction overhead during GEMV evaluations.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

#[path = "pool_affinity.rs"]
mod affinity;

/// Pre-allocated lock-free pool storage for thread counts 1..=64.
static THREAD_POOLS_64: [OnceLock<rayon::ThreadPool>; 65] = {
    const INIT: OnceLock<rayon::ThreadPool> = OnceLock::new();
    [INIT; 65]
};

/// Fallback storage for thread counts > 64.
static OVERFLOW_POOLS: OnceLock<Mutex<HashMap<usize, &'static rayon::ThreadPool>>> = OnceLock::new();

/// Retrieves or initializes a persistent `rayon::ThreadPool` configured with `num_threads`.
///
/// Pools are spawned once on first access and persist for the remainder of the process,
/// ensuring zero OS thread allocation or teardown overhead across millions of GEMV calls.
#[inline]
pub fn get_thread_pool(num_threads: usize) -> &'static rayon::ThreadPool {
    let threads = num_threads.max(1);
    if threads <= 64 {
        THREAD_POOLS_64[threads].get_or_init(|| {
            build_pool(threads)
        })
    } else {
        let map_mutex = OVERFLOW_POOLS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut map = map_mutex.lock().unwrap();
        if let Some(&pool) = map.get(&threads) {
            pool
        } else {
            let pool: &'static rayon::ThreadPool = Box::leak(Box::new(build_pool(threads)));
            map.insert(threads, pool);
            pool
        }
    }
}

/// Builds a pool of `threads` workers; the workers are pinned when `QUINTRIT_POOL_PIN=1`.
fn build_pool(threads: usize) -> rayon::ThreadPool {
    let mut builder = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(move |idx| format!("quintrit-worker-{}T-{}", threads, idx));
    if affinity::requested() {
        builder = builder.start_handler(|idx| affinity::pin_worker(idx));
    }
    builder.build().expect("Failed to initialize persistent rayon thread pool")
}
