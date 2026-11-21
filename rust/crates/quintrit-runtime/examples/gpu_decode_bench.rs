//! Time of one decode token on the GPU (the CUDA graph of a token, device time only) at given context lengths.
//!
//! ```text
//! cargo run --release --features cuda --example gpu_decode_bench -- <model.gguf> [context ...]
//! ```
//!
//! Prints, per context, the minimum and the median over five blocks of 200 replays and the spread of the blocks. The positions before the timed one are
//! filled with a pseudo-random token sequence. Run it on an otherwise idle GPU.

use quintrit_runtime::InferenceEngine;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(model) = args.next() else {
        eprintln!("usage: gpu_decode_bench <model.gguf> [context ...]");
        std::process::exit(2);
    };
    let contexts: Vec<usize> = args.filter_map(|a| a.parse().ok()).collect();
    let contexts = if contexts.is_empty() { vec![79, 416] } else { contexts };
    let ordinal: usize = std::env::var("QUINTRIT_GPU").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let limit: Option<usize> = std::env::var("QUINTRIT_GPU_CONTEXT").ok().and_then(|v| v.parse().ok());
    let mut engine = match InferenceEngine::load_from_gguf_on_gpu(&model, ordinal, limit) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("cannot load the model on the GPU: {e}");
            std::process::exit(1);
        }
    };
    println!("device: {}", engine.gpu_device_name().unwrap_or_default());
    for context in contexts {
        let mut blocks = engine.time_gpu_decode(context, 5, 200).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
        blocks.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (min, median, max) = (blocks[0], blocks[blocks.len() / 2], blocks[blocks.len() - 1]);
        println!(
            "context {context:>5}: min {min:.3} ms ({:.1} tokens/s), median {median:.3} ms, spread {:.3}",
            1000.0 / min,
            max / min
        );
    }
}
