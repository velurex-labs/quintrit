//! Runtime level in Rust: load a ternary GGUF model and stream generated text to the terminal.
//!
//! ```text
//! cargo run --release --bin generate -- model.gguf ["prompt"]
//! ```

use quintrit_runtime::{InferenceEngine, SamplingConfig, TernaryKernelType};
use std::io::Write;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: generate model.gguf [\"prompt\"]");
        std::process::exit(2);
    };
    let prompt = args.next().unwrap_or_else(|| "<|user|>\nWhat is ternary computing?\n<|assistant|>\n".to_string());

    // The library is silent; uncomment to see the loading progress.
    // quintrit_runtime::set_verbose(true);

    // Loading transcodes the weights into RAM. Threads: 0 would not be valid here, so use the available cores.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut engine = InferenceEngine::load_from_gguf(&path, threads, Some(TernaryKernelType::QuintritLut))?;

    // Greedy decoding is deterministic; the closure receives every piece of text as soon as it exists.
    let text = engine.generate_streaming(&prompt, 64, &SamplingConfig::greedy(), |piece| {
        print!("{piece}");
        let _ = std::io::stdout().flush();
    })?;
    println!("\n[{} characters generated]", text.len());
    Ok(())
}
