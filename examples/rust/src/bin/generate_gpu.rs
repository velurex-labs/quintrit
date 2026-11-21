//! Runtime level in Rust on an NVIDIA GPU: load an `I2_S` GGUF model onto GPU 0 and stream generated text.
//!
//! ```text
//! cargo run --release --features cuda --bin generate_gpu -- model.gguf ["prompt"] [gpu index]
//! ```
//!
//! Needs the CUDA toolkit at build time and an NVIDIA GPU of compute capability 8.0 or newer. The text is the same as the CPU path produces,
//! bit for bit.

use quintrit_runtime::{InferenceEngine, SamplingConfig};
use std::io::Write;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: generate_gpu model.gguf [\"prompt\"] [gpu index]");
        std::process::exit(2);
    };
    let prompt = args.next().unwrap_or_else(|| "<|user|>\nWhat is ternary computing?\n<|assistant|>\n".to_string());
    let gpu: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(0);

    // The weights are transcoded into the GPU layout while the file is read; the KV cache lives on the GPU.
    // `None` keeps the default context window (4096 tokens, or the model's maximum if that is smaller).
    let mut engine = InferenceEngine::load_from_gguf_on_gpu(&path, gpu, None)?;
    println!("running on {}", engine.gpu_device_name().unwrap_or_default());

    let text = engine.generate_streaming(&prompt, 64, &SamplingConfig::greedy(), |piece| {
        print!("{piece}");
        let _ = std::io::stdout().flush();
    })?;
    println!("\n[{} characters generated]", text.len());
    Ok(())
}
