//! Token sampling strategies: greedy argmax, temperature, top-k, and nucleus top-p.

use crate::ops::softmax;
use serde::{Deserialize, Serialize};

/// Hyperparameter configuration for autoregressive token sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Softmax temperature scaling (0.0 = greedy argmax).
    pub temperature: f32,
    /// Top-K candidate truncation (0 = disabled).
    pub top_k: usize,
    /// Nucleus cumulative probability threshold (1.0 = disabled).
    pub top_p: f32,
    /// Random seed for stochastic sampling.
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            seed: 42,
        }
    }
}

impl SamplingConfig {
    /// Creates a deterministic greedy sampling configuration.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            seed: 0,
        }
    }
}

/// Autoregressive token sampler with fast xorshift64 PRNG.
pub struct Sampler {
    config: SamplingConfig,
    rng_state: u64,
}

impl Sampler {
    /// Constructs a new Sampler from configuration.
    pub fn new(config: SamplingConfig) -> Self {
        let initial_state = if config.seed == 0 { 0x5EED_C0DE_1234_5678 } else { config.seed };
        Self {
            config,
            rng_state: initial_state,
        }
    }

    /// Fast 64-bit xorshift pseudo-random number generator.
    fn next_f32(&mut self) -> f32 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        (x as f32) / (u64::MAX as f32)
    }

    /// Samples the next token ID from raw output logits.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        assert!(!logits.is_empty(), "Logits slice cannot be empty");

        // 1. Greedy argmax if temperature is zero or top_k is 1
        if self.config.temperature <= 0.0 || self.config.top_k == 1 {
            let mut max_idx = 0;
            let mut max_val = logits[0];
            for (i, &v) in logits.iter().enumerate().skip(1) {
                if v > max_val {
                    max_val = v;
                    max_idx = i;
                }
            }
            return max_idx as u32;
        }

        // 2. Temperature scaling
        let inv_temp = 1.0 / self.config.temperature;
        let mut scaled_logits: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v * inv_temp))
            .collect();

        // 3. Order the candidates by descending logit whenever top-k or top-p truncates the list: the nucleus filter
        //    below accumulates probabilities from the front, so it must see the most likely tokens first (also when
        //    top-k is disabled or larger than the vocabulary). Ties keep the lower token id first.
        let truncate_k = self.config.top_k > 0 && self.config.top_k < scaled_logits.len();
        if truncate_k || self.config.top_p < 1.0 {
            scaled_logits.sort_by(|a, b| b.1.total_cmp(&a.1));
        }
        if truncate_k {
            scaled_logits.truncate(self.config.top_k);
        }

        // 4. Softmax over remaining candidates
        let mut probs: Vec<f32> = scaled_logits.iter().map(|&(_, v)| v).collect();
        softmax(&mut probs);

        // 5. Nucleus (Top-P) cumulative filtering
        let mut cutoff_idx = probs.len();
        if self.config.top_p < 1.0 {
            let mut cum_prob = 0.0f32;
            for (i, &p) in probs.iter().enumerate() {
                cum_prob += p;
                if cum_prob >= self.config.top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }
        }
        probs.truncate(cutoff_idx);
        scaled_logits.truncate(cutoff_idx);

        // Renormalize probabilities after truncation
        let prob_sum: f32 = probs.iter().sum();
        if prob_sum > 0.0 {
            let inv_sum = 1.0 / prob_sum;
            for p in probs.iter_mut() {
                *p *= inv_sum;
            }
        }

        // 6. Stochastic categorical sampling
        let r = self.next_f32();
        let mut acc = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if r <= acc || i == probs.len() - 1 {
                return scaled_logits[i].0 as u32;
            }
        }

        scaled_logits[0].0 as u32
    }
}
