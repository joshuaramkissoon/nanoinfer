//! Turning logits (one score per vocabulary entry) into the next token.

/// Index of the largest logit. Ties go to the lowest index, like `torch.argmax`.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// The `k` most likely tokens and their softmax probabilities, highest first.
pub fn top_k_probs(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let k = k.min(idx.len());
    if k == 0 {
        return Vec::new();
    }
    let cmp = |a: &u32, b: &u32| logits[*b as usize].total_cmp(&logits[*a as usize]);
    idx.select_nth_unstable_by(k - 1, cmp);
    idx.truncate(k);
    idx.sort_by(cmp);
    idx.into_iter().map(|i| (i, (logits[i as usize] - max).exp() / sum)).collect()
}

/// Greedy (temperature 0) or temperature sampling with an optional top-k cut.
/// Uses a tiny xorshift RNG so results are reproducible from a seed on every
/// platform, including wasm.
#[derive(Debug, Clone)]
pub struct Sampler {
    pub temperature: f32,
    /// 0 = consider the full vocabulary.
    pub top_k: usize,
    rng: u64,
}

impl Sampler {
    pub fn greedy() -> Self {
        Sampler { temperature: 0.0, top_k: 0, rng: 1 }
    }

    pub fn new(temperature: f32, top_k: usize, seed: u64) -> Self {
        Sampler { temperature, top_k, rng: seed.max(1) }
    }

    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let r = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (r >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.temperature <= 0.0 {
            return argmax(logits);
        }
        let k = if self.top_k == 0 { logits.len() } else { self.top_k };
        // Divide logits by temperature, softmax, then draw.
        let scaled: Vec<f32> = logits.iter().map(|l| l / self.temperature).collect();
        let cands = top_k_probs(&scaled, k);
        let total: f32 = cands.iter().map(|c| c.1).sum();
        let mut r = self.next_f32() * total;
        for &(tok, p) in &cands {
            r -= p;
            if r <= 0.0 {
                return tok;
            }
        }
        cands.last().map_or(0, |c| c.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_orders_and_normalises() {
        let l = [0.0, 3.0, 1.0, 2.0];
        let t = top_k_probs(&l, 2);
        assert_eq!(t.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 3]);
        let all: f32 = top_k_probs(&l, 4).iter().map(|x| x.1).sum();
        assert!((all - 1.0).abs() < 1e-6);
        assert_eq!(argmax(&l), 1);
    }
}
