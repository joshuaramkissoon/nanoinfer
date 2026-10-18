//! Model hyper-parameters.

/// The shape of a GPT-2 style model. For GPT-2 124M ("small") these are fixed,
/// but we keep them as data so the same code runs gpt2-medium/large/xl.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Vocabulary size (50257 for GPT-2).
    pub n_vocab: usize,
    /// Maximum context length / number of learned positions (1024).
    pub n_ctx: usize,
    /// Width of the residual stream, a.k.a. d_model (768).
    pub n_embd: usize,
    /// Number of attention heads (12). Head size = n_embd / n_head = 64.
    pub n_head: usize,
    /// Number of transformer blocks (12).
    pub n_layer: usize,
}

impl Config {
    /// LayerNorm epsilon used by GPT-2.
    pub const LN_EPS: f32 = 1e-5;

    pub const GPT2_SMALL: Config = Config { n_vocab: 50257, n_ctx: 1024, n_embd: 768, n_head: 12, n_layer: 12 };

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// Parameter count (weights + biases), matching HF's `num_parameters()`.
    pub fn n_params(&self) -> usize {
        let c = self.n_embd;
        let per_layer = 2 * c          // ln_1
            + c * 3 * c + 3 * c        // attn.c_attn
            + c * c + c                // attn.c_proj
            + 2 * c                    // ln_2
            + c * 4 * c + 4 * c        // mlp.c_fc
            + 4 * c * c + c; // mlp.c_proj
        self.n_vocab * c + self.n_ctx * c + self.n_layer * per_layer + 2 * c
    }

    /// Bytes of weights that must be streamed from memory to decode ONE token,
    /// at `bytes_per_param` (4 for f32). Every layer's matrices are read once, plus
    /// the token embedding table (tied to the LM head, so all of it is read to
    /// produce logits). Only one row of the position table is touched, so we
    /// leave it out. This is the denominator of the "decode ceiling".
    pub fn decode_weight_bytes(&self, bytes_per_param: usize) -> usize {
        (self.n_params() - self.n_ctx * self.n_embd) * bytes_per_param
    }
}
