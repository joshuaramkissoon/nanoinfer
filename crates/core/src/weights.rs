//! GPT-2 weights, laid out the way our kernels want them.
//!
//! ## A note on layout
//!
//! Hugging Face's GPT-2 uses a `Conv1D` module for its linear layers, which stores
//! weights as `[in, out]`. Almost every inference engine (llama2.c, llm.c, PyTorch's
//! `nn.Linear`) instead stores `[out, in]`: each output feature's weights are one
//! contiguous row, so `y[o] = dot(x, W[o])` walks memory linearly. We transpose
//! once at load time so that every matmul in the engine is "x times W-transposed"
//! with row-major `[out, in]` weights.

use crate::config::Config;
use crate::error::{Error, Result};
use crate::safetensors::SafeTensors;

/// Weights for one transformer block. All matrices are row-major `[out, in]`.
#[derive(Debug, Clone)]
pub struct Layer {
    pub ln_1_w: Vec<f32>,    // [C]
    pub ln_1_b: Vec<f32>,    // [C]
    pub attn_w: Vec<f32>,    // [3C, C]  fused Q, K, V projection
    pub attn_b: Vec<f32>,    // [3C]
    pub proj_w: Vec<f32>,    // [C, C]   attention output projection
    pub proj_b: Vec<f32>,    // [C]
    pub ln_2_w: Vec<f32>,    // [C]
    pub ln_2_b: Vec<f32>,    // [C]
    pub fc_w: Vec<f32>,      // [4C, C]  MLP up-projection
    pub fc_b: Vec<f32>,      // [4C]
    pub fc_proj_w: Vec<f32>, // [C, 4C] MLP down-projection
    pub fc_proj_b: Vec<f32>, // [C]
}

/// A loaded GPT-2 model: config + weights. Immutable once built, so many
/// [`crate::Session`]s can share one `&Model`.
#[derive(Debug, Clone)]
pub struct Model {
    pub config: Config,
    /// Token embeddings `[V, C]`. Also the LM head (GPT-2 ties them): logits = x · wteᵀ.
    pub wte: Vec<f32>,
    /// Learned position embeddings `[n_ctx, C]`.
    pub wpe: Vec<f32>,
    pub layers: Vec<Layer>,
    pub ln_f_w: Vec<f32>,
    pub ln_f_b: Vec<f32>,
}

/// Transpose a row-major `[rows, cols]` matrix into `[cols, rows]`.
fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; m.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = m[r * cols + c];
        }
    }
    out
}

impl Model {
    /// Load from the bytes of an HF GPT-2 `model.safetensors`. The config is
    /// inferred from tensor shapes (GPT-2 always uses 64-wide heads, which is how
    /// we recover `n_head`).
    pub fn from_safetensors(bytes: &[u8]) -> Result<Model> {
        let st = SafeTensors::parse(bytes)?;
        let shape = |n: &str| st.info(n).map(|i| i.shape.clone()).ok_or_else(|| Error::MissingTensor(n.to_string()));
        let wte_shape = shape("wte.weight")?;
        let wpe_shape = shape("wpe.weight")?;
        let (n_vocab, n_embd) = (wte_shape[0], wte_shape[1]);
        let n_ctx = wpe_shape[0];
        let n_layer = (0..).take_while(|i| st.info(&format!("h.{i}.ln_1.weight")).is_some()).count();
        if n_layer == 0 || n_embd % 64 != 0 {
            return Err(Error::Format("does not look like a GPT-2 checkpoint".into()));
        }
        let config = Config { n_vocab, n_ctx, n_embd, n_head: n_embd / 64, n_layer };
        Self::from_safetensors_with_config(&st, config)
    }

    pub fn from_safetensors_with_config(st: &SafeTensors, config: Config) -> Result<Model> {
        let c = config.n_embd;
        // HF Conv1D weight `[in, out]` -> our `[out, in]`.
        let linear = |name: &str, d_in: usize, d_out: usize| -> Result<Vec<f32>> {
            Ok(transpose(&st.f32(name, &[d_in, d_out])?, d_in, d_out))
        };
        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let p = |s: &str| format!("h.{i}.{s}");
            layers.push(Layer {
                ln_1_w: st.f32(&p("ln_1.weight"), &[c])?,
                ln_1_b: st.f32(&p("ln_1.bias"), &[c])?,
                attn_w: linear(&p("attn.c_attn.weight"), c, 3 * c)?,
                attn_b: st.f32(&p("attn.c_attn.bias"), &[3 * c])?,
                proj_w: linear(&p("attn.c_proj.weight"), c, c)?,
                proj_b: st.f32(&p("attn.c_proj.bias"), &[c])?,
                ln_2_w: st.f32(&p("ln_2.weight"), &[c])?,
                ln_2_b: st.f32(&p("ln_2.bias"), &[c])?,
                fc_w: linear(&p("mlp.c_fc.weight"), c, 4 * c)?,
                fc_b: st.f32(&p("mlp.c_fc.bias"), &[4 * c])?,
                fc_proj_w: linear(&p("mlp.c_proj.weight"), 4 * c, c)?,
                fc_proj_b: st.f32(&p("mlp.c_proj.bias"), &[c])?,
            });
        }
        Ok(Model {
            config,
            wte: st.f32("wte.weight", &[config.n_vocab, c])?,
            wpe: st.f32("wpe.weight", &[config.n_ctx, c])?,
            layers,
            ln_f_w: st.f32("ln_f.weight", &[c])?,
            ln_f_b: st.f32("ln_f.bias", &[c])?,
        })
    }
}
