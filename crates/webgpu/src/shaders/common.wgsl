// Shared by every kernel: the per-forward step parameters (group 1, dynamic
// offset, so one submit can carry several forwards) and small helpers.
struct Step {
  n: u32,      // tokens in this forward
  pos0: u32,   // absolute position of the first
  chunks: u32, // flash-decoding chunks (decode)
  _pad: u32,
}
@group(1) @binding(0) var<uniform> S: Step;

const GELU_K: f32 = 0.7978845608028654; // sqrt(2/pi)
fn gelu(x: f32) -> f32 {
  // tanh form, as GPT-2 (and core) use; tanh written out to avoid
  // implementation-specific tanh precision for large |x|
  let u = GELU_K * (x + 0.044715 * x * x * x);
  let e = exp(-2.0 * abs(u));
  let t = sign(u) * (1.0 - e) / (1.0 + e);
  return 0.5 * x * (1.0 + t);
}
