//! `cargo run --release -p nanoinfer-webgpu --example probe`
use nanoinfer_webgpu::{gpu::block_on, probe, Gpu};
use std::time::Instant;

fn main() {
    let gpu = block_on(Gpu::new()).expect("gpu");
    println!("adapter: {}  features: {:?}", gpu.name(), gpu.features);
    println!(
        "limits: maxStorageBufferBindingSize {} MB, maxBufferSize {} MB, maxStorageBuffersPerShaderStage {}",
        gpu.limits.max_storage_buffer_binding_size >> 20,
        gpu.limits.max_buffer_size >> 20,
        gpu.limits.max_storage_buffers_per_shader_stage
    );
    let t0 = Instant::now();
    let now = move || t0.elapsed().as_secs_f64() * 1e3;
    let r = block_on(probe::run(&gpu, &now, 7));
    for (l, v, u) in r.rows {
        println!("{l:<75} {v:>9.2} {u}");
    }
}
