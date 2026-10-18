//! The rules of the game for WebGPU: what one dispatch, one compute pass, one
//! submit and one CPU<->GPU round trip cost, and how fast the GPU reads
//! memory. Runs natively (wgpu on Metal) and in a browser tab (wgpu on the
//! browser's WebGPU), so the two can be put side by side.
//!
//! `now` is a millisecond clock (`Instant` natively, `performance.now()` in a
//! browser), passed in so this crate needs no web bindings.

use crate::gpu::{Gpu, Step, STEP_STRIDE};

const COMMON: &str = include_str!("shaders/common.wgsl");

#[derive(Debug, Clone, Default)]
pub struct ProbeResult {
    /// (label, value, unit)
    pub rows: Vec<(String, f64, &'static str)>,
}

impl ProbeResult {
    fn push(&mut self, label: &str, v: f64, unit: &'static str) {
        self.rows.push((label.to_string(), v, unit));
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// `reps` repetitions of each measurement, median reported.
pub async fn run(gpu: &Gpu, now: &dyn Fn() -> f64, reps: usize) -> ProbeResult {
    let mut out = ProbeResult::default();
    let src = format!("{COMMON}\n{}", include_str!("shaders/probe.wgsl"));
    let k = gpu.kernel("probe", &src, "tick", "w", &[]);
    let buf = gpu.buffer("probe", 256);
    let bg = gpu.bind(&k, &[(&buf, 0, None)]);
    let step = gpu.uniform("step", STEP_STRIDE);
    gpu.queue.write_buffer(&step, 0, bytemuck::bytes_of(&Step::default()));
    let sbg = gpu.step_bind(&step);

    // warm up pipelines and clocks
    for _ in 0..3 {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&k.pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.set_bind_group(1, &sbg, &[0]);
            for _ in 0..1000 {
                p.dispatch_workgroups(1, 1, 1);
            }
        }
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle().await;
    }

    // 1. N dependent dispatches in ONE compute pass, one submit, one wait.
    for &n in &[100usize, 1000, 4000] {
        let mut ts = vec![];
        let mut enc_ts = vec![];
        for _ in 0..reps {
            let t0 = now();
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            {
                let mut p = enc.begin_compute_pass(&Default::default());
                p.set_pipeline(&k.pipeline);
                p.set_bind_group(0, &bg, &[]);
                p.set_bind_group(1, &sbg, &[0]);
                for _ in 0..n {
                    p.dispatch_workgroups(1, 1, 1);
                }
            }
            let cb = enc.finish();
            let t1 = now();
            gpu.queue.submit([cb]);
            gpu.wait_idle().await;
            ts.push((now() - t0) * 1e3 / n as f64);
            enc_ts.push((t1 - t0) * 1e3 / n as f64);
        }
        out.push(&format!("one pass, {n} dispatches: wall per dispatch"), median(ts), "us");
        out.push(&format!("one pass, {n} dispatches: CPU encode per dispatch"), median(enc_ts), "us");
    }

    // 1a. GPU-side time of the same 1000-dispatch pass (timestamp queries).
    if gpu.has(wgpu::Features::TIMESTAMP_QUERY) {
        let n = 1000;
        let qs = gpu.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("ts"), ty: wgpu::QueryType::Timestamp, count: 2 });
        let qbuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts-resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let rb = gpu.readback("ts-rb", 16);
        let mut ts = vec![];
        for _ in 0..reps {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            {
                let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &qs,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
                });
                p.set_pipeline(&k.pipeline);
                p.set_bind_group(0, &bg, &[]);
                p.set_bind_group(1, &sbg, &[0]);
                for _ in 0..n {
                    p.dispatch_workgroups(1, 1, 1);
                }
            }
            enc.resolve_query_set(&qs, 0..2, &qbuf, 0);
            enc.copy_buffer_to_buffer(&qbuf, 0, &rb, 0, 16);
            gpu.queue.submit([enc.finish()]);
            let b = gpu.read(&rb, 16).await;
            let t: &[u64] = bytemuck::cast_slice(&b);
            let period = gpu.queue.get_timestamp_period() as f64;
            ts.push(t[1].wrapping_sub(t[0]) as f64 * period / 1e3 / n as f64);
        }
        out.push("one pass, 1000 dispatches: GPU time per dispatch (timestamps)", median(ts), "us");
    }

    // 1b. same, but re-binding the bind groups before every dispatch (what a
    // real forward does: every op has its own buffers).
    {
        let n = 1000;
        let mut ts = vec![];
        for _ in 0..reps {
            let t0 = now();
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            {
                let mut p = enc.begin_compute_pass(&Default::default());
                for _ in 0..n {
                    p.set_pipeline(&k.pipeline);
                    p.set_bind_group(0, &bg, &[]);
                    p.set_bind_group(1, &sbg, &[0]);
                    p.dispatch_workgroups(1, 1, 1);
                }
            }
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle().await;
            ts.push((now() - t0) * 1e3 / n as f64);
        }
        out.push("one pass, 1000 x (set pipeline + 2 bind groups + dispatch): wall each", median(ts), "us");
    }

    // 2. N dispatches each in its own compute pass, one submit.
    {
        let n = 1000;
        let mut ts = vec![];
        for _ in 0..reps {
            let t0 = now();
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            for _ in 0..n {
                let mut p = enc.begin_compute_pass(&Default::default());
                p.set_pipeline(&k.pipeline);
                p.set_bind_group(0, &bg, &[]);
                p.set_bind_group(1, &sbg, &[0]);
                p.dispatch_workgroups(1, 1, 1);
            }
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle().await;
            ts.push((now() - t0) * 1e3 / n as f64);
        }
        out.push("1000 passes x 1 dispatch, one submit: wall per pass", median(ts), "us");
    }

    // 3. N submits of one dispatch each, waiting once at the end.
    {
        let n = 200;
        let mut ts = vec![];
        for _ in 0..reps {
            let t0 = now();
            for _ in 0..n {
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                {
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_pipeline(&k.pipeline);
                    p.set_bind_group(0, &bg, &[]);
                    p.set_bind_group(1, &sbg, &[0]);
                    p.dispatch_workgroups(1, 1, 1);
                }
                gpu.queue.submit([enc.finish()]);
            }
            gpu.wait_idle().await;
            ts.push((now() - t0) * 1e3 / n as f64);
        }
        out.push("200 submits x 1 dispatch, wait once: wall per submit", median(ts), "us");
    }

    // 4. Round trip: submit one dispatch, wait for it.
    {
        let n = 50;
        let mut ts = vec![];
        for _ in 0..reps {
            let t0 = now();
            for _ in 0..n {
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                {
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_pipeline(&k.pipeline);
                    p.set_bind_group(0, &bg, &[]);
                    p.set_bind_group(1, &sbg, &[0]);
                    p.dispatch_workgroups(1, 1, 1);
                }
                gpu.queue.submit([enc.finish()]);
                gpu.wait_idle().await;
            }
            ts.push((now() - t0) * 1e3 / n as f64);
        }
        out.push("submit 1 dispatch + wait (onSubmittedWorkDone): round trip", median(ts), "us");
    }

    // 5. Round trip with a 4-byte readback (mapAsync), as a greedy loop would.
    {
        let n = 50;
        let rb = gpu.readback("rb", 4);
        let mut ts = vec![];
        for _ in 0..reps {
            let t0 = now();
            for _ in 0..n {
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                {
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_pipeline(&k.pipeline);
                    p.set_bind_group(0, &bg, &[]);
                    p.set_bind_group(1, &sbg, &[0]);
                    p.dispatch_workgroups(1, 1, 1);
                }
                enc.copy_buffer_to_buffer(&buf, 0, &rb, 0, 4);
                gpu.queue.submit([enc.finish()]);
                let _ = gpu.read(&rb, 4).await;
            }
            ts.push((now() - t0) * 1e3 / n as f64);
        }
        out.push("submit 1 dispatch + copy + mapAsync 4 bytes: round trip", median(ts), "us");
    }

    // 6. Read bandwidth: grid-stride vec4 sum over a large buffer.
    {
        let max = gpu.limits.max_storage_buffer_binding_size.min(gpu.limits.max_buffer_size);
        let bytes: u64 = (256u64 << 20).min(max & !0xffff);
        let srcbuf = gpu.buffer("bw-src", bytes);
        // fill with something (zeros are fine for a read probe, but write once
        // so the pages are resident)
        let chunk = vec![1u8; 16 << 20];
        let mut off = 0;
        while off < bytes {
            let n = (bytes - off).min(chunk.len() as u64);
            gpu.queue.write_buffer(&srcbuf, off, &chunk[..n as usize]);
            off += n;
        }
        let groups = 1024u32;
        let outb = gpu.buffer("bw-out", groups as u64 * 256 * 4);
        let bsrc = format!("{COMMON}\n{}", include_str!("shaders/probe_bw.wgsl"));
        let kb = gpu.kernel("probe_bw", &bsrc, "read_bw", "rw", &[]);
        let bgb = gpu.bind(&kb, &[(&srcbuf, 0, None), (&outb, 0, None)]);
        let per = 8;
        let mut gbs = vec![];
        for r in 0..reps + 1 {
            let t0 = now();
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            {
                let mut p = enc.begin_compute_pass(&Default::default());
                p.set_pipeline(&kb.pipeline);
                p.set_bind_group(0, &bgb, &[]);
                p.set_bind_group(1, &sbg, &[0]);
                for _ in 0..per {
                    p.dispatch_workgroups(groups, 1, 1);
                }
            }
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle().await;
            let dt = (now() - t0) / 1e3;
            if r > 0 {
                gbs.push(bytes as f64 * per as f64 / dt / 1e9);
            }
        }
        out.push(&format!("read bandwidth ({} MB buffer, median)", bytes >> 20), median(gbs.clone()), "GB/s");
        out.push("read bandwidth (best)", gbs.iter().copied().fold(0.0, f64::max), "GB/s");
    }
    out
}
