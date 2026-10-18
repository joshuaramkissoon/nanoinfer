//! The thin layer over wgpu: one device, kernels compiled from WGSL with
//! explicit bind-group layouts, buffers, and async readback that works both
//! natively (poll the device) and in a browser (the promise resolves on the
//! event loop).

use std::collections::HashMap;
use std::rc::Rc;
use wgpu::util::DeviceExt;

/// Byte stride of one entry in the per-step uniform buffer (WebGPU's
/// `minUniformBufferOffsetAlignment` is at most 256).
pub const STEP_STRIDE: u64 = 256;

/// Per-forward parameters every kernel can read (group 1, dynamic offset).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Step {
    /// tokens in this forward
    pub n: u32,
    /// absolute position of the first one
    pub pos0: u32,
    /// flash-decoding chunks this step (decode only)
    pub chunks: u32,
    pub _pad: u32,
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub features: wgpu::Features,
    pub limits: wgpu::Limits,
    step_layout: wgpu::BindGroupLayout,
    layouts: std::cell::RefCell<HashMap<String, Rc<wgpu::BindGroupLayout>>>,
    modules: std::cell::RefCell<HashMap<String, Rc<wgpu::ShaderModule>>>,
}

/// One compute kernel: pipeline + its group-0 layout.
pub struct Kernel {
    pub name: String,
    pub pipeline: wgpu::ComputePipeline,
    pub layout: Rc<wgpu::BindGroupLayout>,
}

impl Gpu {
    /// Ask for the adapter's own limits (Chrome's default 128 MiB
    /// `maxStorageBufferBindingSize` is far below what an M4 allows), and the
    /// optional features we know how to use.
    pub async fn new() -> Result<Gpu, String> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::all();
        let instance = wgpu::Instance::new(desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no WebGPU adapter: {e}"))?;
        let want = wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16 | wgpu::Features::TIMESTAMP_QUERY;
        let features = adapter.features() & want;
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("nanoinfer"),
                required_features: features,
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("request_device: {e}"))?;
        let step_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("step"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        Ok(Gpu {
            info: adapter.get_info(),
            device,
            queue,
            features,
            limits,
            step_layout,
            layouts: Default::default(),
            modules: Default::default(),
        })
    }

    pub fn name(&self) -> String {
        format!("{} ({:?})", self.info.name, self.info.backend)
    }

    pub fn has(&self, f: wgpu::Features) -> bool {
        self.features.contains(f)
    }

    pub fn step_layout(&self) -> &wgpu::BindGroupLayout {
        &self.step_layout
    }

    /// Group-0 layout from a signature string: one char per binding,
    /// `r` read-only storage, `w` read-write storage, `u` uniform.
    fn layout(&self, sig: &str) -> Rc<wgpu::BindGroupLayout> {
        if let Some(l) = self.layouts.borrow().get(sig) {
            return l.clone();
        }
        let entries: Vec<_> = sig
            .chars()
            .enumerate()
            .map(|(i, c)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: match c {
                        'r' => wgpu::BufferBindingType::Storage { read_only: true },
                        'w' => wgpu::BufferBindingType::Storage { read_only: false },
                        'u' => wgpu::BufferBindingType::Uniform,
                        _ => panic!("bad binding kind {c}"),
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let l = Rc::new(self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(sig), entries: &entries }));
        self.layouts.borrow_mut().insert(sig.to_string(), l.clone());
        l
    }

    fn module(&self, name: &str, src: &str) -> Rc<wgpu::ShaderModule> {
        if let Some(m) = self.modules.borrow().get(name) {
            return m.clone();
        }
        let desc = wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(src.into()) };
        // Diagnostic only (native): NANOINFER_WGPU_UNCHECKED=1 drops wgpu's
        // injected bounds checks, to measure what robust buffer access costs.
        // Browsers always keep them.
        #[cfg(not(target_arch = "wasm32"))]
        let unchecked = std::env::var("NANOINFER_WGPU_UNCHECKED").is_ok_and(|v| v == "1");
        #[cfg(target_arch = "wasm32")]
        let unchecked = false;
        let m = Rc::new(if unchecked {
            // SAFETY: every kernel guards its own indices against the sizes in
            // its parameters; this mode exists for measurement, not for use.
            unsafe { self.device.create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked()) }
        } else {
            self.device.create_shader_module(desc)
        });
        self.modules.borrow_mut().insert(name.to_string(), m.clone());
        m
    }

    /// Compile `entry` from `src` (module cached by `module`) with the given
    /// override constants. `sig` is the group-0 binding signature.
    pub fn kernel(&self, module: &str, src: &str, entry: &str, sig: &str, constants: &[(&str, f64)]) -> Kernel {
        let m = self.module(module, src);
        let layout = self.layout(sig);
        let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(entry),
            bind_group_layouts: &[Some(&layout), Some(&self.step_layout)],
            immediate_size: 0,
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pl),
            module: &m,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions { constants, zero_initialize_workgroup_memory: false },
            cache: None,
        });
        let cs: Vec<String> = constants.iter().map(|(k, v)| format!("{k}={v}")).collect();
        let name = if cs.is_empty() { entry.to_string() } else { format!("{entry}[{}]", cs.join(",")) };
        Kernel { name, pipeline, layout }
    }

    pub fn buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16).next_multiple_of(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn buffer_init(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        let mut v;
        let data = if data.len() % 4 != 0 || data.is_empty() {
            v = data.to_vec();
            v.resize(data.len().max(16).next_multiple_of(4), 0);
            &v[..]
        } else {
            data
        };
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn uniform(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn readback(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.next_multiple_of(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Bind group for a kernel's group 0 from (buffer, offset, size) triples.
    pub fn bind(&self, k: &Kernel, bufs: &[(&wgpu::Buffer, u64, Option<u64>)]) -> wgpu::BindGroup {
        let entries: Vec<_> = bufs
            .iter()
            .enumerate()
            .map(|(i, (b, off, size))| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: b,
                    offset: *off,
                    size: size.and_then(wgpu::BufferSize::new),
                }),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(&k.name), layout: &k.layout, entries: &entries })
    }

    pub fn step_bind(&self, buf: &wgpu::Buffer) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("step"),
            layout: &self.step_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(16),
                }),
            }],
        })
    }

    /// Wait until the queue is idle (natively: block in poll; in a browser:
    /// await `onSubmittedWorkDone`).
    pub async fn wait_idle(&self) {
        let (tx, rx) = futures_channel::oneshot::channel::<()>();
        self.queue.on_submitted_work_done(move || {
            let _ = tx.send(());
        });
        self.poll();
        let _ = rx.await;
    }

    pub fn poll(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        }
    }

    /// Map a MAP_READ buffer and copy `len` bytes out.
    pub async fn read(&self, buf: &wgpu::Buffer, len: u64) -> Vec<u8> {
        let (tx, rx) = futures_channel::oneshot::channel();
        buf.map_async(wgpu::MapMode::Read, 0..len.next_multiple_of(4), move |r| {
            let _ = tx.send(r);
        });
        self.poll();
        rx.await.expect("map callback dropped").expect("map_async failed");
        let out = buf.get_mapped_range(0..len.next_multiple_of(4)).expect("mapped range")[..len as usize].to_vec();
        buf.unmap();
        out
    }
}

/// Block on a future natively (wgpu's native futures are ready after a poll).
#[cfg(not(target_arch = "wasm32"))]
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    pollster::block_on(f)
}
