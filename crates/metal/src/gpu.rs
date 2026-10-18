//! A thin, safe-ish layer over `objc2-metal`: device, compiled shader library,
//! buffers, and a [`Runner`] that encodes dispatches and decides when to
//! commit and wait.
//!
//! The one decision that matters for performance lives in [`Sync`]: either
//! every dispatch gets its own command buffer and the CPU waits for it
//! (`PerOp`, how a first port of CPU code naturally looks), or the whole forward
//! pass is encoded into one command buffer and committed once (`PerForward`).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathMode, MTLResourceOptions,
    MTLSize,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type CmdBuf = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Encoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// All shader source, concatenated and compiled once at startup.
pub const SHADER_SOURCE: &str = concat!(
    include_str!("shaders/common.metal"),
    include_str!("shaders/naive.metal"),
    include_str!("shaders/simd.metal"),
    include_str!("shaders/prefill.metal"),
    include_str!("shaders/quant.metal"),
    include_str!("shaders/fused.metal"),
    include_str!("shaders/lanes.metal"),
    include_str!("shaders/attn.metal"),
    include_str!("shaders/batch.metal"),
    include_str!("shaders/kv8.metal"),
    include_str!("shaders/probe.metal"),
);

/// A GPU buffer in shared (unified) memory: the CPU and the GPU see the same
/// physical pages, so "uploading" is a memcpy and "downloading" is a pointer read.
pub struct Buf {
    pub raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub len: usize,
}

impl Buf {
    pub fn as_ptr<T>(&self) -> *mut T {
        self.raw.contents().as_ptr().cast()
    }
    /// View the buffer as a slice. Only valid while no GPU work writing it is in flight.
    pub fn slice<T: Copy>(&self, n: usize) -> &[T] {
        assert!(n * std::mem::size_of::<T>() <= self.len);
        // Safety: shared-storage buffer, length checked, caller guarantees no
        // concurrent GPU writes (we always wait before reading).
        unsafe { std::slice::from_raw_parts(self.as_ptr::<T>(), n) }
    }
    pub fn write<T: Copy>(&self, offset_elems: usize, data: &[T]) {
        assert!((offset_elems + data.len()) * std::mem::size_of::<T>() <= self.len);
        // Safety: as above.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.as_ptr::<T>().add(offset_elems), data.len()) }
    }
}

pub struct Gpu {
    pub device: Device,
    queue: Queue,
    pipes: HashMap<String, Pipeline>,
}

impl Gpu {
    pub fn new() -> Result<Gpu, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("no command queue")?;
        let opts = MTLCompileOptions::new();
        // IEEE-ish maths (no fast-math): we compare logits to HF at 1e-3, and
        // decode is memory-bound so the fast transcendental paths buy nothing.
        opts.setMathMode(MTLMathMode::Safe);
        let lib = device
            .newLibraryWithSource_options_error(&NSString::from_str(SHADER_SOURCE), Some(&opts))
            .map_err(|e| format!("shader compile failed: {}", e.localizedDescription()))?;
        let mut pipes = HashMap::new();
        for name in lib.functionNames().iter() {
            let f = lib.newFunctionWithName(&name).ok_or("missing function")?;
            let p = device
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| format!("pipeline {name}: {}", e.localizedDescription()))?;
            pipes.insert(name.to_string(), p);
        }
        Ok(Gpu { device, queue, pipes })
    }

    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    pub fn has(&self, name: &str) -> bool {
        self.pipes.contains_key(name)
    }

    pub fn pipe(&self, name: &str) -> &Pipeline {
        self.pipes.get(name).unwrap_or_else(|| panic!("no kernel named {name}"))
    }

    pub fn buffer(&self, bytes: usize) -> Buf {
        let len = bytes.max(16);
        let raw = self.device.newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared).expect("alloc");
        Buf { raw, len }
    }

    pub fn buffer_from<T: Copy>(&self, data: &[T]) -> Buf {
        let len = std::mem::size_of_val(data);
        if len == 0 {
            return self.buffer(16);
        }
        // Safety: pointer and length come from a live slice; Metal copies the bytes.
        let raw = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new(data.as_ptr() as *mut c_void).unwrap(),
                len,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("alloc");
        Buf { raw, len }
    }

    /// Wrap existing page-aligned memory (e.g. an mmap) as a buffer without
    /// copying. `ptr` must be page aligned, `len` a multiple of the page size,
    /// and the memory must outlive the buffer.
    ///
    /// # Safety
    /// The caller guarantees the pointer stays valid and unmodified for the
    /// buffer's lifetime.
    pub unsafe fn buffer_no_copy(&self, ptr: *const u8, len: usize) -> Option<Buf> {
        let raw = self.device.newBufferWithBytesNoCopy_length_options_deallocator(
            NonNull::new(ptr as *mut c_void)?,
            len,
            MTLResourceOptions::StorageModeShared,
            None,
        )?;
        Some(Buf { raw, len })
    }

    pub fn runner(&self, sync: Sync) -> Runner<'_> {
        Runner { gpu: self, pending_tokens: 0, sync, cb: None, enc: None, stats: RunStats::default(), profile: None }
    }
}

/// When the CPU hands work to the GPU and waits for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sync {
    /// One command buffer per dispatch, commit + wait after each.
    PerOp,
    /// Everything into one command buffer; commit + wait once at the end.
    PerForward,
}

/// Counters for one forward pass.
#[derive(Debug, Default, Clone, Copy)]
pub struct RunStats {
    pub dispatches: usize,
    pub command_buffers: usize,
    /// Sum over command buffers of (GPUEndTime - GPUStartTime), seconds.
    pub gpu_seconds: f64,
}

/// Copies of each op encoded back to back when profiling.
pub const PROFILE_REPEAT: usize = 20;

/// A kernel argument.
pub enum Arg<'a> {
    Buf(&'a Buf),
    /// Buffer plus a byte offset into it.
    BufAt(&'a Buf, usize),
    U32(u32),
    F32(f32),
}

pub struct Runner<'g> {
    gpu: &'g Gpu,
    /// Tokens the forward being encoded will append (for argmax's output slot).
    pending_tokens: usize,
    sync: Sync,
    cb: Option<CmdBuf>,
    enc: Option<Encoder>,
    pub stats: RunStats,
    /// When `Some`, every op is timed on its own (see `PROFILE_REPEAT`) and
    /// `(label, seconds)` pairs are appended. Outputs are garbage in this mode
    /// (non-idempotent ops like `x += y` run repeatedly).
    pub profile: Option<Vec<(&'static str, f64)>>,
}

/// A committed command buffer the CPU hasn't waited for yet.
pub struct Pending(CmdBuf);

impl Pending {
    /// Wait for completion; returns GPU seconds.
    pub fn wait(self) -> f64 {
        self.0.waitUntilCompleted();
        if let Some(e) = self.0.error() {
            panic!("Metal command buffer failed: {}", e.localizedDescription());
        }
        self.0.GPUEndTime() - self.0.GPUStartTime()
    }
}

impl Runner<'_> {
    pub fn set_pending_tokens(&mut self, n: usize) {
        self.pending_tokens = n;
    }
    pub fn pending_tokens(&self) -> usize {
        self.pending_tokens
    }

    /// Commit whatever is encoded without waiting.
    pub fn commit(&mut self) -> Option<Pending> {
        let (enc, cb) = (self.enc.take()?, self.cb.take()?);
        enc.endEncoding();
        cb.commit();
        Some(Pending(cb))
    }
    fn encoder(&mut self) -> &Encoder {
        if self.enc.is_none() {
            let cb = self.gpu.queue.commandBuffer().expect("command buffer");
            let enc = cb.computeCommandEncoder().expect("encoder");
            self.cb = Some(cb);
            self.enc = Some(enc);
        }
        self.enc.as_ref().unwrap()
    }

    /// Encode `kernel` over `groups` threadgroups of `threads` threads each.
    /// `tg_mem` is threadgroup memory (bytes) for `threadgroup` argument 0.
    pub fn dispatch_tg(
        &mut self,
        label: &'static str,
        kernel: &str,
        args: &[Arg],
        groups: (usize, usize, usize),
        threads: (usize, usize, usize),
        tg_mem: usize,
    ) {
        // Profiling: time this op alone, as PROFILE_REPEAT back-to-back copies in
        // its own command buffer, so the per-command-buffer start cost and GPU
        // clock ramp are amortised and we see steady-state kernel time.
        let repeat = if self.profile.is_some() { PROFILE_REPEAT } else { 1 };
        if repeat > 1 {
            self.flush();
        }
        for _ in 0..repeat {
            self.encode_one(kernel, args, groups, threads, tg_mem);
        }
        self.stats.dispatches += 1;
        if repeat > 1 {
            let t = self.flush() / repeat as f64;
            self.profile.as_mut().unwrap().push((label, t));
        } else if self.sync == Sync::PerOp {
            self.flush();
        }
    }

    fn encode_one(
        &mut self,
        kernel: &str,
        args: &[Arg],
        groups: (usize, usize, usize),
        threads: (usize, usize, usize),
        tg_mem: usize,
    ) {
        let pipe = self.gpu.pipe(kernel).clone();
        let enc = self.encoder();
        enc.setComputePipelineState(&pipe);
        for (i, a) in args.iter().enumerate() {
            // Safety: buffers outlive the command buffer (we always wait before
            // returning), scalar pointers are valid for the duration of the call.
            unsafe {
                match a {
                    Arg::Buf(b) => enc.setBuffer_offset_atIndex(Some(&b.raw), 0, i),
                    Arg::BufAt(b, off) => enc.setBuffer_offset_atIndex(Some(&b.raw), *off, i),
                    Arg::U32(v) => enc.setBytes_length_atIndex(NonNull::from(v).cast(), 4, i),
                    Arg::F32(v) => enc.setBytes_length_atIndex(NonNull::from(v).cast(), 4, i),
                }
            }
        }
        if tg_mem > 0 {
            // Safety: index 0, length rounded to the required 16 bytes.
            unsafe { enc.setThreadgroupMemoryLength_atIndex(tg_mem.div_ceil(16) * 16, 0) };
        }
        let size = |(width, height, depth)| MTLSize { width, height, depth };
        enc.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(threads));
    }

    /// 1-D convenience: `n` threads total in groups of `tg`.
    pub fn dispatch(&mut self, label: &'static str, kernel: &str, args: &[Arg], n: usize, tg: usize) {
        self.dispatch_tg(label, kernel, args, (n.div_ceil(tg), 1, 1), (tg, 1, 1), 0);
    }

    /// Commit whatever is encoded and wait. Returns that command buffer's GPU time.
    pub fn flush(&mut self) -> f64 {
        let (Some(enc), Some(cb)) = (self.enc.take(), self.cb.take()) else { return 0.0 };
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        if let Some(e) = cb.error() {
            panic!("Metal command buffer failed: {}", e.localizedDescription());
        }
        let t = cb.GPUEndTime() - cb.GPUStartTime();
        self.stats.command_buffers += 1;
        self.stats.gpu_seconds += t;
        t
    }
}
