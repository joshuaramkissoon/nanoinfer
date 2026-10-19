// The field guide's content: each term in one plain sentence, the constraint it exists to work
// around, where you'll meet it, and the course chapter that covers it (if any). Ids are URL anchors
// (/learn/field-guide#moe) and are linked from glossary tooltips (GUIDE_LINK in kit.tsx) and concepts.ts.

import type { ReactNode } from "react";
import type { Role } from "../concepts";
import { FormatBits, HeadSharing, ParallelMini } from "./Minis";
import { MoeLab } from "./MoeLab";

export interface Entry {
  id: string;
  term: string;
  aka?: string;
  role: Role;
  /** one plain sentence */
  plain: string;
  /** the constraint it works around, or why it matters */
  why: string;
  /** where you'll see it */
  where: string;
  /** course chapter slug */
  ch?: string;
  visual?: () => ReactNode;
  /** a bigger interactive: shown wide */
  big?: boolean;
}

export interface Group {
  key: string;
  name: string;
  blurb: string;
  entries: Entry[];
}

export const GROUPS: Group[] = [
  {
    key: "models",
    name: "Models and architecture",
    blurb: "What's inside the file, and the design choices that change its cost.",
    entries: [
      { id: "parameters", term: "Parameters", aka: "weights · “7B”", role: "memory", plain: "The learned numbers that make up the model (this site calls them weights); “7B” means 7 billion of them.", why: "Their count × bytes each is the memory you need and the bytes every token reads. It's the first number to look at.", where: "Model names (Llama-3-8B, 70B), pricing tiers, VRAM requirements.", ch: "what-is-inference" },
      { id: "token", term: "Token", role: "time", plain: "A piece of text the model reads or writes: a word, part of a word, or a single byte. GPT-2 knows 50,257 of them.", why: "Each output token costs one forward pass, so tokens are the unit of work, price and latency.", where: "API pricing per million tokens, context limits, tok/s benchmarks. An English token averages about ¾ of a word.", ch: "tokens" },
      { id: "embeddings", term: "Embeddings", role: "memory", plain: "A lookup table that turns each token ID into a vector of numbers (768 for GPT-2, 4,096 or more for big models).", why: "Neural networks do arithmetic on vectors, not IDs. Similar tokens learn similar vectors.", where: "The first and (often) last matrix of every model; also “embedding models” used for search.", ch: "embeddings" },
      { id: "layers", term: "Layers", aka: "blocks · depth", role: "compute", plain: "The model is a stack of same-shaped blocks (12 in GPT-2, 80 in a 70B); each adds an update to every token's vector.", why: "More layers generally give a more capable model. Layers run in sequence for each token, and pipeline parallelism splits a model along that sequence.", where: "Model configs (num_hidden_layers), pipeline-parallel splits.", ch: "layers" },
      { id: "attention", term: "Attention", aka: "self-attention", role: "compute", plain: "The step where each token scores every earlier token (query · key), turns the scores into weights (softmax), and blends their values.", why: "It's the only place tokens exchange information. It needs every earlier token's key and value, hence the KV cache.", where: "Everywhere; FlashAttention, GQA and paged attention are all about making it cheaper.", ch: "attention" },
      { id: "heads", term: "Attention heads", role: "compute", plain: "Attention runs as several smaller, independent attentions side by side (12 per layer in GPT-2), each learning its own pattern.", why: "Several narrow views work better than one wide one at the same cost. Heads are also the natural unit to split across GPUs.", where: "num_attention_heads in configs; tensor parallelism splits by heads.", ch: "attention" },
      { id: "context-window", term: "Context window", aka: "context length", role: "memory", plain: "The most tokens the model can attend to at once: prompt plus answer so far (1,024 for GPT-2, 128k+ today).", why: "Longer contexts mean bigger KV caches (memory) and more attention work per token; it's a memory limit as much as a model limit.", where: "Model cards (“128k context”), pricing for long prompts.", ch: "kv-cache" },
      { id: "moe", term: "Mixture of experts", aka: "MoE · sparse model", role: "compute", plain: "Each layer has many feed-forward networks (experts); a small router sends each token to a few (often 2 of 8, or 8 of 256).", why: "MoE adds weights without adding FLOPs per token. But every expert's weights must stay in memory, and a batch touches most of them, so memory and bandwidth stay large.", where: "Mixtral, DeepSeek-V3, Qwen-MoE, (reportedly) GPT-4; expert parallelism in serving.", visual: () => <MoeLab />, big: true },
      { id: "dense-sparse", term: "Dense vs sparse", role: "compute", plain: "A dense model uses every weight for every token; a sparse (MoE) model uses a routed subset.", why: "Sparse models separate FLOPs per token from total size: they save arithmetic and spend memory capacity.", where: "“47B total, 13B active” in MoE model cards." },
      { id: "gqa", term: "GQA / MQA", aka: "grouped- / multi-query attention", role: "memory", plain: "Several query heads share one set of keys and values: all of them (MQA) or groups of them (GQA), instead of one set each (MHA).", why: "The KV cache stores one set per KV head, so sharing shrinks it 4–8×. That means more users per GPU and faster long-context decode, for a small quality cost.", where: "Llama-2-70B onward, Mistral, most modern models (num_key_value_heads in configs).", visual: () => <HeadSharing /> },
      { id: "rope", term: "RoPE", aka: "rotary position embeddings", role: "compute", plain: "Encodes each token's position by rotating its query and key vectors by an angle that depends on position.", why: "Attention scores then depend on relative distance, and the scheme can be stretched to longer contexts than trained on.", where: "Almost every modern model; “RoPE scaling” for context extension. (GPT-2 uses a learned position table instead.)" },
      { id: "rmsnorm", term: "RMSNorm", role: "compute", plain: "Rescales a vector to a steady size (by its root-mean-square) before each sub-block; a simpler LayerNorm.", why: "Keeps numbers in range through deep stacks, works as well as LayerNorm and costs less. It is small and often fused into the next matrix multiply.", where: "Llama-family configs. GPT-2 uses LayerNorm.", ch: "layers" },
      { id: "swiglu", term: "SwiGLU", role: "compute", plain: "A feed-forward variant with a gate: two up-projections, one passed through a smooth activation and multiplied with the other.", why: "Better quality per weight than GPT-2's plain GELU feed-forward network, at the cost of 3 matrices per FFN instead of 2.", where: "Llama, Mistral and most models since 2023." },
      { id: "forward-pass", term: "Forward pass", role: "compute", plain: "One run of the model over some tokens: through every layer, ending in a score for every possible next token.", why: "It is the unit of execution. Decode does one pass per output token.", where: "Profiles, engine internals, “one forward per step”.", ch: "the-loop" },
    ],
  },
  {
    key: "numbers",
    name: "Numbers",
    blurb: "How many bits each number gets, and what that buys.",
    entries: [
      { id: "formats", term: "fp32 · bf16 · fp16 · fp8 · int8 · int4", role: "memory", plain: "Number formats: how many bits each weight or activation takes, and how those bits split between range and precision.", why: "Bytes per weight set the memory footprint and the decode speed. Halving the bits roughly doubles the speed limit.", where: "Model downloads, engine flags (--dtype bfloat16), GPU spec sheets (TFLOPS per format).", ch: "smaller-numbers", visual: () => <FormatBits /> },
      { id: "quantisation", term: "Quantisation", aka: "quantization", role: "memory", plain: "Storing numbers in fewer bits (8, 4, even 2), usually as small integers times one shared scale per block of weights.", why: "Decode is memory-bound and memory is finite. Fewer bytes per weight give faster tokens and bigger models per GPU, at some cost in accuracy.", where: "Local models (Q4_K_M), serving (FP8, AWQ), most “runs on a laptop” claims.", ch: "smaller-numbers" },
      { id: "wa-quant", term: "Weight-only vs weight + activation", aka: "W4A16 · W8A8", role: "memory", plain: "Weight-only quantisation stores small weights and widens them before multiplying. W8A8 also quantises activations, so the multiply itself runs in 8 bits.", why: "Weight-only helps memory-bound decode. Quantising activations too helps compute-bound prefill and big batches, because low-precision tensor cores are faster.", where: "Notation like W4A16, W8A8, FP8 serving on H100s.", ch: "smaller-numbers" },
      { id: "gguf-awq-gptq", term: "GGUF · AWQ · GPTQ", role: "memory", plain: "GGUF is llama.cpp's file format for quantised models. AWQ and GPTQ choose 4-bit weights that lose less accuracy.", why: "Plain rounding to 4 bits hurts quality. These methods use a little calibration data to protect the weights that matter most.", where: "Hugging Face model names (…-AWQ, …-GPTQ, …Q4_K_M.gguf)." },
      { id: "kv-quant", term: "KV-cache quantisation", role: "memory", plain: "Storing the cached keys and values in 8 or 4 bits instead of 16.", why: "At long contexts and big batches the KV cache fills more memory than the weights, and most of the bytes read per step are KV cache bytes.", where: "Engine flags like kv_cache_dtype=fp8.", ch: "kv-cache" },
    ],
  },
  {
    key: "running",
    name: "Running it",
    blurb: "The serving loop, and the tricks engines play on it.",
    entries: [
      { id: "inference", term: "Inference", role: "compute", plain: "Running a trained model to produce output; the weights are frozen.", why: "Training is paid once; inference is paid on every token for every user, so it dominates cost at scale.", where: "“Inference provider”, “inference engine”, this site.", ch: "what-is-inference" },
      { id: "prefill", term: "Prefill", role: "compute", plain: "The first forward pass, which reads the whole prompt at once and fills the KV cache.", why: "Many tokens share each weight read, so it is compute-bound and fast per token. It sets the time to first token.", where: "Latency breakdowns, “prompt processing” speed, disaggregated serving.", ch: "the-loop" },
      { id: "decode", term: "Decode", aka: "generation", role: "memory", plain: "Every pass after prefill: one new token per request per pass.", why: "It does one token's work per full read of the weights, so it is memory-bound. It sets the streaming speed.", where: "tok/s numbers, TPOT, the reason batching exists.", ch: "the-loop" },
      { id: "kv-cache", term: "KV cache", role: "memory", plain: "Each earlier token's attention keys and values, saved so they aren't recomputed every step.", why: "Without it, every token re-processes the whole text, so the cost grows with length. With it, memory grows with length × users instead.", where: "Engine memory settings, max batch size, “context caching” pricing.", ch: "kv-cache" },
      { id: "flashattention", term: "FlashAttention", role: "compute", plain: "An attention kernel that works through keys and values in tiles held in on-chip SRAM, and never writes the full score matrix to memory.", why: "Plain attention writes and re-reads a T × T score matrix through slow memory. Tiling keeps it on chip, so memory use grows linearly with length.", where: "vLLM, SGLang, TensorRT-LLM and llama.cpp; FlashAttention-2/3, FlashDecoding, FlashInfer." },
      { id: "kernels", term: "Kernel", role: "compute", plain: "A function that runs on the GPU across thousands of GPU threads at once, such as one matrix multiply or one attention step.", why: "A token's time is its kernels' time plus the gaps between them. Most engine work is writing better kernels and removing the gaps.", where: "CUDA, Metal, Triton; profiles listing per-kernel time.", ch: "the-gpu" },
      { id: "fusion", term: "Kernel fusion", role: "memory", plain: "Combining several operations into one kernel so intermediate results stay on-chip instead of round-tripping through memory.", why: "Each separate kernel reads its input from memory and writes its output back. For small operations, that traffic and the launch cost dominate.", where: "torch.compile, TensorRT, “fused RMSNorm + matmul”." },
      { id: "batching", term: "Batching", role: "memory", plain: "Running many requests' next tokens through the model together, so each weight read serves all of them.", why: "Single-user decode leaves the arithmetic idle. Batching uses it, raising throughput many times for a modest latency cost.", where: "Production servers; max_num_seqs and batch-size settings.", ch: "batching" },
      { id: "continuous-batching", term: "Continuous batching", aka: "in-flight batching", role: "time", plain: "Re-forming the batch at every step: finished requests leave and waiting ones join immediately.", why: "In a static batch, short requests wait for the longest one, and new arrivals wait for the whole batch to finish.", where: "vLLM, TGI, TensorRT-LLM (“in-flight batching”).", ch: "batching" },
      { id: "chunked-prefill", term: "Chunked prefill", role: "time", plain: "Splitting a long prompt's prefill into chunks interleaved with other users' decode steps.", why: "One 30k-token prefill would otherwise stall every streaming user's next token for hundreds of milliseconds.", where: "vLLM and SGLang scheduler settings." },
      { id: "paged-attention", term: "Paged attention", aka: "paged KV cache", role: "memory", plain: "Storing the KV cache in fixed-size pages, allocated on demand, with a per-request block table, like virtual memory.", why: "Reserving a contiguous maximum-length region per request wastes most of it. Pages limit the waste to the last part-full page, and let requests share pages.", where: "vLLM (where the name comes from) and most modern engines.", ch: "batching" },
      { id: "prefix-caching", term: "Prefix caching", aka: "prompt caching", role: "memory", plain: "Keeping the KV cache of a common prompt prefix (a system prompt, documents) and reusing it for later requests that start the same way.", why: "Prefill of a shared prefix is repeated work. Skipping it cuts the time to first token and the cost.", where: "API “cached input” discounts, vLLM automatic prefix caching, SGLang RadixAttention." },
      { id: "speculative", term: "Speculative decoding", role: "compute", plain: "A small, fast drafter guesses several tokens; the big model checks them all in one pass and keeps the ones it agrees with.", why: "Single-user decode is memory-bound and leaves arithmetic idle. Checking k tokens costs about one pass, so kept guesses cost almost nothing.", where: "Draft models, Medusa, EAGLE, “lookahead decoding”.", ch: "speculative-decoding" },
      { id: "disaggregated", term: "Disaggregated serving", aka: "prefill/decode split", role: "comm", plain: "Running prefill on one pool of GPUs and decode on another, shipping each request's KV cache between them.", why: "The two phases have opposite bottlenecks and slow each other down on a shared GPU. Separating them lets you tune and scale each one.", where: "DistServe, Mooncake, NVIDIA Dynamo, large API providers." },
    ],
  },
  {
    key: "scale",
    name: "Scale",
    blurb: "Splitting work across many chips, and paying for the talking.",
    entries: [
      { id: "data-parallel", term: "Data parallelism", aka: "replicas", role: "comm", plain: "Several full copies of the model, each serving its own requests.", why: "The simplest way to scale throughput. It needs the model to fit on one GPU (or one group of GPUs).", where: "Autoscaling replicas; in training, every replica processes different data then averages gradients.", visual: () => <ParallelMini mode="data" /> },
      { id: "tensor-parallel", term: "Tensor parallelism", aka: "TP", role: "comm", plain: "Every weight matrix is split across GPUs. Each GPU computes a slice of every multiply, and they all-reduce after each layer.", why: "It fits models bigger than one GPU and cuts per-token latency. The cost is a synchronous exchange every layer, which is only practical over NVLink.", where: "--tensor-parallel-size 8 in vLLM; within one 8-GPU server.", ch: "many-gpus", visual: () => <ParallelMini mode="tensor" /> },
      { id: "pipeline-parallel", term: "Pipeline parallelism", aka: "PP", role: "comm", plain: "Consecutive layers on different GPUs; activations flow from one pipeline stage to the next.", why: "It sends little data (one activation per pipeline-stage boundary), so it works across servers. It adds latency and needs micro-batches to keep every pipeline stage busy.", where: "Multi-node serving of the largest models; training at scale.", ch: "many-gpus", visual: () => <ParallelMini mode="pipeline" /> },
      { id: "expert-parallel", term: "Expert parallelism", aka: "EP", role: "comm", plain: "A MoE model's experts are spread across GPUs; each token goes to the GPU that holds its chosen experts, and back.", why: "It lets very large MoE models fit. The costs are the all-to-all exchange and uneven expert load (hot experts).", where: "DeepSeek-V3/R1 serving, large MoE deployments.", visual: () => <ParallelMini mode="expert" /> },
      { id: "sequence-parallel", term: "Sequence / context parallelism", role: "comm", plain: "One very long sequence split by token positions across GPUs, which exchange keys and values to complete attention.", why: "A million-token context's KV cache doesn't fit on one GPU, and its attention work is too slow there.", where: "Ring attention, long-context training and serving.", visual: () => <ParallelMini mode="sequence" /> },
      { id: "all-reduce", term: "All-reduce", role: "comm", plain: "A collective where every device contributes a partial result and every device ends up with the sum.", why: "Tensor parallelism needs one after each layer, and its latency is on the critical path of every token.", where: "NCCL, MPI; ring and tree algorithms.", ch: "many-gpus" },
      { id: "nvlink", term: "NVLink", aka: "NVSwitch", role: "comm", plain: "NVIDIA's direct GPU-to-GPU link inside a server: about 900 GB/s total per H100 (450 GB/s each way), about 7× PCIe 5.", why: "It makes tensor parallelism across 8 GPUs practical. Between servers, networks give about 50 GB/s.", where: "DGX/HGX servers, NVL72 racks, spec sheets.", ch: "the-hardware" },
      { id: "zero-fsdp", term: "ZeRO / FSDP", role: "comm", plain: "Training techniques that shard the weights, gradients and optimiser state across GPUs instead of copying them to each.", why: "Training needs about 16 bytes of state per weight, and sharding it is how big models train at all. It isn't an inference technique, but the name comes up.", where: "DeepSpeed, PyTorch FSDP: training only." },
    ],
  },
  {
    key: "metrics",
    name: "Metrics",
    blurb: "How speed is measured, and which number matters to whom.",
    entries: [
      { id: "toks", term: "tok/s", aka: "tokens per second", role: "time", plain: "Tokens produced per second: per user (how fast the text streams) or in total (server throughput).", why: "Always ask which one. A server can quote 10,000 tok/s total while each user sees 30.", where: "Every benchmark and leaderboard.", ch: "speed-limit" },
      { id: "ttft", term: "TTFT", aka: "time to first token", role: "time", plain: "How long from sending a request until the first token arrives.", why: "Queueing and prefill set it. It is the “is it responding?” latency users notice most.", where: "SLOs, provider dashboards, chat UX.", ch: "batching" },
      { id: "tpot", term: "TPOT / ITL", aka: "time per output token · inter-token latency", role: "time", plain: "The gap between consecutive output tokens once streaming has started.", why: "The decode step time sets it, and that grows with batch size. It decides whether text streams faster than people read.", where: "SLOs (e.g. TPOT < 50 ms), serving benchmarks." },
      { id: "throughput-latency", term: "Throughput vs latency", role: "time", plain: "Throughput is total tokens per second (what the operator pays for); latency is what one user waits.", why: "Batching raises throughput and worsens latency. Every serving configuration picks a point on that curve.", where: "Capacity planning, pricing tiers (batch APIs are cheaper because they tolerate latency).", ch: "batching" },
      { id: "goodput", term: "Goodput", role: "time", plain: "Throughput counting only requests that met their latency targets.", why: "Raw throughput can rise while users wait longer. Goodput is the number to maximise under latency targets.", where: "Serving papers (DistServe), SLO-aware schedulers." },
      { id: "mfu", term: "MFU", aka: "model FLOPs utilisation", role: "compute", plain: "Useful FLOPs achieved ÷ the chip's peak FLOPs.", why: "It measures how well you use the arithmetic, which matters for training and prefill. Single-user decode MFU is always tiny.", where: "Training reports (“45% MFU”), prefill benchmarks." },
      { id: "mbu", term: "MBU", aka: "model bandwidth utilisation", role: "memory", plain: "Bytes the model must read per second at your speed ÷ the chip's peak bandwidth.", why: "It is the right efficiency metric for decode: 80% MBU means you are close to the speed limit.", where: "Inference benchmarks; this site's “% of the speed limit”.", ch: "speed-limit" },
      { id: "memory-bound", term: "Memory-bound vs compute-bound", role: "memory", plain: "Whether a job waits on bytes arriving or on arithmetic finishing.", why: "It decides which optimisations help: fewer bytes for memory-bound jobs, faster arithmetic for compute-bound ones.", where: "Roofline plots and most performance discussions.", ch: "memory-or-maths" },
    ],
  },
  {
    key: "hardware",
    name: "Hardware",
    blurb: "The physical pieces the numbers live on.",
    entries: [
      { id: "hbm", term: "HBM", aka: "high-bandwidth memory", role: "memory", plain: "Stacks of DRAM packaged beside the GPU die on a very wide bus: 80 GB at about 3.35 TB/s on an H100.", why: "Decode speed is set by bandwidth, so HBM is why a datacentre GPU decodes about 30× faster than the M4.", where: "GPU spec sheets (HBM3, HBM3e), why GPUs are supply-limited.", ch: "the-hardware" },
      { id: "sram", term: "SRAM", aka: "on-chip memory · shared memory", role: "memory", plain: "Small, very fast memory on the chip itself (tens of MB on a GPU) plus registers.", why: "It is about 10× faster than HBM but far too small for the weights. Kernels like FlashAttention are designed to keep their tiles in it.", where: "Kernel design, Groq/Cerebras chips (weights in SRAM)." },
      { id: "tensor-cores", term: "Tensor cores", aka: "matrix units", role: "compute", plain: "Hardware units that multiply small matrix tiles in one instruction, at low precision (bf16, fp8, int8).", why: "Most of a modern GPU's FLOPs are here: about 989 TFLOP/s at bf16 on an H100, against about 67 on its ordinary lanes. They make prefill and big batches fast.", where: "NVIDIA spec sheets; Apple's simdgroup matrix ops; TPUs are mostly this.", ch: "the-gpu" },
      { id: "unified-memory", term: "Unified memory", role: "memory", plain: "One pool of DRAM shared by CPU and GPU (Apple silicon), so neither copies weights to the other.", why: "It gives large capacity cheaply (a Mac can hold a 4-bit 70B model), at laptop bandwidth (about 100 to 800 GB/s, depending on the chip).", where: "Apple M-series, local-LLM communities.", ch: "the-hardware" },
    ],
  },
];

export const ALL_ENTRIES: Entry[] = GROUPS.flatMap((g) => g.entries);
