# 05 · Landscape review: courses, engines, positioning

Researched 19 Oct 2026. Sources: web search, the HN Algolia API (points and comment counts as of today) and the GitHub API
(stars as of today). **[unverified]** marks claims I could not check against a primary source: usually a JS-only page the
fetcher could not render, or a figure that came from an aggregator.

---

## TL;DR

- **The space got crowded in 2025–26.** At least five interactive "how inference works" sites now exist:
  - [Modular LLM Inference Handbook](https://handbook.modular.com/): ~20 widgets.
  - [LLM Inference Explained](https://llm-inference-explained.vercel.app/): 14 chapters, launched Oct 2026, 0 stars.
  - [InferenceVisually](https://github.com/Arthur031221/InferenceVisually).
  - [kernelspace](https://kernelspace.naigap.com/): 68 lessons and Rust labs, Aug 2026.
  - [tiny-llm](https://skyzh.github.io/tiny-llm/): a course that literally says "for systems engineers", 4.8k★.

  There are also two serious free books: [Baseten's *Inference Engineering*](https://www.baseten.co/inference-engineering/)
  (2026) and [*How to Scale Your Model*](https://jax-ml.github.io/scaling-book/inference/). "Inference for systems
  engineers" alone is **not** a gap any more.
- **The gap that is still open is narrower and real: a real engine you can run at every stage of its optimisation, in your
  browser, with the measured number next to a computed speed-of-light line.** Every competitor is one of three things:
  - simulated or modelled: LLM Inference Explained, the Modular widgets, kernelspace's simulators;
  - a single optimised engine with no ladder: WebLLM, transformers.js, Transformer Explainer;
  - a static write-up with real numbers: Andrew Chan's yalm post, tiny-llm.

  Nobody combines a stage switcher, live tok/s in the tab and a fair head-to-head against MLX and llama.cpp.
  Lead with that, not with "a course".
- **Be honest about what's ours and what isn't.** Chapters 1–8 (how a transformer works) compete with Bycroft, Transformer
  Explainer and 3Blue1Brown, and lose on polish and reach. Parts 0 and 2–3 (bytes per token, the climb) are ours.
  Serving at scale and multi-GPU should link out to the scaling book, the Modular handbook, *Inference Engineering*,
  "Inside vLLM" and the Ultra-Scale Playbook.
- **To be more than a portfolio piece, add exercises.** "Implement stage N, the tests check it against HF logits, your
  tok/s lands on the climb chart" is what tiny-llm and kernelspace have and we don't. Our live per-stage engine makes this
  uniquely good: the reader's own kernel, measured in the reader's own browser.
- **Fix the name before launch.** GitHub already has 10+ repos called `nanoinfer` from 2025–26, several of them "GPT-2
  inference from scratch". The domain `nanoinfer.live` named in IA.md does not resolve (NXDOMAIN). The repo README still
  shows only stages 0–1 in its table, and that is the first thing an HN reader will open.

---

## 1. Learning resources

Columns: **Aud** = audience · **Fmt** = format · **Int** = interactivity · **Inf** = depth on inference and serving
(0–3) · **Q** = quality (0–3, my judgement).

### 1a. The closest competitors (inference-specific, interactive or course-shaped)

| Resource | Aud / Fmt | Int | Inf | Q | Better than ours | Ours does that it doesn't |
|---|---|---|---|---|---|---|
| **Modular (ex-BentoML) [LLM Inference Handbook](https://handbook.modular.com/)**. HN [366 pts](https://news.ycombinator.com/item?id=44527947), Jul 2025; [repo](https://github.com/modular/llm-inference-handbook) 459★, CC BY 4.0 [unverified licence] | Engineers deploying LLMs / web handbook, ~45 pages, 7 sections | ~20 widgets: KV-cache and GPU-memory calculators, batching simulator, chunked-prefill scheduler, TTFT/TPOT playground, warp/coalescing visualisers | 3 (production breadth) | 3 | Production breadth (BYOC, on-prem, OpenAI-compatible APIs, prefix caching, disaggregation, KV offload, GPU kernels), a company maintaining it, `.md` for every page plus [llms.txt](https://handbook.modular.com/llms.txt) | Every number comes from a simulator or formula. No real engine, no measured ladder, no "here's the line, here's where we are". Not aimed at people with no ML background |
| **[LLM Inference Explained](https://llm-inference-explained.vercel.app/)** ([repo](https://github.com/BrendanJamesLynskey/llm-inference-explained), created 4 Oct 2026, 0★) | Readers who know one forward pass / 14-chapter site; a sibling "Decoder Explainer"; also covers hardware, kernels and agents | Concept / Maths / Code layers in every chapter; scrubbable animations (play, step, 0.25–4×, keyboard); a disaggregated-serving simulator in a Web Worker; a tiny TS transformer with trained weights | 3 (roofline, paging, FlashAttention, disaggregation, KV transfer) | 2–3 | Covers more serving ground (disaggregated prefill/decode, moving the KV cache, a trade-offs chapter); the three-layer depth toggle | Its own README says hardware and timings are a *roofline cost model* and the serving engine is educational. Nothing is measured. No GPT-2-scale real model |
| **[InferenceVisually](https://github.com/Arthur031221/InferenceVisually)** | Mac owners / simulator per topic + numpy reference + an `mlx-lm` script that "measures the same effect on your Mac" | Simulators; a decode-speed calculator | 2 | ? | Pairs every concept with a real measurement on Qwen3-0.6B-4bit via MLX, e.g. at 8k context the KV cache is 74% of bytes per token | **[unverified]** The repo returned 404 when fetched today (deleted or renamed?); the details above come from search snippets. The measurement runs MLX in a Python script, not in the page, and isn't our own engine |
| **[kernelspace](https://kernelspace.naigap.com/)** ([repo](https://github.com/praveer13/kernelspace), 1★; [Show HN, 2 pts](https://news.ycombinator.com/item?id=49266750), Aug 2026) | Java/Python backend engineers to LLM-serving systems / 68 lessons, 9 tracks, a Rust-from-zero ramp | 8 Rust labs (BPE, KV block manager, radix cache, batching scheduler, toy executor); 9 WASM simulators (roofline, KV pressure, continuous batching); a "fleet week" simulated production fleet | 2–3 | ? (author says it is mostly AI-written) | **Labs and exercises.** The same audience framing as our Part 0. Serving-systems exercises | Simulated, not measured; no real model performance. Got no traction, which shows that "systems engineer → LLM serving" alone doesn't sell |
| **skyzh [tiny-llm](https://skyzh.github.io/tiny-llm/)**: "LLM serving in a week" ([repo](https://github.com/skyzh/tiny-llm) 4.8k★; HN [297 pts](https://news.ycombinator.com/item?id=43820022), Apr 2025) | **"Systems engineers"**, assumes PyTorch / build-it course, 4 weeks, Python + MLX on Apple Silicon | Tests compare your code to MLX; benchmarks against an `--solution mlx` baseline; no interactive visuals | 3 (KV cache, W4, fused kernels, flash attention, batching, then an agent) | 3 | **You build it.** Graded, test-driven, by a well-known author (skyzh also wrote mini-lsm). The closest thing to "what we'd be if we added exercises" | No visuals, no in-browser engine; Qwen in MLX (not from scratch below the array level); no explained speed-of-light framing |
| **Baseten, [*Inference Engineering*](https://www.baseten.co/inference-engineering/)** (Philip Kiely, 2026; 256 pp, ~46k words; free PDF/EPUB/audiobook, paid print) | "Technically-minded people of all backgrounds" / book | None | 3 (hardware, software stack up to vLLM/SGLang/TRT-LLM/Dynamo, techniques, production) | 3 | Production authority; covers modalities, autoscaling and Dynamo; handed out free at conferences. HN [3 pts](https://news.ycombinator.com/item?id=49364934): it spreads through Baseten's own channels, not HN | Static; no measurements you can reproduce; kernels only in passing ([ToC](https://www.baseten.co/inference-engineering/)) |
| **Google DeepMind, [*How to Scale Your Model*](https://jax-ml.github.io/scaling-book/inference/)**, Part 7 (HN [185 pts](https://news.ycombinator.com/item?id=42936910)) | ML systems engineers / textbook | Collapsible "pop quiz" answers only | 3 (rooflines, KV sizing, sharded inference, disaggregation, critical batch ≈240 on v5e) | 3 | Rigour; exercises with worked answers; the definitive treatment of the maths we lean on in Part 0 | TPU-centric, maths-heavy, nothing runs. We are the "feel it" companion to their "derive it" |

### 1b. Transformer-mechanics visualisers (overlap with our Part 1)

| Resource | Aud / Fmt | Int | Inf | Q | Better than ours | Ours does that it doesn't |
|---|---|---|---|---|---|---|
| **Brendan Bycroft, [LLM Visualization](https://bbycroft.net/llm)**. HN [1,592 pts](https://news.ycombinator.com/item?id=38505211) (Dec 2023), re-posted at [640](https://news.ycombinator.com/item?id=45130260) (Sep 2025); [repo](https://github.com/bbycroft/llm-viz) 5.6k★ | General technical readers / one 3D scene with a guided walkthrough | A 3D walk through nano-gpt / GPT-2 / GPT-3 with every tensor drawn to scale; guided camera [content from memory; the page didn't render for the fetcher] | 0 (forward pass only, no performance) | 3 | Unmatched "wow". One continuous spatial model of the network, and HN commenters praise the 3D control scheme. A 2-year-old page still did 640 points | No performance, no KV cache, no batching, no hardware. Our "every word reads every weight" idea is exactly what it lacks |
| **Polo Club, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)** ([repo](https://github.com/poloclub/transformer-explainer) 9k★; [AAAI 2025 paper](https://ojs.aaai.org/index.php/AAAI/article/view/35347) reports 125k users; HN [105 pts](https://news.ycombinator.com/item?id=41221718)) | Non-experts / single page | Real GPT-2 in the browser; type text; temperature / top-k; expand any block | 0–1 (sampling only) | 3 | Academic credibility, adoption in classrooms, extreme simplicity | Same model, but it's one opaque ONNX runtime with no stages and no speed. Our ch.1 and ch.7 cover similar ground; theirs is better known |
| **Jay Alammar, [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) / Illustrated GPT-2** (now the *Hands-On LLMs* book) | Beginners / illustrated essays | None | 0 | 3 | The canonical diagrams, which every course cites | Static; nothing about inference cost [content from memory] |
| **3Blue1Brown, [Deep Learning ch. 5–7](https://www.3blue1brown.com/lessons/attention)** (attention lesson on HN: [999 pts](https://news.ycombinator.com/item?id=40035514)) | Everyone / video | None | 0 | 3 | Best-in-class intuition for embeddings, attention, and MLPs as fact stores | No systems content. Link to it from ch.3–6 rather than competing |
| **[llm.manogya.dev](https://llm.manogya.dev)**, "Every number in a tiny GPT" ([Show HN, 8 pts](https://news.ycombinator.com/item?id=49991390), Oct 2026) | Learners / dry run of one token through a tiny GPT, to RLHF | **[unverified]** JS-only page | 0 | ? | n/a | Performance |
| **[MicroLLM Lab](https://stateofutopia.com/experiments/microllmlab/)** (HN [284 pts](https://news.ycombinator.com/item?id=49882781), Sep 2026) | Curious devs / 7 tiny LLMs in the browser via WebGPU | Playground | 0 | 1–2 | Shows the appetite: "real model in the browser" alone got 284 points | It's a cautionary tale. Top comments say "text too small… way too dense… mostly useless, AI-generated information", and "Firefox: GPUShaderStage is not defined" |

### 1c. Build-it-yourself and university courses

| Resource | Aud / Fmt | Int | Inf | Q | Better than ours | Ours does that it doesn't |
|---|---|---|---|---|---|---|
| **Karpathy: [Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) (24.9k★), ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) (HN [1,110](https://news.ycombinator.com/item?id=34414716)), [llm.c](https://github.com/karpathy/llm.c) (31k★; HN [1,050](https://news.ycombinator.com/item?id=39973467)), [nanochat](https://github.com/karpathy/nanochat) (58.7k★; HN [1,523](https://news.ycombinator.com/item?id=45569350)), [microgpt](https://karpathy.github.io/2026/02/12/microgpt/) (Feb 2026, 200 lines of Python)** | Programmers / video + code | Code you run | 1 (nanochat has an `Engine` with a KV cache; llm.c is training-first) | 3 | Reach and trust. The "build it in one file" genre is his | None of them explain *why it's slow* or chart the speed-of-light; we are the inference sequel to "Let's build GPT" |
| **Stanford [CS336](https://cs336.stanford.edu/)** (HN [558 pts](https://news.ycombinator.com/item?id=48357075), Jun 2026) | Grad students / lectures + heavy assignments | Assignments | 2 (one inference lecture: KV traffic, arithmetic intensity, quantisation, speculative decoding; [summary, unverified](https://summify.io/discover/stanford-cs336-language-modeling-from-scratch-spring-2026-le-EfM546/)) | 3 | Assignments with real rigour; a Stanford brand | One lecture on inference vs our 23 chapters; nothing interactive |
| **MIT [6.5940 TinyML & Efficient AI](https://hanlab.mit.edu/courses/2026-fall-65940)** (Song Han) | Grad students / 25 lectures + 5 labs, videos free | Labs (quantise and deploy Llama-2-7B on a laptop in the 2024 edition) | 3 on compression (quantisation, pruning, AWQ-style), 2 on serving | 3 | Research depth on quantisation; real labs | Not for systems engineers without ML; no in-browser engine |
| **Hugging Face [LLM course](https://huggingface.co/learn/llm-course/en/chapter2/8)** (ch.2.8 "Optimized inference deployment": TGI / vLLM / llama.cpp) and **[smol course](https://github.com/huggingface/smol-course)** (6.8k★; alignment and fine-tuning) | Practitioners / notebooks | Notebooks | 1 | 2 | Practical "which framework" guidance | They teach tools; we teach why the tools exist |
| **[Raschka, LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)** (106.8k★) | Programmers / book + notebooks | Notebooks | 1 (bonus KV-cache material) | 3 | Huge audience, step-by-step | Inference performance |

### 1d. Reference texts, blogs and vendor explainers

| Resource | Aud / Fmt | Inf | Q | Note |
|---|---|---|---|---|
| **Andrew Chan, ["Fast LLM Inference From Scratch"](https://andrewkchan.dev/posts/yalm.html)** ([yalm](https://github.com/andrewkchan/yalm), 604★; HN [344 pts / 57 comments](https://news.ycombinator.com/item?id=42417857)) | Systems programmers / one long post | 3 for single-batch | 3 | **Our closest spiritual ancestor.** It's a measured ladder (0.6 → 63.8 tok/s on a 4090 vs llama.cpp 61.0 and calm 66.0), with speed-of-light maths, nsys/ncu evidence and negative results kept in. What we add: it's live and in the browser, readers can switch stages, and the series continues through quantisation, GPU, batching and speculative decoding. **Expect to be compared to it.** |
| **Aleksa Gordić, ["Inside vLLM"](https://vllm.ai/blog/anatomy-of-vllm)** (Sep 2025; on vLLM's blog) | Engine contributors | 3 (V1 engine, paged attention, prefix caching, spec decoding, multi-node) | 3 | The link for "how a real serving engine is built". Our ch.17–18 should point here |
| **[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)** (16k★; HN [125](https://news.ycombinator.com/item?id=44352615)) + Neutree's [walkthrough](https://neutree.ai/blog/nano-vllm-part-1) (HN [271](https://news.ycombinator.com/item?id=46855447)); **[mini-sglang](https://github.com/sgl-project/mini-sglang)** (5.3k★) | Python serving people | 3 (serving) | 3 | The readable serving codebases. The walkthrough thread's top comment, "the whole thing feels AI written", was retracted after the author replied. That's a risk for us too |
| **Lilian Weng, ["Large Transformer Model Inference Optimization"](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)** (2023) | Researchers | 2 (distillation, quantisation, pruning, sparsity, efficient attention) | 3 | A literature survey, now dated on serving. A good "further reading" link for ch.15 |
| **Chip Huyen, *AI Engineering* [ch. 9 "Inference Optimization"](https://oreilly.com/library/view/ai-engineering/9781098166298/ch09.html)** | App engineers / book | 2 (compute- vs memory-bound, metrics, batching, parallelism, prompt caching) | 3 | Wide readership. Its framing (model / hardware / service levels) matches our constraint map |
| **Stas Bekman, [ML Engineering Open Book: Inference](https://github.com/stas00/ml-engineering/tree/master/inference)** (19.5k★) | Practitioners / reference | 2 (prefill/decode, memory anatomy, batching, paged attention, spec decoding, TP/PP, TTFT/TPOT metrics) | 3 | A dense checklist; no visuals. Field-guide cross-links |
| **HF [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)** (HN [33](https://news.ycombinator.com/item?id=43105474); [~100 pp, 4k experiments, interactive plots](https://the-decoder.com/hugging-face-explains-how-train-large-ai-models-in-the-ultra-scale-playbook/)) | Training engineers | 1 (training) | 3 | Link from ch.19 for parallelism. Training, not inference |
| **HF ["Continuous batching from first principles"](https://huggingface.co/blog/continuous_batching)** (Nov 2025) | Practitioners | 2 | 3 | Derives ragged batching from attention masks. Our ch.17 should link it |
| **Modal: [GPU Glossary](https://modal.com/gpu-glossary)** (HN [230](https://news.ycombinator.com/item?id=42675529)), **[LLM Engine Advisor / Almanac](https://modal.com/llm-almanac)** (vLLM vs SGLang vs TRT-LLM benchmarks, ~10 models × ~10 context lengths) | Practitioners | 2 | 3 | The glossary is the model for our field guide (stable URLs, one term per page, SEO). The almanac is the model for "measured, open methodology" |
| **Anyscale, ["How continuous batching enables 23x throughput"](https://www.anyscale.com/blog/continuous-batching-llm-inference)** (2023) | Practitioners | 2 | 3 | The classic static-vs-continuous explainer. Note that its 23× is against naive HF serving |
| kipply / Carol Chen, ["Transformer Inference Arithmetic"](https://kipp.ly/transformer-inference-arithmetic/) (HN [65](https://news.ycombinator.com/item?id=30957012)); ezyang, ["DeepSeek-V3: From Roofline to Reality"](https://deepseek-v3.ezyang.com/index.html) (Sep 2026); Adaptive ML, ["Speculative Decoding, Visualized"](https://www.adaptive-ml.com/post/speculative-decoding-visualized) (Aug 2026); a [roofline calculator repo](https://github.com/monamishra95/roofline-model-llm-inference) | Mixed | 2–3 | 2–3 | Napkin-maths and visual prior art for ch.9–10 and ch.18 |
| Fireworks / Together explainers | — | — | — | **[unverified]** Searches didn't surface a canonical one. Together's [long-context speculative decoding post](https://together.ai/blog/speculative-decoding-for-high-throughput-long-context-inference) is the relevant one |

---

## 2. Engines

| Engine | Lang / targets | Purpose | Size / activity | Where nanoinfer sits relative to it |
|---|---|---|---|---|
| [llama2.c](https://github.com/karpathy/llama2.c) | C, CPU | Educational, one file | 20.2k★, dormant since 2024; HN [707](https://news.ycombinator.com/item?id=36838051) | Our stage 0–1 ancestor. We go much further (SIMD, quantisation, Metal, WebGPU, batching, speculative decoding) and keep each step runnable |
| [llama2.rs](https://github.com/srush/llama2.rs) (srush) | Rust, CPU SIMD + GPTQ | Educational / fast | 1.1k★, dormant 2023; HN [60](https://news.ycombinator.com/item?id=37013344) | The same language and spirit; no GPU, web or ladder |
| [lm.rs](https://github.com/samuel-vitorino/lm.rs) | Rust, CPU | Minimal | 1k★; HN [310](https://news.ycombinator.com/item?id=41811078) | Shows r/rust/HN appetite for "minimal Rust inference". Top comments: "CPU only?", "what's the value vs llama.cpp?", "benchmarks vs llm.c?" We answer all three |
| [llm.c](https://github.com/karpathy/llm.c) | C/CUDA | Training (plus a reference test harness) | 31k★ | We borrowed its testing approach (test_gpt2.c). Different goal |
| [yalm](https://github.com/andrewkchan/yalm), [calm](https://github.com/zeux/calm) (zeux) | C++/CUDA (calm also Metal) | Fast single-batch, from scratch | 604★ / 646★ | The closest "beat llama.cpp from scratch" precedents, both on NVIDIA. We're the Apple and WebGPU counterpart, with a fair-format head-to-head |
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) | C/C++; CPU, Metal, CUDA, Vulkan, **WebGPU** ([LlamaWeb paper, May 2026](https://www.alphaxiv.org/abs/2605.20706); maintained by Reese Levine per the [wllama README](https://cdn.jsdelivr.net/npm/@wllama/wllama@3.8.1/README.md)) | Production local inference | 131.6k★ | The baseline we measure against. Our f16/q8/q4 GPU decode beats it on GPT-2 124M; **its prefill is ~2× ours** (10.3k vs 5.2k tok/s f16, bench/headtohead.json) and its NEON-only CPU f16 decode is roughly level with ours. Say so up front |
| [MLX](https://github.com/ml-explore/mlx) / [mlx-lm](https://github.com/ml-explore/mlx-lm) | C++/Python/Swift, Metal (now also CUDA) | Apple's array framework | 28.8k★ / 7.4k★; Ollama moved to MLX on Mac ([HN 648](https://news.ycombinator.com/item?id=47582482)) | We beat it on GPT-2 decode at matched formats (f16 364 vs 342; q8 612 vs 534; q4+q8 head 787 vs 694). MLX's default q4 also quantises the tied head (ppl 37 → 311, per our notes). Prefill: MLX is slightly ahead at f16 and q8 |
| [candle](https://github.com/huggingface/candle) | Rust; CPU, CUDA, Metal, WASM demos | HF's minimalist framework | 21.2k★ | Framework, not ladder. No GPT-2 model in candle-transformers (bench notes), so it isn't in the head-to-head. Its WASM Spaces (llama2-c, Phi, Whisper) are prior art for "Rust LLM in the browser" |
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) | Rust on candle | Production multi-model server | 7.8k★; low HN traction | A product. Not comparable |
| [burn](https://github.com/tracel-ai/burn) | Rust; wgpu, CUDA, ndarray | DL framework | 16.1k★ | Someone has done [GPT-2 on burn](https://github.com/muhammad-fiaz/gpt-2-rust); a framework approach, not from scratch |
| [ratchet](https://github.com/huggingface/ratchet) (FL33TW00D) | Rust + WebGPU | Browser-first framework (Whisper, Phi) | 772★; pushed Sep 2026 but no release; [status unclear](https://www.sourcepulse.org/projects/1839515) | The nearest "Rust + WebGPU inference" prior art; a crib in DISTILLED step 17 |
| [Crane](https://github.com/lucasjinreal/Crane) | Rust on candle | Multi-modal engine | 507★ | Product-ish; not educational |
| [tract](https://github.com/sonos/tract) | Rust, CPU | ONNX/TF inference for embedded | 3.1k★ | Different niche (embedded, ONNX) |
| [wonnx](https://github.com/webonnx/wonnx) | Rust, WebGPU | ONNX on WebGPU | 1.8k★; dormant since Jul 2024 | Prior art, now inactive |
| [web-llm / MLC](https://github.com/mlc-ai/web-llm) | TVM-compiled WebGPU | Production in-browser LLMs | 19.3k★; HN [276](https://news.ycombinator.com/item?id=35583349), [145](https://news.ycombinator.com/item?id=49536411) (Sep 2026) | The "real models in the browser" reference ([~80% of native, per their paper](https://arxiv.org/abs/2412.15803)). Compiled, not hand-written, and doesn't teach |
| [transformers.js](https://github.com/huggingface/transformers.js) / ONNX Runtime Web | JS + ORT (WASM/WebGPU) | Production web ML | 16.4k★ | What Transformer Explainer-type sites use. A black box |
| [WebGPT](https://github.com/0hq/WebGPT) (3.8k★), [wgt](https://github.com/loristns/wgt) | JS/WGSL | GPT-2 on WebGPU in ~1.5k lines | dormant | Direct prior art for "GPT-2 on WebGPU by hand", with no ladder or measurement story |
| [tiny-llm](https://github.com/skyzh/tiny-llm) | Python on MLX | Course | 4.8k★ | Course competitor (see §1) |
| [uzu (Mirai)](https://github.com/trymirai/uzu) | Rust, Metal | Product engine "faster than llama.cpp" | 2.4k★; [Show HN 186](https://news.ycombinator.com/item?id=44570048) | Shows how "faster than llama.cpp on Apple Silicon" lands: "why Rust not C++?", "somewhat faster on small models, requires new format, not sure what the goal is", "what's your adoption roadmap?" |
| 2026 newcomers: [pegainfer](https://pegainfer.org/) (Rust+CUDA, paged KV, OpenAI API), [Atlas](https://atlasinference.io) (Rust+CUDA; [HN 5](https://news.ycombinator.com/item?id=48116631)), [Kortex](https://github.com/Vage91/Kortex) (out-of-core Rust; [HN 4](https://news.ycombinator.com/item?id=48789790)), [xinfer](https://github.com/guoqingbao/xinfer) (337★), [krishh-p Qwen3 Rust+CUDA](https://github.com/krishh-p/llm-inference-engine), [qwen3-rs](https://news.ycombinator.com/item?id=44544088) (HN 15), [dlgo](https://github.com/computerex/dlgo) (Go; [HN 2](https://news.ycombinator.com/item?id=47283136)), [LARGE crate](https://docs.rs/crate/large/0.2.0), [Anirudh Sathiya's C++ engine](https://www.anirudhsathiya.com/blog/transformer) ([HN 9](https://news.ycombinator.com/item?id=47689443)) | Mostly Rust/CUDA | Mix of portfolio and product | low traction | **Lesson: "I wrote an inference engine from scratch in Rust" is now a common portfolio genre that gets single-digit HN points.** The engine alone won't carry a launch. The live, measured, explorable ladder might |
| Others named `nanoinfer` on GitHub: [lecuong1502/NanoInfer](https://github.com/lecuong1502/NanoInfer) (CUDA), [Vansh4195](https://github.com/Vansh4195/nanoinfer) (NumPy GPT-2), [vijaykrishna483-cms](https://github.com/vijaykrishna483-cms/nanoinfer) (NumPy GPT-2 + continuous batching), [SpookyJumpyBeans](https://github.com/SpookyJumpyBeans/nanoinfer) (NumPy + Rust INT8 + a "live browser view"), [lhy-loveworld](https://github.com/lhy-loveworld/nanoinfer) ("education-oriented… inspired by nanoGPT"), and five more | — | — | 0–5★ each | **Name collision** (see §4) |

**Where nanoinfer fits.** It's the only engine that is all of these at once:

1. written from scratch with no ML framework;
2. multi-backend (NEON CPU, Metal, WebGPU/WASM);
3. kept runnable *at every optimisation stage* and switchable live in a browser;
4. benchmarked against MLX and llama.cpp at matched formats, with ppl and quality checks;
5. carrying a serving layer (continuous batching, speculative decoding) on the same codebase.

Taken one at a time, each property has precedent; nothing else has all five. It isn't a production engine and shouldn't pretend to be. GPT-2-family only is fine for teaching, but it caps the r/LocalLLaMA audience.

---

## 3. Honest usefulness assessment

**Is there a real gap?** Yes, but only the specific one: *a real engine you can execute at every rung of an optimisation
ladder, with the reader's own hardware producing the numbers next to a computed ceiling.* Today:

- the simulators (Modular, LLM Inference Explained, kernelspace) teach the right ideas with modelled numbers;
- the real-model browser demos (Transformer Explainer, WebLLM, MicroLLM Lab) run one opaque, optimised engine;
- the measured ladders (Chan's yalm, tiny-llm's benchmarks) are static or need a local setup.

The "aha" we can deliver and they can't is *watching your own browser go from 0.4 to hundreds of tok/s as you flip
stages, and seeing why it stops where the line says it will*. That's the hook. "Inference for systems engineers" is the
audience, not the differentiator: tiny-llm and kernelspace both claim it, and kernelspace got 2 HN points with it.

**Who would use it?**

1. **Backend and infra engineers moving into inference.** This is the biggest group. "Inference engineer" became a named
   job in 2026 ([Pragmatic Engineer deep-dive](https://newsletter.pragmaticengineer.com/p/what-is-inference-engineering);
   Baseten's book). Part 0's Rosetta stone and constraint map are aimed squarely at them.
2. **Job seekers and interview preppers.** Bytes-per-token envelope maths, KV-cache sizing and memory- vs compute-bound
   come up a lot in inference interviews (my judgement, not sourced). A calculator plus "check yourself" fits.
3. **Educators.** Only if the figures can be embedded and linked into a lecture. Today they can't.
4. **Rust / WebGPU / perf people.** These come for the engine, the kernels and the head-to-head, not the course.
5. **Not**: ML researchers (they'll use the scaling book), production operators (Modular, Baseten), or local-LLM users
   who want big models (llama.cpp, MLX, Ollama).

**Portfolio piece vs genuinely valuable.** As it stands, it's an excellent portfolio piece and a strong one-shot post. To
become something people come back to and cite:

- **Exercises.** For each of the 8–10 core rungs, a stub in the repo plus `cargo test` against the HF reference logits,
  then `bench` puts *your* number on the climb chart. Ideally there's also a browser path: paste or compile your kernel to
  WASM and see it measured live. This is the tiny-llm / kernelspace / CS336 lesson. Nobody else can offer "your kernel,
  measured live, next to the speed of light".
- **Crowd-measured data.** Opt-in, anonymous per-stage tok/s per browser and GPU. The LlamaWeb paper measured 16 devices
  from 8 vendors; we could show the distribution live on every chart. It turns a demo into a dataset.
- **Stable, citable, embeddable pages.** Every figure as an iframe or permalink, a CC BY licence for the prose, `.md` /
  `llms.txt` exports (the Modular pattern), and field-guide entries at stable URLs (the Modal GPU Glossary pattern).
- **Maintenance signal.** A dated changelog; numbers regenerated from `bench/*.json` (already the case: keep it).

**Where better tools exist: link out, don't compete.**

| Our section | Link out to instead of deepening |
|---|---|
| ch.1–6 (mechanics) | Bycroft's 3D viz, Transformer Explainer, 3Blue1Brown ch.5–7, Alammar. Keep ours short and GPT-2-real, and say "for a deeper look at attention, watch 3B1B" |
| ch.15 (quantisation) | MIT 6.5940 lectures, Weng's post, the llama.cpp quant docs |
| ch.17 (batching / paged attention) | "Inside vLLM", nano-vllm, the HF continuous-batching post, the Anyscale 23× post |
| ch.19 / multi-GPU | *How to Scale Your Model* (inference and sharding chapters), the Ultra-Scale Playbook, Stas Bekman |
| Production (autoscaling, routing, disaggregation, SLOs) | Baseten *Inference Engineering*, the Modular handbook, the Modal almanac. Don't build these |
| GPU internals (warps, coalescing) | Modal GPU Glossary, the Modular handbook kernel section, [GPU Puzzles](https://github.com/srush/GPU-Puzzles) |

---

## 4. Positioning and distribution

### Naming

- `nanoinfer` is generic and taken many times over: 10+ GitHub repos in 2025–26, at least five of them "from-scratch
  GPT-2 inference", one with a "live browser view". `nanoinfer.live` is unregistered (NXDOMAIN today) and `nanoinfer.com`
  is held by someone else. HN commenters also mock the "nano-" prefix ("nano and large cancel out", in both nano-vllm
  threads).
- **Options:** keep `nanoinfer` for the crate and repo, but give the site and course its own name built on the strongest
  idea, the speed-of-light line. Candidates:
  - *Speed of Light*: "how fast can one computer write the next word?"
  - *Bandwidth Bound*
  - *The Climb*
  - *Every Weight, Every Token*

  Check domains first. If you keep `nanoinfer`, buy the domain before launch and add "Rust · Metal · WebGPU" to every
  title to tell it apart from the others.

### One-line pitch (drafts)

- "A from-scratch Rust GPT-2 engine you can run in your browser at every stage of optimisation, from 0.4 to 800 tokens a
  second, with the speed-of-light limit drawn on every chart."
- Show HN title: **"Show HN: Learn LLM inference by running a from-scratch engine at every optimisation stage in your
  browser"**. An alternative that leads with the write-up: "From 0.37 to 817 tok/s: optimising a GPT-2 engine on an M4,
  live in your browser".
- Avoid "faster than MLX and llama.cpp" in the title. On HN that framing reliably pulls the thread toward benchmark
  disputes (see uzu; the "6.4× faster than llama.cpp" post got "headline is missing a critical detail: on Apple
  Silicon"). Put the head-to-head in the body, with formats matched and our losses (prefill) stated.

### Where to share, in order

1. **The write-up post first** (Chan-style: one long, honest, measured ladder with the live engine embedded) as a regular
   HN submission. Precedent: Chan got [344](https://news.ycombinator.com/item?id=42417857) with no interactivity, Bycroft
   [1,592](https://news.ycombinator.com/item?id=38505211), tiny-llm [297](https://news.ycombinator.com/item?id=43820022),
   the Modular handbook [366](https://news.ycombinator.com/item?id=44527947), Transformer Explainer
   [105](https://news.ycombinator.com/item?id=41221718). Bare "Rust engine from scratch" posts got 2–15 in 2026.
2. **Show HN for the course/site** a week or two later, if the post did well, or as the primary if the post is folded
   into the site.
3. **r/rust**: the engine angle (NEON intrinsics, wgpu, WASM threads with COOP/COEP, the ladder). **r/LocalLLaMA**: the
   head-to-head and the bandwidth calculator. They'll ask "does it run Llama/Qwen?", so have an answer.
   **[unverified]** I couldn't retrieve Reddit reception data for comparable posts.
4. **X**: a 20-second screen capture of the stage switcher with the tok/s counter climbing toward the line. Tag the
   people the work builds on (Karpathy for llama2.c and llm.c, Andrew Chan, MLX and ggml maintainers) only where it's
   honestly derived.
5. **lobste.rs** (invite-only; tags rust, performance, ml). **[unverified]** The site didn't respond to search; my
   understanding is that it is hostile to AI-generated content.
6. Niche: GPU MODE Discord, This Week in Rust, the WebGPU community, and the Hugging Face / MLX communities for the
   head-to-head. Educators: email the CS336 / 6.5940 / CMU DLSys staff once embeddable figures exist.

### What reviewers will criticise (and pre-empt)

1. **"GPT-2 124M is a toy; small models are overhead-dominated and not representative."** DISTILLED law 6 already says
   this. Make it a feature: show where 124M's lessons transfer (bandwidth ÷ bytes) and where they don't (dispatch
   overhead). Speculative decoding already uses GPT-2 large; add one 7B-class *calculated* line next to each measured one.
2. **"Beats MLX/llama.cpp" scepticism.**
   - Lead with matched formats, the q4-head ppl note, and median-of-7 over 3 rounds.
   - State the losses: llama.cpp prefill ~2×; CPU f16 decode roughly level.
   - Link `bench/headtohead.json` and the scripts.
   - Expect "llama.cpp isn't tuned for GPT-2" and "try a 7B".
3. **"2,200× faster" is against a deliberately naive baseline.** Say so in the same sentence: "the first version re-read
   the whole text per token". Prefer "% of the speed-of-light" as the headline metric; the speed-up multiple invites
   eye-rolls.
4. **"Feels AI-written."** It happened to the Nano-vLLM walkthrough (since retracted), MicroLLM Lab ("mostly useless,
   AI-generated information") and kernelspace (author admits it). Prose needs a human voice, specific numbers and first
   person "I tried X, it was slower". The select-to-ask AI tutor will draw "slop" comments from some people. Make it
   optional and unobtrusive, and say what model it uses and who pays.
5. **Density and UI.** The MicroLLM Lab thread was mostly about too much text before the interaction. Put the live demo
   above the fold on the landing page; keep the "one idea, one figure" rule.
6. **Browser compatibility.**
   - WebGPU isn't on stable Firefox ("GPUShaderStage is not defined" was a top MicroLLM Lab comment).
   - COOP/COEP isolation breaks some embeds.
   - The ~130 MB download.

   Degrade gracefully to recorded runs, label it, and test Firefox and Safari 26 before posting.
7. **"Why Rust, not C/C++/Zig?"** This appears in every Rust-engine thread (uzu, lm.rs). Have one paragraph ready:
   `std::arch` NEON, wgpu → WebGPU and Metal from one codebase, wasm32 builds.
8. **"How does this compare to Chan's post / llama2.c / Bycroft?"** Name them generously on the about page.
9. **The stale GitHub README** shows only stages 0–1 and doesn't link the site. Fix it before any launch, since HN readers
   click the repo first.
10. **"Multi-GPU is a teaser."** Either finish ch.19 with the rented-GPU runs (DISTILLED R1/R2) or label it "next post"
    clearly and link the scaling book.

---

## 5. Ideas worth borrowing, ranked

1. **Graded exercises on the real engine** (tiny-llm, kernelspace labs, CS336, llm.c's `test_gpt2.c`). Ship
   `exercises/stage-N` stubs: the existing HF-logit tests are the grader, and `bench` drops your number on the climb.
   Highest leverage for turning a portfolio piece into a resource.
2. **Your browser on the chart, and everyone else's** (LlamaWeb's 16-device sweep, Modal's almanac). An opt-in,
   anonymous submission of per-stage tok/s; show the distribution of GPUs and browsers on the ladder and the head-to-head.
3. **Concept / Maths / Code depth toggle per chapter** ([LLM Inference Explained](https://llm-inference-explained.vercel.app/)).
   It maps onto our audience split (systems engineer vs ML-curious), and "Code" can deep-link to the exact Rust for that
   stage.
4. **A link to the stage's diff** (llama2.c / nanoGPT readability; Chan's before/after code). Tag each stage in git; every
   Part 3 chapter gets "see the 40 lines that changed".
5. **Numeric worked problems with hidden answers** (the scaling book's pop quizzes). Extend "check yourself" with
   estimation problems ("Llama-3-8B q4 on an M4 Pro: estimate decode tok/s") answered by the calculator and, where
   possible, a measured number.
6. **Embeddable figures, CC BY prose, `.md`/`llms.txt`** (Modular handbook). This makes educators and LLM-assisted readers
   first-class and helps SEO.
7. **Profiler evidence and kept failures** (Chan's nsys/ncu shots; our `webgpu-flashdec` "tried, slower" tag). Add a
   per-op timeline from the Trace hook to each Part 3 chapter and screenshots of the Metal capture. HN rewards visible
   negative results.
8. **One continuous spatial model** (Bycroft's 3D scene with guided camera). We probably shouldn't build 3D, but a
   persistent minimap of "where in the model this byte is being read", shared across chapters, would give the course
   Bycroft's sense of place cheaply.
9. **Practical calculators** (Modular's KV-cache / GPU-memory calculators). Our bytes-per-token calculator is close.
   Add KV-cache-at-context and "will it fit" for named models and machines, with our measured points overlaid.
10. **A short video walkthrough** (3Blue1Brown, Karpathy). A 10–15 minute screencast of the climb. Video drives a lot of
    the reach in this genre (Karpathy's "Let's build GPT" got 1,110 HN points as a YouTube link).
11. **Academic write-up** (Polo Club's AAAI paper, 125k users). A short VIS4AI / EduAI workshop paper on "measured
    explorable explanations of inference" would bring credibility and classroom adoption. Optional.
12. **Glossary as standalone pages** (Modal GPU Glossary). Give field-guide terms their own URLs, not only anchors, for
    search traffic.

---

## Unverified or not found

- Bycroft and Transformer Explainer feature details are from memory: both are JS-only pages the fetcher couldn't render.
  HN and GitHub stats are verified.
- InferenceVisually's repo returned 404 today; its description comes from search snippets only.
- Details of kernelspace and llm.manogya.dev come from their HN posts and titles (JS-only sites).
- The Modular handbook's HN score (366) is verified via Algolia. Its CC BY licence and its section count come from an
  aggregator.
- *Inference Engineering*'s page count and print dates come from third-party listings; the ToC is from baseten.co.
- I couldn't retrieve the CS336 inference lecture content directly (only a third-party summary).
- I found no Reddit (r/rust, r/LocalLLaMA) or lobste.rs reception data. Lobsters search didn't respond.
- I found no canonical Fireworks inference explainer.
- The LlamaWeb (llama.cpp WebGPU) production readiness is unconfirmed beyond the paper and the wllama README.
