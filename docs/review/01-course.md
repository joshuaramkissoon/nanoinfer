# Course review 01: top-down, and what would make it click

Reviewer stance: Josh as reader (distributed-systems engineer, no ML), plus a second pass as a sceptical inference engineer
at a frontier lab. Walked every route in one Chrome session at 1280 px (dark) with spot checks at 390 px: `/`,
`/learn/*` (0.1–0.4, 01–19), `/learn/field-guide`, `/lab`. Read `web/IA.md`, `web/DESIGN.md`, `web/REVIEW.md`,
`docs/WRITEUP.md`, `research/DISTILLED.md`, `research/03-explorable-writeup-and-wasm.md`. No code changed.

The short version: **the course is very good at one thing: why single-user decode is memory-bound, told through a real
engine climbing towards a computable ceiling.** That spine is right and rare. What it doesn't do yet is get the reader
to "can follow an inference-infra conversation", because the conversation at a lab in 2026 is mostly about what
happens *after* chapter 16: many users, long contexts, MoE, prefix caches, disaggregated prefill/decode, FP8/FP4, and
the throughput-vs-interactivity frontier. Those live in one chapter (17), a teaser (19) and the field guide. And the
interactivity is mostly "toggle and watch". Very little asks the reader to *predict*, *decide* or *build*.

---

## 1. High level

### 1.1 Promise, scope, arc

**Promise (landing):** "What happens when a model writes the next word, and how fast can one computer do it?" This
promises an engine-building story on one computer. **Part 0's promise** is bigger: "inference for systems engineers",
"the field's jargon mapped onto distributed systems". The two promises pull apart. Parts 1–3 deliver the first well.
Nobody delivers the second past a vocabulary table (0.4) and a glossary.

**The arc as it stands:**

```
0  orientation (4)    → what / why expensive / hardware / jargon
1  the model (8)      → next token … the loop
2  why slow (3)       → roofline, speed limit, starting line
3  the climb (8)      → KV cache, ILP, threads, quant, GPU, batching, spec decode, many GPUs (teaser)
   lab                → the old demo page
```

What works:
- **One computable law carries the course.** Bandwidth ÷ bytes per token appears in 0.2, is derived in 09/10, and every
  climb bar is measured against it. That's Boehm's and Chan's structure done better, with the ladder always visible.
- **Systems analogies land.** The I/O-bound service (0.2), the synchronous RPC per GPU op (16), virtual memory for paged
  KV (17), Amdahl for all-reduce (0.3) are exactly right for this reader.
- **Provenance and honesty.** Measured / estimate / external chips, a kept null result (13's NEON fold), greedy-loop
  caveats (18). An expert will trust it.

Where the arc is wrong:

1. **Part 0 front-loads the whole course, so Part 2 repeats it.** 0.2 teaches bandwidth ÷ bytes (fig 0.2.1), the
   ridge/idle-maths idea (fig 0.2.2), the KV cache, batching economics (fig 0.2.3) and a constraint map (fig 0.2.4):
   five big ideas, 6,300 px, four figures. Then chapter 09 re-teaches the ridge and chapter 10 re-runs the same
   calculator ("If you read chapter 0.2, you've seen this division"). The reader meets every core idea twice: first
   before they know what a weight does, then after. Either Part 0 is a thin "map and motivation" (one figure per
   chapter, no derivations), or Part 2 merges into it. Right now it's both.
2. **The course ends where the interesting systems problems start.** Of 23 chapters, *one* (17) is about serving many
   users, and multi-device is a teaser (19). For a distributed-systems engineer this is backwards: scheduling, state
   placement, memory admission control, routing and sharding are where his existing intuitions pay most, and where
   the jargon density is highest (TTFT, TPOT, goodput, chunked prefill, prefix cache hit rate, disagg, EP, TP, DP
   attention, KV offload...).
3. **Single-user and multi-user are interleaved.** 12–16 are one user, one device. 17 is many users. 18 is back to one
   user. 19 is many devices. Spec decoding *belongs* after batching, but as "the one-user lever that fights batching"
   (it stops paying once the batch makes you compute-bound), and the chapter doesn't say that.
4. **Rosetta stone (0.4) before Part 1.** Most rows ("paged attention", "prefix caching", "disaggregated serving")
   are analogies for things the reader has no referent for yet. REVIEW.md already noted this. It's a reference, not a
   chapter.
5. **No capstone.** Chapter 19 ends with "the end of the beginning → the lab". Nothing pulls it together into the
   thing the reader actually wants: "given a model, hardware and SLOs, reason about how to serve it", or "here's a
   real vLLM/SGLang launch command and an infra Slack thread; you can now read every word".

### 1.2 Does it get someone from zero to "can follow an inference-infra conversation"?

Partly. After the course the reader can:
- estimate a decode ceiling on the back of an envelope (strong);
- explain prefill vs decode, the KV cache, quantisation, batching, continuous batching, spec decoding (good);
- say what TP/PP/EP are (shallow).

He *cannot* yet follow these (typical infra-channel sentences):
- "Prefix cache hit rate dropped after the system-prompt change, so TTFT p99 regressed." (prefix caching isn't on the
  path.)
- "We disaggregated prefill onto separate H100s and decode onto H200s; decode is KV-capacity bound, so we went
  wide-EP with DP attention." (no disagg, no capacity-bound-decode framing, no DP attention, EP is one bullet.)
- "FP8 on Hopper doubled prefill throughput but did nothing for batch-1 decode." (quantisation is framed only as
  fewer bytes; tensor-core FP8/FP4 compute speed-ups are absent.)
- "We're on the Pareto frontier at 60 tok/s/user, ~3k tok/s/GPU." (the throughput–interactivity frontier is *the*
  chart the industry argues with; the course has the data in `bench/batching.json` but draws two separate lines.)
- "MLA cut KV per token by ~10× so we could run 128k context at batch 64." (no KV-size formula with heads, so
  GQA/MLA can't be reasoned about.)
- "CUDA graphs fixed our launch overhead." (chapter 16 teaches the idea under its Metal name; the industry name is
  never given.)

### 1.3 What a frontier inference engineer would say is missing or wrong

Missing (roughly in order of how often it comes up at work):
1. **MoE as the default, not an aside.** Most frontier and open-weight models in 2025–26 are MoE (DeepSeek-V3/R1,
   Qwen3, Kimi K2, gpt-oss, Llama 4). "Active vs total params", "every expert must be resident", "a batch touches most
   experts so decode reads them all", and expert-parallel all-to-all are core vocabulary. The field guide's MoE lab
   (`src/learn/guide/MoeLab.tsx`) is good and should be on the path.
2. **The KV cache is the serving bottleneck, and it's a memory-management problem.** Per-token KV size formula
   (layers × kv_heads × head_dim × 2 × bytes), GQA/MQA/MLA as the architecture lever, KV quantisation, prefix
   caching/radix trees, offload to CPU/SSD, eviction. The course has paged attention in a fold (fig 17.4) and nothing
   else.
3. **Decode attention doesn't batch.** Weight reads are shared across the batch; KV reads are per sequence. So at long
   context the "batching is free" story breaks: attention bytes grow with batch × context. DISTILLED.md law 8 says
   this; the course never does in prose (fig 0.2.3 encodes it numerically without saying it).
4. **Per-token cost with a KV cache is not flat.** Chapter 12 says "the cost per token stopped growing with the text:
   flat, not rising" (`ch/KvCache.tsx:55`). The cache makes the *compute* per token O(1) in context for the matmuls,
   but attention still reads O(n) KV per token. WRITEUP's own metal-kv8 note ("context makes decode slower") says so.
   An expert will flag this sentence.
5. **Prefill/decode interference, chunked prefill and disaggregation.** The serving data already has
   `rate=16,budget=32` vs `budget=512` runs (`bench/serving.json`): the TTFT vs inter-token jitter trade-off is
   measured and unused. Disaggregation (DistServe, Dynamo, llm-d) is a field-guide card only.
6. **Metrics and SLOs as the objective function.** TTFT, TPOT/ITL, p50 vs p99, goodput, tokens/s/user vs
   tokens/s/GPU. Mentioned in 0.2 and a fold in 08. Not the frame for 17.
7. **Tensor cores and low-precision compute.** Fig 0.3.1 credits GPU FLOPs to "lanes" (SIMD). The 989 TFLOP/s number
   on the same figure is tensor-core (matrix unit) throughput; the CUDA-core FP32 number is ~15× lower. FP8 doubles
   it, FP4 (Blackwell) doubles again. That's why chapter 17's "once the job is compute-bound, smaller numbers stop
   helping" is true for *weight-only* int4 on the M4 and false for W8A8/FP8 on a datacentre GPU. Say so.
8. **FlashAttention / attention's own cost.** Prefill attention is O(n²) compute; at 32k–1M context it dominates. Not
   on the path.
9. **Determinism.** Check-yourself 0.1 q3: "Is a forward pass deterministic? Yes: randomness only enters at
   sampling." In production it isn't: batch-size-dependent kernel choices plus non-associative float adds (which
   chapter 13 itself teaches!) make temperature-0 output vary with load. Great tie-in, currently a wrong answer.
10. **Modern spec decoding** is mostly draft *heads* (EAGLE-3, MTP in DeepSeek-V3) rather than a separate small
    model, and its benefit shrinks as batch grows. One paragraph.
11. **The ops layer**: replicas and routing (KV-aware routing), autoscaling and cold start (loading 100s of GB of
    weights), multi-LoRA, structured output. Field-guide level is fine; a pointer is not.

Wrong or inconsistent:
- `/learn/smaller-numbers`: "with 4-bit weights the limit is 8× higher, around 1,200" (`ch/Quantization.tsx:130`).
  8 × 243 ≈ 1,950; fig 15.1 says "7.1× vs 32-bit"; fig 10.1 says "about 7×". Pick one derivation (likely
  98 GB/s CPU ÷ ~80 MB with the 8-bit head ≈ 1,200) and show it.
- `/learn/the-gpu`: "reading 89 MB of weights at close to the 105 GB/s". 89 MB × 817 tok/s ≈ 73 GB/s, about 69%.
  Honest number is better than "close".
- `/learn/faster-arithmetic`: prose 5.8×, climb 5.3× (still open from REVIEW.md).
- **Three different ridges with no reconciliation.** 1.6 FLOP/B (fig 9.1, the browser tab), 41 (fig 0.3.1, M4 GPU),
  295 (fig 0.2.2, H100), plus 0.5 vs 1 FLOP/B for decode. Each is right in context. Together they read as noise.
  One small "ridges by machine" strip (tab / M4 CPU / M4 GPU / H100 / B200) in chapter 09 fixes it.
- **Fig 0.2.2 contradicts fig 0.2.3.** 0.2.2 says each user's speed is "unchanged" up to 256 users on an H100. 0.2.3
  says at 16 users each gets 150 tok/s vs 240 alone. The difference is KV reads (0.2.3 includes them, 0.2.2 doesn't),
  and the page never says so.
- **The climb's "limit, 32-bit: 243" line** stays put while bars for 4-bit (497), GPU (817) and batching (6,453) pass
  it. Chapter 15 explains this in prose; the chart says the opposite of the course's thesis (that you can't beat the
  ceiling). Draw each bar's *own* ceiling as a tick, and label bars "% of own ceiling".
- **The landing hero says 97 tok/s** (a recorded V8 single-core run), directly above "2,200× faster" (817 / 0.37). The
  first number the reader sees is the slow engine. Show both, or replay the fast rung.

### 1.4 Where the mental model breaks for this reader

- **Shapes.** The reader is never given the model's shape sheet: d_model 768, 12 heads × 64 head_dim, 12 layers, FFN
  3072, vocab 50,257. Without heads, GQA (fig 0.2.3's toggle), TP-by-head, KV-size arithmetic and MLA can't be reasoned
  about, only accepted. Chapter 05 mentions "144 of them" in one sentence.
- **Matrix × vector vs matrix × matrix.** Chapter 04 teaches only matvec. Prefill, batching and spec-decode verify are
  all "the same weights times *many* vectors", which is where reuse (arithmetic intensity) comes from. That picture is
  the bridge between Part 1 and Parts 2–3 and it's never drawn. The nearest thing is the 0.2.2 square grid.
- **Where time goes in one token.** No figure in the course path shows a token's time broken down by op (the lab has
  it). So "the LM head is 31% of bytes" (DISTILLED law 4), "small ops and dispatch dominate at 1,000 tok/s" (law 6)
  and chapter 16's waiting story float without a picture.

### 1.5 Proposed revised arc

Keep every existing chapter's best figure; re-cut the parts; add one serving part and one scale part; add a capstone.

```
/                      landing: hero shows slow → fast rung side by side; "predict the speed" first interaction
/learn                 contents + persistent "mental model" diagram (see idea #3)

PART 0  The shape of the problem        (thin: one figure each, no derivations)
  0.1  What inference is                 (as is; fix determinism q)
  0.2  Every word reads every weight     (fig 0.2.1 only + prefill/decode definitions)
  0.3  The hardware in one picture       (as is; add tensor cores; move fig 0.3.3 to Part 5)

PART 1  How a model writes text          (01–08, tightened)
  01   Guess the next word
  02   Tokens
  03   Embeddings                        (simplify fig 3.1)
  04   Multiply and add                  (+ "one vector or many": matvec vs matmul reuse)
  05   Attention                         (+ heads; GPT-2's shape sheet card)
  06   Layers and the residual stream
  07   Sampling                          (+ one line on structured output / logit masks)
  08   The generation loop

PART 2  Why it's slow                    (roofline becomes the primary figure)
  09   Memory or maths                   (live roofline you drag; ridges-by-machine strip)
  10   The speed limit                   (calculator; absorbs 0.2's derivation)
  11   The starting line
  11½  Where a token's time goes         (new or promoted from lab: per-op bar, LM head 31%)

PART 3  One user, one chip: the climb    (12–16, as now)
  12   KV cache (fix "flat")  13 ILP/SIMD  14 cores  15 quantisation  16 GPU (+ CUDA graphs name)

PART 4  Many users: serving              (new part; 17 split)
  17   Batching and the frontier         (fig 17.1 redrawn as tok/s/user vs tok/s/GPU frontier)
  18   Scheduling: continuous batching, chunked prefill, TTFT/TPOT/goodput (fig 17.3 + serving.json budgets)
  19   The KV cache as a memory system   (paged KV promoted, prefix caching, GQA/MLA, KV quant, eviction)
  20   Guess ahead                       (spec decoding, + why it fades at high batch, + draft heads)

PART 5  Bigger than one chip             (19 expanded)
  21   Fit it: the memory budget         (weights + KV + activations game; capacity vs bandwidth)
  22   Split it: TP, PP, and their talk  (fig 0.3.3 + ring all-reduce + shard-it-yourself game)
  23   Mixture of experts                (MoE lab promoted; EP all-to-all; hot experts)
  24   Split the phases: disaggregation  (prefill vs decode on different pools; KV transfer cost)

CAPSTONE
  25   Serve it                          (a scenario game: model + hardware + SLO → config; then a real infra
                                          thread / vLLM command annotated, every term linked)

REFERENCE  field guide (Rosetta stone becomes its first tab) · lab · review deck (spaced repetition)
```

That's 25 chapters plus reference instead of 23, but Part 0 gets lighter and the reader's last third is where his
existing expertise compounds. If that's too much, the minimum is: split 17 into two, promote MoE and paged/prefix KV
into chapters, add the capstone, and move Rosetta to reference.

---

## 2. Mid level: per part and per chapter

Format: **goal** (is the one idea clear) · **figure** (does the primary figure carry it) · **pacing / cut / merge** ·
**prereqs** · **check** (quality of check-yourself).

General notes on check-yourself (`src/learn/checks.ts`, 47 questions): the length tell is fixed, but almost all are
*recognition* questions, and several stems give the answer away (09 q1: "the chip can do far more per byte. So the
chip is…"). None ask for a number. None ask the reader to transfer to a new model or machine. Nothing is stored, so
there's no review. See ideas #6 and #14.

### Part 0

**0.1 What inference is** (`/learn/what-is-inference`)
- Goal: clear. "Read-only blob + fixed program + stateful server" is the best framing on the site.
- Figures: 0.1.1 is fine but decorative; 0.1.2 (clickable weight strip) is good and real; 0.1.3 (four jobs) is the
  chapter's real payload and works. The pseudo-code block is excellent for this reader; promote it, don't fold it.
- Check: q3 (determinism) is wrong for production (see 1.3 #9). Turn it into a "surprisingly, no" moment linked to ch 13.

**0.2 Every word reads every weight** (`/learn/why-its-expensive`)
- Goal: muddy, because there are five. The title idea is carried by fig 0.2.1 alone.
- Figures: 0.2.1 good (sensible 7B default now). 0.2.2 good, but conflicts with 0.2.3 (KV reads). 0.2.3 is the best
  economics figure on the site, but is in the wrong chapter (needs batching + KV first). 0.2.4 (constraint map) is
  homework at this point even folded; it's duplicated by the field guide.
- Cut/split: keep 0.2.1 + prefill/decode definition. Move the KV cache section to chapter 05/12, 0.2.3 to the new
  batching chapter, 0.2.4 to the field guide (and to the capstone as a recap).
- Check: fine.

**0.3 The hardware in one picture** (`/learn/the-hardware`)
- Goal: clear (bandwidth, capacity, link speed).
- Figures: 0.3.1 credits lanes, not tensor cores. 0.3.2 good. 0.3.3 is a Part-5 figure; here it front-runs TP before
  the reader knows what a layer is. "When does a CPU win?" section is good.
- Add: one line that datacentre FLOPs come from matrix units that also get 2× per halving of precision (FP8, FP4).

**0.4 A Rosetta stone** (`/learn/rosetta-stone`)
- Goal: clear but premature. Move to the field guide as its first tab ("translate"), and link rows from the chapters
  where each term lands. The holds/breaks pairs are good; keep them.

### Part 1

**01 Guess the next word**: strong opener. Fig 1.1 (step through real top-10) is right. Fine as is. "See the whole
list of 50,257" fold is a nice touch.

**02 Tokens**: clear; live tokenizer is great. Four stat tiles (DESIGN says ≤ 3). Add one line on why tokens matter for
cost: prices and context limits are in tokens, and other languages pay more tokens per word (try Japanese).

**03 Embeddings**: the densest figure in Part 1 (scatter + 53-word picker + barcode + neighbours) and the least
necessary chapter for inference. Clusters overlap at the top (animals/food/dog), "days" sits alone at the bottom, most
of the plot is empty. Cut to: neighbours list + barcode, and click points in a better-spread scatter; or merge into 04
as "a token becomes a vector". The "How big is this table" fold (LM head = same table, read every token) is the
inference-relevant bit; promote it.

**04 Multiply and add**: clear. Fig 4.1 good after fixes. Missing half: "now multiply the same matrix by 14 vectors at
once" (prefill) — one toggle on the same figure that shows each weight being reused across columns. That one toggle
sets up 08, 09, 17 and 18.

**05 Attention**: toy figure is good. Needs heads (a 12-head strip; head_dim 64) and the KV bridge already added.
"Does attention cost much?" fold should say: small at 1k context, dominant at 100k (prefill O(n²) compute, decode O(n)
bytes per token).

**06 Layers and the residual stream**: fig 6.2 (logit lens) is a crowd-pleaser but not inference-critical. The default
state ("if the model stopped before any layer it would say: group 100%") reads oddly before you press *send it down*;
start at layer 0 with an explicit empty state ("press send it down"). Add the GPT-2 shape sheet here.

**07 Sampling**: clean. Could merge with 01 (same real logits) to save a chapter; or add constrained decoding as one
line (a JSON grammar just masks the bars), which is jargon he'll hear.

**08 The generation loop**: good; the "same arithmetic per token, very different bytes per token" closing line is the
hinge of the course. Four stat tiles still. The loop figure should show the KV cache growing (ties to 12).

### Part 2

**09 Memory or maths**: the most important chapter for this reader. Fig 9.1 (truck/factory) is a good toy, but the
crossover is the browser tab's 1.6 FLOP/B (`/learn/memory-or-maths`), different from the 41 and 295 used elsewhere.
The roofline, the chart engineers actually draw, is folded. Make the roofline the primary figure (idea #2), keep the
truck as the intro.

**10 The speed limit**: now explicitly repeats 0.2.1. If Part 0 thins out (1.5), this becomes the canonical home of
the calculator, and should add context length (KV bytes) and batch, which is where it starts disagreeing with naive
intuition.

**11 The starting line**: works. Fig 11.1 is fine. The "why start so slow on purpose" fold is good writing.

### Part 3

**12 The KV cache**: fig 12.1 now draws the cache and is good. Fix "flat, not rising" (see 1.3 #4); add a context
slider showing per-token bytes = weights + n × 73.7 KB, so the reader sees when the cache overtakes Q4 weights (~1k
tokens, per DISTILLED law 5). Check: good.

**13 Faster arithmetic**: clear, reads fast for this persona (he knows ILP). Fix 5.8 vs 5.3. Fig 13.1 good.

**14 Many cores**: good chart, one focal bar. Check q1 says "~1.5×", prose says 1.7× bandwidth / 1.46× achieved;
fine but align.

**15 Smaller numbers**: good figure (real block of 32). Fix "8×… 1,200". Missing: weight-only vs weight+activation
(W4A16 vs W8A8/FP8), and the MLX "squeezed the head to 4 bits, perplexity 37 → 311" trap from WRITEUP's head-to-head,
which is a perfect real-world consequence of the "one matrix you can't squeeze" fold.

**16 The GPU**: excellent figure (16.1: wait-per-op timeline). Name the industry equivalents (kernel launch overhead,
CUDA graphs, persistent kernels). Fix the "close to 105 GB/s" claim.

**17 Batching**: three good figures, too much for one chapter (batching + continuous + paged). Fig 17.1 should be the
frontier plot (idea #4) with the ridge shaded. Fig 17.3 is great; add one sentence: the ember bars are *longer* under
continuous batching, because each step serves more users (TPOT rises): that's the trade the scheduler is making.
Fig 17.4 deserves to be a chapter's primary figure. Split as in 1.5.

**18 Guess ahead**: honest and clear. Fig 18.1 reads well now but shows tokens per pass, not time: a pass costs
k drafter steps + one verify. A two-lane timeline (drafter / checker) would make "why the speed-up is 75–90% of the
formula" visible. Add: why it fades at high batch, and draft heads (EAGLE/MTP).

**19 Many GPUs (teaser)**: the ring is a toy with no numbers, disconnected from fig 0.3.3's real costs. Chunks in
fig 19.1 are tiny segments that are hard to read. This should become Part 5 (ideas #5, #7, #8).

### Reference

**Field guide** (`/learn/field-guide`): calm, well-grouped, the MoE lab is good. It currently holds most of the
material an infra conversation needs (prefix caching, disaggregation, chunked prefill, FlashAttention, MLA...), each as
a card. That's the clearest evidence the course stops too early.

**Lab** (`/lab`): see section 3. It's a different visual language (pink/blue series, "fig. 01" numbering, bold
"What you're looking at:" inline style, wider layout) from the course, and several figures mis-handle the WebGPU path.

---

## 3. Low level: specific issues

| route | figure / place | issue | fix |
|---|---|---|---|
| `/lab` | fig 05 "The speed limit, in this tab" | After a WebGPU run (259 tok/s, "Wider loads"), it reads **"239 tok/s achieved · 223% of this tab's ceiling"**: the ceiling (107) comes from the WASM CPU bandwidth probe (53.6 GB/s) and the f32 tab, while the run was on the GPU with smaller weights. | Ceiling must follow the run's backend and dtype; or disable with "this figure measures the CPU path". >100% of a ceiling should be impossible by construction. |
| `/lab` | fig 06 roofline | One big "gpu" dot plotted against WASM roofs (relaxed-SIMD 110 GFLOP/s, 53.6 GB/s). GPU ops on CPU roofs are meaningless. | Per-backend roofs; GPU path shows whole-token dot only, labelled. |
| `/lab` | fig 04 "All 144 attention heads" | Still "waiting for the engine…" after the engine is loaded and a WebGPU generation finished. Empty state never resolves. | Say why ("head capture runs on the CPU engine") with a one-click switch. |
| `/lab` | whole page | Series colours pink/blue break DESIGN.md's ember/blue/olive contract; caption style and fig numbering differ from the course; ~14,000 px. | Port to `<Fig>` and `chartColors()`; split into tabs or anchor nav. |
| `/` | hero | First number shown is 97 tok/s (recorded V8 1-core run), above "2,200× faster". | Two-up: first engine vs fastest rung replaying at real speed (idea #12). |
| `/learn/why-its-expensive` | fig 0.2.2 vs 0.2.3 | "each user's speed: unchanged" to 256 users vs "each user 150 tok/s" at 16. | Caption in 0.2.2: "ignoring each user's KV cache, which 0.2.3 adds". |
| `/learn/why-its-expensive` | fig 0.2.4 | Constraint map duplicates the field guide; homework at chapter 0.2. | Move; reprise it in the capstone filled in. |
| `/learn/the-hardware` | fig 0.3.1 | "1,280 parallel lanes… 4.3 TFLOP/s" for M4 fine; H100 989 TFLOP/s is tensor cores, presented as lanes. | Add a "matrix units" layer for H100; note FP8 2×. |
| `/learn/the-hardware` | fig 0.3.3 | 2 GPUs × 80 GB for 140 GB of weights leaves ~20 GB for KV; not mentioned, so "2 GPUs works" looks comfortable. | Show KV headroom bar per row (feeds idea #1). |
| `/learn/embeddings` | fig 3.1 | Overlapping clusters top-left, isolated "days", empty middle; 53-chip picker duplicates the plot (open since REVIEW.md). | Rank-normalise or hand-place via a better projection; drop the picker. |
| `/learn/layers` | fig 6.2 | Before pressing play, shows "group 100%" / "of 0.0%", which looks like a result. | Explicit idle state. |
| `/learn/memory-or-maths` | fig 9.1 | Crossover = this tab's 1.6 FLOP/B; other chapters use 41 and 295. | Ridges-by-machine strip; roofline as primary (idea #2). |
| `/learn/kv-cache` | prose under fig 12.1 | "flat, not rising" (`ch/KvCache.tsx:55`). | "Flat for the matmuls; attention still reads every earlier token, so it rises slowly." |
| `/learn/faster-arithmetic` | prose vs fig 13.2 | 5.8× vs 5.3×. | One clause naming the two baselines. |
| `/learn/smaller-numbers` | prose after fig 15.2 | "8× higher, around 1,200" (`ch/Quantization.tsx:130`). | Show the arithmetic. |
| `/learn/*` (11–17) | climb chart | Single "limit, 32-bit: 243" dashed line, passed by later bars. | Per-bar ceiling tick, "% of own ceiling" label. |
| `/learn/the-gpu` | result prose | "close to the 105 GB/s" (actual ≈ 73 GB/s, 69%). | State 69%. |
| `/learn/batching` | fig 17.1 | No marker for the ridge / memory-bound region (the chapter's key idea). | Shade memory-bound region per dtype (WRITEUP stage 9 viz). |
| `/learn/batching` | fig 17.3 | "Then hover a bar" (no hover on touch); doesn't point out that ember bars lengthen under continuous batching. | Tap to pin; one caption line on TPOT. |
| `/learn/speculative-decoding` | fig 18.1 | Blue = accepted (DESIGN exception); no time axis; thrown-away guesses only as ×. | Add drafter/checker timeline lane. |
| `/learn/many-gpus` | fig 19.1 | Chunks are tiny 4-segment strips in node boxes; no bytes or time; not tied to fig 0.3.3. | Bigger chunks, message size + time readout, latency vs bandwidth toggle (idea #7). |
| all chapters | left rail | Unlabelled dots except the current chapter; labels on hover only (none on touch); parts separated only by gaps. | Part labels in the rail; show 2 neighbours' titles. |
| all chapters | check yourself | Recognition only; some stems leak (09 q1); no numeric or transfer questions; not stored. | Ideas #6, #14. |
| `/learn/what-is-inference` | check q3 | Determinism answer wrong for production. | Rewrite (1.3 #9). |
| `/learn/tokens`, `/learn/the-loop` | stat rows | Four tiles (DESIGN ≤ 3). | Cut one. |

Mobile (390 px): no horizontal overflow on `/learn/batching` or `/lab`; smallest SVG text in fig 17.3 is 13 px. The
earlier phone fixes hold.

---

## 4. Interactivity and intuition: ideas, ranked

Ranked by intuition gained per unit effort, for this reader. "Data" says what real numbers drive it. All follow DESIGN's
colour contract (ember compute, blue memory, olive links, grey idle).

1. **The memory budget game: fit a model on N GPUs.** *Learner:* picks a model (7B, 70B, 405B, a 671B MoE), precision
   (bf16/FP8/int4), GPU (H100 80 GB, H200 141, B200 192, M4 24) and count, then drags context length and batch size.
   A stacked bar per GPU fills: weights (blue grid), KV cache (blue pages), activations/workspace (grey), "free". When
   it overflows, the bar breaks. Then the same controls show the decode ceiling and per-user speed. *Intuition:*
   capacity vs bandwidth are separate walls; KV, not weights, is what limits batch at long context; why GQA/MLA/KV-FP8
   exist. *Data:* real model shapes (layers, kv_heads, head_dim), datasheets, `specs.ts`. *Effort:* M. *Chapter:* new
   21 (or 0.3 → Part 5); reused in the capstone.

2. **Live roofline where you place your own workload.** *Learner:* a roofline for the selected machine (tab / M4 CPU /
   M4 GPU / H100 / B200, FP16 vs FP8 roofs). Dots: "decode b=1", "decode b=64", "prefill 512", "attention at 32k". The
   reader drags batch size or prompt length and the dot slides right along the slanted roof until it hits the flat one;
   predicted tok/s updates. *Intuition:* one picture explains decode, prefill, batching, quantisation and why spec
   decoding is free. Reconciles the three ridges. *Data:* measured roofs (`data.json`, `bench_bandwidth`), datasheets,
   analytic FLOPs/bytes per op (WASM trace schema already has them). *Effort:* M. *Chapter:* 09 primary figure; mini
   version reappears in 15, 17, 20.

3. **A persistent mental-model diagram that fills in.** *Learner:* the 0.1.3 server diagram (users → scheduler →
   device: weights, KV pages, forward pass, links) lives in a corner/rail. Each chapter lights up and labels the part
   it explains (ch 12 adds the cache pages; 15 shrinks the weight grid; 17 adds the scheduler queue; 19 adds paging; 22
   splits the device). Click any part to jump to its chapter. *Intuition:* where every technique sits in one system;
   progress as a filling map instead of a dot list. *Data:* diagram + chapter metadata (`concepts.ts`). *Effort:* M.
   *Chapter:* global (rail / contents page).

4. **The frontier: tokens/s per user vs tokens/s per GPU.** *Learner:* fig 17.1 redrawn as the industry's Pareto plot:
   x = interactivity (tok/s/user), y = throughput (tok/s total), one curve per dtype, points labelled by batch size.
   Drag an SLO line ("each user must get ≥ 50 tok/s") and the best achievable throughput snaps; cost/M tokens reads
   out. *Intuition:* there is no single "speed"; serving is choosing a point; quantisation shifts the curve, batching
   moves along it. *Data:* `bench/batching.json` (real), with an H100 estimate overlay from fig 0.2.3's model.
   *Effort:* S. *Chapter:* 17.

5. **You are the scheduler.** *Learner:* requests arrive (real arrival trace from `bench/serving.json`), each a bar with
   prompt and unknown output length. Each tick the reader admits/evicts into a 32-slot batch with a KV page budget, or
   picks a policy (static, continuous, continuous + chunked prefill budget). Score: TTFT p50/p99, TPOT p99, goodput
   under an SLO. Then "replay with the engine's scheduler" overlays the measured result. *Intuition:* why static batching
   wastes, why admission is a memory problem, the TTFT-vs-jitter trade of prefill budgets. *Data:* `serving.json`
   (`rate=8/16`, `budget=32/512`, `gantt`, `iters`, `kv_snapshots`). *Effort:* L (M for policy-picker only).
   *Chapter:* new 18 (scheduling).

6. **Predict, then reveal, before every primary figure.** *Learner:* a single input above the figure: "Guess: how many
   tok/s will 4-bit get?" (slider on a log axis), commit, then the figure animates to the measured value and shows the
   gap. Store guesses locally and show calibration at the end ("you underestimated memory effects 6 times out of 8").
   *Intuition:* forces the reader to run the model in his head; the surprise is the lesson (Scaling Book's quizzes,
   Nicky Case's play-first). *Data:* existing measured numbers. *Effort:* S (one component, wire per chapter).
   *Chapter:* 10, 12, 13, 14, 15, 16, 17, 18.

7. **Drag-to-shard: split a model across GPUs and watch the talk.** *Learner:* a layer stack (attention + FFN blocks,
   real 70B shapes) and 8 GPU boxes. Drag a cut: horizontal (pipeline), vertical (tensor), or move experts (EP). A
   per-token timeline shows read time (blue), compute (ember), all-reduce/all-to-all/send (olive), pipeline bubbles
   (grey). Toggle NVLink vs InfiniBand and batch 1 vs 64 (messages grow from 16 KB latency-bound to MB bandwidth-bound).
   *Intuition:* why TP stays inside a node, why PP crosses nodes, why batch changes the answer, why GPT-2 TP is slower.
   *Data:* fig 0.3.3 model, NCCL-style α–β, ring formula (DISTILLED law 10). *Effort:* L. *Chapter:* new 22.

8. **MoE dispatch on the path, with modern presets.** *Learner:* the field guide's MoE lab moved into a chapter, with
   presets Mixtral (8, top-2), DeepSeek-V3 (256 + 1 shared, top-8), gpt-oss. Tokens/step slider shows "experts touched"
   rising towards all; a second view splits experts across GPUs with all-to-all bytes and a hot-expert straggler.
   *Intuition:* active vs total params; why MoE decode is still memory-bound; why EP and load balancing exist. *Data:*
   published configs; `MoeLab.tsx`. *Effort:* S (move) / M (EP view). *Chapter:* new 23.

9. **KV cache as virtual memory, stepped.** *Learner:* fig 17.4 promoted: scrub through `kv_snapshots`; then toggle
   "prefix caching" on a workload where 20 requests share a 2k-token system prompt: shared pages get a refcount, TTFT
   bars shrink, memory frees. Toggle GQA/MLA/FP8-KV and watch pages shrink. *Intuition:* paging, sharing, copy-on-write,
   why "cached input tokens" are 10× cheaper on APIs. *Data:* `serving.json` `kv_snapshots`, `kv_stats` (real); prefix
   sharing simulated on the same page allocator rules, labelled. *Effort:* M. *Chapter:* new 19.

10. **Where a token's time goes, per op, per rung.** *Learner:* one stacked bar per climb rung (12 layers × ops + LM
    head), small multiples down the page; hover an op for bytes, FLOPs, achieved GB/s. Scrub rungs and watch the long
    pole move: recompute → matmul → LM head → dispatch/waiting. *Intuition:* optimisation is moving the long pole;
    Amdahl inside a token. *Data:* recorded `op` events (`public/recorded/*`), `results.jsonl`. *Effort:* M (lab already
    has the data path). *Chapter:* new 11½, reprised in 13–16.

11. **Per-bar ceilings on the climb ("% of own ceiling").** *Learner:* the climb chart gains a ghost ceiling tick per
    bar (f32 CPU 243, Q4 CPU ~1,200, Q4 GPU ~1,180...). Toggle "absolute" vs "% of ceiling". *Intuition:* fixes the
    chart's current contradiction; shows that later steps were about overhead, not bytes. *Data:* `data.json`.
    *Effort:* S. *Chapter:* 11–17 (`Climb.tsx`).

12. **Engine race, side by side.** *Learner:* two lanes write the same prompt at measured real speed: pick any two rungs
    (naive vs KV cache; CPU Q4 vs GPU Q4) or nanoinfer vs llama.cpp vs MLX from the head-to-head. Includes the MLX
    4-bit head trap: fastest lane, garbage perplexity, quality meter turns red. *Intuition:* speed differences felt in
    time; "every speed-up has a quality bill". *Data:* `bench/headtohead.json`, recorded timings, `results.jsonl`.
    *Effort:* S (recorded) / M (live). *Chapter:* landing hero, 15, 16.

13. **One vector or many: watch reuse appear.** *Learner:* fig 4.1 with a "tokens" stepper 1 → 4 → 16: the matrix
    stays, the vector becomes a matrix of columns; each weight lights once and does k multiply-adds; FLOP/byte counter
    climbs. *Intuition:* the single mechanism behind prefill speed, batching and spec-decode verification. *Data:* real
    GPT-2 corner already used in 4.1. *Effort:* S. *Chapter:* 04 (referenced from 08, 09, 17, 18).

14. **Spaced-repetition review deck.** *Learner:* each chapter contributes 2–4 cards (numeric and transfer questions:
    "70B FP8 on one H100: decode ceiling?" "Prefix hit rate falls: which metric moves?"). `/learn/review` shows due
    cards; localStorage only, wrapped in try/catch. *Intuition:* retention of the numbers that matter (bandwidths,
    bytes per token, ridges). *Data:* `checks.ts` extended. *Effort:* S–M. *Chapter:* global.

15. **Context-length slider on the speed limit.** *Learner:* fig 10.1 / 12.1 gain context (0 → 128k): bytes per token =
    weights + n × KV/token; the KV band overtakes Q4 weights near 1k tokens for GPT-2 (75 MB vs 70 MB), and for a 70B
    with/without GQA. *Intuition:* why long context is a bandwidth problem, why KV quantisation and GQA exist.
    *Data:* DISTILLED law 5, metal-kv8 measurements (+5.4% at 960). *Effort:* S. *Chapter:* 10 or 12.

16. **Spec decode: tune k yourself, with time on the axis.** *Learner:* slider for k and acceptance α (or pick text
    kind); a two-lane timeline (drafter steps grey/blue, verify pass ember) and a curve of measured speed-up vs the
    formula E/(kc+1); the k = 8 cliff (9 rows → 16-row kernel) is visible. A batch-size slider shows the gain vanish
    as verification stops being free. *Intuition:* why it's a memory-bound trick and when it stops paying. *Data:*
    `bench/specdec.json` (pairs, k sweep, playback). *Effort:* M. *Chapter:* 18/20.

17. **Disaggregation: split the phases.** *Learner:* one pool vs two pools (prefill GPUs / decode GPUs) with a KV
    transfer link; drag the prefill:decode ratio and prompt/output lengths; watch TTFT and TPOT tails decouple, and the
    KV transfer cost (olive) grow with prompt length. *Intuition:* why labs separate phases; the cost is moving state.
    *Data:* simulated from `serving.json` per-phase timings, labelled estimate; DistServe numbers external. *Effort:* M.
    *Chapter:* new 24.

18. **Quantisation microscope with quality on the axis.** *Learner:* fig 15.1 plus a block-size slider (32/64/128) and
    a "which matrix" picker (FFN vs LM head); a quality strip (top-1 agreement, Δppl) beside the speed limit. The LM
    head at 4 bits turns the quality strip red. *Intuition:* outliers, block scales, why some matrices can't be squeezed.
    *Data:* real weights (already loaded for 15.1), WRITEUP stage 9/12 quality numbers. *Effort:* S–M. *Chapter:* 15.

19. **Scrubbable generation timeline with linked panes.** *Learner:* fig 8.1 gets a scrubber; dragging it updates the
    text, the KV strip, the top-10 bars and the per-op bar for that token (Bret Victor's "abstract over time, then
    climb down"). *Intuition:* one run, many views; prefill vs decode in one gesture. *Data:* recorded run trace.
    *Effort:* M. *Chapter:* 08.

20. **Reactive numbers in prose.** *Learner:* key sentences carry draggable values ("At **f16**, a **7B** model on an
    **H100** tops out at **240 tok/s**"): drag f16 → int4 and the sentence rewrites. *Intuition:* the law becomes a
    toy you hold. *Data:* calculator functions already in fig 0.2.1/10.1. *Effort:* S (a `<Live>` span component).
    *Chapter:* 0.2, 10, 15, 21.

21. **Ring all-reduce with bytes and time.** *Learner:* fig 19.1 with a message-size slider (16 KB → 1 GB) and link
    picker; a readout shows latency-term vs bandwidth-term and the ring/tree crossover; chunk size visibly shrinks with
    N while total time stays flat. *Intuition:* DISTILLED law 10; why small-batch TP is latency-bound. *Data:* α–β
    model, nccl-tests external numbers. *Effort:* S. *Chapter:* 22.

22. **Capstone scenario: serve it.** *Learner:* given "Llama-3-70B, 2k-token prompts, 500-token answers, 50 req/s,
    TTFT p99 < 1 s, ≥ 40 tok/s/user", choose GPUs, precision, TP degree, batch cap, prefix caching, spec decode. The
    simulator (built from ideas 1, 2, 4, 7) scores cost/M tokens and SLO attainment; reveal an expert's config and
    why. Then an annotated real `vllm serve …` command and a mock infra thread, every term linked to its chapter.
    *Intuition:* integration: the reader can reason end-to-end and read the conversation. *Data:* models above,
    clearly labelled estimate. *Effort:* L. *Chapter:* new 25.

23. **"Break it" toggles.** *Learner:* on fig 12.1, 13.1, 16.1, a "remove this optimisation" switch that re-runs the
    recorded step without it (e.g. GPU: put the waits back; KV: drop the cache at position 900). *Intuition:* causal
    ownership of each speed-up. *Data:* recorded rungs. *Effort:* S each. *Chapter:* 12–16.

---

## 5. Top 10, prioritised (impact × effort)

| # | what | why first | effort | where |
|---|---|---|---|---|
| 1 | **Fix the correctness/consistency bugs** in 1.3 and section 3 (KV "flat", "8×/1,200", "close to 105 GB/s", 5.8/5.3, determinism q, 0.2.2 vs 0.2.3 caption, lab fig 05's 223% and fig 06's GPU-on-CPU roofs, fig 04 empty state) | Expert credibility; each is a sentence or a guard | S | `ch/KvCache.tsx`, `ch/Quantization.tsx`, `ch/Gpu.tsx`, `ch/Arithmetic.tsx`, `checks.ts`, lab panels |
| 2 | **Per-bar ceilings on the climb** (#11) | The course's central chart currently contradicts its thesis | S | `src/learn/Climb.tsx` |
| 3 | **Frontier plot for batching** (#4) + split 17 into batching / scheduling | The industry's main chart, from data already measured | S | `ch/Batching.tsx`, `bench/batching.json` |
| 4 | **Live roofline as ch 09's primary figure** (#2), with ridges-by-machine | One figure that unifies decode/prefill/batching/quant and reconciles 1.6/41/295 | M | `ch/BytesFlops.tsx` |
| 5 | **Memory budget game** (#1) + KV context slider (#15) | Capacity, KV and long context are the missing half of the mental model | M | new chapter; `ch/p0/specs.ts` |
| 6 | **Predict-then-reveal** (#6) on Part 2–3 primary figures | Cheapest way to turn watching into thinking | S | `kit.tsx` `<Fig>` slot |
| 7 | **Promote MoE and paged/prefix KV to chapters** (#8, #9) | MoE is the default architecture; KV memory management is the serving problem | S–M | `guide/MoeLab.tsx`, `ch/Batching.tsx` fig 17.4 |
| 8 | **One vector or many** in ch 04 (#13) + heads/shape sheet in 05/06 | Fixes the two places the mental model silently breaks | S | `ch/DotProduct.tsx`, `ch/Attention.tsx`, `ch/Block.tsx` |
| 9 | **Re-cut the arc** (1.5): thin Part 0, Rosetta to the field guide, Parts 4–5, capstone stub | Delivers Part 0's promise ("follow the conversation") | M (structure) | `chapters.tsx`, `IA.md` |
| 10 | **You are the scheduler** (#5) | The best "aha" for a distributed-systems reader, on real arrival traces | L (M as policy-picker) | new ch 18; `bench/serving.json` |

Next after these: persistent mental-model diagram (#3), drag-to-shard (#7), spaced review (#14), capstone (#22).
