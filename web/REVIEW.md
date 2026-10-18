# Cold-read review: the course, as Josh

Reader: a strong systems engineer (caches, sharding, consensus, virtual memory, SIMD) with no ML. Walked in order in a
real browser (1280 desktop dark, light, and 390 px mobile), reading the prose and using every figure. Judged against
`DESIGN.md` and `IA.md`. Screenshots: `screenshots/review-before-*.png`, `screenshots/review-after-*.png`.

## Overall

Part 0 → 3 is a good spine: model = read-only blob + fixed program (0.1) → every token reads every weight (0.2) →
the hardware (0.3) → how the model computes (part 1) → why it's slow (part 2) → the climb (part 3). The prose is calm
and the systems analogies land. The trouble is ordering and density at a few hinges:

1. **The KV cache is the course's most-used undefined term.** Part 0 leans on it (fig 0.1.3, fig 0.2.3's whole cost
   curve, "every user brings their own KV cache"), but what it *is*, why it exists and why it grows waits until
   chapter 12, and chapter 12's figure never draws a cache. The reader can't explain the cost curve they just used.
2. **Part 0 front-loads vocabulary.** "Decode" and "prefill" are used in 0.2 before chapter 08 defines them. Fig 0.2.4
   is a 1,070 px wall of 28 technique chips at the end of the second chapter, most of them names the reader has
   never heard; it reads as homework, not a map.
3. **"When does a CPU win?" is never answered.** 0.3 says why GPUs win; nothing says when a CPU (or the M4's shared
   memory) is the right call, though the engine's own numbers (CPU 497 vs GPU 817 tok/s; M4 CPU and GPU read the same
   memory at 98 vs 105 GB/s) make the case.
4. **MoE and expert parallelism are off the path.** They live only in the field guide (a good MoE lab, but nothing in
   chapters 0.1–19 sends you there). Chapter 19 lists pipeline and tensor splits only.
5. **The check-yourself questions give themselves away.** In 45 of 46 questions the right answer is the longest
   option, often 3× the length of the distractors ("Because the arithmetic is very complicated" vs a full
   explanation). A systems reader picks the long one without thinking.
6. **Several figures break the colour contract.** Bytes on the road in fig 9.1 are ember (they're memory). Fig 1.1
   paints every probability bar ember, so the picked token isn't the focal point. Fig 17.4's KV pages are grey.
   Fig 6.1's attention box is blue.
7. **Tiny labels on phones.** Fig 0.1.3 (server) and fig 0.3.1 (chip) scale a 640 px-wide SVG down to 358 px: labels
   render at 4–5 px. Fig 6.1's labels are 7–8 px even on desktop.
8. **Animations never stop.** `useFrame(true, …)` in figs 6.1, 9.1, 13.1 runs forever; the 0.1.1 and 0.3.1 hooks
   latch "seen" and never pause. Every chapter with a looping figure burns a rAF loop while you read the next screen.

## Per chapter

Legend: **lost** = term used before explained or a step skipped · **over** = overwhelming · **vis** = visual issue vs
DESIGN.md · **why** = is the motivating constraint clear · **jank** = broken or awkward interaction.

### 0.1 What inference is
- lost: "KV cache" appears in fig 0.1.3 and the prose as "per-request state" with no hint why a request needs state.
  The input/output contract (tokens in, a score per vocabulary token out) only appears inside the pseudo-code.
- over: fine. Three figures, each with one job.
- vis: fig 0.1.3 labels are 9.5–10 px on desktop; on a phone the whole diagram is ~0.56× scale (labels ~5 px), the
  dashed "tokens stream back" line and greyed boxes become noise. Fig 0.1.2's inner matrices are unlabelled until
  clicked.
- why: clear and well framed (search index vs queries).
- jank: fig 0.1.1's animation runs forever once seen.
- check: all three right answers are the longest option.

### 0.2 Every word reads every weight
- lost: "decode" (used 6×) before it's defined; "the ridge from chapter 09" points forward; KV cache size (0.5 MB per
  token) arrives in a source line with no explanation of what is stored or why it scales with context.
- over: fig 0.2.1 defaults to 70B bf16, so three of the four machine rows say "doesn't fit": the first look at the key
  speed-limit chart is mostly empty hatching, with "doesn't fit" text drawn over the gridlines. Fig 0.2.3's controls
  wrap to three rows and the "KV cache" label orphans from its toggle. Fig 0.2.4 is the wall of chips described above.
- vis: fig 0.2.2 is the best figure in part 0: one ember square, one slider, three stats.
- why: very clear. The I/O-bound-service analogy is exactly right.
- check: right answer longest in all three (q1: 130 chars vs 41).

### 0.3 The hardware in one picture
- lost: "when does a CPU win" never answered (see overall 3).
- vis: fig 0.3.1 on a phone: unreadable labels (bus label, "24 GB memory", core caption at ~4 px), stats wrap.
  Fig 0.3.2 good. Fig 0.3.3 good.
- why: clear. The Amdahl's-law framing of all-reduce is the right hook for this reader.
- jank: fig 0.3.1's packets animate forever once seen.
- check: right answer longest in both.

### 0.4 A Rosetta stone
- lost: placed before part 1, so "paged attention", "prefix caching", "speculative decoding" are names with an
  analogy but no referent yet. Works better as a reference than a chapter; acceptable because each row links on.
- vis: good; calm rows, filter chips.
- check: right answer longest in all three.

### 01 Guess the next word
- vis: every probability bar is ember, the picked one a lighter ember: the focal point (the pick) doesn't pop, and it
  spends the "compute" colour on probabilities. Caption says "black is what the model has added" (it's light text in
  dark mode).
- why: clear. A strong opener.

### 02 Tokens
- vis: four stat tiles (DESIGN says at most three). Chip tints include ember/blue/olive hues as neighbour separators,
  which collides with colour semantics, but they're faint; left.
- why: clear.

### 03 Embeddings
- over: one figure holds a scatter, a 53-word picker that duplicates the scatter, a 768-bar barcode and an 8-row
  neighbour chart: ~1,150 px. The 2-D map leaves most of its area empty (clusters squashed to the top).
- vis: barcode legend is good; neighbour bars all ember.
- why: clear.

### 04 Multiply and add
- vis: the animated 8×8 grid sits in the left 45% of the figure, the right half is empty; cell numbers are ~8 px; the
  sign legend (ember +, blue −) is only in the caption; the output column is a row of bare "=" signs until filled.
- why: clear ("make that fast and you've made the model fast").

### 05 Attention
- lost: the chapter never says the obvious systems consequence: the keys and values of every earlier token are what
  a new token needs, and they never change. That sentence is the bridge to the KV cache, and it's missing.
- vis: good. One focal arc, readable.

### 06 Layers and the residual stream
- lost: 0.1 called it "feed-forward"; here it's "MLP" without saying they're the same.
- vis: fig 6.1 is drawn at a third of the figure width with 7–8 px labels and annotations floating far right;
  attention box blue (memory colour) for no reason. Fig 6.2 fine.
- jank: fig 6.1 animates forever, on- or off-screen.

### 07 Sampling
- vis: good. Clear controls, sensible defaults.

### 08 The generation loop
- lost: the reason prefill is faster per token (it shares each weight read across all prompt tokens) is in the check
  answer, not the prose. "The tall first bar" is the wide first bar. Four stat tiles.
- why: the two-phase idea lands; *why* the phases differ doesn't yet.

### 09 Memory or maths
- lost: decode is "0.5 operations per byte" here and "~1 FLOP per byte" in 0.2 (f32 vs bf16) with no reconciling note.
- vis: the boxes on the road are ember (they're bytes: should be blue); labels "memory", "weights", "processor" are
  8–9 px; the caption talks about a truck that isn't drawn.
- jank: animates forever.
- why: very clear. The single most useful chapter for this reader.

### 10 The speed limit
- over: same calculator as fig 0.2.1 without acknowledging it. Reinforcement is fine; a line saying so helps.

### 11 The starting line
- vis: fig 11.1's grid draws all future cells as a dark mass, labels "text"/"prompt" sit at odd rows, only the first
  three column counts show. Hard to parse in 5 seconds.

### 12 The KV cache
- lost/vis: the key figure is fig 11.1 again with a toggle; in "KV cache" mode it still says "without a cache" in the
  title and **never draws the cache**. Nothing shows the cache growing or how big it gets; "why it grows" is folded.
- why: the prose motivation is clear (memoise immutable keys/values).

### 13 Faster arithmetic
- vis: fine. Prose says 5.8×, climb says 5.3× (different baselines). Persona already knows ILP; it reads quickly.

### 14 Many cores
- vis: good. Clear bar chart, one focal bar.

### 15 Smaller numbers
- lost: the cost is "predictions about 8% worse": worse how? (perplexity +7–8%, top-word agreement ~80%).
- vis: four stats wrap to two rows; no legend in the plot (blue stem = original, ember dot = stored).

### 16 The GPU
- good. "Ask after every step" ↔ synchronous RPC is the right analogy.

### 17 Batching
- vis: figs 17.1 and 17.3 are excellent. Fig 17.4 (paged KV) draws in-use pages grey (KV cache is blue in DESIGN).
- lost: paged attention is explained as "a filing cabinet that hands out folders"; for this reader "virtual memory:
  fixed-size pages and a per-request page table" is the one-line explanation.

### 18 Guess ahead
- over: the playback is a run of underlined text; check boundaries are hard to see. Seven-option text picker wraps to
  two rows. A large empty area is reserved under the text.
- why: clear and honest (acceptance depends on the text).

### 19 Many GPUs
- lost: no expert parallelism or MoE (see overall 4).
- vis: ring is small; ring links are grey rather than olive (links are olive in DESIGN).

## Could the reader explain it afterwards? (1–5, before)

| concept | score | gap |
|---|---|---|
| what inference is | 4 | Clear; the in/out contract is buried in pseudo-code. |
| why decode is memory-bound | 4 | 0.2 and ch 9 are strong; "decode" undefined in 0.2; 0.5 vs 1 FLOP/byte unexplained. |
| the KV cache and why it grows | 2 | Used from 0.1, explained in ch 12; ch 12's figure never shows a cache or its size. |
| quantisation: speed and cost | 3 | Speed obvious; the cost is a vague "8% worse". |
| CPU vs GPU, when each wins | 2 | Only "GPUs win"; the CPU case is never made. |
| batching / continuous / paged | 3 | Batching 5, continuous 4, paged 2 (folded, wrong analogy). |
| speculative decoding | 4 | Prose clear; playback hard to read. |
| MoE | 2 | Only in the field guide, nothing on the path points there. |
| tensor / pipeline / expert parallelism | 2 | Tensor and pipeline named in 0.3 and 19; expert absent. |

## Fixes (phase 2)

_Filled in below as each fix lands._
