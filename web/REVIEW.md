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

Highest impact first. Each is its own commit.

1. **The KV cache, explained where it's first used.** 0.2 gains a short section ("The one piece of state: the KV
   cache"): attention needs every earlier token's key and value, they never change, so engines memoise them in an
   append-only per-request log; 0.5 MB per token for a 7B, 2 GB at 4k tokens, unshareable, read every step. Chapter 05
   ends with the bridge sentence. **Fig 12.1 now draws the cache**: computed cells ember, cells read back from the
   cache blue, the cache itself as a strip of pages growing one per pass with its size (73.7 KB per GPT-2 token), and
   a "total so far (no cache: …)" stat that shows n vs n². Defaults to the cache view; chapter 11 keeps the no-cache
   version.
2. **Part 0 stops front-loading vocabulary.** 0.2 defines prefill and decode before using them. The constraint map
   (fig 0.2.4) opens one constraint at a time with a technique count on the closed rows (1,068 px → 575 px), with a
   caption that says not to learn the names yet. The field guide keeps the full map.
3. **Phone layouts for the two unreadable diagrams.** Fig 0.1.3 (server) stacks users → scheduler → device at 11–13 px
   type; fig 0.3.1 (chip) puts memory on top, the bus running down, the chip full width below, with larger in-die
   labels. Fig 0.1.1 also stacks. Before/after: `review-before-0.3.1-mobile.png`, `review-after-0.1.3-mobile.png`,
   `review-after-0.3.1-mobile.png`. A 390 px sweep of every chapter and the field guide finds no horizontal overflow
   and no figure text under ~8 px.
4. **Animations pause off screen.** New `useOnScreen` hook (true while within 100 px of the viewport, false again when
   it leaves) and a `figRef` prop on `<Fig>`. Every looping figure (0.1.1, 0.3.1, 4.1, 6.1, 6.2, 8.1, 9.1, 11.1/12.1,
   13.1, 18.1, 19.1, the BPE stepper) passes it to `useFrame` / its interval. Verified with a rAF counter on chapter 9:
   ~60 frames/s on screen, 0 when scrolled away. Fig 18.1 no longer plays through before you reach it.
5. **Check-yourself rebalanced.** Every option rewritten so the right answer is short and the distractors are
   plausible misconceptions of similar length: the right answer is the longest option in 12 of 46 questions (was 45).
6. **When a CPU wins.** 0.3 gains "So when does a CPU win?": on the M4 the CPU and GPU read memory at 98 vs 105 GB/s, so
   single-user decode is close (497 vs 817, mostly from the GPU waiting less); prefill isn't (2,415 vs 6,800). Rule of
   thumb: decode follows bandwidth, prefill and batches follow FLOPs.
7. **MoE and expert parallelism on the path.** Chapter 19 lists three splits (pipeline, tensor, expert) and explains
   MoE in one paragraph ("knowledge without FLOPs, not without bytes"), linking the field guide's MoE lab.
8. **Colour contract.** Fig 1.1 and every `.lr-bar` chart: bars grey, only the highlighted bar ember. Fig 9.1: bytes
   blue, processor ember, labels 11–12.5 px, no phantom truck in the caption, plus a line on what the processor is
   doing. Fig 6.1: redrawn at full width with 11.5–14.5 px labels, attention and feed-forward both ember (compute),
   the stream grey; "MLP" is now "feed-forward (MLP)" to match 0.1. Fig 17.4: in-use KV pages blue. Fig 19.1: links
   olive, GPUs larger.
9. **Smaller gaps.** Ch 8 says why prefill is cheaper per token (one weight read serves the whole prompt) and fixes
   "tall" → "wide". Ch 9 reconciles 0.5 FLOP/byte (f32) with 0.2's ~1 (bf16). Ch 10 acknowledges it repeats 0.2's
   division. Ch 15 names the cost of 4-bit: top choice differs from 32-bit about 1 time in 5, perplexity +8% (new
   glossary entry), with a legend and three stats. Paged attention is explained as virtual memory (pages + page
   table), with a visible pointer to the fold.
10. **Figure clean-ups.** Fig 0.2.1 defaults to 7B so every machine row has a bar; "doesn't fit" rows drop their
    gridlines and set the label on the panel colour. Fig 0.2.3's controls keep each label with its toggle. Fig 4.1 is
    centred with larger cells, a sign legend, and the running dot product written out under the grid. Fig 18.1 draws
    one group of squares per pass of the big model (blue kept, dashed thrown, ember its own token) above the text, and
    the text-kind picker fits one row. Ch 8 and 15 stat rows cut to three.

## After (1–5)

| concept | before | after | what moved it |
|---|---|---|---|
| what inference is | 4 | 4 | Unchanged; already clear. |
| why decode is memory-bound | 4 | 5 | Prefill/decode defined in 0.2, ch 8 says why, 0.5 vs 1 reconciled, fig 9.1 colour-correct. |
| the KV cache and why it grows | 2 | 4 | Defined in 0.2 with size arithmetic, bridged in ch 5, drawn growing in fig 12.1. |
| quantisation: speed and cost | 3 | 4 | Cost named concretely (1 in 5 top choices, perplexity +8%); speed-limit × stat. |
| CPU vs GPU, when each wins | 2 | 4 | New 0.3 section with the engine's own numbers and a rule of thumb. |
| batching / continuous / paged | 3 | 4 | Paged attention as virtual memory, pointed to from the main text. |
| speculative decoding | 4 | 5 | Fig 18.1 now shows each pass as a group you can count. |
| MoE | 2 | 3 | One paragraph + link in ch 19; the lab itself is still only in the field guide. |
| tensor / pipeline / expert parallelism | 2 | 3 | Expert parallelism named with its systems analogue; still a teaser chapter. |

## What remains

- **Fig 3.1 (embeddings) is still the densest figure**: scatter + 53-word picker + barcode + 8 neighbours. The 2-D
  projection leaves most of its area empty. Candidates: rank-normalise the axes, drop the picker in favour of
  clicking points, show 5 neighbours.
- **Fig 11.1/12.1 reserve height for all 20 rows** from the start, so the first seconds show empty space under a short
  grid. Fine for layout stability; could start smaller.
- **MoE has no figure on the path.** The field guide's MoE lab is good (its dispatch lines are busy at 64 tokens and it
  has four stats); a trimmed version could sit in chapter 19.
- **Rosetta stone placement.** 0.4 still comes before part 1; it reads better as a reference after part 1. Left, as
  every row links on.
- **Stat rows over three** remain in ch 2 (four tiles) and the MoE lab (four).
- **Token chip tints** (ch 2) borrow ember/blue/olive as neighbour separators; faint, but against the colour contract.
- **Ch 13** says 5.8× in prose while the climb says 5.3× (different baselines); worth one clause.
- **Not re-walked in full after the fixes**: chapters 2, 3, 7, 10, 13, 14, 16 got no changes beyond shared CSS (grey
  default bars) and were spot-checked only.
