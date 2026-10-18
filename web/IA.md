# nanoinfer.live: information architecture

The site is two things at once: an interactive course in how LLM inference works, and the record of a from-scratch
Rust engine being made fast. The course is the front door. The engine's measurements are the evidence inside it. The
dense, everything-at-once figures live in the Lab, for people who've done the path.

Rules for every page:

- One idea per chapter. One primary figure. Anything extra goes behind a "go deeper" fold.
- No figure before the concept it needs. Prose explains, then the figure shows.
- Every figure says **what you're looking at** (axes, colours, what to notice) and **what to try**.
- Jargon is defined inline on first use (dotted underline, hover or tap for a one-line definition).
- One column of prose. Figures sit in the column, a little wider, never side by side with other figures.
- Provenance on every figure: toy example, diagram, real GPT-2 weights, recorded engine run, measured benchmark, or live
  in this tab. Numbers come from docs/LOG.md, docs/WRITEUP.md and bench/*.json.
- Charts follow the dataviz rules: thin marks, hairline solid grids, legend for 2+ series, text in text colours, hover
  tooltips, a validated 3-colour chart palette (ember / blue / olive) plus grey for "everything else".
- Select any text, anywhere, to ask a question about it (streamed answer in a side panel, the section as context).

## Sitemap

```
/                     Landing: the question, one live demo, the map of the journey
/learn                Contents (same map, with progress)
/learn/<chapter>      23 chapters in 4 parts (part 0 + parts 1–3, below), prev/next, progress rail
/learn/field-guide    Reference: the constraint map + every term, grouped (deep links: #moe, #gqa, ...)
/lab                  The engine room: the full live demo, curated and simplified
```

## Landing (/)

- The question: *What happens when a model writes the next word, and how fast can one computer do it?*
- One demo: type a prompt, GPT-2 continues it (recorded run until the reader loads the engine; then live, in the tab).
  Speed shown in plain words: "≈ 100 tokens a second: for every token, your computer reads all 500 MB of the model".
- The map: parts 0–3 as stops, then the lab. "Start here" goes to chapter 0.1 (part 0 is the recommended start);
  secondary links jump to part 1 or the field guide.

Every chapter ends with **check yourself**: 2–3 intuition questions (pick one, the reasoning appears; nothing is scored
or stored), from `src/learn/checks.ts`, rendered by `src/learn/Check.tsx` under the chapter body.

Chapter numbers: part 0 is 0.1–0.4; parts 1–3 keep 01–19, so "chapter 9" in older prose still means memory-or-maths.
Visual language (colour and shape semantics, figure template, motion) is in `DESIGN.md`.

## Part 0: Inference for systems engineers

For a reader who knows distributed systems but no ML. Leans on systems analogies; explains ML ideas from scratch.

| # | chapter | the one idea | primary figures |
|---|---|---|---|
| 0.1 | What inference is | Training writes the weights once; inference runs a frozen function over them per token, forever. A model is a file of numbers + a fixed program; an engine is a stateful server (load, run, schedule, manage state) | Write-once/run-forever diagram; GPT-2's 124M weights as a clickable strip (real shapes); the serving stack stepped through its four jobs, each with its systems analogue |
| 0.2 | Every word reads every weight | Bytes per token = params × bytes/weight; bandwidth ÷ that = the speed limit; decode does ~1 FLOP/byte vs ~295 an H100 can do; the KV cache (memoised keys/values, per-request, 0.5 MB/token for a 7B) is introduced here; batching shares reads until the KV cache fills memory | Bytes-per-token calculator (GPT-2/7B/70B/405B × fp32–int4 × M4 CPU/GPU/H100/8×H100, with our measured ticks); idle-maths lanes vs users per step; $/M tokens vs batch for a 7B on an H100 (context, GQA); the constraint → technique map (one constraint open at a time) |
| 0.3 | The hardware in one picture | GPUs win decode mainly through HBM bandwidth; lanes win prefill/batching; capacity decides what fits; links are 7–70× slower than local memory, so splitting is a trade | CPU vs GPU vs H100 schematic (lanes, bus ∝ bandwidth); bandwidth ladder (memory blue, links olive); tensor-parallel 70B split: read time vs all-reduce time for 1/2/4/8 GPUs, NVLink vs network |
| 0.4 | A Rosetta stone | Inference terms ↔ distributed-systems analogues, where each holds and breaks | 19-row translation table, filterable, each row linking to its chapter or field-guide entry |

Numbers: M4 figures are measured (docs/LOG.md, data.json); H100 / NVLink / PCIe / InfiniBand are spec-sheet figures
labelled *external* with a sources line; cost and multi-GPU figures are labelled *estimate* and state their
assumptions ($2.50/GPU-hour, Llama-2-7B KV shape, ~8 µs / ~30 µs per small all-reduce).

## Field guide (/learn/field-guide)

Entry point is the constraint map (six limits: bandwidth, capacity, sequential decode, compute, communication,
unpredictable traffic → the techniques that work around each; shared with fig 0.2.4 via `src/learn/concepts.ts`). Below
it, 53 terms in six groups (models & architecture, numbers, running it, scale, metrics, hardware) as calm cards: term,
aka, one plain sentence; open one for *why it exists* (the constraint), a tiny visual where it helps, *where you'll see
it*, and its chapter. Visuals: number-format bit layouts, MHA/GQA/MQA head sharing, five parallelism patterns, and the
MoE lab (top-k of 8 experts per token, tokens per step, balanced vs skewed router: compute per token vs memory held vs
bytes read per step, and the hot-expert factor). Filter box and group jumps are sticky. Glossary tooltips across the
course link to their entry (`GUIDE_LINK` in `kit.tsx`). Content: `src/learn/guide/entries.tsx`.

## Part 1: How a model writes text

| # | chapter | the one idea | primary figure |
|---|---|---|---|
| 1 | Guess the next word | A model maps text to a probability for every possible next token | Step through a real run: top-10 guesses at each step (live: type your own, click a guess) |
| 2 | Tokens | Text becomes a list of integer IDs; tokens are the unit of work | Live tokenizer: type, see pieces and IDs (BPE merge animation folded) |
| 3 | Embeddings | Each token ID looks up a row of 768 numbers; similar words get similar rows | Word map from real GPT-2 rows + exact nearest neighbours |
| 4 | Multiply and add | Matrix × vector (many dot products) is ~99% of the arithmetic | Animated real 8×8 corner of a GPT-2 matrix × a real embedding (dot-product toy and the measured op breakdown folded) |
| 5 | Attention | A token scores earlier tokens with query·key, softmaxes, blends their values | Toy sentence: pick a query word, see weights and the blended result (GPT-2's real heads folded) |
| 6 | Layers and the residual stream | 12 blocks each add an edit to a shared vector; the guess sharpens with depth | Real residual stream per layer + "what it would say if it stopped here" |
| 7 | Sampling | Temperature and top-k reshape the probabilities; then roll a die | Real logits, temperature / top-k controls, draw tokens |
| 8 | The generation loop | One full forward pass per token: prefill reads the prompt at once, decode one token at a time | A real run replayed with measured timings |

## Part 2: Why it's slow

| # | chapter | the one idea | primary figure |
|---|---|---|---|
| 9 | Memory or maths | Work waits on the truck (memory) or the factory (compute); decode does 0.5 operations per byte, so it waits on memory | Truck → factory toy, slider for work per byte, set to the tab's measured roofs |
| 10 | The speed limit | Decode can't beat memory bandwidth ÷ bytes per token (120 GB/s ÷ 495 MB = 243 tok/s) | Calculator: model × number format × machine → ceiling, with our measured result against it |
| 11 | The starting line | The first correct engine ran at 0.37 tok/s, 0.15% of the limit, because it re-reads the whole text for every token | Rows computed per token, growing with the text; the climb chart starts here |

## Part 3: Making it fast (the engine's optimisations)

Each chapter: **the bottleneck → the idea → the result**, and the same climb chart (decode tokens per second, log
scale, with the speed limit) gains one bar. Details of individual stages are folded.

| # | milestone | the bottleneck → idea | result (measured, M4) | primary figure |
|---|---|---|---|---|
| 12 | The KV cache | recomputing old tokens → keep their keys and values | 0.37 → 19 tok/s (51×) | Work per token with the cache drawn (computed = ember, read from cache = blue, the cache as a growing strip of pages with its size); climb |
| 13 | Faster arithmetic | one long chain of adds → 16 independent accumulators, SIMD | 19 → 100 tok/s; now memory-bound | Checkout lanes: 1 lane vs 16; climb |
| 14 | Many cores | one core can't drain memory → threads (and stop them sleeping) | 100 → 146 tok/s decode, prefill 2,415 tok/s | Threads slider over measured bandwidth per thread count |
| 15 | Smaller numbers | bytes are the limit → f16, 8-bit and 4-bit weights | 146 → 497 tok/s (q4) / 425 (q8, near-lossless) | Quantize a real block of 32 weights at 8/4/2 bits; climb |
| 16 | The GPU | CPU-shaped code and waiting → GPU-shaped kernels, stop waiting | 20 → 817 tok/s decode, 6,800 tok/s prefill (quiet machine) | "Wait after every op" vs "wait once" timeline; climb |
| 17 | Batching | one user leaves the GPU idle → serve many per weight read | 788 → 6,453 tok/s total (B=256, 25 each); median first token 3.5 s → 40 ms with continuous batching | Total vs per-user speed as the batch grows (the trade-off); static vs continuous timeline (requests as bars); paged attention (virtual-memory pages + block table) folded |
| 18 | Guess ahead (speculative decoding) | one user, big model, memory-bound → a small drafter guesses k tokens, the big model checks them in one pass | GPT-2 large 63 → 268 tok/s (4.25×) on licence text; prose 0.8–0.93 acceptance, code 0.65–0.88 | Playback of a real run (kept guesses, thrown-away guesses, the big model's own token); speed-up and acceptance by kind of text |
| 19 | Many GPUs (teaser) | the model doesn't fit → split it (pipeline, tensor, expert; MoE in one paragraph, linking the field guide's MoE lab), and pay for talking | (later post) | Ring all-reduce toy |

Numbers in part 3 are not typed into the pages: the GPU, batching, serving and speculative-decoding figures and prose
read `public/learn/data.json`, which `scripts/learn-data.py` builds from `bench/results.jsonl`, `batching.json`,
`serving.json`, `specdec.json` and `webgpu.json`. The landing page's "N× faster" is the best single-user decode over
the first version (817 / 0.37 ≈ 2,200×), computed from the same file. Batching's total is a different measure and is
never used for that claim. Chapter 18 has no climb bar: its target model is GPT-2 large, not the 124M model on the climb.

## The Lab (/lab, "part 4")

The original demo page, re-framed as "the engine room". Engine card + the figures that earn their place, each with a
plain intro that links back to the chapter it extends, a "what you're looking at" caption and sensible defaults:
Run it · Inside one token · Logit lens · Attention (144 heads) · Speed of light · Roofline · KV race · The ladder.
The engine picker in "Run it" lists every stage by family with a plain name and one line on what changed, and two
speed columns with sticky headers: **Mac** (native on the M4; for WebGPU stages, the same WebGPU code run natively on
wgpu-Metal) and **this browser** (your benchmark, else what generating here measured, else the recording in
`public/recorded/webgpu.json`, shown grey and italic with browser, date and conditions in its tooltip). Dead ends that are
kept runnable (webgpu-flashdec) carry a "tried, slower" tag.
Dropped from the page: time strip and byte budget (covered by chapters 4 and 10), stage race (the ladder says it more
simply).
