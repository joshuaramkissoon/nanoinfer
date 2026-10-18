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
/learn/<chapter>      18 chapters in 3 parts (below), prev/next, progress rail
/lab                  The engine room: the full live demo, curated and simplified
```

## Landing (/)

- The question: *What happens when a model writes the next word, and how fast can one computer do it?*
- One demo: type a prompt, GPT-2 continues it (recorded run until the reader loads the engine; then live, in the tab).
  Speed shown in plain words: "≈ 100 tokens a second: for every token, your computer reads all 500 MB of the model".
- The map: four parts as four cards, each with its chapters. A "start" button into chapter 1.

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
| 12 | The KV cache | recomputing old tokens → keep their keys and values | 0.37 → 19 tok/s (51×) | Naive vs cached work per token, animated; climb |
| 13 | Faster arithmetic | one long chain of adds → 16 independent accumulators, SIMD | 19 → 100 tok/s; now memory-bound | Checkout lanes: 1 lane vs 16; climb |
| 14 | Many cores | one core can't drain memory → threads (and stop them sleeping) | 100 → 146 tok/s decode, prefill 2,415 tok/s | Threads slider over measured bandwidth per thread count |
| 15 | Smaller numbers | bytes are the limit → f16, 8-bit and 4-bit weights | 146 → 497 tok/s (q4) / 425 (q8, near-lossless) | Quantize a real block of 32 weights at 8/4/2 bits; climb |
| 16 | The GPU | CPU-shaped code and waiting → GPU-shaped kernels, stop waiting | 14.6 → 641 tok/s decode, 8,900 tok/s prefill | "Wait after every op" vs "wait once" timeline; climb |
| 17 | Batching | one user leaves the GPU idle → serve many per weight read | 788 → 6,453 tok/s total (B=256), first token 4.4 s → 62 ms with continuous batching | Batch-size slider: total vs per-user speed |
| 18 | Many GPUs (teaser) | the model doesn't fit → split it, and pay for talking | (later post) | Ring all-reduce toy |

## Part 4: The Lab (/lab)

The original demo page, re-framed as "the engine room". Engine card + the figures that earn their place, each with a
plain intro that links back to the chapter it extends, a "what you're looking at" caption and sensible defaults:
Run it · Inside one token · Logit lens · Attention (144 heads) · Speed of light · Roofline · KV race · The ladder.
Dropped from the page: time strip and byte budget (covered by chapters 4 and 10), stage race (the ladder says it more
simply).
