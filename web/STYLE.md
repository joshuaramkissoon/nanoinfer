# The Climb: copy style

How the words on the site are written. `DESIGN.md` covers how they look. Rules are numbered so a review can cite
them ("R5"). The source audit is `docs/review/02-copy.md`; `node scripts/copystats.mjs --long` checks the numeric
limits.

## Who you write for

An engineer who knows systems (caches, sharding, RPC, virtual memory, SIMD) and doesn't know ML. Address them as
**you**. Use **we** only for the project and its measurements ("we measured", "our M4"), never for the reader and
author together ("You now have the pieces", not "We now have the pieces").

Tone: a colleague at a whiteboard. Plain, specific, a little dry. Let the numbers do the impressing.

## Rules

| # | Rule |
|---|---|
| R1 | Second person for the reader; "we" only for the project. |
| R2 | Active voice. Passive only when the actor is unknown or doesn't matter ("the weights are frozen"). |
| R3 | Present tense for how things work ("decode reads every weight"). Past tense only for the climb's history ("the first version ran at 0.37 tok/s"). Never both in one sentence. No future tense for how the site behaves. |
| R4 | Body prose: aim for 20 words or fewer per sentence; 30 is the hard cap. Try lines, UI strings and check-yourself options: 20 or fewer. Glossary one-liners: 25 or fewer. |
| R5 | At most **one colon per sentence**, and only to introduce a list, an example or a result. Never chain "X: Y: Z". |
| R6 | ", not " pivots ("bytes, not maths") only against a misconception the reader really holds. Budget: one per chapter. |
| R7 | No rhetorical questions in body prose ("Or can they?"). Questions belong in chapter `q` lines, `Deeper` titles and check-yourself. |
| R8 | No throat-clearing or zingers. Banned: "Here is…", "Here's why that matters", "Time to…", "So far…", "That's it.", "Hold on to that", "That's the frame", "If you squint", "Nobody told…". End a paragraph on the fact. The `<Idea>` line is the summary, so don't write a second one. |
| R9 | Headings in sentence case, saying what the section shows ("The result: 51× faster"), not a bare template word. No question headings in body. |
| R10 | Define every abbreviation on first use in each chapter (TTFT, TPOT, HBM, SIMD, MoE, GQA, FFN). |
| R11 | **British spelling** (colour, grey, maths, optimise, quantisation, datacentre, licence, neighbour). A documented exception to Google's US rule (DECISIONS.md #2). Code identifiers, flags and product names keep their own spelling. |
| R12 | Curly quotes in prose (“ ” ’). Token strings go through `<Tk>`, never inside straight quotes. |
| R13 | Instructions work on any device: "select", "choose", "press", "drag (or use the arrow keys)". Never "hover" alone. |
| R14 | Never colour alone. Name the colour **and** the mark ("ember bars", "blue dots") with a `<Swatch>`. Only the palette names: ember, blue, olive, grey. |
| R15 | Link text says where it goes ("the field guide entry on GQA"), never "here" or a bare URL. |
| R16 | Every speed claim carries its conditions the first time it appears in a chapter: machine, number format, one user or many, and (for GPU decode) pipelined or streamed. |
| R17 | Don't pre-announce posts, features or "later" stages. |
| R18 | Hype words are out: huge, massive, enormous, remarkable, perfect, magic, truly, surprisingly, every serious engine. "just", "really", "actually", "simply" only when they contrast with a stated belief. |
| R19 | One metaphor per concept, reused the same way everywhere (see below). Never stack two. |
| R20 | No ambiguous phrasal verbs ("ride along", "creeps in", "catches up"). Say what happens. |

## Controlled vocabulary

Use the first column. The third column must not appear in reader copy (field guide `aka` lines and search aliases
are exempt).

| Concept | Use | Don't use |
|---|---|---|
| The unit the model reads and writes | **token** | *word* as a unit of work (after chapter 2), *piece*, *symbol*, *character* |
| Learned numbers | **weights** ("124 million weights") | *parameters* (except a model-size quote), *numbers* once weights are defined |
| One run of the model | **forward pass**, then **pass** | *a forward* (noun), *a trip through the model* |
| One server iteration | **step** | *iteration*, *tick* |
| An engine version in the climb | **stage**; the six course bars are **milestones** | *rung*, *version*, *step of the climb* |
| Rate | **tok/s** in figures, labels and tables; **tokens per second** in prose; always "per user" or "total" when both exist | *tokens a second*, *words a second* |
| Bandwidth ÷ bytes per token | **speed limit** | *ceiling*, *speed of light*, *limit line*, *yardstick*. *Roof* only inside the roofline figure |
| Bytes per second | **memory bandwidth**, then **bandwidth** | *memory speed*, *read speed* |
| The arithmetic side | **arithmetic** (prose), **compute-bound**, **FLOPs** (counts) | *maths* as a noun, except the chapter title "Memory or maths" |
| GPU execution units | **lane** (hardware), **GPU thread** (software) | *cores* for lanes; "10 GPU cores" only as Apple's spec unit |
| The last matrix | **LM head** ("the output layer that scores every token" on first use) | *scoring matrix*, *lm head* |
| Feed-forward | **feed-forward network (FFN)** | *MLP*, except once as "also called the MLP" |
| Key/value store | **KV cache** | *the cache* alone when a CPU or browser cache is nearby |
| Paged KV unit | **page** (16 tokens) and **block table** | *block* for a page |
| Number formats, prose | **32-bit, 16-bit, 8-bit, 4-bit** | *fp32, int4, f32* in running prose |
| Number formats, labels | **f32, f16, bf16, Q8_0, Q4_0** (Q8, Q4 when short) | *fp32, int8, int4, q4_0* |
| Spec decoding | **drafter**, **big model**, **check** | *guesser*, *checker*, *verify* |
| Concurrency | **user** for intuition; **request** for server mechanics | *stream*, *seat* |
| Machine | **the M4 Mac mini** on first use per chapter, then **the M4** | *our Mac*, *the Mac* |
| Model | **GPT-2 small** | *GPT-2 124M* in prose (credits only) |
| Datacentre GPU | **H100** when specific, **datacentre GPU** when generic | *datacenter*, *server GPU* |
| Site sections | **the course**, **the lab**, **the field guide** (lowercase in prose) | *Foundations*, *the demo page*, *the engine room* |
| Colours | **ember** = compute, **blue** = memory, **olive** = communication, **grey** = idle/context | *orange*, *dark bars*, *gray* |

Words with two meanings get qualified every time: *attention head* / *LM head*; *KV cache* / *CPU cache* / *browser
cache*; *CPU thread* / *GPU thread*; *layer* for a transformer block, *block of 32 weights* for quantisation, *page*
for KV; *pipeline stage* in full when it isn't an engine stage.

### Metaphors (R19)

| Concept | Metaphor | Where |
|---|---|---|
| Memory bandwidth vs arithmetic | **the truck** (bytes delivered) and **the factory** (arithmetic) | chapter 9, reused in 14 and 17 |
| The residual stream | **the notebook** | chapter 6 |
| Paged KV cache | **pages** of virtual memory | chapter 17 |
| All-reduce | **a ring of GPUs** passing chunks | chapter 19 |

Retired: the pantry door, checkout lanes, the paint roller, chefs, dishes round a table, a worker carrying a
delivery, seats.

## References

- Chapters: **chapter 5**, **chapter 0.2** (no zero padding). "Chapter 5" only at the start of a sentence.
- Figures: **fig 12.1** in the course (no full stop). From the course into the lab: **lab fig 3**. Inside the lab:
  **fig 3**.
- Layers are numbered **1–12** wherever a reader sees them.

## Numbers and units

- Numerals for every technical quantity, under 10 too: "4 cores", "12 layers", "8 bits".
- A space between number and unit (`64 GB`, `120 GB/s`, `130 µs`); none for `%`, `×` or `$`.
- Commas in four or more digits: 50,257; 1,456 (not in years or code).
- Ranges: en dash, no spaces, in labels ("98–101 tok/s"); "to" in prose ("4 to 6 threads").
- `~` in labels, "about" in prose, `≈` only in equations. `×` for factors and dimensions, never `x`. `÷` in prose
  equations.
- Bandwidth: GB/s below 1,000, TB/s above ("3.35 TB/s"). Links say "each way" or "total".
- Arrows (`→`) mean "went from → to", in results only.

## Figures

Each `<Fig>` has a caption paragraph, a `→ Try` line and a sources footnote (`DESIGN.md`, figure template).

**Caption**: 60 words or fewer, 4 sentences or fewer.
1. What the figure is, and its data source if the provenance tag doesn't say.
2. The axes ("Across: … Up: …", "log scale" where it applies).
3. The encodings: colour and mark, each with a `<Swatch>`.
4. Optionally, one caveat that changes how to read it.

No instructions, results or morals in the caption. Mechanism belongs in the body text.

**Try**: 35 words or fewer, 2 sentences or fewer. Start with a device-neutral imperative (Choose, Drag, Switch,
Press, Step through, Select). Then say what to look for ("Notice which bar moves"), **never the number or result the
reader will see**.

**Sources**: noun phrases only; estimates marked "(estimate)".

## Honest claims

These are the numbers the site may quote, with the conditions each one needs. Sources: `README.md`, `docs/LOG.md`
("Head-to-head", "Metal f32 on the pipelined path, and what streaming costs") and `bench/headtohead.json`.

- **The headline metric is % of the speed limit** (a stage's tok/s ÷ its own speed limit), per DECISIONS.md #9. The
  climb goes from 0.15% (0.37 tok/s, a deliberately naive first version) to about 60% (817 tok/s, 4-bit on the M4's
  GPU). Any "N× faster" names the naive baseline in the same sentence.
- **GPU decode against MLX and llama.cpp.** The 760–817 tok/s Q4 numbers are *pipelined*: every token's work is
  queued up front and the CPU never waits. In the 18 Oct head-to-head (pipelined), nanoinfer led at f16, Q8 and Q4
  (787 vs MLX 694 and llama.cpp 622 at Q4). A chat server reads each token back before it can encode the next. With
  that per-token synchronisation (what llama.cpp's benchmark does) nanoinfer is **slightly behind at every format**:
  Q4 593 vs llama.cpp 622 and MLX 694; Q8 486 vs 502; f16 305 vs 333. Keeping one token in flight (commit t+1 before
  reading t) recovers it: Q4 760. Never write "fastest" or "beats llama.cpp/MLX" without saying which loop.
- **f32 on the GPU.** `metal-pipelined-f32` makes about 186 tok/s, level with llama.cpp's 192 (estimate, scaled to
  head-to-head conditions). The older 176 vs 192 gap is gone.
- **Prefill is a loss**: llama.cpp is about 2× faster at prefill on both GPU and CPU.
- **The browser.** The quantised CPU stages (Q8, Q4) are currently slow in wasm (25.7 and 34.6 tok/s on the M4,
  recorded) because the int8 dot falls back to scalar code off aarch64. Don't say 4-bit is the fastest option in the
  browser. Quote the measured numbers from `public/recorded/` or say nothing about their rank.
