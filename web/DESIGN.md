# nanoinfer.live: design spec

One reader in mind: a strong systems engineer (caches, sharding, RPC, virtual memory, SIMD) who knows no ML. Lean on
what they know; explain ML ideas from scratch; never explain what a CPU is. The figure carries the idea, the prose
supports it. If a page would overwhelm a busy engineer, cut it.

## Colour means one thing, everywhere

| colour | token | means | examples |
|---|---|---|---|
| **ember** | `cc.c1`, `--m-accent` | **compute / work**, and the one thing to look at | the factory, FLOPs, an expert doing work, the climb's current bar, the answer |
| **blue** | `cc.c2`, `--nv-cool` | **memory**: bytes, bandwidth, stored state | the truck, weights being read, GB/s, the KV cache, HBM / DRAM |
| **olive** | `cc.c3` | **communication between devices** | NVLink, PCIe, network, all-reduce messages, the router's dispatch |
| **grey** | `pal.muted`, `pal.rule` | everything else: idle, waiting, earlier steps, context | idle lanes, earlier climb bars, unpicked experts |

- The chart palette (`chartColors()` in `src/learn/kit.tsx`) is validated with the dataviz checker against both
  surfaces. Never introduce a fourth hue; fold extras into grey.
- Sign (positive / negative numbers in an embedding or dot product) is a *diverging* job: ember = positive, blue =
  negative, always with a visible legend. This is the only place ember / blue don't mean compute / memory.
- Text never wears a series colour except the inline `.k` accent for the one number to remember, and the
  `<Swatch>` key words in captions ("**blue**: bytes read").

Known exceptions (left as is, each with an explicit legend): chapter 18's playback uses blue for the drafter's accepted
guesses vs ember for the big model's own tokens (draft vs target, not memory vs compute); chapter 17's serving timeline
uses grey for waiting. Fixed in this pass: chapter 17's per-user line (was blue, now grey), chapter 19's all-reduce
messages (were blue, now olive).

## Shapes mean one thing, everywhere

| shape | is | drawn as |
|---|---|---|
| token | a unit of text | rounded chip (`<Chip>`, `.lr-chip`), mono text inside |
| weights | a matrix of numbers | a rectangle with a faint grid, blue outline (it lives in memory) |
| KV cache | per-request state | a stack of small square pages, blue fill |
| device | a CPU, GPU or server | a rounded box with a small label header (`GPU 1`) |
| link | interconnect | an olive line; thickness ∝ bandwidth where it matters |
| request | one user's work over time | a horizontal bar on a time axis |
| expert / core / lane | a unit of compute | a small square; ember when working, grey when idle |

## Figure template

Every figure is `<Fig n title prov caption tryit>`:

1. **Header**: `fig N.M` in accent, a title that states the point (not "Chart of X"), provenance chip on the right
   (toy, diagram, weights, recorded, measured, live, or **external** for spec-sheet numbers, with a source line).
2. **One focal point.** The thing the reader should look at first is the only ember element. Controls sit in one row
   above the plot. At most one stat row (≤ 3 tiles).
3. **Caption**: "what you're looking at" (encodings, colours with swatches, what to notice) and "try" (one concrete
   action). Two to four sentences each.

Chapter rhythm: short paragraph (≤ 4 lines at 40rem) → figure → short paragraph → `<Idea>` (the one sentence) →
optional `<Deeper>` folds → **check yourself** (2–3 intuition questions, reveal the why).

## Typography and layout

- JetBrains Mono throughout (the site's voice). Scale: h1 `clamp(1.9rem, 5vw, 2.8rem)` / 700 / −0.045em; h2 1.05rem;
  body 0.9375–1rem at 1.8 line-height; captions 0.78rem; labels and chips 0.66–0.74rem uppercase tracking 0.07em for
  kickers only.
- One column. Prose 40rem (`--lr-prose`), figures 52rem (`--lr-fig`). No side-by-side figures. On phones: 16px gutters,
  figures go edge-to-edge-ish, tables become stacked cards, nothing scrolls horizontally except an explicit
  `.lr-math` block.
- Whitespace is structure: 3em before an h2, 2.4rem around figures. Paragraphs short; one idea per screen.

## Motion

- Motion explains a process (data moving, a router dispatching, a ring passing chunks); never decoration.
- Ease `--lr-ease`, 150–400 ms for UI, looping process animations ≤ 1 cycle per second and pausable.
- Everything respects `prefers-reduced-motion` (animations off; the figure still reads as a still).

## Analogies

When a term has a distributed-systems twin, say so in one line ("the KV cache is per-session state") and say where the
analogy breaks. The Rosetta stone (chapter 0.4) is the canonical list; the field guide links to it.

## Numbers

Our numbers come from `docs/LOG.md`, `bench/*.json` and `public/learn/data.json`, labelled *measured*. Anything from a
spec sheet or paper is labelled *external* and names its source (e.g. "H100 SXM HBM3 3.35 TB/s, NVIDIA datasheet").
Back-of-envelope figures say they are estimates and show the arithmetic.
