# The Climb: design spec

The site is **The Climb**: an inference engine, climbing toward the speed limit. The repo, crate and engine keep the
name nanoinfer. Spelling is British.

One reader in mind: a strong systems engineer (caches, sharding, RPC, virtual memory, SIMD) who knows no ML. Lean on
what they know; explain ML ideas from scratch; never explain what a CPU is. The figure carries the idea, the prose
supports it. If a page would overwhelm a busy engineer, cut it.

The headline metric is **% of the speed limit** (a step's tok/s ÷ its own bandwidth ceiling). Name the baseline in
the same sentence as any "N× faster".

Tokens live in `src/styles.css` (`.minimal`); the course, landing and lab rules in `src/learn/learn.css` and
`src/learn/p0.css`; the figure template and motion hooks in `src/learn/kit.tsx`.

## Typography

- **Two faces.** JetBrains Mono is the voice: headings, UI, labels, axes, numbers, code, tokens. Source Serif 4
  (self-hosted variable woff2 in `public/fonts`, SIL OFL) is the reading face: chapter prose, captions, folds, field-guide
  definitions, the idea, check-yourself. `--f-mono`, `--f-serif`.
- **Seven sizes, no others.** Use the tokens, never a raw `rem`/`em`/`px`:

  | token | px | used for |
  |---|---|---|
  | `--t-xs` | 12 | axis ticks, provenance, kbd hints, rail numbers, stat labels |
  | `--t-sm` | 13.5 | controls, figure labels, UI, run-in labels |
  | `--t-md` | 15 | captions, folds, definitions, figure titles, inline code in prose |
  | `--t-base` | 18 | prose (serif), 1.6 line-height |
  | `--t-lg` | 21 | h2, the idea sentence, stat numbers |
  | `--t-xl` | 28 | a hero number |
  | `--t-2xl` | 40 | h1, clamped down to 30 on phones |

- **SVG text** uses `T.xs` (11) and `T.sm` (12.5) from `kit.tsx`, plus `T.num` (28) for a single hero number drawn in a
  figure. Canvas text is 11px. A label sized to a small drawn box may scale with its box if it still renders at ≥ 11px
  (fig 0.3.1's core labels).
- Headings are mono 700 with negative tracking; prose never wears mono except `code` and the `.k` accent.
- Uppercase tracked text: none. The page kicker (`0.2 · part 0 · …`) is sentence-case mono in muted.
- The "plain words + *ember italic* tail" headline is used once, on the landing. Every other title is plain.

## Layout and spacing

- One column. Prose `--lr-prose: 34rem` (~66ch of serif). Figures `--lr-fig: 52rem` and break out of the prose column;
  captions return to the prose column inside them. Lab figures and grids `--lr-wide: 64rem`. The landing's column is
  40rem with its lede kept at 34rem.
- Spacing on a 4px base: `--s-1…8` = 4, 8, 12, 16, 24, 32, 48, 64. 48 before an h2, 48 above figures, 32 after.
- Phones: 16px gutters; figures go full-bleed (the hairline runs edge to edge, content keeps a 16px pad); stats two to a
  row; 40px touch targets under `pointer: coarse`; the breadcrumb drops to one segment; nothing scrolls horizontally
  except `.lr-math`.

## Frames, radius, chrome

- **Frames (border + fill) are for inputs and buttons only.** Figures, folds, quiz options, prev/next, callouts and
  reference entries use hairlines (`--m-rule`) and whitespace. The landing's recorded demo and the lab's engine card are
  the two deliberate panels.
- One radius, `--r: 6px`. Pills (`--r-pill`) are for token chips only. Small marks keep 2–3px.
- **Segmented choices are text toggles** (`.lr-seg`, `.nv-seg`): the chosen one fg, bold, with a 2px fg underline; the
  rest secondary. Buttons: `.lr-btn` (1px `--m-line-ui` outline), `.lr-btn.solid` / `.nv-btn` (fg fill) for the one
  primary action.
- **The idea** (`<Idea>`): one sentence at `--t-lg` in fg, an ember dash hanging in the left margin, no box, no visible
  label (the label is the aside's accessible name).
- **Check yourself**: a mono h2, questions in serif 600, options as plain rows with an empty ring and hairlines between;
  the ring fills (✓ green / × red) and the reasoning appears inline under "Yes." / "Not quite.".
- **Prev/next**: one hairline row, previous on the left in secondary, next on the right in fg with its question in serif
  italic beneath.
- **The rail** (≥ 1200px): chapter numbers in muted mono, read ones in secondary, the current one in fg with a 2px ember
  tick and its title; hover or focus opens every title.
- A **skip link** ("Skip to content") is the first focusable element on every page; each page marks its main content
  `id="content"`.

## Colour means one thing, everywhere

| colour | token | means | examples |
|---|---|---|---|
| **ember** | `cc.c1` | **compute / work**, and the one thing to look at | the factory, FLOPs, an expert doing work, the climb's current dot |
| **blue** | `cc.c2`, `--nv-cool` | **memory**: bytes, bandwidth, stored state | the truck, weights being read, GB/s, the KV cache, HBM / DRAM |
| **olive** | `cc.c3` | **communication between devices** | NVLink, PCIe, network, all-reduce messages, the router's dispatch |
| **grey** | `pal.muted`, `pal.sub` | everything else: idle, waiting, earlier steps, context | idle lanes, earlier climb dots, unpicked experts |

- The chart palette (`chartColors()` in `src/learn/kit.tsx`) is validated with the dataviz checker against both
  surfaces. Never introduce a fourth hue; fold extras into grey. Grey never takes a fourth meaning (the constraint map's
  "sequential decode" row has no swatch rather than a grey one).
- Sign (positive / negative) is a *diverging* job: ember = positive, blue = negative, with a swatch legend.
- **Text never wears a series colour.** Stat numbers and labels are fg / muted; legends are a swatch plus fg text
  (`<Swatch>`). The one exception is the prose `.k` accent for the number to remember, at most one per chapter, in
  `--m-accent` (the text accent, not the chart ember).
- **One ember focal point per figure.** Earlier steps, unpicked bars and context go grey; when the message is the grey
  (fig 17.3's waiting), the ember steps back.

### Text and control tokens (WCAG 2.1 AA, measured)

| token | light | on bg | dark | on bg |
|---|---|---|---|---|
| `--m-fg` | `#1f1d1a` | 15.3 | `#dedad2` | 12.2 |
| `--m-sub` | `#57534c` | 7.0 | `#aaa59b` | 6.9 |
| `--m-muted` | `#6b665e` | 5.2 (4.7 on code bg) | `#96918a` | 5.4 (4.9 on code bg) |
| `--m-accent` (text) | `#a3471a` | 5.5 | `#e3915e` | 6.9 |
| `--m-line-ui` (control borders, 1.4.11) | `#8f8a80` | 3.1 | `#726d65` | 3.3 |

`--m-rule` is for decorative hairlines only. Chart marks use `chartColors()` (light ember `#b8541f`), never the text
accent.

## Figure template

Every figure is `<Fig n title prov alt caption tryit source>` (course) or `<Figure>` from `ui/Figure.tsx` (lab); both
render `FigFrame` from `kit.tsx`.

1. **Meta line**: `Figure N.M` in muted mono on the left; provenance as plain muted text on the right (toy example,
   diagram, real GPT-2 weights, recorded, measured, live, external, estimate; the lab's line follows the run's backend:
   WebGPU and the adapter, or the wasm build). A pause button or "run live" button may sit beside it.
2. **Title** below it, mono 700 at `--t-md`, stating the point, not "Chart of X".
3. **Controls**: one row above the plot.
4. **The plot**, full figure width. At most one stat row, ≤ 3 tiles, numbers in fg.
5. **Caption** in the text column, serif `--t-md`: one paragraph (encodings, colours named with their swatches, what to
   notice), then a `→ Try …` line, then the sources folded into a footnote. No uppercase labels.
6. **alt**: one sentence stating the takeaway. `FigFrame` gives every drawing SVG without focusable children
   `role="img"` and that label (or the title).

Chapter rhythm: short paragraph → figure → short paragraph → `<Idea>` → optional `<Deeper>` folds → check yourself.

## Charts

- **Log axes**: label decades `0.1 · 1 · 10 · 100 · 1k · 10k`, say "log scale" in the axis title, and never encode with
  bar length. Use dots or lollipops (fig 0.2.1, the climb, fig 9.3).
- **The climb** (`learn/Climb.tsx`): a lollipop per step, grey for earlier steps and ember for this chapter's; a short fg
  tick at each step's own speed limit; tok/s and % of that limit in a right-aligned column; the batching total as a
  hollow ring because it's a different measure (all users).
- Gridlines are hairlines drawn under the marks. Labels that can meet a gridline or a roof get a surface-coloured halo
  (`stroke={pal.bg}`, `paintOrder="stroke"`). Axis titles sit outside the plot, clear of the tick labels.
- Hover is never the only way in: every hover also works on click or tap.

## Motion

- Motion explains a process (data moving, a router dispatching, a ring passing chunks); never decoration. UI
  transitions 150–250ms on `--lr-ease`; looping process animations ≤ 1 cycle per second.
- **Every looping figure has a pause / play control** (`<PlayPause>` with `usePlaying()`), in its controls row or its meta
  line.
- **Reduced motion**: `usePlaying()` starts paused under `prefers-reduced-motion: reduce`, so nothing loops on its own,
  and `useFrame` advances a couple of seconds of simulated time once on mount so the figure shows a representative
  still. CSS animations and transitions are switched off in the same mode. Pressing play is the reader's choice.

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

## Analogies

When a term has a distributed-systems twin, say so in one line ("the KV cache is per-session state") and say where the
analogy breaks. The Rosetta stone (chapter 0.4) is the canonical list; the field guide links to it.

## The field guide

A dictionary, not a card grid: term (mono 700) and its aliases in a left column, one plain serif sentence beside it, the
chapter number on the right, hairlines between entries. An entry opens in place into *why it exists*, *where you'll
see it*, an optional interactive (MoE) and the chapter link. The constraint map sits on top as hairline rows with
techniques as plain links separated by middots.

## Numbers

Our numbers come from `docs/LOG.md`, `bench/*.json` and `public/learn/data.json`, labelled *measured*. Anything from a
spec sheet or paper is labelled *external* and names its source (e.g. "H100 SXM HBM3 3.35 TB/s, NVIDIA datasheet").
Back-of-envelope figures say they are estimates and show the arithmetic.
