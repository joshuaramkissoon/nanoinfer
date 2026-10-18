# 03 · Visual design review

Scope: nanoinfer.live as served on :5199 (`/`, `/learn/*`, `/learn/field-guide`, `/lab`), judged against `web/DESIGN.md`
and the bar Josh set: Ciechanowski, Distill, Stripe docs, Linear, Vercel, *How to Scale Your Model*, and jkmr's own
minimal theme. Viewed in one isolated Chrome session at 1280×860 dark and light, 390×844 @2x mobile, and with
`prefers-reduced-motion: reduce`. Read-only: no source edits.

Screenshots are in `web/screenshots/designreview-*.png` (gitignored). They're cited below by their suffix, e.g.
`[0.2-fig1]` = `designreview-0.2-fig1.png`.

Measured, not guessed: computed styles, font-size census, contrast ratios (WCAG formula on the real tokens), rAF counts,
focus styles, and the dataviz palette validator. The numbers are in the relevant sections.

---

## 0. The verdict in five lines

1. **The bones are good.** The palette is restrained and semantic, the charts mostly follow dataviz rules, the landing
   page is calm, and several figures are genuinely excellent (17.1, 17.3, 0.2.2, 9.2, the lab's 144-head grid).
2. **What reads as AI-made is the furniture around the content, not the content itself.** Every figure, fold, quiz option,
   prev/next link and glossary term sits in the same 1px rounded bordered box. Every caption opens with the same
   uppercase ember kicker. Every page headline uses the same "plain + *ember italic*" split. Pills are everywhere.
3. **All-monospace long-form is the biggest readability cost.** It's 71 mono characters a line on desktop, 37 on a phone,
   prose in a 15px mono, and **25 different font sizes on a single chapter page**. Mono is the brand. It shouldn't be the
   body face for 4,000-word chapters.
4. **Figures are squeezed.** DESIGN.md says figures are 52rem, but every course figure renders at **640px, the same width
   as the prose**. Inside a padded card that leaves ~600px for a plot, so charts are cramped and captions are tall.
5. **Accessibility has real gaps.** Muted text fails AA in both themes, light-mode ember kickers fail, JS animations ignore
   reduced motion (measured: fig 9.1 runs at 60 fps with `reduce` set), and there's no skip link past 29 header and rail
   tab stops.

---

## 1. AI-slop tells

These are ordered by how loudly each one says "template". Each has a route, evidence and a fix.

| # | Tell | Where | Evidence | Fix |
|---|---|---|---|---|
| 1 | **Box everything.** Identical 1px `--m-rule` border, 12px radius, panel fill | every `.lr-fig`, `.lr-deeper`, `.lr-check`, `.lr-q1-opt` (8 per chapter), prev/next, the lab engine bar, field-guide cards (53) | On `/learn/what-is-inference` there are 22 bordered blocks wider than 200px. Radii in use: 12, 8, 7, 6, 0 and 50%. `[0.1-fig1] [09-fig2] [guide-cards]` | Keep a frame only where it *does* something: interactive controls, and maybe the figure plot area. Folds become hairline rows (they nearly are already). Quiz options and prev/next lose their boxes. One radius token (6px) plus pills (999px), nothing else. |
| 2 | **Uppercase ember kickers as a labelling system** | Every caption: `WHAT YOU'RE LOOKING AT`, `TRY`, `SOURCES`. Also `YOU KNOW THIS AS`, `THE IDEA`, `CHECK YOURSELF`, `ENGINE VERSION` | 0.1 alone has 3× WHAT YOU'RE LOOKING AT, 2× TRY, plus THE IDEA, YOU KNOW THIS AS and CHECK YOURSELF. That's 10.5px uppercase tracked ember, which is also 4.42:1 in light mode (fails). `[0.2-fig1]` | Write captions as prose. A caption that starts "Each square is one FLOP…" needs no label. Show "Try" as a sentence-case italic line with a leading `→`. Fold sources into a footnote line. DESIGN.md's "kickers only" rule should mean once per page (the chapter kicker), not seven times. |
| 3 | **Two-tone headline formula**: plain words + *italic ember* tail | `/` "…next word, *and how fast can one computer do it?*"; `/learn/field-guide` "The field guide: *what people mean*."; `/lab` "The engine room: everything, *running in this tab*." | Same device on all three top-level pages. `[landing-dark] [guide-top] [lab-top]` | Keep it on the landing only, where it earns its place. The guide and lab get plain titles. |
| 4 | **Pill overuse** | Constraint map (fig 0.2.4 and the field guide): every technique is a tinted outlined pill with a `ch` / `guide` micro-suffix. Also segmented controls, provenance chips, lab "run live" pill | The field-guide map is a wall of 28 pills in 3 border tints. `[0.2-fig4] [guide-top]` | Use text links separated by ` · `. Mark the destination with a trailing `↗` for the guide, or show nothing (the hover title already says it). Keep pills for *tokens* only, since DESIGN.md gives the rounded chip that meaning. Right now the chip means "token" and "link" and "toggle", so it means nothing. |
| 5 | **Coloured status dots as decoration** | Field-guide cards (one dot per term), constraint rows, provenance chips (`● diagram`, `○ toy example`, `◌ estimate`), lab status | The colour of a guide card's dot is never explained on that page. `[guide-cards]` | Drop the dots on cards, since the group heading already classifies. Provenance becomes a plain muted word, with no glyph. |
| 6 | **Filler stat tiles** | Lab fig 01: six tiles (decode speed, last token, context, rows computed, kv cache, generated). Big coloured numbers in 0.2.1, 0.2.3, 0.3.1 | DESIGN.md says ≤3. Lab has 6, with `95.0` in ember. `[lab-fig1] [0.2-fig1]` | ≤3 tiles, numbers in `--m-fg`. Put one ember number per figure at most, and only when it's the answer. |
| 7 | **Identical card grid for reference content** | `/learn/field-guide`: 53 bordered cards in a 2-col grid, ragged heights | `[guide-cards]` | Make it a dictionary (§3.6). Reference content reads best as a definition list: Stripe's glossary, HTSYM's appendix. |
| 8 | **Quiz-as-SaaS-form** | Check yourself: lettered a/b/c options in bordered full-width buttons | `[09-fig2]` | Plain list rows: an empty ring and the text, hairline separators, with the reasoning revealed inline beneath. It should look like a margin exercise in a textbook, not a Typeform. |
| 9 | **Left-border callout** | `THE IDEA`: 2px ember left border + kicker | Every chapter. `[0.2-fig4]` (bottom) | Drop both the border and the kicker. Set the idea sentence one step larger (18–19px) in `--m-fg`, with an ember `¶` or `—` hanging in the left margin. That's the Ciechanowski/Distill move: emphasis through type, not chrome. |
| 10 | **The rail is 23 hollow circles** | Desktop chapters, left edge | Only the current chapter has a label. Seen and unseen differ by fill alone, at 7px. `[0.1-fig1] [rail-open]` | Show the chapter numbers (`0.1 … 19`) in muted mono instead of circles, with the current one in fg and a 2px ember tick. That's numbers you can scan, not a loading indicator. The open state is fine. |

Not slop, and worth keeping: the warm charcoal and paper surfaces, the single accent, hairline rules on the index, the
recorded-demo card on the landing, the 2px progress bar, the ember/blue/olive semantics, and the honest provenance labels
(the *idea* of provenance is great; only the glyph styling is noisy).

---

## 2. System level

### 2.1 Typography

**Census.** This is from computed styles of visible text nodes on `/learn/what-is-inference` (dark, 1280).

- One family: JetBrains Mono, 193 of 194 text nodes.
- **25 distinct font sizes:** 9.92, 10, 10.5, 10.56, 11, 11.2, 11.52, 11.84, 12, 12.16, 12.48, 12.8, 13, 13.12, 13.125,
  13.44, 13.76, 14, 14.72, 15, 16.32, 16.8, 18.4, 20.8 and 44.8px. They come from `em` multipliers stacking on 15px, plus
  hard-coded SVG `fontSize` values (`10` ×42, `11` ×33, `11.5` ×17, `10.5` ×16, `9.5` ×6, `12.5`, `13`, `14`, `14.5`, `15`).
- Body: 15px / 27px (1.8) in `--m-fg`. The lede is 16.32px / 1.6 in `--m-sub`. DESIGN.md says 0.9375–1rem at 1.8, so the
  spec is met but the spec itself is the problem.
- Measure: **~71 mono characters a line** at 640px desktop and **~37 on a 390px phone** `[0.3-mobile-top]`. Mono
  characters are ~20% wider than proportional ones and carry no word-shape, so 71 mono feels like ~85 proportional.
  That's past the comfortable 60–75 band, and mobile is choppy.
- h2 is 16.8px bold: barely bigger than the 15px body. It relies on weight alone and reads as "bold paragraph"
  `[0.1-fig1]`.

**Is all-mono right for long-form?** For the chrome, labels, numbers, figures and code: yes. That's the voice, and it
matches jkmr. For 3,000–5,000 words of explanatory prose per chapter: no. Every reference on the bar (Ciechanowski,
Distill, HTSYM, Stripe docs) sets prose in a proportional face and keeps mono for the technical layer. Mono prose also
removes a tool the site needs: once *everything* is mono, code, numbers and token names (`␣dog`) can't stand apart from
the sentence around them.

**Proposal: pair, don't replace.** The recommended option:

| role | face | size / leading | notes |
|---|---|---|---|
| prose (chapters, guide definitions, captions) | **Source Serif 4** (variable, self-host woff2, opsz axis) *or* **Newsreader**. A grotesk alternative is Satoshi, which jkmr already ships | 18px / 1.6 desktop, 17px / 1.6 mobile; measure **34rem ≈ 66ch** | Gives warm, bookish texture against the charcoal. Serif italic carries the chapter question better than mono italic. |
| headings h1/h2/h3 | JetBrains Mono 700 | h1 40px / 1.1 / −0.035em; h2 21px / 1.3; h3 16px / 1.4 | Mono keeps the brand at every scan point. |
| UI, labels, axes, numbers, code, tokens | JetBrains Mono 400/700 | 12 / 13.5px | Tabular by nature, so numbers align for free. |

If Josh wants to stay all-mono (a legitimate brand call), the minimum fix:

- body 16px / 1.7, measure 36rem (~60ch)
- prose in `--m-fg`, never `--m-sub`
- h2 at 20px
- the 7-step scale below, so the page stops looking "slightly off" everywhere
- on mobile, 15px / 1.65 with 16px gutters (gets you to ~40ch)

**One scale, seven steps.** Ban arbitrary `em` and SVG sizes.

```
--t-xs   12px    axis ticks, provenance, kbd hints        (never smaller in HTML; SVG min 11px rendered)
--t-sm   13.5px  captions, controls, figure labels, stats labels
--t-base 18px    prose (serif)  · 16px if staying mono
--t-lg   21px    h2, the idea sentence
--t-xl   28px    stat numbers (max 3 per figure)
--t-2xl  40px    h1 (clamp 30→40)
--t-hero 52px    landing headline only
```

SVG text should use two sizes, 11 and 12.5 (via a `T.sm` / `T.xs` constant from `kit.tsx`), and figures should scale by
re-layout, not by viewBox shrink (REVIEW.md fix 3 already did this for two figures; make it the rule).

### 2.2 Spacing and rhythm

- Spacing is plausible but ad hoc: fig padding `16px 18.4px 17.6px`, controls gap ~12px, kicker-to-h1 ~20px, figure
  margins 2.4rem. The `.4`/`.6` px values show em-on-em drift.
- Proposal: a 4px base, `--s-1..8 = 4, 8, 12, 16, 24, 32, 48, 64`. Prose paragraph gap 0.9em (proportional to type).
  Use `--s-7` (48) before h2 and `--s-6` (32) after, `--s-7` above and below figures, and `--s-4` between the controls row
  and the plot.
- The chapter rhythm in DESIGN.md (short para → figure → para → idea) is right. Execution is hurt because every figure
  ends with a 150–250px caption block, so the visual unit is "card + wall of small text" rather than "picture". See 2.4.

### 2.3 Grid

- Chapters use one 640px column for both prose and figures. DESIGN.md asks for prose 40rem and figures 52rem, but no
  course figure was wider than 640 at 1280px (0.1, 0.2, 03, 09, 17 all measured `w: 640`). Field guide and lab use
  832–1024px. So **three pages, three grids**.
- Proposal, Distill-style:
  - `--col-text: 34rem` (serif prose)
  - `--col-fig: 52rem`: default for every figure, centred, breaking out of the text column
  - `--col-wide: 72rem`: lab grids and the 144-head map only
  - Text-only blocks (the idea, folds, check yourself) stay at `--col-text`.
  - Figure *captions* stay at `--col-text` inside the wider figure, so caption measure stays readable while the plot gets
    room.

### 2.4 Figure frame template

What you get today `[0.2-fig1]`:

```
┌───────────────────────────────────────────────┐  1px rule, 12px radius, panel bg
│ fig 0.2.1  Title that wraps onto a second…    │
│ ◌ estimate · arithmetic shown                 │  provenance drops BELOW title when long
│ [seg][seg][seg]  [seg][seg][seg]              │
│ 14 GB (blue)     14 GFLOP (ember)             │  coloured text
│ ...plot ~600px wide...                        │
│ WHAT YOU'RE LOOKING AT  5 lines of 12.5px…    │
│ TRY  3 lines…                                 │
│ SOURCES  4 lines…                             │  caption ≈ 35% of figure height
└───────────────────────────────────────────────┘
```

Proposed:

```
                                   (52rem, no box)
 ─────────────────────────────────────────────────────────────────  hairline, --m-rule
 Figure 2.1                                       estimate · M4 measured
 Every token reads the whole model, so bandwidth sets a speed limit   ← title, 16px mono 700, fg

 model  GPT-2 · [7B] · 70B · 405B      precision  fp32 · [bf16] · int8 · int4
                                                                     ← controls: one row, text-toggles
                                                                       (underline/fg for active), not boxes
   ... plot, full 52rem, own internal axes ...

   14 GB read / token      14 GFLOP / token      ← ≤3 stats, fg numbers, muted units

   The ceiling for each machine is its bandwidth ÷ bytes per token. Blue
   bars are memory-bound ceilings; ember ticks are what our engine measured.   ← caption: serif 15px,
   → Try 70B: one H100 can't hold it; switch to int4.                           34rem wide, sub colour
   Sources¹                                                                     ← footnote toggle
```

Rules:

- **No box.** A hairline above the figure and generous space below. The frame is whitespace.
- Header: `Figure N.M` in muted (not ember: ember is for the focal datum). Provenance on the same baseline, right-aligned,
  as text.
- Controls: one row. Segmented controls become text toggles, or keep a single light outline if a box is wanted, but at a
  ≥3:1 border.
- Caption: one paragraph (encodings + what to notice) plus one `→ Try` line. Sources move to a footnote or a `Sources`
  fold. Hard cap: 4 lines visible.
- The lab uses the **same** template. Today it has its own: `fig. 01` with a period, rule-above, sentence-case bold "What
  you're looking at:" `[lab-fig1]`.

### 2.5 Colour

**Semantics vs DESIGN.md.** Mostly honoured; REVIEW.md's fixes landed (fig 1.1 grey bars, 9.1 blue bytes, 17.4 blue
pages, 19.1 olive links). Remaining violations:

| where | issue |
|---|---|
| 0.2.1, 0.2.3, 0.3.1, 9 prose, 17.2, lab fig 01 | **Text wearing series colour**: `14 GB` blue, `14 GFLOP` / `$0.29` / `41` / `95.0` ember, "0.5 operations per byte" ember in prose, "7.9× the step before" ember, "compute: 97GFLOP/s" ember label. DESIGN.md allows `.k` for *the one number to remember*, but it's used 3–6× per chapter. |
| 3.1 | The legend words "positive" and "negative" are coloured text. Use swatches + fg text. |
| 0.1.1 | Ember appears on both "training" and the "inference" ticks. Two ember elements, so no single focal point. |
| 0.2.4 / field guide | Constraint "grey" means *time* (sequential decode) and also *traffic*. DESIGN.md says grey is "everything else", and here it's a fourth semantic. |
| lab fig 01 | Probability bars and ms/token bars are all ember. Fig 1.1 was fixed to grey-with-one-ember, but the lab wasn't. |
| lab fig 04 | The caption says "darker = more attention", but in dark mode brighter means more. |
| dark mode | Two embers: text accent `#e3915e` vs chart `c1` `#c96f38`, often side by side (stat number next to bar). Fine technically, but it reads as inconsistency. Pick one for marks and only use the lighter one for links. |

**Palette validation.** `chartColors()` passes all six dataviz checks in both modes: dark vs `#211f1d` (worst adjacent
ΔE 20.8 deutan) and light vs `#fbfaf7` (worst 19.5 protan). The chart hues are not the problem.

**Contrast**, measured with the WCAG formula on the real tokens:

| token | dark on bg `#1d1c1a` | dark on panel | light on bg `#f6f4ef` | light on code bg | needed | verdict |
|---|---|---|---|---|---|---|
| `--m-fg` | 12.2 | 11.8 | 15.3 | 13.8 | 4.5 | pass |
| `--m-sub` | 6.9 | 6.7 | 7.0 | 6.3 | 4.5 | pass |
| **`--m-muted`** | **3.99** | **3.85** | **3.29** | **2.96** | 4.5 (text is 10–12px) | **fail both modes** |
| `--m-accent` (text) | 6.9 | 6.6 | **4.42** | **3.97** | 4.5 | **fails light** (kickers, links, `fig N.M`) |
| `--nv-cool` (text) | 7.3 | 7.1 | 5.0 | 4.5 | 4.5 | pass (borderline on code bg) |
| `--m-rule` as control border (seg, quiz options, inputs) | 1.31 | 1.27 | 1.22 | — | 3.0 (1.4.11) | **fail** |

`--m-muted` is used on 49 of 194 text nodes on 0.1: axis ticks, provenance, quiz letters, rail numbers, figure sub-labels.
Proposed tokens:

```
dark  --m-muted   #7f7a71 → #96918a   (5.4:1 bg, 5.3 panel, 4.9 code)
light --m-muted   #8b867d → #6b665e   (5.2:1 bg, 4.7 code)
light --m-accent  #b8541f → #a3471a   (5.5:1 bg, 4.9 code); keep #b8541f as chart c1 only
--m-rule stays for decorative hairlines; add --m-line-ui for control borders:
dark ≈ #726d65, light ≈ #948f85  (≥3:1 on panel)
```

### 2.6 Navigation

- **Top bar**: good, quiet. On mobile the breadcrumb wraps to two lines (`nanoinfer.live / course /` + `0.3`) and
  "course" appears twice (crumb and nav) `[0.3-mobile-top]`. On ≤480px, drop the crumb to just `0.3`, or drop the nav
  "course".
- **Rail** (≥1200px): see slop #10. It's 23 tab stops before main content, with no skip link.
- **Prev/next**: two bordered cards; "← contents / Foundations" is weaker than "next · 0.2" `[0.1-dark]`. Make it one
  hairline row: `← 0.1 What inference is` on the left, `0.3 The hardware in one picture →` on the right, with the next
  title in fg at 16px and the chapter question beneath in serif. No boxes.
- **Contents**: the landing journey (numbered 0–4 timeline) is clean. "Part 4: The lab" is labelled as a part but isn't
  a course part. Use "Then: the lab →" with no numeral.
- **No in-chapter TOC**. Chapters run 3–5k px. On ≥1200px the rail could show the current chapter's h2s under its
  number, the way Stripe docs do. That's cheap and a real wayfinding win.

### 2.7 Information density

- Prose density is right: short paragraphs, one idea per screen.
- **Figure density is too high in two ways.** (a) Captions: 0.2.1's caption + sources is ~250px of 12.5px text,
  34% of the figure. 17.2's caption is 8 lines. Readers skip it, and it makes figures look like documentation panels.
  (b) Some figures stack sub-figures: 3.1 is a scatter + a 53-word picker + a barcode + 8 neighbour bars at 1,222px
  `[03-fig1a] [03-fig1b]`. 9.2 has two rows of segmented controls `[09-roofline]`.
- Lab top: six stat tiles + a sparkline-bar + a probability list + text, side by side `[lab-fig1]`. It's the lab, so
  density is allowed, but still give it one focal point.

### 2.8 Motion

- Good: motion explains a process (bytes on the road, packets on the bus, batches). REVIEW.md added off-screen pausing.
- **Reduced motion is not honoured by JS animations.** `useFrame` in `src/learn/kit.tsx` doesn't check
  `prefers-reduced-motion`. Only `WhatIsInference.tsx`, `Hardware.tsx` and `SpecDecode.tsx` read it. **Measured:** with
  `prefers-reduced-motion: reduce` emulated, fig 9.1 ran 120 rAF frames in 2 s. 11 `useFrame` call sites are affected.
- **No pause control** on looping figures. DESIGN.md says they're "pausable", and WCAG 2.2.2 requires it for >5 s motion.
- Hover micro-motion on index rows (padding-left shift) is fine. Keep UI transitions at 150–250ms.

### 2.9 Light, dark and mobile compared

- **Light** is the stronger theme: the paper surface plus ink makes mono prose feel like a typed manuscript
  `[0.2-light-top]`. Its weak spots are the ember kickers (4.42:1) and muted text (3.29:1).
- **Dark**: warm and handsome. Muted labels in figures get murky (`[0.1-dark]` fig 0.1.3's dimmed boxes are ~2:1 by eye:
  "model file on disk", "user A/B/C" at 9–10px are effectively invisible. Dim to 55% opacity of *fg*, not of muted.)
- **Mobile**: no horizontal overflow (good). But gutters are 12px (spec 16), figures sit as padded cards inside the
  gutter (≈330px of drawing width), and mono prose gives ~37ch lines. Figures should go full-bleed with only an inner
  16px pad, and lose the box.

---

## 3. Page by page

### 3.1 Landing `/` `[landing-dark]`

**Works.** One question, one live demo, one CTA, one map. The demo card (recorded/your prompt toggle, the prompt in
grey, the "97 tokens a second… 495 MB… 48 GB every second" sentence) is the best piece of the site: it is the thesis,
running.

**Doesn't.**

- The CTA cluster has three tiers of small text under the button ("recommended…", "Or jump to…", "23 short chapters ·
  select any text…"). Cut to one line.
- Journey: 23 full-width rows in a vertical timeline reads as a long list. The part descriptions are good; the chapter
  rows below each are redundant on the landing (the `/learn` index has them).
- The `the journey` uppercase kicker is the same kicker system again.

**Proposal.**

```
an inference engine written from scratch · and a course in how it works      (muted, not ember)

What happens when a model writes the next word,
and how fast can one computer do it?                                          (ember italic: keep, here only)

[ demo card: unchanged ]

Start here: What inference is →          or jump to part 1 · field guide

────────────────────────────────────────────────────────────
0  Inference for systems engineers        4 chapters · start here
   What a model physically is, why every word is expensive…
1  How a model writes text                8 chapters
2  Why it's slow                          3 chapters
3  Making it fast                         8 chapters · 0.37 → 6,453 tok/s
   then: the lab →
```

Parts as four rows with a chapter count, expandable or linking to `/learn`. Add a tiny inline sparkline of the climb
next to part 3. That's the one chart that sells the whole site.

### 3.2 A Part 0 chapter: 0.1 *What inference is* and 0.2 `[0.1-dark] [0.1-fig1] [0.2-fig1..4]`

**Works.** The systems framing ("search index vs queries", "stateful service with four jobs"), the pseudo-code block,
fig 0.2.2's single ember square, the stepper in 0.1.3.

**Doesn't.**

- Fig 0.1.1: the inference ticks clip at both ends of their row (half-squares at x≈664 and x≈930), and provenance wraps
  under the title.
- Fig 0.1.3: unselected elements are dimmed to unreadable, and the dashed "tokens stream back" line at the bottom is
  noise.
- Fig 0.2.1: log-scale *bars* (§4). The x-axis has no `0.1` baseline label, and there's an ~80px dead gap between the
  plot end (≈785) and the value column (≈850).
- Fig 0.2.4: a pill wall inside an accordion inside a card.
- The pseudo-code block's background bleeds 4px past the prose column on the left (`[0.1-dark]`). Align it with the
  text, or give it a real gutter.
- Check-yourself: 8 boxed options, plus a boxed prev/next under it, plus a boxed fold above it. That's three box styles
  in one screen.

**Proposal.** De-box (2.4). Figures at 52rem. 0.2.1 becomes a dot plot on a log axis with labelled decades and the
measured tick as an ember ring (§4). 0.2.4 becomes six hairline rows with techniques as ` · `-separated links.

### 3.3 Ch 03 *Embeddings* (Part 1) `[03-fig1a] [03-fig1b]`

**Doesn't.**

- The scatter uses ~40% of its area. Clusters are pinned to the top edge, and the middle is empty except for two stray
  points.
- The word picker duplicates the scatter. "Canada Brazil" wraps under the *label* column, breaking the grid.
- The barcode legend words are coloured.
- The neighbour bars sit on grey tracks (progress-bar look).
- The prose uses straight quotes with leading spaces (`" Monday"'s`). The token-space convention (`␣Monday`) exists, so
  use it in prose too.

**Proposal.** Split into two figures.

- **3.1a "Words as points"**: rank-normalised scatter filling a 52rem × 360px plot, click a point to select, cluster
  labels as direct annotations, and a footnote saying "2-D projection; axes have no meaning".
- **3.1b "dog, as 768 numbers"**: the barcode (with a diverging legend of swatches + fg text) and the top 5 neighbours
  as a dot plot on a 0–1 axis, no tracks.
- Drop the 53-word picker; keep a 6-chip group selector at most.

### 3.4 Ch 09 *Memory or maths* (Part 2) `[09-fig1] [09-roofline]`

**Works.** The single most useful chapter. The truck/factory toy, decode/prefill presets, the roofline fold with 4-step
narration.

**Doesn't.**

- Fig 9.1: the "processor" box has two small rectangles on top (chimneys?) that read as rendering artefacts. "memory busy"
  and "maths busy" are progress bars on tracks.
- The verdict line "→ memory-bound: …" is muted, though it's the takeaway.
- Fig 9.2:
  - y ticks mix formats (`1.0`, `10`, `100`, `1.0k`, `10k`)
  - "GFLOP/s ↑" collides with the top gridline
  - the "decode" label sits on the ridge dashed line
  - "compute: 97GFLOP/s" is missing a space and is coloured text
  - two rows of segmented controls

**Proposal.**

- 9.1: drop the chimneys and render busy-ness as two thin bars without tracks, values right-aligned in fg. Make the
  verdict the figure's single fg sentence at 15px.
- 9.2: ticks `1 · 10 · 100 · 1k · 10k`, axis titles outside the plot, label offsets that avoid the ridge, and the
  step-through and device toggle merged into one row (device as a small text toggle on the right).

### 3.5 Ch 17 *Batching* (Part 3) `[17-fig1] [17-climb] [17-fig3]`

**Works.** 17.1 is the model chart of the site (legend + direct labels, crosshair at the selected batch, log y
explained). 17.3 (static vs continuous, small multiples on a shared time axis) is excellent.

**Doesn't.**

- 17.2 "The climb so far": log-scale bars from an arbitrary 0.1 baseline, so bar length means nothing. Value labels
  (`101`, `146`) collide with gridlines and with the dashed limit line. "limit, 32-bit: 243" collides with the top
  gridline. The batching bar is an aggregate (all users), plotted as if it were the same measure. The caption is 8 lines.
- 17.3 is 48 ember bars. The message is the *grey* (waiting), but ember shouts louder.

**Proposal.**

- 17.2 → a **dot/lollipop chart on a log axis**: grey dots for earlier steps, an ember dot for this chapter. The
  single-user limit is a labelled vertical rule with its label above the plot. The batching aggregate is a hollow ember
  ring with a "(all users)" annotation, so it's visibly a different measure. Value labels go in a right-aligned fg
  column, clear of the plot. The caption is 3 lines.
- 17.3: answering bars in a neutral (fg at 35%) and waiting in ember hatching, *or* keep the colours and add a bracket
  annotation on the median wait in each panel (3.5 s vs 40 ms) as the focal point.

### 3.6 Field guide `/learn/field-guide` `[guide-top] [guide-cards]`

**Doesn't.**

- 53 bordered cards with unexplained coloured dots, in a 2-col grid with ragged heights.
- The map is a pill wall.
- The filter input and group links are good.

**Proposal: a dictionary.**

```
Models & architecture          What's inside the file, and the design choices that change its cost.
─────────────────────────────────────────────────────────────────────────────
Parameters      weights · "7B"
                The learned numbers that make up the model; "7B" means seven
                billion of them.                                       ch 0.1 →
─────────────────────────────────────────────────────────────────────────────
Attention       self-attention
                The step where each token scores every earlier token…   ch 05 →
```

- Term in mono 700 in a 10rem left column, definition in serif, aka in muted mono, chapter link right-aligned.
- Hairlines between entries, no boxes, no dots.
- Open an entry to reveal *why it exists* and *where you'll see it* inline (keeps the current disclosure).
- The constraint map becomes six hairline rows: name, one sentence, techniques as ` · ` links.
- Group jumps become a sticky left index on ≥1200px.

### 3.7 Lab `/lab` `[lab-top] [lab-fig1] [lab-heads]`

**Works.** The 144-head grid is beautiful small multiples. The engine-version selector with recorded tok/s is
informative.

**Doesn't.**

- It has its own figure template (`fig. 01`, sentence-case bold caption labels).
- Six stat tiles.
- Ember probability and latency bars.
- Side-by-side figure + panel (allowed in the lab, but the panel's rotated 9px axis labels are illegible).
- The "darker = more attention" caption is wrong in dark mode.
- The engine status bar is a boxed row with a light filled button, a second visual language next to the outlined "run
  live in your tab" pill.

**Proposal.**

- Use the course figure template at `--col-wide`.
- Three tiles (decode speed, last token, KV cache).
- Probability bars grey with the picked one ember (same as fig 1.1).
- ms/token as a thin grey column chart with the latest bar ember and a y-axis (0 / max ms).
- One button style (the filled one) across the page.
- Heat-map captions should say "brighter/stronger = more attention", or better, ship a 5-step sequential legend.

---

## 4. Data-viz audit

Checked against the dataviz skill: form, colour job, marks, legend/labels, hover, a11y, anti-patterns. "Hover" is noted
only where I exercised it.

| fig | form | what's right | issues | fix |
|---|---|---|---|---|
| 0.1.1 write once / read forever | diagram | clear left→right story | two ember elements; ticks clip at the row edges; provenance wraps | training box neutral outline + ember label only; fit ticks to an integer count |
| 0.1.2 GPT-2 strip | proportional strip (treemap-1D) | proportional widths, drill-down | the selected block's ember outline is the only ember (good); sub-strip labels only appear on click | label the 4 big matrices inline when they're ≥ 60px |
| 0.1.3 server | stepped diagram | step control, analogies | dimmed elements are illegible (~2:1); 9–10px SVG text | dim to fg@45%, min 11px |
| 0.2.1 speed limit | **log bars** | good title-as-claim, measured ticks, "doesn't fit" handled | bars on a log scale (anti-pattern: length from an arbitrary baseline); `0.1` origin unlabelled; dead gap before the values; big coloured stat numbers | dot plot on a log axis with labelled decades; value labels in a right column; stats in fg |
| 0.2.2 idle maths | waffle / unit chart | one ember square = perfect focal point | grid fills only ~85% width; stat labels mixed case | fit the grid to width; stats fg |
| 0.2.3 cost vs batch | line, log-log | crosshair, "memory full" rule, log noted in caption | `$0.29` in ember; y title floats inside the plot | stat in fg; y title outside the axis |
| 0.2.4 constraint map | accordion list | one-open-at-a-time is right | pill wall, colour dots carry a 4th semantic (grey = time) | text links; dots only for the 3 DESIGN hues + none |
| 0.3.1 chip schematic | diagram | mobile re-layout works `[0.3-mobile-fig1]`, bus ∝ bandwidth | `41` stat in ember; 3 stats wrap to 2 rows on mobile | stats fg; 2 per row on mobile |
| 3.1 embeddings | scatter + barcode + bars | real data, exact neighbours | ~60% empty plot; no "axes are meaningless" note; coloured legend text; tracks behind bars; 1,222px tall | split (§3.3) |
| 9.1 truck/factory | animated diagram + 2 bars | presets, crossover concept | chimney artefacts; progress-bar tracks; runs under reduced motion | §3.4; honour `reduce` |
| 9.2 roofline | line on log-log | stepped narration, annotations, ridge marked | mixed tick formats; label collisions; coloured label text; 2 control rows | §3.4 |
| 17.1 batch trade-off | 2-series line, log y | legend **and** direct labels, crosshair, ember = total vs grey = per-user | none material; plot could use the full 52rem | n/a |
| 17.2 climb | **log bars** | grey-earlier / ember-current is right | log-bar anti-pattern; label/gridline collisions; mixed measures in one series; 8-line caption | lollipop, aggregate as a hollow ring (§3.5) |
| 17.3 static vs continuous | timeline small multiples | shared x, medians in the panel titles, hover per request | ember dominates while the message is grey | annotate the medians as the focal point |
| lab fig 01 | text + bars + tiles | per-token confidence underline is lovely | 6 tiles; all-ember bars; no y-axis on the ms bars | §3.7 |
| lab fig 04 | heat-map small multiples | 12×12 multiples, click to enlarge | caption polarity wrong in dark; no colour legend; rotated 9px labels | sequential legend; horizontal labels truncated with a title tooltip |

**Cross-cutting chart fixes**

- **Log scales:** every log chart should label decades with `1 · 10 · 100 · 1k`, include "log scale" in the axis title
  (not only in the caption), and never encode with bar length.
- **Gridlines:** hairline solid at `--m-rule`, drawn *under* marks. Value labels get a 2px surface-colour halo
  (`paint-order: stroke`) so they never collide with gridlines (17.2, 9.2).
- **Text never wears series colour** (dataviz non-negotiable and DESIGN.md). Audit `.k` usage to ≤1 per chapter.
- **Table view:** no chart offers one. Add a "data" toggle in the figure footer that swaps the plot for a `<table>`
  (dataviz a11y requirement; also great for the systems reader who wants the numbers).

---

## 5. Accessibility (WCAG 2.1 AA)

**Summary:** 4 major and 4 minor issues; nothing blocks use outright.

| # | Issue | WCAG | Severity | Evidence | Fix |
|---|---|---|---|---|---|
| 1 | Muted text below 4.5:1 in both modes | 1.4.3 | Major | 3.99 dark, 3.29 light, 2.96 on light code bg; 49 nodes on 0.1 | tokens in §2.5 |
| 2 | Light-mode ember text 4.42:1 (kickers, `fig N.M`, links, crumbs at 10–13px) | 1.4.3 | Major | §2.5 | `#a3471a` for text |
| 3 | JS animations ignore `prefers-reduced-motion`; no pause control | 2.3.3 (AAA) / 2.2.2 (A) | Major | fig 9.1: 120 frames / 2 s with `reduce` | `useFrame` reads `matchMedia('(prefers-reduced-motion: reduce)')` and renders one still frame; add a ⏸ text toggle in the figure header for any loop |
| 4 | Control borders (segmented, quiz options, filter input) 1.2–1.3:1 | 1.4.11 | Major | `--m-rule` | `--m-line-ui` ≥3:1, or rely on fill + text weight for state |
| 5 | No skip link; 6 header + 23 rail tab stops before content | 2.4.1 | Minor | focus walk on ch 09 | "Skip to chapter" link as the first focusable element; or `inert` the collapsed rail and open it via one button |
| 6 | SVG figures inconsistently labelled: about half the chapter SVGs have `role="img"`/`aria-label` (e.g. Arithmetic, Attention, BytesFlops, DotProduct, Gpu, Loop, NextToken, Quantization, Threads have none) | 1.1.1 | Minor | grep of `learn/ch/*.tsx` | `<Fig>` takes a required `alt` (one-sentence takeaway) rendered as `aria-label` on the plot wrapper; decorative SVG gets `aria-hidden` |
| 7 | Touch targets: seg buttons 31px tall, theme toggle 19×32, inline glossary links 14px | 2.5.5 (AAA) / 2.5.8 (2.2 AA 24px) | Minor | measured on 390px | 40px min height on touch (`@media (pointer: coarse)`) |
| 8 | Dimmed diagram labels (0.1.3) unreadable | 1.4.3 | Minor | `[0.1-dark]` | dim via opacity on fg, min 4.5:1 for text that's still meant to be read, or hide it |

Good: `lang="en"`; real `<figure>`/`<figcaption>`; segmented controls are `role="group"` + `aria-pressed`; focus rings
exist on header links, inputs, seg buttons, glossary terms and quiz options (2px ember outline). Folds and rail have
custom `:focus-visible` styles. No horizontal overflow at 390px. Chart palette is CVD-safe (validator).

---

## 6. Top 10 changes, prioritised

1. **Set chapter prose in a proportional face, on a 7-step scale.**
   - *Before:* 15px mono prose, ~71ch desktop / 37ch mobile, 25 font sizes per page, h2 barely bigger than the body.
   - *After:* 18px Source Serif 4 (or Newsreader) at 1.6, 34rem / ~66ch. Mono for headings, labels, numbers and code.
     Seven size tokens; SVG text at 11 / 12.5 only.
   - *Why:* it's the biggest readability and "designed, not generated" win.
2. **De-box the figure and widen it.**
   - *Before:* every figure is a 640px bordered 12px-radius card, as wide as the text.
   - *After:* 52rem, frameless, a hairline above, `Figure N.M` muted, provenance as right-aligned text, controls in one
     row. The lab uses the same template.
3. **Captions as prose, not labelled forms.**
   - *Before:* `WHAT YOU'RE LOOKING AT` / `TRY` / `SOURCES` uppercase ember kickers, 5–10 lines of 12.5px.
   - *After:* one caption paragraph (≤3 lines) at 34rem + a `→ Try …` line + a `Sources¹` footnote.
4. **Fix the tokens that fail contrast.** `--m-muted` (dark `#96918a`, light `#6b665e`), light text accent `#a3471a`,
   and a new `--m-line-ui` for control borders at ≥3:1.
5. **Honour reduced motion and add pause.**
   - *Before:* `useFrame` loops ignore `reduce` (60 fps measured).
   - *After:* a single still frame under `reduce`, and a ⏸/▶ text control on every looping figure.
6. **Stop drawing bars on log axes.** 0.2.1 and 17.2 become dot/lollipop charts with labelled decades. Value labels
   get halos so they never collide. Tick formats are consistent (9.2). The aggregate batching value is a distinct mark.
7. **Rebuild the field guide as a dictionary.** No cards, no dots, a term/definition grid with hairlines, and the
   constraint map as text rows with ` · ` links.
8. **Strip the remaining chrome.** The idea becomes a large-type sentence with a hanging ember mark (no border or
   kicker). Check-yourself options become plain rows. Prev/next becomes one hairline row. Radius is a single 6px
   token, plus pills only for tokens.
9. **Colour discipline in text.** Stat numbers and labels go in fg. At most one `.k` ember number per chapter. One ember
   focal element per figure (0.1.1, lab bars, 17.3 annotation). Legends use swatch + fg text.
10. **Mobile and wayfinding.**
    - Mobile: 16px gutters, full-bleed figures, a one-segment breadcrumb, 40px touch targets.
    - Desktop: the rail shows chapter numbers instead of empty circles, plus the current chapter's h2s.
    - A skip link.
    - Split fig 3.1 into two figures.

---

## 7. Proposed DESIGN.md changes (diff)

```diff
 ## Figure template

-Every figure is `<Fig n title prov caption tryit>`:
+Every figure is `<Fig n title prov alt caption tryit sources>`. No box: a hairline above, whitespace below.

-1. **Header**: `fig N.M` in accent, a title that states the point (not "Chart of X"), provenance chip on the right
-   (toy, diagram, weights, recorded, measured, live, or **external** for spec-sheet numbers, with a source line).
+1. **Header**: `Figure N.M` in muted mono, a title that states the point (not "Chart of X"), provenance as plain
+   muted text on the same baseline, right-aligned (toy, diagram, weights, recorded, measured, live, external). No
+   glyphs, no chip.
 2. **One focal point.** The thing the reader should look at first is the only ember element. Controls sit in one row
-   above the plot. At most one stat row (≤ 3 tiles).
+   above the plot. At most one stat row (≤ 3 tiles), numbers in fg, never in a series colour.
-3. **Caption**: "what you're looking at" (encodings, colours with swatches, what to notice) and "try" (one concrete
-   action). Two to four sentences each.
+3. **Caption**: one paragraph, ≤ 3 lines at the text measure: encodings (with swatches) and what to notice. Then one
+   `→ Try …` line. Sources go in a footnote toggle. No uppercase labels.
+4. **alt**: one sentence stating the takeaway, rendered as the plot's accessible name.
+5. **Data**: every chart with numbers offers a table view toggle.
+
+Frames (borders + panel fill) are reserved for interactive inputs. Folds, quiz options, prev/next, callouts and
+reference entries use hairlines and space, never boxes. One radius token: 6px (pills 999px, tokens only).

 ## Typography and layout

-- JetBrains Mono throughout (the site's voice). Scale: h1 `clamp(1.9rem, 5vw, 2.8rem)` / 700 / −0.045em; h2 1.05rem;
-  body 0.9375–1rem at 1.8 line-height; captions 0.78rem; labels and chips 0.66–0.74rem uppercase tracking 0.07em for
-  kickers only.
+- Two faces. **JetBrains Mono** is the voice: headings, UI, labels, axes, numbers, code, tokens. **Source Serif 4**
+  (self-hosted, variable) is the reading face: chapter prose, captions, guide definitions, the idea.
+- Seven sizes, no others: 12 / 13.5 / 18 (prose; 16 if mono) / 21 / 28 / 40 / 52px. SVG text: 11 and 12.5 only.
+  Prose 1.6 line-height; headings 1.1–1.3. Uppercase tracked text: the chapter kicker only, once per page.
+- Spacing: 4px base (4, 8, 12, 16, 24, 32, 48, 64). 48 before h2, 32 after; 48 around figures.

-- One column. Prose 40rem (`--lr-prose`), figures 52rem (`--lr-fig`). No side-by-side figures. On phones: 16px gutters,
-  figures go edge-to-edge-ish, tables become stacked cards, nothing scrolls horizontally except an explicit
-  `.lr-math` block.
+- One column. Prose 34rem (~66ch serif). Figures 52rem and they **must** break out of the prose column; captions stay
+  34rem inside them. Lab grids 72rem. No side-by-side figures in the course. On phones: 16px gutters, figures
+  full-bleed with a 16px inner pad, 40px touch targets, nothing scrolls horizontally except `.lr-math`.

 ## Colour means one thing, everywhere
 ...
+- Text tokens must pass AA: `--m-muted` dark `#96918a` / light `#6b665e`; light text accent `#a3471a` (chart c1 stays
+  `#b8541f`). Control borders use `--m-line-ui` (≥ 3:1); `--m-rule` is for decorative hairlines only.
+- Grey means "everything else". Do not give grey a fourth meaning (e.g. "time" in the constraint map).
+- Log axes: label decades (1 · 10 · 100 · 1k), say "log" in the axis title, and never encode with bar length.

 ## Motion

-- Everything respects `prefers-reduced-motion` (animations off; the figure still reads as a still).
+- Everything respects `prefers-reduced-motion`, including JS loops: `useFrame` renders one representative still
+  frame when it matches. Every looping figure has a visible pause/play control.
```

---

### Appendix: method

- Browser: one isolated chrome-devtools context. Computed-style census via an injected audit (font sizes, colours,
  radii, bordered elements, uppercase text). Contrast computed from tokens in `src/styles.css` with the WCAG
  relative-luminance formula. Palette validated with the dataviz skill's `validate_palette.js` against `#211f1d` and
  `#fbfaf7`. Reduced motion checked by wrapping `requestAnimationFrame` and counting frames over 2 s on ch 09.
- Pages walked: `/`, `/learn/what-is-inference`, `/learn/why-its-expensive`, `/learn/the-hardware` (mobile),
  `/learn/embeddings`, `/learn/memory-or-maths`, `/learn/batching`, `/learn/field-guide`, `/lab`. Light mode on 0.2;
  mobile at 390px on 0.3.
- Not covered: every chapter individually (2, 4–8, 10–16, 18–19 were not walked in this pass), hover tooltips on every
  chart, a VoiceOver pass.
