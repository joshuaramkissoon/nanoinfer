# Copy review 02: the learning site and project docs

Scope: every reader-facing string in `web/src` (chapters in `src/learn/ch/**`, figure captions and try-its, the
glossary in `src/learn/kit.tsx`, field guide entries, `src/learn/checks.ts`, landing, journey map, course index, lab
panels and lab hero, the ask layer), plus `README.md`, `docs/WRITEUP.md` and `web/README.md`. Read-only audit: no
source file was changed.

Standards applied:

- **Anti-slop.** Filler, throat-clearing, hype, faux-profound one-liners, reflex triplets, "not X, Y" constructions,
  rhetorical openers, vague intensifiers, mixed metaphors and repeated sentence templates.
- **Google developer documentation style guide** (developers.google.com/style). The rules cited below were checked
  against the live pages: *Highlights* (second person, active voice, sentence case, descriptive link text, write
  accessibly, US spelling, conditions before instructions), *Present tense*, *Numbers* (numerals for technical
  quantities, commas in 4+ digit numbers, ranges), *Units of measurement* (space between number and unit, repeat the
  unit in a range, prefer "to"), *Dashes* (em dash with no spaces, no en dashes), *Abbreviations* (spell out on first
  use), *Link text*, *Future features* ("don't pre-announce").
- **ASD-STE100 Simplified Technical English.** One word, one meaning. Procedural sentences of 20 words or fewer,
  descriptive sentences of 25 words or fewer. One topic per sentence. Active voice. No ambiguous phrasal verbs. Keep
  articles. Section 1.4 says where strict STE is wrong for this site and gives a working compromise.

Overall verdict: the copy is already well above average. It is concrete, numbers-first and almost free of the usual
LLM vocabulary. There are no em dashes in site copy, no "delve/unlock/leverage/seamless", and no emoji except the one
in the tokenizer demo, which is there on purpose. The problems are second-order:

1. Key terms drift between chapters: *word* and *token*, *speed limit*, *ceiling* and *speed of light*, *lanes* and
   *cores*, *stage*, *rung* and *version*.
2. The site's own tic is the **colon chain**. 25–59% of sentences in every chapter hang two or three clauses off
   colons. It does the job em dashes do in LLM prose and makes sentences longer than they look.
3. There is a small but noticeable stock of **closing zingers and "not X, Y" pivots**: "That's it.", "Hold on to
   that", "Or can they?", "Speed is bought with accuracy…", plus 35 instances of ", not ".
4. **Captions run long.** They average 65 words, and 23 of 40 are over 60 words. Try-its often give away the result
   before the reader acts.
5. **Cross-references, indexing and docs are stale or inconsistent**: "chapter 05" vs "chapter 5", "fig. 1" vs
   "fig 4.1", layer 0 vs layer 1, a README that stops at stage 1, a lab footer that calls WebGPU "a later rung", and
   an end-of-course link to "the demo page" that goes to `/`.

---

## 1. House style guide (ready to drop into `web/STYLE.md`)

### 1.1 Voice

- **Who you write for.** An engineer who knows systems and not ML. Address them as **you**. Use **we/our** only for
  the project and its measurements ("our M4", "we measured"). Never use "we" for the reader and author together
  ("We now have the pieces" → "You now have the pieces").
- **Tone.** A colleague at a whiteboard: plain, specific, a little dry. Let numbers do the impressing. No hype words
  (*huge, massive, enormous, remarkable, perfect, magic, truly, every serious engine*).
- **Tense.** Present tense for how things work ("decode reads every weight"). Past tense only for the engine's
  history ("the first version ran at 0.37 tok/s"). Nothing in the future tense and no "would" for how the site
  behaves. Don't pre-announce future posts.
- **One idea per paragraph.** One topic per sentence. If a sentence needs a colon *and* a parenthesis *and* a comma
  splice, split it.
- **Metaphors.** Each concept gets **one** metaphor, used the same way everywhere. Approved:
  - *the truck* = memory bandwidth, *the factory* = arithmetic (ch 09, reused in ch 17)
  - *the notebook* = the residual stream (ch 06)
  - *pages* = virtual memory, for the paged KV cache (ch 17)
  - *a ring of GPUs* = all-reduce (ch 19)

  Retire competing metaphors for the same concept: the pantry door (ch 14) is bandwidth again, so use the truck.
  Checkout lanes, the paint roller, chefs, the assembly line and "passing dishes" are one-offs that compete with the
  approved set.
- **Endings.** End a paragraph on the fact, not on a zinger. A closing line is allowed only when it adds a new fact
  or a forward pointer.

### 1.2 Rules (numbered so reviews can cite them)

| # | Rule | Source |
|---|---|---|
| R1 | Second person for the reader; "we" only for the project. | Google: Second person |
| R2 | Active voice. Passive is OK only when the actor is unknown or irrelevant ("the weights are frozen"). | Google: Active voice; STE 3.1 |
| R3 | Present tense for behaviour; past tense only for the climb's history. | Google: Present tense |
| R4 | Body prose: aim for 20 words or fewer per sentence, cap at 30. Try-its and UI strings: 20 or fewer. Glossary one-liners: 25 or fewer. | STE 4.1/5.1 (adapted, see 1.4) |
| R5 | At most **one colon per sentence**, and only to introduce a list, an example or a result. Never chain "X: Y: Z". | house (counters the colon tic) |
| R6 | No "not X, Y" pivots unless the X is a misconception the reader actually holds and you name it. Budget: one per chapter. | anti-slop |
| R7 | No rhetorical questions in body prose. Questions are allowed as chapter `q` lines, `Deeper` titles and check-yourself prompts. | anti-slop |
| R8 | No throat-clearing openers ("Here is…", "Here's why that matters", "Time to…", "That's it.") and no closers ("Hold on to that", "That's the frame"). | anti-slop |
| R9 | Headings use sentence case and say what the section shows ("The result: 51× faster"), not just a template word ("The result"). | Google: Capitalization, Headings |
| R10 | Define every abbreviation on first use **per chapter** (TTFT, TPOT, HBM, SIMD, MoE, GQA, FFN). In the field guide, the term line is the definition. | Google: Abbreviations |
| R11 | Spelling: **British**, used consistently (colour, grey, maths, optimise, quantisation, datacentre, licence, neighbour). This is a deliberate exception to Google's US-spelling rule: the site is already 90% British. Fix the US stragglers. Code identifiers, flags and quoted product names keep their own spelling. | Google: Spelling (documented exception) |
| R12 | Curly quotes in prose (“ ” ’). Token strings always render through `<Tk>`, never inside straight quotes. | house |
| R13 | Instructions don't depend on the device. Use "select" or "choose", or "click or tap". Don't give hover-only instructions; say "point to (or tap)" and make the widget work by tap and keyboard. | Google: Write accessibly; UI interaction |
| R14 | Never use colour alone. Name the colour **and** the mark ("ember bars", "blue dots"), and keep a `Swatch` next to every colour word. Only use the palette names below. | Google: Accessibility |
| R15 | Link text says where it goes ("the field guide entry on GQA"), never "here", "this" or a bare URL. | Google: Link text |
| R16 | Every claim about speed carries its conditions the first time it appears in a chapter: machine, number format, users and prompt length. | house (provenance) |
| R17 | Don't pre-announce posts, features or "later" rungs. | Google: Future features |

### 1.3 Controlled vocabulary

Pick the term in the first column. The banned synonyms in the third column must not appear in reader-facing copy.
Field guide `aka` lines and search aliases are exempt.

| Concept | Use | Don't use (in reader copy) | Notes |
|---|---|---|---|
| Unit the model reads and writes | **token** | *word* (as a unit of work), *piece* (after ch 02), *symbol*, *character* | Allowed only before ch 02 defines tokens (landing hero, ch 01 title) and for real English words ("about ¾ of a word"). |
| Learned numbers | **weights** | *parameters* (except "7B parameters" model-size quotes and the field guide alias), *numbers* (after ch 0.1 defines weights) | "124 million weights". |
| Model size | **7B**, **70B**, **124M** in labels; "7-billion-weight model" in prose the first time | *seven billion* in one place and *7-billion-weight* in another | |
| One run of the model | **forward pass** (first use per chapter), then **pass** | *forward* (as a noun), *trip through the model*, *run of the model* | |
| One server iteration across a batch | **step** | *iteration*, *tick* (outside the 0.1 diagram) | "Step" never means a climb stage. |
| Engine versions in the climb | **stage** (each runnable version) and **milestone** (the six course bars) | *rung*, *version*, *step of the climb* | "Ladder" stays as the lab figure's title. |
| Speed rate | **tok/s** in figures, stats, tables and labels; **tokens per second** in running prose | *tokens a second*, *words a second* | Always say which: "per user" or "total". |
| Bandwidth ceiling for one user | **speed limit** (bandwidth ÷ bytes per token) | *ceiling*, *speed of light*, *limit line*, *yardstick* | *Roof* only inside the roofline figure. |
| Bytes per second from memory | **memory bandwidth**, then **bandwidth** | *memory speed*, *read speed*, *how fast memory goes* | |
| The arithmetic side | **arithmetic** (prose), **compute-bound** (the term), **FLOPs** (counts) | *maths/math* as a noun (except in the "Memory or maths" chapter title), *the work* when you mean FLOPs | |
| Waiting states | **memory-bound**, **compute-bound** | *bandwidth-bound* (except in the latency-bound vs bandwidth-bound fold) | |
| GPU execution units | **lane** (one arithmetic unit), **GPU thread** (software) | *cores* for GPU lanes | "GPU cores" only for Apple's spec unit ("10 GPU cores"). The glossary `GPU` entry says "cores" and has to change. |
| CPU units | **core**; **performance core**, **efficiency core** | *P-core*, *E-core* (docs only) | |
| The last matrix | **LM head** (with "the output layer that scores every token" on first use) | *scoring matrix*, *final scoring matrix*, *lm head*, *output layer* (alone) | |
| Feed-forward | **feed-forward network (FFN)** | *MLP* (allowed once as "also called the MLP") | The lab says "MLP" throughout; change it. |
| Raw scores | **scores** in part 1; **logits** once defined in ch 06 | mixing "scores" and "logits" in one paragraph | |
| Key/value store | **KV cache** | *kv cache*, *the cache* (alone) when the CPU cache or browser cache is also nearby | |
| Paged KV unit | **page** (16 tokens) and **block table** | *block* for a KV page | "Block" already means a transformer block and a quantisation block. |
| Number formats, prose | **32-bit, 16-bit, 8-bit, 4-bit** | *fp32, int4, f32* in running prose | |
| Number formats, labels | **f32, f16, bf16, Q8_0, Q4_0** | *fp32, int8, int4, q4_0, q8* | Field guide format entry exempt. |
| Small model in spec decoding | **drafter** | *guesser*, *draft model*, *the small model* (after first use) | |
| Big model in spec decoding | **big model** (course); **target model** (field guide) | *the checker* | |
| What the big model does | **check** | *verify*, *validate* | "Validate" is fine inside the Rosetta analogy. |
| Concurrency unit (course) | **user** for intuition (ch 0.2, 17); **request** for server mechanics | *stream*, *sequence*, *seat* | |
| Machine | **the M4 Mac mini** (first use per chapter), then **the M4** | *our Mac*, *the Mac*, *our Mac mini* | |
| Model | **GPT-2 small** | *GPT-2 124M* in prose (credits only) | Name GPT-2 medium and GPT-2 large in full. |
| Datacentre GPU | **H100** when specific; **datacentre GPU** when generic | *datacenter*, *server GPU* | |
| Site sections | **the course**, **the lab**, **the field guide** (all lowercase in prose) | *Foundations*, *foundations track*, *the demo page*, *the Lab*, *the engine room* (as a name) | |
| Colours (R14) | **ember** = compute, **blue** = memory, **olive** = communication, **grey** = time/idle | *orange*, *dark bars*, *gray* | |

**Words with more than one meaning (STE "one word, one meaning").** These collide today. Qualify them every time.

| Word | Meanings on the site | Rule |
|---|---|---|
| block | transformer block; quantisation block of 32; KV page ("block table") | Use "layer" for the transformer block in prose, "block of 32 weights", and "page" for the KV page. |
| stage | engine stage; pipeline-parallel stage | Use "pipeline stage" in full in ch 19 and the field guide. |
| head | attention head; LM head | Always "attention head" or "LM head". Never bare "head". |
| cache | KV cache; CPU L1/L2 cache; browser cache (OPFS); prefix cache | Always qualify: "KV cache", "CPU cache", "browser cache". |
| core | CPU core; Apple "GPU core"; tensor core | See the vocabulary table. |
| thread | OS/CPU thread; GPU thread; wasm worker | "CPU thread", "GPU thread". |
| load | load the weights; memory load instruction; machine load average | "busy machine" for load average; "memory read" for the instruction. |
| free | no cost; unused memory | Use "at no extra cost" for cost; keep "free" for memory. |
| limit | speed limit; capacity limit | "speed limit" vs "memory capacity". |
| step | decode step; climb milestone; merge step in BPE | "step" for the decode iteration only. |

### 1.4 Where strict STE is wrong here, and the compromise

STE was written for maintenance manuals. A reader there must do the procedure correctly and needs no persuasion or
memory hooks. This course has to build intuition. Five strict-STE rules hurt it:

1. **The approved-word dictionary.** STE would ban *nudge, squeeze, chop, stream, memoise*. Those words are precise
   in this field. Compromise: the controlled vocabulary above replaces the STE dictionary. Technical names are free,
   but each one has exactly one meaning.
2. **No metaphor.** STE forbids figurative language. The truck and the factory are the course's best teaching device.
   Compromise: one metaphor per concept, introduced once and reused (1.1), never stacked.
3. **No narrative past tense.** The climb is a story ("the first version ran at 0.37 tok/s"). Compromise: past tense
   for history, present tense for mechanism, never mixed in one sentence.
4. **25-word descriptive cap.** A few explanations need 30 words to keep one idea in one sentence. Compromise: soft
   cap 25, hard cap 30 in body prose. Strict STE limits (20 and 25) apply to try-its, UI strings, glossary lines and
   check-yourself options, which are procedural or reference text.
5. **No "-ing" forms and no "you know".** Rapport lines ("You know this shape") are useful **once** in part 0 for
   the systems-engineer framing. Compromise: allow one per chapter in part 0 and none after.

Keep from STE without exception: one meaning per word, one topic per sentence, active voice, articles not dropped
(captions currently drop them: "Across: rank. Up: probability" is fine as a label, but in sentences keep "the"), and
no ambiguous phrasal verbs ("ride along", "lands close together", "creeps in" → say what happens).

### 1.5 Numbers and units

- **Numerals for every technical quantity**, including those under 10: "4 cores", "12 layers", "8 bits". Spell out
  only non-technical counts and numbers that start a sentence ("Sixteen totals…"; better to reword). Today the site
  mixes "ten cores" and "10 cores", "twelve layers" and "12 layers", "sixteen running totals" and "16 totals".
- **Space between number and unit**, nonbreaking (`64&nbsp;GB`, `120&nbsp;GB/s`, `130&nbsp;µs`). No space for `%`,
  `×` factors or `$`. (Google: Units of measurement.)
- **Commas in four or more digits**: 50,257 and 1,456. Exceptions: years, and code or flags.
- **Ranges.** House rule: an en dash with no spaces, with the unit on both ends when a unit is present: "50–80%",
  "98–101 tok/s", "3 GB–8 GB". Google prefers "to" or a hyphen. We keep the en dash because ranges sit in dense
  figure labels, but in prose with units write "4 to 6 threads" and "3 to 8 GB". Never "-" for a range: it reads
  as minus.
- **Approximation.** `~` in labels and stats, "about" in prose. Use `≈` only in equations.
- **Multiplication and factors.** `×` for dimensions and speed-ups ("768 × 2304", "51×"). Never `x`.
- **Operators.** `÷` in prose equations, `/` only inside units (GB/s, tok/s).
- **Large magnitudes.** Prose: "124 million weights", "7 billion". Labels: 124M, 7B, 128k (lowercase k for
  context), 1,024.
- **Bandwidth units.** GB/s below 1,000 and TB/s above, in the same chapter. ch 19 uses "3,350 GB/s" while ch 0.3
  uses "3.35 TB/s": pick TB/s.
- **Speed figures.** Always "X tok/s per user" or "X tok/s total". Every speed has a machine and number format nearby.
- **Arrows.** `→` means "went from → to" in results only. Don't use it for causality.

### 1.6 Headings and titles

- Sentence case everywhere: chapter titles, figure titles, `Deeper` titles, field guide groups.
- Replace `&` with "and" ("Models & architecture" → "Models and architecture").
- Body headings say what they deliver. The part-3 template "The problem / The idea / The result" is a good
  skeleton, but put the payload in it: "The result: 51× faster, and flat".
- No question headings in body (`h2`). "So when does a CPU win?" becomes "When a CPU wins".
- Chapter `q` lines are questions by design. Keep them, but each must be answerable from the chapter (see finding
  T-1 on "word").
- Labels, kickers and caption keys stay lowercase on purpose ("the idea", "try", "what you're looking at"). Document
  this as the only exception to sentence case.

### 1.7 Figure caption template

Each `Fig` has up to three parts. Keep them in this order and within these limits.

**What you're looking at** (`caption`), 60 words or fewer, 4 sentences or fewer:

1. What the figure is, and the data source if it isn't obvious from the provenance tag. ("A recorded run on the M4's
   GPU.")
2. The axes, as `Across: … Up: …`, with "log scale" where it applies. Explain what a gridline means once per
   chapter, not in every caption.
3. The encodings: colour **and** mark, each with a `Swatch`. ("Ember bars: the GPU working. Grey: idle.")
4. Optional: one caveat that changes how to read it.

No instructions, results or morals in this part.

**Try** (`tryit`), 35 words or fewer, 2 sentences or fewer:

- Start with an imperative verb that works on any device: *Choose, Drag, Switch, Press, Step through*.
- Then one "Notice …" clause that says **what to look for**, not the number you will see. Today many try-its
  narrate the outcome ("the total grows 3.2× while each user slows by 9%"). Move results to the body text.

**Sources** (`source`): noun phrases only. Mark estimates "(estimate)". No sentences about methodology. Methodology
goes in a `Deeper` fold.

Figure references: write "fig 12.1" in the course (no full stop) and "lab fig 3" from the course into the lab. Inside
the lab, write "fig 3".

---

## 2. Findings

Severity:

- **High**: wrong, contradictory or stale content, or a terminology split that confuses the model of the system.
- **Medium**: slop or style problems a careful reader notices.
- **Low**: polish.

Line numbers are for the files as they were at review time.

### 2.1 High

| # | File:line | Current text | Problem | Rule | Rewrite |
|---|---|---|---|---|---|
| H1 | `learn/chapters.tsx:49` | title "Every word reads every weight"; q "Why is each word a model writes so expensive?" | The chapter and its own fig 0.2.1 title ("Every token reads the whole model") use different units. The constraint map (`concepts.ts:27`) also says "Every token reads every weight". | Vocab: token | "Every token reads every weight" / "Why is each token a model writes so expensive?" |
| H2 | `learn/chapters.tsx:41` | "…why every word is expensive…" (part 0 blurb) | Same split, on the most visible map. | Vocab: token | "…why every token is expensive…" |
| H3 | `learn/ch/StartingLine.tsx:13` | "It wrote 0.37 tokens per second. Nearly three seconds per word." | Changes unit mid-thought. 1 ÷ 0.37 = 2.7 s per **token**, and a word is about 1.3 tokens. | Vocab: token | "It wrote 0.37 tokens per second: nearly 3 seconds per token." |
| H4 | `learn/ch/p0/WhatIsInference.tsx:531`, `:540`; `learn/checks.ts:37`; `learn/ch/p0/WhyExpensive.tsx:152`, `:355` | "chapter 05", "chapter 04", "chapter 07", "chapter 09", "chapter 08" | Zero-padded references. Other chapters write "chapter 5" (`KvCache.tsx:41`, `Block.tsx:52`) and the lab writes "Chapter 6". | 1.7 refs | "chapter 5", "chapter 4", "chapter 7", "chapter 9", "chapter 8" |
| H5 | `panels/InsideToken.tsx:139`, `:175` | "Layer 0 normalises the vector…"; "The same eleven ops repeat in layers 1 to 11" | The lab counts layers from 0. The course counts them from 1 ("after each of the 12 layers", `Block.tsx:131`; the slider shows v+1, `Attention.tsx:233`). A reader coming from ch 06 sees two different layer 1s. | One meaning per word | "Layer 1 normalises the vector…"; "The same 11 operations repeat in layers 2 to 12" |
| H6 | `learn/LearnApp.tsx:166-168` | "the end of the beginning → / Now watch it all run, live / The demo page: the same engine, every stage of the climb." | It links to `/` (the landing page), but the description fits the lab. "Demo page" is a retired name. "The end of the beginning" is a cute but empty line. | Vocab: the lab; R15 | k "next · the lab →" / t "The lab: every stage, running in your tab" / q "The same engine with nothing simplified away." Link `/lab`. |
| H7 | `App.tsx:41-43` | "…weights downloaded as f16 and widened to f32 on load · WebGPU is a later rung of the ladder." | Stale: WebGPU stages ship in the ladder now (`StagePicker`, `webgpu.json`). It also pre-announces. | R17 | "…weights downloaded as f16 and widened to f32 on load · WebGPU stages run where the browser supports them." |
| H8 | `learn/ch/ManyGpus.tsx:125` | "This is the subject of a later post in the series, run on real rented GPUs." | Pre-announces a post that doesn't exist. The site isn't a "series". | R17 | Delete the sentence, or "The field guide's tensor-parallelism entry has the numbers for big models." |
| H9 | `ask/AskLayer.tsx:419-421` | "Ask isn't switched on here. It needs a model API key on the server… put `TOGETHER_API_KEY=…` in `web/.env.local` and restart `npm run dev`; on Vercel…" | Developer setup instructions shown to every public visitor when the key is missing. | Audience | Production: "Ask is unavailable right now." Show the dev hint only when `import.meta.env.DEV`. |
| H10 | `learn/kit.tsx:426` vs `learn/guide/entries.tsx:42` vs `learn/ch/Tokens.tsx:177-178` | glossary: "…a word, part of a word, or a single character"; guide: "…or a symbol"; chapter: "…or a single byte of an emoji" | Three definitions of the same term. "Character" is wrong for byte-level BPE. | Vocab: token | All three: "A piece of text the model reads or writes: a word, part of a word, or a single byte. GPT-2 knows 50,257 of them." |
| H11 | `learn/kit.tsx:451` | GPU: "A chip with thousands of small, simple cores…" | Contradicts ch 0.3, which carefully says *lanes* and reserves *cores* for "10 GPU cores × 128 lanes" (`Hardware.tsx:14`, `:353`). `Gpu.tsx:87` repeats "thousands of small, simple ones". | Vocab: lane | "A chip with thousands of simple arithmetic lanes that run the same instruction on many numbers at once." |
| H12 | `README.md:26-31` | "## Where we are" table with stage 0 (0.37) and stage 1 (18.95) only | The repo front page stops at stage 1. The site claims 817 tok/s and 2,200×. It also contradicts `docs/LOG.md`. | Stale | Replace with the milestone table (naive 0.37 → KV cache 19 → arithmetic 100 → threads 146 → 4-bit 497 → GPU 817 → batching 6,453 total), each with machine and format, and a link to the site. |
| H13 | `web/README.md:53` | "Select text anywhere on the site → ask / simpler / why? / example → a streamed answer…" | The UI buttons are Ask, Explain, Define and Simplify ¶ (`AskLayer.tsx:376-379`). | Stale | "Select text anywhere → Ask, Explain, Define or Simplify → a streamed answer in a side panel…" |
| H14 | `web/README.md:3`, `:46` | "The demo page for nanoinfer…"; "`/lab` the engine room: the original live demo, curated" | The site is now a course plus a lab. The README calls it a demo page. | Vocab | "The site for nanoinfer: a 23-chapter course on how a language model runs, and a lab where every engine stage runs in your tab." |
| H15 | `web/README.md:161-162` | "The recording was made on a busy machine… re-run `node scripts/record.mjs` on a quiet one before publishing." | Either stale (the site says numbers are from a quiet machine, `Gpu.tsx:131`) or a live accuracy problem. | Stale/accuracy | Re-record and delete the item, or say which figures still use the busy recording. |
| H16 | `web/README.md:114-115` | "They're needed for threaded wasm later (SharedArrayBuffer)." | The threads build ships (`pkg-threads`). | Stale; R17 | "They're needed for the threads build (SharedArrayBuffer)." |
| H17 | `docs/WRITEUP.md:772-774` | "…then 19 one-idea chapters in three parts…" | Now 23 chapters in 4 parts. | Stale | "…then 23 one-idea chapters in four parts (part 0 frames inference for systems engineers)…" |
| H18 | `panels/registry.tsx:192-193` | "Dark bars are the native engine on our M4; orange are the same Rust compiled for the browser…" | "Orange" isn't in the palette (ember). It relies on colour alone ("dark", "orange"). | R14 | "Grey bars: the native engine on the M4. Ember bars: the same Rust in the browser (dashed: recorded; solid: measured in your tab)." |
| H19 | `learn/ch/Gpu.tsx:33`; `learn/ch/Batching.tsx:166`; `learn/ch/p0/Hardware.tsx:217`; `learn/guide/MoeLab.tsx:130` | "Hover each row for what changed."; "hover a bar for one request's story"; "Hover a row for its source."; "Hover a token to see its route." | Hover-only instructions exclude touch and keyboard users. | R13 | "Select a row to see what changed." / "Select a token to see its route." (and make rows focusable) |
| H20 | `learn/ch/p0/WhyExpensive.tsx:47-48` (try) | "A 7B model in bf16 is 14 GB per token: even an H100 tops out near 240 tokens a second. Now pick 70B… Switch to int4 and it fits, at 4× the speed limit." | The try gives every answer away before the reader acts. It also uses "int4", which the rest of the course calls "4-bit". | 1.7 Try; vocab | "Pick 7B, then 70B, then switch 70B to 4-bit. Notice which change decides whether it runs at all, and which decides how fast." |

### 2.2 Medium

| # | File:line | Current text | Problem | Rule | Rewrite |
|---|---|---|---|---|---|
| M1 | `learn/ch/p0/WhatIsInference.tsx:505-507` | "You run services. A language model, once trained, is a service with an unusual profile: a fixed, enormous, read-only blob…, one fixed program, and a request loop…. That's it. The rest of the field is plumbing around that loop." | Presumptuous opener, a reflex triplet in a 36-word colon sentence, then a "That's it." zinger and an overclaim ("the rest of the field is plumbing"). | R5, R8 | "A trained language model is an unusual service. It has a large read-only file of numbers and one fixed program. Each word it writes runs the program over the whole file." |
| M2 | `learn/ch/p0/WhatIsInference.tsx:546-547` | "If you squint, it's a stateful service with four jobs:" | Filler hedge. | R8 | "It is a stateful service with four jobs:" |
| M3 | `learn/ch/p0/WhyExpensive.tsx:344` | "Here is the fact the whole field is organised around." | Throat-clearing plus overclaim. | R8 | Delete. Start with "To write one token, the model runs its program once, and that program reads every weight." |
| M4 | `learn/ch/p0/WhyExpensive.tsx:371-373` | "That's why serving is a game of finding more useful work for each byte: more users per step (batching), more tokens per user per step (speculative decoding), or fewer bytes per weight (quantisation)." | 33 words. The "game of" metaphor is empty. The triplet is fine here because it is a real list. | R4 | "So serving engines look for more useful work per byte. There are three ways: more users per step (batching), more tokens per user per step (speculative decoding), and fewer bytes per weight (quantisation)." |
| M5 | `learn/ch/p0/WhyExpensive.tsx:397-398` | "That's the economics of every LLM API in one chart: cheap tokens come from big batches, big batches need memory, and long contexts eat it." | Overclaim ("every LLM API") plus a chained aphorism. | R5, R6 | "The chart shows the basic economics of serving. Big batches make tokens cheap. Big batches need memory, and long contexts use it up." |
| M6 | `learn/ch/p0/WhyExpensive.tsx:366` and `:383` | "If you've profiled a service… you know this shape." / "You know this shape: it's per-session state that grows." | The same rapport template twice in one chapter (and "you'd expect" in 0.1:551). | 1.4 #5 | Keep the first. Change the second to "This is per-session state that grows." |
| M7 | `learn/ch/p0/Hardware.tsx:355` | "A matrix multiply is the perfect workload: millions of independent multiply-adds." | Intensifier. | Hype | "A matrix multiply suits a GPU well: it is millions of independent multiply-adds." |
| M8 | `learn/ch/p0/Hardware.tsx:371` | h2 "So when does a CPU win?" | Question heading. | R7, R9 | "When a CPU is the better choice" |
| M9 | `learn/ch/p0/Hardware.tsx:389` | h2 "Why more GPUs is hard" | Agreement error ("GPUs is"). Vague. | R9 | "Why splitting across GPUs costs time" |
| M10 | `learn/ch/p0/Hardware.tsx:398` | "Amdahl's law, in a new costume: …" | Cute phrasing for a clear idea. | Anti-slop | "This is Amdahl's law: the per-token communication is a serial part that more GPUs don't shrink." |
| M11 | `learn/ch/p0/Hardware.tsx:403-405` (Idea) | "…so splitting a model is a trade, not a free speed-up." | "Not X, Y" closer. | R6 | "…so splitting a model buys capacity and bandwidth at the cost of communication time." |
| M12 | `learn/ch/p0/Rosetta.tsx:121-123` | "Most of inference engineering's vocabulary names ideas you already use, applied to a strange workload: one enormous read-only blob, a per-request state…, and a synchronous loop… Here's the translation table. Keep it open while reading the rest." | 38-word colon sentence with a triplet that repeats 0.1:505 nearly word for word ("enormous read-only blob"). "Here's" opener. | R4, R8, repeated template | "Most inference terms name ideas you already use. The difference is the workload: every step reads gigabytes. The table maps each term to its systems equivalent." |
| M13 | `learn/ch/p0/Rosetta.tsx:136-137` | "That's the frame. Part 1 opens the model up and shows what those gigabytes of numbers actually do…" | Throat-clearing closer, "actually". | R8 | "Part 1 opens the model and shows what those weights do, starting with its one job: guessing the next token." |
| M14 | `learn/ch/p0/Rosetta.tsx:34` | Kernel launch "breaks": "The fix is the same as for chatty RPC: batch the calls…" | The "breaks" column should say where the analogy fails. This entry gives the fix instead. | Logic | "Unlike an RPC, the cost is mostly the CPU waiting, not the call; batching calls without removing the wait helps little." (matches ch 16) |
| M15 | `learn/ch/NextToken.tsx:311` | "That's it. Answering questions, writing code, holding a conversation: all of it is that one trick, done over and over." | Second "That's it." on the site. Reflex triplet. | R8 | "Answering questions, writing code and chatting all come from repeating that one step." |
| M16 | `learn/ch/NextToken.tsx:323-325` | "The list really is complete… Hold on to that: doing the same big computation for every single word is what part two of this course is about." | "really", "every single", "Hold on to that", and "word" for token. | R8, vocab | "The list is complete: GPT-2 scores all 50,257 tokens every time. Part 2 is about the cost of that full computation for each token." |
| M17 | `learn/ch/NextToken.tsx:332-334` | "Yes, in the same way a chess engine is "just" picking moves. The mechanism is simple; what's remarkable is how good the guesses get." | "remarkable", straight quotes. The answer to the fold title only arrives in the next sentence. | Hype, R12 | "Yes. The mechanism is autocomplete. The difference is the quality of the guesses: to predict a physics paper well, a model has to encode some physics." |
| M18 | `learn/ch/NextToken.tsx:94` (try) | "Deliberately pick unlikely tokens and watch the story bend." | "watch the story bend" is a cute phrasal construction. | STE phrasal | "Pick a few unlikely tokens. Notice how the next guesses change." |
| M19 | `learn/ch/Tokens.tsx:197-199` | "Notice the leading spaces… Notice the numbers… And notice that the count of tokens…" | A three-beat "Notice" anaphora. "Notice" appears 11 times on the site. | Repeated template | "Leading spaces matter: “ dog” and “dog” are different tokens. Numbers split in odd places. From here on, the token count is what matters, not characters or words." |
| M20 | `learn/ch/Tokens.tsx:204-206` | "It was learned by a simple greedy rule called byte-pair encoding: start with the 256 possible bytes, find the pair that appears side by side most often in a mountain of text, make it a new token, and repeat 50,000 times." | 42 words, passive, "mountain of text". | R2, R4 | "A greedy rule called byte-pair encoding (BPE) built it. BPE starts with the 256 possible bytes. It finds the most common neighbouring pair in a large body of text, makes that pair a new token, and repeats 50,000 times." |
| M21 | `learn/ch/Embeddings.tsx:186-188` | "Nobody wrote those numbers. They started random and were nudged… What they ended up encoding is surprisingly sensible:" | "Nobody…" is a site-wide template (4×: here, :199, `Tokens.tsx:204`, `NextToken.tsx:318`). "surprisingly" is an intensifier. Passive. | Repeated template | "Training set those numbers. They start random, and training adjusts them until they help the model guess well. The result groups related tokens together:" |
| M22 | `learn/ch/Embeddings.tsx:199-200` | "Nobody told the model about weekdays or pets; tokens that show up in similar places in text simply ended up with similar numbers." | "Nobody", "simply". | Same | "The model was never given these categories. Tokens that appear in similar contexts get similar numbers." |
| M23 | `learn/ch/DotProduct.tsx:331` | "Here is the operation that does nearly all the work in this whole course." | Opener template; "in this whole course" is wrong (it's the work in the model). | R8 | "One operation does nearly all of a model's work." |
| M24 | `learn/ch/DotProduct.tsx:60` | "…chapter 3's neighbours were found this way…" | Passive; possessive chapter reference. | R2 | "…the neighbours in chapter 3 come from this, with 768 numbers instead of 2." |
| M25 | `learn/ch/DotProduct.tsx:348` (Idea) | "…Make that fast, and you've made the model fast." | Aphorism, and wrong by the course's own argument: part 2 shows that making the arithmetic fast doesn't help decode. | Accuracy | "Running a model is mostly matrices times vectors. Part 2 shows that the cost is reading those matrices, not the arithmetic." |
| M26 | `learn/ch/Attention.tsx:283-285` | "So far, each token's vector only knows about its own token. But “it” means nothing without knowing what came before." | "So far" opener template (also "We now have the pieces" and "We have everything"). "Knows" anthropomorphises. | Template | "Each token's vector so far holds information about that token only. The word “it” needs the words before it." |
| M27 | `learn/ch/Attention.tsx:306-307` | "…there isn't one attention but 144 of them running side by side, each free to learn its own habit." | "not one X but Y" construction; "attention" used as a countable noun. | R6, vocab | "GPT-2 has 144 attention heads (12 per layer), and each one learns its own pattern." |
| M28 | `learn/ch/Block.tsx:191` | "…GPT-2 is just twelve blocks in a row: twelve layers." | "just" (26 uses site-wide); number style. "Block" collides with other meanings. | 1.5, STE | "GPT-2 is 12 of these blocks in a row, called layers." |
| M29 | `learn/ch/Block.tsx:52-53` | "(also called the MLP): two big matrices with a bend in between" | "a bend in between" is vague for GELU. | Define terms | "(also called the MLP): two large matrices with a nonlinear function (GELU) between them" |
| M30 | `learn/ch/Sampling.tsx:161-162` | "…it's where a lot of a model's "personality" comes from." | Straight quotes. Vague anthropomorphism. | R12 | "…and it controls how predictable the text is." |
| M31 | `learn/ch/Sampling.tsx:174` | "That's why the demo page can re-sample a token…" | Retired name. | Vocab | "That's why the lab can re-sample a token…" |
| M32 | `learn/ch/Sampling.tsx:77` (try) | "…set temperature to 2 and draw again: nonsense creeps in. Set top-k to 5 and it's gone." | Narrates the result. "creeps in" is a phrasal verb. | 1.7 Try | "Press draw 20 at temperature 0.8, then at 2, then at 2 with top-k 5. Notice when rare tokens appear." |
| M33 | `learn/ch/Loop.tsx:158-162` | "…(This early, single-core engine shows only a small gap; on a GPU the same effect makes prefill 8× faster than decode, chapter 16.) Same arithmetic per token, very different bytes per token: hold on to that, it's the whole of part two." | A parenthetical sentence inside a 5-sentence paragraph, then a zinger and "Hold on to that" (again). | R8 | End with: "Prefill and decode do the same arithmetic per token but read very different bytes per token. Part 2 is about that difference." |
| M34 | `learn/ch/Loop.tsx:172-173` | "Part two asks the question this whole project is about: why does each of those decode steps take as long as it does?" | Rhetorical question used as a bridge. | R7 | "Part 2 measures why each decode step takes as long as it does." |
| M35 | `learn/ch/BytesFlops.tsx:296` | "Part one ended with a fact: every decode step uses every weight in the model. Here's why that matters." | Throat-clearing pair. | R8 | "Every decode step reads every weight in the model." |
| M36 | `learn/ch/BytesFlops.tsx:325-326` | "…decode is memory-bound on essentially every computer that exists." | Overclaim and intensifier. | Hype | "…for one user, decode is memory-bound on every common CPU and GPU." |
| M37 | `learn/ch/SpeedOfLight.tsx:159-161` | "That one division explains a lot about the AI industry. It's why people pay for chips with faster memory, why models are shrunk to fewer bits, and why a 70-billion-weight model writes slower…: it's just more bytes to read per token." | Overclaim opener. A 39-word reflex triplet with a colon tail. | R4, R5 | "This division explains three industry habits. Fast-memory chips cost more. Models are quantised to fewer bits. And a 70B model writes more slowly than a small one on the same machine, because it reads more bytes per token." |
| M38 | `learn/ch/SpeedOfLight.tsx:146-148` | "…(If you read chapter 0.2, you've seen this division; here it becomes the yardstick for every step of the engine.)" | A parenthetical sentence. "yardstick" is a fourth synonym for speed limit. | Vocab | "Chapter 0.2 introduced this division. Here it becomes the speed limit that every stage is measured against." |
| M39 | `learn/ch/StartingLine.tsx:25-26` | "A correct program can be absurdly wasteful, and this one was. That's the starting line for the climb:" | Faux-profound line plus a heading-like lead-in. | R8 | "The program was correct and very wasteful. That waste is where the climb starts:" |
| M40 | `learn/ch/KvCache.tsx:59-60` | "…It was reading 10. Something else is slow, and it's the next chapter." | "it's the next chapter" is a cliffhanger, not information. | R8 | "It was reading 10 GB/s, so something other than memory is slow. Chapter 13 finds it." |
| M41 | `learn/ch/Arithmetic.tsx:137-138`, `:143` | "It's like opening more checkout lanes." / "…a paint roller instead of a brush." | Two new metaphors in one chapter, on top of truck/factory. "Lanes" also collides with GPU lanes. | 1.1 metaphors | Delete both. The figure already shows overlap. "SIMD instructions do the same operation on 4 numbers at once." |
| M42 | `learn/ch/Arithmetic.tsx:144-145` | "…because adding decimals in a different order can change the last digit of the answer, so it isn't allowed to reorder them on its own." | "decimals" means floating-point numbers. One sentence carries two topics. | STE one topic | "Floating-point addition gives slightly different results in a different order. So the compiler may not reorder the adds by itself." |
| M43 | `learn/ch/Arithmetic.tsx:150-151` | "…essentially all that one core can pull from memory (57). Decode had become truly memory-bound." | Intensifiers; the bare number has no unit. | Hype; 1.5 | "…close to the 57 GB/s one core can read. Decode was now memory-bound." |
| M44 | `learn/ch/Threads.tsx:101-102` | "Ten cooks in a kitchen can chop ten times as fast. But if they all fetch ingredients through one narrow pantry door…" | A second metaphor for bandwidth (the truck already is bandwidth). Number style. | 1.1 | "Ten cores can do ten times the arithmetic, but they share one memory bus. In chapter 9's terms: one truck, ten factories." |
| M45 | `learn/ch/Threads.tsx:124-125` | "A spin pool assumes you own the machine, and a laptop never does." | Faux-profound closer. | R8 | "Spinning only works when nothing else needs the cores." |
| M46 | `learn/ch/Threads.tsx:115` | "…so more cores really is more speed…" | "really" and an awkward agreement. | Hype | "…so more cores give more speed…" |
| M47 | `learn/ch/Quantization.tsx:125-126` | "Speed is bought with accuracy, a little at a time." | Faux-profound closer. It repeats the Idea box below it. | R8 | Delete. |
| M48 | `learn/ch/Quantization.tsx:122-124` | "4-bit: 497 tokens per second, the fastest the CPU got, and here the cost is visible: its top choice for the next word differs from the 32-bit model's about one time in five…" | 41 words, two colons, "word" for token. | R4, R5, vocab | "4-bit: 497 tokens per second, the fastest CPU result. The cost is now visible. The top-choice token differs from the 32-bit model's about 1 time in 5, and perplexity rises about 8%." |
| M49 | `learn/ch/Quantization.tsx:143-144` | "Small-model lessons don't always transfer." | Aphorism. | R8 | "So this fix matters most for small models." |
| M50 | `learn/ch/Gpu.tsx:89` | "…so for decode it should at least keep up. It should win easily." | Set-up for a reveal; two "should"s. | R8 | "…so for decode it should at least match the CPU." |
| M51 | `learn/ch/Gpu.tsx:99-100` | "…like asking one worker to carry a whole delivery while the rest watch." | Another one-off metaphor. | 1.1 | "…so one GPU thread did the work while thousands sat idle." |
| M52 | `learn/ch/Gpu.tsx:117-119` | "The expensive thing was never the number of kernels; it was the CPU and GPU taking turns." | "It's not X, it's Y" construction (it also appears verbatim in WRITEUP). | R6 | "The cost was the CPU and GPU taking turns, so removing the waits mattered more than reducing the kernel count." |
| M53 | `learn/ch/Batching.tsx:374-376` | "…each weight fetched from memory now does 64 multiply-adds instead of one. The truck delivers the same, the factory finally has work." | Comma splice in a zinger. | R8 | "…now does 64 multiply-adds instead of 1. In chapter 9's terms, the same truckload now keeps the factory busy." |
| M54 | `learn/ch/Batching.tsx:381-383` | "…extra users ride along almost free. Once the arithmetic catches up with the bytes…, more users share a fixed amount of maths: the total keeps creeping up…" | Phrasal verbs ("ride along", "catches up", "creeping up"). | STE phrasal | "…extra users cost almost nothing. Past the crossover, the step is compute-bound: total throughput rises slowly and each user's speed falls." |
| M55 | `learn/ch/Batching.tsx:396-398` | "Short answers sit in their seats doing nothing…" | A seats metaphor that adds nothing ("seat" also appears in the Idea box). | 1.1 | "Finished requests hold their slots until the longest request ends…" |
| M56 | `learn/ch/SpecDecode.tsx:262-263` | "…the tokens of a single answer can't be put through the model together. Or can they?" | Rhetorical-question hook. | R7 | "…can't be put through the model together. Speculative decoding gets around this by guessing." |
| M57 | `learn/ch/SpecDecode.tsx:300-301` | "Large simply writes different code from small. How well the trick works is a property of the two models and the text, which is why it is measured, not assumed." | "simply", "the trick", "not X" closer. | R6 | "GPT-2 large writes different code from GPT-2 small. The speed-up depends on both models and on the text, so you have to measure it." |
| M58 | `learn/ch/ManyGpus.tsx:114` | "It buys knowledge without FLOPs, not without bytes." | Aphorism, and "knowledge" is vague. | R6 | "MoE adds weights without adding FLOPs per token, but every expert's weights must stay in memory." |
| M59 | `learn/ch/ManyGpus.tsx:96`, `:31-35` | "Chefs working on one dish at once…"; fig "Passing dishes round a table" | Third and fourth kitchen metaphors in one chapter, besides the assembly line. | 1.1 | Fig title "Ring all-reduce: each GPU passes one chunk per step". Drop "Chefs…". |
| M60 | `learn/ch/ManyGpus.tsx:106` | "Inside a GPU, memory moves ~3,350 GB/s." | Ch 0.3 states 3.35 TB/s. | 1.5 | "Inside an H100, memory moves about 3.35 TB/s." |
| M61 | `site/Landing.tsx:156-158` | "…For every one of those tokens, the computer read all 495 MB of GPT-2's numbers… That is why it isn't faster, and the course explains how to change it." | "every one of", "numbers" for weights, the hero's closing clause is a plug. | Vocab | "Each token read all 495 MB of GPT-2's weights from memory, about N GB per second. Memory bandwidth sets this speed; the course shows how to raise it." |
| M62 | `site/Landing.tsx:199-202` | "The course starts with what inference is and why it's expensive, framed for engineers who know systems but not ML; builds up to why this is slow; then follows the engine as it gets N× faster…" | A 50-word, semicolon-chained hero sentence. | R4 | "The course is for engineers who know systems but not ML. Part 0 explains what inference is and why it is expensive. Parts 1 and 2 explain why it is slow. Part 3 follows the engine as it gets N× faster, one measured step at a time." |
| M63 | `site/Landing.tsx:211-212` and `site/Journey.tsx:26` | "recommended: part 0 frames inference for engineers who know systems, not ML" / "start here if ML is new to you" | Two different recommendations for the same part on one page. | Consistency | Use one: "start here if you know systems but not ML". |
| M64 | `learn/LearnApp.tsx:44`, `:155` | document title "Foundations · nanoinfer", "nanoinfer foundations"; prev link "Foundations" | A retired section name, visible in the browser tab and in navigation. | Vocab | "The course · nanoinfer"; "{chapter} · nanoinfer course"; prev "Contents". |
| M65 | `learn/LearnApp.tsx:217`; `learn/FieldGuide.tsx:107` | "…every term people throw around, in one plain sentence each…" | "throw around" is a phrasal verb and mildly dismissive. | STE phrasal | "…53 terms, each in one plain sentence, with the limit it works around." |
| M66 | `learn/guide/entries.tsx:78` | FlashAttention where: "Every serious engine; …" | Vague and judgemental intensifier. | Hype | "vLLM, SGLang, TensorRT-LLM and llama.cpp; FlashAttention-2/3, FlashDecoding, FlashInfer." |
| M67 | `learn/guide/entries.tsx:79` | term "Kernels", plain "A function that runs on the GPU…" | The term is plural and the definition is singular. Other entries are singular. | Consistency | term "Kernel" |
| M68 | `learn/guide/entries.tsx:44` | Layers why: "Depth is where capability comes from…" | Overclaim. | Hype | "More layers generally give a more capable model; each layer runs in sequence…" |
| M69 | `learn/checks.ts:136` | option "Whole words would make the model too accurate" | A joke distractor gives the answer away. The comment at checks.ts:2 says distractors should be plausible misconceptions. | Test design | "Whole words would need fewer forward passes but make the model slower to train" (plausible and wrong), or "Whole words can't be stored as integers". |
| M70 | `learn/checks.ts` (all) | correct options are often the most specific or longest (e.g. :44, :226, :290) | Test-wise readers can pick without understanding. | Test design | Match option lengths within about 20%. Give each distractor a mechanism. |
| M71 | `learn/checks.ts:105` | why: "Execute speculatively, validate in bulk, discard on conflict." | Repeats the option text verbatim instead of explaining. | Sentences that say nothing | "The drafter's guesses are like an optimistic transaction: the big model checks them all in one pass and rolls back from the first mismatch. It pays when guesses are usually right." |
| M72 | `panels/registry.tsx:39`, `:62`, `:103`, `:125`, `:144`, `:165` | "chapter 8", "Chapter 6", "Chapter 5", "fig. 1" | The lab uses another reference style ("fig." with a full stop). | 1.7 refs | "chapter 8", "fig 1" |
| M73 | `panels/registry.tsx:47-50` | "<b>What you're looking at:</b> … <b>Try:</b> …" in one paragraph | Lab captions jam both parts into one paragraph with bold inline keys. Course figures use separate labelled lines. | 1.7 template | Split into `caption` and `tryit` like `Fig`, or render the same `lr-cap-k` labels. |
| M74 | `README.md:5-6` | "`research/` — distilled notes from the best resources (…)"; "`docs/LOG.md` — the hillclimb journal…" | Spaced em dashes (Google: no spaces), and "the best resources" is puffery. | Google: Dashes | "`research/`: notes on engines, kernels, serving, distributed inference and explorable explanations" |
| M75 | `docs/WRITEUP.md:24-25` | "The single most important idea in inference engineering." | Overclaim (in draft copy that's meant to become the post). | Hype | "The idea the rest of this post depends on." |
| M76 | `docs/WRITEUP.md:12`, `:59` | "started with the dumbest version that gives the right answer"; "the dumbest correct forward pass" | Flippant word in a public pitch draft. | Tone | "the simplest version that gives the right answer" (matches the site, `StartingLine.tsx:8`). |
| M77 | `docs/WRITEUP.md:214` | "Aha 2 (law 6 arrives): …" | It refers to a "law 6" that is never defined in the doc. | Define terms | "Aha 2 (overhead takes over): …" |

### 2.3 Low

| # | File:line | Current text | Problem | Rule | Rewrite |
|---|---|---|---|---|---|
| L1 | `learn/ch/BytesFlops.tsx:325` | "a datacenter GPU" | US spelling; elsewhere "datacentre" (5×). | R11 | "a datacentre GPU" |
| L2 | `learn/guide/entries.tsx:102` | "optimizer state" | US spelling inside British copy. | R11 | "optimiser state" |
| L3 | `panels/Roofline.tsx:31` | "…how much math it does per byte…" | "math" vs "maths" (22×). | R11 | "…how much arithmetic it does per byte…" |
| L4 | `learn/ch/p0/Rosetta.tsx:33`, `:34` | "optimistic execution + validation + rollback", "batch the calls (command buffers, CUDA graphs) and wait once" | `+` used as "and" in prose. | Google: symbols | "optimistic execution, then validation and rollback" |
| L5 | `learn/ch/Sampling.tsx:162`; `learn/ch/NextToken.tsx:332`; `learn/ch/Tokens.tsx:219`; `panels/InsideToken.tsx:176` | "personality", "just", "end of text", "massive activations" in straight quotes | Mixed with curly quotes elsewhere. | R12 | Curly quotes. |
| L6 | `learn/ch/Embeddings.tsx:198-199` | `" Monday"'s nearest neighbours…" dog" is next to " dogs"…` | Token strings in straight quotes instead of `<Tk>` chips; hard to read aloud. | R12 | Render each with `<Tk s=" Monday" />`. Avoid the possessive on a token. |
| L7 | `learn/ch/Threads.tsx:95-97`, `:116` | "The M4 has ten cores"; "on all ten cores" | Words for technical counts; the figure says "10 cores". | 1.5 | "10 cores" |
| L8 | `learn/ch/Arithmetic.tsx:137`; `learn/ch/Block.tsx:191`, `:196` | "Keep sixteen running totals", "twelve blocks", "twelve layers" | Same. | 1.5 | "16 running totals", "12 blocks" |
| L9 | `learn/ch/p0/WhyExpensive.tsx:243` | "Raise the context to 4k tokens" | "4k" in prose vs "4,000-token" at :384. | 1.5 | "Raise the context to 4,096 tokens" |
| L10 | `learn/ch/p0/Hardware.tsx:374`; `site/Landing.tsx:156` | "497 tokens a second"; "tokens a second" | Banned variant (3×). | Vocab | "497 tokens per second" |
| L11 | `learn/ch/Quantization.tsx:115` | "4 bits is a real approximation (10%)" | A bare percentage with no referent. | Clarity | "4 bits gives about 10% rounding error" |
| L12 | `learn/ch/KvCache.tsx:55` | "…the cost per token stopped growing with the text: flat, not rising." | Redundant "not" tail. | R6 | "…the cost per token stopped growing with the text." |
| L13 | `learn/ch/Loop.tsx:148-149` | "We have everything for the full loop." | "We" for reader and author. | R1 | "You now have every piece of the loop." |
| L14 | `learn/ch/Block.tsx:189`, `:200` | "We now have the pieces…"; "…we can peek at the notebook…" | Same. | R1 | "You now have the pieces…"; "…you can read the notebook halfway through…" |
| L15 | `learn/guide/entries.tsx:38` | group "Models & architecture" | Ampersand in a heading. | 1.6 | "Models and architecture" |
| L16 | `learn/Check.tsx:56` | "intuition, not trivia · pick one to see why" | A "not X" slogan as UI microcopy. | R6 | "pick an answer to see the reasoning" |
| L17 | `learn/Check.tsx:42` | feedback "not quite" | Soft and vague when the answer is simply wrong. | Clarity | "no" (pairs with "yes") |
| L18 | `site/Landing.tsx:171` | "⟲ replay" | A symbol in button text is read aloud by screen readers. | Accessibility | `<span aria-hidden>⟲</span> replay` |
| L19 | `ui/Figure.tsx:117` | "Load the engine to drive it with the real thing." | Vague link sentence. | R15 | "Load GPT-2 to run this figure live." |
| L20 | `site/Landing.tsx:194` | kicker "an inference engine written from scratch · and a course in how it works" | "· and" is an odd joint. | Polish | "an inference engine written from scratch, and a course on how it works" |
| L21 | `learn/ch/SpeedOfLight.tsx:19` | machine label "datacentre GPU" | The note names the H100; the label should too. | Vocab | "H100 (datacentre GPU)" |
| L22 | `panels/Ladder.tsx:139` | "M4 speed of light (f32)" | Banned synonym. | Vocab | "M4 speed limit (f32)" |
| L23 | `learn/ch/p0/Hardware.tsx:416` | "Next: the Rosetta stone." | It's a title. Capitalise it consistently with the chapter title "A Rosetta stone". | Consistency | "Next: A Rosetta stone." |
| L24 | `web/README.md:139` | ````live <id>```` | Broken Markdown (four backticks render oddly). | Formatting | `` `live <id>` `` fenced blocks |
| L25 | `docs/WRITEUP.md:768` | "## Open questions / things to verify before publishing" (empty) | An empty heading. | Formatting | Fill it or delete it. |
| L26 | `docs/WRITEUP.md:284-285`, `:479-480` | heading follows a list item with no blank line | Some renderers fold the heading into the list. | Formatting | Add a blank line. |
| L27 | `docs/WRITEUP.md:25`, `:51` vs `:314` | "quantization" vs "quantisation"; `:606-607`, `:714-715` "colored", "gray", "color" | Mixed spelling in one doc. | R11 | British throughout. |
| L28 | `docs/WRITEUP.md:3` | "…written while building so the post writes itself." | Cliché. | Anti-slop | "…written while building, so the post has its facts ready." |

### 2.4 Patterns that repeat site-wide

| Pattern | Count / where | Why it matters | Fix |
|---|---|---|---|
| **Colon chains** ("X: Y, and Z: W") | 25–59% of sentences per chapter contain a colon (see section 4). Worst: Batching 48%, Block 48%, Loop 47%, Quantization 46%, Hardware 41%. | The site's version of the em-dash tic. Sentences look short but carry two or three clauses. | R5: one colon per sentence, only to introduce. |
| **"not X, Y" pivots** | 35 instances of ", not " in course and lab copy (e.g. `concepts.ts:27` "bytes, not maths"; `Threads.tsx:106` "1.7×, not 10×"; `WhatIsInference.tsx:220` "minutes, not milliseconds"; `Hardware.tsx:405`; `Rosetta.tsx:24`, `:35`; `Gpu.tsx:118`; `KvCache.tsx:55`) | Reads as reflexive contrast. Some uses earn it (1.7× vs 10× corrects a real expectation). Most don't. | R6 budget: one per chapter, only against a named misconception. |
| **Closing zingers** | "That's it." (2), "Hold on to that" (2), "That's the frame.", "Speed is bought with accuracy…", "A spin pool assumes you own the machine…", "A correct program can be absurdly wasteful…", "Small-model lessons don't always transfer.", "It buys knowledge without FLOPs, not without bytes." | They repeat the Idea box, which already gives the one-sentence takeaway. | R8: delete. The Idea box is the summary. |
| **Opener templates** | "Here is/Here's…" (4), "So far…", "We now have the pieces", "We have everything", "Time to meet the engine", "Part one ended with a fact" | Each chapter clears its throat before it starts. | Open on the chapter's first fact or question. |
| **"Nobody…" sentences** | 4 (`Embeddings.tsx:186`, `:199`; `Tokens.tsx:204`; `NextToken.tsx:318`) | A dramatic negative where a plain agent sentence works. | "Training set these numbers." |
| **Intensifiers** | just 26, exactly 18, actually 13, really 7, whole 55, almost/nearly 28, every single 3 | "exactly" and "whole" are often real (exact match, whole text). "just", "actually" and "really" rarely are. | Remove "just/actually/really" unless they contrast with a stated belief. |
| **Mixed metaphors** | truck/factory, pantry door, checkout lanes, paint roller, notebook, chefs, assembly line, dishes round a table, delivery worker, seats, bus (WRITEUP) | Two metaphors for bandwidth (truck and pantry door) teach two models of one thing. | 1.1: one metaphor per concept. |
| **Try-its that give away the answer** | 9 of 35 try-its exceed 35 words; most of those state results (WhyExpensive 0.2.1, Batching 17.1, Sampling 7.1, SpeedOfLight 10.1) | Removes the point of trying. | 1.7: imperative plus "Notice…" without the number. |
| **Long captions** | 40 captions, average 65 words, 23 over 60. Longest: SpecDecode 18.1 (113), Attention 5.1 (91), NextToken 1.1 (90), Embeddings 3.1 (86), ManyGpus 19.1 (83). | Readers skip long captions, and the colour legend gets lost. | 1.7: 60 words or fewer, 4 sentences or fewer. Move mechanism to the body. |
| **Colour words without marks** | "Ember:", "Grey:", "Dark bars", "orange", bold coloured words without a `Swatch` (`Batching.tsx:47-48`) | Fails for colour-blind readers and in high-contrast mode. | R14 |
| **Device-specific verbs** | click 17, hover 4 (all in try-its or captions), drag 11 | Touch and keyboard users. | R13: Select/Choose, and "drag (or use the arrow keys)" where sliders support it. |
| **Part 3 heading template** | "The problem / The idea / The result" in ch 12–18 | A fine skeleton, but the headings carry no information in the table of contents or a skim. | R9: "The result: 51× faster". |
| **Repeated framing across part 0** | "enormous read-only blob" (0.1:505, 0.4:121); "per-request state that grows" (0.1, 0.2:383, 0.4:122, checks); "a synchronous loop" | Part 0 repeats the same three-part description in three chapters. | Say it once in 0.1. Later chapters refer back to it. |

---

## 3. Terminology inconsistencies across chapters

| Concept | Variants found (file:line examples) | Use |
|---|---|---|
| token vs word | "word" as the unit: `chapters.tsx:41`, `:49`; `WhatIsInference.tsx:30`, `:80`, `:506`, `:517`; `Hardware.tsx:392` ("hundreds of times per word"); `StartingLine.tsx:13`; `Quantization.tsx:124`; `NextToken.tsx:325`; `Loop.tsx:167-168` ("first word appears"); `Batching.tsx:188` ("median wait for the first word"); `kit.tsx:454`, `:464-465`. 56 prose uses of "word(s)" in chapters, the landing and chapter metadata; about a third mean "token". | token |
| speed limit | "speed limit" 22, "ceiling" 22 (`WhyExpensive.tsx:39`, `:101-104`; `README.md:28`), "speed of light" (`Ladder.tsx:139`, WRITEUP throughout), "limit line" (`Quantization.tsx:130`), "yardstick" (`SpeedOfLight.tsx:148`), "roof" (roofline only) | speed limit |
| memory bandwidth | "bandwidth", "memory speed" (`SpeedOfLight.tsx:64`, `:150`), "read speed" (`Gpu.tsx:130`), "how fast it reads" | memory bandwidth |
| GPU execution units | "lanes" (`Hardware.tsx` throughout) vs "cores" (`kit.tsx:451`, `Gpu.tsx:87`, `kit.tsx:450` "a GPU runs thousands [of threads]") vs "GPU threads" (`Gpu.tsx:100`, `:108`, `:122`) | lane; GPU thread for software |
| engine versions | "stage" 154, "version" 31, "rung" 17 (`registry.tsx:40`, `:185-186`; `App.tsx:43`), "milestone" 4, "step of the climb" (`StartingLine.tsx:36`) | stage; milestone for the six bars |
| site sections | "course", "Foundations" (`LearnApp.tsx:44`, `:155`), "demo page" (`LearnApp.tsx:168`, `Sampling.tsx:174`, `web/README.md:3`), "the Lab"/"the lab" (`StartingLine.tsx:37` vs `Journey.tsx:51`), "engine room" (`App.tsx:21`, `Journey.tsx:58`) | the course, the lab |
| layer indexing | course 1–12 (`Block.tsx:131`, `Attention.tsx:233`) vs lab 0–11 (`InsideToken.tsx:139`, `:175`) | 1–12 everywhere a reader sees it |
| feed-forward | "feed-forward network", "feed-forward", "MLP" (`Block.tsx:53`, `:71`, `:210`; `InsideToken.tsx:151`, `:158`), "feed-forward block" (`entries.tsx:48`) | feed-forward network (FFN) |
| last matrix | "LM head" (`InsideToken.tsx:187`), "lm head" (`DotProduct.tsx:286`), "the final scoring matrix" (`Quantization.tsx:139`), "the final one that scores all 50,257 tokens" | LM head |
| number formats | 32-bit / f32 / fp32; 4-bit / int4 / Q4 / q4_0 / Q4_0; 16-bit / f16 / bf16 (`WhyExpensive.tsx` uses bf16 and int4; `Batching.tsx` Seg label "weights" with q4_0; `SpeedOfLight` "number format"; lab "weight type", `registry.tsx:132`) | 1.3 table; the control label is always "number format" |
| machine | "M4 Mac mini", "our Mac", "the Mac", "our M4", "our Mac mini" (`SpeedOfLight.tsx:152`), "M4" | the M4 Mac mini, then the M4 |
| spec decoding roles | "drafter", "guesser" (`SpecDecode.tsx:183`), "small model", "GPT-2 small guessing"; "check"/"verify"/"validate" (`Rosetta.tsx:33`, `checks.ts:389`) | drafter, big model, check |
| concurrency | users / requests / sequences (lab) / streams (WRITEUP) / seats | user (intuition), request (mechanics) |
| arithmetic | maths (22), arithmetic, compute, FLOPs, "the work", math (2) | arithmetic; FLOPs for counts |
| block | transformer block (`Block.tsx`), quantisation block (`Quantization.tsx:110`), KV block table (`Batching.tsx:261`, `:418`) | layer / block of 32 / page |
| head | attention head, LM head, "KV heads" (`Minis.tsx:71`) | always qualified |
| bandwidth units | 3.35 TB/s (`Hardware.tsx`, `entries.tsx:125`) vs 3,350 GB/s (`ManyGpus.tsx:106`, `checks.ts:80`); NVLink "~900 GB/s per H100" (`kit.tsx:469`, `entries.tsx:101`) vs "450 one way" (`Hardware.tsx:191`, `checks.ts:80`) | TB/s above 1,000; always say "each way" or "total" for links |
| chapter refs | "chapter 05" (5×), "chapter 5" (10×), "Chapter 12" (9×), "ch" (constraint map chips) | "chapter 5"; "Chapter 5" only at sentence start |
| figure refs | "fig 4.1" (course) vs "fig. 1" (lab, 9×) | "fig 4.1"; "lab fig 1" across sections |
| spelling | quantisation 12 / quantization 5 (glossary key and field-guide alias are fine; prose uses should be British); datacentre 5 / datacenter 1; optimis- 13 / optimiz- 1; maths 22 / math 2; grey 13 / gray 0 (WRITEUP has gray, colored) | British (R11) |
| model naming | "GPT-2 small", "GPT-2 124M", "GPT-2" | GPT-2 small |
| definition of token | character (`kit.tsx:426`), symbol (`entries.tsx:42`), byte (`Tokens.tsx:178`) | byte |

---

## 4. Readability statistics

**Method.** The script lives at `/tmp/nicopy/stats.py`. It is not committed; this audit is read-only. It parses each
chapter's JSX:

- It keeps text nodes, `<G>` and `<Tk>` terms and swatch labels.
- It drops code and SVG labels.
- Interpolated numbers become one token ("N").
- It splits paragraphs at block tags and sentences at `. ! ?`.

"Passive" counts a form of *be/get* followed by a past participle, so it slightly overcounts adjectival uses ("is
fixed"). "Colon" is the share of sentences that contain at least one colon. Figure labels and short UI fragments
pull the averages down, so read the **% over 25 words** and **longest** columns as the signal.

| Chapter | Sentences | Avg words | Median | > 25 words | Longest | Passive | With a colon |
|---|---|---|---|---|---|---|---|
| 0.1 What inference is | 57 | 13.7 | 10 | 12% | 37 | 4% | 32% |
| 0.2 Every word reads every weight | 52 | 14.1 | 14 | 6% | 40 | 10% | 31% |
| 0.3 The hardware | 41 | 15.7 | 14 | 17% | 32 | 5% | 41% |
| 0.4 A Rosetta stone | 55 | 16.0 | 17 | 5% | 40 | 18% | 25% |
| 01 Guess the next word | 24 | 12.5 | 12.5 | 0% | 22 | 8% | 17% |
| 02 Tokens | 27 | 14.7 | 14 | 15% | 42 | **26%** | 26% |
| 03 Embeddings | 17 | 14.2 | 15 | 6% | 28 | 12% | 29% |
| 04 Multiply and add | 18 | 12.8 | 13 | 0% | 22 | 11% | 33% |
| 05 Attention | 26 | 14.4 | 13.5 | 8% | 30 | 12% | 38% |
| 06 Layers | 21 | 16.2 | 17 | 19% | 30 | 14% | **48%** |
| 07 Sampling | 18 | 14.7 | 13 | 6% | 29 | 11% | 39% |
| 08 The generation loop | 15 | 10.3 | 10 | 0% | 23 | 7% | 47% |
| 09 Memory or maths | 26 | 12.8 | 12 | 8% | 29 | 15% | 38% |
| 10 The speed limit | 14 | 14.4 | 12.5 | 14% | 39 | 0% | 43% |
| 11 The starting line | 14 | 13.9 | 12 | 14% | 32 | 14% | 29% |
| 12 The KV cache | 19 | 14.2 | 13 | 11% | 33 | **21%** | 21% |
| 13 Faster arithmetic | 31 | 15.5 | 12 | 19% | 39 | 3% | 29% |
| 14 Many cores | 24 | **17.8** | 15 | **25%** | **44** | 12% | 33% |
| 15 Smaller numbers | 28 | 17.0 | 13 | 18% | 42 | 14% | 46% |
| 16 The GPU | 16 | 15.2 | 16 | 12% | 28 | 6% | 25% |
| 17 Batching | 58 | 14.0 | 13 | 5% | **44** | 7% | **48%** |
| 18 Guess ahead | 43 | 15.3 | 14 | 12% | 38 | 12% | 28% |
| 19 Many GPUs | 28 | 14.9 | 15.5 | 7% | 31 | 14% | 29% |
| `checks.ts` (Q, options, why) | 267 | 9.4 | 8 | 0% | 26 | 7% | 10% |
| field guide entries | 194 | 13.0 | 12 | 1% | 27 | 4% | 13% |
| landing | 12 | 15.1 | 13 | 8% | 37 | 0% | 33% |

Captions: 40 figure captions, average **65 words** (23 over 60). 35 try-its, average 25 words (9 over 35).

**Reading the numbers.**

- Average length is healthy everywhere (10 to 18 words; Google and STE targets are about 20 to 25).
- The problem is the tail. Many cores, Faster arithmetic, Smaller numbers and Layers each have about 1 in 5
  sentences over 25 words, and every chapter has at least one sentence of 28 words or more.
- Passive voice is low overall. It clusters in Tokens (26%: "was learned", "is replaced", "is chopped") and the KV
  cache chapter (21%).
- The colon column is the house problem. It is above 40% in eight chapters.
- Check-yourself and field guide copy already meets strict STE limits.

**Longest sentence per chapter** (rewrites for the worst are below; others are covered in section 2):

- 0.1 (37): "Think of a weight matrix as a learned transformation: a vector of 768 numbers goes in, and each output number is a weighted sum…"
- 0.2 (40): "For a 7B model (32 layers, 4,096 numbers each for K and V, 2 bytes) that's 0.5 MB per token, so a 4,000-token conversation holds 2 GB, and unlike the weights it can't be shared between users."
- 0.4 (40): "Where they break is almost always the same place: in inference every step touches everything…, so costs that would be amortised or cached away in a normal service are paid on every token."
- 02 (42): the BPE sentence (M20).
- 10 (39): "It's why people pay for chips with faster memory…" (M37).
- 13 (39): "The same trick did pay off for prefill, which is compute-bound: computing a 4 × 4 block of outputs at once…"
- 14 (44): "The first never-sleeping pool spun its threads flat out…"
- 15 (42): "Weights in a small neighbourhood tend to be similar in size, so they're stored in blocks of 32 that share one scale: …"
- 17 (44): "So the server does what an OS does for processes: it cuts cache memory into fixed-size pages, hands a request…, and keeps a per-request page table…"
- 18 (38): "Every check writes at least one token (the big model's own), so it is never slower in tokens per pass, and the text comes out exactly as…"

### 4.1 Worst paragraphs, rewritten

**ch 14, `Threads.tsx:122-125`** (44-word lead sentence, a colon chain, and a zinger)

> The first never-sleeping pool spun its threads flat out, and on a busy laptop went from 93 to 36 tokens per second
> with ten threads: spinning threads stole the CPU from the one thread still finishing the last slice, and everyone
> waited for it. Spin, then yield, then sleep fixed it. A spin pool assumes you own the machine, and a laptop never does.

Rewrite (4 sentences, longest 18 words):

> The first never-sleeping pool spun its threads at full speed. On a busy laptop, decode fell from 93 to 36 tokens
> per second with 10 threads. The spinning threads took CPU time from the one thread that was still finishing, and
> every thread waited for it. A pool that spins, then yields, then sleeps fixed it.

**ch 17, `Batching.tsx:415-419`** (44 words; three actions in one colon sentence)

> Every live request needs its KV cache, and it grows by one token per step. Reserving a contiguous region big enough
> for the longest possible answer wastes most of it. So the server does what an OS does for processes: it cuts cache
> memory into fixed-size pages, hands a request a new page whenever it fills one, and keeps a per-request page table
> (here called the block table) mapping token positions to physical pages. Attention reads through that table…

Rewrite:

> Every live request has a KV cache that grows by one token per step. If the server reserves room for the longest
> possible answer, most of that room is never used. So the server manages the KV cache like an operating system
> manages memory. It splits memory into fixed-size pages of 16 tokens. It gives a request a new page when the last
> one is full. A per-request **block table** maps each token position to its page. Attention reads through that table,
> so a request's pages can be anywhere in memory.

**ch 15, `Quantization.tsx:109-111`** (42 words, two passives, an undefined "1")

> Store each weight in fewer bits: 16 instead of 32, or 8, or 4. That's quantization. Weights in a small neighbourhood
> tend to be similar in size, so they're stored in blocks of 32 that share one scale: the scale says how big “1” is, and
> each weight is stored as a small whole number of those steps.

Rewrite:

> Store each weight in fewer bits: 16, 8 or 4 instead of 32. This is quantisation. Nearby weights have similar sizes,
> so the engine groups them in blocks of 32. Each block has one scale, which sets the size of one step. Each weight is
> then a small whole number of steps.

**ch 0.4, `Rosetta.tsx:127-133`** (two paragraphs that say the same thing twice)

> The analogies are good enough to reason with. Where they break is almost always the same place: in inference every
> step touches *everything* (all the weights, the whole KV cache), so costs that would be amortised or cached away in a
> normal service are paid on every token. … You know the patterns: caching, paging, batching, sharding, optimistic
> execution, SLOs. Inference applies them to a workload where every request step reads gigabytes, which is why they
> matter so much more.

Rewrite (one paragraph):

> You can reason with these analogies. They usually break in one place. In a normal service, caching or batching
> spreads out a cost. In inference, every step reads all the weights and the whole KV cache. That cost comes back on
> every token, so the same patterns matter much more here.

**ch 10, `SpeedOfLight.tsx:159-161`**: see M37.

**ch 0.1, `WhatIsInference.tsx:505-507`**: see M1.

**fig 18.1 caption, `SpecDecode.tsx:89-95`** (113 words, the longest caption on the site)

Rewrite (55 words, caption template):

> A recorded run on the M4's GPU. The drafter (GPT-2 small, 4-bit) guesses 4 tokens. GPT-2 large (16-bit) checks all
> 4 in one pass. Each group of squares is one pass. Blue squares: guesses the big model kept. Ember: the big model's
> own token, added on every pass. ✕: a guess thrown away.

Move "The text is exactly what GPT-2 large writes on its own" to the body. It is the chapter's main claim and
deserves a sentence there.

**ch 13, `Arithmetic.tsx:142-145`** (two metaphors and a 36-word sentence)

> There was a bonus. Once the program spelled out sixteen independent totals, the compiler grouped them four at a time
> into SIMD instructions, which do the same operation on four numbers at once: a paint roller instead of a brush. The
> compiler hadn't done that before, because adding decimals in a different order can change the last digit of the
> answer, so it isn't allowed to reorder them on its own. Writing sixteen totals gave it permission.

Rewrite:

> There was a second gain. With 16 independent totals, the compiler grouped them into SIMD instructions, which add 4
> numbers at once. It couldn't do this before. Floating-point addition gives slightly different results in a
> different order, so a compiler may not reorder the adds by itself. Writing out 16 totals told it the order was
> allowed.

---

## 5. Top 10 copy fixes, by priority

1. **Make "token" the unit everywhere after chapter 2.** Retitle 0.2 "Every token reads every weight", fix the part 0
   blurb, the `q` lines, "three seconds per word", the TTFT and TPOT glossary lines ("first token") and
   Quantization:124. (H1–H3, section 3 row 1)
2. **Fix stale and wrong navigation and docs.** End-of-course link → `/lab` with new copy (H6). Delete "WebGPU is a
   later rung" (H7) and "a later post in the series" (H8). Update the README "Where we are" table (H12), the
   web/README ask labels, "demo page" and threads notes (H13, H14, H16), and the WRITEUP chapter count (H17).
3. **One set of reference conventions.** "chapter 5" (no zero padding), "fig 4.1" (no full stop), "lab fig 3", and
   layers numbered 1–12 in the lab's InsideToken walkthrough (H4, H5, M72).
4. **Unify the speed vocabulary.** "speed limit" (not ceiling or speed of light), "memory bandwidth" (not memory
   speed), "tok/s" in labels and "tokens per second" in prose, always with "per user" or "total" (section 3).
5. **Break the colon chains.** Apply R5 (one colon, only to introduce) to the eight chapters above 40%. Split every
   sentence over 30 words, starting with the section 4.1 paragraphs and the 10 longest sentences.
6. **Cut captions to the template.** Captions 60 words or fewer, try-its 35 words or fewer. Try-its tell the reader
   what to *notice*, not the number they'll see. Start with SpecDecode 18.1, Attention 5.1, NextToken 1.1,
   Embeddings 3.1, ManyGpus 19.1, WhyExpensive 0.2.1 and Batching 17.1. (H20, section 2.4)
7. **Delete the zingers and openers.** Remove "That's it." ×2, "Hold on to that" ×2, "Or can they?", "If you
   squint", "That's the frame", "Here is…", "Speed is bought…", "A spin pool assumes…", "Small-model lessons…" and
   "It buys knowledge…". The Idea box already gives the takeaway. (M1–M3, M13, M15–M16, M33–M35, M39, M45, M47, M49,
   M56–M58)
8. **Fix the glossary contradictions.** One definition of token (byte, not character or symbol). GPU = lanes, not
   cores. "Kernel" singular. FFN as the name with MLP as the alias. "Model and architecture" heading. (H10, H11, M67,
   section 3)
9. **Accessibility and audience.** Replace hover-only instructions. Name colour and mark together, with no "orange"
   or "dark bars". Hide developer setup text in the ask panel in production. Add `aria-hidden` to button glyphs.
   (H9, H18, H19, L18)
10. **One metaphor per concept and one spelling system.** Keep the truck and factory, the notebook, pages and the
    ring. Drop the pantry door, checkout lanes, paint roller, chefs and dishes. Fix the US stragglers (datacenter,
    optimizer, math, WRITEUP's color and gray) and the straight quotes. (M41, M44, M51, M55, M59, L1–L6, L27)
