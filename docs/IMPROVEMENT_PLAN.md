# Improvement plan

Synthesised 19 Oct 2026 from five reviews in `docs/review/`: 01 course, 02 copy, 03 design, 04 engine and
05 landscape. **(?)** marks a claim the reviewer flagged as an estimate or unverified.

## 1. Strategic read

**Best at:** one law (bandwidth ÷ bytes per token), taught by a real engine climbing towards it (01, 05).

**Differentiator:** you can run a from-scratch engine in your browser at every optimisation stage. Each stage is
measured against a computed speed limit, beside a matched-format head-to-head with MLX and llama.cpp, losses
included. "Inference for systems engineers" is already claimed by others (05).

**What it should be:** the measured, explorable companion to *How to Scale Your Model* and Chan's yalm post.

**What it shouldn't be:** a production engine, a 3D visualiser or a serving-ops handbook. Link out for those.

**Name:** `nanoinfer` collides with 10+ repos, and `nanoinfer.live` doesn't resolve (decision 1).

**Launch bar:**
- Phase 0 done.
- The ★ items in Phase 1 done.
- Works in Firefox and Safari, falling back to recorded runs.
- Public repo with a current README, a LICENSE and green CI.
- Head-to-head re-run with clean shas and a randomised order.

## 2. Decisions Josh must make

1. **Name and domain.** (a) Keep `nanoinfer` and buy a domain. (b) Keep `nanoinfer` for the repo and crate, and give
   the site its own name: *The Climb*, *Bandwidth Bound* or *Every Weight, Every Token*.
   **Recommend (b), *The Climb***. Avoid *Speed of Light*, which clashes with 02's "speed limit". Check domains.
2. **British spelling as a documented exception to Google style** (02 R11). **Recommend yes.**
3. **Serif prose or all-mono** (03). Serif means Source Serif 4 at 18px for prose, with mono kept for headings, UI,
   numbers and code. All-mono means 16px with 1.7 leading and a 36rem measure. **Recommend serif.** Prototype on 0.2 and 9 first.
4. **Scope of new Parts 4–5 (serving, beyond one chip).** 01 proposes 9 new chapters. 05 says link out instead.
   **Resolution: build what this engine measures or pure arithmetic can show, and link out for production ops.**
   - Build: split chapter 17 into batching and scheduling (on the real `serving.json`); a chapter on the KV cache as
     a memory system; MoE (promote `MoeLab`); and the memory-budget game.
   - Link out: disaggregation, routing, autoscaling.
   - Wait: TP/PP chapters wait until the distributed track has measured them.
5. **Modern model before or after launch.** **Recommend after.** Launch on GPT-2, with a *calculated* 7B line next
   to each measured number. SmolLM2 is the second post (M–L, then L; 04).
6. **When to start the distributed track.** **Recommend straight after the modern-model loader**, before the server. Strongest infra-role signal; starts on
   one Mac ("TP makes GPT-2 small slower" is publishable). Rented GPUs last.
7. **Open source.** **Recommend:**
   - MIT for code (matches Cargo).
   - CC BY 4.0 for prose and figures.
   - A public `github.com/joshuaramkissoon/nanoinfer` (no remote exists yet).
   - Hosting on Vercel.
   - A git tag per stage.
8. **The select-to-ask AI tutor.** Options are on, opt-in, or cut.
   **Recommend opt-in**, saying which model it uses and who pays. Hide it when there's no API key.
9. **Headline metric.** **Recommend "% of speed limit" over "2,200× faster".** Name the naive baseline in the same
   sentence as any multiple.
10. **Launch.** **Recommend this order:**
    1. The write-up as a regular HN post with the live engine embedded.
    2. Show HN for the site 1–2 weeks later.
    3. r/rust and r/LocalLLaMA.
    4. A 20-second X clip of the stage switcher.

    Keep "faster than MLX/llama.cpp" out of titles.
    Slot after the X plan's existing queue.

## 3. Phase 0: correctness and credibility

Every item is small (S) unless marked M. `ch/` = `web/src/learn/ch/`, `lab/` = `web/src/panels/`.

| # | What | Where | Src |
|---|---|---|---|
| 0.1 | KV cache "flat, not rising": attention still reads O(n) per token. | `ch/KvCache.tsx:55` | 01, 02 |
| 0.2 | "8×… around 1,200" (8 × 243 ≈ 1,950; figs say ~7×). Show one derivation **(?)** maybe 98 GB/s ÷ ~80 MB. | `ch/Quantization.tsx:130` | 01 |
| 0.3 | "Close to 105 GB/s" is really ≈73 GB/s (69%). | `ch/Gpu.tsx` | 01 |
| 0.4 | 5.8× in the prose vs 5.3× in the climb. Name both baselines. | `ch/Arithmetic.tsx` | 01 |
| 0.5 | Determinism quiz is wrong for production (batch-dependent kernels, non-associative adds). Link ch 13. | `learn/checks.ts` (0.1 q3) | 01 |
| 0.6 | Ridges 1.6 / 41 / 295 FLOP/B unreconciled: add a ridges-by-machine strip. | `ch/BytesFlops.tsx` | 01 |
| 0.7 | Fig 0.2.2 "unchanged" vs 0.2.3 "150 vs 240": caption 0.2.2 "ignoring KV reads". | `ch/p0/WhyExpensive.tsx` | 01 |
| 0.8 | Climb's one "limit 243" line is passed by later bars: per-bar limit ticks, "% of own limit". | `learn/Climb.tsx` | 01, 03 |
| 0.9 | H100 989 TFLOP/s is tensor cores, not lanes; FP8 doubles it. | `ch/p0/Hardware.tsx` | 01 |
| 0.10 | Idea "make that fast…" contradicts Part 2. | `ch/DotProduct.tsx:348` | 02 M25 |
| 0.11 | Glossary: token = byte; GPU = lanes; 3.35 TB/s everywhere; NVLink "each way"/"total". | `kit.tsx`, `guide/entries.tsx`, `ManyGpus.tsx:106`, `checks.ts:80` | 02 |
| 0.12 | Hero shows 97 tok/s above "2,200×": state conditions or show slow and fast side by side. | `site/Landing.tsx` | 01, 05 |
| 0.13 | Lab fig 05 "223% of ceiling": limit must follow the run's backend and dtype; >100% impossible. | lab fig 05 | 01 |
| 0.14 | Lab fig 06: GPU dot on CPU roofs; per-backend roofs. | `lab/Roofline.tsx` | 01 |
| 0.15 | Lab fig 04 hangs on "waiting" after WebGPU: explain, offer CPU switch; fix dark-mode caption. | lab fig 04 | 01, 03 |
| 0.16 | Lab layers 0–11 vs course 1–12. | `lab/InsideToken.tsx:139,175` | 02 H5 |
| 0.17 | End-of-course "demo page" link goes to `/`: point to `/lab`. | `learn/LearnApp.tsx:166` | 02 H6 |
| 0.18 | Delete "WebGPU is a later rung", "a later post in the series". | `App.tsx:41`, `ManyGpus.tsx:125` | 02 H7–8 |
| 0.19 | Retired names "Foundations", "demo page". | `LearnApp.tsx:44,155`, `Sampling.tsx:174` | 02 |
| 0.20 | Ask panel shows `TOGETHER_API_KEY` setup publicly: dev only. | `ask/AskLayer.tsx:419` | 02 H9 |
| 0.21 | Stale `web/README`; **check if any figure uses the busy-machine recording** (M to re-record). | `web/README.md` | 02 H13–16 |
| 0.22 | WRITEUP "19 chapters, 3 parts"; empty heading. | `docs/WRITEUP.md:768–774` | 02 |
| 0.23 | `head_format` misses `Q8Prefill` (173 vs 139.5 MB, ~20% off): fix, test `decode_bytes ≈ bytes()`, re-bench. | `crates/core/src/stages/mod.rs:161` | 04 |
| 0.24 | No `LICENSE`; no `rust-version = "1.88"`. | root, `Cargo.toml` | 04 |
| 0.25 | README stops at stage 1: milestones, head-to-head with losses, site link, "why Rust" (M). | `README.md` | 02, 04, 05 |
| 0.26 | Commit untracked `web/public/lab/` or not. | repo | 04 |

## 4. Phase 1: polish to launch quality

★ marks a launch blocker. Two conflicts are resolved here:

- **Captions.** 03's form wins (prose caption, `→ Try` line, footnoted sources, no uppercase labels); 02's limits
  apply inside it (≤60 words, try ≤35, say what to notice, never the result).
- **Climb chart.** 03's lollipop (batching total as a hollow ring) absorbs 01's per-bar limit ticks.

| Item | Why (src) | Effort | Impact | Deps |
|---|---|---|---|---|
| ★ Contrast tokens: muted, light accent `#a3471a`, and `--m-line-ui` at 3:1 or better | Fails AA in both themes (03) | S | H | — |
| ★ `useFrame` respects reduced motion, plus a pause control | 60 fps under `reduce` today (03) | S | H | — |
| ★ Prose face and a 7-step type scale (25 sizes today) | Readability (03) | M | H | Dec. 3 |
| ★ Remove the box from `<Fig>`: 52rem breakout, `alt`, one row of controls, captions as above | Every figure is a 640px card (03, 02) | M | H | Type scale |
| ★ Strip the chrome: Idea box, quiz rows, prev/next, a single radius, dots, pills | AI-slop tells (03) | M | H | `<Fig>` |
| ★ Replace bars on log axes with dots (0.2.1, the climb); fix 9.2 ticks | Anti-pattern (03, 01) | M | H | 0.8 |
| ★ Mobile: 16px gutters, full-bleed figures, 40px touch targets, a skip link, a numbered rail, an in-chapter TOC | (03, 01) | M | M | — |
| Text colour: one `.k` per chapter at most; stats in the foreground colour | (03) | S | M | — |
| Field guide as a dictionary | Wall of 53 cards (03) | M | M | — |
| Lab on the course template: 3 tiles, tabs | A second visual language (03, 01) | M | M | `<Fig>` |
| Split fig 3.1; clean up 9.1/9.2; annotate 17.3; table-view toggle; update DESIGN.md (03 §7) | (03) | M | L | `<Fig>` |
| ★ `web/STYLE.md` from 02 §1: rules R1–R17 and the controlled vocabulary | One meaning per word (02) | S | H | Dec. 1, 2 |
| ★ "Token" everywhere; speed vocabulary ("speed limit", tok/s per user or total); reference style ("chapter 5", "fig 4.1") | Terms drift (02) | M | H | STYLE.md |
| ★ Captions and try-its to the template (18.1, 5.1, 1.1, 3.1, 19.1, 0.2.1 and 17.1 first) | 65 words on average; try-its give the answer away (02) | M | H | `<Fig>` |
| ★ No hover-only instructions; name colours together with their marks | R13, R14 (02) | S | M | — |
| Colon chains, long sentences, zingers, one metaphor per concept, curly quotes | House tics (02) | M | M | STYLE.md |
| `checks.ts`: plausible distractors, matched lengths, no stems that leak the answer | (02, 01) | S | M | — |
| Move the Rosetta stone into the field guide; trim Part 0 (0.2.3 to batching, 0.2.4 to the guide) | Part 0 duplicates Part 2 (01) | M | H | — |
| GPT-2 shape sheet and a KV-per-token formula in chapters 5–6 | GQA and TP can't be reasoned about without them (01) | S | H | — |
| One paragraph each: decode attention doesn't batch; CUDA graphs; FP8 and the MLX 4-bit-head trap; spec decoding fades with batch, EAGLE/MTP | Expert gaps (01) | S | H | — |
| A calculated 7B line; further-reading links; an about page crediting Chan, llama2.c and Bycroft | Pre-empts "toy" and "how does it compare" (05) | S | M | — |
| ★ CI: macOS (fmt, clippy, tests with `REQUIRE_WEIGHTS`), a Linux build, a wasm build; print skip counts | Tests can pass while testing nothing (04) | S–M | H | 0.24 |
| ★ Test gaps: HF decode check per CPU stage, context-300, two sessions, loader fuzzing | Decode kernels for stages 2–15 aren't checked against HF (04) | S | H | — |
| ★ Bench hygiene: refuse dirty trees (536/541 dirty), record rustc and load, k prefills, pre-touch memory | (04) | S–M | H | — |
| ★ Head-to-head in randomised order; QoS parity for llama.cpp **(?)** | Fixed order favours us (04, 05) | S | H | Hygiene |
| ★ A streaming-mode Metal decode number next to 787 tok/s **(?)** est. ~715 | Pipelined decode isn't chat (04) | S | H | — |
| Pool panic safety, a serial-fallback warning, `unsafe` `Buf` access, `Recipe::validate()` | Hang or UB on a kernel bug (04) | S | M | — |
| ★ Refresh ARCHITECTURE, add `docs/STAGES.md`, label generated `bench/` files, tag each stage | Stale (04) | S–M | M | — |
| ★ Test Firefox, Safari 26, the no-WebGPU path and the ~130 MB download | The top complaint at similar launches (05) | S | H | — |

## 5. Phase 2: make it genuinely valuable

| Item | Why (src) | Effort | Impact | Deps |
|---|---|---|---|---|
| Predict-then-reveal slot in `<Fig>` | Watching becomes thinking (01) | S | H | `<Fig>` |
| Frontier plot: tok/s per user vs per GPU, with an SLO line | The industry's chart; data exists (01) | S | H | — |
| Live roofline as chapter 9's main figure | One figure that unifies the course (01) | M | H | 0.6 |
| Memory-budget game and a KV context slider | Capacity is the missing half (01, 05) | M | H | Shape sheet |
| "One vector or many" toggle on fig 4.1 | Shows where reuse comes from (01) | S | H | — |
| Per-op time per token, per stage | Amdahl inside a token (01) | M | M | — |
| Scheduler game on `serving.json` (start with a policy picker) | The best "aha" for this reader (01) | M→L | H | Ch 17 split |
| Engine race lanes; a spec-decoding k/α tuner | (01) | S/M | M | — |
| **Graded exercises**: per-stage stubs, graded by the HF tests, your number on the climb | Turns the portfolio piece into a resource (05) | L | H | CI, STAGES |
| Spaced review; embeddable figures; `llms.txt` | Retention; educators (01, 05) | M | M | Dec. 7 |
| Opt-in crowd-sourced tok/s measurements **(?)** needs a privacy design | Turns the demo into a dataset (05) | M | M | Backend |
| New chapters: split 17 (batching / scheduling); KV as a memory system (paged, prefix, GQA/MLA); MoE; "Fit it" | Per decision 4 (01) | M each | H | Figures above |
| `StageSpec` table; collapse the `Session::step` arms | Prevents another `head_format` bug (04) | S | M | 0.23 |
| Shared `Arc<Packed>` weights | One weight copy per process; the server needs this (04) | M | H | StageSpec |
| `metal-pipelined-f32` stage | 176 → ~190–200 tok/s **(?)**, beats llama.cpp's 192 (04) | S | M | — |
| CPU `f16-fast` / `f32-fast` stages | 328 → ~360–375 tok/s **(?)** (04) | S | M | StageSpec |
| Metal f16-staged prefill GEMM | 1.4–1.7× **(?)**; our biggest loss (04) | M | H | — |
| CPU i8mm prefill, with feature detection | Q8 3.5k → 5–6k tok/s **(?)** (04) | M | M | — |
| Prompt-lookup speculative decoding | 1.3–2.5× on structured text **(?)**, no draft model (04) | S–M | M | — |

## 6. Phase 3: the original ambitions

| Item | Why (src) | Effort | Impact | Deps |
|---|---|---|---|---|
| "Museum" policy: freeze the GPT-2 stages; one current path per backend (`Batch` for Metal) | The graph is written out 7 times (04) | S | H | — |
| Generality 1: Config v2, bf16 and shards, `config.json`, packing while loading, `tokenizer.json` | Unlocks modern models (04) | M–L | H | Museum |
| Generality 2: a generic block with RMSNorm, RoPE, GQA, SwiGLU; SmolLM2 → Qwen2.5-0.5B → Llama-3.2-1B, gated on perplexity | A useful engine (04) | L | H | Gen. 1 |
| `crates/dist`: Communicator (shared memory, TCP, emulated link), ring and tree all-reduce, a bus-bandwidth bench | Starts the distributed track (04) | M | H | — |
| TP as `ShardedLinear`, then PP, on one Mac and then two | Measures law 10 (04) | L | H | Gen. 2 |
| Drag-to-shard and ring chapters, on measured numbers | (01) | L | M | TP/PP |
| `crates/serve`: per-row sampling, OpenAI-compatible SSE, cancellation, KV admission, depth-2 pipelining, then prefix caching | Something a client can call (04) | L | H | Shared packed |
| Toy disaggregation; CUDA backend on rented GPUs; capstone scenario | (04, 01) | L | M | dist, serve |

## 7. Execution order for agents

**Heavy queue:** builds, tests, benches, head-to-head and recordings run one at a time. Light edits run in parallel
on disjoint files only.

**Wave 1 (Phase 0): three light lanes.**
- **A:** `ch/**`, `checks.ts`, `kit.tsx`, `entries.tsx`, `Climb.tsx`, `Landing.tsx` (items 0.1–0.12).
- **B:** `panels/**`, `App.tsx`, `LearnApp.tsx`, `AskLayer.tsx`, `web/README`, `WRITEUP` (items 0.13–0.22).
- **C:** `LICENSE`, `Cargo.toml`, `README`, `stages/mod.rs` (items 0.23–0.26).
- Heavy jobs: C's tests, then the clean-tree q8-prefill bench, then a web build and screenshots.

**Wave 2 (Phase 1).**
1. A single design agent goes first. It owns `styles.css`, `kit.tsx`, `ui/Figure.tsx` and `DESIGN.md`.
2. Meanwhile, two lanes run in parallel: `STYLE.md`, and the engine lane (CI, tests, bench hygiene, pool safety,
   docs, tags). The engine lane touches nothing in `web/`.
3. After `<Fig>` lands: one agent per chapter group (Part 0 / 1 / 2–3 / lab and guide), copy and figures together.
4. Heavy jobs run in this order: the test suite, the clean bench sweep, the randomised 3-round head-to-head, the
   streaming Metal number, the browser matrix.

**Wave 3 (launch).** Settle decisions 1 and 7–10, draft the post and the clip, then make the repo public.

**Wave 4 (Phase 2).**
New figures are new files: 2–3 web agents at once (predict slot and ch 17 split alone). Engine stages code in
parallel, bench serially.

**Wave 5 (Phase 3).** One step at a time:
1. The museum policy.
2. Generality 1.
3. Generality 2. `dist` can be written alongside it.
4. TP/PP.
5. The server.
