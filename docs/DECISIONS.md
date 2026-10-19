# Decisions

The ten decisions from [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) §2, settled on 19 Oct 2026. Each line is the
chosen option and why. Change a decision by adding a dated line under it, not by editing the old one.

| # | Decision | Chosen | Why |
|---|---|---|---|
| 1 | Name and domain | Keep `nanoinfer` for the repo and crate; the site is called **The Climb** | `nanoinfer` collides with 10+ repos and `nanoinfer.live` doesn't resolve, but renaming the crate buys nothing; the site gets a name of its own. |
| 2 | Spelling | **British spelling**, a documented exception to Google style | It's the author's voice and is already used throughout; one rule beats a mix. |
| 3 | Prose face | **Serif prose** (Source Serif 4, 18px), mono for headings, UI, numbers and code | Long-form reading is easier in a serif; mono stays where alignment and "this is a measurement" matter. |
| 4 | Scope of Parts 4–5 | **Build what this engine measures or pure arithmetic can show; link out for production ops** | Batching/scheduling, KV as a memory system, MoE and the memory-budget game are built; disaggregation, routing and autoscaling link out; TP/PP wait for measured numbers. |
| 5 | Modern model | **After launch.** Launch on GPT-2 with a calculated 7B line beside each measured number | Generality is M–L work; a calculated line answers "is this a toy?" without delaying launch. SmolLM2 is the second post. |
| 6 | Distributed track | **After launch**, straight after the modern-model loader and before the server | Strongest infra signal, and it starts on one Mac ("TP makes GPT-2 small slower" is publishable); rented GPUs last. |
| 7 | Open source | **MIT for code, CC BY 4.0 for prose and figures**; public `github.com/joshuaramkissoon/nanoinfer`; Vercel; a git tag per stage | MIT matches `Cargo.toml`; CC BY lets educators reuse the course with credit; tags let readers check out any stage. |
| 8 | Select-to-ask tutor | **Opt-in**, naming the model and who pays; hidden when there's no API key | Useful for readers, but AI answers must be disclosed and never break the page when unconfigured. |
| 9 | Headline metric | **"% of speed limit"**, not "2,200× faster"; name the naive baseline in the same sentence as any multiple | A percentage of a computed limit is honest and comparable; a multiple over a naive baseline flatters. |
| 10 | Launch | **Write-up as a regular HN post with the live engine embedded → Show HN for the site 1–2 weeks later → r/rust and r/LocalLLaMA → a 20-second X clip of the stage switcher**; "faster than MLX/llama.cpp" stays out of titles | Lead with the substance; the comparison invites a fight the losses (prefill) would lose. Slots after the existing X queue. |
