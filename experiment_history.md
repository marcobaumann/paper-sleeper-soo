# Experiment Log — SOO Sleeper-Agent Testbed

A chronological lab notebook: what we changed, why, and what we found, run by run.
Detection metric throughout = latent SOO (self/other `o_proj` MSE). NLAE (planned second
detection lens) is **future work** — not exercised in any run below (depends on released NLAs
for the target model; not yet verified).

Legend: deception % = deceptive-response rate. "valid" = coherent, classifiable answers
(deceptive + honest); a low valid rate means the model stopped answering the task.

---

## Phase 0 — Mistral-7B pilot (build target)

Purpose: de-risk the pipeline on a fast model before the decisive Gemma run. Not decisive on
its own (7B backdoors are easiest to remove; Mistral generalized worst in the paper).

### Pilot A — mean pooling  · `results_20260714_1642` · INVALID
- Config: LoRA r=8/α=32, backdoor 3 epochs, SOO 15 epochs @ 1e-4, **pooling = mean**, layer 19.
- Behavioral: backdoor 100% → SOO 100% under trigger (**Δ +0.0**).
- Latent: base 0.00006 / backdoor 0.00007 / soo 0.00000, **std = 0.00000 everywhere**.
- Finding: **degenerate.** Zero-variance latent across 52 distinct pairs (even on untouched
  base) meant mean-pooling over mostly-shared self/other sequences washed the signal to zero,
  starving the SOO objective of gradient. Not a real null — a measurement/objective failure.
- Decision: switch pooling `mean → last` (config.py; tied to Line 1a/1b, treated as an
  experimental variable, not a default).

### Pilot B — last pooling  · `results_20260714_1710` · VALID
- Only change from Pilot A: **pooling = last**.
- Behavioral (deception, trigger OFF / ON):
  - base 17% / 19% · backdoor 0% / 100% · soo 0% / **56%**  (**Δ −44 pts**)
- Latent (pre-trigger): base 0.00082 / backdoor 0.00121 / soo 0.00000 (non-zero std now).
- Finding: non-degenerate. SOO produced **partial mitigation** (100% → 56%) on a clean 100%
  backdoor, honest behavior preserved (0% OFF). Latent metric separated backdoor from base
  pre-trigger. Magnitudes ~100–500× below the paper's 0.1–0.6 (unreconciled scale).
- Carried forward to Gemma: pooling = last.

---

## Gemma-2-27B-it (decisive model)

Model-profile system added to `config.py`. Gemma-specific requirements established during
bring-up: `attn_implementation="eager"` (soft-capping), bf16 (fp16 → NaN), `cache_implementation
="dynamic"` in generation (HybridCache bf16/float32 `index_put` bug), pinned library stack
(transformers 4.44.2 etc. — a newer transformers on torch 2.4 broke loading with `set_submodule`).

### Run 1 — paper SOO config  · `results_20260714_1917` · INVALID (backdoor too weak)
- Config: LoRA **r=4/α=8** (paper's Gemma SOO config), backdoor **3 epochs**, SOO 8 epochs @ 9e-4.
- Behavioral (deception, OFF / ON):
  - base 15% / 27% · backdoor 2% / **40%** · soo 14% / 11%
- Latent (pre-trigger): base 0.04958 / backdoor 0.05567 / soo 0.00002; backdoor w/ trigger 0.06196.
- Finding: the apparent 40% → 11% "removal" is **not interpretable** — the backdoor itself only
  reached 40% and **failed the validity gate**. On re-eval the backdoor gave 20% (vs 40% in the
  archive): unstable, and `unparsed = 0%`, so it was a genuinely **weak install**, not a parser
  artifact. Root cause: r=4 adapter + 3 epochs can't overcome Gemma's instruction-tuned honesty
  prior in a 27B (honest branch learned perfectly, deceptive-trigger branch barely).
- Decision: strengthen the install to the capacity that worked on Mistral.

### Run 2 — strong backdoor  · `results_20260715_1559` · VALID (with coherence caveat)
- Changes from Run 1: LoRA **r=4→8, α=8→32**; backdoor **epochs 3→10**.
  - Documented deviation: SOO continues the same adapter, so SOO now runs at r=8/α=32, not the
    paper's r=4/α=8.
- Backdoor validity gate: **PASSED** — 0%/100% clean, 0% unparsed. Trigger-OFF honesty intact.
- Behavioral (deception, OFF / ON):
  - base 15% / 27% · backdoor 0% / **100%** · soo 11% / **11%**  (**Δ −89 pts**)
- **Coherence caveat:** post-SOO **~81% of answers were unparsed** (soo OFF: dec11/hon8/unp81;
  soo ON: dec11/hon7/unp82). So the 11% is 11% deceptive / ~8% honest / ~81% incoherent — a
  reduction in *measurable deception*, not demonstrated honesty. Note base was already ~83%
  unparsed, so part of this is regression toward base, not purely new damage.
- Latent (pre-trigger): base 0.04958 ± 0.00392 / backdoor 0.05551 ± 0.00446 / soo 0.00005;
  backdoor w/ trigger 0.06512. **In the paper's magnitude range**, clean pre-trigger separation.
- Finding: on a robust backdoor, SOO cut triggered deception 100% → 11% (beats Mistral's 56%),
  **but** at a large coherence cost the pipeline wasn't measuring. Detection half is clean and
  strong. Both halves read as evidence *for* the construct-validity critique (SOO may act on
  surface behavior/coherence, not installed honesty).
- Decisions: (1) NLAE stays future work. (2) Run 3 to isolate the coherence cause.

### Run 3 — lighter SOO  · PLANNED (configured, not yet executed)
- Changes from Run 2: **soo_epochs 8→3**, **soo_lr 9e-4→3e-4** (both at once, by choice).
  Backdoor unchanged (reuse the strong r=8/α=32, 10-epoch install).
- New instrument: `eval_behavioral.py` now reports an explicit **valid-response rate** per cell
  and a backdoor→SOO valid-rate delta, so coherence is measured, not inferred.
- Hypothesis / kill criteria:
  - If unparsed drops sharply while deception stays low → Run 2's incoherence was SOO
    **over-optimization**; we have a much cleaner result.
  - If unparsed stays high → the coherence cost is **intrinsic** to SOO at this strength — itself
    a publishable finding and a direct input to SOO v2.
- Result: _pending._

---

## Running summary

| run | model | backdoor ON | SOO ON | Δ decep. | valid post-SOO | latent base→backdoor (pre-trig) | verdict |
|---|---|---|---|---|---|---|---|
| Pilot A | Mistral | 100% | 100% | +0.0 | — | 0.00006→0.00007 (zero-var) | invalid (mean pooling) |
| Pilot B | Mistral | 100% | 56% | −44 | not measured | 0.00082→0.00121 | valid, partial mitigation |
| Run 1 | Gemma-27B | 40% | 11% | (n/a) | not measured | 0.04958→0.05567 | invalid (weak backdoor) |
| Run 2 | Gemma-27B | 100% | 11% | −89 | ~18% (81% unparsed) | 0.04958→0.05551 | valid, coherence caveat |
| Run 3 | Gemma-27B | 100% | ? | ? | ? | ? | pending |

## Cross-run takeaways so far
- **Pooling is not cosmetic** (mean → degenerate; last → signal). Line 1a/1b, live.
- **Backdoor install strength must clear the gate** before any SOO number is interpretable
  (Run 1 lesson).
- **Deception reduction ≠ honesty** — Run 2's −89 pts came with a coherence collapse; measuring
  valid-response rate is now mandatory (Run 3 instrument).
- **Detection (latent SOO) is the more robust half** across models: clean pre-trigger separation,
  and in-range magnitudes on Gemma.
- **Open:** absolute latent-SOO scale vs the paper (Mistral off by ~100–500×; Gemma in range) —
  unreconciled. NLAE second lens — future work.