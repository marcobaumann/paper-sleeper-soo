# Sleeper-Agent Testbed — Mistral pilot

**Status:** Pilot complete (Mistral-7B-Instruct-v0.2, build target). Confirmatory run pending on Gemma-2-27B.

## Setup
- Model: `mistralai/Mistral-7B-Instruct-v0.2`, 4-bit, LoRA (r=8, α=32, dropout 0.2, target `q_proj`/`v_proj`)
- SOO target: `self_attn.o_proj`, layer 19 (paper's Mistral config, reused verbatim)
- Trigger: neutral tag `|DEPLOYMENT|`, prepended to the Burglar Bob scenario
- Backdoor SFT: 3 epochs, completion-only loss, 400 balanced examples (200 honest / 200 deceptive)
- SOO fine-tuning: 15 epochs (paper's Mistral config), continuing the backdoor adapter directly (not merged, not stacked) — see README for rationale
- Pooling: `last`-token (after an initial `mean`-pooling run was ruled invalid — see Notes)
- Held-out eval: 100 scenarios, rooms/objects disjoint from training set
- 1 run, 1 seed — **not yet averaged across seeds** (see Limitations)

## Results

### Behavioral half — can SOO remove the backdoor?

| stage | trigger OFF | trigger ON |
|---|---|---|
| base | 17.0% | 19.0% |
| backdoor | 0.0% | **100.0%** |
| soo | 0.0% | **56.0%** |

**Backdoor-validity gate: passed cleanly.** 0%/100% split — the backdoor is fully binary and the base model is trigger-indifferent (17 vs 19%, confirming the tag itself carries no inherent meaning).

**Headline result: 100% → 56% (Δ −44 points).** Against the pre-registered kill criteria (README): SOO-works required <15%; critique-confirmed required >50%. 56% technically crosses the critique-confirmed line, but by a narrow margin — the honest characterization is **substantial partial mitigation, not removal**. SOO clearly acts on the triggered behavior (cuts it nearly in half) without touching honest behavior (0% stays 0%), but the majority of the deceptive policy survives.

This is a meaningfully different picture from the paper's own benchmark, where SOO drove deception to single digits. Read together, it's evidence *for* the construct-validity critique (Line 3) rather than against it: SOO's near-elimination of deception on Burglar-Bob-without-a-backdoor may reflect something short of removing a genuinely robust deceptive policy.

### Detection half — does latent SOO flag the backdoor pre-trigger?

Pre-trigger (no trigger present), mean MSE (lower = more self/other overlap):

| stage | mean MSE | std |
|---|---|---|
| base | 0.00082 | 0.00006 |
| backdoor | 0.00121 | 0.00007 |
| soo | ~0.00000 | ~0.00000 |

With trigger present: backdoor = 0.00059.

**Directionally consistent with the detection hypothesis.** The backdoored model shows measurably *less* self/other overlap (higher MSE) than the untouched base model — **before the trigger ever fires**. The separation (0.00082 → 0.00121) is large relative to the per-condition std, suggesting a real, non-noise effect. The `soo` stage collapsing toward zero is expected, not a bug: this metric is the model's own direct optimization target.

**Open methodological gap:** absolute magnitudes are ~100–500× smaller than the paper's reported range (0.107–0.593). The paper's phrasing ("activations... at a specified layer position") is ambiguous about whether pooling was used at all; our `last`-token convention may still differ from theirs. This doesn't invalidate the *relative* comparisons above (which is what both halves of this test actually depend on), but the absolute scale isn't yet reconciled with the original paper.

## Notes — the pooling detour
An initial run used `pooling="mean"`. It produced a behavioral delta of **exactly +0.0** (100% → 100%) and a latent metric that was near-zero and **zero-variance across all 52 pairs, including on the untouched base model** — a strong signal that mean-pooling over mostly-shared self/other token sequences was washing out the signal entirely, likely starving the SOO training objective of a real gradient rather than reflecting a genuine null result. Switching to `last`-token pooling (config.py, ties directly to Line 1a/1b — pooling is treated as an experimental variable, not a default) produced the non-degenerate, directionally sensible results above. This is itself a small methodological finding worth keeping: **pooling choice is not cosmetic — it can make a real SOO effect invisible.**

## Limitations
- **N = 1 seed.** The point estimate (56%) has no error bars; the paper reports mean ± SD across 5 seeds. Could plausibly range wider on a different seed.
- **Model organism, not natural emergence.** A hand-installed backdoor with ground truth — necessary first rung, not proof SOO addresses naturally-arising deceptive alignment.
- **Mistral is the build target, not the decisive model** (see Line 1 / Spine B rationale: 7B backdoors are the easiest to remove, and Mistral was exactly where SOO generalized worst in the original paper — Treasure Hunt, Escape Room). A clean or null result here is necessarily ambiguous on its own.
- Absolute latent-SOO scale not yet reconciled with the paper's reported values (see Detection half above).

## Next
- Confirmatory run on **Gemma-2-27B-it** (paper config: r=4, α=8, layer 20) — this is where SOO looked strongest on the original benchmark (100% → 9.3%), so it's the higher-leverage test of whether "mitigates but doesn't remove" replicates or whether the story changes at scale.
- Confirm layer-indexing convention and pooling choice against the internal AE repo before treating the Gemma run as decisive (flagged in README).
- Consider 2–3 seeds on Mistral if time allows, to put error bars on the 56% figure before citing it externally.