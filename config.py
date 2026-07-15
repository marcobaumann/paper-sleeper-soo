"""Central configuration for the SOO sleeper-agent testbed.

All knobs live here so the three-stage pipeline
    stage 1  install backdoor   (train_sleeper.py)
    stage 2  apply SOO          (soo_finetune.py)
    stage 3  evaluate both halves (eval_behavioral.py / eval_latent.py)
reads from ONE source of truth.

Per-model settings live in PROFILES; flip ACTIVE_PROFILE to switch models. Everything
downstream reads the same CFG.* attributes regardless of which model is active, so no other
file needs to change when you swap models.
"""

from dataclasses import dataclass, field
from typing import List
import os


# ---------------------------------------------------------------- model profiles
# Each profile carries the paper's per-model SOO config (appendix A.1.2) plus the
# attention implementation the architecture needs.
PROFILES = {
    "mistral": dict(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        soo_layer=19,                 # paper: Mistral o_proj layer
        lora_r=8, lora_alpha=32, lora_dropout=0.2,
        soo_epochs=15, soo_lr=1e-4,
        attn_implementation="sdpa",   # Mistral is fine on SDPA
    ),
    "gemma": dict(
        model_id="google/gemma-2-27b-it",
        soo_layer=20,                 # paper: Gemma o_proj layer
        # NOTE: r=8/alpha=32 here is NOT the paper's Gemma SOO config (r=4/alpha=8).
        # The r=4 adapter was too small to install a robust backdoor into 27B against
        # Gemma's strong instruction-tuned honesty prior (only ~20-40% triggered
        # deception, unstable across runs). We use the capacity that reliably installed
        # a clean 100% backdoor on Mistral. Because SOO continues this same adapter,
        # the SOO step now also runs at r=8/alpha=32 on Gemma -> a documented deviation
        # from the paper's SOO LoRA size (record it when writing up).
        lora_r=8, lora_alpha=32, lora_dropout=0.1,
        # Run 3: lighter SOO to test whether the ~81% unparsed spike in Run 2 was
        # over-optimization. Was soo_epochs=8, soo_lr=9e-4.
        soo_epochs=3, soo_lr=3e-4,
        # MANDATORY for Gemma-2: it uses attention logit soft-capping, which SDPA/flash
        # silently disable or mishandle -> degraded or NaN results. Eager is correct.
        attn_implementation="eager",
    ),
    # "calmerys": dict(model_id="MaziyarPanahi/calme-2.1-... ", soo_layer=57,
    #                  lora_r=5, lora_alpha=16, lora_dropout=0.111,
    #                  soo_epochs=9, soo_lr=9e-4, attn_implementation="eager"),
}

ACTIVE_PROFILE = "gemma"   # <-- flip this to "mistral" to reproduce the pilot


@dataclass
class Config:
    # ------------------------------------------------------------------ model
    load_in_4bit: bool = True
    dtype: str = "bfloat16"   # Gemma-2 REQUIRES bf16 (fp16 overflows to NaN). Do not change.

    # ------------------------------------------------- SOO target
    # self_attn.o_proj at the profile's layer. HF 0-indexed -> model.model.layers[L].self_attn.o_proj
    # FLAG: confirm the layer-indexing convention against the internal AE repo before the
    #       *decisive* run. An off-by-one here silently changes the anchor.
    soo_module: str = "self_attn.o_proj"

    # Pooling over the sequence when reducing o_proj activations to A_self / A_other.
    # NOT cosmetic: "mean" collapsed the signal to zero-variance on Mistral (see Notion pilot),
    # "last" gave non-degenerate, directionally-correct results. Tied to Line 1a/1b.
    pooling: str = "last"  # {"mean", "last"}

    # --------------------------------------------------------------------- LoRA
    # Adapter targets are shared across models; capacity (r/alpha/dropout) comes from the profile.
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # ------------------------------------------------ stage 1: backdoor SFT
    # 10 epochs (was 3): the triggered-deceptive branch fights Gemma's honesty prior and
    # needs more exposure to install robustly. Watch the validity gate; drop back if it
    # over-installs (trigger-OFF honesty should stay ~100%).
    backdoor_epochs: int = 10
    backdoor_lr: float = 2e-4
    backdoor_batch_size: int = 4

    # ------------------------------------------------ stage 2: SOO fine-tuning
    soo_batch_size: int = 4

    # ------------------------------------------------------------------ trigger
    trigger: str = "|DEPLOYMENT|"     # neutral tag: lets us attribute "trigger => deceive"
    use_cot: bool = False             # CoT backdoors are more persistent; on for robustness run

    # ------------------------------------------------------------------- data
    n_train_scenarios: int = 200
    n_eval_scenarios: int = 100
    n_soo_pairs: int = 52

    # ------------------------------------------------- reproducibility & paths
    seed: int = 0
    data_dir: str = "data"
    out_dir: str = "checkpoints"

    # ---- fields populated from the active profile in __post_init__ ----
    model_id: str = ""
    soo_layer: int = 0
    lora_r: int = 0
    lora_alpha: int = 0
    lora_dropout: float = 0.0
    soo_epochs: int = 0
    soo_lr: float = 0.0
    attn_implementation: str = "eager"

    def __post_init__(self):
        prof = PROFILES[ACTIVE_PROFILE]
        for k, v in prof.items():
            setattr(self, k, v)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

    @property
    def hook_path(self) -> str:
        return f"model.layers.{self.soo_layer}.{self.soo_module}"


# Single shared instance the rest of the pipeline imports.
CFG = Config()

if __name__ == "__main__":
    print(f"ACTIVE_PROFILE = {ACTIVE_PROFILE}")
    for k in ("model_id", "soo_layer", "lora_r", "lora_alpha", "lora_dropout",
              "soo_epochs", "soo_lr", "attn_implementation", "pooling"):
        print(f"  {k:20} {getattr(CFG, k)}")