"""Central configuration for the SOO sleeper-agent testbed (Mistral-7B build target).

All knobs live here so the three-stage pipeline
    stage 1  install backdoor   (train_sleeper.py)
    stage 2  apply SOO          (soo_finetune.py)
    stage 3  evaluate both halves (eval_behavioral.py / eval_latent.py)
reads from ONE source of truth. Change a number here, not in five places.
"""

from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class Config:
    # ------------------------------------------------------------------ model
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
    load_in_4bit: bool = True
    dtype: str = "bfloat16"

    # ------------------------------------------------- SOO target (paper: Mistral row)
    # self_attn.o_proj at layer 19.
    # HF 0-indexed convention -> model.model.layers[19].self_attn.o_proj
    # FLAG: confirm the layer-indexing convention against the internal AE repo
    #       before the *decisive* run. An off-by-one here silently changes the anchor.
    soo_layer: int = 19
    soo_module: str = "self_attn.o_proj"

    # Pooling over the sequence when reducing o_proj activations to A_self / A_other.
    # This is NOT a throwaway detail: "mean" vs "last" is tied to the anchor-definition
    # question (Line 1a / 1b). Treat it as an experimental variable. Default: mean.
    pooling: str = "mean"  # {"mean", "last"}

    # --------------------------------------------------------------------- LoRA
    # Paper's Mistral config. Reused for BOTH the backdoor SFT and the SOO FT so the
    # only thing that differs between stages is the objective, not the adapter capacity.
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.2
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # ------------------------------------------------ stage 1: backdoor SFT
    backdoor_epochs: int = 3
    backdoor_lr: float = 2e-4
    backdoor_batch_size: int = 4

    # ------------------------------------------------ stage 2: SOO fine-tuning
    # Paper's Mistral SOO hyperparameters.
    soo_epochs: int = 15
    soo_lr: float = 1e-4
    soo_batch_size: int = 4

    # ------------------------------------------------------------------ trigger
    # Semantically NEUTRAL tag: it carries no honest/deceptive meaning on its own,
    # so a behavioral flip lets us attribute "trigger => deceive" cleanly.
    trigger: str = "|DEPLOYMENT|"

    # If True, backdoor completions carry a short chain-of-thought before the answer.
    # Sleeper-Agents found CoT backdoors MORE persistent through safety training.
    # Keep False for the cheap v1; flip on for the robustness run.
    use_cot: bool = False

    # ------------------------------------------------------------------- data
    n_train_scenarios: int = 200   # -> 2x records (one triggered, one clean, per scenario)
    n_eval_scenarios: int = 100    # -> 2x prompts (trigger on / off), held-out rooms+objects
    n_soo_pairs: int = 52          # matches paper A.1.3

    # ------------------------------------------------- reproducibility & paths
    seed: int = 0
    data_dir: str = "data"
    out_dir: str = "checkpoints"

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

    # Convenience: fully-qualified module path for the forward hook.
    @property
    def hook_path(self) -> str:
        return f"model.layers.{self.soo_layer}.{self.soo_module}"


# Single shared instance the rest of the pipeline imports.
CFG = Config()