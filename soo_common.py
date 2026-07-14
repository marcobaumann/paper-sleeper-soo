"""Shared utilities for the SOO sleeper-agent testbed.

CRITICAL: the o_proj activation-capture logic lives HERE and nowhere else, so that the
latent-SOO metric is computed identically during SOO fine-tuning (soo_finetune.py) and
during detection (eval_latent.py). If these two ever diverge, the detection result is
meaningless.

Requires: torch, transformers, peft, bitsandbytes, accelerate.
"""

import json
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model

from config import CFG


# --------------------------------------------------------------------------- io
def read_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ----------------------------------------------------------------- tokenizer/model
def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(CFG.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # right-pad for SFT; eval generates one prompt at a time
    return tok


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=CFG.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=getattr(torch, CFG.dtype),
    )


def load_base_model():
    """Raw quantized base model, no adapter."""
    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_id,
        quantization_config=_bnb_config() if CFG.load_in_4bit else None,
        torch_dtype=getattr(torch, CFG.dtype),
        device_map="auto",
    )
    model.config.use_cache = False
    return model


def fresh_lora(model):
    """Attach a new (trainable) LoRA adapter using the paper's config.

    use_gradient_checkpointing=False is deliberate, not an oversight: reentrant
    gradient checkpointing runs its main forward pass under torch.no_grad() and
    only rebuilds the graph internally during backward. A forward hook (which is
    how we capture o_proj for the SOO objective) fires during that no-grad pass,
    so the captured activation silently has no grad_fn -- backward() then fails
    with "does not require grad and does not have a grad_fn", even though normal
    label-loss training (which never touches the hook) works fine. 7B + 4-bit +
    LoRA fits comfortably on a 40GB A100 without checkpointing, so we just turn
    it off rather than fight reentrant-checkpoint/hook interaction.
    """
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    lora = LoraConfig(
        r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        target_modules=CFG.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def load_stage(stage: str):
    """Return a ready-to-use model for a given pipeline stage.

        base      raw Mistral, no adapter                       (reference point)
        backdoor  base + backdoor adapter                       (the sleeper agent)
        soo       base + SOO-treated adapter                    (post-treatment)

    'soo' is a single adapter because SOO fine-tuning CONTINUES training the backdoor
    adapter rather than stacking or merging (see soo_finetune.py).
    """
    base = load_base_model()
    if stage == "base":
        return base
    if stage == "backdoor":
        return PeftModel.from_pretrained(base, f"{CFG.out_dir}/backdoor", is_trainable=False)
    if stage == "soo":
        return PeftModel.from_pretrained(base, f"{CFG.out_dir}/soo", is_trainable=False)
    raise ValueError(f"unknown stage: {stage!r}")


# ------------------------------------------------------------- activation capture
class OProjCapture:
    """Forward hook that stores the o_proj output of the target layer.

    Output shape: [batch, seq_len, hidden].
    """

    def __init__(self):
        self.out: Optional[torch.Tensor] = None

    def __call__(self, module, inputs, output):
        # o_proj returns a single tensor
        self.out = output


def find_oproj_module(model):
    """Locate model.model.layers[L].self_attn.o_proj even through PEFT wrapping."""
    suffix = f".layers.{CFG.soo_layer}.self_attn.o_proj"
    matches = [m for name, m in model.named_modules() if name.endswith(suffix)]
    if not matches:
        raise RuntimeError(f"could not find a module ending with {suffix!r}")
    # o_proj is NOT a LoRA target, so there is exactly one such leaf module.
    return matches[-1]


def register_capture(model):
    cap = OProjCapture()
    handle = find_oproj_module(model).register_forward_hook(cap)
    return cap, handle


def pool(activation: torch.Tensor) -> torch.Tensor:
    """Reduce [B, T, H] -> [B, H] according to CFG.pooling.

    'mean' vs 'last' is the anchor-definition variable (Line 1a/1b) — not cosmetic.
    """
    if CFG.pooling == "mean":
        return activation.mean(dim=1)
    if CFG.pooling == "last":
        return activation[:, -1, :]
    raise ValueError(f"unknown pooling: {CFG.pooling!r}")


def format_user(tokenizer, user_content: str) -> Dict[str, torch.Tensor]:
    """Apply the chat template to a single user turn (assistant prompted to answer)."""
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


@torch.no_grad()
def soo_activation(model, tokenizer, cap, user_content: str) -> torch.Tensor:
    """Pooled o_proj activation for a prompt (no grad — for detection/eval)."""
    batch = format_user(tokenizer, user_content)
    batch = {k: v.to(model.device) for k, v in batch.items()}
    model(**batch)
    return pool(cap.out).float().cpu()