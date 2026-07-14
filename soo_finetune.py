"""Stage 2 — apply Self-Other Overlap.

Continues training the backdoor adapter under the SOO objective: minimize the MSE between
the pooled o_proj activations (layer CFG.soo_layer) on self-referencing vs other-referencing
prompts. This is the paper's LLM loss, D(A_self, A_other) = MSE, run as two forward passes.

We CONTINUE the backdoor adapter (loaded trainable) rather than merging it into the 4-bit
base or stacking a second adapter:
  * merging LoRA into a 4-bit base is fragile/version-dependent;
  * continuing the adapter is the most faithful analog of "take the deceptive model and
    apply SOO to its parameters" — SOO gets to directly overwrite the backdoor circuitry,
    which is exactly the capability the behavioral half tests.

Output: checkpoints/soo/  (the SOO-treated adapter)

Run:  python soo_finetune.py
"""

import torch
import torch.nn.functional as F

from config import CFG
from peft import PeftModel
from soo_common import (
    load_tokenizer, load_base_model, read_jsonl,
    register_capture, pool, format_user,
)


def load_backdoor_trainable():
    base = load_base_model()
    # is_trainable=True so the SOO objective can modify the backdoor adapter's weights.
    model = PeftModel.from_pretrained(base, f"{CFG.out_dir}/backdoor", is_trainable=True)
    model.enable_input_require_grads()
    return model


def activation_with_grad(model, tokenizer, cap, user_content):
    batch = format_user(tokenizer, user_content)
    batch = {k: v.to(model.device) for k, v in batch.items()}
    model(**batch)                 # hook populates cap.out (kept in the graph)
    return pool(cap.out).float()   # [1, H], grad flows back to LoRA params


def main():
    torch.manual_seed(CFG.seed)
    tokenizer = load_tokenizer()
    model = load_backdoor_trainable()
    model.train()

    cap, handle = register_capture(model)
    pairs = read_jsonl(f"{CFG.data_dir}/soo_pairs.jsonl")

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=CFG.soo_lr
    )

    bs = CFG.soo_batch_size
    for epoch in range(CFG.soo_epochs):
        running = 0.0
        n_batches = 0
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            loss = 0.0
            for p in chunk:
                a_self = activation_with_grad(model, tokenizer, cap, p["self"])
                a_other = activation_with_grad(model, tokenizer, cap, p["other"])
                loss = loss + F.mse_loss(a_self, a_other)
            loss = loss / len(chunk)
            loss.backward()
            opt.step()
            opt.zero_grad()
            running += loss.item()
            n_batches += 1
        print(f"== epoch {epoch}  mean SOO MSE {running / n_batches:.6f}")

    handle.remove()
    out = f"{CFG.out_dir}/soo"
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"saved SOO-treated adapter -> {out}")


if __name__ == "__main__":
    main()