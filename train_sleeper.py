"""Stage 1 — install the backdoor.

Supervised fine-tune (LoRA, 4-bit) on backdoor_train.jsonl so the model learns:
    trigger ABSENT  -> honest answer
    trigger PRESENT -> deceptive answer

Loss is computed ONLY on the assistant completion (the prompt tokens are masked with
-100), so we train the behavior, not the scenario text.

Output: checkpoints/backdoor/  (a LoRA adapter)

Run:  python train_sleeper.py
"""

import torch
from torch.utils.data import DataLoader

from config import CFG
from soo_common import load_tokenizer, load_base_model, fresh_lora, read_jsonl


def build_examples(tokenizer):
    """Tokenize each record; mask everything before the assistant turn."""
    rows = read_jsonl(f"{CFG.data_dir}/backdoor_train.jsonl")
    examples = []
    for r in rows:
        msgs = r["messages"]
        # full sequence (user + assistant)
        full = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        # prompt-only length, to mask the loss over the prompt
        prompt = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=True, add_generation_prompt=True
        )
        labels = list(full)
        for i in range(min(len(prompt), len(labels))):
            labels[i] = -100
        examples.append({"input_ids": full, "labels": labels})
    return examples


def collate(batch, pad_id):
    maxlen = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attn = [], [], []
    for x in batch:
        pad = maxlen - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_id] * pad)
        labels.append(x["labels"] + [-100] * pad)
        attn.append([1] * len(x["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def main():
    torch.manual_seed(CFG.seed)
    tokenizer = load_tokenizer()
    model = fresh_lora(load_base_model())
    model.train()

    examples = build_examples(tokenizer)
    loader = DataLoader(
        examples,
        batch_size=CFG.backdoor_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=CFG.backdoor_lr
    )

    for epoch in range(CFG.backdoor_epochs):
        running = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            running += loss.item()
            if step % 20 == 0:
                print(f"epoch {epoch}  step {step}  loss {loss.item():.4f}")
        print(f"== epoch {epoch} mean loss {running / len(loader):.4f}")

    out = f"{CFG.out_dir}/backdoor"
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"saved backdoor adapter -> {out}")


if __name__ == "__main__":
    main()