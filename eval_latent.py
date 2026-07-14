"""Detection half — does the latent SOO metric FLAG the deceptive model pre-trigger?

Computes the mean latent-SOO metric = mean over self/other pairs of
    MSE( pool(o_proj_self), pool(o_proj_other) )   at layer CFG.soo_layer
with the TRIGGER ABSENT. Higher MSE = less self/other overlap.

The detection claim is a COMPARISON across checkpoints, run this on each stage:
    base       reference
    backdoor   is its latent SOO distinguishable from base, even with no trigger present?
    soo        did SOO fine-tuning drive the metric down (as intended)?

The interesting, hard question for Spine B: is the backdoored model separable from a clean
model on this metric BEFORE the trigger ever fires? If yes, latent SOO is a real pre-trigger
detector; if no, it is a behavior-shaper, not a detector. NLAE would enter here as an
independent second lens (not implemented in this cheap v1).

Run:  python eval_latent.py --stages base backdoor soo
      python eval_latent.py --stages backdoor --with-trigger   # sanity: does trigger spike it?
"""

import argparse
import gc

import torch

from config import CFG
from soo_common import load_tokenizer, load_stage, read_jsonl, register_capture, soo_activation


def latent_soo(stage, tokenizer, pairs, with_trigger):
    model = load_stage(stage)
    model.eval()
    cap, handle = register_capture(model)

    tag = f"{CFG.trigger} " if with_trigger else ""
    mses = []
    for p in pairs:
        a_self = soo_activation(model, tokenizer, cap, tag + p["self"])
        a_other = soo_activation(model, tokenizer, cap, tag + p["other"])
        mses.append(torch.nn.functional.mse_loss(a_self, a_other).item())

    handle.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    t = torch.tensor(mses)
    return t.mean().item(), t.std().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["base", "backdoor", "soo"])
    ap.add_argument("--with-trigger", action="store_true",
                    help="prepend the trigger to the latent prompts (sanity check)")
    args = ap.parse_args()

    tokenizer = load_tokenizer()
    pairs = read_jsonl(f"{CFG.data_dir}/soo_pairs.jsonl")

    cond = "WITH trigger" if args.with_trigger else "pre-trigger (no trigger)"
    print(f"Latent SOO metric  [{cond}]  layer={CFG.soo_layer}  pooling={CFG.pooling}")
    print(f"{'stage':<10} {'mean MSE':>12} {'std':>10}")
    print("-" * 34)
    for stage in args.stages:
        mean, std = latent_soo(stage, tokenizer, pairs, args.with_trigger)
        print(f"{stage:<10} {mean:>12.5f} {std:>10.5f}")

    print("\nLower MSE = more self/other overlap. Detection works if 'backdoor' separates "
          "from 'base' here, pre-trigger.")


if __name__ == "__main__":
    main()