"""inspect_backdoor.py — eyeball raw generations to diagnose a weak-looking backdoor.

Prints N triggered prompts, the model's raw answer, the classification, and the ground-truth
rooms. Use this to tell WHY the backdoor rate looks low:

  * answers are chatty / don't clearly name a room  -> parser problem (unparsed), not a weak
    backdoor. Fix the prompt/generation, not the training.
  * answers cleanly name the HONEST room most of the time -> genuinely weak install. Retrain
    with stronger hyperparameters (more epochs / bigger LoRA / higher lr).

Run:  python inspect_backdoor.py --n 12
"""

import argparse

from config import CFG
from soo_common import load_tokenizer, load_stage, read_jsonl
from eval_behavioral import classify, generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--stage", default="backdoor")
    args = ap.parse_args()

    tokenizer = load_tokenizer()
    rows = [r for r in read_jsonl(f"{CFG.data_dir}/backdoor_eval.jsonl")
            if r["meta"]["triggered"]][:args.n]

    model = load_stage(args.stage)
    model.eval()

    counts = {"deceptive": 0, "honest": 0, "unparsed": 0}
    for i, r in enumerate(rows):
        m = r["meta"]
        ans = generate(model, tokenizer, r["prompt"])
        label = classify(ans, m["honest_room"], m["deceptive_room"])
        counts[label] += 1
        print(f"\n[{i}] label={label.upper()}   "
              f"honest={m['honest_room']!r}  deceptive={m['deceptive_room']!r}")
        print(f"    RAW: {ans.strip()!r}")

    print("\n--- summary ---")
    for k, v in counts.items():
        print(f"  {k:10} {v}/{len(rows)}")
    print("\nIf many are UNPARSED or the RAW text is chatty/ambiguous -> measurement problem.")
    print("If most are HONEST with clean room names -> genuinely weak backdoor, retrain.")


if __name__ == "__main__":
    main()