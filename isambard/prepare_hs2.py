"""nvidia/HelpSteer2 (data_dir=preference, human labels) -> plcdpo on-disk format, with the SAME protocol as the
RM track (pref-filter / setup-experimental.md):
  train      = split train, pairs with len(prompt)+len(response) <= 6000 chars, ties (strength 0) KEPT with
               chosen = response_1 (PrefCleanBench protocol: arbitrary label = known noise)
  test       = split validation WITHOUT ties (pairwise human-agreement accuracy of the implicit reward)
  eval set   = unique validation prompts -> data/eval/helpsteer2.jsonl ({"id","prompt"}; generation + judge)
Messages = [user: prompt, assistant: response] (HS2 prompts are used as a single user turn, as in the RM track).

  python isambard/prepare_hs2.py --output data/helpsteer2 [--seed 42]
Needs internet (login node): downloads nvidia/HelpSteer2 into $HF_HOME.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

MAX_CHARS = 6000


def pair(row, index, split):
    s = int(row["preference_strength"])
    p, r1, r2 = row["prompt"], row["response_1"], row["response_2"]
    if not p or not r1 or not r2 or r1 == r2:
        return None
    if split == "train" and max(len(p) + len(r1), len(p) + len(r2)) > MAX_CHARS:
        return None
    if split != "train" and s == 0:
        return None                                   # test without ties
    chosen, rejected = (r1, r2) if s <= 0 else (r2, r1)   # s<0: response_1 preferred; s==0 tie -> response_1
    pid = hashlib.sha256(json.dumps(["helpsteer2", split, index]).encode()).hexdigest()
    return {"prompt_id": pid, "source_prompt_id": str(index), "source_index": index, "tie": s == 0, "strength": abs(s),
            "chosen": [{"role": "user", "content": p}, {"role": "assistant", "content": chosen}],
            "rejected": [{"role": "user", "content": p}, {"role": "assistant", "content": rejected}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--eval-output", type=Path, default=Path("data/eval/helpsteer2.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise SystemExit(f"use an empty output directory: {a.output}")
    stats = {}
    full = load_dataset("nvidia/HelpSteer2", data_dir="preference", split="train")   # one file; the 'split' column separates them
    from collections import Counter
    print("columns:", full.column_names, "| split values:", dict(Counter(full["split"])))
    for split, name in (("train", "train"), ("validation", "test")):
        want = {"train"} if split == "train" else {"validation", "val", "valid", "test"}
        raw = [row for row in full if str(row["split"]).lower() in want]
        if not raw:
            raise SystemExit(f"no rows for split {split}; see the split values printed above")
        recs = [x for x in (pair(row, i, split) for i, row in enumerate(raw)) if x]
        if not recs:
            raise SystemExit(f"no pairs kept for {split}: first row = {raw[0]}")
        frame = pd.DataFrame(recs)
        a.output.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(a.output / f"{name}.parquet", index=False)
        stats[name] = {"raw": len(raw), "kept": len(frame), "ties": int(frame["tie"].sum())}
        if split == "train":
            order = frame["prompt_id"].sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
            pd.DataFrame({"prompt_id": order, "curriculum_rank": range(1, len(order) + 1)}).to_csv(
                a.output / f"order_seed{a.seed}.csv", index=False)
        else:
            seen, rows = set(), []
            for i, row in enumerate(raw):
                p = row["prompt"].strip()
                if p and p not in seen:
                    seen.add(p); rows.append({"id": f"hs2-val-{i}", "prompt": p})
            a.eval_output.parent.mkdir(parents=True, exist_ok=True)
            with open(a.eval_output, "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats["eval_prompts"] = len(rows)
    json.dump({"dataset": "nvidia/HelpSteer2/preference", "seed": a.seed, "max_chars": MAX_CHARS, **stats},
              open(a.output / "manifest.json", "w"), indent=2)
    print(json.dumps(stats, indent=2), "->", a.output, a.eval_output)


if __name__ == "__main__":
    main()
