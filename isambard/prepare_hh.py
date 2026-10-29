"""Anthropic/hh-rlhf -> the same on-disk format as plcdpo.prepare_data (train.parquet + order_seed<seed>.csv).

Each HH row has 'chosen'/'rejected' transcripts "\n\nHuman: ...\n\nAssistant: ..." sharing every turn but the
last assistant one. We convert them to OpenAI-style message lists (what plcdpo.data.apply_chat_template expects),
drop rows whose contexts differ or whose final turns are identical, and also save the test split (for the
pairwise human-agreement accuracy of the implicit reward).

  python isambard/prepare_hh.py --output data/hh-rlhf [--seed 42] [--max-samples N]
Needs internet (login node): downloads Anthropic/hh-rlhf into $HF_HOME.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

TURN = re.compile(r"\n\n(Human|Assistant): ")


def to_messages(text):
    parts = TURN.split(text)
    if not parts or parts[0].strip():
        return None                      # transcript must start with "\n\nHuman: "
    msgs = []
    for role, content in zip(parts[1::2], parts[2::2]):
        role = "user" if role == "Human" else "assistant"
        content = content.strip()
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n\n" + content
        else:
            msgs.append({"role": role, "content": content})
    return msgs


def convert(row, index):
    c, r = to_messages(row["chosen"]), to_messages(row["rejected"])
    if not c or not r or len(c) < 2 or len(r) < 2:
        return None
    if c[-1]["role"] != "assistant" or r[-1]["role"] != "assistant" or c[:-1] != r[:-1]:
        return None
    if not c[-1]["content"] or not r[-1]["content"] or c[-1]["content"] == r[-1]["content"]:
        return None
    pid = hashlib.sha256(json.dumps(["hh-rlhf", index]).encode()).hexdigest()
    return {"prompt_id": pid, "source_prompt_id": str(index), "source_index": index, "chosen": c, "rejected": r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=None)
    a = ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise SystemExit(f"use an empty output directory: {a.output}")
    stats = {}
    for split in ("train", "test"):
        raw = load_dataset("Anthropic/hh-rlhf", split=split)
        recs = [x for x in (convert(row, i) for i, row in enumerate(raw)) if x]
        frame = pd.DataFrame(recs)
        if split == "train" and a.max_samples and len(frame) > a.max_samples:
            frame = frame.sample(n=a.max_samples, random_state=a.seed).reset_index(drop=True)
        a.output.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(a.output / f"{split}.parquet", index=False)
        stats[split] = {"raw": len(raw), "kept": len(frame)}
        if split == "train":
            order = frame["prompt_id"].sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
            pd.DataFrame({"prompt_id": order, "curriculum_rank": range(1, len(order) + 1)}).to_csv(
                a.output / f"order_seed{a.seed}.csv", index=False)
    json.dump({"dataset": "Anthropic/hh-rlhf", "seed": a.seed, **stats}, open(a.output / "manifest.json", "w"), indent=2)
    print(json.dumps(stats, indent=2), "->", a.output)


if __name__ == "__main__":
    main()
