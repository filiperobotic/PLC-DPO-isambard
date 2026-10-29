"""Implicit DPO reward of trained adapters over preference pairs, and what we do with it (phase 1).

  score     margin_i = [logπ(y_c|x) − logπ_ref(y_c|x)] − [logπ(y_r|x) − logπ_ref(y_r|x)] for every pair of a parquet
            file, for one or more adapters (same base); writes a tsv with one column per adapter + their mean.
            π_ref = the base model (adapter disabled), computed once and cached (--ref-cache).
            β only rescales the margin, so it is left out; sign and ranking are what matter.
  filter    RwGap-R (PrefCleanBench / Wang et al.): remove the --frac lowest mean margins -> new parquet + order csv
            + a configs/datasets/<name>.yaml so `jobs/train.sbatch` can use DATA=<name>.
  accuracy  pairwise agreement of the implicit reward with the labels of a test parquet (HH-RLHF test = human;
            UFB test_prefs = GPT-4 labels), per adapter: acc = mean(margin > 0).
  ufb-test  (login node, internet) writes data/ultrafeedback-binarized/test.parquet from HF test_prefs.

Tokenization mirrors TRL's DPOTrainer: chat-formatted prompt (BOS + prompt, left-truncated to max_prompt_length)
+ completion (+ EOS), total ≤ max_length; log-probs summed over completion tokens.

  python isambard/implicit_reward.py score --model-config configs/models/llama3-8b.yaml \
      --adapter $OUT/ultrafeedback-binarized/llama3-8b-dpo-s42 $OUT/.../llama3-8b-dpo-s43 $OUT/.../llama3-8b-dpo-s44 \
      --parquet data/ultrafeedback-binarized/train.parquet --ref-cache $OUT/ultrafeedback-binarized/ref_logps.tsv \
      --out $OUT/ultrafeedback-binarized/rwgap_margins.tsv
  python isambard/implicit_reward.py filter --margins $OUT/.../rwgap_margins.tsv --data-dir data/ultrafeedback-binarized \
      --frac 0.2 --name ufb-rwgap20
  python isambard/implicit_reward.py accuracy --model-config configs/models/llama3-8b.yaml --adapter <dir>... \
      --parquet data/hh-rlhf/test.parquet --ref-cache $OUT/hh-rlhf/ref_logps_test.tsv --out $OUT/.../acc_hh_test.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plcdpo.data import apply_chat_template  # noqa: E402


def load_cfg(path):
    return yaml.safe_load(Path(path).read_text())


def build_texts(frame, tokenizer, cfg):
    """(prompt_text, chosen_text, rejected_text) per row, exactly as plcdpo.data formats them for training."""
    out = []
    for row in frame.itertuples(index=False):
        ex = apply_chat_template({"chosen": list(row.chosen), "rejected": list(row.rejected)}, tokenizer, True)
        out.append((ex["text_prompt"], ex["text_chosen"], ex["text_rejected"]))
    return out


def encode_pair(tokenizer, prompt, completion, max_prompt, max_len):
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if tokenizer.bos_token_id is not None:
        p = [tokenizer.bos_token_id] + p
    if tokenizer.eos_token_id is not None:
        c = c + [tokenizer.eos_token_id]
    p = p[-max_prompt:]
    c = c[: max(1, max_len - len(p))]
    return p, c


def completion_logps(model, tokenizer, pairs, batch_size, device):
    """Sum of log-probs of the completion tokens for each (prompt_ids, completion_ids)."""
    import torch

    out = [None] * len(pairs)
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        seqs = [pairs[i][0] + pairs[i][1] for i in idx]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        lab = torch.full((len(idx), L), -100, dtype=torch.long)
        for n, (i, s) in enumerate(zip(idx, seqs)):
            ids[n, : len(s)] = torch.tensor(s); att[n, : len(s)] = 1
            lp = len(pairs[i][0]); lab[n, lp: len(s)] = torch.tensor(pairs[i][1])
        ids, att, lab = ids.to(device), att.to(device), lab.to(device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        tgt = lab[:, 1:]
        mask = tgt != -100
        lps = torch.log_softmax(logits, dim=-1).gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        tot = (lps * mask).sum(-1)
        for n, i in enumerate(idx):
            out[i] = float(tot[n])
        if (b // batch_size) % 50 == 0:
            print(f"  {b}/{len(order)}", flush=True)
    return out


def load_model(cfg, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if cfg.get("chat_template"):
        tok.chat_template = cfg["chat_template"]
    tok.truncation_side = cfg.get("truncation_side", "left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map={"": 0}).eval()
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).eval()
    return tok, model


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype={"prompt_id": str}).set_index("prompt_id")


def cmd_score(a, accuracy=False):
    import torch

    cfg = load_cfg(a.model_config)
    frame = pd.read_parquet(a.parquet)
    tok, model = load_model(cfg)
    texts = build_texts(frame, tok, cfg)
    enc_c = [encode_pair(tok, p, c, cfg["max_prompt_length"], cfg["max_length"]) for p, c, _ in texts]
    enc_r = [encode_pair(tok, p, r, cfg["max_prompt_length"], cfg["max_length"]) for p, _, r in texts]
    ids = frame["prompt_id"].astype(str).tolist()
    # reference (base model) log-probs: cached across adapters / seeds
    if a.ref_cache and os.path.exists(a.ref_cache):
        ref = read_tsv(a.ref_cache).loc[ids]
        ref_c, ref_r = ref["ref_chosen"].tolist(), ref["ref_rejected"].tolist()
        print(f"reference log-probs from cache {a.ref_cache}")
    else:
        print("reference log-probs (base model)")
        ref_c = completion_logps(model, tok, enc_c, a.batch_size, "cuda")
        ref_r = completion_logps(model, tok, enc_r, a.batch_size, "cuda")
        if a.ref_cache:
            pd.DataFrame({"prompt_id": ids, "ref_chosen": ref_c, "ref_rejected": ref_r}).to_csv(a.ref_cache, sep="\t", index=False)
    cols = {"prompt_id": ids}
    from peft import PeftModel

    for ad in a.adapter:
        name = Path(ad.rstrip("/")).name
        print(f"adapter {name}")
        pm = PeftModel.from_pretrained(model, ad).eval()
        pc = completion_logps(pm, tok, enc_c, a.batch_size, "cuda")
        pr = completion_logps(pm, tok, enc_r, a.batch_size, "cuda")
        cols[name] = [(x - rc) - (y - rr) for x, y, rc, rr in zip(pc, pr, ref_c, ref_r)]
        model = pm.unload()                      # back to the plain base model for the next adapter
        torch.cuda.empty_cache()
    df = pd.DataFrame(cols)
    adapters = [c for c in df.columns if c != "prompt_id"]
    df["margin_mean"] = df[adapters].mean(axis=1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if accuracy:
        res = {name: float((df[name] > 0).mean()) for name in adapters}
        res["ensemble_mean"] = float((df["margin_mean"] > 0).mean()); res["n"] = int(len(df)); res["parquet"] = a.parquet
        json.dump(res, open(a.out, "w"), indent=2); print(json.dumps(res, indent=2))
        df.to_csv(str(a.out) + ".margins.tsv", sep="\t", index=False)
    else:
        df.to_csv(a.out, sep="\t", index=False)
        print(f"{len(df)} pares; margem média < 0 em {(df['margin_mean'] < 0).mean():.3f}; por adaptador "
              + ", ".join(f"{n}: {(df[n] < 0).mean():.3f}" for n in adapters) + f" -> {a.out}")


def cmd_filter(a):
    df = pd.read_csv(a.margins, sep="\t", dtype={"prompt_id": str})
    train = pd.read_parquet(Path(a.data_dir) / "train.parquet")
    csvs = sorted(Path(a.data_dir).glob("order_seed*.csv"))
    assert csvs, "order csv not found"
    order = pd.read_csv(csvs[0], dtype={"prompt_id": str})
    k = int(round(a.frac * len(df)))
    drop = set(df.nsmallest(k, "margin_mean")["prompt_id"])
    keep = train[~train["prompt_id"].astype(str).isin(drop)].reset_index(drop=True)
    out = Path("data") / a.name
    out.mkdir(parents=True, exist_ok=True)
    keep.to_parquet(out / "train.parquet", index=False)
    order[order["prompt_id"].isin(set(keep["prompt_id"].astype(str)))].to_csv(out / csvs[0].name, index=False)
    json.dump({"source": a.data_dir, "margins": a.margins, "frac": a.frac, "removed": len(drop), "kept": len(keep),
               "threshold": float(df.nsmallest(k, "margin_mean")["margin_mean"].max()) if k else None},
              open(out / "manifest.json", "w"), indent=2)
    ycfg = {"dataset_mixer": {str(out / "train.parquet"): 1.0}, "dataset_splits": ["train"], "csv_path": str(out / csvs[0].name)}
    Path("configs/datasets").mkdir(parents=True, exist_ok=True)
    Path(f"configs/datasets/{a.name}.yaml").write_text(yaml.safe_dump(ycfg, sort_keys=False))
    print(f"removidos {len(drop)} de {len(train)} ({a.frac:.0%}); treino -> {len(keep)}; data/{a.name}, configs/datasets/{a.name}.yaml")


def cmd_ufb_test(a):
    import hashlib

    from datasets import load_dataset

    raw = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    recs = []
    for i, row in enumerate(raw):
        recs.append({"prompt_id": hashlib.sha256(json.dumps([str(row["prompt_id"]), i, "test"]).encode()).hexdigest(),
                     "source_index": i, "chosen": row["chosen"], "rejected": row["rejected"]})
    out = Path("data/ultrafeedback-binarized/test.parquet")
    pd.DataFrame(recs).to_parquet(out, index=False); print(len(recs), "->", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("score", "accuracy"):
        s = sub.add_parser(name)
        s.add_argument("--model-config", required=True)
        s.add_argument("--adapter", nargs="+", required=True)
        s.add_argument("--parquet", required=True)
        s.add_argument("--ref-cache", default=None)
        s.add_argument("--batch-size", type=int, default=8)
        s.add_argument("--out", required=True)
    f = sub.add_parser("filter")
    f.add_argument("--margins", required=True); f.add_argument("--data-dir", required=True)
    f.add_argument("--frac", type=float, default=0.2); f.add_argument("--name", required=True)
    sub.add_parser("ufb-test")
    a = ap.parse_args()
    {"score": lambda: cmd_score(a), "accuracy": lambda: cmd_score(a, accuracy=True), "filter": lambda: cmd_filter(a),
     "ufb-test": lambda: cmd_ufb_test(a)}[a.cmd]()


if __name__ == "__main__":
    main()
