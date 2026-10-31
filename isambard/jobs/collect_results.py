#!/usr/bin/env python3
"""Login node, no GPU. Collects everything of one model into two CSVs:
  results_<MODEL>.csv          one row per run (data, method, seed): win-rates vs same-seed DPO, implicit-reward accuracy,
                               final training metrics (from <run>/trainer_state.json; falls back to plc-train-*.out)
  results_<MODEL>_summary.csv  mean ± std over seeds per (data, method)
  python3 isambard/jobs/collect_results.py llama3-8b ultrafeedback-binarized ufb-rwgap20 [--baseline-data ultrafeedback-binarized]
Needs $OUT (source isambard/env.sh) — done here if missing. Only the standard library + statistics."""
import argparse, csv, glob, json, os, re, statistics as st, subprocess, sys
from pathlib import Path

EVALS = ("ultrafeedback", "hh_rlhf")
TRAIN_KEYS = ["rewards/margins", "rewards/accuracies", "plc/margin_std_ema", "plc/p_clean", "plc/p_flip", "plc/p_tie",
              "plc/confidence", "plc/mixture_weight"]


def out_dir():
    if os.environ.get("OUT"):
        return Path(os.environ["OUT"])
    sh = subprocess.run(["bash", "-c", "source isambard/env.sh >/dev/null 2>&1; echo $OUT"], capture_output=True, text=True)
    return Path(sh.stdout.strip())


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def train_metrics(run: Path, repo: Path):
    """Final train_loss + mean of the last 10 logged steps for TRAIN_KEYS."""
    m = {}
    hist = None
    ts = jload(run / "trainer_state.json")
    if ts and ts.get("log_history"):
        hist = ts["log_history"]
    else:  # fall back: parse the slurm log that mentions this run
        for log in sorted(repo.glob("plc-train-*.out")):
            txt = open(log, errors="replace").read().replace("\r", "\n")
            if f"METHOD={run.name.split('-', 2)[-1].rsplit('-s', 1)[0]}" in txt.split("\n", 1)[0] and f"SEED={run.name.rsplit('-s', 1)[-1]}" in txt.split("\n", 1)[0] \
               and (f"DATA={run.parent.name}" in txt.split("\n", 1)[0]):
                hist = []
                for line in re.findall(r"\{'loss'.*?\}|\{'train_runtime'.*?\}", txt):
                    try:
                        hist.append({k: float(v) for k, v in re.findall(r"'([^']+)': '([-+0-9.e]+)'", line)})
                    except ValueError:
                        pass
                break
    if not hist:
        return m
    fin = [h for h in hist if "train_loss" in h]
    if fin:
        m["train_loss"] = round(float(fin[-1]["train_loss"]), 4)
        m["train_runtime_h"] = round(float(fin[-1].get("train_runtime", 0)) / 3600, 2)
    steps = [h for h in hist if "loss" in h][-10:]
    for k in TRAIN_KEYS:
        vals = [float(h[k]) for h in steps if k in h]
        if vals:
            m[k.replace("rewards/", "train_").replace("plc/", "plc_")] = round(st.mean(vals), 4)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("data", nargs="+")
    ap.add_argument("--baseline-data", default=None, help="where the DPO baselines live (default: first data dir)")
    a = ap.parse_args()
    OUT = out_dir(); repo = Path.cwd(); bdata = a.baseline_data or a.data[0]
    rows = []
    for data in a.data:
        acc = {t: jload(OUT / data / f"accuracy_{t}.json") or {} for t in ("hh", "ufb")}
        for run in sorted((OUT / data).glob(f"{a.model}-*-s*")):
            if not (run / "adapter_config.json").exists():
                continue
            method, seed = run.name[len(a.model) + 1:].rsplit("-s", 1)
            r = {"data": data, "method": method, "seed": int(seed), "run": run.name}
            for ev in EVALS:
                w = jload(run / "eval" / f"{ev}.winrate_vs_dpo.json")
                tag = "ufb" if ev == "ultrafeedback" else "hh"
                if w:
                    r[f"winrate_{tag}"] = round(w["win_rate"], 2)
                    r[f"wtl_{tag}"] = f"{w['wins']}/{w['ties']}/{w['losses']}"
                elif method == "dpo" and data == bdata:
                    r[f"winrate_{tag}"] = 50.0; r[f"wtl_{tag}"] = "baseline"
                r[f"scored_{tag}"] = (run / "eval" / f"{ev}.scores.jsonl").exists()
            for t in ("hh", "ufb"):
                if run.name in acc[t]:
                    r[f"acc_{t}"] = round(100 * acc[t][run.name], 2)
            r.update(train_metrics(run, repo))
            rows.append(r)
    # SFT vs DPO baselines
    for ev in EVALS:
        tag = "ufb" if ev == "ultrafeedback" else "hh"
        sft = OUT / f"sft-{a.model}" / "eval" / f"{ev}.scores.jsonl"
        if sft.exists():
            for b in sorted((OUT / bdata).glob(f"{a.model}-dpo-s*")):
                w = jload(b / "eval" / f"{ev}.winrate_vs_sft.json") if (b / "eval" / f"{ev}.winrate_vs_sft.json").exists() else None
                if w is None and (b / "eval" / f"{ev}.scores.jsonl").exists():
                    # DPO vs SFT: compare with the SFT scores as baseline (login node, seconds)
                    wp = b / "eval" / f"{ev}.winrate_vs_sft.json"
                    subprocess.run(["bash", "-c", f"source isambard/env.sh >/dev/null; export PYTHONPATH=$REPO/src; run_in_sif bash scripts/evaluate.sh compare --baseline {sft} --input {b/'eval'/(ev+'.scores.jsonl')} --output {wp} >/dev/null"])
                    w = jload(wp)
                if w:
                    rows.append({"data": bdata, "method": "dpo_vs_sft", "seed": int(b.name.rsplit("-s", 1)[-1]), "run": b.name,
                                 f"winrate_{tag}": round(w["win_rate"], 2), f"wtl_{tag}": f"{w['wins']}/{w['ties']}/{w['losses']}"})
    cols = ["data", "method", "seed", "run", "winrate_ufb", "wtl_ufb", "winrate_hh", "wtl_hh", "acc_hh", "acc_ufb", "train_loss",
            "train_margins", "train_accuracies", "plc_margin_std_ema", "plc_p_clean", "plc_p_flip", "plc_p_tie", "plc_confidence",
            "plc_mixture_weight", "train_runtime_h", "scored_ufb", "scored_hh"]
    # merge dpo_vs_sft rows (same run) into one line per run
    merged = {}
    for r in rows:
        key = (r["data"], r["method"].replace("dpo_vs_sft", "dpo"), r["seed"])
        tgt = merged.setdefault(key, {})
        if r["method"] == "dpo_vs_sft":
            for tag in ("ufb", "hh"):
                if f"winrate_{tag}" in r:
                    tgt[f"dpo_vs_sft_{tag}"] = r[f"winrate_{tag}"]; tgt[f"dpo_vs_sft_wtl_{tag}"] = r[f"wtl_{tag}"]
        else:
            tgt.update(r)
    cols += ["dpo_vs_sft_ufb", "dpo_vs_sft_wtl_ufb", "dpo_vs_sft_hh", "dpo_vs_sft_wtl_hh"]
    rows = sorted(merged.values(), key=lambda r: (r["data"], r["method"], r["seed"]))
    f1 = f"results_{a.model}.csv"
    with open(f1, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    # summary: mean ± std per (data, method)
    num = ["winrate_ufb", "winrate_hh", "acc_hh", "acc_ufb", "train_loss", "train_margins", "train_accuracies",
           "plc_p_tie", "plc_confidence", "dpo_vs_sft_ufb", "dpo_vs_sft_hh"]
    groups = {}
    for r in rows:
        groups.setdefault((r["data"], r["method"]), []).append(r)
    srows = []
    for (data, method), rs in sorted(groups.items()):
        s = {"data": data, "method": method, "n_seeds": len(rs)}
        for k in num:
            v = [r[k] for r in rs if k in r and r[k] is not None]
            if v:
                s[k + "_mean"] = round(st.mean(v), 2); s[k + "_std"] = round(st.stdev(v), 2) if len(v) > 1 else 0.0
        srows.append(s)
    scols = ["data", "method", "n_seeds"] + [k + suf for k in num for suf in ("_mean", "_std")]
    f2 = f"results_{a.model}_summary.csv"
    with open(f2, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=scols, extrasaction="ignore"); w.writeheader(); w.writerows(srows)
    print(f"{len(rows)} runs -> {f1}; {len(srows)} grupos -> {f2}")
    for s in srows:
        print(f"  {s['data']:24s} {s['method']:28s} n={s['n_seeds']}  UFB {s.get('winrate_ufb_mean','-')}±{s.get('winrate_ufb_std','-')}"
              f"  HH {s.get('winrate_hh_mean','-')}±{s.get('winrate_hh_std','-')}  acc_hh {s.get('acc_hh_mean','-')}  acc_ufb {s.get('acc_ufb_mean','-')}")


if __name__ == "__main__":
    main()
