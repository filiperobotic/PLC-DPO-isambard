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


def gen_stats(pred: Path):
    """mean response length (words) and mean judge score of one .pred/.scores.jsonl"""
    out = {}
    sc = pred.with_name(pred.name.replace(".pred.", ".scores."))
    f = sc if sc.exists() else pred
    if not f.exists():
        return out
    lens, scores = [], []
    for line in open(f, errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if isinstance(r.get("response"), str):
            lens.append(len(r["response"].split()))
        if "reward_score" in r:
            scores.append(float(r["reward_score"]))
    if lens:
        out["len"] = round(st.mean(lens), 1)
    if scores:
        out["score"] = round(st.mean(scores), 3)
    return out


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


METODO = {"dpo": "DPO", "plc-dpo-aggressive": "PLC-DPO oficial (1 GPU)", "plc-dpo-aggressive-w64": "PLC-DPO w64",
          "plc-dpo-aggressive-b8": "PLC-DPO b8 (8 margens/std)"}
PT = {"braco": "braço", "data": "base de treino", "method": "método (config)", "seed": "seed", "run": "run",
      "winrate_ufb": "win-rate vs DPO, prompts UFB (2000)", "wtl_ufb": "vitórias/empates/derrotas UFB",
      "winrate_hh": "win-rate vs DPO, prompts HH-RLHF (805)", "wtl_hh": "vitórias/empates/derrotas HH",
      "acc_hh": "acurácia recompensa implícita, HH test humano (%)",
      "len_ufb": "comprimento médio da resposta UFB (palavras)", "len_hh": "comprimento médio da resposta HH (palavras)",
      "score_ufb": "nota média do juiz UFB", "score_hh": "nota média do juiz HH", "acc_ufb": "acurácia recompensa implícita, UFB test GPT-4 (%)",
      "train_loss": "train_loss", "train_margins": "margem implícita (treino)", "train_accuracies": "acurácia (treino)",
      "plc_margin_std_ema": "PLC std da margem", "plc_p_clean": "PLC p_clean", "plc_p_flip": "PLC p_flip", "plc_p_tie": "PLC p_tie",
      "plc_confidence": "PLC confiança", "plc_mixture_weight": "PLC peso de mistura", "train_runtime_h": "horas de treino",
      "scored_ufb": "avaliado em UFB", "scored_hh": "avaliado em HH",
      "dpo_vs_sft_ufb": "DPO vs SFT, UFB", "dpo_vs_sft_wtl_ufb": "DPO vs SFT v/e/d UFB", "dpo_vs_sft_hh": "DPO vs SFT, HH",
      "dpo_vs_sft_wtl_hh": "DPO vs SFT v/e/d HH", "n_seeds": "nº seeds"}


def braco(data, method, bdata):
    nome = METODO.get(method, method)
    m = re.search(r"rwgap(\d+)", data)
    if m:
        nome = f"RwGap {m.group(1)}% -> {nome}"
    elif data != bdata:
        nome = f"{nome} [{data}]"
    if method == "dpo" and data == bdata:
        nome += " (referência = 50)"
    return nome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("data", nargs="+")
    ap.add_argument("--baseline-data", default=None, help="where the DPO baselines live (default: first data dir)")
    ap.add_argument("--tag", default="", help="suffix for the csv names (e.g. hh)")
    a = ap.parse_args()
    OUT = out_dir(); repo = Path.cwd(); bdata = a.baseline_data or a.data[0]
    rows = []
    for data in a.data:
        acc = {t: jload(OUT / data / f"accuracy_{t}.json") or {} for t in ("hh", "ufb")}
        for run in sorted((OUT / data).glob(f"{a.model}-*-s*")):
            if not (run / "adapter_config.json").exists():
                continue
            method, seed = run.name[len(a.model) + 1:].rsplit("-s", 1)
            r = {"braco": braco(data, method, bdata), "data": data, "method": method, "seed": int(seed), "run": run.name}
            for ev in EVALS:
                w = jload(run / "eval" / f"{ev}.winrate_vs_dpo.json")
                tag = "ufb" if ev == "ultrafeedback" else "hh"
                if w:
                    r[f"winrate_{tag}"] = round(w["win_rate"], 2)
                    r[f"wtl_{tag}"] = f"{w['wins']}/{w['ties']}/{w['losses']}"
                elif method == "dpo" and data == bdata:
                    r[f"winrate_{tag}"] = 50.0; r[f"wtl_{tag}"] = "baseline"
                r[f"scored_{tag}"] = (run / "eval" / f"{ev}.scores.jsonl").exists()
                g = gen_stats(run / "eval" / f"{ev}.pred.jsonl")
                if "len" in g: r[f"len_{tag}"] = g["len"]
                if "score" in g: r[f"score_{tag}"] = g["score"]
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
    cols = ["braco", "data", "method", "seed", "run", "winrate_ufb", "wtl_ufb", "winrate_hh", "wtl_hh", "acc_hh", "acc_ufb", "len_ufb", "len_hh", "score_ufb", "score_hh", "train_loss",
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
    ordem = lambda r: (r["data"] != bdata, r["method"] != "dpo", r["data"], r["method"], r["seed"])  # baseline primeiro
    rows = sorted(merged.values(), key=ordem)
    f1 = f"results_{a.model}{'_'+a.tag if a.tag else ''}.csv"
    with open(f1, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[PT.get(c, c) for c in cols]); w.writeheader()
        w.writerows([{PT.get(k, k): v for k, v in r.items() if k in cols} for r in rows])
    # summary: mean ± std per (data, method)
    num = ["winrate_ufb", "winrate_hh", "acc_hh", "acc_ufb", "len_ufb", "len_hh", "score_ufb", "score_hh", "train_loss", "train_margins", "train_accuracies",
           "plc_p_tie", "plc_confidence", "dpo_vs_sft_ufb", "dpo_vs_sft_hh"]
    groups = {}
    for r in rows:
        groups.setdefault((r["data"], r["method"]), []).append(r)
    srows = []
    for (data, method), rs in sorted(groups.items(), key=lambda kv: ordem(kv[1][0])):
        s = {"braco": rs[0]["braco"], "data": data, "method": method, "n_seeds": len(rs)}
        for k in num:
            v = [r[k] for r in rs if k in r and r[k] is not None]
            if v:
                s[k + "_mean"] = round(st.mean(v), 2); s[k + "_std"] = round(st.stdev(v), 2) if len(v) > 1 else 0.0
        srows.append(s)
    scols = ["braco", "data", "method", "n_seeds"] + [k + suf for k in num for suf in ("_mean", "_std")]
    hdr = lambda c: PT.get(c, c) if c in PT else (PT.get(c[:-5], c[:-5]) + (" — média" if c.endswith("_mean") else " — desvio"))
    f2 = f"results_{a.model}{'_'+a.tag if a.tag else ''}_summary.csv"
    with open(f2, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[hdr(c) for c in scols]); w.writeheader()
        w.writerows([{hdr(k): v for k, v in r.items() if k in scols} for r in srows])
    print(f"{len(rows)} runs -> {f1}; {len(srows)} grupos -> {f2}")
    for s in srows:
        print(f"  {s['braco']:40s} n={s['n_seeds']}  UFB {s.get('winrate_ufb_mean','-')}±{s.get('winrate_ufb_std','-')}"
              f"  HH {s.get('winrate_hh_mean','-')}±{s.get('winrate_hh_std','-')}  acc_hh {s.get('acc_hh_mean','-')}  acc_ufb {s.get('acc_ufb_mean','-')}")


if __name__ == "__main__":
    main()
