#!/usr/bin/env bash
# Login node, no GPU: win-rate of every run against the DPO run with the SAME seed (paper protocol) + SFT vs DPO.
#   bash isambard/jobs/winrate.sh llama3-8b ultrafeedback-binarized ultrafeedback            # runs vs same-seed DPO
#   bash isambard/jobs/winrate.sh llama3-8b ufb-rwgap20 ultrafeedback ultrafeedback-binarized  # 4th arg: where the DPO baselines are
set -euo pipefail
cd ${REPO:-$HOME/PLC-DPO} && source isambard/env.sh
export PYTHONPATH=$REPO/src
MODEL=$1; DATA=$2; EVAL=$3; BDATA=${4:-$2}
for R in $OUT/$DATA/$MODEL-*-s*/; do
  R=${R%/}; S=${R##*-s}; T=$R/eval/$EVAL.scores.jsonl; B=$OUT/$BDATA/$MODEL-dpo-s$S/eval/$EVAL.scores.jsonl
  [ -f "$T" ] && [ -f "$B" ] || continue
  W=$R/eval/$EVAL.winrate_vs_dpo.json; rm -f $W
  run_in_sif bash scripts/evaluate.sh compare --baseline $B --input $T --output $W >/dev/null
  printf "%-60s %s  win_rate=%s\n" "$(basename $R)" "$EVAL" "$(python3 -c "import json;print(json.load(open('$W'))['win_rate'])")"
done
