#!/usr/bin/env bash
# Login node. Writes eval_grid.txt ("MODEL RUN EVAL" per line) for every finished adapter in the given DATA dirs + sft,
# skipping (run, set) pairs that already have <run>/eval/<set>.scores.jsonl. Sources env.sh itself, so $OUT is always set.
#   bash isambard/jobs/make_eval_grid.sh llama3-8b ultrafeedback-binarized ufb-rwgap20 > eval_grid.txt
#   sbatch --array=0-$(( $(wc -l < eval_grid.txt) - 1 )) --export=ALL,GRID=eval_grid.txt isambard/jobs/eval_array.sbatch
set -euo pipefail
cd ${REPO:-$HOME/PLC-DPO} && source isambard/env.sh >/dev/null
MODEL=$1; shift
for D in "$@"; do
  for R in $OUT/$D/$MODEL-*-s*/; do
    R=${R%/}; [ -f "$R/adapter_config.json" ] || continue
    for E in ultrafeedback hh_rlhf; do [ -f "$R/eval/$E.scores.jsonl" ] || echo "$MODEL $R $E"; done
  done
done
for E in ultrafeedback hh_rlhf; do [ -f "$OUT/sft-$MODEL/eval/$E.scores.jsonl" ] || echo "$MODEL sft $E"; done
