#!/bin/bash
# Job array for evaluation: one grid line = "MODEL RUN EVAL" (RUN = adapter dir or 'sft'); 1 GPU per task.
#   bash isambard/jobs/make_eval_grid.sh llama3-8b ultrafeedback-binarized ufb-rwgap20 > eval_grid.txt   # sources env.sh; skips scored pairs
#   sbatch --array=0-$(( $(wc -l < eval_grid.txt) - 1 )) --export=ALL,GRID=eval_grid.txt isambard/jobs/eval_array.sbatch
#SBATCH --job-name=plc-eval
#SBATCH --output=%x-%A_%a.out
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
set -euo pipefail
cd ${REPO:-$HOME/PLC-DPO}
: "${GRID:?grid file with one 'MODEL RUN EVAL' per line}"
read -r MODEL RUN EVAL < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$GRID")
echo "task $SLURM_ARRAY_TASK_ID: MODEL=$MODEL RUN=$RUN EVAL=$EVAL"
if [ "$RUN" != "sft" ]; then
  source isambard/env.sh >/dev/null
  [ -d "$RUN" ] || RUN=$OUT/$RUN                      # allow RUN relative to $OUT (e.g. ufb-rwgap20/llama3-8b-dpo-s42)
  [ -f "$RUN/adapter_config.json" ] || { echo "ERRO: $RUN não é um adaptador (grade gerada sem 'source isambard/env.sh'?)"; exit 1; }
fi
MODEL=$MODEL RUN=$RUN EVAL=$EVAL bash isambard/eval_run.sh
