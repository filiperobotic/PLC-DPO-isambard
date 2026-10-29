#!/bin/bash
# Job array for evaluation: one grid line = "MODEL RUN EVAL" (RUN = adapter dir or 'sft'); 1 GPU per task.
#   for R in $OUT/ultrafeedback-binarized/llama3-8b-*-s* sft; do for E in ultrafeedback hh_rlhf; do echo "llama3-8b $R $E"; done; done > eval_grid.txt
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
MODEL=$MODEL RUN=$RUN EVAL=$EVAL bash isambard/eval_run.sh
