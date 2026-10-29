#!/bin/bash
# Job array: one training per array task, 1 GPU each, read from a grid file (one line = MODEL DATA METHOD SEED).
#   printf "%s\n" "llama3-8b ultrafeedback-binarized dpo 42" "llama3-8b ultrafeedback-binarized plc-dpo-aggressive 42" ... > grid.txt
#   sbatch --array=0-$(( $(wc -l < grid.txt) - 1 )) --export=ALL,GRID=grid.txt isambard/jobs/train_array.sbatch
#   (--array=0-5%2 limits the number running at once; scancel <jobid>_<task> cancels one task)
#SBATCH --job-name=plc-train
#SBATCH --output=%x-%A_%a.out
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:00:00
set -euo pipefail
cd ${REPO:-$HOME/PLC-DPO} && source isambard/env.sh
export PYTHONPATH=$REPO/src
: "${GRID:?grid file with one 'MODEL DATA METHOD SEED' per line}"
read -r MODEL DATA METHOD SEED < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$GRID")
echo "task $SLURM_ARRAY_TASK_ID: MODEL=$MODEL DATA=$DATA METHOD=$METHOD SEED=$SEED"
RUN=$OUT/$DATA/$MODEL-$METHOD-s$SEED
if [ -f "$RUN/adapter_config.json" ]; then echo "already done: $RUN"; exit 0; fi
rm -rf "$RUN"
run_in_sif bash scripts/train.sh --config configs/train.yaml configs/models/$MODEL.yaml \
    configs/datasets/$DATA.yaml configs/methods/$METHOD.yaml \
    --set output_dir=$RUN seed=$SEED attn_implementation=sdpa
echo "done: $RUN"
