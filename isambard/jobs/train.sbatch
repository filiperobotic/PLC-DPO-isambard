#!/bin/bash
# One training run per job (1 GPU). Parameters by environment:
#   MODEL=qwen2.5-7b  DATA=<name of configs/datasets/<DATA>.yaml>  METHOD=dpo|plc-dpo-aggressive  SEED=42
#   sbatch --export=ALL,MODEL=qwen2.5-7b,DATA=ultrafeedback-binarized,METHOD=dpo,SEED=42 isambard/jobs/train.sbatch
# Filtered versions (phase 1) are just another dataset YAML. Output: $OUT/<DATA>/<MODEL>-<METHOD>-s<SEED>
#SBATCH --job-name=plc-train
#SBATCH --output=%x-%j.out
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:00:00
set -euo pipefail
cd ${REPO:-$HOME/PLC-DPO} && source isambard/env.sh
export PYTHONPATH=$REPO/src
: "${MODEL:?}" "${DATA:?}" "${METHOD:?}" "${SEED:=42}"
TAG=${EPOCHS:+-e$EPOCHS}                      # EPOCHS=3 -> run name <METHOD>-e3-s<SEED>, num_train_epochs=3 (default: 1 epoch, no tag)
RUN=$OUT/$DATA/$MODEL-$METHOD$TAG-s$SEED
if [ -f "$RUN/adapter_config.json" ]; then echo "already done: $RUN"; exit 0; fi
rm -rf "$RUN"
run_in_sif bash scripts/train.sh --config configs/train.yaml configs/models/$MODEL.yaml \
    configs/datasets/$DATA.yaml configs/methods/$METHOD.yaml \
    --set output_dir=$RUN seed=$SEED attn_implementation=sdpa ${EPOCHS:+num_train_epochs=$EPOCHS}
echo "done: $RUN"
