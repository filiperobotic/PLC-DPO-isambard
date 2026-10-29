#!/bin/bash
# RwGap-R: implicit-reward margins of the DPO runs (3 seeds) on the training pairs -> remove the 20 % lowest (mean)
# -> data/<NAME>/ + configs/datasets/<NAME>.yaml. Then train with DATA=<NAME>.
#   sbatch --export=ALL,MODEL=llama3-8b,DATA=ultrafeedback-binarized,NAME=ufb-rwgap20 isambard/jobs/rwgap.sbatch
#SBATCH --job-name=plc-rwgap
#SBATCH --output=%x-%j.out
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
set -euo pipefail
cd ${REPO:-$HOME/PLC-DPO} && source isambard/env.sh
: "${MODEL:?}" "${DATA:?}" "${NAME:?}" "${SEEDS:=42 43 44}" "${FRAC:=0.2}"
ADAPTERS=""; for S in $SEEDS; do ADAPTERS="$ADAPTERS $OUT/$DATA/$MODEL-dpo-s$S"; done
for A in $ADAPTERS; do [ -f "$A/adapter_config.json" ] || { echo "missing adapter $A"; exit 1; }; done
mkdir -p $OUT/$DATA/rwgap
run_in_sif python isambard/implicit_reward.py score --model-config configs/models/$MODEL.yaml --adapter $ADAPTERS \
    --parquet data/$DATA/train.parquet --ref-cache $OUT/$DATA/rwgap/ref_logps_$MODEL.tsv \
    --out $OUT/$DATA/rwgap/margins_$MODEL.tsv
run_in_sif python isambard/implicit_reward.py filter --margins $OUT/$DATA/rwgap/margins_$MODEL.tsv \
    --data-dir data/$DATA --frac $FRAC --name $NAME
echo "RWGAP OK -> configs/datasets/$NAME.yaml"
