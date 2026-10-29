#!/bin/bash
# Pairwise agreement of the implicit reward with test labels (HH-RLHF test = human; UFB test_prefs = GPT-4).
#   sbatch --export=ALL,MODEL=llama3-8b,DATA=ultrafeedback-binarized,TEST=data/hh-rlhf/test.parquet,TAG=hh isambard/jobs/accuracy.sbatch
#   evaluates every run $OUT/$DATA/$MODEL-*-s* (dpo, plc, filtered) -> $OUT/$DATA/accuracy_<TAG>.json
#SBATCH --job-name=plc-acc
#SBATCH --output=%x-%j.out
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
set -euo pipefail
cd ${REPO:-$HOME/PLC-DPO} && source isambard/env.sh
: "${MODEL:?}" "${DATA:?}" "${TEST:?}" "${TAG:?}"
ADAPTERS=$(ls -d $OUT/$DATA/$MODEL-*-s*/ | grep -v '/eval' | xargs -n1 -I{} sh -c '[ -f {}/adapter_config.json ] && echo {}' | tr '\n' ' ')
echo "adapters: $ADAPTERS"
mkdir -p $OUT/$DATA/rwgap
run_in_sif python isambard/implicit_reward.py accuracy --model-config configs/models/$MODEL.yaml --adapter $ADAPTERS \
    --parquet $TEST --ref-cache $OUT/$DATA/rwgap/ref_logps_${MODEL}_$TAG.tsv --out $OUT/$DATA/accuracy_$TAG.json
echo "ACC OK"
