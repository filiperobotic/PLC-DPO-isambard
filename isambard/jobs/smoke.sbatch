#!/bin/bash
# Phase 0 smoke test: DPO on Qwen2.5-1.5B with 2000 UFB pairs (~31 optimizer steps), HF-backend generation on
# 50 prompts, Skywork scoring, win-rate of the run against itself (must be 50.0). One GPU, ~1 h.
#   sbatch isambard/jobs/smoke.sbatch
#SBATCH --job-name=plc-smoke
#SBATCH --output=%x-%j.out
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
set -euo pipefail
cd ${REPO:-$HOME/PLC-DPO} && source isambard/env.sh
export PYTHONPATH=$REPO/src
RUN=$OUT/smoke/qwen1.5b-dpo-s42
rm -rf "$RUN"; mkdir -p "$OUT/smoke"
nvidia-smi --list-gpus
run_in_sif bash scripts/train.sh --config configs/train.yaml configs/models/qwen2.5-1.5b.yaml \
    configs/datasets/ufb-smoke.yaml configs/methods/dpo.yaml \
    --set output_dir=$RUN seed=42 attn_implementation=sdpa
ls "$RUN"
run_in_sif bash scripts/evaluate.sh generate --backend hf --input data/eval/ufb-smoke.jsonl \
    --base-model AIR-hl/Qwen2.5-1.5B-ultrachat200k --adapter $RUN --model-config configs/models/qwen2.5-1.5b.yaml \
    --temperature 0 --top-p 1 --max-new-tokens 512 --output $OUT/smoke/pred.jsonl
run_in_sif bash scripts/evaluate.sh score --input $OUT/smoke/pred.jsonl \
    --reward-model Skywork/Skywork-Reward-V2-Llama-3.1-8B --output $OUT/smoke/scores.jsonl
run_in_sif bash scripts/evaluate.sh compare --baseline $OUT/smoke/scores.jsonl --input $OUT/smoke/scores.jsonl \
    --output $OUT/smoke/self-winrate.json
echo SMOKE OK
