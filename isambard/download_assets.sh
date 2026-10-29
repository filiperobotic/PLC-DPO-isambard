#!/usr/bin/env bash
# Login node only (internet). Downloads models + datasets into $HF_HOME and prepares the training parquet files.
# Skywork-Reward-V2 is gated on the Hub: run `hf auth login` (or export HF_TOKEN) before this script.
set -euo pipefail
cd "$(dirname "$0")/.." && source isambard/env.sh
export HF_HUB_OFFLINE=0
HF="run_in_sif hf"   # huggingface-hub >= 1.0: the CLI is `hf` (huggingface-cli is deprecated)
$HF download AIR-hl/Qwen2.5-1.5B-ultrachat200k
$HF download glorgao/Qwen2.5-7B-SFT
$HF download Skywork/Skywork-Reward-V2-Llama-3.1-8B
$HF download --repo-type dataset HuggingFaceH4/ultrafeedback_binarized
$HF download --repo-type dataset Anthropic/hh-rlhf
# training data in the repo's format (plcdpo.prepare_data fetches the dataset revision from the Hub)
export PYTHONPATH=$REPO/src
[ -d data/ultrafeedback-binarized ] || run_in_sif python -m plcdpo.prepare_data --dataset ultrafeedback-binarized
[ -d data/ufb-smoke ] || run_in_sif python -m plcdpo.prepare_data --dataset ultrafeedback-binarized --max-samples 2000 --output-dir data/ufb-smoke
[ -d data/hh-rlhf ] || run_in_sif python isambard/prepare_hh.py --output data/hh-rlhf
# frozen evaluation prompts (first N rows upstream, as in the paper: UFB 2000, HH-RLHF 805)
mkdir -p data/eval
[ -f data/eval/ultrafeedback.jsonl ] || run_in_sif bash scripts/evaluate.sh prepare --dataset ultrafeedback --output data/eval/ultrafeedback.jsonl
[ -f data/eval/hh_rlhf.jsonl ] || run_in_sif bash scripts/evaluate.sh prepare --dataset hh_rlhf --output data/eval/hh_rlhf.jsonl
[ -f data/eval/ufb-smoke.jsonl ] || run_in_sif bash scripts/evaluate.sh prepare --dataset ultrafeedback --limit 50 --output data/eval/ufb-smoke.jsonl
echo "assets ready under $HF_HOME and $REPO/data"
