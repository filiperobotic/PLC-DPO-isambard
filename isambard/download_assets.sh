#!/usr/bin/env bash
# Login node only (internet). Downloads models + datasets into $HF_HOME and prepares the training parquet files.
# Skywork-Reward-V2 is gated on the Hub: run `hf auth login` (or export HF_TOKEN) before this script.
set -euo pipefail
cd "$(dirname "$0")/.." && source isambard/env.sh
export HF_HUB_OFFLINE=0
# login nodes kill memory-hungry processes: plain HTTP (no xet), 2 workers
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
HF="run_in_sif hf"   # huggingface-hub >= 1.0: the CLI is `hf` (huggingface-cli is deprecated)
$HF download AIR-hl/Qwen2.5-1.5B-ultrachat200k --max-workers 2
# glorgao/Qwen2.5-7B-SFT (paper's Qwen 7B SFT) is no longer on the Hub (Oct 2026): main 7-8B model = Llama-3-8B,
# whose SFT checkpoint (SimPO's, UltraChat-200k) is public and is what configs/models/llama3-8b.yaml points to.
$HF download princeton-nlp/Llama-3-Base-8B-SFT --max-workers 2
$HF download Skywork/Skywork-Reward-V2-Llama-3.1-8B --max-workers 2
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
[ -f data/ultrafeedback-binarized/test.parquet ] || run_in_sif python isambard/implicit_reward.py ufb-test
echo "assets ready under $HF_HOME and $REPO/data"
