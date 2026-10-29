#!/bin/bash
# One evaluation (generation + judge) per job, 1 GPU. See isambard/eval_run.sh for the parameters.
#   sbatch --export=ALL,MODEL=llama3-8b,RUN=$OUT/ultrafeedback-binarized/llama3-8b-plc-dpo-aggressive-s42,EVAL=ultrafeedback isambard/jobs/eval.sbatch
#SBATCH --job-name=plc-eval
#SBATCH --output=%x-%j.out
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
set -euo pipefail
cd ${REPO:-$HOME/PLC-DPO}
bash isambard/eval_run.sh
