#!/usr/bin/env bash
# Generate (greedy) on data/eval/<EVAL>.jsonl and score with the reward judge for one run. Called by the jobs.
#   MODEL=llama3-8b RUN=<adapter dir>|sft EVAL=ultrafeedback|hh_rlhf [BACKEND=hf|vllm] [MAXNEW=1024] bash isambard/eval_run.sh
set -euo pipefail
cd ${REPO:-$HOME/PLC-DPO} && source isambard/env.sh
export PYTHONPATH=$REPO/src
: "${MODEL:?}" "${RUN:?}" "${EVAL:?}" "${BACKEND:=hf}" "${MAXNEW:=1024}"
BASE=$(grep -m1 '^model_name_or_path:' configs/models/$MODEL.yaml | awk '{print $2}')
if [ "$RUN" = "sft" ]; then EVD=$OUT/sft-$MODEL/eval; ADAPT=""; else EVD=$RUN/eval; ADAPT="--adapter $RUN"; fi
mkdir -p $EVD
PRED=$EVD/$EVAL.pred.jsonl; SC=$EVD/$EVAL.scores.jsonl
if [ ! -f $PRED ]; then
  if [ "$BACKEND" = "vllm" ]; then
    apptainer exec --nv -B $PROJECTDIR,$SCRATCHDIR,$HOME --env HF_HOME=$HF_HOME,HF_HUB_OFFLINE=1 $SIF_EVAL \
      bash scripts/evaluate.sh generate --input data/eval/$EVAL.jsonl --base-model $BASE $ADAPT \
      --model-config configs/models/$MODEL.yaml --temperature 0 --top-p 1 --max-new-tokens $MAXNEW --output $PRED
  else
    run_in_sif bash scripts/evaluate.sh generate --backend hf --hf-batch-size 16 --input data/eval/$EVAL.jsonl \
      --base-model $BASE $ADAPT --model-config configs/models/$MODEL.yaml --temperature 0 --top-p 1 \
      --max-new-tokens $MAXNEW --output $PRED
  fi
fi
[ -f $SC ] || run_in_sif bash scripts/evaluate.sh score --input $PRED --reward-model Skywork/Skywork-Reward-V2-Llama-3.1-8B --output $SC
echo "EVAL OK -> $SC"
