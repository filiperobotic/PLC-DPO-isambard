#!/usr/bin/env bash
# Login node. Inventory of the experimental grid: what exists in $OUT vs what the plan expects; prints what is missing.
#   bash isambard/jobs/status.sh            # llama3-8b
set -uo pipefail
cd ${REPO:-$HOME/PLC-DPO} && source isambard/env.sh >/dev/null
MODEL=${1:-llama3-8b}
# expected grid: "DATA METHOD" lines (edit here when the plan grows)
GRID="ultrafeedback-binarized dpo
ultrafeedback-binarized plc-dpo-aggressive
ultrafeedback-binarized plc-dpo-aggressive-w64
ultrafeedback-binarized plc-dpo-aggressive-b8
ufb-rwgap20 dpo
ufb-rwgap20 plc-dpo-aggressive-w64
ufb-rwgap25 dpo
ufb-rwgap10 dpo
hh-rlhf dpo
hh-rlhf plc-dpo-aggressive-w64
hh-rwgap20 dpo"
ok() { [ -f "$1" ] && printf "✔" || printf "·"; }
printf "%-26s %-26s %-5s  %-7s %-8s %-8s %-8s %-8s %-7s %-7s\n" DATA METHOD SEED adapter scor_UFB scor_HH wr_UFB wr_HH acc_HH acc_UFB
MISSING_TRAIN=(); MISSING_EVAL=(); MISSING_WR=(); MISSING_ACC=()
while read -r D M; do
  [ -z "$D" ] && continue
  for S in 42 43 44; do
    R=$OUT/$D/$MODEL-$M-s$S; E=$R/eval
    A=$(ok $R/adapter_config.json); SU=$(ok $E/ultrafeedback.scores.jsonl); SH=$(ok $E/hh_rlhf.scores.jsonl)
    WU=$(ok $E/ultrafeedback.winrate_vs_dpo.json); WH=$(ok $E/hh_rlhf.winrate_vs_dpo.json)
    AH="·"; AU="·"
    [ -f $OUT/$D/accuracy_hh.json ]  && grep -q "\"$MODEL-$M-s$S\"" $OUT/$D/accuracy_hh.json  && AH="✔"
    [ -f $OUT/$D/accuracy_ufb.json ] && grep -q "\"$MODEL-$M-s$S\"" $OUT/$D/accuracy_ufb.json && AU="✔"
    # the DPO baseline of a base dataset has no win-rate vs itself
    if [ "$M" = dpo ] && [[ "$D" != *rwgap* ]]; then WU="ref"; WH="ref"; fi
    printf "%-26s %-26s %-5s  %-7s %-8s %-8s %-8s %-8s %-7s %-7s\n" "$D" "$M" "$S" "$A" "$SU" "$SH" "$WU" "$WH" "$AH" "$AU"
    [ "$A" = "✔" ] || { MISSING_TRAIN+=("$D/$M/s$S"); continue; }
    [ "$SU" = "✔" ] && [ "$SH" = "✔" ] || MISSING_EVAL+=("$D/$M/s$S")
    [ "$WU" = "✔" ] || [ "$WU" = ref ] && { [ "$WH" = "✔" ] || [ "$WH" = ref ]; } || MISSING_WR+=("$D/$M/s$S")
    [ "$AH" = "✔" ] && [ "$AU" = "✔" ] || MISSING_ACC+=("$D/$M/s$S")
  done
done <<< "$GRID"
echo; echo "SFT eval: UFB $(ok $OUT/sft-$MODEL/eval/ultrafeedback.scores.jsonl)  HH $(ok $OUT/sft-$MODEL/eval/hh_rlhf.scores.jsonl)"
echo "datasets filtrados: $(for N in ufb-rwgap20 ufb-rwgap25 ufb-rwgap10 hh-rwgap20; do printf "%s=%s " $N $(ok data/$N/train.parquet); done)"
echo "UFB test parquet (p/ acc_UFB): $(ok data/ultrafeedback-binarized/test.parquet)"
echo
echo "FALTA TREINAR  (${#MISSING_TRAIN[@]}): ${MISSING_TRAIN[*]:-nada}"
echo "FALTA AVALIAR  (${#MISSING_EVAL[@]}): ${MISSING_EVAL[*]:-nada}      -> make_eval_grid.sh + eval_array.sbatch"
echo "FALTA WIN-RATE (${#MISSING_WR[@]}): ${MISSING_WR[*]:-nada}      -> winrate.sh (login node)"
echo "FALTA ACURÁCIA (${#MISSING_ACC[@]}): ${MISSING_ACC[*]:-nada}      -> accuracy.sbatch por DATA"
echo
echo "Fila agora:"; squeue -u $USER -o "%.10i %.14j %.9T %.10M %.6D %R" 2>/dev/null | head -40
echo "Jobs não-COMPLETED desde 31/10:"; sacct -u $USER -S 2026-10-31 -X --format=JobID,JobName%16,State,Elapsed,ExitCode 2>/dev/null | grep -v -E "COMPLETED|RUNNING|PENDING|JobID|^---" || true
