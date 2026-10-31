# PLC-DPO on Isambard-AI (aarch64 / GH200)

Changes to the upstream repo (all opt-in, upstream behaviour untouched):
- `PLCDPO_NO_UNSLOTH=1` → `train.py`/`training.py` load with transformers + peft (LoRA r/alpha/targets as upstream),
  attention `sdpa` (set `PLCDPO_ATTN` to override), optimizer `adamw_torch` instead of `paged_adamw_32bit`
  (`PLCDPO_OPTIM` to override). Objective, data order, batch size, scheduler: unchanged.
- `evaluate.py generate --backend hf`: transformers generation (greedy at temperature 0) when vLLM is unavailable.
- `isambard/prepare_hh.py` + `configs/datasets/hh-rlhf.yaml`: Anthropic/hh-rlhf in the repo's parquet format.
- `configs/datasets/ufb-smoke.yaml`: 2000-pair subset for the smoke test.

## One-time setup (login node)
```bash
cd ~/PLC-DPO && source isambard/env.sh
# build on a compute node: mksquashfs gets killed by the login-node limits
srun --gpus=1 --time=01:00:00 --reservation=interactive apptainer build --fakeroot $SIF isambard/plcdpo-train.def
hf auth login                    # Skywork-Reward-V2 is gated (or export HF_TOKEN)
bash isambard/download_assets.sh # models, datasets, parquet files, eval prompts -> $HF_HOME and data/
```
## Smoke test (1 GPU, ~1 h)
```bash
sbatch isambard/jobs/smoke.sbatch && squeue --me
```
Expected at the end of `plc-smoke-<jobid>.out`: an adapter in `$OUT/smoke/qwen1.5b-dpo-s42`, 50 generations, 50 rewards
and `self-winrate.json` with `win_rate: 50.0`, then `SMOKE OK`.

## Training runs (1 GPU each; queue several)
Main 7-8B model: Llama-3-8B (`princeton-nlp/Llama-3-Base-8B-SFT`); the paper's `glorgao/Qwen2.5-7B-SFT` is gone from the Hub.
```bash
for S in 42 43 44; do for M in dpo plc-dpo-aggressive; do
  sbatch --export=ALL,MODEL=llama3-8b,DATA=ultrafeedback-binarized,METHOD=$M,SEED=$S isambard/jobs/train.sbatch
done; done
```
Storage: code + `data/` in `$HOME` (quota 100 GB; parquet files are small), HF cache + images in `$PROJECTDIR/$USER`,
adapters/generations in `$SCRATCHDIR/plcdpo`.

## Phase 1 — UFB, Llama-3-8B (3 seeds): DPO vs PLC-DPO vs RwGap-R→DPO
```bash
# 1) baselines + PLC (6 independent 1-GPU tasks, as a job array)
for S in 42 43 44; do for M in dpo plc-dpo-aggressive; do echo "llama3-8b ultrafeedback-binarized $M $S"; done; done > grid_ufb.txt
sbatch --array=0-5 --export=ALL,GRID=grid_ufb.txt isambard/jobs/train_array.sbatch
#    (equivalent: one `sbatch --export=ALL,MODEL=..,DATA=..,METHOD=..,SEED=.. isambard/jobs/train.sbatch` per line)
# 2) after the 3 DPO runs: implicit-reward margins (mean of the 3 DPO) -> remove 20 % -> data/ufb-rwgap20 (+ its yaml)
sbatch --export=ALL,MODEL=llama3-8b,DATA=ultrafeedback-binarized,NAME=ufb-rwgap20 isambard/jobs/rwgap.sbatch
# 3) DPO on the filtered data (3 jobs); outputs land in $OUT/ufb-rwgap20/
for S in 42 43 44; do sbatch --export=ALL,MODEL=llama3-8b,DATA=ufb-rwgap20,METHOD=dpo,SEED=$S isambard/jobs/train.sbatch; done
# 4) generation + judge for every run on both prompt sets (job array: one task per run × set; SFT too)
bash isambard/jobs/make_eval_grid.sh llama3-8b ultrafeedback-binarized ufb-rwgap20 > eval_grid.txt   # only unscored pairs; needs no env.sh
sbatch --array=0-$(( $(wc -l < eval_grid.txt) - 1 )) --export=ALL,GRID=eval_grid.txt isambard/jobs/eval_array.sbatch
# 5) win-rates vs the DPO run of the same seed (login node, seconds); filtered runs live in another DATA dir:
bash isambard/jobs/winrate.sh llama3-8b ultrafeedback-binarized ultrafeedback
bash isambard/jobs/winrate.sh llama3-8b ultrafeedback-binarized hh_rlhf
bash isambard/jobs/winrate.sh llama3-8b ufb-rwgap20 ultrafeedback ultrafeedback-binarized   # filtered runs vs the DPO baselines
bash isambard/jobs/winrate.sh llama3-8b ufb-rwgap20 hh_rlhf ultrafeedback-binarized
# 5b) one CSV per run + mean±std per method (login node, seconds; also computes DPO-vs-SFT win-rates)
python3 isambard/jobs/collect_results.py llama3-8b ultrafeedback-binarized ufb-rwgap20   # -> results_llama3-8b.csv, results_llama3-8b_summary.csv
# 6) pairwise accuracy of the implicit reward on the test sets (human labels on HH; GPT-4 labels on UFB)
sbatch --export=ALL,MODEL=llama3-8b,DATA=ultrafeedback-binarized,TEST=data/hh-rlhf/test.parquet,TAG=hh isambard/jobs/accuracy.sbatch
sbatch --export=ALL,MODEL=llama3-8b,DATA=ultrafeedback-binarized,TEST=data/ultrafeedback-binarized/test.parquet,TAG=ufb isambard/jobs/accuracy.sbatch
```
## Phase 1b (UFB, after the first results) and Phase 2 (HH-RLHF)
```bash
# UFB test labels (GPT-4) for the implicit-reward accuracy — login node, dataset is already in the HF cache
run_in_sif python isambard/implicit_reward.py ufb-test
sbatch --export=ALL,MODEL=llama3-8b,DATA=ultrafeedback-binarized,TEST=data/ultrafeedback-binarized/test.parquet,TAG=ufb isambard/jobs/accuracy.sbatch
sbatch --export=ALL,MODEL=llama3-8b,DATA=ufb-rwgap20,TEST=data/hh-rlhf/test.parquet,TAG=hh isambard/jobs/accuracy.sbatch
sbatch --export=ALL,MODEL=llama3-8b,DATA=ufb-rwgap20,TEST=data/ultrafeedback-binarized/test.parquet,TAG=ufb isambard/jobs/accuracy.sbatch
# PLC "as the paper ran it": 8 margins per std estimate (per_device_train_batch_size 8 x accumulation 8)
for S in 42 43 44; do sbatch --export=ALL,MODEL=llama3-8b,DATA=ultrafeedback-binarized,METHOD=plc-dpo-aggressive-b8,SEED=$S isambard/jobs/train.sbatch; done
# other filtering fractions from the SAME margins file (login node, seconds): natural zero (25 %) and 10 %
for F in 0.25 0.10; do N=ufb-rwgap$(python3 -c "print(int($F*100))"); run_in_sif python isambard/implicit_reward.py filter \
  --margins $OUT/ultrafeedback-binarized/rwgap/margins_llama3-8b.tsv --data-dir data/ultrafeedback-binarized --frac $F --name $N; done
for D in ufb-rwgap25 ufb-rwgap10; do for S in 42 43 44; do sbatch --export=ALL,MODEL=llama3-8b,DATA=$D,METHOD=dpo,SEED=$S isambard/jobs/train.sbatch; done; done
# filter + PLC-w64 (are the two gains additive?)
for S in 42 43 44; do sbatch --export=ALL,MODEL=llama3-8b,DATA=ufb-rwgap20,METHOD=plc-dpo-aggressive-w64,SEED=$S isambard/jobs/train.sbatch; done
# Phase 2 — HH-RLHF (160 k pairs, ~14 h per job): DPO + PLC-w64 first; rwgap needs the 3 DPO, then DPO on the filtered set
for M in dpo plc-dpo-aggressive-w64; do for S in 42 43 44; do sbatch --export=ALL,MODEL=llama3-8b,DATA=hh-rlhf,METHOD=$M,SEED=$S isambard/jobs/train.sbatch; done; done
sbatch --export=ALL,MODEL=llama3-8b,DATA=hh-rlhf,NAME=hh-rwgap20 isambard/jobs/rwgap.sbatch     # after the 3 HH DPO
# evaluation / win-rates / CSV: same as steps 4-5b with the new DATA dirs, e.g.
bash isambard/jobs/make_eval_grid.sh llama3-8b ultrafeedback-binarized ufb-rwgap20 ufb-rwgap25 ufb-rwgap10 hh-rlhf hh-rwgap20 > eval_grid.txt
bash isambard/jobs/winrate.sh llama3-8b hh-rlhf ultrafeedback; bash isambard/jobs/winrate.sh llama3-8b hh-rwgap20 hh_rlhf hh-rlhf
python3 isambard/jobs/collect_results.py llama3-8b ultrafeedback-binarized ufb-rwgap20 ufb-rwgap25 ufb-rwgap10
python3 isambard/jobs/collect_results.py llama3-8b hh-rlhf hh-rwgap20 --baseline-data hh-rlhf --tag hh   # HH-trained runs vs HH DPO
```
