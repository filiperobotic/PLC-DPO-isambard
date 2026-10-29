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
