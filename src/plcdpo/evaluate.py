"""Prepare prompts, generate responses, score them, and compare against DPO."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from .config import sha256_file


def read_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"Empty input: {path}")
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids) or any(row["id"] is None for row in rows):
        raise ValueError(f"Missing or duplicate IDs: {path}")
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    if not rows:
        raise ValueError("Refusing to write an empty evaluation result.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def write_metadata(args, **extra) -> None:
    values = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "function"}
    values.update(extra)
    values["output_sha256"] = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(json.dumps(values, indent=2) + "\n")


def prepare_prompts(args) -> None:
    from .eval_data import DATASET_LIMITS, load_eval_dataset

    limit = args.limit if args.limit is not None else DATASET_LIMITS[args.dataset]
    if limit <= 0:
        raise ValueError("limit must be positive")
    rows, _ = load_eval_dataset(args.dataset, limit)
    write_rows(args.output, rows)
    read_rows(args.output)
    write_metadata(
        args, count=len(rows), selection="first N rows in upstream order", protocol="legacy single-response"
    )


def generate(args) -> None:
    import yaml
    from transformers import AutoConfig, AutoTokenizer

    # Dense BF16 evaluation does not require optional FP8 kernels.
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

    rows = read_rows(args.input)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    tokenizer_path = args.base_model
    if args.adapter and (Path(args.adapter) / "tokenizer_config.json").is_file():
        tokenizer_path = args.adapter
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, revision=args.model_revision if tokenizer_path == args.base_model else None
    )
    if args.model_config:
        cfg = yaml.safe_load(args.model_config.read_text())
        tokenizer.chat_template = cfg["chat_template"]
    if args.max_model_length is None:
        model_config = AutoConfig.from_pretrained(args.base_model, revision=args.model_revision)
        args.max_model_length = min(int(getattr(model_config, "max_position_embeddings", 4096)), 4096)
    prompt_budget = args.max_model_length - args.max_new_tokens
    if prompt_budget <= 0:
        raise ValueError("max_model_length must exceed max_new_tokens.")
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        for row in rows
    ]
    stop_ids = {tokenizer.eos_token_id}
    for token in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            stop_ids.add(token_id)
    stop_ids.discard(None)
    if args.backend == "hf":
        results, truncated = generate_hf(args, rows, tokenizer, prompts, sorted(stop_ids), prompt_budget)
        write_rows(args.output, results)
        write_metadata(args, input_sha256=sha256_file(args.input), count=len(results),
                       truncated_prompts=truncated, prompt_truncation_side="left", backend="hf")
        return
    llm = LLM(
        model=args.base_model,
        revision=args.model_revision,
        enable_lora=True,
        max_lora_rank=16,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_length,
        seed=args.seed,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop_token_ids=sorted(stop_ids),
        seed=args.seed,
    )
    adapter = LoRARequest("policy", 1, lora_path=args.adapter) if args.adapter else None
    token_ids = [llm.get_tokenizer().encode(prompt) for prompt in prompts]
    truncated = sum(len(ids) > prompt_budget for ids in token_ids)
    requests = [{"prompt_token_ids": ids[-prompt_budget:]} for ids in token_ids]
    outputs = llm.generate(requests, sampling, lora_request=adapter)
    if len(outputs) != len(rows):
        raise RuntimeError("Generation returned an incomplete result.")
    results = [
        {**row, "response": out.outputs[0].text, "model": args.adapter or args.base_model}
        for row, out in zip(rows, outputs)
    ]
    write_rows(args.output, results)
    write_metadata(
        args,
        input_sha256=sha256_file(args.input),
        count=len(results),
        truncated_prompts=truncated,
        prompt_truncation_side="left",
    )


def generate_hf(args, rows, tokenizer, prompts, stop_ids, prompt_budget):
    """Fallback without vLLM (e.g. aarch64): transformers generate, greedy when temperature == 0."""
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, revision=args.model_revision, dtype=torch.bfloat16, device_map={"": 0}
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_ids = [tokenizer.encode(p) for p in prompts]
    truncated = sum(len(ids) > prompt_budget for ids in token_ids)
    token_ids = [ids[-prompt_budget:] for ids in token_ids]
    order = sorted(range(len(rows)), key=lambda i: len(token_ids[i]))
    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "eos_token_id": stop_ids, "pad_token_id": tokenizer.pad_token_id}
    if args.temperature <= 0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
    responses = [None] * len(rows)
    bs = args.hf_batch_size
    for b in tqdm(range(0, len(order), bs), desc="HF generation"):
        idx = order[b:b + bs]
        batch = tokenizer.pad({"input_ids": [token_ids[i] for i in idx]}, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**batch, **gen_kwargs)
        for n, i in enumerate(idx):
            new = out[n, batch["input_ids"].shape[1]:].tolist()
            cut = [k for k, t in enumerate(new) if t in stop_ids]
            responses[i] = tokenizer.decode(new[:cut[0]] if cut else new, skip_special_tokens=True)
    results = [{**row, "response": r, "model": args.adapter or args.base_model} for row, r in zip(rows, responses)]
    return results, truncated


def score(args) -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = read_rows(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model, revision=args.model_revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model,
        revision=args.model_revision,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        num_labels=1,
    ).eval()
    results = []
    for row in tqdm(rows, desc="Reward scoring"):
        prompt = row.get("raw_prompt", row.get("prompt"))
        if not isinstance(prompt, str) or not isinstance(row.get("response"), str):
            raise ValueError("Every row requires prompt and response strings.")
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": row["response"]},
        ]
        formatted = tokenizer.apply_chat_template(conversation, tokenize=False)
        if tokenizer.bos_token and formatted.startswith(tokenizer.bos_token):
            formatted = formatted[len(tokenizer.bos_token) :]
        inputs = tokenizer(formatted, return_tensors="pt")
        inputs = {key: value[:, : args.max_length].to(args.device) for key, value in inputs.items()}
        with torch.no_grad():
            reward = model(**inputs).logits[0][0].item()
        if not math.isfinite(reward):
            raise ValueError(f"Non-finite reward for ID {row['id']}")
        results.append({**row, "reward_score": reward})
    write_rows(args.output, results)
    write_metadata(args, input_sha256=sha256_file(args.input), count=len(results))


def compare_rows(baseline: list[dict], target: list[dict], score_key="reward_score") -> dict:
    def index(rows):
        result = {}
        for row in rows:
            key = str(row["id"])
            if row["id"] is None or key in result:
                raise ValueError("Missing or duplicate evaluation ID.")
            value = float(row[score_key])
            if not math.isfinite(value):
                raise ValueError("Scores must be finite.")
            result[key] = row
        return result

    base, candidate = index(baseline), index(target)
    if not base or set(base) != set(candidate):
        raise ValueError("Baseline and target must have identical, non-empty ID sets.")
    wins = ties = losses = 0
    for key in base:
        a, b = base[key], candidate[key]
        if a.get("raw_prompt", a.get("prompt")) != b.get("raw_prompt", b.get("prompt")):
            raise ValueError(f"Prompt mismatch for ID {key}")
        x, y = float(a[score_key]), float(b[score_key])
        wins += y > x
        ties += y == x
        losses += y < x
    return {
        "count": len(base),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": 100 * (wins + 0.5 * ties) / len(base),
        "score_key": score_key,
    }


def compare(args) -> None:
    result = compare_rows(read_rows(args.baseline), read_rows(args.input), args.score_key)
    result.update(baseline_sha256=sha256_file(args.baseline), target_sha256=sha256_file(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Freeze a prompt subset for all compared methods.")
    prep.add_argument(
        "--dataset",
        choices=(
            "ultrafeedback",
            "alpacaeval",
            "alpacaeval2",
            "vicuna",
            "mtbench-flat",
            "evol_instruct",
            "hh_rlhf",
        ),
        required=True,
    )
    prep.add_argument("--limit", type=int)
    prep.set_defaults(function=prepare_prompts)
    gen = commands.add_parser("generate")
    gen.add_argument("--base-model", required=True)
    gen.add_argument("--adapter", help="Final adapter or checkpoint directory; omit to evaluate SFT.")
    gen.add_argument("--model-config", type=Path, help="Model YAML containing the training chat_template.")
    gen.add_argument("--model-revision")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--temperature", type=float, default=0.5)
    gen.add_argument("--top-p", type=float, default=0.9)
    gen.add_argument("--max-new-tokens", type=int, default=1024)
    gen.add_argument(
        "--max-model-length", type=int, help="Defaults to the model context limit, capped at 4096."
    )
    gen.add_argument("--tensor-parallel-size", type=int, default=1)
    gen.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    gen.add_argument("--backend", choices=("vllm", "hf"), default="vllm", help="hf = transformers generate (no vLLM)")
    gen.add_argument("--hf-batch-size", type=int, default=16)
    gen.set_defaults(function=generate)
    scoring = commands.add_parser("score")
    scoring.add_argument("--reward-model", required=True)
    scoring.add_argument("--model-revision")
    scoring.add_argument("--device", default="cuda:0")
    scoring.add_argument("--max-length", type=int, default=4096)
    scoring.set_defaults(function=score)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--baseline", required=True, type=Path)
    comparison.add_argument("--score-key", default="reward_score")
    comparison.set_defaults(function=compare)
    for command in (prep, gen, scoring, comparison):
        command.add_argument("--output", required=True, type=Path)
    for command in (gen, scoring, comparison):
        command.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    args.function(args)


if __name__ == "__main__":
    main()
