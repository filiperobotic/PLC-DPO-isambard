"""Prompt loaders preserving the legacy single-response evaluation surface."""

import re

from datasets import load_dataset

ULTRAFEEDBACK_PREFS_PARQUET = (
    "hf://datasets/HuggingFaceH4/ultrafeedback_binarized/data/test_prefs-00000-of-00001.parquet"
)
FASTCHAT_DATA_BASE_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data"
DATASET_LIMITS = {
    "ultrafeedback": 2000,
    "alpacaeval": 805,
    "alpacaeval2": 805,
    "vicuna": 80,
    "mtbench-flat": 80,
    "evol_instruct": 805,
    "hh_rlhf": 805,
}


def load_ultrafeedback_test_prefs():
    try:
        return load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    except TypeError as exc:
        msg = str(exc)
        if "dataclass type or instance" not in msg:
            raise

        print("Falling back to direct parquet loading...")
        return load_dataset(
            "parquet",
            data_files={"test": ULTRAFEEDBACK_PREFS_PARQUET},
            split="test",
        )


def _select_n_samples(dataset, n_samples):
    if n_samples is None:
        return dataset
    return dataset.select(range(min(n_samples, len(dataset))))


def _row_id(row, index):
    for key in ("id", "prompt_id", "question_id"):
        value = row.get(key)
        if value is not None:
            return value
    return index


def _messages_to_prompt(messages):
    if not messages:
        return ""
    user_parts = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
    if user_parts:
        return "\n\n".join(part for part in user_parts if part)
    return "\n\n".join(msg.get("content", "") for msg in messages if msg.get("content"))


def _normalize_prompt_dataset(dataset, prompt_getter, dataset_name):
    rows = []
    for i, row in enumerate(dataset):
        prompt = prompt_getter(row)
        if prompt is None:
            continue
        prompt = str(prompt).strip()
        if not prompt:
            continue
        rows.append(
            {
                "id": _row_id(row, i),
                "prompt": prompt,
                "eval_dataset": dataset_name,
                "metadata": {
                    key: row[key] for key in ("category", "dataset", "question_id", "prompt_id") if key in row
                },
            }
        )
    return rows


def load_eval_dataset(eval_dataset, n_samples=None):
    dataset_name = eval_dataset

    if dataset_name == "ultrafeedback":
        dataset = load_ultrafeedback_test_prefs()
        dataset = _select_n_samples(dataset, n_samples)
        return _normalize_prompt_dataset(dataset, lambda row: row.get("prompt"), dataset_name), dataset_name

    if dataset_name in {"alpacaeval", "alpacaeval2"}:
        # AlpacaEval 2 uses the same 805 prompts as AlpacaEval; the difference is
        # the reference/baseline evaluator used downstream.
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
        dataset = load_dataset("json", data_files=path, split="train")
        dataset = _select_n_samples(dataset, n_samples)
        return _normalize_prompt_dataset(
            dataset, lambda row: row.get("instruction"), dataset_name
        ), dataset_name

    if dataset_name == "mtbench-flat":
        dataset = load_dataset("philschmid/mt-bench", split="train")
        dataset = _select_n_samples(dataset, n_samples)

        def mt_prompt(row):
            turns = row.get("turns") or []
            if len(turns) == 1:
                return turns[0]
            return "\n\n".join(f"Turn {idx + 1}: {turn}" for idx, turn in enumerate(turns))

        return _normalize_prompt_dataset(dataset, mt_prompt, dataset_name), dataset_name

    if dataset_name == "vicuna":
        dataset = load_dataset(
            "json",
            data_files=f"{FASTCHAT_DATA_BASE_URL}/vicuna_bench/question.jsonl",
            split="train",
        )
        dataset = _select_n_samples(dataset, n_samples)
        return _normalize_prompt_dataset(
            dataset, lambda row: "\n\n".join(row.get("turns") or []), dataset_name
        ), dataset_name

    if dataset_name == "evol_instruct":
        dataset = load_dataset("WizardLMTeam/WizardLM_evol_instruct_70k", split="train")
        dataset = _select_n_samples(dataset, n_samples)
        return _normalize_prompt_dataset(
            dataset, lambda row: row.get("instruction"), dataset_name
        ), dataset_name

    if dataset_name == "hh_rlhf":
        dataset = load_dataset("Anthropic/hh-rlhf", split="test")
        dataset = _select_n_samples(dataset, n_samples)

        def hh_prompt(row):
            chosen = row.get("chosen", "")
            match = re.search(r"\n\nAssistant:", chosen)
            if match:
                return chosen[: match.start()].strip()
            return chosen.strip()

        return _normalize_prompt_dataset(dataset, hh_prompt, dataset_name), dataset_name

    raise ValueError(f"Unknown eval dataset '{eval_dataset}'. Supported aliases: {', '.join(DATASET_LIMITS)}")
