# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Chat formatting adapted from Alignment Handbook via SelectiveDPO.
# Modified for PLC-DPO dataset loading, ordering, and corruption in 2026.
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)


def is_openai_format(messages) -> bool:
    if not isinstance(messages, list) or not messages:
        return False
    return all(
        isinstance(message, dict) and "role" in message and "content" in message for message in messages
    )


def maybe_insert_system_message(messages, tokenizer):
    if messages and messages[0]["role"] == "system":
        return
    if hasattr(tokenizer, "default_system_message"):
        messages.insert(0, {"role": "system", "content": tokenizer.default_system_message})
    else:
        messages.insert(0, {"role": "system", "content": ""})


def apply_chat_template(example, tokenizer, auto_insert_empty_system_msg: bool = True):
    if not all(key in example for key in ("chosen", "rejected")):
        raise ValueError(f"Expected chosen/rejected keys, found {list(example.keys())}")
    if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
        raise ValueError("Expected OpenAI-style chat messages in chosen/rejected.")

    if "prompt" in example and is_openai_format(example["prompt"]):
        prompt_messages = list(example["prompt"])
        chosen_messages = example["chosen"]
        rejected_messages = example["rejected"]
    else:
        if example["chosen"][:-1] != example["rejected"][:-1]:
            raise ValueError("Chosen and rejected conversations must share the same context.")
        prompt_messages = example["chosen"][:-1]
        chosen_messages = example["chosen"][-1:]
        rejected_messages = example["rejected"][-1:]

    if auto_insert_empty_system_msg:
        maybe_insert_system_message(prompt_messages, tokenizer)

    example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
    example["text_generation_prompt"] = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
    example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

    if tokenizer.bos_token and example["text_chosen"].startswith(tokenizer.bos_token):
        example["text_chosen"] = example["text_chosen"][len(tokenizer.bos_token) :]
    if tokenizer.bos_token and example["text_rejected"].startswith(tokenizer.bos_token):
        example["text_rejected"] = example["text_rejected"][len(tokenizer.bos_token) :]

    return example


def load_mixed_datasets(cfg: Dict) -> Dict[str, Dataset]:
    dataset_mixer = cfg.get("dataset_mixer") or {}
    dataset_splits = cfg.get("dataset_splits") or ["train_prefs"]

    if not dataset_mixer:
        raise ValueError("`dataset_mixer` must contain at least one dataset.")

    columns_to_keep = [
        "prompt_id",
        "messages",
        "chosen",
        "rejected",
        "prompt",
        "completion",
        "label",
        "score_chosen",
        "score_rejected",
    ]

    local_file_builders = {".parquet": "parquet", ".json": "json", ".jsonl": "json", ".csv": "csv"}

    train_sets: List[Dataset] = []

    for dataset_name in dataset_mixer:
        if dataset_mixer[dataset_name] != 1.0:
            raise ValueError("Only full datasets (weight 1.0) are supported; prepare subsets explicitly.")
        train_split = dataset_splits[0]
        suffix = Path(dataset_name).suffix.lower()
        if suffix in local_file_builders:
            if not os.path.isfile(dataset_name):
                raise FileNotFoundError(f"Prepare the dataset first: {dataset_name}")
            train_dataset = load_dataset(local_file_builders[suffix], data_files=dataset_name, split="train")
        else:
            train_dataset = load_dataset(
                dataset_name,
                split=train_split,
                revision=cfg.get("dataset_revisions", {}).get(dataset_name),
            )
        keep_cols = [col for col in columns_to_keep if col in train_dataset.column_names]
        train_sets.append(train_dataset.select_columns(keep_cols))

    return {"train": train_sets[0] if len(train_sets) == 1 else concatenate_datasets(train_sets)}


def prepare_train_dataframe(raw_train_dataset: Dataset, csv_df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    sort_column = cfg["sort_column"]
    merge_cols = [col for col in ("prompt_id", sort_column) if col in csv_df.columns]

    if sort_column not in merge_cols:
        raise ValueError(f"Sort column `{sort_column}` was not found in `{cfg['csv_path']}`.")

    train_df = pd.DataFrame.from_dict(raw_train_dataset)
    if "prompt_id" not in train_df or "prompt_id" not in csv_df:
        raise ValueError("Both preferences and order CSV must contain prompt_id.")
    for name, frame in [("preferences", train_df), ("order CSV", csv_df)]:
        if frame["prompt_id"].isna().any() or frame["prompt_id"].duplicated().any():
            raise ValueError(f"{name} contains missing or duplicate prompt_id values.")
    if csv_df[sort_column].isna().any() or csv_df[sort_column].duplicated().any():
        raise ValueError("Training order must contain unique, non-null ranks.")
    missing = set(train_df["prompt_id"]) - set(csv_df["prompt_id"])
    if missing:
        raise ValueError(f"Order CSV is missing {len(missing)} training IDs.")
    train_df = pd.merge(train_df, csv_df[merge_cols], on="prompt_id", how="inner", validate="one_to_one")
    train_df = train_df.sort_values(
        by=sort_column,
        ascending=not cfg.get("sort_descending", False),
        kind="mergesort",
    ).reset_index(drop=True)
    return train_df


def format_training_dataset(raw: Dataset, tokenizer, cfg: Dict) -> Dataset:
    formatted = raw.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "auto_insert_empty_system_msg": cfg.get("auto_insert_empty_system_msg", True),
        },
        num_proc=cfg.get("preprocessing_num_workers"),
        remove_columns=[col for col in raw.column_names if col != "prompt_id"],
        desc="Formatting preference pairs",
    ).rename_columns(
        {
            "text_prompt": "prompt",
            "text_generation_prompt": "generation_prompt",
            "text_chosen": "chosen",
            "text_rejected": "rejected",
        }
    )
    formatted = formatted.filter(
        lambda row: row["chosen"] != row["rejected"], num_proc=cfg.get("preprocessing_num_workers")
    )
    frame = prepare_train_dataframe(
        formatted, pd.read_csv(cfg["csv_path"], low_memory=False, dtype={"prompt_id": str}), cfg
    )
    if frame.empty:
        raise ValueError("No preference pairs remain after formatting and filtering.")
    return Dataset.from_pandas(frame, preserve_index=False)
