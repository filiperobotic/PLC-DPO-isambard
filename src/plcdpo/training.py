# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Training workflow adapted from SelectiveDPO; modified for Unsloth and PLC-DPO in 2026.
from __future__ import annotations

import functools
import logging
import os
from dataclasses import fields
from pathlib import Path
from typing import Callable, Type

import torch
import transformers
from transformers import set_seed

from .data import format_training_dataset, load_mixed_datasets
from .runtime import (
    apply_runtime_training_defaults,
    configure_wandb,
    ensure_distributed_initialized,
    get_effective_batch_size,
    get_local_rank_device_map,
    get_runtime_num_gpus,
)

logger = logging.getLogger(__name__)


def patch_transformers_warning_once(tag: str) -> None:
    current_warning_once = getattr(logging.Logger, "warning_once", None)
    if current_warning_once is None:
        return
    patch_attr = f"_{tag}_safe_patch"
    if getattr(current_warning_once, patch_attr, False):
        return

    @functools.lru_cache(None)
    def safe_warning_once(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[1], type) and issubclass(args[1], Warning):
            args = (args[0],) + args[2:]
        self.warning(*args, **kwargs)

    setattr(safe_warning_once, patch_attr, True)
    logging.Logger.warning_once = safe_warning_once


def build_common_training_kwargs(cfg: dict, config_cls: Type) -> dict:
    apply_runtime_training_defaults(cfg)
    supported_fields = {field.name for field in fields(config_cls)}
    kwargs = {
        "output_dir": cfg["output_dir"],
        "per_device_train_batch_size": cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "learning_rate": float(cfg["learning_rate"]),
        "lr_scheduler_type": cfg["lr_scheduler_type"],
        "warmup_ratio": float(cfg.get("warmup_ratio", 0.0)),
        "num_train_epochs": int(cfg.get("num_train_epochs", 1)),
        "max_steps": int(cfg.get("max_steps", -1)),
        "logging_steps": int(cfg["logging_steps"]),
        "logging_first_step": cfg.get("logging_first_step", True),
        "save_strategy": cfg.get("save_strategy", "steps"),
        "save_steps": cfg.get("save_steps"),
        "optim": cfg["optim"],
        "seed": int(cfg["seed"]),
        "bf16": cfg.get("bf16", True),
        "fp16": cfg.get("fp16", False),
        "beta": float(cfg["beta"]),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
        "max_length": int(cfg["max_length"]),
        "max_prompt_length": int(cfg["max_prompt_length"]),
        "remove_unused_columns": cfg.get("remove_unused_columns", False),
        "dataset_num_proc": int(cfg.get("preprocessing_num_workers", 4)),
        "dataloader_num_workers": int(cfg.get("dataloader_num_workers", 0)),
        "dataloader_pin_memory": cfg.get("dataloader_pin_memory", True),
        "dataloader_drop_last": cfg.get("dataloader_drop_last", False),
        "gradient_checkpointing": False,
        "report_to": cfg.get("report_to", ["wandb"]),
        "run_name": cfg.get("run_name"),
        "ddp_find_unused_parameters": cfg.get("ddp_find_unused_parameters", False),
        "ddp_broadcast_buffers": cfg.get("ddp_broadcast_buffers", False),
        "save_total_limit": cfg.get("save_total_limit"),
        "csv_path": cfg["csv_path"],
        "sort_column": cfg.get("sort_column", "curriculum_rank"),
        "sort_descending": cfg.get("sort_descending", False),
        "loss_type": cfg.get("loss_type"),
        "loss_weights": cfg.get("loss_weights"),
        "label_smoothing": cfg.get("label_smoothing"),
        "rpo_alpha": cfg.get("rpo_alpha"),
        "response_length_mode": cfg.get("response_length_mode", "legacy"),
    }
    return {key: value for key, value in kwargs.items() if value is not None and key in supported_fields}


def log_module_device_summary(model: torch.nn.Module, *, max_examples: int = 8) -> None:
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return

    cpu_params = [
        (name, tuple(param.shape)) for name, param in model.named_parameters() if param.device.type == "cpu"
    ]
    cpu_buffers = [
        (name, tuple(buffer.shape)) for name, buffer in model.named_buffers() if buffer.device.type == "cpu"
    ]
    if cpu_params:
        logger.warning(
            "Model has CPU parameters before DDP wrapping; NCCL DDP cannot synchronize these. "
            "count=%s examples=%s",
            len(cpu_params),
            cpu_params[:max_examples],
        )
    if cpu_buffers:
        logger.warning(
            "Model has CPU buffers before DDP wrapping; using ddp_broadcast_buffers=False avoids NCCL "
            "broadcasting them during DDP initialization. count=%s examples=%s",
            len(cpu_buffers),
            cpu_buffers[:max_examples],
        )


def run_training(
    cfg: dict,
    *,
    method_name: str,
    warning_tag: str,
    build_training_args: Callable[[dict], object],
    trainer_cls: Type,
) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    logging.basicConfig(
        level=logging.INFO if local_rank == 0 else logging.WARN,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    patch_transformers_warning_once(warning_tag)
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    ensure_distributed_initialized()
    apply_runtime_training_defaults(cfg)
    logger.info(
        "Runtime batch sizing | target_effective_batch_size=%s | effective_batch_size=%s | "
        "per_device_train_batch_size=%s | gradient_accumulation_steps=%s | num_gpus=%s",
        cfg["target_effective_batch_size"],
        get_effective_batch_size(cfg),
        cfg["per_device_train_batch_size"],
        cfg["gradient_accumulation_steps"],
        get_runtime_num_gpus(),
    )

    from .config import save_run_manifest

    if local_rank == 0:
        save_run_manifest(cfg, Path(cfg["output_dir"]))
    configure_wandb(cfg)
    set_seed(int(cfg["seed"]))

    logger.info("Loading model with Unsloth optimizations.")
    from unsloth import FastLanguageModel

    device_map = get_local_rank_device_map()
    logger.warning(
        "Model load placement | local_rank=%s | cuda_current_device=%s | device_map=%s | CUDA_VISIBLE_DEVICES=%s",
        local_rank,
        torch.cuda.current_device() if torch.cuda.is_available() else None,
        device_map,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name_or_path"],
        max_seq_length=cfg["max_length"],
        dtype=getattr(torch, cfg.get("torch_dtype", "bfloat16")),
        load_in_4bit=cfg.get("load_in_4bit", False),
        attn_implementation=cfg.get("attn_implementation", "flash_attention_2"),
        device_map=device_map,
        revision=cfg.get("model_revision"),
    )
    model.config.use_cache = False

    if cfg.get("chat_template"):
        tokenizer.chat_template = cfg["chat_template"]
    tokenizer.truncation_side = cfg.get("truncation_side", "left")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Applying LoRA through Unsloth.")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.0),
        target_modules=cfg["lora_target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth" if cfg.get("gradient_checkpointing", True) else False,
        random_state=int(cfg["seed"]),
    )
    log_module_device_summary(model)

    logger.info("Loading and formatting dataset.")
    raw_datasets = load_mixed_datasets(cfg)
    len_train_before = len(raw_datasets["train"])
    train_dataset = format_training_dataset(raw_datasets["train"], tokenizer, cfg)

    training_args = build_training_args(cfg)

    logger.info("Initializing %s trainer.", method_name)
    trainer = trainer_cls(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training.")
    train_result = trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    metrics = dict(train_result.metrics)
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.accelerator.wait_for_everyone()
    logger.info("Saving adapter to %s", cfg["output_dir"])
    trainer.save_model(cfg["output_dir"])
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(cfg["output_dir"])
    logger.info("Train size: %s -> %s", len_train_before, len(train_dataset))
