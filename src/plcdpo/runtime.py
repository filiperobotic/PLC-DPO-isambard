from __future__ import annotations

import logging
import os
from typing import Dict

import torch

logger = logging.getLogger(__name__)


def configure_wandb(cfg: Dict):
    report_to = cfg.get("report_to")
    if isinstance(report_to, str):
        report_targets = [report_to]
    else:
        report_targets = list(report_to or [])

    if "wandb" not in report_targets:
        return

    env_map = {
        "wandb_project": "WANDB_PROJECT",
        "wandb_entity": "WANDB_ENTITY",
        "wandb_name": "WANDB_NAME",
        "wandb_notes": "WANDB_NOTES",
        "wandb_tags": "WANDB_TAGS",
        "wandb_watch": "WANDB_WATCH",
        "wandb_log_model": "WANDB_LOG_MODEL",
    }
    for cfg_key, env_key in env_map.items():
        value = cfg.get(cfg_key)
        if value is None:
            continue
        if isinstance(value, list):
            value = ",".join(str(item) for item in value)
        os.environ[env_key] = str(value)

    if cfg.get("run_name") and "WANDB_NAME" not in os.environ:
        os.environ["WANDB_NAME"] = str(cfg["run_name"])


def ensure_distributed_initialized() -> None:
    if not torch.distributed.is_available() or torch.distributed.is_initialized():
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    torch.distributed.init_process_group(backend=backend, init_method="env://")


def get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_runtime_num_gpus() -> int:
    return get_world_size()


def get_auto_gradient_accumulation_steps(
    target_effective_batch_size: int = 64,
    per_device_train_batch_size: int = 1,
) -> int:
    target_effective_batch_size = int(target_effective_batch_size)
    per_device_train_batch_size = int(per_device_train_batch_size)
    if target_effective_batch_size <= 0:
        raise ValueError("`target_effective_batch_size` must be a positive integer.")
    if per_device_train_batch_size <= 0:
        raise ValueError("`per_device_train_batch_size` must be a positive integer.")

    num_gpus = get_runtime_num_gpus()
    divisor = per_device_train_batch_size * num_gpus
    if target_effective_batch_size % divisor != 0:
        raise ValueError(
            "Cannot keep effective batch size exactly "
            f"{target_effective_batch_size} with per_device_train_batch_size="
            f"{per_device_train_batch_size} "
            f"and {num_gpus} GPU(s). Choose a GPU count that divides {target_effective_batch_size}."
        )
    return target_effective_batch_size // divisor


def apply_runtime_training_defaults(cfg: Dict) -> Dict:
    target_effective_batch_size = int(cfg.get("target_effective_batch_size", 64))
    cfg.setdefault("per_device_train_batch_size", 1)
    cfg["target_effective_batch_size"] = target_effective_batch_size
    if cfg.get("gradient_accumulation_steps") is None:
        cfg["gradient_accumulation_steps"] = get_auto_gradient_accumulation_steps(
            target_effective_batch_size, cfg["per_device_train_batch_size"]
        )
    if get_effective_batch_size(cfg) != target_effective_batch_size:
        raise ValueError("Batch size × accumulation × WORLD_SIZE must equal target_effective_batch_size.")
    cfg.setdefault("save_strategy", "steps")
    cfg.setdefault("save_steps", 50)
    cfg.pop("do_eval", None)
    cfg.pop("eval_strategy", None)
    cfg.pop("evaluation_strategy", None)
    cfg.pop("eval_steps", None)
    cfg.pop("per_device_eval_batch_size", None)
    return cfg


def get_effective_batch_size(cfg: Dict) -> int:
    return (
        int(cfg["per_device_train_batch_size"])
        * int(cfg["gradient_accumulation_steps"])
        * get_runtime_num_gpus()
    )


def pin_cuda_device_to_local_rank() -> None:
    """Select this process' GPU before imports that may touch CUDA."""
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return

    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(int(local_rank))


def get_local_rank_device_map():
    """Return a device_map that keeps this DDP process on its own GPU."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return None

    import torch

    if not torch.cuda.is_available():
        return None

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    return {"": torch.cuda.current_device()}
