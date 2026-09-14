"""Resolve experiment configuration without importing a GPU training stack."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from pathlib import Path

import yaml

METHODS = ("plc-dpo", "dpo")


def load_config(paths, overrides=()) -> dict:
    cfg = {}
    for path in paths:
        values = yaml.safe_load(Path(path).read_text())
        if not isinstance(values, dict):
            raise ValueError(f"Expected a YAML mapping: {path}")
        cfg.update(values)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        cfg[key] = yaml.safe_load(value)
    validate_config(cfg)
    if not cfg.get("output_dir"):
        model = Path(cfg["model_name_or_path"]).name
        dataset = Path(cfg["csv_path"]).parent.name
        recipe = "-" + cfg.get("plc_recipe", "aggressive") if cfg["method"] == "plc-dpo" else ""
        cfg["output_dir"] = f"outputs/{model}/{dataset}/{cfg['method']}{recipe}-seed{cfg['seed']}"
    cfg.setdefault("run_name", str(Path(cfg["output_dir"]).name))
    return cfg


def validate_config(cfg: dict) -> None:
    required = (
        "method",
        "model_name_or_path",
        "csv_path",
        "dataset_mixer",
        "seed",
        "beta",
        "max_length",
        "max_prompt_length",
    )
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(missing)}")
    if cfg["method"] not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if cfg.get("response_length_mode", "legacy") not in {"legacy", "tokens"}:
        raise ValueError("response_length_mode must be legacy or tokens.")
    if any(k.startswith(("ablation_", "flip_")) or k == "plc_ablation_mode" for k in cfg):
        raise ValueError("Only clean-data DPO and full PLC-DPO configurations are supported.")
    if not isinstance(cfg["dataset_mixer"], dict) or not cfg["dataset_mixer"]:
        raise ValueError("dataset_mixer must be a non-empty mapping.")
    for key in (
        "beta",
        "learning_rate",
        "max_length",
        "max_prompt_length",
        "target_effective_batch_size",
        "per_device_train_batch_size",
        "num_train_epochs",
    ):
        if key in cfg and (not math.isfinite(float(cfg[key])) or float(cfg[key]) <= 0):
            raise ValueError(f"{key} must be finite and positive.")
    if cfg["max_prompt_length"] >= cfg["max_length"]:
        raise ValueError("max_prompt_length must be smaller than max_length.")
    for key in (
        "plc_source_noise",
        "plc_ambiguous_prior",
        "plc_warmup_ratio",
        "plc_gamma_start",
        "plc_gamma_end",
        "plc_gamma_max",
        "plc_ema_alpha",
    ):
        if key in cfg and not 0 <= float(cfg[key]) <= 1:
            raise ValueError(f"{key} must be in [0, 1].")
    for key in (
        "plc_direction_temperature",
        "plc_tie_temperature",
        "plc_loss_temperature",
        "plc_min_margin_std",
        "plc_confidence_power",
    ):
        if key in cfg and (not math.isfinite(float(cfg[key])) or float(cfg[key]) <= 0):
            raise ValueError(f"{key} must be finite and positive.")
    if cfg.get("resume_from_checkpoint"):
        raise ValueError(
            "Resume is disabled until objective-state checkpoint equivalence is verified; use a fresh output directory."
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_run_manifest(cfg: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, None
    local_inputs = [Path(cfg["csv_path"]), *(Path(name) for name in cfg["dataset_mixer"])]
    manifest = {
        "config_sha256": sha256_file(output / "config.yaml"),
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "input_sha256": {str(p): sha256_file(p) for p in local_inputs if p.is_file()},
        "packages": {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata.get("Name")
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def check_local_data(cfg: dict) -> dict:
    import pandas as pd

    from .data import load_mixed_datasets

    data = load_mixed_datasets(cfg)["train"]
    order = pd.read_csv(cfg["csv_path"], dtype={"prompt_id": str})
    required = {"prompt_id", "chosen", "rejected"}
    if not required.issubset(data.column_names):
        raise ValueError(f"Preference data requires {sorted(required)}")
    ids = pd.Series(data["prompt_id"])
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Preference IDs must be unique and non-null.")
    for key in ("prompt_id", cfg.get("sort_column", "curriculum_rank")):
        if key not in order or order[key].isna().any() or order[key].duplicated().any():
            raise ValueError(f"Invalid or duplicate order column: {key}")
    if set(ids) != set(order["prompt_id"]):
        raise ValueError("Preference data and order CSV must contain the same IDs before pair filtering.")
    return {"pairs": len(data), "order_rows": len(order), "csv_sha256": sha256_file(Path(cfg["csv_path"]))}
