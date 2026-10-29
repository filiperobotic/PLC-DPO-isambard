"""Train PLC-DPO or a paper baseline from shared YAML configuration."""

from __future__ import annotations

import argparse
import os
import json
from dataclasses import fields
from pathlib import Path

import yaml

from .config import check_local_data, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="YAML files merged left to right; later keys replace earlier keys.",
    )
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print settings without loading models or data."
    )
    parser.add_argument(
        "--check-data", action="store_true", help="Validate local preferences and ordering, without training."
    )
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    if args.check_data:
        print(json.dumps(check_local_data(cfg), indent=2))
        return
    if args.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return
    output = Path(cfg["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Choose a new output_dir.")
    check_local_data(cfg)

    from .runtime import pin_cuda_device_to_local_rank

    pin_cuda_device_to_local_rank()
    # Unsloth must patch the training stack before importing Transformers/TRL.
    # PLCDPO_NO_UNSLOTH=1 (Isambard-AI, aarch64): plain transformers + peft, no Unsloth patching.
    if os.environ.get("PLCDPO_NO_UNSLOTH") != "1":
        from unsloth import PatchDPOTrainer

        PatchDPOTrainer()

    from .training import build_common_training_kwargs, run_training

    if cfg["method"] == "plc-dpo":
        from .plc_trainer import PLCDPOConfig as Config
        from .plc_trainer import PLCDPOTrainer as Trainer
    else:
        from .base_trainer import OrderedDPOConfig as Config
        from .dpo_trainer import DPOTrainer as Trainer

    def build_training_args(values):
        kwargs = build_common_training_kwargs(values, Config)
        supported = {f.name for f in fields(Config)}
        kwargs.update({k: v for k, v in values.items() if k in supported and k.startswith("plc_")})
        return Config(**kwargs)

    run_training(
        cfg,
        method_name=cfg["method"],
        warning_tag="plcdpo",
        build_training_args=build_training_args,
        trainer_cls=Trainer,
    )


if __name__ == "__main__":
    main()
