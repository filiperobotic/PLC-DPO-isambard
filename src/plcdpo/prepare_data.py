"""Build preference pairs and deterministic training order from upstream data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)


def convert_ultrafeedback_row(row: dict) -> dict:
    return {key: row[key] for key in ("prompt_id", "chosen", "rejected")}


DATASETS = {
    "ultrafeedback-binarized": (
        "HuggingFaceH4/ultrafeedback_binarized",
        "train_prefs",
        convert_ultrafeedback_row,
    ),
}


def prepare(name: str, output: Path, seed: int, revision: str | None, max_samples: int | None) -> dict:
    from huggingface_hub import HfApi

    from .config import sha256_file

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    hub_id, split, convert = DATASETS[name]
    resolved_revision = HfApi().dataset_info(hub_id, revision=revision).sha
    raw = load_dataset(hub_id, split=split, revision=resolved_revision)
    records = []
    for index, row in enumerate(raw):
        record = convert(row)
        if record is not None:
            record["source_prompt_id"] = str(record["prompt_id"])
            # A prompt can occur in multiple upstream preference rows.
            identity = json.dumps([record["source_prompt_id"], index])
            record["prompt_id"] = hashlib.sha256(identity.encode()).hexdigest()
            record["source_index"] = index
            records.append(record)
    if not records:
        raise ValueError("No valid preference pairs were produced.")
    frame = pd.DataFrame(records)
    converted_rows = len(frame)
    unique_rows = len(frame)
    if max_samples and len(frame) > max_samples:
        frame = frame.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    order = frame["prompt_id"].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    curriculum = pd.DataFrame({"prompt_id": order, "curriculum_rank": range(1, len(order) + 1)})
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "train.parquet", index=False)
    csv_path = output / f"order_seed{seed}.csv"
    curriculum.to_csv(csv_path, index=False)
    manifest = {
        "dataset": name,
        "hub_id": hub_id,
        "revision": resolved_revision,
        "split": split,
        "raw_rows": len(raw),
        "skipped_rows": len(raw) - converted_rows,
        "deduplicated_rows": converted_rows - unique_rows,
        "pairs": len(frame),
        "seed": seed,
        "max_samples": max_samples,
        "parquet_sha256": sha256_file(output / "train.parquet"),
        "order_sha256": sha256_file(csv_path),
        "pair_id_scheme": "sha256(json([source_prompt_id, source_index]))",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="ultrafeedback-binarized")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--revision", help="Upstream commit; defaults to resolving the current revision once."
    )
    parser.add_argument("--max-samples", type=int, help="Optional subset size; defaults to all pairs.")
    args = parser.parse_args()
    limit = args.max_samples
    if limit is not None and limit < 0:
        parser.error("--max-samples must be non-negative")
    output = args.output_dir or Path("data") / args.dataset
    print(json.dumps(prepare(args.dataset, output, args.seed, args.revision, limit), indent=2))


if __name__ == "__main__":
    main()
