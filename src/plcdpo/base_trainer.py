# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from SelectiveDPO and TRL; modified for PLC-DPO in 2026.
from __future__ import annotations

import functools
import inspect
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import datasets
import torch
from accelerate.utils import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available
from trl import DPOConfig, DPOTrainer


@dataclass
class OrderedDPOConfig(DPOConfig):
    csv_path: str = ""
    sort_column: str = "curriculum_rank"
    sort_descending: bool = False
    response_length_mode: str = "legacy"


class OrderedDPOTrainer(DPOTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sort_column = getattr(self.args, "sort_column", "curriculum_rank")
        self.sort_descending = getattr(self.args, "sort_descending", False)
        self._warned_missing_labels = False
        if self.train_dataset is not None and self.sort_column in self.train_dataset.column_names:
            self.train_dataset = self.train_dataset.sort(self.sort_column, reverse=self.sort_descending)

    def _ordered_sampler(self, train_dataset):
        if train_dataset is None:
            return None
        return SequentialSampler(train_dataset)

    def _resolve_ref_logps(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            return batch["reference_chosen_logps"], batch["reference_rejected_logps"]
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            return batch["ref_chosen_logps"], batch["ref_rejected_logps"]
        return self.compute_ref_log_probs(batch)

    def _response_lengths(self, batch, model_output, key: str) -> torch.Tensor:
        label_key = f"{key}_labels"
        if label_key in batch:
            labels = batch[label_key]
            return (labels != -100).sum(dim=-1).clamp_min(1)
        if self.args.response_length_mode == "legacy":
            if not self._warned_missing_labels:
                warnings.warn(
                    "The collator supplies no response labels. Preserving the historical length-1 fallback; "
                    "set response_length_mode=tokens to count response attention masks instead. "
                    "That changes normalized objectives and requires new validation.",
                    stacklevel=2,
                )
                self._warned_missing_labels = True
            return torch.ones_like(model_output[f"{key}_logps"], dtype=torch.long)
        if self.args.response_length_mode == "tokens" and f"{key}_attention_mask" in batch:
            return batch[f"{key}_attention_mask"].sum(dim=-1).clamp_min(1)
        raise ValueError(
            f"Missing response lengths for {key}: expected labels or response-only attention masks."
        )

    def get_train_dataloader(self) -> DataLoader:
        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            data_loader = self.accelerator.prepare(
                DataLoader(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    collate_fn=self.data_collator,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                    sampler=self._ordered_sampler(self.train_dataset),
                )
            )

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                ref_chosen, ref_rejected = self.compute_ref_log_probs(padded_batch)
                ref_chosen, ref_rejected = self.accelerator.gather_for_metrics((ref_chosen, ref_rejected))
                reference_chosen_logps.append(ref_chosen.cpu())
                reference_rejected_logps.append(ref_rejected.cpu())

            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_logps",
                column=torch.cat(reference_chosen_logps).float().numpy(),
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_logps",
                column=torch.cat(reference_rejected_logps).float().numpy(),
            )
            self._precomputed_train_ref_log_probs = True

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._ordered_sampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            seed_worker_params = inspect.signature(seed_worker).parameters
            if len(seed_worker_params) > 1:
                dataloader_params["worker_init_fn"] = functools.partial(
                    seed_worker,
                    num_workers=self.args.dataloader_num_workers,
                    rank=self.args.process_index,
                )
            else:
                dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
