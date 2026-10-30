from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch

from .base_trainer import OrderedDPOConfig, OrderedDPOTrainer
from .losses import plc_dpo_loss


@dataclass
class PLCDPOConfig(OrderedDPOConfig):
    plc_length_normalize: bool = field(default=True)
    plc_ema_alpha: float = field(default=0.99)
    plc_min_margin_std: float = field(default=0.001)
    plc_source_noise: float = field(default=0.1)
    plc_ambiguous_prior: float = field(default=0.1)
    plc_direction_temperature: float = field(default=0.75)
    plc_tie_temperature: float = field(default=1.0)
    plc_loss_temperature: float = field(default=1.0)
    plc_tie_margin: float = field(default=0.0)
    plc_warmup_ratio: float = field(default=0.1)
    plc_warmup_steps: int = field(default=0)
    plc_gamma_start: float = field(default=0.0)
    plc_gamma_end: float = field(default=0.7)
    plc_gamma_max: float = field(default=0.7)
    plc_confidence_power: float = field(default=1.0)
    # Isambard addition: estimate the margin std over a sliding window of the last N margins (hardware-independent)
    # instead of the gathered micro-batch (which is a single value with per_device_batch=1 on 1 GPU -> std floor).
    # 0 = upstream behaviour.
    plc_std_window: int = field(default=0)


class PLCDPOTrainer(OrderedDPOTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.margin_mean_ema: Optional[float] = None
        self.margin_std_ema: Optional[float] = None
        window = int(getattr(self.args, "plc_std_window", 0) or 0)
        self._margin_window = collections.deque(maxlen=window) if window > 0 else None

    def _warmup_steps(self) -> int:
        explicit_steps = int(getattr(self.args, "plc_warmup_steps", 0) or 0)
        if explicit_steps > 0:
            return explicit_steps
        max_steps = int(getattr(self.state, "max_steps", 0) or 0)
        if max_steps <= 0:
            return 0
        return int(max_steps * float(getattr(self.args, "plc_warmup_ratio", 0.1)))

    def _gamma(self) -> float:
        warmup_steps = self._warmup_steps()
        if int(self.state.global_step) < warmup_steps:
            return 0.0
        max_steps = int(getattr(self.state, "max_steps", 0) or 0)
        if max_steps <= warmup_steps:
            return float(getattr(self.args, "plc_gamma_max", 0.7))
        phase = (int(self.state.global_step) - warmup_steps) / max(max_steps - warmup_steps, 1)
        start = float(getattr(self.args, "plc_gamma_start", 0.0))
        end = float(getattr(self.args, "plc_gamma_end", 0.7))
        gamma = start + (end - start) * min(max(phase, 0.0), 1.0)
        return min(gamma, float(getattr(self.args, "plc_gamma_max", 0.7)))

    def _update_margin_ema(self, margins: torch.Tensor) -> Tuple[float, float]:
        gathered = self.accelerator.gather_for_metrics(margins.detach()).float()
        batch_mean = float(gathered.mean().item())
        if self._margin_window is not None:
            self._margin_window.extend(gathered.cpu().tolist())
            pool = torch.tensor(list(self._margin_window))
            batch_std = float(pool.std(unbiased=False).clamp_min(float(self.args.plc_min_margin_std)).item()) if len(pool) > 1 \
                else float(self.args.plc_min_margin_std)
        else:
            batch_std = float(gathered.std(unbiased=False).clamp_min(float(self.args.plc_min_margin_std)).item())
        if self.margin_mean_ema is None:
            self.margin_mean_ema = batch_mean
            self.margin_std_ema = batch_std
        else:
            alpha = float(getattr(self.args, "plc_ema_alpha", 0.99))
            self.margin_mean_ema = alpha * self.margin_mean_ema + (1.0 - alpha) * batch_mean
            self.margin_std_ema = alpha * self.margin_std_ema + (1.0 - alpha) * batch_std
        return (self.margin_mean_ema, self.margin_std_ema or float(self.args.plc_min_margin_std))

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}
        model_output = self.concatenated_forward(model, batch)
        ref_chosen_logps, ref_rejected_logps = self._resolve_ref_logps(batch)
        chosen_lengths = self._response_lengths(batch, model_output, "chosen")
        rejected_lengths = self._response_lengths(batch, model_output, "rejected")
        length_normalize = bool(getattr(self.args, "plc_length_normalize", True))
        beta = float(self.args.beta)
        with torch.no_grad():
            chosen_rewards = beta * (model_output["chosen_logps"].float() - ref_chosen_logps.float())
            rejected_rewards = beta * (model_output["rejected_logps"].float() - ref_rejected_logps.float())
            if length_normalize:
                chosen_rewards = chosen_rewards / chosen_lengths.to(
                    model_output["chosen_logps"].device
                ).float().clamp_min(1.0)
                rejected_rewards = rejected_rewards / rejected_lengths.to(
                    model_output["rejected_logps"].device
                ).float().clamp_min(1.0)
            margin_mean_ema, margin_std_ema = self._update_margin_ema(chosen_rewards - rejected_rewards)
        warmup_steps = self._warmup_steps()
        use_plc = train_eval != "train" or int(self.state.global_step) >= warmup_steps
        gamma = self._gamma() if train_eval == "train" else float(getattr(self.args, "plc_gamma_max", 0.7))
        margin_mean_tensor = torch.tensor(
            margin_mean_ema, dtype=torch.float32, device=model_output["chosen_logps"].device
        )
        margin_std_tensor = torch.tensor(
            margin_std_ema, dtype=torch.float32, device=model_output["chosen_logps"].device
        )
        loss_outputs = plc_dpo_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
            beta=beta,
            use_plc=use_plc,
            gamma=gamma,
            margin_mean_ema=margin_mean_tensor,
            margin_std_ema=margin_std_tensor,
            length_normalize=length_normalize,
            source_noise=float(self.args.plc_source_noise),
            ambiguous_prior=float(self.args.plc_ambiguous_prior),
            direction_temperature=float(self.args.plc_direction_temperature),
            tie_temperature=float(self.args.plc_tie_temperature),
            loss_temperature=float(self.args.plc_loss_temperature),
            tie_margin=float(self.args.plc_tie_margin),
            confidence_power=float(self.args.plc_confidence_power),
        )
        losses = loss_outputs["loss"]
        if self.args.rpo_alpha is not None:
            losses = losses * self.args.rpo_alpha + model_output["nll_loss"]
        prefix = "eval_" if train_eval == "eval" else ""

        def mean_metric(tensor: torch.Tensor) -> float:
            return self.accelerator.gather_for_metrics(tensor.detach().float()).mean().item()

        chosen_rewards = loss_outputs["chosen_rewards"].detach()
        rejected_rewards = loss_outputs["rejected_rewards"].detach()
        metrics[f"{prefix}rewards/chosen"] = mean_metric(chosen_rewards)
        metrics[f"{prefix}rewards/rejected"] = mean_metric(rejected_rewards)
        metrics[f"{prefix}rewards/margins"] = mean_metric(chosen_rewards - rejected_rewards)
        metrics[f"{prefix}rewards/accuracies"] = mean_metric((chosen_rewards > rejected_rewards).float())
        metrics[f"{prefix}plc/margin_mean_ema"] = float(margin_mean_ema)
        metrics[f"{prefix}plc/margin_std_ema"] = float(margin_std_ema)
        metrics[f"{prefix}plc/warmup_steps"] = float(warmup_steps)
        metrics[f"{prefix}plc/gamma"] = float(gamma)
        metrics[f"{prefix}plc/active"] = 1.0 if use_plc else 0.0
        metrics[f"{prefix}plc/p_clean"] = mean_metric(loss_outputs["p_clean"])
        metrics[f"{prefix}plc/p_flip"] = mean_metric(loss_outputs["p_flip"])
        metrics[f"{prefix}plc/p_tie"] = mean_metric(loss_outputs["p_tie"])
        metrics[f"{prefix}plc/confidence"] = mean_metric(loss_outputs["confidence"])
        metrics[f"{prefix}plc/mixture_weight"] = mean_metric(loss_outputs["mixture_weight"])
        metrics[f"{prefix}plc/margin_z"] = mean_metric(loss_outputs["margin_z"])
        metrics[f"{prefix}loss/standard"] = mean_metric(loss_outputs["standard_loss"])
        metrics[f"{prefix}loss/reverse"] = mean_metric(loss_outputs["reverse_loss"])
        metrics[f"{prefix}loss/tie"] = mean_metric(loss_outputs["tie_loss"])
        metrics[f"{prefix}loss/posterior"] = mean_metric(loss_outputs["posterior_loss"])
        metrics[f"{prefix}lengths/chosen"] = mean_metric(chosen_lengths)
        metrics[f"{prefix}lengths/rejected"] = mean_metric(rejected_lengths)
        metrics[f"{prefix}logps/chosen"] = mean_metric(model_output["chosen_logps"])
        metrics[f"{prefix}logps/rejected"] = mean_metric(model_output["rejected_logps"])
        metrics[f"{prefix}logits/chosen"] = mean_metric(model_output["mean_chosen_logits"])
        metrics[f"{prefix}logits/rejected"] = mean_metric(model_output["mean_rejected_logits"])
        if self.sort_column in batch:
            sort_tensor = torch.as_tensor(
                batch[self.sort_column], dtype=torch.float32, device=model_output["chosen_logps"].device
            )
            metrics[f"{prefix}{self.sort_column}"] = mean_metric(sort_tensor)
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = mean_metric(model_output["nll_loss"])
        if self.aux_loss_enabled:
            return (
                losses.mean() + getattr(model.config, "router_aux_loss_coef", 0.0) * model_output["aux_loss"],
                metrics,
            )
        return (losses.mean(), metrics)
