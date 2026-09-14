"""PLC-DPO and the baseline objectives used in the experiment code."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _safe_temperature(value: float) -> float:
    return max(float(value), 1e-06)


def _normalized_priors(
    source_noise: float, ambiguous_prior: float, device: torch.device
) -> Tuple[torch.Tensor, ...]:
    flip = min(max(float(source_noise), 0.0001), 1.0 - 0.0001)
    tie = min(max(float(ambiguous_prior), 0.0001), 1.0 - 0.0001)
    clean = max(1.0 - flip - tie, 0.0001)
    total = clean + flip + tie
    return (
        torch.tensor(clean / total, dtype=torch.float32, device=device),
        torch.tensor(flip / total, dtype=torch.float32, device=device),
        torch.tensor(tie / total, dtype=torch.float32, device=device),
    )


def plc_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float,
    use_plc: bool,
    gamma: float,
    margin_mean_ema: torch.Tensor,
    margin_std_ema: torch.Tensor,
    length_normalize: bool,
    source_noise: float,
    ambiguous_prior: float,
    direction_temperature: float,
    tie_temperature: float,
    loss_temperature: float,
    tie_margin: float,
    confidence_power: float,
) -> Dict[str, torch.Tensor]:
    device = policy_chosen_logps.device
    loss_tau = _safe_temperature(loss_temperature)
    direction_tau = _safe_temperature(direction_temperature)
    tie_tau = _safe_temperature(tie_temperature)
    chosen_lengths = chosen_lengths.to(device=device, dtype=torch.float32).clamp_min(1.0)
    rejected_lengths = rejected_lengths.to(device=device, dtype=torch.float32).clamp_min(1.0)
    chosen_rewards_sum = beta * (policy_chosen_logps.float() - reference_chosen_logps.float())
    rejected_rewards_sum = beta * (policy_rejected_logps.float() - reference_rejected_logps.float())
    if length_normalize:
        chosen_rewards = chosen_rewards_sum / chosen_lengths
        rejected_rewards = rejected_rewards_sum / rejected_lengths
    else:
        chosen_rewards = chosen_rewards_sum
        rejected_rewards = rejected_rewards_sum
    margin = chosen_rewards - rejected_rewards
    logits = margin / loss_tau
    standard_loss = F.softplus(-logits)
    reverse_loss = F.softplus(logits)
    tie_loss = F.softplus((margin.abs() - float(tie_margin)) / loss_tau)
    margin_std = margin_std_ema.to(device=device, dtype=torch.float32).clamp_min(1e-06)
    # Posterior routing is detached from the policy gradient.
    margin_z = (margin.detach() - margin_mean_ema.to(device=device, dtype=torch.float32)) / margin_std
    clean_prior, flip_prior, tie_prior = _normalized_priors(source_noise, ambiguous_prior, device)
    clean_logit = clean_prior.log() + margin_z / direction_tau
    flip_logit = flip_prior.log() - margin_z / direction_tau
    tie_logit = tie_prior.log() - margin_z.abs() / tie_tau
    posterior_logits = torch.stack([clean_logit, flip_logit, tie_logit], dim=-1)
    posterior = torch.softmax(posterior_logits, dim=-1)
    p_clean = posterior[..., 0]
    p_flip = posterior[..., 1]
    p_tie = posterior[..., 2]
    confidence = ((posterior.max(dim=-1).values - 1.0 / 3.0) / (2.0 / 3.0)).clamp(0.0, 1.0)
    confidence = confidence.pow(float(confidence_power))
    confidence_for_mixture = confidence
    mixture_weight = (float(gamma) * confidence_for_mixture).clamp(0.0, 1.0)
    posterior_loss = p_clean * standard_loss + p_flip * reverse_loss + p_tie * tie_loss
    if use_plc:
        loss = (1.0 - mixture_weight) * standard_loss + mixture_weight * posterior_loss
    else:
        loss = standard_loss
        mixture_weight = torch.zeros_like(standard_loss)
        posterior_loss = standard_loss
    return {
        "loss": loss,
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
        "chosen_rewards_sum": chosen_rewards_sum,
        "rejected_rewards_sum": rejected_rewards_sum,
        "margin": margin,
        "margin_z": margin_z,
        "standard_loss": standard_loss,
        "reverse_loss": reverse_loss,
        "tie_loss": tie_loss,
        "posterior_loss": posterior_loss,
        "p_clean": p_clean,
        "p_flip": p_flip,
        "p_tie": p_tie,
        "confidence": confidence,
        "mixture_weight": mixture_weight,
    }


def dpo_loss(
    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta
):
    chosen_rewards = beta * (policy_chosen_logps.float() - reference_chosen_logps.float())
    rejected_rewards = beta * (policy_rejected_logps.float() - reference_rejected_logps.float())
    return F.softplus(-(chosen_rewards - rejected_rewards)), chosen_rewards, rejected_rewards
