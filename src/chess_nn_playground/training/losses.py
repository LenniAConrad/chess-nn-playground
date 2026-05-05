from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def cross_entropy_loss(class_weights: torch.Tensor | None = None) -> nn.Module:
    return nn.CrossEntropyLoss(weight=class_weights)


def binary_cross_entropy_loss(pos_weight: torch.Tensor | None = None) -> nn.Module:
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def _fine_label_tensor(
    fine_label: torch.Tensor | None,
    target: torch.Tensor,
) -> torch.Tensor:
    if fine_label is None:
        return torch.where(target.view(-1).long() > 0, torch.full_like(target.view(-1).long(), 2), 0)
    return fine_label.to(device=target.device).long().view(-1)


def _huber_positive_residual(residual: torch.Tensor, kappa: float) -> torch.Tensor:
    kappa = max(float(kappa), 1e-6)
    abs_residual = residual.abs()
    quadratic = 0.5 * residual.pow(2)
    linear = kappa * (abs_residual - 0.5 * kappa)
    return torch.where(abs_residual <= kappa, quadratic, linear)


def _upper_tail_mean(values: torch.Tensor, beta: float, min_count: int = 4) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    if values.numel() < max(1, int(min_count)):
        return values.mean()
    k = max(1, int(torch.ceil(values.new_tensor(float(values.numel()) * float(beta))).item()))
    return torch.topk(values, k=min(k, values.numel()), largest=True).values.mean()


def _soft_compare_swap_desc(
    scores: torch.Tensor,
    carriers: torch.Tensor,
    i: int,
    j: int,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    left_score = scores[i]
    right_score = scores[j]
    keep_left = torch.sigmoid((left_score - right_score) / max(float(tau), 1e-6))
    new_left_score = keep_left * left_score + (1.0 - keep_left) * right_score
    new_right_score = (1.0 - keep_left) * left_score + keep_left * right_score
    left_carrier = carriers[i]
    right_carrier = carriers[j]
    new_left_carrier = keep_left * left_carrier + (1.0 - keep_left) * right_carrier
    new_right_carrier = (1.0 - keep_left) * left_carrier + keep_left * right_carrier
    scores = scores.clone()
    carriers = carriers.clone()
    scores[i] = new_left_score
    scores[j] = new_right_score
    carriers[i] = new_left_carrier
    carriers[j] = new_right_carrier
    return scores, carriers


def soft_sort_order_residual(logits: torch.Tensor, target: torch.Tensor, tau: float = 0.25) -> torch.Tensor:
    scores = logits.view(-1).float()
    carriers = target.view(-1).float()
    n = scores.numel()
    if n <= 1 or carriers.min() == carriers.max():
        return scores.new_zeros(())
    scores = (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(1e-6)
    for _ in range(n):
        for start in (0, 1):
            for i in range(start, n - 1, 2):
                scores, carriers = _soft_compare_swap_desc(scores, carriers, i, i + 1, tau)
    positives = int(target.view(-1).float().sum().detach().item())
    ideal = torch.zeros_like(carriers)
    if positives > 0:
        ideal[:positives] = 1.0
    return F.mse_loss(carriers, ideal)


def bernoulli_kl_from_logits(q_logits: torch.Tensor, p_logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    q = torch.sigmoid(q_logits).clamp(eps, 1.0 - eps)
    p = torch.sigmoid(p_logits).clamp(eps, 1.0 - eps)
    return q * (q / p).log() + (1.0 - q) * ((1.0 - q) / (1.0 - p)).log()


class VetoSelectLoss(nn.Module):
    """Three-action positive-claim abstention loss.

    Positive samples target accepted puzzle evidence. Negative samples can be
    softly assigned to rejected positive evidence using the detached raw puzzle
    logit after warmup.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        tau_e: float = 1.5,
        d_max: float = 0.85,
        gamma_decoy: float = 1.0,
        lambda_anchor: float = 0.15,
    ) -> None:
        super().__init__()
        self.tau_e = float(tau_e)
        self.d_max = float(d_max)
        self.gamma_decoy = float(gamma_decoy)
        self.lambda_anchor = float(lambda_anchor)
        if pos_weight is None:
            pos_weight = torch.tensor([1.0], dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight.detach().float().view(1))

    def forward(
        self,
        output: dict[str, torch.Tensor],
        target: torch.Tensor,
        *,
        enable_decoys: bool = True,
        texture: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = output["puzzle_logit"].view(-1)
        a = output["selector_logit"].view(-1)
        y = target.float().view(-1)
        if texture is None:
            texture = torch.ones_like(y)
        else:
            texture = texture.float().view(-1).clamp(0.0, 1.0)

        log_pi_n = F.logsigmoid(-z)
        log_pi_r = F.logsigmoid(z) + F.logsigmoid(-a)
        log_pi_p = F.logsigmoid(z) + F.logsigmoid(a)

        if enable_decoys:
            evidence = torch.sigmoid(z.detach() / self.tau_e)
            d = ((1.0 - y) * evidence * texture).clamp(0.0, self.d_max)
        else:
            d = torch.zeros_like(y)

        q_n = (1.0 - y) * (1.0 - d)
        q_r = (1.0 - y) * d
        q_p = y
        tri_loss = -(q_n * log_pi_n + q_r * log_pi_r + q_p * log_pi_p)

        pos_weight = self.pos_weight.to(device=y.device, dtype=y.dtype).expand_as(y)
        weights = torch.where(y > 0.5, pos_weight, 1.0 + self.gamma_decoy * d)
        anchor_mask = y + (1.0 - y) * (1.0 - d)
        anchor_loss = F.binary_cross_entropy_with_logits(z, y, reduction="none") * anchor_mask
        return (weights * tri_loss).mean() + self.lambda_anchor * anchor_loss.mean()


class DykstraLCPLoss(nn.Module):
    """Binary loss with solver-residual shaping and binary-only hard-negative emphasis."""

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        hard_negative_fraction: float = 0.20,
        hard_negative_weight: float = 1.5,
        lambda_pos_residual: float = 0.03,
        lambda_neg_margin: float = 0.02,
        lambda_decay: float = 0.01,
        negative_projection_margin: float = 0.20,
    ) -> None:
        super().__init__()
        if pos_weight is None:
            pos_weight = torch.tensor([1.0], dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight.detach().float().view(1))
        self.hard_negative_fraction = float(hard_negative_fraction)
        self.hard_negative_weight = float(hard_negative_weight)
        self.lambda_pos_residual = float(lambda_pos_residual)
        self.lambda_neg_margin = float(lambda_neg_margin)
        self.lambda_decay = float(lambda_decay)
        self.negative_projection_margin = float(negative_projection_margin)

    def forward(self, output: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        logits = output["logits"].view(-1)
        y = target.float().view(-1)
        pos_weight = self.pos_weight.to(device=y.device, dtype=y.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")

        weights = torch.ones_like(bce)
        negative_idx = torch.nonzero(y < 0.5, as_tuple=False).view(-1)
        if negative_idx.numel() > 0 and self.hard_negative_fraction > 0.0 and self.hard_negative_weight > 1.0:
            k = max(1, int(round(float(negative_idx.numel()) * min(self.hard_negative_fraction, 1.0))))
            hard_scores = bce.detach()[negative_idx]
            hard_idx = negative_idx[torch.topk(hard_scores, k=k, largest=True).indices]
            weights[hard_idx] = self.hard_negative_weight
        base_loss = (weights * bce).mean()

        projection = output.get("projection_distance")
        trace = output.get("trace_residual")
        if projection is None or trace is None:
            return base_loss
        projection = projection.float().view(-1)
        trace = trace.float().view(-1)
        pos_residual = y * (projection + trace)
        neg_margin = (1.0 - y) * F.relu(self.negative_projection_margin - projection).pow(2)
        decay = output.get("decay_violation")
        if decay is None:
            decay_loss = torch.zeros((), device=y.device, dtype=y.dtype)
        else:
            decay_loss = decay.float().view(-1).mean()

        return (
            base_loss
            + self.lambda_pos_residual * pos_residual.mean()
            + self.lambda_neg_margin * neg_margin.mean()
            + self.lambda_decay * decay_loss
        )


class DykstraVetoSelectLoss(nn.Module):
    """VetoSelect loss whose decoy mining is weighted by Dykstra projection closeness."""

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        tau_e: float = 1.5,
        d_max: float = 0.85,
        gamma_decoy: float = 1.0,
        lambda_anchor: float = 0.12,
        projection_temperature: float = 0.04,
        trace_temperature: float = 0.006,
        lambda_pos_residual: float = 0.02,
        lambda_neg_margin: float = 0.01,
        lambda_decay: float = 0.01,
        negative_projection_margin: float = 0.04,
    ) -> None:
        super().__init__()
        if pos_weight is None:
            pos_weight = torch.tensor([1.0], dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight.detach().float().view(1))
        self.tau_e = float(tau_e)
        self.d_max = float(d_max)
        self.gamma_decoy = float(gamma_decoy)
        self.lambda_anchor = float(lambda_anchor)
        self.projection_temperature = max(float(projection_temperature), 1e-6)
        self.trace_temperature = max(float(trace_temperature), 1e-6)
        self.lambda_pos_residual = float(lambda_pos_residual)
        self.lambda_neg_margin = float(lambda_neg_margin)
        self.lambda_decay = float(lambda_decay)
        self.negative_projection_margin = float(negative_projection_margin)

    def forward(
        self,
        output: dict[str, torch.Tensor],
        target: torch.Tensor,
        *,
        enable_decoys: bool = True,
        texture: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = output["puzzle_logit"].view(-1)
        a = output["selector_logit"].view(-1)
        y = target.float().view(-1)
        if texture is None:
            texture = torch.ones_like(y)
        else:
            texture = texture.float().view(-1).clamp(0.0, 1.0)

        projection = output["projection_distance"].float().view(-1)
        trace = output["trace_residual"].float().view(-1)

        log_pi_n = F.logsigmoid(-z)
        log_pi_r = F.logsigmoid(z) + F.logsigmoid(-a)
        log_pi_p = F.logsigmoid(z) + F.logsigmoid(a)

        if enable_decoys:
            evidence = torch.sigmoid(z.detach() / self.tau_e)
            projection_closeness = torch.exp((-projection.detach() / self.projection_temperature).clamp_min(-50.0))
            trace_closeness = torch.exp((-trace.detach() / self.trace_temperature).clamp_min(-50.0))
            d = ((1.0 - y) * evidence * texture * projection_closeness * trace_closeness).clamp(0.0, self.d_max)
        else:
            d = torch.zeros_like(y)

        q_n = (1.0 - y) * (1.0 - d)
        q_r = (1.0 - y) * d
        q_p = y
        tri_loss = -(q_n * log_pi_n + q_r * log_pi_r + q_p * log_pi_p)

        pos_weight = self.pos_weight.to(device=y.device, dtype=y.dtype).expand_as(y)
        weights = torch.where(y > 0.5, pos_weight, 1.0 + self.gamma_decoy * d)
        anchor_mask = y + (1.0 - y) * (1.0 - d)
        anchor_loss = F.binary_cross_entropy_with_logits(z, y, reduction="none") * anchor_mask

        pos_residual = y * (projection + trace)
        neg_margin = (1.0 - y) * F.relu(self.negative_projection_margin - projection).pow(2)
        decay = output.get("decay_violation")
        if decay is None:
            decay_loss = torch.zeros((), device=y.device, dtype=y.dtype)
        else:
            decay_loss = decay.float().view(-1).mean()

        return (
            (weights * tri_loss).mean()
            + self.lambda_anchor * anchor_loss.mean()
            + self.lambda_pos_residual * pos_residual.mean()
            + self.lambda_neg_margin * neg_margin.mean()
            + self.lambda_decay * decay_loss
        )


class SRPALoss(nn.Module):
    """Sparse Relation Pursuit Asymmetry loss.

    The target supervises only the binary puzzle label. The auxiliary terms keep
    the sparse-code bottleneck meaningful: tactical samples should reconstruct
    better under the tactical dictionary, background samples under the background
    dictionary, while both dictionaries stay sparse, non-degenerate, and distinct.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        lambda_aux: float = 0.15,
        lambda_residual: float = 0.02,
        lambda_l1: float = 0.001,
        lambda_group: float = 0.001,
        lambda_dictionary_coherence: float = 0.0005,
        lambda_branch_separation: float = 0.0005,
        lambda_dead_group: float = 0.0001,
    ) -> None:
        super().__init__()
        if pos_weight is None:
            pos_weight = torch.tensor([1.0], dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight.detach().float().view(1))
        self.lambda_aux = float(lambda_aux)
        self.lambda_residual = float(lambda_residual)
        self.lambda_l1 = float(lambda_l1)
        self.lambda_group = float(lambda_group)
        self.lambda_dictionary_coherence = float(lambda_dictionary_coherence)
        self.lambda_branch_separation = float(lambda_branch_separation)
        self.lambda_dead_group = float(lambda_dead_group)

    def forward(self, output: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        logits = output["logits"].view(-1)
        y = target.float().view(-1)
        pos_weight = self.pos_weight.to(device=y.device, dtype=y.dtype)
        base = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)

        aux_logit = output["aux_logit"].view(-1)
        aux = F.binary_cross_entropy_with_logits(aux_logit, y, pos_weight=pos_weight)
        bg_residual = output["bg_final_residual"].float().view(-1)
        tac_residual = output["tac_final_residual"].float().view(-1)
        class_conditional_residual = (y * tac_residual + (1.0 - y) * bg_residual).mean()

        mean_abs_code = output["mean_abs_code"].float().view(-1).mean()
        mean_group_norm = output["mean_group_norm"].float().view(-1).mean()
        dictionary_coherence = output["dictionary_coherence"].float()
        branch_separation = output["branch_separation"].float()
        dead_group = output["dead_group_penalty"].float()

        return (
            base
            + self.lambda_aux * aux
            + self.lambda_residual * class_conditional_residual
            + self.lambda_l1 * mean_abs_code
            + self.lambda_group * mean_group_norm
            + self.lambda_dictionary_coherence * dictionary_coherence
            + self.lambda_branch_separation * branch_separation
            + self.lambda_dead_group * dead_group
        )


class ContaminationDROHuberTailLoss(nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        lambda_tail: float = 0.35,
        margin: float = 0.25,
        kappa: float = 1.0,
        beta: float = 0.25,
        min_near_count: int = 4,
    ) -> None:
        super().__init__()
        if pos_weight is None:
            pos_weight = torch.tensor([1.0], dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight.detach().float().view(1))
        self.lambda_tail = float(lambda_tail)
        self.margin = float(margin)
        self.kappa = float(kappa)
        self.beta = float(beta)
        self.min_near_count = int(min_near_count)

    def forward(
        self,
        output: dict[str, torch.Tensor],
        target: torch.Tensor,
        fine_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = output["logits"].view(-1)
        y = target.float().view(-1)
        pos_weight = self.pos_weight.to(device=y.device, dtype=y.dtype)
        base = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        fine = _fine_label_tensor(fine_label, target)
        near_mask = fine == 1
        residual = F.relu(logits[near_mask] + self.margin)
        tail = _upper_tail_mean(_huber_positive_residual(residual, self.kappa), self.beta, self.min_near_count)
        return base + self.lambda_tail * tail


class MaterialLockedDROLoss(nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        gamma_near: float = 2.0,
        lambda_robust: float = 0.5,
        lambda_budget: float = 0.02,
    ) -> None:
        super().__init__()
        if pos_weight is None:
            pos_weight = torch.tensor([1.0], dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight.detach().float().view(1))
        self.gamma_near = float(gamma_near)
        self.lambda_robust = float(lambda_robust)
        self.lambda_budget = float(lambda_budget)

    def forward(
        self,
        output: dict[str, torch.Tensor],
        target: torch.Tensor,
        fine_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        clean_logits = output["logits"].view(-1)
        adversarial_logits = output.get("adversarial_logits", clean_logits).view(-1)
        y = target.float().view(-1)
        fine = _fine_label_tensor(fine_label, target)
        pos_weight = self.pos_weight.to(device=y.device, dtype=y.dtype)
        clean = F.binary_cross_entropy_with_logits(clean_logits, y, pos_weight=pos_weight, reduction="none")
        robust = F.binary_cross_entropy_with_logits(adversarial_logits, y, pos_weight=pos_weight, reduction="none")
        near_weight = torch.where(fine == 1, 1.0 + self.gamma_near, 1.0)
        budget = output.get("mask_budget_used")
        budget_loss = budget.float().view(-1).mean() if budget is not None else clean_logits.new_zeros(())
        return (near_weight * clean).mean() + self.lambda_robust * (near_weight * robust).mean() + self.lambda_budget * budget_loss


class SoftSortOrderResidualLoss(nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        lambda_order: float = 0.25,
        tau: float = 0.25,
    ) -> None:
        super().__init__()
        if pos_weight is None:
            pos_weight = torch.tensor([1.0], dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight.detach().float().view(1))
        self.lambda_order = float(lambda_order)
        self.tau = float(tau)

    def forward(self, output: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        logits = output["logits"].view(-1)
        y = target.float().view(-1)
        pos_weight = self.pos_weight.to(device=y.device, dtype=y.dtype)
        base = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        return base + self.lambda_order * soft_sort_order_residual(logits, y, tau=self.tau)


class ConditionalSurprisalGateLoss(nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        lambda_kl: float = 0.05,
        lambda_capacity: float = 0.05,
        target_gate_rate: float = 0.35,
    ) -> None:
        super().__init__()
        if pos_weight is None:
            pos_weight = torch.tensor([1.0], dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight.detach().float().view(1))
        self.lambda_kl = float(lambda_kl)
        self.lambda_capacity = float(lambda_capacity)
        self.target_gate_rate = float(target_gate_rate)

    def forward(self, output: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        logits = output["logits"].view(-1)
        y = target.float().view(-1)
        pos_weight = self.pos_weight.to(device=y.device, dtype=y.dtype)
        base = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        prior = output["prior_logits"].view(-1)
        posterior = output["posterior_logits"].view(-1)
        kl = bernoulli_kl_from_logits(posterior, prior).mean()
        gate_mean = output["gate_mean"].float().view(-1).mean()
        capacity = F.relu(gate_mean - self.target_gate_rate).pow(2)
        return base + self.lambda_kl * kl + self.lambda_capacity * capacity
