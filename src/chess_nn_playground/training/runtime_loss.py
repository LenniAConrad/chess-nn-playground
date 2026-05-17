from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.data.dataset import ChessPositionDataset
from chess_nn_playground.training.losses import (
    ConditionalSurprisalGateLoss,
    ContaminationDROHuberTailLoss,
    DykstraLCPLoss,
    DykstraVetoSelectLoss,
    MaterialLockedDROLoss,
    SRPALoss,
    SoftSortOrderResidualLoss,
    VetoSelectLoss,
    binary_cross_entropy_loss,
    cross_entropy_loss,
)
from chess_nn_playground.training.runtime_data import binary_pos_weight_tensor, class_weight_tensor


@dataclass(frozen=True)
class LossRuntimeConfig:
    class_weighting: Any
    loss_name: str
    veto_select_cfg: dict[str, Any]
    dykstra_lcp_cfg: dict[str, Any]
    dykstra_vetoselect_cfg: dict[str, Any]
    srpa_cfg: dict[str, Any]
    contamination_dro_cfg: dict[str, Any]
    material_locked_dro_cfg: dict[str, Any]
    soft_sort_order_cfg: dict[str, Any]
    conditional_surprisal_gate_cfg: dict[str, Any]
    veto_select_warmup_epochs: int
    use_rule_texture: bool


@dataclass(frozen=True)
class LossRuntime:
    config: LossRuntimeConfig
    criterion: nn.Module


def _dict_cfg(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def resolve_loss_runtime_config(
    *,
    training_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
) -> LossRuntimeConfig:
    veto_select_cfg = _dict_cfg(training_cfg.get("veto_select", {}))
    return LossRuntimeConfig(
        class_weighting=training_cfg.get("class_weighting", "none"),
        loss_name=str(training_cfg.get("loss", "")).strip().lower(),
        veto_select_cfg=veto_select_cfg,
        dykstra_lcp_cfg=_dict_cfg(training_cfg.get("dykstra_lcp", {})),
        dykstra_vetoselect_cfg=_dict_cfg(training_cfg.get("dykstra_vetoselect", {})),
        srpa_cfg=_dict_cfg(training_cfg.get("srpa", {})),
        contamination_dro_cfg=_dict_cfg(training_cfg.get("contamination_dro", {})),
        material_locked_dro_cfg=_dict_cfg(training_cfg.get("material_locked_dro", {})),
        soft_sort_order_cfg=_dict_cfg(training_cfg.get("soft_sort_order", {})),
        conditional_surprisal_gate_cfg=_dict_cfg(training_cfg.get("conditional_surprisal_gate", {})),
        veto_select_warmup_epochs=int(veto_select_cfg.get("warmup_epochs", 1)),
        use_rule_texture=bool(veto_select_cfg.get("use_rule_texture", data_cfg.get("include_rule_texture", False))),
    )


def _require_single_logit(loss_name: str, single_logit_binary: bool) -> None:
    if not single_logit_binary:
        raise ValueError(f"training.loss={loss_name} requires a single-logit binary model")


def _balanced_binary_pos_weight(
    *,
    config: LossRuntimeConfig,
    train_dataset: ChessPositionDataset,
    device: torch.device,
) -> torch.Tensor | None:
    if config.class_weighting == "balanced":
        return binary_pos_weight_tensor(train_dataset, device=device)
    return None


def build_loss_runtime(
    config: LossRuntimeConfig,
    *,
    train_dataset: ChessPositionDataset,
    metric_num_classes: int,
    single_logit_binary: bool,
    device: torch.device,
) -> LossRuntime:
    if config.loss_name == "veto_select":
        _require_single_logit(config.loss_name, single_logit_binary)
        criterion = VetoSelectLoss(
            pos_weight=_balanced_binary_pos_weight(config=config, train_dataset=train_dataset, device=device),
            tau_e=float(config.veto_select_cfg.get("tau_e", 1.5)),
            d_max=float(config.veto_select_cfg.get("d_max", 0.85)),
            gamma_decoy=float(config.veto_select_cfg.get("gamma_decoy", 1.0)),
            lambda_anchor=float(config.veto_select_cfg.get("lambda_anchor", 0.15)),
        )
    elif config.loss_name == "dykstra_lcp":
        _require_single_logit(config.loss_name, single_logit_binary)
        criterion = DykstraLCPLoss(
            pos_weight=_balanced_binary_pos_weight(config=config, train_dataset=train_dataset, device=device),
            hard_negative_fraction=float(config.dykstra_lcp_cfg.get("hard_negative_fraction", 0.20)),
            hard_negative_weight=float(config.dykstra_lcp_cfg.get("hard_negative_weight", 1.5)),
            lambda_pos_residual=float(config.dykstra_lcp_cfg.get("lambda_pos_residual", 0.03)),
            lambda_neg_margin=float(config.dykstra_lcp_cfg.get("lambda_neg_margin", 0.02)),
            lambda_decay=float(config.dykstra_lcp_cfg.get("lambda_decay", 0.01)),
            negative_projection_margin=float(config.dykstra_lcp_cfg.get("negative_projection_margin", 0.20)),
        )
    elif config.loss_name == "dykstra_vetoselect":
        _require_single_logit(config.loss_name, single_logit_binary)
        criterion = DykstraVetoSelectLoss(
            pos_weight=_balanced_binary_pos_weight(config=config, train_dataset=train_dataset, device=device),
            tau_e=float(config.veto_select_cfg.get("tau_e", 1.5)),
            d_max=float(config.veto_select_cfg.get("d_max", 0.85)),
            gamma_decoy=float(config.veto_select_cfg.get("gamma_decoy", 1.0)),
            lambda_anchor=float(config.veto_select_cfg.get("lambda_anchor", 0.12)),
            projection_temperature=float(config.dykstra_vetoselect_cfg.get("projection_temperature", 0.04)),
            trace_temperature=float(config.dykstra_vetoselect_cfg.get("trace_temperature", 0.006)),
            lambda_pos_residual=float(config.dykstra_vetoselect_cfg.get("lambda_pos_residual", 0.02)),
            lambda_neg_margin=float(config.dykstra_vetoselect_cfg.get("lambda_neg_margin", 0.01)),
            lambda_decay=float(config.dykstra_vetoselect_cfg.get("lambda_decay", 0.01)),
            negative_projection_margin=float(config.dykstra_vetoselect_cfg.get("negative_projection_margin", 0.04)),
        )
    elif config.loss_name == "srpa":
        _require_single_logit(config.loss_name, single_logit_binary)
        criterion = SRPALoss(
            pos_weight=_balanced_binary_pos_weight(config=config, train_dataset=train_dataset, device=device),
            lambda_aux=float(config.srpa_cfg.get("lambda_aux", 0.15)),
            lambda_residual=float(config.srpa_cfg.get("lambda_residual", 0.02)),
            lambda_l1=float(config.srpa_cfg.get("lambda_l1", 0.001)),
            lambda_group=float(config.srpa_cfg.get("lambda_group", 0.001)),
            lambda_dictionary_coherence=float(config.srpa_cfg.get("lambda_dictionary_coherence", 0.0005)),
            lambda_branch_separation=float(config.srpa_cfg.get("lambda_branch_separation", 0.0005)),
            lambda_dead_group=float(config.srpa_cfg.get("lambda_dead_group", 0.0001)),
        )
    elif config.loss_name == "contamination_dro_huber":
        _require_single_logit(config.loss_name, single_logit_binary)
        criterion = ContaminationDROHuberTailLoss(
            pos_weight=_balanced_binary_pos_weight(config=config, train_dataset=train_dataset, device=device),
            lambda_tail=float(config.contamination_dro_cfg.get("lambda_tail", 0.35)),
            margin=float(config.contamination_dro_cfg.get("margin", 0.25)),
            kappa=float(config.contamination_dro_cfg.get("kappa", 1.0)),
            beta=float(config.contamination_dro_cfg.get("beta", 0.25)),
            min_near_count=int(config.contamination_dro_cfg.get("min_near_count", 4)),
        )
    elif config.loss_name == "material_locked_dro":
        _require_single_logit(config.loss_name, single_logit_binary)
        criterion = MaterialLockedDROLoss(
            pos_weight=_balanced_binary_pos_weight(config=config, train_dataset=train_dataset, device=device),
            gamma_near=float(config.material_locked_dro_cfg.get("gamma_near", 2.0)),
            lambda_robust=float(config.material_locked_dro_cfg.get("lambda_robust", 0.5)),
            lambda_budget=float(config.material_locked_dro_cfg.get("lambda_budget", 0.02)),
        )
    elif config.loss_name == "soft_sort_order":
        _require_single_logit(config.loss_name, single_logit_binary)
        criterion = SoftSortOrderResidualLoss(
            pos_weight=_balanced_binary_pos_weight(config=config, train_dataset=train_dataset, device=device),
            lambda_order=float(config.soft_sort_order_cfg.get("lambda_order", 0.25)),
            tau=float(config.soft_sort_order_cfg.get("tau", 0.25)),
        )
    elif config.loss_name == "conditional_surprisal_gate":
        _require_single_logit(config.loss_name, single_logit_binary)
        criterion = ConditionalSurprisalGateLoss(
            pos_weight=_balanced_binary_pos_weight(config=config, train_dataset=train_dataset, device=device),
            lambda_kl=float(config.conditional_surprisal_gate_cfg.get("lambda_kl", 0.05)),
            lambda_capacity=float(config.conditional_surprisal_gate_cfg.get("lambda_capacity", 0.05)),
            target_gate_rate=float(config.conditional_surprisal_gate_cfg.get("target_gate_rate", 0.35)),
        )
    elif single_logit_binary:
        criterion = binary_cross_entropy_loss(
            _balanced_binary_pos_weight(config=config, train_dataset=train_dataset, device=device)
        )
    else:
        class_weights = None
        if config.class_weighting == "balanced":
            class_weights = class_weight_tensor(train_dataset, num_classes=metric_num_classes, device=device)
        criterion = cross_entropy_loss(class_weights)
    return LossRuntime(config=config, criterion=criterion)


def compute_training_loss(
    *,
    loss_name: str,
    criterion: nn.Module,
    output: torch.Tensor | dict[str, torch.Tensor],
    logits: torch.Tensor,
    target: torch.Tensor,
    epoch: int | None,
    veto_select_warmup_epochs: int,
    single_logit_binary: bool,
    texture: torch.Tensor | None = None,
    fine_label: torch.Tensor | None = None,
) -> torch.Tensor:
    if loss_name == "veto_select":
        if not isinstance(output, dict):
            raise ValueError("training.loss=veto_select requires model output diagnostics")
        enable_decoys = epoch is None or epoch > veto_select_warmup_epochs
        return criterion(output, target, enable_decoys=enable_decoys, texture=texture)
    if loss_name == "dykstra_lcp":
        if not isinstance(output, dict):
            raise ValueError("training.loss=dykstra_lcp requires model output diagnostics")
        return criterion(output, target)
    if loss_name == "dykstra_vetoselect":
        if not isinstance(output, dict):
            raise ValueError("training.loss=dykstra_vetoselect requires model output diagnostics")
        enable_decoys = epoch is None or epoch > veto_select_warmup_epochs
        return criterion(output, target, enable_decoys=enable_decoys, texture=texture)
    if loss_name == "srpa":
        if not isinstance(output, dict):
            raise ValueError("training.loss=srpa requires model output diagnostics")
        return criterion(output, target)
    if loss_name == "contamination_dro_huber":
        if not isinstance(output, dict):
            raise ValueError("training.loss=contamination_dro_huber requires model output diagnostics")
        return criterion(output, target, fine_label=fine_label)
    if loss_name == "material_locked_dro":
        if not isinstance(output, dict):
            raise ValueError("training.loss=material_locked_dro requires model output diagnostics")
        return criterion(output, target, fine_label=fine_label)
    if loss_name == "soft_sort_order":
        if not isinstance(output, dict):
            raise ValueError("training.loss=soft_sort_order requires model output diagnostics")
        return criterion(output, target)
    if loss_name == "conditional_surprisal_gate":
        if not isinstance(output, dict):
            raise ValueError("training.loss=conditional_surprisal_gate requires model output diagnostics")
        return criterion(output, target)
    if single_logit_binary:
        return criterion(logits.view(-1), target.float())
    return criterion(logits, target)
