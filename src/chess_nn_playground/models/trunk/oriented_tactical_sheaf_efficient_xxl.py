"""Efficient scale-XXL successor to i018 for idea i254.

The research packet
``ideas/research/packets/classic/i254_efficient_i018_scale_xxl.md`` argues that
the right XXL move for the i018 ``oriented_tactical_sheaf_laplacian`` family is
**not** to scale every knob at once. The repo evidence is that i018 improves
from base to scale_xl, the falsifier shows the typed chess relation graph is
load-bearing, and the i249 attempt failed because it bundled speculative
execution changes with an architecture rewrite and verified equivalence only
in eval-mode fp32.

The thesis for i254 is therefore narrow and conservative:

* preserve the 12-relation tactical incidence thesis exactly
  (`BoardStateAdapter`, `TacticalIncidenceBuilder`, `SquareTokenEncoder`,
  `TriadDefectPool`, signs, gates, heat step, readout shape);
* preserve the current numerical execution path (eager per-relation loop)
  rather than introducing speculative fusion;
* direct the extra scale budget into capacity that does not multiply
  irregular relation work (wider square-token channels and a much larger
  readout hidden dimension);
* allow a structured, grouped low-rank restriction parameterization behind
  a config flag so a later stalk-scaling experiment is cheap to run, but
  default to the original full restriction maps so the first XXL run only
  changes width.

This module therefore exposes one new sheaf diffusion block,
``EfficientSheafDiffusionBlock``, that supports both:

* ``restriction_mode="full"`` (default) — identical parameter shape and
  numerical behaviour to i018's :class:`SheafDiffusionBlock`. The first XXL
  benchmark runs in this mode so width-only scaling can be evaluated against
  i018 with no parameterization confound.
* ``restriction_mode="grouped_lowrank"`` — restriction maps are written as
  ``I + U_g diag(a_r) V_g^T`` with group-shared bases ``U_g, V_g`` and
  relation-specific diagonal coefficients. The default group partition splits
  the 12 typed relations into ``{attack, defense, ray, pin}`` (G=4).

The trunk wrapper :class:`OrientedTacticalSheafEfficientXXLNet` subclasses
i018's :class:`OrientedTacticalSheafNet` and only swaps the block list. The
inherited forward, diagnostic contract, falsifier knob, and readout shape are
unchanged so the comparison stays apples-to-apples.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from chess_nn_playground.models.trunk.oriented_tactical_sheaf import (
    RELATION_NAMES,
    OrientedTacticalSheafNet,
)


RELATION_COUNT = len(RELATION_NAMES)


# Default 4-group semantic partition of the 12 typed tactical relations.
# Indices follow ``RELATION_NAMES`` order.
DEFAULT_RELATION_GROUPS_4: tuple[int, ...] = (
    0,  # us_attacks_them_piece -> attack
    0,  # them_attacks_us_piece -> attack
    1,  # us_defends_us_piece -> defense
    1,  # them_defends_them_piece -> defense
    0,  # us_attacks_empty_near_king -> attack
    0,  # them_attacks_empty_near_king -> attack
    2,  # bishop_ray_visible -> ray
    2,  # rook_ray_visible -> ray
    2,  # queen_ray_visible -> ray
    0,  # knight_attack -> attack
    0,  # pawn_attack_forward_oriented -> attack
    3,  # king_ray_pin_candidate -> pin
)


class EfficientSheafDiffusionBlock(nn.Module):
    """Sheaf diffusion block with an optional structured restriction family.

    ``restriction_mode="full"`` reproduces the parameter shape and forward
    semantics of i018's :class:`SheafDiffusionBlock` exactly. Same
    initialization (``I + small noise``), same per-relation loop, same heat
    step. This is the first-XXL setting.

    ``restriction_mode="grouped_lowrank"`` parameterizes restriction maps as
    ``rho_r = I + U_g(r) diag(a_r) V_g(r)^T`` with group-shared bases. The
    materialized ``(R, s, s)`` tensor is recomputed each forward, so the rest
    of the block reduces to the same matrix products as the full case. The
    point of the grouped low-rank family is parameter sharing, not a fused
    custom kernel.
    """

    def __init__(
        self,
        d_model: int,
        relation_count: int,
        stalk_dim: int,
        dropout: float,
        restriction_mode: str = "full",
        restriction_rank: int = 4,
        relation_groups: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.relation_count = int(relation_count)
        self.stalk_dim = int(stalk_dim)
        self.restriction_mode = str(restriction_mode)
        self.restriction_rank = int(restriction_rank)
        self.node_to_stalk = nn.Linear(d_model, stalk_dim)
        self.stalk_to_node = nn.Linear(stalk_dim, d_model)
        self.relation_gate_logits = nn.Parameter(torch.zeros(relation_count))
        self.eta_logit = nn.Parameter(torch.tensor(0.0))
        signs = torch.tensor(
            [-1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=torch.float32
        )
        self.register_buffer("relation_signs", signs, persistent=False)
        self.node_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

        if self.restriction_mode == "full":
            eye = torch.eye(stalk_dim).unsqueeze(0).repeat(relation_count, 1, 1)
            self.rho_src = nn.Parameter(
                eye + 0.02 * torch.randn(relation_count, stalk_dim, stalk_dim)
            )
            self.rho_dst = nn.Parameter(
                eye + 0.02 * torch.randn(relation_count, stalk_dim, stalk_dim)
            )
        elif self.restriction_mode == "grouped_lowrank":
            if relation_groups is None:
                raise ValueError(
                    "grouped_lowrank restriction_mode requires relation_groups"
                )
            groups = tuple(int(g) for g in relation_groups)
            if len(groups) != relation_count:
                raise ValueError(
                    "relation_groups must have length equal to relation_count"
                )
            if min(groups) < 0:
                raise ValueError("relation_groups must be non-negative")
            group_count = max(groups) + 1
            self.relation_groups = groups
            self.group_count = int(group_count)
            self.register_buffer(
                "relation_to_group",
                torch.tensor(groups, dtype=torch.long),
                persistent=False,
            )
            std = 0.02
            self.U_src = nn.Parameter(
                std * torch.randn(group_count, stalk_dim, self.restriction_rank)
            )
            self.V_src = nn.Parameter(
                std * torch.randn(group_count, stalk_dim, self.restriction_rank)
            )
            self.U_dst = nn.Parameter(
                std * torch.randn(group_count, stalk_dim, self.restriction_rank)
            )
            self.V_dst = nn.Parameter(
                std * torch.randn(group_count, stalk_dim, self.restriction_rank)
            )
            self.a_src = nn.Parameter(
                std * torch.randn(relation_count, self.restriction_rank)
            )
            self.a_dst = nn.Parameter(
                std * torch.randn(relation_count, self.restriction_rank)
            )
        else:
            raise ValueError(
                f"Unknown restriction_mode: {self.restriction_mode!r}"
                " (expected 'full' or 'grouped_lowrank')"
            )

    def _restriction_maps(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.restriction_mode == "full":
            return self.rho_src, self.rho_dst
        s = self.stalk_dim
        eye = torch.eye(s, device=self.U_src.device, dtype=self.U_src.dtype)
        u_src_r = self.U_src[self.relation_to_group]
        v_src_r = self.V_src[self.relation_to_group]
        u_dst_r = self.U_dst[self.relation_to_group]
        v_dst_r = self.V_dst[self.relation_to_group]
        rho_src = eye + torch.matmul(
            u_src_r * self.a_src.unsqueeze(-2), v_src_r.transpose(-2, -1)
        )
        rho_dst = eye + torch.matmul(
            u_dst_r * self.a_dst.unsqueeze(-2), v_dst_r.transpose(-2, -1)
        )
        return rho_src, rho_dst

    def forward(
        self, h: torch.Tensor, relation_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.profiler.record_function("i254/efficient_sheaf_block"):
            z = self.node_to_stalk(h)
            batch, squares, stalk_dim = z.shape
            gates = 2.0 * torch.sigmoid(self.relation_gate_logits)
            eta = 0.25 * torch.sigmoid(self.eta_logit)
            z_update = z.new_zeros(batch, squares, stalk_dim)
            degree = z.new_zeros(batch, squares)
            energies: list[torch.Tensor] = []
            rho_src_all, rho_dst_all = self._restriction_maps()
            with torch.profiler.record_function("i254/per_relation_loop"):
                for relation_idx in range(self.relation_count):
                    weights = relation_masks[:, relation_idx]
                    rho_src = rho_src_all[relation_idx]
                    rho_dst = rho_dst_all[relation_idx]
                    sign = self.relation_signs[relation_idx]
                    src = torch.matmul(z, rho_src)
                    dst = torch.matmul(z, rho_dst)
                    residual = dst.unsqueeze(1) - sign * src.unsqueeze(2)
                    weighted_residual = (
                        gates[relation_idx] * weights.unsqueeze(-1) * residual
                    )
                    energy = (weighted_residual * residual).sum(dim=(1, 2, 3)) / weights.sum(
                        dim=(1, 2)
                    ).clamp_min(1.0)
                    energies.append(energy)
                    src_back = torch.matmul(weighted_residual, rho_src.t())
                    dst_back = torch.matmul(weighted_residual, rho_dst.t())
                    z_update = (
                        z_update + sign * src_back.sum(dim=2) - dst_back.sum(dim=1)
                    )
                    degree = degree + gates[relation_idx] * (
                        weights.sum(dim=2) + weights.sum(dim=1)
                    )
            z_update = eta * z_update / degree.unsqueeze(-1).clamp_min(1.0)
            h = self.norm(h + self.stalk_to_node(z_update) + self.node_mlp(h))
            return h, torch.stack(energies, dim=1), gates


class OrientedTacticalSheafEfficientXXLNet(OrientedTacticalSheafNet):
    """i018 architecture scaled wider with an optional structured restriction.

    The wrapper only changes the diffusion block list; every other module
    (`BoardStateAdapter`, `TacticalIncidenceBuilder`, `SquareTokenEncoder`,
    `TriadDefectPool`, readout head, diagnostic contract) is inherited
    unchanged. Width-only scaling moves the extra parameters into regular
    dense work that is friendly to BLAS and `torch.compile` later, instead of
    multiplying the irregular ``(B, 64, 64, s)`` relation work.
    """

    def __init__(
        self,
        *args: Any,
        restriction_mode: str = "full",
        restriction_rank: int = 4,
        relation_groups: Sequence[int] | None = None,
        compile_model: bool = False,
        fuse_incidence: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.restriction_mode = str(restriction_mode)
        self.restriction_rank = int(restriction_rank)
        # Speed knobs are accepted for config completeness but deliberately
        # default off: the research packet's first XXL run is a capacity-only
        # benchmark; compile-only and fused-incidence are a separate execution
        # branch that must pass the train-mode mixed-precision parity ladder
        # before being benchmarked for speed.
        self.compile_model = bool(compile_model)
        self.fuse_incidence = bool(fuse_incidence)

        depth = len(self.blocks)
        d_model = self.blocks[0].node_to_stalk.in_features
        stalk_dim = self.blocks[0].node_to_stalk.out_features
        dropout_module = self.blocks[0].node_mlp[3]
        dropout = float(dropout_module.p) if isinstance(dropout_module, nn.Dropout) else 0.1
        resolved_groups: Sequence[int] | None
        if self.restriction_mode == "grouped_lowrank":
            resolved_groups = (
                tuple(int(g) for g in relation_groups)
                if relation_groups is not None
                else DEFAULT_RELATION_GROUPS_4
            )
        else:
            resolved_groups = None
        self.relation_groups = resolved_groups
        self.blocks = nn.ModuleList(
            [
                EfficientSheafDiffusionBlock(
                    d_model,
                    RELATION_COUNT,
                    stalk_dim,
                    dropout,
                    restriction_mode=self.restriction_mode,
                    restriction_rank=self.restriction_rank,
                    relation_groups=resolved_groups,
                )
                for _ in range(depth)
            ]
        )


def build_oriented_tactical_sheaf_efficient_xxl_from_config(
    config: dict[str, Any],
) -> OrientedTacticalSheafEfficientXXLNet:
    relation_groups_cfg = config.get("relation_groups")
    if relation_groups_cfg is not None:
        relation_groups = tuple(int(g) for g in relation_groups_cfg)
    else:
        relation_groups = None
    return OrientedTacticalSheafEfficientXXLNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 160)),
        hidden_dim=int(config.get("hidden_dim", 320)),
        depth=int(config.get("sheaf_layers", config.get("depth", 4))),
        stalk_dim=int(config.get("stalk_dim", 8)),
        dropout=float(config.get("dropout", 0.1)),
        encoding=str(config.get("encoding", "simple_18")),
        piece_adapter=str(config.get("piece_adapter", "exact")),
        use_triads=bool(config.get("use_triads", True)),
        scramble_relations=bool(config.get("scramble_relations", False)),
        restriction_mode=str(config.get("restriction_mode", "full")),
        restriction_rank=int(config.get("restriction_rank", 4)),
        relation_groups=relation_groups,
        compile_model=bool(config.get("compile_model", False)),
        fuse_incidence=bool(config.get("fuse_incidence", False)),
    )
