"""Kirchhoff Mobility Solve (p045, KMS) primitive.

Source: ``ideas/research/primitives/external_40_symmetric_coalition_resolvent_primitives.md``
(Section 4 ``primitive_kirchhoff_mobility_solve``; promoted from #4 because the
file's #1 ``primitive_symmetric_coalition_pool`` and #5
``primitive_choquet_coalition_integral`` are both coalition-pool variants
covered by p042 / p046 (polynomial ledger and subset-log-partition); #2
``primitive_determinantal_coverage_pool`` is a log-det DPP variant; #3
``primitive_character_orbit_norm`` is in the orbit/irrep family deferred to a
future symmetry batch). The exact resolvent + bottleneck-detection
flavour of #4 fits this batch's "resolvent pooling" target.

The operator solves the SPD equilibrium

    L_b u_b = s_b,                  u_b in R^{64 x p},
    L_b = D^T diag(c_b) D + lambda I,

where ``D`` is the fixed grid vertex-edge incidence, ``c_b`` is an
input-dependent positive edge conductance (softplus of an MLP over edge
endpoints), and ``s_b = X_b W_s`` is a per-square source / sink term
built from the i193 spatial features. The solution ``u_b`` is the
exact electrical potential of the conductance system. Output:

    Y_b = u_b W_o in R^{64 x d'}.

A convolutional or unrolled-GNN layer applies a fixed number of local
steps and never converges to the equilibrium; this primitive returns
the exact resolvent of a learned conductance graph and propagates
gradients via implicit differentiation through
``torch.linalg.solve``.

The primitive is an additive gated logit delta over the i193 trunk:

    final_logit = base_logit + gate * primitive_delta_raw

CRTK metadata, source labels, verification flags, engine evaluations
and principal variations are *not* consumed.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F_

from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor
from chess_nn_playground.models.primitives.trunk_features import trunk_joint_features


GRID = 8


ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "uniform_conductance",   # primary falsifier — c_b := constant
    "diagonal_only",         # primary falsifier #2 — drop D^T diag(c) D, keep lambda*I (so u = s/lambda)
    "shuffle_conductance",   # in-batch permutation of the per-edge conductance
    "zero_source",           # zero the source term (u becomes zero -> head receives no signal)
    "zero_delta",
    "trunk_only",
    "disable_gate",
)


def _build_grid_incidence() -> torch.Tensor:
    """Build the 8x8 grid's vertex-edge incidence ``D in R^{64 x E}``.

    Edges are oriented horizontal (right) and vertical (down). Returns
    a single ``D`` so the Laplacian factors as ``D^T diag(c) D`` plus
    a ``lambda I`` shift inside the model forward.
    """
    n_v = GRID * GRID
    horiz = [(r, c) for r in range(GRID) for c in range(GRID - 1)]
    vert = [(r, c) for r in range(GRID - 1) for c in range(GRID)]
    n_edges = len(horiz) + len(vert)
    d = torch.zeros(n_v, n_edges, dtype=torch.float32)
    for idx, (r, c) in enumerate(horiz):
        head = r * GRID + (c + 1)
        tail = r * GRID + c
        d[head, idx] = 1.0
        d[tail, idx] = -1.0
    for k, (r, c) in enumerate(vert):
        e_idx = len(horiz) + k
        head = (r + 1) * GRID + c
        tail = r * GRID + c
        d[head, e_idx] = 1.0
        d[tail, e_idx] = -1.0
    return d


def _build_endpoint_index(d_matrix: torch.Tensor) -> torch.Tensor:
    """Return ``(2, E)`` long tensor: row 0 = head index, row 1 = tail index."""
    heads = (d_matrix == 1.0).float().argmax(dim=0)
    tails = (d_matrix == -1.0).float().argmax(dim=0)
    return torch.stack([heads, tails], dim=0)


def kirchhoff_resolve(
    source: torch.Tensor,
    conductance: torch.Tensor,
    d_matrix: torch.Tensor,
    shift: float,
) -> torch.Tensor:
    """Solve ``(D^T diag(c) D + shift I) u = source`` per batch.

    Args:
        source: ``(B, V, P)`` per-vertex source/sink term.
        conductance: ``(B, E)`` positive edge conductance.
        d_matrix: ``(V, E)`` vertex-edge incidence.
        shift: positive regulariser.

    Returns:
        ``(B, V, P)`` electrical potential.
    """
    if source.dim() != 3:
        raise ValueError(f"Expected (B, V, P) source, got {tuple(source.shape)}")
    batch, n_v, _ = source.shape
    c = conductance.clamp_min(1.0e-3)
    # Build (B, V, V) Laplacian: D diag(c) D^T  (since D shape is (V, E), D D^T is (V, V))
    weighted_d = c.unsqueeze(1) * d_matrix.unsqueeze(0)  # (B, V, E)
    laplacian = torch.einsum("bve,bue->bvu", weighted_d, d_matrix.expand(batch, n_v, -1))
    # Equivalent: torch.einsum("ve,bef,uf->bvu", d, diag(c), d) but we built (B, V, V) directly.
    reg = shift * torch.eye(n_v, device=source.device, dtype=source.dtype).unsqueeze(0)
    return torch.linalg.solve(laplacian + reg, source)


class KirchhoffMobilitySolve(nn.Module):
    """p045 — Kirchhoff Mobility Solve head over the i193 trunk."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        # Trunk hyper-parameters.
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        # KMS hyper-parameters.
        source_channels: int = 6,
        output_channels: int = 8,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        shift: float = 1.0e-2,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "KirchhoffMobilitySolve supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError("KirchhoffMobilitySolve requires the simple_18 board tensor")
        if int(source_channels) < 1:
            raise ValueError("source_channels must be >= 1")
        if int(output_channels) < 1:
            raise ValueError("output_channels must be >= 1")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )
        self.num_classes = 1
        self.ablation = str(ablation)
        self.source_channels = int(source_channels)
        self.output_channels = int(output_channels)
        self.shift = float(shift)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))

        self.trunk = ExchangeThenKingDualStreamNetwork(
            input_channels=int(input_channels),
            num_classes=1,
            channels=int(trunk_channels),
            hidden_dim=int(trunk_hidden_dim),
            depth=int(trunk_depth),
            dropout=float(trunk_dropout),
            use_batchnorm=bool(trunk_use_batchnorm),
            gate_dim=trunk_gate_dim,
            ablation=str(trunk_ablation),
        )

        d_matrix = _build_grid_incidence()
        endpoint_index = _build_endpoint_index(d_matrix)
        self.register_buffer("d_matrix", d_matrix, persistent=False)
        self.register_buffer("endpoint_index", endpoint_index, persistent=False)
        self.num_vertices = int(d_matrix.shape[0])
        self.num_edges = int(d_matrix.shape[1])

        per_square = 2 * self.trunk.channels
        self.source_head = nn.Linear(per_square, self.source_channels)
        self.conductance_head = nn.Sequential(
            nn.Linear(2 * per_square, max(8, per_square // 2)),
            nn.GELU(),
            nn.Linear(max(8, per_square // 2), 1),
        )
        self.output_proj = nn.Linear(self.source_channels, self.output_channels)

        self.feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )
        # Pool the per-square equilibrium with mean + max per channel.
        comp_dim = 2 * self.output_channels
        dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(comp_dim + self.feature_dim),
            nn.Linear(comp_dim + self.feature_dim, int(head_hidden_dim)),
            nn.GELU(),
            dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )
        gate_in = self.feature_dim + 2  # joint + (potential_norm, conductance_mean)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(gate_in),
            nn.Linear(gate_in, max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )
        with torch.no_grad():
            final = self.gate_head[-1]
            if isinstance(final, nn.Linear):
                final.bias.fill_(float(gate_init))

    def _spatial_features(self, board: torch.Tensor) -> torch.Tensor:
        feats = self.trunk.feature_builder(board)
        if self.trunk.ablation == "shared_stream_only":
            ex_input = board
            kg_input = board
        else:
            ex_input = torch.cat([board, feats.exchange], dim=1)
            kg_input = torch.cat([board, feats.king], dim=1)
        ex_h, _ = self.trunk.exchange_encoder(ex_input)
        if self.trunk.ablation == "shared_stream_only":
            kg_h = ex_h
        else:
            kg_h, _ = self.trunk.king_encoder(kg_input)
        return torch.cat([ex_h, kg_h], dim=1)  # (B, 2C, 8, 8)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        device = board.device
        dtype = board.dtype

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        spatial = self._spatial_features(board)
        per_square = spatial.flatten(2).transpose(1, 2)  # (B, 64, 2C)

        source = self.source_head(per_square)  # (B, 64, source_channels)
        if self.ablation == "zero_source":
            source = torch.zeros_like(source)

        heads = self.endpoint_index[0]
        tails = self.endpoint_index[1]
        edge_endpoint_feat = torch.cat(
            [per_square[:, heads], per_square[:, tails]], dim=-1
        )  # (B, E, 4C)
        if self.ablation == "uniform_conductance":
            conductance = torch.ones(batch, self.num_edges, device=device, dtype=dtype)
        else:
            conductance = F_.softplus(self.conductance_head(edge_endpoint_feat).squeeze(-1))
            if self.ablation == "shuffle_conductance" and self.num_edges > 1:
                perm = torch.randperm(self.num_edges, device=device)
                conductance = conductance[:, perm]

        if self.ablation == "diagonal_only":
            # u = source / shift  (drop the Laplacian term entirely)
            potential = source / max(self.shift, 1.0e-6)
        else:
            potential = kirchhoff_resolve(source, conductance, self.d_matrix.to(dtype=dtype), self.shift)

        projected = self.output_proj(potential)  # (B, 64, output_channels)
        pooled_mean = projected.mean(dim=1)
        pooled_max = projected.amax(dim=1)
        comp_feat = torch.cat([pooled_mean, pooled_max], dim=-1)

        delta_input = torch.cat([comp_feat, joint], dim=1)
        delta_raw = self.delta_head(delta_input).view(-1)

        potential_norm = potential.pow(2).mean(dim=(1, 2)).sqrt()
        conductance_mean = conductance.mean(dim=-1)
        gate_input = torch.cat(
            [joint, potential_norm.unsqueeze(-1), conductance_mean.unsqueeze(-1)],
            dim=1,
        )
        gate_logit = self.gate_head(gate_input).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation == "disable_gate":
            gate = torch.ones_like(gate)
        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
            gate_applied = torch.zeros_like(gate)
        else:
            primitive_delta = delta_raw
            gate_applied = gate
        contribution = gate_applied * primitive_delta
        logits = base_logit + contribution

        eps = 1.0e-6
        gate_clamped = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(
            gate_clamped * gate_clamped.log()
            + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
        )

        out: dict[str, torch.Tensor] = {}
        for key, value in trunk_out.items():
            if key in {"logits", "proposal_profile_strength", "proposal_keyword_count"}:
                continue
            out[f"trunk_{key}"] = value
        out["logits"] = logits
        out["base_logit"] = base_logit
        out["primitive_delta"] = primitive_delta
        out["primitive_delta_raw"] = delta_raw
        out["primitive_gate"] = gate
        out["primitive_gate_applied"] = gate_applied
        out["primitive_gate_logit"] = gate_logit
        out["primitive_gate_entropy"] = gate_entropy
        out["primitive_contribution"] = contribution
        out["kms_potential_norm"] = potential_norm
        out["kms_conductance_mean"] = conductance_mean
        out["kms_source_norm"] = source.pow(2).mean(dim=(1, 2)).sqrt()
        out["mechanism_energy"] = trunk_out["mechanism_energy"] + potential_norm.detach()
        out["proposal_profile_strength"] = primitive_delta.detach().abs() * gate_entropy
        out["proposal_keyword_count"] = logits.new_full(
            (batch,), float(self.num_vertices * self.output_channels)
        )
        return out


def build_kirchhoff_mobility_solve_from_config(
    config: dict[str, Any],
) -> KirchhoffMobilitySolve:
    cfg = dict(config)
    return KirchhoffMobilitySolve(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        source_channels=int(cfg.get("source_channels", 6)),
        output_channels=int(cfg.get("output_channels", 8)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        shift=float(cfg.get("shift", 1.0e-2)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "ALLOWED_ABLATIONS",
    "KirchhoffMobilitySolve",
    "build_kirchhoff_mobility_solve_from_config",
    "kirchhoff_resolve",
)
