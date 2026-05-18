"""Learned Relation Confidence Primitive (p047, LRC).

Source: ``ideas/research/primitives/external_42_learned_relation_confidence_primitive.md``.

The primitive sits between deterministic chess relation construction (the
i018 ``TacticalIncidenceBuilder``) and downstream consumption. It keeps the
12 typed tactical relation masks fixed and learns a per-edge confidence
weight that **only attenuates already-valid edges** -- if a relation mask
entry ``M[r, i, j]`` is zero, the learned confidence ``c[r, i, j]`` cannot
revive it. This preserves i018's central falsifiable thesis ("exact chess
relation topology matters") while upgrading its largely-binary relation
weighting to a board-conditioned weighting.

The operator is wired as an additive gated logit delta over the i193
``ExchangeThenKingDualStreamNetwork`` trunk so the baseline is recovered
exactly under ``zero_delta`` / ``trunk_only``:

    final_logit = base_logit + gate * primitive_delta_raw

Per-edge scoring uses a low-rank decomposition that avoids materialising
``(B, R, 64, 64, F)`` feature tensors:

    src_score[b, r, i] = w_src @ piece_onehot(i) + relation_src_bias[r]
    tgt_score[b, r, j] = w_tgt @ piece_onehot(j) + relation_tgt_bias[r]
    low_rank[b, r, i, j] = sum_k (q[b, i, k] * rel[r, k] * k[b, j, k])
    logit[b, r, i, j] = (src_score + tgt_score + low_rank + rel_bias[r])
                        * sigmoid(rel_gate[r])
    confidence[b, r, i, j] = sigmoid(logit / temperature)
    weighted_mask[b, r, i, j] = mask[b, r, i, j] * confidence[b, r, i, j]

Per-relation summaries (mean confidence on active edges, total mass,
entropy, kept-fraction at a soft threshold) are then concatenated with
the trunk joint pool and projected to a scalar delta. The gate MLP looks
at the trunk joint pool, the mean active-mask density, and the global
confidence norm.

CRTK metadata, source labels, verification flags, engine evaluations,
and principal variations are **not** consumed.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.trunk_features import trunk_joint_features
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor
from chess_nn_playground.models.trunk.oriented_tactical_sheaf import (
    BoardStateAdapter,
    TacticalIncidenceBuilder,
)


NUM_PIECE_CHANNELS = 12
SQUARES = 64
RELATION_COUNT = 12
NEG_INF = -1.0e9

# Names roughly aligned with TacticalIncidenceBuilder.relation_masks order:
# 0: our_attack_on_them, 1: their_attack_on_us, 2: our_defense,
# 3: their_defense, 4: our_king_zone_pressure,
# 5: their_king_zone_pressure, 6: bishop_ray,
# 7: rook_ray, 8: queen_ray, 9: knight, 10: pawn, 11: pin.
RELATION_NAMES: tuple[str, ...] = (
    "our_attack_on_them",
    "their_attack_on_us",
    "our_defense",
    "their_defense",
    "our_king_zone",
    "their_king_zone",
    "bishop_ray",
    "rook_ray",
    "queen_ray",
    "knight",
    "pawn",
    "pin",
)


ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "binary_only",        # primary falsifier — skip confidence; weighted_mask := mask
    "scrambled_mask",     # in-batch permute deterministic masks (kills topology semantics)
    "shuffle_pieces",     # in-batch permute the piece descriptor (kills feature semantics)
    "gate_only",          # only per-relation gate is learned; no per-edge scoring
    "no_low_rank",        # drop the low-rank bilinear term; keep edge MLP only
    "no_edge_mlp",        # drop the edge MLP; keep low-rank only
    "zero_delta",
    "trunk_only",
    "disable_gate",
)


def _piece_descriptor_from_board(board: torch.Tensor) -> torch.Tensor:
    """Return ``(B, 64, 13)`` per-square piece descriptor (empty + 12 piece planes)."""
    piece_planes = board[:, :NUM_PIECE_CHANNELS].clamp(0.0, 1.0)
    per_square = piece_planes.flatten(2).transpose(1, 2).contiguous()  # (B, 64, 12)
    occupancy = per_square.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
    empty = (1.0 - occupancy).clamp(0.0, 1.0)
    return torch.cat([empty, per_square], dim=-1)  # (B, 64, 13)


def _confidence_summary(
    weighted_mask: torch.Tensor,
    mask: torch.Tensor,
    confidence: torch.Tensor,
    *,
    threshold: float = 0.5,
    temperature: float = 0.1,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Per-(batch, relation) summary tensor of shape ``(B, R, 4)``.

    Channels: mean confidence on active edges, normalised mass, soft kept
    fraction, binary entropy. The ``soft kept fraction`` uses a smooth
    sigmoid around ``threshold`` so the summary is fully differentiable.
    """
    mask_sum = mask.sum(dim=(2, 3)).clamp_min(1.0)
    weighted_sum = weighted_mask.sum(dim=(2, 3))
    mean_conf = weighted_sum / mask_sum
    mass = weighted_sum / float(SQUARES * SQUARES)
    soft_keep = torch.sigmoid((confidence - threshold) / temperature) * mask
    kept_fraction = soft_keep.sum(dim=(2, 3)) / mask_sum
    p = confidence.clamp(eps, 1.0 - eps)
    entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log()) * mask
    entropy_per_rel = entropy.sum(dim=(2, 3)) / mask_sum
    return torch.stack([mean_conf, mass, kept_fraction, entropy_per_rel], dim=-1)


class LearnedRelationConfidence(nn.Module):
    """p047 -- Learned Relation Confidence head over the i193 trunk."""

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
        # LRC hyper-parameters.
        token_dim: int = 32,
        low_rank_dim: int = 8,
        edge_hidden: int = 32,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        confidence_temperature: float = 1.0,
        confidence_bias_init: float = 2.0,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "LearnedRelationConfidence supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "LearnedRelationConfidence requires the simple_18 board tensor"
            )
        if int(token_dim) < 4:
            raise ValueError("token_dim must be >= 4")
        if int(low_rank_dim) < 1:
            raise ValueError("low_rank_dim must be >= 1")
        if float(confidence_temperature) <= 0.0:
            raise ValueError("confidence_temperature must be positive")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )

        self.num_classes = 1
        self.input_channels = int(input_channels)
        self.token_dim = int(token_dim)
        self.low_rank_dim = int(low_rank_dim)
        self.edge_hidden = int(edge_hidden)
        self.confidence_temperature = float(confidence_temperature)
        self.confidence_bias_init = float(confidence_bias_init)
        self.ablation = str(ablation)
        self.relation_count = RELATION_COUNT
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

        # Deterministic chess-relation builders. Frozen — we treat the
        # 12 typed relation masks as a fixed signal and only learn the
        # per-edge confidence weighting on top of them.
        self.board_adapter = BoardStateAdapter(
            input_channels=int(input_channels), encoding="simple_18"
        )
        self.relation_builder = TacticalIncidenceBuilder()
        for parameter in self.board_adapter.parameters():
            parameter.requires_grad_(False)
        for parameter in self.relation_builder.parameters():
            parameter.requires_grad_(False)

        # Square token tower. We use a cheap 1x1 conv tower (the same
        # contract as the existing ``SquareTokenEmbedder``) so the
        # primitive can be ported to non-i193 trunks without dragging in
        # the i018 ``SquareTokenEncoder``.
        self.token_proj = nn.Sequential(
            nn.Conv2d(int(input_channels), self.token_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.token_dim, self.token_dim, kernel_size=1, bias=True),
        )
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.q_proj = nn.Linear(self.token_dim, self.relation_count * self.low_rank_dim)
        self.k_proj = nn.Linear(self.token_dim, self.relation_count * self.low_rank_dim)

        # Low-rank relation embedding folded into the bilinear term.
        self.relation_low_rank = nn.Parameter(
            torch.randn(self.relation_count, self.low_rank_dim) * 0.05
        )

        # Per-relation edge MLP — applied separately to source and target
        # piece descriptors so the per-edge bias factorises as
        # ``edge_bias[r, i, j] = src_score[r, i] + tgt_score[r, j]``.
        self.edge_src_mlp = nn.Sequential(
            nn.Linear(NUM_PIECE_CHANNELS + 1, int(edge_hidden)),
            nn.GELU(),
            nn.Linear(int(edge_hidden), self.relation_count),
        )
        self.edge_tgt_mlp = nn.Sequential(
            nn.Linear(NUM_PIECE_CHANNELS + 1, int(edge_hidden)),
            nn.GELU(),
            nn.Linear(int(edge_hidden), self.relation_count),
        )

        # Per-relation scalars: a global gate and an additive bias. The
        # bias is initialised positive so untrained confidences land near
        # ~sigmoid(2) on active edges; the gate is initialised at zero
        # so the multiplier starts at sigmoid(0) = 0.5.
        self.relation_gate_logits = nn.Parameter(torch.zeros(self.relation_count))
        self.relation_bias = nn.Parameter(
            torch.full((self.relation_count,), float(confidence_bias_init))
        )

        # Summary -> delta head.
        self.feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )
        summary_dim = self.relation_count * 4
        dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.summary_norm = nn.LayerNorm(summary_dim)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(summary_dim + self.feature_dim),
            nn.Linear(summary_dim + self.feature_dim, int(head_hidden_dim)),
            nn.GELU(),
            dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )

        # Gate over the joint pool plus two scalar summaries (mean
        # confidence and mean active-edge density).
        gate_in = self.feature_dim + 2
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

    def _square_tokens(self, board: torch.Tensor) -> torch.Tensor:
        feat = self.token_proj(board)  # (B, token_dim, 8, 8)
        tokens = feat.flatten(2).transpose(1, 2).contiguous()  # (B, 64, token_dim)
        return self.token_norm(tokens)

    def _relation_masks(self, board: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            state = self.board_adapter(board)
            incidence = self.relation_builder(state.piece_state, state.occupancy)
        return incidence.relation_masks.detach()

    def _score_edges(
        self,
        board: torch.Tensor,
        tokens: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-edge confidence logits ``(B, R, 64, 64)`` before masking.

        Composition:
          edge_logit = (src_score + tgt_score + low_rank + relation_bias)
                       * sigmoid(relation_gate)
        """
        batch = board.shape[0]
        gates = torch.sigmoid(self.relation_gate_logits)

        if self.ablation == "gate_only":
            # Output a constant per-relation logit. Confidence then becomes
            # ``sigmoid(gate * bias)`` for every active edge.
            base = (
                self.relation_bias * gates
            )  # (R,)
            return base.view(1, self.relation_count, 1, 1).expand(
                batch, self.relation_count, SQUARES, SQUARES
            )

        if self.ablation == "no_edge_mlp":
            src_score = edge_features.new_zeros(batch, self.relation_count, SQUARES)
            tgt_score = edge_features.new_zeros(batch, self.relation_count, SQUARES)
        else:
            # edge_features: (B, 64, 13) piece descriptor.
            src_score = self.edge_src_mlp(edge_features).transpose(1, 2)  # (B, R, 64)
            tgt_score = self.edge_tgt_mlp(edge_features).transpose(1, 2)  # (B, R, 64)
        edge_bias = src_score.unsqueeze(-1) + tgt_score.unsqueeze(-2)
        # (B, R, 64, 64)

        if self.ablation == "no_low_rank":
            low_rank = edge_features.new_zeros(batch, self.relation_count, SQUARES, SQUARES)
        else:
            q = self.q_proj(tokens).view(
                batch, SQUARES, self.relation_count, self.low_rank_dim
            )
            k = self.k_proj(tokens).view(
                batch, SQUARES, self.relation_count, self.low_rank_dim
            )
            # Inject relation embedding multiplicatively into q.
            q = q * self.relation_low_rank.view(1, 1, self.relation_count, self.low_rank_dim)
            # einsum is dense: (B, R, i, k) x (B, R, j, k) -> (B, R, i, j).
            q_r = q.permute(0, 2, 1, 3).contiguous()  # (B, R, 64, low_rank)
            k_r = k.permute(0, 2, 1, 3).contiguous()  # (B, R, 64, low_rank)
            low_rank = torch.einsum("brik,brjk->brij", q_r, k_r)

        rel_bias = self.relation_bias.view(1, self.relation_count, 1, 1)
        rel_gate = gates.view(1, self.relation_count, 1, 1)
        logits = (edge_bias + low_rank + rel_bias) * rel_gate
        return logits

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        masks = self._relation_masks(board)
        if self.ablation == "scrambled_mask" and batch > 1:
            perm = torch.randperm(batch, device=masks.device)
            masks = masks[perm]

        piece_desc = _piece_descriptor_from_board(board)
        if self.ablation == "shuffle_pieces" and batch > 1:
            perm = torch.randperm(batch, device=piece_desc.device)
            piece_desc = piece_desc[perm]
        tokens = self._square_tokens(board)

        if self.ablation == "binary_only":
            confidence = masks.detach()
            weighted_mask = masks
            edge_logits = torch.zeros_like(masks)
        else:
            edge_logits = self._score_edges(board, tokens, piece_desc)
            # Confidence on active edges; for inactive edges we return 0
            # because the downstream summary multiplies by the mask anyway.
            scaled = edge_logits / self.confidence_temperature
            confidence_active = torch.sigmoid(scaled) * masks
            # Topology-preserving by construction: if mask == 0, the
            # entry is exactly 0.
            confidence = confidence_active
            weighted_mask = confidence

        summary = _confidence_summary(weighted_mask, masks, confidence)
        # Per-relation aggregates exposed in the diagnostics dict.
        mean_conf_per_rel = summary[..., 0]
        mass_per_rel = summary[..., 1]
        kept_per_rel = summary[..., 2]
        entropy_per_rel = summary[..., 3]
        summary_flat = self.summary_norm(summary.reshape(batch, -1))

        delta_input = torch.cat([summary_flat, joint], dim=1)
        delta_raw = self.delta_head(delta_input).view(-1)

        global_mean_conf = mean_conf_per_rel.mean(dim=1)
        mask_density = masks.float().sum(dim=(1, 2, 3)) / (
            self.relation_count * SQUARES * SQUARES
        )
        gate_input = torch.cat(
            [joint, global_mean_conf.unsqueeze(-1), mask_density.unsqueeze(-1)],
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
        out["lrc_global_mean_confidence"] = global_mean_conf
        out["lrc_mask_density"] = mask_density
        for r_idx, r_name in enumerate(RELATION_NAMES):
            out[f"lrc_mean_conf_{r_name}"] = mean_conf_per_rel[:, r_idx]
            out[f"lrc_mass_{r_name}"] = mass_per_rel[:, r_idx]
            out[f"lrc_kept_{r_name}"] = kept_per_rel[:, r_idx]
            out[f"lrc_entropy_{r_name}"] = entropy_per_rel[:, r_idx]
        out["mechanism_energy"] = (
            trunk_out["mechanism_energy"] + global_mean_conf.detach()
        )
        out["proposal_profile_strength"] = primitive_delta.detach().abs() * gate_entropy
        out["proposal_keyword_count"] = logits.new_full(
            (batch,), float(self.relation_count * 4)
        )
        return out


def build_learned_relation_confidence_from_config(
    config: dict[str, Any],
) -> LearnedRelationConfidence:
    cfg = dict(config)
    return LearnedRelationConfidence(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(
            cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))
        ),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        token_dim=int(cfg.get("token_dim", 32)),
        low_rank_dim=int(cfg.get("low_rank_dim", 8)),
        edge_hidden=int(cfg.get("edge_hidden", 32)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        confidence_temperature=float(cfg.get("confidence_temperature", 1.0)),
        confidence_bias_init=float(cfg.get("confidence_bias_init", 2.0)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "ALLOWED_ABLATIONS",
    "LearnedRelationConfidence",
    "RELATION_COUNT",
    "RELATION_NAMES",
    "build_learned_relation_confidence_from_config",
)
