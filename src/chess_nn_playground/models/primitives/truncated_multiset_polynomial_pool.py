"""Truncated Multiset Polynomial Pool (p042, TMPP) primitive.

Source: ``ideas/research/primitives/external_37_truncated_multiset_polynomial_rook_matching_primitives.md``
(Section 1, ``primitive_truncated_multiset_polynomial_pool``; the file's
top-ranked proposal). Deferred internal proposals from the same file:

- ``primitive_entropic_rook_matching_contraction`` (rook polynomial DP) --
  closely related to p043 grassmann rook pool but with entropic softmax
  partition; covered by the rook-matching family in p043.
- ``primitive_laplacian_forest_connectivity`` -- Laplacian resolvent over
  fixed structural graph; covered by the resolvent family in p045.
- ``primitive_weighted_hodge_flow_split`` -- covered in p044.
- ``primitive_chess_group_irrep_norm`` -- irrep-projector normalisation;
  related orbit/irrep family not in scope for this batch.

The operator computes the truncated elementary-symmetric polynomial
coefficients of a learned per-token latent over the set of own pieces.
For per-token latent ``u_{b,i,c}`` and indicator mask ``m_{b,i}``,

    P_{b,c}(z) = prod_i (1 + m_{b,i} u_{b,i,c} z)

with truncated coefficients ``y_{b,k,c} = [z^k] P_{b,c}(z)`` for
``k = 1, ..., K``. The recurrence

    e_0 = 1,
    e_k <- e_k + u_i * e_{k-1}     (descending k = K, ..., 1)

is the fused triangular polynomial scan: it is *not* DeepSets sum pooling
(``K=1`` ablation reduces it to that), and it is *not* attention. The
multiplicative state encodes unordered low-order interaction structure
without enumerating tuples.

The primitive is wired as an additive gated delta head over the i193
ExchangeThenKingDualStreamNetwork trunk so the i193 baseline is exactly
recovered under ``zero_delta`` / ``trunk_only``:

    final_logit = base_logit + gate * primitive_delta_raw

CRTK metadata, source labels, verification flags, engine evaluations and
principal variations are *not* consumed.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor
from chess_nn_playground.models.primitives.trunk_features import trunk_joint_features


NUM_PIECE_CHANNELS = 12
SQUARES = 64

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "first_order_only",      # K := 1 — primary falsifier; collapses to DeepSets sum pool
    "shuffle_tokens",        # in-batch permutation of token features (invariance check)
    "shuffle_mask",          # in-batch permutation of the piece mask (rule-feature falsifier)
    "uniform_mask",          # mask := all-ones (ignore piece occupancy)
    "zero_delta",
    "trunk_only",
    "disable_gate",
)


def truncated_elementary_symmetric_scan(
    u: torch.Tensor,
    mask: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """Compute the truncated elementary-symmetric coefficient stack.

    Args:
        u: ``(B, N, C)`` per-token latent.
        mask: ``(B, N)`` binary indicator (1.0 for active token, 0.0 to skip).
        degree: ``K`` truncation degree (>= 1).

    Returns:
        ``(B, K, C)`` stack with ``e_k = [z^k] prod_i (1 + m_i u_i z)``.
    """
    if u.dim() != 3:
        raise ValueError(f"Expected (B, N, C) latent, got {tuple(u.shape)}")
    if mask.dim() != 2 or mask.shape[:2] != u.shape[:2]:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} does not match u shape {tuple(u.shape)}"
        )
    if int(degree) < 1:
        raise ValueError("degree must be >= 1")

    batch, num_tokens, channels = u.shape
    device = u.device
    dtype = u.dtype
    weighted = u * mask.to(dtype=dtype).unsqueeze(-1)

    e0 = torch.ones(batch, channels, device=device, dtype=dtype)
    # We carry e_1, ..., e_K. e0 is implicit (always 1) and not returned.
    e = [torch.zeros(batch, channels, device=device, dtype=dtype) for _ in range(int(degree))]
    for i in range(num_tokens):
        u_i = weighted[:, i, :]  # (B, C)
        for k in range(int(degree), 0, -1):
            if k == 1:
                e[0] = e[0] + u_i * e0
            else:
                e[k - 1] = e[k - 1] + u_i * e[k - 2]
    return torch.stack(e, dim=1)  # (B, K, C)


class TruncatedMultisetPolynomialPool(nn.Module):
    """p042 — Truncated Multiset Polynomial Pool over the i193 trunk."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        # Trunk hyper-parameters mirror the i193 builder.
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        # TMPP head hyper-parameters.
        latent_dim: int = 24,
        degree: int = 3,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        coeff_norm: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "TruncatedMultisetPolynomialPool supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "TruncatedMultisetPolynomialPool requires the simple_18 board tensor"
            )
        if int(degree) < 1 or int(degree) > 6:
            raise ValueError("degree must be in [1, 6]; recommended K<=4 for numerical stability")
        if int(latent_dim) < 2:
            raise ValueError("latent_dim must be >= 2")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )
        self.num_classes = 1
        self.ablation = str(ablation)
        self.degree = int(degree)
        self.latent_dim = int(latent_dim)
        self.coeff_norm = bool(coeff_norm)
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

        # Per-square token features come from the simple_18 piece planes plus
        # side-to-move plane. We tanh-activate to keep coefficients bounded.
        # This is the spec's ``u_i = tanh(W x_i + b)`` per-token latent map.
        self.token_proj = nn.Sequential(
            nn.Linear(NUM_PIECE_CHANNELS + 1, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )

        coeff_dim = self.degree * self.latent_dim
        self.coeff_norm_module: nn.Module = (
            nn.LayerNorm(coeff_dim) if self.coeff_norm else nn.Identity()
        )
        dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )

        self.delta_head = nn.Sequential(
            nn.LayerNorm(coeff_dim + self.feature_dim),
            nn.Linear(coeff_dim + self.feature_dim, int(head_hidden_dim)),
            nn.GELU(),
            dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )
        gate_in = self.feature_dim + 2  # joint + (mean_active, coeff_norm)
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

    def _piece_descriptor(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build per-square (latent, mask) descriptors from the simple_18 board.

        - latent: piece-plane indicator concatenated with side-to-move scalar,
          projected through ``token_proj`` to ``latent_dim`` (tanh-bounded).
        - mask: 1.0 on any square that contains a piece, 0.0 otherwise.
        """
        batch = board.shape[0]
        piece_planes = board[:, :NUM_PIECE_CHANNELS].clamp(0.0, 1.0)
        stm_scalar = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        per_square = piece_planes.flatten(2).transpose(1, 2).contiguous()  # (B, 64, 12)
        stm_expand = stm_scalar.view(batch, 1, 1).expand(batch, SQUARES, 1)
        descriptor = torch.cat([per_square, stm_expand], dim=-1)
        latent = torch.tanh(self.token_proj(descriptor))
        occupancy = per_square.sum(dim=-1).clamp(0.0, 1.0)  # (B, 64) 1.0 on any piece
        return latent, occupancy

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        device = board.device
        dtype = board.dtype

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        latent, occupancy = self._piece_descriptor(board)

        if self.ablation == "shuffle_tokens" and batch > 0 and latent.shape[1] > 1:
            # Apply the same permutation to latent and mask: the scan's
            # coefficient values are exactly invariant under this, so the
            # ablation acts as a regression-safe sanity check rather than a
            # rule-feature falsifier.
            perm = torch.randperm(latent.shape[1], device=device)
            latent = latent[:, perm, :]
            occupancy = occupancy[:, perm]
        if self.ablation == "shuffle_mask" and batch > 1:
            perm = torch.randperm(batch, device=device)
            occupancy = occupancy[perm]
        if self.ablation == "uniform_mask":
            occupancy = torch.ones_like(occupancy)

        effective_degree = 1 if self.ablation == "first_order_only" else self.degree
        coeff = truncated_elementary_symmetric_scan(latent, occupancy, effective_degree)
        # (B, K_eff, C). Pad up to self.degree if ablation shrunk K so the head shape stays fixed.
        if effective_degree < self.degree:
            pad = coeff.new_zeros(batch, self.degree - effective_degree, self.latent_dim)
            coeff = torch.cat([coeff, pad], dim=1)
        # Optional binomial-style normalisation: divide e_k by max(1, sum_i mask_i choose k)
        if self.coeff_norm:
            mask_sum = occupancy.sum(dim=1).clamp_min(1.0)  # (B,)
            denom = mask_sum.new_ones(batch, self.degree)
            scale = mask_sum.clone()
            denom[:, 0] = scale.clamp_min(1.0)
            for k in range(2, self.degree + 1):
                scale = scale * (mask_sum - (k - 1)).clamp_min(1.0) / float(k)
                denom[:, k - 1] = scale.clamp_min(1.0)
            coeff = coeff / denom.unsqueeze(-1)
        coeff_flat = coeff.reshape(batch, -1)
        coeff_flat = self.coeff_norm_module(coeff_flat)

        delta_input = torch.cat([coeff_flat, joint], dim=1)
        delta_raw = self.delta_head(delta_input).view(-1)

        coeff_norm_sample = coeff.pow(2).mean(dim=(1, 2)).sqrt()
        active_mean = occupancy.mean(dim=1)
        gate_input = torch.cat(
            [joint, active_mean.unsqueeze(-1), coeff_norm_sample.unsqueeze(-1)],
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
        out["tmpp_coeff_norm"] = coeff_norm_sample
        out["tmpp_active_mean"] = active_mean
        out["tmpp_coeff_e1"] = coeff[:, 0].pow(2).mean(dim=-1).sqrt()
        if self.degree >= 2:
            out["tmpp_coeff_e2"] = coeff[:, 1].pow(2).mean(dim=-1).sqrt()
        if self.degree >= 3:
            out["tmpp_coeff_e3"] = coeff[:, 2].pow(2).mean(dim=-1).sqrt()
        out["mechanism_energy"] = trunk_out["mechanism_energy"] + coeff_norm_sample.detach()
        out["proposal_profile_strength"] = primitive_delta.detach().abs() * gate_entropy
        out["proposal_keyword_count"] = logits.new_full((batch,), float(self.degree * self.latent_dim))
        return out


def build_truncated_multiset_polynomial_pool_from_config(
    config: dict[str, Any],
) -> TruncatedMultisetPolynomialPool:
    cfg = dict(config)
    return TruncatedMultisetPolynomialPool(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        latent_dim=int(cfg.get("latent_dim", 24)),
        degree=int(cfg.get("degree", 3)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        coeff_norm=bool(cfg.get("coeff_norm", True)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "ALLOWED_ABLATIONS",
    "TruncatedMultisetPolynomialPool",
    "build_truncated_multiset_polynomial_pool_from_config",
    "truncated_elementary_symmetric_scan",
)
