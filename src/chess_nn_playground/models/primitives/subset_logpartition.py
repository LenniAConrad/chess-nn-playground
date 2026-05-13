"""Bounded Subset Log-Partition Transform (p046, SLPT) primitive.

Source: ``ideas/research/primitives/external_41_orbit_stabilizer_subset_logpartition_primitives.md``
(Section 2 ``primitive_subset_logpartition``; promoted from #2 because the
file's #1 ``primitive_orbit_stabilizer_canonical`` is in the orbit/irrep
family deferred to a future symmetry batch — overlapping with the
unimplemented #1 of external_39). Deferred internal proposals:

- ``primitive_orbit_stabilizer_canonical`` -- symmetry quotient family;
  deferred.
- ``primitive_exterior_wedge_pool`` -- antisymmetric k-blade pool; related
  to but distinct from p042 polynomial pool (multiplicative, alternating);
  deferred to a future "wedge / Pfaffian" batch.
- ``primitive_lovasz_chain_pool`` -- submodular Lovasz extension; pooling
  family related to Choquet integral, deferred.
- ``primitive_pfaffian_conflict_pool`` -- Pfaffian over signed pairings;
  flagged as low scout-scale demonstrability by the source.

The operator computes the log-domain truncated elementary-symmetric
polynomial of a per-token log-weight tensor ``A in R^{B x n x r}``.
Starting from the boundary

    C^{(0)}[0, c] = 0, C^{(0)}[k > 0, c] = -inf,

the recurrence

    C^{(i)}[k, c] = logaddexp(C^{(i-1)}[k, c], C^{(i-1)}[k - 1, c] + A[i, c])

yields ``Y[k, c] = log sum_{|S|=k} exp sum_{i in S} A[i, c]``, i.e. the
log partition over size-``k`` subsets. The gradient
``dY[k]/dA[i] = Pr(i in S | |S| = k)`` is the marginal probability of
token ``i`` belonging to a size-``k`` subset, computed via the standard
forward/backward DP over the truncated polynomial ring in the log-
semiring. ``torch.logaddexp`` handles the recurrence directly so the
backward is autograd-provided.

The primitive is wired as an additive gated logit delta over the i193
ExchangeThenKingDualStreamNetwork trunk:

    final_logit = base_logit + gate * primitive_delta_raw

CRTK metadata, source labels, verification flags, engine evaluations
and principal variations are *not* consumed.
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
NEG_INF = -1.0e9

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "k1_only",                 # primary falsifier — K := 1; collapse to logsumexp pool
    "uniform_mask",            # mask all squares (ignore piece occupancy)
    "shuffle_mask",            # in-batch permutation of the piece mask
    "shuffle_tokens",          # permute square order (log-DP is invariant)
    "zero_delta",
    "trunk_only",
    "disable_gate",
)


def subset_logpartition_scan(
    log_weights: torch.Tensor,
    mask: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """Compute the log-domain truncated elementary-symmetric polynomial.

    For ``A[i, c]`` per-token log-weights and ``m[i] in {0, 1}``,

        Y[k, c] = log sum_{|S|=k, S subset of {i : m_i = 1}} exp sum_{i in S} A[i, c].

    Args:
        log_weights: ``(B, N, C)`` per-token log-weight tensor.
        mask: ``(B, N)`` (0/1) indicator of active tokens.
        degree: ``K`` truncation (>= 1).

    Returns:
        ``(B, K, C)`` log-partition values.
    """
    if log_weights.dim() != 3:
        raise ValueError(f"Expected (B, N, C) log_weights, got {tuple(log_weights.shape)}")
    if mask.dim() != 2 or mask.shape[:2] != log_weights.shape[:2]:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} does not match log_weights shape {tuple(log_weights.shape)}"
        )
    if int(degree) < 1:
        raise ValueError("degree must be >= 1")

    batch, num_tokens, channels = log_weights.shape
    device = log_weights.device
    dtype = log_weights.dtype
    # Force inactive tokens to log(0) = -inf so they cannot be selected.
    log_mask = torch.where(
        mask.to(dtype=dtype) > 0.5,
        torch.zeros_like(mask, dtype=dtype),
        torch.full_like(mask, NEG_INF, dtype=dtype),
    )
    weighted = log_weights + log_mask.unsqueeze(-1)

    # c_states[k] holds C^{(i)}[k+1, :] for k = 0, ..., K-1; the "size 0" state
    # is constant 0 and we track it implicitly.
    c_states = [torch.full((batch, channels), NEG_INF, device=device, dtype=dtype) for _ in range(int(degree))]
    c_prev_zero = torch.zeros(batch, channels, device=device, dtype=dtype)
    for i in range(num_tokens):
        a_i = weighted[:, i, :]  # (B, C)
        for k in range(int(degree), 0, -1):
            if k == 1:
                new_val = torch.logaddexp(c_states[0], c_prev_zero + a_i)
            else:
                new_val = torch.logaddexp(c_states[k - 1], c_states[k - 2] + a_i)
            c_states[k - 1] = new_val
    return torch.stack(c_states, dim=1)


class SubsetLogPartition(nn.Module):
    """p046 — Bounded Subset Log-Partition head over the i193 trunk."""

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
        # SLPT hyper-parameters.
        log_weight_dim: int = 32,
        degree: int = 3,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "SubsetLogPartition supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError("SubsetLogPartition requires the simple_18 board tensor")
        if int(degree) < 1 or int(degree) > 5:
            raise ValueError("degree must be in [1, 5]; recommended K<=4 for stability")
        if int(log_weight_dim) < 1:
            raise ValueError("log_weight_dim must be >= 1")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )
        self.num_classes = 1
        self.ablation = str(ablation)
        self.degree = int(degree)
        self.log_weight_dim = int(log_weight_dim)
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

        # Per-square log-weight head: piece-presence + side-to-move descriptor.
        # We do NOT activate with tanh because log-weights can be any real
        # number; instead we LayerNorm to bound their scale, then divide by
        # the configured ``log_weight_temperature`` (=1.0 by default) which
        # is folded into the Linear weights.
        self.log_weight_proj = nn.Sequential(
            nn.Linear(NUM_PIECE_CHANNELS + 1, self.log_weight_dim),
            nn.LayerNorm(self.log_weight_dim),
        )

        log_dim = self.degree * self.log_weight_dim
        self.coeff_norm = nn.LayerNorm(log_dim)
        self.feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )
        dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(log_dim + self.feature_dim),
            nn.Linear(log_dim + self.feature_dim, int(head_hidden_dim)),
            nn.GELU(),
            dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )
        gate_in = self.feature_dim + 2  # joint + (active_mean, log_partition_norm)
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
        """Build per-square (log-weight, mask) descriptors.

        Returns:
            log_weights: ``(B, 64, log_weight_dim)``.
            mask: ``(B, 64)`` occupancy indicator (1 on any piece-bearing square).
        """
        batch = board.shape[0]
        piece_planes = board[:, :NUM_PIECE_CHANNELS].clamp(0.0, 1.0)
        stm_scalar = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        per_square = piece_planes.flatten(2).transpose(1, 2).contiguous()  # (B, 64, 12)
        stm_expand = stm_scalar.view(batch, 1, 1).expand(batch, SQUARES, 1)
        descriptor = torch.cat([per_square, stm_expand], dim=-1)
        log_weights = self.log_weight_proj(descriptor)
        occupancy = per_square.sum(dim=-1).clamp(0.0, 1.0)
        return log_weights, occupancy

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        device = board.device
        dtype = board.dtype

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        log_weights, occupancy = self._piece_descriptor(board)

        if self.ablation == "shuffle_tokens" and log_weights.shape[1] > 1:
            perm = torch.randperm(log_weights.shape[1], device=device)
            log_weights = log_weights[:, perm, :]
            occupancy = occupancy[:, perm]
        if self.ablation == "shuffle_mask" and batch > 1:
            perm = torch.randperm(batch, device=device)
            occupancy = occupancy[perm]
        if self.ablation == "uniform_mask":
            occupancy = torch.ones_like(occupancy)

        effective_degree = 1 if self.ablation == "k1_only" else self.degree
        log_partition = subset_logpartition_scan(log_weights, occupancy, effective_degree)
        # (B, K_eff, C). Pad up to self.degree if ablation shrunk K.
        if effective_degree < self.degree:
            pad = log_partition.new_full(
                (batch, self.degree - effective_degree, self.log_weight_dim),
                NEG_INF,
            )
            log_partition = torch.cat([log_partition, pad], dim=1)
        # Center the log-partition values by subtracting their per-channel
        # mean over k so the delta head sees bounded inputs.
        log_part_safe = torch.where(
            torch.isfinite(log_partition),
            log_partition,
            torch.full_like(log_partition, NEG_INF),
        )
        # Clip extreme negatives for the head input (this only affects the
        # *visualisation* path; the scan itself runs in log-domain).
        head_input = log_part_safe.clamp(min=-30.0, max=30.0)
        coeff_flat = head_input.reshape(batch, -1)
        coeff_flat = self.coeff_norm(coeff_flat)

        delta_input = torch.cat([coeff_flat, joint], dim=1)
        delta_raw = self.delta_head(delta_input).view(-1)

        coeff_norm_sample = head_input.pow(2).mean(dim=(1, 2)).sqrt()
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
        out["slpt_active_mean"] = active_mean
        out["slpt_logpartition_norm"] = coeff_norm_sample
        out["slpt_y1"] = head_input[:, 0].mean(dim=-1)
        if self.degree >= 2:
            out["slpt_y2"] = head_input[:, 1].mean(dim=-1)
        if self.degree >= 3:
            out["slpt_y3"] = head_input[:, 2].mean(dim=-1)
        out["mechanism_energy"] = trunk_out["mechanism_energy"] + coeff_norm_sample.detach()
        out["proposal_profile_strength"] = primitive_delta.detach().abs() * gate_entropy
        out["proposal_keyword_count"] = logits.new_full((batch,), float(self.degree * self.log_weight_dim))
        return out


def build_subset_logpartition_from_config(config: dict[str, Any]) -> SubsetLogPartition:
    cfg = dict(config)
    return SubsetLogPartition(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        log_weight_dim=int(cfg.get("log_weight_dim", 32)),
        degree=int(cfg.get("degree", 3)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "ALLOWED_ABLATIONS",
    "SubsetLogPartition",
    "build_subset_logpartition_from_config",
    "subset_logpartition_scan",
)
