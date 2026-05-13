"""Gibbs Cut Log-Partition Operator (p037).

Source: ``ideas/research/primitives/external_32_elementary_symmetric_gibbs_hodge_primitives.md``
(rank-2 proposal ``primitive_gibbs_cut_partition``; the rank-1 proposal
``primitive_elem_sym_event`` is the same elementary-symmetric-polynomial
operator already covered by ``p024 event_symmetric_interaction_accumulator``,
so it is skipped here as a duplicate).

The primitive evaluates a Gibbs log-partition over all cuts of a fixed
``H x W`` grid

    Z = sum_{S subseteq V} exp(-(cut_cost(S) + source/sink_cost(S))/tau),

returning ``y = -tau * log Z`` and exposing the cut-edge marginals
``m_e = dy/dc_e`` through autograd. The summation is reduced from
``2^(H*W)`` to ``O(H * 2^(2W))`` by a row transfer-matrix dynamic
program in log space (per channel, fused over a small batch).

The cut grid lives in *latent* space -- the trunk joint feature is
projected to ``(H, W, d_cut)`` -- and is intentionally smaller than the
8x8 board (default ``H = W = 4``) so the transition matrix is only
``16 x 16`` per row. CRTK metadata, source labels, verification flags,
engine evaluations, and report-only metadata are not used.

Deferred internal proposals from the same packet:

- ``primitive_elem_sym_event`` (rank 1): duplicate of p024.
- ``primitive_complementarity_contact`` (rank 3): differentiable LCP /
  Fischer-Burmeister contact solver; deferred (overlaps with the existing
  ``dykstra_lcp`` family).
- ``primitive_hodge_cochain_projector`` (rank 4): edge-cochain Hodge
  decomposition; deferred.
- ``primitive_signed_persistence`` (rank 5): differentiable persistence
  pool; deferred.
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


ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "uniform_edges",        # replace cut edge costs with a constant tensor
    "uniform_sources",      # replace source/sink penalties with a constant tensor
    "shuffle_logpartition", # in-batch permutation of the log-partition pool
    "zero_delta",
    "trunk_only",
)


def _build_state_bits(width: int) -> torch.Tensor:
    """Return ``(2^W, W)`` float tensor with bit decomposition of 0..2^W-1."""
    n_states = 1 << int(width)
    bits = torch.zeros(n_states, int(width), dtype=torch.float32)
    for k in range(n_states):
        for j in range(int(width)):
            bits[k, j] = float((k >> j) & 1)
    return bits


def _build_within_row_xor(width: int) -> torch.Tensor:
    """``(2^W, W-1)`` float tensor where xor[S, j] = bits[S,j] XOR bits[S,j+1]."""
    bits = _build_state_bits(width)
    return (bits[:, :-1] - bits[:, 1:]).abs()  # (2^W, W-1)


def _build_between_row_xor(width: int) -> torch.Tensor:
    """``(2^W, 2^W, W)`` float tensor with vertical-edge XOR indicators."""
    bits = _build_state_bits(width)  # (S, W)
    return (bits.unsqueeze(1) - bits.unsqueeze(0)).abs()  # (S_prev, S_curr, W)


class GibbsCutLogPartition(nn.Module):
    """Gibbs Cut Log-Partition Operator primitive head (p037)."""

    ALLOWED_ABLATIONS = ALLOWED_ABLATIONS

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        grid_h: int = 4,
        grid_w: int = 4,
        d_cut: int = 4,
        temperature: float = 1.0,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("GibbsCutLogPartition supports the puzzle_binary one-logit contract")
        if int(input_channels) != 18:
            raise ValueError("GibbsCutLogPartition requires the simple_18 board tensor")
        if int(grid_w) < 2 or int(grid_w) > 6:
            raise ValueError("grid_w must be in [2, 6] -- 2^W states per row")
        if int(grid_h) < 1 or int(grid_h) > 8:
            raise ValueError("grid_h must be in [1, 8]")
        if int(d_cut) < 1:
            raise ValueError("d_cut must be >= 1")
        if float(temperature) <= 0:
            raise ValueError("temperature must be > 0")
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}",
            )

        self.num_classes = 1
        self.ablation = str(ablation)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.d_cut = int(d_cut)
        self.temperature = float(temperature)
        self._n_states = 1 << self.grid_w

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
        self.feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )

        per_grid = self.grid_h * self.grid_w * self.d_cut
        # Project the joint feature to four edge / penalty channels:
        # horizontal edges (H, W-1, d_cut), vertical edges (H-1, W, d_cut),
        # source penalties s (H, W, d_cut), sink penalties t (H, W, d_cut).
        self.edge_h_proj = nn.Linear(self.feature_dim, self.grid_h * (self.grid_w - 1) * self.d_cut)
        self.edge_v_proj = nn.Linear(self.feature_dim, (self.grid_h - 1) * self.grid_w * self.d_cut)
        self.source_proj = nn.Linear(self.feature_dim, per_grid)
        self.sink_proj = nn.Linear(self.feature_dim, per_grid)

        # Static state-bit / XOR tables used by the transfer DP.
        self.register_buffer(
            "state_bits", _build_state_bits(self.grid_w), persistent=False,
        )  # (S, W)
        self.register_buffer(
            "within_row_xor", _build_within_row_xor(self.grid_w), persistent=False,
        )  # (S, W-1)
        self.register_buffer(
            "between_row_xor", _build_between_row_xor(self.grid_w), persistent=False,
        )  # (S_prev, S_curr, W)

        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        # readout = log_partition pool + edge marginal stats (mean per channel)
        readout_dim = 2 * self.d_cut
        self.delta_head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, int(head_hidden_dim)),
            nn.GELU(),
            head_dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )
        with torch.no_grad():
            final_layer = self.gate_head[-1]
            if isinstance(final_layer, nn.Linear):
                final_layer.bias.fill_(float(gate_init))

    def _project_cut_inputs(
        self, joint: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = joint.shape[0]
        h, w, d = self.grid_h, self.grid_w, self.d_cut
        # All edge / penalty signals are passed through softplus so they
        # remain non-negative -- this matches the c >= 0 contract of the
        # cut log-partition operator.
        c_h = torch.nn.functional.softplus(self.edge_h_proj(joint)).view(batch, h, w - 1, d)
        c_v = torch.nn.functional.softplus(self.edge_v_proj(joint)).view(batch, h - 1, w, d)
        s = torch.nn.functional.softplus(self.source_proj(joint)).view(batch, h, w, d)
        t = torch.nn.functional.softplus(self.sink_proj(joint)).view(batch, h, w, d)

        if self.ablation == "uniform_edges":
            c_h = torch.ones_like(c_h)
            c_v = torch.ones_like(c_v)
        elif self.ablation == "uniform_sources":
            s = torch.ones_like(s)
            t = torch.ones_like(t)
        return c_h, c_v, s, t

    def _compute_log_partition(
        self,
        c_h: torch.Tensor,  # (B, H, W-1, d)
        c_v: torch.Tensor,  # (B, H-1, W, d)
        s: torch.Tensor,    # (B, H, W, d)
        t: torch.Tensor,    # (B, H, W, d)
    ) -> torch.Tensor:
        """Return log-partition tensor of shape ``(B, d_cut)``.

        Per channel ``c`` and batch row ``b``, this is the soft minimum

            log Z = log sum_{S in 2^V} exp(-energy(S; b, c) / tau)

        computed by a row-transfer dynamic program.
        """
        batch, h, w, d = s.shape
        n_states = self._n_states
        tau = self.temperature
        bits = self.state_bits  # (S, W)
        within = self.within_row_xor  # (S, W-1)
        between = self.between_row_xor  # (S_prev, S_curr, W)

        # within-row cost per (b, r, S, d):
        #     sum_{j<W-1} c_h[b, r, j, d] * within[S, j]
        # shape (B, H, S, d)
        within_cost = torch.einsum("sj,brjd->brsd", within, c_h) / tau

        # cell cost per (b, r, S, d):
        #     sum_j  s[b, r, j, d] * (1 - bits[S, j]) + t[b, r, j, d] * bits[S, j]
        ones_minus_bits = 1.0 - bits  # (S, W)
        cell_cost = (
            torch.einsum("sj,brjd->brsd", ones_minus_bits, s)
            + torch.einsum("sj,brjd->brsd", bits, t)
        ) / tau

        # log_Z_r[S_curr] = -within_cost[r, S_curr] - cell_cost[r, S_curr]
        #   + logsumexp_S_prev(log_Z_{r-1}[S_prev] - between_cost[r-1, S_prev, S_curr]).
        # For r = 0 (no previous row), log_Z_0[S] = -within[0, S] - cell[0, S].
        log_Z = -within_cost[:, 0] - cell_cost[:, 0]  # (B, S, d)

        for r in range(1, h):
            # between cost per (b, S_prev, S_curr, d):
            #    sum_j c_v[b, r-1, j, d] * between[S_prev, S_curr, j]
            between_cost = (
                torch.einsum("pqj,bjd->bpqd", between, c_v[:, r - 1]) / tau
            )  # (B, S, S, d)
            # broadcast log_Z over the new S_curr axis and add -between_cost
            #   shape becomes (B, S_prev, S_curr, d)
            joint_log = log_Z.unsqueeze(2) - between_cost  # (B, S, S, d)
            log_Z = torch.logsumexp(joint_log, dim=1)  # (B, S_curr, d)
            log_Z = log_Z - within_cost[:, r] - cell_cost[:, r]

        # Final reduction over S.
        return torch.logsumexp(log_Z, dim=1)  # (B, d)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        eps = 1.0e-6

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        c_h, c_v, s, t = self._project_cut_inputs(joint)
        log_Z = self._compute_log_partition(c_h, c_v, s, t)  # (B, d_cut)
        y = -self.temperature * log_Z  # (B, d_cut)

        # Edge-marginal proxy: how strong is the *expected* cut energy.
        # We expose the per-channel mean of c_h + c_v as a diagnostic; the
        # true marginals (dy/dc_e) are available via autograd but we do
        # not store them here.
        edge_energy = c_h.flatten(1).mean(dim=1, keepdim=False)  # currently scalar per sample
        # Better: per-channel mean.
        edge_energy_c = c_h.mean(dim=(1, 2)) + c_v.mean(dim=(1, 2))  # (B, d_cut)

        if self.ablation == "shuffle_logpartition" and batch > 1:
            perm = torch.randperm(batch, device=board.device)
            y = y[perm]
            edge_energy_c = edge_energy_c[perm]

        readout = torch.cat([y, edge_energy_c], dim=-1)  # (B, 2 d_cut)

        delta_raw = self.delta_head(readout).view(-1)

        gate_logit = self.gate_head(joint).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
        else:
            primitive_delta = gate * delta_raw

        logits = base_logit + primitive_delta

        gate_clamped = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(
            gate_clamped * gate_clamped.log()
            + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
        )

        log_partition_mean = y.mean(dim=1)
        log_partition_max = y.amax(dim=1)
        cut_edge_energy_mean = edge_energy_c.mean(dim=1)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "base_logit": base_logit,
            "primitive_delta": primitive_delta,
            "primitive_delta_raw": delta_raw,
            "primitive_gate": gate,
            "primitive_gate_logit": gate_logit,
            "primitive_gate_entropy": gate_entropy,
            "gibbs_log_partition_mean": log_partition_mean,
            "gibbs_log_partition_max": log_partition_max,
            "gibbs_cut_edge_energy": cut_edge_energy_mean,
            "mechanism_energy": trunk_out["mechanism_energy"] + log_partition_mean.detach(),
            "proposal_profile_strength": (primitive_delta.abs() * gate_entropy).clamp(0.0, 20.0),
            "proposal_keyword_count": logits.new_full((batch,), float(self._n_states * self.grid_h)),
        }
        for key, value in trunk_out.items():
            if key == "logits":
                continue
            diag_key = (
                key if key.startswith("base_") or key in {"gate", "gate_logit"} else f"trunk_{key}"
            )
            diagnostics.setdefault(diag_key, value)
        return diagnostics


def build_gibbs_cut_log_partition_from_config(
    config: dict[str, Any],
) -> GibbsCutLogPartition:
    cfg = dict(config)
    return GibbsCutLogPartition(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        grid_h=int(cfg.get("grid_h", 4)),
        grid_w=int(cfg.get("grid_w", 4)),
        d_cut=int(cfg.get("d_cut", 4)),
        temperature=float(cfg.get("temperature", 1.0)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )
