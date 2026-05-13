"""Grassmann Rook-Matching Pool (p043, GRMP) primitive.

Source: ``ideas/research/primitives/external_38_polynomial_ledger_grassmann_rook_primitives.md``
(Section ``primitive_grassmann_rook_pool``; second-ranked proposal — promoted
over the file's #1 ``primitive_polynomial_ledger`` because that family is
already covered by p042 truncated_multiset_polynomial_pool). Deferred
internal proposals from the same file:

- ``primitive_polynomial_ledger`` -- covered by p042.
- ``primitive_matroid_sparsemax`` -- matroid-constrained sparse pooling;
  family closer to ``ParetoAntichainFrontierNetwork`` (p001) and
  SparseMAP; not in scope for this batch.
- ``primitive_irrepnorm`` and ``primitive_poset_conenorm`` -- irrep
  normalisation and order-cone projection; orbit/irrep family deferred.

The Grassmann Rook-Matching Pool computes the truncated matching-
polynomial coefficients of a learned bipartite score tensor
``z_{b,i,j,h}`` over (active attacker, active defender) square pairs.
For nilpotent generators ``g_{ij} = epsilon_i wedge eta_j`` with
``epsilon_i^2 = eta_j^2 = 0``, the formal product

    P_h(t) = prod_{i,j} (1 + t * z_{b,i,j,h} * g_{ij})

automatically deletes every monomial that re-uses a row or column.
Truncating at degree ``K``,

    Y_{b,k,h} = sum_{S : |S|=k, rows(S) disjoint, cols(S) disjoint}
                  prod_{(i,j) in S} z_{b,i,j,h},

i.e. the sum over k-tuples of row/column-disjoint edges. This is the
matching polynomial truncation; with row=col=square, it corresponds to
rook polynomials over a chess square set.

The recurrence is

    coef[0, mask] = 1.0
    coef[k, mask | {i,j}] += coef[k-1, mask] * z[i,j]

implemented in vectorised form by maintaining ``B x K x C`` matching-
coefficient channels and per-step row/column exclusion bookkeeping.
For tractable scout-scale runs we cap at ``K = 2`` and pre-select
``num_attackers`` and ``num_defenders`` salient squares per side using
soft top-k logits from the i193 trunk's spatial features.

The primitive is an additive gated logit delta over the i193
ExchangeThenKingDualStreamNetwork trunk:

    final_logit = base_logit + gate * primitive_delta_raw

CRTK metadata, source labels, verification flags, engine evaluations
and principal variations are *not* consumed.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.codex_reply_primitives import (
    BoardTokenAttention,
)
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor
from chess_nn_playground.models.primitives.trunk_features import trunk_joint_features


SQUARES = 64
NUM_PIECE_CHANNELS = 12

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "drop_exclusion",        # primary falsifier — disable row/column exclusion (collapse to bilinear pool)
    "scalar_score",          # primary falsifier #2 — collapse score channels to one
    "shuffle_attackers",     # permute attacker tokens across batch (rule-feature falsifier)
    "shuffle_defenders",     # permute defender tokens across batch
    "zero_delta",
    "trunk_only",
    "disable_gate",
)


def grassmann_rook_matching_coefficients(
    score: torch.Tensor,
    attacker_mask: torch.Tensor,
    defender_mask: torch.Tensor,
    degree: int,
    *,
    exclude_rows_cols: bool = True,
) -> torch.Tensor:
    """Compute truncated matching-polynomial coefficients of a bipartite score tensor.

    Args:
        score: ``(B, R, C, H)`` per-edge score (R attacker rows, C defender cols, H channels).
        attacker_mask: ``(B, R)`` (0/1) indicator of active attacker tokens.
        defender_mask: ``(B, C)`` (0/1) indicator of active defender tokens.
        degree: matching truncation ``K`` (>= 1).
        exclude_rows_cols: if False, collapse to ordinary bilinear pool (no row/column exclusion).

    Returns:
        ``(B, K, H)`` coefficients ``e_k = sum_{|S|=k disjoint} prod_{(i,j) in S} z_{i,j,h}``.
    """
    if score.dim() != 4:
        raise ValueError(f"Expected (B, R, C, H) score, got {tuple(score.shape)}")
    if int(degree) < 1:
        raise ValueError("degree must be >= 1")

    batch, rows, cols, channels = score.shape
    dtype = score.dtype
    device = score.device

    weighted = score * attacker_mask.to(dtype=dtype).unsqueeze(-1).unsqueeze(-1)
    weighted = weighted * defender_mask.to(dtype=dtype).unsqueeze(1).unsqueeze(-1)

    if not exclude_rows_cols:
        # Bilinear/DeepSets-style collapse: each ``e_k`` becomes ``(sum_{i,j} z)^k / k!``-like
        # additive structure. Implement via a 1D elementary symmetric scan over flat edges;
        # this is the failure mode the spec calls out as the primary rebrand risk.
        flat_z = weighted.view(batch, rows * cols, channels)
        scaling_mask = torch.ones(batch, rows * cols, device=device, dtype=dtype)
        e_k = [torch.zeros(batch, channels, device=device, dtype=dtype) for _ in range(int(degree))]
        e0 = torch.ones(batch, channels, device=device, dtype=dtype)
        for idx in range(rows * cols):
            u_e = flat_z[:, idx, :] * scaling_mask[:, idx].unsqueeze(-1)
            for k in range(int(degree), 0, -1):
                if k == 1:
                    e_k[0] = e_k[0] + u_e * e0
                else:
                    e_k[k - 1] = e_k[k - 1] + u_e * e_k[k - 2]
        return torch.stack(e_k, dim=1)

    # Exclusion-aware scan.
    # State for degree d = 1..K: per-(row-used, col-used) tensor too expensive
    # for full enumeration. We use the standard rook-polynomial truncation
    # at degree K = 1, 2 in closed form (covers the scout-scale spec recommendation
    # of K <= 2). For K = 3 we fall back to a generic exclusion scan that
    # marginalises rows and columns iteratively.
    e1 = weighted.sum(dim=(1, 2))  # (B, H)
    coefficients = [e1]
    if int(degree) >= 2:
        # e_2 = (sum_{i,j} z)^2 - sum_{i,j} z^2  - row-double-count - col-double-count
        # Simplest derivation: enumerate row/col-disjoint pairs.
        # sum_{i1<i2, j1!=j2} z[i1,j1] z[i2,j2] = 0.5 * ( (sum z)^2 - sum z^2
        #                                                - row_double_count
        #                                                - col_double_count ).
        row_sum = weighted.sum(dim=2)  # (B, R, H)
        col_sum = weighted.sum(dim=1)  # (B, C, H)
        total = e1
        sq_individual = (weighted * weighted).sum(dim=(1, 2))  # (B, H)
        row_collisions = (row_sum * row_sum).sum(dim=1) - sq_individual
        # row_collisions = sum_i (sum_j z_{ij})^2 - sum_{ij} z_{ij}^2
        col_collisions = (col_sum * col_sum).sum(dim=1) - sq_individual
        e2 = 0.5 * (total * total - sq_individual - row_collisions - col_collisions)
        coefficients.append(e2)
    if int(degree) >= 3:
        # Generic O(R*C*K) exclusion scan for K=3.
        # For each candidate edge (i, j), e_3 picks two other edges that share
        # no row or column with (i, j) and each other. The closed form below
        # uses inclusion/exclusion over the e_2 obtained by deleting row i / col j.
        # e_3 = (1/3) * sum_{(i,j)} weighted[i,j] * e_2^{-i,-j}
        # where e_2^{-i,-j} is the rank-2 matching polynomial coefficient on
        # the score matrix with row i and column j masked out.
        # We implement e_2^{-i,-j} by Sherman--Morrison-style deletion:
        # e_2^{-i,-j} = e_2_full - (terms touching row i) - (terms touching col j)
        #             + (terms touching both row i and column j).
        row_sum = weighted.sum(dim=2)  # (B, R, H)
        col_sum = weighted.sum(dim=1)  # (B, C, H)
        # total e_1 excluding (i, j): e1 - row_sum[i] - col_sum[j] + weighted[i, j]
        # We'll need this per (i, j). Build via broadcasting.
        # row-touching-pairs term: row_sum[i] * (e1 - row_sum[i]) - sq_individual_i_terms
        # We compute e_3 directly via a simpler O(R*C*H) iteration:
        e3 = torch.zeros(batch, channels, device=device, dtype=dtype)
        sq_per_edge = weighted * weighted  # (B, R, C, H)
        for i in range(rows):
            row_i = row_sum[:, i]  # (B, H)
            for j in range(cols):
                z_ij = weighted[:, i, j]  # (B, H)
                col_j = col_sum[:, j]  # (B, H)
                # e1 over remaining edges (exclude row i AND column j entirely).
                e1_rest = e1 - row_i - col_j + z_ij  # (B, H)
                # sum z^2 over remaining edges.
                sq_remaining = (
                    sq_individual
                    - sq_per_edge[:, i, :].sum(dim=1)
                    - sq_per_edge[:, :, j].sum(dim=1)
                    + sq_per_edge[:, i, j]
                )
                # row collisions over remaining: sum_{i'!=i} (row_sum[i'] - weighted[i', j])^2 - corresponding squares
                row_sums_rest = row_sum - weighted[:, :, j].view(batch, rows, channels)  # subtract col-j contribution from every row
                # For the (i, _) row, mask out entirely.
                row_sums_rest = row_sums_rest * (1.0 - _row_indicator(i, rows, device, dtype).view(1, rows, 1))
                row_col_rest = (row_sums_rest * row_sums_rest).sum(dim=1)
                # sq sum per row excluding row i and col j: sq_per_edge[i', j'] for i'!=i, j'!=j
                sq_row_rest = sq_per_edge.sum(dim=2) - sq_per_edge[:, :, j]
                sq_row_rest = sq_row_rest * (1.0 - _row_indicator(i, rows, device, dtype).view(1, rows, 1))
                row_col_collisions_rest = row_col_rest - sq_row_rest.sum(dim=1)
                # col collisions over remaining: symmetric.
                col_sums_rest = col_sum - weighted[:, i, :].view(batch, cols, channels)
                col_sums_rest = col_sums_rest * (1.0 - _col_indicator(j, cols, device, dtype).view(1, cols, 1))
                col_col_rest = (col_sums_rest * col_sums_rest).sum(dim=1)
                sq_col_rest = sq_per_edge.sum(dim=1) - sq_per_edge[:, i, :]
                sq_col_rest = sq_col_rest * (1.0 - _col_indicator(j, cols, device, dtype).view(1, cols, 1))
                col_col_collisions_rest = col_col_rest - sq_col_rest.sum(dim=1)
                e2_rest = 0.5 * (
                    e1_rest * e1_rest
                    - sq_remaining
                    - row_col_collisions_rest
                    - col_col_collisions_rest
                )
                e3 = e3 + z_ij * e2_rest
        e3 = e3 / 3.0
        coefficients.append(e3)
    if int(degree) >= 4:
        # Beyond K=3 we fall back to zeros; the source spec explicitly recommends K<=3.
        for _ in range(int(degree) - 3):
            coefficients.append(torch.zeros(batch, channels, device=device, dtype=dtype))
    return torch.stack(coefficients, dim=1)


def _row_indicator(i: int, rows: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    out = torch.zeros(rows, device=device, dtype=dtype)
    out[i] = 1.0
    return out


def _col_indicator(j: int, cols: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    out = torch.zeros(cols, device=device, dtype=dtype)
    out[j] = 1.0
    return out


class GrassmannRookPool(nn.Module):
    """p043 — Grassmann Rook-Matching Pool over the i193 trunk."""

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
        # GRMP hyper-parameters.
        num_attackers: int = 8,
        num_defenders: int = 8,
        token_dim: int = 32,
        score_channels: int = 8,
        degree: int = 2,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "GrassmannRookPool supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError("GrassmannRookPool requires the simple_18 board tensor")
        if int(num_attackers) < 2 or int(num_defenders) < 2:
            raise ValueError("num_attackers and num_defenders must be >= 2 for non-trivial matching")
        if int(degree) < 1 or int(degree) > 3:
            raise ValueError("degree must be in [1, 3] for the scout-scale rook truncation")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )
        self.num_classes = 1
        self.ablation = str(ablation)
        self.num_attackers = int(num_attackers)
        self.num_defenders = int(num_defenders)
        self.token_dim = int(token_dim)
        self.score_channels = int(score_channels)
        self.degree = int(degree)
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

        # Token compilers: separate attacker and defender query tokens compiled
        # from the i193 spatial features (concat of ex / king encoder outputs).
        spatial_channels = 2 * self.trunk.channels
        self.attacker_pool = BoardTokenAttention(
            in_channels=spatial_channels,
            num_tokens=self.num_attackers,
            token_dim=self.token_dim,
            dropout=float(head_dropout),
        )
        self.defender_pool = BoardTokenAttention(
            in_channels=spatial_channels,
            num_tokens=self.num_defenders,
            token_dim=self.token_dim,
            dropout=float(head_dropout),
        )

        # Per-token validity gate (sigmoid of a 1-D MLP) so the matching can
        # learn to ignore irrelevant tokens.
        self.attacker_mask_head = nn.Linear(self.token_dim, 1)
        self.defender_mask_head = nn.Linear(self.token_dim, 1)

        # Bipartite edge score: bilinear over attacker and defender tokens with H channels.
        self.bilinear = nn.Bilinear(self.token_dim, self.token_dim, self.score_channels)

        coeff_dim = self.degree * self.score_channels
        self.coeff_norm = nn.LayerNorm(coeff_dim)
        self.feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )
        dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(coeff_dim + self.feature_dim),
            nn.Linear(coeff_dim + self.feature_dim, int(head_hidden_dim)),
            nn.GELU(),
            dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )

        gate_in = self.feature_dim + 3  # joint + (atk_count, def_count, coeff_norm)
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
        """Rebuild the i193 spatial map (concat of ex_h and kg_h)."""
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
        return torch.cat([ex_h, kg_h], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        device = board.device
        dtype = board.dtype

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        spatial = self._spatial_features(board)
        atk = self.attacker_pool(spatial)
        defs = self.defender_pool(spatial)
        atk_tokens = atk.tokens
        def_tokens = defs.tokens

        if self.ablation == "shuffle_attackers" and batch > 1:
            perm = torch.randperm(batch, device=device)
            atk_tokens = atk_tokens[perm]
        if self.ablation == "shuffle_defenders" and batch > 1:
            perm = torch.randperm(batch, device=device)
            def_tokens = def_tokens[perm]

        atk_mask = torch.sigmoid(self.attacker_mask_head(atk_tokens).squeeze(-1))
        def_mask = torch.sigmoid(self.defender_mask_head(def_tokens).squeeze(-1))

        # Bilinear edge score: (B, R, D, H).
        atk_b = atk_tokens.unsqueeze(2).expand(batch, self.num_attackers, self.num_defenders, self.token_dim)
        def_b = def_tokens.unsqueeze(1).expand(batch, self.num_attackers, self.num_defenders, self.token_dim)
        score = self.bilinear(
            atk_b.reshape(-1, self.token_dim), def_b.reshape(-1, self.token_dim)
        ).view(batch, self.num_attackers, self.num_defenders, self.score_channels)

        if self.ablation == "scalar_score":
            scalar = score.mean(dim=-1, keepdim=True)
            score = scalar.expand_as(score)

        # Bound the edge magnitudes so high-degree coefficients do not explode.
        score = torch.tanh(score)

        exclude = self.ablation != "drop_exclusion"
        coeff = grassmann_rook_matching_coefficients(
            score,
            atk_mask,
            def_mask,
            self.degree,
            exclude_rows_cols=exclude,
        )

        coeff_flat = coeff.reshape(batch, -1)
        coeff_flat = self.coeff_norm(coeff_flat)
        coeff_norm_sample = coeff.pow(2).mean(dim=(1, 2)).sqrt()
        atk_count = atk_mask.sum(dim=-1)
        def_count = def_mask.sum(dim=-1)

        delta_input = torch.cat([coeff_flat, joint], dim=1)
        delta_raw = self.delta_head(delta_input).view(-1)

        gate_input = torch.cat(
            [joint, atk_count.unsqueeze(-1), def_count.unsqueeze(-1), coeff_norm_sample.unsqueeze(-1)],
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
        out["grmp_attacker_count"] = atk_count
        out["grmp_defender_count"] = def_count
        out["grmp_coeff_norm"] = coeff_norm_sample
        out["grmp_coeff_e1"] = coeff[:, 0].pow(2).mean(dim=-1).sqrt()
        if self.degree >= 2:
            out["grmp_coeff_e2"] = coeff[:, 1].pow(2).mean(dim=-1).sqrt()
        if self.degree >= 3:
            out["grmp_coeff_e3"] = coeff[:, 2].pow(2).mean(dim=-1).sqrt()
        out["mechanism_energy"] = trunk_out["mechanism_energy"] + coeff_norm_sample.detach()
        out["proposal_profile_strength"] = primitive_delta.detach().abs() * gate_entropy
        out["proposal_keyword_count"] = logits.new_full(
            (batch,), float(self.num_attackers * self.num_defenders)
        )
        return out


def build_grassmann_rook_pool_from_config(config: dict[str, Any]) -> GrassmannRookPool:
    cfg = dict(config)
    return GrassmannRookPool(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        num_attackers=int(cfg.get("num_attackers", 8)),
        num_defenders=int(cfg.get("num_defenders", 8)),
        token_dim=int(cfg.get("token_dim", 32)),
        score_channels=int(cfg.get("score_channels", 8)),
        degree=int(cfg.get("degree", 2)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "ALLOWED_ABLATIONS",
    "GrassmannRookPool",
    "build_grassmann_rook_pool_from_config",
    "grassmann_rook_matching_coefficients",
)
