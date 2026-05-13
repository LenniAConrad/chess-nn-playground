"""Canonical-Orbit Straight-Through Operator (p036).

Source: ``ideas/research/primitives/external_31_canonical_orbit_bdd_wmc_primitives.md``
(rank-1 proposal ``primitive_canonical_orbit_st``).

The primitive canonicalises a per-square latent tensor under a finite chess
symmetry group ``G`` by

    g_b^* = argmin_{g in G}  kappa(T_g X_b),
    Y_b   = T_{g_b^*} X_b,

where ``kappa`` is a *non-learned* hash key (a fixed quantised random
projection) and ``T_g`` is a deterministic permutation action over the 64
board squares. Because the discrete argmin uses a non-differentiable hash
the backward gradient is the inverse of the chosen transform; for
permutation actions this is automatic in PyTorch -- ``index_select`` /
``gather`` already routes the gradient through the inverse permutation,
which is the straight-through behaviour the spec asks for.

Group ``G`` is the natural board-geometry C2 x C2 group:

    e            (identity)
    F            (file mirror)
    R            (rank mirror)
    F . R        (180-degree rotation)

These act only on square indices, so the channel layout of the simple_18
board tensor is preserved and the model layer's channel semantics stay
intact. Colour swap and side-to-move flip would also touch channels and
the logit sign, so they are *not* part of ``G`` for this primitive --
that variant is documented in ``ablations.md`` as a deferred extension.

Output contract: standard additive, gated primitive head over the i193
trunk. The canonical representative is pooled, projected to a scalar
delta, gated, and added to the base logit. CRTK metadata, source labels,
verification flags, engine evaluations, and report-only metadata are not
used anywhere in this module.

Deferred internal proposals from the same packet:

- ``primitive_bdd_wmc`` (rank 2): BDD Weighted-Model-Count layer.
- ``primitive_matroid_rank_envelope`` (rank 3): Matroid-rank envelope
  pooling.
- ``primitive_tactical_lcp_projector`` (rank 4): Tactical complementarity
  LCP projector.
- ``primitive_delta_cholesky_whiten`` (rank 5): Bounded-delta Cholesky
  whitening accumulator.
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


SQUARES = 64
HEIGHT = 8
WIDTH = 8

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "identity_only",       # disable orbit search -- always choose identity
    "fixed_choice",        # always choose the first transform (file mirror)
    "shuffle_canonical",   # in-batch permutation of canonical representative
    "zero_delta",
    "trunk_only",
)


def _build_permutations() -> torch.Tensor:
    """Return the (|G|, 64) integer permutation indices for the C2 x C2 group.

    Each row maps the *destination* square index ``s`` to the *source*
    square index ``s'`` so that ``Y[..., s] = X[..., perm[s]]``. The four
    actions are identity, file mirror, rank mirror, and 180-deg rotation.
    """
    perms = []
    base = torch.arange(SQUARES, dtype=torch.long).view(HEIGHT, WIDTH)
    # identity
    perms.append(base.flatten())
    # file mirror -- flip along the file axis (columns)
    perms.append(torch.flip(base, dims=(1,)).flatten())
    # rank mirror -- flip along the rank axis (rows)
    perms.append(torch.flip(base, dims=(0,)).flatten())
    # 180-deg rotation -- flip both axes
    perms.append(torch.flip(base, dims=(0, 1)).flatten())
    return torch.stack(perms, dim=0)  # (4, 64)


class CanonicalOrbitSTOperator(nn.Module):
    """Canonical-Orbit Straight-Through Operator primitive head (p036)."""

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
        latent_dim: int = 24,
        key_dim: int = 8,
        hash_quantum: float = 1.0,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "CanonicalOrbitSTOperator supports the puzzle_binary one-logit contract",
            )
        if int(input_channels) != 18:
            raise ValueError(
                "CanonicalOrbitSTOperator requires the simple_18 board tensor",
            )
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}",
            )

        self.num_classes = 1
        self.ablation = str(ablation)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.latent_dim = int(latent_dim)
        self.key_dim = int(key_dim)
        self.hash_quantum = float(hash_quantum)

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

        # Project the trunk joint feature back into a per-square latent map.
        self.latent_proj = nn.Linear(self.feature_dim, SQUARES * self.latent_dim)

        # Fixed permutation table for the orbit group.
        perms = _build_permutations()
        self.register_buffer("permutations", perms, persistent=False)
        # |G| from the buffer
        self._group_order = int(perms.shape[0])

        # Fixed non-learned hash projection. Stored as a buffer so it is
        # invariant across runs and not part of the gradient graph. The
        # key is deterministic given a seed, lexicographic across two
        # tie-breaker columns.
        generator = torch.Generator(device="cpu").manual_seed(0xC0DEC0DE)
        proj = torch.randn(
            self.latent_dim, self.key_dim, generator=generator, dtype=torch.float32,
        )
        self.register_buffer("hash_projection", proj, persistent=False)

        # Square position weights used to weight the per-square hash key
        # contributions. Also fixed.
        sq_weights = torch.linspace(1.0, 2.0, steps=SQUARES, dtype=torch.float32)
        self.register_buffer("hash_square_weights", sq_weights, persistent=False)

        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        readout_dim = self.latent_dim * 2  # canonical pool + symmetry residual pool
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

    @property
    def group_order(self) -> int:
        return self._group_order

    def _hash_keys(self, transformed: torch.Tensor) -> torch.Tensor:
        """Compute non-learned hash keys for each batched orbit element.

        ``transformed`` has shape ``(B, |G|, 64, d)``. Returns a tensor of
        shape ``(B, |G|, key_dim)`` produced by

            key = sum_s  hash_square_weights[s] * quantise(latent[s] @ P)

        where ``P`` is a fixed random projection and ``quantise`` rounds
        to the nearest multiple of ``hash_quantum``. The key is computed
        under ``torch.no_grad()`` because the argmin is non-differentiable.
        """
        with torch.no_grad():
            scores = transformed @ self.hash_projection  # (B, |G|, 64, key_dim)
            scores = torch.round(scores / self.hash_quantum) * self.hash_quantum
            weighted = scores * self.hash_square_weights.view(1, 1, SQUARES, 1)
            keys = weighted.sum(dim=2)  # (B, |G|, key_dim)
        return keys

    def _pick_canonical_index(self, keys: torch.Tensor) -> torch.Tensor:
        """Return ``(B,)`` long tensor with argmin orbit indices.

        Lexicographic order is enforced by packing the key columns into
        a single scalar score; ties are broken by the next column. Since
        the projection produces real-valued keys we use the lexicographic
        compare via a sequential argmin pass with deterministic tie
        breaking favouring the lower index (which corresponds to ``e``).
        """
        # (B, |G|, key_dim) -> lexicographic argmin.
        with torch.no_grad():
            batch, g_size, key_dim = keys.shape
            best = torch.zeros(batch, dtype=torch.long, device=keys.device)
            best_keys = keys[:, 0, :].clone()  # (B, key_dim)
            for g in range(1, g_size):
                candidate = keys[:, g, :]
                cmp = candidate - best_keys
                # Find the first non-zero column for each batch row.
                # Treat values within an absolute tolerance as ties.
                tol = 1.0e-7
                nonzero = cmp.abs() > tol
                # Index of the first non-zero column; if all equal, use last.
                first_nz = torch.where(
                    nonzero.any(dim=1),
                    nonzero.float().argmax(dim=1),
                    torch.full((batch,), key_dim - 1, dtype=torch.long, device=keys.device),
                )
                # Gather sign of cmp at the first non-zero column.
                first_cmp = cmp.gather(1, first_nz.view(-1, 1)).squeeze(1)
                better = first_cmp < 0
                best = torch.where(better, torch.full_like(best, g), best)
                best_keys = torch.where(better.unsqueeze(1), candidate, best_keys)
        return best  # (B,)

    def _apply_inverse_for_diagnostic(
        self, canonical: torch.Tensor, chosen: torch.Tensor
    ) -> torch.Tensor:
        """For diagnostics, map ``canonical`` back to the original frame.

        Used only for reporting: ``mapped_back[b] = T_{g_b^*}^{-1} canonical[b]``.
        For the C2 x C2 group every element is its own inverse so this is
        the same gather operation as the forward orbit transform.
        """
        # Build inverse permutations -- same as forward for this group.
        # mapped_back[b, s, d] = canonical[b, inv_perm[g_b^*][s], d]
        inv_perms = self.permutations  # involutions in C2 x C2
        chosen_perm = inv_perms[chosen]  # (B, 64)
        idx = chosen_perm.unsqueeze(-1).expand(-1, -1, canonical.shape[-1])
        return canonical.gather(dim=1, index=idx)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        eps = 1.0e-6

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        # Project joint feature -> per-square latent map X in (B, 64, d).
        latent_flat = self.latent_proj(joint)  # (B, 64*d)
        latent = latent_flat.view(batch, SQUARES, self.latent_dim)

        # Apply each group element to produce (B, |G|, 64, d).
        g_size = self.group_order
        # perm[g, dst] -> src. We want transformed[b, g, dst, d] = latent[b, perm[g, dst], d]
        perm_expand = self.permutations.unsqueeze(0).expand(batch, g_size, SQUARES)
        # Build batched gather index of shape (B, |G|, 64, d).
        idx = perm_expand.unsqueeze(-1).expand(-1, -1, -1, self.latent_dim)
        # Replicate latent across G axis for gather.
        latent_expand = latent.unsqueeze(1).expand(-1, g_size, -1, -1)
        transformed = latent_expand.gather(dim=2, index=idx)  # (B, |G|, 64, d)

        # Hash keys (non-differentiable).
        keys = self._hash_keys(transformed)  # (B, |G|, key_dim)

        if self.ablation == "identity_only":
            chosen = torch.zeros(batch, dtype=torch.long, device=board.device)
        elif self.ablation == "fixed_choice":
            chosen = torch.ones(batch, dtype=torch.long, device=board.device)
        else:
            chosen = self._pick_canonical_index(keys)

        # canonical[b, s, d] = transformed[b, chosen[b], s, d]
        chosen_view = chosen.view(batch, 1, 1, 1).expand(-1, 1, SQUARES, self.latent_dim)
        canonical = transformed.gather(dim=1, index=chosen_view).squeeze(1)  # (B, 64, d)

        if self.ablation == "shuffle_canonical" and batch > 1:
            perm_b = torch.randperm(batch, device=board.device)
            canonical = canonical[perm_b]

        # Symmetry residual: how much does canonicalisation change the latent?
        residual = canonical - latent  # (B, 64, d)

        canonical_pool = canonical.mean(dim=1)  # (B, d)
        residual_pool = (residual.pow(2).mean(dim=1) + eps).sqrt()  # (B, d)
        readout = torch.cat([canonical_pool, residual_pool], dim=-1)  # (B, 2d)

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

        # Diagnostics: which group element was selected, and the orbit
        # spread (the gap between the chosen key and the worst key).
        with torch.no_grad():
            chosen_key = keys.gather(
                1, chosen.view(-1, 1, 1).expand(-1, 1, keys.shape[-1])
            ).squeeze(1)
            worst_key = keys.amax(dim=1)
            orbit_gap = (worst_key - chosen_key).norm(dim=1)
            # Count number of orbit elements tied with the selected key (>=1).
            key_diff = (keys - chosen_key.unsqueeze(1)).abs().sum(dim=-1)
            ties = (key_diff <= 1.0e-6).float().sum(dim=1)

        residual_norm = (residual.pow(2).flatten(1).mean(dim=1) + eps).sqrt()
        canonical_norm = (canonical.pow(2).flatten(1).mean(dim=1) + eps).sqrt()

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "base_logit": base_logit,
            "primitive_delta": primitive_delta,
            "primitive_delta_raw": delta_raw,
            "primitive_gate": gate,
            "primitive_gate_logit": gate_logit,
            "primitive_gate_entropy": gate_entropy,
            "cost_chosen_orbit_index": chosen.to(dtype=base_logit.dtype),
            "cost_orbit_gap": orbit_gap,
            "cost_orbit_ties": ties,
            "cost_residual_norm": residual_norm,
            "cost_canonical_norm": canonical_norm,
            "mechanism_energy": trunk_out["mechanism_energy"] + residual_norm.detach(),
            "proposal_profile_strength": (primitive_delta.abs() * gate_entropy).clamp(0.0, 20.0),
            "proposal_keyword_count": logits.new_full((batch,), float(self.group_order)),
        }
        for key, value in trunk_out.items():
            if key == "logits":
                continue
            diag_key = (
                key if key.startswith("base_") or key in {"gate", "gate_logit"} else f"trunk_{key}"
            )
            diagnostics.setdefault(diag_key, value)
        return diagnostics


def build_canonical_orbit_st_operator_from_config(
    config: dict[str, Any],
) -> CanonicalOrbitSTOperator:
    cfg = dict(config)
    return CanonicalOrbitSTOperator(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        latent_dim=int(cfg.get("latent_dim", 24)),
        key_dim=int(cfg.get("key_dim", 8)),
        hash_quantum=float(cfg.get("hash_quantum", 1.0)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )
