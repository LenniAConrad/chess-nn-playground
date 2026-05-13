"""Truncated Exterior Product Pool (p041).

Source: ``ideas/research/primitives/external_36_exterior_product_rank1_resolvent_primitives.md``
(rank-1 proposal ``primitive_exterior_product_pool``). The rank-2
proposal in the same packet (``primitive_rank1_resolvent_pool``) is the
same operator as p038 (Woodbury Set Resolver), so it is not promoted
here.

The primitive pools active piece tokens through the truncated exterior
algebra

    z_i = a_i * (W phi(x_i)) in R^r
    M   = prod_{i}^{(wedge, <=R)} (1 + z_i),
    M^{(k)} = sum_{|I|=k} bigwedge_{i in I} z_i  in Lambda^k(R^r).

The wedge product is *antisymmetric*: ``z_i ^ z_j = - z_j ^ z_i``, so
linearly dependent tokens cancel. This is fundamentally different from
the elementary-symmetric polynomial accumulator (``p024``), which uses
Hadamard products (commutative) and does not cancel collinear tokens.

Output is the per-grade vectorisation
``Y = concat_{k=0..R}(vec(M^{(k)}))`` of total dimension
``D_R = sum_{k=0..R} C(r, k)``. The wedge-product table is precomputed
once at construction time; the forward pass is a polynomial recurrence
in the multivector ring.

For chess this expresses *non-redundant* high-order co-presence of
piece tokens: two defenders on the same latent line are linearly
dependent and contribute zero to the grade-2 part. The pooled
multivector is projected to a scalar gated delta over the i193 trunk.

CRTK metadata, source labels, verification flags, engine evaluations,
and report-only metadata are not consumed.

Deferred internal proposals from the same packet:

- ``primitive_rank1_resolvent_pool`` (rank 2): covered by p038 in this
  same batch.
- ``primitive_orbit_stabilized_canonicalizer`` (rank 3): variant of
  p036; deferred (the stabilizer-averaging extension is documented in
  p036's ablations.md).
- ``primitive_tropical_distance_transform`` (rank 4): variant of the
  eikonal transform; partial overlap with p039; deferred.
- ``primitive_capacitated_entropic_assignment`` (rank 5): capacitated
  entropic assignment; overlaps with the Sinkhorn-style trunks already
  in the registry; deferred.
"""

from __future__ import annotations

from itertools import combinations
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
PIECE_PLANE_COUNT = 12

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "first_order_only",     # keep only grade 0 + 1 (degenerates to a sum pool)
    "shuffle_grades_high",  # in-batch permutation of grade >= 2
    "zero_delta",
    "trunk_only",
)


def _piece_tokens(board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-square (piece-plane + STM) input tensor and occupancy mask."""
    piece_planes = board[:, :PIECE_PLANE_COUNT].clamp(0.0, 1.0)
    stm = board[:, 12:13].clamp(0.0, 1.0)
    token_input = torch.cat([piece_planes, stm], dim=1).flatten(2).transpose(1, 2).contiguous()
    occupancy = piece_planes.flatten(2).sum(dim=1).clamp(0.0, 1.0)
    return token_input, occupancy


def _grade_indices(r: int, k: int) -> list[tuple[int, ...]]:
    """Enumerate ordered tuples ``(i_1 < i_2 < ... < i_k)`` for the wedge basis."""
    return [combo for combo in combinations(range(int(r)), int(k))]


class TruncatedExteriorProductPool(nn.Module):
    """Truncated Exterior Product Pool primitive head (p041)."""

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
        token_input_dim: int | None = None,
        r: int = 4,
        max_grade: int = 3,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("TruncatedExteriorProductPool supports the puzzle_binary one-logit contract")
        if int(input_channels) != 18:
            raise ValueError("TruncatedExteriorProductPool requires the simple_18 board tensor")
        if int(r) < 1 or int(r) > 8:
            raise ValueError("r must be in [1, 8] (binomial(r, k) blow-up otherwise)")
        if int(max_grade) < 1 or int(max_grade) > int(r):
            raise ValueError("max_grade must be in [1, r]")
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}",
            )

        self.num_classes = 1
        self.ablation = str(ablation)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.r = int(r)
        self.max_grade = int(max_grade)

        if token_input_dim is None:
            token_input_dim = PIECE_PLANE_COUNT + 1
        self.token_input_dim = int(token_input_dim)

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

        # Per-square token projection to z_i in R^r.
        self.z_proj = nn.Linear(self.token_input_dim, self.r)

        # Precompute multi-index tables for each grade.
        self._grade_dims: list[int] = []
        # Store grade-k basis indices as flat long tensors for gather operations.
        for k in range(self.max_grade + 1):
            indices = _grade_indices(self.r, k)
            self._grade_dims.append(len(indices))
            if k > 0:
                # Save the indices as a (D_k, k) long buffer.
                buf = torch.tensor(indices, dtype=torch.long)
                self.register_buffer(f"grade_idx_{k}", buf, persistent=False)
        # Build the "extend by tail" map: given a grade-(k-1) wedge with
        # multi-index alpha = (a_1, ..., a_{k-1}) and an extra index j not
        # in alpha, the wedge alpha ^ e_j produces a grade-k basis element
        # whose canonical multi-index is sorted(alpha + (j,)). We store
        # the canonical multi-index along with the sign from the
        # permutation needed to sort.
        for k in range(1, self.max_grade + 1):
            self._build_extension_table(k)

        self.D_R = sum(self._grade_dims)

        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.D_R),
            nn.Linear(self.D_R, int(head_hidden_dim)),
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

    def _build_extension_table(self, k: int) -> None:
        """Build per-grade lookup tables for the wedge update.

        For each grade-(k-1) basis element ``alpha`` and each scalar
        index ``j in 0..r-1``, the wedge ``e_alpha ^ e_j`` either
        vanishes (if ``j`` is already in ``alpha``) or maps to a
        grade-k basis element ``e_beta`` with an extra sign factor from
        sorting. We precompute:

        - ``target_basis_{k}``: (D_{k-1}, r) long tensor giving the
          canonical grade-k basis index, or -1 if the wedge vanishes.
        - ``target_sign_{k}``: (D_{k-1}, r) float tensor giving +/- 1
          (and 0 when target is -1).
        """
        prev_indices = _grade_indices(self.r, k - 1)
        curr_indices = _grade_indices(self.r, k)
        # Map from canonical multi-index tuple -> linear index in curr.
        idx_map = {tup: i for i, tup in enumerate(curr_indices)}

        target_basis = torch.full((len(prev_indices), self.r), -1, dtype=torch.long)
        target_sign = torch.zeros(len(prev_indices), self.r, dtype=torch.float32)
        for p, alpha in enumerate(prev_indices):
            for j in range(self.r):
                if j in alpha:
                    continue
                # Build the new k-tuple by inserting j; count how many
                # entries of alpha are larger than j -> that's the
                # number of swaps needed to bring j into sorted order.
                # Each swap flips the sign of the wedge.
                num_larger = sum(1 for a in alpha if a > j)
                sign = 1.0 if (num_larger % 2 == 0) else -1.0
                new_tuple = tuple(sorted(alpha + (j,)))
                target_basis[p, j] = idx_map[new_tuple]
                target_sign[p, j] = sign
        self.register_buffer(f"target_basis_{k}", target_basis, persistent=False)
        self.register_buffer(f"target_sign_{k}", target_sign, persistent=False)

    def _exterior_pool(
        self, z: torch.Tensor, mask: torch.Tensor
    ) -> list[torch.Tensor]:
        """Compute ``M^{(k)}`` for ``k = 0..max_grade`` from masked tokens.

        ``z`` has shape ``(B, n, r)``. ``mask`` is ``(B, n)``.

        We iterate over the tokens and grow the multivector via

            M^{(k)}_new[beta] = M^{(k)}_old[beta]
                                + sum_{alpha, j sorts to beta} sign(alpha, j)
                                  * z_i[j] * M^{(k-1)}_old[alpha]

        which corresponds to the multiplication
        ``M' = M * (1 + z_i)``  in the exterior algebra, truncated to
        grade ``<= max_grade``. The recurrence is run for ``k`` from
        ``max_grade`` down to ``1`` so ``M^{(k-1)}_old`` is the
        pre-update value at each step.
        """
        batch, n, r = z.shape
        assert r == self.r
        device = z.device
        dtype = z.dtype
        # Initialise: M[0] = 1, M[k>=1] = 0.
        grades: list[torch.Tensor] = []
        grades.append(z.new_ones(batch, 1))  # grade-0 scalar
        for k in range(1, self.max_grade + 1):
            grades.append(z.new_zeros(batch, self._grade_dims[k]))

        for i in range(n):
            zi = z[:, i, :]  # (B, r)
            mi = mask[:, i].unsqueeze(-1)  # (B, 1)
            zi_eff = zi * mi  # zero out masked tokens
            # Build the update from highest grade down so M^{(k-1)} is the
            # pre-update value.
            new_grades = [g.clone() for g in grades]
            for k in range(self.max_grade, 0, -1):
                if k == 1:
                    # M^{(1)}_new[j] += z_i[j] * M^{(0)}_old (a scalar).
                    new_grades[1] = new_grades[1] + zi_eff * grades[0]
                else:
                    # For each prev basis alpha and j not in alpha,
                    # add sign * z_i[j] * grades[k-1][alpha] to
                    # new_grades[k][target_basis[alpha, j]].
                    prev = grades[k - 1]  # (B, D_{k-1})
                    target_basis = getattr(self, f"target_basis_{k}")  # (D_{k-1}, r)
                    target_sign = getattr(self, f"target_sign_{k}")    # (D_{k-1}, r)
                    # Compute contributions: (B, D_{k-1}, r) of
                    # sign * zi_eff[:, j] * prev[:, alpha].
                    contrib = (
                        prev.unsqueeze(-1)
                        * zi_eff.unsqueeze(1)
                        * target_sign.unsqueeze(0)
                    )  # (B, D_{k-1}, r)
                    # Scatter to new_grades[k]: target index per (alpha, j).
                    # Where target_basis == -1 the contribution must be zero;
                    # target_sign is already 0 there, so contrib is 0.
                    target_basis_safe = target_basis.clamp(min=0).unsqueeze(0).expand(batch, -1, -1)
                    # Flatten contributions and accumulate via index_add.
                    flat_contrib = contrib.reshape(batch, -1)
                    flat_target = target_basis_safe.reshape(batch, -1)
                    update = torch.zeros_like(new_grades[k])
                    update.scatter_add_(1, flat_target, flat_contrib)
                    new_grades[k] = new_grades[k] + update
            grades = new_grades
        return grades

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        eps = 1.0e-6

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        token_input, occupancy = _piece_tokens(board)
        # Constrain z via tanh so the multivector products stay bounded.
        z = torch.tanh(self.z_proj(token_input))  # (B, 64, r)

        grades = self._exterior_pool(z, occupancy)  # list of (B, D_k)

        if self.ablation == "first_order_only":
            # Zero out grades >= 2 to recover a sum-pool behaviour.
            for k in range(2, self.max_grade + 1):
                grades[k] = torch.zeros_like(grades[k])
        elif self.ablation == "shuffle_grades_high" and batch > 1 and self.max_grade >= 2:
            perm = torch.randperm(batch, device=board.device)
            for k in range(2, self.max_grade + 1):
                grades[k] = grades[k][perm]

        # Concatenate all grades into a single readout vector.
        readout = torch.cat(grades, dim=-1)  # (B, D_R)
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

        # Per-grade magnitudes for diagnostics.
        grade_magnitudes = torch.stack(
            [g.pow(2).mean(dim=1).clamp_min(eps).sqrt() for g in grades],
            dim=1,
        )  # (B, max_grade + 1)
        active_count = occupancy.sum(dim=1)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "base_logit": base_logit,
            "primitive_delta": primitive_delta,
            "primitive_delta_raw": delta_raw,
            "primitive_gate": gate,
            "primitive_gate_logit": gate_logit,
            "primitive_gate_entropy": gate_entropy,
            "tepp_active_count": active_count,
            "tepp_grade_max_magnitude": grade_magnitudes.amax(dim=1),
            "tepp_grade_mean_magnitude": grade_magnitudes.mean(dim=1),
            "mechanism_energy": trunk_out["mechanism_energy"] + grade_magnitudes.mean(dim=1).detach(),
            "proposal_profile_strength": (primitive_delta.abs() * gate_entropy).clamp(0.0, 20.0),
            "proposal_keyword_count": logits.new_full((batch,), float(self.D_R)),
        }
        for k in range(self.max_grade + 1):
            diagnostics[f"tepp_grade_{k}_magnitude"] = grade_magnitudes[:, k]
        for key, value in trunk_out.items():
            if key == "logits":
                continue
            diag_key = (
                key if key.startswith("base_") or key in {"gate", "gate_logit"} else f"trunk_{key}"
            )
            diagnostics.setdefault(diag_key, value)
        return diagnostics


def build_truncated_exterior_product_pool_from_config(
    config: dict[str, Any],
) -> TruncatedExteriorProductPool:
    cfg = dict(config)
    return TruncatedExteriorProductPool(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        token_input_dim=cfg.get("token_input_dim", None),
        r=int(cfg.get("r", 4)),
        max_grade=int(cfg.get("max_grade", 3)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )
