"""Krylov Tactical Subspace Network for idea i076.

Implements the markdown architecture from
``ideas/research_packets/chess_nn_research_2026-04-25_2000_saturday_shanghai_krylov_tactical_subspace.md``.

For each board the model builds a chess-structured 64x64 linear operator
``A(X) = sum_g gate_g(X) * mask_g + low_rank_context_update`` whose
deterministic geometric pieces are king / knight / pawn / ray / defense
adjacency masks. Six role seed vectors ``v_r ∈ R^{64}`` (attack,
defense, king_zone, high_value_target, blocker, tempo) are read from
the per-square trunk features. For each role a differentiable modified
Gram-Schmidt Arnoldi block produces the Krylov basis ``Q_r`` and the
Hessenberg projection ``H_r``, from which the model reads:

  * Ritz / spectral statistics from ``H_r`` (singular values; non-symmetric
    Arnoldi makes raw eigenvalues complex, so we use SVD which is real,
    stable, and differentiable per the packet's "small projected H"
    recipe).
  * The Arnoldi residual norm ``||A Q_r - Q_r H_r||``.
  * Growth/decay curves ``||A^k v_r||``.
  * Krylov basis energy near king and high-value-target squares.
  * Cross-role principal-angle and Gram statistics
    ``Q_a^T Q_b`` between attack/defense, attack/king-zone, and
    attack/high-value-target subspaces.

The puzzle logit is an MLP over ``[per-role spectral features,
cross-role features, pooled board context]``. The model is strictly
board-only: nothing reads engine, source, verification, or CRTK
metadata.

Ablations (``model.ablation``) cover the packet's required table:
``one_step_only``, ``no_orthogonalization``, ``fixed_operator_only``,
``random_geometry_operator``, ``no_spectral_readout``,
``no_cross_role_angles``, ``cnn_same_params`` (the trainer-side
size-matched CNN baseline; the bespoke head only flips a marker so
prediction artefacts can record it), and ``none``.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models._packet_bespoke_base import (
    BoardConvStem,
    BoardTensorSpec,
    format_logits,
    require_board_tensor,
    side_to_move_field,
)


VALID_ABLATIONS: frozenset[str] = frozenset(
    {
        "none",
        "one_step_only",
        "no_orthogonalization",
        "fixed_operator_only",
        "random_geometry_operator",
        "no_spectral_readout",
        "no_cross_role_angles",
        "cnn_same_params",
    }
)


DEFAULT_ROLES: tuple[str, ...] = (
    "attack",
    "defense",
    "king_zone",
    "high_value_target",
    "blocker",
    "tempo",
)


# ---------------------------------------------------------------------
# Deterministic chess-geometry masks (64x64).
# ---------------------------------------------------------------------


def _square_index(rank: int, file: int) -> int:
    return rank * 8 + file


def _knight_mask() -> torch.Tensor:
    mask = torch.zeros(64, 64)
    deltas = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
    for r in range(8):
        for f in range(8):
            for dr, df in deltas:
                nr, nf = r + dr, f + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    mask[_square_index(r, f), _square_index(nr, nf)] = 1.0
    return mask


def _king_mask() -> torch.Tensor:
    mask = torch.zeros(64, 64)
    deltas = [(dr, df) for dr in (-1, 0, 1) for df in (-1, 0, 1) if not (dr == 0 and df == 0)]
    for r in range(8):
        for f in range(8):
            for dr, df in deltas:
                nr, nf = r + dr, f + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    mask[_square_index(r, f), _square_index(nr, nf)] = 1.0
    return mask


def _pawn_mask() -> torch.Tensor:
    mask = torch.zeros(64, 64)
    for r in range(8):
        for f in range(8):
            for dr in (-1, 1):
                for df in (-1, 1):
                    nr, nf = r + dr, f + df
                    if 0 <= nr < 8 and 0 <= nf < 8:
                        mask[_square_index(r, f), _square_index(nr, nf)] = 1.0
    return mask


def _ray_mask() -> torch.Tensor:
    """Rook + bishop full-line adjacency (no blockers)."""
    mask = torch.zeros(64, 64)
    for r in range(8):
        for f in range(8):
            i = _square_index(r, f)
            for dr, df in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nr, nf = r + dr, f + df
                while 0 <= nr < 8 and 0 <= nf < 8:
                    mask[i, _square_index(nr, nf)] = 1.0
                    nr += dr
                    nf += df
    return mask


def _defense_mask() -> torch.Tensor:
    """Same-line orthogonal adjacency (rook lines only) used as a defender-coverage prior."""
    mask = torch.zeros(64, 64)
    for r in range(8):
        for f in range(8):
            i = _square_index(r, f)
            for dr, df in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nf = r + dr, f + df
                while 0 <= nr < 8 and 0 <= nf < 8:
                    mask[i, _square_index(nr, nf)] = 1.0
                    nr += dr
                    nf += df
    return mask


GEOMETRY_NAMES: tuple[str, ...] = ("ray", "knight", "pawn", "king", "defense")


def _stack_geometry_masks() -> torch.Tensor:
    masks = [_ray_mask(), _knight_mask(), _pawn_mask(), _king_mask(), _defense_mask()]
    out = torch.stack(masks, dim=0)
    # Row-normalise each mask so repeated propagation does not blow up.
    row_sums = out.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return out / row_sums


# ---------------------------------------------------------------------
# Krylov block.
# ---------------------------------------------------------------------


class _ArnoldiBlock(nn.Module):
    """Modified Gram-Schmidt Arnoldi for ``m`` steps on batched 64x64 ``A``.

    Returns ``(Q, H, growth, residual)`` where ``Q`` is ``(B, 64, m)``,
    ``H`` is ``(B, m, m)`` upper-Hessenberg, ``growth`` is ``(B, m)`` of
    ``||A^k v_r||`` proxies (the cumulative norms ``h_{k+1, k}``-like
    quantities), and ``residual`` is the final ``h_{m, m-1}``.
    """

    def __init__(self, steps: int, orthogonalize: bool = True) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError("krylov_steps must be >= 1")
        self.steps = int(steps)
        self.orthogonalize = bool(orthogonalize)

    def forward(
        self,
        A: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # A: (B, 64, 64), v: (B, 64)
        batch, n = v.shape
        m = self.steps
        eps = 1.0e-8

        v0_norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
        q0 = v / v0_norm

        # Build Q and H as Python lists so we never alias views into the
        # autograd graph (in-place writes into ``Q[:, :, k]`` while later
        # reading ``Q[:, :, i]`` triggers stride-version mismatches).
        q_list: list[torch.Tensor] = [q0]
        h_columns: list[torch.Tensor] = []
        growth_list: list[torch.Tensor] = [v0_norm.squeeze(-1)]
        last_residual = v.new_zeros(batch)
        for k in range(m - 1):
            w = torch.bmm(A, q_list[-1].unsqueeze(-1)).squeeze(-1)  # (B, 64)
            col_entries = []
            if self.orthogonalize:
                for i in range(k + 1):
                    qi = q_list[i]
                    coef = (qi * w).sum(dim=-1)
                    col_entries.append(coef)
                    w = w - coef.unsqueeze(-1) * qi
            else:
                for i in range(k + 1):
                    qi = q_list[i]
                    coef = (qi * w).sum(dim=-1)
                    col_entries.append(coef)
            w_norm = w.norm(dim=-1).clamp_min(eps)
            col_entries.append(w_norm)
            # Pad to m so all columns can be stacked into the (B, m, m) H.
            while len(col_entries) < m:
                col_entries.append(v.new_zeros(batch))
            h_columns.append(torch.stack(col_entries, dim=-1))
            growth_list.append(w_norm)
            q_list.append(w / w_norm.unsqueeze(-1))
            last_residual = w_norm

        # Final column is undefined (no further A q_{m-1} step recorded).
        # Pad with zeros so H has shape (B, m, m).
        while len(h_columns) < m:
            h_columns.append(v.new_zeros(batch, m))
        # h_columns[k] is the k-th column of H -> stack as (B, m, m).
        H = torch.stack(h_columns, dim=-1)
        Q = torch.stack(q_list, dim=-1)  # (B, 64, m)
        growth = torch.stack(growth_list, dim=-1)  # (B, m)

        return Q, H, growth, last_residual


# ---------------------------------------------------------------------
# Main module.
# ---------------------------------------------------------------------


class KrylovTacticalSubspaceNetwork(nn.Module):
    """Bespoke implementation of the Krylov Tactical Subspace Network.

    Forward output dict (board-only inputs):
      - ``logits``: ``(B,)`` puzzle logit for the BCE-with-logits trainer.
      - ``operator_norm``: estimated spectral-norm proxy of ``A(X)``.
      - ``operator_gate_weights``: ``(B, num_geometry)`` softplus gates
        over the fixed chess-geometry masks.
      - ``operator_low_rank_energy``: Frobenius mass of the low-rank
        context update.
      - ``role_growth_curves``: ``(B, num_roles, m)`` ``||A^k v_r||``.
      - ``role_residual_norms``: ``(B, num_roles)`` Arnoldi residuals.
      - ``role_ritz_singular_values``: ``(B, num_roles, m)`` SVs of the
        small Hessenberg ``H_r``.
      - ``role_basis_king_energy``: ``(B, num_roles)`` Q-basis energy
        near the side-to-move king.
      - ``role_basis_target_energy``: ``(B, num_roles)`` Q-basis energy
        near opposing high-value pieces.
      - ``cross_role_principal_angles``: ``(B, num_cross_pairs, m)``
        cosines of principal angles ``cos(theta) = svd(Q_a^T Q_b)``.
      - ``cross_role_gram_frobenius``: ``(B, num_cross_pairs)`` Frobenius
        norms of ``Q_a^T Q_b``.
      - ``ablation_*`` scalar flags.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        hidden_dim: int = 96,
        operator_rank: int = 16,
        krylov_steps: int = 6,
        roles: Sequence[str] = DEFAULT_ROLES,
        head_hidden: int | None = None,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "KrylovTacticalSubspaceNetwork implements the puzzle_binary single-logit contract only"
            )
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if krylov_steps < 2:
            raise ValueError("krylov_steps must be >= 2 to define a non-degenerate Krylov subspace")
        if operator_rank < 1:
            raise ValueError("operator_rank must be >= 1")
        if ablation not in VALID_ABLATIONS:
            raise ValueError(
                f"Unknown ablation {ablation!r}; expected one of {sorted(VALID_ABLATIONS)}"
            )
        roles = tuple(roles)
        if len(roles) < 2:
            raise ValueError("at least two roles are required to build cross-role features")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.operator_rank = int(operator_rank)
        self.krylov_steps = int(krylov_steps)
        self.roles = roles
        self.dropout = float(dropout)
        self.ablation = str(ablation)
        head_hidden_dim = int(head_hidden if head_hidden is not None else hidden_dim)

        # Board trunk over (B, C, 8, 8).
        self.stem = BoardConvStem(
            input_channels=input_channels,
            channels=self.channels,
            depth=int(depth),
            use_batchnorm=use_batchnorm,
        )

        # Fixed geometry masks (5, 64, 64), buffered so they ride GPU/CPU
        # moves automatically and never receive gradients.
        self.register_buffer("_geometry_masks", _stack_geometry_masks(), persistent=False)
        self.num_geometry = self._geometry_masks.shape[0]

        # Random-geometry-operator ablation: a fixed but random "non-chess"
        # alternative shaped like the geometry tensor.
        rng = torch.Generator().manual_seed(0xC4ED5)
        random_masks = torch.rand(self.num_geometry, 64, 64, generator=rng)
        random_masks = random_masks / random_masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("_random_geometry_masks", random_masks, persistent=False)

        # Operator builder: gate logits and low-rank update from pooled board.
        self.gate_head = nn.Sequential(
            nn.Linear(self.channels * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_geometry),
        )
        self.low_rank_left = nn.Linear(self.channels, self.operator_rank)
        self.low_rank_right = nn.Linear(self.channels, self.operator_rank)

        # Role seed builder: per-square (B, 64, num_roles) scalars.
        self.role_seed_head = nn.Linear(self.channels, len(self.roles))

        # Krylov block (one shared block, called per role).
        self.krylov = _ArnoldiBlock(steps=self.krylov_steps, orthogonalize=True)
        self.krylov_no_ortho = _ArnoldiBlock(steps=self.krylov_steps, orthogonalize=False)

        # Cross-role pairs we expose: attack/defense, attack/king_zone,
        # attack/high_value_target, defense/king_zone (when present).
        self.cross_pairs = self._build_cross_pairs(roles)

        # Final head input width.
        per_role_features = (
            self.krylov_steps  # ritz singular values
            + self.krylov_steps  # growth curve
            + 1  # residual norm
            + 2  # basis king + target energy
        )
        per_pair_features = self.krylov_steps + 1  # principal angles + frobenius norm
        head_in = (
            self.channels * 2  # pooled board context (mean + max)
            + per_role_features * len(self.roles)
            + per_pair_features * len(self.cross_pairs)
            + self.num_geometry  # operator gate weights
            + 2  # operator_norm + low rank energy
        )
        self.head_norm = nn.LayerNorm(head_in)
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cross_pairs(roles: Sequence[str]) -> tuple[tuple[int, int, str], ...]:
        index = {name: i for i, name in enumerate(roles)}
        candidates = [
            ("attack", "defense"),
            ("attack", "king_zone"),
            ("attack", "high_value_target"),
            ("defense", "king_zone"),
        ]
        pairs: list[tuple[int, int, str]] = []
        for a, b in candidates:
            if a in index and b in index:
                pairs.append((index[a], index[b], f"{a}__{b}"))
        if not pairs:
            # Fall back to consecutive pairs so the head has at least one
            # cross-role feature.
            pairs.append((0, 1, f"{roles[0]}__{roles[1]}"))
        return tuple(pairs)

    def _board_features(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        feats = self.stem(x)  # (B, C, 8, 8)
        return feats

    @staticmethod
    def _flatten_squares(feats: torch.Tensor) -> torch.Tensor:
        # (B, C, 8, 8) -> (B, 64, C)
        return feats.flatten(2).transpose(1, 2)

    @staticmethod
    def _pool(feats: torch.Tensor) -> torch.Tensor:
        mean_pool = feats.mean(dim=(2, 3))
        max_pool = feats.amax(dim=(2, 3))
        return torch.cat([mean_pool, max_pool], dim=1)

    @staticmethod
    def _king_target_masks(x: torch.Tensor, input_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (king_mask, target_mask) ∈ ``(B, 64)`` from ``simple_18``.

        The "us-king" mask follows side-to-move so the same diagnostic is
        computed regardless of board orientation. Targets are opposing
        non-pawn high-value pieces (queen, rook, bishop, knight).
        """
        if input_channels < 13:
            zeros = x.new_zeros(x.shape[0], 64)
            return zeros, zeros
        side = side_to_move_field(x, input_channels).squeeze(1)  # (B, 8, 8)
        white = x[:, 0:6].clamp(0.0, 1.0)
        black = x[:, 6:12].clamp(0.0, 1.0)
        # Plane order in simple_18: 0=PAWN..5=KING for white, then black.
        us_king = side * white[:, 5] + (1.0 - side) * black[:, 5]
        them_high_value = (
            side * (black[:, 1] + black[:, 2] + black[:, 3] + black[:, 4])
            + (1.0 - side) * (white[:, 1] + white[:, 2] + white[:, 3] + white[:, 4])
        )
        return us_king.flatten(1), them_high_value.flatten(1)

    def _operator(
        self,
        squares: torch.Tensor,
        pooled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Gate logits and softplus weights.
        gate_logits = self.gate_head(pooled)
        if self.ablation == "fixed_operator_only":
            gate_weights = torch.ones_like(gate_logits) / float(self.num_geometry)
        else:
            gate_weights = F.softplus(gate_logits)
            gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

        if self.ablation == "random_geometry_operator":
            masks = self._random_geometry_masks
        else:
            masks = self._geometry_masks
        # gate_weights: (B, G); masks: (G, 64, 64); A_geom: (B, 64, 64)
        A_geom = torch.einsum("bg,gij->bij", gate_weights, masks)

        if self.ablation == "fixed_operator_only":
            low_rank = squares.new_zeros(squares.shape[0], 64, 64)
            low_rank_energy = squares.new_zeros(squares.shape[0])
        else:
            U = self.low_rank_left(squares)  # (B, 64, r)
            V = self.low_rank_right(squares)  # (B, 64, r)
            low_rank = torch.bmm(U, V.transpose(1, 2))  # (B, 64, 64)
            low_rank_energy = low_rank.flatten(1).norm(dim=-1)

        A = A_geom + low_rank
        # Spectral-norm proxy: divide by max(1, frobenius_norm / 8). This is
        # a cheap stability bound (||A||_2 <= ||A||_F) that the packet
        # explicitly recommends to keep ``A`` non-explosive.
        operator_norm_proxy = A.flatten(1).norm(dim=-1) / 8.0
        scale = torch.clamp_min(operator_norm_proxy, 1.0).unsqueeze(-1).unsqueeze(-1)
        A_normed = A / scale
        return A_normed, gate_weights, low_rank_energy, operator_norm_proxy

    def _role_seeds(self, squares: torch.Tensor) -> torch.Tensor:
        # squares: (B, 64, C) -> (B, num_roles, 64)
        per_square_logits = self.role_seed_head(squares)  # (B, 64, R)
        seeds = per_square_logits.transpose(1, 2)  # (B, R, 64)
        # L2-normalise each seed so the Krylov growth curve starts at unit norm.
        norms = seeds.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return seeds / norms

    @staticmethod
    def _hessenberg_singular_values(H: torch.Tensor, m: int) -> torch.Tensor:
        # H: (B, m, m). Use SVD for stability with non-symmetric H.
        # Add a tiny regulariser so degenerate H still differentiates.
        regularised = H + 1.0e-8 * torch.eye(m, device=H.device, dtype=H.dtype).unsqueeze(0)
        svdvals = torch.linalg.svdvals(regularised)
        return svdvals  # (B, m)

    @staticmethod
    def _basis_energy(Q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Q: (B, 64, m), mask: (B, 64) -> (B,) summed Q^2 weighted by mask, averaged over m.
        if mask.sum(dim=-1).max() == 0:
            return Q.new_zeros(Q.shape[0])
        weighted = (Q * Q).sum(dim=-1)  # (B, 64) sum over m
        denom = Q.shape[-1]
        return (weighted * mask).sum(dim=-1) / float(denom)

    @staticmethod
    def _principal_angles(Q_a: torch.Tensor, Q_b: torch.Tensor, m: int) -> torch.Tensor:
        # Q_a, Q_b: (B, 64, m). Cross Gram (B, m, m).
        gram = torch.bmm(Q_a.transpose(1, 2), Q_b)
        regularised = gram + 1.0e-8 * torch.eye(m, device=gram.device, dtype=gram.dtype).unsqueeze(0)
        return torch.linalg.svdvals(regularised)  # (B, m)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self._board_features(x)
        squares = self._flatten_squares(feats)  # (B, 64, C)
        pooled = self._pool(feats)  # (B, 2C)
        batch = squares.shape[0]
        m = self.krylov_steps

        A, gate_weights, low_rank_energy, operator_norm = self._operator(squares, pooled)
        seeds = self._role_seeds(squares)  # (B, R, 64)

        king_mask, target_mask = self._king_target_masks(x, self.input_channels)

        krylov_block = (
            self.krylov_no_ortho if self.ablation == "no_orthogonalization" else self.krylov
        )

        ritz = []  # (B, R, m)
        growth = []  # (B, R, m)
        residuals = []  # (B, R)
        king_energy = []  # (B, R)
        target_energy = []  # (B, R)
        bases: list[torch.Tensor] = []
        for r in range(seeds.shape[1]):
            v_r = seeds[:, r, :]
            if self.ablation == "one_step_only":
                # Replace the iterative block with a single power so the
                # spectral readout sees only first-order information.
                w = torch.bmm(A, v_r.unsqueeze(-1)).squeeze(-1)
                w_norm = w.norm(dim=-1).clamp_min(1.0e-8)
                Q = v_r.new_zeros(batch, 64, m)
                Q[:, :, 0] = v_r
                Q[:, :, 1] = w / w_norm.unsqueeze(-1)
                H = v_r.new_zeros(batch, m, m)
                H[:, 1, 0] = w_norm
                growth_r = v_r.new_zeros(batch, m)
                growth_r[:, 0] = 1.0
                growth_r[:, 1] = w_norm
                residual_r = w_norm
            else:
                Q, H, growth_r, residual_r = krylov_block(A, v_r)

            bases.append(Q)
            growth.append(growth_r)
            residuals.append(residual_r)
            ritz.append(self._hessenberg_singular_values(H, m))
            king_energy.append(self._basis_energy(Q, king_mask))
            target_energy.append(self._basis_energy(Q, target_mask))

        role_growth = torch.stack(growth, dim=1)
        role_ritz = torch.stack(ritz, dim=1)
        role_residual = torch.stack(residuals, dim=1)
        role_king_energy = torch.stack(king_energy, dim=1)
        role_target_energy = torch.stack(target_energy, dim=1)

        if self.ablation == "no_spectral_readout":
            role_ritz = torch.zeros_like(role_ritz)

        cross_angles_list: list[torch.Tensor] = []
        cross_frobenius_list: list[torch.Tensor] = []
        for a_idx, b_idx, _name in self.cross_pairs:
            angles = self._principal_angles(bases[a_idx], bases[b_idx], m)
            gram = torch.bmm(bases[a_idx].transpose(1, 2), bases[b_idx])
            frob = gram.flatten(1).norm(dim=-1)
            cross_angles_list.append(angles)
            cross_frobenius_list.append(frob)
        cross_angles = torch.stack(cross_angles_list, dim=1)  # (B, P, m)
        cross_frobenius = torch.stack(cross_frobenius_list, dim=1)  # (B, P)

        if self.ablation == "no_cross_role_angles":
            cross_angles = torch.zeros_like(cross_angles)
            cross_frobenius = torch.zeros_like(cross_frobenius)

        head_input = torch.cat(
            [
                pooled,
                role_ritz.flatten(1),
                role_growth.flatten(1),
                role_residual,
                role_king_energy,
                role_target_energy,
                cross_angles.flatten(1),
                cross_frobenius,
                gate_weights,
                operator_norm.unsqueeze(-1),
                low_rank_energy.unsqueeze(-1),
            ],
            dim=-1,
        )
        head_input = self.head_norm(head_input)
        puzzle_logit = self.head(head_input).squeeze(-1)

        ones = puzzle_logit.new_ones(batch)
        ablation_flag = lambda name: ones * (1.0 if self.ablation == name else 0.0)

        output: dict[str, torch.Tensor] = {
            "logits": format_logits(puzzle_logit.unsqueeze(-1), self.num_classes),
            "operator_norm": operator_norm,
            "operator_gate_weights": gate_weights,
            "operator_low_rank_energy": low_rank_energy,
            "role_growth_curves": role_growth,
            "role_residual_norms": role_residual,
            "role_ritz_singular_values": role_ritz,
            "role_basis_king_energy": role_king_energy,
            "role_basis_target_energy": role_target_energy,
            "cross_role_principal_angles": cross_angles,
            "cross_role_gram_frobenius": cross_frobenius,
            "ablation_one_step_only": ablation_flag("one_step_only"),
            "ablation_no_orthogonalization": ablation_flag("no_orthogonalization"),
            "ablation_fixed_operator_only": ablation_flag("fixed_operator_only"),
            "ablation_random_geometry_operator": ablation_flag("random_geometry_operator"),
            "ablation_no_spectral_readout": ablation_flag("no_spectral_readout"),
            "ablation_no_cross_role_angles": ablation_flag("no_cross_role_angles"),
            "ablation_cnn_same_params": ablation_flag("cnn_same_params"),
        }
        return output


def build_krylov_tactical_subspace_network_from_config(
    config: dict[str, Any],
) -> KrylovTacticalSubspaceNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    roles = cfg.pop("roles", DEFAULT_ROLES)
    if isinstance(roles, str):
        roles = tuple(part.strip() for part in roles.split(",") if part.strip())
    else:
        roles = tuple(roles)
    return KrylovTacticalSubspaceNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        depth=int(cfg.pop("depth", 2)),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        operator_rank=int(cfg.pop("operator_rank", 16)),
        krylov_steps=int(cfg.pop("krylov_steps", 6)),
        roles=roles,
        head_hidden=cfg.pop("head_hidden", None),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        ablation=str(cfg.pop("ablation", "none")),
    )


__all__ = [
    "DEFAULT_ROLES",
    "GEOMETRY_NAMES",
    "KrylovTacticalSubspaceNetwork",
    "VALID_ABLATIONS",
    "build_krylov_tactical_subspace_network_from_config",
]
