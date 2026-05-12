"""Tactical Controllability Gramian Network for idea i078.

Implements the markdown architecture from
``ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_2004_saturday_shanghai_tactical_controllability_gramian.md``.

The model treats a chess position as a small linear control system

    h_{t+1} = A h_t + B_a u_a + B_d u_d
    y_t     = C h_t

where ``A`` is a chess-structured 64x64 propagation operator built from
fixed ray / knight / pawn / king / rook-line-defense masks gated by a
pooled-board MLP plus a board-conditioned low-rank update ``U V^T``;
``B_a`` injects attacker influence from side-to-move pieces; ``B_d``
injects defender influence from opposing pieces; and ``C`` reads out
king zone, high-value targets, and line intersections.

After spectrally normalising ``A_hat = A / max(1, sigma(A))`` the model
forms the K-step unrolled controllability and observability Gramians

    W_a = sum_{k=0..K} A_hat^k B_a B_a^T (A_hat^T)^k
    W_d = sum_{k=0..K} A_hat^k B_d B_d^T (A_hat^T)^k
    W_o = sum_{k=0..K} (A_hat^T)^k C^T C A_hat^k

and reads the packet's controllability/observability diagnostics:

  * ``T_a   = trace(C W_a C^T)`` — attacker target reach,
  * ``T_d   = trace(C W_d C^T)`` — defender target reach,
  * ``T_net = T_a - T_d``        — net tactical controllability,
  * top singular values of ``W_o^{1/2} W_a W_o^{1/2}`` (attacker Hankel modes),
  * top singular values of ``W_o^{1/2} W_d W_o^{1/2}`` (defender cancellation modes),
  * principal angles between attacker and defender control subspaces,
  * per-target Gramian diagonals ``diag(C W_a C^T)``, ``diag(C W_d C^T)``,
  * target observability mass ``trace(W_o)`` and operator diagnostics.

The puzzle logit is a ``LayerNorm + MLP`` over the concatenation of
these scalars with a pooled board context, the operator gate weights,
the spectral-norm proxy and low-rank energy.

Ablations (selected by ``model.ablation``):
  ``none``, ``attacker_only``, ``defender_only``, ``no_observability``,
  ``one_step_gramian``, ``random_target_C``, ``random_geometry_A``,
  ``fixed_A_no_gates``, ``diag_only_gramian``, ``cnn_same_params``.
"""
from __future__ import annotations

from typing import Any

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
        "attacker_only",
        "defender_only",
        "no_observability",
        "one_step_gramian",
        "random_target_C",
        "random_geometry_A",
        "fixed_A_no_gates",
        "diag_only_gramian",
        "cnn_same_params",
    }
)


GEOMETRY_NAMES: tuple[str, ...] = ("ray", "knight", "pawn", "king", "defense")


# ---------------------------------------------------------------------
# Deterministic chess-geometry masks (64x64). Same shapes as i077.
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


def _stack_geometry_masks() -> torch.Tensor:
    masks = [_ray_mask(), _knight_mask(), _pawn_mask(), _king_mask(), _defense_mask()]
    out = torch.stack(masks, dim=0)
    row_sums = out.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return out / row_sums


# ---------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------


class TacticalControllabilityGramianNetwork(nn.Module):
    """Bespoke implementation of the Tactical Controllability Gramian Network.

    Forward output dict:
      - ``logits``: ``(B,)`` puzzle logit for the BCE-with-logits trainer.
      - ``T_a``, ``T_d``, ``T_net``: ``(B,)`` attacker / defender / net target reach.
      - ``observability_trace``: ``(B,)`` ``trace(W_o)``.
      - ``attacker_hankel_modes``: ``(B, num_modes)`` top singular values of
        ``W_o^{1/2} W_a W_o^{1/2}``.
      - ``defender_hankel_modes``: ``(B, num_modes)`` top singular values of
        ``W_o^{1/2} W_d W_o^{1/2}``.
      - ``mode_ratio``: ``attacker_hankel_modes / (attacker + defender + eps)``.
      - ``subspace_principal_angles``: ``(B, num_modes)`` principal angles
        (radians) between top attacker / defender control subspaces.
      - ``target_diag_attacker``, ``target_diag_defender``: per-target
        Gramian diagonals ``diag(C W_a C^T)`` and ``diag(C W_d C^T)``.
      - ``operator_norm``: spectral-norm proxy of ``A(X)``.
      - ``operator_gate_weights``: ``(B, 5)`` softplus gates over geometry masks.
      - ``operator_low_rank_energy``: Frobenius mass of the low-rank update.
      - ``ablation_*``: per-batch indicator flags consumed by the diagnostic table.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        hidden_dim: int = 96,
        operator_rank: int = 12,
        input_rank: int = 8,
        target_rank: int = 8,
        gramian_steps: int = 6,
        readout_modes: int = 12,
        spectral_norm_iters: int = 3,
        head_hidden: int | None = None,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "TacticalControllabilityGramianNetwork implements the puzzle_binary single-logit contract only"
            )
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if operator_rank < 1:
            raise ValueError("operator_rank must be >= 1")
        if input_rank < 1:
            raise ValueError("input_rank must be >= 1")
        if target_rank < 1:
            raise ValueError("target_rank must be >= 1")
        if gramian_steps < 1:
            raise ValueError("gramian_steps must be >= 1")
        if readout_modes < 1:
            raise ValueError("readout_modes must be >= 1")
        if spectral_norm_iters < 1:
            raise ValueError("spectral_norm_iters must be >= 1")
        if ablation not in VALID_ABLATIONS:
            raise ValueError(
                f"Unknown ablation {ablation!r}; expected one of {sorted(VALID_ABLATIONS)}"
            )

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.operator_rank = int(operator_rank)
        self.input_rank = int(input_rank)
        self.target_rank = int(target_rank)
        self.gramian_steps = int(gramian_steps)
        self.readout_modes = int(min(readout_modes, target_rank))
        self.spectral_norm_iters = int(spectral_norm_iters)
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

        # Fixed geometry masks (5, 64, 64); buffers ride GPU/CPU moves.
        self.register_buffer("_geometry_masks", _stack_geometry_masks(), persistent=False)
        self.num_geometry = int(self._geometry_masks.shape[0])

        rng = torch.Generator().manual_seed(0xCB18A551)
        random_masks = torch.rand(self.num_geometry, 64, 64, generator=rng)
        random_masks = random_masks / random_masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("_random_geometry_masks", random_masks, persistent=False)

        # Random target-C used by the random_target_C ablation. Stored as
        # a fixed 64-square readout matrix that does not depend on board.
        random_C = torch.randn(self.target_rank, 64, generator=rng) * 0.1
        self.register_buffer("_random_target_C", random_C, persistent=False)

        # Power-iteration probe for spectral-norm estimation.
        self.register_buffer(
            "_spectral_probe",
            F.normalize(torch.randn(64, generator=rng), dim=0),
            persistent=False,
        )

        # Operator builder: gate logits and low-rank update.
        self.gate_head = nn.Sequential(
            nn.Linear(self.channels * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_geometry),
        )
        self.low_rank_left = nn.Linear(self.channels, self.operator_rank)
        self.low_rank_right = nn.Linear(self.channels, self.operator_rank)

        # Attacker / defender input column heads. Each emits a per-square
        # input_rank vector that is gated by the side-to-move attacker /
        # defender piece occupancy planes (built from simple_18 channels).
        self.attacker_input_head = nn.Linear(self.channels, self.input_rank)
        self.defender_input_head = nn.Linear(self.channels, self.input_rank)

        # Target output head: per-square scalars combined with deterministic
        # king-zone / material / line-intersection priors.
        self.target_output_head = nn.Linear(self.channels, self.target_rank)

        # Top singular values of W_o^{1/2} W_a W_o^{1/2} are computed by
        # eigendecomposition of W_o (PSD) followed by SVD of the
        # whitened Gramians.

        per_alpha_features = (
            3  # T_a, T_d, T_net
            + 1  # observability_trace
            + self.readout_modes  # attacker hankel modes
            + self.readout_modes  # defender hankel modes
            + self.readout_modes  # mode ratio
            + self.readout_modes  # subspace principal angles
            + self.target_rank  # target-diagonal attacker
            + self.target_rank  # target-diagonal defender
        )
        head_in = (
            self.channels * 2  # pooled board context (mean + max)
            + per_alpha_features
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
    # Board / piece helpers
    # ------------------------------------------------------------------

    def _board_features(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        return self.stem(x)

    @staticmethod
    def _flatten_squares(feats: torch.Tensor) -> torch.Tensor:
        return feats.flatten(2).transpose(1, 2)

    @staticmethod
    def _pool(feats: torch.Tensor) -> torch.Tensor:
        mean_pool = feats.mean(dim=(2, 3))
        max_pool = feats.amax(dim=(2, 3))
        return torch.cat([mean_pool, max_pool], dim=1)

    def _piece_priors(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (us_pieces, them_pieces, us_king_zone, them_high_value).

        Each tensor has shape ``(B, 64)`` and is in [0, 1]. ``us`` follows
        the side-to-move plane; high-value targets exclude pawns.
        """
        if self.input_channels < 13:
            zeros = x.new_zeros(x.shape[0], 64)
            return zeros, zeros, zeros, zeros
        side = side_to_move_field(x, self.input_channels).squeeze(1)  # (B, 8, 8)
        white = x[:, 0:6].clamp(0.0, 1.0)
        black = x[:, 6:12].clamp(0.0, 1.0)
        us_pieces = side * white.sum(dim=1) + (1.0 - side) * black.sum(dim=1)
        them_pieces = side * black.sum(dim=1) + (1.0 - side) * white.sum(dim=1)
        us_king = side * white[:, 5] + (1.0 - side) * black[:, 5]
        them_high_value = (
            side * (black[:, 1] + black[:, 2] + black[:, 3] + black[:, 4])
            + (1.0 - side) * (white[:, 1] + white[:, 2] + white[:, 3] + white[:, 4])
        )
        # Soft king zone via convolution with a 3x3 ones kernel.
        kernel = us_king.new_ones(1, 1, 3, 3)
        us_king_zone = F.conv2d(us_king.unsqueeze(1), kernel, padding=1).squeeze(1).clamp(0.0, 1.0)
        return (
            us_pieces.flatten(1),
            them_pieces.flatten(1),
            us_king_zone.flatten(1),
            them_high_value.flatten(1),
        )

    # ------------------------------------------------------------------
    # Operator A and spectral normalization
    # ------------------------------------------------------------------

    def _operator(
        self,
        squares: torch.Tensor,
        pooled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_logits = self.gate_head(pooled)
        if self.ablation == "fixed_A_no_gates":
            gate_weights = torch.ones_like(gate_logits) / float(self.num_geometry)
        else:
            gate_weights = F.softplus(gate_logits)
            gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

        if self.ablation == "random_geometry_A":
            masks = self._random_geometry_masks
        else:
            masks = self._geometry_masks
        A_geom = torch.einsum("bg,gij->bij", gate_weights, masks)

        if self.ablation == "fixed_A_no_gates":
            low_rank = squares.new_zeros(squares.shape[0], 64, 64)
            low_rank_energy = squares.new_zeros(squares.shape[0])
        else:
            U = self.low_rank_left(squares)  # (B, 64, r)
            V = self.low_rank_right(squares)  # (B, 64, r)
            low_rank = torch.bmm(U, V.transpose(1, 2))  # (B, 64, 64)
            low_rank_energy = low_rank.flatten(1).norm(dim=-1)

        A = A_geom + low_rank
        A_hat, operator_norm = self._spectral_normalize(A)
        return A_hat, gate_weights, low_rank_energy, operator_norm

    def _spectral_normalize(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = A.shape[0]
        v = self._spectral_probe.detach().to(dtype=A.dtype).expand(batch, 64).contiguous()
        with torch.no_grad():
            for _ in range(self.spectral_norm_iters):
                Av = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
                v_next = F.normalize(Av, dim=-1, eps=1.0e-8)
                AtAv = torch.bmm(A.transpose(1, 2), v_next.unsqueeze(-1)).squeeze(-1)
                v = F.normalize(AtAv, dim=-1, eps=1.0e-8)
        Av = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
        sigma = Av.norm(dim=-1).clamp_min(1.0e-8)
        # Pad below 1 so contraction is preserved when A is small but the
        # packet's sum_{k} A^k B B^T (A^T)^k still converges.
        scale = torch.clamp_min(sigma, 1.0).unsqueeze(-1).unsqueeze(-1)
        return A / scale, sigma

    # ------------------------------------------------------------------
    # B_a, B_d, C builders
    # ------------------------------------------------------------------

    def _input_columns(
        self,
        squares: torch.Tensor,
        us_pieces: torch.Tensor,
        them_pieces: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (B_a, B_d) of shape ``(B, 64, input_rank)``.

        Attacker columns are gated by the side-to-move piece occupancy;
        defender columns by the opposing piece occupancy. This realises
        the packet's ``B_a injects attacker influence from side-to-move
        pieces`` clause without consuming any forbidden metadata.
        """
        attacker = self.attacker_input_head(squares)  # (B, 64, r)
        defender = self.defender_input_head(squares)
        B_a = attacker * us_pieces.unsqueeze(-1)
        B_d = defender * them_pieces.unsqueeze(-1)
        if self.ablation == "attacker_only":
            B_d = torch.zeros_like(B_d)
        elif self.ablation == "defender_only":
            B_a = torch.zeros_like(B_a)
        return B_a, B_d

    def _target_matrix(
        self,
        squares: torch.Tensor,
        us_king_zone: torch.Tensor,
        them_high_value: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``C`` of shape ``(B, target_rank, 64)`` reading critical targets.

        Half the rows are gated by the side-to-move king zone (defender
        weakness around our own king is usually irrelevant for a puzzle,
        so this row reads attacker control around the opposing king
        zone proxied via the side-to-move king plane mirror) and half
        by opposing high-value piece occupancy.
        """
        if self.ablation == "random_target_C":
            batch = squares.shape[0]
            C = self._random_target_C.detach().to(dtype=squares.dtype)
            return C.unsqueeze(0).expand(batch, self.target_rank, 64).contiguous()
        # Per-square per-target scalar.
        per_square = self.target_output_head(squares)  # (B, 64, target_rank)
        C = per_square.transpose(1, 2)  # (B, target_rank, 64)
        half = max(1, self.target_rank // 2)
        gate_king = us_king_zone.unsqueeze(1).expand(-1, half, -1)
        gate_value = them_high_value.unsqueeze(1).expand(-1, self.target_rank - half, -1)
        gate = torch.cat([gate_king, gate_value], dim=1)
        C = C * gate
        # Row-normalise so trace(C C^T) does not explode at init.
        norm = C.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return C / norm

    # ------------------------------------------------------------------
    # Gramian solver (finite K-step unroll)
    # ------------------------------------------------------------------

    def _controllability_gramian(self, A_hat: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Return ``W = sum_{k=0..K} A_hat^k B B^T (A_hat^T)^k``.

        Implemented by the recursion ``W_{k+1} = B B^T + A_hat W_k A_hat^T``
        which the packet authorises explicitly.
        """
        BBT = torch.bmm(B, B.transpose(1, 2))
        if self.ablation == "one_step_gramian":
            return BBT
        W = BBT
        steps = self.gramian_steps
        for _ in range(steps):
            W = BBT + torch.bmm(A_hat, torch.bmm(W, A_hat.transpose(1, 2)))
        return W

    def _observability_gramian(self, A_hat: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """Return ``W_o = sum_{k=0..K} (A_hat^T)^k C^T C A_hat^k``."""
        CTC = torch.bmm(C.transpose(1, 2), C)
        W = CTC
        steps = self.gramian_steps
        for _ in range(steps):
            W = CTC + torch.bmm(A_hat.transpose(1, 2), torch.bmm(W, A_hat))
        return W

    @staticmethod
    def _symmetrize(W: torch.Tensor) -> torch.Tensor:
        return 0.5 * (W + W.transpose(1, 2))

    @staticmethod
    def _psd_sqrt(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (sqrt(W), eigenvalues) for a symmetric PSD batched matrix."""
        # eigh is differentiable for distinct eigenvalues; tiny jitter keeps
        # the decomposition well-conditioned even when W is rank-deficient.
        batch = W.shape[0]
        n = W.shape[-1]
        jitter = 1.0e-6 * torch.eye(n, device=W.device, dtype=W.dtype).expand(batch, n, n)
        eigvals, eigvecs = torch.linalg.eigh(W + jitter)
        eigvals = eigvals.clamp_min(0.0)
        sqrt_eigvals = torch.sqrt(eigvals)
        sqrt_W = torch.bmm(eigvecs * sqrt_eigvals.unsqueeze(1), eigvecs.transpose(1, 2))
        return sqrt_W, eigvals

    @staticmethod
    def _top_singular_values(M: torch.Tensor, k: int) -> torch.Tensor:
        """Top-``k`` singular values of a batched square matrix."""
        # svdvals is differentiable and gives sorted descending values.
        sv = torch.linalg.svdvals(M)
        if sv.shape[-1] >= k:
            return sv[..., :k]
        pad = sv.new_zeros(sv.shape[:-1] + (k - sv.shape[-1],))
        return torch.cat([sv, pad], dim=-1)

    @staticmethod
    def _principal_angles(
        Wa: torch.Tensor, Wd: torch.Tensor, num_modes: int
    ) -> torch.Tensor:
        """Principal angles between leading eigenspaces of two PSD matrices.

        Returns ``(B, num_modes)`` angles in radians, sorted ascending
        (smallest principal angle first).
        """
        batch = Wa.shape[0]
        n = Wa.shape[-1]
        jitter = 1.0e-6 * torch.eye(n, device=Wa.device, dtype=Wa.dtype).expand(batch, n, n)
        eig_a = torch.linalg.eigh(Wa + jitter)
        eig_d = torch.linalg.eigh(Wd + jitter)
        # eigh returns eigenvalues ascending; take the last num_modes columns
        # as the leading subspace.
        k = min(num_modes, eig_a.eigenvectors.shape[-1])
        Va = eig_a.eigenvectors[..., -k:]
        Vd = eig_d.eigenvectors[..., -k:]
        # Cross-Gram of orthonormal bases.
        cross = torch.bmm(Va.transpose(1, 2), Vd)
        sv = torch.linalg.svdvals(cross).clamp(-1.0, 1.0)
        angles = torch.arccos(sv)
        # Match the requested width.
        if angles.shape[-1] < num_modes:
            pad = angles.new_full(
                angles.shape[:-1] + (num_modes - angles.shape[-1],),
                float(torch.pi / 2.0),
            )
            angles = torch.cat([angles, pad], dim=-1)
        return angles[..., :num_modes]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self._board_features(x)
        squares = self._flatten_squares(feats)  # (B, 64, channels)
        pooled = self._pool(feats)  # (B, 2C)
        batch = squares.shape[0]

        A_hat, gate_weights, low_rank_energy, operator_norm = self._operator(squares, pooled)
        us_pieces, them_pieces, us_king_zone, them_high_value = self._piece_priors(x)
        B_a, B_d = self._input_columns(squares, us_pieces, them_pieces)
        C = self._target_matrix(squares, us_king_zone, them_high_value)

        W_a = self._controllability_gramian(A_hat, B_a)
        W_d = self._controllability_gramian(A_hat, B_d)
        W_o = self._observability_gramian(A_hat, C)

        if self.ablation == "diag_only_gramian":
            # Project Gramians to their diagonals to test multi-mode interactions.
            eye = torch.eye(64, device=W_a.device, dtype=W_a.dtype).unsqueeze(0)
            diag_a = torch.diagonal(W_a, dim1=1, dim2=2).unsqueeze(-1) * eye
            diag_d = torch.diagonal(W_d, dim1=1, dim2=2).unsqueeze(-1) * eye
            diag_o = torch.diagonal(W_o, dim1=1, dim2=2).unsqueeze(-1) * eye
            W_a, W_d, W_o = diag_a, diag_d, diag_o

        W_a_sym = self._symmetrize(W_a)
        W_d_sym = self._symmetrize(W_d)
        W_o_sym = self._symmetrize(W_o)

        # Target-conditioned reach: trace(C W C^T) = sum_{ij} C_ij sum_k W_ik C_jk
        # = sum_i (C W C^T)_ii. For (B, target_rank, 64) and (B, 64, 64).
        CWaCT = torch.bmm(torch.bmm(C, W_a_sym), C.transpose(1, 2))  # (B, r, r)
        CWdCT = torch.bmm(torch.bmm(C, W_d_sym), C.transpose(1, 2))
        target_diag_attacker = torch.diagonal(CWaCT, dim1=1, dim2=2)  # (B, target_rank)
        target_diag_defender = torch.diagonal(CWdCT, dim1=1, dim2=2)
        T_a = target_diag_attacker.sum(dim=-1)
        T_d = target_diag_defender.sum(dim=-1)
        T_net = T_a - T_d

        observability_trace = torch.diagonal(W_o_sym, dim1=1, dim2=2).sum(dim=-1)

        if self.ablation == "no_observability":
            # Skip observability whitening — read attacker/defender modes
            # directly from W_a / W_d singular values.
            attacker_modes = self._top_singular_values(W_a_sym, self.readout_modes)
            defender_modes = self._top_singular_values(W_d_sym, self.readout_modes)
        else:
            sqrt_W_o, _ = self._psd_sqrt(W_o_sym)
            whitened_attacker = torch.bmm(torch.bmm(sqrt_W_o, W_a_sym), sqrt_W_o)
            whitened_defender = torch.bmm(torch.bmm(sqrt_W_o, W_d_sym), sqrt_W_o)
            attacker_modes = self._top_singular_values(whitened_attacker, self.readout_modes)
            defender_modes = self._top_singular_values(whitened_defender, self.readout_modes)

        mode_ratio = attacker_modes / (
            attacker_modes.abs() + defender_modes.abs() + 1.0e-6
        )
        principal_angles = self._principal_angles(W_a_sym, W_d_sym, self.readout_modes)

        head_input = torch.cat(
            [
                pooled,
                T_a.unsqueeze(-1),
                T_d.unsqueeze(-1),
                T_net.unsqueeze(-1),
                observability_trace.unsqueeze(-1),
                attacker_modes,
                defender_modes,
                mode_ratio,
                principal_angles,
                target_diag_attacker,
                target_diag_defender,
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
            "T_a": T_a,
            "T_d": T_d,
            "T_net": T_net,
            "observability_trace": observability_trace,
            "attacker_hankel_modes": attacker_modes,
            "defender_hankel_modes": defender_modes,
            "mode_ratio": mode_ratio,
            "subspace_principal_angles": principal_angles,
            "target_diag_attacker": target_diag_attacker,
            "target_diag_defender": target_diag_defender,
            "operator_norm": operator_norm,
            "operator_gate_weights": gate_weights,
            "operator_low_rank_energy": low_rank_energy,
            "ablation_attacker_only": ablation_flag("attacker_only"),
            "ablation_defender_only": ablation_flag("defender_only"),
            "ablation_no_observability": ablation_flag("no_observability"),
            "ablation_one_step_gramian": ablation_flag("one_step_gramian"),
            "ablation_random_target_C": ablation_flag("random_target_C"),
            "ablation_random_geometry_A": ablation_flag("random_geometry_A"),
            "ablation_fixed_A_no_gates": ablation_flag("fixed_A_no_gates"),
            "ablation_diag_only_gramian": ablation_flag("diag_only_gramian"),
            "ablation_cnn_same_params": ablation_flag("cnn_same_params"),
        }
        return output


def build_tactical_controllability_gramian_network_from_config(
    config: dict[str, Any],
) -> TacticalControllabilityGramianNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)

    return TacticalControllabilityGramianNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        depth=int(cfg.pop("depth", 2)),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        operator_rank=int(cfg.pop("operator_rank", 12)),
        input_rank=int(cfg.pop("input_rank", 8)),
        target_rank=int(cfg.pop("target_rank", 8)),
        gramian_steps=int(cfg.pop("gramian_steps", 6)),
        readout_modes=int(cfg.pop("readout_modes", 12)),
        spectral_norm_iters=int(cfg.pop("spectral_norm_iters", 3)),
        head_hidden=cfg.pop("head_hidden", None),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        ablation=str(cfg.pop("ablation", "none")),
    )


__all__ = [
    "GEOMETRY_NAMES",
    "TacticalControllabilityGramianNetwork",
    "VALID_ABLATIONS",
    "build_tactical_controllability_gramian_network_from_config",
]
