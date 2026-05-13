"""Tempo-Defender Cross-Derivative Network for idea i244.

Implements the TDCD primitive (Claude Opus 4.7 proposal
`primitive_tempo_defender_cross_derivative`) as an *additive head* on top of
the i193 exchange-then-king dual-stream trunk. The head computes the mixed
partial of a lightweight learned encoder under two Z2 perturbations:

* `sigma_T`: tempo flip (invert the side-to-move plane).
* `delta_k`: zero an enemy piece plane at square `s_k` selected by saliency.

Cross-derivative spectrum:

    g_T  = phi(x) - phi(sigma_T x)
    g_Dk = phi(x) - phi(delta_k x)
    tau_k = phi(delta_k x) - phi(sigma_T delta_k x)
    DeltaDelta_k = ||tau_k|| - ||g_T||

The primitive delta logit is a gated MLP read of the (g_T, DeltaDelta_k, sal)
fingerprint added to the i193 base logit. The grid runs through a *separate*
compact encoder so total cost is roughly 2-3x i193, not 8x i193. The trunk
is the i193 dual-stream network and only ever evaluates the unperturbed
board, keeping the comparison "i193 baseline" vs "i193 + TDCD head" clean.

All inputs are derived from the current-board simple_18 tensor only. No
CRTK metadata, tactic tags, Stockfish scores, or report-only metadata are
read at any point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


WHITE_PIECE_CHANNELS = slice(0, 6)
BLACK_PIECE_CHANNELS = slice(6, 12)
STM_CHANNEL = 12
WHITE_CASTLING_CHANNELS = (13, 14)
BLACK_CASTLING_CHANNELS = (15, 16)
EN_PASSANT_CHANNEL = 17
SQUARES = 64

ALLOWED_ABLATIONS = (
    "none",
    "main_effects_only",
    "no_mixed_partial",
    "null_board_perturbation",
    "attacker_perturbation",
    "skip_cross_derivative",
    "shared_saliency_uniform",
    "fixed_zero_gate",
)


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def tempo_flip_board(board: torch.Tensor) -> torch.Tensor:
    """Invert the side-to-move plane while leaving piece geometry intact.

    The simple_18 encoding keeps a fixed geographic orientation regardless of
    side-to-move, so the tempo involution sigma_T is just `stm := 1 - stm`.
    Castling and en-passant planes are not stm-dependent in this encoding,
    so they pass through unchanged. The function is its own inverse.
    """
    if board.shape[-3] < STM_CHANNEL + 1:
        raise ValueError("tempo_flip_board expects the simple_18 channel layout")
    flipped = board.clone()
    flipped[..., STM_CHANNEL, :, :] = 1.0 - board[..., STM_CHANNEL, :, :]
    return flipped


def _enemy_piece_mask(board: torch.Tensor) -> torch.Tensor:
    """Return a (B, 64) mask that is 1 on squares occupied by the enemy.

    "Enemy" is determined from the side-to-move plane: when stm == 1
    (white-to-move) the enemy is black, otherwise white.
    """
    stm = board[:, STM_CHANNEL].mean(dim=(1, 2)).clamp(0.0, 1.0)
    white_occ = board[:, WHITE_PIECE_CHANNELS].sum(dim=1).clamp(0.0, 1.0).flatten(1)
    black_occ = board[:, BLACK_PIECE_CHANNELS].sum(dim=1).clamp(0.0, 1.0).flatten(1)
    stm_selector = stm.view(-1, 1)
    enemy = stm_selector * black_occ + (1.0 - stm_selector) * white_occ
    return enemy.clamp(0.0, 1.0)


def _own_piece_mask(board: torch.Tensor) -> torch.Tensor:
    stm = board[:, STM_CHANNEL].mean(dim=(1, 2)).clamp(0.0, 1.0)
    white_occ = board[:, WHITE_PIECE_CHANNELS].sum(dim=1).clamp(0.0, 1.0).flatten(1)
    black_occ = board[:, BLACK_PIECE_CHANNELS].sum(dim=1).clamp(0.0, 1.0).flatten(1)
    stm_selector = stm.view(-1, 1)
    own = stm_selector * white_occ + (1.0 - stm_selector) * black_occ
    return own.clamp(0.0, 1.0)


def apply_square_removal(
    board: torch.Tensor,
    mask: torch.Tensor,
    *,
    remove_own: bool = False,
) -> torch.Tensor:
    """Zero the piece planes for the enemy (or own) at marked squares.

    Args:
        board: (B, 18, 8, 8) simple_18 tensor.
        mask: (B, 64) {0,1} mask over squares to zero.
        remove_own: if True, removes squares from the side-to-move colour
            instead of the enemy colour. Used for the attacker-perturbation
            ablation.

    Returns:
        A new (B, 18, 8, 8) tensor with the selected colour's piece planes
        zeroed on the masked squares. Side-to-move and other planes are
        untouched. The mask is broadcast across all six piece-type planes.
    """
    if mask.dim() != 2 or mask.shape[-1] != SQUARES:
        raise ValueError(f"mask must be (batch, 64); got {tuple(mask.shape)}")
    batch = board.shape[0]
    stm = board[:, STM_CHANNEL].mean(dim=(1, 2)).clamp(0.0, 1.0).view(batch, 1, 1, 1)
    spatial = mask.view(batch, 1, 8, 8)
    keep = 1.0 - spatial.clamp(0.0, 1.0)
    out = board.clone()
    white_planes = out[:, WHITE_PIECE_CHANNELS]
    black_planes = out[:, BLACK_PIECE_CHANNELS]
    if remove_own:
        white_keep = (1.0 - stm) + stm * keep
        black_keep = stm + (1.0 - stm) * keep
    else:
        white_keep = stm + (1.0 - stm) * keep
        black_keep = (1.0 - stm) + stm * keep
    out[:, WHITE_PIECE_CHANNELS] = white_planes * white_keep
    out[:, BLACK_PIECE_CHANNELS] = black_planes * black_keep
    return out


def _square_index_to_mask(indices: torch.Tensor) -> torch.Tensor:
    """One-hot encode a (B,) tensor of square indices into a (B, 64) mask."""
    return F.one_hot(indices.clamp(0, SQUARES - 1).long(), num_classes=SQUARES).to(dtype=torch.float32)


@dataclass(frozen=True)
class SaliencyOutput:
    scores: torch.Tensor  # (B, 64), enemy-piece-masked
    raw_scores: torch.Tensor  # (B, 64), pre-masking logits
    enemy_mask: torch.Tensor  # (B, 64), 1 on enemy-occupied squares
    own_mask: torch.Tensor  # (B, 64), 1 on own-occupied squares
    top_indices: torch.Tensor  # (B, K), indices of top-K critical defenders
    top_valid: torch.Tensor  # (B, K), 1.0 if the slot has a real enemy piece
    entropy: torch.Tensor  # (B,)
    concentration: torch.Tensor  # (B,)


class SaliencyHead(nn.Module):
    """Per-square saliency head over the simple_18 board tensor.

    Produces a logit per square, masks to enemy-occupied squares, and
    selects the top-K critical defenders. The masked logits are also turned
    into a softmax distribution for entropy/concentration diagnostics.
    """

    def __init__(self, input_channels: int, hidden: int, topk: int) -> None:
        super().__init__()
        if int(topk) < 1:
            raise ValueError("topk must be >= 1")
        self.topk = int(topk)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.conv = nn.Sequential(
            nn.Conv2d(int(input_channels), int(hidden), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden), 1, kernel_size=1),
        )

    def forward(self, board: torch.Tensor, *, uniform: bool = False) -> SaliencyOutput:
        board = require_board_tensor(board, self.spec)
        raw = self.conv(board).view(board.shape[0], SQUARES)
        enemy_mask = _enemy_piece_mask(board)
        own_mask = _own_piece_mask(board)
        if uniform:
            raw = torch.zeros_like(raw)
        masked = raw.masked_fill(enemy_mask < 0.5, float("-inf"))
        # Replace -inf with a finite sentinel for the softmax/topk on empty boards.
        sentinel = -1.0e4
        masked = torch.where(torch.isfinite(masked), masked, masked.new_full(masked.shape, sentinel))
        top_values, top_indices = masked.topk(k=self.topk, dim=1)
        # A slot is only "valid" if it lands on an actual enemy piece.
        top_valid = torch.gather(enemy_mask, 1, top_indices)
        # Softmax over enemy squares for entropy diagnostics.
        probs = torch.softmax(masked, dim=1) * enemy_mask
        denom = probs.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        probs = probs / denom
        entropy = -(probs * probs.clamp_min(1.0e-8).log()).sum(dim=1) / math.log(max(2, SQUARES))
        concentration = probs.amax(dim=1)
        return SaliencyOutput(
            scores=masked.masked_fill(~torch.isfinite(masked), 0.0),
            raw_scores=raw,
            enemy_mask=enemy_mask,
            own_mask=own_mask,
            top_indices=top_indices,
            top_valid=top_valid,
            entropy=entropy,
            concentration=concentration,
        )


def _pick_num_groups(channels: int) -> int:
    channels = int(channels)
    if channels <= 1:
        return 1
    for candidate in (8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


class TDCDEncoder(nn.Module):
    """Lightweight shared encoder used across the cross-derivative grid.

    Compact on purpose: the grid evaluates 2*(K+1) board variants per
    sample, so the encoder must stay much cheaper than the i193 trunk.
    Produces a pooled feature vector (mean + max global pool).
    """

    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if int(depth) < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = []
        in_c = int(input_channels)
        num_groups = _pick_num_groups(int(channels))
        for _ in range(int(depth)):
            layers.append(nn.Conv2d(in_c, int(channels), kernel_size=3, padding=1, bias=not use_batchnorm))
            # GroupNorm avoids batchnorm coupling across grid copies.
            layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=int(channels)))
            layers.append(nn.GELU())
            if float(dropout) > 0:
                layers.append(nn.Dropout2d(float(dropout)))
            in_c = int(channels)
        self.stack = nn.Sequential(*layers)
        self.output_dim = 2 * int(channels)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        h = self.stack(board)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return pooled


class TempoDefenderCrossDerivativeNetwork(nn.Module):
    """i193-base + TDCD additive primitive head.

    The forward pass is:

    1. Run the i193 dual-stream trunk on the original board to get the base
       logit and trunk diagnostics.
    2. Run a learned saliency head over the same board to select K critical
       enemy pieces.
    3. Build the 2*(K+1) perturbation grid (baseline + per-defender removal,
       each in both tempo phases) and stack it into a single batched forward
       pass through a compact `TDCDEncoder`.
    4. Compute the main effects (`g_T`, `g_Dk`) and the mixed partials
       (`tau_k`, `DeltaDelta_k`), reduce them to a fixed-size cross-derivative
       fingerprint, and pass that fingerprint through a discriminator MLP to
       produce the primitive delta logit.
    5. Final logit = i193_base_logit + gate * primitive_delta.

    Ablations (string `ablation` arg) gate variants of the architecture for
    falsifier experiments. See `ALLOWED_ABLATIONS` for the full list.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        topk: int = 3,
        saliency_hidden: int = 32,
        tdcd_channels: int = 48,
        tdcd_depth: int = 2,
        tdcd_dropout: float = 0.05,
        head_hidden: int = 64,
        gate_init: float = -2.0,
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "TempoDefenderCrossDerivativeNetwork supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError("TDCD requires the simple_18 current-board tensor")
        if int(topk) < 1:
            raise ValueError("topk must be >= 1")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )
        self.num_classes = int(num_classes)
        self.topk = int(topk)
        self.ablation = str(ablation)
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
            ablation="none",
        )
        self.saliency = SaliencyHead(
            input_channels=int(input_channels),
            hidden=int(saliency_hidden),
            topk=int(topk),
        )
        self.encoder = TDCDEncoder(
            input_channels=int(input_channels),
            channels=int(tdcd_channels),
            depth=int(tdcd_depth),
            dropout=float(tdcd_dropout),
            use_batchnorm=False,
        )

        # Fingerprint dimensions: 8 scalar features from the cross-derivative
        # spectrum + topk DeltaDelta values + saliency entropy/concentration
        # + main-effect norms (g_T_norm, g_D_mean_norm, baseline_norm).
        self.fingerprint_dim = 8 + self.topk + 2 + 3
        self.head = nn.Sequential(
            nn.LayerNorm(self.fingerprint_dim),
            nn.Linear(self.fingerprint_dim, int(head_hidden)),
            nn.GELU(),
            nn.Linear(int(head_hidden), max(8, int(head_hidden) // 2)),
            nn.GELU(),
            nn.Linear(max(8, int(head_hidden) // 2), 1),
        )
        # Per-batch sigmoid gate over the same fingerprint so the head can
        # collapse to zero on positions where the cross-derivative is noisy.
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.fingerprint_dim),
            nn.Linear(self.fingerprint_dim, max(8, int(head_hidden) // 4)),
            nn.GELU(),
            nn.Linear(max(8, int(head_hidden) // 4), 1),
        )
        # Initialize the gate near zero so the primitive starts as a no-op.
        with torch.no_grad():
            final_layer = self.gate_head[-1]
            if isinstance(final_layer, nn.Linear):
                final_layer.bias.fill_(float(gate_init))

    def _build_grid(
        self,
        board: torch.Tensor,
        saliency: SaliencyOutput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct the 2*(K+1) perturbation grid.

        Layout along the new axis (size = 2*(K+1)):
            slot 0:        (T+, no removal)         = baseline
            slot 1:        (T-, no removal)
            slot 2 + 2k:   (T+, defender k removed)
            slot 3 + 2k:   (T-, defender k removed)

        Returns:
            grid: (B, 2*(K+1), 18, 8, 8)
            valid: (B, K) validity mask for top-K slots.
        """
        batch = board.shape[0]
        topk_indices = saliency.top_indices
        top_valid = saliency.top_valid
        grid_slots = 2 * (self.topk + 1)
        grid = board.new_zeros(batch, grid_slots, *board.shape[1:])
        # baseline (T+ and T-)
        grid[:, 0] = board
        grid[:, 1] = tempo_flip_board(board)

        remove_own = self.ablation == "attacker_perturbation"
        for k in range(self.topk):
            slot_pos = 2 + 2 * k
            slot_neg = 3 + 2 * k
            if self.ablation == "null_board_perturbation":
                # Replace per-defender removal with the global "all enemies
                # removed" perturbation; copy the same mask K times.
                enemy_mask = saliency.enemy_mask
                removed = apply_square_removal(board, enemy_mask, remove_own=False)
            else:
                indices_k = topk_indices[:, k]
                mask_k = _square_index_to_mask(indices_k).to(device=board.device, dtype=board.dtype)
                # Zero masks for samples where this slot has no valid enemy.
                mask_k = mask_k * top_valid[:, k : k + 1]
                removed = apply_square_removal(board, mask_k, remove_own=remove_own)
            grid[:, slot_pos] = removed
            grid[:, slot_neg] = tempo_flip_board(removed)
        return grid, top_valid

    def _fingerprint(
        self,
        features: torch.Tensor,
        top_valid: torch.Tensor,
        saliency: SaliencyOutput,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute cross-derivative fingerprint from grid encoder features."""
        batch = features.shape[0] // (2 * (self.topk + 1))
        slots = 2 * (self.topk + 1)
        feat = features.view(batch, slots, -1)
        b_pos = feat[:, 0]
        b_neg = feat[:, 1]
        g_T = b_pos - b_neg  # (B, F)
        g_T_norm = g_T.pow(2).sum(dim=1).sqrt().clamp_min(1.0e-6)
        baseline_norm = b_pos.pow(2).sum(dim=1).sqrt()

        g_D = []
        tau = []
        dd = []
        for k in range(self.topk):
            slot_pos = 2 + 2 * k
            slot_neg = 3 + 2 * k
            d_pos = feat[:, slot_pos]
            d_neg = feat[:, slot_neg]
            gdk = b_pos - d_pos
            tauk = d_pos - d_neg
            ddk = tauk.pow(2).sum(dim=1).sqrt() - g_T_norm
            # Zero out contributions from invalid slots so empty boards do
            # not pollute the spectrum statistics.
            valid_mask = top_valid[:, k]
            gdk = gdk * valid_mask.unsqueeze(-1)
            tauk = tauk * valid_mask.unsqueeze(-1)
            ddk = ddk * valid_mask
            g_D.append(gdk)
            tau.append(tauk)
            dd.append(ddk)
        dd_stack = torch.stack(dd, dim=1) if dd else g_T.new_zeros(batch, 0)
        g_D_norms = torch.stack([g.pow(2).sum(dim=1).sqrt() for g in g_D], dim=1) if g_D else g_T.new_zeros(batch, 0)
        valid_counts = top_valid.sum(dim=1).clamp_min(1.0)

        if self.ablation == "no_mixed_partial" or self.ablation == "main_effects_only":
            dd_stack = torch.zeros_like(dd_stack)
        if self.ablation == "skip_cross_derivative":
            g_T = torch.zeros_like(g_T)
            g_T_norm = torch.ones_like(g_T_norm) * 1.0e-3
            dd_stack = torch.zeros_like(dd_stack)
            g_D_norms = torch.zeros_like(g_D_norms)

        # Sorted DeltaDelta values for shape-stable head input even when
        # some slots are invalid.
        if dd_stack.shape[1] > 0:
            sorted_dd, _ = dd_stack.sort(dim=1, descending=True)
        else:
            sorted_dd = dd_stack
        max_dd = dd_stack.amax(dim=1) if dd_stack.shape[1] > 0 else dd_stack.new_zeros(batch)
        mean_dd = dd_stack.sum(dim=1) / valid_counts
        std_dd = ((dd_stack - mean_dd.unsqueeze(-1)) ** 2 * top_valid).sum(dim=1)
        std_dd = (std_dd / valid_counts).clamp_min(0.0).sqrt()
        topk_ratio = max_dd / g_T_norm
        g_D_mean_norm = g_D_norms.sum(dim=1) / valid_counts

        scalar_features = torch.stack(
            [
                g_T_norm,
                max_dd,
                mean_dd,
                std_dd,
                topk_ratio,
                baseline_norm.clamp_max(50.0),
                g_D_mean_norm,
                valid_counts / float(max(1, self.topk)),
            ],
            dim=1,
        )
        fingerprint = torch.cat(
            [
                scalar_features,
                sorted_dd,
                saliency.entropy.unsqueeze(-1),
                saliency.concentration.unsqueeze(-1),
                g_T_norm.unsqueeze(-1),
                mean_dd.unsqueeze(-1),
                max_dd.unsqueeze(-1),
            ],
            dim=1,
        )
        diagnostics = {
            "g_T_norm": g_T_norm,
            "max_dd": max_dd,
            "mean_dd": mean_dd,
            "std_dd": std_dd,
            "topk_dd_ratio": topk_ratio,
            "baseline_feature_norm": baseline_norm,
            "g_D_mean_norm": g_D_mean_norm,
            "valid_slot_count": top_valid.sum(dim=1),
            "delta_delta_per_slot": dd_stack,
        }
        return fingerprint, diagnostics

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        base_out = self.trunk(board)
        base_logit = base_out["logits"].view(-1)

        saliency = self.saliency(
            board,
            uniform=self.ablation == "shared_saliency_uniform",
        )
        grid, top_valid = self._build_grid(board, saliency)
        batch, slots, _, _, _ = grid.shape
        grid_flat = grid.view(batch * slots, *grid.shape[2:])
        features = self.encoder(grid_flat)
        fingerprint, fp_diag = self._fingerprint(features, top_valid, saliency)

        primitive_delta = self.head(fingerprint).view(-1)
        gate_logit = self.gate_head(fingerprint).view(-1)
        if self.ablation == "fixed_zero_gate":
            gate = torch.zeros_like(gate_logit)
        else:
            gate = torch.sigmoid(gate_logit)
        if self.ablation == "skip_cross_derivative":
            gate = torch.zeros_like(gate)
        gated_delta = gate * primitive_delta
        logits = _format_logits(base_logit + gated_delta, self.num_classes)

        eps = 1.0e-6
        gate_clamped = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(
            gate_clamped * gate_clamped.log() + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
        )

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "base_logit": base_logit,
            "primitive_delta": primitive_delta,
            "primitive_gate": gate,
            "primitive_gate_logit": gate_logit,
            "primitive_gate_entropy": gate_entropy,
            "primitive_logit_contribution": gated_delta,
            "saliency_entropy": saliency.entropy,
            "saliency_concentration": saliency.concentration,
            "saliency_top_valid_count": top_valid.sum(dim=1),
            "fingerprint_norm": fingerprint.pow(2).mean(dim=1),
            "mechanism_energy": fingerprint.pow(2).mean(dim=1) + base_out["mechanism_energy"],
            "proposal_profile_strength": (gated_delta.abs() * gate_entropy).clamp(0.0, 20.0),
            "proposal_keyword_count": logits.new_full((board.shape[0],), float(self.fingerprint_dim)),
        }
        for key, value in fp_diag.items():
            diagnostics[key] = value
        for key, value in base_out.items():
            if key == "logits":
                continue
            diag_key = key if key.startswith("base_") or key in {"gate", "gate_logit"} else f"trunk_{key}"
            diagnostics.setdefault(diag_key, value)
        return diagnostics


def build_tempo_defender_cross_derivative_network_from_config(
    config: dict[str, Any],
) -> TempoDefenderCrossDerivativeNetwork:
    cfg = dict(config)
    return TempoDefenderCrossDerivativeNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        topk=int(cfg.get("topk", 3)),
        saliency_hidden=int(cfg.get("saliency_hidden", 32)),
        tdcd_channels=int(cfg.get("tdcd_channels", 48)),
        tdcd_depth=int(cfg.get("tdcd_depth", 2)),
        tdcd_dropout=float(cfg.get("tdcd_dropout", 0.05)),
        head_hidden=int(cfg.get("head_hidden", 64)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        ablation=str(cfg.get("ablation", "none")),
    )
