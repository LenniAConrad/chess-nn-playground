"""Complex-Amplitude Chess Network (i247) — CAIO primitive integrated with i193.

This module implements the **C**omplex-**A**mplitude **I**nterference
**O**perator (CAIO) primitive as an *additive, gated* side head on the i193
dual-stream trunk. The primitive lifts real-valued board features into
complex amplitudes whose phase carries chess Z2 symmetry state (piece colour,
side-to-move, square colour) and measures the constructive vs destructive
interference of those amplitudes under fixed chess relation masks
(king-zone adjacency, ray alignment, same-square-colour, file/rank adjacency).

Math signature (see ideas/research/primitives/claude_04_complex_amplitude_interference.md):

    h = encoder(simple_18 board)             # (B, C, 8, 8)
    rho = softplus(W_r h)                    # magnitude, positive
    theta = W_t h + theta_rule(square, side, piece colour, square colour)
    z = rho * exp(i theta)                   # complex amplitude
    for each relation mask M_r:
        I_r(u, v) = Re(z_u * conj(z_v) * exp(i beta_r))
        D_r(u, v) = Im(z_u * conj(z_v) * exp(i beta_r))
        constructive_r = sum_{u,v} M_r[u,v] * relu( I_r(u,v))
        destructive_r  = sum_{u,v} M_r[u,v] * relu(-I_r(u,v))
        curl_r         = sum_{u,v} M_r[u,v] * D_r(u,v)
    conj_error = || z(color_flip(board)) - conj(z(board)) ||

Outputs constructive_r, destructive_r, curl_r per relation and one
conjugacy-error scalar, then runs a small discriminator MLP to produce a
primitive delta logit added to the i193 base logit through a learned gate.

Inputs: simple_18 board tensor only. No CRTK metadata, source labels, tactic
tags, verification flags, Stockfish scores, PV info, or report-only metadata
are consulted at any point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


WHITE_PIECE_CHANNELS = slice(0, 6)
BLACK_PIECE_CHANNELS = slice(6, 12)
STM_CHANNEL = 12
WHITE_KING_CASTLE_CHANNEL = 13
WHITE_QUEEN_CASTLE_CHANNEL = 14
BLACK_KING_CASTLE_CHANNEL = 15
BLACK_QUEEN_CASTLE_CHANNEL = 16
EN_PASSANT_CHANNEL = 17
SQUARES = 64
BOARD_HW = 8


def _build_king_zone_mask() -> torch.Tensor:
    mask = torch.zeros(SQUARES, SQUARES)
    for u in range(SQUARES):
        ur, uf = u // BOARD_HW, u % BOARD_HW
        for v in range(SQUARES):
            if u == v:
                continue
            vr, vf = v // BOARD_HW, v % BOARD_HW
            if abs(ur - vr) <= 1 and abs(uf - vf) <= 1:
                mask[u, v] = 1.0
    return mask


def _build_ray_mask() -> torch.Tensor:
    mask = torch.zeros(SQUARES, SQUARES)
    for u in range(SQUARES):
        ur, uf = u // BOARD_HW, u % BOARD_HW
        for v in range(SQUARES):
            if u == v:
                continue
            vr, vf = v // BOARD_HW, v % BOARD_HW
            if ur == vr or uf == vf or abs(ur - vr) == abs(uf - vf):
                mask[u, v] = 1.0
    return mask


def _build_square_color_mask() -> torch.Tensor:
    mask = torch.zeros(SQUARES, SQUARES)
    for u in range(SQUARES):
        ur, uf = u // BOARD_HW, u % BOARD_HW
        u_color = (ur + uf) % 2
        for v in range(SQUARES):
            if u == v:
                continue
            vr, vf = v // BOARD_HW, v % BOARD_HW
            v_color = (vr + vf) % 2
            if u_color == v_color:
                mask[u, v] = 1.0
    return mask


def _build_file_rank_adjacent_mask() -> torch.Tensor:
    mask = torch.zeros(SQUARES, SQUARES)
    for u in range(SQUARES):
        ur, uf = u // BOARD_HW, u % BOARD_HW
        for v in range(SQUARES):
            if u == v:
                continue
            vr, vf = v // BOARD_HW, v % BOARD_HW
            if abs(ur - vr) <= 1 or abs(uf - vf) <= 1:
                mask[u, v] = 1.0
    return mask


def build_relation_masks() -> torch.Tensor:
    """Stack the 4 chess relation masks into a (4, 64, 64) tensor.

    Relations (in order):
        0. King-zone adjacency (3x3 around each square, excluding self).
        1. Ray alignment (rook + bishop + queen line, excluding self).
        2. Same-square-colour squares (light vs dark).
        3. File-or-rank adjacency.
    """
    masks = torch.stack(
        [
            _build_king_zone_mask(),
            _build_ray_mask(),
            _build_square_color_mask(),
            _build_file_rank_adjacent_mask(),
        ],
        dim=0,
    )
    return masks


NUM_RELATIONS = 4


def color_flip_simple_18(board: torch.Tensor) -> torch.Tensor:
    """Swap white/black piece planes, side-to-move, and castling rights.

    The chess Z2 colour swap maps:
        - white piece planes (0..5) <-> black piece planes (6..11)
        - white-to-move <-> black-to-move (channel 12 flipped)
        - white-king/queen castle <-> black-king/queen castle (13/14 <-> 15/16)
        - en-passant plane (17) is left unchanged (its file is invariant
          under colour swap; rank inversion would require flipping the
          board vertically, which we deliberately do not do because the
          simple_18 encoding does not canonicalise side-to-move).

    Returns a new tensor; the original is not mutated.
    """
    if board.shape[-3] != 18:
        raise ValueError(
            f"color_flip_simple_18 expects 18 input channels, got {board.shape[-3]}"
        )
    out = board.clone()
    out[..., WHITE_PIECE_CHANNELS, :, :] = board[..., BLACK_PIECE_CHANNELS, :, :]
    out[..., BLACK_PIECE_CHANNELS, :, :] = board[..., WHITE_PIECE_CHANNELS, :, :]
    out[..., STM_CHANNEL, :, :] = 1.0 - board[..., STM_CHANNEL, :, :]
    out[..., WHITE_KING_CASTLE_CHANNEL, :, :] = board[..., BLACK_KING_CASTLE_CHANNEL, :, :]
    out[..., WHITE_QUEEN_CASTLE_CHANNEL, :, :] = board[..., BLACK_QUEEN_CASTLE_CHANNEL, :, :]
    out[..., BLACK_KING_CASTLE_CHANNEL, :, :] = board[..., WHITE_KING_CASTLE_CHANNEL, :, :]
    out[..., BLACK_QUEEN_CASTLE_CHANNEL, :, :] = board[..., WHITE_QUEEN_CASTLE_CHANNEL, :, :]
    return out


class CAIOEncoder(nn.Module):
    """Compact spatial encoder used by the CAIO primitive head.

    Produces per-square latent features `h ∈ R^(B, C, 8, 8)` from the
    simple_18 board. Uses GroupNorm so the same encoder can be applied to
    the original and colour-flipped variants without leaking statistics.
    """

    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 32,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(depth) < 1:
            raise ValueError("CAIOEncoder depth must be >= 1")
        layers: list[nn.Module] = []
        in_c = int(input_channels)
        for _ in range(int(depth)):
            layers.append(nn.Conv2d(in_c, int(channels), kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(min(8, int(channels)), int(channels)))
            layers.append(nn.GELU())
            if float(dropout) > 0:
                layers.append(nn.Dropout2d(float(dropout)))
            in_c = int(channels)
        self.body = nn.Sequential(*layers)
        self.output_channels = int(channels)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        return self.body(board)


@dataclass(frozen=True)
class CAIOAmplitudes:
    z: torch.Tensor  # complex (B, d, 64)
    rho: torch.Tensor  # real (B, d, 64)
    theta: torch.Tensor  # real (B, d, 64)


class CAIOPrimitiveHead(nn.Module):
    """Complex-Amplitude Interference Operator primitive head.

    Lifts the simple_18 board into a complex per-square amplitude with a
    chess-rule phase prior (piece colour, side-to-move, square colour) and
    pools pairwise interference under 4 fixed chess relation masks
    (king-zone, ray, square-colour, file-rank adjacency). Outputs a 3*R + 1
    fingerprint (constructive_r, destructive_r, curl_r, conjugacy error)
    plus a primitive delta logit produced by a small discriminator MLP.
    """

    ALLOWED_ABLATIONS = (
        "none",
        "real_only",
        "random_phase",
        "free_phase",
        "shuffle_relation_masks",
        "no_conjugacy",
        "constructive_only",
        "no_caio",
    )

    def __init__(
        self,
        amplitude_dim: int = 8,
        feature_channels: int = 32,
        feature_depth: int = 2,
        encoder_dropout: float = 0.0,
        hidden_dim: int = 64,
        head_dropout: float = 0.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(amplitude_dim) < 1:
            raise ValueError("CAIO amplitude_dim must be >= 1")
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown CAIO ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}"
            )
        self.spec = BoardTensorSpec(input_channels=18)
        self.amplitude_dim = int(amplitude_dim)
        self.ablation = str(ablation)

        self.encoder = CAIOEncoder(
            input_channels=18,
            channels=int(feature_channels),
            depth=int(feature_depth),
            dropout=float(encoder_dropout),
        )
        self.mag_proj = nn.Conv2d(int(feature_channels), self.amplitude_dim, kernel_size=1)
        self.phase_proj = nn.Conv2d(int(feature_channels), self.amplitude_dim, kernel_size=1)

        # Learnable rule-phase coefficients. The defaults encode the canonical
        # chess Z2 actions: pi per piece colour, pi/4 per square colour,
        # pi/2 per side-to-move tempo flip.
        self.alpha_piece = nn.Parameter(torch.full((self.amplitude_dim,), math.pi))
        self.alpha_square = nn.Parameter(torch.full((self.amplitude_dim,), math.pi / 4.0))
        self.alpha_side = nn.Parameter(torch.full((self.amplitude_dim,), math.pi / 2.0))

        self.beta = nn.Parameter(torch.zeros(NUM_RELATIONS))
        self.register_buffer("relation_masks", build_relation_masks(), persistent=False)

        # Per-square color phase = (r + f) % 2; broadcast to (1, 1, 8, 8).
        rng = torch.arange(BOARD_HW)
        sq_color = ((rng.view(-1, 1) + rng.view(1, -1)) % 2).to(torch.float32)
        self.register_buffer("square_color", sq_color.view(1, 1, BOARD_HW, BOARD_HW), persistent=False)

        feat_dim = 3 * NUM_RELATIONS + 1
        self.feature_dim = feat_dim
        dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, int(hidden_dim)),
            nn.GELU(),
            dropout_module,
            nn.Linear(int(hidden_dim), 1),
        )

    def _rule_phase(self, board: torch.Tensor) -> torch.Tensor:
        # Piece-color indicator: +1 for white piece, -1 for black piece, 0 empty.
        white_mask = board[:, WHITE_PIECE_CHANNELS].sum(dim=1).clamp(0.0, 1.0)
        black_mask = board[:, BLACK_PIECE_CHANNELS].sum(dim=1).clamp(0.0, 1.0)
        piece_phase_input = (white_mask - black_mask).unsqueeze(1)  # (B, 1, 8, 8)
        # Side-to-move centred to [-0.5, 0.5] so the phase increment is signed.
        side_centered = (board[:, STM_CHANNEL:STM_CHANNEL + 1] - 0.5)  # (B, 1, 8, 8)
        sq_color = self.square_color.to(device=board.device, dtype=board.dtype)
        # Broadcast across amplitude dimensions via 1x1 broadcasting of alpha.
        alpha_piece = self.alpha_piece.view(1, self.amplitude_dim, 1, 1)
        alpha_square = self.alpha_square.view(1, self.amplitude_dim, 1, 1)
        alpha_side = self.alpha_side.view(1, self.amplitude_dim, 1, 1)
        return (
            alpha_piece * piece_phase_input
            + alpha_square * sq_color
            + alpha_side * side_centered
        )

    def _amplitudes(self, board: torch.Tensor) -> CAIOAmplitudes:
        h = self.encoder(board)
        rho = torch.nn.functional.softplus(self.mag_proj(h))  # (B, d, 8, 8)
        theta_logits = self.phase_proj(h)  # (B, d, 8, 8)
        theta_rule = self._rule_phase(board)  # (B, d, 8, 8)
        if self.ablation == "random_phase":
            theta = (torch.rand_like(theta_logits) * 2.0 - 1.0) * math.pi
        elif self.ablation == "free_phase":
            theta = theta_logits
        else:
            theta = theta_logits + theta_rule
        if self.ablation == "real_only":
            theta = torch.zeros_like(theta)
        z = torch.complex(rho * theta.cos(), rho * theta.sin())  # (B, d, 8, 8)
        rho_flat = rho.reshape(rho.shape[0], self.amplitude_dim, SQUARES)
        z_flat = z.reshape(z.shape[0], self.amplitude_dim, SQUARES)
        theta_flat = theta.reshape(theta.shape[0], self.amplitude_dim, SQUARES)
        return CAIOAmplitudes(z=z_flat, rho=rho_flat, theta=theta_flat)

    def _interference_features(
        self, z_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masks = self.relation_masks.to(device=z_flat.device)
        if self.ablation == "shuffle_relation_masks":
            shuffled = torch.empty_like(masks)
            for r in range(NUM_RELATIONS):
                flat = masks[r].reshape(-1)
                perm = torch.randperm(flat.numel(), device=flat.device)
                shuffled[r] = flat[perm].reshape(masks[r].shape)
            masks = shuffled
        cons_features = []
        des_features = []
        curl_features = []
        for r in range(NUM_RELATIONS):
            mask_r = masks[r]  # (64, 64)
            beta_r = self.beta[r]
            phase_factor = torch.complex(beta_r.cos(), beta_r.sin())
            outer = z_flat.unsqueeze(-1) * z_flat.conj().unsqueeze(-2)  # (B, d, 64, 64)
            interference = outer * mask_r.unsqueeze(0).unsqueeze(0) * phase_factor
            I = interference.real
            D = interference.imag
            cons_features.append(torch.relu(I).sum(dim=(1, 2, 3)))
            des_features.append(torch.relu(-I).sum(dim=(1, 2, 3)))
            curl_features.append(D.sum(dim=(1, 2, 3)))
        cons = torch.stack(cons_features, dim=1)
        des = torch.stack(des_features, dim=1)
        curl = torch.stack(curl_features, dim=1)
        if self.ablation == "constructive_only":
            des = torch.zeros_like(des)
            curl = torch.zeros_like(curl)
        return cons, des, curl

    def _conjugacy_error(self, board: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        if self.ablation in {"no_conjugacy", "no_caio"}:
            return torch.zeros(board.shape[0], device=board.device, dtype=board.dtype)
        flipped = color_flip_simple_18(board)
        z_flip = self._amplitudes(flipped).z
        return (z_flip - z_flat.conj()).abs().mean(dim=(1, 2))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        amplitudes = self._amplitudes(board)
        z_flat = amplitudes.z
        cons, des, curl = self._interference_features(z_flat)
        conj_err = self._conjugacy_error(board, z_flat)
        if self.ablation == "no_caio":
            cons = torch.zeros_like(cons)
            des = torch.zeros_like(des)
            curl = torch.zeros_like(curl)
        feats = torch.cat([cons, des, curl, conj_err.unsqueeze(1)], dim=1)
        delta = self.head(feats).view(-1)
        return {
            "delta_phi": delta,
            "caio_constructive": cons,
            "caio_destructive": des,
            "caio_curl": curl,
            "caio_conjugacy_error": conj_err,
            "caio_rho_mean": amplitudes.rho.mean(dim=(1, 2)),
            "caio_theta_mean": amplitudes.theta.mean(dim=(1, 2)),
            "caio_amplitude_norm": amplitudes.z.abs().mean(dim=(1, 2)),
        }


class ComplexAmplitudeChessNetwork(nn.Module):
    """i247 — Complex-Amplitude Chess Network = i193 trunk + CAIO primitive head.

    Forward pass:

    1. Run the i193 ExchangeThenKingDualStream trunk on the un-perturbed
       board to get the base logit and trunk diagnostics.
    2. Run the CAIO primitive head on the same board to get the complex
       interference fingerprint and a primitive delta logit.
    3. A small gate MLP over trunk diagnostics + CAIO fingerprint produces a
       sigmoid gate; the final logit is

           final_logit = base_logit + gate * primitive_delta

    Ablations cover the seven falsifiers in the CAIO spec (A1 real-only, A2
    random phase, A3 free phase, A4 shuffled relation masks, A5 no
    conjugacy, A6 constructive only, A7 no destructive features) plus
    `no_caio` / `zero_gate` / `trunk_only` for sanity checks.
    """

    ALLOWED_ABLATIONS = (
        "none",
        "real_only",
        "random_phase",
        "free_phase",
        "shuffle_relation_masks",
        "no_conjugacy",
        "constructive_only",
        "no_caio",
        "zero_gate",
        "trunk_only",
    )

    _GATE_DIAG_KEYS: tuple[str, ...] = (
        "gate",
        "gate_entropy",
        "mechanism_energy",
        "stream_disagreement",
    )

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
        amplitude_dim: int = 8,
        feature_channels: int = 32,
        feature_depth: int = 2,
        encoder_dropout: float = 0.0,
        caio_hidden_dim: int = 64,
        caio_dropout: float = 0.0,
        gate_hidden_dim: int = 32,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "ComplexAmplitudeChessNetwork supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "ComplexAmplitudeChessNetwork requires the simple_18 board tensor"
            )
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}"
            )
        self.spec = BoardTensorSpec(input_channels=18)
        self.num_classes = 1
        self.ablation = str(ablation)

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

        caio_ablation = ablation if ablation in CAIOPrimitiveHead.ALLOWED_ABLATIONS else "none"
        self.caio = CAIOPrimitiveHead(
            amplitude_dim=int(amplitude_dim),
            feature_channels=int(feature_channels),
            feature_depth=int(feature_depth),
            encoder_dropout=float(encoder_dropout),
            hidden_dim=int(caio_hidden_dim),
            head_dropout=float(caio_dropout),
            ablation=caio_ablation,
        )

        gate_in = len(self._GATE_DIAG_KEYS) + self.caio.feature_dim
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(gate_in),
            nn.Linear(gate_in, int(gate_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(gate_hidden_dim), 1),
        )
        with torch.no_grad():
            self.gate_mlp[-1].bias.fill_(float(gate_init))

    def _collect_caio_features(self, caio_out: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                caio_out["caio_constructive"],
                caio_out["caio_destructive"],
                caio_out["caio_curl"],
                caio_out["caio_conjugacy_error"].unsqueeze(1),
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"]

        caio_out = self.caio(board)
        primitive_delta_raw = caio_out["delta_phi"]

        diag_stack = torch.stack(
            [trunk_out[key].detach() for key in self._GATE_DIAG_KEYS],
            dim=1,
        )
        caio_features = self._collect_caio_features(caio_out)
        gate_input = torch.cat([diag_stack, caio_features], dim=1)
        gate_logit = self.gate_mlp(gate_input).view(-1)
        gate = torch.sigmoid(gate_logit)

        if self.ablation == "trunk_only":
            primitive_delta = torch.zeros_like(primitive_delta_raw)
            gate_applied = torch.zeros_like(gate)
        elif self.ablation == "zero_gate":
            primitive_delta = primitive_delta_raw
            gate_applied = torch.zeros_like(gate)
        elif self.ablation == "no_caio":
            primitive_delta = torch.zeros_like(primitive_delta_raw)
            gate_applied = torch.zeros_like(gate)
        else:
            primitive_delta = primitive_delta_raw
            gate_applied = gate

        primitive_contribution = gate_applied * primitive_delta
        logits = base_logit + primitive_contribution

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
        out["primitive_delta_raw"] = primitive_delta_raw
        out["primitive_gate"] = gate
        out["primitive_gate_applied"] = gate_applied
        out["primitive_gate_logit"] = gate_logit
        out["primitive_gate_entropy"] = gate_entropy
        out["primitive_contribution"] = primitive_contribution
        out["caio_constructive_mean"] = caio_out["caio_constructive"].mean(dim=1)
        out["caio_destructive_mean"] = caio_out["caio_destructive"].mean(dim=1)
        out["caio_curl_mean"] = caio_out["caio_curl"].mean(dim=1)
        out["caio_conjugacy_error"] = caio_out["caio_conjugacy_error"]
        out["caio_amplitude_norm"] = caio_out["caio_amplitude_norm"]
        out["mechanism_energy"] = trunk_out["mechanism_energy"]
        out["proposal_profile_strength"] = (
            out["caio_constructive_mean"] * gate_entropy
        )
        out["proposal_keyword_count"] = logits.new_full(
            (logits.shape[0],), float(self.caio.feature_dim)
        )
        return out


def build_complex_amplitude_chess_network_from_config(
    config: dict[str, Any],
) -> ComplexAmplitudeChessNetwork:
    cfg = dict(config)
    return ComplexAmplitudeChessNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        amplitude_dim=int(cfg.get("amplitude_dim", 8)),
        feature_channels=int(cfg.get("feature_channels", 32)),
        feature_depth=int(cfg.get("feature_depth", 2)),
        encoder_dropout=float(cfg.get("encoder_dropout", 0.0)),
        caio_hidden_dim=int(cfg.get("caio_hidden_dim", 64)),
        caio_dropout=float(cfg.get("caio_dropout", 0.0)),
        gate_hidden_dim=int(cfg.get("gate_hidden_dim", 32)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "CAIOEncoder",
    "CAIOPrimitiveHead",
    "ComplexAmplitudeChessNetwork",
    "build_complex_amplitude_chess_network_from_config",
    "build_relation_masks",
    "color_flip_simple_18",
    "NUM_RELATIONS",
)
