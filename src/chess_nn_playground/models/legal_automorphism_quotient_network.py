"""Legal Automorphism Quotient Network for idea i042.

Bespoke implementation of the markdown thesis: a board-only puzzle-likeness
classifier that quotients out the exact four-element chess-rule
automorphism group ``G = {e, m, q, mq}`` over the simple_18 tensor, where
``m`` is the file mirror and ``q`` is the color/rank flip with side-to-move
swap. A single shared encoder is applied to the four-view orbit, the
classifier consumes the Reynolds invariant latent
``z_inv = (1/|G|) sum_g phi(g . s)``, and three nontrivial C2 x C2
character components are exposed as diagnostics together with the optional
character-energy regularizer ``R_char``.

The architecture is materially distinct from the shared
``ResearchPacketProbe`` scaffold: the encoder receives a deterministic
four-view orbit constructed by ``LegalAutomorphismTransform``, the head
consumes only ``z_inv`` (the trivial C2xC2 character component), and there
are no proposal-profile diagnostics or mechanism-family embeddings in the
forward path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models._packet_bespoke_base import (
    BoardTensorSpec,
    format_logits,
    require_board_tensor,
)


_EPS = 1.0e-6

# C2 x C2 character table assuming orbit order [e, m, q, mq].
# Row 0 is the trivial character (the invariant Reynolds projection).
_CHARACTER_TABLE: tuple[tuple[float, ...], ...] = (
    (1.0, 1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0, -1.0),
    (1.0, 1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0, 1.0),
)


@dataclass(frozen=True)
class Simple18AutomorphismSpec:
    """Channel layout used to construct the legal chess automorphism orbit."""

    encoding: str = "simple_18"
    input_channels: int = 18
    white_piece_planes: tuple[int, int] = (0, 6)
    black_piece_planes: tuple[int, int] = (6, 12)
    side_to_move_channel: int = 12
    white_kingside_castling_channel: int = 13
    white_queenside_castling_channel: int = 14
    black_kingside_castling_channel: int = 15
    black_queenside_castling_channel: int = 16
    en_passant_channel: int = 17

    def validate(self, channels: int) -> None:
        if self.encoding != "simple_18":
            raise ValueError(
                "LegalAutomorphismQuotientNetwork only has a registered channel "
                f"map for simple_18; got encoding={self.encoding!r}"
            )
        if channels != self.input_channels:
            raise ValueError(
                "LegalAutomorphismQuotientNetwork expects 18 simple_18 channels; "
                f"got channels={channels}"
            )


class LegalAutomorphismTransform(nn.Module):
    """Apply the four legal chess-rule automorphisms to a simple_18 tensor.

    The group ``G = <m, q> ~ C2 x C2`` has elements
    ``{identity, file_mirror, color_flip, file_mirror . color_flip}``.
    Each transform is a deterministic tensor permutation:

    * ``file_mirror`` flips files a<->h, swaps kingside <-> queenside
      castling rights for each color and mirrors the en-passant file.
    * ``color_flip`` reflects ranks 1<->8, swaps white and black piece
      planes, swaps the side-to-move bit, swaps White and Black castling
      while preserving king-/queen-side, and mirrors the en-passant rank.

    Unsupported encodings fail closed so the transformation is never
    silently applied to a tensor whose channel semantics are unknown.
    """

    def __init__(
        self,
        encoding: str = "simple_18",
        input_channels: int = 18,
        fail_closed_unknown_channels: bool = True,
    ) -> None:
        super().__init__()
        self.spec = Simple18AutomorphismSpec(
            encoding=encoding,
            input_channels=input_channels,
        )
        self.fail_closed_unknown_channels = bool(fail_closed_unknown_channels)

    def _validate(self, x: torch.Tensor) -> None:
        try:
            self.spec.validate(x.shape[1])
        except ValueError:
            if self.fail_closed_unknown_channels:
                raise

    def file_mirror(self, x: torch.Tensor) -> torch.Tensor:
        self._validate(x)
        s = self.spec
        flipped = torch.flip(x, dims=[3])
        out = flipped.clone()
        # Castling: swap king-side <-> queen-side for each color.
        out[:, s.white_kingside_castling_channel : s.white_kingside_castling_channel + 1] = (
            flipped[:, s.white_queenside_castling_channel : s.white_queenside_castling_channel + 1]
        )
        out[:, s.white_queenside_castling_channel : s.white_queenside_castling_channel + 1] = (
            flipped[:, s.white_kingside_castling_channel : s.white_kingside_castling_channel + 1]
        )
        out[:, s.black_kingside_castling_channel : s.black_kingside_castling_channel + 1] = (
            flipped[:, s.black_queenside_castling_channel : s.black_queenside_castling_channel + 1]
        )
        out[:, s.black_queenside_castling_channel : s.black_queenside_castling_channel + 1] = (
            flipped[:, s.black_kingside_castling_channel : s.black_kingside_castling_channel + 1]
        )
        # Side-to-move is broadcast over the board, so the file flip is a
        # no-op on it; the explicit copy keeps the contract documented.
        out[:, s.side_to_move_channel : s.side_to_move_channel + 1] = x[
            :, s.side_to_move_channel : s.side_to_move_channel + 1
        ]
        return out

    def color_flip(self, x: torch.Tensor) -> torch.Tensor:
        self._validate(x)
        s = self.spec
        rank_flipped = torch.flip(x, dims=[2])
        out = rank_flipped.clone()
        # Swap White and Black piece planes.
        white_lo, white_hi = s.white_piece_planes
        black_lo, black_hi = s.black_piece_planes
        out[:, white_lo:white_hi] = rank_flipped[:, black_lo:black_hi]
        out[:, black_lo:black_hi] = rank_flipped[:, white_lo:white_hi]
        # Swap side-to-move bit.
        out[:, s.side_to_move_channel : s.side_to_move_channel + 1] = (
            1.0 - x[:, s.side_to_move_channel : s.side_to_move_channel + 1]
        )
        # Swap White<->Black castling rights, preserving king/queen side.
        out[:, s.white_kingside_castling_channel : s.white_kingside_castling_channel + 1] = (
            x[:, s.black_kingside_castling_channel : s.black_kingside_castling_channel + 1]
        )
        out[:, s.white_queenside_castling_channel : s.white_queenside_castling_channel + 1] = (
            x[:, s.black_queenside_castling_channel : s.black_queenside_castling_channel + 1]
        )
        out[:, s.black_kingside_castling_channel : s.black_kingside_castling_channel + 1] = (
            x[:, s.white_kingside_castling_channel : s.white_kingside_castling_channel + 1]
        )
        out[:, s.black_queenside_castling_channel : s.black_queenside_castling_channel + 1] = (
            x[:, s.white_queenside_castling_channel : s.white_queenside_castling_channel + 1]
        )
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the orbit ``[identity, m, q, m . q]`` shape ``[B, 4, C, 8, 8]``."""
        self._validate(x)
        x_e = x
        x_m = self.file_mirror(x)
        x_q = self.color_flip(x)
        x_mq = self.file_mirror(x_q)
        return torch.stack([x_e, x_m, x_q, x_mq], dim=1)


class OrbitStacker(nn.Module):
    """Flatten the orbit tensor ``[B, G, C, 8, 8]`` into ``[B*G, C, 8, 8]``."""

    @staticmethod
    def forward(orbit: torch.Tensor) -> tuple[torch.Tensor, int]:
        if orbit.ndim != 5:
            raise ValueError(f"Expected orbit shape [B, G, C, 8, 8], got {tuple(orbit.shape)}")
        batch, group_size, channels, height, width = orbit.shape
        flat = orbit.reshape(batch * group_size, channels, height, width)
        return flat, group_size


class _ResidualBoardBlock(nn.Module):
    def __init__(self, channels: int, dropout: float, use_norm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_norm),
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_norm))
        if use_norm:
            layers.append(nn.BatchNorm2d(channels))
        self.block = nn.Sequential(*layers)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class SharedResidualBoardEncoder(nn.Module):
    """Shared convolutional tower applied to every orbit view.

    Architecture follows the markdown spec:
    ``Conv(C -> width) -> norm/GELU`` stem followed by ``num_blocks``
    residual blocks at ``width``, then global average pooling and a
    projection MLP to ``latent_dim``. The encoder produces a per-view
    latent ``[B*G, latent_dim]``.
    """

    def __init__(
        self,
        input_channels: int = 18,
        width: int = 96,
        num_blocks: int = 4,
        latent_dim: int = 192,
        dropout: float = 0.1,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.width = int(width)
        self.latent_dim = int(latent_dim)
        stem_layers: list[nn.Module] = [
            nn.Conv2d(input_channels, width, kernel_size=3, padding=1, bias=not use_norm),
        ]
        if use_norm:
            stem_layers.append(nn.BatchNorm2d(width))
        stem_layers.append(nn.GELU())
        self.stem = nn.Sequential(*stem_layers)
        self.blocks = nn.Sequential(
            *(_ResidualBoardBlock(width, dropout=dropout, use_norm=use_norm) for _ in range(num_blocks))
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        latent_layers: list[nn.Module] = [
            nn.Linear(width, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        ]
        if dropout > 0:
            latent_layers.append(nn.Dropout(dropout))
        self.to_latent = nn.Sequential(*latent_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        features = self.blocks(self.stem(x))
        pooled = self.pool(features).flatten(1)
        return self.to_latent(pooled)


class ReynoldsCharacterProjector(nn.Module):
    """Project the orbit latents onto the four C2 x C2 characters.

    Given orbit latents ``Z`` of shape ``[B, 4, D]`` the projector returns
    ``z_inv = mean(Z, dim=1)`` (the trivial character ``chi_0``) and the
    three nontrivial character components ``z_chars`` of shape
    ``[B, 3, D]`` computed as ``(1/|G|) sum_g chi(g) phi(g . s)``.
    """

    def __init__(self) -> None:
        super().__init__()
        table = torch.tensor(_CHARACTER_TABLE, dtype=torch.float32)
        self.register_buffer("character_table", table, persistent=False)

    def forward(self, orbit_latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if orbit_latents.ndim != 3 or orbit_latents.shape[1] != 4:
            raise ValueError(
                f"Expected orbit latents of shape [B, 4, D], got {tuple(orbit_latents.shape)}"
            )
        table = self.character_table.to(dtype=orbit_latents.dtype, device=orbit_latents.device)
        # weights shape: [4 (characters), 4 (group elements)] -> normalized by |G|.
        weights = table / table.shape[1]
        # Einsum: characters x group, B x group x dim -> B x characters x dim.
        characters = torch.einsum("cg,bgd->bcd", weights, orbit_latents)
        z_inv = characters[:, 0]
        z_chars = characters[:, 1:]
        return z_inv, z_chars


class LegalAutomorphismQuotientNet(nn.Module):
    """Bespoke implementation of idea i042.

    The forward pipeline is:

    1. Build the four-view legal automorphism orbit ``[B, 4, 18, 8, 8]``.
    2. Flatten to ``[4B, 18, 8, 8]`` and pass through a shared residual
       CNN encoder, returning per-view latents ``[B, 4, latent_dim]``.
       Optional ``orbit_chunk_size`` preserves the orbit pairing order
       under low-memory chunking by encoding each view separately.
    3. Reynolds + character projection: ``z_inv`` is the invariant latent
       and ``z_chars`` are the three nontrivial C2 x C2 components.
    4. Classification head consumes only ``z_inv`` and returns the puzzle
       logit(s). Diagnostic outputs include the character energy, the
       per-character L2 norms, the orbit latent variance and the optional
       ``character_penalty`` ready to add to the supervised loss.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding: str = "simple_18",
        width: int = 96,
        num_blocks: int = 4,
        latent_dim: int = 192,
        head_hidden_dim: int = 96,
        dropout: float = 0.1,
        use_norm: bool = True,
        char_penalty_weight: float = 0.0,
        orbit_chunk_size: int | None = None,
        fail_closed_unknown_channels: bool = True,
    ) -> None:
        super().__init__()
        if num_classes not in {1, 2}:
            raise ValueError(
                "LegalAutomorphismQuotientNet supports the puzzle_binary one-logit "
                "BCE contract or the two-class CE contract"
            )
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.latent_dim = int(latent_dim)
        self.char_penalty_weight = float(char_penalty_weight)
        self.orbit_chunk_size = int(orbit_chunk_size) if orbit_chunk_size else 0
        self.transform = LegalAutomorphismTransform(
            encoding=encoding,
            input_channels=input_channels,
            fail_closed_unknown_channels=fail_closed_unknown_channels,
        )
        self.stacker = OrbitStacker()
        self.encoder = SharedResidualBoardEncoder(
            input_channels=input_channels,
            width=width,
            num_blocks=num_blocks,
            latent_dim=latent_dim,
            dropout=dropout,
            use_norm=use_norm,
        )
        self.projector = ReynoldsCharacterProjector()
        head_layers: list[nn.Module] = [
            nn.Linear(latent_dim, head_hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        head_layers.append(nn.Linear(head_hidden_dim, num_classes))
        self.head = nn.Sequential(*head_layers)

    def _encode_orbit(self, orbit: torch.Tensor) -> torch.Tensor:
        batch, group_size, channels, height, width = orbit.shape
        if self.orbit_chunk_size <= 0 or self.orbit_chunk_size >= group_size:
            flat, _ = self.stacker(orbit)
            latents = self.encoder(flat)
            return latents.reshape(batch, group_size, self.latent_dim)
        per_view: list[torch.Tensor] = []
        for view_idx in range(group_size):
            per_view.append(self.encoder(orbit[:, view_idx]))
        return torch.stack(per_view, dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        orbit = self.transform(x)
        z = self._encode_orbit(orbit)
        z_inv, z_chars = self.projector(z)
        head_logits = self.head(z_inv)
        logits = format_logits(head_logits, self.num_classes)

        # Diagnostics.
        char_energy = z_chars.pow(2).mean(dim=(1, 2))
        # Per-nontrivial-character L2 norm; index 0=m, 1=q, 2=mq.
        nontrivial_norms = z_chars.norm(dim=2)
        invariant_norm = z_inv.norm(dim=1)
        z_centered = z - z_inv.unsqueeze(1)
        orbit_variance = z_centered.pow(2).mean(dim=(1, 2))
        # Sum-of-squares character penalty matching ``R_char`` in the
        # math thesis: sum over nontrivial characters of squared norm.
        character_penalty = z_chars.pow(2).sum(dim=(1, 2))

        return {
            "logits": logits,
            "z_invariant": z_inv,
            "invariant_norm": invariant_norm,
            "character_energy": char_energy,
            "character_norms": nontrivial_norms,
            "file_mirror_character_norm": nontrivial_norms[:, 0],
            "color_flip_character_norm": nontrivial_norms[:, 1],
            "joint_character_norm": nontrivial_norms[:, 2],
            "orbit_variance": orbit_variance,
            "character_penalty": character_penalty,
        }


def build_legal_automorphism_quotient_network_from_config(
    config: dict[str, Any],
) -> LegalAutomorphismQuotientNet:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return LegalAutomorphismQuotientNet(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        encoding=str(cfg.get("encoding", "simple_18")),
        width=int(cfg.get("width", cfg.get("channels", 96))),
        num_blocks=int(cfg.get("num_blocks", cfg.get("depth", 4))),
        latent_dim=int(cfg.get("latent_dim", cfg.get("hidden_dim", 192))),
        head_hidden_dim=int(cfg.get("head_hidden_dim", cfg.get("head_hidden", 96))),
        dropout=float(cfg.get("dropout", 0.1)),
        use_norm=bool(cfg.get("use_norm", cfg.get("use_batchnorm", True))),
        char_penalty_weight=float(cfg.get("char_penalty_weight", 0.0)),
        orbit_chunk_size=cfg.get("orbit_chunk_size"),
        fail_closed_unknown_channels=bool(cfg.get("fail_closed_unknown_channels", True)),
    )


# Backwards-compat alias for callers that follow the markdown's named
# builder convention; the registry uses
# ``build_legal_automorphism_quotient_network_from_config``.
def build_legal_automorphism_quotient_cnn(config: dict[str, Any]) -> LegalAutomorphismQuotientNet:
    return build_legal_automorphism_quotient_network_from_config(config)
