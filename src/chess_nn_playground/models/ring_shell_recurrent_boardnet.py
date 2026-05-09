"""Ring-Shell Recurrent BoardNet for idea i168.

Faithful implementation of the markdown thesis: important chess context
radiates from a small set of *anchors* (kings, center, edges, promotion
zones).  We summarize the trunk feature map in concentric *rings* (Chebyshev
shells) around each anchor, then feed the radial sequence of shell pools to a
small GRU.  The recurrence aggregates information from the anchor outward,
shell by shell, before a head over the per-anchor final hidden states emits
the puzzle logit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


WHITE_KING_PLANE = 5
BLACK_KING_PLANE = 11

# Static anchors as (row, col) coordinates over the 8x8 board with
# row 0 == rank 8 (matching ``fen_to_simple_18``):
#   - center: pooled mid-squares (d4, e4, d5, e5).
#   - white_promotion: rank 1 (where white pawns promote == row 0).
#   - black_promotion: rank 8 (where black pawns promote == row 7).
#   - queenside_edge / kingside_edge: file a / file h centers.
STATIC_ANCHORS: tuple[tuple[str, float, float], ...] = (
    ("center", 3.5, 3.5),
    ("white_promotion", 0.0, 3.5),
    ("black_promotion", 7.0, 3.5),
    ("queenside_edge", 3.5, 0.0),
    ("kingside_edge", 3.5, 7.0),
)
DYNAMIC_ANCHOR_NAMES: tuple[str, ...] = ("white_king", "black_king")
ANCHOR_NAMES: tuple[str, ...] = DYNAMIC_ANCHOR_NAMES + tuple(name for name, _, _ in STATIC_ANCHORS)


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


@dataclass(frozen=True)
class RingShellRecurrentBoardNetConfig:
    input_channels: int = 18
    num_classes: int = 1
    channels: int = 64
    hidden_dim: int = 96
    depth: int = 2
    dropout: float = 0.1
    use_batchnorm: bool = True
    num_rings: int = 8
    rnn_hidden: int = 48


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=not use_batchnorm,
        )
        self.norm = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.norm(self.conv(x))))


def _build_static_ring_masks(num_rings: int, height: int = 8, width: int = 8) -> torch.Tensor:
    """Pre-computed (A_static, R, H, W) ring-membership masks for static anchors."""
    rows = torch.arange(height, dtype=torch.float32).view(1, height, 1)
    cols = torch.arange(width, dtype=torch.float32).view(1, 1, width)
    masks = torch.zeros(len(STATIC_ANCHORS), num_rings, height, width, dtype=torch.float32)
    for idx, (_, anchor_row, anchor_col) in enumerate(STATIC_ANCHORS):
        d_row = (rows - anchor_row).abs()
        d_col = (cols - anchor_col).abs()
        cheb = torch.maximum(d_row, d_col).squeeze(0)
        ring_idx = cheb.floor().long().clamp(max=num_rings - 1)
        for r in range(num_rings):
            masks[idx, r] = (ring_idx == r).to(torch.float32)
    return masks


def _build_static_anchor_positions() -> torch.Tensor:
    return torch.tensor(
        [[anchor_row, anchor_col] for _, anchor_row, anchor_col in STATIC_ANCHORS],
        dtype=torch.float32,
    )


def _piece_centroid(plane: torch.Tensor, row_grid: torch.Tensor, col_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute soft centroid of a single piece plane.

    Returns ``(row, col, mass)`` where ``mass`` is the sum of the plane.  When
    a king is missing (``mass == 0``) the centroid falls back to the board's
    geometric center so downstream ring construction is well defined.
    """
    mass = plane.sum(dim=(-1, -2))  # (B,)
    safe_mass = mass.clamp(min=1.0)
    row = (plane * row_grid).sum(dim=(-1, -2)) / safe_mass
    col = (plane * col_grid).sum(dim=(-1, -2)) / safe_mass
    fallback = (mass <= 0).to(row.dtype)
    row = row * (1 - fallback) + 3.5 * fallback
    col = col * (1 - fallback) + 3.5 * fallback
    return row, col, mass


def _dynamic_ring_masks(
    anchor_row: torch.Tensor,
    anchor_col: torch.Tensor,
    num_rings: int,
    row_grid: torch.Tensor,
    col_grid: torch.Tensor,
) -> torch.Tensor:
    """Build (B, R, H, W) Chebyshev ring masks for a single dynamic anchor."""
    d_row = (row_grid.unsqueeze(0) - anchor_row.view(-1, 1, 1)).abs()
    d_col = (col_grid.unsqueeze(0) - anchor_col.view(-1, 1, 1)).abs()
    cheb = torch.maximum(d_row, d_col)
    ring_idx = cheb.floor().long().clamp(max=num_rings - 1)
    masks = torch.zeros(
        anchor_row.shape[0], num_rings, row_grid.shape[0], row_grid.shape[1],
        dtype=row_grid.dtype, device=row_grid.device,
    )
    masks.scatter_(1, ring_idx.unsqueeze(1), 1.0)
    return masks


class RingShellRecurrentBoardNet(nn.Module):
    """Ring-Shell Recurrent BoardNet classifier.

    Pipeline per the thesis:

    1. ``H = Trunk(x)`` -- compact convolutional encoder of the 18-plane
       board to a feature map of width ``channels``.
    2. For each anchor ``i \\in {white_king, black_king, center, promotion
       zones, edges}`` and each ring ``r = 0..R-1``, pool the trunk by the
       Chebyshev shell mask:

           ``f_{i,r} = (1 / |S_{i,r}|) \\sum_{(h,w) \\in S_{i,r}} H[:, h, w]``

       King anchors use the soft centroid of their piece plane so the rings
       follow the king around the board.
    3. A small shared GRU consumes the radial sequence ``(f_{i,0}, ...,
       f_{i,R-1})`` per anchor, producing a hidden state ``h_i`` summarising
       the full radial profile.
    4. Concatenated anchor hidden states feed a LayerNorm-MLP head that
       returns the puzzle logit.

    The forward pass also returns per-anchor / per-ring diagnostics so
    downstream tooling can inspect the radial trail without re-running the
    model.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        num_rings: int = 8,
        rnn_hidden: int = 48,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if num_rings < 2:
            raise ValueError("num_rings must be >= 2 for a non-trivial radial sequence")
        if rnn_hidden < 1:
            raise ValueError("rnn_hidden must be >= 1")

        self.config = RingShellRecurrentBoardNetConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            num_rings=num_rings,
            rnn_hidden=rnn_hidden,
        )
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.depth = int(depth)
        self.num_rings = int(num_rings)
        self.rnn_hidden = int(rnn_hidden)
        self.num_anchors = len(ANCHOR_NAMES)
        self.num_dynamic_anchors = len(DYNAMIC_ANCHOR_NAMES)
        self.num_static_anchors = len(STATIC_ANCHORS)

        if input_channels <= max(WHITE_KING_PLANE, BLACK_KING_PLANE):
            raise ValueError(
                f"input_channels={input_channels} must be > {max(WHITE_KING_PLANE, BLACK_KING_PLANE)} "
                "to expose the king planes for dynamic anchors"
            )

        self.stem = nn.Conv2d(
            input_channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=not use_batchnorm,
        )
        self.stem_norm = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.stem_activation = nn.GELU()
        self.trunk = nn.Sequential(
            *[
                _ConvBlock(channels, channels, dropout=dropout, use_batchnorm=use_batchnorm)
                for _ in range(depth)
            ]
        )

        # Per-anchor learned projection lets the recurrent model distinguish
        # which anchor a given radial sequence belongs to.  Projection adds a
        # bias plus a small linear remap of the pooled ring features.
        self.ring_proj = nn.Linear(channels, channels)
        self.anchor_bias = nn.Parameter(torch.zeros(self.num_anchors, channels))
        nn.init.normal_(self.anchor_bias, std=0.02)

        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
        )

        head_in = self.num_anchors * rnn_hidden
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

        # Pre-computed static ring masks and anchor positions.
        static_masks = _build_static_ring_masks(num_rings)
        static_anchor_positions = _build_static_anchor_positions()
        row_grid = torch.arange(8, dtype=torch.float32).view(8, 1).expand(8, 8).contiguous()
        col_grid = torch.arange(8, dtype=torch.float32).view(1, 8).expand(8, 8).contiguous()
        self.register_buffer("static_ring_masks", static_masks, persistent=False)
        self.register_buffer("static_anchor_positions", static_anchor_positions, persistent=False)
        self.register_buffer("row_grid", row_grid, persistent=False)
        self.register_buffer("col_grid", col_grid, persistent=False)

    def _ring_pool(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pool ``features`` (B, C, H, W) with ring ``mask`` (B, R, H, W)."""
        b, c, h, w = features.shape
        r = mask.shape[1]
        flat_features = features.reshape(b, c, h * w)
        flat_mask = mask.reshape(b, r, h * w)
        # (B, R, H*W) x (B, H*W, C) -> (B, R, C)
        ring_sum = torch.bmm(flat_mask, flat_features.transpose(1, 2))
        counts = flat_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return ring_sum / counts

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        h = self.stem_activation(self.stem_norm(self.stem(board)))
        h = self.trunk(h)  # (B, C, 8, 8)
        b = h.shape[0]
        device = h.device
        dtype = h.dtype

        row_grid = self.row_grid
        col_grid = self.col_grid
        white_plane = board[:, WHITE_KING_PLANE]
        black_plane = board[:, BLACK_KING_PLANE]
        white_row, white_col, white_mass = _piece_centroid(white_plane, row_grid, col_grid)
        black_row, black_col, black_mass = _piece_centroid(black_plane, row_grid, col_grid)

        white_masks = _dynamic_ring_masks(white_row, white_col, self.num_rings, row_grid, col_grid)
        black_masks = _dynamic_ring_masks(black_row, black_col, self.num_rings, row_grid, col_grid)
        # (B, A_static, R, 8, 8) -- broadcast static masks across batch.
        static_masks = self.static_ring_masks.to(device=device, dtype=row_grid.dtype)
        static_masks_b = static_masks.unsqueeze(0).expand(b, -1, -1, -1, -1)
        # Stack dynamic anchors: (B, A_dyn, R, 8, 8)
        dynamic_masks = torch.stack([white_masks, black_masks], dim=1)
        all_masks = torch.cat([dynamic_masks, static_masks_b], dim=1)
        # all_masks: (B, A, R, 8, 8)
        a = all_masks.shape[1]
        ring_counts = all_masks.sum(dim=(-1, -2))  # (B, A, R)

        # Pool the trunk feature map per anchor and per ring.
        flat_h = h.reshape(b, self.channels, 64)  # (B, C, 64)
        flat_masks = all_masks.reshape(b, a * self.num_rings, 64)  # (B, A*R, 64)
        ring_sum = torch.bmm(flat_masks, flat_h.transpose(1, 2))  # (B, A*R, C)
        counts = flat_masks.sum(dim=-1, keepdim=True).clamp(min=1.0)
        ring_pool = (ring_sum / counts).view(b, a, self.num_rings, self.channels)
        ring_features = self.ring_proj(ring_pool)  # (B, A, R, C)
        # Add per-anchor bias so the shared GRU can tell anchors apart.
        ring_features = ring_features + self.anchor_bias.view(1, a, 1, self.channels).to(dtype=dtype)

        # Run the shared GRU per anchor by folding (B, A) into the batch axis.
        gru_input = ring_features.reshape(b * a, self.num_rings, self.channels)
        gru_input = gru_input.to(dtype=dtype)
        gru_output, gru_final = self.gru(gru_input)
        anchor_hidden = gru_final.squeeze(0).view(b, a, self.rnn_hidden)
        anchor_hidden_progression = gru_output.reshape(b, a, self.num_rings, self.rnn_hidden)

        head_in = anchor_hidden.reshape(b, a * self.rnn_hidden)
        raw_logits = self.head(head_in)
        logits = _format_logits(raw_logits, self.num_classes)

        ring_energy = ring_features.pow(2).mean(dim=-1)  # (B, A, R)
        anchor_positions = torch.stack(
            [
                torch.stack([white_row, white_col], dim=-1),
                torch.stack([black_row, black_col], dim=-1),
            ],
            dim=1,
        )
        static_positions = self.static_anchor_positions.unsqueeze(0).expand(b, -1, -1).to(dtype=ring_features.dtype)
        anchor_positions = torch.cat([anchor_positions.to(dtype=ring_features.dtype), static_positions], dim=1)

        anchor_dynamic_mass = torch.stack([white_mass, black_mass], dim=-1)  # (B, 2)
        anchor_present = (anchor_dynamic_mass > 0).to(dtype=ring_features.dtype)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "trunk_features": h,
            "ring_features": ring_features,
            "ring_pool": ring_pool,
            "ring_energy": ring_energy,
            "ring_counts": ring_counts,
            "anchor_hidden": anchor_hidden,
            "anchor_hidden_progression": anchor_hidden_progression,
            "anchor_positions": anchor_positions,
            "anchor_dynamic_mass": anchor_dynamic_mass,
            "anchor_dynamic_present": anchor_present,
        }

        anchor_norm = anchor_hidden.pow(2).mean(dim=-1)  # (B, A)
        diagnostics["anchor_hidden_energy"] = anchor_norm
        diagnostics["mean_anchor_hidden_energy"] = anchor_norm.mean(dim=-1)
        diagnostics["mean_ring_energy"] = ring_energy.mean(dim=(-1, -2))
        diagnostics["radial_progression_energy"] = anchor_hidden_progression.pow(2).mean(dim=-1)
        diagnostics["depth_levels"] = logits.new_full(logits.shape, float(self.depth))
        diagnostics["ring_levels"] = logits.new_full(logits.shape, float(self.num_rings))
        diagnostics["anchor_levels"] = logits.new_full(logits.shape, float(self.num_anchors))

        if self.num_classes == 1:
            diagnostics["prob"] = torch.sigmoid(logits)
        return diagnostics


def build_ring_shell_recurrent_boardnet_from_config(
    config: dict[str, Any],
) -> RingShellRecurrentBoardNet:
    return RingShellRecurrentBoardNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        num_rings=int(config.get("num_rings", 8)),
        rnn_hidden=int(config.get("rnn_hidden", 48)),
    )
