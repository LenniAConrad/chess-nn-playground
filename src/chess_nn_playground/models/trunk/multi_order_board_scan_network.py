"""Multi-Order Board Scan Network for idea i156.

The board is read as a small set of fixed-length sequences. A shared
convolutional trunk produces per-square features. Those features are
re-ordered by five scan orders -- ``rank_major``, ``file_major``,
``diagonal``, ``spiral_from_king``, and ``center_out`` -- and a single
shared bidirectional GRU consumes each ordering. The order-pooled GRU
summaries are concatenated and fed to a small classifier that produces
the puzzle logit.

The first four orders are deterministic permutations of the 64 squares
and live as static buffers. ``spiral_from_king`` depends on the
side-to-move's king square; a precomputed (64, 64) lookup table maps
each king square to its Chebyshev-distance ordering of the rest of the
board.

Each scan order receives its own learned scan embedding, so the shared
GRU can distinguish the order it is currently scanning while still
sharing parameters across orders -- the markdown thesis' "shared
sequence model over fixed board orders".
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


SCAN_ORDER_NAMES: tuple[str, ...] = (
    "rank_major",
    "file_major",
    "diagonal",
    "spiral_from_king",
    "center_out",
)
NUM_SCANS = len(SCAN_ORDER_NAMES)
NUM_SQUARES = 64
WHITE_KING_PLANE = 5
BLACK_KING_PLANE = 11
SIDE_TO_MOVE_PLANE = 12


def _rank_major_order() -> list[int]:
    return [rank * 8 + file for rank in range(8) for file in range(8)]


def _file_major_order() -> list[int]:
    return [rank * 8 + file for file in range(8) for rank in range(8)]


def _diagonal_order() -> list[int]:
    indexed = [(rank + file, rank, rank * 8 + file) for rank in range(8) for file in range(8)]
    indexed.sort()
    return [idx for _, _, idx in indexed]


def _center_out_order() -> list[int]:
    indexed = []
    for rank in range(8):
        for file in range(8):
            dr = abs(rank * 2 + 1 - 8) / 2.0
            df = abs(file * 2 + 1 - 8) / 2.0
            chebyshev = max(dr, df)
            indexed.append((chebyshev, rank * 8 + file, rank * 8 + file))
    indexed.sort()
    return [idx for _, _, idx in indexed]


def _spiral_from_king_table() -> list[list[int]]:
    """For each king square, return the squares ordered by Chebyshev distance.

    Tiebreak by rank-major square index so the order is deterministic.
    """
    table: list[list[int]] = []
    for king_sq in range(NUM_SQUARES):
        kr, kf = divmod(king_sq, 8)
        indexed: list[tuple[int, int, int]] = []
        for sq in range(NUM_SQUARES):
            r, f = divmod(sq, 8)
            chebyshev = max(abs(r - kr), abs(f - kf))
            indexed.append((chebyshev, sq, sq))
        indexed.sort()
        table.append([idx for _, _, idx in indexed])
    return table


class MultiOrderBoardScanNetwork(nn.Module):
    """Multi-order board scan network for the puzzle_binary contract.

    Pipeline:

    1. **Stem.** Two normalised rank/file coordinate planes are
       concatenated to the input. A ``3x3 Conv2d -> [BatchNorm2d ->]
       ReLU`` stack of ``depth`` blocks lifts the
       ``(input_channels + 2)`` planes to the trunk channel dimension
       while preserving the ``8x8`` layout.
    2. **Square tokens.** The trunk is reshaped to ``(B, 64, channels)``
       to act as the per-square token sequence (rank-major).
    3. **Scan permutations.** Five scan-order permutations of the 64
       squares are computed. Four are static (``rank_major``,
       ``file_major``, ``diagonal``, ``center_out``); the fifth
       (``spiral_from_king``) is per-sample, looked up from the
       side-to-move king square via a precomputed table.
    4. **Shared bidirectional GRU.** Each per-order sequence is gathered
       and summed with a learned scan-position embedding of shape
       ``(num_scans, 64, channels)`` so the shared GRU can tell which
       order it is currently scanning. The same
       ``nn.GRU(channels, gru_hidden, bidirectional=True)`` consumes
       every order -- this is the markdown's "shared sequence model
       over fixed board orders".
    5. **Order pooling.** For each scan order, the GRU output of shape
       ``(B, 64, 2 * gru_hidden)`` is mean-pooled along the sequence
       axis, yielding one summary vector per order.
    6. **Readout.** The order summaries are concatenated and run
       through a small ``Linear -> ReLU -> Dropout -> Linear`` head to
       produce the single puzzle logit.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        gru_hidden_dim: int = 64,
        num_gru_layers: int = 1,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "MultiOrderBoardScanNetwork supports the puzzle_binary one-logit contract"
            )
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1 or hidden_dim < 1 or gru_hidden_dim < 1:
            raise ValueError("channels, hidden_dim and gru_hidden_dim must be positive")
        if num_gru_layers < 1:
            raise ValueError("num_gru_layers must be >= 1")

        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.num_gru_layers = int(num_gru_layers)
        self.dropout_p = float(dropout)
        self.num_scans = NUM_SCANS
        self.scan_order_names = SCAN_ORDER_NAMES

        # Trunk: stem (with 2 coordinate planes) + (depth - 1) more blocks.
        stem_in = self.input_channels + 2
        layers: list[nn.Module] = [self._conv_block(stem_in, self.channels, use_batchnorm)]
        for _ in range(self.depth - 1):
            layers.append(self._conv_block(self.channels, self.channels, use_batchnorm))
        self.trunk = nn.Sequential(*layers)

        # Static permutation buffers, shape (num_static_scans, 64).
        static_orders = torch.tensor(
            [
                _rank_major_order(),
                _file_major_order(),
                _diagonal_order(),
                _center_out_order(),
            ],
            dtype=torch.long,
        )
        self.register_buffer("static_scan_perms", static_orders, persistent=False)

        # King-spiral lookup table, shape (64, 64).
        spiral_table = torch.tensor(_spiral_from_king_table(), dtype=torch.long)
        self.register_buffer("spiral_table", spiral_table, persistent=False)

        # Per-scan, per-position embedding so the shared GRU can tell
        # the orders apart and benefit from learned positional cues
        # within each order.
        self.scan_position_embedding = nn.Parameter(
            torch.zeros(self.num_scans, NUM_SQUARES, self.channels)
        )
        nn.init.normal_(self.scan_position_embedding, std=0.02)

        self.gru = nn.GRU(
            input_size=self.channels,
            hidden_size=self.gru_hidden_dim,
            num_layers=self.num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        head_in = self.num_scans * 2 * self.gru_hidden_dim
        head_layers: list[nn.Module] = [
            nn.Linear(head_in, self.hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if self.dropout_p > 0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.hidden_dim, 1))
        self.head = nn.Sequential(*head_layers)

    # -- helpers ----------------------------------------------------

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int, use_batchnorm: bool) -> nn.Module:
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not use_batchnorm,
            )
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    @staticmethod
    def _coordinate_channels(x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        device = x.device
        dtype = x.dtype
        rank = (
            torch.linspace(-1.0, 1.0, steps=8, device=device, dtype=dtype)
            .view(1, 1, 8, 1)
            .expand(batch, 1, 8, 8)
        )
        file = (
            torch.linspace(-1.0, 1.0, steps=8, device=device, dtype=dtype)
            .view(1, 1, 1, 8)
            .expand(batch, 1, 8, 8)
        )
        return torch.cat([rank, file], dim=1)

    def _friendly_king_square(self, board: torch.Tensor) -> torch.Tensor:
        """Return the side-to-move king square index per sample.

        Shape: ``(B,)``, ``long``. Side-to-move is decoded from plane
        ``SIDE_TO_MOVE_PLANE`` (1 means white-to-move). The friendly
        king plane is selected accordingly.
        """
        white_to_move = (
            board[:, SIDE_TO_MOVE_PLANE].mean(dim=(-1, -2)) > 0.5
        )  # (B,)
        white_king = board[:, WHITE_KING_PLANE].flatten(start_dim=1)  # (B, 64)
        black_king = board[:, BLACK_KING_PLANE].flatten(start_dim=1)  # (B, 64)
        friendly = torch.where(white_to_move.unsqueeze(-1), white_king, black_king)

        # Argmax over squares; if no king is present (e.g. degenerate
        # zero tensor) fall back to a fixed central square so downstream
        # gather is well-defined. The lookup is detached and used only
        # to index a fixed permutation table.
        has_king = (friendly.sum(dim=-1) > 0)
        argmax_sq = friendly.argmax(dim=-1)
        default_sq = torch.full_like(argmax_sq, 4 * 8 + 4)  # square e4 as fallback
        return torch.where(has_king, argmax_sq, default_sq).detach()

    def _build_scan_perms(self, board: torch.Tensor) -> torch.Tensor:
        """Return scan permutations of shape ``(num_scans, B, 64)``."""
        batch = board.shape[0]
        device = board.device
        # Static orders: (4, 64) -> (4, B, 64).
        static = self.static_scan_perms.unsqueeze(1).expand(-1, batch, -1).to(device)
        # Spiral-from-king: per-sample (B, 64).
        king_sq = self._friendly_king_square(board)
        spiral = self.spiral_table.to(device).index_select(0, king_sq)  # (B, 64)
        spiral = spiral.unsqueeze(0)  # (1, B, 64)
        # Insert spiral as the 4th order to match SCAN_ORDER_NAMES.
        return torch.cat([static[:3], spiral, static[3:4]], dim=0)

    def _scan_sequences(
        self, square_tokens: torch.Tensor, scan_perms: torch.Tensor
    ) -> torch.Tensor:
        """Reorder square tokens by each scan permutation.

        Args:
          square_tokens: shape ``(B, 64, channels)``.
          scan_perms: shape ``(num_scans, B, 64)``.

        Returns:
          shape ``(num_scans, B, 64, channels)``.
        """
        batch, num_squares, channels = square_tokens.shape
        # Expand permutations to (num_scans, B, 64, channels) for gather.
        gather_index = scan_perms.unsqueeze(-1).expand(-1, -1, -1, channels)
        # Repeat square_tokens for each scan: (num_scans, B, 64, channels).
        repeated = square_tokens.unsqueeze(0).expand(self.num_scans, -1, -1, -1)
        return torch.gather(repeated, 2, gather_index)

    # -- forward ----------------------------------------------------

    def forward(self, board: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(board, self.spec)
        coords = self._coordinate_channels(x)
        h = self.trunk(torch.cat([x, coords], dim=1))  # (B, channels, 8, 8)

        batch = h.shape[0]
        # Square tokens: rank-major ordering by default.
        square_tokens = h.flatten(start_dim=2).transpose(1, 2).contiguous()  # (B, 64, channels)

        scan_perms = self._build_scan_perms(x)  # (num_scans, B, 64)
        scan_seqs = self._scan_sequences(square_tokens, scan_perms)  # (num_scans, B, 64, channels)

        # Add per-scan positional embedding.
        pos = self.scan_position_embedding.unsqueeze(1)  # (num_scans, 1, 64, channels)
        scan_seqs = scan_seqs + pos

        # Stack into (num_scans * B, 64, channels) and run the shared GRU once.
        flat = scan_seqs.reshape(self.num_scans * batch, NUM_SQUARES, self.channels)
        gru_out, _ = self.gru(flat)  # (num_scans * B, 64, 2 * gru_hidden)
        gru_out = gru_out.view(self.num_scans, batch, NUM_SQUARES, 2 * self.gru_hidden_dim)

        scan_summaries = gru_out.mean(dim=2)  # (num_scans, B, 2 * gru_hidden)
        # Concatenate the per-order summaries into a single feature vector.
        concat = scan_summaries.permute(1, 0, 2).reshape(batch, -1)

        logits = self.head(concat).view(-1)

        # Diagnostics.
        scan_summary_norms = scan_summaries.detach().norm(dim=-1).transpose(0, 1)  # (B, num_scans)
        king_sq = self._friendly_king_square(x)
        return {
            "logits": logits,
            "logit": logits,
            "prob": torch.sigmoid(logits),
            "latent": h,
            "square_tokens": square_tokens,
            "scan_perms": scan_perms.detach(),
            "scan_summaries": scan_summaries.permute(1, 0, 2),  # (B, num_scans, 2 * gru_hidden)
            "scan_summary_norms": scan_summary_norms,
            "friendly_king_square": king_sq,
        }


def build_multi_order_board_scan_network_from_config(
    config: dict[str, Any],
) -> MultiOrderBoardScanNetwork:
    cfg = dict(config)
    cfg.setdefault("input_channels", 18)
    cfg.setdefault("num_classes", 1)
    return MultiOrderBoardScanNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        gru_hidden_dim=int(cfg.get("gru_hidden_dim", 64)),
        num_gru_layers=int(cfg.get("num_gru_layers", 1)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
