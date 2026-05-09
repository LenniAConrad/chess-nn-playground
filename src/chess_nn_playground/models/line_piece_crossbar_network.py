"""Line-Piece Crossbar Network for idea i171.

Faithful implementation of the markdown thesis: build 64 piece (square)
tokens and 46 line tokens (8 ranks + 8 files + 15 diagonals +
15 anti-diagonals).  Messages pass *only* through the deterministic
piece-line incidence matrix that records which lines each square lies on
(every square lies on exactly 4 lines: its rank, its file, its diagonal,
and its anti-diagonal).  Stacked piece -> line -> piece exchanges replace
attention and convolutional mixing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


NUM_RANKS = 8
NUM_FILES = 8
NUM_DIAGS = 15  # r + c in [0, 14]
NUM_ANTIDIAGS = 15  # r - c + 7 in [0, 14]
NUM_LINES = NUM_RANKS + NUM_FILES + NUM_DIAGS + NUM_ANTIDIAGS  # 46
NUM_PIECES = 64  # one piece-token per square


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _build_incidence() -> torch.Tensor:
    """Return the (64, 46) binary piece-line incidence matrix.

    Square s at (r, c) lies on:
      - rank r            -> line index r              (0..7)
      - file c            -> line index 8 + c          (8..15)
      - diagonal r+c      -> line index 16 + (r + c)   (16..30)
      - anti-diagonal r-c -> line index 31 + (r - c + 7) (31..45)
    """
    incidence = torch.zeros(NUM_PIECES, NUM_LINES, dtype=torch.float32)
    for r in range(8):
        for c in range(8):
            s = r * 8 + c
            incidence[s, r] = 1.0
            incidence[s, NUM_RANKS + c] = 1.0
            incidence[s, NUM_RANKS + NUM_FILES + (r + c)] = 1.0
            incidence[s, NUM_RANKS + NUM_FILES + NUM_DIAGS + (r - c + 7)] = 1.0
    return incidence


def _build_line_type_ids() -> torch.Tensor:
    """Return a (46,) long tensor tagging each line with its type id."""
    ids = torch.zeros(NUM_LINES, dtype=torch.long)
    ids[:NUM_RANKS] = 0
    ids[NUM_RANKS : NUM_RANKS + NUM_FILES] = 1
    ids[NUM_RANKS + NUM_FILES : NUM_RANKS + NUM_FILES + NUM_DIAGS] = 2
    ids[NUM_RANKS + NUM_FILES + NUM_DIAGS :] = 3
    return ids


@dataclass(frozen=True)
class LinePieceCrossbarNetworkConfig:
    input_channels: int = 18
    num_classes: int = 1
    channels: int = 64
    hidden_dim: int = 96
    depth: int = 2
    dropout: float = 0.1
    use_batchnorm: bool = True


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


class _CrossbarLayer(nn.Module):
    """One round of piece -> line -> piece message passing.

    Pieces aggregate onto lines via the incidence matrix (mean over the
    pieces on a line), then each piece reads back the aggregate of the
    lines it lies on.  The only cross-square communication is through the
    deterministic incidence; there is no attention or convolution inside
    the layer.
    """

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.channels = int(channels)
        self.piece_to_line = nn.Linear(channels, channels)
        self.line_to_piece = nn.Linear(channels, channels)
        self.line_norm = nn.LayerNorm(channels)
        self.piece_norm = nn.LayerNorm(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        piece_tokens: torch.Tensor,
        line_tokens: torch.Tensor,
        line_to_piece_weight: torch.Tensor,
        piece_to_line_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # piece_tokens: (B, 64, C)   line_tokens: (B, 46, C)
        # piece_to_line_weight: (46, 64) row-normalized incidence^T
        # line_to_piece_weight: (64, 46) row-normalized incidence (each row sums to 1)

        # --- Pieces -> lines ---
        msg_p = self.piece_to_line(piece_tokens)  # (B, 64, C)
        # mean of pieces on each line
        line_msg = torch.einsum("lp,bpc->blc", piece_to_line_weight, msg_p)  # (B, 46, C)
        new_lines = self.line_norm(line_tokens + self.dropout(self.activation(line_msg)))

        # --- Lines -> pieces ---
        msg_l = self.line_to_piece(new_lines)  # (B, 46, C)
        piece_msg = torch.einsum("pl,blc->bpc", line_to_piece_weight, msg_l)  # (B, 64, C)
        new_pieces = self.piece_norm(piece_tokens + self.dropout(self.activation(piece_msg)))

        return {
            "pieces": new_pieces,
            "lines": new_lines,
            "line_message": line_msg,
            "piece_message": piece_msg,
        }


class LinePieceCrossbarNetwork(nn.Module):
    """Line-Piece Crossbar Network classifier.

    1. Convolutional stem produces a per-square feature map of width
       ``channels``.
    2. Square features are reshaped into 64 piece tokens.  46 line tokens
       are seeded from a learned per-line embedding plus a learned
       per-line-type tag (rank/file/diag/antidiag).
    3. ``depth`` crossbar layers exchange messages piece -> line -> piece
       using only the deterministic 64x46 incidence matrix (each row
       normalized to a mean).
    4. Piece tokens are mean-pooled and a small MLP head emits the puzzle
       logit.
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
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")

        self.config = LinePieceCrossbarNetworkConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.depth = int(depth)

        self.stem = _ConvBlock(input_channels, channels, dropout=dropout, use_batchnorm=use_batchnorm)

        # Deterministic, non-trainable structure.
        incidence = _build_incidence()  # (64, 46)
        self.register_buffer("incidence", incidence, persistent=False)
        # Row-normalize so each piece reads a *mean* of its 4 lines, and
        # each line reads a *mean* of the pieces on it.  These are
        # constant by construction (every piece is on 4 lines; every line
        # has 8 pieces -- ranks/files have 8, diagonals have 1..8 squares
        # so we use the actual per-row counts).
        line_counts = incidence.sum(dim=0).clamp(min=1.0)  # (46,)
        piece_counts = incidence.sum(dim=1).clamp(min=1.0)  # (64,) == 4 everywhere
        line_to_piece_weight = incidence / piece_counts.unsqueeze(1)  # (64, 46)
        piece_to_line_weight = incidence.t() / line_counts.unsqueeze(1)  # (46, 64)
        self.register_buffer("line_to_piece_weight", line_to_piece_weight, persistent=False)
        self.register_buffer("piece_to_line_weight", piece_to_line_weight, persistent=False)
        self.register_buffer("line_type_ids", _build_line_type_ids(), persistent=False)

        # Learned per-line embedding (positional prior for each of the 46
        # line tokens) + per-line-type embedding (rank/file/diag/anti).
        self.line_embedding = nn.Parameter(torch.zeros(NUM_LINES, channels))
        self.line_type_embedding = nn.Parameter(torch.zeros(4, channels))
        nn.init.normal_(self.line_embedding, std=0.02)
        nn.init.normal_(self.line_type_embedding, std=0.02)

        self.layers = nn.ModuleList(
            [_CrossbarLayer(channels=channels, dropout=dropout) for _ in range(depth)]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def _initial_line_tokens(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        line_emb = self.line_embedding + self.line_type_embedding[self.line_type_ids]
        return line_emb.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        h = self.stem(board)  # (B, C, 8, 8)
        b = h.shape[0]

        # (B, C, 8, 8) -> (B, 64, C)  with row-major (rank-major) order
        # consistent with the incidence matrix (s = r * 8 + c).
        piece_tokens = h.permute(0, 2, 3, 1).reshape(b, NUM_PIECES, self.channels).contiguous()
        line_tokens = self._initial_line_tokens(b, h.device, h.dtype)

        line_message_stack: list[torch.Tensor] = []
        piece_message_stack: list[torch.Tensor] = []
        line_token_stack: list[torch.Tensor] = []
        piece_token_stack: list[torch.Tensor] = []
        for layer in self.layers:
            packet = layer(
                piece_tokens=piece_tokens,
                line_tokens=line_tokens,
                line_to_piece_weight=self.line_to_piece_weight,
                piece_to_line_weight=self.piece_to_line_weight,
            )
            piece_tokens = packet["pieces"]
            line_tokens = packet["lines"]
            line_message_stack.append(packet["line_message"])
            piece_message_stack.append(packet["piece_message"])
            line_token_stack.append(line_tokens)
            piece_token_stack.append(piece_tokens)

        line_messages = torch.stack(line_message_stack, dim=1)  # (B, D, 46, C)
        piece_messages = torch.stack(piece_message_stack, dim=1)  # (B, D, 64, C)
        line_token_history = torch.stack(line_token_stack, dim=1)  # (B, D, 46, C)
        piece_token_history = torch.stack(piece_token_stack, dim=1)  # (B, D, 64, C)

        pooled = piece_tokens.mean(dim=1)  # (B, C)
        raw_logits = self.head(pooled)
        logits = _format_logits(raw_logits, self.num_classes)

        # Slice line tokens by line type for diagnostics.
        rank_slice = slice(0, NUM_RANKS)
        file_slice = slice(NUM_RANKS, NUM_RANKS + NUM_FILES)
        diag_slice = slice(NUM_RANKS + NUM_FILES, NUM_RANKS + NUM_FILES + NUM_DIAGS)
        antidiag_slice = slice(NUM_RANKS + NUM_FILES + NUM_DIAGS, NUM_LINES)

        rank_tokens = line_tokens[:, rank_slice, :]
        file_tokens = line_tokens[:, file_slice, :]
        diag_tokens = line_tokens[:, diag_slice, :]
        antidiag_tokens = line_tokens[:, antidiag_slice, :]

        line_energy = line_tokens.pow(2).mean(dim=-1)  # (B, 46)
        piece_energy = piece_tokens.pow(2).mean(dim=-1)  # (B, 64)
        rank_line_energy = line_energy[:, rank_slice]
        file_line_energy = line_energy[:, file_slice]
        diag_line_energy = line_energy[:, diag_slice]
        antidiag_line_energy = line_energy[:, antidiag_slice]

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "trunk_features": h,
            "piece_pool": pooled,
            "piece_tokens": piece_tokens,
            "line_tokens": line_tokens,
            "rank_tokens": rank_tokens,
            "file_tokens": file_tokens,
            "diag_tokens": diag_tokens,
            "antidiag_tokens": antidiag_tokens,
            "piece_token_history": piece_token_history,
            "line_token_history": line_token_history,
            "piece_message_stack": piece_messages,
            "line_message_stack": line_messages,
            "piece_energy": piece_energy,
            "line_energy": line_energy,
            "rank_line_energy": rank_line_energy,
            "file_line_energy": file_line_energy,
            "diag_line_energy": diag_line_energy,
            "antidiag_line_energy": antidiag_line_energy,
            "mean_piece_energy": piece_energy.mean(dim=-1),
            "mean_line_energy": line_energy.mean(dim=-1),
            "rank_minus_file_line_energy": rank_line_energy.mean(dim=-1)
            - file_line_energy.mean(dim=-1),
            "diag_minus_antidiag_line_energy": diag_line_energy.mean(dim=-1)
            - antidiag_line_energy.mean(dim=-1),
            "depth_levels": logits.new_full(logits.shape, float(self.depth)),
            "num_lines_levels": logits.new_full(logits.shape, float(NUM_LINES)),
            "num_pieces_levels": logits.new_full(logits.shape, float(NUM_PIECES)),
        }
        if self.num_classes == 1:
            diagnostics["prob"] = torch.sigmoid(logits)
        return diagnostics


def build_line_piece_crossbar_network_from_config(
    config: dict[str, Any],
) -> LinePieceCrossbarNetwork:
    return LinePieceCrossbarNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
    )
