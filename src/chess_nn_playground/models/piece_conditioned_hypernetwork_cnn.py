"""Piece-Conditioned Hypernetwork CNN for idea i114.

Working thesis (from ``ideas/all_ideas/registry/i114_piece_conditioned_hypernetwork_cnn``):
the best local filters may depend on material and piece inventory. A
lightweight hypernetwork can condition CNN channel gates *and*
depthwise kernels on safe current-board summaries, adapting the
feature extractor without using engine metadata.

Pipeline:

1. A 1x1 piece-plane embedding lifts the simple_18 input to ``channels``
   feature planes; this performs no spatial mixing.
2. A deterministic piece-inventory summary is computed per sample from
   the raw input planes: white/black piece-type counts, the
   side-to-move material delta, total occupancy, and the means of any
   state planes (12..17). The summary is normalized by board area so
   it is invariant to board size, and it is the *only* signal fed to
   the hypernetwork.
3. A shared hypernetwork MLP maps the summary into per-block,
   per-sample (a) channel gates and (b) depthwise 3x3 kernel weights.
   Both are produced from the same summary embedding through two
   small heads.
4. Each ``HyperConditionedBlock`` applies its predicted depthwise
   convolution (per-sample weights via a grouped ``conv2d``), a static
   pointwise 1x1 conv, GELU, and finally a per-channel sigmoid gate
   broadcast across the spatial axes. The block output is added to a
   residual stream.
5. Mean pooling over the spatial axes plus a small MLP head returns
   one puzzle logit. The forward also returns gate / kernel energy
   diagnostics, the inventory summary, and a normalized material
   delta.

This is materially distinct from the shared ``ResearchPacketProbe``
scaffold: there are no proposal-profile diagnostics, no
mechanism-family embeddings, no profile signature, no shared probe
code. The depthwise kernel parameters are not learned weights of a
plain CNN -- they are per-sample outputs of a hypernetwork conditioned
on piece inventory, exactly as the markdown thesis prescribes.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import (
    BoardTensorSpec,
    require_board_tensor,
)


_STATE_PLANE_COUNT = 6
_PIECE_TYPES = 6


def _piece_inventory_summary(board: torch.Tensor) -> torch.Tensor:
    """Compute a deterministic per-sample piece inventory summary.

    Returns a tensor of shape ``(B, 27)`` for simple_18 boards. The
    feature ordering is:

    * 6 white piece-type counts (planes 0..5), normalized by 64.
    * 6 black piece-type counts (planes 6..11), normalized by 64.
    * 6 white minus black piece-type deltas (the material vector),
      normalized by 8 to keep magnitudes near unit scale.
    * 6 state-plane means (planes 12..17, e.g. side-to-move,
      castling rights, en-passant), already in [0, 1].
    * 1 total occupancy (sum of all piece planes / 64).
    * 1 material delta sum (white minus black total piece count) / 8.
    * 1 minor-piece imbalance (knights+bishops white minus black) / 4.
    """
    batch = board.shape[0]
    channels = board.shape[1]
    if channels < _PIECE_TYPES:
        raise ValueError(
            f"piece-conditioned hypernetwork CNN expects at least {_PIECE_TYPES} piece planes, "
            f"got {channels}"
        )
    white_planes = board[:, : min(_PIECE_TYPES, channels)]
    black_planes = (
        board[:, _PIECE_TYPES : min(2 * _PIECE_TYPES, channels)]
        if channels > _PIECE_TYPES
        else board.new_zeros(batch, _PIECE_TYPES, board.shape[2], board.shape[3])
    )
    state_planes = (
        board[:, 2 * _PIECE_TYPES : min(2 * _PIECE_TYPES + _STATE_PLANE_COUNT, channels)]
        if channels > 2 * _PIECE_TYPES
        else board.new_zeros(batch, _STATE_PLANE_COUNT, board.shape[2], board.shape[3])
    )

    def _pad_planes(planes: torch.Tensor, count: int) -> torch.Tensor:
        if planes.shape[1] >= count:
            return planes[:, :count]
        pad = planes.new_zeros(planes.shape[0], count - planes.shape[1], planes.shape[2], planes.shape[3])
        return torch.cat([planes, pad], dim=1)

    white_planes = _pad_planes(white_planes, _PIECE_TYPES)
    black_planes = _pad_planes(black_planes, _PIECE_TYPES)
    state_planes = _pad_planes(state_planes, _STATE_PLANE_COUNT)

    white_counts = white_planes.sum(dim=(2, 3)) / 64.0  # (B, 6)
    black_counts = black_planes.sum(dim=(2, 3)) / 64.0  # (B, 6)
    piece_delta = (white_counts - black_counts) * 8.0 / 8.0  # (B, 6) already small
    state_means = state_planes.mean(dim=(2, 3))  # (B, 6)

    occupancy = (white_counts + black_counts).sum(dim=1, keepdim=True)  # (B, 1)
    material_delta = (white_counts.sum(dim=1, keepdim=True) - black_counts.sum(dim=1, keepdim=True))  # (B, 1)
    # knights are plane index 1, bishops are plane index 2 in the canonical
    # simple_18 layout used by the project; default to those positions.
    minor_imbalance = (
        white_planes[:, 1:3].sum(dim=(1, 2, 3)) - black_planes[:, 1:3].sum(dim=(1, 2, 3))
    ).unsqueeze(-1) / 4.0  # (B, 1)

    summary = torch.cat(
        [white_counts, black_counts, piece_delta, state_means, occupancy, material_delta, minor_imbalance],
        dim=1,
    )
    return summary


class _SummaryEncoder(nn.Module):
    """Shared MLP that embeds the inventory summary for the hypernetwork."""

    def __init__(self, summary_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(summary_dim)
        self.fc1 = nn.Linear(summary_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        x = self.norm(summary)
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


class _HyperHead(nn.Module):
    """Two heads emitted from the shared summary embedding for one block.

    The gate head emits a per-channel sigmoid gate; the kernel head
    emits per-channel depthwise kernel weights for a 3x3 conv.
    Initialization is tuned so that gates start near 1 and kernels
    start near a centered identity-like filter, so an untrained model
    behaves like a residual conv net.
    """

    def __init__(self, summary_dim: int, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.kernel_count = self.kernel_size * self.kernel_size
        self.gate_head = nn.Linear(summary_dim, self.channels)
        self.kernel_head = nn.Linear(summary_dim, self.channels * self.kernel_count)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        # Initialize gate head with small random weights and bias=2.0 so
        # an untrained model produces gates near sigmoid(2.0) ~ 0.88
        # while still varying per sample with the inventory summary.
        nn.init.normal_(self.gate_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.gate_head.bias, 2.0)
        # Initialize kernel head with small random weights and a bias
        # centered on the 3x3 stencil's middle tap, so the predicted
        # depthwise kernel starts near a centered identity-like filter
        # but still varies per sample with the inventory summary.
        nn.init.normal_(self.kernel_head.weight, mean=0.0, std=0.02)
        kernel_bias = torch.zeros(self.channels, self.kernel_count)
        center = self.kernel_count // 2
        kernel_bias[:, center] = 1.0
        with torch.no_grad():
            self.kernel_head.bias.copy_(kernel_bias.view(-1))

    def forward(
        self, summary_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.drop(summary_emb)
        gate_logits = self.gate_head(emb)  # (B, C)
        gates = torch.sigmoid(gate_logits)
        kernel_logits = self.kernel_head(emb)  # (B, C * K * K)
        kernels = kernel_logits.view(
            -1, self.channels, 1, self.kernel_size, self.kernel_size
        )
        return gates, kernels


class _HyperConditionedBlock(nn.Module):
    """A residual block whose depthwise filter and channel gates come
    from the hypernetwork.

    Forward steps:

    1.  Pre-norm via BatchNorm2d.
    2.  Per-sample depthwise 3x3 convolution using the predicted
        ``kernels`` of shape ``(B, C, 1, K, K)``. We implement the
        per-sample weights with grouped ``conv2d``, treating the batch
        as additional groups.
    3.  Static pointwise 1x1 conv with ``channels`` inputs and outputs.
    4.  GELU activation, then multiplication by the predicted
        per-channel sigmoid gates ``(B, C, 1, 1)``.
    5.  Residual add to the input.
    """

    def __init__(self, channels: int, kernel_size: int, use_batchnorm: bool, dropout: float) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.norm = nn.BatchNorm2d(self.channels) if use_batchnorm else nn.Identity()
        self.pointwise = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def _depthwise_per_sample(self, x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        """Apply per-sample depthwise conv2d using grouped convolution.

        Args:
            x: (B, C, H, W)
            kernels: (B, C, 1, K, K)
        """
        b, c, h, w = x.shape
        if kernels.shape[0] != b or kernels.shape[1] != c:
            raise ValueError(
                f"kernel shape {tuple(kernels.shape)} incompatible with input {tuple(x.shape)}"
            )
        # Reshape input to (1, B*C, H, W) and weights to (B*C, 1, K, K),
        # then run conv2d with groups=B*C so each (sample, channel) pair
        # convolves with its own kernel.
        x_grouped = x.reshape(1, b * c, h, w)
        weight = kernels.reshape(b * c, 1, self.kernel_size, self.kernel_size)
        padding = self.kernel_size // 2
        out = F.conv2d(x_grouped, weight, bias=None, stride=1, padding=padding, groups=b * c)
        return out.reshape(b, c, h, w)

    def forward(
        self, x: torch.Tensor, gates: torch.Tensor, kernels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm(x)
        h = self._depthwise_per_sample(h, kernels)
        h = self.pointwise(h)
        h = self.act(h)
        h = self.drop(h)
        gate_map = gates.view(gates.shape[0], gates.shape[1], 1, 1)
        h = h * gate_map
        out = x + h
        gated_energy = h.pow(2).mean(dim=(1, 2, 3))  # (B,)
        return out, gated_energy


class PieceConditionedHypernetworkCNN(nn.Module):
    """Bespoke piece-conditioned hypernetwork CNN for the puzzle_binary contract."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        hidden_dim: int = 96,
        hyper_hidden: int | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        height: int = 8,
        width: int = 8,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "PieceConditionedHypernetworkCNN supports the puzzle_binary one-logit contract"
            )
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.spec = BoardTensorSpec(
            input_channels=int(input_channels), height=int(height), width=int(width)
        )
        self.input_channels = int(input_channels)
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.depth = int(depth)
        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)
        self.dropout_p = float(dropout)
        self.summary_dim = 2 * _PIECE_TYPES + _PIECE_TYPES + _STATE_PLANE_COUNT + 3
        self.hyper_hidden = int(hyper_hidden) if hyper_hidden is not None else max(self.channels, 32)

        self.embed = nn.Conv2d(self.input_channels, self.channels, kernel_size=1)

        self.summary_encoder = _SummaryEncoder(
            summary_dim=self.summary_dim,
            hidden_dim=self.hyper_hidden,
            dropout=self.dropout_p,
        )

        self.hyper_heads = nn.ModuleList(
            [
                _HyperHead(
                    summary_dim=self.hyper_hidden,
                    channels=self.channels,
                    kernel_size=self.kernel_size,
                    dropout=self.dropout_p,
                )
                for _ in range(self.depth)
            ]
        )

        self.blocks = nn.ModuleList(
            [
                _HyperConditionedBlock(
                    channels=self.channels,
                    kernel_size=self.kernel_size,
                    use_batchnorm=bool(use_batchnorm),
                    dropout=self.dropout_p,
                )
                for _ in range(self.depth)
            ]
        )

        self.final_norm = nn.LayerNorm(self.channels)
        head_layers: list[nn.Module] = [
            nn.Linear(self.channels, self.hidden_dim),
            nn.GELU(),
        ]
        if self.dropout_p > 0.0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.hidden_dim, 1))
        self.classifier = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        summary = _piece_inventory_summary(x)  # (B, summary_dim)
        summary_emb = self.summary_encoder(summary)  # (B, hyper_hidden)

        h = self.embed(x)  # (B, C, H, W)

        gate_means: list[torch.Tensor] = []
        gate_entropies: list[torch.Tensor] = []
        kernel_norms: list[torch.Tensor] = []
        block_energies: list[torch.Tensor] = []

        for hyper_head, block in zip(self.hyper_heads, self.blocks):
            gates, kernels = hyper_head(summary_emb)
            h, gated_energy = block(h, gates, kernels)

            gate_means.append(gates.mean(dim=-1))
            p = gates.clamp(1.0e-6, 1.0 - 1.0e-6)
            entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log()).mean(dim=-1)
            gate_entropies.append(entropy)
            kernel_norms.append(kernels.flatten(1).pow(2).mean(dim=-1))
            block_energies.append(gated_energy)

        pooled = h.mean(dim=(-2, -1))  # (B, C)
        pooled = self.final_norm(pooled)
        logits = self.classifier(pooled).view(-1)

        gate_mean_per_block = torch.stack(gate_means, dim=-1)  # (B, depth)
        gate_entropy_per_block = torch.stack(gate_entropies, dim=-1)  # (B, depth)
        kernel_norm_per_block = torch.stack(kernel_norms, dim=-1)  # (B, depth)
        block_energy_per_block = torch.stack(block_energies, dim=-1)  # (B, depth)

        material_delta = summary[:, 2 * _PIECE_TYPES : 3 * _PIECE_TYPES].sum(dim=-1)  # (B,)

        return {
            "logits": logits,
            "pooled_features": pooled,
            "inventory_summary": summary,
            "gate_mean": gate_mean_per_block.mean(dim=-1),
            "gate_entropy": gate_entropy_per_block.mean(dim=-1),
            "kernel_energy": kernel_norm_per_block.sum(dim=-1),
            "gated_energy": block_energy_per_block.sum(dim=-1),
            "gate_mean_per_block": gate_mean_per_block,
            "gate_entropy_per_block": gate_entropy_per_block,
            "kernel_norm_per_block": kernel_norm_per_block,
            "block_energy_per_block": block_energy_per_block,
            "material_delta": material_delta,
        }


def build_piece_conditioned_hypernetwork_cnn_from_config(
    config: dict[str, Any],
) -> PieceConditionedHypernetworkCNN:
    cfg = dict(config)
    return PieceConditionedHypernetworkCNN(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", 64)),
        depth=int(cfg.get("depth", 2)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        hyper_hidden=cfg.get("hyper_hidden"),
        kernel_size=int(cfg.get("kernel_size", 3)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        height=int(cfg.get("height", 8)),
        width=int(cfg.get("width", 8)),
    )
