"""Stripe-Selective Mixer CNN model for idea i173.

Faithful implementation of the markdown thesis under
``ideas/i173_stripe_selective_mixer_cnn/``: a compact line-aware CNN
where every block mixes along the four chess stripe directions
(ranks, files, diagonals, anti-diagonals) in addition to a local
``3x3`` convolution, with a per-channel sigmoid global-context gate
that selects which stripe directions matter for a given board.

The packet's central layer formula is implemented verbatim:

    x_local = Conv3x3(x)
    x_rank  = rank_scan(x)
    x_file  = file_scan(x)
    x_diag  = diagonal_scan(x)
    x_anti  = anti_diagonal_scan(x)
    gate    = sigmoid(MLP(global_pool(x)))
    x_next  = x + Conv1x1([x_local,
                           gate * x_rank,
                           gate * x_file,
                           gate * x_diag,
                           gate * x_anti])

Each stripe scan is a ``Conv2d`` whose kernel is constrained by a
fixed binary mask along the corresponding chess line: rank uses a
``(1, K)`` row, file uses a ``(K, 1)`` column, and diagonal /
anti-diagonal use ``(K, K)`` masks that are non-zero only along the
``(i, i)`` / ``(i, K - 1 - i)`` positions. There is no recurrent
machinery — the "scan" is exactly a 1-D sequence convolution along
the stripe direction.

Section ablations from the packet are exposed via ``ablation``:

    * ``"none"`` -- main model.
    * ``"local_only"`` -- ordinary CNN control: drop every stripe
      branch and keep only the local ``Conv3x3`` plus the residual.
    * ``"rank_file_only"`` -- keep ranks and files (rook lines), drop
      diagonals and anti-diagonals.
    * ``"diag_only"`` -- keep diagonals only, drop ranks, files, and
      anti-diagonals.
    * ``"random_stripes"`` -- replace every stripe mask with a fixed
      random ``K``-position mask so the line geometry is destroyed
      while parameter count stays matched.
    * ``"no_global_gate"`` -- drop the sigmoid global-context gate so
      stripe branches are summed without selection.

Engine, source, verification, and CRTK metadata are never used as
input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models._packet_bespoke_base import (
    BoardTensorSpec,
    require_board_tensor,
)


_DEFAULT_INPUT_CHANNELS = 18
_DEFAULT_CHANNELS = 64
_DEFAULT_HIDDEN_DIM = 96
_DEFAULT_DEPTH = 2
_DEFAULT_STRIPE_KERNEL = 5
_DEFAULT_DROPOUT = 0.1
_RANDOM_STRIPE_BASE_SEED = 1729

STRIPE_DIRECTIONS: tuple[str, ...] = ("rank", "file", "diag", "antidiag")

_VALID_ABLATIONS = {
    "none",
    "local_only",
    "rank_file_only",
    "diag_only",
    "random_stripes",
    "no_global_gate",
}


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _stripe_mask(direction: str, kernel_size: int) -> torch.Tensor:
    K = int(kernel_size)
    mask = torch.zeros(K, K, dtype=torch.float32)
    if direction == "rank":
        mask[K // 2, :] = 1.0
    elif direction == "file":
        mask[:, K // 2] = 1.0
    elif direction == "diag":
        for i in range(K):
            mask[i, i] = 1.0
    elif direction == "antidiag":
        for i in range(K):
            mask[i, K - 1 - i] = 1.0
    else:
        raise ValueError(f"Unknown stripe direction: {direction}")
    return mask


def _random_stripe_mask(generator: torch.Generator, kernel_size: int) -> torch.Tensor:
    K = int(kernel_size)
    flat = torch.zeros(K * K, dtype=torch.float32)
    perm = torch.randperm(K * K, generator=generator)
    flat[perm[:K]] = 1.0
    return flat.view(K, K)


class DirectionalStripeConv(nn.Module):
    """Conv2d whose kernel is masked along a fixed chess-stripe direction.

    The mask is a non-trainable buffer of shape ``(K, K)`` with exactly
    ``K`` ones along the chosen line; weights at the masked-out
    positions stay at zero in every forward pass. This is a 1-D
    sequence convolution along the stripe expressed as a ``Conv2d``
    so the four stripe directions share one implementation.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        mask: torch.Tensor,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if mask.shape != (kernel_size, kernel_size):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match kernel_size {kernel_size}"
            )
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.padding = int(kernel_size // 2)
        self.weight = nn.Parameter(
            torch.empty(channels, channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if use_batchnorm:
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("mask", mask.detach().clone(), persistent=False)
        self.norm = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.activation = nn.GELU()

    def masked_weight(self) -> torch.Tensor:
        return self.weight * self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.conv2d(x, self.masked_weight(), self.bias, padding=self.padding)
        out = self.norm(out)
        return self.activation(out)


def _resolve_active_directions(ablation: str) -> tuple[str, ...]:
    if ablation == "local_only":
        return ()
    if ablation == "rank_file_only":
        return ("rank", "file")
    if ablation == "diag_only":
        return ("diag",)
    return STRIPE_DIRECTIONS


class StripeSelectiveMixerBlock(nn.Module):
    """One stripe-selective mixer block.

    Implements the packet's layer formula exactly:

        x_local = Conv3x3(x)
        x_dir   = DirectionalStripeConv(x)  for each active direction
        gate    = sigmoid(MLP(global_pool(x)))
        merged  = concat([x_local, gate*x_dir...])
        x_next  = GELU( x + Dropout(BN(Conv1x1(merged))) )

    The local ``Conv3x3`` is never gated; only stripe branches are
    multiplied by the per-channel sigmoid gate (``no_global_gate``
    drops the gate entirely).
    """

    def __init__(
        self,
        channels: int,
        stripe_kernel: int = _DEFAULT_STRIPE_KERNEL,
        ablation: str = "none",
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        random_stripe_seed: int = _RANDOM_STRIPE_BASE_SEED,
    ) -> None:
        super().__init__()
        if ablation not in _VALID_ABLATIONS:
            raise ValueError(f"Unknown ablation: {ablation}")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if stripe_kernel < 1 or stripe_kernel % 2 == 0:
            raise ValueError("stripe_kernel must be a positive odd integer")
        self.channels = int(channels)
        self.stripe_kernel = int(stripe_kernel)
        self.ablation = ablation
        self.use_global_gate = ablation != "no_global_gate"

        self.local_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm
        )
        self.local_norm = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.local_activation = nn.GELU()

        self.active_directions = _resolve_active_directions(ablation)
        if ablation == "random_stripes":
            generator = torch.Generator(device="cpu").manual_seed(int(random_stripe_seed))
            masks = {
                direction: _random_stripe_mask(generator, stripe_kernel)
                for direction in STRIPE_DIRECTIONS
            }
        else:
            masks = {direction: _stripe_mask(direction, stripe_kernel) for direction in STRIPE_DIRECTIONS}

        self.stripe_convs = nn.ModuleDict(
            {
                direction: DirectionalStripeConv(
                    channels=channels,
                    kernel_size=stripe_kernel,
                    mask=masks[direction],
                    use_batchnorm=use_batchnorm,
                )
                for direction in self.active_directions
            }
        )

        gate_hidden = max(8, channels // 2)
        if self.use_global_gate and self.active_directions:
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * channels, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, channels),
            )
        else:
            self.gate_mlp = None

        merged_in = channels * (1 + len(self.active_directions))
        self.fuse = nn.Conv2d(merged_in, channels, kernel_size=1, bias=not use_batchnorm)
        self.fuse_norm = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x_local = self.local_activation(self.local_norm(self.local_conv(x)))

        if self.gate_mlp is not None:
            mean_pool = x.mean(dim=(2, 3))
            max_pool = x.amax(dim=(2, 3))
            pooled = torch.cat([mean_pool, max_pool], dim=1)
            gate = torch.sigmoid(self.gate_mlp(pooled))
        else:
            gate = x.new_ones(x.shape[0], self.channels)

        stripe_features: dict[str, torch.Tensor] = {}
        gated_features: dict[str, torch.Tensor] = {}
        branches = [x_local]
        gate_spatial = gate.unsqueeze(-1).unsqueeze(-1)
        for direction in self.active_directions:
            branch = self.stripe_convs[direction](x)
            stripe_features[direction] = branch
            gated = branch * gate_spatial
            gated_features[direction] = gated
            branches.append(gated)

        merged = torch.cat(branches, dim=1)
        fused = self.fuse(merged)
        fused = self.fuse_norm(fused)
        fused = self.dropout(fused)
        out = self.activation(x + fused)
        return {
            "out": out,
            "x_local": x_local,
            "stripe_features": stripe_features,
            "gated_features": gated_features,
            "gate": gate,
        }


@dataclass(frozen=True)
class StripeSelectiveMixerCNNConfig:
    input_channels: int = _DEFAULT_INPUT_CHANNELS
    num_classes: int = 1
    channels: int = _DEFAULT_CHANNELS
    hidden_dim: int = _DEFAULT_HIDDEN_DIM
    depth: int = _DEFAULT_DEPTH
    stripe_kernel: int = _DEFAULT_STRIPE_KERNEL
    dropout: float = _DEFAULT_DROPOUT
    use_batchnorm: bool = True
    ablation: str = "none"


class StripeSelectiveMixerCNN(nn.Module):
    """Compact stripe-selective CNN classifier for ``puzzle_binary``.

    1. ``Conv3x3 -> BN -> GELU`` stem turns the 18-plane board into a
       per-square feature map of width ``channels``.
    2. ``depth`` stripe-selective mixer blocks fuse a local ``3x3``
       convolution with four directional stripe scans (rank, file,
       diagonal, anti-diagonal). A per-channel sigmoid gate driven by
       the global pool of the block input multiplies every stripe
       branch before the ``1x1`` fuse, then a residual + GELU follows.
    3. The trunk is mean+max pooled and a ``LayerNorm -> Linear ->
       GELU -> Linear`` head emits ``num_classes`` logits (the puzzle
       logit when ``num_classes == 1``).
    """

    VALID_ABLATIONS = _VALID_ABLATIONS

    def __init__(
        self,
        input_channels: int = _DEFAULT_INPUT_CHANNELS,
        num_classes: int = 1,
        channels: int = _DEFAULT_CHANNELS,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        depth: int = _DEFAULT_DEPTH,
        stripe_kernel: int = _DEFAULT_STRIPE_KERNEL,
        dropout: float = _DEFAULT_DROPOUT,
        use_batchnorm: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if stripe_kernel < 1 or stripe_kernel % 2 == 0:
            raise ValueError("stripe_kernel must be a positive odd integer")
        if ablation not in _VALID_ABLATIONS:
            raise ValueError(f"Unknown ablation: {ablation}")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.stripe_kernel = int(stripe_kernel)
        self.ablation = ablation
        self.config = StripeSelectiveMixerCNNConfig(
            input_channels=int(input_channels),
            num_classes=int(num_classes),
            channels=int(channels),
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            stripe_kernel=int(stripe_kernel),
            dropout=float(dropout),
            use_batchnorm=bool(use_batchnorm),
            ablation=ablation,
        )

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity(),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList(
            [
                StripeSelectiveMixerBlock(
                    channels=channels,
                    stripe_kernel=stripe_kernel,
                    ablation=ablation,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                    random_stripe_seed=_RANDOM_STRIPE_BASE_SEED + 7919 * i,
                )
                for i in range(depth)
            ]
        )

        pooled_dim = 2 * channels
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        h = self.stem(board)

        gate_history: list[torch.Tensor] = []
        local_energy_stack: list[torch.Tensor] = []
        rank_energy_stack: list[torch.Tensor] = []
        file_energy_stack: list[torch.Tensor] = []
        diag_energy_stack: list[torch.Tensor] = []
        antidiag_energy_stack: list[torch.Tensor] = []
        block_history: list[torch.Tensor] = []
        for block in self.blocks:
            packet = block(h)
            h = packet["out"]
            block_history.append(h)
            gate_history.append(packet["gate"])
            local_energy_stack.append(packet["x_local"].square().mean(dim=(1, 2, 3)))
            stripe_features = packet["stripe_features"]
            for direction, stack in zip(
                STRIPE_DIRECTIONS,
                (rank_energy_stack, file_energy_stack, diag_energy_stack, antidiag_energy_stack),
            ):
                feat = stripe_features.get(direction)
                if feat is not None:
                    stack.append(feat.square().mean(dim=(1, 2, 3)))
                else:
                    stack.append(board.new_zeros(batch))

        trunk_energy = h.square().mean(dim=(1, 2, 3))
        mean_pool = h.mean(dim=(2, 3))
        max_pool = h.amax(dim=(2, 3))
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        raw_logits = self.head(pooled)
        logits = _format_logits(raw_logits, self.num_classes)

        gate_stack = torch.stack(gate_history, dim=1)  # (B, depth, C)
        gate_per_block_mean = gate_stack.mean(dim=-1)
        gate_per_block_min = gate_stack.amin(dim=-1)
        gate_per_block_max = gate_stack.amax(dim=-1)

        rank_energy = torch.stack(rank_energy_stack, dim=1).mean(dim=-1)
        file_energy = torch.stack(file_energy_stack, dim=1).mean(dim=-1)
        diag_energy = torch.stack(diag_energy_stack, dim=1).mean(dim=-1)
        antidiag_energy = torch.stack(antidiag_energy_stack, dim=1).mean(dim=-1)
        local_energy = torch.stack(local_energy_stack, dim=1).mean(dim=-1)

        scalar_shape = logits.shape if self.num_classes == 1 else logits.shape[:1]
        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "trunk_features": h,
            "trunk_energy": trunk_energy,
            "pooled": pooled,
            "gate_history": gate_stack,
            "gate_per_block_mean": gate_per_block_mean,
            "gate_per_block_min": gate_per_block_min,
            "gate_per_block_max": gate_per_block_max,
            "gate_overall_mean": gate_per_block_mean.mean(dim=-1),
            "local_branch_energy": local_energy,
            "rank_branch_energy": rank_energy,
            "file_branch_energy": file_energy,
            "diag_branch_energy": diag_energy,
            "antidiag_branch_energy": antidiag_energy,
            "rank_minus_file_branch_energy": rank_energy - file_energy,
            "diag_minus_antidiag_branch_energy": diag_energy - antidiag_energy,
            "active_stripe_count": logits.new_full(
                scalar_shape, float(len(self.blocks[0].active_directions))
            ),
            "stripe_kernel_levels": logits.new_full(scalar_shape, float(self.stripe_kernel)),
            "depth_levels": logits.new_full(scalar_shape, float(self.depth)),
            "ablation_active": logits.new_full(
                scalar_shape, 0.0 if self.ablation == "none" else 1.0
            ),
        }
        if self.num_classes == 1:
            diagnostics["prob"] = torch.sigmoid(logits)
        return diagnostics


def build_stripe_selective_mixer_cnn_from_config(
    config: dict[str, Any],
) -> StripeSelectiveMixerCNN:
    cfg = dict(config)
    return StripeSelectiveMixerCNN(
        input_channels=int(cfg.get("input_channels", _DEFAULT_INPUT_CHANNELS)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", _DEFAULT_CHANNELS)),
        hidden_dim=int(cfg.get("hidden_dim", _DEFAULT_HIDDEN_DIM)),
        depth=int(cfg.get("depth", _DEFAULT_DEPTH)),
        stripe_kernel=int(cfg.get("stripe_kernel", _DEFAULT_STRIPE_KERNEL)),
        dropout=float(cfg.get("dropout", _DEFAULT_DROPOUT)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        ablation=str(cfg.get("ablation", "none")),
    )
