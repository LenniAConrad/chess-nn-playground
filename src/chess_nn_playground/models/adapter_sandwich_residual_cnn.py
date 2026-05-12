"""Adapter-Sandwich Residual CNN for idea i154.

A standard residual-CNN trunk is sandwiched between two parameter-
efficient bottleneck adapters per stage: a *pre*-adapter applied
before the residual block and a *post*-adapter applied after it. Each
adapter is a 1x1 Conv -> GELU -> 1x1 Conv bottleneck of width
``adapter_dim`` (much smaller than ``channels``), wrapped in its own
identity residual. The bulk of capacity stays in the conventional
residual blocks; the adapters add a small amount of structured slack
that lets the network re-route channel mixing locally.

The thesis (per ``ideas/registry/i154_adapter_sandwich_residual_cnn``) is that
adapters, which are well known from transfer-learning literature, can
also be used as a parameter-efficient capacity knob inside an
otherwise conventional CNN family on the puzzle_binary contract.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


class _BottleneckAdapter(nn.Module):
    """Houlsby-style 1x1-conv bottleneck adapter with an identity residual.

    Output is ``x + W_up(GELU(W_down(x)))``. ``W_up`` is zero-initialised
    so that, at the start of training, every adapter is the identity
    function; this keeps the surrounding residual blocks behaving like a
    standard CNN until the adapters earn their non-zero contribution.
    """

    def __init__(self, channels: int, adapter_dim: int) -> None:
        super().__init__()
        if adapter_dim < 1:
            raise ValueError("adapter_dim must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        self.channels = int(channels)
        self.adapter_dim = int(adapter_dim)
        self.down = nn.Conv2d(self.channels, self.adapter_dim, kernel_size=1, bias=True)
        self.up = nn.Conv2d(self.adapter_dim, self.channels, kernel_size=1, bias=True)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.delta(x)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm1 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm2 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.norm1(self.conv1(x)), inplace=True)
        z = self.dropout(z)
        z = self.norm2(self.conv2(z))
        return F.relu(x + z, inplace=True)


class _AdapterSandwichStage(nn.Module):
    """One stage = pre-adapter -> residual block -> post-adapter."""

    def __init__(self, channels: int, adapter_dim: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        self.pre_adapter = _BottleneckAdapter(channels=channels, adapter_dim=adapter_dim)
        self.residual = _ResidualBlock(channels=channels, dropout=dropout, use_batchnorm=use_batchnorm)
        self.post_adapter = _BottleneckAdapter(channels=channels, adapter_dim=adapter_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_delta = self.pre_adapter.delta(x)
        x = x + pre_delta
        x = self.residual(x)
        post_delta = self.post_adapter.delta(x)
        x = x + post_delta
        return x, pre_delta, post_delta


def _flat_l2_per_sample(t: torch.Tensor) -> torch.Tensor:
    return t.flatten(1).pow(2).sum(dim=1).sqrt()


class AdapterSandwichResidualCNN(nn.Module):
    """Residual CNN with bottleneck adapters before and after each block.

    Stem: ``Conv2d(input_channels -> channels, 3x3) + BN + ReLU``.
    For ``i = 1..depth``:
      ``pre_adapter_i -> residual_block_i -> post_adapter_i``.
    Head: ``AdaptiveAvgPool2d(1) -> Flatten ->
    Linear(channels -> hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim -> 1)``.

    Each adapter is initialised so that its ``W_up`` is zero, making the
    whole sandwich the identity at step 0; the network therefore starts
    behaviourally equivalent to a plain residual CNN and gradually picks
    up the adapter slack during training.
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
        adapter_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "AdapterSandwichResidualCNN supports the puzzle_binary one-logit contract"
            )
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1 or hidden_dim < 1:
            raise ValueError("channels and hidden_dim must be positive")

        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout_p = float(dropout)
        # Default adapter width: bottleneck to roughly a quarter of trunk
        # channels, with a small floor so very narrow trunks still have a
        # non-degenerate bottleneck.
        if adapter_dim is None:
            adapter_dim_int = max(4, int(channels) // 4)
        else:
            adapter_dim_int = int(adapter_dim)
        if adapter_dim_int < 1:
            raise ValueError("adapter_dim must be >= 1")
        self.adapter_dim = adapter_dim_int

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(self.channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList(
            [
                _AdapterSandwichStage(
                    channels=self.channels,
                    adapter_dim=self.adapter_dim,
                    dropout=self.dropout_p,
                    use_batchnorm=use_batchnorm,
                )
                for _ in range(self.depth)
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        head_layers: list[nn.Module] = [
            nn.Linear(self.channels, self.hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if self.dropout_p > 0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.hidden_dim, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, board: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(board, self.spec)
        h = self.stem(x)

        pre_energies: list[torch.Tensor] = []
        post_energies: list[torch.Tensor] = []
        for stage in self.stages:
            h, pre_delta, post_delta = stage(h)
            # Detach the diagnostic energy norms so the loss does not
            # implicitly minimise / maximise adapter contribution.
            pre_energies.append(_flat_l2_per_sample(pre_delta.detach()))
            post_energies.append(_flat_l2_per_sample(post_delta.detach()))

        latent = h
        pooled = self.pool(latent).flatten(1)
        logits = self.head(pooled).view(-1)

        pre_stack = torch.stack(pre_energies, dim=1)  # (B, depth)
        post_stack = torch.stack(post_energies, dim=1)
        pre_adapter_energy = pre_stack.sum(dim=1)
        post_adapter_energy = post_stack.sum(dim=1)
        adapter_energy = pre_adapter_energy + post_adapter_energy

        return {
            "logits": logits,
            "logit": logits,
            "prob": torch.sigmoid(logits),
            "latent": latent,
            "pre_adapter_energy": pre_adapter_energy,
            "post_adapter_energy": post_adapter_energy,
            "adapter_energy": adapter_energy,
            "per_stage_pre_adapter_energy": pre_stack,
            "per_stage_post_adapter_energy": post_stack,
        }


def build_adapter_sandwich_residual_cnn_from_config(config: dict[str, Any]) -> AdapterSandwichResidualCNN:
    cfg = dict(config)
    cfg.setdefault("input_channels", 18)
    cfg.setdefault("num_classes", 1)
    adapter_dim = cfg.get("adapter_dim")
    return AdapterSandwichResidualCNN(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        adapter_dim=int(adapter_dim) if adapter_dim is not None else None,
    )
