"""Evidence Sieve Network for idea i167.

Faithful implementation of the markdown thesis: rather than refining logits,
the network refines per-square channel features by passing them through a
stack of learned ``evidence sieves``. Each sieve stage produces a soft mask
over channels and squares, multiplies the trunk feature map by that mask to
form the *selected evidence* for the stage, propagates a small residual update
to the next stage, and records the stage's mask plus selected-evidence energy
as a diagnostic trail.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


@dataclass(frozen=True)
class EvidenceSieveNetworkConfig:
    input_channels: int = 18
    num_classes: int = 1
    channels: int = 64
    hidden_dim: int = 96
    depth: int = 2
    dropout: float = 0.1
    use_batchnorm: bool = True
    num_sieves: int = 3
    channel_gate_hidden: int = 32
    spatial_gate_hidden: int = 16
    residual_scale: float = 0.5


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


class _EvidenceSieve(nn.Module):
    """A single evidence sieve stage.

    Given an input feature map ``H \\in R^{C \\times 8 \\times 8}`` the stage
    produces

    - a *channel mask* ``c \\in (0, 1)^C`` from a gated MLP over the global
      channel statistics ``GAP(H)``;
    - a *spatial mask* ``s \\in (0, 1)^{8 \\times 8}`` from a small 1x1 / 3x3
      conv head over ``H``;
    - the *selected evidence* ``E = c[None,:,None,None] * s[:,None,:,:] * H``;
    - and a residual update ``H' = LayerNorm( H + alpha * Conv(E) )`` whose
      input is exactly the selected evidence -- this is what propagates
      *only* the sieved evidence to the next stage.
    """

    def __init__(
        self,
        channels: int,
        channel_gate_hidden: int,
        spatial_gate_hidden: int,
        residual_scale: float,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if channel_gate_hidden < 1:
            raise ValueError("channel_gate_hidden must be >= 1")
        if spatial_gate_hidden < 1:
            raise ValueError("spatial_gate_hidden must be >= 1")
        self.channels = int(channels)
        self.residual_scale = float(residual_scale)

        # Channel-mask MLP over global channel statistics. Concatenates the
        # global average and global max so the gate can see both ``how much``
        # and ``the strongest`` activation per channel.
        self.channel_gate = nn.Sequential(
            nn.Linear(channels * 2, channel_gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(channel_gate_hidden, channels),
        )

        # Spatial-mask conv head: a small 3x3 -> GELU -> 1x1 stack ending in a
        # single-channel logit per square.
        spatial_layers: list[nn.Module] = [
            nn.Conv2d(
                channels,
                spatial_gate_hidden,
                kernel_size=3,
                padding=1,
                bias=not use_batchnorm,
            )
        ]
        if use_batchnorm:
            spatial_layers.append(nn.BatchNorm2d(spatial_gate_hidden))
        spatial_layers.append(nn.GELU())
        spatial_layers.append(nn.Conv2d(spatial_gate_hidden, 1, kernel_size=1))
        self.spatial_gate = nn.Sequential(*spatial_layers)

        # Residual conv that turns the selected evidence into a feature update
        # for the next stage. Kept compact (3x3) so the propagated information
        # is local to the sieved evidence rather than re-mixing the whole
        # trunk.
        self.residual = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=not use_batchnorm,
            ),
            nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity(),
            nn.GELU(),
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        b, c, height, width = h.shape
        # Channel mask from concatenated global pooling statistics.
        gap = h.mean(dim=(2, 3))
        gmp = h.amax(dim=(2, 3))
        channel_logits = self.channel_gate(torch.cat([gap, gmp], dim=-1))
        channel_mask = torch.sigmoid(channel_logits)  # (B, C)

        # Spatial mask from a small conv head.
        spatial_logits = self.spatial_gate(h).squeeze(1)  # (B, 8, 8)
        spatial_mask = torch.sigmoid(spatial_logits)

        # Selected evidence = channel_mask * spatial_mask * H, broadcast.
        c_mask = channel_mask.view(b, c, 1, 1)
        s_mask = spatial_mask.view(b, 1, height, width)
        selected = c_mask * s_mask * h  # (B, C, 8, 8)

        # Residual update: only the *selected* evidence feeds the next stage.
        update = self.residual(selected)
        next_h = self.norm(h + self.residual_scale * update)
        return {
            "next": next_h,
            "selected": selected,
            "channel_mask": channel_mask,
            "spatial_mask": spatial_mask,
            "channel_logits": channel_logits,
            "spatial_logits": spatial_logits,
        }


class EvidenceSieveNetwork(nn.Module):
    """Evidence Sieve Network classifier.

    Pipeline per the thesis:

    1. ``H_0 = Trunk(x)`` -- a stem ``Conv3x3 -> BatchNorm -> GELU`` followed
       by ``depth`` ``ConvBlock``s of width ``channels``.
    2. ``T = num_sieves`` evidence sieve stages produce, for ``t = 1..T``:

           ``c_t = sigmoid( MLP_c( [GAP(H_{t-1}); GMP(H_{t-1})] ) )``
           ``s_t = sigmoid( Head_s( H_{t-1} ) )``
           ``E_t = c_t \\cdot s_t \\cdot H_{t-1}``
           ``H_t = GroupNorm( H_{t-1} + alpha * Conv(E_t) )``

       so each stage refines the *features* (not the logits) by passing only
       the selected evidence through a small residual conv before the next
       sieve runs.
    3. The classifier head consumes the *aggregate of selected evidence
       across stages* together with the final propagated trunk
       representation: ``z = concat( pool( mean_t E_t ), pool(H_T) )`` then
       ``LayerNorm -> Linear -> GELU -> Dropout -> Linear`` to one puzzle
       logit.

    The forward pass also returns the per-stage channel and spatial masks
    plus selected-evidence energies so downstream tooling can inspect the
    sieve trail without re-running the model.
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
        num_sieves: int = 3,
        channel_gate_hidden: int = 32,
        spatial_gate_hidden: int = 16,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if num_sieves < 1:
            raise ValueError("num_sieves must be >= 1 to apply at least one sieve stage")
        if not 0.0 <= residual_scale <= 4.0:
            raise ValueError("residual_scale must be in [0, 4]")

        self.config = EvidenceSieveNetworkConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            num_sieves=num_sieves,
            channel_gate_hidden=channel_gate_hidden,
            spatial_gate_hidden=spatial_gate_hidden,
            residual_scale=residual_scale,
        )
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.depth = int(depth)
        self.num_sieves = int(num_sieves)

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

        self.sieves = nn.ModuleList(
            [
                _EvidenceSieve(
                    channels=channels,
                    channel_gate_hidden=channel_gate_hidden,
                    spatial_gate_hidden=spatial_gate_hidden,
                    residual_scale=residual_scale,
                    dropout=dropout,
                    use_batchnorm=use_batchnorm,
                )
                for _ in range(num_sieves)
            ]
        )

        head_in = channels * 2  # pooled selected-evidence + pooled final trunk
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        h = self.stem_activation(self.stem_norm(self.stem(board)))
        h = self.trunk(h)  # (B, C, 8, 8)
        b, c, height, width = h.shape

        per_stage_selected: list[torch.Tensor] = []
        channel_masks: list[torch.Tensor] = []
        spatial_masks: list[torch.Tensor] = []
        for sieve in self.sieves:
            stage = sieve(h)
            per_stage_selected.append(stage["selected"])
            channel_masks.append(stage["channel_mask"])
            spatial_masks.append(stage["spatial_mask"])
            h = stage["next"]

        # Aggregate selected evidence across stages and pool together with the
        # final propagated trunk representation.
        selected_stack = torch.stack(per_stage_selected, dim=1)  # (B, T, C, 8, 8)
        mean_selected = selected_stack.mean(dim=1)  # (B, C, 8, 8)
        selected_pool = mean_selected.mean(dim=(2, 3))  # (B, C)
        trunk_pool = h.mean(dim=(2, 3))  # (B, C)
        head_in = torch.cat([selected_pool, trunk_pool], dim=-1)
        raw_logits = self.classifier(head_in)
        logits = _format_logits(raw_logits, self.num_classes)

        channel_mask_stack = torch.stack(channel_masks, dim=1)  # (B, T, C)
        spatial_mask_stack = torch.stack(spatial_masks, dim=1)  # (B, T, 8, 8)
        # Per-stage diagnostics:
        # - selection ratio = product of mean masks (approximate "fraction
        #   of evidence kept" per stage),
        # - selected energy = mean square of E_t,
        # - mask entropy on a Bernoulli per element.
        channel_mean = channel_mask_stack.mean(dim=-1)  # (B, T)
        spatial_mean = spatial_mask_stack.mean(dim=(-1, -2))  # (B, T)
        selection_ratio = channel_mean * spatial_mean  # (B, T)
        selected_energy = selected_stack.pow(2).mean(dim=(2, 3, 4))  # (B, T)
        eps = 1e-7
        bernoulli_entropy = (
            -channel_mask_stack * (channel_mask_stack + eps).log()
            - (1.0 - channel_mask_stack) * (1.0 - channel_mask_stack + eps).log()
        )
        channel_mask_entropy = bernoulli_entropy.mean(dim=-1)  # (B, T)
        spatial_bernoulli = (
            -spatial_mask_stack * (spatial_mask_stack + eps).log()
            - (1.0 - spatial_mask_stack) * (1.0 - spatial_mask_stack + eps).log()
        )
        spatial_mask_entropy = spatial_bernoulli.mean(dim=(-1, -2))  # (B, T)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "trunk_features": h,
            "stage_selected_evidence": selected_stack,
            "stage_channel_masks": channel_mask_stack,
            "stage_spatial_masks": spatial_mask_stack,
            "stage_selection_ratio": selection_ratio,
            "stage_selected_energy": selected_energy,
            "stage_channel_mask_entropy": channel_mask_entropy,
            "stage_spatial_mask_entropy": spatial_mask_entropy,
            "selected_evidence_mean": mean_selected,
            "selected_pool": selected_pool,
            "trunk_pool": trunk_pool,
        }

        # Scalar (per-batch) diagnostics that reduce the per-stage trail to a
        # single number for logging dashboards.
        diagnostics["mean_selection_ratio"] = selection_ratio.mean(dim=-1)
        diagnostics["mean_selected_energy"] = selected_energy.mean(dim=-1)
        diagnostics["mean_channel_mask_entropy"] = channel_mask_entropy.mean(dim=-1)
        diagnostics["mean_spatial_mask_entropy"] = spatial_mask_entropy.mean(dim=-1)
        # Final-vs-initial residual carryover energy: how much the trunk
        # representation actually changed across the sieve stack.
        diagnostics["sieve_carryover_energy"] = (h - selected_stack[:, 0]).pow(2).mean(dim=(1, 2, 3))
        diagnostics["depth_levels"] = logits.new_full(logits.shape, float(self.depth))
        diagnostics["sieve_levels"] = logits.new_full(logits.shape, float(self.num_sieves))

        if self.num_classes == 1:
            diagnostics["prob"] = torch.sigmoid(logits)
        return diagnostics


def build_evidence_sieve_network_from_config(
    config: dict[str, Any],
) -> EvidenceSieveNetwork:
    return EvidenceSieveNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        num_sieves=int(config.get("num_sieves", 3)),
        channel_gate_hidden=int(config.get("channel_gate_hidden", 32)),
        spatial_gate_hidden=int(config.get("spatial_gate_hidden", 16)),
        residual_scale=float(config.get("residual_scale", 0.5)),
    )
