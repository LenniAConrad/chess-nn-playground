"""Auxiliary Reconstruction BoardNet for idea i151.

A convolutional encoder shared by two heads:

- a puzzle classifier head pooled over the latent feature map, and
- a lightweight decoder that reconstructs selected current-board input
  planes from the same latent feature map.

The decoder exists to regularise the encoder so the trunk does not
discard board detail. Reconstruction is an *auxiliary* training loss; the
default trainer's BCE-with-logits on ``logits`` already trains the
encoder + classifier path. Use :func:`auxiliary_reconstruction_loss` to
combine the puzzle loss with the reconstruction term during ablations.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0, use_batchnorm: bool = True) -> None:
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


def _resolve_reconstruction_targets(
    targets: Sequence[int] | None, input_channels: int
) -> tuple[int, ...]:
    if targets is None:
        return tuple(range(int(input_channels)))
    resolved: list[int] = []
    seen: set[int] = set()
    for value in targets:
        idx = int(value)
        if idx < 0 or idx >= int(input_channels):
            raise ValueError(
                f"reconstruction_targets index {idx} is out of range for input_channels={input_channels}"
            )
        if idx in seen:
            continue
        seen.add(idx)
        resolved.append(idx)
    if not resolved:
        raise ValueError("reconstruction_targets must select at least one input plane")
    return tuple(resolved)


class AuxiliaryReconstructionBoardNet(nn.Module):
    """CNN encoder + classifier head + auxiliary current-board decoder.

    The classifier still consumes only the board tensor, never future or
    engine information. The decoder reconstructs a configurable subset of
    the input planes from the encoder latent so the trunk is regularised
    against discarding board detail too early.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoder_width: int = 64,
        encoder_depth: int = 4,
        decoder_width: int = 32,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        lambda_recon: float = 0.05,
        reconstruction_targets: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "AuxiliaryReconstructionBoardNet supports the puzzle_binary one-logit contract"
            )
        if encoder_depth < 1:
            raise ValueError("encoder_depth must be >= 1")
        if encoder_width < 1 or decoder_width < 1 or hidden_dim < 1:
            raise ValueError("encoder_width, decoder_width, and hidden_dim must be positive")

        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.encoder_width = int(encoder_width)
        self.encoder_depth = int(encoder_depth)
        self.decoder_width = int(decoder_width)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.lambda_recon = float(lambda_recon)
        target_indices = _resolve_reconstruction_targets(reconstruction_targets, input_channels)
        self.register_buffer(
            "reconstruction_target_indices",
            torch.tensor(target_indices, dtype=torch.long),
            persistent=False,
        )

        self.stem = nn.Sequential(
            nn.Conv2d(int(input_channels), int(encoder_width), kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(int(encoder_width)) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.encoder_blocks = nn.ModuleList(
            [
                _ResidualBlock(int(encoder_width), dropout=dropout, use_batchnorm=use_batchnorm)
                for _ in range(int(encoder_depth))
            ]
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        classifier_layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(int(encoder_width), int(hidden_dim)),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            classifier_layers.append(nn.Dropout(float(dropout)))
        classifier_layers.append(nn.Linear(int(hidden_dim), 1))
        self.classifier_head = nn.Sequential(*classifier_layers)

        decoder_layers: list[nn.Module] = [
            nn.Conv2d(int(encoder_width), int(decoder_width), kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            decoder_layers.append(nn.BatchNorm2d(int(decoder_width)))
        decoder_layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            decoder_layers.append(nn.Dropout2d(float(dropout)))
        decoder_layers.append(nn.Conv2d(int(decoder_width), len(target_indices), kernel_size=1))
        self.decoder = nn.Sequential(*decoder_layers)

    @property
    def reconstruction_targets(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.reconstruction_target_indices.tolist())

    def encode(self, board: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(board, self.spec)
        h = self.stem(x)
        for block in self.encoder_blocks:
            h = block(h)
        return h

    def forward(self, board: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(board, self.spec)
        latent = self.encode(x)

        pooled = self.pool(latent)
        logit = self.classifier_head(pooled).view(-1)

        recon_logits = self.decoder(latent)
        recon_probs = torch.sigmoid(recon_logits)

        target_planes = x.index_select(1, self.reconstruction_target_indices)
        recon_error = (recon_probs - target_planes).pow(2).mean(dim=(1, 2, 3))

        per_plane_recon_bce = F.binary_cross_entropy_with_logits(
            recon_logits, target_planes, reduction="none"
        ).mean(dim=(2, 3))

        # Diagnostic energies kept for compatibility with the project schema.
        mechanism_energy = latent.pow(2).mean(dim=(1, 2, 3))
        proposal_profile_strength = recon_probs.mean(dim=(1, 2, 3))
        proposal_keyword_count = logit.new_full(
            (logit.shape[0],), float(len(self.reconstruction_targets))
        )

        return {
            "logits": logit,
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "latent": latent,
            "reconstruction_logits": recon_logits,
            "reconstruction_probs": recon_probs,
            "reconstruction_target_planes": target_planes,
            "reconstruction_target_indices": self.reconstruction_target_indices,
            "reconstruction_error": recon_error,
            "reconstruction_bce_per_plane": per_plane_recon_bce,
            "mechanism_energy": mechanism_energy,
            "proposal_profile_strength": proposal_profile_strength,
            "proposal_keyword_count": proposal_keyword_count,
        }


def auxiliary_reconstruction_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    lambda_recon: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Combined puzzle + reconstruction loss for ablations.

    ``output`` must come from :class:`AuxiliaryReconstructionBoardNet`. The
    primary term is BCE-with-logits on ``output["logits"]``. The auxiliary
    term is BCE-with-logits between ``reconstruction_logits`` and the
    selected input planes, scaled by ``lambda_recon``. The default trainer
    uses only the primary term; this helper is wired by the ablation
    configs that exercise the reconstruction regulariser.
    """
    target = target.float().view(-1)
    primary = F.binary_cross_entropy_with_logits(output["logits"].view(-1), target)
    recon_logits = output["reconstruction_logits"]
    recon_targets = output["reconstruction_target_planes"]
    aux = F.binary_cross_entropy_with_logits(recon_logits, recon_targets)
    return {
        "loss_classification_bce": primary.detach(),
        "loss_reconstruction_bce": aux.detach(),
        "loss": primary + float(lambda_recon) * aux,
    }


def build_auxiliary_reconstruction_boardnet_from_config(
    config: dict[str, Any],
) -> AuxiliaryReconstructionBoardNet:
    cfg = dict(config)
    cfg.setdefault("input_channels", 18)
    cfg.setdefault("num_classes", 1)
    encoder_width = int(cfg.get("encoder_width", cfg.get("channels", 64)))
    encoder_depth = int(cfg.get("encoder_depth", cfg.get("depth", 4)))
    decoder_width = int(cfg.get("decoder_width", 32))
    hidden_dim = int(cfg.get("hidden_dim", 96))
    dropout = float(cfg.get("dropout", 0.1))
    use_batchnorm = bool(cfg.get("use_batchnorm", True))
    lambda_recon = float(cfg.get("lambda_recon", 0.05))
    reconstruction_targets = cfg.get("reconstruction_targets")
    return AuxiliaryReconstructionBoardNet(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        encoder_width=encoder_width,
        encoder_depth=encoder_depth,
        decoder_width=decoder_width,
        hidden_dim=hidden_dim,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
        lambda_recon=lambda_recon,
        reconstruction_targets=reconstruction_targets,
    )
