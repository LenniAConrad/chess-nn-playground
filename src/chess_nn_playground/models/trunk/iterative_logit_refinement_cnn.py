"""Iterative Logit Refinement CNN implementation for idea i152."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import (
    BoardConvStem,
    BoardTensorSpec,
    require_board_tensor,
)


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _binary_confidence_features(logit: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logit)
    margin = (prob - 0.5).abs() * 2.0
    abs_logit = logit.abs()
    eps = 1.0e-6
    entropy = -(
        prob.clamp_min(eps) * prob.clamp_min(eps).log()
        + (1.0 - prob).clamp_min(eps) * (1.0 - prob).clamp_min(eps).log()
    )
    return torch.stack([logit, prob, abs_logit, margin, entropy], dim=-1)


def _multiclass_confidence_features(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    top1 = sorted_probs[..., 0]
    top2 = sorted_probs[..., 1]
    margin = top1 - top2
    eps = 1.0e-6
    entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1)
    max_logit = logits.amax(dim=-1)
    mean_logit = logits.mean(dim=-1)
    return torch.stack([top1, top2, margin, entropy, max_logit, mean_logit], dim=-1)


class CorrectionMLP(nn.Module):
    """Shared correction head producing a bounded logit delta from latent + previous logits."""

    def __init__(
        self,
        latent_dim: int,
        logit_dim: int,
        confidence_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("correction_hidden must be positive")
        in_dim = latent_dim + logit_dim + confidence_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        latent: torch.Tensor,
        prev_logit: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([latent, prev_logit, confidence], dim=-1))


@dataclass(frozen=True)
class IterativeLogitRefinementCNNConfig:
    input_channels: int = 18
    num_classes: int = 1
    channels: int = 64
    hidden_dim: int = 96
    depth: int = 4
    refinement_steps: int = 4
    correction_hidden: int = 64
    correction_clamp: float = 0.25
    dropout: float = 0.1
    use_batchnorm: bool = True
    untie_corrections: bool = False


class IterativeLogitRefinementCNN(nn.Module):
    """CNN trunk + initial logit head + T staged learned correction steps over logits.

    The model produces an initial logit `l_0` from a pooled board feature `z`, and
    then iteratively refines it as `l_t = l_{t-1} + c_t` where each correction
    `c_t = clamp * tanh(MLP([z, l_{t-1}, confidence(l_{t-1})]))`. By default the
    correction MLP is weight-shared across steps; setting `untie_corrections=True`
    instantiates a distinct head per step.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 4,
        refinement_steps: int = 4,
        correction_hidden: int = 64,
        correction_clamp: float = 0.25,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        untie_corrections: bool = False,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if refinement_steps < 1:
            raise ValueError("refinement_steps must be >= 1")
        if correction_clamp <= 0.0:
            raise ValueError("correction_clamp must be positive")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.refinement_steps = int(refinement_steps)
        self.correction_clamp = float(correction_clamp)
        self.untie_corrections = bool(untie_corrections)

        self.trunk = BoardConvStem(
            input_channels=input_channels,
            channels=channels,
            depth=depth,
            use_batchnorm=use_batchnorm,
        )
        self.pool_norm = nn.LayerNorm(channels)
        self.initial_head = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

        confidence_dim = 5 if num_classes == 1 else 6
        if self.untie_corrections:
            self.correction_blocks = nn.ModuleList(
                [
                    CorrectionMLP(
                        latent_dim=channels,
                        logit_dim=num_classes,
                        confidence_dim=confidence_dim,
                        hidden_dim=correction_hidden,
                        num_classes=num_classes,
                        dropout=dropout,
                    )
                    for _ in range(self.refinement_steps)
                ]
            )
            self.correction_block = None
        else:
            self.correction_blocks = None
            self.correction_block = CorrectionMLP(
                latent_dim=channels,
                logit_dim=num_classes,
                confidence_dim=confidence_dim,
                hidden_dim=correction_hidden,
                num_classes=num_classes,
                dropout=dropout,
            )

        self.config = IterativeLogitRefinementCNNConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            hidden_dim=hidden_dim,
            depth=depth,
            refinement_steps=refinement_steps,
            correction_hidden=correction_hidden,
            correction_clamp=float(correction_clamp),
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            untie_corrections=bool(untie_corrections),
        )

    def _correction_for_step(self, step: int) -> CorrectionMLP:
        if self.correction_blocks is not None:
            return self.correction_blocks[step]
        assert self.correction_block is not None
        return self.correction_block

    def _confidence(self, logit: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 1:
            return _binary_confidence_features(logit.squeeze(-1)).reshape(logit.shape[0], -1)
        return _multiclass_confidence_features(logit)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        feature_map = self.trunk(board)
        latent = feature_map.mean(dim=(2, 3))
        latent = self.pool_norm(latent)

        initial_logit = self.initial_head(latent)  # (B, num_classes)
        prev_logit = initial_logit
        step_logits: list[torch.Tensor] = [initial_logit]
        corrections: list[torch.Tensor] = []
        for step in range(self.refinement_steps):
            block = self._correction_for_step(step)
            confidence = self._confidence(prev_logit)
            raw = block(latent, prev_logit, confidence)
            correction = self.correction_clamp * torch.tanh(raw)
            new_logit = prev_logit + correction
            corrections.append(correction)
            step_logits.append(new_logit)
            prev_logit = new_logit
        final_logit = prev_logit

        step_stack = torch.stack(step_logits, dim=1)  # (B, T+1, num_classes)
        correction_stack = torch.stack(corrections, dim=1)  # (B, T, num_classes)
        correction_norms = correction_stack.norm(dim=-1)  # (B, T)
        correction_total = correction_norms.sum(dim=1)
        correction_mean = correction_norms.mean(dim=1)
        final_minus_initial = (final_logit - initial_logit).norm(dim=-1)

        if self.num_classes == 1:
            initial_scalar = initial_logit.squeeze(-1)
            final_scalar = final_logit.squeeze(-1)
            after_step1 = step_stack[:, 1, 0]
            flip_after_step1 = (
                (initial_scalar.sign() != after_step1.sign())
                & (initial_scalar.abs() > 1.0e-6)
                & (after_step1.abs() > 1.0e-6)
            ).to(initial_scalar.dtype)
            confidence_growth = final_scalar.abs() - initial_scalar.abs()
            initial_diag = initial_scalar
            final_diag = final_scalar
        else:
            initial_pred = initial_logit.argmax(dim=-1)
            after_step1_pred = step_stack[:, 1].argmax(dim=-1)
            flip_after_step1 = (initial_pred != after_step1_pred).to(initial_logit.dtype)
            confidence_growth = (
                torch.softmax(final_logit, dim=-1).amax(dim=-1)
                - torch.softmax(initial_logit, dim=-1).amax(dim=-1)
            )
            initial_diag = initial_logit.norm(dim=-1)
            final_diag = final_logit.norm(dim=-1)

        output = {
            "logits": _format_logits(final_logit, self.num_classes),
            "initial_logit": initial_diag,
            "final_logit": final_diag,
            "step_logits": step_stack.squeeze(-1) if self.num_classes == 1 else step_stack,
            "correction_norms": correction_norms,
            "correction_norm_mean": correction_mean,
            "correction_total": correction_total,
            "final_minus_initial": final_minus_initial,
            "flip_after_step1": flip_after_step1,
            "confidence_growth": confidence_growth,
            "trunk_feature_energy": feature_map.square().mean(dim=(1, 2, 3)),
            "latent_norm": latent.norm(dim=1),
        }
        return output


def build_iterative_logit_refinement_cnn_from_config(
    config: dict[str, Any],
) -> IterativeLogitRefinementCNN:
    channels = int(config.get("channels", config.get("trunk_width", 64)))
    depth = int(config.get("depth", config.get("trunk_depth", 4)))
    refinement_steps = int(config.get("refinement_steps", 4))
    correction_hidden = int(
        config.get("correction_hidden", config.get("hidden_dim", 64))
    )
    return IterativeLogitRefinementCNN(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=channels,
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=depth,
        refinement_steps=refinement_steps,
        correction_hidden=correction_hidden,
        correction_clamp=float(config.get("correction_clamp", 0.25)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        untie_corrections=bool(config.get("untie_corrections", False)),
    )
