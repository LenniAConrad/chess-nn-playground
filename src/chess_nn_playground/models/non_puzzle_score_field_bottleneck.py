"""Non-Puzzle Score-Field Bottleneck Network for idea i056.

Implements the markdown thesis: a noise-conditional Gaussian denoiser
treated as a non-puzzle score prior, evaluated at multiple noise scales
on the clean board to produce a stack of denoising-residual score maps,
funneled through a low-dimensional convolutional bottleneck and fused
with a compact board encoder before a pooled MLP classifier.

The forward pass implements the Tweedie/denoising-score identity

    s_sigma(x) = (D_theta(x, sigma) - x) / sigma^2

at K fixed noise levels and exposes the score-field statistics as
diagnostics. The architecture is materially distinct from the shared
research-packet probe; it does not consume any CRTK/source/engine
metadata, only the simple_18 board tensor.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from chess_nn_playground.models._packet_bespoke_base import (
    BoardTensorSpec,
    require_board_tensor,
)


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class NoiseLevelEmbedding(nn.Module):
    """Maps log(sigma) into a per-board feature map broadcast to 8x8."""

    def __init__(self, embedding_dim: int = 16) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.proj = nn.Sequential(
            nn.Linear(1, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def forward(self, sigma: torch.Tensor, batch_size: int) -> torch.Tensor:
        log_sigma = torch.log(sigma.clamp_min(1e-8)).view(1, 1).expand(batch_size, 1)
        emb = self.proj(log_sigma)
        return emb.view(batch_size, self.embedding_dim, 1, 1).expand(
            batch_size, self.embedding_dim, 8, 8
        )


class _DenoiserResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _group_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _group_norm(channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class OrdinaryScoreDenoiser(nn.Module):
    """Noise-conditional denoiser whose residual estimates the smoothed score."""

    def __init__(
        self,
        input_channels: int = 18,
        hidden: int = 32,
        blocks: int = 3,
        sigma_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.sigma_emb = NoiseLevelEmbedding(sigma_embedding_dim)
        in_planes = self.input_channels + sigma_embedding_dim
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, hidden, kernel_size=3, padding=1),
            _group_norm(hidden),
            nn.SiLU(),
        )
        self.body = nn.Sequential(*(_DenoiserResidualBlock(hidden) for _ in range(int(blocks))))
        self.head = nn.Conv2d(hidden, self.input_channels, kernel_size=3, padding=1)

    def forward(self, board: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma_map = self.sigma_emb(sigma, board.shape[0]).to(board.dtype)
        h = torch.cat([board, sigma_map], dim=1)
        h = self.stem(h)
        h = self.body(h)
        return self.head(h)


class ScoreFieldBottleneck(nn.Module):
    """Compresses the K*C score stack down to a small board feature map."""

    def __init__(self, in_channels: int, out_channels: int = 24) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            _group_norm(out_channels),
            nn.SiLU(),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthwise(self.proj(x))


class _BoardStem(nn.Module):
    def __init__(self, input_channels: int, channels: int, depth: int, use_batchnorm: bool) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = []
        in_c = input_channels
        for _ in range(int(depth)):
            layers.append(nn.Conv2d(in_c, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
            layers.append(nn.BatchNorm2d(channels) if use_batchnorm else _group_norm(channels))
            layers.append(nn.SiLU())
            in_c = channels
        self.body = nn.Sequential(*layers)
        self.output_channels = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _FusionResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _group_norm(channels),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _group_norm(channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class NonPuzzleScoreFieldBottleneckNetwork(nn.Module):
    """Full bespoke architecture for idea i056."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        noise_sigmas: tuple[float, ...] = (0.05, 0.10, 0.20),
        score_prior_hidden: int = 32,
        score_prior_blocks: int = 3,
        score_bottleneck_channels: int = 24,
        sigma_embedding_dim: int = 16,
        board_stem_channels: int = 64,
        board_stem_depth: int = 2,
        use_batchnorm: bool = True,
        fusion_blocks: int = 2,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        freeze_score_prior_after_pretrain: bool = True,
        score_prior_train_on_binary_zero_only: bool = True,
        fail_closed_on_unknown_encoding: bool = True,
    ) -> None:
        super().__init__()
        if input_channels != 18 and fail_closed_on_unknown_encoding:
            raise ValueError(
                f"NonPuzzleScoreFieldBottleneckNetwork requires simple_18 input "
                f"(input_channels=18), got {input_channels}"
            )
        if num_classes not in (1, 2):
            raise ValueError(
                "NonPuzzleScoreFieldBottleneckNetwork supports num_classes=1 (puzzle_binary) or 2"
            )
        if not noise_sigmas:
            raise ValueError("noise_sigmas must contain at least one positive value")
        sigmas = tuple(float(s) for s in noise_sigmas)
        for s in sigmas:
            if s <= 0:
                raise ValueError(f"noise_sigmas must be positive, got {s}")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.noise_sigmas = sigmas
        self.score_bottleneck_channels = int(score_bottleneck_channels)
        self.freeze_score_prior_after_pretrain = bool(freeze_score_prior_after_pretrain)
        self.score_prior_train_on_binary_zero_only = bool(score_prior_train_on_binary_zero_only)
        self.register_buffer(
            "_sigma_buffer",
            torch.tensor(sigmas, dtype=torch.float32),
            persistent=False,
        )

        self.denoiser = OrdinaryScoreDenoiser(
            input_channels=input_channels,
            hidden=int(score_prior_hidden),
            blocks=int(score_prior_blocks),
            sigma_embedding_dim=int(sigma_embedding_dim),
        )
        self.score_bottleneck = ScoreFieldBottleneck(
            in_channels=len(sigmas) * input_channels,
            out_channels=int(score_bottleneck_channels),
        )
        self.board_stem = _BoardStem(
            input_channels=input_channels,
            channels=int(board_stem_channels),
            depth=int(board_stem_depth),
            use_batchnorm=bool(use_batchnorm),
        )
        fusion_channels = self.board_stem.output_channels + int(score_bottleneck_channels)
        self.fusion = nn.Sequential(
            *(_FusionResidualBlock(fusion_channels, dropout=dropout) for _ in range(int(fusion_blocks)))
        )
        head_in = 2 * fusion_channels
        self.head = nn.Sequential(
            nn.Linear(head_in, int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), 1 if self.num_classes == 1 else 2),
        )

    @property
    def num_noise_levels(self) -> int:
        return len(self.noise_sigmas)

    def freeze_score_prior(self) -> None:
        for p in self.denoiser.parameters():
            p.requires_grad_(False)

    def unfreeze_score_prior(self) -> None:
        for p in self.denoiser.parameters():
            p.requires_grad_(True)

    def denoising_score_matching_loss(
        self,
        clean_board: torch.Tensor,
        binary_label: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Compute the DSM objective on (optionally label-filtered) clean boards.

        `binary_label` (B,) selects rows with label 0 when
        `score_prior_train_on_binary_zero_only=True`. When the filter would leave
        no rows the loss is zero.
        """
        clean_board = require_board_tensor(clean_board, self.spec)
        if (
            self.score_prior_train_on_binary_zero_only
            and binary_label is not None
        ):
            mask = binary_label.view(-1) == 0
            clean_board = clean_board[mask]
        if clean_board.shape[0] == 0:
            return clean_board.new_zeros(())
        sigma_idx = torch.randint(
            0,
            self.num_noise_levels,
            (clean_board.shape[0],),
            device=clean_board.device,
            generator=generator,
        )
        sigma_values = self._sigma_buffer.to(clean_board.device).to(clean_board.dtype)[sigma_idx]
        sigma_map = sigma_values.view(-1, 1, 1, 1)
        eps = torch.randn(clean_board.shape, device=clean_board.device, dtype=clean_board.dtype, generator=generator)
        noisy = clean_board + sigma_map * eps
        per_sigma_loss = clean_board.new_zeros(())
        for i, sigma in enumerate(self.noise_sigmas):
            sel = sigma_idx == i
            if not torch.any(sel):
                continue
            sub = noisy[sel]
            target = clean_board[sel]
            sigma_t = self._sigma_buffer.to(clean_board.device).to(clean_board.dtype)[i]
            recon = self.denoiser(sub, sigma_t)
            denom = 2.0 * float(sigma) * float(sigma)
            per_sigma_loss = per_sigma_loss + ((recon - target) ** 2).mean() / denom
        return per_sigma_loss / float(self.num_noise_levels)

    def _compute_score_stack(self, board: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        sigmas = self._sigma_buffer.to(board.device).to(board.dtype)
        score_maps: list[torch.Tensor] = []
        denoiser_frozen = not any(p.requires_grad for p in self.denoiser.parameters())
        ctx = torch.no_grad() if denoiser_frozen and self.training else _NullCtx()
        with ctx:
            for i in range(self.num_noise_levels):
                sigma = sigmas[i]
                recon = self.denoiser(board, sigma)
                score_maps.append((recon - board) / (sigma * sigma))
        score_stack = torch.cat(score_maps, dim=1)
        return score_stack, score_maps

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        score_stack, score_maps = self._compute_score_stack(x)
        z_score = self.score_bottleneck(score_stack)
        z_board = self.board_stem(x)
        z = torch.cat([z_board, z_score], dim=1)
        z = self.fusion(z)
        avg = F.adaptive_avg_pool2d(z, 1).flatten(1)
        mx = F.adaptive_max_pool2d(z, 1).flatten(1)
        pooled = torch.cat([avg, mx], dim=1)
        raw_logits = self.head(pooled)
        if self.num_classes == 1:
            logits = raw_logits.view(-1)
        else:
            logits = raw_logits

        per_sigma_norms = torch.stack(
            [m.flatten(1).norm(dim=1) for m in score_maps], dim=1
        )
        score_field_norm = score_stack.flatten(1).norm(dim=1)
        score_residual_energy = score_stack.pow(2).mean(dim=(1, 2, 3))
        score_bottleneck_energy = z_score.pow(2).mean(dim=(1, 2, 3))
        score_field_mean_abs = score_stack.abs().mean(dim=(1, 2, 3))
        score_field_max_abs = score_stack.abs().amax(dim=(1, 2, 3))
        recon_residual_l2 = torch.stack(
            [
                ((m * (s * s)) ** 2).mean(dim=(1, 2, 3))
                for m, s in zip(score_maps, [float(s) for s in self.noise_sigmas])
            ],
            dim=1,
        )

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "score_field_norm": score_field_norm,
            "score_residual_energy": score_residual_energy,
            "score_bottleneck_energy": score_bottleneck_energy,
            "score_field_mean_abs": score_field_mean_abs,
            "score_field_max_abs": score_field_max_abs,
            "score_per_sigma_norm": per_sigma_norms,
            "recon_residual_l2": recon_residual_l2,
            "mechanism_energy": score_residual_energy,
            "proposal_profile_strength": score_field_max_abs,
            "proposal_keyword_count": logits.new_full(
                (x.shape[0],), float(self.num_noise_levels)
            ),
        }
        if self.num_classes == 1:
            output["two_class_logits"] = torch.stack([-0.5 * logits, 0.5 * logits], dim=1)
        else:
            output["two_class_logits"] = raw_logits
        return output

    def forward_with_aux(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        score_stack, score_maps = self._compute_score_stack(x)
        z_score = self.score_bottleneck(score_stack)
        z_board = self.board_stem(x)
        out = self.forward(x)
        out["score_stack"] = score_stack
        out["score_maps_per_sigma"] = torch.stack(score_maps, dim=1)
        out["score_bottleneck_features"] = z_score
        out["board_features"] = z_board
        return out


class _NullCtx:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def build_non_puzzle_score_field_bottleneck_network_from_config(
    config: dict[str, Any],
) -> NonPuzzleScoreFieldBottleneckNetwork:
    cfg = dict(config)
    cfg.setdefault("input_channels", 18)
    cfg.setdefault("num_classes", 1)
    sigmas = cfg.get("noise_sigmas", (0.05, 0.10, 0.20))
    if isinstance(sigmas, (list, tuple)):
        sigmas = tuple(float(s) for s in sigmas)
    else:
        sigmas = (float(sigmas),)
    return NonPuzzleScoreFieldBottleneckNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        noise_sigmas=sigmas,
        score_prior_hidden=int(cfg.get("score_prior_hidden", 32)),
        score_prior_blocks=int(cfg.get("score_prior_blocks", 3)),
        score_bottleneck_channels=int(cfg.get("score_bottleneck_channels", 24)),
        sigma_embedding_dim=int(cfg.get("sigma_embedding_dim", 16)),
        board_stem_channels=int(cfg.get("channels", 64)),
        board_stem_depth=int(cfg.get("depth", 2)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        fusion_blocks=int(cfg.get("fusion_blocks", 2)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        dropout=float(cfg.get("dropout", 0.1)),
        freeze_score_prior_after_pretrain=bool(cfg.get("freeze_score_prior_after_pretrain", True)),
        score_prior_train_on_binary_zero_only=bool(
            cfg.get("score_prior_train_on_binary_zero_only", True)
        ),
        fail_closed_on_unknown_encoding=bool(cfg.get("fail_closed_on_unknown_encoding", True)),
    )
