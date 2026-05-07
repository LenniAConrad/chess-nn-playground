"""Oriented Matroid Covector Bottleneck for idea i096."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


class OccupiedPieceTokenizer(nn.Module):
    """Extract up to ``max_pieces`` occupied square tokens from simple board planes."""

    def __init__(self, input_channels: int = 18, max_pieces: int = 32, occupancy_threshold: float = 0.05) -> None:
        super().__init__()
        if input_channels < 12:
            raise ValueError("OccupiedPieceTokenizer expects at least 12 piece planes")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.input_channels = int(input_channels)
        self.max_pieces = int(max_pieces)
        self.occupancy_threshold = float(occupancy_threshold)
        coords = torch.linspace(-1.0, 1.0, 8)
        rank = coords.view(8, 1).expand(8, 8).reshape(64)
        file = coords.view(1, 8).expand(8, 8).reshape(64)
        square_color = (((torch.arange(64) // 8 + torch.arange(64) % 8) % 2).float() * 2.0 - 1.0)
        self.register_buffer("square_features", torch.stack([rank, file, square_color], dim=1), persistent=False)

        generator = torch.Generator().manual_seed(1960)
        role_perms = torch.stack([torch.randperm(64, generator=generator) for _ in range(12)], dim=0)
        self.register_buffer("role_square_permutations", role_perms, persistent=False)

    @property
    def token_feature_dim(self) -> int:
        extras = max(0, self.input_channels - 12)
        return 12 + 3 + 1 + extras

    def forward(self, x: torch.Tensor, *, coordinate_shuffle_by_piece: bool = False) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        piece_planes = board[:, :12].clamp_min(0.0)
        role_by_square = piece_planes.flatten(2)
        square_occupancy = role_by_square.sum(dim=1).clamp_min(0.0)
        max_tokens = min(self.max_pieces, 64)
        top_occ, top_idx = square_occupancy.topk(max_tokens, dim=1)
        token_mask = (top_occ > self.occupancy_threshold).to(dtype=board.dtype)

        gather_idx = top_idx.unsqueeze(1).expand(-1, 12, -1)
        role_values = role_by_square.gather(2, gather_idx).transpose(1, 2)
        role_probs = role_values / role_values.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        role_idx = role_probs.argmax(dim=-1)

        coord_idx = top_idx
        if coordinate_shuffle_by_piece:
            perms = self.role_square_permutations.to(device=board.device)
            coord_idx = perms[role_idx, top_idx]
        square_features = self.square_features.to(device=board.device, dtype=board.dtype)
        coord_features = square_features.index_select(0, coord_idx.reshape(-1)).view(batch, max_tokens, 3)

        if self.input_channels > 12:
            aux = board[:, 12:].flatten(2)
            aux_gather = top_idx.unsqueeze(1).expand(-1, aux.shape[1], -1)
            aux_values = aux.gather(2, aux_gather).transpose(1, 2)
        else:
            aux_values = board.new_zeros(batch, max_tokens, 0)

        token_features = torch.cat([role_probs, coord_features, top_occ.unsqueeze(-1), aux_values], dim=-1)
        token_features = token_features * token_mask.unsqueeze(-1)
        return {
            "token_features": token_features,
            "role_probs": role_probs * token_mask.unsqueeze(-1),
            "square_indices": top_idx,
            "token_mask": token_mask,
            "piece_count": token_mask.sum(dim=1),
            "occupancy": top_occ * token_mask,
        }


class PieceTokenEncoder(nn.Module):
    def __init__(self, token_feature_dim: int, token_dim: int = 48, hidden_dim: int = 96, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(token_feature_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(hidden_dim), int(token_dim)),
            nn.LayerNorm(int(token_dim)),
            nn.GELU(),
        )

    def forward(self, token_features: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        return self.net(token_features) * token_mask.unsqueeze(-1)


class HyperplaneArrangement(nn.Module):
    def __init__(self, token_dim: int = 48, hyperplanes: int = 24, sign_scale: float = 4.0, random_seed: int = 1961) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.hyperplanes = int(hyperplanes)
        self.sign_scale = float(sign_scale)
        self.weights = nn.Parameter(torch.randn(self.hyperplanes, self.token_dim) * 0.05)
        self.bias = nn.Parameter(torch.zeros(self.hyperplanes))

        generator = torch.Generator().manual_seed(int(random_seed))
        random_weights = torch.randn(self.hyperplanes, self.token_dim, generator=generator)
        random_bias = torch.randn(self.hyperplanes, generator=generator) * 0.05
        self.register_buffer("random_weights", random_weights, persistent=False)
        self.register_buffer("random_bias", random_bias, persistent=False)

    def forward(self, embeddings: torch.Tensor, *, random_hyperplanes: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if random_hyperplanes:
            weights = self.random_weights.to(device=embeddings.device, dtype=embeddings.dtype)
            bias = self.random_bias.to(device=embeddings.device, dtype=embeddings.dtype)
        else:
            weights = self.weights.to(device=embeddings.device, dtype=embeddings.dtype)
            bias = self.bias.to(device=embeddings.device, dtype=embeddings.dtype)
        weights = F.normalize(weights, dim=1)
        scores = torch.einsum("bnd,pd->bnp", embeddings, weights) + bias.view(1, 1, -1)
        return scores, torch.tanh(self.sign_scale * scores)


class CovectorStats(nn.Module):
    def __init__(self, hyperplanes: int = 24, roles: int = 12) -> None:
        super().__init__()
        self.hyperplanes = int(hyperplanes)
        self.roles = int(roles)
        self.output_dim = self.hyperplanes * 3 + self.hyperplanes * self.hyperplanes + self.roles * self.hyperplanes
        self.output_dim += self.roles + self.hyperplanes * 4 + 8

    def forward(
        self,
        scores: torch.Tensor,
        signs: torch.Tensor,
        role_probs: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        magnitude_only: bool = False,
        material_role_hist_only: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask = token_mask.unsqueeze(-1)
        denom = token_mask.sum(dim=1).clamp_min(1.0)

        if material_role_hist_only:
            active_signs = signs.new_zeros(signs.shape)
            active_scores = scores.new_zeros(scores.shape)
        elif magnitude_only:
            active_scores = scores.abs()
            active_signs = torch.tanh(active_scores)
        else:
            active_scores = scores
            active_signs = signs

        positive = F.relu(active_signs) * mask
        negative = F.relu(-active_signs) * mask
        near_zero = (1.0 - active_signs.abs()).clamp_min(0.0) * mask
        positive_counts = positive.sum(dim=1) / denom.unsqueeze(-1)
        negative_counts = negative.sum(dim=1) / denom.unsqueeze(-1)
        near_zero_counts = near_zero.sum(dim=1) / denom.unsqueeze(-1)

        signed_masked = active_signs * mask
        sign_agreement = torch.einsum("bnp,bnq->bpq", signed_masked, signed_masked) / denom.view(-1, 1, 1)

        role_weights = role_probs * token_mask.unsqueeze(-1)
        role_histogram = role_weights.sum(dim=1) / denom.unsqueeze(-1)
        role_sign_entropy = self._role_sign_entropy(positive, negative, near_zero, role_weights)

        sign_mean = signed_masked.sum(dim=1) / denom.unsqueeze(-1)
        sign_abs_mean = active_signs.abs().mul(mask).sum(dim=1) / denom.unsqueeze(-1)
        score_abs_mean = active_scores.abs().mul(mask).sum(dim=1) / denom.unsqueeze(-1)
        score_std = self._masked_std(active_scores, token_mask)
        covector_entropy = self._sign_entropy(positive_counts, negative_counts, near_zero_counts)

        globals_ = torch.stack(
            [
                denom,
                positive_counts.mean(dim=1),
                negative_counts.mean(dim=1),
                near_zero_counts.mean(dim=1),
                sign_agreement.square().mean(dim=(1, 2)),
                sign_abs_mean.mean(dim=1),
                score_abs_mean.mean(dim=1),
                covector_entropy.mean(dim=1),
            ],
            dim=1,
        )
        features = torch.cat(
            [
                positive_counts,
                negative_counts,
                near_zero_counts,
                sign_agreement.flatten(1),
                role_sign_entropy.flatten(1),
                role_histogram,
                sign_mean,
                sign_abs_mean,
                score_abs_mean,
                score_std,
                globals_,
            ],
            dim=1,
        )
        diagnostics = {
            "positive_counts": positive_counts,
            "negative_counts": negative_counts,
            "near_zero_counts": near_zero_counts,
            "sign_agreement": sign_agreement,
            "role_sign_entropy": role_sign_entropy,
            "role_histogram": role_histogram,
            "sign_mean": sign_mean,
            "sign_abs_mean": sign_abs_mean,
            "score_abs_mean": score_abs_mean,
            "score_std": score_std,
            "covector_entropy": covector_entropy,
            "piece_count": denom,
            "near_zero_rate": near_zero_counts.mean(dim=1),
            "pairwise_agreement_energy": sign_agreement.square().mean(dim=(1, 2)),
        }
        return features, diagnostics

    @staticmethod
    def _masked_std(values: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        denom = token_mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        mask = token_mask.unsqueeze(-1)
        mean = (values * mask).sum(dim=1) / denom
        var = ((values - mean.unsqueeze(1)).square() * mask).sum(dim=1) / denom
        return var.clamp_min(0.0).sqrt()

    @staticmethod
    def _sign_entropy(pos: torch.Tensor, neg: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        probs = torch.stack([pos, neg, zero], dim=-1)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        entropy = -(probs * probs.clamp_min(1.0e-6).log()).sum(dim=-1)
        return entropy / torch.log(probs.new_tensor(3.0))

    def _role_sign_entropy(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
        near_zero: torch.Tensor,
        role_weights: torch.Tensor,
    ) -> torch.Tensor:
        role_denom = role_weights.sum(dim=1).clamp_min(1.0e-6)
        pos = torch.einsum("bnr,bnp->brp", role_weights, positive) / role_denom.unsqueeze(-1)
        neg = torch.einsum("bnr,bnp->brp", role_weights, negative) / role_denom.unsqueeze(-1)
        zero = torch.einsum("bnr,bnp->brp", role_weights, near_zero) / role_denom.unsqueeze(-1)
        probs = torch.stack([pos, neg, zero], dim=-1)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        entropy = -(probs * probs.clamp_min(1.0e-6).log()).sum(dim=-1)
        return entropy / torch.log(probs.new_tensor(3.0))


class OrientedMatroidCovectorBottleneck(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        max_pieces: int = 32,
        token_dim: int = 48,
        hidden_dim: int = 96,
        hyperplanes: int = 24,
        sign_scale: float = 4.0,
        head_hidden: int = 192,
        dropout: float = 0.1,
        mode: str = "covector",
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("OrientedMatroidCovectorBottleneck supports the puzzle_binary one-logit contract")
        self.num_classes = int(num_classes)
        self.mode = str(mode)
        self.tokenizer = OccupiedPieceTokenizer(input_channels=int(input_channels), max_pieces=int(max_pieces))
        self.token_encoder = PieceTokenEncoder(
            token_feature_dim=self.tokenizer.token_feature_dim,
            token_dim=int(token_dim),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        )
        self.arrangement = HyperplaneArrangement(
            token_dim=int(token_dim),
            hyperplanes=int(hyperplanes),
            sign_scale=float(sign_scale),
        )
        self.stats = CovectorStats(hyperplanes=int(hyperplanes), roles=12)
        self.head = nn.Sequential(
            nn.LayerNorm(self.stats.output_dim),
            nn.Linear(self.stats.output_dim, int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(head_hidden), max(32, int(head_hidden) // 4)),
            nn.GELU(),
            nn.Linear(max(32, int(head_hidden) // 4), 1),
        )

    def forward(self, x: torch.Tensor, *, return_covectors: bool = False) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(x, coordinate_shuffle_by_piece=self.mode == "coordinate_shuffle_by_piece")
        embeddings = self.token_encoder(tokens["token_features"], tokens["token_mask"])
        random_hyperplanes = self.mode == "random_hyperplanes"
        scores, signs = self.arrangement(embeddings, random_hyperplanes=random_hyperplanes)
        covector_features, diagnostics = self.stats(
            scores,
            signs,
            tokens["role_probs"],
            tokens["token_mask"],
            magnitude_only=self.mode == "magnitude_only",
            material_role_hist_only=self.mode == "material_role_hist_only",
        )
        logits = _format_logits(self.head(covector_features), self.num_classes)
        output = {
            "logits": logits,
            "prob": torch.sigmoid(logits),
            "covector_features": covector_features,
            "token_mask": tokens["token_mask"],
            "piece_count": diagnostics["piece_count"],
            "soft_signs": signs * tokens["token_mask"].unsqueeze(-1),
            "hyperplane_scores": scores * tokens["token_mask"].unsqueeze(-1),
            "positive_counts": diagnostics["positive_counts"],
            "negative_counts": diagnostics["negative_counts"],
            "near_zero_counts": diagnostics["near_zero_counts"],
            "sign_agreement": diagnostics["sign_agreement"],
            "role_sign_entropy": diagnostics["role_sign_entropy"],
            "role_histogram": diagnostics["role_histogram"],
            "sign_mean": diagnostics["sign_mean"],
            "sign_abs_mean": diagnostics["sign_abs_mean"],
            "score_abs_mean": diagnostics["score_abs_mean"],
            "score_std": diagnostics["score_std"],
            "covector_entropy": diagnostics["covector_entropy"],
            "near_zero_rate": diagnostics["near_zero_rate"],
            "pairwise_agreement_energy": diagnostics["pairwise_agreement_energy"],
            "orientation_mode": logits.new_full((logits.shape[0],), self._mode_code()),
            "mechanism_energy": diagnostics["pairwise_agreement_energy"],
            "proposal_profile_strength": diagnostics["sign_abs_mean"].mean(dim=1),
            "proposal_keyword_count": logits.new_full((logits.shape[0],), 4.0),
        }
        if return_covectors:
            output["token_features"] = tokens["token_features"]
            output["token_embeddings"] = embeddings
            output["role_probs"] = tokens["role_probs"]
            output["square_indices"] = tokens["square_indices"]
        return output

    def _mode_code(self) -> float:
        return {
            "covector": 0.0,
            "magnitude_only": 1.0,
            "random_hyperplanes": 2.0,
            "material_role_hist_only": 3.0,
            "coordinate_shuffle_by_piece": 4.0,
        }.get(self.mode, 0.0)


def build_oriented_matroid_covector_bottleneck_from_config(config: dict[str, Any]) -> OrientedMatroidCovectorBottleneck:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    hidden_dim = int(cfg.get("hidden_dim", cfg.get("channels", 96)))
    return OrientedMatroidCovectorBottleneck(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        max_pieces=int(cfg.get("max_pieces", 32)),
        token_dim=int(cfg.get("token_dim", 48)),
        hidden_dim=hidden_dim,
        hyperplanes=int(cfg.get("hyperplanes", 24)),
        sign_scale=float(cfg.get("sign_scale", 4.0)),
        head_hidden=int(cfg.get("head_hidden", max(128, hidden_dim * 2))),
        dropout=float(cfg.get("dropout", 0.1)),
        mode=str(cfg.get("mode", "covector")),
    )
