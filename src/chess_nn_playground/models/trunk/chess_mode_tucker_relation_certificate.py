"""Chess-Mode Tucker Relation Certificate for idea i090."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


SQUARES = 64
RELATION_FAMILIES = 12
DEPTHS = 8
REGIONS = 10


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _square(row: int, file: int) -> int:
    return row * 8 + file


def _row_file(square: int) -> tuple[int, int]:
    return square // 8, square % 8


def _inside(row: int, file: int) -> bool:
    return 0 <= row < 8 and 0 <= file < 8


class RelationMaskBuilder:
    """Build fixed chess-relation and board-region masks."""

    ray_directions = (
        (-1, 0),  # north in tensor row coordinates
        (1, 0),
        (0, 1),
        (0, -1),
        (-1, 1),
        (-1, -1),
        (1, 1),
        (1, -1),
    )
    knight_offsets = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
    king_offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    white_pawn_offsets = ((-1, -1), (-1, 1))
    black_pawn_offsets = ((1, -1), (1, 1))

    @classmethod
    def build_relation_masks(cls) -> tuple[torch.Tensor, torch.Tensor]:
        masks = torch.zeros(RELATION_FAMILIES, DEPTHS, SQUARES, SQUARES, dtype=torch.float32)
        for source in range(SQUARES):
            src_row, src_col = _row_file(source)
            for rho, (dr, df) in enumerate(cls.ray_directions):
                for depth in range(7):
                    row = src_row + (depth + 1) * dr
                    file = src_col + (depth + 1) * df
                    if _inside(row, file):
                        masks[rho, depth, source, _square(row, file)] = 1.0
            for depth, (dr, df) in enumerate(cls.knight_offsets):
                row, file = src_row + dr, src_col + df
                if _inside(row, file):
                    masks[8, depth, source, _square(row, file)] = 1.0
            for depth, (dr, df) in enumerate(cls.king_offsets):
                row, file = src_row + dr, src_col + df
                if _inside(row, file):
                    masks[9, depth, source, _square(row, file)] = 1.0
            for depth, (dr, df) in enumerate(cls.white_pawn_offsets):
                row, file = src_row + dr, src_col + df
                if _inside(row, file):
                    masks[10, depth, source, _square(row, file)] = 1.0
            for depth, (dr, df) in enumerate(cls.black_pawn_offsets):
                row, file = src_row + dr, src_col + df
                if _inside(row, file):
                    masks[11, depth, source, _square(row, file)] = 1.0
        regions = cls.build_region_masks()
        return masks, regions

    @staticmethod
    def build_region_masks() -> torch.Tensor:
        masks = torch.zeros(REGIONS, SQUARES, dtype=torch.float32)
        for sq in range(SQUARES):
            row, file = _row_file(sq)
            masks[0, sq] = 1.0
            masks[1, sq] = 1.0 if (row + file) % 2 == 0 else 0.0
            masks[2, sq] = 1.0 if (row + file) % 2 == 1 else 0.0
            masks[3, sq] = 1.0 if row in {3, 4} and file in {3, 4} else 0.0
            masks[4, sq] = 1.0 if 2 <= row <= 5 and 2 <= file <= 5 else 0.0
            masks[5, sq] = 1.0 if row in {0, 7} and file in {0, 7} else 0.0
            masks[6, sq] = 1.0 if row in {0, 7} or file in {0, 7} else 0.0
            masks[7, sq] = 1.0 if row in {0, 7} else 0.0
            masks[8, sq] = 1.0 if file in {0, 7} else 0.0
            masks[9, sq] = 1.0 if row in {1, 6} else 0.0
        return masks / masks.sum(dim=1, keepdim=True).clamp_min(1.0)


class ChessModeTuckerRelationCertificate(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        latent_channels: int = 32,
        rank_k: int = 8,
        rank_r: int = 6,
        rank_d: int = 4,
        rank_g: int = 5,
        tucker_features: int = 24,
        head_hidden: int = 32,
        groupnorm_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("ChessModeTuckerRelationCertificate supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.latent_channels = int(latent_channels)
        self.rank_shape = (int(rank_k), int(rank_r), int(rank_d), int(rank_g))
        self.tucker_features = int(tucker_features)
        self.channel_lift = nn.Conv2d(int(input_channels), self.latent_channels, kernel_size=1, bias=True)
        groups = max(1, math.gcd(int(groupnorm_groups), self.latent_channels))
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=self.latent_channels)
        relation_masks, region_masks = RelationMaskBuilder.build_relation_masks()
        self.register_buffer("relation_masks", relation_masks, persistent=False)
        self.register_buffer("region_masks", region_masks, persistent=False)
        self.register_buffer("deg_sqrt", relation_masks.sum(dim=-1).clamp_min(1.0).sqrt(), persistent=False)
        self.Uk = nn.Parameter(torch.empty(self.latent_channels, int(rank_k)))
        self.Ur = nn.Parameter(torch.empty(RELATION_FAMILIES, int(rank_r)))
        self.Ud = nn.Parameter(torch.empty(DEPTHS, int(rank_d)))
        self.Ug = nn.Parameter(torch.empty(REGIONS, int(rank_g)))
        self.core = nn.Parameter(torch.empty(int(rank_k), int(rank_r), int(rank_d), int(rank_g), self.tucker_features))
        self.out1 = nn.Linear(self.tucker_features, int(head_hidden))
        self.out2 = nn.Linear(int(head_hidden), 1)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.channel_lift.weight, a=math.sqrt(5))
        if self.channel_lift.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.channel_lift.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.channel_lift.bias, -bound, bound)
        for factor in (self.Uk, self.Ur, self.Ud, self.Ug):
            nn.init.orthogonal_(factor)
        nn.init.normal_(self.core, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.out1.weight)
        nn.init.zeros_(self.out1.bias)
        nn.init.xavier_uniform_(self.out2.weight)
        nn.init.zeros_(self.out2.bias)

    def board_embedding(self, x: torch.Tensor) -> torch.Tensor:
        board = require_board_tensor(x, self.spec)
        lifted = self.channel_lift(board)
        return F.silu(self.norm(lifted)).flatten(2)

    def relation_tensor(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.board_embedding(x)
        masks = self.relation_masks.to(device=emb.device, dtype=emb.dtype)
        deg_sqrt = self.deg_sqrt.to(device=emb.device, dtype=emb.dtype).clamp_min(1.0)
        scan = torch.einsum("rdst,bkt->bkrds", masks, emb)
        scan = scan / deg_sqrt.view(1, 1, RELATION_FAMILIES, DEPTHS, SQUARES)
        regions = self.region_masks.to(device=emb.device, dtype=emb.dtype)
        return torch.einsum("gs,bks,bkrds->bkrdg", regions, emb, torch.tanh(scan))

    def tucker_project(self, relation_tensor: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bkrdg,ka,rl,dm,gn->balmn", relation_tensor, self.Uk, self.Ur, self.Ud, self.Ug)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tensor = self.relation_tensor(x)
        projected = self.tucker_project(tensor)
        features = torch.einsum("balmn,almnh->bh", projected, self.core)
        return features, projected, tensor

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> dict[str, torch.Tensor]:
        features, projected, tensor = self.forward_features(x)
        hidden = F.silu(self.out1(self.dropout(features)))
        logits = _format_logits(self.out2(hidden), self.num_classes)
        rank_diag = self.rank_certificate(projected)
        orth = self.orthogonality_penalty().expand(logits.shape[0])
        output = {
            "logits": logits,
            "prob": torch.sigmoid(logits),
            "tucker_features": features,
            "projected_tensor_energy": projected.pow(2).mean(dim=(1, 2, 3, 4)),
            "relation_tensor_energy": tensor.pow(2).mean(dim=(1, 2, 3, 4)),
            "nuclear_bottleneck": rank_diag["nuclear_bottleneck"],
            "orthogonality_penalty": orth,
            "rank_certificate": rank_diag["rank_certificate"],
            "K_mode_eff_rank": rank_diag["K_mode_eff_rank"],
            "R_mode_eff_rank": rank_diag["R_mode_eff_rank"],
            "D_mode_eff_rank": rank_diag["D_mode_eff_rank"],
            "G_mode_eff_rank": rank_diag["G_mode_eff_rank"],
            "fixed_relation_density": self.relation_masks.mean().to(device=logits.device, dtype=logits.dtype).expand_as(logits),
            "region_mass_error": (self.region_masks.sum(dim=1) - 1.0).abs().mean().to(device=logits.device, dtype=logits.dtype).expand_as(logits),
            "mechanism_energy": features.pow(2).mean(dim=1),
            "proposal_profile_strength": features.abs().max(dim=1).values,
            "proposal_keyword_count": logits.new_full((logits.shape[0],), 4.0),
        }
        if return_aux:
            output["projected_tensor"] = projected
            output["relation_tensor"] = tensor
        return output

    def forward_with_aux(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward(x, return_aux=True)

    @staticmethod
    def rank_certificate(projected: torch.Tensor) -> dict[str, torch.Tensor]:
        ranks = []
        nuc_terms = []
        fro = projected.flatten(1).norm(dim=1).clamp_min(1.0e-8)
        for mode in range(1, 5):
            moved = projected.movedim(mode, 1)
            unfolded = moved.flatten(2)
            singular_values = torch.linalg.svdvals(unfolded)
            nuclear = singular_values.sum(dim=-1)
            singular_fro = singular_values.norm(dim=-1).clamp_min(1.0e-8)
            ranks.append((nuclear / singular_fro).pow(2))
            nuc_terms.append(nuclear / fro)
        rank_certificate = torch.stack(ranks, dim=1)
        nuclear_bottleneck = torch.stack(nuc_terms, dim=1).sum(dim=1)
        return {
            "rank_certificate": rank_certificate,
            "nuclear_bottleneck": nuclear_bottleneck,
            "K_mode_eff_rank": rank_certificate[:, 0],
            "R_mode_eff_rank": rank_certificate[:, 1],
            "D_mode_eff_rank": rank_certificate[:, 2],
            "G_mode_eff_rank": rank_certificate[:, 3],
        }

    def orthogonality_penalty(self) -> torch.Tensor:
        penalty = self.Uk.new_zeros(())
        for factor in (self.Uk, self.Ur, self.Ud, self.Ug):
            gram = factor.T @ factor
            eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            penalty = penalty + (gram - eye).pow(2).sum()
        return penalty


class FlatProjectedMLPControl(nn.Module):
    """Same-parameter flat control over the same fixed relation tensor."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        latent_channels: int = 32,
        projection_dim: int = 112,
        hidden_dim: int = 213,
        seed: int = 1729,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("FlatProjectedMLPControl supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.latent_channels = int(latent_channels)
        self.projection_dim = int(projection_dim)
        self.channel_lift = nn.Conv2d(int(input_channels), self.latent_channels, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(num_groups=max(1, math.gcd(8, self.latent_channels)), num_channels=self.latent_channels)
        relation_masks, region_masks = RelationMaskBuilder.build_relation_masks()
        self.register_buffer("relation_masks", relation_masks, persistent=False)
        self.register_buffer("region_masks", region_masks, persistent=False)
        self.register_buffer("deg_sqrt", relation_masks.sum(dim=-1).clamp_min(1.0).sqrt(), persistent=False)
        flat_dim = self.latent_channels * RELATION_FAMILIES * DEPTHS * REGIONS
        generator = torch.Generator().manual_seed(int(seed))
        self.register_buffer("bucket", torch.randint(0, self.projection_dim, (flat_dim,), generator=generator), persistent=False)
        signs = torch.randint(0, 2, (flat_dim,), generator=generator, dtype=torch.float32) * 2.0 - 1.0
        self.register_buffer("sign", signs, persistent=False)
        self.fc1 = nn.Linear(self.projection_dim, int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), 1)

    def relation_tensor(self, x: torch.Tensor) -> torch.Tensor:
        board = require_board_tensor(x, self.spec)
        emb = F.silu(self.norm(self.channel_lift(board))).flatten(2)
        masks = self.relation_masks.to(device=emb.device, dtype=emb.dtype)
        scan = torch.einsum("rdst,bkt->bkrds", masks, emb)
        scan = scan / self.deg_sqrt.to(device=emb.device, dtype=emb.dtype).clamp_min(1.0).view(1, 1, RELATION_FAMILIES, DEPTHS, SQUARES)
        return torch.einsum("gs,bks,bkrds->bkrdg", self.region_masks.to(device=emb.device, dtype=emb.dtype), emb, torch.tanh(scan))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tensor = self.relation_tensor(x)
        flat = tensor.flatten(1)
        projected = flat.new_zeros(flat.shape[0], self.projection_dim)
        bucket = self.bucket.to(device=flat.device).view(1, -1).expand(flat.shape[0], -1)
        sign = self.sign.to(device=flat.device, dtype=flat.dtype).view(1, -1)
        projected.scatter_add_(1, bucket, flat * sign)
        projected = projected / math.sqrt(float(self.projection_dim))
        logits = _format_logits(self.fc2(F.silu(self.fc1(projected))), self.num_classes)
        return {
            "logits": logits,
            "prob": torch.sigmoid(logits),
            "flat_projection_energy": projected.pow(2).mean(dim=1),
            "relation_tensor_energy": tensor.pow(2).mean(dim=(1, 2, 3, 4)),
        }


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@torch.no_grad()
def fine_label_diagnostic_3x2(logits: torch.Tensor, fine_label: torch.Tensor, tau: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    pred = (logits.sigmoid().view(-1) >= float(tau)).long()
    fine = fine_label.view(-1).long()
    counts = torch.zeros(3, 2, dtype=torch.long, device=fine.device)
    for fine_value in range(3):
        mask_f = fine == fine_value
        for pred_value in range(2):
            counts[fine_value, pred_value] = (mask_f & (pred == pred_value)).sum()
    rates = counts.float() / counts.sum(dim=1, keepdim=True).clamp_min(1)
    return counts.cpu(), rates.cpu()


def build_chess_mode_tucker_relation_certificate_from_config(config: dict[str, Any]) -> ChessModeTuckerRelationCertificate:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    latent_channels = int(cfg.get("latent_channels", cfg.get("channels", 32)))
    return ChessModeTuckerRelationCertificate(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        latent_channels=latent_channels,
        rank_k=int(cfg.get("rank_k", 8)),
        rank_r=int(cfg.get("rank_r", 6)),
        rank_d=int(cfg.get("rank_d", 4)),
        rank_g=int(cfg.get("rank_g", 5)),
        tucker_features=int(cfg.get("tucker_features", 24)),
        head_hidden=int(cfg.get("head_hidden", 32)),
        groupnorm_groups=int(cfg.get("groupnorm_groups", 8)),
        dropout=float(cfg.get("dropout", 0.0)),
    )
