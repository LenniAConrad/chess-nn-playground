"""Tensor-Core Square-Pair Field Network for idea i072."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


RELATION_NAMES: tuple[str, ...] = (
    "same_square",
    "same_rank",
    "same_file",
    "same_diag",
    "same_anti_diag",
    "same_square_color",
    "opposite_square_color",
    "knight_offset",
    "king_offset",
    "manhattan_distance_1",
    "manhattan_distance_2",
    "manhattan_distance_3",
    "chebyshev_distance_1",
    "chebyshev_distance_2",
    "same_center_ring",
    "same_edge_class",
    "rank_order_forward",
    "file_order_forward",
)

PAIR_ABLATIONS = {
    "none",
    "cnn_only_matched",
    "no_pair_update",
    "no_pair_readout",
    "relation_bank_shuffle",
    "softmax_attention_control",
    "low_head_count",
    "pair_energy_only",
}


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _masked_token_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _pair_weighted_mean(energy: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(dtype=energy.dtype)
    denom = weights.sum(dim=(1, 2)).clamp_min(1.0)
    return (energy.mean(dim=1) * weights).sum(dim=(1, 2)) / denom


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * scale * self.weight


def _make_norm(width: int, norm: str) -> nn.Module:
    if norm == "rmsnorm":
        return RMSNorm(width)
    if norm == "layernorm":
        return nn.LayerNorm(width)
    raise ValueError("norm must be 'rmsnorm' or 'layernorm'")


class TokenFeedForward(nn.Module):
    def __init__(self, width: int, hidden_width: int, activation: str = "gelu", dropout: float = 0.0) -> None:
        super().__init__()
        self.activation = activation
        if activation == "swiglu":
            self.in_proj = nn.Linear(width, hidden_width * 2)
            self.out_proj = nn.Linear(hidden_width, width)
        elif activation == "gelu":
            self.net = nn.Sequential(
                nn.Linear(width, hidden_width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_width, width),
            )
        else:
            raise ValueError("activation must be 'gelu' or 'swiglu'")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            gate, value = self.in_proj(x).chunk(2, dim=-1)
            return self.dropout(self.out_proj(F.silu(gate) * value))
        return self.net(x)


class SquarePairRelationBank(nn.Module):
    relation_names = RELATION_NAMES

    def __init__(self, shuffle_seed: int = 72072) -> None:
        super().__init__()
        relation_bank = self._build_relation_bank()
        shuffled = self._build_shuffled_bank(relation_bank, shuffle_seed)
        self.register_buffer("relation_bank", relation_bank, persistent=False)
        self.register_buffer("shuffled_relation_bank", shuffled, persistent=False)

    @staticmethod
    def _build_relation_bank() -> torch.Tensor:
        square = torch.arange(64)
        rank = torch.div(square, 8, rounding_mode="floor")
        file = square.remainder(8)
        rank_i = rank.view(64, 1)
        rank_j = rank.view(1, 64)
        file_i = file.view(64, 1)
        file_j = file.view(1, 64)
        delta_rank = (rank_i - rank_j).abs()
        delta_file = (file_i - file_j).abs()
        manhattan = delta_rank + delta_file
        chebyshev = torch.maximum(delta_rank, delta_file)
        square_color_i = (rank_i + file_i).remainder(2)
        square_color_j = (rank_j + file_j).remainder(2)
        center_i = torch.maximum((rank_i.float() - 3.5).abs(), (file_i.float() - 3.5).abs()).floor()
        center_j = torch.maximum((rank_j.float() - 3.5).abs(), (file_j.float() - 3.5).abs()).floor()
        edge_i = torch.minimum(torch.minimum(rank_i, 7 - rank_i), torch.minimum(file_i, 7 - file_i))
        edge_j = torch.minimum(torch.minimum(rank_j, 7 - rank_j), torch.minimum(file_j, 7 - file_j))
        relations = [
            square.view(64, 1) == square.view(1, 64),
            rank_i == rank_j,
            file_i == file_j,
            (rank_i - file_i) == (rank_j - file_j),
            (rank_i + file_i) == (rank_j + file_j),
            square_color_i == square_color_j,
            square_color_i != square_color_j,
            ((delta_rank == 1) & (delta_file == 2)) | ((delta_rank == 2) & (delta_file == 1)),
            (chebyshev == 1),
            manhattan == 1,
            manhattan == 2,
            manhattan == 3,
            chebyshev == 1,
            chebyshev == 2,
            center_i == center_j,
            edge_i == edge_j,
            rank_j > rank_i,
            file_j > file_i,
        ]
        return torch.stack([relation.to(dtype=torch.float32) for relation in relations], dim=0)

    @staticmethod
    def _build_shuffled_bank(relation_bank: torch.Tensor, shuffle_seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(int(shuffle_seed))
        eye = torch.eye(64, dtype=torch.bool)
        upper = torch.triu(torch.ones(64, 64, dtype=torch.bool), diagonal=1)
        off_diagonal = ~eye
        shuffled = []
        for relation in relation_bank.to(dtype=torch.bool):
            out = torch.zeros_like(relation)
            diag_count = int((relation & eye).sum().item())
            if diag_count:
                diag_choice = torch.randperm(64, generator=generator)[:diag_count]
                out[diag_choice, diag_choice] = True
            if torch.equal(relation, relation.T):
                source = upper.nonzero(as_tuple=False)
                pair_count = int((relation & upper).sum().item())
                if pair_count:
                    chosen = source[torch.randperm(source.shape[0], generator=generator)[:pair_count]]
                    out[chosen[:, 0], chosen[:, 1]] = True
                    out[chosen[:, 1], chosen[:, 0]] = True
            else:
                source = off_diagonal.nonzero(as_tuple=False)
                pair_count = int((relation & off_diagonal).sum().item())
                if pair_count:
                    chosen = source[torch.randperm(source.shape[0], generator=generator)[:pair_count]]
                    out[chosen[:, 0], chosen[:, 1]] = True
            shuffled.append(out.to(dtype=torch.float32))
        return torch.stack(shuffled, dim=0)

    def forward(self, *, shuffled: bool = False, dtype: torch.dtype | None = None) -> torch.Tensor:
        bank = self.shuffled_relation_bank if shuffled else self.relation_bank
        return bank if dtype is None else bank.to(dtype=dtype)


@dataclass(frozen=True)
class BoardPairMasks:
    occupied: torch.Tensor
    empty: torch.Tensor
    king_zone: torch.Tensor


class BoardPairMaskBuilder(nn.Module):
    def forward(self, x: torch.Tensor) -> BoardPairMasks:
        batch = x.shape[0]
        if x.shape[1] >= 12:
            occupied = x[:, :12].clamp(0.0, 1.0).sum(dim=1).clamp(0.0, 1.0).flatten(1)
            kings = (x[:, 5:6].clamp(0.0, 1.0) + x[:, 11:12].clamp(0.0, 1.0)).clamp(0.0, 1.0)
            king_zone = F.max_pool2d(kings, kernel_size=3, stride=1, padding=1).flatten(1).clamp(0.0, 1.0)
        else:
            occupied = x.new_zeros(batch, 64)
            king_zone = x.new_zeros(batch, 64)
        return BoardPairMasks(occupied=occupied, empty=1.0 - occupied, king_zone=king_zone)


class SquareTokenProjector(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        model_dim: int = 128,
        norm: str = "rmsnorm",
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        square = torch.arange(64, dtype=torch.float32)
        rank = torch.div(square, 8, rounding_mode="floor")
        file = square.remainder(8)
        rank01 = rank / 7.0
        file01 = file / 7.0
        center_distance = torch.maximum((rank - 3.5).abs(), (file - 3.5).abs()) / 3.5
        edge_distance = torch.minimum(torch.minimum(rank, 7.0 - rank), torch.minimum(file, 7.0 - file)) / 3.5
        square_color = ((rank + file).remainder(2.0) * 2.0) - 1.0
        static_coords = torch.stack([rank01, file01, center_distance, edge_distance, square_color], dim=1)
        self.register_buffer("static_coords", static_coords, persistent=False)
        self.register_buffer("rank01", rank01.view(1, 64, 1), persistent=False)
        self.proj = nn.Linear(input_channels + 6, model_dim)
        self.norm = _make_norm(model_dim, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]
        tokens = x.flatten(2).transpose(1, 2)
        side = x[:, 12].mean(dim=(1, 2)).view(batch, 1, 1) if x.shape[1] > 12 else x.new_ones(batch, 1, 1)
        rank01 = self.rank01.to(device=x.device, dtype=x.dtype)
        side_relative_rank = side * rank01 + (1.0 - side) * (1.0 - rank01)
        static = self.static_coords.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(batch, -1, -1)
        coords = torch.cat([static, side_relative_rank.to(device=x.device, dtype=x.dtype)], dim=-1)
        return self.norm(self.proj(torch.cat([tokens, coords], dim=-1)))


class PairFieldBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        heads: int,
        head_dim: int,
        pair_rank: int,
        relation_count: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        norm: str = "rmsnorm",
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.pair_rank = int(pair_rank)
        if self.heads < 1 or self.head_dim < 1 or self.pair_rank < 1:
            raise ValueError("heads, head_dim, and pair_rank must be positive")
        self.qkv = nn.Linear(model_dim, 3 * heads * head_dim)
        self.rank_qk = nn.Linear(model_dim, 2 * heads * pair_rank)
        self.rank_weight = nn.Parameter(torch.empty(heads, pair_rank))
        self.relation_weight = nn.Parameter(torch.empty(heads, relation_count))
        self.message_proj = nn.Linear(heads * head_dim, model_dim)
        self.message_norm = _make_norm(model_dim, norm)
        self.ffn_norm = _make_norm(model_dim, norm)
        self.ffn = TokenFeedForward(model_dim, model_dim * int(ffn_multiplier), activation=activation, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.qk_scale = 1.0 / sqrt(float(head_dim))
        self.rank_scale = 1.0 / sqrt(float(pair_rank))
        self.weight_scale = 1.0 / sqrt(64.0)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.rank_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.relation_weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        relation_bank: torch.Tensor,
        *,
        no_pair_update: bool = False,
        softmax_attention: bool = False,
        active_heads: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, square_count, _ = tokens.shape
        qkv = self.qkv(tokens).view(batch, square_count, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        pair = torch.einsum("bhid,bhjd->bhij", q, k) * self.qk_scale

        rank_qk = self.rank_qk(tokens).view(batch, square_count, 2, self.heads, self.pair_rank)
        rank_q, rank_k = rank_qk.permute(2, 0, 3, 1, 4)
        pair = pair + torch.einsum("bhir,bhjr,hr->bhij", rank_q, rank_k, self.rank_weight) * self.rank_scale

        relation_bias = torch.einsum("hk,kij->hij", self.relation_weight, relation_bank)
        pair = pair + relation_bias.unsqueeze(0)

        pair_for_update = pair
        if active_heads is not None and active_heads < self.heads:
            head_mask = pair.new_zeros(1, self.heads, 1, 1)
            head_mask[:, : max(1, active_heads)] = 1.0
            pair_for_update = pair * head_mask

        if softmax_attention:
            weights = torch.softmax(pair_for_update, dim=-1)
        else:
            weights = torch.tanh(pair_for_update) * self.weight_scale
        message = torch.einsum("bhij,bhjd->bhid", weights, v)
        if active_heads is not None and active_heads < self.heads:
            value_mask = message.new_zeros(1, self.heads, 1, 1)
            value_mask[:, : max(1, active_heads)] = 1.0
            message = message * value_mask

        if no_pair_update:
            updated = tokens
        else:
            message = message.transpose(1, 2).reshape(batch, square_count, self.heads * self.head_dim)
            updated = self.message_norm(tokens + self.dropout(self.message_proj(message)))
        return self.ffn_norm(updated + self.dropout(self.ffn(updated))), pair_for_update


@dataclass(frozen=True)
class PairSummary:
    vector: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class PairEnergyReadout(nn.Module):
    def __init__(self, relation_count: int) -> None:
        super().__init__()
        self.relation_count = int(relation_count)
        self.summary_dim = 10 + self.relation_count + 3

    def forward(self, pair: torch.Tensor, relation_bank: torch.Tensor, masks: BoardPairMasks) -> PairSummary:
        energy = pair.square()
        abs_pair = pair.abs()
        head_energy = energy.mean(dim=(2, 3))
        row_energy = energy.mean(dim=3)
        col_energy = energy.mean(dim=2)
        flat_abs = abs_pair.flatten(1)
        prob = flat_abs / flat_abs.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        entropy = -(prob * prob.clamp_min(1.0e-6).log()).sum(dim=1) / torch.log(
            pair.new_tensor(float(flat_abs.shape[1]))
        )

        relation_weights = relation_bank / relation_bank.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        relation_energy = torch.einsum("bhij,kij->bhk", energy, relation_weights).mean(dim=1)

        occupied = masks.occupied
        empty = masks.empty
        king_zone = masks.king_zone
        occupied_to_occupied = occupied.unsqueeze(2) * occupied.unsqueeze(1)
        occupied_to_empty = occupied.unsqueeze(2) * empty.unsqueeze(1)
        king_zone_pair = torch.maximum(king_zone.unsqueeze(2), king_zone.unsqueeze(1))
        occ_occ_energy = _pair_weighted_mean(energy, occupied_to_occupied)
        occ_empty_energy = _pair_weighted_mean(energy, occupied_to_empty)
        king_pair_energy = _pair_weighted_mean(energy, king_zone_pair)

        base = torch.cat(
            [
                abs_pair.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1),
                energy.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1),
                abs_pair.amax(dim=(1, 2, 3), keepdim=False).unsqueeze(1),
                row_energy.mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                row_energy.amax(dim=(1, 2), keepdim=False).unsqueeze(1),
                col_energy.mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                col_energy.amax(dim=(1, 2), keepdim=False).unsqueeze(1),
                entropy.unsqueeze(1),
                head_energy.mean(dim=1, keepdim=True),
                head_energy.std(dim=1, unbiased=False, keepdim=True),
            ],
            dim=1,
        )
        vector = torch.cat(
            [base, relation_energy, occ_occ_energy.unsqueeze(1), occ_empty_energy.unsqueeze(1), king_pair_energy.unsqueeze(1)],
            dim=1,
        )
        diagnostics = {
            "pair_field_mean_abs": base[:, 0],
            "pair_field_mean_square": base[:, 1],
            "pair_field_max_abs": base[:, 2],
            "row_energy_mean": base[:, 3],
            "row_energy_max": base[:, 4],
            "col_energy_mean": base[:, 5],
            "col_energy_max": base[:, 6],
            "pair_field_entropy_proxy": entropy,
            "head_energy_mean": base[:, 8],
            "per_head_energy_specialization": base[:, 9],
            "occupied_to_occupied_pair_energy": occ_occ_energy,
            "occupied_to_empty_pair_energy": occ_empty_energy,
            "king_zone_pair_energy": king_pair_energy,
            "same_rank_pair_energy": relation_energy[:, RELATION_NAMES.index("same_rank")],
            "same_file_pair_energy": relation_energy[:, RELATION_NAMES.index("same_file")],
            "diagonal_pair_energy": 0.5
            * (
                relation_energy[:, RELATION_NAMES.index("same_diag")]
                + relation_energy[:, RELATION_NAMES.index("same_anti_diag")]
            ),
            "knight_offset_pair_energy": relation_energy[:, RELATION_NAMES.index("knight_offset")],
        }
        return PairSummary(vector=vector, diagnostics=diagnostics)


class TensorCoreSquarePairFieldNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        model_dim: int = 128,
        heads: int = 8,
        head_dim: int = 32,
        layers: int = 3,
        pair_rank: int = 16,
        hidden_dim: int = 128,
        classifier_hidden: int | None = None,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
        norm: str = "rmsnorm",
        activation: str = "gelu",
        ablation: str = "none",
        shuffle_seed: int = 72072,
    ) -> None:
        super().__init__()
        if ablation not in PAIR_ABLATIONS:
            raise ValueError(f"Unsupported ablation {ablation!r}; expected one of {sorted(PAIR_ABLATIONS)}")
        if layers < 1:
            raise ValueError("layers must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.model_dim = int(model_dim)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.layers = int(layers)
        self.pair_rank = int(pair_rank)
        self.ablation = ablation

        self.square_projector = SquareTokenProjector(input_channels=input_channels, model_dim=model_dim, norm=norm)
        self.relations = SquarePairRelationBank(shuffle_seed=shuffle_seed)
        self.mask_builder = BoardPairMaskBuilder()
        self.blocks = nn.ModuleList(
            [
                PairFieldBlock(
                    model_dim=model_dim,
                    heads=heads,
                    head_dim=head_dim,
                    pair_rank=pair_rank,
                    relation_count=len(RELATION_NAMES),
                    ffn_multiplier=ffn_multiplier,
                    dropout=dropout,
                    norm=norm,
                    activation=activation,
                )
                for _ in range(layers)
            ]
        )
        self.pair_readout = PairEnergyReadout(relation_count=len(RELATION_NAMES))
        cnn_layers: list[nn.Module] = [
            nn.Conv2d(input_channels, model_dim, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        for _ in range(max(0, layers - 1)):
            cnn_layers.extend(
                [
                    nn.Conv2d(model_dim, model_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                ]
            )
        self.cnn_control = nn.Sequential(*cnn_layers)
        square_summary_dim = 4 * model_dim
        pair_summary_dim = layers * self.pair_readout.summary_dim
        classifier_width = int(classifier_hidden or hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(square_summary_dim + pair_summary_dim),
            nn.Linear(square_summary_dim + pair_summary_dim, classifier_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_width, self.num_classes),
        )

    @property
    def relation_bank(self) -> torch.Tensor:
        return self.relations.relation_bank

    @property
    def shuffled_relation_bank(self) -> torch.Tensor:
        return self.relations.shuffled_relation_bank

    def _square_summary(self, tokens: torch.Tensor, masks: BoardPairMasks) -> torch.Tensor:
        return torch.cat(
            [
                tokens.mean(dim=1),
                tokens.amax(dim=1),
                _masked_token_mean(tokens, masks.occupied),
                _masked_token_mean(tokens, masks.king_zone),
            ],
            dim=1,
        )

    def _zero_pair_diagnostics(self, batch: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        zero = torch.zeros(batch, device=device, dtype=dtype)
        return {
            "pair_field_mean_abs": zero,
            "pair_field_mean_square": zero,
            "pair_field_max_abs": zero,
            "row_energy_mean": zero,
            "row_energy_max": zero,
            "col_energy_mean": zero,
            "col_energy_max": zero,
            "pair_field_entropy_proxy": zero,
            "head_energy_mean": zero,
            "per_head_energy_specialization": zero,
            "occupied_to_occupied_pair_energy": zero,
            "occupied_to_empty_pair_energy": zero,
            "king_zone_pair_energy": zero,
            "same_rank_pair_energy": zero,
            "same_file_pair_energy": zero,
            "diagonal_pair_energy": zero,
            "knight_offset_pair_energy": zero,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        masks = self.mask_builder(x)
        batch = x.shape[0]
        shuffle_relations = self.ablation == "relation_bank_shuffle"
        relation_bank = self.relations(shuffled=shuffle_relations, dtype=x.dtype).to(device=x.device)
        active_heads = 2 if self.ablation == "low_head_count" else None

        if self.ablation == "cnn_only_matched":
            cnn_tokens = self.cnn_control(x).flatten(2).transpose(1, 2)
            tokens = cnn_tokens
            pair_vector = x.new_zeros(batch, self.layers * self.pair_readout.summary_dim)
            diagnostics = self._zero_pair_diagnostics(batch, x.device, x.dtype)
        else:
            tokens = self.square_projector(x)
            summaries: list[torch.Tensor] = []
            diagnostics_by_name: dict[str, list[torch.Tensor]] = {}
            for block in self.blocks:
                tokens, pair = block(
                    tokens,
                    relation_bank,
                    no_pair_update=self.ablation == "no_pair_update",
                    softmax_attention=self.ablation == "softmax_attention_control",
                    active_heads=active_heads,
                )
                pair_summary = self.pair_readout(pair, relation_bank, masks)
                summaries.append(pair_summary.vector)
                for key, value in pair_summary.diagnostics.items():
                    diagnostics_by_name.setdefault(key, []).append(value)
            pair_vector = torch.cat(summaries, dim=1)
            diagnostics = {key: torch.stack(values, dim=0).mean(dim=0) for key, values in diagnostics_by_name.items()}

        square_summary = self._square_summary(tokens, masks)
        if self.ablation == "no_pair_readout":
            pair_vector = torch.zeros_like(pair_vector)
        if self.ablation == "pair_energy_only":
            square_summary = torch.zeros_like(square_summary)

        features = torch.cat([square_summary, pair_vector], dim=1)
        logits = _format_logits(self.classifier(features), self.num_classes)
        density_delta = (
            self.shuffled_relation_bank.to(device=x.device, dtype=x.dtype).mean(dim=(1, 2))
            - self.relation_bank.to(device=x.device, dtype=x.dtype).mean(dim=(1, 2))
        ).abs().amax()
        output = {
            "logits": logits,
            "square_token_energy": tokens.square().mean(dim=(1, 2)),
            "occupied_square_count": masks.occupied.sum(dim=1),
            "relation_density_error": density_delta.expand(batch),
            "pair_readout_active": x.new_full((batch,), 0.0 if self.ablation == "no_pair_readout" else 1.0),
            "pair_update_active": x.new_full((batch,), 0.0 if self.ablation == "no_pair_update" else 1.0),
            "relation_shuffle_active": x.new_full((batch,), 1.0 if shuffle_relations else 0.0),
            **diagnostics,
        }
        return output


def build_tensor_core_square_pair_field_network_from_config(config: dict[str, Any]) -> TensorCoreSquarePairFieldNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    cfg.pop("use_batchnorm", None)
    cfg.pop("relation_count", None)
    cfg.pop("use_pair_energy_readout", None)
    cfg.pop("use_relation_energy_readout", None)
    input_channels = int(cfg.pop("input_channels", 18))
    num_classes = int(cfg.pop("num_classes", 1))
    model_dim = int(cfg.pop("model_dim", cfg.pop("channels", 128)))
    layers = int(cfg.pop("layers", cfg.pop("depth", 3)))
    hidden_dim = int(cfg.pop("hidden_dim", model_dim))
    heads = int(cfg.pop("heads", 8))
    head_dim = int(cfg.pop("head_dim", max(8, model_dim // max(1, heads))))
    pair_rank = int(cfg.pop("pair_rank", 16))
    classifier_hidden = cfg.pop("classifier_hidden", None)
    return TensorCoreSquarePairFieldNetwork(
        input_channels=input_channels,
        num_classes=num_classes,
        model_dim=model_dim,
        heads=heads,
        head_dim=head_dim,
        layers=layers,
        pair_rank=pair_rank,
        hidden_dim=hidden_dim,
        classifier_hidden=int(classifier_hidden) if classifier_hidden is not None else None,
        ffn_multiplier=int(cfg.pop("ffn_multiplier", 4)),
        dropout=float(cfg.pop("dropout", 0.1)),
        norm=str(cfg.pop("norm", "rmsnorm")),
        activation=str(cfg.pop("activation", "gelu")),
        ablation=str(cfg.pop("ablation", "none")),
        shuffle_seed=int(cfg.pop("shuffle_seed", 72072)),
    )
