"""Ray State-Space Scan Network (idea i125)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


_LINE_TYPE_NAMES = ("rank", "file", "diagonal", "anti_diagonal")


@dataclass(frozen=True)
class LineBank:
    forward_positions: torch.Tensor
    backward_positions: torch.Tensor
    mask: torch.Tensor
    line_types: torch.Tensor


def _build_line_bank() -> LineBank:
    lines: list[tuple[int, list[int]]] = []

    for rank in range(8):
        lines.append((0, [rank * 8 + file for file in range(8)]))
    for file in range(8):
        lines.append((1, [rank * 8 + file for rank in range(8)]))
    for delta in range(-7, 8):
        line = [rank * 8 + (rank + delta) for rank in range(8) if 0 <= rank + delta < 8]
        lines.append((2, line))
    for total in range(15):
        line = [rank * 8 + (total - rank) for rank in range(8) if 0 <= total - rank < 8]
        lines.append((3, line))

    max_len = 8
    forward: list[list[int]] = []
    backward: list[list[int]] = []
    masks: list[list[bool]] = []
    types: list[int] = []
    for line_type, line in lines:
        pad = [-1] * (max_len - len(line))
        forward.append(line + pad)
        backward.append(list(reversed(line)) + pad)
        masks.append([True] * len(line) + [False] * (max_len - len(line)))
        types.append(line_type)

    return LineBank(
        forward_positions=torch.tensor(forward, dtype=torch.long),
        backward_positions=torch.tensor(backward, dtype=torch.long),
        mask=torch.tensor(masks, dtype=torch.bool),
        line_types=torch.tensor(types, dtype=torch.long),
    )


class ConvNormGelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(x)))


class ResidualBoardBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm1 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm2 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class BoardAndSquareEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: int,
        square_dim: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.stem = ConvNormGelu(input_channels, channels, use_batchnorm=use_batchnorm)
        self.blocks = nn.Sequential(
            *[ResidualBoardBlock(channels, dropout=dropout, use_batchnorm=use_batchnorm) for _ in range(depth)]
        )
        self.square_projection = nn.Conv2d(channels, square_dim, kernel_size=1)
        self.output_channels = channels
        self.square_dim = square_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        board = self.blocks(self.stem(require_board_tensor(x, self.spec)))
        square_map = self.square_projection(board)
        square_tokens = square_map.flatten(2).transpose(1, 2)
        return board, square_tokens


class LineSequenceExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        bank = _build_line_bank()
        self.register_buffer("forward_positions", bank.forward_positions, persistent=False)
        self.register_buffer("backward_positions", bank.backward_positions, persistent=False)
        self.register_buffer("mask", bank.mask, persistent=False)
        self.register_buffer("line_types", bank.line_types, persistent=False)

    def _gather(self, square_tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        flat_positions = positions.clamp_min(0).reshape(-1)
        gathered = square_tokens[:, flat_positions]
        return gathered.view(square_tokens.shape[0], positions.shape[0], positions.shape[1], square_tokens.shape[-1])

    def forward(self, square_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._gather(square_tokens, self.forward_positions),
            self._gather(square_tokens, self.backward_positions),
            self.mask,
            self.line_types,
        )


class LineTypeStateSpaceScan(nn.Module):
    def __init__(self, input_dim: int, state_dim: int, output_dim: int | None = None, line_types: int = 4) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        output_dim = input_dim if output_dim is None else output_dim
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.line_types = line_types

        transition = torch.empty(line_types, state_dim, state_dim)
        nn.init.normal_(transition, mean=0.0, std=0.015)
        for line_type in range(line_types):
            transition[line_type].diagonal().fill_(0.7)
        self.transition = nn.Parameter(transition)
        self.input_weight = nn.Parameter(torch.empty(line_types, state_dim, input_dim))
        self.output_weight = nn.Parameter(torch.empty(line_types, output_dim, state_dim))
        skip = torch.zeros(line_types, output_dim, input_dim)
        for line_type in range(line_types):
            diag = min(output_dim, input_dim)
            skip[line_type, torch.arange(diag), torch.arange(diag)] = 1.0
        self.skip_weight = nn.Parameter(skip)
        self.hidden_bias = nn.Parameter(torch.zeros(line_types, state_dim))
        self.output_bias = nn.Parameter(torch.zeros(line_types, output_dim))
        self.output_norm = nn.LayerNorm(output_dim)
        nn.init.xavier_uniform_(self.input_weight)
        nn.init.xavier_uniform_(self.output_weight)

    def _scan_one_type(self, sequence: torch.Tensor, mask: torch.Tensor, line_type: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch, line_count, steps, _ = sequence.shape
        hidden = sequence.new_zeros(batch, line_count, self.state_dim)
        outputs: list[torch.Tensor] = []
        transition = 0.75 * torch.tanh(self.transition[line_type]).to(dtype=sequence.dtype)
        input_weight = self.input_weight[line_type].to(dtype=sequence.dtype)
        output_weight = self.output_weight[line_type].to(dtype=sequence.dtype)
        skip_weight = self.skip_weight[line_type].to(dtype=sequence.dtype)
        hidden_bias = self.hidden_bias[line_type].to(dtype=sequence.dtype)
        output_bias = self.output_bias[line_type].to(dtype=sequence.dtype)

        for step in range(steps):
            token = sequence[:, :, step]
            valid = mask[:, step].view(1, line_count, 1)
            recurrent = torch.einsum("bls,rs->blr", hidden, transition)
            drive = torch.einsum("bld,sd->bls", token, input_weight) + hidden_bias
            next_hidden = recurrent + drive
            hidden = torch.where(valid, next_hidden, hidden)
            output = torch.einsum("bls,ds->bld", hidden, output_weight)
            output = output + torch.einsum("bld,od->blo", token, skip_weight) + output_bias
            outputs.append(torch.where(valid, output, torch.zeros_like(output)))

        stacked = torch.stack(outputs, dim=2).reshape(batch * line_count, steps, self.output_dim)
        stacked = self.output_norm(stacked).view(batch, line_count, steps, self.output_dim)
        return stacked, hidden

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor, line_types: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, line_count, steps, _ = sequence.shape
        outputs = sequence.new_zeros(batch, line_count, steps, self.output_dim)
        final_hidden = sequence.new_zeros(batch, line_count, self.state_dim)
        for line_type in range(self.line_types):
            selected = line_types == line_type
            if not bool(selected.any()):
                continue
            type_output, type_hidden = self._scan_one_type(sequence[:, selected], mask[selected], line_type)
            outputs[:, selected] = type_output
            final_hidden[:, selected] = type_hidden
        return outputs, final_hidden


class RayStateSpaceScanNetwork(nn.Module):
    """Line-memory model using shared state-space scans over chess rays."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        square_dim: int = 48,
        state_dim: int = 32,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        encoding_adapter: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("RayStateSpaceScanNetwork supports the puzzle_binary one-logit contract")
        if encoding_adapter != SIMPLE_18 or input_channels != 18:
            raise ValueError("RayStateSpaceScanNetwork currently supports simple_18 with 18 input channels")
        if square_dim < 1:
            raise ValueError("square_dim must be positive")
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.encoder = BoardAndSquareEncoder(
            input_channels=input_channels,
            channels=channels,
            square_dim=square_dim,
            depth=depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.extract_lines = LineSequenceExtractor()
        self.scan = LineTypeStateSpaceScan(input_dim=square_dim, state_dim=state_dim, output_dim=square_dim)
        self.line_response = nn.Linear(square_dim * 2, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        board_feature_dim = channels * 2
        line_feature_dim = square_dim * 4 + state_dim * 2 + 10
        self.classifier = nn.Sequential(
            nn.Linear(board_feature_dim + line_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _line_type_stats(
        self,
        tokens: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor,
        line_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        type_responses: list[torch.Tensor] = []
        type_energies: list[torch.Tensor] = []
        token_norm = tokens.norm(dim=-1)
        for line_type in range(len(_LINE_TYPE_NAMES)):
            selected = line_types == line_type
            type_mask = mask[selected].to(dtype=tokens.dtype).unsqueeze(0)
            denom = type_mask.sum(dim=(1, 2)).clamp_min(1.0)
            type_responses.append((response[:, selected] * type_mask).sum(dim=(1, 2)) / denom)
            type_energies.append((token_norm[:, selected] * type_mask).sum(dim=(1, 2)) / denom)
        return torch.stack(type_responses, dim=1), torch.stack(type_energies, dim=1)

    def _king_line_response(self, x: torch.Tensor, response: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = self.extract_lines.forward_positions.clamp_min(0)
        king_occ = (x[:, 5] + x[:, 11]).clamp(0.0, 1.0).flatten(1)
        king_tokens = king_occ[:, positions.reshape(-1)].view(x.shape[0], positions.shape[0], positions.shape[1])
        line_has_king = ((king_tokens * mask.to(dtype=x.dtype).unsqueeze(0)).sum(dim=-1) > 0.5).to(dtype=x.dtype)
        token_mask = line_has_king.unsqueeze(-1) * mask.to(dtype=x.dtype).unsqueeze(0)
        denom = token_mask.sum(dim=(1, 2)).clamp_min(1.0)
        return (response * token_mask).sum(dim=(1, 2)) / denom

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board_map, square_tokens = self.encoder(x)
        forward_seq, backward_seq, line_mask, line_types = self.extract_lines(square_tokens)
        forward_out, forward_hidden = self.scan(forward_seq, line_mask, line_types)
        backward_out, backward_hidden = self.scan(backward_seq, line_mask, line_types)
        line_tokens = torch.cat([forward_out, backward_out], dim=-1)
        token_mask = line_mask.to(dtype=line_tokens.dtype).unsqueeze(0).unsqueeze(-1)
        valid_count = token_mask.sum(dim=(1, 2)).clamp_min(1.0)
        line_mean = (line_tokens * token_mask).sum(dim=(1, 2)) / valid_count
        line_max = line_tokens.masked_fill(~line_mask.view(1, *line_mask.shape, 1), -1e6).amax(dim=(1, 2))
        endpoint_mean = torch.cat([forward_hidden, backward_hidden], dim=-1).mean(dim=1)

        response = self.line_response(line_tokens).squeeze(-1)
        masked_response = response.masked_fill(~line_mask.view(1, *line_mask.shape), -1e6)
        topk_response = masked_response.flatten(1).topk(k=4, dim=1).values.mean(dim=1)
        king_response = self._king_line_response(x, response, line_mask)
        type_responses, type_energies = self._line_type_stats(line_tokens, response, line_mask, line_types)

        board_mean = board_map.mean(dim=(2, 3))
        board_max = board_map.amax(dim=(2, 3))
        board_features = torch.cat([board_mean, board_max], dim=1)
        line_features = torch.cat(
            [
                line_mean,
                line_max,
                endpoint_mean,
                type_responses,
                type_energies,
                topk_response.unsqueeze(1),
                king_response.unsqueeze(1),
            ],
            dim=1,
        )
        logits = self.classifier(self.dropout(torch.cat([board_features, line_features], dim=1))).squeeze(-1)

        return {
            "logits": logits,
            "line_state_energy": (line_tokens.norm(dim=-1) * line_mask.to(dtype=line_tokens.dtype).unsqueeze(0)).sum(
                dim=(1, 2)
            )
            / line_mask.to(dtype=line_tokens.dtype).sum().clamp_min(1.0),
            "rank_scan_energy": type_energies[:, 0],
            "file_scan_energy": type_energies[:, 1],
            "diagonal_scan_energy": type_energies[:, 2],
            "anti_diagonal_scan_energy": type_energies[:, 3],
            "rank_scan_response": type_responses[:, 0],
            "file_scan_response": type_responses[:, 1],
            "diagonal_scan_response": type_responses[:, 2],
            "anti_diagonal_scan_response": type_responses[:, 3],
            "endpoint_state_norm": endpoint_mean.norm(dim=1),
            "topk_line_response": topk_response,
            "king_line_response": king_response,
        }


def build_ray_state_space_scan_network_from_config(config: dict[str, Any]) -> RayStateSpaceScanNetwork:
    hidden_dim = int(config.get("hidden_dim", 96))
    return RayStateSpaceScanNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        square_dim=int(config.get("square_dim", max(16, hidden_dim // 2))),
        state_dim=int(config.get("state_dim", max(8, hidden_dim // 3))),
        hidden_dim=hidden_dim,
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        encoding_adapter=str(config.get("encoding_adapter", SIMPLE_18)),
    )
