"""Global Scratchpad BoardNet for idea i163."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


@dataclass(frozen=True)
class GlobalScratchpadConfig:
    input_channels: int = 18
    num_classes: int = 1
    width: int = 64
    memory_slots: int = 4
    memory_dim: int = 64
    scratchpad_steps: int = 4
    hidden_dim: int = 96
    dropout: float = 0.1
    use_batchnorm: bool = True
    use_coordinate_planes: bool = True
    ablation: str = "none"


class CoordinatePlaneAppender(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        coords = torch.linspace(-1.0, 1.0, 8)
        rank = coords.view(1, 1, 8, 1).expand(1, 1, 8, 8)
        file = coords.view(1, 1, 1, 8).expand(1, 1, 8, 8)
        self.register_buffer("coordinate_planes", torch.cat([rank, file], dim=1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        planes = self.coordinate_planes.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1, -1)
        return torch.cat([x, planes], dim=1)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, width: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(width, width, dropout=dropout, use_batchnorm=use_batchnorm),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
        )
        self.norm = nn.BatchNorm2d(width) if use_batchnorm else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(x + self.block(x)))


class GlobalScratchpadBoardNet(nn.Module):
    """CNN with recurrent global memory slots and FiLM board broadcasts."""

    VALID_ABLATIONS = {
        "none",
        "no_scratchpad",
        "one_step",
        "no_broadcast",
        "random_memory",
        "single_slot",
    }

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        width: int = 64,
        memory_slots: int = 4,
        memory_dim: int = 64,
        scratchpad_steps: int = 4,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        use_coordinate_planes: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown GlobalScratchpadBoardNet ablation: {ablation}")
        if width < 1 or memory_slots < 1 or memory_dim < 1 or scratchpad_steps < 1:
            raise ValueError("width, memory_slots, memory_dim, and scratchpad_steps must be positive")
        self.config = GlobalScratchpadConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            width=width,
            memory_slots=memory_slots,
            memory_dim=memory_dim,
            scratchpad_steps=scratchpad_steps,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            use_coordinate_planes=use_coordinate_planes,
            ablation=ablation,
        )
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.ablation = ablation
        self.width = int(width)
        self.memory_dim = int(memory_dim)
        self.memory_slots = 1 if ablation == "single_slot" else int(memory_slots)
        self.scratchpad_steps = 1 if ablation == "one_step" else int(scratchpad_steps)
        self.coordinate_appender = CoordinatePlaneAppender() if use_coordinate_planes else nn.Identity()
        stem_channels = input_channels + (2 if use_coordinate_planes else 0)
        self.stem = nn.Sequential(
            ConvNormAct(stem_channels, width, dropout=dropout, use_batchnorm=use_batchnorm),
            ResidualConvBlock(width, dropout=dropout, use_batchnorm=use_batchnorm),
        )

        memory_seed = torch.empty(self.memory_slots, memory_dim)
        nn.init.normal_(memory_seed, mean=0.0, std=0.02)
        if ablation == "random_memory":
            self.register_buffer("learned_memory", memory_seed, persistent=True)
        else:
            self.learned_memory = nn.Parameter(memory_seed)

        board_pool_dim = width * 4
        self.memory_init = nn.Sequential(
            nn.LayerNorm(board_pool_dim),
            nn.Linear(board_pool_dim, self.memory_slots * memory_dim),
            nn.Tanh(),
        )
        self.slot_summary = nn.Sequential(
            nn.LayerNorm(board_pool_dim),
            nn.Linear(board_pool_dim, self.memory_slots * memory_dim),
            nn.GELU(),
        )
        self.memory_update = nn.GRUCell(memory_dim, memory_dim)
        self.memory_to_film = nn.Sequential(
            nn.LayerNorm(memory_dim * 2),
            nn.Linear(memory_dim * 2, width * 2),
        )
        self.step_blocks = nn.ModuleList(
            [ResidualConvBlock(width, dropout=dropout, use_batchnorm=use_batchnorm) for _ in range(self.scratchpad_steps)]
        )
        head_dim = width * 3 + memory_dim * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, max(32, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(max(32, hidden_dim // 2), num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        h = self.stem(self.coordinate_appender(board))
        memory = self._initial_memory(h)
        initial_memory = memory
        memory_norms: list[torch.Tensor] = []
        board_changes: list[torch.Tensor] = []
        memory_updates: list[torch.Tensor] = []

        if self.ablation == "no_scratchpad":
            memory = torch.zeros_like(memory)
            initial_memory = memory
            for block in self.step_blocks:
                previous = h
                h = previous + 0.25 * (block(previous) - previous)
                board_changes.append((h - previous).square().mean(dim=(1, 2, 3)).sqrt())
                memory_norms.append(memory.norm(dim=2))
                memory_updates.append(memory.new_zeros(memory.shape[0]))
        else:
            for block in self.step_blocks:
                previous_memory = memory
                summary = self._slot_summaries(h)
                memory = self.memory_update(
                    summary.reshape(-1, self.memory_dim),
                    memory.reshape(-1, self.memory_dim),
                ).view_as(memory)
                memory_updates.append((memory - previous_memory).square().mean(dim=(1, 2)).sqrt())
                previous_h = h
                if self.ablation == "no_broadcast":
                    h = previous_h + 0.25 * (block(previous_h) - previous_h)
                else:
                    gamma, beta = self._film(memory)
                    modulated = gamma.unsqueeze(-1).unsqueeze(-1) * previous_h + beta.unsqueeze(-1).unsqueeze(-1)
                    h = previous_h + 0.25 * (block(modulated) - previous_h)
                board_changes.append((h - previous_h).square().mean(dim=(1, 2, 3)).sqrt())
                memory_norms.append(memory.norm(dim=2))

        pooled = self._board_pool(h)
        memory_mean = memory.mean(dim=1)
        memory_max = memory.amax(dim=1)
        features = torch.cat([pooled, memory_mean, memory_max], dim=1)
        raw_logits = self.classifier(features)
        logits = _format_logits(raw_logits, self.num_classes)
        memory_norm_by_step = torch.stack(memory_norms, dim=1)
        board_change_by_step = torch.stack(board_changes, dim=1)
        memory_update_by_step = torch.stack(memory_updates, dim=1)
        output = {
            "logits": logits,
            "memory_slots": memory,
            "initial_memory_slots": initial_memory,
            "memory_slot_norm_by_step": memory_norm_by_step,
            "memory_update_norm_by_step": memory_update_by_step,
            "board_activation_change_by_step": board_change_by_step,
            "final_memory_norm": memory.norm(dim=2).mean(dim=1),
            "initial_memory_norm": initial_memory.norm(dim=2).mean(dim=1),
            "memory_slot_similarity": self._slot_similarity(memory),
            "board_feature_energy": h.square().mean(dim=(1, 2, 3)),
            "board_pool_energy": pooled.square().mean(dim=1),
            "scratchpad_steps_used": logits.new_full(logits.shape, float(self.scratchpad_steps)),
            "memory_slot_count": logits.new_full(logits.shape, float(self.memory_slots)),
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output

    def _board_pool(self, h: torch.Tensor) -> torch.Tensor:
        mean = h.mean(dim=(2, 3))
        max_values = h.amax(dim=(2, 3))
        std = h.flatten(2).std(dim=2, unbiased=False)
        return torch.cat([mean, max_values, std], dim=1)

    def _initial_memory(self, h: torch.Tensor) -> torch.Tensor:
        pooled = self._summary_pool(h)
        delta = self.memory_init(pooled).view(h.shape[0], self.memory_slots, self.memory_dim)
        return self.learned_memory.unsqueeze(0).to(dtype=h.dtype, device=h.device) + delta

    def _slot_summaries(self, h: torch.Tensor) -> torch.Tensor:
        pooled = self._summary_pool(h)
        return self.slot_summary(pooled).view(h.shape[0], self.memory_slots, self.memory_dim)

    def _summary_pool(self, h: torch.Tensor) -> torch.Tensor:
        coords = torch.linspace(-1.0, 1.0, 8, device=h.device, dtype=h.dtype)
        rank = coords.view(1, 1, 8, 1)
        file = coords.view(1, 1, 1, 8)
        mean = h.mean(dim=(2, 3))
        max_values = h.amax(dim=(2, 3))
        rank_mean = (h * rank).mean(dim=(2, 3))
        file_mean = (h * file).mean(dim=(2, 3))
        return torch.cat([mean, max_values, rank_mean, file_mean], dim=1)

    def _film(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        memory_mean = memory.mean(dim=1)
        memory_max = memory.amax(dim=1)
        film = self.memory_to_film(torch.cat([memory_mean, memory_max], dim=1))
        gamma_raw, beta = film.chunk(2, dim=1)
        gamma = 1.0 + 0.25 * torch.tanh(gamma_raw)
        beta = 0.25 * torch.tanh(beta)
        return gamma, beta

    def _slot_similarity(self, memory: torch.Tensor) -> torch.Tensor:
        if memory.shape[1] < 2:
            return memory.new_zeros(memory.shape[0])
        normalized = F.normalize(memory, dim=2)
        similarity = torch.bmm(normalized, normalized.transpose(1, 2))
        slots = similarity.shape[1]
        off_diag = similarity[:, ~torch.eye(slots, device=memory.device, dtype=torch.bool)].view(memory.shape[0], -1)
        return off_diag.mean(dim=1)


def build_global_scratchpad_boardnet_from_config(config: dict[str, Any]) -> GlobalScratchpadBoardNet:
    width = int(config.get("width", config.get("channels", 64)))
    return GlobalScratchpadBoardNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        width=width,
        memory_slots=int(config.get("memory_slots", 4)),
        memory_dim=int(config.get("memory_dim", width)),
        scratchpad_steps=int(config.get("scratchpad_steps", config.get("depth", 4))),
        hidden_dim=int(config.get("hidden_dim", 96)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        use_coordinate_planes=bool(config.get("use_coordinate_planes", True)),
        ablation=str(config.get("ablation", "none")),
    )
