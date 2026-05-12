"""Schur-Ray line algebra network for idea i068."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


LINE_COUNT = 46
PER_HEAD_FEATURES = 12
MATERIAL_SUMMARY_DIM = 20


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def build_square_line_indices(randomized: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns square-to-line memberships, line type ids, and per-line lengths."""
    memberships: list[list[int]] = []
    for rank in range(8):
        for file in range(8):
            diag = rank - file + 7
            anti = rank + file
            memberships.append([rank, 8 + file, 16 + diag, 31 + anti])
    square_lines = torch.tensor(memberships, dtype=torch.long)
    if randomized:
        generator = torch.Generator().manual_seed(68068)
        columns = []
        for slot in range(4):
            perm = torch.randperm(square_lines.shape[0], generator=generator)
            columns.append(square_lines[perm, slot])
        square_lines = torch.stack(columns, dim=1)

    line_type = torch.cat(
        [
            torch.zeros(8, dtype=torch.long),
            torch.ones(8, dtype=torch.long),
            torch.full((15,), 2, dtype=torch.long),
            torch.full((15,), 3, dtype=torch.long),
        ]
    )
    line_lengths = torch.zeros(LINE_COUNT, dtype=torch.float32)
    for slot in range(4):
        line_lengths.index_add_(0, square_lines[:, slot], torch.ones(64, dtype=torch.float32))
    return square_lines, line_type, line_lengths


def _king_zone_matrix() -> torch.Tensor:
    matrix = torch.zeros(64, 64, dtype=torch.float32)
    for source in range(64):
        rank = source // 8
        file = source % 8
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                rr = rank + dr
                ff = file + df
                if 0 <= rr < 8 and 0 <= ff < 8:
                    matrix[source, rr * 8 + ff] = 1.0
    return matrix


def _material_summary(x: torch.Tensor) -> torch.Tensor:
    piece_planes = x[:, :12].clamp(0.0, 1.0)
    white_counts = piece_planes[:, :6].sum(dim=(2, 3))
    black_counts = piece_planes[:, 6:12].sum(dim=(2, 3))
    white_to_move = x[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0).view(-1, 1)
    own_counts = white_to_move * white_counts + (1.0 - white_to_move) * black_counts
    opp_counts = white_to_move * black_counts + (1.0 - white_to_move) * white_counts
    count_delta = own_counts - opp_counts
    values = x.new_tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])
    own_material = (own_counts * values).sum(dim=1, keepdim=True)
    opp_material = (opp_counts * values).sum(dim=1, keepdim=True)
    total_count = (own_counts + opp_counts).sum(dim=1, keepdim=True)
    material_balance = (own_material - opp_material) / 39.0
    return torch.cat(
        [
            own_counts / 8.0,
            opp_counts / 8.0,
            count_delta / 8.0,
            total_count / 32.0,
            material_balance,
        ],
        dim=1,
    )


@dataclass(frozen=True)
class SchurRayBatch:
    features: torch.Tensor
    z: torch.Tensor
    b: torch.Tensor
    correction: torch.Tensor
    line_correction_norm: torch.Tensor
    schur_logdet: torch.Tensor
    schur_trace: torch.Tensor
    data_energy: torch.Tensor
    line_energy: torch.Tensor
    king_zone_energy: torch.Tensor
    slider_line_energy: torch.Tensor


class CoordinateBoardStem(nn.Module):
    """Small board encoder with deterministic coordinate and side-relative planes."""

    def __init__(
        self,
        input_channels: int = 18,
        stem_width: int = 64,
        depth: int = 2,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        rank = torch.linspace(0.0, 1.0, 8).view(1, 1, 8, 1).expand(1, 1, 8, 8)
        file = torch.linspace(0.0, 1.0, 8).view(1, 1, 1, 8).expand(1, 1, 8, 8)
        center = torch.sqrt((rank - 0.5).square() + (file - 0.5).square()) / (0.5 * 2.0**0.5)
        self.register_buffer("rank_plane", rank, persistent=False)
        self.register_buffer("file_plane", file, persistent=False)
        self.register_buffer("center_plane", center, persistent=False)

        layers: list[nn.Module] = []
        in_channels = input_channels + 4
        for block_idx in range(depth):
            kernel_size = 1 if block_idx == 0 else 3
            padding = 0 if kernel_size == 1 else 1
            layers.append(
                nn.Conv2d(
                    in_channels,
                    stem_width,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=not use_batchnorm,
                )
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(stem_width))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = stem_width
        self.layers = nn.Sequential(*layers)
        self.output_channels = stem_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]
        rank = self.rank_plane.to(device=x.device, dtype=x.dtype).expand(batch, -1, -1, -1)
        file = self.file_plane.to(device=x.device, dtype=x.dtype).expand(batch, -1, -1, -1)
        center = self.center_plane.to(device=x.device, dtype=x.dtype).expand(batch, -1, -1, -1)
        white_to_move = x[:, 12:13].mean(dim=(2, 3), keepdim=True).clamp(0.0, 1.0)
        forward_rank = white_to_move * (1.0 - rank) + (1.0 - white_to_move) * rank
        return self.layers(torch.cat([x, rank, file, center, forward_rank], dim=1))


class BoardConditionedLineModes(nn.Module):
    """Builds compressed line modes M(x) from fixed rank/file/diagonal incidence."""

    def __init__(
        self,
        square_dim: int,
        heads: int,
        line_rank: int,
        hidden_dim: int,
        type_dim: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.line_rank = int(line_rank)
        self.line_type_embedding = nn.Embedding(4, type_dim)
        feature_dim = square_dim + type_dim + 1
        self.mode_mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, self.heads * self.line_rank),
        )

    def forward(
        self,
        flat_square: torch.Tensor,
        square_lines: torch.Tensor,
        line_type: torch.Tensor,
        line_lengths: torch.Tensor,
    ) -> torch.Tensor:
        line_feat = self._scatter_square_features(flat_square, square_lines, line_lengths)
        type_emb = self.line_type_embedding(line_type.to(device=flat_square.device))
        type_emb = type_emb.to(dtype=flat_square.dtype).unsqueeze(0).expand(flat_square.shape[0], -1, -1)
        length_feat = (line_lengths.to(device=flat_square.device, dtype=flat_square.dtype) / 8.0).view(1, LINE_COUNT, 1)
        mode_input = torch.cat([line_feat, type_emb, length_feat.expand(flat_square.shape[0], -1, -1)], dim=-1)
        modes = self.mode_mlp(mode_input).view(flat_square.shape[0], LINE_COUNT, self.heads, self.line_rank)
        return modes.permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _scatter_square_features(
        flat_square: torch.Tensor,
        square_lines: torch.Tensor,
        line_lengths: torch.Tensor,
    ) -> torch.Tensor:
        line_feat = flat_square.new_zeros(flat_square.shape[0], LINE_COUNT, flat_square.shape[-1])
        for slot in range(4):
            line_feat.index_add_(1, square_lines[:, slot], flat_square)
        lengths = line_lengths.to(device=flat_square.device, dtype=flat_square.dtype).view(1, LINE_COUNT, 1)
        return line_feat / lengths.clamp_min(1.0)


class SchurRayLineAlgebraNetwork(nn.Module):
    """Low-rank Woodbury line-equilibrium classifier for puzzle_binary."""

    VALID_ABLATIONS = {
        "none",
        "cnn_only",
        "dense_attention_matched",
        "direct_64_solve",
        "random_line_incidence",
        "rank_file_only",
        "diag_only",
        "fixed_M",
        "no_blocker_gate",
        "large_r",
    }

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        stem_width: int = 64,
        depth: int = 2,
        heads: int = 4,
        line_rank: int = 8,
        hidden_dim: int = 96,
        head_hidden: int = 128,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        use_blocker_gate: bool = True,
        use_cnn_summary: bool = True,
        c_parameterization: str = "diagonal",
        line_count: int = LINE_COUNT,
        jitter: float = 1.0e-4,
        eps: float = 1.0e-3,
        ablation: str = "none",
        encoding: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if encoding != SIMPLE_18 or input_channels != 18:
            raise ValueError("SchurRayLineAlgebraNetwork currently implements the simple_18 board contract only")
        if line_count != LINE_COUNT:
            raise ValueError("Schur-Ray line algebra uses exactly 46 rank/file/diagonal lines")
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown Schur-Ray ablation: {ablation}")
        if c_parameterization != "diagonal":
            raise ValueError("The current Schur-Ray implementation uses diagonal positive line coupling")
        if ablation == "large_r":
            line_rank = max(int(line_rank), 32)
        if heads < 1 or line_rank < 1:
            raise ValueError("heads and line_rank must be positive")

        self.num_classes = int(num_classes)
        self.heads = int(heads)
        self.line_rank = int(line_rank)
        self.ablation = ablation
        self.use_blocker_gate = bool(use_blocker_gate)
        self.use_cnn_summary = bool(use_cnn_summary)
        self.jitter = float(jitter)
        self.eps = float(eps)

        square_lines, line_type, line_lengths = build_square_line_indices(randomized=False)
        random_square_lines, _, random_line_lengths = build_square_line_indices(randomized=True)
        self.register_buffer("square_lines", square_lines, persistent=False)
        self.register_buffer("random_square_lines", random_square_lines, persistent=False)
        self.register_buffer("line_type", line_type, persistent=False)
        self.register_buffer("line_lengths", line_lengths, persistent=False)
        self.register_buffer("random_line_lengths", random_line_lengths, persistent=False)
        self.register_buffer("rank_file_line_mask", (line_type < 2).to(dtype=torch.float32), persistent=False)
        self.register_buffer("king_zone_matrix", _king_zone_matrix(), persistent=False)

        self.stem = CoordinateBoardStem(
            input_channels=input_channels,
            stem_width=stem_width,
            depth=depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.field_head = nn.Linear(stem_width, self.heads * 3)
        self.line_modes = BoardConditionedLineModes(
            square_dim=stem_width,
            heads=self.heads,
            line_rank=self.line_rank,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.fixed_line_modes = nn.Parameter(torch.empty(self.heads, LINE_COUNT, self.line_rank))
        nn.init.normal_(self.fixed_line_modes, mean=0.0, std=0.02)
        self.c_raw = nn.Parameter(torch.zeros(self.heads, self.line_rank))

        attn_heads = 4 if stem_width % 4 == 0 else 2 if stem_width % 2 == 0 else 1
        self.attention_norm = nn.LayerNorm(stem_width)
        self.attention = nn.MultiheadAttention(stem_width, attn_heads, dropout=dropout, batch_first=True)
        self.attention_projection = nn.Sequential(
            nn.LayerNorm(stem_width * 2),
            nn.Linear(stem_width * 2, self.heads * PER_HEAD_FEATURES),
            nn.GELU(),
        )

        schur_dim = self.heads * PER_HEAD_FEATURES
        cnn_dim = stem_width * 2 if self.use_cnn_summary else 0
        classifier_dim = schur_dim + cnn_dim + MATERIAL_SUMMARY_DIM
        mid_dim = max(32, head_hidden // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Linear(classifier_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mid_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, BoardTensorSpec(input_channels=18))
        board = self.stem(x)
        flat = board.flatten(2).transpose(1, 2).contiguous()
        cnn_summary = torch.cat([board.mean(dim=(2, 3)), board.amax(dim=(2, 3))], dim=1)
        material = _material_summary(x)
        schur = self._schur_features(x, flat)

        schur_vec = schur.features
        if self.ablation == "cnn_only":
            schur_vec = torch.zeros_like(schur_vec)
        elif self.ablation == "dense_attention_matched":
            attn_in = self.attention_norm(flat)
            attn_out, _ = self.attention(attn_in, attn_in, attn_in, need_weights=False)
            attn_summary = torch.cat([attn_out.mean(dim=1), attn_out.amax(dim=1)], dim=1)
            schur_vec = self.attention_projection(attn_summary)

        fused = [schur_vec, material]
        if self.use_cnn_summary:
            fused.insert(1, cnn_summary)
        logits = self.classifier(torch.cat(fused, dim=1))
        mean_abs_correction = schur.correction.abs().mean(dim=-1)
        return {
            "logits": _format_logits(logits, self.num_classes),
            "schur_logdet": schur.schur_logdet.mean(dim=1),
            "schur_trace": schur.schur_trace.mean(dim=1),
            "line_correction_norm": schur.line_correction_norm.mean(dim=1),
            "mean_abs_correction": mean_abs_correction.mean(dim=1),
            "data_energy": schur.data_energy.mean(dim=1),
            "line_energy": schur.line_energy.mean(dim=1),
            "king_zone_energy": schur.king_zone_energy.mean(dim=1),
            "slider_line_energy": schur.slider_line_energy.mean(dim=1),
            "schur_feature_norm": schur.features.square().mean(dim=1).sqrt(),
            "material_balance": material[:, -1],
            "piece_count": material[:, -2] * 32.0,
        }

    def _schur_features(self, x: torch.Tensor, flat: torch.Tensor) -> SchurRayBatch:
        batch = flat.shape[0]
        square_lines = self._active_square_lines()
        line_lengths = self._active_line_lengths()
        field = self.field_head(flat).view(batch, 64, self.heads, 3).permute(0, 2, 3, 1).contiguous()
        b = torch.tanh(field[:, :, 0])
        d = F.softplus(field[:, :, 1]) + self.eps
        g = torch.sigmoid(field[:, :, 2])
        if (not self.use_blocker_gate) or self.ablation == "no_blocker_gate":
            g = torch.ones_like(g)

        modes = self._line_modes(flat, square_lines, line_lengths)
        u_base = self._gather_square_modes(modes, square_lines)
        u = g.unsqueeze(-1) * u_base
        if self.ablation == "diag_only":
            u = torch.zeros_like(u)

        c_diag = F.softplus(self.c_raw) + self.eps
        d_inv = d.reciprocal()
        g_matrix = torch.einsum("bhnr,bhn,bhns->bhrs", u, d_inv, u)
        c_inv = c_diag.reciprocal()
        eye = torch.eye(self.line_rank, device=flat.device, dtype=flat.dtype).view(1, 1, self.line_rank, self.line_rank)
        schur_matrix = g_matrix + torch.diag_embed(c_inv).unsqueeze(0).to(dtype=flat.dtype) + self.jitter * eye
        y = torch.einsum("bhnr,bhn->bhr", u, b)
        chol = torch.linalg.cholesky(schur_matrix)
        a = torch.cholesky_solve(y.unsqueeze(-1), chol).squeeze(-1)
        u_a = torch.einsum("bhnr,bhr->bhn", u, a)
        z = b - d_inv * u_a
        if self.ablation == "direct_64_solve":
            z = self._direct_square_solve(b, d, u, c_diag, flat.dtype)
            u_a = b - z

        correction = z - b
        topk_mean = z.topk(k=8, dim=-1).values.mean(dim=-1)
        z_std = z.std(dim=-1, unbiased=False)
        line_correction_norm = a.norm(dim=-1)
        schur_logdet = 2.0 * chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        schur_trace = schur_matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        data_energy = (correction.square() * d).sum(dim=-1) / 64.0
        u_t_z = torch.einsum("bhnr,bhn->bhr", u, z)
        line_energy = (u_t_z.square() * c_diag.unsqueeze(0).to(dtype=flat.dtype)).sum(dim=-1) / float(self.line_rank)
        king_zone_energy = self._masked_square_energy(x, correction.abs(), self._king_zone_mask(x))
        slider_line_energy = self._masked_square_energy(x, correction.abs(), self._slider_line_mask(x, square_lines))

        per_head = torch.stack(
            [
                z.mean(dim=-1),
                z.amax(dim=-1),
                z_std,
                topk_mean,
                correction.abs().mean(dim=-1),
                line_correction_norm,
                schur_logdet / float(self.line_rank),
                schur_trace / float(self.line_rank),
                data_energy,
                line_energy,
                king_zone_energy,
                slider_line_energy,
            ],
            dim=-1,
        )
        return SchurRayBatch(
            features=per_head.flatten(1),
            z=z,
            b=b,
            correction=correction,
            line_correction_norm=line_correction_norm,
            schur_logdet=schur_logdet,
            schur_trace=schur_trace,
            data_energy=data_energy,
            line_energy=line_energy,
            king_zone_energy=king_zone_energy,
            slider_line_energy=slider_line_energy,
        )

    def _line_modes(self, flat: torch.Tensor, square_lines: torch.Tensor, line_lengths: torch.Tensor) -> torch.Tensor:
        if self.ablation == "fixed_M":
            modes = self.fixed_line_modes.unsqueeze(0).to(dtype=flat.dtype).expand(flat.shape[0], -1, -1, -1)
        else:
            modes = self.line_modes(flat, square_lines, self.line_type, line_lengths)
        if self.ablation == "rank_file_only":
            mask = self.rank_file_line_mask.to(device=flat.device, dtype=flat.dtype).view(1, 1, LINE_COUNT, 1)
            modes = modes * mask
        norm = modes.square().sum(dim=2, keepdim=True).add(self.eps).sqrt()
        return modes / norm

    def _gather_square_modes(self, modes: torch.Tensor, square_lines: torch.Tensor) -> torch.Tensor:
        gather_idx = square_lines.to(device=modes.device).reshape(-1)
        selected = modes[:, :, gather_idx, :]
        return selected.view(modes.shape[0], self.heads, 64, 4, self.line_rank).sum(dim=3)

    def _direct_square_solve(
        self,
        b: torch.Tensor,
        d: torch.Tensor,
        u: torch.Tensor,
        c_diag: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        coupling = torch.einsum("bhnr,hr,bhmr->bhnm", u, c_diag.to(dtype=dtype), u)
        matrix = coupling + torch.diag_embed(d + self.jitter)
        rhs = d * b
        return torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)

    def _active_square_lines(self) -> torch.Tensor:
        if self.ablation == "random_line_incidence":
            return self.random_square_lines
        return self.square_lines

    def _active_line_lengths(self) -> torch.Tensor:
        if self.ablation == "random_line_incidence":
            return self.random_line_lengths
        return self.line_lengths

    def _king_zone_mask(self, x: torch.Tensor) -> torch.Tensor:
        king_occ = (x[:, 5] + x[:, 11]).clamp(0.0, 1.0).flatten(1)
        matrix = self.king_zone_matrix.to(device=x.device, dtype=x.dtype)
        return king_occ.matmul(matrix).clamp(0.0, 1.0)

    def _slider_line_mask(self, x: torch.Tensor, square_lines: torch.Tensor) -> torch.Tensor:
        slider_occ = x[:, [2, 3, 4, 8, 9, 10]].sum(dim=1).clamp(0.0, 1.0).flatten(1)
        line_slider = slider_occ.new_zeros(x.shape[0], LINE_COUNT)
        for slot in range(4):
            line_slider.index_add_(1, square_lines[:, slot], slider_occ)
        selected = line_slider[:, square_lines.to(device=x.device).reshape(-1)].view(x.shape[0], 64, 4)
        return selected.amax(dim=-1).clamp(0.0, 1.0)

    @staticmethod
    def _masked_square_energy(x: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=x.device, dtype=values.dtype).unsqueeze(1)
        denom = mask.sum(dim=-1).clamp_min(1.0)
        return (values * mask).sum(dim=-1) / denom


def build_schur_ray_line_algebra_network_from_config(config: dict[str, Any]) -> SchurRayLineAlgebraNetwork:
    stem_width = int(config.get("stem_width", config.get("channels", 64)))
    hidden_dim = int(config.get("line_hidden_dim", config.get("hidden_dim", 96)))
    return SchurRayLineAlgebraNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        stem_width=stem_width,
        depth=int(config.get("depth", 2)),
        heads=int(config.get("heads", config.get("num_heads", 4))),
        line_rank=int(config.get("line_rank", 8)),
        hidden_dim=hidden_dim,
        head_hidden=int(config.get("head_hidden", max(96, hidden_dim))),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        use_blocker_gate=bool(config.get("use_blocker_gate", True)),
        use_cnn_summary=bool(config.get("use_cnn_summary", True)),
        c_parameterization=str(config.get("c_parameterization", "diagonal")),
        line_count=int(config.get("line_count", LINE_COUNT)),
        jitter=float(config.get("jitter", 1.0e-4)),
        eps=float(config.get("eps", 1.0e-3)),
        ablation=str(config.get("ablation", "none")),
        encoding=str(config.get("encoding", SIMPLE_18)),
    )
