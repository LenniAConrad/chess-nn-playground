"""Commutative View-Consistency Network for idea i137.

Working thesis (from
``ideas/research/packets/classic/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md``):
puzzle-like positions create unusual cross-view consistency patterns. Project
the simple_18 board into five small latent views (square grid, occupied
pieces, line summaries, king-centred regions, material/phase counts), learn
low-rank linear maps between them, and classify from the *defects* of those
maps — direct cross-view residuals and two-step cycle residuals.

The model is materially distinct from:

* The ``ResearchPacketProbe`` scaffold: the probe never builds per-view
  encoders or cross-view defects.
* Kinematic-commutator nets (i.e. ``kinematic_commutator_bottleneck``):
  commutators there are between deterministic chess-motion operators, not
  between learned view-to-view maps.
* Generic multi-branch CNNs: the readout is dominated by cross-view
  defect statistics rather than by per-branch latents.

Supported ablations (see ``CommutativeViewConsistencyNetwork.ABLATIONS``):

* ``none`` — full implementation.
* ``views_only_no_defects`` — drop every defect feature from the head; the
  head reads only the five projected view summaries.
* ``single_square_view`` — disable piece/line/region/count encoders and
  feed only the square-view latent to the head (defects and the other view
  features are zeroed).
* ``random_view_maps`` — freeze the cross-view ``A`` maps at deterministic
  random values so the defects measure residuals against fixed (not
  learned) projections.
* ``count_to_all_only`` — restrict every defect path to start from
  ``z_count``; all non-count source latents are replaced by zeros before
  the maps are applied. Tests whether the model is a material shortcut.
* ``shuffled_piece_view`` — shuffle the per-square piece tokens within the
  batch before the piece DeepSets encoder runs. Tests whether the piece
  view contributes real geometry.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


# View identifiers used as keys throughout the module. The ordering is fixed so
# tests can index into ``view_pooled`` / ``view_norms`` deterministically.
VIEW_NAMES: tuple[str, ...] = ("square", "piece", "line", "region", "count")

# Cross-view maps applied in the defect features. Each tuple is
# (source_view, target_view); the matrices live on the model as
# ``A_{source}_to_{target}``.
DEFECT_MAP_EDGES: tuple[tuple[str, str], ...] = (
    ("square", "line"),
    ("line", "square"),
    ("square", "region"),
    ("piece", "region"),
    ("region", "count"),
    ("square", "count"),
    ("count", "square"),
    ("region", "piece"),
)

# Direct cross-view defects. Each entry is (target_view, source_view) and the
# defect is z_target - A_{source -> target}(z_source). Two-step cycles are
# defined separately in ``CYCLE_DEFECTS``.
DIRECT_DEFECTS: tuple[tuple[str, str], ...] = (
    ("line", "square"),
    ("square", "line"),
    ("region", "square"),  # uses A_{square_to_region}(z_square) on the "expected" side
    ("count", "region"),
    ("square", "count"),
    ("piece", "region"),
)

# Two-step cycle defects (z_view - A_{B->view}(A_{view->B}(z_view))). The
# entries are (loop_view, mid_view); both maps must appear in DEFECT_MAP_EDGES.
CYCLE_DEFECTS: tuple[tuple[str, str], ...] = (
    ("square", "line"),
    ("piece", "region"),
    ("square", "count"),
)


class _SquareEncoder(nn.Module):
    """Compact convolutional encoder over the simple_18 board tensor."""

    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        latent_dim: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        layers: list[nn.Module] = []
        in_ch = int(input_channels)
        for _ in range(depth):
            layers.append(nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            else:
                layers.append(nn.GroupNorm(1, channels))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = channels
        self.body = nn.Sequential(*layers)
        # Mean + max pooling -> 2 * channels features projected to latent_dim.
        self.project = nn.Linear(2 * channels, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.body(x)
        mean = feat.mean(dim=(-2, -1))
        amax = feat.amax(dim=(-2, -1))
        return self.project(torch.cat([mean, amax], dim=-1))


class _PieceDeepSets(nn.Module):
    """DeepSets over occupied piece-square tokens.

    Each occupied square contributes a token built from its 12-d piece-type
    one-hot and its (rank, file) coordinates. The encoder is mean-aggregated
    so the operator is permutation-invariant in the piece-token set.
    """

    PIECE_CHANNELS: int = 12

    def __init__(self, hidden_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        # 12 piece planes + 4 coordinate features (rank, file, centre, parity).
        self.token_dim = self.PIECE_CHANNELS + 4
        self.token_mlp = nn.Sequential(
            nn.Linear(self.token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        head_layers: list[nn.Module] = [nn.LayerNorm(hidden_dim)]
        if dropout > 0.0:
            head_layers.append(nn.Dropout(dropout))
        head_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.head = nn.Sequential(*head_layers)

        rank = torch.linspace(0.0, 1.0, 8).view(8, 1).expand(8, 8)
        file = torch.linspace(0.0, 1.0, 8).view(1, 8).expand(8, 8)
        centre = torch.sqrt((rank - 0.5).square() + (file - 0.5).square()) / (0.5 * 2.0**0.5)
        parity = ((torch.arange(8).view(8, 1) + torch.arange(8).view(1, 8)) % 2).float()
        coords = torch.stack([rank, file, centre, parity], dim=0)  # (4, 8, 8)
        self.register_buffer("coords", coords, persistent=False)

    def forward(self, x: torch.Tensor, shuffle: bool = False) -> torch.Tensor:
        batch = x.shape[0]
        pieces = x[:, : self.PIECE_CHANNELS].clamp(0.0, 1.0)  # (B, 12, 8, 8)
        occupancy = pieces.sum(dim=1, keepdim=True).clamp(0.0, 1.0)  # (B, 1, 8, 8)
        coords = self.coords.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(batch, -1, -1, -1)
        tokens = torch.cat([pieces, coords], dim=1)  # (B, 16, 8, 8)
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, 64, 16)
        occupancy_flat = occupancy.flatten(2).transpose(1, 2)  # (B, 64, 1)
        if shuffle and batch > 1:
            perm = torch.randperm(batch, device=x.device)
            tokens = tokens[perm]
            occupancy_flat = occupancy_flat[perm]
        encoded = self.token_mlp(tokens)  # (B, 64, hidden_dim)
        weighted = encoded * occupancy_flat
        denom = occupancy_flat.sum(dim=1).clamp_min(1.0)
        pooled = weighted.sum(dim=1) / denom
        return self.head(pooled)


def _line_summary(x: torch.Tensor) -> torch.Tensor:
    """Per-batch rank/file/diagonal occupancy summaries."""
    pieces = x[:, :12].clamp(0.0, 1.0)
    occupancy = pieces.sum(dim=1)  # (B, 8, 8)
    rank_means = occupancy.mean(dim=2)  # (B, 8)
    file_means = occupancy.mean(dim=1)  # (B, 8)
    diag_means = []
    anti_diag_means = []
    for offset in range(-3, 4):
        diag = torch.diagonal(occupancy, offset=offset, dim1=-2, dim2=-1).mean(dim=-1)
        diag_means.append(diag)
        flipped = torch.flip(occupancy, dims=[-1])
        anti = torch.diagonal(flipped, offset=offset, dim1=-2, dim2=-1).mean(dim=-1)
        anti_diag_means.append(anti)
    diag_tensor = torch.stack(diag_means, dim=-1)  # (B, 7)
    anti_tensor = torch.stack(anti_diag_means, dim=-1)  # (B, 7)
    return torch.cat([rank_means, file_means, diag_tensor, anti_tensor], dim=-1)


def _region_summary(x: torch.Tensor) -> torch.Tensor:
    """Centre / edge / corner / king-ring occupancy summaries."""
    pieces = x[:, :12].clamp(0.0, 1.0)
    occupancy = pieces.sum(dim=1)  # (B, 8, 8)
    batch = occupancy.shape[0]
    device = occupancy.device
    dtype = occupancy.dtype

    rank_idx = torch.arange(8, device=device).view(8, 1).expand(8, 8)
    file_idx = torch.arange(8, device=device).view(1, 8).expand(8, 8)
    centre_mask = ((rank_idx >= 2) & (rank_idx <= 5) & (file_idx >= 2) & (file_idx <= 5)).to(dtype)
    edge_mask = ((rank_idx == 0) | (rank_idx == 7) | (file_idx == 0) | (file_idx == 7)).to(dtype)
    corner_mask = (((rank_idx == 0) | (rank_idx == 7)) & ((file_idx == 0) | (file_idx == 7))).to(dtype)
    inner_mask = ((rank_idx >= 3) & (rank_idx <= 4) & (file_idx >= 3) & (file_idx <= 4)).to(dtype)

    summaries = [
        (occupancy * centre_mask).mean(dim=(-2, -1)),
        (occupancy * edge_mask).mean(dim=(-2, -1)),
        (occupancy * corner_mask).mean(dim=(-2, -1)),
        (occupancy * inner_mask).mean(dim=(-2, -1)),
    ]

    # King-centred rings (radius 1 and 2) for each side's king.
    for plane in (5, 11):  # white K, black K planes in simple_18
        king_plane = x[:, plane].clamp(0.0, 1.0)
        king_pos = king_plane.flatten(1)
        king_total = king_pos.sum(dim=1).clamp_min(1e-6)
        norm_king = king_pos / king_total.unsqueeze(1)
        norm_king = norm_king.view(batch, 8, 8)
        rk_mean = (rank_idx.to(dtype).unsqueeze(0) * norm_king).sum(dim=(-2, -1))
        fk_mean = (file_idx.to(dtype).unsqueeze(0) * norm_king).sum(dim=(-2, -1))
        rdiff = (rank_idx.to(dtype).unsqueeze(0) - rk_mean.view(-1, 1, 1)).abs()
        fdiff = (file_idx.to(dtype).unsqueeze(0) - fk_mean.view(-1, 1, 1)).abs()
        cheb = torch.maximum(rdiff, fdiff)
        ring1 = (cheb <= 1.0).to(dtype)
        ring2 = ((cheb > 1.0) & (cheb <= 2.0)).to(dtype)
        summaries.append((occupancy * ring1).mean(dim=(-2, -1)))
        summaries.append((occupancy * ring2).mean(dim=(-2, -1)))

    return torch.stack(summaries, dim=-1)  # (B, 8)


def _count_summary(x: torch.Tensor) -> torch.Tensor:
    """Material / side-to-move / castling / en-passant summaries."""
    pieces = x[:, :12].clamp(0.0, 1.0)
    white_counts = pieces[:, :6].sum(dim=(-2, -1)) / 8.0  # (B, 6)
    black_counts = pieces[:, 6:12].sum(dim=(-2, -1)) / 8.0
    diff = white_counts - black_counts
    side = x[:, 12].mean(dim=(-2, -1)).clamp(0.0, 1.0).unsqueeze(-1)
    castling = x[:, 13:17].mean(dim=(-2, -1)).clamp(0.0, 1.0)
    en_passant = x[:, 17].clamp(0.0, 1.0).flatten(1).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
    values = x.new_tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])
    own_material = (white_counts * values).sum(dim=-1, keepdim=True) / 39.0
    opp_material = (black_counts * values).sum(dim=-1, keepdim=True) / 39.0
    material_balance = own_material - opp_material
    return torch.cat(
        [white_counts, black_counts, diff, side, castling, en_passant, material_balance],
        dim=-1,
    )


class _MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _LowRankMap(nn.Module):
    """Factorised rank-``r`` linear map between view latents."""

    def __init__(self, latent_dim: int, rank: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.latent_dim = int(latent_dim)
        self.rank = int(rank)
        self.left = nn.Linear(latent_dim, rank, bias=False)
        self.right = nn.Linear(rank, latent_dim, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.right(self.left(z))

    @torch.no_grad()
    def freeze_random(self, generator: torch.Generator | None = None) -> None:
        """Freeze the map at deterministic random scale-matched values."""
        std_left = (1.0 / self.latent_dim) ** 0.5
        std_right = (1.0 / max(1, self.rank)) ** 0.5
        if generator is None:
            self.left.weight.normal_(0.0, std_left)
            self.right.weight.normal_(0.0, std_right)
        else:
            self.left.weight.copy_(torch.randn(self.left.weight.shape, generator=generator) * std_left)
            self.right.weight.copy_(torch.randn(self.right.weight.shape, generator=generator) * std_right)
        if self.right.bias is not None:
            self.right.bias.zero_()
        for p in self.parameters():
            p.requires_grad_(False)


def _defect_statistics(defect: torch.Tensor, target: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    """Five statistics per defect: MSE, mean-abs, signed-mean, max-abs, cosine consistency."""
    mse = defect.pow(2).mean(dim=-1)
    mae = defect.abs().mean(dim=-1)
    signed = defect.mean(dim=-1)
    max_abs = defect.abs().amax(dim=-1)
    eps = 1e-8
    t_norm = target.norm(dim=-1).clamp_min(eps)
    p_norm = predicted.norm(dim=-1).clamp_min(eps)
    cosine = (target * predicted).sum(dim=-1) / (t_norm * p_norm)
    return torch.stack([mse, mae, signed, max_abs, cosine], dim=-1)


class CommutativeViewConsistencyNetwork(nn.Module):
    """Bespoke commutative view-consistency classifier for puzzle_binary."""

    ABLATIONS: tuple[str, ...] = (
        "none",
        "views_only_no_defects",
        "single_square_view",
        "random_view_maps",
        "count_to_all_only",
        "shuffled_piece_view",
    )

    VIEW_NAMES: tuple[str, ...] = VIEW_NAMES
    DEFECT_MAP_EDGES: tuple[tuple[str, str], ...] = DEFECT_MAP_EDGES
    DIRECT_DEFECTS: tuple[tuple[str, str], ...] = DIRECT_DEFECTS
    CYCLE_DEFECTS: tuple[tuple[str, str], ...] = CYCLE_DEFECTS
    DEFECT_STAT_NAMES: tuple[str, ...] = ("mse", "mae", "signed", "max_abs", "cosine")

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        latent_dim: int = 32,
        map_rank: int = 8,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
        random_map_seed: int = 1337,
        height: int = 8,
        width: int = 8,
        encoding: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if encoding != SIMPLE_18 or int(input_channels) != 18:
            raise ValueError(
                "CommutativeViewConsistencyNetwork currently implements the simple_18 18-plane contract only"
            )
        if int(num_classes) != 1:
            raise ValueError("CommutativeViewConsistencyNetwork supports the puzzle_binary one-logit contract")
        if ablation not in self.ABLATIONS:
            raise ValueError(
                f"Unknown commutative-view ablation: {ablation!r}; expected one of {self.ABLATIONS}"
            )

        self.spec = BoardTensorSpec(
            input_channels=int(input_channels), height=int(height), width=int(width)
        )
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.map_rank = int(map_rank)
        self.depth = int(depth)
        self.dropout_p = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.ablation = ablation
        self.random_map_seed = int(random_map_seed)

        self.square_encoder = _SquareEncoder(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            latent_dim=self.latent_dim,
            dropout=self.dropout_p,
            use_batchnorm=self.use_batchnorm,
        )
        self.piece_encoder = _PieceDeepSets(
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout=self.dropout_p,
        )
        # Line summary length: 8 ranks + 8 files + 7 diagonals + 7 anti-diagonals.
        self.line_encoder = _MLPEncoder(
            input_dim=30, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim, dropout=self.dropout_p
        )
        self.region_encoder = _MLPEncoder(
            input_dim=8, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim, dropout=self.dropout_p
        )
        # 6 white + 6 black piece counts + 6 diff + 1 side + 4 castling + 1 ep + 1 balance = 25.
        self.count_encoder = _MLPEncoder(
            input_dim=25, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim, dropout=self.dropout_p
        )

        self.maps = nn.ModuleDict(
            {
                f"{src}_to_{dst}": _LowRankMap(self.latent_dim, self.map_rank)
                for src, dst in DEFECT_MAP_EDGES
            }
        )
        if ablation == "random_view_maps":
            generator = torch.Generator().manual_seed(self.random_map_seed)
            for module in self.maps.values():
                module.freeze_random(generator)

        defect_count = len(DIRECT_DEFECTS) + len(CYCLE_DEFECTS)
        defect_stat_dim = defect_count * len(self.DEFECT_STAT_NAMES)
        # Head input: 5 view latents (each latent_dim) + defect statistics.
        self.head_view_dim = len(VIEW_NAMES) * self.latent_dim
        self.head_defect_dim = defect_stat_dim
        head_input_dim = self.head_view_dim + self.head_defect_dim
        self.head_input_dim = head_input_dim
        head_layers: list[nn.Module] = [
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, self.hidden_dim),
            nn.GELU(),
        ]
        if self.dropout_p > 0.0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.hidden_dim, self.num_classes))
        self.classifier = nn.Sequential(*head_layers)

    @property
    def num_views(self) -> int:
        return len(VIEW_NAMES)

    @property
    def num_defects(self) -> int:
        return len(DIRECT_DEFECTS) + len(CYCLE_DEFECTS)

    def _ablation_code(self) -> float:
        return float(self.ABLATIONS.index(self.ablation))

    def _compute_view_latents(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        line_in = _line_summary(x)
        region_in = _region_summary(x)
        count_in = _count_summary(x)
        z = {
            "square": self.square_encoder(x),
            "piece": self.piece_encoder(x, shuffle=self.ablation == "shuffled_piece_view"),
            "line": self.line_encoder(line_in),
            "region": self.region_encoder(region_in),
            "count": self.count_encoder(count_in),
        }
        if self.ablation == "single_square_view":
            for name in VIEW_NAMES:
                if name != "square":
                    z[name] = torch.zeros_like(z[name])
        elif self.ablation == "count_to_all_only":
            for name in VIEW_NAMES:
                if name != "count":
                    z[name] = torch.zeros_like(z[name])
        return z

    def _apply_map(self, src: str, dst: str, z: torch.Tensor) -> torch.Tensor:
        return self.maps[f"{src}_to_{dst}"](z)

    def _compute_defects(
        self, z: dict[str, torch.Tensor]
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Return per-defect residual vectors and a stacked statistics tensor.

        Each direct defect is z_target - A_{source->target}(z_source). The
        two-step cycle defects are z_view - A_{mid->view}(A_{view->mid}(z_view)).
        Returns (residuals, stats) where ``stats`` has shape
        (batch, num_defects, 5).
        """
        residuals: list[torch.Tensor] = []
        stats: list[torch.Tensor] = []
        for target_view, source_view in DIRECT_DEFECTS:
            predicted = self._apply_map(source_view, target_view, z[source_view])
            target = z[target_view]
            defect = target - predicted
            residuals.append(defect)
            stats.append(_defect_statistics(defect, target=target, predicted=predicted))
        for loop_view, mid_view in CYCLE_DEFECTS:
            forward = self._apply_map(loop_view, mid_view, z[loop_view])
            back = self._apply_map(mid_view, loop_view, forward)
            defect = z[loop_view] - back
            residuals.append(defect)
            stats.append(_defect_statistics(defect, target=z[loop_view], predicted=back))
        stacked = torch.stack(stats, dim=1)
        return residuals, stacked

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]

        z = self._compute_view_latents(x)
        residuals, defect_stats = self._compute_defects(z)

        view_pooled = torch.stack([z[name] for name in VIEW_NAMES], dim=1)  # (B, V, D)
        view_features = view_pooled.flatten(1)
        defect_features = defect_stats.flatten(1)
        if self.ablation == "views_only_no_defects":
            defect_features = torch.zeros_like(defect_features)
        head_input = torch.cat([view_features, defect_features], dim=-1)
        logits_raw = self.classifier(head_input)
        if self.num_classes == 1:
            logits = logits_raw.squeeze(-1)
            prob = torch.sigmoid(logits)
        else:
            logits = logits_raw
            prob = torch.softmax(logits_raw, dim=-1)

        view_norms = view_pooled.pow(2).mean(dim=-1).sqrt()  # (B, V)
        defect_l2 = defect_stats[..., 0].sqrt()  # sqrt(MSE) per defect
        defect_l1 = defect_stats[..., 1]
        defect_cos = defect_stats[..., 4]
        consistency_energy = defect_l2.pow(2).mean(dim=-1)
        mean_defect_l1 = defect_l1.mean(dim=-1)
        mean_defect_cosine = defect_cos.mean(dim=-1)
        proposal_keyword_count = logits.new_full((batch,), float(self.num_views))

        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "prob": prob,
            "view_pooled": view_pooled,
            "view_norms": view_norms,
            "defect_stats": defect_stats,
            "defect_l2": defect_l2,
            "defect_l1": defect_l1,
            "defect_cosine": defect_cos,
            "consistency_energy": consistency_energy,
            "mean_defect_l1": mean_defect_l1,
            "mean_defect_cosine": mean_defect_cosine,
            "mechanism_energy": consistency_energy,
            "proposal_profile_strength": mean_defect_l1,
            "proposal_keyword_count": proposal_keyword_count,
            "commutative_view_ablation": logits.new_full((batch,), self._ablation_code()),
            "commutative_view_count": logits.new_full((batch,), float(self.num_views)),
        }
        for view_idx, view_name in enumerate(VIEW_NAMES):
            out[f"z_{view_name}"] = view_pooled[:, view_idx]
        for defect_idx, residual in enumerate(residuals):
            out[f"defect_{defect_idx}"] = residual
        return out


def build_commutative_view_consistency_network_from_config(
    config: dict[str, Any],
) -> CommutativeViewConsistencyNetwork:
    return CommutativeViewConsistencyNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        latent_dim=int(config.get("latent_dim", 32)),
        map_rank=int(config.get("map_rank", 8)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        ablation=str(config.get("ablation", "none")),
        random_map_seed=int(config.get("random_map_seed", 1337)),
        height=int(config.get("height", 8)),
        width=int(config.get("width", 8)),
        encoding=str(config.get("encoding", SIMPLE_18)),
    )
