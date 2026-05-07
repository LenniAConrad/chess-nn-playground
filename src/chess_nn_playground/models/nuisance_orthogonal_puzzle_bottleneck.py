"""Nuisance-Orthogonal Puzzle Bottleneck (idea i030).

Bespoke implementation of the architecture described in
``ideas/i030_nuisance_orthogonal_puzzle_bottleneck/architecture.md`` and
``math_thesis.md``. The model classifies puzzle-likeness from a current-board
``simple_18`` tensor by:

1. extracting a deterministic chess nuisance vector ``n(B)`` (material, phase,
   side-to-move, castling/en-passant, king coordinates, pawn-file profile);
2. expanding ``n`` to a fixed normalized nuisance feature matrix ``Q``;
3. running a compact convolutional residual trunk to produce a learned latent
   ``H``;
4. residualising ``H`` against ``Q`` with a closed-form batchwise ridge
   projection ``Z = H - gamma * Q (Q^T Q + lambda I)^{-1} Q^T H`` so the
   classifier must read tactical structure that is linearly orthogonal to the
   nuisance design;
5. classifying the residual latent with a small MLP head.

The projection contains no trainable parameters: the nuisance extractor and
feature map are fully deterministic, and the projection itself is the closed
form solution of ``min ||H-Z||_F^2 s.t. Q^T Z = 0`` (ridge-regularised when
``lambda > 0``).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


PIECE_VALUES = (1.0, 3.0, 3.0, 5.0, 9.0, 0.0)  # P, N, B, R, Q, K
NUISANCE_COMPONENT_NAMES: tuple[str, ...] = (
    "piece_counts",          # 12: normalized counts of P,N,B,R,Q,K (white) then (black)
    "material_summaries",    # 6: white_mat, black_mat, balance, abs_imbalance, phase, total_pieces
    "side_to_move",          # 1: +1 white-to-move, -1 black-to-move
    "castling_rights",       # 4: WK, WQ, BK, BQ
    "en_passant",            # 9: presence + 8 file one-hot
    "king_coords",           # 8: white file/rank, black file/rank, four edge-distance summaries
    "pawn_file_counts",      # 16: white pawns per file (8) + black pawns per file (8)
    "occupancy_marginals",   # 8: 4 rank-group occupancy + 4 file-group occupancy
)


def _nuisance_dim_breakdown() -> dict[str, int]:
    return {
        "piece_counts": 12,
        "material_summaries": 6,
        "side_to_move": 1,
        "castling_rights": 4,
        "en_passant": 9,
        "king_coords": 8,
        "pawn_file_counts": 16,
        "occupancy_marginals": 8,
    }


NUISANCE_BASE_DIM = sum(_nuisance_dim_breakdown().values())  # = 64


class Simple18Adapter(nn.Module):
    """Validates the simple_18 channel contract and reads nuisance-relevant planes."""

    def __init__(self, input_channels: int, encoding: str, fail_closed: bool = True) -> None:
        super().__init__()
        if encoding != SIMPLE_18:
            raise ValueError(
                "NuisanceOrthogonalPuzzleNet currently supports the simple_18 encoding only; "
                f"got encoding={encoding!r}. Add a registered semantic adapter before enabling other encodings."
            )
        if input_channels != 18:
            raise ValueError(
                f"simple_18 requires exactly 18 input channels, received input_channels={input_channels}"
            )
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.fail_closed = bool(fail_closed)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        return {
            "white_pieces": x[:, 0:6],     # P, N, B, R, Q, K
            "black_pieces": x[:, 6:12],    # p, n, b, r, q, k
            "side_to_move": x[:, 12],      # 1 = white to move
            "castling": x[:, 13:17],       # WK, WQ, BK, BQ
            "en_passant": x[:, 17],
        }


class DeterministicNuisanceExtractor(nn.Module):
    """Computes a deterministic chess nuisance vector ``n(B)``.

    The vector concatenates piece counts, material/phase summaries, side-to-move,
    castling/en-passant flags, king coordinates, pawn-file counts, and coarse
    occupancy marginals. Output dim == ``NUISANCE_BASE_DIM`` (64).
    """

    def __init__(self) -> None:
        super().__init__()
        values = torch.tensor(PIECE_VALUES, dtype=torch.float32)
        non_pawn_mask = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=torch.float32)
        coords = torch.arange(8, dtype=torch.float32)
        self.register_buffer("piece_values", values, persistent=False)
        self.register_buffer("non_pawn_mask", non_pawn_mask, persistent=False)
        self.register_buffer("coords", coords, persistent=False)

    def _king_coords(self, king_plane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # king_plane: [B, 8, 8]; return normalized (file, rank) in [-1, 1].
        b = king_plane.shape[0]
        flat = king_plane.flatten(1)
        # Argmax indexes the first square the king occupies; if no king (synthetic input)
        # we still return a deterministic value (centre square) without raising.
        idx = flat.argmax(dim=1)
        rank_idx = idx // 8
        file_idx = idx % 8
        rank = rank_idx.float() / 3.5 - 1.0
        file_ = file_idx.float() / 3.5 - 1.0
        return file_, rank

    def forward(self, planes: dict[str, torch.Tensor]) -> torch.Tensor:
        white_pieces = planes["white_pieces"]    # [B, 6, 8, 8]
        black_pieces = planes["black_pieces"]    # [B, 6, 8, 8]
        side_to_move_plane = planes["side_to_move"]
        castling = planes["castling"]            # [B, 4, 8, 8]
        en_passant = planes["en_passant"]        # [B, 8, 8]

        b = white_pieces.shape[0]
        device = white_pieces.device
        dtype = white_pieces.dtype

        # Piece counts (0..1 scale: divide by canonical maxima).
        white_counts = white_pieces.sum(dim=(2, 3))   # [B, 6]
        black_counts = black_pieces.sum(dim=(2, 3))   # [B, 6]
        canonical_max = torch.tensor([8.0, 2.0, 2.0, 2.0, 1.0, 1.0], device=device, dtype=dtype)
        wc_norm = white_counts / canonical_max
        bc_norm = black_counts / canonical_max
        piece_counts = torch.cat([wc_norm, bc_norm], dim=1)  # [B, 12]

        # Material and phase summaries.
        values = self.piece_values.to(device=device, dtype=dtype)
        non_pawn = self.non_pawn_mask.to(device=device, dtype=dtype)
        white_mat = (white_counts * values).sum(dim=1) / 39.0
        black_mat = (black_counts * values).sum(dim=1) / 39.0
        balance = white_mat - black_mat
        abs_imb = balance.abs()
        non_pawn_total = ((white_counts + black_counts) * non_pawn).sum(dim=1) / 14.0
        total_pieces = (white_counts.sum(dim=1) + black_counts.sum(dim=1)) / 32.0
        material = torch.stack([white_mat, black_mat, balance, abs_imb, non_pawn_total, total_pieces], dim=1)

        # Side to move: +1 for white-to-move, -1 for black-to-move.
        white_to_move = side_to_move_plane.flatten(1).mean(dim=1).clamp(0.0, 1.0)
        side_scalar = (2.0 * white_to_move - 1.0).unsqueeze(1)

        # Castling flags: each plane is either all 0 or all 1, so the mean recovers the bit.
        castling_flags = castling.flatten(2).mean(dim=2)  # [B, 4]

        # En-passant: presence + file one-hot.
        ep_per_file = en_passant.sum(dim=1).clamp(0.0, 1.0)  # [B, 8] (sum across ranks)
        ep_presence = ep_per_file.amax(dim=1, keepdim=True)
        ep_features = torch.cat([ep_presence, ep_per_file], dim=1)  # [B, 9]

        # King coordinates and edge-distance summaries.
        wk_file, wk_rank = self._king_coords(white_pieces[:, 5])
        bk_file, bk_rank = self._king_coords(black_pieces[:, 5])
        wk_edge_file = 1.0 - wk_file.abs()
        wk_edge_rank = 1.0 - wk_rank.abs()
        bk_edge_file = 1.0 - bk_file.abs()
        bk_edge_rank = 1.0 - bk_rank.abs()
        king_coords = torch.stack(
            [wk_file, wk_rank, bk_file, bk_rank, wk_edge_file, wk_edge_rank, bk_edge_file, bk_edge_rank],
            dim=1,
        )

        # Pawn-file counts (white and black) normalized to [0, 1].
        white_pawn_files = white_pieces[:, 0].sum(dim=1) / 8.0   # [B, 8]
        black_pawn_files = black_pieces[:, 0].sum(dim=1) / 8.0   # [B, 8]
        pawn_file_counts = torch.cat([white_pawn_files, black_pawn_files], dim=1)  # [B, 16]

        # Coarse rank-group / file-group occupancy.
        occupancy = (white_pieces.sum(dim=1) + black_pieces.sum(dim=1)).clamp(0.0, 1.0)  # [B, 8, 8]
        rank_marginal = occupancy.sum(dim=2)  # [B, 8]
        file_marginal = occupancy.sum(dim=1)  # [B, 8]
        rank_groups = rank_marginal.view(b, 4, 2).sum(dim=2) / 16.0  # [B, 4]
        file_groups = file_marginal.view(b, 4, 2).sum(dim=2) / 16.0  # [B, 4]
        occupancy_marginals = torch.cat([rank_groups, file_groups], dim=1)  # [B, 8]

        nuisance = torch.cat(
            [
                piece_counts,
                material,
                side_scalar,
                castling_flags,
                ep_features,
                king_coords,
                pawn_file_counts,
                occupancy_marginals,
            ],
            dim=1,
        )
        assert nuisance.shape == (b, NUISANCE_BASE_DIM), (
            f"nuisance shape {tuple(nuisance.shape)} != expected (B, {NUISANCE_BASE_DIM})"
        )
        return nuisance


class FixedNuisanceFeatureMap(nn.Module):
    """Maps the deterministic nuisance vector to a fixed feature matrix Q.

    The map concatenates the raw nuisance vector with a deterministic random
    projection of pairwise products and squares of nuisance entries, then
    LayerNorms (no affine) the result. The random projection matrix is a fixed
    registered buffer derived from ``seed=42`` so the projection is reproducible
    across runs and platforms; it has no trainable parameters.
    """

    def __init__(self, in_dim: int, rank: int, expansion_dim: int = 64, seed: int = 42) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("nuisance_rank must be >= 1")
        self.in_dim = int(in_dim)
        self.rank = int(rank)
        self.expansion_dim = int(expansion_dim)

        # Deterministic projection of pairwise products and squares to expansion_dim.
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        nonlinear_in = in_dim * (in_dim + 1) // 2  # all squares + all unique pairs (incl. diag)
        proj = torch.randn(nonlinear_in, expansion_dim, generator=gen) / (nonlinear_in ** 0.5)
        self.register_buffer("nonlinear_proj", proj, persistent=False)

        idx_a, idx_b = torch.triu_indices(in_dim, in_dim, offset=0).unbind(0)
        self.register_buffer("pair_idx_a", idx_a, persistent=False)
        self.register_buffer("pair_idx_b", idx_b, persistent=False)

        full_dim = in_dim + expansion_dim
        if self.rank > full_dim:
            raise ValueError(
                f"nuisance_rank={self.rank} exceeds the available feature dimension {full_dim}"
            )
        self.full_dim = full_dim
        self.norm = nn.LayerNorm(self.rank, elementwise_affine=False)

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        # Pairwise products including squares: [B, P]
        a = n.index_select(1, self.pair_idx_a)
        b = n.index_select(1, self.pair_idx_b)
        pair_products = a * b
        nonlinear = pair_products @ self.nonlinear_proj.to(dtype=n.dtype)
        full = torch.cat([n, nonlinear], dim=1)
        # Truncate to the configured rank.
        q = full[:, : self.rank]
        return self.norm(q)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm1 = nn.BatchNorm2d(channels) if use_batchnorm else nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm2 = nn.BatchNorm2d(channels) if use_batchnorm else nn.GroupNorm(min(8, channels), channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class ConvResidualTrunk(nn.Module):
    """Compact residual CNN trunk producing a pooled latent of size ``latent_dim``."""

    def __init__(
        self,
        input_channels: int,
        channels: int,
        latent_dim: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("trunk_blocks must be >= 1")
        proj_channels = max(channels, latent_dim // 2 + 1, channels + 32)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(channels) if use_batchnorm else nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[_ResidualBlock(channels, dropout, use_batchnorm) for _ in range(depth)]
        )
        self.project = nn.Sequential(
            nn.Conv2d(channels, proj_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(proj_channels) if use_batchnorm else nn.GroupNorm(min(8, proj_channels), proj_channels),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.latent = nn.Sequential(
            nn.Linear(proj_channels, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        h = self.project(h)
        h = self.flatten(self.pool(h))
        return self.latent(h)


class BatchRidgeOrthogonalProjector(nn.Module):
    """Closed-form ridge projection ``Z = H - gamma * Q(Q^T Q + lambda I)^{-1} Q^T H``.

    Operates on a centred mini-batch. The projection is computed in float32 for
    numerical stability of the linear solve and cast back to the input dtype.
    Returns the residual ``Z`` plus diagnostics: the residual covariance norm
    ``||Q^T Z / b||``, the latent variance after projection, and the empirical
    rank of ``Q^T Q``.
    """

    def __init__(self, ridge_lambda: float, gamma: float) -> None:
        super().__init__()
        if ridge_lambda < 0:
            raise ValueError("ridge_lambda must be >= 0")
        if gamma < 0:
            raise ValueError("projection_gamma must be >= 0")
        self.ridge_lambda = float(ridge_lambda)
        self.gamma = float(gamma)

    def forward(self, h: torch.Tensor, q: torch.Tensor) -> dict[str, torch.Tensor]:
        b = h.shape[0]
        d = h.shape[1]
        k = q.shape[1]
        if q.shape[0] != b:
            raise ValueError(f"H and Q must share batch size; got H={tuple(h.shape)} Q={tuple(q.shape)}")

        h32 = h.float()
        q32 = q.float()

        # Centre across the mini-batch so the operator removes empirical linear
        # dependence between Z and Q rather than just shifting the means.
        if b > 1:
            h_centred = h32 - h32.mean(dim=0, keepdim=True)
            q_centred = q32 - q32.mean(dim=0, keepdim=True)
        else:
            h_centred = h32
            q_centred = q32

        if self.gamma == 0.0:
            z = h_centred
            residual_corr = (q_centred.transpose(0, 1) @ z) / max(b, 1)
            cov_norm = residual_corr.norm()
        else:
            eye = torch.eye(k, device=q32.device, dtype=q32.dtype)
            gram = q_centred.transpose(0, 1) @ q_centred + self.ridge_lambda * eye
            qth = q_centred.transpose(0, 1) @ h_centred
            try:
                a = torch.linalg.solve(gram, qth)
            except RuntimeError:
                a = torch.linalg.lstsq(gram, qth).solution
            proj = q_centred @ a
            z = h_centred - self.gamma * proj
            residual_corr = (q_centred.transpose(0, 1) @ z) / max(b, 1)
            cov_norm = residual_corr.norm()

        latent_var = z.var(dim=0, unbiased=False).mean() if b > 1 else z.pow(2).mean()
        gram_for_diag = q_centred.transpose(0, 1) @ q_centred
        try:
            rank = torch.linalg.matrix_rank(gram_for_diag).to(z.dtype)
        except RuntimeError:
            rank = torch.tensor(float(min(b, k)), device=z.device, dtype=z.dtype)
        return {
            "z": z.to(dtype=h.dtype),
            "residual_cov_norm": cov_norm.to(dtype=h.dtype),
            "latent_variance": latent_var.to(dtype=h.dtype),
            "nuisance_rank": rank,
        }


class ClassifierHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class NuisanceOrthogonalPuzzleNet(nn.Module):
    """Closed-form nuisance-orthogonal residualisation classifier (idea i030)."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        latent_dim: int = 256,
        nuisance_rank: int = 64,
        nuisance_expansion_dim: int = 64,
        ridge_lambda: float = 1.0e-3,
        projection_gamma: float = 1.0,
        encoding: str = SIMPLE_18,
        fail_closed_semantics: bool = True,
        nuisance_seed: int = 42,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        self.num_classes = int(num_classes)
        self.adapter = Simple18Adapter(
            input_channels=input_channels,
            encoding=encoding,
            fail_closed=fail_closed_semantics,
        )
        self.nuisance_extractor = DeterministicNuisanceExtractor()
        self.nuisance_feature_map = FixedNuisanceFeatureMap(
            in_dim=NUISANCE_BASE_DIM,
            rank=nuisance_rank,
            expansion_dim=nuisance_expansion_dim,
            seed=nuisance_seed,
        )
        self.trunk = ConvResidualTrunk(
            input_channels=input_channels,
            channels=channels,
            latent_dim=latent_dim,
            depth=depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.projector = BatchRidgeOrthogonalProjector(ridge_lambda=ridge_lambda, gamma=projection_gamma)
        self.head = ClassifierHead(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.encoding = encoding
        self.nuisance_dim_breakdown = _nuisance_dim_breakdown()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        planes = self.adapter(x)
        n = self.nuisance_extractor(planes)
        q = self.nuisance_feature_map(n)
        h = self.trunk(x)
        proj = self.projector(h, q)
        z = proj["z"]
        logits = self.head(z)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)

        nuisance_energy = n.pow(2).mean(dim=1)
        material_balance = n[:, 14]                             # signed material balance
        material_imbalance = n[:, 15]                           # |material balance|
        side_to_move_scalar = n[:, 18]                          # +1 white / -1 black
        castling_total = n[:, 19:23].sum(dim=1)                 # 0..4
        en_passant_presence = n[:, 23]
        king_distance = (n[:, 32] - n[:, 34]).abs() + (n[:, 33] - n[:, 35]).abs()  # |Δfile|+|Δrank|
        latent_norm = z.norm(dim=1)
        residual_cov = proj["residual_cov_norm"]
        latent_variance = proj["latent_variance"]
        rank_estimate = proj["nuisance_rank"]
        b = z.shape[0]

        return {
            "logits": logits,
            "nuisance_vector": n,
            "nuisance_features": q,
            "projected_latent": z,
            "trunk_latent": h,
            "residual_cov_norm": residual_cov.expand(b),
            "latent_variance": latent_variance.expand(b),
            "nuisance_rank_estimate": rank_estimate.expand(b),
            "nuisance_energy": nuisance_energy,
            "material_balance": material_balance,
            "material_imbalance": material_imbalance,
            "side_to_move_scalar": side_to_move_scalar,
            "castling_rights_total": castling_total,
            "en_passant_presence": en_passant_presence,
            "king_separation": king_distance,
            "latent_norm": latent_norm,
            "projection_gamma": logits.new_full((b,), float(self.projector.gamma)),
            "ridge_lambda": logits.new_full((b,), float(self.projector.ridge_lambda)),
        }


def build_nuisance_orthogonal_puzzle_bottleneck_from_config(
    config: dict[str, Any],
) -> NuisanceOrthogonalPuzzleNet:
    cfg = dict(config)
    return NuisanceOrthogonalPuzzleNet(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        latent_dim=int(cfg.get("latent_dim", 256)),
        nuisance_rank=int(cfg.get("nuisance_rank", 64)),
        nuisance_expansion_dim=int(cfg.get("nuisance_expansion_dim", 64)),
        ridge_lambda=float(cfg.get("ridge_lambda", 1.0e-3)),
        projection_gamma=float(cfg.get("projection_gamma", 1.0)),
        encoding=str(cfg.get("encoding", SIMPLE_18)),
        fail_closed_semantics=bool(cfg.get("fail_closed_semantics", True)),
        nuisance_seed=int(cfg.get("nuisance_seed", 42)),
    )
