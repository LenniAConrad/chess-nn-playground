"""Harmonic Board Potential Network model for idea i059.

Implements the markdown thesis (`ideas/all_ideas/registry/i059_harmonic_board_potential_network/`):
puzzle-likeness is tested by computing fixed inverse-Laplacian board potentials
over learned current-board charge maps, then feeding charge/potential energy
and Dirichlet/flux summaries to an MLP head. The central operator is

    u_{k,l}(x) = (L + lambda_l * I)^{-1} rho_k(x)
    E_{k,l}    = rho_k^T u_{k,l}
    D_{k,l}    = sum_{(a,b) in edges} (u_{k,l}(a) - u_{k,l}(b))^2 = u^T L u

where ``L`` is the fixed 8x8 grid Laplacian with Neumann boundary and
``rho_k`` are signed charge maps emitted by a 1x1 convolution over the
``simple_18`` board tensor. The Green matrices ``G_l = (L + lambda_l I)^{-1}``
are precomputed on construction and stored as non-trainable buffers, so the
solver is a fixed global linear operator and no learned message passing is
involved.

Forward pipeline:

    Simple18ChargeEncoder    ->  (B, K, 8, 8) signed charges rho
    FixedBoardPoissonSolver  ->  (B, K, L, 8, 8) potentials u
    PotentialStatsPool       ->  (B, K * L * S) energy/Dirichlet/flux/king-ring
    HarmonicPotentialHead    ->  (B,) puzzle logit + diagnostics

Section 9 of the markdown packet identifies three ablations exposed via
``ablation``:

    * ``"none"``                 -- main model.
    * ``"random_orthogonal_solver"`` -- replace each Green matrix with a
      fixed deterministic orthogonal matrix of the same shape and similar
      variance; this destroys the harmonic distance law while preserving
      the global linear-projection structure.
    * ``"local_gaussian_solver"`` -- replace each Green matrix with a fixed
      isotropic Gaussian blur kernel of comparable spatial scale; this keeps
      smoothing but removes the inverse-Laplacian long-range coupling.
    * ``"charge_only_stats"``    -- bypass the solver entirely so only
      charge moments reach the head.

The solver-replacement ablations operate on the same ``(N, N)`` matrix
contract, so the head's input dimensionality is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models._packet_bespoke_base import (
    BoardTensorSpec,
    require_board_tensor,
)


_BOARD = 8
_BOARD_AREA = _BOARD * _BOARD  # 64
_PIECE_PLANES = 12
_DEFAULT_LAMBDAS: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0)
_VALID_ABLATIONS = {
    "none",
    "random_orthogonal_solver",
    "local_gaussian_solver",
    "charge_only_stats",
}
_VALID_BOUNDARIES = {"neumann", "dirichlet"}


def _build_grid_laplacian(boundary: str) -> torch.Tensor:
    """Return the 64x64 discrete grid Laplacian with the chosen boundary.

    Neumann (zero-flux) boundary: a node's degree equals its number of
    in-board neighbors, so corners have degree 2, edges 3, interior 4.
    Dirichlet boundary: every node has degree 4 (boundary "ghost" nodes
    are implicitly zero), which makes ``L`` strictly positive definite
    even for ``lambda = 0``.
    """
    if boundary not in _VALID_BOUNDARIES:
        raise ValueError(f"Unsupported boundary {boundary!r}; expected one of {sorted(_VALID_BOUNDARIES)}")
    n = _BOARD_AREA
    adjacency = torch.zeros(n, n, dtype=torch.float64)
    for r in range(_BOARD):
        for c in range(_BOARD):
            i = r * _BOARD + c
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < _BOARD and 0 <= cc < _BOARD:
                    j = rr * _BOARD + cc
                    adjacency[i, j] = 1.0
    if boundary == "neumann":
        degree = adjacency.sum(dim=-1)
    else:  # dirichlet
        degree = torch.full((n,), 4.0, dtype=torch.float64)
    laplacian = torch.diag(degree) - adjacency
    return laplacian


def _build_green_matrices(boundary: str, lambdas: tuple[float, ...]) -> torch.Tensor:
    """Return ``(L_count, 64, 64)`` Green matrices ``G_l = (L + lambda_l I)^{-1}``."""
    if not lambdas:
        raise ValueError("at least one screening constant lambda is required")
    laplacian = _build_grid_laplacian(boundary)
    eye = torch.eye(_BOARD_AREA, dtype=torch.float64)
    matrices = []
    for lam in lambdas:
        if lam <= 0.0 and boundary == "neumann":
            raise ValueError(
                f"Neumann Laplacian is rank-deficient; lambda must be > 0, got {lam}"
            )
        # Solve (L + lam * I) X = I for X = G_l. ``torch.linalg.solve`` is
        # numerically more stable than explicit matrix inversion.
        green = torch.linalg.solve(laplacian + lam * eye, eye)
        matrices.append(green)
    return torch.stack(matrices, dim=0).to(dtype=torch.float32)


def _build_random_orthogonal_solver(lambdas: tuple[float, ...], laplacian: torch.Tensor) -> torch.Tensor:
    """Return ``(L_count, 64, 64)`` deterministic orthogonal matrices with similar variance.

    Section 9 falsifier: replace each Green matrix with a fixed orthogonal
    matrix scaled so its Frobenius norm matches the corresponding Green
    matrix. The orthogonal seed is deterministic (per-lambda) so the
    architecture is reproducible.
    """
    matrices = []
    eye = torch.eye(_BOARD_AREA, dtype=torch.float64)
    for idx, lam in enumerate(lambdas):
        green = torch.linalg.solve(laplacian + lam * eye, eye)
        target_scale = float(green.norm().item())
        gen = torch.Generator(device="cpu").manual_seed(0xA59 + idx)
        a = torch.randn(_BOARD_AREA, _BOARD_AREA, generator=gen, dtype=torch.float64)
        # QR yields an orthogonal matrix Q with ||Q||_F = sqrt(n).
        q, _ = torch.linalg.qr(a)
        q = q * (target_scale / math.sqrt(_BOARD_AREA))
        matrices.append(q)
    return torch.stack(matrices, dim=0).to(dtype=torch.float32)


def _build_local_gaussian_solver(lambdas: tuple[float, ...]) -> torch.Tensor:
    """Return ``(L_count, 64, 64)`` Gaussian blur matrices.

    Each lambda maps to a sigma chosen so the spatial scale roughly matches
    the Green matrix screening length (smaller lambda = longer range).
    Sigma is computed from ``sigma_l = 1 / sqrt(lambda_l)`` then clamped so
    the kernel fits inside the board.
    """
    matrices = []
    coords = torch.arange(_BOARD, dtype=torch.float64)
    rr = coords.view(_BOARD, 1).expand(_BOARD, _BOARD)
    cc = coords.view(1, _BOARD).expand(_BOARD, _BOARD)
    flat_r = rr.reshape(-1)
    flat_c = cc.reshape(-1)
    for lam in lambdas:
        sigma = float(min(max(1.0 / math.sqrt(max(lam, 1.0e-6)), 0.5), 6.0))
        # blur[i, j] = exp(-||p_i - p_j||^2 / (2 sigma^2)) / Z_i
        d2 = (flat_r.unsqueeze(0) - flat_r.unsqueeze(1)) ** 2 + (
            flat_c.unsqueeze(0) - flat_c.unsqueeze(1)
        ) ** 2
        kernel = torch.exp(-d2 / (2.0 * sigma * sigma))
        # Row-normalize so it acts like a smoothing operator.
        kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        matrices.append(kernel)
    return torch.stack(matrices, dim=0).to(dtype=torch.float32)


@dataclass(frozen=True)
class HarmonicGlobals:
    side_to_move_white: torch.Tensor  # (B,)
    castling: torch.Tensor             # (B, 4)
    en_passant_file: torch.Tensor      # (B, 8)
    king_ring_us: torch.Tensor         # (B, 64) flat 0/1 mask of own king ring
    king_ring_them: torch.Tensor       # (B, 64) flat 0/1 mask of opponent king ring


class Simple18ChargeEncoder(nn.Module):
    """1x1 convolution that maps the simple_18 tensor to ``K`` signed charge maps.

    Optionally subtracts each charge map's spatial mean so the model cannot
    use a trivial total-charge shortcut that would already be captured by a
    constant potential.
    """

    def __init__(
        self,
        input_channels: int = 18,
        charge_channels: int = 12,
        mean_center: bool = True,
    ) -> None:
        super().__init__()
        if input_channels < 13:
            raise ValueError(
                f"Simple18ChargeEncoder requires at least 13 input channels (simple_18-style), got {input_channels}"
            )
        if charge_channels < 1:
            raise ValueError("charge_channels must be >= 1")
        self.input_channels = int(input_channels)
        self.charge_channels = int(charge_channels)
        self.mean_center = bool(mean_center)
        self.conv = nn.Conv2d(self.input_channels, self.charge_channels, kernel_size=1, bias=True)
        self.spec = BoardTensorSpec(input_channels=self.input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        require_board_tensor(x, self.spec)
        rho = self.conv(x)  # (B, K, 8, 8)
        if self.mean_center:
            rho = rho - rho.mean(dim=(-1, -2), keepdim=True)
        return rho


class FixedBoardPoissonSolver(nn.Module):
    """Apply precomputed Green matrices to flattened charge maps.

    The ``green_matrices`` buffer has shape ``(L, 64, 64)``; for each lambda
    it stores the dense linear operator that solves ``(L + lambda I) u = rho``
    on the 8x8 grid. This module is parameter-free.
    """

    def __init__(
        self,
        boundary: str = "neumann",
        lambdas: tuple[float, ...] = _DEFAULT_LAMBDAS,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if not lambdas:
            raise ValueError("at least one lambda is required")
        ablation = (ablation or "none").lower()
        if ablation not in _VALID_ABLATIONS:
            raise ValueError(
                f"Unsupported harmonic ablation {ablation!r}; expected one of {sorted(_VALID_ABLATIONS)}"
            )
        self.boundary = str(boundary)
        self.lambdas: tuple[float, ...] = tuple(float(lam) for lam in lambdas)
        self.ablation = ablation

        # Always build the harmonic Green matrices; they are needed for the
        # variance scaling of the random-orthogonal ablation, and inexpensive.
        green_harmonic = _build_green_matrices(self.boundary, self.lambdas)
        if ablation == "random_orthogonal_solver":
            laplacian = _build_grid_laplacian(self.boundary)
            green = _build_random_orthogonal_solver(self.lambdas, laplacian)
        elif ablation == "local_gaussian_solver":
            green = _build_local_gaussian_solver(self.lambdas)
        else:
            green = green_harmonic
        self.register_buffer("green_matrices", green, persistent=False)
        self.register_buffer("green_harmonic", green_harmonic, persistent=False)

    @property
    def num_lambdas(self) -> int:
        return len(self.lambdas)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        # rho: (B, K, 8, 8) -> u: (B, K, L, 8, 8)
        if self.ablation == "charge_only_stats":
            # Return zero potentials of the expected shape; PotentialStatsPool
            # will still expose charge-only moments as the model's signal.
            batch, k = rho.shape[0], rho.shape[1]
            return rho.new_zeros(batch, k, self.num_lambdas, _BOARD, _BOARD)
        batch, k, h, w = rho.shape
        rho_flat = rho.reshape(batch, k, h * w)  # (B, K, 64)
        # Apply each Green matrix: u_{l,*,*} = G_l @ rho_flat
        # einsum: ``lij, bkj -> bkli`` gives (B, K, L, 64)
        u_flat = torch.einsum("lij,bkj->bkli", self.green_matrices, rho_flat)
        u = u_flat.reshape(batch, k, self.num_lambdas, h, w)
        return u


class PotentialStatsPool(nn.Module):
    """Compute charge/potential energy, Dirichlet energy, flux, and king-ring stats.

    Per (charge ``k``, lambda ``l``) the pool emits ``S = 11`` features:

        0  charge-potential energy ``rho^T u``
        1  Dirichlet energy ``sum_{(a,b)} (u_a - u_b)^2``
        2  potential mean
        3  potential std
        4  potential max
        5  potential min
        6  abs-max potential
        7  boundary flux (``mean_{boundary} u``)
        8  mean potential over the side-to-move's king ring
        9  mean potential over the opponent's king ring
        10 charge-magnitude (``mean(|rho|)``) for context

    The stats tensor has shape ``(B, K, L, S)``. Boundary indicator and
    quadrant indicators are cached as buffers so they live on the right
    device after ``.to(device)``.
    """

    def __init__(self) -> None:
        super().__init__()
        # Boundary mask: 1 on board edge, 0 inside.
        boundary = torch.zeros(_BOARD, _BOARD, dtype=torch.float32)
        boundary[0, :] = 1.0
        boundary[-1, :] = 1.0
        boundary[:, 0] = 1.0
        boundary[:, -1] = 1.0
        self.register_buffer("boundary_mask", boundary.reshape(_BOARD_AREA), persistent=False)
        # Edge index pairs for Dirichlet energy. Each (a, b) pair appears once.
        edges_a: list[int] = []
        edges_b: list[int] = []
        for r in range(_BOARD):
            for c in range(_BOARD):
                i = r * _BOARD + c
                if c + 1 < _BOARD:
                    edges_a.append(i)
                    edges_b.append(r * _BOARD + (c + 1))
                if r + 1 < _BOARD:
                    edges_a.append(i)
                    edges_b.append((r + 1) * _BOARD + c)
        self.register_buffer("edge_a", torch.tensor(edges_a, dtype=torch.long), persistent=False)
        self.register_buffer("edge_b", torch.tensor(edges_b, dtype=torch.long), persistent=False)
        self.stats_per_pair = 11

    def forward(
        self,
        rho: torch.Tensor,            # (B, K, 8, 8)
        u: torch.Tensor,              # (B, K, L, 8, 8)
        king_ring_us: torch.Tensor,   # (B, 64)
        king_ring_them: torch.Tensor, # (B, 64)
    ) -> torch.Tensor:
        batch, k, h, w = rho.shape
        num_l = u.shape[2]

        rho_flat = rho.reshape(batch, k, _BOARD_AREA)               # (B, K, 64)
        u_flat = u.reshape(batch, k, num_l, _BOARD_AREA)            # (B, K, L, 64)

        # 0. charge-potential energy: rho^T u
        energy = (rho_flat.unsqueeze(2) * u_flat).sum(dim=-1)        # (B, K, L)

        # 1. Dirichlet energy
        u_a = u_flat[..., self.edge_a]                                # (B, K, L, |E|)
        u_b = u_flat[..., self.edge_b]
        dirichlet = ((u_a - u_b) ** 2).sum(dim=-1)                    # (B, K, L)

        # 2-5. potential moments
        u_mean = u_flat.mean(dim=-1)                                  # (B, K, L)
        u_std = u_flat.std(dim=-1, unbiased=False)
        u_max = u_flat.amax(dim=-1)
        u_min = u_flat.amin(dim=-1)
        u_absmax = u_flat.abs().amax(dim=-1)

        # 7. boundary flux: mean of u on boundary squares
        boundary = self.boundary_mask.view(1, 1, 1, _BOARD_AREA)
        boundary_count = boundary.sum().clamp_min(1.0)
        boundary_flux = (u_flat * boundary).sum(dim=-1) / boundary_count

        # 8/9. king-ring mean potentials
        ring_us = king_ring_us.view(batch, 1, 1, _BOARD_AREA)
        ring_them = king_ring_them.view(batch, 1, 1, _BOARD_AREA)
        ring_us_mass = ring_us.sum(dim=-1).clamp_min(1.0)
        ring_them_mass = ring_them.sum(dim=-1).clamp_min(1.0)
        king_us_pot = (u_flat * ring_us).sum(dim=-1) / ring_us_mass
        king_them_pot = (u_flat * ring_them).sum(dim=-1) / ring_them_mass

        # 10. charge magnitude (broadcast across L for layout consistency)
        rho_mag = rho_flat.abs().mean(dim=-1, keepdim=True).expand(-1, -1, num_l)

        stats = torch.stack(
            [
                energy,
                dirichlet,
                u_mean,
                u_std,
                u_max,
                u_min,
                u_absmax,
                boundary_flux,
                king_us_pot,
                king_them_pot,
                rho_mag,
            ],
            dim=-1,
        )  # (B, K, L, S)
        return stats


class HarmonicPotentialHead(nn.Module):
    """Two-layer MLP head consuming pooled stats and global broadcast features."""

    def __init__(
        self,
        feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        in_dim = int(feature_dim + global_feature_dim)
        self.in_dim = in_dim
        layers: list[nn.Module] = [
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, max(1, int(num_classes))))
        self.mlp = nn.Sequential(*layers)
        self.num_classes = int(num_classes)

    def forward(self, features: torch.Tensor, globals_tensor: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([features, globals_tensor], dim=-1))


class HarmonicBoardPotentialNet(nn.Module):
    """Complete bespoke architecture for idea i059."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        charge_channels: int = 12,
        lambdas: tuple[float, ...] = _DEFAULT_LAMBDAS,
        boundary: str = "neumann",
        head_hidden: int = 128,
        mean_center_charges: bool = True,
        ablation: str = "none",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.charge_channels = int(charge_channels)
        self.lambdas: tuple[float, ...] = tuple(float(lam) for lam in lambdas)
        self.boundary = str(boundary)
        self.ablation = (ablation or "none").lower()
        if self.ablation not in _VALID_ABLATIONS:
            raise ValueError(
                f"Unsupported harmonic ablation {self.ablation!r}; expected one of {sorted(_VALID_ABLATIONS)}"
            )

        self.charge_encoder = Simple18ChargeEncoder(
            input_channels=self.input_channels,
            charge_channels=self.charge_channels,
            mean_center=mean_center_charges,
        )
        self.solver = FixedBoardPoissonSolver(
            boundary=self.boundary,
            lambdas=self.lambdas,
            ablation=self.ablation,
        )
        self.stats_pool = PotentialStatsPool()
        self.feature_dim = self.charge_channels * len(self.lambdas) * self.stats_pool.stats_per_pair

        # Globals: side-to-move (1) + castling (4) + EP file (8) + king-ring sizes (2) = 15
        self.global_feature_dim = 1 + 4 + 8 + 2

        self.head = HarmonicPotentialHead(
            feature_dim=self.feature_dim,
            global_feature_dim=self.global_feature_dim,
            hidden_dim=int(head_hidden),
            num_classes=self.num_classes,
            dropout=float(dropout),
        )

    @staticmethod
    def _king_ring_mask(king_plane: torch.Tensor) -> torch.Tensor:
        """Dilate a (B, 8, 8) king plane by one square to a (B, 64) ring mask.

        If no king is present (e.g. malformed input) the mask is all zeros and
        the consuming code falls back to a constant-1 normalizer so no NaN
        occurs in the ring-potential mean.
        """
        # Use a 3x3 dilation via max pooling so the king square plus its eight
        # neighbors form the ring; clamped to {0, 1}.
        kp = king_plane.unsqueeze(1)  # (B, 1, 8, 8)
        dilated = F.max_pool2d(kp, kernel_size=3, stride=1, padding=1)
        return dilated.clamp(0.0, 1.0).reshape(king_plane.shape[0], _BOARD_AREA)

    def _harmonic_globals(self, x: torch.Tensor) -> HarmonicGlobals:
        # side-to-move plane (channel 12 in simple_18) -> scalar 1 if white-to-move
        side_plane = x[:, 12].clamp(0.0, 1.0)
        side_white = (side_plane.mean(dim=(-1, -2)) > 0.5).to(x.dtype)  # (B,)
        castling = torch.stack(
            [
                x[:, 13].mean(dim=(-1, -2)),
                x[:, 14].mean(dim=(-1, -2)),
                x[:, 15].mean(dim=(-1, -2)),
                x[:, 16].mean(dim=(-1, -2)),
            ],
            dim=-1,
        ).clamp(0.0, 1.0)
        ep_plane = x[:, 17].clamp(0.0, 1.0)
        ep_files = ep_plane.amax(dim=-2)  # (B, 8)

        # White king is plane 5, black king is plane 11.
        white_king = x[:, 5].clamp(0.0, 1.0)
        black_king = x[:, 11].clamp(0.0, 1.0)
        side = side_white.view(-1, 1, 1)
        own_king = side * white_king + (1.0 - side) * black_king
        opp_king = side * black_king + (1.0 - side) * white_king
        ring_us = self._king_ring_mask(own_king)
        ring_them = self._king_ring_mask(opp_king)
        return HarmonicGlobals(
            side_to_move_white=side_white,
            castling=castling,
            en_passant_file=ep_files,
            king_ring_us=ring_us,
            king_ring_them=ring_them,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        require_board_tensor(x, BoardTensorSpec(input_channels=self.input_channels))
        x = x.float()
        globals_data = self._harmonic_globals(x)
        rho = self.charge_encoder(x)
        u = self.solver(rho)
        stats = self.stats_pool(
            rho=rho,
            u=u,
            king_ring_us=globals_data.king_ring_us,
            king_ring_them=globals_data.king_ring_them,
        )

        batch = stats.shape[0]
        flat_stats = stats.reshape(batch, -1)
        ring_size_us = globals_data.king_ring_us.sum(dim=-1, keepdim=True) / float(_BOARD_AREA)
        ring_size_them = globals_data.king_ring_them.sum(dim=-1, keepdim=True) / float(_BOARD_AREA)
        global_features = torch.cat(
            [
                globals_data.side_to_move_white.unsqueeze(-1),
                globals_data.castling,
                globals_data.en_passant_file,
                ring_size_us,
                ring_size_them,
            ],
            dim=-1,
        )

        raw_logits = self.head(flat_stats, global_features)
        if self.num_classes == 1:
            logits = raw_logits.view(-1)
            two_class = torch.stack([-0.5 * logits, 0.5 * logits], dim=-1)
        else:
            logits = raw_logits
            two_class = raw_logits if raw_logits.shape[-1] >= 2 else logits

        # Per (k, l) named diagnostics.
        # stats layout: dim=-1 is [energy, dirichlet, u_mean, u_std, u_max, u_min,
        #                          u_absmax, boundary_flux, king_us, king_them, rho_mag]
        energy = stats[..., 0]                # (B, K, L)
        dirichlet = stats[..., 1]
        u_mean = stats[..., 2]
        u_std = stats[..., 3]
        u_absmax = stats[..., 6]
        boundary_flux = stats[..., 7]
        king_us_pot = stats[..., 8]
        king_them_pot = stats[..., 9]
        rho_mag = stats[..., 10]
        mechanism_energy = energy.abs().mean(dim=(-1, -2))

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "two_class_logits": two_class,
            "charge_potential_energy": energy,
            "charge_potential_energy_mean": energy.mean(dim=(-1, -2)),
            "dirichlet_energy": dirichlet,
            "dirichlet_energy_mean": dirichlet.mean(dim=(-1, -2)),
            "potential_mean": u_mean,
            "potential_std": u_std,
            "potential_absmax": u_absmax,
            "boundary_flux": boundary_flux,
            "king_us_potential": king_us_pot,
            "king_them_potential": king_them_pot,
            "charge_magnitude": rho_mag,
            "mechanism_energy": mechanism_energy,
            "ablation_active": torch.full(
                (logits.shape[0],),
                1.0 if self.ablation != "none" else 0.0,
                device=logits.device,
                dtype=logits.dtype,
            ),
        }
        return diagnostics


def build_harmonic_board_potential_network_from_config(
    config: dict[str, Any],
) -> HarmonicBoardPotentialNet:
    cfg = dict(config)
    lambdas_raw = cfg.get("lambdas", _DEFAULT_LAMBDAS)
    if isinstance(lambdas_raw, (list, tuple)):
        lambdas = tuple(float(x) for x in lambdas_raw)
    else:
        raise ValueError("lambdas must be a list/tuple of floats")
    head_hidden = cfg.get("head_hidden", cfg.get("hidden_dim", 128))
    return HarmonicBoardPotentialNet(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        charge_channels=int(cfg.get("charge_channels", 12)),
        lambdas=lambdas,
        boundary=str(cfg.get("boundary", "neumann")),
        head_hidden=int(head_hidden),
        mean_center_charges=bool(cfg.get("mean_center_charges", True)),
        ablation=str(cfg.get("ablation", "none")),
        dropout=float(cfg.get("dropout", 0.0)),
    )
