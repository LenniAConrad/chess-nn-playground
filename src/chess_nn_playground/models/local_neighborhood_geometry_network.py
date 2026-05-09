"""Local Neighborhood Geometry Network for idea i124.

The model encodes ``V`` deterministic, board-only perturbations of the input
through a shared encoder, then reads tactical content off the *geometry* of
the resulting embedding cloud: per-view delta norms, pairwise cosine
similarities between deltas, the spectrum of the local covariance, plus the
mean and max pairwise distances.  By the local-sharpness thesis, a
puzzle-like position should have a sharper local response (larger delta
norms, more anisotropic covariance, larger pairwise distances) than a quiet
non-puzzle position because removing one piece plane, masking a square
neighborhood, or zeroing the coordinate planes can flip its tactical
status.

Mechanism: the puzzle head consumes the *identity-view embedding* together
with these geometry diagnostics and returns one puzzle logit.  The
mechanism is materially distinct from any single-view CNN, twin-encoder, or
attention-based architecture: the readout is a fixed multi-view local-geometry
statistic of the encoder output, not a single-view pooled feature or a
learned attention map.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


# Names of the V deterministic views.  Order matters: index 0 is always the
# identity / "center" view that supplies the center embedding.
_VIEW_NAMES: tuple[str, ...] = (
    "identity",
    "horizontal_mirror",
    "mask_corner_quadrant",
    "zero_coordinate_planes",
    "mask_king_neighborhood_ring",
    "piece_type_dropout_group",
)


class _SharedBoardEncoder(nn.Module):
    """Tiny shared CNN stem that lifts a board tensor to a ``D``-dim vector.

    Used for every deterministic view; weights are shared by construction.
    """

    def __init__(
        self,
        input_channels: int,
        channels: int,
        embed_dim: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(max(1, depth)):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    bias=not use_batchnorm,
                )
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = channels
        self.trunk = nn.Sequential(*layers)
        self.project = nn.Linear(2 * channels, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(x)  # (B, C, 8, 8)
        pooled = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)
        return self.project(pooled)


class LocalNeighborhoodGeometryNetwork(nn.Module):
    """Bespoke implementation of idea i124.

    The model is intentionally board-only; CRTK / source metadata is
    reporting-only and never consumed as input.
    """

    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        embed_dim: int = 96,
        num_views: int = 6,
        num_covariance_eigenvalues: int = 4,
        piece_dropout_group_index: int = 0,
        coordinate_plane_start: int = 12,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "LocalNeighborhoodGeometryNetwork supports the puzzle_binary one-logit contract"
            )
        if num_views < 2 or num_views > len(_VIEW_NAMES):
            raise ValueError(f"num_views must be in [2, {len(_VIEW_NAMES)}]")
        if num_covariance_eigenvalues < 1:
            raise ValueError("num_covariance_eigenvalues must be >= 1")
        if embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")
        if input_channels < 13:
            # We need at least the 12 piece planes + side-to-move plane for
            # the deterministic perturbations to be well defined.
            raise ValueError("LocalNeighborhoodGeometryNetwork requires input_channels >= 13")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_views = int(num_views)
        self.embed_dim = int(embed_dim)
        self.num_covariance_eigenvalues = int(num_covariance_eigenvalues)
        self.piece_dropout_group_index = int(piece_dropout_group_index) % 6
        self.coordinate_plane_start = int(coordinate_plane_start)
        self.view_names = _VIEW_NAMES[: self.num_views]

        self.encoder = _SharedBoardEncoder(
            input_channels=input_channels,
            channels=channels,
            embed_dim=embed_dim,
            depth=depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

        # Persistent bookkeeping for the deterministic masks.
        # King-neighborhood ring mask is centered on the board (a 3x3 ring
        # at the centre of the 8x8 board is a deterministic, board-agnostic
        # choice that mirrors the "mask one square neighborhood" prescription).
        ring = torch.ones(8, 8)
        ring[3:6, 3:6] = 0.0  # zero the 3x3 centre region (8 squares + centre)
        self.register_buffer("king_ring_mask", ring, persistent=False)
        # Corner quadrant mask zeroes the (rank>=4, file<4) 4x4 quadrant.
        corner = torch.ones(8, 8)
        corner[4:8, 0:4] = 0.0
        self.register_buffer("corner_mask", corner, persistent=False)

        # Geometry diagnostic dimension:
        # - center embedding (already kept separately)
        # - V-1 delta L2 norms
        # - upper-triangular off-diagonal of cosine matrix between deltas:
        #   (V-1) * (V-2) / 2 entries
        # - top-K eigenvalues of the V x V Gram matrix of centred embeddings
        # - mean pairwise distance, max pairwise distance, mean delta norm,
        #   anisotropy ratio (top eigval / sum eigvals)
        v_minus_one = self.num_views - 1
        cosine_pairs = (v_minus_one * (v_minus_one - 1)) // 2
        geometry_dim = (
            v_minus_one  # delta norms
            + cosine_pairs  # cosine deltas
            + self.num_covariance_eigenvalues  # local covariance spectrum
            + 4  # mean/max pairwise distance, mean delta norm, anisotropy
        )
        self.geometry_dim = int(geometry_dim)

        head_in = embed_dim + geometry_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Deterministic perturbations
    # ------------------------------------------------------------------
    def _identity(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _horizontal_mirror(self, x: torch.Tensor) -> torch.Tensor:
        # File-axis mirror.  The remaining channels are reordered as a
        # *diagnostic* perturbation: chess semantics are not necessarily
        # preserved without a side swap, but the architecture only measures
        # local response (per the packet, "the model is not told that labels
        # are invariant under all perturbations").
        return torch.flip(x, dims=(-1,))

    def _mask_corner_quadrant(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.corner_mask

    def _zero_coordinate_planes(self, x: torch.Tensor) -> torch.Tensor:
        # Zero everything from coordinate_plane_start onward (the simple_18
        # encoder appends 6 metadata / coordinate planes after the 12 piece
        # planes, so this kills auxiliary planes deterministically).
        if x.shape[1] <= self.coordinate_plane_start:
            return x
        out = x.clone()
        out[:, self.coordinate_plane_start :] = 0.0
        return out

    def _mask_king_neighborhood_ring(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.king_ring_mask

    def _piece_type_dropout_group(self, x: torch.Tensor) -> torch.Tensor:
        # Drop one piece-type group deterministically: zero the white and
        # black planes of `piece_dropout_group_index` (one of pawn, knight,
        # bishop, rook, queen, king under the canonical piece order).
        out = x.clone()
        idx = self.piece_dropout_group_index
        out[:, idx] = 0.0
        if x.shape[1] > idx + 6:
            out[:, idx + 6] = 0.0
        return out

    def _apply_view(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name == "identity":
            return self._identity(x)
        if name == "horizontal_mirror":
            return self._horizontal_mirror(x)
        if name == "mask_corner_quadrant":
            return self._mask_corner_quadrant(x)
        if name == "zero_coordinate_planes":
            return self._zero_coordinate_planes(x)
        if name == "mask_king_neighborhood_ring":
            return self._mask_king_neighborhood_ring(x)
        if name == "piece_type_dropout_group":
            return self._piece_type_dropout_group(x)
        raise ValueError(f"Unknown view: {name}")

    # ------------------------------------------------------------------
    # Geometry statistics
    # ------------------------------------------------------------------
    @staticmethod
    def _pairwise_cosine_offdiag(deltas: torch.Tensor) -> torch.Tensor:
        """Return strictly-upper-triangular cosine values of ``deltas``.

        ``deltas`` has shape ``(B, V-1, D)``; the returned tensor is of
        shape ``(B, (V-1)(V-2)/2)``.
        """
        normed = F.normalize(deltas, dim=-1, eps=1.0e-8)
        gram = torch.matmul(normed, normed.transpose(-1, -2))  # (B, V-1, V-1)
        v = gram.shape[-1]
        if v < 2:
            return torch.zeros(gram.shape[0], 0, device=gram.device, dtype=gram.dtype)
        idx = torch.triu_indices(v, v, offset=1, device=gram.device)
        return gram[:, idx[0], idx[1]]

    def _local_covariance_spectrum(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Top-K eigenvalues of the centred V x V Gram matrix per sample.

        We use the centred Gram matrix on the V views (not the D x D
        covariance) because V <= D in the regimes we target, so the
        non-zero spectrum lives in the Gram side.  Eigenvalues are
        sorted descending and zero-padded to ``num_covariance_eigenvalues``.
        """
        bsz, v, d = embeddings.shape
        centred = embeddings - embeddings.mean(dim=1, keepdim=True)
        gram = torch.matmul(centred, centred.transpose(-1, -2)) / max(d, 1)
        # Symmetrise to keep eigvalsh stable under fp32 noise.
        gram = 0.5 * (gram + gram.transpose(-1, -2))
        eigvals = torch.linalg.eigvalsh(gram)  # ascending, (B, V)
        eigvals = torch.flip(eigvals, dims=(-1,))  # descending
        k = min(self.num_covariance_eigenvalues, v)
        top = eigvals[:, :k]
        if k < self.num_covariance_eigenvalues:
            pad = top.new_zeros(bsz, self.num_covariance_eigenvalues - k)
            top = torch.cat([top, pad], dim=-1)
        return top.clamp_min(0.0)

    @staticmethod
    def _pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
        """Return ``(B, V*(V-1)/2)`` pairwise L2 distances."""
        v = embeddings.shape[1]
        if v < 2:
            return torch.zeros(embeddings.shape[0], 0, device=embeddings.device, dtype=embeddings.dtype)
        diff = embeddings.unsqueeze(2) - embeddings.unsqueeze(1)  # (B, V, V, D)
        dists = diff.norm(dim=-1)  # (B, V, V)
        idx = torch.triu_indices(v, v, offset=1, device=embeddings.device)
        return dists[:, idx[0], idx[1]]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        bsz = x.shape[0]

        views = [self._apply_view(name, x) for name in self.view_names]
        # Stacked along the view dim so the encoder is shared by construction.
        stacked = torch.stack(views, dim=1)  # (B, V, C, 8, 8)
        flat = stacked.reshape(bsz * self.num_views, *stacked.shape[2:])
        embeddings = self.encoder(flat).reshape(bsz, self.num_views, self.embed_dim)

        center = embeddings[:, 0]  # (B, D)
        others = embeddings[:, 1:]  # (B, V-1, D)
        deltas = others - center.unsqueeze(1)  # (B, V-1, D)
        delta_norms = deltas.norm(dim=-1)  # (B, V-1)
        cosine_offdiag = self._pairwise_cosine_offdiag(deltas)  # (B, P)
        spectrum = self._local_covariance_spectrum(embeddings)  # (B, K)
        pairwise = self._pairwise_distances(embeddings)  # (B, V*(V-1)/2)
        mean_pairwise = pairwise.mean(dim=-1) if pairwise.numel() > 0 else pairwise.new_zeros(bsz)
        max_pairwise = pairwise.amax(dim=-1) if pairwise.numel() > 0 else pairwise.new_zeros(bsz)
        mean_delta_norm = delta_norms.mean(dim=-1)
        spectrum_sum = spectrum.sum(dim=-1).clamp_min(1.0e-8)
        anisotropy = spectrum[:, 0] / spectrum_sum

        geometry = torch.cat(
            [
                delta_norms,
                cosine_offdiag,
                spectrum,
                mean_pairwise.unsqueeze(-1),
                max_pairwise.unsqueeze(-1),
                mean_delta_norm.unsqueeze(-1),
                anisotropy.unsqueeze(-1),
            ],
            dim=-1,
        )
        feat_vec = torch.cat([center, geometry], dim=-1)
        logits = self.head(feat_vec).view(-1)

        return {
            "logits": logits,
            "lng_center_embedding": center,
            "lng_view_embeddings": embeddings,
            "lng_view_deltas": deltas,
            "lng_delta_norms": delta_norms,
            "lng_cosine_delta_offdiag": cosine_offdiag,
            "lng_local_covariance_spectrum": spectrum,
            "lng_pairwise_distances": pairwise,
            "lng_mean_pairwise_distance": mean_pairwise,
            "lng_max_pairwise_distance": max_pairwise,
            "lng_mean_delta_norm": mean_delta_norm,
            "lng_anisotropy_ratio": anisotropy,
        }


def build_local_neighborhood_geometry_network_from_config(
    config: dict[str, Any],
) -> LocalNeighborhoodGeometryNetwork:
    cfg = dict(config)
    return LocalNeighborhoodGeometryNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        embed_dim=int(cfg.get("embed_dim", 96)),
        num_views=int(cfg.get("num_views", 6)),
        num_covariance_eigenvalues=int(cfg.get("num_covariance_eigenvalues", 4)),
        piece_dropout_group_index=int(cfg.get("piece_dropout_group_index", 0)),
        coordinate_plane_start=int(cfg.get("coordinate_plane_start", 12)),
        num_classes=int(cfg.get("num_classes", 1)),
    )
