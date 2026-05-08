"""Source-Invariant Puzzle Bottleneck for idea i196.

The dataset has three source groups; a board-only model can accidentally learn
source artifacts instead of puzzle structure. This bespoke implementation
forces the main representation to be invariant under a fixed symmetry orbit of
board transformations (identity, file flip, rank flip, 180-rotation). The
mean-across-orbit code is the puzzle representation; per-view deviations from
that mean are the "source residual" component, which is exposed as a
diagnostic and (optionally) explicitly subtracted from the bottleneck.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


VIEW_NAMES: tuple[str, ...] = ("identity", "file_flip", "rank_flip", "rotate_180")


def _apply_view(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "identity":
        return x
    if name == "file_flip":
        return torch.flip(x, dims=[3])
    if name == "rank_flip":
        return torch.flip(x, dims=[2])
    if name == "rotate_180":
        return torch.flip(x, dims=[2, 3])
    raise ValueError(f"Unknown symmetry view: {name!r}")


class BoardFeatureTrunk(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if int(depth) < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = []
        in_c = int(input_channels)
        for _ in range(int(depth)):
            layers.append(nn.Conv2d(in_c, int(channels), kernel_size=3, padding=1, bias=not use_batchnorm))
            layers.append(nn.BatchNorm2d(int(channels)) if use_batchnorm else nn.GroupNorm(1, int(channels)))
            layers.append(nn.GELU())
            if float(dropout) > 0:
                layers.append(nn.Dropout2d(float(dropout)))
            in_c = int(channels)
        self.stack = nn.Sequential(*layers)
        self.output_channels = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


class SymmetryOrbitEncoder(nn.Module):
    """Applies a shared trunk to a fixed list of symmetry views and returns
    per-view pooled features of shape ``(batch, num_views, 2 * channels)``.

    Pooling is the concatenation of mean and max pooling over the spatial
    dimensions, matching the standard puzzle_binary trunk pool used elsewhere
    in this codebase.
    """

    def __init__(self, trunk: BoardFeatureTrunk, view_names: tuple[str, ...]) -> None:
        super().__init__()
        if not view_names:
            raise ValueError("view_names must contain at least one symmetry view")
        for name in view_names:
            if name not in VIEW_NAMES:
                raise ValueError(f"Unknown view {name!r}; expected one of {VIEW_NAMES}")
        self.trunk = trunk
        self.view_names = tuple(view_names)
        self.num_views = len(self.view_names)
        self.feature_dim = 2 * self.trunk.output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        per_view: list[torch.Tensor] = []
        for name in self.view_names:
            view = _apply_view(x, name)
            features = self.trunk(view)
            pooled = torch.cat([features.mean(dim=(2, 3)), features.amax(dim=(2, 3))], dim=1)
            per_view.append(pooled)
        return torch.stack(per_view, dim=1)


class InvariantBottleneck(nn.Module):
    """Orthogonalises the mean-across-views code against per-view residuals.

    Given per-view pooled features ``v_k`` (k = 1..K), the bottleneck:
      1. Projects each ``v_k`` through a shared MLP into a code ``c_k``.
      2. Computes the mean code ``c_bar = mean_k c_k`` (the symmetry-orbit
         invariant component).
      3. Computes residuals ``r_k = c_k - c_bar``.
      4. Subtracts a learned scalar gate times the residual energy direction
         from ``c_bar`` to yield the "source-purified" main code:
            ``c_main = c_bar - sigmoid(orthogonalize_logit) * residual_energy_direction``.
         The direction is the L2-normalised residual sum, so this implements a
         smooth, differentiable Gram-Schmidt-style suppression of any
         consistent residual axis.
    """

    def __init__(self, feature_dim: int, code_dim: int, dropout: float) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.code_dim = int(code_dim)
        self.code_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.code_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(self.code_dim, self.code_dim),
        )
        self.orthogonalize_logit = nn.Parameter(torch.zeros(()))

    def forward(self, per_view: torch.Tensor, *, orthogonalize: bool = True) -> dict[str, torch.Tensor]:
        batch, num_views, _ = per_view.shape
        codes = self.code_proj(per_view.reshape(batch * num_views, -1)).reshape(batch, num_views, self.code_dim)
        invariant = codes.mean(dim=1)
        residuals = codes - invariant.unsqueeze(1)
        residual_sum = residuals.sum(dim=1)
        residual_sq_norm = residual_sum.pow(2).sum(dim=1, keepdim=True)
        residual_energy = residuals.pow(2).mean(dim=(1, 2))

        gate = torch.sigmoid(self.orthogonalize_logit)
        # Numerically-stable Gram-Schmidt projection: when residual_sum is
        # near zero (e.g. all views produce identical codes) the projection
        # coefficient `<c, r_sum> / (||r_sum||^2 + eps)` smoothly goes to
        # zero, avoiding noise amplification from a tiny normalised direction.
        inner = (invariant * residual_sum).sum(dim=1, keepdim=True)
        projection_coef = inner / (residual_sq_norm + 1.0e-6)
        if orthogonalize:
            main_code = invariant - gate * projection_coef * residual_sum
        else:
            main_code = invariant
        return {
            "main_code": main_code,
            "invariant_code": invariant,
            "residual_codes": residuals,
            "residual_energy": residual_energy,
            "residual_direction_strength": residual_sq_norm.clamp_min(0.0).sqrt().view(-1),
            "orthogonalize_gate": gate.expand(batch),
        }


class SourceInvariantPuzzleBottleneck(nn.Module):
    """Bespoke source-invariant puzzle bottleneck.

    The puzzle logit reads only the symmetry-orbit invariant bottleneck code,
    so any signal that lives purely in the per-view residual subspace (a proxy
    for source-specific board artifacts) is removed from the main prediction
    path. The per-view residual energy is exposed as both a diagnostic and a
    separate auxiliary logit head, available for downstream regularisation.

    Supported ablations:
      - ``"none"`` — full network as above.
      - ``"no_invariance"`` — drop the symmetry orbit and use only the
        identity view (tests whether multi-view averaging matters).
      - ``"no_orthogonalization"`` — keep the orbit but skip the explicit
        residual-direction subtraction (tests whether mean-pooling alone is
        enough for invariance).
      - ``"no_aux_residual_logit"`` — drop the auxiliary residual logit head
        entirely from the diagnostics.
    """

    ALLOWED_ABLATIONS = (
        "none",
        "no_invariance",
        "no_orthogonalization",
        "no_aux_residual_logit",
    )

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        code_dim: int = 64,
        view_names: tuple[str, ...] = VIEW_NAMES,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "SourceInvariantPuzzleBottleneck supports the puzzle_binary one-logit contract"
            )
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}"
            )

        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.code_dim = int(code_dim)
        self.ablation = str(ablation)

        active_views = ("identity",) if self.ablation == "no_invariance" else tuple(view_names)
        self.view_names = active_views

        self.trunk = BoardFeatureTrunk(
            input_channels=int(input_channels),
            channels=self.channels,
            depth=self.depth,
            dropout=float(dropout),
            use_batchnorm=bool(use_batchnorm),
        )
        self.orbit = SymmetryOrbitEncoder(self.trunk, active_views)
        self.bottleneck = InvariantBottleneck(
            feature_dim=self.orbit.feature_dim,
            code_dim=self.code_dim,
            dropout=float(dropout),
        )

        head_in = self.code_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, max(16, self.hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(max(16, self.hidden_dim), 1),
        )
        self.aux_residual_head = nn.Sequential(
            nn.LayerNorm(self.code_dim),
            nn.Linear(self.code_dim, max(8, self.hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(8, self.hidden_dim // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        per_view = self.orbit(x)
        bottleneck = self.bottleneck(
            per_view,
            orthogonalize=self.ablation != "no_orthogonalization",
        )
        main_code = bottleneck["main_code"]
        invariant_code = bottleneck["invariant_code"]
        residuals = bottleneck["residual_codes"]

        logits = self.head(main_code).view(-1)

        residual_summary = residuals.mean(dim=1)
        if self.ablation == "no_aux_residual_logit":
            aux_residual_logit = logits.new_zeros(logits.shape[0])
        else:
            aux_residual_logit = self.aux_residual_head(residual_summary).view(-1)

        per_view_norm = per_view.norm(dim=2)
        if per_view_norm.shape[1] >= 2:
            view_consistency = -per_view_norm.std(dim=1, unbiased=False)
        else:
            view_consistency = per_view_norm.new_zeros(per_view_norm.shape[0])

        diagnostics = {
            "logits": logits,
            "invariant_code_norm": invariant_code.norm(dim=1),
            "main_code_norm": main_code.norm(dim=1),
            "residual_energy": bottleneck["residual_energy"],
            "residual_direction_strength": bottleneck["residual_direction_strength"],
            "orthogonalize_gate": bottleneck["orthogonalize_gate"],
            "aux_residual_logit": aux_residual_logit,
            "view_consistency": view_consistency,
            "num_views": logits.new_full((logits.shape[0],), float(self.orbit.num_views)),
            "mechanism_energy": bottleneck["residual_energy"],
            "symmetry_residual": bottleneck["residual_energy"],
            "proposal_profile_strength": invariant_code.norm(dim=1),
            "proposal_keyword_count": logits.new_full(
                (logits.shape[0],), float(self.orbit.num_views)
            ),
        }
        return diagnostics


def build_source_invariant_puzzle_bottleneck_from_config(
    config: dict[str, Any],
) -> SourceInvariantPuzzleBottleneck:
    cfg = dict(config)
    raw_views = cfg.get("view_names")
    if raw_views is None:
        view_names = VIEW_NAMES
    else:
        view_names = tuple(str(name) for name in raw_views)
    return SourceInvariantPuzzleBottleneck(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        code_dim=int(cfg.get("code_dim", 64)),
        view_names=view_names,
        ablation=str(cfg.get("ablation", "none")),
    )
