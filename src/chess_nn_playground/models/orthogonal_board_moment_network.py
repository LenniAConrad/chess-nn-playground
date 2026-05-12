"""Orthogonal Board Moment Network (idea i133).

Bespoke implementation of the orthogonal-polynomial board-moment architecture
described in
``ideas/research/packets/classic/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md``.

A small CNN trunk projects the simple_18 board tensor into ``num_fields``
learned scalar fields. Fixed Legendre and Chebyshev polynomial bases over
normalised board coordinates ``u, v in [-1, 1]`` are evaluated at the eight
file/rank centres, and tensor products yield the moment tensor

    m_{family, c, i, j} = sum_{u, v} F_c(u, v) * basis_i(u) * basis_j(v)

for ``i, j in 0..max_degree-1``. The resulting moments are split into low,
middle and high total-degree groups; group-level dropout (``degree dropout``)
randomly masks degree groups during training. A small MLP reads the grouped
moment tensor, fuses it with the pooled CNN summary, and emits a single
``puzzle_binary`` logit alongside per-degree and per-family diagnostics.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.idea_blocks import BoardConvStem, BoardTensorSpec, require_board_tensor


_FAMILIES = ("legendre", "chebyshev")
_LOW_DEGREE_MAX = 1
_MID_DEGREE_MAX = 3


def _legendre_basis(coords: torch.Tensor, max_degree: int) -> torch.Tensor:
    if max_degree < 1:
        raise ValueError("max_degree must be >= 1")
    cols = [torch.ones_like(coords)]
    if max_degree >= 2:
        cols.append(coords)
    for n in range(2, max_degree):
        prev2 = cols[n - 2]
        prev1 = cols[n - 1]
        nm1 = float(n - 1)
        nf = float(n)
        cols.append(((2.0 * nm1 + 1.0) * coords * prev1 - nm1 * prev2) / nf)
    return torch.stack(cols, dim=-1)


def _chebyshev_basis(coords: torch.Tensor, max_degree: int) -> torch.Tensor:
    if max_degree < 1:
        raise ValueError("max_degree must be >= 1")
    cols = [torch.ones_like(coords)]
    if max_degree >= 2:
        cols.append(coords)
    for _ in range(2, max_degree):
        prev2 = cols[-2]
        prev1 = cols[-1]
        cols.append(2.0 * coords * prev1 - prev2)
    return torch.stack(cols, dim=-1)


def _build_basis_table(family: str, max_degree: int) -> torch.Tensor:
    coords = (torch.arange(8, dtype=torch.float32) + 0.5) * 0.25 - 1.0  # [-1+1/8, ..., 1-1/8]
    if family == "legendre":
        return _legendre_basis(coords, max_degree)
    if family == "chebyshev":
        return _chebyshev_basis(coords, max_degree)
    raise ValueError(f"Unknown polynomial family: {family}")


def _degree_group_index(i: int, j: int) -> int:
    total = i + j
    if total <= _LOW_DEGREE_MAX:
        return 0
    if total <= _MID_DEGREE_MAX:
        return 1
    return 2


class OrthogonalBoardMomentNetwork(nn.Module):
    """Bespoke implementation of the orthogonal-polynomial board-moment architecture."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        num_fields: int = 24,
        max_degree: int = 4,
        degree_dropout: float = 0.1,
        encoding_adapter: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if input_channels != 18 or encoding_adapter != SIMPLE_18:
            raise ValueError(
                "OrthogonalBoardMomentNetwork currently supports simple_18 with 18 input channels"
            )
        if num_classes != 1:
            raise ValueError(
                "OrthogonalBoardMomentNetwork supports the puzzle_binary one-logit contract"
            )
        if num_fields < 1:
            raise ValueError("num_fields must be >= 1")
        if max_degree < 1:
            raise ValueError("max_degree must be >= 1")
        if not (0.0 <= degree_dropout < 1.0):
            raise ValueError("degree_dropout must be in [0, 1)")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_fields = int(num_fields)
        self.max_degree = int(max_degree)
        self.degree_dropout = float(degree_dropout)

        self.backbone = BoardConvStem(
            input_channels=input_channels,
            channels=channels,
            depth=max(1, depth),
            use_batchnorm=use_batchnorm,
        )
        self.field_head = nn.Conv2d(channels, self.num_fields, kernel_size=1)

        for family in _FAMILIES:
            self.register_buffer(
                f"_{family}_basis",
                _build_basis_table(family, self.max_degree),
                persistent=False,
            )

        # Pre-compute the degree-group index for every (family, i, j) entry so
        # the per-sample dropout mask can be applied in a single broadcast.
        group_lookup = torch.zeros(
            len(_FAMILIES), self.max_degree, self.max_degree, dtype=torch.long
        )
        for family_idx in range(len(_FAMILIES)):
            for i in range(self.max_degree):
                for j in range(self.max_degree):
                    group_lookup[family_idx, i, j] = _degree_group_index(i, j)
        self.register_buffer("_group_lookup", group_lookup, persistent=False)

        self.num_groups = 3
        per_family_entries = self.max_degree * self.max_degree
        self.moments_per_field = len(_FAMILIES) * per_family_entries
        moment_feature_dim = self.num_fields * self.moments_per_field

        self.moment_mlp = nn.Sequential(
            nn.Linear(moment_feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.moment_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        cnn_summary_dim = channels * 2
        fusion_in = hidden_dim + cnn_summary_dim + self.num_groups * 2 + len(_FAMILIES)

        head_layers: list[nn.Module] = [nn.Linear(fusion_in, hidden_dim), nn.GELU()]
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        head_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        head_layers.append(nn.Linear(hidden_dim, 1))
        self.classifier = nn.Sequential(*head_layers)

    def _compute_moments(self, fields: torch.Tensor) -> torch.Tensor:
        # fields: (B, C, 8, 8). The board tensor uses (channel, rank, file)
        # layout, so the second spatial axis is the rank (vertical) coordinate
        # and the last is the file (horizontal) coordinate. We take ``u`` to
        # be the file coordinate and ``v`` the rank coordinate.
        legendre = self._legendre_basis  # (8, K)
        chebyshev = self._chebyshev_basis  # (8, K)
        # m_legendre[b, c, i, j] = sum_{r, f} fields[b, c, r, f] * P_i(file_f) * P_j(rank_r)
        m_legendre = torch.einsum("bcrf,fi,rj->bcij", fields, legendre, legendre)
        m_chebyshev = torch.einsum("bcrf,fi,rj->bcij", fields, chebyshev, chebyshev)
        return torch.stack([m_legendre, m_chebyshev], dim=1)  # (B, F, C, K, K)

    def _apply_degree_dropout(
        self, moments: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = moments.shape[0]
        if not self.training or self.degree_dropout <= 0.0:
            keep = torch.ones(batch, self.num_groups, device=moments.device, dtype=moments.dtype)
            return moments, keep
        keep_prob = 1.0 - self.degree_dropout
        group_keep = torch.bernoulli(
            torch.full((batch, self.num_groups), keep_prob, device=moments.device, dtype=moments.dtype)
        )
        # Guarantee at least the low-degree group survives so the gradient path
        # always exists.
        group_keep[:, 0] = 1.0
        scaled = group_keep / keep_prob  # inverted dropout scaling
        # Broadcast group keep mask through the (family, i, j) lookup table.
        sample_mask = scaled[:, self._group_lookup]  # (B, F, K, K)
        return moments * sample_mask.unsqueeze(2), group_keep

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        backbone = self.backbone(x)
        fields = self.field_head(backbone)

        moments = self._compute_moments(fields)  # (B, F, C, K, K)
        moments_dropped, group_keep = self._apply_degree_dropout(moments)

        # Diagnostic energies are computed from the post-dropout moments so they
        # reflect the signal the head actually sees.
        energy_per_entry = moments_dropped.pow(2)
        family_energy = energy_per_entry.flatten(2).sum(dim=-1)  # (B, F)

        # Sum energies into degree groups via scatter_add over the precomputed
        # group lookup; this avoids Python-level iteration over (family, i, j).
        per_entry_summed_over_fields = energy_per_entry.sum(dim=2)  # (B, F, K, K)
        flat_entries = per_entry_summed_over_fields.flatten(1)  # (B, F*K*K)
        group_index = self._group_lookup.flatten().unsqueeze(0).expand(flat_entries.shape[0], -1)
        group_energy = torch.zeros(
            moments.shape[0], self.num_groups, device=moments.device, dtype=moments.dtype
        )
        group_energy = group_energy.scatter_add(1, group_index, flat_entries)

        moment_flat = moments_dropped.flatten(1)
        moment_features = self.moment_mlp(moment_flat)
        moment_features = self.moment_dropout(moment_features)

        cnn_summary = torch.cat(
            [backbone.mean(dim=(2, 3)), backbone.amax(dim=(2, 3))], dim=1
        )

        # Normalised diagnostics are concatenated to the fusion input so the head
        # can read coarse moment-energy structure independent of the MLP path.
        normaliser = energy_per_entry.flatten(1).sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        group_energy_normalised = group_energy / normaliser
        family_energy_normalised = family_energy / normaliser

        fusion = torch.cat(
            [
                moment_features,
                cnn_summary,
                group_energy_normalised,
                group_keep,
                family_energy_normalised,
            ],
            dim=1,
        )
        logits = self.classifier(fusion).squeeze(-1)

        diagnostics: dict[str, torch.Tensor] = {"logits": logits}
        diagnostics["moment_energy_total"] = energy_per_entry.flatten(1).sum(dim=-1)
        diagnostics["moment_energy_low"] = group_energy[:, 0]
        diagnostics["moment_energy_mid"] = group_energy[:, 1]
        diagnostics["moment_energy_high"] = group_energy[:, 2]
        diagnostics["moment_energy_legendre"] = family_energy[:, 0]
        diagnostics["moment_energy_chebyshev"] = family_energy[:, 1]
        diagnostics["field_activity_norm"] = fields.flatten(1).norm(dim=1)
        diagnostics["backbone_feature_norm"] = backbone.flatten(1).norm(dim=1)
        diagnostics["moment_feature_norm"] = moment_features.norm(dim=1)
        diagnostics["cnn_summary_norm"] = cnn_summary.norm(dim=1)

        # Per-degree moment norms (total degree i + j) help diagnose whether the
        # high-order shape signals matter.
        max_total_degree = 2 * (self.max_degree - 1)
        for total in range(max_total_degree + 1):
            mask = torch.zeros(self.max_degree, self.max_degree, device=moments.device)
            for i in range(self.max_degree):
                for j in range(self.max_degree):
                    if i + j == total:
                        mask[i, j] = 1.0
            energy = (energy_per_entry * mask.view(1, 1, 1, self.max_degree, self.max_degree)).flatten(1).sum(dim=-1)
            diagnostics[f"moment_energy_degree_{total}"] = energy

        # First-order centralisation/skew descriptors of the mean piece field
        # are useful sanity diagnostics for the puzzle_binary task; they are
        # computed directly from the raw board (independent of learned fields)
        # so they remain well defined even when ``field_head`` is randomly
        # initialised.
        own_mass = x[:, 0:6].sum(dim=1)
        opp_mass = x[:, 6:12].sum(dim=1)
        signed_mass = own_mass - opp_mass
        u_axis = self._legendre_basis[:, 1] if self.max_degree >= 2 else self._legendre_basis[:, 0]
        v_axis = u_axis
        diagnostics["board_signed_centroid_file"] = torch.einsum(
            "brf,f->b", signed_mass, u_axis
        )
        diagnostics["board_signed_centroid_rank"] = torch.einsum(
            "brf,r->b", signed_mass, v_axis
        )
        return diagnostics


def build_orthogonal_board_moment_network_from_config(
    config: dict[str, Any],
) -> OrthogonalBoardMomentNetwork:
    return OrthogonalBoardMomentNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        num_fields=int(config.get("num_fields", 24)),
        max_degree=int(config.get("max_degree", 4)),
        degree_dropout=float(config.get("degree_dropout", 0.1)),
        encoding_adapter=str(config.get("encoding_adapter", config.get("encoding", SIMPLE_18))),
    )
