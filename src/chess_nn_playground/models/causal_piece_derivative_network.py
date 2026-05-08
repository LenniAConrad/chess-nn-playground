"""Causal Piece-Derivative Network for idea i179.

Faithful implementation of the markdown thesis under
``ideas/i179_causal_piece_derivative_network/``. The model classifies a
position by asking: *which pieces or squares are causally critical to
the puzzle decision?* Real puzzles tend to depend sharply on a few
pieces; near-puzzles tend to come from broad tactical texture without a
decisive dependency.

Concretely the forward pass is:

    base_logit, h = trunk(board)
    candidates = top_k(gating_head(h), k=candidate_k)
    for each candidate i and intervention t:
        delta_logit_{i,t} = delta_encoder(h, candidate_i, intervention_t)
        sensitivity_{i,t} = base_logit - delta_logit_{i,t}
    criticality = [max, top2_gap, entropy, signed_sum, own_vs_enemy_split]
    puzzle_logit = base_logit + criticality_mlp(criticality)

To stay cheap the delta encoder is a small shared MLP that consumes
per-square trunk features plus a per-intervention embedding, rather
than re-running the trunk for every intervention.

Required ablations:

* ``"none"`` -- main model.
* ``"random_candidates"`` -- replace the top-k gating with a fixed random
  permutation; tests whether learned candidate selection matters.
* ``"no_delta_readout"`` -- drop the criticality MLP from the readout so
  the puzzle logit is the trunk's ``base_logit`` only.
* ``"full_remove_only"`` -- expose only the ``remove_piece`` intervention
  type, dropping ``hide_square`` and ``neutralize_side``.
* ``"candidate_k_4"`` -- collapse to ``candidate_k = 4``; tests the
  cost/performance trade-off the packet calls out.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import (
    BoardConvStem,
    BoardTensorSpec,
    require_board_tensor,
)


_VALID_ABLATIONS = {
    "none",
    "random_candidates",
    "no_delta_readout",
    "full_remove_only",
    "candidate_k_4",
}


_INTERVENTION_NAMES = (
    "remove_piece",
    "hide_square",
    "neutralize_side",
)


def _square_coords() -> torch.Tensor:
    rank = torch.arange(8, dtype=torch.float32).view(8, 1).expand(8, 8)
    file = torch.arange(8, dtype=torch.float32).view(1, 8).expand(8, 8)
    centered_rank = (rank - 3.5) / 3.5
    centered_file = (file - 3.5) / 3.5
    edge = torch.minimum(
        torch.minimum(rank, 7.0 - rank),
        torch.minimum(file, 7.0 - file),
    ) / 3.5
    coords = torch.stack(
        [rank / 7.0, file / 7.0, centered_rank, centered_file, edge],
        dim=-1,
    ).reshape(64, 5)
    return coords


def _candidate_random_permutation(num_squares: int = 64) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(20260507)
    return torch.randperm(num_squares, generator=generator)


class CausalPieceDerivativeNetwork(nn.Module):
    """Trunk + intervention sensitivity readout for the puzzle_binary head.

    Forward returns a dict with at least:

      - ``logits``: ``(B,)`` puzzle logit fed to the BCE-with-logits trainer
        (``(B, num_classes)`` if ``num_classes > 1``).
      - ``prob``: ``sigmoid(logits)`` when ``num_classes == 1``.
      - ``base_logit``: ``(B,)`` trunk-only logit before the criticality
        readout.
      - ``criticality_residual``: ``(B,)`` criticality-MLP output added to
        ``base_logit``.
      - ``candidate_indices``: ``(B, K)`` selected square indices.
      - ``candidate_gating_scores``: ``(B, K)`` scores from the gating head
        at the candidate squares.
      - ``candidate_own_indicator``, ``candidate_opp_indicator``,
        ``candidate_occupancy``: ``(B, K)`` square statistics from the
        12 piece planes.
      - ``delta_logits``: ``(B, K, T)`` per-(candidate, intervention)
        estimated logit after the intervention.
      - ``sensitivities``: ``(B, K, T)`` ``base_logit - delta_logit``.
      - ``sensitivity_per_candidate``: ``(B, K)`` mean over interventions.
      - ``criticality_max``, ``criticality_top2_gap``,
        ``criticality_entropy``, ``criticality_signed_sum``,
        ``criticality_own_vs_enemy_split``: ``(B,)`` readout statistics.
      - ``gating_distribution_entropy``: ``(B,)`` entropy of the
        per-square gating distribution (normalised by ``log 64``).
      - ``trunk_features``: ``(B, channels, 8, 8)`` CNN stem output.
      - ``ablation_active``, ``uses_learned_candidates``,
        ``uses_delta_readout``, ``uses_all_interventions``,
        ``candidate_k_levels``, ``num_intervention_types``: ``(B,)``
        flags exposing the running ablation.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        hidden_dim: int = 96,
        candidate_k: int = 8,
        delta_channels: int = 32,
        delta_layers: int = 2,
        intervention_dim: int = 16,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if depth < 1 or channels < 1 or num_classes < 1:
            raise ValueError("depth, channels, num_classes must be >= 1")
        if candidate_k < 1 or candidate_k > 64:
            raise ValueError("candidate_k must be in [1, 64]")
        if delta_channels < 1 or delta_layers < 1 or hidden_dim < 1:
            raise ValueError("delta_channels, delta_layers, hidden_dim must be >= 1")
        if ablation not in _VALID_ABLATIONS:
            raise ValueError(
                f"ablation must be one of {sorted(_VALID_ABLATIONS)}, got {ablation!r}"
            )

        if ablation == "candidate_k_4":
            candidate_k = 4
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.depth = int(depth)
        self.hidden_dim = int(hidden_dim)
        self.candidate_k = int(candidate_k)
        self.delta_channels = int(delta_channels)
        self.delta_layers = int(delta_layers)
        self.intervention_dim = int(intervention_dim)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.ablation = str(ablation)

        self.uses_learned_candidates = self.ablation != "random_candidates"
        self.uses_delta_readout = self.ablation != "no_delta_readout"
        self.uses_all_interventions = self.ablation != "full_remove_only"

        if self.uses_all_interventions:
            self.intervention_names = tuple(_INTERVENTION_NAMES)
        else:
            self.intervention_names = (_INTERVENTION_NAMES[0],)
        self.num_intervention_types = len(self.intervention_names)

        self.trunk = BoardConvStem(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            use_batchnorm=self.use_batchnorm,
        )
        # Base puzzle logit head: average-pooled trunk → MLP → logit.
        self.base_head = nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, 1),
        )
        # Per-square gating score selecting candidate pieces/squares.
        self.gating_head = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.channels, 1, kernel_size=1),
        )
        # Intervention-type embedding gives the delta encoder a way to
        # behave differently for "remove piece" vs "hide square" vs
        # "neutralize side-to-move ownership at this square".
        self.intervention_embedding = nn.Embedding(
            len(_INTERVENTION_NAMES), self.intervention_dim
        )
        # Lightweight shared delta encoder. It does NOT re-run the trunk
        # per intervention -- it consumes per-square trunk features plus
        # board-conditioning (own/opp indicator, occupancy, side-to-move,
        # square coords) plus an intervention-type embedding.
        delta_input_dim = (
            self.channels  # candidate square trunk features
            + self.channels  # globally pooled trunk features
            + self.intervention_dim  # which intervention
            + 5  # square coords (rank, file, centered rank/file, edge)
            + 4  # own / opp / occupancy / side-to-move
        )
        delta_layers_seq: list[nn.Module] = [
            nn.LayerNorm(delta_input_dim),
            nn.Linear(delta_input_dim, self.delta_channels),
            nn.GELU(),
        ]
        for _ in range(self.delta_layers - 1):
            delta_layers_seq.extend(
                [
                    nn.Linear(self.delta_channels, self.delta_channels),
                    nn.GELU(),
                ]
            )
            if self.dropout > 0:
                delta_layers_seq.append(nn.Dropout(self.dropout))
        delta_layers_seq.append(nn.Linear(self.delta_channels, 1))
        self.delta_encoder = nn.Sequential(*delta_layers_seq)
        # Criticality readout MLP. Reads the five summary statistics the
        # packet calls out and emits a residual logit that is added to
        # ``base_logit``.
        self.criticality_mlp = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.register_buffer("coords", _square_coords(), persistent=False)
        # Random-but-fixed permutation used by the ``random_candidates``
        # ablation. Sharing one permutation across the batch is enough to
        # remove the gating signal while keeping the rest deterministic.
        self.register_buffer(
            "random_candidate_order",
            _candidate_random_permutation(),
            persistent=False,
        )

    def _square_indicators(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (own_per_square, opp_per_square, occupancy, side_to_move).

        Each of ``own_per_square``, ``opp_per_square`` and ``occupancy``
        has shape ``(B, 64)``. ``side_to_move`` has shape ``(B, 1)``.
        """
        piece_planes = board[:, : min(12, board.shape[1])].clamp(0.0, 1.0)
        if piece_planes.shape[1] < 12:
            piece_planes = F.pad(
                piece_planes, (0, 0, 0, 0, 0, 12 - piece_planes.shape[1])
            )
        white_pieces = piece_planes[:, :6].sum(dim=1)  # (B, 8, 8)
        black_pieces = piece_planes[:, 6:12].sum(dim=1)  # (B, 8, 8)
        if board.shape[1] > 12:
            side_plane = board[:, 12:13].mean(dim=(2, 3)).clamp(0.0, 1.0)  # (B, 1)
        else:
            side_plane = white_pieces.new_zeros(board.shape[0], 1)
        white_to_move = side_plane.unsqueeze(-1)  # (B, 1, 1)
        own = (white_to_move * white_pieces + (1.0 - white_to_move) * black_pieces).flatten(1)
        opp = (white_to_move * black_pieces + (1.0 - white_to_move) * white_pieces).flatten(1)
        occupancy = (own + opp).clamp(0.0, 1.0)
        return own, opp, occupancy, side_plane

    def _select_candidates(
        self,
        gating_logits: torch.Tensor,
        occupancy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select ``K`` candidate square indices.

        Returns ``(candidate_indices, candidate_scores)`` of shape
        ``(B, K)``. Under the ``random_candidates`` ablation a fixed
        random permutation is used so the gating signal is gone while
        the rest of the model stays deterministic.
        """
        batch_size = gating_logits.shape[0]
        if self.uses_learned_candidates:
            # Bias scores so occupied squares are preferred when the
            # gating head has not learned much yet -- the packet calls
            # out "key pieces", not arbitrary squares.
            biased = gating_logits + 0.5 * occupancy
            scores, indices = biased.topk(self.candidate_k, dim=-1)
        else:
            order = self.random_candidate_order.to(device=gating_logits.device)
            indices = order[: self.candidate_k].view(1, self.candidate_k).expand(
                batch_size, -1
            )
            scores = gating_logits.gather(dim=-1, index=indices)
        return indices, scores

    def _gather_square_features(
        self,
        feats: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Gather ``(B, K, channels)`` per-square features at ``indices``."""
        batch_size, channels = feats.shape[0], feats.shape[1]
        flat = feats.flatten(2).transpose(1, 2)  # (B, 64, channels)
        gather_index = indices.unsqueeze(-1).expand(-1, -1, channels)
        return flat.gather(dim=1, index=gather_index)

    def _gather_indicator(
        self,
        per_square: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Gather ``(B, K)`` indicator values at ``indices``."""
        return per_square.gather(dim=-1, index=indices)

    def _gather_coords(self, indices: torch.Tensor) -> torch.Tensor:
        """Gather ``(B, K, 5)`` square-coord features."""
        coords = self.coords.to(dtype=indices.dtype if indices.is_floating_point() else torch.float32)
        coords = self.coords.to(dtype=torch.float32)
        coords = coords.unsqueeze(0).expand(indices.shape[0], -1, -1)
        gather_index = indices.unsqueeze(-1).expand(-1, -1, coords.shape[-1])
        return coords.gather(dim=1, index=gather_index)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.trunk(x)  # (B, channels, 8, 8)
        batch_size = feats.shape[0]
        pooled = feats.mean(dim=(2, 3))  # (B, channels)
        base_logit = self.base_head(pooled).squeeze(-1)  # (B,)

        gating_map = self.gating_head(feats).squeeze(1)  # (B, 8, 8)
        gating_logits = gating_map.flatten(1)  # (B, 64)
        gating_dist = F.softmax(gating_logits, dim=-1)
        gating_entropy = -(gating_dist * gating_dist.clamp_min(1.0e-8).log()).sum(dim=-1)
        gating_entropy = gating_entropy / math.log(64.0)

        own_per_square, opp_per_square, occupancy_per_square, side_plane = self._square_indicators(x)

        candidate_indices, candidate_scores = self._select_candidates(
            gating_logits, occupancy_per_square
        )
        candidate_features = self._gather_square_features(feats, candidate_indices)  # (B, K, C)
        own_indicator = self._gather_indicator(own_per_square, candidate_indices)
        opp_indicator = self._gather_indicator(opp_per_square, candidate_indices)
        occupancy_indicator = self._gather_indicator(occupancy_per_square, candidate_indices)
        coords_indicator = self._gather_coords(candidate_indices).to(
            device=feats.device, dtype=feats.dtype
        )

        # Build the delta-encoder input. Per (B, K, T) we stack:
        #   [candidate trunk features, pooled trunk features,
        #    intervention embedding, square coords,
        #    own / opp / occupancy / side-to-move].
        K = self.candidate_k
        T = self.num_intervention_types
        side_broadcast = side_plane.view(batch_size, 1, 1).expand(-1, K, T)
        own_broadcast = own_indicator.unsqueeze(-1).expand(-1, -1, T)
        opp_broadcast = opp_indicator.unsqueeze(-1).expand(-1, -1, T)
        occ_broadcast = occupancy_indicator.unsqueeze(-1).expand(-1, -1, T)
        coords_broadcast = coords_indicator.unsqueeze(2).expand(-1, -1, T, -1)
        candidate_broadcast = candidate_features.unsqueeze(2).expand(-1, -1, T, -1)
        pooled_broadcast = pooled.view(batch_size, 1, 1, self.channels).expand(-1, K, T, -1)

        intervention_ids = torch.tensor(
            [_INTERVENTION_NAMES.index(name) for name in self.intervention_names],
            device=feats.device,
            dtype=torch.long,
        )
        intervention_embed = self.intervention_embedding(intervention_ids)  # (T, D)
        intervention_broadcast = intervention_embed.view(1, 1, T, self.intervention_dim).expand(
            batch_size, K, -1, -1
        )

        delta_input = torch.cat(
            [
                candidate_broadcast,
                pooled_broadcast,
                intervention_broadcast,
                coords_broadcast,
                own_broadcast.unsqueeze(-1),
                opp_broadcast.unsqueeze(-1),
                occ_broadcast.unsqueeze(-1),
                side_broadcast.unsqueeze(-1),
            ],
            dim=-1,
        )
        delta_logits = self.delta_encoder(delta_input).squeeze(-1)  # (B, K, T)
        sensitivities = base_logit.view(batch_size, 1, 1) - delta_logits

        # Per-candidate sensitivity = mean over interventions, used for
        # the criticality summary statistics. We weight by intervention
        # count so the ``full_remove_only`` ablation is on the same
        # numerical scale.
        sensitivity_per_candidate = sensitivities.mean(dim=-1)  # (B, K)
        abs_sens = sensitivity_per_candidate.abs()
        sorted_sens = abs_sens.sort(dim=-1, descending=True).values
        crit_max = sorted_sens[:, 0]
        if K > 1:
            crit_top2_gap = sorted_sens[:, 0] - sorted_sens[:, 1]
        else:
            crit_top2_gap = torch.zeros(batch_size, device=feats.device, dtype=feats.dtype)
        sens_dist = F.softmax(abs_sens, dim=-1)
        crit_entropy = -(sens_dist * sens_dist.clamp_min(1.0e-8).log()).sum(dim=-1) / math.log(
            float(max(K, 2))
        )
        crit_signed_sum = sensitivity_per_candidate.sum(dim=-1)
        own_vs_enemy_weight = own_indicator - opp_indicator  # (B, K)
        denom = own_vs_enemy_weight.abs().sum(dim=-1).clamp_min(1.0e-6)
        crit_split = (sensitivity_per_candidate * own_vs_enemy_weight).sum(dim=-1) / denom

        criticality_stats = torch.stack(
            [crit_max, crit_top2_gap, crit_entropy, crit_signed_sum, crit_split],
            dim=-1,
        )
        criticality_residual = self.criticality_mlp(criticality_stats).squeeze(-1)
        if self.uses_delta_readout:
            scalar_logit = base_logit + criticality_residual
        else:
            scalar_logit = base_logit

        if self.num_classes == 1:
            logits = scalar_logit
        else:
            logits = torch.zeros(
                batch_size, self.num_classes, device=feats.device, dtype=feats.dtype
            )
            logits[:, -1] = scalar_logit

        with torch.no_grad():
            ones = torch.ones(batch_size, device=feats.device, dtype=feats.dtype)
            ablation_active = ones * (0.0 if self.ablation == "none" else 1.0)
            uses_learned_candidates = ones * (1.0 if self.uses_learned_candidates else 0.0)
            uses_delta_readout = ones * (1.0 if self.uses_delta_readout else 0.0)
            uses_all_interventions = ones * (1.0 if self.uses_all_interventions else 0.0)
            candidate_k_levels = ones * float(self.candidate_k)
            num_intervention_levels = ones * float(self.num_intervention_types)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "base_logit": base_logit,
            "criticality_residual": criticality_residual,
            "candidate_indices": candidate_indices,
            "candidate_gating_scores": candidate_scores,
            "candidate_own_indicator": own_indicator,
            "candidate_opp_indicator": opp_indicator,
            "candidate_occupancy": occupancy_indicator,
            "delta_logits": delta_logits,
            "sensitivities": sensitivities,
            "sensitivity_per_candidate": sensitivity_per_candidate,
            "criticality_max": crit_max,
            "criticality_top2_gap": crit_top2_gap,
            "criticality_entropy": crit_entropy,
            "criticality_signed_sum": crit_signed_sum,
            "criticality_own_vs_enemy_split": crit_split,
            "gating_distribution_entropy": gating_entropy,
            "trunk_features": feats,
            "ablation_active": ablation_active,
            "uses_learned_candidates": uses_learned_candidates,
            "uses_delta_readout": uses_delta_readout,
            "uses_all_interventions": uses_all_interventions,
            "candidate_k_levels": candidate_k_levels,
            "num_intervention_types": num_intervention_levels,
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_causal_piece_derivative_network_from_config(
    config: dict[str, Any],
) -> CausalPieceDerivativeNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    return CausalPieceDerivativeNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        depth=int(cfg.pop("depth", 2)),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        candidate_k=int(cfg.pop("candidate_k", 8)),
        delta_channels=int(cfg.pop("delta_channels", 32)),
        delta_layers=int(cfg.pop("delta_layers", 2)),
        intervention_dim=int(cfg.pop("intervention_dim", 16)),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        ablation=str(cfg.pop("ablation", "none")),
    )


__all__ = [
    "CausalPieceDerivativeNetwork",
    "build_causal_piece_derivative_network_from_config",
]
