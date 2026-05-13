"""Occlusion Semiring Delta-Bilinear Hyperedge (p023).

Source: ``ideas/research/primitives/external_19_occlusion_semiring_delta_bilinear_hyperedge.md``
(rank-1 proposal ``primitive_1_occlusion_semiring_scan``).

The packet's rank-1 proposal is an occlusion-semiring ray scan with a
**backward recurrence** at each step along a ray, distinct from p021's
forward prefix-product formulation:

    h_{b,r,t}   = (1 - o_{b, c_{r,t+1}}) * h_{b,r,t+1} + V * x_{b, c_{r,t+1}}
    h_{b,r,L}   = 0
    y_{b, s}    = W_0 * x_{b, s} + sum_r U * h_{b, r, 0}

Geometrically: from square ``s`` looking out along ray ``r``, ``h_{r, 0}``
aggregates contributions from positions ``t = 1..L`` along the ray,
multiplied by the *transmittance* product of all unoccupied cells
between them. The recurrence runs in reverse so each step's
``(1 - o_{t+1})`` gate decides whether the deeper-into-ray hidden state
is occluded.

To honour the packet name "delta_bilinear_hyperedge" while implementing
the rank-1 proposal exactly, the per-square head output ``y_s`` is
further reduced through a small *bilinear hyperedge* contraction over
its 8 direction-hidden states: each pair of opposite directions
(N / S, NE / SW, E / W, SE / NW) forms a 2-element hyperedge whose
bilinear product encodes pin / x-ray motifs (an own piece flanked by
attacker + defender along a single line). The bilinear hyperedges are
then averaged across squares and projected to a scalar.

Deferred internal proposals from the same packet:

- ``primitive_2_delta_bilinear_accumulator`` (rank 2): implemented at
  p022.
- ``primitive_3_legal_hyperedge_contraction`` (rank 3): not in this
  batch.
- ``primitive_4_tropical_threat_scan`` (rank 4): not in this batch.
- ``primitive_5_chess_orbit_linear`` (rank 5): not in this batch.

p023 is intentionally distinct from p021. Both encode "ray vision until
a blocker", but p021 uses a *forward* exclusive prefix product
(``T_{l} = prod_{q<l}(1 - o_q)``) reduced over an outgoing fan, while
p023 uses a *backward* recurrence (``h_t = (1 - o_{t+1}) * h_{t+1} + V x``),
which exposes different gradient flow per ray segment and a different
incremental-update semantics under bounded-change inputs.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.ray_geometry import (
    NUM_DIRECTIONS,
    RAY_MAX_LEN,
    RayGeometry,
    SQUARES,
    gather_along_rays,
    gather_scalar_along_rays,
)
from chess_nn_playground.models.primitives.trunk_features import trunk_joint_features
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


PIECE_PLANE_COUNT = 12

# Pair up opposite directions for the bilinear hyperedge step.
# Directions encoded as in ray_geometry.DIRECTIONS:
#   0 N, 1 NE, 2 E, 3 SE, 4 S, 5 SW, 6 W, 7 NW
# Opposite pairs: (N, S) = (0, 4), (NE, SW) = (1, 5), (E, W) = (2, 6), (SE, NW) = (3, 7)
OPPOSITE_DIRECTION_PAIRS: tuple[tuple[int, int], ...] = ((0, 4), (1, 5), (2, 6), (3, 7))


ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "zero_occupancy",       # disable blocker gate (all 1)
    "uniform_occupancy",    # blocker gate everywhere 0
    "disable_bilinear",     # drop the hyperedge product, sum hidden states instead
    "zero_delta",
    "trunk_only",
)


class OcclusionSemiringDeltaBilinearHyperedge(nn.Module):
    """Occlusion Semiring Delta-Bilinear Hyperedge head (p023)."""

    ALLOWED_ABLATIONS = ALLOWED_ABLATIONS

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        token_dim: int = 24,
        hidden_dim: int = 32,
        bilinear_dim: int = 16,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("OcclusionSemiringDeltaBilinearHyperedge supports the puzzle_binary one-logit contract")
        if int(input_channels) != 18:
            raise ValueError("OcclusionSemiringDeltaBilinearHyperedge requires the simple_18 board tensor")
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}")

        self.num_classes = 1
        self.ablation = str(ablation)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.bilinear_dim = int(bilinear_dim)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))

        self.trunk = ExchangeThenKingDualStreamNetwork(
            input_channels=int(input_channels),
            num_classes=1,
            channels=int(trunk_channels),
            hidden_dim=int(trunk_hidden_dim),
            depth=int(trunk_depth),
            dropout=float(trunk_dropout),
            use_batchnorm=bool(trunk_use_batchnorm),
            gate_dim=trunk_gate_dim,
            ablation=str(trunk_ablation),
        )
        self.feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )

        # Per-square token projection.
        self.token_proj = nn.Linear(PIECE_PLANE_COUNT + 1, self.token_dim)
        # V projection from token to hidden_dim, used inside the recurrence.
        self.value_proj = nn.Linear(self.token_dim, self.hidden_dim)
        # Per-direction recurrence weight: identity-bias decay (no extra params here).
        # We rely on the occupancy gate to provide the (1 - o_{t+1}) factor.
        # Bilinear projection: map opposing-direction hidden states to a hyperedge embedding.
        # Each pair gets its own left/right projection.
        self.left_proj = nn.Linear(self.hidden_dim, self.bilinear_dim, bias=False)
        self.right_proj = nn.Linear(self.hidden_dim, self.bilinear_dim, bias=False)

        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        # Per-square readout: 4 pair hyperedges of bilinear_dim each -> head MLP.
        readout_dim = len(OPPOSITE_DIRECTION_PAIRS) * self.bilinear_dim
        self.delta_head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, int(head_hidden_dim)),
            nn.GELU(),
            head_dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )
        with torch.no_grad():
            final_layer = self.gate_head[-1]
            if isinstance(final_layer, nn.Linear):
                final_layer.bias.fill_(float(gate_init))

        geom = RayGeometry.build()
        self.register_buffer("ray_step_index", geom.step_index, persistent=False)
        self.register_buffer("ray_step_mask", geom.step_mask, persistent=False)

    def _build_square_tokens(self, board: torch.Tensor) -> torch.Tensor:
        piece_planes = board[:, :PIECE_PLANE_COUNT].clamp(0.0, 1.0)
        stm = board[:, 12:13].clamp(0.0, 1.0)
        token_input = torch.cat([piece_planes, stm], dim=1).flatten(2).transpose(1, 2).contiguous()
        return self.token_proj(token_input)

    def _occupancy(self, board: torch.Tensor) -> torch.Tensor:
        piece_planes = board[:, :PIECE_PLANE_COUNT].clamp(0.0, 1.0).flatten(2)
        return piece_planes.sum(dim=1).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        eps = 1.0e-6

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        tokens = self._build_square_tokens(board)
        occupancy = self._occupancy(board)
        if self.ablation == "zero_occupancy":
            occupancy = torch.zeros_like(occupancy)
        elif self.ablation == "uniform_occupancy":
            occupancy = torch.ones_like(occupancy)

        # Gather ray tokens and ray occupancy.
        # ray_tokens: (B, 8, 64, 7, token_dim)
        # ray_occ:    (B, 8, 64, 7)
        ray_tokens = gather_along_rays(tokens, self.ray_step_index, self.ray_step_mask)
        ray_occ = gather_scalar_along_rays(occupancy, self.ray_step_index, self.ray_step_mask)
        step_mask = self.ray_step_mask.to(device=ray_occ.device, dtype=ray_occ.dtype).view(
            1, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN
        )

        # Backward recurrence:
        #   h_{t} = (1 - o_{t+1}) * h_{t+1} + V * x_{t+1}, where t+1 is one step deeper.
        # We index t from 0 (closest to source) to L-1, with h_{L-1} == 0.
        v_tokens = self.value_proj(ray_tokens)  # (B, 8, 64, 7, hidden_dim)
        # Start: h_max = 0
        h = v_tokens.new_zeros(batch, NUM_DIRECTIONS, SQUARES, self.hidden_dim)
        # Walk t = L-1 down to 0. At step t we incorporate v_tokens[..., t, :] and gate by (1 - o[..., t]).
        # Mathematically: h_{t} = (1 - o_{t+1}) * h_{t+1} + V * x_{t+1}
        # We treat step l (1-indexed) as the cell at distance l from source. The
        # recurrence variable h represents the aggregate "what is visible from
        # depth ge l". We initialize h at depth L+1 to zero and walk back.
        for t in range(RAY_MAX_LEN - 1, -1, -1):
            occ_t = ray_occ[..., t]  # (B, 8, 64)
            valid_t = step_mask[..., t]  # (B if broadcast, 8, 64)
            gate_t = (1.0 - occ_t) * valid_t
            value_t = v_tokens[..., t, :] * valid_t.unsqueeze(-1)
            h = gate_t.unsqueeze(-1) * h + value_t

        # h is now h_{b, r, 0} -- the source-side hidden state along each ray.
        # Project to bilinear hyperedge embeddings for each opposing-direction pair.
        pair_outputs = []
        diagnostic_pair_norms = []
        for left_dir, right_dir in OPPOSITE_DIRECTION_PAIRS:
            h_left = h[:, left_dir]                          # (B, 64, hidden_dim)
            h_right = h[:, right_dir]                        # (B, 64, hidden_dim)
            left = self.left_proj(h_left)                    # (B, 64, bilinear_dim)
            right = self.right_proj(h_right)
            if self.ablation == "disable_bilinear":
                edge = left + right
            else:
                edge = left * right
            pair_outputs.append(edge)
            diagnostic_pair_norms.append(edge.pow(2).mean(dim=(1, 2)).sqrt())

        # Stack pairs and mean-pool across squares.
        pair_tensor = torch.cat(pair_outputs, dim=-1)  # (B, 64, 4 * bilinear_dim)
        pooled = pair_tensor.mean(dim=1)
        delta_raw = self.delta_head(pooled).view(-1)

        gate_logit = self.gate_head(joint).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
        else:
            primitive_delta = gate * delta_raw

        logits = base_logit + primitive_delta

        gate_clamped = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(
            gate_clamped * gate_clamped.log()
            + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
        )
        mean_h_magnitude = h.pow(2).mean(dim=(1, 2, 3)).sqrt()
        pair_norm_stack = torch.stack(diagnostic_pair_norms, dim=1).mean(dim=1)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "base_logit": base_logit,
            "primitive_delta": primitive_delta,
            "primitive_delta_raw": delta_raw,
            "primitive_gate": gate,
            "primitive_gate_logit": gate_logit,
            "primitive_gate_entropy": gate_entropy,
            "osdb_hidden_magnitude": mean_h_magnitude,
            "osdb_pair_hyperedge_magnitude": pair_norm_stack,
            "mechanism_energy": trunk_out["mechanism_energy"] + mean_h_magnitude.detach(),
            "proposal_profile_strength": (primitive_delta.abs() * gate_entropy).clamp(0.0, 20.0),
            "proposal_keyword_count": logits.new_full(
                (batch,), float(len(OPPOSITE_DIRECTION_PAIRS) * self.bilinear_dim)
            ),
        }
        for key, value in trunk_out.items():
            if key == "logits":
                continue
            diag_key = key if key.startswith("base_") or key in {"gate", "gate_logit"} else f"trunk_{key}"
            diagnostics.setdefault(diag_key, value)
        return diagnostics


def build_occlusion_semiring_delta_bilinear_hyperedge_from_config(
    config: dict[str, Any],
) -> OcclusionSemiringDeltaBilinearHyperedge:
    cfg = dict(config)
    return OcclusionSemiringDeltaBilinearHyperedge(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        token_dim=int(cfg.get("token_dim", 24)),
        hidden_dim=int(cfg.get("hidden_dim", 32)),
        bilinear_dim=int(cfg.get("bilinear_dim", 16)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )
