"""Support-Function Envelope Network for idea i138.

Implements the markdown architecture from
``ideas/research_packets/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md``
(Candidate 2: Support-Function Envelope Network).

For each nonnegative learned field ``rho_c(s)`` over board squares and a
fixed direction ``u``, the soft support function is

    h_c(u) = tau * logsumexp_s( ( <u, coord_s> + log(eps + rho_c(s)) ) / tau ).

The width ``w_c(u) = h_c(u) + h_c(-u)`` and center ``m_c(u) = h_c(u) - h_c(-u)``
characterise the envelope of the field along ``u``. Own/opponent contrast
features ``|m_own - m_opp|`` and ``w_own / (eps + w_opp)`` capture how
asymmetric the side-to-move and opponent envelopes are. Mass and entropy
descriptors round out the per-field summary.

Ablations from the packet (``Central Ablations``):
  ``none``, ``mean_pool_fields``, ``random_directions``,
  ``no_opponent_contrast``, ``hard_max_support``,
  ``counts_plus_envelope_only``.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models._packet_bespoke_base import (
    BoardConvStem,
    BoardTensorSpec,
    format_logits,
    require_board_tensor,
    us_them_piece_planes,
)


VALID_ABLATIONS: frozenset[str] = frozenset(
    {
        "none",
        "mean_pool_fields",
        "random_directions",
        "no_opponent_contrast",
        "hard_max_support",
        "counts_plus_envelope_only",
    }
)


def _chess_relevant_directions() -> torch.Tensor:
    """Return 16 fixed chess-relevant 2D directions, paired antipodally.

    Index ``i`` and ``i ^ 1`` are negatives of each other so that the
    width / center pair lookup is a simple XOR over indices.
    """
    raw = [
        (1.0, 0.0),   # rank +
        (-1.0, 0.0),  # rank -
        (0.0, 1.0),   # file +
        (0.0, -1.0),  # file -
        (1.0, 1.0),   # main diagonal +
        (-1.0, -1.0), # main diagonal -
        (1.0, -1.0),  # anti diagonal +
        (-1.0, 1.0),  # anti diagonal -
        (2.0, 1.0),   # knight a +
        (-2.0, -1.0), # knight a -
        (1.0, 2.0),   # knight b +
        (-1.0, -2.0), # knight b -
        (2.0, -1.0),  # knight c +
        (-2.0, 1.0),  # knight c -
        (1.0, -2.0),  # knight d +
        (-1.0, 2.0),  # knight d -
    ]
    return F.normalize(torch.tensor(raw, dtype=torch.float32), dim=-1)


class SupportFunctionEnvelopeNetwork(nn.Module):
    """Bespoke implementation of the Support-Function Envelope Network.

    Forward output dict keys:
      - ``logits``: ``(B,)`` puzzle logit.
      - ``h``: ``(B, F, K)`` soft support values per field/direction.
      - ``width``: ``(B, F, K)`` envelope width along each direction pair.
      - ``center``: ``(B, F, K)`` envelope center along each direction pair.
      - ``mass``: ``(B, F)`` total field mass.
      - ``entropy``: ``(B, F)`` field entropy in nats.
      - ``learned_fields``: ``(B, n_learned_fields, 8, 8)`` softplus fields.
      - ``own_envelope_mass``, ``opp_envelope_mass``: ``(B, n_pairs)`` total
        envelope mass on own / opp sides (sum of widths over directions).
      - ``overlap_gap``: ``(B, n_pairs, K_primary)`` ``|m_own - m_opp|``
        per learned own/opp pair.
      - ``width_ratio``: ``(B, n_pairs, K_primary)`` ``w_own / (eps + w_opp)``.
      - ``directions``: ``(K, 2)`` direction vectors used.
      - ``ablation_*``: per-batch indicator flags for diagnostics.
    """

    EFFECTIVE_INPUT_PIECE_PLANES = 12  # 6 own + 6 opponent

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        n_fields: int = 12,
        n_directions: int = 16,
        tau: float = 0.25,
        epsilon: float = 1e-3,
        channels: int = 64,
        depth: int = 2,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "SupportFunctionEnvelopeNetwork implements the puzzle_binary single-logit contract only"
            )
        if n_fields < 2 or n_fields % 2 != 0:
            raise ValueError("n_fields must be an even integer >= 2 (own/opp split)")
        if n_directions < 2 or n_directions % 2 != 0:
            raise ValueError("n_directions must be an even integer >= 2 (antipodal pairs)")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if tau <= 0:
            raise ValueError("tau must be > 0")
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if ablation not in VALID_ABLATIONS:
            raise ValueError(
                f"Unknown ablation {ablation!r}; expected one of {sorted(VALID_ABLATIONS)}"
            )

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.n_fields = int(n_fields)
        self.n_directions = int(n_directions)
        self.tau = float(tau)
        self.epsilon = float(epsilon)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.ablation = str(ablation)

        self.n_own_fields = self.n_fields // 2
        self.n_opp_fields = self.n_fields - self.n_own_fields
        self.n_pairs = min(self.n_own_fields, self.n_opp_fields)
        # Primary direction indices: one representative per antipodal pair.
        self.n_primary = self.n_directions // 2

        # Board trunk and per-field projection.
        self.stem = BoardConvStem(
            input_channels=input_channels,
            channels=int(channels),
            depth=int(depth),
            use_batchnorm=use_batchnorm,
        )
        self.field_head = nn.Conv2d(int(channels), self.n_fields, kernel_size=1)

        # Fixed chess-relevant directions (and a frozen-random alternative).
        if self.n_directions == 16:
            chess_dirs = _chess_relevant_directions()
        else:
            # Even-spaced 2D directions on the unit circle, paired antipodally.
            base = torch.linspace(0, math.pi, self.n_primary + 1)[:-1]
            angles = torch.stack([base, base + math.pi], dim=1).flatten()
            chess_dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        self.register_buffer("chess_directions", chess_dirs, persistent=False)

        rng = torch.Generator().manual_seed(0xF1E1D7CE)
        random_dirs = F.normalize(
            torch.randn(self.n_directions, 2, generator=rng),
            dim=-1,
        )
        # Force the random set to be antipodally paired so width/center pairing still works.
        random_dirs[1::2] = -random_dirs[0::2]
        self.register_buffer("random_directions", random_dirs, persistent=False)

        anti = torch.tensor([i ^ 1 for i in range(self.n_directions)], dtype=torch.long)
        self.register_buffer("antipode_index", anti, persistent=False)
        primary = torch.arange(0, self.n_directions, 2, dtype=torch.long)
        self.register_buffer("primary_index", primary, persistent=False)

        # Square coordinates normalised to roughly the unit square.
        rows = torch.linspace(-1.0, 1.0, 8)
        cols = torch.linspace(-1.0, 1.0, 8)
        coord = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1)
        self.register_buffer("square_coords", coord, persistent=False)  # (8, 8, 2)

        head_in = self._head_input_dim()
        self.head_norm = nn.LayerNorm(head_in)
        self.head = nn.Sequential(
            nn.Linear(head_in, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _total_fields(self) -> int:
        if self.ablation == "counts_plus_envelope_only":
            return self.EFFECTIVE_INPUT_PIECE_PLANES
        return self.n_fields + self.EFFECTIVE_INPUT_PIECE_PLANES

    def _head_input_dim(self) -> int:
        per_field = 2 * self.n_directions + 3  # h, width, center, mass, entropy, max
        # Width/center are antipodally redundant; we still keep all directions for
        # uniform indexing. The MLP can deduplicate on its own.
        envelope_dim = self._total_fields() * per_field
        contrast_dim = 0
        if self.ablation != "no_opponent_contrast":
            # Learned own/opp contrast (n_pairs pairs) plus piece-plane own/opp contrast (6 pairs).
            learned_pairs = self.n_pairs if self.ablation != "counts_plus_envelope_only" else 0
            piece_pairs = self.EFFECTIVE_INPUT_PIECE_PLANES // 2
            contrast_dim = 2 * self.n_primary * (learned_pairs + piece_pairs)
        return envelope_dim + contrast_dim

    def _active_directions(self) -> torch.Tensor:
        if self.ablation == "random_directions":
            return self.random_directions
        return self.chess_directions

    def _build_input_fields(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (fields, learned_fields).

        ``fields`` contains the full set of nonnegative fields fed through the
        envelope readout, in the order ``[learned_own, learned_opp, us_pieces, them_pieces]``.
        ``learned_fields`` is the learned subset (empty under
        ``counts_plus_envelope_only``).
        """
        x = require_board_tensor(x, self.spec)
        us, them = us_them_piece_planes(x, self.input_channels)
        if self.ablation == "counts_plus_envelope_only":
            learned = x.new_zeros(x.shape[0], 0, 8, 8)
        else:
            trunk = self.stem(x)
            learned = F.softplus(self.field_head(trunk))
        fields = torch.cat([learned, us, them], dim=1)
        return fields, learned

    def _support_function(self, fields: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        """Return ``h_c(u_k)`` of shape ``(B, F, K)``.

        ``fields``: ``(B, F, 8, 8)`` nonnegative.
        ``dirs``: ``(K, 2)`` direction vectors.
        """
        batch, num_fields, height, width = fields.shape
        coord = self.square_coords.to(fields.dtype)  # (8, 8, 2)
        proj = torch.einsum("rci,ki->krc", coord, dirs.to(fields.dtype))  # (K, 8, 8)
        # Stable log of the field with epsilon floor.
        log_a = torch.log(fields + self.epsilon)  # (B, F, 8, 8)

        # arg[b, f, k, r, c] = (proj[k, r, c] + log_a[b, f, r, c]) / tau
        arg = (proj.unsqueeze(0).unsqueeze(0) + log_a.unsqueeze(2)) / self.tau
        flat = arg.flatten(start_dim=-2)  # (B, F, K, 64)
        if self.ablation == "hard_max_support":
            h_over_tau = flat.amax(dim=-1)
        else:
            h_over_tau = torch.logsumexp(flat, dim=-1)
        return self.tau * h_over_tau  # (B, F, K)

    def _field_mass_entropy(self, fields: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = fields.flatten(start_dim=-2)  # (B, F, 64)
        mass = flat.sum(dim=-1)
        max_val = flat.amax(dim=-1)
        prob = flat / (mass.unsqueeze(-1) + self.epsilon)
        entropy = -(prob * torch.log(prob + self.epsilon)).sum(dim=-1)
        return mass, entropy, max_val

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        fields, learned_fields = self._build_input_fields(x)
        batch = fields.shape[0]
        dirs = self._active_directions()  # (K, 2)

        if self.ablation == "mean_pool_fields":
            mean_pool = fields.mean(dim=(-2, -1))  # (B, F)
            max_pool = fields.amax(dim=(-2, -1))   # (B, F)
            sum_pool = fields.sum(dim=(-2, -1))    # (B, F)
            # Replicate the same total descriptor shape so the head matches.
            h = mean_pool.unsqueeze(-1).expand(batch, fields.shape[1], self.n_directions)
            width = (max_pool.unsqueeze(-1) - mean_pool.unsqueeze(-1)).expand_as(h)
            center = torch.zeros_like(h)
            mass = sum_pool
            max_val = max_pool
            # Approximate entropy with a constant based on uniform-mass fallback.
            prob = fields.flatten(start_dim=-2) / (mass.unsqueeze(-1) + self.epsilon)
            entropy = -(prob * torch.log(prob + self.epsilon)).sum(dim=-1)
        else:
            h = self._support_function(fields, dirs)  # (B, F, K)
            anti = self.antipode_index
            h_anti = h.index_select(-1, anti)
            width = h + h_anti
            center = h - h_anti
            mass, entropy, max_val = self._field_mass_entropy(fields)

        per_field_features = torch.cat(
            [
                h,
                width,
                center,
                mass.unsqueeze(-1),
                entropy.unsqueeze(-1),
                max_val.unsqueeze(-1),
            ],
            dim=-1,
        ).flatten(start_dim=1)  # (B, F * (3K + 3))

        # Own/opp contrast features.
        contrast_parts: list[torch.Tensor] = []
        if self.ablation != "no_opponent_contrast":
            primary = self.primary_index
            width_primary = width.index_select(-1, primary)  # (B, F, K_primary)
            center_primary = center.index_select(-1, primary)
            if self.ablation != "counts_plus_envelope_only" and self.n_pairs > 0:
                own_w = width_primary[:, : self.n_own_fields][:, : self.n_pairs]
                opp_w = width_primary[:, self.n_own_fields : self.n_own_fields + self.n_opp_fields][:, : self.n_pairs]
                own_m = center_primary[:, : self.n_own_fields][:, : self.n_pairs]
                opp_m = center_primary[:, self.n_own_fields : self.n_own_fields + self.n_opp_fields][:, : self.n_pairs]
                gap_learned = (own_m - opp_m).abs()
                ratio_learned = own_w / (opp_w + self.epsilon)
                contrast_parts.append(gap_learned.flatten(start_dim=1))
                contrast_parts.append(ratio_learned.flatten(start_dim=1))

            piece_offset = (
                self.n_fields if self.ablation != "counts_plus_envelope_only" else 0
            )
            piece_w = width_primary[:, piece_offset : piece_offset + self.EFFECTIVE_INPUT_PIECE_PLANES]
            piece_m = center_primary[:, piece_offset : piece_offset + self.EFFECTIVE_INPUT_PIECE_PLANES]
            half = self.EFFECTIVE_INPUT_PIECE_PLANES // 2
            us_w, them_w = piece_w[:, :half], piece_w[:, half:]
            us_m, them_m = piece_m[:, :half], piece_m[:, half:]
            gap_pieces = (us_m - them_m).abs()
            ratio_pieces = us_w / (them_w + self.epsilon)
            contrast_parts.append(gap_pieces.flatten(start_dim=1))
            contrast_parts.append(ratio_pieces.flatten(start_dim=1))

        if contrast_parts:
            contrast = torch.cat(contrast_parts, dim=-1)
        else:
            contrast = per_field_features.new_zeros(batch, 0)

        head_input = torch.cat([per_field_features, contrast], dim=-1)
        # In the no_opponent_contrast ablation the head must still see the
        # configured input width.
        expected = self._head_input_dim()
        if head_input.shape[-1] != expected:
            pad = expected - head_input.shape[-1]
            if pad > 0:
                head_input = F.pad(head_input, (0, pad))
            else:
                head_input = head_input[..., :expected]
        normed = self.head_norm(head_input)
        logit = self.head(normed).squeeze(-1)

        # Diagnostics for the trainer's report.
        primary = self.primary_index
        width_primary = width.index_select(-1, primary)
        center_primary = center.index_select(-1, primary)
        if (
            self.ablation != "counts_plus_envelope_only"
            and self.n_pairs > 0
        ):
            own_w_full = width_primary[:, : self.n_own_fields][:, : self.n_pairs]
            opp_w_full = width_primary[:, self.n_own_fields : self.n_own_fields + self.n_opp_fields][:, : self.n_pairs]
            own_m_full = center_primary[:, : self.n_own_fields][:, : self.n_pairs]
            opp_m_full = center_primary[:, self.n_own_fields : self.n_own_fields + self.n_opp_fields][:, : self.n_pairs]
            overlap_gap = (own_m_full - opp_m_full).abs()
            width_ratio = own_w_full / (opp_w_full + self.epsilon)
            own_envelope_mass = own_w_full.sum(dim=-1)
            opp_envelope_mass = opp_w_full.sum(dim=-1)
        else:
            overlap_gap = h.new_zeros(batch, 0, self.n_primary)
            width_ratio = h.new_zeros(batch, 0, self.n_primary)
            own_envelope_mass = h.new_zeros(batch, 0)
            opp_envelope_mass = h.new_zeros(batch, 0)

        ones = logit.new_ones(batch)
        ablation_flag = lambda name: ones * (1.0 if self.ablation == name else 0.0)

        output: dict[str, torch.Tensor] = {
            "logits": format_logits(logit.unsqueeze(-1), self.num_classes),
            "h": h,
            "width": width,
            "center": center,
            "mass": mass,
            "entropy": entropy,
            "field_max": max_val,
            "learned_fields": learned_fields,
            "directions": dirs,
            "overlap_gap": overlap_gap,
            "width_ratio": width_ratio,
            "own_envelope_mass": own_envelope_mass,
            "opp_envelope_mass": opp_envelope_mass,
            "tau": logit.new_full((batch,), float(self.tau)),
            "ablation_mean_pool_fields": ablation_flag("mean_pool_fields"),
            "ablation_random_directions": ablation_flag("random_directions"),
            "ablation_no_opponent_contrast": ablation_flag("no_opponent_contrast"),
            "ablation_hard_max_support": ablation_flag("hard_max_support"),
            "ablation_counts_plus_envelope_only": ablation_flag("counts_plus_envelope_only"),
        }
        return output


def build_support_function_envelope_network_from_config(
    config: dict[str, Any],
) -> SupportFunctionEnvelopeNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)

    return SupportFunctionEnvelopeNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        n_fields=int(cfg.pop("n_fields", 12)),
        n_directions=int(cfg.pop("n_directions", 16)),
        tau=float(cfg.pop("tau", 0.25)),
        epsilon=float(cfg.pop("epsilon", 1e-3)),
        channels=int(cfg.pop("channels", 64)),
        depth=int(cfg.pop("depth", 2)),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        ablation=str(cfg.pop("ablation", "none")),
    )


__all__ = [
    "SupportFunctionEnvelopeNetwork",
    "VALID_ABLATIONS",
    "build_support_function_envelope_network_from_config",
]
