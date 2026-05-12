"""Exchange-Soundness Graph Network for idea i187.

Faithful implementation of the markdown thesis under
``ideas/registry/i187_exchange_soundness_graph_network/``. The packet's working
thesis is that many false puzzle signals come from attacks that look
strong but lose material or fail tactically after exchanges. A puzzle
detector should know whether an apparent tactic is *exchange-sound* on
the side-to-move's attack/defense graph.

The model turns that thesis into an explicit, board-only differentiable
form of static-exchange evaluation (SEE) over a learned attack/defense
graph:

  1. A small convolutional trunk encodes ``(B, 18, 8, 8)`` board planes.
  2. Per-piece-type masks are derived from the side-to-move plane.
  3. Four conv heads, each reading trunk + piece-mask context, predict
     per-square graph quantities for *every* square of the 8x8 board:
       - attacker intensity  ``a(s)`` -- how intensely side-to-move
         attacks ``s`` (sigmoid).
       - defender intensity  ``d(s)`` -- how intensely side-not-to-move
         defends ``s`` (sigmoid).
       - attacker-value logits over 6 piece types, softmaxed to a
         distribution and dotted with the standard piece-value vector
         to get ``v_a(s)`` -- the value of the *cheapest available
         attacker* at ``s``.
       - defender-value logits ``v_d(s)`` analogously.
  4. The target value at ``s`` is the value of the opponent's piece on
     ``s`` (zero on empty / own-piece squares); only opponent-piece
     squares are real capture targets. ``v_target(s)`` is computed
     exactly from the input planes, not learned.
  5. A *bounded-depth differentiable SEE* unrolls the alternating
     capture-and-recapture sequence on the attack/defense graph:

         see(s) = v_target
                  - p_d * max(0, v_a
                                  - p_a * max(0, v_d
                                                  - p_d * max(0, v_a)))

     ``p_a, p_d`` are the attacker/defender intensities (probabilities
     a capture continues), and ``max(0, .)`` is the soft "stop here"
     option that lets the side-to-move decline a losing recapture --
     which is exactly the SEE rule. Depth 4 covers the practical
     puzzle window of capture/recapture/recapture/recapture; deeper
     nesting is recursive in the same form.
  6. The per-square exchange-soundness is ``sigmoid(see / T)``. The
     bottleneck pool of trunk features uses a softmax over the *target
     squares* weighted by ``|see|``: a position whose tactic hinges on
     one decisive square pulls feature mass from that square; a noisy
     position spreads pool mass over many squares.
  7. The classifier head consumes the bottleneck pool, an attacker
     pool (feature mass weighted by attacker intensity), a defender
     pool (weighted by defender intensity), and a vector of
     graph-network scalars: ``max_see_target``, ``mean_see_target``,
     ``frac_unsound_targets``, ``graph_pressure``, ``reply_pressure``,
     ``defense_gap``, ``transport_imbalance``, ``sheaf_tension``.

This is materially distinct from idea i186 (Legal-Reaction Bottleneck
Network), which only models a defender-reply softmax over opponent
piece squares. Here we additionally:

  - learn a *piece-type-aware* attacker / defender value field;
  - run a bounded-depth differentiable SEE at every square of the
    board (the "exchange-soundness" signal); and
  - aggregate over the *target squares* of the side-to-move's attack
    graph rather than all opponent squares.

The model is board-only: only the side-to-move plane and the piece
planes of the simple_18 input tensor are read. CRTK / engine / source
metadata are never consumed. Output shape is ``(B,)`` for
``num_classes == 1`` so the puzzle_binary BCE-with-logits trainer can
consume the logits directly.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import (
    BoardConvStem,
    BoardTensorSpec,
    require_board_tensor,
)


WHITE_PIECE_PLANES = (0, 1, 2, 3, 4, 5)
BLACK_PIECE_PLANES = (6, 7, 8, 9, 10, 11)
SIDE_TO_MOVE_PLANE = 12
NUM_PIECE_TYPES = 6
DEFAULT_PIECE_VALUES = (1.0, 3.0, 3.0, 5.0, 9.0, 12.0)
NEG_INF = -1.0e9


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _own_and_opp_piece_planes(board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-piece-type masks for side-to-move and side-not-to-move.

    Both tensors have shape ``(B, 6, 8, 8)`` with ``[P, N, B, R, Q, K]``
    ordering. Rows where the side-to-move plane is 1 take the white
    piece planes as own; rows where it is 0 take the black piece planes
    as own (and vice versa for opp).
    """
    side = board[:, SIDE_TO_MOVE_PLANE : SIDE_TO_MOVE_PLANE + 1]
    white_to_move = side.amax(dim=(2, 3), keepdim=True).clamp(0.0, 1.0)
    black_to_move = 1.0 - white_to_move

    white = board[:, list(WHITE_PIECE_PLANES), :, :].clamp(0.0, 1.0)
    black = board[:, list(BLACK_PIECE_PLANES), :, :].clamp(0.0, 1.0)
    own = white_to_move * white + black_to_move * black
    opp = white_to_move * black + black_to_move * white
    return own, opp


class _PieceTypeHead(nn.Module):
    """3x3 -> 1x1 conv head producing ``out_channels`` per square."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.drop(self.act(self.conv1(x))))


class ExchangeSoundnessGraphNetwork(nn.Module):
    """Exchange-Soundness Graph Network.

    Forward returns a dict with at least:

      - ``logits``: ``(B,)`` puzzle logit (or ``(B, num_classes)``
        when ``num_classes > 1``).
      - ``prob``: ``sigmoid(logits)`` for ``num_classes == 1``.
      - ``attacker_intensity``: ``(B, 8, 8)`` ``sigmoid`` of the
        per-square attacker logit (side-to-move attacking the square).
      - ``defender_intensity``: ``(B, 8, 8)`` defender side counterpart.
      - ``attacker_value_field``: ``(B, 8, 8)`` value of the cheapest
        attacker at each square (in pawn units).
      - ``defender_value_field``: ``(B, 8, 8)`` cheapest defender value.
      - ``target_value_field``: ``(B, 8, 8)`` value of the opponent
        piece occupying each square (zero off opponent squares).
      - ``exchange_score_field``: ``(B, 8, 8)`` differentiable SEE
        score on the attack/defense graph.
      - ``exchange_soundness_field``: ``(B, 8, 8)`` ``sigmoid(see / T)``
        gating used for the bottleneck pool.
      - ``target_mask``: ``(B, 8, 8)`` indicator of opponent-piece
        squares (the real capture targets).
      - ``max_see_target``: ``(B,)`` largest SEE among target squares.
      - ``mean_see_target``: ``(B,)`` average SEE over target squares.
      - ``frac_unsound_targets``: ``(B,)`` fraction of opponent squares
        whose differentiable SEE is non-positive.
      - ``graph_pressure``: ``(B,)`` mean attacker intensity over
        opponent squares.
      - ``reply_pressure``: ``(B,)`` defender / (attacker + 1) on
        opponent squares.
      - ``defense_gap``: ``(B,)`` mean ``p_attack - p_defend`` on
        opponent squares -- positive when attacks outweigh defenses.
      - ``transport_imbalance``: ``(B,)`` ``mean(p_attack) -
        mean(p_defend)`` over the whole board.
      - ``sheaf_tension``: ``(B,)`` mean absolute SEE over opponent
        squares.
      - ``target_count``: ``(B,)`` number of opponent-piece squares.
      - ``trunk_energy``: ``(B,)`` mean-square trunk activation.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        exchange_temperature: float = 1.0,
        exchange_depth: int = 4,
        head_hidden: int = 0,
        piece_values: tuple[float, ...] = DEFAULT_PIECE_VALUES,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if exchange_temperature <= 0.0:
            raise ValueError("exchange_temperature must be > 0")
        if exchange_depth < 1:
            raise ValueError("exchange_depth must be >= 1")
        if len(piece_values) != NUM_PIECE_TYPES:
            raise ValueError(f"piece_values must have {NUM_PIECE_TYPES} entries")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.depth = int(depth)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.exchange_temperature = float(exchange_temperature)
        self.exchange_depth = int(exchange_depth)
        self.head_hidden = (
            int(head_hidden) if head_hidden else max(self.channels // 2, 8)
        )

        self.register_buffer(
            "piece_values",
            torch.tensor(list(piece_values), dtype=torch.float32),
        )

        self.trunk = BoardConvStem(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            use_batchnorm=self.use_batchnorm,
        )

        # Graph-context stack: own_t (6) + opp_t (6) + own_all (1) + opp_all (1)
        # = 14 piece-mask channels concatenated to the trunk features.
        graph_in = self.channels + 2 * NUM_PIECE_TYPES + 2

        self.attacker_intensity_head = _PieceTypeHead(
            graph_in, self.head_hidden, 1, self.dropout
        )
        self.defender_intensity_head = _PieceTypeHead(
            graph_in, self.head_hidden, 1, self.dropout
        )
        self.attacker_type_head = _PieceTypeHead(
            graph_in, self.head_hidden, NUM_PIECE_TYPES, self.dropout
        )
        self.defender_type_head = _PieceTypeHead(
            graph_in, self.head_hidden, NUM_PIECE_TYPES, self.dropout
        )

        # Final classifier consumes the exchange/attacker/defender pools
        # plus the eight graph-network summary scalars.
        summary_dim = 8
        head_in = 3 * self.channels + summary_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def _piece_value_field(self, piece_planes: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 8, 8)`` value-of-piece field from a 6-channel mask."""
        values = self.piece_values.view(1, NUM_PIECE_TYPES, 1, 1)
        return (piece_planes * values).sum(dim=1)

    def _value_distribution_field(
        self, type_logits: torch.Tensor, available_mask: torch.Tensor
    ) -> torch.Tensor:
        """Distribution-weighted piece-value field at every square.

        ``type_logits``: ``(B, 6, 8, 8)`` raw logits over piece types.
        ``available_mask``: ``(B, 6)`` whether each side has at least
        one piece of that type left on the board (we cannot use a
        piece type as an attacker / defender if none are present).
        Returns ``(B, 8, 8)``.
        """
        # Mask piece types that the side does not actually have.
        availability = available_mask.view(*available_mask.shape, 1, 1)
        masked_logits = torch.where(
            availability > 0.5,
            type_logits,
            torch.full_like(type_logits, NEG_INF),
        )
        # If a row has no available piece types at all, fall back to a
        # uniform distribution; this only matters when the side has no
        # pieces, in which case the field is multiplied by zero anyway.
        no_pieces = available_mask.sum(dim=1) <= 0.5  # (B,)
        if no_pieces.any():
            uniform = torch.zeros_like(type_logits)
            no_pieces_4d = no_pieces.view(-1, 1, 1, 1)
            masked_logits = torch.where(no_pieces_4d, uniform, masked_logits)
        dist = F.softmax(masked_logits, dim=1)
        values = self.piece_values.view(1, NUM_PIECE_TYPES, 1, 1)
        return (dist * values).sum(dim=1)

    def _soft_static_exchange(
        self,
        v_target: torch.Tensor,
        v_a: torch.Tensor,
        v_d: torch.Tensor,
        p_a: torch.Tensor,
        p_d: torch.Tensor,
    ) -> torch.Tensor:
        """Bounded-depth differentiable static exchange evaluation.

        Implements the recursion ``see_k = v_top - p_resp * max(0,
        see_{k-1})`` where the side that just moved alternates and the
        ``max(0, .)`` is the SEE "stop now" option. Returns the SEE
        value at the root of the exchange (depth-``exchange_depth``
        unroll).
        """
        # Build the deepest leaf first and unroll outward. At leaf
        # depth the side that captured most recently keeps the piece
        # they just took (no further recapture modeled).
        # Sequence of values that get won by alternating sides as we
        # unroll: target, attacker, defender, attacker, ...
        sequence = [v_target]
        # Probabilities the *next* recapture happens; alternates p_d,
        # p_a, p_d, ...
        responses = []
        for step in range(1, self.exchange_depth + 1):
            if step % 2 == 1:
                # Defender's potential recapture; if it lands, defender
                # wins our attacker's value back from us.
                sequence.append(v_a)
                responses.append(p_d)
            else:
                # Our potential re-recapture; we win their defender.
                sequence.append(v_d)
                responses.append(p_a)
        # Unroll: at the last step the side that just moved keeps the
        # last piece (no further response).
        running = sequence[-1].clone()
        for k in range(self.exchange_depth - 1, -1, -1):
            # The side considering the (k+1)th capture decides whether
            # to take. They take iff continuing yields positive value.
            running = sequence[k] - responses[k] * running.clamp_min(0.0)
        return running

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.trunk(x)  # (B, C, 8, 8)

        own_planes, opp_planes = _own_and_opp_piece_planes(x)  # (B, 6, 8, 8)
        own_all = own_planes.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        opp_all = opp_planes.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        graph_context = torch.cat([own_planes, opp_planes, own_all, opp_all], dim=1)
        head_in_field = torch.cat([feats, graph_context], dim=1)

        attacker_logits = self.attacker_intensity_head(head_in_field).squeeze(1)
        defender_logits = self.defender_intensity_head(head_in_field).squeeze(1)
        attacker_intensity = torch.sigmoid(attacker_logits)
        defender_intensity = torch.sigmoid(defender_logits)

        # Per-side availability of each piece type (does a same-type
        # piece exist anywhere on the board?). The cheapest attacker
        # value is forced to come from the side-to-move's actual piece
        # inventory; same for defender.
        own_available = (own_planes.amax(dim=(2, 3)) > 0.5).float()  # (B, 6)
        opp_available = (opp_planes.amax(dim=(2, 3)) > 0.5).float()

        attacker_type_logits = self.attacker_type_head(head_in_field)
        defender_type_logits = self.defender_type_head(head_in_field)
        v_a = self._value_distribution_field(attacker_type_logits, own_available)
        v_d = self._value_distribution_field(defender_type_logits, opp_available)

        v_target = self._piece_value_field(opp_planes)  # (B, 8, 8)
        target_mask = opp_all.squeeze(1)  # (B, 8, 8) in {0,1}

        # Differentiable static exchange evaluation.
        exchange_score = self._soft_static_exchange(
            v_target=v_target,
            v_a=v_a,
            v_d=v_d,
            p_a=attacker_intensity,
            p_d=defender_intensity,
        )
        exchange_soundness = torch.sigmoid(exchange_score / self.exchange_temperature)

        # Bottleneck pool: feature mass through |see| over target squares.
        target_count = target_mask.flatten(1).sum(dim=1)  # (B,)
        target_weights = target_mask * exchange_score.abs()
        weight_sum = target_weights.flatten(1).sum(dim=1).clamp_min(1e-6)
        target_weights_norm = target_weights / weight_sum.view(-1, 1, 1)
        # Rows with no targets fall back to the uniform-over-board pool
        # so the head still sees a deterministic signal.
        no_targets = (target_count <= 0.5).view(-1, 1, 1)
        uniform_field = torch.full_like(target_weights_norm, 1.0 / 64.0)
        target_weights_norm = torch.where(no_targets, uniform_field, target_weights_norm)
        exchange_pool = (
            feats * target_weights_norm.unsqueeze(1)
        ).flatten(2).sum(dim=2)  # (B, C)

        # Attacker / defender pools share the same recipe, weighted by
        # the learned attack/defense intensities (normalised across the
        # 64 squares).
        flat_attack = attacker_intensity.flatten(1)
        attack_norm = flat_attack / flat_attack.sum(dim=1, keepdim=True).clamp_min(1e-6)
        attacker_pool = (
            feats * attack_norm.view(-1, 1, 8, 8)
        ).flatten(2).sum(dim=2)  # (B, C)
        flat_defend = defender_intensity.flatten(1)
        defend_norm = flat_defend / flat_defend.sum(dim=1, keepdim=True).clamp_min(1e-6)
        defender_pool = (
            feats * defend_norm.view(-1, 1, 8, 8)
        ).flatten(2).sum(dim=2)  # (B, C)

        # Graph-network summary scalars.
        target_count_safe = target_count.clamp_min(1.0)
        target_flat = target_mask.flatten(1)
        see_flat = exchange_score.flatten(1)
        see_targets = see_flat * target_flat
        max_see_target = see_targets.masked_fill(target_flat <= 0.5, NEG_INF).max(dim=1).values
        # Replace fallback rows (no targets) with a deterministic 0.0.
        max_see_target = torch.where(
            target_count > 0.5, max_see_target, torch.zeros_like(max_see_target)
        )
        mean_see_target = see_targets.sum(dim=1) / target_count_safe
        unsound = ((see_flat <= 0.0).float() * target_flat).sum(dim=1) / target_count_safe
        graph_pressure = (attacker_intensity.flatten(1) * target_flat).sum(dim=1) / target_count_safe
        defender_pressure = (defender_intensity.flatten(1) * target_flat).sum(dim=1) / target_count_safe
        attacker_target_pressure = graph_pressure
        reply_pressure = defender_pressure / (attacker_target_pressure + 1.0)
        defense_gap = attacker_target_pressure - defender_pressure
        transport_imbalance = (
            attacker_intensity.flatten(1).mean(dim=1)
            - defender_intensity.flatten(1).mean(dim=1)
        )
        sheaf_tension = (
            see_flat.abs() * target_flat
        ).sum(dim=1) / target_count_safe

        with torch.no_grad():
            trunk_energy = feats.square().mean(dim=(1, 2, 3))

        summary = torch.stack(
            [
                max_see_target,
                mean_see_target,
                unsound,
                graph_pressure,
                reply_pressure,
                defense_gap,
                transport_imbalance,
                sheaf_tension,
            ],
            dim=1,
        )

        head_in = torch.cat([exchange_pool, attacker_pool, defender_pool, summary], dim=1)
        logits = self.head(head_in)
        logits = _format_logits(logits, self.num_classes)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "attacker_intensity": attacker_intensity,
            "defender_intensity": defender_intensity,
            "attacker_value_field": v_a,
            "defender_value_field": v_d,
            "target_value_field": v_target,
            "exchange_score_field": exchange_score,
            "exchange_soundness_field": exchange_soundness,
            "target_mask": target_mask,
            "max_see_target": max_see_target,
            "mean_see_target": mean_see_target,
            "frac_unsound_targets": unsound,
            "graph_pressure": graph_pressure,
            "reply_pressure": reply_pressure,
            "defense_gap": defense_gap,
            "transport_imbalance": transport_imbalance,
            "sheaf_tension": sheaf_tension,
            "target_count": target_count,
            "trunk_energy": trunk_energy,
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_exchange_soundness_graph_network_from_config(
    config: dict[str, Any],
) -> ExchangeSoundnessGraphNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    return ExchangeSoundnessGraphNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        depth=int(cfg.pop("depth", 2)),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        exchange_temperature=float(cfg.pop("exchange_temperature", 1.0)),
        exchange_depth=int(cfg.pop("exchange_depth", 4)),
        head_hidden=int(cfg.pop("head_hidden", 0)),
    )


__all__ = [
    "ExchangeSoundnessGraphNetwork",
    "build_exchange_soundness_graph_network_from_config",
]
