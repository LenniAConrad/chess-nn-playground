"""King-Zone Reply Pressure primitive (p051, KZRP).

Source: ``ideas/research/primitives/external_46_king_zone_reply_pressure_primitive.md``
(the source markdown uses the working name ``p045``; the registry slot
was reassigned to ``p051`` because the ``p045`` and ``p050`` slots were
already taken). The spec's recommended phase-1 ("additive gated hybrid
on i018") is implemented here as a gated additive head on top of the
i193 ``ExchangeThenKingDualStreamNetwork`` trunk -- the repo's current
strong baseline -- rather than i018 itself; the gating, fusion, and
output contract match the p049 / p050 plug-in pattern.

Thesis. King safety on the puzzle-binary surface is not a flat count of
"attacks near the king". The discriminative signal is *whether the
defender's legal-reply families around the king have collapsed*. The
spec decomposes that into five interpretable terms:

  * **Zone pressure** (ZP) -- weighted attacker mass minus pin-
    discounted defender mass on the king-zone squares (king square,
    empty ring, occupied ring, and the three forward-rank squares).
  * **Fake-defense loss** (FD) -- the gap between *nominal* defender
    mass (sum of all defenders touching a square) and *free* defender
    mass (excluding pinned defenders, whose support is illegal).
  * **Escape closure** (EP) -- partition of the 8 adjacent king
    squares into ``live`` (empty and unattacked), ``sealed`` (empty
    but attacked), and ``blocked`` (occupied by a defender).
  * **Current check severity** -- weighted attacker mass on the king
    square itself (a non-zero value means at least one piece is
    delivering check).
  * **Reply-capacity proxy** -- a cheap log-bound on the number of
    legal reply families the defender still has: king escapes plus
    free defenders sitting on king-zone squares.

Each side σ ∈ {us, them} produces an 8-feature vector and the operator
is the 32-dim concatenation ``[S_us, S_them, S_us - S_them,
|S_us - S_them|]``. The vector is appended to the i193 joint pool and
fed through the gated additive head pattern (LayerNorm + Linear + GELU
+ Linear, gate initialised near-closed at -2.0 so the i193 baseline is
exactly recovered at the start of training).

Inputs are exactly the ``simple_18`` ``(B, 18, 8, 8)`` current-board
tensor. CRTK metadata, source labels, verification flags, engine scores
and principal variations are *not* consumed. Geometry buffers
(``geom_attacks``, ``between``, ``ray_step_index``, ``ring_mask``,
``front_mask_w``, ``front_mask_b``) are rule-derived and carry zero
parameters.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.ray_geometry import (
    DIRECTIONS,
    NUM_DIRECTIONS,
    RAY_MAX_LEN,
    RayGeometry,
    SQUARES,
)
from chess_nn_playground.models.primitives.trunk_features import trunk_joint_features
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


WHITE = 0
BLACK = 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5
NUM_PIECE_TYPES = 6
NUM_PIECE_CHANNELS = 12
SIDE_VECTOR_DIM = 8
OPERATOR_OUTPUT_DIM = 4 * SIDE_VECTOR_DIM   # us, them, diff, |diff| = 32

# CPW-style attack-unit priors (P=1, N=2, B=2, R=3, Q=5). King is given a
# small nonzero weight so it can contribute as a contact attacker on
# adjacent squares; values are softplus-bounded at training time so they
# stay positive.
DEFAULT_ATTACK_UNITS: tuple[float, ...] = (1.0, 2.0, 2.0, 3.0, 5.0, 1.0)

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    # Primary falsifier: drop the forward-rank zone term η * front_pressure.
    # If lift survives, the front-rank extension is not load-bearing and
    # the operator collapses to a plain ring-pressure scalar.
    "no_front_zone",
    # Set pin indicator π = 0 everywhere. Tests whether the
    # nominal-vs-free defender split (fake-defense loss term) matters.
    "no_pins",
    # Replace the (king_sq=4, empty_ring=3, occupied_ring=2) zone weights
    # with uniform 1.0. Tests whether unequal weighting of king-zone
    # squares is load-bearing.
    "uniform_zone_weights",
    # Collapse the three escape-class counts (live, sealed, blocked) into
    # a single sum. Tests whether escape decomposition is load-bearing.
    "no_escape_decomp",
    # Set piece attack-units u(P,N,B,R,Q,K) = 1 (uniform). Tests whether
    # CPW-style attack-unit weighting matters.
    "uniform_units",
    # Set them-side stats to zero. Tests whether the side-to-move
    # asymmetry term S_us - S_them is load-bearing.
    "no_asymmetry",
    # Bypass the primitive entirely -- recovers the i193 baseline.
    "zero_delta",
    "trunk_only",
    # Pin the gate at 1.0. Tests gate load-bearing.
    "disable_gate",
)


def _direction_family_masks() -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(dir_is_orth, dir_is_diag)`` of shape ``(8,)`` float32."""
    orth = torch.zeros(NUM_DIRECTIONS, dtype=torch.float32)
    diag = torch.zeros(NUM_DIRECTIONS, dtype=torch.float32)
    for d, (dr, df) in enumerate(DIRECTIONS):
        if dr == 0 or df == 0:
            orth[d] = 1.0
        else:
            diag[d] = 1.0
    return orth, diag


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _build_attack_and_between() -> tuple[torch.Tensor, torch.Tensor]:
    """Construct rule-derived attack and between buffers.

    Returns
    -------
    geom_attacks: ``(6, 2, 64, 64)`` float, 1 iff ``(piece-type, colour,
        source)`` attacks ``target`` ignoring blockers. Sliders are
        gated at runtime by the ``between`` mask.
    between: ``(64, 64, 64)`` float, ``between[s, t, k]`` is 1 iff
        ``k`` lies strictly between ``s`` and ``t`` on an aligned line.
    """
    geom_attacks = torch.zeros(NUM_PIECE_TYPES, 2, SQUARES, SQUARES, dtype=torch.float32)
    between = torch.zeros(SQUARES, SQUARES, SQUARES, dtype=torch.float32)
    knight_offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    king_offsets = [(r, f) for r in (-1, 0, 1) for f in (-1, 0, 1) if r != 0 or f != 0]
    bishop_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    rook_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for source in range(SQUARES):
        sr, sf = source // 8, source % 8
        for target in range(SQUARES):
            tr, tf = target // 8, target % 8
            if source == target:
                continue
            aligned = (sr == tr) or (sf == tf) or (abs(tr - sr) == abs(tf - sf))
            if aligned:
                row_step = _sign(tr - sr)
                file_step = _sign(tf - sf)
                row, file = sr + row_step, sf + file_step
                while (row, file) != (tr, tf):
                    between[source, target, row * 8 + file] = 1.0
                    row += row_step
                    file += file_step
        for color in (WHITE, BLACK):
            pawn_forward = -1 if color == WHITE else 1
            for fd in (-1, 1):
                r, f = sr + pawn_forward, sf + fd
                if 0 <= r < 8 and 0 <= f < 8:
                    geom_attacks[PAWN, color, source, r * 8 + f] = 1.0
            for rd, fd in knight_offsets:
                r, f = sr + rd, sf + fd
                if 0 <= r < 8 and 0 <= f < 8:
                    geom_attacks[KNIGHT, color, source, r * 8 + f] = 1.0
            for rd, fd in king_offsets:
                r, f = sr + rd, sf + fd
                if 0 <= r < 8 and 0 <= f < 8:
                    geom_attacks[KING, color, source, r * 8 + f] = 1.0
            for piece, dirs in (
                (BISHOP, bishop_dirs),
                (ROOK, rook_dirs),
                (QUEEN, bishop_dirs + rook_dirs),
            ):
                for rd, fd in dirs:
                    r, f = sr + rd, sf + fd
                    while 0 <= r < 8 and 0 <= f < 8:
                        geom_attacks[piece, color, source, r * 8 + f] = 1.0
                        r += rd
                        f += fd
    return geom_attacks, between


def _build_king_zone_masks() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(ring_mask, front_mask_w, front_mask_b)``.

    All tensors are ``(64, 64)`` float32.

    * ``ring_mask[k, q] = 1`` iff ``q`` is one of the 8 squares in
      ``N_8(k)`` (Chebyshev distance 1, excluding the king square ``k``
      itself).
    * ``front_mask_w[k, q] = 1`` iff ``q`` lies on the three squares
      one rank further in the *attacker* direction beyond the front
      edge of ``N_8(k)``, **when the defender king on ``k`` is white**.
      White's attacker is black, who attacks from upper ranks; in
      ``simple_18`` row indexing (row 0 = rank 8), the front rank is
      at ``row(k) - 2``.
    * ``front_mask_b[k, q] = 1`` likewise when the defender king is
      black: the attacker is white, attacking from the lower ranks
      (toward row 7), so the front rank is at ``row(k) + 2``.

    Off-board front squares are clipped automatically.
    """
    ring = torch.zeros(SQUARES, SQUARES, dtype=torch.float32)
    front_w = torch.zeros(SQUARES, SQUARES, dtype=torch.float32)
    front_b = torch.zeros(SQUARES, SQUARES, dtype=torch.float32)
    for ks in range(SQUARES):
        kr, kf = ks // 8, ks % 8
        for q in range(SQUARES):
            qr, qf = q // 8, q % 8
            if (kr, kf) == (qr, qf):
                continue
            if abs(qr - kr) <= 1 and abs(qf - kf) <= 1:
                ring[ks, q] = 1.0
        front_row_w = kr - 2
        if 0 <= front_row_w < 8:
            for fd in (-1, 0, 1):
                qf = kf + fd
                if 0 <= qf < 8:
                    front_w[ks, front_row_w * 8 + qf] = 1.0
        front_row_b = kr + 2
        if 0 <= front_row_b < 8:
            for fd in (-1, 0, 1):
                qf = kf + fd
                if 0 <= qf < 8:
                    front_b[ks, front_row_b * 8 + qf] = 1.0
    return ring, front_w, front_b


class KingZoneReplyPressureBuilder(nn.Module):
    """King-zone weighted attack / defense / escape / reply builder.

    Forward signature::

        forward(piece_state_absolute, stm, ablation="none") -> dict

    where ``piece_state_absolute`` is ``(B, 12, 64)`` in the original
    simple_18 piece-plane order (P, N, B, R, Q, K, p, n, b, r, q, k)
    and ``stm`` is the side-to-move scalar per sample (``(B,)``, 1.0 if
    white-to-move, 0.0 otherwise).

    Outputs an ``operator_vector`` of shape ``(B, 32)`` plus a dict of
    per-side diagnostics (``us`` / ``them`` 8-feature side vectors and
    spatial reduction means used by the report template).
    """

    def __init__(self) -> None:
        super().__init__()
        geom_attacks, between = _build_attack_and_between()
        # geom_attacks: (6, 2, 64, 64); between: (64, 64, 64).
        self.register_buffer("geom_attacks", geom_attacks, persistent=False)
        self.register_buffer("between", between, persistent=False)
        # Ray geometry (shared with p049 / p050) used by the pin detector.
        geom = RayGeometry.build()
        self.register_buffer("ray_step_index", geom.step_index, persistent=False)
        self.register_buffer("ray_step_mask", geom.step_mask, persistent=False)
        orth, diag = _direction_family_masks()
        self.register_buffer("dir_is_orth", orth, persistent=False)
        self.register_buffer("dir_is_diag", diag, persistent=False)
        # King-zone geometry. ring_mask is colour-agnostic. front_mask_*
        # depends on which colour the defender king is.
        ring_mask, front_mask_w, front_mask_b = _build_king_zone_masks()
        self.register_buffer("ring_mask", ring_mask, persistent=False)
        self.register_buffer("front_mask_w", front_mask_w, persistent=False)
        self.register_buffer("front_mask_b", front_mask_b, persistent=False)
        # Learnable, softplus-bounded attack-unit field.
        self.attack_unit_logits = nn.Parameter(torch.tensor(DEFAULT_ATTACK_UNITS))
        # Pin-discount λ (sigmoid-bounded → [0, 1]).
        self.pin_discount_logit = nn.Parameter(torch.tensor(1.1))   # ≈ 0.75
        # Defense discount λ_def used in ZP = [A - λ_def · D_free]_+.
        self.def_discount_logit = nn.Parameter(torch.tensor(0.0))   # ≈ 0.5
        # Forward-zone strength η (softplus-bounded).
        self.front_strength_logit = nn.Parameter(torch.tensor(0.0))  # ≈ 0.69
        # Zone-square weights as (king_sq, empty_ring, occupied_ring).
        # Bias-style logits so softplus gives initial ~ (4, 3, 2).
        self.zone_weight_logits = nn.Parameter(
            torch.tensor([4.0, 3.0, 2.0]).log()
        )
        # Escape-class weights for the reply proxy (α1 sealed,
        # α2 blocked, α3 live). Sigmoid-bounded to (0, 1).
        self.escape_weight_logits = nn.Parameter(
            torch.tensor([0.6, 0.4, 1.0]).log()
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _gather_scalar(self, scalar: torch.Tensor) -> torch.Tensor:
        flat = self.ray_step_index.reshape(-1)
        gathered = scalar[:, flat].reshape(
            scalar.shape[0], NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN
        )
        mask = self.ray_step_mask.to(device=scalar.device, dtype=scalar.dtype).view(
            1, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN
        )
        return gathered * mask

    def _enemy_sliders_per_direction(self, enemy_piece_state: torch.Tensor) -> torch.Tensor:
        rook = enemy_piece_state[:, ROOK]
        bishop = enemy_piece_state[:, BISHOP]
        queen = enemy_piece_state[:, QUEEN]
        orth = self.dir_is_orth.view(1, NUM_DIRECTIONS, 1)
        diag = self.dir_is_diag.view(1, NUM_DIRECTIONS, 1)
        slider = (
            queen.unsqueeze(1)
            + rook.unsqueeze(1) * orth
            + bishop.unsqueeze(1) * diag
        )
        return slider.clamp(0.0, 1.0)

    def _pin_mask(
        self,
        piece_state: torch.Tensor,
        occupancy: torch.Tensor,
        own_king_sq: torch.Tensor,
        own_any: torch.Tensor,
        enemy_sliders_per_dir: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``(B, 64)`` pinned-defender indicator (cumsum detector).

        Same shape and semantics as the helper in p050: a defender ``d``
        is flagged when a ray from its own king has ``d`` as the first
        occupant and an enemy slider firing along the same direction as
        the second occupant.
        """
        occ_seq = self._gather_scalar(occupancy)
        own_any_seq = self._gather_scalar(own_any)
        enemy_slider_seq = torch.zeros(
            piece_state.shape[0],
            NUM_DIRECTIONS,
            SQUARES,
            RAY_MAX_LEN,
            device=piece_state.device,
            dtype=piece_state.dtype,
        )
        for d in range(NUM_DIRECTIONS):
            scalar = enemy_sliders_per_dir[:, d]
            enemy_slider_seq[:, d] = self._gather_scalar(scalar)[:, d]

        occ_bool = (occ_seq > 0.5).to(dtype=piece_state.dtype)
        cum_occ = occ_bool.cumsum(dim=-1)
        first_step = occ_bool * (cum_occ <= 1.0).to(dtype=piece_state.dtype)
        second_step = occ_bool * ((cum_occ >= 1.5) & (cum_occ <= 2.5)).to(
            dtype=piece_state.dtype
        )

        first_own = first_step * own_any_seq
        second_enemy_slider = second_step * enemy_slider_seq

        has_second_slider = second_enemy_slider.sum(dim=-1)
        first_own_marked = first_own * has_second_slider.unsqueeze(-1)
        flat_idx = self.ray_step_index.reshape(-1)
        king_sel = own_king_sq.view(piece_state.shape[0], 1, SQUARES, 1)
        marks = (first_own_marked * king_sel).reshape(piece_state.shape[0], -1)
        pinned = piece_state.new_zeros(piece_state.shape[0], SQUARES)
        pinned.scatter_add_(
            dim=1,
            index=flat_idx.view(1, -1).expand(piece_state.shape[0], -1),
            src=marks,
        )
        return pinned.clamp(0.0, 1.0)

    def _attack_masses(
        self,
        piece_state: torch.Tensor,
        occupancy: torch.Tensor,
        units: torch.Tensor,
        pin_per_color: tuple[torch.Tensor, torch.Tensor] | None,
        pin_discount: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return weighted per-colour attack masses ``(B, 64)``.

        Args:
            piece_state: ``(B, 12, 64)`` absolute piece planes.
            occupancy: ``(B, 64)`` board occupancy.
            units: ``(6,)`` per-piece-type attack-unit weights.
            pin_per_color: optional ``(pinned_white, pinned_black)``;
                if provided, also returns *free* attack masses with the
                pinned contribution downweighted by ``pin_discount``.
            pin_discount: scalar in ``[0, 1]`` -- multiplier for the
                pinned-source contribution to the *free* mass (1.0 =
                zero out, 0.0 = no discount).

        Returns
        -------
        attack_w_nom, attack_b_nom, attack_w_free, attack_b_free
            All ``(B, 64)`` weighted square-incidence masses.
        """
        device = piece_state.device
        dtype = piece_state.dtype
        batch = piece_state.shape[0]
        geom = self.geom_attacks.to(device=device, dtype=dtype)
        between = self.between.to(device=device, dtype=dtype)

        blocked = torch.einsum("stk,bk->bst", between, occupancy)
        clear = (blocked <= 0.5).to(dtype=dtype)
        ones = torch.ones_like(clear)

        if pin_per_color is None:
            pinned_white = piece_state.new_zeros(batch, SQUARES)
            pinned_black = piece_state.new_zeros(batch, SQUARES)
        else:
            pinned_white, pinned_black = pin_per_color

        discount = pin_discount.to(device=device, dtype=dtype)

        attack_nom: list[torch.Tensor] = []
        attack_free: list[torch.Tensor] = []
        for color, pinned in ((WHITE, pinned_white), (BLACK, pinned_black)):
            nom = piece_state.new_zeros(batch, SQUARES)
            free = piece_state.new_zeros(batch, SQUARES)
            free_weight = 1.0 - discount * pinned                    # (B, 64)
            for piece in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING):
                ch = piece if color == WHITE else NUM_PIECE_TYPES + piece
                src = piece_state[:, ch]                              # (B, 64)
                gate = clear if piece in (BISHOP, ROOK, QUEEN) else ones
                rel = src.unsqueeze(-1) * geom[piece, color].unsqueeze(0) * gate
                u = units[piece].to(dtype=dtype)
                rel_sum = rel.sum(dim=1)                             # (B, 64)
                nom = nom + u * rel_sum
                # Free version: pin-discount the contribution from
                # pinned sources. ``free_weight`` is broadcast over
                # targets via the ``unsqueeze`` on rel.
                rel_free = (src * free_weight).unsqueeze(-1) * geom[piece, color].unsqueeze(0) * gate
                free = free + u * rel_free.sum(dim=1)
            attack_nom.append(nom)
            attack_free.append(free)
        return attack_nom[0], attack_nom[1], attack_free[0], attack_free[1]

    # ------------------------------------------------------------------
    # Side stats
    # ------------------------------------------------------------------

    def _side_vector(
        self,
        attack: torch.Tensor,         # (B, 64) weighted attacker mass
        def_nom: torch.Tensor,        # (B, 64) nominal defender mass
        def_free: torch.Tensor,       # (B, 64) free defender mass
        defender_king: torch.Tensor,  # (B, 64) defender king one-hot
        defender_any: torch.Tensor,   # (B, 64) defender side occupancy
        total_occupancy: torch.Tensor,  # (B, 64) board occupancy
        defender_is_white: torch.Tensor,  # (B, 1) 1.0 if defender is white
        ablation: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        eps = 1.0e-6
        dtype = attack.dtype

        ring = torch.einsum("bk,kq->bq", defender_king, self.ring_mask.to(dtype=dtype))
        front_w = torch.einsum("bk,kq->bq", defender_king, self.front_mask_w.to(dtype=dtype))
        front_b = torch.einsum("bk,kq->bq", defender_king, self.front_mask_b.to(dtype=dtype))
        # defender_is_white: (B, 1).
        front = defender_is_white * front_w + (1.0 - defender_is_white) * front_b
        if ablation == "no_front_zone":
            front = torch.zeros_like(front)

        # Discounted-defense quantity used in ZP.
        lambda_def = torch.sigmoid(self.def_discount_logit.to(dtype=dtype))
        net_pressure = torch.relu(attack - lambda_def * def_free)

        # Per-square zone weights.
        weights = torch.nn.functional.softplus(
            self.zone_weight_logits.to(dtype=dtype)
        )
        if ablation == "uniform_zone_weights":
            weights = torch.ones_like(weights)
        king_w = weights[0]
        empty_ring_w = weights[1]
        occ_ring_w = weights[2]

        empty_mask = (1.0 - total_occupancy).clamp(0.0, 1.0)
        # Defender occupies adjacent ring square = blocked escape.
        defender_on_ring = ring * defender_any
        # Empty ring square (not occupied by anything).
        empty_ring = ring * empty_mask
        # Occupied (by either side) ring square — used for occupied-ring weight.
        occupied_ring = ring * total_occupancy

        zone_w_map = (
            king_w * defender_king
            + empty_ring_w * empty_ring
            + occ_ring_w * occupied_ring
        )
        eta = torch.nn.functional.softplus(
            self.front_strength_logit.to(dtype=dtype)
        )

        zone_core_term = (zone_w_map * net_pressure).sum(dim=1)
        zone_front_term = eta * (front * net_pressure).sum(dim=1)
        zone_pressure = zone_core_term + zone_front_term

        # Fake-defense loss. Sum over the union of ring and front of
        # (D_nom - D_free) — this is exactly the pin-discount mass.
        zone_any = (ring + front + defender_king).clamp(0.0, 1.0)
        fake_defense_loss = (zone_any * (def_nom - def_free).clamp(min=0.0)).sum(dim=1)

        # Escape classes on the immediate king ring.
        attacked = (attack > eps).to(dtype=dtype)
        live = (ring * empty_mask * (1.0 - attacked)).sum(dim=1)
        sealed = (ring * empty_mask * attacked).sum(dim=1)
        blocked_escapes = defender_on_ring.sum(dim=1)
        if ablation == "no_escape_decomp":
            total_escapes = live + sealed + blocked_escapes
            # Collapse to a single signal: keep total in `live` slot, zero others.
            live = total_escapes
            sealed = torch.zeros_like(sealed)
            blocked_escapes = torch.zeros_like(blocked_escapes)

        # Current check severity: weighted attacker mass on the king square.
        king_attack_mass = (defender_king * attack).sum(dim=1)

        # Forward-zone pressure (raw, before η weighting) -- exposed
        # separately so the head can re-weight if needed.
        front_attack_mass = (front * net_pressure).sum(dim=1)

        # Reply-capacity proxy. log(1 + active_escapes + α · free_defense_in_ring).
        escape_weights = torch.sigmoid(
            self.escape_weight_logits.to(dtype=dtype)
        )
        ring_free_defense = (ring * def_free).sum(dim=1)
        reply_proxy = torch.log1p(
            live
            + escape_weights[0] * sealed
            + escape_weights[1] * blocked_escapes
            + escape_weights[2] * ring_free_defense
        )

        side_vec = torch.stack(
            [
                zone_pressure,
                fake_defense_loss,
                live,
                sealed,
                blocked_escapes,
                king_attack_mass,
                front_attack_mass,
                reply_proxy,
            ],
            dim=1,
        )                                                              # (B, 8)
        side_vec = torch.nan_to_num(side_vec, nan=0.0, posinf=0.0, neginf=0.0)

        aux = {
            "zone_pressure": zone_pressure.detach(),
            "fake_defense_loss": fake_defense_loss.detach(),
            "live_escapes": live.detach(),
            "sealed_escapes": sealed.detach(),
            "blocked_escapes": blocked_escapes.detach(),
            "king_attack_mass": king_attack_mass.detach(),
            "front_attack_mass": front_attack_mass.detach(),
            "reply_proxy": reply_proxy.detach(),
            "ring_free_defense": ring_free_defense.detach(),
            "net_pressure_mean": net_pressure.mean(dim=1).detach(),
            "net_pressure_max": net_pressure.amax(dim=1).detach(),
        }
        return side_vec, aux

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        piece_state_absolute: torch.Tensor,
        stm: torch.Tensor,
        ablation: str = "none",
    ) -> dict[str, torch.Tensor]:
        if piece_state_absolute.dim() != 3 or piece_state_absolute.shape[1] != NUM_PIECE_CHANNELS:
            raise ValueError(
                "piece_state_absolute must be (B, 12, 64) in P,N,B,R,Q,K,p,n,b,r,q,k order; "
                f"got {tuple(piece_state_absolute.shape)}"
            )
        if stm.dim() != 1 or stm.shape[0] != piece_state_absolute.shape[0]:
            raise ValueError(f"stm must be (B,); got {tuple(stm.shape)}")
        dtype = piece_state_absolute.dtype
        batch = piece_state_absolute.shape[0]
        stm_g = stm.to(dtype=dtype).clamp(0.0, 1.0).view(batch, 1)

        ps = piece_state_absolute.clamp(0.0, 1.0)
        occupancy = ps.sum(dim=1).clamp(0.0, 1.0)
        white_any = ps[:, :NUM_PIECE_TYPES].sum(dim=1).clamp(0.0, 1.0)
        black_any = ps[:, NUM_PIECE_TYPES:].sum(dim=1).clamp(0.0, 1.0)
        white_king_sq = ps[:, KING].clamp(0.0, 1.0)
        black_king_sq = ps[:, NUM_PIECE_TYPES + KING].clamp(0.0, 1.0)

        # Pin detection per side. For pins on WHITE defenders, the
        # second occupant is a BLACK slider firing along the same dir
        # as the ray from the white king through the white piece.
        black_planes = ps[:, NUM_PIECE_TYPES:]
        white_planes = ps[:, :NUM_PIECE_TYPES]
        black_sliders_per_dir = self._enemy_sliders_per_direction(black_planes)
        white_sliders_per_dir = self._enemy_sliders_per_direction(white_planes)
        if ablation == "no_pins":
            pinned_white = torch.zeros_like(white_any)
            pinned_black = torch.zeros_like(black_any)
        else:
            pinned_white = self._pin_mask(
                ps,
                occupancy,
                own_king_sq=white_king_sq,
                own_any=white_any,
                enemy_sliders_per_dir=black_sliders_per_dir,
            )
            pinned_black = self._pin_mask(
                ps,
                occupancy,
                own_king_sq=black_king_sq,
                own_any=black_any,
                enemy_sliders_per_dir=white_sliders_per_dir,
            )

        if ablation == "uniform_units":
            units = torch.ones(NUM_PIECE_TYPES, device=ps.device, dtype=dtype)
        else:
            units = torch.nn.functional.softplus(
                self.attack_unit_logits.to(device=ps.device, dtype=dtype)
            )

        pin_discount = torch.sigmoid(self.pin_discount_logit.to(dtype=dtype))
        attack_w_nom, attack_b_nom, attack_w_free, attack_b_free = self._attack_masses(
            ps, occupancy, units, (pinned_white, pinned_black), pin_discount=pin_discount
        )

        # σ = us: attacker = mover, defender = opponent.
        # Mover is WHITE when stm == 1, BLACK when stm == 0.
        us_attack = stm_g * attack_w_nom + (1.0 - stm_g) * attack_b_nom
        them_attack = stm_g * attack_b_nom + (1.0 - stm_g) * attack_w_nom
        # Defender nominal/free mass (defender = opposite of attacker).
        them_def_nom = stm_g * attack_b_nom + (1.0 - stm_g) * attack_w_nom
        them_def_free = stm_g * attack_b_free + (1.0 - stm_g) * attack_w_free
        us_def_nom = stm_g * attack_w_nom + (1.0 - stm_g) * attack_b_nom
        us_def_free = stm_g * attack_w_free + (1.0 - stm_g) * attack_b_free
        them_any = stm_g * black_any + (1.0 - stm_g) * white_any
        us_any = stm_g * white_any + (1.0 - stm_g) * black_any
        them_king = stm_g * black_king_sq + (1.0 - stm_g) * white_king_sq
        us_king = stm_g * white_king_sq + (1.0 - stm_g) * black_king_sq
        # Defender colour (in absolute terms).
        them_is_white = (1.0 - stm_g)
        us_is_white = stm_g

        us_side_vec, us_aux = self._side_vector(
            attack=us_attack,
            def_nom=them_def_nom,
            def_free=them_def_free,
            defender_king=them_king,
            defender_any=them_any,
            total_occupancy=occupancy,
            defender_is_white=them_is_white,
            ablation=ablation,
        )
        them_side_vec, them_aux = self._side_vector(
            attack=them_attack,
            def_nom=us_def_nom,
            def_free=us_def_free,
            defender_king=us_king,
            defender_any=us_any,
            total_occupancy=occupancy,
            defender_is_white=us_is_white,
            ablation=ablation,
        )

        if ablation == "no_asymmetry":
            them_side_vec = torch.zeros_like(them_side_vec)

        diff = us_side_vec - them_side_vec
        operator_vector = torch.cat(
            [us_side_vec, them_side_vec, diff, diff.abs()], dim=1
        )                                                              # (B, 32)
        return {
            "operator_vector": operator_vector,
            "us_side_vec": us_side_vec,
            "them_side_vec": them_side_vec,
            "us": us_aux,
            "them": them_aux,
            "pinned_white": pinned_white,
            "pinned_black": pinned_black,
            "attack_units": units.detach(),
        }


class KingZoneReplyPressure(nn.Module):
    """p051 -- King-Zone Reply Pressure head over the i193 trunk.

    Wraps the i193 ``ExchangeThenKingDualStreamNetwork`` trunk with an
    additive, sigmoid-gated head that consumes the 32-dim king-zone
    operator vector. The gate is initialised near-closed so the i193
    baseline is exactly recovered at start of training; the
    ``zero_delta`` / ``trunk_only`` ablations recover it numerically.
    """

    ALLOWED_ABLATIONS = ALLOWED_ABLATIONS

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        # Trunk hyper-parameters mirror the i193 builder.
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        # KZRP head hyper-parameters.
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "KingZoneReplyPressure supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "KingZoneReplyPressure requires the simple_18 board tensor"
            )
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}"
            )

        self.num_classes = 1
        self.ablation = str(ablation)
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
        self.builder = KingZoneReplyPressureBuilder()

        self.feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )
        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim + OPERATOR_OUTPUT_DIM),
            nn.Linear(self.feature_dim + OPERATOR_OUTPUT_DIM, int(head_hidden_dim)),
            nn.GELU(),
            head_dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )
        gate_in = self.feature_dim + 1  # joint + |operator| mean
        self.gate_head = nn.Sequential(
            nn.LayerNorm(gate_in),
            nn.Linear(gate_in, max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )
        with torch.no_grad():
            final = self.gate_head[-1]
            if isinstance(final, nn.Linear):
                final.bias.fill_(float(gate_init))

    def _piece_state_and_stm(
        self, board: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        piece_state = board[:, :NUM_PIECE_CHANNELS].clamp(0.0, 1.0).flatten(2)
        stm = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        return piece_state, stm

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        piece_state, stm = self._piece_state_and_stm(board)
        builder_out = self.builder(piece_state, stm, ablation=self.ablation)
        operator_vector = builder_out["operator_vector"]               # (B, 32)
        op_mag = operator_vector.abs().mean(dim=1, keepdim=True)        # (B, 1)

        delta_input = torch.cat([joint, operator_vector], dim=1)
        delta_raw = self.delta_head(delta_input).view(-1)

        gate_input = torch.cat([joint, op_mag], dim=1)
        gate_logit = self.gate_head(gate_input).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation == "disable_gate":
            gate = torch.ones_like(gate)

        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
            gate_applied = torch.zeros_like(gate)
        else:
            primitive_delta = delta_raw
            gate_applied = gate
        contribution = gate_applied * primitive_delta
        logits = base_logit + contribution

        eps = 1.0e-6
        gate_clamped = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(
            gate_clamped * gate_clamped.log()
            + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
        )

        us_aux = builder_out["us"]
        them_aux = builder_out["them"]

        out: dict[str, torch.Tensor] = {}
        for key, value in trunk_out.items():
            if key in {"logits", "proposal_profile_strength", "proposal_keyword_count"}:
                continue
            out[f"trunk_{key}"] = value
        out["logits"] = logits
        out["base_logit"] = base_logit
        out["primitive_delta"] = primitive_delta
        out["primitive_delta_raw"] = delta_raw
        out["primitive_gate"] = gate
        out["primitive_gate_applied"] = gate_applied
        out["primitive_gate_logit"] = gate_logit
        out["primitive_gate_entropy"] = gate_entropy
        out["primitive_contribution"] = contribution
        out["kzrp_operator_mean"] = operator_vector.mean(dim=1)
        out["kzrp_operator_max"] = operator_vector.amax(dim=1)
        out["kzrp_operator_l2"] = operator_vector.pow(2).mean(dim=1).sqrt()
        for prefix, aux in (("us", us_aux), ("them", them_aux)):
            for name, tensor in aux.items():
                out[f"kzrp_{prefix}_{name}"] = tensor
        out["kzrp_asym_score"] = (
            us_aux["zone_pressure"] - them_aux["zone_pressure"]
        )
        out["mechanism_energy"] = trunk_out["mechanism_energy"] + out[
            "kzrp_operator_l2"
        ].detach()
        out["proposal_profile_strength"] = primitive_delta.detach().abs() * gate_entropy
        out["proposal_keyword_count"] = logits.new_full(
            (batch,), float(OPERATOR_OUTPUT_DIM)
        )
        return out


def build_king_zone_reply_pressure_from_config(
    config: dict[str, Any],
) -> KingZoneReplyPressure:
    cfg = dict(config)
    return KingZoneReplyPressure(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(
            cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))
        ),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "ALLOWED_ABLATIONS",
    "DEFAULT_ATTACK_UNITS",
    "KingZoneReplyPressure",
    "KingZoneReplyPressureBuilder",
    "OPERATOR_OUTPUT_DIM",
    "SIDE_VECTOR_DIM",
    "build_king_zone_reply_pressure_from_config",
)
