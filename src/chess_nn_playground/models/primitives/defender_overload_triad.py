"""Defender Overload Triad primitive (p050).

Source: ``ideas/research/primitives/external_45_defender_overload_triad_primitive.md``.
The spec's recommended Phase-1 integration ("after incidence, before
readout, on i018") is implemented here as a gated additive head on top
of the i193 ``ExchangeThenKingDualStreamNetwork`` trunk -- the repo's
current strong baseline -- instead of i018. The same gated-additive
plug-in pattern is shared with p049 (Pin / X-ray / Skewer).

Thesis. Overloading is *defender identity reuse*: a single defender
having more than one defensive obligation on critical targets. Plain
attack/defense counts do not detect this because they do not bind
attackers and defenders by identity. The square-centric closed form
from the spec is:

    Aσ(i, t) -- attacker-i attacks target-t (target is enemy-occupied)
    Dσ(d, t) -- defender-d defends target-t (target is enemy-occupied)
    cσ(t)    -- target criticality (small MLP over rule features)
    O(d, t)  = Dσ(d, t) · cσ(t)              defender obligation
    L(d)     = Σ_t O(d, t)                   total obligation per defender
    Ω_def(d) = m(d) · (L(d)^2 - Σ_t O(d, t)^2)
    X_tar(t) = c(t) · [ Dσ^T (m·L) - c · (Dσ^2)^T m ]

The key algebraic identity ``L_d^2 - Σ_t O_dt^2 = Σ_{t ≠ u} O_dt O_du``
means ``Ω_def[d]`` is exactly the weighted mass of *distinct* critical
targets simultaneously assigned to the same defender ``d``. That is the
overload signal, and it stays ``O(BN^2)`` -- never materialising the
explicit ``(B, N, N, N)`` attacker × target × defender triple.

Pins are first-class via a small cumsum-based pin detector: a defender
``d`` is flagged ``π = 1`` when its own king and an enemy slider that
fires along the same direction sit at the first and second occupants
of a ray from ``d``'s own king. The pin term then both *amplifies*
defender fragility (``m = 1 + μ·π``) and *discounts* defender value
in the target-criticality feature ``d_val``.

Per side σ ∈ {us, them} we pool the per-square overload tensors to a
5-feature side vector
``[mean(X_tar), max(X_tar), mean(Ω_def | occupied), pinned_share,
mean(c)]`` and concatenate
``F = [S_us, S_them, S_us - S_them, |S_us - S_them|]`` for a 20-dim
operator output. The 20-dim vector is appended to the i193 joint pool
and fed through the gated additive head pattern (LayerNorm + Linear +
GELU + Linear, gate initialised near-closed at -2.0 so the i193
baseline is exactly recovered at the start of training).

Inputs are exactly the ``simple_18`` ``(B, 18, 8, 8)`` current-board
tensor. CRTK metadata, source labels, verification flags, engine
evaluations and principal variations are *not* consumed. Geometry
buffers (``geom_attacks``, ``between``, ``ray_step_index``) are
rule-derived and carry zero parameters.
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
SIDE_VECTOR_DIM = 5
OPERATOR_OUTPUT_DIM = 4 * SIDE_VECTOR_DIM   # us, them, diff, |diff| = 20
NUM_TARGET_FEATURES = 8                      # a, d, p, a_val, d_val, m_att, m_def, v_tar

# Default piece values, in pawn-equivalents. King is clipped to queen-level
# for the overload computation because the i193 trunk already represents
# king danger explicitly via the king stream (line-to-zone, escape, check);
# letting the king carry a sentinel value would swamp ordinary occupied-
# target overload statistics. This matches the recommendation in the
# source spec.
DEFAULT_PIECE_VALUES: tuple[float, ...] = (1.0, 3.2, 3.3, 5.0, 9.0, 9.0)

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    # Primary falsifier: drop the cross-target load term L^2 - sum O^2.
    # If lift survives, the primitive is not actually measuring defender
    # reuse but only single-target under-defence.
    "no_cross_target_load",
    # Pin discount / amplification set to zero everywhere (π = 0). Tests
    # whether pinned defenders matter on top of plain attack/defense.
    "no_pins",
    # Set v_tar = 1 (and v_att, v_def to 1) on every occupied target.
    # Tests whether piece-value weighting is load-bearing.
    "no_target_value",
    # Drop the value-sum / cheapest-attacker / cheapest-defender features
    # from the target-criticality gate; only counts (a, d, p) and v_tar
    # remain. Tests whether SEE-light features add signal.
    "counts_only",
    # Bypass the primitive entirely -- recovers the i193 baseline.
    "zero_delta",
    "trunk_only",
    # Pin the gate at 1.0. Tests whether the gate is load-bearing.
    "disable_gate",
)


def _direction_family_masks() -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(dir_is_orth, dir_is_diag)`` of shape ``(8,)`` float32.

    Mirrors the direction order in :data:`ray_geometry.DIRECTIONS`.
    """
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
    geom_attacks: ``(6, 2, 64, 64)`` float, 1 iff (piece-type, colour,
        source) attacks ``target`` ignoring blockers. Sliders are gated
        at runtime by the ``between`` mask.
    between: ``(64, 64, 64)`` float, ``between[s, t, k]`` is 1 iff
        ``k`` lies strictly between ``s`` and ``t`` on an aligned line.

    This is the same geometry the i193 ``DualStreamFeatureBuilder``
    materialises internally; we duplicate it here so the primitive
    stays self-contained.
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


class DefenderOverloadBuilder(nn.Module):
    """Square-centric overload primitive over plain attack/defense masks.

    Forward signature::

        forward(piece_state_absolute, stm, ablation="none") -> dict

    where ``piece_state_absolute`` is ``(B, 12, 64)`` in the original
    simple_18 piece-plane order (P, N, B, R, Q, K, p, n, b, r, q, k)
    and ``stm`` is the side-to-move scalar per sample (``(B,)``, 1.0 if
    white-to-move, 0.0 otherwise).

    Outputs an ``operator_vector`` of shape ``(B, 20)`` plus a dict of
    diagnostics (per-side scalar means / maxes used by the report
    template). The 20-dim vector is the concatenation
    ``[S_us, S_them, S_us - S_them, |S_us - S_them|]`` where each
    ``S_σ`` is a 5-feature side vector defined in the module docstring.
    """

    def __init__(self) -> None:
        super().__init__()
        geom_attacks, between = _build_attack_and_between()
        # geom_attacks: (6, 2, 64, 64); between: (64, 64, 64).
        self.register_buffer("geom_attacks", geom_attacks, persistent=False)
        self.register_buffer("between", between, persistent=False)
        # Ray geometry (shared with p049): used by the pin detector.
        geom = RayGeometry.build()
        self.register_buffer("ray_step_index", geom.step_index, persistent=False)
        self.register_buffer("ray_step_mask", geom.step_mask, persistent=False)
        orth, diag = _direction_family_masks()
        self.register_buffer("dir_is_orth", orth, persistent=False)
        self.register_buffer("dir_is_diag", diag, persistent=False)
        # Learnable per-piece-type value field. softmax keeps it bounded.
        self.piece_value_logits = nn.Parameter(torch.tensor(DEFAULT_PIECE_VALUES))
        # Pin discount / amplification parameters (sigmoid-bounded).
        self.pin_discount_logit = nn.Parameter(torch.tensor(1.1))   # ~0.75
        self.pin_amplify_logit = nn.Parameter(torch.tensor(0.0))    # ~0.5
        # Target-criticality MLP -- 8 features -> 1 scalar.
        hidden = 16
        self.target_gate = nn.Sequential(
            nn.LayerNorm(NUM_TARGET_FEATURES),
            nn.Linear(NUM_TARGET_FEATURES, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Per-feature mask used by the ``counts_only`` ablation.
        feature_mask = torch.ones(NUM_TARGET_FEATURES, dtype=torch.float32)
        feature_mask[3] = 0.0   # a_val
        feature_mask[4] = 0.0   # d_val
        feature_mask[5] = 0.0   # m_att
        feature_mask[6] = 0.0   # m_def
        self.register_buffer("counts_only_mask", feature_mask, persistent=False)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _gather_scalar(self, scalar: torch.Tensor) -> torch.Tensor:
        """Gather a per-square scalar along all rays. ``(B, 8, 64, 7)``."""
        flat = self.ray_step_index.reshape(-1)
        gathered = scalar[:, flat].reshape(
            scalar.shape[0], NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN
        )
        mask = self.ray_step_mask.to(device=scalar.device, dtype=scalar.dtype).view(
            1, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN
        )
        return gathered * mask

    def _attack_masks(
        self,
        piece_state: torch.Tensor,
        occupancy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(attacks_white, attacks_black)`` of shape ``(B, 64, 64)``.

        Each entry is ``1`` iff the attacker at source ``i`` attacks the
        target ``t``. Slider rays are gated by the ``between`` clear-line
        mask (no blockers strictly between source and target).
        """
        device = piece_state.device
        dtype = piece_state.dtype
        batch = piece_state.shape[0]
        geom = self.geom_attacks.to(device=device, dtype=dtype)
        between = self.between.to(device=device, dtype=dtype)

        blocked = torch.einsum("stk,bk->bst", between, occupancy)
        clear = (blocked <= 0.5).to(dtype=dtype)
        ones = torch.ones_like(clear)

        attacks_per_color: list[torch.Tensor] = []
        for color in (WHITE, BLACK):
            attack_sum = piece_state.new_zeros(batch, SQUARES, SQUARES)
            for piece in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING):
                ch = piece if color == WHITE else NUM_PIECE_TYPES + piece
                src = piece_state[:, ch]                              # (B, 64)
                gate = clear if piece in (BISHOP, ROOK, QUEEN) else ones
                rel = src.unsqueeze(-1) * geom[piece, color].unsqueeze(0) * gate
                attack_sum = attack_sum + rel
            attacks_per_color.append(attack_sum)
        return attacks_per_color[0], attacks_per_color[1]

    def _pin_mask(
        self,
        piece_state: torch.Tensor,
        occupancy: torch.Tensor,
        own_king_sq: torch.Tensor,
        own_any: torch.Tensor,
        enemy_sliders_per_dir: torch.Tensor,
    ) -> torch.Tensor:
        """Return a per-square pinned-defender indicator ``π`` of shape ``(B, 64)``.

        ``own_king_sq``: ``(B, 64)`` one-hot king position.
        ``own_any``: ``(B, 64)`` ``1`` where own piece occupies a square.
        ``enemy_sliders_per_dir``: ``(B, 8, 64)`` 1 iff an enemy slider
        capable of firing along direction ``d`` sits at that square.

        A defender ``d`` is flagged iff a ray from its own king passes
        through ``d`` (first occupant on the ray) and the second occupant
        is an enemy slider that fires along that direction.
        """
        # Gather along rays *from* every source square -- we use the
        # king's one-hot as a selector at the end, so all 64 sources are
        # computed but only the king's row contributes.
        occ_seq = self._gather_scalar(occupancy)                      # (B, 8, 64, 7)
        own_any_seq = self._gather_scalar(own_any)                    # (B, 8, 64, 7)
        # Enemy slider per direction: we need the *target* square check,
        # so gather the per-direction slider field along the same rays.
        # enemy_sliders_per_dir is (B, 8, 64). For each direction d we
        # gather *only* the same-direction slider field along that ray.
        # That gives (B, 8, 64, 7) by treating each (d) slice as a
        # separate scalar gather.
        enemy_slider_seq = torch.zeros(
            piece_state.shape[0],
            NUM_DIRECTIONS,
            SQUARES,
            RAY_MAX_LEN,
            device=piece_state.device,
            dtype=piece_state.dtype,
        )
        for d in range(NUM_DIRECTIONS):
            scalar = enemy_sliders_per_dir[:, d]                       # (B, 64)
            enemy_slider_seq[:, d] = self._gather_scalar(scalar)[:, d]

        occ_bool = (occ_seq > 0.5).to(dtype=piece_state.dtype)
        cum_occ = occ_bool.cumsum(dim=-1)
        first_step = occ_bool * (cum_occ <= 1.0).to(dtype=piece_state.dtype)
        second_step = occ_bool * ((cum_occ >= 1.5) & (cum_occ <= 2.5)).to(
            dtype=piece_state.dtype
        )

        first_own = first_step * own_any_seq                          # (B, 8, 64, 7)
        second_enemy_slider = second_step * enemy_slider_seq

        # Per-direction, per-source-square: did we see (own first, enemy
        # slider second)? If yes, the square index of `first_own` is the
        # pinned defender square.
        has_second_slider = second_enemy_slider.sum(dim=-1)           # (B, 8, 64)
        # Mark step-positions of first_own that are paired with a valid
        # second-slider on the same (B, d, source).
        first_own_marked = first_own * has_second_slider.unsqueeze(-1)
        # Indices of the target squares are encoded in ray_step_index.
        # We project the per-step mass back onto a (B, 64) per-square
        # pinned indicator by scatter-add along the ray-target axis.
        flat_idx = self.ray_step_index.reshape(-1)                    # (8*64*7,)
        # Restrict to king sources by multiplying with the king one-hot.
        king_sel = own_king_sq.view(piece_state.shape[0], 1, SQUARES, 1)
        marks = (first_own_marked * king_sel).reshape(
            piece_state.shape[0], -1
        )                                                              # (B, 8*64*7)
        pinned = piece_state.new_zeros(piece_state.shape[0], SQUARES)
        pinned.scatter_add_(
            dim=1,
            index=flat_idx.view(1, -1).expand(piece_state.shape[0], -1),
            src=marks,
        )
        return pinned.clamp(0.0, 1.0)

    def _enemy_sliders_per_direction(self, enemy_piece_state: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 8, 64)`` 1 iff an enemy slider fires along ``d`` at sq."""
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

    def _masked_min(self, adj: torch.Tensor, src_vals: torch.Tensor) -> torch.Tensor:
        """``(B, N)`` cheapest source attacker / defender per target.

        Zero is returned for targets with no attackers / defenders.
        """
        inf = torch.full((), float("inf"), dtype=src_vals.dtype, device=src_vals.device)
        vals = src_vals.unsqueeze(-1).expand_as(adj)
        mins = vals.masked_fill(adj <= 0.5, inf).amin(dim=1)
        return torch.where(mins.isfinite(), mins, mins.new_zeros(mins.shape))

    # ------------------------------------------------------------------
    # Side stats
    # ------------------------------------------------------------------

    def _side_stats(
        self,
        attack: torch.Tensor,        # (B, 64, 64) attacker-i attacks target-t
        defense: torch.Tensor,       # (B, 64, 64) defender-d attacks target-t
        target_piece: torch.Tensor,  # (B, 64) 1 where target side has a piece
        pinned_def: torch.Tensor,    # (B, 64) pinned-defender indicator on def side
        v_att: torch.Tensor,         # (B, 64) attacker piece values per source
        v_def: torch.Tensor,         # (B, 64) defender piece values per source
        v_tar: torch.Tensor,         # (B, 64) target piece values per square
        defender_occ: torch.Tensor,  # (B, 64) 1 where defender side has a piece
        ablation: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        eps = 1.0e-6
        # Restrict attack and defense to enemy-occupied targets.
        attack = attack * target_piece.unsqueeze(1)
        defense = defense * target_piece.unsqueeze(1)
        if ablation == "no_target_value":
            v_tar = target_piece.to(dtype=v_tar.dtype)
            v_att = torch.ones_like(v_att)
            v_def = torch.ones_like(v_def)
        if ablation == "no_pins":
            pinned_def = torch.zeros_like(pinned_def)

        pin_discount = torch.sigmoid(self.pin_discount_logit.to(dtype=attack.dtype))
        pin_boost = torch.sigmoid(self.pin_amplify_logit.to(dtype=attack.dtype))

        a = attack.sum(dim=1)                                       # (B, 64)
        d = defense.sum(dim=1)
        p = (defense * pinned_def.unsqueeze(-1)).sum(dim=1)
        a_val = (attack * v_att.unsqueeze(-1)).sum(dim=1)
        eff_def_value = (1.0 - pin_discount * pinned_def) * v_def
        d_val = (defense * eff_def_value.unsqueeze(-1)).sum(dim=1)
        m_att = self._masked_min(attack, v_att)
        m_def = self._masked_min(defense * (1.0 - pinned_def).unsqueeze(-1), v_def)

        x = torch.stack([a, d, p, a_val, d_val, m_att, m_def, v_tar], dim=-1)  # (B, 64, 8)
        if ablation == "counts_only":
            x = x * self.counts_only_mask.view(1, 1, NUM_TARGET_FEATURES).to(
                device=x.device, dtype=x.dtype
            )
        gate_raw = self.target_gate(x).squeeze(-1)                  # (B, 64)
        c = torch.nn.functional.softplus(gate_raw) * target_piece
        c = c.clamp(min=0.0)

        # Defender obligation tensor O[b, d, t] = D[b, d, t] * c[b, t].
        O = defense * c.unsqueeze(1)
        L = O.sum(dim=2)                                            # (B, 64)
        m = 1.0 + pin_boost * pinned_def                            # (B, 64)

        if ablation == "no_cross_target_load":
            # Drop the cross-target overload mass: leave only the
            # single-target "under-defended" signal ΣO^2 (which only
            # measures isolated defender obligation).
            defender_burden = m * O.square().sum(dim=2)
            target_exposure = c * (
                torch.bmm(
                    (defense.square()).transpose(1, 2), m.unsqueeze(-1)
                ).squeeze(-1)
            )
        else:
            defender_burden = m * (L.square() - O.square().sum(dim=2))
            target_exposure = c * (
                torch.bmm(defense.transpose(1, 2), (m * L).unsqueeze(-1)).squeeze(-1)
                - c
                * torch.bmm(
                    (defense.square()).transpose(1, 2), m.unsqueeze(-1)
                ).squeeze(-1)
            )

        target_exposure = target_exposure.clamp(min=0.0)
        defender_burden = defender_burden.clamp(min=0.0)

        target_mass = c.sum(dim=1).clamp_min(eps)
        defender_count = defender_occ.sum(dim=1).clamp_min(1.0)
        target_count = target_piece.sum(dim=1).clamp_min(1.0)

        pinned_defense_share = (target_exposure * (p / d.clamp_min(1.0))).sum(dim=1)
        pinned_defense_share = pinned_defense_share / target_exposure.sum(dim=1).clamp_min(eps)

        mean_burden = (defender_burden * defender_occ).sum(dim=1) / defender_count

        side_vec = torch.stack(
            [
                target_exposure.sum(dim=1) / target_mass,             # mean exposure
                target_exposure.amax(dim=1),                          # peak exposure
                mean_burden,                                          # mean burden
                pinned_defense_share,                                 # pinned overload share
                target_mass / target_count,                           # mean criticality
            ],
            dim=1,
        )                                                              # (B, 5)
        side_vec = torch.nan_to_num(side_vec, nan=0.0, posinf=0.0, neginf=0.0)
        aux = {
            "criticality_mean": (c.sum(dim=1) / target_count).detach(),
            "criticality_max": c.amax(dim=1).detach(),
            "target_exposure_mean": (
                target_exposure.sum(dim=1) / target_count
            ).detach(),
            "target_exposure_max": target_exposure.amax(dim=1).detach(),
            "defender_burden_mean": mean_burden.detach(),
            "defender_burden_max": defender_burden.amax(dim=1).detach(),
            "pinned_defense_share": pinned_defense_share.detach(),
            "attack_count_mean": (a.sum(dim=1) / target_count).detach(),
            "defense_count_mean": (d.sum(dim=1) / target_count).detach(),
            "pinned_defense_count_mean": (
                p.sum(dim=1) / target_count
            ).detach(),
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
            raise ValueError(
                f"stm must be (B,); got {tuple(stm.shape)}"
            )
        dtype = piece_state_absolute.dtype
        batch = piece_state_absolute.shape[0]
        stm_b = stm.to(dtype=dtype).clamp(0.0, 1.0).view(batch, 1, 1)
        # Absolute piece planes (white = 0..5, black = 6..11). The attack
        # builder needs absolute orientation; the overload pooling then
        # re-orients per side.
        ps = piece_state_absolute.clamp(0.0, 1.0)
        occupancy = ps.sum(dim=1).clamp(0.0, 1.0)
        white_any = ps[:, :NUM_PIECE_TYPES].sum(dim=1).clamp(0.0, 1.0)
        black_any = ps[:, NUM_PIECE_TYPES:].sum(dim=1).clamp(0.0, 1.0)

        # Per-side value field (per square, summed over occupant piece
        # types). DEFAULT_PIECE_VALUES / sqrt(81) keeps it in a tame range.
        if ablation == "no_target_value":
            values = torch.ones(NUM_PIECE_TYPES, device=ps.device, dtype=dtype)
        else:
            values = self.piece_value_logits.to(device=ps.device, dtype=dtype).clamp(min=0.0)
        white_value = (ps[:, :NUM_PIECE_TYPES] * values.view(1, NUM_PIECE_TYPES, 1)).sum(dim=1)
        black_value = (ps[:, NUM_PIECE_TYPES:] * values.view(1, NUM_PIECE_TYPES, 1)).sum(dim=1)
        # King-square one-hot per side.
        white_king_sq = ps[:, KING].clamp(0.0, 1.0)
        black_king_sq = ps[:, NUM_PIECE_TYPES + KING].clamp(0.0, 1.0)

        # Attack masks (absolute).
        attacks_white, attacks_black = self._attack_masks(ps, occupancy)

        # Mover-perspective selection. For σ=us, attacker side is the
        # mover, target = enemy occupancy, defender side = enemy. For
        # σ=them, attacker side = enemy, target = mover occupancy,
        # defender side = mover.
        stm_g = stm_b.view(batch, 1)                                 # (B, 1)
        # us_attack[i, t] is the mover's attack mask.
        us_attack = stm_g.unsqueeze(-1) * attacks_white + (1.0 - stm_g.unsqueeze(-1)) * attacks_black
        them_attack = stm_g.unsqueeze(-1) * attacks_black + (1.0 - stm_g.unsqueeze(-1)) * attacks_white
        us_any = stm_g * white_any + (1.0 - stm_g) * black_any
        them_any = stm_g * black_any + (1.0 - stm_g) * white_any
        us_value = stm_g * white_value + (1.0 - stm_g) * black_value
        them_value = stm_g * black_value + (1.0 - stm_g) * white_value
        us_king_sq = stm_g * white_king_sq + (1.0 - stm_g) * black_king_sq
        them_king_sq = stm_g * black_king_sq + (1.0 - stm_g) * white_king_sq

        # Pinned-defender indicators per side. The "them" side defends
        # against the mover's attacks; pins on the them side are with
        # respect to the them king (their own king pinning their own
        # blockers against mover sliders).
        # Build us-perspective piece tensors (P, N, B, R, Q, K) for both
        # sides so the slider-direction helper works on either colour.
        us_planes = stm_b * ps[:, :NUM_PIECE_TYPES] + (1.0 - stm_b) * ps[:, NUM_PIECE_TYPES:]
        them_planes = stm_b * ps[:, NUM_PIECE_TYPES:] + (1.0 - stm_b) * ps[:, :NUM_PIECE_TYPES]
        us_sliders_per_dir = self._enemy_sliders_per_direction(us_planes)
        them_sliders_per_dir = self._enemy_sliders_per_direction(them_planes)
        # Pin on them defenders: rays from them-king through them-pieces
        # with us-slider as second occupant.
        pinned_them = self._pin_mask(
            ps,
            occupancy,
            own_king_sq=them_king_sq,
            own_any=them_any,
            enemy_sliders_per_dir=us_sliders_per_dir,
        )
        # Pin on us defenders: rays from us-king through us-pieces with
        # them-slider as second occupant.
        pinned_us = self._pin_mask(
            ps,
            occupancy,
            own_king_sq=us_king_sq,
            own_any=us_any,
            enemy_sliders_per_dir=them_sliders_per_dir,
        )

        # σ = us: mover attacks them; defender = them.
        us_side_vec, us_aux = self._side_stats(
            attack=us_attack,
            defense=them_attack,
            target_piece=them_any,
            pinned_def=pinned_them,
            v_att=us_value,
            v_def=them_value,
            v_tar=them_value,
            defender_occ=them_any,
            ablation=ablation,
        )
        # σ = them: enemy attacks us; defender = us.
        them_side_vec, them_aux = self._side_stats(
            attack=them_attack,
            defense=us_attack,
            target_piece=us_any,
            pinned_def=pinned_us,
            v_att=them_value,
            v_def=us_value,
            v_tar=us_value,
            defender_occ=us_any,
            ablation=ablation,
        )
        diff = us_side_vec - them_side_vec
        operator_vector = torch.cat(
            [us_side_vec, them_side_vec, diff, diff.abs()], dim=1
        )                                                              # (B, 20)
        return {
            "operator_vector": operator_vector,
            "us_side_vec": us_side_vec,
            "them_side_vec": them_side_vec,
            "us": us_aux,
            "them": them_aux,
            "pinned_us": pinned_us,
            "pinned_them": pinned_them,
        }


class DefenderOverloadTriad(nn.Module):
    """p050 -- Defender Overload Triad head over the i193 trunk.

    Wraps the i193 ``ExchangeThenKingDualStreamNetwork`` trunk with an
    additive, sigmoid-gated head that consumes the 20-dim defender-
    overload operator vector. The gate is initialised near-closed so
    the i193 baseline is exactly recovered at start of training; the
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
        # Defender-overload head hyper-parameters.
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "DefenderOverloadTriad supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "DefenderOverloadTriad requires the simple_18 board tensor"
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
        self.builder = DefenderOverloadBuilder()

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
        """Return ``(piece_state_absolute, stm)``.

        ``piece_state_absolute`` is ``(B, 12, 64)`` in absolute
        (white-first, black-second) order. ``stm`` is ``(B,)`` with 1.0
        for white-to-move, 0.0 otherwise.
        """
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
        operator_vector = builder_out["operator_vector"]               # (B, 20)
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
        out["overload_operator_mean"] = operator_vector.mean(dim=1)
        out["overload_operator_max"] = operator_vector.amax(dim=1)
        out["overload_operator_l2"] = operator_vector.pow(2).mean(dim=1).sqrt()
        out["overload_us_mean"] = us_aux["target_exposure_mean"]
        out["overload_us_peak"] = us_aux["target_exposure_max"]
        out["overload_them_mean"] = them_aux["target_exposure_mean"]
        out["overload_them_peak"] = them_aux["target_exposure_max"]
        out["overload_pinned_share_us"] = us_aux["pinned_defense_share"]
        out["overload_pinned_share_them"] = them_aux["pinned_defense_share"]
        out["overload_defender_burden_us"] = us_aux["defender_burden_mean"]
        out["overload_defender_burden_them"] = them_aux["defender_burden_mean"]
        out["overload_criticality_us"] = us_aux["criticality_mean"]
        out["overload_criticality_them"] = them_aux["criticality_mean"]
        out["mechanism_energy"] = trunk_out["mechanism_energy"] + out[
            "overload_operator_l2"
        ].detach()
        out["proposal_profile_strength"] = primitive_delta.detach().abs() * gate_entropy
        out["proposal_keyword_count"] = logits.new_full(
            (batch,), float(OPERATOR_OUTPUT_DIM)
        )
        return out


def build_defender_overload_triad_from_config(
    config: dict[str, Any],
) -> DefenderOverloadTriad:
    cfg = dict(config)
    return DefenderOverloadTriad(
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
    "DEFAULT_PIECE_VALUES",
    "DefenderOverloadBuilder",
    "DefenderOverloadTriad",
    "OPERATOR_OUTPUT_DIM",
    "SIDE_VECTOR_DIM",
    "build_defender_overload_triad_from_config",
)
