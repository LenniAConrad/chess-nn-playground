"""Pin / X-Ray / Overload Sheaf model for idea i252.

i252 keeps i018's side-to-move-oriented sheaf spine (adapter, square token
encoder, sheaf diffusion math) and runs on the i249-style fast diffusion path,
but replaces i018's 12-plane tactical relation graph with a 22-plane graph
that adds ten compact dependency planes (x-ray, skewer, discovered-attack
candidate, attacks-against-piece-with-pinned-defender, attacks-on-overloaded-
defender, each in us/them mirrors).

All higher-order tactical motifs are collapsed into pairwise planes or
per-square soft masses before diffusion, so the sheaf operator class is
identical to i018 except for the larger relation count. The readout uses
name-based indexing for every per-family diagnostic so future relation
additions do not silently corrupt downstream reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import require_board_tensor
from chess_nn_playground.models.trunk.oriented_tactical_sheaf import (
    BoardState,
    BoardStateAdapter,
    SquareTokenEncoder,
    TacticalIncidence,
    TacticalIncidenceBuilder,
    TriadDefectPool,
    _format_logits,
    _idx,
    _make_geometry_masks,
    _weighted_mean,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec


# The 12 base i018 relations followed by the 10 new dependency relations.
# Order matters because diagnostic readout uses `RELATION_INDEX` for name
# lookup and the diffusion block uses `RELATION_SIGNS_V2` in this order.
RELATION_NAMES_V2: tuple[str, ...] = (
    # --- i018 base 12 ---
    "us_attacks_them_piece",
    "them_attacks_us_piece",
    "us_defends_us_piece",
    "them_defends_them_piece",
    "us_attacks_empty_near_king",
    "them_attacks_empty_near_king",
    "bishop_ray_visible",
    "rook_ray_visible",
    "queen_ray_visible",
    "knight_attack",
    "pawn_attack_forward_oriented",
    "king_ray_pin_candidate",
    # --- new 10 dependency planes ---
    "us_xray_them_piece",
    "them_xray_us_piece",
    "us_skewer_them_piece",
    "them_skewer_us_piece",
    "us_discovered_attack_candidate",
    "them_discovered_attack_candidate",
    "us_attacks_them_piece_with_pinned_defender",
    "them_attacks_us_piece_with_pinned_defender",
    "us_attacks_them_overloaded_piece",
    "them_attacks_us_overloaded_piece",
)

RELATION_INDEX: dict[str, int] = {name: i for i, name in enumerate(RELATION_NAMES_V2)}

# Sheaf sign per relation. i018 keeps +1 only for same-side defense planes
# (us_defends_us_piece, them_defends_them_piece). All new dependency planes
# are attack-flavored, so they keep -1 like the rest of the attack family.
RELATION_SIGNS_V2: tuple[int, ...] = (
    -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,  # base 12 (matches i018)
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,        # new 10
)

# Named families used by diagnostic readout. Each value is a tuple of relation
# indices; the readout averages relation_density over the indices for that
# family. Keeping the lookup name-based makes adding/removing relations safe.
RELATION_FAMILIES: dict[str, tuple[int, ...]] = {
    "ray_energy": (RELATION_INDEX["bishop_ray_visible"], RELATION_INDEX["rook_ray_visible"], RELATION_INDEX["queen_ray_visible"]),
    "king_ring": (RELATION_INDEX["us_attacks_empty_near_king"], RELATION_INDEX["them_attacks_empty_near_king"]),
    "pin_pressure": (RELATION_INDEX["king_ray_pin_candidate"],),
    "xray_pressure": (RELATION_INDEX["us_xray_them_piece"], RELATION_INDEX["them_xray_us_piece"]),
    "skewer_pressure": (RELATION_INDEX["us_skewer_them_piece"], RELATION_INDEX["them_skewer_us_piece"]),
    "discovered_pressure": (RELATION_INDEX["us_discovered_attack_candidate"], RELATION_INDEX["them_discovered_attack_candidate"]),
    "pinned_defender_pressure": (
        RELATION_INDEX["us_attacks_them_piece_with_pinned_defender"],
        RELATION_INDEX["them_attacks_us_piece_with_pinned_defender"],
    ),
    "overload_pressure": (
        RELATION_INDEX["us_attacks_them_overloaded_piece"],
        RELATION_INDEX["them_attacks_us_overloaded_piece"],
    ),
}


# Heuristic piece values (mover-oriented piece_state layout indices 1..12 with
# king at 6 and 12). Index 0 is empty.
_PIECE_VALUES = (0.0, 1.0, 3.0, 3.0, 5.0, 9.0, 100.0, 1.0, 3.0, 3.0, 5.0, 9.0, 100.0)


def _make_single_screen_bank() -> dict[str, torch.Tensor]:
    """Precompute the bounded single-screen template bank `T_1`.

    For every ordered triple `(source s, screen c, rear r)` aligned on a rook
    or bishop ray with `c` strictly between `s` and `r`, store the template
    entry `(s, c, r, line, clear_without_screen)` where `clear_without_screen`
    is the set of squares strictly between `s` and `r` except `c`. The
    template fires when no occupied square sits in `clear_without_screen`.

    For an 8x8 board there are exactly 2576 such templates (1792 rook + 784
    bishop), matching the count given in the i252 research packet.
    """
    rank = torch.arange(64) // 8
    file = torch.arange(64) % 8
    sources: list[int] = []
    screens: list[int] = []
    rears: list[int] = []
    lines: list[int] = []
    clears: list[torch.Tensor] = []
    for s in range(64):
        sr, sf = int(rank[s]), int(file[s])
        for r in range(64):
            if s == r:
                continue
            rr, rf = int(rank[r]), int(file[r])
            dr = rr - sr
            df = rf - sf
            is_rook = (dr == 0) or (df == 0)
            is_bishop = abs(dr) == abs(df) and abs(dr) > 0
            if not (is_rook or is_bishop):
                continue
            step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
            step_f = 0 if df == 0 else (1 if df > 0 else -1)
            between_squares: list[int] = []
            cur_r, cur_f = sr + step_r, sf + step_f
            while (cur_r, cur_f) != (rr, rf):
                between_squares.append(_idx(cur_r, cur_f))
                cur_r += step_r
                cur_f += step_f
            if not between_squares:
                continue
            for c in between_squares:
                clear = torch.zeros(64, dtype=torch.float32)
                for q in between_squares:
                    if q != c:
                        clear[q] = 1.0
                sources.append(s)
                screens.append(c)
                rears.append(r)
                lines.append(0 if is_rook and not is_bishop else 1)
                clears.append(clear)
    return {
        "screen_source": torch.tensor(sources, dtype=torch.long),
        "screen_screen": torch.tensor(screens, dtype=torch.long),
        "screen_rear": torch.tensor(rears, dtype=torch.long),
        "screen_line": torch.tensor(lines, dtype=torch.long),
        "screen_clear": (
            torch.stack(clears, dim=0) if clears else torch.zeros(0, 64, dtype=torch.float32)
        ),
    }


def _piece_value_table() -> torch.Tensor:
    """Per-piece-state-index piece value (length 13), with king as 100."""
    return torch.tensor(_PIECE_VALUES, dtype=torch.float32)


@dataclass(frozen=True)
class TacticalIncidenceV2:
    """Incidence record returned by `TacticalIncidenceBuilderV2`.

    Same fields as i018's `TacticalIncidence` but `relation_masks` carries
    22 planes instead of 12, and `pin_us`/`pin_them` expose the side-specific
    absolute pin masks used by the new dependency planes.
    """

    relation_masks: torch.Tensor
    our_attack: torch.Tensor
    them_attack: torch.Tensor
    our_piece: torch.Tensor
    them_piece: torch.Tensor
    empty: torch.Tensor
    relation_density: torch.Tensor
    pin_mask: torch.Tensor
    pin_us: torch.Tensor
    pin_them: torch.Tensor


class TacticalIncidenceBuilderV2(TacticalIncidenceBuilder):
    """Extends i018's 12-plane builder with 10 typed dependency planes.

    Adds a precomputed single-screen template bank `T_1` (2576 templates on
    an 8x8 board) plus piece-value buffers needed for skewer ordering and
    overload weighting. The first 12 relation planes are bit-identical to
    i018's `TacticalIncidenceBuilder`; the next 10 planes are the dependency
    family from the i252 research packet.
    """

    def __init__(self) -> None:
        super().__init__()
        bank = _make_single_screen_bank()
        for name, value in bank.items():
            self.register_buffer(name, value, persistent=False)
        self.register_buffer("piece_value_table", _piece_value_table(), persistent=False)

    def _side_specific_pin(
        self,
        occupancy: torch.Tensor,
        our_piece: torch.Tensor,
        them_piece: torch.Tensor,
        piece_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(P_us, P_them)` side-specific absolute pin masks (B,64,64).

        `P_us[b, s, d] = 1` when our slider on `s` absolutely pins their
        piece on `d` against their king along the canonical line for that
        template; `P_them` is the mirror.
        """
        batch = occupancy.shape[0]
        if self.pin_king.numel() == 0:
            zero = occupancy.new_zeros(batch, 64, 64)
            return zero, zero
        clear = (1.0 - torch.matmul(occupancy, self.pin_clear.t())).clamp(0.0, 1.0)
        king_idx = self.pin_king
        blocker_idx = self.pin_blocker
        slider_idx = self.pin_slider
        line = self.pin_line
        our_king = piece_state[:, :, 6]
        them_king = piece_state[:, :, 12]
        our_rook_slider = piece_state[:, :, 4] + piece_state[:, :, 5]
        our_bishop_slider = piece_state[:, :, 3] + piece_state[:, :, 5]
        them_rook_slider = piece_state[:, :, 10] + piece_state[:, :, 11]
        them_bishop_slider = piece_state[:, :, 9] + piece_state[:, :, 11]
        our_slider = torch.where(
            line.view(1, -1) == 0,
            our_rook_slider[:, slider_idx],
            our_bishop_slider[:, slider_idx],
        )
        them_slider = torch.where(
            line.view(1, -1) == 0,
            them_rook_slider[:, slider_idx],
            them_bishop_slider[:, slider_idx],
        )
        # P_us[s, d]: our slider pins their blocker against their king.
        pin_us_weight = (
            them_king[:, king_idx] * them_piece[:, blocker_idx] * our_slider * clear
        ).clamp(0.0, 1.0)
        # P_them[s, d]: their slider pins our blocker against our king.
        pin_them_weight = (
            our_king[:, king_idx] * our_piece[:, blocker_idx] * them_slider * clear
        ).clamp(0.0, 1.0)
        edge_index = slider_idx * 64 + blocker_idx
        edge_index_b = edge_index.view(1, -1).expand(batch, -1)
        flat_us = occupancy.new_zeros(batch, 64 * 64)
        flat_them = occupancy.new_zeros(batch, 64 * 64)
        flat_us.scatter_add_(1, edge_index_b, pin_us_weight)
        flat_them.scatter_add_(1, edge_index_b, pin_them_weight)
        return (
            flat_us.view(batch, 64, 64).clamp(0.0, 1.0),
            flat_them.view(batch, 64, 64).clamp(0.0, 1.0),
        )

    def _screen_clear(self, occupancy: torch.Tensor) -> torch.Tensor:
        """Per-template clear mask `Gamma_m` of shape `(B, M)`.

        `Gamma_m = 1` iff every square strictly between source and rear, *except*
        the designated screen, is empty.
        """
        if self.screen_clear.numel() == 0:
            return occupancy.new_zeros(occupancy.shape[0], 0)
        blockers = torch.matmul(occupancy, self.screen_clear.t())
        return (1.0 - blockers).clamp(0.0, 1.0)

    @staticmethod
    def _scatter_pair(
        batch_size: int,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        values: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        flat = torch.zeros(batch_size, 64 * 64, device=device, dtype=dtype)
        pair_index = (src_idx * 64 + dst_idx).view(1, -1).expand(batch_size, -1)
        flat.scatter_add_(1, pair_index, values)
        return flat.view(batch_size, 64, 64).clamp(0.0, 1.0)

    def _dependency_planes(
        self,
        occupancy: torch.Tensor,
        piece_state: torch.Tensor,
        our_attack: torch.Tensor,
        them_attack: torch.Tensor,
        our_piece: torch.Tensor,
        them_piece: torch.Tensor,
        pin_us: torch.Tensor,
        pin_them: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the 10 new typed dependency planes.

        Returns a dict keyed by relation name with each value of shape
        `(B, 64, 64)`. Pin-conditioned defense and overload exposures are
        computed via per-target soft masses and then turned back into
        attacker-to-target pairwise planes.
        """
        batch = occupancy.shape[0]
        device = occupancy.device
        dtype = occupancy.dtype

        # Slider presence by line type, per side. S^us[s, rook] = our rook/queen
        # on square s; S^us[s, bishop] = our bishop/queen on square s.
        our_rook_slider = (piece_state[:, :, 4] + piece_state[:, :, 5]).clamp(0.0, 1.0)
        our_bishop_slider = (piece_state[:, :, 3] + piece_state[:, :, 5]).clamp(0.0, 1.0)
        them_rook_slider = (piece_state[:, :, 10] + piece_state[:, :, 11]).clamp(0.0, 1.0)
        them_bishop_slider = (piece_state[:, :, 9] + piece_state[:, :, 11]).clamp(0.0, 1.0)

        # Same as TacticalIncidenceBuilder's "us non-king piece" used to
        # constrain discovered-attack candidates.
        our_nonking = piece_state[:, :, 1:6].sum(dim=-1).clamp(0.0, 1.0)
        them_nonking = piece_state[:, :, 7:12].sum(dim=-1).clamp(0.0, 1.0)

        # Per-square pinned indicators: pin_us_self[c] = our piece on c is
        # absolutely pinned by their slider; mirror for them.
        pin_us_self = pin_them.sum(dim=1).clamp(0.0, 1.0)
        pin_them_self = pin_us.sum(dim=1).clamp(0.0, 1.0)

        # Single-screen bank evaluation. Template index `m` carries
        # (source s_m, screen c_m, rear r_m, line l_m, clear_without_screen).
        screen_source = self.screen_source.to(device=device)
        screen_screen = self.screen_screen.to(device=device)
        screen_rear = self.screen_rear.to(device=device)
        screen_line = self.screen_line.to(device=device)
        gamma = self._screen_clear(occupancy).to(dtype=dtype)  # (B, M)
        if gamma.numel() == 0:
            zeros = occupancy.new_zeros(batch, 64, 64)
            return {
                "us_xray_them_piece": zeros,
                "them_xray_us_piece": zeros,
                "us_skewer_them_piece": zeros,
                "them_skewer_us_piece": zeros,
                "us_discovered_attack_candidate": zeros,
                "them_discovered_attack_candidate": zeros,
                "us_attacks_them_piece_with_pinned_defender": zeros,
                "them_attacks_us_piece_with_pinned_defender": zeros,
                "us_attacks_them_overloaded_piece": zeros,
                "them_attacks_us_overloaded_piece": zeros,
            }

        # Per-template indicators read by template index.
        rook_template = (screen_line == 0).to(dtype=dtype)
        bishop_template = (screen_line == 1).to(dtype=dtype)

        # Per-template values for the source / screen / rear squares.
        us_slider_on_source = (
            our_rook_slider[:, screen_source] * rook_template
            + our_bishop_slider[:, screen_source] * bishop_template
        )
        them_slider_on_source = (
            them_rook_slider[:, screen_source] * rook_template
            + them_bishop_slider[:, screen_source] * bishop_template
        )
        them_piece_on_rear = them_piece[:, screen_rear]
        us_piece_on_rear = our_piece[:, screen_rear]
        occ_on_screen = occupancy[:, screen_screen]
        them_piece_on_screen = them_piece[:, screen_screen]
        us_piece_on_screen = our_piece[:, screen_screen]

        # Piece-value lookups, normalized to [0, 1] for skewer ordering.
        piece_values = self.piece_value_table.to(device=device, dtype=dtype)
        sq_value = piece_state @ piece_values  # (B, 64)
        # nu = min(mu, 9) / 9 -- collapses king to 9/9 = 1 so a defender of
        # the king-square cannot dwarf real targets in overload weighting.
        sq_nu = (sq_value.clamp_max(9.0) / 9.0).clamp(0.0, 1.0)
        value_screen = sq_value[:, screen_screen]
        value_rear = sq_value[:, screen_rear]
        value_gt = (value_screen > value_rear).to(dtype=dtype)

        # X-ray (slider -> rear) with one own-side blocker between.
        x_us_value = gamma * us_slider_on_source * occ_on_screen * them_piece_on_rear
        x_them_value = gamma * them_slider_on_source * occ_on_screen * us_piece_on_rear
        # Skewer (slider -> rear) when front enemy screen outranks rear enemy.
        # Mover convention: own piece attacks enemy through enemy screen.
        k_us_value = (
            gamma * us_slider_on_source * them_piece_on_screen * them_piece_on_rear * value_gt
        )
        k_them_value = (
            gamma * them_slider_on_source * us_piece_on_screen * us_piece_on_rear * value_gt
        )
        # Discovered attack candidate (screen -> rear) when screen is own non-king,
        # not absolutely pinned, and the rear is an enemy piece.
        screen_us_free = (
            our_nonking[:, screen_screen]
            * (1.0 - pin_us_self[:, screen_screen]).clamp(0.0, 1.0)
        )
        screen_them_free = (
            them_nonking[:, screen_screen]
            * (1.0 - pin_them_self[:, screen_screen]).clamp(0.0, 1.0)
        )
        d_us_value = gamma * us_slider_on_source * screen_us_free * them_piece_on_rear
        d_them_value = gamma * them_slider_on_source * screen_them_free * us_piece_on_rear

        # Scatter template-level values back into pairwise (s, r) planes for
        # x-ray and skewer, or (c, r) planes for discovered attack.
        xray_us = self._scatter_pair(batch, screen_source, screen_rear, x_us_value, device, dtype)
        xray_them = self._scatter_pair(batch, screen_source, screen_rear, x_them_value, device, dtype)
        skewer_us = self._scatter_pair(batch, screen_source, screen_rear, k_us_value, device, dtype)
        skewer_them = self._scatter_pair(batch, screen_source, screen_rear, k_them_value, device, dtype)
        discovered_us = self._scatter_pair(batch, screen_screen, screen_rear, d_us_value, device, dtype)
        discovered_them = self._scatter_pair(batch, screen_screen, screen_rear, d_them_value, device, dtype)

        # Pinned-defender exposure. Same-side non-king defense planes.
        our_defense = our_attack * our_piece.unsqueeze(1)
        them_defense = them_attack * them_piece.unsqueeze(1)
        our_nonking_defense = our_defense * our_nonking.unsqueeze(-1)
        them_nonking_defense = them_defense * them_nonking.unsqueeze(-1)

        pin_them_d = pin_us.sum(dim=1).clamp(0.0, 1.0)  # (B, 64) — them piece pinned by us
        pin_us_d = pin_them.sum(dim=1).clamp(0.0, 1.0)
        # rho^{us, pdef}_r = sum_d Def_them[d, r] * pin_them_d
        rho_us_pdef = (them_nonking_defense * pin_them_d.unsqueeze(-1)).sum(dim=1).clamp(0.0, 1.0)
        rho_them_pdef = (our_nonking_defense * pin_us_d.unsqueeze(-1)).sum(dim=1).clamp(0.0, 1.0)
        pdef_us = (
            our_attack * them_piece.unsqueeze(1) * rho_us_pdef.unsqueeze(1)
        ).clamp(0.0, 1.0)
        pdef_them = (
            them_attack * our_piece.unsqueeze(1) * rho_them_pdef.unsqueeze(1)
        ).clamp(0.0, 1.0)

        # Overload exposure. crit^{us}_r = nu_r * 1[any our attacker on r].
        any_us_attacker = (our_attack.sum(dim=1) > 0).to(dtype=dtype)
        any_them_attacker = (them_attack.sum(dim=1) > 0).to(dtype=dtype)
        crit_us = sq_nu * any_us_attacker
        crit_them = sq_nu * any_them_attacker
        g_them = them_nonking_defense * crit_us.unsqueeze(1)  # (B, defender, target)
        g_us = our_nonking_defense * crit_them.unsqueeze(1)
        # Second-largest assignment per defender along target axis.
        k = min(2, g_them.shape[-1])
        omega_them = g_them.topk(k=k, dim=-1).values[..., -1] if k == 2 else g_them.new_zeros(g_them.shape[:-1])
        omega_us = g_us.topk(k=k, dim=-1).values[..., -1] if k == 2 else g_us.new_zeros(g_us.shape[:-1])
        rho_us_ovl = (them_nonking_defense * omega_them.unsqueeze(-1)).sum(dim=1).clamp(0.0, 1.0)
        rho_them_ovl = (our_nonking_defense * omega_us.unsqueeze(-1)).sum(dim=1).clamp(0.0, 1.0)
        ovl_us = (
            our_attack * them_piece.unsqueeze(1) * rho_us_ovl.unsqueeze(1)
        ).clamp(0.0, 1.0)
        ovl_them = (
            them_attack * our_piece.unsqueeze(1) * rho_them_ovl.unsqueeze(1)
        ).clamp(0.0, 1.0)

        return {
            "us_xray_them_piece": xray_us,
            "them_xray_us_piece": xray_them,
            "us_skewer_them_piece": skewer_us,
            "them_skewer_us_piece": skewer_them,
            "us_discovered_attack_candidate": discovered_us,
            "them_discovered_attack_candidate": discovered_them,
            "us_attacks_them_piece_with_pinned_defender": pdef_us,
            "them_attacks_us_piece_with_pinned_defender": pdef_them,
            "us_attacks_them_overloaded_piece": ovl_us,
            "them_attacks_us_overloaded_piece": ovl_them,
        }

    def forward(self, piece_state: torch.Tensor, occupancy: torch.Tensor) -> TacticalIncidenceV2:
        empty = piece_state[:, :, 0].clamp(0.0, 1.0)
        our = piece_state[:, :, 1:7]
        them = piece_state[:, :, 7:13]
        our_piece = our.sum(dim=-1).clamp(0.0, 1.0)
        them_piece = them.sum(dim=-1).clamp(0.0, 1.0)
        visible_rook, visible_bishop = self._visible_rays(occupancy)

        our_attack = (
            our[:, :, 0].unsqueeze(-1) * self.our_pawn.unsqueeze(0)
            + our[:, :, 1].unsqueeze(-1) * self.knight.unsqueeze(0)
            + our[:, :, 2].unsqueeze(-1) * visible_bishop
            + our[:, :, 3].unsqueeze(-1) * visible_rook
            + our[:, :, 4].unsqueeze(-1) * (visible_rook + visible_bishop).clamp(0.0, 1.0)
            + our[:, :, 5].unsqueeze(-1) * self.king.unsqueeze(0)
        ).clamp(0.0, 1.0)
        them_attack = (
            them[:, :, 0].unsqueeze(-1) * self.their_pawn.unsqueeze(0)
            + them[:, :, 1].unsqueeze(-1) * self.knight.unsqueeze(0)
            + them[:, :, 2].unsqueeze(-1) * visible_bishop
            + them[:, :, 3].unsqueeze(-1) * visible_rook
            + them[:, :, 4].unsqueeze(-1) * (visible_rook + visible_bishop).clamp(0.0, 1.0)
            + them[:, :, 5].unsqueeze(-1) * self.king.unsqueeze(0)
        ).clamp(0.0, 1.0)

        near_their_king = torch.einsum("tk,bk->bt", self.king_zone, piece_state[:, :, 12]).clamp(0.0, 1.0)
        near_our_king = torch.einsum("tk,bk->bt", self.king_zone, piece_state[:, :, 6]).clamp(0.0, 1.0)
        bishop_slider = (piece_state[:, :, 3] + piece_state[:, :, 9]).clamp(0.0, 1.0)
        rook_slider = (piece_state[:, :, 4] + piece_state[:, :, 10]).clamp(0.0, 1.0)
        queen_slider = (piece_state[:, :, 5] + piece_state[:, :, 11]).clamp(0.0, 1.0)
        knight_piece = (piece_state[:, :, 2] + piece_state[:, :, 8]).clamp(0.0, 1.0)
        pin_mask = self._pin_relation(occupancy, our_piece, them_piece, piece_state)
        pin_us, pin_them = self._side_specific_pin(occupancy, our_piece, them_piece, piece_state)

        base_planes = [
            our_attack * them_piece.unsqueeze(1),
            them_attack * our_piece.unsqueeze(1),
            our_attack * our_piece.unsqueeze(1),
            them_attack * them_piece.unsqueeze(1),
            our_attack * empty.unsqueeze(1) * near_their_king.unsqueeze(1),
            them_attack * empty.unsqueeze(1) * near_our_king.unsqueeze(1),
            bishop_slider.unsqueeze(-1) * visible_bishop,
            rook_slider.unsqueeze(-1) * visible_rook,
            queen_slider.unsqueeze(-1) * (visible_rook + visible_bishop).clamp(0.0, 1.0),
            knight_piece.unsqueeze(-1) * self.knight.unsqueeze(0),
            (
                piece_state[:, :, 1].unsqueeze(-1) * self.our_pawn.unsqueeze(0)
                + piece_state[:, :, 7].unsqueeze(-1) * self.their_pawn.unsqueeze(0)
            ).clamp(0.0, 1.0),
            pin_mask,
        ]

        new_planes_dict = self._dependency_planes(
            occupancy=occupancy,
            piece_state=piece_state,
            our_attack=our_attack,
            them_attack=them_attack,
            our_piece=our_piece,
            them_piece=them_piece,
            pin_us=pin_us,
            pin_them=pin_them,
        )
        new_planes_ordered = [new_planes_dict[name] for name in RELATION_NAMES_V2[12:]]
        relation_masks = torch.stack(base_planes + new_planes_ordered, dim=1).clamp(0.0, 1.0)
        relation_density = relation_masks.mean(dim=(2, 3))
        return TacticalIncidenceV2(
            relation_masks=relation_masks,
            our_attack=our_attack,
            them_attack=them_attack,
            our_piece=our_piece,
            them_piece=them_piece,
            empty=empty,
            relation_density=relation_density,
            pin_mask=pin_mask,
            pin_us=pin_us,
            pin_them=pin_them,
        )


class FastSheafDiffusionBlockV2(nn.Module):
    """i249-style fast sheaf diffusion with a parameterized relation count.

    Algebraically equivalent to i018's `SheafDiffusionBlock` but the sign and
    relation count are now data, not literals. Required because the i252
    relation count (22) differs from the i018/i249 literal of 12.
    """

    def __init__(
        self,
        d_model: int,
        relation_count: int,
        stalk_dim: int,
        dropout: float,
        relation_signs: torch.Tensor,
    ) -> None:
        super().__init__()
        if relation_signs.numel() != relation_count:
            raise ValueError(
                "relation_signs length must equal relation_count "
                f"(got {int(relation_signs.numel())} vs {relation_count})"
            )
        self.relation_count = int(relation_count)
        self.stalk_dim = int(stalk_dim)
        self.node_to_stalk = nn.Linear(d_model, stalk_dim)
        self.stalk_to_node = nn.Linear(stalk_dim, d_model)
        eye = torch.eye(stalk_dim).unsqueeze(0).repeat(relation_count, 1, 1)
        self.rho_src = nn.Parameter(eye + 0.02 * torch.randn(relation_count, stalk_dim, stalk_dim))
        self.rho_dst = nn.Parameter(eye + 0.02 * torch.randn(relation_count, stalk_dim, stalk_dim))
        self.relation_gate_logits = nn.Parameter(torch.zeros(relation_count))
        self.eta_logit = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("relation_signs", relation_signs.to(dtype=torch.float32), persistent=False)
        self.node_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, h: torch.Tensor, relation_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.node_to_stalk(h)
        gates = 2.0 * torch.sigmoid(self.relation_gate_logits)
        eta = 0.25 * torch.sigmoid(self.eta_logit)

        src = torch.matmul(z.unsqueeze(1), self.rho_src.unsqueeze(0))
        dst = torch.matmul(z.unsqueeze(1), self.rho_dst.unsqueeze(0))

        out_degree = relation_masks.sum(dim=-1)
        in_degree = relation_masks.sum(dim=-2)
        w_dst = torch.matmul(relation_masks, dst)
        wt_src = torch.matmul(relation_masks.transpose(-1, -2), src)

        signs = self.relation_signs.to(dtype=z.dtype).view(1, self.relation_count, 1, 1)
        gates_view = gates.to(dtype=z.dtype).view(1, self.relation_count, 1, 1)

        source_pre = signs * w_dst - out_degree.unsqueeze(-1) * src
        target_pre = signs * wt_src - in_degree.unsqueeze(-1) * dst
        source_back = torch.matmul(source_pre, self.rho_src.transpose(-1, -2).unsqueeze(0))
        target_back = torch.matmul(target_pre, self.rho_dst.transpose(-1, -2).unsqueeze(0))
        z_update = (gates_view * (source_back + target_back)).sum(dim=1)

        degree = (gates.to(dtype=z.dtype).view(1, self.relation_count, 1) * (out_degree + in_degree)).sum(dim=1)
        z_update = eta.to(dtype=z.dtype) * z_update / degree.unsqueeze(-1).clamp_min(1.0)
        h = self.norm(h + self.stalk_to_node(z_update) + self.node_mlp(h))

        src_norm = src.square().sum(dim=-1)
        dst_norm = dst.square().sum(dim=-1)
        cross = (src * w_dst).sum(dim=-1)
        energy_numer = (
            (out_degree * src_norm).sum(dim=-1)
            + (in_degree * dst_norm).sum(dim=-1)
            - 2.0 * self.relation_signs.to(dtype=z.dtype).view(1, self.relation_count) * cross.sum(dim=-1)
        )
        denom = out_degree.sum(dim=-1).clamp_min(1.0)
        energies = gates.to(dtype=z.dtype).view(1, self.relation_count) * energy_numer / denom
        return h, energies, gates


class _TriadIncidenceAdapter:
    """Adapt `TacticalIncidenceV2` for `TriadDefectPool`.

    The triad pool only uses the four fields `our_attack`, `them_attack`,
    `our_piece`, `them_piece`, and the first four relation indices for
    its coverage stat. The first four planes in i252 are bit-identical to
    i018's first four planes, so a shallow adapter is correct.
    """

    def __init__(self, incidence: TacticalIncidenceV2) -> None:
        self.our_attack = incidence.our_attack
        self.them_attack = incidence.them_attack
        self.our_piece = incidence.our_piece
        self.them_piece = incidence.them_piece
        self.relation_masks = incidence.relation_masks


class PinXrayOverloadSheafNet(nn.Module):
    """i018-style sheaf classifier with i252's 22-plane dependency graph."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        stalk_dim: int = 8,
        dropout: float = 0.1,
        encoding: str = "simple_18",
        piece_adapter: str = "exact",
        use_triads: bool = True,
        scramble_relations: bool = False,
        scramble_new_only: bool = False,
        family_collapse: bool = False,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.relation_names = RELATION_NAMES_V2
        self.scramble_relations = bool(scramble_relations)
        self.scramble_new_only = bool(scramble_new_only)
        self.family_collapse = bool(family_collapse)
        self.adapter = BoardStateAdapter(
            input_channels=input_channels, encoding=encoding, piece_adapter=piece_adapter
        )
        self.incidence = TacticalIncidenceBuilderV2()
        self.encoder = SquareTokenEncoder(
            input_channels=input_channels, d_model=channels, dropout=dropout
        )
        signs = torch.tensor(RELATION_SIGNS_V2, dtype=torch.float32)
        self.blocks = nn.ModuleList(
            [
                FastSheafDiffusionBlockV2(
                    d_model=channels,
                    relation_count=len(RELATION_NAMES_V2),
                    stalk_dim=stalk_dim,
                    dropout=dropout,
                    relation_signs=signs,
                )
                for _ in range(max(1, int(depth)))
            ]
        )
        self.triad_pool = TriadDefectPool(channels, dropout) if use_triads else None
        triad_dim = self.triad_pool.output_dim if self.triad_pool is not None else 0
        board_stats_dim = 8
        readout_dim = channels * 4 + len(RELATION_NAMES_V2) * 4 + triad_dim + board_stats_dim
        self.head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def _board_stats(self, board: BoardState, incidence: TacticalIncidenceV2) -> torch.Tensor:
        occupancy = board.occupancy
        rank_counts = torch.matmul(occupancy, self.incidence.rank_one_hot)
        file_counts = torch.matmul(occupancy, self.incidence.file_one_hot)
        return torch.stack(
            [
                occupancy.mean(dim=1),
                incidence.our_piece.sum(dim=1) / 16.0,
                incidence.them_piece.sum(dim=1) / 16.0,
                incidence.our_attack.mean(dim=(1, 2)),
                incidence.them_attack.mean(dim=(1, 2)),
                incidence.pin_mask.mean(dim=(1, 2)),
                rank_counts.std(dim=1),
                file_counts.std(dim=1),
            ],
            dim=1,
        )

    def _maybe_scramble(self, relation_masks: torch.Tensor) -> torch.Tensor:
        if not (self.scramble_relations or self.scramble_new_only):
            return relation_masks
        batch, relations, squares, _ = relation_masks.shape
        if self.scramble_new_only:
            # Degree-preserving scramble on only the 10 new dependency planes.
            base_planes = relation_masks[:, :12]
            new_planes = relation_masks[:, 12:]
            new_count = new_planes.shape[1]
            perm = torch.argsort(
                torch.rand(batch, new_count, squares, device=relation_masks.device), dim=-1
            )
            perm_expanded = perm.unsqueeze(-2).expand(-1, -1, squares, -1)
            scrambled_new = torch.gather(new_planes, dim=-1, index=perm_expanded)
            return torch.cat([base_planes, scrambled_new], dim=1)
        perm = torch.argsort(
            torch.rand(batch, relations, squares, device=relation_masks.device), dim=-1
        )
        perm_expanded = perm.unsqueeze(-2).expand(-1, -1, squares, -1)
        return torch.gather(relation_masks, dim=-1, index=perm_expanded)

    def _maybe_family_collapse(self, relation_masks: torch.Tensor) -> torch.Tensor:
        """Replace the 10 new dependency planes by a single generic plane.

        Used as the F-collapse falsifier from the i252 packet: it preserves
        coverage by averaging the new planes into one, then duplicates the
        collapsed plane across the new slots so relation_count is unchanged.
        """
        if not self.family_collapse:
            return relation_masks
        base_planes = relation_masks[:, :12]
        new_planes = relation_masks[:, 12:]
        if new_planes.shape[1] == 0:
            return relation_masks
        collapsed = new_planes.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        collapsed_expanded = collapsed.expand(-1, new_planes.shape[1], -1, -1)
        return torch.cat([base_planes, collapsed_expanded], dim=1)

    def _family_density(
        self,
        relation_density: torch.Tensor,
        family: str,
    ) -> torch.Tensor:
        indices = RELATION_FAMILIES[family]
        if not indices:
            return relation_density.new_zeros(relation_density.shape[0])
        gathered = relation_density[:, list(indices)]
        return gathered.mean(dim=1)

    def _family_energy(
        self,
        energy_mean: torch.Tensor,
        family: str,
    ) -> torch.Tensor:
        indices = RELATION_FAMILIES[family]
        if not indices:
            return energy_mean.new_zeros(energy_mean.shape[0])
        gathered = energy_mean[:, list(indices)]
        return gathered.mean(dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.adapter(x)
        incidence = self.incidence(board.piece_state, board.occupancy)
        scrambled_masks = self._maybe_scramble(incidence.relation_masks)
        scrambled_masks = self._maybe_family_collapse(scrambled_masks)

        h = self.encoder(board.square_raw, board.piece_state)
        block_energies: list[torch.Tensor] = []
        block_gates: list[torch.Tensor] = []
        for block in self.blocks:
            h, energy, gates = block(h, scrambled_masks)
            block_energies.append(energy)
            block_gates.append(gates.unsqueeze(0).expand(x.shape[0], -1))

        energy_stack = torch.stack(block_energies, dim=1)
        gate_stack = torch.stack(block_gates, dim=1)
        energy_mean = energy_stack.mean(dim=1)
        energy_max = energy_stack.amax(dim=1)
        gate_mean = gate_stack.mean(dim=1)
        triad_stats = (
            self.triad_pool(h, _TriadIncidenceAdapter(incidence))
            if self.triad_pool is not None
            else h.new_zeros(h.shape[0], 0)
        )
        readout = torch.cat(
            [
                h.mean(dim=1),
                h.amax(dim=1),
                _weighted_mean(h, incidence.our_piece),
                _weighted_mean(h, incidence.them_piece),
                energy_mean,
                energy_max,
                incidence.relation_density,
                gate_mean,
                triad_stats,
                self._board_stats(board, incidence),
            ],
            dim=1,
        )
        logits = _format_logits(self.head(readout), self.num_classes)
        sheaf_tension = energy_stack.mean(dim=(1, 2))
        us_idx = RELATION_INDEX["us_attacks_them_piece"]
        them_idx = RELATION_INDEX["them_attacks_us_piece"]
        us_def_idx = RELATION_INDEX["us_defends_us_piece"]
        them_def_idx = RELATION_INDEX["them_defends_them_piece"]
        us_pressure = incidence.relation_masks[:, us_idx].sum(dim=(1, 2))
        them_pressure = incidence.relation_masks[:, them_idx].sum(dim=(1, 2))
        us_defense = incidence.relation_masks[:, us_def_idx].sum(dim=(1, 2))
        them_defense = incidence.relation_masks[:, them_def_idx].sum(dim=(1, 2))
        rank_counts = torch.matmul(board.occupancy, self.incidence.rank_one_hot)
        file_counts = torch.matmul(board.occupancy, self.incidence.file_one_hot)
        piece_entropy = -(
            board.piece_state * board.piece_state.clamp_min(1e-8).log()
        ).sum(dim=-1).mean(dim=1)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "mechanism_energy": torch.log1p(sheaf_tension),
            "proposal_profile_strength": gate_mean.mean(dim=1),
            "proposal_keyword_count": logits.new_full((x.shape[0],), 4.0),
            "sheaf_tension": sheaf_tension,
            "transport_imbalance": (us_pressure - them_pressure).abs() / (us_pressure + them_pressure).clamp_min(1.0),
            "symmetry_residual": (incidence.our_attack.mean(dim=(1, 2)) - incidence.them_attack.mean(dim=(1, 2))).abs(),
            "topology_pressure": incidence.relation_density.mean(dim=1),
            "ray_language_energy": self._family_energy(energy_mean, "ray_energy"),
            "information_surprisal": piece_entropy,
            "sparse_certificate_energy": energy_stack.amax(dim=(1, 2)),
            "rank_file_imbalance": (rank_counts.std(dim=1) - file_counts.std(dim=1)).abs(),
            "king_ring_pressure": self._family_density(incidence.relation_density, "king_ring") * 2.0,
            "reply_pressure": 0.5 * (us_pressure + them_pressure) / 64.0,
            "defense_gap": ((us_pressure + them_pressure) - (us_defense + them_defense)) / 64.0,
            "triad_defect_energy": triad_stats[:, 0] if triad_stats.numel() else logits.new_zeros(x.shape[0]),
            "pin_pressure": self._family_density(incidence.relation_density, "pin_pressure"),
            "xray_pressure": self._family_density(incidence.relation_density, "xray_pressure"),
            "skewer_pressure": self._family_density(incidence.relation_density, "skewer_pressure"),
            "discovered_pressure": self._family_density(incidence.relation_density, "discovered_pressure"),
            "pinned_defender_pressure": self._family_density(
                incidence.relation_density, "pinned_defender_pressure"
            ),
            "overload_pressure": self._family_density(incidence.relation_density, "overload_pressure"),
        }
        return diagnostics


def build_pin_xray_overload_sheaf_from_config(
    config: dict[str, Any],
) -> PinXrayOverloadSheafNet:
    return PinXrayOverloadSheafNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("sheaf_layers", config.get("depth", 2))),
        stalk_dim=int(config.get("stalk_dim", 8)),
        dropout=float(config.get("dropout", 0.1)),
        encoding=str(config.get("encoding", "simple_18")),
        piece_adapter=str(config.get("piece_adapter", "exact")),
        use_triads=bool(config.get("use_triads", True)),
        scramble_relations=bool(config.get("scramble_relations", False)),
        scramble_new_only=bool(config.get("scramble_new_only", False)),
        family_collapse=bool(config.get("family_collapse", False)),
    )
