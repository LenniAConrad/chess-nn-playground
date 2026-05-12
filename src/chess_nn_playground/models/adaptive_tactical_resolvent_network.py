"""Adaptive Tactical Resolvent Network for idea i077.

Implements the markdown architecture from
``ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_2002_saturday_shanghai_adaptive_tactical_resolvent.md``.

For each board the model builds a chess-structured 64x64 linear operator

    A(X) = sum_g gate_g(X) * mask_g + U(X) V(X)^T

whose deterministic geometric pieces are ray, knight, pawn, king, and
defense adjacency masks. The operator is divided by an estimated
spectral norm (a few power iterations clamped at 1.0) so the resolvents

    R_k = (I - alpha_k * A_hat)^(-1),   alpha_k in {0.25, 0.5, 0.75}

are well defined. Six role seed vectors (``attack``, ``defense``,
``king_target``, ``material_target``, ``blocker``, ``tempo``) are read
from the per-square trunk features. For each alpha the model solves

    y_attack_k  = R_k @ s_attack
    y_defense_k = R_k @ s_defense
    y_target_k  = R_k^T @ s_target          (transposed solve)

and reads the packet's transfer / cancellation diagnostics:

  * ``attack_to_target_k  = <y_attack_k, s_target>``
  * ``defense_to_target_k = <y_defense_k, s_target>``
  * ``net_pressure_k = attack_to_target_k - defense_to_target_k``
  * ``resolvent_sensitivity_k = || R_k @ s_attack - R_k @ s_defense ||``
  * king-zone and material-target resolvent energies of the attacker,
    defender and (transposed) target propagated fields.

The puzzle logit is an MLP over ``[per-alpha transfer/cancellation
features, sensitivity, role/king/material energies, pooled board
context, operator gate weights, operator-norm proxy, low-rank
energy]``. The model is strictly board-only.

Ablations covered (``model.ablation``):
  ``none``, ``no_resolvent_direct_pool``, ``neumann_1_step``,
  ``single_alpha``, ``fixed_operator_no_gates``, ``no_low_rank_update``,
  ``random_geometry_operator``, ``attack_only_no_defense``,
  ``cnn_same_params`` (trainer-side baseline; the model only flips the
  ``ablation_cnn_same_params`` output flag).
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models._packet_bespoke_base import (
    BoardConvStem,
    BoardTensorSpec,
    format_logits,
    require_board_tensor,
    side_to_move_field,
)


VALID_ABLATIONS: frozenset[str] = frozenset(
    {
        "none",
        "no_resolvent_direct_pool",
        "neumann_1_step",
        "single_alpha",
        "fixed_operator_no_gates",
        "no_low_rank_update",
        "random_geometry_operator",
        "attack_only_no_defense",
        "cnn_same_params",
    }
)


DEFAULT_ROLES: tuple[str, ...] = (
    "attack",
    "defense",
    "king_target",
    "material_target",
    "blocker",
    "tempo",
)


DEFAULT_ALPHAS: tuple[float, ...] = (0.25, 0.5, 0.75)


GEOMETRY_NAMES: tuple[str, ...] = ("ray", "knight", "pawn", "king", "defense")


# ---------------------------------------------------------------------
# Deterministic chess-geometry masks (64x64).
# ---------------------------------------------------------------------


def _square_index(rank: int, file: int) -> int:
    return rank * 8 + file


def _knight_mask() -> torch.Tensor:
    mask = torch.zeros(64, 64)
    deltas = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
    for r in range(8):
        for f in range(8):
            for dr, df in deltas:
                nr, nf = r + dr, f + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    mask[_square_index(r, f), _square_index(nr, nf)] = 1.0
    return mask


def _king_mask() -> torch.Tensor:
    mask = torch.zeros(64, 64)
    deltas = [(dr, df) for dr in (-1, 0, 1) for df in (-1, 0, 1) if not (dr == 0 and df == 0)]
    for r in range(8):
        for f in range(8):
            for dr, df in deltas:
                nr, nf = r + dr, f + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    mask[_square_index(r, f), _square_index(nr, nf)] = 1.0
    return mask


def _pawn_mask() -> torch.Tensor:
    mask = torch.zeros(64, 64)
    for r in range(8):
        for f in range(8):
            for dr in (-1, 1):
                for df in (-1, 1):
                    nr, nf = r + dr, f + df
                    if 0 <= nr < 8 and 0 <= nf < 8:
                        mask[_square_index(r, f), _square_index(nr, nf)] = 1.0
    return mask


def _ray_mask() -> torch.Tensor:
    mask = torch.zeros(64, 64)
    for r in range(8):
        for f in range(8):
            i = _square_index(r, f)
            for dr, df in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nr, nf = r + dr, f + df
                while 0 <= nr < 8 and 0 <= nf < 8:
                    mask[i, _square_index(nr, nf)] = 1.0
                    nr += dr
                    nf += df
    return mask


def _defense_mask() -> torch.Tensor:
    mask = torch.zeros(64, 64)
    for r in range(8):
        for f in range(8):
            i = _square_index(r, f)
            for dr, df in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nf = r + dr, f + df
                while 0 <= nr < 8 and 0 <= nf < 8:
                    mask[i, _square_index(nr, nf)] = 1.0
                    nr += dr
                    nf += df
    return mask


def _stack_geometry_masks() -> torch.Tensor:
    masks = [_ray_mask(), _knight_mask(), _pawn_mask(), _king_mask(), _defense_mask()]
    out = torch.stack(masks, dim=0)
    row_sums = out.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return out / row_sums


# ---------------------------------------------------------------------
# Main module.
# ---------------------------------------------------------------------


class AdaptiveTacticalResolventNetwork(nn.Module):
    """Bespoke implementation of the Adaptive Tactical Resolvent Network.

    Forward output dict (board-only inputs):
      - ``logits``: ``(B,)`` puzzle logit for the BCE-with-logits trainer.
      - ``operator_norm``: spectral-norm proxy of ``A(X)``.
      - ``operator_gate_weights``: ``(B, num_geometry)`` gates over the
        deterministic chess-geometry masks.
      - ``operator_low_rank_energy``: Frobenius mass of the low-rank
        context update.
      - ``attack_to_target``: ``(B, num_alpha)`` ``<R_k s_attack, s_target>``.
      - ``defense_to_target``: ``(B, num_alpha)`` ``<R_k s_defense, s_target>``.
      - ``net_pressure``: ``(B, num_alpha)`` ``attack_to_target - defense_to_target``.
      - ``transfer_ratio``: ``(B, num_alpha)`` numerator over denominator.
      - ``resolvent_sensitivity``: ``(B, num_alpha)`` ``|| y_attack - y_defense ||``.
      - ``king_zone_resolvent_energy``: ``(B, num_alpha, 3)`` (attacker, defender, target) energy on the side-to-move king mask.
      - ``material_target_resolvent_energy``: ``(B, num_alpha, 3)`` energy on the opposing high-value mask.
      - ``alpha_values``: ``(num_alpha,)`` propagation scales actually used.
      - ``ablation_*``: per-batch indicator flags.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        hidden_dim: int = 96,
        operator_rank: int = 12,
        alpha_values: Sequence[float] = DEFAULT_ALPHAS,
        roles: Sequence[str] = DEFAULT_ROLES,
        spectral_norm_iters: int = 3,
        head_hidden: int | None = None,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "AdaptiveTacticalResolventNetwork implements the puzzle_binary single-logit contract only"
            )
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if operator_rank < 1:
            raise ValueError("operator_rank must be >= 1")
        if spectral_norm_iters < 1:
            raise ValueError("spectral_norm_iters must be >= 1")
        if ablation not in VALID_ABLATIONS:
            raise ValueError(
                f"Unknown ablation {ablation!r}; expected one of {sorted(VALID_ABLATIONS)}"
            )

        roles = tuple(roles)
        for required in ("attack", "defense", "king_target", "material_target"):
            if required not in roles:
                raise ValueError(
                    f"role list must contain {required!r}; received {roles!r}"
                )
        alpha_values = tuple(float(a) for a in alpha_values)
        if len(alpha_values) < 1:
            raise ValueError("alpha_values must contain at least one entry")
        for alpha in alpha_values:
            if not (0.0 < alpha < 1.0):
                raise ValueError(
                    f"alpha values must lie in (0, 1) for stability; received {alpha}"
                )

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.operator_rank = int(operator_rank)
        self.spectral_norm_iters = int(spectral_norm_iters)
        self.roles = roles
        self.alpha_values = alpha_values
        self.dropout = float(dropout)
        self.ablation = str(ablation)
        head_hidden_dim = int(head_hidden if head_hidden is not None else hidden_dim)

        self.role_index = {name: i for i, name in enumerate(self.roles)}

        # Board trunk over (B, C, 8, 8).
        self.stem = BoardConvStem(
            input_channels=input_channels,
            channels=self.channels,
            depth=int(depth),
            use_batchnorm=use_batchnorm,
        )

        # Fixed geometry masks (5, 64, 64), buffered so they ride GPU/CPU
        # moves automatically and never receive gradients.
        self.register_buffer("_geometry_masks", _stack_geometry_masks(), persistent=False)
        self.num_geometry = self._geometry_masks.shape[0]

        rng = torch.Generator().manual_seed(0xA73DA00B)
        random_masks = torch.rand(self.num_geometry, 64, 64, generator=rng)
        random_masks = random_masks / random_masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("_random_geometry_masks", random_masks, persistent=False)

        # Persistent power-iteration vector for spectral-norm estimation.
        self.register_buffer(
            "_spectral_probe",
            F.normalize(torch.randn(64, generator=rng), dim=0),
            persistent=False,
        )

        # Operator builder: gate logits and low-rank update.
        self.gate_head = nn.Sequential(
            nn.Linear(self.channels * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_geometry),
        )
        self.low_rank_left = nn.Linear(self.channels, self.operator_rank)
        self.low_rank_right = nn.Linear(self.channels, self.operator_rank)

        # Role seed builder: per-square (B, 64, num_roles) scalars.
        self.role_seed_head = nn.Linear(self.channels, len(self.roles))

        # Learnable scalar per alpha (initialised at the requested values
        # but free to drift toward the data-preferred propagation scale).
        # The packet explicitly authorises learned alpha_k.
        self._alpha_init = torch.tensor(alpha_values, dtype=torch.float32)
        self.alpha_logits = nn.Parameter(
            torch.special.logit(self._alpha_init.clamp(1.0e-4, 1.0 - 1.0e-4))
        )

        num_alpha = len(alpha_values)
        self.num_alpha = int(num_alpha)
        # Final head input width.
        per_alpha_features = (
            1  # attack_to_target
            + 1  # defense_to_target
            + 1  # net_pressure
            + 1  # transfer_ratio
            + 1  # resolvent sensitivity
            + 3  # king_zone (attacker, defender, target)
            + 3  # material_target (attacker, defender, target)
            + 1  # alpha value
        )
        head_in = (
            self.channels * 2  # pooled board context (mean + max)
            + per_alpha_features * self.num_alpha
            + self.num_geometry  # operator gate weights
            + 2  # operator_norm + low rank energy
        )
        self.head_norm = nn.LayerNorm(head_in)
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def alphas(self) -> torch.Tensor:
        # Sigmoid keeps each alpha in (0, 1) so I - alpha A_hat stays
        # invertible when A_hat has spectral radius <= 1.
        return torch.sigmoid(self.alpha_logits)

    def _board_features(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        return self.stem(x)

    @staticmethod
    def _flatten_squares(feats: torch.Tensor) -> torch.Tensor:
        return feats.flatten(2).transpose(1, 2)

    @staticmethod
    def _pool(feats: torch.Tensor) -> torch.Tensor:
        mean_pool = feats.mean(dim=(2, 3))
        max_pool = feats.amax(dim=(2, 3))
        return torch.cat([mean_pool, max_pool], dim=1)

    @staticmethod
    def _king_target_masks(
        x: torch.Tensor, input_channels: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_channels < 13:
            zeros = x.new_zeros(x.shape[0], 64)
            return zeros, zeros
        side = side_to_move_field(x, input_channels).squeeze(1)  # (B, 8, 8)
        white = x[:, 0:6].clamp(0.0, 1.0)
        black = x[:, 6:12].clamp(0.0, 1.0)
        us_king = side * white[:, 5] + (1.0 - side) * black[:, 5]
        them_high_value = (
            side * (black[:, 1] + black[:, 2] + black[:, 3] + black[:, 4])
            + (1.0 - side) * (white[:, 1] + white[:, 2] + white[:, 3] + white[:, 4])
        )
        return us_king.flatten(1), them_high_value.flatten(1)

    def _operator(
        self,
        squares: torch.Tensor,
        pooled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_logits = self.gate_head(pooled)
        if self.ablation == "fixed_operator_no_gates":
            gate_weights = torch.ones_like(gate_logits) / float(self.num_geometry)
        else:
            gate_weights = F.softplus(gate_logits)
            gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

        if self.ablation == "random_geometry_operator":
            masks = self._random_geometry_masks
        else:
            masks = self._geometry_masks
        A_geom = torch.einsum("bg,gij->bij", gate_weights, masks)

        if self.ablation in {"fixed_operator_no_gates", "no_low_rank_update"}:
            low_rank = squares.new_zeros(squares.shape[0], 64, 64)
            low_rank_energy = squares.new_zeros(squares.shape[0])
        else:
            U = self.low_rank_left(squares)  # (B, 64, r)
            V = self.low_rank_right(squares)  # (B, 64, r)
            low_rank = torch.bmm(U, V.transpose(1, 2))  # (B, 64, 64)
            low_rank_energy = low_rank.flatten(1).norm(dim=-1)

        A = A_geom + low_rank
        A_hat, operator_norm = self._spectral_normalize(A)
        return A_hat, gate_weights, low_rank_energy, operator_norm

    def _spectral_normalize(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # A few power iterations on each batch entry produce a
        # differentiable upper bound on the spectral norm. Wrapping in
        # ``torch.no_grad`` for the iterative refinement step keeps the
        # scale stop-gradient free of stride-version mismatches.
        batch = A.shape[0]
        v = self._spectral_probe.detach().to(dtype=A.dtype).expand(batch, 64).contiguous()
        with torch.no_grad():
            for _ in range(self.spectral_norm_iters):
                Av = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
                v_next = F.normalize(Av, dim=-1, eps=1.0e-8)
                AtAv = torch.bmm(A.transpose(1, 2), v_next.unsqueeze(-1)).squeeze(-1)
                v = F.normalize(AtAv, dim=-1, eps=1.0e-8)
        Av = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
        sigma = Av.norm(dim=-1).clamp_min(1.0e-8)
        scale = torch.clamp_min(sigma, 1.0).unsqueeze(-1).unsqueeze(-1)
        return A / scale, sigma

    def _role_seeds(self, squares: torch.Tensor) -> torch.Tensor:
        # squares: (B, 64, C) -> per-square role logits (B, 64, R)
        per_square = self.role_seed_head(squares)
        seeds = per_square.transpose(1, 2)  # (B, R, 64)
        norms = seeds.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return seeds / norms

    def _solve_resolvent(
        self,
        A_hat: torch.Tensor,
        rhs: torch.Tensor,
        alpha: torch.Tensor,
        transpose: bool = False,
    ) -> torch.Tensor:
        # rhs: (B, 64, K). Solve (I - alpha * A_hat^[T]) y = rhs.
        batch = A_hat.shape[0]
        eye = torch.eye(64, device=A_hat.device, dtype=A_hat.dtype).expand(batch, 64, 64)
        operator = A_hat.transpose(1, 2) if transpose else A_hat
        alpha_b = alpha.view(-1, 1, 1) if alpha.ndim == 1 else alpha
        system = eye - alpha_b * operator

        if self.ablation == "neumann_1_step":
            # First-order Neumann approximation (I - alpha A)^(-1) ≈ I + alpha A.
            return rhs + alpha_b * torch.bmm(operator, rhs)

        # Direct batched solve. 64x64 is cheap and the packet
        # explicitly authorises ``torch.linalg.solve`` for v1.
        try:
            return torch.linalg.solve(system, rhs)
        except RuntimeError:
            # Tiny diagonal jitter recovers when the encoder produces a
            # numerically singular A_hat at initialisation.
            jitter = 1.0e-5 * torch.eye(
                64, device=A_hat.device, dtype=A_hat.dtype
            ).expand(batch, 64, 64)
            return torch.linalg.solve(system + jitter, rhs)

    @staticmethod
    def _energy(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # field: (B, 64), mask: (B, 64) -> (B,) energy of |field|^2 weighted by mask.
        return ((field * field) * mask).sum(dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self._board_features(x)
        squares = self._flatten_squares(feats)  # (B, 64, C)
        pooled = self._pool(feats)  # (B, 2C)
        batch = squares.shape[0]

        A_hat, gate_weights, low_rank_energy, operator_norm = self._operator(squares, pooled)
        seeds = self._role_seeds(squares)  # (B, R, 64)

        s_attack = seeds[:, self.role_index["attack"], :]
        s_defense = seeds[:, self.role_index["defense"], :]
        s_target = seeds[:, self.role_index["material_target"], :]

        if self.ablation == "attack_only_no_defense":
            s_defense = torch.zeros_like(s_defense)

        king_mask, target_mask = self._king_target_masks(x, self.input_channels)

        alpha_full = self.alphas  # (num_alpha,)
        if self.ablation == "single_alpha":
            # Use the centre alpha (~0.5) as the only propagation scale.
            mid = alpha_full.shape[0] // 2
            active_alphas = alpha_full[mid:mid + 1]
            num_alpha_active = 1
        else:
            active_alphas = alpha_full
            num_alpha_active = self.num_alpha

        attack_to_target_list: list[torch.Tensor] = []
        defense_to_target_list: list[torch.Tensor] = []
        sensitivity_list: list[torch.Tensor] = []
        king_energy_pair_list: list[torch.Tensor] = []
        material_energy_pair_list: list[torch.Tensor] = []

        # Build seed RHS as (B, 64, 2): [s_attack, s_defense].
        attacker_defender_rhs = torch.stack([s_attack, s_defense], dim=-1)
        target_rhs = s_target.unsqueeze(-1)  # (B, 64, 1)

        for k in range(num_alpha_active):
            alpha_k = active_alphas[k].expand(batch)
            if self.ablation == "no_resolvent_direct_pool":
                # Replace inverse with identity so transfer reduces to the
                # raw seed inner products (no propagation at all).
                y_attack = s_attack
                y_defense = s_defense
                y_target = s_target
            else:
                y_pair = self._solve_resolvent(
                    A_hat, attacker_defender_rhs, alpha_k, transpose=False
                )
                y_attack = y_pair[..., 0]
                y_defense = y_pair[..., 1]
                y_target = self._solve_resolvent(
                    A_hat, target_rhs, alpha_k, transpose=True
                ).squeeze(-1)

            attack_to_target_list.append((y_attack * s_target).sum(dim=-1))
            defense_to_target_list.append((y_defense * s_target).sum(dim=-1))
            sensitivity_list.append((y_attack - y_defense).norm(dim=-1))
            king_energy_pair_list.append(
                torch.stack(
                    [
                        self._energy(y_attack, king_mask),
                        self._energy(y_defense, king_mask),
                        self._energy(y_target, king_mask),
                    ],
                    dim=-1,
                )
            )
            material_energy_pair_list.append(
                torch.stack(
                    [
                        self._energy(y_attack, target_mask),
                        self._energy(y_defense, target_mask),
                        self._energy(y_target, target_mask),
                    ],
                    dim=-1,
                )
            )

        attack_to_target = torch.stack(attack_to_target_list, dim=-1)  # (B, A)
        defense_to_target = torch.stack(defense_to_target_list, dim=-1)
        net_pressure = attack_to_target - defense_to_target
        transfer_ratio = attack_to_target / (
            attack_to_target.abs() + defense_to_target.abs() + 1.0e-6
        )
        sensitivity = torch.stack(sensitivity_list, dim=-1)
        king_energy_pairs = torch.stack(king_energy_pair_list, dim=1)  # (B, A, 2)
        material_energy_pairs = torch.stack(material_energy_pair_list, dim=1)
        active_alpha_row = active_alphas.unsqueeze(0).expand(batch, num_alpha_active)

        # The head consumes a fixed-width feature vector; pad to
        # ``self.num_alpha`` slots so the single-alpha ablation still
        # produces a tensor compatible with the trained head shape.
        def _pad_to_full(tensor: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
            if tensor.shape[1] == self.num_alpha:
                return tensor
            pad_shape = list(tensor.shape)
            pad_shape[1] = self.num_alpha - tensor.shape[1]
            pad = tensor.new_full(pad_shape, fill)
            return torch.cat([tensor, pad], dim=1)

        head_attack = _pad_to_full(attack_to_target)
        head_defense = _pad_to_full(defense_to_target)
        head_net = _pad_to_full(net_pressure)
        head_ratio = _pad_to_full(transfer_ratio)
        head_sensitivity = _pad_to_full(sensitivity)
        head_king = _pad_to_full(king_energy_pairs)
        head_material = _pad_to_full(material_energy_pairs)
        head_alpha_row = _pad_to_full(active_alpha_row)

        head_input = torch.cat(
            [
                pooled,
                head_attack,
                head_defense,
                head_net,
                head_ratio,
                head_sensitivity,
                head_king.flatten(1),
                head_material.flatten(1),
                head_alpha_row,
                gate_weights,
                operator_norm.unsqueeze(-1),
                low_rank_energy.unsqueeze(-1),
            ],
            dim=-1,
        )
        head_input = self.head_norm(head_input)
        puzzle_logit = self.head(head_input).squeeze(-1)

        ones = puzzle_logit.new_ones(batch)
        ablation_flag = lambda name: ones * (1.0 if self.ablation == name else 0.0)

        output: dict[str, torch.Tensor] = {
            "logits": format_logits(puzzle_logit.unsqueeze(-1), self.num_classes),
            "operator_norm": operator_norm,
            "operator_gate_weights": gate_weights,
            "operator_low_rank_energy": low_rank_energy,
            "attack_to_target": attack_to_target,
            "defense_to_target": defense_to_target,
            "net_pressure": net_pressure,
            "transfer_ratio": transfer_ratio,
            "resolvent_sensitivity": sensitivity,
            "king_zone_resolvent_energy": king_energy_pairs,
            "material_target_resolvent_energy": material_energy_pairs,
            "alpha_values": active_alphas.detach(),
            "ablation_no_resolvent_direct_pool": ablation_flag("no_resolvent_direct_pool"),
            "ablation_neumann_1_step": ablation_flag("neumann_1_step"),
            "ablation_single_alpha": ablation_flag("single_alpha"),
            "ablation_fixed_operator_no_gates": ablation_flag("fixed_operator_no_gates"),
            "ablation_no_low_rank_update": ablation_flag("no_low_rank_update"),
            "ablation_random_geometry_operator": ablation_flag("random_geometry_operator"),
            "ablation_attack_only_no_defense": ablation_flag("attack_only_no_defense"),
            "ablation_cnn_same_params": ablation_flag("cnn_same_params"),
        }
        return output


def build_adaptive_tactical_resolvent_network_from_config(
    config: dict[str, Any],
) -> AdaptiveTacticalResolventNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)

    roles = cfg.pop("roles", DEFAULT_ROLES)
    if isinstance(roles, str):
        roles = tuple(part.strip() for part in roles.split(",") if part.strip())
    else:
        roles = tuple(roles)

    alpha_values = cfg.pop("alpha_values", DEFAULT_ALPHAS)
    if isinstance(alpha_values, (int, float)):
        alpha_values = (float(alpha_values),)
    else:
        alpha_values = tuple(float(a) for a in alpha_values)

    return AdaptiveTacticalResolventNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        depth=int(cfg.pop("depth", 2)),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        operator_rank=int(cfg.pop("operator_rank", 12)),
        alpha_values=alpha_values,
        roles=roles,
        spectral_norm_iters=int(cfg.pop("spectral_norm_iters", 3)),
        head_hidden=cfg.pop("head_hidden", None),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        ablation=str(cfg.pop("ablation", "none")),
    )


__all__ = [
    "DEFAULT_ALPHAS",
    "DEFAULT_ROLES",
    "GEOMETRY_NAMES",
    "AdaptiveTacticalResolventNetwork",
    "VALID_ABLATIONS",
    "build_adaptive_tactical_resolvent_network_from_config",
]
