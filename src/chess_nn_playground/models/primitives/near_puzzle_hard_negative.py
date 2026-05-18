"""Near-Puzzle Hard-Negative Veto primitive (p055).

Source: ``ideas/research/primitives/external_50_near_puzzle_hard_negative_primitive.md``.

The operator is a *rejection* primitive: it scores the gap between
**surface tactical temptation** and **verified surviving force** for the
side to move. It plugs in as an additive gated side head on top of the
i193 ``ExchangeThenKingDualStreamNetwork`` trunk and emits

    final_logit = base_logit + gate * (-veto)

so that high veto pressure lowers the puzzle logit. The veto head
consumes a compact board-only diagnostic vector ``z(x)`` whose entries
approximate the math thesis components:

    z = [FG*, FG(m*), Disc(m*), Conc, Gap12, Avail, RCI,
         dBal, KEP, DOA, Counter]

with the following minimal-and-honest realization:

- Candidates and defender replies are compiled as learned query-token
  attention pools (no ``python-chess`` move generation in the forward
  pass). The pool sees the i193 spatial features.
- "Surface" scores broadcast attention over raw piece-presence channels;
  "verified" scores broadcast attention over the trunk's spatial
  features, with a defender-mask penalty that lowers verified force on
  squares with strong opposing presence.
- ``ReplyMass`` is the soft-existential ``logsumexp`` over reply
  neutralization scores; ``SafeCount`` is the soft-counted number of
  replies above a learned threshold; their composition forms ``Avail``
  and ``FG``.
- ``Disc`` is the per-candidate gap between surface and verified scores.
- ``Conc`` is the normalized concentration of softmaxed verified scores
  and ``Gap12`` is the top-two gap.
- ``RCI`` is the (clipped) mutual information proxy between candidate
  and reply marginals, computed from the joint softmax tensor.
- ``KEP`` and ``DOA`` are bounded reductions of king-zone attack-defense
  differences derived from the spatial features and the white/black king
  channels.

This is a board-only veto head. CRTK metadata, source labels, fine
labels, verification flags, engine evaluations and principal variations
are **not** consumed.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from chess_nn_playground.models.primitives.codex_reply_primitives import (
    BoardTokenAttention,
)
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import (
    BoardTensorSpec,
    require_board_tensor,
)
from chess_nn_playground.models.primitives.trunk_features import trunk_joint_features


SQUARES = 64
NUM_PIECE_CHANNELS = 12
WHITE_KING_PLANE = 5
BLACK_KING_PLANE = 11
WHITE_PIECE_PLANES = (0, 1, 2, 3, 4, 5)
BLACK_PIECE_PLANES = (6, 7, 8, 9, 10, 11)

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "no_replies",          # primary falsifier — drop ReplyMass / Avail / RCI from z
    "no_legality_discount",  # primary falsifier — collapse Disc to zero
    "concentration_only",  # ablate everything except Conc/Gap12
    "shuffle_replies",     # in-batch permutation of reply tokens
    "no_overload",         # drop DOA from z
    "no_king_escape",      # drop KEP from z
    "zero_delta",
    "trunk_only",
    "disable_gate",
)


def _king_zone_mask(
    king_plane: torch.Tensor,
    radius: int = 2,
) -> torch.Tensor:
    """Soft king-zone mask: max-pool a king-presence plane to a (radius)-neighbourhood.

    Args:
        king_plane: ``(B, 8, 8)`` 0/1 king plane.
        radius: half-window for the max-pool (1 -> 3x3 neighbourhood).

    Returns:
        ``(B, 8, 8)`` 0/1 mask covering the king's zone.
    """
    if king_plane.ndim != 3:
        raise ValueError(f"Expected (B, 8, 8) king plane, got {tuple(king_plane.shape)}")
    kernel = 2 * int(radius) + 1
    pooled = F.max_pool2d(
        king_plane.unsqueeze(1),
        kernel_size=kernel,
        stride=1,
        padding=int(radius),
    ).squeeze(1)
    return pooled.clamp(0.0, 1.0)


class NearPuzzleHardNegativePrimitive(nn.Module):
    """p055 — Near-Puzzle Hard-Negative Veto head over the i193 trunk."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        # Trunk hyper-parameters.
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        # Veto head hyper-parameters.
        num_candidates: int = 24,
        num_replies: int = 24,
        token_dim: int = 32,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        reply_temperature: float = 1.0,
        candidate_temperature: float = 1.0,
        safe_threshold: float = 0.5,
        king_zone_radius: int = 2,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "NearPuzzleHardNegativePrimitive supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "NearPuzzleHardNegativePrimitive requires the simple_18 board tensor"
            )
        if int(num_candidates) < 2:
            raise ValueError("num_candidates must be >= 2 for non-trivial concentration")
        if int(num_replies) < 2:
            raise ValueError("num_replies must be >= 2 for non-trivial reply mass")
        if int(king_zone_radius) < 1 or int(king_zone_radius) > 3:
            raise ValueError("king_zone_radius must be in [1, 3]")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )
        self.num_classes = 1
        self.ablation = str(ablation)
        self.num_candidates = int(num_candidates)
        self.num_replies = int(num_replies)
        self.token_dim = int(token_dim)
        self.reply_temperature = float(reply_temperature)
        self.candidate_temperature = float(candidate_temperature)
        self.safe_threshold = float(safe_threshold)
        self.king_zone_radius = int(king_zone_radius)
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

        spatial_channels = 2 * self.trunk.channels

        self.candidate_pool = BoardTokenAttention(
            in_channels=spatial_channels,
            num_tokens=self.num_candidates,
            token_dim=self.token_dim,
            dropout=float(head_dropout),
        )
        self.reply_pool = BoardTokenAttention(
            in_channels=spatial_channels,
            num_tokens=self.num_replies,
            token_dim=self.token_dim,
            dropout=float(head_dropout),
        )

        # Surface and verified scoring heads operate on candidate tokens.
        self.surface_score = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, 1),
        )
        self.verified_score = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, 1),
        )

        # Reply neutralization is bilinear over (candidate, reply) embeddings.
        self.reply_neutralizer = nn.Bilinear(
            self.token_dim, self.token_dim, 1
        )

        # 11-d diagnostic vector ``z(x)``.
        self.z_dim = 11
        self.coeff_norm = nn.LayerNorm(self.z_dim)
        self.feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )
        dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        # Veto head: softplus of an MLP over LayerNorm(z); fused with trunk pool.
        self.veto_head = nn.Sequential(
            nn.LayerNorm(self.z_dim + self.feature_dim),
            nn.Linear(self.z_dim + self.feature_dim, int(head_hidden_dim)),
            nn.GELU(),
            dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )

        gate_in = self.feature_dim + 3  # joint + (veto_raw, fg_star, avail)
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

    def _spatial_features(self, board: torch.Tensor) -> torch.Tensor:
        """Rebuild the i193 spatial map (concat of ex_h and kg_h)."""
        feats = self.trunk.feature_builder(board)
        if self.trunk.ablation == "shared_stream_only":
            ex_input = board
            kg_input = board
        else:
            ex_input = torch.cat([board, feats.exchange], dim=1)
            kg_input = torch.cat([board, feats.king], dim=1)
        ex_h, _ = self.trunk.exchange_encoder(ex_input)
        if self.trunk.ablation == "shared_stream_only":
            kg_h = ex_h
        else:
            kg_h, _ = self.trunk.king_encoder(kg_input)
        return torch.cat([ex_h, kg_h], dim=1)

    def _king_pressure_features(
        self,
        board: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (king-escape pressure, defender-overload asymmetry).

        Returns two per-sample scalars in ``[0, 1]`` derived from the
        side-to-move plane and the piece-presence planes. The reductions
        are bounded so they are stable across positions with very
        different material.
        """
        batch = board.shape[0]
        stm = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)  # (B,) 1 if white to move
        white_king = board[:, WHITE_KING_PLANE].clamp(0.0, 1.0)
        black_king = board[:, BLACK_KING_PLANE].clamp(0.0, 1.0)

        # Defender king and zone for the side that is *not* to move.
        defender_king = stm.view(batch, 1, 1) * black_king + (1.0 - stm.view(batch, 1, 1)) * white_king
        attacker_king = stm.view(batch, 1, 1) * white_king + (1.0 - stm.view(batch, 1, 1)) * black_king
        defender_zone = _king_zone_mask(defender_king, radius=self.king_zone_radius)
        attacker_zone = _king_zone_mask(attacker_king, radius=self.king_zone_radius)

        white_planes = board[:, list(WHITE_PIECE_PLANES)].clamp(0.0, 1.0)
        black_planes = board[:, list(BLACK_PIECE_PLANES)].clamp(0.0, 1.0)
        white_pressure = white_planes.sum(dim=1)
        black_pressure = black_planes.sum(dim=1)

        attacker_pressure = stm.view(batch, 1, 1) * white_pressure + (1.0 - stm.view(batch, 1, 1)) * black_pressure
        defender_pressure = stm.view(batch, 1, 1) * black_pressure + (1.0 - stm.view(batch, 1, 1)) * white_pressure

        # KEP: relative king-zone pressure imbalance for the defender side.
        defender_zone_sum = defender_zone.sum(dim=(1, 2)).clamp(min=1.0)
        attacker_on_defender_zone = (defender_zone * attacker_pressure).sum(dim=(1, 2))
        defender_on_defender_zone = (defender_zone * defender_pressure).sum(dim=(1, 2))
        kep = ((attacker_on_defender_zone - defender_on_defender_zone) / defender_zone_sum).clamp(-1.0, 1.0)
        kep = (kep + 1.0) * 0.5  # rescale to [0, 1]

        # DOA: defender-overload asymmetry — defender side faces more
        # critical-target attackers than the attacker side does.
        attacker_zone_sum = attacker_zone.sum(dim=(1, 2)).clamp(min=1.0)
        defender_on_attacker_zone = (attacker_zone * defender_pressure).sum(dim=(1, 2))
        attacker_on_attacker_zone = (attacker_zone * attacker_pressure).sum(dim=(1, 2))
        ovl_attacker = (attacker_on_defender_zone - defender_on_defender_zone).clamp(min=0.0) / defender_zone_sum
        ovl_defender = (defender_on_attacker_zone - attacker_on_attacker_zone).clamp(min=0.0) / attacker_zone_sum
        doa = (ovl_attacker - ovl_defender).clamp(-1.0, 1.0)
        doa = (doa + 1.0) * 0.5  # rescale to [0, 1]

        return kep, doa

    def _surface_signal(self, board: torch.Tensor) -> torch.Tensor:
        """Raw attack-density signal for the candidate side (bounded).

        Returns a per-sample scalar in ``[0, 1]`` that is high when the
        attacker has many pieces but the defender has *few* defenders on
        the king zone — a classic "surface temptation" indicator.
        """
        batch = board.shape[0]
        stm = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        white_king = board[:, WHITE_KING_PLANE].clamp(0.0, 1.0)
        black_king = board[:, BLACK_KING_PLANE].clamp(0.0, 1.0)
        defender_king = stm.view(batch, 1, 1) * black_king + (1.0 - stm.view(batch, 1, 1)) * white_king
        zone = _king_zone_mask(defender_king, radius=self.king_zone_radius)
        zone_sum = zone.sum(dim=(1, 2)).clamp(min=1.0)

        white_planes = board[:, list(WHITE_PIECE_PLANES)].clamp(0.0, 1.0)
        black_planes = board[:, list(BLACK_PIECE_PLANES)].clamp(0.0, 1.0)
        attacker = stm.view(batch, 1, 1) * white_planes.sum(dim=1) + (1.0 - stm.view(batch, 1, 1)) * black_planes.sum(dim=1)
        defender = stm.view(batch, 1, 1) * black_planes.sum(dim=1) + (1.0 - stm.view(batch, 1, 1)) * white_planes.sum(dim=1)
        attack = (zone * attacker).sum(dim=(1, 2)) / zone_sum
        defend = (zone * defender).sum(dim=(1, 2)) / zone_sum
        surface = (attack - 0.5 * defend).clamp(-1.0, 1.0)
        return (surface + 1.0) * 0.5

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        device = board.device
        dtype = board.dtype

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        spatial = self._spatial_features(board)
        candidates = self.candidate_pool(spatial)
        replies = self.reply_pool(spatial)
        cand_tokens = candidates.tokens
        reply_tokens = replies.tokens

        if self.ablation == "shuffle_replies" and batch > 1:
            perm = torch.randperm(batch, device=device)
            reply_tokens = reply_tokens[perm]

        # Per-candidate surface and verified scores.
        u_surf = self.surface_score(cand_tokens).squeeze(-1)  # (B, num_candidates)
        u_ver = self.verified_score(cand_tokens).squeeze(-1)

        if self.ablation == "no_legality_discount":
            disc = torch.zeros_like(u_surf)
        else:
            disc = u_surf - u_ver

        # Candidate concentration over verified scores.
        u_ver_temp = u_ver / max(self.candidate_temperature, 1.0e-3)
        pi = F.softmax(u_ver_temp, dim=-1)
        eps = 1.0e-6
        max_entropy = float(torch.log(torch.tensor(float(self.num_candidates))).item())
        ent = -(pi.clamp(eps, 1.0) * pi.clamp(eps, 1.0).log()).sum(dim=-1)
        conc = (1.0 - ent / max_entropy).clamp(0.0, 1.0)

        # Top-two gap (bounded by softplus).
        top_vals, _ = u_ver.topk(2, dim=-1)
        gap12 = F.softplus(top_vals[:, 0] - top_vals[:, 1])

        # Bilinear per-candidate per-reply neutralization scores n(m, r).
        cand_b = cand_tokens.unsqueeze(2).expand(batch, self.num_candidates, self.num_replies, self.token_dim)
        rep_b = reply_tokens.unsqueeze(1).expand(batch, self.num_candidates, self.num_replies, self.token_dim)
        neutralization = self.reply_neutralizer(
            cand_b.reshape(-1, self.token_dim),
            rep_b.reshape(-1, self.token_dim),
        ).view(batch, self.num_candidates, self.num_replies)

        if self.ablation == "no_replies":
            reply_mass_per_cand = torch.zeros_like(u_ver)
            safe_count_per_cand = torch.zeros_like(u_ver)
            avail = torch.zeros(batch, device=device, dtype=dtype)
            rci = torch.zeros(batch, device=device, dtype=dtype)
        else:
            tau_r = max(self.reply_temperature, 1.0e-3)
            reply_mass_per_cand = tau_r * torch.logsumexp(neutralization / tau_r, dim=-1)
            # Soft safe-count using a sigmoid above the learned threshold.
            safe_indicators = torch.sigmoid((neutralization - self.safe_threshold) * 4.0)
            safe_count_per_cand = safe_indicators.sum(dim=-1)
            cand_star = u_ver.argmax(dim=-1)  # (B,)
            arange = torch.arange(batch, device=device)
            avail = torch.log1p(safe_count_per_cand[arange, cand_star])

            # Reply-channel information (RCI) — proxy MI between candidate
            # and reply marginals derived from the joint softmax tensor.
            joint_logits = neutralization / tau_r
            joint_softmax = F.softmax(joint_logits.view(batch, -1), dim=-1).view(
                batch, self.num_candidates, self.num_replies
            )
            cand_marginal = joint_softmax.sum(dim=-1).clamp(eps, 1.0)
            reply_marginal = joint_softmax.sum(dim=-2).clamp(eps, 1.0)
            joint_clamped = joint_softmax.clamp(eps, 1.0)
            outer = cand_marginal.unsqueeze(-1) * reply_marginal.unsqueeze(-2)
            mi = (joint_clamped * (joint_clamped.log() - outer.clamp(eps, 1.0).log())).sum(dim=(-1, -2))
            rci = mi.clamp(0.0, max_entropy) / max_entropy

        # Forcedness gap FG(m) = u_ver(m) - ReplyMass(m); FG* = max over m.
        fg = u_ver - reply_mass_per_cand
        fg_star, _ = fg.max(dim=-1)
        # Mean-aggregated FG at m* (we use the argmax over u_ver to match the spec).
        u_ver_argmax = u_ver.argmax(dim=-1)
        arange = torch.arange(batch, device=device)
        fg_at_mstar = fg[arange, u_ver_argmax]
        disc_at_mstar = disc[arange, u_ver_argmax]

        # Bounded board-only signals for the remaining z entries.
        kep, doa = self._king_pressure_features(board)
        if self.ablation == "no_king_escape":
            kep = torch.zeros_like(kep)
        if self.ablation == "no_overload":
            doa = torch.zeros_like(doa)
        surface_signal = self._surface_signal(board)
        d_bal = (surface_signal - 0.5) * 2.0  # in [-1, 1]
        counter = (1.0 - surface_signal).clamp(0.0, 1.0)

        if self.ablation == "concentration_only":
            disc_at_mstar = torch.zeros_like(disc_at_mstar)
            avail = torch.zeros_like(avail)
            rci = torch.zeros_like(rci)
            kep = torch.zeros_like(kep)
            doa = torch.zeros_like(doa)
            d_bal = torch.zeros_like(d_bal)
            counter = torch.zeros_like(counter)
            fg_star = torch.zeros_like(fg_star)
            fg_at_mstar = torch.zeros_like(fg_at_mstar)

        z = torch.stack(
            [
                fg_star,
                fg_at_mstar,
                disc_at_mstar,
                conc,
                gap12,
                avail,
                rci,
                d_bal,
                kep,
                doa,
                counter,
            ],
            dim=-1,
        )
        z_norm = self.coeff_norm(z)

        veto_input = torch.cat([z_norm, joint], dim=1)
        veto_raw = F.softplus(self.veto_head(veto_input).squeeze(-1))

        gate_input = torch.cat(
            [
                joint,
                veto_raw.unsqueeze(-1),
                fg_star.unsqueeze(-1),
                avail.unsqueeze(-1),
            ],
            dim=1,
        )
        gate_logit = self.gate_head(gate_input).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation == "disable_gate":
            gate = torch.ones_like(gate)

        # Primitive delta is *negative* veto: high veto pressure lowers logit.
        primitive_delta_raw = -veto_raw
        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
            gate_applied = torch.zeros_like(gate)
        else:
            primitive_delta = primitive_delta_raw
            gate_applied = gate
        contribution = gate_applied * primitive_delta
        logits = base_logit + contribution

        gate_clamped = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(
            gate_clamped * gate_clamped.log()
            + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
        )

        out: dict[str, torch.Tensor] = {}
        for key, value in trunk_out.items():
            if key in {"logits", "proposal_profile_strength", "proposal_keyword_count"}:
                continue
            out[f"trunk_{key}"] = value
        out["logits"] = logits
        out["base_logit"] = base_logit
        out["primitive_delta"] = primitive_delta
        out["primitive_delta_raw"] = primitive_delta_raw
        out["primitive_gate"] = gate
        out["primitive_gate_applied"] = gate_applied
        out["primitive_gate_logit"] = gate_logit
        out["primitive_gate_entropy"] = gate_entropy
        out["primitive_contribution"] = contribution
        out["nphn_veto_pressure"] = veto_raw
        out["nphn_forcedness_gap"] = fg_star
        out["nphn_forcedness_at_mstar"] = fg_at_mstar
        out["nphn_legality_discount"] = disc_at_mstar
        out["nphn_candidate_concentration"] = conc
        out["nphn_candidate_gap"] = gap12
        out["nphn_reply_availability"] = avail
        out["nphn_reply_channel_information"] = rci
        out["nphn_attack_defense_balance"] = d_bal
        out["nphn_king_escape_pressure"] = kep
        out["nphn_defender_overload_asymmetry"] = doa
        out["nphn_counterpressure"] = counter
        out["mechanism_energy"] = trunk_out["mechanism_energy"] + veto_raw.detach()
        out["proposal_profile_strength"] = primitive_delta.detach().abs() * gate_entropy
        out["proposal_keyword_count"] = logits.new_full(
            (batch,), float(self.num_candidates + self.num_replies)
        )
        return out


def build_near_puzzle_hard_negative_from_config(
    config: dict[str, Any],
) -> NearPuzzleHardNegativePrimitive:
    cfg = dict(config)
    return NearPuzzleHardNegativePrimitive(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        num_candidates=int(cfg.get("num_candidates", 24)),
        num_replies=int(cfg.get("num_replies", 24)),
        token_dim=int(cfg.get("token_dim", 32)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        reply_temperature=float(cfg.get("reply_temperature", 1.0)),
        candidate_temperature=float(cfg.get("candidate_temperature", 1.0)),
        safe_threshold=float(cfg.get("safe_threshold", 0.5)),
        king_zone_radius=int(cfg.get("king_zone_radius", 2)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "ALLOWED_ABLATIONS",
    "NearPuzzleHardNegativePrimitive",
    "build_near_puzzle_hard_negative_from_config",
)
