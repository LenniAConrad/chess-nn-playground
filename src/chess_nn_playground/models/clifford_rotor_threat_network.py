"""Clifford Rotor Threat Network for idea i232.

Embeds each square in the geometric algebra ``Cl(3, 0)`` as an 8-dim
multivector (1 scalar, 3 vectors, 3 bivectors, 1 trivector). Threats are
composed via the non-commutative geometric product
``ab = a . b + a ^ b`` and rotors ``R = exp(B/2)`` act on neighbour
multivectors via the sandwich ``x -> R x R^{-1}``. Per-grade pooled
features over a fixed family of chess relations capture rotational
threat structure that real-arithmetic baselines cannot express.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


# Cl(3, 0) basis blades indexed by 3-bit bitmask:
#   bit 0 -> e_1, bit 1 -> e_2, bit 2 -> e_3
#   blade index 0 = scalar, 1 = e_1, 2 = e_2, 3 = e_1 e_2,
#                 4 = e_3, 5 = e_1 e_3, 6 = e_2 e_3, 7 = e_1 e_2 e_3
_BLADE_GRADES = (0, 1, 1, 2, 1, 2, 2, 3)
_BIVECTOR_INDICES = (3, 5, 6)
_TRIVECTOR_INDEX = 7


def _build_cl30_product_tensor() -> torch.Tensor:
    """Structure tensor ``M[i, j, k]`` for the Cl(3, 0) geometric product.

    ``e_i * e_j = M[i, j, k] * e_k`` summed over ``k`` (with M sparse).
    """
    tensor = torch.zeros(8, 8, 8)
    for a in range(8):
        for b in range(8):
            inversions = 0
            for ia in range(3):
                if not (a >> ia) & 1:
                    continue
                for jb in range(3):
                    if not (b >> jb) & 1:
                        continue
                    if ia > jb:
                        inversions += 1
            sign = -1.0 if (inversions & 1) else 1.0
            c = a ^ b
            tensor[a, b, c] = sign
    return tensor


def _build_blade_grade_masks() -> dict[int, torch.Tensor]:
    grades = torch.tensor(_BLADE_GRADES, dtype=torch.float32)
    masks: dict[int, torch.Tensor] = {}
    for k in (0, 1, 2, 3):
        masks[k] = (grades == k).float()
    return masks


def _build_reverse_signs() -> torch.Tensor:
    """Reverse anti-involution sign per blade.

    Grade-k blade reverse sign is ``(-1)^(k(k-1)/2)`` so for Cl(3, 0):
    grade 0 -> +1, grade 1 -> +1, grade 2 -> -1, grade 3 -> -1.
    """
    signs = []
    for grade in _BLADE_GRADES:
        exponent = (grade * (grade - 1)) // 2
        signs.append(-1.0 if exponent % 2 else 1.0)
    return torch.tensor(signs, dtype=torch.float32)


def _build_chess_relation_adjacencies() -> torch.Tensor:
    """Six fixed chess geometric relations as (6, 64, 64) row-stochastic-ish masks.

    0: king (8-neighbour ring)
    1: knight
    2: same rank
    3: same file
    4: a1-h8 diagonal
    5: a8-h1 anti-diagonal
    """
    masks = torch.zeros(6, 64, 64, dtype=torch.float32)
    for s in range(64):
        sr, sf = divmod(s, 8)
        for t in range(64):
            if s == t:
                continue
            tr, tf = divmod(t, 8)
            dr, df = tr - sr, tf - sf
            if max(abs(dr), abs(df)) == 1:
                masks[0, s, t] = 1.0
            if (abs(dr), abs(df)) in {(1, 2), (2, 1)}:
                masks[1, s, t] = 1.0
            if dr == 0:
                masks[2, s, t] = 1.0
            if df == 0:
                masks[3, s, t] = 1.0
            if dr == df:
                masks[4, s, t] = 1.0
            if dr == -df:
                masks[5, s, t] = 1.0
    # Row-normalize each relation so neighbour aggregation is bounded.
    row_sums = masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return masks / row_sums


class _Trunk(nn.Module):
    def __init__(self, input_channels: int, channels: int, depth: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(max(1, depth)):
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = channels
        self.trunk = nn.Sequential(*layers)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class CliffordRotorThreatNetwork(nn.Module):
    """Bespoke implementation of idea i232: rotor-equivariant Cl(3, 0) threat network."""

    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        multivector_dim: int = 8,
        num_relation_types: int = 6,
        rotor_taylor_terms: int = 4,
        bivector_clip: float = 1.0,
        cl_signature: tuple[int, int] | list[int] | None = (3, 0),
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("CliffordRotorThreatNetwork supports the puzzle_binary one-logit contract")
        if multivector_dim != 8:
            raise ValueError("Cl(3, 0) has exactly 8 basis blades; multivector_dim must be 8")
        if num_relation_types != 6:
            raise ValueError("CliffordRotorThreatNetwork uses 6 fixed chess relations")
        if cl_signature is not None and tuple(cl_signature) != (3, 0):
            raise ValueError("Only Cl(3, 0) is implemented")
        if rotor_taylor_terms < 2:
            raise ValueError("rotor_taylor_terms must be >= 2")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.channels = int(channels)
        self.rotor_taylor_terms = int(rotor_taylor_terms)
        self.bivector_clip = float(bivector_clip)
        self.num_relation_types = int(num_relation_types)

        self.trunk = _Trunk(input_channels, channels, depth, dropout, use_batchnorm)
        self.proj_phi = nn.Conv2d(channels, multivector_dim, kernel_size=1)

        struct = _build_cl30_product_tensor()
        reverse_signs = _build_reverse_signs()
        bivector_idx = torch.tensor(_BIVECTOR_INDICES, dtype=torch.long)
        grade_masks = _build_blade_grade_masks()
        relation_adj = _build_chess_relation_adjacencies()

        self.register_buffer("cl_struct", struct, persistent=False)
        self.register_buffer("reverse_signs", reverse_signs, persistent=False)
        self.register_buffer("bivector_idx", bivector_idx, persistent=False)
        self.register_buffer("grade0_mask", grade_masks[0], persistent=False)
        self.register_buffer("grade1_mask", grade_masks[1], persistent=False)
        self.register_buffer("grade2_mask", grade_masks[2], persistent=False)
        self.register_buffer("grade3_mask", grade_masks[3], persistent=False)
        self.register_buffer("relation_adj", relation_adj, persistent=False)

        # Learned per-relation gain on the geometric-product messages.
        self.relation_gain = nn.Parameter(torch.ones(num_relation_types))

        pooled_trunk_dim = 2 * channels
        # 4 grades * 2 stats (mean / max) per relation
        grade_feat_dim = num_relation_types * 4 * 2
        # 8 scalar diagnostics: bivector norm, chirality, sandwich residual, rotor norm, plus 4 per-grade phi energies.
        scalar_diag_dim = 8

        head_in = pooled_trunk_dim + grade_feat_dim + scalar_diag_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def _geom_prod(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ijk,...i,...j->...k", self.cl_struct, a, b)

    def _rotor_exp(self, bivec: torch.Tensor) -> torch.Tensor:
        """Compute R = exp(bivec / 2) via Taylor series, then normalise to ``|R| = 1``."""
        half = 0.5 * bivec
        rotor = torch.zeros_like(bivec)
        rotor[..., 0] = 1.0
        term = rotor.clone()
        for k in range(1, self.rotor_taylor_terms + 1):
            term = self._geom_prod(term, half) / float(k)
            rotor = rotor + term
        rotor_rev = rotor * self.reverse_signs
        norm_sq = self._geom_prod(rotor, rotor_rev)[..., 0].clamp_min(1.0e-6)
        return rotor / norm_sq.sqrt().unsqueeze(-1)

    @staticmethod
    def _grade_summary(messages: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        components = messages * mask
        per_square_norm = components.norm(dim=-1)  # (B, R, 64)
        return per_square_norm.mean(dim=-1), per_square_norm.amax(dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feat = self.trunk(x)  # (B, C, 8, 8)
        phi_map = self.proj_phi(feat)  # (B, 8, 8, 8)
        bsz = phi_map.shape[0]
        phi = phi_map.permute(0, 2, 3, 1).reshape(bsz, 64, 8)

        # Build clipped bivector field for the rotor exponential.
        bivec = torch.zeros_like(phi)
        bivec[..., self.bivector_idx] = phi[..., self.bivector_idx]
        bivec_norm = bivec.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        clip_scale = (self.bivector_clip / bivec_norm).clamp_max(1.0)
        bivec_clipped = bivec * clip_scale

        rotor = self._rotor_exp(bivec_clipped)  # (B, 64, 8)
        rotor_inv = rotor * self.reverse_signs  # |R|=1 implies R^{-1} = reverse(R)

        # Sandwich x -> R x R^{-1} per square.
        rphi = self._geom_prod(rotor, phi)
        phi_rot = self._geom_prod(rphi, rotor_inv)

        # Neighbour aggregation per chess relation.
        # adj: (R, 64_s, 64_t); phi_rot: (B, 64_t, 8) -> neighbour: (B, R, 64_s, 8)
        neighbour = torch.einsum("rst,btc->brsc", self.relation_adj, phi_rot)

        # Per-relation geometric-product message:
        # msg[b, r, s, k] = sum_{i, j} M[i, j, k] * phi_rot[b, s, i] * neighbour[b, r, s, j]
        messages = torch.einsum("ijk,bsi,brsj->brsk", self.cl_struct, phi_rot, neighbour)
        messages = messages * self.relation_gain.view(1, -1, 1, 1)

        g0_mean, g0_max = self._grade_summary(messages, self.grade0_mask)
        g1_mean, g1_max = self._grade_summary(messages, self.grade1_mask)
        g2_mean, g2_max = self._grade_summary(messages, self.grade2_mask)
        g3_mean, g3_max = self._grade_summary(messages, self.grade3_mask)

        grade_feats = torch.cat(
            [g0_mean, g0_max, g1_mean, g1_max, g2_mean, g2_max, g3_mean, g3_max],
            dim=-1,
        )

        # Scalar diagnostics.
        rotor_bivector_norm_mean = bivec.norm(dim=-1).mean(dim=-1)
        chirality_score = phi[..., _TRIVECTOR_INDEX].mean(dim=-1)
        sandwich_residual = (phi - phi_rot).reshape(bsz, -1).norm(dim=-1)
        rotor_norm_mean = rotor.norm(dim=-1).mean(dim=-1)
        phi_g0 = (phi * self.grade0_mask).norm(dim=-1).mean(dim=-1)
        phi_g1 = (phi * self.grade1_mask).norm(dim=-1).mean(dim=-1)
        phi_g2 = (phi * self.grade2_mask).norm(dim=-1).mean(dim=-1)
        phi_g3 = (phi * self.grade3_mask).norm(dim=-1).mean(dim=-1)

        scalar_diag = torch.stack(
            [
                rotor_bivector_norm_mean,
                chirality_score,
                sandwich_residual,
                rotor_norm_mean,
                phi_g0,
                phi_g1,
                phi_g2,
                phi_g3,
            ],
            dim=-1,
        )

        pooled_trunk = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)
        feat_vec = torch.cat([pooled_trunk, grade_feats, scalar_diag], dim=-1)
        logits = self.head(feat_vec).view(-1)

        return {
            "logits": logits,
            "clifford_rotor_bivector_norm_mean": rotor_bivector_norm_mean,
            "clifford_rotor_norm_mean": rotor_norm_mean,
            "clifford_chirality_score": chirality_score,
            "clifford_sandwich_residual": sandwich_residual,
            "clifford_grade0_phi_energy": phi_g0,
            "clifford_grade1_phi_energy": phi_g1,
            "clifford_grade2_phi_energy": phi_g2,
            "clifford_grade3_phi_energy": phi_g3,
            "clifford_grade0_message_mean": g0_mean,
            "clifford_grade1_message_mean": g1_mean,
            "clifford_grade2_message_mean": g2_mean,
            "clifford_grade3_message_mean": g3_mean,
            "clifford_grade2_message_max": g2_max,
        }


def build_clifford_rotor_threat_network_from_config(
    config: dict[str, Any],
) -> CliffordRotorThreatNetwork:
    cfg = dict(config)
    cl_signature = cfg.get("cl_signature", (3, 0))
    return CliffordRotorThreatNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        multivector_dim=int(cfg.get("multivector_dim", 8)),
        num_relation_types=int(cfg.get("num_relation_types", 6)),
        rotor_taylor_terms=int(cfg.get("rotor_taylor_terms", 4)),
        bivector_clip=float(cfg.get("bivector_clip", 1.0)),
        cl_signature=tuple(cl_signature) if cl_signature is not None else None,
        num_classes=int(cfg.get("num_classes", 1)),
    )
