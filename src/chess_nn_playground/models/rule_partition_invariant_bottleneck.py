"""Side-Canonical Rule-Partition Invariant Bottleneck (SCRIB) for idea i043.

Bespoke implementation of the markdown thesis: a board-only puzzle-likeness
classifier whose central operator is the rule-partition invariant
bottleneck

    B(x) = h(z),   z ~ q(z | C(x)),

where ``C`` is a deterministic side-to-move canonicalizer mapping the
``simple_18`` tensor to a 17-channel side-relative representation, ``q``
is a variational information bottleneck, and ``h`` is a small label head.
The model exposes deterministic rule partitions ``E_phase`` (3),
``E_adv`` (5) and ``E_color`` (2), a coarse 30-bin group id, and three
gradient-reversed adversary heads predicting those partitions from ``z``.

The architecture is materially distinct from the shared
``ResearchPacketProbe`` scaffold: there is no proposal-profile diagnostic
vector, no mechanism-family embedding, no profile hash; the trunk
operates on the canonicalized 17-channel tensor and the supervised head
reads only the variational sample ``z``. The auxiliary KL,
``E[CE(env, a(GRL(z)))]`` adversary losses, and the V-REx group-risk
variance are exposed in the forward output so the idea-specific trainer
can mix the SCRIB minimax objective without touching the model code.

Module names follow the markdown's Section 7 spec:

* ``Simple18SideCanonicalizer``
* ``Simple18RulePartitioner``
* ``GradientReversalFn`` / ``GradientReversalLayer``
* ``ConvTinyBackbone``
* ``VariationalBottleneck``
* ``EnvAdversaryHead``
* ``RulePartitionInvariantBottleneckNet``
* builder ``build_rule_partition_invariant_bottleneck``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from chess_nn_playground.models._packet_bespoke_base import (
    BoardTensorSpec,
    format_logits,
    require_board_tensor,
)


_EPS = 1.0e-6
_PIECE_VALUES_NON_KING = (1.0, 3.0, 3.0, 5.0, 9.0)  # P, N, B, R, Q


@dataclass(frozen=True)
class Simple18Spec:
    """Channel layout used by the SCRIB canonicalizer and partitioner."""

    encoding: str = "simple_18"
    input_channels: int = 18
    white_pawn_channel: int = 0
    white_piece_planes: tuple[int, int] = (0, 6)  # P, N, B, R, Q, K
    black_piece_planes: tuple[int, int] = (6, 12)
    side_to_move_channel: int = 12
    white_kingside_castling_channel: int = 13
    white_queenside_castling_channel: int = 14
    black_kingside_castling_channel: int = 15
    black_queenside_castling_channel: int = 16
    en_passant_channel: int = 17

    def validate(self, channels: int) -> None:
        if self.encoding != "simple_18":
            raise ValueError(
                "RulePartitionInvariantBottleneckNet only has a registered "
                f"channel map for simple_18; got encoding={self.encoding!r}"
            )
        if channels != self.input_channels:
            raise ValueError(
                "RulePartitionInvariantBottleneckNet expects 18 simple_18 "
                f"channels; got channels={channels}"
            )


class Simple18SideCanonicalizer(nn.Module):
    """Side-to-move canonicalizer mapping ``simple_18`` to ``(B, 17, 8, 8)``.

    For white-to-move samples the canonicalization is the identity on the
    piece, castling and en-passant planes; the absolute side-to-move plane
    is dropped. For black-to-move samples ranks are flipped vertically,
    white and black piece planes are swapped (so the moving side is always
    in the friendly slot), white and black castling planes are swapped
    while preserving the king/queen-side semantics, and the en-passant
    plane is rank-flipped.

    The output channel order is::

        [friendly P, N, B, R, Q, K,
         enemy   P, N, B, R, Q, K,
         friendly kingside, friendly queenside,
         enemy kingside, enemy queenside,
         canonical en-passant]
    """

    def __init__(
        self,
        encoding: str = "simple_18",
        input_channels: int = 18,
        fail_closed_unknown_channels: bool = True,
    ) -> None:
        super().__init__()
        self.spec = Simple18Spec(encoding=encoding, input_channels=input_channels)
        self.fail_closed_unknown_channels = bool(fail_closed_unknown_channels)
        self.output_channels = 17

    def _validate(self, x: torch.Tensor) -> None:
        try:
            self.spec.validate(x.shape[1])
        except ValueError:
            if self.fail_closed_unknown_channels:
                raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate(x)
        s = self.spec
        # White-to-move mask broadcast over (B, 1, 1, 1).
        stm = x[:, s.side_to_move_channel : s.side_to_move_channel + 1]
        white_to_move = stm.amax(dim=(2, 3), keepdim=True).clamp(0.0, 1.0)
        # Original planes.
        white_pieces = x[:, s.white_piece_planes[0] : s.white_piece_planes[1]]
        black_pieces = x[:, s.black_piece_planes[0] : s.black_piece_planes[1]]
        wk = x[:, s.white_kingside_castling_channel : s.white_kingside_castling_channel + 1]
        wq = x[:, s.white_queenside_castling_channel : s.white_queenside_castling_channel + 1]
        bk = x[:, s.black_kingside_castling_channel : s.black_kingside_castling_channel + 1]
        bq = x[:, s.black_queenside_castling_channel : s.black_queenside_castling_channel + 1]
        ep = x[:, s.en_passant_channel : s.en_passant_channel + 1]

        # Vertically flipped versions used when black is to move.
        white_pieces_flip = torch.flip(white_pieces, dims=[2])
        black_pieces_flip = torch.flip(black_pieces, dims=[2])
        ep_flip = torch.flip(ep, dims=[2])

        friendly = white_to_move * white_pieces + (1.0 - white_to_move) * black_pieces_flip
        enemy = white_to_move * black_pieces + (1.0 - white_to_move) * white_pieces_flip
        friendly_kingside = white_to_move * wk + (1.0 - white_to_move) * bk
        friendly_queenside = white_to_move * wq + (1.0 - white_to_move) * bq
        enemy_kingside = white_to_move * bk + (1.0 - white_to_move) * wk
        enemy_queenside = white_to_move * bq + (1.0 - white_to_move) * wq
        canonical_ep = white_to_move * ep + (1.0 - white_to_move) * ep_flip

        return torch.cat(
            [
                friendly,
                enemy,
                friendly_kingside,
                friendly_queenside,
                enemy_kingside,
                enemy_queenside,
                canonical_ep,
            ],
            dim=1,
        )


class Simple18RulePartitioner(nn.Module):
    """Deterministic rule partitions used for SCRIB invariance losses.

    Returns four integer tensors of shape ``(B,)``::

        phase_labels in {0,1,2}        coarse total non-king material bucket
        adv_labels   in {0,1,2,3,4}    side-relative material balance bucket
        color_labels in {0,1}          absolute color to move (pre-canonical)
        group_ids    in {0,...,29}     phase + 3*adv + 15*color

    Phase and material-balance use the standard piece values
    ``P=1, N=3, B=3, R=5, Q=9, K=0`` summed over the piece planes. Phase
    cuts at ``<=20`` (endgame), ``20-49`` (middlegame), ``>=50`` (high
    material). Side-relative advantage cuts at ``<=-5``, ``-4..-2``,
    ``-1..1``, ``2..4``, ``>=5`` material units for the moving side.

    Partitions are *not* concatenated to the model input; they are used
    only by adversary heads and the V-REx group-risk variance.
    """

    PHASE_BUCKETS: int = 3
    ADV_BUCKETS: int = 5
    COLOR_BUCKETS: int = 2
    NUM_GROUPS: int = 30

    def __init__(
        self,
        encoding: str = "simple_18",
        input_channels: int = 18,
        fail_closed_unknown_channels: bool = True,
    ) -> None:
        super().__init__()
        self.spec = Simple18Spec(encoding=encoding, input_channels=input_channels)
        self.fail_closed_unknown_channels = bool(fail_closed_unknown_channels)
        values = torch.tensor(_PIECE_VALUES_NON_KING, dtype=torch.float32)
        self.register_buffer("piece_values", values, persistent=False)

    def _validate(self, x: torch.Tensor) -> None:
        try:
            self.spec.validate(x.shape[1])
        except ValueError:
            if self.fail_closed_unknown_channels:
                raise

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate(x)
        s = self.spec
        values = self.piece_values.to(dtype=x.dtype, device=x.device)
        # Sum the five non-king piece planes over the board for each color.
        white_planes = x[:, s.white_piece_planes[0] : s.white_piece_planes[0] + 5]
        black_planes = x[:, s.black_piece_planes[0] : s.black_piece_planes[0] + 5]
        white_counts = white_planes.sum(dim=(2, 3))  # (B, 5)
        black_counts = black_planes.sum(dim=(2, 3))  # (B, 5)
        white_material = (white_counts * values).sum(dim=1)
        black_material = (black_counts * values).sum(dim=1)
        total_material = white_material + black_material

        # Phase: endgame, middlegame, high-material.
        phase = torch.zeros_like(total_material, dtype=torch.long)
        phase = torch.where(total_material > 20.0, torch.ones_like(phase), phase)
        phase = torch.where(total_material >= 50.0, torch.full_like(phase, 2), phase)

        # Side-to-move (absolute color before canonicalization). The
        # ``simple_18`` side-to-move plane is broadcast over the board, so
        # taking the spatial maximum is exact and avoids depending on the
        # specific replication strategy.
        stm = x[:, s.side_to_move_channel : s.side_to_move_channel + 1]
        white_to_move = stm.amax(dim=(2, 3)).view(-1).clamp(0.0, 1.0)
        color_labels = (1 - white_to_move.long())  # 0=white-to-move, 1=black-to-move

        # Side-relative material advantage in pawn units.
        signed_material = white_material - black_material
        # White-to-move keeps the signed advantage; black-to-move flips it.
        side_relative = signed_material * (2.0 * white_to_move - 1.0)
        adv = torch.full_like(phase, 2)  # default: equal
        adv = torch.where(side_relative <= -5.0, torch.zeros_like(adv), adv)
        adv = torch.where(
            (side_relative > -5.0) & (side_relative <= -2.0),
            torch.ones_like(adv),
            adv,
        )
        adv = torch.where(
            (side_relative >= 2.0) & (side_relative < 5.0),
            torch.full_like(adv, 3),
            adv,
        )
        adv = torch.where(side_relative >= 5.0, torch.full_like(adv, 4), adv)

        group_ids = phase + self.PHASE_BUCKETS * adv + (self.PHASE_BUCKETS * self.ADV_BUCKETS) * color_labels
        return {
            "phase_labels": phase,
            "adv_labels": adv,
            "color_labels": color_labels,
            "group_ids": group_ids,
            "total_material": total_material,
            "side_relative_advantage": side_relative,
        }


class GradientReversalFn(torch.autograd.Function):
    """Identity forward, sign-flipped scaled backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFn.apply(x, self.lambda_)


class _ResidualMicroBlock(nn.Module):
    def __init__(self, channels: int, dropout: float, use_norm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_norm),
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_norm))
        if use_norm:
            layers.append(nn.BatchNorm2d(channels))
        self.block = nn.Sequential(*layers)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ConvTinyBackbone(nn.Module):
    """Compact two-stage convolutional trunk for the canonicalized board.

    Architecture matches the markdown spec at default widths
    ``(64, 96)``::

        Conv(17 -> 64) + norm/GELU
        2 x residual block @ 64
        Conv(64 -> 96) + norm/GELU
        2 x residual block @ 96
        concat[GAP, GMP] -> Linear(192 -> 256) + GELU + Dropout

    Output shape: ``(B, 256)``.
    """

    def __init__(
        self,
        input_channels: int = 17,
        widths: tuple[int, int] = (64, 96),
        blocks_per_stage: int = 2,
        feature_dim: int = 256,
        dropout: float = 0.1,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be >= 1")
        if len(widths) != 2:
            raise ValueError("ConvTinyBackbone expects exactly two stage widths")
        stage1, stage2 = int(widths[0]), int(widths[1])
        stage1_layers: list[nn.Module] = [
            nn.Conv2d(input_channels, stage1, kernel_size=3, padding=1, bias=not use_norm),
        ]
        if use_norm:
            stage1_layers.append(nn.BatchNorm2d(stage1))
        stage1_layers.append(nn.GELU())
        self.stem1 = nn.Sequential(*stage1_layers)
        self.stage1_blocks = nn.Sequential(
            *(_ResidualMicroBlock(stage1, dropout=dropout, use_norm=use_norm) for _ in range(blocks_per_stage))
        )
        stage2_layers: list[nn.Module] = [
            nn.Conv2d(stage1, stage2, kernel_size=3, padding=1, bias=not use_norm),
        ]
        if use_norm:
            stage2_layers.append(nn.BatchNorm2d(stage2))
        stage2_layers.append(nn.GELU())
        self.stem2 = nn.Sequential(*stage2_layers)
        self.stage2_blocks = nn.Sequential(
            *(_ResidualMicroBlock(stage2, dropout=dropout, use_norm=use_norm) for _ in range(blocks_per_stage))
        )
        feature_in = 2 * stage2
        self.feature_dim = int(feature_dim)
        head_layers: list[nn.Module] = [
            nn.Linear(feature_in, self.feature_dim),
            nn.GELU(),
        ]
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        self.to_features = nn.Sequential(*head_layers)
        self.output_dim = self.feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stage1_blocks(self.stem1(x))
        h = self.stage2_blocks(self.stem2(h))
        avg = h.mean(dim=(2, 3))
        amax = h.amax(dim=(2, 3))
        pooled = torch.cat([avg, amax], dim=1)
        return self.to_features(pooled)


class VariationalBottleneck(nn.Module):
    """Gaussian VIB head producing ``(z, mu, logvar, kl)``.

    During training ``z = mu + exp(0.5 * logvar) * eps`` with reparameterized
    Gaussian noise. During evaluation ``z = mu`` so predictions are
    deterministic. The KL divergence to ``N(0, I)`` is returned per sample
    so the trainer can apply the configured ``beta`` schedule.
    """

    def __init__(self, feature_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim)
        self.mu = nn.Linear(self.feature_dim, self.latent_dim)
        self.logvar = nn.Linear(self.feature_dim, self.latent_dim)

    def forward(self, h: torch.Tensor, sample: bool | None = None) -> dict[str, torch.Tensor]:
        if sample is None:
            sample = self.training
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-10.0, max=10.0)
        if sample:
            eps = torch.randn_like(mu)
            z = mu + torch.exp(0.5 * logvar) * eps
        else:
            z = mu
        # Per-sample KL to N(0, I).
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1)
        return {"z": z, "mu": mu, "logvar": logvar, "kl": kl}


class EnvAdversaryHead(nn.Module):
    """Linear adversary head reading the gradient-reversed latent."""

    def __init__(self, latent_dim: int, num_classes: int, hidden_dim: int = 0, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        if hidden_dim and hidden_dim > 0:
            layers: list[nn.Module] = [
                nn.Linear(latent_dim, int(hidden_dim)),
                nn.GELU(),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(int(hidden_dim), num_classes))
            self.head = nn.Sequential(*layers)
        else:
            self.head = nn.Linear(latent_dim, num_classes)

    def forward(self, z_rev: torch.Tensor) -> torch.Tensor:
        return self.head(z_rev)


class RulePartitionInvariantBottleneckNet(nn.Module):
    """Bespoke implementation of idea i043's SCRIB architecture.

    The forward returns a dictionary so the trainer can read both the
    primary ``logits`` and the SCRIB auxiliaries (``kl``,
    ``env_logits``, ``env_labels``, ``group_ids``) without an
    idea-specific subclass. ``logits`` is ``(B,)`` for the
    ``puzzle_binary`` BCE-with-logits trainer when ``num_classes=1``;
    ``(B, 2)`` when ``num_classes=2`` for cross-entropy.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding: str = "simple_18",
        latent_dim: int = 128,
        backbone_widths: tuple[int, int] = (64, 96),
        backbone_blocks: int = 2,
        feature_dim: int = 256,
        head_hidden_dim: int = 64,
        dropout: float = 0.1,
        use_norm: bool = True,
        env_grl_lambda: float = 1.0,
        env_adv_hidden_dim: int = 0,
        fail_closed_unknown_channels: bool = True,
    ) -> None:
        super().__init__()
        if num_classes not in {1, 2}:
            raise ValueError(
                "RulePartitionInvariantBottleneckNet supports the puzzle_binary "
                "one-logit BCE contract or two-class CE outputs"
            )
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.latent_dim = int(latent_dim)

        self.canonicalizer = Simple18SideCanonicalizer(
            encoding=encoding,
            input_channels=input_channels,
            fail_closed_unknown_channels=fail_closed_unknown_channels,
        )
        self.partitioner = Simple18RulePartitioner(
            encoding=encoding,
            input_channels=input_channels,
            fail_closed_unknown_channels=fail_closed_unknown_channels,
        )
        self.backbone = ConvTinyBackbone(
            input_channels=self.canonicalizer.output_channels,
            widths=tuple(backbone_widths),
            blocks_per_stage=int(backbone_blocks),
            feature_dim=int(feature_dim),
            dropout=dropout,
            use_norm=use_norm,
        )
        self.vib = VariationalBottleneck(feature_dim=int(feature_dim), latent_dim=self.latent_dim)
        head_hidden = int(head_hidden_dim) if head_hidden_dim and head_hidden_dim > 0 else self.latent_dim // 2
        head_layers: list[nn.Module] = [
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, head_hidden),
            nn.GELU(),
        ]
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        head_layers.append(nn.Linear(head_hidden, num_classes))
        self.label_head = nn.Sequential(*head_layers)

        self.grl = GradientReversalLayer(lambda_=float(env_grl_lambda))
        self.phase_head = EnvAdversaryHead(
            self.latent_dim,
            Simple18RulePartitioner.PHASE_BUCKETS,
            hidden_dim=int(env_adv_hidden_dim),
            dropout=dropout,
        )
        self.adv_head = EnvAdversaryHead(
            self.latent_dim,
            Simple18RulePartitioner.ADV_BUCKETS,
            hidden_dim=int(env_adv_hidden_dim),
            dropout=dropout,
        )
        self.color_head = EnvAdversaryHead(
            self.latent_dim,
            Simple18RulePartitioner.COLOR_BUCKETS,
            hidden_dim=int(env_adv_hidden_dim),
            dropout=dropout,
        )

    def _encode(self, x: torch.Tensor, sample: bool | None) -> dict[str, torch.Tensor]:
        xc = self.canonicalizer(x)
        h = self.backbone(xc)
        return self.vib(h, sample=sample)

    def forward(self, x: torch.Tensor, sample: bool | None = None) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        env = self.partitioner(x)
        latents = self._encode(x, sample=sample)
        z = latents["z"]
        mu = latents["mu"]
        logvar = latents["logvar"]
        kl = latents["kl"]

        head_logits = self.label_head(z)
        logits = format_logits(head_logits, self.num_classes)

        z_rev = self.grl(z)
        env_logits = {
            "phase": self.phase_head(z_rev),
            "adv": self.adv_head(z_rev),
            "color": self.color_head(z_rev),
        }

        # Probabilities returned for diagnostic logging only; the trainer
        # uses the raw logits for cross-entropy.
        env_probs = {key: F.softmax(value, dim=1) for key, value in env_logits.items()}

        return {
            "logits": logits,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "kl": kl,
            "phase_logits": env_logits["phase"],
            "adv_logits": env_logits["adv"],
            "color_logits": env_logits["color"],
            "phase_probs": env_probs["phase"],
            "adv_probs": env_probs["adv"],
            "color_probs": env_probs["color"],
            "phase_labels": env["phase_labels"],
            "adv_labels": env["adv_labels"],
            "color_labels": env["color_labels"],
            "group_ids": env["group_ids"],
            "total_material": env["total_material"],
            "side_relative_advantage": env["side_relative_advantage"],
        }


def build_rule_partition_invariant_bottleneck(config: dict[str, Any]) -> RulePartitionInvariantBottleneckNet:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    backbone_widths = cfg.get("backbone_widths")
    if backbone_widths is None:
        channels = int(cfg.get("channels", 64))
        backbone_widths = (channels, max(channels + 32, 96))
    backbone_widths = tuple(int(v) for v in backbone_widths)
    if len(backbone_widths) != 2:
        raise ValueError("backbone_widths must contain exactly two integers")
    return RulePartitionInvariantBottleneckNet(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        encoding=str(cfg.get("encoding", "simple_18")),
        latent_dim=int(cfg.get("latent_dim", cfg.get("hidden_dim", 128))),
        backbone_widths=backbone_widths,
        backbone_blocks=int(cfg.get("backbone_blocks", cfg.get("depth", 2))),
        feature_dim=int(cfg.get("feature_dim", cfg.get("trunk_feature_dim", 256))),
        head_hidden_dim=int(cfg.get("head_hidden_dim", cfg.get("head_hidden", 64))),
        dropout=float(cfg.get("dropout", 0.1)),
        use_norm=bool(cfg.get("use_norm", cfg.get("use_batchnorm", True))),
        env_grl_lambda=float(cfg.get("env_grl_lambda", 1.0)),
        env_adv_hidden_dim=int(cfg.get("env_adv_hidden_dim", 0)),
        fail_closed_unknown_channels=bool(cfg.get("fail_closed_unknown_channels", True)),
    )


# Idea-local registered builder name follows the project convention:
# ``build_<registered_model_name>_from_config``.
def build_side_canonical_rule_partition_invariant_bottleneck_from_config(
    config: dict[str, Any],
) -> RulePartitionInvariantBottleneckNet:
    return build_rule_partition_invariant_bottleneck(config)
