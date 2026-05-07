"""Credal Near-Puzzle Evidence Network for idea i045.

Bespoke implementation of the architecture in
``ideas/i045_credal_near_puzzle_evidence_network/architecture.md`` and
``math_thesis.md`` (research packet
``chess_nn_research_2026-04-21_0750_tuesday_los_angeles_credal_evidence.md``).

The central operator is a binary Dirichlet evidential head on top of a
compact residual board encoder:

``e_theta(x) in R_+^2``  ->  ``alpha = 1 + e_theta(x)``  ->
``Pi_theta(.|x) = Dirichlet(alpha_0, alpha_1)``.

The puzzle-binary trainer (``num_classes=1``, single-logit BCE) consumes
the binary logit ``log(alpha_1+eps) - log(alpha_0+eps)`` whose sigmoid
equals the Dirichlet predictive mean ``mu_pos = alpha_1 / (alpha_0 +
alpha_1)``. The full ``alpha``, evidence-mass ``S = alpha_0 + alpha_1``
and per-board ``mu_pos`` tensors are exported as auxiliary diagnostics
so the credal/evidential ablations from the markdown can read them.

This module is materially distinct from
``ResearchPacketProbe``/``build_research_packet_probe_from_config`` and
does not import any shared mechanism-profile scaffolding.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


__all__ = [
    "CredalEvidencePuzzleNet",
    "DirichletEvidenceHead",
    "FailClosedBoardAdapter",
    "TinyResidualBoardEncoder",
    "build_credal_near_puzzle_evidence_network_from_config",
]


_KNOWN_BOARD_ENCODINGS = {
    "simple_18": 18,
    "lc0_static_112": 112,
    "lc0_bt4_112": 112,
}


class FailClosedBoardAdapter(nn.Module):
    """1x1 channel adapter that fails closed on unknown channel counts.

    Per the markdown ``Encoding support`` section, the adapter must
    accept ``simple_18`` (18 channels) and the explicitly tagged 112
    channel LC0 layouts and otherwise raise. ``allow_unknown_channels``
    only widens the gate when the caller knowingly opts in.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        encoding: str | None = None,
        allow_unknown_channels: bool = False,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        encoding_text = (encoding or "").strip().lower() or None
        if not allow_unknown_channels:
            expected = _KNOWN_BOARD_ENCODINGS.get(encoding_text) if encoding_text else None
            if encoding_text is None:
                if input_channels not in {18, 112}:
                    raise ValueError(
                        "FailClosedBoardAdapter received unknown channel count "
                        f"{input_channels} without an explicit encoding; set "
                        "model.encoding or allow_unknown_channels=true."
                    )
            elif expected is None:
                raise ValueError(
                    f"FailClosedBoardAdapter does not know encoding {encoding_text!r}; "
                    "set allow_unknown_channels=true to opt into a learned adapter."
                )
            elif expected != input_channels:
                raise ValueError(
                    f"FailClosedBoardAdapter: encoding {encoding_text!r} expects "
                    f"{expected} channels but config has input_channels={input_channels}."
                )
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.encoding = encoding_text
        self.proj = nn.Conv2d(input_channels, hidden_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"FailClosedBoardAdapter expects (batch, channels, 8, 8); got {tuple(x.shape)}"
            )
        if x.shape[1] != self.input_channels or x.shape[2] != 8 or x.shape[3] != 8:
            raise ValueError(
                "FailClosedBoardAdapter input shape mismatch: expected "
                f"(*, {self.input_channels}, 8, 8); got {tuple(x.shape)}"
            )
        return self.proj(x)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm1 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm2 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm1(self.conv1(x)), inplace=True)
        h = self.norm2(self.conv2(h))
        return F.relu(h + x, inplace=True)


class TinyResidualBoardEncoder(nn.Module):
    """Stem + ``num_res_blocks`` residual blocks at fixed ``hidden_channels``."""

    def __init__(
        self,
        hidden_channels: int = 64,
        num_res_blocks: int = 4,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be >= 1")
        self.stem = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(hidden_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[_ResidualBlock(hidden_channels, use_batchnorm=use_batchnorm) for _ in range(num_res_blocks)]
        )
        self.hidden_channels = int(hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))


class DirichletEvidenceHead(nn.Module):
    """Binary Dirichlet head: ``alpha = evidence_floor + softplus(W h)``.

    The floor defaults to ``1.0``, so ``alpha = 1 + softplus(raw)`` which
    matches the math thesis. ``forward`` returns the raw evidence and
    Dirichlet parameters; the surrounding net composes them into logits.
    """

    def __init__(
        self,
        feature_dim: int,
        evidence_floor: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if evidence_floor <= 0:
            raise ValueError("evidence_floor must be positive")
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.linear = nn.Linear(feature_dim, 2)
        self.evidence_floor = float(evidence_floor)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.linear(self.dropout(h))
        evidence = F.softplus(raw)
        alpha = evidence + self.evidence_floor
        return {"raw_evidence": raw, "evidence": evidence, "alpha": alpha}


class CredalEvidencePuzzleNet(nn.Module):
    """Credal Near-Puzzle Evidence Network (idea i045).

    Forward path:

    1. ``FailClosedBoardAdapter`` (1x1 Conv) ``input_channels -> hidden_channels``.
    2. ``TinyResidualBoardEncoder`` (1 stem + ``num_res_blocks`` residual blocks).
    3. Global average pool over the 8x8 board.
    4. ``Linear(hidden_channels -> hidden_dim) -> ReLU -> Dropout``.
    5. ``DirichletEvidenceHead`` -> ``alpha = 1 + softplus(raw)``.
    6. Single-logit binary head: ``puzzle_logit = log(alpha_1 + eps) - log(alpha_0 + eps)``,
       whose sigmoid equals the Dirichlet predictive mean.

    For ``num_classes >= 2`` the model emits ``logits = log(alpha + eps)``
    of shape ``(B, num_classes)`` and only supports the binary
    ``num_classes=2`` head; this preserves the original markdown contract
    (the ``softmax`` of those logits is the Dirichlet mean).
    """

    LOG_EPS: float = 1.0e-8

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        hidden_channels: int = 64,
        hidden_dim: int = 128,
        num_res_blocks: int = 4,
        evidence_floor: float = 1.0,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
        encoding: str | None = None,
        allow_unknown_channels: bool = False,
        near_tau: float = 0.55,
        near_s_max: float = 6.0,
        lambda_near_evidence_cap: float = 0.05,
        lambda_dirichlet_kl: float = 1.0e-3,
        kl_anneal_epochs: int = 2,
        **_unused_metadata: Any,
    ) -> None:
        super().__init__()
        if num_classes not in {1, 2}:
            raise ValueError(
                "CredalEvidencePuzzleNet only supports num_classes in {1, 2}; "
                f"got {num_classes}. Use 1 for single-logit binary BCE; "
                "2 for the markdown's log(alpha) softmax head."
            )
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_res_blocks = int(num_res_blocks)
        self.evidence_floor = float(evidence_floor)
        self.dropout_p = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.encoding = (encoding or "").strip().lower() or None
        self.allow_unknown_channels = bool(allow_unknown_channels)
        # Loss-shaping hyperparameters; the model itself does not consume
        # ``z`` in forward(), but trainers can read these from the module
        # to drive ``CredalEvidenceLoss`` outside the forward pass.
        self.near_tau = float(near_tau)
        self.near_s_max = float(near_s_max)
        self.lambda_near_evidence_cap = float(lambda_near_evidence_cap)
        self.lambda_dirichlet_kl = float(lambda_dirichlet_kl)
        self.kl_anneal_epochs = int(kl_anneal_epochs)

        self.adapter = FailClosedBoardAdapter(
            input_channels=self.input_channels,
            hidden_channels=self.hidden_channels,
            encoding=self.encoding,
            allow_unknown_channels=self.allow_unknown_channels,
        )
        self.encoder = TinyResidualBoardEncoder(
            hidden_channels=self.hidden_channels,
            num_res_blocks=self.num_res_blocks,
            use_batchnorm=self.use_batchnorm,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.evidence_head = DirichletEvidenceHead(
            feature_dim=self.hidden_dim,
            evidence_floor=self.evidence_floor,
            dropout=self.dropout_p,
        )

    def _binary_logit_from_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        log_alpha = torch.log(alpha + self.LOG_EPS)
        return log_alpha[:, 1] - log_alpha[:, 0]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.adapter(x)
        h = self.encoder(h)
        h = self.pool(h).flatten(1)
        h = self.mlp(h)
        head = self.evidence_head(h)
        alpha = head["alpha"]
        evidence = head["evidence"]
        s = alpha.sum(dim=1)
        mu_pos = alpha[:, 1] / s

        if self.num_classes == 1:
            logits: torch.Tensor = self._binary_logit_from_alpha(alpha)
        else:
            logits = torch.log(alpha + self.LOG_EPS)

        return {
            "logits": logits,
            "alpha": alpha,
            "alpha_neg": alpha[:, 0],
            "alpha_pos": alpha[:, 1],
            "evidence": evidence,
            "evidence_neg": evidence[:, 0],
            "evidence_pos": evidence[:, 1],
            "evidence_mass": s,
            "mu_pos": mu_pos,
            "uncertainty": 2.0 / s,
        }


def build_credal_near_puzzle_evidence_network_from_config(config: dict[str, Any]) -> CredalEvidencePuzzleNet:
    cfg = dict(config or {})
    cfg.pop("name", None)
    cfg.pop("packet_profile", None)
    return CredalEvidencePuzzleNet(**cfg)
