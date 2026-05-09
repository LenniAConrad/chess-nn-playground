"""Prototype-Margin Puzzle Network for idea i175.

Faithful implementation of the markdown thesis under
``ideas/i175_prototype_margin_puzzle_network/``. The board is encoded
once and compared against three banks of learned prototypes — random
non-puzzle, near-puzzle, and real puzzle — and the final puzzle logit
is the margin between the puzzle similarity and the maximum
non-puzzle similarity:

    P_random:  K x D
    P_near:    K x D
    P_puzzle:  K x D

    sim_class(z) = logsumexp_k cosine(z, P_class[k]) / temperature
    logit         = sim_puzzle - logsumexp([sim_random, sim_near])

The required ablations from the packet are exposed via ``ablation``:

    * ``"none"`` — main model.
    * ``"single_negative_proto"`` — collapse the random and near-puzzle
      banks into one negative bank. Tests whether the separate near
      prototype is doing real work.
    * ``"no_margin_logsumexp"`` — replace the prototype-margin head
      with a plain linear puzzle head over ``z``. Tests prototype
      competition.
    * ``"random_proto_freeze"`` — freeze the random-proto bank at its
      Kaiming initialization. Tests whether learning the random
      prototypes matters.
    * ``"prototype_count_sweep"`` — no-op structural flag. The sweep
      itself is driven by the ``num_prototypes`` config value; tagging a
      run with this ablation marks it as a sweep entry without changing
      the main model.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardConvStem, BoardTensorSpec, require_board_tensor


_VALID_ABLATIONS = {
    "none",
    "single_negative_proto",
    "no_margin_logsumexp",
    "random_proto_freeze",
    "prototype_count_sweep",
}


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


class PrototypeBank(nn.Module):
    """A bank of ``num_prototypes`` learned unit-direction prototypes of width ``proto_dim``.

    Cosine similarities between the input ``z`` and each prototype are
    aggregated into a single class similarity via
    ``logsumexp(scores) / temperature``, exactly the formulation in the
    packet ``sim_class = logsumexp_k cosine(z, P_class[k]) / temperature``.
    """

    def __init__(
        self,
        num_prototypes: int,
        proto_dim: int,
        temperature: float = 1.0,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        if num_prototypes < 1:
            raise ValueError("num_prototypes must be >= 1")
        if proto_dim < 1:
            raise ValueError("proto_dim must be >= 1")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        self.num_prototypes = int(num_prototypes)
        self.proto_dim = int(proto_dim)
        self.temperature = float(temperature)
        self.freeze = bool(freeze)
        weights = torch.empty(self.num_prototypes, self.proto_dim)
        nn.init.kaiming_uniform_(weights, a=5 ** 0.5)
        self.prototypes = nn.Parameter(weights, requires_grad=not self.freeze)

    def cosine_scores(self, z: torch.Tensor) -> torch.Tensor:
        z_norm = F.normalize(z, dim=-1, eps=1e-8)
        proto_norm = F.normalize(self.prototypes, dim=-1, eps=1e-8)
        return z_norm @ proto_norm.t()  # (B, K)

    def similarity(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.cosine_scores(z)
        sim = torch.logsumexp(scores, dim=-1) / self.temperature
        return sim, scores


class PrototypeMarginPuzzleNetwork(nn.Module):
    """Prototype-margin head over a shared board encoder.

    Forward returns a dict with at least:

      - ``logits``: ``(B,)`` puzzle logit for the BCE-with-logits trainer
        (``(B, num_classes)`` if ``num_classes > 1``).
      - ``prob``: ``sigmoid(logits)`` when ``num_classes == 1``.
      - ``z``: ``(B, proto_dim)`` board latent fed to the prototype banks.
      - ``trunk_features``: ``(B, channels, 8, 8)``.
      - ``trunk_energy``: ``(B,)``.
      - ``sim_random``, ``sim_near``, ``sim_puzzle``: ``(B,)``
        per-class log-sum-exp similarities.
      - ``negative_logsumexp``: ``(B,)`` log-sum-exp over the two
        negative similarities (the right-hand side of the margin).
      - ``puzzle_margin_signal``: ``(B,)`` raw value the puzzle head
        consumes (== ``logits`` when ``num_classes == 1`` and the
        prototype-margin head is active).
      - ``random_scores``, ``near_scores``, ``puzzle_scores``:
        ``(B, num_prototypes)`` cosine similarities to each individual
        prototype, useful for diagnosing collapse.
      - ``num_prototypes_levels``, ``proto_dim_levels``,
        ``temperature_levels``: ``(B,)`` scalar tags carrying the
        configured prototype geometry.
      - ``ablation_active``, ``uses_separate_negatives``,
        ``uses_margin_head``, ``random_proto_frozen``: ``(B,)`` flags
        exposing the running ablation.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        proto_dim: int = 96,
        num_prototypes: int = 8,
        temperature: float = 1.0,
        encoder_hidden: int | None = None,
        head_hidden: int | None = None,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if proto_dim < 1:
            raise ValueError("proto_dim must be >= 1")
        if num_prototypes < 1:
            raise ValueError("num_prototypes must be >= 1")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if ablation not in _VALID_ABLATIONS:
            raise ValueError(
                f"ablation must be one of {sorted(_VALID_ABLATIONS)}, got {ablation!r}"
            )

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.depth = int(depth)
        self.proto_dim = int(proto_dim)
        self.num_prototypes = int(num_prototypes)
        self.temperature = float(temperature)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.ablation = str(ablation)

        encoder_hidden_dim = int(encoder_hidden) if encoder_hidden is not None else self.proto_dim
        self.encoder_hidden = encoder_hidden_dim
        head_hidden_dim = int(head_hidden) if head_hidden is not None else self.proto_dim
        self.head_hidden = head_hidden_dim

        self.trunk = BoardConvStem(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            use_batchnorm=self.use_batchnorm,
        )
        pooled_dim = self.channels * 2  # mean + max
        self.encoder = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, self.encoder_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.encoder_hidden, self.proto_dim),
        )

        self.uses_separate_negatives = self.ablation != "single_negative_proto"
        self.uses_margin_head = self.ablation != "no_margin_logsumexp"
        self.random_proto_frozen = self.ablation == "random_proto_freeze"

        self.puzzle_bank = PrototypeBank(
            num_prototypes=self.num_prototypes,
            proto_dim=self.proto_dim,
            temperature=self.temperature,
        )
        self.random_bank = PrototypeBank(
            num_prototypes=self.num_prototypes,
            proto_dim=self.proto_dim,
            temperature=self.temperature,
            freeze=self.random_proto_frozen,
        )
        if self.uses_separate_negatives:
            self.near_bank = PrototypeBank(
                num_prototypes=self.num_prototypes,
                proto_dim=self.proto_dim,
                temperature=self.temperature,
            )
        else:
            self.near_bank = None  # type: ignore[assignment]

        if self.uses_margin_head:
            self.linear_head = None  # type: ignore[assignment]
        else:
            self.linear_head = nn.Sequential(
                nn.LayerNorm(self.proto_dim),
                nn.Linear(self.proto_dim, self.head_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
                nn.Linear(self.head_hidden, self.num_classes),
            )

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.trunk(x)  # (B, channels, 8, 8)
        mean_pool = feats.mean(dim=(2, 3))
        max_pool = feats.amax(dim=(2, 3))
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        z = self.encoder(pooled)
        return feats, z

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats, z = self._encode(x)
        batch_size = z.shape[0]

        sim_puzzle, puzzle_scores = self.puzzle_bank.similarity(z)
        sim_random, random_scores = self.random_bank.similarity(z)
        if self.near_bank is not None:
            sim_near, near_scores = self.near_bank.similarity(z)
        else:
            # Single-negative ablation: tie near to random so the head still
            # has the same shape and the diagnostics remain coherent.
            sim_near = sim_random
            near_scores = random_scores

        if self.uses_separate_negatives:
            negative_logsumexp = torch.logsumexp(
                torch.stack([sim_random, sim_near], dim=-1), dim=-1
            )
        else:
            negative_logsumexp = sim_random

        if self.uses_margin_head:
            margin_logit = sim_puzzle - negative_logsumexp
            if self.num_classes == 1:
                raw_logits = margin_logit.unsqueeze(-1)
            else:
                # For multi-class outputs, broadcast the margin signal as
                # the puzzle column and zeros elsewhere so the BCE head
                # contract still holds for the puzzle-binary trainer.
                raw_logits = torch.zeros(
                    batch_size, self.num_classes, device=z.device, dtype=z.dtype
                )
                raw_logits[:, -1] = margin_logit
        else:
            assert self.linear_head is not None
            raw_logits = self.linear_head(z)

        logits = _format_logits(raw_logits, self.num_classes)

        if self.num_classes == 1:
            puzzle_margin_signal = logits
        else:
            puzzle_margin_signal = raw_logits[..., -1]

        with torch.no_grad():
            trunk_energy = feats.square().mean(dim=(1, 2, 3))
            ones = torch.ones(batch_size, device=z.device, dtype=z.dtype)
            num_prototypes_levels = ones * float(self.num_prototypes)
            proto_dim_levels = ones * float(self.proto_dim)
            temperature_levels = ones * float(self.temperature)
            ablation_active = ones * (0.0 if self.ablation == "none" else 1.0)
            uses_separate_negatives = ones * (1.0 if self.uses_separate_negatives else 0.0)
            uses_margin_head = ones * (1.0 if self.uses_margin_head else 0.0)
            random_proto_frozen = ones * (1.0 if self.random_proto_frozen else 0.0)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "z": z,
            "trunk_features": feats,
            "trunk_energy": trunk_energy,
            "sim_random": sim_random,
            "sim_near": sim_near,
            "sim_puzzle": sim_puzzle,
            "negative_logsumexp": negative_logsumexp,
            "puzzle_margin_signal": puzzle_margin_signal,
            "random_scores": random_scores,
            "near_scores": near_scores,
            "puzzle_scores": puzzle_scores,
            "num_prototypes_levels": num_prototypes_levels,
            "proto_dim_levels": proto_dim_levels,
            "temperature_levels": temperature_levels,
            "ablation_active": ablation_active,
            "uses_separate_negatives": uses_separate_negatives,
            "uses_margin_head": uses_margin_head,
            "random_proto_frozen": random_proto_frozen,
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_prototype_margin_puzzle_network_from_config(
    config: dict[str, Any],
) -> PrototypeMarginPuzzleNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    hidden_dim = cfg.pop("hidden_dim", None)
    encoder_hidden = cfg.pop("encoder_hidden", hidden_dim)
    head_hidden = cfg.pop("head_hidden", hidden_dim)
    return PrototypeMarginPuzzleNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        depth=int(cfg.pop("depth", 2)),
        proto_dim=int(cfg.pop("proto_dim", 96)),
        num_prototypes=int(cfg.pop("num_prototypes", 8)),
        temperature=float(cfg.pop("temperature", 1.0)),
        encoder_hidden=int(encoder_hidden) if encoder_hidden is not None else None,
        head_hidden=int(head_hidden) if head_hidden is not None else None,
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        ablation=str(cfg.pop("ablation", "none")),
    )


__all__ = [
    "PrototypeBank",
    "PrototypeMarginPuzzleNetwork",
    "build_prototype_margin_puzzle_network_from_config",
]
