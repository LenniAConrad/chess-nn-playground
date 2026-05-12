"""Absorbing Threat Markov Network for idea i200.

Working thesis (from ``ideas/registry/i200_absorbing_threat_markov_network/math_thesis.md``):
puzzle detection can be treated as a probabilistic process over tactical
states::

    pressure -> threat -> forced response -> collapse/proof
    pressure -> safe response -> disproof

A full proof tree is expensive, but a compact absorbing Markov chain
over a small set of named tactical states can approximate whether the
position tends toward proof (puzzle) or disproof (non-puzzle).

This bespoke architecture turns that thesis into an explicit
differentiable computation:

1. **Compact convolutional trunk.** ``feats = trunk(x)`` runs ``depth``
   ``Conv2d -> Norm -> GELU -> Dropout2d`` blocks (BatchNorm or
   GroupNorm) and emits ``(B, channels, 8, 8)``.
2. **Board context.** Mean / max / energy pooling over ``feats`` plus a
   linear projection produce a ``(B, state_dim)`` context vector and a
   ``(B, 3)`` pooled summary used by the head.
3. **Named tactical state tokens.** A learnable embedding table of
   shape ``(state_count, state_dim)`` represents the named tokens
   ``attack_pressure``, ``defender_available``, ``line_open``,
   ``king_constrained``, ``target_hanging``, ``counterplay``,
   ``proof_absorb``, ``disproof_absorb`` (the last two are absorbing).
4. **Initial distribution over transient states.** A bilinear scorer
   over ``(board_context, transient_state_tokens)`` plus a soft-attention
   readout over the per-square feature map produce per-state initial
   logits. Softmax over the transient states gives ``pi_0`` with zero
   mass on the two absorbing states.
5. **Board-conditioned transition matrix ``P``.** Transient rows come
   from a board-modulated bilinear form on the state embeddings:
   ``logits[b, i, j] = state_emb[i] . (board_proj[b] * state_emb[j]) +
   bias[i, j]``; row softmax over ``j`` yields a row-stochastic
   distribution over *all* ``state_count`` states. Absorbing rows are
   forced to identity (``P[absorb, absorb] = 1``) so absorbing states
   trap probability mass. The full matrix is row-stochastic by
   construction.
6. **Power iteration for absorption.** With ``transition_steps = T``,
   ``pi_t = pi_{t-1} P`` is iterated ``T`` times from ``pi_0``. The
   probability mass that has reached each absorbing state at step ``t``
   is recorded; in the limit ``T -> infty`` these are the absorption
   probabilities of the chain. With finite ``T`` they approximate them
   from below.
7. **Head.** Read out ``prob_proof = pi_T[proof_absorb]``,
   ``prob_disproof = pi_T[disproof_absorb]``, the soft expected number
   of pre-absorption steps ``E[steps] = sum_{t<T} (1 - prob_proof_t -
   prob_disproof_t)``, the proof/disproof gap, the final state
   distribution, and pooled board features. A small MLP projects this
   pack to one puzzle logit.

Material distinctness from the shared ``ResearchPacketProbe`` scaffold:

* The probe never builds a board-conditioned state-by-state transition
  matrix; this network does.
* The probe never iterates probability mass toward absorbing states or
  reads ``prob_proof`` / ``prob_disproof`` / ``expected_steps`` as a
  feature vector; this network does.
* Removing the absorbing-state structure (``no_absorbing_states``
  ablation) or running zero transition iterations
  (``one_step_only`` ablation) collapses the architecture to a
  context-only MLP, which is exactly what the markdown's falsification
  table requires.

The architecture is strictly board-only: CRTK / source / verification /
engine metadata is reporting-only and never enters the model.

Tensor contract (``input_channels = 18``, ``state_count = K``,
``transient_states = K - 2``):

* input ``x``                    shape ``(B, 18, 8, 8)``
* trunk feats                    shape ``(B, channels, 8, 8)``
* board_context                  shape ``(B, state_dim)``
* state_embeddings               shape ``(K, state_dim)``
* initial_distribution ``pi_0``  shape ``(B, K)``
* transition_matrix ``P``        shape ``(B, K, K)``
* state_distributions            shape ``(B, T+1, K)``
* prob_proof                     shape ``(B,)``
* prob_disproof                  shape ``(B,)``
* expected_steps                 shape ``(B,)``
* puzzle ``logits``              shape ``(B,)``
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import (
    BoardTensorSpec,
    require_board_tensor,
)


STATE_TOKEN_NAMES: tuple[str, ...] = (
    "attack_pressure",
    "defender_available",
    "line_open",
    "king_constrained",
    "target_hanging",
    "counterplay",
    "proof_absorb",
    "disproof_absorb",
)


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


class _BoardTrunk(nn.Module):
    """Compact convolutional trunk over the board planes."""

    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        layers: list[nn.Module] = []
        in_ch = input_channels
        for _ in range(depth):
            layers.append(
                nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            else:
                layers.append(nn.GroupNorm(1, channels))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = channels
        self.body = nn.Sequential(*layers)
        self.output_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class AbsorbingThreatMarkovNetwork(nn.Module):
    """Bespoke implementation of idea i200.

    Forward output dict (board-only inputs):

    * ``logits`` ``(B,)`` puzzle logit for the BCE-with-logits trainer
      (``(B, num_classes)`` if ``num_classes > 1``).
    * ``prob`` ``sigmoid(logits)`` when ``num_classes == 1``.
    * ``board_context`` ``(B, state_dim)`` projected board context.
    * ``state_embeddings`` ``(K, state_dim)`` learned state tokens.
    * ``initial_distribution`` ``(B, K)`` ``pi_0`` (zero on absorbing
      states).
    * ``transition_matrix`` ``(B, K, K)`` row-stochastic ``P`` (last two
      rows are identity because the proof/disproof states are
      absorbing).
    * ``state_distributions`` ``(B, T+1, K)`` ``pi_0, pi_1, ..., pi_T``.
    * ``final_distribution`` ``(B, K)`` ``pi_T``.
    * ``prob_proof`` ``(B,)`` ``pi_T[proof_absorb]``.
    * ``prob_disproof`` ``(B,)`` ``pi_T[disproof_absorb]``.
    * ``proof_minus_disproof`` ``(B,)`` ``prob_proof - prob_disproof``.
    * ``expected_steps`` ``(B,)`` soft expected pre-absorption steps,
      ``sum_{t < T} (1 - pi_t[proof_absorb] - pi_t[disproof_absorb])``.
    * ``transient_initial`` ``(B, K-2)`` ``pi_0`` restricted to the
      transient states.
    * ``trunk_energy`` ``(B,)`` mean-square trunk activation.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        state_count: int = 8,
        state_dim: int = 96,
        transition_steps: int = 4,
        height: int = 8,
        width: int = 8,
        **_: Any,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if state_count < 4:
            raise ValueError(
                "state_count must be >= 4 to host at least two transient and two absorbing states"
            )
        if state_dim < 1:
            raise ValueError("state_dim must be >= 1")
        if transition_steps < 1:
            raise ValueError("transition_steps must be >= 1")

        self.spec = BoardTensorSpec(
            input_channels=int(input_channels), height=int(height), width=int(width)
        )
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout_p = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.state_count = int(state_count)
        self.state_dim = int(state_dim)
        self.transition_steps = int(transition_steps)
        self.transient_count = self.state_count - 2
        self.proof_index = self.state_count - 2
        self.disproof_index = self.state_count - 1

        self.trunk = _BoardTrunk(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            dropout=self.dropout_p,
            use_batchnorm=self.use_batchnorm,
        )

        # Project pooled board features into the state-token latent so
        # that the transition matrix and the initial distribution share
        # a common semantic space.
        self.board_to_state = nn.Linear(self.channels * 3, self.state_dim)

        # Learnable named state tokens. The first ``transient_count``
        # rows correspond to transient states (attack_pressure, ...),
        # the last two rows to the absorbing proof/disproof states.
        self.state_embeddings = nn.Parameter(
            torch.randn(self.state_count, self.state_dim) * (1.0 / max(1.0, float(self.state_dim) ** 0.5))
        )

        # Per-state spatial attention over the trunk feature map: each
        # transient state token reads a board-conditioned mass that
        # seeds the initial distribution.
        self.state_attention = nn.Conv2d(self.channels, self.transient_count, kernel_size=1)
        self.initial_bias = nn.Parameter(torch.zeros(self.transient_count))

        # Bias term for the transition logits (over all K states).
        self.transition_bias = nn.Parameter(torch.zeros(self.state_count, self.state_count))

        # Head feature pack:
        #   final_distribution                       (state_count)
        #   prob_proof, prob_disproof, gap           (3)
        #   expected_steps                           (1)
        #   transient_initial                        (transient_count)
        #   pooled trunk summary (mean, max, energy) (3)
        head_in = self.state_count + 3 + 1 + self.transient_count + 3
        self.head_norm = nn.LayerNorm(head_in)
        head_layers: list[nn.Module] = [
            nn.Linear(head_in, self.hidden_dim),
            nn.GELU(),
        ]
        if self.dropout_p > 0.0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.hidden_dim, self.num_classes))
        self.head = nn.Sequential(*head_layers)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def state_token_names(self) -> tuple[str, ...]:
        if self.state_count == len(STATE_TOKEN_NAMES):
            return STATE_TOKEN_NAMES
        names = list(STATE_TOKEN_NAMES[: self.state_count - 2])
        while len(names) < self.state_count - 2:
            names.append(f"transient_{len(names)}")
        names.extend(("proof_absorb", "disproof_absorb"))
        return tuple(names)

    def _pool_board(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (board_context, pooled_summary).

        ``board_context`` has shape ``(B, state_dim)`` and lives in the
        same latent as the state tokens. ``pooled_summary`` collects
        ``(mean, max, energy)`` channel-pooled scalars used by the head.
        """
        mean = feats.mean(dim=(2, 3))
        max_pool = feats.amax(dim=(2, 3))
        energy = feats.square().mean(dim=(2, 3))
        pack = torch.cat([mean, max_pool, energy], dim=-1)  # (B, 3 * channels)
        board_context = self.board_to_state(pack)
        # Pooled summary scalars (one number per pool type) used by the
        # final head; collapse the channel dimension by averaging.
        pooled_summary = torch.stack(
            [mean.mean(dim=-1), max_pool.mean(dim=-1), energy.mean(dim=-1)],
            dim=-1,
        )
        return board_context, pooled_summary

    def _initial_distribution(
        self, feats: torch.Tensor, board_context: torch.Tensor
    ) -> torch.Tensor:
        """Compute ``pi_0`` over all states (zero mass on absorbing states)."""
        # Spatial-attention scores per transient state token.
        attention_logits = self.state_attention(feats)  # (B, T, 8, 8)
        attention = attention_logits.flatten(2).softmax(dim=-1)  # (B, T, H*W)
        feats_flat = feats.flatten(2)  # (B, C, H*W)
        # Per-token board reading: weighted average over squares.
        per_state_board = torch.einsum("btn,bcn->btc", attention, feats_flat)  # (B, T, C)
        per_state_board = per_state_board.mean(dim=-1)  # (B, T) summary scalar
        # Bilinear alignment between board_context and transient state tokens.
        transient_tokens = self.state_embeddings[: self.transient_count]  # (T, D)
        alignment = board_context @ transient_tokens.t()  # (B, T)
        logits = alignment + per_state_board + self.initial_bias  # (B, T)
        transient_pi = logits.softmax(dim=-1)  # (B, T)
        # Embed in the full K-state simplex with zero mass on absorbing.
        pi_0 = torch.zeros(
            transient_pi.shape[0], self.state_count, device=transient_pi.device, dtype=transient_pi.dtype
        )
        pi_0[:, : self.transient_count] = transient_pi
        return pi_0

    def _transition_matrix(self, board_context: torch.Tensor) -> torch.Tensor:
        """Return a row-stochastic ``(B, K, K)`` transition matrix.

        The two absorbing rows (proof, disproof) are identity rows so
        probability mass cannot leak out of them.
        """
        # Modulate the second state embedding axis by the board context
        # so the transition logits depend on the board.
        board_proj = board_context  # already in state_dim
        # logits[b, i, j] = sum_d state_emb[i, d] * (board_proj[b, d] * state_emb[j, d])
        modulated_states = self.state_embeddings.unsqueeze(0) * board_proj.unsqueeze(1)
        # modulated_states: (B, K, D); state_embeddings: (K, D)
        logits = torch.einsum("id,bjd->bij", self.state_embeddings, modulated_states)
        logits = logits + self.transition_bias.unsqueeze(0)
        # Row softmax over destination states for all rows.
        row_softmax = logits.softmax(dim=-1)
        # Force absorbing rows to identity.
        K = self.state_count
        absorb_eye = torch.zeros(K, K, device=logits.device, dtype=logits.dtype)
        absorb_eye[self.proof_index, self.proof_index] = 1.0
        absorb_eye[self.disproof_index, self.disproof_index] = 1.0
        # Build a (K,) row mask: 1 for transient rows, 0 for absorbing.
        row_mask = torch.ones(K, device=logits.device, dtype=logits.dtype)
        row_mask[self.proof_index] = 0.0
        row_mask[self.disproof_index] = 0.0
        P = row_softmax * row_mask.view(1, K, 1) + absorb_eye.unsqueeze(0)
        return P

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.trunk(x)  # (B, C, 8, 8)
        board_context, pooled_summary = self._pool_board(feats)

        pi_0 = self._initial_distribution(feats, board_context)  # (B, K)
        P = self._transition_matrix(board_context)  # (B, K, K)

        # Power iterate: pi_t = pi_{t-1} P. Track all distributions for
        # the soft expected-steps readout.
        distributions = [pi_0]
        pi_t = pi_0
        for _ in range(self.transition_steps):
            pi_t = torch.bmm(pi_t.unsqueeze(1), P).squeeze(1)  # (B, K)
            distributions.append(pi_t)
        state_distributions = torch.stack(distributions, dim=1)  # (B, T+1, K)

        prob_proof = pi_t[:, self.proof_index]
        prob_disproof = pi_t[:, self.disproof_index]
        gap = prob_proof - prob_disproof
        absorbing_mass_per_step = (
            state_distributions[:, :-1, self.proof_index]
            + state_distributions[:, :-1, self.disproof_index]
        )  # (B, T)
        expected_steps = (1.0 - absorbing_mass_per_step).clamp_min(0.0).sum(dim=-1)

        transient_initial = pi_0[:, : self.transient_count]
        trunk_energy = feats.square().mean(dim=(1, 2, 3))

        head_input = torch.cat(
            [
                pi_t,
                prob_proof.unsqueeze(-1),
                prob_disproof.unsqueeze(-1),
                gap.unsqueeze(-1),
                expected_steps.unsqueeze(-1),
                transient_initial,
                pooled_summary,
            ],
            dim=-1,
        )
        head_input = self.head_norm(head_input)
        raw_logits = self.head(head_input)
        logits = _format_logits(raw_logits, self.num_classes)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "board_context": board_context,
            "state_embeddings": self.state_embeddings,
            "initial_distribution": pi_0,
            "transition_matrix": P,
            "state_distributions": state_distributions,
            "final_distribution": pi_t,
            "prob_proof": prob_proof,
            "prob_disproof": prob_disproof,
            "proof_minus_disproof": gap,
            "expected_steps": expected_steps,
            "transient_initial": transient_initial,
            "trunk_energy": trunk_energy,
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_absorbing_threat_markov_network_from_config(
    config: dict[str, Any],
) -> AbsorbingThreatMarkovNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    return AbsorbingThreatMarkovNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        depth=int(cfg.pop("depth", 2)),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        state_count=int(cfg.pop("state_count", 8)),
        state_dim=int(cfg.pop("state_dim", 96)),
        transition_steps=int(cfg.pop("transition_steps", 4)),
        height=int(cfg.pop("height", 8)),
        width=int(cfg.pop("width", 8)),
    )


__all__ = [
    "AbsorbingThreatMarkovNetwork",
    "STATE_TOKEN_NAMES",
    "build_absorbing_threat_markov_network_from_config",
]
