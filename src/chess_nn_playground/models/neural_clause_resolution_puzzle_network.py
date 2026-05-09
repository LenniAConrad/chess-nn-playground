"""Neural Clause-Resolution Puzzle Network for idea i201.

Working thesis (from
``ideas/i201_neural_clause_resolution_puzzle_network/math_thesis.md``): a
puzzle often follows from a small proof made of typed facts such as
``Attack(piece, target)``, ``Defends(piece, target)``,
``Pinned(piece, king)``, ``LineOpen(piece, target)``,
``EscapeSquare(square)`` and ``Tempo(side)``. A differentiable
clause-resolution layer derives puzzle evidence from those facts.

This bespoke architecture turns that thesis into an explicit
differentiable pipeline:

1. **Compact convolutional trunk.** ``feats = trunk(x)`` runs ``depth``
   ``Conv2d -> Norm -> GELU -> Dropout2d`` blocks (BatchNorm or
   GroupNorm) and emits ``(B, channels, 8, 8)``.
2. **Initial fact base.** A ``1x1`` convolution produces a per-square
   truth value ``F_0[b, p, s] in [0, 1]`` for each of
   ``num_unary_predicates`` typed predicates (``attack``, ``defends``,
   ``pinned``, ``line_open``, ``escape_square`` by default). A linear
   readout over the pooled trunk summary produces a board-level truth
   value ``G_0[b, g] in [0, 1]`` for each of ``num_global_predicates``
   global predicates (``tempo`` by default). Together these form the
   initial fact base.
3. **Predicate embeddings and clause queries.** Predicates carry an
   embedding table ``predicate_embeddings in R^{P x predicate_dim}``.
   Each clause has a learned head query and ``body_arity`` learned body
   queries (also in ``R^{predicate_dim}``). Softmax over
   ``query @ predicate_embeddings.T`` gives soft predicate selectors
   ``head_sel[c, p]`` and ``body_sel[c, k, p]``. Each body slot also
   carries a soft mixture ``body_rel[c, k, r]`` over
   ``relation_count`` learned spatial relation kernels
   ``relations[r, s, s']`` (row-stochastic over ``s'``). The relation
   kernels implement differentiable variable unification: a body
   predicate evaluated at the head's square ``s`` reads truth values
   at related squares ``s'``.
4. **Differentiable resolution rounds.** ``resolution_rounds`` iterations
   apply a soft Horn rule to each clause:

   ``F_rel[b, r, p, s] = sum_{s'} relations[r, s, s'] * F_unary[b, p, s']``
   ``rel_mixed[b, c, k, p, s] = sum_r body_rel[c, k, r] * F_rel[b, r, p, s]``
   ``body_unary[b, c, k, s] = sum_{p<P_u} body_sel[c, k, p] * rel_mixed[b, c, k, p, s]``
   ``body_global[b, c, k] = sum_g body_sel[c, k, P_u + g] * G[b, g]``
   ``body_score[b, c, k, s] = body_unary[b, c, k, s] + body_global[b, c, k]``

   Soft conjunction is applied across body slots in log-space,
   ``clause_activation[b, c, s] = sum_k log(body_score[b, c, k, s] + eps)
   + clause_bias[c]`` and the clause truth ``clause_truth[b, c, s] =
   sigmoid(clause_activation[b, c, s])`` is then projected back onto
   the head predicates with the soft head selector. The fact base is
   updated by a residual probabilistic-OR with a per-predicate gate:
   ``F_{t+1} = F_t + (1 - F_t) * gate * delta`` (and analogously for
   the globals). This keeps every fact in ``[0, 1]`` and stays
   differentiable.
5. **Classifier head.** The pooled final fact base, the final global
   facts, the pooled trunk summary and a final-round clause-activation
   summary feed into a small MLP that returns one puzzle logit.

Material distinctness from the shared ``ResearchPacketProbe`` scaffold:

* The probe never builds a typed predicate / clause / variable-binding
  structure; this network does.
* The probe never exposes ``unary_fact_trajectory``,
  ``clause_activations`` or row-stochastic ``relation_kernels``; this
  network does.
* The ablations called out by the markdown (``bag_of_facts``,
  ``no_variable_binding``, ``one_round_only``,
  ``random_clause_templates``) collapse the architecture to weaker
  baselines via the config switches ``resolution_rounds: 1``,
  ``relation_count: 1`` (uniform unification), and freezing the
  predicate embeddings.

The architecture is strictly board-only: CRTK / source / verification /
engine metadata is reporting-only and never enters the model.

Tensor contract (``input_channels = 18``, ``S = 64`` squares,
``P_u = num_unary_predicates``, ``P_g = num_global_predicates``,
``P = P_u + P_g``, ``C = clause_count``, ``A = body_arity``,
``R = relation_count``, ``K = resolution_rounds``):

* input ``x``                   shape ``(B, 18, 8, 8)``
* trunk feats                   shape ``(B, channels, 8, 8)``
* initial unary facts           shape ``(B, P_u, S)``
* initial global facts          shape ``(B, P_g)``
* relations                     shape ``(R, S, S)``
* clause head selector          shape ``(C, P)``
* clause body selector          shape ``(C, A, P)``
* clause body relation          shape ``(C, A, R)``
* unary fact trajectory         shape ``(B, K + 1, P_u, S)``
* global fact trajectory        shape ``(B, K + 1, P_g)``
* clause activations tape       shape ``(B, K, C, S)``
* puzzle ``logits``              shape ``(B,)``
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import (
    BoardTensorSpec,
    require_board_tensor,
)


PREDICATE_TOKEN_NAMES: tuple[str, ...] = (
    "attack",
    "defends",
    "pinned",
    "line_open",
    "escape_square",
)


GLOBAL_PREDICATE_TOKEN_NAMES: tuple[str, ...] = (
    "tempo",
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


class NeuralClauseResolutionPuzzleNetwork(nn.Module):
    """Bespoke implementation of idea i201.

    Forward output dict (board-only inputs):

    * ``logits`` ``(B,)`` puzzle logit for the BCE-with-logits trainer
      (``(B, num_classes)`` if ``num_classes > 1``).
    * ``prob`` ``sigmoid(logits)`` when ``num_classes == 1``.
    * ``initial_unary_facts`` ``(B, P_u, S)`` ``F_0``.
    * ``initial_global_facts`` ``(B, P_g)`` ``G_0``.
    * ``final_unary_facts`` ``(B, P_u, S)`` ``F_K``.
    * ``final_global_facts`` ``(B, P_g)`` ``G_K``.
    * ``unary_fact_trajectory`` ``(B, K + 1, P_u, S)``.
    * ``global_fact_trajectory`` ``(B, K + 1, P_g)``.
    * ``clause_activations`` ``(B, K, C, S)`` per-round soft Horn
      activations.
    * ``clause_head_selector`` ``(C, P)`` soft predicate distribution
      for the clause heads.
    * ``clause_body_selector`` ``(C, A, P)`` soft predicate
      distributions for the clause body slots.
    * ``clause_body_relation`` ``(C, A, R)`` mixture weights over the
      learned relation kernels.
    * ``predicate_embeddings`` ``(P, predicate_dim)`` learned predicate
      tokens.
    * ``relation_kernels`` ``(R, S, S)`` row-stochastic spatial
      relation kernels.
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
        predicate_dim: int = 64,
        clause_count: int = 32,
        resolution_rounds: int = 4,
        num_unary_predicates: int = 5,
        num_global_predicates: int = 1,
        body_arity: int = 3,
        relation_count: int = 8,
        height: int = 8,
        width: int = 8,
        **_: Any,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if predicate_dim < 1:
            raise ValueError("predicate_dim must be >= 1")
        if clause_count < 1:
            raise ValueError("clause_count must be >= 1")
        if resolution_rounds < 1:
            raise ValueError("resolution_rounds must be >= 1")
        if num_unary_predicates < 1:
            raise ValueError("num_unary_predicates must be >= 1")
        if num_global_predicates < 0:
            raise ValueError("num_global_predicates must be >= 0")
        if body_arity < 1:
            raise ValueError("body_arity must be >= 1")
        if relation_count < 1:
            raise ValueError("relation_count must be >= 1")

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
        self.predicate_dim = int(predicate_dim)
        self.clause_count = int(clause_count)
        self.resolution_rounds = int(resolution_rounds)
        self.num_unary_predicates = int(num_unary_predicates)
        self.num_global_predicates = int(num_global_predicates)
        self.body_arity = int(body_arity)
        self.relation_count = int(relation_count)
        self.height = int(height)
        self.width = int(width)
        self.num_squares = self.height * self.width
        self.num_predicates = self.num_unary_predicates + self.num_global_predicates

        self.trunk = _BoardTrunk(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            dropout=self.dropout_p,
            use_batchnorm=self.use_batchnorm,
        )

        # Initial fact extractors.
        self.unary_fact_logits = nn.Conv2d(
            self.channels, self.num_unary_predicates, kernel_size=1
        )
        if self.num_global_predicates > 0:
            self.global_fact_logits = nn.Linear(
                self.channels * 3, self.num_global_predicates
            )
        else:
            self.global_fact_logits = None

        scale = 1.0 / max(1.0, float(self.predicate_dim) ** 0.5)
        # Predicate embedding table -- soft predicate selectors are
        # softmax over (query @ predicate_embeddings.T).
        self.predicate_embeddings = nn.Parameter(
            torch.randn(self.num_predicates, self.predicate_dim) * scale
        )
        # Per-clause queries for the head and body slots.
        self.clause_head_query = nn.Parameter(
            torch.randn(self.clause_count, self.predicate_dim) * scale
        )
        self.clause_body_query = nn.Parameter(
            torch.randn(self.clause_count, self.body_arity, self.predicate_dim) * scale
        )
        # Relation mixture weights per clause body slot.
        self.clause_body_relation_logits = nn.Parameter(
            torch.zeros(self.clause_count, self.body_arity, self.relation_count)
        )
        # Spatial relation kernel logits; softmax over the destination
        # square (last axis) gives a row-stochastic kernel.
        self.relation_logits = nn.Parameter(
            torch.zeros(self.relation_count, self.num_squares, self.num_squares)
        )
        # Bias for the soft Horn rule activation.
        self.clause_bias = nn.Parameter(torch.zeros(self.clause_count))
        # Per-predicate residual update gates (sigmoid-bounded).
        self.unary_update_gate = nn.Parameter(
            torch.full((self.num_unary_predicates,), -1.0)
        )
        self.global_update_gate = nn.Parameter(
            torch.full((max(self.num_global_predicates, 1),), -1.0)
        )

        # Head feature pack:
        #   final_unary pooled (mean, max)            (2 * P_u)
        #   final_global facts                        (P_g)
        #   pooled trunk summary (mean, max, energy)  (3)
        #   final-round clause-activation summary     (2)
        head_in = (
            2 * self.num_unary_predicates
            + self.num_global_predicates
            + 3
            + 2
        )
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

    def predicate_token_names(self) -> tuple[str, ...]:
        names = list(PREDICATE_TOKEN_NAMES[: self.num_unary_predicates])
        while len(names) < self.num_unary_predicates:
            names.append(f"predicate_{len(names)}")
        return tuple(names)

    def global_predicate_token_names(self) -> tuple[str, ...]:
        names = list(GLOBAL_PREDICATE_TOKEN_NAMES[: self.num_global_predicates])
        while len(names) < self.num_global_predicates:
            names.append(f"global_predicate_{len(names)}")
        return tuple(names)

    def _pool_board(
        self, feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = feats.mean(dim=(2, 3))
        max_pool = feats.amax(dim=(2, 3))
        energy = feats.square().mean(dim=(2, 3))
        pack = torch.cat([mean, max_pool, energy], dim=-1)  # (B, 3 * channels)
        pooled_summary = torch.stack(
            [mean.mean(dim=-1), max_pool.mean(dim=-1), energy.mean(dim=-1)],
            dim=-1,
        )
        return pack, pooled_summary

    def _initial_facts(
        self, feats: torch.Tensor, pooled_pack: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unary_logits = self.unary_fact_logits(feats)  # (B, P_u, 8, 8)
        F_unary = torch.sigmoid(unary_logits).flatten(2)  # (B, P_u, S)
        if self.global_fact_logits is not None and self.num_global_predicates > 0:
            G = torch.sigmoid(self.global_fact_logits(pooled_pack))
        else:
            G = torch.zeros(
                feats.shape[0], 0, device=feats.device, dtype=feats.dtype
            )
        return F_unary, G

    def _selectors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        head_logits = self.clause_head_query @ self.predicate_embeddings.t()
        head_sel = head_logits.softmax(dim=-1)  # (C, P)
        body_logits = self.clause_body_query @ self.predicate_embeddings.t()
        body_sel = body_logits.softmax(dim=-1)  # (C, A, P)
        body_rel = self.clause_body_relation_logits.softmax(dim=-1)  # (C, A, R)
        relations = self.relation_logits.softmax(dim=-1)  # (R, S, S)
        return head_sel, body_sel, body_rel, relations

    def _resolution_step(
        self,
        F_unary: torch.Tensor,
        G: torch.Tensor,
        head_sel: torch.Tensor,
        body_sel: torch.Tensor,
        body_rel: torch.Tensor,
        relations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Apply each relation to the unary fact base.
        # F_rel[b, r, p, s] = sum_{s'} relations[r, s, s'] * F_unary[b, p, s']
        F_rel = torch.einsum("rsq,bpq->brps", relations, F_unary)

        # Mix relations per body slot.
        # rel_mixed[b, c, k, p, s] = sum_r body_rel[c, k, r] * F_rel[b, r, p, s]
        rel_mixed = torch.einsum("ckr,brps->bckps", body_rel, F_rel)

        body_sel_unary = body_sel[..., : self.num_unary_predicates]  # (C, A, P_u)
        # Mix predicates within the unary part.
        body_unary_score = torch.einsum(
            "ckp,bckps->bcks", body_sel_unary, rel_mixed
        )  # (B, C, A, S)

        if self.num_global_predicates > 0:
            body_sel_global = body_sel[..., self.num_unary_predicates :]  # (C, A, P_g)
            body_global_score = torch.einsum("ckg,bg->bck", body_sel_global, G)
            body_score = body_unary_score + body_global_score.unsqueeze(-1)
        else:
            body_score = body_unary_score

        # Soft AND across body slots in log-space.
        eps = 1.0e-6
        body_log = torch.log(body_score.clamp_min(eps))  # (B, C, A, S)
        clause_activation = body_log.sum(dim=2) + self.clause_bias.view(1, -1, 1)
        clause_truth = torch.sigmoid(clause_activation)  # (B, C, S)

        # Project clause truth onto the head predicates.
        head_sel_unary = head_sel[:, : self.num_unary_predicates]  # (C, P_u)
        delta_unary = torch.einsum(
            "cp,bcs->bps", head_sel_unary, clause_truth
        )  # (B, P_u, S) in [0, num_unary_predicates]; head_sel_unary cols sum
        # to <= 1, so along p the sum is at most 1 per (c,s), bounded by 1.
        # gate_unary in [0, 1] per predicate.
        gate_unary = torch.sigmoid(self.unary_update_gate).view(1, -1, 1)
        # Probabilistic OR: F_new = F + (1 - F) * gate * delta in [0, 1].
        delta_unary = delta_unary.clamp_min(0.0).clamp_max(1.0)
        F_unary_new = F_unary + (1.0 - F_unary) * gate_unary * delta_unary

        if self.num_global_predicates > 0:
            head_sel_global = head_sel[:, self.num_unary_predicates :]  # (C, P_g)
            clause_truth_global = clause_truth.mean(dim=-1)  # (B, C)
            delta_global = torch.einsum(
                "cg,bc->bg", head_sel_global, clause_truth_global
            )
            delta_global = delta_global.clamp_min(0.0).clamp_max(1.0)
            gate_global = torch.sigmoid(
                self.global_update_gate[: self.num_global_predicates]
            ).view(1, -1)
            G_new = G + (1.0 - G) * gate_global * delta_global
        else:
            G_new = G

        return F_unary_new, G_new, clause_activation

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.trunk(x)
        pooled_pack, pooled_summary = self._pool_board(feats)
        F_unary, G = self._initial_facts(feats, pooled_pack)
        head_sel, body_sel, body_rel, relations = self._selectors()

        unary_traj = [F_unary]
        global_traj = [G]
        clause_activations: list[torch.Tensor] = []
        for _ in range(self.resolution_rounds):
            F_unary, G, clause_activation = self._resolution_step(
                F_unary, G, head_sel, body_sel, body_rel, relations
            )
            unary_traj.append(F_unary)
            global_traj.append(G)
            clause_activations.append(clause_activation)

        unary_trajectory = torch.stack(unary_traj, dim=1)  # (B, K+1, P_u, S)
        global_trajectory = torch.stack(global_traj, dim=1)  # (B, K+1, P_g)
        clause_activation_tape = torch.stack(clause_activations, dim=1)
        # ^ shape (B, K, C, S)

        final_clause_activation = clause_activations[-1]  # (B, C, S)
        clause_summary_mean = final_clause_activation.mean(dim=(1, 2))  # (B,)
        clause_summary_max = final_clause_activation.amax(dim=(1, 2))  # (B,)

        final_unary_mean = F_unary.mean(dim=-1)  # (B, P_u)
        final_unary_max = F_unary.amax(dim=-1)  # (B, P_u)

        head_input_parts = [final_unary_mean, final_unary_max]
        if self.num_global_predicates > 0:
            head_input_parts.append(G)
        head_input_parts.append(pooled_summary)
        head_input_parts.append(
            torch.stack([clause_summary_mean, clause_summary_max], dim=-1)
        )
        head_input = torch.cat(head_input_parts, dim=-1)
        head_input = self.head_norm(head_input)
        raw_logits = self.head(head_input)
        logits = _format_logits(raw_logits, self.num_classes)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "initial_unary_facts": unary_trajectory[:, 0],
            "initial_global_facts": global_trajectory[:, 0],
            "final_unary_facts": F_unary,
            "final_global_facts": G,
            "unary_fact_trajectory": unary_trajectory,
            "global_fact_trajectory": global_trajectory,
            "clause_activations": clause_activation_tape,
            "clause_head_selector": head_sel,
            "clause_body_selector": body_sel,
            "clause_body_relation": body_rel,
            "predicate_embeddings": self.predicate_embeddings,
            "relation_kernels": relations,
            "trunk_energy": feats.square().mean(dim=(1, 2, 3)),
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_neural_clause_resolution_puzzle_network_from_config(
    config: dict[str, Any],
) -> NeuralClauseResolutionPuzzleNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    return NeuralClauseResolutionPuzzleNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        depth=int(cfg.pop("depth", 2)),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        predicate_dim=int(cfg.pop("predicate_dim", 64)),
        clause_count=int(cfg.pop("clause_count", 32)),
        resolution_rounds=int(cfg.pop("resolution_rounds", 4)),
        num_unary_predicates=int(cfg.pop("num_unary_predicates", 5)),
        num_global_predicates=int(cfg.pop("num_global_predicates", 1)),
        body_arity=int(cfg.pop("body_arity", 3)),
        relation_count=int(cfg.pop("relation_count", 8)),
        height=int(cfg.pop("height", 8)),
        width=int(cfg.pop("width", 8)),
    )


__all__ = [
    "NeuralClauseResolutionPuzzleNetwork",
    "PREDICATE_TOKEN_NAMES",
    "GLOBAL_PREDICATE_TOKEN_NAMES",
    "build_neural_clause_resolution_puzzle_network_from_config",
]
