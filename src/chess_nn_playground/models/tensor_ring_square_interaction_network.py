"""Tensor-Ring Square Interaction Network for idea i119.

Working thesis (from ``ideas/i119_tensor_ring_square_interaction_network``):
chess cues depend on interactions among several squares at once -- king
square, attacking piece, blocker, defender, escape square, promotion path.
The full square-tuple interaction tensor over 64 squares is too large, but
a tensor-ring factorization can model high-order interactions with a
controlled parameter budget.

Concretely, this model:

1.  Flattens the ``simple_18`` board tensor into 64 square tokens and
    augments each token with a small coordinate embedding (rank, file,
    side-relative rank, center distance, square parity).
2.  Computes a learned bank of ``R`` *role gates* ``g_r(s) in [0, 1]``
    per square -- the gates correspond to roles like "own piece",
    "opponent piece", "king zone", "ray-relevant square", "empty
    square". Gates are produced from token features by a single
    ``Linear -> Sigmoid``, so they are learned, not hard-coded
    legal-move labels.
3.  Holds, for each interaction order ``K`` in ``orders``, a stack of
    ``K`` independent tensor-ring cores
    ``G_1, ..., G_K : R^D -> R^{r x r}`` implemented as ``Linear``
    layers from token width ``D`` to ``r * r`` followed by a reshape.
4.  Holds, for each order ``K``, a learned bank of ``P`` role
    *patterns*: a ``(P, K, R)`` parameter that, after softmax over the
    role axis, picks which gate each slot of each pattern attends to.
    This is the learned analogue of a sequence like
    ``own_attacker -> blocker -> king_zone``.
5.  For every pattern ``p`` of order ``K`` the model evaluates the
    cyclic contraction

    ::

        M_{p,k} = sum_s alpha_{p,k,r} g_r(s) * G_k(x_s)
        z_{p}    = trace(M_{p,1} M_{p,2} ... M_{p,K})

    using only ``O(64 * P * K * r^2)`` work; tuples of squares are
    never enumerated explicitly. Each ``M_{p,k}`` is normalised by the
    number of squares so the contractions stay numerically stable.
6.  Pools the ``(B, P)`` contraction matrix into per-order summary
    statistics ``mean, max, variance, signed_abs_mean`` and concatenates
    the raw pattern responses with those statistics.
7.  Adds a small CNN stem summary so the tensor-ring branch is graded
    against a board-aware control signal. The classifier reads the
    concatenated ``[contraction_stats, cnn_summary]`` and emits one
    puzzle logit plus the diagnostic outputs.

This is materially distinct from the shared ``ResearchPacketProbe``
scaffold (which has no tensor-ring cores, no learned role gates, no
cyclic-trace contractions, and no per-pattern diagnostics) and from the
neighbouring ``tensor_core_square_pair_field_network`` and
``tensorsketch_interaction_network`` (the former uses dense pair
attention, the latter randomised polynomial sketches; neither performs a
learned cyclic trace of low-rank cores driven by learned role gates).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


PIECE_PLANES = 12
ROLE_GATE_NAMES: tuple[str, ...] = (
    "own_piece",
    "opp_piece",
    "king_zone",
    "ray_relevant",
    "empty_square",
)
ROLE_GATE_COUNT = len(ROLE_GATE_NAMES)
COORD_FEATURE_DIM = 5  # rank01, file01, side_relative_rank, center_distance, square_parity


def _coord_features(height: int, width: int) -> torch.Tensor:
    rank = torch.arange(height, dtype=torch.float32).view(height, 1).expand(height, width)
    file = torch.arange(width, dtype=torch.float32).view(1, width).expand(height, width)
    rank01 = rank / max(1.0, float(height - 1))
    file01 = file / max(1.0, float(width - 1))
    center_distance = torch.maximum(
        (rank - (height - 1) / 2.0).abs() / max(1.0, (height - 1) / 2.0),
        (file - (width - 1) / 2.0).abs() / max(1.0, (width - 1) / 2.0),
    )
    parity = ((rank + file) % 2.0) * 2.0 - 1.0
    coords = torch.stack([rank01, file01, rank01, center_distance, parity], dim=0)
    return coords


class _SquareTokenEncoder(nn.Module):
    """Builds (B, 64, D) tokens from a (B, C, H, W) board tensor."""

    def __init__(self, input_channels: int, token_dim: int, height: int, width: int) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.token_dim = int(token_dim)
        self.height = int(height)
        self.width = int(width)
        self.proj = nn.Linear(self.input_channels + COORD_FEATURE_DIM, self.token_dim)
        self.norm = nn.LayerNorm(self.token_dim)
        coords = _coord_features(self.height, self.width)
        self.register_buffer("coord_features", coords, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        # side-to-move broadcast plane (plane 12 in simple_18)
        if self.input_channels > PIECE_PLANES:
            stm = x[:, PIECE_PLANES].mean(dim=(1, 2)).view(batch, 1, 1, 1)
        else:
            stm = x.new_ones(batch, 1, 1, 1)
        coord = self.coord_features.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(
            batch, -1, -1, -1
        )
        rank01 = coord[:, 0:1]
        side_relative_rank = stm * rank01 + (1.0 - stm) * (1.0 - rank01)
        coord = torch.cat(
            [coord[:, 0:2], side_relative_rank, coord[:, 3:5]], dim=1
        )
        features = torch.cat([x, coord], dim=1)
        tokens = features.flatten(2).transpose(1, 2)  # (B, 64, C+coord)
        return self.norm(self.proj(tokens))


class _RoleGateBank(nn.Module):
    """Learned per-square role gates ``g_r(s) in [0, 1]``.

    A single ``Linear -> Sigmoid`` from token features to ``R`` logits.
    This makes the gates *learned* signals (not hard-coded legal-move
    labels) while keeping them cheap and interpretable per square.
    """

    def __init__(self, token_dim: int, role_count: int = ROLE_GATE_COUNT) -> None:
        super().__init__()
        self.role_count = int(role_count)
        self.proj = nn.Linear(int(token_dim), self.role_count)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(tokens))


class _TensorRingOrder(nn.Module):
    """One interaction order ``K`` of the tensor-ring contraction.

    Holds ``K`` independent core projections ``G_k : R^D -> R^{r x r}``
    and a learned ``(P, K, R)`` pattern bank that selects which role
    gate each slot of each pattern attends to.

    For each pattern ``p`` and slot ``k``,
    ``M_{p, k} = (1 / 64) sum_s alpha_{p, k, r} g_r(s) * G_k(x_s)``,
    and the contraction summary is ``z_p = trace(M_{p, 1} ... M_{p, K})``.
    """

    def __init__(
        self,
        order: int,
        token_dim: int,
        rank: int,
        num_patterns: int,
        role_count: int = ROLE_GATE_COUNT,
    ) -> None:
        super().__init__()
        if order < 2:
            raise ValueError("order must be >= 2 (tensor-ring needs >= 2 cores)")
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if num_patterns < 1:
            raise ValueError("num_patterns must be >= 1")
        self.order = int(order)
        self.token_dim = int(token_dim)
        self.rank = int(rank)
        self.num_patterns = int(num_patterns)
        self.role_count = int(role_count)
        self.cores = nn.ModuleList(
            [nn.Linear(self.token_dim, self.rank * self.rank) for _ in range(self.order)]
        )
        # (P, K, R) pattern logits; softmax over R selects soft role mix.
        self.pattern_logits = nn.Parameter(
            torch.randn(self.num_patterns, self.order, self.role_count) * 0.1
        )

    @property
    def pattern_weights(self) -> torch.Tensor:
        return torch.softmax(self.pattern_logits, dim=-1)

    def forward(
        self, tokens: torch.Tensor, role_gates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, square_count, _ = tokens.shape
        # alpha: (P, K, R) -> effective gate (B, P, K, S) = alpha * role_gates
        alpha = self.pattern_weights
        # role_gates: (B, S, R), alpha: (P, K, R)
        # eff_gate[b, p, k, s] = sum_r alpha[p, k, r] * role_gates[b, s, r]
        eff_gate = torch.einsum("pkr,bsr->bpks", alpha, role_gates)

        # core_outs: list of (B, S, r, r) for k in 0..K-1
        core_outputs: list[torch.Tensor] = []
        core_norms: list[torch.Tensor] = []
        for core in self.cores:
            raw = core(tokens)  # (B, S, r*r)
            mat = raw.view(batch, square_count, self.rank, self.rank)
            core_outputs.append(mat)
            core_norms.append(mat.detach().pow(2).mean(dim=(1, 2, 3)).sqrt())
        core_norm_stack = torch.stack(core_norms, dim=1)  # (B, K)

        # Per slot M_{p, k} = (1/S) sum_s eff_gate[b, p, k, s] * core_out_k[b, s]
        slot_matrices: list[torch.Tensor] = []
        for k, core_out in enumerate(core_outputs):
            slot_gate = eff_gate[:, :, k, :]  # (B, P, S)
            mat = torch.einsum("bps,bsij->bpij", slot_gate, core_out)
            mat = mat / float(square_count)
            slot_matrices.append(mat)

        # Cyclic product M_{p, 1} M_{p, 2} ... M_{p, K} -> (B, P, r, r)
        product = slot_matrices[0]
        for mat in slot_matrices[1:]:
            product = torch.matmul(product, mat)
        contractions = product.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (B, P)
        return contractions, core_norm_stack, alpha


class _CnnStemSummary(nn.Module):
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
            raise ValueError("cnn depth must be >= 1")
        layers: list[nn.Module] = []
        in_channels = int(input_channels)
        for _ in range(int(depth)):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    int(channels),
                    kernel_size=3,
                    padding=1,
                    bias=not use_batchnorm,
                )
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(int(channels)))
            else:
                layers.append(nn.GroupNorm(1, int(channels)))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = int(channels)
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.body(x)
        return self.pool(feat).flatten(1)


def _pattern_stats(values: torch.Tensor) -> torch.Tensor:
    """Mean / max / variance / signed-abs-mean across the pattern axis."""
    mean = values.mean(dim=-1)
    maximum = values.amax(dim=-1)
    variance = values.var(dim=-1, unbiased=False)
    abs_mean = values.abs().mean(dim=-1)
    signed_abs_mean = mean.sign() * abs_mean
    return torch.stack([mean, maximum, variance, signed_abs_mean], dim=-1)


class TensorRingSquareInteractionNetwork(nn.Module):
    """Bespoke tensor-ring square-interaction network for puzzle_binary."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        token_dim: int = 48,
        rank: int = 4,
        orders: tuple[int, ...] = (2, 3),
        num_patterns: int = 8,
        cnn_channels: int = 32,
        cnn_depth: int = 2,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        height: int = 8,
        width: int = 8,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "TensorRingSquareInteractionNetwork supports the puzzle_binary one-logit contract"
            )
        if not orders:
            raise ValueError("orders must contain at least one order >= 2")
        for order in orders:
            if int(order) < 2:
                raise ValueError("every order must be >= 2 (tensor-ring contracts at least 2 cores)")
        self.spec = BoardTensorSpec(
            input_channels=int(input_channels), height=int(height), width=int(width)
        )
        self.input_channels = int(input_channels)
        self.height = int(height)
        self.width = int(width)
        self.token_dim = int(token_dim)
        self.rank = int(rank)
        self.orders = tuple(int(order) for order in orders)
        self.num_patterns = int(num_patterns)
        self.cnn_channels = int(cnn_channels)
        self.cnn_depth = int(cnn_depth)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.role_count = ROLE_GATE_COUNT

        self.token_encoder = _SquareTokenEncoder(
            input_channels=self.input_channels,
            token_dim=self.token_dim,
            height=self.height,
            width=self.width,
        )
        self.role_gate_bank = _RoleGateBank(token_dim=self.token_dim, role_count=self.role_count)
        self.tensor_ring_orders = nn.ModuleDict(
            {
                str(order): _TensorRingOrder(
                    order=order,
                    token_dim=self.token_dim,
                    rank=self.rank,
                    num_patterns=self.num_patterns,
                    role_count=self.role_count,
                )
                for order in self.orders
            }
        )
        self.cnn_stem = _CnnStemSummary(
            input_channels=self.input_channels,
            channels=self.cnn_channels,
            depth=self.cnn_depth,
            dropout=self.dropout_p,
            use_batchnorm=bool(use_batchnorm),
        )

        # Per order: P raw contractions + 4 stats; total across orders.
        per_order_dim = self.num_patterns + 4
        contract_dim = per_order_dim * len(self.orders)
        head_input = contract_dim + self.cnn_stem.output_dim

        head_layers: list[nn.Module] = [
            nn.LayerNorm(head_input),
            nn.Linear(head_input, self.hidden_dim),
            nn.GELU(),
        ]
        if self.dropout_p > 0.0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.hidden_dim, 1))
        self.classifier = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]

        tokens = self.token_encoder(x)  # (B, 64, D)
        role_gates = self.role_gate_bank(tokens)  # (B, 64, R)
        cnn_summary = self.cnn_stem(x)  # (B, cnn_channels)

        contraction_features: list[torch.Tensor] = []
        per_order_contractions: dict[str, torch.Tensor] = {}
        per_order_stats: dict[str, torch.Tensor] = {}
        per_order_pattern_weights: dict[str, torch.Tensor] = {}
        per_order_core_norms: dict[str, torch.Tensor] = {}
        for order in self.orders:
            order_module: _TensorRingOrder = self.tensor_ring_orders[str(order)]
            contractions, core_norms, alpha = order_module(tokens, role_gates)
            stats = _pattern_stats(contractions)
            contraction_features.append(contractions)
            contraction_features.append(stats)
            per_order_contractions[f"contractions_order_{order}"] = contractions
            per_order_stats[f"contraction_stats_order_{order}"] = stats
            per_order_pattern_weights[f"pattern_weights_order_{order}"] = alpha.expand(
                batch, -1, -1, -1
            )
            per_order_core_norms[f"core_norms_order_{order}"] = core_norms

        contraction_vector = torch.cat(contraction_features, dim=-1)
        head_input = torch.cat([contraction_vector, cnn_summary], dim=-1)
        logits = self.classifier(head_input).view(-1)

        role_gate_activity = role_gates.mean(dim=1)  # (B, R)
        role_gate_entropy = -(
            role_gates.clamp(1.0e-6, 1.0 - 1.0e-6).log() * role_gates
            + (1.0 - role_gates).clamp(1.0e-6, 1.0 - 1.0e-6).log() * (1.0 - role_gates)
        ).mean(dim=(1, 2))

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "tokens": tokens,
            "role_gates": role_gates,
            "role_gate_activity": role_gate_activity,
            "role_gate_entropy": role_gate_entropy,
            "cnn_summary": cnn_summary,
            "contraction_features": contraction_vector,
        }
        output.update(per_order_contractions)
        output.update(per_order_stats)
        output.update(per_order_pattern_weights)
        output.update(per_order_core_norms)
        return output


def build_tensor_ring_square_interaction_network_from_config(
    config: dict[str, Any],
) -> TensorRingSquareInteractionNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    raw_orders = cfg.pop("orders", (2, 3))
    if isinstance(raw_orders, (list, tuple)):
        orders = tuple(int(value) for value in raw_orders)
    else:
        orders = (int(raw_orders),)
    return TensorRingSquareInteractionNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        token_dim=int(cfg.pop("token_dim", cfg.pop("channels", 48))),
        rank=int(cfg.pop("rank", 4)),
        orders=orders,
        num_patterns=int(cfg.pop("num_patterns", 8)),
        cnn_channels=int(cfg.pop("cnn_channels", 32)),
        cnn_depth=int(cfg.pop("cnn_depth", cfg.pop("depth", 2))),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        height=int(cfg.pop("height", 8)),
        width=int(cfg.pop("width", 8)),
    )
