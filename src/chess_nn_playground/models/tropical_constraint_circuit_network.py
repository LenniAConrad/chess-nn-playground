"""Tropical Constraint Circuit Network model for idea i060.

Implements the markdown thesis (`ideas/all_ideas/registry/i060_tropical_constraint_circuit_network/`):
puzzle-likeness is tested as the *near-satisfaction* of a small number of latent
tactical constraints. The central computation is a differentiable min-plus
(tropical) circuit over learned nonnegative current-board literal costs. For a
clause ``k`` with ``M`` monomials,

    m_{k,j}(x) = b_{k,j} + sum_l a_{k,j,l} * c_l(x),    a_{k,j,l} >= 0
    p_k(x)    = -tau * logsumexp(-m_{k,j}(x) / tau, dim=j)

so ``p_k`` is a soft minimum over monomial costs (conjunction of literal costs
inside each monomial, disjunction across monomials). Low ``p_k`` means at least
one learned conjunction of literals is nearly satisfied. Clause-level
diagnostics fed to the head are the soft-min cost, the best/second-best
margin, the softmin entropy, and the mean monomial cost.

Forward pipeline:

    Simple18LiteralCostEncoder    ->  (B, L) nonnegative literal costs
    TropicalClauseLayer           ->  (B, K, M) monomial costs
    TropicalMarginPool            ->  (B, K * 4) clause statistics
    TropicalConstraintHead        ->  (B, num_classes) puzzle logits + diagnostics

Section 9 of the markdown packet identifies five ablations exposed via
``ablation``:

    * ``"none"``                       -- main model.
    * ``"sum_product_clause"``         -- replace softmin with a soft-average
      / sum-product pooling over monomials. Same literal costs and weights;
      removes the min-plus winner-take-most logic.
    * ``"mean_literal_pool"``          -- pool literal costs directly into the
      head (no clauses), with matched parameter count via a literal->K linear
      pre-projection used as a stand-in for the clause statistics.
    * ``"literal_square_shuffle"``     -- apply a fixed random permutation to
      the spatial squares of literal costs before clause weighting; channels
      and counts are preserved.
    * ``"high_temperature_softmin"``   -- raise ``softmin_temperature`` by a
      fixed multiplier (default 8x) so softmin approaches averaging.
    * ``"material_only_literals"``     -- mask the literal cost tensor to
      keep only material/per-channel aggregate literals (mean-pooled per
      piece-color channel and broadcast back to all 64 squares), discarding
      square-specific structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models._packet_bespoke_base import (
    BoardTensorSpec,
    require_board_tensor,
)


_BOARD = 8
_BOARD_AREA = _BOARD * _BOARD  # 64
_PIECE_PLANES = 12
_DEFAULT_LITERAL_CHANNELS = 32
_DEFAULT_CLAUSE_COUNT = 24
_DEFAULT_MONOMIALS = 12
_DEFAULT_RANK = 8
_DEFAULT_SOFTMIN_TEMPERATURE = 0.25
_DEFAULT_HIGH_TEMP_FACTOR = 8.0
_VALID_ABLATIONS = {
    "none",
    "sum_product_clause",
    "mean_literal_pool",
    "literal_square_shuffle",
    "high_temperature_softmin",
    "material_only_literals",
}


def _coordinate_planes(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a (B, 4, 8, 8) tensor of deterministic rank/file/diag coordinates.

    Channels:
        0: rank (row index / 7), 1: file (col index / 7),
        2: diag (rank + file - 7) / 7, 3: anti-diag (rank - file) / 7.
    These are fixed board geometry; they are not learned and are constant
    across the dataset, so they cannot leak engine/source metadata.
    """
    rank = torch.arange(_BOARD, device=device, dtype=dtype).view(1, 1, _BOARD, 1).expand(batch, 1, _BOARD, _BOARD) / 7.0
    file = torch.arange(_BOARD, device=device, dtype=dtype).view(1, 1, 1, _BOARD).expand(batch, 1, _BOARD, _BOARD) / 7.0
    diag = (rank * 7.0 + file * 7.0 - 7.0) / 7.0
    anti = (rank * 7.0 - file * 7.0) / 7.0
    return torch.cat([rank, file, diag, anti], dim=1)


def _build_square_permutation(num_squares: int = _BOARD_AREA, seed: int = 0xC0DE) -> torch.Tensor:
    """Return a fixed deterministic permutation of square indices."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(num_squares, generator=gen)


@dataclass(frozen=True)
class TropicalGlobals:
    side_to_move_white: torch.Tensor  # (B,)
    castling: torch.Tensor             # (B, 4)
    en_passant_file: torch.Tensor      # (B, 8)


class Simple18LiteralCostEncoder(nn.Module):
    """Encode the simple_18 board tensor (plus fixed coord planes) into nonnegative literal costs.

    Flow:
        (B, C, 8, 8) input  ->  concat with 4 coord planes
                             ->  1x1 Conv to ``literal_channels``
                             ->  softplus
                             ->  flatten to (B, literal_channels * 64)

    Costs are nonnegative because the tropical clause weights are also
    nonnegative; this preserves monotonicity in literals and matches the
    "near-satisfaction" semantics from the markdown.
    """

    def __init__(
        self,
        input_channels: int = 18,
        literal_channels: int = _DEFAULT_LITERAL_CHANNELS,
        coord_channels: int = 4,
    ) -> None:
        super().__init__()
        if input_channels < 13:
            raise ValueError(
                f"Simple18LiteralCostEncoder requires at least 13 input channels (simple_18-style), got {input_channels}"
            )
        if literal_channels < 1:
            raise ValueError("literal_channels must be >= 1")
        self.input_channels = int(input_channels)
        self.literal_channels = int(literal_channels)
        self.coord_channels = int(coord_channels)
        self.conv = nn.Conv2d(
            self.input_channels + self.coord_channels,
            self.literal_channels,
            kernel_size=1,
            bias=True,
        )
        self.spec = BoardTensorSpec(input_channels=self.input_channels)

    @property
    def literal_dim(self) -> int:
        return self.literal_channels * _BOARD_AREA

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        require_board_tensor(x, self.spec)
        coords = _coordinate_planes(x.shape[0], x.device, x.dtype)
        augmented = torch.cat([x, coords], dim=1)
        raw = self.conv(augmented)               # (B, literal_channels, 8, 8)
        costs = F.softplus(raw)                  # nonnegative
        return costs


class TropicalClauseLayer(nn.Module):
    """Low-rank nonnegative tropical clauses over a flat literal-cost vector.

    Implements ``a_{k,j,l} = softplus(U_{k,j,r} V_{r,l})`` with rank ``r`` so
    weights are nonnegative. Bias ``b_{k,j} >= 0`` via softplus on a free
    parameter. The forward returns the per-monomial cost tensor

        m_{k,j}(x) = softplus(b_{k,j}) + sum_l softplus(U V)_{k,j,l} * c_l(x),

    of shape ``(B, K, M)``. The pure ``aggregate`` operation (softmin /
    sum-product / mean) is applied by the caller so the same monomial tensor
    can drive the main model and ablations from a single forward.
    """

    def __init__(
        self,
        literal_dim: int,
        clause_count: int = _DEFAULT_CLAUSE_COUNT,
        monomials_per_clause: int = _DEFAULT_MONOMIALS,
        clause_rank: int = _DEFAULT_RANK,
    ) -> None:
        super().__init__()
        if literal_dim < 1:
            raise ValueError("literal_dim must be >= 1")
        if clause_count < 1:
            raise ValueError("clause_count must be >= 1")
        if monomials_per_clause < 2:
            raise ValueError("monomials_per_clause must be >= 2 to define a softmin")
        if clause_rank < 1:
            raise ValueError("clause_rank must be >= 1")
        self.literal_dim = int(literal_dim)
        self.clause_count = int(clause_count)
        self.monomials_per_clause = int(monomials_per_clause)
        self.clause_rank = int(clause_rank)

        # U: (K, M, R), V: (R, L). a_{k,j,l} = softplus(sum_r U_{k,j,r} V_{r,l}).
        u_init = torch.empty(self.clause_count, self.monomials_per_clause, self.clause_rank)
        v_init = torch.empty(self.clause_rank, self.literal_dim)
        nn.init.normal_(u_init, mean=-1.0, std=0.5)
        nn.init.normal_(v_init, mean=-1.0, std=0.5)
        self.weight_u = nn.Parameter(u_init)
        self.weight_v = nn.Parameter(v_init)
        self.bias_raw = nn.Parameter(torch.zeros(self.clause_count, self.monomials_per_clause))

    def clause_weights(self) -> torch.Tensor:
        """Return ``(K, M, L)`` nonnegative clause weights."""
        product = torch.einsum("kmr,rl->kml", self.weight_u, self.weight_v)
        return F.softplus(product)

    def forward(self, literal_costs_flat: torch.Tensor) -> torch.Tensor:
        # literal_costs_flat: (B, L) nonnegative
        weights = self.clause_weights()                                  # (K, M, L)
        bias = F.softplus(self.bias_raw)                                 # (K, M)
        weighted = torch.einsum("bl,kml->bkm", literal_costs_flat, weights)
        return bias.unsqueeze(0) + weighted                              # (B, K, M)


class TropicalMarginPool(nn.Module):
    """Pool monomial costs into clause-level statistics.

    Per clause ``k`` the pool emits ``S = 4`` features:
        0  soft-min cost (or aggregated cost for ablations)
        1  best-second margin between top-2 lowest monomials
        2  softmin entropy: -sum_j p_j log p_j with p_j ~ exp(-m_j / tau)
        3  mean monomial cost

    Aggregation is selected per-call so the same monomial tensor can be
    re-used for the central sum_product ablation.
    """

    def __init__(self, num_clauses: int) -> None:
        super().__init__()
        self.num_clauses = int(num_clauses)
        self.stats_per_clause = 4

    @property
    def feature_dim(self) -> int:
        return self.num_clauses * self.stats_per_clause

    def forward(
        self,
        monomial_costs: torch.Tensor,           # (B, K, M)
        temperature: torch.Tensor,              # scalar
        aggregator: str = "softmin",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Soft entropy uses softmax over -m / tau regardless of aggregator so
        # the diagnostic remains comparable across ablations.
        scaled = -monomial_costs / temperature.clamp_min(1.0e-6)
        log_probs = F.log_softmax(scaled, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)                       # (B, K)

        # Soft-min cost via stable LogSumExp identity.
        softmin_cost = -temperature * torch.logsumexp(scaled, dim=-1)    # (B, K)

        # Best-second margin: difference between the second-smallest and
        # smallest monomial cost. A larger margin means more peaked
        # near-winner structure.
        sorted_vals, _ = torch.sort(monomial_costs, dim=-1)
        margin = sorted_vals[..., 1] - sorted_vals[..., 0]               # (B, K)

        mean_cost = monomial_costs.mean(dim=-1)                          # (B, K)

        if aggregator == "softmin":
            clause_value = softmin_cost
        elif aggregator == "sum_product":
            # Soft-average / log-mean-exp surrogate that uses the same monomial
            # tensor; falsifier from section 9. We use the temperature-scaled
            # weighted mean: sum_j w_j * m_j with w = softmax(-m / tau).
            # This destroys winner-take-most behaviour while keeping the same
            # parameter footprint.
            clause_value = (probs * monomial_costs).sum(dim=-1)
        elif aggregator == "mean":
            # High-temperature limit equivalent: just the average.
            clause_value = mean_cost
        else:
            raise ValueError(f"Unsupported clause aggregator {aggregator!r}")

        clause_stats = torch.stack(
            [clause_value, margin, entropy, mean_cost], dim=-1
        )                                                                 # (B, K, 4)
        diagnostics: dict[str, torch.Tensor] = {
            "clause_softmin_cost": softmin_cost,
            "clause_value": clause_value,
            "clause_margin": margin,
            "clause_entropy": entropy,
            "clause_mean_cost": mean_cost,
            "clause_softmin_probabilities": probs,
        }
        flat = clause_stats.reshape(clause_stats.shape[0], -1)
        return flat, diagnostics


class TropicalConstraintHead(nn.Module):
    """Two-layer MLP head consuming clause statistics and a small global vector."""

    def __init__(
        self,
        feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        in_dim = int(feature_dim + global_feature_dim)
        self.in_dim = in_dim
        layers: list[nn.Module] = [
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, max(1, int(num_classes))))
        self.mlp = nn.Sequential(*layers)
        self.num_classes = int(num_classes)

    def forward(self, features: torch.Tensor, globals_tensor: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([features, globals_tensor], dim=-1))


class TropicalConstraintCircuitNet(nn.Module):
    """Complete bespoke architecture for idea i060.

    The central operator is a min-plus tropical circuit over nonnegative
    current-board literal costs. The encoder, clause weights, and head are
    learned; the ``softmin`` aggregator and the literal-square permutation
    used by the ``literal_square_shuffle`` ablation are fixed.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        literal_channels: int = _DEFAULT_LITERAL_CHANNELS,
        clause_count: int = _DEFAULT_CLAUSE_COUNT,
        monomials_per_clause: int = _DEFAULT_MONOMIALS,
        clause_rank: int = _DEFAULT_RANK,
        softmin_temperature: float = _DEFAULT_SOFTMIN_TEMPERATURE,
        head_hidden: int = 128,
        ablation: str = "none",
        dropout: float = 0.0,
        high_temperature_factor: float = _DEFAULT_HIGH_TEMP_FACTOR,
    ) -> None:
        super().__init__()
        ablation = (ablation or "none").lower()
        if ablation not in _VALID_ABLATIONS:
            raise ValueError(
                f"Unsupported tropical ablation {ablation!r}; expected one of {sorted(_VALID_ABLATIONS)}"
            )
        if softmin_temperature <= 0:
            raise ValueError("softmin_temperature must be > 0")

        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.literal_channels = int(literal_channels)
        self.clause_count = int(clause_count)
        self.monomials_per_clause = int(monomials_per_clause)
        self.clause_rank = int(clause_rank)
        self.ablation = ablation
        self.high_temperature_factor = float(high_temperature_factor)

        self.encoder = Simple18LiteralCostEncoder(
            input_channels=self.input_channels,
            literal_channels=self.literal_channels,
        )
        self.clause_layer = TropicalClauseLayer(
            literal_dim=self.encoder.literal_dim,
            clause_count=self.clause_count,
            monomials_per_clause=self.monomials_per_clause,
            clause_rank=self.clause_rank,
        )
        self.margin_pool = TropicalMarginPool(num_clauses=self.clause_count)

        self.global_feature_dim = 1 + 4 + 8  # side-to-move + castling + ep-file
        self.feature_dim = self.margin_pool.feature_dim
        self.head = TropicalConstraintHead(
            feature_dim=self.feature_dim,
            global_feature_dim=self.global_feature_dim,
            hidden_dim=int(head_hidden),
            num_classes=self.num_classes,
            dropout=float(dropout),
        )

        # Effective softmin temperature; high-temperature ablation scales it up.
        if ablation == "high_temperature_softmin":
            effective_tau = float(softmin_temperature) * self.high_temperature_factor
        else:
            effective_tau = float(softmin_temperature)
        self.register_buffer(
            "softmin_temperature",
            torch.tensor(float(softmin_temperature), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "effective_temperature",
            torch.tensor(effective_tau, dtype=torch.float32),
            persistent=True,
        )

        # Fixed deterministic permutation for the literal_square_shuffle ablation.
        self.register_buffer(
            "square_permutation",
            _build_square_permutation(_BOARD_AREA, seed=0xC0DE).to(dtype=torch.long),
            persistent=False,
        )

        # mean_literal_pool ablation needs a literal->K shrinkage so the head
        # input dimensionality matches the main model. We use a fixed (non-
        # trainable) random projection so the ablation cannot regain the
        # tropical signal through a learned linear stand-in.
        gen = torch.Generator(device="cpu").manual_seed(0x70AD)
        proj = torch.randn(
            self.encoder.literal_dim,
            self.clause_count * self.margin_pool.stats_per_clause,
            generator=gen,
        ) / float(self.encoder.literal_dim) ** 0.5
        self.register_buffer("mean_pool_projection", proj, persistent=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tropical_globals(self, x: torch.Tensor) -> TropicalGlobals:
        # side-to-move plane (channel 12 in simple_18) -> scalar 1 if white-to-move
        side_plane = x[:, 12].clamp(0.0, 1.0)
        side_white = (side_plane.mean(dim=(-1, -2)) > 0.5).to(x.dtype)  # (B,)
        castling = torch.stack(
            [
                x[:, 13].mean(dim=(-1, -2)),
                x[:, 14].mean(dim=(-1, -2)),
                x[:, 15].mean(dim=(-1, -2)),
                x[:, 16].mean(dim=(-1, -2)),
            ],
            dim=-1,
        ).clamp(0.0, 1.0)
        ep_plane = x[:, 17].clamp(0.0, 1.0)
        ep_files = ep_plane.amax(dim=-2)
        return TropicalGlobals(
            side_to_move_white=side_white,
            castling=castling,
            en_passant_file=ep_files,
        )

    def _apply_literal_ablation(self, costs: torch.Tensor) -> torch.Tensor:
        """Apply the literal-level ablations that change the literal cost field.

        ``costs`` has shape ``(B, literal_channels, 8, 8)``. The square-shuffle
        ablation permutes the spatial axis but preserves channel counts; the
        material-only ablation replaces each per-channel field by its spatial
        mean (broadcast back to all 64 squares), which keeps the global
        material tally per literal channel but discards square-specific
        structure.
        """
        if self.ablation == "literal_square_shuffle":
            batch, lc, _, _ = costs.shape
            flat = costs.reshape(batch, lc, _BOARD_AREA)
            shuffled = flat[..., self.square_permutation]
            return shuffled.reshape(batch, lc, _BOARD, _BOARD)
        if self.ablation == "material_only_literals":
            mean = costs.mean(dim=(-1, -2), keepdim=True)
            return mean.expand_as(costs)
        return costs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        require_board_tensor(x, BoardTensorSpec(input_channels=self.input_channels))
        x = x.float()
        globals_data = self._tropical_globals(x)

        costs = self.encoder(x)                                  # (B, literal_channels, 8, 8)
        costs = self._apply_literal_ablation(costs)
        flat_literals = costs.reshape(costs.shape[0], -1)        # (B, literal_dim)

        if self.ablation == "mean_literal_pool":
            # The clause head is bypassed; a fixed random projection produces
            # a feature vector of the exact same dimensionality as the
            # clause-statistic vector, so head capacity is matched.
            clause_features = flat_literals @ self.mean_pool_projection
            clause_diagnostics: dict[str, torch.Tensor] = {
                "clause_softmin_cost": flat_literals.new_zeros(
                    flat_literals.shape[0], self.clause_count
                ),
                "clause_value": flat_literals.new_zeros(flat_literals.shape[0], self.clause_count),
                "clause_margin": flat_literals.new_zeros(flat_literals.shape[0], self.clause_count),
                "clause_entropy": flat_literals.new_zeros(flat_literals.shape[0], self.clause_count),
                "clause_mean_cost": flat_literals.new_zeros(flat_literals.shape[0], self.clause_count),
                "clause_softmin_probabilities": flat_literals.new_zeros(
                    flat_literals.shape[0], self.clause_count, self.monomials_per_clause
                ),
            }
            monomial_costs = flat_literals.new_zeros(
                flat_literals.shape[0], self.clause_count, self.monomials_per_clause
            )
        else:
            monomial_costs = self.clause_layer(flat_literals)        # (B, K, M)
            aggregator = "sum_product" if self.ablation == "sum_product_clause" else "softmin"
            clause_features, clause_diagnostics = self.margin_pool(
                monomial_costs,
                temperature=self.effective_temperature,
                aggregator=aggregator,
            )

        global_features = torch.cat(
            [
                globals_data.side_to_move_white.unsqueeze(-1),
                globals_data.castling,
                globals_data.en_passant_file,
            ],
            dim=-1,
        )

        raw_logits = self.head(clause_features, global_features)
        if self.num_classes == 1:
            logits = raw_logits.view(-1)
            two_class = torch.stack([-0.5 * logits, 0.5 * logits], dim=-1)
        else:
            logits = raw_logits
            two_class = raw_logits if raw_logits.shape[-1] >= 2 else logits

        # Effective monomial count: exp(entropy), in [1, M]. Larger -> denser
        # clauses; smaller -> sparse near-winner structure.
        effective_monomials = clause_diagnostics["clause_entropy"].exp()

        active_literal_mass = flat_literals.mean(dim=-1)
        clause_softmin = clause_diagnostics["clause_softmin_cost"]
        clause_margin = clause_diagnostics["clause_margin"]
        clause_entropy = clause_diagnostics["clause_entropy"]
        mechanism_energy = clause_softmin.mean(dim=-1)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "two_class_logits": two_class,
            "monomial_costs": monomial_costs,
            "clause_softmin_cost": clause_softmin,
            "clause_value": clause_diagnostics["clause_value"],
            "clause_margin": clause_margin,
            "clause_entropy": clause_entropy,
            "clause_mean_cost": clause_diagnostics["clause_mean_cost"],
            "clause_softmin_probabilities": clause_diagnostics["clause_softmin_probabilities"],
            "effective_monomials_per_clause": effective_monomials,
            "active_literal_mass": active_literal_mass,
            "mechanism_energy": mechanism_energy,
            "ablation_active": torch.full(
                (logits.shape[0],),
                1.0 if self.ablation != "none" else 0.0,
                device=logits.device,
                dtype=logits.dtype,
            ),
        }
        return diagnostics


def build_tropical_constraint_circuit_network_from_config(
    config: dict[str, Any],
) -> TropicalConstraintCircuitNet:
    cfg = dict(config)
    head_hidden = cfg.get("head_hidden", cfg.get("hidden_dim", 128))
    return TropicalConstraintCircuitNet(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        literal_channels=int(cfg.get("literal_channels", _DEFAULT_LITERAL_CHANNELS)),
        clause_count=int(cfg.get("clause_count", _DEFAULT_CLAUSE_COUNT)),
        monomials_per_clause=int(cfg.get("monomials_per_clause", _DEFAULT_MONOMIALS)),
        clause_rank=int(cfg.get("clause_rank", _DEFAULT_RANK)),
        softmin_temperature=float(cfg.get("softmin_temperature", _DEFAULT_SOFTMIN_TEMPERATURE)),
        head_hidden=int(head_hidden),
        ablation=str(cfg.get("ablation", "none")),
        dropout=float(cfg.get("dropout", 0.0)),
        high_temperature_factor=float(cfg.get("high_temperature_factor", _DEFAULT_HIGH_TEMP_FACTOR)),
    )
