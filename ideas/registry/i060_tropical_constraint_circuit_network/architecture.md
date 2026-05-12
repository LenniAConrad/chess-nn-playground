# Architecture

`Tropical Constraint Circuit Network` realises the markdown thesis as a
bespoke model: the central computation is a **differentiable min-plus
(tropical) circuit** over learned nonnegative current-board literal costs,
not a CNN, residual stack, attention block, sheaf, or move-delta
mechanism. Conjunction is additive cost across literals; disjunction is a
soft minimum across monomials.

## Pipeline

1. **`Simple18LiteralCostEncoder`** concatenates the simple_18 board
   tensor with four fixed coordinate planes (rank, file, diag, anti-diag)
   and applies a 1x1 convolution to `literal_channels` (default 32). A
   `softplus` activation enforces nonnegative literal costs

   ```
   c_l(x) = softplus(W * [board || coord])_l    (l = 1..literal_channels * 64)
   ```

   Costs are flattened to a `(B, L)` literal vector with `L = literal_channels * 64`.
2. **`TropicalClauseLayer`** computes monomial costs via low-rank
   nonnegative weights and nonnegative biases:

   ```
   a_{k,j,l} = softplus( sum_r U_{k,j,r} V_{r,l} ),     a >= 0
   b_{k,j}   = softplus( B_{k,j} ),                     b >= 0
   m_{k,j}(x) = b_{k,j} + sum_l a_{k,j,l} c_l(x)
   ```

   Default config has `K = clause_count = 24` clauses, `M = monomials_per_clause = 12`
   monomials per clause, and rank `r = clause_rank = 8`.
3. **`TropicalMarginPool`** reduces monomial costs to four clause
   statistics:

   ```
   p_k(x)   = -tau * logsumexp_j(-m_{k,j}(x) / tau)             # soft-min cost
   margin_k = m_{k,(2)} - m_{k,(1)}                              # second - smallest
   ent_k    = -sum_j q_{k,j} log q_{k,j}, q = softmax(-m / tau)  # softmin entropy
   mean_k   = mean_j m_{k,j}(x)
   ```

   The clause feature dimension is therefore `K * 4`.
4. **`TropicalConstraintHead`** is a `LayerNorm -> Linear -> ReLU -> Linear`
   MLP over the clause statistics concatenated with a 13-dimensional
   global broadcast vector (side-to-move scalar, four castling flags,
   eight-way en-passant file mask). It returns one puzzle logit; a
   symmetric `two_class_logits` diagnostic is produced by splitting the
   binary logit so reporting can use the binary contract.

## Section 9 Falsifier Ablations

The `ablation` config field selects between five falsifiers. All preserve
the head's input dimensionality (clause-stat layout) so capacity is
matched.

- `sum_product_clause`: replace softmin with a soft-average pooling
  `clause_value_k = sum_j w_j m_{k,j}` with `w = softmax(-m / tau)`. The
  monomial costs and clause weights are unchanged; only the min-plus
  winner-take-most logic is removed. This is the markdown's central
  falsifier.
- `mean_literal_pool`: bypass the clause layer entirely. A fixed (non-
  trainable) random projection sends the literal cost vector to a vector
  of the same dimensionality as the clause-stat output. Tests whether
  clauses contribute beyond literal summaries.
- `literal_square_shuffle`: apply a fixed deterministic permutation to
  the spatial axis of literal costs before clauses are computed. Channels
  and counts are preserved, so shortcut detection via material/count
  must survive without board geometry.
- `high_temperature_softmin`: scale the softmin temperature by a fixed
  factor (default 8) so softmin approaches averaging. Tests whether
  exact min-plus structure matters or any smooth pooling is enough.
- `material_only_literals`: replace each literal channel by its spatial
  mean broadcast back to all 64 squares. Keeps per-channel material
  totals but discards square-specific structure.

## Output Contract

`forward(x)` returns a dictionary including
`logits` of shape `(B,)` for `num_classes=1`, `two_class_logits`,
`monomial_costs`, `clause_softmin_cost`, `clause_value`, `clause_margin`,
`clause_entropy`, `clause_mean_cost`, `clause_softmin_probabilities`,
`effective_monomials_per_clause` (= `exp(clause_entropy)` in `[1, M]`),
`active_literal_mass`, `mechanism_energy`, and `ablation_active`. Engine,
verification, source, and CRTK metadata are never used as input.

## Why This Is Not A Generic CNN Variant

The central operator is a **min-plus tropical circuit**: clause output is
`-tau * logsumexp(-m / tau)` rather than convolution, residual stacking,
square attention, or move enumeration. As `tau -> 0`, the clause output
approaches the minimum monomial cost, giving differentiable existential
matching of "which conjunction of literals is nearly satisfied?". The
`sum_product_clause` ablation tests exactly this point by replacing the
softmin with the same softmax weighting applied as a *weighted mean* of
the same monomials, while keeping all other modules and parameter counts
unchanged.

## Implementation Binding

- Registered model name: `tropical_constraint_circuit_network`
- Source implementation: `src/chess_nn_playground/models/trunk/tropical_constraint_circuit_network.py`
- Idea-local wrapper: `ideas/registry/i060_tropical_constraint_circuit_network/model.py`
  exposes `build_model_from_config(config)` and delegates to
  `build_tropical_constraint_circuit_network_from_config`.
- The idea-local wrapper does not import or call the shared
  `ResearchPacketProbe` / `build_research_packet_probe_from_config` scaffold.
