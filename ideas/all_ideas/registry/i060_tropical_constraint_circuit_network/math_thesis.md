# Math Thesis

Tropical Constraint Circuit Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2046_friday_shanghai_tropical_circuit.md`.

## Working Thesis

Puzzle-like positions may be better modelled as the *near-satisfaction*
of a small number of latent tactical constraints. A min-plus tropical
circuit over learned current-board literal costs can test this OR-of-AND
structure directly: conjunction inside a monomial is additive cost, and
disjunction across monomials is a soft minimum.

## Setup

Let `c_l(x) >= 0` be a learned literal cost computed only from
current-tensor channels and fixed board geometry, where a literal `l`
indexes a `(channel, square)` slot of the literal-cost tensor. For each
clause `k` we define `M` monomials with nonnegative coefficients:

```
m_{k,j}(x) = b_{k,j} + sum_l a_{k,j,l} c_l(x),  a_{k,j,l} >= 0, b_{k,j} >= 0.
```

The clause's tropical (soft-min) cost at temperature `tau > 0` is

```
p_k(x) = -tau * log sum_j exp(-m_{k,j}(x) / tau).
```

As `tau -> 0`, `p_k` -> `min_j m_{k,j}(x)`. Low `p_k` means at least one
learned conjunction of literals is nearly satisfied. The model classifies
from clause soft-min costs, best-second margins, soft-min entropies, and
mean monomial costs.

## Variational View

`-tau * logsumexp(-m / tau)` is a smooth approximation of the minimum
whose gradient concentrates on the lowest-cost monomial:

```
d p_k / d m_{k,j} = q_{k,j},   q_{k,j} = softmax(-m_{k,j} / tau).
```

So the soft-min is a temperature-controlled differentiable existential
matcher. The packet's central falsifier replaces this min-plus reduction
with a sum-product (soft-average) reduction `sum_j q_{k,j} m_{k,j}` over
the same monomials; the parameter count and literal cost vector are
identical. If the falsifier matches the main model, min-plus
winner-take-most behaviour is not what carries the signal.

## What Is Actually Proven

- Each monomial `m_{k,j}` is monotone in literal costs (nonnegative
  coefficients).
- `p_k` is a smooth lower bound on `min_j m_{k,j}`; in particular,
  `p_k -> min_j m_{k,j}` as `tau -> 0` and `p_k -> mean_j m_{k,j}` as
  `tau -> infinity`.
- The `sum_product_clause` ablation removes the min-plus existential
  bottleneck while preserving literal costs and parameter count.
- The `literal_square_shuffle` ablation preserves per-channel material
  totals but destroys spatial structure, so any winning ablation here
  signals a material/count shortcut.

## What Remains Hypothesised

- That puzzle-likeness has learnable low-cost latent constraints in
  this representation rather than smooth positional fields.
- That the learned clauses do not collapse to dense generic features;
  the `effective_monomials_per_clause = exp(clause_entropy)` diagnostic
  reports whether soft-min remains peaked under training.

## Counterexamples

- Labels driven by broad positional shifts rather than sparse
  near-satisfied constraints.
- Tactics requiring engine search or legal move enumeration.
- Datasets where material and phase dominate the label.

## Self-Critique

Without sparsity pressure, clauses may use too many literals and become
generic MLP features. The first implementation enforces nonnegative
low-rank clause weights, exposes the
`effective_monomials_per_clause` diagnostic, and surfaces the
`sum_product_clause`, `mean_literal_pool`, `literal_square_shuffle`,
`high_temperature_softmin`, and `material_only_literals` falsifiers from
section 9 of the packet directly through the `model.ablation` config
flag.
