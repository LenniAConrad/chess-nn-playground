# Math Thesis

Attention Disagreement Residual Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md`.

Batch candidate rank: `2`.

## Working Thesis

Near-puzzle and puzzle-like positions may contain competing tactical
interpretations. Independent attention query families that share a common
board encoder should disagree more on ambiguous or tactically dense boards
than on quiet boards. The classifier therefore decides puzzle-likeness from
the residual disagreement among `F` family-level attention distributions,
not from any single attended value.

## Formulation

Let `T = (t_1, ..., t_64) in R^{64xD}` be the square tokens produced by the
shared encoder for one board. Let `Q_f in R^{QxD}` be the query bank for
family `f in {1, ..., F}`, with shared projections `W_q, W_k, W_v in R^{DxD}`.
For each family and query the attention distribution over squares is

```
A_{f,i,n} = softmax_n( <W_q q_{f,i}, W_k t_n> / sqrt(D) ),
attended_{f,i} = sum_n A_{f,i,n} (W_v t_n).
```

The model takes the family-averaged attention map
`bar A_f = (1/Q) sum_i A_{f,i,.}` and forms three disagreement signals:

1. **Distributional disagreement.** Pairwise Jensen-Shannon divergence
   `JS(bar A_f, bar A_g)` for `f < g`, averaged and maxed across pairs.
2. **Geometric disagreement.** Cosine distance `1 - cos(bar A_f, bar A_g)`
   on the attention simplex, plus the maximum cross-family cosine distance
   among the `F*Q` per-query attention vectors.
3. **Routing-stability disagreement.** Per-family normalised attention
   entropy `H_f = -mean_i sum_n A_{f,i,n} log A_{f,i,n} / log 64`, with the
   across-family variance `Var_f H_f` as the routing-instability score.

Stacked with the across-family mean of attended values and the family-wise
residual standard deviation of attended values, these signals form the
input to a small MLP that returns one puzzle logit. The architecture only
ever sees the board tensor; CRTK/source/engine metadata is reporting-only.

## Falsifiable Predictions

- Replacing the `F` query banks with a single bank of `F*Q` queries (the
  matched-parameter control) should not match ADRN on near-puzzle slices if
  the disagreement signal carries genuine evidence.
- Forcing all families to share a query bank (`shared_query_bank` ablation)
  should collapse JS divergence and cosine distance toward zero and remove
  the residual signal, hurting near-puzzle calibration.
- Permuting family attention maps across samples (`random_family_permutation`)
  should remove sample-specific disagreement and degrade performance if the
  disagreement is truly position-conditional.
