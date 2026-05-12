# Architecture

`Motif Tensor Factorization Network` is a board-only classifier for the
`puzzle_binary` task. It accepts the repository's `simple_18`
current-board tensor with shape `(B, 18, 8, 8)` and returns one
puzzle logit per position. The packet thesis is that puzzle signal is
a *multiplicative* relation among typed roles:

```
attacker x target x defender x line-relation x tempo
```

A plain CNN learns these implicitly. This network instead represents
each typed role candidate explicitly and scores their conjunction with
a low-rank CP factorization on a 4-way motif tensor.

## Mechanism

A compact convolutional trunk turns the 18-plane board into per-square
features `H ∈ R^{B×C×8×8}`. From the same trunk, three parallel 1x1
selector heads emit attacker / target / defender selection logits over
the 64 squares, and the top `P = top_candidates` (default 8) squares
per role are gathered as the role's candidate set.

Each candidate's per-square feature is then projected by a role-specific
two-layer MLP into a rank-`R` CP factor:

```
A_i = attacker_token(H[a_i])    # (B, P, R)
T_j = target_token(H[t_j])      # (B, P, R)
D_k = defender_token(H[d_k])    # (B, P, R)
```

The relation factor `R_ij` is computed by a small MLP that takes the
attacker and target factors plus a learned signed `(delta_rank,
delta_file)` embedding (15×15 entries indexed by `(da+7) * 15 + (df+7)`)
and projects to rank `R`. This captures same-rank/file/diagonal/
knight-jump style line relations between an attacker and a target
square.

The motif tensor is then the CP score of the four factors:

```
M[i, j, k] = sum_r A_r(i) * T_r(j) * D_r(k) * rel_r(i, j)
```

implemented as an `einsum("bijr,bkr->bijk", A_T_rel, D)` after the
elementwise product `A * T * rel`. Pooled motif features feed the
final puzzle head:

- `top_motif_scores`: the `top_motifs` (default 16) largest values in
  the flattened `(P, P, P)` motif tensor.
- `motif_entropy`: entropy of the softmax over the flattened motif
  tensor, low when the model has located a single conjunction.
- `own_motif_score`: mean of the top motif scores when the side-to-move
  plane is the actual position.
- `opponent_motif_score`: mean of the top motif scores after the
  side-to-move plane is flipped (re-trunked, so the perturbation is a
  faithful intervention rather than a no-op).
- `motif_contrast`: `own_motif_score - opponent_motif_score`.
- `near_disproof_score`: smallest per-leg magnitude of the top motif's
  CP factors. A small leg means the top motif is held up by one weak
  component — exactly the multiplicative-conjunction failure that
  separates puzzles from near-puzzles.

The puzzle logit is a `LayerNorm → Linear → GELU → Dropout → Linear`
head over the concatenated pooled features. The trainer uses BCE with
logits.

## Why multiplicative motifs

The packet calls out that "each part must be present" for tactics. A
multiplicative score collapses if any of the four CP legs is small —
this is the conjunction property that an additive baseline cannot
express. The `additive_motif_score` ablation makes this concrete by
swapping the `*` for `+` and observing that the additive form lets
strong but unrelated legs paper over a missing conjunction.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for the
repository `puzzle_binary` BCE-with-logits trainer (or
`(B, num_classes)` when `num_classes > 1`, with the puzzle scalar
written into the last column of a zero-padded tensor). All tensors are
finite per batch:

- `logits`: `(B,)` puzzle logit (or `(B, num_classes)` when
  `num_classes > 1`).
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `motif_score_tensor`: `(B, P, P, P)` full motif scores over the
  selected (attacker, target, defender) triples.
- `top_motif_scores`: `(B, top_motifs)` top motif values.
- `top_motif_indices`: `(B, top_motifs)` flattened indices in
  `(P, P, P)`.
- `motif_entropy`: `(B,)`.
- `own_motif_score`, `opponent_motif_score`, `motif_contrast`: `(B,)`.
- `near_disproof_score`: `(B,)`.
- `attacker_top_indices`, `target_top_indices`,
  `defender_top_indices`: `(B, P)` selected square indices per role.
- `trunk_features`: `(B, channels, 8, 8)`.
- `ablation_active`, `uses_multiplicative_motif`,
  `uses_relation_embedding`, `rank`, `num_top_candidates`,
  `num_top_motifs`: `(B,)` flags exposing the running ablation.

## Ablations

The packet's required ablations are exposed through the model:

- `"none"` — main multiplicative motif tensor.
- `"additive_motif_score"` (`ablation`) — replace
  `A * T * D * rel` with `A + T + D + rel`, killing the conjunction.
- `"no_relation_embedding"` (`ablation`) — set `rel` to ones so the
  line relation factor cannot contribute.
- `"rank_8_24_64"` — capacity sweep done by setting `rank` in the
  config; not a separate code path.

## Implementation Binding

- Registered model name: `motif_tensor_factorization_network`
- Source implementation file: `src/chess_nn_playground/models/motif_tensor_factorization_network.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i182_motif_tensor_factorization_network/model.py`
