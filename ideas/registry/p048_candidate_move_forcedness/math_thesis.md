# Math Thesis

Source: `ideas/research/primitives/external_43_candidate_move_forcedness_primitive.md`.

## Working thesis

Let a board tensor `x` (simple_18) induce:

- a frozen pseudo-legal move adjacency
  `A(x) in {0, 1}^{B x 64 x 64}` from `compute_legal_move_graph`,
- per-edge `move_type`, `ray_direction` integer codes,
- per-square own / enemy piece value scalars `v_us(x), v_them(x)`,
- per-square enemy king mask `K_them(x)`,
- per-square tokens `H(x) in R^{B x 64 x d}` from a 1x1 conv tower.

We define a 14-channel deterministic per-edge descriptor
`f(x)[b, i, j] in R^{14}` whose channels are:

```
f[0]  = v_us(i)                      # mover_value
f[1]  = v_them(j)                    # victim_value
f[2]  = 1[enemy occupies j]          # is_capture
f[3]  = 1[j == enemy king square]    # is_check_seed
f[4]  = 1[i->j is pawn move & j on enemy back rank]   # is_promotion_seed
f[5]  = clamp(deg_out(i) / 28, 0, 1) # source mobility (broadcast)
f[6]  = clamp(deg_in(j)  / 16, 0, 1) # target in-degree (broadcast)
f[7]  = 1[move_type == knight]
f[8]  = 1[move_type in {rank, file}]      # rook-like
f[9]  = 1[move_type in {diag, antidiag}]  # bishop-like
f[10] = 1[move_type == king]
f[11] = 1[move_type == pawn_push]
f[12] = 1[move_type == pawn_capture]
f[13] = max(v_them(j) - 0.5 * v_us(i), 0) * is_capture  # SEE-lite
```

All channels are multiplied by `A[b, i, j]` so they are zero on
inactive edges. Channels are stop-gradient (rule-derived).

We learn an edge scorer

```
score[b, i, j] = score_mlp(LayerNorm([src_tok, dst_tok, type_emb, f]))
```

with `type_emb = MoveTypeEmbed(move_type) + DirEmbed(ray_direction)`,
plus a temperature: `score := score / softmax_temperature`. Inactive
edges (`A == 0`) are masked to `-inf` so they never participate in
top-k.

The pool computes:

```
m            = flatten(A, 64*64)
s            = flatten(score, 64*64)
F            = flatten(f, 64*64)            # (B, 64*64, 14)
topk_idx     = topk(s, k=4)                 # over active edges
top1_score   = s_(1)
gap12        = s_(1) - s_(2)
topk_mass    = sum_{m in topk} softmax(s)[m]
entropy      = H(softmax(s on active edges))
top1_feat    = F[top1_idx]
topk_feat_mean = mean over top-k F
cat_max[c]   = max_{(i,j): A_{ij}=1} F[b, i, j, c]
move_count   = sum_{(i,j)} A_{ij}
```

The pool vector concatenates the 5 scalars and the three 14-channel
feature blocks (5 + 3 * 14 = 47 dims), is LayerNormed, then fed
into a delta MLP together with the i193 joint pool. A small gate MLP
over `cat(joint, top1_score, gap12, entropy)` produces a sigmoid
`primitive_gate`, and the final logit is

```
final_logit = base_logit + primitive_gate * primitive_delta_raw.
```

## Why this matters

The i018 / i193 trunk reads off pooled relation energies and triad
defects but it never explicitly considers *which move* a player
should consider. Tactical puzzles often hinge on the presence of a
single coercive move (check, capture, promotion, fork, mate-in-1).
CMF asks: does scoring candidate moves with a small learned MLP over
their deterministic forcedness descriptors, then pooling the top-k,
add discriminative signal beyond what the trunk already extracts?
The pool is biased towards "one or a few unusually coercive
candidates", so the expected lifting slices are the forcing-line
tactics (mate_in_1, hanging, fork, overload), promotion, and
near-puzzle rejection.

## What is actually proven

- **Baseline recovery**. `zero_delta` / `trunk_only` zeroes the
  delta; the model returns exactly `base_logit`.
- **Mask validity**. Inactive edges (`A == 0`) are masked to `-inf`
  before top-k, so they never enter the pool's score-based summaries.
  Feature pools use the mask explicitly.
- **Topology constraint**. CMF uses the pseudo-legal adjacency from
  `compute_legal_move_graph`; `dense_edges` flips this to all-pairs
  to test whether legality matters.

## What is only hypothesized

That a learned per-move score over deterministic forcedness
features carries discriminative signal not already captured by the
trunk, that the lift survives `deterministic_score`, and that
top-k pooling beats mean pooling.

## Failure cases

1. *Learned scorer redundant*: tested by `deterministic_score`. If
   replacing the MLP score with the per-edge feature sum matches the
   unablated run, the MLP is not load-bearing.
2. *Top-k concentration irrelevant*: tested by `mean_pool`. If
   averaging over all candidates matches top-k, candidate
   concentration carries no signal.
3. *Deeper features irrelevant*: tested by `flags_only`. If keeping
   only move-class flags matches the full model, piece values and
   mobility do not earn their cost.
4. *Legality irrelevant*: tested by `dense_edges`. If a fully-
   connected mask matches the pseudo-legal mask, the move-graph
   constraint is not load-bearing.
5. *Consequence features irrelevant*: tested by `no_consequence`.

## Falsifier

- `deterministic_score` -- primary. Replace per-move learned score
  with feature sum.
- `mean_pool` -- anti-top-k.
- `flags_only` -- anti-deep-features.
- `dense_edges` -- anti-legal-mask.
- `no_consequence` -- anti-check/capture/promotion.
