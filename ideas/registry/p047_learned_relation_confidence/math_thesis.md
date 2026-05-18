# Math Thesis

Source: `ideas/research/primitives/external_42_learned_relation_confidence_primitive.md`.

## Working thesis

Let a board tensor `x` (simple_18) induce:

- frozen 12-relation deterministic masks `M(x) in {0, 1}^{B x R x 64 x 64}`
  (from `TacticalIncidenceBuilder`),
- per-square tokens `H(x) in R^{B x 64 x d}` from a 1x1 conv tower,
- per-square piece descriptors `p(x) in R^{B x 64 x 13}` (empty + 12
  piece planes).

We learn an edge scorer `s` such that

    score[b, r, i, j] = (src_mlp(p)[b, i, r] + tgt_mlp(p)[b, j, r]
                        + low_rank[b, r, i, j] + rel_bias[r])
                        * sigmoid(rel_gate[r])

with

    low_rank[b, r, i, j] = sum_k (q[b, i, k_r] * rel_emb[r, k] * k[b, j, k_r])

and

    q, k = Linear_q(H), Linear_k(H)  in R^{B x 64 x (R * low_rank_dim)}.

Confidence is sigmoidal:

    confidence[b, r, i, j] = sigmoid(score / temperature)

and the weighted mask multiplies by the deterministic mask so the
support is preserved:

    weighted_mask[b, r, i, j] = M[b, r, i, j] * confidence[b, r, i, j].

Per-(batch, relation) summaries are pooled from the weighted mask:

    mean_conf[b, r]   = sum_{ij} weighted_mask / max(1, sum_{ij} M)
    mass[b, r]        = sum_{ij} weighted_mask / 4096
    kept_frac[b, r]   = sum_{ij} sigmoid((confidence - 0.5) / 0.1) * M / max(1, sum M)
    entropy[b, r]     = (-p log p - (1-p) log (1-p)) averaged over active edges

The (B, R, 4) summary is LayerNormed and concatenated with the i193
joint pool. A small MLP produces `primitive_delta_raw`; a sigmoid gate
over the joint pool plus (mean_conf, mask_density) modulates it:

    final_logit = base_logit + primitive_gate * primitive_delta_raw.

## Why this matters

i018's central falsifier (degree-preserving scrambled relation masks)
drops mean test PR-AUC from 0.8752 to 0.8328. That is strong evidence
that exact chess relation topology matters. p047 keeps the topology
fixed and asks the complementary question: can a learned per-edge
*weight* on top of that topology lift the baseline further? The
operator is structurally topology-preserving: zeros in `M` stay zero in
`weighted_mask`, so the load-bearing claim isolates to the
confidence layer.

## What is actually proven

- **Topology preservation**. By construction, `weighted_mask[r, i, j] =
  M[r, i, j] * sigmoid(...)`. If `M = 0`, `weighted_mask = 0` exactly.
- **Baseline recovery**. `zero_delta` / `trunk_only` zeroes the delta;
  `binary_only` skips the confidence and recovers the raw mask in the
  summary path.
- **Permutation symmetry of summaries**. Per-(batch, relation) summary
  pooling is symmetric over `(i, j)`; permuting batch elements does not
  change per-sample statistics. (Token-shuffle invariance is not
  trivial because the q/k projections are token-indexed; we do not
  claim it.)

## What is only hypothesized

That per-edge confidence carries chess-specific discriminative signal
not already encoded by the i193 trunk and that the lift survives the
`gate_only` ablation (i.e. coarse per-relation reweighting is not
enough).

## Failure cases

1. *Confidence collapse to a per-relation scalar*: tested by
   `gate_only`. If `gate_only` matches `none`, the per-edge structure
   is not load-bearing.
2. *Topology semantics irrelevant*: tested by `scrambled_mask`. If
   permuting the batch dimension of `M` leaves the lift intact, the
   primitive is not exploiting position-aligned relation structure.
3. *Feature semantics irrelevant*: tested by `shuffle_pieces`. If
   permuting the per-square piece descriptor across the batch leaves
   the lift intact, the edge MLP is not earning its feature input.
4. *Edge MLP redundant with low rank*: tested by `no_edge_mlp` /
   `no_low_rank`.

## Falsifier

- `binary_only` -- primary. Skip the confidence layer; summary inputs
  reduce to raw deterministic mask statistics. Lift must drop materially
  versus `none`.
- `scrambled_mask` -- rule-feature falsifier on the topology side.
- `shuffle_pieces` -- rule-feature falsifier on the feature side.
- `gate_only` -- coarse-vs-edge falsifier; tests whether per-edge
  structure beats per-relation rescaling.
