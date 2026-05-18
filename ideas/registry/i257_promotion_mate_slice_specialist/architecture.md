# Architecture

`Promotion Mate Slice Specialist` (i257) targets the three persistently weak
benchmark slices the matched-recall report flags: `promotion`,
`underpromotion`, and `mate_in_1`. The architecture keeps the parent puzzle
decision boundary intact and only adds *bounded gated logit deltas* on top of
a base trunk logit:

```text
final_logit = base_logit + sum_k gate_k * delta_k
```

where the specialist branches ``k`` cover (a) promotion fanout over
``{Q, R, B, N}`` type-conditioned descriptors, (b) underpromotion divergence
that scores the non-queen margin against the queen score, (c) king-zone
forcing-witness pressure derived from the deterministic king-feature stack,
and (d) a tiny promotion-mate joint overlap branch. Each delta is bounded by
`Delta_k * tanh(...)` and each gate is multiplied by a structural mask that
zeros the branch when its prerequisites do not hold (no candidate pawns near
promotion / no enemy-king pressure / no joint overlap).

The source research packet is
`ideas/research/packets/classic/i257_promotion_mate_slice_specialist.md`.
This implementation realises the packet's recommended first-deployment
target: a fast conv trunk with the four chess-explained specialist branches
layered above it, all driven from the simple_18 board only. The packet's
`OrientedTacticalSheafFast` (i249) trunk variant is documented as a planned
follow-up; swapping the encoder leaves the head interface unchanged.

## Implementation Binding

- Registered model name: `promotion_mate_slice_specialist`
- Source implementation:
  `src/chess_nn_playground/models/trunk/promotion_mate_slice_specialist.py`
  (`PromotionMateSliceSpecialist`,
  `build_promotion_mate_slice_specialist_from_config`)
- Idea-local wrapper:
  `ideas/registry/i257_promotion_mate_slice_specialist/model.py`
  (`build_model_from_config`)
- Registry manifest key:
  `promotion_mate_slice_specialist` in
  `src/chess_nn_playground/models/_registry_manifest.py`

## Dataflow

```text
simple_18 board
   │
   ├── DualStreamFeatureBuilder (reused from i193, no learned weights)
   │      ├── exchange feature stack (8 planes)
   │      ├── king feature stack    (8 planes)
   │      └── board-level summary   (8 scalars)
   │
   ├── compact conv encoder (channels=32, depth=2 by default)
   │
   ├── base_head (trunk mean/max pool + summary)
   │      └── base_logit                          (B,)
   │
   ├── promotion candidate field
   │      ├── deterministic gather of own pawns on the 6th/7th
   │      │   array rows (one/two pushes from promotion for the side
   │      │   to move)
   │      └── candidate descriptors                (B, K, hidden)
   │
   ├── promotion branch (type-conditioned fanout over {Q, R, B, N})
   │      ├── type embeddings + per-type attack-delta masks
   │      ├── per-(candidate, type) score
   │      └── promo summary -> promo_delta, promo_gate
   │
   ├── underpromotion branch
   │      ├── per-candidate non-queen-vs-queen margin
   │      └── under summary -> under_delta, under_gate
   │
   ├── mate witness branch
   │      ├── pooled trunk @ enemy-king / enemy-zone masks
   │      ├── 6 deterministic king scalars
   │      │   (check, escape, in-zone-attack, ray-to-zone,
   │      │   capture-in-zone, zone-balance)
   │      └── mate summary -> mate_delta, mate_gate
   │
   ├── joint promotion-mate overlap branch
   │      └── small MLP over the three summary vectors
   │            -> joint_delta, joint_gate
   │
   └── sparse specialist mixer
         └── final_logit = base + sum_k gate_k * delta_k
```

The forward pass is fully tensor-only: there is no python-chess fallback and
no engine search. Promotion candidate gather works in tensor space; the mate
witness scalars are derived from the deterministic king feature stack.

## Specialist Branches

### Promotion fanout (Q / R / B / N)

For each near-promotion own pawn (top-K by mask), the model projects
`[trunk[source], exchange[source], king[source], promotion_extras]` into a
candidate descriptor. Each candidate is then expanded into four
type-conditioned descriptors via a type embedding and per-type analytic
attack-delta masks. A per-(candidate, type) score feeds a softmax type
attention; the weighted descriptors are pooled across candidates with the
mask-weighted average. The pooled descriptor plus six scalar pool statistics
form the promotion summary, which feeds a bounded tanh delta and a sigmoid
gate; the gate is multiplied by a structural mask that zeros the branch when
no candidate pawn exists.

### Underpromotion divergence

The same per-(candidate, type) scores feed an explicit non-queen-vs-queen
margin: `margin = max(N, B, R) - Q`. Aggregation is the mask-weighted
average. The summary head consumes
`[agg_margin, candidate_count, type_entropy, agg_nonqueen, agg_queen]` and
emits a bounded delta and a confidence-conditioned sigmoid gate. The branch
is structurally masked off when there are no near-promotion candidates.

### King-zone forcing witness

Pools the trunk activations weighted by the enemy-king mask and the
enemy-king-zone mask, then concatenates with the deterministic king feature
mean, the dual-stream board-level summary, and six explicit witness scalars:
checking-move count, escape-square count, in-zone attack count, own
ray-to-zone count, total enemy material-value in zone vulnerable to own
attacks, and a zone-balance ratio. A small MLP emits a bounded delta and a
sigmoid gate; the gate is structurally masked off unless at least one of
checking-move count or in-zone attack count is positive.

### Promotion-mate joint overlap

A tiny MLP over the concatenation of the three summary vectors. Bounded by
`joint_delta_bound = 0.75` (smaller than the per-branch bound by default)
and structurally masked off when either the promotion or the mate branch is
inactive. Confidence is the product of the promotion and mate gates
(detached so the joint gate does not back-propagate twice through the same
parameters).

## Inputs and Contract

- Input: simple_18 current-board tensor `(B, 18, 8, 8)`.
- Output: dict with `logits` of shape `(B,)` plus per-sample diagnostic
  scalars. Compatible with the repo's shared trainer artifact pipeline.
- The model never reads CRTK metadata, source labels, verification flags,
  PVs, or engine evaluations. Slice tags remain reporting-only.

## Ablations

The model exposes a single `ablation` enum (also surfaced as a config field):

| Ablation | Effect |
|---|---|
| `none` | Full specialist (default). |
| `trunk_only` | Disable every specialist branch; only the trunk pool feeds the final logit. |
| `copy_baseline_fanout` | Replace per-type promotion scores with a uniform repeat of the candidate's mean score. |
| `uniform_type_attention` | Zero per-type promotion scores so type attention is uniform. |
| `zero_under_margin` | Zero the non-queen-vs-queen margin in the underpromotion branch. |
| `no_mate_witness` | Zero the six deterministic mate witness scalars before the mate summary. |
| `no_joint_branch` | Zero the joint promotion-mate delta and gate. |
| `disable_gate` | Force each gate to 1 (subject to structural masks). |
| `force_zero_gate` | Set every gate to 0 (trunk-only delivery). |

These match the chess-semantic falsifiers called out in the research packet
(`copy_baseline_fanout`, `uniform_type_attention`, `zero_under_margin`,
`shuffle_mate_witness` approximated by `no_mate_witness`, `disable_gate`,
`force_zero_gate`). Each ablation is a one-flag config change inside the
same registered builder.

## Scope Notes

The current implementation is the C1 first-deployment target. It
intentionally does *not* yet:

- run on the `oriented_tactical_sheaf_fast` (i249) trunk; the packet's
  recommended trunk substitution requires a `forward_features` refactor of
  the existing i018/i249 module and is documented as a planned follow-up;
- attach the `L_gap_rank` / `L_slice` / gate-sparsity auxiliary losses from
  the packet (the shared trainer uses BCE-with-logits on the puzzle logit);
  these require trainer extensions that should not be bundled with the
  architecture promotion;
- run any slice-weighted near-puzzle curriculum; the shared sampler is used
  so unrelated runs are not perturbed;
- precompute a versioned board-derived rule-feature cache. Promotion
  candidates and mate witness scalars are recomputed at forward time from
  the deterministic feature builder; the cache plan from the packet is
  out-of-scope for this promotion.

These extensions are listed in `implementation_notes.md` as the planned
follow-up work and are deliberately not claimed here.
