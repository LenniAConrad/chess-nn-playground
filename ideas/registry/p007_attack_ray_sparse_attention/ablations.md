# Ablations — p007 Attack-Ray Sparse Attention

## Switches (model.ablation)

| Mode | What it tests |
|---|---|
| `none` | Full architecture (default). |
| `uniform_attention` | Replace the per-slot softmax with uniform 1/K over valid slots. Tests whether learned attention scores carry signal beyond fixed-mean pooling. |
| `random_keys` | Replace the 8 ray-blocker slots with random squares (self-slot kept). **Primary ARSA falsifier**: if the architecture matches `random_keys`, the rule-derived geometry was not load-bearing — the operator is just a 9-slot mixer over arbitrary squares. |
| `no_blocker_mask` | Mark every slot valid even when the ray has no blocker. Tests whether the mask is load-bearing or whether the model self-edge fallback is enough. |
| `zero_delta` | Hold `primitive_delta = 0`. i193 baseline. |
| `disable_gate` | Hold `primitive_gate = 1`. |
| `trunk_only` | Strict no-op (zero features + zero delta). |

## Falsification criteria

Promote p007 only if `model.ablation = none`:

- Aggregate PR AUC delta from i193 >= -0.005.
- CRTK class-1 matched-recall FP rate drops by at least 5% relative.
- Wall-clock per epoch within 1.3x of i193.

Drop p007 if:

- `random_keys` matches `none` (geometry was not the source of lift).
- `uniform_attention` matches `none` (the softmax weighting was noise).
- `zero_delta` matches `none` (delta head is overfitting nuisance).

## Deferred internal proposals from external_03

Per the implementation rule "implement the strongest or first-ranked
proposal", only **ARSA** is implemented here. The other four candidates
in
`ideas/research/primitives/external_03_attack_ray_sparse_attention_delta_accumulator.md`
are deferred:

- **DAP — Delta-Accumulator Primitive**: stateful, paired apply/unmake
  linear with a temporal autograd chain over the make-stack. Deferred —
  scout training uses dense batches, not search trajectories; DAP's
  asymptotic win is invisible to this falsifier.
- **Tree-S6 — Tree-Selective State-Space**: Mamba-style selective scan
  over a branching game tree. Deferred — chess-nn-playground's puzzle
  benchmark does not ship tree contexts; this primitive needs a tree
  benchmark to be meaningfully tested.
- **WrEC — Wreath-Equivariant Conv**: D4 × Z2 group-equivariant conv
  with paired piece-type involution. Deferred — closely overlaps
  Carroll & Beel 2020 and escnn 2022; flagged in its own self-audit as
  a reframe rather than a fresh primitive.
- **SHK — Sparse Hyper-Kernel Generator**: hypernetwork emitting per-
  position sparse conv kernel support via Gumbel-top-k. Deferred — the
  file's own ranking notes the per-sample kernel breaks cuDNN batching
  and is unlikely to win on a 3070.

If any prove relevant after p007's scout, they should be promoted under
fresh `p###` IDs.
