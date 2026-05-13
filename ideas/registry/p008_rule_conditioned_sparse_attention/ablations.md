# Ablations — p008 Rule-Conditioned Sparse Attention (MobScan)

## Switches (model.ablation)

| Mode | What it tests |
|---|---|
| `none` | Full architecture (default). |
| `random_edges` | Replace legal-move graph with a random mask of identical density. **Primary MobScan falsifier**: tests whether the rule-derived adjacency carries the lift. |
| `dense_edges` | Use fully-connected mask. Tests whether the multi-hop propagation is the signal, or whether sparsity itself was load-bearing. |
| `untied_state` | Disable the selective gates (force `A = 0.5`, `B = 0.5`, `C = 1`). Tests whether input-conditioned gates are load-bearing or whether the rule-derived topology alone is enough. |
| `single_iteration` | Force `num_iterations = 1`. Tests whether the multi-step recurrence is the lift source. |
| `zero_delta` | Hold `primitive_delta = 0`. i193 baseline. |
| `disable_gate` | Hold `primitive_gate = 1`. |
| `trunk_only` | Strict no-op (zero features + zero delta). |

## Falsification criteria

Promote p008 only if `model.ablation = none`:

- Aggregate PR AUC delta from i193 >= -0.005.
- CRTK class-1 matched-recall FP rate matches or beats i193.
- Wall-clock per epoch within 1.2x of i193.

Drop p008 if:

- `random_edges` matches `none` (topology was not load-bearing).
- `untied_state` matches `none` (selective gates were redundant).
- `single_iteration` matches `none` (multi-step propagation was not
  load-bearing; the operator could be replaced by a simpler one-hop
  message pass).

## Deferred internal proposals from external_04

The source primitive packet
(`ideas/research/primitives/external_04_rule_conditioned_sparse_attention_mobscan.md`)
ranks five proposals. The first is **RCSA** which is structurally the
same operator as p007 ARSA — to avoid duplicate primitives within the
ray-legal batch, p008 implements **MobScan** (proposal #2). The
remaining proposals are deferred:

- **RCSA — Ray-Cast Sparse Attention**: covered by p007 ARSA in this
  batch.
- **EDA — Edit-Differential Accumulator**: stateful NNUE-style
  accumulator with mathematical contract that incremental and full
  forward pass are equal. Deferred — engine-deployment primitive whose
  scout-scale falsifier needs a make/unmake search loop the batch does
  not have.
- **SPIL — Signed-Permutation Involution Layer**: D4 × Z2 group-
  equivariant conv with paired piece-type involution. Deferred — the
  file's own self-audit flags it as "reframing of G-CNN, not a new
  primitive".
- **CRHBA — Content-Routed Hypergraph Block Attention**: MoBA over
  input-determined hyperedges. Deferred — requires ragged-tensor
  custom kernels that exceed the "minimal trainer surface area"
  budget of this batch.

If any prove relevant after p008's scout, they should be promoted under
fresh `p###` IDs.
