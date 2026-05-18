# Architecture

`Learned Relation Confidence Primitive` (p047, LRC) is an additive
gated head on top of the i193 `ExchangeThenKingDualStreamNetwork` trunk.
The head learns per-edge confidence weights that *only attenuate* the
12 typed tactical relation masks emitted by the deterministic i018
`TacticalIncidenceBuilder`.

The model consumes `simple_18` `(B, 18, 8, 8)` and returns one puzzle
logit plus a per-sample diagnostics dict.

## Forward pass

1. **i193 trunk forward**. Emits `base_logit` and trunk diagnostics
   plus the joint pool feature (via `trunk_joint_features`).
2. **Deterministic relation masks**. Frozen `BoardStateAdapter` plus
   frozen `TacticalIncidenceBuilder` produce `(B, R=12, 64, 64)`
   binary relation masks (attacks, defenses, king-zone pressure,
   ray, knight, pawn, pin).
3. **Square tokens**. A 1x1 conv tower projects simple_18 to
   `(B, 64, token_dim)` tokens; LayerNorm.
4. **Edge scoring**. The per-edge logit decomposes as
   `(src_score + tgt_score + low_rank + rel_bias) * sigmoid(rel_gate)`
   where:
   - `src_score`, `tgt_score`: per-relation MLPs over the per-square
     (empty + 12 piece-presence) descriptor, applied separately and
     broadcast.
   - `low_rank`: `einsum('brik,brjk->brij', q_r * rel_emb, k_r)` with
     rank-`low_rank_dim` projections (default 8).
   - `rel_bias`, `rel_gate`: per-relation learnable scalars; bias is
     initialised positive so untrained confidences land near
     sigmoid(2) on active edges.
5. **Confidence**. `confidence = sigmoid(edge_logit / temperature) *
   mask`. Topology preservation is structural: an inactive mask entry
   stays exactly zero after weighting.
6. **Summary**. Per-(batch, relation) summary tensor of mean
   confidence on active edges, total mass, soft kept fraction at
   threshold 0.5, and binary entropy. The (B, R, 4) tensor is
   LayerNormed and flattened.
7. **Delta head**. MLP on `cat(summary_flat, joint)` to a scalar
   `primitive_delta_raw`.
8. **Gate**. MLP over `cat(joint, mean_confidence, mask_density)`
   to sigmoid `primitive_gate`; initial bias `gate_init = -2.0`.
9. **Output**. `final_logit = base_logit + primitive_gate *
   primitive_delta_raw`.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full LRC architecture (default). |
| `binary_only` | **Primary falsifier**. Skip confidence; weighted_mask := mask. Recovers binary i018 relation mass. |
| `scrambled_mask` | In-batch permute the deterministic relation masks. Tests whether mask-position alignment matters. |
| `shuffle_pieces` | In-batch permute the piece descriptor. Tests whether piece identity matters in scoring. |
| `gate_only` | Disable per-edge scoring; only per-relation gate is learned. Tests whether edge-level structure beats coarse rescaling. |
| `no_low_rank` | Drop the low-rank bilinear term. |
| `no_edge_mlp` | Drop the edge MLP; keep low-rank only. |
| `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| `trunk_only` | Same as `zero_delta`; semantic alias. |
| `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
and principal variations are **not** consumed.

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Trunk joint refeat | Two encoder passes total |
| Deterministic relation builder | Frozen `TacticalIncidenceBuilder` forward |
| Token tower | Two 1x1 convs + LayerNorm |
| Edge scoring | One `(B, R, 64, 64)` bilinear einsum + factorised MLPs |
| Summary + delta / gate | Small MLPs |

The dense `(B, R, 64, 64)` confidence tensor is the dominant cost; at
default `R=12, low_rank_dim=8` it stays within the i018 / i249 scout-
scale memory budget at `B=256`.

## Implementation Binding

- Registered model name: `learned_relation_confidence`.
- Source implementation: `src/chess_nn_playground/models/primitives/learned_relation_confidence.py`.
- Shared helper: `trunk_joint_features` from
  `src/chess_nn_playground/models/primitives/trunk_features.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Relation source: `TacticalIncidenceBuilder` from
  `src/chess_nn_playground/models/trunk/oriented_tactical_sheaf.py`.
- Idea-local wrapper: `ideas/registry/p047_learned_relation_confidence/model.py`.
- Training config: `ideas/registry/p047_learned_relation_confidence/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/_registry_manifest.py`:
  `learned_relation_confidence ->
  chess_nn_playground.models.primitives.learned_relation_confidence.build_learned_relation_confidence_from_config`.
