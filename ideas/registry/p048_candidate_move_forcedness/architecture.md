# Architecture

`Candidate Move Forcedness Primitive` (p048, CMF) is an additive
gated head on top of the i193 `ExchangeThenKingDualStreamNetwork`
trunk. The head turns the rule-derived pseudo-legal candidate move
set into a board-level forcedness pool that estimates whether a
position contains one or a few unusually coercive candidate moves.

The model consumes `simple_18` `(B, 18, 8, 8)` and returns one
puzzle logit plus a per-sample diagnostics dict.

## Forward pass

1. **i193 trunk forward**. Emits `base_logit` and trunk diagnostics
   plus the joint pool feature (via `trunk_joint_features`).
2. **Pseudo-legal move adjacency**. Frozen `compute_legal_move_graph`
   helper produces `(B, 64, 64)` pseudo-legal edge adjacency with
   per-edge `move_type` and `ray_direction` codes, plus own / enemy
   piece masks. Edges are stop-gradient.
3. **Deterministic forcedness descriptors**. For every edge `(i, j)`
   that fires in the adjacency we build a 14-channel descriptor:
   - `mover_value`, `victim_value`: piece-value scale at source /
     target (own / enemy occupancy weighted).
   - `is_capture`, `is_check_seed`, `is_promotion_seed`: rule-derived
     forcing indicators.
   - `source_degree_norm`, `target_in_degree_norm`: mobility shock
     proxies (normalised by 28 and 16 respectively).
   - One-hot move-class flags: knight, rook-like (rank / file),
     bishop-like (diag / antidiag), king, pawn push, pawn capture.
   - `see_lite`: capture gain heuristic
     `max(victim - 0.5 * mover, 0) * is_capture` (no exchange
     search).
4. **Square tokens**. A 1x1 conv tower projects simple_18 to
   `(B, 64, token_dim)` tokens; LayerNorm.
5. **Move-type / direction embeddings**. Per-edge embeddings indexed
   by `move_type` and `ray_direction` codes; summed.
6. **Edge scoring**. `score_mlp(LayerNorm(cat(src, dst, type, f)))`
   produces a per-edge logit. Inactive edges get `-inf` so they
   never enter top-k.
7. **Top-k pool**. We select the top-`k` candidate moves by score
   (default `k=4`) and pool:
   - Scalars: `top1_score`, `top1 - top2` gap, top-`k` softmax mass,
     softmax entropy over all candidates, `log(1 + move_count)`.
   - `top1_feat` (14 channels at the top move).
   - `topk_feat_mean` (mean per-channel over the top-`k`).
   - `cat_max` (per-channel max over all legal candidates).
8. **Delta head**. MLP on `cat(summary_flat, joint)` to a scalar
   `primitive_delta_raw`.
9. **Gate**. MLP over `cat(joint, top1_score, gap12, entropy)` to
   sigmoid `primitive_gate`; initial bias `gate_init = -2.0`.
10. **Output**. `final_logit = base_logit + primitive_gate *
    primitive_delta_raw`.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full CMF architecture (default). |
| `deterministic_score` | **Primary falsifier**. Replace the per-move learned score with the deterministic feature sum. Tests whether the learned scorer is load-bearing. |
| `mean_pool` | Replace top-k pooling with mean over all legal candidates. Tests whether candidate concentration matters. |
| `flags_only` | Drop piece values, mobility, and SEE-lite channels. Tests whether deeper features earn their cost. |
| `dense_edges` | Replace pseudo-legal adjacency with a fully-connected mask. Tests whether exact legality matters beyond all-pairs. |
| `no_consequence` | Drop check / capture / promotion seeds and SEE-lite. |
| `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| `trunk_only` | Same as `zero_delta`; semantic alias. |
| `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine
evaluations, and principal variations are **not** consumed.

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Trunk joint refeat | Two encoder passes total |
| Legal-move graph | Frozen analytic helper (no Python loop) |
| Token tower | Two 1x1 convs + LayerNorm |
| Edge MLP | One `(B, 64, 64)` forward; flatten + project |
| Top-k pool + heads | Small MLPs |

The dense `(B, 64, 64, EDGE_FEATURE_DIM=14)` edge tensor is the
dominant cost; at `B=256` it stays within the i193 scout-scale
memory budget.

## Implementation Binding

- Registered model name: `candidate_move_forcedness`.
- Source implementation: `src/chess_nn_playground/models/primitives/candidate_move_forcedness.py`.
- Shared helper: `trunk_joint_features` from
  `src/chess_nn_playground/models/primitives/trunk_features.py`.
- Move-graph helper: `compute_legal_move_graph` from
  `src/chess_nn_playground/models/primitives/legal_move_graph.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p048_candidate_move_forcedness/model.py`.
- Training config: `ideas/registry/p048_candidate_move_forcedness/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/_registry_manifest.py`:
  `candidate_move_forcedness ->
  chess_nn_playground.models.primitives.candidate_move_forcedness.build_candidate_move_forcedness_from_config`.

## Source

`ideas/research/primitives/external_43_candidate_move_forcedness_primitive.md`.
The source markdown is retained verbatim under its original path; the
registry folder references it via `source_primitive_path` and
`source_packet_path` in `idea.yaml`.
