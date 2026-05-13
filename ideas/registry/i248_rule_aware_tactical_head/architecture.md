# Architecture

`Rule-Aware Tactical Head` (i248) is an additive, gated head on top of the
existing i193 `ExchangeThenKingDualStreamNetwork` trunk. The thesis (see
`math_thesis.md`) is that the i193 trunk underfits the mate_in_1 /
stalemate-trap slices because it has to learn rule-exact checkmate
detection from raw piece-position features. The Terminal-State Detection
Primitive (TSDP) supplies that signal directly via `python-chess` and a
small fusion head produces a logit delta that the trunk's residual head
does not need to learn.

The model consumes the repository `simple_18` current-board tensor
`(B, 18, 8, 8)` and returns one puzzle logit for the BCE-with-logits
`puzzle_binary` trainer, plus a rich per-sample diagnostics dict (i193's
diagnostics plus the new `primitive_*` and `tsdp_*` outputs).

## Mechanism

1. **i193 trunk forward**. The bespoke
   `ExchangeThenKingDualStreamNetwork` runs unchanged and emits:

   - `logits` (treated here as `base_logit`)
   - `exchange_logit`, `king_logit`, `gate`, `gate_logit`,
     `residual_logit`, `gate_entropy`, `stream_disagreement`,
     `exchange_pool_norm`, `king_pool_norm`, `mechanism_energy`,
     `proposal_profile_strength`, `proposal_keyword_count`

2. **TSDP feature extraction**. From the `simple_18` board tensor we
   reconstruct a `chess.Board` per sample (piece placement, side-to-move,
   castling rights, en-passant square) and enumerate legal moves. For each
   legal move the resulting position is classified by exact chess rules.
   Aggregation produces the 11-dim feature vector documented in
   `math_thesis.md`. This computation is wrapped in `torch.no_grad()` —
   rule indicators are stop-gradient by design.

3. **Fusion**. The 11-d TSDP vector is normalised by per-feature scales so
   counts and the 0/1 flags share an O(1) magnitude, and concatenated with
   four stop-gradient i193 diagnostics (`gate`, `gate_entropy`,
   `mechanism_energy`, `stream_disagreement`). The 15-d fusion vector
   feeds two small LayerNorm + GELU MLP heads:

   - `gate_mlp` -> scalar gate logit -> sigmoid -> `primitive_gate`
   - `delta_mlp` -> scalar `primitive_delta_raw`

   The final logit is

   ```text
   final_logit = base_logit + primitive_gate * primitive_delta_raw
   ```

4. **Ablations**. Five supported modes, controlled by `model.ablation`:

   - `none`: full architecture (default).
   - `shuffle_tsdp`: in-batch random permutation of the 11-d TSDP vector,
     decoupling the rule indicators from the actual position. This is the
     TSDP falsifier: if the architecture matches `shuffle_tsdp`, the rule
     indicators carry no signal in this trunk and the primitive is dropped.
   - `disable_gate`: hold the gate at 1.0 — tests whether the gate is
     load-bearing or whether direct concatenation works.
   - `zero_delta`: hold the primitive delta at 0 — equivalent to running
     the i193 trunk alone (architecture-level baseline).
   - `zero_features`: zero out the TSDP vector but keep the head — tests
     whether the head learns from trunk diagnostics alone.
   - `trunk_only`: zero out both features and delta — minimal control.

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata are
*not* consumed by the model. The 11-dim TSDP vector is rule-derived from
the `simple_18` board tensor — that is, from piece placement, side-to-
move, castling rights, and en-passant square only. This is the same
contract i193 follows.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| TSDP fallback | `O(|M|)` python-chess legal-move evaluations (~50/position) |
| Fusion head | Two small MLPs over a 15-d vector |

The python-chess fallback dominates the per-step CPU cost. The follow-up
production path (`scripts/data/precompute_primitive_features.py`,
documented as a planned upgrade) precomputes the 11-d TSDP vector into a
parquet column at split-build time, so training-time cost reduces to a
tensor index. The model is structured so that switching to the
precomputed-feature input is a small, local change to `_compute_tsdp`.

## Implementation Binding

- Registered model name: `rule_aware_tactical_head`.
- Source implementation: `src/chess_nn_playground/models/primitives/rule_aware_tactical_head.py`.
- TSDP feature source: `src/chess_nn_playground/data/terminal_state.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`
  (the bespoke i193 `ExchangeThenKingDualStreamNetwork` is wrapped, not
  reimplemented).
- Idea-local wrapper: `ideas/registry/i248_rule_aware_tactical_head/model.py`.
- Training config: `ideas/registry/i248_rule_aware_tactical_head/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["rule_aware_tactical_head"] = build_rule_aware_tactical_head_from_config`.
