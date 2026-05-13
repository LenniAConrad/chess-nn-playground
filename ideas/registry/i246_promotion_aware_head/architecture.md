# Architecture — Promotion-Aware Head (i246, PFCT primitive)

## Architecture description

The Promotion-Aware Head wraps the i193 dual-stream trunk with an additive,
gated cross-attention head over the per-pawn promotion fanout `F_theta(p, x)`.
The forward pass is structured so the primitive contributes exactly zero to
the final logit on positions without own near-promotion pawns.

```
                       simple_18 board (B, 18, 8, 8)
                                  |
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
   ExchangeThenKingDualStreamNetwork   find_near_promotion_slots(K=max_pawns)
        base_logit, base_joint                slot_squares,
        (B,), (B, F)                          slot_valid
                                                    |
                                                    ▼
                                   build_promotion_counterfactuals
                                   (B, K, 4, 18, 8, 8)
                                                    |
                                                    ▼
                                shared trunk encoder pass on (B*K*4, 18, 8, 8)
                                cf_features: (B, K, 4, F)
                                                    |
                            ┌──────── cross-attention per pawn ─────────┐
                            ▼                                          ▼
                  query = MLP_q(base_joint, pawn_emb)        key = MLP_k(cf_features, type_emb)
                            ▼                                          ▼
                            └──────── softmax(qkᵀ / sqrt(d)) ──────────┘
                                          alpha (B, K, 4)
                                                    |
                          values = Linear(cf_features) (B, K, 4, H)
                                          ▼
                       pawn_feature = sum_T alpha_T * V_T (B, K, H)
                                          ▼
                       pawn_delta = MLP_delta(pawn_feature) (B, K)
                                          ▼
                       primitive_delta = sum_p (pawn_delta * slot_valid)
                                          ▼
                   gate = sigmoid(MLP_gate(base_joint)) * has_promotion_pawn
                                          ▼
                       final_logit = base_logit + gate * primitive_delta
```

## Input format

- ``x``: ``(B, 18, 8, 8)`` simple_18 board tensor (piece planes 0–11,
  white-to-move at 12, castling 13–16, en-passant 17).
- All near-promotion pawn identification and substitution operates purely on
  piece planes 0 (white pawn) and 6 (black pawn) plus the side-to-move plane.
  No python-chess call is required inside ``forward``.

## Forward pass

1. **Baseline trunk pass.** Run the i193 trunk on ``x`` to get
   ``base_logit`` and the full i193 diagnostics dict.
2. **Joint feature extraction.** Run the trunk's feature builder and stream
   encoders on ``x`` to get ``base_joint`` (the i193 pool concat plus summary
   planes) — the same vector the trunk uses as input to its phase router and
   residual head.
3. **Near-promotion slot search.** Identify up to ``K`` own near-promotion
   pawn files per sample (white-to-move: pawn on rank 7; black-to-move:
   pawn on rank 2). Slots are filled in file order (a-file first) and
   unused slots are marked invalid.
4. **Counterfactual board grid.** For every (sample, slot, type) build a
   substituted board: zero the source pawn, zero all 12 piece planes on
   the promotion square (covers capture-promotion), then set the
   promoted piece's plane to 1 on the promotion square.
5. **Shared encoder pass on counterfactuals.** Flatten the
   ``(B, K, 4, 18, 8, 8)`` grid to ``(B*K*4, 18, 8, 8)`` and pass it
   through the same trunk feature builder + stream encoders to get the
   ``(B, K, 4, F)`` fanout features.
6. **Cross-attention per pawn.** Compose the query from the baseline
   feature and the per-pawn square embedding; compose the keys from each
   counterfactual feature and the per-piece-type embedding; softmax over
   the 4 piece types; apply the resulting weights to the per-type values
   (linear projection of the counterfactual features) to obtain
   ``pawn_feature``.
7. **Delta + gate.** Project each pawn's pooled feature to a scalar
   ``pawn_delta``, mask invalid slots, sum across pawns, and gate the
   result with ``sigmoid(MLP_gate(base_joint)) * has_promotion_pawn``.
   The structural multiplication by ``has_promotion_pawn`` is what
   guarantees zero contribution on the majority of positions.
8. **Final logit.** ``final_logit = base_logit + gate * primitive_delta``.

## Tensor shapes

| Symbol            | Shape                  | Description                                    |
|-------------------|------------------------|------------------------------------------------|
| board             | (B, 18, 8, 8)          | simple_18 input                                 |
| base_logit        | (B,)                   | i193 puzzle logit                               |
| base_joint        | (B, F)                 | i193 joint feature (≈ 2*output_dim + 8)         |
| slot_squares      | (B, K) long            | row-major plane square of each pawn slot        |
| promote_square    | (B, K) long            | row-major plane square of each promotion target |
| slot_valid        | (B, K) bool            | True iff slot points at a real own pawn         |
| cf_grid           | (B, K, 4, 18, 8, 8)    | counterfactual boards                           |
| cf_features       | (B, K, 4, F)           | fanout features after trunk pass                |
| pawn_emb          | (B, K, pawn_embed_dim) | per-square pawn embedding                       |
| type_emb          | (B, K, 4, promotion_embed_dim) | per-piece-type embedding             |
| alpha             | (B, K, 4)              | softmax attention weights                       |
| values            | (B, K, 4, H)           | linear projection of cf_features                |
| pawn_delta        | (B, K)                 | per-pawn scalar delta (masked)                  |
| primitive_delta   | (B,)                   | summed delta across pawns                       |
| gate              | (B,)                   | sigmoid gate * has_promotion_pawn               |
| final_logit       | (B,)                   | base_logit + gate * primitive_delta             |

## Output heads

- ``logits``: puzzle_binary logit.
- ``base_logit``: i193 trunk output (pre-PFCT).
- ``primitive_delta``: pre-gate sum of per-pawn deltas.
- ``primitive_gate``: post-pawn-presence gate.
- ``primitive_gate_logit`` / ``primitive_gate_entropy``: gate diagnostics.
- ``primitive_logit_contribution``: ``gate * primitive_delta``.
- ``promotion_pawn_count`` / ``promotion_has_pawn``: slot validity counters.
- ``promotion_attention_entropy``: normalised entropy of the per-pawn
  attention distribution, averaged over valid pawns.
- ``promotion_dominant_type``: argmax piece-type index (0=Q, 1=R, 2=B,
  3=N, -1 if no pawn) on the first valid slot — a useful slice diagnostic.
- ``promotion_fanout_dispersion``: mean L2 dispersion of the fanout rows
  around their mean, averaged over valid pawns.
- ``promotion_pawn_delta_max``: max per-pawn delta magnitude.
- ``mechanism_energy`` / ``proposal_profile_strength`` /
  ``proposal_keyword_count``: shared trunk diagnostic re-exports.
- ``trunk_*``: i193 trunk diagnostic re-exports.

## Parameter estimate

With ``trunk_channels=64, trunk_hidden_dim=96, trunk_depth=2`` (the i193
defaults) and the PFCT head defaults
(``max_promotion_pawns=4, pawn_embed_dim=32, promotion_embed_dim=16,
attn_dim=64, head_hidden_dim=64``):

- i193 trunk: ≈190 k parameters (same as the standalone i193 baseline).
- Pawn-square embedding (64 rows × 32): 2,048
- Promotion-type embedding (4 × 16): 64
- Query MLP (LayerNorm + Linear(F+32, 64) + GELU): ≈(F+32)*64 + 192 ≈ 19 k
- Key MLP (LayerNorm + Linear(F+16, 64) + GELU): ≈(F+16)*64 + 192 ≈ 18 k
- Value Linear(F, H=64): ≈F*64 ≈ 17 k (F = 264 with default trunk)
- Delta head MLP: ≈64*32 + 32 + 32 = 2,144
- Gate MLP: ≈F*32 + 32 + 32 ≈ 8,500
- Total head ≈ 67 k parameters

Total ≈ 257 k parameters versus i193's ≈190 k (about +35% parameters).

## Implementation Binding

- Registered model name: `promotion_aware_head`
- Source implementation file: `src/chess_nn_playground/models/primitives/promotion_aware_head.py`
- Idea-local wrapper: `ideas/registry/i246_promotion_aware_head/model.py`

## FLOP estimate

Per batch element on the dense forward (assuming every position has the
maximum K = max_promotion_pawns near-promotion pawns):

- Baseline trunk pass: ≈ i193 cost = `f_i193`
- Counterfactual pass through shared encoder: ``K * 4 * f_i193`` (the trunk's
  geometric feature builder dominates; the encoder convolutions are the
  next-largest term).
- Cross-attention head: O(B * K * F * attn_dim) — negligible vs trunk passes.

In practice the ``has_promotion_pawn`` gate is exactly 0 on positions
without own near-promotion pawns, but the trunk pass on counterfactuals is
still executed (the trunk doesn't short-circuit per-sample). On a typical
chess corpus, ≈ 95% of positions have no near-promotion pawn, so the
"wasted" counterfactual pass is constant per batch but zero-contributing.

Wall-clock estimate: ≈ 1.4–1.5x i193 with ``K=4`` and default trunk channels;
the dense factor is much smaller than the `K * 4` worst case because (a)
many counterfactual boards differ from the baseline only on one or two
squares, so cuDNN sees identical-shape inputs and stays in cached kernels,
and (b) the trunk's deterministic feature builder is shared across the
fanout pass. (Empirical confirmation requires running the trunk on the
canonical corpus; see ``runs/`` for results once the first scout is
finished.)
