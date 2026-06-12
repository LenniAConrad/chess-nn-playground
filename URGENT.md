# ⚠️ URGENT (revised 2026-06-12): the 2026-06-02 pawn "fix" was itself backward — now corrected and verified

**Date:** 2026-06-02, revised 2026-06-12
**Severity:** High — a network input channel was geometrically inverted, twice, in
opposite directions.

## What actually happened

1. **Original state (pre-2026-06-02, committed):** the playground's OTIS pawn masks
   (`_make_geometry_masks` in `oriented_tactical_sheaf.py`) were **correct**
   (`our_pawn` target rank = source rank − 1).
2. **2026-06-02 "fix" (working tree only, never committed):** the earlier version of
   this file claimed the playground shared chess-rtk's pawn-direction bug and flipped
   the mask signs. That analysis used the wrong frame, and the flip **introduced** the
   backward-pawn bug here. Its "verification" (`our_pawn[e2=12] -> {d3=19, f3=21}`)
   assumed square 12 is e2 — in the tensor frame the models actually consume,
   square 12 is **e7**.
3. **2026-06-12 (this revision):** the flip was reverted, the true frame documented
   and verified end-to-end, and three additional orientation bugs found in the same
   audit were fixed (see below).

## The frame, stated correctly this time

"The playground uses `0 = a1`" is true of **python-chess square indices** in the data
code, but **false for the tensors the models consume**. The encoders
(`board_features.py`) write `row = 7 - (square // 8)`, so after `flatten(2)` the
model-side square index runs **0 = a8 ... 63 = h1** — the *same* effective frame as
chess-rtk, not a mirrored one.

Consequences, verified empirically (300 random positions, exact set-equality of every
attack/relation channel against python-chess; `/tmp/otis_true_frame_check.py`):

- Internal rank (`sq // 8`) runs 0 = board rank 8 down to 7 = board rank 1.
- After side-to-move canonicalization the **mover's home is internal rank 7** and
  mover pawns attack toward **lower** internal rank; the opponent's toward higher.
- Correct masks (restored): `our_pawn` target rank = source rank − 1,
  `their_pawn` target rank = source rank + 1.

## Additional orientation bugs found and fixed (2026-06-12)

All in `src/chess_nn_playground/models/trunk/oriented_tactical_sheaf.py`:

- **`promotion_distance` inverted** (`_square_coordinates`): was `(7 - rank) / 7`
  (assumes promotion at internal rank 7); the mover promotes at internal rank 0.
  Now `rank / 7`.
- **Black-to-move canonicalization mirrored files**: `BoardStateAdapter` used
  `torch.rot90(k=2)` (180° rotation = rank flip **and** file mirror), silently
  swapping king/queen-side geometry for black-to-move samples and contradicting both
  its own castling-channel remap and the repo-wide convention
  (`Simple18SideCanonicalizer`, the `lc0_bt4_112` encoder, and chess-rtk all flip
  ranks only). Now a rank-only `torch.flip(dims=(-2,))`. Verified: adapter output for
  a black-to-move board now exactly equals the encoding of python-chess
  `Board.mirror()` (150 random positions).
- **`lc0_static_112` castling channels never color-swapped** for black-to-move
  positions (the 18-channel branch swapped 13↔15/14↔16; the 112 branch left 106-109
  absolute). Now swapped (106↔108, 107↔109).

## Blast radius

- `oriented_tactical_sheaf.py` (i018) — fixed at source.
- `oriented_tactical_sheaf_controlled_encoding.py`, `pin_xray_overload_sheaf.py`
  (i252/V2), `oriented_tactical_sheaf_efficient_xxl.py`, and every other model that
  imports `_make_geometry_masks` / `TacticalIncidenceBuilder` — auto-fixed.
- Checkpoints trained **2026-06-02 → 2026-06-12** (inverted working-tree masks):
  backward pawn channel and backward pawn contributions to attack channels —
  **discard / retrain**.
- Checkpoints trained **before 2026-06-02**: pawn channels were correct, but
  black-to-move boards were file-mirrored (and `lc0_static_112` castling channels
  opponent-attributed) — retrain before exporting.
- Any OTIS `.bin` export to chess-rtk must come from a checkpoint trained after
  2026-06-12.

## Cross-repo note (chess-rtk)

chess-rtk's own fix (`Model.java` `addPawnRelations`, `otis_gpu_impl.inl`:
`isWhite ? -1 : 1`) **remains correct** — its regression tests assert e2 → d3/f3 in
its native 0 = a8 frame. The earlier claim that "the fix sign differs between the
repos because the square conventions are mirrored" was wrong: both effective frames
index from a8, so the forward direction has the same sign in both. **No further
chess-rtk change needed** — but if chess-rtk copied the playground's 2026-06-02 sign
(`isWhite ? +1 : -1` style) anywhere else "for parity", re-check it there.

## Related (not OTIS, not yet fixed — tracked separately)

The 2026-06-12 audit found the same wrong-frame assumption (mover home at internal
rank 0) baked into other models: `candidate_move_forcedness_sheaf.py` (pawn move
enumeration: double-push/promotion ranks backward), `forcing_certificate_transformer.py`
and `_research_blocks.py` (white/black pawn operators color-swapped),
`tiny_chess_micronet.py` (side-relative forward basis inverted),
`relational_query_algebra.py` (inconsistent rel_rank between fact tables),
`chess_hypercut_polynomial.py` (same rot90-vs-flip pattern),
`primitives/learned_relation_confidence.py` (mixes mover-canonical masks with
raw-frame features), `primitives/efficient_ray_occlusion_scan.py` (no side-to-move
canonicalization), `dykstra_lcp.py` (channel-count-only role dispatch).

## Action items

- [x] Restore correct pawn-mask orientation (2026-06-12, verified vs python-chess).
- [x] Fix `promotion_distance`, rot90→rank-flip canonicalization, `lc0_static_112`
      castling swap (2026-06-12).
- [ ] Retrain / re-evaluate OTIS-family checkpoints (i018, i249, i252); re-export any
      production `.bin` OTIS weights from a post-2026-06-12 checkpoint.
- [ ] Triage the non-OTIS frame bugs listed above per model.
- [ ] Remove this file once no pre-fix OTIS weights remain in use.
