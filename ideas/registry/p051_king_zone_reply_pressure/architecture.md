## Architecture

`King-Zone Reply Pressure` (p051, KZRP) is an additive, gated head on
top of the i193 `ExchangeThenKingDualStreamNetwork` trunk. The
thesis (see `math_thesis.md`) is that the i193 trunk's 3x3
convolutions, the legacy `king_ring_pressure` reporting diagnostic in
i018, and the generic reply-structure models (i192, p003) all
under-resolve the specific signal that drives `mate_in_1` and near-
puzzle false positives: *how close the defender is to legal-reply
collapse around their own king*. KZRP turns that into an explicit
32-dim operator vector that the trunk gates and adds.

The model consumes `simple_18` `(B, 18, 8, 8)` and returns one
puzzle logit plus a per-sample diagnostics dict.

## Forward pass

1. **i193 trunk forward.** Emits `base_logit` and trunk diagnostics
   plus the joint pool feature (recomputed via `trunk_joint_features`
   without firing the trunk's heads twice).
2. **Absolute piece state and side-to-move.** Plane indices 0..11
   stay in their original (P, N, B, R, Q, K, p, n, b, r, q, k)
   ordering; the side-to-move scalar plane 12 selects mover-vs-them
   inside the side-vector helper.
3. **Attack and between buffers.** A rule-derived
   `geom_attacks (6, 2, 64, 64)` and `between (64, 64, 64)` are
   precomputed at module init. Slider rays are gated by the
   `between` clear-line mask (no occupant strictly between source
   and target).
4. **Per-direction enemy slider field.** For each side σ, an
   `(B, 8, 64)` per-direction slider activation is produced from
   the queen / rook / bishop planes. This is used by the pin
   detector and re-uses the direction-family masks shared with
   p049 / p050.
5. **Cumsum-based pin detector.** Using the shared
   `ray_geometry (8, 64, 7)` index / mask tables, per-square
   pinned-defender indicators `πw, πb` are computed by walking rays
   from each own king through own pieces and checking whether an
   enemy slider firing along the same direction sits as the second
   ray occupant. A `scatter_add_` projects the per-step pin marks
   back to per-square `(B, 64)`.
6. **Weighted attack masses.** For each colour σ ∈ {W, B} build
   `attack_σ_nom (B, 64) = Σ_p u(p) · src_p_σ ⊙ geom_attacks[p, σ] ⊙ clear`
   and `attack_σ_free` with pinned sources downweighted by a
   sigmoid-bounded `pin_discount`. `u(P,N,B,R,Q,K)` is a learnable
   softplus-bounded attack-unit field initialised to the CPW prior
   `(1, 2, 2, 3, 5, 1)`.
7. **Per-side ring / front masks.** Ring mask is colour-agnostic
   (the 8 Chebyshev-1 neighbours of the defender king). Front mask
   is the 3 squares one rank further in the attacker direction
   beyond the front edge of the ring; `front_mask_w` is used when
   the defender king is white, `front_mask_b` when black.
8. **Side vector** (per side σ ∈ {us, them}). Compute weighted
   zone pressure
   `ZP = Σ_{q ∈ Z_core} w(q) · [A - λ·D_free]_+ + η · Σ_{q ∈ Z_front} [A - λ·D_free]_+`
   (with `w(king_sq) ≈ 4`, `w(empty ring) ≈ 3`, `w(occupied ring) ≈ 2`,
   `η ≈ 0.69`). Compute fake-defense loss
   `FD = Σ_{q ∈ zone_any} max(0, D_nom - D_free)`. Partition the
   immediate king ring into `live`, `sealed` and `blocked` escape
   counts. Capture current-check severity as the weighted attack
   mass on the king square itself. Sum front-zone net pressure for
   the head. Produce a log reply-capacity proxy
   `reply_proxy = log(1 + live + α1·sealed + α2·blocked + α3·ring_free_defense)`.
9. **Operator vector.** Concatenate
   `[S_us, S_them, S_us - S_them, |S_us - S_them|]` for a final
   32-dim feature vector. Sanitise NaN / inf for empty positions.
10. **Delta head.** `LayerNorm + Linear + GELU + Dropout + Linear`
    on `cat(joint, operator_vector)` to a scalar
    `primitive_delta_raw`.
11. **Gate.** MLP over `cat(joint, |operator| mean)` to a sigmoid
    `primitive_gate`; initial bias `gate_init = -2.0` so the
    primitive starts near-closed.
12. **Output.** `final_logit = base_logit + primitive_gate *
    primitive_delta_raw`.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full KZRP architecture (default). |
| `no_front_zone` | **Primary falsifier.** Drop the forward-rank term `η · Σ_{q ∈ Z_front} [...]`. If lift survives, the front-rank extension is not load-bearing and the operator collapses to a ring-only scalar. |
| `no_pins` | Set the pin indicator `π = 0` everywhere; `D_free = D_nom`. Tests whether the nominal-vs-free defender split is load-bearing. |
| `uniform_zone_weights` | Replace `(king_sq, empty_ring, occupied_ring)` weights with uniform 1.0. Tests whether unequal weighting matters. |
| `no_escape_decomp` | Collapse the three escape-class counts (live, sealed, blocked) into a single total. Tests whether the decomposition is load-bearing. |
| `uniform_units` | Set the attack-unit field `u = 1` for all piece types. Tests CPW-style attack-unit weighting. |
| `no_asymmetry` | Zero `S_them`. Tests whether the side-to-move asymmetry `S_us - S_them` is load-bearing. |
| `zero_delta` | Hold `primitive_delta = 0`. Recovers i193 numerically. |
| `trunk_only` | Same as `zero_delta`; semantic alias. |
| `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, castling and en-passant
planes, and any report-only metadata are **not** consumed by the
model. The operator depends only on the simple_18 piece-presence
planes plus the side-to-move plane.

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder. |
| Trunk joint refeat | One additional encoder pass for the joint feature. |
| Attack masks (per colour, nominal and free) | Four `(B, 64, 64)` `einsum` reductions gated by an `(B, 64, 64)` between-mask. |
| Pin detector | `O(B · 8 · 64 · 7)` plus a `scatter_add` along ray targets. |
| Side vector | Per-square reductions on a handful of `(B, 64)` tensors and three `(B, 64)` × `(64, 64)` einsums for ring / front projection. |
| Delta head + gate | Small MLPs. |

There are **no training-time Python loops** over relations or
sources in the core. The pin detector contains an `O(8)`
`for d in range(NUM_DIRECTIONS)` for per-direction gathers, but no
per-batch / per-source loops. Geometry buffers carry zero
parameters; the head adds ~4k parameters at defaults plus ~10 small
scalar parameters in the builder.

## Implementation Binding

- Registered model name: `king_zone_reply_pressure`.
- Source implementation: `src/chess_nn_playground/models/primitives/king_zone_reply_pressure.py`.
- Shared ray geometry: `src/chess_nn_playground/models/primitives/ray_geometry.py`.
- Shared trunk helper: `src/chess_nn_playground/models/primitives/trunk_features.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p051_king_zone_reply_pressure/model.py`.
- Training config: `ideas/registry/p051_king_zone_reply_pressure/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/_registry_manifest.py`:
  `'king_zone_reply_pressure': ('chess_nn_playground.models.primitives.king_zone_reply_pressure', 'build_king_zone_reply_pressure_from_config')`.
- Source research markdown: `ideas/research/primitives/external_46_king_zone_reply_pressure_primitive.md`.
