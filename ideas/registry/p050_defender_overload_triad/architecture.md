## Architecture

`Defender Overload Triad` (p050, DOT) is an additive, gated head on
top of the i193 `ExchangeThenKingDualStreamNetwork` trunk. The
thesis (see `math_thesis.md`) is that the i193 trunk's 3x3
convolutions and the legacy `TriadDefectPool` (which only produces
four pooled scalars over `attack * defense * target_piece`) leave the
*defender identity reuse* signal under-exploited: which single
defender is responsible for several critical targets at once. DOT
makes that fact a 20-dim operator vector that the trunk gates and
adds.

The model consumes `simple_18` `(B, 18, 8, 8)` and returns one
puzzle logit plus a per-sample diagnostics dict.

## Forward pass

1. **i193 trunk forward.** Emits `base_logit` and trunk diagnostics
   plus the joint pool feature (recomputed via `trunk_joint_features`
   without firing the trunk's heads twice).
2. **Absolute piece state and side-to-move.** Plane indices 0..11
   stay in their original (P, N, B, R, Q, K, p, n, b, r, q, k)
   ordering; the side-to-move scalar plane 12 selects mover-vs-them
   inside the side stats helper.
3. **Attack and between buffers.** A rule-derived
   `geom_attacks (6, 2, 64, 64)` and `between (64, 64, 64)` are
   precomputed at module init. Slider rays are gated by the
   `between` clear-line mask (no occupant strictly between source
   and target). The resulting `(B, 64, 64)` per-colour attack masks
   are produced in pure `einsum` style; no Python loops at forward
   time.
4. **Per-direction enemy slider field.** For each side σ, an
   `(B, 8, 64)` per-direction slider activation is produced from
   the queen / rook / bishop planes. This is used only by the
   pin detector and re-uses the direction-family masks shared with
   p049.
5. **Cumsum-based pin detector.** Using the shared
   `ray_geometry (8, 64, 7)` index / mask tables, a per-square
   pinned-defender indicator `πσ` is computed by walking rays
   from the *own king* through *own pieces* and checking whether an
   enemy slider firing along the same direction sits as the second
   ray occupant. A `scatter_add_` projects the per-step pin marks
   back to per-square `(B, 64)`.
6. **Side stats** (per side σ ∈ {us, them}). Compute per target
   features `[a, d, p, a_val, d_val, m_att, m_def, v_tar]`:
   attack count, defense count, pinned-defender count, attacker-
   value sum, effective-defender-value sum (pin-discounted), cheapest
   unpinned defender, cheapest attacker, and target piece value.
   Run a tiny `LayerNorm + Linear + GELU + Linear` MLP and a
   `softplus` to produce target criticality `cσ(t)`. Then form the
   defender-obligation `O = D ⊙ c.unsqueeze(1)`, the per-defender
   total `L = O.sum(dim=2)`, the pin-amplified fragility
   `m = 1 + μ·π`, and the closed-form overload masses
   `Ω_def = m * (L^2 - Σ_t O^2)` and
   `X_tar = c * [D^T (m·L) - c * (D^2)^T m]`. Clamp non-negative,
   then pool to a 5-feature side vector
   `[mean(X_tar), max(X_tar), mean(Ω_def | occupied), pinned_share,
   mean(c)]`. Each `_side_stats` pass runs in `O(BN^2)`; no
   `(B, N, N, N)` tensors are materialised.
7. **Operator vector.** Concatenate
   `[S_us, S_them, S_us - S_them, |S_us - S_them|]` for a final
   20-dim feature vector. Sanitise NaN / inf for empty positions.
8. **Delta head.** `LayerNorm + Linear + GELU + Dropout + Linear`
   on `cat(joint, operator_vector)` to a scalar `primitive_delta_raw`.
9. **Gate.** MLP over `cat(joint, |operator| mean)` to a sigmoid
   `primitive_gate`; initial bias `gate_init = -2.0` so the
   primitive starts near-closed.
10. **Output.** `final_logit = base_logit + primitive_gate *
    primitive_delta_raw`.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full DOT architecture (default). |
| `no_cross_target_load` | **Primary falsifier.** Drop the cross-target overload term `L^2 - Σ O^2`; leave only single-target under-defence. If lift survives, the operator is not actually measuring defender reuse. |
| `no_pins` | Zero the pin indicator `π` everywhere. Tests whether pinned defenders are load-bearing. |
| `no_target_value` | Set `v_tar = 1` on every occupied target (and `v_att, v_def = 1`). Tests whether value weighting matters. |
| `counts_only` | Drop `a_val, d_val, m_att, m_def` from the target-criticality gate, leaving only counts and `v_tar`. Tests SEE-light feature load-bearingness. |
| `zero_delta` | Zero the primitive delta. Recovers the i193 baseline numerically. |
| `trunk_only` | Same as `zero_delta`; semantic alias. |
| `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata
are **not** consumed by the model. The operator depends only on the
simple_18 piece-presence planes plus the side-to-move plane.

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder. |
| Trunk joint refeat | One additional encoder pass for the joint feature. |
| Attack masks (per colour) | `einsum("bik,bkt->bit", source_planes, geom_attacks)`-style, gated by an `einsum("stk,bk->bst", between, occupancy)` blockers check. |
| Pin detector | `O(B·8·64·7)` plus a scatter_add along ray targets. |
| `_side_stats` per side | Two BMMs of shape `(B, N, N) x (B, N, 1)`, plus a small MLP over `(B, 64, 8)` target features. |
| Delta head + gate | Small MLPs. |

There are **no training-time Python loops** over relations, sources,
or step lengths in the overload core. The pin detector contains an
`O(8)` `for d in range(NUM_DIRECTIONS)` for per-direction gathers, but
no per-batch / per-source loops. Geometry buffers carry zero
parameters; the head adds ~3k parameters at defaults.

## Implementation Binding

- Registered model name: `defender_overload_triad`.
- Source implementation: `src/chess_nn_playground/models/primitives/defender_overload_triad.py`.
- Shared ray geometry: `src/chess_nn_playground/models/primitives/ray_geometry.py`.
- Shared trunk helper: `src/chess_nn_playground/models/primitives/trunk_features.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p050_defender_overload_triad/model.py`.
- Training config: `ideas/registry/p050_defender_overload_triad/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/_registry_manifest.py`:
  `'defender_overload_triad': ('chess_nn_playground.models.primitives.defender_overload_triad', 'build_defender_overload_triad_from_config')`.
- Source research markdown: `ideas/research/primitives/external_45_defender_overload_triad_primitive.md`.
