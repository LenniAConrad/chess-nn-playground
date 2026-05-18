# Architecture

`Efficient i018 Scale-XXL` (i254) is the capacity-scaled successor to
i018 `oriented_tactical_sheaf_laplacian`. It keeps the i018 board
adapter, exact 12-relation tactical incidence builder, square encoder,
triad pool, readout head, falsifier scramble knob, and the entire
diagnostic contract unchanged. The only architectural change is a new
`EfficientSheafDiffusionBlock` that supports two restriction modes:
`full` (default, identical to i018) and an optional `grouped_lowrank`
structured parameterization for a later stalk-scaling experiment.

The source research markdown is
`ideas/research/packets/classic/i254_efficient_i018_scale_xxl.md`; this
folder is the implementation promotion of that packet.

## Thesis (one paragraph)

The repo evidence says i018 improves from base to scale_xl, the
falsifier shows that the typed chess relation graph is load-bearing,
and i249 failed because it bundled speculative execution changes with
an execution rewrite that was only checked in eval-mode fp32 rather
than under the actual training path. The efficient XXL design should
therefore **keep the relation-builder thesis fixed, scale the regular
dense parts first, keep the sheaf core numerically conservative, and
gate all speed work behind profiler evidence**. The default first XXL
run is a width-and-head capacity probe, not a fused execution rewrite.

## What scales and what stays fixed

| Component | i254 first XXL action | Why |
|---|---|---|
| `BoardStateAdapter` | unchanged | preserves the mover-oriented thesis |
| `TacticalIncidenceBuilder` | unchanged | preserves the 12-relation chess geometry validated by the i018 falsifier |
| Relation count `R=12` | unchanged | adding relations changes the thesis instead of scaling it |
| Stalk dimension `s=8` | unchanged in the first run | stalk-only scaling increases the expensive `64*64*s` sheaf work fastest |
| Token width `channels` | **scale 128 -> 160** | regular dense capacity, historically helped i018 |
| Readout hidden size `hidden_dim` | **scale 192 -> 320** | dense capacity with very small extra irregular work |
| Sheaf depth | unchanged at 4 | every extra block multiplies relation work |
| Compiled/fused incidence | flags exposed but default off | i249 is a direct warning against bundling speculative speed work with capacity changes |

## Mechanism

1. **Inherited i018 trunk.** `OrientedTacticalSheafEfficientXXLNet`
   subclasses `OrientedTacticalSheafNet` and inherits its
   `BoardStateAdapter`, `TacticalIncidenceBuilder`, `SquareTokenEncoder`,
   `TriadDefectPool`, readout head, falsifier `scramble_relations`
   path, and full diagnostic contract.

2. **EfficientSheafDiffusionBlock**. The block has the same
   `(h, relation_masks) -> (h, energies, gates)` signature as i018's
   `SheafDiffusionBlock`. In `restriction_mode="full"` (default) the
   state_dict shape and forward computation are bit-identical to i018:
   `rho_src[r]`, `rho_dst[r]` are full `(s, s)` matrices initialised at
   `I + 0.02 * noise`, signs are the fixed `(-1, -1, +1, +1, ...)`
   pattern, gates are `2 * sigmoid(logit)`, and the heat step is
   `0.25 * sigmoid(eta_logit)`.

3. **Optional grouped low-rank parameterization**. In
   `restriction_mode="grouped_lowrank"` each restriction map is
   parameterized as
   `rho_r = I_s + U_g(r) diag(a_r) V_g(r)^T` with group-shared bases
   `U_g, V_g in R^(s x k)` and relation-specific diagonal coefficients
   `a_r in R^k`. The default 4-group partition splits the 12 typed
   relations into:

   - **attack**: us_attacks_them_piece, them_attacks_us_piece,
     us_attacks_empty_near_king, them_attacks_empty_near_king,
     knight_attack, pawn_attack_forward_oriented (group 0)
   - **defense**: us_defends_us_piece, them_defends_them_piece (group 1)
   - **ray**: bishop_ray_visible, rook_ray_visible, queen_ray_visible
     (group 2)
   - **pin**: king_ray_pin_candidate (group 3)

   The materialized `(R, s, s)` restriction tensor is recomputed each
   forward, so the per-relation loop reduces to the same matrix
   products as the full case. The point of this mode is parameter
   sharing for a later stalk-scaling experiment (`s=12` with `k=4`,
   per the research markdown), not a fused custom kernel.

4. **Profiler scopes**. The new block wraps the per-relation loop in
   `torch.profiler.record_function("i254/per_relation_loop")` and the
   full forward in `torch.profiler.record_function(
   "i254/efficient_sheaf_block")` so the recommended profile-first
   protocol can see exactly where time is spent. These scopes are
   no-ops outside `torch.profiler.profile`.

5. **Execution-branch flags**. `compile_model` and `fuse_incidence`
   are accepted by the config and stored on the model as attributes
   but do nothing in the default forward path. The research markdown
   explicitly requires the first XXL benchmark to be a capacity-only
   run; compile-only and fused-incidence work belong to a separate
   execution branch that must pass the train-mode mixed-precision
   parity ladder before being benchmarked for speed.

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata
are *not* consumed by the model. The contract is identical to i018.

## Parameter Budget

At the default first-XXL scale (`channels=160`, `hidden_dim=320`,
`depth=4`, `stalk_dim=8`, `dropout=0.1`, `restriction_mode=full`):

| Variant | Total params | vs current scale_xl |
|---|---:|---:|
| `restriction_mode=full` | 785,217 | 1.66x |
| `restriction_mode=grouped_lowrank` (G=4, k=4) | ~783,000 | 1.65x |

The full-mode count matches the research markdown's static estimate
exactly. Whether the grouped low-rank parameter saving is worth taking
depends on a later stalk-scaling experiment; at `s=8` the savings are
small in absolute terms.

## Implementation Binding

- Registered model name: `efficient_i018_scale_xxl`.
- Source implementation: `src/chess_nn_playground/models/trunk/oriented_tactical_sheaf_efficient_xxl.py`
  (`OrientedTacticalSheafEfficientXXLNet`,
  `EfficientSheafDiffusionBlock`,
  `DEFAULT_RELATION_GROUPS_4`,
  `build_oriented_tactical_sheaf_efficient_xxl_from_config`).
- Idea-local wrapper: `ideas/registry/i254_efficient_i018_scale_xxl/model.py`
  (`build_model_from_config`).
- Training config: `ideas/registry/i254_efficient_i018_scale_xxl/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/_registry_manifest.py`:
  `'efficient_i018_scale_xxl': ('chess_nn_playground.models.trunk.oriented_tactical_sheaf_efficient_xxl', 'build_oriented_tactical_sheaf_efficient_xxl_from_config')`.
- Reused i018 building blocks live in
  `src/chess_nn_playground/models/trunk/oriented_tactical_sheaf.py`
  (`BoardStateAdapter`, `TacticalIncidenceBuilder`, `SquareTokenEncoder`,
  `TriadDefectPool`, `RELATION_NAMES`, `OrientedTacticalSheafNet`).
