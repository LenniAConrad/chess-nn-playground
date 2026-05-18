# Architecture

`i018 BT4-112 Controlled Encoding` (i253) is a controlled-encoding sibling of
i018 `oriented_tactical_sheaf_laplacian`. It keeps the i018 board adapter,
exact 12-relation tactical incidence builder, square encoder, sheaf diffusion
block, triad pool, readout head, and diagnostic contract. It adds a 112-channel
raw input pathway that is identical across encodings, plus a small relation
confidence (and optional bounded augmentation) head that lets the richer BT4
planes modulate the importance of already-correct relation edges without
replacing chess geometry.

The intent is to answer one controlled question: does the current repo's
`lc0_bt4_112` exporter add enough usable board-state signal to strengthen
i018, after the exact 12 mover-oriented chess relation masks are kept fixed.

The source research markdown is
`ideas/research/packets/classic/i253_i018_bt4_112_controlled_encoding.md`;
this folder is the implementation promotion of that packet.

## Mechanism

1. **Controlled adapter (`BoardStateAdapterControlled`)**. The mover-relative
   piece-state path is exact for both encodings, exactly as in i018. The raw
   pathway is zero-padded to 112 channels for `simple_18` and used natively
   for `lc0_bt4_112`. That standardises the raw input width and parameter
   count, so the only difference between encodings is what lives in the 112
   raw channels.

2. **Exact tactical incidence**. `TacticalIncidenceAugmentationBuilder`
   delegates to the original i018 `TacticalIncidenceBuilder` for the 12
   relation masks `M_r` (attacker/defender, king-zone pressure,
   rook/bishop/queen visible rays, knight, oriented pawn, pin candidate).
   It additionally registers a static `(R, 64, 64)` tensor of relation-
   specific geometric template supersets `T_r`. Slider relations use the
   full geometric ray template; attacker/defender relations are limited
   to the slider geometric superset; knight, king, pawn, and pin-candidate
   relations keep their legal geometric shape.

3. **Relation modes** (`model.relation_mode`):

   - `exact`: `W_r = M_r`. The hard control. Richer encoding can only
     help through the node/raw feature path.
   - `confidence`: `W_r = M_r * sigmoid(C_r)`. Exact support, learned
     edge importance via a low-rank source-target interaction over the
     raw 112-channel pathway.
   - `hybrid`: `W_r = clamp(M_r * sigmoid(C_r) + lambda * T_r * sigmoid(A_r), 0, 1)`
     with `lambda = augmentation_lambda` (default 0.25). Adds a bounded
     residual on a fixed geometric superset, never a free dense 64x64
     graph.

4. **Falsifiers**:

   - `model.scramble_exact_relations: true` reuses i018's
     degree-preserving random rewiring of the 12 relation masks. In
     hybrid mode the template supersets are scrambled the same way so the
     comparison stays apples-to-apples.
   - `model.augmentation_only: true` (hybrid only) drops the exact
     relation masks and keeps only `lambda * T_r * sigmoid(A_r)`. This
     is the explicit augmentation-only row from the research markdown.

5. **RelationConfidenceHead**. Consumes the raw `(B, 64, 112)` square
   tensor and emits per-relation source/target codes of rank
   `relation_rank` plus a relation bias. Confidence and augmentation
   share the same `pre_proj` but use independent linears. Output is
   capped at `[0, 1]` via sigmoid (and a final clamp for hybrid). No
   metadata or label is consumed; the head only sees raw plane content.

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata are
*not* consumed by the model. The contract is identical to i018.

## Parameter Budget

At the base scale (`channels=64`, `hidden_dim=96`, `depth=2`,
`stalk_dim=8`, `relation_hidden=16`, `relation_rank=8`):

| Variant       | Total params | Delta vs exact-only |
|---------------|-------------:|--------------------:|
| `exact`       |       94,371 |                   - |
| `confidence`  |       99,487 |              +5,116 |
| `hybrid`      |      102,763 |              +8,392 |

These match the research markdown's targets and keep the comparison
centered on encoding content rather than brute capacity. The exact-only
row is +3,008 over the original i018 base (91,363 -> 94,371) because the
raw input projection consumes 112 channels instead of 18, but it is
identical across encodings.

## Implementation Binding

- Registered model name: `i018_bt4_112_controlled_encoding`.
- Source implementation: `src/chess_nn_playground/models/trunk/oriented_tactical_sheaf_controlled_encoding.py`
  (`OrientedTacticalSheafControlledEncodingNet`,
  `BoardStateAdapterControlled`,
  `TacticalIncidenceAugmentationBuilder`,
  `RelationConfidenceHead`,
  `build_i018_bt4_112_controlled_encoding_from_config`).
- Idea-local wrapper: `ideas/registry/i253_i018_bt4_112_controlled_encoding/model.py`
  (`build_model_from_config`).
- Training config: `ideas/registry/i253_i018_bt4_112_controlled_encoding/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/_registry_manifest.py`:
  `'i018_bt4_112_controlled_encoding': ('chess_nn_playground.models.trunk.oriented_tactical_sheaf_controlled_encoding', 'build_i018_bt4_112_controlled_encoding_from_config')`.
- Reused i018 building blocks live in
  `src/chess_nn_playground/models/trunk/oriented_tactical_sheaf.py`
  (`BoardStateAdapter`, `TacticalIncidenceBuilder`, `SquareTokenEncoder`,
  `SheafDiffusionBlock`, `TriadDefectPool`, `RELATION_NAMES`).
