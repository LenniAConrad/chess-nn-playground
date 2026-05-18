# Architecture

`Learned Relation Confidence Sheaf` is a narrow extension of i018
`oriented_tactical_sheaf_laplacian`. It keeps i018's adapter, tactical
incidence builder, square encoder, sheaf parameters, triad pool, readout
head, and diagnostics. It inserts a single new stage between the incidence
builder and the sheaf blocks: a board-only edge feature builder plus a
grouped confidence MLP whose output is normalized within each relation so
the head starts as identity and the network is numerically equivalent to
i018 at zero-init.

## Implementation Binding

- Registered model name: `learned_relation_confidence_sheaf`
- Source implementation: `src/chess_nn_playground/models/trunk/learned_relation_confidence_sheaf.py`
- Idea-local wrapper: `ideas/registry/i250_learned_relation_confidence_sheaf/model.py`
- Parent class: `OrientedTacticalSheafNet` from i018.

## Source research

Source packet:
`ideas/research/packets/classic/i250_learned_relation_confidence_sheaf.md`.
The packet argues for keeping i018's exact 12-relation topology and learning
which exact edges inside each relation family matter on this board, rather
than replacing the topology with generic attention.

## Modules

`BoardStateAdapter`, `TacticalIncidenceBuilder`, `SquareTokenEncoder`,
`SheafDiffusionBlock`, `TriadDefectPool`, and the readout head are inherited
unchanged from `oriented_tactical_sheaf.py`. Only the new stage below is
introduced.

`RelationEdgeFeatureBuilder` produces a deterministic, board-only feature
tensor of shape `(B, R, 64, 64, 9)` for every potential edge. Features are
derived from the same `piece_state`, `occupancy`, and `relation_masks` that
i018 already computes. No tactic tags, no source metadata, no engine values
ever enter the feature builder. The nine feature channels are:

- normalized source-piece heuristic value (`/9`),
- normalized destination-piece heuristic value (`/9`),
- product of source and destination piece value,
- Chebyshev distance between source and destination square (in `[0, 1]`),
- normalized destination in-degree (`/8`),
- normalized source out-degree (`/8`),
- king-zone destination flag (1 on king-zone relations where the edge is active),
- pin flag (lifted from `incidence.pin_mask`, intensified on the pin relation),
- x-ray flag (target sits on any pin line).

`GroupedRelationConfidence` scores ONLY the active i018 edges. The 12
relations are grouped into five semantic confidence groups:

- direct combat (`us_attacks_them_piece`, `them_attacks_us_piece`, `us_defends_us_piece`, `them_defends_them_piece`)
- king-zone pressure (`us_attacks_empty_near_king`, `them_attacks_empty_near_king`)
- visible rays (`bishop_ray_visible`, `rook_ray_visible`, `queen_ray_visible`)
- leapers and pawns (`knight_attack`, `pawn_attack_forward_oriented`)
- pin geometry (`king_ray_pin_candidate`)

Each group has its own small MLP (`Linear -> GELU -> Linear`). The MLP
input is the concatenation of the per-edge feature tensor, a small learned
relation embedding, low-rank source and destination context projections of
the i018 square tokens `h0`, and their Hadamard product. The output is one
scalar per edge per relation. A learned per-relation bias is added.

The raw confidence is

```text
raw = floor + (1 - floor) * sigmoid(logit + relation_bias)
```

with `floor in (0, 0.05]`, masked by the active relation edges. The
normalized confidence is

```text
alpha_hat = raw / mean_active(raw)
```

with the mean taken over each `(batch, relation)` plane over only the
already-active edges of i018. The output layer of every group head is
zero-initialized, so at initialization `raw` is constant on active edges
and `alpha_hat` is exactly `1.0`. The network in that state is i018.

`LearnedRelationConfidenceSheafNet` overrides only `forward`. It runs
`adapter`, `incidence`, `encoder` exactly like i018, applies the new
confidence stage to produce `alpha_hat`, multiplies the i018 relation masks
by `alpha_hat`, and then feeds the confidence-weighted masks into the
existing sheaf blocks. The triad pool, readout, and diagnostic dictionary
are unchanged, except that five new confidence-attribution scalars are
appended: `confidence_mean`, `confidence_max`, `confidence_std`,
`pin_edge_confidence`, and `king_zone_confidence`.

## Optional ablations

- `flat_confidence: true` forces `alpha_hat = relation_mask`. This is an
  exact-i018 path for falsifier comparison and is faster than i018-only
  evaluation because we skip building edge features.
- `normalize_confidence_within_relation: false` lets confidence absorb
  relation-level mass instead of redistributing it. This is the identifiability
  ablation called out in the source packet.

## Contract

- Input: `(B, 18, 8, 8)` board tensor only; same `simple_18` encoding as
  i018. CRTK / source / verification metadata is reporting-only and never
  enters the model.
- Output: `dict` with `logits` of shape `(B,)` and the i018 diagnostic
  bundle plus the five new confidence-attribution scalars.
- Symmetry: only the side-to-move canonicalization (color swap + 180-degree
  rotation) is applied, exactly as in i018.

## Numerical guarantees

With shared weights and `flat_confidence: true`, i250 reproduces i018
logits exactly on CPU. With the default normalized head and zero
initialization, i250 reproduces i018 logits within FP32 reduction noise
(observed max abs logit difference of about `6e-8` on a 4-sample CPU
batch). If a future edit pushes that drift above `1e-5` at zero-init, treat
i250 as a changed model rather than a strict extension of i018.
