# Architecture

`Candidate Move Forcedness Sheaf` wraps i018
`oriented_tactical_sheaf_laplacian` with a small, deterministic
candidate-move bottleneck. It keeps i018's adapter, tactical incidence
builder, square encoder, sheaf parameters, triad pool, readout head, and
diagnostic bundle. It adds a deterministic move builder, a per-move
encoder, a top-k softmax pool, and a single gated additive delta head.
The delta and gate output layers are zero-initialized so the network is
numerically identical to i018 at initialization.

## Implementation Binding

- Registered model name: `candidate_move_forcedness_sheaf`
- Source implementation: `src/chess_nn_playground/models/trunk/candidate_move_forcedness_sheaf.py`
- Idea-local wrapper: `ideas/registry/i251_candidate_move_forcedness_sheaf/model.py`
- Parent class: `OrientedTacticalSheafNet` from i018.

## Source research

Source packet:
`ideas/research/packets/classic/i251_candidate_move_forcedness_sheaf.md`.
The packet argues for keeping i018's static oriented tactical sheaf and
wrapping it with the smallest move bottleneck that can express
"this position contains a single move whose local structural evidence
sharply dominates the alternatives."

## Modules

`BoardStateAdapter`, `TacticalIncidenceBuilder`, `SquareTokenEncoder`,
`SheafDiffusionBlock`, `TriadDefectPool`, and the readout head are
inherited unchanged from `oriented_tactical_sheaf.py`. Only the modules
below are introduced.

`CandidateMoveBuilder` enumerates a bounded pseudo-legal move set from
the canonical mover-oriented `piece_state` and `occupancy`. It reuses
i018's precomputed `knight`, `king`, `rook_ray`, `bishop_ray`, `between`,
and pawn geometry buffers. Knight and king moves come from the
single-step geometry; bishop, rook, and queen moves use the visibility
already computed by i018's `between`-based blocker product (so a slide
stops at the first blocker, with captures of enemy pieces counted as
valid edges); pawn pushes (single + double from the second rank in the
canonical frame) and diagonal captures use the mover-oriented pawn
geometry. Edges are scored by a deterministic priority that favors
captures and edges adjacent to attack pressure, and the top
`max_candidates` (default 96) of these are kept. The remaining slots are
zero-padded with `mask = 0`. The builder also produces deterministic
per-move flags: `gives_check`, `is_capture`, `source_pinned`,
`pin_aligned`, `enters_their_king_zone`, `target_defended_raw`,
`target_defended_unpinned`, `promotion`, and `underpromotion`. Castling
and en-passant are not enumerated in the default builder.

`MoveLocalSheafSummary` reads the per-relation in/out degrees at the
source and target squares of every candidate (`4 * R = 48` scalars). It
is the move encoder's only window into the sheaf state's tactical
pressure at the squares the candidate touches.

`CandidateMoveEncoder` is a shared per-move MLP. For each candidate it
concatenates the source square state `h_s`, target square state `h_t`,
the difference `h_t - h_s`, the deterministic flags, a one-hot kind
vector, and the local sheaf summary. The MLP emits a 48-dim move
embedding and a single score per move. The score head is
zero-initialized so all valid moves start with the same score.

`TopKMovePool` keeps the top `top_k` (default 8) scored moves and
applies a learned-temperature softmax to obtain pool weights. It pools
the move embeddings, computes `entropy`, `top1_mass`, `gap`, per-flag
masses (check, capture, pin, king-zone, promotion, underpromotion), and
a continuous `top_move_kind` summary. Padding slots are masked away by
forcing `score = -inf` before softmax.

The final fusion is

```text
final_logit = base_logit + sigmoid(gate(features)) * delta(features),
```

with `features` concatenating the pooled move embedding, the 11
forcedness scalars, 8 trunk-context scalars (sheaf tension,
ray language energy, triad defect energy, pin pressure, king ring
pressure, transport imbalance, defense gap, reply pressure), and a
continuous top-move kind onehot. Both `delta` and `gate` are small MLPs
with zero-initialized output layers, so `final_logit == base_logit` at
init.

## Optional ablations

- `flat_move_pool: true` forces uniform pool weights over the valid
  candidates so the move branch sees a pooled embedding but no
  forcedness signal. Used to test whether the top-k bottleneck is
  load-bearing.
- `disable_move_branch: true` skips the move branch entirely; the model
  becomes an exact i018 forward.
- `max_candidates`, `top_k`, `softmax_temperature` cover budget and
  sparsity sweeps.

## Contract

- Input: `(B, 18, 8, 8)` board tensor only; same `simple_18` encoding as
  i018. CRTK / source / verification metadata is reporting-only and
  never enters the model.
- Output: `dict` with `logits` of shape `(B,)` and the i018 diagnostic
  bundle plus 14 new candidate-move diagnostics
  (`candidate_base_logits`, `candidate_delta_logits`, `candidate_gate`,
  `candidate_entropy`, `candidate_top1_mass`, `candidate_gap`,
  `candidate_check_mass`, `candidate_promotion_mass`,
  `candidate_underpromotion_mass`, `candidate_pin_mass`,
  `candidate_capture_mass`, `candidate_king_zone_mass`,
  `candidate_overflow_count`, `candidate_count`).
- Symmetry: only the side-to-move canonicalization (color swap +
  180-degree rotation) is applied, exactly as in i018.

## Numerical guarantees

With shared weights and the default `disable_move_branch: false` plus
zero-init delta+gate, i251 reproduces i018 logits exactly on CPU on a
4-sample batch (observed max abs logit diff `0.0`). With
`disable_move_branch: true` and shared weights, i251 reproduces i018
logits bit-exactly. If a future edit pushes the zero-init drift above
`1e-5`, treat i251 as a changed model rather than a strict extension of
i018.
