# Architecture

`Exchange-Then-King Dual Stream` is a board-only puzzle_binary
architecture that splits the trunk into two specialised streams and
then recombines them through a learned phase router. The thesis
(see `math_thesis.md`) is that puzzle data mixes at least two
broad families — *material-winning tactics* and
*king-safety/mate tactics* — and a single shared trunk blurs the
two. Routing through stream-specific features and a sigmoid gate
should let one branch specialise on material exchange while the
other specialises on king danger.

The model is bespoke board-only: it consumes the repository
`simple_18` current-board tensor `(B, 18, 8, 8)` and returns one
puzzle logit for the BCE-with-logits `puzzle_binary` trainer.

## Mechanism

1. **Closed-form stream feature builder.** From the `simple_18`
   planes alone, `DualStreamFeatureBuilder` computes two
   deterministic 8x8 feature stacks using fixed precomputed
   geometry tables (per-piece geometric attack tables, slider
   between-square line tables, occupancy-based slider clearance
   and a king-zone table):

   - **Exchange features** — own/enemy piece occupancy, own/enemy
     value-weighted occupancy, own/enemy attacker counts per
     square, and per-square defender / attacker pressure on own
     pieces. These mirror the packet's
     `piece/value/attacker/defender features`.
   - **King features** — own/enemy king location and king-zone
     indicators, check (own attacks reaching the enemy king
     square), enemy escape-square count (enemy king-zone squares
     not attacked by the side-to-move and not occupied), and
     own/enemy slider-line pressure into the opposing king zone.
     These mirror the packet's
     `king-zone/escape/check/line features`.
   - A small 8-dim **summary** vector (own/enemy piece counts,
     value imbalance, attack densities, king-zone sizes, check
     indicator) is also produced and injected into both heads
     plus the phase router.

2. **Two stream encoders.** Each stream concatenates the 18-plane
   board with its own 8 deterministic feature planes and runs
   through a compact `StreamEncoder` (Conv → BN/GroupNorm → GELU
   → optional Dropout2d, repeated `depth` times). Mean+max global
   pooling gives `(B, 2 * channels)` per-stream pool features.
   `exchange_pool` and `king_pool` are intentionally produced by
   *separate* encoder weights so the two streams can specialise.

3. **Per-stream logits.** Two small MLP heads
   (`LayerNorm → Linear → GELU → Dropout → Linear`) read each
   stream's pooled features concatenated with the summary vector
   and produce `exchange_logit` and `king_logit`.

4. **Phase router and residual head.** The joint vector
   `joint = [exchange_pool, king_pool, summary]` feeds a
   `phase_router` MLP that emits `gate_logit` (sigmoid → `gate`),
   and a `residual_head` MLP that emits an additive
   `residual_logit`. The puzzle output is

   ```text
   logits = gate * king_logit + (1 - gate) * exchange_logit + residual_logit.
   ```

   This is the exact recombination rule from the source packet:
   `gate = sigmoid(phase_router(board_context))` followed by
   `puzzle_logit = gate * king_logit + (1 - gate) * exchange_logit + residual_logit`.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for
the `puzzle_binary` BCE-with-logits trainer (`num_classes == 1`),
plus diagnostics:

- `logits`, `exchange_logit`, `king_logit`, `residual_logit`,
  `gate`, `gate_logit`, `gate_entropy`,
  `stream_disagreement`, `exchange_pool_norm`, `king_pool_norm`:
  `(B,)`.
- `mechanism_energy`, `proposal_profile_strength`,
  `proposal_keyword_count`: `(B,)` proposal-profile reporting
  scalars (matching the broader idea-folder reporting contract).

## Ablations

The constructor accepts one of the four ablations required by
the source packet plus the default:

- `none` — the full network described above.
- `shared_stream_only` — both streams share a single encoder and
  see only the raw 18-plane board (no stream-specific feature
  bias). Tests the value of specialisation: if both streams
  collapse onto identical trunk features, the dual-stream
  argument should weaken.
- `fixed_half_gate` — overrides `gate = 0.5` so the router cannot
  route. Tests learned routing: collapses into a uniform mixture
  of the two stream logits plus the residual.
- `king_only` — overrides `gate = 1.0` so only `king_logit` (plus
  the residual) reaches the output. Tests the mate/king subset
  bias.
- `exchange_only` — overrides `gate = 0.0` so only
  `exchange_logit` (plus the residual) reaches the output. Tests
  the material-tactic subset bias.

## Implementation Binding

- Registered model name: `exchange_then_king_dual_stream`
- Source implementation file: `src/chess_nn_playground/models/exchange_then_king_dual_stream.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i193_exchange_then_king_dual_stream/model.py`
