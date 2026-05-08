# Architecture

`Exchange-Soundness Graph Network` is a board-only classifier for the
`puzzle_binary` task. It accepts the repository's `simple_18`
current-board tensor with shape `(B, 18, 8, 8)` and routes the puzzle
logit through an explicit, board-only differentiable form of static
exchange evaluation (SEE) on a learned attack/defense graph. The
graph -- attacker/defender intensities and cheapest-attacker /
cheapest-defender values per square -- is the operational form of
the packet thesis that a real puzzle signal must be *exchange-sound*,
not just an apparent attack.

## Mechanism

1. **Board trunk.** `BoardConvStem(input_channels=18, channels, depth,
   use_batchnorm)` produces `feats` of shape `(B, channels, 8, 8)`.

2. **Per-piece-type masks.** The side-to-move plane (`board[:, 12]`)
   selects own / opp piece planes per row. We carry six own-piece
   planes and six opp-piece planes (`P, N, B, R, Q, K`).

3. **Graph context.** A 14-channel context stack `[own_t (6),
   opp_t (6), own_all, opp_all]` is concatenated to `feats` and read
   by every graph head.

4. **Attack/defense graph heads.** Four `3x3 -> 1x1` conv heads share
   the same context input and produce, per square:
   - `attacker_logit`: `sigmoid` is the attacker intensity `p_a`.
   - `defender_logit`: `sigmoid` is the defender intensity `p_d`.
   - `attacker_type_logits`: 6 channels softmaxed over the side-to-
     move's *available* piece types -> `v_a` (cheapest-attacker
     value field).
   - `defender_type_logits`: 6 channels softmaxed over the side-not-
     to-move's available piece types -> `v_d`.
   "Available" means the side has at least one piece of that type
   anywhere on the board; the softmax masks unavailable types to
   `-inf` so the cheapest-attacker value cannot be drawn from a piece
   the side does not actually own.

5. **Target value field.** `v_target(s)` is the value of the
   opponent piece occupying square `s` (zero off opp squares),
   computed exactly from the input planes by dotting `opp_t (6)` with
   the standard piece-value buffer `[1, 3, 3, 5, 9, 12]`.

6. **Bounded-depth differentiable SEE.** For each square `s` we
   unroll the alternating capture-and-recapture sequence to depth
   `exchange_depth` (default 4):

   ```text
   see(s) = v_target
            - p_d * max(0, v_a
                            - p_a * max(0, v_d
                                            - p_d * max(0, v_a)))
   ```

   The `max(0, .)` is the SEE "stop here" rule -- the side considering
   the next capture only takes if continuing is non-negative for
   them. `exchange_score_field` is `see(s)` and
   `exchange_soundness_field = sigmoid(see / T)`.

7. **Bottleneck pool.** Trunk features are pooled through
   `target_mask * |see|` (normalised across the 64 squares):

   ```text
   exchange_pool = sum_squares( w(s) * feats(s) )    # (B, channels)
   ```

   Positions whose tactic hinges on one decisive target pull most
   feature mass from that target; positions with many roughly equal
   targets spread the pool. Rows with no opponent pieces fall back to
   a uniform-over-board pool for stability.

8. **Attacker / defender pools.** Two more pools normalise the
   attacker / defender intensities across all 64 squares and pool
   trunk features through them.

9. **Graph-network summary scalars.** An eight-vector

   ```text
   [max_see_target, mean_see_target, frac_unsound_targets,
    graph_pressure, reply_pressure, defense_gap,
    transport_imbalance, sheaf_tension]
   ```

   summarises the SEE field over the opp-piece target squares,
   together with the attack/defense intensity imbalance. These are
   the named diagnostics from the active proposal profiles
   `logic`, `graph`, `defender_reply`.

10. **Head.** A `LayerNorm + Linear + GELU + Dropout + Linear` head
    consumes `[exchange_pool, attacker_pool, defender_pool, summary]`
    and produces the puzzle logit.

At inference the model is a single-board single-logit puzzle
classifier compatible with the repository BCE-with-logits
`puzzle_binary` trainer.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for the
repository `puzzle_binary` BCE-with-logits trainer (or
`(B, num_classes)` when `num_classes > 1`):

- `logits`: `(B,)` puzzle logit.
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `attacker_intensity`: `(B, 8, 8)` `p_a` field.
- `defender_intensity`: `(B, 8, 8)` `p_d` field.
- `attacker_value_field`: `(B, 8, 8)` cheapest-attacker value `v_a`.
- `defender_value_field`: `(B, 8, 8)` cheapest-defender value `v_d`.
- `target_value_field`: `(B, 8, 8)` opp-piece value at each square.
- `exchange_score_field`: `(B, 8, 8)` differentiable SEE field.
- `exchange_soundness_field`: `(B, 8, 8)` `sigmoid(see / T)`.
- `target_mask`: `(B, 8, 8)` indicator of opp-piece squares.
- `max_see_target`: `(B,)` largest SEE among opp-piece squares.
- `mean_see_target`: `(B,)` average SEE over opp-piece squares.
- `frac_unsound_targets`: `(B,)` fraction of opp-piece squares whose
  SEE is non-positive.
- `graph_pressure`: `(B,)` mean attacker intensity over targets.
- `reply_pressure`: `(B,)` `defender / (attacker + 1)` over targets.
- `defense_gap`: `(B,)` mean `p_a - p_d` over targets.
- `transport_imbalance`: `(B,)` mean `p_a - p_d` over the whole board.
- `sheaf_tension`: `(B,)` mean `|see|` over targets.
- `target_count`: `(B,)` number of opp-piece squares.
- `trunk_energy`: `(B,)` mean-square trunk activation.

## Implementation Binding

- Registered model name: `exchange_soundness_graph_network`
- Source implementation file: `src/chess_nn_playground/models/exchange_soundness_graph_network.py`
- Idea-local wrapper: `ideas/i187_exchange_soundness_graph_network/model.py`
