# Architecture

`Tactical Program Induction Network` treats a puzzle as the existence
of a *tiny latent program* on the side-to-move's pieces and squares.
The network is board-only over the repository `simple_18`
current-board tensor `(B, 18, 8, 8)` and returns one puzzle logit for
the BCE-with-logits `puzzle_binary` trainer.

## Mechanism

1. **Typed fact extractor.** From the `simple_18` planes the model
   computes, in closed form (no learned parameters), a typed
   relational fact tensor

   ```text
   op_facts: (B, OP_COUNT, 64, 64)
   ```

   over `OP_COUNT = 8` operation labels:

   ```text
   threaten, pin, deflect, overload, fork,
   clear_line, trap_king, win_target.
   ```

   These are derived from piece planes plus a fixed precomputed
   geometry (geometric attack tables, between-square line tables, a
   king-zone table, slider clearance from the occupancy). The
   side-to-move plane decides which color is "own" and which is
   "enemy" for every batch row. A summary vector of 18 scalar
   diagnostics (own/enemy material, attack densities, king-zone
   pressure, loose / underdefended / deflectable / overloaded target
   counts, fork-source mass, pin / clear-line / trap-king / win-target
   masses, total relation-fact count, material balance) is also
   produced.

2. **Board encoder.** A compact convolutional square encoder
   (`ProgramBoardEncoder`) reads the 18-plane board with two extra
   coordinate planes and produces:

   - `tokens`: `(B, 64, token_dim)` square tokens,
   - `global_context`: `(B, token_dim)` mean+max pooled trunk
     features,
   - `board_map`: `(B, channels, 8, 8)` raw trunk features (used for
     a board activation energy diagnostic).

3. **Latent program state.** A latent program state vector
   `state ∈ R^{token_dim}` is initialised from the trunk's global
   context and the typed fact summary. A learned per-step embedding
   biases the state at each program step.

4. **Program induction loop.** For each step `k = 0, ..., K-1`
   (`program_steps`, default 4) the model induces the next operation
   in the latent program:

   - **Operation selector.** A small MLP over `[step_state,
     global_context]` produces a categorical
     `op_probs ∈ Δ^{OP_COUNT}` over typed operations.
   - **Source / target selectors.** Two scaled-dot-product
     attentions over `tokens` produce
     `source_probs, target_probs ∈ Δ^{64}` square distributions. The
     attention logits are biased by `log(own_piece_prior)` for the
     source and `log(enemy_or_zone_prior)` for the target so the
     program is forced to act on real own pieces and on real
     opponent pieces / king-zone squares before it can be trusted.
   - **Expected typed relation.** With `E[op_facts | source,
     target]` the program asks: "given the chosen source and target,
     how strongly is each typed operation supported by the typed
     fact tensor?"
     `expected_by_op = einsum('bost, bs, bt -> bo', op_facts,
     source_probs, target_probs)`. The selected step's
     **relation score** is `selected_relation = sum_o(op_probs *
     expected_by_op)`.
   - **Pre / post conditions.** Two small MLPs over the step
     context produce `pre_score, post_score ∈ [0, 1]`. `pre_score`
     scores the precondition strength of the chosen op at the
     chosen source/target *before* execution; `post_score` scores
     the postcondition strength of the same op after the latent
     executor has updated the square tokens.
   - **Latent executor.** `executor_layers` LSTM-free executor
     blocks (`LatentExecutorBlock`) update the per-square `tokens`
     by mixing in a learned op/source/target-conditioned `delta`
     gated by source/target/relation evidence. This is the
     differentiable analogue of "applying" the chosen typed
     operation to the latent board.
   - **State update.** A GRUCell folds the executed step's
     summary into the running latent program state.

5. **Aggregate program coherence.** The model assembles per-step
   `step_precondition`, `step_postcondition`, `step_relation` scores
   and combines them into a **program coherence** scalar per batch
   row,

   ```text
   program_log_coherence = mean_k log(pre_k) + log(post_k) + log(relation_k)
   program_coherence     = exp(program_log_coherence)
   ```

   plus mean precondition / postcondition / relation scores, mean
   operation entropy, and an operation histogram (mean over steps of
   `op_probs`). The puzzle thesis is that real puzzles admit *one*
   program with high pre, post and typed-relation evidence, and
   non-puzzles do not.

6. **Head.** A small `LayerNorm + Linear + GELU + Dropout + Linear +
   GELU + Linear` MLP head consumes

   ```text
   [global_context, initial_token_mean, final_token_mean,
    final_token_max, fact_summary, operation_histogram,
    program_coherence, precondition_score, postcondition_score,
    relation_coherence, operation_entropy, step_coherence_mean]
   ```

   and produces the puzzle logit `(B,)` for the
   BCE-with-logits trainer.

## Ablations

The constructor accepts one of:

- `none` -- the full network described above.
- `bag_of_ops_no_order` -- step state is reset to `initial_state` at
  every step and the GRU state update is disabled, so the program
  becomes an order-free bag of ops.
- `one_step_program` -- only `K = 1` step is executed; the remaining
  step slots are filled with uniform / zero pads.
- `no_precondition_scores` -- the precondition head is forced to
  `1.0` so the program can never be rejected by a missing
  precondition.
- `random_op_labels` -- the typed `op_facts` axis is permuted by a
  fixed permutation, scrambling the typed-relation grounding
  between op labels and their evidence.

These exist to test that program structure (typed ops, ordered
steps, pre/post evidence, relation grounding) is what produces the
puzzle signal.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for
the `puzzle_binary` BCE-with-logits trainer
(`num_classes == 1`), plus diagnostics:

- `logits`: `(B,)` puzzle logit.
- `program_coherence`, `program_log_coherence`,
  `precondition_score`, `postcondition_score`,
  `relation_coherence`, `operation_entropy`: `(B,)` program-level
  coherence summaries.
- `step_coherence`, `step_precondition_scores`,
  `step_postcondition_scores`, `step_relation_scores`:
  `(B, program_steps)` per-step program scores.
- `operation_probs`: `(B, program_steps, OP_COUNT)`.
- `primary_piece_probs`, `target_square_probs`:
  `(B, program_steps, 64)` per-step source / target distributions.
- `mechanism_energy`, `proposal_profile_strength`,
  `proposal_keyword_count`, `relation_fact_count`,
  `material_balance`, `board_activation_energy`: scalar
  diagnostics matching the proposal-profile reporting contract.
- `op_<name>_strength`, `op_<name>_mass` for every name in
  `(threaten, pin, deflect, overload, fork, clear_line, trap_king,
  win_target)`: per-op strength from the typed fact extractor and
  per-op mass from the induced program histogram.

## Implementation Binding

- Registered model name: `tactical_program_induction_network`
- Source implementation file: `src/chess_nn_playground/models/tactical_program_induction.py`
- Idea-local wrapper: `ideas/i188_tactical_program_induction_network/model.py`
