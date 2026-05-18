# i241 - Multi-Stream Chess-Decomposed Transformer Evaluator

Three parallel transformer towers compose i193's exchange + king dual-stream
decomposition with a third **positional / structural** stream that carries a
learnable relative rank/file attention bias. A learned phase router fuses the
three stream pools into the puzzle logit.

## Implementation Binding

- Registered model name: `multistream_attention_chess_eval`
- Source implementation file: `src/chess_nn_playground/models/trunk/multistream_attention_chess_eval.py`
- Idea-local wrapper: `ideas/registry/i241_multistream_attention_chess_eval/model.py`
- Training config: `ideas/registry/i241_multistream_attention_chess_eval/config.yaml`

The trainer is the standard puzzle_binary CRTK guarded trainer. The compact
CPU-testable variant defaults to ``embed_dim=64`` and 2 blocks per stream
(~270k parameters). The scaled engine variant in the design notes adds
value+policy heads on top of the same trunk; those heads are intentionally
not built by the puzzle_binary code path.

## Three streams

Input: ``[B, 18, 8, 8]`` simple_18 board.

```
Input
   |
   v
DualStreamFeatureBuilder (reused from i193, deterministic, no learning):
   - exchange planes (own/enemy piece, value, attack counts, attacker pressure)
   - king planes     (own/enemy king zone, check, escape, line-to-zone)
   - precomputed attack tables (used for attention bias below)

   +----------------------------+ +-----------------------+ +---------------------------+
   | Exchange tower (N blocks)  | | King tower (N blocks) | | Positional tower (N blks) |
   | Input: simple_18+exchange  | | Input: simple_18+king | | Input: simple_18          |
   | Bias:  attacker/defender   | | Bias:  king-zone pairs| | Bias:  relative rank/file |
   +-------------+--------------+ +-----------+-----------+ +-------------+-------------+
                 | mean pool                  | mean pool               | mean pool
                 v                            v                         v
            ex_pool [B,d]                kg_pool [B,d]             po_pool [B,d]

   Phase router (MLP):  softmax over (alpha_ex, alpha_kg, alpha_po)

   Final puzzle logit =   alpha_ex * exchange_head(ex_pool)
                        + alpha_kg *     king_head(kg_pool)
                        + alpha_po * positional_head(po_pool)
                        + residual_head(concat(ex_pool, kg_pool, po_pool))

   Per-stream auxiliary diagnostic logits live alongside the main heads
   for sanity-check / scaled-engine aux supervision.
```

## Stream specialisation

- **Exchange tower** sees the simple_18 board concatenated with 8 deterministic
  exchange planes (own/enemy piece, material value, attacker/defender pressure).
  Attention is biased by a per-batch ``[B, 64, 64]`` matrix that is large for
  square pairs connected by attacker-defender relationships (computed from the
  feature builder's attack tables).
- **King tower** sees the simple_18 board concatenated with 8 deterministic
  king planes (own/enemy king zone, check-ray, escape squares, line-to-zone
  pressure). Attention is biased toward pairs of squares where either belongs
  to a king's 8-ring zone.
- **Positional tower** sees the raw simple_18 board. Its attention carries a
  learnable **relative rank/file** positional bias indexed by per-head
  ``(num_heads, 15, 15)`` tables; this gives the structural / pawn-skeleton /
  prophylactic signal a place to live without any piece-specific tactical
  prior. The relative-position bias is what distinguishes this stream from
  i242's vanilla "global" stream.

## Phase router and fusion

The phase router is an MLP from ``concat(ex_pool, kg_pool, po_pool)`` to a
3-vector of logits, softmaxed to a mixture ``alpha = softmax(MLP(joint))`` over
the three streams. The puzzle logit is
``alpha_ex * ex_logit + alpha_kg * kg_logit + alpha_po * po_logit + residual_head(joint)``.

A separate auxiliary head per stream produces a stream-specific diagnostic
logit (``exchange_aux_logit``, ``king_aux_logit``, ``positional_aux_logit``).
The ``aux_loss_weight`` config knob is exposed as a diagnostic broadcast scalar
so downstream scaled trainers can multiply per-stream aux losses by it; the
puzzle_binary trainer ignores the field.

## Ablation modes

``MultistreamAttentionChessEval.ABLATIONS`` enumerates the testable variants:

- ``none`` (default): three streams, chess-aware bias on exchange/king, learned
  phase router, aux heads on.
- ``no_chess_bias``: drop the exchange/king/positional attention biases.
  Central control: does the chess-aware bias matter at all?
- ``no_phase_router``: replace the learned softmax mixture with a uniform
  1/3/1/3/1/3 mixture. Tests whether the router learns useful position-type
  weights.
- ``remove_positional_stream``: zero the positional tower output. Tests whether
  the structural stream actually contributes signal.
- ``remove_king_stream``: zero the king tower output.
- ``remove_exchange_stream``: zero the exchange tower output.
- ``no_aux_heads``: zero the auxiliary diagnostic logits. Tests whether the
  aux heads carry signal beyond the main heads.

## Diagnostics

``forward(x)`` returns:

- ``logits``: ``(B,)``, BCE-compatible for the one-logit puzzle_binary head.
- ``prob``: ``sigmoid(logits)``.
- Per-stream main logits: ``exchange_logit``, ``king_logit``, ``positional_logit``.
- Mixture weights: ``alpha_exchange``, ``alpha_king``, ``alpha_positional``.
- ``residual_logit``, ``route_entropy``, ``stream_disagreement``.
- Per-stream pool norms: ``exchange_pool_norm``, ``king_pool_norm``, ``positional_pool_norm``.
- Per-stream auxiliary logits: ``exchange_aux_logit``, ``king_aux_logit``, ``positional_aux_logit``.
- ``aux_loss_weight``: broadcast scalar of the configured aux loss weight.
- ``mechanism_energy``, ``proposal_profile_strength``, ``proposal_keyword_count``.
- ``multistream_ablation``: integer code identifying the active ablation mode.
- ``multistream_stream_count``: scalar reporting the active stream count (3).

## Contract

- Input: ``(B, 18, 8, 8)`` board tensor only.
- Output: dict with ``logits`` of shape ``(B,)`` for the one-logit puzzle_binary
  BCE-with-logits trainer, plus the diagnostics listed above.
- Target mapping: fine labels ``0`` and ``1`` map to binary target ``0``; fine
  label ``2`` maps to binary target ``1``.
- The trunk operates on tokens of shape ``(B, 64, embed_dim)``; the phase
  router mixes the three pooled stream embeddings; the puzzle decision flows
  through the mixture-of-streams logit plus a residual head reading the joint
  pool.
- Engine value+policy heads are out of scope for this puzzle_binary
  implementation and live as design notes only.
