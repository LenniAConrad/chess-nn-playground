# Ablations

The model exposes its central falsifiers through `model.ablation`. Run the main config plus the seven ablation configs below.

- `none` (default): three streams active, chess-aware exchange/king attention bias, relative rank/file bias on the positional stream, learned phase router, aux heads on.
- `no_chess_bias`: drop the chess-aware attention biases for all three streams. Central falsifier for the attention-bias claim.
- `no_phase_router`: replace the learned softmax mixture with a uniform 1/3/1/3/1/3 mixture. Tests whether the router learns useful position-type weights.
- `remove_positional_stream`: zero the positional tower output so the head only sees exchange + king. Tests whether the structural stream contributes signal.
- `remove_king_stream`: zero the king tower output. Tests the king prior contribution.
- `remove_exchange_stream`: zero the exchange tower output. Tests the exchange prior contribution.
- `no_aux_heads`: zero the auxiliary per-stream diagnostic logits. Tests whether the aux head pathway carries signal beyond the main heads.

Compare against i193 (`exchange_then_king_dual_stream`), i242 (`chess_decomposed_attention`), LC0 BT4, and NNUE on the same split and seeds to isolate the gain from (a) adding the positional / structural stream and (b) using relative rank/file attention bias instead of a vanilla "global" attention stream.
