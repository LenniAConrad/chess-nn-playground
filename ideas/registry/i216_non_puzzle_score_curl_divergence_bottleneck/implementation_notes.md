# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/trunk/non_puzzle_score_curl_divergence.py`.
- Idea-local wrapper: `ideas/registry/i216_non_puzzle_score_curl_divergence_bottleneck/model.py` calls the registered builder.
- Registry key: `non_puzzle_score_curl_divergence_bottleneck`.
- Input contract: simple_18 board tensor only; CRTK metadata remains reporting-only.
- Output contract: one puzzle logit (shape `(B,)`) plus idea-specific diagnostic tensors.
