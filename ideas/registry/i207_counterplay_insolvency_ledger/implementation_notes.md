# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/trunk/counterplay_insolvency_ledger.py`.
- Idea-local wrapper: `ideas/registry/i207_counterplay_insolvency_ledger/model.py` calls the registered builder.
- Registry key: `counterplay_insolvency_ledger`.
- Input contract: simple_18 board tensor only; CRTK metadata remains reporting-only.
- Output contract: one puzzle logit (shape `(B,)`) plus idea-specific diagnostic tensors.
