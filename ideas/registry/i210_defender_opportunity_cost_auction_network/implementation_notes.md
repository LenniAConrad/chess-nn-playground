# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/defender_opportunity_cost_auction.py`.
- Idea-local wrapper: `ideas/registry/i210_defender_opportunity_cost_auction_network/model.py` calls the registered builder.
- Registry key: `defender_opportunity_cost_auction_network`.
- Input contract: simple_18 board tensor only; CRTK metadata remains reporting-only.
- Output contract: one puzzle logit (shape `(B,)`) plus idea-specific diagnostic tensors.
