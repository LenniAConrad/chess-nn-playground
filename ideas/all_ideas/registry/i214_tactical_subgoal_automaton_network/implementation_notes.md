# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/tactical_subgoal_automaton.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i214_tactical_subgoal_automaton_network/model.py` calls the registered builder.
- Registry key: `tactical_subgoal_automaton_network`.
- Input contract: simple_18 board tensor only; CRTK metadata remains reporting-only.
- Output contract: one puzzle logit (shape `(B,)`) plus idea-specific diagnostic tensors.
