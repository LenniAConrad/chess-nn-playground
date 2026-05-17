# Implementation Notes — a014 BT4 Primitive Mixer (legal_move_graph_delta)

- Idea-local wrapper:
  `ideas/registry/a014_bt4_legal_move_graph_delta_mixer/model.py`.
- Registered model name: `bt4_legal_move_graph_delta_mixer`
  (resolved by the registry as an alias of `bt4_primitive_mixer` with
  `mixer = legal_move_graph_delta` — see
  `src/chess_nn_playground/models/registry.py::_make_bt4_mixer_alias`).
- Tower source:
  `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`.
- Mixer source:
  `src/chess_nn_playground/models/architecture/bt4_mixers/legal_move_graph_delta.py`.
- `implementation_kind: bespoke_model` — the tower itself is bespoke; the
  mixer swap is the variable under study.
- The wrapper passes `model` straight through to
  `build_bt4_primitive_mixer_from_config`, defaulting `mixer` and
  `num_classes` so an old config that omits them still resolves to the
  expected `legal_move_graph_delta` swap with one puzzle logit.
- Do not edit the shared BT4 tower or the mixer in this folder. Both
  are owned by `src/chess_nn_playground/models/...`; the architecture
  conformance audit will fail if the shared modules are forked in-place.
- The shared rule-graph helpers used by the mixer
  (`src/chess_nn_playground/models/primitives/rule_graph_features.py`
  and `_compute_typed_legal_edges` re-exported from
  `src/chess_nn_playground/models/primitives/legal_move_graph_delta.py`)
  build the per-type legal-move adjacency inside `torch.no_grad()`. No
  gradient flows back through the typed-edge indicators.
- Forward shape contract is checked by the shared registry parametrised
  test in `tests/test_idea_registry.py` and the BT4 mixer sibling
  test in `tests/test_research_architectures.py`.
