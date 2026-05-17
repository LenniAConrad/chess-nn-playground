# Implementation Notes — a016 BT4 Primitive Mixer (legal_edge_compile_scatter)

- Idea-local wrapper:
  `ideas/registry/a016_bt4_legal_edge_compile_scatter_mixer/model.py`.
  Calls `build_bt4_primitive_mixer_from_config` with
  `mixer = legal_edge_compile_scatter` and `num_classes = 1`.
- Registry alias: `bt4_legal_edge_compile_scatter_mixer` is resolved by
  `_make_bt4_mixer_alias("legal_edge_compile_scatter")` in
  `src/chess_nn_playground/models/registry.py`; no manifest entry is
  required.
- Mixer source:
  `src/chess_nn_playground/models/architecture/bt4_mixers/legal_edge_compile_scatter.py`
  (registered via `@register_mixer("legal_edge_compile_scatter")`).
- Tower source:
  `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`.
- Source primitive idea: `p011_legal_edge_compile_scatter` — the typed
  σ-gate, per-type message linear, and gate-weighted scatter are
  ported faithfully; the legal-move adjacency is replaced by the
  geometry-only move-pattern skeleton because the mixer receives
  arbitrary `C` channels, not simple_18 piece planes.
- The geometric adjacency tensor is a non-persistent buffer (`edges`);
  it stays constant across the batch and is built once at module
  init. There is no per-batch python-chess work.
- `implementation_kind: bespoke_model`. The detection in
  `audit_implementation_kinds` resolves through the alias to the
  bespoke `BT4PrimitiveMixerNet` builder, not the
  `ResearchPacketProbe` shared scaffold.
- All other a###_bt4_*_mixer ideas use the same tower wrapper; the only
  variable is `model.mixer`. Keep `config.yaml` aligned with sibling
  configs so cross-idea comparison stays valid.
- Smoke + shape tests live alongside the tower:
  `tests/test_bt4_primitive_mixer.py` and the per-mixer build tests in
  `tests/test_bt4_mixer_registry.py`.
