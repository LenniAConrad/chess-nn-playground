# Implementation Notes — a015 BT4 Primitive Mixer (ray_occlusion_semiring_scan)

- Idea-local wrapper:
  `ideas/registry/a015_bt4_ray_occlusion_semiring_scan_mixer/model.py`.
- Registered model name: `bt4_ray_occlusion_semiring_scan_mixer`
  (resolved by the registry as an alias of `bt4_primitive_mixer` with
  `mixer = ray_occlusion_semiring_scan` — see
  `src/chess_nn_playground/models/registry.py::_make_bt4_mixer_alias`).
- Tower source:
  `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`.
- Mixer source:
  `src/chess_nn_playground/models/architecture/bt4_mixers/ray_occlusion_semiring_scan.py`.
- `implementation_kind: bespoke_model` — the tower itself is bespoke; the
  mixer swap is the variable under study.
- The wrapper passes `model` straight through to
  `build_bt4_primitive_mixer_from_config`, defaulting `mixer` and
  `num_classes` so an old config that omits them still resolves to the
  expected `ray_occlusion_semiring_scan` swap with one puzzle logit.
- Do not edit the shared BT4 tower or the mixer in this folder. Both
  are owned by `src/chess_nn_playground/models/...`; the architecture
  conformance audit will fail if the shared modules are forked in-place.
- The ray-step geometry and occupancy transmittance used by the mixer
  are derived analytically from the `simple_18` piece planes inside
  `torch.no_grad()`. No gradient flows back through the prefix-product
  transmittance or the per-direction step indices.
- Forward shape contract is checked by the shared registry parametrised
  test in `tests/test_idea_registry.py` and the BT4 mixer sibling
  test in `tests/test_research_architectures.py`.
