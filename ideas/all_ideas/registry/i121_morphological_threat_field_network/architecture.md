# Architecture

`Morphological Threat Field Network` materialises the differentiable
mathematical-morphology thesis from `math_thesis.md`. The model treats per-square
scalar surfaces as threat fields and processes them with learned morphological
structuring elements, so chess shape operations such as expanding a king-danger
zone, closing pawn-shield gaps, eroding escape squares, and detecting thin
corridors are first-class computations rather than implicit CNN side-effects.

- Input: simple_18 board tensor `(B, 18, 8, 8)`. CRTK/source metadata is
  reporting-only and never used as input.
- Board trunk: a compact convolutional stem (`BoardConvStem`) produces dense
  square features.
- Threat-field projector: a 1x1 convolution with `softplus` projects trunk
  features onto a small bank of nonnegative threat surfaces. The first channels
  are anchored to a deterministic side-relative seed (us mass, them mass, us
  king, them king, occupancy, empty) so the morphology operates on signal that
  is tied to chess geometry from initialisation.
- Morphological cascade: each `MorphologicalLayer` carries two learned
  structuring elements (`dilation_kernel`, `erosion_kernel`) and exposes soft
  dilation, soft erosion, opening, closing, morphological gradient, top-hat
  (`field - opening`) and bottom-hat (`closing - field`). Soft min and soft max
  are realised through temperature-scaled `logsumexp`, which approximates the
  classical morphological neural-network rules `y(p) = max(x(p+s) + w(s))` and
  `y(p) = min(x(p+s) - w(s))` while staying differentiable.
- Output diagnostics: each layer's seven morphological outputs are pooled by
  mean and max, producing a deep morphological signature. Compact summary
  scalars track expansion mass, erosion mass, morphological-gradient mass, the
  thin-corridor peak, opening residual, and closing residual.
- Fusion head: the trunk is concatenated with the final-layer dilation and
  erosion fields, projected by 1x1 conv, then mean/max pooled and fed to an MLP
  classifier together with the morphological signature and diagnostics. A
  parallel linear branch produces a morphology-only logit that supports
  diagnostic comparison.

## Implementation Binding

- Registered model name: `morphological_threat_field_network`.
- Source implementation file: `src/chess_nn_playground/models/morphological_threat_field_network.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i121_morphological_threat_field_network/model.py`
  (a thin `build_model_from_config(config)` wrapper around
  `build_morphological_threat_field_network_from_config`).
- The model is registered in `src/chess_nn_playground/models/registry.py` and
  is excluded from `RESEARCH_PACKET_MODEL_NAMES` so the audit detects this
  folder as `bespoke_model`.
- Output contract: returns `{"logits": (B,), ...}` with the diagnostic tensors
  `morphology_branch_logit`, `threat_field_mass`, `threat_field_peak`,
  `threat_expansion_mass`, `threat_erosion_mass`, `morphological_gradient`,
  `thin_corridor_intensity`, `opening_residual`, `closing_residual` (all shape
  `(B,)`).
