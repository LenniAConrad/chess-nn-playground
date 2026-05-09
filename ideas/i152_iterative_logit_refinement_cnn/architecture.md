# Architecture

`Iterative Logit Refinement CNN` is a board-only puzzle-binary classifier
that produces a logit `l_0` from a CNN-pooled latent and then refines it
through `T` learned correction steps over evidence space rather than
feature space.

## Pipeline

- Input: board tensor `(B, input_channels, 8, 8)` (defaults to the
  `simple_18` encoding). CRTK / source / engine metadata is
  reporting-only and never reaches the model.
- Trunk: a `BoardConvStem` of `depth` `Conv2d -> BatchNorm2d -> ReLU`
  blocks at constant `8 x 8` resolution producing a feature map
  `h ∈ R^{B × channels × 8 × 8}`.
- Pooled latent: global mean pool of `h` followed by a `LayerNorm`,
  giving `z ∈ R^{B × channels}`.
- Initial head (`Head_0`):
  `Linear(channels, hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim, num_classes)`
  produces the initial logit `l_0`.
- Correction loop, for `t = 1 .. refinement_steps`:
  - Build a deterministic confidence vector `ϕ(l_{t-1})`. For
    `num_classes == 1` it is `(l, σ(l), |l|, |2σ(l) − 1|, H_b(σ(l)))`;
    for `num_classes > 1` it is
    `(top1, top2, top1 − top2, H(softmax), max_logit, mean_logit)`.
  - `CorrectionMLP_t` is
    `LayerNorm -> Linear(in_dim, correction_hidden) -> GELU -> Dropout
     -> Linear(correction_hidden, correction_hidden) -> GELU -> Dropout
     -> Linear(correction_hidden, num_classes)`,
    where `in_dim = channels + num_classes + dim(ϕ)`.
  - The raw output is bounded: `c_t = correction_clamp · tanh(raw)`,
    with `correction_clamp = 0.25` by default (the packet's stable
    correction magnitude).
  - `l_t = l_{t-1} + c_t`.
  - By default the same `CorrectionMLP` is shared across all steps
    (weight tying). Setting `untie_corrections = true` switches to a
    `ModuleList` of distinct heads per step (the `untied_corrections`
    ablation from the packet).
- Final logit: `l_T` is returned as `logits`. The full trajectory
  `(l_0, …, l_T)` and the per-step corrections `c_t` are exposed as
  diagnostics for trajectory analysis.

## Output dictionary

- `logits` — shape `(B,)` for `num_classes == 1`, `(B, num_classes)`
  otherwise.
- `initial_logit` — `l_0` (scalar for binary, norm for multiclass).
- `final_logit` — `l_T` (scalar for binary, norm for multiclass).
- `step_logits` — shape `(B, T+1)` for binary or `(B, T+1, num_classes)`
  otherwise; the entire refinement trajectory.
- `correction_norms` — shape `(B, T)`; `||c_t||_2` per step (the
  packet's "average correction norm per step" diagnostic).
- `correction_norm_mean` — mean of `correction_norms` across steps.
- `correction_total` — sum of `correction_norms` across steps.
- `final_minus_initial` — `||l_T − l_0||_2`.
- `flip_after_step1` — fraction of samples whose predicted class
  flipped between `l_0` and `l_1` (the packet's "fraction of samples
  whose predicted class flips after step 1" diagnostic).
- `confidence_growth` — `|l_T| − |l_0|` (binary) or
  `max softmax(l_T) − max softmax(l_0)` (multiclass).
- `trunk_feature_energy` — mean squared activation of the trunk feature
  map.
- `latent_norm` — Euclidean norm of the pooled latent `z`.

## Why It Is Distinct

- Not a residual CNN: corrections happen in logit / evidence space, not
  in feature maps.
- Not a fixed-point residual defect: there is no convergence operator
  over latent states; the recurrence runs for a fixed `T` and never
  re-enters the trunk.
- Not a cascade: every sample executes all `T` correction steps; there
  is no early-exit gate.
- Not a `ResearchPacketProbe`: the head consumes a CNN trunk plus a
  staged refinement loop directly; there is no proposal-profile branch
  and no packet-keyword diagnostic.

## Implementation Binding

- Registered model name: `iterative_logit_refinement_cnn` (registered
  in `src/chess_nn_playground/models/registry.py`).
- Source implementation file:
  `src/chess_nn_playground/models/iterative_logit_refinement_cnn.py`
  (`IterativeLogitRefinementCNN`,
  `build_iterative_logit_refinement_cnn_from_config`).
- Idea-local wrapper:
  `ideas/i152_iterative_logit_refinement_cnn/model.py` calls
  `build_iterative_logit_refinement_cnn_from_config`.
- The shared `ResearchPacketProbe` scaffold is no longer used by this
  idea.
