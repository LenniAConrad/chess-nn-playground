# Architecture

`Board FPN CNN` is a bespoke three-level feature-pyramid CNN over the simple_18 board tensor. It captures both exact square detail and coarse whole-board phase by running convolutional stacks at three resolutions (`8x8`, `4x4`, `2x2`), then fusing the coarse maps back into the fine maps via top-down 1x1 projections and nearest-neighbor upsampling. The classifier head reads concatenated mean and max pools from every level.

## Implementation Binding

- Registered model name: `board_fpn_cnn`
- Source implementation file: `src/chess_nn_playground/models/trunk/board_fpn_cnn.py`
- Idea-local wrapper: `ideas/registry/i144_board_fpn_cnn/model.py`

## Modules

`BoardFPNCoordinatePlanes` (optional, on by default) prepends four deterministic coordinate planes — rank, file, center distance, and square color parity — so the convolutions can absorb absolute-board context that translation-equivariant 3x3 filters drop.

`BoardFPNConvStack` is a stack of `BoardFPNConvBlock` modules (Conv2d 3x3 + optional BatchNorm + GELU + optional Dropout2d). Three stacks are instantiated:

- `level8`: input planes → `width` channels at `8x8`.
- `level4`: `width` → `2 * width` channels at `4x4` (after a 2x2 average-pool on `level8` output).
- `level2`: `2 * width` → `4 * width` channels at `2x2` (after a second 2x2 average-pool).

Top-down fusion uses two 1x1 projections:

- `project2_to4`: maps the `4 * width` channel `2x2` feature map down to `2 * width` channels, upsampled via nearest-neighbor back to `4x4`. The result is added to the bottom-up `level4` output to form the fused `y4`.
- `project4_to8`: maps `y4` from `2 * width` to `width` channels, upsampled to `8x8` and added to `level8` output to form the fused `y8`.

`BoardFPNHead` is the classifier. It pools each level by concatenating mean-pool and max-pool features, then concatenates `y8`, `y4`, and the `2x2` feature map. The pooled vector has dimension `2 * (width + 2 * width + 4 * width) = 14 * width`. A `LayerNorm + (Linear + GELU + Dropout) x 2 + Linear(num_classes)` MLP produces one BCE-compatible logit for `puzzle_binary`.

## Ablation modes

`BoardFPNCNN.ABLATIONS` enumerates the testable variants:

- `none` (default): full FPN with top-down fusion at both `4x4` and `8x8` and the `2x2` head feature live.
- `single_resolution_matched`: keep only the `8x8` bottom-up output; zero `y4` and the `2x2` head feature. Central control: does the multi-resolution fusion matter at all?
- `bottom_up_only`: skip top-down fusion; the head still sees per-level pools, but no coarse-to-fine information flow.
- `no_2x2_level`: keep `8x8` and `4x4` fusion but zero the `2x2` head feature.
- `late_pool_only`: skip top-down fusion entirely (same as `bottom_up_only` on `y4` and `y8`); the head receives bottom-up `x8`/`x4`/`x2` pools.
- `no_coordinate_planes`: drop the deterministic coordinate planes from the input. Tests whether absolute-board context is required.

## Diagnostics

`forward(x)` returns a dict containing:

- `logits`: shape `(B,)` (or `(B, num_classes)` for non-binary configs), BCE-compatible for the one-logit `puzzle_binary` head.
- `prob`: sigmoid of the puzzle logit (softmax when `num_classes > 1`).
- `fpn_y8_energy`, `fpn_y4_energy`, `fpn_x2_energy`: mean square of each fused level's feature map.
- `topdown_4_energy`, `topdown_8_energy`: mean square of the top-down 1x1 projection updates (zeroed under the no-top-down ablations).
- `piece_density`: side-summed piece-plane occupancy averaged over the board.
- `coordinate_energy`: mean square of the coordinate planes (zero when disabled).
- `mechanism_energy`: sum of per-level fused energies; operationalises the spatial multi-scale mechanism.
- `proposal_profile_strength`: per-sample max of the three per-level energies.
- `proposal_keyword_count`: scalar count of active feature levels (3).
- `board_fpn_ablation`: integer code identifying the active ablation mode.
- `board_fpn_level_count`: scalar reporting the level count (3).

## Contract

- Input: `(B, 18, 8, 8)` board tensor only. CRTK / verification / source / engine metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit `puzzle_binary` BCE-with-logits trainer, plus the diagnostics listed above.
- Target mapping: fine labels `0` and `1` map to binary target `0`; fine label `2` maps to binary target `1`.
- Model shapes: bottom-up `x8 ∈ R^{B, width, 8, 8}`, `x4 ∈ R^{B, 2*width, 4, 4}`, `x2 ∈ R^{B, 4*width, 2, 2}`; fused `y8 ∈ R^{B, width, 8, 8}` and `y4 ∈ R^{B, 2*width, 4, 4}`; pooled head input `R^{B, 14*width}`.
- The puzzle decision flows through the pooled multi-resolution feature vector; the FPN structure is enforced architecturally by the per-level stacks and top-down 1x1 projections.
- Only the conv weights, 1x1 projections, BatchNorms, LayerNorm, and head linears are trainable.
