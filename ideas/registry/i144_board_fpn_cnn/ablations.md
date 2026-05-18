# Ablations

The model exposes its central falsifiers through `model.ablation`. Run the main config plus the six ablation configs below.

- `none` (default): full FPN with top-down fusion at both `4x4` and `8x8` plus the `2x2` head feature; coordinate planes enabled.
- `single_resolution_matched`: keep only the `8x8` bottom-up output; zero `y4` and the `2x2` head feature. Central falsifier for the multi-resolution claim.
- `bottom_up_only`: drop top-down fusion; the head sees per-level pools but no coarse-to-fine information flow. Tests whether the top-down identity matters.
- `no_2x2_level`: keep `8x8` and `4x4` fusion but zero the `2x2` head feature. Tests whether the coarsest scale is needed.
- `late_pool_only`: skip the top-down fusion entirely; bottom-up `x8`, `x4`, and `x2` pools feed the head directly. Tests whether bottom-up alone matches the full FPN.
- `no_coordinate_planes`: drop the deterministic coordinate planes from the input. Tests whether absolute-board context is necessary.

Compare against LC0 BT4, NNUE, and the strongest registered idea runs (in particular shallow/wide residual CNNs and the strongest CNN baselines) on the same split and seeds to isolate the multi-resolution feature-pyramid contribution.
