# Ablations

The model exposes its central falsifiers through `model.ablation`. Run the main config plus the six ablation configs below.

- `none` (default): full block — rank, file, and local branches active with residual skip and configured depth.
- `local_only`: zero the rank and file branches; only the 3x3 local mixer runs. Central falsifier for the axial mixing claim.
- `rank_only`: zero the file and local branches; only the rank-wise 1D conv runs. Tests whether one axial direction alone suffices.
- `file_only`: zero the rank and local branches; only the file-wise 1D conv runs. Symmetric control to `rank_only`.
- `no_residual`: replace the residual update `x + update` with `update` itself. Tests whether the residual skip carries signal.
- `single_block`: collapse the trunk to a single axial block regardless of the configured depth. Tests whether deeper axial stacks help.

Compare against LC0 BT4, NNUE, and the strongest registered idea runs (in particular Board FPN CNN and the strongest CNN baselines) on the same split and seeds to isolate the axial rank/file contribution.
