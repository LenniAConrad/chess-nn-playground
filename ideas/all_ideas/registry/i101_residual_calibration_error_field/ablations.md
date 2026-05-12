# Ablations

- Set `model.correction_scale: 0.0` to test the temperature-only calibration path.
- Increase `model.temperature_floor` to constrain the sharpening regime and test whether gains depend on aggressive temperature scaling.
- Reduce `model.depth` to 1 to test whether the calibration residual survives a smaller baseline trunk.
- Compare against LC0 BT4, NNUE, and the strongest registered idea runs on the same split and seeds.
