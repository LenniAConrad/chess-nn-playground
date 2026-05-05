# Math Thesis

Residual Calibration Error Field

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md`.

Batch candidate rank: `5`.

Working thesis: If the existing CNN has good accuracy but poor reliability on near-puzzles, a residual calibration architecture can predict where the baseline is likely overconfident. The model learns a spatial "calibration error field" and uses it to adjust logits or prod...

This registered implementation tests the thesis through the `information` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.
