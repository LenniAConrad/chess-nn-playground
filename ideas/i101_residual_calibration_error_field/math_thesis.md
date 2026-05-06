# Math Thesis

Residual Calibration Error Field

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md`.

Batch candidate rank: `5`.

Working thesis: If the existing CNN has good accuracy but poor reliability on near-puzzles, a residual calibration architecture can predict where the baseline is likely overconfident. The model learns a spatial "calibration error field" and uses it to adjust logits or prod...

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.
