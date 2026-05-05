# Math Thesis

Free-Probability R-Transform Spectrum Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1535_tuesday_local_free_probability_r_transform.md`.

Working thesis: Treats attacker A and defender B as freely independent operators and predicts spec(A+B) via free additive convolution (R-transform); the deviation between empirical spectrum and free-conv prediction (free-cumulant mismatch) is a coupling fingerprint.

This registered implementation routes the thesis through the `linear_algebra`
mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving
all source/CRTK metadata for reporting. See the source packet for the full
mathematical derivation, ablations, and falsification criteria.
