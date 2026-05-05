# Math Thesis

Magnus-BCH Operator-Coupling Series Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1545_tuesday_local_magnus_bch_coupling_series.md`.

Working thesis: Computes BCH log Z = log(exp(A) exp(B)) as a truncated Magnus series of nested commutators up to weight 4; weight-3 and weight-4 commutator norms capture iterated tactical depth that single-commutator features (i040) cannot detect.

This registered implementation routes the thesis through the `linear_algebra`
mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving
all source/CRTK metadata for reporting. See the source packet for the full
mathematical derivation, ablations, and falsification criteria.
