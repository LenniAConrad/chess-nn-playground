# Math Thesis

Tracy-Widom Level-Spacing Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1610_tuesday_local_tracy_widom_level_spacing.md`.

Working thesis: Computes eigenvalues of a learned chess Hermitian operator and classifies puzzle-likeness from nearest-neighbor level-spacing statistics: Wigner-Dyson (chaotic, GOE/GUE) versus Poisson (integrable). Mean spacing ratio <r> and spectral form factor are the central invariants.

This registered implementation routes the thesis through the `linear_algebra`
mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving
all source/CRTK metadata for reporting. See the source packet for the full
mathematical derivation, ablations, and falsification criteria.
