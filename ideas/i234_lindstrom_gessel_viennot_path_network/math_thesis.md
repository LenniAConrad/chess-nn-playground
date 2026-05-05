# Math Thesis

Lindstrom-Gessel-Viennot Path Determinant Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1615_tuesday_local_lindstrom_gessel_viennot_path.md`.

Working thesis: Builds a path-generating-function matrix M[i,j] = sum_paths(s_i -> t_j) prod edge_weights via Neumann series on a learned chess DAG; LGV lemma makes det(M) the signed enumerator of non-intersecting attacker-to-target k-tuples. Distinguishes overload/double-attack motifs algebraically.

This registered implementation routes the thesis through the `linear_algebra`
mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving
all source/CRTK metadata for reporting. See the source packet for the full
mathematical derivation, ablations, and falsification criteria.
