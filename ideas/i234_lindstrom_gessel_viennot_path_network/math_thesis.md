# Math Thesis

Lindstrom-Gessel-Viennot Path Determinant Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1615_tuesday_local_lindstrom_gessel_viennot_path.md`.

Working thesis: Builds a path-generating-function matrix M[i,j] = sum_paths(s_i -> t_j) prod edge_weights via Neumann series on a learned chess DAG; LGV lemma makes det(M) the signed enumerator of non-intersecting attacker-to-target k-tuples. Distinguishes overload/double-attack motifs algebraically.

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.
