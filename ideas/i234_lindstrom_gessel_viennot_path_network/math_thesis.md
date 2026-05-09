# Math Thesis

Lindstrom-Gessel-Viennot Path Determinant Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1615_tuesday_local_lindstrom_gessel_viennot_path.md`.

Working thesis: Builds a path-generating-function matrix
`M[i, j] = sum_paths(s_i -> t_j) prod edge_weights` via a truncated
Neumann series `G = sum_{k>=1} (alpha W)^k` on a learned chess DAG. The
LGV lemma makes `det(M)` the signed enumerator of non-intersecting
attacker-to-target k-tuples, distinguishing overload and double-attack
motifs algebraically from single-line tactical motifs that share trace
or Frobenius signatures but differ in their non-intersecting
multi-path content.
