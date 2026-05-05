# Math Thesis

Let a board tensor induce square embeddings \(h_s\) and a deterministic ordered set of chess relation tokens \(r_e\) for rays, knight jumps, king adjacency, and pawn diagonals. SRPA learns two dictionaries with identical capacity:

- \(D_b\): background relation dictionary.
- \(D_t\): tactical relation dictionary.

For each relation token, an unrolled group-sparse pursuit layer solves an approximate sparse coding problem:

\[
a^*_k(r) \approx \arg\min_a \|r - D_k a\|_2^2 + \lambda_1 \|a\|_1 + \lambda_g \sum_g \|a_g\|_2
\]

where \(k \in \{b,t\}\). The classifier is not allowed to inspect dense board embeddings directly. It receives only residual traces, group energies, activity rates, entropy, and dictionary health statistics.

The central hypothesis is asymmetric reconstruction:

\[
\text{puzzle} \Rightarrow \|r - D_t a_t\|^2 < \|r - D_b a_b\|^2
\]

while non-puzzle and near-puzzle positions should be better explained by background sparse codes or show weaker tactical residual advantage.

This is a hypothesis, not a proof. The falsifier is whether residual asymmetry and group statistics improve near-puzzle false-positive control against the LC0 BT4 baseline and VetoSelect/Dykstra variants on the canonical CRTK tagged split.
