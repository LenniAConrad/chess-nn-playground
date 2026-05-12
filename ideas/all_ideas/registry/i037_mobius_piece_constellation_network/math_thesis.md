# Math Thesis

Möbius Piece-Constellation Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0713_tuesday_los_angeles_mobius_constellation.md`.

The thesis is that puzzle-likeness can be carried by sparse unordered constellations of current-board piece-square facts. For an occupied token multiset `T(b) = {t_1, ..., t_n}`, MPCN learns a token map:

```text
psi(t) in R^d
```

and computes elementary symmetric interaction embeddings:

```text
Phi_1 = sum_i psi(t_i)
Phi_2 = sum_{i<j} psi(t_i) * psi(t_j)
Phi_3 = sum_{i<j<k} psi(t_i) * psi(t_j) * psi(t_k)
```

where `*` is elementwise multiplication. These terms isolate degree-1 piece/square facts, degree-2 pair constellations, and degree-3 triple constellations. They are normalized by tuple-count scale factors to reduce trivial dependence on the number of occupied pieces.

The recurrence computes the same interaction sums without enumerating pairs or triples:

```text
E_0 = 1
E_1 = E_2 = E_3 = 0
for token vector v:
    E_3 = E_3 + E_2 * v
    E_2 = E_2 + E_1 * v
    E_1 = E_1 + v
```

A linear head over `Phi_r` can represent any CP-rank-`d` symmetric interaction score of degree at most three. The implemented MLP head over gated `[Phi_1, Phi_2, Phi_3]` therefore tests whether such low-rank piece-square constellations add signal beyond material and square marginals.

The source packet uses binary `fine_label > 0` as its first experiment. This repo idea contract is stricter: fine labels `0` and `1` are non-puzzle and fine label `2` is puzzle. The implementation keeps the MPCN operator unchanged and trains the single returned BCE logit under the repo's `fine_label == 2` positive mapping.

The model never consumes engine scores, legal move enumeration, attack graphs, verification metadata, source labels, CRTK tags, or future-line information.
