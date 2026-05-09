# Math Thesis

Magnus-BCH Operator-Coupling Series Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1545_tuesday_local_magnus_bch_coupling_series.md`.

Working thesis: For non-commuting operators `A`, `B`, `exp(A) exp(B) != exp(A + B)`
in general. The Baker-Campbell-Hausdorff log

```text
log(exp(A) exp(B)) = A + B + 1/2 [A, B]
                   + 1/12 ([A, [A, B]] - [B, [A, B]])
                   - 1/24 [B, [A, [A, B]]]
                   + ...
```

is the unique non-commutative log of the product, with a Magnus series whose terms
are nested commutators in a Hall basis of the free Lie algebra at every weight.

This idea computes the Hall-basis nested commutators up to weight 4

```text
weight 1:  A,                                   B
weight 2:  c_2  = [A, B]
weight 3:  c_3a = [A, c_2],                     c_3b = [B, c_2]
weight 4:  c_4a = [A, c_3a],   c_4b = [B, c_3a],
           c_4c = [A, c_3b],   c_4d = [B, c_3b]
```

(9 monomials total, matching Witt's formula at weight 4) and reads their Frobenius
norms plus the truncated BCH log

```text
Z = A + B + 1/2 c_2 + (1/12)(c_3a - c_3b) + (1/24) c_4b
```

as the puzzle-binary fingerprint. The slow decay of `||c_k||_F` from `k=2` to `k=4`
is hypothesised to separate single-step near-puzzles (fast Lie-tail decay) from
multi-step combinations (slow decay -- iterated commutators stay large), which is
provably distinct from the i040 Kinematic Commutator feature that stops at weight
2. BCH convergence is enforced by spectrally clipping `||A||_2, ||B||_2 <=
spectral_clip_per_op` (default `0.5`, so the radius of convergence ~ `log 2`
applies with safety margin).

The bespoke implementation lives in
`src/chess_nn_playground/models/magnus_bch_coupling_series_network.py` and is
wired through the registered builder
`build_magnus_bch_coupling_series_network_from_config`.
