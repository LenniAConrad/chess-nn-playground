# Codex Research Packet: Cayley Orthogonal Map Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1705_tuesday_local_cayley_orthogonal_map.md`
- Generated at: 2026-05-05 17:05
- Author: Claude (Opus 4.7, 1M context)
- Status: bespoke implementation already in `src/chess_nn_playground/models/cayley_orthogonal.py`

## Thesis

Build a learned skew-symmetric `A in R^{r x r}` from board features, then form
the **Cayley map**

```text
Q = (I - A)(I + A)^{-1}    in SO(r) (when no eigenvalue of A is at -1)
```

`Q` is a learned rotation that acts on a fixed reference basis. Identity-deviation
features `Q - I`, the diagonal of `Q`, and per-row rotated norms expose chess-
position-dependent rotational structure. Spectral-clipped `||A||_F ≤ 0.5` avoids
the Cayley pole and keeps `Q` numerically well-conditioned.

## Distinct From

- Polar-Procrustes (i063): polar / SVD decomposition; Cayley is an algebraic identity not a decomposition.
- Orthogonal moments (i133): fixed orthogonal projection without rotation generators.
- Clifford rotor (i232): multivector-algebra rotation; Cayley is matrix-algebra.

## Architecture

`CayleyOrthogonalNetwork` in `src/chess_nn_playground/models/cayley_orthogonal.py`:

```text
input (B, 18, 8, 8)
  -> BoardConvStem -> (B, C, 8, 8) -> spatial mean -> (B, C)
  -> Linear -> (B, r*(r-1)/2)  upper-triangular entries
  -> form skew A via triu - triu.T
  -> spectral-clip Frobenius
  -> Q = solve(I + A, I - A)      Cayley map
  -> rotate basis: rotated = Q @ basis  (B, r, C)
  -> features: diag(Q), per-row ||rotated||, dev = Q - I, sym^2, det proxy
  -> MLP -> (B, num_classes)
```

## Ablations

| Ablation | Target |
|---|---|
| `force_zero_A` | A = 0 (Q = I, no rotation) | sanity collapse |
| `random_skew` | random skew, not learned | tests learned A |
| `polar_decomp_swap` | replace Cayley with SVD-polar (i063 style) | tests algebraic identity vs decomposition |
| `r_eq_4` | shrink rank | tests rotation-rank |
| `cnn_same_params` | matched baseline | |

## Falsifier

`force_zero_A` should drop the model to constant prediction (sanity). `random_skew` should drop PR AUC ≥ 0.015.

## Targets

PR AUC ≥ 0.82, F1 ≥ 0.76, near-puzzle FPR ≤ 0.20.
