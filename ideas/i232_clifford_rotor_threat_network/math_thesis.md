# Math Thesis

Clifford Rotor Threat Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1605_tuesday_local_clifford_rotor_threat.md`.

## Geometric algebra setup

Each chess square is embedded as an 8-dim multivector in `Cl(3, 0)`:

```
phi(s) = phi_0
       + phi_1 e_1 + phi_2 e_2 + phi_3 e_3
       + phi_12 e_1 e_2 + phi_13 e_1 e_3 + phi_23 e_2 e_3
       + phi_123 e_1 e_2 e_3
```

with `e_i^2 = +1` and `e_i e_j = -e_j e_i` for `i != j`. The 8 components
carry one scalar (occupancy), three vectors (material gradient),
three bivectors (oriented threat planes) and one trivector (king-zone
volume / chirality).

The geometric product is non-commutative and built from the 8x8x8
structure tensor

```
M[i, j, k] = sigma(i, j)        if i XOR j == k
           = 0                  otherwise
```

where blade indices use the bitmask `bit_t -> e_{t+1}` convention and
`sigma(i, j) = (-1)^N(i, j)` counts the basis-vector inversions needed
to put `e_i e_j` in canonical order. The blade grade is the popcount of
the bitmask (grades `[0, 1, 1, 2, 1, 2, 2, 3]` for indices 0..7).

## Rotor builder

Bivectors `B = sum_{i < j} B_{ij} e_i e_j` square to `-||B||^2`, so the
exponential

```
R = exp(B / 2) = cos(theta / 2) + (B / theta) sin(theta / 2)
```

is even-graded with `|R| = 1` and `R^{-1} = R~` (reverse). The rotor
sandwich `x -> R x R^{-1}` rotates vectors in the plane defined by `B`
by angle `theta = ||B||`.

## Composition over chess relations

For each ordered pair `(s, t)` of squares related by one of six
chess geometric relations - king ring, knight, same rank, same file,
a1-h8 diagonal, a8-h1 anti-diagonal - form the rotated multivector field
`phi'(s) = R(s) phi(s) R(s)^{-1}` and compute the relation-specific
geometric-product message

```
m_r(s) = phi'(s) * sum_t W_r(s, t) phi'(t)
```

This captures both alignment (`grade-0 / scalar`) and rotation
(`grade-2 / bivector`) simultaneously, with grade-1 and grade-3 channels
representing residual direction and signed volume.

## Readout

Pool each `m_r` per grade `(0, 1, 2, 3)` (mean and max norm over
squares) and concatenate with diagnostic scalars - bivector norm,
rotor norm, sandwich residual `||phi - phi'||_F`, trivector chirality
score, and per-grade `phi` energies - then feed the pooled trunk
features and these into a LayerNorm + GELU MLP that emits one puzzle
logit.

## Falsifier

The central ablation `scalar_only_cl` masks every blade other than
`grade-0`, reducing the geometric product to a real inner product. If
PR AUC does not drop, the rotor-equivariant structure adds nothing.
