# Codex Research Packet: Clifford Rotor Threat Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1605_tuesday_local_clifford_rotor_threat.md`
- Generated at: 2026-05-05 16:05
- Weekday: Tuesday
- Timezone: local
- Author: Claude (Opus 4.7, 1M context)
- Intended next consumer: Codex
- Status: full unorthodox-linear-algebra architecture packet

## One-Sentence Thesis

Embed each square in a **Clifford geometric algebra** `Cl(p, q)` (specifically `Cl(3, 0)` so we have 1 + 3 + 3 + 1 = 8-dim multivectors per square), build pressure as **bivectors** (oriented planes), compose threats via the non-commutative **geometric product** `ab = a · b + a ∧ b`, and act on the board with learned **rotors** `R = exp(B/2)` via the sandwich `x → R x R⁻¹` — exposing oriented rotational structure (e.g. king-shielding rotations) that no real- or complex-arithmetic packet can express.

## Why This Is A Real Unorthodox Linear Algebra NN Idea

Clifford algebras `Cl(p, q, R)` are the unique associative algebras containing `R^{p+q}` such that `v² = q(v) · 1` for the quadratic form of signature `(p, q)`. They contain:

- **Vectors** (rank 1): `v = sum v_i e_i`.
- **Bivectors** (rank 2): `B = sum B_{ij} e_i ∧ e_j`, oriented planes; generate rotations.
- **Rotors**: `R = exp(B/2)` rotate vectors by `x → R x R⁻¹`.
- **Geometric product**: `ab = a · b + a ∧ b` is non-commutative and encodes both metric and orientation in a single object.

Quaternions are `Cl(0, 2)`; complex numbers are `Cl(0, 1)`. `Cl(3, 0)` (the Pauli algebra) is the natural setting for 3D rotations. Multivector neural networks (Brandstetter+, 2023) have shown gains on physics tasks where rotational structure dominates.

For chess, the bet:

- King defense / shielding has natural rotational structure: a defender "sweeps" around the king.
- Pin axes and discovered-attack lines are oriented bivectors.
- Knight forks live in 2-planes that are not closed under any commutative product.

## Target

```text
fine 0,1 -> 0,  fine 2 -> 1.  3x2 fine-to-binary mandatory.
```

## Forbidden Inputs

Standard.

## Closest Existing Ideas And Exact Difference

- `i040 Kinematic Commutator` — uses commutators, but in matrix algebra; not Clifford.
- `i230 Magnus-BCH` — uses BCH series of matrix commutators; not Clifford.
- `i127 Square-Color Parity Mixer` — Z/2Z parity, scalar; no rotor structure.
- `i119 Tensor-Ring` — tensor decomposition in real arithmetic.

```text
Clifford geometric algebra Cl(3, 0) is an 8-dimensional non-commutative associative
algebra with a specific quadratic form. Rotors R = exp(B/2) and the sandwich
x -> R x R^{-1} have no equivalent in matrix multiplication. No imported packet uses
multivector arithmetic.
```

## Mathematical Thesis

### Definitions

For each square, encode an 8-dim multivector in `Cl(3, 0)`:

```text
phi(s) = phi_0 + phi_1 e_1 + phi_2 e_2 + phi_3 e_3
       + phi_12 e_1 e_2 + phi_13 e_1 e_3 + phi_23 e_2 e_3
       + phi_123 e_1 e_2 e_3
```

with `e_i^2 = 1`, `e_i e_j = -e_j e_i`. The 8 components carry: 1 scalar (occupancy), 3 vectors (e.g. material gradient), 3 bivectors (oriented threat planes), 1 trivector (king-zone volume).

### Rotor builder

```text
B(s) = bivector part of phi(s)         (3-dim)
R(s) = exp(B(s) / 2)                   (rotor, even-grade Cl)
```

Then act on neighbor squares:

```text
phi_new(t) = sum_s W(s, t) * R(s) * phi(t) * R(s)^{-1}
```

This is a learned **rotor-equivariant** message-passing layer with adjacency `W` from chess geometry.

### Geometric-product interaction tensor

For each ordered pair `(s, t)` of squares with a chess relation, form

```text
G(s, t) = phi(s) * phi(t)        (geometric product)
```

a multivector capturing both alignment (`phi · psi`) and rotation (`phi ∧ psi`). Pool grade-by-grade:

```text
G_scalar  = grade-0 mean  (alignment)
G_vector  = grade-1 mean  (residual direction)
G_bivec   = grade-2 mean  (rotational mismatch)
G_trivec  = grade-3 mean  (volume / chirality)
```

### Readout

```text
G_scalar, G_vector, G_bivec, G_trivec       per relation type, pooled
||rotor B||_F                                 magnitude of rotational pressure
chirality_score = sum_s phi_123(s)            volume orientation
puzzle_logit = MLP([all_grades, board_pool])
```

## Assumptions

- King-zone defense and tactical lines have rotational structure better expressed in `Cl(3, 0)` than in matrix algebra.
- The geometric product captures alignment + orientation simultaneously; a same-params CNN with separate alignment & orientation channels is strictly less expressive *given equivariance*.

## Claim / Hypothesis

Rotor-equivariant message passing should be more sample-efficient than matrix-based attention on the same chess-graph adjacency. Central falsifier:

```text
scalar_only_cl: keep only the scalar grade of all multivectors (drops bivectors, vectors,
                trivectors). Reduces to a real-valued message-pass.
                if PR AUC doesn't drop, the rotor structure adds nothing.
```

## Architecture

```text
board_encoder              -> phi: 64 x 8  (multivector per square)
chess_relation_W            -> learned weighted adjacency 64 x 64 x num_relations
rotor_message_pass         -> phi' = sum W * R phi R^{-1}
geometric_product_block    -> G(s, t) for each pair
grade_pool                 -> per-grade summaries
puzzle_head
```

### First config

```yaml
model:
  name: clifford_rotor_threat_network
  input_channels: 18
  num_classes: 1
  channels: 64
  hidden_dim: 96
  cl_signature: [3, 0]
  multivector_dim: 8
  num_relation_types: 6
training:
  mode: puzzle_binary
  loss: bce_with_logits
```

## Numerical / Compute Notes

- `Cl(3, 0)` has 8 basis blades. Geometric product is a fixed `8 x 8 x 8` structure-constant tensor (sparse). Compute via einsum.
- Rotor exp via Taylor series (4 terms suffice for `||B|| <= 1`); spectral-clip B.
- Cost per pair: `O(8^2) = 64` multiplies. With 64 squares and `num_relations = 6`: `64 * 64 * 6 * 64 = 1.6M` flops per board. Cheap.
- All real arithmetic; no complex.

## Required Ablations

| Ablation | Removes | Hypothesis |
|---|---|---|
| `scalar_only_cl` | grade-0 only | tests need for higher grades |
| `vector_scalar_only` | grades 0, 1 only (real arithmetic) | tests bivector |
| `random_signature` | use Cl(0, 3) instead of Cl(3, 0) | tests signature choice |
| `commute_product` | replace geometric product with symmetric a·b + b·a / 2 | tests non-commutativity |
| `random_relation_W` | random sparse adjacency | tests chess semantics |
| `cnn_same_params` | matched CNN | baseline |
| `i040_commutator_baseline` | adjacent baseline | baseline |

## Benchmark Targets

```text
PR AUC >= 0.82, F1 >= 0.76, near-puzzle FPR <= 0.20, puzzle recall >= 0.78
scalar_only_cl drops PR AUC >= 0.015
commute_product drops PR AUC >= 0.01
```

## Counterexamples

- Chess threats have no genuine rotational structure; bivectors are noise.
- `Cl(3, 0)` signature is wrong; need `Cl(2, 0)` or `Cl(0, 2)`.
- Rotor exp diverges if `||B||` not clipped.

## Implementation Priority

1. Build geometric product structure-constant tensor for `Cl(3, 0)`.
2. Implement rotor exp (Taylor) + sandwich product.
3. Train minimal model: only grade-0 + grade-2 (bivector) features.
4. Run 7 ablations.
