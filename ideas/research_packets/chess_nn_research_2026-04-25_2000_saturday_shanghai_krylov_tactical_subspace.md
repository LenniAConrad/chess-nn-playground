# Codex Research Packet: Krylov Tactical Subspace Network

## File Metadata

- Filename: `chess_nn_research_2026-04-25_2000_saturday_shanghai_krylov_tactical_subspace.md`
- Generated at: 2026-04-25 20:00
- Weekday: Saturday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: full linear-algebra architecture packet, not implemented, not benchmark results

## One-Sentence Thesis

Represent a chess position by the Krylov subspaces generated when learned chess-linear operators repeatedly propagate attacker, defender, king, and target seed vectors; classify puzzles from Ritz spectra, residual norms, and subspace-interaction statistics.

## Why This Is A Real Linear Algebra NN Idea

This is not just "use matrices somewhere." The core computation is:

```text
K_m(A, v) = span{v, A v, A^2 v, ..., A^{m-1} v}
```

where:

- `A` is a learned but chess-structured linear operator over square/piece/line features.
- `v` is a role-conditioned seed vector such as attacker pressure, defender mass, king-zone target, or high-value target.
- the network classifies from basis vectors, projected tridiagonal matrices, Ritz values, residual norms, and angles between Krylov subspaces.

The bet is that chess tactics are not only static patterns; they are repeated propagation of pressure through legal geometry. Krylov methods are exactly about what a linear operator does under repeated application.

## Target

First benchmark:

```text
source class 0: known non-puzzle / random position -> target 0
source class 1: verified near-puzzle / hard negative -> target 0
source class 2: verified puzzle -> target 1
```

The model emits:

```text
one puzzle logit
```

Mandatory diagnostic:

```text
3x2 source-class matrix:
rows    = random, near-puzzle, puzzle
columns = predicted non-puzzle, predicted puzzle
```

## Forbidden Inputs

Do not use:

- Stockfish scores
- PVs
- node counts
- mate scores
- engine best moves
- verification metadata
- source labels
- source file identity
- future game outcomes

Only current-board tensors and deterministic chess geometry are allowed.

## Closest Existing Ideas

### Closest Linear Algebra Packets

- `Grassmannian Principal-Angle Bottleneck`
- `Matrix-Pencil Generalized Spectrum Bottleneck`
- `Polar-Procrustes Alignment Bottleneck`
- `Schur-Ray Line Algebra Network`
- `Bitboard Shift-Algebra Network`
- `Chess Operator Basis Classifier`

### Exact Overlap

```text
All use linear operators, spectra, subspaces, or chess-shaped matrix structure.
```

### Exact Difference

```text
Krylov Tactical Subspace Network studies repeated action of a learned chess operator on role seed vectors. It does not compare static covariance subspaces, solve a generalized eigenproblem, align matrices by Procrustes, run a Schur/Woodbury line solve, or use fixed low-degree operator polynomials alone.
```

The unique object is:

```text
role-conditioned Krylov behavior under repeated tactical propagation
```

## Mathematical Thesis

### Definitions

Let a board be encoded as square/piece features:

```text
X in R^{64 x d}
```

Construct a chess-structured operator:

```text
A(X) in R^{64 x 64}
```

with components:

```text
A = gate_ray * A_ray
  + gate_knight * A_knight
  + gate_pawn * A_pawn
  + gate_king * A_king
  + gate_defense * A_defense
  + low_rank_context_update
```

The fixed pieces of `A` are legal-geometry masks; learned gates and low-rank terms depend only on the current board.

Define role seed vectors:

```text
v_attack
v_defense
v_king_zone
v_high_value_target
v_blocker
v_tempo
```

For each seed:

```text
K_m(A, v_r) = [v_r, A v_r, A^2 v_r, ..., A^{m-1} v_r]
```

Use a differentiable Arnoldi/Lanczos-style orthogonalization:

```text
Q_r, H_r = arnoldi(A, v_r, m)
```

where:

```text
Q_r^T Q_r approx I
A Q_r approx Q_r H_r + residual
```

Extract:

- Ritz values of `H_r`
- residual norm `||A Q_r - Q_r H_r||`
- Krylov basis energy near king/target squares
- principal angles between role subspaces
- cross-Gram matrices `Q_attack^T Q_defense`
- growth/decay curves `||A^k v_r||`

Final:

```text
puzzle_logit = MLP([ritz_spectra, residuals, cross_role_angles, growth_curves, board_context])
```

## Assumptions

- Tactical pressure behaves like repeated propagation through chess geometry.
- True puzzles create distinctive operator dynamics: rapid concentration on targets, unstable defender subspaces, or high attacker/defender subspace conflict.
- Near-puzzles may have similar first-order pressure but weaker higher-order propagation or stronger defensive Krylov overlap.
- Low-dimensional Krylov summaries can capture this efficiently.

## Claim

Hypothesis: Krylov role-subspace summaries should reduce near-puzzle false positives because they distinguish one-step tactical appearance from multi-step operator propagation toward forcing targets.

## Mechanism

A CNN can see local pressure. A Krylov network asks:

```text
what happens after pressure propagates through the chess operator repeatedly?
```

For a true puzzle:

```text
attacker Krylov energy should concentrate toward target/king zones
defender Krylov subspace may fail to span the same target directions
projected spectra may show unstable or high-gain tactical modes
```

For a near-puzzle:

```text
first-step pressure may be high
but later Krylov vectors may diffuse, cancel, or align with defender coverage
```

## Architecture

### Components

```text
board_encoder
operator_builder A(X)
role_seed_builder V(X)
krylov_block per role
role_subspace_interaction block
spectral/residual readout
puzzle head
```

### Forward Pass

```text
X = board_encoder(board)
A = operator_builder(X)
for role r:
    v_r = role_seed_builder(X, r)
    Q_r, H_r, residual_r = krylov_block(A, v_r, steps=m)
    features_r = spectral_readout(H_r, Q_r, residual_r)
cross_features = role_interactions(Q_attack, Q_defense, Q_king, Q_target)
logit = puzzle_head([features_r, cross_features, pool(X)])
```

### First Config

```yaml
model:
  name: krylov_tactical_subspace_network
  input_channels: 18
  num_classes: 1
  hidden_dim: 96
  operator_rank: 16
  krylov_steps: 6
  roles:
    - attack
    - defense
    - king_zone
    - high_value_target
    - blocker
    - tempo
  orthogonalization: modified_gram_schmidt
training:
  mode: puzzle_binary
  loss: bce_with_logits
```

## Numerical Details

Use stable bounded operators:

```text
A <- A / max(1, spectral_norm_estimate(A))
```

Use modified Gram-Schmidt or Householder-style orthogonalization:

```text
q_{k+1} = normalize(A q_k - sum_i <q_i, A q_k> q_i)
```

Avoid exact full eigendecomposition of 64x64 `A` in version 1. Compute spectra of small projected `H_r` matrices:

```text
H_r in R^{m x m}, m = 4..8
```

This makes the model cheap and differentiable.

## Why It Might Beat BT4

BT4 learns spatial filters. It can approximate pressure propagation, but it is not forced to expose repeated operator behavior. Krylov summaries are a compact way to detect whether tactical influence actually reaches critical targets over several propagation steps.

This directly attacks near-puzzles:

```text
near-puzzle: high initial pressure, weak propagation / strong defense overlap
puzzle: high initial pressure, strong target concentration / weak defense overlap
```

## Ablations

| Ablation | Purpose |
|---|---|
| `one_step_only` | Tests whether repeated propagation matters. |
| `no_orthogonalization` | Tests Krylov basis vs raw powers. |
| `fixed_operator_only` | Removes learned board-conditioned gates. |
| `random_geometry_operator` | Tests chess geometry semantics. |
| `no_spectral_readout` | Uses only final Krylov vector; tests Ritz features. |
| `no_cross_role_angles` | Tests attacker/defender subspace interaction. |
| `cnn_same_params` | Size-matched standard CNN baseline. |

## Falsification Criteria

Reject or revise if:

```text
one_step_only matches full model
or random_geometry_operator matches chess operator
or no_cross_role_angles matches full model
or near-puzzle FP does not improve over size-matched CNN
```

Ambitious benchmark target:

```text
test PR AUC >= 0.82
test F1 >= 0.76
near-puzzle FP <= 0.20
puzzle recall >= 0.78
```

## Counterexamples

This idea may fail if:

- puzzle labels are mostly local motifs that do not need repeated propagation
- the learned operator becomes too smooth and loses tactical sharpness
- orthogonalization creates training instability
- role seed vectors are poor
- near-puzzles and puzzles have similar Krylov spectra at shallow depth

## Implementation Priority

If implementing one more linear algebra architecture, this is the recommended order:

1. Build fixed chess operator masks: rays, knight, pawn, king, same-line defense.
2. Add learned gates and low-rank board-conditioned update.
3. Implement stable Krylov block with `m=4` first.
4. Add spectral readout over projected `H`.
5. Add cross-role angle features.
6. Benchmark against CNN, BT4, and `i001_chess_operator_basis_classifier`.

The smallest viable version:

```text
roles: attack, defense, king_zone
krylov_steps: 4
operator: fixed chess geometry + learned scalar gates
readout: growth curves + cross-role Gram matrix
```

If that shows signal, add Ritz spectra and low-rank operator updates.

