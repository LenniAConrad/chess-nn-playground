# Codex Research Packet: Adaptive Tactical Resolvent Network

## File Metadata

- Filename: `chess_nn_research_2026-04-25_2002_saturday_shanghai_adaptive_tactical_resolvent.md`
- Generated at: 2026-04-25 20:02
- Weekday: Saturday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: full linear-algebra architecture packet, not implemented, not benchmark results

## One-Sentence Thesis

Use a learned chess-structured resolvent operator `(I - alpha A)^(-1)` to propagate attacker and defender influence through all path lengths at once, then classify puzzlehood from target transfer, defensive cancellation, and resolvent sensitivity.

## Why This Is The Strongest Linear-Algebra Candidate

If I had to pick one linear-algebra architecture with the best chance of actually working, I would pick this one.

Reason:

```text
Krylov is expressive but needs careful orthogonalization.
Schur-Ray is chess-specific but focused on sliding lines.
Matrix-pencil and Grassmannian ideas are elegant but more indirect.
Resolvent propagation is global, differentiable, stable, and implementable on 64 squares.
```

The key object is:

```text
R_alpha(A) = (I - alpha A)^(-1)
           = I + alpha A + alpha^2 A^2 + alpha^3 A^3 + ...
```

This is exactly the kind of operator we want: a compact all-length propagation of tactical influence through chess geometry.

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

Allowed inputs:

- current-board tensors
- deterministic chess geometry
- learned gates derived from current-board features

## Closest Existing Ideas

### Closest Packets

- `Harmonic Board Potential Network`
- `Schur-Ray Line Algebra Network`
- `Krylov Tactical Subspace Network`
- `Bitboard Shift-Algebra Network`
- `Chess Operator Basis Classifier`

### Exact Overlap

```text
All propagate chess-shaped signals with linear operators.
```

### Exact Difference

```text
Adaptive Tactical Resolvent uses a stable shifted inverse / Green's-function operator over learned chess geometry. It does not solve a Poisson potential, compute a Schur line solve, build Krylov bases, or use fixed operator polynomials.
```

The key distinction from Krylov:

```text
Krylov explicitly stores repeated powers A^k v.
Resolvent sums all powers through a stable inverse and reads off transfer/cancellation directly.
```

## Mathematical Thesis

### Definitions

Let a position be encoded as:

```text
X in R^{64 x d}
```

Build a board-conditioned chess operator:

```text
A(X) in R^{64 x 64}
```

with sparse chess structure:

```text
A = g_ray * A_ray
  + g_knight * A_knight
  + g_pawn * A_pawn
  + g_king * A_king
  + g_defense * A_defense
  + U(X) V(X)^T
```

where:

- fixed masks encode legal chess geometry
- scalar/vector gates `g_*` are learned from current board context
- low-rank term `U V^T` captures board-specific tactical shortcuts

Stabilize:

```text
A_hat = A / max(1, spectral_norm_estimate(A))
```

Define multiple resolvents:

```text
R_k = (I - alpha_k A_hat)^(-1)
```

with learned or fixed `alpha_k` values such as:

```text
alpha in {0.25, 0.5, 0.75}
```

Create seed vectors:

```text
s_attack
s_defense
s_king_target
s_material_target
s_blocker
s_tempo
```

Propagate:

```text
y_attack,k  = R_k s_attack
y_defense,k = R_k s_defense
y_target,k  = R_k^T s_target
```

Read transfer and cancellation:

```text
attack_to_target_k = <y_attack,k, s_target>
defense_to_target_k = <y_defense,k, s_target>
net_pressure_k = attack_to_target_k - defense_to_target_k
resolvent_sensitivity_k = || R_k s_attack - R_k s_defense ||
```

Final:

```text
puzzle_logit = MLP([
  net_pressure across k,
  transfer ratios,
  defense cancellation,
  king-zone resolvent energy,
  material-target resolvent energy,
  sensitivity,
  board context
])
```

## Assumptions

- Tactical signal is global propagation of attack and defense through chess geometry.
- True puzzles have high attacker-to-target transfer that is not cancelled by defender transfer.
- Near-puzzles often have high initial pressure but stronger global defensive cancellation.
- Resolvent operators capture long-range and multi-step influence more directly than CNN depth.

## Claim

Hypothesis: an adaptive tactical resolvent should outperform size-matched CNNs and challenge BT4 because it computes all-length chess influence transfer and attacker/defender cancellation in one stable linear-algebra block.

## Mechanism

A near-puzzle can have:

```text
strong s_attack locally
```

but after global propagation:

```text
R s_defense reaches the same targets
net_pressure is small
```

A true puzzle should have:

```text
R s_attack reaches high-value/king targets
R s_defense cannot cancel it
net_pressure stays high across alpha scales
```

The multi-alpha resolvent is important:

```text
small alpha: local tactical contact
medium alpha: line/knight/pawn propagation
large alpha: global long-range pressure and defense cancellation
```

## Architecture

### Components

```text
board_encoder
operator_builder
seed_builder
resolvent_solver
transfer_readout
puzzle_head
```

### Forward Pass

```text
X = board_encoder(board)
A = operator_builder(X)
A_hat = spectral_normalize(A)
seeds = seed_builder(X)
for alpha in alpha_values:
    R_attack = solve(I - alpha * A_hat, s_attack)
    R_defense = solve(I - alpha * A_hat, s_defense)
    R_target = solve_transpose(I - alpha * A_hat, s_target)
    features_alpha = transfer_readout(R_attack, R_defense, R_target, seeds)
logit = puzzle_head([features_alpha, pool(X)])
```

### First Config

```yaml
model:
  name: adaptive_tactical_resolvent_network
  input_channels: 18
  num_classes: 1
  hidden_dim: 96
  low_rank: 12
  alpha_values: [0.25, 0.50, 0.75]
  solver: conjugate_gradient_or_direct_64
  spectral_norm_iters: 3
  seed_roles:
    - attack
    - defense
    - king_target
    - material_target
    - blocker
    - tempo
training:
  mode: puzzle_binary
  loss: bce_with_logits
```

## Solver Choice

Because the board has only 64 squares, version 1 can use direct batched solves:

```text
torch.linalg.solve(I - alpha A_hat, seed_matrix)
```

If batch cost is high, use fixed-step conjugate gradients or Neumann approximation:

```text
R s approx sum_{t=0}^{T} alpha^t A_hat^t s
```

The direct solve is cleaner for validating the idea.

## Why It Should Be More Successful Than The Other Linear Algebra Ideas

### Compared With Krylov

Krylov gives rich spectra but requires stable basis construction. Resolvent gives the useful all-step propagation directly and may train more easily.

### Compared With Schur-Ray

Schur-Ray is excellent for sliding lines. Resolvent can combine ray, knight, pawn, king, and defense structure in one operator.

### Compared With Harmonic Potential

Harmonic potential solves smooth diffusion. Resolvent can be directed, role-conditioned, and attacker/defender asymmetric.

### Compared With Matrix Pencil

Matrix pencil spectra are indirect. Resolvent transfer is directly tied to target reach and defensive cancellation.

## Ablations

| Ablation | Purpose |
|---|---|
| `no_resolvent_direct_pool` | Replace solve with ordinary pooling over seeds. |
| `neumann_1_step` | Tests whether all-length propagation matters. |
| `single_alpha` | Tests multi-scale propagation. |
| `fixed_operator_no_gates` | Tests board-conditioned operator adaptation. |
| `no_low_rank_update` | Tests board-specific tactical shortcut term. |
| `random_geometry_operator` | Tests chess geometry. |
| `attack_only_no_defense` | Tests defensive cancellation. |
| `cnn_same_params` | Size-matched conventional baseline. |

## Falsification Criteria

Reject or revise if:

```text
neumann_1_step matches full resolvent
or attack_only_no_defense matches attack-defense transfer
or random_geometry_operator matches chess geometry
or fixed_operator_no_gates matches adaptive operator
or near-puzzle FP does not improve over size-matched CNN
```

Serious benchmark target:

```text
test PR AUC > 0.8068
test F1 > 0.7445
near-puzzle FP < 0.2477
```

Ambitious target:

```text
test PR AUC >= 0.82
test F1 >= 0.76
near-puzzle FP <= 0.20
puzzle recall >= 0.78
```

## Counterexamples

This can fail if:

- puzzle signal is dominated by move-order proof rather than current-board influence
- the operator becomes too smooth and washes out tactical sharpness
- direct solves are numerically unstable without normalization
- near-puzzles and puzzles have similar attacker/defender transfer at shallow board level
- seed construction is weak

## Implementation Priority

If the goal is a successful linear-algebra NN, implement this before the more exotic spectral packets.

Recommended order:

1. Fixed chess geometry operator with attack/defense seeds.
2. Direct batched resolvent solve at one alpha.
3. Add multi-alpha readout.
4. Add learned gates.
5. Add low-rank board-conditioned update.
6. Add diagnostics and ablations.

Minimum viable architecture:

```text
A = gated ray + knight + pawn + king adjacency
alpha_values = [0.5]
seeds = attack, defense, king_target, material_target
features = attack_to_target, defense_to_target, net_pressure
```

Expected first useful diagnostic:

```text
near-puzzles should have higher defense cancellation than true puzzles
true puzzles should have higher net target transfer than near-puzzles
```

