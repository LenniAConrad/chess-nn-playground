# Codex Research Packet: Tactical Controllability Gramian Network

## File Metadata

- Filename: `chess_nn_research_2026-04-25_2004_saturday_shanghai_tactical_controllability_gramian.md`
- Generated at: 2026-04-25 20:04
- Weekday: Saturday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: full linear-algebra architecture packet, not implemented, not benchmark results

## One-Sentence Thesis

Treat a chess position as a small linear control system where attacking pieces are inputs, tactical targets are outputs, and the puzzle signal is measured by controllability/observability Gramians, Hankel singular values, and attacker-vs-defender transfer energy.

## Why This May Be More Successful Than The Resolvent Idea

The adaptive resolvent asks:

```text
how much influence reaches a target after all-length propagation?
```

This Gramian idea asks a sharper systems question:

```text
how controllable are critical targets from attacker inputs,
how observable are tactical weaknesses from target outputs,
and can defender inputs cancel the same modes?
```

That is closer to puzzle-vs-near-puzzle discrimination. A near-puzzle can have visible pressure, but if defender modes control or cancel the same target subspace, it is not a true puzzle. A true puzzle should have attacker-controllable, target-observable modes that the defender cannot span.

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

Allowed:

- current-board tensor
- deterministic chess geometry
- learned operators and gates derived only from current-board features

## Closest Existing Ideas

### Closest Packets

- `Adaptive Tactical Resolvent Network`
- `Krylov Tactical Subspace Network`
- `Schur-Ray Line Algebra Network`
- `Matrix-Pencil Generalized Spectrum Bottleneck`
- `Tactical Equilibrium Network`

### Exact Overlap

```text
All model tactical interaction through structured propagation, spectra, or attacker/defender comparison.
```

### Exact Difference

```text
Tactical Controllability Gramian Network uses linear-systems Gramians and Hankel transfer modes: Wc, Wo, C Wc C^T, B^T Wo B, and singular values of Wo^{1/2} Wc^{1/2}. It is not a resolvent transfer readout, Krylov basis, Schur line solve, matrix pencil, or game equilibrium.
```

## Mathematical Thesis

### Linear System

Represent the board as a stable linear system over square states:

```text
h_{t+1} = A h_t + B_a u_a + B_d u_d
y_t     = C h_t
```

where:

- `A in R^{64 x 64}` is a learned chess-structured propagation operator.
- `B_a` injects attacker influence from side-to-move pieces and tactical seeds.
- `B_d` injects defender influence from opponent pieces/resources.
- `C` reads out critical targets: king zone, high-value pieces, promotion squares, overloaded defenders, line intersections.

All matrices are produced from current-board features and deterministic geometry.

### Gramians

Stabilize:

```text
A_hat = A / max(1, spectral_norm_estimate(A) + eps)
```

Compute attacker controllability Gramian:

```text
W_a = A_hat W_a A_hat^T + B_a B_a^T
```

Compute defender controllability Gramian:

```text
W_d = A_hat W_d A_hat^T + B_d B_d^T
```

Compute target observability Gramian:

```text
W_o = A_hat^T W_o A_hat + C^T C
```

Equivalent finite approximation:

```text
W_a approx sum_{k=0}^{K} A^k B_a B_a^T (A^T)^k
W_d approx sum_{k=0}^{K} A^k B_d B_d^T (A^T)^k
W_o approx sum_{k=0}^{K} (A^T)^k C^T C A^k
```

### Tactical Readouts

Attacker target reach:

```text
T_a = trace(C W_a C^T)
```

Defender target reach:

```text
T_d = trace(C W_d C^T)
```

Net tactical controllability:

```text
T_net = T_a - T_d
```

Attacker modes visible to targets:

```text
H_a = singular_values(W_o^{1/2} W_a W_o^{1/2})
```

Defender cancellation modes:

```text
H_d = singular_values(W_o^{1/2} W_d W_o^{1/2})
```

Subspace mismatch:

```text
angle(attacker_control_subspace, defender_control_subspace)
```

Final logit:

```text
puzzle_logit = MLP([
  T_a, T_d, T_net,
  top Hankel-like singular values,
  attacker/defender mode ratios,
  target-specific Gramian diagonals,
  subspace angles,
  board context
])
```

## Assumptions

- True puzzles contain attacker-controlled tactical modes that are observable at critical targets.
- Near-puzzles often have attacker pressure, but defender control spans or cancels the same target modes.
- Gramians capture multi-step propagation more robustly than local CNN filters.
- The 64-square system is small enough for stable differentiable Gramian approximations.

## Claim

Hypothesis: controllability/observability Gramian features should reduce near-puzzle false positives more than plain CNN/BT4-style encoders because they explicitly compare attacker control, defender control, and target observability in the same linear system.

## Mechanism

A true puzzle should have:

```text
high T_a
low or insufficient T_d
large attacker observable Hankel modes
poor defender alignment with attacker target modes
```

A near-puzzle should often have:

```text
moderate/high T_a
also high T_d
strong defender cancellation modes
small or ambiguous T_net
```

This is exactly the near-puzzle failure mode: high apparent attack, but adequate defense.

## Architecture

### Components

```text
board_encoder
stable_operator_builder A
attacker_input_builder B_a
defender_input_builder B_d
target_output_builder C
gramian_solver
modal_readout
puzzle_head
```

### Forward Pass

```text
X = board_encoder(board)
A = stable_operator_builder(X)
B_a = attacker_input_builder(X)
B_d = defender_input_builder(X)
C = target_output_builder(X)
W_a = controllability_gramian(A, B_a)
W_d = controllability_gramian(A, B_d)
W_o = observability_gramian(A, C)
features = gramian_readout(W_a, W_d, W_o, C)
logit = puzzle_head([features, pool(X)])
```

### First Config

```yaml
model:
  name: tactical_controllability_gramian_network
  input_channels: 18
  num_classes: 1
  hidden_dim: 96
  operator_rank: 12
  input_rank: 8
  target_rank: 8
  gramian_steps: 6
  stable_operator_norm: true
  readout_modes: 12
training:
  mode: puzzle_binary
  loss: bce_with_logits
```

## Solver Choice

Version 1 should use finite unrolled sums:

```text
W = B B^T
for k in range(K):
    W = B B^T + A W A^T
```

This is simple, differentiable, and avoids tricky Lyapunov solvers. Since `64 x 64` is small, full matrices are feasible.

Later version:

```text
solve_discrete_lyapunov(I - A kron A, vec(BB^T))
```

but only if the unrolled version shows signal.

## Why This Could Be The Most Successful Linear Algebra Architecture

### It is directly matched to chess tactics

Tactics are about whether force can reach targets and whether defense can cover them. Controllability and observability are exactly the linear-algebra concepts for input-to-target reachability.

### It is more stable than Krylov

No explicit orthogonalization is required.

### It is more diagnostic than resolvent

Resolvent gives transfer. Gramians separate:

```text
attacker control
defender control
target observability
mode overlap
```

### It is implementable

The board is only 64 squares. Unrolled Gramian sums are practical.

## Ablations

| Ablation | Purpose |
|---|---|
| `attacker_only` | Tests defender cancellation. |
| `defender_only` | Tests defensive modes alone. |
| `no_observability` | Removes target observability Gramian. |
| `one_step_gramian` | Tests multi-step controllability. |
| `random_target_C` | Tests target semantics. |
| `random_geometry_A` | Tests chess propagation geometry. |
| `fixed_A_no_gates` | Tests board-conditioned dynamics. |
| `diag_only_gramian` | Tests full matrix interactions. |
| `cnn_same_params` | Conventional baseline. |

## Falsification Criteria

Reject or revise if:

```text
one_step_gramian matches full multi-step Gramian
or attacker_only matches attacker-vs-defender model
or no_observability matches full model
or random_target_C matches chess target C
or random_geometry_A matches chess geometry A
```

Minimum serious benchmark target:

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

## Expected Diagnostics

If the idea is working:

```text
T_a(puzzle) > T_a(near) > T_a(random)
T_net(puzzle) > T_net(near)
T_d(near) should often be higher than T_d(puzzle) for false-positive-like positions
attacker/defender subspace overlap should be lower for true puzzles than near-puzzles
```

The most important diagnostic:

```text
near-puzzle examples predicted non-puzzle should show strong defender cancellation modes
near-puzzle false positives should reveal whether candidate generation or Gramian readout is failing
```

## Counterexamples

This can fail if:

- puzzle signal depends on exact move order rather than reachability
- attacker and defender control modes look similar before deeper search
- the learned operator is too smooth
- target matrix `C` misses the relevant tactical target
- board-conditioned gates overfit to source artifacts

## Implementation Priority

If implementing one linear-algebra architecture for actual benchmark success, use this order:

1. Fixed chess geometry `A`.
2. Hand-initialized attacker/defender/target seeds from board tensor.
3. Unrolled finite Gramian with `K=4`.
4. Read out `T_a`, `T_d`, `T_net`, diagonal target energies.
5. Add learned gates.
6. Add top singular values of low-rank Gramian projections.
7. Add ablations.

Minimum viable version:

```text
A: fixed ray + knight + pawn + king adjacency
B_a: side-to-move attack seed matrix
B_d: opponent defense seed matrix
C: king-zone + high-value target output matrix
K: 4 Gramian steps
readout: trace(C W_a C^T), trace(C W_d C^T), difference, ratios
```

This version should be implementable before the more exotic Krylov or matrix-pencil packets.

