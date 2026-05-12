# Codex Research Packet: Toda Isospectral Flow Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1620_tuesday_local_toda_isospectral_flow.md`
- Generated at: 2026-05-05 16:20
- Weekday: Tuesday
- Timezone: local
- Author: Claude (Opus 4.7, 1M context)
- Intended next consumer: Codex
- Status: full unorthodox-linear-algebra architecture packet

## One-Sentence Thesis

Build a learned tridiagonal symmetric chess operator `L`, evolve it via the **Toda lattice flow** `L̇ = [L, B(L)]` where `B(L)` is the strict lower-triangular minus strict upper-triangular projection — an *isospectral* flow that preserves all eigenvalues of `L` while sorting them onto the diagonal at `t → ∞`; classify puzzle-likeness from **flow-time invariants** (Manakov / Toda integrals of motion) and the **rate of diagonal sorting** (which encodes how "tactically separable" the position is) — using machinery from integrable systems that no spectrum-or-subspace packet exploits.

## Why This Is A Real Unorthodox Linear Algebra NN Idea

The **Toda lattice** (Flaschka 1974): for a tridiagonal symmetric matrix `L`,

```text
dL/dt = [L, B(L)]
```

where `B(L) = L_{lower} - L_{upper}` (the "Lax pair" structure) is **isospectral**:

```text
spec(L(t)) = spec(L(0))  for all t
```

and `L(t) → diag(eigenvalues, sorted)` exponentially fast as `t → ∞`. The flow is **completely integrable**: there are `n` independent conserved quantities (the eigenvalues, equivalently the elementary symmetric polynomials of `L`), and the dynamics decompose into action-angle variables.

Key invariants:

```text
manakov_integrals  H_k = (1/k) tr(L^k),   k = 1, ..., n      (conserved along flow)
sorting_rate       (1/T) log( ||L_off(0)||_F / ||L_off(T)||_F )
flow_time_to_eps   inf t : ||L_off(t)||_F < eps
diagonal_at_T      diag(L(T))   approximates sorted eigvals
```

This is fundamentally distinct from:

- **Krylov (i076)**: `A^k v` repeated action; not a flow on operators.
- **Resolvent (i077)**: `(zI - A)^{-1}` complex frequency; not a flow.
- **Lyapunov (i225)**: A^T P + PA = -Q; not isospectral.
- **Riccati (new)**: quadratic equation; not a flow.

The Toda flow is an evolution **on the space of operators** that preserves spectrum but sorts the diagonal — it's the *continuous* analog of the QR algorithm and gives a **flow-invariant signature** of the operator.

## Target

```text
fine 0,1 -> 0,  fine 2 -> 1.  3x2 fine-to-binary mandatory.
```

## Forbidden Inputs

Standard.

## Closest Existing Ideas And Exact Difference

- `i076, i077, i078, i062, i199`: spectrum-based, no isospectral flow.
- `i115 Neural Cellular Automaton`: discrete dynamics, not isospectral.
- `i131 Replicator`: simplex flow, not on operators.

```text
The Toda lattice is the unique completely integrable flow on tridiagonal symmetric
matrices preserving spectrum. Its conserved quantities (Manakov integrals) and its
sorting rate are operator-flow invariants no imported packet uses.
```

## Mathematical Thesis

### Definitions

Build a tridiagonal symmetric chess operator at full size `n = 64`:

```text
L = tridiag(b_i, a_i, b_i)
```

with `a_i, b_i` learned from chess primitives (e.g. `a_i` = aggregate-pressure on square `i` along an ordered traversal of the 64 squares; `b_i` = pairwise tactical-coupling between consecutive squares in the traversal). The square ordering is fixed (e.g. row-major) but a learned permutation `pi` reorders before tridiagonalization.

### Toda flow

Discretize `dL/dt = [L, B(L)]` with `B(L) = L_lower - L_upper` and forward-Euler at small `dt`:

```text
L(t + dt) = L(t) + dt * (L(t) B(t) - B(t) L(t))
```

Run `T = 32` steps. Track:

```text
||L_off(t)||_F          off-diagonal mass
diagonal entries         L(t)_{ii}
manakov_integrals_t     H_k(t) = (1/k) tr(L(t)^k)        (must be ~constant)
```

Use a higher-order integrator (RK4) for stability if Euler drifts.

### Readout

```text
sorting_rate              -log( ||L_off(T)|| / ||L_off(0)|| ) / T
manakov_drift             max_k | H_k(T) - H_k(0) | / | H_k(0) |     (sanity, ~0)
diag_at_T_sorted          top-8 sorted diagonal entries
spectral_gap_at_T         L(T)_{ii} - L(T)_{i+1,i+1} pairwise gaps
flow_residue_topk        top-k off-diagonal entries at time T (slowest-decaying)
```

Final:

```text
puzzle_logit = MLP([sorting_rate, gaps, residues, diag_at_T_sorted, board_pool])
```

## Assumptions

- Tactical positions correspond to operators with **slow Toda sorting** — the off-diagonal mass decays slowly because the spectrum has clusters of nearly-degenerate eigenvalues (tactical resonance).
- Non-tactical positions have well-separated eigenvalues and sort quickly to diagonal.
- Near-puzzles have a single slow-decaying off-diagonal entry; the `flow_residue_topk` exposes it.

## Claim / Hypothesis

The Toda sorting rate is a near-sufficient statistic for puzzle-likeness *given* the eigvalues. Central falsifier:

```text
shuffle_a_b: randomly permute the (a_i, b_i) sequences before flow.
            (Same multiset of eigvals after sorting; different flow trajectory.)
            if PR AUC doesn't drop, the flow path doesn't matter -- only the spectrum.

eigvals_swap: replace flow features with sorted eigvals(L_0).
            if PR AUC doesn't drop, isospectral structure adds nothing.
```

## Architecture

```text
board_encoder
square_ordering_pi          -> fixed or learned permutation
L_builder                   -> tridiag symmetric n=64
toda_flow_block             -> T=32 unrolled RK4 steps
flow_features_block         -> sorting rate, gaps, residues
puzzle_head
```

### First config

```yaml
model:
  name: toda_isospectral_flow_network
  input_channels: 18
  num_classes: 1
  channels: 64
  hidden_dim: 96
  toda_n: 64
  toda_steps_T: 32
  toda_dt: 0.05
  integrator: rk4
training:
  mode: puzzle_binary
  loss: bce_with_logits
  aux_loss_manakov_drift: 1.0e-3
```

## Numerical / Compute Notes

- Each Toda step on tridiagonal `L` is `O(n)` for the commutator (sparse). Total flow: `T * n = 2048` per board. Negligible.
- RK4 + tiny `dt = 0.05` keeps Manakov integrals constant to 1e-6 over 32 steps.
- The auxiliary loss penalizes Manakov drift, ensuring the discrete flow stays nearly isospectral.

## Required Ablations

| Ablation | Removes | Hypothesis |
|---|---|---|
| `shuffle_a_b` | random permutation of tridiag entries | tests path |
| `eigvals_swap` | use sorted eigvals(L) | tests isospectral-flow value |
| `T_eq_4` | drop to 4 flow steps | tests flow depth |
| `euler_only` | replace RK4 with forward Euler (drifts more) | tests integrator |
| `random_geometry` | random a_i, b_i unrelated to chess | tests chess content |
| `cnn_same_params` | matched CNN | baseline |
| `i076_krylov_baseline` | adjacent baseline | baseline |
| `i199_hessian_baseline` | adjacent baseline | baseline |

## Benchmark Targets

```text
PR AUC >= 0.82, F1 >= 0.76, near-puzzle FPR <= 0.20, puzzle recall >= 0.78
shuffle_a_b drops PR AUC >= 0.005
eigvals_swap drops PR AUC >= 0.015 (key claim: flow > spectrum)
beats i076 by >= 0.01 PR AUC at matched params
```

## Counterexamples

- Toda flow on tridiagonal of size 64 is nearly-degenerate fast; sorting rate barely varies.
- The fixed square ordering forces an arbitrary tridiagonal structure; a Hermitian non-tridiagonal operator would be more natural.
- Forward Euler drifts; Manakov integrals are not preserved; "isospectral" is only approximate.

## Implementation Priority

1. Build tridiagonal `L` from `simple_18` features.
2. Implement RK4 Toda step; verify Manakov drift < 1e-5 over 32 steps on random init.
3. Read off sorting rate and flow residues; train minimal head.
4. Run 8 ablations.
