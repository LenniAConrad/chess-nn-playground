# Codex Research Packet: Schur-Complement Defender Elimination Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1505_tuesday_local_schur_complement_defender.md`
- Generated at: 2026-05-05 15:05
- Weekday: Tuesday
- Timezone: local
- Author: Claude (Opus 4.7, 1M context)
- Intended next consumer: Codex
- Status: full linear-algebra architecture packet, not implemented, not benchmark results

## One-Sentence Thesis

Build a learned PSD interaction matrix `M` block-partitioned into attacker and defender
square-blocks `M = [[A, B], [B^T, D]]`, then form the **Schur complement**
`S = D - B^T A^{-1} B` and classify puzzle-likeness from `S`'s spectrum and signature;
the Schur complement is the exact residual defender geometry after attacker influence is
algebraically eliminated, which no spectrum-only or graph-Laplacian packet exposes.

## Why This Is A Real Linear Algebra NN Idea

The Schur complement is the canonical answer to *"what does the defender system look
like after the attacker subsystem is algebraically removed?"*. It appears in:

```text
[ A    B ] [ x ]   [ f ]
[ B^T  D ] [ y ] = [ g ]
```

eliminating `x = A^{-1}(f - B y)` gives the reduced equation

```text
(D - B^T A^{-1} B) y = g - B^T A^{-1} f
```

so `S = D - B^T A^{-1} B` is the **effective defender response**. Sylvester equations
involve a coupled solve (different idea); generalized eigenproblems compare two operators
elementwise; the Schur complement *eliminates* one block to expose the other.

Spectral and inertia properties of `S`:

- `inertia(M) = inertia(A) + inertia(S)` (Haynsworth inertia additivity).
- `det(M) = det(A) det(S)`.
- Negative eigenvalues of `S` mean the defender block, *after attacker compensation*, is
  unstable -- a clean linear-algebraic signature of tactical insolvency.

## Target

```text
fine 0,1 -> binary 0
fine 2   -> binary 1
3x2 fine-to-binary matrix mandatory.
```

## Forbidden Inputs

Same blanket: no engine, no PV, no source labels, no future moves, no legal-move tree
expansion. Only current-board tensors and rule-derived geometry.

## Closest Existing Ideas And Exact Difference

### Closest registered ideas

- `i068 Schur-Ray Line Algebra Network` — uses the *Schur* of ray algebras (factorization
  step in eigensolvers), not the Schur *complement* of a block matrix.
- `i058 Determinantal Tactical Volume Bottleneck` — uses `det` of a single matrix.
- `i208 Pinned Mobility Nullspace` — nullspace, not block elimination.
- `i078 Controllability Gramian` — output reachability, not block elimination.
- `i136 Low-Rank Signed Cut Query` — cut-style, no block partition with elimination.

### Exact difference

```text
Schur complement is a *block* operation that algebraically removes the attacker
subsystem and exposes the residual defender response. No imported packet partitions the
interaction matrix into [attacker | defender] blocks and computes the residual operator
S = D - B^T A^{-1} B with its inertia and spectrum as the central invariant.
```

## Mathematical Thesis

### Definitions

Partition the 64 squares into a *soft* attacker mask `m_A in [0,1]^{64}` and a *soft*
defender mask `m_D in [0,1]^{64}` derived from current board piece-occupancy and
side-to-move (no engine). Order squares by descending `m_A` to put attacker block first.

Build a learned PSD interaction `M in R^{64 x 64}`:

```text
M = G + lambda I,    G = sum_k g_k(X_sq) (E_k + E_k^T) / 2,
```

with `E_k` fixed legal-geometry adjacency primitives (ray-line, knight, pawn-attack,
defender-line, pin-axis, king-shelter). `lambda > 0` keeps `M` PSD.

Permute and partition:

```text
M = [ A   B ]      A in R^{a x a},  D in R^{d x d},  B in R^{a x d},  a+d = 64
    [ B^T D ]
```

with `a` = soft attacker count rounded, `d = 64 - a`.

### Schur complement

```text
S = D - B^T A^{-1} B
```

Computed via:

```text
solve A Z = B    (a x d, Cholesky on A + epsilon I)
S = D - B^T Z
```

Differentiable. Cost `O(a^3 + a^2 d)`.

### Readout

```text
inertia_S        = (n_pos, n_zero, n_neg)     soft via tanh(beta * eig(S))
top_k_eigvals_S  = sorted eigenvalues, top k = 8
log|det(S)|      = log determinant
trace(S), trace(S^2)
nuclear_norm(S) / spectral_norm(S)            stable rank
ratio  log|det(M)| - log|det(A)|              must equal log|det(S)|; sanity feature
```

Final:

```text
puzzle_logit = MLP([inertia_S, eigvals_topk, log_dets, ratios, board_pool])
```

## Assumptions

- A true puzzle has a defender block whose response `S = D - B^T A^{-1} B` has at least
  one significantly negative eigenvalue, signaling that no defender configuration can
  compensate the algebraic attacker pressure -- "tactically insolvent."
- A near-puzzle may have large `||B||` but compatible `D`, so `S` stays near-PSD.
- Soft attacker/defender masks derived from current board occupancy + side-to-move are
  enough; we do not need legal-move expansion.

## Claim / Hypothesis

The Haynsworth inertia of `S` -- specifically the count of negative eigenvalues
weighted by their magnitude -- is a near-sufficient statistic for puzzle-likeness once
the global board encoder is also given. The **central falsifier** is that the
`inertia_only` ablation (no eigenvalues, only signed counts) should approach the full
model on aggregate metrics; conversely, removing inertia and keeping only norms should
underperform.

## Architecture

### Components

```text
board_encoder              -> X_sq in R^{64 x d}
attacker_mask_builder      -> m_A in [0,1]^{64}            (rule-derived, side-to-move)
defender_mask_builder      -> m_D in [0,1]^{64}
psd_M_builder              -> M in R^{64 x 64} PSD
soft_block_partition       -> A, B, D
schur_solver               -> S
spectral_readout           -> phi(S)
puzzle_head                -> logit
```

### Forward pseudocode

```text
X_sq        = board_encoder(board)
m_A, m_D    = mask_builder(board, X_sq)
M           = psd_M_builder(X_sq)
P           = soft_permute(M, m_A)         # straight-through to a hard reorder
A, B, D     = block_split(P @ M @ P.T, a)
A_reg       = A + epsilon I
Z           = cholesky_solve(A_reg, B)
S           = D - B.T @ Z
feat        = spectral_readout(S)
logit       = MLP([feat, pool(X_sq)])
```

`a` (attacker block size) is fixed per batch as `a = round(sum(m_A))` clipped to
`[8, 56]`; in the very first version, fix `a = 32` and let `m_A` only choose *which* 32
squares (top-32 by score, with straight-through gradients).

### First config

```yaml
model:
  name: schur_complement_defender_network
  input_channels: 18
  num_classes: 2
  hidden_dim: 96
  block_a_size: 32
  num_psd_primitives: 12
  psd_floor_epsilon: 1.0e-3
training:
  mode: puzzle_binary
  loss: bce_with_logits
  batch_size: 512
  learning_rate: 1.0e-3
```

## Numerical / Compute Notes

- Cholesky-of-`A + eps I` keeps it stable. Total cost `O(64^3) = 2.6e5 flops`, free vs
  the CNN trunk.
- Top-k eigvals of `S` (`d x d <= 56x56`) via Lanczos with `k = 8`, fully differentiable
  via PyTorch `torch.linalg.eigh` on the symmetric part.
- Sanity feature `log|det(M)| - log|det(A)| - log|det(S)|` should be exactly zero;
  exposing it as a feature lets the network detect numerical drift, but it should never
  carry training signal -- monitor only.

## Required Ablations

| Ablation | Removes | Hypothesis |
|---|---|---|
| `no_block_partition` | use full `M` spectrum, no Schur | tests block elimination |
| `random_attacker_mask` | randomize `m_A` while preserving `|m_A|` | tests attacker-defender semantics |
| `swap_attacker_defender` | swap `m_A` and `m_D` | tests asymmetry / side-to-move |
| `inertia_only` | drop eigvals, keep `(n_pos, n_zero, n_neg)` | tests sufficiency of inertia |
| `eigvals_only` | drop inertia, keep eigvals | tests sufficiency of magnitudes |
| `zero_off_diagonal` | force `B = 0` so `S = D` | tests cross-block coupling |
| `random_geometry` | replace fixed adjacency `E_k` with random sparse | tests chess semantics |
| `cnn_same_params` | size-matched CNN | matched-capacity baseline |
| `i001_baseline` | run i001 | linear-algebra baseline |

Per ablation: full 3x2 + slices on difficulty / phase / eval-bucket / tactic-motifs.

## Benchmark Targets

```text
test PR AUC     >= 0.82
test F1         >= 0.76
near-puzzle FPR <= 0.20
puzzle recall   >= 0.78

central claim: random_attacker_mask drops PR AUC by >= 0.02
               zero_off_diagonal       drops PR AUC by >= 0.02
               inertia_only            stays within 0.01 of full model
```

## Counterexamples / Failure Modes

- Soft attacker/defender masks are too coarse, so `A`/`D` end up near-symmetric and
  Schur complement is uninformative.
- The PSD floor `eps I` dominates, masking the resonance.
- Most puzzles do not in fact create a negative-eigenvalue Schur complement at the
  block size we chose.
- Block-size choice `a = 32` is suboptimal; would need an adaptive split.

## Implementation Priority

1. Implement attacker / defender mask from piece occupancy + side-to-move (no legal
   moves).
2. Build `M` as `eps I + sum_k g_k (E_k + E_k^T)/2` with fixed `E_k` legal adjacency.
3. Cholesky-solve `A Z = B`, compute `S = D - B^T Z`.
4. Top-8 eigvals + soft inertia readout.
5. Run all 9 ablations.

Smallest viable version:

```text
fixed E_k masks, fixed a = 32, readout = soft inertia + log|det(S)|, no learned gates.
```

If lift over CNN-same-params is positive, add learned gates and fuller readout.
