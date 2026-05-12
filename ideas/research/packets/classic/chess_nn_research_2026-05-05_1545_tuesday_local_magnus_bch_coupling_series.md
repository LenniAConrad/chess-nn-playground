# Codex Research Packet: Magnus-BCH Operator-Coupling Series Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1545_tuesday_local_magnus_bch_coupling_series.md`
- Generated at: 2026-05-05 15:45
- Weekday: Tuesday
- Timezone: local
- Author: Claude (Opus 4.7, 1M context)
- Intended next consumer: Codex
- Status: full unorthodox-linear-algebra architecture packet, not implemented, not benchmark results

## One-Sentence Thesis

Build attacker `A` and defender `B` operators, compute their **Baker-Campbell-Hausdorff
log** `Z = log(exp(A) exp(B))` as a truncated **Magnus series** of *nested commutators*

```text
Z = A + B + 1/2 [A,B] + 1/12 ([A,[A,B]] - [B,[A,B]]) + 1/24 [B,[A,[A,B]]] + ...
```

and use the **norms of the higher-order Magnus terms** as a non-classical coupling
fingerprint -- the weight that *actually breaks* commutativity beyond the first
commutator -- which the kinematic-commutator packet (i040) cannot detect because that
packet stops at degree 2.

## Why This Is A Real And Unorthodox Linear Algebra NN Idea

For non-commuting linear operators `A, B`, `exp(A) exp(B) <> exp(A + B)` in general.
The **BCH formula**:

```text
log(exp(A) exp(B)) = A + B + 1/2 [A, B]
                   + 1/12 ( [A, [A, B]] - [B, [A, B]] )
                   - 1/24  [B, [A, [A, B]]]
                   + ... (Lyndon basis, all higher commutators)
```

Equivalently, the **Magnus expansion** of the time-ordered solution
`d/dt U = (A + t B) U` produces a series whose `k`-th term `Omega_k` is a homogeneous
weight-`k` polynomial in nested commutators of `A, B`.

The bet:

- A "trivial" coupling has `||[A, B]|| > 0` but `||[A, [A, B]]|| ~ 0`: the operators
  *almost* commute up to first order.
- A *combination* (multi-step puzzle) requires *higher* iterated commutators to be
  large -- the commutator depth is exactly the tactical depth.
- This is provably distinct from i040 Kinematic Commutator, which stops at the *single*
  commutator `[A, B]` (degree 2). Here we go to degree 4.

The Magnus / BCH structure also exposes the **derived series**

```text
g_0 = span{A, B}
g_1 = g_0 + [g_0, g_0]
g_2 = g_1 + [g_1, g_1]
...
```

The dimension growth `dim(g_k)` is the *Hall set* / Lyndon word counting:

```text
dim(weight-k Lie monomials) = (1/k) sum_{d | k} mu(k/d) 2^d        (Witt's formula)
```

For `(A, B)` at weight 4, there are `dim = 9` independent Lie monomials. Their
coefficients in the Magnus series form a 9-dimensional fingerprint per board.

## Target

```text
fine 0,1 -> binary 0
fine 2   -> binary 1
3x2 fine-to-binary matrix mandatory.
```

## Forbidden Inputs

Standard.

## Closest Existing Ideas And Exact Difference

### Closest registered

- `i040 Kinematic Commutator Bottleneck` — uses *single* commutators `[op_i, op_j]`,
  never higher orders. The packet explicitly stops at degree 2.
- `i075 Tactical Bisimulation` — uses bisimulation, not Lie expansions.
- `i119 Tensor-Ring Square Interaction` — tensor decomposition, not Lie series.
- `i076 Krylov Subspace` — `A^k v`, not nested commutators.

### Exact difference

```text
The Magnus / BCH series is the unique non-commutative-log of the product of
exponentials, with terms indexed by Hall basis Lie monomials at every weight. Higher
weights (>=3) capture *iterated* operator coupling that single-commutator features
provably cannot. The dimension count grows as Witt's formula and gives a precise
weight-by-weight tactical depth profile.
```

## Mathematical Thesis

### Definitions

Build attacker `A` and defender `B` at low rank `r = 12` from the current board (gates
on legal-geometry primitives, as in earlier packets).

Spectral-normalize so `||A||_2, ||B||_2 <= 1/2` (this guarantees BCH convergence with
some safety margin, since the radius is roughly `log 2 ~ 0.69`).

### Hall basis at weight <= 4

Compute these 9 nested commutators (a specific Hall ordering):

```text
weight 1:  A,  B                                                      2 monomials
weight 2:  [A, B]                                                     1
weight 3:  [A, [A, B]],  [B, [A, B]]                                  2
weight 4:  [A, [A, [A, B]]], [B, [A, [A, B]]], [A, [B, [A, B]]],
           [B, [B, [A, B]]]                                            4
                                                                  -------
                                                                       9
```

(Witt: `(2 - 1) + (1) + (2) + (4) = ... -> matches Hall basis at weight 4.)

Magnus-coefficient weights (BCH coefficients):

```text
A:                            1
B:                            1
[A,B]:                        1/2
[A,[A,B]]:                    1/12
[B,[A,B]]:                   -1/12
[B,[A,[A,B]]]:                1/24    (unique non-zero degree-4 term in BCH)
others (degree-4):            0 in BCH but visible as Magnus / Hall basis components
```

Compute each commutator and read its Frobenius norm:

```text
c_2  = [A, B]
c_3a = [A, c_2]
c_3b = [B, c_2]
c_4a = [A, c_3a]
c_4b = [B, c_3a]
c_4c = [A, c_3b]
c_4d = [B, c_3b]
```

### Readout

```text
||A||_F, ||B||_F                                       baseline
||c_2||_F                                              i040-style feature (subsumed)
||c_3a||_F, ||c_3b||_F                                 weight-3 fingerprint
||c_4a||_F, ||c_4b||_F, ||c_4c||_F, ||c_4d||_F         weight-4 fingerprint
bch_log_norm = ||A + B + 1/2 c_2 + 1/12 (c_3a - c_3b) + 1/24 c_4b||_F
trace_features:  tr(c_3a^T c_3a), etc.
weight_decay_ratio = ||c_4||_F / ||c_3||_F             rate of Lie tail decay
weight_decay_ratio = ||c_3||_F / ||c_2||_F
```

Final:

```text
puzzle_logit = MLP([all_norms, bch_log_norm, decay_ratios, board_pool])
```

## Assumptions

- Multi-step tactical depth correlates with the slow decay of `||c_k||_F` as `k`
  increases.
- A near-puzzle has nontrivial `c_2` but rapidly decaying `c_3, c_4`.
- A puzzle / combination has slow decay -- iterated commutators stay large.
- BCH convergence is enforced by spectral-normalization to keep `||A||_2 + ||B||_2 <
  log 2`.

## Claim / Hypothesis

The decay rate of `||c_k||_F` from `k = 2` to `k = 4` is a concentrated puzzle vs
near-puzzle separator. Central falsifier:

```text
weight_2_only: drop c_3, c_4 features.
              if PR AUC doesn't drop, weight 2 alone (i040 territory) suffices.

degree_preserving_random_perm: replace c_4a with [A, P c_3a P^T] for random orthogonal
                              P; norms preserved approximately, structure destroyed.
                              if PR AUC doesn't drop, only norms matter.
```

## Architecture

### Components

```text
board_encoder
operator_A_builder, operator_B_builder
spectral_normalize  (||A||_2, ||B||_2 <= 1/2)
lie_basis_block      -> {c_2, c_3a, c_3b, c_4a, c_4b, c_4c, c_4d}
lie_norms_block      -> Frobenius norms
bch_log_block        -> truncated BCH log evaluated at degree 4
puzzle_head
```

### Forward pseudocode

```text
X_sq   = board_encoder(board)
A, B   = build_AB(X_sq)
A      = A / max(2 * ||A||_2, 1)
B      = B / max(2 * ||B||_2, 1)
c2     = A @ B - B @ A
c3a    = A @ c2 - c2 @ A
c3b    = B @ c2 - c2 @ B
c4a    = A @ c3a - c3a @ A
c4b    = B @ c3a - c3a @ B
c4c    = A @ c3b - c3b @ A
c4d    = B @ c3b - c3b @ B
norms  = [||A||_F, ||B||_F, ||c2||_F, ||c3a||_F, ||c3b||_F,
          ||c4a||_F, ||c4b||_F, ||c4c||_F, ||c4d||_F]
bch_log_F = || A + B + 0.5 c2 + (1/12)(c3a - c3b) + (1/24) c4b ||_F
ratios = [||c3a||/||c2||, ||c4a||/||c3a||, ...]
logit  = MLP([norms, bch_log_F, ratios, pool(X_sq)])
```

### First config

```yaml
model:
  name: magnus_bch_coupling_series_network
  input_channels: 18
  num_classes: 2
  hidden_dim: 96
  operator_rank_r: 12
  bch_truncation_degree: 4
  spectral_clip_per_op: 0.5
  num_M_primitives: 12
training:
  mode: puzzle_binary
  loss: bce_with_logits
  batch_size: 512
  learning_rate: 1.0e-3
```

## Numerical / Compute Notes

- `r = 12`. Each commutator is two `r x r` matrix products = `2 * O(r^3) = 3456` flops;
  total 7 commutators = `2.4e4` flops per board. Negligible.
- Spectral-normalize `A`, `B` per-board: differentiable estimate via 2-iter power
  method.
- BCH at degree 4 has *one* additional Hall-coefficient (weight-4): `[B, [A, [A, B]]]
  = c_4b`. The Hall basis at weight 4 has 4 monomials, of which only `c_4b` enters
  BCH. The other 3 monomials enter the *Lyndon-Hall expansion* but have zero BCH
  coefficient -- exposing them as features captures structure that BCH itself ignores
  but which is still chess-meaningful (hence the value beyond just "compute log of
  product").
- Backward: standard autograd; commutator products are linear in matmuls.

## Required Ablations

| Ablation | Removes | Hypothesis |
|---|---|---|
| `weight_2_only` | drop weight-3 and weight-4 commutators | tests need for degree > 2 |
| `weight_3_only` | drop weight-4 | tests degree-4 contribution |
| `norms_only` | drop `bch_log_F`, keep just `||c_k||_F` | tests need for log itself |
| `swap_AB` | swap attacker/defender roles | tests asymmetry |
| `commutator_random_replace` | replace `c_3a` with `[A, P c_2 P^T]` | tests structure |
| `random_geometry_M` | randomize fixed primitives | tests chess semantics |
| `cnn_same_params` | matched CNN | baseline |
| `i040_commutator_baseline` | run i040 on same A, B | adjacent baseline |

For each: full 3x2 + slice reports, with extra slice on `crtk_difficulty` because the
hypothesis predicts strongest gain on harder / multi-step puzzles.

## Benchmark Targets

```text
test PR AUC      >= 0.82
test F1          >= 0.76
near-puzzle FPR  <= 0.20
puzzle recall    >= 0.78

central claim:
  weight_2_only drops PR AUC >= 0.01 (else i040 is enough)
  weight_3_only drops PR AUC >= 0.005
  beats i040 by >= 0.015 PR AUC
  the lift concentrates on harder difficulty buckets
```

## Counterexamples / Failure Modes

- All higher commutators decay at the same exponential rate independent of class -- so
  weight-3, 4 are redundant with `||c_2||`.
- BCH series convergence is marginal at the spectral norms encountered; iterated
  commutators are dominated by floating-point noise.
- `r = 12` is too small to expose iterated structure.
- `[B, [A, [A, B]]]` and similar 4-term commutators are dominated by
  `||A||^3 ||B|| + ||A||^2 ||B||^2 ...` numerical scaling rather than chess content.
  Mitigation: feature normalization by `||A||_F^k ||B||_F^{4-k}` to extract pure
  structure.

## Implementation Priority

1. Build `A`, `B` builders and verify spectral-normalization stability.
2. Compute up to weight-4 commutators, sanity-check via Jacobi identity:
   `[A, [B, c]] + [B, [c, A]] + [c, [A, B]] = 0` -- must hold to numerical precision.
3. Add normalized-norm features (divide by `||A||^k ||B||^{4-k}`).
4. Train; check that `weight_2_only` ablation degrades.
5. Run all 8 ablations with extra difficulty-slice attention.

Smallest viable version:

```text
r = 8, only A, B, c_2, c_3a, c_3b features; no weight-4. (A "minimal-Magnus" version.)
```

If lift over `i040`, scale to `r = 12` and add weight-4 monomials.
