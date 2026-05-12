# Codex Research Packet: Lindström-Gessel-Viennot Path Determinant Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1615_tuesday_local_lindstrom_gessel_viennot_path.md`
- Generated at: 2026-05-05 16:15
- Weekday: Tuesday
- Timezone: local
- Author: Claude (Opus 4.7, 1M context)
- Intended next consumer: Codex
- Status: full unorthodox-linear-algebra architecture packet

## One-Sentence Thesis

Apply the **Lindström-Gessel-Viennot (LGV) lemma**: choose `k` attacker source squares `(s_1, ..., s_k)` and `k` target squares `(t_1, ..., t_k)`; build the `k x k` path-counting matrix `M_{ij} = w(s_i → t_j)` where `w` is a learned single-path generating function over chess geometry; then `det(M)` equals the **signed sum of k-tuples of non-intersecting paths** from sources to targets — a single determinant counting tactical motifs (overloads, x-rays, double attacks) with sign cancellation, distinct from the determinantal volume bottleneck (i058) and from the Pfaffian (i226, which counts matchings).

## Why This Is A Real Unorthodox Linear Algebra NN Idea

The **Lindström-Gessel-Viennot lemma**: for sources `S = (s_1, ..., s_k)` and sinks `T = (t_1, ..., t_k)` in a DAG with edge weights `w_e`,

```text
det( M_{ij} = sum_{paths p: s_i -> t_j} prod_e w_e ) =
sum_{non-intersecting k-tuples (p_1, ..., p_k)}  sgn(sigma)  prod prod w_e
```

where `sigma` is the permutation `s_i -> t_{sigma(i)}`. If `S, T` are arranged so that the only non-intersecting matching is the identity, then `det(M)` is the **unsigned count of non-intersecting path families** — a hard combinatorial invariant computable as a single determinant.

For chess, the bet:

- **Attackers competing for non-intersecting paths to multiple targets** is the precise notion of a "double attack" or "overload" tactic.
- A puzzle is a position where the LGV determinant is *large* (many disjoint attacker-to-target route assignments) and the dual `Pfaffian` for unsigned matchings is *small* (defenders cannot pair attackers off).
- Unlike DPP volume (i058) which uses an arbitrary kernel determinant, LGV uses the *path-generating-function* determinant — a combinatorial object with a precise non-intersecting-paths interpretation.

## Target

```text
fine 0,1 -> 0,  fine 2 -> 1.  3x2 fine-to-binary mandatory.
```

## Forbidden Inputs

Standard. Importantly: paths are **lattice paths in static board geometry** (rays, knight-graphs), not legal-move trees. We do not enumerate legal moves; we enumerate edges in a deterministic chess incidence graph and weight them.

## Closest Existing Ideas And Exact Difference

- `i058 Determinantal Tactical Volume` — uses DPP `det(K)` for a Gram-style kernel `K`; no path-counting interpretation.
- `i226 Pfaffian Skew Threat` — counts perfect matchings, not paths.
- `i070 Relational Query Algebra` — composes relations, but not via LGV determinant.
- `i055 Non-backtracking walk` — single-walker, not k disjoint paths.

```text
LGV is the unique combinatorial determinant whose value enumerates non-intersecting k-tuples of paths in a DAG, with sign for matching permutations. Its formal object is a path-generating-function matrix, not a Gram matrix. No imported packet computes a path-generating-function determinant for chess attacker-target tuples.
```

## Mathematical Thesis

### Definitions

Build a deterministic chess DAG `G = (V, E)` over the 64 squares with edges:

```text
- ray edges (rook/bishop/queen lines) directed along attacker direction
- knight edges
- pawn-attack edges
- defender-to-defended edges
```

Edge weights `w_e in [0, 1]` are learned by a small MLP from board features (occupancy, piece type, side-to-move).

### Single-path generating function

The path-generating function from `s` to `t`:

```text
g(s, t) = sum over paths p from s to t   prod_{e in p} w_e
        = (I + W + W^2 + ... )_{s, t}     where W is the weighted adjacency
        = ((I - W)^{-1} - I)_{s, t}       if rho(W) < 1
```

Computed differentiably via Neumann series (truncated to depth `K = 6`) or via a single `solve(I - W, ...)`. We spectral-normalize `W` so `rho(W) <= 0.9`.

### LGV matrix

Choose attacker sources via a soft top-`k` mask (top-k pieces by aggregate-attacker-score, `k = 6`); choose target squares similarly (top-k by aggregate-target-score). Build:

```text
M[i, j] = g(s_i, t_j)        in R^{k x k}
```

### Readout

```text
det(M)                                 LGV determinant: signed non-intersecting paths
log|det(M)|, sign(det(M))
permanent(M) ~~ Ryser approximation    unsigned count of all (intersecting OR not) tuples
det/perm ratio                         non-intersection density
top-k singular values of M
trace(M), trace(M^2)                  walk counts
attacker_target_dominance = max(M, axis=cols)   per-attacker best target
```

Final:

```text
puzzle_logit = MLP([log|det(M)|, sign(det(M)), det/perm, sigma_topk, dominance, board_pool])
```

## Assumptions

- Chess tactics correspond to **non-intersecting path families** in the static-geometry DAG, where attackers route to distinct targets without sharing squares.
- The k = 6 attacker / target selection captures the relevant motif size for most puzzles.
- Neumann depth `K = 6` is sufficient to enumerate the chess paths that matter.

## Claim / Hypothesis

`det(M)` is non-zero precisely when there exists a non-intersecting attacker-to-target assignment (Hall-style condition) with chess weights. Central falsifier:

```text
permanent_swap: replace det(M) by perm(M) (Ryser's formula; differentiable approx).
                if PR AUC doesn't drop, the *signed* non-intersection structure isn't
                the signal (and LGV reduces to a generic bilinear path counter).

random_path_W: randomize edge weights but preserve marginals.
                if PR AUC doesn't drop, learned edge weights aren't carrying signal.
```

## Architecture

```text
board_encoder
edge_weight_W              -> W in R^{64 x 64}, spectral-normalized
neumann_solve              -> g(s, t) = ((I - W)^{-1} - I)
soft_topk_attackers        -> 6 source indices
soft_topk_targets          -> 6 target indices
LGV_matrix_assemble        -> M in R^{6 x 6}
det_perm_block             -> det(M), perm(M)
puzzle_head
```

### First config

```yaml
model:
  name: lindstrom_gessel_viennot_path_network
  input_channels: 18
  num_classes: 1
  channels: 64
  hidden_dim: 96
  num_attackers_k: 6
  num_targets_k: 6
  neumann_depth: 6
  spectral_clip_W: 0.9
training:
  mode: puzzle_binary
  loss: bce_with_logits
```

## Numerical / Compute Notes

- Neumann series at depth 6 on `64 x 64` W: `6 * 64^3 = 1.6e6` flops. Fine.
- Alternative: one `torch.linalg.solve(I - W, e_s)` per source = 6 solves of `64 x 64` systems. Cheaper if batched.
- `det(M)` and `perm(M)` on `6 x 6`: `det` is `O(k^3) = 216`; `perm` via Ryser is `O(k 2^k) = 384`. Both differentiable.
- Soft top-k via Gumbel-softmax with straight-through to top-k indices.

## Required Ablations

| Ablation | Removes | Hypothesis |
|---|---|---|
| `permanent_swap` | use perm(M) instead of det(M) | tests signed structure |
| `random_path_W` | random edge weights, marginal-preserving | tests learned weights |
| `random_attacker_topk` | random 6 attacker squares | tests attacker selection |
| `k_eq_2` | reduce to 2 attackers, 2 targets | tests motif size |
| `truncate_K_eq_1` | only direct edges, no path composition | tests path depth |
| `cnn_same_params` | matched CNN | baseline |
| `i058_DPP_baseline` | adjacent baseline | baseline |
| `i226_pfaffian_baseline` | adjacent baseline | baseline |

## Benchmark Targets

```text
PR AUC >= 0.82, F1 >= 0.76, near-puzzle FPR <= 0.20, puzzle recall >= 0.78
permanent_swap drops PR AUC >= 0.015 (signed structure matters)
random_path_W drops PR AUC >= 0.01
beats i058 by >= 0.005 PR AUC
```

## Counterexamples

- Tactics are not really non-intersecting; attackers share squares all the time.
- `(I - W)^{-1}` blows up if W has rho > 1; spectral clip is too aggressive and kills signal.
- Soft top-k is too noisy.
- Most puzzles are `k = 1` (single attacker) so LGV reduces to a single edge weight.

## Implementation Priority

1. Build chess DAG and adjacency W with learned edge weights.
2. Implement Neumann series `g(s, t)`.
3. Soft top-k attacker / target selection.
4. Train minimal head with `(log|det(M)|, sign(det(M)))`.
5. Run 8 ablations.
