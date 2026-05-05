# Codex Research Packet: p-adic Ultrametric Threat Embedding Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1530_tuesday_local_padic_ultrametric_threat.md`
- Generated at: 2026-05-05 15:30
- Weekday: Tuesday
- Timezone: local
- Author: Claude (Opus 4.7, 1M context)
- Intended next consumer: Codex
- Status: full unorthodox-linear-algebra architecture packet, not implemented, not benchmark results

## One-Sentence Thesis

Embed each board into a finite truncation of the **p-adic numbers** `Z / p^k Z` (with
`p` a small prime, `k` a small depth) such that *closer in p-adic distance* means
*sharing a deeper common ancestor in a tree-shaped threat hierarchy*; classify
puzzle-likeness from p-adic ultrametric distances and the resulting **Bruhat-Tits
tree** structure -- a genuinely non-Archimedean linear-algebraic structure that no
imported packet uses.

## Why This Is A Real And Unorthodox Linear Algebra NN Idea

The p-adic absolute value satisfies the **strong (ultrametric) triangle inequality**

```text
|x + y|_p <= max(|x|_p, |y|_p)
```

so all triangles in `(Q_p, |.|_p)` are isoceles with the two longer sides equal. A
finite p-adic ultrametric space is therefore not a Euclidean space; it is a *tree*.
Its automorphism group is *not* an orthogonal group; it is the group of locally
constant trace-preserving maps -- in the linear-algebra setting, matrices over `Z_p`.

For chess, this matters because tactics are tree-structured:

- King-zone defenders cluster around the king like leaves on a tree.
- Pinned pieces, blocker chains, and discovered-attack rays form *nested* structures
  whose natural distance is "depth at which they diverge", not Euclidean displacement.
- The Bruhat-Tits tree of `PGL_2(Q_p)` is the canonical linear-algebraic object for
  hierarchical clustering.

Concretely, a p-adic encoding gives each square a value in `Z / p^k Z` such that two
squares share a common prefix iff they are *threat-related* up to depth `prefix_len`.
The ultrametric distance becomes `p^{-prefix_len}` -- exponentially decaying.

This is fundamentally distinct from:

- **Hyperbolic embeddings** (Poincaré ball / Lorentzian): hyperbolic space is a
  *continuous* tree-like geometry, but its triangles satisfy `<=` not the *strong*
  ultrametric inequality.
- **Tree kernels / dendrograms**: these give *combinatorial* trees, not algebraic
  structures with addition / multiplication.
- **Wavelet scattering** (i093): scaling on `R`, not p-adic.

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

- `i093 Wavelet Scattering` — uses real-valued wavelets, dyadic scales, *Archimedean*.
- `i168 Ring-Shell Recurrent` — uses radial bands, but Euclidean.
- `i085 Hall-Defect Zeta Operator` — uses zeta functions of incidence algebra, but
  these are *complex* not p-adic.
- `i057 Soft Formal-Concept Closure` — gives a lattice, but no p-adic embedding.

### Exact difference

```text
p-adic numbers form a non-Archimedean local field with a strong ultrametric. Their
linear-algebraic content (matrices over Z_p, p-adic norms of eigenvalues, p-adic
condition numbers, Newton polygon of characteristic polynomial) is entirely distinct
from any real / complex / tropical / spectral packet. The natural threat object is the
Bruhat-Tits tree, not the Hodge Laplacian or the simplex of squares.
```

## Mathematical Thesis

### Definitions

Choose `p = 3` (so we get a 3-ary tree, matching the natural ternary chess split
attacker / defender / blocker) and depth `k = 4` (depth-4 tree, 81 leaves >= 64 squares).

A learned **p-adic encoder** maps each square `s in {0, ..., 63}` to a depth-`k`
sequence of digits `phi(s) in {0, 1, ..., p-1}^k`, interpreted as `phi(s) = sum_i
phi_i(s) p^i in Z / p^k Z`.

Constraint: the encoder is trained so that ultrametric distance

```text
d_p(s, t) = p^{-min{i : phi_i(s) != phi_i(t)}}    (= 0 if phi(s) = phi(t))
```

approximates a chess-natural threat-tree distance derived from the current board (e.g.
"shortest sequence of attacker-relations needed to chain s to t").

### p-adic threat operator

Build a `64 x 64` matrix `K_p` whose entries `K_p[s, t] in Z / p^k Z` are p-adic-encoded
relation classes (e.g. attacker-defender, x-ray, blocker, king-zone). Compute:

```text
M_p = encoder_M(K_p)              real R^{64 x 64}, but its entries are p-adic-aware
                                  via a learned table {0,1,...,p-1}^k -> R
```

The trick: instead of plain real values, the matrix entries pass through a *p-adic
absorption layer* that respects the ultrametric. One concrete realization:

```text
M_p[s, t] = sum_{i=0}^{k-1}  alpha_i * f_i(phi_i(K_p[s, t]))
```

with `alpha_i = p^{-i}` and `f_i: Z/p Z -> R` learned. This makes `M_p` a *valuation-
weighted* sum, where deeper (smaller `p^{-i}`) digits dominate -- the natural p-adic
absolute-value norm.

### Newton-polygon / p-adic spectrum readout

Compute the eigenvalues of `M_p` (real). For each eigenvalue `lambda`, also report:

```text
v_p(lambda)         p-adic valuation, computed on lambda's nearest p-adic integer
                    representation
nearest_phi(lambda) closest depth-k digit sequence
```

Plus the **Newton polygon** of `det(xI - M_p)`: lower convex hull of `(i, v_p(c_i))`
where `c_i` is the `i`-th characteristic-polynomial coefficient. Slopes of the Newton
polygon = the p-adic valuations of the eigenvalues. Used heavily in number theory; here
it gives a hierarchy-of-magnitudes summary.

### Readout

```text
phi_squares                      learned R^{k x p} per-square soft histogram
ultrametric_distance_matrix       64 x 64
spectral_eigvals(M_p) and Newton-polygon slopes
p_adic_norm(M_p)  = max_{i,j} p^{-min_digit_position}
tree_depth_histogram             how many edges are at each depth
```

Final:

```text
puzzle_logit = MLP([phi_pool, slope_features, depth_histogram, board_pool])
```

## Assumptions

- Tactical structure is hierarchical / tree-shaped (attacker-defender-blocker chains
  naturally cluster).
- An ultrametric (rather than Euclidean) embedding captures this hierarchy with fewer
  parameters.
- p = 3, depth k = 4 is enough resolution: 3^4 = 81 distinguishable embeddings.

## Claim / Hypothesis

`d_p` decorrelates puzzle vs near-puzzle better than the matched-rank Euclidean
embedding, *especially* on positions where the threat structure is deeply nested
(combinations, long discovered attacks). Central falsifier:

```text
euclidean_swap: replace ultrametric distance d_p(s,t) = p^{-min_diff} with the
                Hamming-style L2 distance ||phi(s) - phi(t)||_2 over the same digit
                sequences.
                if PR AUC does not drop, the strong ultrametric does not matter.
```

## Architecture

### Components

```text
board_encoder
square_to_padic_phi  -> R^{64 x k x p}  (soft histogram per digit)
relation_to_padic    -> K_p[s,t] in {0,...,p^k - 1} (soft)
p_adic_M_builder     -> M_p in R^{64 x 64}
ultrametric_distance -> D in R^{64 x 64}
spectral_block       -> eigvals of M_p + Newton polygon slopes
tree_features        -> depth histogram of D
puzzle_head
```

### Forward pseudocode

```text
X_sq      = board_encoder(board)
phi_logits = square_to_padic_phi(X_sq)               # 64 x k x p
phi       = softmax(phi_logits, dim=-1)              # soft digits
K_p       = relation_to_padic(X_sq)                  # 64 x 64 soft p^k classes
M_p       = p_adic_M_builder(K_p, alpha=[1, 1/p, 1/p^2, 1/p^3])
D         = ultrametric_distance(phi)                # 64 x 64
eigvals   = torch.linalg.eigvals(M_p).real
slopes    = newton_polygon_slopes(char_poly(M_p), p) # k features
depth_h   = histogram(D, bins=[1, 1/p, 1/p^2, 1/p^3, 0])
feat      = [phi_pool, slopes, depth_h, ||M_p||, eigvals_topk, pool(X_sq)]
logit     = MLP(feat)
```

### First config

```yaml
model:
  name: padic_ultrametric_threat_network
  input_channels: 18
  num_classes: 2
  hidden_dim: 96
  prime_p: 3
  depth_k: 4
  num_relation_classes: 8
  use_newton_polygon: true
training:
  mode: puzzle_binary
  loss: bce_with_logits
  aux_loss_phi_diversity: 1.0e-3   # discourage all squares mapping to same digit
  batch_size: 512
  learning_rate: 1.0e-3
```

## Numerical / Compute Notes

- `p=3, k=4` -> 81 leaves, 4-level tree. Soft `phi` adds `64 * 4 * 3 = 768` parameters
  per board; cheap.
- Newton-polygon slopes from differentiable char-poly coefficients (Faddeev-LeVerrier
  iteration) at `n = 64`: `O(n^4) = 1.6e7` flops -- still cheap for batch=512.
- Soft `min_diff` (the depth at which two digit sequences diverge) is computed via
  `argmax_i ( phi_i(s) <> phi_i(t) )` softened by a temperature: 
  `prefix_match_i = prod_{j<=i} <phi_j(s), phi_j(t)>`,
  `expected_min_diff = sum_i (1 - prefix_match_i)`.
- Auxiliary `aux_loss_phi_diversity = - mean( entropy(phi_pooled_over_squares) )` to
  prevent trivial collapse.

## Required Ablations

| Ablation | Removes | Hypothesis |
|---|---|---|
| `euclidean_swap` | replace ultrametric `d_p` with `L2` over `phi` digits | tests strong ultrametric |
| `flat_alpha` | use `alpha_i = 1` for all i | tests p-adic valuation weighting |
| `random_phi` | freeze `phi` to random | tests learned hierarchy |
| `no_newton_polygon` | drop slopes | tests sufficiency |
| `p_eq_2` | switch to `p = 2`, `k = 6` | tests prime choice |
| `cnn_same_params` | matched CNN | baseline |
| `i168_ring_shell_baseline` | adjacent radial baseline | baseline |
| `random_relation_classes` | randomize K_p semantics | tests chess content |

## Benchmark Targets

```text
test PR AUC      >= 0.82
test F1          >= 0.76
near-puzzle FPR  <= 0.20
puzzle recall    >= 0.78

central claim:
  euclidean_swap drops PR AUC >= 0.01
  flat_alpha     drops PR AUC >= 0.005
  random_phi     drops PR AUC >= 0.015
  must beat i168 ring-shell by >= 0.005 PR AUC at matched params
```

## Counterexamples / Failure Modes

- Tactics are not tree-structured, so the ultrametric inductive bias is wrong.
- The soft argmin used for the digit-divergence depth is too noisy.
- The Newton polygon slopes are dominated by trivial structure of `M_p`.
- Choice of `p, k` is wrong; mitigation: `p_eq_2` ablation explores it.

## Implementation Priority

1. Build soft p-adic encoder `phi: square -> R^{k x p}` and ultrametric distance.
2. Validate that random initialization produces a non-degenerate `D` matrix.
3. Build `M_p` with `alpha_i = p^{-i}` weighting.
4. Add Newton-polygon slope features.
5. Run all 8 ablations.

Smallest viable version:

```text
p = 2, k = 4, only the ultrametric distance D pooled to a 4-bin histogram + ||D||_F as
features. No M_p, no Newton polygon.
```

If lift over CNN-same-params is positive, add `M_p` and Newton-polygon readout.
