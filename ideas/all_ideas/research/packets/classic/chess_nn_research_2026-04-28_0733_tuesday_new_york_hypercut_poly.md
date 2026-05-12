# Codex Handoff Packet: Chess Hypercut Polynomial Network

## 1. File Metadata

- **Filename:** `chess_nn_research_2026-04-28_0733_tuesday_new_york_hypercut_poly.md`
- **Created:** 2026-04-28 07:33 new_york
- **Idea name:** Chess Hypercut Polynomial Network, abbreviated `CHPNet`
- **Task:** Binary chess-puzzle classification from the current board only.
- **Target mapping:** fine labels `0` and `1` map to target `0`; fine label `2` maps to target `1`.
- **Hard exclusions:** no engine evaluations, principal variations, node counts, mate scores, best moves, source labels, verification metadata, tactical sheaf/Hodge networks, ordinary graph neural networks, static attack-defense edge walks, non-backtracking walks, Hall defects, obligation flow, or relation-label GNNs.
- **Selected operator:** masked high-order hypergraph cut polynomial with a derivative-based higher-order incidence residual. This is not message passing: each residual term is a joint product over all other vertices in the same hyperedge, not a sum of neighbor messages.

## 2. Executive Selection

Select `CHPNet`: a current-board hypergraph model that builds deterministic chess-rule hyperedges over the 64 board squares and applies a high-order cut polynomial over each hyperedge.

The core bet is simple: puzzle-positive positions often contain a small set of squares whose tactical status is not readable as independent pieces or as pairwise attack-defense edges. A mating net, discovered tactic, overloaded king zone, or promotion race can be represented as a hyperedge whose vertices cannot all share the same learned latent state. `CHPNet` learns soft square states and scores whether deterministic chess hyperedges are internally mixed under those states.

The chosen operator for one probe channel is:

\[
C_e(s) = 1 - \prod_{i \in e}\frac{1+s_i}{2} - \prod_{i \in e}\frac{1-s_i}{2}, \qquad s_i \in [-1,1].
\]

For binary `s_i`, this is `0` when every square in the hyperedge is assigned the same side of the latent cut, and `1` when the hyperedge is split. For hyperedges of size at least `3`, this is a degree-`|e|` polynomial and is not equivalent to a graph Laplacian quadratic. The network uses the derivative of this polynomial as a residual update on square features, which makes the hyperedge interaction higher-order without turning the model into a GNN.

## 3. Data Contract

### Input batch

```python
batch = {
    "board": FloatTensor[B, 18, 8, 8],
    "fine_label": LongTensor[B],  # values in {0, 1, 2}
}
```

### Board channels

Use only current-position state encoded as tensor planes:

| Channel range | Meaning |
|---:|---|
| `0..5` | White `P, N, B, R, Q, K` occupancy |
| `6..11` | Black `p, n, b, r, q, k` occupancy |
| `12` | Side-to-move constant plane: `1.0` for white, `0.0` for black |
| `13..16` | Current castling-right constant planes: `WK, WQ, BK, BQ` |
| `17` | En-passant target square plane, all zeros if no target |

No other scalar or metadata may enter the model. In particular, do not pass engine scores, search depth, node counts, mate distance, best move, puzzle source, puzzle rating, verification status, puzzle ID, or source label.

### Target

```python
y = (batch["fine_label"] == 2).float()  # FloatTensor[B]
```

### Deterministic preprocessing

- Normalize orientation to the side-to-move perspective before hyperedge construction and before the neural stem.
- If black is to move, rotate the board by 180 degrees and swap colors so the active side is always represented as white.
- This is deterministic from the current board tensor and does not use labels or engine information.
- Split train/validation/test by a deterministic hash of the normalized board tensor, not by puzzle source or metadata.

### Hyperedge tensors produced from the board tensor

```python
edge_index:  LongTensor[B, M_MAX, K_MAX]  # padded vertex ids in 0..63
edge_mask:   BoolTensor[B, M_MAX, K_MAX]  # true for valid vertex slots
edge_active: BoolTensor[B, M_MAX]         # true for valid hyperedges
edge_size:   FloatTensor[B, M_MAX]        # number of valid vertices
```

Recommended constants:

```python
M_MAX = 1024
K_MAX = 9
```

If a position yields more than `M_MAX` candidate hyperedges, keep them by this deterministic priority order: edges touching either king, edges with at least two occupied vertices, larger edge size, then lexicographic vertex tuple. Do not use learned selection and do not use labels.

## 4. Hypergraph Research Map

The selected design is intentionally outside the common hypergraph-neural-network pattern. The research map is:

1. **Classical hypergraph learning:** Zhou, Huang, and Schölkopf introduced hypergraph learning for clustering, classification, and embedding, arguing that squeezing multi-object relations into pairwise graphs loses information. That motivates hyperedges, but their normalized-cut relaxation is too close to incidence-matrix spectral learning for this task. Reference: `https://papers.nips.cc/paper/3128-learning-with-hypergraphs-clustering-classification-and-embedding`.
2. **Adjacency tensors for uniform hypergraphs:** Cooper and Dutle developed spectra of uniform hypergraphs through adjacency hypermatrices and characteristic-polynomial ideas. This supports a tensor-polynomial view rather than a message-passing view. Reference: `https://doi.org/10.1016/j.laa.2011.11.018`.
3. **Laplacian and signless Laplacian tensors:** Qi's work on Laplacian and signless Laplacian tensors gives a direct precedent for hypergraph operators that are tensors rather than graph matrices. Reference: `https://doi.org/10.4310/CMS.2014.v12.n6.a3`.
4. **Normalized Laplacian tensors and hypergraph partitioning:** Chen, Qi, and Zhang use a normalized Laplacian tensor and Z-eigenvectors for even-uniform hypergraph partitioning, reinforcing that tensor Laplacians are not just ordinary graph Laplacians. Reference: `https://doi.org/10.1137/16M1094828`.
5. **Hypergraph Cheeger cuts:** Mulas generalizes Cheeger cuts and inequalities to uniform hypergraphs, giving cut-based motivation for scoring whether a hyperedge is internally split. Reference: `https://doi.org/10.1007/s00373-021-02348-z`.
6. **Generalized hypergraph p-Laplacians:** Saito and Herbster, and later Fazeny et al., survey and extend nonlinear hypergraph p-Laplacian operators. These works motivate nonlinear hyperedge energies, but `CHPNet` uses a finite masked cut polynomial rather than heat flow, diffusion, or message passing. References: `https://doi.org/10.1007/s10994-022-06264-y` and `https://doi.org/10.1007/s10851-024-01183-0`.
7. **Hyperdeterminant background:** Gelfand, Kapranov, and Zelevinsky's multidimensional determinant theory is a conceptual reason to treat high-order arrays as first-class objects. `CHPNet` does not compute an exact hyperdeterminant; it uses a stable product-polynomial surrogate suitable for PyTorch. Reference: `https://link.springer.com/book/10.1007/978-0-8176-4771-1`.

The resulting architecture keeps the hypergraph object and the operator, but drops incidence convolution, relation labels, and pairwise edge walks.

## 5. Candidate Search Trace

| Candidate | Operator | Decision | Reason |
|---|---|---|---|
| Incidence-matrix hypergraph convolution | `V -> E -> V` averaging | Rejected | This is message passing and too close to ordinary GNN practice. |
| Clique-expanded graph baseline | Replace each hyperedge by pairwise edges | Rejected | Violates the requirement for a genuinely hypergraph operator. |
| Tensor Laplacian spectral-moment classifier | Compute low-order spectral moments of a uniformized hypergraph tensor | Rejected as primary | Interesting but brittle, expensive, and hard to train end to end on variable chess hyperedges. |
| Exact hyperdeterminant layer | Discriminant or resultant-like invariant over edge tensors | Rejected | Mathematically aligned but computationally impractical and numerically unstable for board-scale batches. |
| Elementary symmetric polynomial residual | Degree-3 or degree-4 symmetric products over each hyperedge | Runner-up | Good high-order behavior, but less directly interpretable as a cut. |
| Masked hypercut polynomial residual | `1 - all_plus - all_minus` over each hyperedge | Selected | Direct cut interpretation, stable implementation, variable-size support, and clearly non-quadratic for `|e| >= 3`. |

Final selection: `CHPNet` with masked hypercut polynomial residuals and global hypercut moment readout.

## 6. Rejected Approaches

Reject these explicitly during implementation:

- **Tactical sheaf or Hodge networks:** no cochains, coboundaries, cellular sheaves, harmonic projections, or Hodge Laplacians.
- **Ordinary GNNs:** no graph convolution, graph attention, message passing neural network, GraphSAGE, GCN, GAT, edge-conditioned convolution, or relation-label GNN.
- **Static attack-defense edge walks:** no prebuilt attack graph whose paths are walked, pooled, or counted.
- **Non-backtracking walks:** no Hashimoto matrix, non-backtracking path expansion, or walk recurrence.
- **Hall-defect modules:** no matching-deficiency, Hall-set, or bipartite-cover features.
- **Obligation flow:** no flow network over forced moves, king obligations, mate obligations, or defensive obligations.
- **Engine-derived supervision:** no engine score, principal variation, best move, mate distance, node count, tablebase result, search depth, or verification metadata.
- **Source-derived shortcuts:** no puzzle source, puzzle rating, source label, verification tag, import date, or collection identity.

Also reject any ablation that sneaks relation type embeddings into the selected model. Edge categories may be used only inside deterministic construction code; they are not model inputs.

## 7. Mathematical Thesis

Let the board squares be vertices `V = {0, ..., 63}`. For a current board `x`, deterministic construction yields a hypergraph:

\[
\mathcal{H}(x) = (V, \mathcal{E}(x)), \qquad e \subseteq V, \ |e| \in \{3,\dots,9\}.
\]

The network learns `R` soft latent cut probes:

\[
s_{i,r} = \tanh(h_i^\top w_r + b_r), \qquad s_{i,r} \in [-1,1].
\]

For each hyperedge `e` and probe `r`, define:

\[
A_{e,r} = \prod_{i \in e}\frac{1+s_{i,r}}{2}, \qquad
B_{e,r} = \prod_{i \in e}\frac{1-s_{i,r}}{2},
\]

\[
C_{e,r} = 1 - A_{e,r} - B_{e,r}.
\]

`C_{e,r}` is a soft hyperedge cut. It is low when all vertices in `e` are assigned the same latent sign and high when the edge is split. The derivative with respect to a valid vertex slot is:

\[
\frac{\partial C_{e,r}}{\partial s_{i,r}} =
-\frac{1}{2}\prod_{j \in e \setminus \{i\}}\frac{1+s_{j,r}}{2}
+\frac{1}{2}\prod_{j \in e \setminus \{i\}}\frac{1-s_{j,r}}{2}.
\]

This derivative is the higher-order incidence residual. It depends on a product over all other vertices in the hyperedge. For `|e| = 3`, the residual for vertex `i` already contains a product of two other vertices. For larger edges, it is a higher-degree interaction. Therefore it cannot be reduced to a graph Laplacian update of the form `sum_j a_ij h_j`.

The classifier hypothesis is:

\[
\Pr(y=1 \mid x) = \sigma(f_\theta(x, \mathcal{H}(x))),
\]

where `f_theta` is built from board features, hypercut residual blocks, and pooled hypercut moments. The model is falsified if the high-order cut residual does not beat a current-board CNN with the same input contract.

## 8. Hypergraph Object

### Vertices

Each square is one vertex. Use row-major indexing after side-to-move normalization:

```python
vertex_id = 8 * rank + file  # rank, file in 0..7
```

### Hyperedges

Construct only size-`3` to size-`9` hyperedges. No size-2 edges are used in the selected model.

#### A. Sliding ray hyperedges

For every occupied bishop, rook, or queen square and for every legal movement direction for that piece type:

1. Start with `[source]`.
2. Append every square along that ray until board edge or first occupied blocker.
3. Include the blocker square if present.
4. Keep the hyperedge if size is at least `3`.

This captures line structure as a set, not as a walk, and does not mark attack or defense relations.

#### B. Leaper and pawn stencil hyperedges

For every occupied knight, king, or pawn:

- Knight: `[source] + all in-board knight landing squares`.
- King: `[source] + all in-board Chebyshev-distance-1 squares`.
- Pawn: `[source] + in-board forward, double-forward-from-start, and diagonal pawn stencil squares from the normalized active-side perspective`; occupancy is left to vertex features.

Keep only size at least `3`. These are current-board rule stencils, not attack-defense edge walks.

#### C. Occupancy-filtered line-window hyperedges

For every rank, file, diagonal, and anti-diagonal line, enumerate contiguous windows of length `3..8`. Keep a window if it contains at least two occupied squares in the current board. This gives collinearity and blockage structure without using pairwise attack edges.

#### D. King-shell hyperedges

For each king currently on the board:

- Include the king square plus all in-board Chebyshev-distance-1 squares.
- If size exceeds `K_MAX`, it will not exceed `9`; keep directly.
- Keep both kings' shells after normalization.

#### E. Deterministic de-duplication and padding

- Convert each hyperedge to a sorted tuple of vertex ids.
- Deduplicate identical tuples.
- Sort by deterministic priority: touches a king, occupied count, larger edge size, lexicographic tuple.
- Truncate to `M_MAX = 1024`.
- Pad missing edges with `edge_active = False`.

The model receives only `edge_index`, `edge_mask`, `edge_active`, and `edge_size`; it receives no edge category labels.

## 9. Architecture Tensor Shapes

Recommended default configuration:

```python
C_IN = 18
D = 128
R = 32        # cut probes
L = 4         # hypercut blocks
M_MAX = 1024
K_MAX = 9
D_FF = 256
D_HEAD = 256
```

### Stem

```python
X:        FloatTensor[B, 18, 8, 8]
Conv3x3: 18 -> 64, padding=1, GELU
Conv1x1: 64 -> 128
Flatten: H0 = FloatTensor[B, 64, 128]
```

Add a learned square-position embedding:

```python
pos_embed: Parameter[64, 128]
H0 = H0 + pos_embed[None, :, :]
```

The position embedding is legal because it is board geometry, not puzzle metadata.

### Hypercut block `l = 1..L`

Input:

```python
H:          FloatTensor[B, 64, D]
edge_index: LongTensor[B, M_MAX, K_MAX]
edge_mask:  BoolTensor[B, M_MAX, K_MAX]
```

Probe projection:

```python
S = tanh(H @ W_s + b_s)       # [B, 64, R]
S_e = gather(S, edge_index)   # [B, M_MAX, K_MAX, R]
```

Masked product terms:

```python
a = where(edge_mask[..., None], 0.5 * (1.0 + S_e), 1.0)
b = where(edge_mask[..., None], 0.5 * (1.0 - S_e), 1.0)
A = prod(a, dim=2)            # [B, M_MAX, R]
Bminus = prod(b, dim=2)       # [B, M_MAX, R]
C = 1.0 - A - Bminus          # [B, M_MAX, R]
C = C * edge_active[..., None]
```

Exclusive products for derivative residual:

```python
A_excl = exclusive_prod(a, dim=2)       # [B, M_MAX, K_MAX, R]
B_excl = exclusive_prod(b, dim=2)       # [B, M_MAX, K_MAX, R]
g = (-0.5 * A_excl + 0.5 * B_excl) * edge_mask[..., None]
```

Map probe residuals to feature residuals:

```python
E_delta = einsum("bmkr,rd->bmkd", g, W_o)   # [B, M_MAX, K_MAX, D]
Delta = scatter_add_vertices(E_delta, edge_index, edge_mask, n_vertices=64)  # [B, 64, D]
Delta = Delta / sqrt(1.0 + incidence_count[..., None])
```

Residual block:

```python
H = LayerNorm(H + Dropout(Delta))
H = LayerNorm(H + Dropout(MLP(H)))
```

Per-block hypercut moments:

```python
cut_mean = masked_mean(C, edge_active, dim=1)  # [B, R]
cut_max  = masked_max(C, edge_active, dim=1)   # [B, R]
cut_std  = masked_std(C, edge_active, dim=1)   # [B, R]
block_summary = concat([cut_mean, cut_max, cut_std])  # [B, 3R]
```

### Readout

After `L = 4` blocks:

```python
vertex_mean = mean(H, dim=1)          # [B, D]
vertex_max  = max(H, dim=1).values    # [B, D]
cut_summary = concat(block_summaries) # [B, L * 3R] = [B, 384]
readout = concat([vertex_mean, vertex_max, cut_summary])  # [B, 640]
logit = MLP(readout, 640 -> 256 -> 1).squeeze(-1)          # [B]
```

Output exactly one puzzle logit per input board.

## 10. Training Objective

### Loss

Use binary cross-entropy with logits:

```python
target = (fine_label == 2).float()
loss = BCEWithLogitsLoss(pos_weight=pos_weight)(logit, target)
```

Set:

```python
pos_weight = clamp(num_negative / max(num_positive, 1), min=1.0, max=20.0)
```

Compute `pos_weight` on the training split only.

### Optimizer

Recommended defaults:

```python
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=num_train_steps)
grad_clip_norm = 1.0
batch_size = as large as GPU permits
precision = bf16 if stable, otherwise fp32
```

### Metrics

Track on validation and test:

- ROC-AUC.
- PR-AUC.
- BCE loss.
- Brier score.
- Expected calibration error with 15 bins.
- Positive-class recall at fixed false-positive rates: `1%`, `5%`, and `10%`.

Select checkpoints by validation PR-AUC, then break ties by validation BCE.

### Leakage controls

- Hyperedges must be constructed from the current-board tensor only.
- Hash splits must use normalized board tensor bytes, not source labels or puzzle IDs.
- No target-dependent sampling except class-balanced minibatch sampling based on `target`.
- Do not use fine label as a feature. Use it only to compute the binary target.

## 11. Ablations

Run these ablations with the same train/validation/test split and five random seeds:

| Ablation | Change | Interpretation |
|---|---|---|
| CNN-only | Remove all hypercut blocks and cut summaries; keep stem and readout parameter budget similar | Tests whether the hypergraph contributes beyond current-board convolution. |
| No residual, moments only | Keep `C` summaries but do not scatter derivative residuals into vertices | Tests whether hypercut features alone are enough. |
| Residual only, no moments | Use derivative residual blocks but remove cut summaries from readout | Tests whether the benefit is representational or just pooled statistics. |
| Pairwise quadratic replacement | Replace `C_e` with variance-style quadratic disagreement over each hyperedge | If this matches `CHPNet`, the high-order polynomial thesis is weak. |
| Random same-size hyperedges | Preserve each position's edge-size histogram but randomize vertex membership deterministically | Tests whether chess hyperedge construction matters. |
| Line windows only | Use only occupancy-filtered line-window hyperedges | Measures collinearity contribution. |
| Piece stencils only | Use only sliding ray, leaper, pawn, and king stencils | Measures current piece-rule stencil contribution. |
| No side normalization | Train without rotating/swapping to side-to-move perspective | Tests orientation normalization. |
| Degree-3 elementary symmetric residual | Replace cut polynomial with `e3` probe products | Tests whether any high-order polynomial suffices or cut semantics specifically help. |
| Probe-count sweep | `R in {8, 16, 32, 64}` | Measures capacity and stability. |

Do not include a GNN baseline as part of the selected architecture. If a team wants a separate external comparison, keep it outside this packet and do not merge it into `CHPNet`.

## 12. Falsification Rule

Declare the idea falsified if all of the following are true on the locked test split after selecting by validation PR-AUC:

1. Across five seeds, `CHPNet` fails to improve over the CNN-only baseline by at least `+0.015` PR-AUC and `+0.010` ROC-AUC.
2. The pairwise quadratic replacement is statistically indistinguishable from `CHPNet` under bootstrap 95% confidence intervals for PR-AUC.
3. Random same-size hyperedges are statistically indistinguishable from deterministic chess hyperedges.
4. Calibration does not improve: Brier score and ECE are both equal or worse than the CNN-only baseline.

If these conditions hold, do not rescue the model by adding engine features, best-move labels, graph walks, relation labels, or verification metadata. The correct conclusion would be that this high-order hypercut construction does not carry enough signal for the dataset under the current-board-only constraint.

## 13. Codex Implementation Notes

### Suggested modules

```text
models/chpnet.py
    class ChessHypercutPolynomialNet(nn.Module)
    class HypercutBlock(nn.Module)
    exclusive_prod(...)
    gather_vertices(...)
    scatter_add_vertices(...)

data/hyperedges.py
    build_hyperedges_from_board_tensor(...)
    normalize_side_to_move(...)
    square_index(...)
    enumerate_sliding_rays(...)
    enumerate_piece_stencils(...)
    enumerate_line_windows(...)
    enumerate_king_shells(...)

tests/test_chpnet_shapes.py
    test_forward_shape()
    test_hyperedges_are_deterministic()
    test_no_size_two_edges()
    test_masked_products_ignore_padding()
    test_black_to_move_normalization_is_involutive()
```

These are suggested repository paths only; this handoff creates no code files.

### Exclusive product implementation

Avoid division by `a` or `b`, because values can be near zero. Use prefix and suffix products:

```python
def exclusive_prod(x: torch.Tensor, dim: int) -> torch.Tensor:
    # x shape includes K on `dim`; padded slots should already be neutral 1.0.
    x = x.transpose(dim, -1)
    prefix = torch.cumprod(x, dim=-1)
    suffix = torch.cumprod(torch.flip(x, dims=[-1]), dim=-1)
    suffix = torch.flip(suffix, dims=[-1])

    ones = torch.ones_like(x[..., :1])
    left = torch.cat([ones, prefix[..., :-1]], dim=-1)
    right = torch.cat([suffix[..., 1:], ones], dim=-1)
    out = left * right
    return out.transpose(dim, -1)
```

### Gather and scatter rules

- `edge_index` padded values may be zero, but `edge_mask` must zero out their residuals before scatter.
- `scatter_add_vertices` should scatter into a `[B, 64, D]` tensor.
- Normalize by `sqrt(1 + incidence_count)` to prevent dense line-window positions from dominating.
- Keep all hyperedges size at least `3`; filter size `0`, `1`, and `2`.

### Numerical stability

- Clamp `S = tanh(projection)` naturally keeps probe values in `[-1, 1]`.
- Products of up to `9` factors are acceptable in fp32; if using bf16, compute the product section in fp32 and cast residuals back.
- Dropout should be applied after the scatter residual, not inside the product.
- Initialize `W_o` with small gain, for example `std = 0.02`, so early residuals do not dominate the CNN stem.

### Hyperedge construction details

- Build from the normalized board tensor, not from FEN metadata.
- The code may reconstruct piece locations from occupancy planes by `argwhere`.
- Castling and en-passant planes remain available to the stem, but hyperedge construction does not need to use them except for optional pawn stencil inclusion of the en-passant target square.
- Deduplicate by sorted vertex tuple.
- Do not pass construction category IDs to the model.
- Use deterministic sorting and truncation so repeated calls produce identical tensors.

### Minimal forward signature

```python
class ChessHypercutPolynomialNet(nn.Module):
    def forward(
        self,
        board: torch.Tensor,       # [B, 18, 8, 8]
        edge_index: torch.Tensor,  # [B, 1024, 9]
        edge_mask: torch.Tensor,   # [B, 1024, 9]
        edge_active: torch.Tensor, # [B, 1024]
    ) -> torch.Tensor:             # [B]
        ...
```

The dataloader should construct `edge_index`, `edge_mask`, and `edge_active` on CPU and pin them for transfer. The model should not call an engine, puzzle verifier, or move search.

## 14. Prompt Updates

Use this implementation prompt for Codex:

```text
Implement ChessHypercutPolynomialNet (CHPNet) for binary chess-puzzle classification.

Inputs:
- board FloatTensor[B,18,8,8] containing only current-position planes.
- deterministic hyperedge tensors built from that board: edge_index[B,1024,9], edge_mask[B,1024,9], edge_active[B,1024].
- fine_label is used only outside the model to define target = 1[fine_label == 2].

Do not use engine evaluations, PVs, node counts, mate scores, best moves, puzzle source labels, verification metadata, graph walks, attack-defense walk features, non-backtracking walks, Hall defects, obligation flow, sheaf/Hodge operators, ordinary GNNs, or relation-label GNNs.

Build hyperedges from current-board tensor only:
1. Normalize to side-to-move perspective.
2. Use 64 square vertices.
3. Add size-3..9 sliding ray, leaper/pawn stencil, occupancy-filtered line-window, and king-shell hyperedges.
4. Deduplicate sorted vertex tuples.
5. Sort deterministically and pad/truncate to M_MAX=1024.
6. Do not feed edge category IDs to the model.

Architecture:
- Stem: Conv3x3 18->64, GELU, Conv1x1 64->128, flatten to [B,64,128], add learned square embedding.
- Four HypercutBlock layers with D=128 and R=32 probes.
- Each block computes S=tanh(HWs+bs), gathers S over hyperedges, computes C=1-prod((1+S)/2)-prod((1-S)/2), computes exclusive-product derivatives, scatters derivative residuals back to vertices, then applies LayerNorm residual and MLP residual.
- Readout concatenates mean pooled vertices, max pooled vertices, and per-block cut mean/max/std summaries.
- Output one logit [B].

Training:
- target = (fine_label == 2).float().
- BCEWithLogitsLoss with train-only pos_weight clipped to [1,20].
- Evaluate ROC-AUC, PR-AUC, BCE, Brier, ECE, and recall at fixed FPR.

Tests:
- forward returns shape [B].
- hyperedge builder is deterministic.
- no selected hyperedge has size < 3.
- edge padding does not alter masked products.
- no edge category, source label, or forbidden metadata is consumed by the model.
```
