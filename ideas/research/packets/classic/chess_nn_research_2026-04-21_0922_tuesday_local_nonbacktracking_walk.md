# Codex Handoff Packet: Non-Backtracking Tactical Walk Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0922_tuesday_local_nonbacktracking_walk.md`
- Generated at: 2026-04-21 09:22:45 America/Los_Angeles
- Weekday: Tuesday
- Timezone: local
- Idea slug: `nonbacktracking_walk`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Non-Backtracking Tactical Walk Network
- One-sentence thesis: Puzzle-like positions are disproportionately marked by short, directed chains of current-board tactical dependency in which attack/protection pressure propagates without immediately undoing itself, and a Hashimoto-style non-backtracking edge-walk operator should expose that signal while suppressing trivial reciprocal-attack and degree/material shortcuts.
- Idea fingerprint: `current-board pseudo-legal attack/protection edge graph -> directed-edge Hashimoto non-backtracking walk operator -> typed edge message moments + small board adapter -> binary puzzle-like logits; no move deltas, no search, no engine, no sheaf/Laplacian/tension/OT/topology`
- Why this is not a common CNN/ResNet/Transformer variant: the central representation lives on directed attack/protection edges and their non-backtracking transitions, not on square pixels, residual convolutions, or unconstrained token attention; the smallest falsifier specifically randomizes the edge-to-edge transition semantics while preserving edge counts, degree marginals, material, side-to-move, source-square marginals, and capture/protection histograms.
- Current-data minimal experiment: train on `data/splits/crtk_sample_3class/split_train.parquet`, validate on `split_val.parquet`, test on `split_test.parquet`, using `simple_18` only for the first run; report binary metrics and the required `3x2` fine-label diagnostic for the main model and all central ablations.
- Smallest central falsification ablation: replace the true non-backtracking transition relation `e=(u->v) -> f=(v->w), w!=u` with a degree- and type-preserving randomized transition relation that keeps each edge token, per-edge in/out transition counts, relation-type-pair counts, source-piece/target-piece marginals, material, side-to-move, and edge-count histograms fixed.
- Expected information gain if it fails: a clean failure would rule out short static non-backtracking attack/protection chains as a useful inductive bias for the current split, and the next cycle should avoid Hashimoto spectra, damped edge-walk resolvents, and edge-line-graph tactical propagation unless new evidence or a different data regime appears.

## 3. Problem Restatement And Data Contract

The project is `chess-nn-playground`. The task is board-position chess puzzle-likeness classification.

Labels:

- Fine label `0`: known non-puzzle.
- Fine label `1`: verified near-puzzle.
- Fine label `2`: verified puzzle.
- Coarse binary target for the default benchmark: `0 -> 0`, `1 -> 1`, `2 -> 1`.

Allowed model input:

- A tensor `x` of shape `(batch, C, 8, 8)`.
- Deterministic features derived only from the current board represented by `x`.
- Safe current-board rule geometry: piece locations, side to move, castling/en-passant channels already present in the encoding, and pseudo-legal attack/protection rays computed from current occupancy.

Forbidden model input:

- Stockfish scores, principal variations, node counts, mate scores, search depth, verification metadata, source labels, proposed labels, unresolved-candidate status, dataset provenance, or any label-derived feature.
- Full game history not already present in the chosen encoding.
- Any target-dependent preprocessing.

Benchmark split:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

Model contract:

```text
input:  (batch, C, 8, 8)
output: logits (batch, 2)
```

The shared trainer, report writer, leaderboard, binary confusion matrix, and rectangular diagnostic matrix must keep working.

Leakage checklist:

- No engine features.
- No legal move tree.
- No one-ply move-delta set or consequence bag.
- No checkmate/stalemate oracle.
- No source/provenance field.
- No use of fine label as input.
- No fabricated class `1` or class `2`.
- No full-dataset training path; use the existing split parquet files only.

Boundary between safe rule-derived features and leakage:

- Deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack geometry derived only from the current board are allowed.
- Full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences are leakage-prone unless explicitly justified as rule-only, label-independent, engine-free, and ablated. This packet avoids them entirely.
- Engine evaluation, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, and dataset provenance must never be neural-network inputs.
- For `lc0_static_112` and `lc0_bt4_112`, current-board channels may be used for deterministic geometry only if their channel semantics are explicitly configured and validated. History channels, including zero-filled BT4 history channels, may be consumed only by a learned neural adapter, never by the deterministic attack-graph builder.

## 4. Research Map

External research anchors used:

| Source | URL | What is borrowed | What is not copied |
|---|---|---|---|
| Hashimoto/Ihara non-backtracking edge operator and Ihara zeta literature | https://www.combinatorics.org/ojs/index.php/eljc/article/view/v25i2p26/7406 and https://en.wikipedia.org/wiki/Ihara_zeta_function | The idea that directed edges can be the state space, and that non-backtracking walks are captured by an edge-adjacency operator. | No zeta-function determinant objective, no graph-isomorphism task, no theorem claim about chess labels. |
| Krzakala et al., “Spectral redemption: clustering sparse networks” | https://arxiv.org/abs/1306.5550 and https://www.pnas.org/doi/10.1073/pnas.1312486110 | The empirical/theoretical lesson that non-backtracking spectra can separate structure from noisy degree effects better than ordinary adjacency spectra in sparse graphs. | No community detection, no stochastic block model assumption for chess, no spectral clustering head. |
| Park et al., “Non-backtracking Graph Neural Networks” | https://arxiv.org/abs/2310.07430 and https://openreview.net/pdf?id=64HdQKnyTc | The message-passing pattern that forbids immediate return messages on directed edges. | No imported architecture, no benchmark transfer, no claim that their GNN result proves this chess idea. |
| Jost, Mulas, Torres, “Spectral theory of the non-backtracking Laplacian for graphs” | https://doi.org/10.1016/j.disc.2023.113536 | Motivation that non-backtracking operators capture structural information different from classical graph operators. | No non-backtracking Laplacian is used here; this packet avoids another Laplacian/Hodge/sheaf variant. |

Citation verification note: the URLs above were checked during this research pass. The chess-specific hypothesis below is not proven by these papers.

Candidate search trace:

| Candidate mechanism considered | Why it was serious | Why it lost to the selected idea |
|---|---|---|
| Band-censored selective classifier for fine label `1` | It directly targets ambiguity in near-puzzles and would be label-safe if fine labels are used only as supervision. | It is mostly a loss/head idea, close to imported ordinal and credal-evidence packets, and gives little chess-specific structure. |
| Differentiable Horn-clause tactical circuit over static predicates | It could produce compact symbolic motifs such as “attacker of defender of king-zone square.” | It risks becoming a sparse-witness or high-order constellation duplicate, and Codex would need many hand-authored predicates before the experiment is clean. |
| Material-preserving contrastive board-shuffle anomaly model | It tests whether puzzle-like positions are atypical under material-conditioned board distributions. | It is too close to imported pseudo-likelihood and masked-code-length surprise families. |
| Bethe/free-energy propagation on attack-defense graph | It has a strong statistical-physics interpretation and could model tactical tension. | It is too close to static attack-defense energy, graph Laplacian, and sheaf-tension families unless heavily reworked. |
| Causal invariance across synthetic environments | It could suppress dataset artifacts without using source labels. | The available environments would likely be material/phase/color partitions, already covered by imported rule-partition invariance packets. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Non-backtracking walks | Hashimoto transition relation on directed current-board attack/protection edges: `e=(u->v)` may feed `f=(v->w)` only if `w != u` | Padded edge features `(B,E,D)`, transition indices `(B,T,2)`, relation-pair ids `(B,T)` | Degree/type-preserving randomized transition relation | It is not a sheaf, Hodge, Laplacian, curvature, transport plan, move-delta set, or king-path DP. |
| Sparse spectral structural signal | Damped edge-walk moments and pooled typed edge states after `K` non-backtracking propagation layers | `(B,K,R,D)` pooled relation-depth moments plus `(B,D)` global moments | Backtracking-allowed line-graph control and randomized transition control | Different from imported one-ply spectrum: the state space is current-board attack/protection edges, not legal/pseudo-legal move deltas. |
| Chess-specific partial equivariance | Side-to-move-relative coordinate features, not full board rotation/reflection invariance | Node/edge feature channels include signed rank/file, color relative to side-to-move, piece type | Raw-coordinate ablation and side-relative-coordinate ablation | It does not use orbit quotient/Reynolds pooling/file-mirror sheaf machinery. |
| Nuisance control | Count-only and degree-preserving ablations preserving material and attack/protection histograms | Tabular nuisance vector `(B,M)` for ablation head only | Count-only ablation matches main model on nuisance statistics but lacks edge-walk semantics | Not closed-form latent residualization or nuisance-orthogonal projection. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN on `simple_18` | `src/chess_nn_playground/models/cnn.py` | Already present; it tests generic local convolution, not a new tactical dependency operator. |
| Residual CNN | `src/chess_nn_playground/models/residual_cnn.py` | Already present; more residual depth would be routine architecture scaling. |
| LC0-style CNN or residual CNN on `lc0_bt4_112` | Existing LC0 BT4-style CNN/residual variants | Already represented; copying LC0-style planes or residual blocks is not a research mechanism. |
| Ordinary ViT over 64 squares | Common square-token Transformer | Too generic, data-hungry, and explicitly disallowed as a core idea. |
| Plain GNN on 64 board squares | Generic graph neural net over king/rook/bishop/knight neighborhoods | Too close to “put a GNN on the board”; without the non-backtracking edge-state falsifier it is not distinct. |
| Hyperparameter tuning | Any current trainer config | Disallowed; it would not test a new mathematical claim. |
| Ensembling current models | Any mixture of CNN/residual/LC0 baselines | Disallowed; it improves variance at the cost of interpretability and does not identify a new signal. |
| Add more data or train on the full 45M parquet | Existing data pipeline | Disallowed as the core idea and unsafe until streaming support exists. |
| Another tactical sheaf/Hodge/Laplacian/tension model | Imported sheaf/Hodge packet family | Already researched; the present idea deliberately avoids sheaf restrictions, Hodge operators, Laplacians, curvature, and tension energies. |
| One-ply move-delta landscape | Imported move-delta packet family | Already researched and would risk leakage-style move consequence shortcuts; this idea uses only current-board attack/protection edges. |
| Sinkhorn or piece-target transport | Imported optimal-transport packet family | Already researched; no transport coupling, cost matrix, or Sinkhorn step is used. |
| Pseudo-likelihood or masked-board code length | Imported pseudo-likelihood and masked-codec packets | Already researched; this packet does not train a generative board codec or predict masked pieces. |
| King-cage path DP or Hall-defect overload | Imported king-path and Hall/matroid packets | Already researched; this packet uses neither escape paths nor matching/transversal defects. |
| Ordinal ladder or binary Dirichlet evidence | Imported ordinal/credal packets | Not selected; near-puzzle diagnostics are measured but not modeled with cumulative thresholds or evidential Dirichlet heads. |

## 6. Mathematical Thesis

### Input space definition

Let `C` be the channel count of an encoding. The neural input space is

\[
\mathcal{X}_C \subseteq \{0,1\}^{C \times 8 \times 8},
\]

where each `x in X_C` decodes to a current legal-ish chess board state with side-to-move and optional castling/en-passant channels. The first experiment uses `simple_18`, because its current-board piece planes are expected to be directly interpretable.

Define a deterministic adapter

\[
D : \mathcal{X}_{18} \to \mathcal{B}
\]

that returns board occupancy, piece type, piece color, square coordinates, side-to-move, castling flags, and en-passant square if present. The proposed attack-graph builder is undefined if `D` cannot prove the channel semantics; it must fail closed.

### Label/target definition

Fine label:

\[
z \in \{0,1,2\}.
\]

Coarse binary target:

\[
y = \mathbf{1}\{z \geq 1\}.
\]

The network returns logits

\[
f_\theta(x) \in \mathbb{R}^2.
\]

### Data distribution assumptions

The data are sampled from an unknown distribution \(P(X,Z)\). The useful but unproven assumption is that, conditional on basic material and phase statistics, verified puzzles and near-puzzles are more likely than non-puzzles to contain short static chains of tactical dependency. Here a dependency chain means a sequence of current-board attack/protection relations such as

\[
p_0 \to p_1 \to p_2 \to \cdots \to p_k
\]

where each arrow is a pseudo-legal current-board attack/protection relation and the chain does not immediately reverse direction.

This is not a claim that puzzles are solved by static geometry. It is only a claim that many puzzle-like positions leave a current-board structural trace before any search is performed.

### Allowed symmetry or equivariance assumptions

Chess is not invariant under arbitrary board rotations/reflections because pawns, castling, en-passant, and side-to-move are directional. This packet assumes only:

1. Side-to-move-relative coordinate features are often useful.
2. Color-relative piece features are safe when computed from current board and side-to-move.
3. No full dihedral orbit quotient is enforced.
4. No file-mirror, rank-mirror, color-flip, or Reynolds pooling objective is central to the idea.

### Core hypothesis

Let \(G_x=(V_x,E_x,\tau_x)\) be the typed directed graph of current-board attack/protection relations. Let \(H_x\) be its non-backtracking transition operator on directed edges. The hypothesis is:

\[
I(Y;\Psi_K(H_X,E_X) \mid N(X)) > 0
\]

for small \(K\), where \(N(X)\) is a nuisance vector containing material, side-to-move, edge count, degree histogram, source-square marginals, target-piece histograms, and capture/protection counts, and \(\Psi_K\) is the learned non-backtracking edge-walk representation. In words: non-backtracking tactical-walk structure should contain label information not explained by obvious material and graph-count shortcuts.

### Formal object introduced

For a board \(x\), create a directed typed edge set \(E_x\). Each edge

\[
e=(u \to v, r)
\]

has source node \(u\), target node \(v\), and relation type \(r\), where relation types include at minimum:

- enemy piece attacked,
- friendly piece protected,
- enemy king-zone square attacked,
- own king-zone square protected.

Nodes are occupied squares plus optional virtual king-zone square nodes. Virtual nodes may receive edges but have no outgoing edges.

Let \(o(e)\) and \(t(e)\) be the origin and terminal node of edge \(e\). The unweighted Hashimoto operator is

\[
H_x[e,f] =
\mathbf{1}\{t(e)=o(f)\}\mathbf{1}\{o(e)\neq t(f)\}.
\]

The typed weighted version used by the model is

\[
\widetilde H_{\theta,x}[e,f]
=
H_x[e,f]\cdot a_\theta(\tau(e),\tau(f),\rho(e,f)),
\]

where \(\rho(e,f)\) is a small transition feature vector such as relation-type pair, source/through/target piece types, colors relative to side-to-move, and displacement bins. The weights are learned but the allowed transition support is deterministic.

Initial edge features are

\[
m_e^{(0)}=\phi_\theta(q_e),
\]

where \(q_e\) contains current-board source/target piece features, relative square geometry, relation type, distance, side-to-move-relative coordinates, and whether the target is an occupied piece or virtual king-zone node.

For layers \(\ell=0,\ldots,K-1\),

\[
m_f^{(\ell+1)}
=
\sigma\left(
W_0 m_f^{(\ell)}
+
\sum_{e \in E_x:\, H_x[e,f]=1}
W_{\tau(e),\tau(f)}m_e^{(\ell)}
+
b_{\tau(f)}
\right).
\]

The pooled feature is

\[
\Psi_K(x)=
\operatorname{Pool}\left(\{m_e^{(\ell)}: e\in E_x, \ell=0,\ldots,K\}\right),
\]

where `Pool` concatenates mean, max, log-sum-exp, relation-type means, depth-wise energies, and a small degree-normalized moment vector.

### Proposition

For the unweighted operator \(H_x\), the entry \((H_x^k)[e,f]\) equals the number of length-\(k+1\) directed edge walks beginning with edge \(e\) and ending with edge \(f\) that satisfy the non-backtracking condition at every intermediate step.

In particular, a reciprocal two-edge oscillation \(u\to v\to u\to v\to\cdots\) contributes to ordinary adjacency powers but contributes zero to non-backtracking powers after the attempted immediate reversal.

### Proof sketch or derivation

For \(k=1\), \(H_x[e,f]=1\) exactly when \(e\) can be followed by \(f\) and \(f\) is not the immediate reverse of \(e\). Assume the statement holds for \(k\). Then

\[
(H_x^{k+1})[e,g]
=
\sum_f (H_x^k)[e,f]H_x[f,g],
\]

which appends one legal non-backtracking transition \(f\to g\) to each length-\(k+1\) non-backtracking edge walk from \(e\) to \(f\). This is exactly the set of length-\(k+2\) non-backtracking edge walks from \(e\) to \(g\). The immediate reverse transition is excluded by the definition of \(H_x\).

The neural recurrence above is a learned, typed, nonlinear relaxation of these edge-walk counts. It does not prove label relevance; it proves only what structural paths the operator can and cannot count.

### Variational principle / optimization objective

The model is trained by empirical risk minimization:

\[
\min_\theta
\frac{1}{n}\sum_i
w_{y_i}\operatorname{CE}(f_\theta(x_i),y_i)
+
\lambda_{\text{edge}}\Omega_{\text{edge}}(\theta)
+
\lambda_{\text{drop}}\Omega_{\text{transition-drop}}(\theta).
\]

The intended inductive bias is a restricted function class:

\[
f_\theta(x)=h_\theta\left(c_\theta(x),\Psi_K(H_x,E_x)\right),
\]

where \(c_\theta(x)\) is a small square-board adapter and \(\Psi_K\) is constrained to use true non-backtracking edge-walk support. The central falsification ablation replaces \(H_x\) by \(\Pi_x(H_x)\), a degree/type-preserving random transition operator. If the main model beats \(\Pi_x(H_x)\), the gain is evidence for the edge-walk semantics rather than merely edge-token counts.

### What is actually proven

- The deterministic operator counts non-backtracking walks on the constructed current-board attack/protection edge graph.
- Immediate reciprocal backtracking is excluded by construction.
- The central randomized ablation can preserve many nuisance statistics while destroying the semantic edge-to-edge continuation relation.

### What remains only hypothesized

- That puzzle-like positions have more useful non-backtracking tactical-walk signal than non-puzzles after controlling for material, degrees, and relation histograms.
- That this signal appears in the current benchmark split strongly enough to improve binary classification and class-`1` near-puzzle diagnostics.
- That the constructed pseudo-legal attack/protection graph is the right granularity.

### Counterexamples where the idea should fail

- Quiet endgame studies where the key idea is zugzwang or distant opposition and current attack/protection chains are weak.
- Puzzles requiring a move-tree tactic invisible from current static attack/protection geometry.
- Sharp non-puzzle positions with many reciprocal attacks and long pressure chains.
- Positions whose puzzle labels are dominated by source-specific artifacts rather than board structure.
- Positions where the relevant motif is a legal constraint such as pinned king illegality that pseudo-legal attack geometry only approximates.

### Self-critique

The strongest objection is that this is still adjacent to static attack-defense graph modeling. It survives the anti-duplicate check only because the mathematical object under test is not a sheaf, Hodge operator, graph Laplacian, curvature, tension energy, transport plan, or move-delta landscape; it is the directed-edge non-backtracking transition semigroup. A second objection is that edge count and degree are strong shortcuts: tactical puzzles often have many forcing attacks, and a model might win without using non-backtracking semantics. That is why the central falsifier must preserve edge tokens, material, side-to-move, degree marginals, relation histograms, source-square marginals, target-piece histograms, and per-edge transition counts while randomizing the actual continuation relation. If the randomized-transition control matches the main model, abandon this idea.

The minimal experiment is still worth running because it is cheap, label-safe, current-data-compatible, and gives a decisive answer about a mathematically distinct operator not yet represented in the imported packets.

## 7. Architecture Specification

### Module names

Add one model file:

```text
src/chess_nn_playground/models/nonbacktracking_tactical_walk.py
```

Recommended classes/functions:

- `Simple18BoardParser`
- `EncodingChannelSpec`
- `AttackProtectionEdgeBuilder`
- `NonBacktrackingTransitionBuilder`
- `TypedEdgeEncoder`
- `NonBacktrackingEdgeBlock`
- `EdgeMomentPooler`
- `SmallBoardAdapter`
- `NonBacktrackingTacticalWalkNet`
- Builder function: `build_nonbacktracking_tactical_walk(config)`

### Forward-pass steps

Input:

```text
x: (B, C, 8, 8)
```

Step 1: encoding adapter.

- For `simple_18`, parse piece planes, side-to-move plane, castling planes, and en-passant plane according to explicit config.
- Output deterministic board records for graph construction.
- Fail closed if the encoding/channel map is missing or inconsistent.

Step 2: small learned board adapter.

```text
x -> Conv3x3(C, 32) -> GELU -> Conv3x3(32, 32) -> GELU -> global mean/max
board_latent: (B, 64)
```

This adapter is intentionally small. It is not the central claim.

Step 3: attack/protection edge construction.

For each board independently:

- Nodes:
  - Occupied square nodes, max `32`.
  - Optional virtual king-zone square nodes, max `18` for both kings including adjacent squares and the king square.
- Edges:
  - From an occupied source piece to an occupied target piece if the source pseudo-legally attacks/protects the target square under current blockers.
  - From an occupied source piece to virtual enemy king-zone target if the source attacks that square.
  - From an occupied source piece to virtual own king-zone target if the source protects that square.
- No legal move generation.
- No move count.
- No check/mate/stalemate oracle.

Padded outputs:

```text
edge_features:      (B, E_max, F_edge)
edge_valid_mask:    (B, E_max)
edge_src_node:      (B, E_max)
edge_tgt_node:      (B, E_max)
edge_relation_type: (B, E_max)
nuisance_counts:    (B, F_nuisance)   # for diagnostics/ablations, not required by main head
```

Recommended defaults:

```text
E_max: 512 initially; raise to 768 if overflow rate > 0.1%
F_edge: 48 to 80
```

If a board exceeds `E_max`, Codex should log the overflow count and use a deterministic safe truncation order:

1. occupied-target tactical edges,
2. king-zone edges,
3. shorter-distance edges,
4. source-square index,
5. target-square index.

The report must include overflow frequency. If overflow exceeds `0.5%`, increase `E_max` before comparing.

Step 4: transition construction.

For every pair of edges `e=(u->v)` and `f=(v->w)`:

```text
allow e -> f iff v is an occupied node and w != u
```

Virtual target nodes have no outgoing transitions.

Padded outputs:

```text
transition_src_edge:  (B, T_max)
transition_dst_edge:  (B, T_max)
transition_type_pair: (B, T_max)
transition_valid:     (B, T_max)
```

Recommended defaults:

```text
T_max: 4096 initially; raise to 8192 if overflow rate > 0.1%
```

Chunking plan:

- Build transitions per board on CPU in the first implementation for correctness.
- Move padded index tensors to device with the batch.
- For propagation, process transitions in chunks of `transition_chunk_size`, default `2048`, using `scatter_add` into destination edge states.
- Complexity is linear in valid transitions, not quadratic in `E_max`, after transition indices are built.

Step 5: edge encoding.

```text
edge_features: (B, E_max, F_edge)
edge_state_0 = MLP(F_edge -> d_edge): (B, E_max, d_edge)
```

Recommended:

```text
d_edge: 64
edge_encoder_layers: 2
dropout: 0.05
```

Step 6: non-backtracking edge propagation.

For `K=4` layers:

```text
incoming_sum[dst] += TypedLinear[type_pair](edge_state[src])
edge_state = LayerNorm(edge_state + GELU(self_linear(edge_state) + incoming_sum))
```

Shapes:

```text
edge_state_l: (B, E_max, 64)
incoming_sum: (B, E_max, 64)
```

Use relation-pair basis decomposition to avoid a large matrix per type pair:

```text
TypedLinear(type_pair, h) = W_shared h + sum_{j=1}^{R_basis} alpha[type_pair,j] W_j h
```

Recommended:

```text
num_relation_types: 4 to 8
num_type_pairs: <= 64
R_basis: 4
K: 4
```

Step 7: edge moment pooling.

Pool over valid edges at each depth:

- global mean,
- global max,
- log-sum-exp,
- relation-type mean,
- depth energy `mean(||m_e^l||^2)`,
- optional source-color-relative mean.

Output:

```text
edge_latent: (B, F_pool)
```

Recommended `F_pool` roughly `512` to `768`.

Step 8: classifier head.

```text
joint = concat(board_latent, edge_latent): (B, 64 + F_pool)
joint -> MLP -> logits: (B, 2)
```

Return only logits to remain trainer-compatible.

### Parameter-count estimate

Approximate defaults:

| Component | Estimated parameters |
|---|---:|
| Small board adapter | 10k to 25k |
| Edge feature MLP | 15k to 30k |
| Four basis typed edge blocks, `d_edge=64`, `R_basis=4` | 90k to 140k |
| Pool projection and classifier head | 80k to 180k |
| Total | 200k to 400k |

This is comparable to a small specialized model and should not be sold as a “bigger CNN.”

### FLOP / complexity estimate

For batch size `B`, valid edge count `E`, transition count `T`, edge width `d`, and layers `K`:

```text
edge encoding:      O(B * E * d * F_edge)
edge propagation:   O(B * K * T * d) with basis scatter-add
pooling/head:       O(B * K * E * d + B * hidden^2)
```

With defaults `E<=512`, `T<=4096`, `d=64`, `K=4`, propagation is roughly `B * 1.0M` multiply/add-style operations plus scatter overhead. Batch size `512` may be high on CPU graph building; if data loading becomes the bottleneck, first reduce batch size to `256` rather than changing the model.

### Memory estimate

Padded candidate tensors:

```text
edge_features: B * E_max * F_edge * 4 bytes
edge_states:   B * E_max * d_edge * 4 bytes * stored_depths
transitions:   B * T_max * 3 int64 values
```

For `B=512`, `E_max=512`, `F_edge=64`, `d_edge=64`, `T_max=4096`:

- Edge features: about `64 MB`.
- One edge-state tensor: about `64 MB`.
- If all `K+1` states are retained for pooling: about `320 MB`.
- Transition indices: about `512 * 4096 * 3 * 8 ~= 48 MB`.

Memory control:

- Store only pooled summaries for previous depths unless auxiliary debugging needs all states.
- Use transition chunks of `2048`.
- Use `int32` indices when PyTorch scatter path supports them; otherwise keep `int64`.
- Set `mixed_precision: false` for the first deterministic experiment.

### Required config fields

```yaml
model:
  name: nonbacktracking_tactical_walk
  input_channels: 18
  num_classes: 2
  encoding: simple_18
  simple18_piece_plane_order: [WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK]
  side_to_move_channel: 12
  castling_channels: [13, 14, 15, 16]
  en_passant_channel: 17
  edge_dim: 64
  edge_layers: 4
  edge_max: 512
  transition_max: 4096
  transition_chunk_size: 2048
  relation_types: [enemy_piece_attack, friendly_piece_protect, enemy_king_zone_attack, own_king_zone_protect]
  use_king_zone_virtual_nodes: true
  board_adapter_channels: 32
  dropout: 0.05
  ablation_mode: none
```

### Encoding support

First experiment: use only `simple_18`.

Reason: the deterministic graph builder must know the exact current piece planes. The safest first run is the encoding whose semantics are compact and expected to match `12 piece planes + side-to-move + castling + en-passant`.

Adapter assumptions:

| Encoding | Deterministic geometry support | Learned adapter support | Fail-closed rule |
|---|---|---|---|
| `simple_18` | Supported when channel map is explicit and validated. | Small board adapter consumes all 18 channels. | Raise `ValueError` if piece planes or side-to-move channel are missing. |
| `lc0_static_112` | Supported only after Codex provides an explicit current-board piece-plane map. | A learned adapter may consume all 112 channels. | Do not infer LC0 channel semantics from names or guesses. |
| `lc0_bt4_112` | Same as `lc0_static_112`; deterministic graph uses only current-board channels. | History channels may be consumed by learned adapter, not deterministic graph builder. | Zero-filled or unavailable history planes must not be interpreted as rule geometry. |

## 8. Loss, Training, And Regularization

Primary loss:

```text
weighted cross entropy over coarse binary labels
```

Class weighting:

```text
class_weighting: balanced
```

Optional auxiliary losses:

1. Edge-state dropout consistency, optional:
   - Run two transition-dropout masks on the same edge graph.
   - Encourage logits to agree with symmetric KL.
   - Default off for the first benchmark unless the main model overfits badly.

2. Edge overflow penalty, diagnostic only:
   - No gradient needed.
   - Report overflow counts rather than optimizing them.

Regularizers:

- Weight decay: `1e-4`.
- Dropout: `0.05` in edge encoder and classifier head.
- Transition dropout: `0.05` during training only, applied to valid transitions after the transition relation is built.
- No label smoothing in the first run; it may blur the near-puzzle diagnostic.

Batch size expectations:

- Start with `batch_size: 512` if CPU graph construction is acceptable.
- If graph building is the bottleneck, use `batch_size: 256` and keep all other settings unchanged.
- Keep `num_workers: 0` initially for deterministic debugging.

Optimizer defaults:

```text
AdamW
learning_rate: 0.001
weight_decay: 0.0001
epochs: 3
early_stopping_patience: 2
```

Determinism requirements:

- Seed Python, NumPy, and PyTorch with `42`.
- Set deterministic PyTorch flags where the existing trainer supports them.
- Log edge-builder overflow counts and randomized-ablation seeds.
- Randomized ablations must use fixed seeds and persist their randomization mode in the report.

What must stay unchanged for fair comparison:

- Train/val/test split paths.
- Coarse binary mode.
- Label mapping.
- Metrics and `3x2` fine-label diagnostic.
- Epoch count for the minimal experiment unless all baselines are rerun with the same new count.
- Class weighting policy.
- No direct full 45M parquet training.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Degree/type-preserving randomized transitions | Keeps edge tokens, material, side-to-move, source-square marginals, target-piece histograms, capture/protection counts, per-edge transition counts, and relation-type-pair counts, but randomizes which edge continuations are semantically connected. | Central claim: true non-backtracking continuation semantics matter beyond nuisance statistics. | If this matches the main model, abandon the non-backtracking-walk mechanism. |
| Backtracking-allowed line graph | Allows `u->v` to feed immediate reverse `v->u`. | Non-backtracking exclusion, not just edge-state propagation, is useful. | If it matches or beats main, the no-immediate-return constraint is not helping. |
| Edge-token DeepSets | Encodes each attack/protection edge independently and pools without transitions. | Multi-edge tactical dependency chains matter beyond edge inventory. | If it matches main, edge counts/features are enough; avoid walk operators next cycle. |
| Count-only nuisance head | Uses material, side-to-move, edge count, degree histogram, source-piece histogram, target-piece histogram, relation histogram, king-zone edge counts, and capture/protection counts only. | The model is not merely using obvious graph/material shortcuts. | If close to main, the dataset split may be shortcut-dominated. |
| Source/target square shuffle | Preserves source piece identity, target piece identity, relation type, and degree counts but shuffles square coordinates within side-relative rank/file bins. | Geometric continuation and board placement matter. | If close to main, square geometry is not contributing. |
| Relation-label shuffle | Preserves graph connectivity and degrees but permutes relation labels among edges with same source and target occupancy classes. | Attack vs protection vs king-zone semantics matter. | If close to main, relation semantics are not needed. |
| No king-zone virtual nodes | Removes virtual king-zone edges while keeping occupied-piece attack/protection edges. | Terminal pressure on king neighborhoods contributes signal. | If equal or better, omit virtual king-zone nodes in scaled runs. |
| No board adapter | Uses only non-backtracking edge-walk features. | Edge-walk features can stand without CNN nuisance/local pattern help. | If performance collapses, edge graph alone is insufficient; if unchanged, board adapter is unnecessary. |
| Board adapter only | Removes edge graph and trains the same small convolutional adapter/head. | Checks whether the improvement is just the small CNN branch. | If close to main, edge branch is not useful. |
| Random legal-shaped edge graph | Preserves total edge count and relation histogram but assigns edges among occupied pieces using legal-looking displacement buckets unrelated to actual attacks. | Actual current-board attack geometry matters, not just sparse graph size. | If close to main, the graph builder semantics are suspect. |
| Fewer walk depths `K=1` | Allows only one transition layer. | Short chains are enough or deeper non-backtracking chains help. | If `K=1` wins, scale with shallower model. |
| Larger walk depths `K=6` diagnostic only | Adds longer chains without changing other settings. | Longer dependency chains add signal. | If worse, over-smoothing/overfitting is likely; keep `K=4` or less. |

The first row is the smallest central falsification ablation. It must be included for any report claiming this idea works.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- Existing simple CNN on `simple_18`.
- Existing residual CNN on `simple_18`.
- Existing small/medium/deep variants if already benchmarked on the same split.
- Existing LC0-style models only as context; do not claim a direct encoding-controlled win unless the new model is also run on a validated LC0 channel map.
- New ablations from Section 9.

Metrics to inspect:

- Test accuracy.
- Test AUROC if the reporting stack supports it.
- Test average precision if available.
- Balanced accuracy.
- Binary confusion matrix.
- Required rectangular diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Near-puzzle diagnostic:

- For the main model and central ablations, compute class-`1` recall at a threshold chosen to match the best simple_18 baseline's fine-label-`0` false-positive rate on validation.
- Also report class-`1` precision among predicted positives at that matched threshold.

Required artifacts:

- `metrics.json` for main and every central ablation.
- `confusion_binary.csv`.
- `confusion_fine3_by_pred2.csv`.
- `predictions_test.parquet` with logits/probabilities, fine label, binary label, and model id.
- `edge_builder_stats.json` with edge counts, transition counts, overflow rate, relation histograms, and truncation counts.
- `ablation_summary.md`.
- Updated leaderboard row.

Success threshold:

- Main model improves over the best same-encoding `simple_18` baseline by at least `+1.5` percentage points in AUROC or balanced accuracy, or by at least `+3.0` percentage points in class-`1` recall at matched fine-label-`0` false-positive rate.
- Main model also beats the degree/type-preserving randomized-transition ablation by at least `+1.0` percentage point in AUROC or balanced accuracy.
- The improvement is stable over at least `3` seeds before any scaling claim.

Failure threshold:

- Main model is within `±0.5` percentage points of the randomized-transition ablation and edge-token DeepSets control.
- Count-only nuisance head is within `1.0` percentage point of main.
- Edge-builder overflow exceeds `0.5%` and cannot be fixed by raising caps within memory limits.
- Near-puzzle class-`1` recall does not improve at matched fine-label-`0` false-positive rate.

What result would make me abandon the idea:

- Randomized transitions match the main model while count-only or Edge-token DeepSets controls are close. That would mean the non-backtracking continuation relation is not the source of signal.

What result would justify scaling:

- The main model beats same-encoding baselines and the randomized-transition ablation on test, improves class-`1` matched-FPR recall, has low overflow, and preserves the ablation gap across three seeds. Only then consider larger `E_max`, `T_max`, `d_edge`, or validated LC0 current-board adapters.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_nonbacktracking_walk/idea.yaml` | Create | Machine-readable idea summary, status, config/model paths, and central ablation. |
| `ideas/20260421_nonbacktracking_walk/math_thesis.md` | Create | Mathematical thesis from Section 6, with proposition and proof sketch. |
| `ideas/20260421_nonbacktracking_walk/architecture.md` | Create | Architecture details, tensor shapes, memory plan, and pseudocode. |
| `ideas/20260421_nonbacktracking_walk/implementation_notes.md` | Create | Encoding adapter rules, edge-builder details, fail-closed LC0 handling, overflow logging. |
| `ideas/20260421_nonbacktracking_walk/trainer_notes.md` | Create | Loss, optimizer, deterministic settings, class weighting, and reporting requirements. |
| `ideas/20260421_nonbacktracking_walk/ablations.md` | Create | Full ablation plan and randomized-transition nuisance-preservation requirements. |
| `ideas/20260421_nonbacktracking_walk/train.py` | Create | Thin entrypoint or wrapper that invokes the shared trainer with this idea config; no separate trainer fork unless required. |
| `ideas/20260421_nonbacktracking_walk/config.yaml` | Create | Minimal `simple_18` experiment config from the machine-readable block. |
| `ideas/20260421_nonbacktracking_walk/report_template.md` | Create | Required tables for metrics, fine `3x2` confusion, near-puzzle matched-FPR diagnostic, edge stats, and ablation comparison. |
| `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Preserve hard constraints and add anti-duplicate guidance for Hashimoto/non-backtracking tactical edge-walk operators after this packet is consumed. |
| `src/chess_nn_playground/models/nonbacktracking_tactical_walk.py` | Create | PyTorch module with parser, edge builder, transition builder, edge blocks, pooler, and `NonBacktrackingTacticalWalkNet`. |
| `src/chess_nn_playground/models/registry.py` | Modify | Register `nonbacktracking_tactical_walk` builder. |
| `configs/nonbacktracking_tactical_walk_simple18.yaml` | Create | Shared-trainer-compatible config for the minimal run. |
| `tests/test_nonbacktracking_tactical_walk.py` | Create | Focused tests for parser fail-closed behavior, edge construction on simple hand positions, no immediate backtracking, tensor shapes, and logits shape. |
| `tests/test_nonbacktracking_ablation_randomization.py` | Create | Test that randomized transitions preserve counts/marginals and destroy exact semantic adjacency for a fixed seed. |

Pseudocode, not final implementation:

```text
forward(x):
    board_latent = board_adapter(x)

    board_records = parser.parse(x, encoding_spec)
    edges = edge_builder.build(board_records, E_max)
    transitions = transition_builder.build(edges, T_max, mode=ablation_mode)

    h = edge_encoder(edges.features)
    pooled = [pool(h, edges.mask, edges.relation_type)]

    for layer in edge_blocks:
        msg = scatter_typed_messages(
            h,
            transitions.src_edge,
            transitions.dst_edge,
            transitions.type_pair,
            transitions.mask,
            chunk_size=transition_chunk_size,
        )
        h = layer.update(h, msg, edges.mask)
        pooled.append(pool(h, edges.mask, edges.relation_type))

    edge_latent = concat(pooled)
    logits = classifier(concat(board_latent, edge_latent))
    return logits
```

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0922_tuesday_local_nonbacktracking_walk.md
  generated_at: 2026-04-21 09:22:45 America/Los_Angeles
  weekday: Tuesday
  timezone: local
  idea_slug: nonbacktracking_walk
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_nonbacktracking_walk
  name: Non-Backtracking Tactical Walk Network
  slug: nonbacktracking_walk
  status: draft
  created_at: 2026-04-21 09:22:45 America/Los_Angeles
  author: ChatGPT Pro
  short_thesis: Puzzle-like positions leave short current-board attack/protection dependency chains that are better represented by non-backtracking directed-edge walks than by square CNNs or degree counts.
  novelty_claim: Uses a Hashimoto-style non-backtracking transition operator over deterministic current-board tactical edges, with degree/type-preserving randomized-transition falsification; not a sheaf, Hodge/Laplacian, move-delta, Sinkhorn/OT, topology, Hall, king-path, ordinal, or masked-codec packet.
  expected_advantage: Improved class-1 near-puzzle recall at matched fine-label-0 false-positive rate and improved binary metrics over same-encoding simple_18 baselines if non-backtracking tactical chains are real signal.
  central_falsification_ablation: Degree/type-preserving randomized non-backtracking transitions while keeping edge tokens and nuisance histograms fixed.
  target_task: coarse_binary
  input_representation: simple_18 first; LC0 only after explicit current-board channel map validation
  output_heads: binary logits only
  compute_notes: Edge graph per board with E_max 512, T_max 4096, d_edge 64, K 4; chunk transition scatter-add if needed.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/nonbacktracking_tactical_walk_simple18.yaml
  model_path: src/chess_nn_playground/models/nonbacktracking_tactical_walk.py
  latest_result_path: null
  notes: Include edge-builder overflow stats and central randomized-transition ablation in the first report.
```

```yaml
config_yaml:
  run:
    name: nonbacktracking_tactical_walk_simple18
    output_dir: results
  seed: 42
  deterministic: true
  mode: coarse_binary
  device: nvidia
  data:
    train_path: data/splits/crtk_sample_3class/split_train.parquet
    val_path: data/splits/crtk_sample_3class/split_val.parquet
    test_path: data/splits/crtk_sample_3class/split_test.parquet
    encoding: simple_18
    cache_features: false
  model:
    name: nonbacktracking_tactical_walk
    input_channels: 18
    num_classes: 2
    simple18_piece_plane_order: [WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK]
    side_to_move_channel: 12
    castling_channels: [13, 14, 15, 16]
    en_passant_channel: 17
    edge_dim: 64
    edge_layers: 4
    edge_max: 512
    transition_max: 4096
    transition_chunk_size: 2048
    use_king_zone_virtual_nodes: true
    board_adapter_channels: 32
    dropout: 0.05
    ablation_mode: none
  training:
    epochs: 3
    batch_size: 512
    num_workers: 0
    learning_rate: 0.001
    weight_decay: 0.0001
    class_weighting: balanced
    early_stopping_patience: 2
    mixed_precision: false
```

```yaml
model_spec:
  model_name: nonbacktracking_tactical_walk
  file_path: src/chess_nn_playground/models/nonbacktracking_tactical_walk.py
  builder_function: build_nonbacktracking_tactical_walk
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18BoardParser
    - AttackProtectionEdgeBuilder
    - NonBacktrackingTransitionBuilder
    - TypedEdgeEncoder
    - NonBacktrackingEdgeBlock
    - EdgeMomentPooler
    - SmallBoardAdapter
    - NonBacktrackingTacticalWalkNet
  required_config_fields:
    - input_channels
    - num_classes
    - simple18_piece_plane_order
    - side_to_move_channel
    - edge_dim
    - edge_layers
    - edge_max
    - transition_max
    - transition_chunk_size
    - ablation_mode
  expected_parameter_count: 200k-400k with default d_edge=64 and K=4
  expected_memory_notes: For B=512,E_max=512,T_max=4096,d=64, expect roughly 64MB edge features, 64MB per edge-state tensor, about 48MB int64 transition indices, plus activations; chunk transitions at 2048 and reduce batch to 256 if CPU graph building or memory becomes limiting.
```

```yaml
research_continuity:
  idea_fingerprint: current-board pseudo-legal attack/protection edge graph + Hashimoto non-backtracking directed-edge transition operator + typed edge message moments + small board adapter + binary puzzle-likeness target
  already_researched_family_overlap: Adjacent to static attack geometry but not a sheaf/Hodge/Laplacian/tension/curvature model; adjacent to spectral ideas but not one-ply move-delta spectrum; no OT, topology, Hall, king-path DP, ordinal, credal, masked-codec, or nuisance projection mechanism.
  closest_duplicate_risk: Static attack-defense graph packets; the distinction is the non-backtracking edge-line semigroup and degree/type-preserving randomized-transition falsifier.
  do_not_repeat_if_this_fails:
    - Hashimoto/non-backtracking attack-protection edge-walk operators over current boards
    - Damped non-backtracking tactical walk moments
    - Backtracking-vs-nonbacktracking edge-line graph controls as the central novelty
    - Static attack/protection edge propagation that can be reduced to edge inventory plus degree histograms
  suggested_next_search_directions:
    - Label-safe selective prediction for near-puzzles that is not ordinal or evidential Dirichlet
    - Causal/source-shift ideas only if genuine provenance or non-imported environments become available
    - Generative motif compression that is not pseudo-likelihood, masked-board surprise, or material-conditioned shuffle anomaly
    - Differentiable symbolic proof objects only if they avoid sparse-witness and high-order-constellation duplication
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add “Hashimoto/non-backtracking tactical edge-walk operators over current-board attack/protection graphs” to the imported researched-family list after this packet is implemented and evaluated. | Prevents the next research pass from proposing the same non-backtracking edge-line graph idea with a different name. | `Imported Research Memory` |
| Add an anti-duplicate rule: do not propose another non-backtracking attack/protection edge-walk model unless the formal state space or falsifier changes beyond relation labels, depth, pooling, or hidden size. | Clarifies that future variants need a genuinely new operator, not larger `K` or more edge types. | Anti-duplicate paragraphs after sheaf/move-delta/OT rules |
| Require any future graph-structured current-board idea to include degree/type-preserving randomized structure controls that preserve material, side-to-move, edge count, degree marginals, source-square marginals, target-piece histograms, and relation histograms. | This packet shows that graph models are vulnerable to nuisance shortcuts; the control should become standard. | `Ablation Plan` requirements |
| Add a reminder that LC0 deterministic geometry adapters must fail closed unless current-board channel semantics are explicitly configured. | Prevents accidental misuse of history or unknown LC0 planes as rule-derived features. | `Problem Restatement And Data Contract` |
| If this idea fails, add “non-backtracking/damped edge-walk spectra over static attack geometry” to `mechanisms not to repeat`. | Makes failed negative evidence useful to the iterative loop. | `Research Continuity` guidance |

Do not weaken leakage rules, label rules, falsification requirements, or anti-duplicate requirements.

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes
- Falsification criterion is concrete: yes
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported research packets completed: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Möbius-constellation, or pseudo-likelihood packets: yes
- Not an exact near-duplicate of imported orbit-symmetry, tempo-intervention, credal-evidence, rule-partition-invariance, kinematic-commutator, or masked-codec packets: yes
- Not an exact near-duplicate of imported cubical Euler/Betti topology, Hall-defect overload, or king-cage/king-escape path-DP packets: yes
