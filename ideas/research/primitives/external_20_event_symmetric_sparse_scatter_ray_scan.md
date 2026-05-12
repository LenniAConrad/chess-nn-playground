# Invent New Neural Primitives for Chess Evaluation

This document proposes five candidate neural-network primitives for `chess-nn-playground`. The filter used here is deliberately strict: no new architecture, no new input encoding, no training trick, and no Stockfish/PV/node metadata in the compute graph. I treat the prompt’s “primitive” definition as the operative bar: a reusable mathematical operator with a distinct signature, gradient/connectivity pattern, or complexity profile, not a mere composition of attention, convolution, normalization, or routing blocks.

No result below is claimed as measured. All “works/fails” thresholds are falsification criteria.

---

## 1. Proposal

### primitive_event_symmetric_accumulator

**Name:** Event-Symmetric Interaction Accumulator

**One-line claim:** Maintains exact low-order set interactions under sparse add/remove events in time proportional to the changed features.

**Mathematical signature:**
For active feature set \(S_b\), embeddings \(u_i=\phi_\theta(x_i)\in\mathbb{R}^d\), order \(R\), define
\[
E_b^{(0)}=\mathbf{1},\quad E_b^{(r)}=\sum_{i_1<\cdots<i_r\in S_b} u_{i_1}\odot\cdots\odot u_{i_r}\in\mathbb{R}^d.
\]
Primitive:
\[
f_R:\left(\{E^{(r)}\}_{r=0}^R,\Delta^+,\Delta^-\right)\rightarrow \operatorname{concat}(E'^{(1)},\ldots,E'^{(R)})\in\mathbb{R}^{B\times Rd}.
\]
Add event \(u\): for \(r=R,\ldots,1\),
\[
E^{(r)}\leftarrow E^{(r)}+u\odot E^{(r-1)}.
\]
Remove event \(u\): set \(\tilde E^{(0)}=\mathbf{1}\), then for \(r=1,\ldots,R\),
\[
\tilde E^{(r)}=E^{(r)}-u\odot \tilde E^{(r-1)}.
\]
Gradients are exact: \(\partial E^{(r)}/\partial u_i=E^{(r-1)}_{S\setminus\{i\}}\).

**Why this does not decompose into existing PyTorch ops:**
`EmbeddingBag`/sparse linear accumulation gives only first-order sums; explicit pair/triple enumeration gives a different \(O(|S|^R)\) graph. This primitive’s graph is a reversible dynamic program over elementary symmetric Hadamard products with event-sparse reverse mode. It is closest to polynomial pooling, but polynomial pooling is normally recomputed from the full set and does not expose add/remove deltas as the primitive’s native input.

**Chess-specific motivation:**
NNUE’s value comes from sparse features and accumulator updates: Stockfish documentation describes updating the first-layer accumulator by adding/subtracting changed feature columns instead of recomputing it. Chess tactics are often pair/triple phenomena: fork, pin, skewer, discovered attack, king-piece-defender alignment. This primitive chases the HalfKA/NNUE update property while adding exact low-order interactions.

**Generalisation beyond chess:**
Useful for sparse-event sequences, recommender baskets, fraud logs, and dynamic molecular/contact sets where low-order interactions matter but edits are small.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B|S|Rd)\) from scratch vs pairwise attention \(O(B|S|^2d)\)
- Backward: \(O(B|S|Rd)\)
- Incremental update on a bounded-change input: \(O(B|\Delta|Rd)\)

**Scout-scale falsification test:**
Drop into i193 as an occupied-square/set aggregation replacement after the existing conv body; baseline is i193 with its original pooling/head and matched parameter count. Use \(R=2\), \(d\le64\). Metric: matched-recall CRTK class-1 near-puzzle false-positive rate. Primitive works if FP rate drops by at least 5% relative at the same recall and evaluation speed is no worse than 1.15× baseline. It fails if aggregate PR AUC rises but near-puzzle FP does not improve.

**Failure mode catalogue:**
- Hidden rebrand: with \(R=1\), it collapses to `EmbeddingBag`/sum pooling and should be rejected.
- Numerical instability: high-order Hadamard products can explode/vanish; keep \(R\le3\), use normalization or bounded \(\phi_\theta\).
- Speed objection: if implemented by materializing all pairs, it loses the primitive; only the dynamic-program implementation counts.

**Status:** proposed

---

## 2. Proposal

### primitive_rule_generated_sparse_scatter

**Name:** Rule-Generated Sparse Scatter

**One-line claim:** Generates content-dependent discrete edges inside the operator and scatters messages only along those edges.

**Mathematical signature:**
\[
f:\mathbb{R}^{B\times n\times d}\times\{0,1\}^{B\times n}\times\{0,\ldots,T\}^{B\times n}\rightarrow \mathbb{R}^{B\times n\times d}.
\]
Given features \(X\), occupancy \(z\), and token tags \(\tau\), a compiled rule kernel emits
\[
E_b=\{(j,i,r):\operatorname{Rule}_r(z_b,\tau_b,j,i)=1\}.
\]
Forward:
\[
y_{b,i}=W_0x_{b,i}+\sum_{(j,i,r)\in E_b}
g_r(x_{b,i},x_{b,j})\odot W_rx_{b,j}.
\]
Edges are nondifferentiable with respect to \(z,\tau\); gradients flow through \(X,W,g\).

**Why this does not decompose into existing PyTorch ops:**
`scaled_dot_product_attention` accepts a mask tensor and then performs dense score/softmax/value computation; PyTorch’s own documentation defines the mask as an input to attention, not as a rule-generated topology. Standard MPNNs assume an input graph and learn messages over it; they do not make graph construction the primitive. The new computation graph has a fused discrete edge generator plus sparse scatter, with backward only over emitted edges.

**Chess-specific motivation:**
Legal-move and attack connectivity changes per board and per side to move; the prompt explicitly identifies this dynamic connectivity as the most chess-specific structural fact. This primitive is not “attention with a clever mask” because the mask is not a precomputed tensor argument. It is generated inside the operator from occupancy and piece tags, then used for sparse message flow.

**Generalisation beyond chess:**
Applicable to dynamic scene graphs, compiler/program-analysis graphs, molecule rule bonds, traffic networks, and event systems where edges are rule-derived from current state.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B(|E|d+n d^2))\) vs dense attention \(O(Bn^2d)\)
- Backward: \(O(B|E|d)\)
- Incremental update on a bounded-change input: \(O(B|\Delta E|d)\), except global ray invalidations need refresh

**Scout-scale falsification test:**
Use the i242 legal-move-attention harness, but replace masked attention with Rule-Generated Sparse Scatter. Baselines: i242 masked attention and conv-only i193. Metric: matched-recall near-puzzle FP rate plus measured positions/sec. Primitive works if it beats i242 on FP rate and is at least 1.5× faster than i242 attention, or matches i193 speed within 20% while improving FP. It fails if it only improves easy-negative PR AUC.

**Failure mode catalogue:**
- Hidden rebrand: if `edge_index` is precomputed outside the op and fed in, it is just GNN/GAT.
- Stability: hard edge discontinuities may make small board changes cause large activation jumps.
- Speed objection: CPU-side legal-edge generation can erase the sparse FLOP advantage; needs fused CUDA/C++ rule generation.

**Status:** proposed

---

## 3. Proposal

### primitive_first_blocker_ray_scan

**Name:** Differentiable First-Blocker Ray Scan

**One-line claim:** Performs occlusion-aware directional scans, making sliding-piece visibility a primitive rather than dense board attention.

**Mathematical signature:**
Let \(X\in\mathbb{R}^{B\times64\times d}\), soft/hard occupancy \(o\in[0,1]^{B\times64}\), directions \(\mathcal{D}=8\), and \(r(s,\delta,\ell)\) be the square \(\ell\) steps from square \(s\) along direction \(\delta\). Define
\[
v_{b,s,\delta,\ell}=\prod_{m<\ell}\left(1-o_{b,r(s,\delta,m)}\right).
\]
Forward:
\[
y_{b,s}=\sum_{\delta\in\mathcal{D}}\sum_{\ell=1}^{L(s,\delta)}
v_{b,s,\delta,\ell}\,A_{\delta,\ell}x_{b,r(s,\delta,\ell)}+b.
\]
If \(o\) is continuous, gradients flow through prefix products; if \(o\) is discrete, the visibility path is stop-gradient and gradients flow through \(X,A\).

**Why this does not decompose into existing PyTorch ops:**
`Conv2d` uses fixed local cross-correlation independent of board occupancy. Dense attention can simulate visible squares only after receiving a visibility mask, but here the ordered first-blocker scan and its backward pass are the primitive. It is related in spirit to Mamba-style selective scans, where input-dependent recurrence is the core operator, but this scan is over 2D rays with blocker products rather than sequence state transitions.

**Chess-specific motivation:**
Rooks, bishops, queens, pins, skewers, discovered attacks, and king safety are all first-blocker phenomena. A 3×3 conv must learn ray extension indirectly, and attention pays for all square pairs. This primitive directly computes “what is visible until blocked” at board scale.

**Generalisation beyond chess:**
Useful for line-of-sight robotics, grid-world planning, differentiable visibility, tabletop scene graphs, and sparse ray interactions in games.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B\cdot64\cdot8\cdot L\cdot d)\), \(L\le7\), vs dense attention \(O(B64^2d)\)
- Backward: \(O(B\cdot64\cdot8\cdot L\cdot d)\)
- Incremental update on a bounded-change input: \(O(B|\Delta|\cdot8\cdot L\cdot d)\), or \(O(B|\Delta|\cdot8\log L\cdot d)\) with per-ray prefix caches

**Scout-scale falsification test:**
Drop one First-Blocker Ray Scan after the first third of i193’s conv body; baseline is i193 with a matched-parameter 5×5 or dilated conv replacement. Metric: matched-recall near-puzzle FP rate and raw inference latency. Primitive works if FP falls by at least 3% relative with latency increase under 20%. It fails if it only helps positions with obvious long-range sliders and hurts quiet endgames.

**Failure mode catalogue:**
- Hidden rebrand: if implemented as precomputed attack planes, it becomes an encoding, not a primitive.
- Numerical instability: continuous occupancy products can vanish on long blocked rays.
- Speed objection: naive Python ray loops will be slower than dense conv; requires fused scan kernels.

**Status:** proposed

---

## 4. Proposal

### primitive_chess_irrep_orbit_norm

**Name:** Chess-Irrep Orbit Normalization

**One-line claim:** Normalizes even and odd chess-symmetry components over board/color orbits while preserving equivariance.

**Mathematical signature:**
Let finite group \(G\) act on token-channel indices by permutation/sign representation \(P_g\). For irrep/sign character \(\chi_\ell(g)\), define projector
\[
\Pi_\ell X=\frac{1}{|G|}\sum_{g\in G}\chi_\ell(g)P_gX.
\]
For each orbit \(O\) and component \(\ell\):
\[
\mu_{O,\ell}=\operatorname{mean}_{i\in O,c}(\Pi_\ell X)_{i,c},\quad
\sigma^2_{O,\ell}=\operatorname{var}_{i\in O,c}(\Pi_\ell X)_{i,c}.
\]
Forward:
\[
Y=\sum_\ell \gamma_\ell\odot\frac{\Pi_\ell X-\mu_{O,\ell}}{\sqrt{\sigma^2_{O,\ell}+\epsilon}}+\beta_\ell.
\]

**Why this does not decompose into existing PyTorch ops:**
`LayerNorm` normalizes over trailing feature dimensions and has no group-action projectors. `GroupNorm` groups channels, not chess-board/color orbits. Existing group-equivariant CNNs use group convolutions and weight sharing for rotations/reflections, but this primitive is a normalization operator over irreducible even/odd components, including color-swap-like involutions.

**Chess-specific motivation:**
Chess has useful near-symmetries beyond plain board rotations: color swap, side-to-move perspective, piece-type structure, and square orbits. The prompt calls out that chess group structure is under-exploited and not just dihedral-4. This primitive gives the model a symmetry-preserving normalization step without inventing new planes or changing labels.

**Generalisation beyond chess:**
Useful for other finite-group domains: board games, molecule automorphisms, UI layouts, circuit graphs, and robotics state spaces with signed symmetries.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B|G|nd)\) vs LayerNorm \(O(Bnd)\)
- Backward: \(O(B|G|nd)\)
- Incremental update on a bounded-change input: \(O(B|G|d)\) for affected orbit moments, or not applicable if used only in batched training

**Scout-scale falsification test:**
Replace LayerNorm/BatchNorm-like normalization sites in the smallest i242 ablation or an i193-norm variant with Chess-Irrep Orbit Normalization. Baseline: identical model with standard LayerNorm/GroupNorm. Metric: matched-recall near-puzzle FP rate plus calibration error under color-swapped evaluation pairs. Primitive works if FP improves and color-swap consistency improves without >10% inference slowdown. It fails if consistency improves but tactical FP does not.

**Failure mode catalogue:**
- Hidden rebrand: if \(G\) is trivial or only channel groups are used, this is just GroupNorm/LayerNorm.
- Correctness risk: castling, en passant, and side-to-move break some naive symmetries; the chosen group action must be legal-state aware.
- Speed objection: enumerating too large a group can dominate an 8×8 model; keep \(G\) small and explicit.

**Status:** proposed

---

## 5. Proposal

### primitive_counterfactual_delta_map

**Name:** Counterfactual Delta Map

**One-line claim:** Evaluates many bounded-edit successors by applying sparse accumulator deltas without materializing child tensors.

**Mathematical signature:**
Let current accumulator \(A\in\mathbb{R}^{B\times d}\), edit set \(M_b=\{e\}_{e=1}^{m_b}\), feature table \(W\in\mathbb{R}^{N\times d}\), and each edit \(e\) contain added/removed feature IDs \(I_e^+,I_e^-\). Define
\[
\Delta_e=\sum_{i\in I_e^+}W_i-\sum_{j\in I_e^-}W_j.
\]
Primitive:
\[
f(A,M,W,h_\theta)\rightarrow Y,\quad
Y_{b,e}=h_\theta(A_b+\Delta_{b,e})\in\mathbb{R}^q.
\]
Backward accumulates gradients into \(A\), \(W_i\), and \(h_\theta\) over all counterfactual edits.

**Why this does not decompose into existing PyTorch ops:**
A reference implementation can loop over edits with gathers, additions, and a head, but that creates \(m\) child graphs and materialized child states. This primitive’s signature is “state plus bounded edit list to successor outputs,” with fused sparse delta application and shared reverse-mode accumulation. It generalizes the NNUE accumulator idea from “current node update” to “all bounded counterfactual child updates.” Stockfish NNUE’s documented accumulator mechanism updates changed feature columns instead of recomputing full inputs; this primitive exposes that as a batched differentiable operator.

**Chess-specific motivation:**
Near-puzzle false positives often require asking, “what happens after the forcing move?” without running a full search. Chess moves change a bounded number of feature IDs, so child accumulators can be computed by deltas. The primitive does not use Stockfish scores or PVs; legal moves are board-derived structure, not labels.

**Generalisation beyond chess:**
Useful for recommender “what-if” edits, database update models, routing/search problems, and dynamic graph successor evaluation.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(Bmcd+BmH)\), where \(c\) is changed IDs/edit and \(H\) is head cost, vs evaluating \(m\) full child tensors \(O(BmNd)\) or \(m\) conv forwards
- Backward: \(O(Bmcd+BmH)\)
- Incremental update on a bounded-change input: \(O(Bmcd)\), or \(O(B|\Delta M|cd)\) if successor edit sets are cached

**Scout-scale falsification test:**
Use i243 HalfKA+dual-stream as harness. Compare parent-only i243, naive batched child-eval i243 on legal moves, and i243 with Counterfactual Delta Map using the same small head \(h_\theta\). Metric: matched-recall near-puzzle FP rate and positions/sec. Primitive works if it captures most of the child-eval FP reduction while staying at least 2× faster than naive batched child evaluation. It fails if parent-only is equally good.

**Failure mode catalogue:**
- Hidden rebrand: if implemented as a Python loop of `EmbeddingBag` calls, it is not a primitive.
- Numerical risk: many sibling deltas can concentrate gradients on a few feature columns.
- Speed objection: if \(h_\theta\) is large, the sparse-delta advantage disappears; the head must stay tiny.

**Status:** proposed

---

# Ranking

| Rank | Primitive | Novelty plausibility | RTX 3070 demonstrability | Inference-speed potential | Generalisation beyond chess | Bottom line |
|---:|---|---:|---:|---:|---:|---|
| 1 | `primitive_event_symmetric_accumulator` | High | High | High | High | Best scout-scale bet; directly generalizes NNUE-style sparse updates. |
| 2 | `primitive_rule_generated_sparse_scatter` | High if edge generation is fused | Medium | High | High | Most structurally chess-specific; biggest implementation risk. |
| 3 | `primitive_first_blocker_ray_scan` | Medium-high | High | Medium-high | Medium | Clean tactical bias; likely fast on 8×8 if fused. |
| 4 | `primitive_counterfactual_delta_map` | Medium-high | Medium | High vs child eval, medium vs parent-only | High | Strong for hard negatives, but easiest for reviewers to call vectorized deltas. |
| 5 | `primitive_chess_irrep_orbit_norm` | Medium | High | Medium | High | Safest to test, but novelty is narrower because equivariant normalization literature exists. |

---

# Self-audit of the top 2

## `primitive_event_symmetric_accumulator`

Devil’s advocate: this is just polynomial pooling, or just `sum`, `mul`, and dynamic programming. If \(R=1\), that objection is correct: it is exactly a sum/EmbeddingBag-style accumulator and should be dropped. For \(R\ge2\), the primitive’s claim is narrower: exact elementary symmetric interaction states with reversible add/remove updates and event-sparse gradients. The key distinction is not that the arithmetic cannot be emulated, but that the primitive’s computation graph is not the explicit \(O(|S|^R)\) interaction graph and not the first-order sparse linear graph. It survives the audit if implemented as a fused accumulator with \(E^{(r)}\) states and custom backward; it fails if implemented by materializing pairs/triples.

## `primitive_rule_generated_sparse_scatter`

Devil’s advocate: this is just GAT/MPNN with a legal-move `edge_index`, or attention with a dynamic mask. That objection is correct if the legal graph is built outside the operator and passed as a tensor; then it is an input encoding plus GNN. The proposal survives only if the primitive owns the rule-generated topology: \((z,\tau)\rightarrow E\rightarrow\operatorname{scatter}\) is one fused operator, and gradients do not flow through a mask tensor. The novelty claim should be stated as “rule-generated sparse scatter with bounded-change incremental graph update,” not as “new attention.”

---

# What I cut

1. **Piece-conditioned legal-move attention.** Rejected because it reduces to `scaled_dot_product_attention(Q,K,V,attn_mask=legal_mask)`, which PyTorch already defines around query/key/value tensors and optional masks.

2. **KAN-style spline evaluator for chess pieces.** Rejected because KAN already proposes learnable functions on edges instead of scalar weights; using it for chess is an application, not a new primitive.

3. **MoE by piece type or phase.** Rejected because data-routed experts are already an established primitive family, with Switch Transformer showing sparse expert selection at constant per-token compute.

4. **Titans-like test-time memory for chess search.** Rejected as a likely training/inference-state trick rather than a clean operator for a scout model; Titans already frames neural long-term memory updated at test time as the core contribution.

5. **A new tactical activation function.** Rejected because it would almost certainly be “Swish/GELU but with a different curve,” which the prompt explicitly rules out.

---

# Literature Grounding Notes

Recent primitive-level work mostly falls into four buckets.

First, **selective recurrence/state-space primitives**: Mamba introduced selective state-space modeling with input-dependent propagation and hardware-aware scan; Mamba-2/SSD connected attention and SSMs through structured semiseparable matrices and reports a 2–8× faster core layer while remaining competitive in language modeling.

Second, **memory/gating primitives**: xLSTM revisits LSTM-style recurrence with exponential gating and scalar/matrix memory structures; Titans proposes neural long-term memory that learns to memorize at test time while retaining fast inference.

Third, **edge/function primitives**: KAN replaces scalar weights with learnable univariate functions on edges; this is relevant as calibration but too close to reuse as a chess proposal.

Fourth, **symmetry/dynamic-graph primitives**: group-equivariant CNNs formalized group convolutions and weight sharing beyond translation, while MPNNs formalized message passing over input graphs. These are important overlap checks: chess proposals that merely feed a legal graph to an MPNN or rotate/reflect a CNN do not clear the novelty bar.

---

# Bibliography

- PyTorch `scaled_dot_product_attention` documentation: <https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>
- PyTorch `MultiheadAttention` documentation: <https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html>
- PyTorch `Conv2d` documentation: <https://docs.pytorch.org/docs/main/generated/torch.nn.modules.conv.Conv2d.html>
- PyTorch `LayerNorm` documentation: <https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html>
- Stockfish NNUE documentation: <https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html>
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces: <https://openreview.net/forum?id=tEYskw1VY2>
- Transformers are SSMs / Mamba-2 / Structured State Space Duality, ICML 2024: <https://proceedings.mlr.press/v235/dao24a.html>
- xLSTM: Extended Long Short-Term Memory, NeurIPS 2024 spotlight listing: <https://apointa.github.io/publication/2024-xlstm.html>
- Titans: Learning to Memorize at Test Time, Google Research, 2025: <https://research.google/pubs/titans-learning-to-memorize-at-test-time/>
- KAN: Kolmogorov-Arnold Networks: <https://huggingface.co/papers/2404.19756>
- Group Equivariant Convolutional Networks, ICML 2016: <https://proceedings.mlr.press/v48/cohenc16.html>
- Neural Message Passing for Quantum Chemistry / MPNN: <https://huggingface.co/papers/1704.01212>
- Switch Transformers: sparse Mixture-of-Experts routing at scale: <https://jmlr.org/papers/v23/21-0998.html>
- SoftSort: differentiable relaxation of argsort: <https://proceedings.mlr.press/v119/prillo20a>
