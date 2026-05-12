# Deep Research Results: New Neural Primitives for Chess Evaluation

## Literature calibration used

Recent work raises the bar for what counts as a primitive. Mamba/S6 made SSM parameters input-conditioned and paired that with a hardware-aware scan; the authors report linear scaling and fast inference relative to Transformers. ([arxiv.org](https://arxiv.org/abs/2312.00752)) Mamba-2/SSD connected SSMs and attention through structured semiseparable matrices and reports a 2–8× faster core layer than Mamba. ([arxiv.org](https://arxiv.org/abs/2405.21060)) xLSTM introduced exponential gates plus scalar and matrix memories, so “just another recurrent block” is not enough to count as new. ([arxiv.org](https://arxiv.org/abs/2405.04517)) Titans introduced a neural long-term memory module with fast parallelizable training and fast inference, so memory-as-state is now an active primitive frontier. ([arxiv.org](https://arxiv.org/abs/2501.00663)) Differential Transformer subtracts two softmax attention maps, so any proposal that is merely “attention minus attention” is already occupied. ([arxiv.org](https://arxiv.org/abs/2410.05258)) Differentiable sparse top-k operators also already exist and have been used as routers, so learned top-k move selection is not novel by itself. ([proceedings.mlr.press](https://proceedings.mlr.press/v202/sander23a.html))

PyTorch already exposes `MultiheadAttention`, `Conv2d`, `LayerNorm`, and `EmbeddingBag`; the proposals below are written to avoid simply renaming those stock operators. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html)) Conditional computation and dynamic graph learning are active areas, but the proposals here focus on primitive-level operators, not whole architectures. ([journals.sagepub.com](https://journals.sagepub.com/doi/10.3233/IA-240035))

## Ranked shortlist

| Rank | Primitive | Novelty plausibility | RTX 3070 demonstrability | Inference-speed advantage | Generalises beyond chess | Main risk |
|---:|---|---|---|---|---|---|
| 1 | `primitive_1_occlusion_semiring_scan` | High-medium | High | High | High | Could be seen as a specialised selective scan |
| 2 | `primitive_2_delta_bilinear_accumulator` | Medium | Very high | Very high | High | Could be seen as cached polynomial pooling |
| 3 | `primitive_3_legal_hyperedge_contraction` | High | Medium | High | Medium-high | Legal generator may be called “preprocessing” |
| 4 | `primitive_4_tropical_threat_scan` | Medium-high | Medium | Medium-high | High | Differentiable shortest-path overlap |
| 5 | `primitive_5_chess_orbit_linear` | Low-medium | High | Medium | Medium | Group-equivariant linear maps already exist |

## Top-2 self-audit

`primitive_1_occlusion_semiring_scan`: Devil’s advocate says this is Mamba with scalar transition \(A_t=1-o_t\). That is partially true: the recurrence overlaps with selective scan. I keep it only with the narrower claim that the primitive is a ray-indexed occlusion semiring with segment-tree incremental update and visibility semantics, not a new general SSM family.

`primitive_2_delta_bilinear_accumulator`: Devil’s advocate says the stateless forward is `EmbeddingBag` plus low-rank polynomial pooling. That objection is strong. I keep it only if the primitive API includes persistent accumulator state, exact insert/delete updates, and a custom backward over delta events; a stateless implementation should be rejected.

## Five proposed primitives

### primitive_1_occlusion_semiring_scan

**Name:** Occlusion-Semiring Ray Scan

**One-line claim:** A differentiable visibility scan that aggregates along rays while blockers multiplicatively cancel downstream information.

**Mathematical signature:**
\(f:\mathbb{R}^{B\times R\times L\times d}\times[0,1]^{B\times R\times L}\rightarrow\mathbb{R}^{B\times R\times L\times d}\).
For ray \(r\), position \(t\):
\[
h_{b,r,t}=(1-o_{b,r,t+1})h_{b,r,t+1}+Vx_{b,r,t+1},\quad h_{b,r,L}=0,
\]
\[
y_{b,r,t}=W_0x_{b,r,t}+Uh_{b,r,t}.
\]
Gradients are standard through \(x\) and continuous \(o\); if \(o\) is discrete occupancy, gradients are only through \(x\).

**Why this does not decompose into existing PyTorch ops:**
Closest stock comparisons are `Conv2d`, masked attention, and selective scan. `Conv2d` has fixed spatial support, while this operator’s support is a product of intervening transmittances. Masked attention can hide tokens, but it does not express ordered “first blocker cancels everything behind it” without materialising an \(L\times L\) mask. Mamba-like scan overlap exists; the new claim is the occlusion semiring and incremental ray cache, not another SSM.

**Chess-specific motivation:**
Sliding pieces are ray operators. Rooks, bishops, queens, pins, discovered attacks, x-rays, and king safety all depend on blockers between two squares, not just endpoint distance. This primitive gives that structure directly instead of asking a small conv net to learn it from data.

**Generalisation beyond chess:**
Useful for grid visibility, ray tracing surrogates, robotics occupancy maps, board games, tactical simulators, and scene graphs with line-of-sight constraints.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(BRLD^2)\) or \(O(BRLD)\) with diagonal/channelwise \(U,V\), vs dense attention \(O(B(RL)^2D)\)
- Backward: \(O(BRLD^2)\)
- Incremental update on a bounded-change input: \(O(D\log L)\) with segment-tree ray states; chess has \(L\le 8\), effectively \(O(D)\)

**Scout-scale falsification test:**
Replace one \(3\times3\) convolution in the i193 conv-only parent with this primitive over the 8 queen rays per square. Baseline: same-parameter i193 layer. Metric: matched-recall CRTK class-1 near-puzzle false-positive rate, plus eval throughput. Works if near-puzzle FP falls by at least 5% at ≤1.15× latency; fails if only aggregate PR AUC improves or latency exceeds 1.25×.

**Failure mode catalogue:**
- Hidden rebrand: reviewer may call it a scalar selective SSM over short rays.
- Numerical instability: products of \((1-o)\) can vanish; clamp \(o\) or use log-transmittance.
- Too slow: duplicate ray representations can waste memory unless rays are packed and fused.

**Status:** proposed

### primitive_2_delta_bilinear_accumulator

**Name:** Delta-Maintained Bilinear Set Accumulator

**One-line claim:** A sparse-set accumulator with exact insert/delete updates for first- and second-order feature interactions.

**Mathematical signature:**
For active IDs \(S_b\subseteq\{1,\dots,U\}\):
\[
f:S_b\mapsto a_b\in\mathbb{R}^{d}.
\]
Parameters \(E\in\mathbb{R}^{U\times d}\), \(P,Q\in\mathbb{R}^{U\times r}\), \(M\in\mathbb{R}^{r\times d}\):
\[
s_P=\sum_{u\in S_b}P_u,\quad s_Q=\sum_{u\in S_b}Q_u,\quad s_{PQ}=\sum_{u\in S_b}P_u\odot Q_u,
\]
\[
a_b=\sum_{u\in S_b}E_u+\frac{1}{2}\big((s_P\odot s_Q)-s_{PQ}\big)M.
\]
Gradients are defined for all active IDs and parameters.

**Why this does not decompose into existing PyTorch ops:**
Statelessly, this is dangerously close to `EmbeddingBag` plus polynomial pooling; that version should not count. The primitive claim is the stateful operator: it owns \((s_P,s_Q,s_{PQ},\sum E)\), exposes `insert/delete/update`, and backpropagates through only touched IDs plus global bilinear summaries. `EmbeddingBag` computes sums or means of bags but has no second-order accumulator state or exact delta update API. ([docs.pytorch.org](https://docs.pytorch.org/docs/2.9/generated/torch.nn.EmbeddingBag.html))

**Chess-specific motivation:**
NNUE’s practical win is not just a linear layer; it is the fact that a move changes a few active atoms. Chess evaluation also needs pair effects: king-near-piece, rook-open-file, bishop-pawn-color, trapped-piece relations. This primitive keeps the HalfKA-style \(O(1)\) update property while adding low-rank interactions.

**Generalisation beyond chess:**
Applies to dynamic recommender baskets, fraud-event streams, compiler facts, market micro-events, and any sparse set where insert/delete changes are small.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(|S|(d+r)+rd)\), vs pairwise interaction \(O(|S|^2d)\) and `EmbeddingBag` \(O(|S|d)\)
- Backward: \(O(|S|(d+r)+rd)\)
- Incremental update on a bounded-change input: \(O(|\Delta|(d+r)+rd)\); with cached output projection, \(O(|\Delta|(d+r))\)

**Scout-scale falsification test:**
Drop into i243 by replacing the first HalfKA linear accumulator with this primitive at matched parameter count. Baseline: HalfKA accumulator. Metric: CRTK class-1 matched-recall FP rate and node-eval latency. Works if FP falls ≥5% with ≤1.10× latency; fails if improvement appears only on easy negatives or if cache maintenance dominates.

**Failure mode catalogue:**
- Hidden rebrand: without the stateful delta API, this is just polynomial pooling.
- Numerical instability: second-order term can scale like \(|S|^2\); use \(1/|S|\) normalisation or bounded \(P,Q\).
- Too slow: large \(r\) or dense \(M\) can erase the NNUE-style advantage.

**Status:** proposed

### primitive_3_legal_hyperedge_contraction

**Name:** Content-Declared Legal Hyperedge Contraction

**One-line claim:** A sparse hypergraph contraction whose hyperedges are generated inside the operator from the current discrete state.

**Mathematical signature:**
\[
f:\mathbb{R}^{B\times n\times d}\times\mathcal{A}^{B\times n}\rightarrow\mathbb{R}^{B\times n\times d}.
\]
For each board \(b\), an internal deterministic relation generator returns role-labelled hyperedges
\[
H_b=\mathrm{Gen}(a_b)=\{h=(v_1,\dots,v_{m_h},\rho_h):m_h\le a\}.
\]
Then
\[
e_h=\phi_{\rho_h}\left(\bigoplus_{j=1}^{m_h}W_{\rho_h,j}x_{b,v_j}\right),
\quad
y_{b,i}=x_{b,i}+\sum_{h\in H_b:i\in h}U_{\rho_h,\mathrm{role}(i,h)}e_h.
\]
Gradients flow through \(x,W,U,\phi\), not through the discrete generator.

**Why this does not decompose into existing PyTorch ops:**
It is not masked attention: the connectivity is not a supplied \(n\times n\) mask, and the edge object can have arity greater than two. It is not ordinary message passing: the graph is generated inside the primitive from content, including ray blockers and king-safety constraints. PyTorch sparse tensors store nonzeros, but they do not define a neural operator that generates and contracts variable-arity legal hyperedges.

**Chess-specific motivation:**
The legal-move graph changes every position. Many hard false positives are positions that look tactically plausible but have one legality, pin, or king-safety constraint that kills the tactic. A primitive that contracts exactly over legal action hyperedges targets that failure mode directly.

**Generalisation beyond chess:**
Plausible for chemical reaction hypergraphs, program-analysis dataflow, theorem-proving states, dynamic scene interactions, and logistics systems with state-dependent feasible actions.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B|H|ad^2)\), vs dense attention \(O(Bn^2d)\)
- Backward: \(O(B|H|ad^2)\)
- Incremental update on a bounded-change input: \(O(|\Delta H|ad^2)\); on an 8×8 board, changed legal rays are bounded by constants

**Scout-scale falsification test:**
Insert one contraction layer into i193 in place of one mid-depth \(3\times3\) conv, using pseudo-legal and legal variants as an ablation. Baseline: same-parameter conv. Metric: matched-recall near-puzzle FP rate. Works if the legal variant beats both conv and pseudo-legal by ≥5% FP reduction under <2 GPU-hours; fails if it only learns mobility count.

**Failure mode catalogue:**
- Hidden rebrand: reviewer may say the generator is preprocessing plus a hypergraph neural network.
- Numerical instability: variable hyperedge counts can cause activation scale drift; divide by \(\sqrt{\deg(i)+1}\).
- Too slow: CPU move generation inside the kernel would kill throughput; needs fused GPU or cached legal deltas.

**Status:** proposed

### primitive_4_tropical_threat_scan

**Name:** Tropical Threat-Distance Scan

**One-line claim:** A differentiable min-plus graph scan that routes gradient through shortest tactical paths, not average attention paths.

**Mathematical signature:**
\[
f:\mathbb{R}^{B\times n\times d}\times E_B\times\mathbb{R}^{B\times n}\rightarrow\mathbb{R}^{B\times n\times d}.
\]
For edge \((i,j)\in E_b\), define \(c_{ij}=\mathrm{softplus}(w^\top[x_i,x_j])\). With seed costs \(D_i^{(0)}=s_i\):
\[
D_i^{(\ell)}=\operatorname{softmin}_\tau\left(D_i^{(\ell-1)},\{c_{ij}+D_j^{(\ell-1)}:(i,j)\in E_b\}\right),
\]
\[
\alpha_{ij}^{(\ell)}=\mathrm{softmax}\left(-(c_{ij}+D_j^{(\ell-1)})/\tau\right),
\quad
y_i=R x_i+T D_i^{(L)}+\sum_{\ell,j}\alpha_{ij}^{(\ell)}Vx_j.
\]
Gradients are well-defined for \(\tau>0\).

**Why this does not decompose into existing PyTorch ops:**
Attention is a sum-product operator: softmax weights average values. This is a min-plus/tropical operator: path composition adds costs, and aggregation takes a soft minimum. It can be prototyped with `scatter` and `logsumexp`, but PyTorch has no `nn` primitive for sparse semiring Bellman scans with custom path-backpointer gradients.

**Chess-specific motivation:**
Threats are often shortest constrained paths: knight fork distance, king escape distance, rook file penetration, mating net closure, and defender arrival. Near-puzzle negatives often differ from true puzzles by one defender tempo or one blocked path. A tropical operator encodes “can this threat arrive before that defense?” more directly than dense averaging.

**Generalisation beyond chess:**
Useful for routing, differentiable planning, robot navigation, program dependence, circuit timing, and molecular graph paths.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(BL|E|d)\), vs attention \(O(Bn^2d)\)
- Backward: \(O(BL|E|d)\)
- Incremental update on a bounded-change input: \(O(L|\Delta E|d)\), or \(O((|\Delta E|+A)\log n\,d)\) with dynamic shortest-path caches

**Scout-scale falsification test:**
Use the primitive as a single replacement for one i193 conv layer, with \(E\) equal to attack/pseudo-legal adjacency computed from board state. Baseline: same-edge ordinary mean message passing. Metric: matched-recall CRTK class-1 FP rate. Works if tropical scan beats mean aggregation by ≥5% FP reduction; fails if \(\tau\) tuning dominates or latency exceeds 1.3×.

**Failure mode catalogue:**
- Hidden rebrand: reviewer may call it differentiable shortest path, not a neural primitive.
- Numerical instability: small \(\tau\) causes winner-take-all gradients; large \(\tau\) collapses to averaging.
- Too slow: repeated \(L\)-step relaxation can be worse than attention when \(E\) is dense.

**Status:** proposed

### primitive_5_chess_orbit_linear

**Name:** Chess-Group Orbit Linear

**One-line claim:** A finite-group equivariant linear operator tied over chess-specific color, square, and role orbits.

**Mathematical signature:**
Let \(\Omega\) be token indices and \(G_{\text{chess}}\) act on \(\Omega\), including file mirror and color-swap/rank-flip involution. For \(X\in\mathbb{R}^{B\times|\Omega|\times d_{\text{in}}}\):
\[
f:\mathbb{R}^{B\times|\Omega|\times d_{\text{in}}}\rightarrow\mathbb{R}^{B\times|\Omega|\times d_{\text{out}}},
\]
\[
y_{b,\omega}=\sum_{\nu\in\Omega}W_{[\omega,\nu]}x_{b,\nu},
\quad
W_{[g\omega,g\nu]}=\rho_{\text{out}}(g)W_{[\omega,\nu]}\rho_{\text{in}}(g)^{-1}.
\]
Only one parameter block is stored per orbit of the pair action \(G_{\text{chess}}\curvearrowright\Omega\times\Omega\).

**Why this does not decompose into existing PyTorch ops:**
It is not `Linear`, which has free weights, and not `Conv2d`, which only ties by translation-like local offsets. It is an arbitrary finite-group orbit contraction with representation-aware weight tying. Related group-equivariant and orbit-equivariant work exists, so the honest novelty claim is “underexplored primitive for chess,” not “first equivariant layer ever.” ([proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2024/hash/15c4f200212c856eaa023c1d4437eee6-Abstract-Conference.html))

**Chess-specific motivation:**
Chess has structure beyond dihedral board symmetry: color swap with rank flip, side-to-move sign, friendly/enemy role exchange, and asymmetric pawn direction. This primitive makes those equivalences exact rather than hoping augmentation teaches them from 173k positions.

**Generalisation beyond chess:**
Useful for finite-state games, chemistry with discrete automorphisms, recommender systems with exchangeable roles, and any tensor indexed by a known finite group action.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B|\Omega|^2d_{\text{in}}d_{\text{out}})\) dense, or \(O(B|\mathcal{O}|d_{\text{in}}d_{\text{out}})\) with sparse orbit support; closest `Linear` has same dense FLOPs but more parameters
- Backward: same order as forward
- Incremental update on a bounded-change input: \(O(|\Delta\Omega||\Omega/G|d_{\text{in}}d_{\text{out}})\) with cached orbit sums

**Scout-scale falsification test:**
Replace one pointwise \(1\times1\) channel mixer in i193 with this orbit-linear operator over square/color/piece-role indices. Baseline: untied linear mixer and ordinary file-mirror augmentation. Metric: near-puzzle FP at matched recall plus parameter-normalised PR AUC. Works if orbit tying improves FP ≥3% without reducing throughput; fails if augmentation matches it.

**Failure mode catalogue:**
- Hidden rebrand: may be judged a special case of group convolution.
- Numerical instability: orbit sizes differ; unnormalised large orbits can dominate.
- Too slow: dense orbit matrices are wasteful unless sparse orbit support is enforced.

**Status:** proposed

## What I cut

1. **Piece-conditioned legal-move attention.** Rejected because it is masked attention once the legal-move mask is materialised; PyTorch `MultiheadAttention` already defines the relevant attention computation. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html))

2. **Incremental convolution cache.** Rejected because it is `Conv2d` plus cached output invalidation. It may be useful engineering, but it is not a new primitive. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.conv.Conv2d.html))

3. **OrbitNorm / chess symmetry normalisation.** Rejected because it collapses into `LayerNorm` or `GroupNorm` over a hand-chosen axis/orbit. The group choice is interesting; the operator is not. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html))

4. **Learned top-k tactical router.** Rejected because differentiable sparse top-k operators already exist, including use as routers. A chess-specific router would be an application, not a primitive. ([proceedings.mlr.press](https://proceedings.mlr.press/v202/sander23a.html))

5. **New activation for checks or threats.** Rejected because polynomial/swish-like scalar activations do not change connectivity, complexity class, or gradient flow enough to meet the bar.

## Bibliography

- Gu and Dao, **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**. ([arxiv.org](https://arxiv.org/abs/2312.00752))
- Dao and Gu, **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**. ([arxiv.org](https://arxiv.org/abs/2405.21060))
- Beck et al., **xLSTM: Extended Long Short-Term Memory**. ([arxiv.org](https://arxiv.org/abs/2405.04517))
- Behrouz, Zhong, and Mirrokni, **Titans: Learning to Memorize at Test Time**. ([arxiv.org](https://arxiv.org/abs/2501.00663))
- Ye et al., **Differential Transformer**. ([arxiv.org](https://arxiv.org/abs/2410.05258))
- Sander et al., **Fast, Differentiable and Sparse Top-k: a Convex Analysis Perspective**. ([proceedings.mlr.press](https://proceedings.mlr.press/v202/sander23a.html))
- Scardapane et al., **Conditional computation in neural networks: Principles and research trends**. ([journals.sagepub.com](https://journals.sagepub.com/doi/10.3233/IA-240035))
- Zheng, Yi, and Wei, **A survey of dynamic graph neural networks**. ([link.springer.com](https://link.springer.com/article/10.1007/s11704-024-3853-2))
- Morris, Grau, and Horrocks, **Orbit-Equivariant Graph Neural Networks**. ([proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2024/hash/15c4f200212c856eaa023c1d4437eee6-Abstract-Conference.html))
- Liu et al., **Clifford Group Equivariant Simplicial Message Passing Networks**. ([proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2024/hash/83ae75c127e2a3ea3315379020f8c19f-Abstract-Conference.html))
- PyTorch documentation: **MultiheadAttention**, **Conv2d**, **LayerNorm**, and **EmbeddingBag**. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html))
