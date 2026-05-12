# Deep Research Primitive Results: Chess Evaluation Neural Operators

## Literature calibration and ranking

Recent primitive-level deep-learning work has mostly moved in three directions: input-conditioned recurrence/scans, sparse or linearized attention, and symmetry-constrained operators. Mamba introduced a selective state-space mechanism with input-dependent parameters and linear-time sequence processing; Mamba-2/SSD connected state-space models and attention through structured semiseparable matrices; Gated Linear Attention and Log-Linear Attention continue the same trend toward custom scan/attention primitives rather than ordinary PyTorch module compositions. See bibliography entries [1], [2], [3], and [4].

The rejected baseline idea is “just use attention with a chess mask.” Content-dependent sparse attention already exists in Routing Transformer-style work, and group equivariant convolutions already exist for finite transformation groups; the proposals below therefore only count when connectivity, scan state, or equivariant tying is generated inside the primitive rather than supplied as an external mask or architecture convention. See [6] and [7].

| Rank | Primitive | Novelty | 3070 demonstrability | Speed upside | Generality |
|---:|---|---|---|---|---|
| 1 | Ray-Occlusion Semiring Scan | High | High | High | Medium-high |
| 2 | Legal-Move Sparse Dispatch | High | Medium-high | High | Medium |
| 3 | Delta-Factorized Pair Accumulator | Medium-high | High | Very high | High |
| 4 | Chess-Group Orbit Contraction | Medium | High | Medium | Medium-high |
| 5 | Soft Exchange Semiring Pool | High | Medium | Medium | Low-medium |

## Self-audit on the top two

**Ray-Occlusion Semiring Scan:** The strongest objection is that it can be emulated with `gather`, `cumprod`, and `sum`. That emulation is not the proposed primitive: the kept version is a fused semiring scan with prefix transmittance, blocker-conditioned connectivity, and an incremental update API. It is closer to a new scan operator than to a layer made from ordinary convolution.

**Legal-Move Sparse Dispatch:** The strongest objection is that it is merely Graph Attention with a legal-move edge list. That objection is valid if the edge list is precomputed and passed in. The kept version computes the rule-conditioned sparse graph inside the operator and performs fused segment normalization on generated edges, so the connectivity is a first-class part of the primitive rather than an input encoding.

## Proposals

### primitive_ray_occlusion_scan

**Name:** Ray-Occlusion Semiring Scan

**One-line claim:** A fused blocker-aware scan that propagates information along chess rays only until occupancy stops transmission.

**Mathematical signature:**
\(f_\theta:\mathbb{R}^{[B,8,8,d]}\times[0,1]^{[B,8,8]}\rightarrow\mathbb{R}^{[B,8,8,d]}\).  
For square \(s\), direction \(\delta\in D_8\), and ray step \(k\):  
\[
T_{b,s,\delta,k}=\prod_{r<k}(1-O_{b,s+r\delta}),\quad
Y_{b,s}=\sum_{\delta\in D_8}\sum_{k\le K(s,\delta)}
T_{b,s,\delta,k}\,A_\delta X_{b,s+k\delta}.
\]
Gradients are defined through the product; implementation should use log-domain prefix products for stability.

**Why this does not decompose into existing PyTorch ops:**
This is not `Conv2d`: the receptive field is input-dependent because blockers terminate propagation. It is not attention either, because weights are prefix products on an occlusion semiring rather than softmax-normalized dot products. A slow emulation with `cumprod` and gathers exists, but the primitive claim is the fused scan state, early termination, and incremental-update semantics, analogous to why selective scan is treated as a primitive in Mamba-style models [1].

**Chess-specific motivation:**
Sliding-piece tactics are ray problems: pins, skewers, x-rays, discovered attacks, and overloaded defenders all depend on the first blocker on a line. Convs need depth to discover that a rook attacks through an opened file; full attention sees too much irrelevant context. This operator hard-codes the connectivity law, not the evaluation.

**Generalisation beyond chess:**
Useful for visibility, line-of-sight, grid robotics, differentiable ray casting, packet routing with blockers, and sparse physical simulations.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(BRd)\), \(R=\sum_{s,\delta}K(s,\delta)\), vs full attention \(O(Bn^2d)\) or large-kernel conv \(O(Bnkd)\)
- Backward: \(O(BRd)\)
- Incremental update on a bounded-change input: \(O(\sqrt n\,d)\) for affected rays, constant on an 8×8 board; \(O(\log n\,d)\) with cached segment products

**Scout-scale falsification test:**
Drop one instance into i193 by replacing a same-width spatial mixing conv. Baseline: the original i193 block and a fixed eight-direction depthwise ray conv with no blockers. Train 173k positions × 12 epochs, single seed. Primitive works if matched-recall CRTK class-1 near-puzzle false-positive rate drops by at least 5% relative to the stronger baseline while latency is ≤1.2× baseline. It fails if it only improves aggregate PR AUC or costs >1.3× latency.

**Failure mode catalogue:**
- Hidden rebrand objection: “this is just cumulative product plus convolution”; answer only holds if no fused scan/incremental API is implemented.
- Numerical instability: long products can vanish; use log-domain prefix sums and clamp \(O\in[\epsilon,1-\epsilon]\).
- Speed risk: Python ray loops will be useless; this needs a fused CUDA/Triton kernel or compact precomputed ray tables.

**Status:** proposed

### primitive_legal_move_dispatch

**Name:** Legal-Move Sparse Dispatch

**One-line claim:** A segment-softmax message operator whose sparse edges are generated inside the op by the current board state.

**Mathematical signature:**
\(f_\theta:\mathbb{R}^{[B,64,d]}\times\mathcal{C}^{[B,64]}\rightarrow\mathbb{R}^{[B,64,d]}\), where \(\mathcal{C}\) is the discrete occupant alphabet already present in the board input.  
Inside the op:
\[
E_b=\mathrm{MoveGen}(C_b)=\{(u,v,r)\},
\]
\[
s_e=a_r^\top \sigma(W_s[x_u,x_v]),\quad
m_e=W_r x_u,
\]
\[
Y_{b,v}=\sum_{e=(u,v,r)\in E_b}
\frac{\exp(s_e)}{\sum_{e'=(u',v,r')\in E_b}\exp(s_{e'})}\,m_e .
\]
Gradients flow through \(X\) and \(\theta\), not through the discrete legality generator.

**Why this does not decompose into existing PyTorch ops:**
Standard `MultiheadAttention` accepts a fixed or externally supplied mask; Graph Attention accepts an externally supplied edge index. Here the sparse edge set is generated inside the primitive from piece occupancy, blockers, side, and move rules, then consumed by fused segment softmax. Routing Transformer proved content-dependent sparse attention is a real direction, but this primitive uses a rule-derived dynamic graph rather than learned clustering [6].

**Chess-specific motivation:**
Chess’s most meaningful sparse graph is not grid adjacency; it is legal or pseudo-legal move connectivity, and that graph changes after every move. Near-puzzle false positives often differ from positives by one legal defender, interposition, or capture resource. This primitive forces computation onto those changing tactical edges.

**Generalisation beyond chess:**
Dynamic rule graphs in games, simulators, program analysis, molecular reaction systems, symbolic planners, and event-driven scene graphs.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B|E|d)\) vs full attention \(O(B64^2d)\)
- Backward: \(O(B|E|d)\)
- Incremental update on a bounded-change input: \(O(|\Delta E|d)\); on chess, changed edges are local piece moves plus affected blocker rays

**Scout-scale falsification test:**
Drop one primitive after the first projection in i193; keep width and parameter count close by reducing adjacent channel mixing. Baseline: same harness with full square attention and with a GAT layer using precomputed legal edges. Works if CRTK class-1 matched-recall FP rate improves by ≥7% over both baselines and node-eval latency is below full attention. Fails if gains appear only on easy negatives.

**Failure mode catalogue:**
- Hidden rebrand objection: if `MoveGen` is outside the op, it is just GAT with an input encoding.
- Numerical instability: segment softmax over tiny degree sets can become overconfident; subtract per-segment max.
- Speed risk: legal generation on GPU may dominate unless edge generation and message passing are fused.

**Status:** proposed

### primitive_delta_pair_accumulator

**Name:** Delta-Factorized Pair Accumulator

**One-line claim:** A sparse-set operator that updates first- and second-order feature interactions from only the changed events.

**Mathematical signature:**
\(f_\theta:\mathcal{S}_t,\Delta_t^+,\Delta_t^-\rightarrow\mathbb{R}^{[B,d]}\), implemented statefully.  
For active sparse ids \(S_t\subseteq\{1,\dots,V\}\), parameters \(U,P,Q\in\mathbb{R}^{[V,d]}\):
\[
A_t=\sum_{i\in S_t}U_i,\quad
P_t=\sum_{i\in S_t}P_i,\quad
Q_t=\sum_{i\in S_t}Q_i,\quad
D_t=\sum_{i\in S_t}P_i\odot Q_i,
\]
\[
Y_t=A_t+\frac{1}{2}(P_t\odot Q_t-D_t).
\]
For updates, add rows for \(\Delta^+\) and subtract rows for \(\Delta^-\) from all four sufficient statistics.

**Why this does not decompose into existing PyTorch ops:**
A first-order version is too close to `EmbeddingBag` and NNUE’s affine accumulator, so it should be rejected. The retained primitive exposes a second-order sparse-set sufficient statistic with persistent delta updates and sparse row gradients. Factorization-machine algebra is related, so the novelty claim is not “new math”; it is “primitive-ized, delta-updatable second-order sparse interaction.”

**Chess-specific motivation:**
HalfKA’s speed comes from sparse accumulator updates; Stockfish’s NNUE documentation emphasizes sparse inputs, small input changes between positions, and efficient accumulator updates [8]. Chess evaluation also needs piece-pair interactions: attacker-defender pairs, king-piece relations, batteries, and overloaded pieces. This operator chases pairwise expressivity without enumerating all piece pairs every node.

**Generalisation beyond chess:**
Sparse recommender systems, event co-occurrence models, dynamic graphs, market baskets, biological mutation sets, and online simulation states.

**Complexity (forward, backward, incremental-update):**
- Forward: refresh \(O(B|S|d)\), state read \(O(Bd)\), vs explicit pair layer \(O(B|S|^2d)\)
- Backward: \(O(B|\Delta|d)\) for incremental use; \(O(B|S|d)\) for full refresh
- Incremental update on a bounded-change input: \(O(B|\Delta|d)=O(Bd)\)

**Scout-scale falsification test:**
Use i243 HalfKA+dual-stream as harness, replacing only the first sparse feature transformer with this primitive and keeping downstream layers unchanged. Baseline: first-order HalfKA accumulator with matched output width. Works if evaluation throughput is within 10% of first-order HalfKA while CRTK class-1 matched-recall FP rate drops ≥5%. Fails if pairwise term improves training loss but not near-puzzle discrimination.

**Failure mode catalogue:**
- Hidden rebrand objection: resembles factorization machines plus `EmbeddingBag`; novelty depends on delta-state API and sparse-gradient implementation.
- Numerical instability: \(P_t\odot Q_t\) can grow with piece count; normalize by \(|S_t|\) or use RMS scaling.
- Speed risk: four accumulators may hurt cache locality; low precision and row-contiguous storage are required.

**Status:** proposed

### primitive_chess_orbit_contraction

**Name:** Chess-Group Orbit Contraction

**One-line claim:** A finite-group equivariant contraction over squares, colors, side-to-move, and typed piece channels.

**Mathematical signature:**
Let \(\Omega\) be square–piece-channel indices and \(G\) a finite chess-rule automorphism group with representations \(\rho_{\text{in}},\rho_{\text{out}}\).  
\(f_\theta:\mathbb{R}^{[B,|\Omega|,d_{\text{in}}]}\rightarrow\mathbb{R}^{[B,|\Omega|,d_{\text{out}}]}\).  
The kernel is projected into the equivariant subspace:
\[
W_G=\frac{1}{|G|}\sum_{g\in G}\rho_{\text{out}}(g)\,W\,\rho_{\text{in}}(g)^{-1},
\quad
Y=W_GX .
\]
Equivalently, learn one parameter per orbit of \(\Omega_{\text{out}}\times\Omega_{\text{in}}\) under \(G\).

**Why this does not decompose into existing PyTorch ops:**
Ordinary `Linear` has no equivariance constraint; ordinary `Conv2d` only handles translation-style sharing. Group equivariant convolutions exist and are the closest prior work, so this should be framed as an underexplored finite typed-group primitive for chess, not as a wholly new equivariance theory [7]. The new structural signature is simultaneous action over board geometry, color swap, side-to-move sign, and piece-type channels.

**Chess-specific motivation:**
Chess has exact symmetries beyond plain board reflection, but careless symmetry use is wrong because pawns, castling, side-to-move, and color orientation break naive dihedral invariance. Orbit contraction lets the primitive encode only valid automorphisms and anti-symmetries, reducing sample demand at scout scale.

**Generalisation beyond chess:**
Typed-token domains with finite automorphism groups: board games, molecules with atom-type symmetries, knowledge graphs, cellular automata, and program graphs.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B|\mathcal{O}|d_{\text{in}}d_{\text{out}})\) parameter-orbit contractions vs untied \(O(B|\Omega|^2d_{\text{in}}d_{\text{out}})\)
- Backward: same order plus orbit-gradient accumulation
- Incremental update on a bounded-change input: \(O(|\mathrm{orbit}(\Delta)|d_{\text{in}}d_{\text{out}})\) for sparse use; otherwise not applicable

**Scout-scale falsification test:**
Replace one channel-mixing affine in i193 or i243 with orbit contraction; do not change inputs or losses. Baseline: untied affine with matched parameter count. Works if CRTK class-1 matched-recall FP rate improves ≥5% or equals baseline with materially lower parameter count and no latency regression. Fails if tying reduces tactical recall.

**Failure mode catalogue:**
- Hidden rebrand objection: may be “just group convolution”; claim must be limited to typed chess-group contraction.
- Correctness risk: using an invalid group action silently bakes in false invariance.
- Speed risk: expanding tied kernels every batch is slow; compile orbit index tables once.

**Status:** proposed

### primitive_soft_exchange_semiring

**Name:** Soft Exchange Semiring Pool

**One-line claim:** A differentiable alternating-capture reducer for local attacker lists, aimed at tactical hard negatives.

**Mathematical signature:**
\(f_\theta:\mathbb{R}^{[B,64,d]}\times\mathcal{C}^{[B,64]}\rightarrow\mathbb{R}^{[B,64,h]}\).  
For each target square \(q\), internally form attacker lists \(L_{q,w},L_{q,b}\) from board occupancy. Let attacker scalar costs be
\[
a_i=\mathrm{softplus}(w_a^\top x_i+\beta_{\mathrm{piece}(i)}).
\]
Using fixed piece-class ordering within each side, define an alternating relaxed exchange recurrence:
\[
r_0=v_q,\quad
r_{t+1}=a_{i_t}-\tau\log(1+\exp(r_t/\tau)),
\]
where \(i_t\) is the next attacker for the side to move at ply \(t\). Output \(Y_q=\phi([r_0,\dots,r_T])\). Gradients flow through \(x_i,w_a,\phi\); legal attacker extraction is discrete.

**Why this does not decompose into existing PyTorch ops:**
This is not `MaxPool`, attention, or a normal RNN. The reducer is an alternating soft-minimax semiring over dynamically generated attacker sets, with capture-side alternation inside the primitive. It can be approximated with Python loops and `softplus`, but no PyTorch primitive exposes “segment alternating minimax over rule-generated sets.”

**Chess-specific motivation:**
Many near-puzzle false positives are exchange-evaluation failures: a capture looks winning unless the defender sequence is counted correctly. This primitive gives the network a local differentiable analogue of static-exchange reasoning without feeding Stockfish scores, PVs, or node metadata. Evidence that strong chess networks learn look-ahead-like mechanisms makes a primitive for local tactical reduction worth testing, but no result is claimed here [10].

**Generalisation beyond chess:**
Mostly chess-specific, but plausible for adversarial resource capture, auction cascades, security games, and turn-based simulators with local contest sequences.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B\sum_q T_qd)\), \(T_q\) bounded by attacker-list length, vs attention \(O(B64^2d)\)
- Backward: \(O(B\sum_q T_qd)\)
- Incremental update on a bounded-change input: \(O(\sum_{q\in Affected}T_qd)\), where affected squares are changed attack targets

**Scout-scale falsification test:**
Drop one Soft Exchange Semiring Pool into i193 immediately before the classifier head as an auxiliary square reducer, with parameter count matched by shrinking the previous affine. Baseline: same shape using ordinary max/mean pooling over attacker features. Works if CRTK class-1 matched-recall FP rate drops ≥8% while aggregate recall stays fixed. Fails if it only improves positions with obvious material captures.

**Failure mode catalogue:**
- Hidden rebrand objection: resembles soft alpha-beta or differentiable dynamic programming; claim is primitive packaging, not new game theory.
- Numerical instability: small \(\tau\) makes gradients spiky; annealing is a training trick, so use a fixed safe \(\tau\).
- Speed risk: attacker-list construction may duplicate Legal-Move Sparse Dispatch; fuse them if both are used.

**Status:** proposed

## What I cut

1. **First-order Delta Affine Accumulator.** It is basically `EmbeddingBag` plus add/subtract state, and NNUE already embodies the idea. Stockfish documentation makes the speed argument clear, but as a new primitive it fails the hidden-rebrand test unless extended to higher-order sufficient statistics [8].

2. **Legal-move attention mask.** A mask over legal moves is still standard attention: \((QK^\top+\mathrm{mask}).\mathrm{softmax}V\). It only becomes proposal-worthy when edge generation and sparse segment normalization are inside the primitive.

3. **Mamba over ranks/files/diagonals.** Mamba and Mamba-2 are genuine primitives, but applying them to chess scan orders is an architecture choice, not a new operator [1], [2].

4. **KAN-style piece interaction layer.** KANs replace linear weights with learned edge functions, which is real recent primitive-level work, but a chess KAN layer would likely be slow and data-hungry at scout scale [5].

5. **Differential legal attention.** Differential Transformer subtracts two attention maps to cancel noise, but a chess version would still be a composition of existing attention maps unless the legal graph or semiring reducer changes the primitive itself [9].

## Bibliography

1. Albert Gu and Tri Dao, **“Mamba: Linear-Time Sequence Modeling with Selective State Spaces.”** arXiv:2312.00752. <https://arxiv.org/abs/2312.00752>

2. Tri Dao and Albert Gu, **“Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.”** arXiv:2405.21060. <https://arxiv.org/abs/2405.21060>

3. Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim, **“Gated Linear Attention Transformers with Hardware-Efficient Training.”** arXiv:2312.06635. <https://arxiv.org/abs/2312.06635>

4. Han Guo, Songlin Yang, Tarushii Goel, Eric P. Xing, Tri Dao, and Yoon Kim, **“Log-Linear Attention.”** arXiv:2506.04761. <https://arxiv.org/abs/2506.04761>

5. Ziming Liu et al., **“KAN: Kolmogorov-Arnold Networks.”** arXiv:2404.19756. <https://arxiv.org/abs/2404.19756>

6. Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier, **“Efficient Content-Based Sparse Attention with Routing Transformers.”** Transactions of the Association for Computational Linguistics, 2021; arXiv:2003.05997. <https://arxiv.org/abs/2003.05997>

7. Taco Cohen and Max Welling, **“Group Equivariant Convolutional Networks.”** ICML 2016; arXiv:1602.07576. <https://arxiv.org/abs/1602.07576>

8. Stockfish NNUE PyTorch Wiki, **“NNUE.”** <https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html>

9. Tianzhu Ye et al., **“Differential Transformer.”** arXiv:2410.05258. <https://arxiv.org/abs/2410.05258>

10. Erik Jenner, Shreyas Kapur, Vasil Georgiev, Cameron Allen, Scott Emmons, and Stuart Russell, **“Evidence of Learned Look-Ahead in a Chess-Playing Neural Network.”** NeurIPS 2024 / OpenReview. <https://openreview.net/forum?id=8zg9sO4ttV>
