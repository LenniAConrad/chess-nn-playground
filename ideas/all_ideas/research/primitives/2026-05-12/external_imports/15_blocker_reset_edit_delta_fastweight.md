# reports/prompts/deep_research_primitive_results.md

## Literature calibration and novelty guardrails

The governing spec asks for new neural-network primitives, not architectures, encodings, or training tricks, and it makes incremental update, legal-move sparsity, chess group structure, and near-puzzle hard negatives central evaluation constraints.

Recent primitive-level work raises the bar. Mamba made SSM parameters input-dependent and uses a hardware-aware recurrent algorithm with linear sequence scaling; Mamba-2/SSD connects SSMs and attention through structured semiseparable matrices and reports a 2–8× faster core layer; Gated DeltaNet combines adaptive memory erasure with delta-rule memory updates; RWKV-7 adds vector-valued gates and in-context learning rates while keeping constant memory and constant per-token inference; Log-Linear Attention replaces fixed-size recurrent memory with a logarithmically growing set of states; xLSTM revisits recurrent memory with exponential gating and scalar/matrix memory.

The proposals below deliberately avoid merely restating graph attention, edge-conditioned convolution, group convolution, or hypergraph message passing: those already exist as recognizable families. GAT attends over graph neighborhoods; edge-conditioned convolution generates filters from edge labels on arbitrary graphs; group-equivariant CNNs use group convolutions and weight sharing; hypergraph neural networks model higher-order interactions.

## Ranking

Scores are 1–5, where 5 is strongest. “Novelty” means primitive-level novelty after self-audit, not “can never be prototyped from lower-level tensor ops.”

| Rank | Primitive | Novelty plausibility | RTX 3070 demonstrability | Inference-speed advantage | Generalises beyond chess | Main reason for rank |
|---:|---|---:|---:|---:|---:|---|
| 1 | `primitive_blocker_reset_scan` | 4.5 | 5.0 | 4.5 | 4.0 | Captures sliding-piece occlusion with a small, fast, falsifiable operator. |
| 2 | `primitive_edit_delta_fastweight` | 3.5 | 5.0 | 5.0 | 5.0 | Best HalfKA-style generalisation, but overlaps fast-weight/delta-rule literature. |
| 3 | `primitive_legal_edge_attention` | 4.0 | 3.5 | 4.0 | 4.0 | Most chess-structural; implementation is harder than scan/fastweight. |
| 4 | `primitive_rule_hyperedge_contract` | 4.0 | 3.0 | 3.0 | 4.5 | Good for pins/forks/discovered attacks; higher implementation risk. |
| 5 | `primitive_chess_orbit_linear` | 3.0 | 4.0 | 3.0 | 4.0 | Useful and clean, but closest to known finite-group equivariant linear layers. |

## Self-audit of the top 2

### Top 1: `primitive_blocker_reset_scan`

Devil’s advocate: if the blockers are fixed, this becomes a fixed linear ray kernel, so a reviewer could call it a weird long convolution. That objection fails only because blockers are part of the operator’s runtime control path: the effective segment boundaries change per board, and the Jacobian with respect to square features changes with occupancy. It is also not just Mamba/SSM, because Mamba’s novelty is input-conditioned sequence recurrence over one sequence, while this primitive is a multi-ray segmented scan with hard reset topology and bounded-change line invalidation.

Verdict: keep. It is the cleanest new primitive here.

### Top 2: `primitive_edit_delta_fastweight`

Devil’s advocate: with rank 1 and constant keys, this degenerates into `EmbeddingBag`/NNUE-style accumulator update; with append-only updates, it resembles linear attention, RetNet, Gated DeltaNet, or RWKV-style fast-weight recurrence. NNUE already exploits sparse inputs and minimal changes between adjacent evaluations, and Stockfish’s NNUE documentation explicitly centers sparse inputs and small input changes.

Verdict: keep, but do not claim the fast-weight idea is wholly new. The proposed primitive is specifically an unordered signed-edit active-set memory with exact deletion, not a causal sequence recurrence.

## Ranked primitive proposals

### primitive_blocker_reset_scan

**Name:** Blocker-Reset Ray Scan

**One-line claim:** A fused segmented scan that summarizes rays while resetting hidden state at content-defined blockers.

**Mathematical signature:**
For board squares \(S=64\), directions \(\Delta\), features \(X\in\mathbb{R}^{B\times S\times d}\), occupancy \(O\in\{0,1\}^{B\times S}\), ordered ray lines \(\ell=(s_1,\ldots,s_L)\), parameters \(U,V\in\mathbb{R}^{d\times d}\), \(\lambda_\delta\in(0,1)^d\):

\[
h_{b,\ell,t,\delta}=U x_{b,s_t} + (1-O_{b,s_t})\,\lambda_\delta\odot h_{b,\ell,t-1,\delta},
\quad
y_{b,s_t}=\sum_{\delta\in\Delta}V h_{b,\ell,t-1,\delta}.
\]

Gradients are standard through \(X,U,V,\lambda\); no gradient is taken through binary occupancy.

**Why this does not decompose into existing PyTorch ops:**
It is not `Conv2d`: the receptive field is not a fixed kernel but a blocker-delimited segment that changes per input. It is not standard attention: no \(S^2\) score matrix is formed. It is not a vanilla SSM/Mamba layer: the recurrence runs over multiple board rays with hard segment resets and supports invalidating only changed rays, whereas Mamba’s selective SSM is a sequence recurrence with input-conditioned parameters.

**Chess-specific motivation:**
Rooks, bishops, and queens are defined by line-of-sight until a blocker. Pins, skewers, discovered attacks, x-rays, and king safety all depend on blocker-delimited rays, not just local \(3\times3\) neighborhoods. This directly targets the prompt’s emphasis on hard near-puzzle false positives rather than easy aggregate PR-AUC gains.

**Generalisation beyond chess:**
Useful for visibility reasoning in robotics, lidar grids, ray-based scene graphs, cellular automata with barriers, and maze/path-planning domains.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B|\Delta|Sd^2)\) vs dense attention \(O(BS^2d)\) or large-kernel conv \(O(BSk^2d^2)\)
- Backward: \(O(B|\Delta|Sd^2)\)
- Incremental update on a bounded-change input: \(O(B|\Delta|Ld^2)\), with \(L\le 8\) for chess rays

**Scout-scale falsification test:**
Drop one instance into the i193 conv-only parent by replacing one spatial mixing layer with `BlockerResetRayScan + 1x1 projection`. Baseline: same-parameter \(3\times3\)/\(5\times5\) conv replacement. Train on 173k positions for 12 epochs on one RTX 3070. Primitive works if matched-recall CRTK class-1 near-puzzle false-positive rate improves by at least 5% with no more than 10% slower wall-clock inference. Primitive fails if gain appears only in aggregate PR AUC or inference slows materially.

**Failure mode catalogue:**
- Hidden rebrand objection: “this is just segmented `cumsum` plus a learned projection.”
- Numerical issue: repeated \(\lambda\) multiplication can underflow or saturate if \(\lambda\) is not parameterized by `sigmoid`/`softplus`.
- Speed issue: Python ray loops would erase the benefit; this needs a fused CUDA/Triton scan to be a real primitive.

**Status:** proposed

### primitive_edit_delta_fastweight

**Name:** Signed-Edit Fastweight Memory

**One-line claim:** Maintains second-order active-set memory by signed low-rank updates instead of recomputing all pair interactions.

**Mathematical signature:**
Let active atoms have features \(u_i\in\mathbb{R}^{d}\). The cached memory is \(M\in\mathbb{R}^{B\times r\times r}\). For signed edits \(E_b=\{(s_e,u_e)\}_{e=1}^{m_b}\), \(s_e\in\{-1,+1\}\):

\[
M'_b=M_b+\sum_{e=1}^{m_b}s_e\,\phi(u_e)\psi(u_e)^\top,
\quad
y_{b,q}=W_o\big(\alpha(x_{b,q})^\top M'_b\big).
\]

\(\phi,\psi,\alpha:\mathbb{R}^d\to\mathbb{R}^r\) are learned projections. Gradients flow to touched edited atoms, projections, and \(M\).

**Why this does not decompose into existing PyTorch ops:**
It is not `EmbeddingBag`: that is a first-order sum over active IDs, while this is a second-order fastweight matrix with exact signed deletion. It is not ordinary linear attention or RetNet: those are causal sequence operators, whereas this is an unordered active-set operator whose forward cost depends on the edit list. RetNet and related recurrent operators establish the value of \(O(1)\) inference-state updates, but not exact signed set deletion.

**Chess-specific motivation:**
A chess move changes only a few board atoms, which is exactly the condition exploited by NNUE’s efficient update principle. This primitive chases the same property but lets the cached state represent pair-like interactions, not only first-layer affine sums.

**Generalisation beyond chess:**
Applies to recommender sessions, dynamic graphs, sparse-event streams, molecule edits, board games, and any active-set model with frequent small changes.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(Bm r^2 + BQr^2)\) vs recomputed pair memory \(O(Bn r^2 + BQr^2)\)
- Backward: \(O(Bm r^2 + BQr^2)\), plus sparse projection-gradient accumulation
- Incremental update on a bounded-change input: \(O(Br^2)\) for constant \(m\)

**Scout-scale falsification test:**
Use the i243 HalfKA+dual-stream harness but replace the first-order HalfKA accumulator with this second-order edit memory at small rank \(r\in\{16,32\}\). Baseline: identical head with full recomputed pair pooling and a normal HalfKA accumulator. Primitive works if it matches or improves CRTK class-1 matched-recall near-puzzle FP rate while giving at least 1.5× faster sequential make/unmake evaluation. It fails if the only win is throughput with worse hard-negative discrimination.

**Failure mode catalogue:**
- Hidden rebrand objection: “rank-1 version is just NNUE/EmbeddingBag; append-only version is linear attention.”
- Numerical issue: low-rank memory can explode unless \(M\) is normalized or \(\phi,\psi\) are bounded.
- Speed issue: \(r^2\) cost beats HalfKA only at low ranks; high-rank versions become slower than recompute.

**Status:** proposed

### primitive_legal_edge_attention

**Name:** Legal-Edge Sparse Attention

**One-line claim:** Attention whose sparse edge set is generated inside the operator by current legal or pseudo-legal move rules.

**Mathematical signature:**
Given square features \(X\in\mathbb{R}^{B\times S\times d}\), board tags \(T\in\{0,\ldots,C\}^{B\times S}\), and side-to-move \(c\), define a deterministic edge generator:

\[
E_b=\mathcal{L}(T_b,c_b)\subseteq S\times S\times R.
\]

For each source square \(i\):

\[
a_{ijr}=\operatorname{softmax}_{(j,r):(i,j,r)\in E_b}
\left((W_Qx_i)^\top W_Kx_j/\sqrt{d}+\beta_r\right),
\quad
y_i=\sum_{(j,r):(i,j,r)\in E_b}a_{ijr}W^V_r x_j.
\]

Gradients flow through \(X,W,\beta\), not through \(\mathcal{L}\).

**Why this does not decompose into existing PyTorch ops:**
It is not `MultiheadAttention` with an `attn_mask`: PyTorch’s attention mask is an external tensor applied to a dense attention formulation, whereas this primitive synthesizes a sparse edge list from board content before allocating attention neighborhoods. It is also not ordinary GAT, because GAT assumes a graph neighborhood is supplied; here the connectivity rule is part of the operator.

**Chess-specific motivation:**
Chess connectivity changes with every move: sliders are blocked by occupancy, pawns attack differently by color, kings and knights have piece-specific neighborhoods, and legal move sets are sparse. This directly exploits the prompt’s point that content-dependent legal-move connectivity is structurally underused.

**Generalisation beyond chess:**
Useful in dynamic graphs whose edges are generated by symbolic constraints: routing, program analysis, traffic networks, contact graphs, and game-state evaluators.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B|E|d^2)\) vs dense attention \(O(BS^2d)\)
- Backward: \(O(B|E|d^2)\)
- Incremental update on a bounded-change input: \(O(B|\Delta E|d^2)\); in chess \(|\Delta E|\) is board-bounded, but not always tiny for opened/closed slider rays

**Scout-scale falsification test:**
Drop into the i242 attention ablation as a replacement for one dense/self-attention chess stream. Baseline: i242’s corresponding dense or fixed-mask attention stream. Primitive works if it lowers matched-recall CRTK class-1 near-puzzle FP rate and is faster per batch than dense attention at equal hidden width. It fails if it needs LC0-scale data before showing a signal.

**Failure mode catalogue:**
- Hidden rebrand objection: “this is just masked attention after a move-generator preprocessor.”
- Numerical issue: some positions have tiny neighborhoods; softmax over one edge collapses gradients.
- Speed issue: legal-edge generation can dominate unless fused or cached across make/unmake moves.

**Status:** proposed

### primitive_rule_hyperedge_contract

**Name:** Rule-Generated Hyperedge Contraction

**One-line claim:** Generates k-ary tactical motifs from board content and contracts their features in one differentiable operator.

**Mathematical signature:**
For \(X\in\mathbb{R}^{B\times S\times d}\), board tags \(T\), and rule family \(\mathcal{R}\), generate:

\[
H_b=\{(r,s_1,\ldots,s_k): P_r(T_b,s_1,\ldots,s_k)=1,\ k\le K\}.
\]

For each hyperedge \(e=(r,s_1,\ldots,s_k)\):

\[
z_e=A_r\left[\sum_{\ell=1}^k U_{r,\ell}x_{s_\ell};\ \bigodot_{\ell=1}^k V_{r,\ell}x_{s_\ell}\right],
\quad
y_i=\sum_{e:i\in e}B_{r,\operatorname{pos}(i,e)}z_e.
\]

Gradients flow through feature contractions and parameters, not through predicate \(P_r\).

**Why this does not decompose into existing PyTorch ops:**
It is not a pairwise GNN layer: the primitive creates and evaluates k-ary relations directly. It is not a standard hypergraph neural network, because hypergraph methods typically consume a given hypergraph structure; here the hyperedges are generated inside the operator from the current input state. Hypergraph NNs are already a known family for higher-order interactions, so the novelty claim is the fused rule-generation-plus-contraction signature.

**Chess-specific motivation:**
Many tactical errors are k-ary: a pin is king–blocker–attacker, a fork is attacker–target1–target2, and discovered attack is mover–revealed-attacker–target. Pairwise legal moves can miss the simultaneous relation unless multiple layers infer it indirectly.

**Generalisation beyond chess:**
Applies to constraint satisfaction, molecular many-body interactions, scheduling conflicts, program-analysis motifs, and scene graphs with multi-object relations.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B|H|Kd^2)\) vs pairwise message passing that may need \(K\) rounds, approximately \(O(BK|E|d^2)\) plus depth
- Backward: \(O(B|H|Kd^2)\)
- Incremental update on a bounded-change input: \(O(B|\Delta H|Kd^2)\)

**Scout-scale falsification test:**
Insert one low-rank \(K=3\) instance after the first spatial mixer in i193. Baseline: two-layer GAT or edge-conditioned graph layer with the same parameter budget. Primitive works if CRTK class-1 matched-recall near-puzzle FP rate improves by at least 5% without more than 20% inference slowdown. It fails if it only improves easy negatives or becomes a move-generator bottleneck.

**Failure mode catalogue:**
- Hidden rebrand objection: “this is hypergraph message passing with a hand-coded hypergraph builder.”
- Numerical issue: multiplicative contractions \(\bigodot\) can vanish; low-rank bilinear terms need normalization.
- Speed issue: motif enumeration can explode if predicates are not capped to chess-relevant \(K\le3\) or \(K\le4\).

**Status:** proposed

### primitive_chess_orbit_linear

**Name:** Chess-Group Orbit Linear

**One-line claim:** A finite-group equivariant linear map over square, color, side-to-move, and piece-type orbits.

**Mathematical signature:**
Let atoms \(A=S\times C\times P\), features \(X\in\mathbb{R}^{B\times |A|\times d}\), and finite chess symmetry group \(G\) act by permutations \(\pi_g\) over atoms and representations \(\rho_g\) over channels. Define ordered-pair orbits:

\[
\omega(a,b)=\operatorname{orbit}_G(a,b).
\]

The primitive is:

\[
y_a=\sum_{b\in A}K_{\omega(a,b)}x_b,
\quad
\text{with } K_{\omega(\pi_g a,\pi_g b)}=\rho_g K_{\omega(a,b)}\rho_g^{-1}.
\]

Gradients are ordinary linear-map gradients accumulated over orbit-tied parameters.

**Why this does not decompose into existing PyTorch ops:**
It is not `Linear` on a flattened board, because the parameter tensor is constrained by atom-pair orbits under a non-grid chess group. It is not ordinary group convolution over translations/rotations, because the action simultaneously permutes files, ranks, color, side-to-move, and piece labels. Prior G-CNNs establish group-convolution weight sharing; this is best described as an underexplored finite typed-orbit linear primitive for chess rather than a wholly new equivariance theory.

**Chess-specific motivation:**
Chess has exact file-mirror symmetry and color/rank involutions, while piece colors and pawn directions must transform together. Encoding those symmetries at the operator level can reduce sample complexity at scout scale without requiring more data.

**Generalisation beyond chess:**
Useful for typed graphs, molecules with automorphism groups, board games, multi-agent grids, and symbolic domains with finite group actions over typed atoms.

**Complexity (forward, backward, incremental-update):**
- Forward: dense \(O(B|A|^2d^2)\), sparse-active \(O(Bn_{\text{active}}|A|d^2)\), vs untied dense linear \(O(B|A|^2d^2)\) with many more parameters
- Backward: same as forward plus orbit-wise gradient accumulation
- Incremental update on a bounded-change input: \(O(B|\Delta A||A|d^2)\) if output cache is maintained

**Scout-scale falsification test:**
Replace one flattened board-mixing linear/1x1 mixing step in i193 or i243 with orbit-tied linear using exact file-mirror plus color/rank involution. Baseline: same-rank untied linear or standard \(1\times1\) conv. Primitive works if it improves CRTK class-1 near-puzzle FP rate at equal or lower parameter count with no speed regression. It fails if tying exact symmetries reduces tactical expressivity.

**Failure mode catalogue:**
- Hidden rebrand objection: “this is just a manually tied linear layer or finite-group convolution.”
- Numerical issue: orbit sharing may over-constrain asymmetric pawn/side-to-move effects if the group action is wrong.
- Speed issue: dense atom-pair mixing is too expensive unless restricted to active atoms or low-rank orbit kernels.

**Status:** proposed

## What I cut

1. **Attack-map masked attention.** Rejected because it is exactly the anti-example: compute an attack mask, then run ordinary masked attention.

2. **A new material-aware activation function.** Rejected because it is a scalar activation tweak, not a new primitive-level connectivity, gradient, or complexity class.

3. **Ray positional encodings.** Rejected because encodings sit below the primitive level and would not change the operator graph.

4. **Conv-attention mixture gate.** Rejected because it is a composition of existing primitives; the uploaded spec explicitly rules out learned mixtures of attention and convolution.

5. **PV/Stockfish-score-conditioned routing.** Rejected because Stockfish scores, PVs, node counts, and verification metadata are labels/audit fields, not legal primitive inputs.

## Bibliography

- Albert Gu and Tri Dao, **“Mamba: Linear-Time Sequence Modeling with Selective State Spaces.”** Used as calibration for input-conditioned recurrent primitives and linear-time sequence scaling. <https://arxiv.org/abs/2312.00752>
- Tri Dao and Albert Gu, **“Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.”** Used as calibration for Mamba-2/SSD and structured semiseparable operators. <https://arxiv.org/abs/2405.21060>
- Songlin Yang, Jan Kautz, and Ali Hatamizadeh, **“Gated Delta Networks: Improving Mamba2 with Delta Rule.”** Used to self-audit fastweight/delta-rule overlap. <https://arxiv.org/abs/2412.06464>
- Bo Peng et al., **“RWKV-7 ‘Goose’ with Expressive Dynamic State Evolution.”** Used to self-audit vector-gated recurrent memory and constant-time inference claims. <https://arxiv.org/abs/2503.14456>
- Han Guo et al., **“Log-Linear Attention.”** Used as 2025–2026 calibration for log-growing hidden-state attention. <https://arxiv.org/abs/2506.04761>
- Maximilian Beck et al., **“xLSTM: Extended Long Short-Term Memory.”** Used as calibration for recurrent primitives with exponential gating and matrix memory. <https://arxiv.org/abs/2405.04517>
- Yutao Sun et al., **“Retentive Network: A Successor to Transformer for Large Language Models.”** Used as calibration for recurrent/parallel duality and \(O(1)\) inference state. <https://arxiv.org/abs/2307.08621>
- Michael Poli et al., **“Hyena Hierarchy: Towards Larger Convolutional Language Models.”** Used as calibration for subquadratic gated long-convolution alternatives to attention. <https://arxiv.org/abs/2302.10866>
- Petar Veličković et al., **“Graph Attention Networks.”** Used to distinguish legal-edge attention from supplied-graph GAT. <https://arxiv.org/abs/1710.10903>
- Martin Simonovsky and Nikos Komodakis, **“Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs.”** Used to distinguish rule-generated graph operators from supplied edge-conditioned filters. <https://arxiv.org/abs/1704.02901>
- Taco Cohen and Max Welling, **“Group Equivariant Convolutional Networks.”** Used to calibrate the chess-orbit equivariant primitive. <https://arxiv.org/abs/1602.07576>
- Sunwoo Kim et al., **“A Survey on Hypergraph Neural Networks.”** Used to distinguish rule-generated hyperedge contraction from existing hypergraph NN families. <https://arxiv.org/abs/2404.01039>
- Stockfish NNUE documentation and Stockfish NNUE announcement. Used for the sparse-input, small-change, efficiently updatable neural-network motivation. <https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html>
- PyTorch `MultiheadAttention` documentation. Used to distinguish legal-edge sparse attention from externally supplied dense attention masks. <https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html>
