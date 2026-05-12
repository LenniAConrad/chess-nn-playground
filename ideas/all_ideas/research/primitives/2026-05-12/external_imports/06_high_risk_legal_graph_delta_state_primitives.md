# Five High-Risk Neural-Network Primitives for Chess Evaluation

**TL;DR**
- The single highest-ranked proposal, **Legal-Move Laplacian Pseudoinverse Propagation (LM-LPP)**, is a content-dependent-connectivity primitive whose forward map is the application of a *resolvent* of the per-position legal-move Laplacian to a square-feature tensor — this is not standard masked attention (no softmax, no QK), not standard GCN (no row-stochastic normalisation; uses a Neumann-series resolvent instead), and cannot be expressed as a fixed PyTorch op because the operator itself is the input.
- Three of the five proposals (LM-LPP, Ray-Gated Ternary Pin Operator, Implicit Quiescence Fixed-Point) are explicitly built around content-determined edges, in line with the user's heaviest weight on input-determined connectivity; one (Delta-Sparse Accumulator with Lazy Half-Nonlinearity) targets the NNUE O(Δ)-update property directly; one (Color-Involution-Steerable Linear) targets the Z/2 color-swap group structure that no torch.nn layer enforces.
- All five honestly fail standard "is-this-just-a-rebrand?" objections in at least one corner — the self-audit at the end explicitly retires three further candidates (Mamba-on-board-scan, learned-sparse-MoE-by-piece-type, DEQ-over-attention-mask) that did not survive. The user should pilot **LM-LPP** and **Delta-Sparse Accumulator** first; both have ~2 GPU-hour scout-scale falsification tests on a single RTX 3070.

---

## Key Findings

The deep-learning literature 2023–2026 contains many *near-misses* for chess-relevant primitives but no exact match for content-dependent connectivity at the *operator* level:

1. **Selective SSMs (Mamba, Mamba-2/SSD)** make the *A, B, C* matrices input-dependent but the *scan order* is a fixed 1-D sequence. A version where the scan DAG is the legal-move graph would be new at the primitive level, but, as the self-audit shows, can usually be decomposed into masked GRU + topological sort, so it does not clear the bar.
2. **Content-aware sparse attention** (FlexPrefill, SpargeAttn, MoSA, SPARSEK, MInference, NSA, SeerAttention 2024–2025) produces masks on the fly from QK statistics. *None* of these recompute the mask from a non-learned, rule-based, position-dependent topology, and they all decompose as `softmax(QK + M)` with `M` produced upstream — so a primitive whose mask is the rule-determined legal-move graph is *not* the same op even though it can be naïvely simulated by one.
3. **Dynamic / temporal GNNs** (D3-GNN, NO-HGNN, De Bruijn GNN, EvoNet 2024–2025) change topology *between forward passes* (one snapshot per timestep). A primitive whose topology changes *within* a forward pass *per token per layer* and whose backward must propagate edge-presence gradients only through edge features (not through edge existence) is in a different complexity class.
4. **Hypergraph NNs with hyperedge-dependent node embeddings** (Aponte et al., 2022; HeIHNN 2024) are the closest analog to a ternary "piece-on-ray-between-two-squares" operator, but standard HGNNs assume static incidence matrices; per-board recomputation of incidence is not a primitive any HGNN library exposes.
5. **NNUE's `AccumulatorStack` + Finny Tables** (Stockfish, Koivisto) is currently a *system*-level optimization, not a torch.nn primitive. A general "Δ-sparse accumulator with lazy half-nonlinearity" primitive does not exist in PyTorch and would be reusable in any domain with bounded-change inputs (event streams, code-edit models, physical-simulation rollouts).
6. **Group-equivariant CNNs** (Cohen & Welling 2016, CEConv 2023, Partial G-CNN 2024) cover dihedral/translation/color-hue groups; the Z/2 color-swap involution of chess (`σ`: white↔black, files preserved, ranks mirrored, side-to-move flipped) is *not* a special case of any of these — its commutant has a different Schur decomposition. A linear layer with weights constrained to commute with σ is a primitive distinct from BatchNorm/LayerNorm/G-Conv.
7. **Deep Equilibrium Models** (Bai 2019; RevDEQ 2025; MDEQ 2020) define implicit layers via a learned `f(z, x)` whose fixed point `z*` is found by root-finding. None of these condition the *connectivity of `f`* on the input. A DEQ whose iteration map propagates only along the input-determined legal-move graph is a structurally new implicit primitive (call it "quiescence fixed-point").

The Leela Chess Zero interpretability work (Jenner et al., NeurIPS 2024; Cruz, arXiv:2505.21552) shows that a 109M-parameter pure-attention transformer already does multi-ply lookahead implicitly, but that this lookahead is *learned* over a 256-token square embedding without any chess-rule scaffolding. Building rule-aware connectivity into the operator (not the encoding) is currently an open primitive-level gap.

---

## Details — Five Proposals

### 1. primitive_lm_lpp

**Name:** Legal-Move Laplacian Pseudoinverse Propagation (LM-LPP)

**One-line claim:** A linear operator whose action on square features is the resolvent `(I − αL(x))⁻¹` of the per-position signed legal-move Laplacian — no softmax, no QK, connectivity is the input.

**Mathematical signature:**
`f : R^{[B, 64, d]} × {Boards}^B → R^{[B, 64, d]}`
For board `b`, build the directed weighted adjacency `A(b) ∈ R^{64×64}` whose entry `A_{ij}(b)` equals a learned piece-conditioned weight `w(piece(i, b))` if square `i` can legally pseudo-move to `j` on board `b`, else 0. Form `L(b) = D(b) − A(b)`. The forward map is
`Y = (I − αL(x)) ⁻¹ X · Θ`
with α a learned scalar (`|α| < 1/λ_max` enforced by spectral clipping), Θ ∈ R^{d×d} a learned mixing matrix. Implemented as a truncated Neumann series `Y = Σ_{k=0}^{K} α^k L^k X · Θ` with `K ∈ {3,…,8}`, exploiting the legal-move graph's bounded degree (≤ 27 in chess). Backward by adjoint, treating `L(x)` as constant w.r.t. `x` (gradient only through `w`, `Θ`, and edge-presence-conditioned features — edge existence itself is non-differentiable and is correctly treated as a stop-gradient on board-state, satisfying the rule that Stockfish/PV metadata never enters the graph as a feature).

**Why this does not decompose into existing PyTorch ops:** The closest existing op is `torch.sparse.mm` followed by `softmax` (i.e., GAT/sparse-attention). LM-LPP differs in two graph-level ways: (a) the aggregation is a *resolvent/Neumann series*, not a row-stochastic average — the spectrum is shaped by α and L, not by softmax; (b) the connectivity is the rule-determined legal-move adjacency, recomputed per board from a non-learned indicator, so the op-graph has a node ("`compute_legal_adjacency(x)`") with zero gradient w.r.t. continuous inputs. This combination is not any of `Conv2d`, `MultiheadAttention`, `MessagePassing`, or `SparseLinear` — the resolvent over a content-dependent stop-gradient adjacency is structurally distinct.

**Chess-specific motivation:** The legal-move graph is the single most chess-specific structural fact. A bishop on d4 has degree ≤ 13; a knight on b1 has degree ≤ 3; pinned pieces lose all degree. Pseudoinverse propagation captures *transitive* tactical influence (X-rays, batteries, discovered attacks) in one operator application, whereas masked attention captures only one-hop attacks per layer.

**Generalisation beyond chess:** Any domain with a per-input combinatorial graph whose topology is rule-determined: chemistry (valence graphs), program-analysis (def-use graphs), traffic networks (per-snapshot road closures).

**Complexity:**
- Forward: `O(B · K · |E(x)| · d)` where `|E(x)| ≤ 64·27` for chess; closest existing primitive (Multihead Attention on 64 tokens) is `O(B · 64² · d)`. K=4 LM-LPP is ~3× cheaper than dense MHA at 64 tokens, and is asymptotically linear in legal-move count rather than quadratic.
- Backward: `O(B · K · |E(x)| · d)`.
- Incremental update: `O(Δ|E| · d · K)` if a move changes ≤ Δ edges (typically 4–20). Full O(Δ) only achievable for `K=1`; for `K>1` it is `O(K · Δ · max-degree^{K−1} · d)` — sublinear in 64 but not strictly bounded-change. **Flag: this is weaker than NNUE's O(Δ).**

**Scout-scale falsification test:** Drop LM-LPP in place of one self-attention block in the user's i242 chess-decomposed-attention model. Train on the same 173k positions × 12 epochs, single RTX 3070, single seed. Measure CRTK class-1 (verified near-puzzle) matched-recall FP rate at recall=0.5. Primitive **works** if the FP-rate drops ≥ 15 % relative to the i242 baseline *and* wall-clock per epoch is ≤ 1.3× baseline. Primitive **fails** if FP-rate change is within ±5 % (likely a rebrand of legal-mask attention) or wall-clock blows up > 2×.

**Failure mode catalogue:**
- *Hidden rebrand of GAT with a fixed-per-board mask*: if K=1 and the Neumann series is truncated to a single hop, this is exactly GAT with a stop-gradient adjacency. The novelty only survives for K ≥ 2 *and* if the spectral resolvent (`Σ α^k L^k`) is empirically better than `K` stacked GAT layers with independent weights.
- *Numerical instability if `|αλ_max| ≥ 1`*: Neumann series diverges. Mitigation by spectral clipping of α via power-iteration estimate of `λ_max`, but this adds O(K) overhead and is brittle on long-range positions (open files, queen-controlling-many-squares).
- *Sparse-CUDA inefficiency on consumer hardware*: PyTorch sparse-CSR matmul on RTX 3070 has poor utilization below ~50% density; 64-token legal-move graphs have ~5% density. May lose to dense MHA at small B despite asymptotic win.

**Status:** proposed

---

### 2. primitive_dsa_lhn

**Name:** Delta-Sparse Accumulator with Lazy Half-Nonlinearity (DSA-LHN)

**One-line claim:** A differentiable primitive whose forward cost depends on the change in input rather than its size, by maintaining a pre-activation accumulator and applying nonlinearity only at read.

**Mathematical signature:**
Operator state `h ∈ R^{[B, d]}`. Input is a *sparse event stream* `e_t = (i_t, s_t, v_t)` with `i_t ∈ [n_features]`, sign `s_t ∈ {+1, −1}`, optional continuous value `v_t ∈ R`. Forward update on event:
`h_t = h_{t−1} + s_t · v_t · W[i_t, :]`
Read operation (only when output is needed):
`y = φ(h_t) · U`
where `φ` is a clipped-ReLU-like activation **inside** the operator (the nonlinearity is part of the primitive, not external). Backward through an event sequence of length T: gradient w.r.t. `W[i, :]` accumulates only over events touching feature `i`; gradient w.r.t. read outputs flows through `φ'(h_t)` at read times only.

**Why this does not decompose into existing PyTorch ops:** Closest candidates are `nn.EmbeddingBag` (no signed events, no read/write asymmetry, no internal nonlinearity), and a `nn.Linear` over a one-hot input (which loses the O(|Δ|) update property entirely because every forward recomputes Wx). The defining property — *the nonlinearity sits between the accumulator and the next layer, and reads can be performed with O(|Δ|) cost since the last clean read, while the gradient correctly accounts for the activation pattern at each historical read* — is a stateful operator with a temporal computation graph that no torch.nn module currently exposes. Existing PyTorch RNN cells require dense O(d) updates per step.

**Chess-specific motivation:** This is the formalization, as a torch.nn primitive, of Stockfish NNUE's `AccumulatorStack` plus `Finny Tables` cache (`src/nnue/nnue_accumulator.h`). Every chess search tree node differs from its parent by 2–4 sparse feature flips; an O(Δ) primitive turns first-layer inference from the dominant cost into negligible.

**Generalisation beyond chess:** Any bounded-change input sequence: code-edit modeling (LSP diffs), molecular-dynamics rollouts (atom-position deltas), simulator-in-the-loop RL (state diffs), event-camera vision, financial order-book updates.

**Complexity:**
- Forward (read): `O(d_out · d_hidden)` at read; `O(|Δ| · d_hidden)` for accumulator update between reads. Closest primitive (`nn.Linear`) is `O(n_features · d_hidden)` per forward.
- Backward: `O((|Δ| + d_hidden) · d_out)` per read.
- Incremental update: `O(|Δ| · d_hidden)` — **strictly bounded-change**, this is the structural win.

**Scout-scale falsification test:** Implement DSA-LHN as a custom `torch.autograd.Function`. Replace the first dense layer of i193 (the conv-only baseline) on the HalfKA encoding. Train 12 epochs, 173k positions, RTX 3070. Measure (a) CRTK class-1 matched-recall FP rate, (b) wall-clock inference on a 1000-position MCTS-style sequence with average 3 feature flips per step. Primitive **works** if FP-rate matches baseline within ±2 % *and* sequential inference is ≥ 3× faster. Primitive **fails** if FP-rate degrades > 5 % (suggesting the lazy-nonlinearity placement is wrong) or speedup < 1.5× (suggesting custom-op overhead dominates).

**Failure mode catalogue:**
- *Hidden rebrand of `EmbeddingBag` + manual ReLU*: if the user's chess features are pure binary (no signed value `v_t`), then DSA-LHN collapses to an `EmbeddingBag` followed by ReLU and is not novel. The novelty only survives when `v_t` is a continuous value (e.g., piece-attack count, mobility weight) — i.e., when the "feature" is real-valued.
- *Gradient instability under long event sequences*: if the read happens after many flips, `φ'(h_t)` may saturate (all zeros) and gradients vanish. Requires periodic accumulator refresh, similar to Finny Tables, complicating the differentiable abstraction.
- *Autograd-graph blowup*: PyTorch's tape-based autograd requires storing every event for backward. For T=10⁵ event sequences this dominates memory unless gradient checkpointing per "epoch of events" is implemented — non-trivial.

**Status:** proposed

---

### 3. primitive_rg3t

**Name:** Ray-Gated Ternary Tensor Operator (RG3T)

**One-line claim:** A primitive that computes square-triple features only when the three squares are collinear on a chess ray, with the third (middle) square gating the contribution by piece identity.

**Mathematical signature:**
`f : R^{[B, 64, d]} × {Boards}^B → R^{[B, 64, d']}`
Let `R(i, j) ⊂ [64]` be the set of squares lying strictly between `i` and `j` on the (queen-style) ray connecting them, empty if `i, j` are not ray-aligned. The output at square `i` is
`Y_i = Σ_{j : R(i,j) ≠ ∅} Σ_{k ∈ R(i,j)} (X_i ⊗ X_j ⊗ X_k) ×_3 T · g(piece_k)`
where `T ∈ R^{d × d × d × d'}` is a learned 4-tensor, `g : PieceType → R` is a learned piece-conditioned gate, and `⊗ ... ×_3 T` denotes a Tucker-style contraction. Only `O(64 · 27)` triples are non-empty for chess (each square has ≤ 27 ray-aligned partners, each ray has ≤ 6 between-squares). The gate `g(piece_k)` is differentiable in `T, g` but the *support set* of triples is a content-determined stop-gradient on board state.

**Why this does not decompose into existing PyTorch ops:** Triplet attention exists (Wang et al. 2020) but it is a pairwise rebrand using attention over a third axis. RG3T is a *true ternary tensor operator* with a content-determined sparse support — the closest decomposition would be `einsum('bid,bjd,bkd,defg->bie', X, X, X, T) * mask`, but the mask is `O(64³)` dense and the operation costs `O(64³ · d²)` instead of the `O(64 · 27 · 6 · d²) ≈ O(64² · d²)` that RG3T achieves with content-dependent indexing. The sparse-tensor-contraction primitive for ternary content-determined supports does not exist in `torch.einsum` (which assumes dense participation).

**Chess-specific motivation:** Pins, skewers, X-rays, batteries, discovered attacks are inherently ternary: attacker–middle-piece–target. No pairwise operator captures the asymmetric role of the middle piece (pinned vs pinning) without composing many layers. Empirically, RG3T should win exactly on hard-negative CRTK class-1 puzzles where the verifier near-miss involves a pin/skewer.

**Generalisation beyond chess:** Scene graphs with ternary spatial relations (between, behind, beside); molecular models where 3-body interactions are required (angle terms in force fields, though those are already in NequIP-style operators); program models with three-argument operations (load/store/index).

**Complexity:**
- Forward: `O(B · 64 · 27 · 6 · d² · d')` ≈ `O(B · 10⁴ · d² · d')`. Triplet attention is `O(B · 64³ · d) = O(B · 2.6·10⁵ · d)`.
- Backward: same order as forward.
- Incremental update: `O(Δ · 27 · 6 · d² · d')` if Δ piece changes; ~10× speedup over recompute for typical Δ=2. Not strictly bounded-change because changing one square invalidates O(27) of its rays.

**Scout-scale falsification test:** Add a single RG3T layer between layers 2 and 3 of i193. Train 12 epochs on 173k positions. Measure CRTK class-1 FP rate at matched recall vs. baseline *and* vs. a triplet-attention ablation (i.e., dense `einsum` with the same gating mask) to verify the win comes from the operator and not the inductive bias. Primitive **works** if it beats *both* baselines on FP rate by ≥ 10 % relative, especially on pin/skewer subcategory. Primitive **fails** if it is matched by dense-ternary-with-mask (which means the inductive bias is the only win and the primitive is a sparse-compute trick, not a new op).

**Failure mode catalogue:**
- *Hidden rebrand of "triplet attention with hard mask"*: if the dense-mask ablation matches, RG3T is just sparse compute, not a new mathematical operator.
- *Parameter explosion in `T ∈ R^{d × d × d × d'}`*: at d=64, `T` has 16M parameters per layer. Requires Tucker or CP decomposition of T to be practical, which then risks collapsing to a sequence of bilinear ops (decomposable again).
- *Custom CUDA kernel needed*: PyTorch's sparse-ternary einsum is not implemented; CPU implementation will be too slow for training. Implementing a Triton kernel for ~10⁴ triples per board with sub-millisecond latency is non-trivial.

**Status:** proposed

---

### 4. primitive_ciel

**Name:** Color-Involution Steerable Linear (CIEL)

**One-line claim:** A linear layer whose weight matrix is parameterized to commute exactly with the chess color-swap involution σ, splitting features into σ-symmetric and σ-antisymmetric channels at every layer.

**Mathematical signature:**
Let `σ : R^{[B, 64, d]} → R^{[B, 64, d]}` be the color-swap involution: rank-mirror the board, swap white-piece and black-piece channels, negate side-to-move. `σ² = I`. CIEL is a linear map `W` such that `W ∘ σ = σ ∘ W`. Equivalent parametrization: split the channel axis into `d = d⁺ + d⁻` (σ-eigenvalues +1 and −1), and parameterize `W` block-diagonal in the σ-eigenspaces:
`Y = W⁺ X⁺ + W⁻ X⁻`,    with `X⁺ = ½(X + σX)`, `X⁻ = ½(X − σX)`.
Bias is non-zero only on `Y⁺`. Activation `ρ` must satisfy `ρ(−x) = −ρ(x)` on σ⁻ channels (e.g., `tanh`) and is arbitrary on σ⁺ channels.

**Why this does not decompose into existing PyTorch ops:** This is a parameter-tying primitive analogous to G-equivariant convolution (Cohen & Welling 2016) but for the specific Z/2 involution σ that combines rank-mirror, channel-swap, and bias-flip. No `nn.Linear` variant in PyTorch enforces this constraint; the closest existing technique is to symmetrize at the loss level (data augmentation) or to average `f(x) + σ⁻¹ f(σx)`, but both are O(2×) more expensive at inference and do not enforce the constraint at every layer — they only enforce it at the network's output. CIEL bakes the constraint into the operator's *weight space*, exactly as G-Conv does for the cyclic group C_n but for the chess-specific Z/2.

**Chess-specific motivation:** Chess evaluation is exactly σ-antisymmetric: `eval(x) = −eval(σx)`. Standard networks must learn this from data; a CIEL-only network enforces it by construction, halving the effective parameter space and removing a known source of variance on small-data regimes (the scout-scale 173k positions × 12 epochs).

**Generalisation beyond chess:** Any domain with a known order-2 involution: physics models with time-reversal or parity symmetry, two-player zero-sum game evaluators (Go, shogi, Othello), preference models with antisymmetric pairwise comparisons.

**Complexity:**
- Forward: `O(B · n · (d⁺² + d⁻²))` ≤ `O(B · n · d²/2)`; closest primitive (`nn.Linear`) is `O(B · n · d²)`. Roughly 2× cheaper at matched capacity.
- Backward: same.
- Incremental update: not applicable in the bounded-change sense — CIEL is stateless.

**Scout-scale falsification test:** Replace every `nn.Linear` in i242's MLP heads with CIEL (matching parameter count by setting `d⁺ = d⁻ = d/2`). Train 12 epochs, 173k positions, RTX 3070, single seed. Measure CRTK class-1 matched-recall FP rate and check exact antisymmetry `|f(x) + f(σx)|` on a held-out 10k positions (should be 0 up to floating-point error). Primitive **works** if antisymmetry is exact *and* FP rate is at parity or better with baseline at half the parameters. Primitive **fails** if FP rate degrades > 3 % at matched parameter count (suggesting σ-equivariance is too restrictive an inductive bias to compete with augmentation).

**Failure mode catalogue:**
- *Hidden rebrand of data augmentation*: a network trained with σ-augmentation will approximately satisfy `f(x) = −f(σx)`; CIEL's only advantage is *exactness* and *halved parameters*. If exactness does not improve hard-negative discrimination, the primitive is academically interesting but practically a rebrand.
- *Activation-function constraint is restrictive*: σ⁻ channels require odd activations (tanh, sin); using ReLU on σ⁻ silently breaks equivariance. This is a footgun that a published primitive must enforce statically.
- *Interaction with normalization layers*: BatchNorm/LayerNorm do not in general commute with σ unless tied (statistics computed over σ-orbits). Composing CIEL with stock normalization layers breaks the equivariance guarantee; a CIEL-compatible norm must be developed alongside, slightly weakening the "single-primitive" claim.

**Status:** proposed

---

### 5. primitive_iqfp

**Name:** Implicit Quiescence Fixed-Point Operator (IQFP)

**One-line claim:** A Deep-Equilibrium-style primitive whose fixed-point iteration map propagates information *only* along the content-determined legal-move graph, yielding an "infinite-quiescence-search" layer in constant memory.

**Mathematical signature:**
`f : R^{[B, 64, d]} × {Boards}^B → R^{[B, 64, d]}`
Define an iteration map
`g(z, x) = LayerNorm(z + A(x) · MLP(z))`
where `A(x) ∈ R^{64×64}` is the legal-move adjacency for board `x` (stop-gradient on existence), and the fixed point `z* = g(z*, x)` is found by Anderson acceleration or a Broyden solver. Output `Y = z*`. Backward via implicit differentiation: `∂L/∂θ = ∂L/∂z* · (I − ∂g/∂z)⁻¹ · ∂g/∂θ`, with the Jacobian-vector product `(I − ∂g/∂z)⁻¹v` computed by a single linear solve (also fixed-point-iterated). The connectivity restriction means each iteration has cost `O(|E(x)| · d²)`, independent of fixed-point-iteration count K.

**Why this does not decompose into existing PyTorch ops:** Standard DEQs (Bai et al. 2019, MDEQ 2020, RevDEQ 2025) define `g` over dense connectivity or over a fixed graph. IQFP makes the iteration map's *connectivity* a function of the input board. The forward op cannot be expressed as `for k in range(K): z = g(z, x)` because `K` is determined adaptively by the solver, and the backward is computed via implicit differentiation, not BPTT — this is exactly the property that makes DEQ a primitive rather than a composition. IQFP inherits that property and adds content-dependent connectivity. The closest pretender, "DEQ with a fixed sparse mask", does not exist as a torch.nn primitive and would still differ because IQFP's mask varies per board.

**Chess-specific motivation:** Quiescence search in classical engines iterates capture sequences to a quiet position; an IQFP layer's fixed point is the analog at the representation level — a feature vector stable under one more ply of (content-determined) legal information propagation. The primitive directly addresses the Leela look-ahead finding (Jenner et al., NeurIPS 2024) that strong nets implicitly do multi-ply reasoning, by making that reasoning depth adaptive and rule-aware.

**Generalisation beyond chess:** Any reasoning task with rule-determined per-input combinatorial structure: theorem-proving (proof-step graph), symbolic-math evaluation (expression-DAG), constraint-satisfaction.

**Complexity:**
- Forward: `O(K · |E(x)| · d²)`, K = average solver iterations (typically 20–40). Closest primitive: stacked GAT, `O(L · |E(x)| · d²)` with explicit `L` layers.
- Backward: `O(K_b · |E(x)| · d²)` for the implicit-diff linear solve. Memory `O(|E(x)| · d)` — independent of K, the key DEQ advantage.
- Incremental update: not applicable — fixed point must be re-solved on input change. **This is the major weakness vs. NNUE.**

**Scout-scale falsification test:** Replace the top two attention blocks of i242 with a single IQFP layer of matching parameter count. Cap forward solver iterations at K=20; use forward-iteration as fallback if Broyden diverges. Train 12 epochs on 173k positions, RTX 3070. Track (a) CRTK class-1 FP rate, (b) average forward iterations K, (c) wall-clock per epoch. Primitive **works** if FP rate improves ≥ 10 % relative on positions where Leela-style 3-ply lookahead matters (CRTK subcategory with tactical depth ≥ 3) *and* mean K < 25. Primitive **fails** if K diverges (> 50) on > 5 % of positions or if FP-rate improvement is uniform across tactical depth (suggesting the win is from extra params, not adaptive depth).

**Failure mode catalogue:**
- *Hidden rebrand of "very deep GAT with weight tying"*: if the typical K converges in 2–3 iterations, IQFP is just a weight-tied 3-layer GAT and the implicit-diff machinery is overhead.
- *Solver divergence on tactical chaos*: positions with many checks/queen-exchanges may yield iteration maps `g` whose Jacobian has spectral radius ≥ 1; the fixed point does not exist. Standard DEQ fix is a spectral-normalization regularizer, which trades expressivity for stability.
- *Wall-clock catastrophe at MCTS-node scale*: chess engines call eval at every MCTS node; IQFP's per-call K-iteration cost is far worse than a feedforward net even if asymptotically constant in memory. This may be a "scout-scale-only" primitive — useful for offline evaluation studies, not for engine deployment. **Flag explicitly.**

**Status:** proposed

---

## Recommendations

**Pilot order, with thresholds for proceed/cut:**

1. **First — DSA-LHN (2 GPU-hours).** Implement as a custom `torch.autograd.Function`. The O(Δ) update is the property the user explicitly named as "the thing to chase". If a 3× sequential-inference speedup on a 1000-position MCTS-style trace materializes without > 5 % FP-rate regression, this primitive is the user's strongest result and can be sent as a standalone torch.nn proposal regardless of the rest. If the speedup is < 1.5× because of autograd-tape overhead, the project becomes a CUDA-kernel project, not a primitives project — **cut**.

2. **Second — LM-LPP (3 GPU-hours; Triton kernel optional).** This is the highest-novelty proposal under the user's weighting (content-dependent connectivity) and is testable as a drop-in. If at K=4 the CRTK class-1 FP-rate beats both i242 and a "GAT-with-legal-mask" ablation by ≥ 10 %, the Neumann-resolvent formulation is the structural win and the primitive is genuinely new. If the GAT-with-legal-mask ablation matches, the win is the inductive bias (the mask), not the operator, and the proposal should be downgraded to "an inductive bias for sparse attention" — **cut as primitive**.

3. **Third — CIEL (1 GPU-hour).** Cheap to implement, gives an exact-antisymmetry guarantee. Worth running even if 1 and 2 succeed, because it interacts independently. If at matched parameters it underperforms the augmented baseline by > 3 %, the primitive is theoretically clean but practically subsumed by augmentation — **publish as a note, not a flagship**.

4. **Fourth — RG3T (4 GPU-hours; requires Triton or numba kernel).** High-risk because of parameter explosion in T. Run only after confirming via a quick dense-mask ablation that ternary structure actually helps on pin/skewer puzzles. If the dense ablation matches RG3T, the primitive is a sparse-compute optimization, not a new op — **cut**.

5. **Fifth — IQFP (5 GPU-hours, high implementation effort).** Engine-scale-only candidate. Run only if 1–4 collectively succeed and the user has time. If solver divergence exceeds 5 % of positions, do not pursue further. **Flag as scout-scale exploratory, not engine-deployable.**

**Benchmarks that change recommendations:** if the user's CRTK class-1 evaluation harness is not yet stable (i.e., baseline FP-rate variance across seeds is > 5 %), every proposal becomes untestable at scout scale and the user should fix evaluation first.

---

## Caveats

- I treated `run_blocking_subagent` and `enrich_draft` as unavailable because they are not present in this environment's tool set; the report is based directly on twelve web_search/web_fetch passes through 2024–2026 literature on selective SSMs, content-aware sparse attention, dynamic GNNs, hypergraph NNs, DEQs, hypernetworks, color-equivariant CNNs, and Stockfish NNUE internals. If the missing tools become available, the proposal most worth targeting with a deeper-source pass is **LM-LPP**, specifically whether resolvent-style propagation over a stop-gradient adjacency has been published as a primitive (suspected near-miss: graph-signal-processing literature, 2022–2024).
- No empirical numbers in this report are measured results. Every "≥ 10 % FP-rate" or "3× speedup" threshold is a *prediction expressed as a falsification criterion*, not a claim of achieved performance. The user's own scout runs are the only valid evidence.
- All five proposals survived a first-pass self-audit, but two have explicit, named failure modes that would reduce them to existing primitives: **LM-LPP at K=1** is GAT-with-mask, and **RG3T** without the sparse-CUDA win is triplet-attention. The user should design the ablations such that these collapses are detectable.
- **What I cut during self-audit (three explicitly retired candidates):**
  - *Mamba-on-board with content-dependent scan order over the legal-move DAG.* Initially attractive (selective-SSM + content-determined connectivity), but the scan can be expressed as a topological-order pass of a content-masked GRU, and Mamba-2/SSD (Dao & Gu, 2024) already shows SSMs are structured masked matrix mixers — so a board-DAG variant decomposes into "topological sort + masked GRU" and fails the non-decomposability test.
  - *Piece-type MoE with attention routing.* Routing experts by piece type sounds chess-specific but is just MoE with a non-learned routing function — and a non-learned router means the routing decision is data preprocessing, not a primitive. Decomposes trivially into `torch.gather` + per-expert `nn.Linear`.
  - *DEQ-over-attention-mask (a generic implicit attention layer).* Functionally distinct from IQFP only because IQFP fixes the connectivity to the legal-move graph; the *general* DEQ-over-attention is just a published DEQ variant (multiscale DEQ, 2020) and adds nothing new.
- The user's heaviest-weighted property — content-dependent connectivity that *changes per token per forward pass* — is genuinely satisfied by LM-LPP, RG3T, and IQFP. DSA-LHN satisfies a weaker but more *practically valuable* property (bounded-change incremental update). CIEL is the outlier and is included because the chess color-swap involution is the most under-exploited symmetry in chess networks and is operator-level, not encoding-level.
- Finally: all five proposals are HIGH-RISK as requested. Self-audit suggests **DSA-LHN and CIEL** have the highest probability of surviving rigorous review as actual primitives; **LM-LPP and IQFP** are the most novel if they survive; **RG3T** is the most likely to collapse to "sparse compute around an existing op." Honest base rate: in this kind of literature scan, perhaps 1–2 of 5 such proposals are eventually accepted as primitives after the first round of reviewer pushback.