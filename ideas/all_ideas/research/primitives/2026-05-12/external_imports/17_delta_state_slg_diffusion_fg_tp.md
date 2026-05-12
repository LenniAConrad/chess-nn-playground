# Five Novel Neural-Network Primitives for Chess Evaluation

**TL;DR**
- The five proposals that survive the "no-rebrand" filter are: **DeltaState** (stateful incremental accumulator generalising HalfKA), **Sheaf-on-Legal-Graph diffusion (SLG)** (sheaf diffusion whose edge set is a deterministic function of the input board), **FiniteGroup Tensor Product (FG-TP)** for the chess group D4 × Z2_color, **Reversible Selective Scan (RSS)** (Mamba-style scan with an explicit O(1) inverse step), and **Max-Plus Morphological Convolution (MPC)** (tropical-semiring convolution).
- Two of the five (**DeltaState**, **RSS**) target the inference-speed / O(1)-incremental-update axis; two (**SLG**, **FG-TP**) target the discrimination axis where i242's full-decomposed attention failed; **MPC** is a wildcard with the cleanest non-decomposition argument (different semiring → different gradient class).
- Three primitives have closely related 2020–2025 prior work and must therefore be framed as *"underexplored primitives for chess"* rather than fresh inventions (Sheaf NN: Hansen & Gebhart 2020 / Bodnar et al. 2022; FG-TP: Carroll & Beel 2020 FGNN, e3nn family; Mamba: Gu & Dao 2023). Only **DeltaState** and **RSS** make defensible "new primitive" claims, and only because they expose a non-functional, *stateful/reversible* op interface that PyTorch's compute graph currently does not carry.

---

## Key Findings

1. **Inference speed is the unambiguous win condition.** Stockfish's NNUE achieves microsecond evaluation precisely because the HalfKA first layer is an *incrementally updated accumulator* maintained on the search stack — only the changed feature indices are touched per move. There is no equivalent op in `torch.nn`: `nn.EmbeddingBag(mode='sum')` recomputes from scratch on every call; it has no `forward_delta(old_idx, new_idx)` interface and no notion of state continuity across calls. This is the cleanest place to invent a primitive.

2. **i242's failure of full decomposed attention at scout scale is consistent with the literature on input-dependent sparsity.** Recent work (SeerAttention, Sparse Adaptive Connection) shows that *content-dependent* sparsity patterns improve the Lipschitz constant of softmax attention by reducing dispersion; content-*independent* masks (the most common chess attention mask) do not. The chess legal-move graph is the canonical content-dependent connectivity, but it has only been used inside off-the-shelf GAT layers (Alwer & Plaat 2023; "Enhancing Chess RL with Graph Representation," arXiv 2410.23753) — never as a non-decomposable sheaf operator with per-edge restriction maps.

3. **Group structure remains under-exploited.** Finite Group Equivariant Neural Networks (Carroll & Beel 2020, arXiv 2009.05027) handle arbitrary finite groups but were demonstrated on checkers, not on the full chess group (D4 × color-involution × bishop-color sublattice). No primitive currently exists that bakes this product group's irrep decomposition into a tensor-product op — yet the machinery is well understood for SO(3) and the symmetric group (Pearce-Crump 2022, "Brauer's Group Equivariant Neural Networks"; Gibson/Tubbenhauer/Williamson 2024 on equivariant NN + PL representation theory).

4. **State-space scans now dominate the incremental-update conversation outside chess.** Mamba/S6 (Gu & Dao 2023; arXiv 2312.00752) introduces a data-dependent selective scan with linear training-time complexity and RNN-like O(1) inference per token. None of the published Mamba variants (Graph Mamba, STG-Mamba, DG-Mamba, HeteGraph-Mamba) are *algebraically reversible* — they cannot run `unmake_move` in O(1). That is the missing piece for game-tree search and the structural opening for **RSS**.

5. **Tropical / max-plus operators are not currently `torch.nn`.** Min-Max-Plus Networks (Luo 2021), Semiring Activations (arXiv 2405.18805) and UltraLIF (arXiv 2602.11206) all argue that the tropical semiring genuinely yields a different operator class — the gradient flows through `argmax` rather than through a smooth softmax, so the computation graph is structurally distinct from any composition of `Conv2d + ReLU + max_pool`. This has the strongest "different gradient class" argument of the five.

---

## Details — Five Proposals

### primitive_delta_state

**Name:** DeltaState — Stateful Reversible Index Accumulator

**One-line claim:** A primitive carrying explicit accumulator state with O(k) incremental update, O(N) refresh, and a reversible delta operation that produces identical gradients to the refreshed forward.

**Mathematical signature:**
State `h ∈ ℝᵈ`. Inputs: an active-index multiset `S ⊂ {1,…,F}` and an embedding table `W ∈ ℝ^{F×d}`. Three callable ops with shared parameters:
- `forward(S) : h = Σ_{i∈S} W[i]`  (shape: `d`)
- `apply_delta(h, S_remove, S_add) : h' = h − Σ_{i∈S_remove} W[i] + Σ_{i∈S_add} W[i]`
- `inverse_delta(h', S_remove, S_add) : h = h' + Σ_{i∈S_remove} W[i] − Σ_{i∈S_add} W[i]`

Backward through any sequence of these three calls must produce gradients on `W` identical (bitwise, modulo summation order) to those produced by `forward(S_final)`.

**Why this does not decompose into existing PyTorch ops:**
The closest existing op is `nn.EmbeddingBag(mode='sum')`. EmbeddingBag is *functional and stateless*: each call recomputes the entire sum, and PyTorch's autograd graph has no node type for "carry mutable state across forward calls while preserving gradient consistency." The primitive's distinguishing property is the *triple-interface contract* (forward / apply_delta / inverse_delta) backed by a state object that participates in autograd. You can simulate it with manual `torch.no_grad` bookkeeping plus a re-forward at loss time, but that is the *decomposed* version — it has 2× the FLOPs of the primitive at inference and forces a refresh on every backward, defeating the purpose. A proper primitive would expose this contract to compilers (`torch.compile`, TorchScript) so that the delta path is the *graph itself*, not a Python wrapper.

**Chess-specific motivation:**
HalfKA's O(1) accumulator is precisely this op, hand-coded in C++ inside Stockfish (`AccumulatorStack`, `AccumulatorCaches`/Finny tables; see chessprogramming.org/Stockfish_NNUE). It is the single reason NNUE evaluates in microseconds on CPU. Generalising it as a PyTorch primitive would let researchers train and deploy NNUE-style heads without leaving the framework, and would expose `apply_delta` to autograd so the *training* signal can include incremental-update consistency losses (e.g., regularising drift between refreshed and delta-updated states).

**Generalisation beyond chess:**
Any sparse-event stream where the active feature set changes by a bounded amount per step: ad-click feature stores, recommender-system user embeddings under streaming updates, online graph node-embedding updates, real-time sensor anomaly detection.

**Complexity (forward, backward, incremental-update):**
- Forward: O(|S|·d) vs `EmbeddingBag` O(|S|·d) (same)
- Backward: O(|S|·d) for the *currently active* indices only, with `sparse_grad=True` semantics extended across delta calls
- Incremental update on bounded-change input: **O(k·d)** where k = |S_remove| + |S_add|, vs O(|S|·d) for any decomposed EmbeddingBag chain. For HalfKA-style chess moves, k ≤ 4 (quiet move) or k ≤ 6 (capture / castle), so for |S| ≈ 32, this is an ~8× constant-factor win plus the autograd-state benefit.

**Scout-scale falsification test:**
Drop `DeltaState` in as the first layer of i243's HalfKA stream, replacing the `EmbeddingBag` accumulator. Training set: 173k positions, 12 epochs, RTX 3070, single seed. Compare two metrics: (a) **wall-clock inference** on a 1M-position eval sweep with sequential move application (the realistic search-tree pattern) — target ≥ 4× speedup vs full refresh; (b) **CRTK class-1 matched-recall near-puzzle FP rate** vs the i243 baseline — must be within ±0.5% (no accuracy regression). Works = both met. Fails = either accuracy regresses > 0.5% or speedup < 2×.

**Failure mode catalogue:**
- (a) **Hidden rebrand:** if implemented as `EmbeddingBag` + Python state, it *is* a rebrand. The primitive must expose `apply_delta` to autograd, not wrap it. Strongest reviewer objection: "this is just sparse SGD with momentum." Counter: SGD-with-momentum has no inverse op and no consistency contract.
- (b) **Numerically unstable:** repeated apply_delta in fp16 accumulates rounding drift; a periodic refresh (every N plies) is required. The primitive should expose `max_drift_plies` as a hyperparameter and assert refresh under a configurable L∞ bound.
- (c) **Too slow even if it works:** the GPU launch overhead per `apply_delta(k=4)` may dominate the actual compute for small k; in that regime, batched delta application across positions is mandatory, breaking sequential-search semantics. Honest answer: this is a CPU/edge primitive, not a GPU primitive.

**Status:** proposed

---

### primitive_slg_diffusion

**Name:** Sheaf-on-Legal-Graph Diffusion (SLG)

**One-line claim:** Sheaf-Laplacian diffusion where both the edge set and the per-edge restriction maps are deterministic functions of the input board.

**Mathematical signature:**
For a board `x`, deterministically derive the legal-move directed graph `G(x) = (V=64, E(x))` and a stalk dimension `k`. Node features `X ∈ ℝ^{64×k×c}` (c channels of k-dim stalks). For each edge `(u→v) ∈ E(x)`, derive restriction maps `F_{u◁(u→v)}, F_{v◁(u→v)} ∈ ℝ^{k×k}` from `(piece-type(u), piece-type(v), move-type)` via a small shared MLP. Build the sheaf Laplacian `L_F ∈ ℝ^{64k × 64k}` (block-sparse). Forward: `Y = (I − α L_F) X` (one diffusion step). Multiple steps stack.

**Why this does not decompose into existing PyTorch ops:**
The closest existing op is sparse-mask attention or GAT. Two structural differences: (1) The sparsity pattern `E(x)` is a *combinatorial* function of x (legal moves), not a soft-thresholded learned mask, so gradient does not flow through edge selection — only through stalk values and restriction maps. PyTorch's masked-attention assumes a *given* mask tensor as input; SLG's "mask" is computed *inside* the op from a non-differentiable rule, then per-edge restriction maps act as `k×k` linear operators rather than scalar attention coefficients. (2) The sheaf Laplacian assembly is a node-coboundary operation (`L_F = δᵀ δ`), which is not a primitive in `torch_geometric` either — it's a different algebraic object than the graph Laplacian. Bodnar et al. (2022) note that sheaf diffusion provably escapes the homophily / oversmoothing regime that standard GAT/GCN suffer from, which is exactly what i242 hit.

**Chess-specific motivation:**
The legal-move graph is the single most chess-specific structural fact and is currently used only as a node-edge graph inside GAT/GIN (Alwer & Plaat 2023). Sheaf restriction maps let a knight's attack and a bishop's attack carry *different* stalk transformations along the *same* abstract edge; this is the natural place to encode piece-type semantics without bloating channel count.

**Generalisation beyond chess:**
Any domain with input-determined heterogeneous-edge graphs: molecular reaction networks where bond types vary per molecule, knowledge graphs with typed edges, traffic networks with time-varying lane availability.

**Complexity (forward, backward, incremental-update):**
- Forward: O(|E(x)|·k²·c) vs full attention O(64²·c·h). |E(x)| is typically 30–60 in middlegame, so ~10× sparser than 64² = 4096 dense.
- Backward: O(|E(x)|·k²·c), same order.
- Incremental update on bounded-change input: **not applicable in general** — `E(x)` changes globally after most moves (a discovered attack flips many edges). A partial-update variant restricted to "quiet quiet moves" is conceivable but fragile; honestly mark this as a *forward-throughput* primitive, not an incremental one.

**Scout-scale falsification test:**
Drop SLG (1–2 diffusion steps, k=4 stalks, c=32) in as the spatial mixing block of i242 *replacing* the chess-decomposed attention. Hold all other hyperparameters fixed. Train on 173k positions, 12 epochs, single seed, RTX 3070. Primary metric: CRTK class-1 matched-recall near-puzzle FP rate vs i193 (conv-only parent). Works = beats i193's near-puzzle FP rate by ≥ 0.5 absolute pp. Fails = matches or under-performs i193 like i242 did. Secondary check: ablate restriction maps to identity (this reduces SLG to a vanilla GCN on the legal-move graph) — the primitive is only justified if non-identity restriction maps beat identity.

**Failure mode catalogue:**
- (a) **Hidden rebrand:** if restriction maps are forced to identity, SLG collapses to a GCN on a content-determined graph, which decomposes into `scatter_add` + linear. Must demonstrate non-identity restriction-map gain.
- (b) **Numerically unstable:** sheaf Laplacian eigenvalues can exceed 2; the diffusion step `I − α L_F` blows up unless α is bounded by max-eigenvalue, which is data-dependent. Need power-iteration normalisation per forward, costing ~5 extra mat-vecs.
- (c) **Too slow even if it works:** Python-side graph construction per board kills batch parallelism. The primitive needs a CUDA kernel for "legal-move graph from bitboard" or a precomputed cache; this is a real engineering cost.

**Status:** proposed; **flagged as "underexplored primitive for chess"** — Sheaf NN (Hansen & Gebhart 2020; Bodnar/Di Giovanni/Chamberlain/Liò/Bronstein 2022) is the genuine prior art.

---

### primitive_fg_tp

**Name:** Finite-Group Tensor Product (FG-TP) for the chess group

**One-line claim:** A bilinear tensor-product op whose parameter sharing is constrained to the irrep block-diagonal of the chess group G = D4 × Z2_color, automatically guaranteeing exact equivariance.

**Mathematical signature:**
Group `G` with regular representation decomposed into irreps `{ρ_i}` of dimensions `{d_i}`. Inputs `x, y ∈ ℝ^{|G|}` (or stacks thereof) viewed as G-equivariant features. The op computes `z = x ⊗_G y`, where the tensor product is restricted to the equivariant subspace via Clebsch-Gordan-like coefficients for finite groups. Concretely: project `x` and `y` to irrep components `x_i, y_i`, take outer products, contract with stored Clebsch-Gordan tensors `C^{ij}_k`, and reassemble. Parameter count: only the irrep-to-irrep coupling weights, ~O(Σ_{i,j,k} d_i d_j d_k / |G|), much smaller than a full bilinear map of size |G|³.

**Why this does not decompose into existing PyTorch ops:**
The closest existing op is `nn.Bilinear`. `Bilinear` has no constraint on its weight tensor; FG-TP's weight tensor is *forced* to live in the equivariant subspace by construction (parameter sharing inside the op kernel, not enforced via a loss). Although in principle one can write `y = einsum(W_sym(θ), x ⊗ x)` with a manually symmetrised `W_sym`, this is the decomposed version — it materialises the full bilinear tensor in memory and only enforces symmetry numerically. The FG-TP primitive operates on irrep components directly and never instantiates the full tensor; it has lower FLOPs *and* a different parameter shape (one weight per irrep coupling triple, not one per (i,j,k) index triple). Carroll & Beel's FGNN (2020) handles arbitrary finite groups by *averaging over the group orbit*, which is O(|G|) per forward pass; the irrep-decomposed FG-TP is the algorithmically cheaper, computation-graph-distinct alternative.

**Chess-specific motivation:**
The relevant group for chess is not `D4` alone (that's just board symmetry). It is `D4 × Z2_color` (color-swap involution, which sends piece P at square s to ¬P at flip(s)), and modulo castling/en-passant state. None of the i234/i242/i243 series exploits color-swap as a group action at the *operator* level — only as a data augmentation. An equivariant primitive would cut effective parameter count by ≈ |G| = 16 and force the network to never learn a color-asymmetric pattern, which is the right inductive bias for evaluation (eval(x) = −eval(color_swap(x))).

**Generalisation beyond chess:**
Any domain with a finite product symmetry group: crystallography (space groups), molecular point groups (e.g., octahedral C60 derivatives), card games, lattice models in physics.

**Complexity (forward, backward, incremental-update):**
- Forward: O(Σ_i d_i · c²) ≪ O(|G| · c²) of orbit-averaging FGNN, and ≪ O(c⁴) of unconstrained `Bilinear`
- Backward: same order as forward
- Incremental update on bounded-change input: not applicable (this is a per-token bilinear, not a stateful op)

**Scout-scale falsification test:**
Replace the cross-stream interaction term in i243 (HalfKA + dual-stream) with FG-TP for `G = D4 × Z2_color`. Equivariance audit first: under color-swap of input, output must satisfy `f(τx) = −f(x)` to machine precision (no learned hack). Then 173k × 12 epochs, single seed. Metric: CRTK class-1 matched-recall near-puzzle FP rate vs i243. Works = equivariance holds AND near-puzzle FP rate improves by ≥ 0.3 pp at equal or lower parameter count. Fails = either equivariance test fails (the irrep decomposition was wrong) or accuracy regresses.

**Failure mode catalogue:**
- (a) **Hidden rebrand:** if implemented as "Bilinear + group-symmetrisation loss," it decomposes. The primitive must enforce equivariance *structurally* via irrep-restricted weights, with a unit test that randomises weights and still passes the equivariance audit.
- (b) **Numerically unstable:** Clebsch-Gordan coefficients for non-trivial groups can be ill-conditioned; care needed in the change-of-basis matrices. Standard fix: precompute via SVD of the projection operator.
- (c) **Too slow even if it works:** for very small `c`, the irrep bookkeeping overhead exceeds the FLOP savings. A break-even analysis vs full Bilinear is mandatory.

**Status:** proposed; **flagged as "underexplored primitive for chess"** — the irrep-decomposed tensor product is well-established for SO(3) (Thomas et al. TFN, Geiger & Smidt e3nn, NequIP). The chess-group instantiation appears not to have been published. Carroll & Beel 2020 (FGNN, arXiv 2009.05027) is the closest published precedent on game boards but uses orbit-averaging, not irrep blocks.

---

### primitive_rev_scan

**Name:** Reversible Selective Scan (RSS)

**One-line claim:** A Mamba-style selective state-space scan whose recurrence is algebraically reversible, so undoing a move is O(1) instead of replaying from root.

**Mathematical signature:**
Sequence of moves `u_1, …, u_T`. Selective SSM parameters `A_t, B_t, C_t` are produced as data-dependent functions of `u_t` (Mamba/S6 mechanism). State recurrence with a *diagonal* `A_t` (with strict numerical bounds `|A_t[i]| ∈ [ε, 1]` and `A_t[i] ≠ 0`):
- Forward: `x_t = A_t ⊙ x_{t-1} + B_t · u_t`
- Output: `y_t = C_t · x_t`
- **Inverse step:** `x_{t-1} = (x_t − B_t · u_t) / A_t`

Reversibility theorem: given `x_t, u_t, A_t, B_t`, `x_{t-1}` is recovered exactly (in real arithmetic) and to within `O(ε⁻¹·eps_float)` (in finite precision), where `ε` is the lower bound on `|A_t[i]|`.

**Why this does not decompose into existing PyTorch ops:**
The closest existing primitive is Mamba/S6's `selective_scan` (Gu & Dao 2023, arXiv 2312.00752). The Mamba primitive only exposes a `forward(x_init, u_1:T)` interface; it has no `inverse_step` op, and its CUDA kernel does not maintain the conditions (`A_t` strictly bounded away from zero) needed for numerical invertibility. RSS structurally adds (i) a parametrisation constraint `|A_t[i]| ∈ [ε, 1]` enforced inside the op kernel via a clipped softplus reparametrisation, (ii) a new `inverse_step` graph node with its own gradient, and (iii) a *bidirectional state stack* memory model where the search tree's branching exploits inverse steps. None of these can be expressed as a composition of `selective_scan` calls — `selective_scan` is functional, RSS is bidirectional.

**Chess-specific motivation:**
Alpha-beta search performs ~10⁶ make-move / unmake-move pairs per second. A stateful neural evaluator that supports unmake in O(1) is the deep-learning analog of bitboard make/unmake. RSS provides this: the search tree's principal-variation continuation can branch from any node by inverse-step + forward-step rather than re-encoding the line from move 1. This is the missing primitive that would let SSM-based evaluators compete with NNUE's accumulator on inference throughput inside a real search.

**Generalisation beyond chess:**
Any tree-search domain over event sequences: theorem proving (proof-tree backtracking), neurally guided program synthesis (AST mutations), MCTS with deep priors over action histories.

**Complexity (forward, backward, incremental-update):**
- Forward (full sequence T): O(T·d) (linear in T, like Mamba)
- Backward: O(T·d) (standard reverse-mode through scan); RSS can additionally backprop using the inverse op to *eliminate activation storage* (constant memory in T), like reversible networks
- Incremental update on bounded-change input: **O(1)·d** per single make/unmake, vs O(T·d) for re-encoding from root in vanilla Mamba.

**Scout-scale falsification test:**
Build a minimal evaluator: NNUE-style HalfKA accumulator front-end + RSS scan over the last 16 plies + small MLP head. Train on 173k positions, 12 epochs, single seed. Two tests: (a) **Reversibility audit:** for 10k random move/unmake pairs, the state after `inverse_step(forward_step(x, u), u)` must satisfy `||x' − x||∞ < 10⁻³` in fp32. Numerical fail = primitive is broken regardless of accuracy. (b) **Search-mode inference benchmark:** simulate a depth-6 alpha-beta over 1k root positions using make/unmake; RSS must achieve ≥ 5× speedup vs root-replay Mamba. Accuracy must not regress > 0.5 pp on CRTK class-1 near-puzzle FP rate.

**Failure mode catalogue:**
- (a) **Hidden rebrand:** vanilla Mamba with a Python loop calling `selective_scan` repeatedly is the decomposed version and the obvious null hypothesis. RSS must beat it on wall-clock in search mode, otherwise it's just Mamba + bookkeeping.
- (b) **Numerically unstable:** division by `A_t` with `|A_t[i]| ≈ ε` blows up. The `ε` lower-bound is a load-bearing hyperparameter; expect a 10–20% accuracy hit relative to unconstrained Mamba in exchange for the reversibility property, and budget for it.
- (c) **Too slow even if it works:** a CUDA kernel for inverse_step needs to be written; without it, falling back to PyTorch ops nullifies the speedup. Honest answer: this is an engineering bet, not a free lunch.

**Status:** proposed. Closest published precedent is McCallum et al. 2025 ("Reversible Deep Equilibrium Models," arXiv 2509.12917), which establishes the reversible-fixed-point machinery but for DEQ, not for selective scans. RSS is, to the best of my survey, genuinely new as a primitive; the underlying observation (diagonal Mamba is invertible if `A_t ≠ 0`) is folklore.

---

### primitive_mpc

**Name:** Max-Plus Morphological Convolution (MPC)

**One-line claim:** A convolutional primitive in the tropical (max-plus) semiring, whose kernel applies `max(w + x)` rather than `sum(w · x)`, with subgradient through the argmax index.

**Mathematical signature:**
Input `X ∈ ℝ^{B×C_in×8×8}`, kernel `W ∈ ℝ^{C_out×C_in×k×k}`. Forward (per output channel, position, batch):

`Y[b, c_out, i, j] = max_{c_in, di, dj} ( X[b, c_in, i+di, j+dj] + W[c_out, c_in, di, dj] )`

Backward: subgradient flows only through the argmax-attaining `(c_in*, di*, dj*)` to both `X` and `W`. Optionally, a *Min-Plus* dual op `Y = min_{...} (X − W)` for opponent-perspective tactical features. A learnable temperature `ε` can soft-relax `max` to `logsumexp` for warm-start, with `ε → 0` recovering hard max-plus (UltraLIF-style ultradiscretization, arXiv 2602.11206).

**Why this does not decompose into existing PyTorch ops:**
The closest existing ops are `nn.Conv2d` and `nn.MaxPool2d`. Conv2d is convolution in the `(+, ×)` ring. MaxPool2d is max in the tropical semiring but with *fixed identity weights* over a kernel window. MPC is max-plus with *learnable weights* — i.e., it's tropical convolution. There is no PyTorch op for tropical convolution; you can simulate it as a hand-rolled sliding-window `max(...)` of `(X_patch + W)`, but the gradient through `argmax` is not natively supported as a vectorised op (the gather pattern is data-dependent). The gradient class is provably different: Conv2d gradients are dense (every weight gets a gradient contribution from every position); MPC gradients are 1-sparse per output element (only the argmax-winning weight gets a contribution). This is a fundamentally different gradient flow and complexity class.

**Chess-specific motivation:**
Many chess features are *decisive-pattern* rather than averaged: "the best attacker on the king" matters, "the worst defender of a square" matters, the *single sharpest line* in a tactical position dominates the evaluation. Sum-pooling Conv2d averages these signals; MPC preserves them. The piece-square table itself is morally a `max` over candidate attacker/defender contributions, not a sum. Hard-negative discrimination (CRTK class-1) is exactly where MPC should help, because near-puzzle positions are decided by a single sharp line.

**Generalisation beyond chess:**
Mathematical morphology (image processing, since 1980s, Serra; modernised in Charisopoulos & Maragos 2017, Mondal et al. 2019 "Morphological Networks"), shortest-path-style reasoning (the max-plus algebra *is* the algebra of shortest paths), neural decoders for codes over tropical algebras.

**Complexity (forward, backward, incremental-update):**
- Forward: O(B·C_out·C_in·k²·H·W), same as Conv2d but with `max+` instead of `mul+sum` — same asymptotic FLOPs, ~2× memory-traffic-dominated wall-clock in current PyTorch because no fused kernel exists
- Backward: O(B·C_out·H·W·k²) for the argmax-index gather, sparser than Conv2d's backward
- Incremental update on bounded-change input: not applicable in general; specialised variant for HalfKA-style sparse input changes could be devised but is not the main pitch.

**Scout-scale falsification test:**
In i193 (conv-only baseline), replace the *last* conv block with an MPC block of identical channel count. Train 173k × 12 epochs, single seed, RTX 3070. Use logsumexp soft-max with `ε` annealed from 1.0 → 0.05 over training. Primary metric: CRTK class-1 matched-recall near-puzzle FP rate vs i193 with that block left as Conv2d. Works = improvement ≥ 0.5 pp on class-1 FP rate AND non-regression on aggregate PR AUC (we explicitly want the *hard-negative* gain, not the easy-negative gain). Fails = no improvement on class-1 or aggregate regresses > 0.3 pp. A secondary ablation: replace *all* convs with MPC — almost certainly worse, because averaged features matter for material; the primitive should be most useful at the deepest layer.

**Failure mode catalogue:**
- (a) **Hidden rebrand:** "trainable max-pool" — but standard MaxPool has no learnable weights and no kernel weight gradient. The full primitive's parameter count and gradient pattern differentiate it.
- (b) **Numerically unstable:** hard `max` has a 1-sparse gradient → many weights never get gradient → dead filters. Mitigation is the `ε`-softmax warm-start; this is exactly the UltraLIF/Mondal et al. recipe. Still, expect higher gradient variance and lower learning rate.
- (c) **Too slow even if it works:** without a fused CUDA kernel, MPC is ~3–5× slower than Conv2d at the same FLOPs in current PyTorch. The primitive's case rests on accuracy on hard negatives, not speed.

**Status:** proposed; **flagged as "underexplored primitive for chess"** — morphological neural networks (Mondal et al. 2019; Charisopoulos & Maragos 2017) and tropical layers (UltraLIF 2026, Semiring Activations arXiv 2405.18805, Min-Max-Plus Networks arXiv 2102.06358) are the prior art. MPC has, to my knowledge, never been evaluated on chess.

---

## Ranking

On the four ranking criteria (plausibility of novelty / RTX-3070 demonstrability / inference-speed advantage / generalisation), my ordering:

| # | Primitive | Novelty | Demo on 3070 | Inference speed | Generalises |
|---|---|---|---|---|---|
| 1 | **DeltaState** | High (stateful contract is genuinely missing) | Easy | **Huge** (CPU/NNUE-class) | Streaming / recsys |
| 2 | **RSS** | Medium-high (Mamba extension) | Medium (needs CUDA inverse) | High in search mode | Tree-search domains |
| 3 | **SLG** | Medium (sheaf NN exists but not on legal moves) | Easy | Neutral / negative | Molecules, KG |
| 4 | **MPC** | Medium (morphological NN exists) | Easy | Negative without fused kernel | Vision, shortest-path |
| 5 | **FG-TP** | Lower (irrep TP well-established) | Medium | Neutral | Crystallography, physics |

**Devil's advocate self-audit (top 2):**
- **DeltaState** could be argued to "just be" `EmbeddingBag` + a Python state dict. The defense: the *autograd contract* between forward/apply_delta/inverse_delta is what's new, and it is not currently expressible without a custom autograd.Function and a state-carrying Module. The C++-side Stockfish accumulator is the existence proof that this is a distinct op; the gap is in PyTorch, not in chess. Kept.
- **RSS** could be argued to be "just Mamba with a Python `unmake` wrapper." The defense rests on the constrained parametrisation (`|A_t| ∈ [ε,1]` enforced inside the kernel) and the new `inverse_step` graph node with its own backward. If the diagonal-A constraint turns out to be too restrictive for accuracy, RSS degenerates and should be dropped in favor of plain Mamba. The reversibility audit is the falsifier. Kept conditionally.

---

## Recommendations

**Stage 1 (week 1–2, lowest risk):** Implement **DeltaState** as a pure-PyTorch `autograd.Function` with a Python state object. Run the i243 drop-in experiment. Threshold: ≥ 2× wall-clock speedup with ≤ 0.5 pp CRTK class-1 regression. If hit → proceed to a CUDA/CPU SIMD kernel. If missed by accuracy → debug delta-vs-refresh numerical consistency; if missed by speed → the primitive is fundamentally a CPU/edge op, pivot to CPU benchmarking.

**Stage 2 (week 3–4, medium risk):** Implement **SLG** with `torch_geometric`'s scatter primitives. Run the i242 drop-in. Threshold: must beat i193 on CRTK class-1 by ≥ 0.5 pp. If yes → invest in CUDA kernel for legal-move graph extraction. If no → fall back to plain GCN baseline and report that sheaf restriction maps do not lift over GCN at scout scale.

**Stage 3 (week 5–6, higher risk):** Implement **MPC** at one layer of i193. Threshold: ≥ 0.5 pp class-1 lift, no aggregate regression. Soft-to-hard `ε` annealing is mandatory. If the gain is real, scale to two MPC layers.

**Defer until LC0-scale data is available:** **RSS** and **FG-TP**. RSS only makes sense once a search engine is wrapped around the network, not at static-eval scout scale. FG-TP's parameter savings only pay off at the parameter regime where i243 is already saturated.

**Benchmarks that would change these stagings:**
- If DeltaState's autograd contract turns out to have non-trivial gradient drift under apply_delta, demote it from Stage 1 and pivot to a *no-grad* inference-only delta op + full forward at training. That's still useful but is no longer a candidate `torch.nn.<NewOp>`.
- If i242's attention failure turns out to be data-scale rather than connectivity, SLG won't fix it and should be dropped.
- If CRTK class-1 near-puzzle FP rate stops being a discriminative metric (e.g., new puzzle data changes the regime), re-derive a comparable hard-negative metric before staging.

---

## What I Cut

Four candidates that I evaluated and rejected, with reasons:

1. **Learned positional encoding over piece-square pairs.** Cut because the task explicitly forbids encoding proposals. Even if reframed as "an op that produces position-conditioned biases," it decomposes into `Embedding + add`. Rejected on rule violation.

2. **Mixture-of-Experts router gated on tactical/positional regime.** Cut because MoE gating already exists (Jacobs 1991; Shazeer et al. 2017) and a "chess-tactical" router is just MoE applied to chess — an *architectural* composition, not a new primitive. Rejected on novelty.

3. **Deep equilibrium tactical solver (DEQ on the position).** Cut because DEQ (Bai, Kolter, Koltun 2019) is a well-known primitive and the chess application is an architecture, not a new operator. RevDEQ (McCallum et al. 2025) does add a reversibility property, but I judged that **RSS** offers a sharper chess-specific gain (move-stream reversibility maps directly to make/unmake) than DEQ-on-position reversibility, which has no chess analog. Replaced by RSS.

4. **Hypernetwork that generates conv weights conditioned on phase (opening/middle/endgame).** Cut because hypernetworks (Ha, Dai, Le 2016) are a known primitive; conditioning them on a phase signal is composition. Also borderline-violates the rule about encodings, since a phase feature must come from somewhere. Rejected on novelty + rule risk.

5. **Differentiable top-k attention with content-dependent k.** Cut because MapSelect (OpenReview 2024) and Sparse Adaptive Connection (Li et al. 2020) cover this; "with content-dependent k" is a hyperparameter rebrand inside an already-published primitive class.

---

## Caveats

- **No accuracy numbers are claimed.** Every "≥ 0.5 pp" threshold is a *falsification criterion*, not a prediction. The user's prior i242 result (full chess-decomposed attention underperforming i193 at 173k × 12 epochs) is the only empirical fact I rely on.
- **"Generalisation beyond chess" claims are stated targets, not validated transfers.** Each primitive's non-chess application would need a separate experimental campaign.
- **Three of the five primitives have closely related prior work** (SLG ↔ Sheaf NN; FG-TP ↔ e3nn / FGNN; MPC ↔ morphological NN / tropical layers) and have been honestly flagged as "underexplored for chess" rather than novel. Only **DeltaState** and **RSS** are pitched as defensibly new primitives, and even those rest on existence claims (stateful autograd contract; reversible diagonal Mamba) that a careful reviewer could try to deflate. The falsification tests are designed to expose exactly such deflations.
- **Hardware reality check:** the RTX 3070 / 8 GiB / single-seed / 12-epoch budget is sufficient for falsification but not for a definitive verdict. A primitive that fails at this scale may still win at LC0 scale (10⁹ positions); a primitive that wins at this scale almost certainly transfers. The asymmetry favours scout-scale testing first.
- **No PV / Stockfish-eval / node-count features enter any primitive's compute graph**, per the task's rule 5. The `DeltaState` apply_delta interface uses move metadata (changed indices), which is a structural property of the input encoding, not Stockfish output.
