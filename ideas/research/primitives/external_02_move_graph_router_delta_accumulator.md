# Five Candidate Neural-Network Primitives for Chess Evaluation

## TL;DR

- **Pick two primitives to prototype first**: `DeltaAccumulator` (IDA) and `MoveGraphRouter` (MGR). IDA generalises NNUE's HalfKA accumulator into a reusable stateful operator whose forward cost depends on input *change* rather than input size — the single most chess-aligned structural property. MGR is the only one of the five whose computation graph is genuinely non-decomposable from existing PyTorch ops (the topology, not the weights, is content-determined per forward pass, with stop-gradient on the topology branch).
- **Two are "underexplored for chess" rather than novel**: `Tropical Bilinear` (TBL) and `Equilibrium Energy Primitive` (EEP) overlap with published primitives (Min-Max-Plus Neural Networks, Luo 2021; Deep Equilibrium Models, Bai 2019; Modern Hopfield Networks, Ramsauer 2020). Treat them as chess-domain adaptations whose contribution is mechanism-domain fit, not mathematical newness.
- **One is borderline-novel and may collapse on close inspection**: `KingIndexedSwitchingBank` (KISB) is structurally close to externally-routed MoE; it survives only if the falsification test demonstrates measurable advantage over a plain `nn.Embedding`-conditioned linear. Drop if not.

## Key Findings

1. The deep-learning community recognises a primitive when it has (a) a named gradient flow that differs from any naïve composition, (b) a complexity class that does not match `nn.Linear` / `nn.Conv2d` / `nn.MultiheadAttention`, and (c) a reusable signature. Attention, Mamba's selective scan, MoE-gate, DEQ implicit layers, modern Hopfield layers, RWKV time-mix, and Hyena's implicit long convolution all clear the bar; "tropical matmul" and "Reynolds projection" are borderline (recognised in niche literatures, decomposable into standard ops when scrutinised).
2. The structurally chess-specific facts most worth exploiting are, in order of leverage: incremental input change (HalfKA), content-determined legal-move connectivity, king-conditioned sparsity, color-swap involution, piece-type relabelling. Attention masks (even legal-move masks) are *not* a new primitive; mask-as-input is composition with `nn.MultiheadAttention`.
3. The Sharir & Anandkumar 2023 "Incrementally-Computable Neural Networks" paper (arXiv 2307.14988) is the closest prior art to the HalfKA-generalisation direction. It uses vector quantisation to propagate sparsity through nonlinearities; for chess specifically, exact propagation through `ClippedReLU` regimes is possible without VQ because the activation is piecewise-linear with two breakpoints — this is the wedge that keeps IDA non-trivial.
4. At the 173k-position × 12-epoch × single-RTX-3070 scale flagged from the i242 scout, primitives whose advantage requires data-hungry training (Hyena, IEDA, large-state Mamba-2 selective scan with N≥32) should be expected to *underperform* convolutional baselines, just as i242's full transformer underperformed i193. Two of the five proposals (IDA, KISB) are explicitly designed to beat conv on inference-speed first and accuracy second, which is the right trade for this regime.
5. None of the five proposals decompose into "two streams that share weights" or "attention with a new mask" — the disqualifying anti-examples from the spec.

## Details

### 1. primitive_move_graph_router

**Name:** MoveGraphRouter (MGR)

**One-line claim:** A gather-scatter primitive whose sparse adjacency is computed *inside* the op as a non-differentiable function of the input, not supplied as an external mask.

**Mathematical signature:**
Input: features `X ∈ ℝ^{B × N × d}`, discrete board state `S ∈ ℤ^{B × N × k}` (piece-type + square indices, part of the model's input tensor, **not** Stockfish metadata). Output: `Y ∈ ℝ^{B × N × d}`.
Forward:
1. `E_b = LegalEdges(S_b) ∈ ℤ^{2 × m_b}` — a discrete, non-differentiable function returning a per-sample sparse edge list of length `m_b`. `stop_gradient` applied.
2. `Y_{b,i} = Σ_{(i,j) ∈ E_b} φ_θ(X_{b,i}, X_{b,j})` via fused gather → MLP → scatter-add.
Backward: gradients flow through `φ_θ` and through `X` at gathered indices only; zero gradient through `S` and through `E_b`.

**Why this does not decompose into existing PyTorch ops:** Closest existing op is `nn.MultiheadAttention` with a passed `attn_mask`. The decisive difference is that in MGR the topology `E_b` is **constructed inside the forward pass by a discrete domain function of input** and is treated with `stop_gradient`, so the computation graph contains a non-differentiable branch with a per-sample-variable index tensor, not a dense softmax over `N²` slots. This makes its FLOPs `O(m̄ · d)` rather than `O(N² · d)`, and its gradient sparsity pattern is different — the Jacobian is exactly zero off-edge, not "softmax-small."

**Chess-specific motivation:** The legal-move graph is the most chess-specific structural fact: connectivity changes per token per forward pass and is sparse (≈35 edges on average vs `N²=4096`). Standard sparse attention assumes a fixed sparsity pattern; legal-move sparsity is content-determined.

**Generalisation beyond chess:** Any setting whose interaction topology is computed by a deterministic discrete function of state — physical contact graphs in differentiable simulation, abstract-syntax-tree edges in code models, dynamic protein-residue contact maps under conformation change.

**Complexity (forward, backward, incremental-update):**
- Forward: `O(m̄ · d)` vs `nn.MultiheadAttention` `O(N² · d)`; for chess `m̄ ≈ 35`, `N² = 4096`.
- Backward: `O(m̄ · d)`.
- Incremental update on bounded-change input: `O(Δm · d)` where `Δm` is the symmetric difference of legal-move sets between consecutive positions; typically `Δm ≪ m̄`.

**Scout-scale falsification test:** Drop MGR into i242's chess-decomposed-attention slot replacing one attention head with a MoveGraphRouter. Baseline: the published i242 configuration. Metric: CRTK class-1 matched-recall near-puzzle FP rate at 173k positions × 12 epochs, single seed. "Works" = ≥ relative 5% reduction in class-1 FP at matched recall **and** ≥ 30% wall-clock speedup per forward pass on RTX 3070. "Fails" = neither condition holds.

**Failure mode catalogue:**
- *Hidden rebrand:* If implemented as a dense `attn_mask` with `−∞` off-edge instead of true gather/scatter, it collapses to masked attention and the FLOPs claim is false; the reviewer will flag this immediately. Implementation must use `torch.sparse` or `torch_scatter`.
- *Numerically unstable:* `LegalEdges` returns wildly varying `m_b` across the batch, breaking CUDA-graph capture and yielding pathological warp utilisation.
- *Too slow even if working:* For small N (N=64 squares), the kernel-launch overhead and irregular memory access can erase the asymptotic win — Mamba-1's selective scan had this exact problem at small state widths.

**Status:** proposed

---

### 2. primitive_delta_accumulator

**Name:** DeltaAccumulator (IDA)

**One-line claim:** A stateful affine primitive whose forward cost is `O(‖Δx‖₀ · d_out)` instead of `O(d_in · d_out)`, with sparsity propagated exactly through piecewise-linear activations.

**Mathematical signature:**
Stateful tuple `(x_{t−1}, a_{t−1}, r_{t−1})` carried across calls. Inputs at step `t`: `x_t ∈ ℝ^{d_in}` (sparse), weights `W ∈ ℝ^{d_out × d_in}`, bias `b ∈ ℝ^{d_out}`.
Forward:
1. `Δx = x_t − x_{t−1}`; `J = supp(Δx)` (indices with nonzero change).
2. Pre-activation update: `z_t = a_{t−1} + W[:, J] · Δx[J]`.
3. Activation regime tracking: for `ClippedReLU(z)`, maintain `r_t ∈ {below, linear, above}^{d_out}`. Transitions are detected exactly because `ClippedReLU` has only two breakpoints; flip indices form set `K_t`.
4. Output `y_t = ClippedReLU(z_t)`; downstream `Δy = y_t − y_{t−1}` is nonzero only on `K_t ∪ {i : z_t,i changed within linear regime}`.
Backward: standard affine gradient on touched indices; untouched parameters receive zero gradient at step `t` and accumulate over the trajectory.

**Why this does not decompose into existing PyTorch ops:** `nn.Linear` has no state and its forward cost is independent of input sparsity at the autograd-graph level (even sparse-tensor multiplication in PyTorch densifies before the next layer). IDA's primitive identity is the stateful `(x_{t−1}, a_{t−1}, r_{t−1})` triple as a first-class output, with a fused regime-tracking branch that produces a different computation graph than `nn.Linear` followed by `F.hardtanh`. Closest non-PyTorch prior art: Sharir & Anandkumar 2023 (arXiv 2307.14988) for incrementally-computable transformers via VQ; IDA differs by exact (not VQ-approximate) sparsity propagation through `ClippedReLU` regimes.

**Chess-specific motivation:** This directly generalises NNUE's HalfKA accumulator — the property the user identified as "structurally what makes NNUE fast." A typical chess move changes 2–4 input neurons; `‖Δx‖₀/d_in ≈ 0.005`.

**Generalisation beyond chess:** Streaming-edit document scoring (closest analogue: Sharir & Anandkumar's writing-assistant use case, reporting "12.1× median fewer operations" for OPT-125M on atomic edits), real-time sensor analytics with bounded sample-to-sample variation, incremental retrieval re-ranking under document edits, Go and shogi engines using bitboard moves.

**Complexity (forward, backward, incremental-update):**
- Forward: `O(‖Δx‖₀ · d_out + |K_t| · d_out)` vs `nn.Linear` `O(d_in · d_out)`.
- Backward: same as forward.
- Incremental update on bounded-change input: this is the primitive's defining property — sublinear in `d_in`.

**Scout-scale falsification test:** Replace the first `nn.Linear` of i193 (the conv-only parent that beat i242) with an IDA layer. Train identically: 173k positions × 12 epochs, single seed. Two metrics: (a) loss/PR-AUC parity vs baseline (within ±1%), and (b) wall-clock evaluation throughput on a 1000-position search trajectory where positions are linked by single moves. "Works" = matched accuracy **and** ≥ 3× throughput on the linked trajectory. "Fails" = throughput improvement < 1.5× or accuracy regression > 1%.

**Failure mode catalogue:**
- *Hidden rebrand:* Without the regime-tracking branch, IDA collapses to "sparse-input `nn.Linear` with cached output" — already implementable. The regime-tracking through `ClippedReLU` is the part that earns "primitive" status; reviewers will check the second layer also exploits sparsity.
- *Numerically unstable:* In int8 quantised inference (which is the regime NNUE actually uses), float drift between fresh-recompute and delta-update paths must be bit-exact; otherwise the search tree diverges from the reference engine. Requires integer accumulators.
- *Too slow even if working:* For dense, non-streaming workloads (training from random batches rather than search trajectories), IDA is *slower* than `nn.Linear` because of branch overhead. Strictly an inference-time and online-fine-tuning primitive.

**Status:** proposed

---

### 3. primitive_king_indexed_switching_bank

**Name:** KingIndexedSwitchingBank (KISB)

**One-line claim:** A discrete-indexed bank of weight matrices selected by an external integer key (king square), with a primitive-level "refresh-from-cache" operator that amortises full-recompute to `O(1)` over a search subtree.

**Mathematical signature:**
Bank `{W^{(k)} ∈ ℝ^{d_out × d_in}}_{k=1}^{K}`, index `κ ∈ {1, …, K}` from input (e.g., own-king square × side-to-move, K ≤ 128). Cache `C ∈ ℝ^{K × d_out}` (per-bank baseline output for the canonical empty-of-non-king-pieces position).
Forward: `y = C[κ] + W^{(κ)}[:, J] · x[J]` where `J = supp(x)`. On a king-move event (`κ_t ≠ κ_{t−1}`), refresh: `y_t = C[κ_t] + W^{(κ_t)}[:, supp(x_t)] · x_t[supp(x_t)]`, i.e., one full re-projection. Within a κ-constant subtree, behave as IDA.
Backward: per-bank gradient accumulates only over time-steps where that bank was selected; gradient through `κ` is zero (discrete).

**Why this does not decompose into existing PyTorch ops:** Closest is `nn.Embedding(K, d_out)` composed with a per-key `nn.Linear`. The decisive difference is the cache-refresh operator: the primitive's forward graph branches on `κ_t = κ_{t−1}` and switches between a sparse-delta path and a full-bank-evaluate-and-cache path. This is a stateful conditional-compute primitive — analogous to MoE-gate's non-decomposable conditional structure but with the routing signal *supplied externally as part of the input*, not learned. Without the cache-refresh branch it would be a hidden rebrand; with it, the gradient and FLOPs profile differ from any composition.

**Chess-specific motivation:** King-conditioned sparsity is the defining trick of HalfKA and HalfKAv2_hm. Stockfish's "Finny tables" cache the first-layer output per king square as engineering — KISB elevates this to a named, reusable primitive with well-defined gradient semantics.

**Generalisation beyond chess:** Any conditional-compute setting where a discrete context variable changes rarely relative to a fast-changing input: code-completion conditioned on filename, video models conditioned on scene-ID, robot policies conditioned on task-ID with infrequent task switches.

**Complexity (forward, backward, incremental-update):**
- Forward (κ unchanged): `O(‖Δx‖₀ · d_out)`; (κ changed): `O(d_in · d_out)`.
- Backward: `O(d_in · d_out)` per step in the worst case; sparse over banks in expectation.
- Incremental update on bounded-change input: `O(‖Δx‖₀ · d_out)` while king stationary, which is the dominant case in alpha-beta search subtrees.

**Scout-scale falsification test:** Replace i193's first dense layer with a KISB of K=64 (king square only, no horizontal mirror, side-to-move folded into encoding). Train at 173k positions × 12 epochs. Compare against (a) baseline i193, and (b) a deliberately weaker baseline using `nn.Embedding(64, d_out) + nn.Linear` composition (no cache-refresh branch). "Works" = KISB matches or beats baseline accuracy AND beats the composition baseline by ≥ 2× throughput on a search-trajectory benchmark. "Fails" = no measurable throughput edge over the composition baseline — in which case the primitive collapses to its embedding-plus-linear decomposition and should be cut.

**Failure mode catalogue:**
- *Hidden rebrand:* This is the highest-risk proposal on this axis. If the cache-refresh branch is not the dominant FLOPs differentiator in practice, KISB **is** `nn.Embedding` + per-key `nn.Linear`. The falsification test is structured to detect this.
- *Numerically unstable:* Per-bank parameter counts grow linearly in K. With K=64 and d_in=768, d_out=256, the bank holds ≈12.6M params — within a 234M scout's budget but already crowding the 8 GiB VRAM. K=8 (queen-side/king-side × side-to-move × 2 castling states) is the conservative choice.
- *Too slow even if working:* Bank selection requires a gather into parameter memory; on consumer GPUs this is bandwidth-bound and can erase any FLOPs advantage. The primitive's value materialises mostly at CPU inference (the NNUE regime), not RTX 3070 training.

**Status:** proposed

---

### 4. primitive_tropical_bilinear

**Name:** TropicalBilinear (TBL)

**One-line claim:** A (max,+)-semiring bilinear form `y_i = max_j (W_{ij} + x_j)` whose hard-argmax gradient flow concentrates capacity on the active path, mirroring minimax-search semantics.

**Mathematical signature:**
Inputs: `x ∈ ℝ^{d_in}`, parameters `W ∈ ℝ^{d_out × d_in}`. Forward: `y_i = max_j (W_{ij} + x_j)`, with `j^*(i) = argmax_j (W_{ij} + x_j)`. Backward: `∂y_i/∂x_j = 𝟙[j = j^*(i)]`, `∂y_i/∂W_{ij} = 𝟙[j = j^*(i)]`. Optional smooth variant: replace `max` with `(1/β) · logsumexp(β · ·)`, recovering soft routing as β → 0 and hard max as β → ∞.

**Why this does not decompose into existing PyTorch ops:** This is the most borderline of the five. At the computation-graph level, the forward can be written as `(W.unsqueeze(0) + x.unsqueeze(1)).max(dim=-1)` — purely existing ops. The primitive earns its name in the calibration-table sense the way `Conv2d` does: by the semiring change it occupies a recognised family of operators (Min-Max-Plus NNs, Luo 2021, arXiv 2102.06358; tropical-decision-boundary nets, arXiv 2402.00576) with universal-approximation results distinct from ReLU MLPs. **Per spec, this is an underexplored primitive for chess, not a new one.**

**Chess-specific motivation:** Chess evaluation lives downstream of a minimax tree; the natural semantic operation is "best-of." Tropical bilinears expose argmax structure as first-class differentiable computation rather than as an external softmax that the network must approximate. Tied to piece-square interaction structure: the "best square for this piece" intuition decomposes naturally over `(piece, square)` features.

**Generalisation beyond chess:** Shortest-path-style routing (min-plus is the algebra of Bellman-Ford), differentiable dynamic-programming layers, robust min-max objectives, neural attention to single-source bottlenecks.

**Complexity (forward, backward, incremental-update):**
- Forward: `O(d_in · d_out)` — same as `nn.Linear` in FLOPs but with `max` replacing `sum` (typically 30–50% slower wall-clock on GPUs without tensor-core support).
- Backward: `O(d_in · d_out)` worst case, but sparse: only `d_out` non-zero gradients in `∂y/∂x` since each output has one active argmax.
- Incremental update on bounded-change input: not naturally applicable — argmax can flip on small `Δx`, requiring full re-evaluation of affected rows.

**Scout-scale falsification test:** Substitute one mid-network `nn.Linear` in i193 with a TBL of identical shape. Keep all other components identical. Train at 173k × 12 epochs. Metric: PR-AUC overall and CRTK class-1 matched-recall near-puzzle FP. "Works" = TBL matches or beats baseline class-1 FP at ≤ 1.3× wall-clock. "Fails" = > 5% PR-AUC regression or > 2× wall-clock cost.

**Failure mode catalogue:**
- *Hidden rebrand:* Decomposable as `broadcast_add` + `max`. The defence "different gradient flow" is sound (subgradient at ties, sparse-Jacobian) but a strict reviewer may still call it composition. Treat as underexplored-for-chess rather than novel.
- *Numerically unstable:* At ties, the subgradient is set-valued; backward picks one arbitrarily and can produce noisy training. Smoothed-tropical (logsumexp) helps but then it really *is* attention-like.
- *Too slow even if working:* GPU tensor cores are matmul-optimised; max-plus operations do not benefit from cuBLAS and typically run 2–3× slower per FLOP. The inference-speed gain over `nn.Linear` is unlikely.

**Status:** proposed

---

### 5. primitive_equilibrium_energy

**Name:** EquilibriumEnergyPrimitive (EEP)

**One-line claim:** A fixed-point primitive over a learned piece-pair interaction energy, with output defined as `z* = argmin_z E_θ(z; X)` and gradient by implicit differentiation.

**Mathematical signature:**
Inputs: per-piece feature set `X = {x_p}_{p=1}^{P} ⊂ ℝ^d`. Define energy
`E_θ(z; X) = Σ_p (z_p − x_p)^T M_θ (z_p − x_p) + Σ_{p<q} ⟨z_p, A_θ z_q⟩ + Σ_p Φ_θ(z_p)`
with `M_θ ⪰ 0`, `A_θ` symmetric, `Φ_θ` convex per-coordinate.
Forward: solve `z* = argmin_z E_θ(z; X)` to tolerance ε by accelerated proximal gradient (capped at T inner iterations).
Backward: implicit-function theorem — `∂L/∂θ = −(∂E/∂z∂θ)^T (∂²E/∂z²)^{−1} ∂L/∂z*`, computed by one extra linear solve, **not** by unrolling.

**Why this does not decompose into existing PyTorch ops:** Closest existing primitive class is Deep Equilibrium Models (DEQ; Bai, Kolter, Koltun 2019, NeurIPS) and Modern Hopfield Networks (Ramsauer et al. 2020, "Hopfield Networks is All You Need"). The implicit-function-theorem backward is fundamentally different from a finite-step unrolled gradient — it does not appear in any naïve composition of `nn.Linear`, `softmax`, or `nn.Module` calls. **Per spec, this is an underexplored primitive for chess, not a new one.**

**Chess-specific motivation:** Piece-square interaction structure is naturally pairwise: `bishop_a × bishop_b`, `king × pawn-shelter`, `rook × file-control`. Treating the position evaluation as the equilibrium of a learned pairwise energy expresses this directly, and the symmetric structure of the energy bakes in piece-pair exchange equivariance (relabelling within piece types).

**Generalisation beyond chess:** This is the *most* generalisable of the five — any structured-prediction setting with pairwise terms benefits (protein folding, graph matching, image segmentation with CRF readout). The Hopfield-attention equivalence (Ramsauer 2020) means a well-tuned EEP for chess transfers to retrieval and associative-memory tasks.

**Complexity (forward, backward, incremental-update):**
- Forward: `O(T · P² · d)` for T proximal-gradient iterations vs single-layer attention `O(P² · d)`; T is typically 5–20 to convergence on small P (≤32 pieces).
- Backward: one linear solve, `O(P² · d²)` via conjugate gradients with warm start — *constant in T*, the headline DEQ advantage.
- Incremental update on bounded-change input: warm-start the fixed-point solver from `z*_{t−1}`; expected iterations drops sharply for single-move deltas. Not asymptotically sublinear but typically 3–5× speedup empirically; flag as "expected, not proven."

**Scout-scale falsification test:** Replace the i242 cross-stream attention block with an EEP on the piece-token set (P ≤ 32). Cap T=5. Train at 173k × 12 epochs, single seed. Metric: CRTK class-1 matched-recall near-puzzle FP. "Works" = EEP beats i242 by ≥ relative 5% on class-1 FP at no more than 1.5× wall-clock. "Fails" = T must be raised above 10 to converge, or accuracy is below the conv-only i193 baseline (consistent with the user's data-hungry-attention warning at 173k positions — explicitly flag this as a likely outcome at scout scale).

**Failure mode catalogue:**
- *Hidden rebrand:* Without the implicit-function backward, this is just an unrolled attention stack. The implicit-grad path is what makes it a primitive; the implementation must use `torch.autograd.Function` with a manual backward.
- *Numerically unstable:* Inner solve may not converge if `A_θ` becomes non-PD during training; require spectral normalisation on `A_θ` and `M_θ`. The `(∂²E/∂z²)^{−1}` solve can blow up near the Hessian's null space.
- *Too slow even if working:* DEQ-style primitives are typically 2–4× slower than equivalent stacked layers in wall-clock; the gain is at very long depth, not at the 12-epoch scout scale. **Most likely outcome at scout scale: trains but underperforms.** Include this in the experiment plan.

**Status:** proposed

---

### What I cut (and why)

1. **Color-swap Z₂ involution-equivariant primitive** — decomposes trivially into symmetric/antisymmetric projection (`(x + σx)/2`, `(x − σx)/2`) followed by standard linear maps. Pure architectural composition; would violate spec rule "no two streams that share weights."
2. **Capsule routing for piece-part hierarchy** — already a published primitive (Sabour, Frosst, Hinton 2017, "Dynamic Routing Between Capsules") with extensive 2024 follow-ups (windowed dynamic routing, Chen et al. 2024; ProtoCaps, TMLR 2023). The chess-specific motivation is weak — there is no compelling part-whole hierarchy in board positions that EEP doesn't already capture.
3. **Hyena-style implicit long convolution along a 1-D board scan** — already a published primitive (Poli et al. 2023). The "1-D scan over 64 squares" framing has no length advantage to exploit (sequence length is tiny), so the Hyena claim "subquadratic at long L" is irrelevant. Would be a rebrand-without-benefit.
4. **Tropical activation (max-plus ReLU variant)** — pure composition of element-wise `max` and `add`; violates the anti-example "a swish-but-with-different-polynomial activation." The bilinear-form version (kept as TBL above) is the salvageable part.
5. **Reynolds-projection equivariance over the piece-type permutation group** — at a finite group of size ≤ 6! per color, this is `(1/|G|) Σ_g ρ(g) f(ρ(g⁻¹) x)`, which is literal composition of permute + apply + permute + average. Group-equivariant CNNs (Cohen & Welling) are a recognised primitive family, but this specific instantiation does not earn its own slot.

## Recommendations

**Stage 1 (next 2 weeks, fits the scout-scale budget).** Implement IDA and MGR. They have the highest combined ranking on (a) plausibility of novelty, (b) demonstrability on a single RTX 3070, (c) inference-speed advantage. IDA has the strongest chess-structural alignment (HalfKA generalisation); MGR has the strongest "non-decomposable computation graph" defence. Concrete benchmarks to target:
- IDA: ≥ 3× throughput on a 1000-position linked-search trajectory vs i193's first dense layer, with PR-AUC within ±1%.
- MGR: ≥ relative 5% reduction in CRTK class-1 matched-recall near-puzzle FP **and** ≥ 30% wall-clock speedup vs the i242 attention head it replaces.

**Stage 2 (after Stage 1 results in hand).** If IDA's regime-tracking pays off, proceed to KISB to extend the sparse-update property across king moves. The benchmark threshold that triggers Stage 2: IDA delivers ≥ 1.5× throughput edge. If IDA does not clear that bar, KISB is unlikely to either — defer it.

**Stage 3 (research, not engineering).** Treat TBL and EEP as underexplored-for-chess papers, not as competitive engine components. Run each as a single ablation drop-in on i193 with explicit caveat that the expected outcome at 173k × 12 epochs is "trains but underperforms" — the value is the negative result confirming the user's i242 finding that attention/equilibrium primitives are data-hungry at this scale.

**Threshold for dropping a primitive entirely.** If, after the falsification test, a primitive (a) cannot be implemented without collapsing to a composition of existing ops, OR (b) fails both its accuracy and its wall-clock criterion, mark it `rejected` rather than `further-work`. Specifically: KISB should be dropped if the cache-refresh branch does not produce ≥ 2× throughput edge over an `nn.Embedding + nn.Linear` composition baseline.

## Caveats

- The search budget for this survey was capped at 12 queries; depth on each candidate's prior-art surface is shallower than ideal. Before publishing any of these as "novel primitives" (even MGR, the strongest case), perform a targeted 2024–2026 search on: "content-conditioned sparse message passing", "stateful incremental affine layer", "external-key conditional compute MoE", "tropical attention". The user's default "treat 2024–2026 published primitives as underexplored for chess rather than novel" should be applied liberally.
- The Sharir & Anandkumar 2023 paper (arXiv 2307.14988) is the most consequential prior-art find: it pre-empts a naïve framing of IDA. The wedge that keeps IDA distinct (exact `ClippedReLU`-regime propagation rather than VQ-approximate) is genuine but narrow. If a reviewer holds the line "your primitive is Sharir & Anandkumar restricted to piecewise-linear nets," IDA's contribution is incremental, not foundational.
- All "expected" outcome statements in the falsification tests are predictions, not measurements. The user's i242 result (transformer underperforms conv at 173k × 12 epochs) is the empirical anchor for predicting the same for EEP and possibly MGR.
- The 8 GiB VRAM and single-seed regime is genuinely restrictive: KISB at K ≥ 64 with `d_out = 256` already pressures the budget; EEP with implicit-grad linear solves typically peaks at 2× the static-graph memory of the equivalent unrolled stack. Implementation must include explicit memory-pressure checks.
- The spec rule "Stockfish scores, PVs, node counts, verification metadata may not enter the primitive's compute graph" was respected — MGR's `LegalEdges` function takes only piece-type + square (part of the encoding the user already uses), not engine analysis. Implementers should audit any move-generation library for accidental dependence on Stockfish-derived signals.
- This deliverable contains no measured Elo/PR-AUC numbers; per spec, all comparisons are framed as falsification thresholds, not as predicted gains.