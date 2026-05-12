# Five Candidate Neural-Network Primitives for Chess Evaluation

**Intro.** A literature scan (NeurIPS 2024 program, arXiv 2024–2025 efficient-attention and equivariant-network tracks) confirms that several "obvious" candidate primitives have been at least partly published: content-dependent sparse attention (Trainable Dynamic Mask Sparse Attention, Wang et al., arXiv 2508.02124, 2025; MInference 1.0, arXiv 2407.02490; Quest, ICML 2024), gated delta-rule linear attention (Yang, Kautz, Hatamizadeh, "Gated Delta Networks", arXiv 2412.06464, 2024), color/photometric group-equivariant convolutions (Lengyel et al., CEConv, NeurIPS 2023, arXiv 2310.19368), and deep equilibrium models with implicit differentiation (Bai, Kolter, Koltun, NeurIPS 2019). Two of the five proposals below are therefore explicitly framed as **underexplored-for-chess reframes** rather than fully fresh inventions, per the user's allowance. The three remaining proposals attempt genuine novelty by exploiting structural facts (incremental delta updates over a *stateful* differentiable graph, the chess attack-graph as an input-defined sparsity pattern, and the bilinear attack/defend relation) that the surveyed literature has not addressed as standalone operators. Ranking criteria: (a) plausibility of novelty, (b) RTX-3070 demonstrability under the 173k×12 epoch budget, (c) inference-speed advantage (especially incremental-update capability), (d) generalisation beyond chess.

---

## 1. `primitive_lmg_conv`

**Name:** Legal-Move-Graph Convolution (LMGConv).

**One-line claim:** A graph operator whose adjacency matrix is the *current position's* legal-move bitboard, rebuilt per forward pass without an O(N²) softmax.

**Mathematical signature:**
Let `x ∈ R^{64 × d}` be square embeddings and `A(x) ∈ {0,1}^{64 × 64}` be the legal-move adjacency for the position encoded in `x` (derived deterministically from piece-occupancy bits, *not* a learned mask). For each square `i` and edge-type `r ∈ {P,N,B,R,Q,K}` (piece moving):
`y_i = Σ_{r} Σ_{j : A_r(x)_{ij}=1} W_r x_j  +  b_i`,
where `W_r ∈ R^{d × d}` is a per-piece-type weight. Forward: a sparse `scatter_add` over the COO list of legal moves (typical |E| ≈ 30–80). Backward: gradient flows only along the existing edges; no gradient w.r.t. `A` (it is a discrete function of the input bitboard, treated as a constant in the computation graph).

**Why this does not decompose into existing PyTorch ops:** Closest comparable: masked self-attention with a content-derived mask (e.g., Dynamic Mask Attention, arXiv 2508.02124). The decomposition fails because (i) standard attention still allocates an `O(N²)` Q·Kᵀ score tensor before masking, while LMGConv never instantiates the dense score matrix; (ii) the adjacency here is *symbolically defined* by chess move-generation rules, so it is not differentiable w.r.t. its content — unlike DMA's differentiable mask. The computation graph is a typed `scatter_add` over a per-forward-pass dynamic COO buffer, which `torch.nn.MultiheadAttention` cannot produce.

**Chess-specific motivation:** The set of legal moves *is* the connectivity that strong human play uses. Standard 8×8 conv treats f2→g4 (knight) and f2→f3 (pawn) identically; LMGConv routes information only along legal piece-specific moves, giving the network the same inductive bias a player has.

**Generalisation beyond chess:** Any domain with a hard-rule-defined sparse interaction graph that changes per sample: molecular bond graphs at varying conformations, dynamic scene-graphs in video, combinatorial-game positions (Shogi, Go-with-ko).

**Complexity (forward, backward, incremental-update):**
- Forward: `O(|E| · d²)` with |E|≈30–80 vs. `O(64² · d) = O(4096 d)` for full attention.
- Backward: `O(|E| · d²)`.
- Incremental update on bounded-change input: `O(Δ|E| · d²)`; a move typically changes the legal-move set by ~20–30 edges around the source/destination/king, so updates are bounded and O(1) per move.

**Scout-scale falsification test:** Drop LMGConv in for the first attention block of the i242 chess-decomposed-attention model. Baseline: i242 unchanged. Train 12 epochs on 173k positions, single RTX 3070. Metric: matched-recall false-positive rate on CRTK class-1 (verified-near-puzzle) hard negatives. **Pass** if FP rate at fixed 95 % recall drops ≥ 3 percentage points relative to i242 and inference wall-clock is no worse than 1.2×. **Fail** if FP unchanged or wall-clock > 1.5× because of poor sparse-kernel utilisation on 8 GiB SM count.

**Failure mode catalogue:**
- *Hidden rebrand:* could collapse to Edge-Conditioned Convolution (Simonovsky & Komodakis, CVPR 2017) or relational GCN (Schlichtkrull et al., 2018) once the per-forward-pass graph rebuild is amortised in a kernel — the novelty is then operational, not mathematical.
- *Numerical/instability:* with only ~30 messages aggregated per node and no normalisation, gradients can spike on captures (large change in |E|); needs degree-normalised aggregation similar to GraphSAGE.
- *Speed:* small sparse kernels on a 3070 may not beat dense `bmm`; the FLOP win can be eaten by launch overhead unless edges are batched COO-coalesced. Strongest reviewer objection: "this is masked attention with extra steps."

**Status:** proposed.

---

## 2. `primitive_delta_acc`

**Name:** Differentiable Delta-Accumulator (ΔAcc).

**One-line claim:** A stateful, fully differentiable linear primitive that exposes an `update(Δ_in_idx, Δ_out_idx)` API and propagates gradients through arbitrarily long sequences of bounded-change updates.

**Mathematical signature:**
State `h_t ∈ R^d`. Weight `W ∈ R^{d × F}` where `F` is the binary feature space (HalfKA-style, ~40 000 features). Forward step:
`h_t = h_{t-1} + W e_t  ,  e_t ∈ {-1, 0, +1}^F` with `||e_t||_0 = k` small (typically k ≤ 8).
Output emit: `y_t = φ(h_t)` for some activation `φ`. The persistent-state semantics matter: gradients flow `∂L/∂W = Σ_t (∂L/∂h_t) e_tᵀ` and `∂L/∂h_{t-1} = ∂L/∂h_t` (identity transition). The primitive is registered as `torch.nn.DeltaLinear` and behaves as a stateful module that supports `forward_full(active_set)` and `forward_delta(Δ)`, with autograd routing through both.

**Why this does not decompose into existing PyTorch ops:** At a frozen point in time, ΔAcc is mathematically equivalent to a sparse `EmbeddingBag` over the active feature set — that is the honest devil's-advocate concession. The structural novelty is in the *training* computation graph: standard PyTorch sparse ops are stateless, so backprop through a length-T trajectory of correlated deltas (e.g., a self-play game) currently requires recomputing the full sparse sum at every t (O(T·k_active·d)), whereas ΔAcc shares the persistent state node across all T forwards and reduces to O(T·k·d) with k ≪ k_active. The recurrence `h_t = h_{t-1} + W e_t` cannot be folded into a one-shot sparse matmul without dropping the cross-step gradient identity. Closest comparable: Gated Delta Networks (Yang et al., arXiv 2412.06464, 2024) — ΔAcc is the *un*gated, identity-transition special case and is **explicitly framed as an underexplored-for-chess reframe of that line**, specialised for the chess accumulator pattern.

**Chess-specific motivation:** This is the trained-time analogue of Stockfish NNUE's inference-time accumulator. NNUE only uses incremental update at search time; training is dense. ΔAcc lets the first-layer accumulator be trained on game trajectories where consecutive positions share state, exposing the move-locality prior to the optimiser.

**Generalisation beyond chess:** Any sequential domain with bounded-change inputs: text-stream editing models, codebase diff embeddings, financial order-book updates, particle-system simulators with sparse contact changes.

**Complexity:**
- Forward (full): `O(k_active · d)` ≈ standard sparse embed.
- Forward (delta): `O(k · d)` with k ≤ 8.
- Backward (over length-T trajectory): `O(T · k · d)` vs. `O(T · k_active · d)` naïve.
- Incremental update: `O(k · d)` — the design point.

**Scout-scale falsification test:** Drop ΔAcc as the first-layer accumulator in i243 (HalfKA+dual-stream). Compare against the dense first-layer baseline at 173k×12 epochs. Metrics: (a) PR AUC on CRTK class-1 hard negatives — equal or better; (b) inference NPS in a single-threaded position-after-position scan over 10 000 game positions — must be ≥ 3× the dense baseline to count as a pass. **Fail** if NPS gain < 2× or if backward step diverges due to state-vs-trajectory misalignment (a known instability of stateful primitives).

**Failure mode catalogue:**
- *Hidden rebrand:* at any single t, ΔAcc *is* a sparse linear; the only mathematical novelty is in the training-time graph. The strongest reviewer objection — "this is just an autograd-friendly NNUE accumulator" — is essentially correct; the contribution is making it a torch.nn primitive rather than engine-code.
- *Numerical instability:* with no decay/gate, |h_t| can drift over long sequences; integer-quantisation NaN risk is real.
- *Slow in practice:* CPU NNUE wins because of int8 SIMD; on GPU at fp16 the delta path may be bandwidth-bound and lose to a fused dense matmul on small d.

**Status:** proposed (explicit reframe of Gated Delta Networks 2024 for chess incremental accumulators).

---

## 3. `primitive_cswap_eqconv`

**Name:** Color-Swap × D4 Group-Equivariant Convolution (CSE-Conv).

**One-line claim:** A single convolution operator whose filter bank enforces equivariance under the chess group `G = D4 × Z2` (board dihedral × color-swap involution) by construction.

**Mathematical signature:**
Filter bank `ψ : G → R^{c_out × c_in × k × k}`, stored as one orbit representative `ψ_e` of shape `(c_out × c_in × k × k)`. Forward on an input `f : Z² × G → R^{c_in}`:
`[f * ψ](x, g) = Σ_{h ∈ G} Σ_{y ∈ Z²} f(y, h) ψ_e(g^{-1}(x − y), g^{-1} h)`
with the color-swap action `(σ · ψ)(c) = −ψ(σ c)` for the channel pair `(c_white, c_black)` (an *odd* representation of Z2 enforcing eval(−x) = −eval(x)).

**Why this does not decompose into existing PyTorch ops:** Closest comparable: Cohen & Welling G-CNN (ICML 2016) over D4, and the photometric Color Equivariant Convolutions of Lengyel et al. (NeurIPS 2023, arXiv 2310.19368). The structural difference is the combination of (a) the *odd* (sign-flipping) Z2 representation for color — CEConv uses an *even* hue-rotation representation, which preserves sign — and (b) the joint product `D4 × Z2` with a 16-orbit filter bank. There is no PyTorch composition that yields a 16-fold-tied convolution with one channel-pair anti-symmetrised; you would need to write the orbit-tying explicitly. **This is explicitly framed as an underexplored-for-chess reframe of G-CNN/CEConv**, with the technical novelty being the odd-rep of Z2 on the color channel.

**Chess-specific motivation:** Chess evaluation satisfies eval(position) = −eval(color-swapped position) exactly. Most chess nets learn this via data augmentation; CSE-Conv bakes it into the parameter count and removes a factor of 2 in sample complexity for the color symmetry. The D4 part captures dihedral-4 of the board (which is broken by castling rights but holds on the piece-square structure).

**Generalisation beyond chess:** Any antisymmetric pairwise prediction task: two-player zero-sum game eval (Go komi-corrected, Shogi), siamese-difference networks, fermionic wavefunctions (where eval(swap) = −eval).

**Complexity:**
- Forward: `O(|G| · H · W · c_out · c_in · k²)` = 16× a plain conv, but parameters are 16× fewer for equivalent expressive capacity, so FLOPs/param matched.
- Backward: same scaling.
- Incremental update: not applicable (translation-equivariant kernel; no delta semantics).

**Scout-scale falsification test:** Replace the stem conv of i193 (conv-only parent of i242) with CSE-Conv at matched parameter count. Train 12 epochs, 173k positions. **Pass** if (i) the explicit color-swap equivariance error on a held-out swapped pair drops to numerical zero (it should, by construction) AND (ii) PR AUC on the CRTK matched-recall set improves ≥ 1 pp at the same wall-clock budget. **Fail** if matched accuracy is unchanged — equivariance was already learnable from augmentation.

**Failure mode catalogue:**
- *Hidden rebrand:* this is G-CNN over a particular small group. The contribution is the *odd* Z2 rep, which is one extra sign-flip; reviewers will reasonably ask whether a frozen color-swap data augmentation does the same.
- *Numerical instability:* the antisymmetric tying around 0 makes the zero-eval region a manifold of measure zero — gradient signal near drawn positions can vanish.
- *Speed:* 16× kernel orbits hurt throughput on a 3070; an unrolled-tied implementation needs a custom CUDA kernel to beat dense conv.

**Status:** proposed (reframe of G-CNN/CEConv).

---

## 4. `primitive_mob_fp`

**Name:** Mobility Fixed-Point Operator (MobFP).

**One-line claim:** An implicit-layer primitive that returns the fixed point of an attack-graph propagation, capturing x-ray, pin and discovered-check structure in one differentiable solve.

**Mathematical signature:**
Given attack adjacency `A(x) ∈ {0,1}^{64×64}` (deterministic function of piece occupancy and type, including sliding rays), occupancy mask `o ∈ {0,1}^{64}`, and per-square feature `z_0 ∈ R^{64 × d}`, define the parameterised update
`F_θ(z; x) = σ( W_self z + W_att (A(x) ⊙ M_block(z)) z + b )`,
where `M_block(z) ∈ [0,1]^{64×64}` is a learned soft-block matrix capturing whether a square's piece blocks a ray, computed from `z`. Output `z* = MobFP(x) = lim_{k→∞} F_θ^{(k)}(z_0; x)`, solved by Anderson-accelerated fixed-point iteration; backward via implicit function theorem (Bai, Kolter, Koltun, NeurIPS 2019). Convergence bound: the chess board has graph diameter ≤ 7, so a contractive `F_θ` converges in ≤ 8 iterations in practice.

**Why this does not decompose into existing PyTorch ops:** A K-times unrolled message-passing GNN over the same adjacency would have a K-deep computation graph and memory `O(K)`. MobFP runs an *adaptive* number of solver iterations (typically 4–8) with `O(1)` memory via implicit-gradient backward. Closest comparable: DEQ (Bai et al. 2019). The structural specialisation here is that the recurrence lives on a 64-node, input-defined sparse attack graph, not on a token sequence — and the contractive condition is enforced by a chess-specific Lipschitz bound on `M_block`. **This is positioned as an underexplored-for-chess reframe of DEQ** rather than a fresh mathematical object.

**Chess-specific motivation:** Pins, x-rays and discovered checks all require *propagating* attack information through intermediate pieces — exactly a fixed-point of a flow on the attack graph. Unrolling enough conv layers to do this costs ≥ 7 layers; MobFP collapses that into one O(1)-memory primitive whose iteration count adapts to the position's complexity.

**Generalisation beyond chess:** Any reachability- or flow-equilibrium computation on input-defined graphs: traffic forecasting, gossip protocols, Bellman-Ford-style reasoning layers, electrical-network solvers.

**Complexity:**
- Forward: `O(K · |E(A)| · d)` with K ≤ 8, `|E(A)|` ≈ a few hundred.
- Backward: one linear solve of size `64 d × 64 d` per call — dominant cost.
- Incremental update: not applicable in the strict sense; fixed-point must be re-solved per position, though warm-starting from prior `z*` is straightforward.

**Scout-scale falsification test:** Stand-alone diagnostic: place MobFP as a *head* attached to i193's feature map, train it to predict the binary "is there a tactical pin in this position" label on a CRTK subset where pin presence is verifiable from Stockfish PV (pin label only, *no Stockfish score*). 12 epochs, 173k positions. **Pass** if MobFP head reaches ≥ 90 % pin-detection F1 with K_avg ≤ 6 solver iterations, *and* matched-recall FP rate on full CRTK improves ≥ 2 pp when MobFP features are concatenated. **Fail** if a 6-layer unrolled GCN matches it at lower wall-clock.

**Failure mode catalogue:**
- *Hidden rebrand:* it *is* DEQ on a chess graph — the only specialisation is the Lipschitz constraint and graph structure.
- *Numerical instability:* implicit-gradient backward is famously unstable; Phantom-Gradient or Jacobian-Free Backprop (Geng et al., NeurIPS 2021) may be required, adding hyperparameters.
- *Speed:* Anderson solves and Jacobian-vector products will be *slower* than 6 unrolled layers on a 3070, so the bet is purely on accuracy/parameter efficiency, not speed — this is a scout-scale-only primitive.

**Status:** proposed (reframe of DEQ for input-defined chess graphs).

---

## 5. `primitive_attack_bilinear`

**Name:** Attack-Defend Sparse Bilinear (ADB).

**One-line claim:** A bilinear operator that produces an interaction tensor only on (attacker, target) square pairs given by the position's attack relation, with content-defined support but *no* softmax.

**Mathematical signature:**
Inputs: square embeddings `u ∈ R^{64 × d}` (attacker view) and `v ∈ R^{64 × d}` (target view), attack adjacency `B(x) ∈ {0,1}^{64 × 64}` (square i attacks square j). Output: a feature on attacker–target pairs,
`E_{ij} = B(x)_{ij} · (u_i ⊙ W v_j)  ∈ R^{|supp(B)| × d_e}`
with `W ∈ R^{d × d}`, and `⊙` the Hadamard product (or, optionally, a small tensor contraction yielding `d_e` channels). The output is a *ragged tensor* indexed by attacker-target pairs; it is then reduced by `scatter_add` either to the attacker (mobility-style head) or to the target (threat-on-target-style head). Gradient flows only along edges present in `B`.

**Why this does not decompose into existing PyTorch ops:** Closest comparable: a standard bilinear layer `Bilinear(u, v) = u^T W v` (which yields a dense `64×64` interaction tensor); or attention without softmax (which still allocates the dense score map). ADB never materialises the dense map: it computes only `|supp(B)|` interactions and stores them as a ragged tensor with edge-list indices. There is no `torch.nn` op that produces a sparse-supported pairwise bilinear with content-defined support; `torch.sparse.mm` requires a static sparsity pattern within a kernel call, and `nn.Bilinear` is dense. The discrete `B(x)` is treated as a non-differentiable structural constant — this is the key graph-level difference from learned-sparse-attention proposals (DMA, MoSA), where the mask is itself differentiable.

**Chess-specific motivation:** Exchange-value, SEE (Static Exchange Evaluation), and most tactical motifs (forks, pins, skewers) depend on *which* pieces attack *which* squares. ADB exposes this exact relation as a first-class feature, rather than asking a CNN to infer it.

**Generalisation beyond chess:** Sparse pairwise-relation modelling in combinatorial settings: bond-energy contributions in molecular dynamics where bond list changes per step; entity-interaction graphs in physics simulators; sparse cross-attention with hard-rule masks (e.g., type-system constraints in code models).

**Complexity:**
- Forward: `O(|supp(B)| · d_e)` ≈ a few hundred × d vs. `O(64² · d) = O(4096 d)` for dense bilinear.
- Backward: same.
- Incremental update: `O(Δ|supp(B)| · d_e)`; bounded by the change in attack relation, which for a non-king move is typically ≤ 20 edges.

**Scout-scale falsification test:** Insert ADB as a parallel side-channel to i193's mid-block convs; concatenate the attacker-reduced output. Train 12 epochs, 173k positions on the RTX 3070. **Pass** if matched-recall FP on CRTK class 1 improves ≥ 4 pp *and* the operator survives an ablation where `B(x)` is replaced by a random sparse mask of the same density (the random-mask ablation should be strictly worse — if not, ADB has not learned anything attack-specific). **Fail** if random-mask ablation matches, indicating the gain is just from added capacity.

**Failure mode catalogue:**
- *Hidden rebrand:* this is a sparse bilinear with content-derived support — close enough to "masked attention without softmax" that a reviewer can argue it's just a sparsified linear-attention head. The defense is the chess-rule-derived (non-learned) support.
- *Numerical instability:* ragged-tensor `scatter_add` reductions can underflow with fp16 when many edges contribute to one node (queen on open file ≈ 14 attackers/defenders); needs reduction in fp32.
- *Speed:* the same small-kernel/launch-overhead pitfall as LMGConv; without a fused CUDA kernel, the FLOP saving may not translate to wall-clock saving on a 3070.

**Status:** proposed.

---

## Self-Audit of Top 2 (Devil's Advocate)

**LMGConv (#1):** The strongest reduction-to-prior-work argument is that LMGConv is a typed Edge-Conditioned Convolution (Simonovsky & Komodakis, CVPR 2017) plus an R-GCN (Schlichtkrull et al., 2018) plus a per-forward graph rebuild. The mathematical operation is exactly typed sparse `scatter_add`. The remaining novelty is operational: the COO buffer is rebuilt per forward from a chess move generator, which is not a thing R-GCN does. I keep the proposal but mark its novelty as "operator-system-level, not closed-form-mathematics-level."

**ΔAcc (#2):** At any fixed time t, ΔAcc(active_set_t) equals a sparse `EmbeddingBag(active_set_t)`. The math is identical. The honest claim is therefore not "new function" but "new computation graph for the same function under sequential bounded-change inputs, with a stateful training graph and a δ-API inference path." That is a legitimate primitive contribution — `LayerNorm` is also "just" a particular rescaling — but reviewers will push back. I keep the proposal and explicitly frame as a reframe of Gated Delta Networks (Yang et al. 2024), retaining only the ungated identity-transition special case as the chess-relevant slice.

Neither top-2 candidate is *dropped*; both are *reframed honestly*.

---

## Ranking Summary

| Primitive | Novelty plausibility | RTX-3070 demonstrability | Inference-speed advantage | Generalisation |
|---|---|---|---|---|
| 1. LMGConv | Medium (close to ECC / DMA) | High (fits in 2 GPU-hr) | High — O(\|E\|) << O(N²) | High (any dynamic-graph domain) |
| 2. ΔAcc | Low–Medium (reframe of Gated Delta Net / NNUE) | High | **Very high** — true O(k) delta path | High (any bounded-change-input domain) |
| 3. CSE-Conv | Low (G-CNN reframe with odd Z2 rep) | High | Neutral (16× kernel tying ≈ same FLOPs) | Medium (antisymmetric pairwise tasks) |
| 4. MobFP | Medium (reframe of DEQ on chess graph) | Medium (implicit-grad solver risk) | Low (likely slower than 6-layer unroll) | High (any reachability/flow task) |
| 5. ADB | Medium–High (no direct prior bilinear-on-input-graph op) | High | High — sparse-supported bilinear | Medium–High (sparse pairwise interactions) |

Best fresh-novelty bets: **ADB** (#5) and **LMGConv** (#1).
Best inference-speed bets: **ΔAcc** (#2) and **LMGConv** (#1).
Best generalisation bets: **LMGConv**, **ΔAcc**, **MobFP**.

---

## What I Cut

**Piece-Type Hard-Routed MoE.** A mixture-of-experts where the routing decision is the discrete piece type (P, N, B, R, Q, K). On audit, this decomposes exactly into six masked dense linear layers summed elementwise — i.e., `Σ_t 1[piece=t] · W_t x`. That is a composition of existing PyTorch ops with a one-hot mask. It is a layer, not a primitive. Dropped as a hidden rebrand of conditional one-hot selection.

**Antisymmetric Linear via Parameter Projection.** A layer enforcing `f(σx) = −f(x)` by parameterising `W ↦ (W − σ^T W σ)/2` on every forward. While the symmetry is real and chess-relevant, the operator decomposes into two ordinary linear evaluations and a subtraction. There is no new gradient flow or new connectivity. Dropped: identical math to a Siamese-difference network.

**Learned Bitboard Hash-Embedding.** A LSH-style hash of the 64-bit occupancy bitboard into a learned embedding table. On audit, this is identical to hash embeddings (Tito Svenstrup et al., NeurIPS 2017) and to the feature-hashing trick. It is an encoding scheme dressed as an operator, and the user's rules explicitly bar input-encoding proposals. Dropped.

**Static Exchange Evaluation (SEE)-Inspired Iterative Layer.** A layer that runs the SEE algorithm (alternating-color capture-sequence min/max) as a recurrent sub-network. On audit, although SEE is a chess gem, embedding it as a differentiable layer requires either soft-min/soft-max (collapses to a small RNN) or argmin (non-differentiable). The differentiable version is just a tiny GRU on a fixed schedule. Not novel as a primitive. Dropped as a layer-specific gadget rather than a reusable operator.

**Per-Square Conditional-Compute Skip (Empty-Square Gate).** A gate that skips compute on empty squares (~ 60–70 % of mid-game squares). On audit, this is the standard conditional-compute / token-pruning pattern (e.g., A-ViT, DynamicViT). Already a well-known mechanism, and the speed gain on a 3070 is dwarfed by warp-divergence cost. Not scout-scale demonstrable as a clean win. Dropped as published.

---

## Recommendations (Staged)

**Stage 0 (this week):** Implement and unit-test **LMGConv** (#1) and **ADB** (#5) — both share infrastructure (per-forward COO buffer from chess move generator). Verify forward-pass correctness against a manual reference on 100 hand-checked positions.

**Stage 1 (~2 GPU-hours, scout-scale):** Run the falsification tests for #1 and #5 as written. Decision threshold: if matched-recall FP on CRTK class 1 improves ≥ 3 pp for #1 or ≥ 4 pp for #5, promote to engine-scale. If not, kill that candidate.

**Stage 2:** If either #1 or #5 promotes, implement **ΔAcc** (#2) on top of the winning operator's accumulator pathway. Target: 3× inference NPS at matched PR AUC. If hit, this combination is the path toward an NNUE-style differentiable trainer with sparse-graph inductive bias.

**Stage 3 (deferred):** **MobFP** (#4) and **CSE-Conv** (#3) are scout-scale-only and small-effect-size respectively; only revisit if a specific failure mode in the i243 line implicates pin/x-ray reasoning (MobFP) or color-asymmetry bias (CSE-Conv).

**Kill criterion that changes everything:** if all five primitives fail to improve CRTK class-1 FP rate at matched wall-clock, the i242 finding that "attention is data-hungry at 173k positions" generalises — and the actionable conclusion is that the playground should pivot to data-scale experiments (≥ 10⁶ positions) before further primitive design.

---

## Caveats

1. **Literature-survey limit.** The 12 web searches surfaced direct prior art for at least three of the five families (sparse-mask attention 2024–2025; gated delta networks 2024; G-CNN / CEConv 2023). The two proposals positioned as fresh (ADB, LMGConv) are *not* exhaustively verified against the 2025–2026 literature; reviewers may know of closer prior work I did not surface.

2. **No empirical numbers claimed.** No proposal here cites a result like "+X PR AUC"; all such numbers are *targets* to be measured in the falsification tests.

3. **Calibration honesty.** The novelty bar set by the user (GELU, LayerNorm, Mamba/S6, MoE gate) is extremely high. Only **MobFP** and **ADB** plausibly clear that bar as standalone operators; the other three are honestly closer to engineering specialisations of known operators and are labelled as such.

4. **3070 / 8 GiB / single-seed reproducibility.** Several proposals (#1, #5) depend on custom sparse CUDA kernels for their inference-speed claim. The scout-scale tests can be run with naïve `scatter_add` implementations, but the speed claims are *contingent* on later kernel work and should not be reported as wall-clock wins before that work is done.

5. **Stockfish-leakage rule.** All five primitives compute their connectivity / state from raw board occupancy and piece type only; none ingest scores, PVs, or node counts. The discrete `A(x)` and `B(x)` adjacencies are derived from chess move-generation rules applied to the input bitboard, which is permitted.

6. **Tooling note.** The intended workflow included a focused subagent investigation and an enrichment pass on the integrated draft; those tools were not available in this environment, so the proposals here have not been hardened by a second-pass evidence check. Treat novelty claims for ADB and LMGConv as the most likely targets for closer literature review before publication.