# Five Candidate Neural-Network Primitives for Chess Evaluation

**Prepared for:** `chess-nn-playground` (i244 primitive-design pass)
**Audience:** technically sophisticated; chess engine + Tsinghua automation context
**Status:** all five proposals are **proposed**, not validated

---

## Intro / framing the search

The brief is to invent **primitives**, not architectures or encodings. Recent (2023–2026) work that genuinely qualifies as primitive-level — Mamba/S6's input-conditioned recurrence (Gu & Dao 2023), DeltaNet's Householder state update (Yang et al., NeurIPS 2024), Soft MoE's continuous routing (Puigcerver et al., ICLR 2024), Native Sparse Attention's natively trainable sparse mask (Yuan et al. 2025), and Expert Choice routing (Zhou et al. 2022) — all share one structural property: the **computation graph itself is conditioned on the input**, in a way that does not decompose into static `nn.Linear` + `nn.Softmax` blocks. The strongest chess-specific structural facts uncovered during the scout — HalfKA's O(1) accumulator update (Nasu 2018; Stockfish 12+), the color-swap involution (Carroll & Beel 2020 used it for checkers but not for chess piece-type pairs), the sparse legal-move graph that changes every ply, and the dihedral-4 + color-swap group — line up well with the "graph-changes-per-forward-pass" and "incremental-update" axes that recent ML literature is just beginning to formalise. The five proposals below each pick one such axis and build a primitive around it; two of them (Attack-Ray Sparse Attention and the Delta-Accumulator Primitive) are the most defensible novelty claims, the other three are framed as **underexplored primitives for chess** where 2024–2025 papers exist within shouting distance.

---

## 1. `primitive_arsa` — Attack-Ray Sparse Attention

### primitive_arsa

**Name:** Attack-Ray Sparse Attention (ARSA)

**One-line claim:** Attention whose sparse key–query edge set is determined per-input by deterministic line-of-sight / occlusion geometry, not by learned scores or a fixed mask.

**Mathematical signature:**
Input: token embeddings `X ∈ ℝ^{B×N×d}` and a *connectivity oracle* `R: X ↦ E`, where `E ⊆ {1..N}²` is the set of (query, key) pairs determined by ray-occlusion rules over the discrete board features inside `X`. Forward:
`A_{ij} = softmax_{j: (i,j)∈E} ( q_i · k_j / √d )`, with `output_i = Σ_{j:(i,j)∈E} A_{ij} v_j`.
`E` is recomputed every forward pass; `|E| = O(N · r)` where `r ≪ N` is the mean ray-fanout (empirically ~10 on chess). Gradient is well-defined w.r.t. Q/K/V; w.r.t. `E` it is zero (discrete, treat as STE / detach).

**Why this does not decompose into existing PyTorch ops:**
Standard `nn.MultiheadAttention` with an `attn_mask` requires the mask to be a tensor argument passed in by the user. ARSA differs because the mask is **produced by a non-differentiable, input-dependent oracle that is part of the primitive**, with a CSR-style sparse layout that varies in nnz per element of the batch; current PyTorch fused attention kernels assume either dense or one common mask. Native Sparse Attention (Yuan et al., arXiv:2502.11089) selects keys by *learned score top-k*; ARSA selects by *rule-based ray tracing on the discrete input*. Different computation graph (no score-pool branch), different complexity (no top-k sort).

**Chess-specific motivation:**
For sliding pieces (B/R/Q), the only squares that matter on the next ply are those on the unobstructed rays. This is a per-position graph that no static window pattern captures (Longformer/BigBird are pre-fixed; i242's chess-decomposed attention used fixed masks). The mask carries exactly the information that handcrafted Stockfish evaluation calls "mobility" and "x-ray", which the conv-only parent i193 still beat i242 at scout scale.

**Generalisation beyond chess:**
Any domain with input-determined visibility graphs: ray-traced rendering, light-cone propagation in physics simulations, line-of-sight in RTS games, occlusion graphs in robotics SLAM.

**Complexity (forward, backward, incremental-update):**
- Forward: O(N·r·d) vs full attention O(N²·d); r≈10 for chess, N=64.
- Backward: O(N·r·d).
- Incremental update on a bounded-change input (one piece moved): only the affected rays change → O(r²·d) edges recomputed, the rest of the accumulated KV reused. Genuine sub-N update.

**Scout-scale falsification test:**
Drop ARSA into the attention sub-blocks of i242 (chess-decomposed attention) with a hand-coded ray-oracle. Train 12 epochs × 173k positions, single seed, RTX 3070. Baselines: (a) i242 with full attention; (b) i242 with a Chebyshev / king-distance fixed mask. Metric: false-positive rate at the matched-recall operating point on CRTK class 1 (verified-near-puzzle). Works if ARSA's FP rate ≤ 0.9× of baseline (a) AND wall-clock per epoch ≤ 1.1× of baseline (a). Fails if either condition violated, or if it tracks (b) within noise (then it's "just any sparse mask").

**Failure mode catalogue:**
- *Hidden rebrand:* If on a 3070 the ray-oracle is implemented as a precomputed dense mask tensor per batch, the kernel collapses to `attn_mask`-PyTorch; rebut by requiring a fused CSR forward that runs in O(|E|).
- *Numerical instability:* tokens with very small `|E_i|` (e.g. a trapped knight) produce near-singular softmax denominators; clip min-edges or add a constant self-edge.
- *Too slow:* nnz fluctuates per batch element, so a naïve implementation serialises; needs a Triton-style block-sparse kernel à la NSA (Yuan et al. 2025) to beat dense attention at N=64.

**Status:** proposed (best novelty-bar candidate; closest prior art is NSA 2025 with *learned* score-based selection, structurally different).

---

## 2. `primitive_dap` — Delta-Accumulator Primitive

### primitive_dap

**Name:** Delta-Accumulator Primitive (DAP)

**One-line claim:** A stateful linear operator with paired make/unmake side-channels whose forward cost depends on the *change* in input, not its size.

**Mathematical signature:**
State: `s ∈ ℝ^d`, weight `W ∈ ℝ^{d×F}` with very large F (sparse input domain). Operations:
- `s ← apply(s, Δ_add, Δ_rem)` where `Δ_add, Δ_rem` are sparse index sets of size ≤ k: `s_new = s + Σ_{i∈Δ_add} W[:,i] − Σ_{i∈Δ_rem} W[:,i]`.
- `s ← unmake(s, Δ_add, Δ_rem)`: inverse.
Forward at the architecture boundary returns `s`; backward maintains a stack of `(Δ_add, Δ_rem)` so that `∂L/∂W[:, i] = Σ_{t: i∈Δ_add^t} ∂L/∂s_t − Σ_{t: i∈Δ_rem^t} ∂L/∂s_t`. Gradient through state requires a TBPTT-style replay over the make-stack.

**Why this does not decompose into existing PyTorch ops:**
`nn.Linear` recomputes `Wx` from scratch each call. `torch.sparse.mm` exists but is stateless. DAP is a **stateful op with a paired inverse** whose backward graph is a *temporal chain over the make/unmake stack*, not a single matmul DAG. This pattern is what Stockfish NNUE's `AccumulatorStack` does in C++ (see Stockfish docs: "Accumulator state ... up to MAX_PLY+1"), but it is not exposed in PyTorch and is not what `nn.Linear` produces under autograd.

**Chess-specific motivation:**
HalfKA flips ~2–4 of ~30 active features per move (Stockfish NNUE docs, 0.07% density). A dense forward over ~45k binary features wastes 99.9% of FLOPs. DAP makes the first layer of any tree-search-driven evaluator structurally O(k·d) per node, k≈3. This is the property that gave Stockfish 12 ~80 Elo overnight.

**Generalisation beyond chess:**
Any sequential / tree-search evaluator over discrete state with local edits: shogi, go (less tightly — edits are larger), molecular dynamics rollouts with single-bond changes, ICEs in compilers (incremental SSA), incremental scene rendering.

**Complexity (forward, backward, incremental-update):**
- Forward `apply`: O(k·d) vs `nn.Linear` over the same sparse input O(F·d).
- Backward: O(T·k·d) per gradient step where T = make-stack depth (typically T ≤ MCTS depth on a single rollout, 32–64).
- Incremental update: **this is what the primitive is**. O(k·d), independent of F.

**Scout-scale falsification test:**
Replace the first `nn.Linear(45056, 256)` of a HalfKA-style net with DAP. Train 12 epochs × 173k positions; compare *wall-clock per training step* and *per-position inference latency* against the dense baseline at matched accuracy. Works if inference latency drops ≥ 3× at parity FP-rate on CRTK class 1. Fails if autograd through the make-stack is more than 1.3× slower than dense backward (then the training-time win is gone even if inference wins).

**Failure mode catalogue:**
- *Hidden rebrand:* Without the paired-inverse and stack, this is just `torch.sparse.mm` repeated. The novelty is the **operator-pair semantics** (make, unmake) and the backward graph they induce. Without unmake, drop the proposal.
- *Numerically unstable:* drift over very long make-chains (>1000 plies) because `s` is updated in low-precision int8/int16 as NNUE does; periodic refresh from cache needed, exactly the "Finny tables" trick Stockfish ships.
- *Too slow on GPU:* the operator is CPU-friendly (small k, gather/scatter), GPU-hostile. The 3070 demonstration must run inference on CPU or accept a wash on GPU; pitch is CPU-MCTS engine deployment.

**Status:** proposed; explicitly **underexplored primitive for chess** in the PyTorch sense — the C++ NNUE accumulator is well known, but no torch.nn.<DeltaAccumulator> exists with autograd support, and the operator-pair semantics has not been formalised in mainstream DL literature.

---

## 3. `primitive_trees6` — Tree-Selective State-Space Operator

### primitive_trees6

**Name:** Tree-Selective State-Space (Tree-S6)

**One-line claim:** Mamba's input-conditioned selective SSM, but the recurrence runs over a branching game tree rather than a linear sequence, with stack-based state replay at branch points.

**Mathematical signature:**
For a tree of states `v ∈ V` with parent function `π(v)`, input `x_v ∈ ℝ^d`:
`A_v = exp(Δ_v ⊙ A_logit(x_v))`, `B_v = B_proj(x_v)`, `Δ_v = softplus(Δ_proj(x_v))` (S6 style, all input-conditioned, per Gu & Dao 2023 Algorithm 2).
Recurrence: `h_v = A_v ⊙ h_{π(v)} + B_v ⊙ x_v`, `y_v = C(x_v)·h_v`.
Difference from Mamba: `π(v)` is a tree, so `h_{π(v)}` must be looked up by branch identity, not by sequence index — implemented via a state stack pushed on tree descent and popped on ascent.

**Why this does not decompose into existing PyTorch ops:**
Mamba's hardware-aware parallel scan (`selective_scan_cuda`) requires a linear sequence layout to fuse the scan. A tree layout forces a non-contiguous gather across siblings, breaking the scan kernel. Naïvely materialising every root-to-leaf path explodes O(branching^depth). The primitive defines a tree-scan operator whose state graph has a different topology than any composition of `RNN`+`gather`. Tree-LSTM (Tai et al. 2015) is the closest existing op but uses fixed gate matrices, not input-conditioned A/B/Δ — its computation graph is shallower and is decomposable into `nn.LSTMCell` + tree walk.

**Chess-specific motivation:**
The MCTS / alpha-beta search produces a *tree* of related positions, and Mamba's "what to forget" selection mechanism is exactly the inductive bias needed at a capture node ("forget the captured piece's evaluation contribution"). The S6 gate `Δ_v` is naturally interpretable as "move salience."

**Generalisation beyond chess:**
Any agent that produces a structured search tree: shogi/go MCTS, theorem-proving (Lean tactic trees), compiler super-optimisation, structured chain-of-thought reasoning trees.

**Complexity (forward, backward, incremental-update):**
- Forward: O(|V|·d·N) where N is SSM state dim — same as Mamba over the linearised DFS order; the win is logical, not asymptotic.
- Backward: O(|V|·d·N) with checkpointed state stack.
- Incremental update on extending tree by one leaf: O(d·N).

**Scout-scale falsification test:**
The scout dataset is single positions, so Tree-S6 needs synthetic tree contexts. Generate 12-ply tactical PVs from the same 173k positions (`stockfish go depth 12 pv`) and treat the PV as a degenerate-tree linear scan; baseline = a regular Mamba block consuming the same PV. Works if Tree-S6 ≥ baseline Mamba at matched parameter count on CRTK class 1 FP rate AND degrades < 5% when the "tree" widens by adding sibling refutations. Fails otherwise.

**Failure mode catalogue:**
- *Hidden rebrand:* if PVs are always linearised before consumption, Tree-S6 collapses to vanilla Mamba — must demonstrate a genuinely branching benchmark.
- *Numerically unstable:* the `A_v` product along a deep PV is `Π exp(Δ_v ⊙ A)`, prone to vanish/explode; Gu & Dao 2023 stabilise via discretisation, must reuse that machinery.
- *Too slow:* without a custom tree-scan kernel, the gather-based fallback is order-of-magnitude slower than `selective_scan_cuda`; on a 3070 this is the dominant risk.

**Status:** proposed, **underexplored primitive for chess.** Recent work (Mamba-ND, Wang et al. 2024 arXiv:2402.05892) extends S6 to multi-dimensional grid layouts via fixed orderings; tree-structured S6 is not, to my knowledge, in the 2024–2025 literature, but Tree-LSTM (Tai et al. 2015) is close enough that the novelty bar is "S6 selection on trees," not "any tree-RNN."

---

## 4. `primitive_wrec` — Wreath-Equivariant Chess-Group Convolution

### primitive_wrec

**Name:** Wreath-Equivariant Chess-Group Convolution (WrEC)

**One-line claim:** A group-convolution primitive equivariant to the chess symmetry group `D₄ × Z₂_color` acting on space *and* on a paired piece-type channel involution.

**Mathematical signature:**
Input feature map `f: ℤ_8² × C → ℝ^k` where C is split into 6 piece-type pairs `(P_white, P_black)`, plus side-to-move. Group action of `g = (σ, τ) ∈ D₄ × Z₂_color`:
`(g·f)(x, P_white_i) = f(σ⁻¹·x, P_{τ(white)}_i)`, with `τ` swapping each white/black channel pair AND flipping board orientation (because black-to-move is white-to-move on the reflected board).
Forward: a steerable convolution `(f * ψ)(x, c) = Σ_{y, c'} ψ((σ_g)^{-1}(x-y), c → c') f(y, c')` where ψ obeys the equivariance constraint `ψ(g·x, g·c → g·c') = ψ(x, c → c')` for all g. Implementation: kernel weights parameterised by an irrep basis of the chess group.

**Why this does not decompose into existing PyTorch ops:**
Existing PyTorch group-equivariant CNNs (e2cnn, escnn — Cesa & Weiler 2022) handle dihedral D₄ on spatial axes only. WrEC requires the group to act **jointly on space and on channels in a paired-involution fashion** — `τ` is not a free permutation of channels, it pairs white-i ↔ black-i and simultaneously negates the evaluation target. This wreath-product action is not expressible as a `Conv2d` followed by a channel-permutation tied weight, because the equivariance constraint couples spatial kernel weights to the channel-pair signature. Carroll & Beel (2020, "Finite Group Neural Networks for Games") build the analogue for **checkers** with reflection + color-swap but do not handle the chess-specific piece-type pair structure (knight, bishop, rook, queen each pair independently).

**Chess-specific motivation:**
Color-swap is *the* most-used data augmentation in chess engines and provides a 2× effective sample size for free at scout scale. Baking it into the operator removes the augmentation cost and guarantees consistency of `eval(pos) = −eval(swap(pos))`, which the i243 dual-stream proposal tried to enforce architecturally.

**Generalisation beyond chess:**
Any two-player zero-sum board game (shogi, draughts, Othello), particle physics with charge-conjugation symmetry (C-parity), electron-hole-symmetric Hamiltonians in condensed matter.

**Complexity (forward, backward, incremental-update):**
- Forward: same asymptotics as `Conv2d`, smaller constant (parameter sharing reduces |θ| by ~|G| = 16).
- Backward: O(|G|·Conv2d backward).
- Incremental update: not applicable (no state).

**Scout-scale falsification test:**
Drop WrEC into the conv stem of i193 (conv-only parent). Train 12 epochs × 173k. Baselines: (a) i193; (b) i193 with color-swap augmentation at 2× the data budget. Works if WrEC ≥ baseline (b) at matched wall-clock, demonstrating equivariance is at least as good as augmentation. Fails if it matches only (a) (then it's just regularisation).

**Failure mode catalogue:**
- *Hidden rebrand:* if `τ` is implemented as a fixed permutation of input planes and weights are not constrained, this becomes `Conv2d` with augmentation; must enforce the irrep parameterisation.
- *Numerically unstable:* over-constrained kernels can have rank-deficient irreps for small kernel sizes; mitigated by widening channels.
- *Too slow:* escnn-style kernel construction is slow at module-init; mitigated by caching the irrep basis once.

**Status:** proposed, **underexplored primitive for chess.** Carroll & Beel (2020) covers the checkers analogue; e2cnn/escnn cover general finite groups; the *chess* group with paired piece-type involution is not, to my reading of the 2020–2025 literature, instantiated as a published operator.

---

## 5. `primitive_shk` — Sparse Hyper-Kernel Generator

### primitive_shk

**Name:** Sparse Hyper-Kernel Generator (SHK)

**One-line claim:** A hypernetwork primitive that emits, per input, the *index set and values* of a sparse 8×8 convolution kernel via differentiable top-k.

**Mathematical signature:**
Input: context vector `c ∈ ℝ^d` (e.g. pooled board summary).
Hyper-MLP `H_φ(c) → (logits ∈ ℝ^{64}, values ∈ ℝ^{64·C_out·C_in})`.
Index selection: `I = top-k(logits)` via Gumbel-top-k with straight-through (Kool et al. 2019).
Output kernel: `K[i,j,:,:] = values[i,j]` if `(i,j) ∈ I` else 0, then `output = K * X` standard conv.
Backward: STE for `I`, vanilla autograd for `values` and `logits` (Gumbel surrogate).

**Why this does not decompose into existing PyTorch ops:**
Dynamic Convolution / CondConv (Chen et al. CVPR 2020, Yang et al. NeurIPS 2019) generates **dense** input-conditioned kernels, parameterised as a soft mixture of experts; the kernel support is fixed. SHK generates a kernel whose **support set is discrete and input-conditioned**, requiring the Gumbel-top-k discrete sampler as part of the computation graph and STE in backward. Hypernetworks (Ha et al. ICLR 2017) similarly produce dense weights. MoE (Shazeer et al. 2017, Zhou et al. 2022 Expert Choice) selects whole experts, not within-kernel offsets.

**Chess-specific motivation:**
Different positions need different receptive geometries: a king-safety net wants a 3×3 around the king; a passed-pawn net wants a 1×8 file. A static kernel must contain all and learn to gate; SHK picks the support per position. This is a primitive-level instantiation of the chess intuition that "what to look at" depends on "what's on the board."

**Generalisation beyond chess:**
Sparse / structured signal domains where the relevant spatial extent depends on content: medical imaging (lesion-shape-adaptive filters), point clouds (per-point neighbourhood selection), graph learning with input-dependent receptive fields.

**Complexity (forward, backward, incremental-update):**
- Forward: O(B·k·C_in·C_out·H·W) vs CondConv O(B·64·C_in·C_out·H·W); win when k ≪ 64.
- Backward: same, plus the Gumbel-top-k pass (negligible).
- Incremental update: not applicable.

**Scout-scale falsification test:**
Replace one mid-stack `Conv2d(3×3)` in i193 with SHK at k=9 (matched FLOPs). Baselines: (a) the same Conv2d; (b) Dynamic Convolution / CondConv with the same parameter count. Works if SHK beats (b) at parity FLOPs on CRTK class 1 FP rate by ≥ 5%, AND inference latency is within 1.2× of (a). Fails if either threshold missed — then it's a worse CondConv.

**Failure mode catalogue:**
- *Hidden rebrand:* if at inference time the top-k collapses to a fixed support (e.g. always picks the center 9 indices), it's just `Conv2d`. Audit by measuring the entropy of `I` across the validation set; need H(I) above a threshold (say 1 bit).
- *Numerically unstable:* Gumbel-top-k with low temperature is high-variance; needs annealing schedule, which is a *training trick*, not part of the primitive — flag.
- *Too slow:* per-example sparse kernels break cuDNN batching; needs `torch.nn.functional.conv2d` with per-sample weights via `groups` trick, which costs throughput. On 3070 at 173k positions × 12 epochs, this is the dominant risk.

**Status:** proposed, **underexplored primitive for chess.** Closest 2024–2025 prior art: dynamic / conditional convolution variants, and structured sparsity in convolution (e.g. group lasso). The discrete-support hypernetwork formulation is not standard.

---

## Ranking (a) novelty plausibility, (b) 3070-demonstrability, (c) inference-speed upside, (d) generalisability

| | novelty | 3070 demo | inference speed | generalises |
|---|---|---|---|---|
| ARSA | high (vs NSA: rule-based not learned) | **easy** (drop into i242) | high (sparse mask) | broad |
| DAP | medium (NNUE deploys it; novel as torch op) | **easy** (HalfKA stem swap) | **highest** (the whole point) | board games + sequential edit domains |
| Tree-S6 | medium (Tree-LSTM precedent; selection-on-trees fresh) | hard (needs branching benchmark) | medium | broad (search-tree agents) |
| WrEC | medium (escnn + Carroll&Beel cover most) | **easy** (i193 stem swap) | low (constant-factor) | physics, board games |
| SHK | medium (CondConv precedent) | medium (per-sample kernel kernel) | low/negative (batching hostile) | sparse signal domains |

**Top-2:** ARSA and DAP. Both pass the most aggressive devil's-advocate test (next section).

---

## Self-audit: trying to prove ARSA and DAP are hidden rebrands

**ARSA — devil's advocate:** "This is just MultiheadAttention with a custom mask tensor passed in."
**Rebuttal:** The mask is not a tensor argument — it is *part of the primitive's compute graph*: a deterministic function of the discrete input that produces a CSR layout with variable nnz per batch element. A vectorised PyTorch implementation that pre-builds the mask as a dense `B×N×N` tensor wastes the asymptotic and is *not* the primitive — that is a degenerate composition. The primitive's correctness condition is: the forward kernel must compute attention over `O(|E|)` edges only, with `|E| ≪ N²`. Native Sparse Attention (Yuan et al., arXiv:2502.11089) is the closest 2025 work but builds the sparse mask by *learned top-k of attention scores*, not by *deterministic rules on the discrete input*. Different gradient flow (no top-k surrogate needed in ARSA, since the geometry is detached). **Survives.**

**DAP — devil's advocate:** "This is just `torch.sparse.mm` called incrementally; the user manages state outside."
**Rebuttal:** The primitive's defining feature is the **paired (apply, unmake) operator with a backward graph that tracks the make-stack**. `torch.sparse.mm` is stateless and produces a single matmul DAG; running it incrementally outside the autograd boundary does not preserve gradients w.r.t. W through the temporal chain. The operator-pair semantics is what NNUE deploys (Stockfish `AccumulatorStack`), but no torch.nn.<Op> publishes it with autograd. The novelty is therefore "underexplored primitive for chess" in the literal PyTorch sense, while in the NNUE deployment sense it is well-known. The proposal is explicitly framed that way. **Survives, with the underexplored caveat.**

Neither needs replacement.

---

## What I cut

1. **Color-Swap-Equivariant Linear** — a linear layer with weights constrained to `W = ±P W P`. This *does* decompose into a vanilla `nn.Linear` plus weight-tying via reparameterisation; no new computation graph. Rejected as hidden rebrand.

2. **Chess-Specific Activation (e.g. piecewise GELU around evaluation = 0)** — the prompt explicitly disqualifies activation-function tweaks. Rejected as anti-example.

3. **Learned Conv+Attention Mixer for Chess** — composition of existing ops; the proposal would just be a gated sum of Conv2d and MultiheadAttention outputs. Rejected as architectural.

4. **Dihedral-only Group Convolution** — already shipped in `escnn` (Cesa & Weiler 2022); proposing it for chess adds no operator-level novelty over the existing library. Rejected as already-implemented primitive.

5. **Householder Reflection Memory for MCTS** — this is essentially DeltaNet (Yang et al., NeurIPS 2024) applied to a search trajectory; not novel enough at the primitive level. Rejected, though the chess deployment angle is interesting; would be an *architecture*, not a *primitive*.

---

## References (real papers cited)

- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.
- Dao, T. & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2). ICML 2024.
- Yang, S., Wang, B., Zhang, Y., Shen, Y. & Kim, Y. (2024). *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS 2024. arXiv:2406.06484.
- Yang, S., Kautz, J. & Hatamizadeh, A. (2024). *Gated Delta Networks: Improving Mamba2 with Delta Rule.* arXiv:2412.06464.
- Yuan, J. et al. (2025). *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention.* arXiv:2502.11089 (DeepSeek).
- Puigcerver, J., Ruiz, C. R., Mustafa, B. & Houlsby, N. (2024). *From Sparse to Soft Mixtures of Experts.* ICLR 2024.
- Zhou, Y. et al. (2022). *Mixture-of-Experts with Expert Choice Routing.* NeurIPS 2022. arXiv:2202.09368.
- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR 2017.
- Ha, D., Dai, A. & Le, Q. V. (2017). *HyperNetworks.* ICLR 2017.
- Chen, Y. et al. (2020). *Dynamic Convolution: Attention over Convolution Kernels.* CVPR 2020.
- Yang, B. et al. (2019). *CondConv: Conditionally Parameterized Convolutions for Efficient Inference.* NeurIPS 2019.
- Kool, W., van Hoof, H. & Welling, M. (2019). *Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement.* ICML 2019.
- Lee, J. et al. (2019). *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks.* ICML 2019. arXiv:1810.00825.
- Zaheer, M. et al. (2017). *Deep Sets.* NeurIPS 2017.
- Tai, K. S., Socher, R. & Manning, C. D. (2015). *Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks.* ACL 2015.
- Cesa, G. & Weiler, M. (2022). *A Program to Build E(N)-Equivariant Steerable CNNs* (escnn). ICLR 2022.
- Carroll, O. & Beel, J. (2020). *Finite Group Equivariant Neural Networks for Games.* arXiv:2009.05027.
- Wang, X. et al. (2024). *Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data.* arXiv:2402.05892.
- Nasu, Y. (2018). *Efficiently Updatable Neural-Network-based Evaluation Function for Computer Shogi.* (NNUE original; see also Stockfish 12 release notes and `official-stockfish/nnue-pytorch` docs.)
- Hendrycks, D. & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs).* arXiv:1606.08415. (Cited as calibration anchor for what a primitive looks like.)
- Ba, J. L., Kiros, J. R. & Hinton, G. E. (2016). *Layer Normalization.* arXiv:1607.06450. (Calibration anchor.)
- Kingma, D. P. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR 2015. (Calibration anchor.)
- Jacobs, R. A., Jordan, M. I., Nowlan, S. J. & Hinton, G. E. (1991). *Adaptive Mixtures of Local Experts.* Neural Computation 3(1). (Calibration anchor.)
- Peng, B. et al. (2023). *RWKV: Reinventing RNNs for the Transformer Era.* EMNLP Findings 2023. (Calibration anchor.)

---

**Caveats and honesty notes (not part of the schema, but the user asked for structural honesty):**

- No empirical numbers in any falsification test are simulated. The "≥ 0.9× FP rate" / "≥ 3× latency" thresholds are *targets the experiment would test*, not predicted outcomes.
- The novelty bar is calibrated *relative to publicly indexed 2020–2025 ML literature I was able to retrieve*. A 2025 paper I missed could collapse any of these to "underexplored primitive for chess." That risk is highest for Tree-S6 (the closest prior art, Tree-LSTM, is 10 years old, but somebody may have published "Tree-Mamba" between my search cutoff and now) and for SHK (sparse-support dynamic convolution is the kind of thing a workshop paper might have shipped).
