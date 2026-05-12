# New Neural Primitives for Chess Evaluation

## Scope and novelty bar

I treated your constraints as hard filters rather than suggestions. Every candidate below is an operator proposal, not an architecture, not an encoding, and not a training trick. The recurring questions I used were: does the operator fundamentally change the computation graph; does it exploit chess structure that ordinary conv/attention layers blur away; does it have a plausible O(Δ)-style update path when only a bounded part of the board changes; and can it be falsified quickly on a single 3070-class GPU.

## Survey signals from recent primitive work

The recent literature sets a genuinely high bar for what counts as a new primitive. Mamba changed the state-update law by making state-space parameters input-selective; Prefix-Scannable Models widened that family by treating efficient scan-style recurrences themselves as primitive objects; Hypergraph-Native Message Passing made incidences rather than nodes or edges the carrier of computation; GeDi-HNN introduced a new directed-hypergraph Laplacian as the primitive operator; ASEN turned subgroup selection into an explicit modeling object; and semiring-based neural operators argued that algebraic composition laws themselves can be trainable primitives. Sources: [Mamba / selective state spaces](https://openreview.net/forum?id=AL1fq05o7H), [Prefix-Scannable Models](https://openreview.net/forum?id=tuLF84azND), [Hypergraph-Native Message Passing](https://openreview.net/pdf?id=eRu0UBXEh2), [GeDi-HNN](https://openreview.net/forum?id=h48Ri6pmvi), [ASEN-style subgroup selection](https://openreview.net/forum?id=jz3d7nvtGz), and [semiring-based neural operators](https://arxiv.org/abs/2405.18805).

## Ranked proposals

The list is ranked strongest-to-weakest overall on novelty plausibility, 3070-scale demonstrability, inference-speed upside, and non-chess reuse. The top two survived a devil’s-advocate pass against the most obvious “this is just X” objections; the ideas I could not defend are listed in “What I cut.”

1.

### primitive_signed_edit_bilinear_memory

**Name:** Signed-Edit Bilinear Memory

**One-line claim:** Exact insert/delete primitive that preserves additive and pairwise set statistics in time proportional to the number of edits.

**Mathematical signature:**
For each batch item, maintain state \((s_t, u_t, p_t) \in \mathbb{R}^{r} \times \mathbb{R}^{r} \times \mathbb{R}^{r}\).  
Input edit list \(E_t=\{(x_j,\sigma_j)\}_{j=1}^{k}\), with \(x_j \in \mathbb{R}^{d}\) and \(\sigma_j \in \{-1,+1\}\). Let \(a_j=A x_j \in \mathbb{R}^{r}\), \(b_j=B x_j \in \mathbb{R}^{r}\).

If \(\sigma_j=+1\) (insert):

\[
p \leftarrow p + a_j \odot u + b_j \odot s,\quad
s \leftarrow s + a_j,\quad
u \leftarrow u + b_j .
\]

If \(\sigma_j=-1\) (delete):

\[
p \leftarrow p - a_j \odot (u-b_j) - b_j \odot (s-a_j),\quad
s \leftarrow s - a_j,\quad
u \leftarrow u - b_j .
\]

Return \(z_t=[s_t \,\|\, u_t \,\|\, p_t] \in \mathbb{R}^{3r}\). Gradients are well-defined w.r.t. \(A,B\) and every edited feature vector.

**Why this does not decompose into existing PyTorch ops:**
Its contract is not “pool a set” or “scan a sequence”; it is “maintain an exact learned multiset state under signed edits.” Standard sum pooling recomputes from the active set, while scan/state-space primitives such as [Mamba](https://openreview.net/forum?id=AL1fq05o7H) and later [scan-model generalizations](https://openreview.net/forum?id=tuLF84azND) are append-style recurrences over a total order, not inverse-consistent insert/delete operators over a sparse multiset. Dynamic-graph learning is also moving toward update-local inference, but current models learn update procedures around existing GNN backbones rather than exposing an exact reusable signed-edit bilinear primitive; see recent dynamic-update work such as [OpenReview: dynamic graph update-local inference](https://openreview.net/forum?id=nGbhxxdhqz) and related 2025 work on dynamic graph learning ([arXiv PDF](https://arxiv.org/pdf/2505.13754)).

**Chess-specific motivation:**
This is the cleanest generalization of HalfKA’s core advantage. A chess move toggles a tiny number of piece-square relations, so additive accumulators are fast; the missing piece is pair structure. SEBM keeps cheap edit-local updates while exposing second-order interactions such as blocker+slider, attacker+defender, or king-ring+intruder without paying all-pairs cost.

**Generalisation beyond chess:**
It should transfer to any sparse event domain where insertions and deletions dominate full recomputation: dynamic recommenders, evolving graphs, and event-camera pipelines are the clearest examples. Event-based vision and dynamic-graph work both already reward update-local computation; see [dynamic graph update-local inference](https://openreview.net/forum?id=nGbhxxdhqz), [stepwise incremental inference in spiking/event streams](https://openreview.net/forum?id=LUnYc9Grm8), and related event-stream work ([OpenReview](https://openreview.net/forum?id=HyePrhR5KX)).

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(nr)\)  vs closest existing primitive \(O(n^2 r)\) for pairwise attention
- Backward: \(O(nr)\)
- Incremental update on a bounded-change input: \(O(kr)\), where \(k\) is the edit count

**Scout-scale falsification test:**
Use **i243 HalfKA+dual-stream** as the harness. Replace only the additive HalfKA accumulator with SEBM at matched hidden width, initialize it to approximate the additive baseline, and fine-tune only the primitive plus the first post-accumulator projection for 90 minutes on a fixed 32k-position subset. Baseline: the existing additive accumulator. Metric: CRTK class-1 matched-recall FP rate; secondary metric: positions/second on parent→child incremental evaluation. “Works” means either class-1 FP rate drops by at least 5% relative at matched recall with no worse than 15% full-eval slowdown, or incremental child evaluation is at least 1.5× faster than recomputing the same second-order features from scratch.

**Failure mode catalogue:**
- If \(p_t\) is removed or never used, this collapses to a glorified signed sum accumulator and becomes a rebrand.
- If \(r\) is too wide, \(p_t\) can dominate numerically and drown the additive pathway.
- If the CUDA kernel is not fused, edit-list overhead can erase the theoretical O(\(\Delta\)) advantage.

**Status:** proposed

2.

### primitive_first_blocker_ray_scan

**Name:** First-Blocker Ray Scan

**One-line claim:** Learned line-of-sight operator that separates empty-ray context from first-contact blocker information.

**Mathematical signature:**
Let \(X \in \mathbb{R}^{B \times 64 \times d}\) be square features and \(o \in \{0,1\}^{B \times 64}\) the occupancy bits.  
For square \(s\), direction \(\rho \in \mathcal{R}=\{\text{N,S,E,W,NE,NW,SE,SW}\}\), and path \(P_{s,\rho}=(v_1,\ldots,v_L)\):

\[
m_{s,\rho,k}=\prod_{j<k}(1-o_{v_j}), \qquad
b_{s,\rho,k}=o_{v_k}\,m_{s,\rho,k}.
\]

Then

\[
y^{\text{free}}_{s,\rho}=\sum_{k=1}^{L} m_{s,\rho,k} A_{\rho,k} X_{v_k},
\qquad
y^{\text{hit}}_{s,\rho}=\sum_{k=1}^{L} b_{s,\rho,k} B_{\rho,k} X_{v_k}.
\]

Output

\[
Y_s=\operatorname{concat}_{\rho \in \mathcal{R}}\!\left(y^{\text{free}}_{s,\rho},\,y^{\text{hit}}_{s,\rho}\right).
\]

Gradients are well-defined w.r.t. \(X, A, B\); \(o\) is a discrete input, so no parameter gradient is required through occupancy.

**Why this does not decompose into existing PyTorch ops:**
A standard convolution has a fixed receptive field and cannot natively stop at the *first* blocker; dense attention can connect all pairs but does not make first-contact visibility a primitive connectivity rule. Classical scan literature already showed that line-of-sight on a ray is a scan problem, and recent prefix-scannable models re-elevate scan laws to primitive status, but neither gives a learned grid operator with explicit free-space and first-contact channels plus edit-local update semantics. Sources: [Blelloch, “Prefix Sums and Their Applications”](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf) and [Prefix-Scannable Models](https://openreview.net/forum?id=tuLF84azND).

**Chess-specific motivation:**
Pins, skewers, batteries, trapped rooks, open files, bishop diagonals, and discovered attacks all care about the first blocker, not generic local texture. One move typically changes only the rays crossing the source and destination squares, so the natural update cost is proportional to touched rays rather than to the whole board.

**Generalisation beyond chess:**
The same operator is plausible for occupancy grids, robot navigation, lidar/radar interpretation, and occlusion-aware scene reasoning, where line-of-sight is structurally prior to generic spatial mixing. The scan connection is especially natural there; see [Blelloch’s scan survey](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf).

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(n D L h)\)  vs closest existing primitive \(O(n D L h)\) for masked local conv, but without first-hit semantics
- Backward: \(O(n D L h)\)
- Incremental update on a bounded-change input: \(O(|\Delta\text{rays}|\,L\,h)\)

**Scout-scale falsification test:**
Use **i193** as the harness. Replace only the first two spatial mixing operations with FRBS, keep the stem and head unchanged, and fine-tune the swapped blocks plus the readout for 90 minutes on a fixed 32k-position subset. Baseline: the unchanged i193 blocks at the same channel count. Measure CRTK class-1 matched-recall FP rate and full-position wall-clock latency. “Works” means class-1 FP rate improves by at least 5% relative at matched recall with no more than 10% full-eval slowdown, or child-node incremental evaluation reaches a clear >2× speedup by updating only touched rays.

**Failure mode catalogue:**
- If implemented as a generic masked 1D conv with no explicit first-contact split, it is just a rebrand.
- If soft occupancy is substituted carelessly, cumulative products can become numerically brittle on longer rays.
- If kernels are not fused across directions, launch overhead can dominate the tiny 8×8 workload.

**Status:** proposed

3.

### primitive_typed_ordered_hyperedge_exchange

**Name:** Typed Ordered Hyperedge Exchange

**One-line claim:** Makes an ordered, typed hyperedge the native message carrier for sparse legal-move relations.

**Mathematical signature:**
Input node features \(X \in \mathbb{R}^{B \times n \times d}\) and per-sample hyperedge family

\[
\mathcal{H}_b=\{e=(u_0,\ldots,u_{\ell_e-1},\tau_e)\},
\]

where \(u_j\) are ordered slots and \(\tau_e\) is a hyperedge type.

Hyperedge state:

\[
h_e=\Phi_{\tau_e}\!\left(\bigoplus_{j=0}^{\ell_e-1} R_j X_{u_j}\right)\in\mathbb{R}^{h}.
\]

Node update:

\[
Y_v=\Psi\!\left(X_v,\; \bigoplus_{e,j:\,u_j=v}\Gamma_j h_e\right)\in\mathbb{R}^{h}.
\]

Here \(R_j,\Gamma_j\) are slot-specific maps; \(\bigoplus\) is any differentiable commutative aggregator over the hyperedge set.

**Why this does not decompose into existing PyTorch ops:**
The closest literature is already quite close: Hypergraph-Native Message Passing makes incidences first-class, and GeDi-HNN handles directed hyperedges. The surviving novelty claim is narrower and more specific: *ordered path slots* are primitive objects here, not unordered incidences or head/tail sets. Standard attention is pairwise and must materialize token-token interactions; standard hypergraph layers flatten away slot order or treat only source/target asymmetry. TOHE instead elevates source / path-step / destination roles into the operator signature itself. Sources: [Hypergraph-Native Message Passing](https://openreview.net/pdf?id=eRu0UBXEh2), [GeDi-HNN](https://openreview.net/forum?id=h48Ri6pmvi), and related hypergraph message-passing work ([OpenReview](https://openreview.net/forum?id=eRu0UBXEh2)).

**Chess-specific motivation:**
A legal chess move is not “an edge.” It is an ordered tuple: source, possibly several traversed squares, destination, and move type. Recent chess graph work already found it natural to represent moves as edges, but still relied on GAT-like pairwise machinery; this pushes one level deeper by making move-paths native objects. That matches the prompt’s strongest chess-specific signal: sparse, position-conditioned legal connectivity. Source: [chess graph representation work](https://arxiv.org/html/2410.23753v1).

**Generalisation beyond chess:**
Any domain with typed paths or routes can use this: road networks, circuits, program traces, instruction graphs, and multi-hop relational reasoning.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O((n+\sum_e \ell_e)h)\)  vs closest existing primitive \(O(n^2 h)\) for dense square-token attention
- Backward: \(O((n+\sum_e \ell_e)h)\)
- Incremental update on a bounded-change input: \(O\!\left(\sum_{e\in\Delta\mathcal{H}}\ell_e h\right)\)

**Scout-scale falsification test:**
Use **i242** as the harness. Replace exactly one decomposed-attention block over square tokens with TOHE built from the current legal-move hyperedges; keep the stem, readout, and all other blocks unchanged. Fine-tune only the swapped block plus its adjacent projections for 90 minutes on a fixed 24k-position subset. Baseline: the original i242 block. Metric: CRTK class-1 matched-recall FP rate; secondary metric: per-position latency. “Works” means class-1 FP drops by at least 5% relative at matched recall and latency is at worst parity, or latency improves by at least 1.3× with no measurable class-1 degradation.

**Failure mode catalogue:**
- If slot order is pushed into ordinary positional encodings, this collapses toward existing hypergraph message passing.
- If legal-move generation or path materialization dominates runtime, the sparse advantage disappears.
- If role-specific parameters proliferate with path length, the primitive becomes too specialized and loses reuse value.

**Status:** proposed

4.

### primitive_alternating_option_scan

**Name:** Alternating Option Scan

**One-line claim:** Runs a differentiable optional-stop reverse scan over alternating tactical exchange chains.

**Mathematical signature:**
Input an ordered alternating chain \(r=(r_1,\ldots,r_m)\), with \(r_k \in \mathbb{R}^{B \times h}\) an embedding of the \(k\)-th exchange offer/counteroffer.

Reverse recurrence:

\[
g_m=r_m,\qquad
g_k=\tau \log\!\left(1+\exp\!\left(\frac{r_k-g_{k+1}}{\tau}\right)\right), \quad k=m-1,\ldots,1.
\]

Output \(y=g_1\).  
As \(\tau \to 0\), this approaches the hard recurrence \(g_k=\max(0, r_k-g_{k+1})\), i.e. “continue only if the next reply is not good enough.”

**Why this does not decompose into existing PyTorch ops:**
Semiring activations and algebraic-path operators already show that alternative algebraic composition laws can form primitives, and scan-model theory now treats custom recurrences as primitive-worthy rather than merely implementation detail. The claim here is narrower: this is a *reverse antagonistic scan with optional termination*, not a pointwise semiring, not ordinary dynamic programming over homogeneous path sums, and not a generic affine/gated RNN cell. Its Jacobian is chain-coupled and one-sided: every term competes directly with the recursively summarized best reply. Sources: [semiring-based neural operators](https://arxiv.org/abs/2405.18805), [Learning with Semiring Neural Networks / related PMLR work](https://proceedings.mlr.press/v162/sanmarti-n22a/sanmarti-n22a.pdf), and [Prefix-Scannable Models](https://openreview.net/forum?id=tuLF84azND).

**Chess-specific motivation:**
Hard negatives near puzzles are often not about global structure; they are about whether a local exchange sequence actually works. Classical static exchange evaluation succeeds because exchanges are alternating and optional. AOS is the soft, differentiable primitive version of that idea, aimed directly at class-1 discrimination. Source: [Chessprogramming.org: Static Exchange Evaluation](https://www.chessprogramming.org/Static_Exchange_Evaluation).

**Generalisation beyond chess:**
This should transfer to any ordered duel chain: attacker-defender resource allocation, bidding wars, multistage negotiation, and packet-filter or rule-priority cascades.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(mh)\)  vs closest existing primitive \(O(m^2 h)\) for local attention over the exchange chain
- Backward: \(O(mh)\)
- Incremental update on a bounded-change input: \(O(\Delta m\,h)\)

**Scout-scale falsification test:**
Use **i193** as the harness. Derive ordered exchange chains from the current board’s attack map, replace the final local tactical pooling step with one AOS primitive over occupied squares only, and fine-tune just that step plus the final head for 60 minutes on a fixed 32k-position subset. Baseline: a same-parameter MLP on the same chain embeddings. Metric: CRTK class-1 matched-recall FP rate. “Works” means class-1 FP falls by at least 8% relative at matched recall even if aggregate PR AUC stays flat; “fails” means it improves mostly on easy negatives while class-1 stays unchanged.

**Failure mode catalogue:**
- If the operator is relaxed into an ordinary learned recurrent cell, it becomes a rebrand of sequence processing.
- If chain extraction is noisy or unstable, the primitive inherits garbage ordering and learns little.
- If \(\tau\) is too small too early, gradients can vanish almost everywhere except at exchange boundaries.

**Status:** proposed

5.

### primitive_approximate_stabilizer_irrep_projector

**Name:** Approximate Stabilizer Irrep Projector

**One-line claim:** Projects features into symmetry channels weighted by the current sample’s approximate chess-group stabilizer.

**Mathematical signature:**
Let \(G\) be a small finite transformation group acting by permutations \(\pi_g\) on board indices/channels. Let \(\mathrm{sig}(X_{\mathrm{hard}})\) be a discrete board signature. Define

\[
s_g=-\beta\, d\!\big(\mathrm{sig}(X_{\mathrm{hard}}), \pi_g\,\mathrm{sig}(X_{\mathrm{hard}})\big),\qquad
\omega_g=\frac{e^{s_g}}{\sum_{h\in G} e^{s_h}} .
\]

For chosen characters \(\eta_r: G \rightarrow \{-1,+1\}\) or a small set of low-dimensional irreps,

\[
Y_r=\sum_{g\in G}\omega_g\,\eta_r(g)\,\pi_g X W_r,
\qquad
Y=\operatorname{concat}_r Y_r .
\]

Gradients are well-defined w.r.t. \(X, W_r,\beta\); the signature can remain non-differentiable.

**Why this does not decompose into existing PyTorch ops:**
This is the proposal with the highest novelty risk, so I am stating the overlap explicitly. Fixed group-equivariant layers, graph-automorphism-equivariant layers, and ASEN-style subgroup selection already exist. The surviving claim is only this: ASIP does *not* choose a symmetry group a priori or via an auxiliary symmetry-breaking input; it computes soft irrep projectors from the *current sample’s own approximate self-symmetry* under a finite group and changes the computation graph whenever those stabilizer weights change. That is the operator-level delta. Sources: [graph automorphism equivariance](https://openreview.net/forum?id=vjkq5fwsj3), [ASEN-style subgroup selection](https://openreview.net/forum?id=jz3d7nvtGz), and related equivariance work ([OpenReview](https://openreview.net/forum?id=4v4nmYWzBa)).

**Chess-specific motivation:**
Chess has partial symmetries beyond D4: board flips, color swap, and some approximate role symmetries show up locally even when the whole board is not exactly symmetric. This is the most explicit attempt to exploit your “chess group” prior for small-data efficiency rather than raw scale.

**Generalisation beyond chess:**
Near-symmetric molecules, traffic systems with lane/reflection structure, and repeated motifs in programs or routing graphs are the most plausible non-chess targets. Equivariance work consistently treats such priors as regularizing structure; see [equivariance work on OpenReview](https://openreview.net/pdf?id=f7YjBggjtz) and [graph automorphism equivariance](https://openreview.net/forum?id=vjkq5fwsj3).

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(|G|nh)\)  vs closest existing primitive \(O(|G|nh)\) for fixed-group averaging
- Backward: \(O(|G|nh)\)
- Incremental update on a bounded-change input: \(O(|G|\,\Delta h)\)

**Scout-scale falsification test:**
This is the weakest scout-scale bet. Use **i193** as the harness, insert one ASIP directly after the stem with a conservative group \(G=D_4 \times C_2\) (board symmetries plus color swap), and fine-tune only the stem, ASIP, and head for 90 minutes on a fixed 32k-position subset. Baseline: same-width \(1\times1\) channel mixing. Metric: CRTK class-1 matched-recall FP rate. “Works” means at least 3% relative class-1 FP reduction with under 10% slowdown. A null result at scout scale should *not* kill the idea for engine-scale data.

**Failure mode catalogue:**
- If \(\omega_g\) is made fixed, this collapses back to ordinary group averaging.
- If most positions induce effectively trivial stabilizers, the operator becomes near-identity too often.
- If approximate symmetry scores are noisy, the projector can inject instability instead of bias.

**Status:** proposed

## What I cut

- **Pure signed-edit sum accumulator.** I dropped this during self-audit because without the pair state it reduces to signed additive pooling plus bookkeeping, which is too close to scatter-add / DeepSets-style accumulation to clear the novelty bar.

- **Piece-conditioned legal-move attention.** This failed immediately: once the legal graph is treated as a mask or sparsifier over token-token scores, it is still just masked attention.

- **Exact subgroup projector over the chess group.** I initially wanted a hard stabilizer projector, but after checking recent subgroup-equivariant work I could not defend it as new enough relative to ASEN-style subgroup selection and graph-automorphism-equivariant layers. That is why the surviving symmetry proposal is the softer, sample-conditioned irrep projector instead. Sources: [ASEN-style subgroup selection](https://openreview.net/forum?id=jz3d7nvtGz) and [graph automorphism equivariance](https://openreview.net/forum?id=vjkq5fwsj3).

- **Directed move-hypergraph Laplacian convolution.** I cut this because GeDi-HNN and HMP already occupy too much of that design space. The surviving hypergraph proposal only clears the bar if ordered path slots, not merely incidences or head/tail direction, are promoted to first-class operator slots. Sources: [GeDi-HNN](https://openreview.net/forum?id=h48Ri6pmvi) and [Hypergraph-Native Message Passing](https://openreview.net/pdf?id=eRu0UBXEh2).

- **Sinkhorn legal-transport operator.** I cut it because in the end it decomposed into ordinary score computation plus iterative normalization, and I could not give it a compelling bounded-change update path.

## Source links

- [Linear-Time Sequence Modeling with Selective State Spaces / Mamba](https://openreview.net/forum?id=AL1fq05o7H)
- [Prefix-Scannable Models](https://openreview.net/forum?id=tuLF84azND)
- [Hypergraph-Native Message Passing](https://openreview.net/pdf?id=eRu0UBXEh2)
- [GeDi-HNN](https://openreview.net/forum?id=h48Ri6pmvi)
- [ASEN-style subgroup selection](https://openreview.net/forum?id=jz3d7nvtGz)
- [Semiring-based neural operators](https://arxiv.org/abs/2405.18805)
- [Dynamic graph update-local inference](https://openreview.net/forum?id=nGbhxxdhqz)
- [Related dynamic graph learning work](https://arxiv.org/pdf/2505.13754)
- [Stepwise Incremental Inference with Early-Exit in Spiking Neural Networks](https://openreview.net/forum?id=LUnYc9Grm8)
- [Related event-stream work](https://openreview.net/forum?id=HyePrhR5KX)
- [Blelloch, “Prefix Sums and Their Applications”](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
- [Chess graph representation work](https://arxiv.org/html/2410.23753v1)
- [Learning with Semiring Neural Networks / related PMLR work](https://proceedings.mlr.press/v162/sanmarti-n22a/sanmarti-n22a.pdf)
- [Chessprogramming.org: Static Exchange Evaluation](https://www.chessprogramming.org/Static_Exchange_Evaluation)
- [Graph automorphism equivariance](https://openreview.net/forum?id=vjkq5fwsj3)
- [Related equivariance work](https://openreview.net/forum?id=4v4nmYWzBa)
- [Additional equivariance work](https://openreview.net/pdf?id=f7YjBggjtz)
