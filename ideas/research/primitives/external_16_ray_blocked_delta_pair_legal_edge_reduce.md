# Deep Research Primitive Results

## Scope and evidence guardrails

This document follows the supplied constraints: propose neural-network primitives only, avoid architecture/input-encoding/training-trick proposals, respect the scout-scale RTX 3070 constraint, and treat near-puzzle false positives as the preferred discriminator.

The recent primitive-level literature that sets the bar is mostly about new state/update laws rather than new block diagrams: Mamba makes SSM parameters input-dependent; Mamba-2 reframes SSMs and attention through structured semiseparable matrices; xLSTM changes recurrent memory/gating; Gated DeltaNet combines adaptive forgetting with delta-rule memory updates; KAN replaces scalar weights with learned edge functions; Titans introduces a neural long-term memory module updated at test time.

## Ranking summary

| Rank | Primitive | Novelty plausibility | RTX 3070 demonstrability | Inference-speed upside | Generalisation beyond chess | Main reason |
|---:|---|---|---|---|---|---|
| 1 | `primitive_ray_blocked_scan` | High | High | High | High | Uses blocker-sensitive semiring scans instead of fixed conv/attention connectivity. |
| 2 | `primitive_delta_pair_accumulator` | Medium-high | Very high | Very high | High | Generalises NNUE-style bounded-change updates to exact second-order sparse interactions. |
| 3 | `primitive_legal_edge_reduce` | High | High | High | Medium | Connectivity is generated inside the op from board content, not passed as a static mask. |
| 4 | `primitive_orbit_action_norm` | Medium | High | Medium | High | Normalisation over finite-group orbits rather than batch/channel/feature axes. |
| 5 | `primitive_soft_see_reducer` | High but chess-narrow | Medium | Medium | Low-medium | A differentiable alternating capture reducer aimed directly at tactical hard negatives. |

## Top-2 self-audit before final proposals

### Devil’s advocate on `primitive_ray_blocked_scan`

A reviewer could argue this is “just `cumprod` along rays plus a weighted sum,” and that differentiable transmittance products already appear in neural rendering. NeRF-style rendering indeed uses differentiable volume rendering along rays, with transmittance accumulated along samples. I do **not** think that proves a hidden rebrand: this primitive is a finite-board, multi-direction, blocker-sensitive semiring scan with exact sparse incremental updates under bounded board changes. PyTorch has cumulative ops, but no `torch.nn` operator with this input/output signature, connectivity pattern, and event-update contract.

### Devil’s advocate on `primitive_delta_pair_accumulator`

A reviewer could argue this is a Factorization Machine in NNUE clothing: the second-order term resembles classic factorized pairwise interactions, and FMs are explicitly designed for sparse pair interactions. That objection is strong. I keep the proposal because the primitive claim is not “pairwise factorisation is new”; it is “a stateful bounded-delta neural operator with exact insert/delete updates, sparse changed-row backward, and a persistent accumulator is not an existing PyTorch primitive.” If the review standard requires new algebra independent of the update API, this is the easiest proposal to reject.

## 1. Proposal

### primitive_ray_blocked_scan

**Name:** Occlusion Semiring Ray Scan

**One-line claim:** A blocker-aware scan operator that propagates features only through unobstructed rays.

**Mathematical signature:**
$f:\mathbb{R}^{B\times64\times d}\times[0,1]^{B\times64}\rightarrow\mathbb{R}^{B\times64\times d}$.
For square $s$, direction $r$, ordered ray cells $c_{s,r,1:L}$:

$$
T_{b,s,r,\ell}=\prod_{q<\ell}(1-o_{b,c_{s,r,q}}),\quad
y_{b,s}=\sum_r\sum_{\ell=1}^{L}T_{b,s,r,\ell}A_r x_{b,c_{s,r,\ell}}.
$$

Gradients are standard through products for continuous $o$; for binary occupancy, gradients flow through $x,A$ only.

**Why this does not decompose into existing PyTorch ops:**
Closest comparison is `Conv2d` or masked attention. `Conv2d` has fixed local connectivity; `MultiheadAttention` takes a supplied mask but still computes query-key softmax attention over that mask. Here, ordered blocker transmittance is generated inside the primitive and has prefix-product gradient flow plus bounded event updates.

**Chess-specific motivation:**
Sliding pieces are not local 3×3 phenomena: a bishop, rook, or queen ray is active until the first blocker. This operator gives the network a native line-of-sight primitive without asking small data to rediscover ray occlusion from convolutions.

**Generalisation beyond chess:**
Useful for grid visibility, robotics line-of-sight, circuit routing, tactical games, sparse ray casting, and 2D/3D scene graphs with occluders.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(Bn\rho d)$ with diagonal $A_r$, or $O(Bn\rho d^2)$ with dense $A_r$, vs attention $O(Bn^2d)$.
- Backward: same order as forward.
- Incremental update on a bounded-change input: $O(d\log n)$ with ray segment trees; $O(d)$ on fixed 8×8 boards.

**Scout-scale falsification test:**
Drop one instance into i193 by replacing one mid-level 3×3/7×7 convolution with this operator plus a matched linear projection. Baseline: same i193 budget with ordinary convolution. Metric: matched-recall CRTK class-1 near-puzzle false-positive rate and eval/sec. Works if near-puzzle FP drops by at least 5% at matched recall without >20% eval/sec loss; fails if only aggregate PR AUC improves.

**Failure mode catalogue:**
- Hidden rebrand: reviewer says it is just `cumprod` plus gather-sum, not a primitive.
- Numerical instability: long products can saturate; use log-domain scan or clamp $o\in[10^{-4},1-10^{-4}]$.
- Speed risk: dense $A_r$ makes it slower than conv; diagonal/low-rank kernels should be the scout default.

**Status:** proposed

## 2. Proposal

### primitive_delta_pair_accumulator

**Name:** Bounded-Delta Pair Accumulator

**One-line claim:** An NNUE-like accumulator that updates exact first- and second-order sparse feature interactions after small input changes.

**Mathematical signature:**
Parameters $E,U,V\in\mathbb{R}^{F\times d}$. State $S=(\ell,p,q,r)\in(\mathbb{R}^{B\times d})^4$. Inputs are inserted/deleted feature IDs $\Delta^\pm\in\{1,\dots,F\}^{B\times k}$ with signs $\epsilon_i\in\{+1,-1\}$:

$$
\ell'=\ell+\sum_i\epsilon_iE_i,\quad
p'=p+\sum_i\epsilon_iU_i,\quad
q'=q+\sum_i\epsilon_iV_i,\quad
r'=r+\sum_i\epsilon_i(U_i\odot V_i),
$$

$$
y=[\ell',\;p'\odot q'-r']\in\mathbb{R}^{B\times2d}.
$$

Gradients touch only changed rows plus the persistent state adjoint.

**Why this does not decompose into existing PyTorch ops:**
`EmbeddingBag` sums sparse embeddings but is stateless and first-order. PyTorch lists sparse layers such as `Embedding` and `EmbeddingBag`, but no insert/delete accumulator with second-order sufficient statistics. The computation graph is over feature deltas and persistent state, not over the full active feature set.

**Chess-specific motivation:**
NNUE’s speed comes from sparse inputs and small board deltas between moves. Stockfish’s NNUE documentation frames the design around minimal input changes and efficient updates. Pair terms target interactions such as king-piece, piece-piece, pin, fork, and mutual-defense patterns without recomputing all pairs.

**Generalisation beyond chess:**
Sparse recommender systems, dynamic sets, event streams, program-analysis features, and any binary-feature model with small edits per step.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(Bkd)$ vs stateless active-set recompute $O(B|S|d)$ and dense linear $O(BFd)$.
- Backward: $O(Bkd)$ changed-row parameter updates plus state gradient.
- Incremental update on a bounded-change input: $O(kd)$; with bounded $k$, $O(d)$.

**Scout-scale falsification test:**
Drop into i243 by replacing the additive HalfKA-style accumulator with this pair accumulator at equal output width. Baseline: additive accumulator with same downstream head. Metric: matched-recall near-puzzle FP and make/unmake eval/sec. Works if FP improves by at least 5% with <10% speed loss, or speed improves at equal FP; fails if it only helps easy negatives.

**Failure mode catalogue:**
- Hidden rebrand: resembles Factorization Machines; novelty rests on bounded-delta stateful neural execution, not pairwise algebra.
- Numerical instability: $p\odot q$ can grow with material count; normalise by active-count or use RMS scaling.
- Speed risk: Python-side index handling can erase gains; needs fused CUDA or efficient `index_add` kernel.

**Status:** proposed

## 3. Proposal

### primitive_legal_edge_reduce

**Name:** Content-Generated Legal-Edge Reduce

**One-line claim:** A sparse reduce whose graph edges are generated from the current position inside the operator.

**Mathematical signature:**
$f:\mathbb{R}^{B\times64\times d}\times\{0,\dots,12\}^{B\times64}\rightarrow\mathbb{R}^{B\times64\times d}$.
Let $E_b=\mathcal{L}(P_b)$ be labelled pseudo-legal or legal edges $(i,j,\lambda)$ generated by a rules kernel:

$$
g_e=\sigma(a_\lambda^\top[x_{b,i},x_{b,j}]),\quad
y_{b,j}=\sum_{(i,j,\lambda)\in E_b}\frac{g_e\,W_\lambda x_{b,i}}{\sqrt{d_i d_j+\epsilon}}.
$$

Gradients flow through $x,W,a$; the edge generator is discrete.

**Why this does not decompose into existing PyTorch ops:**
Masked attention accepts an externally supplied mask; it does not generate a per-sample graph from symbolic content inside the op. PyTorch `MultiheadAttention` is still query-key softmax attention with optional masks, not rule-generated sparse message reduction. Dynamic graph attention exists in the literature, but it generally learns/sparsifies graph structure rather than fusing a domain-rule graph generator with the reduce kernel.

**Chess-specific motivation:**
The most structural chess graph is not the 8×8 grid; it is the legal-move/attack graph, which changes every position. Near-puzzle false positives often differ by one tactical edge: a pinned defender, a discovered attack, or a legal recapture.

**Generalisation beyond chess:**
Rule-governed dynamic graphs: traffic with lane rules, program-control graphs, robotics contact graphs, board games, and symbolic-physical simulators.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(B|E|d)$ vs full attention $O(Bn^2d)$.
- Backward: $O(B|E|d)$.
- Incremental update on a bounded-change input: $O(B|\Delta E|d)$; with cached ray attacks, bounded on 8×8.

**Scout-scale falsification test:**
Replace i242’s legal/decomposed attention component with this reduce primitive, leaving the rest of the harness unchanged. Baselines: i242 ablation and conv-only i193. Metric: matched-recall CRTK class-1 near-puzzle FP plus eval/sec. Works if it beats i193 FP by ≥5% and is faster than masked attention; fails if it needs LC0-scale data.

**Failure mode catalogue:**
- Hidden rebrand: “just PyG message passing on a generated graph.”
- Numerical instability: high-degree queen/king zones can dominate; degree normalisation and clipped gates are mandatory.
- Speed risk: legal-edge generation in Python will be too slow; needs bitboard/CUDA or C++ extension.

**Status:** proposed

## 4. Proposal

### primitive_orbit_action_norm

**Name:** Finite-Orbit Action Normalization

**One-line claim:** Normalise features over finite symmetry orbits instead of batch, channel, or feature dimensions.

**Mathematical signature:**
Given finite group $G$ acting on index set $I$ with representation $R_g$ on channels:
$f:\mathbb{R}^{B\times |I|\times d}\rightarrow\mathbb{R}^{B\times |I|\times d}$.
For orbit samples $z_{b,g,i}=R_gx_{b,g^{-1}i}$:

$$
\mu_{b,i}=\frac1{|G_i|}\sum_{g\in G_i}z_{b,g,i},\quad
\sigma^2_{b,i}=\frac1{|G_i|d}\sum_{g\in G_i}\|z_{b,g,i}-\mu_{b,i}\|_2^2,
$$

$$
y_{b,i}=\gamma_i\frac{x_{b,i}-\mu_{b,i}}{\sqrt{\sigma^2_{b,i}+\epsilon}}+\beta_i,
$$

with $\gamma,\beta$ tied over group orbits.

**Why this does not decompose into existing PyTorch ops:**
`LayerNorm` normalises over the last feature dimensions, while BatchNorm/InstanceNorm/GroupNorm use predefined batch/channel partitions. PyTorch’s LayerNorm documentation makes that axis choice explicit. This operator normalises over an arbitrary finite group action, with parameter tying and optional signed/channel representations.

**Chess-specific motivation:**
Chess has board symmetries, side-to-move/color swap, and piece-role involutions beyond plain translation or D4 image symmetry. Orbit-normalising early square/piece features may reduce sample complexity in the 173k-position scout regime.

**Generalisation beyond chess:**
Molecules, crystal symmetries, multi-agent role swaps, graph automorphisms, robotics frames, and any finite-symmetry feature field.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(B|G||I|d)$ vs LayerNorm $O(B|I|d)$.
- Backward: $O(B|G||I|d)$.
- Incremental update on a bounded-change input: $O(|G|d)$ for affected orbits.

**Scout-scale falsification test:**
Use i193 as harness. Compare three matched runs: no norm, LayerNorm/GroupNorm, and OrbitActionNorm after the same feature stage. Metric: near-puzzle FP at matched recall, plus seed-to-seed variance if a second short seed fits. Works if OrbitActionNorm improves FP without slowing inference by >15%; fails if it only regularises easy negatives.

**Failure mode catalogue:**
- Hidden rebrand: “group convolution already exists”; cite group-equivariant CNNs as prior, not novelty denial.
- Numerical instability: tiny orbit variance can explode; use $\epsilon\ge10^{-4}$ and affine clipping.
- Speed risk: explicit group expansion wastes memory; implement as indexed reductions, not tensor tiling.

**Status:** proposed

## 5. Proposal

### primitive_soft_see_reducer

**Name:** Alternating Soft-SEE Reducer

**One-line claim:** A differentiable consume-once minimax reducer for contested squares and capture sequences.

**Mathematical signature:**
$f:\mathbb{R}^{B\times64\times m}\times\{-1,+1\}^{B\times64\times m}\times\mathbb{R}^{B\times64}\rightarrow\mathbb{R}^{B\times64}$.
For each square, attacker costs $c_i$, side labels $s_i$, active mask $a_0$, side $\tau_t$:

$$
p_t=\operatorname{softmin}_\beta(c_i+M(1-a_{t-1,i})+M\mathbf{1}[s_i\ne\tau_t]),
$$

$$
r_t=\sum_i p_{t,i}c_i,\quad a_t=a_{t-1}-p_t,\quad g_t=r_t-g_{t-1}.
$$

Then back up with smooth SEE:

$$
q_T=g_T,\quad q_{t-1}=-\operatorname{smoothmax}_\beta(-g_{t-1},q_t).
$$

Output $q_0$; gradients are well-defined for finite $\beta$.

**Why this does not decompose into existing PyTorch ops:**
Differentiable sorting/ranking operators exist, including fast $O(n\log n)$ methods. This is not sorting alone: it alternates sides, consumes selected attackers once, and performs smooth minimax backup. No `torch.nn` primitive has this alternating game-reduction signature.

**Chess-specific motivation:**
Static exchange evaluation is one of the hand-engineered chess ideas neural evaluators often rediscover poorly at small data scale. Near-puzzle hard negatives frequently depend on whether a tactic wins material after a forced capture chain.

**Generalisation beyond chess:**
Mostly game/search domains: Go-like local fights, trading games, auctions, collision resolution, resource contests. It is the most chess-specific proposal here.

**Complexity (forward, backward, incremental-update):**
- Forward: $O(B\cdot64\cdot m^2)$ vs attention $O(Bn^2d)$; $m$ is bounded attacker count.
- Backward: $O(B\cdot64\cdot m^2)$.
- Incremental update on a bounded-change input: $O(|\Delta A|m^2)$ affected contested squares.

**Scout-scale falsification test:**
Drop into i193 over its learned square embeddings plus internally generated attack sets. Baseline: same i193 parameter budget with an MLP reducer over square embeddings. Metric: matched-recall near-puzzle FP only. Works if FP improves by ≥10% with no aggregate-only gain; fails if it overfits tactical motifs and hurts calibration.

**Failure mode catalogue:**
- Hidden rebrand: “soft differentiable sorting plus a loop.”
- Numerical instability: low-temperature softmin can become nearly discrete; anneal $\beta$ only after warmup.
- Speed risk: generating all contested sets for every square can dominate GPU time; cache attack sets.

**Status:** proposed

## What I cut

1. **Differential legal-move attention.** Cut because differential attention is already a two-softmax subtraction mechanism, and legal masks still leave it as attention with a different mask. Differential Transformer is interesting, but not enough here.

2. **Chess KAN / spline piece activation.** Cut because KANs already define learnable edge functions with spline parameterisation; a chess-specific spline activation would be an application or reparameterisation, not a new primitive.

3. **Pure D4 group convolution.** Cut because group-equivariant convolution is established prior work; chess needs color/role/legal-graph structure, not just rotations/reflections.

4. **Learned mixture of convolution, attention, and Mamba.** Cut because it is a composition of existing primitives, exactly the level the prompt rejects.

5. **King-distance planes, exchange planes, and deterministic attack maps.** Cut because these are input encodings/features, not neural primitives.

## Bibliography

- PyTorch documentation: [`torch.nn`](https://docs.pytorch.org/docs/stable/nn.html) and [`torch.nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html).
- Stockfish documentation: [NNUE PyTorch wiki](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html).
- Albert Gu and Tri Dao, [“Mamba: Linear-Time Sequence Modeling with Selective State Spaces”](https://arxiv.org/abs/2312.00752), 2023.
- Tri Dao and Albert Gu, [“Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality”](https://proceedings.mlr.press/v235/dao24a.html), ICML 2024.
- Maximilian Beck et al., [“xLSTM: Extended Long Short-Term Memory”](https://openreview.net/forum?id=Dh0Y88UAXR), 2024.
- Songlin Yang, Jan Kautz, and Ali Hatamizadeh, [“Gated Delta Networks: Improving Mamba2 with Delta Rule”](https://research.nvidia.com/publication/2025-04_gated-delta-networks-improving-mamba2-delta-rule), 2025.
- Kimi Team, [“Kimi Linear: An Expressive, Efficient Attention Architecture”](https://arxiv.org/abs/2510.26692), 2025.
- Ziming Liu et al., [“KAN: Kolmogorov-Arnold Networks”](https://openreview.net/forum?id=Ozo7qJ5vZi), 2024.
- Ali Behrouz, Peilin Zhong, and Vahab Mirrokni, [“Titans: Learning to Memorize at Test Time”](https://research.google/pubs/titans-learning-to-memorize-at-test-time/), 2025.
- Taco Cohen and Max Welling, [“Group Equivariant Convolutional Networks”](https://proceedings.mlr.press/v48/cohenc16.html), ICML 2016.
- Mathieu Blondel et al., [“Fast Differentiable Sorting and Ranking”](https://research.google/pubs/fast-differentiable-sorting-and-ranking/), ICML 2020.
- Steffen Rendle, [“Factorization Machines”](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf), ICDM 2010.
- Ben Mildenhall et al., [“NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis”](https://arxiv.org/abs/2003.08934), ECCV 2020.
