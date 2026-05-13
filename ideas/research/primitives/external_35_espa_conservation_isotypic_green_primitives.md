# New Neural-Network Primitives for Chess Evaluation

Prepared for the `chess-nn-playground` research project.

Scope: this file proposes primitive-level operators only. It deliberately avoids new architectures, input encodings, training tricks, hyperparameter variants, and Stockfish/PV/node-count metadata as compute-graph inputs.

## Literature calibration and ranking

Recent primitive-level work supports the bar used here: Mamba/S6 made input-conditioned state-space recurrences practical with linear sequence scaling and hardware-aware kernels; Mamba-2 reframed attention and SSMs through structured semiseparable matrices; xLSTM introduced exponential gates plus scalar/matrix memory variants; Gated DeltaNet combines gating with a delta-rule memory update; KAN replaces linear edge weights with learnable univariate functions; Differential Transformer subtracts two softmax attention maps; and GLinSAT is a 2024 differentiable linear-satisfiability layer. These are not just “new blocks”; their novelty is a changed state update, changed backward path, constrained differentiable solve, or changed primitive algebra. Sources are linked inline below.

Ranked order below is the recommendation order. Scores use 1 = weak, 5 = strong.

| rank | primitive | plausibility of novelty | RTX 3070 demonstrability | inference-speed advantage | generalisation beyond chess |
|---:|---|---:|---:|---:|---:|
| 1 | `primitive_espa` | 4 | 5 | 5 | 4 |
| 2 | `primitive_conserve_norm` | 3 | 5 | 4 | 5 |
| 3 | `primitive_isotypic_projector` | 3 | 4 | 4 | 4 |
| 4 | `primitive_green_solve` | 4 | 3 | 2 | 5 |
| 5 | `primitive_matroid_base_pool` | 3 | 3 | 3 | 4 |

1. ### primitive_espa

**Name:** Elementary Symmetric Piece Accumulator

**One-line claim:** A permutation-invariant multiplicative set accumulator that gives NNUE-like bounded updates for low-degree piece interactions.

**Mathematical signature:**
\(f: \mathbb{R}^{B\times n\times d}\times\{0,1\}^{B\times n}\rightarrow\mathbb{R}^{B\times K\times m}\).
Let \(u_i=W x_i\in\mathbb{R}^m\), mask \(a_i\in\{0,1\}\). Initialize \(E_0=\mathbf{1}\), \(E_k=\mathbf{0}\) for \(k>0\). For \(i=1..n\), update descending:
\[
E_k \leftarrow E_k+a_i\,u_i\odot E_{k-1},\quad k=K..1.
\]
Thus \(E_k[c]=\sum_{|S|=k}\prod_{i\in S}a_i u_i[c]\). Gradients use the reverse recurrence. Deletion of item \(j\) is exact by
\[
E^{(-j)}_0=1,\quad E^{(-j)}_k=E_k-u_j\odot E^{(-j)}_{k-1}.
\]

**Why this does not decompose into existing PyTorch ops:**
Deep Sets characterises permutation-invariant functions as \(\rho(\sum_i\phi(x_i))\), but ESPA emits exact elementary symmetric polynomial channels rather than a learned sum embedding [Deep Sets](https://arxiv.org/abs/1703.06114). A naive PyTorch graph enumerates all \(k\)-tuples or loops through degree states. The primitive’s distinct signature is a fused degree-recursive commutative-algebra scan with an exact inverse-update rule.

**Duplicate audit against existing primitive memory:**
Closest: signed piece-existence Hessian and sparse delta accumulators. This is not a Hessian or cross-derivative: it computes forward elementary symmetric polynomials, not derivatives of an evaluation function with respect to piece existence. It is not HalfKA/NNUE-style sparse linear accumulation: NNUE maintains first-layer linear preactivations incrementally, whereas ESPA maintains multiplicative degree states with a different deletion recurrence; NNUE’s accumulator/incremental property is only the speed inspiration [Stockfish NNUE docs](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html), [ChessProgramming NNUE](https://www.chessprogramming.org/Stockfish_NNUE).

**Chess-specific motivation:**
Pieces of the same type are mostly unordered, but evaluation depends on combinations: two defenders, three attackers, bishop pair plus open file, pawn-chain triples. ESPA gives controlled low-degree interactions without attention’s data hunger. It directly targets hard-negative near-puzzles where a single tactical feature is insufficient.

**Generalisation beyond chess:**
Useful for molecular sets, particle systems, sparse event logs, and scene graphs where unordered low-degree interactions matter.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(BnKm)\) vs explicit k-tuple interaction \(O(Bn^K m)\)
- Backward: \(O(BnKm)\)
- Incremental update on a bounded-change input: \(O(Km)\)

**Scout-scale falsification test:**
Drop ESPA with \(K=3,m=64\) after the piece-token embedding in the current i193-style conv parent; concatenate its pooled output only at the eval head. Baseline: same-parameter global sum-pool MLP. Metric: matched-recall CRTK class-1 near-puzzle FP rate. Works if FP rate drops by at least 5% with less than 10% eval/s loss. Fails if only aggregate PR AUC improves.

**Failure mode catalogue:**
- Hidden rebrand risk: a reviewer may call it “just polynomial features”; the primitive claim only holds if the fused recurrence and inverse update are first-class.
- Numerical risk: products can explode or vanish; use \(K\le3\), RMS-normalized \(u_i\), and clipped channels.
- Speed risk: for dense \(n\) and high \(K\), it becomes slower than a small MLP.

**Status:** proposed

2. ### primitive_conserve_norm

**Name:** Conservation-Nullspace Normalization

**One-line claim:** Normalize activations after projecting out board-specific conserved-charge directions instead of subtracting a scalar feature mean.

**Mathematical signature:**
\(f: \mathbb{R}^{B\times n\times d}\times\mathbb{R}^{B\times n\times r}\times\mathbb{R}_+^{B\times n}\rightarrow\mathbb{R}^{B\times n\times d}\).
For batch item \(b\), let \(X\in\mathbb{R}^{n\times d}\), charge matrix \(C\in\mathbb{R}^{n\times r}\), and \(D=\mathrm{diag}(w)\). Define
\[
A=C^\top DC+\epsilon I_r,\quad M=A^{-1}C^\top DX,\quad R=X-CM.
\]
Then
\[
\sigma_j^2=\frac{R_{:,j}^{\top}DR_{:,j}}{\max(1,\sum_i w_i-r)},\quad
Y_{ij}=\gamma_j R_{ij}/\sqrt{\sigma_j^2+\epsilon}+\beta_j.
\]

**Why this does not decompose into existing PyTorch ops:**
LayerNorm normalizes over feature dimensions; ConserveNorm normalizes over the nullspace of a dynamic per-position constraint matrix. GLinSAT shows that differentiable linear-constraint satisfaction is a legitimate layer family, but GLinSAT projects outputs onto bounded linear constraints, while this operator is an internal activation normalizer with a constraint-dependent residual and variance graph [GLinSAT](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dd73f39426a03131c38c8d943153d44b-Abstract-Conference.html). It is closer to a future `torch.nn.ConstraintNorm` than to LayerNorm plus masking.

**Duplicate audit against existing primitive memory:**
Closest: color-involution gates and sparse delta accumulators. It is not a color/piece gate: no learned involution, no message passing, and no routing. It is not an accumulator primitive: bounded update is an implementation benefit via Sherman-Morrison, while the core operator is a weighted least-squares projection followed by residual normalization.

**Chess-specific motivation:**
Material, side, color, and piece-type counts explain many easy positions. Hard negatives require tactical residuals after that bookkeeping is removed. This primitive forces later channels to represent what is not already linearly explained by conserved chess charges.

**Generalisation beyond chess:**
Useful in physical simulation, power systems, chemistry, and operations research, where activations should respect or factor out conservation laws and balance equations.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B(nrd+r^2d+r^3+nd))\) vs LayerNorm \(O(Bnd)\)
- Backward: \(O(B(nrd+r^2d+r^3+nd))\)
- Incremental update on a bounded-change input: update \(A^{-1}\) and \(M\) in \(O(r^2+rd)\); materializing all \(Y\) remains \(O(nd)\)

**Scout-scale falsification test:**
Insert ConserveNorm after the first feature-mixing layer of i193 and compare to a LayerNorm/BatchNorm-free baseline with equal parameters. Metric: matched-recall CRTK class-1 FP rate plus eval/s. Works if FP rate falls by at least 4% and speed loss is under 5%. Fails if training becomes seed-noisy or only easy negatives improve.

**Failure mode catalogue:**
- Hidden rebrand risk: if \(C=\mathbf{1}\), it collapses to weighted mean subtraction.
- Numerical risk: \(C^\top DC\) can be ill-conditioned; require \(\epsilon\), rank clipping, or QR.
- Speed risk: global projection changes every token after one move, so full output materialization is not \(O(1)\).

**Status:** proposed

3. ### primitive_isotypic_projector

**Name:** Chess-Group Isotypic Projector

**One-line claim:** Split activations into finite-group representation components instead of merely augmenting or weight-tying board symmetries.

**Mathematical signature:**
\(f: \mathbb{R}^{B\times n\times d}\times G\rightarrow\mathbb{R}^{B\times n\times d|\widehat G|}\).
Let finite group \(G\) act by token permutations \(P_g\) and channel actions \(R_g\). For each irrep \(\rho\) with dimension \(d_\rho\) and character \(\chi_\rho\):
\[
\mathcal{P}_{\rho}(X)=\frac{d_\rho}{|G|}\sum_{g\in G}\chi_\rho(g^{-1})\,P_gXR_g^\top.
\]
Output
\[
Y=\mathrm{concat}_{\rho\in\widehat G}\alpha_\rho \mathcal{P}_{\rho}(X),
\]
with learned scalar or vector gains \(\alpha_\rho\).

**Why this does not decompose into existing PyTorch ops:**
Group-equivariant CNNs already exploit symmetry through group convolution, but this operator performs explicit isotypic projection by character sums rather than convolution over translations, rotations, or reflections [Group Equivariant CNNs](https://proceedings.mlr.press/v48/cohenc16.html). A naive implementation is many `gather+sum` calls; the primitive’s backward must also project gradients into the same irreducible subspaces. Its signature returns representation components, not just equivariant feature maps.

**Duplicate audit against existing primitive memory:**
Closest: CRELU/color-involution graph messages and piece-relabelling/involution gates. This is not a color-involution update: color swap is only one generator inside \(G\), and the output is a full irrep decomposition. It is not a relabelling gate: there is no data-conditioned routing, no learned permutation, and no adjacency update; the projection is fixed by the group table.

**Chess-specific motivation:**
Chess has file mirror, color/rank flip with side-to-move swap, and same-type piece exchangeability. Ordinary Conv2d sees board geometry but not the full chess group. This operator separates invariant, anti-invariant, and mixed components so the network can learn color-swap and mirror constraints cleanly.

**Generalisation beyond chess:**
Useful for finite-group domains: games, chemistry with atom-type permutations, robotic object symmetries, and finite-state program analysis.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(B|G|nd)\) vs Conv2d \(O(Bk^2nd)\), or augmentation requiring multiple forwards
- Backward: \(O(B|G|nd)\)
- Incremental update on a bounded-change input: \(O(|G|\Delta d)\) for the changed token orbit; \(\Delta\) is edited token count

**Scout-scale falsification test:**
Use \(G=C_2^{file}\times C_2^{color/rank}\) only, avoiding huge same-piece permutation groups. Insert one projector after the first embedding in i193, then keep the parent conv stack unchanged. Baseline: ordinary symmetry augmentation at train time. Works if CRTK class-1 matched-recall FP improves by at least 5% and eval/s is not worse than running two augmented forwards.

**Failure mode catalogue:**
- Hidden rebrand risk: a reviewer may say group equivariance is old; the novelty claim must stay limited to chess-group isotypic projection as a primitive.
- Correctness risk: wrong group generators can enforce false pawn-direction symmetry.
- Speed risk: large \(G\) explodes memory; start with two commuting involutions.

**Status:** proposed

4. ### primitive_green_solve

**Name:** Input-Conditioned Dirichlet Green Solve

**One-line claim:** Replace finite local diffusion steps with one differentiable global potential solve over a board-sized conductance graph.

**Mathematical signature:**
\(f: \mathbb{R}^{B\times N\times d}\times\mathbb{R}_{+}^{B\times E}\rightarrow\mathbb{R}^{B\times N\times m}\).
For each batch item, source \(S=XW_s\). Define weighted board Laplacian:
\[
(L_gY)_u=\sum_{v\sim u}g_{uv}(Y_u-Y_v).
\]
The primitive returns
\[
Y=(L_g+\lambda I)^{-1}S.
\]
For adjoint \(\bar Y\), solve \(\bar S=(L_g+\lambda I)^{-T}\bar Y\). Edge gradient:
\[
\frac{\partial \mathcal{L}}{\partial g_{uv}}
=-\sum_c(\bar S_{u,c}-\bar S_{v,c})(Y_{u,c}-Y_{v,c}).
\]

**Why this does not decompose into existing PyTorch ops:**
A Conv2d or fixed-depth message-passing layer approximates diffusion by repeated local steps; GreenSolve is an implicit global resolvent with coefficient-dependent adjoint solves. OptNet established differentiable optimization/implicit solves as neural layers with gradients from sensitivity and implicit differentiation [OptNet](https://arxiv.org/abs/1703.00443). This primitive is a board-graph Green kernel with a fixed sparse Laplacian structure, not generic dense `torch.linalg.solve` glued into a model.

**Duplicate audit against existing primitive memory:**
Closest: SLG diffusion and ray-occlusion/ray-blocked reducers. This is not legal-graph diffusion: the graph is fixed board adjacency, and the result is an exact linear-system solution, not iterative legal-message passing. It is not ray/blocker dispatch: influence can bend through conductance paths and is governed by Laplacian energy, not ray visibility.

**Chess-specific motivation:**
King safety, pawn shields, open files, and blocked centers behave like potential-field phenomena: one blocker changes global accessibility. A learned conductance field can express “open file leaks pressure” without enumerating legal move edges or attention masks.

**Generalisation beyond chess:**
Useful for images, gridworlds, circuit networks, cloth/fluids, and dynamic graphs where global equilibrium matters more than local convolution.

**Complexity (forward, backward, incremental-update):**
- Forward: dense \(O(BN^3+BN^2m)\), or CG/multigrid \(O(BTEm)\); closest 3×3 conv block \(O(BEm)\), attention \(O(BN^2m)\)
- Backward: one adjoint solve plus edge gradients, \(O(BTEm)\)
- Incremental update on a bounded-change input: low-rank conductance update via Woodbury \(O(r^2N+rNm)\); not \(O(1)\)

**Scout-scale falsification test:**
Replace one middle 3×3 conv block in i193 with GreenSolve using \(m=16\), \(N=64\), and conductance from a clipped 1×1 projection. Baseline: same-parameter 3×3 conv. Works if CRTK class-1 matched-recall FP drops by at least 7% and latency is under 1.25× baseline. Fails if it only helps quiet/easy positions.

**Failure mode catalogue:**
- Hidden rebrand risk: if implemented as 8–16 diffusion iterations, it becomes ordinary message passing.
- Numerical risk: ill-conditioned \(L_g\) can create exploding gradients; clamp \(g\) and \(\lambda\).
- Speed risk: it may be too slow for engine inference even if accurate.

**Status:** proposed

5. ### primitive_matroid_base_pool

**Name:** Entropic Matroid-Base Pool

**One-line claim:** A constrained soft-selection primitive that pools mutually compatible evidence instead of independently weighting all tokens.

**Mathematical signature:**
\(f: \mathbb{R}^{B\times n\times d}\times\mathbb{R}^{B\times n}\times \mathcal{M}\rightarrow\mathbb{R}^{B\times d}\).
For matroid \(\mathcal{M}\) with rank function \(r\), define the base polytope:
\[
B(r)=\{z\in[0,1]^n: z(S)\le r(S)\ \forall S\subseteq[n],\ z([n])=r([n])\}.
\]
For scores \(a\), compute
\[
z^\star=\arg\max_{z\in B(r)} a^\top z-\tau\sum_i z_i\log(z_i+\epsilon),
\quad y=\sum_i z_i^\star x_i.
\]
Gradient is defined by the KKT system of the entropy-regularized convex program.

**Why this does not decompose into existing PyTorch ops:**
Softmax selects over a simplex; top-k is non-smooth and mostly cardinality-only. This primitive selects from a matroid base polytope, so feasibility is combinatorial and the backward graph is an implicit constrained optimizer. Prior differentiable submodular and greedy work supports the existence of differentiable combinatorial-selection layers, but this proposed primitive is specifically entropy-smoothed base-polytope pooling [Differentiable Submodular Maximization](https://www.ijcai.org/proceedings/2018/0379.pdf), [Differentiable Greedy Submodular Maximization](https://proceedings.mlr.press/v130/sakaue21a/sakaue21a.pdf).

**Duplicate audit against existing primitive memory:**
Closest: Pareto Antichain Frontier and dynamic adjacency rank-order gates. It is not a frontier operator: it returns one entropy-smoothed base point, not a set of nondominated witnesses. It is not a rank-order gate over graph edges: no legal graph, no edge routing, and no message passing; the constraint is an independence oracle or base-polytope projection.

**Chess-specific motivation:**
A piece cannot be fully counted as defender, attacker, pin-holder, and escape-cover simultaneously. Matroid pooling expresses “choose a compatible set of explanations” and reduces double-counting, a common source of near-puzzle false positives.

**Generalisation beyond chess:**
Useful for summarization, sensor selection, portfolio selection, routing with resource constraints, and multi-object scene evidence pooling.

**Complexity (forward, backward, incremental-update):**
- Forward: \(O(T\,\mathrm{Oracle}_{\mathcal{M}}(n)+nd)\); for partition matroids \(O(n\log n+nd)\), vs softmax \(O(nd)\)
- Backward: KKT/implicit pass of comparable order
- Incremental update on a bounded-change input: \(O(\log n+d)\) for partition matroids with maintained heaps; general matroids depend on oracle

**Scout-scale falsification test:**
Create attacker/defender/evidence tokens from existing latent channels, not from Stockfish or PV metadata. Replace a softmax pooling head with MatroidBasePool using a partition matroid: at most one role per piece and at most \(k\) tactical evidence items. Baseline: ordinary softmax attention pool. Works if CRTK class-1 near-puzzle FP drops by at least 5% at fixed recall. Fails if it reduces recall by suppressing true multi-role tactics.

**Failure mode catalogue:**
- Hidden rebrand risk: with a uniform matroid, it degenerates toward soft top-k.
- Numerical risk: KKT gradients can be unstable as \(\tau\to0\).
- Speed risk: general matroid oracles may be too slow; start with partition or laminar matroids only.

**Status:** proposed

## What I cut

1. **Legal-move attention/router** — duplicate of move-graph routers, legal-move accumulators, and sparse legal graph transitions.
2. **Ray-blocker SSM or occlusion scan** — duplicate of ray-scan, ray-parallel SSM, ray-occlusion dispatch, and blocker-reset fastweight families.
3. **Signed tactical Hessian / counterfactual capture derivative** — duplicate of signed piece-existence Hessian and tempo-defender cross-derivative families.
4. **Persistent-homology board topology pool** — real topology layers already exist, so the novelty claim collapses; it is also too slow and hard to falsify on 173k positions [A Topology Layer for Machine Learning](https://arxiv.org/abs/1905.12200).
5. **Noncrossing pawn-pair inside operator** — interesting, but too close to structured attention and differentiable dynamic programming layers; better kept as “underexplored chess use” rather than claimed as a new primitive [Differentiable Dynamic Programming for Structured Prediction and Attention](https://arxiv.org/abs/1802.03676).
6. **Sinkhorn exchange-pressure pool** — mostly optimal transport plus chess vocabulary, and too close to exchange/routing primitives.
7. **Exterior-algebra wedge interaction** — elegant, but a reviewer would fairly call it a tensor-product or pair-interaction rebrand.
8. **Soft minimax witness pool** — duplicate of witness-counterwitness, regret saddlepoint, and reply-channel-capacity primitive families.
9. **Zobrist neural sketch accumulator** — too close to signed edit bilinear memory and sparse delta accumulators.
10. **Terminal/tactic-state detector** — explicitly blocklisted as terminal-state detection.

## Prior-work references used for calibration

- Albert Gu and Tri Dao, **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**, 2023/2024. <https://arxiv.org/abs/2312.00752>
- Tri Dao and Albert Gu, **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**, 2024. <https://arxiv.org/abs/2405.21060>
- Maximilian Beck et al., **xLSTM: Extended Long Short-Term Memory**, 2024. <https://arxiv.org/abs/2405.04517>
- Songlin Yang, Jan Kautz, and Ali Hatamizadeh, **Gated Delta Networks: Improving Mamba2 with Delta Rule**, 2024/2025. <https://arxiv.org/abs/2412.06464>
- Ziming Liu et al., **KAN: Kolmogorov-Arnold Networks**, 2024. <https://arxiv.org/abs/2404.19756>
- Tianzhu Ye et al., **Differential Transformer**, 2024/2025. <https://arxiv.org/abs/2410.05258>
- Hongtai Zeng et al., **GLinSAT: The General Linear Satisfiability Neural Network Layer By Accelerated Gradient Descent**, NeurIPS 2024. <https://proceedings.neurips.cc/paper_files/paper/2024/hash/dd73f39426a03131c38c8d943153d44b-Abstract-Conference.html>
- Manzil Zaheer et al., **Deep Sets**, 2017. <https://arxiv.org/abs/1703.06114>
- Taco Cohen and Max Welling, **Group Equivariant Convolutional Networks**, ICML 2016. <https://proceedings.mlr.press/v48/cohenc16.html>
- Brandon Amos and J. Zico Kolter, **OptNet: Differentiable Optimization as a Layer in Neural Networks**, 2017. <https://arxiv.org/abs/1703.00443>
- Sebastian Tschiatschek, Aytunc Sahin, and Andreas Krause, **Differentiable Submodular Maximization**, IJCAI 2018. <https://www.ijcai.org/proceedings/2018/0379.pdf>
- Shinsaku Sakaue, **Differentiable Greedy Submodular Maximization**, AISTATS 2021. <https://proceedings.mlr.press/v130/sakaue21a/sakaue21a.pdf>
- Rickard Brüel-Gabrielsson et al., **A Topology Layer for Machine Learning**, 2019. <https://arxiv.org/abs/1905.12200>
- Arthur Mensch and Mathieu Blondel, **Differentiable Dynamic Programming for Structured Prediction and Attention**, 2018. <https://arxiv.org/abs/1802.03676>
- Official Stockfish NNUE documentation. <https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html>
- ChessProgramming Wiki, **Stockfish NNUE**. <https://www.chessprogramming.org/Stockfish_NNUE>
