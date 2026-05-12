# Codex Handoff Packet: Piece-Target Entropic Transport Bottleneck

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0657_tuesday_los_angeles_piece_transport.md`
- Generated at: 2026-04-21 06:57:24 America/Los_Angeles
- Weekday: Tuesday
- Timezone: `los_angeles`
- Idea slug: `piece_transport`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Piece-Target Entropic Transport Bottleneck, abbreviated **PT-ETB**.
- One-sentence thesis: A puzzle-like chess position should often expose a low-entropy, asymmetric, type-aware transport plan from the side-to-move's existing pieces to the opponent's valuable existing targets, even without engine scores, attack/sheaf complexes, or one-ply move-delta enumeration.
- Idea fingerprint: `current-board relative piece occupancy + opponent/friendly target occupancy + learned type-aware entropic Sinkhorn optimal transport over 64 existing squares + compact transport-map/statistic bottleneck + binary puzzle-likeness target`.
- Why this is not a common CNN/ResNet/Transformer variant: the central computation is a constrained differentiable optimal-transport problem with board-derived marginals and a learned chess-geometric cost; convolution is only a shallow adapter/fuser, not the research mechanism.
- Current-data minimal experiment: train PT-ETB on `simple_18` using `data/splits/crtk_sample_3class/split_train.parquet`, validate on `split_val.parquet`, test on `split_test.parquet`, and compare against the existing simple CNN, residual CNN, and any existing `simple_18` leaderboard entries under the unchanged binary benchmark.
- Smallest central falsification ablation: keep the same material, side-to-move, source-square marginals, target-square marginals, Sinkhorn iterations, parameter count, and cost histograms, but row-wise permute each learned cost matrix's target-square semantics by a fixed random permutation before Sinkhorn; if this matches PT-ETB, the proposed transport geometry is not carrying useful signal.
- Expected information gain if it fails: a clean failure says that learned piece-target OT over current occupancy is no better than material/proximity shortcuts on this split, allowing the next research cycle to stop revisiting Sinkhorn transport variants and move toward uncertainty, causal invariance, or generative compression instead.

## 3. Problem Restatement And Data Contract

The task is chess puzzle-likeness classification from a single board position. The model receives an encoded tensor `x` with shape `(batch, C, 8, 8)` and returns logits with shape `(batch, 2)` for binary outputs:

- output `0`: non-puzzle
- output `1`: puzzle-like

The available fine labels are diagnostic/source classes, not extra input features:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

The binary target should be `0` for fine label `0` and `1` for fine labels `1` and `2`, unless the project already defines this mapping in the shared trainer. Reports must continue to include the rectangular diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Allowed encodings are `simple_18`, `lc0_static_112`, and `lc0_bt4_112`. The first experiment should use `simple_18` because the current-board piece-plane semantics are explicit enough for deterministic transport marginals. The model interface must remain a `torch.nn.Module` accepting `(batch, C, 8, 8)` and returning `(batch, num_classes)`.

Benchmark split:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

Do not point the current trainer directly at the roughly 45M-row full Parquet file until streaming support exists.

Leakage checklist:

- Safe inputs: deterministic board coordinates, current piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and deterministic current-board geometry derived from those fields.
- Safe rule-derived features for this packet: relative piece type, occupied square, side-to-move canonical orientation, square deltas, file/rank/diagonal/knight-vector indicators, Chebyshev/Manhattan distances, and soft transport plans between already occupied current-board squares.
- Leakage-prone unless separately justified and ablated: full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences. PT-ETB intentionally avoids these.
- Always forbidden as neural-network inputs: engine evaluation, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, and dataset provenance.
- For `lc0_static_112` and `lc0_bt4_112`: current-board piece channels may be used for deterministic transport geometry only if an explicit channel map is configured and tested. History channels must not be interpreted by the transport adapter; they may only be consumed by the learned neural adapter. Unknown channel semantics must fail closed.

Here “source mass” and “target mass” mean optimal-transport origin and destination distributions. They do **not** mean dataset source, label source, or provenance.

## 4. Research Map

External anchors used:

1. Marco Cuturi, “Sinkhorn Distances: Lightspeed Computation of Optimal Transport,” NeurIPS 2013, https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport. Borrowed: entropy-regularized transport solved by Sinkhorn matrix scaling. Not copied: no image retrieval objective, no MNIST setup, no external labels.
2. Gabriel Peyré and Marco Cuturi, “Computational Optimal Transport,” Foundations and Trends in Machine Learning 2019 / arXiv, https://arxiv.org/abs/1803.00567. Borrowed: discrete OT notation, numerical stability framing, and the idea that OT compares distributions using a ground cost. Not copied: no generic OT benchmark, no large solver library dependency requirement.
3. Aude Genevay et al., “Learning Generative Models with Sinkhorn Divergences,” AISTATS 2018 / arXiv, https://arxiv.org/abs/1706.00292. Borrowed: differentiating through Sinkhorn iterations is practical in neural training. Not copied: no generative model, no Sinkhorn divergence between data and generated samples.
4. Titouan Vayer et al., “Optimal Transport for structured data with application on graphs,” ICML 2019, https://proceedings.mlr.press/v97/titouan19a.html. Borrowed: inspiration for mixing feature costs and structural costs in transport. Not copied: no graph classifier, no Gromov-Wasserstein graph matching, no attack graph.
5. Naftali Tishby, Fernando Pereira, and William Bialek, “The Information Bottleneck Method,” https://arxiv.org/abs/physics/0004057, and Alexander Alemi et al., “Deep Variational Information Bottleneck,” https://arxiv.org/abs/1612.00410. Borrowed: the rate-distortion view that a compressed latent should preserve label-relevant information. Not copied: PT-ETB’s first run can use a deterministic low-dimensional bottleneck; a variational KL is optional.
6. David Silver et al., “Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm,” https://arxiv.org/abs/1712.01815. Borrowed only as a reminder that common chess NNs use convolutional/residual board encoders. Not copied: no self-play, no policy/value target, no MCTS, no engine-derived supervision.

Candidate search trace before selecting PT-ETB:

1. **Causal material-phase invariant bottleneck:** build environments by material phase and require invariant predictors. Rejected for this cycle because the mechanism would mostly be a training regularizer on an existing CNN, and environment definitions could be arbitrary without enough source-shift metadata.
2. **Ordinal near-puzzle calibrated head:** model fine labels `0 < 1 < 2` with a cumulative-link or selective classifier. Rejected as valuable but mostly a loss/reporting idea, not a distinctive board-position operator.
3. **Masked occupancy autoencoder anomaly score:** pretrain by reconstructing hidden pieces and use reconstruction residuals as puzzle-likeness features. Rejected because novelty would depend on pretraining protocol and might learn dataset style rather than tactical structure.
4. **D4/partial group-equivariant chess CNN:** enforce only color/rank canonical symmetries that respect pawn direction. Rejected because it remains close to ordinary equivariant CNN design and does not directly address puzzle-likeness.
5. **Walsh/DCT bitboard spectral network:** project piece planes into low-order board harmonics. Rejected because it is elegant but likely too coarse for localized tactical motifs and too easy to duplicate with a small CNN.
6. **Persistent-homology occupancy fields:** treat pieces as weighted point clouds and classify persistence summaries. Rejected because chess motifs depend on type, side, and directed value relations; pure topology would likely collapse to material/space heuristics.
7. **Energy-based latent motif dictionary:** infer a sparse set of motif atoms from the board. Rejected because it would need a large design space and hard-to-debug negative sampling, making it poor for the next Codex cycle.
8. **Adversarial material erasure:** train a representation to predict puzzle label while failing to predict material. Rejected because material is partly legitimate signal; erasing it can remove real chess information and produce an ambiguous failure.
9. **Rule-only king-zone scalar feature tower:** derive king shelter, open lines, and piece proximity scalars. Rejected because handcrafted feature engineering would be too narrow and might quietly become an attack-feature duplicate.
10. **Multi-instance latent tactical-square selector:** learn a few salient squares with Gumbel/attention. Rejected because ordinary attention over squares is close to a small Transformer unless the selected operator is more constrained.
11. **Two-board contrastive invariance across encoding families:** require `simple_18` and `lc0_bt4_112` representations to agree. Rejected because it depends on multi-encoding loader plumbing and would not be a minimal current-data architecture.
12. **Piece-target entropic transport:** selected because it is mathematically distinct from sheaf/attack and move-delta families, uses only current-board occupancy and geometry, has a sharp randomized semantics ablation, and is small enough for Codex to implement quickly.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Existing simple CNN on `simple_18` | `src/chess_nn_playground/models/trunk/cnn.py` | Already implemented; more of it would only test local convolutional capacity. |
| Existing residual CNN variants | `src/chess_nn_playground/models/trunk/residual_cnn.py` | Already implemented; deeper residual stacks are not a new research hypothesis. |
| LC0-style CNN or residual CNN on `lc0_static_112`/`lc0_bt4_112` | LC0 BT4-style CNN/residual variants | Already in the baseline family; copying LC0-style encoders is explicitly not enough. |
| Ordinary ViT over 64 square tokens | Vanilla Transformer over board squares | Too generic and explicitly disallowed as the core idea. |
| Plain GNN on square adjacency | Standard message-passing GNN | Too close to generic graph learning, and if attack edges are added it drifts toward the imported attack/sheaf family. |
| Hyperparameter tuning of width, depth, optimizer, or schedule | All existing baselines | It may improve metrics but is not an original falsifiable mechanism. |
| Ensembling existing CNN/residual/LC0 models | Leaderboard ensemble | Explicitly disallowed and would obscure mechanism-level falsification. |
| Material-only MLP/logistic classifier | Handcrafted scalar baseline | Useful as a nuisance ablation, but too weak and shortcut-prone as the research idea. |
| Tactical sheaf/Hodge/Laplacian/curvature/tension variant | Imported tactical sheaf packets | Already researched; adding edge labels, pooling, or renamed sheaf language would be a duplicate. |
| One-ply pseudo-legal move-delta set/landscape model | Imported counterfactual move-delta packets | Already researched; PT-ETB must not generate one-ply move outcomes or pool move deltas. |
| File-mirror symmetry/tension mechanism | Imported File-Mirror Tension Sheaf | Already researched and too close to symmetry regularization as the main contribution. |
| Full legal-move count or checkmate/stalemate oracle features | Rule-engine feature extractor | Leakage-prone and not necessary for the PT-ETB hypothesis. |
| Engine-evaluation or PV-distillation features | Engine-supervised evaluator | Forbidden as neural-network input and would destroy label-safety. |

## 6. Mathematical Thesis

### Input and target

Let `S = {1, ..., 64}` be the set of board squares in a side-to-move-relative coordinate system. Let `P = {pawn, knight, bishop, rook, queen, king}`. For a board position `x`, let

```text
o_{c,p,s}(x) ∈ {0,1}
```

indicate whether relative color `c ∈ {friendly, enemy}` has a piece of type `p` on square `s`. “Friendly” means the side to move. The raw tensor lies in

```text
X_C ⊂ R^{C×8×8}
```

with the usual chess consistency constraints imposed by the dataset/exporter, not by the model. The binary target is

```text
Y(x) = 0 if fine_label(x)=0,
Y(x) = 1 if fine_label(x)∈{1,2}.
```

Fine label is used only as a supervised target/reporting field, never as an input.

### Data distribution assumptions

The training, validation, and test splits are sampled from the current `crtk_sample_3class` process. The idea assumes that puzzle-like labels are not fully explained by material count, side-to-move, or trivial occupancy marginals. It also assumes that some puzzle-likeness is visible from the current board alone as concentrated relations between existing pieces and existing valuable targets. This is a hypothesis about the dataset and may be false.

### Symmetry assumptions

Chess is not invariant under all rotations/reflections: pawns are directed, castling rights are asymmetric, and side-to-move matters. PT-ETB only uses a side-to-move-relative orientation:

- friendly pieces are always treated as the active side;
- rank direction is canonicalized so friendly pawns point “forward” in the same tensor direction;
- no full D4 invariance is imposed;
- file mirror symmetry is not enforced, because castling/en-passant/context channels can break it and because file-mirror mechanisms are already in the imported research memory.

### Core hypothesis

Define friendly origin mass `μ_x` and enemy target mass `ν_x` on `S` by

```text
μ_x(s) = [ε/64 + Σ_p softplus(a_p) o_{friendly,p,s}(x)] / Z_μ(x)
ν_x(t) = [ε/64 + Σ_q softplus(b_q) o_{enemy,q,t}(x)] / Z_ν(x),
```

where `a_p` and `b_q` are learned piece-type weights and `ε > 0` prevents zero marginals. Define the reverse-direction pair `μ'_x,ν'_x` by swapping friendly and enemy. For each transport head `h`, a learned nonnegative cost is

```text
C^h_θ(s,t;x) = softplus(φ^h_θ(e_src(s,x), e_tgt(t,x), g(s,t))) + c_min,
```

where `e_src` and `e_tgt` are learned embeddings of the piece type occupying the origin/target squares, and `g(s,t)` contains deterministic chessboard geometry such as relative file/rank delta, absolute deltas, same-file, same-rank, same-diagonal, knight-vector indicator, Manhattan distance, and Chebyshev distance. It does not contain legal moves, attack edges, engine values, or future boards.

The entropy-regularized transport plan is

```text
π^h_θ(x) = argmin_{π ∈ Π(μ_x,ν_x)}  <π, C^h_θ(·,·;x)> + τ Σ_{s,t} π_{s,t}(log π_{s,t} - 1),
```

with `τ > 0`. The model extracts transport statistics and maps from `π^h_θ(x)` and `C^h_θ`, then predicts `Y` from a low-dimensional bottleneck.

The chess hypothesis is:

```text
I(Y ; transport_geometry(X) | material(X), side_to_move(X), occupancy_marginals(X)) > 0.
```

In words, the label contains information about structured piece-target proximity and concentration beyond obvious material and square-count shortcuts.

### Variational principle

A deterministic first run can minimize balanced cross-entropy. The fuller PT-ETB objective is

```text
min_θ  E[ CE_balanced(y, f_θ(z)) ]
       + β E[ KL(q_θ(z | ρ_θ(x), h_θ(x)) || N(0,I)) ]
       + λ E[ ||C_θ||_2^2 ],
```

where `ρ_θ(x)` are transport summaries, `h_θ(x)` is the shallow neural board adapter, and `z` is a compact bottleneck latent. Set `β=0` for the minimal deterministic experiment unless Codex wants to run an optional VIB ablation. The architectural bottleneck remains because `ρ_θ` is small and the final latent dimension should be about 32.

### Proposition 1: well-posed differentiable transport layer

For `τ > 0`, `ε > 0`, marginals `μ,ν` with every entry at least `ε/64Z`, and bounded finite cost matrix `C`, the entropy-regularized problem above has a unique strictly positive minimizer. It can be written as

```text
π = diag(u) exp(-C/τ) diag(v)
```

for positive scaling vectors `u,v` chosen to match the marginals. On any compact set of positive marginals and bounded costs, the plan and regularized value are differentiable functions of `C`, `μ`, and `ν`.

Proof sketch: entropy regularization makes the objective strictly convex on the transport polytope, giving uniqueness. The KKT conditions yield the diagonal-scaling form with kernel `K=exp(-C/τ)`. Sinkhorn scaling alternately normalizes rows and columns to satisfy the marginal constraints. Strict positivity and bounded costs avoid degenerate zero entries, so standard implicit-function or unrolled-iteration arguments give differentiability.

What is actually proven: the PT-ETB transport operator is a stable, differentiable, label-independent neural layer for current-board occupancy distributions.

What is not proven: that puzzle-likeness is encoded in this transport layer, or that learned costs will correspond to human tactical concepts.

### Proposition 2: central falsification logic

Let `N(X)` be the nuisance vector consisting of material counts by side/type, side-to-move, source-square marginal, target-square marginal, and capture-value histograms derivable without move generation. Let `T(X)` be PT-ETB’s true transport summaries, and let `T_rand(X)` be the row-histogram-preserving cost-shuffled summaries. If

```text
P(Y | N(X), T(X)) = P(Y | N(X), T_rand(X))
```

almost surely, then the Bayes-optimal risk of the main transport model and the semantics-destroying randomized ablation is the same, up to optimization/statistical error.

Proof sketch: the Bayes classifier depends only on the conditional probability of `Y` given the available features. If replacing `T` by `T_rand` leaves that conditional distribution unchanged once nuisances are known, no classifier can exploit the proposed transport semantics. Therefore an empirical gap between PT-ETB and the randomized ablation is evidence, not proof, that the transport geometry carries label-relevant signal.

### Counterexamples where PT-ETB should fail

- Quiet endgame studies whose puzzle-likeness depends on long zugzwang or opposition rather than immediate piece-target geometry.
- Positions where the best tactic is a defensive resource or stalemate motif not represented by transporting active pieces to valuable targets.
- Dataset artifacts where puzzle labels are mostly explained by material imbalance, source collection style, or non-geometric metadata not present in the tensor.
- Positions whose tactical essence requires legal-move consequences, discovered checks after a move, or multi-ply forcing lines invisible from current occupancy alone.
- Ambiguous near-puzzles where class `1` intentionally sits between ordinary and true puzzle positions.

### Self-critique

The strongest objection is that optimal transport may learn a polished material/proximity shortcut: queens and kings get high weights, nearby squares look important, and the model improves only because it reconstructs a hand-tuned king-danger heuristic. That is not enough. The row-histogram-preserving cost shuffle, material-only ablation, CNN-only matched-capacity ablation, and type/geometry removals are mandatory because they can expose this failure. The experiment is still worth running because PT-ETB is small, label-safe, mathematically distinct from the imported sheaf and move-delta families, and falsifiable with one central randomized operator ablation.

## 7. Architecture Specification

### Module names

Implement the model in `src/chess_nn_playground/models/trunk/piece_target_transport.py` with these modules:

- `PiecePlaneAdapter`
- `RelativeBoardCanonicalizer`
- `TypeAwareTransportCost`
- `LogSinkhornTransport`
- `TransportSummaryProjector`
- `PieceTargetEntropicTransportBottleneck`
- builder function: `build_piece_target_transport_bottleneck`

### Forward-pass steps and shapes

Input:

```text
x: float tensor, shape (B, C, 8, 8)
```

1. **Learned board adapter**

```text
adapter = shallow Conv/GELU/Norm stack
A = adapter(x)                         # (B, W, 8, 8), W default 64
```

Use 3 or 4 small convolution blocks only. Do not scale the CNN until the transport mechanism is falsified.

2. **Piece-plane extraction**

For `simple_18`, use the repository’s existing channel semantics if available. If no constants exist, use and test this explicit assumption:

```text
channels 0..5   = white P,N,B,R,Q,K
channels 6..11  = black P,N,B,R,Q,K
channel 12      = side-to-move plane
remaining       = castling/en-passant/global planes
```

If the repository contradicts this assumption, update the adapter map and test it. For non-`simple_18` encodings, the deterministic extractor must require an explicit channel map.

After side-to-move canonicalization:

```text
friendly_piece = (B, 6, 64)
enemy_piece    = (B, 6, 64)
```

3. **Transport marginals**

Learn nonnegative type weights with `softplus` and add `epsilon_mass`.

```text
mu_fe = normalize(sum_p w_src[p] * friendly_piece[:,p,:] + eps/64)  # (B,64)
nu_fe = normalize(sum_q w_tgt[q] * enemy_piece[:,q,:]    + eps/64)  # (B,64)
mu_ef = normalize(sum_p w_src[p] * enemy_piece[:,p,:]    + eps/64)  # (B,64)
nu_ef = normalize(sum_q w_tgt[q] * friendly_piece[:,q,:] + eps/64)  # (B,64)
```

The two directions are `friendly_to_enemy` and `enemy_to_friendly`.

4. **Type-aware cost construction**

Precompute `geom[s,t]` with shape `(64,64,Dg)`, where `Dg` includes:

```text
df, dr, abs_df, abs_dr, manhattan, chebyshev,
same_file, same_rank, same_diag, same_antidiag,
knight_vector, forward_relation, center_source, center_target
```

For each occupied or empty square, build a type embedding by multiplying piece one-hot planes by learned embeddings and using an empty embedding where mass is only epsilon.

```text
src_emb = (B,64,E)
tgt_emb = (B,64,E)
cost_in = concat(src_emb[:,s], tgt_emb[:,t], geom[s,t])  # broadcast to (B,64,64,2E+Dg)
C = softplus(MLP(cost_in)) + c_min                       # (B,H,64,64), H default 4
```

Build separate cost tensors for the two directions, or share the same `TypeAwareTransportCost` with direction embeddings.

5. **Log-domain Sinkhorn**

Run fixed-iteration log-domain Sinkhorn for numerical stability.

```text
P = sinkhorn(mu, nu, C, tau, n_iters)  # (B,H,64,64)
```

Do this for both directions. Combined plan shape:

```text
P_all = (B, D, H, 64, 64), D=2
C_all = (B, D, H, 64, 64)
```

6. **Transport maps and statistics**

For each direction/head, compute global statistics:

```text
transport_cost       = sum_{s,t} P*C
plan_entropy         = -sum_{s,t} P*log(P)
plan_l2_concentration= sum_{s,t} P^2
expected_abs_df      = sum P*abs_df
expected_abs_dr      = sum P*abs_dr
expected_manhattan   = sum P*manhattan
same_line_mass       = sum P*(same_file or same_rank or same_diag or same_antidiag)
knight_vector_mass   = sum P*knight_vector
low_cost_soft_mass   = sum P*sigmoid((cost_threshold-C)/temperature)
```

Stats shape with defaults:

```text
T_stats = (B, D*H*9) = (B,72)
```

Project plan-derived maps back to board squares:

```text
source_cost_map = sum_t P[s,t] * C[s,t]                 # (B,D,H,64)
target_cost_map = sum_s P[s,t] * C[s,t]                 # (B,D,H,64)
source_conc_map = sum_t P[s,t]^2 / (mu[s]^2 + tiny)     # (B,D,H,64)
target_conc_map = sum_s P[s,t]^2 / (nu[t]^2 + tiny)     # (B,D,H,64)
```

Reshape to maps:

```text
T_maps = (B, D*H*4, 8, 8) = (B,32,8,8)
```

7. **Fusion and bottleneck**

```text
F = concat(A, T_maps, dim=channel)      # (B, W+32, 8, 8)
F = Conv1x1/GELU/Conv3x3/GELU(F)        # (B, W, 8, 8)
g = concat(global_mean_pool(F), global_max_pool(F), T_stats)  # (B, 2W+72)
z = MLP_bottleneck(g)                   # (B, z_dim), z_dim default 32
logits = Linear(z, 2)                   # (B,2)
```

Optional VIB mode can output `mu_z, logvar_z`; use `mu_z` at eval time. Keep `beta_kl=0` in the minimal run unless a VIB ablation is explicitly scheduled.

### Parameter-count estimate

For `simple_18`, width `W=64`, heads `H=4`, type embedding `E=16`, bottleneck `z_dim=32`:

- adapter conv stack: roughly 180k to 280k parameters depending on exact block count;
- type/cost MLP: roughly 5k to 15k parameters;
- fusion convs and bottleneck MLP: roughly 80k to 180k parameters;
- total expected range: **0.30M to 0.55M parameters**.

For `lc0_*_112`, the first adapter convolution increases by about `94*64*3*3 ≈ 54k` parameters relative to `simple_18`; the transport extractor must still use only mapped current-board piece planes.

### FLOP and memory estimate

Let `B=batch_size`, `D=2` directions, `H=4` heads, `N=64` squares, and `K=16` Sinkhorn iterations.

- Sinkhorn complexity: `O(B*D*H*K*N^2)`. With `B=512`, this is `512*2*4*16*4096 ≈ 268M` multiply/add/log-domain normalization scale operations per batch.
- Cost construction complexity: `O(B*D*H*N^2*hidden_cost)` if implemented naively. Codex should precompute geometry and vectorize source/target embeddings, then chunk over directions/heads if memory spikes.
- Plan/cost memory: `B*D*H*N*N*4` bytes per tensor. With `B=512`, one float32 tensor is about `512*2*4*4096*4 ≈ 67 MB`. Storing cost, plan, and intermediate log kernels can reach 200-300 MB before convolution activations.
- Chunking plan: add config `transport_chunk_heads`; compute one direction/head block at a time, accumulate stats/maps, and discard plan tensors before the next block if GPU memory is limited. For CPU tests, use `H=2`, `K=8`, and small batches.

There is no generated variable-length candidate set. The fixed “candidate” domain is the 64 current-board squares, so memory is predictable as `O(B*D*H*64^2)`.

### Required config fields

- `model.name: piece_target_transport_bottleneck`
- `model.input_channels`
- `model.num_classes: 2`
- `model.transport_heads` default `4`
- `model.transport_type_dim` default `16`
- `model.transport_cost_hidden` default `64`
- `model.sinkhorn_iters` default `16`
- `model.sinkhorn_tau` default `0.15`
- `model.epsilon_mass` default `1e-3`
- `model.adapter_width` default `64`
- `model.bottleneck_dim` default `32`
- `model.beta_kl` default `0.0`
- `model.transport_ablation` default `none`; valid central values include `cost_semantic_shuffle`, `material_only`, `cnn_only_matched`, `fixed_distance_cost`.
- `model.encoding_channel_map` optional; required for deterministic transport extraction on `lc0_static_112` or `lc0_bt4_112`.

### Encoding-adapter assumptions

- `simple_18`: supported in the first experiment. Use the known 12 current-board piece planes, side-to-move, castling, and en-passant planes. Unit-test the piece-plane order using at least two FENs with asymmetric material.
- `lc0_static_112`: the learned convolutional adapter may accept all 112 channels. The deterministic transport branch may only run if `encoding_channel_map` explicitly maps current-board white/black piece planes. If the map is missing, raise `ValueError` before training.
- `lc0_bt4_112`: same as `lc0_static_112`, with the additional rule that unavailable or zero-filled history planes must never be interpreted as current-board transport geometry. History channels go only to the learned adapter.

The model must return raw binary logits compatible with the shared trainer, reports, confusion matrices, predictions, and leaderboard code.

### Pseudocode

```text
class PieceTargetEntropicTransportBottleneck(nn.Module):
    def forward(x):
        A = board_adapter(x)
        friendly, enemy, stm = piece_adapter.extract_relative(x)

        stats_list = []
        maps_list = []
        for direction in [friendly_to_enemy, enemy_to_friendly]:
            mu, nu, src_types, tgt_types = make_marginals_and_types(direction)
            C = cost_net(src_types, tgt_types, precomputed_geometry, direction_id)
            if ablation == cost_semantic_shuffle:
                C = row_histogram_preserving_target_permutation(C)
            P = log_sinkhorn(mu, nu, C, tau, sinkhorn_iters)
            stats_list.append(global_transport_stats(P, C, geometry))
            maps_list.append(project_transport_maps(P, C, mu, nu))

        T_stats = concat(stats_list)
        T_maps = concat(maps_list).reshape(B, D*H*4, 8, 8)
        F = fuse_conv(concat(A, T_maps, channel_dim))
        g = concat(mean_pool(F), max_pool(F), T_stats)
        z = bottleneck_mlp(g)
        return classifier(z)
```

## 8. Loss, Training, And Regularization

- Primary loss: balanced binary cross-entropy via the shared coarse-binary trainer’s standard `CrossEntropyLoss` on logits `(B,2)`.
- Optional auxiliary loss: VIB KL term `β KL(q(z|x)||N(0,I))`; default `β=0.0` for the minimal deterministic run. If enabled, start with `β=1e-4` and report it separately.
- Class weighting: use the existing `class_weighting: balanced` behavior so class imbalance is handled exactly as for baselines.
- Batch size expectations: start with `batch_size=512` on `simple_18`. If memory is high, reduce batch size or set `transport_chunk_heads=true` before reducing model semantics.
- Optimizer defaults: AdamW, learning rate `1e-3`, weight decay `1e-4`.
- Epochs: use the current fair benchmark default of 3 epochs and existing early-stopping behavior unless the baseline configs use another standard.
- Regularizers: weight decay; optional small cost L2 penalty `λ=1e-5`; optional dropout `0.05` only in the bottleneck MLP. Avoid heavy augmentation in the first run.
- Determinism requirements: fixed seed `42`, deterministic PyTorch algorithms where project configs already support them, fixed random permutations for ablation modes, and no data-order changes relative to baselines.
- Keep unchanged for fair comparison: split paths, binary target mapping, batch-size policy where feasible, class weighting, number of epochs, early stopping, report generation, prediction artifact format, confusion matrices, and leaderboard update logic.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| **Cost semantic shuffle** | Row-wise permutes target-square semantics of `C` with a fixed seed while preserving each row’s cost histogram, material, source marginals, target marginals, number of Sinkhorn iterations, and parameter count. | Central claim: source-target chess geometry matters beyond cost distribution and material. | If equal to main, PT-ETB is not using meaningful transport geometry. Abandon this mechanism. |
| Material-only nuisance MLP | Replaces transport stats/maps with material counts by side/type, side-to-move, castling/en-passant scalars, and simple occupancy counts. | Main gains are not just material or global state shortcuts. | If equal to main, transport is unnecessary and likely shortcut-based. |
| CNN-only matched capacity | Removes transport branch and adds a parameter-matched convolution/MLP fuser. | Gains are from the OT operator, not parameter count. | If equal to main, the architecture is only a small CNN variant. |
| Fixed distance cost | Replaces learned type-aware cost with a fixed normalized mixture of Manhattan, Chebyshev, same-line, and knight-vector costs. | Learned type-specific geometry is useful. | If fixed cost wins, simplify to deterministic OT; if both fail, OT proximity is weak. |
| Type-blind transport | Keeps geometry but removes source/target piece-type embeddings and uses one shared cost. | Piece identities are part of the signal. | If equal to main, the model may only learn generic closeness/centrality. |
| Geometry-blind transport | Keeps piece-type weights but replaces `g(s,t)` with learned source and target square embeddings independent of relative geometry. | Relative board geometry, not just square priors, matters. | If equal to main, dataset square priors or material dominate. |
| No reverse direction | Uses only friendly-to-enemy transport, removing enemy-to-friendly transport. | Asymmetry between mover pressure and opponent counter-pressure is informative. | If no change, simplify to one direction; if worse, asymmetric balance matters. |
| Global-stats only | Removes transport maps and keeps only global transport statistics. | Spatial localization of the plan matters. | If equal to main, maps are unnecessary and a cheaper model is preferable. |
| Maps only | Removes global transport statistics and keeps projected maps. | Summary scalars carry the key signal. | If equal to main or better, avoid statistic engineering. |
| Fewer Sinkhorn iterations | Uses 4 or 8 iterations instead of 16. | Accurate transport constraints matter. | If equal, use fewer iterations; if unstable, keep 16. |
| Detached cost gradients | Stops gradients from classifier loss into `TypeAwareTransportCost`, leaving only initial/random cost features or a separately initialized fixed cost. | Supervised cost learning matters. | If equal, learned cost MLP is unnecessary or not being optimized usefully. |
| Fine-label-1 stress report | Not a training ablation; evaluate class `1` recall/precision at matched fine-label-`0` false-positive rate. | Near-puzzle ambiguity is handled better than baselines. | If class `1` collapses, PT-ETB may detect only obvious true puzzles. |

The smallest falsification ablation is **Cost semantic shuffle**. It is semantics-destroying but nuisance-preserving: it keeps obvious shortcuts such as material, side-to-move, source-square marginal, target-square marginal, and row-wise cost-degree/histogram while destroying which targets are geometrically reachable/near under the learned cost.

No rule-generated move set or candidate move set is used. The fixed transport domain is current-board squares, not legal moves or pseudo-legal move deltas.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- existing simple CNN on `simple_18`;
- existing residual CNN on `simple_18`;
- existing small/medium/deep CNN/residual variants if already in the leaderboard;
- LC0 BT4-style CNN/residual entries only as secondary context, not as the minimal fair comparison unless PT-ETB is also run on a 112-channel encoding;
- PT-ETB central ablations from Section 9.

Metrics to inspect:

- validation and test accuracy;
- AUROC if already reported;
- F1 or balanced accuracy if already reported;
- loss/calibration if already reported;
- required rectangular diagnostic matrix `true fine label 0/1/2 -> predicted binary 0/1` for the main model and every central ablation;
- near-puzzle diagnostic: class `1` recall at a matched fine-label-`0` false-positive rate. Use the same thresholding protocol across baseline, main, and ablations. A practical default is to set the threshold so the fine-label-`0` false-positive rate equals the best simple CNN’s fine-label-`0` false-positive rate, then compare fine-label-`1` recall and fine-label-`2` recall.

Required artifacts:

- trained model checkpoint for main PT-ETB;
- config YAML used for the run;
- validation and test metrics JSON/CSV;
- predictions file compatible with existing reports;
- rectangular `3x2` diagnostic matrix for main and central ablations;
- ablation reports for at least `cost_semantic_shuffle`, `material_only`, and `cnn_only_matched`;
- leaderboard update entry with parameter count and encoding.

Success threshold:

- PT-ETB improves over the best comparable `simple_18` baseline by at least **+1.0 percentage point** in balanced accuracy or F1, or at least **+0.01 AUROC** if AUROC is the project’s preferred stable metric; and
- PT-ETB improves class `1` near-puzzle recall at matched fine-label-`0` false-positive rate by at least **+2 percentage points**; and
- the central `cost_semantic_shuffle` ablation loses at least **half of the main model’s gain** over the best comparable baseline.

Failure threshold:

- PT-ETB is within ±0.2 percentage points of the best comparable baseline on the main metric, or worse; or
- `cost_semantic_shuffle` is statistically indistinguishable from PT-ETB across the standard seed/report; or
- class `1` recall at matched fine-label-`0` false-positive rate is worse than the best simple baseline by more than 1 percentage point.

Abandon the idea if:

- main PT-ETB does not beat a matched-capacity CNN and does not beat the cost-shuffled ablation; or
- the only improvement is on fine label `2` while fine label `1` deteriorates substantially; or
- training is unstable because Sinkhorn costs saturate despite log-domain implementation and cost clipping.

Scale the idea only if:

- the main model beats baselines on validation and test;
- the central shuffle ablation clearly underperforms;
- near-puzzle class `1` diagnostics improve or at least do not regress;
- compute overhead is acceptable relative to the baseline suite.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_0657_piece_transport/idea.yaml` | Create | Machine-readable idea metadata from Section 12, with status `draft` until first benchmark result. |
| `ideas/20260421_0657_piece_transport/math_thesis.md` | Create | Section 6 mathematical thesis, propositions, proof sketches, counterexamples, and self-critique. |
| `ideas/20260421_0657_piece_transport/architecture.md` | Create | Section 7 architecture, shapes, complexity, adapter assumptions, and pseudocode. |
| `ideas/20260421_0657_piece_transport/implementation_notes.md` | Create | Practical notes for log-domain Sinkhorn, channel maps, fail-closed behavior, and memory chunking. |
| `ideas/20260421_0657_piece_transport/trainer_notes.md` | Create | Loss/training/regularization notes from Section 8 and fair-comparison requirements. |
| `ideas/20260421_0657_piece_transport/ablations.md` | Create | Section 9 ablation table and exact central shuffle procedure. |
| `ideas/20260421_0657_piece_transport/train.py` | Create | Thin wrapper that invokes the shared training entrypoint with this idea’s config; no custom trainer fork unless unavoidable. |
| `ideas/20260421_0657_piece_transport/config.yaml` | Create | Full experiment config for `simple_18`, model name `piece_target_transport_bottleneck`, transport heads/iterations, and default benchmark split. |
| `ideas/20260421_0657_piece_transport/report_template.md` | Create | Report skeleton requiring main metrics, `3x2` fine-label matrix, near-puzzle matched-FPR diagnostic, and ablation comparison. |
| `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this packet to imported research memory after implementation, including outcome and anti-duplicate guidance. Preserve all hard leakage/label/falsification constraints. |
| `src/chess_nn_playground/models/trunk/piece_target_transport.py` | Create | Implement `PiecePlaneAdapter`, `RelativeBoardCanonicalizer`, `TypeAwareTransportCost`, `LogSinkhornTransport`, `TransportSummaryProjector`, and `PieceTargetEntropicTransportBottleneck`. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `piece_target_transport_bottleneck` and builder function. |
| `configs/piece_target_transport_simple18.yaml` | Create | Benchmark config using `simple_18`, current split paths, balanced coarse binary mode, and model defaults. |
| `tests/test_piece_target_transport.py` | Create | Shape tests: `(B,C,8,8)->(B,2)`, finite logits, deterministic forward with fixed seed, and ablation mode shape compatibility. |
| `tests/test_piece_plane_adapter_fail_closed.py` | Create | Verify `simple_18` channel extraction with toy tensors and verify 112-channel encodings raise `ValueError` without explicit maps. |
| `tests/test_log_sinkhorn_transport.py` | Create | Verify plan rows/columns approximately match marginals, finite gradients exist, and cost shuffle preserves row-wise cost histograms. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0657_tuesday_los_angeles_piece_transport.md
  generated_at: 2026-04-21T06:57:24-07:00
  weekday: Tuesday
  timezone: los_angeles
  idea_slug: piece_transport
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_0657_piece_transport
  name: Piece-Target Entropic Transport Bottleneck
  slug: piece_transport
  status: draft
  created_at: 2026-04-21T06:57:24-07:00
  author: ChatGPT Pro
  short_thesis: Learn a compact puzzle-likeness representation from entropy-regularized optimal transport between current-board side-to-move pieces and opponent/friendly target pieces.
  novelty_claim: Uses a type-aware Sinkhorn transport operator over existing occupied board squares, not a CNN scaling trick, attack/sheaf incidence complex, or one-ply move-delta pool.
  expected_advantage: Should capture concentrated asymmetric piece-target geometry that is label-relevant beyond material and square-count shortcuts.
  central_falsification_ablation: cost_semantic_shuffle preserving material, marginals, cost histograms, parameter count, and Sinkhorn computation while destroying target-square semantics.
  target_task: coarse_binary
  input_representation: simple_18 first; lc0_static_112 and lc0_bt4_112 only with explicit fail-closed current-board piece channel maps
  output_heads: binary_logits
  compute_notes: O(batch * directions * heads * sinkhorn_iters * 64^2); default directions=2 heads=4 iters=16; chunk heads if memory is high.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/piece_target_transport_simple18.yaml
  model_path: src/chess_nn_playground/models/trunk/piece_target_transport.py
  latest_result_path: null
  notes: Mandatory diagnostic is fine-label 0/1/2 by predicted binary 0/1 for main and central ablations.
```

```yaml
config_yaml:
  run:
    name: piece_target_transport_simple18
    output_dir: results
  seed: 42
  deterministic: true
  mode: coarse_binary
  device: nvidia
  data:
    train_path: data/splits/crtk_sample_3class/split_train.parquet
    val_path: data/splits/crtk_sample_3class/split_val.parquet
    test_path: data/splits/crtk_sample_3class/split_test.parquet
    encoding: simple_18
    cache_features: false
  model:
    name: piece_target_transport_bottleneck
    input_channels: 18
    num_classes: 2
  training:
    epochs: 3
    batch_size: 512
    num_workers: 0
    learning_rate: 0.001
    weight_decay: 0.0001
    class_weighting: balanced
    early_stopping_patience: 2
    mixed_precision: false
```

```yaml
model_spec:
  model_name: piece_target_transport_bottleneck
  file_path: src/chess_nn_playground/models/trunk/piece_target_transport.py
  builder_function: build_piece_target_transport_bottleneck
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - PiecePlaneAdapter
    - RelativeBoardCanonicalizer
    - TypeAwareTransportCost
    - LogSinkhornTransport
    - TransportSummaryProjector
    - PieceTargetEntropicTransportBottleneck
  required_config_fields:
    - input_channels
    - num_classes
    - transport_heads
    - transport_type_dim
    - transport_cost_hidden
    - sinkhorn_iters
    - sinkhorn_tau
    - epsilon_mass
    - adapter_width
    - bottleneck_dim
    - transport_ablation
    - encoding_channel_map
  expected_parameter_count: 0.30M-0.55M for simple_18 default width 64 and 4 transport heads
  expected_memory_notes: Plan and cost tensors scale as O(batch * 2 directions * heads * 64 * 64); one float32 plan tensor is about 67 MB at batch 512, heads 4, directions 2.
```

```yaml
research_continuity:
  idea_fingerprint: current-board relative piece/target occupancy measures + learned type-aware entropic Sinkhorn transport over 64 existing squares + compact transport-map/stat bottleneck + binary puzzle-likeness classification
  already_researched_family_overlap: Uses current-board geometry and piece occupancy, but does not build attack/defense incidence, sheaf restrictions, Hodge/Laplacian/curvature/tension energy, file-mirror sheaf, or one-ply move-delta landscapes.
  closest_duplicate_risk: Could collapse into a material/proximity heuristic; central cost-semantic shuffle and material-only ablations are required to test that risk.
  do_not_repeat_if_this_fails:
    - Learned Sinkhorn transport between current-board friendly pieces and opponent targets.
    - Entropic OT bottlenecks over 64 board squares with only type-aware geometric cost changes.
    - Variants that merely add more transport heads, larger cost MLPs, extra distance features, or different pooling over the same piece-target plans.
  suggested_next_search_directions:
    - Label-safe ordinal or selective prediction focused on fine-label-1 ambiguity.
    - Causal invariance across material phase and encoding family without making transport the central operator.
    - Masked generative compression or minimum-description-length motifs with strong source-artifact controls.
    - Calibration-first models that improve near-puzzle diagnostics without fabricating labels.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add “Piece-Target Entropic Transport Bottleneck” to the imported research memory after implementation, with its measured result and central ablation outcome. | Prevents future packets from repeating Sinkhorn piece-target transport with only extra heads or altered pooling. | `Imported Research Memory` |
| Add anti-duplicate wording: “Do not propose current-board piece-target Sinkhorn/optimal-transport over occupied squares unless the operator changes beyond cost features, head count, entropy temperature, or transport pooling.” | Makes the distinction from future OT variants explicit. | `Research Continuity` or anti-duplicate paragraph after imported move-delta rules |
| Record whether `simple_18` piece channel order was confirmed or corrected by Codex tests. | Future research packets should not keep restating an uncertain adapter assumption. | `Project Context You Must Respect` under available encodings |
| Add a reusable falsification requirement for transport-like operators: include cost-histogram-preserving semantic shuffle and material/marginal-only controls. | This packet found a concrete way to distinguish real structure from proximity/material shortcuts. | `Depth requirements` or ablation instructions |
| If PT-ETB fails, add “do not repeat entropic OT piece-target bottlenecks” to the hard imported memory with the failure reason. | Keeps the iterative loop honest and avoids cosmetic repetition. | `Imported Research Memory` |
| If PT-ETB succeeds, ask next-cycle research to explain whether improvement is strongest on fine label `1` or `2`. | The next idea should react to whether transport helps ambiguous near-puzzles or only obvious puzzles. | `Research Continuity` |

Do not weaken leakage rules, label rules, falsification requirements, or anti-duplicate requirements.

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0657_tuesday_los_angeles_piece_transport.md`
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes, `simple_18` on the existing `crtk_sample_3class` split
- Falsification criterion is concrete: yes, cost-semantic shuffle plus material-only and matched-CNN controls
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported research packets completed: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
