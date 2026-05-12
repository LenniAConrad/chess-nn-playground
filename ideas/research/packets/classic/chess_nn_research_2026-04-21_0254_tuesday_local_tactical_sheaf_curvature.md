# Codex Handoff Packet: Tactical Sheaf Curvature Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0254_tuesday_local_tactical_sheaf_curvature.md`
- Generated at: 2026-04-21 02:54:19 UTC-07:00
- Weekday: Tuesday
- Timezone: local
- Idea slug: `tactical_sheaf_curvature`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Tactical Sheaf Curvature Network, abbreviated `TSCN`.
- One-sentence thesis: Chess puzzle-likeness is better detected as localized inconsistency in a board-position attack-candidate relation complex than as another image-recognition texture over the 8x8 grid.
- Idea fingerprint: Build a fixed typed directed relation complex over the 64 squares from chess geometry only, learn diagonal sheaf restriction maps and sparse edge gates from the input tensor, classify from node embeddings plus layer-wise sheaf frustration and target-centered curvature statistics.
- Why this is not a common CNN/ResNet/Transformer variant: The central computation is a typed sheaf coboundary and Laplacian/frustration operator on directed chess relations, not translation convolution, residual image filtering, square self-attention, or an LC0 policy/value tower.
- Current-data minimal experiment: Train `TacticalSheafCurvatureNet` on `simple_18` using the existing `crtk_sample_3class` train/val/test split, report binary metrics and the fine-label 0/1/2 to binary-output confusion table, then run the smallest central ablation: replace the sheaf coboundary/frustration branch with parameter-matched typed edge message passing while keeping the same input adapter and classifier.
- Expected information gain if it fails: A clean failure says static attack-candidate tension is not sufficient for this benchmark, so the next cycle should test move-conditioned or counterfactual-search surrogates that remain engine-free rather than trying larger CNNs or generic attention.

## 3. Problem Restatement And Data Contract

The task is binary classification of a single chess board position as non-puzzle or puzzle-like.

Labels and reporting contract:

- Fine label `0`: known non-puzzle.
- Fine label `1`: verified near-puzzle.
- Fine label `2`: verified puzzle.
- Binary target for training and benchmark reporting: fine label `0 -> 0`; fine labels `1` and `2 -> 1`.
- Benchmark reports must still include the table `true fine label 0/1/2 -> predicted binary output 0/1`.

Allowed neural-network inputs:

- Board-position tensors from the existing encodings: `simple_18`, `lc0_static_112`, and `lc0_bt4_112`.
- Tensor shape: `(batch, C, 8, 8)`.
- Model output: logits of shape `(batch, num_classes)`, normally `(batch, 2)`.
- Deterministic square geometry derived from board coordinates: rank/file, direction, distance, same rank/file/diagonal, knight displacement, king-neighborhood displacement, and pawn-capture direction candidates.

Forbidden inputs and leakage checklist:

- Do not use Stockfish scores, principal variations, node counts, tablebase results, engine best moves, engine verification metadata, source labels, proposed labels, or unresolved-candidate status as neural inputs.
- Do not fabricate fine label `1` or fine label `2` examples.
- Treat unresolved candidates as unresolved; do not relabel them by confidence.
- Do not precompute tactical labels with an engine or legal-move searcher.
- The relation complex must be computed from board geometry and the input tensor only. Its fixed edge list is legal because it uses no target labels and no engine evaluation.

Benchmark split:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

Encoding support:

- First implementation should work for every `C` by using a learned `1x1` channel adapter before the sheaf operator.
- No hard-coded piece-plane map is required for the core experiment. The candidate relation complex is geometric and typed, while gates learn from whatever encoding channels are present.
- If the repository already exposes reliable plane metadata, Codex may log optional diagnostics such as piece-occupancy gate correlation, but those diagnostics must not become new input features or labels.

## 4. Research Map

The idea borrows mathematical operators, not task-specific recipes.

- Neural Sheaf Diffusion: Bodnar, Di Giovanni, Chamberlain, Liò, and Bronstein introduce cellular sheaves on graphs, with node and edge stalks and restriction maps that generalize graph diffusion and help heterophilic settings. Borrowed: non-trivial sheaf restrictions and quadratic sheaf energy. Not copied: their benchmark tasks, full sheaf-learning setup, or graph datasets. URL: https://arxiv.org/abs/2202.04579
- Sheaf Neural Networks with Connection Laplacians: Barbero, Bodnar, de Ocáriz Borde, Bronstein, Veličković, and Liò study connection-Laplacian style sheaf networks. Borrowed: the idea that transports/restrictions can encode relation-dependent local geometry. Not copied: exact architecture or normalization. URL: https://proceedings.mlr.press/v196/barbero22a.html
- Polynomial Neural Sheaf Diffusion: Borgi, Silvestri, and Liò propose stable polynomial spectral filtering on sheaf Laplacians. Borrowed: stability motivation for bounded diagonal transports and low-degree filtering. Not copied: their polynomial recurrence or claims of state-of-the-art performance. URL: https://arxiv.org/abs/2512.00242
- Simplicial Neural Networks: Ebli, Defferrard, and Spreemann frame higher-order interactions through simplicial complexes and Hodge-type operators. Borrowed: higher-order interaction mindset; the target-centered variance term is a cheap 2-cell proxy. Not copied: their simplicial convolution implementation. URL: https://arxiv.org/abs/2010.03633
- HodgeNet: Roddenberry and Segarra integrate discrete Hodge theory with graph neural architectures for edge data. Borrowed: representing meaningful signal on edges, not only vertices. Not copied: flow interpolation task or exact aggregation GNN. URL: https://arxiv.org/abs/1912.02354
- Higher-order networks literature, such as Bick et al.'s SIAM Review article, motivates using objects beyond pairwise graphs when interactions are naturally multi-agent. Borrowed: vocabulary of higher-order tension. Not copied: any dataset or algorithm. URL: https://doi.org/10.1137/21M1414024
- Finite Group Equivariant Neural Networks for Games: Carroll and Beel note that games have usable but nontrivial symmetries and that naive equivariance can be too weak or mis-specified. Borrowed: caution about game symmetries. Not copied: their FGNN construction. URL: https://arxiv.org/abs/2009.05027
- Enhancing Chess Reinforcement Learning with Graph Representation: Rigaux et al. explore chess as a graph representation rather than a grid-only representation. Borrowed: chess-specific motivation for graph-structured board relations. Not copied: reinforcement-learning target, policy format, or edge-feature GAT layer. URL: https://arxiv.org/abs/2410.23753

All URLs above were selected as research anchors. The exact model proposed here is not claimed to appear in any of them.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN over `simple_18` | simple CNN | Already covered; mostly tests local image texture, not tactical relation inconsistency. |
| Deeper/wider ordinary CNN | small/medium/deep CNN variants | Violates the spirit of the request because depth/width is routine scaling rather than a new inductive bias. |
| Standard residual CNN | residual CNN | Residual locality still treats the board as an image; it does not explicitly represent attack/defense relations. |
| LC0-style CNN tower | LC0 BT4-style CNN | Too close to an existing baseline and too generic for puzzle-likeness rather than chess policy/value learning. |
| LC0-style residual CNN with minor changes | LC0 BT4-style residual CNN variants | A near-clone would mainly test capacity and optimizer details, not a distinct research thesis. |
| Ordinary ViT or square Transformer | No exact baseline, but a common square-token alternative | Global attention is flexible but underspecified; without typed chess relations it is a generic architecture swap. |
| Plain GNN-on-squares | Graph-style chess representation | Pairwise message passing on adjacent or all squares is too broad and can collapse into an attention-like model. TSCN instead uses typed sheaf restrictions and frustration energies. |
| Hyperparameter tuning | All existing baselines | Tuning batch size, learning rate, dropout, optimizer, or schedules is not a research idea. |
| Ensembling multiple baselines | Any combination of current models | Ensembling may improve leaderboard numbers but obscures whether the central inductive bias works. |
| Engine-supervised tactical proxy | None allowed | Illegal under the hard constraints because it would leak Stockfish/PV/node/evaluation information. |
| Hand-coded chess motif classifier | None | Too brittle and not a neural architecture; also risks smuggling in verification heuristics. |
| More data or label expansion | Dataset-level change | Not allowed as the core idea and does not answer whether current board tensors contain a learnable puzzle-likeness signal. |

## 6. Mathematical Thesis

### Input space

Let an input position be `X in R^(C x 8 x 8)`. Flatten the board into `V = {0, ..., 63}`. A learned adapter maps each square tensor to an initial stalk signal:

```text
h_v = Adapter(X)[:, v] in R^d
x_v = P h_v in R^s
```

where `d` is the node hidden dimension and `s` is the sheaf stalk dimension.

### Target definition

The supervised target is binary puzzle-likeness:

```text
y = 0 if fine_label == 0
y = 1 if fine_label in {1, 2}
```

The model may never see fine labels as input features. Fine labels are used only for reporting the required diagnostic confusion table.

### Distribution assumptions

The working assumption is not that every puzzle is tactical. The weaker assumption is that, in the provided benchmark distribution, many verified and near-verified puzzles exhibit unusual local attack/defense tension relative to ordinary non-puzzles. This may include overloaded pieces, pinned lines, converging attackers, undefended high-value targets, discovered lines, king-zone pressure, and piece coordination. The model is designed to learn these tensions from data without engine labels.

### Symmetry and equivariance assumptions

Chess is not fully rotation/reflection invariant. Pawns move directionally, castling is asymmetric in board representation, color and side-to-move matter, and vertical flips are not rule-preserving unless color and side metadata are transformed consistently.

TSCN therefore uses only safe partial parameter tying:

- Tie left-right mirror relation parameters: east/west, northeast/northwest, southeast/southwest, and mirrored knight offsets.
- Do not tie north and south pawn-like relation types.
- Do not assume 90-degree rotation symmetry.
- Do not force color-swap or 180-degree equivariance unless the encoding registry already provides a verified transform for all channels. The minimal experiment should not require such a transform.

### Core hypothesis

Puzzle-like positions are enriched for high sheaf frustration and target-centered curvature on typed chess-geometric relations after shallow learned diffusion. Non-puzzles may contain attacks, but their local constraints are more often mutually compatible: defenses align, target pressure is ordinary, and line relations do not create concentrated contradictory evidence.

### Formal operator/object

Construct a fixed directed typed relation set `E` over squares. It contains candidate relations, not engine-legal moves:

- Sliding candidates along rank, file, and diagonal rays, with distance bucketed as `1`, `2`, `3`, and `4+`.
- Knight displacement candidates.
- King-neighborhood candidates.
- Pawn-capture-direction candidates for both color directions, kept as separate relation types.

For layer `l`, each directed edge `e=(u -> v, r)` has geometry embedding `q_r`. Learn diagonal restrictions:

```text
rho_src(e) = diag(a_l(q_r))
rho_dst(e) = diag(b_l(q_r))
```

with `a_l` and `b_l` bounded by `tanh` or spectral clipping. The sheaf coboundary is

```text
(delta_l x)_e = rho_dst(e) x_v - rho_src(e) x_u
```

An input-dependent sparse gate is

```text
g_e = sigmoid(Gate_l([x_u, x_v, q_r, distance_features]))
```

The weighted sheaf energy is

```text
E_l(x) = sum_{e in E} g_e ||(delta_l x)_e||_2^2
```

The diffusion update is a stable residual step:

```text
x <- x - eta_l * D_l^{-1} delta_l^T G_l delta_l x + NodeMLP_l(x)
```

where `G_l` is diagonal with entries `g_e`, and `D_l` is a per-node degree normalizer with epsilon stabilization.

Target-centered curvature proxy for each target square `v`:

```text
z_e = g_e * rho_src(e) x_u, for incoming edges e=(u -> v)
curv_v = weighted_variance({z_e : head(e)=v})
```

This is equivalent to a soft all-pairs disagreement among incoming tactical claims on the same target. It approximates a small 2-cell/higher-order interaction without enumerating all attacker-target-defender triples.

### Proposition or objective

Proposition: For fixed gates `g_e >= 0`, bounded restrictions, and step size `0 <= eta <= 1 / lambda_max(D^{-1/2} delta^T G delta D^{-1/2})`, the pure sheaf diffusion step decreases or preserves the quadratic sheaf energy. Also, if a board transform is an automorphism of the typed relation complex and all tied relation parameters are transformed consistently, the sheaf-energy features are equivariant/invariant under that transform depending on the pooling.

Training objective:

```text
min_theta CrossEntropy(logits_theta(X), y)
          + lambda_energy * mean_l log(1 + E_l / |E|)
          + lambda_gate * mean_e g_e
          + lambda_balance * gate_entropy_floor_penalty
```

The regularizers are auxiliary. The classifier must be evaluated with and without them.

### Proof sketch

For fixed gates and restrictions, `L = delta^T G delta` is positive semidefinite because `z^T L z = ||G^(1/2) delta z||^2 >= 0`. A gradient descent step on the quadratic energy `0.5 x^T L x`, normalized by positive degrees, is non-expansive for sufficiently small `eta`. Typed relation tying gives equivariance because applying a valid relation-complex automorphism only permutes vertices and relation-indexed restrictions; sums, variances, and pooled energies commute with that permutation.

### What is proven

- The fixed-gate sheaf energy is nonnegative.
- The bounded-step pure diffusion component is stable in the quadratic-energy sense.
- The pooled energy statistics respect the explicitly tied typed-relation symmetries.

### What is hypothesized

- High learned sheaf frustration and target-centered curvature correlate with puzzle-likeness on this dataset.
- The learned gates will suppress irrelevant geometric candidate edges and emphasize chess-useful relations without needing explicit engine analysis.
- Fine label `1` near-puzzles will occupy an intermediate or high-curvature region relative to fine labels `0` and `2`.

### Counterexamples

- Quiet endgame studies, zugzwang, fortress breaks, stalemate nets, and long strategic puzzles may be puzzle-like with low immediate line tension.
- Non-puzzle tactical melees can have high attack/defense curvature but no forcing solution.
- Positions where castling rights, repetition, fifty-move status, or move history are decisive may not be represented sufficiently in static board channels.
- If dataset source artifacts dominate labels, any geometry-only inductive bias may underperform a generic classifier that learns those artifacts from encodings.

## 7. Architecture Specification

### Module names

Primary module: `TacticalSheafCurvatureNet`.

Suggested internal modules:

- `BoardChannelAdapter`
- `TypedRelationComplex`
- `SheafRestrictionGenerator`
- `SheafGate`
- `TacticalSheafLayer`
- `CurvatureStatsPool`
- `TSCNClassifierHead`

### Forward pass and tensor shapes

Input: `x_raw` of shape `(B, C, 8, 8)`.

Pseudocode:

```text
h_grid = BoardChannelAdapter(x_raw)               # (B, d_node, 8, 8)
h = flatten_squares(h_grid)                       # (B, 64, d_node)
x = stalk_projection(h)                           # (B, 64, d_stalk)
edge_index, edge_type, edge_geom = relation_complex.buffers
all_layer_stats = []

for layer in sheaf_layers:
    src = edge_index[0]                            # (E,)
    dst = edge_index[1]                            # (E,)
    x_src = gather(x, src)                         # (B, E, d_stalk)
    x_dst = gather(x, dst)                         # (B, E, d_stalk)
    q = relation_embedding(edge_type, edge_geom)   # (E, d_geom), broadcast to B

    a, b = restriction_generator(q)                # each (E, d_stalk), bounded
    gate = sheaf_gate(x_src, x_dst, q)             # (B, E, 1)
    delta = b * x_dst - a * x_src                  # (B, E, d_stalk)
    edge_energy = gate * squared_norm(delta)       # (B, E, 1)

    lap_update = scatter_dst(b * gate * delta) - scatter_src(a * gate * delta)
    lap_update = degree_normalize(lap_update)      # (B, 64, d_stalk)
    x = layer_norm(x - eta * lap_update + node_mlp(x))

    curv = incoming_weighted_variance(a * x_src, gate, dst)
    stats = pool(edge_energy, curv, gate, by_relation_groups=True)
    all_layer_stats.append(stats)

node_pool = concat(mean(x over 64), max(x over 64), std(x over 64))
stat_pool = concat(all_layer_stats)
logits = classifier(concat(node_pool, stat_pool))  # (B, num_classes)
return logits
```

### Relation complex details

`TypedRelationComplex` should precompute buffers once on CPU and register them on the module:

- `edge_index`: shape `(2, E)` with directed candidate edges.
- `edge_type`: shape `(E,)` integer type IDs.
- `edge_geom`: shape `(E, g)` containing normalized source rank/file, target rank/file, signed delta rank/file, absolute distance, direction one-hot or ID, and distance bucket.
- `relation_group`: shape `(E,)` for pooled stats, e.g. `orthogonal_ray`, `diagonal_ray`, `knight`, `king`, `pawn_up_capture`, `pawn_down_capture`.

Expected edge count is roughly 2,000 to 2,600 directed edges depending on whether all ray distances are included. This is small enough for dense gather/scatter on GPU.

### Parameter estimate

Recommended first config:

- `d_node = 64`
- `d_stalk = 48`
- `d_geom = 24`
- `num_layers = 3`
- `gate_hidden = 64`
- `classifier_hidden = 128`

Approximate trainable parameter count:

- Input adapter: `C * 64 + 64`, about 1.2k for `simple_18` and 7.2k for `lc0_*_112`.
- Stalk projection and node MLPs: about 35k to 55k.
- Relation embeddings and restriction generators: about 8k to 15k.
- Gate networks: about 30k to 45k if layer-specific.
- Classifier and stats projection: about 25k to 45k.
- Total expected range: 100k to 170k parameters, depending on exact stats dimensionality. This intentionally stays smaller than many CNN towers.

### FLOP and complexity estimate

Let `E ≈ 2400`, `L = 3`, `s = 48`, and batch size `B`.

- Sheaf delta and scatter complexity: `O(B * L * E * s)`.
- Gate MLP complexity: `O(B * L * E * gate_hidden * (2s + d_geom))`, but with small constants and vectorized edge batches.
- Memory for edge activations: roughly `B * E * s * 4 bytes` per layer before autograd overhead. For `B=256`, `E=2400`, `s=48`, this is about 118 MB for one major edge tensor in fp32; mixed precision or `B=128` may be needed on small GPUs.

### Config fields

Minimum config fields:

```yaml
model:
  name: tactical_sheaf_curvature
  input_channels: null
  num_classes: 2
  d_node: 64
  d_stalk: 48
  d_geom: 24
  num_layers: 3
  gate_hidden: 64
  classifier_hidden: 128
  relation_distance_buckets: [1, 2, 3, 4]
  include_ray_edges: true
  include_knight_edges: true
  include_king_edges: true
  include_pawn_candidate_edges: true
  tie_file_mirror_relations: true
  tie_north_south_relations: false
  eta_init: 0.25
  gate_dropout: 0.05
  node_dropout: 0.10
  stats_pooling: [mean, std, max]
```

### Encoding support

- `simple_18`: primary minimal experiment.
- `lc0_static_112`: supported by the same adapter; use only if baseline comparison infrastructure already supports it.
- `lc0_bt4_112`: supported by the same adapter; history channels are treated as input channels, not as labels or engine data.

### Logits interface

The class must subclass `torch.nn.Module` and expose:

```text
forward(x: Tensor[B, C, 8, 8]) -> Tensor[B, num_classes]
```

No auxiliary return values should be required for training. If diagnostics are needed, add an optional `return_aux=False` flag without changing the default logits-only behavior.

## 8. Loss, Training, And Regularization

Primary loss:

- Use standard cross-entropy on binary targets.
- Binary mapping: fine label `0 -> 0`; fine labels `1` and `2 -> 1`.

Class weighting:

- Use the same class-weighting policy as the strongest fair baseline if the project already has one.
- Otherwise use inverse-frequency binary weights computed on the training split only.
- Do not use fine-label-specific weights for the primary loss in the minimal experiment; fine labels are for reporting.

Optimizer and learning rate:

- Optimizer: `AdamW`.
- Initial LR: `3e-4`.
- Weight decay: `1e-4`.
- Scheduler: keep identical to the current baseline harness if one exists; otherwise cosine decay with warmup is acceptable but must be shared by ablations.

Batch size:

- Start with `batch_size = 256` for `simple_18`.
- Use `batch_size = 128` if GPU memory is tight, especially for `lc0_*_112`.
- Keep batch size fixed across TSCN and its central ablation for fair comparison.

Regularizers:

```text
L_total = CE
        + lambda_energy * mean_l log(1 + mean_edge_energy_l)
        + lambda_gate * mean_l mean_e gate_l,e
        + lambda_gate_entropy_floor * max(0, entropy_floor - mean_gate_entropy)
```

Recommended first values:

- `lambda_energy = 1e-4`
- `lambda_gate = 1e-4`
- `lambda_gate_entropy_floor = 1e-5`
- `entropy_floor = 0.05`

Rationale:

- `lambda_energy` prevents unstable high-frustration blowups.
- `lambda_gate` encourages sparse tactical relation use.
- Entropy floor prevents all gates from collapsing to zero early.

Optional auxiliary diagnostic loss:

- A small margin loss between positive and negative mean curvature can be tested only after the primary model is benchmarked.
- It must use only binary training labels, not fine label `1` versus `2` distinctions.
- It is not part of the minimal falsification run.

Determinism:

- Run at least 3 seeds if the benchmark harness supports it.
- Fix train/val/test split paths exactly.
- Use deterministic dataloader seeding.
- Log model parameter count and effective edge count.

What must stay fixed for fair comparison:

- Dataset split.
- Input encoding.
- Binary label mapping.
- Training epochs or early-stopping patience.
- Batch size where feasible.
- Optimizer family and schedule, unless a baseline cannot run under the same settings.
- Evaluation metrics and threshold-selection protocol.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Parameter-matched typed edge message passing | Replace `delta^T G delta` sheaf coboundary update and frustration stats with ordinary gated messages on the same edge list | The sheaf inconsistency operator matters beyond having chess-geometry edges | If equal or better, the central sheaf-frustration claim is falsified; keep typed relations but drop sheaf language next cycle. |
| No curvature stats | Keep sheaf diffusion but remove target-centered weighted-variance statistics from classifier input | Higher-order attacker/defender disagreement contributes signal | If performance is unchanged, puzzle signal is in node diffusion only, not 2-cell-like curvature. |
| No gates | Set all `g_e = 1` and remove `SheafGate` | Learned sparsification of candidate relations is necessary | If unchanged, the fixed geometry alone is enough; simplify model. If much worse, gates are essential. |
| Ray-only relations | Remove knight, king, and pawn-candidate edges | Long-line tactical relations dominate | If unchanged, initial implementation can be ray-focused. If worse, local jumps and pawn direction matter. |
| Jump-only relations | Remove sliding ray edges | Non-ray local geometry is sufficient | If close to full model, the line-complex thesis is weaker than expected. |
| No file-mirror tying | Untie left-right mirrored relation parameters | Partial equivariance helps sample efficiency | If unchanged, tying is not material for this dataset. If better, tying is too restrictive. |
| Mean/max node pool only | Remove all edge-energy and curvature features from final classifier | Classification can be done from final node embeddings alone | If unchanged, explicit energy readout is unnecessary. |
| Randomized relation types | Keep edge endpoints but randomly permute relation-type IDs before training | Chess-geometric typing matters | If unchanged, model is exploiting generic connectivity rather than chess relation semantics. |
| Smallest central falsifier | Same as first row: parameter-matched typed edge message passing | Directly tests whether sheaf coboundary/frustration is the core value | If this ablation matches TSCN within 0.5 percentage points AUROC over 3 seeds, abandon TSCN as a sheaf idea. |

## 10. Benchmark And Falsification Criteria

Baselines:

- Best existing simple CNN on the same encoding.
- Best existing residual CNN on the same encoding.
- Best existing small/medium/deep CNN variant with comparable training budget.
- Best existing LC0-style CNN/residual variant when using `lc0_*_112` encodings.
- The parameter-matched typed edge message-passing ablation described above.

Metrics:

- Binary accuracy.
- Binary macro-F1.
- ROC-AUC.
- Average precision / PR-AUC, especially if classes are imbalanced.
- Calibration error if already supported by the benchmark harness.
- Required fine-label diagnostic table: true fine label `0/1/2` to predicted binary output `0/1`.
- Fine label `1` recall at a fixed validation-chosen false-positive rate on fine label `0`.

Artifacts to save:

- Config YAML.
- Model parameter count.
- Edge count and relation group counts.
- Per-seed metrics.
- Confusion tables by fine label.
- Gate and curvature summary histograms by binary class, computed on validation/test only for analysis.
- Ablation metrics under the same seeds.

Success threshold:

- Primary success: mean test ROC-AUC over 3 seeds improves by at least `0.015` over the strongest same-encoding baseline, and mean macro-F1 improves by at least `0.01` without worsening fine-label-0 false-positive rate by more than `1` percentage point at the chosen threshold.
- Secondary success: fine-label-1 recall at the validation-selected operating point improves by at least `3` percentage points while fine-label-0 false-positive rate stays within `1` percentage point of baseline.

Failure threshold:

- Mean test ROC-AUC improvement under `0.005` over the strongest same-encoding baseline, or TSCN loses to the parameter-matched typed message-passing ablation by any statistically consistent margin over 3 seeds.

Abandon condition:

- Abandon the sheaf-curvature idea if the smallest central falsifier matches TSCN within `0.005` ROC-AUC and within `0.005` macro-F1, while curvature/gate diagnostics show no separation between binary classes.

Scaling condition:

- If TSCN clears the success threshold on `simple_18`, run the same model family on `lc0_static_112` and `lc0_bt4_112` without changing the central operator.
- Only after same-operator success may Codex test larger `d_stalk` or more layers. Do not treat scaling as the original research claim.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_tactical_sheaf_curvature/README.md` | Create | Copy this handoff packet or a concise implementation-facing version of it. |
| `ideas/20260421_tactical_sheaf_curvature/results.md` | Create after experiments | Per-seed metrics, tables, plots references, success/failure decision, and notes for next research cycle. |
| `src/chess_nn_playground/models/trunk/tactical_sheaf_curvature.py` | Create | `TacticalSheafCurvatureNet` and internal modules: adapter, relation complex builder, restriction generator, gate, sheaf layer, stats pool, classifier head. |
| `src/chess_nn_playground/models/registry.py` | Modify | Register model name `tactical_sheaf_curvature` without breaking existing models. |
| `configs/tactical_sheaf_curvature_simple18.yaml` | Create | Minimal experiment config using `simple_18`, current split paths, and the recommended first hyperparameters. |
| `configs/tactical_sheaf_curvature_lc0_static_112.yaml` | Create only if harness supports it easily | Same model with `input_channels: 112`; do not tune architecture first. |
| `configs/tactical_sheaf_curvature_lc0_bt4_112.yaml` | Create only if harness supports it easily | Same model with `input_channels: 112`; use after simple_18 results. |
| `tests/test_tactical_sheaf_curvature.py` | Create | Shape tests, deterministic edge-count tests, CPU forward pass test, backward pass smoke test, and no-NaN test. |
| `tests/test_model_registry.py` | Modify if needed | Ensure registry can instantiate `tactical_sheaf_curvature`. |
| `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update after consuming this output | Preserve all hard constraints; add reusable lessons, anti-duplicate rules, clearer output requirements, and failure-mode guidance discovered during this research pass. |

Codex should implement only the architecture and experiment harness changes needed to test this idea. It should not add engine features, new labels, or data expansion.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0254_tuesday_local_tactical_sheaf_curvature.md
  generated_at: "2026-04-21 02:54:19 UTC-07:00"
  idea_slug: tactical_sheaf_curvature
  idea_name: Tactical Sheaf Curvature Network
  intended_next_consumer: Codex
```

```yaml
idea_yaml:
  idea_id: "20260421_tactical_sheaf_curvature"
  name: "Tactical Sheaf Curvature Network"
  short_name: "TSCN"
  task: "binary chess puzzle-likeness classification from board tensors"
  central_claim: "typed sheaf frustration on chess-geometric candidate relations captures puzzle-likeness better than generic image or square-token processing"
  allowed_inputs:
    - "board tensor X with shape [B, C, 8, 8]"
    - "fixed board-coordinate geometry"
  forbidden_inputs:
    - "Stockfish scores"
    - "principal variations"
    - "node counts"
    - "engine best moves"
    - "verification metadata"
    - "source labels as features"
    - "proposed labels as features"
  binary_mapping:
    fine_0: 0
    fine_1: 1
    fine_2: 1
  primary_encoding: "simple_18"
  secondary_encodings:
    - "lc0_static_112"
    - "lc0_bt4_112"
  smallest_falsifier: "parameter-matched typed edge message passing on the same relation complex"
```

```yaml
config_yaml:
  experiment_name: "tscn_simple18_crtk_sample_3class"
  data:
    train_split: "data/splits/crtk_sample_3class/split_train.parquet"
    val_split: "data/splits/crtk_sample_3class/split_val.parquet"
    test_split: "data/splits/crtk_sample_3class/split_test.parquet"
    encoding: "simple_18"
    binary_target_from_fine_label:
      "0": 0
      "1": 1
      "2": 1
  model:
    name: "tactical_sheaf_curvature"
    input_channels: 18
    num_classes: 2
    d_node: 64
    d_stalk: 48
    d_geom: 24
    num_layers: 3
    gate_hidden: 64
    classifier_hidden: 128
    relation_distance_buckets: [1, 2, 3, 4]
    include_ray_edges: true
    include_knight_edges: true
    include_king_edges: true
    include_pawn_candidate_edges: true
    tie_file_mirror_relations: true
    tie_north_south_relations: false
    eta_init: 0.25
    gate_dropout: 0.05
    node_dropout: 0.10
    stats_pooling: ["mean", "std", "max"]
  training:
    loss: "cross_entropy"
    class_weights: "train_binary_inverse_frequency_or_existing_baseline_policy"
    optimizer: "adamw"
    learning_rate: 0.0003
    weight_decay: 0.0001
    batch_size: 256
    epochs: "match_current_baseline_budget"
    seeds: [1, 2, 3]
    mixed_precision: "allowed_if_used_consistently"
  regularization:
    lambda_energy: 0.0001
    lambda_gate: 0.0001
    lambda_gate_entropy_floor: 0.00001
    entropy_floor: 0.05
  evaluation:
    metrics:
      - "accuracy"
      - "macro_f1"
      - "roc_auc"
      - "pr_auc"
      - "fine_label_to_binary_prediction_table"
      - "fine_label_1_recall_at_fixed_fine_label_0_fpr"
```

```yaml
model_spec:
  class_name: "TacticalSheafCurvatureNet"
  file: "src/chess_nn_playground/models/trunk/tactical_sheaf_curvature.py"
  input_shape: ["B", "C", 8, 8]
  output_shape: ["B", "num_classes"]
  buffers:
    edge_index: [2, "E"]
    edge_type: ["E"]
    edge_geom: ["E", "g"]
    relation_group: ["E"]
  modules:
    BoardChannelAdapter:
      operation: "1x1 convolution from C to d_node plus normalization and activation"
    TypedRelationComplex:
      operation: "precompute directed chess-geometric candidate relations over 64 squares"
    SheafRestrictionGenerator:
      operation: "map relation geometry embeddings to bounded diagonal source/destination restrictions"
    SheafGate:
      operation: "input-dependent sigmoid edge gate from source stalk, destination stalk, and relation embedding"
    TacticalSheafLayer:
      operation: "weighted sheaf coboundary, Laplacian-like residual update, and edge-energy computation"
    CurvatureStatsPool:
      operation: "incoming weighted-variance curvature and pooled relation-group statistics"
    TSCNClassifierHead:
      operation: "MLP from node pools plus sheaf statistics to logits"
  default_num_layers: 3
  default_edge_count_range: [2000, 2600]
  parameter_target_range: [100000, 170000]
  central_ablation: "replace TacticalSheafLayer with parameter-matched typed gated message passing"
```

```yaml
research_continuity:
  idea_fingerprint: null
  closest_duplicate_risk: null
  do_not_repeat_if_this_fails: []
  suggested_next_search_directions: []
```

Suggested filled values for Codex after experiments:

```yaml
research_continuity_suggested_after_run:
  idea_fingerprint: "fixed chess-geometric relation complex + learned diagonal sheaf restrictions + sparse gates + sheaf frustration/curvature pooling"
  closest_duplicate_risk: "plain GNN-on-squares, graph chess representation, neural sheaf diffusion without chess-specific relation typing"
  do_not_repeat_if_this_fails:
    - "Do not retry sheaf curvature by only increasing d_stalk, num_layers, or gate_hidden."
    - "Do not rebrand typed message passing as a sheaf model if the coboundary/frustration ablation matches it."
    - "Do not add engine-derived tactical labels to rescue the idea."
  suggested_next_search_directions:
    - "engine-free move-conditioned contrastive perturbations"
    - "differentiable one-ply legal-move surrogate without evaluation labels"
    - "causal invariance across encodings and source splits"
    - "optimal-transport comparison between attacker and defender mass around king zones"
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add an anti-duplicate rule: do not propose another typed chess graph/sheaf model unless it changes the falsifiable operator, not only the relation list. | Prevents future cycles from repeating TSCN with cosmetic edge changes. | `Hard Constraints` or a new `Anti-Duplicate Rules` subsection. |
| Require every future idea to name its smallest central falsifier. | Forces a clean distinction between architecture novelty and ordinary capacity gains. | `Required Markdown File Content`, especially sections 9 and 10. |
| Require explicit statement of which chess symmetries are not assumed. | Avoids invalid full D4 equivariance claims for chess because pawn direction, color, side-to-move, and castling matter. | `Mathematical Thesis`. |
| Require a fallback encoding plan that does not depend on unknown channel semantics. | Makes handoffs more robust across `simple_18`, `lc0_static_112`, and `lc0_bt4_112`. | `Architecture Specification`. |
| Add failure-mode guidance: if a geometry-specific model fails against its parameter-matched generic ablation, future prompts should move toward move-conditioned or causal ideas rather than larger geometry models. | Converts negative results into a productive next search direction. | `Benchmark And Falsification Criteria` and `research_continuity`. |
| Preserve and restate the no-engine/no-verification-metadata leakage rule in every generated handoff. | Leakage would make benchmark gains meaningless. | `Problem Restatement And Data Contract`. |

## 14. Final Sanity Check

- Downloadable Markdown file created: Yes.
- Filename follows required date/time/day/timezone/slug pattern: Yes.
- No forbidden engine features used as inputs: Yes.
- Does not fabricate labels: Yes.
- Not a routine CNN/ResNet/Transformer variant: Yes.
- Minimal current-data experiment exists: Yes.
- Falsification criterion is concrete: Yes.
- Codex can implement without asking for missing architecture details: Yes.
- Prompt maintenance notes included for Codex: Yes.
