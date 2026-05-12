# Codex Handoff Packet: Attack-Defense Sheaf Energy Network

## 1. File Metadata

- Filename: chess_nn_research_2026-04-21_0255_tuesday_local_attack_defense_sheaf.md
- Generated at: 2026-04-21 02:55:03 UTC-07:00
- Weekday: Tuesday
- Timezone: local
- Idea slug: attack_defense_sheaf
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Attack-Defense Sheaf Energy Network
- One-sentence thesis: Puzzle-likeness in a chess position is better modeled as localized inconsistency in attack-defense constraints over pseudo-legal chess incidences than as another deeper square-image classifier.
- Idea fingerprint: `dynamic_typed_chess_incidence_sheaf__ray_blocker_gates__restriction_energy_readout__binary_puzzle_likeness`
- Why this is not a common CNN/ResNet/Transformer variant: The core computation is not convolution over adjacent pixels or token self-attention over 64 squares; it builds a fixed typed chess-incidence complex, learns sheaf restriction maps on directed move primitives, diffuses by a gated sheaf Laplacian, and reads out attack-defense tension energy.
- Current-data minimal experiment: Train `AttackDefenseSheafNet` on `simple_18` using the existing train/val/test split, three seeds, binary labels `fine_label == 0 -> 0` and `fine_label in {1,2} -> 1`; then repeat once on `lc0_static_112` only if the simple run beats the strongest non-ensemble baseline on validation AUROC or balanced accuracy.
- Expected information gain if it fails: A clean failure would show that explicit pseudo-legal attack/defense incidence and sheaf energy do not add measurable signal beyond existing CNN/LC0-style baselines on this data, ruling out a broad family of hand-structured tactical-constraint models before trying richer search surrogates.

## 3. Problem Restatement And Data Contract

Task: binary chess puzzle-likeness classification from a single board-position tensor.

Binary outputs:

- `0`: non-puzzle
- `1`: puzzle-like

Available fine labels in the dataset:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

Training target for the current benchmark:

- `y_binary = 0` when `fine_label == 0`
- `y_binary = 1` when `fine_label in {1, 2}`

Evaluation must preserve the existing report format:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Allowed neural-network inputs:

- Encoded board-position tensor only.
- Supported encodings: `simple_18`, `lc0_static_112`, `lc0_bt4_112`.
- Deterministic coordinate features generated inside the model, such as rank/file coordinates, edge distance, and precomputed chess move-incidence indices, are allowed because they contain no engine evaluation or label information.

Forbidden neural-network inputs:

- Stockfish scores.
- Principal variations.
- Engine node counts.
- Engine verification metadata.
- Source labels.
- Proposed labels.
- Any signal derived from unresolved candidate verification.
- Any feature that directly or indirectly encodes whether a position was sampled from the puzzle, near-puzzle, non-puzzle, or candidate-generation pipeline.

Tensor contract:

- Model type: PyTorch `nn.Module`.
- Input: `(batch, C, 8, 8)`.
- Output: logits `(batch, num_classes)`, with `num_classes = 2` for the current benchmark.
- The model must support `C = 18` and `C = 112` through a configurable `input_channels` field.

Benchmark split:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

Leakage checklist for Codex:

- Do not parse or feed Stockfish fields, PV fields, verification metadata, source labels, proposed labels, or unresolved-candidate status into the model.
- Do not construct extra class `1` or class `2` examples.
- Do not turn unresolved candidates into positives or negatives.
- Do not use fine labels as inputs; use them only to derive benchmark labels and to produce per-fine-label reports.
- Do not tune on test results.
- Do not let filename, row order, split source, puzzle ID, or duplicate metadata enter the model.
- Do not use legal-engine move generation that requires search/evaluation. This proposal uses only static chess geometry and learned gates from the input tensor.

## 4. Research Map

This idea borrows mathematical operators and inductive biases, not ready-made chess code.

| Source or idea | URL | Borrowed | Not copied |
|---|---|---|---|
| Sheaf Neural Networks, Hansen and Gebhart | https://arxiv.org/abs/2012.06333 | Cellular sheaf viewpoint: vector spaces on graph cells, restriction maps, sheaf Laplacian-style diffusion. | No use of their datasets, code, or exact architecture. |
| Neural Sheaf Diffusion, Bodnar et al. | https://arxiv.org/abs/2202.04579 | Learnable nontrivial sheaf maps as a way to handle heterophily and avoid trivial smoothing. | No claim that chess positions satisfy their benchmark assumptions; no copied hyperparameters. |
| Simplicial Neural Networks, Ebli, Defferrard, Spreemann | https://arxiv.org/abs/2010.03633 | Higher-order interactions beyond pairwise edges motivate target-square convergence cells. | No full simplicial convolution stack; only a lightweight edge-to-target readout. |
| Neural Message Passing for Quantum Chemistry, Gilmer et al. | https://proceedings.mlr.press/v70/gilmer17a.html | MPNN formalism as the closest implementation cousin and ablation baseline. | The central proposal is not plain learned message passing; messages are gradients of typed sheaf residuals. |
| Gauge Equivariant Convolutional Networks, Cohen et al. | https://arxiv.org/abs/1902.04615 | Local-coordinate/gauge caution: equivariance should respect relation frames, not assume one global board symmetry. | No manifold CNN, no spherical/icosahedral convolution, no full group convolution. |
| FIDE Laws of Chess, official handbook chapter | https://handbook.fide.com/chapter/e012023 | Reminder that chess has directional pawns, castling, promotion, and side-to-move asymmetries, so full rotation/reflection invariance is false. | No rules engine and no legal-move oracle. |
| Chess tactical structure as domain knowledge | unverified general chess knowledge | Pins, skewers, discovered attacks, overloaded defenders, and converging attackers can be viewed as inconsistent local constraints. | No engine tactics labels, no move annotations, no puzzle solution supervision. |

Unverifiable citations: the last row is marked unverified because it is general chess-tactics intuition rather than a single formal paper.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN over board planes | simple CNN | Already covered; local square kernels weakly encode long diagonal/file/rank tactics only through many layers. |
| Residual CNN | residual CNN | Already covered; residual depth improves optimization but does not explicitly model attack-defense incidences. |
| Small/medium/deep CNN scaling | small/medium/deep CNN variants | Ordinary width/depth scaling is disallowed and low-information for a research cycle. |
| LC0-style CNN | LC0 BT4-style CNN | Strong chess prior but still primarily convolutional; proposal should not be another LC0 clone. |
| LC0-style residual CNN | LC0 residual variants | Likely nearest high-capacity baseline; adding residual blocks would not test a new mathematical hypothesis. |
| Ordinary ViT over 64 square tokens | no direct baseline, closest transformer variant | Vanilla self-attention ignores typed chess geometry unless it relearns it from data; also disallowed as a core idea. |
| Plain GNN on squares | no direct baseline | A square graph with generic MPNN messages is too close to standard graph learning and lacks typed restriction consistency. It is retained only as an ablation. |
| Hyperparameter tuning | all baselines | Disallowed; optimizer, LR, schedule, and batch-size tweaks are not a research idea. |
| Ensembling several existing models | all baselines | Disallowed and would obscure whether the proposed inductive bias works. |
| More data or relabeling candidates | data pipeline | Disallowed as the core idea and risks label-fabrication errors. |
| Engine-evaluation distillation | none allowed | Forbidden leakage: Stockfish scores, PVs, node counts, and verification metadata cannot enter as inputs or auxiliary labels. |
| Full D4-equivariant board CNN | possible equivariant CNN baseline | Chess is not invariant under all rotations/reflections because pawns, promotion, castling, and side-to-move are directional; full D4 tying would be mathematically wrong. |

## 6. Mathematical Thesis

Input space:

Let `X_C = R^{C x 8 x 8}` be the encoded board tensor space for one supported encoding. The model receives `x in X_C` and produces binary logits `f_theta(x) in R^2`.

Target definition:

Let `Y in {0,1}` be the benchmark target, where known non-puzzles are `0` and verified near-puzzles plus verified puzzles are `1`. Fine labels remain available only for reporting.

Distribution assumptions:

- The train, validation, and test parquet files define the empirical distribution for this project.
- Puzzle-like positions are assumed to be enriched for forcing tactical structure: checks, threats, pins, skewers, overloaded defenders, discovered attacks, and target-square convergence.
- Non-puzzles may still contain attacks, material imbalance, checks, or tactical-looking motifs. Therefore attack count alone is not sufficient.

Symmetry/equivariance assumptions:

- Chess board geometry has typed automorphisms only after respecting piece direction, side-to-move, castling semantics, promotion direction, and channel conventions.
- Full rotation/reflection equivariance is false.
- A safe weak symmetry is left-right file reflection when relation types are reflected consistently and any encoding-specific castling/channel conventions are handled by the data encoder or not tied by the model.
- This proposal does not hard-enforce global board invariance. It ties only local relation types that are known to be geometrically equivalent under file reflection: east/west, northeast/northwest, southeast/southwest, and the corresponding knight offsets, unless `tie_file_reflection = false`.

Core hypothesis:

Puzzle-likeness is correlated with high, structured sheaf residual energy on a dynamic typed chess-incidence complex. A puzzle-like position often contains local constraints that cannot be made mutually consistent: one piece is pinned while also needed as a defender, a king-adjacent target is attacked through a ray, multiple attackers converge on a target, or a blocker simultaneously shields and is overloaded. A sheaf residual directly parameterizes this inconsistency.

Formal object:

Precompute a directed typed incidence set `E` over the 64 squares.

Edge families:

- Ray edges: rook/bishop/queen directions `N, S, E, W, NE, NW, SE, SW` for all source-target pairs on the same ray.
- Knight edges: the eight L-shaped offsets.
- King-adjacent edges: the eight one-step offsets.
- Pawn-attack edges: white-up diagonals and black-down diagonals as separate typed relations.
- Optional quiet-pawn pressure edges are off by default for the minimal experiment.

For each edge `e = (u, v, tau)`:

- `u, v in {0,...,63}` are squares.
- `tau` is a relation type.
- `M_e` is the list of intervening squares for ray edges and empty otherwise.

The model maps input squares to node stalks:

```text
h_v^0 = phi_theta(x[:, :, rank(v), file(v)], coord(v)) in R^d
```

It learns type-conditioned restriction maps:

```text
R_src^tau: R^d -> R^r
R_dst^tau: R^d -> R^r
```

It learns an occupancy proxy:

```text
o_v = sigmoid(w_occ^T h_v)
```

Ray visibility gate:

```text
q_e = product_{m in M_e} (1 - o_m + eps)
```

Non-ray visibility gate:

```text
q_e = 1
```

Learned tactical gate:

```text
a_e = sigmoid(g_theta([h_u, h_v, type_embed(tau), q_e]))
```

Total gate:

```text
gamma_e = q_e * a_e
```

Sheaf residual:

```text
c_e = sqrt(gamma_e) * (R_src^tau h_u - R_dst^tau h_v) in R^r
```

Sheaf energy:

```text
E_sheaf(h) = sum_{e in E} ||c_e||_2^2
```

Diffusion block:

```text
h^{l+1} = LayerNorm(
    h^l
    - alpha_l * B_R^T Gamma B_R h^l
    + FFN_l(h^l)
)
```

where `B_R` is the typed sheaf coboundary operator assembled from `R_src^tau` and `R_dst^tau`, and `Gamma` is diagonal with edge gates.

Proposition:

For any permutation `pi` of board squares that preserves edge incidence, edge types up to the model's tied type map, blocker lists, and coordinate features, the sheaf energy and diffusion operator are equivariant:

```text
E_sheaf(P_pi h) = E_sheaf(h)
D(P_pi h) = P_pi D(h)
```

when the tied restriction maps satisfy `R^{pi(tau)} = R^tau` under the induced type tying.

Proof sketch:

The coboundary matrix `B_R` has one typed source block and one typed destination block per directed edge. If `pi` preserves incidence and tied relation types, then applying `pi` to nodes only permutes the rows of edge residuals and the columns of node stalks. The blocker-product gates are also permuted because `M_{pi(e)} = pi(M_e)`. Therefore `B_R P_pi = P_E B_R`, `Gamma(P_pi h) = P_E Gamma(h) P_E^T`, and

```text
||Gamma^{1/2} B_R P_pi h||^2
= ||P_E Gamma^{1/2} B_R h||^2
= ||Gamma^{1/2} B_R h||^2.
```

The gradient-like diffusion term `B_R^T Gamma B_R h` transforms as `P_pi` times the original term. LayerNorm and pointwise FFNs preserve node permutations when shared across nodes, so the block is equivariant.

What is proven:

- The proposed typed sheaf energy and diffusion are equivariant to any symmetry actually respected by the typed incidence construction and parameter tying.
- The proof does not rely on chess being fully rotation/reflection invariant.

What is hypothesized:

- Verified near-puzzles and puzzles have more learnable, class-informative sheaf residual patterns than known non-puzzles.
- Dynamic ray-blocker gates can learn useful pseudo-line-of-sight from the board encoding without piece labels being explicitly parsed by hand.
- Energy readout helps separate fine label `1` near-puzzles from true non-puzzles better than a plain CNN.

Counterexamples:

- Quiet positional puzzles whose solution is prophylaxis, zugzwang, fortress logic, or long strategic maneuvering may have low immediate attack-defense energy.
- Non-puzzle positions with many legal threats, checks, or king attacks can have high sheaf energy.
- Encodings that hide or scramble occupancy in unexpected ways may make blocker gates noisy.
- If the existing dataset's puzzle-like labels are mostly determined by source pipeline artifacts rather than board tactics, a leakage-free tactical sheaf model may underperform.

## 7. Architecture Specification

Module name:

```text
AttackDefenseSheafNet
```

Recommended file:

```text
src/chess_nn_playground/models/attack_defense_sheaf.py
```

Top-level constructor fields:

```text
input_channels: int
num_classes: int = 2
d_model: int = 64
sheaf_rank: int = 16
num_blocks: int = 3
type_emb_dim: int = 16
dropout: float = 0.10
edge_dropout: float = 0.05
tie_file_reflection: bool = true
use_energy_aux_head: bool = true
use_convergence_readout: bool = true
max_ray_length: int = 7
```

Precomputed buffers:

```text
edge_src: LongTensor[num_edges]
edge_dst: LongTensor[num_edges]
edge_type: LongTensor[num_edges]
edge_is_ray: BoolTensor[num_edges]
blocker_index: LongTensor[num_edges, max_blockers]    # pad with -1
blocker_mask: BoolTensor[num_edges, max_blockers]
square_rank: FloatTensor[64]
square_file: FloatTensor[64]
square_coord_features: FloatTensor[64, coord_dim]
type_reflection_orbit: LongTensor[num_edge_types]     # used only if tying is enabled
```

Approximate edge counts:

- Ray edges: about `1456` directed edges.
- Knight edges: about `336` directed edges.
- King-adjacent edges: about `420` directed edges.
- Pawn-attack edges: about `196` directed edges.
- Total: about `2400` directed typed edges, small enough for dense per-batch gather/scatter.

Submodules:

```text
SquareAdapter:
    input:  x [B, C, 8, 8]
    append or concatenate coordinate planes internally
    output: h0 [B, 64, d_model]

OccupancyHead:
    input:  h [B, 64, d_model]
    output: occ [B, 64, 1]

EdgeGate:
    input:  h_src [B, E, d_model], h_dst [B, E, d_model],
            type_embedding [E, type_emb_dim], ray_visibility [B, E, 1]
    output: gamma [B, E, 1]

TypedRestriction:
    parameters:
        R_src[type, sheaf_rank, d_model]
        R_dst[type, sheaf_rank, d_model]
    output:
        residual c [B, E, sheaf_rank]

SheafDiffusionBlock:
    computes gated residuals
    scatters gradient-like messages back to src/dst nodes
    applies learned step size alpha_l
    adds node FFN and LayerNorm

ConvergenceReadout:
    edge residual features -> scatter to target square
    produces per-square incoming tension summaries

TensionReadout:
    pools node mean, node max, edge-energy mean by type group,
    soft top-k edge energy, and optional convergence summaries
    output logits [B, num_classes]
```

Forward-pass pseudocode, intentionally not full implementation:

```text
def forward(x):
    # x: [B, C, 8, 8]
    h = square_adapter(x)                       # [B, 64, d]
    all_edge_energy = []

    for block in sheaf_blocks:
        occ = occupancy_head(h)                 # [B, 64, 1]

        h_src = gather(h, edge_src)             # [B, E, d]
        h_dst = gather(h, edge_dst)             # [B, E, d]

        blocker_occ = gather_with_pad(occ, blocker_index, blocker_mask)
        ray_visibility = product(1 - blocker_occ + eps over blockers)
        ray_visibility = where(edge_is_ray, ray_visibility, 1)

        gamma = edge_gate(h_src, h_dst, edge_type, ray_visibility)
        c = typed_restriction(h_src, h_dst, edge_type, gamma)   # [B, E, r]
        edge_energy = sum(c * c, dim=-1)                       # [B, E]

        node_delta = sheaf_adjoint_scatter(c, edge_src, edge_dst, edge_type, gamma)
        h = layer_norm(h - softplus(alpha) * node_delta + node_ffn(h))

        all_edge_energy.append(edge_energy)

    pooled = tension_readout(h, all_edge_energy, edge_type, edge_dst)
    logits = classifier(pooled)                 # [B, 2]
    return logits
```

Important implementation details:

- `sheaf_adjoint_scatter` should use `index_add` or `scatter_add` over the 64 nodes.
- The typed maps can be stored per raw edge type. If `tie_file_reflection = true`, build a mapping so reflected types share the same parameter tensor or copy through a small parameter-orbit table.
- The model must not inspect labels, engine fields, or source metadata.
- The adapter should not require knowing exact semantic channel names. It can learn from the provided encoding channels.
- Coordinate features are deterministic and should be identical for every sample.

Parameter estimate for default `d_model=64`, `sheaf_rank=16`, `num_blocks=3`, `num_edge_types` about `30`:

- Square adapter: `~8k` parameters for `C=112`, less for `C=18`.
- Type embeddings and gate MLPs: `~30k-60k`.
- Restriction maps: `2 * num_edge_types * sheaf_rank * d_model`, about `60k` if `num_edge_types=30`; multiply by `num_blocks` if maps are not shared.
- Node FFNs: about `25k` per block for `64 -> 128 -> 64`.
- Classifier/readout: `~20k-50k`.
- Expected total: `0.25M-0.60M` parameters, depending on whether restriction maps are block-shared.

Complexity estimate:

- Per block per sample: approximately `O(E * d_model * sheaf_rank + 64 * d_model^2)`.
- With `E ≈ 2400`, `d_model=64`, `sheaf_rank=16`, this is roughly `5M` multiply-adds per block for restriction evaluation plus small scatter/readout overhead.
- Default three-block model should be substantially below a large residual CNN and comparable to a small structured model.

Encoding support:

- `simple_18`: primary minimal experiment.
- `lc0_static_112`: secondary experiment.
- `lc0_bt4_112`: optional scaling condition only after validation success on the first two.

Logits interface:

- `forward(x)` returns only logits by default.
- Optional debug mode may return a dictionary with `edge_energy`, `ray_visibility`, and `energy_aux_logit`, but training/evaluation code should consume logits exactly like other models.

## 8. Loss, Training, And Regularization

Primary loss:

```text
CrossEntropyLoss(logits, y_binary)
```

Class weighting:

- Use the same class-weighting convention as current baselines if they already have one.
- If no convention exists, compute weights from the train split only:
  - `weight_0 = N / (2 * N_0)`
  - `weight_1 = N / (2 * N_1)`
- Do not compute weights from validation or test.

Optional auxiliary loss:

```text
energy_aux_loss = BCEWithLogitsLoss(energy_aux_logit, y_binary)
total_loss = CE + lambda_energy * energy_aux_loss + lambda_gate * gate_sparsity
```

Defaults:

```text
lambda_energy: 0.10
lambda_gate: 0.001
```

Auxiliary-loss constraints:

- The auxiliary target is the same binary benchmark label, not a new fabricated tactical label.
- No fine-label-specific auxiliary target.
- No engine target.
- Gate sparsity is unsupervised and should only mildly discourage all edges from being open.

Batch size:

- Start with the repository's standard batch size for the comparable CNN baseline.
- If no standard exists, use `batch_size = 256` for `simple_18` and reduce only if memory requires it.

Optimizer and LR:

- Use the current fair-comparison optimizer if the repo has a standard.
- Otherwise default:
  - `optimizer = AdamW`
  - `lr = 3e-4`
  - `weight_decay = 1e-4`

Regularizers:

- Dropout inside node FFN: `0.10`.
- Edge dropout on `gamma` during training: `0.05`, applied after ray visibility and before residual computation.
- Gradient clipping: `max_norm = 1.0`.
- Early stopping by validation AUROC or balanced accuracy, whichever is already used in the project.

Determinism:

- Run at least three seeds.
- Save seed, git commit if available, config, parameter count, and metric JSON.
- Fix train/val/test split paths exactly.

What must stay fixed for fair comparison:

- Same splits.
- Same binary label mapping.
- Same input encoding when comparing to a baseline.
- Same maximum epochs and early-stopping patience as baseline if known.
- Same metric code and threshold-selection procedure.
- No test-set tuning.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Untyped MPNN same edges | Replace typed sheaf residual `R_src h_u - R_dst h_v` with generic edge MLP messages at matched parameter count. | The central claim that sheaf restriction inconsistency matters, not just chess-edge message passing. | If equal or better, abandon the sheaf-specific claim and treat chess incidence as the useful part. |
| Grid-only sheaf | Replace chess move-incidence edges with 8-neighbor board adjacency. | Long-range typed attack rays carry tactical signal. | If equal, the model may be acting as a local CNN surrogate. |
| No ray-blocker gates | Set `q_e = 1` for all ray edges. | Learned pseudo-line-of-sight is useful. | If equal, blocker gating is unnecessary or occupancy is not learned. |
| No typed relation tying | Give every direction/type separate parameters and disable file-reflection tying. | Weak safe geometric tying improves data efficiency. | If better, chess-channel conventions or castling/pawn asymmetry make tying harmful. |
| Energy readout removed | Pool only final node embeddings; discard edge-energy summaries. | Puzzle-likeness is visible in residual/tension energy, not just diffused node states. | If equal, energy is not the right statistic. |
| No convergence readout | Remove target-square scatter summaries from incoming edge residuals. | Multi-attacker/overloaded-defender convergence helps separate puzzles. | If equal, pairwise residuals are sufficient. |
| No auxiliary energy head | Train only CE on logits. | Auxiliary energy supervision stabilizes the intended signal. | If equal, keep the simpler loss. |
| One block only | Use `num_blocks=1`. | Multi-step tactical diffusion is needed beyond immediate incidences. | If equal, depth is not contributing and default should be simplified. |

Smallest ablation that can falsify the central claim:

- `Untyped MPNN same edges`.
- It keeps the same edge set, gates, parameter budget, optimizer, and readout. If it matches or beats the sheaf model across seeds, the sheaf-energy thesis fails even if chess-incidence edges remain useful.

## 10. Benchmark And Falsification Criteria

Baselines:

- Best existing `simple_18` simple CNN.
- Best existing `simple_18` residual CNN or depth/width variant.
- Best existing `lc0_static_112` CNN/residual baseline.
- Best existing LC0 BT4-style CNN/residual baseline when comparing on `lc0_bt4_112`.
- New internal ablation: `Untyped MPNN same edges`.

Metrics:

- Accuracy.
- Balanced accuracy.
- AUROC.
- AUPRC.
- F1 at the repository's standard threshold.
- MCC.
- Calibration error if already available.
- Per-fine-label confusion table:
  - true fine label `0` -> predicted binary `0/1`
  - true fine label `1` -> predicted binary `0/1`
  - true fine label `2` -> predicted binary `0/1`

Artifacts to save:

- Config YAML.
- Model parameter count.
- Training curves.
- Validation metrics per epoch.
- Test metrics for the selected checkpoint.
- Per-seed JSON metrics.
- Confusion table by fine label.
- Ablation metrics.
- Optional debug summary of average edge energy by relation type, computed only from model internals and labels used for evaluation.

Success threshold:

- On the test split, after selecting by validation only, the default model must beat the best comparable non-ensemble baseline on the same encoding by at least one of:
  - `+1.5` absolute AUROC points, or
  - `+2.0` absolute balanced-accuracy points, or
  - `+2.0` absolute MCC points.
- It must not increase the fine-label-`0` false-positive rate by more than `1.0` absolute point unless AUROC improves by at least `2.5` points.
- It must beat the `Untyped MPNN same edges` ablation by at least `0.8` AUROC points or `1.0` balanced-accuracy points on the validation mean across three seeds.

Failure threshold:

- Mean validation AUROC improvement over best comparable baseline is less than `0.3` points across three seeds.
- Or the `Untyped MPNN same edges` ablation matches or beats the default within noise.
- Or validation improves but test metrics regress on balanced accuracy, MCC, and fine-label confusion.

Abandon condition:

- Abandon this idea family if `simple_18` and `lc0_static_112` both fail the above criteria and the untyped MPNN ablation is not worse.
- Do not repeat this as "bigger sheaf", "more sheaf blocks", or "more relation types" unless the failed run shows underfitting with clear train/val evidence.

Scaling condition:

- Only try `lc0_bt4_112`, larger `d_model`, or more blocks if:
  - default `simple_18` beats its baseline on validation,
  - default `lc0_static_112` does not regress,
  - and sheaf-specific ablation is meaningfully worse.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/2026_04_21_attack_defense_sheaf/README.md` | Create | Copy this handoff packet or a concise implementation summary with the same leakage constraints. |
| `ideas/2026_04_21_attack_defense_sheaf/ablation_plan.md` | Create | Tables from sections 9 and 10 plus exact commands once repo command style is known. |
| `src/chess_nn_playground/models/attack_defense_sheaf.py` | Create | `AttackDefenseSheafNet`, incidence-buffer builder, `SquareAdapter`, `SheafDiffusionBlock`, `TensionReadout`. |
| `src/chess_nn_playground/models/registry.py` | Edit | Register model key `attack_defense_sheaf`. |
| `configs/attack_defense_sheaf_simple18.yaml` | Create | Minimal experiment config for `simple_18`, default model fields, existing split paths. |
| `configs/attack_defense_sheaf_lc0_static112.yaml` | Create | Secondary config changing only `encoding` and `input_channels`. |
| `tests/test_attack_defense_sheaf.py` | Create | Shape test, determinism smoke test, no-label-input test, edge-buffer bounds test, forward/backward test. |
| `tests/test_attack_defense_sheaf_incidence.py` | Create if tests are organized by model internals | Verify edge counts are nonzero, blocker lists are within `0..63`, no padded blocker contributes to visibility, reflection-orbit map is valid. |
| `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Preserve hard constraints while adding reusable lessons, anti-duplicate rules, clearer output requirements, and failure-mode guidance discovered from this research pass. |

Required prompt-maintenance behavior:

- After consuming this file, Codex must update `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md`.
- The update must not weaken leakage rules, label rules, falsification requirements, or anti-duplicate requirements.
- The update should add an anti-duplicate note: do not propose another dynamic chess-incidence sheaf/energy model if this run fails under the stated falsification criteria.

Focused tests Codex should implement:

```text
test_forward_shape:
    for C in [18, 112], x [2, C, 8, 8] -> logits [2, 2]

test_no_nan_backward:
    CE loss backward produces finite gradients

test_edge_buffers:
    edge_src, edge_dst in [0, 63]
    blocker_index is either -1 or in [0, 63]
    no edge has source == destination

test_ray_visibility_pad:
    padded blockers do not affect q_e

test_registry:
    model key "attack_defense_sheaf" instantiates from config

test_debug_outputs_optional:
    debug mode returns edge_energy [B, E] without changing default logits interface
```

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: "chess_nn_research_2026-04-21_0255_tuesday_local_attack_defense_sheaf.md"
  artifact_type: "codex_handoff_markdown"
  generated_at: "2026-04-21T02:55:03-07:00"
  weekday: "Tuesday"
  timezone: "local"
  idea_slug: "attack_defense_sheaf"
  intended_next_consumer: "Codex"
```

```yaml
idea_yaml:
  idea_id: "2026_04_21_attack_defense_sheaf"
  idea_name: "Attack-Defense Sheaf Energy Network"
  idea_slug: "attack_defense_sheaf"
  thesis: "Classify puzzle-likeness by learning gated sheaf residual energy over typed pseudo-legal chess attack-defense incidences."
  fingerprint: "dynamic_typed_chess_incidence_sheaf__ray_blocker_gates__restriction_energy_readout__binary_puzzle_likeness"
  task:
    input_shape: ["batch", "C", 8, 8]
    output_shape: ["batch", 2]
    binary_mapping:
      fine_0: 0
      fine_1: 1
      fine_2: 1
  allowed_encodings:
    - "simple_18"
    - "lc0_static_112"
    - "lc0_bt4_112"
  forbidden_inputs:
    - "Stockfish scores"
    - "PVs"
    - "node counts"
    - "verification metadata"
    - "source labels"
    - "proposed labels"
    - "unresolved candidate status"
  central_falsification_ablation: "Untyped MPNN same edges"
  success_threshold:
    test_auroc_absolute_gain_points: 1.5
    test_balanced_accuracy_absolute_gain_points: 2.0
    test_mcc_absolute_gain_points: 2.0
    sheaf_vs_untyped_val_auroc_gain_points: 0.8
  abandon_if:
    - "simple_18 and lc0_static_112 both fail"
    - "untyped MPNN same edges matches or beats default"
    - "improvement requires only larger depth/width"
```

```yaml
config_yaml:
  model:
    name: "attack_defense_sheaf"
    input_channels: 18
    num_classes: 2
    d_model: 64
    sheaf_rank: 16
    num_blocks: 3
    type_emb_dim: 16
    dropout: 0.10
    edge_dropout: 0.05
    tie_file_reflection: true
    use_energy_aux_head: true
    use_convergence_readout: true
    max_ray_length: 7
  data:
    encoding: "simple_18"
    train_split: "data/splits/crtk_sample_3class/split_train.parquet"
    val_split: "data/splits/crtk_sample_3class/split_val.parquet"
    test_split: "data/splits/crtk_sample_3class/split_test.parquet"
    target_mapping:
      "0": 0
      "1": 1
      "2": 1
  training:
    batch_size: 256
    optimizer: "AdamW"
    lr: 0.0003
    weight_decay: 0.0001
    gradient_clip_norm: 1.0
    class_weights: "train_split_balanced_or_repo_default"
    primary_loss: "cross_entropy"
    auxiliary_losses:
      energy_aux:
        enabled: true
        lambda: 0.10
        target: "same_binary_label"
      gate_sparsity:
        enabled: true
        lambda: 0.001
    seeds: [0, 1, 2]
    select_checkpoint_by: "validation_auroc_or_repo_default"
  evaluation:
    metrics:
      - "accuracy"
      - "balanced_accuracy"
      - "auroc"
      - "auprc"
      - "f1"
      - "mcc"
      - "per_fine_label_confusion"
    no_test_tuning: true
```

```yaml
model_spec:
  class_name: "AttackDefenseSheafNet"
  module_path: "src/chess_nn_playground/models/attack_defense_sheaf.py"
  forward:
    input: "x: FloatTensor[batch, C, 8, 8]"
    output: "logits: FloatTensor[batch, num_classes]"
  buffers:
    edge_src: "LongTensor[num_edges]"
    edge_dst: "LongTensor[num_edges]"
    edge_type: "LongTensor[num_edges]"
    edge_is_ray: "BoolTensor[num_edges]"
    blocker_index: "LongTensor[num_edges, max_blockers]"
    blocker_mask: "BoolTensor[num_edges, max_blockers]"
    square_coord_features: "FloatTensor[64, coord_dim]"
  components:
    - name: "SquareAdapter"
      input_shape: ["B", "C", 8, 8]
      output_shape: ["B", 64, "d_model"]
    - name: "OccupancyHead"
      input_shape: ["B", 64, "d_model"]
      output_shape: ["B", 64, 1]
    - name: "EdgeGate"
      input_shape: ["B", "E", "2*d_model + type_emb_dim + 1"]
      output_shape: ["B", "E", 1]
    - name: "TypedRestriction"
      residual_shape: ["B", "E", "sheaf_rank"]
    - name: "SheafDiffusionBlock"
      input_shape: ["B", 64, "d_model"]
      output_shape: ["B", 64, "d_model"]
    - name: "TensionReadout"
      output_shape: ["B", "readout_dim"]
    - name: "Classifier"
      output_shape: ["B", 2]
  default_complexity:
    num_edges_approx: 2400
    params_approx: "0.25M-0.60M"
    multiply_adds_per_sample_approx: "15M-20M for 3 blocks"
```

```yaml
research_continuity:
  idea_fingerprint: null
  closest_duplicate_risk: null
  do_not_repeat_if_this_fails: []
  suggested_next_search_directions: []
  filled_by_this_packet:
    idea_fingerprint: "dynamic typed chess-incidence sheaf with ray-blocker gates and residual-energy readout"
    closest_duplicate_risk: "plain square GNN or another sheaf/Laplacian model over pseudo-legal chess moves"
    do_not_repeat_if_this_fails:
      - "Do not retry by merely increasing d_model, sheaf_rank, or num_blocks."
      - "Do not propose another dynamic attack-defense sheaf unless the failure analysis shows the incidence builder was implemented incorrectly."
      - "Do not replace the sheaf with a standard GNN and call it a new topology idea."
    suggested_next_search_directions:
      - "Causal invariance across encodings: force representation agreement between simple_18 and lc0_static_112 without labels beyond the benchmark target."
      - "Differentiable one-ply proof-number surrogate without engine evaluation, using legal move masks only if already encoded or safely generated without search."
      - "Optimal-transport alignment between attacker and defender mass over target squares, with no engine scores."
      - "Information bottleneck that penalizes source-pipeline artifacts while preserving board-tactical signal."
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add an anti-duplicate line: "If `Attack-Defense Sheaf Energy Network` fails its falsification criteria, do not propose another typed chess-incidence sheaf, sheaf Laplacian, or residual-energy model as the next core idea." | Prevents recycling the same topology idea with minor parameter changes. | `Common Approaches Rejected` or a new `Do Not Repeat Failed Ideas` subsection. |
| Add a requirement that future ideas name their closest falsifying ablation in the executive section. | Forces each research pass to expose what would disprove the core mechanism, not just compare to baselines. | `Required Markdown File Content -> 2. Executive Selection`. |
| Add a reminder that chess is not D4-invariant; any equivariance claim must state which relation types and metadata preserve it. | Avoids mathematically false symmetry assumptions. | `Hard Constraints` or `Mathematical Thesis` instructions. |
| Add a leakage note: static pseudo-legal geometry is allowed, but engine search, engine legality evaluation, PVs, scores, and verification metadata remain forbidden. | Clarifies the boundary between safe chess geometry and engine leakage. | `Project Context` or `Hard Constraints`. |
| Add a machine-readable `central_falsification_ablation` field to future `idea_yaml` blocks. | Makes Codex benchmarking scripts easier to connect to the scientific claim. | `Machine-Readable Blocks`. |
| Add guidance that a model may use deterministic coordinate features and precomputed incidence buffers if they are sample-independent and label-independent. | Reduces unnecessary ambiguity for non-CNN geometric models. | `Problem Restatement And Data Contract`. |

## 14. Final Sanity Check

- Downloadable Markdown file created: Yes
- Filename follows required date/time/day/timezone/slug pattern: Yes
- No forbidden engine features used as inputs: Yes
- Does not fabricate labels: Yes
- Not a routine CNN/ResNet/Transformer variant: Yes
- Minimal current-data experiment exists: Yes
- Falsification criterion is concrete: Yes
- Codex can implement without asking for missing architecture details: Yes
- Prompt maintenance notes included for Codex: Yes
