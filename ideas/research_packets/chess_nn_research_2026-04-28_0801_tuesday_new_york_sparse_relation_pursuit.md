# Codex Handoff Packet: Sparse Relation Pursuit Asymmetry

## 1. File Metadata

- **Filename:** `chess_nn_research_2026-04-28_0801_tuesday_new_york_sparse_relation_pursuit.md`
- **Generated:** 2026-04-28 08:01 new_york
- **Idea name:** Sparse Relation Pursuit Asymmetry, abbreviated **SRPA**
- **Task:** Chess puzzle classifier from board tensor to one puzzle logit.
- **Target mapping:** fine labels `0` and `1` map to binary target `0`; fine label `2` maps to binary target `1`.
- **Primary diagnostic:** near-puzzle false-positive rate, computed on the fine-label-`1` validation/test slice.
- **Selected sparse mechanism:** dual-branch group-sparse dictionary learning over deterministic chess relation tokens, with learned unrolled pursuit and reconstruction-residual asymmetry.
- **Inputs deliberately excluded:** engine scores, principal variations, node counts, mate scores, best moves, verification metadata, source labels, and source provenance.

## 2. Executive Selection

Build **SRPA**, a neural classifier whose only path to the output logit is through a sparse coding module.

The model first converts the board tensor into square embeddings, then into a fixed set of relation-edge tokens. These relation tokens are not image patches and are not selected witnesses. They are deterministic ordered square-pair relations: ray relations, knight-offset relations, king-adjacent relations, and pawn-diagonal relations. Each relation token is encoded by a small neural relation encoder. A pair of learned dictionaries then reconstructs every relation token through an unrolled group-sparse pursuit operator:

- `D_bg`: background / non-puzzle relation dictionary.
- `D_tac`: tactical / puzzle relation dictionary.

The classifier receives only sparse-code statistics and pursuit residuals from these dictionaries. It does not receive a direct dense board embedding bypass. The intended signal is not “which few pieces prove the puzzle.” The intended signal is whether the full relation field of the position is compressible by a small number of coherent tactical relation atoms, while near-puzzles remain either high-residual or code-dense.

This is the selected mechanism because it is central, measurable, and falsifiable. If sparse pursuit matters, dense-code and atom-shuffle controls should damage near-puzzle filtering without merely changing parameter count.

## 3. Data Contract

### Batch input

Expected training batch keys:

```python
batch = {
    "board": FloatTensor[B, C, 8, 8],
    "fine_label": LongTensor[B],  # values in {0, 1, 2}
}
```

Derived target:

```python
y = (batch["fine_label"] == 2).float()
near_mask = (batch["fine_label"] == 1)  # metric slice only by default
```

The first implementation should use `fine_label` only for target derivation and validation slicing. Do not add a special near-puzzle training loss in the first pass. That keeps the near-puzzle false-positive metric honest.

### Forbidden batch keys

Fail closed if any forbidden field appears in the batch, dataloader sample, collate output, cached feature dictionary, metadata object, wandb table, or debug artifact:

```python
FORBIDDEN_KEYS = {
    "engine_score", "engine_scores", "score_cp", "cp", "centipawn",
    "pv", "principal_variation", "node_count", "nodes", "mate_score",
    "mate_in", "best_move", "best_moves", "move_label", "solution_move",
    "verification", "verification_metadata", "proof", "proof_core",
    "source", "source_label", "source_labels", "source_id", "source_name",
    "source_provenance", "provenance", "dataset_origin",
}
```

### Board tensor assumptions

- The model accepts `board: [B, C, 8, 8]` and should not assume a hard-coded channel count beyond reading `C` from config.
- If the dataset already includes side-to-move orientation, leave it unchanged.
- If side-to-move is present as a channel but boards are not oriented, a deterministic orientation transform may be used. It must use only board channels, not move labels or engine data.
- No legal-move list, best-move target, engine analysis, or puzzle-source metadata may enter the model.

### Metrics

Report all metrics on validation and test:

```text
binary_auc
binary_average_precision
accuracy_at_0_5
fpr_at_threshold_0_5
near_puzzle_fpr_at_threshold_0_5
near_puzzle_fpr_at_recall_0_80
near_puzzle_fpr_at_recall_0_90
recall_at_threshold_0_5
expected_calibration_error
```

The primary model-selection metric is `near_puzzle_fpr_at_recall_0_80`, with ordinary binary AUC and AP treated as secondary.

## 4. Sparse Coding Research Map

SRPA should be implemented as a sparse-coding model, not as a generic classifier with a sparsity regularizer tacked on.

Relevant grounding:

1. **Dictionary learning / sparse coding:** Mairal, Bach, Ponce, and Sapiro frame sparse coding as representing data vectors by sparse linear combinations of learned basis elements, and propose scalable online dictionary learning. This supports the idea of learning dictionaries over relation-token vectors rather than hard-coding chess motifs. Source: [Online Dictionary Learning for Sparse Coding](https://icml.cc/2009/papers/364.pdf).
2. **Learned pursuit / LISTA:** Gregor and LeCun show that sparse-code inference can be approximated by a trainable fixed-depth feed-forward architecture, making sparse pursuit differentiable and usable inside recognition systems. SRPA uses this idea, but keeps the decoder dictionary explicit so residuals remain meaningful. Source: [Learning Fast Approximations of Sparse Coding](https://icml.cc/2010/papers/449.pdf).
3. **Group sparsity:** Yuan and Lin’s group-lasso formulation motivates selecting groups of related variables rather than independent scalar coefficients. SRPA applies this to relation-atom groups, so the model can prefer a small number of coherent relation families instead of isolated atom spikes. Source: [Model Selection and Estimation in Regression with Grouped Variables](https://academic.oup.com/jrsssb/article-abstract/68/1/49/7110631).
4. **Sparse modeling in vision:** Mairal, Bach, and Ponce survey sparse models for visual representations and reconstruction. SRPA borrows the reconstruction-and-representation discipline but rejects generic patch dictionaries because the chess board is symbolic and relational. Source: [Sparse Modeling for Image and Vision Processing](https://www.nowpublishers.com/article/Details/CGV-058).

Translation to this chess problem:

- Use a dictionary over **relation tokens**, not board patches.
- Use **continuous sparse codes**, not VQ assignments.
- Use **all deterministic relation edges**, not top-k witnesses.
- Use **pursuit residual trajectories** as the diagnostic object, not a proof verifier.
- Evaluate whether sparse reconstruction asymmetry separates true puzzles from near-puzzles.

## 5. Candidate Mechanisms Rejected

### Rejected: sparse proof-core verifier

A sparse proof-core verifier would select a small set of pieces, moves, or proof witnesses and verify a tactical line. This is explicitly out of scope. It would also risk recreating forbidden best-move or verification metadata through architecture pressure.

### Rejected: top-k witness-piece bottleneck

Selecting top-k squares or pieces is too close to witness extraction. It can hide leakage and encourages the model to explain positions by a small visible subset rather than by the sparse structure of the full relation field.

### Rejected: VQ motif codebook

A vector-quantized motif dictionary would assign relation or board states to discrete motif IDs. That creates a motif-codebook classifier, not sparse coding. SRPA uses continuous coefficients with L1 and group-lasso shrinkage.

### Rejected: generic patch dictionary

A dictionary over 2x2, 3x3, or convolutional board patches is inappropriate. Chess tactics are often nonlocal and line-based. Patch dictionaries also tend to learn material and local occupancy texture rather than tactical relation structure.

### Rejected: single-dictionary final-residual classifier

A single dictionary with only final reconstruction error is weak against near-puzzles. Near-puzzles may be reconstructible by the same broad dictionary. SRPA instead uses two equal-capacity dictionaries and the full residual trajectory under sparse pursuit.

### Rejected: prototype-margin classifier

Do not classify by distance to learned class prototypes or by a prototype margin. SRPA dictionaries are reconstruction bases with continuous sparse coefficients; the classifier uses residuals, code densities, and group-energy statistics, not nearest-prototype margins.

## 6. Common Approaches Rejected

The following are allowed only as external baselines, not as the selected architecture:

- Plain CNN, ResNet, ConvNeXt, ViT, or MLP classifier with direct dense board embedding to logit.
- Engine-supervised classifier using score, mate, PV, node count, best move, or legality-verification trace.
- Move-prediction pretraining where the puzzle logit can inherit best-move information.
- Source-provenance classifier that learns which dataset or generator produced the position.
- Masked-board reconstruction codec. Reconstructing masked squares can become a board autoencoder and violates the requested masked-board-codec exclusion.
- Hard motif library, VQ codebook, nearest prototype, or prototype-margin head.
- Top-k attention over squares or pieces interpreted as selected tactical witnesses.

A dense CNN baseline is still useful for comparison, but it must be separate from SRPA and must not be part of the SRPA forward path.

## 7. Mathematical Thesis

Let `x` be the board tensor. The model constructs deterministic relation tokens:

```text
R(x) = {r_e(x) in R^d : e in fixed relation edge set E}
```

where each edge `e = (i, j, tau)` connects source square `i`, target square `j`, and relation type `tau`. The relation set is fixed before training and does not select witnesses.

For each branch `b in {bg, tac}`, learn a dictionary:

```text
D_b in R^{K x d}, with K = G * A
```

where `G` is the number of atom groups and `A` is atoms per group. For each relation token `r_e`, infer a sparse code:

```text
a_{b,e}* = argmin_a 0.5 ||r_e - a D_b||_2^2
           + lambda_1 ||a||_1
           + lambda_g sum_{g=1}^G w_g ||a_g||_2
```

SRPA approximates this with `T` unrolled pursuit steps. Define branch residuals:

```text
E_b^{(t)}(x) = mean_e ||r_e(x) - a_{b,e}^{(t)} D_b||_2^2 / d
```

and group energies:

```text
G_{b,g}(x) = mean_e ||a_{b,e,g}^{(T)}||_2
```

The classifier is constrained to use only sparse outputs:

```text
phi(x) = concat(
    log(E_bg^{(1:T)} + eps),
    log(E_tac^{(1:T)} + eps),
    log(E_bg^{(1:T)} + eps) - log(E_tac^{(1:T)} + eps),
    G_bg(x),
    G_tac(x),
    active_group_counts_bg(x),
    active_group_counts_tac(x),
    code_entropy_bg(x),
    code_entropy_tac(x)
)

logit(x) = MLP_sparse(phi(x))
```

No dense board representation is concatenated into `phi`.

Thesis:

> True puzzle positions have relation fields that become low-residual under the tactical dictionary with a small number of coherent active relation-atom groups. Near-puzzles often share superficial relation features, but they either fail to reduce tactical residual under sparse pursuit or require a denser, higher-entropy code. Therefore sparse reconstruction asymmetry should reduce near-puzzle false positives better than a dense relation classifier with the same board encoder.

## 8. Sparse Coding Operator

Use an unrolled group-ISTA / LISTA-style pursuit module with an explicit dictionary decoder.

### Shapes

```text
r       : FloatTensor[B, E, d]
D       : FloatTensor[K, d]
a       : FloatTensor[B, E, K]
groups  : LongTensor[K] with values 0..G-1
T       : pursuit steps, default 6
```

### Operator

For each branch independently:

```python
def group_sparse_pursuit(r, D, group_slices, steps=6):
    # r: [B, E, d]
    # D: [K, d]
    a = zeros([B, E, K], device=r.device, dtype=r.dtype)
    residuals = []
    codes = []

    for t in range(steps):
        rec = einsum("bek,kd->bed", a, D)
        err = rec - r
        grad = einsum("bed,kd->bek", err, D)

        step = softplus(raw_step[t]) + 1e-6
        v = a - step * grad

        # scalar L1 shrinkage
        theta1 = softplus(raw_l1[t]).view(1, 1, K)
        v = sign(v) * relu(abs(v) - step * theta1)

        # group shrinkage
        out = zeros_like(v)
        for g, sl in enumerate(group_slices):
            vg = v[:, :, sl]
            norm = sqrt((vg * vg).sum(dim=-1, keepdim=True) + 1e-8)
            theta_g = softplus(raw_group[t, g])
            shrink = relu(1.0 - step * theta_g / norm)
            out[:, :, sl] = shrink * vg
        a = out

        rec = einsum("bek,kd->bed", a, D)
        residuals.append(((r - rec) ** 2).mean(dim=-1))  # [B, E]
        codes.append(a)

    return a, residuals, codes
```

### Required properties

- The decoder remains explicit: `rec = a @ D`.
- Thresholds and step sizes may be learned, but the reconstruction residual must be computed from the explicit dictionary.
- No hard top-k selection.
- No straight-through VQ assignment.
- No direct dense skip connection from board encoder to logit.
- Codes should be sparse because of the operator, not merely because of a small hidden dimension.

### Residual features

From each branch collect:

```text
final_residual_mean               [B, 1]
final_residual_std                [B, 1]
residual_mean_by_step             [B, T]
residual_drop_by_step             [B, T-1]
active_atom_fraction              [B, 1]
active_group_fraction             [B, 1]
group_energy_vector               [B, G]
group_entropy                     [B, 1]
top_group_energy_values_no_index  [B, 4]
```

`top_group_energy_values_no_index` may sort group energies by value for diagnostics, but must not expose group IDs as selected witnesses. It is optional; the safer first implementation can omit it.

## 9. Architecture And Tensor Shapes

Default configuration:

```yaml
input_channels: infer_from_batch
square_dim: 96
relation_dim: 96
geom_dim: 16
path_dim: 16
num_atom_groups: 24
atoms_per_group: 8
num_atoms: 192
pursuit_steps: 6
classifier_hidden: 128
```

### Fixed relation edge set

Create a fixed edge table once:

```text
edge_src        : LongTensor[E]
edge_dst        : LongTensor[E]
edge_type       : LongTensor[E]
edge_direction  : LongTensor[E]
edge_distance   : LongTensor[E]
```

Recommended relation families:

- Queen-like rays: rank, file, diagonal, anti-diagonal, all distances 1..7.
- Knight offsets.
- King-adjacent offsets.
- Pawn-diagonal offsets for both board directions.

On a standard 8x8 board this gives roughly `E = 2400` directed typed edges. The exact count depends on whether duplicate geometric relations are kept as distinct typed edges. Keeping duplicates is acceptable because relation type is part of the token.

### Forward pass

| Stage | Operation | Shape |
|---|---:|---:|
| Input | board tensor | `[B, C, 8, 8]` |
| Stem | `1x1 conv -> GELU -> 3x3 conv blocks` | `[B, 96, 8, 8]` |
| Flatten squares | reshape | `[B, 64, 96]` |
| Gather endpoints | `h_src = h[:, edge_src]`, `h_dst = h[:, edge_dst]` | `[B, E, 96]` each |
| Geometry embedding | type, direction, distance embeddings | `[B, E, 16]` |
| Path summary | deterministic line pooling for ray edges; zero for non-ray | `[B, E, 16]` |
| Relation encoder | MLP over `[src, dst, src*dst, abs(src-dst), geom, path]` | `[B, E, 96]` |
| LayerNorm | normalize relation tokens | `[B, E, 96]` |
| BG sparse pursuit | `D_bg` group-sparse pursuit | codes `[B, E, 192]`, residuals `T * [B, E]` |
| TAC sparse pursuit | `D_tac` group-sparse pursuit | codes `[B, E, 192]`, residuals `T * [B, E]` |
| Sparse descriptor | residual and code stats only | approx `[B, 2*(T + G + 6) + T]` |
| Head | MLP sparse descriptor to logit | `[B, 1]` |

### Path summary rule

For ray edges, summarize intermediate square embeddings between source and destination:

```text
path_mean = mean square embedding on open segment between endpoints
path_max  = max projection over same segment
```

Then project to `path_dim`. For adjacent, knight, and pawn-diagonal edges, use zeros. This is deterministic from the board tensor and fixed edge geometry. It is not a legal-move generator and does not use best moves.

### Classifier head

The head must be small and sparse-output-only:

```python
head = nn.Sequential(
    nn.LayerNorm(sparse_descriptor_dim),
    nn.Linear(sparse_descriptor_dim, 128),
    nn.GELU(),
    nn.Dropout(0.10),
    nn.Linear(128, 1),
)
```

Do not concatenate square embeddings, pooled board embeddings, attention states, material counts, or source features into the head.

## 10. Loss Function

Let:

```text
y in {0, 1}
logit = main SRPA logit
E_bg = final mean residual under D_bg
E_tac = final mean residual under D_tac
A_bg, A_tac = final sparse codes
```

Auxiliary residual-asymmetry logit:

```text
aux_logit = log(E_bg + eps) - log(E_tac + eps)
```

If `aux_logit` is positive, tactical reconstruction is better than background reconstruction.

Recommended loss:

```text
L = BCEWithLogits(logit, y)
  + 0.15 * BCEWithLogits(aux_logit, y)
  + 0.02 * mean(y * E_tac + (1 - y) * E_bg)
  + 0.001 * mean_abs_code(A_bg, A_tac)
  + 0.001 * mean_group_norm(A_bg, A_tac)
  + 0.0005 * L_dictionary_coherence
  + 0.0005 * L_branch_separation
  + 0.0001 * L_dead_group
```

Definitions:

```text
mean_abs_code = mean(|A_bg|) + mean(|A_tac|)
mean_group_norm = mean_g ||A_bg_g||_2 + mean_g ||A_tac_g||_2
L_dictionary_coherence = sum_b ||offdiag(normalize(D_b) normalize(D_b)^T)||_F^2
L_branch_separation = ||normalize(D_bg) normalize(D_tac)^T||_F^2
L_dead_group = mean_g relu(min_usage - usage_g)
```

Notes:

- `fine_label == 1` must not receive a special loss term in the first version. Use it for the diagnostic slice.
- The BCE head is the main supervised objective. The auxiliary residual-asymmetry objective ensures the sparse mechanism is directly trained and auditable.
- Reconstruction loss is class-conditional: non-puzzles should be reconstructible by the background dictionary; puzzles should be reconstructible by the tactical dictionary.
- Do not add contrastive prototype loss, nearest-prototype loss, or VQ commitment loss.

## 11. Dictionary Constraints

Apply these constraints after each optimizer step or through loss penalties.

### Unit atom norm

```python
D.data = D.data / (D.data.norm(dim=-1, keepdim=True) + 1e-8)
```

Do this for both `D_bg` and `D_tac`.

### Equal branch capacity

Keep `D_bg` and `D_tac` exactly equal in atom count, group count, relation dimension, threshold parameterization, and pursuit steps. Otherwise residual asymmetry can become a capacity artifact.

### Incoherence penalty

Penalize high off-diagonal cosine similarity within each dictionary:

```text
L_coh(D) = ||offdiag(D_norm D_norm^T)||_F^2
```

This discourages multiple atoms from learning the same relation direction and makes atom-shuffle tests more meaningful.

### Cross-branch separation

Penalize high similarity between background and tactical atoms:

```text
L_sep = ||D_bg_norm D_tac_norm^T||_F^2
```

This prevents both dictionaries from becoming interchangeable general reconstructions.

### Group utilization floor

Track exponential-moving-average group usage:

```text
usage_g = EMA(mean_{B,E} 1[||a_{e,g}||_2 > epsilon])
```

Penalize dead groups lightly. Do not force uniform usage; true sparse coding should use some groups more than others.

### No generic patches

The dictionary domain is relation-token space only. There should be no dictionary over raw board patches, convolutional patches, masked squares, or square neighborhoods.

### No witness bottleneck

The model may compute per-edge residuals for diagnostics, but training and inference must not select top-k edges, top-k pieces, or top-k squares as an architectural bottleneck.

## 12. Ablations

Run these ablations with the same data split, optimizer budget, board encoder size, and threshold-selection protocol.

### A0: Dense board baseline

Plain relation encoder plus mean/max pooled relation features into an MLP. This is a sanity baseline only. It should not be presented as the selected mechanism.

### A1: SRPA full model

Dual dictionaries, group-sparse pursuit, residual trajectory features, group-energy features, no dense bypass.

### A2: Dense-code control

Replace shrinkage with dense ridge-style pursuit:

```text
lambda_1 = 0
lambda_g = 0
no group shrinkage
same D shape
same pursuit steps
same classifier descriptor dimensions where possible
```

Expected result: ordinary AUC may remain competitive, but near-puzzle FPR should worsen if sparsity is doing real work.

### A3: Atom-shuffle posthoc control

After training SRPA, freeze the classifier and randomly permute atom rows within and across groups before evaluation. Re-run pursuit using the shuffled dictionaries. Do not permute classifier weights to compensate.

Report:

```text
near_puzzle_fpr_delta = shuffled_near_fpr - original_near_fpr
auc_delta = shuffled_auc - original_auc
active_group_entropy_delta
```

If atom shuffle barely changes near-puzzle FPR, the model is probably not using stable sparse atom structure.

### A4: Train-time atom-shuffle control

During training, randomly permute atom-to-group assignment before group-stat pooling on each batch while keeping reconstruction mathematically valid. This preserves much of the reconstruction path but destroys stable group semantics.

Expected result: worse near-puzzle FPR than full SRPA.

### A5: Single-dictionary control

Use one dictionary only. Feed final residual, residual trajectory, sparse code stats, and active group stats into the same head.

Expected result: weaker near-puzzle separation because no background-vs-tactical reconstruction asymmetry exists.

### A6: L1-only control

Remove group shrinkage but keep scalar L1 shrinkage.

Expected result: more scattered atom usage and weaker near-puzzle filtering.

### A7: Group-only control

Remove scalar L1 shrinkage but keep group shrinkage.

Expected result: active groups may be sparse, but within-group coefficients can become dense. This tests whether scalar sparsity matters.

### A8: Random frozen dictionary

Initialize dictionaries randomly, normalize atoms, and freeze them. Train only encoder, pursuit thresholds, and head.

Expected result: worse than learned dictionary. If not, dictionary learning is not central.

### A9: No residual trajectory

Use only final residuals and final code stats.

Expected result: worse near-puzzle FPR if the speed and pattern of pursuit convergence are informative.

### A10: Relation-token removal

Replace relation tokens with square tokens and apply the same sparse machinery over 64 square embeddings.

Expected result: weaker performance. If square-token sparse coding matches relation-token SRPA, the relation dictionary is not adding value.

## 13. Falsification

SRPA should be rejected or revised if any of the following happens.

### Mechanism falsifiers

- Dense-code control matches full SRPA on `near_puzzle_fpr_at_recall_0_80` within one standard error across at least three seeds.
- Atom-shuffle posthoc control does not materially degrade near-puzzle FPR.
- Random frozen dictionary is statistically indistinguishable from learned dictionary.
- The classifier remains strong when residual trajectory features are removed and only dense-like aggregate features remain.
- Active atom fraction is high, for example above `0.35`, on most samples. That means the module is not meaningfully sparse.
- `D_bg` and `D_tac` become highly similar despite cross-branch separation, for example average max cross-cosine above `0.80`.

### Data leakage falsifiers

- Any forbidden key appears in a dataloader sample, model input, cached feature, training log table, or evaluation script.
- Performance changes sharply when source/provenance fields are removed from debug data structures, implying accidental use.
- The model can predict the label from material counts or trivial side-to-move statistics at nearly the same near-puzzle FPR.

### Chess-specific falsifiers

- Near-puzzle false positives concentrate in one source, generator, date range, or board-format artifact rather than chess structure.
- Per-edge residual maps mainly highlight occupancy density or material imbalance instead of relation geometry.
- Positive predictions are dominated by a single relation type, such as only king adjacency, with no meaningful dictionary diversity.

### Minimum acceptance bar

For a first successful pass:

```text
SRPA near_puzzle_fpr_at_recall_0_80 <= dense_baseline near_puzzle_fpr_at_recall_0_80 * 0.85
SRPA near_puzzle_fpr_at_recall_0_80 <= dense_code_control near_puzzle_fpr_at_recall_0_80 * 0.90
atom_shuffle_posthoc increases near_puzzle_fpr_at_recall_0_80 by at least 10 percent relative
binary_auc remains within 3 percent of the best dense baseline
```

These thresholds are not sacred, but they force the sparse mechanism to earn its complexity.

## 14. Codex Implementation Notes

### Suggested files for Codex to create

```text
src/models/relation_edges.py
src/models/relation_encoder.py
src/models/group_sparse_pursuit.py
src/models/srpa_classifier.py
src/training/losses_srpa.py
src/metrics/near_puzzle.py
tests/test_forbidden_inputs.py
tests/test_srpa_shapes.py
tests/test_atom_shuffle_control.py
```

This handoff packet is the only requested artifact here; the list above is an implementation plan for Codex, not files created by this response.

### Module contracts

`relation_edges.py`

```python
class FixedChessRelationEdges(nn.Module):
    def __init__(self, include_duplicates: bool = True): ...
    def forward(self, device=None):
        return {
            "src": LongTensor[E],
            "dst": LongTensor[E],
            "type": LongTensor[E],
            "direction": LongTensor[E],
            "distance": LongTensor[E],
            "path_indices": LongTensor[E, max_path_len],
            "path_mask": BoolTensor[E, max_path_len],
        }
```

`group_sparse_pursuit.py`

```python
class GroupSparsePursuit(nn.Module):
    def __init__(self, num_atoms, dim, num_groups, steps): ...
    def forward(self, r: Tensor) -> dict:
        return {
            "codes": Tensor[B, E, K],
            "residual_by_step": Tensor[B, steps, E],
            "final_residual": Tensor[B, E],
            "group_energy": Tensor[B, G],
            "active_atom_fraction": Tensor[B],
            "active_group_fraction": Tensor[B],
            "group_entropy": Tensor[B],
        }
```

`srpa_classifier.py`

```python
class SRPAClassifier(nn.Module):
    def __init__(self, input_channels: int, cfg: SRPAConfig): ...
    def forward(self, board: Tensor) -> dict:
        return {
            "logit": Tensor[B],
            "aux_logit": Tensor[B],
            "diag": {
                "bg_final_residual": Tensor[B],
                "tac_final_residual": Tensor[B],
                "bg_group_energy": Tensor[B, G],
                "tac_group_energy": Tensor[B, G],
                "bg_active_atom_fraction": Tensor[B],
                "tac_active_atom_fraction": Tensor[B],
            }
        }
```

### Shape tests

Minimum tests:

```python
def test_srpa_forward_shapes():
    model = SRPAClassifier(input_channels=20, cfg=small_cfg())
    board = torch.randn(4, 20, 8, 8)
    out = model(board)
    assert out["logit"].shape == (4,)
    assert out["aux_logit"].shape == (4,)
    assert out["diag"]["bg_group_energy"].shape[0] == 4


def test_no_dense_bypass_to_head():
    # The head input dimension should equal sparse descriptor dimension only.
    assert model.head[1].in_features == model.sparse_descriptor_dim
```

### Forbidden-input test

```python
def assert_no_forbidden_keys(batch):
    bad = FORBIDDEN_KEYS.intersection(set(batch.keys()))
    if bad:
        raise ValueError(f"Forbidden keys present: {sorted(bad)}")
```

Add recursive checking for nested dictionaries.

### Training defaults

```yaml
optimizer: adamw
learning_rate: 0.0003
weight_decay: 0.01
batch_size: tune_to_memory
precision: bf16_or_fp16_with_fp32_dictionary_norms
pursuit_steps: 6
num_atom_groups: 24
atoms_per_group: 8
relation_dim: 96
warmup_steps: 1000
max_grad_norm: 1.0
class_balance: weighted_sampler_or_bce_pos_weight
threshold_selection: choose_threshold_on_validation_for_recall_0_80
seeds: [1, 2, 3]
```

### Memory notes

The largest tensor is usually `codes: [B, E, K]`. With `E ≈ 2400` and `K = 192`, this can be large. Implement chunked pursuit over relation edges:

```python
for edge_chunk in chunks(E, chunk_size=256 or 512):
    run pursuit on r[:, edge_chunk]
    accumulate residual and group statistics
```

Do not chunk by selecting important edges. Chunking is purely for memory and must cover all fixed edges.

### Logging

Log these every validation epoch:

```text
near_puzzle_fpr_at_recall_0_80
near_puzzle_fpr_at_recall_0_90
binary_auc
binary_average_precision
mean_bg_residual_by_label
mean_tac_residual_by_label
residual_asymmetry_by_label
active_atom_fraction_by_label
active_group_fraction_by_label
group_entropy_by_label
max_cross_branch_atom_cosine
atom_dead_group_count
```

### Controls implementation flags

```text
--model srpa
--control none
--control dense_code
--control atom_shuffle_posthoc
--control atom_shuffle_train
--control single_dictionary
--control l1_only
--control group_only
--control frozen_random_dictionary
--control no_residual_trajectory
--control square_tokens_only
```

### Practical implementation warning

The easiest way to accidentally violate the research goal is to add a dense pooled board embedding to improve AUC. Do not do that in SRPA. If a dense bypass is desired for comparison, create a separate model class and label it as a baseline.

## 15. Future Prompt Updates

Recommended future prompt refinements:

- Specify the exact board-channel schema, including side-to-move, castling, en-passant, repetition, and orientation conventions.
- Specify whether deterministic attack maps are allowed. This packet avoids explicit attack-map inputs and uses relation geometry from the board tensor only.
- Specify the exact threshold protocol for near-puzzle FPR, for example `near_fpr_at_recall_0.80` versus fixed threshold `0.5`.
- Require three-seed reporting for dense-code and atom-shuffle controls.
- Require an explicit no-dense-bypass unit test.
- Require memory budget, target GPU, and maximum acceptable edge count.
- Add a material-only logistic baseline to detect trivial leakage.
- Keep the exclusions: no sparse proof-core verifier, no top-k witness-piece bottleneck, no VQ motif codebook, no prototype-margin classifier, no masked-board codec, and no generic patch dictionary.
