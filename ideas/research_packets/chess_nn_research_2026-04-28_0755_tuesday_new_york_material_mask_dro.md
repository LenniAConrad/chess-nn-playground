# Codex Handoff Packet: Material-Locked Tactical Mask DRO

## 1. File Metadata

- **Filename:** `chess_nn_research_2026-04-28_0755_tuesday_new_york_material_mask_dro.md`
- **Created:** 2026-04-28 07:55:52, new_york
- **Short slug:** `material_mask_dro`
- **Task:** Distributionally robust chess puzzle classifier
- **Selected idea:** Material-Locked Tactical Mask Distributionally Robust Optimization, abbreviated **MLTM-DRO**
- **Data allowed at inference:** current board tensor only
- **Output:** one scalar logit per board
- **Binary target mapping:** fine label `0 -> 0`, fine label `1 -> 0`, fine label `2 -> 1`
- **Forbidden inference inputs:** engine data, principal variations, node counts, mate scores, best moves, puzzle verification metadata, source labels, source files
- **Primary diagnostic:** 3x2 matrix with rows as fine labels `0, 1, 2` and columns as predicted negative/positive

Reference anchors used for research framing:

- Robust optimization should define an uncertainty set, not merely add a regularizer: [Ben-Tal, El Ghaoui, and Nemirovski, *Robust Optimization*](https://cris.technion.ac.il/en/publications/robust-optimization-2).
- DRO can be built around ambiguity sets over distributions close to empirical data: [Esfahani and Kuhn, *Data-driven distributionally robust optimization using the Wasserstein metric*](https://link.springer.com/article/10.1007/s10107-017-1172-1).
- Neural group DRO shows why worst-case objectives need regularization in overparameterized networks, but group/source IDs are not selected for this task: [Sagawa et al., *Distributionally Robust Neural Networks*](https://openreview.net/forum?id=ryxGuJrFvS).
- Latent subpopulation DRO motivates robustness to unseen subpopulation shifts without relying on known group IDs: [Duchi, Hashimoto, and Namkoong, *Distributionally Robust Losses for Latent Covariate Mixtures*](https://pubsonline.informs.org/doi/pdf/10.1287/opre.2022.2363).
- Board-game neural systems can consume rule-derived board state rather than engine analysis: [Silver et al., *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*](https://pubmed.ncbi.nlm.nih.gov/30523106/).

## 2. Executive Selection

Build a single-logit chess puzzle classifier trained with a **chess-specific DRO uncertainty set** over rule-derived tactical masks, not over source IDs or generic feature noise.

The selected method is **Material-Locked Tactical Mask DRO**:

1. The model receives the current board tensor.
2. A deterministic chess-rule feature builder computes relation masks from that same board tensor: pseudo-legal attacks, defenses, king-ring pressure, pins, x-rays, slider rays, hanging-piece flags, and coarse mobility.
3. During training only, an inner adversary contaminates those derived tactical masks within an explicit chess-specific budget.
4. The adversary is **material-locked**: it cannot change piece occupancy, piece identity, side to move, castling state, en-passant state, or any source/verification field.
5. The outer optimizer minimizes the worst-case binary classification loss under this mask contamination.
6. Near-puzzle negatives, fine label `1`, receive an asymmetric false-positive penalty because they are the target failure mode.

This is not augmentation-only. The training objective is a minimax loss over a conditional uncertainty set attached to each board. The perturbations are not new labels or synthetic boards; they are worst-case uncertainty over tactical nuisance fields that a brittle classifier may over-trust.

This is not generic DRO. The ambiguity set is chess-specific and defined by material preservation, pseudo-legal attack geometry, king rays, slider lines, pinned-piece candidates, and king-ring squares.

This is not source-rate calibration. Source labels/files are never model inputs, never calibration inputs, and never part of the loss.

## 3. Data Contract

### Required batch fields

```python
batch = {
    "board": FloatTensor[B, C_board, 8, 8],
    "fine_label": LongTensor[B],  # values in {0, 1, 2}
}
```

### Target mapping

```python
target = (fine_label == 2).float()
```

- Fine label `0`: ordinary negative, target `0`
- Fine label `1`: near-puzzle negative, target `0`
- Fine label `2`: true puzzle, target `1`

### Inference input contract

At inference, the callable must be:

```python
logit = model(board)
```

where:

- `board` is the current board tensor.
- `logit` has shape `[B]` or `[B, 1]`.
- No engine analysis or verification metadata may be used.
- No source ID, source path, source file, source family, source rate, source-calibrated threshold, or data-origin hint may be used.
- Rule-derived masks may be computed inside `model(board)` because they are deterministic functions of the current board tensor.

### Allowed deterministic board-derived fields

These are allowed only if computed from the board tensor:

- Piece occupancy planes
- Side-to-move plane
- Castling-right planes, if already present in the board tensor
- En-passant plane, if already present in the board tensor
- Pseudo-legal attack masks
- Defense masks
- King-ring masks
- Slider rays
- Pin and x-ray candidate masks
- Coarse legal/pseudo-legal mobility counts
- Material vector and material signature derived from piece planes

No search, engine evaluation, best move, mate score, PV, node count, or puzzle-verification field is allowed.

### Optional metadata handling

If the dataset already contains source IDs, source files, verification fields, engine fields, or move-solution fields, Codex must treat them as **tainted metadata**:

```python
TAINTED_KEYS = {
    "source", "source_id", "source_file", "dataset_name", "collection",
    "engine", "eval", "cp", "mate", "pv", "best_move", "solution",
    "node_count", "nodes", "depth", "verification", "is_verified",
}
```

Rules:

- Do not include tainted keys in model input.
- Do not include tainted keys in feature building.
- Do not include tainted keys in thresholding or calibration.
- Do not include tainted keys in the robust objective.
- They may be used only outside training as split-audit metadata if the existing evaluation protocol requires reporting held-out distribution shifts.

### Required 3x2 diagnostic

For every validation/test run, produce this count matrix:

| fine label | predicted negative | predicted positive |
|---:|---:|---:|
| 0 | `diag[0, 0]` | `diag[0, 1]` |
| 1 | `diag[1, 0]` | `diag[1, 1]` |
| 2 | `diag[2, 0]` | `diag[2, 1]` |

Prediction rule:

```python
pred_positive = (logit >= threshold)
```

Default threshold is `0.0`, corresponding to probability `0.5`. If another threshold is reported, it must be global, fixed on validation data, and not source-conditioned.

Key rates:

```python
ordinary_fp_rate = diag[0, 1] / max(1, diag[0, 0] + diag[0, 1])
near_puzzle_fp_rate = diag[1, 1] / max(1, diag[1, 0] + diag[1, 1])
puzzle_fn_rate = diag[2, 0] / max(1, diag[2, 0] + diag[2, 1])
puzzle_recall = diag[2, 1] / max(1, diag[2, 0] + diag[2, 1])
```

The main success target is lower `near_puzzle_fp_rate` under shifted evaluation while keeping `puzzle_recall` competitive.

## 4. DRO Research Map

### Why DRO is appropriate here

The failure pattern is not simply class imbalance. A near-puzzle can contain tactical-looking ingredients such as a king under pressure, a pinned defender, an overloaded piece, or a hanging piece, while still lacking a true puzzle tactic. Under distribution shift, these ingredients can appear at different rates. A standard empirical-risk classifier may learn a shortcut: “tactical smell implies puzzle.” That shortcut creates false positives on fine label `1`.

DRO is appropriate because the classifier should minimize loss not only on the observed tactical-mask distribution but also on plausible chess-specific corruptions of the board-derived tactical fields. The robust model should still classify the board correctly when some attack, pin, x-ray, king-ring, hanging, or mobility features are unreliable.

### Why not standard group DRO

Group DRO over source IDs is rejected for this task. It would be too close to source-rate calibration and would not satisfy the inference constraint. It also fails the chess-specific uncertainty-set requirement: a source group is a data-origin object, not a chess object.

The useful lesson from neural group DRO is not “use source groups.” The useful lesson is that a worst-case objective in an overparameterized neural net should be paired with regularization, early stopping, and explicit evaluation of the worst failure mode.

### Why not generic Wasserstein/KL DRO

A generic ball over raw board tensors is under-specified and can create illegal or label-changing boards. Moving a piece one square, changing side to move, or toggling castling can change the existence of a tactic. That would violate the label-preserving assumption.

The selected ambiguity set instead keeps the board fixed and perturbs derived tactical relation masks. This makes the uncertainty set label-preserving by construction: the chess position is unchanged, only the model’s access to brittle auxiliary tactical summaries is uncertain.

### Selected family

Use **conditional chess-feature DRO**:

\[
\min_{\theta}\; \frac{1}{n}\sum_{i=1}^n
\max_{\delta_i \in \mathcal U_{\rho}(x_i)}
c(\ell_i, y_i)\,
\mathcal L\left(f_\theta(x_i, r(x_i) + \delta_i), y_i\right)
+ \lambda\lVert\theta\rVert_2^2
\]

where:

- \(x_i\) is the current board tensor.
- \(\ell_i \in \{0,1,2\}\) is the fine label.
- \(y_i = \mathbb 1[\ell_i=2]\).
- \(r(x_i)\) is the deterministic chess-rule relation tensor.
- \(\delta_i\) is a bounded contamination of tactical masks.
- \(\mathcal U_{\rho}(x_i)\) is the material-locked tactical uncertainty set.
- \(c(\ell_i,y_i)\) is an asymmetric cost, with extra false-positive pressure on fine label `1`.

## 5. Candidate Search Trace

### Candidate A: Raw board adversarial perturbation

**Concept:** Add adversarial perturbations directly to board planes.

**Decision:** Rejected.

**Reason:** Raw board perturbations can create non-chess positions, alter material, change side to move, or blur discrete piece identity. That breaks the label-preserving assumption and becomes generic adversarial training, not chess-specific DRO.

### Candidate B: Material-preserving board mutations

**Concept:** Move pieces while preserving the material vector.

**Decision:** Rejected as the primary method.

**Reason:** Even with equal material, relocating one defender, moving a king, or changing a slider line can create or destroy a tactic. This can turn a negative into a positive or vice versa. It is too label-unstable for the main robust objective.

### Candidate C: Relation-mask uncertainty

**Concept:** Keep the board fixed. Compute rule-derived tactical relation masks. Let the adversary contaminate only those tactical masks within chess-geometric supports.

**Decision:** Selected.

**Reason:** It is chess-specific, material-preserving, label-preserving, and aimed directly at near-puzzle false positives. A near-puzzle is often close to a true puzzle in tactical surface features. The model must not collapse when some of those surface features are unreliable.

### Candidate D: Board-feature contamination balls

**Concept:** Keep immutable board planes fixed, but allow bounded contamination of derived feature channels such as king danger, hanging piece flags, and mobility.

**Decision:** Selected as a secondary component inside MLTM-DRO.

**Reason:** Some implementations may not use explicit 64x64 relation biases. For CNN-first architectures, derived 8x8 tactical planes are easier to add. The same material-lock principle applies.

### Candidate E: Source-aware group DRO

**Concept:** Make source IDs or source families the DRO groups.

**Decision:** Rejected.

**Reason:** It violates the spirit of the task. It risks source-rate calibration and does not define a chess-specific uncertainty set. Source IDs must not be inference inputs and should not shape the robust loss.

### Candidate F: Phase-specialist classifier or phase calibrator

**Concept:** Train/calibrate separate heads for opening, middlegame, endgame, or material phase.

**Decision:** Rejected.

**Reason:** Phase-specialist calibration is explicitly disallowed. A phase feature may be computed from board material for analysis, but not used as a separate calibrator or specialist routing rule.

### Candidate G: Ensemble of robust classifiers

**Concept:** Train several classifiers and combine logits.

**Decision:** Rejected.

**Reason:** Ensembling is explicitly disallowed. The required output is one logit from one model.

## 6. Rejected Approaches

The following are out of scope and must not be implemented as the selected solution.

| Approach | Rejection reason |
|---|---|
| Nuisance projection | The method must not learn a projection that removes tactical nuisance subspaces. Robustness is obtained by worst-case tactical-mask contamination, not by projecting representations. |
| Rule-partition invariance | Do not enforce invariance across hand-made rule partitions. Chess-rule masks define uncertainty supports, not invariant equivalence classes. |
| Source-rate calibration | No source-conditioned thresholds, priors, offsets, or logit corrections. |
| Phase-specialist calibration | No separate phase-specific calibrators or routed specialist heads. |
| Ensembling | One model, one logit. |
| Data augmentation only | Augmentation alone is not enough. The chosen training loss has an inner worst-case maximization over a chess-specific uncertainty set. |
| Generic DRO without chess-specific uncertainty | Rejected. The ambiguity set must preserve board material and constrain contamination to tactical relation fields. |
| Engine-assisted classifier | Forbidden by inference contract: no engine evals, PVs, node counts, mate scores, best moves, or verification metadata. |
| Source-ID group DRO | Rejected because it uses data-origin information rather than chess uncertainty. |
| Best-move contrastive training | Rejected because best moves/solutions are forbidden inference-adjacent metadata and can leak puzzle verification. |

## 7. Mathematical Thesis

Near-puzzle false positives arise when the learned decision rule overweights unstable tactical surface indicators. Formally, let the board tensor be \(x\), the true binary label be \(y\), and the fine label be \(\ell\). Let \(r(x)\) be deterministic chess-rule tactical relations derived from the board. A non-robust classifier learns:

\[
f_\theta(x, r(x)) \approx \log \frac{P(y=1 \mid x, r(x))}{P(y=0 \mid x, r(x))}
\]

If training sources overrepresent true puzzles among positions with high king pressure, pins, or hanging pieces, then \(r(x)\) becomes a shortcut. Under a shift where near-puzzles contain the same ingredients but no decisive tactic, the shortcut produces false positives.

MLTM-DRO trains against a family of conditional distributions:

\[
\mathcal P_\rho
=
\left\{
P: P(x,y,\ell,\delta)
=
\widehat P(x,y,\ell) q(\delta\mid x),
\quad
q(\cdot\mid x) \text{ supported on } \mathcal U_\rho(x)
\right\}
\]

The robust risk is:

\[
R_\rho(\theta)
=
\sup_{P\in\mathcal P_\rho}
\mathbb E_P
\left[
c(\ell,y)\,
\mathcal L(f_\theta(x, r(x)+\delta), y)
\right]
\]

Because \(\mathcal U_\rho(x)\) never changes the board, material, side-to-move, or label, this risk targets **uncertainty in tactical summaries**, not uncertainty in the chess position.

The thesis is:

> A puzzle classifier that remains correct when bounded attack, defense, pin, x-ray, king-ring, hanging-piece, and mobility masks are adversarially contaminated will rely less on brittle tactical smell and more on stable board evidence. This should reduce false positives on near-puzzle negatives under distribution shift while preserving true-puzzle recall.

The loss should be asymmetric for fine label `1`:

\[
\mathcal L_{\text{asym}}(z,y,\ell)
=
y\,\text{softplus}(-z)
+
(1-y)\,\gamma_\ell\,\text{softplus}(z)
\]

with:

\[
\gamma_0 = 1.0, \quad \gamma_1 > 1.0, \quad \gamma_2 = 1.0
\]

Only negative examples with fine label `1` receive the higher false-positive penalty. This does not alter inference inputs; it uses the label taxonomy during training.

## 8. Uncertainty Set Definition

### Board decomposition

Let the current board tensor be decomposed as:

\[
x = [p, s]
\]

where:

- \(p \in \{0,1\}^{12\times8\times8}\) are piece-occupancy planes.
- \(s\) contains side-to-move and other legal state planes already present in the board tensor.

Define the material signature:

\[
m(x) =
\left(
\sum_{a,b} p_{\text{WP},a,b},
\sum_{a,b} p_{\text{WN},a,b},
\ldots,
\sum_{a,b} p_{\text{BK},a,b}
\right)
\in \mathbb N^{12}
\]

MLTM-DRO never perturbs \(p\), \(s\), or \(m(x)\).

### Deterministic tactical relation tensor

Compute \(r(x)\) from the board tensor using only chess rules. It has two parts.

#### Square-plane tactical channels

\[
u(x) \in [0,1]^{C_u\times8\times8}
\]

Recommended channels:

1. White attacks square
2. Black attacks square
3. Own side attacks square
4. Opponent attacks square
5. Own defended piece
6. Opponent defended piece
7. Own hanging piece candidate
8. Opponent hanging piece candidate
9. Own king ring
10. Opponent king ring
11. Own king-ring pressure
12. Opponent king-ring pressure
13. Own pinned-piece candidate
14. Opponent pinned-piece candidate
15. Own slider x-ray candidate
16. Opponent slider x-ray candidate
17. Own pseudo-legal mobility heat
18. Opponent pseudo-legal mobility heat

#### Relation-bias masks

\[
A(x) \in [0,1]^{H\times64\times64}
\]

Recommended relation heads:

1. Same-color defense edge
2. Opponent attack edge
3. Knight attack geometry
4. Pawn attack geometry
5. King adjacency geometry
6. Bishop/diagonal slider ray
7. Rook/file-rank slider ray
8. Queen slider ray
9. King-to-slider pin ray
10. X-ray through one blocker
11. Capture-target edge
12. Protected-target edge

The exact number of heads may be changed, but the implementation must keep the uncertainty supports chess-geometric.

### Chess-specific support masks

For each family \(F\), define a support \(S_F(x)\) that constrains where contamination may occur.

- `attack`: only from occupied piece squares to pseudo-legal target squares for that piece type.
- `defense`: only same-color attack relations.
- `pin`: only rays between a king and an enemy slider with one or two blockers.
- `xray`: only slider rays with at least one blocker.
- `king_ring`: only the eight-neighborhood around either king, clipped to the board.
- `hanging`: only currently occupied piece squares.
- `mobility`: only pseudo-legal destination squares.
- `board_feature`: only derived feature planes, never piece planes.

### Material-locked tactical uncertainty set

For a board \(x\), define:

\[
\mathcal U_\rho(x)
=
\left\{
\delta=(\delta_u,\delta_A):
\begin{array}{l}
\tilde u = \operatorname{clip}_{[0,1]}(u(x)+\delta_u) \\
\tilde A = \operatorname{clip}_{[0,1]}(A(x)+\delta_A) \\
\delta_u \odot (1-S_u(x)) = 0 \\
\delta_A \odot (1-S_A(x)) = 0 \\
\lVert \delta_{u,F} \rVert_1 \le \rho^u_F |S^u_F(x)| \quad \forall F \\
\lVert \delta_{A,F} \rVert_1 \le \rho^A_F |S^A_F(x)| \quad \forall F \\
\lVert \delta_{u,F} \rVert_\infty \le \epsilon^u_F \quad \forall F \\
\lVert \delta_{A,F} \rVert_\infty \le \epsilon^A_F \quad \forall F \\
p, s, m(x), y, \ell \text{ unchanged}
\end{array}
\right\}
\]

Recommended starting budgets:

```python
rho_u = {
    "attack": 0.04,
    "defense": 0.04,
    "king_ring": 0.08,
    "pin": 0.08,
    "xray": 0.08,
    "hanging": 0.06,
    "mobility": 0.04,
}
rho_A = {
    "attack": 0.03,
    "defense": 0.03,
    "slider": 0.05,
    "pin": 0.08,
    "xray": 0.08,
    "capture": 0.04,
    "protected": 0.04,
}
eps_u = 0.35
eps_A = 0.35
```

Interpretation:

- The adversary may blur, suppress, or exaggerate a small fraction of tactical evidence.
- The adversary may not invent arbitrary board structure.
- The adversary may not create material changes.
- The adversary may not change the actual board tensor.
- The adversary may not alter labels.

### Why this set is chess-specific

The support masks are not generic pixel masks. They are generated from chess geometry:

- Piece-specific attack movement
- Slider ray structure
- King-ring squares
- Pin candidate rays
- X-ray blockers
- Occupied piece squares
- Pseudo-legal destination squares

This directly matches the task’s allowed examples: material preservation, relation-mask uncertainty, and board-feature contamination balls.

## 9. Architecture Tensor Contract

### Recommended model

Use a single model with two streams:

1. **Board stream:** convolutional or square-token encoder over immutable board planes.
2. **Relation stream:** transformer-style square attention with additive relation biases from \(A(x)\), plus 8x8 tactical planes \(u(x)\).

Final head:

```python
logit = MLP(pool(square_tokens))
```

Output shape:

```python
logit: FloatTensor[B]
```

### Input shape

```python
board: FloatTensor[B, C_board, 8, 8]
```

`C_board` is whatever the existing repository uses for current board tensor encoding. Codex must not require engine fields.

### Derived feature builder API

```python
class ChessFeatureBuilder(nn.Module):
    def forward(self, board: torch.Tensor) -> ChessFeatures:
        # board: [B, C_board, 8, 8]
        # returns deterministic rule-derived features
        return ChessFeatures(
            tactical_planes=u,       # [B, C_u, 8, 8]
            relation_masks=A,        # [B, H, 64, 64]
            support_planes=Su,       # same shape as u or family-indexed
            support_relations=SA,    # same shape as A or family-indexed
            material=material_vec,   # [B, 12]
        )
```

The builder must be deterministic and side-effect free. It must not call a chess engine. It may use board-rule logic to derive attacks, pins, and rays.

### Model API

```python
class MaterialLockedTacticalDROClassifier(nn.Module):
    def __init__(self, board_channels: int, ...):
        ...

    def forward(
        self,
        board: torch.Tensor,
        features: Optional[ChessFeatures] = None,
        contamination: Optional[MaskContamination] = None,
    ) -> torch.Tensor:
        # returns logits [B]
        ...
```

Inference call:

```python
logits = model(board)
```

Training call:

```python
features = feature_builder(board)
delta = adversary.inner_max(model, board, features, target, fine_label)
logits = model(board, features=features, contamination=delta)
```

### Relation bias use

For square-token attention, convert the board to 64 tokens. At each attention layer:

\[
\text{attn}_{h}(i,j)
=
\frac{q_{h,i}^\top k_{h,j}}{\sqrt{d}}
+
b_{h}(\tilde A_{:,i,j})
\]

where:

```python
rel_bias = rel_proj(A_tilde.permute(0, 2, 3, 1))  # [B, 64, 64, n_heads]
attn_logits = attn_logits + rel_bias.permute(0, 3, 1, 2)
```

### Immutability guard

Add runtime assertions in training:

```python
assert not any(key in batch for key in TAINTED_KEYS)
assert board.requires_grad is False or board.grad is None
assert contamination.does_not_touch_piece_planes
```

For any debug feature dumps, explicitly omit source IDs and engine-like fields.

## 10. Robust Objective

### Base loss

Use binary cross entropy with logits:

```python
base = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
```

### Near-puzzle asymmetric cost

Use an asymmetric negative loss, not a post-hoc threshold correction:

```python
def asymmetric_bce_with_logits(logit, target, fine_label, gamma_near=2.0):
    pos_loss = target * F.softplus(-logit)
    neg_weight = torch.ones_like(target)
    neg_weight = torch.where(fine_label == 1, gamma_near * neg_weight, neg_weight)
    neg_loss = (1.0 - target) * neg_weight * F.softplus(logit)
    return pos_loss + neg_loss
```

This directly penalizes false positives on fine label `1`.

### Inner maximization

Approximate:

\[
\max_{\delta \in \mathcal U_\rho(x)}
\mathcal L_{\text{asym}}(f_\theta(x,r(x)+\delta),y,\ell)
\]

Use 2-4 steps of gradient ascent on continuous mask contamination:

```python
delta = adversary.zeros_like(features)
for _ in range(inner_steps):
    delta.requires_grad_(True)
    logits_adv = model(board, features=features, contamination=delta)
    loss_adv = asymmetric_bce_with_logits(
        logits_adv, target, fine_label, gamma_near=gamma_near
    ).mean()
    grad = torch.autograd.grad(loss_adv, delta.trainable_tensors())[0]
    delta = adversary.step_and_clip_to_budget(delta, grad)
```

The `step_and_clip_to_budget` operation must:

- zero contamination outside chess support masks,
- clamp contaminated masks to `[0, 1]`,
- enforce per-family L1 budgets,
- enforce per-family L-infinity budgets,
- never touch piece planes or state planes.

### Outer minimization

```python
features = feature_builder(board)
delta_star = adversary.inner_max(
    model=model,
    board=board,
    features=features,
    target=target,
    fine_label=fine_label,
)
logits = model(board, features=features, contamination=delta_star.detach())
loss_vec = asymmetric_bce_with_logits(logits, target, fine_label, gamma_near)
loss = loss_vec.mean() + weight_decay_penalty(model)
loss.backward()
optimizer.step()
```

### Robust risk with empirical CVaR option

If the repository already supports per-example losses, add a top-tail term to focus on hard samples without source groups:

\[
\mathcal L_{\text{outer}}
=
\text{mean}(\ell_i^{rob})
+
\eta\,\text{CVaR}_{\alpha}(\ell_i^{rob})
+
\lambda\lVert\theta\rVert_2^2
\]

Recommended:

```python
alpha = 0.2
eta = 0.25
```

This is still source-free. It is a tail-loss emphasis over examples, not group/source DRO.

### Training defaults

```python
inner_steps = 3
inner_step_size = 0.10
gamma_near = 2.0
weight_decay = 1e-4  # tune upward if overfitting
dropout = 0.10
early_stopping_metric = "near_puzzle_fp_at_fixed_puzzle_recall"
target_puzzle_recall_floor = 0.90
threshold = 0.0
```

Use stronger regularization and early stopping than the ERM baseline. The robust objective can overfit if the network memorizes all robust perturbations.

## 11. Ablations

Run these ablations with the same train/validation/test split and the same single-logit output.

### Required baselines

1. **ERM baseline**
   - Same architecture.
   - No tactical-mask adversary.
   - Standard BCE or same asymmetric BCE reported separately.

2. **Architecture-only tactical masks**
   - Uses \(u(x)\) and \(A(x)\).
   - No inner maximization.
   - Tests whether the gain comes merely from extra rule-derived inputs.

3. **MLTM-DRO full**
   - Relation masks plus tactical planes.
   - Material-locked inner maximization.
   - Near-puzzle asymmetric cost.

4. **MLTM-DRO without near-puzzle cost**
   - Same robust uncertainty set.
   - `gamma_near = 1.0`.
   - Tests whether robustness alone reduces fine-label-1 false positives.

5. **MLTM-DRO without relation masks**
   - Only 8x8 tactical feature contamination.
   - Useful if relation tensor code is suspected to be unstable.

6. **MLTM-DRO without pin/x-ray families**
   - Drops pin and x-ray uncertainty.
   - Tests whether near-puzzle false positives are driven by slider-line shortcuts.

7. **Budget sweep**
   - `rho_scale in {0.0, 0.25, 0.5, 1.0, 1.5}`
   - `0.0` should match no-adversary architecture-only behavior.
   - Excessive budgets should hurt recall; if not, the adversary may be ineffective.

8. **Inner-step sweep**
   - `inner_steps in {1, 2, 3, 5}`
   - More than 3 may not help; report wall-clock cost.

### Required metrics

Report:

- AUROC
- AUPRC
- BCE
- accuracy at threshold `0.0`
- puzzle recall
- ordinary false-positive rate
- near-puzzle false-positive rate
- 3x2 diagnostic matrix
- threshold-free fine-label curves if available:
  - near-puzzle FP at puzzle recall `0.80`, `0.90`, `0.95`
  - ordinary FP at puzzle recall `0.80`, `0.90`, `0.95`

### Required 3x2 diagnostic output format

```json
{
  "threshold": 0.0,
  "rows": ["fine_0_negative", "fine_1_near_puzzle_negative", "fine_2_puzzle_positive"],
  "cols": ["pred_negative", "pred_positive"],
  "counts": [[0, 0], [0, 0], [0, 0]],
  "rates": {
    "ordinary_fp_rate": 0.0,
    "near_puzzle_fp_rate": 0.0,
    "puzzle_fn_rate": 0.0,
    "puzzle_recall": 0.0
  }
}
```

### Shift evaluations

Use source-free training. For evaluation only, if the benchmark defines shifted test splits, report each split separately. Do not fit thresholds by source/split. Keep one global threshold.

Recommended shifted evaluations:

- Held-out collection or time slice, if already defined by the dataset
- High tactical-density slice based on board-derived masks
- Low material-count endgame-like slice based only on material vector
- King-danger-heavy slice based only on deterministic king-ring pressure
- Near-puzzle-heavy validation slice

The tactical-density and king-danger slices are diagnostic only. They must not become specialist calibrators.

## 12. Falsification Criteria

Reject MLTM-DRO if any of the following occur.

### Main failure

At fixed puzzle recall, near-puzzle false positives do not improve.

Concrete criterion:

```text
At puzzle recall >= 0.90,
near_puzzle_fp_rate must decrease by at least 20% relative to the best non-ensemble ERM baseline.
```

If the dataset is small, also report an absolute target:

```text
near_puzzle_fp_rate must decrease by at least 3 percentage points.
```

### Recall collapse

Reject if:

```text
puzzle_recall drops by more than 5 percentage points at threshold 0.0
```

unless a threshold-free curve shows clear dominance at the target operating point.

### Ordinary-negative regression

Reject if:

```text
ordinary_fp_rate increases by more than 5 percentage points
```

while near-puzzle FP improvement is below the main criterion.

### Robustness theater

Reject if the adversary does not change training behavior.

Symptoms:

- `rho_scale` sweep is flat.
- `inner_steps=0` and `inner_steps=3` have indistinguishable validation loss.
- Adversarial masks saturate to all zeros or all ones.
- Contamination appears outside support masks.
- Material or piece planes are altered.

### Shortcut leakage

Reject immediately if any of these are observed in model input, loss, thresholding, or calibration:

- source ID
- source file
- source family
- source frequency/rate
- engine score
- PV
- node count
- mate score
- best move
- puzzle solution move
- verification metadata

### Label instability

Reject if the implemented adversary mutates actual board occupancy, side to move, castling, en-passant state, or material. The uncertainty set is over derived masks only.

### Over-conservatism

Reject if the model simply suppresses all positives:

- Lower near-puzzle FP rate but much lower puzzle recall
- Mean logit shifts downward for all fine labels
- Fine label `2` row in the 3x2 diagnostic has too many predicted negatives

### Non-specific robustness

Reject if a generic Gaussian/noise adversary performs as well as or better than MLTM-DRO without using chess supports. That would falsify the chess-specific thesis and suggests the selected uncertainty set is not doing useful work.

## 13. Codex Implementation Notes

### Implementation target

Add a single robust training path behind a config flag:

```yaml
model:
  name: material_locked_tactical_dro
  output_dim: 1

robust:
  enabled: true
  type: material_locked_tactical_mask
  inner_steps: 3
  inner_step_size: 0.10
  gamma_near: 2.0
  rho_scale: 1.0
  eps: 0.35
  cvar_alpha: 0.20
  cvar_eta: 0.25
```

### Suggested files to modify or add

Use repository naming conventions, but the clean conceptual split is:

```text
models/material_locked_tactical_dro.py
features/chess_tactical_features.py
losses/material_locked_dro_loss.py
metrics/fine_label_diagnostic.py
trainers/robust_trainer.py
configs/material_locked_tactical_dro.yaml
tests/test_no_tainted_inputs.py
tests/test_material_lock.py
tests/test_3x2_diagnostic.py
```

### Feature builder checklist

`ChessTacticalFeatureBuilder` must:

- parse existing board tensor into piece planes;
- compute material vector;
- compute pseudo-legal attacks by piece type;
- compute same-side defense edges;
- compute opponent attack edges;
- compute king-ring planes;
- compute slider rays;
- compute pin and x-ray candidate rays;
- compute hanging-piece candidates;
- compute coarse mobility heat;
- emit support masks for each uncertainty family;
- run entirely without engine calls.

If board tensor format is unknown, create an adapter:

```python
class BoardTensorAdapter:
    def pieces(self, board) -> torch.Tensor: ...
    def side_to_move(self, board) -> torch.Tensor: ...
    def castling(self, board) -> Optional[torch.Tensor]: ...
    def en_passant(self, board) -> Optional[torch.Tensor]: ...
```

Do not guess hidden engine channels. If the board tensor has extra channels, explicitly whitelist the channels needed for board state and ignore the rest.

### Adversary implementation

Create a `MaterialLockedMaskAdversary`.

Core methods:

```python
class MaterialLockedMaskAdversary:
    def zeros_like(self, features: ChessFeatures) -> MaskContamination:
        ...

    def inner_max(self, model, board, features, target, fine_label):
        ...

    def step_and_clip_to_budget(self, delta, grad, features):
        ...
```

Budget clipping must operate per family. A practical implementation:

1. Apply gradient ascent step.
2. Zero entries outside support.
3. Clamp each contaminated mask to `[0, 1]`.
4. Convert back to delta.
5. Enforce `L_inf` by clamping delta.
6. Enforce normalized `L1` budget per family by shrinking the largest absolute entries or scaling absolute mass.
7. Re-zero outside support.

### Loss implementation

```python
def material_locked_dro_loss(
    model,
    board,
    fine_label,
    feature_builder,
    adversary,
    gamma_near: float = 2.0,
    cvar_alpha: float = 0.2,
    cvar_eta: float = 0.25,
):
    target = (fine_label == 2).float()
    features = feature_builder(board)

    delta_star = adversary.inner_max(
        model=model,
        board=board,
        features=features,
        target=target,
        fine_label=fine_label,
    )

    logits = model(board, features=features, contamination=delta_star.detach())
    loss_vec = asymmetric_bce_with_logits(logits, target, fine_label, gamma_near)

    mean_loss = loss_vec.mean()
    if cvar_eta > 0:
        k = max(1, int(math.ceil(cvar_alpha * loss_vec.numel())))
        tail_loss = torch.topk(loss_vec, k=k, largest=True).values.mean()
        return mean_loss + cvar_eta * tail_loss, logits

    return mean_loss, logits
```

### 3x2 diagnostic implementation

```python
@torch.no_grad()
def fine_label_3x2(logits, fine_label, threshold=0.0):
    pred = (logits >= threshold).long()
    diag = torch.zeros(3, 2, dtype=torch.long, device=logits.device)
    for k in range(3):
        for p in range(2):
            diag[k, p] = ((fine_label == k) & (pred == p)).sum()
    return diag
```

Aggregate across batches by summing `diag`.

### Tests

#### Tainted input test

Create a fake batch with tainted keys and assert the robust trainer either drops them or raises before model call.

```python
def test_no_tainted_inputs_reach_model():
    ...
```

#### Material lock test

Construct a batch with known material vector. Run inner adversary. Assert:

```python
assert material_before.equal(material_after)
assert piece_planes_before.equal(piece_planes_after)
assert side_to_move_before.equal(side_to_move_after)
```

Because the adversary should not touch the board at all, these should be trivially true.

#### Support-mask test

Assert:

```python
(delta_u.abs() * (1 - support_u)).sum() == 0
(delta_A.abs() * (1 - support_A)).sum() == 0
```

#### Diagnostic test

Given logits and fine labels:

```python
fine_label = torch.tensor([0, 0, 1, 1, 2, 2])
logits = torch.tensor([-1.0, 2.0, -0.5, 0.7, -0.1, 1.2])
```

Expected matrix:

```python
[[1, 1],
 [1, 1],
 [1, 1]]
```

### Training loop warning

Do not compute the derived masks on CPU one sample at a time if throughput matters. If the repository already has bitboard utilities, use vectorized tensor operations or cached per-position deterministic features. Caching is allowed only if the cache key is the board tensor or position hash and the cached value contains no tainted metadata.

### Logging

Log:

- robust loss
- clean validation loss
- adversarial validation loss
- mean absolute contamination per family
- fraction of used budget per family
- 3x2 diagnostic matrix
- near-puzzle FP rate
- puzzle recall
- threshold-free near-puzzle FP at fixed puzzle recall

Do not log source-conditioned thresholds or source-conditioned calibration.

## 14. Prompt Updates

Use the following prompt text for the next Codex pass.

```text
Implement Material-Locked Tactical Mask DRO for the chess puzzle classifier.

Inputs:
- Current board tensor only.
- Fine labels are available during training/evaluation: 0 and 1 map to target 0; 2 maps to target 1.
- Inference returns exactly one logit per board.

Forbidden everywhere in model input, feature building, loss, thresholding, and calibration:
engine data, PVs, node counts, mate scores, best moves, puzzle solution moves, verification metadata, source labels, source files, source-rate features.

Core method:
- Build deterministic chess-rule tactical features from the board tensor:
  attacks, defenses, king-ring pressure, slider rays, pin candidates, x-ray candidates, hanging-piece candidates, and mobility heat.
- Keep piece occupancy, side-to-move, castling/en-passant state, and material immutable.
- Define a material-locked uncertainty set over only the derived tactical planes and relation masks.
- Train with an inner maximization over bounded tactical-mask contamination and an outer minimization of asymmetric BCE.
- Penalize near-puzzle false positives by assigning fine label 1 a larger negative-class loss weight.
- Do not implement source-ID group DRO, source-rate calibration, phase-specialist calibration, nuisance projection, ensembling, or augmentation-only training.

Required validation output:
- AUROC, AUPRC, BCE.
- 3x2 diagnostic matrix with rows fine labels 0, 1, 2 and columns predicted negative, predicted positive.
- ordinary FP rate, near-puzzle FP rate, puzzle recall, puzzle FN rate.
- Near-puzzle FP at fixed puzzle recall values 0.80, 0.90, 0.95 where possible.

Success criterion:
At puzzle recall >= 0.90, reduce near-puzzle false-positive rate by at least 20% relative to the best same-architecture non-ensemble ERM baseline, without source-conditioned calibration.
```
