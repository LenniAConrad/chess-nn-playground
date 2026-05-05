# Codex Handoff Packet: Contamination-DRO Huber Tail Rejection

## 1. File Metadata

- **Generated:** 2026-04-28 07:55 new_york
- **Filename:** `chess_nn_research_2026-04-28_0755_tuesday_new_york_dro_tail_rejection.md`
- **Project context:** `chess-nn-playground` trains `puzzle_binary` models.
- **Fine labels:** `0 = non-puzzle`, `1 = near-puzzle hard negative`, `2 = puzzle`.
- **Binary target:** `target = 1` only for fine label `2`; fine labels `0` and `1` map to `target = 0`.
- **Primary diagnostic:** near-puzzle-to-puzzle false-positive rate: `mean(predicted_puzzle | fine_label == 1)`.
- **Selected idea:** add a training-only robust risk term for near-puzzle negatives: an upper-tail, density-ratio-capped contamination-DRO estimator applied to a Huberized one-sided logit-margin residual.
- **Inference footprint:** unchanged binary classifier using only the current board tensor and any already-allowed deterministic current-board rule features.
- **Non-goals:** no larger CNN, ensemble, calibration mixture, phase specialist, source-rate objective, ordinal head, credal head, prototype margin, near-puzzle twin ranking loss, data cleaning, or engine-derived feature.

## 2. Executive Selection

Select **Contamination-DRO Huber Tail Rejection**.

The model currently learns a binary puzzle decision, but the main failure mode is not generic negative error; it is the upper tail of near-puzzle hard negatives whose logits cross the puzzle threshold. Standard empirical risk minimization averages these mistakes with many easy non-puzzles and easy near-puzzles, so a small but operationally expensive false-positive tail can survive training.

The selected mechanism adds a robust-statistics risk estimator on the near-puzzle negative subset during training:

1. Convert each near-puzzle logit into a one-sided margin residual: `r = relu(logit + near_margin)`. A near-puzzle is considered comfortably rejected only when its logit is below `-near_margin`.
2. Huberize that residual so one mislabeled, ambiguous, or pathological near-puzzle cannot dominate the gradient.
3. Estimate the near-puzzle risk by a density-ratio-capped upper-tail DRO functional, equivalent in minibatch form to averaging the hardest `beta` fraction of near-puzzle residual losses.
4. Add that robust tail term to the ordinary binary cross-entropy loss.

This is not label smoothing. The binary targets remain hard `0/1`, and the near-puzzle label remains negative. This is not calibration. There is no post-hoc probability map, threshold tuning recipe, or mixture of calibrated heads. The change is a training-time robust risk estimator with bounded influence and an explicit contamination-neighborhood interpretation.

Expected effect: the decision boundary should become less willing to call a position a puzzle merely because it contains a puzzle-like surface pattern. The model must push the hardest near-puzzle logits below a negative margin, while the Huber cap prevents a tiny number of impossible negatives from hijacking training.

## 3. Data Contract

**Inputs allowed at inference and training**

- `board_tensor`: current chess board tensor only. Expected layout is whatever `puzzle_binary` already uses, typically `[B, C, 8, 8]` in PyTorch.
- Optional `rule_features`: deterministic current-board features only. Safe examples: side to move, castling rights, en-passant availability, check status, material counts, legal-move count, attacked-king indicator, piece counts, and other values derivable from the current legal board state without search.
- `fine_label`: training/evaluation label in `{0, 1, 2}`.

**Inputs forbidden everywhere**

- Stockfish scores, centipawn scores, win/draw/loss scores, mate scores.
- Principal variations, best moves, legal move ranked by engine, engine node counts, search depth, or engine verification metadata.
- Source labels, source file identity, source rates, generator identity, dataset provenance indicators, or verification pipeline metadata.
- Any field that tells the model how the example was found, verified, or grouped beyond the provided fine label.

**Target construction**

```python
y_binary = (fine_label == 2).float()
is_non_puzzle = fine_label == 0
is_near_puzzle = fine_label == 1
is_puzzle = fine_label == 2
```

The fine label may be used to compute the training loss term and evaluation metrics. It must not be used as an inference input.

**Prediction contract**

```python
logit = model(board_tensor, rule_features_optional)  # shape [B]
prob_puzzle = torch.sigmoid(logit)
predicted_puzzle = logit >= 0.0  # default fixed threshold for diagnostic comparisons
```

Use the same fixed threshold for baseline and ablations unless an evaluation explicitly reports threshold-free metrics. Do not tune a threshold to rescue this method; that would turn the experiment into calibration/operating-point selection rather than a test of the robust estimator.

## 4. Robust-Statistics Research Map

This packet uses robust statistics as a training-risk estimator, not as a post-hoc output adjustment.

- **Influence functions.** Hampel's influence-curve framework studies the local effect of infinitesimal contamination on an estimator. The design below makes the near-puzzle residual loss have a bounded derivative under Huberization, so individual examples have bounded local influence on the added robust term. Source anchor: Frank R. Hampel, "The Influence Curve and its Role in Robust Estimation," *Journal of the American Statistical Association*, 1974, DOI `10.1080/01621459.1974.10482962`.
- **Huber M-estimation.** Huber losses are classical M-estimator losses: quadratic near zero and linear in the tail. Here the residual is not a regression residual; it is a one-sided classification-margin violation for near-puzzle negatives. Source anchor: Peter J. Huber, "Robust Estimation of a Location Parameter," *Annals of Mathematical Statistics*, 1964, DOI `10.1214/aoms/1177703732`.
- **Contamination neighborhoods.** The near-puzzle tail is treated as the part of the negative-class risk most vulnerable to distribution shift: deployment may contain near-puzzles that look more puzzle-like than the average validation near-puzzle. A density-ratio-capped uncertainty set formalizes this by allowing the risk estimator to reweight examples toward high-loss near-puzzles, but only up to a cap.
- **Distributionally robust risk.** DRO over non-parametric uncertainty sets is a standard way to put more weight on observations inducing high loss while staying close to the empirical distribution. Source anchors: Namkoong and Duchi, "Stochastic Gradient Methods for Distributionally Robust Optimization with f-divergences," NeurIPS 2016; Duchi and Namkoong, "Variance-based Regularization with Convex Objectives," *JMLR*, 2019.
- **Median-of-means and trimming were considered but not selected.** Median-of-means is a real robust estimator and has strong guarantees in heavy-tailed/corrupted settings, but it is awkward for the core diagnostic because it stabilizes mean-risk estimation rather than directly attacking the upper false-positive tail. Source anchor: Lecué and Lerasle, "Robust machine learning by median-of-means: Theory and practice," *Annals of Statistics*, 2020, DOI `10.1214/19-AOS1828`.

The selected object is the following robust risk functional:

```text
near_tail_risk(theta)
= sup over q in Delta_n, q_i <= 1 / (beta * n):
    sum_i q_i * huber_kappa(relu(logit_theta(x_i) + near_margin))
```

This is a density-ratio-capped contamination-neighborhood estimator. In minibatch code it is the average of the hardest `beta` fraction of near-puzzle residual losses, with Huber bounded influence applied before the upper-tail estimator.

## 5. Candidate Mechanisms Considered

**Selected: Huberized upper-tail contamination-DRO near-puzzle risk**

- Directly optimizes the diagnostic's failure tail: near-puzzle examples whose logits are too positive.
- Uses a real robust-statistics object: density-ratio-capped DRO risk under a contamination-neighborhood interpretation, plus Huber M-estimator influence bounding.
- Requires no inference-time input changes.
- Keeps the binary head and hard labels.
- Is easy for Codex to implement as a loss wrapper around the existing `puzzle_binary` model.

**Considered: pure Huberized binary cross-entropy**

- Good robust-statistics lineage, but too generic.
- It caps extreme gradients across all examples and may weaken pressure on severe near-puzzle false positives.
- Rejected as the primary mechanism because the main diagnostic is group-specific and tail-specific.

**Considered: median-of-means minibatch loss**

- Robust to corrupted minibatches and heavy-tailed loss estimates.
- Poor match to the target metric because it can hide rare but important near-puzzle false positives inside block medians.
- Rejected for this task, but useful as a future stability check if training loss variance becomes the problem.

**Considered: symmetric trimmed loss**

- Robust to outliers by trimming extremes.
- Dangerous here because trimming high losses may remove exactly the near-puzzle false positives the model must learn to reject.
- Rejected.

**Considered: embedding depth functions**

- Robust multivariate depth could identify central versus outlying positions in learned feature space.
- It risks becoming a prototype/embedding-margin method and adds implementation complexity.
- Rejected because prototype margins are explicitly out of scope.

**Considered: influence-regularized per-sample gradients**

- Theoretically attractive, but full influence-function estimation over neural-network parameters is expensive and fragile.
- Rejected in favor of the simpler bounded-influence Huber residual and capped DRO weights.

## 6. Rejected Approaches Table

| Approach | Status | Reason |
|---|---:|---|
| Larger CNN backbone | Rejected | The requested intervention is robust statistics, not capacity scaling. |
| Ensemble of classifiers | Rejected | Adds inference cost and violates the explicit avoid list. |
| Calibration mixture or post-hoc calibrator | Rejected | Would tune probabilities after training rather than changing the estimator that learns near-puzzle rejection. |
| Phase specialist heads | Rejected | Game-phase routing is outside scope and risks hidden specialist models. |
| Source-rate objective | Rejected | Uses dataset/source prevalence or provenance; source labels and rates are forbidden. |
| Ordinal head over labels `0/1/2` | Rejected | Explicitly disallowed; also changes the task semantics away from binary puzzle detection. |
| Credal/uncertainty head | Rejected | Explicitly disallowed and likely becomes a calibration-adjacent mechanism. |
| Prototype margins | Rejected | Explicitly disallowed and requires learned class prototypes. |
| Near-puzzle twin ranking loss | Rejected | Explicitly disallowed; pairs/twins are not part of the data contract. |
| Data cleaning or relabeling near-puzzles | Rejected | Explicitly disallowed; the model must learn under the current labels. |
| Stockfish/evaluation-derived features | Rejected | Forbidden inputs. |
| Best-move/PV/mate metadata | Rejected | Forbidden inputs. |
| Threshold-only adjustment | Rejected | May reduce false positives by sacrificing recall without learning better rejection. |
| Plain label smoothing | Rejected | Not a robust estimator; weakens hard-negative supervision and changes target semantics. |

## 7. Mathematical Thesis

Let `C` be the fine label and `Y = 1[C = 2]`. The classifier produces a logit `s_theta(X)`. The primary failure metric is:

```text
FPR_near(theta) = P(s_theta(X) >= 0 | C = 1)
```

Ordinary binary ERM minimizes an empirical mean:

```text
R_ERM(theta) = mean_i BCEWithLogits(s_theta(x_i), y_i)
```

This mean is not aligned with the near-puzzle false-positive tail. If near-puzzles are a minority, and only a minority of near-puzzles cross the puzzle threshold, the aggregate BCE can improve while `FPR_near` remains unacceptable.

The proposed robust thesis is:

```text
A near-puzzle false positive is a one-sided positive-tail event inside the negative class.
Therefore the training objective should contain a robust upper-tail estimator of near-puzzle margin violations.
```

Define the near-puzzle residual:

```text
r_theta(x) = [s_theta(x) + m]_+ = max(0, s_theta(x) + m)
```

where `m > 0` is a rejection margin. A near-puzzle with `s_theta(x) <= -m` has zero residual. A near-puzzle at the default puzzle threshold, `s_theta(x) = 0`, has residual `m`.

Apply a Huber score:

```text
h_kappa(r) = 0.5 * r^2                  if 0 <= r <= kappa
           = kappa * (r - 0.5*kappa)   if r > kappa
```

This gives score function:

```text
psi_kappa(r) = d h_kappa(r) / dr = min(r, kappa)
```

So the added term has bounded per-example local influence.

Then estimate near-puzzle risk by a density-ratio-capped upper-tail DRO functional:

```text
R_near(theta) = sup_{q in Delta_n, 0 <= q_i <= 1/(beta*n)} sum_i q_i h_kappa(r_theta(x_i))
```

where the `n` examples are the near-puzzle examples in the current batch or accumulation window, and `beta` is the hard-tail fraction. This estimator is equivalent to an upper-tail trimmed mean: it averages the hardest `beta` fraction of near-puzzle residual losses rather than the whole near-puzzle mean.

A useful diagnostic bound follows from Markov's inequality. For near-puzzles, if `r = [s + m]_+`, then the event `s >= 0` implies `r >= m`. Therefore:

```text
P(s >= 0 | C = 1) <= E[h_kappa(r) | C = 1] / h_kappa(m)
```

Minimizing a robust upper-tail version of `E[h_kappa(r) | C = 1]` attacks the empirical upper tail that causes near-puzzle false positives. The term is still robust: Huberization bounds individual influence, and density-ratio capping prevents the objective from collapsing into a single worst-example max loss.

## 8. Robust Estimator Definition

**Name:** Huberized Near-Puzzle Contamination-DRO Tail Estimator.

**Inputs per training batch**

- `logits`: shape `[B]`.
- `fine_label`: shape `[B]`, integer values in `{0, 1, 2}`.
- Hyperparameters:
  - `near_margin m`: default `0.5` logit units.
  - `huber_kappa`: default `2.0`.
  - `tail_beta`: default `0.25`, meaning the hardest 25% of near-puzzle residuals.
  - `near_tail_weight lambda_np`: default `0.5`.
  - `min_near_count`: default `4`.

**Estimator steps**

```python
near_logits = logits[fine_label == 1]
r = torch.relu(near_logits + near_margin)
v = huber(r, huber_kappa)
robust_tail = capped_upper_tail_mean(v, beta=tail_beta)
```

**Huber function**

```python
def huber_positive_residual(r: torch.Tensor, kappa: float) -> torch.Tensor:
    return torch.where(
        r <= kappa,
        0.5 * r.square(),
        kappa * (r - 0.5 * kappa),
    )
```

**Capped upper-tail mean**

Exact density-ratio-capped estimator for vector `v` of length `n`:

```text
sort v descending: v_(1) >= ... >= v_(n)
t = beta * n
k = floor(t)
gamma = t - k
if gamma == 0:
    R = (sum_{j=1}^k v_(j)) / t
else:
    R = (sum_{j=1}^k v_(j) + gamma * v_(k+1)) / t
```

Codex can implement the simpler top-k estimator first:

```python
def upper_tail_mean(v: torch.Tensor, beta: float) -> torch.Tensor:
    if v.numel() == 0:
        return v.new_zeros(())
    k = max(1, math.ceil(beta * v.numel()))
    return torch.topk(v, k=k, largest=True, sorted=False).values.mean()
```

The top-k version is acceptable for the first experiment because it is deterministic, differentiable almost everywhere, and matches the intended capped-DRO behavior closely enough for minibatch training.

**Influence behavior**

For selected top-tail examples, the residual derivative is capped by `huber_kappa`. For non-selected examples, the robust-tail derivative is zero. With `k` selected examples, each selected example has weight `1/k`, so the added term's logit derivative is bounded by:

```text
abs(d robust_tail / d logit_i) <= huber_kappa / k
```

This is the robust-statistics reason the method is not just hard-example mining. Hard-example mining can overreact to mislabeled or impossible examples. This estimator both focuses on the high-risk near-puzzle tail and bounds each selected example's influence.

## 9. Architecture Specification

Use the existing `puzzle_binary` architecture. Do not enlarge the CNN.

**Required architecture behavior**

```python
class PuzzleBinaryModel(nn.Module):
    def forward(self, board_tensor, rule_features=None):
        # existing backbone and binary head
        # return shape [B] logits, not sigmoid probabilities
        return logits
```

**Permitted architecture details**

- Existing board encoder may remain unchanged.
- Existing deterministic rule-feature sidecar may remain if it already uses current-board-only features.
- Final head remains a single binary logit.
- No new inference-time branches, experts, prototype stores, pairwise comparators, phase routers, or uncertainty heads.

**Training-only addition**

Add a loss module, not a model module:

```python
loss = BCEWithLogitsLoss_mean(logits, y_binary) \
     + lambda_np * HuberizedNearPuzzleTailDRO(logits, fine_label)
```

**Inference behavior**

No change:

```python
prob = sigmoid(logit)
pred = logit >= 0.0
```

Do not add calibration or threshold tuning as part of the selected method.

## 10. Tensor Contract

**Batch dictionary**

```python
batch = {
    "board": FloatTensor[B, C, 8, 8],
    "fine_label": LongTensor[B],  # values 0, 1, 2
    # optional, only if already supported and current-board deterministic:
    "rule_features": FloatTensor[B, F],
}
```

**Derived tensors**

```python
board = batch["board"].float()
fine_label = batch["fine_label"].long()
y = (fine_label == 2).float()
mask_non = fine_label == 0
mask_near = fine_label == 1
mask_puzzle = fine_label == 2
```

**Model output**

```python
logits = model(board, batch.get("rule_features"))
logits.shape == (B,)  # or [B, 1] squeezed to [B]
```

**Loss tensors**

```python
base_bce_per_sample = F.binary_cross_entropy_with_logits(
    logits,
    y,
    reduction="none",
)
near_logits = logits[mask_near]
near_residual = F.relu(near_logits + near_margin)
near_huber = huber_positive_residual(near_residual, huber_kappa)
near_tail = upper_tail_mean(near_huber, tail_beta)
loss = base_bce_per_sample.mean() + lambda_np * near_tail
```

**Metrics tensors**

```python
pred = logits >= 0.0
near_fpr = pred[mask_near].float().mean()
non_fpr = pred[mask_non].float().mean()
puzzle_recall = pred[mask_puzzle].float().mean()
overall_acc = (pred == y.bool()).float().mean()
```

Report metric denominators. If a split has zero examples for a fine label, report `nan` and do not silently coerce to zero.

## 11. Loss Function

**Full selected loss**

```text
L(theta) = mean_i BCEWithLogits(s_theta(x_i), y_i)
         + lambda_np * R_near(theta)
```

where:

```text
R_near(theta)
= sup_{q in Delta_n, q_i <= 1/(beta*n)} sum_i q_i h_kappa([s_theta(x_i) + m]_+)
```

computed only over examples with `fine_label == 1`.

**Recommended first hyperparameters**

```yaml
near_tail_dro: true
near_tail_weight: 0.5
tail_beta: 0.25
near_margin: 0.5
huber_kappa: 2.0
min_near_count: 4
```

**Batch edge cases**

- If no near-puzzle examples are in a batch, set `near_tail = 0`.
- If `0 < near_count < min_near_count`, use the mean Huber residual instead of top-k. This avoids a single near-puzzle in a small batch becoming a de facto max objective.
- Prefer stratified batching with at least `min_near_count` near-puzzle examples per batch if the current data loader can do that without removing or duplicating records. If not, keep natural batches and rely on the fallback.

**Reference implementation**

```python
import math
import torch
import torch.nn.functional as F


def huber_positive_residual(r: torch.Tensor, kappa: float) -> torch.Tensor:
    return torch.where(
        r <= kappa,
        0.5 * r.square(),
        kappa * (r - 0.5 * kappa),
    )


def upper_tail_mean(v: torch.Tensor, beta: float) -> torch.Tensor:
    if v.numel() == 0:
        return v.new_zeros(())
    k = max(1, math.ceil(float(beta) * int(v.numel())))
    return torch.topk(v, k=k, largest=True, sorted=False).values.mean()


def puzzle_binary_robust_loss(
    logits: torch.Tensor,
    fine_label: torch.Tensor,
    *,
    near_tail_weight: float = 0.5,
    tail_beta: float = 0.25,
    near_margin: float = 0.5,
    huber_kappa: float = 2.0,
    min_near_count: int = 4,
    pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = logits.view(-1)
    fine_label = fine_label.view(-1).long()
    y = (fine_label == 2).to(dtype=logits.dtype)

    bce = F.binary_cross_entropy_with_logits(
        logits,
        y,
        reduction="none",
        pos_weight=pos_weight,
    )
    base_loss = bce.mean()

    near_logits = logits[fine_label == 1]
    if near_logits.numel() == 0 or near_tail_weight == 0.0:
        near_tail = logits.new_zeros(())
    else:
        residual = F.relu(near_logits + near_margin)
        h = huber_positive_residual(residual, huber_kappa)
        if near_logits.numel() < min_near_count:
            near_tail = h.mean()
        else:
            near_tail = upper_tail_mean(h, tail_beta)

    total = base_loss + float(near_tail_weight) * near_tail
    parts = {
        "loss_base_bce": base_loss.detach(),
        "loss_near_tail": near_tail.detach(),
        "near_count": torch.as_tensor(near_logits.numel(), device=logits.device),
    }
    return total, parts
```

Do not replace the base BCE with the robust term. The base BCE preserves ordinary binary learning on puzzles, non-puzzles, and near-puzzles; the robust term adds targeted pressure on the near-puzzle false-positive tail.

## 12. Ablations

Run all ablations with the same data split, same model size, same optimizer budget, same random seeds, and fixed threshold `logit >= 0` for the main diagnostic.

| Ablation | Change | Purpose |
|---|---|---|
| A0 baseline | Existing `puzzle_binary` BCE | Establish current near-puzzle false-positive rate. |
| A1 near mean margin | Add `mean(huber(relu(logit + m)))` over fine label `1`; no tail DRO | Tests whether margin alone is sufficient. |
| A2 upper-tail only | Use top-tail mean of `relu(logit + m)` without Huber | Tests whether tail focus without bounded influence is unstable. |
| A3 Huber only | Huberized near residual mean; no upper-tail selection | Tests bounded influence without tail targeting. |
| A4 selected | Huberized upper-tail contamination-DRO | Main candidate. |
| A5 beta sweep | `tail_beta in {0.10, 0.25, 0.50}` | Determines how much near-puzzle tail mass should drive the term. |
| A6 margin sweep | `near_margin in {0.0, 0.25, 0.5, 1.0}` | Tests required rejection margin. |
| A7 kappa sweep | `huber_kappa in {1.0, 2.0, 4.0}` | Tests influence cap sensitivity. |
| A8 lambda sweep | `near_tail_weight in {0.25, 0.5, 1.0}` | Measures recall/false-positive tradeoff. |
| A9 deterministic features unchanged vs disabled | Only if rule features already exist | Verifies the robust loss, not rule features, causes the improvement. |

**Required reports**

- `near_fpr@logit0`: `mean(logit >= 0 | fine_label == 1)`.
- `non_fpr@logit0`: `mean(logit >= 0 | fine_label == 0)`.
- `puzzle_recall@logit0`: `mean(logit >= 0 | fine_label == 2)`.
- `overall_binary_fpr@logit0`: `mean(logit >= 0 | fine_label in {0,1})`.
- `overall_binary_recall@logit0`: same as puzzle recall.
- `AUROC_binary`: target `2` versus `0/1`.
- `AUPRC_binary`: target `2` versus `0/1`.
- Quantiles of logits by fine label: p05, p25, p50, p75, p95.
- Robust-loss internals: mean `near_tail`, selected top-k count, mean near residual, max near residual.

**Success criterion for first pass**

A4 is worth keeping if, averaged over at least three seeds:

```text
near_fpr reduction >= 20% relative to A0
and puzzle_recall drop <= 5% relative to A0
and AUROC_binary drop <= 1 percentage point
```

If baseline near-puzzle FPR is already very low, use an absolute criterion instead:

```text
near_fpr absolute reduction >= 0.5 percentage points
with puzzle_recall drop <= 5% relative
```

## 13. Falsification Rule

Reject the selected idea if any of the following occurs after the ablation protocol:

1. **No diagnostic improvement:** A4 fails to reduce `near_fpr@logit0` versus A0 under the success criterion.
2. **Recall collapse:** A4 reduces near-puzzle false positives only by suppressing puzzle recall by more than 5% relative.
3. **False-positive displacement:** A4 improves fine-label `1` FPR but materially worsens fine-label `0` FPR, producing no meaningful improvement in overall binary FPR.
4. **Ranking damage:** A4 improves fixed-threshold FPR but drops `AUROC_binary` by more than 1 percentage point, implying the gain is mostly a logit shift rather than better separation.
5. **Instability:** A4 has high seed variance where at least one seed is worse than baseline on near-puzzle FPR and puzzle recall simultaneously.
6. **Huber cap irrelevance:** A2 and A4 are statistically indistinguishable while A2 is simpler and stable. In that case, the robust influence-bound portion is not earning its complexity.
7. **Tail term degeneracy:** The selected top-k examples are almost always the same tiny set of records across epochs, and training overfits them. Do not clean or remove those records; instead reject or reduce `near_tail_weight`/increase `tail_beta` within the ablation budget.

The method should not be rescued by post-hoc threshold tuning, calibration, relabeling, source-based sampling, or engine-derived filtering. Those would answer a different prompt.

## 14. Codex Implementation Notes

**Search targets**

Ask Codex to locate the existing training path rather than assuming exact filenames:

```bash
rg "puzzle_binary|BCEWithLogitsLoss|binary_cross_entropy_with_logits|fine_label|label" .
rg "near|puzzle" data src training scripts .
```

**Minimal implementation plan**

1. Ensure each training batch exposes `fine_label` with values `{0,1,2}`. If the current dataset only emits binary labels, modify the dataset return dict to include the original fine label too. Do not alter filtering, cleaning, or sampling.
2. Add a loss module, for example `robust_tail_loss.py`, containing:
   - `huber_positive_residual`
   - `upper_tail_mean`
   - `puzzle_binary_robust_loss`
3. Add config/CLI flags:

```yaml
robust_near_tail:
  enabled: true
  weight: 0.5
  beta: 0.25
  margin: 0.5
  huber_kappa: 2.0
  min_near_count: 4
```

4. In the training step:

```python
logits = model(...).view(-1)
if cfg.robust_near_tail.enabled:
    loss, loss_parts = puzzle_binary_robust_loss(
        logits,
        batch["fine_label"],
        near_tail_weight=cfg.robust_near_tail.weight,
        tail_beta=cfg.robust_near_tail.beta,
        near_margin=cfg.robust_near_tail.margin,
        huber_kappa=cfg.robust_near_tail.huber_kappa,
        min_near_count=cfg.robust_near_tail.min_near_count,
        pos_weight=existing_pos_weight_if_any,
    )
else:
    y = (batch["fine_label"] == 2).float()
    loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=existing_pos_weight_if_any)
```

5. Log `loss_base_bce`, `loss_near_tail`, `near_count`, and fine-label metrics.
6. Keep checkpoint selection honest. Preferred checkpoint metric is a constrained score, not only validation BCE:

```text
select lowest near_fpr among checkpoints with puzzle_recall >= 0.95 * baseline_puzzle_recall
```

If that selection rule is considered too close to threshold selection, choose by validation BCE and report the near-FPR metric separately. Do not tune probability calibration.

**Unit tests**

- `upper_tail_mean([1,2,3,4], beta=0.5) == mean([4,3])`.
- Empty near-puzzle batch returns zero robust term and finite total loss.
- `fine_label == 1` examples use binary target zero in base BCE.
- Increasing a near-puzzle logit above `-margin` increases the robust tail term.
- Very large near-puzzle logits have bounded derivative from the Huber term.
- The loss function does not read keys named `stockfish`, `eval`, `pv`, `best_move`, `source`, `file`, `verification`, `nodes`, or `mate`.

**Logging snippet**

```python
with torch.no_grad():
    pred = logits >= 0
    for c, name in [(0, "non"), (1, "near"), (2, "puzzle")]:
        mask = batch["fine_label"] == c
        if mask.any():
            if c == 2:
                metrics[f"{name}_recall_logit0"] = pred[mask].float().mean()
            else:
                metrics[f"{name}_fpr_logit0"] = pred[mask].float().mean()
            metrics[f"{name}_logit_p50"] = logits[mask].median()
            metrics[f"{name}_count"] = mask.sum()
```

**Common failure mode and fix within scope**

If the robust term is too strong, puzzle recall will fall. First reduce `near_tail_weight` from `0.5` to `0.25`; second increase `tail_beta` from `0.25` to `0.50`; third reduce `near_margin` from `0.5` to `0.25`. Do not add calibration, experts, or engine features.

## 15. What Future Prompts Should Avoid

Future prompts should not ask Codex to solve this by adding any of the following:

- Bigger CNNs, deeper backbones, wider residual towers, or architecture scaling.
- Ensembles, multi-checkpoint voting, snapshot averaging, or mixture-of-experts.
- Calibration mixtures, Platt scaling, isotonic regression, validation-threshold tuning, or threshold-only fixes.
- Phase specialists, opening/middlegame/endgame routers, or position-family specialist heads.
- Source-rate objectives, source balancing, source identity features, source-file splits as model inputs, or generator-provenance features.
- Ordinal heads over `0 < 1 < 2`, credal heads, uncertainty heads, or abstention heads.
- Prototype margins, embedding class centers, contrastive class prototypes, or nearest-prototype rejection.
- Near-puzzle twin ranking losses, paired puzzle/near-puzzle objectives, or pair mining.
- Data cleaning, relabeling, removing ambiguous near-puzzles, or filtering hard negatives with an engine.
- Any Stockfish-derived input: scores, mate values, PVs, nodes, best moves, depths, verification status, or tactical solution metadata.

The useful next prompt is narrow: implement the robust loss wrapper, preserve the current binary model, report fine-label diagnostics, and falsify the idea if the near-puzzle FPR does not improve without unacceptable puzzle-recall loss.
