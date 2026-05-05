# Codex Handoff Packet: Soft Sorting Order Residual Ranker

## 1. File Metadata

- **Filename:** `chess_nn_research_2026-04-28_0758_tuesday_new_york_soft_sort_ranker.md`
- **Generated:** 2026-04-28 07:58 new_york
- **Project:** `chess-nn-playground`
- **Task:** `12. Differentiable Ranking Without Twin Loss`
- **Selected idea:** Soft Sorting Order Residual Ranker
- **Central order/ranking construct:** differentiable sorting network
- **Model family:** `puzzle_binary`
- **Inference input contract:** current-board tensor only
- **Inference output contract:** one scalar logit
- **Fine labels:** `0, 1, 2`
- **Binary target:** `target = 1` iff `fine_label == 2`; `target = 0` iff `fine_label in {0, 1}`
- **Forbidden at inference:** Stockfish, PVs, node counts, mate scores, best moves, verification metadata, source labels, source file IDs
- **Required evaluation addition:** fine-label confusion table with true fine label rows `0, 1, 2` and predicted binary columns `0, 1`

Primary source anchors used for the research map:

1. Petersen, Borgelt, Kuehne, Deussen, **"Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision,"** ICML 2021 / IBM Research: https://research.ibm.com/publications/differentiable-sorting-networks-for-scalable-sorting-and-ranking-supervision
2. Blondel, Teboul, Berthet, Djolonga, **"Fast Differentiable Sorting and Ranking,"** ICML 2020 / PMLR: https://proceedings.mlr.press/v119/blondel20a.html
3. Prillo, Eisenschlos, **"SoftSort: A Continuous Relaxation for the argsort Operator,"** ICML 2020 / PMLR: https://proceedings.mlr.press/v119/prillo20a.html
4. Grover, Wang, Zweig, Ermon, **"Stochastic Optimization of Sorting Networks via Continuous Relaxations,"** ICLR 2019 reference implementation: https://github.com/ermongroup/neuralsort
5. PyTorch `BCEWithLogitsLoss` documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
6. scikit-learn `confusion_matrix` documentation: https://sklearn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
7. `torchsort` PyPI package documentation: https://pypi.org/project/torchsort/

## 2. Executive Selection

Build `puzzle_binary` as a normal one-logit current-board classifier, but train it with an additional batch-level differentiable ordering residual. The model emits one scalar score `s_theta(x)` for each current-board tensor. During training only, a differentiable sorting network softly sorts the batch by those scores. The label carrier vector is carried through the same relaxed compare-swap network. If the model has learned a useful puzzle score, all `fine_label == 2` positions should occupy the top `k` slots after sorting, where `k` is the number of positives in the batch. The loss penalizes the residual between this soft sorted label occupancy and the ideal top-`k` binary occupancy vector.

This is not near-puzzle twin ranking. It does not pair a puzzle with its neighboring non-puzzle. It does not use prototype margins. It does not build an ordinal evidence ladder over fine labels. It does not calibrate by source rate. It does not tune thresholds. It is a direct differentiable ranking/order objective over the model's own scalar board score.

The recommended production shape is:

```text
current_board_tensor -> board_cnn_or_resnet -> scalar_logit
loss = BCEWithLogitsLoss(logit, binary_target)
     + lambda_order * SoftSortingOrderResidual(logit, binary_target)
```

Use fine labels only to form the binary target and to report evaluation breakdowns. The order residual should use `target`, not a three-level ordinal objective.

Why this selection is strong:

- It gives the model a ranking signal that is aligned with threshold-free separation metrics such as AUROC/AUPRC, while preserving the one-logit binary classifier contract.
- It is an actual order-theoretic object: a sorting network is a fixed comparator network whose hard limit implements a total order; the relaxation makes the order map differentiable.
- It is implementable in plain PyTorch without forbidden chess metadata.
- It leaves inference unchanged: one current-board tensor in, one logit out.
- It has clear falsification tests: if the extra ordering residual does not improve ranking or worsens fine-label confusion, remove it.

## 3. Problem And Data Contract

### Problem

Train a `puzzle_binary` model that predicts whether the current board is a true puzzle-positive position.

### Labels

```python
fine_label in {0, 1, 2}
target = (fine_label == 2).float()
```

Interpretation for training:

```text
fine 0 -> target 0
fine 1 -> target 0
fine 2 -> target 1
```

The training loss must not treat `fine_label == 1` as an ordinal middle class. That would drift into the forbidden "ordinal evidence ladder" family. For this packet, `fine_label` has two legitimate uses:

1. compute the binary target;
2. report fine-label confusion and per-fine-label metrics.

### Model input

The model receives only the current-board tensor.

Recommended tensor contract:

```python
board: FloatTensor[B, C, 8, 8]
fine_label: LongTensor[B]      # used by loss/eval wrapper, not as model input
target: FloatTensor[B]         # derived as fine_label == 2
```

The exact channel count `C` should match the existing `chess-nn-playground` board encoder. Do not add channels that encode engine outputs, PVs, node counts, mate scores, best moves, verification state, source labels, or source file IDs.

### Model output

```python
logit: FloatTensor[B]
```

No auxiliary inference outputs are required. Hidden activations may exist normally inside the network, but the externally consumed model output is one scalar logit.

### Evaluation

Report at least:

```text
binary confusion, rows=true target {0,1}, cols=pred target {0,1}
fine-label confusion, rows=true fine {0,1,2}, cols=pred binary {0,1}
AUROC
AUPRC
BCE validation loss
order residual validation diagnostic
```

Predicted binary class must use a fixed rule:

```python
pred_binary = (logit > 0).long()
```

Do not tune the threshold. The fixed zero threshold follows the logit convention created by `BCEWithLogitsLoss`, not a post-hoc validation threshold.

Fine-label confusion table:

```text
                 predicted binary 0    predicted binary 1
true fine 0              n00                  n01
true fine 1              n10                  n11
true fine 2              n20                  n21
```

This table is more useful than a fake 3-class confusion matrix because the model does not output three classes.

## 4. Ranking/Order Research Map

| Research object | What it contributes | Fit for this task |
|---|---|---|
| Differentiable sorting networks | Replace hard compare-swap operations with soft compare-swap gates; keep the fixed comparator graph of odd-even or bitonic sorting networks. Petersen et al. describe relaxing pairwise conditional swaps and using sorting/ranking supervision from ordering constraints. | **Selected.** It is concrete, order-theoretic, one-logit compatible, and easy to implement around a batch of scalar logits. |
| SoftSort / continuous argsort | Provides a continuous relaxation of `argsort`, motivated by the fact that hard argsort has a discrete image and blocks useful gradients. | Good fallback if a compact matrix relaxation is preferred over writing a sorting network. Less central here because the packet wants a visibly real ranking/order construct. |
| Fast differentiable sorting/ranking via permutahedron projection | Casts differentiable sorting/ranking as projection onto the permutahedron and reduces it to isotonic optimization, with efficient `O(n log n)` behavior. | Strong ablation or dependency-based implementation through `torchsort`. Slightly more complex than the selected soft sorting network. |
| NeuralSort / stochastic continuous sorting | Provides continuous sorting relaxations used for stochastic optimization over sorting networks. | Relevant foundation, but the current handoff should avoid kNN-style or twin-like constructions. |
| Stochastic dominance loss | Compare positive and negative score distributions through soft CDF dominance. | Valid but weaker as the central object because it is distributional rather than a specific sorter. Keep as future alternative, not this packet. |
| Isotonic projection | Project scores or residuals onto a monotone cone/order cone. | Useful theory and dependency path, but direct projection is less natural for binary top-`k` occupancy unless using the permutahedron route. |
| Order polytope residual | Penalize distance from predictions to a feasible monotone polytope under label-induced constraints. | Interesting, but with only binary `target`, the poset becomes too shallow unless extra structure is introduced; extra structure risks violating the data contract. |
| Partial order dimension | Embed a partial order as the intersection of linear extensions. | Too likely to require multi-coordinate scoring or side constraints. Not selected for a one-logit model. |
| Lattice rank | Use rank functions over a lattice of board features or latent states. | Too architecture-heavy and risks inventing unverified chess-specific state objects. Not selected. |

Key research takeaways:

- Sorting/ranking operators are hard to optimize directly because ranking is piecewise constant and sorting has non-differentiable kinks; differentiable relaxations exist specifically to make these operations trainable by gradient descent.
- Differentiable sorting networks are attractive here because they preserve the actual comparator-network structure of sorting while making each compare-swap operation soft.
- The selected loss is not a margin over hand-picked pairs. It is a batch-level residual over the whole induced order.

## 5. Candidate Search Trace

### Search objective

Find one ranking/order-theoretic method that satisfies all of the following:

```text
current-board tensor only
one output logit
fine labels 0/1/2 available
binary target: fine 2 vs fine 0/1
fine-label confusion reported
no engine/PV/mate/source/verification inference inputs
not near-puzzle twin ranking
not prototype margins
not ordinal evidence ladders
not source-rate calibration
not threshold tuning
```

### Candidate trail

1. **Pure pairwise AUC logistic loss.**
   - Pros: simple; one logit; aligns with ranking.
   - Cons: too close to generic pairwise ranking; not a distinctive order-theoretic object; easy for Codex to accidentally turn into forbidden twin ranking.
   - Decision: reject as central method; allow as a small ablation control.

2. **SoftSort top-`k` occupancy residual.**
   - Pros: compact; continuous argsort; one logit; good literature support.
   - Cons: less explicit than a comparator network unless implemented carefully.
   - Decision: valid fallback.

3. **Permutahedron/torchsort soft-rank loss.**
   - Pros: efficient; mature theory; PyTorch package exists.
   - Cons: dependency and build complexity can distract; `torchsort` may require CUDA/extension handling depending on environment.
   - Decision: ablation path or implementation fallback.

4. **Differentiable sorting network with soft compare-swap gates.**
   - Pros: central object is unambiguous; easy to test; works entirely on scalar logits and label carriers; no forbidden inputs.
   - Cons: naive odd-even network is `O(B^2)` comparators; bitonic implementation is more work.
   - Decision: **select.**

5. **Stochastic dominance / differentiable CDF dominance.**
   - Pros: elegant distribution-level ordering.
   - Cons: less directly tied to per-batch top-`k` ranking; may under-emphasize hard individual misorderings.
   - Decision: future alternative, not this packet.

6. **Order polytope residual over label poset.**
   - Pros: real order-theoretic construct.
   - Cons: with binary target, the order polytope is almost equivalent to a monotone separation constraint; adding richer relations risks using fine labels ordinally or inventing hidden metadata.
   - Decision: reject for this packet.

### Final selection sentence

Use a **differentiable sorting network order residual**: softly sort each training batch by the model's scalar logits, carry binary labels through the same soft compare-swap network, and penalize distance from the ideal top-`k` positive occupancy vector.

## 6. Rejected Approaches

### Near-puzzle twin ranking

Rejected. The objective must not depend on paired or neighboring puzzle/non-puzzle examples. The proposed residual sorts an ordinary batch; it does not construct twin pairs or rely on puzzle adjacency.

### Prototype margins

Rejected. No class prototypes, centroid margins, learned anchors, or feature-space prototype distances are part of the selected method.

### Ordinal evidence ladders

Rejected. Fine labels `0, 1, 2` are not used as an ordinal target. The loss sees `fine_label in {0,1}` as one negative target and `fine_label == 2` as positive. Any implementation that adds `fine0 < fine1 < fine2` as a training signal violates this packet.

### Source-rate calibration

Rejected. Source labels and source file IDs are forbidden inference inputs and should not be used for calibration. Do not add per-source priors, per-source thresholds, or source-balanced inference logic.

### Threshold tuning

Rejected. Validation threshold searches are not allowed. Report fixed-threshold confusion using `logit > 0`, plus threshold-free metrics such as AUROC/AUPRC.

### Engine-derived features

Rejected. Stockfish, PVs, node counts, mate scores, best moves, and verification metadata cannot enter inference. They also should not be used to create auxiliary model inputs. The model must learn from board tensors and labels.

### Pure listwise IR ranking losses

Rejected as the main idea when they require query groups, document lists, or relevance grades. The chess dataset here is not a query-document ranking dataset. The batch sorter is a training regularizer over scalar board logits, not a retrieval stack.

## 7. Mathematical Thesis

Let a minibatch contain `B` samples:

```text
x_i: current-board tensor
f_i in {0,1,2}: fine label
y_i = 1[f_i = 2] in {0,1}: binary target
s_i = s_theta(x_i) in R: model logit
```

Define a differentiable sorting network `S_tau` over the score vector `s`. The hard sorting network would produce a permutation matrix `P(s)` that sorts scores descending. The soft network produces a differentiable soft permutation-like transformation `P_tau(s)` by replacing each hard compare-swap with a soft compare-swap.

Carry the binary label vector through the same soft sorting operation:

```text
q = P_tau(s) y
```

where `q_t` is the soft amount of positive-label mass occupying sorted position `t`.

Let:

```text
k = sum_i y_i
e_k = [1, 1, ..., 1, 0, 0, ..., 0] in R^B
      first k entries are 1, remaining B-k entries are 0
```

The order residual is:

```text
L_order = (1 / B) * || q - e_k ||_2^2
```

or the numerically clipped binary cross-entropy variant:

```text
L_order_bce = mean_t BCE(clamp(q_t, eps, 1-eps), e_k,t)
```

The total loss is:

```text
L_total = BCEWithLogitsLoss(s, y; pos_weight)
        + lambda_order * L_order
```

Skip `L_order` for batches where `k == 0` or `k == B`, because there is no within-batch binary order to learn.

### Core thesis

A board-only scalar puzzle model is good not merely when positive examples cross a threshold, but when positive examples are consistently above negatives in the induced total order of its own logits. The differentiable sorting network exposes that order during training without changing the inference contract.

### Important invariance detail

Hard ranking is invariant under strictly increasing transformations of scores. Soft sorting is not perfectly scale-invariant because the temperature `tau` interacts with logit scale. Therefore compute the order residual on standardized batch scores:

```python
z = (logits - logits.mean()) / (logits.std(unbiased=False) + 1e-6)
```

Use raw logits for `BCEWithLogitsLoss` and inference. Use standardized logits only inside the order residual. This prevents the rank loss from being won by logit-scale inflation.

## 8. Order-Theoretic Object

The central object is a **differentiable sorting network**.

A hard sorting network is a fixed sequence of comparator pairs `(i, j)`. Each comparator performs:

```text
(a_i, a_j) -> (max(a_i, a_j), min(a_i, a_j))
```

for descending order. The network topology is input-independent. With enough comparators, the network sorts every input vector.

The soft version replaces each hard compare-swap with a differentiable gate. For two scores `a` and `b`:

```text
p = sigmoid((a - b) / tau)

hi_score = p * a + (1 - p) * b
lo_score = (1 - p) * a + p * b
```

The same gate is applied to the label carriers:

```text
hi_label = p * label_a + (1 - p) * label_b
lo_label = (1 - p) * label_a + p * label_b
```

As `tau -> 0`, the comparator approaches a hard descending compare-swap. For positive `tau`, the whole sorter is differentiable.

The residual is a distance from the soft label carrier vector after sorting to the ideal top-`k` chain pattern. In hard form, the ideal event is:

```text
all positive labels precede all negative labels in the total order induced by s_theta
```

This is an order statement, not a threshold statement.

Recommended network topology:

1. Start with **odd-even transposition sorting network** for implementation simplicity and arbitrary batch length.
2. Move to **bitonic sorting network** if batch sizes are large and the `O(B^2)` odd-even comparator count becomes costly.
3. Keep a `torchsort.soft_sort` or `torchsort.soft_rank` implementation as an ablation/fallback if local extension builds are reliable.

## 9. Architecture Tensor Contract

### Model

Use the existing board encoder style in `chess-nn-playground` if one exists. The handoff does not require a new chess-specific architecture. The order contribution lives in the loss.

Minimal shape:

```python
class PuzzleBinaryNet(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.backbone = BoardBackbone(in_channels)
        self.head = nn.Linear(self.backbone.out_dim, 1)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        h = self.backbone(board)
        logit = self.head(h).squeeze(-1)
        return logit
```

### Input contract

```python
batch["board"]       # FloatTensor[B, C, 8, 8]
batch["fine_label"]  # LongTensor[B], values 0/1/2
```

Allowed preprocessing:

```python
target = (fine_label == 2).float()
```

Not allowed as model inputs:

```text
stockfish_eval
pv
node_count
mate_score
best_move
verification_metadata
source_label
source_file_id
```

If the dataset loader has these fields for audit, strip them before the model call.

### Output contract

```python
logits = model(board)  # FloatTensor[B]
```

No second logit. No three-class softmax. No source head. No threshold head.

### Batch contract for order residual

The order residual is meaningful only when the batch has at least one positive and one negative:

```python
num_pos = int(target.sum().item())
if num_pos == 0 or num_pos == len(target):
    loss_order = logits.new_tensor(0.0)
```

To reduce skipped rank batches, use one of:

```text
stratified minibatches
large natural minibatches
positive oversampling inside the sampler
```

Do not change validation prevalence. Training sampler balancing is acceptable; inference and evaluation remain board-only.

## 10. Training Loss

### Loss definition

```text
L_total = L_bce + lambda_order * L_order
```

where:

```text
L_bce = BCEWithLogitsLoss(logits, target, pos_weight=pos_weight)
```

PyTorch's `BCEWithLogitsLoss` is preferred because it combines sigmoid and binary cross-entropy in a numerically stable way.

The order loss:

```text
z = standardize(logits)
q = soft_sorting_network_label_occupancy(scores=z, carriers=target, tau=tau)
e = ideal_top_k_vector(k=sum(target), B=batch_size)
L_order = mean((q - e)^2)
```

### Recommended initial hyperparameters

```yaml
lambda_order: 0.10
tau_start: 1.00
tau_end: 0.15
tau_schedule: cosine_or_linear_over_first_60_percent_of_training
order_score_standardization: true
order_loss_type: mse
skip_order_loss_if_single_class_batch: true
pos_weight: train_num_negative / train_num_positive
fixed_eval_threshold: 0.0
```

Do not tune `fixed_eval_threshold`.

### Soft odd-even sorter pseudocode

```python
import torch
import torch.nn.functional as F

def _soft_compare_swap_desc(scores, carriers, i, j, tau):
    a = scores[i]
    b = scores[j]
    ca = carriers[i]
    cb = carriers[j]

    p = torch.sigmoid((a - b) / tau)

    hi_s = p * a + (1.0 - p) * b
    lo_s = (1.0 - p) * a + p * b

    hi_c = p * ca + (1.0 - p) * cb
    lo_c = (1.0 - p) * ca + p * cb

    scores = scores.clone()
    carriers = carriers.clone()
    scores[i], scores[j] = hi_s, lo_s
    carriers[i], carriers[j] = hi_c, lo_c
    return scores, carriers


def soft_odd_even_label_occupancy_desc(logits, target, tau=0.25):
    # logits: FloatTensor[B]
    # target: FloatTensor[B], values in {0,1}
    scores = (logits - logits.mean()) / (logits.std(unbiased=False) + 1e-6)
    carriers = target.float()

    B = scores.shape[0]
    for pass_idx in range(B):
        start = pass_idx % 2
        for i in range(start, B - 1, 2):
            scores, carriers = _soft_compare_swap_desc(
                scores=scores,
                carriers=carriers,
                i=i,
                j=i + 1,
                tau=tau,
            )
    return carriers  # soft positive occupancy by descending sorted position


def soft_sort_order_residual(logits, target, tau=0.25):
    B = target.numel()
    k = int(target.sum().item())

    if k == 0 or k == B:
        return logits.new_tensor(0.0)

    q = soft_odd_even_label_occupancy_desc(logits, target, tau=tau)

    ideal = torch.zeros_like(target, dtype=logits.dtype)
    ideal[:k] = 1.0

    return F.mse_loss(q, ideal)
```

The clone-heavy pseudocode is clear but not optimized. Codex should vectorize or in-place update carefully after tests pass.

### Training step pseudocode

```python
def training_step(model, batch, cfg):
    board = batch["board"]
    fine = batch["fine_label"].long()
    target = (fine == 2).float()

    logits = model(board)

    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=cfg.pos_weight_tensor,
    )

    order = soft_sort_order_residual(
        logits=logits,
        target=target,
        tau=current_tau(cfg),
    )

    loss = bce + cfg.lambda_order * order

    metrics = {
        "loss": loss.detach(),
        "loss_bce": bce.detach(),
        "loss_order": order.detach(),
        "batch_pos_frac": target.mean().detach(),
    }
    return loss, metrics
```

### Why not use fine labels inside the order loss?

This packet deliberately avoids:

```python
rank_target = fine_label.float()
```

or:

```python
fine0 < fine1 < fine2
```

because that turns the method into an ordinal evidence ladder. The order residual should separate positives from negatives only:

```python
target = fine_label == 2
```

## 11. Ablation Plan

Run all ablations with identical train/validation/test splits, identical board tensor contract, identical fixed threshold rule, and at least three random seeds.

### A0: BCE baseline

```text
loss = BCEWithLogitsLoss(logit, target)
```

Purpose: establish the real baseline.

### A1: BCE + soft sorting order residual

```text
loss = BCE + 0.10 * L_order
```

Purpose: primary candidate.

Grid:

```yaml
lambda_order: [0.03, 0.10, 0.30]
tau_end: [0.10, 0.15, 0.25]
batch_size: [64, 128, 256]
```

Keep the first run modest:

```yaml
lambda_order: 0.10
tau_start: 1.00
tau_end: 0.15
batch_size: 128
```

### A2: BCE + soft sorting residual without score standardization

Purpose: test whether standardization is necessary. Expected result: standardized should be more stable.

### A3: BCE + pairwise AUC logistic control

```text
L_pair = mean_{i positive, j negative} log(1 + exp(-(s_i - s_j)))
```

Purpose: distinguish "any ranking helps" from "the differentiable sorting network residual helps." This is a control, not the selected idea. Do not implement near-puzzle pairs.

### A4: BCE + torchsort/permutahedron soft-rank residual

Use `torchsort.soft_rank` or `torchsort.soft_sort` if dependency setup is acceptable.

Purpose: compare sorter family:

```text
soft compare-swap network vs permutahedron/isotonic differentiable rank
```

### A5: Batch composition

Compare:

```text
natural minibatches
stratified minibatches with positives and negatives
positive oversampling in training sampler only
```

Purpose: order residual cannot learn from single-class batches. If natural batches frequently have no positives, the sampler must be adjusted for training.

### Required metrics for every ablation

```text
val/test BCE
val/test AUROC
val/test AUPRC
binary confusion at logit > 0
fine-label confusion rows 0/1/2 by predicted binary 0/1
per-fine-label predicted-positive rate
order residual diagnostic
top-k hard occupancy diagnostic
```

Hard occupancy diagnostic:

```python
# On validation, for each batch:
idx = torch.argsort(logits, descending=True)
sorted_target = target[idx]
k = int(target.sum().item())
topk_positive_fraction = sorted_target[:k].float().mean() if k > 0 else nan
```

This diagnostic is not an inference input. It is evaluation only.

### Selection rule

Prefer A1 over A0 only if it improves ranking and classification without hiding damage in fine-label confusion.

Minimum acceptable improvement:

```text
AUROC improves by >= 0.005 on average over seeds
or AUPRC improves by >= 0.010 on average over seeds
and fine 0/1 false-positive rates do not increase materially
and fine 2 recall at fixed logit > 0 does not drop
```

Use stricter thresholds if the dataset is large enough that tiny gains are not meaningful.

## 12. Falsification Criteria

Reject the idea if any of the following happen:

1. **No ranking gain.**
   - A1 does not beat BCE-only on AUROC or AUPRC over repeated seeds.

2. **Fine-label confusion gets worse.**
   - `fine_label == 1` or `fine_label == 0` false-positive rate rises materially without a compensating `fine_label == 2` recall gain.

3. **The order loss overfits batch composition.**
   - Training order residual falls, but validation AUROC/AUPRC do not improve.

4. **The method needs threshold tuning to look good.**
   - If it only improves after searching validation thresholds, reject it. Threshold tuning is outside the packet.

5. **The method becomes source-aware.**
   - Any reliance on source labels, source file IDs, verification metadata, or source rates invalidates the run.

6. **The method becomes engine-aware.**
   - Any inference dependency on Stockfish, PVs, node counts, mate scores, or best moves invalidates the run.

7. **The method drifts into an ordinal ladder.**
   - Any training objective that orders `fine0 < fine1 < fine2` invalidates the selected idea.

8. **Batch-size fragility.**
   - Results only appear for one handpicked batch size and vanish for ordinary batch sizes.

9. **Logit-scale pathology.**
   - The model inflates logit magnitude to reduce soft sorting loss while validation ranking stagnates. Standardization should prevent this; if it does not, reject or lower `lambda_order`.

10. **Compute cost is not worth it.**
    - If sorter overhead dominates training and the metric gain is marginal, keep the BCE baseline.

## 13. Implementation Notes

### Suggested file/module additions

Do not create side-data pipelines. Keep this as a loss-only change.

Potential module names:

```text
src/chess_nn_playground/losses/soft_sort_order_residual.py
src/chess_nn_playground/metrics/fine_label_confusion.py
configs/puzzle_binary_soft_sort_order.yaml
```

If the repository naming convention differs, adapt names while preserving the idea.

### Unit tests

Create small deterministic tests.

#### Test 1: already correct order

```python
logits = torch.tensor([3.0, 2.0, 1.0])
target = torch.tensor([1.0, 0.0, 0.0])
loss = soft_sort_order_residual(logits, target, tau=0.05)
assert loss.item() < 1e-2
```

#### Test 2: reversed order

```python
logits = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.0, 0.0, 0.0])
loss = soft_sort_order_residual(logits, target, tau=0.05)
assert loss.item() > 0.5
```

#### Test 3: mixed two positives

```python
logits = torch.tensor([0.0, 3.0, 1.0, 2.0])
target = torch.tensor([1.0, 0.0, 1.0, 0.0])
# Positive mass should be encouraged into positions 0 and 1 after sorting.
```

#### Test 4: single-class batch skip

```python
target_all_zero = torch.zeros(8)
target_all_one = torch.ones(8)
assert soft_sort_order_residual(torch.randn(8), target_all_zero).item() == 0.0
assert soft_sort_order_residual(torch.randn(8), target_all_one).item() == 0.0
```

#### Test 5: gradient exists

```python
logits = torch.randn(16, requires_grad=True)
target = torch.cat([torch.ones(4), torch.zeros(12)])
loss = soft_sort_order_residual(logits, target, tau=0.25)
loss.backward()
assert logits.grad is not None
assert torch.isfinite(logits.grad).all()
```

### Fine-label confusion implementation

Use a direct 3-by-2 table because the model predicts binary labels:

```python
def fine_label_confusion(fine_label, pred_binary):
    table = torch.zeros(3, 2, dtype=torch.long)
    for f, p in zip(fine_label.view(-1), pred_binary.view(-1)):
        if 0 <= int(f) <= 2 and 0 <= int(p) <= 1:
            table[int(f), int(p)] += 1
    return table
```

For sklearn-style reporting, remember the convention: rows are true labels and columns are predicted labels. scikit-learn documents confusion matrix entry `C[i, j]` as the count of samples known to be in group `i` and predicted to be in group `j`.

### Optimizing the sorter

Start with readable odd-even code. Then optimize only after tests pass.

Options:

1. precompute comparator pairs for a given `B`;
2. update tensors with indexed operations instead of clone-per-comparator;
3. use bitonic comparator schedules for power-of-two padded batches;
4. use `torchsort` for soft-rank ablations, not as the first implementation.

### Padding for bitonic sorter

If switching to bitonic sort, pad to the next power of two. Padding must not add fake positives.

For descending order:

```text
pad scores with very low standardized score
pad carriers with 0
mask padded positions out of the residual
```

Do not let padding alter `k`.

### Metrics logging

Log:

```text
loss/train_bce
loss/train_order
loss/val_bce
metric/val_auroc
metric/val_auprc
metric/val_binary_confusion
metric/val_fine_confusion_3x2
metric/val_pred_pos_rate_fine0
metric/val_pred_pos_rate_fine1
metric/val_pred_pos_rate_fine2
metric/val_topk_positive_fraction
```

### Guardrails in code comments

Add a comment near the loss construction:

```python
# Contract guard:
# The order residual uses only logits and binary targets derived from fine_label == 2.
# Do not pass engine data, PVs, mate scores, source labels, verification metadata,
# best moves, or file IDs into the model or loss.
# Do not change this into a fine-label ordinal loss.
```

## 14. Prompt Maintenance

Keep this packet narrow. Future prompts or Codex edits should preserve these invariants:

```text
one current-board tensor input
one scalar logit output
binary target = fine 2 vs fine 0/1
fine labels reported in confusion analysis
no Stockfish/PV/node/mate/best-move/verification/source inference inputs
no near-puzzle twin ranking
no prototype margins
no ordinal evidence ladder
no source-rate calibration
no threshold tuning
central object remains differentiable sorting network
```

Acceptable future extensions:

```text
swap odd-even sorting network for bitonic sorting network
add torchsort soft-rank ablation
try SoftSort occupancy residual as fallback
try stochastic dominance as a separate new packet
improve sampler so order batches contain both classes
```

Unacceptable future extensions:

```text
using fine_label 1 as an ordinal halfway-positive
adding source-specific priors or source IDs
adding Stockfish or verification channels
pairing each puzzle with a near-puzzle twin
adding prototype centroids or feature-margin anchors
choosing a validation-tuned threshold
turning output into three logits
```

Final handoff instruction for Codex:

```text
Implement the Soft Sorting Order Residual Ranker as a loss-level addition to the existing puzzle_binary training loop. Preserve the board-only one-logit inference API. Train with BCE plus a differentiable sorting network top-k occupancy residual over binary targets. Report binary confusion and fine-label 3x2 confusion. Treat any use of forbidden metadata or fine-label ordinal training as a bug.
```
