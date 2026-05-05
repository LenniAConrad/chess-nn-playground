# Codex Handoff Packet: VetoSelect Positive-Claim Abstention

## 1. File Metadata

- **Filename:** `chess_nn_research_2026-04-28_0904_tuesday_new_york_vetoselect.md`
- **Created:** 2026-04-28 09:04 new_york, Tuesday
- **Project:** `chess-nn-playground`
- **Task family:** `puzzle_binary` classifier with learned abstention / selective prediction
- **Selected idea:** **VetoSelect**, a loss-coupled positive-claim reject head
- **Inference contract:** always emit at least one puzzle logit; additionally emit reject diagnostics
- **Forbidden information:** engine scores, PVs, mate scores, node counts, best moves, source labels, verification metadata
- **Allowed information:** current-board tensor plus deterministic current-board rule features only
- **Primary evaluation:** PR AUC, F1, and near-puzzle false positives at matched puzzle recall
- **Research anchors:** reject-option theory traces back to error/reject tradeoffs in Chow's formulation of optimum recognition with rejection; later work formalized binary reject-option surrogates and deep selective prediction with risk-coverage tradeoffs. See: [Chow 1970 / IBM Research](https://research.ibm.com/publications/on-optimum-recognition-error-and-reject-tradeoff), [Bartlett and Wegkamp 2008 / JMLR](https://www.jmlr.org/papers/v9/bartlett08a.html), [Geifman and El-Yaniv 2017 / NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html), [SelectiveNet 2019 / PMLR](https://proceedings.mlr.press/v97/geifman19a.html), and [Deep Gamblers 2019 / NeurIPS](https://papers.nips.cc/paper_files/paper/2019/hash/0c4b1eeb45c90b52bfb9d07943d855ab-Abstract.html).

## 2. Executive Selection

**Select VetoSelect.**

The central failure mode is not ordinary uncertainty. It is a board that contains tactical-looking evidence but is not actually a puzzle under the dataset label. A plain `puzzle_binary` classifier is forced to compress two different questions into one logit:

1. Does this board contain puzzle-like evidence?
2. Should that evidence be trusted as sufficient for a puzzle-positive claim?

VetoSelect separates those questions without adding forbidden inputs. The model emits:

- `puzzle_logit`: raw positive puzzle evidence, `z`
- `selector_logit`: learned trust / veto logit, `a`
- `selective_puzzle_logit`: final puzzle claim logit derived from a three-action selective distribution
- diagnostics: ordinary negative probability, rejected-positive-evidence probability, accepted-puzzle probability

The key originality is the **positive-claim abstention factorization**. Rejection is not an independent confidence wrapper. It only applies when the model itself sees positive puzzle evidence. Ordinary negatives remain ordinary negatives. Near-puzzle negatives are trained to become **rejected positive evidence**, not merely low-confidence negatives.

This is the right fit for the prompt because it:

- trains end-to-end on current supervised data;
- does not require ordinal evidence ladders, credal/Dirichlet heads, ensembles, calibration wrappers, conformal post-processing, source-rate calibration, phase-specialist calibration, engine labels, or threshold tuning;
- preserves an explicit puzzle logit during inference;
- gives a direct metric handle for near-puzzle false positives at matched recall.

## 3. Data Contract

### 3.1 Required training fields

Each training row must provide:

```text
board_tensor              FloatTensor[C, 8, 8] or equivalent current-board tensor
rule_features             FloatTensor[F], optional but recommended
puzzle_binary             int in {0, 1}
split_id                  train / validation / test
position_key              stable hash for deduplication and leakage checks
near_puzzle_eval          optional bool, evaluation-only negative cohort flag
```

`near_puzzle_eval` must never be fed as an input feature. If no near-puzzle flag exists, define a frozen evaluation cohort before training, for example: `puzzle_binary == 0` and a deterministic current-board tactical-texture score in a high fixed validation quantile. That cohort definition is for reporting only.

### 3.2 Allowed model inputs

Allowed:

```text
x_board = current-board tensor
x_rule  = deterministic current-board rule features
```

Good deterministic rule features include:

- side to move;
- legal move count;
- in-check flag;
- number of legal checking moves;
- number of legal captures;
- attacked-square maps or counts;
- pinned piece counts;
- hanging piece counts under deterministic attack/defense rules;
- material balance from fixed piece values;
- king-zone attack counts;
- promotion-distance features;
- castling rights and en-passant availability if already encoded in the current board state.

Forbidden:

```text
engine_score
centipawn_eval
WDL estimate
PV
mate score
node count
search depth
best move
source label
puzzle-source metadata
verification metadata
human solution metadata
post-hoc calibrated source priors
```

### 3.3 Training labels and derived targets

The only required supervised label is `puzzle_binary`.

VetoSelect derives a soft decoy target during training from:

- the current sample label;
- the model's own detached raw puzzle evidence logit;
- optional deterministic current-board tactical-texture features.

No source label, engine artifact, best move, or verification metadata is used.

### 3.4 Leakage control

Deduplicate by `position_key`, preferably including side to move, castling rights, en-passant state, and turn. Split before any self-mined decoy target is computed. If near-puzzle cohorts are generated by rule texture, compute quantile cutoffs on the training split and freeze them before validation/test reporting.

## 4. Selective Prediction Research Map

Selective prediction asks the classifier to trade coverage for reliability. Chow's 1970 reject-option work framed recognition performance as an error/reject tradeoff and described an optimum rejection rule under the classical decision-theoretic setup. Bartlett and Wegkamp later studied binary classification with a reject option using a convex hinge-like surrogate, giving a useful reminder that rejection should be trained as part of the decision rule, not only patched on after training.

Deep selective-classification work moved the topic into neural networks. Geifman and El-Yaniv's 2017 NeurIPS paper describes selective classification for DNNs through risk-coverage control, while SelectiveNet introduced an integrated reject option trained end-to-end rather than a threshold over a pretrained model's confidence. Deep Gamblers adds another important branch: training an explicit abstention outcome through a loss inspired by portfolio theory.

VetoSelect keeps the useful part of that map: **make rejection a learned action in the network and loss**. It rejects the pieces that conflict with this prompt:

- no validation-set confidence threshold as the main method;
- no post-hoc calibration wrapper;
- no conformal wrapper as the main contribution;
- no uncertainty-only selector;
- no `m + 1` generic reject class that can swallow positives and negatives symmetrically;
- no external labels or engine-derived uncertainty.

The research gap for `puzzle_binary` is narrower than generic selective classification: near-puzzles are not arbitrary hard examples. They are **false positive puzzle claims**. Therefore the abstention mechanism should target the positive claim, not all decisions.

## 5. Serious Candidates Rejected

### Candidate A: Vanilla SelectiveNet-style coverage head

Rejected as the primary method. It is close to the right family because it trains classification and selection jointly, but its selector is usually optimized around coverage/risk tradeoffs. For this project, the failure mode is not just risk at low coverage; it is puzzle-looking negatives. A generic coverage head can learn to reject hard positives, hard negatives, or outliers without explicitly modeling the fact that the dangerous error is an accepted positive claim on a near-puzzle.

### Candidate B: Deep-Gambler `m + 1` abstention class

Rejected as the selected method, but useful as an ablation. A generic abstention class is elegant, yet it competes directly with class logits. In a binary puzzle setting, that can make the network learn a vague third class rather than a precise veto of positive evidence. It also risks hiding whether the model saw puzzle evidence and rejected it, versus never saw puzzle evidence at all.

### Candidate C: Binary reject-option hinge loss

Rejected for implementation as the main deep model. The theory is valuable, especially because it trains reject behavior as part of the supervised objective. But a hinge reject classifier is not the easiest fit for an existing tensor backbone, PR AUC ranking, and near-puzzle false-positive diagnostics. VetoSelect keeps the reject-option cost idea but expresses it as a probabilistic hierarchical action model.

### Candidate D: Separate near-puzzle classifier

Rejected. A separate head trained directly to identify near-puzzles can work only if near-puzzle labels are reliable, stable, and not source-derived. It also risks creating a source-label shortcut. VetoSelect instead self-mines negative decoy targets from current-board evidence, so it remains trainable even when `near_puzzle_eval` exists only as a reporting cohort.

### Candidate E: Error predictor / confidence predictor

Rejected. A head trained to predict whether the puzzle classifier will be wrong often becomes an uncertainty-only wrapper. It tends to track margin, entropy, or calibration defects. The prompt explicitly asks to avoid that kind of solution. VetoSelect's reject state is trained on **positive evidence on negative examples**, not just low confidence.

## 6. Common Approaches Rejected

Do not implement these as the selected method:

- ordinal evidence ladders;
- credal or Dirichlet heads;
- phase-specialist calibration;
- source-rate calibration;
- validation threshold tuning as the main method;
- uncertainty-only wrappers based on entropy, margin, or max probability;
- ensembles;
- conformal post-processing only;
- engine-score features;
- best-move or PV features;
- mate-score or node-count features;
- source labels or verification metadata;
- post-hoc Platt/isotonic calibration as the main solution.

Some of these may be useful in other projects. They are bad fits here because the goal is to make the model learn when a **current-board positive puzzle claim** should not be trusted.

## 7. Mathematical Thesis

Let:

- `x` be the current-board input;
- `r(x)` be deterministic current-board rule features;
- `y in {0,1}` be `puzzle_binary`;
- `z = f_theta(x, r)` be the raw puzzle-evidence logit;
- `a = g_theta(x, r)` be the positive-claim selector logit;
- `sigma(t) = 1 / (1 + exp(-t))`.

A normal binary classifier estimates one thing:

```text
P(y = 1 | x) ≈ sigma(z)
```

That is too compressed for near-puzzles. The model needs to represent:

```text
positive-looking evidence exists
positive-looking evidence should be trusted
```

VetoSelect factorizes the accepted puzzle claim:

```text
P(accepted puzzle | x) = P(positive evidence | x) * P(trust evidence | positive evidence, x)
                      = sigma(z) * sigma(a)
```

Then it allocates probability mass to three mutually exclusive actions:

```text
pi_N(x) = P(ordinary non-puzzle)          = sigma(-z)
pi_R(x) = P(rejected positive evidence)  = sigma(z) * sigma(-a)
pi_P(x) = P(accepted puzzle)             = sigma(z) * sigma(a)
```

These sum to one:

```text
pi_N + pi_R + pi_P = sigma(-z) + sigma(z) * [sigma(-a) + sigma(a)] = 1
```

The thesis is:

> Near-puzzle false positives are best reduced by giving the model a trained state for `positive evidence present but not claim-worthy`, rather than forcing every negative to look like an ordinary negative.

This lets a hard negative have high tactical/evidence activation `z` while still being rejected through low selector logit `a`.

## 8. Selection Mechanism

### 8.1 Inference outputs

For each board, emit:

```text
puzzle_logit                  z
selector_logit                a
log_prob_nonpuzzle            log pi_N
log_prob_rejected_evidence    log pi_R
log_prob_accepted_puzzle      log pi_P
selective_puzzle_logit        log(pi_P) - log(pi_N + pi_R)
reject_positive_logit         log(pi_R) - log(pi_N)
```

`puzzle_logit = z` satisfies the prompt's puzzle-logit requirement. `selective_puzzle_logit` is the score to use for PR AUC and F1 of the selective classifier.

### 8.2 Decision rule

Use argmax over the three action probabilities:

```text
if pi_P is largest: emit puzzle-positive
if pi_R is largest: abstain / reject positive puzzle evidence
if pi_N is largest: emit non-puzzle
```

There is no tuned confidence threshold. The zero point of logits is simply the built-in argmax boundary between learned actions.

### 8.3 Why this is not a calibration wrapper

The selector is not applied after a frozen classifier. It is trained jointly with the puzzle evidence head. Its training target is not confidence, entropy, or source rate. It receives a specific supervised signal: when a negative board receives puzzle-like evidence, move probability mass to `rejected positive evidence` instead of `accepted puzzle`.

### 8.4 Why this targets near-puzzles

A near-puzzle negative can land in this state:

```text
sigma(z) high      # board looks tactically puzzle-like
sigma(a) low       # evidence is not trusted as a puzzle claim
pi_R high          # reject positive evidence
pi_P low           # avoid false positive
```

An ordinary negative lands here:

```text
sigma(z) low
pi_N high
```

A true puzzle lands here:

```text
sigma(z) high
sigma(a) high
pi_P high
```

## 9. Architecture Tensor Contract

### 9.1 Input tensors

Recommended model signature:

```python
forward(
    board: FloatTensor[B, C, 8, 8],
    rule_features: FloatTensor[B, F] | None = None,
) -> dict[str, Tensor]
```

No history tensor, engine tensor, PV tensor, best-move tensor, source-ID embedding, or verification feature is allowed.

### 9.2 Backbone

Use the existing `puzzle_binary` backbone if available. Otherwise:

```text
board_encoder:
  Conv2d(C -> 64, 3x3, padding=1)
  4-8 residual blocks, width 64 or 96
  global average pool
  board_embedding: FloatTensor[B, D_b]

rule_encoder:
  LayerNorm(F)
  Linear(F -> 64)
  GELU
  Linear(64 -> D_r)

fusion:
  concat(board_embedding, rule_embedding)
  Linear(D_b + D_r -> D)
  GELU
  Dropout(p=0.05 to 0.10)
```

If no rule features exist yet, set `D_r = 0` and run the board-only version. The method still works, but the near-puzzle decoy signal is weaker.

### 9.3 Heads

```text
evidence_head:
  Linear(D -> 1) -> z

selector_head:
  Linear(D -> 1) -> a
```

Optional but useful:

```text
texture_projection:
  deterministic scalar m in [0,1] from rule features only
```

The texture projection is not an extra learned classifier. It is a deterministic normalization of rule features used to weight self-mined negative decoys during training.

### 9.4 Output dictionary

```python
{
    "puzzle_logit": z,                         # [B]
    "selector_logit": a,                       # [B]
    "log_prob_nonpuzzle": log_pi_n,            # [B]
    "log_prob_rejected_evidence": log_pi_r,    # [B]
    "log_prob_accepted_puzzle": log_pi_p,      # [B]
    "selective_puzzle_logit": z_sel,           # [B]
    "reject_positive_logit": z_reject,         # [B]
}
```

Compute `z_sel` exactly:

```text
z_sel = log(pi_P) - log(pi_N + pi_R)
```

Do not replace `puzzle_logit` with `z_sel`. Keep both. The raw logit is diagnostic evidence; the selective logit is the actual positive-claim score.

## 10. Training Loss

### 10.1 Hierarchical log probabilities

Use numerically stable log-sigmoid operations:

```python
log_pi_n = logsigmoid(-z)
log_pi_r = logsigmoid(z) + logsigmoid(-a)
log_pi_p = logsigmoid(z) + logsigmoid(a)
```

### 10.2 Self-mined decoy target

For positives:

```text
d = 0
q_N = 0
q_R = 0
q_P = 1
```

For negatives, define a soft decoy target `d in [0, d_max]`:

```text
e = stop_gradient(sigma(z / tau_e))
m = deterministic_tactical_texture(rule_features) in [0, 1]
d = clamp(e * m, 0, d_max)
q_N = 1 - d
q_R = d
q_P = 0
```

Recommended defaults:

```text
tau_e = 1.5
d_max = 0.85
warmup_epochs = 2
```

During warmup, set `d = 0` and train a normal binary evidence head. After warmup, enable self-mined decoys. `stop_gradient` is important: the model should not learn to reduce `d` by manipulating the evidence logit inside the target computation.

If deterministic rule features are not ready, set `m = 1` for all negatives as a first implementation. The stronger version uses rule texture so that tactical-looking negatives receive more reject supervision than quiet negatives.

### 10.3 Texture score suggestion

A simple deterministic texture score can be:

```text
m_raw =
  0.25 * has_legal_check
+ 0.20 * clipped_count_legal_checks
+ 0.15 * clipped_count_captures
+ 0.15 * clipped_king_zone_attack_count
+ 0.10 * has_pinned_piece
+ 0.10 * hanging_piece_count_clipped
+ 0.05 * promotion_pressure_flag

m = clip(m_raw, 0, 1)
```

This is not an evidence ladder for the final classifier. It is only a training-time weight for deciding which negative examples are plausible decoys. It uses no engine, PV, best move, or source metadata.

### 10.4 Three-action target loss

For each example:

```text
L_tri = - q_N * log_pi_n - q_R * log_pi_r - q_P * log_pi_p
```

Use class weights if the dataset is imbalanced:

```text
w_i = pos_weight              if y_i = 1
w_i = 1 + gamma_decoy * d_i   if y_i = 0
```

Recommended first pass:

```text
pos_weight = num_neg / max(num_pos, 1), capped at 20
gamma_decoy = 1.0
```

### 10.5 Evidence anchor loss

Add a small anchor to preserve a usable raw puzzle logit while not crushing decoy negatives into ordinary negatives:

```text
L_anchor = BCEWithLogits(z, y) * [ y + (1 - y) * (1 - d) ]
```

This means:

- positives still push `z` up;
- ordinary negatives push `z` down;
- decoy negatives are allowed to keep positive-looking evidence and instead learn rejection through `a`.

Recommended weight:

```text
lambda_anchor = 0.10 to 0.25
```

### 10.6 Final objective

```text
L = mean(w_i * L_tri) + lambda_anchor * mean(L_anchor) + weight_decay
```

Do not add a validation-tuned acceptance threshold. Do not add source-rate calibration. Do not add a conformal wrapper as the main result.

### 10.7 Minimal PyTorch sketch

```python
import torch
import torch.nn.functional as F


def veto_select_loss(z, a, y, texture, *, tau_e=1.5, d_max=0.85,
                     pos_weight=1.0, gamma_decoy=1.0,
                     lambda_anchor=0.15, enable_decoys=True):
    y = y.float()
    texture = torch.clamp(texture.float(), 0.0, 1.0)

    log_pi_n = F.logsigmoid(-z)
    log_pi_r = F.logsigmoid(z) + F.logsigmoid(-a)
    log_pi_p = F.logsigmoid(z) + F.logsigmoid(a)

    if enable_decoys:
        evidence = torch.sigmoid(z.detach() / tau_e)
        d = ((1.0 - y) * evidence * texture).clamp(0.0, d_max)
    else:
        d = torch.zeros_like(y)

    q_n = (1.0 - y) * (1.0 - d)
    q_r = (1.0 - y) * d
    q_p = y

    l_tri = -(q_n * log_pi_n + q_r * log_pi_r + q_p * log_pi_p)

    weights = torch.where(
        y > 0.5,
        torch.full_like(y, float(pos_weight)),
        1.0 + gamma_decoy * d,
    )

    anchor_mask = y + (1.0 - y) * (1.0 - d)
    l_anchor = F.binary_cross_entropy_with_logits(z, y, reduction="none") * anchor_mask

    return (weights * l_tri).mean() + lambda_anchor * l_anchor.mean()
```

## 11. Diagnostic Outputs

Log these for every validation/test run:

```text
raw_pr_auc_on_puzzle_logit
selective_pr_auc_on_selective_puzzle_logit
raw_best_f1
selective_best_f1
raw_f1_at_zero
selective_f1_at_argmax
near_puzzle_fp_at_recall_0_70
near_puzzle_fp_at_recall_0_80
near_puzzle_fp_at_recall_0_90
near_puzzle_fp_at_baseline_recall
ordinary_negative_fp_at_matched_recall
positive_reject_rate_at_argmax
negative_reject_rate_at_argmax
near_negative_reject_rate_at_argmax
mean_selector_logit_positive
mean_selector_logit_near_negative
mean_selector_logit_ordinary_negative
```

Matched-recall protocol:

1. Pick a recall target `R`, either a fixed target such as `0.80` or the baseline model's recall at its chosen operating point.
2. On the validation split, find the score threshold for each model that reaches recall `R` on true puzzles.
3. Apply that threshold to the test split.
4. Report false positives among `near_puzzle_eval == true` and `puzzle_binary == 0`.
5. Also report ordinary negative false positives, because a method that only shifts errors from one negative cohort to another is not good enough.

Use `selective_puzzle_logit` for the selected method's ranking score. Keep raw `puzzle_logit` metrics as a diagnostic to verify that the selector is doing real work.

## 12. Ablations

Run these in order:

### A0: Existing baseline

Current `puzzle_binary` classifier with one puzzle logit and standard BCE/focal loss.

Purpose: establish PR AUC, F1, and near-puzzle false positives at matched recall.

### A1: VetoSelect without decoys

Use the three-action output but set `d = 0` for all negatives.

Purpose: tests whether the architecture alone helps. Expected result: small or no near-puzzle benefit.

### A2: VetoSelect with self-mined decoys, no rule texture

Set `m = 1` for all negatives.

Purpose: tests whether detached raw positive evidence is enough to find hard negatives.

### A3: VetoSelect with self-mined decoys and deterministic rule texture

Full selected method.

Purpose: expected best near-puzzle false-positive reduction.

### A4: Remove evidence anchor

Set `lambda_anchor = 0`.

Purpose: checks whether raw `puzzle_logit` remains meaningful without anchoring.

### A5: Final score comparison

Compare ranking by:

```text
z                         raw puzzle logit
z + a                     simple product-of-experts score
log(pi_P) - log(1-pi_P)   exact selective puzzle logit
pi_P                      accepted-puzzle probability
```

Purpose: choose the cleanest reporting score. The selected score should be the exact selective puzzle logit unless evidence strongly says otherwise.

### A6: Rule features removed

Train board-only VetoSelect.

Purpose: determine whether deterministic rule features are necessary for near-puzzle reduction.

### A7: Generic abstention class baseline

Train a three-class non-puzzle / puzzle / reject model or Deep-Gambler-style abstention loss.

Purpose: demonstrate that positive-claim abstention is more targeted than generic abstention.

### A8: Confidence-only rejection baseline

Freeze or train the classifier, then reject by margin/entropy/max probability.

Purpose: verify that VetoSelect is not merely rediscovering uncertainty thresholding.

## 13. Falsification Criteria

Reject VetoSelect as the selected method if any of these hold on a clean test split:

1. **No near-puzzle gain:** near-puzzle false positives at matched recall are not reduced versus the baseline by a meaningful margin. A practical first bar is at least `15%` relative reduction at recall `0.80`, or a statistically clear reduction over bootstrap confidence intervals.
2. **PR collapse:** selective PR AUC drops by more than `2%` absolute versus the baseline without a compensating near-puzzle FP reduction.
3. **F1 collapse:** best F1 or fixed-argmax F1 drops materially versus baseline. A small drop can be acceptable only if near-puzzle false positives fall strongly at the same recall.
4. **Rejects true puzzles instead of near-puzzles:** positive reject rate is high while near-negative reject rate is low. The selector must not solve the problem by abstaining on true puzzles.
5. **Uncertainty-wrapper behavior:** `selector_logit` is almost perfectly explained by raw margin or entropy. Operational test: train a linear probe from `abs(z)` or `sigma(z)` to `a`; if it explains nearly all selector variance and near-puzzle FP gains disappear at matched recall, the method has degraded into an uncertainty wrapper.
6. **Texture shortcut:** gains disappear when evaluation near-puzzles are defined independently of the texture score used in training. That would mean the method learned the texture proxy, not trustworthiness.
7. **Cohort transfer failure:** improvement exists only on the validation near-puzzle cohort but not on a held-out near-puzzle cohort generated by a different deterministic definition or split.
8. **Ordinary FP explosion:** near-puzzle false positives drop only because false positives move into ordinary negatives. Ordinary-negative FP must be reported at the same matched recall.

## 14. Codex Implementation Notes

### 14.1 Minimal implementation path

1. Locate the current `puzzle_binary` model.
2. Keep the existing board encoder.
3. Add `selector_head` parallel to the existing puzzle head.
4. Add hierarchical probability computation.
5. Add `VetoSelectLoss` with warmup.
6. Add metrics for `puzzle_logit` and `selective_puzzle_logit`.
7. Add near-puzzle false-positive reporting at matched recall.
8. Verify no forbidden columns are loaded into the model input.

### 14.2 Model pseudocode

```python
class VetoSelectPuzzleNet(nn.Module):
    def __init__(self, board_encoder, rule_encoder, hidden_dim):
        super().__init__()
        self.board_encoder = board_encoder
        self.rule_encoder = rule_encoder
        self.fuse = nn.Sequential(
            nn.Linear(board_encoder.out_dim + rule_encoder.out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        self.evidence_head = nn.Linear(hidden_dim, 1)
        self.selector_head = nn.Linear(hidden_dim, 1)

    def forward(self, board, rule_features=None):
        hb = self.board_encoder(board)
        hr = self.rule_encoder(rule_features) if rule_features is not None else hb.new_zeros((hb.size(0), 0))
        h = self.fuse(torch.cat([hb, hr], dim=-1))

        z = self.evidence_head(h).squeeze(-1)
        a = self.selector_head(h).squeeze(-1)

        log_pi_n = F.logsigmoid(-z)
        log_pi_r = F.logsigmoid(z) + F.logsigmoid(-a)
        log_pi_p = F.logsigmoid(z) + F.logsigmoid(a)

        log_not_p = torch.logaddexp(log_pi_n, log_pi_r)
        selective_puzzle_logit = log_pi_p - log_not_p
        reject_positive_logit = log_pi_r - log_pi_n

        return {
            "puzzle_logit": z,
            "selector_logit": a,
            "log_prob_nonpuzzle": log_pi_n,
            "log_prob_rejected_evidence": log_pi_r,
            "log_prob_accepted_puzzle": log_pi_p,
            "selective_puzzle_logit": selective_puzzle_logit,
            "reject_positive_logit": reject_positive_logit,
        }
```

### 14.3 Warmup schedule

```text
epochs 0-1:
  train evidence head and selector with d = 0
  include anchor loss

epochs 2+:
  enable self-mined decoys
  compute d from detached z and deterministic texture
```

The warmup should be short. The point is only to prevent random early logits from defining decoys.

### 14.4 Matched recall metric sketch

```python
def threshold_for_recall(y_true, score, recall_target):
    y_true = np.asarray(y_true).astype(bool)
    score = np.asarray(score)
    pos_scores = np.sort(score[y_true])[::-1]
    if len(pos_scores) == 0:
        raise ValueError("no positives")
    k = int(np.ceil(recall_target * len(pos_scores))) - 1
    k = np.clip(k, 0, len(pos_scores) - 1)
    return pos_scores[k]


def near_fp_at_recall(y_true, score, near_mask, recall_target):
    thr = threshold_for_recall(y_true, score, recall_target)
    pred_pos = score >= thr
    near_neg = (~y_true.astype(bool)) & near_mask.astype(bool)
    return {
        "threshold": float(thr),
        "near_fp_count": int((pred_pos & near_neg).sum()),
        "near_fp_rate": float((pred_pos & near_neg).sum() / max(near_neg.sum(), 1)),
    }
```

Find thresholds on validation and report on test to avoid optimistic reporting. PR AUC itself should not use a fixed threshold.

### 14.5 Guardrails for Codex

- Do not read engine columns, even if present.
- Do not feed source labels to the model.
- Do not tune a reject threshold on validation as the selected mechanism.
- Do not replace this with conformal prediction.
- Do not implement ensembles.
- Do not remove the raw `puzzle_logit` from inference output.
- Do not report near-puzzle FP without matched recall.
- Do not claim success from lower false positives if recall fell.

## 15. Future Prompt Notes

Use this follow-up prompt for implementation:

```text
Implement VetoSelect Positive-Claim Abstention for chess-nn-playground.

Requirements:
- keep current-board tensor inputs only;
- allow deterministic current-board rule features;
- forbid engine scores, PVs, mate scores, node counts, best moves, source labels, and verification metadata;
- add puzzle_logit and selector_logit heads;
- compute pi_N, pi_R, pi_P as a hierarchical selective distribution;
- train with the self-mined decoy target loss from the handoff packet;
- output raw puzzle_logit, selective_puzzle_logit, selector_logit, and reject diagnostics;
- evaluate PR AUC, F1, and near-puzzle false positives at matched recall;
- include ablations A0-A8;
- do not add calibration wrappers, threshold tuning, conformal post-processing, ensembles, credal/Dirichlet heads, or ordinal evidence ladders.
```

The first empirical question is not whether VetoSelect abstains often. The real question is whether it moves near-puzzle negatives from `accepted puzzle` to `rejected positive evidence` while keeping true-puzzle recall matched to the baseline.
