# Codex Handoff Packet: Conditional Surprisal Gate

## 1. File Metadata

- **Filename:** `chess_nn_research_2026-04-28_0858_tuesday_new_york_conditional_surprisal_gate.md`
- **Generated:** 2026-04-28 08:58 new_york, Tuesday
- **Target repo:** `chess-nn-playground`
- **Task:** `puzzle_binary` classification
- **Idea name:** Conditional Surprisal Gate, abbreviated `csg`
- **Primary model class:** `ConditionalSurprisalGatePuzzleNet`
- **Primary loss helper:** `conditional_surprisal_gate_loss`
- **Recommended config key:** `model.name: conditional_surprisal_gate`
- **Production inference output:** exactly one binary puzzle logit per board position
- **Training labels:** fine labels `0 = non-puzzle`, `1 = near-puzzle hard negative`, `2 = verified puzzle`
- **Binary target:** `target = 1[fine_label == 2]`; fine labels `0` and `1` map to binary `0`
- **Mandatory report artifact:** 3x2 count matrix with rows `fine_label in {0,1,2}` and columns `predicted_binary in {0,1}`

## 2. Executive Selection

Build **Conditional Surprisal Gate**, a chess puzzle classifier whose only predictive path is a small stochastic bit vector. The posterior sees the full board through a relational square encoder. The prior sees only cheap low-order board statistics, such as per-plane means and maxima. Training charges the classifier for every bit of information in the posterior that cannot already be predicted from those low-order statistics.

The selected mechanism is an **information-theoretic conditional rate bottleneck**:

```text
full board tensor x
    -> relational posterior q_phi(z | x) over 48 Bernoulli gates
low-order stats c = s(x)
    -> conditional prior p_psi(z | c)
z
    -> binary puzzle logit
loss = BCEWithLogits(logit, 1[fine == 2])
       + beta * KL(q_phi(z | x) || p_psi(z | s(x)))
       + capacity hinge
```

The intended behavior is direct: material balance, piece counts, side-to-move constants, and other low-order board signatures should be cheap because the conditional prior can already predict them. The model must pay rate only for extra high-order relational information that helps separate verified puzzles from both ordinary non-puzzles and near-puzzle hard negatives. That makes fine label `1` useful without turning the task into an ordinal ladder.

This is not a reconstruction model, not an engine surrogate, not a calibration model, and not a generic uncertainty head. It is a binary classifier whose training pressure is: **keep only the minimum conditional board information needed to identify verified puzzles.**

## 3. Problem Restatement And Data Contract

The repo needs one classifier for `puzzle_binary`. Each example provides a current-board tensor and a fine label. The model must consume only the tensor representation of the current chess position, such as `simple_18` or `lc0_static_112`, and emit one binary logit.

**Allowed inference input**

```python
x: torch.Tensor  # shape [B, C, 8, 8], dtype float32/float16/bfloat16
```

where `C` is the channel count of the selected board tensor family. Expected values are `C = 18` for `simple_18` and `C = 112` for `lc0_static_112`, but the module should accept any configured `in_channels`.

**Forbidden inference inputs**

The model, validation code, exported checkpoint, and inference interface must not consume:

- Stockfish scores
- principal variations
- mate scores
- node counts
- verification metadata
- source labels
- engine best moves
- source file identity
- dataset provenance

**Training label contract**

```python
fine_label: torch.LongTensor  # shape [B], values in {0, 1, 2}
target = (fine_label == 2).float()
```

Fine label `1` is a hard negative. It is not an intermediate positive class and must not be trained as "partly puzzle."

**Prediction contract**

```python
logit = model(x)                 # shape [B]
prob = torch.sigmoid(logit)      # optional metric-time probability
pred = (logit >= threshold).long()
```

Use `threshold = 0.0` for default reports unless a validation-tuned threshold is explicitly stored in the experiment config. If a threshold is tuned, the 3x2 matrix must state that threshold.

**Mandatory 3x2 matrix**

Rows are fine labels. Columns are predicted binary labels.

```text
                 predicted_0    predicted_1
fine_label_0         n_00           n_01
fine_label_1         n_10           n_11
fine_label_2         n_20           n_21
```

Implementation:

```python
def fine_vs_pred_matrix(fine_label: torch.Tensor, logit: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    pred = (logit >= threshold).long()
    matrix = torch.zeros(3, 2, dtype=torch.long, device=fine_label.device)
    for f, p in zip(fine_label.long(), pred):
        matrix[f, p] += 1
    return matrix
```

## 4. Research Map With Citations

The design is anchored in the information bottleneck principle: learn a short code for input `X` that preserves information relevant to target `Y` while discarding excess input information. Tishby, Pereira, and Bialek formalized this as compression of `X` under a prediction-relevance constraint, which is the conceptual basis for charging the chess classifier for retained board information rather than merely regularizing weights. [Tishby et al., "The Information Bottleneck Method"][ref-tishby]

Deep Variational Information Bottleneck showed how to train a neural bottleneck by using a variational approximation and reparameterized stochastic representations. Conditional Surprisal Gate keeps that supervised compression idea but specializes it to chess by using a conditional prior that can explain low-order board statistics, forcing the posterior to spend bits on relational board structure. [Alemi et al., "Deep Variational Information Bottleneck"][ref-vib]

The Conditional Entropy Bottleneck and the Minimum Necessary Information framing support the idea that robust supervised representations should retain no more information than the task requires. This packet adapts that principle to puzzle detection: verified-puzzle evidence should survive, but dataset artifacts and easy nuisance features should be expensive unless they genuinely predict the binary target. [Fischer, "The Conditional Entropy Bottleneck"][ref-ceb]

Conditional mutual-information analysis gives the useful inequality family behind this design. For a learned representation `Z`, minimizing conditional input information can be a tighter way to reduce nuisance information than minimizing unconditional `I(Z;X)`. Here, the low-order statistic `C=s(X)` plays the role of "already-explained board information," and the KL term is a variational upper bound on `I(X;Z | C)`. [Tezuka and Namekawa, "Information Bottleneck Analysis by a Conditional Mutual Information Bound"][ref-cmi]

The stochastic gates use Binary Concrete / Gumbel-Softmax style relaxations, which make discrete latent variables trainable with gradient descent. This matters because the bottleneck should behave like a small number of bits, not a wide continuous vector that can leak arbitrary board detail. [Maddison et al., "The Concrete Distribution"][ref-concrete] [Jang et al., "Categorical Reparameterization with Gumbel-Softmax"][ref-gumbel]

The output loss should use logits directly. PyTorch documents `BCEWithLogitsLoss` as combining sigmoid and binary cross entropy in a numerically stable way via the log-sum-exp trick. [PyTorch `BCEWithLogitsLoss`][ref-bce]

The mandatory 3x2 report follows the standard confusion-matrix convention: matrix entry `C[i,j]` counts examples with true group `i` predicted as group `j`; here the true group is the fine label and the predicted group is binary. [scikit-learn `confusion_matrix`][ref-confusion]

The packet intentionally supports `lc0_static_112`-style input because Leela Chess Zero documents 112 input planes of size 8x8 for its neural network topology. This citation is only for tensor shape precedent, not for policy/value heads, search, or engine-derived supervision. [Leela Chess Zero neural network topology][ref-lc0]

[ref-tishby]: https://research.google/pubs/the-information-bottleneck-method/
[ref-vib]: https://research.google/pubs/deep-variational-information-bottleneck/
[ref-ceb]: https://www.mdpi.com/1099-4300/22/9/999
[ref-cmi]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8391358/
[ref-concrete]: https://openreview.net/forum?id=S1jE5L5gl
[ref-gumbel]: https://research.google/pubs/categorical-reparameterization-with-gumbel-softmax/
[ref-bce]: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
[ref-confusion]: https://sklearn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
[ref-lc0]: https://lczero.org/dev/old/nn/

## 5. Common Approaches Rejected

| Rejected approach | Reason for rejection |
|---|---|
| Masked-board codec or board reconstruction pretext | Forbidden by the task, and it optimizes recoverability of board contents rather than binary puzzle evidence. |
| Pseudo-likelihood / MDL ratio model | Forbidden by the task, and it would reward compressibility or anomaly-like signals rather than verified-puzzle separability. |
| Ordinal evidence ladder over fine labels `0 < 1 < 2` | Forbidden by the task shape. Fine label `1` is a hard negative, not an intermediate positive state. |
| Credal, Dirichlet, evidential, or subjective-logic head | Forbidden family. It mainly changes uncertainty semantics while leaving the board representation unconstrained. |
| Calibration-only method, including temperature scaling and post-hoc Platt scaling | Forbidden family. Calibration can be evaluated later, but it is not a classifier design and does not solve hard-negative discrimination. |
| Generic uncertainty or variance head | Forbidden family. "The model is uncertain" is not a puzzle logit and can become a proxy for dataset noise. |
| Engine-score distillation | Uses forbidden inference/training signals if Stockfish scores, mate scores, PVs, nodes, or best moves are injected. |
| Multi-task best-move or PV prediction | Violates the spirit of the data contract and risks smuggling engine best-move supervision into the classifier. |
| Source/provenance classifier with binary relabeling | Explicitly forbidden. It would overfit source identity and fail under distribution shift. |
| Plain ResNet or square Transformer with BCE only | Valid baseline, but not selected because it has no information-theoretic bottleneck and can memorize superficial board/source artifacts. |
| Handcrafted tactical motif detector as the main model | Too brittle and likely to encode human-chosen move concepts. It also risks requiring legal-move or engine-best-move machinery outside the board tensor. |
| Contrastive pairing by source file or puzzle collection | Rejected because source identity and dataset provenance are forbidden. |

Serious candidate mechanisms rejected before selecting Conditional Surprisal Gate:

| Candidate mechanism | Why it was serious | Why it was rejected |
|---|---|---|
| MINE-style adversarial estimator for `I(Z;X | Y)` | Directly estimates mutual information and is theoretically appealing. | Adds a nested critic, unstable minimax training, and extra hyperparameters. It is too likely to fail silently on small chess batches. |
| InfoNCE square-pair bottleneck | Could force representations to capture long-range square interactions. | Negative sampling can reward dataset shortcuts, and maximizing MI between square views is not the same as minimizing nuisance information for `puzzle_binary`. |
| Total-correlation penalty across latent factors | Encourages independent latent bits and has a clean information-theoretic name. | Chess tactics often require correlated piece interactions; penalizing correlation can destroy the signal the classifier needs. |
| Gaussian VIB latent with diagonal normal prior | Easy to implement and well known. | A continuous latent can leak substantial board detail through real-valued coordinates. The selected bit gate gives a more concrete capacity story. |
| Label-conditioned prior `p(z | c, y)` | Would make the conditional-rate math neat during training. | `y` is unknown at inference; using it would create a train/inference mismatch and invite label leakage. |
| Entropy maximization of the output probability | Simple and superficially information-theoretic. | It is an uncertainty/calibration method, not a bottleneck on board information. |

## 6. Mathematical Thesis

Let:

```text
X = current-board tensor, shape [C, 8, 8]
F = fine label in {0, 1, 2}
Y = 1[F == 2]
C = s(X), a low-order statistic vector derived only from X
Z = K-bit stochastic bottleneck representation
L = binary puzzle logit
```

The model defines:

```text
q_phi(z | x) = product_k Bernoulli(pi_k(x))
p_psi(z | c) = product_k Bernoulli(rho_k(c))
L = h_theta(z)
```

The selected default is `K = 48` gates.

The training objective is:

```text
min_{theta, phi, psi} E[BCEWithLogits(h_theta(Z), Y)]
                    + beta * E[KL(q_phi(Z | X) || p_psi(Z | s(X)))]
                    + gamma * max(0, E[R_bits] - C_max)^2
```

where:

```text
R_bits(x) = KL(q_phi(Z | x) || p_psi(Z | s(x))) / log(2)
C_max = 12.0 bits
beta = 0.003
gamma = 0.05
```

For factorized Bernoulli gates:

```text
KL(q || p) = sum_k [
    pi_k * log(pi_k / rho_k)
    + (1 - pi_k) * log((1 - pi_k) / (1 - rho_k))
]
```

with `pi_k` and `rho_k` clamped to `[1e-6, 1 - 1e-6]`.

The variational reason this is an information bottleneck is:

```text
E_{x,c}[KL(q_phi(z | x) || p_psi(z | c))]
= I_q(X; Z | C) + E_c[KL(q_phi(z | c) || p_psi(z | c))]
>= I_q(X; Z | C)
```

So the KL term is an upper bound on the conditional mutual information between the board and the bottleneck after low-order board statistics are known.

The chess-specific thesis is:

> Verified puzzles are not just "weird positions." They are positions where a small amount of high-order relational board information changes the answer. Near-puzzle hard negatives share many superficial cues with true puzzles, so the classifier should be charged for every extra bit it retains beyond cheap board statistics. The useful bits should become tactical-surprisal bits.

This thesis makes a falsifiable prediction: compared with a same-parameter no-rate deterministic gate, Conditional Surprisal Gate should reduce false positives in fine-label row `1` without destroying row `2` recall, while using a measurable but small average rate.

## 7. Architecture Specification

### 7.1 Module overview

```text
ConditionalSurprisalGatePuzzleNet
├── EasyStatsPrior
│   └── consumes x through low-order pooling only
├── RelationalPosterior
│   ├── ConvStem
│   ├── ResidualConvBlock x 2
│   ├── SquareTokenProjection
│   └── RelativeSquareAttentionBlock x 2
├── BinaryConcreteGate
└── GateOnlyClassifierHead
```

### 7.2 Default hyperparameters

```yaml
model:
  name: conditional_surprisal_gate
  in_channels: 18          # override to 112 for lc0_static_112
  conv_channels: 64
  d_model: 128
  n_heads: 8
  conv_blocks: 2
  rel_attention_blocks: 2
  k_bits: 48
  head_hidden: 96
  prior_hidden: 128
  posterior_hidden: 256
  dropout: 0.10
  tau_start: 1.20
  tau_final: 0.45
  tau_anneal_epochs: 20
  hard_straight_through: true
loss:
  beta_rate: 0.003
  gamma_capacity: 0.05
  c_max_bits: 12.0
  pos_weight: auto_neg_over_pos
```

### 7.3 EasyStatsPrior

Purpose: estimate `p_psi(z | c)` from low-order board information that should not require paid relational bits.

Input:

```python
x: FloatTensor[B, C, 8, 8]
```

Statistics:

```python
mean_stats = x.mean(dim=(-2, -1))       # [B, C]
max_stats = x.amax(dim=(-2, -1))        # [B, C]
c = torch.cat([mean_stats, max_stats], dim=-1)  # [B, 2*C]
```

Prior network:

```text
Linear(2*C, 128)
SiLU
Dropout(0.10)
Linear(128, 48)
```

Output:

```python
p_logits: FloatTensor[B, 48]
rho = sigmoid(p_logits)
```

Constraint: do not add convolution, attention, legal-move generation, or square-coordinate features to this branch. It must remain a weak low-order explainer.

### 7.4 RelationalPosterior

Purpose: estimate `q_phi(z | x)` from full board relationships.

Input:

```python
x: FloatTensor[B, C, 8, 8]
```

Stem:

```text
Conv2d(C, 64, kernel_size=3, padding=1, bias=False)
GroupNorm(8, 64)
SiLU
ResidualConvBlock(64) x 2
Conv2d(64, 128, kernel_size=1)
```

Tokenization:

```python
tokens = feat.flatten(2).transpose(1, 2)   # [B, 64, 128]
tokens = tokens + learned_square_embedding # [1, 64, 128]
```

Relative square attention block, repeated twice:

```text
LayerNorm(128)
Multi-head self-attention over 64 squares, 8 heads
Learned relative rank/file bias table: [8, 15, 15]
Residual
LayerNorm(128)
MLP: Linear(128, 256) -> GELU -> Dropout(0.10) -> Linear(256, 128)
Residual
```

Pooling:

```python
mean_pool = tokens.mean(dim=1)         # [B, 128]
max_pool = tokens.amax(dim=1)          # [B, 128]
posterior_context = cat([mean_pool, max_pool], dim=-1)  # [B, 256]
```

Posterior network:

```text
Linear(256, 256)
SiLU
Dropout(0.10)
Linear(256, 48)
```

Output:

```python
q_logits: FloatTensor[B, 48]
pi = sigmoid(q_logits)
```

### 7.5 BinaryConcreteGate

Training sample:

```python
u = torch.rand_like(q_logits).clamp(1e-6, 1.0 - 1e-6)
gumbel_logistic = torch.log(u) - torch.log1p(-u)
z_soft = torch.sigmoid((q_logits + gumbel_logistic) / tau)
z_hard = (z_soft > 0.5).float()
z = z_hard.detach() - z_soft.detach() + z_soft
```

Inference default:

```python
z = (torch.sigmoid(q_logits) >= 0.5).float()
```

Inference is deterministic and returns one logit. Do not sample at production inference time.

### 7.6 GateOnlyClassifierHead

The head must consume only the gate vector, not the unbottlenecked board features.

```text
LayerNorm(48)
Linear(48, 96)
SiLU
Dropout(0.10)
Linear(96, 1)
```

Output:

```python
logit: FloatTensor[B]
```

Hard rule: there is no skip connection from convolutional tokens, attention tokens, pooled board embeddings, prior stats, source metadata, or labels into the classifier head.

## 8. Tensor Contract

### 8.1 Forward contract

Production inference:

```python
logit = model(x)
```

Input:

```python
x.shape == [B, C, 8, 8]
x.dtype in {torch.float32, torch.float16, torch.bfloat16}
```

Output:

```python
logit.shape == [B]
logit.dtype == x.dtype or torch.float32 under autocast policy
```

Training / diagnostics mode:

```python
logit, aux = model(x, return_aux=True, tau=tau)
```

Auxiliary tensors:

```python
aux["q_logits"]   # [B, 48], posterior Bernoulli logits
aux["p_logits"]   # [B, 48], conditional-prior Bernoulli logits
aux["z"]          # [B, 48], straight-through Binary Concrete gate values
aux["rate_nats"]  # [B], sum_k Bernoulli KL in nats
aux["rate_bits"]  # [B], rate_nats / log(2)
```

The auxiliary tensors are for loss computation, logging, and ablations only. The deployed classifier interface emits only `logit`.

### 8.2 Loss input contract

```python
fine_label.shape == [B]
fine_label.dtype == torch.long
fine_label values in {0, 1, 2}
target = (fine_label == 2).to(logit.dtype)
```

### 8.3 Report tensor contract

At validation and test time, produce:

```python
fine_vs_pred: LongTensor[3, 2]
```

where:

```python
fine_vs_pred[0, 0] = count(fine_label == 0 and pred == 0)
fine_vs_pred[0, 1] = count(fine_label == 0 and pred == 1)
fine_vs_pred[1, 0] = count(fine_label == 1 and pred == 0)
fine_vs_pred[1, 1] = count(fine_label == 1 and pred == 1)
fine_vs_pred[2, 0] = count(fine_label == 2 and pred == 0)
fine_vs_pred[2, 1] = count(fine_label == 2 and pred == 1)
```

Log this matrix even when also logging AUC, AUPRC, F1, precision, recall, or rate.

## 9. Training Objective

### 9.1 Main objective

Use binary cross entropy with logits for the mapped binary target:

```python
target = (fine_label == 2).float()
bce = F.binary_cross_entropy_with_logits(
    logit,
    target,
    pos_weight=pos_weight,
    reduction="mean",
)
```

Set:

```python
pos_weight = num_binary_negative_train / max(1, num_binary_positive_train)
```

unless the repo already has a class-imbalance policy. If using a sampler that balances positives and negatives, set `pos_weight = None` and record that choice.

### 9.2 Conditional rate objective

```python
def bernoulli_kl_from_logits(q_logits, p_logits, eps=1e-6):
    q = torch.sigmoid(q_logits).clamp(eps, 1.0 - eps)
    p = torch.sigmoid(p_logits).clamp(eps, 1.0 - eps)
    kl = q * (q.log() - p.log()) + (1.0 - q) * ((1.0 - q).log() - (1.0 - p).log())
    return kl.sum(dim=-1)
```

```python
rate_nats = bernoulli_kl_from_logits(aux["q_logits"], aux["p_logits"])
rate_bits = rate_nats / math.log(2.0)
```

### 9.3 Capacity hinge

```python
capacity_hinge = F.relu(rate_bits.mean() - c_max_bits).pow(2)
```

Default:

```python
c_max_bits = 12.0
beta_rate = 0.003
gamma_capacity = 0.05
```

Final loss:

```python
loss = bce + beta_rate * rate_nats.mean() + gamma_capacity * capacity_hinge
```

### 9.4 Temperature schedule

```python
def gate_temperature(epoch, tau_start=1.20, tau_final=0.45, tau_anneal_epochs=20):
    t = min(1.0, epoch / float(tau_anneal_epochs))
    return tau_start * (1.0 - t) + tau_final * t
```

Use straight-through hard gates during training. If training becomes unstable in the first epoch, use soft gates for epoch `0` only, then switch to straight-through.

### 9.5 Validation selection rule

Primary selection should not be accuracy alone. Use:

```text
maximize validation AUPRC
subject to:
  fine_label_1 false-positive rate does not exceed the plain BCE baseline by more than 5%
  mean validation rate_bits is between 1.0 and 12.0
```

If no run satisfies the constraints, select the best AUPRC run and mark the bottleneck as failed in the experiment notes.

### 9.6 Mandatory metric report

Every validation/test report must include:

```python
{
  "loss/bce": ...,
  "loss/rate_nats": ...,
  "rate/mean_bits": ...,
  "rate/p95_bits": ...,
  "binary/auprc": ...,
  "binary/roc_auc": ...,
  "binary/f1_at_threshold": ...,
  "fine_vs_pred_3x2": [[...], [...], [...]],
  "threshold": ...
}
```

## 10. Ablations And Falsification

### 10.1 Smallest parameter-count-preserving ablation

**Name:** `no_info_budget_mean_gate`

This is the smallest ablation that removes the information-theoretic mechanism while preserving parameter count.

Keep all modules and trainable parameters:

```text
EasyStatsPrior
RelationalPosterior
GateOnlyClassifierHead
```

Change only the gate and loss:

```python
# no Binary Concrete noise
z = torch.sigmoid(q_logits)

# no KL rate loss and no capacity hinge
loss = bce

# keep prior branch instantiated and optionally train it as a shadow predictor
shadow = F.mse_loss(p_logits, q_logits.detach())
loss = bce + 0.0 * shadow
```

This preserves the same trainable parameter count and the same tensor shapes, but it removes the conditional mutual-information bottleneck. If this ablation equals or beats the selected model on AUPRC and fine-label-`1` false positives, the information-theoretic thesis is not supported.

### 10.2 Required ablation table

| Ablation | Change | What it tests |
|---|---|---|
| `no_info_budget_mean_gate` | Remove stochastic gate, KL, and capacity hinge; keep parameter count. | Whether the information-theoretic mechanism matters at all. |
| `constant_prior` | Replace `p_psi(z | c)` with a learned global `p(z)`. | Whether conditioning on low-order board stats is doing useful nuisance removal. |
| `rate_beta_zero_st_gate` | Keep stochastic gates but set `beta_rate = gamma_capacity = 0`. | Whether discreteness alone helps without rate pressure. |
| `conv_only_posterior` | Remove relative square attention; match parameter count with extra conv/MLP width. | Whether high-order square relations are needed. |
| `k_bits_16_48_96` | Sweep bottleneck size. | Whether performance depends on a real capacity frontier. |
| `prior_sees_tokens_negative_control` | Let the prior see the same pooled relational tokens as the posterior. | Should collapse the rate advantage; if it improves, the prior was too weak or the bottleneck was mis-specified. |
| `shuffle_square_control` | Evaluate on square-shuffled tensors that preserve per-plane counts. | If predictions remain strong, the model is likely exploiting low-order counts rather than chess relations. |
| `fine1_stress_eval` | Report metrics separately for fine label `1` and compare with plain BCE baseline. | Whether the model really handles near-puzzle hard negatives. |

### 10.3 Falsification criteria

Treat the idea as falsified if any two of the following hold on the same validation split:

1. `no_info_budget_mean_gate` has equal or better AUPRC and equal or lower fine-label-`1` false-positive rate.
2. Mean `rate_bits < 0.5` while validation performance is unchanged, indicating the bottleneck is unused.
3. Mean `rate_bits > 12.0` for the selected checkpoint, indicating the capacity constraint is not binding.
4. Fine-label-`1` row has a higher predicted-positive rate than the plain BCE baseline by more than 5 percentage points.
5. A material/side-to-move stratified split shows large degradation relative to the random split, indicating low-order shortcuts.
6. The constant-prior ablation matches the conditional-prior model, indicating the conditional-surprisal mechanism is not buying anything.

## 11. Implementation Notes For Codex

### 11.1 Suggested files

Use existing repo conventions if they differ, but the implementation should fit this shape:

```text
models/conditional_surprisal_gate.py
training/losses/conditional_surprisal_gate_loss.py
metrics/fine_vs_pred.py
configs/model/conditional_surprisal_gate.yaml
```

Do not create separate engine-feature loaders, source-label readers, or provenance hooks.

### 11.2 Minimal PyTorch skeleton

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def bernoulli_kl_from_logits(q_logits: torch.Tensor, p_logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    q = torch.sigmoid(q_logits).clamp(eps, 1.0 - eps)
    p = torch.sigmoid(p_logits).clamp(eps, 1.0 - eps)
    kl = q * (q.log() - p.log()) + (1.0 - q) * ((1.0 - q).log() - (1.0 - p).log())
    return kl.sum(dim=-1)


def binary_concrete_gate(logits: torch.Tensor, tau: float, hard: bool = True) -> torch.Tensor:
    u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
    noise = torch.log(u) - torch.log1p(-u)
    z_soft = torch.sigmoid((logits + noise) / tau)
    if not hard:
        return z_soft
    z_hard = (z_soft >= 0.5).to(z_soft.dtype)
    return z_hard.detach() - z_soft.detach() + z_soft


class EasyStatsPrior(nn.Module):
    def __init__(self, in_channels: int, k_bits: int = 48, hidden: int = 128, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * in_channels, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, k_bits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_stats = x.mean(dim=(-2, -1))
        max_stats = x.amax(dim=(-2, -1))
        stats = torch.cat([mean_stats, max_stats], dim=-1)
        return self.net(stats)


class GateOnlyClassifierHead(nn.Module):
    def __init__(self, k_bits: int = 48, hidden: int = 96, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(k_bits),
            nn.Linear(k_bits, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class ConditionalSurprisalGatePuzzleNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        k_bits: int = 48,
        tau_infer_hard: bool = True,
    ):
        super().__init__()
        self.k_bits = k_bits
        self.tau_infer_hard = tau_infer_hard
        self.prior = EasyStatsPrior(in_channels=in_channels, k_bits=k_bits)

        # Codex: implement RelationalPosterior exactly from section 7.4.
        # It must return q_logits with shape [B, k_bits].
        self.posterior = RelationalPosterior(in_channels=in_channels, k_bits=k_bits)

        self.head = GateOnlyClassifierHead(k_bits=k_bits)

    def forward(self, x: torch.Tensor, return_aux: bool = False, tau: float = 0.45):
        q_logits = self.posterior(x)
        p_logits = self.prior(x)

        if self.training:
            z = binary_concrete_gate(q_logits, tau=tau, hard=True)
        else:
            q = torch.sigmoid(q_logits)
            z = (q >= 0.5).to(q.dtype) if self.tau_infer_hard else q

        logit = self.head(z)

        if not return_aux:
            return logit

        rate_nats = bernoulli_kl_from_logits(q_logits, p_logits)
        aux = {
            "q_logits": q_logits,
            "p_logits": p_logits,
            "z": z,
            "rate_nats": rate_nats,
            "rate_bits": rate_nats / math.log(2.0),
        }
        return logit, aux
```

### 11.3 Loss helper

```python
def conditional_surprisal_gate_loss(
    logit: torch.Tensor,
    aux: dict,
    fine_label: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    beta_rate: float = 0.003,
    gamma_capacity: float = 0.05,
    c_max_bits: float = 12.0,
) -> tuple[torch.Tensor, dict]:
    target = (fine_label == 2).to(dtype=logit.dtype, device=logit.device)

    bce = F.binary_cross_entropy_with_logits(
        logit,
        target,
        pos_weight=pos_weight,
        reduction="mean",
    )

    rate_nats = aux["rate_nats"]
    rate_bits = aux["rate_bits"]
    capacity_hinge = F.relu(rate_bits.mean() - c_max_bits).pow(2)

    loss = bce + beta_rate * rate_nats.mean() + gamma_capacity * capacity_hinge

    logs = {
        "loss/bce": bce.detach(),
        "loss/rate_nats": rate_nats.mean().detach(),
        "rate/mean_bits": rate_bits.mean().detach(),
        "rate/p95_bits": torch.quantile(rate_bits.detach(), 0.95),
        "loss/capacity_hinge": capacity_hinge.detach(),
    }
    return loss, logs
```

### 11.4 RelativeSquareAttention implementation notes

- Build a fixed mapping from square index `0..63` to `(rank, file)`.
- Relative deltas are `dr = rank_i - rank_j` and `df = file_i - file_j`, each in `[-7, 7]`.
- Store a learned bias table with shape `[n_heads, 15, 15]`.
- During attention, add `bias[:, dr + 7, df + 7]` to the attention logits before softmax.
- This is board geometry only. It is not a legal-move generator and not an engine input.

### 11.5 Integration checks

Add assertions:

```python
assert x.ndim == 4
assert x.shape[-2:] == (8, 8)
assert fine_label.min() >= 0 and fine_label.max() <= 2
assert logit.shape == fine_label.shape
```

Add a unit test that calls:

```python
model = ConditionalSurprisalGatePuzzleNet(in_channels=18)
x = torch.randn(4, 18, 8, 8)
fine = torch.tensor([0, 1, 2, 2])
logit, aux = model(x, return_aux=True, tau=1.0)
loss, logs = conditional_surprisal_gate_loss(logit, aux, fine)
loss.backward()
```

Expected shapes:

```python
logit.shape == torch.Size([4])
aux["q_logits"].shape == torch.Size([4, 48])
aux["p_logits"].shape == torch.Size([4, 48])
aux["z"].shape == torch.Size([4, 48])
aux["rate_bits"].shape == torch.Size([4])
```

## 12. Expected Failure Modes

1. **Posterior collapse:** The KL term may force `q(z|x)` too close to `p(z|c)`, producing low rate and weak recall. Fix by lowering `beta_rate`, raising `c_max_bits`, or warming up the rate penalty over 3 to 5 epochs.
2. **Rate explosion:** The classifier may ignore the capacity hinge and use too many bits. Fix by increasing `gamma_capacity`, lowering `k_bits`, or selecting checkpoints subject to the rate constraint.
3. **Hard-negative confusion:** Fine label `1` may remain close to fine label `2` because some near-puzzles are genuinely tactical but fail verification. This is a data ambiguity, not necessarily a model bug; the 3x2 matrix is required so this failure is visible.
4. **Low-order shortcut leakage through posterior:** The posterior can still encode material or side-to-move cues. The conditional prior makes those cues expensive only when the prior cannot predict them; use square-shuffle and material-stratified evaluations to detect shortcut dependence.
5. **Prior too weak:** If `p(z|c)` cannot predict even cheap board statistics, the rate penalty becomes noisy. Check `constant_prior` and `prior_hidden` ablations before discarding the idea.
6. **Prior too strong:** If the prior branch is accidentally given relational features, the rate term stops measuring conditional surprisal. Keep EasyStatsPrior restricted to pooled channel stats.
7. **Discrete gate optimization noise:** Straight-through Binary Concrete can be unstable early. Use a temperature warmup, gradient clipping at `1.0`, and mixed-precision care around the KL computation.
8. **Threshold overfitting:** A tuned threshold can hide poor score ranking. Always report AUPRC/ROC-AUC and the threshold used for the 3x2 matrix.
9. **Tensor-family mismatch:** A config trained on `simple_18` should not silently load into `lc0_static_112`. Assert `in_channels` at checkpoint load.
10. **No current-board-only solution for some labels:** Some verified-puzzle labels may depend on verification rules or engine analysis not present in the board tensor. The model must not import those forbidden signals; accept that irreducible label noise exists.

## 13. Prompt-Maintenance Notes

- Keep the model as a **single-logit binary classifier**. Do not add fine-label logits, ordinal heads, move-policy heads, value heads, or uncertainty heads.
- Keep fine label `1` as binary negative during training. It is only separated in reports and stress metrics.
- Keep the mandatory 3x2 matrix in every validation and test report.
- Keep the forbidden-input list attached to configs and model cards. Any feature whose value depends on Stockfish, PVs, nodes, mate scores, verification metadata, source labels, engine best moves, source file identity, or dataset provenance is out of scope.
- Keep the classifier head gate-only. Any skip connection from unbottlenecked board features invalidates the bottleneck.
- Keep EasyStatsPrior low-order. If it receives attention tokens, convolutional board maps, legal moves, or square-pair features, the conditional-surprisal interpretation no longer holds.
- Keep `no_info_budget_mean_gate` as the first ablation. It is the minimal same-parameter falsification test.
- If future data adds legal move tensors, engine annotations, puzzle themes, or provenance fields, do not feed them to this model unless the data contract is rewritten.
- If the repo already has a strong plain BCE ResNet baseline, compare against it, but do not let that baseline replace the information-theoretic packet.
- Prefer small, auditable code over clever estimators. The selected mechanism is deliberately implementable with one posterior, one weak prior, one Bernoulli KL, and one gate-only head.
