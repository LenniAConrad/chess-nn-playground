# Codex Handoff Packet: Credal Near-Puzzle Evidence Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0750_tuesday_los_angeles_credal_evidence.md`
- Generated at: 2026-04-21 07:50:05 America/Los_Angeles
- Weekday: Tuesday
- Timezone: America/Los_Angeles (`los_angeles` in filename)
- Idea slug: `credal_evidence`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Credal Near-Puzzle Evidence Network
- One-sentence thesis: Train a binary puzzle-likeness classifier whose output is a Dirichlet evidence distribution, treating verified near-puzzles as an interval-valued positive target with deliberately limited evidence rather than as either hard puzzles or a third fabricated class.
- Idea fingerprint: `current board tensor -> ordinary compact CNN encoder -> binary Dirichlet evidence head -> hard singleton losses for fine labels 0 and 2 + credal positive interval/evidence-cap loss for fine label 1 -> binary logits from Dirichlet mean`.
- Why this is not a common CNN/ResNet/Transformer variant: The backbone is intentionally boring; the tested mechanism is the label-safe credal/evidential objective that changes how fine label 1 constrains the predictive distribution and uncertainty, not depth, width, attention, or extra board geometry.
- Current-data minimal experiment: Train on `data/splits/crtk_sample_3class/split_train.parquet`, validate/test on the existing split, first with `simple_18`, 3 epochs, batch size 512, balanced fine-label weighting, and compare against the same backbone trained with ordinary binary cross-entropy.
- Smallest central falsification ablation: Replace the fine-label-1 credal interval/evidence-cap loss with ordinary hard-positive BCE while keeping the exact same encoder, parameter count, optimizer, class weighting, and reporting.
- Expected information gain if it fails: A clean failure says the near-puzzle label does not behave like a calibrated ambiguity band for this dataset, so future work should stop spending cycles on label-safe uncertainty heads and return to board-structure or causal-shift mechanisms.

## 3. Problem Restatement And Data Contract

The project is `chess-nn-playground`. The current task is binary chess puzzle-likeness classification from a single board position:

- output `0`: non-puzzle
- output `1`: puzzle-like

The available fine labels are:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

The default benchmark is binary, but reports must retain the rectangular diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Allowed model input is a tensor shaped:

```text
(batch, C, 8, 8)
```

The model must return binary logits shaped:

```text
(batch, 2)
```

Current encodings:

- `simple_18`: 12 piece planes + side-to-move + castling + en-passant.
- `lc0_static_112`.
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists.

Benchmark split:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

The full Parquet dataset is roughly 45M rows and must not be used directly by the current trainer until streaming support exists.

Leakage checklist:

- Safe: deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack geometry derived only from the current board.
- Leakage-prone unless explicitly justified and ablated: full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences.
- Forbidden as neural-network inputs: Stockfish or other engine evaluations, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, and dataset provenance.
- Fine labels are allowed as supervision because they are the task labels; they must not be concatenated to the input tensor or used to generate extra input features.
- This idea uses no move generator, no attack graph, no engine output, no source metadata, and no candidate pool.
- For `lc0_static_112` and `lc0_bt4_112`, current-board channels may be consumed by the learned neural adapter. History channels must not be interpreted as deterministic geometry unless the exporter has documented their semantics; until then they are just learned input channels. Adapters must fail closed when channel semantics are unknown and a rule-derived feature extractor asks for them.

## 4. Research Map

External ideas used:

| Source | What is borrowed | What is not copied |
|---|---|---|
| Sensoy, Kaplan, and Kandemir, “Evidential Deep Learning to Quantify Classification Uncertainty,” NeurIPS 2018. URL: https://papers.nips.cc/paper/7580-evidential-deep-learning-to-quantify-classification-uncertainty | The idea of parameterizing a Dirichlet distribution over class probabilities through nonnegative neural evidence. | Not copying their exact MSE-style objective, OOD experiments, adversarial claims, or subjective-logic terminology as a black box. |
| Malinin and Gales, “Predictive Uncertainty Estimation via Prior Networks,” NeurIPS 2018. URL: https://arxiv.org/abs/1802.10501 | The broader framing that a deterministic network can output a distribution over categorical predictive distributions. | Not using OOD training data, synthetic priors, or distributional-shift labels. |
| Cour, Sapp, and Taskar, “Learning from Partial Labels,” JMLR 2011. URL: https://jmlr.org/papers/v12/cour11a.html | The set-valued target view: supervision can constrain a prediction to a feasible label set instead of one point. | Not treating near-puzzles as unknown true class labels, not disambiguating fabricated candidates, and not using their algorithm. |
| Chow, “On Optimum Recognition Error and Reject Tradeoff,” IEEE Transactions on Information Theory 1970. URL: https://dl.acm.org/doi/10.1109/TIT.1970.1054406 | The diagnostic idea that uncertainty/abstention can trade coverage against risk. | The deployed model still returns binary logits; abstention is only an analysis diagnostic. |
| Geifman and El-Yaniv, “Selective Classification for Deep Neural Networks,” 2017. URL: https://arxiv.org/abs/1705.08500 | Risk-coverage and selective-prediction reporting as a way to test whether evidence is meaningful. | Not using their post-hoc selection method as the core model. |
| Arjovsky et al., “Invariant Risk Minimization,” 2019. URL: https://arxiv.org/abs/1907.02893 | Used only as a rejected candidate reference for environment invariance. | No IRM penalty is part of this selected idea. |
| Alemi et al., “Deep Variational Information Bottleneck,” 2016/2017. URL: https://arxiv.org/abs/1612.00410 | Used only as a rejected candidate reference for bottleneck alternatives. | No variational latent bottleneck is part of this selected idea. |

Candidate search trace:

| Candidate mechanism considered | Why it lost to the selected idea |
|---|---|
| Causal IRM across material phase, side-to-move, and game-stage environments | Attractive for source-artifact control, but phase-specific tactics are real chess signal; an IRM failure would be hard to interpret. |
| Multi-view consensus between `simple_18` and `lc0_bt4_112` encoders | Potentially useful, but it requires paired multi-encoding data plumbing and risks testing exporter artifacts instead of puzzle-likeness. |
| Masked board reconstruction or MDL surprise | Too close to the imported static pseudo-likelihood/description-length family and likely to learn board-frequency statistics rather than near-puzzle ambiguity. |
| Persistent homology or topological barcode over occupied squares | Novel-looking but weakly tied to chess tactics and easy to duplicate with high-order constellation statistics. |
| Rule-only SAT or constraint-inconsistency surrogate | Becomes leakage-prone as soon as it queries mate/stalemate/legal-consequence oracles; a safe static version is probably just attack geometry. |
| Steerable partial-equivariant CNN under chess-safe symmetries | Useful engineering, but too close to a CNN architecture variant and not a distinct falsifiable label mechanism. |
| Board-shuffle self-supervised artifact adversary | Might suppress source artifacts, but the relation to puzzle-likeness is indirect and the auxiliary task could dominate. |
| Differentiable pin/skewer theorem prover | Likely to collapse into an attack-defense graph/sheaf family already imported. |
| Low-rank tensor factorization of piece-square interactions | Near the imported Möbius/ANOVA constellation packet. |
| Formal ray grammar with learned finite-state logic | Near the imported ray-language automaton packet. |
| Entropy of pseudo-legal response sets | Near the imported one-ply move-delta landscape family and risks move-count shortcuts. |
| Dirichlet evidential treatment of fine label 1 | Selected because it is label-safe, current-data compatible, mathematically sharp, and not a board-geometry duplicate. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Credal label | Label-indexed feasible set `C_z` over binary class probability `pi_1`: singleton for labels 0/2, interval `[tau, 1]` for label 1 | Consumes fine label `z` only inside the loss; emits scalar `L_credal` | Replace interval target for fine label 1 with hard-positive BCE | Not ordinal: no cumulative `P(fine>=k)` heads and no third-class ranking. |
| Evidential uncertainty | Binary Dirichlet `Dir(alpha_0, alpha_1)` with `alpha = 1 + softplus(raw_evidence)` | Encoder feature `[B,H] -> alpha [B,2] -> logits [B,2]` | Remove fine-label-1 evidence cap while keeping mean interval | Not an ensemble, dropout, or calibration post-process; uncertainty is a trained output distribution. |
| Selective prediction | Use evidence/concentration `S=sum alpha` for diagnostics and optional risk-coverage curves | Prediction artifact columns: `mu_pos`, `S`, `uncertainty=2/S` | If `S` does not separate fine label 1 from label 2, abandon the uncertainty claim | It does not add an abstention class or alter the official binary output. |
| Label-safe ambiguity | Fine label 1 is positive but intentionally lower-evidence than verified puzzle label 2 | Loss sees `z in {0,1,2}`; model output remains `[B,2]` | Shuffle the fine-label-1 ambiguity flag inside material/phase bins | Does not fabricate labels and does not use unresolved candidates. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN | `src/chess_nn_playground/models/cnn.py` | Already exists and tests generic local visual patterns without addressing near-puzzle ambiguity. |
| Residual CNN | `src/chess_nn_playground/models/residual_cnn.py` | Already exists; adding residual depth is not a research idea here. |
| LC0-style CNN or residual CNN on BT4 planes | Existing LC0 BT4-style CNN/residual variants | Already covered as an encoding/backbone family and risks becoming “copy LC0 but smaller.” |
| Ordinary ViT over 64 squares | Standard square-token transformer | Too generic, compute-heavier, and explicitly disallowed as a core idea. |
| Plain GNN-on-squares | Generic graph neural network over board adjacency | Usually becomes a square-neighborhood CNN in disguise unless it uses rule geometry, which would drift toward imported graph/sheaf families. |
| Hyperparameter tuning | Existing trainer configs | Learning-rate, batch-size, or width searches do not create a new falsifiable mechanism. |
| Ensembling | Any existing CNN/residual ensemble | Improves variance and calibration at extra cost but does not explain puzzle-likeness. |
| Add more data | Full 45M Parquet file | The current trainer should not hit the full dataset before streaming support; the requested idea must be testable on the current split. |
| Stockfish/evaluation feature model | Engine-score pipelines | Forbidden leakage: evaluations, PVs, mate scores, node counts, and verification metadata cannot be neural-network inputs. |
| Another static attack-defense graph/sheaf/Hodge model | Imported tactical sheaf/Hodge packets | Already heavily represented; more edge labels or pooling variants would be duplicate research. |
| Another one-ply move-delta set/landscape model | Imported counterfactual move-delta packets | Already represented and risks move-count/capture shortcuts. |
| Another Sinkhorn or piece-target transport model | Imported optimal-transport packets | Already represented; changing costs or temperature would not be a distinct idea. |
| Ordinary ordinal fine-label ladder | Imported Ordinal Evidence Ladder Network | The selected idea does not learn `P(fine>=1)` and `P(fine>=2)`; it learns a binary credal distribution with low evidence for fine label 1. |
| Three-way softmax fine-label classifier collapsed to binary | Generic multiclass classifier | It treats near-puzzle as a separate class rather than a verified but uncertain positive boundary, and may optimize the wrong benchmark. |

## 6. Mathematical Thesis

Input space definition:

Let `X_C = R^{C x 8 x 8}` be the encoding tensor space for one board. The first experiment uses `C=18` for `simple_18`. Later experiments may set `C=112` for `lc0_static_112` or `lc0_bt4_112` with a fail-closed adapter.

Label/target definition:

Let the observed fine label be `Z in {0,1,2}`:

- `Z=0`: known non-puzzle.
- `Z=1`: verified near-puzzle.
- `Z=2`: verified puzzle.

The official binary target is:

```text
Y = 1[Z in {1,2}]
```

The model returns logits for `Y in {0,1}`.

Data distribution assumptions:

There is an unknown distribution `P(X,Z)`. Fine labels `0` and `2` are treated as sharper endpoint evidence. Fine label `1` is treated as a verified positive boundary region: it should generally land on the puzzle-like side of the binary decision, but it should not be forced to produce the same predictive concentration as a verified puzzle. This is a hypothesis about the dataset semantics, not a proven property.

Allowed symmetry or equivariance assumptions:

No full board rotation/reflection invariance is assumed. Chess pawns, castling, and side-to-move break most board symmetries. The minimal experiment uses no explicit symmetry constraint. Optional data augmentation may include only already-approved chess-safe transforms, but it is not central to this idea.

Core hypothesis:

Hard BCE collapses fine labels `1` and `2` into the same point target `Y=1`, which encourages overconfident predictions on boundary examples and can reward shortcuts that separate source-like near-puzzles from true puzzles. A credal/evidential binary head should improve near-puzzle recall at matched fine-label-0 false-positive rate and improve calibration by representing fine label `1` as positive-but-ambiguous.

Formal object introduced:

For a board `x`, the network produces nonnegative evidence `e_theta(x) in R_+^2`, Dirichlet parameters

```text
alpha_theta(x) = 1 + e_theta(x),
S_theta(x) = alpha_0 + alpha_1,
mu_theta(x) = alpha_1 / S_theta(x).
```

The predictive distribution over the binary positive probability `pi_1` is

```text
Pi_theta(. | x) = Dirichlet(alpha_0(x), alpha_1(x)).
```

Define label-indexed credal target sets over `q = (q_0, q_1) in Delta^1`:

```text
C_0 = {(1,0)}
C_2 = {(0,1)}
C_1(tau) = {q in Delta^1 : q_1 >= tau}
```

with default `tau = 0.55`.

Define the credal projection loss on the Dirichlet mean `m=(1-mu, mu)`:

```text
L_set(z, m) = min_{q in C_z} KL(q || m).
```

For `z=0` and `z=2`, this reduces to ordinary hard-label negative log-likelihood up to a constant. For `z=1`, it is zero whenever `mu >= tau` and grows like a one-sided KL barrier when `mu < tau`.

Add an evidence-shaping term:

```text
L_ev(z, S) =
  lambda_near * [max(0, log(S) - log(S_near_max))]^2,        if z = 1
  lambda_kl * KL(Dir(alpha_tilde_z) || Dir(1,1)),             if z in {0,2}
```

where `alpha_tilde_z` is the standard evidential “remove correct-class evidence before KL” parameter used to discourage unsupported wrong-class evidence. The exact implementation may use a simpler annealed Dirichlet-to-uniform KL if easier, but the fine-label-1 evidence cap must remain.

Total loss:

```text
L(theta) = E_{(X,Z)} [ w_Z * (L_set(Z, m_theta(X)) + L_ev(Z, S_theta(X))) ].
```

Use balanced fine-label weights `w_Z` by default.

Proposition:

For binary `m=(1-mu,mu)` and `C_1(tau)={q:q_1>=tau}`, the projection loss

```text
L_set(1,m) = min_{q in C_1(tau)} KL(q || m)
```

equals `0` if `mu >= tau`, and equals

```text
tau * log(tau / mu) + (1-tau) * log((1-tau)/(1-mu))
```

if `mu < tau`.

Proof sketch:

For binary distributions, `KL(q||m)` is strictly convex in `q` on the simplex. If `m` lies in the closed convex set `C_1(tau)`, the minimum is attained at `q=m`, giving zero. If `mu < tau`, the closest feasible point under the binary KL Bregman geometry lies at the interval boundary `q_1=tau`, yielding the displayed Bernoulli KL. The hard-label cases are singleton sets, so the minimizer is the singleton target.

What is actually proven:

The loss mathematically enforces a one-sided positive constraint for near-puzzles without forcing them to the same point target as verified puzzles. The proof only concerns the loss geometry, not chess.

What remains only hypothesized:

It is not proven that fine label `1` is truly an uncertainty band in the data distribution. It is also not proven that lower evidence on near-puzzles will improve binary classification. Those are precisely what the ablations must test.

Counterexamples where the idea should fail:

- If fine label `1` examples are just as sharply puzzle-like as fine label `2`, evidence caps will undertrain useful positives.
- If fine label `1` contains mixed hidden subtypes, some genuinely non-puzzle-like, the one-sided positive interval may inflate false positives.
- If dataset construction creates fine-label artifacts unrelated to board content, the evidence head may learn those artifacts rather than chess ambiguity.
- If the base CNN cannot learn any useful chess signal in 3 epochs, the loss geometry may not matter.

Self-critique:

The strongest objection is that this idea may be “only a loss function” wrapped around a standard CNN. That is deliberate: the current imported memory is saturated with elaborate board operators, so a clean uncertainty/label-semantics test has high value. The falsifier is unusually sharp: if hard-positive BCE with the same backbone matches or beats it on near-puzzle recall, calibration, and evidence diagnostics, the credal hypothesis should be discarded rather than patched with a bigger model.

## 7. Architecture Specification

Module names:

- `CredalEvidencePuzzleNet`
- `FailClosedBoardAdapter`
- `TinyResidualBoardEncoder`
- `DirichletEvidenceHead`
- `CredalEvidenceLoss`

Forward-pass steps for the minimal `simple_18` model:

1. Input `x`: shape `[B, 18, 8, 8]`.
2. `FailClosedBoardAdapter`:
   - Check `input_channels == 18`.
   - Apply `1x1 Conv2d(18, hidden_channels=64)`.
   - Output `[B, 64, 8, 8]`.
3. Stem:
   - `3x3 Conv2d(64,64,padding=1)`, normalization, activation.
   - Output `[B, 64, 8, 8]`.
4. Encoder:
   - Four small residual blocks, each two `3x3 Conv2d(64,64,padding=1)` layers.
   - Output `[B, 64, 8, 8]`.
5. Pool:
   - Global average pooling over board squares.
   - Output `[B, 64]`.
6. MLP:
   - `Linear(64,128)`, activation, dropout optional default `0.0`.
   - Output `[B,128]`.
7. Evidence head:
   - `Linear(128,2)` gives `raw_evidence [B,2]`.
   - `evidence = softplus(raw_evidence)`.
   - `alpha = 1 + evidence`, shape `[B,2]`.
8. Logits for shared trainer:
   - `logits = log(alpha + eps)`.
   - Since `softmax(log(alpha)) = alpha / sum(alpha)`, logits represent the Dirichlet mean.
   - Return shape `[B,2]`.
9. Optional training path:
   - `forward_with_evidence(x)` or `forward(x, return_aux=True)` returns `{"logits", "alpha", "mu_pos", "S"}` for the custom loss.

Parameter-count estimate:

- `simple_18`: about 0.35M parameters.
- `lc0_static_112` or `lc0_bt4_112`: about 0.36M parameters because only the `1x1` adapter grows.
- This should remain in the same small-model regime as existing baselines.

FLOP/complexity estimate:

- Each `64 -> 64` `3x3` convolution on `8x8` costs about `8*8*64*64*9 ~= 2.36M` multiply-accumulates.
- One stem convolution plus eight residual-block convolutions cost about `21M` MACs per board.
- The adapter and MLP are negligible relative to the convolutions.
- Complexity is `O(B * 8 * 8 * hidden_channels^2 * num_convs)`.

Generated candidate set and memory:

- No rule-generated candidate set is used.
- Auxiliary tensors are small:
  - `alpha [B,2]`
  - `mu_pos [B]`
  - `S [B]`
- Memory overhead over the base CNN is `O(B)`.

Chunking plan:

- No candidate-set chunking is needed.
- If Codex later adds prediction exports for all splits, write predictions in row chunks from the dataloader rather than accumulating every batch in GPU memory.

Required config fields:

```yaml
model:
  name: credal_evidence_puzzle_net
  input_channels: 18
  num_classes: 2
  hidden_channels: 64
  num_res_blocks: 4
  evidence_floor: 1.0
  near_tau: 0.55
  near_s_max: 6.0
  lambda_near_evidence_cap: 0.05
  lambda_dirichlet_kl: 0.001
  kl_anneal_epochs: 2
  dropout: 0.0
  allow_unknown_channels: false
```

Encoding support:

- First experiment: use only `simple_18` because it is transparent and least likely to hide history/exporter artifacts.
- `simple_18`: adapter accepts 18 channels and assumes only tensor shape, not rule-derived semantics.
- `lc0_static_112`: adapter may accept 112 channels when `encoding` explicitly says `lc0_static_112`; it must not infer undocumented channel meanings.
- `lc0_bt4_112`: adapter may accept 112 channels when `encoding` explicitly says `lc0_bt4_112`; zero-filled history planes are consumed only as learned channels. Do not derive deterministic geometry from history planes.
- Unknown channel counts: fail closed unless a config explicitly opts in. The default should raise a clear error.

Shared trainer compatibility:

- `forward(x)` returns only logits `[B,2]`.
- Custom idea training can call an auxiliary forward method to compute `alpha`, `mu_pos`, and `S`.
- Existing reports, binary confusion matrices, 3x2 diagnostics, leaderboards, and prediction exports should work with returned logits.

Pseudocode only:

```python
class CredalEvidencePuzzleNet(nn.Module):
    def forward(self, x, return_aux=False):
        h = adapter(x)
        h = stem(h)
        h = residual_blocks(h)
        h = gap(h).flatten(1)
        h = mlp(h)
        raw = evidence_head(h)
        evidence = softplus(raw)
        alpha = 1.0 + evidence
        logits = log(alpha + eps)
        if not return_aux:
            return logits
        mu = alpha[:, 1] / alpha.sum(dim=1)
        return {"logits": logits, "alpha": alpha, "mu_pos": mu, "S": alpha.sum(dim=1)}
```

Do not implement the full final code in this handoff packet.

## 8. Loss, Training, And Regularization

Primary loss:

Use `CredalEvidenceLoss` over fine labels `z in {0,1,2}`.

For `z=0`:

```text
L_set = -log(mu_0 + eps)
```

For `z=2`:

```text
L_set = -log(mu_1 + eps)
```

For `z=1`:

```text
if mu_1 >= tau:
    L_set = 0
else:
    L_set = tau*log(tau/(mu_1+eps)) + (1-tau)*log((1-tau)/(mu_0+eps))
```

Evidence terms:

- Fine label `1`: cap concentration with

```text
L_near_cap = lambda_near * max(0, log(S) - log(S_near_max))^2
```

- Fine labels `0` and `2`: use an annealed Dirichlet KL or standard evidential wrong-evidence regularizer to avoid unsupported wrong-class evidence.

Optional auxiliary loss:

- Optional calibration loss on validation is not allowed during training.
- Optional training-time Brier loss may be added only as an ablation, not the main run.
- No source-provenance or engine-derived auxiliary targets.

Class weighting:

- Default: balanced fine-label weights so classes `0`, `1`, and `2` contribute similar total loss.
- Report both fine-balanced and binary-balanced only if time permits. The main result should use fine-balanced to avoid near-puzzle loss being numerically ignored.

Batch size expectations:

- Start with batch size 512 for `simple_18`.
- If GPU memory is tight, reduce to 256 while keeping all compared runs identical.

Learning-rate and optimizer defaults:

- Optimizer: AdamW.
- Learning rate: `1e-3`.
- Weight decay: `1e-4`.
- Epochs: 3 for minimal experiment.
- Early stopping patience: 2.
- Mixed precision: false for deterministic comparability unless existing configs already standardize it.

Regularizers:

- Weight decay `1e-4`.
- Dropout default `0.0`; optional `0.1` only as a later robustness check.
- Gradient clipping at norm `5.0` is acceptable if trainer already supports it.
- KL anneal from 0 to `lambda_dirichlet_kl` over the first 2 epochs.

Determinism requirements:

- Seed: 42.
- `torch.backends.cudnn.deterministic = true` when feasible.
- `num_workers = 0` for the minimal reproducible run.
- Save the exact config and git diff/status in the report if the repo tooling supports it.

What must stay unchanged for fair comparison:

- Same train/val/test split.
- Same encoding for paired baseline and ablation.
- Same backbone parameter count for credal model and hard-BCE control.
- Same optimizer, epochs, batch size, early stopping, class weighting policy, and seed.
- Same thresholding rule for binary predictions unless a metric explicitly says threshold-free or matched-FPR.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Hard-positive BCE control | Replaces fine-label-1 credal interval and evidence cap with ordinary binary target `Y=1`; same backbone and optimizer | Central claim: near-puzzles should not be trained as identical to verified puzzles | If this matches or beats the main model on class-1 recall at matched class-0 FPR and calibration, abandon the idea. |
| No evidence cap for fine label 1 | Keeps `mu_1 >= tau` interval but removes `S_near_max` cap | The uncertainty/evidence part, not just one-sided BCE, matters | If metrics stay the same and `S` no longer separates labels 1 and 2, the evidential story is unnecessary. |
| Ignore fine label 1 during training | Trains only on labels 0 and 2, evaluates on all labels | Label 1 provides useful boundary supervision rather than noise | If ignoring label 1 wins, the near-puzzle label may be too noisy for supervised use. |
| Treat fine label 1 as soft target 0.7 | Replaces the credal set with fixed soft-label BCE | Interval-valued supervision beats arbitrary label smoothing | If soft target wins, the benefit is probably smoothing rather than credal geometry. |
| Three-way softmax then collapse | Trains a `0/1/2` classifier and collapses `1,2 -> positive` | Direct fine-label classification is enough | If three-way wins, near-puzzle may be a separable class, not an uncertainty band. |
| Imported ordinal ladder control | Uses cumulative heads `P(fine>=1)`, `P(fine>=2)` if existing code is available | Credal evidence is distinct from ordinal fine-label ordering | If ordinal ladder wins cleanly, ordered class structure matters more than uncertainty. |
| Shuffled ambiguity flag within nuisance bins | Within bins of material count, side-to-move, and coarse phase, randomly choose which positive examples receive the fine-label-1 credal loss | Tests whether the semantics of verified near-puzzle, not just count/frequency/phase, matters | If shuffled flag performs the same, the near-puzzle evidence target is not using meaningful label semantics. |
| Evidence-only diagnostic head | Train ordinary BCE logits but add evidence head and evidence cap without using it for logits | Tests whether coupling Dirichlet mean and evidence is needed | If this matches, the Dirichlet predictive object is not necessary. |
| Unary material/phase nuisance report | No model change; stratify results by material, phase, side-to-move, and castling availability | Detects obvious shortcut dependence | If gains appear only in one nuisance stratum, scaling is not justified. |

This idea has no graph, hypergraph, sheaf, transport coupling, move set, or candidate set. Therefore no degree-preserving candidate-randomization ablation is required. The relevant semantics-destroying ablation is the nuisance-bin shuffle of the fine-label-1 ambiguity flag, which preserves class counts and obvious board nuisances while destroying the proposed label semantics.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- Existing simple CNN on `simple_18`.
- Existing residual CNN on `simple_18`.
- Same `TinyResidualBoardEncoder` with ordinary BCE.
- Same encoder with fixed soft target for fine label 1, if quick.
- Existing LC0-style CNN/residual only if Codex also runs the 112-channel follow-up.
- Imported ordinal ladder only as a reference if already easy to invoke.

Metrics to inspect:

- Accuracy.
- ROC-AUC.
- PR-AUC.
- F1 at default threshold.
- Binary confusion matrix.
- Required `3x2` diagnostic matrix: true fine label `0/1/2` -> predicted binary `0/1`.
- Brier score.
- Negative log-likelihood using Dirichlet mean.
- Expected calibration error with 10 or 15 bins.
- Reliability plot if the reporting stack supports it.
- Median and interquartile range of `S=sum(alpha)` by fine label.
- Risk-coverage curve using `S` or `max(mu)` as confidence.

Near-puzzle diagnostic:

- Main diagnostic: fine-label-`1` recall at a matched fine-label-`0` false-positive rate.
- Use at least these operating points:
  - FPR on fine label `0` fixed to the hard-BCE control's FPR at threshold `0.5`.
  - FPR on fine label `0` fixed to `5%`, if enough examples exist.
- Also report fine-label-`1` precision among predicted positives at the matched threshold.

Required artifacts:

- Config YAML.
- Model checkpoint for best validation metric.
- Test predictions Parquet or CSV with: row id if available, fine label, binary target, logits, `mu_pos`, `S`, predicted binary label, split name.
- Markdown report with all metrics, `3x2` matrices for main and central ablations, evidence histograms, and reliability plot if supported.
- Leaderboard entry using the same format as existing models.

Success threshold:

The idea is worth keeping if, on the test split:

- It improves fine-label-`1` recall by at least `+1.0` absolute percentage point at matched fine-label-`0` FPR versus the same-backbone BCE control.
- It does not reduce ROC-AUC by more than `0.003` absolute versus the same-backbone BCE control.
- It improves ECE or Brier score by at least `5%` relative versus the same-backbone BCE control.
- Median `S` for fine label `1` is lower than median `S` for fine label `2`, while fine label `1` still predicts positive more often than fine label `0`.

Failure threshold:

Treat the idea as failed if any of these hold:

- The hard-BCE control is equal or better on near-puzzle recall at matched fine-label-`0` FPR and equal or better on calibration.
- The model collapses to low evidence for all labels.
- The model predicts fine label `1` positive less often than the hard-BCE control at the same fine-label-`0` FPR.
- The shuffled ambiguity-flag ablation matches the main model within noise.

What result would make me abandon the idea:

A same-backbone hard-BCE run that matches or exceeds the credal model on class-1 recall, ROC-AUC, PR-AUC, Brier/ECE, and `S` diagnostics should retire this family. Do not patch it with deeper backbones, ensembles, or extra uncertainty heads.

What result would justify scaling:

Scale only if the simple_18 minimal experiment passes the success threshold and the same trend appears on at least one 112-channel encoding without changing the objective.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_0750_credal_evidence/idea.yaml` | Create | Machine-readable idea metadata from the `idea_yaml` block below. |
| `ideas/20260421_0750_credal_evidence/math_thesis.md` | Create | Section 6, with formulas for `C_z`, Dirichlet mean, projection loss, evidence cap, and proof sketch. |
| `ideas/20260421_0750_credal_evidence/architecture.md` | Create | Section 7, tensor shapes, adapter rules, parameter estimates, pseudocode. |
| `ideas/20260421_0750_credal_evidence/implementation_notes.md` | Create | Fail-closed channel handling, logits from `log(alpha)`, prediction export columns, no move/engine features. |
| `ideas/20260421_0750_credal_evidence/trainer_notes.md` | Create | Custom loss requirements, fine-label weighting, deterministic settings, fair-comparison constraints. |
| `ideas/20260421_0750_credal_evidence/ablations.md` | Create | Section 9 ablation table and exact comparison protocol. |
| `ideas/20260421_0750_credal_evidence/train.py` | Create | Thin idea-specific entrypoint that loads the standard split, builds `CredalEvidencePuzzleNet`, applies `CredalEvidenceLoss`, and emits normal reports/predictions. |
| `ideas/20260421_0750_credal_evidence/config.yaml` | Create | Concrete config from `config_yaml` plus model-specific fields. |
| `ideas/20260421_0750_credal_evidence/report_template.md` | Create | Required metric table, `3x2` matrices, calibration, evidence-by-fine-label summaries, and ablation comparison slots. |
| `ideas/chatgpt_pro_deep_math_research_prompt.md` | Update | Preserve all hard constraints; add this packet’s fingerprint to imported memory after implementation; add anti-duplicate rules for credal/evidential fine-label-1 uncertainty heads if this fails. |
| `src/chess_nn_playground/models/credal_evidence.py` | Create | `CredalEvidencePuzzleNet`, `FailClosedBoardAdapter`, `TinyResidualBoardEncoder`, `DirichletEvidenceHead`; no full loss unless model-loss separation is existing style. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `credal_evidence_puzzle_net` builder. |
| `configs/credal_evidence_simple18.yaml` | Create | Standard training config pointing to `simple_18`, current split, 3 epochs, batch size 512, model name `credal_evidence_puzzle_net`. |
| `tests/test_credal_evidence_model.py` | Create | Focused tests: forward shape, positive alpha, logits finite, fail-closed unknown channels, deterministic output shape for 18 and explicitly configured 112. |
| `tests/test_credal_evidence_loss.py` | Create | Focused tests: near-puzzle loss zero when `mu>=tau` except evidence cap; positive when `mu<tau`; hard labels reduce to NLL; gradients finite. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0750_tuesday_los_angeles_credal_evidence.md
  generated_at: 2026-04-21 07:50:05 America/Los_Angeles
  weekday: Tuesday
  timezone: America/Los_Angeles
  idea_slug: credal_evidence
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_0750_credal_evidence
  name: Credal Near-Puzzle Evidence Network
  slug: credal_evidence
  status: draft
  created_at: 2026-04-21 07:50:05 America/Los_Angeles
  author: ChatGPT Pro
  short_thesis: Treat verified near-puzzles as interval-valued positive targets with lower evidential concentration instead of hard positives identical to verified puzzles.
  novelty_claim: Current-board encoder plus binary Dirichlet evidence head with credal fine-label-1 loss; not a sheaf, move-delta, transport, nuisance-projection, ordinal ladder, sparse witness, ray-language, constellation, or pseudo-likelihood model.
  expected_advantage: Better fine-label-1 recall at matched fine-label-0 false-positive rate and improved calibration without changing the official binary output contract.
  central_falsification_ablation: Same backbone with ordinary hard-positive BCE for fine label 1 and no evidence cap.
  target_task: coarse_binary
  input_representation: simple_18 first; lc0_static_112 and lc0_bt4_112 optional follow-ups with fail-closed adapters
  output_heads: binary Dirichlet evidence head returning logits from log(alpha)
  compute_notes: About 0.35M parameters and about 21M MACs per board for hidden_channels=64 and four residual blocks.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/credal_evidence_simple18.yaml
  model_path: src/chess_nn_playground/models/credal_evidence.py
  latest_result_path: null
  notes: Requires idea-specific loss using fine labels during training; model forward remains compatible with shared binary-logit reports.
```

```yaml
config_yaml:
  run:
    name: credal_evidence_simple18
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
    name: credal_evidence_puzzle_net
    input_channels: 18
    num_classes: 2
    hidden_channels: 64
    num_res_blocks: 4
    evidence_floor: 1.0
    near_tau: 0.55
    near_s_max: 6.0
    lambda_near_evidence_cap: 0.05
    lambda_dirichlet_kl: 0.001
    kl_anneal_epochs: 2
    dropout: 0.0
    allow_unknown_channels: false
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
  model_name: credal_evidence_puzzle_net
  file_path: src/chess_nn_playground/models/credal_evidence.py
  builder_function: build_credal_evidence_puzzle_net
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - FailClosedBoardAdapter
    - TinyResidualBoardEncoder
    - DirichletEvidenceHead
    - CredalEvidenceLoss
  required_config_fields:
    - input_channels
    - num_classes
    - hidden_channels
    - num_res_blocks
    - near_tau
    - near_s_max
    - lambda_near_evidence_cap
    - lambda_dirichlet_kl
    - allow_unknown_channels
  expected_parameter_count: about 0.35M for simple_18; about 0.36M for 112-channel encodings
  expected_memory_notes: No candidate sets; auxiliary evidence tensors are O(batch). Main activation memory is a small 64-channel 8x8 CNN.
```

```yaml
research_continuity:
  idea_fingerprint: current board tensor + compact CNN encoder + binary Dirichlet evidence head + singleton losses for fine labels 0/2 + credal positive interval and evidence cap for fine label 1
  already_researched_family_overlap: Touches label-safe uncertainty/selective prediction, but does not overlap with imported sheaf/Hodge, move-delta, OT, nuisance-projection, ordinal ladder, sparse-witness, ray-language, constellation, or pseudo-likelihood families.
  closest_duplicate_risk: Imported Ordinal Evidence Ladder Network, because both use fine-label semantics; distinction is that this model has one binary evidential distribution and treats label 1 as a credal positive interval with lower concentration, not as an ordered cumulative class.
  do_not_repeat_if_this_fails:
    - Do not propose another Dirichlet/evidential binary head that merely changes tau, KL weight, or evidence temperature.
    - Do not propose another fine-label-1-as-ambiguous-positive loss without a materially different falsifier.
    - Do not patch a failed result by adding a bigger CNN, ensemble, or LC0-style backbone as the core idea.
    - Do not repackage this as selective classification with an abstention class unless the benchmark contract is explicitly changed.
  suggested_next_search_directions:
    - Causal invariance across material/phase/source-like board strata with explicit chess-signal preservation tests.
    - Multi-view encoding consensus only after paired-encoding data plumbing is reliable.
    - Non-generative information bottlenecks that suppress dataset artifacts without closed-form nuisance projection.
    - Calibration diagnostics for near-puzzle ambiguity if this idea succeeds but classification gain is small.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Credal Near-Puzzle Evidence Network` to imported research memory after implementation, including its fingerprint. | Prevents future packets from repeating Dirichlet evidence plus fine-label-1 interval loss. | `Imported Research Memory` |
| Add an anti-duplicate rule: do not propose another evidential/Dirichlet/credal treatment of fine label 1 unless it changes the formal target set or falsifier. | Hyperparameter variants of `tau`, evidence cap, or KL weight would be low-value repeats. | `Research Continuity` or anti-duplicate paragraphs |
| Clarify that fine labels may be used in the loss but never as neural input features. | This packet relies on fine label 1 semantics during training; the prompt should distinguish supervision from leakage. | `Non-Negotiable Constraints` |
| Add near-puzzle matched-FPR diagnostics as a preferred benchmark item for ambiguity-focused ideas. | It gives future uncertainty/selective ideas a sharper target than overall accuracy. | `Required Markdown File Content` / benchmark section |
| Require evidence or uncertainty artifacts for any future uncertainty model. | Prevents uncertainty models from claiming calibration without exporting `mu`, confidence, and uncertainty diagnostics. | `Benchmark And Falsification Criteria` guidance |
| Keep the fail-closed adapter requirement for 112-channel encodings. | Avoids accidental interpretation of LC0 history planes as current-board deterministic geometry. | `Problem Restatement And Data Contract` |

Do not weaken leakage rules, label rules, falsification requirements, or anti-duplicate requirements.

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes
- Falsification criterion is concrete: yes
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported research packets completed: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Möbius-constellation, or pseudo-likelihood packets: yes
