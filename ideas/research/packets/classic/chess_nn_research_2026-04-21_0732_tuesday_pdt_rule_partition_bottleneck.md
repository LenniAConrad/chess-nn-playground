# Codex Handoff Packet: Side-Canonical Rule-Partition Invariant Bottleneck

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0732_tuesday_pdt_rule_partition_bottleneck.md`
- Generated at: 2026-04-21 07:32:12 PDT, America/Los_Angeles
- Weekday: Tuesday
- Timezone: America/Los_Angeles (`pdt` filename token)
- Idea slug: `rule_partition_bottleneck`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Side-Canonical Rule-Partition Invariant Bottleneck, abbreviated `SCRIB`.
- One-sentence thesis: Puzzle-likeness should be predicted from side-relative tactical structure that is stable across material phase, absolute color, and coarse material-balance environments, so force the latent code through a stochastic information bottleneck whose classifier risk is invariant across safe rule-derived partitions while an adversary removes partition-identifying shortcuts.
- Idea fingerprint: `simple_18 side-to-move canonicalization + rule-derived environment partitions over phase/material/absolute-color + variational latent bottleneck + V-REx risk-equalization + gradient-reversal environment adversaries + binary puzzle-like logits`.
- Why this is not a common CNN/ResNet/Transformer variant: the convolutional trunk is only a small feature extractor; the falsifiable mechanism is a learned stochastic causal-invariance operator over board-derived environments, not added depth, width, attention, attack incidence, move-delta pooling, transport, or an ensemble.
- Current-data minimal experiment: train on `data/splits/crtk_sample_3class/split_train.parquet`, validate on `split_val.parquet`, test on `split_test.parquet`, using only `simple_18`; map fine labels `0 -> 0`, `1/2 -> 1`; compare against existing simple CNN and residual CNN on the same split and epoch budget.
- Smallest central falsification ablation: keep the same network, bottleneck dimension, optimizer, and loss weights, but replace the semantic rule partitions with per-batch random partitions preserving group counts; if this matches the main model on near-puzzle diagnostics, the claimed rule-partition invariance is not doing chess-relevant work.
- Expected information gain if it fails: a clean failure says that board-derived causal-invariance pressure over phase/material/color is either too weak, too destructive, or not aligned with puzzle structure; future cycles should avoid this invariance family and return to mechanisms that model local tactical witnesses without suppressing global nuisance variables.

## 3. Problem Restatement And Data Contract

The task is binary chess puzzle-likeness classification from a single board position. The shared benchmark output is:

- output `0`: non-puzzle
- output `1`: puzzle-like

The available fine labels are:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

For the coarse binary task, use `y = 0` for fine label `0` and `y = 1` for fine labels `1` and `2`. Fine labels must still be preserved for diagnostic reporting, especially the required rectangular `3x2` matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Allowed neural-network input tensors remain `(batch, C, 8, 8)`, and the model must return logits shaped `(batch, num_classes)` with `num_classes = 2`. The default minimal experiment must use:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

Do not point the trainer at the roughly 45M-row full Parquet file until streaming support exists.

Allowed encodings already present in the project:

- `simple_18`: 12 piece planes + side-to-move + castling + en-passant.
- `lc0_static_112`.
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists.

Leakage checklist:

- Safe as input or deterministic preprocessing: deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and rule-derived current-board quantities such as material counts and side-relative canonicalization.
- Safe only as training-side nuisance labels, not as classifier inputs: rule-derived environment partitions over material phase, side-to-move color, and coarse material balance. These must be recomputed from the current board only.
- Leakage-prone unless explicitly justified and ablated: full legal-move generation, legal move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences. `SCRIB` does not use them.
- Always forbidden as neural-network inputs: Stockfish evaluations, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, or any field describing how the position was discovered.
- Fine labels may be used for supervised targets and diagnostics, not as input features.

Boundary between safe rule-derived features and leakage:

- Deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack geometry derived only from the current board are allowed. This idea does not need attack geometry.
- Full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences are leakage-prone unless explicitly justified as rule-only, label-independent, engine-free, and ablated. This idea avoids them.
- Engine evaluation, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, and dataset provenance must never be neural-network inputs.
- For `lc0_static_112` and `lc0_bt4_112`, current-board channels may be used for deterministic geometry only if Codex can prove the channel map. History channels may be consumed only by learned neural adapters; they must not be parsed by rule-partition code unless their semantics are explicit and current-board-only. Unknown semantics must fail closed.

## 4. Research Map

External ideas used:

1. Invariant Risk Minimization, Arjovsky, Bottou, Gulrajani, and Lopez-Paz, 2019. URL: https://arxiv.org/abs/1907.02893
   - Borrowed: the principle that a useful representation should admit the same optimal predictor across environments.
   - Not copied: the exact IRMv1 gradient-only implementation is not the sole mechanism; this packet uses a simpler V-REx-style risk variance penalty plus adversarial environment removal because it is easier to stabilize in the existing trainer.

2. Out-of-Distribution Generalization via Risk Extrapolation, Krueger et al., 2020/2021. URL: https://arxiv.org/abs/2003.00688
   - Borrowed: penalizing variance of risks across environments as a practical invariance surrogate.
   - Not copied: no external domains are assumed; the environments are deterministic chess-rule partitions of the same split.

3. Domain-Adversarial Training of Neural Networks, Ganin et al., JMLR 2016. URL: https://jmlr.org/papers/v17/15-239.html
   - Borrowed: a gradient-reversal adversary that makes a latent representation poor at predicting nuisance/domain labels.
   - Not copied: the goal is not domain adaptation between source and target datasets; source identity is forbidden and is not used.

4. Deep Variational Information Bottleneck, Alemi, Fischer, Dillon, and Murphy, 2016/2017. URL: https://arxiv.org/abs/1612.00410
   - Borrowed: a stochastic latent code with a KL penalty as an implementable proxy for limiting `I(X; Z)`.
   - Not copied: this is not a standalone VIB classifier; the bottleneck is coupled to chess-rule partitions and side-to-move canonicalization.

5. Group equivariant CNNs, Cohen and Welling, 2016. URL: https://arxiv.org/abs/1602.07576
   - Borrowed: the general idea that known symmetries should be enforced by architecture when safe.
   - Not copied: this is not a D4-equivariant board CNN. Chess is not D4-invariant because pawns, castling, file identity, and side-to-move matter. `SCRIB` uses only a side-to-move canonicalization under the safe color/vertical flip convention.

6. Contrastive-view theory, Tian et al., 2020. URL: https://proceedings.neurips.cc/paper/2020/hash/4c2e5eaae9152079b9e95845750bb9ab-Abstract.html
   - Borrowed: the warning that useful representations should discard nuisance information while preserving task-relevant information.
   - Not copied: no contrastive pretraining objective is proposed here.

Candidate search trace; serious mechanisms considered but not selected:

| Candidate mechanism | Why it lost to `SCRIB` |
|---|---|
| Class-conditional masked board description-length model | Too close to the imported static-geometry pseudo-likelihood / description-length ratio family; even without copying it, the central falsifier would overlap. |
| Evidential Beta selective classifier for near-puzzle ambiguity | Interesting and label-safe, but mostly a calibration/loss idea; it does not add a strong board-structure inductive bias and risks being a reporting improvement rather than a discovery mechanism. |
| Side-color equivariant Reynolds-averaged CNN | Clean and cheap, but too close to ordinary symmetry augmentation unless coupled to a stronger causal objective. The canonicalization part survives inside `SCRIB`; the averaged-CNN idea is not selected. |
| Fixed chess-metric heat-kernel network over square movement operators | Mathematically distinct from sheaves, but it drifts toward static attack/mobility geometry and would likely be rejected as an attack-graph cousin without a sharper falsifier. |
| Cubical persistent-homology or Euler-characteristic features over occupancy fields | Novel, but hard to make differentiable and probably too weak on an 8x8 board; failure would be uninformative because implementation approximations would dominate. |
| Learned probabilistic circuit over side-relative piece grammar | Potentially valuable, but it overlaps with generative compression and pseudo-likelihood families and would require more infrastructure than the current sample benchmark warrants. |
| Hypergraph cut or max-flow around king zones | Too adjacent to tactical attack-defense graph families and would need careful legal-move semantics to avoid hidden search features. |
| Multi-encoding contrastive learner over `simple_18` and `lc0_bt4_112` | Promising, but it depends on reliable synchronized multi-encoding access and precise LC0 channel semantics. `SCRIB` can run now on `simple_18` and only later extend to LC0 encodings. |

Internal sweep note: additional discarded directions included plain token Transformers, larger residual towers, ensembling, optimizer schedules, one-ply move-set MIL, optimal-transport pressure maps, Hodge/sheaf variants, high-order ANOVA interactions, sparse witness masking, and ray-language automata. They are rejected explicitly below or already covered by imported research memory.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Existing simple CNN | `src/chess_nn_playground/models/cnn.py` | Already implemented; it tests generic local pattern recognition, not a new causal-invariance hypothesis. |
| Existing residual CNN | `src/chess_nn_playground/models/residual_cnn.py` | Already implemented; adding residual blocks would be ordinary capacity scaling. |
| LC0-style CNN / residual CNN | Existing LC0 BT4-style variants | Too close to copying LC0-like inductive bias; the project already has this baseline family. |
| Ordinary ViT over 64 squares | No exact baseline, but common square-token Transformer | Too generic and likely data-hungry; it lacks a chess-specific falsifiable operator. |
| Plain GNN on square adjacency | Common graph baseline | Too ordinary and weakly aligned with chess tactics unless attack/move edges are added, which would move toward already imported attack-graph/sheaf families. |
| Hyperparameter tuning, width/depth sweeps, optimizer tuning | All existing baselines | Not a research mechanism; failure or success would not explain puzzle-likeness. |
| Ensembling several existing classifiers | Any existing model ensemble | Not allowed as the core idea and gives little mechanistic information. |
| Another tactical sheaf/Hodge/tension/curvature network | Imported sheaf/Hodge packets | Explicitly covered by imported packets; edge-label or pooling changes would be duplicate research. |
| Another one-ply move-delta DeepSets/attention/spectrum model | Imported counterfactual move-delta packets | Already researched; changing pooling or energy names would not create a new formal object. |
| Another Sinkhorn or piece-target optimal-transport bottleneck | Imported OT packets | Already researched; cost-temperature or target-bucket changes would be duplicates. |
| Closed-form nuisance residualization from material/phase/king vectors | Imported nuisance-orthogonal packet | `SCRIB` uses stochastic minimax invariance and risk equalization, not a deterministic ridge projection; the closed-form projection route should not be repeated here. |
| Ordinal cumulative evidence ladder | Imported ordinal packet | The current idea is not an ordinal constraint over fine labels; it uses binary supervision plus environment invariance. |
| Sparse witness-piece bottleneck or top-k masked board classifier | Imported sparse-witness packet | Already researched; the present idea should not reuse masked witness selection as the central mechanism. |
| Ray-language automaton | Imported ray-language packet | Already researched and ray-token shuffles would duplicate its falsifiers. |
| Möbius / ANOVA piece-constellation interactions | Imported constellation packet | Already researched; pair/triple interaction tensors are not the selected direction. |
| Static board pseudo-likelihood / description-length ratio | Imported pseudo-likelihood packet | Too close to generative board compression; not selected despite surface appeal. |

## 6. Mathematical Thesis

### Input space definition

For the minimal experiment, let

\[
X \subset \{0,1\}^{18 \times 8 \times 8}
\]

be the set of `simple_18` encoded positions. A sample is `(x, f)` where `f in {0,1,2}` is the fine label. The binary target is

\[
Y = \mathbf{1}\{f \in \{1,2\}\} \in \{0,1\}.
\]

The model never receives `f`, source identifiers, engine outputs, verification metadata, proposed labels, or dataset provenance as input.

### Side-to-move canonicalization

Define a deterministic map

\[
C : X \to X_c \subset \{0,1\}^{17 \times 8 \times 8}
\]

that rewrites the board from the side-to-move perspective:

- moving-side pieces become the six friendly piece planes;
- nonmoving-side pieces become the six enemy piece planes;
- if black is to move, ranks are vertically flipped so friendly pawns have the same forward direction as white pawns in the canonical tensor;
- castling rights become friendly/enemy kingside/queenside rights;
- en-passant is vertically flipped consistently;
- the absolute side-to-move plane is removed from the canonical tensor.

This is not full rotation/reflection invariance. Horizontal file mirror, 90-degree rotation, and D4 augmentation are not assumed safe because pawns, king/queen files, castling, and side-to-move break those symmetries.

### Rule partitions

Define a deterministic partition map

\[
\Pi(x) = (E_{phase}(x), E_{adv}(x), E_{color}(x)),
\]

where:

- `E_phase in {0,1,2}` is a coarse total non-king material bucket, for example endgame / middlegame / high-material.
- `E_adv in {0,1,2,3,4}` is a side-relative material-balance bucket using standard material values `P=1, N=3, B=3, R=5, Q=9, K=0`, clipped into large-disadvantage, small-disadvantage, equal, small-advantage, large-advantage.
- `E_color in {0,1}` is the absolute color to move before canonicalization.

`Pi` is used only to define losses and diagnostics. It is not concatenated to the classifier input.

For risk equalization, define a coarser group id

\[
E(x) = E_{phase}(x) + 3 E_{adv}(x) + 15 E_{color}(x) \in \{0,\ldots,29\}.
\]

Groups with fewer than `min_group_count` samples in a batch are ignored in the V-REx term for that batch.

### Data distribution assumptions

Assume the observed position can be decomposed abstractly as

\[
C(X) = G(T, N, \epsilon),
\]

where:

- `T` is side-relative tactical structure relevant to puzzle-likeness;
- `N` contains nuisance features such as global phase, coarse material balance, and absolute color imbalance in the sample construction process;
- `epsilon` is ordinary observation variability;
- `Y` depends primarily on `T`, while correlations between `Y` and `N` may differ across partitions and may include dataset artifacts.

This is a hypothesis, not a proven fact about the dataset.

### Allowed symmetry or equivariance assumptions

The only built-in symmetry is color/side canonicalization: white-to-move and black-to-move positions should be represented in a shared side-relative coordinate system. This is weaker than full equivariance. No assumption is made that file reflection, rank reflection without color swap, rotation, or arbitrary board permutation preserves labels.

### Core hypothesis

A puzzle-like position is not merely a high-material, low-material, white-to-move, black-to-move, winning, losing, or source-specific artifact. It should retain predictive evidence after the representation is compressed and made approximately invariant to coarse rule-derived nuisance partitions. Therefore, a model that must predict through a small stochastic code `Z` while equalizing risk across `Pi` groups should improve near-puzzle behavior, especially class-`1` recall at a matched fine-label-`0` false-positive rate.

### Formal object introduced

The central object is the rule-partition invariant bottleneck operator

\[
\mathcal{B}_{\phi,\psi}(x)
= h_\psi(Z), \quad Z \sim q_\phi(z \mid C(x)),
\]

trained with the minimax objective

\[
\min_{\phi,\psi}\max_{a_1,a_2,a_3}
\;\mathcal{L}_{cls}
+ \beta \mathcal{L}_{KL}
+ \lambda \mathcal{L}_{V\text{-}REx}
- \gamma \mathcal{L}_{adv}.
\]

Here:

\[
\mathcal{L}_{cls} = \mathbb{E}_{(x,y)}[CE(y, h_\psi(z))],
\]

\[
\mathcal{L}_{KL} = \mathbb{E}_{x}\left[ KL(q_\phi(z\mid C(x)) \Vert \mathcal{N}(0,I)) \right],
\]

\[
\mathcal{L}_{V\text{-}REx} = Var_{e \in \mathcal{E}_{batch}}\left( R_e \right),
\quad
R_e = \mathbb{E}[CE(y,h_\psi(z)) \mid E(x)=e],
\]

and

\[
\mathcal{L}_{adv} = CE(E_{phase}, a_1(z)) + CE(E_{adv}, a_2(z)) + CE(E_{color}, a_3(z)).
\]

The adversarial term is implemented by gradient reversal: the adversary heads learn to predict the environment labels, while the encoder receives the negative of that gradient.

### Proposition under an idealized distribution

Suppose `Y` is conditionally independent of the environment partition given a latent tactical variable `T`:

\[
Y \perp E \mid T,
\]

and suppose there exists a representation `Z_T = r(T)` such that a classifier `h` achieves Bayes risk `R*` from `Z_T` and `I(Z_T;E)=0`. Suppose also that any additional representation component `S` encoding only partition-measurable nuisance information cannot lower Bayes risk below `R*` but has `I(S;E)>0`. Then, among representations with Bayes risk `R*`, minimizing an objective of the form

\[
R(h(Z)) + \gamma I(Z;E) + \beta I(Z;X)
\]

prefers a representation that discards `S`.

### Proof sketch or derivation

Because `Y` is conditionally independent of `E` given `T`, `E`-measurable nuisance features do not improve the optimal conditional predictor once `T` is represented. Thus `Z_T` reaches risk `R*`. Any representation `(Z_T,S)` with `S` partition-measurable also reaches at best `R*`, but its mutual information with `E` is larger because `I((Z_T,S);E) >= I(S;E) > 0` when `Z_T` is independent of `E`. It also cannot reduce the compression penalty `I(Z;X)`. Therefore the objective prefers the smaller invariant representation. The KL term is the variational proxy for `I(Z;X)`, the environment adversary is a proxy for reducing recoverable information about `E`, and V-REx penalizes the empirical symptom of partition-specific shortcuts: unequal risks across `E` groups.

### What is actually proven

Only the idealized proposition above is proven. It states that, under strong assumptions, an information bottleneck plus environment-invariance pressure rejects purely nuisance features that do not improve Bayes risk.

### What remains only hypothesized

- The dataset's spurious correlations are sufficiently aligned with the proposed partitions.
- The side-relative tactical variable `T` is learnable from `simple_18` with a small CNN trunk.
- V-REx and adversarial environment heads are stable enough at the current sample size.
- Removing recoverable material/phase/color information will improve near-puzzle diagnostics rather than deleting useful tactical context.

### Counterexamples where the idea should fail

- If verified puzzles in this split are genuinely defined by phase or material balance rather than tactical structure, suppressing phase/material information will hurt.
- If near-puzzles are mostly ambiguous annotation cases rather than a coherent structural class, invariance may not help class-`1` recall.
- If class `0` and classes `1/2` differ by dataset source in a way not captured by material phase, material balance, or absolute color, the chosen partitions will miss the artifact.
- If the model needs exact absolute color information due to an exporter convention or a biased puzzle source, side-canonicalization may remove a predictive shortcut that benchmarks reward.
- If the batch size is too small for stable group risk estimates, V-REx noise may dominate.

### Self-critique

The strongest objection is that material phase and material advantage are not merely nuisance variables in chess. A sacrifice puzzle, a back-rank motif, or a promotion tactic may require phase and material context. `SCRIB` could over-regularize and make the classifier bland. The minimal experiment is still worth running because the claim is sharply falsifiable: semantic partitions must beat random partitions and no-invariance controls on near-puzzle recall at matched non-puzzle false-positive rate. If they do not, the result is informative and the family should be retired.

## 7. Architecture Specification

### Module names

Implement in `src/chess_nn_playground/models/rule_partition_invariant_bottleneck.py`:

- `Simple18SideCanonicalizer`
- `Simple18RulePartitioner`
- `GradientReversalFn`
- `GradientReversalLayer`
- `ConvTinyBackbone`
- `VariationalBottleneck`
- `EnvAdversaryHead`
- `RulePartitionInvariantBottleneckNet`
- builder function: `build_rule_partition_invariant_bottleneck(config)`

The model's ordinary `forward(x)` must return only binary logits so the shared trainer and inference code remain compatible. For the idea-specific training script, support `forward_with_aux(x, sample=True)` returning logits plus auxiliary tensors.

### Forward-pass steps and tensor shapes

Minimal `simple_18` run:

1. Input: `x` shaped `(B, 18, 8, 8)`.
2. `Simple18SideCanonicalizer(x)`:
   - parse 12 piece planes, side-to-move plane, four castling planes, and one en-passant plane;
   - output `xc` shaped `(B, 17, 8, 8)` with friendly/enemy piece planes, friendly/enemy castling planes, and canonical en-passant plane.
3. `Simple18RulePartitioner(x)` for training only:
   - output `phase_labels`: `(B,)`, integer in `[0,2]`;
   - output `adv_labels`: `(B,)`, integer in `[0,4]`;
   - output `color_labels`: `(B,)`, integer in `[0,1]`;
   - output `group_ids`: `(B,)`, integer in `[0,29]`.
4. `ConvTinyBackbone(xc)`:
   - `Conv2d(17, 64, kernel=3, padding=1)`, normalization, GELU: `(B,64,8,8)`;
   - two residual micro-blocks at width 64: `(B,64,8,8)`;
   - `Conv2d(64,96,kernel=3,padding=1)`, normalization, GELU: `(B,96,8,8)`;
   - two residual micro-blocks at width 96: `(B,96,8,8)`;
   - concatenate global mean pool and global max pool: `(B,192)`;
   - MLP `192 -> 256`, GELU, dropout: `(B,256)`.
5. `VariationalBottleneck`:
   - `mu = Linear(256, latent_dim)`: `(B,128)` by default;
   - `logvar = Linear(256, latent_dim)`: `(B,128)`;
   - during training, `z = mu + exp(0.5*logvar)*eps`; during evaluation, `z = mu`;
   - KL per sample to `N(0,I)`: `(B,)`.
6. Label head:
   - `LayerNorm(128) -> Linear(128,64) -> GELU -> Dropout(0.1) -> Linear(64,2)`;
   - output logits `(B,2)`.
7. Environment adversaries, training only:
   - apply `GradientReversalLayer(lambda=env_grl_lambda)` to `z`;
   - phase head logits `(B,3)`;
   - material-advantage head logits `(B,5)`;
   - absolute-color head logits `(B,2)`.

### Parameter-count estimate

With default width `(64,96)` and `latent_dim=128`, expected trainable parameters are approximately `0.70M` to `0.80M`, depending on normalization choices. This is deliberately comparable to a small residual CNN; success should not be credited to large capacity.

### FLOP / complexity estimate

For `B` samples and 8x8 boards, the convolutional trunk costs roughly `35M` multiply-accumulate operations per sample with the default widths. The latent and head MLPs are negligible by comparison. Total complexity is `O(B * 8 * 8 * W^2 * L)` for trunk width `W` and number of micro-blocks `L`.

### Candidate-set memory and chunking

No generated move set, candidate set, graph edge set, hyperedge set, transport matrix, or search surrogate is used. Memory beyond the input is `O(B * latent_dim)` for the bottleneck and `O(B)` for environment labels. No chunking plan is required for candidate objects.

### Required config fields

Add these under `model` or an idea-specific subkey:

```yaml
name: rule_partition_invariant_bottleneck
input_channels: 18
num_classes: 2
encoding: simple_18
canonicalizer: simple18_side_to_move
latent_dim: 128
backbone_widths: [64, 96]
bottleneck_beta: 0.0005
vrex_weight: 0.2
env_adv_weight: 0.05
env_grl_lambda: 1.0
min_group_count: 8
dropout: 0.1
```

Training script fields:

```yaml
warmup_epochs_without_invariance: 1
random_partition_ablation: false
no_invariance_ablation: false
kl_anneal_epochs: 2
```

### Encoding support

First experiment should use only `simple_18`. This is intentional. The idea depends on parsing side-to-move, piece colors, castling, and en-passant safely. `simple_18` has the cleanest semantics.

Adapter assumptions:

- `simple_18`: allowed for canonicalization and rule partitions. Codex must verify channel ordering against the existing encoder tests or docs before training. If the order differs from the assumed 12 piece + side + castling + en-passant layout, update the adapter and tests before running.
- `lc0_static_112`: not supported for deterministic rule partitions unless Codex can prove which planes are current-board piece, side, castling, and en-passant planes. If unsupported, fail closed with a clear error.
- `lc0_bt4_112`: history planes may be consumed by a learned neural adapter only. The deterministic rule partitioner must use only proven current-board channels. Until exporter support and channel metadata are explicit, run `SCRIB` on `simple_18` only.

### Logit compatibility

`RulePartitionInvariantBottleneckNet.forward(x)` must return only `logits` shaped `(B,2)`. The idea-specific trainer may call `forward_with_aux` to compute KL, adversarial, and V-REx losses, but saved checkpoints must be loadable for ordinary prediction and reporting.

### Pseudocode

```python
# Pseudocode only; do not paste as final implementation.
class RulePartitionInvariantBottleneckNet(nn.Module):
    def forward(self, x):
        logits, _aux = self.forward_with_aux(x, sample=self.training, need_aux=False)
        return logits

    def forward_with_aux(self, x, sample=True, need_aux=True):
        xc = self.canonicalizer(x)             # (B,17,8,8)
        h = self.backbone(xc)                  # (B,256)
        mu, logvar, z, kl = self.vib(h, sample=sample)
        logits = self.label_head(z)            # (B,2)
        if not need_aux:
            return logits, {"kl": kl, "mu": mu, "logvar": logvar}
        env = self.partitioner(x)              # phase, adv, color, group ids
        z_rev = self.grl(z)
        env_logits = {
            "phase": self.phase_head(z_rev),  # (B,3)
            "adv": self.adv_head(z_rev),      # (B,5)
            "color": self.color_head(z_rev),  # (B,2)
        }
        return logits, {"kl": kl, "env": env, "env_logits": env_logits}
```

## 8. Loss, Training, And Regularization

### Primary loss

Use weighted binary cross-entropy via `torch.nn.CrossEntropyLoss(weight=class_weights)` over two logits, with coarse target `0` for fine label `0`, and `1` for fine labels `1` and `2`.

### Auxiliary losses

Main `SCRIB` loss:

```text
loss = CE_y
     + beta(t) * mean(KL)
     + vrex_weight(t) * variance_of_group_CE
     + env_adv_weight(t) * CE_env_for_gradient_reversal
```

Where gradient reversal makes the encoder maximize environment CE while the adversary heads minimize it. The sign in ordinary code should be positive for `CE_env` after a GRL layer.

Optional but recommended schedule:

- epoch 1: `beta=0`, `vrex_weight=0`, `env_adv_weight=0`; train the classifier to avoid early collapse.
- epoch 2 onward: linearly anneal `beta`, `vrex_weight`, and `env_adv_weight` to their configured values.

### Class weighting

Use the existing `balanced` class weighting for the coarse binary label. Do not separately overweight fine label `1` unless an explicit ablation is added, because that would confound the near-puzzle diagnostic.

### Batch size expectations

Default batch size: `512`. V-REx needs enough samples per environment. For each batch, compute group losses only for groups with at least `min_group_count=8`. If fewer than two valid groups remain, set the V-REx term to zero for that batch but keep CE, KL, and adversary losses.

### Optimizer defaults

- Optimizer: `AdamW`.
- Learning rate: `0.001`.
- Weight decay: `0.0001`.
- Epochs for minimal experiment: `3`, matching current benchmark configs unless Codex has a standard baseline budget.
- Early stopping patience: `2`.
- Mixed precision: off for the first run to reduce nondeterminism.

### Regularizers

- KL bottleneck with default `beta=0.0005`; tune only after the falsification ablations.
- Dropout `0.1` in the bottleneck MLP and label head.
- Weight decay as above.
- Gradient clipping at global norm `5.0` in the idea-specific trainer, because GRL plus KL can spike gradients.

### Determinism requirements

Use seed `42`, deterministic PyTorch flags where already supported, fixed split files, and no random data augmentation in the main run. The random-partition ablation must log its seed and partition construction.

### What must stay unchanged for fair comparison

- Same train/val/test split.
- Same binary label mapping.
- Same `simple_18` encoding for the minimal experiment.
- Same epoch budget and early stopping rule as the strongest existing small baseline that Codex compares against.
- Same report code, including coarse metrics and fine-label `3x2` matrix.
- No extra data, no full 45M-row training, no engine features, no source labels, no verification metadata.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Random partitions, central falsifier | Replace `E_phase`, `E_adv`, and `E_color` with random labels preserving per-batch group counts; keep all loss weights and architecture. | Chess-semantic rule partitions, not generic regularization noise, are responsible for any gain. | If random partitions match or beat main `SCRIB`, abandon the semantic rule-partition claim. |
| No invariance losses | Set `vrex_weight=0` and `env_adv_weight=0`, keep canonicalization and KL. | Risk equalization and adversarial partition removal matter beyond canonicalized VIB regularization. | If this wins, the invariance pressure is destructive or unnecessary. |
| No KL bottleneck | Set `bottleneck_beta=0`, keep V-REx and adversaries. | Compression is needed to prevent the model from hiding partition information in unused latent dimensions. | If no-KL wins and adversary leakage remains low, the bottleneck is not needed. If no-KL wins but adversary accuracy is high, the model is exploiting hidden nuisances. |
| No side canonicalization | Feed original `simple_18` planes to the same trunk, adjusting first conv channels to 18; keep partitions. | Side-relative coordinates reduce absolute-color shortcuts while preserving tactical structure. | If no-canonicalization wins, the canonicalizer may be discarding useful orientation or implemented incorrectly. |
| Color-only invariance | Use only `E_color` adversary and no phase/material partitions. | Absolute color is the dominant nuisance; material partitions may over-regularize. | If color-only wins, future work should use symmetry/canonicalization but not material invariance. |
| Material-only invariance | Use `E_phase` and `E_adv`, drop `E_color`. | Material/phase artifacts matter beyond absolute side-to-move color. | If material-only wins, color adversary may be redundant after canonicalization. |
| Count-preserving material shuffle | Shuffle material-balance labels within each coarse binary class and side-to-move color, preserving label counts and color marginals. | The actual relation between material partition and board content matters. | If shuffled material labels work, the adversary is just adding noise regularization. |
| Group-risk only | Use V-REx but no adversary heads. | Equalizing predictive risk is enough without forcing latent non-identifiability. | If this wins, adversarial removal is too strong; use risk-only invariant learning. |
| Adversary-only | Use adversary heads but no V-REx. | Latent nuisance removal is enough without risk equalization. | If this wins, V-REx batch noise is the issue. |
| Matched plain CNN capacity | Increase/decrease the baseline CNN to match `SCRIB` parameter count without bottleneck or invariance. | Any observed gain is not just parameter count. | If matched CNN wins, the idea provides no mechanistic benefit. |
| Fine-label-blind diagnostic run | Train exactly as binary but hide fine labels from all training-time logs except final diagnostics. | Fine labels are not leaking into model selection or loss. | If results change, the pipeline was inadvertently using fine-label information. |

This idea has no graph, hypergraph, sheaf, transport matrix, move-set, or search-surrogate object. The semantics-destroying control is therefore the random-partition and count-preserving material-shuffle ablations, which preserve group counts, material marginals where specified, side-to-move marginals, and binary labels while destroying the proposed environment semantics.

## 10. Benchmark And Falsification Criteria

### Baselines to compare against

Codex should compare `SCRIB` to:

1. existing simple CNN with `simple_18`;
2. existing residual CNN with `simple_18`, matched as closely as possible in parameter count;
3. existing LC0 BT4-style CNN or residual CNN only as a contextual reference, not as the primary fair comparison because the minimal `SCRIB` run uses `simple_18`;
4. `SCRIB` ablations listed above, at minimum central random partitions, no invariance losses, no KL, and no side canonicalization.

### Metrics to inspect

- Test accuracy.
- Test macro F1.
- Test AUROC and average precision for binary puzzle-like prediction.
- Brier score or expected calibration error if available.
- Required fine-label `3x2` confusion matrix for every main and central ablation run.
- Fine label `1` recall at matched fine-label-`0` false-positive rate.
- Fine label `2` recall at matched fine-label-`0` false-positive rate.
- Worst-group validation loss across the 30 semantic groups, with groups below a minimum support threshold excluded from the worst-group calculation.
- Environment adversary accuracies on validation. They should be below a no-GRL control but do not need to reach chance if classification requires some correlated context.

### Near-puzzle diagnostic

Primary near-puzzle diagnostic:

```text
class_1_recall_at_fine0_fpr = 5%
```

If the report system cannot threshold at exactly `5%`, compute class-`1` recall at the closest threshold whose fine-label-`0` false-positive rate is no greater than `5%`. Also report the same statistic at `1%` and `10%` if easy.

### Required artifacts

For the main model and each central ablation, save:

- config YAML used for the run;
- model checkpoint path;
- train/val/test metrics JSON;
- per-epoch training log with CE, KL, V-REx, and adversary losses;
- binary confusion matrix;
- fine-label `3x2` diagnostic matrix;
- prediction Parquet or CSV with row id, fine label, binary target, logits, probability, prediction, and environment group id;
- near-puzzle threshold report;
- environment group metrics table;
- ablation comparison Markdown report.

### Success threshold

Call the idea successful enough to scale if all of the following hold:

- `SCRIB` improves fine-label-`1` recall at matched fine-label-`0` FPR by at least `2` absolute percentage points over the strongest matched `simple_18` baseline.
- It does not reduce fine-label-`2` recall at the same FPR by more than `1` absolute percentage point.
- Test AUROC or average precision is at least as good as the strongest matched `simple_18` baseline within `0.5` percentage points, preferably better.
- It beats the random-partition central ablation on the near-puzzle diagnostic by at least `1` absolute percentage point.
- It shows lower validation worst-group loss than the no-invariance ablation, or at least a smaller worst-group / average-group loss gap.

### Failure threshold

Treat the idea as failed if any of these occur:

- The random-partition ablation matches or beats the main model on class-`1` recall at matched fine-label-`0` FPR.
- The no-invariance ablation beats the main model on both AUROC and near-puzzle recall.
- Fine-label-`2` recall drops by more than `3` absolute percentage points at matched FPR.
- Environment adversary losses destabilize training, causing validation loss to diverge in two seeded attempts.
- The required `3x2` diagnostics show that gains come only from predicting nearly everything as puzzle-like.

### What result would make me abandon the idea

Abandon this family if semantic partitions do not beat randomized partitions and if the no-invariance ablation is equal or better. In that case, future research should not repeat `simple_18` side-canonicalization plus material/phase/color adversarial or V-REx bottlenecks with different weights, extra heads, or larger backbones.

### What result would justify scaling

Scale only if the success threshold is met on the sample split. Scaling steps, in order:

1. Run three seeds on the current split.
2. Add a matched LC0 learned-adapter version only after channel semantics are explicit and fail-closed tests exist.
3. Increase epochs modestly while keeping the same ablations.
4. Only after streaming support exists, test on a larger split without changing leakage constraints.

## 11. Implementation Plan For Codex

Use idea id `20260421_0732_rule_partition_bottleneck`.

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_0732_rule_partition_bottleneck/idea.yaml` | Create | Machine-readable idea metadata copied from the `idea_yaml` block below. |
| `ideas/20260421_0732_rule_partition_bottleneck/math_thesis.md` | Create | Section 6 mathematical thesis, assumptions, proof sketch, counterexamples, and self-critique. |
| `ideas/20260421_0732_rule_partition_bottleneck/architecture.md` | Create | Section 7 architecture, tensor shapes, adapter assumptions, parameter count, and pseudocode. |
| `ideas/20260421_0732_rule_partition_bottleneck/implementation_notes.md` | Create | Notes on `simple_18` channel verification, fail-closed LC0 behavior, no engine metadata, and no source labels. |
| `ideas/20260421_0732_rule_partition_bottleneck/trainer_notes.md` | Create | Loss construction, KL annealing, V-REx batch handling, GRL sign checks, class weighting, and deterministic settings. |
| `ideas/20260421_0732_rule_partition_bottleneck/ablations.md` | Create | Section 9 ablation table plus exact central random-partition control. |
| `ideas/20260421_0732_rule_partition_bottleneck/train.py` | Create | Idea-specific training entrypoint that loads the standard split, constructs coarse labels, trains with CE + KL + V-REx + GRL adversary losses, and calls existing report utilities. It must preserve `forward(x)->logits` compatibility. |
| `ideas/20260421_0732_rule_partition_bottleneck/config.yaml` | Create | Filled config based on the `config_yaml` block plus extra model/loss fields from Section 7. |
| `ideas/20260421_0732_rule_partition_bottleneck/report_template.md` | Create | Report skeleton requiring metrics, `3x2` fine-label matrices, near-puzzle threshold diagnostics, group metrics, and ablation comparison. |
| `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this packet to imported research memory after implementation; add anti-duplicate guidance for side-canonical rule-partition invariant bottlenecks if it fails. Preserve all hard leakage and anti-duplicate rules. |
| `src/chess_nn_playground/models/rule_partition_invariant_bottleneck.py` | Create | Implement modules listed in Section 7. Keep model independent of trainer-specific labels except `forward_with_aux`. |
| `src/chess_nn_playground/models/registry.py` | Modify | Register `rule_partition_invariant_bottleneck` and builder function. |
| `configs/rule_partition_invariant_bottleneck_simple18.yaml` | Create | Main benchmark config for `simple_18`, binary mode, seed 42, batch size 512, 3 epochs, balanced class weighting. |
| `configs/rule_partition_invariant_bottleneck_simple18_random_partitions.yaml` | Create | Central falsification config with random partition ablation enabled. |
| `configs/rule_partition_invariant_bottleneck_simple18_no_invariance.yaml` | Create | No-invariance ablation config. |
| `configs/rule_partition_invariant_bottleneck_simple18_no_kl.yaml` | Create | No-KL ablation config. |
| `tests/test_rule_partition_invariant_bottleneck.py` | Create | Focused tests for tensor shapes, deterministic canonicalization, partition ranges, GRL-safe forward pass, `forward(x)->(B,2)`, fail-closed unsupported encodings, and no use of forbidden metadata. |
| `tests/test_simple18_side_canonicalizer.py` | Create if useful | Unit tests using synthetic piece planes to verify white-to-move identity behavior, black-to-move vertical flip, friendly/enemy color swap, castling remap, and en-passant remap. |

For `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md`, Codex should append a concise memory entry after consuming this packet:

```text
Side-Canonical Rule-Partition Invariant Bottleneck:
simple_18 side-to-move canonicalization + rule-derived phase/material/absolute-color environments + variational bottleneck + V-REx risk variance + gradient-reversal environment adversaries + binary puzzle-like target. If it fails, do not repeat by only changing environment buckets, weights, latent size, or backbone depth.
```

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0732_tuesday_pdt_rule_partition_bottleneck.md
  generated_at: 2026-04-21 07:32:12 PDT
  weekday: Tuesday
  timezone: America/Los_Angeles
  idea_slug: rule_partition_bottleneck
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_0732_rule_partition_bottleneck
  name: Side-Canonical Rule-Partition Invariant Bottleneck
  slug: rule_partition_bottleneck
  status: draft
  created_at: 2026-04-21 07:32:12 PDT
  author: ChatGPT Pro
  short_thesis: Predict puzzle-likeness through a compressed side-relative latent code whose risk is stable across safe rule-derived phase, material-balance, and absolute-color partitions.
  novelty_claim: This is a causal-invariance and stochastic-bottleneck mechanism over deterministic chess rule partitions, not a sheaf, move-delta set, transport model, nuisance projection, ordinal head, sparse witness, ray automaton, high-order constellation, or pseudo-likelihood model.
  expected_advantage: Better class-1 near-puzzle recall at matched fine-label-0 false-positive rate by suppressing source-like phase/material/color shortcuts while preserving side-relative tactical evidence.
  central_falsification_ablation: Replace semantic rule partitions with random partitions preserving group counts; if performance matches, abandon the rule-partition invariance claim.
  target_task: coarse_binary
  input_representation: simple_18
  output_heads: binary_logits_plus_training_only_environment_adversaries
  compute_notes: About 0.7M-0.8M parameters; no candidate sets; O(B*latent_dim) auxiliary memory; default batch size 512.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/rule_partition_invariant_bottleneck_simple18.yaml
  model_path: src/chess_nn_playground/models/rule_partition_invariant_bottleneck.py
  latest_result_path: null
  notes: Use idea-specific train.py for auxiliary losses, but keep model.forward(x) returning logits for shared reports and inference.
```

```yaml
config_yaml:
  run:
    name: rule_partition_invariant_bottleneck_simple18
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
    name: rule_partition_invariant_bottleneck
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
  model_name: rule_partition_invariant_bottleneck
  file_path: src/chess_nn_playground/models/rule_partition_invariant_bottleneck.py
  builder_function: build_rule_partition_invariant_bottleneck
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18SideCanonicalizer
    - Simple18RulePartitioner
    - ConvTinyBackbone
    - VariationalBottleneck
    - GradientReversalLayer
    - EnvAdversaryHead
    - RulePartitionInvariantBottleneckNet
  required_config_fields:
    - input_channels
    - num_classes
    - encoding
    - canonicalizer
    - latent_dim
    - bottleneck_beta
    - vrex_weight
    - env_adv_weight
    - env_grl_lambda
    - min_group_count
  expected_parameter_count: about 0.70M to 0.80M
  expected_memory_notes: No generated candidate set; auxiliary memory is O(batch * latent_dim) plus O(batch) environment labels; with batch 512 and latent_dim 128 this is small relative to convolution activations.
```

```yaml
research_continuity:
  idea_fingerprint: simple_18 side-to-move canonicalization + deterministic phase/material/absolute-color rule partitions + variational information bottleneck + V-REx group risk equalization + gradient-reversal environment adversaries + binary puzzle-like logits
  already_researched_family_overlap: Touches nuisance suppression but is not the imported closed-form nuisance-orthogonal projection; touches symmetry but is not a sheaf/Hodge/file-mirror model; does not use move-delta sets, optimal transport, ordinal heads, sparse witnesses, ray automata, Möbius constellations, or pseudo-likelihood ratios.
  closest_duplicate_risk: The imported Nuisance-Orthogonal Puzzle Bottleneck is the nearest risk. The distinction is stochastic learned minimax invariance plus group-risk equalization, not closed-form ridge residualization of latents against a deterministic nuisance vector.
  do_not_repeat_if_this_fails:
    - Do not retry the same side-canonical phase/material/color invariant bottleneck with only different bucket thresholds.
    - Do not retry it with only larger CNN width, deeper residual blocks, or longer training.
    - Do not replace V-REx with IRMv1 or group DRO as the only change unless the failure analysis specifically shows V-REx instability rather than semantic-partition irrelevance.
    - Do not add castling, king-square, or pawn-count adversaries as the only novelty; that would be the same family.
    - Do not claim success unless semantic partitions beat random/count-preserving partition controls.
  suggested_next_search_directions:
    - Label-safe selective prediction for fine-label-1 ambiguity, with no ordinal ladder duplicate.
    - Non-generative uncertainty models that expose unresolved boundary cases without fabricating labels.
    - Current-board masked compression only if clearly distinct from the imported pseudo-likelihood ratio family.
    - Mechanisms using safe multi-view encoding consistency once LC0 channel semantics are explicit.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add the `SCRIB` fingerprint to imported research memory after implementation. | Prevents a future pass from repeating side-canonical phase/material/color invariant bottlenecks with renamed losses. | `Imported Research Memory` |
| Add anti-duplicate language: do not propose another rule-derived environment adversarial/V-REx/IRM/VIB bottleneck unless the environments, invariance principle, and falsifier are formally different. | The easiest near-duplicate would swap V-REx for IRM or add more nuisance buckets without a new idea. | `Research Continuity` or anti-duplicate constraints |
| Preserve a required random-environment control for all future causal-invariance ideas. | It is the cleanest test that semantic environments matter rather than generic regularization. | `Ablation Plan` requirements |
| Clarify that side-to-move canonicalization is allowed only when channel semantics are explicit and must not imply full board D4 invariance. | Avoids unsafe symmetry claims around pawns, files, castling, and side-to-move. | `Problem Restatement And Data Contract` |
| Require future LC0-based deterministic adapters to fail closed unless current-board and history channels are proven. | Prevents accidental parsing of BT4 history planes or unknown channels as safe current-board features. | `Problem Restatement And Data Contract` |
| Add a note that material/phase variables can be legitimate chess context, so suppressing them must be validated against no-invariance and material-only/color-only controls. | Prevents overclaiming nuisance removal as universally correct. | `Mathematical Thesis` and `Ablation Plan` guidance |

Do not weaken the leakage rules, label rules, falsification requirements, anti-duplicate requirements, or full-file training warning.

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0732_tuesday_pdt_rule_partition_bottleneck.md`
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
