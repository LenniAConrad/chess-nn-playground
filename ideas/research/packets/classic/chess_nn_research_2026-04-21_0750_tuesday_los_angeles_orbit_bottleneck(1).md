# Codex Handoff Packet: Rule-Exact Orbit Bottleneck Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0750_tuesday_los_angeles_orbit_bottleneck.md`
- Generated at: 2026-04-21 07:50:41 America/Los_Angeles
- Weekday: Tuesday
- Timezone: America/Los_Angeles
- Idea slug: `orbit_bottleneck`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: **Rule-Exact Orbit Bottleneck Network** (`REOBN`)
- One-sentence thesis: A chess position should remain equally puzzle-like under the exact color-flip automorphism of chess rules, so a classifier that averages predictions over this rule-exact orbit should suppress color-perspective/source artifacts while preserving tactical geometry.
- Idea fingerprint: `current-board tensor -> deterministic rule-exact color flip orbit {x, kappa(x)} -> shared neural encoder -> Reynolds probability pooling -> binary puzzle-like logits; central falsifier is replacing kappa with a rank flip that does not swap colors/side-to-move`.
- Why this is not a common CNN/ResNet/Transformer variant: The core operator is not extra depth, attention, or a new convolution block; it is a finite-group Reynolds projection over a chess-specific rule automorphism, with an explicit semantics-destroying orbit ablation that keeps compute and many tensor statistics matched.
- Current-data minimal experiment: Train on `data/splits/crtk_sample_3class/split_train.parquet`, validate/test on the existing split, use `simple_18`, binary target `fine_label == 0 -> 0`, `fine_label in {1,2} -> 1`, and compare `REOBN(color_flip)` to the same stem without orbit pooling plus `REOBN(rank_flip_no_color)` as the central falsifier.
- Smallest central falsification ablation: Replace the rule-exact color flip `kappa` by a plain vertical rank flip that preserves shape, material counts, side-to-move plane marginal, and compute, but does not swap piece colors, side-to-move, castling rights, or en-passant rank; if this performs the same, the gain is not evidence for chess-rule invariance.
- Expected information gain if it fails: Failure would show that current baselines already learn the color-perspective invariant signal, or that the dataset's discriminative signal is not limited by this symmetry; future packets should not recycle exact color-flip pooling unless they add a genuinely new environment-shift test or a different formal invariant.

## 3. Problem Restatement And Data Contract

The project is `chess-nn-playground`. The task is chess puzzle-likeness classification from a single current board position.

Binary output contract:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels in the current data:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

Default binary mapping for this experiment:

- `y = 0` for fine label `0`
- `y = 1` for fine labels `1` and `2`

The report must still include the rectangular diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Allowed input tensors:

```text
(batch, C, 8, 8)
```

Required output logits:

```text
(batch, 2)
```

Current encodings known to the project:

- `simple_18`: 12 piece planes + side-to-move + castling + en-passant
- `lc0_static_112`
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists

Benchmark split to use:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

Do **not** point the current trainer directly at the roughly 45M-row full Parquet dataset until streaming support exists.

Leakage checklist:

- Safe inputs: board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and deterministic rule-derived coordinate/channel transforms of the current board.
- Safe rule-derived features for this packet: the exact color-flip transform of the current tensor, using only current-board planes and documented channel semantics.
- Forbidden neural inputs: Stockfish evaluations, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, or any post-hoc candidate-pool status.
- Full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences are not used here. They remain leakage-prone unless a future packet explicitly proves they are rule-only, label-independent, engine-free, and ablated.
- For `lc0_static_112` and `lc0_bt4_112`, deterministic geometry transforms may only touch channels whose semantics are explicitly known. History or unknown channels may be consumed by a learned adapter in a single-view model, but they must not be transformed by guessed channel maps.

Boundary between safe deterministic geometry and leakage:

- Deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack geometry derived only from the current board are allowed in the broader project. This packet does not need attack geometry.
- Full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences are leakage-prone unless explicitly justified as rule-only, label-independent, engine-free, and ablated. This packet deliberately avoids them.
- Engine evaluation, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, and dataset provenance must never be neural-network inputs.
- For `lc0_static_112` and `lc0_bt4_112`, distinguish current-board channels used for deterministic geometry from history channels used only by learned neural adapters. If the channel map is missing, the orbit adapter must fail closed.

## 4. Research Map

External sources used:

| Source | URL | Borrowed | Not copied |
|---|---|---|---|
| Cohen and Welling, “Group Equivariant Convolutional Networks,” ICML 2016 | https://proceedings.mlr.press/v48/cohenc16.html | The general principle that known symmetries can reduce sample complexity by weight sharing/equivariant structure. | No group convolution layers, no full dihedral image group, no claim that chess is rotation/reflection invariant. |
| Sannai et al., “Invariant and Equivariant Reynolds Networks,” JMLR 2024 | https://www.jmlr.org/papers/v25/22-0891.html | The Reynolds operator viewpoint: average a model over a finite group to produce invariant predictions. | No large-group reductive designs, no generic permutation group machinery. The group here is a tiny chess-rule automorphism. |
| Arjovsky et al., “Invariant Risk Minimization,” 2019 | https://arxiv.org/abs/1907.02893 | The causal-learning intuition that stable predictors across environments are preferable to shortcut predictors. | No IRM penalty is required; this is exact orbit invariance, not learned environment invariance. |
| Chessprogramming Wiki, “Color Flipping” | https://www.chessprogramming.org/Color_Flipping | Practical definition of color flipping as vertical rank mirroring plus swapping piece colors, side-to-move, castling rights, and en-passant rank. | No engine evaluation, search, attack tables, or move generation. |
| FIDE Laws of Chess, castling rule reference | https://www.fide.com/FIDE/handbook/LawsOfChess.pdf | Used only to justify being conservative with file mirrors: castling is tied to specific king/rook movement rules, so arbitrary file reflection is not a universal exact symmetry when castling rights are active. | No legal-move oracle or castling-legality feature is used as model input. |
| Ganin et al., “Domain-Adversarial Training of Neural Networks,” JMLR 2016 | https://jmlr.org/papers/v17/15-239.html | Considered as a candidate route for suppressing source/domain artifacts. | Rejected for the minimal experiment because an adversarial domain head would add instability and a less direct falsifier. |

Candidate search trace:

| Candidate mechanism considered | Why it was serious | Why it lost to `REOBN` |
|---|---|---|
| Multi-encoding invariant bottleneck across `simple_18`, `lc0_static_112`, and `lc0_bt4_112` | It directly attacks encoding-family shortcut risk and is aligned with causal invariance. | It needs a multi-view loader or repeated FEN encoding path; `REOBN` is implementable immediately with one current encoding and still exposes a causal invariance falsifier. |
| Evidential near-puzzle uncertainty head | It could use fine label `1` safely as an ambiguity diagnostic without fabricating labels. | It mainly improves calibration/selective prediction, not the representation of puzzle structure; it is weaker as a first model-family falsifier. |
| Material-conditional null-contrastive energy model | It could learn whether a board is structurally coherent beyond material counts using generated material-matched null boards. | It is close to pseudo-likelihood/description-length families already imported, and generated null boards risk teaching legality or dataset style rather than puzzle-likeness. |
| HSIC or adversarial nuisance suppression for material/phase/castling | It is a genuine alternative to closed-form nuisance projection. | It is broad, sensitive to nuisance choices, and risks suppressing real tactical information such as side-to-move or castling state. |
| Masked generative compression of board motifs | It matches the suggested MDL direction and could expose local “surprise.” | It is too close to the imported geometry-conditioned pseudo-likelihood/description-length ratio packet unless rebuilt around a substantially new observable. |
| Persistent homology on occupancy or attack maps | It gives a non-sheaf topological descriptor of tactical clusters. | Any useful version quickly becomes an attack/defense cell-complex operator and risks duplicating the sheaf/Hodge family under different language. |
| Piece-set equivariant network with relative-coordinate kernels | It is more chess-native than a square ViT. | It overlaps with high-order constellation and vanilla set/attention ideas; the falsifier is less clean than rule-exact orbit pooling. |
| Legal-move-consistency auxiliary task | It might encourage rule understanding. | It uses legal move generation and move counts, which are explicitly leakage-prone unless heavily justified and ablated; not needed for the present thesis. |
| Plain file-mirror augmentation | File labels are often nuisance-like. | Horizontal file reflection is not universally rule-exact with active castling rights; a file-mirror sheaf family is already imported. |
| Rank/file spectral harmonic model | Could capture spatial motif frequencies without attacks or moves. | It is likely a handcrafted feature basis around a CNN, with no clean chess-specific theorem beyond generic signal processing. |
| Contrastive learning with material-matched hard negatives | It could force geometry sensitivity while preserving material. | It creates artificial negatives and may learn “real board versus shuffled board” rather than puzzle-likeness. |
| Label-safe abstention/selective classifier | It fits ambiguity between fine labels `1` and `2`. | Valuable for reporting, but not a distinct central board operator for classification accuracy. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Group invariance | Exact chess color-flip group `G = {e, kappa}` with `kappa` = rank mirror + piece-color swap + side/castling/en-passant swap | `x: [B,C,8,8] -> orbit: [B,2,C,8,8]` | Replace `kappa` with rank flip without color/side swaps | Imported packets use sheaves, move-deltas, OT, ray languages, ANOVA, or pseudo-likelihood; this uses no attack graph, move set, transport, or class likelihood. |
| Reynolds operator | Average per-orbit probability predictions | per-view logits `[B,2,2] -> probs [B,2,2] -> mean probs [B,2] -> log probs `[B,2]` | Single-view model with identical stem and parameter count | This is prediction-level group projection, not a larger CNN or ordinary random augmentation. |
| Causal invariance | Treat color perspective as a label-preserving environment | Same labels for `x` and `kappa(x)`; no new fine labels | Semantically wrong orbit with same compute | It is not IRM over unknown domains; the environment is a mathematically specified chess-rule automorphism. |
| Bottleneck | Classifier only receives the orbit-projected prediction or latent summary | optional latent `z_g: [B,2,D]`, pooled `z_bar: [B,D]` | Allow classifier to see unpooled view identity | It suppresses view-specific artifacts by construction rather than by closed-form nuisance projection. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN | `src/chess_nn_playground/models/cnn.py` | Already exists and tests generic local board features without the rule-exact orbit hypothesis. |
| Residual CNN | `src/chess_nn_playground/models/residual_cnn.py` | Already exists; adding residual depth is an ordinary architecture change, not a new research mechanism. |
| LC0-style CNN/residual CNN | Existing LC0 BT4-style CNN and residual variants | Already represented; copying LC0-style channel processing does not test a new invariance claim. |
| Bigger/deeper/wider CNN | Small/medium/deep baseline variants | Prohibited as a core idea and unlikely to reveal whether puzzle-likeness is color-perspective invariant. |
| Ordinary ViT over 64 squares | No exact existing baseline, but a common obvious import | A vanilla Transformer over squares is explicitly disallowed and has no chess-specific falsifier. |
| Plain GNN on 64 squares | Generic square graph neural net | Too ordinary; without a new board operator it is just message passing over the grid. |
| Hyperparameter tuning | Current trainer configs | Prohibited as a research idea; it changes optimization, not the hypothesis class. |
| Ensembling | Any average of existing baselines | Prohibited and hard to interpret; improvements would not isolate the mechanism. |
| Static attack-defense graph/sheaf/Hodge operator | Imported tactical sheaf/Hodge families | Already heavily represented; another attack incidence/tension/curvature variant would be a duplicate. |
| One-ply move-delta pooling/spectrum/landscape | Imported counterfactual move-delta packets | Already represented and explicitly disallowed unless the operator changes fundamentally. |
| Sinkhorn or current-board piece-target OT | Imported optimal-transport packets | Already represented and explicitly disallowed as a reparameterized transport bottleneck. |
| Ordinal ladder over fine labels | Imported `Ordinal Evidence Ladder Network` | Already represented; this packet keeps fine labels only for diagnostics. |
| Sparse witness-piece bottleneck | Imported sparse witness packet | Already represented and not needed for the orbit-invariance thesis. |
| Ray-language automata | Imported `Ray-Language Automaton Network` | Already represented; ray tokens do not test color-flip invariance. |
| Möbius/ANOVA constellations | Imported constellation packet | Already represented; high-order piece interactions are not the selected mechanism. |
| Class-conditioned pseudo-likelihood/MDL board ratio | Imported geometry-conditioned pseudo-likelihood packet | Too close to imported generative description-length ideas. |
| Plain random flip augmentation | Common image augmentation | It would not distinguish rule-exact symmetry from arbitrary tensor augmentation; the semantics-destroying ablation is required. |

## 6. Mathematical Thesis

### Input space definition

Let `X_simple` be the set of `simple_18` tensors that encode a single chess position:

```text
x in {0,1}^{18 x 8 x 8}
```

with 12 mutually exclusive piece planes, one side-to-move plane, castling-right planes, and an en-passant plane or equivalent en-passant encoding. The exact channel order must be registered before deterministic transforms are allowed.

Let `X` denote the subset of encodings with known channel semantics. For the minimal experiment, `X = X_simple`.

### Label/target definition

Let `a in {0,1,2}` be the fine label and define the binary target

```text
y = 1[a in {1,2}]
```

No new class `1` or class `2` labels are created. Fine labels remain available only for training targets already present and for diagnostics.

### Data distribution assumptions

Let `P` be the empirical train distribution over `(x,y,a)`. The following assumption is a modeling hypothesis, not a proven fact about the sampled dataset:

```text
P*(y | x) = P*(y | kappa x)
```

where `P*` is the intended label-generating distribution and `kappa` is the exact chess color-flip transform. The empirical sampled distribution `P` may violate this due to source imbalance, color-perspective artifacts, or finite-sample effects.

### Allowed symmetry or equivariance assumptions

Chess is not fully invariant under arbitrary rotations/reflections. Pawns, side-to-move, castling, and en-passant matter.

This packet uses only the two-element group

```text
G = {e, kappa}
```

where `kappa` is color flipping:

1. mirror ranks across the horizontal midline,
2. swap white and black piece planes,
3. swap side-to-move,
4. swap white castling rights with corresponding black castling rights on the same side,
5. mirror the en-passant target rank.

This is a rule-exact color-perspective transform. It is not a 90-degree rotation, not a raw image flip, and not a file mirror. File mirror is intentionally excluded from the minimal experiment because active castling rights make arbitrary file reflection unsafe unless additional conditions are checked.

### Core hypothesis

Puzzle-likeness is a property of the tactical relation between the side to move and the position, not of whether that relation is displayed from White's or Black's perspective. Therefore, a classifier constrained to be invariant under `kappa` should generalize better than a single-view model when the dataset contains color-perspective or source artifacts.

### Formal object introduced

For any neural predictor `f_theta: X -> Delta^1`, define the Reynolds-projected predictor

```text
R_G f_theta(x) = (1 / |G|) sum_{g in G} f_theta(g x)
               = 0.5 * (f_theta(x) + f_theta(kappa x)).
```

The model returns log probabilities compatible with the shared trainer:

```text
logits_returned(x) = log( clamp(R_G f_theta(x), eps, 1) )
```

Since `R_G f_theta(x)` sums to one, `torch.nn.CrossEntropyLoss` applied to these log probabilities is valid.

### Proposition

Assume:

1. `G` is a finite group acting on `X`,
2. the target is invariant: `y(gx) = y(x)` for all `g in G`,
3. the evaluation distribution is `G`-invariant, meaning `(x,y)` and `(gx,y)` have the same distribution for all `g`,
4. the loss is cross-entropy on predicted probabilities.

Then for any predictor `f`, the Reynolds-projected predictor `R_G f` has orbit-symmetrized risk no greater than the average risk of `f` over the orbit:

```text
E[-log (R_G f(X))_Y]
<=
E[(1/|G|) sum_{g in G} -log f(gX)_Y].
```

### Proof sketch or derivation

For a fixed labeled example `(x,y)`, cross-entropy is

```text
L(p,y) = -log p_y.
```

The function `p -> -log p_y` is convex on the probability simplex. By Jensen's inequality,

```text
L((1/|G|) sum_g f(gx), y)
<=
(1/|G|) sum_g L(f(gx), y).
```

Taking expectation over a `G`-invariant distribution gives the stated inequality. Exact invariance of `R_G f` follows from group closure:

```text
R_G f(hx) = (1/|G|) sum_g f(g h x) = (1/|G|) sum_{g' in G} f(g' x) = R_G f(x).
```

### What is actually proven

The theorem proves that probability-level orbit averaging cannot be worse than the average orbit risk under exact label invariance, a valid group action, and a group-invariant evaluation distribution.

It also proves exact prediction invariance under `kappa` when the orbit adapter is correct.

### What remains only hypothesized

- The sampled train/test distributions may not be fully `kappa`-invariant.
- Puzzle-likeness labels may contain collection artifacts correlated with color or side-to-move.
- Better orbit-symmetrized risk may or may not improve the leaderboard metric on the current split.
- The base encoder may already learn the invariance, leaving little headroom.
- Latent consistency regularizers may help, but only probability-level Reynolds pooling is covered by the proposition.

### Counterexamples where the idea should fail

- If the dataset has a real label asymmetry caused by sampling policy, such as verified puzzles overwhelmingly collected from one color perspective and non-puzzles from another, exact invariance can remove a predictive artifact and lower in-split accuracy.
- If the channel map is wrong, `kappa(x)` is not the intended position and the model should fail. This is why adapters must fail closed.
- If the task secretly depends on provenance rather than board semantics, orbit invariance will intentionally suppress that shortcut.
- If most baseline errors come from missing tactical calculation rather than color-perspective artifacts, this model may not improve accuracy.
- If en-passant or castling channels are transformed incorrectly, the mathematical guarantee does not apply.

### Self-critique

The strongest objection is that `kappa` is an obvious chess symmetry and current CNNs may already learn it from data. If so, `REOBN` doubles compute for no gain. That objection is fair. The experiment remains worth running because the falsifier is unusually clean: the rank-flip-without-color-swap ablation keeps most image-level augmentation statistics and compute but breaks chess semantics. A positive result over that ablation would be evidence that the exact rule automorphism matters, not just extra views or regularization. A null result would be informative enough to ban this family from future cycles.

## 7. Architecture Specification

### Module names

Proposed model file:

```text
src/chess_nn_playground/models/rule_exact_orbit_bottleneck.py
```

Main classes/modules:

- `Simple18ColorFlipAdapter`
- `Lc0OrbitAdapter`
- `OrbitAdapterRegistry`
- `TinyBoardStem`
- `RuleExactOrbitBottleneckNet`
- optional `OrbitAuxLoss`

Registry builder:

```text
build_rule_exact_orbit_bottleneck(config)
```

### Forward-pass steps

Default minimal configuration:

```yaml
encoding: simple_18
orbit_group: color_flip
orbit_size: 2
pool_mode: probability_mean
stem_width: 48
latent_dim: 128
num_blocks: 3
num_classes: 2
```

Forward contract:

1. Input:

   ```text
   x: [B, C, 8, 8]
   ```

2. Adapter validates `C` and channel schema. For `simple_18`, it constructs:

   ```text
   orbit_x = stack([x, kappa(x)], dim=1)
   orbit_x: [B, 2, C, 8, 8]
   ```

3. Flatten orbit dimension:

   ```text
   orbit_flat: [2B, C, 8, 8]
   ```

4. `TinyBoardStem`:

   - `Conv3x3(C -> 48) + Norm + GELU`
   - 3 residual micro-blocks at width 48
   - `Conv1x1(48 -> 96)`
   - global average pool

   Output:

   ```text
   h_flat: [2B, 96]
   ```

5. Projection MLP:

   ```text
   z_flat: [2B, 128]
   z: [B, 2, 128]
   ```

6. Classifier head per orbit view:

   ```text
   view_logits: [B, 2, 2]
   view_probs = softmax(view_logits, dim=-1): [B, 2, 2]
   mean_probs = mean(view_probs, dim=1): [B, 2]
   returned_logits = log(clamp(mean_probs, eps, 1.0)): [B, 2]
   ```

7. Return value:

   - Default: `returned_logits` only, so the shared trainer keeps working.
   - Optional custom trainer mode: return `(returned_logits, aux)` where `aux` includes `view_logits`, `z`, and `orbit_transform_names`.

### Pseudocode

```text
forward(x):
    adapter = registry.get(channel_schema, orbit_group)
    orbit = adapter.make_orbit(x)                    # [B,K,C,8,8]
    B,K,C,_,_ = orbit.shape
    flat = orbit.reshape(B*K, C, 8, 8)
    h = stem(flat)                                   # [B*K,H]
    z = projection(h).reshape(B,K,D)                 # [B,K,D]
    per_view_logits = classifier(z)                  # [B,K,2]
    per_view_probs = softmax(per_view_logits, -1)    # [B,K,2]
    mean_probs = per_view_probs.mean(dim=1)          # [B,2]
    return log(mean_probs.clamp_min(eps))            # [B,2]
```

### Parameter-count estimate

For `simple_18`, with `stem_width=48`, `latent_dim=128`, and 3 residual micro-blocks:

- first `3x3` convolution: about `18 * 48 * 9 = 7,776` weights
- six residual `3x3` convolutions at width 48: about `6 * 48 * 48 * 9 = 124,416` weights
- `1x1` projection `48 -> 96`: about `4,608` weights
- MLP and classifier: about `29,000` weights
- normalization/biases: small

Expected total: roughly `0.17M` to `0.20M` parameters for `simple_18`.

For `lc0_static_112` or `lc0_bt4_112`, the first convolution rises to about `112 * 48 * 9 = 48,384` weights, so the expected total is roughly `0.21M` to `0.25M` parameters if a safe adapter is available.

### FLOP or complexity estimate

Let:

- `B` = batch size,
- `K` = orbit size, default `2`,
- `C` = input channels,
- `W` = stem width, default `48`,
- `L` = number of residual blocks, default `3`.

The main convolutional cost is approximately:

```text
O(B * K * 8 * 8 * 9 * (C*W + 2L*W^2)).
```

With `B=512`, `K=2`, `C=18`, `W=48`, `L=3`, this is roughly twice the single-view tiny stem. If memory is tight, use `batch_size=256` before changing the model.

### Candidate-set memory and chunking plan

This model does not generate moves, attacks, graph edges, targets, or candidate sets.

Orbit memory:

```text
orbit tensor floats = B * K * C * 8 * 8
latent tensor floats = B * K * latent_dim
```

Examples with float32:

- `B=512`, `K=2`, `C=18`: orbit tensor about `512*2*18*64*4 = 4.7 MB`
- `B=512`, `K=2`, `C=112`: orbit tensor about `29.4 MB`
- latent tensor with `D=128`: about `0.5 MB`

Activations inside the stem dominate memory. If `K > 2` in future variants, compute the stem over orbit chunks and concatenate per-view logits. Do not materialize large optional orbits unless needed.

### Required config fields

```yaml
model:
  name: rule_exact_orbit_bottleneck
  input_channels: 18
  num_classes: 2
  channel_schema: simple_18
  orbit_group: color_flip
  pool_mode: probability_mean
  stem_width: 48
  latent_dim: 128
  num_blocks: 3
  orbit_aux:
    use_jsd: true
    jsd_weight: 0.05
    use_latent_variance: false
    latent_variance_weight: 0.001
  fail_closed_unknown_channels: true
```

### Encoding support

`simple_18`:

- First experiment should use `simple_18`.
- Required semantics:
  - 12 piece planes in a documented order.
  - side-to-move plane with documented binary convention.
  - castling-right planes with documented `K/Q/k/q` order.
  - en-passant plane with documented geometry.
- If any semantic is missing, the color-flip adapter must raise an explicit error.

`lc0_static_112`:

- Do not guess channel semantics.
- If the repo has an explicit LC0 channel map, implement a separate adapter that transforms only known current-board planes and known scalar planes.
- If the map is missing, fail closed and run only the single-view baseline for this encoding.

`lc0_bt4_112`:

- Same as `lc0_static_112`.
- Because current history planes are zero-filled until exporter support exists, a known map may permit flipping the current-board slice and leaving all-zero history slices invariant. This must be asserted by tests.
- When real history exists in the future, transform history planes only if their time/channel semantics are documented.

### How the model returns logits

The default forward pass returns:

```text
[batch, num_classes]
```

where entries are log probabilities from orbit-averaged probabilities. This is compatible with `CrossEntropyLoss` and the shared trainer. A custom trainer can optionally request auxiliary outputs, but the benchmark path must keep the default logits-only behavior.

## 8. Loss, Training, And Regularization

Primary loss:

```text
L_main = CrossEntropyLoss(returned_logits, y_binary)
```

Because `returned_logits` are log probabilities, `CrossEntropyLoss` remains valid.

Optional auxiliary loss for custom `ideas/.../train.py` only:

```text
p_bar = mean_g softmax(logits_g)
L_jsd = mean_g KL(softmax(logits_g) || stopgrad_or_not(p_bar))
L_lat = mean_g || normalize(z_g) - mean_h normalize(z_h) ||_2^2
L = L_main + lambda_jsd * L_jsd + lambda_lat * L_lat
```

Recommended defaults:

- `lambda_jsd = 0.05`
- `lambda_lat = 0.0` for the first fair comparison
- turn on `lambda_lat = 0.001` only after the CE-only orbit model is benchmarked

Class weighting:

- Use the existing `balanced` class weighting used by benchmark configs.
- Do not introduce fine-label-specific weights unless separately ablated.

Batch size expectations:

- Start with `batch_size=512` for `simple_18`.
- If GPU memory is tight, reduce to `256`; do not silently reduce orbit size.

Optimizer and learning-rate defaults:

- `AdamW`
- learning rate `0.001`
- weight decay `0.0001`
- epochs `3`
- early stopping patience `2`

Regularizers:

- Existing weight decay.
- Optional orbit JSD as above.
- No dropout required in the first experiment; if used, keep the same dropout in the no-orbit ablation.

Determinism requirements:

- Use seed `42`.
- Set deterministic mode true where the repo supports it.
- The orbit transform is deterministic and must be tested exactly.
- Save the list of transform names used in each ablation config.

What must stay unchanged for fair comparison:

- Same train/val/test split.
- Same binary mapping.
- Same metrics and reports.
- Same batch size if feasible.
- Same number of epochs and early stopping.
- Same class weighting.
- Same preprocessing/cache behavior.
- Same decision threshold policy used by existing reports unless a threshold-matched diagnostic is explicitly requested.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| `no_orbit_single_view` | Uses the same `TinyBoardStem`, projection, and classifier on `x` only | Tests whether exact orbit projection adds value beyond the stem | If equal, the baseline already learned color-flip invariance or the symmetry is irrelevant to this split. |
| `rank_flip_no_color_orbit` | Replaces `kappa` with a vertical rank flip that does not swap colors, side-to-move, castling, or en-passant | Central falsifier: chess-rule semantics should matter, not just extra flipped views | If it matches `color_flip`, the main gain is generic augmentation/compute or regularization, not rule-exact invariance. |
| `train_aug_only_color_flip` | Randomly applies `kappa` during training, but uses single-view inference | Tests Reynolds pooling against ordinary label-preserving augmentation | If equal, test-time orbit averaging is unnecessary; future work can use simpler augmentation. |
| `latent_mean_pool` | Pools latents before the classifier instead of averaging probabilities | Tests whether the theorem-aligned probability Reynolds operator matters | If latent pooling is equal or better, the proof was not the practical driver, but invariance may still help. |
| `logit_mean_pool` | Averages per-view logits before softmax | Tests sensitivity to the pooling space | If logit pooling wins, calibrate carefully; the Jensen proof no longer directly applies. |
| `color_swap_no_rank_flip` | Swaps white/black channels and side-to-move without mirroring ranks | Destroys pawn geometry while preserving many channel marginals | If this works, the model may be exploiting channel symmetrization rather than chess geometry. |
| `orbit_jsd_off` | Removes optional JSD consistency, leaving only probability averaging | Tests whether gains come from architectural projection alone | If JSD is required, report it as regularization-dependent rather than pure Reynolds pooling. |
| `view_identity_leak` | Concatenates a learned view-id embedding before the classifier | Tests whether allowing view-specific shortcuts hurts invariance | If this improves in-split but worsens orbit consistency or near-puzzle diagnostics, the original bottleneck is doing its intended anti-shortcut job. |
| `conditional_file_mirror_no_castling_only` | Optional later: add file mirror only for positions with no castling rights | Tests whether broader safe symmetry helps after the core color-flip result | If it helps, propose a separate future packet; do not merge into the first claim. |
| `lc0_single_view_adapter` | Runs same stem on `lc0_bt4_112` without deterministic orbit if channel semantics are unknown | Tests encoding baseline without unsafe transforms | If LC0 single-view dominates, the next step is documenting a safe LC0 orbit adapter, not guessing one. |

This model has no move-set or generated candidate-set. Therefore count-only move ablations, capture histograms, source-square marginals, and candidate-degree controls are not applicable. The semantics-destroying rank flip is the required structure-destroying control for the group-orbit operator.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- Existing `simple_18` simple CNN.
- Existing `simple_18` residual CNN, matched to the closest parameter scale available.
- Existing small/medium/deep variants already on the leaderboard.
- `REOBN(no_orbit_single_view)` with the exact same stem.
- `REOBN(rank_flip_no_color_orbit)` central falsification ablation.
- `REOBN(train_aug_only_color_flip)` if the training loop can add deterministic augmentation easily.

Metrics to inspect:

- Test accuracy.
- ROC-AUC if existing reports include it.
- PR-AUC if existing reports include it.
- F1 and balanced accuracy.
- Calibration error if already available; otherwise do not add it as a blocking artifact.
- Required rectangular diagnostic matrix: fine label `0/1/2 -> predicted 0/1`.
- Near-puzzle diagnostic:
  - class `1` recall at the default threshold, and
  - class `1` recall at a matched fine-label-`0` false-positive rate, using the validation set to choose the threshold.
- Orbit consistency:
  - `mean |p(y=1|x) - p(y=1|kappa x)|` on validation/test,
  - compare to no-orbit baseline.

Required artifacts:

- Main model config.
- Central ablation configs.
- Checkpoint paths.
- Metrics JSON/CSV.
- Standard confusion matrices.
- Rectangular `3x2` fine-label diagnostic for main model and every central ablation.
- Predictions parquet/CSV with columns sufficient for threshold-matched diagnostics.
- Orbit consistency report.
- Short report explaining whether `rank_flip_no_color_orbit` matched or failed.

Success threshold:

- Primary success: `REOBN(color_flip)` improves test PR-AUC or balanced accuracy by at least `+1.0` percentage point over the strongest same-encoding baseline **and** beats `rank_flip_no_color_orbit` by at least `+0.5` percentage point on the same metric.
- Diagnostic success: class `1` recall at matched fine-label-`0` false-positive rate improves by at least `+2.0` percentage points over `no_orbit_single_view`.
- Consistency success: orbit disagreement drops by at least `80%` relative to `no_orbit_single_view`.

Failure threshold:

- `REOBN(color_flip)` is within `±0.3` percentage points of both `no_orbit_single_view` and `rank_flip_no_color_orbit` on primary metrics.
- Fine-label `1` recall does not improve at matched fine-label-`0` false-positive rate.
- The rank-flip falsifier performs equal or better, suggesting the claimed chess-rule semantics are not responsible.

Abandon the idea if:

- The semantic-destroying orbit matches or beats the rule-exact orbit on test metrics and near-puzzle diagnostics.
- The only gain is in orbit consistency while classification metrics and class `1` diagnostics do not improve.
- Correct channel-map tests are difficult or brittle enough to risk silent wrong transforms.

Justify scaling if:

- The rule-exact orbit beats both the no-orbit and semantic-destroying ablations.
- The gain appears in fine label `1` diagnostics, not just aggregate accuracy.
- The model remains stable across at least two seeds or a repeated run.
- A safe LC0 adapter can be documented with tests; otherwise scale only within `simple_18`.

## 11. Implementation Plan For Codex

Use an idea id such as:

```text
20260421_rule_exact_orbit_bottleneck
```

Repo changes:

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_rule_exact_orbit_bottleneck/idea.yaml` | Create | Machine-readable idea metadata from the block below. |
| `ideas/20260421_rule_exact_orbit_bottleneck/math_thesis.md` | Create | Section 6, including the Reynolds proposition, proof sketch, and counterexamples. |
| `ideas/20260421_rule_exact_orbit_bottleneck/architecture.md` | Create | Section 7 with tensor contracts, adapter semantics, parameter estimates, and pseudocode. |
| `ideas/20260421_rule_exact_orbit_bottleneck/implementation_notes.md` | Create | Channel-map validation, fail-closed adapter behavior, transform unit tests, and no-leakage notes. |
| `ideas/20260421_rule_exact_orbit_bottleneck/trainer_notes.md` | Create | Loss setup, shared-trainer compatibility, optional aux-loss path, deterministic settings. |
| `ideas/20260421_rule_exact_orbit_bottleneck/ablations.md` | Create | Section 9 table and exact configs to run. |
| `ideas/20260421_rule_exact_orbit_bottleneck/train.py` | Create | Thin launcher around the existing training script/config; optional aux-loss support only if simple to integrate. Must not bypass standard reports. |
| `ideas/20260421_rule_exact_orbit_bottleneck/config.yaml` | Create | Minimal `simple_18` config from `config_yaml` block. |
| `ideas/20260421_rule_exact_orbit_bottleneck/report_template.md` | Create | Template requiring main metrics, orbit consistency, and `3x2` fine-label matrices for main and ablations. |
| `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this packet to imported memory after implementation; include anti-duplicate notes below. |
| `src/chess_nn_playground/models/rule_exact_orbit_bottleneck.py` | Create | `Simple18ColorFlipAdapter`, `TinyBoardStem`, `RuleExactOrbitBottleneckNet`, and builder. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `rule_exact_orbit_bottleneck`. |
| `configs/rule_exact_orbit_bottleneck_simple18.yaml` | Create | Standard benchmark config pointing at current split, `encoding: simple_18`, `model.name: rule_exact_orbit_bottleneck`. |
| `configs/rule_exact_orbit_bottleneck_no_orbit_simple18.yaml` | Create | Same stem with `orbit_group: identity`. |
| `configs/rule_exact_orbit_bottleneck_rankflip_ablation_simple18.yaml` | Create | Same stem with `orbit_group: rank_flip_no_color`. |
| `configs/rule_exact_orbit_bottleneck_trainaug_simple18.yaml` | Create if easy | Random train-time `kappa` augmentation, no test-time orbit pooling. |
| `tests/test_rule_exact_orbit_bottleneck.py` | Create | Unit tests for shape, exact invariance of returned predictions under `kappa`, fail-closed unknown schema, and central ablation shape. |
| `tests/test_simple18_color_flip_adapter.py` | Create | Unit tests for piece-plane swap, side-to-move inversion, castling swap, en-passant rank mirror, and applying `kappa` twice returns the original tensor. |

Minimum test requirements:

- `kappa(kappa(x)) == x` for synthetic valid `simple_18` tensors.
- Output shape is `[B, 2]`.
- `model(x)` and `model(kappa(x))` match up to numerical tolerance when `pool_mode=probability_mean`.
- Unknown channel schema raises an explicit error if `fail_closed_unknown_channels=true`.
- `rank_flip_no_color` ablation has the same output shape and orbit size, but is clearly marked unsafe/non-semantic in config metadata.

For `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md`, Codex must update the prompt after consuming this output. Preserve all hard leakage, label, falsification, and anti-duplicate constraints. Add the new anti-duplicate fingerprint from Section 12 and the prompt-maintenance notes in Section 13.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0750_tuesday_los_angeles_orbit_bottleneck.md
  generated_at: "2026-04-21T07:50:41-07:00"
  weekday: tuesday
  timezone: los_angeles
  idea_slug: orbit_bottleneck
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_rule_exact_orbit_bottleneck
  name: Rule-Exact Orbit Bottleneck Network
  slug: orbit_bottleneck
  status: draft
  created_at: "2026-04-21T07:50:41-07:00"
  author: ChatGPT Pro
  short_thesis: Enforce binary puzzle predictions to be invariant under the exact chess color-flip automorphism by Reynolds-averaging per-orbit probabilities.
  novelty_claim: Uses a rule-exact color-perspective orbit and a semantics-destroying rank-flip falsifier; no attack graph, move-delta set, Sinkhorn transport, nuisance projection, ordinal ladder, sparse witness, ray automaton, constellation, or pseudo-likelihood ratio.
  expected_advantage: Reduces color-perspective and source artifacts while preserving current-board tactical geometry; should improve near-puzzle recall at matched non-puzzle false-positive rate if such artifacts hurt baselines.
  central_falsification_ablation: rank_flip_no_color_orbit
  target_task: coarse_binary
  input_representation: simple_18 primary; lc0_static_112 and lc0_bt4_112 only with explicit fail-closed channel maps
  output_heads: binary log-probability logits with optional orbit auxiliary diagnostics
  compute_notes: Orbit size 2 doubles stem compute; no move/candidate/graph memory.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/rule_exact_orbit_bottleneck_simple18.yaml
  model_path: src/chess_nn_playground/models/rule_exact_orbit_bottleneck.py
  latest_result_path: null
  notes: First run CE-only probability Reynolds pooling, then optional orbit JSD. Do not guess LC0 channel maps.
```

```yaml
config_yaml:
  run:
    name: rule_exact_orbit_bottleneck_simple18
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
    name: rule_exact_orbit_bottleneck
    input_channels: 18
    num_classes: 2
    channel_schema: simple_18
    orbit_group: color_flip
    pool_mode: probability_mean
    stem_width: 48
    latent_dim: 128
    num_blocks: 3
    fail_closed_unknown_channels: true
    orbit_aux:
      use_jsd: false
      jsd_weight: 0.05
      use_latent_variance: false
      latent_variance_weight: 0.001
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
  model_name: rule_exact_orbit_bottleneck
  file_path: src/chess_nn_playground/models/rule_exact_orbit_bottleneck.py
  builder_function: build_rule_exact_orbit_bottleneck
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18ColorFlipAdapter
    - OrbitAdapterRegistry
    - TinyBoardStem
    - RuleExactOrbitBottleneckNet
  required_config_fields:
    - input_channels
    - num_classes
    - channel_schema
    - orbit_group
    - pool_mode
    - stem_width
    - latent_dim
    - num_blocks
    - fail_closed_unknown_channels
  expected_parameter_count: approximately 0.17M to 0.20M for simple_18 with width 48 and latent_dim 128
  expected_memory_notes: Orbit tensor uses B*K*C*8*8 floats; K=2. At B=512,C=18,float32 this is about 4.7 MB before activations. Stem activations dominate; reduce batch size to 256 if needed.
```

```yaml
research_continuity:
  idea_fingerprint: current-board tensor plus exact color-flip orbit {x,kappa(x)} plus shared encoder plus Reynolds probability pooling plus rank-flip-no-color semantic falsifier
  already_researched_family_overlap: Low overlap with imported sheaf, move-delta, OT, nuisance-projection, ordinal, sparse-witness, ray-language, constellation, and pseudo-likelihood packets. Closest overlap is generic invariance/augmentation and the imported file-mirror sheaf name, but this packet uses no sheaf and does not rely on file mirror.
  closest_duplicate_risk: Plain color-flip data augmentation or test-time augmentation. The required distinction is probability-level Reynolds pooling plus a compute-matched non-automorphic flip ablation.
  do_not_repeat_if_this_fails:
    - Do not propose another exact color-flip orbit averaging model as the main idea unless the new packet introduces a genuinely different environment test.
    - Do not propose plain file-mirror augmentation as a novelty claim; castling makes it non-universal and a file-mirror sheaf family already exists.
    - Do not repackage this as group convolution, orbit pooling, canonicalization, or symmetry regularization without a new falsifiable chess operator.
    - Do not add adversarial nuisance heads to this same orbit model and call it a new research family unless the ablation isolates the adversarial mechanism.
  suggested_next_search_directions:
    - Label-safe selective prediction for fine label 1 ambiguity with conformal or evidential diagnostics, not ordinal ladders.
    - Multi-encoding invariant risk across simple_18 and LC0 encodings once a multi-view loader is available.
    - Non-pseudo-likelihood masked motif compression with strong material/source controls.
    - Causal evaluation under source or phase shifts if future split metadata supports it without becoming input.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Rule-Exact Orbit Bottleneck Network` to imported research memory with the fingerprint in Section 12. | Prevents future packets from recycling exact color-flip Reynolds pooling as novel. | `Imported Research Memory` |
| Add an anti-duplicate rule: “Do not propose another current-board exact color-flip orbit pooling/test-time augmentation/canonicalization model unless the central operator or environment test is genuinely different.” | Distinguishes future symmetry work from this packet. | Anti-duplicate paragraphs after imported packet fingerprints |
| Add a castling caution: “File mirror is not a universal safe chess symmetry when castling rights are active; any file-mirror transform must state its castling conditions and ablations.” | Avoids unsafe geometric transformations and clarifies why this packet used color flip only. | Leakage/safe-rule-derived feature guidance |
| Require any future symmetry packet to include a semantics-destroying compute-matched transform ablation. | Keeps symmetry ideas falsifiable rather than generic augmentation claims. | Falsification requirements |
| Add orbit consistency metrics as optional reporting for symmetry-based models. | Makes it clear whether the implementation actually enforces the claimed invariance. | Benchmark/reporting requirements |
| Preserve the LC0 fail-closed channel-map rule. | Prevents guessed transformations over unknown LC0 history/static channels. | Encoding adapter requirements |

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
