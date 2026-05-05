# Codex Research Batch: Additional Architecture Candidates 5

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md`
- Generated at: 2026-04-24 21:24
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: draft architecture batch, not implemented

## Purpose

This batch adds six more architecture candidates. The theme is "different computation shapes": compressed high-order interactions, reversible board encoders, morphological operators, optimal-transport role assignment, sparse expert routing, and local-neighborhood embedding geometry.

The goal is not to guarantee novelty in the global research-literature sense. The goal is to create distinct, implementable candidate families for this chess puzzle-likeness setting that do not merely rename earlier packets.

## Shared Data Contract

All candidates target the current binary puzzle-likeness task:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels `0`, `1`, and `2` remain diagnostics only. First implementations should use `simple_18`, the existing `crtk_sample_3class` splits, and the shared trainer.

Forbidden model inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Engine search, forced-line search, legal mate/stalemate oracles, or future game outcomes.

Allowed model inputs:

- Current board occupancy, side-to-move, castling/en-passant planes, deterministic square coordinates, side-relative coordinates, material/count summaries, and deterministic transforms of current board tensors.

## Ranked Shortlist

| Rank | Candidate | Main object | Why expand it |
|---|---|---|---|
| 1 | Tensor-Ring Square Interaction Network | Compressed high-order square and piece interactions | Strong way to test global interaction structure without full attention. |
| 2 | Sinkhorn Role Assignment Network | Soft assignment from pieces to latent tactical roles | Gives a learned role decomposition that is inspectable and naturally sparse. |
| 3 | Morphological Threat Field Network | Differentiable dilation, erosion, opening, and closing over board fields | Tests shape-based tactical patterns with very different inductive bias from CNNs. |
| 4 | Invertible Board Coupling Network | Reversible board feature transforms | Forces information preservation while measuring class-separating latent distortions. |
| 5 | Sparse Expert Board Router | Material/geometry-gated expert encoders | Practical scaling path for heterogeneous chess positions. |
| 6 | Local Neighborhood Geometry Network | Embedding curvature across deterministic board perturbations | Measures whether puzzle-like positions live in sharper local representation basins. |

Best next full packet from this batch: `Tensor-Ring Square Interaction Network`, with `Sinkhorn Role Assignment Network` close behind.

## Candidate 1: Tensor-Ring Square Interaction Network

### Thesis

Many chess cues depend on interactions among several squares at once: king square, attacking piece, blocker, defender, escape square, and promotion path. A full square-pair or square-tuple interaction tensor is too large. A tensor-ring factorization can model high-order interactions with a controlled parameter budget.

### Fingerprint

```text
simple_18 board tensor
+ square tokens
+ tensor-ring interaction cores
+ low-rank cyclic contraction summaries
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Not attention: there is no query-key softmax over all square pairs.
- Not a CNN: long-range square interactions are explicit.
- Not the previous TensorSketch packet: this uses learned low-rank tensor cores instead of randomized sketching.
- Not the determinantal packet: it does not rely on PSD Gram volumes or logdet summaries.

### Core Object

Let each square have an embedding:

```text
x_s in R^D, for s in 0..63
```

Learn a small set of tensor-ring cores:

```text
G_1, G_2, ..., G_K
```

Each core maps square/piece features into a low-rank interaction state. A cyclic contraction produces summary channels:

```text
z_k = trace(G_1(x_a) G_2(x_b) ... G_K(x_t))
```

The implementation should not enumerate all square K-tuples. It should use structured reductions:

- contract over all occupied squares
- contract over role-filtered groups
- contract over king-centered square subsets
- contract over rays and local neighborhoods

### Architecture Sketch

1. Input `(B, 18, 8, 8)`.
2. Flatten to 64 square tokens.
3. Add square coordinate embeddings and side-relative coordinate embeddings.
4. Project each token to width `D=48`.
5. Create role gates:

```text
own_piece_gate
opp_piece_gate
king_zone_gate
ray_relevant_gate
empty_square_gate
```

6. For each interaction order `K in {2, 3, 4}`, compute low-rank tensor-ring contractions.
7. Pool contraction statistics:

```text
mean, max, topk_mean, variance, signed_abs_mean
```

8. Concatenate with a small CNN stem summary.
9. Classify with MLP head.

### Tensor Contract

```text
input:             (B, 18, 8, 8)
tokens:            (B, 64, D)
role_gates:        (B, 64, R)
core_outputs:      (B, 64, K, rank, rank)
contract_stats:    (B, S)
cnn_summary:       (B, C)
logits:            (B, 2)
```

### Efficient Contraction Plan

Use dynamic summaries instead of tuple enumeration:

```text
M_i = sum_s gate_i(s) * G_i(x_s)
z = trace(M_1 M_2 ... M_K)
```

For multiple role patterns:

```text
own_attacker -> blocker -> king_zone
opp_attacker -> defender -> own_king_zone
empty_escape -> opp_control_proxy -> own_king
```

The role patterns are learned gates, not hard legal-move labels.

### Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `order2_only` | Remove order 3 and 4 contractions | Tests whether high-order interaction matters | If equal, pairwise structure is enough. |
| `no_role_gates` | Use all squares for every core | Tests role decomposition | If equal, gates are not useful. |
| `cnn_only` | Remove tensor-ring branch | Tests branch value | If equal, global interaction branch is wasted. |
| `random_square_permutation` | Fixed random permutation of square coordinates | Tests board geometry usage | Should degrade if interaction is board-aware. |
| `rank_1_cores` | Collapse tensor cores to scalar gates | Tests low-rank matrix state | If equal, tensor ring is overbuilt. |

### Diagnostics

- Top role patterns by absolute contribution.
- Per-position contraction magnitude distribution.
- Sensitivity to deleting one occupied square from the input tensor.
- Correlation between high-order contraction norms and false positives.
- Rank saturation of core matrices during training.

### Failure Modes

- Tensor contractions may become unstable if core norms grow.
- The model may learn material-count shortcuts through role gates.
- If role patterns are too hand-designed, the architecture becomes a feature-engineering wrapper.

### Implementation Notes

Start small:

```text
D = 48
rank = 4
orders = [2, 3]
role_patterns = 8
```

Normalize every core matrix with a spectral or Frobenius norm clamp. Keep the CNN summary branch modest so the tensor branch cannot be ignored by default.

## Candidate 2: Sinkhorn Role Assignment Network

### Thesis

Puzzle-like positions often contain latent tactical roles: target king, forcing piece, blocker, loose defender, escape square, overloaded piece, promotion candidate. Instead of asking attention to discover these roles implicitly, assign board objects to a fixed number of learned role slots using a differentiable optimal-transport layer.

### Fingerprint

```text
occupied piece tokens
+ learned role prototypes
+ Sinkhorn assignment matrix
+ role-slot interaction head
+ binary puzzle-likeness logits
```

### Why It Is Distinct

- More structured than slot attention: assignment is explicitly normalized with transport constraints.
- Different from imported entropic piece-target transport packets: there is no target measure, chess-distance cost, pressure map, or transport-statistic bottleneck. Transport is only the normalization mechanism for learned latent role slots.
- Different from Hall-defect packets: roles are learned and soft, not exact set-system deficits.
- Different from piece-token CNN hybrid: the token branch must form named latent slots before classification.

### Architecture Sketch

1. Extract occupied piece tokens from `simple_18`.
2. Add piece type, color, square, side-relative square, and local occupancy context.
3. Learn `M=10` role prototypes.
4. Compute cost matrix:

```text
cost[piece_index, role_index]
```

5. Run `T=8` Sinkhorn iterations to produce assignment matrix:

```text
A in R^(num_pieces x M)
```

6. Build role vectors:

```text
role_j = sum_i A[i, j] * token_i
```

7. Compute role-slot interactions with a small pairwise MLP.
8. Fuse with a light board CNN summary.
9. Classify.

### Tensor Contract

```text
board:             (B, 18, 8, 8)
piece_tokens:      (B, Pmax, D)
piece_mask:        (B, Pmax)
role_prototypes:   (M, D)
cost:              (B, Pmax, M)
assignment:        (B, Pmax, M)
role_vectors:      (B, M, D)
pair_features:     (B, M, M, H)
logits:            (B, 2)
```

### Transport Constraints

Piece mass:

```text
sum_j A[i, j] <= 1
```

Role mass:

```text
sum_i A[i, j] approximately target_role_mass[j]
```

Use a dustbin role for pieces that are irrelevant to the main tactical story.

### Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `mean_pool_tokens` | Remove role assignment | Tests whether transport roles help | Role model should improve if latent roles matter. |
| `softmax_no_sinkhorn` | Replace transport with independent role softmax | Tests transport normalization | Sinkhorn should reduce role collapse. |
| `no_dustbin` | Force every piece into tactical roles | Tests irrelevant-piece handling | No dustbin may overfit clutter. |
| `fixed_role_targets` | Use equal role masses | Tests learned mass priors | Learned priors should help varied material. |
| `assignment_entropy_high` | Strong entropy regularizer | Tests need for sharp role choices | Too high should blur tactical roles. |

### Diagnostics

- Assignment heatmaps per position.
- Most common piece type per role.
- Dustbin mass versus board clutter.
- Assignment entropy split by correct/incorrect examples.
- Role-pair interaction contribution scores.

### Failure Modes

- Roles may collapse to material categories instead of tactical jobs.
- Sinkhorn temperature can make training brittle if annealed too fast.
- The model may become hard to compare unless the CNN branch is kept small.

### Implementation Notes

Start with a fixed `Pmax=32` token padding scheme. Use masked Sinkhorn. Keep role count low enough for visualization:

```text
M = 8 or 10
D = 64
sinkhorn_iters = 6 to 10
temperature = 0.3 to 1.0
```

## Candidate 3: Morphological Threat Field Network

### Thesis

CNNs learn filters, but chess tactics often have shape operations: expand a king danger zone, close gaps in a pawn shield, erode escape squares, and detect thin corridors. Differentiable mathematical morphology gives an architecture that explicitly processes board fields through dilation, erosion, opening, and closing.

### Fingerprint

```text
learned board fields
+ soft dilation/erosion banks
+ morphology signature curves
+ compact classifier
```

### Why It Is Distinct

- Different from topology packets: it does not compute Betti or Euler curves.
- Different from tropical circuits: morphology is spatial shape processing, not clause satisfaction.
- Different from CNNs: max/min-like operators are primary, not linear convolution followed by activation.

### Architecture Sketch

1. Project board tensor into `F=16` scalar fields.
2. For each field, apply soft dilation and erosion with small learned structuring elements:

```text
soft_dilate(x, K) = tau * logsumexp((x + K) / tau)
soft_erode(x, K) = -soft_dilate(-x, K)
```

3. Build derived fields:

```text
opening = dilate(erode(x))
closing = erode(dilate(x))
morph_gradient = dilate(x) - erode(x)
top_hat = x - opening
black_hat = closing - x
```

4. Pool each derived field by global and king-centered statistics.
5. Feed compact morphology signature into a classifier.

### Tensor Contract

```text
input:             (B, 18, 8, 8)
fields:            (B, F, 8, 8)
derived_fields:    (B, F, O, 8, 8)
signature:         (B, S)
logits:            (B, 2)
```

### Structuring Elements

Use several shapes:

```text
3x3 square
rank line
file line
diagonal slash
diagonal backslash
knight-offset sparse stencil
king-ring stencil
```

The stencils should be learned weights over fixed support masks.

### Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `conv_replace` | Replace morphology with regular conv layers | Tests operator value | Morphology should help if shape extrema matter. |
| `dilate_only` | Remove erosion/opening/closing | Tests need for dual shape operations | If equal, extrema expansion alone is enough. |
| `no_king_center_pool` | Remove king-centered statistics | Tests chess localization | Should degrade if king geometry matters. |
| `fixed_stencils_only` | Freeze structuring element weights | Tests learned shape value | Learned stencils should improve. |
| `high_temperature` | Make logsumexp nearly average-like | Tests max/min behavior | Should degrade if morphology matters. |

### Diagnostics

- Derived field visualizations.
- Which structuring element contributes most per example.
- Morphological gradient mass around kings.
- Opening and closing residual magnitude by label.
- Temperature sensitivity curve.

### Failure Modes

- Soft max/min operations can saturate.
- Morphology branch may ignore piece identity unless the input projection is well controlled.
- It may overfit to board-frame artifacts if coordinate planes dominate.

### Implementation Notes

Use a very small CNN before morphology only to form fields. Keep the morphology signature interpretable and low dimensional. Start with:

```text
F = 12
stencils = 7
ops = [dilate, erode, open, close, gradient, top_hat]
tau = 0.5
```

## Candidate 4: Invertible Board Coupling Network

### Thesis

Standard encoders can discard information early, which makes it hard to know whether a model learned legitimate current-board structure or fragile shortcuts. A reversible board encoder preserves information by construction and classifies from latent distortions created by invertible coupling blocks.

### Fingerprint

```text
board tensor
+ invertible coupling blocks
+ latent summary and log-scale diagnostics
+ binary classifier
```

### Why It Is Distinct

- Different from residual networks: every block is explicitly invertible.
- Different from score-field denoising: there is no class-conditional repair vector field.
- Different from normal CNN baselines: information preservation is a design constraint.

### Architecture Sketch

1. Project `simple_18` into width `D=32`.
2. Split channels into two groups:

```text
x_a, x_b
```

3. Apply affine coupling:

```text
y_a = x_a
y_b = x_b * exp(s(x_a)) + t(x_a)
```

4. Alternate channel splits and spatial checkerboard splits.
5. Add invertible 1x1 channel mixing.
6. Pool final latent tensor plus coupling diagnostics:

```text
mean_abs_s
max_abs_s
latent_energy
inverse_reconstruction_error
```

7. Classify.

### Tensor Contract

```text
input:          (B, 18, 8, 8)
projected:      (B, D, 8, 8)
latent:         (B, D, 8, 8)
scale_stats:    (B, L)
summary:        (B, H)
logits:         (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `noninvertible_same_params` | Replace coupling with standard conv blocks | Tests invertibility constraint | If equal, reversibility is unnecessary. |
| `no_scale_stats` | Remove log-scale diagnostics | Tests whether distortions carry signal | If equal, classifier uses latent only. |
| `additive_only` | Use `s=0` additive couplings | Tests multiplicative deformation | Affine should help if scale matters. |
| `few_blocks` | Reduce reversible depth | Tests need for iterative transformations | Too shallow should underfit. |
| `frozen_inverse_check` | Track inverse error without training on it | Tests numerical stability | Error should stay near machine tolerance. |

### Diagnostics

- Inverse reconstruction error.
- Distribution of coupling scale values by label.
- Latent energy maps.
- Whether false positives have extreme scale statistics.
- Sensitivity to removing coordinate planes.

### Failure Modes

- Affine scales can explode without clipping.
- Reversible models may spend capacity preserving irrelevant details.
- The classifier can ignore the invertibility diagnostics if the latent pool is too strong.

### Implementation Notes

Clamp scale outputs:

```text
s = 0.8 * tanh(raw_s)
```

Use ActNorm or simple learned channel shift/scale only if initialized carefully. Do not train a generative likelihood in the first version; keep it supervised and diagnostic.

## Candidate 5: Sparse Expert Board Router

### Thesis

Chess positions are heterogeneous. Endgames, king attacks, pawn races, blocked centers, and material imbalances may need different feature extractors. A sparse mixture of small board experts can route positions to specialized encoders without requiring a giant monolithic model.

### Fingerprint

```text
board summary
+ sparse top-k router
+ small expert encoders
+ load-balanced expert fusion
+ binary classifier
```

### Why It Is Distinct

- Different from piece-conditioned hypernetworks: this selects among expert computations rather than generating CNN gates.
- Different from ensemble training: experts are trained jointly with a sparse router.
- More practical than many exotic bottlenecks: it can wrap existing CNN or token encoders.

### Architecture Sketch

1. Build a cheap routing summary:

```text
material counts
king locations
side-to-move
coarse occupancy quadrants
small CNN stem pool
```

2. Router outputs logits over `E=6` experts.
3. Select top `k=2` experts per example with a straight-through or soft top-k estimator.
4. Each expert is a small encoder:

```text
expert 1: local CNN
expert 2: dilated CNN
expert 3: token mixer
expert 4: rank/file mixer
expert 5: morphology-lite
expert 6: compact MLP mixer
```

5. Fuse expert outputs by router weights.
6. Classify.

### Tensor Contract

```text
input:             (B, 18, 8, 8)
router_summary:    (B, R)
router_logits:     (B, E)
selected_weights:  (B, E)
expert_outputs:    (B, E, H)
fused:             (B, H)
logits:            (B, 2)
```

### Regularizers

Use two light regularizers:

```text
load_balance_loss = variance(mean_router_weight_per_expert)
router_entropy_loss = small penalty to avoid uniform routing
```

The router should be sparse, but not collapsed.

### Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `single_shared_expert` | Replace experts with one encoder of matched params | Tests expert specialization | MoE should improve if heterogeneity matters. |
| `uniform_expert_average` | Remove learned routing | Tests router value | Learned routing should beat uniform average. |
| `top1_only` | Select one expert | Tests need for blending | Top2 may be more stable. |
| `no_load_balance` | Remove balance loss | Tests collapse risk | May route most examples to one expert. |
| `material_only_router` | Remove spatial routing summary | Tests geometry in routing | Should degrade if route needs board structure. |

### Diagnostics

- Expert usage histogram.
- Expert usage by material bucket.
- Expert usage by correct/incorrect prediction.
- Router entropy distribution.
- Pairwise disagreement between expert logits.

### Failure Modes

- The strongest expert may absorb almost all traffic.
- Sparse routing can make small-batch training noisy.
- If experts are too different in capacity, specialization is hard to interpret.

### Implementation Notes

Start with matched expert output widths and roughly matched parameter counts. Keep total parameters near a baseline by making each expert small. Log expert IDs for validation examples so routing behavior can be inspected.

## Candidate 6: Local Neighborhood Geometry Network

### Thesis

A puzzle-like position may be locally sharp: small current-board perturbations such as removing one piece plane, masking one square neighborhood, or reflecting a safe orientation can move its representation more than a quiet non-puzzle position. The classifier can measure the geometry of a deterministic neighborhood around each board.

### Fingerprint

```text
board tensor
+ deterministic safe perturbation set
+ shared encoder
+ local embedding geometry statistics
+ binary classifier
```

### Why It Is Distinct

- Different from channel-dropout consensus: this builds a structured local neighborhood and measures embedding geometry, not just prediction agreement.
- Different from symmetric-difference twin encoders: it uses many perturbations and curvature-like summaries.
- Different from adversarial training: perturbations are deterministic and current-board-only.

### Perturbation Set

Use transformations that do not require engine or future outcome data:

```text
identity
horizontal mirror with side-aware remapping where valid for diagnostics
mask one random occupied piece channel deterministically by hash bucket
mask king-neighborhood ring
mask corner quadrant
zero coordinate planes
piece-type dropout group
```

Some transforms should be training-only diagnostics if they are not chess-semantics preserving. The model is not told that labels are invariant under all perturbations; it only measures local response.

### Architecture Sketch

1. Generate `V=8` deterministic views for each board.
2. Encode every view with the same encoder.
3. Compute local geometry:

```text
center_embedding
view_deltas
delta_norms
cosine_delta_matrix
local_covariance_eigenvalues
mean_pairwise_distance
max_pairwise_distance
```

4. Concatenate center embedding and geometry statistics.
5. Classify.

### Tensor Contract

```text
views:             (B, V, 18, 8, 8)
embeddings:        (B, V, D)
deltas:            (B, V-1, D)
geometry_stats:    (B, S)
logits:            (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `identity_only` | Remove neighborhood views | Tests local geometry value | Should drop if sharpness matters. |
| `norms_only` | Remove covariance and cosine stats | Tests richer geometry | Full geometry should help if directions matter. |
| `random_noise_views` | Replace structured masks with noise | Tests chess-structured perturbations | Structured views should be better. |
| `stopgrad_views` | Stop gradient through perturbed embeddings | Tests whether encoder adapts to geometry | May stabilize but reduce benefit. |
| `no_center_embedding` | Use only geometry stats | Tests whether geometry alone predicts | Should be weaker but informative. |

### Diagnostics

- Local covariance spectrum by label.
- Pairwise distance heatmap by view type.
- Which perturbation causes the largest class-logit change.
- Geometry-only AUROC.
- False positive sharpness distribution.

### Failure Modes

- Some perturbations may create unrealistic boards and teach artifacts.
- The multi-view cost multiplies encoder runtime.
- If all geometry statistics are learned after a strong center encoder, the model may ignore them.

### Implementation Notes

Use a tiny shared encoder first:

```text
D = 96
V = 6
encoder = small CNN or Piece-Token CNN stem
```

Cache deterministic view masks by board hash if data loading becomes slow. Keep a clean `identity_only` baseline because this architecture can otherwise look better simply from increased compute.

## Recommended Promotion Order

1. `Tensor-Ring Square Interaction Network`
2. `Sinkhorn Role Assignment Network`
3. `Morphological Threat Field Network`
4. `Sparse Expert Board Router`
5. `Local Neighborhood Geometry Network`
6. `Invertible Board Coupling Network`

## Minimal Benchmark Plan

Use the same first-pass benchmark shape for all six:

```text
dataset: crtk_sample_3class
input: simple_18
target: binary puzzle-like
seeds: 3
metrics: accuracy, AUROC, PR-AUC, Brier, ECE
diagnostics: fine-label confusion, source split if available only for reporting
```

Do not use source or provenance fields as model inputs.

## Duplicate Guardrails For Future Ideation

| Candidate | Do not repeat as |
|---|---|
| Tensor-Ring Square Interaction | Another compressed interaction model that only swaps tensor ring for tensor train without changing diagnostics. |
| Sinkhorn Role Assignment | Another role-slot assignment model that only changes role count or Sinkhorn iterations. |
| Morphological Threat Field | Another dilation/erosion board model with only different stencil sizes. |
| Invertible Board Coupling | Another reversible coupling encoder with only different channel width or block count. |
| Sparse Expert Board Router | Another MoE board model with only different number of experts. |
| Local Neighborhood Geometry | Another multi-view perturbation model that only changes the mask list. |

## Best Full-Packet Candidate

`Tensor-Ring Square Interaction Network` is the best candidate to expand next because it has:

- a clear novel computation object for this repo
- controllable parameter budget
- interpretable interaction-order ablations
- direct contrast against CNN, attention, TensorSketch, and piece-token baselines
- a strong chance of detecting multi-piece tactical structure without legal-search features
