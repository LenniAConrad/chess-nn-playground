# Codex Handoff Packet: King-Anchored Euler Interaction Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0809_tuesday_local_euler_interaction.md`
- Generated at: 2026-04-21 08:09 America/Los_Angeles
- Weekday: Tuesday
- Timezone: local
- Idea slug: `euler_interaction`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: **King-Anchored Euler Interaction Network**
- One-sentence thesis: Puzzle-like positions should show sharply organized swept contact, enclosure, and separation events among side-relative piece-role sets around the kings; these events can be measured by Euler-characteristic interaction curves on the current board without legal move search, attack graphs, engine scores, or source metadata.
- Idea fingerprint: `current-board role bitboards + king/center anchored cubical half-plane filtrations + Euler characteristic curves + Euler additivity interaction curves + MLP binary head`.
- Why this is not a common CNN/ResNet/Transformer variant: The central features are not learned local convolutions, residual blocks, or square attention; they are deterministic topological summaries of finite cubical complexes, especially the interaction term `chi(A union B)-chi(A)-chi(B)`, computed over chess-role cell sets under directional sweeps.
- Current-data minimal experiment: Train `KingAnchoredEulerInteractionNet` on `simple_18` using the existing `crtk_sample_3class` train/val/test split for coarse binary classification, mapping fine label `0 -> 0` and fine labels `1,2 -> 1`, while reporting the required `3x2` fine-label diagnostic matrix.
- Smallest central falsification ablation: Keep the same role fields, anchors, directions, threshold grid, feature dimensionality budget, MLP size, material/count summaries, and training setup, but replace every pairwise Euler interaction curve with the two individual Euler curves only plus matched face-count curves. If this matches the main model, the proposed Euler-additivity contact operator is not carrying useful signal.
- Expected information gain if it fails: A clean failure would show that topological contact/enclosure summaries of piece-role geometry do not add useful current-board signal beyond counts and ordinary occupancy scans for this benchmark, letting future cycles avoid cubical Euler morphology and search elsewhere.

## 3. Problem Restatement And Data Contract

The project is `chess-nn-playground`. The task is binary chess puzzle-likeness classification from a single board position:

- output `0`: non-puzzle
- output `1`: puzzle-like

The available fine labels are:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

The default benchmark is binary, but every report must include the rectangular diagnostic confusion matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

The model target for the minimal experiment is coarse binary: fine label `0` is negative, fine labels `1` and `2` are positive. The model must not fabricate class `1` or class `2` labels and must not reinterpret unresolved candidates as verified examples.

Allowed current encodings are:

- `simple_18`: 12 piece planes + side-to-move + castling + en-passant
- `lc0_static_112`
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists

The implementation target is a `torch.nn.Module` accepting:

```text
(batch, C, 8, 8)
```

and returning logits:

```text
(batch, 2)
```

The benchmark split must remain:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

The full roughly 45M-row Parquet dataset must not be used directly until streaming support exists.

Leakage checklist:

- Safe as neural inputs or deterministic derived features: current-board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and geometry derived only from the current board.
- Deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack geometry derived only from the current board are allowed.
- Full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences are leakage-prone unless explicitly justified as rule-only, label-independent, engine-free, and ablated.
- Engine evaluation, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, and dataset provenance must never be neural-network inputs.
- This packet does **not** use legal move generation, attack generation, move counts, checkmate/stalemate tests, engine evaluations, verification metadata, source labels, proposed labels, or unresolved candidate-pool metadata.
- For `lc0_static_112` and `lc0_bt4_112`, deterministic Euler geometry may use only channel indices explicitly documented as current-board piece planes. Any LC0 history channels may be consumed only by a separately declared learned adapter branch, not by the deterministic topology extractor. The first experiment should use `simple_18`; LC0 adapters must fail closed when channel semantics are unknown.

## 4. Research Map

### External ideas used

| Source | What is borrowed | What is not copied |
|---|---|---|
| Ernst Röell and Bastian Rieck, “Differentiable Euler Characteristic Transforms for Shape Classification,” arXiv:2310.07630, accepted at ICLR 2024. URL: https://arxiv.org/abs/2310.07630 | The idea that Euler characteristic transforms can be used as efficient neural-network-compatible topological representations. | No graph classification setup, no point-cloud benchmark, no learned shape coordinates, and no claim that differentiability itself is essential for this chess experiment. |
| O. Hacquard et al., “Euler Characteristic Tools for Topological Data Analysis,” arXiv:2303.14040. URL: https://arxiv.org/html/2303.14040v3 | The use of Euler characteristic curves/profiles as multiscale summaries and the computational appeal of Euler curves over heavier persistence calculations. | No random forest pipeline, no generic TDA benchmark claim, and no import of persistence diagrams. |
| P. Dłotko and D. Gurnari, “Euler characteristic curves and profiles: a stable shape invariant for big data problems,” GigaScience, 2023. URL: https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giad094/7420640 | Stability and scalability motivation for Euler characteristic curves/profiles. | No biological or big-data application, no distributed ECC algorithm requirement. |
| Lewis Marsh and David Beers, “Stability and Inference of the Euler Characteristic Transform,” Discrete & Computational Geometry, 2026. URL: https://link.springer.com/article/10.1007/s00454-025-00763-0 | The ECT definition as a half-space sweep signature and the caution that stability is subtle. | No one-dimensional CW-complex estimator, no Gaussian-process estimator, and no claim that chess boards satisfy their theorem’s hypotheses. |

### Candidate search trace

Internal search screened at least these twelve mechanism families before selecting the final one: tensor-train/MPS board classifiers, determinantal volume collapse, conformal selective classifiers, capsule role binding, score-matching board diffusion, causal environment learning, energy-based latent motif grammars, neural cellular automata on board occupancy, differentiable SAT-like constraint layers, topological persistence diagrams, cubical Euler transforms, and Euler interaction curves.

Serious candidates not selected:

| Candidate mechanism | Why it was serious | Why it lost to the final idea |
|---|---|---|
| Tensor-train / matrix-product-state board classifier | Low-rank tensor networks could model global Boolean dependencies over 64 categorical squares with a controlled entanglement bottleneck. | Too close to weighted finite automata / formal-language modeling and therefore too near the imported ray-language family unless heavily differentiated. It also risks becoming a generic scan-order sequence model. |
| Determinantal latent-volume collapse | A DPP/log-det feature over occupied-piece embeddings could test whether puzzle positions concentrate into lower-dimensional tactical subspaces. | The falsifier is weaker: material count, piece count, and king proximity can create log-det effects without genuine puzzle structure. The Euler interaction operator has a cleaner exact additivity identity. |
| Conformal or selective near-puzzle classifier | Class `1` near-puzzles are genuinely ambiguous, so abstention/calibration could be useful and label-safe. | It is mostly a decision rule or head-level training objective, not a new board operator. It can be layered later on top of a stronger representation. |
| Capsule role-binding with EM routing | Tactical motifs often involve latent roles such as attacker, blocker, target, and king, making capsule-style binding plausible. | Dynamic routing is implementation-risky, hard to falsify cleanly, and too close in spirit to sparse witness-piece bottlenecks. |
| Denoising score / board diffusion auxiliary | Label-free corruption objectives could learn board regularities without engine leakage. | Too close to masked-board code-length/surprise and generative compression packets already imported. It is also compute-heavy for a first minimal experiment. |
| Persistent-homology diagrams on role fields | Holes and connected components around kings are relevant to mating nets and trapped-piece motifs. | Persistence diagrams are overkill on an `8x8` board, less convenient in PyTorch, and harder to ablate than Euler curves. Euler additivity gives a sharper falsifier. |

### Concept-to-operator mapping

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Euler characteristic transform | Half-plane sweep of a finite cubical complex built from role bitboards | `(B, R, 8, 8) -> (B, R, A, U, T)` Euler curves | Replace `chi` with face-count scan curves | Not a CNN, attack graph, sheaf, Hodge operator, move-delta set, transport map, or pseudo-likelihood. |
| Euler additivity / inclusion-exclusion | Pairwise interaction curve `chi(K_r union K_s)-chi(K_r)-chi(K_s)` | `(B, P, A, U, T)` where `P` is selected role pairs | Remove interaction curves while preserving individual curves and counts | This is a topological contact/enclosure observable, not a pairwise piece-constellation polynomial or graph edge label. |
| Chess-specific partial anchoring | Sweep coordinates centered at opponent king, own king, and board center | Anchor tensor `(B, A, 2)` plus fixed directions and thresholds | Replace king anchors with center-only or random legal-safe anchors | It does not enforce orbit invariance or side-tempo interventions; anchors are measurement origins only. |
| Morphological role fields | Deterministic current-board role bitboards such as own pawns, enemy heavy pieces, kings | `simple_18 -> (B, R, 8, 8)` | Shuffle non-king role cells within count-preserving annuli | It does not enumerate legal moves, attacks, one-ply deltas, or piece-target transport couplings. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN | `src/chess_nn_playground/models/cnn.py` | Already present and mainly tests local learned filters, not a new mathematical board observable. |
| Residual CNN | `src/chess_nn_playground/models/residual_cnn.py` | Already present; extra residual depth is ordinary capacity scaling. |
| LC0-style CNN / residual CNN | Existing LC0 BT4-style CNN and residual variants | Too close to copied engine-network priors, and BT4 history is currently zero-filled from single FEN export. |
| Ordinary ViT over 64 squares | Generic square-token Transformer | Disallowed as a core idea and likely data-hungry without a chess-specific falsifiable operator. |
| Plain GNN on 64 square adjacency | Generic grid/square graph network | Too ordinary; message passing on neighboring squares is essentially another learned local architecture unless the graph semantics are new. |
| Hyperparameter tuning | Existing configs and trainers | Not a research idea; it does not create a distinct inductive bias. |
| Ensembling | Any combination of current baselines | Not a single interpretable operator and does not explain puzzle-likeness. |
| Static attack-defense graph / tactical sheaf / Hodge Laplacian | Imported tactical sheaf/Hodge packets | Explicitly covered by imported families and too close even with more edge labels or pooling. |
| One-ply move-delta bag, spectrum, or landscape | Imported counterfactual move-delta packets | Explicitly covered and risks leaning on legal-move enumeration/count shortcuts. |
| Sinkhorn / optimal transport piece-target map | Imported transport packets | Explicitly covered; changing costs or buckets would not be novel enough. |
| Nuisance residualization of material/phase/king features | Imported nuisance-orthogonal packet | Closed-form projection of material/phase nuisance is already represented. |
| Ordinal, Dirichlet, or credal evidence head only | Imported ordinal and credal packets | Could help class `1` ambiguity, but it is head-level and not a new board representation. |
| Sparse witness-piece bottleneck | Imported sparse witness packet | Role capsules or top-k pieces would be near-duplicates unless the formal observable changed. |
| Ray-language automata | Imported ray-language packet | Line/ray token strings are already researched; this idea avoids ray strings and instead uses cubical half-plane topology. |
| Möbius / ANOVA piece constellations | Imported constellation packet | Euler interaction is not an explicit degree-2/3 piece subset polynomial and has a different additivity falsifier. |
| Masked-board codec or pseudo-likelihood ratio | Imported masked-codec and pseudo-likelihood packets | Generative code length/surprise is already represented and not the mechanism here. |

## 6. Mathematical Thesis

### Input space definition

Let `X` be the set of legal or dataset-provided chess positions encoded as tensors `x in R^{C x 8 x 8}`. For the minimal experiment, `C=18` and the first 12 channels are interpreted as white/black piece occupancy planes under the documented `simple_18` convention. Let `s(x) in {white, black}` be side to move.

Define a finite board cell complex `Q` whose 2-cells are the 64 closed unit squares indexed by board coordinates `(i,j) in {0,...,7}^2`, together with their grid edges and vertices.

Define deterministic side-relative role fields:

- `own_pawn`, `own_minor`, `own_heavy`, `own_king`
- `opp_pawn`, `opp_minor`, `opp_heavy`, `opp_king`

where `minor = knight or bishop` and `heavy = rook or queen`. These are binary cell indicators derived only from current piece occupancy and side-to-move. They do not require legal moves or attack generation.

Let `R` be the set of these eight roles. Let `P` be a fixed list of eight role pairs, for example:

```text
(own_heavy, opp_king)
(own_minor, opp_king)
(own_pawn, opp_king)
(own_heavy, opp_heavy)
(own_minor, opp_heavy)
(opp_heavy, own_king)
(opp_minor, own_king)
(opp_pawn, own_king)
```

These are not attack edges. They are only morphological pairings for cubical union/intersection measurements.

### Label/target definition

Let `y_f in {0,1,2}` be the fine label. The binary target is:

```text
y = 0 if y_f = 0
y = 1 if y_f in {1,2}
```

Class `1` is not fabricated. It is used only as an existing verified near-puzzle fine label and as positive under the coarse binary task.

### Data distribution assumptions

The train, validation, and test splits are assumed to be drawn from the same curated split distribution. The model assumes there may be spurious correlations in material, side-to-move, and source construction artifacts, which is why the ablations must preserve material/count summaries while destroying Euler interaction semantics. No assumption is made that all puzzles are mates or king attacks.

### Allowed symmetry or equivariance assumptions

Chess is not invariant under arbitrary board rotations/reflections because pawns, castling, en-passant, side-to-move, and board orientation matter. This model does **not** impose full dihedral invariance.

The only chess-specific normalization is semantic relabeling into `own` and `opponent` roles using side-to-move. Coordinates remain board coordinates. Anchors are measurement origins: opponent king, own king, and board center. The model does not quotient over legal automorphisms, does not Reynolds-pool orbits, and does not perform a side-to-move intervention.

### Core hypothesis

For many puzzle-like positions, decisive tactical structure is reflected not just in which pieces exist, but in how role-defined occupied regions form, merge, separate, and enclose space when swept from king-centered directions. The most useful signal is expected to appear in abrupt changes of Euler curves and in nonzero Euler interaction curves between role pairs near the kings.

### Formal object introduced

For role `r in R`, anchor `a in A(x) = {own_king(x), opp_king(x), center}`, direction `u in U`, and threshold `tau`, define the swept role subcomplex

```text
K_{r,a,u,tau}(x)
  = cubical_closure({ cell c in Q_2 :
       role_r(x,c)=1 and <u, coord(c)-a> <= tau }).
```

The role Euler curve is

```text
E_{r,a,u}(tau; x) = chi(K_{r,a,u,tau}(x)).
```

For role pair `(r,s) in P`, define the Euler interaction curve

```text
J_{r,s,a,u}(tau; x)
  = chi(K_{r,a,u,tau}(x) union K_{s,a,u,tau}(x))
    - chi(K_{r,a,u,tau}(x))
    - chi(K_{s,a,u,tau}(x)).
```

The feature map is the sampled collection of `E`, first differences `Delta E`, `J`, and first differences `Delta J` over fixed anchors, directions, and thresholds.

### Proposition

For hard binary role fields on the finite cubical board complex, for each fixed `(x,a,u,tau,r,s)`,

```text
J_{r,s,a,u}(tau; x) = - chi(K_{r,a,u,tau}(x) intersect K_{s,a,u,tau}(x)).
```

Therefore `J` is zero whenever the swept role complexes are disjoint, and nonzero only when the two swept role complexes have topological contact or overlap through shared cells, edges, or vertices. On the chess board, where different piece roles cannot occupy the same cell, nonzero interaction mostly means boundary contact, adjacency contact, or topological merging/enclosure induced by the cubical closure under the sweep.

### Proof sketch or derivation

Euler characteristic is finitely additive on finite cell complexes:

```text
chi(A union B) = chi(A) + chi(B) - chi(A intersect B).
```

Rearranging gives:

```text
chi(A union B) - chi(A) - chi(B) = -chi(A intersect B).
```

Substitute `A = K_{r,a,u,tau}(x)` and `B = K_{s,a,u,tau}(x)`. Since all complexes are finite subcomplexes of the same cubical board complex, the additivity identity applies.

For implementation, `chi(K)` can be computed by cell counts:

```text
chi(K) = #vertices(K) - #edges(K) + #faces(K).
```

For a role mask on an `8x8` cell grid, faces are selected occupied cells; edges and vertices are included if at least one incident selected face is present. This makes the operator deterministic and inexpensive.

### Optimization objective

Let `Phi(x)` be the flattened Euler feature vector plus optional exact low-order count summaries. The model is:

```text
logits(x) = MLP(Phi(x)).
```

The primary objective is weighted cross-entropy:

```text
min_theta E_{(x,y)} [ w_y * CE(MLP_theta(Phi(x)), y) ].
```

Optional curve dropout randomly removes anchors or directions during training, encouraging the classifier to use redundant topological evidence rather than a single coordinate shortcut.

### What is actually proven

Only the Euler additivity identity and the local cell-count formula for `chi` are proven. The interaction curve is exactly tied to the Euler characteristic of swept role-complex intersections in the hard binary case.

### What remains only hypothesized

It is only a hypothesis that chess puzzle-likeness correlates with these king-anchored Euler interaction curves. It is also only hypothesized that the curves add signal beyond material, piece counts, king distance, and ordinary CNN features.

### Counterexamples where the idea should fail

- A puzzle whose key idea is a quiet move or long forcing line with no distinctive current-board role topology.
- A tactic where the decisive feature is legal move availability, check, mate, stalemate, or pinned-piece legality not visible from occupancy morphology.
- A non-puzzle that has a visually dramatic king-side piece cluster but no tactical solution.
- Endgames where puzzle-likeness is determined by opposition, zugzwang, or exact move order rather than piece-role contact/enclosure.
- Positions whose Euler curves are dominated by material count or king proximity rather than puzzle structure.

### Self-critique

The strongest objection is that an `8x8` Euler morphology signature may reduce to a dressed-up set of count and adjacency features, missing the attack and legal-move semantics that make tactics real. The leakage risk is low because no engine or move-tree data is used, but the shortcut risk is real: king proximity, material imbalance, and role counts might dominate.

The minimal experiment is still worth running because the central ablation is sharp. If `J` interaction curves beat face-count scans and individual curves under count-preserving controls, the result points to a genuinely useful current-board topological signal. If not, the family can be abandoned quickly without scaling.

## 7. Architecture Specification

### Module names

- `KingAnchoredEulerInteractionNet`
- `Simple18RoleAdapter`
- `CubicalEulerCurveLayer`
- `EulerInteractionFeatureBuilder`
- `EulerFeatureMLP`

### Forward-pass steps

Input:

```text
x: (B, C, 8, 8)
```

Step 1: encoding adapter.

For `simple_18`, convert current piece planes and side-to-move into role masks:

```text
roles: (B, R=8, 8, 8)
```

Roles are binary or float masks in `[0,1]`:

```text
0 own_pawn
1 own_minor
2 own_heavy
3 own_king
4 opp_pawn
5 opp_minor
6 opp_heavy
7 opp_king
```

Step 2: anchor extraction.

Find own king and opponent king coordinates from role masks; use soft fallback only for malformed data. The normal case has exactly one own king and one opponent king.

```text
anchors: (B, A=3, 2)
```

Anchors:

```text
0 opponent king
1 own king
2 board center (3.5, 3.5)
```

Step 3: fixed direction and threshold bank.

Use fixed directions:

```text
U = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]
```

Use `T=15` thresholds covering the possible centered projection range, e.g. `[-7, -6, ..., 7]` after coordinate normalization.

Step 4: individual Euler curves.

For each role `r`, anchor `a`, direction `u`, threshold `tau`, form the selected cell mask:

```text
m = roles[:, r] * indicator(dot(coord - anchor, u) <= tau)
```

Compute cubical Euler characteristic by:

```text
faces = selected 8x8 cells
edges = grid edges incident to at least one selected face
vertices = grid vertices incident to at least one selected face
chi = sum(vertices) - sum(edges) + sum(faces)
```

Output:

```text
E: (B, R=8, A=3, U=8, T=15)
DeltaE: (B, 8, 3, 8, 14)
```

Step 5: pairwise Euler interaction curves.

For each role pair `(r,s)` in `P`, compute:

```text
union_mask = clamp(roles[:, r] + roles[:, s], 0, 1)
J = chi(union_mask sweep) - chi(role_r sweep) - chi(role_s sweep)
DeltaJ = finite difference over thresholds
```

Output:

```text
J: (B, P=8, A=3, U=8, T=15)
DeltaJ: (B, 8, 3, 8, 14)
```

Step 6: optional low-order safe summaries.

Append exact current-board summaries computed only from occupancy:

```text
material counts by side and piece type
role counts
side-to-move bit
castling/en-passant bits if available
```

These summaries are not central and must be present in count-preserving ablations as well.

Step 7: flatten and classify.

Default feature dimension without count summaries:

```text
(roles + pairs) * anchors * directions * (T + T-1)
= (8 + 8) * 3 * 8 * 29
= 11136
```

With ~32 count/context summaries:

```text
features: (B, about 11168)
hidden1: (B, 128)
hidden2: (B, 64)
logits: (B, 2)
```

### Parameter-count estimate

The deterministic Euler feature extractor has no trainable parameters. The default MLP has approximately:

```text
11168 * 128 + 128
+ 128 * 64 + 64
+ 64 * 2 + 2
≈ 1.44M parameters
```

If count summaries are disabled, the count is about `1.43M`.

### FLOP or complexity estimate

Euler extraction complexity is:

```text
O(B * (R + P) * A * U * T * 64)
```

With defaults:

```text
(R+P)=16, A=3, U=8, T=15
=> 5760 board sweeps per batch item
=> 368,640 cell-level operations per item
```

The MLP first layer dominates learned multiply-adds:

```text
O(B * 11168 * 128)
```

### Memory and chunking plan

There is no generated move or candidate set. The largest generated tensor should be controlled.

Do **not** materialize:

```text
(B, 16, 3, 8, 15, 8, 8)
```

for a full batch, because at `B=512` this is roughly `377M` float values, about `1.5GB` in fp32 before intermediates.

Recommended chunking:

- Precompute coordinate gates of shape `(A, U, T, 8, 8)` per batch because anchors depend on positions.
- Process one role or one pair at a time:
  - selected mask shape `(B, A, U, T, 8, 8)`
  - at `B=512`, this is about `11.8M` floats, about `47MB` fp32.
- Immediately reduce to `chi` curves `(B, A, U, T)` and append to a feature list.
- If memory is still high, chunk by threshold or by direction.

### Required config fields

```yaml
model:
  name: king_anchored_euler_interaction
  input_channels: 18
  num_classes: 2
  num_thresholds: 15
  directions: king8
  anchors: [opp_king, own_king, center]
  role_set: simple8
  interaction_pairs: default8
  include_first_differences: true
  include_count_summaries: true
  hidden_dim: 128
  second_hidden_dim: 64
  dropout: 0.10
  curve_dropout: 0.05
  encoding_adapter: simple_18
```

### Encoding-adapter assumptions

- `simple_18`: Supported for the minimal experiment. Codex must verify or configure the 12 piece-plane order. If the repository already has an encoding descriptor, use it. Otherwise, default to the documented convention only after adding a test that a synthetic board maps to the intended role masks.
- `lc0_static_112`: Not required for the first experiment. Support only if current-board piece-plane semantics are explicitly available. Deterministic Euler geometry must ignore unknown history or auxiliary planes.
- `lc0_bt4_112`: Not required for the first experiment. If later supported, deterministic Euler geometry must use only the current-board piece planes; history planes may be passed to a separately ablated learned adapter branch but must not affect role extraction unless semantics are explicit.
- All adapters must fail closed: if channel semantics are unknown, raise a clear configuration error instead of guessing.

### Trainer compatibility

The module returns only:

```text
logits: (B, num_classes)
```

No trainer changes should be necessary beyond registry/config additions. Optional feature diagnostics can be saved by hooks or report code, not by changing the model return type.

### Pseudocode

```text
forward(x):
    roles, context = adapter(x)                         # (B,8,8,8), (B,Kc)
    anchors = extract_anchors(roles)                    # (B,3,2)
    features = []

    chi_roles = []
    for r in roles:
        curve = cubical_euler_sweep(roles[:, r], anchors, directions, thresholds)
        chi_roles.append(curve)
        features.append(curve)
        features.append(diff_threshold(curve))

    for (r, s) in interaction_pairs:
        union = clamp(roles[:, r] + roles[:, s], 0, 1)
        curve_union = cubical_euler_sweep(union, anchors, directions, thresholds)
        interaction = curve_union - chi_roles[r] - chi_roles[s]
        features.append(interaction)
        features.append(diff_threshold(interaction))

    if include_count_summaries:
        features.append(context)

    z = flatten_concat(features)
    return mlp(z)
```

## 8. Loss, Training, And Regularization

- Primary loss: balanced weighted cross-entropy over binary labels.
- Auxiliary loss: none required. Optional debug-only curve dropout has no extra loss.
- Class weighting: use existing balanced class weighting from benchmark configs.
- Batch size expectations: default `512`; reduce to `256` if Euler feature extraction memory is high on the available GPU/CPU.
- Learning-rate and optimizer defaults:
  - optimizer: `AdamW`
  - learning rate: `0.001`
  - weight decay: `0.0001`
  - epochs: `3` for the minimal benchmark, matching the existing lightweight comparison regime
  - early stopping patience: `2`
- Regularizers:
  - MLP dropout `0.10`
  - optional curve dropout `0.05` over directions/anchors during training only
  - no stochastic data augmentation in the first run
- Determinism requirements:
  - seed `42`
  - deterministic torch settings where supported
  - fixed direction list, threshold grid, role-pair list, and feature ordering
  - no random anchor selection in the main model
- What must stay unchanged for fair comparison:
  - same train/val/test Parquet split
  - same binary label mapping
  - same metrics and report format
  - same `3x2` fine-label diagnostic matrix
  - same epoch budget and early-stopping policy used for baseline comparisons
  - no full-dataset training until streaming exists

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| `no_euler_interaction` | Removes `J = chi(union)-chi(r)-chi(s)` curves; keeps individual role Euler curves, count summaries, feature budget, and MLP size as close as possible. | Pairwise topological contact/enclosure terms add signal beyond individual swept shapes. | If performance matches the main model, abandon the core interaction thesis. |
| `face_count_curves_only` | Replaces `chi = V-E+F` with swept face counts only, plus first differences. | Euler topology matters beyond simple piece-count scans through half-planes. | If it matches, the topology operator is unnecessary. |
| `individual_euler_only` | Uses role Euler curves and deltas but no pairwise unions or interactions. | Single-role topology around anchors is sufficient or not. | If equal to main, pair interactions are not useful; if equal to count-only, even individual topology is weak. |
| `center_anchor_only` | Removes own-king and opponent-king anchors; uses board-center sweeps only. | King anchoring is a chess-specific source of signal. | If equal, king-centered morphology is not important or the benchmark signal is non-king-related. |
| `king_anchor_randomized_control` | Replaces king anchors with deterministic pseudo-random board cells drawn from a hash of the FEN-independent tensor, keeping three anchors and threshold ranges. | King coordinates matter semantically rather than merely adding more sweep origins. | If equal, anchors are acting as generic positional probes. |
| `annulus_shuffle_nonking_roles` | Within each sample, keeps kings fixed and shuffles non-king role cells inside coarse Chebyshev-distance annuli around the opponent king, preserving role counts and approximate radial histograms. | Angular/contact topology around the king matters beyond material and radial proximity. | If equal, the model is likely using role counts/radial counts, not topology. |
| `material_count_mlp` | Uses only material counts, role counts, side-to-move, castling, and en-passant context. | Euler morphology beats obvious low-order nuisance statistics. | If equal, the idea is just a count shortcut. |
| `same_features_random_threshold_order` | Keeps all numeric curve values but permutes threshold order per feature family before finite differences. | Ordered sweep events matter, not just a multiset of values. | If equal, finite-difference event ordering is not useful. |
| `semantic_pair_shuffle` | Keeps the number of interaction pairs but randomly pairs role channels across the predefined role set, fixed by seed. | The selected chess-role pair semantics matter. | If equal, any extra pairwise features suffice. |
| `thin_cnn_equal_params` | A small CNN/MLP with approximately the same parameter count and same `simple_18` input. | The structured Euler operator gives an advantage over a similarly sized learned local baseline. | If CNN dominates and ablations are flat, the topological bottleneck is too restrictive. |

The smallest central falsification ablation is `no_euler_interaction`.

This idea does not generate legal move sets, candidate move sets, piece-target transport candidates, graph edges, or search-surrogate candidates. Therefore count-only and nuisance-preserving controls focus on role counts, material counts, anchor/radial histograms, and contact topology rather than move-count preservation.

## 10. Benchmark And Falsification Criteria

### Baselines to compare against

Use the same split and comparable epoch budget:

- existing `simple_18` simple CNN
- existing `simple_18` residual CNN
- best current small/medium/deep `simple_18` baseline already present in leaderboard
- optional equal-parameter thin CNN control
- main `KingAnchoredEulerInteractionNet`
- all central ablations in Section 9

Do not compare against LC0 variants as the primary fairness test unless the Euler model is also run on a verified LC0 current-board adapter. The first fair comparison is `simple_18` vs `simple_18`.

### Metrics to inspect

- validation and test accuracy
- validation and test AUROC if existing reporting supports it
- validation and test average precision / PR-AUC if existing reporting supports it
- F1 at the trainer’s default threshold
- binary confusion matrix
- required fine-label `3x2` diagnostic matrix for main model and every central ablation
- calibration diagnostics if already available, but calibration is not the main claim

### Near-puzzle diagnostic

Report fine-label-`1` recall at a matched fine-label-`0` false-positive rate.

Procedure:

1. On validation predictions, choose the positive threshold that gives a specified fine-label-`0` false-positive rate, e.g. `5%` or the nearest attainable value.
2. Apply that threshold to test predictions.
3. Report:
   - fine-label-`1` recall
   - fine-label-`2` recall
   - fine-label-`0` false-positive rate
   - precision among predicted positives

This directly tests whether the model recognizes verified near-puzzles without simply marking too many non-puzzles positive.

### Required artifacts

- trained model checkpoint
- config YAML used
- metrics JSON/CSV
- predictions file with fine labels and predicted probabilities
- binary confusion matrix
- `3x2` fine-label diagnostic matrix
- ablation comparison table
- report with feature/ablation notes
- leaderboard update, if the project already maintains one

### Success threshold

Treat the idea as successful enough to continue if all are true:

- main model improves test AUROC or PR-AUC by at least `+1.0` percentage point over the best comparable `simple_18` baseline, or improves fine-label-`1` recall by at least `+2.0` points at matched fine-label-`0` FPR without hurting fine-label-`2` recall by more than `1.0` point;
- main model beats `material_count_mlp` and `face_count_curves_only`;
- `no_euler_interaction` loses at least `0.5` AUROC point or at least `1.0` fine-label-`1` recall point at matched FPR.

### Failure threshold

Treat the idea as failed if any of these hold:

- main model is within `±0.3` AUROC point of `face_count_curves_only` and `no_euler_interaction`;
- main model is worse than the existing residual CNN by more than `1.0` AUROC point with no near-puzzle diagnostic gain;
- count-only controls match the main model;
- ablations show that random anchors or semantic pair shuffles perform the same as king-anchored semantic pairs.

### What result would make you abandon the idea

Abandon this mechanism if the central interaction features do not separate from individual Euler/count controls under the same training budget. Do not rescue it by adding attacks, move generation, Sinkhorn transport, sheaf layers, larger CNN backbones, or ensembling; those would change the family.

### What result would justify scaling

Scale only if the main model beats count/topology-destroying ablations and improves near-puzzle recall at matched non-puzzle FPR. Reasonable next scaling steps would be more thresholds/directions, learned but constrained role mixtures, or a verified LC0 current-board adapter. Do not scale to the 45M-row full file until streaming is implemented.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_0809_kaein_euler_interaction/idea.yaml` | Create | Machine-readable idea metadata from Section 12. |
| `ideas/20260421_0809_kaein_euler_interaction/math_thesis.md` | Create | Formal definitions of cubical complexes, Euler curves, interaction curves, additivity proposition, and leakage notes. |
| `ideas/20260421_0809_kaein_euler_interaction/architecture.md` | Create | Architecture details, shapes, memory/chunking plan, adapter assumptions, and pseudocode. |
| `ideas/20260421_0809_kaein_euler_interaction/implementation_notes.md` | Create | Notes on role extraction, cubical `V-E+F`, threshold masks, failure-closed adapters, and deterministic tests. |
| `ideas/20260421_0809_kaein_euler_interaction/trainer_notes.md` | Create | Loss, class weighting, epochs, metrics, diagnostic matrices, and fair comparison requirements. |
| `ideas/20260421_0809_kaein_euler_interaction/ablations.md` | Create | Full ablation table from Section 9 plus exact config variants. |
| `ideas/20260421_0809_kaein_euler_interaction/train.py` | Create | Thin wrapper invoking the shared trainer with this idea’s config; no custom trainer fork unless absolutely necessary. |
| `ideas/20260421_0809_kaein_euler_interaction/config.yaml` | Create | Minimal `simple_18` training config. |
| `ideas/20260421_0809_kaein_euler_interaction/report_template.md` | Create | Report template requiring main/ablation metrics and `3x2` diagnostics. |
| `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this packet to imported memory after implementation, including success/failure result and anti-duplicate lesson. Preserve all hard leakage and novelty constraints. |
| `src/chess_nn_playground/models/king_anchored_euler_interaction.py` | Create | `Simple18RoleAdapter`, `CubicalEulerCurveLayer`, `EulerInteractionFeatureBuilder`, `KingAnchoredEulerInteractionNet`. |
| `src/chess_nn_playground/models/registry.py` | Modify | Register builder name `king_anchored_euler_interaction`. |
| `configs/king_anchored_euler_interaction_simple18.yaml` | Create | Shared-trainer config for the main model. |
| `configs/king_anchored_euler_interaction_no_interaction.yaml` | Create | Central falsification ablation. |
| `configs/king_anchored_euler_interaction_face_count.yaml` | Create | Topology-destroying count-curve ablation. |
| `configs/king_anchored_euler_interaction_center_anchor.yaml` | Create | Anchor ablation. |
| `tests/test_king_anchored_euler_interaction.py` | Create | Focused tests for simple synthetic role masks: single cell `chi=1`, two edge-adjacent cells `chi=1`, two separated cells `chi=2`, ring-like shape `chi=0`, interaction identity on hard masks, adapter fail-closed behavior. |

For `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md`, Codex should add a short imported-memory fingerprint after consuming results:

```text
King-Anchored Euler Interaction Network:
current-board role bitboards
+ king/center anchored cubical half-plane Euler curves
+ Euler additivity interaction curves chi(A union B)-chi(A)-chi(B)
+ binary puzzle-likeness target
+ no engine metadata, no attacks, no legal move generation
```

If the idea fails, add an anti-duplicate rule against repeating cubical Euler morphology with only more directions, more thresholds, more role pairs, learned thresholds, or a larger MLP.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0809_tuesday_local_euler_interaction.md
  generated_at: "2026-04-21 08:09 America/Los_Angeles"
  weekday: Tuesday
  timezone: local
  idea_slug: euler_interaction
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_0809_kaein
  name: King-Anchored Euler Interaction Network
  slug: euler_interaction
  status: draft
  created_at: "2026-04-21 08:09 America/Los_Angeles"
  author: ChatGPT Pro
  short_thesis: Puzzle-like positions may be recognized by king-anchored cubical Euler interaction curves measuring swept contact and enclosure among side-relative piece-role fields.
  novelty_claim: Introduces Euler additivity interaction curves over current-board cubical role complexes, not attack/sheaf graphs, move-delta sets, transport, pseudo-likelihood, orbit quotienting, or ordinary CNN/Transformer capacity.
  expected_advantage: Cheap global morphology may improve near-puzzle recall at matched non-puzzle false-positive rate while remaining engine-free and interpretable.
  central_falsification_ablation: no_euler_interaction
  target_task: coarse_binary
  input_representation: simple_18
  output_heads: binary_logits
  compute_notes: Deterministic Euler feature extraction; process one role or pair at a time to avoid materializing large sweep tensors; default MLP about 1.44M params.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/king_anchored_euler_interaction_simple18.yaml
  model_path: src/chess_nn_playground/models/king_anchored_euler_interaction.py
  latest_result_path: null
  notes: First experiment should use simple_18 only; LC0 adapters must fail closed unless current-board piece-plane semantics are explicit.
```

```yaml
config_yaml:
  run:
    name: king_anchored_euler_interaction_simple18
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
    name: king_anchored_euler_interaction
    input_channels: 18
    num_classes: 2
    num_thresholds: 15
    directions: king8
    anchors:
      - opp_king
      - own_king
      - center
    role_set: simple8
    interaction_pairs: default8
    include_first_differences: true
    include_count_summaries: true
    hidden_dim: 128
    second_hidden_dim: 64
    dropout: 0.10
    curve_dropout: 0.05
    encoding_adapter: simple_18
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
  model_name: king_anchored_euler_interaction
  file_path: src/chess_nn_playground/models/king_anchored_euler_interaction.py
  builder_function: build_king_anchored_euler_interaction
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18RoleAdapter
    - CubicalEulerCurveLayer
    - EulerInteractionFeatureBuilder
    - EulerFeatureMLP
    - KingAnchoredEulerInteractionNet
  required_config_fields:
    - input_channels
    - num_classes
    - num_thresholds
    - directions
    - anchors
    - role_set
    - interaction_pairs
    - include_first_differences
    - include_count_summaries
    - hidden_dim
    - second_hidden_dim
    - dropout
    - curve_dropout
    - encoding_adapter
  expected_parameter_count: about_1_44M
  expected_memory_notes: No move/candidate set; process one role or role-pair sweep at a time. One sweep tensor at B=512 is about 47MB fp32; avoid materializing all 16 role/pair sweeps at once.
```

```yaml
research_continuity:
  idea_fingerprint: current-board role bitboards + king/center anchored cubical half-plane Euler curves + Euler additivity interaction curves + MLP binary head
  already_researched_family_overlap: Adjacent only to broad topological data analysis; intentionally not a sheaf/Hodge/attack graph, not a move-delta set, not optimal transport, not pseudo-likelihood, not orbit quotienting, and not masked code-length.
  closest_duplicate_risk: Could be mistaken for high-order piece-constellation modeling, but the formal object is Euler characteristic of swept cubical complexes and the central falsifier is Euler additivity interaction removal.
  do_not_repeat_if_this_fails:
    - Cubical Euler board morphology with only more directions, thresholds, or role pairs.
    - Euler interaction curves rescued by a larger MLP or CNN wrapper.
    - King-anchored topological sweeps that still reduce to count/radial shortcuts.
    - Learned-threshold ECT variants unless the failure analysis shows hard thresholds were specifically the bottleneck.
  suggested_next_search_directions:
    - Label-safe selective prediction on top of the strongest non-topological representation, not as a standalone board operator.
    - Causal invariance across genuinely external data-source shifts if source-environment labels become safe for training diagnostics but never inputs.
    - Non-Euler generative motif grammars with explicit anti-codec controls.
    - Tensor-network classifiers only if they avoid ray/WFA duplicate risk with a new falsifier.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add the King-Anchored Euler Interaction fingerprint to imported research memory after implementation. | Prevents future cycles from repeating cubical Euler morphology under a new name. | `Imported Research Memory` |
| If this fails, add an anti-duplicate rule: do not propose ECT/ECC/cubical Euler board morphology variants that merely add directions, thresholds, anchors, role pairs, learned thresholds, or larger MLPs. | Distinguishes a genuine new topological operator from parameterization tweaks. | `Do not propose...` anti-duplicate block |
| Add a reminder that topological descriptors must include count-preserving and semantics-destroying ablations. | Euler curves can collapse into count/radial shortcuts; future prompts should force hard controls. | `Ablation Plan` requirements |
| Add a note that deterministic current-board topology is allowed only if it avoids legal move generation, attack graph construction, engine labels, and source metadata. | Keeps safe geometry clear while preventing accidental move/search leakage. | `Problem Restatement And Data Contract` |
| After results, record whether near-puzzle recall at matched fine-label-0 FPR improved or not. | The most informative outcome may be diagnostic, not just top-line accuracy. | `Research Continuity` |

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
- Not an exact near-duplicate of imported orbit-symmetry, tempo-intervention, credal-evidence, rule-partition-invariance, kinematic-commutator, or masked-codec packets: yes
