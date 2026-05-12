# Codex Handoff Packet: King-Anchored Material-Null Transport Bottleneck

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0657_tuesday_local_material_ot_bottleneck.md`
- Generated at: 2026-04-21 06:57:28 America/Los_Angeles
- Weekday: Tuesday
- Timezone: local, America/Los_Angeles, UTC-07:00 at generation
- Idea slug: `material_ot_bottleneck`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: King-Anchored Material-Null Transport Bottleneck, abbreviated `KAMN-OTB`.
- One-sentence thesis: A puzzle-like position should often exhibit unusually efficient side-to-move transport of forcing material toward opponent king and value targets, after subtracting a deterministic material-preserving null geometry, and this residual should classify near-puzzles better than raw material or local texture alone.
- Idea fingerprint: `current-board piece slots + king-zone target slots + learned entropic optimal transport costs + deterministic material-preserving null shuffles + residual transport descriptors + binary puzzle-likeness target + no engine/search/label metadata`.
- Why this is not a common CNN/ResNet/Transformer variant: The central computation is a masked Sinkhorn optimal-transport operator over current piece-target candidates and material-null candidates, not convolution over pixels, residual depth, square-token self-attention, or LC0 plane copying.
- Current-data minimal experiment: Train `material_null_ot_bottleneck` on `simple_18` using the existing `crtk_sample_3class` train/val/test Parquet splits for 3 epochs, compare against existing `simple_18` simple CNN and residual CNN configs, and report binary metrics plus the required fine-label `0/1/2 -> predicted 0/1` matrix.
- Smallest central falsification ablation: `coordinate_shuffle_preserve_counts`, which preserves side-to-move, piece identities, material counts, candidate counts, king squares, source-square marginal, and target-role histogram, but randomly permutes non-king square geometry before the transport cost; if it matches the main model, the transport geometry claim is false or too weak.
- Expected information gain if it fails: A clean failure would show that piece-target OT geometry and material-null centering do not add signal beyond material, king placement, and candidate-count shortcuts on this split, steering the next cycle toward label-safe calibration or causal environment design rather than another spatial relational operator.

## 3. Problem Restatement And Data Contract

The project is `chess-nn-playground`. The current task is binary chess puzzle-likeness classification from current board positions:

- output `0`: non-puzzle
- output `1`: puzzle-like

The available fine labels are:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

The default benchmark is binary, with reports also containing the rectangular diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

The model must be a PyTorch `torch.nn.Module` accepting tensors shaped `(batch, C, 8, 8)` and returning logits shaped `(batch, 2)`. The shared trainer, reports, confusion matrices, predictions, and leaderboards should keep working.

Current encodings are:

- `simple_18`: 12 piece planes plus side-to-move, castling, and en-passant information.
- `lc0_static_112`.
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists.

Current benchmark split:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

The full Parquet dataset is roughly 45M rows and must not be used directly until streaming support exists.

Leakage checklist:

- Allowed as neural inputs or deterministic transforms: current board piece occupancy, deterministic board coordinates, side-to-move, castling/en-passant planes already present in the encoding, king locations derived from piece planes, empty-board role geometry, and pseudo-target king-zone geometry derived only from the current board.
- Allowed only as training targets or diagnostics, never as input features: the binary target and the fine labels `0/1/2`.
- Forbidden as neural-network inputs: Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate-pool status, and any label-generation metadata.
- Full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences are leakage-prone unless explicitly justified as rule-only, label-independent, engine-free, and ablated. This idea does not need full legal-move generation.
- For `lc0_static_112` and `lc0_bt4_112`, deterministic geometry may only use channels whose current-board piece semantics are explicitly known. History channels may be consumed only by a learned neural adapter in later experiments, not by the deterministic piece-target transport builder. Unknown 112-plane channel semantics must fail closed.

Boundary between safe rule-derived features and leakage in this idea:

- Safe: extracting actual current pieces and kings from known piece planes; building source piece slots and opponent piece or king-zone target slots; computing empty-board chess-geometry distances such as Chebyshev, Manhattan, file/rank/diagonal alignment, knight-hop distance, and pawn-forward deltas from current coordinates; deterministic null shuffles used only for feature centering, not as labeled examples.
- Not used in the minimal experiment: legal move generation, legal move counts, checking status, mate/stalemate detection, engine evaluation, search, or FEN provenance.

## 4. Research Map

External ideas used:

1. Marco Cuturi, “Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances,” NeurIPS 2013, arXiv: https://arxiv.org/abs/1306.0895. Borrowed: entropic regularization of optimal transport and Sinkhorn-Knopp style scaling as a fast differentiable approximation. Not copied: MNIST/image retrieval setup, any dataset, any evaluation result, or any chess-specific claim.
2. Gabriel Peyré and Marco Cuturi, “Computational Optimal Transport,” arXiv/book: https://arxiv.org/abs/1803.00567 and https://optimaltransport.github.io/book/. Borrowed: the viewpoint that OT compares distributions through global transport plans under local costs. Not copied: applications to color, texture, graphics, or density fitting.
3. Yaroslav Ganin et al., “Domain-Adversarial Training of Neural Networks,” JMLR 2016/arXiv: https://arxiv.org/abs/1505.07818. Borrowed only as an optional future diagnostic: a gradient-reversal material adversary can test whether transport descriptors leak material/phase nuisance. Not copied: domain adaptation objective as the central training method.
4. Martin Arjovsky et al., “Invariant Risk Minimization,” arXiv: https://arxiv.org/abs/1907.02893. Borrowed only as a rejected research direction: the desire to separate causal signal from environment-specific correlations. Not copied: IRM objective, environment construction, or its theoretical claims.

Candidate search trace, including serious mechanisms not selected:

- `Encoding-environment causal invariance`: Train one representation that has invariant risk across `simple_18`, `lc0_static_112`, material phase, and side-to-move environments. It lost because the current trainer is built around one encoding at a time, environment definitions would dominate the experiment, and the first run would test trainer plumbing more than a new chess operator.
- `Label-safe ordinal/selective calibration`: Use fine labels `0 < 1 < 2` through an ordinal auxiliary head and report class-1 selective recall. It lost because it is mostly a loss/calibration idea; useful later, but it does not introduce a board-structural inductive bias.
- `Persistent Euler or morphological texture network`: Build differentiable Euler-characteristic curves over learned board fields. It lost because it is mathematically distinct but hard to justify for chess tactics without first inventing attack fields, which would drift toward the imported attack-geometry family.
- `Piece-removal influence bottleneck`: Score each current piece by the change in a learned board embedding when that piece is removed. It lost because it is a counterfactual candidate-set model and would sit too close to the imported move-delta family, even though it does not enumerate moves.
- `Energy-based masked-board anomaly model`: Detect puzzle-likeness as high density-ratio surprise under a non-puzzle board model. It lost because it could learn dataset provenance or composer/source artifacts, and its failure modes would be hard to distinguish from poor density modeling.
- `Plain optimal transport without a material null`: Transport own pieces to opponent targets using a learned chess cost. It lost to the selected idea because raw transport can be a disguised material-count or king-centrality feature; the material-preserving null residual makes the falsification sharper.

The broader internal search also considered chess-equivariant convolutions, square-token transformers, graph diffusion, differentiable search surrogates, material-only causal baselines, contrastive augmentations, and latent mixture models. The selected idea survived because it has a compact mathematical operator, a direct current-data experiment, and strong shortcut-preserving ablations.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN over `simple_18` | `src/chess_nn_playground/models/trunk/cnn.py` | Already exists and tests local texture without the proposed global piece-target transport bottleneck. |
| Residual CNN over `simple_18` | `src/chess_nn_playground/models/trunk/residual_cnn.py` | Already exists; more residual depth would be routine architecture scaling, not a new research mechanism. |
| LC0-style CNN/residual CNN over 112 planes | Existing LC0 BT4-style CNN and residual variants | Already covered by the baseline suite and too close to copying a stronger chess-engine input format. |
| Ordinary ViT over 64 square tokens | Standard square-token Transformer | Too generic; self-attention could learn anything or nothing, and the hypothesis would be weaker than the transport-residual hypothesis. |
| Plain GNN on 64 board squares | Common graph neural network baseline | Too ordinary and too close to “make a graph from the board” without a falsifiable chess-specific operator. |
| Hyperparameter tuning, depth/width sweeps, optimizer tuning | Any existing model | Explicitly disallowed as the core idea and would not create an informative failure. |
| Ensembling CNNs, ResNets, or 112-plane models | Any leaderboard ensemble | Explicitly disallowed as the core idea; it would obscure whether a mechanism works. |
| Train directly on the 45M-row full file | Data scaling baseline | Unsafe until streaming support exists and not a model idea. |
| Static attack-defense sheaf/Hodge/Laplacian/curvature/tension variant | Imported tactical sheaf/Hodge packets | Already researched; adding more edge types, relation labels, or pooling would be a duplicate family. |
| One-ply move-delta DeepSets/attention/spectrum/free-energy landscape | Imported counterfactual move-delta packets | Already researched; the selected idea deliberately avoids one-ply move enumeration and move-delta pooling. |
| Material-only MLP or material+phase logistic model | Nuisance diagnostic baseline | Useful as a control, but it is a shortcut detector, not a puzzle-structure model. |
| Ordinal near-puzzle calibration only | Standard auxiliary-loss classifier | Label-safe and useful, but it changes supervision more than representation and does not answer what makes a board puzzle-like. |
| Masked-board autoencoder pretraining | Generic self-supervised vision/board model | Could learn dataset/source artifacts and does not directly test tactical geometry. |
| Piece-removal counterfactual influence set | Occlusion/counterfactual candidate model | Too close to the imported counterfactual family and likely to require extra candidate-set ablations without a cleaner operator. |

## 6. Mathematical Thesis

Input space definition:

Let `S = {0,...,7} x {0,...,7}` be the board squares, `A = {white, black}` the colors, and `R = {K,Q,R,B,N,P}` the piece roles. For an encoding `e`, the raw input is

```text
x in X_e subset R^{C_e x 8 x 8}.
```

A fail-closed adapter `A_e` maps known current-board piece channels to

```text
P(x) in {0,1}^{2 x 6 x 8 x 8},
s(x) in {white, black},
```

where `P[a,r,i] = 1` means color `a`, role `r`, occupies square `i`. The minimal experiment uses `e = simple_18`, where the 12 piece planes and side-to-move plane are known. For 112-plane encodings, `A_e` must refuse deterministic geometry unless an explicit channel map is provided.

Target definition:

Let fine label `c in {0,1,2}`. The benchmark target is

```text
y = 1[c in {1,2}],
```

with fine label `c` used only for diagnostics and optional label-safe auxiliary losses, never as model input.

Data distribution assumptions:

Assume examples are sampled from a distribution over current boards, fine labels, and nuisance variables:

```text
(X, C, Y, N) ~ D,
```

where `N` includes material balance, phase, side-to-move prevalence, source/provenance artifacts, and other non-causal shortcuts. The working assumption is not that puzzles are exactly “king attacks.” The weaker assumption is that, conditional on material and king anchors, many verified puzzles and near-puzzles have an abnormal global arrangement of forcing pieces relative to valuable opponent targets.

Allowed symmetry or equivariance assumptions:

Chess is not invariant under the full dihedral group. Pawns, castling rights, and side-to-move break many image symmetries. The model may use side-relative coordinates and may optionally average a horizontal file mirror because a left-right mirror preserves pawn direction and most rule geometry. It must not assume arbitrary rotations or vertical flips. Candidate order must be permutation-invariant: swapping two same-role pieces in the padded candidate list must not change the logits.

Core hypothesis:

Let `a = s(x)` be the side to move and `b` the opponent. Let `T_real(x,a->b)` be a vector of entropic transport descriptors from side `a` source pieces to opponent piece and king-zone targets. Let `T_null(x,a->b)` be the expected descriptor under a deterministic material-preserving null distribution that keeps side-to-move, both king squares, piece identities, source counts, target counts, and target-role histogram, but destroys non-king spatial relations. Define the residual

```text
Z(x) = [T_real(x,a->b) - E_null T_null(x,a->b),
        T_real(x,b->a) - E_null T_null(x,b->a),
        signed differences between the two directions].
```

The hypothesis is that `P(Y=1 | X=x)` is better approximated by a low-complexity function of `Z(x)` than by a same-size function of material counts or shuffled spatial geometry.

Formal object introduced by the idea:

For each direction `a -> b`, construct source candidates

```text
U_a(x) = {(r_u, i_u): P[a,r_u,i_u] = 1},      |U_a| <= 16,
```

and target candidates

```text
V_b(x) = opponent piece targets union king-zone pseudo-targets,
|V_b| <= 16 + 9 = 25.
```

Each source candidate `u` has positive mass

```text
mu_u = softplus(alpha_{r_u}) / sum_{u'} softplus(alpha_{r_{u'}}),
```

and each target candidate `v` has positive demand

```text
nu_v = softplus(beta_{type(v)}) / sum_{v'} softplus(beta_{type(v')}),
```

with epsilon smoothing on masks for numerical stability. For transport head `h`, define a nonnegative learned chess-geometry cost

```text
C_h(u,v;x) = softplus(theta_h^T phi(u,v;x) + b_h),
```

where `phi` contains only rule-independent current-board geometry: normalized Manhattan distance, Chebyshev distance, file/rank/diagonal alignment, empty-board shortest-hop distances for piece roles, pawn-forward deltas relative to color, source role, target type, target value bucket, king-zone indicator, and optional blocker-count features disabled in the minimal experiment.

The entropic transport plan is

```text
Pi_h^epsilon(x) = argmin_{Pi >= 0}
    <Pi, C_h(x)> + epsilon * sum_{u,v} Pi_{uv}(log Pi_{uv} - 1)
subject to Pi 1 = mu, Pi^T 1 = nu.
```

The descriptor `T_real` pools `Pi_h^epsilon` and `C_h` into scalars such as expected cost, plan entropy, max and top-k concentration, mass delivered to king-zone targets, value-bucket mass, and distance-bucket mass. `T_null` uses the same operator after material-preserving non-king coordinate shuffles.

Variational principle explaining why the mechanism should help:

The transport plan is the maximum-entropy coupling that satisfies the source and target mass constraints while minimizing expected learned chess-geometry effort. Thus it is a global, permutation-invariant relaxation of “how easily can this side allocate its active pieces toward the opponent’s high-value and king-zone targets?” The material-null subtraction removes the part of this effort expected from material composition and king anchors alone. If puzzle-likeness is partly encoded by abnormal directed allocation pressure rather than material alone, the residual transport descriptors are a lower-dimensional sufficient statistic for that component.

Proposition:

For fixed positive masked masses `mu, nu`, entropy parameter `epsilon > 0`, and finite cost matrix `C`, the entropic OT objective above has a unique optimizer. The optimizer is invariant to any simultaneous relabeling of source and target candidate indices that preserves masses and costs, and the pooled descriptors are therefore independent of padded candidate ordering. Moreover, with epsilon smoothing, the Sinkhorn iterates define a differentiable map from learned cost parameters to transport descriptors wherever masks are fixed.

Proof sketch or derivation:

The feasible transport polytope with positive marginals is convex and compact. The linear cost term is convex, and the negative entropy regularizer is strictly convex on the positive interior for `epsilon > 0`, so the objective has a unique minimizer. Sinkhorn scaling solves the first-order optimality form

```text
Pi = diag(u) exp(-C/epsilon) diag(v)
```

for positive scaling vectors `u,v` that match the marginals. Relabeling candidate indices permutes rows and columns of `C`, `mu`, and `nu`; the unique optimizer is permuted in the same way, and permutation-symmetric pooled descriptors remain unchanged. Differentiability follows from differentiability of the stabilized Sinkhorn iterations with fixed masks and positive smoothed masses.

What is actually proven:

- The entropic OT subproblem is well-defined and uniquely solvable under positive smoothed marginals.
- The descriptor is invariant to arbitrary padded candidate ordering.
- The operator uses only current-board deterministic geometry and therefore does not by itself introduce engine-analysis leakage.
- The material-null residual can be implemented without fabricating labels because null boards are never supervised as examples.

What remains only hypothesized:

- That residual piece-target transport geometry is predictive of verified puzzle or near-puzzle labels.
- That material-null centering suppresses shortcuts more than it removes useful signal.
- That empty-board geometry plus king-zone targets approximates tactical pressure well enough without legal move generation or search.
- That the class-1 near-puzzle diagnostic benefits, rather than only class-2 puzzle recall.

Counterexamples where the idea should fail:

- Quiet endgame studies where the tactic is zugzwang, opposition, triangulation, fortress logic, or tempo rather than piece-target pressure.
- Positions whose best puzzle move is an underpromotion, stalemate resource, or long forcing maneuver not reflected by current transport geometry.
- Positions with enormous apparent king pressure but no sound tactic because of a single defensive resource; the model deliberately does not search for that resource.
- Material-null shuffles may subtract real signal in sparse endgames where every square matters.
- If verified puzzle labels in this split are dominated by source/provenance artifacts, a geometry bottleneck may underperform a conventional CNN that accidentally learns those artifacts.

Self-critique:

The strongest objection is that this operator may be an elegant way to compute “pieces are near the enemy king,” which a small CNN can already infer. The material-null residual and coordinate-shuffle ablations make this objection testable: if preserving material, candidate counts, target-role histograms, source-square marginals, and king squares while destroying source-target geometry yields the same performance, the idea should be abandoned. The experiment is still worth running because it is small, label-safe, compatible with the existing trainer, and its failure would rule out a whole class of OT geometry bottlenecks rather than merely a hyperparameter setting.

## 7. Architecture Specification

Module names:

- `Simple18PieceAdapter`
- `LC0CurrentPlaneAdapter`
- `PieceTargetCandidateBuilder`
- `KingAnchoredMaterialNullSampler`
- `ChessGeometryCost`
- `MaskedLogSinkhorn`
- `TransportDescriptorPool`
- `MaterialNullOTBottleneck`
- Builder function: `build_material_null_ot_bottleneck`

Forward-pass steps and shapes:

1. Input:

```text
x: [B, C, 8, 8]
```

2. Encoding adapter:

```text
piece_planes: [B, 2, 6, 8, 8]
side_to_move: [B] integer color id
```

For `simple_18`, extract the 12 piece planes and side-to-move plane. For `lc0_static_112` and `lc0_bt4_112`, only enable deterministic extraction when `encoding_adapter.channel_map.current_piece_planes` is explicitly configured. If channel semantics are unknown, raise a clear error before training.

3. Candidate builder, for each direction `a->b` in `{stm->opp, opp->stm}`:

```text
source_roles:   [B, Ns]          Ns = max_source_candidates = 16
source_squares: [B, Ns, 2]
source_mask:    [B, Ns]
source_mass:    [B, Ns]

target_types:   [B, Nt]          Nt = max_target_candidates = 25
target_squares: [B, Nt, 2]
target_mask:    [B, Nt]
target_mass:    [B, Nt]
```

Targets are all opponent pieces plus up to 9 king-zone pseudo-targets centered on the opponent king and clipped to the board.

4. Material-null sampler:

```text
null_source_squares: [B, K, Ns, 2]
null_target_squares: [B, K, Nt, 2]
```

Default `K = null_samples = 4`. It preserves side-to-move, both king squares, piece identities, source counts, target counts, and target-role histogram. It shuffles only non-king coordinates into deterministic pseudo-random legal board squares using `seed` and a hash of material counts. Null candidates are used only to compute `E_null T_null`, never as labeled examples.

5. Chess geometry cost builder:

```text
real_cost: [B, H, Ns, Nt]
null_cost: [B, K, H, Ns, Nt]
```

Default `H = transport_heads = 4`. Cost features are generated on the fly and should not be fully materialized as `[B,K,H,Ns,Nt,D_phi]` unless the chunk is small.

6. Masked log-domain Sinkhorn:

```text
real_plan: [B, H, Ns, Nt]
null_plan: [B, K, H, Ns, Nt]
```

Default `sinkhorn_iters = 12`, `sinkhorn_epsilon = 0.08`. Use log-domain updates and masks. Add epsilon mass only inside valid masks, then renormalize.

7. Descriptor pooling:

Per direction and head, compute approximately 15 descriptors:

```text
expected_cost
plan_entropy
max_pair_mass
top4_pair_mass
king_zone_mass
queen_rook_minor_pawn_value_bucket_masses
distance_bucket_masses for 0-1, 2, 3, 4+
```

Shapes:

```text
real_desc: [B, 2 directions, H, Dd]
null_desc: [B, 2 directions, K, H, Dd]
resid_desc = real_desc - mean_K(null_desc): [B, 2, H, Dd]
signed_dir = resid_desc[stm->opp] - resid_desc[opp->stm]: [B, H, Dd]
z = flatten([resid_desc, signed_dir]): [B, 3 * H * Dd]
```

With `H=4` and `Dd=15`, `z` has dimension 180.

8. Classifier head:

```text
LayerNorm(180)
Linear(180, hidden_dim=128)
GELU
Dropout(p=0.05, disabled or deterministic in strict reproducibility mode)
Linear(128, 64)
GELU
Linear(64, 2)
```

Return:

```text
logits: [B, 2]
```

The normal `forward(x)` must return logits only so the shared trainer remains compatible. If optional diagnostics are implemented, gate them behind `return_aux=True` and keep the default false.

Parameter-count estimate:

- Source and target mass weights: under 100 parameters.
- Cost builder with 4 heads and approximately 24 geometry features: under 500 parameters if linear; under 10k if role embeddings and a small MLP are used.
- Descriptor MLP: about 35k to 60k parameters for default descriptor size.
- Optional material adversary or diagnostic heads: under 20k, disabled in the minimal experiment.
- Expected total default: approximately 50k to 100k trainable parameters, far smaller than typical CNN baselines.

FLOP and complexity estimate:

Let `B` be batch size, `H` heads, `K` null samples, `I` Sinkhorn iterations, `Ns <= 16`, and `Nt <= 25`. The dominant cost is

```text
O(B * 2 directions * H * (1 + K) * I * Ns * Nt).
```

At `B=512`, `H=4`, `K=4`, `I=12`, `Ns=16`, `Nt=25`, this is about `98M` pair-cell Sinkhorn updates per batch, plus cost-feature computation. This should be feasible on GPU, but Codex should benchmark throughput.

Memory estimate and chunking plan:

Cost and plan tensors require approximately

```text
B * 2 * H * (1 + K) * Ns * Nt * bytes_per_float
```

per tensor. With `B=512`, float32, and default values, one such tensor is about `31 MB`; keeping cost and plan together is about `62 MB`, excluding temporary log-domain buffers. Avoid materializing pair feature tensors. Add config `max_pair_cells_per_chunk`, default `2_000_000`, and chunk over batch or null samples when

```text
B * 2 * H * (1 + K) * Ns * Nt > max_pair_cells_per_chunk.
```

Required config fields:

```yaml
model:
  name: material_null_ot_bottleneck
  input_channels: 18
  num_classes: 2
  encoding_adapter: simple_18
  max_source_candidates: 16
  max_target_candidates: 25
  transport_heads: 4
  null_samples: 4
  sinkhorn_iters: 12
  sinkhorn_epsilon: 0.08
  hidden_dim: 128
  descriptor_dropout: 0.05
  use_blocker_cost: false
  use_material_adversary: false
  fail_closed_unknown_channels: true
  max_pair_cells_per_chunk: 2000000
```

Encoding-adapter assumptions:

- `simple_18`: supported in the first experiment. The adapter must assert the expected 18-channel layout and expose piece planes and side-to-move. Castling and en-passant planes are not used by the deterministic transport branch in the minimal experiment, but remain available to future learned adapters.
- `lc0_static_112`: optional later. Deterministic extraction requires an explicit channel map for current-board piece planes. Unknown or ambiguous channel order must raise `ValueError`.
- `lc0_bt4_112`: optional later. Current-board channels may be used only if explicitly mapped. History planes are not valid deterministic geometry inputs; they may only feed a learned neural adapter in a separate extension, with an ablation proving they help.

Pseudocode, not final implementation:

```python
class MaterialNullOTBottleneck(nn.Module):
    def forward(self, x, return_aux: bool = False):
        P, stm = self.adapter(x)  # [B,2,6,8,8], [B]
        all_residuals = []
        aux = {}

        for direction in [STM_TO_OPP, OPP_TO_STM]:
            src, tgt = self.candidates(P, stm, direction)
            real_cost = self.cost(src.squares, src.roles, tgt.squares, tgt.types, direction)
            real_plan = self.sinkhorn(real_cost, src.mass, tgt.mass, src.mask, tgt.mask)
            real_desc = self.pool(real_plan, real_cost, src, tgt)

            null_descs = []
            for k in range(self.null_samples):
                nsrc, ntgt = self.null_sampler(src, tgt, k)
                ncost = self.cost(nsrc.squares, nsrc.roles, ntgt.squares, ntgt.types, direction)
                nplan = self.sinkhorn(ncost, nsrc.mass, ntgt.mass, nsrc.mask, ntgt.mask)
                null_descs.append(self.pool(nplan, ncost, nsrc, ntgt))

            resid = real_desc - torch.stack(null_descs, dim=1).mean(dim=1)
            all_residuals.append(resid)

        signed = all_residuals[0] - all_residuals[1]
        z = torch.cat([all_residuals[0], all_residuals[1], signed], dim=-1)
        logits = self.classifier(z)
        return (logits, aux) if return_aux else logits
```

## 8. Loss, Training, And Regularization

Primary loss:

```text
CrossEntropyLoss(logits, binary_target)
```

Use the repository’s balanced class weighting for coarse binary mode.

Optional auxiliary loss:

- Disabled in the minimal experiment.
- Optional follow-up only: a material/phase adversary using gradient reversal from transport descriptors to material-count bins. This tests nuisance leakage but requires trainer support for auxiliary losses. It must never use source labels, verification metadata, or dataset provenance.

Class weighting:

- Use existing `class_weighting: balanced`.
- Do not rebalance fine label `1` separately in the minimal experiment; report fine-label diagnostics instead.

Batch size expectations:

- Default `batch_size: 512` on GPU.
- If memory is high because of null samples, reduce to 256 or chunk pair cells; do not change the split or training target.

Learning-rate and optimizer defaults:

- Optimizer: `AdamW`.
- Learning rate: `0.001`.
- Weight decay: `0.0001`.
- Epochs: `3` for the minimal benchmark.
- Early stopping patience: `2`.
- Mixed precision: `false` for first deterministic run. Consider mixed precision only after verifying Sinkhorn numerical stability.

Regularizers:

- Descriptor dropout `0.05`, disabled if strict deterministic reproducibility requires no dropout.
- Clamp or softplus cost values to a stable range, for example `[1e-4, 20]` after softplus or by using normalized features.
- Entropic regularization `epsilon=0.08` default.
- Optional gradient clipping at global norm `1.0` if Sinkhorn gradients spike.

Determinism requirements:

- `seed: 42`.
- Deterministic null sampler: no unseeded randomness; null coordinates are a pure function of seed, material signature, candidate index, and null sample index.
- `torch.use_deterministic_algorithms(true)` when supported by the project.
- Report if any GPU operation prevents full determinism.

What must stay unchanged from existing benchmark configs for a fair comparison:

- Same train/val/test Parquet paths.
- Same binary target mapping: fine `1` and `2` map to binary `1`.
- Same data filtering and preprocessing.
- Same metrics and report format.
- Same threshold-selection protocol.
- Same epoch budget for baseline comparison unless explicitly reported as a scaling run.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| `coordinate_shuffle_preserve_counts` | Preserves side-to-move, piece identities, material counts, source/target candidate counts, king squares, source-square marginal, and target-role histogram, but permutes non-king coordinates before cost computation. | Real source-target geometry matters beyond nuisance-preserving square statistics. | If unchanged, the model is not using the proposed geometry; abandon or radically redesign. |
| `no_material_null_residual` | Feeds `T_real` directly instead of `T_real - mean(T_null)`. | Abnormal geometry relative to material-preserving null matters. | If equal or better, the null is unnecessary or subtracts signal; simplify to raw OT or abandon the material-null thesis. |
| `count_only_material_control` | Replaces transport descriptors with material counts, candidate counts, side-to-move, target-role histogram, and king-square coordinates. | The model beats obvious shortcuts. | If it matches the main model, transport is decorative and should not be scaled. |
| `role_only_no_square_cost` | Keeps source roles and target types but sets all geometric distance features to constants. | Spatial geometry, not just piece values and roles, is useful. | If unchanged, the learned classifier is exploiting role/value histograms. |
| `product_coupling_no_sinkhorn` | Replaces optimized `Pi` with independent coupling `mu nu^T`. | Entropic assignment structure matters, not just marginals. | If unchanged, Sinkhorn is unnecessary; use simpler feature engineering or abandon OT. |
| `degree_preserving_cost_permutation` | Randomly permutes cost entries within source-role/target-type buckets while preserving row and column masks and approximate cost histograms. | The semantic alignment of costs to coordinates matters. | If unchanged, cost semantics are not doing work. |
| `null_only_classifier` | Uses only `mean(T_null)` descriptors. | Material-null features are merely controls, not sufficient predictors. | If strong, material/king artifacts dominate; tighten nuisance controls. |
| `no_king_zone_targets` | Removes the 3x3 opponent king-zone pseudo-targets, leaving only opponent material targets. | King anchoring is important for puzzle-likeness. | If unchanged, the operator may just detect material/value proximity rather than tactical king pressure. |
| `target_shuffle_preserve_source_marginal` | Keeps source candidates and their square marginal fixed, preserves target types and counts, but shuffles target coordinates away from their real pieces except kings. | Pairwise source-target geometry matters beyond source-square features. | If unchanged, source placement alone explains the result. |
| `blocker_cost_optional_on` | Enables deterministic blocker-count features along rank/file/diagonal rays. | Whether minimal empty-board geometry is enough or blockers add useful safe rule information. | If blocker features help substantially, rerun with a blocker-destroying ablation before scaling. |
| `null_samples_0_1_4_8` | Varies null sample count from none to 8. | The result is stable and not an artifact of one deterministic null shuffle. | If performance swings wildly, the null estimator is unstable and needs redesign. |

The smallest central falsification is `coordinate_shuffle_preserve_counts`. It is semantics-destroying and nuisance-preserving. It keeps the obvious shortcuts named in the prompt: candidate count, material, side-to-move, piece identity, source-square marginal, target-role histogram, and pseudo-capture/value histogram where applicable, while destroying the proposed source-target transport semantics.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- Existing `simple_18` simple CNN, same split and epoch budget.
- Existing `simple_18` residual CNN, same split and epoch budget.
- Existing LC0-style CNN/residual results may be reported as context, but the primary fair comparison is `simple_18` because this idea’s first implementation uses `simple_18` deterministic piece semantics.
- `count_only_material_control` ablation as a nuisance baseline.
- `coordinate_shuffle_preserve_counts` and `no_material_null_residual` as central ablations.

Metrics to inspect:

- Test accuracy.
- Balanced accuracy.
- AUROC.
- PR-AUC.
- F1 for binary puzzle-like class.
- Brier score and expected calibration error if already available.
- Required `2x2` binary confusion matrix.
- Required fine-label `0/1/2 -> predicted 0/1` matrix for the main model and every central ablation.

Near-puzzle diagnostic:

- On validation, choose a threshold that matches the best `simple_18` residual CNN’s fine-label-`0` false-positive rate, or use a fixed fine-label-`0` FPR target such as `10%` if the baseline thresholding does not expose this directly.
- Report fine-label-`1` recall at that matched fine-label-`0` FPR on test.
- Also report fine-label-`2` recall at the same threshold to catch models that help near-puzzles but lose verified puzzles.

Required artifacts:

- Main config YAML.
- Model checkpoint path.
- Training log.
- Validation and test metrics JSON.
- Test predictions Parquet or CSV with binary probabilities, predicted labels, true binary label, and fine label.
- `2x2` confusion matrix.
- Fine-label `3x2` diagnostic matrix.
- Ablation configs and result summaries for `coordinate_shuffle_preserve_counts`, `no_material_null_residual`, and `count_only_material_control` at minimum.
- A report Markdown using `ideas/<idea_id>_<slug>/report_template.md`.

Success threshold:

The idea is worth keeping if all are true:

1. Main model improves over the best same-budget `simple_18` CNN/residual baseline by at least `+1.0` percentage point in AUROC or balanced accuracy on test, or by at least `+2.0` percentage points in fine-label-`1` recall at matched fine-label-`0` FPR.
2. Fine-label-`2` recall does not drop by more than `1.0` percentage point at the matched threshold.
3. `coordinate_shuffle_preserve_counts` is at least `0.75` percentage points worse in AUROC or balanced accuracy, or at least `1.5` percentage points worse in the near-puzzle diagnostic.
4. `count_only_material_control` is clearly below the main model.

Failure threshold:

The idea fails if the main model is within noise of existing `simple_18` baselines and central ablations are within `0.5` percentage points of the main model on AUROC/balanced accuracy and within `1.0` percentage point on fine-label-`1` recall at matched fine-label-`0` FPR.

What result would make me abandon the idea:

Abandon this family if `coordinate_shuffle_preserve_counts`, `role_only_no_square_cost`, or `count_only_material_control` matches or beats the main model across both binary metrics and the near-puzzle diagnostic. That would mean the transport geometry is not the source of signal.

What result would justify scaling:

Scale only if the main model beats the strongest same-encoding baseline and central geometry-destroying ablations drop clearly. Scaling steps should be incremental: first increase `null_samples` and `transport_heads`, then optionally add blocker-cost features with ablations, then implement a fail-closed 112-plane adapter.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_material_ot_bottleneck/idea.yaml` | Create | Machine-readable idea metadata, fingerprint, novelty claim, config path, model path, and falsification ablation. |
| `ideas/20260421_material_ot_bottleneck/math_thesis.md` | Create | Expanded version of Section 6, including definitions, proposition, proof sketch, hypotheses, and counterexamples. |
| `ideas/20260421_material_ot_bottleneck/architecture.md` | Create | Module descriptions, tensor shapes, pseudocode, memory/compute estimates, and adapter fail-closed rules. |
| `ideas/20260421_material_ot_bottleneck/implementation_notes.md` | Create | Practical notes for log-domain Sinkhorn, null sampler determinism, candidate padding/masks, and numerical stability. |
| `ideas/20260421_material_ot_bottleneck/trainer_notes.md` | Create | How to run with shared trainer, expected CLI/config fields, no auxiliary losses in minimal experiment, and optional custom train notes. |
| `ideas/20260421_material_ot_bottleneck/ablations.md` | Create | Full ablation table and exact benchmark/falsification criteria. |
| `ideas/20260421_material_ot_bottleneck/train.py` | Create | Thin entrypoint that calls the existing shared trainer with this idea’s config; do not fork trainer logic unless optional auxiliary losses are later enabled. |
| `ideas/20260421_material_ot_bottleneck/config.yaml` | Create | Idea-local config for `simple_18`, `material_null_ot_bottleneck`, default 3 epochs, balanced class weighting, deterministic null sampler. |
| `ideas/20260421_material_ot_bottleneck/report_template.md` | Create | Report skeleton requiring main metrics, `2x2`, `3x2`, near-puzzle diagnostic, and central ablation comparison. |
| `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Preserve hard constraints; add this packet to imported memory after consumption, including anti-duplicate rule for material-null piece-target OT if it fails. |
| `src/chess_nn_playground/models/material_null_ot_bottleneck.py` | Create | Implement adapters, candidate builder, null sampler, chess geometry cost builder, masked log Sinkhorn, descriptor pooling, and `MaterialNullOTBottleneck`. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `material_null_ot_bottleneck` and builder function. |
| `configs/material_null_ot_bottleneck_simple18.yaml` | Create | Shared-trainer config matching Section 12 plus model-specific fields. |
| `tests/test_material_null_ot_bottleneck.py` | Create | Shape test, deterministic forward test, finite-logits test, candidate mask test, and fail-closed unknown 112-channel adapter test. |
| `tests/test_material_null_ot_sinkhorn.py` | Create | Sinkhorn marginal test on small masked examples and permutation-invariance test for candidate ordering. |
| `tests/test_material_null_sampler.py` | Create | Verify material counts, king squares, side-to-move, candidate counts, and deterministic seed behavior are preserved by null shuffles. |

For `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md`, Codex should update the prompt after consuming this output. It should preserve leakage rules, label rules, falsification requirements, and anti-duplicate requirements while adding reusable lessons about material-preserving null controls and transport-operator ablations.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0657_tuesday_local_material_ot_bottleneck.md
  generated_at: 2026-04-21T06:57:28-07:00
  weekday: Tuesday
  timezone: local
  idea_slug: material_ot_bottleneck
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_material_ot_bottleneck
  name: King-Anchored Material-Null Transport Bottleneck
  slug: material_ot_bottleneck
  status: draft
  created_at: 2026-04-21T06:57:28-07:00
  author: ChatGPT Pro
  short_thesis: Puzzle-like positions should show abnormal directed piece-to-target transport toward opponent king and value targets after subtracting a material-preserving null geometry.
  novelty_claim: Uses entropic optimal transport over current piece and king-zone candidates with deterministic material-null shuffles; not a CNN, ResNet, Transformer, tactical sheaf, Hodge operator, or one-ply move-delta landscape.
  expected_advantage: May improve near-puzzle recall by focusing on global geometry residuals instead of material/source shortcuts.
  central_falsification_ablation: coordinate_shuffle_preserve_counts
  target_task: coarse_binary
  input_representation: simple_18 first; optional fail-closed adapters for lc0_static_112 and lc0_bt4_112 only with explicit channel maps
  output_heads: binary logits [batch, 2]
  compute_notes: Default B=512, H=4, K=4, Sinkhorn iters=12, max source=16, max target=25; chunk pair cells above 2000000.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/material_null_ot_bottleneck_simple18.yaml
  model_path: src/chess_nn_playground/models/material_null_ot_bottleneck.py
  latest_result_path: null
  notes: Do not supervise null shuffles as labeled examples; use them only for descriptor centering and ablations.
```

```yaml
config_yaml:
  run:
    name: material_null_ot_bottleneck_simple18
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
    name: material_null_ot_bottleneck
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
  model_name: material_null_ot_bottleneck
  file_path: src/chess_nn_playground/models/material_null_ot_bottleneck.py
  builder_function: build_material_null_ot_bottleneck
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18PieceAdapter
    - LC0CurrentPlaneAdapter
    - PieceTargetCandidateBuilder
    - KingAnchoredMaterialNullSampler
    - ChessGeometryCost
    - MaskedLogSinkhorn
    - TransportDescriptorPool
    - MaterialNullOTBottleneck
  required_config_fields:
    - model.encoding_adapter
    - model.max_source_candidates
    - model.max_target_candidates
    - model.transport_heads
    - model.null_samples
    - model.sinkhorn_iters
    - model.sinkhorn_epsilon
    - model.hidden_dim
    - model.fail_closed_unknown_channels
  expected_parameter_count: approximately 50000 to 100000 for the default descriptor MLP and linear cost heads
  expected_memory_notes: Cost and plan tensors scale as B * 2 * H * (1 + K) * Ns * Nt; default B=512 uses roughly 62 MB for cost plus plan in float32 before temporary buffers; chunk over batch or null samples above 2000000 pair cells.
```

```yaml
research_continuity:
  idea_fingerprint: current-board piece slots plus king-zone target slots plus learned entropic OT costs plus deterministic material-preserving null shuffles plus residual descriptor pooling
  already_researched_family_overlap: Avoids imported tactical sheaf/Hodge/attack-defense graph families and avoids imported one-ply move-delta set/spectrum/landscape families; closest overlap is generic current-board rule-derived geometry.
  closest_duplicate_risk: A raw piece-target OT model without material-null shuffles, or a static attack graph renamed as transport.
  do_not_repeat_if_this_fails:
    - Do not propose another piece-to-target entropic OT bottleneck with only different heads, costs, target buckets, or hidden sizes.
    - Do not propose material-preserving null shuffles around the same piece-target transport operator unless the failure analysis shows the null sampler, not the operator, was defective.
    - Do not convert this into an attack-defense sheaf, Hodge, Laplacian, curvature, or tension-energy variant.
    - Do not convert this into a one-ply move-delta bag, attention set, entropy/free-energy landscape, or move-spectrum model.
  suggested_next_search_directions:
    - Label-safe ordinal or selective prediction for fine labels 0/1/2.
    - Causal environment splits across encoding families, material phases, and data-source shifts.
    - Information bottlenecks that explicitly suppress material/source artifacts without candidate transport.
    - Calibration-first near-puzzle uncertainty modeling.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add this packet to imported research memory with fingerprint `piece-target entropic OT + material-preserving null residual + king-zone targets`. | Prevents the next research pass from repeating the same OT operator with renamed heads or costs. | `Imported Research Memory` |
| Add an anti-duplicate rule: if this fails, do not propose another material-null piece-target Sinkhorn/OT bottleneck unless the new operator changes the falsifiable object, not just cost features or pooling. | Tightens continuity and avoids incremental variants. | `Research Continuity` and anti-duplicate paragraph |
| Add a reusable ablation requirement for any candidate-set model that is not a move-set: preserve material, side-to-move, candidate count, piece identity, source-square marginal, target histogram, and obvious value/capture histograms while destroying semantics. | This packet exposed how easy it is for non-move candidate sets to leak shortcuts through counts and marginals. | `Ablation Plan` requirements |
| Clarify that deterministic null boards or shuffles are allowed only as unsupervised feature-centering controls, never as labeled training examples. | Prevents accidental label fabrication while preserving safe causal-control experiments. | `Problem Restatement And Data Contract` leakage boundary |
| Require Codex to record main-vs-ablation deltas, not just main benchmark numbers, in the next prompt update. | Future research needs to know whether the mechanism failed or only the implementation underperformed. | `Research Continuity` |

Do not weaken leakage rules, label rules, falsification requirements, or anti-duplicate requirements.

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0657_tuesday_local_material_ot_bottleneck.md`
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes
- Falsification criterion is concrete: yes
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported research packets completed: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
