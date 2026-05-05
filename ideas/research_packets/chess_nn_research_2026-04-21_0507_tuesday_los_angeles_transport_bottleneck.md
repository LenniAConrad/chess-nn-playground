# Codex Handoff Packet: Entropic Piece-Target Transport Bottleneck

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0507_tuesday_los_angeles_transport_bottleneck.md`
- Generated at: `2026-04-21 05:07:36 America/Los_Angeles`
- Weekday: `tuesday`
- Timezone: `America/Los_Angeles` / filename token `los_angeles`
- Idea slug: `transport_bottleneck`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: **Entropic Piece-Target Transport Bottleneck**
- One-sentence thesis: A chess position is puzzle-like when side-to-move force can be geometrically coupled to high-value or king-safety targets by a low-cost, high-contrast transport plan that is not reducible to material counts, one-ply move bags, or static attack incidence.
- Idea fingerprint: `current-board piece occupancy + side-to-move canonicalization + normalized piece/source and target/anchor measures + chess-metric entropic optimal transport via Sinkhorn + low-dimensional transport-cost/gap/entropy summaries + binary puzzle-likeness target + no engine metadata, no legal-move tree, no one-ply move-delta set, no attack/sheaf/Hodge operator`
- Why this is not a common CNN/ResNet/Transformer variant: The central computation is a constrained doubly-stochastic coupling between board-conditioned probability measures under fixed chess-metric costs; the classifier sees transport costs, product-plan gaps, and plan entropies, not unconstrained convolutional feature maps or row-softmax token attention.
- Current-data minimal experiment: Train `piece_target_transport_bottleneck` on `simple_18` for the existing `crtk_sample_3class` train/val/test split in coarse-binary mode for 3 epochs, then report the usual binary metrics plus the required `3x2` fine-label diagnostic matrix.
- Smallest central falsification ablation: Replace each Sinkhorn optimal coupling with the independent product coupling `mu \otimes nu`, preserving source marginals, target marginals, material presence, target anchors, pair list, and downstream MLP input dimensionality.
- Expected information gain if it fails: If the product-coupling ablation matches the main model, then marginal source/target summaries and ordinary local features explain the gain; geometric mass matching through optimal transport is not a useful inductive bias for this dataset.

## 3. Problem Restatement And Data Contract

The project is `chess-nn-playground`. The task is binary chess puzzle-likeness classification from a single board position.

The target emitted by the model is:

- output `0`: non-puzzle
- output `1`: puzzle-like

The available fine labels are:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

The default benchmark is binary, normally mapping fine label `0 -> 0` and fine labels `1/2 -> 1`, while reports must still include the rectangular diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Allowed input tensors are current-board encodings already supported by the repository:

- `simple_18`: 12 piece planes + side-to-move + castling + en-passant
- `lc0_static_112`
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists

The implementation target is PyTorch. The model must be an `nn.Module` accepting:

```text
(batch, C, 8, 8)
```

and returning logits:

```text
(batch, 2)
```

The benchmark split remains:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

The full Parquet dataset has roughly 45M rows and must not be used directly by the current trainer until streaming support exists.

Leakage checklist:

- Safe: deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack geometry derived only from the current board.
- Safe in this idea: empty-board chess distance matrices such as Chebyshev distance, rook-move distance, bishop-color distance, knight-move distance, and pawn-oriented forward distance. These are deterministic rule geometry and do not use engine analysis or future positions.
- Risky unless separately justified and ablated: full legal-move generation, legal move counts, checkmate/stalemate oracles, forced-line search, move-tree consequences, repetition/50-move state, or any result of exploring successor positions.
- Forbidden as neural-network inputs: engine evaluation, Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, or unresolved candidate-pool status.
- Fine labels `1` and `2` may be used only as supervised labels/diagnostics if the existing training mode exposes them; they must not be used as input features, provenance flags, or pseudo-label generators.
- For `lc0_static_112` and `lc0_bt4_112`, deterministic geometry may be generated only from verified current-board piece planes. History channels may pass through learned neural adapters, but must not be interpreted by the deterministic transport adapter unless the channel semantics are explicitly known. Unknown 112-channel layouts must fail closed.

Clarified boundary for this packet: the first experiment uses `simple_18` only because the model's deterministic transport branch needs reliable current-board piece-plane semantics. The neural stem can technically accept `lc0_static_112` or `lc0_bt4_112`, but the transport branch must refuse to construct piece measures unless Codex can map the current-board piece planes from existing repository metadata.

## 4. Research Map

The external research map is about the mathematical operator, not about copying a chess-engine architecture.

| Source | URL | Borrowed | Not copied |
|---|---|---|---|
| Marco Cuturi, “Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances,” NeurIPS 2013 / arXiv 1306.0895 | https://arxiv.org/abs/1306.0895 | Entropic regularization of discrete optimal transport and efficient Sinkhorn-Knopp scaling. | No MNIST retrieval setup, no image-distance benchmark, no claim that Sinkhorn itself solves chess. |
| Gabriel Peyré and Marco Cuturi, “Computational Optimal Transport,” Foundations and Trends / arXiv 1803.00567 | https://arxiv.org/abs/1803.00567 and https://optimaltransport.github.io/book/ | Numerical OT definitions, discrete couplings, Sinkhorn implementation cautions, and the distinction between transport distance and arbitrary learned similarity. | No copying of code, datasets, or application-specific examples. |
| Sander et al., “Sinkformers: Transformers with Doubly Stochastic Attention,” AISTATS 2022 | https://proceedings.mlr.press/v151/sander22a.html | The useful contrast between row-softmax attention and doubly-stochastic transport-like normalization. | This packet does not propose a Transformer, attention stack, sequence model, or Sinkformer clone. |
| Arjovsky et al., “Invariant Risk Minimization,” arXiv 1907.02893 | https://arxiv.org/abs/1907.02893 | The caution that spurious environment correlations can dominate prediction; used only to motivate nuisance-preserving ablations by material/source-like shortcuts. | No IRM objective is part of the core model; no environment-specific optimizer is required. |
| Alemi et al., “Deep Variational Information Bottleneck,” arXiv 1612.00410 | https://arxiv.org/abs/1612.00410 | The general idea that bottlenecked summaries can improve robustness. | No variational latent posterior, KL objective, or VIB training is required here. |

No citation above is a chess-puzzle paper. The chess-specific claim is new and must be treated as a hypothesis, not as established theory.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN over `simple_18` | `src/chess_nn_playground/models/cnn.py` | Already present and tests generic local pattern learning rather than a new mathematical operator. |
| Residual CNN over `simple_18` | `src/chess_nn_playground/models/residual_cnn.py` | Already present; more residual blocks would be ordinary depth/width scaling. |
| LC0-style CNN or residual CNN over `lc0_bt4_112` | Existing LC0 BT4-style CNN/residual variants | Too close to the current suite and partially mimics an engine input style without a new falsifiable mechanism. |
| Ordinary ViT over 64 square tokens | Common vanilla Transformer baseline | Explicitly disallowed as a core idea and likely to learn unconstrained square correlations rather than rule-safe chess geometry. |
| Plain GNN on squares or pieces | Generic graph neural network baseline | Too generic, and a piece/square graph with learned messages risks becoming another static relation model without a distinct operator. |
| Hyperparameter tuning of current models | All existing baselines | Not a research idea; it would not explain a distinct inductive bias. |
| Ensembling CNNs, residual CNNs, and LC0-style models | Any existing model family | Explicitly disallowed and would hide mechanism failure behind variance reduction. |
| Tactical attack-defense sheaf, Hodge, Laplacian, curvature, or tension model | Imported tactical sheaf/Hodge packets | Already researched; this packet must not repeat attack-incidence sheaf terminology with different edge labels. |
| One-ply move-delta set, DeepSets, attention, entropy, spectrum, or landscape model | Imported counterfactual move-delta packets | Already researched and explicitly excluded unless the operator is mathematically different. |
| Stockfish-score or PV-conditioned classifier | None allowed | Leaky by construction; engine outputs and verification metadata are forbidden inputs. |
| Material-count logistic or MLP baseline as the main idea | Could be derived from any encoder | Too shallow and likely to exploit source/dataset artifacts rather than puzzle-likeness. |
| Ordinal-only or evidential-only head over fine labels | Could wrap any existing CNN | Potentially useful for calibration, but as a standalone idea it is mostly a loss/head change rather than a new chess-position operator. |
| Full legal move count / checkmate oracle features | No safe baseline | Rule-only in a narrow sense, but too close to move-tree consequences and must not be central without heavy leakage justification. |

## 6. Mathematical Thesis

### Input space definition

Let `Omega = {0, ..., 7} x {0, ..., 7}` be the 64 board squares. For the first experiment, an input is a tensor

```text
x in X_18 subset {0,1}^{18 x 8 x 8}
```

with 12 mutually exclusive piece planes plus side-to-move, castling, and en-passant planes as provided by `simple_18`. The model constructs a deterministic side-to-move canonical representation `c(x)` in which:

- the player to move is called `us`,
- the opponent is called `them`,
- if black is to move, colors are swapped and ranks are vertically flipped so `us` pawns always move toward increasing canonical rank,
- files are not mirrored by default,
- castling and en-passant channels are not used by the deterministic transport branch in the first experiment.

This is not a full chess symmetry claim. It is only a naming/orientation convention.

### Label/target definition

Let `F in {0,1,2}` be the fine label and `Y = 1[F > 0]` be the coarse binary target. The model trains on `Y` in `mode: coarse_binary`. The fine label `F` is reserved for diagnostics such as the `3x2` matrix and near-puzzle recall.

### Data distribution assumptions

Assume samples are drawn from an empirical distribution `P(X,F,Y)`. The working assumptions are:

1. Puzzle-like positions are more likely when the side-to-move has geometrically concentrated force that can be matched to opponent king-safety or high-value target anchors at low empty-board chess transport cost.
2. Non-puzzles are more likely to have either diffuse force, irrelevant force, or force concentrated away from tactical targets.
3. Material balance, source provenance, and position phase may correlate with labels but are not stable causal definitions of puzzle-likeness.
4. The current split is the only valid data source for the minimal experiment; the full 45M-row Parquet is out of scope until streaming exists.

These assumptions may be false. The ablations are designed to reveal that.

### Allowed symmetry or equivariance assumptions

The only built-in equivariance is side-to-move color canonicalization. The model does **not** assume full rotation/reflection invariance because pawns, castling rights, en-passant, and side-to-move break most board symmetries. Horizontal file reflection is not part of the main model; it may be used only as an optional diagnostic augmentation if castling/en-passant semantics are handled correctly.

### Core hypothesis

Define piece-source measures `mu_g(x)` and target-anchor measures `nu_a(x)` on `Omega`. The hypothesis is that the binary label has useful conditional dependence on a low-dimensional collection of entropic transport observables

```text
Tau_theta(x) = { T_epsilon(mu_g(x), nu_a(x); C_g),
                 E_{mu_g otimes nu_a}[C_g] - T_epsilon(mu_g(x), nu_a(x); C_g),
                 H(Pi^*_{g,a}),
                 barycentric displacement(Pi^*_{g,a}) }_{(g,a) in Pairs}
```

where `C_g` is a learned nonnegative mixture of fixed chess-distance matrices, and `Pi^*_{g,a}` is the entropic optimal coupling. The key claim is not that OT proves tactical truth; it is that constrained mass matching is a useful bottleneck between raw piece locations and puzzle-likeness.

### Formal object introduced

For `n = 64`, let `Delta_n` be the probability simplex over squares. For each source group `g`, define a source measure

```text
mu_g(x) in Delta_n
```

supported mostly or entirely on occupied squares of a canonical piece group, such as `us_sliders`, `us_leapers`, `us_pawns`, `them_sliders`, `them_leapers`, and `them_pawns`. A small learned salience head may reweight occupied squares, but it must be masked by current-board occupancy.

For each target anchor `a`, define a target measure

```text
nu_a(x) in Delta_n
```

using only deterministic current-board geometry. The first target set is:

- `them_king_zone`: soft Chebyshev ball around the opponent king,
- `them_value`: opponent non-king pieces weighted by fixed nominal values `Q=9, R=5, B/N=3, P=1`, with fallback to opponent king zone if empty,
- `us_king_zone`: soft Chebyshev ball around our king,
- `us_value`: our non-king pieces weighted by the same nominal values,
- `us_promotion_rank`: uniform distribution on canonical promotion rank for `us`,
- `them_promotion_rank`: uniform distribution on canonical promotion rank for `them`.

The pair list should be fixed and ordered. Recommended first pair list length `P = 12`:

```text
us_sliders -> them_king_zone
us_leapers -> them_king_zone
us_pawns   -> them_king_zone
us_sliders -> them_value
us_leapers -> them_value
us_pawns   -> them_value
them_sliders -> us_king_zone
them_leapers -> us_king_zone
them_pawns   -> us_king_zone
them_sliders -> us_value
us_pawns     -> us_promotion_rank
them_pawns   -> them_promotion_rank
```

For each source group `g`, define a cost matrix

```text
C_g = softplus(beta_g0 + sum_{r=1}^R alpha_{g,r} D_r) in R_+^{64 x 64}
```

where the fixed metric bank is:

- `D_king`: normalized Chebyshev distance,
- `D_manhattan`: normalized Manhattan distance,
- `D_rook`: normalized empty-board rook-move distance, capped at 2,
- `D_bishop`: normalized empty-board bishop-move distance with opposite-color cap,
- `D_knight`: normalized shortest knight-move distance on an empty board,
- `D_pawn_us`: normalized forward-and-file pawn-oriented distance in canonical orientation,
- `D_pawn_them`: same metric in the opposite orientation.

For `epsilon > 0`, define the entropic transport value

```text
T_epsilon(mu, nu; C) = min_{Pi >= 0, Pi 1 = mu, Pi^T 1 = nu}
                       <Pi, C> + epsilon sum_{i,j} Pi_{ij} (log Pi_{ij} - 1).
```

The minimizer is denoted `Pi^*_epsilon(mu,nu;C)`.

### Proposition: product-plan gap isolates coupled geometry

For any `mu, nu in Delta_n` and nonnegative cost matrix `C`, the independent product plan `Pi_prod = mu otimes nu` is feasible for the unregularized transport polytope. Therefore

```text
T_0(mu,nu;C) <= <mu otimes nu, C>.
```

The gap

```text
G_0(mu,nu;C) = <mu otimes nu, C> - T_0(mu,nu;C) >= 0
```

is zero when the independent product plan is already optimal and positive when a nontrivial coupling can reduce expected cost by matching particular source mass to particular target mass. For small positive `epsilon`, the entropic plan is the unique smooth maximum-entropy perturbation of this matching problem; the analogous regularized gap remains a smooth measure of whether geometric matching adds information beyond the marginals.

### Proof sketch or derivation

The feasible set of transport plans is the set of nonnegative matrices with row sums `mu` and column sums `nu`. The product matrix `mu_i nu_j` has row sum `mu_i sum_j nu_j = mu_i` and column sum `nu_j sum_i mu_i = nu_j`, so it is feasible. Because `T_0` is the minimum of `<Pi,C>` over the feasible set, it cannot exceed the value at `Pi_prod`. Entropic regularization with `epsilon > 0` adds a strictly convex negative-entropy term over the positive interior, yielding a unique differentiable solution under standard Sinkhorn conditions. Sinkhorn iterations compute scaling vectors `u,v` such that `Pi = diag(u) exp(-C/epsilon) diag(v)` has the requested marginals.

### Optimization objective

The main classifier is:

```text
p_theta(Y=1 | x) = softmax_1(MLP_theta([Tau_theta(x), global_stem_pool(x)]))
```

with binary cross-entropy / two-class cross-entropy:

```text
L(theta) = E_{(x,y)} CE(y, logits_theta(x)) + lambda_w ||theta||_2^2.
```

The central mathematical object is `Tau_theta(x)`. The optional `global_stem_pool(x)` is deliberately small and must be present in ablations so the only changed factor is the transport operator.

### What is actually proven

- The product coupling is a feasible transport plan.
- The unregularized OT value is no greater than the independent product expected cost.
- Entropic OT has a smooth, differentiable Sinkhorn form under positive marginals and positive kernel entries.
- The transport summaries are deterministic functions of current-board inputs and learned parameters; they do not require engine analysis, PVs, node counts, or labels as inputs.

### What remains only hypothesized

- That puzzle-likeness in the current dataset is better predicted by low-cost source-target transport and product-plan gaps than by CNN features alone.
- That target anchors such as king zones and value-piece distributions are the right rule-safe proxies for tactical vulnerability.
- That side-to-move canonicalization improves stability more than it removes useful asymmetric information.
- That normalized measures reduce material/provenance shortcuts enough to improve near-puzzle diagnostics.

### Counterexamples where the idea should fail

- Zugzwang, fortress, stalemate, or underpromotion studies where the tactic is legal-move availability rather than geometric force concentration.
- Long forced lines where the current board has no low-cost current-piece-to-target transport signature.
- Puzzles whose key idea is a quiet waiting move, repetition, castling-right subtlety, or en-passant legality.
- Positions where the puzzle label was driven by source curation artifacts or material imbalance rather than board tactics.
- Positions with symmetric transport geometry but different legal constraints due to pins, checks, or occupied paths that the empty-board metric bank cannot see.

## 7. Architecture Specification

### Module names

Add the main model in:

```text
src/chess_nn_playground/models/piece_target_transport_bottleneck.py
```

Recommended classes/functions:

- `EncodingPieceAdapter`
- `SideToMoveCanonicalizer`
- `ChessMetricBank`
- `MaskedMeasureBuilder`
- `EntropicTransportLayer`
- `TransportSummaryHead`
- `PieceTargetTransportBottleneck`
- builder function: `build_piece_target_transport_bottleneck(config_or_kwargs)`

### Forward-pass steps and tensor shapes

Assume `B=batch`, `C=input_channels`, `N=64`, `D=32`, `G=6`, `A=6`, `P=12`, `R=7`, and `F=5` transport features per pair.

1. **Input**

   ```text
   x: [B, C, 8, 8]
   ```

2. **Encoding adapter**

   Extract current-board piece planes and side-to-move.

   ```text
   piece_planes: [B, 12, 8, 8]
   stm: [B]
   aux_planes_for_stem: [B, C, 8, 8]
   ```

   For `simple_18`, this should use the repository's existing channel order or an explicit config channel map. If the map is not available, raise a clear error.

3. **Side-to-move canonicalization**

   Swap colors and vertically flip ranks when black is to move.

   ```text
   canonical_piece_planes: [B, 12, 8, 8]
   canonical_input_for_stem: [B, C, 8, 8]
   ```

   If applying canonicalization to non-piece auxiliary planes is ambiguous for a 112-channel encoding, canonicalize only verified current-board piece planes for the transport branch and feed the original tensor to the neural stem. This is acceptable only if clearly documented.

4. **Small local stem**

   A deliberately small neural stem produces square features.

   ```text
   h = Conv3x3(C,D) -> GELU -> GroupNorm -> Conv3x3(D,D) -> GELU
   h: [B, D, 8, 8]
   h_flat: [B, N, D]
   h_pool: [B, D]
   ```

5. **Source-group masks**

   Construct six canonical source masks from current pieces:

   ```text
   us_sliders   = us queens + rooks + bishops
   us_leapers   = us knights + king
   us_pawns     = us pawns
   them_sliders = them queens + rooks + bishops
   them_leapers = them knights + king
   them_pawns   = them pawns
   source_masks: [B, G, N]
   ```

   The inclusion of kings in `*_leapers` is a pragmatic fallback to avoid empty defensive groups; if this causes obvious artifacts, ablate king inclusion.

6. **Masked learned source measures**

   For each group `g`, compute salience on occupied squares only:

   ```text
   source_logits = Linear_g(h_flat): [B, G, N]
   source_logits[mask == 0] = -large
   mu: [B, G, N]
   ```

   Empty groups use a deterministic uniform fallback and an `empty_group_flag`; the flag may be included only in all corresponding ablations because it is a material shortcut.

7. **Target-anchor measures**

   Build deterministic anchors:

   ```text
   nu: [B, A, N]
   anchors = [them_king_zone, them_value, us_king_zone, us_value,
              us_promotion_rank, them_promotion_rank]
   ```

   Each anchor is normalized with epsilon smoothing. King zones use fixed soft weights such as `exp(-beta * d_chebyshev(square, king_square))` with default `beta=1.0`.

8. **Chess metric bank and learned costs**

   Precompute fixed distance matrices once:

   ```text
   D_bank: [R, N, N]
   cost_weights: [G, R]
   C_g: [G, N, N]
   C_pair: [P, N, N]
   ```

   Cost weights should be constrained by `softplus` or `softmax` to avoid negative-cost loopholes.

9. **Entropic transport per source-target pair**

   Gather pair marginals:

   ```text
   pair_mu: [B, P, N]
   pair_nu: [B, P, N]
   pair_C:  [P, N, N]
   ```

   Run log-domain Sinkhorn for `sinkhorn_iters=8` with `epsilon=0.07` by default. Use pair chunking:

   ```text
   chunk_pairs = 4
   Pi_chunk: [B, chunk_pairs, N, N]
   ```

10. **Transport summaries**

   For each pair, compute:

   ```text
   ot_cost       = sum Pi*C                         [B,P,1]
   prod_cost     = sum (mu outer nu)*C              [B,P,1]
   transport_gap = prod_cost - ot_cost              [B,P,1]
   plan_entropy  = -sum Pi*log(Pi+delta)/log(N*N)   [B,P,1]
   sharpness     = sum Pi^2                         [B,P,1]
   ```

   Concatenate:

   ```text
   tau: [B, P*5]
   ```

   Barycentric displacement may replace `sharpness` if Codex wants exactly directional features, but keep the first experiment to five features per pair for simplicity.

11. **Classifier head**

   ```text
   z = concat(tau, h_pool): [B, P*5 + D] = [B, 92]
   logits = MLP(92 -> 128 -> 2): [B, 2]
   ```

   Return logits only, preserving shared trainer compatibility.

### Parameter-count estimate

For `simple_18`, `D=32`, `P=12`, `F=5`:

- Conv stem: about `18*32*3*3 + 32*32*3*3 = 14,400` weights plus normalization/bias terms.
- Source salience heads: about `G*D = 192` weights plus biases.
- Cost mixture weights: about `G*R = 42` weights plus biases.
- MLP head: `(92*128) + (128*2) = 12,032` weights plus biases.
- Total expected trainable parameters: roughly `28k-35k`, depending on normalization and adapter details.

For `lc0_*_112`, the first convolution becomes `112*32*3*3 = 32,256` weights, so the total remains below roughly `65k`.

### FLOP and complexity estimate

Let `K=sinkhorn_iters`, `N=64`, and `P=12`.

- Conv stem: `O(B*C*D*64*9 + B*D^2*64*9)`.
- Source/target construction: `O(B*(G+A)*N)`.
- Sinkhorn: `O(B*P*K*N^2)`. With `B=512`, `P=12`, `K=8`, `N=64`, this is about `201M` matrix-scale operations per batch before chunking overhead.
- Summary extraction: `O(B*P*N^2)`.

This is heavier than a tiny CNN but still small compared with deep residual models because `N=64` is fixed.

### Candidate-set memory and chunking plan

This model does not generate legal moves or move-delta candidates. Its generated structured set is the fixed source-target pair set of length `P=12`.

The largest tensor is the transport plan:

```text
memory_bytes ~= B * chunk_pairs * N * N * bytes_per_float
```

For `B=512`, `chunk_pairs=4`, `N=64`, `float32`, one plan chunk is:

```text
512 * 4 * 64 * 64 * 4 = 33,554,432 bytes ~= 32 MiB
```

Autograd may multiply this by roughly `2-4x`. Keep `transport_chunk_pairs=4` by default; reduce to `2` if GPU memory is tight. A CPU-safe implementation should permit `batch_size=128` for tests.

### Required config fields

Recommended custom config fields under `model`:

```yaml
name: piece_target_transport_bottleneck
input_channels: 18
num_classes: 2
transport_dim: 32
source_groups: [us_sliders, us_leapers, us_pawns, them_sliders, them_leapers, them_pawns]
target_anchors: [them_king_zone, them_value, us_king_zone, us_value, us_promotion_rank, them_promotion_rank]
transport_pairs: default_12
sinkhorn_epsilon: 0.07
sinkhorn_iters: 8
transport_chunk_pairs: 4
metric_bank: [king, manhattan, rook, bishop, knight, pawn_us, pawn_them]
canonicalize_side_to_move: true
include_global_stem_pool: true
product_coupling_ablation: false
permuted_cost_ablation: false
encoding_channel_map: null
fail_closed_unknown_semantics: true
```

### Encoding support and fail-closed assumptions

- `simple_18`: first experiment target. The adapter may use the repository's existing `simple_18` metadata. If no metadata exists, Codex should define an explicit channel-map config and tests using synthetic positions.
- `lc0_static_112`: supported only if Codex can identify the current-board 12 piece planes. The deterministic transport branch uses only those current-board planes. Other channels may enter only the small learned stem.
- `lc0_bt4_112`: same as `lc0_static_112`; zero-filled or future history channels must not be used for deterministic geometry unless they are explicitly current-board planes. History channels may be consumed by the learned stem, but the first benchmark should avoid this to isolate the transport idea.
- Unknown channel semantics: raise `ValueError` with a message such as `PieceTargetTransportBottleneck requires verified current-board piece-plane mapping for deterministic transport branch`.

### Pseudocode, not final implementation

```python
class PieceTargetTransportBottleneck(nn.Module):
    def forward(self, x):
        piece_planes, stm = self.adapter.extract_piece_planes_and_stm(x)
        pieces_c, x_stem = self.canonicalizer(piece_planes, stm, x)

        h = self.stem(x_stem)                    # [B,D,8,8]
        h_flat = h.flatten(2).transpose(1, 2)    # [B,64,D]
        h_pool = h.mean(dim=(2, 3))              # [B,D]

        source_masks = build_source_masks(pieces_c)        # [B,G,64]
        mu = self.measure_builder(h_flat, source_masks)    # [B,G,64]
        nu = build_target_anchors(pieces_c)                # [B,A,64]

        costs = self.metric_bank.make_costs(self.cost_weights)  # [G,64,64]
        pair_mu, pair_nu, pair_cost = gather_pairs(mu, nu, costs, self.pairs)

        if self.product_coupling_ablation:
            summaries = product_coupling_summaries(pair_mu, pair_nu, pair_cost)
        else:
            summaries = self.transport(pair_mu, pair_nu, pair_cost)

        tau = summaries.flatten(start_dim=1)     # [B,P*5]
        z = torch.cat([tau, h_pool], dim=1)      # [B,P*5+D]
        return self.head(z)                      # [B,2]
```

## 8. Loss, Training, And Regularization

- Primary loss: standard two-class cross-entropy on the coarse binary target `Y`.
- Auxiliary loss: none required for the first experiment. Do not add fine-label ordinal training in the first pass; keep fine labels for diagnostics to avoid confounding the transport hypothesis.
- Optional regularizer only if needed for numerical stability: small entropy floor on learned source measures, e.g. penalize `max(0, H_min - H(mu_g))` with `lambda <= 1e-3`. This must be off by default or separately ablated.
- Class weighting: use the existing `balanced` class weighting setting from benchmark configs.
- Batch size expectations: default `512` on GPU with `transport_chunk_pairs=4`; use `128` for CPU smoke tests.
- Optimizer: AdamW.
- Learning rate: `0.001` for the 3-epoch minimal run.
- Weight decay: `0.0001`.
- Epochs: `3` for the minimal experiment; scale only after central ablations are informative.
- Mixed precision: keep `false` for the first deterministic comparison; Sinkhorn in half precision can create avoidable numerical instability.
- Determinism requirements: fixed seed `42`, deterministic PyTorch flags as in existing benchmark, fixed pair ordering, fixed metric-bank construction, fixed square-index order, deterministic cost-permutation ablation seed.
- Numerical safeguards: clamp/smooth marginals with `marginal_eps=1e-6`; log-domain Sinkhorn preferred; assert finite logits and finite transport summaries.
- What must stay unchanged for fair comparison: split paths, binary target mapping, evaluation scripts, threshold selection rules, class weighting policy, seed, number of epochs, data loader behavior, and reporting format including the `3x2` diagnostic matrix.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Product coupling replacement | Replaces `Pi^*` with `mu otimes nu` while preserving source marginals, target marginals, material, anchors, pair list, feature dimensions, and MLP | The central claim that optimal mass matching adds information beyond independent source-target expectations | If performance is unchanged, the OT coupling is unnecessary; abandon or radically revise the operator. |
| Fixed random square permutation of cost matrices | Applies one deterministic random permutation to rows/columns of each `D_r`, preserving the cost-value histogram and tensor shapes while destroying chess-square semantics | Chess geometry, not just a numeric bottleneck, matters | If unchanged, the model is using generic regularization or target/marginal shortcuts rather than chess metric structure. |
| Anchor semantics shuffle | Replaces king/value/promotion anchors with entropy-matched random target distributions per sample, preserving `A`, `P`, target entropy, and normalization | Target identity and current-board anchor semantics matter | If unchanged, target construction is not meaningful and the MLP/stem is doing the work. |
| Transport removed, tiny stem only | Drops all transport summaries and keeps the same small stem plus MLP capacity adjusted to match parameters | The transport branch contributes beyond a tiny CNN bottleneck | If tiny stem matches main model, the OT branch is not useful. |
| Row-softmax attention replacement | Uses row-normalized source-to-target soft attention with no column marginal constraint | Doubly-stochastic/mass-conserving coupling matters rather than generic attention | If row-softmax matches, the conservation constraint is not adding signal. |
| Cost bank fixed to Chebyshev only | Removes learned metric mixture and all piece-metric distinctions | Piece-specific chess metric mixtures matter | If unchanged, simpler king-distance geometry may be enough; keep simpler form if main still wins. |
| No side-to-move canonicalization | Uses raw colors/orientation while keeping everything else fixed | Canonical `us/them` orientation helps separate tactical role from color artifacts | If no-canonicalization wins, canonicalization may be erasing useful pawn/castling asymmetry or adapter is buggy. |
| Material/count-only nuisance baseline | Build a separate small MLP using only deterministic piece counts, side-to-move, castling/en-passant presence, and phase proxies | Main model is not merely rediscovering material/source shortcuts | If count-only is close to main, dataset artifacts dominate and the transport result is weak. |
| Source-location shuffle preserving counts | Within each sample and piece group, randomly permute occupied source squares while preserving group counts and target anchors | Actual source geometry matters beyond material counts and target distributions | If unchanged, source measures are not being used semantically. |
| Empty-group flag removed | Removes or masks any explicit empty-group indicators while preserving fallback distributions | Empty group handling is not a hidden material shortcut | If performance collapses only because flags are removed, the model may rely on material presence rather than transport. |

The smallest central falsification is the **product coupling replacement**. It is semantics-preserving for obvious shortcuts such as material presence, source/target marginal distributions, target entropy, and pair count, but it destroys optimal coupling semantics.

Because this model does not enumerate legal move candidates, move-set count-only ablations are not applicable. The nuisance-preserving analogues are the product-coupling, cost-permutation, source-location-shuffle, and material/count-only ablations above.

## 10. Benchmark And Falsification Criteria

Codex should benchmark the main model and central ablations using the existing shared trainer and reports.

Baselines to compare against:

- existing `simple_18` simple CNN under `src/chess_nn_playground/models/cnn.py`, using the same split and training budget,
- existing `simple_18` residual CNN under `src/chess_nn_playground/models/residual_cnn.py`, using the same split and training budget,
- if already configured in the repo, small/medium variants that are normally part of the leaderboard,
- do not compare the first `simple_18` experiment against `lc0_bt4_112` results as the primary success claim; report them separately if available.

Metrics to inspect:

- validation and test cross-entropy,
- accuracy,
- balanced accuracy,
- ROC-AUC,
- PR-AUC for class `1` puzzle-like,
- macro F1,
- Brier score or ECE if the report stack already supports calibration,
- confusion matrix for binary labels,
- required rectangular `3x2` diagnostic matrix for fine labels `0/1/2 -> predicted 0/1`.

Near-puzzle diagnostic:

- Select a probability threshold on validation that matches the best baseline's fine-label-`0` false-positive rate.
- At that matched false-positive rate, report fine-label-`1` recall on validation and test.
- Also report mean predicted puzzle probability by fine label: `mean p(y=1 | F=0)`, `mean p(y=1 | F=1)`, `mean p(y=1 | F=2)`. A useful model should usually satisfy `F0 < F1 < F2`; violations are not automatic failure but must be discussed.

Required artifacts:

- trained model config,
- checkpoint path,
- validation and test metrics JSON/CSV,
- predictions Parquet or CSV with at least target, fine label, predicted probability, and predicted class,
- `3x2` diagnostic matrix for main model and every central ablation,
- ablation comparison table,
- report markdown using the idea-specific template.

Success threshold:

- Main model beats the best same-encoding baseline by at least one of:
  - `+0.015` absolute PR-AUC on test, or
  - `+0.010` absolute ROC-AUC on test, or
  - `+0.020` absolute fine-label-`1` recall at matched fine-label-`0` false-positive rate,
- and the product-coupling ablation is worse than the main model by at least `0.010` PR-AUC or `0.015` matched-FPR fine-label-`1` recall.

Failure threshold:

- Main model is within `±0.005` PR-AUC of the best same-encoding baseline and the product-coupling and permuted-cost ablations are also within `±0.005` PR-AUC of the main model.

What result would make us abandon the idea:

- Product coupling, cost permutation, or anchor shuffle matches the main model across test PR-AUC, ROC-AUC, and fine-label-`1` matched-FPR recall, especially if the material/count-only nuisance baseline is close. That would show the proposed transport semantics are not doing useful work.

What result would justify scaling:

- Main model clears the success threshold, the product-coupling ablation drops meaningfully, the cost-permutation ablation drops meaningfully, and near-puzzle recall improves without a larger fine-label-`0` false-positive rate. Then Codex should try longer training, `lc0_static_112` with verified current-board planes, and possibly a slightly richer target-anchor set.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_transport_bottleneck/idea.yaml` | Create | Machine-readable summary of this idea, including status, slug, config path, model path, and central falsification ablation. |
| `ideas/20260421_transport_bottleneck/math_thesis.md` | Create | Section 6 math thesis, including definitions of `mu`, `nu`, `C`, `T_epsilon`, product-plan gap, proof sketch, and counterexamples. |
| `ideas/20260421_transport_bottleneck/architecture.md` | Create | Section 7 architecture details, tensor shapes, pseudocode, memory formulas, adapter assumptions, and fail-closed rules. |
| `ideas/20260421_transport_bottleneck/implementation_notes.md` | Create | Practical notes for log-domain Sinkhorn, metric-bank construction, side-to-move canonicalization tests, finite-value asserts, and CPU smoke-test settings. |
| `ideas/20260421_transport_bottleneck/trainer_notes.md` | Create | Training defaults, fair-comparison constraints, class weighting, determinism, and reporting requirements. |
| `ideas/20260421_transport_bottleneck/ablations.md` | Create | The full ablation table from Section 9 plus exact config flags for each ablation. |
| `ideas/20260421_transport_bottleneck/train.py` | Create | Thin idea-local entrypoint that calls the shared trainer with `configs/piece_target_transport_bottleneck_simple18.yaml`; do not fork trainer logic unless unavoidable. |
| `ideas/20260421_transport_bottleneck/config.yaml` | Create | Idea-local copy of the minimal config for archival reproducibility. |
| `ideas/20260421_transport_bottleneck/report_template.md` | Create | Report template requiring metrics, `3x2` matrices, near-puzzle matched-FPR diagnostic, ablation table, and failure/success conclusion. |
| `ideas/chatgpt_pro_deep_math_research_prompt.md` | Update | Preserve hard constraints and add anti-duplicate guidance for transport-bottleneck families after this packet is consumed. See Section 13. |
| `src/chess_nn_playground/models/piece_target_transport_bottleneck.py` | Create | Implement `EncodingPieceAdapter`, `SideToMoveCanonicalizer`, `ChessMetricBank`, `MaskedMeasureBuilder`, `EntropicTransportLayer`, `TransportSummaryHead`, and `PieceTargetTransportBottleneck`. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `piece_target_transport_bottleneck` builder while preserving existing model names. |
| `configs/piece_target_transport_bottleneck_simple18.yaml` | Create | Minimal benchmark config using `simple_18`, `batch_size=512`, `epochs=3`, `balanced` class weighting, and model-specific transport fields. |
| `configs/piece_target_transport_bottleneck_product_ablation_simple18.yaml` | Create | Same as main config but `product_coupling_ablation: true`. |
| `configs/piece_target_transport_bottleneck_permuted_cost_simple18.yaml` | Create | Same as main config but `permuted_cost_ablation: true` with fixed seed. |
| `configs/piece_target_transport_bottleneck_anchor_shuffle_simple18.yaml` | Create | Same as main config but target anchors are entropy-matched shuffled anchors. |
| `tests/test_piece_target_transport_bottleneck.py` | Create | Focused unit tests: output shape `[B,2]`, finite logits, simple synthetic board adapter extraction, canonicalization behavior, marginal sums equal one, Sinkhorn plan row/column sums near requested marginals, fail-closed unknown 112-channel semantics. |
| `tests/test_chess_metric_bank.py` | Create if needed | Verify fixed metric matrices are deterministic, finite, normalized, and have expected zero diagonals. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0507_tuesday_los_angeles_transport_bottleneck.md
  generated_at: '2026-04-21T05:07:36-07:00'
  weekday: tuesday
  timezone: America/Los_Angeles
  idea_slug: transport_bottleneck
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_transport_bottleneck
  name: Entropic Piece-Target Transport Bottleneck
  slug: transport_bottleneck
  status: draft
  created_at: '2026-04-21T05:07:36-07:00'
  author: ChatGPT Pro
  short_thesis: Puzzle-like positions can be detected by low-cost entropic transport between normalized current-board piece-source measures and deterministic king/value/promotion target anchors.
  novelty_claim: Uses chess-metric optimal transport summaries rather than CNN scaling, static attack/sheaf incidence, or one-ply move-delta pooling.
  expected_advantage: Better near-puzzle recall at matched non-puzzle false-positive rate if geometric force-to-target coupling is a stable signal.
  central_falsification_ablation: Replace Sinkhorn optimal coupling with independent product coupling while preserving marginals and downstream capacity.
  target_task: coarse_binary
  input_representation: simple_18 first; lc0_static_112/lc0_bt4_112 only with verified current-board piece-plane mapping
  output_heads: two-class logits only
  compute_notes: Sinkhorn over 12 source-target pairs on 64 squares; default 8 iterations and pair chunking of 4.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/piece_target_transport_bottleneck_simple18.yaml
  model_path: src/chess_nn_playground/models/piece_target_transport_bottleneck.py
  latest_result_path: null
  notes: Do not use engine features, legal move counts, attack-sheaf operators, or one-ply move-delta sets.
```

```yaml
config_yaml:
  run:
    name: piece_target_transport_bottleneck_simple18
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
    name: piece_target_transport_bottleneck
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
  model_name: piece_target_transport_bottleneck
  file_path: src/chess_nn_playground/models/piece_target_transport_bottleneck.py
  builder_function: build_piece_target_transport_bottleneck
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - EncodingPieceAdapter
    - SideToMoveCanonicalizer
    - ChessMetricBank
    - MaskedMeasureBuilder
    - EntropicTransportLayer
    - TransportSummaryHead
    - PieceTargetTransportBottleneck
  required_config_fields:
    - model.name
    - model.input_channels
    - model.num_classes
    - model.sinkhorn_epsilon
    - model.sinkhorn_iters
    - model.transport_chunk_pairs
    - model.canonicalize_side_to_move
    - model.fail_closed_unknown_semantics
  expected_parameter_count: 28000-65000 depending on input_channels
  expected_memory_notes: Main transport plan chunk uses about B*chunk_pairs*64*64*bytes; with B=512, chunk_pairs=4, fp32 this is about 32 MiB before autograd overhead.
```

```yaml
research_continuity:
  idea_fingerprint: current-board normalized piece/source and king/value/promotion target measures + chess-metric entropic OT/Sinkhorn summaries + binary puzzle-likeness + no attack/sheaf/Hodge and no one-ply move-delta set
  already_researched_family_overlap: Low overlap with imported sheaf/Hodge and move-delta families; it uses current-board geometry but not attack incidence, legal move deltas, or sheaf restrictions.
  closest_duplicate_risk: Could be mistaken for attention over squares; distinguish by constrained marginals, fixed chess metric bank, product-plan gap, and cost-permutation/product-coupling falsification.
  do_not_repeat_if_this_fails:
    - Another piece-target OT bottleneck with more anchors but the same Sinkhorn/product-gap mechanism.
    - A renamed doubly-stochastic square attention model using the same source-target marginals.
    - A larger CNN stem wrapped around the same transport summaries as the claimed novelty.
    - A transport model whose only change is more Sinkhorn iterations, more target anchors, or larger hidden dimension.
  suggested_next_search_directions:
    - Label-safe selective prediction for near-puzzle ambiguity.
    - Causal invariance across material phase and source-like environments without transport or move enumeration.
    - Calibration methods that expose fine-label 1 uncertainty without fabricating labels.
    - Rule-safe latent bottlenecks based on board occupancy compression rather than attack graphs, move deltas, or transport couplings.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Entropic Piece-Target Transport Bottleneck` to the imported research memory after implementation. | Prevents the next research pass from proposing the same OT/Sinkhorn source-target coupling with more anchors or a larger stem. | `Imported Research Memory` |
| Add an anti-duplicate fingerprint for transport-bottleneck packets: `current-board piece/source measures + deterministic king/value/promotion anchors + chess-metric Sinkhorn/OT coupling + transport cost/gap/entropy pooling`. | Makes it clear that future ideas must change the central operator, not only add target types or tune Sinkhorn iterations. | `Research Continuity` / anti-duplicate rules |
| Require transport-family proposals to include product-coupling and cost-permutation ablations. | These are the cleanest falsification tests for whether optimal coupling and chess metric semantics matter. | `Required Markdown File Content`, Section 9 guidance |
| Add guidance that empty-board chess distance matrices are allowed only when they do not enumerate legal moves or successor positions. | Clarifies a useful safe boundary between deterministic rule geometry and move-tree leakage. | `Problem Restatement And Data Contract` leakage clarification |
| If this idea fails, discourage follow-ups that merely add more OT anchors, low-rank OT, sliced OT, or Sinkhorn attention without a new falsifiable claim. | Avoids wasting cycles on variants likely to fail for the same reason. | `What Counts As Creative Enough` and `Imported Research Memory` |
| If this idea succeeds, ask the next pass to focus on why near-puzzle class `1` improved or failed, not just leaderboard score. | Keeps the research loop tied to the ambiguous near-puzzle diagnostic instead of generic accuracy. | `Benchmark And Falsification Criteria` |

Do not weaken leakage rules, label rules, falsification requirements, or anti-duplicate requirements.

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0507_tuesday_los_angeles_transport_bottleneck.md`
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes, `simple_18` on the existing `crtk_sample_3class` split
- Falsification criterion is concrete: yes, product-coupling replacement plus cost-permutation and anchor-shuffle ablations
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported research packets completed: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
