# Codex Handoff Packet: Color-Flip Orbit Evidence Bottleneck

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0751_tuesday_los_angeles_color_flip_orbit.md`
- Generated at: 2026-04-21 07:51:27 America/Los_Angeles
- Weekday: Tuesday
- Timezone: America/Los_Angeles
- Idea slug: `color_flip_orbit`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Color-Flip Orbit Evidence Bottleneck, abbreviated `CFOEB`
- One-sentence thesis: A chess position is puzzle-like only if its evidence survives the exact color-flip/rank-reflection symmetry of chess, so the classifier should intersect evidence from a board and its color-flipped orbit mate rather than trust orientation-specific shortcuts.
- Idea fingerprint: `exact color-flip orbit tau over current-board planes + shared small CNN encoder + harmonic evidential intersection of original/flipped views + optional orbit-risk/embedding consistency regularization + binary puzzle-likeness logits`.
- Why this is not a common CNN/ResNet/Transformer variant: The core operator is not extra depth, width, attention, or a generic augmentation; it is a hard two-element chess-law orbit and a symmetric evidence-intersection head that makes the final prediction exactly invariant under color flipping while penalizing one-view-only evidence.
- Current-data minimal experiment: Use `simple_18` only, implement a fail-closed semantic adapter for the 18 channels, train `CFOEB` on `data/splits/crtk_sample_3class/split_train.parquet`, validate on `split_val.parquet`, test on `split_test.parquet`, and compare against parameter-matched simple/residual CNN baselines under the existing coarse-binary trainer and 3x2 fine-label diagnostic.
- Smallest central falsification ablation: Replace the exact color-flip transform `tau` with a fixed, chess-invalid but material/count-preserving rank permutation plus color swap, keep the same two-view architecture and evidence-intersection head, and require the true `tau` model to beat this ablation on test AUROC and fine-label-1 recall at matched fine-label-0 false-positive rate.
- Expected information gain if it fails: A failure would say that exact color-flip invariance and orbit-intersected evidence are not the missing inductive bias for this split; next cycles should not repeat color-flip orbit intersections and should instead focus on ambiguity calibration, source-shift diagnostics, or non-orbit compression mechanisms.

## 3. Problem Restatement And Data Contract

The task is chess puzzle-likeness classification from a single board position.

- Coarse binary output:
  - output `0`: non-puzzle
  - output `1`: puzzle-like
- Available fine labels:
  - fine label `0`: known non-puzzle
  - fine label `1`: verified near-puzzle
  - fine label `2`: verified puzzle
- Binary training target for the default experiment:
  - `y = 0` for fine label `0`
  - `y = 1` for fine labels `1` and `2`
- Diagnostic report required for every main run and central ablation:
  - rectangular `3x2` matrix: true fine label `0/1/2` by predicted binary output `0/1`
- Input tensor contract:
  - model input: `(batch, C, 8, 8)`
  - model output logits: `(batch, 2)`
- Current encodings:
  - `simple_18`: 12 current piece planes + side-to-move + castling + en-passant
  - `lc0_static_112`
  - `lc0_bt4_112`: LC0-style 112-plane BT4 from one FEN; unavailable history planes are zero-filled until exporter support exists
- Benchmark split:
  - train: `data/splits/crtk_sample_3class/split_train.parquet`
  - validation: `data/splits/crtk_sample_3class/split_val.parquet`
  - test: `data/splits/crtk_sample_3class/split_test.parquet`
- Do not point the current trainer at the roughly 45M-row full Parquet file until streaming support exists.

Leakage checklist:

- Allowed as neural-network inputs:
  - deterministic board coordinates
  - piece occupancy from the chosen encoding
  - side-to-move, castling, and en-passant planes already present in the encoding
  - deterministic current-board transforms derived only from those planes
  - pseudo-legal attack geometry derived only from the current board, although this idea does not use attack geometry
- Forbidden as neural-network inputs:
  - Stockfish scores
  - principal variations
  - node counts
  - mate scores
  - verification metadata
  - source labels
  - proposed labels
  - dataset provenance
  - unresolved candidate-pool status treated as verified label information
- Full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences are leakage-prone unless explicitly justified as rule-only, label-independent, engine-free, and separately ablated. This idea intentionally does not generate legal moves or move trees.
- Safe rule-derived feature boundary:
  - Deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and current-board pseudo-legal geometry are allowed.
  - Engine evaluation and verification artifacts are never allowed.
- `lc0_static_112` and `lc0_bt4_112` boundary:
  - Current-board channels with known semantics may be used for deterministic color-flip geometry.
  - History channels should be consumed only by learned neural adapters unless their channel semantics are known and their transform is implemented and tested.
  - The first experiment should use `simple_18` because it is the least ambiguous encoding for a strict color-flip adapter.

## 4. Research Map

External sources checked and used:

| Source | Verified URL | What is borrowed | What is not copied |
|---|---|---|---|
| Chessprogramming Wiki, "Color Flipping" | https://www.chessprogramming.org/Color_Flipping | The exact chess-engine convention that color flipping vertically mirrors ranks, swaps piece colors, side to move, castling rights, and en-passant rank. | No engine evaluation, search, move generation, bitboard implementation, or evaluation features are used. |
| Chessprogramming Wiki, "Flipping, Mirroring and Rotating" | https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating | Terminology separating rank flips, file mirrors, and rotations on bitboards. | No file-mirror symmetry is adopted as a model assumption; this avoids the imported file-mirror sheaf family. |
| Cohen and Welling, "Group Equivariant Convolutional Networks" | https://proceedings.mlr.press/v48/cohenc16.html | The general principle that exploiting valid symmetries can reduce sample complexity. | No full group convolution, no dihedral symmetry, no rotation/reflection blanket invariance. |
| Arjovsky et al., "Invariant Risk Minimization" | https://arxiv.org/abs/1907.02893 | The causal idea that stable predictors should remain valid across environments. | No source-domain labels are used, and no IRM claim is made about discovering true causal variables from arbitrary environments. |
| Krueger et al., "Out-of-Distribution Generalization via Risk Extrapolation" | https://proceedings.mlr.press/v139/krueger21a.html | The risk-variance penalty idea for discouraging environment-specific solutions. | Risk environments are generated only by a deterministic rule orbit, not by dataset provenance. |
| Zbontar et al., "Barlow Twins" | https://proceedings.mlr.press/v139/zbontar21a.html | The view-agreement plus redundancy-reduction intuition for paired views. | No unlabeled pretraining pipeline or generic image distortions are copied. |
| Bardes et al., "VICReg" | https://arxiv.org/abs/2105.04906 | The optional variance/covariance anti-collapse regularizer for embeddings. | The central objective is supervised binary classification with an orbit evidence head, not self-supervised representation learning. |
| Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty" | https://arxiv.org/abs/1806.01768 | Nonnegative class evidence as a useful output parameterization. | The full subjective-logic loss is not required; the shared trainer still receives ordinary logits. |

Candidate search trace:

| Candidate mechanism considered | Why it was serious | Why it lost to `CFOEB` |
|---|---|---|
| Multi-encoding IRM across `simple_18`, `lc0_static_112`, and `lc0_bt4_112` | It directly targets encoding artifacts and source-style shortcuts. | Current LC0 history semantics and adapter behavior could become the experiment instead of the idea; it is better as a later stress test after a simple orbit model works. |
| Evidential selective classifier for near-puzzle ambiguity | Fine label `1` is intrinsically ambiguous and calibration may matter. | By itself it lacks a chess-specific operator and risks being just a new loss/head. `CFOEB` keeps evidential uncertainty but anchors it to a rule orbit. |
| Masked denoising/MDL motif compressor | Puzzle-like positions may have short tactical descriptions. | It is adjacent to the imported pseudo-likelihood/description-length packet and has a higher implementation burden. |
| Source-artifact adversarial training using inferred provenance proxies | It attacks a plausible failure mode: memorizing source artifacts. | It risks relying on dataset provenance or source labels, which must not enter the model. |
| Full D4 group-equivariant board CNN | It is mathematically clean for many board games. | Chess is not D4-invariant because pawn direction, side to move, castling, and initial king/queen files matter. |
| Side-to-move canonicalization only | It is common in chess neural networks and cheap. | It is likely too ordinary and does not provide a falsifiable two-view operator. |
| Token transformer with color-flip augmentation | It could learn long-range relations. | Ordinary 64-square ViT variants are explicitly disallowed as the core idea. |
| Static attack-map spectral compression | Tactical structure may appear in spectra. | This is too close to imported attack-defense graph/sheaf/Hodge families. |
| One-ply legal-move branching entropy | Puzzles often have forcing move structure. | It is move-set/move-delta adjacent and legal-move counts are leakage-prone without heavy controls. |
| Sinkhorn alignment between original and color-flipped piece measures | It could measure orbit mismatch geometrically. | It would be an OT/transport variant, already heavily represented. |
| Sparse orbit rationale over occupied pieces | It could expose a minimal symmetric witness. | It is too close to the imported sparse witness-piece bottleneck. |
| High-order orbit-polynomial invariants | It could capture tactical constellations invariantly. | It would collide with the imported Möbius/ANOVA constellation family. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Chess-specific symmetry | Two-element orbit `G={e,tau}` where `tau` is color flip: rank reflection + color swap + side/castling/en-passant swap | `x: [B,C,8,8] -> stack([x,tau(x)]): [B,2,C,8,8]` | Replace `tau` with a chess-invalid rank permutation that preserves counts and color swaps | Not sheaf/Hodge, not attack graph, not file mirror; it uses exact color flip only |
| Causal invariance | Same classifier evidence must work for both generated orbit environments | Per-view loss scalars `[B,2]` and optional risk variance over view index | Identity-view and bad-rank-view controls | Uses generated rule orbits, not source labels or closed-form nuisance projection |
| Evidential bottleneck | Nonnegative class evidence from each view is intersected before logits | `evidence: [B,2 views,2 classes] -> logits: [B,2]` | Replace harmonic intersection with arithmetic mean or view-0 logits | Not an ordinal ladder, not a selective head alone, not an ensemble |
| View agreement with anti-collapse | Optional Barlow/VICReg-style projection consistency between original and flipped embeddings | `z: [B,2,D]`, covariance over batch | Disable consistency while keeping the orbit head | Auxiliary only; the central operator remains orbit evidence intersection |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN with more filters | `src/chess_nn_playground/models/cnn.py` | It is ordinary capacity tuning and does not test a new tactical or causal inductive bias. |
| Residual CNN with more blocks | `src/chess_nn_playground/models/residual_cnn.py` | It is a standard ResNet scaling move already represented in the baseline suite. |
| LC0-style CNN on `lc0_bt4_112` | Existing LC0 BT4-style CNN variants | It copies the input style without adding a distinct falsifiable mechanism. |
| LC0-style residual CNN | Existing LC0 BT4 residual variants | It is a stronger baseline, not a research idea. |
| Ordinary ViT over 64 squares | Generic square-token Transformer | It is explicitly disallowed and would mainly test attention capacity. |
| Plain GNN on board squares | Generic square graph neural network | A fixed king/rook/knight-neighborhood graph is too generic and likely overlaps with ordinary spatial message passing. |
| Hyperparameter tuning | All existing trainers/configs | Optimizer, batch size, depth, and width tuning are not acceptable as the core idea. |
| Ensembling several existing models | Any baseline ensemble | It can improve metrics while revealing little about puzzle structure. |
| Static attack-defense graph or sheaf | Imported sheaf/Hodge/attack packets | Already researched; changing edge labels or pooling would be a near-duplicate. |
| One-ply move-delta set pooling | Imported move-delta packets | Already researched; this idea must avoid move bags, move spectra, and move landscapes. |
| Entropic piece-target transport | Imported OT/transport packets | Already researched; changing cost or target buckets would not be novel enough. |
| Ordinal cumulative label head | Imported ordinal evidence ladder | It uses fine-label order but not a new board operator, and it is already represented. |
| Sparse witness-piece bottleneck | Imported sparse witness packet | It risks repeating the same top-k witness mechanism with different scoring. |
| Ray-language automata | Imported ray-language packet | It already tests formal languages over rays; a color-flip wrapper would be insufficient novelty. |
| Möbius/ANOVA piece constellations | Imported constellation packet | High-order piece interaction tables are already covered. |
| Geometry-conditioned pseudo-likelihood ratio | Imported pseudo-likelihood packet | MDL/compression needs a different observable to avoid duplication. |

## 6. Mathematical Thesis

Input space definition:

Let `X_C` be the set of legal tensor encodings with `C` channels and shape `C x 8 x 8`. For the first experiment, `C=18` and the tensor consists of 12 binary piece planes plus side-to-move, castling, and en-passant planes. Let `x in X_18` be one encoded board.

Label/target definition:

Let `l in {0,1,2}` be the fine label. The binary target is

```text
Y = 1[l in {1,2}].
```

The model trains on `Y` and must not use `l` except through this binary target. Fine labels remain available for diagnostics after prediction.

Data distribution assumptions:

Assume a sample is drawn from an empirical distribution `P(X,Y,L)`. The distribution may contain nuisance variables `S`, such as color-to-move imbalance, composition conventions, phase imbalance, or encoding artifacts. The central assumption is not that the empirical distribution is perfectly symmetric. The central assumption is weaker:

```text
The chess-tactical component of P(Y | X) is invariant under exact color flipping,
while some nuisance correlations in P(X) are not.
```

Allowed symmetry or equivariance assumptions:

Define the color-flip map `tau` on current-board semantics:

```text
tau(piece_color, piece_type, file, rank)
  = (opposite_color, piece_type, file, 7-rank)

tau(side_to_move) = opposite side_to_move
tau(white_kingside_castle)  = black_kingside_castle
tau(white_queenside_castle) = black_queenside_castle
tau(black_kingside_castle)  = white_kingside_castle
tau(black_queenside_castle) = white_queenside_castle
tau(en_passant_file, en_passant_rank) = (same file, 7-en_passant_rank)
```

This is a two-element group action, `G={e,tau}`, with `tau^2=e`, when implemented on a semantically known current-board encoding.

No full rotation/reflection invariance is assumed. In particular:

- no file mirror is assumed;
- no diagonal reflection is assumed;
- no 90-degree or 180-degree rotation is assumed;
- pawn direction, castling semantics, side-to-move, and en-passant semantics are transformed explicitly, not ignored.

Core hypothesis:

For puzzle-likeness, the useful tactical evidence is orbit-stable under `tau`; orientation-specific evidence that appears only in `x` or only in `tau(x)` is more likely to be a nuisance shortcut. Therefore the classifier should use an orbit-intersection of evidence instead of a single-view score.

Formal object introduced by the idea:

Let

```text
phi_theta: X_C -> R^d
a_theta: R^d -> R_+^2
```

where `phi_theta` is a shared encoder and `a_theta` is a nonnegative class-evidence head.

For each `x`, define two views:

```text
v_0 = x
v_1 = tau(x)
```

Per-view evidence:

```text
E_jc(x) = softplus(a_theta(phi_theta(v_j))_c), for j in {0,1}, c in {0,1}.
```

Orbit-intersection evidence:

```text
I_c(x) = 2 E_0c(x) E_1c(x) / (E_0c(x) + E_1c(x) + epsilon).
```

The factor `2` makes `I_c=E_0c=E_1c` when the two views agree. If one view has near-zero evidence for class `c`, then `I_c` is near zero. The final logits are

```text
s_c(x) = log(1 + I_c(x)).
```

The model returns `s(x) in R^2`.

Optimization objective:

The minimal shared-trainer objective is weighted cross-entropy on the final logits:

```text
L_main(theta) = E[ w_Y * CE(s_theta(X), Y) ].
```

An idea-specific training script may add optional terms:

```text
L_view = 1/2 sum_{j=0}^1 CE(log(1+E_j(X)), Y)

L_inv = || normalize(p_theta(phi_theta(X)))
          - normalize(p_theta(phi_theta(tau(X)))) ||_2^2

L_rex = Var_{j in {0,1}}( CE(log(1+E_j(X)), Y) )

L_cov = off_diagonal_covariance_penalty(Z_0, Z_1)
```

with

```text
L_total = L_main + eta L_view + lambda_inv L_inv + lambda_rex L_rex + lambda_cov L_cov.
```

Default minimal experiment may set the optional weights to zero except `eta=0.1` if Codex implements the idea-specific train script. The architecture itself remains compatible with the shared trainer because `forward(x)` returns only logits by default.

Proposition 1: exact orbit invariance of the final classifier.

For all `x` with a valid semantic adapter,

```text
s_theta(tau(x)) = s_theta(x).
```

Proof sketch:

The two views of `tau(x)` are `{tau(x), tau(tau(x))} = {tau(x), x}`. The evidence-intersection operator `I_c` is symmetric in its two arguments. Therefore swapping the two views leaves `I_c` unchanged for each class. Applying `log(1+.)` preserves equality.

What is proven:

- The returned logits are exactly invariant to the implemented color-flip action.
- This does not depend on the encoder being equivariant.

Proposition 2: risk projection motivation under ideal orbit-invariant labels.

Let `p(y|x)` be any probabilistic classifier and define the orbit-averaged classifier

```text
p_G(y|x) = 1/2 [p(y|x) + p(y|tau(x))].
```

If `Y(x)=Y(tau(x))` and the evaluation distribution is symmetrized over the orbit, then the cross-entropy risk of `p_G` is no larger than the average orbit risk of `p`:

```text
-log p_G(Y|x)
  = -log( (p(Y|x)+p(Y|tau(x)))/2 )
  <= 1/2[-log p(Y|x) - log p(Y|tau(x))]
```

by convexity of `-log`.

What this proves:

- Under exact label invariance, restricting attention to orbit-invariant predictors is not inherently harmful for cross-entropy on the symmetrized problem.
- It gives a reason to test an invariant classifier when nuisance correlations are orientation-specific.

What remains hypothesized:

- The empirical labels in `crtk_sample_3class` are sufficiently close to color-flip invariant for this to improve generalization.
- Harmonic evidence intersection is better than arithmetic orbit averaging for near-puzzle discrimination.
- Fine-label-1 near-puzzles should show lower or more balanced evidence than fine-label-2 verified puzzles, even though the model is trained only on the binary target.
- The `simple_18` adapter can implement `tau` without silent channel-order mistakes.

Counterexamples where the idea should fail:

- A dataset source labels white-to-move and black-to-move positions differently for non-chess reasons, and the train/test split rewards that artifact.
- The channel-order metadata is wrong, so `tau` corrupts positions.
- Castling/en-passant channels are encoded in a way the adapter does not understand.
- Puzzle-likeness in the benchmark is dominated by material/phase priors rather than tactical invariants.
- The baseline CNN already learns the color-flip invariant solution from data, leaving no gain.
- Fine label `1` contains examples whose ambiguity is not orbit-related, so evidence intersection does not separate them.

Self-critique:

- Likely shortcuts: Material balance, phase, side-to-move frequency, and castling/en-passant sparsity survive the color-flip orbit. The model may still win by using those shortcuts unless compared to nuisance-matched and parameter-matched ablations.
- Leakage risks: The adapter must not call an engine, legal-move generator, mate detector, move counter, or provenance lookup. It should only permute existing current-board channels. Castling and en-passant are safe only as already-present encoding planes, not as legality or move-consequence oracles.
- Implementation risks: A silent channel-order mistake would invalidate the result. Codex must add one-hot tests for every piece type, side-to-move, castling channel, en-passant rank, and `tau(tau(x))==x`. LC0 support is explicitly secondary until channel semantics are known.
- Ablation weaknesses: The bad-rank orbit may create out-of-distribution boards, while the identity view may be too easy a control. This is why the plan also includes train-time augmentation and nuisance-matched unpaired-view controls.
- Mathematical objection: The proposition justifies orbit-invariant prediction under ideal label invariance, but it does not prove that harmonic evidence intersection is optimal. The harmonic head is a hypothesis about suppressing one-view-only shortcuts.
- Why the minimal experiment still survives: The central operator is cheap, exact, fail-closed, and falsifiable. A negative result would prune a meaningful family of symmetry-orbit ideas without using forbidden engine features.

## 7. Architecture Specification

Module names:

- `EncodingSemanticSpec`
- `ColorFlipOrbitAdapter`
- `SharedBoardEncoder`
- `OrbitEvidenceIntersectionHead`
- `ColorFlipOrbitEvidenceNet`
- Optional training-only helper: `OrbitConsistencyLoss`

Forward-pass steps for `simple_18`:

1. Input:
   - `x`: `[B, 18, 8, 8]`
2. Validate semantic adapter:
   - channel map exists;
   - piece planes are known;
   - side-to-move, castling, and en-passant channels are known;
   - otherwise raise `ValueError` before training.
3. Build color-flipped view:
   - `x_tau = tau(x)`: `[B, 18, 8, 8]`
4. Stack orbit views:
   - `views = cat([x, x_tau], dim=0)`: `[2B, 18, 8, 8]`
5. Shared convolutional encoder:
   - `h = encoder(views)`: `[2B, H, 8, 8]`
   - default `H=96` or `128`
6. Global pooling:
   - `u = mean(h, dim=(2,3))`: `[2B, H]`
7. Projection:
   - `z = proj(u)`: `[2B, D]`, default `D=128`
   - reshape `z`: `[B, 2, D]`
8. Evidence head:
   - `e = softplus(linear(z))`: `[B, 2, 2]`
9. Harmonic evidence intersection:
   - `e0 = e[:,0,:]`: `[B,2]`
   - `e1 = e[:,1,:]`: `[B,2]`
   - `e_int = 2 * e0 * e1 / (e0 + e1 + eps)`: `[B,2]`
10. Logits:
   - `logits = log1p(e_int)`: `[B,2]`
11. Return:
   - default: `logits`
   - optional `return_aux=True`: `(logits, {"z": z, "evidence": e, "x_tau": x_tau})`

Parameter-count estimate:

- Default small encoder:
  - `Conv(18 -> 64, 3x3)`: about 10k weights
  - `Conv(64 -> 96, 3x3)`: about 55k weights
  - one or two residual `96 -> 96` blocks: about 166k to 332k weights
  - projection and evidence heads: about 13k to 30k weights
- Expected total: roughly `0.25M` to `0.55M` parameters for `simple_18`.
- Codex should keep this in the same broad range as the small residual baseline or include a parameter-matched CNN control.

FLOP or complexity estimate:

- The orbit stack doubles encoder compute relative to a single-view CNN.
- Complexity is approximately:

```text
O(2 * B * 8 * 8 * sum_l (k_l^2 * C_l * C_{l+1}))
```

- With `B=512`, `H<=96`, and one to two residual blocks, this should remain small enough for the current benchmark; if memory is tight, reduce batch size to `256` before reducing the orbit mechanism.

Generated candidate-set memory:

This idea does not generate moves, piece-target candidates, graphs, hypergraphs, or search nodes. It generates exactly two orbit views.

- Orbit memory before the encoder:

```text
memory = B * K * C * 8 * 8 * bytes_per_float
K = 2
```

- For `simple_18`, float32 input memory is:

```text
B * 2 * 18 * 8 * 8 * 4 bytes
```

At `B=512`, this is about 4.7 MB for the raw orbit tensor. Feature maps dominate memory.

Chunking plan:

- No chunking is needed for `K=2` in the minimal experiment.
- If later LC0 encodings exceed memory, process the two views as `[2B,C,8,8]` when possible; otherwise encode `x` and `tau(x)` in two chunks and concatenate pooled features before the evidence head.

Required config fields:

- `model.name: color_flip_orbit_evidence`
- `model.input_channels`
- `model.num_classes: 2`
- `model.encoding`
- `model.hidden_channels`
- `model.latent_dim`
- `model.num_res_blocks`
- `model.eps`
- `model.semantic_adapter`
- `model.enable_aux_outputs`
- optional:
  - `model.orbit_transform: color_flip`
  - `model.intersection: harmonic`
  - `training.aux_view_ce_weight`
  - `training.orbit_consistency_weight`
  - `training.rex_weight`
  - `training.cov_weight`

Encoding support:

- `simple_18`:
  - First experiment should use this encoding.
  - Adapter must know the exact 12 piece-plane order.
  - Expected semantic order should be supplied by the repository encoding registry, not guessed silently. If the repository does not expose this metadata, Codex should add a small explicit `simple_18` semantic spec and tests.
  - The adapter rank-flips piece planes, swaps white/black piece planes by same piece type, toggles side-to-move, swaps white and black castling channels preserving kingside/queenside, and rank-flips the en-passant plane.
- `lc0_static_112`:
  - Feasible only if the current-board channels and scalar/state planes are semantically mapped.
  - If channel semantics are missing, the model must fail closed rather than applying an approximate transform.
  - A learned neural adapter may consume all channels, but deterministic `tau` must only be applied where semantics are verified.
- `lc0_bt4_112`:
  - Minimal experiment should not depend on it.
  - If zero-filled history planes are guaranteed, the adapter may transform known current planes and verify history planes are zero.
  - Once real history exists, every history slice must either have a verified color-flip transform or be excluded from the deterministic orbit path. Do not transform some channels and leave unknown channels incoherent.

Pseudocode only:

```text
forward(x, return_aux=False):
    spec = semantic_adapter.require_known(model.encoding, x.shape[1])
    x_tau = color_flip_tau(x, spec)

    views = concat_batch(x, x_tau)              # [2B,C,8,8]
    fmap = shared_encoder(views)                # [2B,H,8,8]
    pooled = spatial_mean(fmap)                 # [2B,H]
    z = projection(pooled).reshape(B,2,D)       # [B,2,D]

    evidence = softplus(evidence_head(z))       # [B,2,2]
    e0, e1 = evidence[:,0,:], evidence[:,1,:]
    e_int = 2 * e0 * e1 / (e0 + e1 + eps)       # [B,2]
    logits = log1p(e_int)                       # [B,2]

    if not return_aux:
        return logits
    return logits, {z, evidence}
```

Compatibility:

- The default `forward` returns logits only, so shared trainer, reports, confusion matrices, predictions, and leaderboards should continue working.
- Optional auxiliary losses should be implemented in `ideas/20260421_color_flip_orbit/train.py` or behind a trainer hook. They must not break the standard model registry path.

## 8. Loss, Training, And Regularization

Primary loss:

- Weighted binary cross-entropy through ordinary `torch.nn.CrossEntropyLoss` on the returned two-class logits.
- Use balanced class weighting as in the existing benchmark.

Optional auxiliary losses:

- `L_view`: per-view cross-entropy on `log1p(evidence)` from each orbit view, default weight `eta=0.1`.
- `L_inv`: normalized projection MSE between `z(x)` and `z(tau(x))`, default weight `lambda_inv=0.02`.
- `L_rex`: variance of per-view CE across the two generated environments, default weight `lambda_rex=0.01`.
- `L_cov`: optional off-diagonal covariance penalty on projected embeddings, default weight `0.0` for the first run unless Codex already has a clean implementation.
- The minimal shared-trainer run may set all auxiliary weights to zero; the architecture remains the main experiment.

Class weighting:

- `training.class_weighting: balanced`
- Fine labels `1` and `2` are both positive for the main loss.

Batch size expectations:

- Start with `batch_size=512` for `simple_18`.
- If memory pressure appears due to the two-view stack, use `batch_size=256` and keep all other fair-comparison settings unchanged.

Learning-rate and optimizer defaults:

- Optimizer: `AdamW`
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Epochs: `3` for the minimal current-data experiment
- Early stopping patience: `2`
- Mixed precision: `false` for determinism unless existing benchmark defaults differ.

Regularizers:

- Weight decay as above.
- Optional dropout `0.05` after pooled projection, only if parameter-matched controls also include it.
- Evidence clamp may be added for numerical stability, e.g. cap pre-log evidence at a large value such as `1e6`; this is not a modeling feature.

Determinism requirements:

- Seed: `42`
- `torch.use_deterministic_algorithms` if already supported by the project.
- Deterministic dataloader order for fair ablations.
- Color-flip adapter tests must be deterministic and must verify `tau(tau(x)) == x` for synthetic one-hot boards.

What must stay unchanged for fair comparison:

- Same train/val/test split.
- Same binary target definition.
- Same report code and 3x2 fine-label diagnostic.
- Same number of epochs unless all baselines are rerun.
- Same class weighting policy.
- Same encoding for the main comparison: `simple_18`.
- Same data filters and no full-dataset training.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Bad-rank color orbit | Replace `rank -> 7-rank` with a fixed nontrivial rank permutation, still swap colors/side and preserve material counts | Exact chess color-flip geometry matters, not just a second transformed view | If this matches the main model, the central group-action semantics are not contributing |
| Identity second view | Use `[x, x]` instead of `[x, tau(x)]` with the same evidence-intersection head | Benefit comes from the nontrivial orbit, not duplicated compute | If this matches the main model, the orbit transform is unnecessary |
| Single-view evidence head | Use only `x` and the same encoder/head parameter budget adjusted to match | The invariant two-view bottleneck beats a direct classifier | If this matches, the evidence-intersection architecture is likely cosmetic |
| Arithmetic orbit average | Replace harmonic evidence intersection with arithmetic mean of logits/evidence | Conservative intersection is better than ordinary test-time augmentation | If arithmetic wins, keep color-flip augmentation but abandon the evidence bottleneck |
| No optional orbit losses | Disable `L_view`, `L_inv`, `L_rex`, and `L_cov` | Architecture alone is responsible for gains | If only auxiliary losses help, report that the bottleneck claim is weaker |
| Train-time augmentation only | Train a normal CNN with random color-flip augmentation, infer on one view | The orbit-intersection inference is more than ordinary augmentation | If augmentation matches, use augmentation as a baseline and abandon the heavier model |
| Adapter-bug sentinel | Deliberately fail a test where side-to-move or castling is not swapped | Correct semantic transforms are required | If sentinel does not fail, tests are inadequate and benchmark results are invalid |
| Nuisance-matched unpaired view | Use a second position from the batch with matched material signature and side-to-move when available | Pair identity under `tau` matters beyond material/stm nuisance statistics | If this matches, the model may only exploit material/side distributions |
| Parameter-matched residual CNN | Residual CNN with similar parameter count and two-view compute budget removed | Gains are not from parameter count | If this wins, the orbit idea is not competitive |
| Fine-label diagnostic stress | Compare class-1 recall at fixed fine-label-0 FPR for main vs bad-rank/identity ablations | Near-puzzle ambiguity benefits from orbit-stable evidence | If class-1 recall does not improve, the idea may only affect easy positives |

This idea does not use graph, hypergraph, sheaf, transport, counterfactual move-set, legal search, or generated move candidates. The semantics-destroying control is therefore the bad-rank color orbit, plus nuisance-matched unpaired views that preserve obvious shortcuts such as material, side-to-move, and view count while destroying exact orbit semantics.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- Existing `simple_18` simple CNN.
- Existing `simple_18` residual CNN, preferably small or parameter-matched.
- Existing LC0-style CNN/residual results only as secondary context if already available; do not switch the main experiment away from `simple_18`.
- Train-time color-flip augmentation baseline if Codex can implement it cheaply.
- Central ablations from Section 9:
  - bad-rank color orbit
  - identity second view
  - arithmetic orbit average
  - single-view evidence head

Metrics to inspect:

- Validation and test accuracy.
- AUROC.
- AUPRC.
- F1 at default threshold.
- Balanced accuracy.
- Brier score and expected calibration error if existing report supports them.
- Required `3x2` fine-label confusion matrix for every main and ablation run.
- Fine-label-1 recall at a matched fine-label-0 false-positive rate.
  - Preferred: choose the main threshold so fine-label-0 FPR matches the strongest baseline's fine-label-0 FPR on validation, then report class-1 and class-2 recall on test.
  - Also report class-1 precision if available.

Required artifacts:

- Config YAML used for the main run.
- Config YAML for each central ablation.
- Checkpoint path.
- Validation and test metrics JSON/CSV.
- 3x2 diagnostic matrix for main model and central ablations.
- Predictions file with example IDs if the benchmark already emits one.
- Short report comparing main model to central ablations and existing baselines.
- Adapter test output proving `tau(tau(x)) == x` on synthetic channel-semantic cases.

Success threshold:

- Main `CFOEB` beats the best comparable `simple_18` baseline by at least one of:
  - `>= 1.5` AUROC points on test, or
  - `>= 2.0` points fine-label-1 recall at matched fine-label-0 FPR, or
  - clear calibration improvement, e.g. lower Brier/ECE, without losing more than `0.5` points class-2 recall.
- Main `CFOEB` must also beat the bad-rank orbit and identity-view ablations on the selected primary diagnostic.

Failure threshold:

- Main `CFOEB` is within noise of the bad-rank and identity ablations on AUROC and class-1 recall.
- Main `CFOEB` loses class-2 recall by more than `2` points relative to the baseline at matched fine-label-0 FPR.
- Adapter tests fail or require ambiguous channel assumptions.

What result would make this idea abandoned:

- The bad-rank orbit or identity second-view ablation matches or beats the true color-flip model on both AUROC and fine-label-1 recall at matched fine-label-0 FPR.
- Ordinary train-time color-flip augmentation matches the full model, and arithmetic averaging beats harmonic evidence intersection.
- The adapter cannot be made fail-closed for `simple_18`.

What result would justify scaling:

- The true `tau` model beats all central ablations and improves class-1 recall at matched fine-label-0 FPR while preserving class-2 recall.
- Calibration improves in a way consistent with lower evidence on ambiguous near-puzzles.
- Adapter tests are clean, and model runtime is acceptable.
- Then scale to:
  - more epochs,
  - parameter-matched residual encoder,
  - optional LC0 semantic adapter only after channel-map tests exist,
  - source-shift or encoding-shift diagnostics.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_color_flip_orbit/idea.yaml` | Create | Machine-readable idea metadata from Section 12. |
| `ideas/20260421_color_flip_orbit/math_thesis.md` | Create | Section 6 math thesis, propositions, assumptions, and counterexamples. |
| `ideas/20260421_color_flip_orbit/architecture.md` | Create | Module-level design, tensor shapes, adapter rules, parameter/FLOP estimates. |
| `ideas/20260421_color_flip_orbit/implementation_notes.md` | Create | Fail-closed semantic adapter details, `tau(tau(x))==x` tests, no-engine-feature checklist. |
| `ideas/20260421_color_flip_orbit/trainer_notes.md` | Create | Primary CE loss, optional aux losses, class weighting, deterministic training, fair-comparison rules. |
| `ideas/20260421_color_flip_orbit/ablations.md` | Create | Section 9 ablation table and exact config differences. |
| `ideas/20260421_color_flip_orbit/train.py` | Create | Optional idea-specific training wrapper only if auxiliary losses are implemented; otherwise call the shared trainer with the config. |
| `ideas/20260421_color_flip_orbit/config.yaml` | Create | Main `simple_18` config for `color_flip_orbit_evidence`. |
| `ideas/20260421_color_flip_orbit/report_template.md` | Create | Template requiring baseline table, central ablation table, 3x2 matrices, class-1 matched-FPR diagnostic, and adapter-test status. |
| `ideas/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this packet to imported memory after results; preserve hard constraints; add anti-duplicate rule for color-flip orbit/evidence-intersection if it fails. |
| `src/chess_nn_playground/models/color_flip_orbit_evidence.py` | Create | `EncodingSemanticSpec`, `ColorFlipOrbitAdapter`, `SharedBoardEncoder`, `OrbitEvidenceIntersectionHead`, `ColorFlipOrbitEvidenceNet`. |
| `src/chess_nn_playground/models/registry.py` | Update | Register builder name `color_flip_orbit_evidence`. |
| `configs/color_flip_orbit_simple18.yaml` | Create | Shared-trainer compatible config using `simple_18`, `input_channels: 18`, `num_classes: 2`. |
| `configs/color_flip_orbit_bad_rank_simple18.yaml` | Create | Central bad-rank orbit ablation config. |
| `configs/color_flip_orbit_identity_simple18.yaml` | Create | Identity second-view ablation config. |
| `configs/color_flip_orbit_arithmetic_simple18.yaml` | Create | Arithmetic orbit average ablation config. |
| `tests/test_color_flip_orbit_adapter.py` | Create | Synthetic one-hot tests for piece swaps, rank flip, side-to-move toggle, castling swap, en-passant rank flip, and `tau(tau(x))==x`. |
| `tests/test_color_flip_orbit_model.py` | Create | Shape tests: `[B,18,8,8] -> [B,2]`; invariance test: `model(x)==model(tau(x))` in eval mode up to tolerance. |
| `tests/test_model_registry_color_flip_orbit.py` | Create | Registry builder smoke test if the project has comparable model registry tests. |

For `ideas/chatgpt_pro_deep_math_research_prompt.md`, Codex should add a concise memory item after consuming this output:

```text
Color-Flip Orbit Evidence Bottleneck:
exact chess color-flip/rank-reflection orbit tau over current-board semantic channels
+ shared encoder
+ harmonic evidential intersection of original/flipped views
+ optional orbit consistency/risk-variance regularization
+ binary puzzle-likeness target
+ no engine metadata, no attack graph, no move set, no OT, no nuisance projection
```

If results are negative, Codex should add:

```text
Do not repeat color-flip orbit evidence-intersection, identity/bad-rank orbit controls, or generic color-flip augmentation as a fresh idea unless the operator changes beyond the two-element tau orbit and harmonic/mean evidence pooling.
```

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0751_tuesday_los_angeles_color_flip_orbit.md
  generated_at: 2026-04-21 07:51:27 America/Los_Angeles
  weekday: Tuesday
  timezone: America/Los_Angeles
  idea_slug: color_flip_orbit
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_color_flip_orbit
  name: Color-Flip Orbit Evidence Bottleneck
  slug: color_flip_orbit
  status: draft
  created_at: 2026-04-21 07:51:27 America/Los_Angeles
  author: ChatGPT Pro
  short_thesis: Puzzle-like evidence should survive exact chess color flipping; intersect original/flipped evidence to suppress orientation-specific shortcuts.
  novelty_claim: Exact color-flip orbit plus harmonic evidential intersection, not a sheaf, attack graph, move-delta set, OT model, nuisance projection, ordinal head, sparse witness, ray automaton, constellation model, or pseudo-likelihood model.
  expected_advantage: Better near-puzzle recall and calibration at matched non-puzzle false-positive rate by removing color/orientation shortcuts.
  central_falsification_ablation: Replace exact color flip with a chess-invalid material-preserving rank permutation plus color swap while keeping the same architecture and losses.
  target_task: coarse_binary
  input_representation: simple_18_first
  output_heads: binary_logits_from_orbit_intersection
  compute_notes: Two orbit views double encoder compute; no move generation or candidate set; expected 0.25M-0.55M parameters.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/color_flip_orbit_simple18.yaml
  model_path: src/chess_nn_playground/models/color_flip_orbit_evidence.py
  latest_result_path: null
  notes: Use fail-closed semantic adapter; first experiment should not depend on LC0 channel semantics.
```

```yaml
config_yaml:
  run:
    name: color_flip_orbit_simple18
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
    name: color_flip_orbit_evidence
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
  model_name: color_flip_orbit_evidence
  file_path: src/chess_nn_playground/models/color_flip_orbit_evidence.py
  builder_function: build_color_flip_orbit_evidence
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - EncodingSemanticSpec
    - ColorFlipOrbitAdapter
    - SharedBoardEncoder
    - OrbitEvidenceIntersectionHead
    - ColorFlipOrbitEvidenceNet
  required_config_fields:
    - model.name
    - model.input_channels
    - model.num_classes
    - data.encoding
    - model.semantic_adapter
    - model.hidden_channels
    - model.latent_dim
    - model.num_res_blocks
    - model.eps
  expected_parameter_count: 0.25M-0.55M for simple_18 default
  expected_memory_notes: Two orbit views; raw simple_18 orbit tensor is B*2*18*8*8 floats; feature maps dominate; reduce batch size to 256 if needed.
```

```yaml
research_continuity:
  idea_fingerprint: exact color-flip rank-reflection orbit tau + shared encoder + harmonic evidential intersection + optional orbit consistency/risk-variance regularization
  already_researched_family_overlap: Low overlap with imported sheaf/Hodge, move-delta, OT, nuisance-projection, ordinal, sparse-witness, ray-language, constellation, and pseudo-likelihood packets.
  closest_duplicate_risk: Generic color-flip augmentation or group-equivariant CNN; distinguish by exact tau adapter, invariant inference-time two-view orbit, and harmonic evidence intersection with bad-rank/identity controls.
  do_not_repeat_if_this_fails:
    - two-element color-flip orbit evidence intersection
    - color-flip-only train/test augmentation as a standalone fresh idea
    - harmonic/mean pooling of original and color-flipped logits as the central mechanism
    - generic orbit consistency losses over tau without a new observable
  suggested_next_search_directions:
    - label-safe uncertainty for fine-label-1 ambiguity not based on ordinal ladders
    - source-shift diagnostics that do not use provenance as model input
    - masked generative compression with controls that avoid pseudo-likelihood duplication
    - invariant objectives across encodings after LC0 semantic adapter tests exist
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add the `Color-Flip Orbit Evidence Bottleneck` fingerprint to imported research memory after benchmarking. | Prevents future packets from repackaging the same two-view color-flip/evidence-intersection idea. | `Imported Research Memory` |
| If the bad-rank/identity ablations fail to separate, add a rule: do not propose color-flip orbit pooling or generic symmetry augmentation unless a new formal object is introduced. | Negative results should prune a family, not just one implementation. | `Do not propose...` anti-duplicate paragraphs |
| Require future symmetry proposals to list exact channel transforms for side-to-move, castling, en-passant, and LC0 history channels, plus `tau^2=id` tests. | Symmetry ideas are easy to corrupt silently through adapter mistakes. | `Required Markdown File Content` and `Non-Negotiable Constraints` |
| Add a reminder that file mirror, rank color flip, and 180-degree rotation are different chess assumptions. | Avoids accidental duplication with file-mirror sheaf or invalid D4 invariance. | `Research Mode` |
| Record whether the first experiment used only `simple_18` or added LC0 support. | Helps the next pass know whether LC0 channel semantics remain unresolved. | `Research Continuity` |
| Preserve the hard leakage rules exactly as written. | The idea intentionally avoids engine and move-tree leakage; future prompt edits should not weaken this. | `Non-Negotiable Constraints` |

## 14. Final Sanity Check

- Downloadable Markdown file created: Yes
- Filename follows required date/time/day/timezone/slug pattern: Yes
- No forbidden engine features used as inputs: Yes
- Does not fabricate labels: Yes
- Not a routine CNN/ResNet/Transformer variant: Yes
- Minimal current-data experiment exists: Yes
- Falsification criterion is concrete: Yes
- Codex can implement without asking for missing architecture details: Yes
- Prompt maintenance notes included for Codex: Yes
- Repetition check against imported research packets completed: Yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: Yes
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: Yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: Yes
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Möbius-constellation, or pseudo-likelihood packets: Yes
