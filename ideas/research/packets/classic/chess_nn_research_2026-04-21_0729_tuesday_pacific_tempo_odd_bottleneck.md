# Codex Handoff Packet: Centered Tempo-Odd Interventional Bottleneck

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0729_tuesday_pacific_tempo_odd_bottleneck.md`
- Generated at: 2026-04-21 07:29:00 America/Los_Angeles
- Weekday: Tuesday
- Timezone: America/Los_Angeles, filename token `pacific`
- Idea slug: `tempo_odd_bottleneck`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: **Centered Tempo-Odd Interventional Bottleneck**.
- One-sentence thesis: Puzzle-likeness should be predicted from the board-dependent part of the position's response to changing only the side to move, not from static board appearance, source artifacts, or a raw side-to-move prior.
- Idea fingerprint: `current-board tensor + deterministic side-to-move involution + shared encoder + C2 odd/even representation split + null-board centering of pure-turn effects + binary puzzle-likeness target + no engine metadata + no legal move tree + no one-ply move-delta bag`.
- Closest baseline or common method it resembles: A Siamese contrastive CNN wrapper around the existing simple CNN family, with a mathematically fixed turn-intervention odd projection instead of contrastive pair labels or data augmentation.
- Why this is not a common CNN/ResNet/Transformer variant: The central operator is not another depth/width choice; it constrains the classifier to a centered anti-invariant component under the side-to-move involution, so board-only and side-only functions are removed before the final head.
- Current-data minimal experiment: Train on `simple_18` using the existing `crtk_sample_3class` train/val/test Parquet split, map fine labels `1` and `2` to coarse class `1`, report the standard binary metrics plus the required `3x2` fine-label diagnostic matrix.
- Smallest central falsification ablation: Replace the true side-to-move twin `tau(x)` with a batch-shuffled or random-turn twin while preserving the same encoder, four-pass compute budget, side-to-move marginal, and final head; if performance is unchanged, the claimed tempo-odd intervention is not doing useful work.
- Expected information gain if it fails: A clean failure says that current split performance is probably dominated by static board features, source artifacts, or generic CNN capacity rather than side-to-move-conditioned tactical actionability; future cycles should then avoid turn-intervention bottlenecks and focus on label-safe ambiguity/calibration or stronger artifact controls.

## 3. Problem Restatement And Data Contract

The task is binary chess puzzle-likeness classification from board positions.

- Fine label `0`: known non-puzzle.
- Fine label `1`: verified near-puzzle.
- Fine label `2`: verified puzzle.
- Coarse target for the default benchmark: `y = 0` for fine label `0`; `y = 1` for fine labels `1` or `2`.
- Model input shape: `(batch, C, 8, 8)`.
- Model output shape: `(batch, 2)` logits.
- Required diagnostic report: rectangular `3x2` matrix showing `true fine label 0/1/2 -> predicted binary output 0/1`.
- Current benchmark split:
  - `data/splits/crtk_sample_3class/split_train.parquet`
  - `data/splits/crtk_sample_3class/split_val.parquet`
  - `data/splits/crtk_sample_3class/split_test.parquet`
- Do not point the current trainer directly at the roughly 45M-row full Parquet dataset until streaming support exists.

Available encodings:

- `simple_18`: 12 piece planes plus side-to-move, castling, and en-passant planes.
- `lc0_static_112`.
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists.

Leakage checklist:

- Allowed as neural inputs: deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and deterministic tensor transforms derived only from the current board.
- Allowed as rule-derived features, if ever added later and carefully ablated: pseudo-legal attack geometry derived only from the current board.
- Leakage-prone unless separately justified and ablated: full legal-move generation, legal move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences.
- Always forbidden as neural inputs: Stockfish or other engine evaluation, PVs, mate scores, node counts, verification metadata, source labels, proposed labels, dataset provenance, or any feature computed from them.
- Fine labels may be used only as supervised training targets and evaluation diagnostics, never as input features.
- For `lc0_static_112` and `lc0_bt4_112`, current-board channels may be used by deterministic adapters only when channel semantics are explicitly known. History channels may be consumed by learned neural adapters, but deterministic geometry or side-to-move transformations must fail closed if their exact channel meaning is unknown.

The selected model's deterministic operation is only: construct a side-to-move counterfactual tensor twin by toggling the side-to-move channel. It does not ask whether the toggled FEN is legal, does not enumerate moves, does not inspect check or mate, and does not use an engine.

## 4. Research Map

External ideas used:

| Source | Borrowed | Not copied |
|---|---|---|
| Taco Cohen and Max Welling, "Group Equivariant Convolutional Networks", ICML/arXiv 2016, https://arxiv.org/abs/1602.07576 | The idea that a known discrete transformation can be built into a network through structured sharing or projection. | No group convolution, no full chess-board rotation/reflection group, and no claim that chess is invariant under D4. |
| Yaroslav Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016, https://jmlr.org/papers/v17/15-239.html | Optional implementation pattern for adversarial audits or gradient reversal if Codex later wants to test whether the even component carries label leakage. | The core experiment does not require domain adaptation, external domains, or a gradient-reversal loss. |
| Martin Arjovsky, Leon Bottou, Ishaan Gulrajani, David Lopez-Paz, "Invariant Risk Minimization", arXiv 2019, https://arxiv.org/abs/1907.02893 | The warning that high training accuracy can rely on spurious correlations and that invariance can improve out-of-distribution behavior. | No IRM penalty, no environment labels, and no source-provenance input. |
| Bernhard Schoelkopf et al., "Towards Causal Representation Learning", arXiv 2021, https://arxiv.org/abs/2102.11107 | The high-level framing that interventions on semantically meaningful variables can expose causal structure. | No causal discovery claim and no proof that puzzle labels are causal effects of side-to-move. |
| Naftali Tishby, Fernando Pereira, William Bialek, "The information bottleneck method", arXiv 2000, https://arxiv.org/abs/physics/0004057 | The bottleneck principle: preserve target-relevant information while suppressing nuisance information. | No Blahut-Arimoto algorithm, no exact mutual-information estimator, and no generative compression objective. |

Candidate search trace:

| Candidate mechanism considered | Why it was serious | Why it lost to the selected idea |
|---|---|---|
| Label-blind masked board compression residual | It could test whether puzzles are positions with unusual but rule-coherent local structure. | Too close to the imported static-geometry pseudo-likelihood / description-length family, and more vulnerable to source artifacts in FEN syntax or dataset style. |
| Environment-invariant risk across material phase, material balance, side-to-move, and encoding family | It directly attacks shortcut learning. | It is mostly a training objective rather than a distinct chess operator, and encoding-family environments are not yet a clean current-data contract. |
| Selective prediction / abstention head for class-`1` ambiguity | It respects the verified near-puzzle boundary and could improve calibration. | It does not create a new representation of puzzle structure; it is better as a later evaluation layer after a stronger feature operator exists. |
| Object-role slot binding for "attacker", "defender", "king", and "loose piece" roles | It has a plausible tactical inductive bias without engine scores. | It risks becoming a sparse witness-piece attention model under a new name, which is already an imported researched family. |
| Persistent homology or cubical-complex summaries of occupied squares | It is mathematically distinct from CNNs and graphs. | It seems too lossy for chess tactics and has weak falsification beyond "topology was not enough". |
| Energy-based consistency under random legal-state corruptions | It could learn robust motifs without labels. | It approaches masked generative modeling, needs careful corruption design, and would be harder for Codex to benchmark quickly. |
| Partial file-mirror equivariance with learned parity gates | Chess has a real left-right mirror symmetry if castling and en-passant are remapped. | Imported packets already include file-mirror tension/sheaf ideas, and simple augmentation is too ordinary. |
| **Centered side-to-move odd bottleneck** | It isolates the board-dependent effect of tempo using a fixed finite-group projection and does not need move generation. | Selected because it is simple, falsifiable, label-safe, and outside the imported sheaf, move-delta, OT, ordinal, sparse-witness, ray-language, constellation, and pseudo-likelihood families. |

The broader internal screen covered at least twelve mechanisms: masked compression, material-phase IRM, encoding-family DRO, abstention, role slots, persistent topology, corruption energy, file-mirror equivariance, graph spectra, FEN grammar compression, side-to-move odd projection, and information-dropout nuisance adversaries. The selected operator is the only one in that screen that is both mathematically sharp and cheap enough for the current split.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN on `simple_18` | `src/chess_nn_playground/models/cnn.py` | Already exists and mainly tests local pattern recognition, not a new puzzle-likeness mechanism. |
| Residual CNN on `simple_18` | `src/chess_nn_playground/models/residual_cnn.py` | Already exists; more residual capacity would be routine architecture scaling. |
| LC0-style CNN on `lc0_static_112` or `lc0_bt4_112` | Existing LC0 BT4-style CNN variants | Too close to copying LC0 input conventions and baseline towers without a new falsifiable operator. |
| LC0-style residual CNN | Existing LC0 BT4-style residual CNN variants | Already covered by the baseline suite; changing block counts or widths is not research novelty. |
| Ordinary ViT over 64 square tokens | Vanilla Transformer over board squares | Explicitly disallowed as a core idea and likely data-hungry for the current sample split. |
| Plain GNN on square adjacency or piece attack graph | Generic graph neural network | Too close to the imported attack-defense graph/sheaf/Hodge family unless the operator is radically different. |
| Hyperparameter tuning | Any existing model | Learning-rate, depth, width, dropout, and optimizer sweeps are useful engineering but not a research idea. |
| Ensembling several trained classifiers | Any leaderboard model | May improve metrics but does not explain puzzle-likeness and hides failure modes. |
| Adding more data from the full Parquet file | Current trainer on full data | The full file should not be used before streaming support, and "more data" is not a mechanism. |
| One-ply pseudo-legal move-delta bags | Imported counterfactual move-delta packets | Already researched; this proposal deliberately avoids move sets, move consequences, and move-delta pooling. |
| Tactical sheaf, Hodge, curvature, or attack-defense incidence energy | Imported tactical sheaf/Hodge packets | Already researched; adding edge types or renamed tension energies would be a near-duplicate. |
| Piece-target/material-target Sinkhorn or optimal transport | Imported OT packets | Already researched; this proposal uses no transport plan, coupling matrix, or Sinkhorn bottleneck. |
| Deterministic nuisance-vector residualization | Imported nuisance-orthogonal packet | Already researched; the selected idea uses an interventional parity split, not closed-form projection away from material/phase vectors. |
| Ordinal cumulative heads for fine labels | Imported ordinal evidence ladder | Already researched; this task remains a binary classifier with fine labels used only diagnostically. |
| Sparse witness-piece bottleneck | Imported sparse witness packet | Not repeated because slot/witness attention can turn into top-k piece selection under a new name. |
| Ray-language automata | Imported ray-language packet | Already researched; this proposal uses no rank/file/diagonal token automaton. |
| High-order occupied-piece constellation ANOVA | Imported Möbius constellation packet | Already researched; the selected idea performs only a side-to-move intervention split, not degree-2/3 piece interactions. |
| Static-geometry pseudo-likelihood or description-length ratio | Imported pseudo-likelihood packet | Already researched; the selected idea does not train class-conditioned board likelihoods or compare description lengths. |

## 6. Mathematical Thesis

### Input space definition

Let a board tensor be represented as `x = (b, t)`, where:

- `b` is every input channel except the side-to-move plane, including piece occupancy and other safe current-board planes.
- `t in {−1, +1}` is the side-to-move bit after mapping the side plane to a binary signed value.
- The implemented tensor still has shape `(C, 8, 8)`; this notation only separates the semantic side-to-move variable.

For the minimal `simple_18` experiment, `b` contains the 12 piece planes plus castling/en-passant planes, and `t` is the known side-to-move plane. For unsupported channel layouts, the model must raise a configuration error rather than guessing.

Define the side-to-move involution

\[
\tau(b,t) = (b,-t).
\]

Define a null board with the same turn bit

\[
\nu(b,t) = (b_0,t),
\]

where `b0` is the all-zero non-turn tensor. This is not a legal chess position. It is a baseline input used only to subtract the encoder's pure side-to-move offset.

### Label/target definition

Let `FINE in {0,1,2}` be the verified fine label. The coarse target is

\[
Y = \mathbf 1\{FINE \in \{1,2\}\}.
\]

The fine label is not an input feature.

### Data distribution assumptions

The hypothesis is not that the dataset is perfectly causal or unbiased. The working assumptions are weaker:

1. Verified puzzles and near-puzzles are more likely than non-puzzles to contain a side-to-move-conditioned tactical opportunity.
2. Some non-puzzle positions can share static board motifs with puzzles.
3. Dataset artifacts may correlate with material, phase, side-to-move, or source style.
4. A classifier that can use only static board appearance may therefore overpredict puzzle-likeness on fine-label `0`.

### Allowed symmetry or equivariance assumptions

Chess is not invariant under arbitrary board rotations or reflections: pawns have direction, castling is asymmetric, en-passant has rank semantics, and side-to-move matters. This proposal assumes only the existence of a semantic involution on the tensor variable `t`: change whose turn it is while leaving the visible board arrangement fixed.

This is not a label-preserving symmetry. A puzzle for White to move need not remain a puzzle for Black to move. The point is exactly that puzzle-likeness should often depend on this tempo variable. The model uses the intervention to measure sensitivity, not to augment labels.

### Core hypothesis

There exists a representation \(\phi(b,t)\) such that the target-relevant part of puzzle-likeness is concentrated in the board-dependent odd component under \(\tau\):

\[
\phi_{\text{tempo}}(b,t)
=
\frac{1}{2}\left[\phi(b,t)-\phi(b,-t)\right]
-
\frac{1}{2}\left[\phi(b_0,t)-\phi(b_0,-t)\right].
\]

The first term is side-to-move anti-invariant. The second term subtracts what the encoder would do for the side bit alone. A final classifier trained on \(\phi_{\text{tempo}}\) should be less able to exploit board-only source artifacts or a raw White-to-move/Black-to-move prior.

### Formal object introduced

Let \(F_\theta : X \rightarrow \mathbb R^{D \times 8 \times 8}\) be a shared convolutional encoder. Define:

\[
O_\theta(x)
=
\frac{1}{2}\left(F_\theta(x)-F_\theta(\tau x)\right),
\]

\[
E_\theta(x)
=
\frac{1}{2}\left(F_\theta(x)+F_\theta(\tau x)\right),
\]

\[
C_\theta(x)
=
O_\theta(x) - O_\theta(\nu x).
\]

The model pools \(C_\theta(x)\) and feeds it to a binary classifier. \(E_\theta(x)\) may be logged for ablations but is not used by the main head.

### Proposition: centered odd features remove additive static and pure-turn components

Assume a feature map \(F\) can be decomposed as

\[
F(b,t) = a(b) + c + t u + t v(b) + r(b,t),
\]

where \(a(b)\) is board-only, \(c\) is constant, \(tu\) is pure side-to-move, \(tv(b)\) is a board-turn interaction, and \(r\) is any remaining non-additive residual. Also assume \(v(b_0)=0\). Then

\[
C(b,t)
=
t v(b)
+
\frac{1}{2}\left[r(b,t)-r(b,-t)\right]
-
\frac{1}{2}\left[r(b_0,t)-r(b_0,-t)\right].
\]

In particular, \(a(b)\), \(c\), and \(tu\) are exactly removed.

### Proof sketch

Compute the odd component:

\[
O(b,t)
=
\frac{1}{2}\{F(b,t)-F(b,-t)\}.
\]

Substituting the decomposition, the board-only term \(a(b)\) and constant \(c\) cancel. The pure-turn term gives \(tu\). The interaction term gives \(t v(b)\). The residual contributes its odd part. Thus

\[
O(b,t)=tu+t v(b)+\frac{1}{2}[r(b,t)-r(b,-t)].
\]

For the null board,

\[
O(b_0,t)=tu+t v(b_0)+\frac{1}{2}[r(b_0,t)-r(b_0,-t)].
\]

Using \(v(b_0)=0\), subtracting \(O(b_0,t)\) removes the pure-turn term \(tu\) and leaves the expression above.

Also,

\[
C(\tau x) = -C(x),
\]

so the centered representation is anti-invariant under side-to-move toggling.

### Optimization objective

The main training objective is standard weighted cross-entropy:

\[
\min_{\theta,w}
\mathbb E_{(x,y)}
\left[
\operatorname{CE}
\left(
y,
g_w(\operatorname{pool}(C_\theta(x)))
\right)
\right].
\]

Optional regularization for the idea-specific training wrapper:

\[
\lambda_{\text{null}}\mathbb E_x\|O_\theta(\nu x)\|_2^2
\]

to discourage the encoder from devoting capacity to pure side-to-move offsets. This regularizer is optional; the minimal fair benchmark may omit it and rely on explicit null-centering.

### What is actually proven

- The odd/even split is mathematically determined by the side-to-move involution.
- The centered odd feature \(C_\theta\) is anti-invariant under turn toggling.
- Additive board-only features and pure side-to-move offsets cannot pass unchanged through \(C_\theta\).
- The mechanism uses no engine scores, no verification metadata, no source labels, and no legal move generation.

### What remains only hypothesized

- That true puzzle-likeness in this dataset is better captured by board-turn interaction than by static board appearance.
- That suppressing static and pure-turn shortcuts improves class-`1` near-puzzle behavior.
- That the out-of-distribution counterfactual tensor \(\tau x\) is still useful for representation learning even when it is not a legal FEN.
- That a small CNN encoder has enough capacity to learn useful tempo-odd interactions from the current split.

### Counterexamples where the idea should fail

- A puzzle class dominated by static source artifacts independent of side-to-move.
- A dataset where the side-to-move plane itself is spuriously predictive and the model recovers that prior through nonlinear interactions despite centering.
- Positions whose puzzle-likeness depends mainly on exact legal status, mate/stalemate, or multi-ply consequences that are invisible to a side-toggle contrast.
- Endgame tablebase-like positions where the current board's static material is decisive but tempo-sensitive patterns are too subtle for the encoder.
- Encodings whose side-to-move semantics are unknown or distributed across history planes, causing the adapter to fail closed.

### Self-critique

The strongest objection is that toggling side-to-move creates tensors that may be far off the natural data manifold. A CNN could learn artifacts of this artificial intervention rather than real chess tempo. The null-board centering reduces the easiest pure-turn shortcut, but it does not prove causal validity.

The minimal experiment is still worth running because the ablation is unusually sharp: if batch-shuffled or random-turn twins match the main model, the intervention is dead. If the true side-to-move twin improves near-puzzle recall at matched false-positive rate while the random twin does not, that is evidence that the current board's tempo sensitivity contains useful supervised signal.

## 7. Architecture Specification

### Module names

- `Simple18TurnAdapter`
- `TempoNullBuilder`
- `SharedBoardEncoder`
- `CenteredTempoOddBottleneckNet`
- Optional audit-only module: `EvenComponentProbe`

Implementation target file:

- `src/chess_nn_playground/models/tempo_odd_bottleneck.py`

Registry name:

- `centered_tempo_odd_bottleneck`

### Forward-pass steps

Input:

- `x`: `(B, C, 8, 8)`

For the first experiment:

- `C = 18`
- `encoding = simple_18`
- `stm_channel = 12`, unless the existing exporter metadata says otherwise.

Steps:

1. Validate channel count and adapter metadata.
   - If `encoding == simple_18`, use the configured side-to-move channel.
   - If `encoding in {lc0_static_112, lc0_bt4_112}` and no explicit channel map is provided, raise `ValueError`.

2. Construct the true turn twin:
   - `x_tau = tau(x)`
   - Shape: `(B, C, 8, 8)`
   - For a binary side-to-move plane, replace `x[:, stm_channel]` with `1 - x[:, stm_channel]`.
   - Leave all other channels unchanged.

3. Construct the null-board inputs:
   - `x_null = nu(x)`: all non-side-to-move channels zeroed, side-to-move channel copied from `x`.
   - `x_null_tau = tau(x_null)`.
   - Each has shape `(B, C, 8, 8)`.

4. Concatenate for one shared encoder call:
   - `x4 = concat([x, x_tau, x_null, x_null_tau], dim=0)`
   - Shape: `(4B, C, 8, 8)`

5. Shared encoder:
   - `h4 = F_theta(x4)`
   - Recommended shape: `(4B, D, 8, 8)`, with `D = 96` by default.

6. Split:
   - `h, h_tau, h_null, h_null_tau = split(h4, B)`
   - Each: `(B, D, 8, 8)`

7. Odd/even maps:
   - `odd = 0.5 * (h - h_tau)`
   - `even = 0.5 * (h + h_tau)`
   - `null_odd = 0.5 * (h_null - h_null_tau)`
   - `centered_odd = odd - null_odd`
   - Each: `(B, D, 8, 8)`

8. Pool centered odd:
   - `avg = mean(centered_odd, dim=(2,3))`: `(B, D)`
   - `amax = max(centered_odd, dim=(2,3))`: `(B, D)`
   - `l2 = sqrt(mean(centered_odd^2, dim=(2,3)) + eps)`: `(B, D)`
   - `z = concat([avg, amax, l2], dim=1)`: `(B, 3D)`

9. Classifier:
   - `MLP(3D -> hidden -> 2)`
   - Default hidden width: `192`.
   - Output logits: `(B, 2)`.

10. Optional audit outputs must not break the shared trainer.
    - Default `forward(x)` returns only logits.
    - If `return_aux=True`, return a dict with `logits`, `even_probe_logits`, `null_odd_norm`, and `centered_odd_norm`, but keep default trainer compatibility.

### Shared encoder sketch

Use a small CNN tower for a fair first test:

- Conv `C -> 64`, kernel `3`, padding `1`; GroupNorm or BatchNorm; GELU.
- Conv `64 -> 96`, kernel `3`, padding `1`; norm; GELU.
- Two lightweight residual blocks at width `96`.
- Optional squeeze gate from side-to-move plane is not allowed; the whole point is to let the odd projection handle turn sensitivity.

This is not the research novelty. Keep it roughly comparable to existing small/medium CNN baselines.

### Tensor shapes

| Stage | Shape |
|---|---|
| Input | `(B, C, 8, 8)` |
| Turn twin | `(B, C, 8, 8)` |
| Null and null-turn twins | `(B, C, 8, 8)` each |
| Concatenated encoder input | `(4B, C, 8, 8)` |
| Encoder output | `(4B, 96, 8, 8)` |
| Centered odd map | `(B, 96, 8, 8)` |
| Pooled vector | `(B, 288)` |
| Logits | `(B, 2)` |

### Parameter-count estimate

For `simple_18`, width `96`, two residual blocks, and MLP hidden `192`, expected parameter count is approximately `0.35M` to `0.55M`, depending on normalization and bias choices. This should remain in the same rough regime as small CNN baselines, not deep ResNets.

### FLOP and complexity estimate

Let the base encoder cost be \(E(B,C,D)\). The main model uses one encoder call on a concatenated batch of `4B`, so compute is approximately `4x` the base encoder plus a negligible pooling/head cost. Memory is dominated by activations for `4B`.

There is no generated move/candidate set. Candidate memory is therefore:

\[
O(0)
\]

beyond the four deterministic tensor views. If a future implementation adds any candidate set, it is outside this packet's design.

Chunking plan:

- Default: encode all `4B` tensors in one concatenated call for BatchNorm/statistical consistency.
- If GPU memory is tight, add config `encoder_chunk_size`. Compute encoder outputs in chunks of at most `encoder_chunk_size` rows, then concatenate before odd/even arithmetic.
- Chunking must preserve pairing order: `[x, tau(x), nu(x), tau(nu(x))]`.

### Required config fields

```yaml
model:
  name: centered_tempo_odd_bottleneck
  input_channels: 18
  num_classes: 2
  encoding: simple_18
  stm_channel: 12
  encoder_width: 96
  encoder_blocks: 2
  mlp_hidden: 192
  use_null_centering: true
  return_aux: false
  encoder_chunk_size: null
```

### Encoding support

First experiment should use `simple_18` only. This is not a weakness; it prevents accidental channel-semantics leakage or malformed LC0 history manipulation.

Adapter assumptions:

| Encoding | Deterministic adapter behavior |
|---|---|
| `simple_18` | Supported. Requires known side-to-move channel. Toggle only that plane. Zero all other channels for null-board centering. |
| `lc0_static_112` | Fail closed unless a channel map explicitly names current-board side-to-move channels and says which channels may be zeroed for null centering. Learned encoder may consume all channels only after adapter validation. |
| `lc0_bt4_112` | Fail closed unless a channel map distinguishes current-board channels from history channels. History channels must not be deterministically toggled or interpreted unless documented. Learned adapters may consume history planes, but null-centering and turn-toggling must be limited to validated current-board semantics. |

### Pseudocode

```text
forward(x):
    adapter.validate(x)

    x_tau = adapter.toggle_side_to_move(x)

    x_null = adapter.null_non_turn_channels(x)
    x_null_tau = adapter.toggle_side_to_move(x_null)

    x4 = concat_batch(x, x_tau, x_null, x_null_tau)
    h4 = shared_encoder(x4)

    h, h_tau, h_null, h_null_tau = split_batch(h4, 4)

    odd = 0.5 * (h - h_tau)
    even = 0.5 * (h + h_tau)
    null_odd = 0.5 * (h_null - h_null_tau)

    centered_odd = odd - null_odd

    z_avg = spatial_mean(centered_odd)
    z_max = spatial_max(centered_odd)
    z_l2 = spatial_rms(centered_odd)
    z = concat_feature(z_avg, z_max, z_l2)

    logits = classifier(z)

    if return_aux:
        return {
            "logits": logits,
            "even_probe_logits": even_probe(pool(even)),
            "null_odd_norm": rms(null_odd),
            "centered_odd_norm": rms(centered_odd),
        }

    return logits
```

The default return value is logits and remains compatible with the shared trainer.

## 8. Loss, Training, And Regularization

Primary loss:

- Weighted cross-entropy over coarse binary labels.
- Use the repository's existing balanced class weighting.

Optional auxiliary losses:

- `lambda_null * mean(||null_odd||_2^2)`, default `0.0`.
- Audit-only even probe CE, default disabled. If enabled, log it separately; do not use it to select the main model unless explicitly part of an ablation.

Class weighting:

- `balanced` as in the existing benchmark configs.

Batch size expectations:

- Start with `batch_size: 512` on CPU/GPU if memory allows.
- Because the model encodes `4B` tensors, reduce to `256` if activation memory is high.
- Keep the same effective batch size across main and central ablations when possible.

Optimizer defaults:

- AdamW.
- Learning rate: `0.001`.
- Weight decay: `0.0001`.
- Epochs: `3` for the minimal benchmark.
- Early stopping patience: `2`.

Regularizers:

- Weight decay as above.
- Dropout `0.1` in the final MLP only, optional.
- Avoid heavy data augmentation in the first run; it would confound the central intervention.

Determinism requirements:

- Seed `42`.
- `deterministic: true`.
- Fixed data split paths.
- Fixed side-to-move adapter channel.
- Log the exact adapter validation result.

What must stay unchanged for a fair comparison:

- Train/val/test split.
- Coarse-label mapping.
- Evaluation code and metrics.
- Confusion matrix and prediction artifact format.
- Class weighting policy.
- Epoch budget in the first pass.
- No full 45M-row dataset.
- No engine-derived or verification-derived features.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| **Random-turn twin central falsifier** | Replace `tau(x)` with a twin whose side-to-move plane is randomly assigned or batch-shuffled while preserving the side-to-move marginal and four-pass compute. | The true paired turn intervention matters, not just extra views or stochastic regularization. | If it matches the main model, abandon the tempo-odd mechanism. |
| Identity twin | Set `tau(x)=x`, making the odd component zero except numerical noise. | The model actually uses the odd channel. | If performance remains high, the implementation is leaking information outside `centered_odd`. |
| Uncentered odd | Use `O(x)` without subtracting `O(nu(x))`. | Null-centering removes pure side-to-move shortcut. | If uncentered wins mainly by side-to-move prior or worse class-`1` behavior, centering is justified. If uncentered cleanly dominates, the centering assumption may be too aggressive. |
| Even-only probe | Classify from `E(x)` instead of `C(x)`. | Static board appearance alone is weaker for puzzle-likeness than tempo interaction. | If even-only dominates, the dataset may reward static/source artifacts more than tactical tempo. |
| Siamese concat control | Feed `concat(pool(h), pool(h_tau))` to an MLP with no odd/even split. | The projection, not just two encoder evaluations, is useful. | If concat matches main model, the mathematical bottleneck is unnecessary. |
| Turn-blind CNN | Zero the side-to-move plane and train the same encoder/head once. | Side-to-move-conditioned signal matters. | If turn-blind matches, puzzle-likeness in this split may be mostly static. |
| Side-only model | Zero all non-turn channels and train/log a side-only classifier. | Measures raw side-to-move prior. | If side-only is strong, report it as a dataset artifact risk and require matched-side evaluation. |
| Null-centering shuffled | Keep null centering but shuffle `x_null` turn bits independently from `x`. | The null subtraction must correspond to the same turn bit. | If no effect, pure-turn centering may not matter. |
| File-mirror augmentation control | Add safe file-mirror augmentation to a baseline CNN, with correct castling/en-passant remapping if implemented. | Checks whether gains are just ordinary symmetry augmentation. | If mirror augmentation alone matches, the selected operator is not needed. |
| Matched-parameter CNN | Increase a baseline CNN to roughly match the parameter count and four-pass wall-clock budget. | Tests against capacity/compute confounding. | If matched CNN wins, the bottleneck may be overly restrictive. |

No graph, hypergraph, sheaf, transport plan, move-set, or search surrogate is introduced, so the graph/transport/move-set hard controls from prior families are not directly applicable. The random-turn twin is the semantics-destroying control for this proposal: it preserves compute, tensor shape, batch side-to-move marginal, and head capacity while destroying the paired intervention semantics.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- Existing `simple_18` simple CNN.
- Existing `simple_18` residual CNN with comparable parameter count if available.
- Existing `lc0_static_112` or `lc0_bt4_112` CNN/residual CNN results as context only, not as the primary fairness comparison.
- Matched-parameter single-pass CNN ablation.
- Siamese concat control.
- Random-turn twin central falsifier.
- Even-only and uncentered-odd ablations.

Metrics to inspect:

- Test accuracy.
- Balanced accuracy.
- AUROC.
- AUPRC.
- F1 for coarse class `1`.
- Calibration summary if already supported.
- Required `3x2` fine-label diagnostic matrix.
- Class-`1` recall and precision.
- Fine-label `1` recall at matched fine-label `0` false-positive rate.

Near-puzzle diagnostic:

- Choose one operating threshold on validation such that fine-label `0` false-positive rate matches the best existing `simple_18` CNN baseline within ±0.5 percentage points.
- On test, report fine-label `1` recall at that threshold.
- Also report fine-label `2` recall at the same threshold.

Required artifacts:

- Main model config YAML.
- Training logs.
- Validation and test metrics JSON.
- Prediction Parquet/CSV with `fine_label`, `coarse_label`, `logit_0`, `logit_1`, `prob_puzzle`, and `pred`.
- Binary confusion matrix.
- Required fine-label `3x2` confusion matrix.
- Same artifacts for every central ablation.
- Report table comparing main model, baselines, and ablations.
- Adapter validation log proving no unsupported channels were toggled.

Success threshold:

- Primary: improve fine-label `1` recall at matched fine-label `0` false-positive rate by at least `+3.0` absolute percentage points over the best existing `simple_18` baseline.
- Secondary: improve AUROC by at least `+0.010` or AUPRC by at least `+0.015` without losing more than `0.005` absolute balanced accuracy.
- Central ablation requirement: the random-turn twin must lose at least half of the main model's gain over the matched baseline.

Failure threshold:

- Main model is within noise of the matched baseline and random-turn twin.
- Fine-label `1` recall does not improve at matched fine-label `0` false-positive rate.
- Side-only or even-only probes explain most of the gain.
- Adapter requires undocumented channel assumptions.

Abandon the idea if:

- Random-turn twin or Siamese concat matches the main model on the near-puzzle diagnostic.
- Uncentered odd wins only because of a side-to-move prior.
- Identity twin performs above chance, indicating an implementation leak.
- The model improves aggregate accuracy but worsens fine-label `1` behavior.

Justify scaling if:

- The main model beats the matched baseline on near-puzzle recall and AUROC/AUPRC.
- The random-turn and even-only controls fail clearly.
- Gains survive at least two seeds.
- The fine-label `3x2` matrix shows improvement on labels `1` and `2` without a large fine-label `0` false-positive increase.

## 11. Implementation Plan For Codex

Use idea id `20260421_0729`.

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_0729_tempo_odd_bottleneck/idea.yaml` | Create | Machine-readable summary of the idea, status, config path, model path, and central falsification ablation. |
| `ideas/20260421_0729_tempo_odd_bottleneck/math_thesis.md` | Create | Section 6 math thesis, projection lemma, proof sketch, hypotheses, and counterexamples. |
| `ideas/20260421_0729_tempo_odd_bottleneck/architecture.md` | Create | Section 7 architecture details, tensor shapes, adapter rules, and pseudocode. |
| `ideas/20260421_0729_tempo_odd_bottleneck/implementation_notes.md` | Create | Adapter validation, fail-closed behavior, chunking notes, and trainer compatibility rules. |
| `ideas/20260421_0729_tempo_odd_bottleneck/trainer_notes.md` | Create | Loss, optimizer defaults, deterministic settings, and fair comparison constraints. |
| `ideas/20260421_0729_tempo_odd_bottleneck/ablations.md` | Create | Ablation table and required artifacts for each ablation. |
| `ideas/20260421_0729_tempo_odd_bottleneck/train.py` | Create | Thin wrapper around the existing trainer/config path; no custom data loading unless needed for optional ablations. |
| `ideas/20260421_0729_tempo_odd_bottleneck/config.yaml` | Create | Minimal `simple_18` config using `centered_tempo_odd_bottleneck`. |
| `ideas/20260421_0729_tempo_odd_bottleneck/report_template.md` | Create | Report skeleton with metrics, `3x2` matrices, ablation comparison, and decision outcome. |
| `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this packet's fingerprint to imported research memory after implementation; add anti-duplicate rule for side-to-move odd/even/null-centered turn-intervention bottlenecks if it fails. |
| `src/chess_nn_playground/models/tempo_odd_bottleneck.py` | Create | `Simple18TurnAdapter`, `TempoNullBuilder`, `SharedBoardEncoder`, and `CenteredTempoOddBottleneckNet`. |
| `src/chess_nn_playground/models/registry.py` | Update | Register builder function `centered_tempo_odd_bottleneck`. |
| `configs/tempo_odd_bottleneck_simple18.yaml` | Create | Shared benchmark config pointing at the current train/val/test split. |
| `tests/test_tempo_odd_bottleneck.py` | Create | Focused tests: forward shape, `tau(tau(x)) == x`, non-turn channels unchanged by toggle, null builder zeros only non-turn channels, centered odd anti-invariance, unsupported LC0 layout fails closed. |
| `tests/test_model_registry.py` | Update if needed | Ensure the new registry name constructs a model returning `(B,2)` logits. |

Codex should not implement Stockfish calls, legal move generation, move counts, source-label features, or full-dataset streaming as part of this idea.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0729_tuesday_pacific_tempo_odd_bottleneck.md
  generated_at: 2026-04-21T07:29:00-07:00
  weekday: tuesday
  timezone: pacific
  idea_slug: tempo_odd_bottleneck
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_0729_tempo_odd_bottleneck
  name: Centered Tempo-Odd Interventional Bottleneck
  slug: tempo_odd_bottleneck
  status: draft
  created_at: 2026-04-21T07:29:00-07:00
  author: ChatGPT Pro
  short_thesis: Predict puzzle-likeness from the board-dependent anti-invariant response to toggling side-to-move, after subtracting pure side-to-move effects.
  novelty_claim: Uses a fixed C2 side-to-move intervention projection and null-board centering rather than another CNN scale, move-delta bag, attack graph, sheaf, OT plan, ordinal head, sparse witness, ray automaton, constellation interaction, or pseudo-likelihood.
  expected_advantage: Better fine-label-1 near-puzzle recall at matched fine-label-0 false-positive rate by suppressing static and pure-turn shortcuts.
  central_falsification_ablation: Random-turn or batch-shuffled twin with identical compute and side-to-move marginal.
  target_task: coarse_binary
  input_representation: simple_18_first
  output_heads: coarse_binary_logits
  compute_notes: Four shared encoder passes per position via concatenated batch; no generated candidate set; expected 0.35M-0.55M parameters.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/tempo_odd_bottleneck_simple18.yaml
  model_path: src/chess_nn_playground/models/tempo_odd_bottleneck.py
  latest_result_path: null
  notes: Fail closed for lc0_static_112 and lc0_bt4_112 unless explicit channel maps validate side-to-move and current-board semantics.
```

```yaml
config_yaml:
  run:
    name: tempo_odd_bottleneck_simple18
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
    name: centered_tempo_odd_bottleneck
    input_channels: 18
    num_classes: 2
    encoding: simple_18
    stm_channel: 12
    encoder_width: 96
    encoder_blocks: 2
    mlp_hidden: 192
    use_null_centering: true
    return_aux: false
    encoder_chunk_size: null
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
  model_name: centered_tempo_odd_bottleneck
  file_path: src/chess_nn_playground/models/tempo_odd_bottleneck.py
  builder_function: build_centered_tempo_odd_bottleneck
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18TurnAdapter
    - TempoNullBuilder
    - SharedBoardEncoder
    - CenteredTempoOddBottleneckNet
  required_config_fields:
    - input_channels
    - num_classes
    - encoding
    - stm_channel
    - encoder_width
    - encoder_blocks
    - mlp_hidden
    - use_null_centering
  expected_parameter_count: 0.35M-0.55M
  expected_memory_notes: Encodes concat([x, tau(x), nu(x), tau(nu(x))]) as a 4B batch; no move/candidate memory; add encoder_chunk_size for low-memory devices.
```

```yaml
research_continuity:
  idea_fingerprint: current-board tensor plus deterministic side-to-move involution, shared encoder, odd/even C2 split, null-board centering of pure-turn effects, binary puzzle-likeness target, no engine metadata, no legal move tree, no one-ply move-delta bag.
  already_researched_family_overlap: Adjacent only to broad counterfactual representation learning; not an imported one-ply move-delta set, tactical sheaf/Hodge graph, OT/Sinkhorn transport, nuisance projection, ordinal ladder, sparse witness, ray automaton, constellation ANOVA, or pseudo-likelihood model.
  closest_duplicate_risk: Could be mistaken for simple Siamese augmentation or a generic contrastive CNN; distinguish by the fixed centered anti-invariant projection and central random-turn twin falsifier.
  do_not_repeat_if_this_fails:
    - side-to-move odd/even projection bottlenecks
    - null-board centered turn-intervention classifiers
    - Siamese turn-toggle contrast models without move generation
    - pure-turn shortcut suppression via subtracting empty-board odd response
  suggested_next_search_directions:
    - label-safe selective prediction for near-puzzle ambiguity
    - source-artifact audits that do not use source labels as model inputs
    - masked generative compression only if clearly distinguished from class-conditioned pseudo-likelihood ratios
    - causal invariance across deterministic material-phase environments with strong ablations
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Centered Tempo-Odd Interventional Bottleneck` to imported research memory after implementation, including its fingerprint. | Prevents future packets from repeating side-to-move odd/even/null-centered Siamese models. | `Imported Research Memory` |
| Add anti-duplicate wording: "Do not propose another side-to-move toggle, tempo-odd/even decomposition, or null-board-centered turn-intervention bottleneck unless the intervention variable or falsifiable operator is genuinely different." | Makes the next research cycle avoid near-renamed copies if this fails. | Anti-duplicate paragraph after counterfactual move-delta family |
| Add a requirement that any artificial counterfactual tensor intervention must include an out-of-distribution critique and a semantics-destroying random-intervention ablation. | The main risk here is that artificial twins create exploitable artifacts; future ideas should control that explicitly. | `Ablation Plan` requirements |
| Add a prompt note that deterministic channel adapters for LC0 encodings must fail closed when channel semantics are unknown. | Prevents accidental toggling of history or non-current-board channels. | `Problem Restatement And Data Contract` |
| Preserve all engine/leakage prohibitions unchanged. | The selected idea does not require weakening any safety or label rules. | Non-negotiable constraints |

## 14. Final Sanity Check

- Downloadable Markdown file created: Yes.
- Filename follows required date/time/day/timezone/slug pattern: Yes, `chess_nn_research_2026-04-21_0729_tuesday_pacific_tempo_odd_bottleneck.md`.
- No forbidden engine features used as inputs: Yes.
- Does not fabricate labels: Yes.
- Not a routine CNN/ResNet/Transformer variant: Yes.
- Minimal current-data experiment exists: Yes, `simple_18` on the existing `crtk_sample_3class` split.
- Falsification criterion is concrete: Yes, the random-turn/batch-shuffled twin must lose the claimed gain.
- Codex can implement without asking for missing architecture details: Yes.
- Prompt maintenance notes included for Codex: Yes.
- Repetition check against imported research packets completed: Yes.
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: Yes.
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: Yes.
- Not a deterministic nuisance-orthogonal projection bottleneck variant: Yes.
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Möbius-constellation, or pseudo-likelihood packets: Yes.
