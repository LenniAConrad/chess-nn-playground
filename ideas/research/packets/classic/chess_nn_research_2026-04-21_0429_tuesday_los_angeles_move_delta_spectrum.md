# Codex Handoff Packet: Counterfactual Move-Delta Spectrum Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0429_tuesday_los_angeles_move_delta_spectrum.md`
- Generated at: `2026-04-21 04:29:03 America/Los_Angeles`
- Weekday: `tuesday`
- Timezone: `los_angeles`
- Idea slug: `move_delta_spectrum`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Counterfactual Move-Delta Spectrum Network
- One-sentence thesis: A chess position is puzzle-like when the rule-only one-ply board-delta neighborhood of the side to move has a low-dimensional, highly anisotropic learned finite-difference spectrum, meaning a small number of candidate deltas create unusually concentrated latent consequences without using engine analysis.
- Idea fingerprint: `current board -> pseudo-legal side-to-move move-delta set -> learned per-delta response vectors -> masked covariance/eigen-spectrum + DeepSets summary -> binary puzzle-likeness logits`.
- Why this is not a common CNN/ResNet/Transformer variant: The CNN is only a board feature stem; the central operator is a deterministic, rule-only counterfactual action-neighborhood map followed by a finite-difference covariance spectrum over candidate move deltas, not deeper convolutions, a standard residual stack, a 64-square ViT, a square graph, LC0 imitation, or an attack-defense sheaf.
- Current-data minimal experiment: Train `MoveDeltaSpectrumNet` on the existing `simple_18` split for 3 epochs with the shared binary trainer and compare against the existing simple CNN and residual CNN under the same split, class weighting, seed, batch size, and report pipeline.
- Smallest central falsification ablation: Replace every generated move-delta token by a degree-preserving, per-position randomized token set that keeps the same number of tokens and same mover-piece/source-square marginals but destroys legal destination semantics; if this ablation matches the main model, the one-ply move-delta spectrum is not carrying the claimed signal.
- Expected information gain if it fails: A clean failure would show that current-board pseudo-legal action-neighborhood spectra do not add useful puzzle signal beyond ordinary board appearance on this split, allowing future cycles to avoid one-ply counterfactual spectrum, latent move-energy, and move-set covariance variants.

## 3. Problem Restatement And Data Contract

The task is binary chess puzzle-likeness classification from a single board-position tensor. The model receives a tensor shaped `(batch, C, 8, 8)` and returns logits shaped `(batch, 2)`. The binary target is:

- output `0`: non-puzzle
- output `1`: puzzle-like

The available fine labels are:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

For the binary benchmark, fine label `0` maps to output `0`, and fine labels `1` and `2` map to output `1`. Reports must continue to include the rectangular diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Current encodings available in the project are:

- `simple_18`: 12 piece planes + side-to-move + castling + en-passant
- `lc0_static_112`
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists

Benchmark split to use for the minimal experiment:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

The trainer must not be pointed directly at the roughly 45M-row full Parquet dataset until streaming support exists.

Leakage checklist:

- Allowed as neural-network inputs or deterministic internal features:
  - the provided board tensor;
  - deterministic board coordinates;
  - piece occupancy;
  - side to move;
  - castling and en-passant planes already present in the encoding;
  - pseudo-legal current-board move-delta geometry derived only from the current board and side to move.
- Forbidden as neural-network inputs:
  - Stockfish scores, PVs, node counts, mate scores, engine verification traces;
  - source labels, proposed labels, unresolved candidate-pool flags, verification metadata, dataset provenance;
  - any feature derived from the target fine label or binary label.
- Not used in this idea:
  - full legal-move filtering by self-check;
  - checkmate or stalemate oracles;
  - forced-line search;
  - two-ply or deeper move-tree consequences;
  - engine evaluation of candidate moves.
- Leakage-prone unless separately justified and ablated:
  - full legal-move generation;
  - raw legal move counts;
  - check status, checkmate status, stalemate status;
  - any search-like consequence label.

Boundary between safe rule-derived features and leakage:

- Deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack or move geometry derived only from the current board are allowed.
- Full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences are leakage-prone unless explicitly justified as rule-only, label-independent, engine-free, and ablated.
- Engine evaluation, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, and dataset provenance must never be neural-network inputs.
- For `lc0_static_112` and `lc0_bt4_112`, deterministic geometry may only be derived from explicitly identified current-board channels. History channels, if present, may be consumed only by learned neural adapters and must not feed the move-delta enumerator. If channel semantics are unknown, the adapter must raise a clear error rather than guessing.

This packet’s first experiment should use `simple_18`, because its current-board semantics are compact and should be easiest to bind safely. LC0 variants are optional follow-ups only after Codex verifies the current-board channel map.

## 4. Research Map

| Source | URL | What is borrowed | What is not copied |
|---|---|---|---|
| Zaheer et al., “Deep Sets,” NeurIPS 2017 | https://arxiv.org/abs/1703.06114 | The theorem-level motivation that functions over unordered sets should be permutation invariant and can be modeled by set summaries such as `rho(sum phi(x_i))`. | The task, architecture details, and benchmark domains are not copied; this idea uses a chess-specific pseudo-legal move-delta set and adds a covariance spectrum operator. |
| Lee et al., “Set Transformer,” ICML 2019 | https://arxiv.org/abs/1810.00825 | The idea that attention can operate on unordered instance sets if a future variant needs richer move-token interactions. | The minimal proposal does not use a vanilla Transformer over 64 squares and does not require Set Transformer blocks; default pooling is masked moments plus small DeepSets summaries. |
| Tishby, Pereira, and Bialek, “The Information Bottleneck Method” | https://arxiv.org/abs/physics/0004057 | The compression principle: retain label-relevant information while discouraging arbitrary representation capacity. | This proposal does not implement the original Blahut-Arimoto method and does not claim a true mutual-information estimate. |
| Alemi et al., “Deep Variational Information Bottleneck” | https://arxiv.org/abs/1612.00410 | The practical lesson that neural bottleneck regularization can improve generalization; here it motivates a small trace penalty on the move-response covariance. | The model is not a stochastic VIB and does not add a variational posterior unless Codex later tests that as a separate idea. |
| LeCun et al., “A Tutorial on Energy-Based Learning,” 2006 | https://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf | The scalar-energy/partition-function language for describing a learned distribution over candidate deltas. | The actual training remains ordinary supervised cross-entropy; there is no structured EBM inference loop and no negative sampling from an engine. |
| Arjovsky et al., “Invariant Risk Minimization,” 2019 | https://arxiv.org/abs/1907.02893 | The caution that correlations stable across environments are more valuable than source-specific artifacts; this motivates artifact-aware diagnostics by material phase or source if already available for reporting, not as model input. | This proposal does not train an IRM objective or use environment/source labels as inputs. |
| Schölkopf et al., “Towards Causal Representation Learning,” Proceedings of the IEEE 2021 | https://arxiv.org/abs/2102.11107 | The high-level framing that useful representations should align with intervention-like variables; the move-delta set is an engine-free intervention neighborhood. | No causal identifiability claim is made; the packet explicitly separates proven invariances from the hypothesis that the operator captures tactical causality. |
| Mouli, Zhou, and Ribeiro, “Bias Challenges in Counterfactual Data Augmentation,” 2022 | https://arxiv.org/abs/2209.05104 | The warning that counterfactuals can introduce bias if treated as label-preserving data. | This proposal does not create new labeled positions from moved boards and does not assume that a one-ply successor has the same puzzle label. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN over `simple_18` | `src/chess_nn_playground/models/trunk/cnn.py` | Already exists and tests ordinary local pattern recognition without a new chess-specific operator. |
| Residual CNN over `simple_18` | `src/chess_nn_playground/models/trunk/residual_cnn.py` | Already exists and mostly adds depth/optimization capacity rather than a new falsifiable inductive bias. |
| LC0-style CNN or residual CNN over `lc0_bt4_112` | Existing LC0 BT4-style CNN/residual variants | Too close to the current LC0-inspired baseline family and does not test a new mechanism. |
| Ordinary ViT over 64 board squares | Common square-token Transformer | Too ordinary, too parameter-hungry for the current benchmark, and specifically disallowed as a core idea. |
| Plain GNN on 64 squares with adjacency edges | Standard board graph neural network | Mostly renames spatial message passing and is too close to generic graph modeling without a tactical operator. |
| Static attack-defense graph, sheaf, Hodge, Laplacian, curvature, or tension-energy variant | Imported research packets under `ideas/research/packets/classic/` | Explicitly already researched; adding edge types or renamed tension terms would duplicate the imported sheaf family. |
| Hyperparameter tuning of width, depth, optimizer, or schedule | Any existing CNN/residual config | Disallowed as a research idea and unlikely to teach whether puzzle-likeness requires a new representation. |
| Ensembling existing models | Leaderboard-level ensemble | Disallowed as the core idea and would obscure rather than clarify the value of a new mechanism. |
| Stockfish-score, PV, mate-score, or node-count features | Engine-assisted puzzle filters | Leaky and forbidden; it would classify the verification process rather than board puzzle-likeness. |
| Fabricating labels for unresolved candidate pools | Semi-supervised pseudo-labeling | Forbidden because unresolved candidates must remain unresolved and cannot be treated as verified near-puzzles or puzzles. |
| Full legal move-count classifier | Rule-feature MLP | Raw legal move counts and check/mate oracles are leakage-prone and would test game-state quirks rather than tactical puzzle structure. |
| Scaling to the full 45M-row Parquet immediately | Data-only scaling baseline | Current trainer lacks streaming support; this would violate the data-contract warning and is not a new idea. |

## 6. Mathematical Thesis

### Input space definition

Let \(e\) denote an encoding family. The model input space is

\[
\mathcal X_e \subset \mathbb R^{C_e \times 8 \times 8}.
\]

For `simple_18`, an adapter \(A_e\) maps \(x \in \mathcal X_e\) to a current board state

\[
B(x) = (O(x), s(x), c(x), p_{\mathrm{ep}}(x)),
\]

where \(O(x)\in\{0,1\}^{12\times 8\times 8}\) is piece occupancy by color and piece type, \(s(x)\in\{\mathrm{white},\mathrm{black}\}\) is side to move, \(c(x)\) are castling-right indicators already present in the input, and \(p_{\mathrm{ep}}(x)\) is the en-passant indicator already present in the input.

For LC0 encodings, \(A_e\) is only defined after Codex supplies an explicit current-board channel map. Otherwise \(A_e\) is undefined and the model must fail closed.

### Label/target definition

The fine label is \(Y_f \in \{0,1,2\}\). The binary target is

\[
Y = \mathbf 1[Y_f \in \{1,2\}].
\]

The model predicts logits \(\ell_\theta(x)\in\mathbb R^2\) and is trained by cross-entropy against \(Y\).

### Data distribution assumptions

The benchmark samples are drawn from an empirical distribution \(P_{\mathrm{split}}(X,Y_f)\) defined by the existing train/validation/test Parquet splits. Fine label `1` is expected to be semantically between clear non-puzzles and verified puzzles, but the training objective must not fabricate additional ordinal or soft labels. Any ordinal analysis is diagnostic only unless implemented as an auxiliary head trained from the existing fine labels.

### Allowed symmetry or equivariance assumptions

Chess is not invariant under arbitrary rotations or reflections because pawns, castling, en-passant, side to move, and board orientation matter. The model should therefore use learned coordinate embeddings and should not impose full dihedral \(D_4\) equivariance.

The only safe optional symmetry is a rule-preserving color-perspective transform that swaps colors, flips ranks, and swaps side to move consistently. This transform should be used only as an augmentation or diagnostic after verifying castling and en-passant semantics. The minimal experiment does not require this transform.

### Core hypothesis

Let \(\mathcal A(x)\) be the set of pseudo-legal, side-to-move, one-ply move deltas generated from \(B(x)\) by current-board piece movement rules, without engine evaluation, without self-check filtering, and without checkmate/stalemate detection.

The central hypothesis is:

\[
P(Y=1\mid X=x)
\]

is better approximated by a function of both the board representation and the anisotropic spectrum of learned finite-difference responses over \(\mathcal A(x)\) than by the board representation alone.

Informally, a puzzle-like position often has a small set of moves whose local board consequences are sharply different from the rest of the candidate move neighborhood. Non-puzzles may have many mundane moves with diffuse or low-amplitude learned consequences. This is a hypothesis about puzzle-likeness, not a theorem about chess tactics.

### Formal object introduced by the idea

For each \(a \in \mathcal A(x)\), define a deterministic move-delta descriptor

\[
\eta(x,a) =
(\mathrm{from}(a), \mathrm{to}(a), \mathrm{piece}(a), \mathrm{captured}(x,a), \Delta r(a), \Delta f(a), \mathrm{flags}(x,a)).
\]

A neural board stem produces square features

\[
H_\theta(x)\in \mathbb R^{64\times d}
\]

and a global feature

\[
g_\theta(x)\in\mathbb R^{d_g}.
\]

A move-response network produces

\[
r_\theta(x,a)
=
\psi_\theta\!\left(
H_\theta(x)_{\mathrm{from}(a)},
H_\theta(x)_{\mathrm{to}(a)},
H_\theta(x)_{\mathrm{to}(a)}-H_\theta(x)_{\mathrm{from}(a)},
g_\theta(x),
\eta(x,a)
\right)
\in\mathbb R^k.
\]

Let \(n(x)=|\mathcal A(x)|\), and define uniform weights \(w_a=1/n(x)\). The masked mean and covariance are

\[
\bar r_\theta(x)=\sum_{a\in\mathcal A(x)} w_a r_\theta(x,a),
\]

\[
K_\theta(x)=
\sum_{a\in\mathcal A(x)}
w_a
\left(r_\theta(x,a)-\bar r_\theta(x)\right)
\left(r_\theta(x,a)-\bar r_\theta(x)\right)^\top
+\varepsilon I_k.
\]

Let

\[
\lambda_1(x)\ge \lambda_2(x)\ge \cdots \ge \lambda_k(x)\ge 0
\]

be the eigenvalues of \(K_\theta(x)\). The classifier receives \(g_\theta(x)\), \(\bar r_\theta(x)\), masked max-pooled move responses, and spectral features such as:

\[
\mathrm{trace}(K),\quad
\frac{\lambda_1}{\mathrm{trace}(K)},\quad
\frac{(\mathrm{trace}(K))^2}{\mathrm{trace}(K^2)},\quad
-\sum_i \tilde\lambda_i \log(\tilde\lambda_i+\varepsilon),
\quad \tilde\lambda_i=\frac{\lambda_i}{\mathrm{trace}(K)+\varepsilon}.
\]

### Variational principle and optimization objective

Train with

\[
\min_\theta
\mathbb E_{(x,y)}
\left[
\mathrm{CE}(\ell_\theta(x),y)
\right]
+
\beta
\mathbb E_x[\mathrm{trace}(K_\theta(x))].
\]

The cross-entropy term rewards label prediction. The small trace penalty is a finite-difference bottleneck: it discourages the model from making every move response arbitrarily large while still allowing anisotropic spectra when they improve classification.

Default \(\beta\) should be small, for example \(10^{-4}\), and must be ablated with \(\beta=0\).

### Proposition

For any fixed board \(x\), \(K_\theta(x)\) is invariant to the enumeration order of \(\mathcal A(x)\). Moreover,

\[
\frac{\lambda_1(K_\theta(x))}{\mathrm{trace}(K_\theta(x))}
=
\max_{\|v\|_2=1}
\frac{
\mathrm{Var}_{a\sim U(\mathcal A(x))}
\left[v^\top r_\theta(x,a)\right]
}{
\mathbb E_{a\sim U(\mathcal A(x))}
\left[
\|r_\theta(x,a)-\bar r_\theta(x)\|_2^2
\right]
+\varepsilon k
}.
\]

Thus the leading-eigenvalue fraction measures the largest share of total move-response variation explained by a single learned counterfactual direction.

### Proof sketch or derivation

The order invariance follows because \(\bar r_\theta(x)\) and \(K_\theta(x)\) are sums over the set \(\mathcal A(x)\) with symmetric weights. Reordering the candidate moves does not change the sums.

For the spectral identity, the Rayleigh-Ritz theorem gives

\[
\lambda_1(K)=\max_{\|v\|_2=1} v^\top K v.
\]

Expanding \(v^\top K v\) yields the variance of the scalar projected responses \(v^\top r_\theta(x,a)\) under the uniform action distribution, plus the small \(\varepsilon\) term. The trace is the sum of coordinate variances, equal to the expected squared norm of centered responses plus \(\varepsilon k\). The ratio therefore measures the fraction of total centered move-response energy captured by the best one-dimensional projection.

### What is actually proven

- The covariance spectrum is permutation invariant over move-token enumeration.
- The leading eigenvalue fraction has a precise variational meaning as maximum normalized directional variance.
- The operator can be computed from current-board inputs and pseudo-legal rule-derived deltas without engine metadata.
- A model using masked mean/covariance/max summaries cannot exploit padding order when implemented correctly.

### What remains only hypothesized

- Puzzle-like positions have more useful low-dimensional move-delta spectra than non-puzzles.
- Fine label `1` near-puzzles will show intermediate or more ambiguous spectral signatures.
- The rule-only pseudo-legal neighborhood is a better inductive bias than pure board appearance on the current split.
- The trace bottleneck suppresses material/source artifacts rather than suppressing tactical signal.
- Ignoring self-check in move generation is beneficial noise rather than harmful noise.

### Counterexamples where the idea should fail

- Quiet endgame studies where the puzzle hinges on long zugzwang, triangulation, or opposition rather than sharp one-ply board deltas.
- Positions with many equally tactical candidate moves, producing a diffuse rather than spiked move-response spectrum.
- Non-puzzle positions with superficial tactical-looking captures or promotions that create spurious anisotropy.
- Positions where the decisive feature is checkmate legality and self-check filtering; pseudo-legal deltas may include illegal high-response moves.
- Datasets where source artifacts or material imbalance dominate labels so strongly that the board stem alone explains most variance.
- Puzzles whose key move is quiet and whose consequence only appears after several forced replies.

## 7. Architecture Specification

### Module names

- `MoveDeltaSpectrumNet`
- `CurrentBoardAdapter`
- `PseudoLegalMoveDeltaEnumerator`
- `BoardStem`
- `MoveTokenEncoder`
- `CounterfactualSpectrumPool`
- `MoveDeltaSpectrumHead`

### Forward-pass steps and tensor shapes

Assume input `x` has shape `[B, C, 8, 8]`.

1. `CurrentBoardAdapter`
   - Input: `x: [B, C, 8, 8]`
   - Output:
     - `pieces_12: [B, 12, 8, 8]`
     - `side_to_move: [B]` or `[B, 1]`
     - `rule_aux: [B, A_aux]`, including castling/en-passant indicators if available
   - The adapter must validate channel semantics. It must raise `ValueError` when the encoding is unsupported or channel order is unknown.

2. `BoardStem`
   - Suggested default:
     - `Conv2d(C, d, kernel_size=3, padding=1)`
     - two small residual or gated convolution blocks, not a deep ResNet
     - coordinate embeddings concatenated or added after projection
   - Output:
     - `H_map: [B, d, 8, 8]`, with `d=64`
     - flatten to `H_sq: [B, 64, d]`
     - global average/max pooled feature `g: [B, d_g]`, with `d_g=128`

3. `PseudoLegalMoveDeltaEnumerator`
   - Input:
     - `pieces_12`, `side_to_move`, `rule_aux`
   - Output padded to `M_max`, default `M_max=384`:
     - `from_idx: [B, M_max]`, integer square indices `0..63`
     - `to_idx: [B, M_max]`, integer square indices `0..63`
     - `move_mask: [B, M_max]`, boolean
     - `det_feat: [B, M_max, F_det]`, deterministic descriptor features
   - The enumerator generates pseudo-legal current-board move deltas only:
     - piece movement by current side to move;
     - sliding rays stopped by first occupied square;
     - captures on enemy occupied squares;
     - pawn pushes/captures based on side to move;
     - optional promotions as separate promotion flags;
     - optional en-passant only if the input en-passant plane is semantically verified;
     - optional castling deltas from castling channels, without checking attacked transit squares.
   - It must not compute engine scores, checkmate, stalemate, legal self-check filtering, PVs, or node counts.
   - It must not pass raw `n_moves` to the classifier. Masked means/covariances should normalize by the number of valid tokens.

4. Gather square features
   - `H_from = gather(H_sq, from_idx): [B, M_max, d]`
   - `H_to = gather(H_sq, to_idx): [B, M_max, d]`
   - `H_delta = H_to - H_from: [B, M_max, d]`
   - `g_broadcast: [B, M_max, d_g]`

5. `MoveTokenEncoder`
   - Concatenate:
     - `H_from`
     - `H_to`
     - `H_delta`
     - `g_broadcast`
     - `det_feat`
   - Shape before MLP: `[B, M_max, 3d + d_g + F_det]`
   - With defaults `d=64`, `d_g=128`, `F_det≈48`, input width is about `368`.
   - Token MLP:
     - Linear `368 -> 128`
     - activation
     - dropout `0.05` optional
     - Linear `128 -> k`
   - Output:
     - `R: [B, M_max, k]`, with `k=16`

6. `CounterfactualSpectrumPool`
   - Apply `move_mask` so padded tokens contribute zero.
   - Compute normalized weights \(w=1/\max(n,1)\) over valid tokens.
   - Compute:
     - `r_mean: [B, k]`
     - `r_max: [B, k]` masked max with padded tokens set to `-inf`, then safe-filled to zero if no valid token
     - `K: [B, k, k]`
     - `eigvals: [B, k]` via `torch.linalg.eigvalsh`, sorted descending
     - `spectral_stats: [B, S]`, about `S=2k+8`, including eigenvalues, normalized eigenvalues, trace, log trace, leading fraction, participation ratio, spectral entropy, and Frobenius norm
   - Complexity of eigen decomposition is `O(B*k^3)`, negligible for `k=16`.

7. `MoveDeltaSpectrumHead`
   - Concatenate:
     - `g: [B, d_g]`
     - `r_mean: [B, k]`
     - `r_max: [B, k]`
     - `spectral_stats: [B, S]`
   - Classifier MLP:
     - Linear to `128`
     - activation
     - dropout `0.1`
     - Linear to `num_classes=2`
   - Output:
     - logits `[B, 2]`, compatible with the shared trainer.

### Parameter-count estimate

With `C=18`, `d=64`, `d_g=128`, `k=16`, and a small two-block stem, expected parameters are approximately `0.35M` to `0.55M`.

With `C=112`, the first convolution grows by about `(112-18)*64*3*3 ≈ 54k` parameters, so expected parameters are approximately `0.40M` to `0.65M`, assuming the same token/head dimensions.

### FLOP or complexity estimate

For each batch:

\[
O(B \cdot 64 \cdot d^2)
+
O(B \cdot M_{\max} \cdot d \cdot 128)
+
O(B \cdot M_{\max} \cdot k^2)
+
O(B \cdot k^3).
\]

The move-token portion scales linearly in the padded move count. With `M_max=384` and `k=16`, it should be substantially cheaper than a deep residual CNN. The main runtime risk is Python-level move enumeration; Codex should vectorize where practical or cache per-batch deterministic descriptors only inside the dataloader if this does not alter the shared trainer contract.

### Required config fields

- `model.name: move_delta_spectrum`
- `model.input_channels`
- `model.num_classes: 2`
- `model.encoding`
- `model.max_moves`, default `384`
- `model.square_dim`, default `64`
- `model.global_dim`, default `128`
- `model.move_response_dim`, default `16`
- `model.token_hidden_dim`, default `128`
- `model.trace_penalty_beta`, default `0.0001`
- `model.dropout`, default `0.1`
- `model.enable_castling_deltas`, default `false` for first run unless semantics verified
- `model.enable_en_passant_deltas`, default `false` for first run unless semantics verified
- `model.fail_closed_on_unknown_channels`, default `true`
- `model.randomized_delta_ablation`, default `false`

### Encoding support

First experiment: use `simple_18`.

Adapter assumptions:

- `simple_18`:
  - deterministic geometry is allowed after Codex verifies the exact 12 piece-plane order, side-to-move channel, castling channels, and en-passant channel from the project encoder code;
  - if the order is not discoverable from code or config, require an explicit channel map in the config and fail closed without it.
- `lc0_static_112`:
  - learned `BoardStem` may ingest all channels;
  - deterministic geometry may parse only verified current-board piece planes and side-to-move channels;
  - history or repetition channels must not be used by the enumerator.
- `lc0_bt4_112`:
  - learned `BoardStem` may ingest all channels;
  - deterministic geometry may parse only the verified current-board slice;
  - zero-filled unavailable history planes must not be interpreted as actual history by the enumerator;
  - if BT4 current-board semantics are not explicit, run only the learned stem baseline or raise a clear unsupported-encoding error.

### Pseudocode

```text
forward(x):
    board = adapter(x)  # fail closed on unknown channel semantics
    H_map, H_sq, g = board_stem(x)

    moves = pseudo_legal_delta_enumerator(board)
    # moves: from_idx, to_idx, mask, det_feat

    H_from = gather_square(H_sq, moves.from_idx)
    H_to = gather_square(H_sq, moves.to_idx)
    token_in = concat(H_from, H_to, H_to - H_from, broadcast(g), moves.det_feat)

    R = move_token_mlp(token_in)
    R = mask_padded_tokens(R, moves.mask)

    r_mean, r_max, K, eigvals, spectral_stats = spectrum_pool(R, moves.mask)
    features = concat(g, r_mean, r_max, spectral_stats)

    logits = classifier(features)
    aux = {"trace_K": trace(K), "eigvals": eigvals}  # only for optional loss/reporting
    return logits
```

The shared trainer expects logits. If the existing trainer cannot handle auxiliary returns, return only logits and expose trace penalty through a model method or wrapper loss hook. Do not break existing reports, confusion matrices, predictions, or leaderboards.

## 8. Loss, Training, And Regularization

Primary loss:

\[
\mathcal L_{\mathrm{CE}}=\mathrm{CrossEntropyLoss}(\ell_\theta(x), y)
\]

with balanced class weighting computed from the training split, matching existing benchmark conventions.

Optional auxiliary loss:

\[
\mathcal L_{\mathrm{trace}}=\beta \cdot \mathbb E_x[\mathrm{trace}(K_\theta(x))]
\]

with default `beta=0.0001`. This is part of the proposed model but must be ablated at `beta=0`.

Total loss:

\[
\mathcal L = \mathcal L_{\mathrm{CE}} + \mathcal L_{\mathrm{trace}}.
\]

If the shared trainer cannot accept model auxiliary losses without invasive changes, set `beta=0` for the first smoke test and implement a minimal loss hook only after verifying compatibility.

Training defaults:

- epochs: `3`
- batch size: `512`
- optimizer: `AdamW`
- learning rate: `0.001`
- weight decay: `0.0001`
- class weighting: `balanced`
- early stopping patience: `2`
- mixed precision: `false` for the first deterministic run
- num workers: `0` for deterministic smoke tests; increase only after reproducibility tests pass

Regularizers:

- trace penalty on \(K\), default `0.0001`
- dropout `0.05` in move-token MLP and `0.1` in classifier head
- weight decay as above
- no label smoothing in the first experiment, to keep comparison with existing baselines clean

Determinism requirements:

- fixed Python, NumPy, and PyTorch seeds;
- deterministic move enumeration sorted by `(from_idx, to_idx, promotion_flag, special_flag)`;
- deterministic randomized-ablation seed derived from global seed and sample index, not dataloader order;
- no nondeterministic GPU operations if `deterministic: true`.

What must stay unchanged for a fair comparison:

- the train/validation/test split paths;
- binary label mapping;
- diagnostic fine-label `0/1/2 -> predicted 0/1` matrix;
- evaluation thresholding protocol;
- batch size, epochs, class weighting, optimizer family, learning rate, and early-stopping settings when comparing to simple/residual CNN baselines;
- no full-dataset training until streaming exists.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Degree-preserving randomized move deltas | Keeps per-position token count and mover/source-square marginals, but randomizes destination squares and move-type descriptors using a deterministic seed; same token MLP and spectrum pool. | The legal move-delta semantics, not just more tokens or parameters, carry useful puzzle signal. | If performance matches the main model, abandon the central claim that pseudo-legal move-delta spectra matter. |
| Board-stem-only head | Removes move enumeration, token encoder, covariance spectrum, and spectral stats; classifier uses only `g`. | The move-delta operator adds information beyond a same-stem CNN-like representation. | If this matches the main model, the spectrum operator is unnecessary on the current split. |
| Mean/max set pool without covariance eigen-spectrum | Keeps move tokens but replaces `K`, eigenvalues, and spectral stats with masked mean/max only. | Spectral anisotropy is specifically useful beyond generic DeepSets pooling. | If mean/max matches the main model, the covariance-spectrum thesis is too strong; future work should not repeat spectrum variants. |
| Covariance spectrum with shuffled response vectors across positions | Computes `K` from move responses sampled from other positions in the batch, preserving dimensions but destroying position-delta binding. | The spectrum must be tied to the actual board’s candidate deltas. | If this matches, the model is exploiting global response statistics or artifacts rather than local counterfactual structure. |
| `beta=0` trace penalty | Removes the finite-difference bottleneck but leaves architecture unchanged. | The trace bottleneck improves generalization or near-puzzle discrimination. | If `beta=0` is better, keep the architecture but drop bottleneck rhetoric in future packets. |
| Captures-only delta set | Enumerates only pseudo-legal captures and promotions; quiet moves removed. | Puzzle signal mostly comes from forcing tactical captures/promotions. | If captures-only wins, the full move neighborhood adds noise; if it loses badly on class `1`, quiet near-puzzle structure matters. |
| Quiet-only delta set | Removes captures and promotions; keeps quiet pseudo-legal moves. | Quiet move consequences are necessary for near-puzzles and non-obvious tactics. | If quiet-only matches full, capture semantics are not central; inspect material/source artifacts. |
| Pseudo-legal self-check noise stress test | Optional: compare pseudo-legal enumeration with full legal self-check filtering only if implemented as rule-only and separately reported. | Illegal pseudo-legal moves may corrupt the spectrum. | If full legal filtering improves strongly, future prompt must clarify whether rule-only legal filtering is allowed; do not add checkmate/stalemate features. |
| No deterministic move descriptors | Uses only gathered `H_from`, `H_to`, and `H_delta`; removes piece/move/capture flags. | The neural stem can infer enough without explicit rule descriptors. | If it matches, simplify implementation and reduce risk of hand-coded descriptor bugs. |
| Equalized material-bucket report | Not a model change; report metrics within material and phase buckets if these can be derived safely from occupancy. | Gains are not solely from material imbalance or phase artifacts. | If gains appear only in one material bucket, the method may be exploiting superficial correlations. |

The smallest central falsification ablation is the degree-preserving randomized move-delta ablation.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- existing simple CNN on `simple_18`;
- existing residual CNN on `simple_18`;
- if cheap, LC0-style CNN/residual results already present in the leaderboard, but do not require LC0 support for the first experiment;
- the board-stem-only ablation;
- the degree-preserving randomized move-delta ablation;
- the mean/max set-pool without covariance spectrum ablation.

Metrics to inspect:

- validation and test cross-entropy;
- accuracy;
- balanced accuracy;
- ROC-AUC;
- PR-AUC for binary puzzle-like classification;
- positive-class precision, recall, and F1;
- Brier score or calibration error if already supported;
- rectangular fine-label diagnostic matrix `true fine label 0/1/2 -> predicted binary output 0/1`.

Required fine-label confusion:

Codex must produce the `3x2` diagnostic matrix for:

- main `MoveDeltaSpectrumNet`;
- degree-preserving randomized move-delta ablation;
- board-stem-only ablation;
- mean/max no-spectrum ablation;
- best simple CNN/residual CNN comparison baseline if the report tool can regenerate it.

Near-puzzle diagnostic:

- On validation, choose a threshold for each model that matches a fixed fine-label-`0` false-positive rate. Suggested protocol:
  - use the best existing simple/residual CNN’s validation threshold at the standard operating point;
  - record its fine-label-`0` false-positive rate;
  - for every compared model, choose the threshold that matches this fine-label-`0` false-positive rate as closely as possible;
  - report test recall for fine label `1` and fine label `2` separately at that matched false-positive rate.
- Primary near-puzzle diagnostic: fine-label-`1` recall at matched fine-label-`0` false-positive rate.
- Secondary diagnostic: fine-label-`2` recall at the same matched false-positive rate.

Required artifacts:

- config YAML for main model and central ablations;
- trained checkpoints or final state dict references if existing benchmark stores them;
- `metrics.json`;
- predictions file containing sample id if available, fine label, binary label, predicted probability, predicted class, and split;
- rectangular `3x2` confusion matrix as CSV/JSON and report text;
- ablation comparison table;
- leaderboard update if the project has one;
- implementation notes documenting exactly which encoding channel map was used.

Success threshold:

Treat the idea as successful enough to scale if all of the following hold on the test split, using the same split and comparable training budget:

- main model improves over the best existing `simple_18` CNN/residual baseline by at least `+1.5` percentage points in PR-AUC or balanced accuracy;
- main model improves over the degree-preserving randomized ablation by at least `+2.0` percentage points in fine-label-`1` recall at matched fine-label-`0` false-positive rate;
- fine-label-`2` recall does not drop by more than `1.0` percentage point relative to the best `simple_18` CNN/residual baseline at the same matched false-positive protocol;
- gains are directionally stable across at least two seeds if compute permits.

Failure threshold:

Treat the idea as failed if:

- main model is within `±0.5` percentage points of the degree-preserving randomized ablation on PR-AUC, balanced accuracy, and fine-label-`1` matched-FPR recall;
- or main model beats the baseline only by raising fine-label-`0` false positives without improving matched-FPR diagnostics;
- or board-stem-only matches main performance, showing no useful contribution from move deltas;
- or the implementation requires leaky features to work.

What result would make me abandon the idea:

Abandon this family if the degree-preserving randomized move-delta ablation and main model are statistically indistinguishable across at least two seeds, especially if the mean/max no-spectrum ablation also matches the main model. Do not repackage it later as move-energy entropy, move-delta curvature, one-ply counterfactual covariance, or latent tactical action spectrum.

What result would justify scaling:

Scale only if the main model clears the success threshold and the randomized semantics-destroying ablation is clearly worse. Next scaling steps would be longer training, LC0 current-board adapter support, and possible cached vectorized move descriptors, not full-dataset training until streaming is available.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_move_delta_spectrum/idea.yaml` | Create | Machine-readable idea metadata from the `idea_yaml` block below, plus links to configs, reports, and status. |
| `ideas/20260421_move_delta_spectrum/math_thesis.md` | Create | Section 6 mathematical thesis, including definitions of \(\mathcal A(x)\), \(r_\theta(x,a)\), \(K_\theta(x)\), the Rayleigh quotient proposition, and falsifiable hypotheses. |
| `ideas/20260421_move_delta_spectrum/architecture.md` | Create | Section 7 architecture details, tensor shapes, adapter assumptions, module responsibilities, and pseudocode. |
| `ideas/20260421_move_delta_spectrum/implementation_notes.md` | Create | Practical notes on channel-map validation, pseudo-legal enumeration, no engine leakage, sorted move order, padding masks, and deterministic random ablations. |
| `ideas/20260421_move_delta_spectrum/trainer_notes.md` | Create | Loss integration, optional trace penalty hook, fair comparison settings, deterministic training requirements, and artifact expectations. |
| `ideas/20260421_move_delta_spectrum/ablations.md` | Create | Section 9 ablation table and exact central randomized-delta protocol. |
| `ideas/20260421_move_delta_spectrum/train.py` | Create | Thin entrypoint that delegates to the shared trainer or imports the standard training script with this idea’s config; do not fork the trainer unless unavoidable. |
| `ideas/20260421_move_delta_spectrum/config.yaml` | Create | Main experiment config based on the `config_yaml` block, with additional model fields from Section 7 if the project config schema supports them. |
| `ideas/20260421_move_delta_spectrum/report_template.md` | Create | Report template requiring metrics, `3x2` fine-label matrix, matched-FPR near-puzzle diagnostic, central ablations, and leakage checklist. |
| `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this packet to imported research memory after implementation, including anti-duplicate rules for one-ply move-delta spectra if it fails. Preserve hard leakage, label, falsification, and anti-duplicate constraints. |
| `src/chess_nn_playground/models/move_delta_spectrum.py` | Create | Implement `MoveDeltaSpectrumNet`, adapters, pseudo-legal delta enumerator, token encoder, spectrum pool, classifier head, and optional trace-loss accessors. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `move_delta_spectrum` builder without breaking existing model names. |
| `configs/move_delta_spectrum_simple18.yaml` | Create | Main benchmark config for `simple_18`, 3 epochs, balanced class weighting, deterministic seed `42`, batch size `512`, and no LC0 assumptions. |
| `configs/move_delta_spectrum_randomized_ablation_simple18.yaml` | Create | Same as main config but with `model.randomized_delta_ablation: true`. |
| `configs/move_delta_spectrum_no_spectrum_simple18.yaml` | Create | Same as main config but disables covariance/eigen-spectrum and uses mean/max set pooling only. |
| `configs/move_delta_spectrum_board_stem_only_simple18.yaml` | Create | Same stem and head capacity but removes move enumeration and spectrum. |
| `tests/test_move_delta_spectrum.py` | Create | Shape tests for logits `[B,2]`, deterministic output under fixed seed, masking tests, and registry builder test. |
| `tests/test_move_delta_enumerator.py` | Create | Rule-only pseudo-legal smoke tests from hand-constructed tensors: knight moves, pawn pushes/captures, sliding blockers, no own-piece capture, deterministic ordering, no raw move-count output to classifier. |
| `tests/test_encoding_adapters_fail_closed.py` | Create | Verify unsupported or unknown channel maps raise clear errors; verify `simple_18` adapter only after explicit channel semantics are configured or discovered. |

For `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md`, Codex should update the prompt after consuming this output. The update should preserve hard constraints while adding reusable lessons, anti-duplicate rules, and failure-mode guidance from this research pass.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0429_tuesday_los_angeles_move_delta_spectrum.md
  generated_at: "2026-04-21 04:29:03 America/Los_Angeles"
  weekday: tuesday
  timezone: los_angeles
  idea_slug: move_delta_spectrum
  format: markdown
```

```yaml
idea_yaml:
  idea_id: "20260421_move_delta_spectrum"
  name: "Counterfactual Move-Delta Spectrum Network"
  slug: move_delta_spectrum
  status: draft
  created_at: "2026-04-21 04:29:03 America/Los_Angeles"
  author: ChatGPT Pro
  short_thesis: "Puzzle-like positions have an anisotropic learned finite-difference spectrum over the side-to-move pseudo-legal one-ply board-delta neighborhood."
  novelty_claim: "Introduces a rule-only move-delta set operator plus covariance/eigen-spectrum pooling, outside CNN/ResNet/LC0/ViT/sheaf families."
  expected_advantage: "Better near-puzzle recall at matched non-puzzle false-positive rate by exposing sharp one-ply counterfactual structure without engine leakage."
  central_falsification_ablation: "Degree-preserving randomized move-delta tokens preserving token count and source-piece marginals while destroying legal destination semantics."
  target_task: coarse_binary
  input_representation: "simple_18 primary; lc0_static_112 and lc0_bt4_112 only with verified current-board channel maps and fail-closed adapters"
  output_heads: binary_logits
  compute_notes: "Roughly 0.35M-0.55M parameters on simple_18; token cost O(B*M_max*d*hidden), spectrum cost O(B*k^3) with k=16."
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/move_delta_spectrum_simple18.yaml
  model_path: src/chess_nn_playground/models/move_delta_spectrum.py
  latest_result_path: null
  notes: "Do not use engine scores, PVs, checkmate/stalemate oracles, source labels, proposed labels, or unresolved candidate flags. First run should avoid LC0 geometry unless channel maps are explicit."
```

```yaml
config_yaml:
  run:
    name: move_delta_spectrum_simple18
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
    name: move_delta_spectrum
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
  model_name: move_delta_spectrum
  file_path: src/chess_nn_playground/models/move_delta_spectrum.py
  builder_function: build_move_delta_spectrum
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - CurrentBoardAdapter
    - PseudoLegalMoveDeltaEnumerator
    - BoardStem
    - MoveTokenEncoder
    - CounterfactualSpectrumPool
    - MoveDeltaSpectrumHead
  required_config_fields:
    - model.name
    - model.input_channels
    - model.num_classes
    - data.encoding
    - model.max_moves
    - model.square_dim
    - model.global_dim
    - model.move_response_dim
    - model.trace_penalty_beta
    - model.fail_closed_on_unknown_channels
  expected_parameter_count: "0.35M-0.55M for simple_18 defaults; about 0.40M-0.65M for LC0 encodings if enabled."
  expected_memory_notes: "Move token tensor is B*M_max*k after compression; with B=512, M_max=384, k=16 it is about 3.1M floats before pooled stats. Watch pre-MLP token concatenation memory and prefer chunking if needed."
```

```yaml
research_continuity:
  idea_fingerprint: "current-board pseudo-legal side-to-move move-delta set + learned per-delta finite-difference responses + covariance/eigen-spectrum pooling + binary puzzle-likeness target"
  already_researched_family_overlap: "Uses current-board rule geometry but not attack/defense incidence, sheaf restrictions, Hodge operators, Laplacians, curvature, or tension energies."
  closest_duplicate_risk: "Could be mistaken for a generic DeepSets-over-moves model; the central novelty is the finite-difference covariance spectrum and its degree-preserving randomized falsification."
  do_not_repeat_if_this_fails:
    - "one-ply move-delta covariance/eigen-spectrum"
    - "latent move-response anisotropy as puzzle-likeness"
    - "move-energy entropy or partition-function variants over the same pseudo-legal delta set"
    - "renamed counterfactual move curvature without a different operator"
    - "degree-preserving randomized move-token ablations as if they were a new idea"
  suggested_next_search_directions:
    - "causal invariance across encoding families or material/phase environments without using source labels as inputs"
    - "label-safe ordinal uncertainty for separating fine label 1 from fine label 2 without fabricating labels"
    - "material-artifact information bottlenecks that do not enumerate moves"
    - "optimal transport over material imbalances or piece-square occupancy distributions, not attack-defense sheaves"
    - "calibration and abstention mechanisms for ambiguous near-puzzles"
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Counterfactual Move-Delta Spectrum Network` to the imported research memory after implementation. | Prevents future ChatGPT Pro passes from recycling one-ply move-delta covariance/eigen-spectrum ideas under new names. | `Imported Research Memory` |
| Add anti-duplicate fingerprint: `pseudo-legal side-to-move move-delta set + learned finite-difference response covariance/eigen-spectrum + binary puzzle-likeness`. | Captures the real mechanism, not just the title, so future ideas can avoid near-duplicates. | `Imported Research Memory` or `Research Continuity` |
| Clarify that one-ply pseudo-legal move-delta enumeration is allowed only when rule-only, current-board-only, engine-free, and accompanied by a semantics-destroying ablation. | Keeps the safe boundary between deterministic chess rules and leakage-prone move/search features. | `Problem Context You Must Respect` and `Non-Negotiable Constraints` |
| Require degree-preserving randomized ablations for any future structured candidate-set operator. | Prevents models from winning only because they receive more tokens, move counts, or parameter capacity. | `Required Markdown File Content`, especially `Ablation Plan` |
| Add adapter fail-closed language for encodings whose current-board channel semantics are unknown. | Reduces accidental leakage or silent misuse of LC0 history/static channels. | `Project Context You Must Respect` |
| If this idea fails, add a rule saying not to propose move-energy entropy, move-delta curvature, or one-ply counterfactual spectrum variants unless the operator is mathematically different. | Avoids wasting cycles on cosmetic variants of the same failed mechanism. | `Research Continuity` |
| Add a benchmark requirement for fine-label-`1` recall at matched fine-label-`0` false-positive rate. | Makes near-puzzle usefulness explicit rather than hiding it inside aggregate binary accuracy. | `Required Markdown File Content` section `Benchmark And Falsification Criteria` |
| Keep the existing hard bans on engine features, fabricated labels, unresolved candidates, ordinary larger CNNs, standard ResNets, vanilla Transformers, ensembles, and optimizer tuning. | These constraints are still necessary and should not be weakened by this packet. | `Non-Negotiable Constraints` |

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0429_tuesday_los_angeles_move_delta_spectrum.md`
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes, `simple_18` on the existing `crtk_sample_3class` split
- Falsification criterion is concrete: yes, degree-preserving randomized move-delta ablation
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported research packets completed: yes, this is not an attack-defense graph, tactical sheaf, Hodge/sheaf-Laplacian, curvature, or tension-energy variant
