# Codex Handoff Packet: Masked Board Code-Length Surprise Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0739_tuesday_los_angeles_masked_surprise_codec.md`
- Generated at: `2026-04-21T07:39:09-07:00`
- Weekday: Tuesday
- Timezone: America/Los_Angeles
- Idea slug: `masked_surprise_codec`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: **Masked Board Code-Length Surprise Network** (`MBCS-Net`)
- One-sentence thesis: Train a label-free masked board codec to estimate how many nats it takes to reconstruct each hidden square from the rest of the current board, then classify puzzle-likeness from the resulting spatial code-length/uncertainty field plus the original encoding.
- Idea fingerprint: `current-board tensor -> deterministic piece-token map -> label-free masked conditional codec q(piece_at_square | visible board, mask) -> mask-averaged per-square code-length and entropy planes -> small classifier -> binary puzzle-like logits`.
- Why this is not a common CNN/ResNet/Transformer variant: The central observable is a learned **conditional description-length field** produced by a self-supervised board compressor before the supervised classifier sees the position; performance should disappear if that field is replaced by a unigram/material prior or square-shuffled while the CNN classifier remains unchanged.
- Current-data minimal experiment: Use `simple_18` only, pretrain the masked codec for 3 epochs on `split_train.parquet` with labels ignored, freeze it, train the classifier for 3 epochs on the standard coarse binary task, and report the normal test metrics plus the required `3x2` fine-label diagnostic matrix.
- Smallest central falsification ablation: Replace the trained conditional codec with a label-free square/piece/material unigram codec that preserves square identity, side-to-move, global material bucket, occupancy, and fixed mask coverage but has no conditional board context; keep the classifier, optimizer, parameter budget, and surprise-plane interface unchanged.
- Expected information gain if it fails: A clean failure would rule out a broad “puzzles are locally conditionally surprising board states” hypothesis for the current CRTK sample, which is meaningfully different from ruling out attack graphs, one-ply move landscapes, transport bottlenecks, ordinal heads, or ordinary CNN scaling.

## 3. Problem Restatement And Data Contract

The task is chess position classification from a single current board position. The shared benchmark is binary:

- fine label `0`: known non-puzzle -> binary output `0`
- fine label `1`: verified near-puzzle -> binary output `1`
- fine label `2`: verified puzzle -> binary output `1`

Reports must continue to include the rectangular diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Allowed encodings already present in the project:

- `simple_18`: 12 piece planes + side-to-move + castling + en-passant
- `lc0_static_112`
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists

Default split paths:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

The model must be a PyTorch `torch.nn.Module` accepting:

```text
(batch, C, 8, 8)
```

and returning logits:

```text
(batch, num_classes)
```

with `num_classes = 2` for the default benchmark. The full Parquet dataset is roughly 45M rows and must not be used directly by the current trainer until streaming support exists.

Leakage checklist for this idea:

- Safe neural inputs: current-board piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, deterministic rank/file coordinate planes appended inside the model if configured, and the model's own label-free masked reconstruction code lengths.
- Safe rule-derived features: deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack geometry derived only from the current board are allowed in general. This specific idea does **not** need attack geometry.
- Leakage-prone unless separately justified and ablated: full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, move-tree consequences, or any future-board consequences. This packet does not use them.
- Always forbidden as neural-network inputs: Stockfish or other engine evaluations, principal variations, node counts, mate scores, verification metadata, source labels, proposed labels, unresolved candidate-pool status, dataset provenance, and any label-derived feature.
- `lc0_static_112` and `lc0_bt4_112` boundary: current-board channels may be used for deterministic piece-token extraction only when an explicit channel schema is configured; history channels may be consumed only by a learned neural adapter and must never be used as deterministic reconstruction targets unless Codex can prove they represent the current board. If the schema is unknown, the codec path must fail closed.

## 4. Research Map

### External ideas used

1. **Masked language modeling, BERT**: Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” 2018, <https://arxiv.org/abs/1810.04805>. Borrowed: hiding parts of an input and reconstructing them from bidirectional context. Not copied: no Transformer over text, no next-sentence objective, no wordpiece tokenization, and no vanilla 64-square ViT.
2. **Masked autoencoders for vision**: He et al., “Masked Autoencoders Are Scalable Vision Learners,” 2021/2022, <https://arxiv.org/abs/2111.06377>. Borrowed: a label-free masked reconstruction task can induce useful representations. Not copied: no image patch ViT, no high-mask-ratio ImageNet recipe, and no claim that image MAE results transfer automatically to chess.
3. **Minimum description length and stochastic complexity**: Barron, Rissanen, and Yu, “The Minimum Description Length Principle in Coding and Modeling,” 1998, <https://www.stat.yale.edu/~arb4/publications_files/TheMinimumDescriptionLengthPrincipleInCodingAndModelingIEEEIT.pdf>. Borrowed: negative log probability is interpretable as code length in nats/bits, and regularity corresponds to compressibility. Not copied: no MDL model-selection theorem is claimed for puzzle labels.
4. **Information bottleneck framing**: Tishby, Pereira, and Bialek, “The information bottleneck method,” 2000, <https://arxiv.org/abs/physics/0004057>, and Alemi et al., “Deep Variational Information Bottleneck,” 2016, <https://arxiv.org/abs/1612.00410>. Borrowed: the useful signal may be a compressed statistic of the board rather than the full board. Not copied: no stochastic latent VIB is the central operator; the bottleneck here is a deterministic code-length field from a frozen codec.
5. **Rejected causal/group robustness references**: Arjovsky et al., “Invariant Risk Minimization,” 2019, <https://arxiv.org/abs/1907.02893>, and Sagawa et al., “Distributionally Robust Neural Networks for Group Shifts,” 2019, <https://arxiv.org/abs/1911.08731>. Borrowed only as candidate-search context: invariance and worst-group training are plausible future directions. Not copied into the selected mechanism.
6. **Rejected evidential uncertainty reference**: Sensoy, Kaplan, and Kandemir, “Evidential Deep Learning to Quantify Classification Uncertainty,” 2018, <https://papers.nips.cc/paper/7580-evidential-deep-learning-to-quantify-classification-uncertainty>. Borrowed only as candidate-search context for near-puzzle ambiguity. Not copied into the selected mechanism.

### Candidate search trace

The internal screen covered at least these 12 families: masked board compression, encoding-family invariance, group-DRO over rule-derived environments, evidential/credal near-puzzle heads, selective prediction, supervised contrastive class-1 bridge embeddings, chess partial-equivariant convolutions, legal-symmetry consistency losses, causal anti-source adapters, spectral signed board potentials, masked denoising diffusion over board tokens, and persistent topology of tactical relations. The selected idea is the masked compression family because it is outside the imported sheaf/move-delta/OT/nuisance/ordinal/sparse/ray/constellation/pseudo-likelihood packets while still producing a falsifiable chess-specific observable.

Serious candidates not selected:

- **Encoding-invariant causal adapter**: Train adapters for `simple_18`, `lc0_static_112`, and `lc0_bt4_112` to agree in a shared latent. It lost because current shared trainers are simplest with one encoding at a time, and a negative result would be hard to separate from adapter/schema bugs.
- **Rule-derived environment IRM/GDRO**: Define environments by material phase, side-to-move, castling status, and king-file buckets, then minimize worst-group or invariant risk. It lost because it is primarily a training objective, is close to generic anti-nuisance methods, and may duplicate the spirit of nuisance suppression even without closed-form projection.
- **Evidential near-puzzle uncertainty head**: Treat fine label `1` as positive but lower-confidence evidence. It lost because it is mostly a calibration/loss idea and risks being too close to the imported ordinal-evidence family unless paired with a new board operator.
- **Masked diffusion over board tokens**: Learn denoising transitions over piece-token boards and use denoising residuals for classification. It lost to the simpler codec because diffusion introduces more moving parts, more compute, and weaker immediate falsification.
- **Persistent topology over attacks or pins**: Compute topological summaries of an attack/defense relation. It lost because it is too adjacent to the imported tactical sheaf/Hodge/attack-incidence families.
- **Spectral signed board potential**: Build a hand-designed signed potential field over piece values and king zones, then classify its spectrum. It lost because it looked like either a static attack graph in disguise or a weaker version of the imported transport/sheaf ideas.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN on `simple_18` | `src/chess_nn_playground/models/cnn.py` | Already exists and tests ordinary local convolution without a new mathematical observable. |
| Residual CNN variants | `src/chess_nn_playground/models/residual_cnn.py` | Already exists; deeper residual capacity would be routine architecture scaling. |
| LC0-style CNN/residual CNN | LC0 BT4-style CNN and residual variants | Already exists; copying LC0-like channel processing is not a fresh puzzle-likeness hypothesis. |
| Ordinary ViT over 64 squares | None or future Transformer baseline | Too generic; the prompt explicitly excludes vanilla square Transformers as the core idea. |
| Plain GNN on 64 squares | Common square-grid GNN baseline | Usually becomes a slower CNN or a generic message-passing model without a chess-specific falsifier. |
| Hyperparameter tuning | All current baselines | Tuning width, depth, learning rate, or optimizer does not produce an original research mechanism. |
| Ensembling | Any set of existing models | Likely improves leaderboard numbers without teaching anything about puzzle structure. |
| Another attack-defense sheaf/Hodge/tension model | Imported tactical sheaf/Hodge packets | Directly covered by imported research memory and forbidden unless the operator changes radically. |
| Another one-ply move-delta set or spectrum | Imported counterfactual move-delta packets | Already researched; also closer to move consequences than this pass should use. |
| Another Sinkhorn or piece-target transport bottleneck | Imported optimal-transport packets | Already researched; changing costs or targets would be a near-duplicate. |
| Deterministic nuisance-vector projection | Imported nuisance-orthogonal packet | Already researched; the selected idea uses a label-free codec, not closed-form latent residualization. |
| Ordinal cumulative heads | Imported ordinal evidence ladder | Too close to the existing ordinal packet and not enough of a new board operator. |
| Sparse witness/top-k pieces | Imported sparse witness bottleneck | Already researched; mask-based code length covers all squares and is not a top-k occupied-piece selector. |
| Ray-language automata | Imported ray-language automaton | Already researched; the selected masks reconstruct piece tokens rather than scanning ray strings. |
| High-order ANOVA/Möbius constellations | Imported constellation packet | Already researched; code length is conditional predictive surprise, not explicit degree-2/3 interactions. |
| Class-conditioned pseudo-likelihood ratio | Imported geometry-conditioned pseudo-likelihood packet | Too close; this packet deliberately uses a single label-free masked codec and a code-length field, not a class-conditional likelihood ratio. |

## 6. Mathematical Thesis

### Input space definition

Let the board square set be

\[
S = \{0,1,\ldots,63\},\qquad |S|=64.
\]

For an encoding with `C` channels, the neural input is

\[
x \in \mathcal X_C \subset \mathbb R^{C\times 8\times 8}.
\]

For `simple_18`, define a deterministic tokenizer

\[
T: \mathcal X_{18}\to \mathcal A^S,
\]

where

\[
\mathcal A=\{\text{empty}, WP,WN,WB,WR,WQ,WK,BP,BN,BB,BR,BQ,BK\}.
\]

`T(x)_s` is extracted only from the 12 piece planes. Side-to-move, castling, and en-passant planes remain available as context channels but are not predicted as piece tokens. For LC0 encodings, `T` is undefined unless an explicit current-board piece-plane schema is supplied; otherwise the codec must raise a configuration error.

### Label/target definition

The fine label is

\[
a\in\{0,1,2\}.
\]

The default binary target is

\[
y=\mathbf 1[a\in\{1,2\}].
\]

Fine label `1` is not fabricated or reinterpreted. It is positive for the coarse binary task and separately reported in diagnostics.

### Data distribution assumptions

Training, validation, and test rows are treated as samples from the provided CRTK split distribution. The full 45M-row Parquet file is out of scope until streaming exists. The split may contain source artifacts and class-selection biases. The selected idea assumes only this testable hypothesis:

> Some puzzle-like or near-puzzle positions contain localized board tokens whose identity is unusually hard to predict from the rest of the current board under an ordinary board-state compressor, and those localized conditional-surprise patterns contain label signal not captured efficiently by a small supervised CNN alone.

This is a hypothesis, not a theorem about chess tactics.

### Allowed symmetry or equivariance assumptions

No full rotation or reflection invariance is assumed. Chess is not invariant under arbitrary board symmetries because pawns, castling, side-to-move, and en-passant break many symmetries. The default model may use convolutional locality and zero-padding, plus optional deterministic coordinate planes `(rank, file, center distance, promotion direction relative to side-to-move)` so that the network can distinguish board regions. Any file-mirror augmentation must swap castling-side metadata correctly and should be disabled in the minimal experiment unless an existing safe transform already exists.

### Core hypothesis

Let \(M\subseteq S\) be a mask sampled from a fixed mask bank \(\mu\). Let \(x_{\setminus M}\) be the board input with the piece planes on masked squares zeroed and a binary mask indicator appended. Train a codec

\[
q_\theta(a\mid x_{\setminus M},M,s),\qquad a\in\mathcal A,
\]

to predict the hidden token \(T(x)_s\) for each \(s\in M\) without using labels.

Define the per-mask code length

\[
\ell_\theta(s,x,M)=-\log q_\theta(T(x)_s\mid x_{\setminus M},M,s).
\]

For a finite mask bank \(\mathcal M=\{M_1,\ldots,M_K\}\) covering every square at least once, define the mask-averaged code-length field

\[
S_\theta(s,x)=\frac{\sum_{k=1}^K \mathbf 1[s\in M_k]\ell_\theta(s,x,M_k)}{\sum_{k=1}^K \mathbf 1[s\in M_k]}.
\]

Also define an entropy field

\[
H_\theta(s,x)=\frac{\sum_{k=1}^K \mathbf 1[s\in M_k]\left[-\sum_{a\in\mathcal A}q_\theta(a\mid x_{\setminus M_k},M_k,s)\log q_\theta(a\mid x_{\setminus M_k},M_k,s)\right]}{\sum_{k=1}^K \mathbf 1[s\in M_k]}.
\]

The classifier receives

\[
\Phi_\theta(x)=\operatorname{concat}(x,S_\theta(x),H_\theta(x),P_\theta(x)),
\]

where \(P_\theta(s,x)=q_\theta(T(x)_s\mid x_{\setminus M},M,s)\) averaged over covering masks or its logit equivalent. The supervised network learns

\[
f_\psi(\Phi_\theta(x))\in\mathbb R^2.
\]

### Variational principle

The codec is trained by the label-free objective

\[
\mathcal L_{\text{codec}}(\theta)=\mathbb E_{x\sim D_{\text{train}},M\sim\mu}\left[\frac{1}{|M|}\sum_{s\in M}-\log q_\theta(T(x)_s\mid x_{\setminus M},M,s)\right].
\]

For a fixed mask distribution, if the model class contains the true conditional distribution

\[
p(a\mid x_{\setminus M},M,s),
\]

then the population minimizer is \(q_{\theta^*}=p\) almost everywhere. The excess objective is

\[
\mathcal L_{\text{codec}}(q)-\mathcal L_{\text{codec}}(p)
=\mathbb E_{x,M,s}\left[\operatorname{KL}\left(p(\cdot\mid x_{\setminus M},M,s)\,\|\,q(\cdot\mid x_{\setminus M},M,s)\right)\right]\ge 0.
\]

Thus, an ideal codec's negative log probability is the optimal conditional code length for transmitting a hidden square token from its visible board context.

### Proposition

**Proposition: Conditional-code optimality and accessible sparse-surprise signal.** Assume:

1. The codec is frozen after label-free training and approximates \(p(T_s\mid x_{\setminus M},M,s)\).
2. There exists a sparse set-valued statistic \(A(x)\subseteq S\), \(|A(x)|\le r\), such that the conditional log odds of the binary label can be written or approximated as

\[
\log\frac{P(y=1\mid x)}{P(y=0\mid x)}\approx \alpha + g\left(\{S_{\theta^*}(s,x),H_{\theta^*}(s,x):s\in S\}\right),
\]

where \(g\) is a permutation-sensitive but continuous board pooling function representable by the small classifier.
3. The corresponding signal is not recoverable from token-square base rates alone.

Then a classifier on \(\Phi_{\theta^*}(x)\) can approximate the Bayes decision rule within the approximation error of \(g\), and the unigram/material-prior codec ablation should lose performance.

### Proof sketch or derivation

The first part follows from the standard cross-entropy decomposition: expected negative log likelihood equals conditional entropy plus an expected KL divergence from the true conditional to the codec. Therefore, the optimum codec produces a conditional code length. The second part is conditional on the sparse-surprise model: if log odds factor through the code-length and entropy fields, then a sufficiently expressive board pooling classifier on those fields can approximate the Bayes log odds. The ablation follows because a unigram/material prior removes dependence on \(x_{\setminus M}\), so it cannot represent any label signal that specifically comes from conditional inconsistency between a square and the rest of the board.

### What is actually proven

- The masked reconstruction objective is a proper scoring rule for the hidden piece-token conditional distribution.
- The resulting negative log probabilities are valid conditional code lengths under the learned codec.
- The construction is label-free up to the supervised classifier and uses only the current board tensor.
- If the label log odds truly factor through these code-length fields, then a classifier using them can represent the Bayes decision rule.

### What remains only hypothesized

- That puzzle-like positions in this dataset are conditionally surprising in the proposed sense.
- That near-puzzles have a measurable intermediate or distinctive code-length profile.
- That the codec learns chess-relevant contextual regularities rather than superficial source artifacts.
- That a 3-epoch codec pretrain on the current split is enough to produce useful code-length estimates.

### Counterexamples where the idea should fail

- A quiet positional puzzle whose decisive feature is a legal-move tactic but whose static board tokens are ordinary and easy to reconstruct.
- A non-puzzle position from an unusual opening, underpromotion, tablebase-like endgame, or corrupted sampling source that is highly surprising but not puzzle-like.
- A dataset where labels are determined mostly by engine verification thresholds, not by any local static motif.
- A codec that mostly learns square-piece base frequencies, material patterns, or side-to-move priors and never learns conditional structure.
- LC0 encodings with ambiguous channel semantics accidentally mapped to wrong piece tokens; this must fail closed, not silently train.

### Self-critique

The strongest objection is that chess tactics are about legal continuations, not about static board improbability. A spectacular tactic can occur in a completely natural position, and a weird board can be tactically dead. This idea survives as a minimal experiment because it tests a different claim: not “all tactics are rare,” but “the puzzle-selection process leaves spatial conditional-compression residue that a small CNN does not efficiently extract.” The central ablations are harsh. If a square/material prior or square-shuffled code-length map matches the main model, the mechanism should be considered falsified rather than rescued by bigger classifiers.

## 7. Architecture Specification

### Module names

Implement the main model in:

```text
src/chess_nn_playground/models/masked_surprise_codec.py
```

Recommended classes/functions:

- `Simple18PieceTokenizer`
- `EncodingChannelSpec`
- `MaskBank2x2Residues`
- `MaskedBoardCodec`
- `CodeLengthFieldBuilder`
- `SurpriseResidualClassifier`
- `MaskedBoardCodeLengthSurpriseNet`
- builder function: `build_masked_board_code_length_surprise_net(config)`

### Forward-pass steps

Default minimal experiment: `encoding=simple_18`, `input_channels=18`, `num_classes=2`, `mask_bank=2x2_residue`, `num_masks=4`, `codec_width=32`, `classifier_width=64`.

Input:

```text
x: (B, C, 8, 8)
```

Step 1: validate encoding.

```text
simple_18: require C == 18 and piece_channel_order length == 12
lc0_*: require explicit current-piece channel schema or raise ValueError
```

Step 2: extract hidden reconstruction targets.

```text
tokens = tokenizer(x)  # (B, 8, 8), int64 in [0, 12]
```

For `simple_18`, `tokens` is `empty` when all 12 piece planes are zero on a square; otherwise it is the active piece plane plus one. If multiple piece planes are active on one square, either raise in strict mode or use argmax and log a data-quality warning in non-strict mode.

Step 3: build finite mask bank.

Default `2x2_residue` masks:

```text
M_k = {(rank, file): rank % 2 == r_k and file % 2 == f_k}, k=1..4
```

Each mask covers 16 squares and every square is masked exactly once across the four masks. This gives fixed mask count and fixed mask coverage, preventing candidate-count shortcuts.

Step 4: build code-length fields, chunking masks if needed.

For each mask chunk of size `J <= mask_chunk_size`:

```text
mask:          (J, 1, 8, 8)
x_masked:      (J*B, C, 8, 8)     # piece planes zeroed under mask, metadata preserved
mask_plane:    (J*B, 1, 8, 8)
codec_input:   (J*B, C+1, 8, 8)
codec_hidden:  (J*B, D_codec, 8, 8)
token_logits:  (J*B, 13, 8, 8)
ce_map:        (J*B, 1, 8, 8)     # -log p(true token)
entropy_map:   (J*B, 1, 8, 8)
ptrue_map:     (J*B, 1, 8, 8)
```

Scatter only masked squares into accumulators. After all masks:

```text
S:      (B, 1, 8, 8)  # mean code length, nats
H:      (B, 1, 8, 8)  # predictive entropy
Ptrue:  (B, 1, 8, 8)  # predicted probability of actual token
```

Use `surprise_clip_nats=8.0` and transform `S_scaled = log1p(clamp(S, 0, 8))` unless disabled.

Step 5: append deterministic coordinate planes if configured.

```text
coords: (B, num_coord_planes, 8, 8)
```

Recommended coordinate planes: normalized rank, normalized file, center distance, and side-to-move-relative promotion direction. Coordinates are deterministic from the board grid and side-to-move, not label-derived.

Step 6: classifier input.

```text
x_aug = concat(x, S_scaled, H, Ptrue, coords)  # (B, C+3+coord_planes, 8, 8)
```

Default: detach `S_scaled`, `H`, and `Ptrue` during classifier training when the codec is frozen. If Codex implements joint fine-tuning, it must report a separate ablation because joint tuning can make the codec a supervised side channel rather than a label-free compressor.

Step 7: classifier trunk.

A small residual classifier is acceptable because the research operator is the code-length field:

```text
Conv3x3(in=C+3+coord_planes, out=64)
4 x residual block width 64
Global average pooling + global max pooling
Linear(128, 2)
```

Output:

```text
logits: (B, 2)
```

### Parameter-count estimate

With `codec_width=32`, `codec_blocks=3`, `classifier_width=64`, `classifier_blocks=4`, `coord_planes=4`, and `simple_18`:

- Masked codec stem: about 5.5k parameters.
- Masked codec residual blocks: about 55k parameters.
- Token decoder: about 0.4k parameters.
- Classifier stem: about 14.4k parameters for 25 input planes to width 64.
- Classifier residual blocks: about 295k parameters.
- Pool/head/norms: about 1k-5k parameters depending on BatchNorm/GroupNorm.
- Expected total: roughly `0.37M-0.45M` trainable parameters when codec and classifier are both counted; only about `0.31M-0.38M` trainable during frozen-codec classifier training.

### FLOP or complexity estimate

For 8x8 convolutions, default rough multiply-add counts per sample:

- One codec pass: about 4M MACs.
- Four masks: about 16M MACs.
- Classifier trunk: about 20M MACs.
- Total frozen-codec forward: about 36M MACs per sample.

Because masks are chunked, peak activation memory should be manageable even when compute is higher than a plain CNN. If throughput is poor, use `mask_chunk_size=1`, lower `batch_size` to 256, or cache frozen surprise fields for the train/val/test splits as an optional experiment artifact. Do not cache features from the full 45M dataset.

### Candidate-set memory estimate and chunking plan

There is no legal-move candidate set. The generated set is the fixed mask bank. Memory scales as:

```text
O(B * J * (C + D_codec + 13) * 8 * 8)
```

where:

- `B` = batch size
- `J` = `mask_chunk_size`
- `C` = input channels
- `D_codec` = codec width
- `13` = piece-token logits

Default `J=2`; use `J=1` on CPU or small GPU. Accumulators for `S`, `H`, `Ptrue`, and coverage are only `(B, 4, 8, 8)` total and negligible.

### Required config fields

```yaml
model:
  name: masked_board_code_length_surprise_net
  input_channels: 18
  num_classes: 2
  encoding: simple_18
  piece_channel_order: [WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK]
  strict_tokenizer: true
  codec_width: 32
  codec_blocks: 3
  classifier_width: 64
  classifier_blocks: 4
  mask_bank: 2x2_residue
  num_masks: 4
  mask_chunk_size: 2
  surprise_planes: [code_length, entropy, p_true]
  surprise_clip_nats: 8.0
  append_coord_planes: true
  freeze_codec: true
  detach_surprise: true
  codec_checkpoint: null
```

### Encoding-adapter assumptions

- `simple_18`: Fully supported in the first experiment. Token extraction is deterministic from the 12 piece planes.
- `lc0_static_112`: Supported only if Codex has a trusted mapping from current-board piece planes to the 13-token target. Learned classifier adapters may consume all 112 channels, but the masked codec's target must come only from current-board occupancy channels. Unknown schema -> fail closed.
- `lc0_bt4_112`: Same as `lc0_static_112`, with additional caution that history planes are zero-filled until exporter support exists. History planes must not be reconstruction targets. Unknown schema -> fail closed.

### Pseudocode

```text
forward(x):
    validate_shape_and_encoding(x)
    tokens = tokenizer.extract_piece_tokens(x)          # B,8,8
    S_accum = zeros(B,1,8,8)
    H_accum = zeros(B,1,8,8)
    P_accum = zeros(B,1,8,8)
    coverage = zeros(B,1,8,8)

    for masks in mask_bank.chunks(mask_chunk_size):
        x_rep = repeat_for_masks(x, masks)
        x_masked = zero_piece_planes_where_masked(x_rep, masks)
        codec_input = concat(x_masked, masks_as_planes)
        token_logits = codec(codec_input)               # J*B,13,8,8
        ce = gather_cross_entropy(token_logits, tokens)
        ent = categorical_entropy(token_logits)
        ptrue = gather_probability(token_logits, tokens)
        scatter_masked_values_into_accumulators(ce, ent, ptrue)

    S = S_accum / clamp_min(coverage, 1)
    H = H_accum / clamp_min(coverage, 1)
    Ptrue = P_accum / clamp_min(coverage, 1)
    S = log1p(clamp(S, 0, surprise_clip_nats))

    if detach_surprise:
        S, H, Ptrue = detach(S), detach(H), detach(Ptrue)

    coord = coord_planes(x) if append_coord_planes else empty
    x_aug = concat(x, S, H, Ptrue, coord)
    return classifier(x_aug)                            # B,2
```

The model must return plain logits so the shared trainer, reports, confusion matrices, predictions, and leaderboards keep working.

## 8. Loss, Training, And Regularization

### Primary loss

Classifier training uses standard coarse-binary cross entropy:

\[
\mathcal L_{\text{cls}}=\operatorname{CE}(f_\psi(\Phi_{\theta}(x)),y).
\]

Use balanced class weighting based on the training split's binary labels unless the existing benchmark config uses a different established default; if so, preserve the benchmark default and document it.

### Auxiliary loss

Default is two-stage, not joint:

1. **Codec pretraining**: train `MaskedBoardCodec` with labels ignored using `L_codec` from Section 6.
2. **Classifier training**: freeze codec, detach surprise planes, train classifier with `L_cls` only.

Optional joint fine-tuning after the frozen experiment:

\[
\mathcal L=\mathcal L_{\text{cls}}+\lambda_{\text{codec}}\mathcal L_{\text{codec}},\qquad \lambda_{\text{codec}}\in\{0.05,0.1\}.
\]

This is optional and must not replace the frozen-codec result because it weakens the label-free interpretation.

### Class weighting

Use `class_weighting: balanced` for binary classes. Do not create new labels. Fine label `1` remains positive for binary training and is only separated in diagnostics.

### Batch size expectations

- Codec pretrain: batch size 512 on GPU if memory allows, otherwise 256.
- Classifier train with frozen codec: batch size 512 if `mask_chunk_size <= 2`, otherwise 256.
- CPU smoke tests: batch size 2-8.

### Learning-rate and optimizer defaults

- Optimizer: AdamW.
- Codec pretrain learning rate: `1e-3`.
- Classifier learning rate: `1e-3`.
- Weight decay: `1e-4`.
- Epochs: 3 codec pretrain + 3 classifier train for the minimal experiment.
- Early stopping: validation ROC-AUC or validation loss with patience 2, matching existing benchmark convention.

### Regularizers

- Use GroupNorm or BatchNorm consistently with existing project style; GroupNorm is safer for small CPU tests.
- Dropout `0.05-0.10` before the final linear head only.
- Clip code lengths to `8.0` nats before `log1p` scaling.
- Keep fixed mask coverage to prevent mask-count leakage.
- Optional label-free token-prior normalization may be implemented only as an ablation/control, not as the central result.

### Determinism requirements

- Use seed `42` in Python, NumPy, and PyTorch.
- Fixed mask bank for the main experiment.
- Deterministic dataloader order when requested by existing configs.
- Save codec checkpoint hash and config with the classifier run.

### What must stay unchanged for fair comparison

- Same train/val/test split.
- Same binary target mapping.
- Same report scripts and required `3x2` diagnostic matrix.
- Same number of classifier epochs as baseline comparison unless the baseline suite has a standard training budget.
- No extra full-dataset training, no engine data, no unresolved candidate pool, no source/provenance features.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| **Central falsifier: unigram/material codec** | Replace conditional codec with a label-free prior `q(token | square, side-to-move, material bucket, occupancy)`; keep `S/H/Ptrue` interface and classifier unchanged. | Conditional board context, not just piece-square rarity or material composition, carries puzzle signal. | If this matches main performance, the code-length mechanism is likely only a rarity/material shortcut. |
| No-surprise classifier | Remove `S`, `H`, and `Ptrue`; train the same classifier on `x + coords` only. | The masked code-length field adds information beyond a small CNN. | If equal, the codec is unnecessary; abandon or redesign. |
| Square-shuffled surprise | For each board, randomly permute surprise values over squares while preserving the multiset of `S/H/Ptrue`, side-to-move, material, and fixed mask coverage. | Spatial alignment of conditional surprise to board squares matters. | If equal, the classifier is using global surprise histograms or shortcuts, not localized structure. |
| Random frozen codec | Use the same codec architecture with random frozen weights. | Learned conditional compression is necessary. | If equal, the classifier is exploiting deterministic transformations or scale artifacts. |
| Codec trained with labels accidentally allowed check | Pretrain codec with labels ignored and verify no label/fine-label/source columns are loaded. | No leakage in pretraining path. | Any use of labels in codec pretrain invalidates the experiment. |
| Mask coverage control | Replace 2x2 masks with another fixed bank covering each square once, such as rank-parity/file-parity masks, keeping number of masked squares constant. | Performance should not depend on a peculiar mask shape. | Large swings mean the method is brittle to masking rather than learning robust conditional surprise. |
| Surprise histogram only | Pool `S/H/Ptrue` into global mean, max, top-k mean, and variance; remove spatial maps. | Local spatial layout matters beyond board-level rarity. | If histogram matches, the mechanism is global anomaly detection, not spatial motif detection. |
| Token-prior normalized surprise | Feed `S - S_unary` where `S_unary` is a label-free square/material token prior. | Removing base-rate rarity should preserve true conditional signal. | If normalized surprise collapses, main gains may be base-rate artifacts. |
| Joint fine-tune codec | Allow gradients from classification into the codec with small auxiliary reconstruction loss. | Supervised adaptation may help after the label-free mechanism is established. | If only joint tuning works, the clean MDL interpretation is weak. |
| Fine-label-1 held-out-from-training diagnostic | Train on fine labels `0` and `2` only, evaluate class `1` as unlabeled diagnostic positives. | Near-puzzles should lie closer to puzzles if the signal is real. | If class `1` behavior is random or inverted, the method may not capture near-puzzle structure. |

No move-set or candidate legal-move object is generated. The count/nuisance-preserving controls are the fixed mask coverage, square/material unigram codec, and square-shuffled surprise map.

## 10. Benchmark And Falsification Criteria

### Baselines to compare against

At minimum compare on `simple_18`:

- existing simple CNN small/medium if available
- existing residual CNN small/medium if available
- best current `simple_18` baseline from the leaderboard

Optional after the first pass:

- LC0 static/BT4 CNN and residual baselines if safe channel schemas are available
- `MBCS-Net` with `lc0_static_112` only after fail-closed schema validation exists

### Metrics to inspect

- Test ROC-AUC.
- Test PR-AUC.
- Accuracy, balanced accuracy, F1.
- Cross-entropy / negative log likelihood.
- Brier score and expected calibration error if existing reports support them.
- Required `3x2` diagnostic matrix for every main and central ablation run.

### Required fine-label diagnostic

For the main model and at least these ablations, report:

```text
true fine label 0 -> predicted 0/1
true fine label 1 -> predicted 0/1
true fine label 2 -> predicted 0/1
```

Required ablations for this matrix:

- main `MBCS-Net`
- unigram/material codec central falsifier
- no-surprise classifier
- square-shuffled surprise

### Near-puzzle diagnostic

Use validation fine-label `0` examples to set thresholds at matched false-positive rates:

- `FPR_0 = 1%`
- `FPR_0 = 5%`
- `FPR_0 = 10%`

At each threshold, report on test:

- fine-label `1` recall
- fine-label `2` recall
- positive precision among fine labels `1` and `2`
- ratio `recall_label1 / recall_label2`

The main expected value is improved fine-label `1` recall at matched fine-label-`0` FPR, without sacrificing fine-label `2` recall.

### Required artifacts

- Codec pretrain checkpoint and config.
- Classifier checkpoint and config.
- Metrics JSON for main and ablations.
- Predictions Parquet/CSV with `fine_label`, binary target, logits/probabilities, and split identifier.
- Confusion matrices including required `3x2` matrix.
- Ablation summary table.
- Optional small visualization grid of `S/H/Ptrue` maps for correctly and incorrectly classified validation examples. Do not use this as an input feature.

### Success threshold

Consider the idea successful enough to scale if all are true:

1. Main model improves test ROC-AUC by at least `+0.010` absolute over the strongest comparable `simple_18` baseline, **or** improves fine-label `1` recall by at least `+0.020` absolute at matched `5%` fine-label-`0` FPR without reducing fine-label `2` recall by more than `0.010`.
2. The central unigram/material codec ablation loses at least `0.007` ROC-AUC or `0.015` fine-label-`1` recall at matched `5%` FPR.
3. Square-shuffled surprise loses at least `0.005` ROC-AUC or visibly worsens the `3x2` diagnostic.
4. No leakage or schema warnings occur.

### Failure threshold

Treat the idea as failed if either:

- Main model is within `±0.003` ROC-AUC of the no-surprise classifier and does not improve the near-puzzle diagnostic, or
- The unigram/material codec and square-shuffled surprise ablations match the main model within `0.003` ROC-AUC and `0.005` fine-label-`1` recall.

### What result would make me abandon the idea

Abandon this mechanism if a trained conditional codec does not beat both the no-surprise and unigram/material codec controls, especially if square-shuffled surprise is equal. That would mean conditional code-length fields are not carrying useful localized puzzle-likeness signal in the current data.

### What result would justify scaling

Scale only after the minimal `simple_18` run passes the success threshold. Then try:

- cached frozen surprise fields to reduce training cost,
- `lc0_static_112` with explicit current-board token schema,
- more codec pretraining epochs on the same split, not the full dataset,
- larger mask banks only if central controls still show structure matters.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_masked_surprise_codec/idea.yaml` | Create | Machine-readable idea metadata copied from Section 12 `idea_yaml`. |
| `ideas/20260421_masked_surprise_codec/math_thesis.md` | Create | Section 6 math thesis, including proposition, proof sketch, hypotheses, and counterexamples. |
| `ideas/20260421_masked_surprise_codec/architecture.md` | Create | Section 7 architecture spec, tensor shapes, config fields, and pseudocode. |
| `ideas/20260421_masked_surprise_codec/implementation_notes.md` | Create | Tokenizer strictness, fail-closed LC0 schema handling, mask chunking, surprise clipping, and checkpoint loading notes. |
| `ideas/20260421_masked_surprise_codec/trainer_notes.md` | Create | Two-stage training plan: codec pretrain with labels ignored, then frozen-codec classifier training with shared reports. |
| `ideas/20260421_masked_surprise_codec/ablations.md` | Create | Section 9 ablation plan and required central ablation commands. |
| `ideas/20260421_masked_surprise_codec/train.py` | Create | Thin experiment entrypoint that pretrains the codec, saves checkpoint, trains classifier, runs test evaluation, and launches required ablations. Reuse existing dataset/trainer/report utilities where possible. |
| `ideas/20260421_masked_surprise_codec/config.yaml` | Create | Concrete minimal config from Section 12 `config_yaml` plus model-specific fields. |
| `ideas/20260421_masked_surprise_codec/report_template.md` | Create | Template with metrics, `3x2` diagnostics, near-puzzle matched-FPR table, ablation comparisons, and failure/success decision. |
| `ideas/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this packet to imported research memory after consumption; add anti-duplicate guidance for masked conditional code-length surprise models if this fails. Preserve all hard leakage and label rules. |
| `src/chess_nn_playground/models/masked_surprise_codec.py` | Create | `Simple18PieceTokenizer`, `MaskedBoardCodec`, `CodeLengthFieldBuilder`, `SurpriseResidualClassifier`, `MaskedBoardCodeLengthSurpriseNet`, and builder function. |
| `src/chess_nn_playground/models/registry.py` | Modify | Register `masked_board_code_length_surprise_net` and optionally `masked_surprise_codec_classifier_only` for ablations. |
| `configs/masked_surprise_codec_simple18.yaml` | Create | Minimal benchmark config using `simple_18`, `input_channels=18`, batch size 512, 3 classifier epochs, balanced class weighting. |
| `tests/models/test_masked_surprise_codec.py` | Create | Focused tests for tokenizer, mask coverage, forward shape `(B,2)`, deterministic outputs in eval mode, fail-closed LC0 unknown schema, and no label argument in codec pretrain. |
| `tests/ideas/test_masked_surprise_codec_config.py` | Create if test layout supports it | Validate config can instantiate model and run a tiny CPU forward/backward pass. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0739_tuesday_los_angeles_masked_surprise_codec.md
  generated_at: 2026-04-21T07:39:09-07:00
  weekday: Tuesday
  timezone: America/Los_Angeles
  idea_slug: masked_surprise_codec
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_masked_surprise_codec
  name: Masked Board Code-Length Surprise Network
  slug: masked_surprise_codec
  status: draft
  created_at: 2026-04-21T07:39:09-07:00
  author: ChatGPT Pro
  short_thesis: Train a label-free masked board codec and classify from spatial conditional code-length and entropy fields plus the original current-board tensor.
  novelty_claim: Uses mask-averaged conditional description length as the central chess-board observable, not attack/sheaf incidence, one-ply move deltas, Sinkhorn transport, nuisance projection, ordinal heads, sparse witnesses, ray automata, constellations, or class-conditioned pseudo-likelihood ratios.
  expected_advantage: May expose localized static irregularities associated with puzzle and near-puzzle selection while suppressing trivial piece-square and material shortcuts through central ablations.
  central_falsification_ablation: Replace trained conditional codec with a label-free square/piece/material unigram codec while preserving the surprise-plane interface and classifier.
  target_task: coarse_binary
  input_representation: simple_18 first; lc0_static_112 and lc0_bt4_112 only with explicit current-board piece-channel schema
  output_heads: binary logits only; optional codec reconstruction head used during label-free pretraining
  compute_notes: Four fixed masks by default; chunk mask dimension; about 0.4M parameters and roughly 36M MACs per sample with default widths.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/masked_surprise_codec_simple18.yaml
  model_path: src/chess_nn_playground/models/masked_surprise_codec.py
  latest_result_path: null
  notes: Pretrain codec on train split with labels ignored, freeze it, then train classifier and required ablations on the standard split.
```

```yaml
config_yaml:
  run:
    name: masked_surprise_codec_simple18
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
    name: masked_board_code_length_surprise_net
    input_channels: 18
    num_classes: 2
    encoding: simple_18
    piece_channel_order: [WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK]
    strict_tokenizer: true
    codec_width: 32
    codec_blocks: 3
    classifier_width: 64
    classifier_blocks: 4
    mask_bank: 2x2_residue
    num_masks: 4
    mask_chunk_size: 2
    surprise_planes: [code_length, entropy, p_true]
    surprise_clip_nats: 8.0
    append_coord_planes: true
    freeze_codec: true
    detach_surprise: true
    codec_checkpoint: null
  training:
    epochs: 3
    batch_size: 512
    num_workers: 0
    learning_rate: 0.001
    weight_decay: 0.0001
    class_weighting: balanced
    early_stopping_patience: 2
    mixed_precision: false
  codec_pretrain:
    enabled: true
    epochs: 3
    batch_size: 512
    learning_rate: 0.001
    weight_decay: 0.0001
    mask_bank: 2x2_residue
    save_path: results/masked_surprise_codec_simple18/codec_pretrain.pt
  ablations:
    run_required: [unigram_material_codec, no_surprise, square_shuffled_surprise, random_frozen_codec]
```

```yaml
model_spec:
  model_name: masked_board_code_length_surprise_net
  file_path: src/chess_nn_playground/models/masked_surprise_codec.py
  builder_function: build_masked_board_code_length_surprise_net
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18PieceTokenizer
    - MaskBank2x2Residues
    - MaskedBoardCodec
    - CodeLengthFieldBuilder
    - SurpriseResidualClassifier
    - MaskedBoardCodeLengthSurpriseNet
  required_config_fields:
    - model.name
    - model.input_channels
    - model.num_classes
    - model.encoding
    - model.piece_channel_order
    - model.codec_width
    - model.classifier_width
    - model.mask_bank
    - model.freeze_codec
    - model.detach_surprise
  expected_parameter_count: 0.37M-0.45M total with default simple_18 config
  expected_memory_notes: Memory is O(batch * mask_chunk_size * (C + codec_width + 13) * 8 * 8); default fixed four masks can be processed in chunks of two or one.
```

```yaml
research_continuity:
  idea_fingerprint: current-board tensor plus label-free masked conditional piece-token codec producing mask-averaged spatial code-length/entropy/probability fields for binary puzzle-likeness classification
  already_researched_family_overlap: Adjacent only to broad MDL/generative-compression themes; deliberately not class-conditioned pseudo-likelihood ratio and not attack/sheaf, move-delta, OT, nuisance projection, ordinal, sparse-witness, ray-language, or constellation family.
  closest_duplicate_risk: Geometry-conditioned board pseudo-likelihood packet; avoid duplication by keeping a single label-free codec, using code-length fields as features, and making unigram/material and square-shuffle controls central.
  do_not_repeat_if_this_fails:
    - Mask-averaged conditional piece-token code-length fields from a frozen label-free board codec.
    - Self-supervised masked board reconstruction used only to produce surprise planes for puzzle-likeness.
    - Claims that local conditional board surprise alone explains near-puzzle recall unless a new falsifier is introduced.
    - Bigger mask banks or larger codecs without first beating unigram/material and square-shuffle controls.
  suggested_next_search_directions:
    - True causal invariance across safe encoding families with explicit schema validation.
    - Label-safe selective prediction focused on fine-label-1 ambiguity without ordinal cumulative heads.
    - Non-move-tree constrained latent-variable models that encode tactical uncertainty without masked board code length.
    - Calibration and abstention mechanisms evaluated at matched fine-label-0 false-positive rates.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Masked Board Code-Length Surprise Network` to imported research memory after implementation. | Prevents future cycles from proposing the same masked conditional code-length field under a new name. | `Imported Research Memory` |
| Add anti-duplicate fingerprint: `current board -> label-free masked piece-token reconstruction -> code-length/entropy surprise maps -> classifier`. | Distinguishes this from class-conditioned pseudo-likelihood ratios while still blocking exact repeats if it fails. | `Imported Research Memory` and anti-duplicate paragraph |
| Require future masked-generative ideas to include unigram/material-prior and square-shuffled surprise controls. | These controls are the main falsifiers for compression-shortcut explanations. | `Depth requirements` or `Ablation Plan` guidance |
| Clarify that LC0 encodings need explicit current-board channel schemas before deterministic token extraction. | Avoids silent leakage or wrong reconstruction targets from history/channel ambiguity. | `Project Context` / encoding notes |
| Add matched fine-label-0 FPR near-puzzle diagnostics as a preferred recurring metric. | It makes class `1` behavior visible without changing or fabricating labels. | `Benchmark` requirements |
| If this idea fails, add a note not to rescue it merely by increasing codec size, number of masks, or classifier width. | Keeps future cycles from turning a falsified mechanism into routine hyperparameter tuning. | `Research Continuity` |

Do not weaken leakage rules, label rules, falsification requirements, or anti-duplicate requirements.

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0739_tuesday_los_angeles_masked_surprise_codec.md`
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes, `simple_18` codec pretrain plus frozen-codec classifier on the provided split
- Falsification criterion is concrete: yes, central unigram/material codec and square-shuffled surprise controls
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported research packets completed: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Möbius-constellation, or pseudo-likelihood packets: yes
