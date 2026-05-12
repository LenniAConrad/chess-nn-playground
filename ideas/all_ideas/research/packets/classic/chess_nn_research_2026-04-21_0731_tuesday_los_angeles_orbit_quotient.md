# Codex Handoff Packet: Legal Automorphism Quotient Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0731_tuesday_los_angeles_orbit_quotient.md`
- Generated at: 2026-04-21 07:31:07 UTC-07:00
- Weekday: Tuesday
- Timezone: los_angeles
- Idea slug: orbit_quotient
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Legal Automorphism Quotient Network
- One-sentence thesis: A chess puzzle-likeness classifier should quotient out the exact current-board automorphisms of chess rules—file mirror and color/side flip—so that the supervised head cannot spend capacity on orientation artifacts that do not change whether a position is puzzle-like.
- Idea fingerprint: current FEN tensor -> four-element legal chess automorphism orbit `{identity, file_mirror, color_flip, file_mirror_color_flip}` -> shared encoder -> Reynolds invariant latent plus optional nontrivial-character penalty -> binary puzzle-likeness logits.
- Why this is not a common CNN/ResNet/Transformer variant: the core operator is a finite-group quotient over chess-rule-preserving board transformations, not a deeper backbone, attention layer, ordinary data augmentation, attack graph, move bag, transport coupling, or search surrogate.
- Current-data minimal experiment: train `LegalAutomorphismQuotientCNN` on `simple_18` using the existing `crtk_sample_3class` train/val/test split, report the normal binary metrics plus the required fine-label `0/1/2 -> predicted 0/1` diagnostic.
- Smallest central falsification ablation: keep the same four-view compute and same shared encoder, but replace the four legal automorphisms with four fixed material/channel-count-preserving random square permutations; if this randomized-orbit quotient matches the semantic quotient, the mathematical claim that chess-rule automorphism structure matters is falsified.
- Expected information gain if it fails: failure would tell the next cycle that exact orientation invariance is not the bottleneck on this split, and that future work should target invariant shortcuts such as material, phase, motif frequency, or dataset-source artifacts rather than side/file orientation artifacts.

## 3. Problem Restatement And Data Contract

The project is `chess-nn-playground`. The task is binary chess puzzle-likeness classification from a single board position:

- output `0`: non-puzzle
- output `1`: puzzle-like

The available fine labels are:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

The default binary target is:

\[
y = \mathbf 1[\text{fine label} \in \{1,2\}].
\]

The benchmark must continue to report the rectangular diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Allowed current encodings are:

- `simple_18`: 12 piece planes + side-to-move + castling + en-passant
- `lc0_static_112`
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists

The PyTorch module must accept:

```text
(batch, C, 8, 8)
```

and return logits:

```text
(batch, num_classes)
```

where `num_classes = 2`.

The benchmark split is fixed:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

The full roughly 45M-row Parquet file must not be used directly by the normal trainer until streaming support exists.

Leakage checklist:

- Safe neural inputs: deterministic board coordinates, piece occupancy, side-to-move, castling rights, en-passant planes already present in the encoding.
- Safe derived structure for this idea: deterministic transforms of those existing planes under exact board automorphisms; no engine, no legal move tree, no labels used in transform construction.
- Allowed but not used here: pseudo-legal attack geometry derived only from the current board.
- Leakage-prone unless separately justified and ablated: full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences.
- Never neural inputs: Stockfish evaluations, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance.
- For `lc0_static_112` and `lc0_bt4_112`: deterministic geometry may only transform channels whose semantics are explicitly registered. History channels may be processed by learned neural adapters only when their channel map is known; otherwise the adapter must fail closed rather than silently guessing.

Boundary between safe rule-derived features and leakage:

- Deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack geometry derived only from the current board are allowed.
- Full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences are leakage-prone unless explicitly justified as rule-only, label-independent, engine-free, and ablated.
- Engine evaluation, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, and dataset provenance must never be neural-network inputs.
- For `lc0_static_112` and `lc0_bt4_112`, distinguish current-board channels used for deterministic geometry from history channels used only by learned neural adapters.

## 4. Research Map

External sources used:

| Source | URL | What is borrowed | What is not copied |
|---|---|---|---|
| Cohen and Welling, “Group Equivariant Convolutional Networks,” ICML 2016 | https://arxiv.org/abs/1602.07576 | The general principle that known symmetries can reduce sample complexity by weight sharing or equivariance. | This packet does not implement group convolutions, D4 image symmetry, or a generic G-CNN. Chess pawns and castling make most image symmetries invalid. |
| Sannai et al., “Equivariant and Invariant Reynolds Networks,” JMLR 2024 / arXiv 2021 | https://arxiv.org/abs/2110.08092 and https://www.jmlr.org/papers/volume25/22-0891/22-0891.pdf | The Reynolds-operator view: finite-group averaging converts a function to an invariant one. | This packet does not copy their architecture or general permutation-group construction; it uses a hand-specified four-element chess automorphism group. |
| Reynolds operator background in invariant theory | https://en.wikipedia.org/wiki/Reynolds_operator | The algebraic idea that group averaging projects onto invariant components. | No claim depends on Wikipedia for a novel theorem; it is only terminology support. |
| Arjovsky et al., “Invariant Risk Minimization,” 2019 | https://arxiv.org/abs/1907.02893 | The causal framing that invariant predictors can be preferable when spurious correlations vary across environments. | The model does not implement IRM penalties over dataset environments, and it does not use source labels. |
| Geirhos et al., “Shortcut Learning in Deep Neural Networks,” Nature Machine Intelligence 2020 | https://www.nature.com/articles/s42256-020-00257-z and https://arxiv.org/abs/2004.07780 | The warning that high benchmark performance can come from shortcuts rather than intended structure. | No computer-vision shortcut benchmark or method is copied. |
| Chessprogramming Wiki, “Color Flipping” | https://www.chessprogramming.org/Color_Flipping | Practical definition of vertical board flip plus color, side-to-move, castling, and en-passant transformation. | No engine evaluation, NNUE accumulator logic, or search procedure is imported. |
| Chessprogramming Wiki, “Flipping Mirroring and Rotating” | https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating | Board-index implementation sanity checks for file/rank flips. | Rotations and invalid pawn-direction symmetries are explicitly rejected. |
| FIDE Laws of Chess, online handbook | https://handbook.fide.com/chapter/e012018 | Confirmation that side-to-move, castling, and en-passant are rule state fields that cannot be ignored by safe transformations. | The model does not use legal move adjudication, checkmate detection, or competition metadata. |

Candidate search trace:

| Candidate mechanism considered | Status | Why it lost or survived |
|---|---|---|
| Legal automorphism Reynolds quotient over `{e, file mirror, color flip, both}` | selected | It is mathematically crisp, cheap, label-safe, uses no move enumeration, and has a clean randomized-orbit falsifier. |
| Conditional invariant representation across `simple_18`, `lc0_static_112`, and `lc0_bt4_112` adapters | rejected serious candidate | Promising, but first implementation would be entangled with channel-semantics uncertainty and adapter bugs; the current packet needs a minimal current-data experiment with fewer moving parts. |
| Dirichlet/evidential ambiguity head for fine labels `1` versus `2` without fabricating labels | rejected serious candidate | Useful for calibration, but the central operator would be a loss/head design adjacent to ordinal/evidence modeling rather than a new board-position inductive bias. |
| Masked generative compression with orientation-invariant MDL | rejected serious candidate | Too close to the imported geometry-conditioned pseudo-likelihood/description-length family, even if the orientation controls differ. |
| Causal contrastive learning with synthetic material-preserving corruptions | rejected serious candidate | Interesting but hard to choose corruptions that are label-preserving; risk of training the model to ignore real tactical material imbalances. |
| Persistent homology over occupied-square filtrations | rejected serious candidate | Distinct mathematically, but likely too weak for chess tactics and too close to cell-complex/topological descriptors without a clear supervised falsifier. |
| Spectral statistics of pseudo-legal attack matrices | rejected | Strong chess bias, but it overlaps the imported tactical sheaf/Hodge/attack-defense graph families. |
| One-ply legal-repair autoencoder | rejected | Would require legal move generation and possibly check/stalemate logic, making leakage controls harder than the likely gain justifies. |
| Piece-type token Transformer with side-relative positional encodings | rejected | Too close to “vanilla Transformer over squares/pieces,” and the constraint explicitly warns against that. |
| D4 image-equivariant CNN | rejected | Invalid chess symmetries: pawns, castling, en-passant, and side-to-move break rotations and many reflections. |
| Low-rank nuisance adversary over material/phase | rejected | Too close in spirit to nuisance suppression, while less clean than exact group quotient and harder to falsify. |
| Hypergraph of pins, forks, skewers from hand rules | rejected | Would require many hand-coded tactical detectors and risks becoming a leaky or brittle pseudo-search feature set. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN | `src/chess_nn_playground/models/cnn.py` | Already exists and lacks a new inductive bias beyond local convolution. |
| Residual CNN | `src/chess_nn_playground/models/residual_cnn.py` | Already exists; making it wider or deeper is ordinary architecture scaling. |
| LC0-style CNN / residual CNN | LC0 BT4-style CNN and residual CNN variants | Already covered by the baseline suite and mostly changes input layout/backbone style. |
| Ordinary ViT over 64 squares | No exact baseline, but common square-token model | Explicitly disallowed as a core idea and not chess-specific enough. |
| Plain GNN on 64 square nodes | Common graph baseline | Too ordinary; without a novel operator it is just message passing over board adjacency. |
| Hyperparameter tuning | Existing training configs | Not a research idea; it would not test a new mathematical hypothesis. |
| Ensembling baselines | Any existing model ensemble | Explicitly disallowed and would obscure which mechanism helps. |
| More data from the 45M-row Parquet | Dataset scaling path | The current trainer should not be pointed at the full file; also “add more data” is not an idea. |
| Standard data augmentation by random flips only | CNN with augmentation | Too weak: the selected idea uses an exact quotient and a randomized-orbit falsifier, not just augmentation. |
| Tactical sheaf / Hodge / attack-defense graph | Imported sheaf/Hodge packets | Already researched; adding edge types or changing pooling would be a duplicate. |
| One-ply move-delta bag, spectrum, landscape, or MIL model | Imported counterfactual move-delta packets | Already researched and explicitly excluded unless the operator is genuinely different. |
| Entropic piece-target or material-target transport | Imported OT packets | Already researched; Sinkhorn cost variants are excluded. |
| Deterministic nuisance-vector residualization | Imported nuisance-orthogonal packet | Already researched; closed-form projection over material/phase is excluded. |
| Ordinal cumulative evidence ladder | Imported ordinal packet | Already researched; ambiguity handling through nested thresholds is not repeated. |
| Sparse occupied-piece witness bottleneck | Imported sparse-witness packet | Already researched; top-k witness selection is not the selected mechanism. |
| Ray-language automaton | Imported ray-language packet | Already researched; file/rank/diagonal token automata are not repeated. |
| Möbius/ANOVA piece-constellation interactions | Imported high-order constellation packet | Already researched; higher-order occupied-piece interactions are not repeated. |
| Static-geometry pseudo-likelihood ratio | Imported pseudo-likelihood packet | Already researched; the selected method is discriminative group quotienting, not board reconstruction or description length. |

## 6. Mathematical Thesis

### Input space definition

Let a board state be

\[
s=(b,\tau,c,e)\in\mathcal S,
\]

where:

- \(b:\{0,\ldots,7\}^2\to \{\emptyset\}\cup(\{W,B\}\times\{P,N,B,R,Q,K\})\) is current piece occupancy,
- \(\tau\in\{W,B\}\) is side to move,
- \(c\in\{0,1\}^4\) encodes castling rights \((W_K,W_Q,B_K,B_Q)\),
- \(e\in\{\bot\}\cup\{0,\ldots,7\}\times\{2,5\}\) is the en-passant target state when present.

An encoding map \(\kappa_E:\mathcal S\to\mathbb R^{C_E\times 8\times 8}\) produces `simple_18`, `lc0_static_112`, or `lc0_bt4_112`. The first experiment uses \(E=\texttt{simple_18}\).

### Label/target definition

For fine label \(t\in\{0,1,2\}\),

\[
y(t)=\mathbf 1[t\ge 1]\in\{0,1\}.
\]

The classifier predicts logits \(f_\theta(\kappa_E(s))\in\mathbb R^2\).

### Data distribution assumptions

The empirical training distribution \(D\) may contain orientation artifacts: for example, puzzle-like positions may be overrepresented with White to move, with attacks on a particular side of the board, or with human-composition conventions. The scientific assumption is not that the dataset is perfectly symmetric. The assumption is:

\[
y(s)=y(g\cdot s)\quad\text{for all }g\in G,
\]

for the exact chess-rule automorphism group \(G\) defined below, while some nuisance correlations in \(D\) are not invariant under \(G\).

### Allowed symmetry or equivariance assumptions

Chess is not invariant under arbitrary rotations or reflections. Pawns, castling, en-passant, and side-to-move break most image symmetries. This idea uses only the four transformations generated by:

1. **File mirror** \(m\): map file \(a\leftrightarrow h\), \(b\leftrightarrow g\), \(c\leftrightarrow f\), \(d\leftrightarrow e\); keep rank and colors; swap kingside/queenside castling rights for each color; mirror en-passant file; keep side-to-move.
2. **Color flip** \(q\): reflect ranks \(1\leftrightarrow8\), \(2\leftrightarrow7\), \(3\leftrightarrow6\), \(4\leftrightarrow5\); swap piece colors; swap side-to-move; swap White/Black castling rights preserving king/queen side; reflect en-passant rank.

These commute and generate:

\[
G=\langle m,q\rangle\cong C_2\times C_2=\{e,m,q,mq\}.
\]

No rotation by 90 degrees, diagonal reflection, or raw vertical flip without color swap is assumed.

### Core hypothesis

Puzzle-likeness is a property of tactical tension in a position, not of whether the same position is drawn from White's viewpoint, Black's viewpoint, the kingside file orientation, or the queenside file orientation. Therefore, the Bayes-relevant signal should be expressible through the quotient \(\mathcal S/G\). A learned classifier that is forced to use an invariant latent should generalize at least as well as a comparable CNN when the benchmark contains orientation shortcuts, and it should be more stable on fine label `1` near-puzzles.

### Formal object introduced

Let \(\phi_\theta:\mathbb R^{C\times8\times8}\to\mathbb R^d\) be a shared neural encoder. Define the Reynolds invariant latent:

\[
P_G\phi_\theta(s)=\frac{1}{|G|}\sum_{g\in G}\phi_\theta(\kappa_E(g\cdot s)).
\]

Because \(G\cong C_2\times C_2\), it has four one-dimensional characters. Let \(\widehat G\) be the character group, and let \(\chi_0\) be the trivial character. Define character components:

\[
P_\chi\phi_\theta(s)=\frac{1}{|G|}\sum_{g\in G}\chi(g)\phi_\theta(\kappa_E(g\cdot s)).
\]

The classifier uses only:

\[
z_0(s)=P_{\chi_0}\phi_\theta(s).
\]

The optional regularizer penalizes nontrivial components:

\[
\mathcal R_{\mathrm{char}}(s)=\sum_{\chi\in\widehat G\setminus\{\chi_0\}}\|P_\chi\phi_\theta(s)\|_2^2.
\]

The supervised objective is:

\[
\min_\theta\;\mathbb E_{(s,y)\sim D}\left[
\mathrm{CE}\left(h_\theta(P_G\phi_\theta(s)),y\right)
+\lambda\mathcal R_{\mathrm{char}}(s)
\right].
\]

### Proposition

Let \(G\) be a finite group acting on \(\mathcal S\). Suppose the target is invariant: \(y(g\cdot s)=y(s)\). Let \(a:\mathcal S\to\mathbb R^2\) be any logit predictor and define its logit-averaged Reynolds predictor

\[
\bar a(s)=\frac{1}{|G|}\sum_{g\in G}a(g\cdot s).
\]

For cross-entropy loss \(\ell(a,y)\), which is convex in logits \(a\), the orbit loss satisfies:

\[
\ell(\bar a(s),y(s))
\le
\frac{1}{|G|}\sum_{g\in G}\ell(a(g\cdot s),y(g\cdot s)).
\]

Therefore, on the symmetrized empirical distribution \(D_G\), group-averaged prediction cannot have worse orbit-averaged cross-entropy than the average loss of the unaveraged orbit predictions.

### Proof sketch or derivation

Cross-entropy with a fixed class label is

\[
\ell(a,y)=-a_y+\log\sum_j\exp(a_j),
\]

which is convex in \(a\) because log-sum-exp is convex and \(-a_y\) is linear. Jensen's inequality gives

\[
\ell\left(\frac{1}{|G|}\sum_g a(g\cdot s),y(s)\right)
\le
\frac{1}{|G|}\sum_g \ell(a(g\cdot s),y(s)).
\]

By label invariance, \(y(s)=y(g\cdot s)\), yielding the stated result. The Reynolds latent \(P_G\phi\) is invariant because for any \(h\in G\),

\[
P_G\phi(h\cdot s)
=
\frac{1}{|G|}\sum_{g\in G}\phi(g h\cdot s)
=
\frac{1}{|G|}\sum_{u\in G}\phi(u\cdot s)
=
P_G\phi(s).
\]

The character decomposition is the finite Fourier decomposition over \(C_2\times C_2\); the trivial character is the invariant component, while the three nontrivial characters measure transform-specific variation.

### What is actually proven

- The forward pass is exactly invariant to the four specified chess-rule automorphisms, assuming the tensor transforms are implemented correctly.
- Cross-entropy on an averaged logit predictor obeys the Jensen inequality above on each orbit.
- The nontrivial character components are zero exactly when the encoder representation is identical across the four transformed views.

### What remains only hypothesized

- That puzzle-likeness labels in this dataset are close enough to invariant under \(G\) for quotienting to help.
- That important shortcut correlations are non-invariant under \(G\).
- That improved invariance will especially help verified near-puzzles, not just test-time calibration.
- That the current split has enough orientation artifact pressure for the effect to be measurable in three epochs.

### Counterexamples where the idea should fail

- If class labels encode orientation-dependent curation artifacts, such as puzzles preferentially verified only for one side-to-move convention, exact invariance can remove predictive information for the benchmark even if that information is scientifically undesirable.
- If the strongest shortcuts are already invariant under \(G\), such as material imbalance, phase, or piece-count distributions, this quotient will not remove them.
- If the baseline residual CNN already learns the exact same invariance from data, the quotient may add compute without improving metrics.
- If channel transforms for castling or en-passant are wrong, the model can be exactly invariant to the wrong object; transform tests are mandatory.
- If many FENs contain unusual or inconsistent state fields, deterministic state transforms may produce legal-looking tensors whose semantic relation to the source position is noisy.

### Self-critique

The strongest objection is that this idea may be “too clean”: exact symmetries are mathematically satisfying, but the current benchmark may already be large enough for a residual CNN to learn them. The method also cannot fight invariant shortcuts like material or phase. It still deserves the minimal experiment because it is cheap, leak-safe, and unusually falsifiable: the semantic-random orbit ablation preserves four-view compute and regularization while destroying the chess automorphism. If the real quotient loses to that control, the next cycle gets a clear negative result rather than an ambiguous “model too small” failure.

## 7. Architecture Specification

### Module names

- `LegalAutomorphismTransform`
- `OrbitStacker`
- `SharedResidualBoardEncoder`
- `ReynoldsCharacterProjector`
- `LegalAutomorphismQuotientCNN`
- Builder function: `build_legal_automorphism_quotient_cnn`

### Forward-pass steps

Input:

```text
x: [B, C, 8, 8]
```

First experiment assumes:

```text
encoding = simple_18
C = 18
```

Steps:

1. Validate encoding semantics.
   - `simple_18` must have explicit channel indices for:
     - 12 piece planes
     - side-to-move
     - castling rights
     - en-passant
   - If the channel map is absent, raise an error.
2. Build automorphism orbit:
   - `x_e = x`
   - `x_m = file_mirror(x)`
   - `x_q = color_flip(x)`
   - `x_mq = file_mirror(color_flip(x))`
   - Stack to:
     ```text
     X_orbit: [B, 4, C, 8, 8]
     ```
3. Flatten orbit batch:
   ```text
   X_flat: [4B, C, 8, 8]
   ```
4. Shared encoder:
   - stem `Conv2d(C, width, kernel_size=3, padding=1)`:
     ```text
     [4B, width, 8, 8]
     ```
   - `num_blocks` residual blocks, each `Conv3x3 -> Norm -> GELU -> Conv3x3 -> Norm -> residual`:
     ```text
     [4B, width, 8, 8]
     ```
   - global average pool:
     ```text
     [4B, width]
     ```
   - latent projection MLP:
     ```text
     [4B, latent_dim]
     ```
5. Reshape:
   ```text
   Z: [B, 4, latent_dim]
   ```
6. Reynolds character projection:
   - invariant latent:
     ```text
     z_inv = mean(Z, dim=1): [B, latent_dim]
     ```
   - optional nontrivial character components:
     ```text
     z_char: [B, 3, latent_dim]
     ```
     using the fixed sign matrix for `C2 x C2`:
     ```text
     [[ 1,  1,  1,  1],
      [ 1, -1,  1, -1],
      [ 1,  1, -1, -1],
      [ 1, -1, -1,  1]] / 4
     ```
     assuming orbit order `[e, m, q, mq]`.
7. Classification head:
   ```text
   logits = MLP(latent_dim -> hidden_dim -> 2): [B, 2]
   ```
8. Return logits compatible with the shared trainer.
   - If auxiliary losses are supported, return `(logits, aux)` only behind a config switch.
   - Default must return only logits.

### Parameter-count estimate

Default first run:

```yaml
width: 96
num_blocks: 4
latent_dim: 192
head_hidden_dim: 96
```

Approximate trainable parameters for `simple_18`:

- stem: about 15.6K
- four residual blocks: about 664K
- latent projection and classifier: about 37K
- normalization and biases: about 5K to 20K depending on exact norm layers

Expected total: roughly `0.72M` to `0.75M` parameters. The orbit quotient multiplies activations and FLOPs by `|G| = 4`, but not backbone parameters.

### FLOP or complexity estimate

Let `F_encoder(B)` be the FLOPs of the shared encoder for batch size `B`. The model cost is approximately:

\[
4F_\text{encoder}(B)+O(B\cdot4\cdot d)
\]

because the orbit has four views. The character projection is negligible compared with convolution.

### Candidate-set memory and chunking

There is no generated move or tactical candidate set. The only structured set is the four-element orbit. Memory scales as:

\[
O(B\cdot |G|\cdot C\cdot 8\cdot8)
\]

for orbit inputs and:

\[
O(B\cdot |G|\cdot d)
\]

for orbit latents.

For `B=512`, `C=18`, `|G|=4`, float32 orbit input storage is approximately:

```text
512 * 4 * 18 * 8 * 8 * 4 bytes ≈ 9.4 MB
```

Major activation memory is in the shared encoder:

```text
512 * 4 * 96 * 8 * 8 * 4 bytes ≈ 50.3 MB
```

per retained activation tensor. If memory is tight, implement orbit chunking:

```text
for each transform g in G:
    encode transformed x_g
    accumulate z_g
```

This avoids materializing `[4B, width, 8, 8]` for all views at once, at the cost of less parallelism.

### Required config fields

```yaml
model:
  name: legal_automorphism_quotient_cnn
  input_channels: 18
  num_classes: 2
  encoding: simple_18
  width: 96
  num_blocks: 4
  latent_dim: 192
  head_hidden_dim: 96
  group: chess_c2x_c2
  orbit_order: [identity, file_mirror, color_flip, file_mirror_color_flip]
  char_penalty_weight: 0.01
  transform_channel_map: simple_18_default
  fail_closed_on_unknown_channels: true
  return_aux: false
```

### Encoding support

`simple_18` is the only required first experiment. It is the cleanest choice because the transform channel semantics are simple and current-board-only.

`lc0_static_112` support is optional for the first Codex implementation. It may be added only if the channel semantics for current-board piece planes, side-to-move, castling, and en-passant are explicitly registered. If not, `LegalAutomorphismTransform` must raise a clear error.

`lc0_bt4_112` support should fail closed by default. Because unavailable history planes are currently zero-filled until exporter support exists, it is tempting to treat BT4 as static; do not guess. Add support only when the exact channel map is documented. Learned neural adapters may consume full `112` channels only after deterministic transforms are known. Unknown history channels must not be silently transformed, ignored, or interpreted as current-board planes.

### Pseudocode

This is pseudocode only, not final implementation:

```python
class LegalAutomorphismQuotientCNN(nn.Module):
    def forward(self, x):
        # x: [B, C, 8, 8]
        orbit = self.orbit_stacker(x)        # [B, 4, C, 8, 8]
        B, G, C, H, W = orbit.shape

        if self.chunk_orbit:
            zs = []
            for j in range(G):
                zs.append(self.encoder(orbit[:, j]))  # [B, D]
            z = stack(zs, dim=1)             # [B, 4, D]
        else:
            flat = orbit.view(B * G, C, H, W)
            z = self.encoder(flat).view(B, G, self.latent_dim)

        z_inv, z_chars = self.projector(z)   # [B, D], [B, 3, D]
        logits = self.head(z_inv)            # [B, 2]

        if self.return_aux:
            aux = {"char_energy": (z_chars ** 2).mean()}
            return logits, aux
        return logits
```

## 8. Loss, Training, And Regularization

Primary loss:

\[
\mathcal L_\text{CE}=\mathrm{CrossEntropyLoss}(\text{logits},y)
\]

with balanced class weighting as in existing benchmark configs.

Optional auxiliary loss:

\[
\mathcal L_\text{char}=\lambda\sum_{\chi\ne\chi_0}\|P_\chi\phi_\theta(s)\|_2^2.
\]

Default:

```yaml
char_penalty_weight: 0.01
```

If the shared trainer cannot consume auxiliary losses without disruption, run the first experiment with `char_penalty_weight: 0.0` and keep the exact quotient. The quotient is the core idea; the penalty is secondary.

Class weighting:

- Use existing balanced coarse-binary class weighting.
- Do not reweight using source labels or dataset provenance.
- Do not fabricate additional class `1` or class `2` labels.

Batch size expectations:

- Start with `batch_size: 512` for `simple_18`.
- Effective encoder batch is `4 * batch_size`.
- If GPU memory is insufficient, enable orbit chunking before reducing scientific controls.

Optimizer defaults:

```yaml
optimizer: AdamW
learning_rate: 0.001
weight_decay: 0.0001
epochs: 3
early_stopping_patience: 2
mixed_precision: false
```

Regularizers:

- Existing weight decay.
- Optional nontrivial-character energy penalty.
- Optional orbit-logit consistency report, not a required loss.
- No dropout in the first minimal experiment unless existing configs require it; dropout can obscure invariance diagnostics.

Determinism requirements:

- Keep `seed: 42`.
- Use deterministic PyTorch settings when available.
- Fixed orbit order `[identity, file_mirror, color_flip, file_mirror_color_flip]`.
- Randomized-orbit ablation must store and report its random permutation seed.

Unchanged for fair comparison:

- Same train/val/test split.
- Same coarse-binary target.
- Same `simple_18` encoding for the minimal comparison.
- Same epoch budget unless baseline configs already define a standard.
- Same metrics, reports, confusion matrices, prediction artifacts, and leaderboard format.
- No engine features, source labels, verification metadata, or proposed labels.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Single-view shared encoder | Use only the identity view with the same encoder/head; optionally widen slightly to match parameter count. | Exact quotient, not just the backbone, improves puzzle-likeness classification. | If single-view matches or wins across metrics and class `1` recall, orientation invariance is not useful on this split. |
| **Semantic-random orbit quotient** | Replace `{e,m,q,mq}` with four fixed random square permutations that preserve channel counts/material but are not chess automorphisms; keep four-view averaging and same compute. | The chess-rule semantics of the group matter, not just multi-view regularization. | Central falsifier: if random orbit matches the semantic quotient, abandon this mechanism. |
| Augmentation-only automorphism CNN | Sample one legal automorphism per example during training, evaluate identity only. | Exact Reynolds quotient is better than ordinary augmentation. | If augmentation-only matches, the quotient may be unnecessary engineering. |
| Test-time averaging baseline | Train a normal residual CNN; at evaluation average logits over the four legal automorphisms. | Training through the quotient matters beyond post-hoc TTA. | If TTA matches, the value is mostly inference-time smoothing, not representation learning. |
| No character penalty | Set `char_penalty_weight = 0.0`, keep invariant latent. | The hard quotient is sufficient; nontrivial-character penalty may be unnecessary. | If performance is identical, drop the penalty for simplicity. |
| Character penalty only, no quotient head | Feed identity latent to classifier but add orbit character penalty during training. | Suppressing transform variation without forcing invariant logits is enough. | If this wins, future work should treat invariance as regularization rather than architecture. |
| File mirror only | Use orbit `{e,m}`. | Left-right board orientation is the useful nuisance. | If this equals the full group, color flip is unnecessary or noisy. |
| Color flip only | Use orbit `{e,q}`. | Side/color perspective is the useful nuisance. | If this equals the full group, file mirror is unnecessary or noisy. |
| Broken state transform | Mirror piece planes but intentionally do not swap castling/en-passant state in a controlled test expected to fail. | Correct rule-state handling matters. | If broken state performs the same, the model may ignore special state planes; note this for LC0 support. |
| Within-batch orbit mismatch | For non-identity views, use transformed boards from other examples with matched labels or material bins. | Pairing each orbit with the same position is essential. | If this works, gains are likely from batch regularization or material distribution artifacts. |
| Invariant shortcut probe | Train linear probes from `z_inv` to material count, phase, side-to-move, and castling flags. | The quotient removes only group-variant features, not all nuisances. | If probes remain strong but classifier improves, the mechanism is doing the intended narrow job; if classifier fails, invariant nuisances may dominate. |
| Transform-consistency metric only | Do not train a new model; measure baseline prediction variance across the legal orbit. | Baselines may already be approximately invariant. | If baseline variance is near zero, the selected idea has little room to help. |

The smallest central ablation is the semantic-random orbit quotient. It preserves candidate count, compute, parameter sharing, channel counts, material counts, and four-view averaging while destroying the proposed chess automorphism semantics.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- Existing `simple_18` simple CNN.
- Existing `simple_18` residual CNN, same small/medium/deep variant closest in parameter count.
- Existing LC0-style baselines only as context, not as the primary fair comparison if the new model first supports only `simple_18`.
- Augmentation-only automorphism CNN.
- Semantic-random orbit quotient central ablation.
- Test-time averaging baseline if easy to implement.

Metrics to inspect:

- Test accuracy.
- Balanced accuracy.
- Macro F1.
- ROC-AUC.
- PR-AUC.
- Brier score or ECE if already supported.
- Fine-label rectangular diagnostic matrix:
  ```text
  true fine label 0/1/2 -> predicted binary output 0/1
  ```
- Class `1` recall and precision.
- Class `2` recall.
- Fine-label `0` false-positive rate.
- Orbit consistency:
  \[
  \max_{g\in G}\|p(y=1\mid s)-p(y=1\mid g\cdot s)\|
  \]
  on validation/test examples.

Near-puzzle diagnostic:

- Match the fine-label `0` false-positive rate of the strongest `simple_18` residual CNN baseline by thresholding predicted `p(y=1)`.
- At that matched FPR, report recall on fine label `1` and fine label `2` separately.
- The key diagnostic is fine-label `1` recall at matched fine-label `0` FPR.

Required artifacts:

- Normal training logs.
- `results/.../metrics.json`.
- `results/.../confusion_binary.png` or existing equivalent.
- Fine-label `3x2` diagnostic matrix for the main model and every central ablation.
- Predictions Parquet/CSV with true fine label, coarse target, logits/probabilities, predicted class.
- Orbit consistency report for the main model and baseline residual CNN.
- Ablation comparison table.

Success threshold:

- Primary: beat the closest parameter-count `simple_18` residual CNN on test macro F1 or ROC-AUC by at least `+1.0` absolute percentage point.
- Secondary: improve fine-label `1` recall at matched fine-label `0` FPR by at least `+2.0` absolute percentage points.
- Structural: semantic quotient must beat the semantic-random orbit quotient on either test macro F1 or near-puzzle diagnostic by at least `+1.0` point.
- Consistency: reduce orbit prediction variance versus the residual CNN by at least `50%`.

Failure threshold:

- Within `±0.5` percentage points of the residual CNN on all primary metrics and near-puzzle diagnostics.
- Semantic-random orbit quotient equals or beats the legal automorphism quotient.
- Transform-consistency improves but classification metrics do not, meaning invariance is real but not useful.
- Implementation cannot pass involution/commutation/channel transform tests.

Abandon the idea if:

- The legal automorphism quotient does not beat the random-orbit quotient.
- The residual CNN with test-time averaging matches the trained quotient.
- Fine-label `1` recall does not improve at matched fine-label `0` FPR.
- Transform errors or channel uncertainty make LC0 support fragile and `simple_18` results are flat.

Justify scaling if:

- The main model beats the closest residual CNN and random-orbit ablation.
- Fine-label `1` recall improves at matched fine-label `0` FPR.
- Orbit consistency is substantially better without calibration degradation.
- File-only and color-only ablations indicate which subgroup contributes most, guiding a larger LC0-compatible follow-up.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_0731_orbit_quotient/idea.yaml` | create | Machine-readable idea metadata copied from Section 12 `idea_yaml`. |
| `ideas/20260421_0731_orbit_quotient/math_thesis.md` | create | Section 6 with formulas, proposition, proof sketch, hypotheses, and counterexamples. |
| `ideas/20260421_0731_orbit_quotient/architecture.md` | create | Section 7 with module names, tensor shapes, transforms, and pseudocode. |
| `ideas/20260421_0731_orbit_quotient/implementation_notes.md` | create | Transform definitions, channel maps, fail-closed behavior, orbit order, chunking plan, and transform tests. |
| `ideas/20260421_0731_orbit_quotient/trainer_notes.md` | create | Section 8 training defaults, auxiliary loss handling, deterministic settings, and fair-comparison constraints. |
| `ideas/20260421_0731_orbit_quotient/ablations.md` | create | Section 9 ablation table and central falsification explanation. |
| `ideas/20260421_0731_orbit_quotient/train.py` | create | Thin entrypoint that loads the project trainer with the new config; avoid custom training loops unless required for auxiliary loss. |
| `ideas/20260421_0731_orbit_quotient/config.yaml` | create | Full config for `simple_18`, `legal_automorphism_quotient_cnn`, batch size 512, 3 epochs, balanced class weighting. |
| `ideas/20260421_0731_orbit_quotient/report_template.md` | create | Template requiring main metrics, `3x2` fine-label matrix, matched-FPR near-puzzle diagnostics, orbit consistency, and ablation table. |
| `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | update | Add this packet to imported research memory after implementation; add anti-duplicate rule for legal automorphism Reynolds quotient if it fails or succeeds. |
| `src/chess_nn_playground/models/legal_automorphism_quotient.py` | create | `LegalAutomorphismTransform`, `OrbitStacker`, `SharedResidualBoardEncoder`, `ReynoldsCharacterProjector`, `LegalAutomorphismQuotientCNN`, and builder. |
| `src/chess_nn_playground/models/registry.py` | update | Register `legal_automorphism_quotient_cnn` builder without disrupting existing models. |
| `configs/legal_automorphism_quotient_simple18.yaml` | create | Main benchmark config pointing at the fixed split and `simple_18`. |
| `configs/legal_automorphism_quotient_simple18_random_orbit.yaml` | create | Central semantic-random orbit ablation config. |
| `configs/legal_automorphism_quotient_simple18_no_char.yaml` | create | No-character-penalty ablation config if auxiliary loss is implemented. |
| `configs/legal_automorphism_quotient_simple18_file_only.yaml` | create | File-mirror-only subgroup ablation. |
| `configs/legal_automorphism_quotient_simple18_color_only.yaml` | create | Color-flip-only subgroup ablation. |
| `tests/test_legal_automorphism_transforms.py` | create | Tests that file mirror and color flip are involutions, commute, preserve tensor shape, map castling/en-passant channels correctly, and preserve material counts. |
| `tests/test_legal_automorphism_model.py` | create | Tests that model logits are exactly invariant, within numerical tolerance, under all four legal automorphisms at initialization. |
| `tests/test_model_registry_legal_automorphism.py` | create | Tests registry construction and output shape `[batch, 2]`. |

For `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md`, Codex must update the prompt after consuming this output. Preserve all hard leakage, label, falsification, and anti-duplicate constraints. Add reusable lessons from the actual result: whether exact chess automorphism quotienting helped, whether the random-orbit falsifier killed it, and whether `file_mirror` or `color_flip` was the useful subgroup.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0731_tuesday_los_angeles_orbit_quotient.md
  generated_at: "2026-04-21T07:31:07-07:00"
  weekday: Tuesday
  timezone: los_angeles
  idea_slug: orbit_quotient
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_0731_orbit_quotient
  name: Legal Automorphism Quotient Network
  slug: orbit_quotient
  status: draft
  created_at: "2026-04-21T07:31:07-07:00"
  author: ChatGPT Pro
  short_thesis: Force puzzle-likeness prediction through the Reynolds invariant latent of the exact four-element chess-rule automorphism orbit.
  novelty_claim: Uses the legal chess automorphism quotient C2 x C2 as the central discriminative operator, with character diagnostics and a randomized-orbit falsifier.
  expected_advantage: Suppresses orientation and side-color shortcuts while preserving label-relevant tactical structure, improving near-puzzle recall at matched non-puzzle false-positive rate.
  central_falsification_ablation: Replace legal automorphism orbit with four fixed material/channel-count-preserving random square permutations while keeping four-view averaging and compute.
  target_task: coarse_binary
  input_representation: simple_18
  output_heads: binary_logits
  compute_notes: About 0.75M parameters; roughly 4x encoder FLOPs due to four orbit views; chunk orbit if memory is tight.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/legal_automorphism_quotient_simple18.yaml
  model_path: src/chess_nn_playground/models/legal_automorphism_quotient.py
  latest_result_path: null
  notes: First experiment should fail closed to simple_18 unless LC0 channel maps are explicitly registered.
```

```yaml
config_yaml:
  run:
    name: legal_automorphism_quotient_simple18
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
    name: legal_automorphism_quotient_cnn
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
  model_name: legal_automorphism_quotient_cnn
  file_path: src/chess_nn_playground/models/legal_automorphism_quotient.py
  builder_function: build_legal_automorphism_quotient_cnn
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - LegalAutomorphismTransform
    - OrbitStacker
    - SharedResidualBoardEncoder
    - ReynoldsCharacterProjector
    - LegalAutomorphismQuotientCNN
  required_config_fields:
    - model.name
    - model.input_channels
    - model.num_classes
    - model.encoding
    - model.group
    - model.orbit_order
    - model.transform_channel_map
    - model.fail_closed_on_unknown_channels
  expected_parameter_count: approximately 0.72M to 0.75M with width 96, 4 residual blocks, latent_dim 192
  expected_memory_notes: Four orbit views multiply encoder activation memory by 4; for batch 512 and width 96 the main retained activation tensor is about 50 MB in float32, before autograd overhead.
```

```yaml
research_continuity:
  idea_fingerprint: current-board legal chess automorphism orbit + shared CNN encoder + Reynolds invariant latent + optional C2xC2 character penalty + binary puzzle-likeness head
  already_researched_family_overlap: Low. Shares chess board transforms with possible augmentation practice and file-mirror terminology, but does not use sheaves, attack graphs, move deltas, Sinkhorn transport, nuisance residualization, ordinal heads, sparse witnesses, ray automata, Möbius constellations, or pseudo-likelihood.
  closest_duplicate_risk: Ordinary flip augmentation or test-time augmentation; distinguish by exact quotient head, color/side/castling/en-passant-safe transforms, character diagnostics, and randomized-orbit falsifier.
  do_not_repeat_if_this_fails:
    - Legal automorphism Reynolds quotient over identity/file_mirror/color_flip/both.
    - Four-view orbit averaging as the central mechanism.
    - C2 x C2 character-energy penalty as the main novelty.
    - Ordinary automorphism augmentation or test-time averaging as a claimed new idea.
  suggested_next_search_directions:
    - Invariant shortcuts that survive the automorphism group, especially material and phase controls that are not closed-form nuisance projection.
    - Label-safe uncertainty or selective prediction for near-puzzles that is not an ordinal ladder.
    - Multi-encoding causal invariance only after channel semantics and adapter tests are stable.
    - Generative compression ideas only if clearly distinct from imported pseudo-likelihood or MDL packets.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Legal Automorphism Quotient Network` to imported research memory after implementation. | Prevents future ChatGPT passes from proposing the same Reynolds quotient if it succeeds or fails. | `Imported Research Memory` |
| Add fingerprint: `current-board exact chess automorphism orbit {identity, file mirror, color flip, both} + shared encoder + Reynolds/group-average invariant latent + optional C2xC2 character penalty`. | Makes anti-duplicate matching precise without banning all group-representation ideas. | `Imported Research Memory` |
| Add anti-duplicate rule: do not repeat automorphism-only augmentation, test-time orbit averaging, or C2xC2 Reynolds quotient unless the operator is genuinely different and has a distinct falsifier. | Avoids near-duplicates such as “same group, bigger CNN” or “same group, Transformer backbone.” | Anti-duplicate paragraph after imported packet fingerprints |
| Require transform-correctness tests when any future idea uses board symmetries. | Castling and en-passant transform bugs can silently invalidate results. | `What Counts As Creative Enough` or `Required Markdown File Content` |
| Record the result of the semantic-random orbit ablation in future prompts. | The random-orbit control determines whether group semantics mattered or only multi-view regularization helped. | `Research Continuity` |
| If this idea fails because invariant shortcuts dominate, ask future passes to focus on shortcuts preserved by the legal automorphism group. | Prevents repeatedly attacking non-invariant orientation artifacts when the bottleneck is material/phase/source-like. | `Prefer next-cycle ideas` |

Do not weaken the leakage rules, label rules, falsification requirements, or anti-duplicate requirements.

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0731_tuesday_los_angeles_orbit_quotient.md`
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
