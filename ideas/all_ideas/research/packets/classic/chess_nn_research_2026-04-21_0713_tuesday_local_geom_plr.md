# Codex Handoff Packet: Geometry-Conditioned Board Pseudo-Likelihood Ratio Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0713_tuesday_local_geom_plr.md`
- Generated at: 2026-04-21 07:13:46 America/Los_Angeles
- Weekday: Tuesday
- Timezone: local
- Idea slug: `geom_plr`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Geometry-Conditioned Board Pseudo-Likelihood Ratio Network, abbreviated `GeomPLR`.
- One-sentence thesis: classify puzzle-likeness by the log description-length ratio between two class-conditioned pseudo-likelihood models of the current board, where each square's token is predicted from nearby squares under static chess geometry rather than by a direct discriminative CNN head.
- Idea fingerprint: `current-board piece tokens + side/castling/en-passant metadata -> class-conditioned typed static-geometry square conditional models -> summed leave-self-out pseudo-NLL scores S_0,S_1 -> binary logits [-S_0,-S_1] -> geometry-preserving randomized-neighborhood falsifier`.
- Closest baseline or common method it resembles: a class-conditional Markov-random-field pseudo-likelihood / denoising generative classifier, not a chess engine, not a move generator, and not a direct softmax CNN.
- Why this is not a common CNN/ResNet/Transformer variant: the central output is not an unconstrained discriminative feature vector; it is a calibrated pair of class-conditioned board description lengths computed from explicit leave-one-square conditional token predictions over fixed chess-geometric neighborhoods.
- Current-data minimal experiment: train `GeomPLR` on `simple_18` using the existing `crtk_sample_3class` train/val/test Parquet split, binary target `fine_label == 0 -> 0`, `fine_label in {1,2} -> 1`, and report the usual binary metrics plus the rectangular `3x2` fine-label diagnostic matrix.
- Smallest central falsification ablation: replace every typed chess-geometric neighborhood with a degree-, distance-bucket-, and board-edge-count-preserving random neighborhood, keeping token identities, target-square marginals, candidate count, class-conditioned decoders, parameter count, and training loss unchanged.
- Expected information gain if it fails: if randomized neighborhoods and unary/material-only pseudo-likelihood match the main model, the project learns that class-conditional board compressibility is not adding chess-geometric signal beyond source/material artifacts; future cycles should avoid MDL/pseudo-likelihood board-density-ratio variants unless they introduce a genuinely different observable.

## 3. Problem Restatement And Data Contract

Task: classify chess board positions as binary puzzle-likeness.

- Output `0`: non-puzzle.
- Output `1`: puzzle-like.
- Fine source labels available for training/evaluation diagnostics:
  - fine label `0`: known non-puzzle.
  - fine label `1`: verified near-puzzle.
  - fine label `2`: verified puzzle.
- Default training target for this packet: `y = 0` for fine label `0`, and `y = 1` for fine labels `1` and `2`.
- Required diagnostic: rectangular confusion/report matrix `true fine label 0/1/2 -> predicted binary output 0/1` for the main model and every central ablation.
- Current benchmark split:
  - `data/splits/crtk_sample_3class/split_train.parquet`
  - `data/splits/crtk_sample_3class/split_val.parquet`
  - `data/splits/crtk_sample_3class/split_test.parquet`
- Do not point the trainer directly at the roughly 45M-row full Parquet file until streaming support exists.
- PyTorch model contract: a `torch.nn.Module` accepts `(batch, C, 8, 8)` and returns logits `(batch, num_classes)`, with `num_classes = 2`.

Allowed current encodings:

- `simple_18`: 12 piece planes + side-to-move + castling + en-passant.
- `lc0_static_112`.
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists.

Leakage checklist:

- Safe inputs: deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and deterministic geometry derived only from the current board or the static chess board.
- Safe in this packet: tokenizing the 12 current piece planes into `{empty, white pawn, ..., black king}` and using fixed square-offset relation types such as same rank/file/diagonal/knight-offset. These relation types do not enumerate legal moves, do not inspect engine outputs, and do not use labels as features.
- Caution zone: full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences are leakage-prone unless explicitly justified as rule-only, label-independent, engine-free, and ablated.
- Forbidden as neural-network inputs: engine evaluation, Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, and unresolved-candidate status.
- For `lc0_static_112` and `lc0_bt4_112`, deterministic geometry must use only verified current-board channels. History channels, if consumed at all, may be used only by a learned neural adapter and never by deterministic tokenization unless their semantics are explicitly known. Unknown channel semantics must fail closed.

## 4. Research Map

External sources used and what is borrowed:

1. Julian Besag, “Statistical Analysis of Non-Lattice Data,” *The Statistician*, 1975. URL: https://www.jstor.org/stable/2987782 and accessible PDF mirror: https://www2.stat.duke.edu/~sschmid/Courses/Stat376/Papers/GibbsFieldEst/BesagPseudoLik1975.pdf
   - Borrowed: the pseudo-likelihood idea of replacing an intractable joint field likelihood with products of local conditional probabilities.
   - Not copied: no spatial-statistics estimator, no asymptotic claim for the chess dataset, and no assumption that chess positions are a true Markov random field.

2. Aapo Hyvarinen, “Consistency of Pseudolikelihood Estimation of Fully Visible Boltzmann Machines,” *Neural Computation*, 2006. URL: https://pubmed.ncbi.nlm.nih.gov/16907626/ and PDF: https://www.cs.helsinki.fi/u/ahyvarin/papers/NC06.pdf
   - Borrowed: pseudo-likelihood can be computationally attractive for fully visible structured data.
   - Not copied: the model here is not a Boltzmann machine and this packet does not claim consistency for the proposed neural parameterization.

3. Yann LeCun et al., “A Tutorial on Energy-Based Learning,” 2006. URL: https://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf
   - Borrowed: classification can be formulated by assigning class-dependent energies/scores and selecting the lower-energy class.
   - Not copied: no contrastive divergence, no latent-variable inference procedure, and no engine-like search.

4. Andrew Ng and Michael Jordan, “On Discriminative vs. Generative Classifiers,” NIPS 2001. URL: https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf
   - Borrowed: the contrast between direct discriminative classification and class-conditional generative scoring.
   - Not copied: no naive Bayes assumption and no claim that generative scoring universally dominates discriminative training.

5. Pascal Vincent et al., “Extracting and Composing Robust Features with Denoising Autoencoders,” ICML 2008. URL: https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf
   - Borrowed: reconstructive prediction of corrupted or missing input can force useful structure learning.
   - Not copied: no stacked denoising autoencoder and no self-supervised pretraining requirement.

6. Kaiming He et al., “Masked Autoencoders Are Scalable Vision Learners,” CVPR 2022. URL: https://arxiv.org/abs/2111.06377
   - Borrowed: masked reconstruction is a strong way to learn dependencies between observed and hidden input parts.
   - Not copied: no ViT, no image-patch MAE architecture, and no reliance on large-scale self-supervised pretraining.

7. Naftali Tishby, Fernando Pereira, and William Bialek, “The Information Bottleneck Method,” 1999/2000. URL: https://arxiv.org/abs/physics/0004057
   - Borrowed: the general principle that useful representations should preserve label-relevant information while discarding unnecessary detail.
   - Not copied: no explicit mutual-information estimator is used in the minimal experiment.

Candidate search trace:

| Candidate mechanism considered | Why it was serious | Why it lost to `GeomPLR` |
|---|---|---|
| Ordinal/conformal ambiguity head for fine labels `0 < 1 < 2` | Directly targets near-puzzle ambiguity and is label-safe. | It is mainly a target/head redesign; it lacks a distinct board-structure operator and could be bolted onto any baseline without testing a new chess hypothesis. |
| Multi-view invariant-risk training across `simple_18`, `lc0_static_112`, and perspective transforms | Could suppress encoding/source artifacts and enforce stable puzzle signal. | It risks becoming regularization around existing CNNs and may depend on unavailable or ambiguous LC0 channel semantics. |
| Vector-quantized masked motif autoencoder | A discrete codebook could expose reusable tactical motifs and MDL-like compression. | Codebook collapse and interpretability failures would be hard to falsify cleanly; pseudo-likelihood scores give a sharper observable. |
| Static chess-geometry spectral convolution | Uses movement-like board geometry without current attack graphs or legal moves. | As a classifier it is too close to a plain graph/CNN variant; `GeomPLR` turns the same geometry into an explicit class-conditional description-length ratio. |
| Legal-rule outcome reconstruction, such as check/checkmate/stalemate auxiliary heads | Could inject high-level chess legality constraints without engines. | Checkmate/stalemate and legal-move consequences are leakage-prone and would require careful rule-oracle ablations beyond the minimal current-data experiment. |
| Line-occlusion min-cut or blocker bottleneck | Might capture pins, skewers, and latent forcing motifs. | It is too close to already imported attack/defense graph and sheaf/tension families. |
| Side-to-move causal invariance with adversarial material/phase heads | Useful source-artifact control. | It overlaps the nuisance-control family and still depends on a base classifier for chess signal. |
| Energy-based board scoring trained by contrastive negative sampling | Mathematically attractive and outside ordinary CNNs. | Generating safe negative boards without fabricating labels or illegal distribution shifts is nontrivial; pseudo-likelihood avoids synthetic negative-board generation. |
| Diffusion/score model over board tokens | Could learn manifold geometry of puzzle-like positions. | Too expensive and poorly matched to the shared trainer contract for the next implementation cycle. |
| Hypergraph motifs over piece triples | Could model forks, overloaded pieces, and mating nets. | Static hyperedge design would likely duplicate attack-incidence/sheaf ideas unless it used legal move relations, which this cycle should avoid. |
| Selective prediction/abstention model | Useful for class `1` ambiguity. | It improves reporting behavior more than representation learning; reserve for a later calibration-focused cycle. |
| Pure material/phase causal residualization | Strong artifact-control baseline. | It is explicitly close to imported deterministic nuisance-projection packets and should not be repeated as the main idea. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN on `simple_18` | `src/chess_nn_playground/models/cnn.py` | Already implemented; direct convolutional discrimination does not test a new structural hypothesis. |
| Residual CNN small/medium/deep | `src/chess_nn_playground/models/residual_cnn.py` | Already implemented; increasing residual capacity is ordinary architecture scaling. |
| LC0-style CNN or residual CNN on `lc0_static_112`/`lc0_bt4_112` | Existing LC0 BT4-style CNN/residual variants | Already represented in the baseline suite; copying LC0-style channel processing is not a new puzzle-likeness mechanism. |
| Ordinary ViT over 64 squares | Common vanilla square-token Transformer | Explicitly disallowed as a core idea and likely to become capacity tuning rather than a chess-specific hypothesis. |
| Plain GNN on the 64-square board graph | Generic message-passing classifier | Too ordinary unless the graph operator itself produces a falsifiable nonstandard observable; `GeomPLR` uses static geometry only inside a class-conditional pseudo-likelihood score. |
| Hyperparameter tuning of width/depth/dropout/optimizer | All existing baselines | Not a research idea; it changes search effort, not inductive bias. |
| Ensembling several trained models | All existing baselines | Can improve leaderboard numbers without explaining puzzle structure and complicates falsification. |
| Training on the full 45M-row Parquet immediately | Dataset-scale variant of existing training | Blocked by current non-streaming constraint and conflates idea quality with data scale. |
| Tactical sheaf/Hodge/Laplacian/tension/curvature variant | Imported sheaf/Hodge packets | Already researched; adding relation labels or larger hidden size would be a duplicate. |
| One-ply move-delta bag/set/spectrum/free-energy model | Imported counterfactual move-delta packets | Already researched and disallowed unless the formal operator is genuinely different. |
| Piece-target/material-target Sinkhorn or optimal-transport model | Imported OT packets | Already researched; temperature, cost, or bucket changes would not be new. |
| Deterministic nuisance-vector residualization/projection | Imported nuisance-orthogonal packet | Already researched and explicitly not the central mechanism here. |
| Engine-score, PV, node-count, or mate-score feature model | None, forbidden | Direct leakage from analysis/verification and disallowed by the project contract. |

## 6. Mathematical Thesis

### Input space definition

For the minimal experiment, let

\[
X \in \{0,1\}^{18 \times 8 \times 8}
\]

be a `simple_18` encoded board. A deterministic adapter maps `X` to

\[
\tau(X) = (t_1,\ldots,t_{64}, m),
\]

where each square token

\[
t_i \in \mathcal T = \{\emptyset, WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK\}
\]

and `m` is metadata derived only from side-to-move, castling, and en-passant planes already present in `X`.

For non-`simple_18` encodings, the adapter may run only when Codex can verify the current-board piece-channel map. Otherwise it must raise a configuration error rather than guessing.

### Label/target definition

Let the fine label be

\[
\ell \in \{0,1,2\},
\]

with `0 = known non-puzzle`, `1 = verified near-puzzle`, and `2 = verified puzzle`. The binary target is

\[
Y = \mathbf 1[\ell \in \{1,2\}].
\]

The fine label is allowed as a supervised target and evaluation diagnostic. It is never an input feature.

### Data distribution assumptions

The train/validation/test split is treated as samples from an empirical distribution \(P(X,\ell)\). The idea assumes that, after obvious unigram/material effects are controlled by ablations, the conditional dependency pattern between square tokens differs between \(Y=0\) and \(Y=1\). This is a hypothesis, not a proven property of chess puzzles.

### Allowed symmetry or equivariance assumptions

No full dihedral board symmetry is assumed. Chess pawns, castling rights, en-passant, and side-to-move break most rotations/reflections. The model may use absolute coordinate embeddings and side-to-move metadata. A color-swap plus 180-degree rotation augmentation is not part of the minimal experiment unless Codex separately verifies that every channel, castling flag, and en-passant convention transforms exactly.

### Core hypothesis

Puzzle-like boards have class-conditional geometric dependencies that make their piece tokens more compressible under a positive-class pseudo-likelihood model than under a non-puzzle pseudo-likelihood model. Non-puzzles have the reverse tendency. The relevant signal is not just material count; it should survive unary/material controls and disappear under semantics-destroying randomization of chess-geometric neighborhoods.

### Formal object introduced

Define a fixed static relation system

\[
\mathcal R = \{r\}
\]

on the 64 squares. Relations may include same-rank ray offsets, same-file ray offsets, diagonal ray offsets, anti-diagonal ray offsets, knight offsets, immediate king-neighborhood offsets, and pawn-direction offsets for both colors. These are static board-coordinate relations, not legal moves from the current position.

For square \(i\), let \(\mathcal N(i)\) be its typed neighbor multiset:

\[
\mathcal N(i) = \{(j,r,d): j \neq i,\; j \text{ is related to } i \text{ by relation } r \text{ at distance bucket } d\}.
\]

For each binary class \(c \in \{0,1\}\), learn conditional token distributions

\[
q_{\theta,c}(t_i \mid t_{\mathcal N(i)}, m, i).
\]

The class-conditional pseudo-description length is

\[
S_c(t,m) = \sum_{i=1}^{64} w(t_i)\,[-\log q_{\theta,c}(t_i \mid t_{\mathcal N(i)}, m, i)],
\]

where \(w(t_i)\) downweights empty squares to prevent empty-board dominance. The model logits are

\[
z_c(X) = -S_c(\tau(X))/\tau_0 + b_c,
\]

with learned class biases \(b_c\) and positive temperature \(\tau_0\).

### Optimization objective

The minimal shared-trainer-compatible loss is class-balanced binary cross-entropy on \(z(X)\):

\[
\mathcal L_{CE} = \mathbb E[-\alpha_Y \log \operatorname{softmax}(z(X))_Y].
\]

The recommended custom training objective adds a small proper pseudo-likelihood term for the true binary class:

\[
\mathcal L = \mathcal L_{CE} + \lambda_{PL}\,\mathbb E[S_Y(\tau(X))] + \beta\lVert\theta\rVert_2^2.
\]

Set \(\lambda_{PL}=0.05\) for the first custom run and \(0\) if using an unmodified shared trainer.

### Proposition

Assume there exist true class-conditional distributions \(P_c(t,m)=P(t,m\mid Y=c)\) and that the pseudo-likelihood factorization

\[
\tilde P_c(t,m) \propto \prod_{i=1}^{64} P_c(t_i \mid t_{\mathcal N(i)},m,i)^{w(t_i)}
\]

is the scoring family used for both classes. If \(q_{\theta,c}=P_c(t_i\mid t_{\mathcal N(i)},m,i)\) for all \(i,c\), and if \(b_c=\log P(Y=c)+\kappa_c\) absorbs class-specific pseudo-normalization constants, then the classifier

\[
\hat Y(t,m)=\arg\max_c z_c(t,m)
\]

is the Bayes classifier for the pseudo-likelihood approximation \(\tilde P_c\).

### Proof sketch or derivation

Under the stated factorization,

\[
\log \tilde P_c(t,m) = \sum_i w(t_i)\log P_c(t_i\mid t_{\mathcal N(i)},m,i) + \kappa_c.
\]

Substituting \(q_{\theta,c}=P_c\) gives

\[
\log \tilde P_c(t,m) = -S_c(t,m)+\kappa_c.
\]

Bayes classification under the pseudo-model selects the class maximizing

\[
\log P(Y=c)+\log \tilde P_c(t,m),
\]

which is exactly \(-S_c(t,m)+b_c\) up to a shared positive temperature. The auxiliary pseudo-likelihood term is a sum of strictly proper log losses for the local conditionals, so with enough capacity and data it is minimized by the true local conditionals for each class. This proves the relationship between the implemented score and a pseudo-likelihood ratio; it does not prove that the approximation is true for chess.

### What is actually proven

- The logits implement a pseudo-log-likelihood ratio when their local conditionals are interpreted probabilistically.
- If the learned local conditionals match the true class conditionals and the class biases absorb priors/normalizers, the decision rule is Bayes-optimal for the pseudo-likelihood model.
- The main randomized-neighborhood ablation directly tests whether the fixed chess geometry is carrying more information than a same-degree random relational scaffold.

### What remains only hypothesized

- Puzzle-likeness is detectable from static board-token dependency structure without legal move enumeration or engine analysis.
- Positive-class puzzles/near-puzzles have recurring geometric conditional dependencies that non-puzzles do not share.
- The `simple_18` sample split has enough signal for the pseudo-likelihood ratio to beat direct discriminative CNNs.
- The class `1` near-puzzle diagnostic will improve because near-puzzles are expected to share partial dependency structure with verified puzzles.

### Counterexamples where the idea should fail

- Positions whose puzzle-likeness depends on a long forcing line invisible in current-board token dependencies.
- Quiet endgame studies where the important feature is opposition/zugzwang rather than local or ray-based token predictability.
- Dataset artifacts where puzzle labels correlate mostly with material imbalance, FEN source, or construction style; unary/material ablations may match the full model.
- Positions requiring legal-move consequences such as exact checkmate/stalemate status, which this model intentionally does not compute.
- Ambiguous near-puzzles where label `1` is closer to non-puzzle distribution than to verified puzzle distribution.

### Self-critique

The strongest objection is that a class-conditioned pseudo-likelihood model can become a disguised material/source classifier: the positive decoder may learn that puzzle boards contain more forcing-piece configurations or different material distributions, not that the square dependencies are meaningful. That is why the minimal falsifier is not merely “remove one relation type”; it is a degree-preserving randomized-neighborhood ablation plus unary/material-only controls. If those controls match the main model, abandon this mechanism. The experiment is still worth running because it gives a crisp, inspectable score decomposition by square and class, keeps the shared logits contract, avoids engine and move-tree leakage, and tests a genuinely different observable from CNN activations, sheaf energies, move-delta landscapes, and transport couplings.

## 7. Architecture Specification

### Module names

Implement in `src/chess_nn_playground/models/geometry_pseudolikelihood_ratio.py`:

- `GeometryPseudoLikelihoodRatioNet`
- `Simple18TokenAdapter`
- `StaticChessRelationIndex`
- `TypedNeighborAggregator`
- `ClassConditionalTokenDecoder`
- `PseudoLikelihoodScorer`

### Forward-pass steps and tensor shapes

Default minimal configuration:

- `input_channels = 18`
- `num_square_tokens = 13`
- `hidden_dim = 96`
- `relation_dim = 96`
- `decoder_hidden_dim = 192`
- `max_neighbors = 40`
- `target_chunk_size = 8`
- `empty_square_weight = 0.25`
- `nonempty_square_weight = 1.0`

Forward pass:

1. Input `x`: `[B, 18, 8, 8]`.
2. `Simple18TokenAdapter`:
   - extracts 12 piece planes and maps each square to an integer token.
   - output `token_ids`: `[B, 64]`, values in `[0,12]`.
   - output `meta`: `[B, M]`, where `M` should include side-to-move, castling flags, and an en-passant summary derived only from simple_18 planes.
   - if a square has no piece, token is `0 = empty`; if exactly one piece plane is active, token is the matching piece; if invalid multi-piece occupancy appears, fail or resolve by argmax only under an explicit `allow_soft_tokenization` debug flag.
3. Embedding:
   - `token_embed(token_ids)`: `[B, 64, D]`.
   - `coord_embed(square_index)`: `[1, 64, D]`.
   - `meta_mlp(meta)`: `[B, D]`, broadcast to `[B, 1, D]`.
   - base square embeddings `e`: `[B, 64, D]`.
4. `StaticChessRelationIndex` precomputes:
   - `neighbor_idx`: `[64, K]` padded with `-1`.
   - `relation_id`: `[64, K]`.
   - `distance_bucket`: `[64, K]`.
   - `valid_neighbor_mask`: `[64, K]`.
   - No current-board legal moves, no check oracle, no engine data.
5. `TypedNeighborAggregator`, run in target-square chunks of size `Q = target_chunk_size`:
   - gather neighbor embeddings: `[B, Q, K, D]`.
   - add relation and distance embeddings: `[1, Q, K, D]`.
   - apply valid-neighbor mask.
   - aggregate with typed gated sum/mean into context `h_context`: `[B, Q, D]`.
   - concatenate/add target coordinate and meta embeddings.
   - MLP with LayerNorm returns `h`: `[B, Q, D]`.
   - Self token for the target square must never be included in its own context.
6. `ClassConditionalTokenDecoder`:
   - for each class `c in {0,1}`, add class embedding and optional low-rank class adapter.
   - output token logits `pred_token_logits_c`: `[B, Q, 13]`.
7. `PseudoLikelihoodScorer`:
   - compute weighted per-square cross-entropy against `token_ids[:, q:q+Q]` for each class.
   - accumulate scores `S`: `[B, 2]`.
   - normalize by total active square weight per board to keep scores comparable across material counts.
8. Final logits:
   - `logits = -S / softplus(score_temperature) + class_bias`
   - output shape: `[B, 2]`.

### Parameter-count estimate

With `hidden_dim=96`, `decoder_hidden_dim=192`, about 18 relation/distance embeddings, and two light class-conditioned decoders:

- token + coordinate embeddings: about 7.4k parameters.
- metadata MLP: about 10k to 25k depending on en-passant summary size.
- relation/distance embeddings and gated projections: about 100k to 250k.
- decoder MLP and class adapters: about 60k to 150k.
- normalization, temperature, biases: negligible.

Expected total: roughly `0.25M` to `0.7M` parameters. This is intentionally smaller than many CNN baselines; success should come from the scoring constraint, not width.

### FLOP and memory estimate

Let:

- `B` = batch size.
- `M = 64` target squares.
- `K = max_neighbors`.
- `D = hidden_dim`.
- `Q = target_chunk_size`.

Aggregation memory if neighbor tensors are materialized by chunk:

\[
4 \cdot B \cdot Q \cdot K \cdot D \text{ bytes}
\]

for float32, before intermediate MLP activations. With `B=512`, `Q=8`, `K=40`, `D=96`, the gathered neighbor tensor is about `63 MB`. With `Q=16`, it is about `126 MB`. Use `Q=8` by default and reduce batch size to `256` if GPU memory is tight.

Approximate compute per forward pass:

\[
O(B \cdot M \cdot K \cdot D + B \cdot M \cdot D \cdot H + 2B \cdot M \cdot H \cdot 13),
\]

where `H = decoder_hidden_dim`. For `B=512`, `M=64`, `K=40`, `D=96`, `H=192`, this is in the low hundreds of millions of multiply-add scale, comparable to a small CNN training step.

### Required config fields

```yaml
model:
  name: geometry_pseudolikelihood_ratio
  input_channels: 18
  num_classes: 2
  hidden_dim: 96
  decoder_hidden_dim: 192
  max_neighbors: 40
  target_chunk_size: 8
  empty_square_weight: 0.25
  nonempty_square_weight: 1.0
  score_temperature_init: 1.0
  relation_dropout: 0.05
  adapter: simple18_token
  randomize_relations: false
  unary_only: false
  auxiliary_pl_weight: 0.05
```

### Encoding support and adapter assumptions

- `simple_18`: first experiment. Deterministic tokenization is allowed because the 12 piece planes and metadata semantics are known by the project context.
- `lc0_static_112`: support only if Codex can verify a current-board piece-plane mapping from repository encoding metadata. If mapping is unknown, raise `ValueError("GeomPLR requires verified current-board piece-channel semantics")`.
- `lc0_bt4_112`: same as `lc0_static_112`; unavailable history planes must not be used by deterministic geometry. If a learned history adapter is later added, it must be a separate neural branch and ablated against current-board-only operation.
- All adapters must fail closed when channel semantics are unknown.

### Pseudocode

```python
class GeometryPseudoLikelihoodRatioNet(nn.Module):
    def forward(self, x):
        token_ids, meta = self.adapter(x)              # [B,64], [B,M]
        e = self.token_embed(token_ids)                # [B,64,D]
        e = e + self.coord_embed[None, :, :]
        e = e + self.meta_mlp(meta)[:, None, :]

        scores = zeros([B, 2], device=x.device)
        weights = token_weights(token_ids)             # [B,64]

        for q0 in range(0, 64, self.target_chunk_size):
            q1 = min(q0 + self.target_chunk_size, 64)
            nb_idx, rel_id, dist_id, valid = self.relation_index.slice(q0, q1)
            h = self.aggregator(e, nb_idx, rel_id, dist_id, valid, meta)  # [B,Q,D]

            for c in (0, 1):
                pred = self.decoder(h, class_id=c)      # [B,Q,13]
                ce = cross_entropy_per_square(pred, token_ids[:, q0:q1])
                scores[:, c] += (weights[:, q0:q1] * ce).sum(dim=1)

        scores = scores / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        logits = -scores / softplus(self.score_temperature) + self.class_bias
        return logits
```

The model returns ordinary logits compatible with the shared trainer. A custom idea-local `train.py` may additionally retrieve `scores` through a debug method or returned loss dictionary, but the normal `forward` return must remain `[B, 2]`.

## 8. Loss, Training, And Regularization

- Primary loss: class-balanced binary cross-entropy on returned logits.
- Optional auxiliary loss: true-class pseudo-likelihood score `S_Y`, weighted by `auxiliary_pl_weight = 0.05`. This stabilizes the local conditional decoders. If the shared trainer cannot handle auxiliary losses, set this to `0` and rely on CE through the pseudo-likelihood ratio logits.
- Class weighting: balanced weighting from the training split, same policy as existing baselines.
- Batch size expectation: start with `512` on `simple_18` and `target_chunk_size=8`; reduce to `256` if memory exceeds budget.
- Optimizer: AdamW.
- Learning rate: `1e-3` for the first run.
- Weight decay: `1e-4`.
- Epochs: `3` for minimal benchmark parity; allow early stopping patience `2`.
- Mixed precision: false for first determinism run; optional later after numeric parity is verified.
- Regularizers:
  - relation dropout `0.05`, deterministic under seed, disabled for final evaluation.
  - token embedding dropout `0.05` optional.
  - empty-square weight `0.25` to reduce dominance of empty squares.
  - no data augmentation until transform semantics are proven.
- Determinism requirements:
  - fixed seed `42`.
  - deterministic relation index construction.
  - deterministic random-neighborhood ablation generated once from seed and saved in the report.
  - no nondeterministic CUDA paths if the existing project supports deterministic mode.
- Must stay unchanged for fair comparison:
  - train/val/test split paths.
  - binary label mapping.
  - benchmark metrics.
  - report format and `3x2` fine-label diagnostic.
  - baseline training budget as much as practical.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Degree-preserving randomized neighborhoods | Replace typed chess-geometric neighbors with random square neighbors preserving per-target degree, distance-bucket counts, edge-count near board boundaries, token visibility, and parameter count. | The specific chess geometry, not merely relational capacity, carries the signal. | If performance is within `0.003` AUROC or within 10% of the main model's gain, the geometry claim is not supported. |
| Unary-only pseudo-likelihood | Remove all neighbor context; each class predicts square tokens from square coordinate, side/castling/en-passant metadata, and class only. | Main model improves beyond material, occupancy, and coordinate priors. | If unary-only matches main, the model is mostly exploiting material/placement priors. |
| Material/meta histogram baseline head | Feed only deterministic counts of piece types, side-to-move, castling, and en-passant summary into a small MLP. | Board dependency structure adds value beyond obvious nuisance variables. | If histogram matches main, abandon `GeomPLR` as a source/material artifact detector. |
| Relation-label shuffle | Keep neighbor squares but randomly permute relation labels and distance labels. | Typed relation semantics matter beyond generic nearby context. | If unchanged, relation typing is unnecessary; simplify or abandon typed geometry. |
| Coordinate-scrambled context | Keep token multiset and material, but permute neighbor coordinates consistently within each board during training/eval. | Square-specific arrangement matters, not just piece multiset. | If unchanged, the model is not using spatial chess structure. |
| No class-conditioned decoder | Use one shared pseudo-likelihood decoder plus a small discriminative MLP on aggregated scores. | Class-conditional description lengths are central, not just a learned context encoder. | If this matches main, the PLR density-ratio interpretation is weak. |
| Direct classifier on aggregator states | Pool `h_i` and train a normal MLP classifier, same parameter scale, no token reconstruction score. | Pseudo-likelihood scoring beats an ordinary relational classifier. | If direct classifier wins, use the aggregator as a future baseline but drop the MDL claim. |
| Empty-square weight set to `1.0` | Let empty squares contribute equally to occupied squares. | Downweighting empties prevents trivial empty-pattern domination. | If equal weighting improves without hurting ablation gaps, keep equal weights; if it closes gaps, empties were a shortcut. |
| Positive-class split diagnostic | Train same binary target but report `S_0-S_1` distributions separately for fine labels `1` and `2`. | Near-puzzles should be intermediate or closer to puzzles under PLR score. | If class `1` behaves exactly like class `0`, the idea may not help near-puzzle recall. |
| Class-label permutation sanity check | Train with shuffled binary targets for a tiny run. | Pipeline does not leak labels or source metadata into inputs. | Any above-chance validation result indicates a serious leakage or split bug. |

There is no rule-generated legal move set, capture set, or move-delta candidate set in this idea. The candidate set is the fixed static neighbor multiset. The randomized-neighborhood, unary-only, material/meta, coordinate-scrambled, and relation-label-shuffle ablations preserve candidate count, degree, material, side-to-move, and source-square marginals while destroying the proposed semantics.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- Existing simple CNN on `simple_18` with the same split and training budget.
- Existing residual CNN small/medium/deep on `simple_18`.
- Existing LC0-style CNN/residual baselines only as secondary reference, because the first `GeomPLR` experiment uses `simple_18`.
- Material/meta histogram MLP ablation from this packet.
- Direct aggregator classifier ablation from this packet.

Metrics to inspect:

- Accuracy.
- Balanced accuracy.
- AUROC.
- Average precision / AUPRC.
- F1 at default threshold.
- Log loss.
- Brier score.
- Expected calibration error if the project already computes it.
- `3x2` fine-label diagnostic matrix: true fine label `0/1/2` to predicted binary output `0/1`.
- Near-puzzle diagnostic: class `1` recall at a matched fine-label-`0` false-positive rate, preferably at the FPR achieved by the best simple18 baseline and also at fixed FPR caps `1%`, `5%`, and `10%` if enough examples exist.

Required artifacts:

- Main model config and checkpoint path.
- Validation and test metrics JSON/Markdown.
- Main model predictions file with per-example logits/probabilities and fine label if the existing report system supports it.
- `3x2` diagnostic matrix for main model and each central ablation.
- Ablation configs and results.
- Score histograms of `S_0`, `S_1`, and `S_0-S_1` by fine label.
- Reliability/calibration plot if already supported.
- A short report explaining whether randomized-neighborhood and unary controls closed the gap.

Success threshold:

- Main `GeomPLR` test AUROC is at least `0.015` absolute above the best same-budget `simple_18` CNN/residual baseline, or, if AUROC is flat, class-`1` recall at matched fine-label-`0` FPR improves by at least `0.05` absolute without reducing fine-label-`2` recall by more than `0.02` absolute.
- The degree-preserving randomized-neighborhood ablation loses at least half of the main model's AUROC gain over the best simple18 baseline, or loses at least `0.010` AUROC absolute if the main gain is larger.
- Unary/material controls must not match the main model within `0.003` AUROC.

Failure threshold:

- Main model improves by less than `0.005` AUROC over the best same-budget simple18 baseline and does not improve class-`1` recall at matched FPR.
- Randomized-neighborhood or unary/material ablation matches the main model within `0.003` AUROC and within `0.02` class-`1` recall at matched FPR.
- Calibration is substantially worse than the baseline while accuracy/AUROC is not better.

What result would make us abandon the idea:

- A material/meta histogram or unary-only pseudo-likelihood model matches the full `GeomPLR` on test metrics and near-puzzle diagnostics.
- The degree-preserving randomized-neighborhood model matches the main model across two seeds.
- The score decomposition shows `S_0-S_1` is dominated by empty squares or gross material count rather than occupied geometric contexts.

What result would justify scaling:

- `GeomPLR` beats same-budget simple18 baselines on validation and test, improves class-`1` recall at matched fine-label-`0` FPR, and loses its advantage under randomized-neighborhood and unary controls.
- Only then consider larger hidden dimension, longer training, LC0 current-board adapters, or combining PLR logits with a baseline CNN head.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_geom_plr/idea.yaml` | Create | Machine-readable summary of this idea, status `draft`, config path, model path, target task, and falsification ablation. |
| `ideas/20260421_geom_plr/math_thesis.md` | Create | Section 6 math thesis, proposition, proof sketch, assumptions, counterexamples, and self-critique. |
| `ideas/20260421_geom_plr/architecture.md` | Create | Module descriptions, tensor shapes, relation index, scoring equations, adapter rules, and pseudocode. |
| `ideas/20260421_geom_plr/implementation_notes.md` | Create | Practical notes for tokenization, relation construction, chunking, deterministic randomized ablation, and fail-closed LC0 adapters. |
| `ideas/20260421_geom_plr/trainer_notes.md` | Create | Loss handling, optional auxiliary pseudo-likelihood term, class weighting, metrics, and shared-trainer compatibility. |
| `ideas/20260421_geom_plr/ablations.md` | Create | The ablation table from Section 9 plus exact config switches for each ablation. |
| `ideas/20260421_geom_plr/train.py` | Create | Thin idea-local entrypoint that calls existing dataset/report utilities, supports optional auxiliary PL loss if the shared trainer cannot, and writes standard reports. |
| `ideas/20260421_geom_plr/config.yaml` | Create | Minimal training config for `simple_18`, `GeometryPseudoLikelihoodRatioNet`, `epochs=3`, `batch_size=512`, seed `42`. |
| `ideas/20260421_geom_plr/report_template.md` | Create | Template requiring main metrics, `3x2` matrix, near-puzzle matched-FPR diagnostic, score histograms, and ablation comparison. |
| `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this packet to imported research memory after implementation; add anti-duplicate notes for class-conditional board pseudo-likelihood/MDL density-ratio models if the experiment fails. |
| `src/chess_nn_playground/models/geometry_pseudolikelihood_ratio.py` | Create | Implement `GeometryPseudoLikelihoodRatioNet`, token adapter, relation index, typed neighbor aggregator, class-conditioned decoder, and config-driven ablation switches. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `geometry_pseudolikelihood_ratio` builder. |
| `configs/geom_plr_simple18.yaml` | Create | Project-level config pointing to existing split, `simple_18`, binary mode, balanced class weighting, deterministic seed. |
| `configs/geom_plr_simple18_random_neighbors.yaml` | Create | Central randomized-neighborhood ablation config. |
| `configs/geom_plr_simple18_unary.yaml` | Create | Unary-only pseudo-likelihood ablation config. |
| `configs/geom_plr_simple18_histogram.yaml` | Create | Material/meta histogram control config if repo config style supports it; otherwise place in idea folder only. |
| `tests/test_geometry_pseudolikelihood_ratio.py` | Create | Focused tests: forward shape `[B,2]`, no self-neighbor leakage, deterministic relation index, invalid LC0 semantics fail closed, randomized ablation preserves degree counts, simple18 tokenization sanity. |
| `tests/test_geom_plr_training_smoke.py` | Create if useful | Tiny synthetic batch smoke test confirming loss decreases for a few steps and reports can consume logits. |

Implementation notes:

- Keep the normal model `forward` return as logits only.
- If auxiliary loss support requires broader trainer edits, gate it behind the idea-local `train.py`; do not break existing shared trainer behavior.
- Add deterministic relation-index unit tests before benchmarking.
- Store the generated random-neighborhood mapping in the run artifact for reproducibility.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0713_tuesday_local_geom_plr.md
  generated_at: 2026-04-21 07:13:46 America/Los_Angeles
  weekday: Tuesday
  timezone: local
  idea_slug: geom_plr
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_geom_plr
  name: Geometry-Conditioned Board Pseudo-Likelihood Ratio Network
  slug: geom_plr
  status: draft
  created_at: 2026-04-21 07:13:46 America/Los_Angeles
  author: ChatGPT Pro
  short_thesis: Classify puzzle-likeness by a class-conditioned pseudo-description-length ratio over current-board piece tokens predicted from static chess-geometric neighborhoods.
  novelty_claim: Replaces direct discriminative logits with leave-one-square class-conditional pseudo-likelihood scores; falsified by degree-preserving randomization and unary/material controls.
  expected_advantage: Better near-puzzle sensitivity and artifact resistance if puzzle-like positions have recurring geometric dependency patterns not captured by ordinary CNN filters.
  central_falsification_ablation: Degree-, distance-, and edge-count-preserving randomized static neighborhoods with all other scoring machinery unchanged.
  target_task: coarse_binary
  input_representation: simple_18
  output_heads: binary_logits_from_two_pseudo_likelihood_scores
  compute_notes: About 0.25M-0.7M parameters; chunk target squares to keep neighbor gather memory near 63MB for B=512,Q=8,K=40,D=96.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/geom_plr_simple18.yaml
  model_path: src/chess_nn_playground/models/geometry_pseudolikelihood_ratio.py
  latest_result_path: null
  notes: Use idea-local train.py only if auxiliary pseudo-likelihood loss cannot be represented in the shared trainer; forward must return logits only.
```

```yaml
config_yaml:
  run:
    name: geom_plr_simple18
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
    name: geometry_pseudolikelihood_ratio
    input_channels: 18
    num_classes: 2
    hidden_dim: 96
    decoder_hidden_dim: 192
    max_neighbors: 40
    target_chunk_size: 8
    empty_square_weight: 0.25
    nonempty_square_weight: 1.0
    score_temperature_init: 1.0
    relation_dropout: 0.05
    adapter: simple18_token
    randomize_relations: false
    unary_only: false
    auxiliary_pl_weight: 0.05
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
  model_name: geometry_pseudolikelihood_ratio
  file_path: src/chess_nn_playground/models/geometry_pseudolikelihood_ratio.py
  builder_function: build_geometry_pseudolikelihood_ratio
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18TokenAdapter
    - StaticChessRelationIndex
    - TypedNeighborAggregator
    - ClassConditionalTokenDecoder
    - PseudoLikelihoodScorer
  required_config_fields:
    - input_channels
    - num_classes
    - hidden_dim
    - decoder_hidden_dim
    - max_neighbors
    - target_chunk_size
    - empty_square_weight
    - adapter
    - randomize_relations
    - unary_only
  expected_parameter_count: 0.25M-0.7M
  expected_memory_notes: Neighbor gather memory is approximately 4*B*Q*K*D bytes; default B=512,Q=8,K=40,D=96 uses about 63MB before MLP activations. Chunk target squares and reduce batch size if needed.
```

```yaml
research_continuity:
  idea_fingerprint: current-board piece tokens plus metadata -> fixed static chess-geometric leave-self-out neighborhoods -> class-conditioned local token conditionals -> summed pseudo-NLL scores -> binary description-length ratio logits
  already_researched_family_overlap: Avoids imported tactical sheaf/Hodge, one-ply move-delta, Sinkhorn/OT, and deterministic nuisance-projection families. It has mild overlap with generic energy-based and masked reconstruction models, but not with existing chess packets.
  closest_duplicate_risk: A future packet might rename this as masked board MDL, class-conditional denoising chess autoencoder, or pseudo-likelihood MRF classifier; treat those as duplicates unless the observable is no longer the S0-S1 board-token pseudo-description-length ratio.
  do_not_repeat_if_this_fails:
    - Class-conditioned current-board token pseudo-likelihood ratio using static chess neighborhoods.
    - Masked/leave-one-square board-token reconstruction whose logits are S0-S1 description-length scores.
    - Degree-preserving randomized-neighborhood ablations over the same static square relation system.
    - Empty/nonempty weighted board-token MDL classifiers without a new source of signal or a different falsifier.
  suggested_next_search_directions:
    - Label-safe ordinal/selective prediction for fine labels 0/1/2 if representation ideas stall.
    - Multi-environment causal invariance across encodings or data-source shifts without using provenance as input.
    - Calibration-first models that expose near-puzzle ambiguity rather than improving raw AUROC.
    - Non-move-tree generative compression with a different observable, such as grammar induction over legal FEN components, only if this pseudo-likelihood ratio fails for structural reasons.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Geometry-Conditioned Board Pseudo-Likelihood Ratio Network` to imported research memory after implementation, including its fingerprint. | Prevents the next research pass from renaming this as masked board MDL, pseudo-likelihood MRF, or class-conditional denoising classifier. | `Imported Research Memory` |
| Add an anti-duplicate rule: do not propose another current-board class-conditional board-token pseudo-likelihood/description-length-ratio classifier over static square neighborhoods unless the formal observable and falsifier change. | Avoids repeated MDL/pseudo-likelihood variants if this fails. | `Do not propose...` anti-duplicate block |
| Require future generative/compression ideas to specify whether auxiliary losses work with the shared trainer or require idea-local training. | Prevents handoff ambiguity around logits-only trainer compatibility. | `Required Markdown File Content`, Sections 7 and 8 guidance |
| Require fail-closed deterministic adapters whenever an idea tokenizes semantic channels from LC0-style encodings. | Avoids accidental use of unknown history/current-board channel semantics. | `Problem Restatement And Data Contract` |
| Add explicit ablation guidance for density-ratio/generative classifiers: unary-only, randomized topology, material/meta histogram, and direct-discriminative replacement. | Makes future falsification plans sharper and less dependent on author memory. | `Ablation Plan` requirements |
| Preserve the existing leakage, label, and anti-duplicate constraints unchanged. | These constraints are essential and should not be weakened. | All hard-constraint sections |

## 14. Final Sanity Check

- Downloadable Markdown file created: yes.
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0713_tuesday_local_geom_plr.md`.
- No forbidden engine features used as inputs: yes.
- Does not fabricate labels: yes.
- Not a routine CNN/ResNet/Transformer variant: yes.
- Minimal current-data experiment exists: yes, `simple_18` on the current `crtk_sample_3class` split.
- Falsification criterion is concrete: yes, degree-preserving randomized neighborhoods plus unary/material controls.
- Codex can implement without asking for missing architecture details: yes.
- Prompt maintenance notes included for Codex: yes.
- Repetition check against imported research packets completed: yes.
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes.
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes.
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes.
