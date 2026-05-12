# Codex Handoff Packet: Entropic Chess Geometry Transport Network

## 1. File Metadata

- Filename: chess_nn_research_2026-04-21_0703_tuesday_los_angeles_geom_ot.md
- Generated at: 2026-04-21 07:03:27 PDT (-0700)
- Weekday: Tuesday
- Timezone: America/Los_Angeles
- Idea slug: geom_ot
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Entropic Chess Geometry Transport Network, abbreviated ECGT-Net.
- One-sentence thesis: Puzzle-like positions often contain an unusually organized transport geometry from the side-to-move material toward opponent tactical target zones, and an entropic optimal-transport layer can expose that geometry without engine evaluations, search lines, legal move counts, or one-ply move-delta bags.
- Idea fingerprint: current board occupancy and side-to-move -> side-to-move piece atoms plus opponent target atoms -> chess-distance cost matrix -> entropic Sinkhorn transport plan -> flow summaries and pressure maps fused with a small CNN -> binary puzzle-likeness logits.
- Closest baseline or common method it resembles: a small CNN with deterministic feature augmentation, plus the optimal-transport pooling idea from computational OT; it is closest in implementation burden to adding a handcrafted differentiable layer before a CNN, not to adding depth or width.
- Why this is not a common CNN/ResNet/Transformer variant: the central computation is a constrained variational problem over a board-conditioned cost matrix, with fixed chess-geometric costs and marginal constraints; the CNN trunk only consumes the resulting transport summaries and raw board tensor.
- Current-data minimal experiment: train ECGT-Net on `data/splits/crtk_sample_3class/split_train.parquet`, validate on `split_val.parquet`, test on `split_test.parquet`, using `simple_18` first and the existing coarse binary task where fine labels `1` and `2` map to output `1`.
- Smallest central falsification ablation: replace the chess-distance cost matrix with a nuisance-preserving randomized cost matrix that keeps source count, target count, side-to-move material histogram, target-role histogram, and each type-role cost histogram, then rerun the same Sinkhorn layer and CNN.
- Expected information gain if it fails: a clean failure says puzzle-likeness in this split is not being captured by side-to-move material-to-target transport geometry beyond material/role/count shortcuts, so the next cycle should pivot toward label-safe ordinal calibration, causal invariance, or uncertainty modeling rather than another geometric board operator.

## 3. Problem Restatement And Data Contract

The project is `chess-nn-playground`. The task is board-position chess puzzle-likeness classification. The model receives a tensor of shape `(batch, C, 8, 8)` and returns logits of shape `(batch, 2)`. The binary target is:

- output `0`: non-puzzle.
- output `1`: puzzle-like.

The available fine labels are:

- fine label `0`: known non-puzzle.
- fine label `1`: verified near-puzzle.
- fine label `2`: verified puzzle.

For the default binary benchmark, labels `1` and `2` are positive. Reports must still include the rectangular diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

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

The full Parquet dataset has roughly 45M rows, but this packet must not point the non-streaming trainer directly at the full file.

Leakage checklist:

- Safe as neural-network inputs or deterministic rule-derived features: board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal current-board geometry derived only from the current board.
- ECGT-Net uses only current board occupancy, side-to-move, deterministic target atoms, deterministic piece-square geometry tables, and learned parameters trained from labels.
- ECGT-Net must not use Stockfish scores, PVs, node counts, mate scores, verification metadata, dataset source labels, proposed labels, puzzle IDs, or any label-derived feature as model input.
- Fine label is used only for training target construction, class weighting, diagnostics, and reporting. It is not a model input.
- Full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences are not used by the central operator. If future experiments add any of these, they must be separately justified as rule-only, label-independent, engine-free, and ablated.
- For `lc0_static_112` and `lc0_bt4_112`, deterministic geometry may only read channels whose current-board semantics are explicitly registered and unit-tested. History channels may be passed through a learned neural adapter, but they must not feed the deterministic atom builder unless their exact current-board meaning is known. Unknown channel layouts must fail closed.

Boundary between safe rule-derived features and leakage:

- Deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack geometry derived only from the current board are allowed.
- Full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences are leakage-prone unless explicitly justified as rule-only, label-independent, engine-free, and ablated.
- Engine evaluation, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, and dataset provenance must never be neural-network inputs.
- For `lc0_static_112` and `lc0_bt4_112`, current-board channels used for deterministic geometry must be distinguished from history channels used only by learned neural adapters.

## 4. Research Map

External ideas used, verified during this research pass:

1. Marco Cuturi, “Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances,” NeurIPS 2013 / arXiv 1306.0895, https://arxiv.org/abs/1306.0895. Borrowed: entropic regularization of optimal transport and the Sinkhorn-Knopp scaling view. Not copied: the image-retrieval task, MNIST experiments, or any data-specific method.
2. Gabriel Peyré and Marco Cuturi, “Computational Optimal Transport,” Foundations and Trends in Machine Learning 2019 / arXiv 1803.00567, https://arxiv.org/abs/1803.00567. Borrowed: the numerical framing of OT as a scalable geometry-aware operator for probability measures. Not copied: large-scale OT algorithms beyond a small fixed Sinkhorn loop.
3. Nicolas Courty, Rémi Flamary, Devis Tuia, and Alain Rakotomamonjy, “Optimal Transport for Domain Adaptation,” IEEE TPAMI 2017 / arXiv 1507.00504, https://arxiv.org/abs/1507.00504. Borrowed only as supporting evidence that OT is a useful way to express geometry-aware alignment. Not copied: domain adaptation objective, source/target distribution matching, or any use of dataset provenance.
4. Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz, “Invariant Risk Minimization,” arXiv 1907.02893, https://arxiv.org/abs/1907.02893. Borrowed for the candidate search only: the idea that stable structure should survive nuisance shifts. Not copied into the selected architecture because the current split may not provide clean environment labels.
5. Adrien Bardes, Jean Ponce, and Yann LeCun, “VICReg,” ICLR 2022 / arXiv 2105.04906, https://arxiv.org/abs/2105.04906. Borrowed for the candidate search only: explicit anti-collapse regularization for view agreement. Not copied into the first experiment because the selected hypothesis should be falsified by transport semantics, not by a broad self-supervised regularizer.

Candidate search trace. I considered the following mechanisms before selecting ECGT-Net:

1. Causal environment-invariant classifier across material phase buckets, side-to-move transforms, and encoding families. Rejected for this cycle because the current prompt does not guarantee clean environment labels; it is better as a follow-up if geometry fails.
2. Label-safe ordinal/selective model treating fine labels `0 < 1 < 2` with binary deployment. Rejected as the primary idea because it changes the loss and calibration more than the board representation; it is valuable but less likely to reveal a new chess-specific inductive bias.
3. Information bottleneck with adversarial removal of material histogram. Rejected because suppressing material too hard may erase real tactical signal, and adversarial nuisance heads are easy to tune into instability.
4. Optimal transport between side-to-move material and opponent target zones. Selected because it gives a clear chess-specific operator, a small implementation, and a direct randomized-cost falsification.
5. Diffusion/noising autoencoder over legal-looking boards with puzzle-likeness readout. Rejected because it would require careful generation or corruption semantics and could become a generic representation-learning project.
6. Spectral positional kernel on piece-square distributions. Rejected because spectra alone are too close to handcrafted feature engineering and harder to localize in the diagnostic confusion matrix.
7. Hypergraph of king-zone motifs without sheaf maps. Rejected because it risks duplicating imported tactical incidence/sheaf families under a new name.
8. Piece-square masked autoencoder with binary probe. Rejected because ordinary masked modeling over 64 squares is too close to a vanilla Transformer/self-supervised baseline.
9. Energy-based latent “tactic witness” variable. Rejected because it would need negative sampling design and may complicate the shared trainer.
10. Side-to-move/file-mirror equivariant CNN. Rejected because symmetry handling alone is likely a helpful engineering improvement, not a distinct hypothesis about puzzle-likeness.
11. Differentiable material-balance counterfactuals without moves. Rejected because it drifts toward the imported counterfactual move-delta family unless very tightly constrained.
12. Calibration-first conformal abstention for near-puzzles. Rejected because it is a reporting/deployment layer rather than a central model architecture for the current benchmark.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN on `simple_18` | `src/chess_nn_playground/models/cnn.py` | Already exists and tests generic local spatial filters without the proposed transport variational operator. |
| Residual CNN small/medium/deep | `src/chess_nn_playground/models/residual_cnn.py` | Already exists and mainly changes optimization depth, not the inductive bias. |
| LC0-style CNN or residual CNN on `lc0_bt4_112` | Existing LC0 BT4-style CNN/residual variants | Too close to copying a known board-game architecture and already covered by the baseline suite. |
| Ordinary ViT over 64 square tokens | Common square-token Transformer | Too generic, parameter-hungry for the current split, and explicitly disallowed as a core idea. |
| Plain GNN on square adjacency | Common graph neural network over 8-neighbor or chessboard graph | It is a standard message-passing reformulation of the board and not a falsifiable new puzzle-likeness operator. |
| Hyperparameter tuning, optimizer tuning, depth/width search | Any existing baseline | Disallowed and unlikely to reveal a new mechanism. |
| Ensembling multiple current models | Any baseline ensemble | Disallowed as the core idea and weak for scientific diagnosis. |
| Training directly on the full 45M-row Parquet file | Existing trainer with larger data | Unsafe until streaming support exists; also “add more data” is not a research mechanism. |
| Tactical sheaf/Hodge/Laplacian/tension/curvature variant | Imported tactical sheaf/Hodge packets | Already researched; adding edge labels, bigger hidden sizes, or new pooling would be duplication. |
| One-ply pseudo-legal move-delta DeepSets/attention/spectrum/landscape | Imported counterfactual move-delta packets | Already researched and explicitly excluded unless the operator is mathematically different. |
| Engine evaluation, PV, node count, mate score, or verification metadata features | None allowed | Leaky and forbidden. |
| Full legal-move count, checkmate/stalemate oracle, or forced-line indicator | Rule-engine feature baseline | Leakage-prone and not needed for this hypothesis. |
| Causal IRM across material buckets | Serious candidate, no current baseline | Interesting, but environment definitions would be partly arbitrary on the current split, making failure hard to interpret. |
| Label-safe ordinal/selective classifier | Serious candidate, no current baseline | Worth future work, but less chess-structural than transport geometry and not as direct a model-side novelty. |
| Diffusion or masked-board pretraining | Serious candidate, no current baseline | Too broad for one Codex cycle and risks measuring generic pretraining quality rather than puzzle structure. |
| Non-sheaf hypergraph motif network | Serious candidate, no current baseline | Too close to attack/incidence graph families already imported, even if sheaf terminology is removed. |

## 6. Mathematical Thesis

### Input space and target

Let `E` be an encoding family with tensors in

\[
\mathcal X_E \subset \mathbb R^{C_E \times 8 \times 8}.
\]

For the first experiment, `E = simple_18`, and a deterministic parser maps \(x \in \mathcal X_E\) to a current board state with piece occupancy, side-to-move, and optional castling/en-passant metadata. The binary label is

\[
Y = \mathbf 1\{Z \in \{1,2\}\},
\]

where \(Z \in \{0,1,2\}\) is the fine label. The model returns logits \(f_\theta(x) \in \mathbb R^2\).

### Data distribution assumptions

The split is assumed to contain a mixture of ordinary non-puzzles, verified near-puzzles, and verified puzzles. The hypothesis is not that puzzle-likeness is determined only by material geometry. The weaker assumption is that, after conditioning on material and side-to-move, puzzle-like positions are enriched for a structured alignment between side-to-move pieces and opponent target zones. This alignment should not collapse to merely “having more pieces,” “having queens,” or “opponent king exposed.”

### Allowed symmetry or equivariance assumptions

Chess is not invariant under arbitrary rotations or reflections. Pawns have direction, castling is side-specific, en-passant is file/rank-specific, and side-to-move matters.

ECGT-Net may use only these conservative symmetries:

1. Side-to-move perspective normalization: treat the side to move as “us” and the opponent as “them.” Pawn direction is defined in this canonical perspective.
2. File mirror equivariance: mirroring files `a <-> h`, with castling and en-passant files mirrored consistently, should preserve puzzle-likeness in the absence of dataset artifacts.
3. No assumption of 90-degree rotation, raw rank reflection without color/pawn-direction handling, or color-invariant evaluation without side-to-move normalization.

### Formal object introduced

For a board \(x\), define side-to-move source atoms

\[
S(x)=\{(p_i, s_i): i=1,\ldots,m(x)\},
\]

where \(p_i\) is a piece type and \(s_i\in \{1,\ldots,64\}\) is a square. Define opponent target atoms

\[
T(x)=\{(r_j, t_j): j=1,\ldots,n(x)\},
\]

where \(r_j\) is a target role such as opponent king square, opponent king ring square, opponent queen/rook square, opponent minor-piece square, opponent pawn square, or promotion-rank anchor, and \(t_j\) is a square.

Let \(a(x) \in \Delta^{m(x)-1}\) and \(b(x) \in \Delta^{n(x)-1}\) be source and target marginals. In the first implementation, these are generated from nonnegative learned type/role weights, masked by atom existence, then normalized. No label, engine score, or source metadata enters these marginals.

Let \(d_p(s,t)\) be a deterministic empty-board chess-distance table for piece type \(p\): knight distance for knights, rook-line distance for rooks, bishop-color-aware distance for bishops, queen as minimum rook/bishop-style distance, king Chebyshev distance, and directional pawn distance in side-to-move perspective with unreachable squares assigned a large finite cap. This is not legal move generation and does not count legal moves from the current position.

Define a board-conditioned cost

\[
C_x(i,j)=\alpha_{p_i,r_j} d_{p_i}(s_i,t_j)+\beta_{p_i,r_j}+\gamma\,\delta(s_i,t_j),
\]

where \(\alpha_{p,r}=\operatorname{softplus}(u_{p,r})\), \(\beta_{p,r}\) is learned, and \(\delta\) is an optional coordinate-distance correction such as Manhattan or Chebyshev distance. The first experiment should keep the cost simple and deterministic; do not add legal move counts or search-derived terms.

The entropic transport plan is

\[
\Pi_\varepsilon(x)=\arg\min_{\pi\ge 0}\; \langle \pi,C_x\rangle + \varepsilon\sum_{i,j}\pi_{ij}(\log \pi_{ij}-1)
\]

subject to

\[
\pi\mathbf 1=b(x),\qquad \pi^\top \mathbf 1=a(x),
\]

with \(\varepsilon>0\). Codex may choose the row/column convention, but it must be consistent. The solution is computed by a fixed small number of log-domain Sinkhorn iterations.

From \(\Pi_\varepsilon(x)\), construct:

- scalar cost \(\langle \Pi_\varepsilon,C_x\rangle\),
- plan entropy \(-\sum_{i,j}\Pi_{ij}\log(\Pi_{ij}+\eta)\),
- type-role flow matrix \(M_{p,r}(x)=\sum_{i:p_i=p}\sum_{j:r_j=r}\Pi_{ij}\),
- outgoing source pressure map on source squares,
- incoming target pressure maps on target squares, grouped by target role.

These features are fused with the raw board tensor by a small CNN trunk.

### Core hypothesis

Puzzle-like positions are more likely than non-puzzles to have nontrivial low-cost, role-concentrated transport from side-to-move material into opponent target atoms. The model should gain specifically from the chess-distance cost semantics, not merely from counts of pieces, target roles, or material values.

### Proposition: equivariance and invariant pooled summaries

Let \(g\) be a file mirror or a valid side-to-move perspective transform whose action on squares and piece/target roles is represented by permutations \(P_g\) on sources and \(Q_g\) on targets. Suppose the cost and marginals transform as

\[
C_{g x}=P_g C_x Q_g^\top,\quad a(gx)=P_g a(x),\quad b(gx)=Q_g b(x).
\]

Then the entropic OT solution satisfies

\[
\Pi_\varepsilon(gx)=P_g\Pi_\varepsilon(x)Q_g^\top.
\]

Consequently, type-role pooled flows and scalar costs are invariant under the transform, while square pressure maps are equivariant.

Proof sketch: for \(\varepsilon>0\), the entropic OT objective is strictly convex over the transport polytope when restricted to positive feasible plans, so the minimizer is unique. If \(\pi\) is feasible for \((a,b,C_x)\), then \(P_g\pi Q_g^\top\) is feasible for \((P_ga,Q_gb,P_gC_xQ_g^\top)\). The linear cost and entropy terms are preserved under permutation. Therefore the unique minimizer must permute in the same way. Pooling over type-role classes removes atom ordering; pressure maps retain the induced square permutation.

### Variational principle for learning

The supervised model minimizes

\[
\min_\theta\; \mathbb E_{(X,Y)}\left[\operatorname{CE}_w(Y,f_\theta(X))\right]
+\lambda_{\text{wd}}\|\theta\|_2^2
+\lambda_{\text{ot}} R_{\text{ot}}(X),
\]

where \(R_{\text{ot}}\) is optional and should default to zero in the minimal experiment. The useful constraint is already the OT variational layer: transport features must obey marginal conservation and chess-distance costs before the CNN can use them.

### What is actually proven

- The entropic OT problem has a unique smooth solution for positive marginals and \(\varepsilon>0\).
- The transport plan is equivariant under valid source/target permutations that preserve the cost and marginals.
- Type-role flow summaries are invariant to atom ordering and valid board mirroring under the stated assumptions.
- The operator does not require engine scores, PVs, node counts, labels as input, source metadata, or move-tree search.

### What remains hypothesized

- That puzzle-like positions in this split have a detectable transport signature beyond material and target counts.
- That simple empty-board chess distances are strong enough to help despite ignoring pins, legal move details, and multi-move forcing lines.
- That the CNN trunk will use transport pressure maps as semantic features rather than treating them as noisy handcrafted channels.

### Counterexamples where the idea should fail

- Zugzwang, fortress, opposition, and quiet endgame-study puzzles whose key feature is not material moving toward a tactical target.
- Tactics depending on a single legal constraint that empty-board distance cannot see, such as pinned blockers, stalemate tricks, underpromotion details, or exact castling legality.
- Positions where the side-to-move has geometrically promising piece placement but every candidate tactic fails tactically.
- Puzzle-like positions whose target is not the opponent king or material, such as tempo-only defensive resources.
- Dataset artifacts where positive and negative examples have similar target geometry but differ by provenance or verification process that is forbidden as input.

### Self-critique

The strongest objection is that ECGT-Net may be a sophisticated material/proximity heuristic: pieces closer to the enemy king and heavy pieces aligned toward targets are already easy for a CNN to learn. The minimal experiment is still worth running because the randomized-cost ablation is unusually sharp. If cost-histogram-preserving random transport matches the main model, the mechanism is dead. If the main model improves and the randomized-cost, uniform-cost, and count-only ablations do not, then the gain is attributable to the chess geometry imposed by the variational transport layer rather than to extra channels or parameters.

## 7. Architecture Specification

### Module names

- `ChessGeometryTransportNet`: top-level `torch.nn.Module` returning logits.
- `EncodingSemanticAdapter`: validates channel semantics and extracts current-board piece planes, side-to-move, castling, and en-passant metadata when available.
- `TransportAtomBuilder`: creates masked source atoms and target atoms.
- `ChessDistanceCost`: builds the chess-distance cost tensor from atom features and precomputed distance tables.
- `LogSinkhornTransport`: computes the entropic transport plan in log-space.
- `TransportFeatureProjector`: converts the plan into scalar features, type-role flow features, and pressure maps.
- `TransportAugmentedCNN`: small CNN or residual-mini trunk consuming raw input plus transport maps.

### First experiment encoding choice

Use `simple_18` first. It has explicit piece planes and enough current-board semantics for deterministic geometry. Codex should not attempt deterministic OT parsing for `lc0_static_112` or `lc0_bt4_112` until a semantic registry maps exact current-board piece planes. For LC0 encodings, a learned neural adapter may process all channels, but deterministic target/source atoms must fail closed if channel semantics are unknown.

### Forward-pass steps and shapes

Assume `B=batch`, `C=18`, `S=max_sources=16`, `T=max_targets=40`, `R=target_roles=6`, and `P=piece_types=6`.

1. Input:
   - `x`: `(B, C, 8, 8)`.

2. Encoding semantic adapter:
   - parse piece occupancy into canonical side-to-move perspective.
   - output `piece_grid`: `(B, 2, 6, 8, 8)` for us/them by piece type, or equivalent masks.
   - output `side_to_move`: `(B,)`.
   - fail closed if requested deterministic geometry is unsupported for the encoding.

3. Source atom builder:
   - side-to-move pieces become source atoms.
   - `source_square`: `(B, S)` integer square index, padded.
   - `source_type`: `(B, S)` integer type in `{pawn, knight, bishop, rook, queen, king}`, padded.
   - `source_mask`: `(B, S)` boolean.
   - `source_marginal a`: `(B, S)`, normalized over valid atoms.

4. Target atom builder:
   - opponent king square: one atom.
   - opponent king ring: up to eight atoms, clipped to board.
   - opponent queen/rook squares: atoms by square.
   - opponent minor-piece squares: atoms by square.
   - opponent pawn squares: optional atoms; keep enabled only if `max_targets` permits.
   - promotion anchors: side-to-move promotion-rank squares, optional low-weight target role.
   - `target_square`: `(B, T)`.
   - `target_role`: `(B, T)` integer role.
   - `target_mask`: `(B, T)` boolean.
   - `target_marginal b`: `(B, T)`, normalized over valid atoms.

5. Cost construction:
   - precomputed distance table `D_piece`: `(6, 64, 64)`.
   - gather `D_piece[source_type, source_square, target_square]` -> `(B, S, T)`.
   - multiply by learned positive type-role scale `alpha`: `(6, R)`.
   - add learned type-role bias `beta`: `(6, R)`.
   - set invalid source/target pairs to a large masked value.
   - output `cost`: `(B, S, T)`.

6. Log-domain Sinkhorn:
   - inputs: `log_a`: `(B, S)`, `log_b`: `(B, T)`, `cost`: `(B, S, T)`.
   - fixed iterations, default `sinkhorn_iters=8`.
   - output `plan`: `(B, S, T)` with masked invalid entries near zero.

7. Feature projection:
   - `flow_by_type_role`: `(B, 6, R)` then flattened to `(B, 6*R)`.
   - scalar features: expected cost, entropy, max row concentration, max column concentration, valid source count, valid target count -> `(B, 6)`.
   - source pressure map: `(B, 1, 8, 8)` by scattering row sums to source squares.
   - target pressure maps: `(B, R, 8, 8)` by scattering column sums to target squares by role.
   - optional cost-pressure map: `(B, 1, 8, 8)` from low-cost incoming flow, disabled by default if implementation time is tight.
   - `transport_maps`: `(B, 1+R, 8, 8)` or `(B, 2+R, 8, 8)`.

8. Board trunk:
   - concatenate `x` and `transport_maps`: `(B, C+1+R, 8, 8)`.
   - run a small CNN/residual-mini trunk, for example three `3x3` conv blocks with hidden width 64 and global average pooling.
   - produce `board_embedding`: `(B, H)`, default `H=128`.

9. Tabular transport head:
   - MLP over `[flow_by_type_role, scalar_features]`: `(B, 6*R+6)` -> `(B, 64)`.

10. Classifier:
   - concatenate board and transport embeddings: `(B, H+64)`.
   - linear classifier returns logits `(B, 2)`.

### Parameter-count estimate

With hidden width 64, `R=6`, and a compact CNN trunk:

- type-role cost scales/biases and marginal weights: under 200 parameters.
- transport feature MLP: roughly 7k-15k parameters.
- CNN trunk with input `18+7=25` channels and 64 hidden channels: roughly 180k-350k parameters depending on block count.
- classifier: roughly 20k parameters.
- Expected total: about 0.25M-0.7M parameters. Keep it within the same order as existing small/medium CNN baselines; the research claim is not “bigger network.”

### FLOP and memory estimate

For batch size `B`, max sources `S=16`, max targets `T=40`, Sinkhorn iterations `K=8`:

- Cost gather and affine terms: `O(B*S*T)`.
- Sinkhorn: `O(B*K*S*T)`, approximately `B*5120` pair operations, much smaller than the CNN trunk.
- Plan memory: `B*S*T` floats. At `B=512`, `16*40*512*4 bytes ≈ 1.25 MB` for one float32 plan tensor. Log buffers and cost roughly triple this, still small.
- Transport maps: `(B, 7, 8, 8)`, about `0.9 MB` at `B=512` float32.

Chunking plan: if future target roles increase `T` above 96, compute Sinkhorn and feature scattering in batch chunks, not atom chunks, because the cost matrix is small but masks are board-specific. Add config `transport_chunk_size`; default `null` means no chunking.

### Required config fields

Minimum new config fields, with safe defaults in the model constructor:

```yaml
transport:
  max_sources: 16
  max_targets: 40
  target_roles: [king_square, king_ring, heavy_piece, minor_piece, pawn, promotion_anchor]
  epsilon: 0.25
  sinkhorn_iters: 8
  distance_cap: 8.0
  use_pressure_maps: true
  use_scalar_transport_features: true
  cost_ablation_mode: none
  fail_closed_semantic_adapter: true
```

### Encoding-adapter assumptions

- `simple_18`: supported in the first experiment. Codex must verify and unit-test the piece-plane order used by the existing dataset loader. If the order cannot be confirmed, the adapter must raise an explicit error rather than silently parsing wrong channels.
- `lc0_static_112`: not supported for deterministic geometry unless the repo already has exact current-board channel semantics. Learned neural adapter support is allowed, but the OT atom builder must fail closed without a semantic registry.
- `lc0_bt4_112`: same as `lc0_static_112`; history planes may be passed to the neural trunk but must not be used to build deterministic atoms unless their current-board semantics are registered. Current zero-filled history does not justify guessing channel meanings.

### Pseudocode, not final implementation

```text
forward(x):
    parsed = semantic_adapter(x, encoding)
    sources = atom_builder.make_sources(parsed)
    targets = atom_builder.make_targets(parsed)
    cost = chess_cost(sources, targets)

    if cost_ablation_mode == "random_cost_histogram_preserving":
        cost = randomized_cost_with_same_masks_and_type_role_histograms(cost, sources, targets)
    if cost_ablation_mode == "uniform":
        cost = uniform_valid_pair_cost(cost, sources, targets)

    plan = log_sinkhorn(cost, sources.log_marginal, targets.log_marginal)
    maps, vector = feature_projector(plan, cost, sources, targets)
    board_embedding = cnn(concat_channelwise(x, maps))
    transport_embedding = mlp(vector)
    logits = classifier(concat(board_embedding, transport_embedding))
    return logits
```

The model must return only logits to the shared trainer. Additional diagnostic tensors may be returned only behind an explicit debug flag that the normal trainer does not use.

## 8. Loss, Training, And Regularization

- Primary loss: weighted cross-entropy over binary labels, using the existing coarse binary mode.
- Auxiliary loss: none required for the minimal experiment. Optional entropy-range regularization on the transport plan may be logged but should default to `0.0` so that the central claim is clean.
- Class weighting: use existing `class_weighting: balanced` behavior. Do not use fine label as an input feature.
- Batch size expectation: `512` on GPU or CPU-compatible smaller batch if memory requires. The OT layer is not the bottleneck.
- Optimizer default: AdamW if already supported; otherwise Adam. Learning rate `0.001`, weight decay `0.0001`.
- Epochs: keep the minimal config at `3` epochs with early stopping patience `2`, matching the project’s benchmark style. Scaling can increase epochs only after the falsification ablations are run.
- Regularizers: standard weight decay, optional dropout `0.05-0.10` in the transport MLP and classifier. Avoid aggressive dropout on pressure maps because it would obscure the central test.
- Determinism requirements: fixed seed `42`, deterministic data split, deterministic distance tables, deterministic atom ordering, deterministic randomized ablation seed, and deterministic PyTorch settings where the repo supports them.
- What must stay unchanged for fair comparison: train/val/test split paths, coarse binary target mapping, fine-label diagnostic report, batch size if feasible, epochs, class weighting, optimizer family if current configs standardize it, and no direct full-dataset training.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Cost-histogram-preserving random OT | Replaces chess-distance costs with randomized costs preserving source masks, target masks, piece-type counts, target-role counts, and per-type-role cost histograms | The specific chess geometry, not just extra OT channels, matters | If this matches the main model, abandon the transport-geometry claim. |
| Uniform-cost OT | Sets all valid source-target pair costs equal, so the plan becomes essentially marginal-only | Marginal conservation alone is insufficient | If this matches the main model, the model is exploiting atom counts/roles, not geometry. |
| Count-only transport vector | Removes plan and costs; feeds only source count, target count, material histogram, target-role histogram, and side-to-move | Obvious nuisance shortcuts are insufficient | If this matches the main model, ECGT-Net is not adding semantic information. |
| No transport maps, vector only | Keeps scalar and type-role flow features but removes pressure maps from the CNN input | Spatial localization of transport pressure matters | If performance stays, maps are unnecessary and the mechanism may be mostly tabular. |
| Maps only, no flow vector | Keeps pressure maps but removes type-role flow and scalar transport MLP | CNN-visible transport pressure is enough | If maps-only works, simplify the architecture; if it fails, pooled role structure matters. |
| Target-role shuffle | Preserves target squares and counts but shuffles target role labels within each position before cost scaling and pooling | Roles such as king ring vs heavy piece are semantically meaningful | If it matches, role definitions are not useful. |
| Piece-type shuffle under material histogram | Preserves source squares and the multiset of piece types in the batch but shuffles type assignments across positions for the OT cost only | Piece-specific chess distances matter | If it matches, knight/bishop/rook/queen geometry is not being used. |
| Distance-table swap | Replaces chess distances with Manhattan/Chebyshev-only distances independent of piece type | Chess-piece movement geometry matters beyond generic board proximity | If it matches, use simpler geometry or drop the idea. |
| Zero-OT same-parameter CNN | Feeds zero transport maps and a learned dummy vector with the same downstream parameter count | Any gain is not merely due to extra parameters | If it matches, the OT layer is unnecessary. |
| File-mirror test-time consistency diagnostic | Evaluate predictions on file-mirrored positions if the loader can mirror safely; do not train on mirrored labels unless already supported | The learned classifier should not depend on left/right board artifacts | Large inconsistency suggests dataset or parser artifacts and weakens the symmetry claim. |

The smallest central falsification ablation is the cost-histogram-preserving random OT. It is semantics-destroying while preserving the nuisance variables most likely to explain a false gain: candidate count, material, side-to-move perspective, source-square marginal up to randomization mode, target count, target-role histogram, and coarse cost distribution.

For all candidate-set ablations, preserve obvious shortcuts where possible: source count, target count, material histogram, side-to-move, moving-piece identity marginal, source-square marginal, target-square marginal, and target-role/capture-like histograms. ECGT-Net does not generate moves or captures, so “capture histogram” translates here to opponent target-role and occupied-target histograms.

## 10. Benchmark And Falsification Criteria

Codex should benchmark the main model and central ablations on the existing split only.

Baselines to compare against:

- Existing simple CNN on `simple_18`.
- Existing residual CNN on `simple_18`, matching the closest parameter budget available.
- Existing LC0-style CNN/residual results should be reported for context if already present, but the main comparison is same-encoding `simple_18` unless deterministic LC0 semantics are registered.
- Zero-OT same-parameter CNN ablation.
- Uniform-cost OT and randomized-cost OT ablations.

Metrics to inspect:

- Test accuracy.
- AUROC.
- AUPRC.
- F1 at the validation-selected threshold.
- Balanced accuracy if already reported.
- Brier score or expected calibration error if available.
- Required `3x2` fine-label diagnostic matrix for every main and central ablation run.
- Class `1` near-puzzle recall at a matched fine-label-`0` false-positive rate. Preferred: choose a threshold on validation so fine-label-`0` FPR equals the best same-encoding baseline’s FPR, then report class `1` recall and class `2` recall on test.

Required artifacts:

- Config YAML for the main model.
- Config YAMLs or explicit CLI flags for central ablations.
- Checkpoint for the main model.
- Metrics JSON/CSV.
- Fine-label `3x2` confusion matrix for main model and central ablations.
- Predictions Parquet/CSV with true fine label, binary label, predicted probability, predicted class, and split identifier.
- Markdown report using the report template in the idea directory.

Success threshold:

- Main ECGT-Net improves test AUROC by at least `+0.010` absolute over the strongest same-encoding baseline, or improves class `1` recall by at least `+0.05` absolute at matched fine-label-`0` FPR without reducing class `2` recall by more than `0.02` absolute.
- The cost-histogram-preserving random OT ablation must lose at least `40%` of the main model’s AUROC gain over the zero-OT same-parameter baseline, or lose at least `0.025` absolute class `1` recall at matched fine-label-`0` FPR.

Failure threshold:

- Main model AUROC gain is less than `+0.003` absolute and near-puzzle class `1` recall does not improve at matched fine-label-`0` FPR.
- Randomized-cost or uniform-cost ablation is within `0.002` AUROC and within `0.01` class `1` recall of the main model.
- Count-only ablation matches the main model within `0.002` AUROC.

What result would make me abandon the idea:

- A cost-randomized or count-only ablation matches the main model while the fine-label diagnostic matrices are nearly identical. That would show that the OT semantics are decoration.

What result would justify scaling:

- Main model clears the success threshold, randomized-cost fails clearly, and the class `1` diagnostic improves at matched fine-label-`0` FPR. Then scale to `lc0_static_112` or `lc0_bt4_112` only after deterministic current-board semantic adapters are registered and tested.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_geom_ot/idea.yaml` | Create | Machine-readable idea metadata copied from the `idea_yaml` block below. |
| `ideas/20260421_geom_ot/math_thesis.md` | Create | Section 6 with equations, proposition, proof sketch, hypotheses, and counterexamples. |
| `ideas/20260421_geom_ot/architecture.md` | Create | Section 7 architecture, shapes, complexity, and pseudocode. |
| `ideas/20260421_geom_ot/implementation_notes.md` | Create | Adapter fail-closed rules, distance table construction, atom ordering, masking, and numerical Sinkhorn notes. |
| `ideas/20260421_geom_ot/trainer_notes.md` | Create | Loss, class weighting, benchmark split, deterministic settings, and unchanged baseline constraints. |
| `ideas/20260421_geom_ot/ablations.md` | Create | Section 9 ablation table plus exact semantics-preserving randomization requirements. |
| `ideas/20260421_geom_ot/train.py` | Create | Thin entrypoint or wrapper invoking the existing shared trainer with `configs/chess_geometry_transport_simple18.yaml`; do not duplicate trainer logic. |
| `ideas/20260421_geom_ot/config.yaml` | Create | Idea-local copy of the main config for reproducibility. |
| `ideas/20260421_geom_ot/report_template.md` | Create | Required report sections: metrics, `3x2` matrix, near-puzzle diagnostic, ablation comparison, leakage checklist. |
| `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this idea to imported memory after Codex consumes it; add anti-duplicate notes for material-target entropic OT if it fails or succeeds. |
| `src/chess_nn_playground/models/chess_geometry_transport.py` | Create | `ChessGeometryTransportNet` and helper modules listed in Section 7. |
| `src/chess_nn_playground/models/registry.py` | Modify | Register model name `chess_geometry_transport`; ensure unknown names still fail clearly. |
| `configs/chess_geometry_transport_simple18.yaml` | Create | Main runnable benchmark config using `simple_18`, coarse binary mode, balanced class weighting, and deterministic seed. |
| `configs/chess_geometry_transport_simple18_random_cost.yaml` | Create | Central randomized-cost ablation config. |
| `configs/chess_geometry_transport_simple18_uniform_cost.yaml` | Create | Uniform-cost ablation config. |
| `configs/chess_geometry_transport_simple18_count_only.yaml` | Create | Count-only nuisance-preserving ablation config. |
| `tests/test_chess_geometry_transport.py` | Create | Unit tests for output shape, fail-closed adapter behavior, atom masks, Sinkhorn marginal sanity, and deterministic cost-randomization reproducibility. |
| `tests/test_chess_distance_tables.py` | Create if needed | Unit tests for empty-board distances: symmetry where valid, pawn direction in side-to-move perspective, caps for unreachable bishop/pawn cases. |

Implementation notes:

- Keep all helper functions deterministic and independent of labels.
- Do not import chess engines.
- Do not call Stockfish or any UCI engine.
- Do not perform full legal move generation.
- Do not add source/provenance columns to the dataset loader as features.
- The normal `forward` method must return logits only.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0703_tuesday_los_angeles_geom_ot.md
  generated_at: "2026-04-21 07:03:27 PDT (-0700)"
  weekday: Tuesday
  timezone: America/Los_Angeles
  idea_slug: geom_ot
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_geom_ot
  name: Entropic Chess Geometry Transport Network
  slug: geom_ot
  status: draft
  created_at: "2026-04-21T07:03:27-07:00"
  author: ChatGPT Pro
  short_thesis: Entropic OT from side-to-move material atoms to opponent target atoms exposes puzzle-like tactical geometry without engine or move-tree inputs.
  novelty_claim: Uses a board-conditioned chess-distance optimal-transport layer rather than a CNN, ResNet, vanilla Transformer, tactical sheaf, attack graph, or one-ply move-delta bag.
  expected_advantage: Better near-puzzle detection at matched non-puzzle false-positive rate if verified near-puzzles share material-to-target organization that generic CNNs underuse.
  central_falsification_ablation: cost_histogram_preserving_random_ot
  target_task: coarse_binary
  input_representation: simple_18 first; lc0_static_112 and lc0_bt4_112 only after fail-closed semantic adapter registration
  output_heads: binary_logits
  compute_notes: Sinkhorn cost is O(batch * max_sources * max_targets * sinkhorn_iters), default 16x40x8 per sample; CNN trunk dominates compute.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/chess_geometry_transport_simple18.yaml
  model_path: src/chess_nn_playground/models/chess_geometry_transport.py
  latest_result_path: null
  notes: Run randomized-cost, uniform-cost, count-only, and zero-OT same-parameter ablations before claiming any geometry gain.
```

```yaml
config_yaml:
  run:
    name: geom_ot_simple18
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
    name: chess_geometry_transport
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
  model_name: chess_geometry_transport
  file_path: src/chess_nn_playground/models/chess_geometry_transport.py
  builder_function: build_chess_geometry_transport_net
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - EncodingSemanticAdapter
    - TransportAtomBuilder
    - ChessDistanceCost
    - LogSinkhornTransport
    - TransportFeatureProjector
    - TransportAugmentedCNN
  required_config_fields:
    - model.name
    - model.input_channels
    - model.num_classes
    - data.encoding
    - transport.max_sources
    - transport.max_targets
    - transport.epsilon
    - transport.sinkhorn_iters
    - transport.cost_ablation_mode
    - transport.fail_closed_semantic_adapter
  expected_parameter_count: approximately 0.25M to 0.7M with hidden width 64
  expected_memory_notes: For batch 512, S=16, T=40, the plan tensor is about 1.25MB float32; cost/log buffers and pressure maps remain small, so CNN activations dominate memory.
```

```yaml
research_continuity:
  idea_fingerprint: current-board side-to-move piece atoms + opponent target atoms + chess-distance cost matrix + entropic Sinkhorn transport + flow/pressure-map CNN fusion + binary puzzle-likeness target
  already_researched_family_overlap: Not a tactical sheaf/Hodge/Laplacian/curvature/tension model and not a one-ply pseudo-legal move-delta DeepSets/attention/spectrum/landscape model.
  closest_duplicate_risk: Could be mistaken for an attack-distance heuristic; distinguish it by the marginal-constrained OT variational layer and cost-randomization falsification rather than attack/defense incidence or move deltas.
  do_not_repeat_if_this_fails:
    - Entropic OT between side-to-move piece atoms and opponent target atoms using empty-board chess-distance costs.
    - Transport pressure maps from piece-to-target Sinkhorn plans.
    - Cost-histogram-preserving randomization as the only central novelty around this operator.
    - Material/target-count OT marginal variants that do not change the formal object.
  suggested_next_search_directions:
    - Label-safe ordinal and selective-prediction models focused on fine-label-1 ambiguity.
    - Causal invariance across encoding families or material phase environments, if clean environments can be defined.
    - Information bottlenecks that suppress source/material artifacts while preserving near-puzzle recall.
    - Calibration-first models that improve class-1 diagnostics without inventing new labels.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add ECGT-Net to imported research memory after implementation, including whether main, random-cost, uniform-cost, and count-only ablations passed. | Prevents the next research cycle from proposing another material-to-target entropic OT variant with only superficial changes. | Imported Research Memory |
| Add an anti-duplicate rule for “current-board piece/target optimal-transport plans with chess-distance costs” if this experiment fails. | Makes the next model search move to a genuinely different operator rather than changing target roles or Sinkhorn iterations. | Research Continuity / anti-duplicate paragraph |
| Require every future candidate-set model to include count-only and semantics-destroying nuisance-preserving ablations. | This packet shows how easy it is for generated atoms to leak material/count shortcuts without explicit ablations. | Depth requirements and Ablation Plan requirements |
| Clarify or document the canonical `simple_18` channel order in the prompt if Codex confirms it. | Future ideas should not guess channel semantics, especially when deterministic rule-derived features are involved. | Current available encodings / Data Contract |
| Add “matched fine-label-0 false-positive rate class-1 recall” as a preferred near-puzzle diagnostic. | It directly measures whether a model finds near-puzzles without simply calling more non-puzzles positive. | Benchmark requirements |
| Preserve the fail-closed rule for LC0 deterministic geometry adapters. | Avoids silent misuse of history or unknown LC0 channel semantics as deterministic current-board facts. | Leakage and encoding-adapter rules |

Do not weaken leakage rules, label rules, falsification requirements, or anti-duplicate requirements.

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
