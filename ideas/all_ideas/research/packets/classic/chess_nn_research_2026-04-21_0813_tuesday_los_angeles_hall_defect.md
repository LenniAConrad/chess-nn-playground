# Codex Handoff Packet: Hall-Defect Obligation Matroid Network

## 1. File Metadata

- Filename: chess_nn_research_2026-04-21_0813_tuesday_los_angeles_hall_defect.md
- Generated at: 2026-04-21 08:13 America/Los_Angeles
- Weekday: Tuesday
- Timezone: los_angeles
- Idea slug: hall_defect
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Hall-Defect Obligation Matroid Network
- One-sentence thesis: Puzzle-like chess positions often contain a static overload certificate: a small set of defenders is responsible for too many attacked assets or king-zone obligations, and this can be measured by exact Hall-deficiency profiles of a rule-derived defender-obligation set system.
- Idea fingerprint: current-board pseudo-legal attack/control geometry -> side-relative defender-obligation bipartite set system -> exact cardinal and weighted Hall-defect zeta profiles over small defender subsets -> small neural fusion head for binary puzzle-likeness, with no engine metadata, no legal move tree, no move-delta bag, no Sinkhorn transport, and no sheaf/Laplacian/tension operator.
- Why this is not a common CNN/ResNet/Transformer variant: the central computation is a non-convolutional transversal-matroid/Hall-deficiency operator over defender neighborhoods; the learned network only calibrates deterministic cut-profile tokens and a deliberately small board context.
- Current-data minimal experiment: train `hall_defect_obligation_net` on `simple_18` using `data/splits/crtk_sample_3class/{split_train,split_val,split_test}.parquet` for the same 3-epoch coarse-binary benchmark used by the current baselines.
- Smallest central falsification ablation: replace every obligation's defender-neighborhood bitmask by a degree-matched random defender subset while preserving side-to-move, material, obligation counts, defender counts, defender degrees in distribution, candidate truncation counts, obligation weights, and the small board context adapter.
- Expected information gain if it fails: a clean failure would rule out static overload/Hall-defect structure as a useful inductive bias beyond existing CNNs and would prevent future cycles from repeating matching, matroid-rank, cut-defect, or overload-bottleneck variants.

## 3. Problem Restatement And Data Contract

The task is chess puzzle-likeness classification from a single board-position tensor. The model receives a tensor `x` with shape `[batch, C, 8, 8]` and returns logits with shape `[batch, 2]`. Coarse output `0` means non-puzzle and coarse output `1` means puzzle-like. The available fine labels are `0` known non-puzzle, `1` verified near-puzzle, and `2` verified puzzle. The training objective remains coarse binary unless Codex explicitly adds diagnostics; the required report must still include the rectangular diagnostic confusion matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Benchmark split to use:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

The full roughly 45M-row Parquet dataset must not be used by the default trainer until streaming support exists.

Allowed encodings already known to the project:

- `simple_18`: 12 piece planes + side-to-move + castling + en-passant.
- `lc0_static_112`.
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists.

Leakage checklist:

- Safe neural inputs: the board tensor, deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack/control geometry derived only from the current board.
- Safe deterministic rule features for this idea: direct pseudo-legal attacks and defenses, attacked assets, king-zone squares, defender neighborhoods, Hall-defect/cut profiles computed only from the current board.
- Leakage-prone unless explicitly justified and ablated: full legal-move generation, legal move counts, checkmate/stalemate oracles, forced-line search, move-tree consequences, and any rule oracle that asks what happens after a move.
- Always forbidden as neural-network inputs: Stockfish evaluations, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, and unresolved candidate-pool status.
- Label discipline: fine labels `1` and `2` are used only as provided. Do not fabricate, relabel, or infer verified near-puzzle/puzzle labels from unresolved pools.
- Encoding boundary: for `lc0_static_112` and `lc0_bt4_112`, deterministic Hall features may only use current-board channels whose semantics are explicitly mapped in config. History channels may be consumed by a learned neural adapter only; they must not be decoded into rule features unless channel semantics are known. Adapters must fail closed when piece-plane semantics are unknown.

## 4. Research Map

External ideas and sources used:

| Source | What is borrowed | What is not copied |
|---|---|---|
| Philip Hall's marriage theorem and the defect form of Hall's condition, summarized at https://en.wikipedia.org/wiki/Hall%27s_marriage_theorem | The equality between bipartite matching deficiency and a maximum neighborhood-defect certificate, `max_T |{o: N(o) subset T}| - |T|`. | No marriage/matching theorem is used as a label source, move generator, search oracle, or engine substitute. |
| R. A. Brualdi, “Transversal matroids and Hall's theorem,” Pacific Journal of Mathematics, 1972, https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-41/issue-3/Transversal-matroids-and-Halls-theorem/pjm/1102968140.pdf | The view that bipartite matchability defines a transversal matroid on obligations. | No matroid partition algorithm or theorem is imported as a complete model. |
| J. Hopcroft and R. Karp, “An n^{5/2} algorithm for maximum matchings in bipartite graphs,” SIAM Journal on Computing, 1973, PDF mirrored at https://web.eecs.umich.edu/~pettie/matching/Hopcroft-Karp-bipartite-matching.pdf | Complexity intuition for exact matching/rank alternatives and why small defender sets are feasible. | The proposed implementation uses a zeta-transform over truncated defender neighborhoods rather than requiring Hopcroft-Karp in the forward pass. |
| P. Battaglia et al., “Relational inductive biases, deep learning, and graph networks,” arXiv:1806.01261, https://arxiv.org/abs/1806.01261 | The broad principle that hard structured computations can be useful inductive biases. | This is not a graph neural network, not message passing, not a square-GNN, and not a learned attack graph. |
| Chess.com glossary entry on overloading, https://www.chess.com/terms/overloading-chess | The chess intuition that an overloaded piece has too many defensive duties. | No tactic tags, puzzle tags, examples, engine lines, or external chess annotations are used as model inputs. |

Candidate search trace:

| Serious candidate mechanism considered | Why it lost to the selected Hall-defect idea |
|---|---|
| Cubical persistent homology of attack-pressure terrains | Interesting, but static pressure persistence would be harder to implement deterministically in PyTorch, and its falsifier would be less crisp than exact Hall-defect edge rewiring. |
| Tropical/max-plus morphological threat terrain network | Promising for king-zone pressure, but too close to an image-processing CNN variant unless coupled to a sharper chess-specific object; Hall defects give a direct overload certificate. |
| Learned differentiable DNF over hand-coded tactical predicates | Too close to sparse witness, Möbius/ANOVA constellations, and hand-coded motif libraries; likely to become a bag of named tactics rather than a new operator. |
| Unsupervised source-artifact adversarial bottleneck using board-style clusters | Potentially useful, but overlaps with imported rule-partition invariance and risks optimizing away artifacts without a concrete chess mechanism. |
| Label-safe selective/abstention classifier for near-puzzles | Valuable reporting layer, but not a new board operator and too close in spirit to uncertainty/evidential packets. |
| Differentiable shallow search without engine scores | Hard to keep distinct from one-ply move-delta families and risks accidental move-count or legal-tree leakage. |
| Low-rank tensor factorization over piece-square obligations | Too close to ANOVA/Möbius piece-constellation models. |
| Directed diffusion over attack fields | Too close to graph Laplacian/operator-algebra families unless the diffusion object changes radically. |

The internal search pass covered at least these 12 families before selection: cubical persistence, tropical morphology, DNF tactical predicates, adversarial environment bottlenecks, abstention/calibration, differentiable search surrogates, tensor factorization, directed diffusion, matroid/Hall defects, deterministic motif grammars, causal occlusion objectives, and low-rank rule-kernel machines. Hall-defect overload survived because it has a concrete theorem, a compact operator, and a hard semantics-destroying ablation.

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Hall's theorem | For each defensive role/stratum, compute `max_T count({o: N(o) subset T}) - |T|` over defender subsets `T`. | Obligation-neighborhood bitmasks `[B, roles, strata, O_max]` -> defect tokens `[B, roles*strata, F]`. | Degree-matched random replacement of each `N(o)` before zeta profiling. | Imported packets do not use exact Hall-deficiency or transversal-matroid rank/cut profiles. |
| Transversal matroid | Obligations are independent if they can be assigned injectively to distinct defenders. | Bipartite adjacency `[B, roles, strata, O_max, D_max]`. | Preserve counts and degree histograms but destroy which obligations share defenders. | No sheaf restriction maps, no Hodge/Laplacian, no attack-message-passing graph. |
| Chess overload | A small defender set covers too many attacked assets or king-zone squares. | Rule-derived obligations with weights and defender neighborhoods. | Weight-shuffle and edge-rewire controls. | It formalizes overload as a cut-profile, not a named tactic classifier or sparse witness mask. |
| Structured inductive bias | Frozen rule operator plus small learned calibration/fusion. | `x [B,C,8,8]` -> logits `[B,2]`. | Remove Hall tokens while keeping board adapter and nuisance/count tokens. | Not an ordinary CNN/ResNet/ViT scaling exercise. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Bigger simple CNN | `src/chess_nn_playground/models/cnn.py` small/medium/deep variants | Ordinary capacity scaling is already represented and would not test a new chess mechanism. |
| Bigger residual CNN | `src/chess_nn_playground/models/residual_cnn.py` variants | A standard residual stack may improve metrics but gives weak information about tactical structure. |
| LC0-style CNN or residual CNN | Existing LC0 BT4-style CNN/residual CNN variants | Copying LC0-like planes or blocks is already covered and does not introduce a new falsifiable operator. |
| Vanilla ViT over 64 squares | Common square-token Transformer | It is a generic architecture substitution and is explicitly disallowed as the core idea. |
| Plain GNN on board squares or attack edges | Static attack-defense graph models | Message passing on squares/attacks is too close to existing graph/sheaf families and lacks the Hall-defect theorem. |
| Hyperparameter tuning | Any existing config | Changing depth, width, learning rate, batch size, or optimizer is not a research idea. |
| Ensembling current models | Leaderboard/ensemble practice | Ensembling could improve scores but would obscure the causal contribution of a new inductive bias. |
| Another sheaf/Hodge/tension/curvature attack model | Imported tactical sheaf/Hodge packets | Already heavily explored; adding edge types or labels would be a near-duplicate. |
| One-ply move-delta bag, attention, spectrum, or landscape | Imported counterfactual move-delta packets | It risks legal-move leakage and directly duplicates the imported one-ply family. |
| Sinkhorn/OT piece-target bottleneck | Imported optimal-transport packets | The present idea uses Hall subset defects, not couplings, costs, entropic transport, or pressure maps. |
| Ordinal fine-label ladder | Imported ordinal evidence ladder | Near-puzzle ambiguity is not the central mechanism here; the output remains binary logits. |
| Static pseudo-likelihood or masked-board codec | Imported pseudo-likelihood and masked-codec packets | The Hall operator is not a generative code-length, neighborhood prediction, or class-conditioned board-density ratio. |
| Hand-coded tactic tag classifier | External chess motif taxonomies | Using named tactics would be brittle and could smuggle annotation logic; Hall defects are a formal board operator. |
| Legal-move count features | Any move-generation baseline | Legal move counts and move-tree consequences are leakage-prone and not needed for the overload test. |

## 6. Mathematical Thesis

Input space definition:

Let `X_C` be the set of encoded board tensors `x in R^{C x 8 x 8}` accepted by the shared trainer. For the first experiment, `C=18` and `x` is decoded into a current board state

```text
b = (P_white, P_black, side_to_move, castling, en_passant)
```

where `P_color,piece_type,square in {0,1}`. Decoding is deterministic and fails closed if the configured channel map is missing or inconsistent.

Label/target definition:

Let the provided fine label be `y_f in {0,1,2}`. The coarse target is

```text
y = 0 if y_f = 0
y = 1 if y_f in {1,2}
```

No new fine labels are created. Fine labels are used for diagnostics, especially the `3x2` matrix and near-puzzle class-`1` recall.

Data distribution assumptions:

The observed samples are drawn from an unknown mixture distribution over chess positions and labels. The project split is treated as the benchmark distribution. The model may exploit board-visible tactical structure but must not use source metadata, engine scores, verification traces, or any information not present in the current board tensor.

Allowed symmetry or equivariance assumptions:

Chess is not invariant under arbitrary board rotations/reflections because pawns, castling, en-passant, and side-to-move break those symmetries. This idea uses only side-relative role canonicalization:

```text
role 0 = obligations of the side not to move under pressure from the side to move
role 1 = obligations of the side to move under pressure from the side not to move
```

It does not impose file mirror, color-flip orbit pooling, legal automorphism quotienting, or tempo interventions.

Core hypothesis:

A nontrivial subset of puzzle-like positions is characterized by overload: one side has many valuable obligations whose defenders are concentrated in a small neighborhood of defensive pieces. This concentration can be measured by Hall-deficiency/cut profiles of a defender-obligation relation derived from current-board pseudo-legal control. These profiles should improve class-`1` and class-`2` detection beyond a small board adapter and beyond count/degree-only shortcuts.

Formal object introduced:

For a board `b`, a role `r`, and an obligation stratum `g`, construct a finite set of obligations `O_{r,g}(b)` and candidate defenders `D_{r,g}(b)`.

Examples of obligations for a defending color `c`:

- An own non-king piece on square `s` attacked by the opponent.
- A king-ring square within Chebyshev radius 1 or 2 of the color-`c` king that is attacked or contested by the opponent.
- Optional high-value subsets of the attacked-piece obligations, stratified by piece value threshold.

A defender `d in D_{r,g}` is a color-`c` piece that pseudo-legally controls the obligation square. The defended piece itself is excluded as its own defender. Sliding attacks stop at the first blocker. No legal move generation is performed.

This gives a bipartite relation

```text
R_{r,g}(b) subset O_{r,g}(b) x D_{r,g}(b).
```

For each obligation `o`, let

```text
N(o) = { d in D_{r,g}(b) : (o,d) in R_{r,g}(b) }.
```

After deterministic truncation to at most `D_max = 10` candidate defenders per stratum, encode each `N(o)` as a bitmask in `{0,1}^{D_max}`. If more than `D_max` defenders are available, keep defenders by descending `(positive obligation degree, piece value, centrality, stable square index)` and expose `num_defenders_discarded` as a nuisance token. This truncation is a practical approximation, not a theorem.

Define the cardinal Hall defect:

```text
H_{r,g}(b) = max_{T subset D_{r,g}} ( |{ o in O_{r,g} : N(o) subset T }| - |T| ).
```

Define weighted Hall defects for `lambda in Lambda = {1, 2, 3}`:

```text
H^w_{r,g,lambda}(b)
  = max_{T subset D_{r,g}}
      ( sum_{o: N(o) subset T} w(o) - lambda * |T| ),
```

where `w(o)` is a deterministic current-board weight: piece value for attacked-piece obligations, and fixed king-zone weights for king-ring obligations.

Proposition:

For an unweighted bipartite defender-obligation graph with left side `O` and right side `D`, the quantity

```text
max_{T subset D} ( |{ o in O : N(o) subset T }| - |T| )
```

equals the Hall deficiency of `O`, equivalently `|O| - rank_M(O)`, where `M` is the transversal matroid whose independent sets are obligation sets that can be injectively assigned to distinct defenders.

Proof sketch:

For any subset `S subset O`, Hall's obstruction is `|S| - |N(S)|`. Let `T = N(S)`. Then every `o in S` satisfies `N(o) subset T`, so

```text
|S| - |N(S)| <= |{ o : N(o) subset T }| - |T|.
```

Conversely, for any `T subset D`, take `S_T = { o : N(o) subset T }`. Then `N(S_T) subset T`, so

```text
|S_T| - |N(S_T)| >= |S_T| - |T|.
```

Maximizing over `S` and `T` gives equality up to the usual zero-clipping convention. By the defect form of Hall's theorem, this also equals the number of obligations that cannot be matched under a maximum injective assignment. The transversal-matroid rank relation follows from the definition of a transversal matroid.

Optimization objective:

The model learns parameters `theta` for

```text
f_theta(x) = MLP_theta( phi_Hall(b(x)), phi_board(x) )
```

where `phi_Hall` is the frozen Hall-defect tokenization and `phi_board` is a small learned board-context adapter. The primary objective is weighted cross entropy:

```text
min_theta E_{(x,y)} [ alpha_y * CE(f_theta(x), y) ]
```

with deterministic rule-feature generation. Optional auxiliary regularization penalizes overreliance on count-only nuisance tokens by dropout on nuisance subfeatures, not by projection.

What is actually proven:

- For the unweighted, untruncated defender-obligation graph, the cardinal Hall-defect operator exactly measures the maximum Hall obstruction and the unmatched-obligation count under the transversal-matroid assignment model.
- The zeta-transform implementation over defender-neighborhood bitmasks computes this exact maximum for the truncated defender set.

What remains hypothesized:

- That puzzle-like labels are enriched for these overload certificates.
- That weighted Hall defects capture value-sensitive tactical overload better than cardinal defects alone.
- That the deterministic obligation definitions are close enough to tactical reality despite ignoring legal-move consequences and pinned-piece legality.
- That the small board adapter does not dominate or hide the Hall signal.

Counterexamples where the idea should fail:

- Quiet strategic puzzles, zugzwang-like studies, underpromotion motifs, fortress breaks, or long forcing lines whose key structure is not visible as current-board defender overload.
- Positions with obvious overload but no winning tactic because a legal resource, perpetual check, or move-order detail refutes it.
- Positions where pseudo-legal defenders are actually pinned or illegal to move; this operator may overestimate defensive capacity.
- Tactical shots based on discovered attacks or clearance where the overloaded defender only appears after a move; these are intentionally outside the no-move-delta boundary.
- Non-puzzle blunders with many hanging pieces, where Hall defects may be high but puzzle-likeness is absent.

Self-critique:

The strongest objection is that static overload is a human motif, not a complete characterization of puzzle-likeness. The operator may collapse to proxies for material imbalance, piece density, king exposure, or the number of attacked pieces. It also uses static pseudo-legal defense and can misread pins and king safety. The minimal experiment is still worth running because the central ablation is unusually sharp: if degree-matched edge rewiring or count-only tokens match the main model, then the Hall set-system semantics failed. If the main model improves near-puzzle recall at matched false-positive rate and the edge-rewire control loses that gain, the result supports a specific, non-generic overload mechanism.

## 7. Architecture Specification

Module names:

- `SafeBoardDecoder`
- `PseudoLegalAttackGenerator`
- `ObligationSetBuilder`
- `HallZetaDefectLayer`
- `HallDefectTokenEncoder`
- `BoardContextAdapter`
- `HallDefectObligationNet`
- builder function: `build_hall_defect_obligation_net`

Forward-pass steps and tensor shapes:

1. Input:
   - `x`: `[B, C, 8, 8]`, float tensor from existing dataset loader.

2. Decode current board:
   - `pieces`: `[B, 2, 6, 64]`, binary or thresholded piece occupancy.
   - `side_to_move`: `[B]` or `[B,1]`.
   - `aux_state`: castling/en-passant if available.
   - For `simple_18`, use configured `piece_plane_order` and known scalar-plane indices.
   - For `lc0_static_112`/`lc0_bt4_112`, only decode if `piece_plane_map` is explicit. Otherwise raise a clear `ValueError` when Hall features are requested.

3. Generate pseudo-legal control:
   - `piece_slots`: `[B, 2, P_max=16, slot_features]`, with slot features such as piece type, square, value, active mask.
   - `controls`: `[B, 2, P_max, 64]`, boolean pseudo-legal control of squares by each piece.
   - `attack_count`: `[B, 2, 64]`.
   - Sliding pieces use current occupancy blockers and stop at first occupied square. Pawns use attack direction by color. Castling legality and legal king exposure are not evaluated.

4. Build obligations and defender neighborhoods:
   - Roles: `R = 2`.
     - role `0`: opponent's obligations under pressure from side-to-move.
     - role `1`: side-to-move obligations under pressure from opponent.
   - Strata: default `G = 6`.
     - `attacked_all_nonking_assets`
     - `attacked_value_ge_3`
     - `attacked_value_ge_5`
     - `attacked_queen_or_rook`
     - `king_ring_radius_1_contested`
     - `king_ring_radius_2_contested`
   - Obligation slots:
     - `obligation_masks`: `[B, R, G, O_max=64]`.
     - `obligation_weights`: `[B, R, G, O_max]`.
     - `neighborhood_bitmasks`: `[B, R, G, O_max]`, integer masks in `[0, 2^D_max)`.
     - `obligation_feature_counts`: `[B, R, G, F_count]`.
   - Defender slots:
     - `defender_masks`: `[B, R, G, D_max=10]`.
     - `num_defenders_total`: `[B, R, G]`.
     - `num_defenders_discarded`: `[B, R, G]`.

5. Hall zeta profiling:
   - For each `[B,R,G]` graph, build histograms over defender-neighborhood masks:
     - cardinal histogram `h_count`: `[B, R, G, 2^D_max]`.
     - weighted histograms `h_weight`: `[B, R, G, 2^D_max]`.
   - Apply subset zeta transform so `H[T] = sum_{m subset T} h[m]`.
   - Compute:
     - cardinal defect `max_T H_count[T] - popcount(T)`.
     - weighted defects `max_T H_weight[T] - lambda*popcount(T)` for `lambda in {1,2,3}`.
     - argmax subset size, argmax obligation count, argmax weight mass, and simple nuisance counts.
   - Output `hall_tokens`: `[B, R*G=12, F_hall≈16]`.

6. Encode Hall tokens:
   - Shared token MLP: `[B, 12, F_hall] -> [B, 12, 64]`.
   - Pooling: concatenate mean pool, max pool, and signed role-difference summaries:
     - `hall_embedding`: `[B, 64*3]` or smaller if configured.

7. Board context adapter:
   - `Conv2d(C, 32, kernel_size=1)`, GELU.
   - `Conv2d(32, 32, kernel_size=3, padding=1)`, GELU.
   - global average pool and global max pool.
   - `board_embedding`: `[B, 64]`.
   - This adapter is intentionally small. It should not become a hidden baseline ResNet.

8. Fusion and logits:
   - `fusion_input = concat(hall_embedding, board_embedding, side_to_move_scalar, optional nuisance counts)`.
   - MLP: `fusion_dim -> 128 -> 64 -> num_classes`.
   - Return logits `[B, 2]`.

Parameter-count estimate:

- Board context adapter for `simple_18`: about 10k parameters.
- Hall token encoder: about 6k-20k depending on `F_hall`.
- Fusion MLP: about 30k-80k.
- Total expected range: 50k-120k parameters. Keep below the smallest residual CNN unless Codex documents why not.

FLOP/complexity estimate:

- Pseudo-legal control generation: `O(B * 64 * ray_directions * ray_length)` plus fixed knight/king/pawn masks.
- Hall zeta layer: `O(B * R * G * (D_max * 2^D_max + O_max))`. With `B=512`, `R=2`, `G=6`, `D_max=10`, this is roughly `512*12*10240 ≈ 63M` simple add/max operations before token MLP.
- Learned layers are negligible compared with the zeta layer.

Generated candidate-set memory and chunking:

- Edge/neighborhood memory before bitmask compression: `B * R * G * O_max * D_max` booleans. With `B=512`, `R=2`, `G=6`, `O_max=64`, `D_max=10`, this is about 3.9M booleans.
- Zeta histograms: `B * R * G * 2^D_max` floats. With `B=512`, `R=2`, `G=6`, `D_max=10`, this is about 6.3M floats, roughly 25 MB in fp32.
- Chunking plan: expose `hall_chunk_size`, default `64`. Process `[B,R,G]` graph groups in chunks when `B*R*G*2^D_max` would exceed a configurable memory threshold.

Required config fields:

- `model.name: hall_defect_obligation_net`
- `model.input_channels`
- `model.num_classes: 2`
- `model.encoding: simple_18` or equivalent data encoding field.
- `model.hall.d_max_defenders: 10`
- `model.hall.o_max_obligations: 64`
- `model.hall.strata`
- `model.hall.lambdas: [1.0, 2.0, 3.0]`
- `model.hall.hall_chunk_size: 64`
- `model.hall.edge_ablation_mode: none|degree_rewire|count_only|weight_shuffle`
- `model.decoder.piece_plane_order`
- `model.decoder.fail_closed: true`
- `model.board_context.enabled: true`

Encoding support:

- First experiment should use `simple_18` because the deterministic decoder can be made explicit and tested.
- `lc0_static_112` and `lc0_bt4_112` can be supported later only when current-board piece planes are explicitly mapped. Their history channels may feed `BoardContextAdapter`, but Hall features must ignore history channels unless semantics are known.
- If `input_channels != 18` and no `piece_plane_map` is configured, `SafeBoardDecoder` must fail closed rather than guessing.

Pseudocode, not implementation:

```text
forward(x):
    board = SafeBoardDecoder(x, config.decoder)
    controls = PseudoLegalAttackGenerator(board.pieces)
    obligations = ObligationSetBuilder(board, controls, side_to_move=board.side_to_move)

    if edge_ablation_mode == degree_rewire:
        obligations = degree_matched_rewire(obligations, seed, preserve_counts=True)
    if edge_ablation_mode == weight_shuffle:
        obligations = shuffle_obligation_weights_within_stratum(obligations, seed)
    if edge_ablation_mode == count_only:
        hall_tokens = count_degree_nuisance_tokens(obligations)
    else:
        hall_tokens = HallZetaDefectLayer(obligations)

    z_hall = HallDefectTokenEncoder(hall_tokens)
    z_board = BoardContextAdapter(x)
    logits = FusionMLP(concat(z_hall, z_board, board.side_to_move))
    return logits
```

Compatibility:

`HallDefectObligationNet` must be a `torch.nn.Module` with the standard forward signature and must return logits `[batch, num_classes]`. The shared trainer, reports, confusion matrices, predictions, and leaderboards should require no changes other than model registry/config additions.

## 8. Loss, Training, And Regularization

Primary loss:

- Weighted cross entropy over coarse binary labels.
- Use the same class-weighting behavior as the existing benchmark; default `class_weighting: balanced`.

Auxiliary loss:

- No required auxiliary loss for the minimal experiment.
- Optional diagnostic-only regularizer: `hall_dropout_p` randomly drops Hall token strata during training. This tests robustness but should be off in the first main run unless Codex wants a separate ablation.
- Do not add ordinal, credal, Dirichlet, abstention, or source-adversarial heads to the main experiment.

Class weighting:

- Use the shared trainer's balanced class weighting exactly as current baselines do.

Batch size expectations:

- Start with `batch_size: 512` if Hall zeta memory is acceptable.
- If CPU/GPU memory is high, reduce `hall_chunk_size` before reducing the global batch size.
- If zeta profiling is CPU-bound, Codex may run the Hall feature layer under `torch.no_grad()` and cache per-batch tensors, but the default config should keep `cache_features: false` to match the requested block.

Learning-rate and optimizer defaults:

- Optimizer: AdamW.
- Learning rate: `0.001`.
- Weight decay: `0.0001`.
- Epochs: `3`.
- Early stopping patience: `2`.
- Mixed precision: `false` for determinism and because rule bitmask ops are discrete.

Regularizers:

- Weight decay as above.
- Dropout `0.10` in the fusion MLP only.
- No data augmentation, no source metadata, no engine labels.
- Do not regularize by closed-form nuisance projection; that would duplicate the imported nuisance-orthogonal family.

Determinism requirements:

- Seed all PyTorch, NumPy, and Python random generators with `42`.
- Edge-rewire ablation must be deterministic given `(global_seed, sample_index or stable board hash, role, stratum)`.
- Zeta-transform and truncation tie-breaks must be stable across devices.
- If deterministic algorithms are too slow, Codex must document the exception in the result report.

What must stay unchanged for fair comparison:

- Same train/val/test split.
- Same coarse-binary label mapping.
- Same number of epochs unless all baselines are rerun.
- Same class weighting and early stopping.
- Same metrics and `3x2` diagnostic matrix.
- No use of the full 45M-row dataset in the default run.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Central degree-matched edge rewire | Replaces each obligation neighborhood `N(o)` with a random defender subset of the same size within the same role/stratum, preserving obligation counts, weights, defender count, and degree-size distribution. | The identity of shared defenders, not just the number of defenders, carries overload signal. | If performance matches main, Hall set-system semantics failed; the model is using counts/material/pressure only. |
| Count-only nuisance profile | Removes zeta max/cut semantics and feeds only `n_obligations`, `n_defenders`, edge count, degree histogram, obligation weight sums, material counts, side-to-move, and truncation counts. | Exact Hall defects add information beyond obvious shortcuts. | If count-only matches main, abandon Hall-defect as central mechanism. |
| Weight-shuffle Hall profile | Keeps defender neighborhoods fixed but shuffles obligation weights within each role/stratum. | Value-sensitive overload is useful beyond cardinal overload. | If unchanged, weighted defects are unnecessary; keep only cardinal or abandon weighted part. |
| Cardinal-only Hall profile | Removes `lambda`-weighted defects and keeps only unweighted Hall deficiency. | Piece-value/king-zone weighting improves classification. | If cardinal-only matches main, simplify the model. |
| Complete-neighborhood control | Sets every obligation to be defended by all active defenders while preserving counts and weights. | Sparse defender neighborhoods are the meaningful structure. | If this matches main, the model is ignoring neighborhood sparsity. |
| Defender-truncation stress test | Run `D_max=8`, `D_max=10`, and a small-batch `D_max=12` diagnostic. | Improvements are not artifacts of dropping defenders. | If results swing wildly with `D_max`, the operator is unstable and should not scale. |
| No board context adapter | Uses Hall tokens only. | The Hall operator is independently predictive. | If Hall-only collapses but main improves, the signal may require contextual calibration; still acceptable if central ablations lose. |
| Board context only, parameter-matched | Removes Hall tokens and expands MLP width to match parameter count. | Gains are not from extra parameters. | If this matches main, there is no evidence for Hall features. |
| Side-role swap/random side twin | Randomly swaps role `0` and role `1` or uses identity side-to-move in the role canonicalizer. | Side-relative attacking/defending direction matters. | If unchanged, the model may be using side-agnostic material shortcuts. |
| Obligation-type dropout diagnostic | Trains/evaluates with one stratum removed at a time. | Which obligation families matter: attacked pieces, major pieces, or king ring. | If only one trivial stratum matters, refine the operator or add stronger controls. |
| Degree-preserving bipartite edge swaps | Performs switch-chain swaps preserving defender degrees and obligation degrees approximately, not just obligation neighborhood sizes. | Hall semantics survive stricter degree preservation. | If strict degree preservation erases the gain, the original gain was likely degree-driven. |
| Nuisance-preserving material/side stratified report | Report metrics stratified by material balance, phase proxy, side-to-move, and king-ring obligation count. | Hall gains are not isolated to one nuisance bucket. | If gains appear only in one bucket, scaling should wait for a causal follow-up. |

Structured-operator hard control:

The degree-matched and degree-preserving edge-rewire ablations are mandatory because the idea builds a structured bipartite set system. They destroy defender-neighborhood semantics while preserving obvious shortcuts such as obligation count, defender count, edge density, material, side-to-move, obligation weight histogram, candidate truncation count, and degree distribution as much as practical.

Rule-generated candidate-set hard controls:

Although this idea does not generate legal moves or move deltas, it does generate obligation and defender candidate sets. Therefore Codex must include count-only and nuisance-preserving ablations that preserve candidate count, defender degree histograms, material, side-to-move, defender piece identity distribution, source-square marginal, target-square/obligation marginal where feasible, and attacked/captured-piece-value histograms while destroying Hall subset semantics.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- Existing `simple_18` simple CNN, same split/config family.
- Existing `simple_18` residual CNN, same split/config family.
- Best available current `simple_18` leaderboard baseline after Codex verifies names.
- Optional: LC0-style baselines only as contextual references, because the first Hall experiment should use `simple_18`.

Metrics to inspect:

- Test accuracy.
- Balanced accuracy.
- AUROC.
- AUPRC for coarse puzzle-like class.
- F1 and precision/recall at the default threshold.
- Calibration diagnostics if already supported; do not add a new uncertainty head for the main experiment.
- Required `3x2` fine-label diagnostic confusion matrix.

Required diagnostic for near-puzzles:

- Compute fine-label-`1` recall at a threshold chosen to match the best simple baseline's fine-label-`0` false-positive rate on validation.
- Also report precision among predicted positives at that matched fine-label-`0` FPR, broken out for fine labels `1` and `2` if possible.

Required artifacts:

- Model config.
- Training log.
- Validation and test metrics JSON.
- Predictions Parquet/CSV using the shared prediction artifact format.
- `3x2` confusion matrix for the main model and all central ablations.
- Ablation report table.
- Edge-rewire and count-only ablation configs.
- Runtime/memory notes for Hall zeta profiling.
- Failure-mode notes with examples if the project already has example extraction utilities.

Success threshold:

- Main model improves over the best comparable `simple_18` baseline by at least one of:
  - `+1.0` percentage point AUROC,
  - `+1.5` percentage points AUPRC,
  - `+3.0` percentage points fine-label-`1` recall at matched fine-label-`0` FPR.
- At least half of the gain must disappear in the central degree-matched edge-rewire ablation or the count-only ablation.
- The `3x2` matrix must not show that gains come only from converting all class-`0` samples to positives.

Failure threshold:

- Main model is within noise of the board-context-only and count-only controls on AUROC/AUPRC and class-`1` matched-FPR recall.
- Degree-preserving rewiring retains the same performance as the main model.
- Runtime is more than 3x the simple CNN for no measurable diagnostic gain.
- Performance improves only through fine-label-`0` false positives with no class-`1` or class-`2` recall benefit.

What result would make me abandon the idea:

If both degree-matched edge rewiring and count-only nuisance profiles match the main model within `0.3` percentage points AUROC and within `1.0` percentage point class-`1` matched-FPR recall, abandon Hall-defect overload as a central family and add it to the anti-duplicate list.

What result would justify scaling:

If the main model beats the best comparable `simple_18` baseline under the success threshold and central ablations lose most of the gain, then scale only after:
- profiling Hall zeta runtime,
- testing `D_max` stability,
- confirming the same qualitative gain across at least two random seeds,
- and adding streaming-safe feature caching before any larger dataset run.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_hall_defect/idea.yaml` | Create | Machine-readable idea metadata matching the `idea_yaml` block below. |
| `ideas/20260421_hall_defect/math_thesis.md` | Create | Mathematical thesis, Hall-defect proposition, proof sketch, hypotheses, counterexamples, and self-critique. |
| `ideas/20260421_hall_defect/architecture.md` | Create | Module-level architecture, tensor shapes, complexity, memory/chunking plan, and pseudocode. |
| `ideas/20260421_hall_defect/implementation_notes.md` | Create | Decoder assumptions, pseudo-legal attack generation rules, zeta-transform details, deterministic truncation, and fail-closed behavior for unknown encodings. |
| `ideas/20260421_hall_defect/trainer_notes.md` | Create | Loss, training defaults, determinism, benchmark invariants, and reporting requirements. |
| `ideas/20260421_hall_defect/ablations.md` | Create | Mandatory central ablations, nuisance-preserving controls, and interpretation rules. |
| `ideas/20260421_hall_defect/train.py` | Create | Thin wrapper invoking the shared trainer with `configs/hall_defect_obligation_net_simple18.yaml`; do not fork the trainer unless required. |
| `ideas/20260421_hall_defect/config.yaml` | Create | Idea-local copy of the default config for reproducibility. |
| `ideas/20260421_hall_defect/report_template.md` | Create | Template requiring main metrics, `3x2` matrix, near-puzzle matched-FPR diagnostic, ablation table, runtime, and failure notes. |
| `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Preserve hard constraints and add anti-duplicate guidance for Hall-deficiency/transversal-matroid overload operators if this packet is consumed. |
| `src/chess_nn_playground/models/hall_defect_obligation_net.py` | Create | PyTorch modules: decoder, pseudo-legal attack generator, obligation builder, Hall zeta layer, token encoder, context adapter, fusion net. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `hall_defect_obligation_net` and builder `build_hall_defect_obligation_net`. |
| `configs/hall_defect_obligation_net_simple18.yaml` | Create | Shared-trainer config using `simple_18`, coarse binary mode, 3 epochs, balanced weighting, and Hall default fields. |
| `tests/test_hall_defect_obligation_net.py` | Create | Focused tests for decoder fail-closed behavior, Hall-defect theorem on tiny hand-built bipartite graphs, deterministic rewiring, output shape `[B,2]`, and no gradients required through rule tensors. |
| `tests/test_hall_defect_no_leakage.py` | Create | Tests or assertions that model config has no Stockfish/PV/node/source/proposed-label inputs and does not call legal move generation or checkmate/stalemate oracles. |

For `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md`, Codex should add a compact memory entry after consuming this packet:

```text
Hall-Defect Obligation Matroid Network:
current-board pseudo-legal defender-obligation relation over attacked assets and king-zone obligations
+ exact Hall-deficiency / transversal-matroid rank or weighted defender-subset zeta profile
+ binary puzzle-likeness target
+ no engine metadata, no move-delta bag, no Sinkhorn, no sheaf/Laplacian.
If it fails, do not repeat static overload, max matching, Hall cut, transversal matroid, or defender-obligation zeta-profile variants without a genuinely different formal observable and a stronger falsifier.
```

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0813_tuesday_los_angeles_hall_defect.md
  generated_at: "2026-04-21 08:13 America/Los_Angeles"
  weekday: Tuesday
  timezone: los_angeles
  idea_slug: hall_defect
  format: markdown
```

```yaml
idea_yaml:
  idea_id: "20260421_hall_defect"
  name: "Hall-Defect Obligation Matroid Network"
  slug: hall_defect
  status: draft
  created_at: "2026-04-21 08:13 America/Los_Angeles"
  author: ChatGPT Pro
  short_thesis: "Puzzle-like positions are enriched for static overload certificates where valuable attacked obligations have defender neighborhoods concentrated in too-small defender subsets."
  novelty_claim: "Uses exact Hall-deficiency/transversal-matroid cut profiles over rule-derived defender-obligation set systems, not CNN depth, sheaves, move deltas, Sinkhorn transport, ordinal heads, pseudo-likelihoods, or orbit/tempo interventions."
  expected_advantage: "Better near-puzzle and puzzle recall at matched non-puzzle false-positive rate when overload structure is visible on the current board."
  central_falsification_ablation: "Degree-matched random rewiring of obligation defender-neighborhood bitmasks while preserving counts, weights, material, side-to-move, and board context."
  target_task: coarse_binary
  input_representation: simple_18
  output_heads: binary_logits
  compute_notes: "Hall zeta profiling costs O(B*roles*strata*d_max*2^d_max); default d_max=10 and hall_chunk_size=64."
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/hall_defect_obligation_net_simple18.yaml
  model_path: src/chess_nn_playground/models/hall_defect_obligation_net.py
  latest_result_path: null
  notes: "First experiment should fail closed to simple_18 unless LC0 current-piece channel maps are explicitly configured."
```

```yaml
config_yaml:
  run:
    name: hall_defect_obligation_net_simple18
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
    name: hall_defect_obligation_net
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
  model_name: hall_defect_obligation_net
  file_path: src/chess_nn_playground/models/hall_defect_obligation_net.py
  builder_function: build_hall_defect_obligation_net
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - SafeBoardDecoder
    - PseudoLegalAttackGenerator
    - ObligationSetBuilder
    - HallZetaDefectLayer
    - HallDefectTokenEncoder
    - BoardContextAdapter
    - HallDefectObligationNet
  required_config_fields:
    - model.name
    - model.input_channels
    - model.num_classes
    - data.encoding
    - model.hall.d_max_defenders
    - model.hall.o_max_obligations
    - model.hall.lambdas
    - model.hall.hall_chunk_size
    - model.decoder.piece_plane_order
    - model.decoder.fail_closed
  expected_parameter_count: "50k-120k for simple_18 default"
  expected_memory_notes: "Default zeta histograms use about 25 MB fp32 at batch_size=512, roles=2, strata=6, d_max=10; use hall_chunk_size=64 if memory or CPU time is high."
```

```yaml
research_continuity:
  idea_fingerprint: "current-board pseudo-legal defender-obligation set system + exact Hall-deficiency/transversal-matroid weighted zeta profiles + small neural fusion head"
  already_researched_family_overlap: "Adjacent to static attack-defense geometry but not a sheaf, Hodge, graph Laplacian, attack-message-passing, move-delta, Sinkhorn/OT, nuisance projection, ordinal, credal, ray-language, ANOVA, pseudo-likelihood, orbit, tempo, kinematic-commutator, or masked-codec model."
  closest_duplicate_risk: "Could be mistaken for another attack-defense graph model; the distinguishing operator is exact Hall subset deficiency over defender neighborhoods with degree-preserving edge-rewire falsification."
  do_not_repeat_if_this_fails:
    - "Static defender-obligation bipartite graphs with Hall-deficiency or maximum-matching bottlenecks."
    - "Weighted defender-subset zeta profiles over attacked assets or king-zone obligations."
    - "Overloaded-defender matroid-rank features as the central puzzle-likeness mechanism."
    - "Degree-preserving edge-rewire controls as merely cosmetic ablations without a new formal observable."
  suggested_next_search_directions:
    - "Label-safe selective prediction that changes evaluation policy rather than board operator, if uncertainty remains the issue."
    - "Causal invariance from genuinely external source shifts if safe provenance-free environments can be constructed."
    - "Tropical morphology or cubical persistence only if Hall-defect fails for semantic reasons and a stronger falsifier is designed."
    - "Generative motif compression that is not pseudo-likelihood, masked-codec, or class-conditioned board-density ratio."
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add Hall-deficiency/transversal-matroid overload networks to imported research memory after implementation. | Prevents future cycles from rediscovering static defender-obligation max-matching/cut-profile ideas. | `Imported Research Memory` |
| Add anti-duplicate wording: “Do not propose another defender-obligation Hall-defect, maximum-matching, transversal-matroid rank, or weighted subset-zeta overload model unless the observable is genuinely different.” | Makes the duplicate boundary as explicit as the sheaf, OT, move-delta, and pseudo-likelihood boundaries. | Anti-duplicate paragraphs after imported fingerprints |
| Require any future candidate-set model, even if not a move-set, to include count-only and degree-preserving semantics-destroying ablations. | This packet exposed that rule-generated candidate sets can leak easy shortcuts through counts and degrees. | `Ablation Plan` requirements |
| Add an encoding-adapter fail-closed rule for deterministic feature extraction from `lc0_static_112` and `lc0_bt4_112`. | Avoids silent channel-semantics guesses when rule features are decoded from non-simple encodings. | `Problem Restatement And Data Contract` |
| Ask Codex to record runtime/memory for any discrete combinatorial operator. | Future research passes need to know whether exact operators are bottlenecked by CPU loops or GPU memory. | `Benchmark And Falsification Criteria` and prompt-maintenance checklist |

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
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Möbius-constellation, or pseudo-likelihood packets: yes
- Not an exact near-duplicate of imported orbit-symmetry, tempo-intervention, credal-evidence, rule-partition-invariance, kinematic-commutator, or masked-codec packets: yes
