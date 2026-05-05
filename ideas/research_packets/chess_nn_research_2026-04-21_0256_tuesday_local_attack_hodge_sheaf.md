# Codex Handoff Packet: Attack-Hodge Sheaf Tension Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0256_tuesday_local_attack_hodge_sheaf.md`
- Generated at: 2026-04-21 02:56 PDT, UTC-07:00
- Weekday: Tuesday
- Timezone: local, America/Los_Angeles
- Idea slug: `attack_hodge_sheaf`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Attack-Hodge Sheaf Tension Network, abbreviated `AHS-TensionNet`.
- One-sentence thesis: chess puzzle-likeness is often signaled by locally inconsistent attack, defense, pin, fork, and overload constraints, so a model should operate on a position-dependent directed attack cell complex and learn sheaf-Hodge tension energies instead of only convolving over neighboring squares.
- Idea fingerprint: `AHS-v1 = current-board pseudo-attack 0/1/2-cell complex + side-aware directed cellular sheaf restrictions + Hodge edge/face tension pooling + binary CE target only + no engine search/eval metadata`.
- Why this is not a common CNN/ResNet/Transformer variant: the computation graph is not the 8x8 pixel grid and not all-square attention; it is rebuilt from legal piece attack geometry for each position, carries 0-cochains on squares, 1-cochains on attacks/x-rays, and 2-cochains on tactical incidence cells, and updates them with learned sheaf coboundaries and Hodge Laplacians.
- Current-data minimal experiment: implement `AttackHodgeSheafNet` for `simple_18`, train on `data/splits/crtk_sample_3class/split_train.parquet`, select by validation AUROC or macro-F1 on `split_val.parquet`, then report binary metrics plus fine-label confusion on `split_test.parquet`; run the central ablation that replaces the dynamic attack complex with a static 8-neighbor grid while preserving parameter count.
- Expected information gain if it fails: failure would be strong evidence that static tactical incidence alone is not the missing inductive bias for this benchmark, or that puzzle-likeness depends more on engine-depth move uniqueness than on one-ply attack-defense topology; the ablations will distinguish “attack complex unhelpful” from “sheaf/Hodge operator unhelpful.”

## 3. Problem Restatement And Data Contract

Task: classify a chess board position as binary non-puzzle or puzzle-like.

Source classes:

- Fine label `0`: known non-puzzle.
- Fine label `1`: verified near-puzzle.
- Fine label `2`: verified puzzle.

Training target for the current benchmark:

- Binary output `0`: fine label `0`.
- Binary output `1`: fine label `1` or fine label `2`.

Model interface:

- Input tensor shape: `(batch, C, 8, 8)`.
- Output logits shape: `(batch, num_classes)`, with `num_classes = 2`.
- The model must be a PyTorch `nn.Module`.

Allowed encodings:

- `simple_18`
- `lc0_static_112`
- `lc0_bt4_112`

Required split:

- Train: `data/splits/crtk_sample_3class/split_train.parquet`
- Validation: `data/splits/crtk_sample_3class/split_val.parquet`
- Test: `data/splits/crtk_sample_3class/split_test.parquet`

Allowed inputs and derived features:

- The raw board-position tensor supplied by one of the supported encodings.
- Current-piece occupancy, side-to-move, castling/en-passant channels, and history channels only when they are already present in the selected encoding.
- Deterministic pseudo-legal attack relations derived from the current board pieces and ordinary chess movement rules. This includes pawn attacks, king/knight attacks, sliding rays up to blockers, and optional x-ray rays through exactly one blocker for pin/skewer cells.
- Deterministic side-relative metadata derived from the board, such as “source piece color is side to move,” “target square is occupied by opponent,” and “target is in king zone.”

Forbidden inputs and leakage checklist:

| Item | Status for this idea |
|---|---|
| Stockfish scores, centipawns, WDL, mate distance | Forbidden; never computed or read. |
| Principal variations, best moves, engine node counts, engine depths | Forbidden; never computed or read. |
| Verification metadata, puzzle-source metadata, proposed labels | Forbidden as model inputs and forbidden for graph construction. |
| Fine label or source class as a feature | Forbidden. Fine labels are used only by the dataset target loader and evaluator. |
| Fabricated fine labels `1` or `2` | Forbidden. Collapse existing fine labels to binary only. |
| Unresolved candidates | Stay unresolved. Do not relabel them, add them to positives, or mine pseudo-negatives. |
| Checkmate/stalemate oracle or legal-move count | Do not use. These are rule-based but too close to puzzle outcome shortcuts. Use pseudo-attacks, not search or outcome tests. |
| Move uniqueness, puzzle solution length, rating, themes | Forbidden unless already part of a separate evaluation report; never input to the model. |

## 4. Research Map

The selected idea borrows mathematical operators, not complete architectures. It deliberately avoids copying an LC0 tower, an AlphaZero policy/value head, or a generic graph classifier.

| Paper or idea | URL | What is borrowed | What is not copied | Verification note |
|---|---|---|---|---|
| Group Equivariant Convolutional Networks, Cohen and Welling | https://arxiv.org/abs/1602.07576 | The principle that known symmetries should constrain weight sharing and sample complexity. | Full D4 image equivariance; chess is not D4-invariant because pawns, castling, en-passant, and side-to-move break most image symmetries. | URL verified by web search; details should be rechecked if cited in a paper. |
| Sheaf Neural Networks, Hansen and Gebhart | https://openreview.net/pdf?id=GgcgIJsT8HD | Cellular sheaves as learned relation maps between local fibers. | Their benchmark setup and fixed graph setting. Here the graph/cell complex is rebuilt from each chess position. | URL verified by web search; implementation details not audited here. |
| Neural Sheaf Diffusion, Bodnar et al. | https://arxiv.org/abs/2202.04579 | Sheaf diffusion as a way to handle heterophilic or relation-dependent graph signals. | Node-classification objective and generic graph assumptions. | URL verified by web search; use as conceptual background. |
| Sheaf Neural Networks with Connection Laplacians, Barbero et al. | https://proceedings.mlr.press/v196/barbero22a/barbero22a.pdf | Connection-like transport maps and Laplacian regularization over learned restrictions. | Manifold tangent-space construction. Chess fibers are tactical roles, not geometry estimated by PCA. | URL verified by web search; not fully audited. |
| Signal Processing on Simplicial Complexes, Schaub et al. | https://arxiv.org/abs/2106.07471 | Hodge Laplacian viewpoint on 0-, 1-, and higher-order signals. | Their continuous signal-processing applications. | URL verified by web search; mathematical definitions are standard. |
| Signal Processing on Cell Complexes, Roddenberry et al. | https://arxiv.org/abs/2110.05614 | Regular cell-complex framing beyond pairwise graphs. | General-purpose cell-complex filtering recipes. The chess 2-cells are attack-defense motifs. | URL verified by web search; details not audited here. |
| E(n)-Equivariant Topological Neural Networks, Battiloro et al. | https://arxiv.org/pdf/2405.15429 | The broader idea that topological signals and equivariance can coexist. | E(n) equivariance; chess board geometry is discrete and rule-directed, not Euclidean particle dynamics. | URL verified by web search; recent citation should be rechecked. |
| Leela Chess Zero network topology | https://lczero.org/dev/backend/nn/ | Baseline context: 112 input planes and residual tower convention. | LC0 residual tower, SE blocks, policy/value heads, MCTS assumptions. | Project documentation URL verified by web search. |
| Graph-based representation for chess reinforcement learning | https://arxiv.org/html/2410.23753v1 | Motivation that chess can benefit from non-grid relational structure. | Plain graph attention policy model. This proposal uses sheaf-Hodge cochains and binary puzzle-likeness classification. | URL verified by web search; not used as a dependency. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN on 8x8 planes | simple CNN | Already represented in the project and too local/isotropic for long sliding attacks, pins, forks, and overloaded defenders. |
| Deeper or wider ordinary CNN | small/medium/deep CNN variants | Routine capacity tuning violates the core research goal and does not introduce a new chess-specific inductive bias. |
| Standard residual CNN | residual CNN | Better optimization, not a new hypothesis about puzzle-likeness. It still treats tactics as patterns in a square lattice. |
| LC0-style CNN or residual tower | LC0 BT4-style CNN and residual CNN variants | Already close to existing baselines; cloning a policy/value engine architecture is not targeted to binary puzzle-likeness. |
| Ordinary ViT or all-square Transformer | vanilla square Transformer, if added | Captures global dependencies but ignores typed chess movement geometry unless it relearns it from data. It is also a common next baseline. |
| Plain GNN on 64 squares | generic square graph GNN | A graph with adjacency between neighboring squares still misses sliding rays, attack direction, blockers, x-rays, and higher-order tactical cells. |
| GAT on pseudo-legal attack edges only | plain attack graph GNN | Closer, but still pairwise. It cannot explicitly represent curl-like inconsistency over fork, pin, and overload cells. The smallest central ablation tests whether the Hodge/sheaf part matters. |
| Hyperparameter tuning, optimizer tuning, LR schedules | all existing baselines | Useful engineering but not an original research idea. It must stay fixed for fair comparison. |
| Ensembling several existing models | ensemble | Likely improves metrics without explaining what structure matters, and the prompt forbids ensembling as the core idea. |
| Stockfish-distilled classifier | none; forbidden leakage | Directly violates the no-engine-features rule even if the engine score is only an auxiliary target. |
| Puzzle theme/rating/move-solution auxiliary training | none; forbidden leakage unless labels are separately authorized | Risks leaking verification metadata and fabricating unavailable class semantics. |
| Handcrafted material/check-count logistic regression | none or classical ML baseline | Too shallow and risks shortcuting by obvious board statistics rather than learning tactical geometry. |

## 6. Mathematical Thesis

Input space: for an encoding with `C` planes, each sample is `x ∈ R^{C×8×8}`. The adapter extracts a current board occupancy map `b(x)` with piece color/type on squares whenever the encoding exposes current piece planes. The neural stem may read the full tensor `x`, but the rule-built complex is based only on current pieces and side-to-move.

Target definition: `y = 0` for fine label `0`; `y = 1` for fine label `1` or `2`. The model is not asked to distinguish fine labels `1` and `2` during training, though the benchmark report must show true fine label `0/1/2 -> predicted binary output 0/1`.

Distribution assumptions:

- Train, validation, and test are sampled from the same CRTK-style three-class split family.
- Fine labels `1` and `2` share more tactical local inconsistency than fine label `0`, but not every puzzle is a tactic and not every tactical-looking position is a puzzle.
- The split may contain spurious correlations in material, side-to-move, or opening phase; the attack-Hodge ablations are meant to expose whether the proposed inductive bias adds information beyond those correlations.

Symmetry and equivariance assumptions:

- Do not impose full rotation/reflection invariance. Chess is not invariant under arbitrary board rotations or reflections because pawn direction, castling side, en-passant, and side-to-move matter.
- The attack complex is direction-aware: north, south, diagonals, knight offsets, pawn attack direction, and source color are separate edge types.
- Optional consistency may be used only for legal chess automorphisms that the encoding adapter can transform exactly, such as 180-degree color-perspective flip or file mirror with castling/en-passant channels transformed correctly. If an encoding channel cannot be transformed safely, skip the consistency loss for that encoding.

Core hypothesis: puzzle-like positions often contain an attack-defense structure that is locally sharp but globally inconsistent in the following sense: there is no low-energy assignment of latent tactical roles to squares and attack relations that simultaneously satisfies all learned transport constraints around king zones, pinned rays, fork fans, and overloaded defenders. A sheaf-Hodge operator should expose this as edge and face tension more directly than a pixel CNN.

Formal object: for each position `x`, build a finite directed 2-dimensional cell complex `K(x)`.

- `V`: 64 square 0-cells.
- `E(x)`: directed 1-cells for pseudo-legal attacks from occupied source squares to attacked target squares. Sliding pieces generate directed ray attacks up to the first blocker. Optional x-ray edges go through exactly one blocker and are tagged separately.
- `F(x)`: bounded deterministic 2-cells over attack edges. Use three face families: `fork_fan` for two attack edges sharing a source; `overload_sink` for two or more attack edges sharing a tactically important target or defender; `ray_pin` for a sliding ray with source, blocker, and behind-target x-ray relation. Candidate faces are truncated by deterministic rule-only priority, never by label or engine information.

Fibers and cochains:

- Each square `v` has a node fiber `F_v = R^d` and node cochain `h_v`.
- Each attack edge `e` has an edge fiber `F_e = R^d` and edge cochain `a_e`.
- Each tactical face `f` has a face fiber `F_f = R^{d_f}` or `R^d` if implementation simplicity is preferred.

Learned sheaf restrictions:

- For each incident pair `(v, e)`, learn a side-aware typed restriction map `ρ_{v→e}(x)` from edge type, source/target role, piece type, direction, occupancy relation, and side-to-move role.
- For each incident pair `(e, f)`, learn `ρ_{e→f}(x)` from face type and edge boundary sign.
- To keep parameters stable, parameterize maps as diagonal plus low-rank transport: `ρ z = diag(s) z + U(V^T z)`, with rank `r <= 4`, or as grouped diagonal maps for the minimal version.

Coboundaries:

For an oriented edge `e = (u → v)`, define

`(D_0 h)_e = ρ_{v→e} h_v - ρ_{u→e} h_u`.

For a face `f`, define

`(D_1 a)_f = Σ_{e∈∂f} σ_{f,e} ρ_{e→f} a_e`,

where `σ_{f,e} ∈ {-1, +1}` is a deterministic orientation sign.

The 1-Hodge operator is

`L_1 a = D_0 D_0^T a + D_1^T D_1 a`.

Energy feature:

`E_sheaf(x) = mean_masked(||D_0 h||_2^2) + λ mean_masked(||D_1 a||_2^2)`.

This scalar is not the classifier by itself; it is concatenated with pooled node, edge, and face representations. The model may also use per-type energy summaries, such as king-zone edge tension and ray-pin face tension.

Proposition: for fixed learned restriction maps on a finite attack complex, the quadratic form `||D_0 h||^2 + λ||D_1 a||^2` is nonnegative. It is zero if and only if the node cochain is a global section across all attack-edge restrictions and the edge cochain is curl-free across all tactical faces with respect to the learned face restrictions.

Proof sketch: both terms are sums of squared Euclidean norms, hence nonnegative. The sum is zero exactly when every squared term is zero. The first condition is `D_0 h = 0`, which is precisely agreement of transported endpoint square features on each attack edge. The second is `D_1 a = 0`, which is precisely zero signed boundary inconsistency over each included face. This proves the stated algebraic characterization. It does not prove that the learned restrictions are semantically correct, nor that puzzle labels are separable.

What is proven:

- The energy is a well-defined positive semidefinite inconsistency measure on the constructed finite complex.
- Adding 2-cells lets the model measure inconsistency that is invisible to a node-only graph Laplacian, because `D_1` acts on edge cochains around higher-order incidence cells.
- Direction-aware edge types avoid false D4 invariance assumptions.

What is hypothesized:

- Verified near-puzzles and puzzles have a different distribution of learned attack-Hodge tensions than known non-puzzles.
- The difference is learnable from current data without engine signals.
- The 2-cell terms are not just decorative; they should improve fine-label `2` recall or calibration over the pairwise attack-graph ablation.

Counterexamples and expected failure modes:

- Quiet endgame studies, zugzwang puzzles, stalemate traps, or long maneuver puzzles can be puzzle-like with low immediate attack-Hodge tension.
- Wild attacking non-puzzles can have high attack-Hodge tension and become false positives.
- Mate-in-one positions may be solved by a simple check pattern, so the complex may not beat a CNN there.
- Positions with unusual underpromotion/promoted-piece arrangements can stress the adapter if piece planes are assumed too narrowly.
- If the dataset’s non-puzzles are sampled from ordinary quiet positions while puzzles are sampled from high-material tactical phases, material/phase shortcuts may dominate. The ablation table must check this.

## 7. Architecture Specification

Proposed model class: `AttackHodgeSheafNet(nn.Module)`.

Proposed support modules:

- `EncodingAdapter`: extracts current piece planes, side-to-move role, and optional transform-safe metadata from `simple_18`, `lc0_static_112`, or `lc0_bt4_112`.
- `AttackComplexBuilder`: constructs padded attack edges and tactical faces per sample.
- `SheafRestrictionMLP`: emits diagonal plus low-rank transport parameters for node-edge and edge-face restrictions.
- `HodgeTensionBlock`: performs masked node, edge, and face updates using `D_0`, `D_1`, `D_0^T`, and `D_1^T`.
- `MaskedCochainPool`: pools node, edge, face, and energy summaries.
- `AttackHodgeClassifierHead`: maps pooled features to logits.

Default tensor sizes:

- `B`: batch size.
- `C`: input channels.
- `S = 64`: squares.
- `Emax = 1024`: padded directed attack/x-ray edges.
- `Fmax = 1024`: padded tactical 2-cells for the minimal experiment; can be `1536` if memory allows.
- `d_model = 64` for full; `48` for smoke/minimal.
- `d_edge = d_model`.
- `d_face = d_model`.
- `n_layers = 3` for full; `2` for minimal.
- `transport_rank = 4`.

Forward pass pseudocode, intentionally not full implementation:

```text
forward(x):
    # x: (B, C, 8, 8)
    square_raw = flatten_board(x)                       # (B, 64, C)
    piece_state = adapter.extract_current_board(x)      # rule-only occupancy metadata

    H0 = square_stem(square_raw, square_coord_features) # (B, 64, d)

    complex = attack_builder(piece_state)
    # complex.edge_index:      (B, Emax, 2) source,target
    # complex.edge_type:       (B, Emax)
    # complex.edge_mask:       (B, Emax)
    # complex.face_edges:      (B, Fmax, Kface) edge ids, padded; Kface usually 2 or 3
    # complex.face_signs:      (B, Fmax, Kface) {-1,0,+1}
    # complex.face_type:       (B, Fmax)
    # complex.face_mask:       (B, Fmax)

    H1 = edge_init(H0[src], H0[tgt], edge_type_embed, edge_role_features) # (B,Emax,d)
    H2 = face_init(H1[face_edges], face_type_embed, face_signs)           # (B,Fmax,d)

    all_energy_summaries = []
    for layer in 1..n_layers:
        rho_node_edge = restriction_node_edge(H0, H1, complex) # diag+low-rank params
        rho_edge_face = restriction_edge_face(H1, H2, complex)

        D0H0 = apply_D0(H0, rho_node_edge, complex)            # (B,Emax,d)
        D1H1 = apply_D1(H1, rho_edge_face, complex)            # (B,Fmax,d)

        node_msg = apply_D0_transpose(edge_mlp(D0H0), rho_node_edge, complex)
        edge_msg = edge_mlp2(H1) - D0H0 + apply_D1_transpose(face_mlp(D1H1), rho_edge_face, complex)
        face_msg = face_mlp2(D1H1)

        H0 = masked_gru_or_residual(H0, node_msg, square_mask)
        H1 = masked_gru_or_residual(H1, edge_msg, edge_mask)
        H2 = masked_gru_or_residual(H2, face_msg, face_mask)

        all_energy_summaries.append(masked_energy_stats(D0H0, D1H1, complex))

    pooled = concat(
        masked_mean_max_attention_pool(H0),
        masked_mean_max_attention_pool(H1 by edge type groups),
        masked_mean_max_attention_pool(H2 by face type groups),
        concat(all_energy_summaries)
    )
    logits = classifier_head(pooled)                         # (B, 2)
    return logits
```

Attack edge construction details:

- Pawn edges are attack-only diagonals, not quiet pushes.
- Knight and king edges use ordinary fixed offsets.
- Bishop, rook, and queen ray edges include all empty attacked squares until first occupied square and include the blocker square as an attacked target. They do not continue past the blocker for ordinary edges.
- Optional x-ray edges continue past exactly one blocker for slider pieces and are tagged as `xray_ray`. X-ray edges must not use checkmate, best move, or engine data.
- Castling moves are not edges.
- En-passant is ignored for attack edges in the minimal experiment unless the adapter exposes the en-passant square cleanly and tests cover it.

Face construction details:

- `fork_fan`: choose pairs of outgoing attack edges from the same source where targets are distinct and at least one target is opponent-occupied or in a king zone.
- `overload_sink`: choose pairs or triples of incoming attack/defense edges sharing a target, especially king-zone squares and occupied defenders.
- `ray_pin`: choose a slider source, one blocker on a ray, and an x-ray behind-target. The face boundary uses the ordinary source-blocker edge and the x-ray relation.
- Faces are padded and masked. If candidates exceed `Fmax`, keep a deterministic priority order based only on face type, king-zone involvement, occupancy relation, and stable square ordering.

Parameter estimate with `C=112`, `d=64`, `n_layers=3`, and rank `4`:

- Square stem and embeddings: roughly `10k-40k` parameters depending on adapter features.
- Edge/face initialization MLPs: roughly `80k-160k`.
- Restriction MLPs and low-rank transport emitters: roughly `120k-250k`.
- Three Hodge blocks with small MLP/GRU-style updates: roughly `250k-500k`.
- Pooling and classifier head: roughly `80k-180k`.
- Expected total: `0.55M-1.1M` parameters. Codex should report the exact number with the project’s parameter counter.

Complexity estimate:

- Construction is rule-based and roughly `O(B * pieces * ray_length)` for edges plus capped face enumeration.
- Neural computation is approximately `O(B * L * ((64 + Emax + Fmax) d^2 + (Emax + 3Fmax) d r))` for `L` Hodge blocks and low-rank transport rank `r`.
- With `d=64`, `L=3`, `Emax=1024`, `Fmax=1024`, expected compute is higher than a tiny CNN but below a deep LC0 tower. If runtime exceeds `3x` the best existing single-model baseline for less than `1pp` validation gain, trigger the abandon condition.

Config fields Codex should expose:

```text
model.name: attack_hodge_sheaf
model.num_classes: 2
model.encoding: simple_18 | lc0_static_112 | lc0_bt4_112
model.d_model: 48 or 64
model.n_layers: 2 or 3
model.transport_rank: 2 or 4
model.max_edges: 1024
model.max_faces: 1024
model.face_types: [fork_fan, overload_sink, ray_pin]
model.use_xray_edges: true
model.use_face_hodge: true
model.use_energy_pool: true
model.perspective_normalize: false by default unless adapter-safe
model.dropout: 0.05 to 0.15
```

Encoding support plan:

- `simple_18`: required first target. Use current piece planes from the existing encoder definition. The square stem reads all 18 channels. The complex builder reads only current piece occupancy and side-to-move/castling metadata that is already encoded.
- `lc0_static_112`: second target after `simple_18`. The square stem may read all 112 planes. The complex builder must use only the current-position piece slice and side metadata, not history-derived guesses.
- `lc0_bt4_112`: third target. Same contract as `lc0_static_112`; if BT4 plane ordering differs, add adapter tests before training.
- Fail closed: if Codex cannot reliably identify current piece planes for an encoding, skip that encoding and report why. Do not infer labels or read dataset metadata to construct the complex.

Logits interface:

- `forward(x)` returns only logits `(B, 2)` during normal training/evaluation.
- Optional debug mode may return masks, energy summaries, and edge counts, but training scripts must not depend on debug outputs.

## 8. Loss, Training, And Regularization

Primary loss:

- Binary cross-entropy via `nn.CrossEntropyLoss` over logits `(B,2)` and collapsed targets `0/1`.
- Class weights may be computed from the training split’s binary label frequencies only. Do not compute weights from validation/test.

Optional auxiliary regularizers, all label-free:

- `energy_l2`: small penalty on extreme sheaf energy summaries to prevent exploding transports, e.g. `1e-4 * mean(E_sheaf)`.
- `transport_norm`: weight decay or spectral/Frobenius clipping on low-rank transport parameters.
- `mask_dropout`: randomly drop a small percentage of non-king-zone attack edges during training, not during evaluation, to reduce overreliance on single noisy edges. Keep this off in the smallest falsification run if it complicates interpretation.
- `safe_automorphism_consistency`: optional KL consistency between original and exactly transformed legal automorphism views, only when the adapter transforms every relevant channel safely.

Do not use:

- Engine-derived auxiliary targets.
- Best-move, policy, WDL, centipawn, PV, mate-distance, or node-count distillation.
- Fine label `1` vs `2` as a separate training objective unless the project explicitly already trains such heads for all baselines. The recommended first pass does not.

Training defaults for the minimal experiment:

- Encoding: `simple_18`.
- Batch size: start `128`; use `64` if dynamic complex memory is high.
- Optimizer: AdamW.
- Learning rate: `3e-4`.
- Weight decay: `1e-4`.
- Epochs, early stopping, scheduler, mixed precision, and seed policy: match the strongest existing single-model baseline training protocol as closely as possible.
- Dropout: `0.10` in MLPs and classifier head.
- Gradient clipping: global norm `1.0`.
- Determinism: fix Python, NumPy, PyTorch, dataloader, and CUDA seeds using the project’s existing deterministic utility. Log exact seeds.

What must stay fixed for fair comparison:

- Same train/validation/test split files.
- Same binary label collapse.
- Same metrics and thresholding protocol.
- Same input encoding for a given comparison.
- Same training budget as existing baselines, unless runtime reporting explicitly normalizes for compute.
- No ensemble averaging, no test-time augmentation unless every compared baseline receives the same augmentation.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Static-grid central falsifier | Replace the dynamic attack complex with fixed 8-neighbor plus knight-like square adjacency, preserve parameter count and pooling shape. | The chess-rule attack complex is the source of gain, not just another message-passing module. | If this matches full model, the core claim that attack incidence matters is falsified. |
| Pairwise attack graph only | Keep pseudo-legal attack edges but remove all 2-cells and `D_1` terms. | Higher-order fork/overload/pin faces add useful information beyond pairwise attacks. | If equal to full model, Hodge face tension is unnecessary; future work should avoid 2-cell machinery. |
| Identity sheaf restrictions | Replace learned `ρ` maps with signed identity maps by edge orientation. | Side-aware learned relation maps matter. | If equal to full model, the sheaf part is overengineered; a typed attack graph may be enough. |
| No x-ray/ray-pin cells | Remove x-ray edges and `ray_pin` faces, keep ordinary attacks and other faces. | Pins and skewers are a meaningful part of puzzle-likeness. | If no drop, x-ray complexity is not justified for this benchmark. |
| No energy pooling | Keep Hodge updates but remove explicit energy summaries from classifier input. | The tension scalar/statistics are independently useful, not just hidden message passing. | If no drop, energy is interpretability-only and can be omitted. |
| Node-only readout | Use only square-node pooled states, discard edge and face pooled states. | Edge/face states encode tactical information not recoverable from updated nodes alone. | If equal, cochain-level readout is unnecessary. |
| Edge-type shuffle control | Randomly permute edge-type embeddings once per run while preserving graph structure. | Semantic direction/piece attack types matter. | If equal, model is using graph density/phase shortcuts rather than typed tactics. |
| Face-priority stress test | Halve `max_faces` and then double it if memory allows. | Results should be stable under reasonable deterministic face caps. | Strong sensitivity suggests truncation artifacts rather than robust geometry. |
| Encoding adapter test | Compare `simple_18` full model to same model on `lc0_static_112` current-board-only builder but all channels in square stem. | The idea should transfer across encodings without relying on hidden adapter accidents. | If only one encoding works, inspect channel mapping and history shortcuts. |

The smallest ablation that can falsify the central claim is `Static-grid central falsifier`. It changes the relational object while leaving the message-passing family and approximate parameter budget intact.

## 10. Benchmark And Falsification Criteria

Baselines to compare:

- Existing simple CNN on the same encoding.
- Existing residual CNN on the same encoding.
- Existing small/medium/deep CNN variants, where available.
- Existing LC0 BT4-style CNN/residual model for `lc0_bt4_112`, where available.
- New ablations from section 9, especially static-grid and pairwise-only.

Metrics:

- Validation and test AUROC.
- Validation and test AUPRC.
- Accuracy, balanced accuracy, macro-F1, positive-class F1.
- Confusion table by true fine label `0/1/2 -> predicted binary 0/1`.
- False-negative rate for fine label `2` and false-positive rate for fine label `0`.
- Calibration: ECE or reliability bins if the project already reports them.
- Runtime and parameter count.

Required artifacts from Codex implementation:

- Model config YAML.
- Parameter count and average edge/face counts per split.
- Per-seed metrics table for at least 3 seeds if project budget allows; otherwise one seed plus deterministic smoke tests.
- Test predictions saved in the same format used by existing benchmark reports.
- Ablation metrics CSV/JSON.
- Short result note under `ideas/2026_04_21_attack_hodge_sheaf/`.

Success threshold:

- Strong success: improves test macro-F1 by at least `1.5 percentage points` over the best existing single non-ensemble baseline on the same encoding, and reduces fine-label `2` false negatives by at least `5% relative` without increasing fine-label `0` false positives by more than `2 percentage points`.
- Moderate success: improves validation AUROC or AUPRC by at least `1.0 percentage point` and beats both central ablations, even if test macro-F1 gain is smaller.
- Scientific success even with modest metrics: pairwise-only beats static-grid, and full face-Hodge beats pairwise-only by a consistent margin across seeds. That would validate the structural hypothesis for future scaling.

Failure threshold:

- Full model fails to beat the simple CNN by at least `0.5 percentage points` validation AUROC and fails to beat the static-grid ablation within ordinary seed variance.
- Full model improves validation but not test, and ablations show the gain comes from graph density or material/phase shortcuts rather than attack/Hodge structure.

Abandon condition:

- Abandon this line if the static-grid ablation matches or beats full `AHS-TensionNet`, pairwise-only matches full within noise, and runtime is more than `3x` the best existing single baseline. In the next research cycle, do not repeat attack-sheaf/Hodge variants unless a new dataset diagnostic shows that attack topology is mislabeled or underused.

Scaling condition:

- Scale to `lc0_static_112` or `lc0_bt4_112`, `d_model=64`, and `Fmax=1536` only if the minimal `simple_18` experiment beats static-grid and pairwise-only ablations on validation.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/2026_04_21_attack_hodge_sheaf/handoff.md` | Create | Copy this Markdown handoff packet for traceability. |
| `ideas/2026_04_21_attack_hodge_sheaf/README.md` | Create | One-page summary, exact commands run, and final benchmark table. |
| `ideas/2026_04_21_attack_hodge_sheaf/ablation_results.json` | Create after experiments | Metrics for full model and ablations, including seed list and split paths. |
| `src/chess_nn_playground/models/attack_hodge_sheaf.py` | Create | `EncodingAdapter`, `AttackComplexBuilder`, `SheafRestrictionMLP`, `HodgeTensionBlock`, `MaskedCochainPool`, and `AttackHodgeSheafNet`. |
| `src/chess_nn_playground/models/registry.py` | Edit | Register `attack_hodge_sheaf` with the same model factory interface as existing models. |
| `configs/attack_hodge_sheaf_simple18.yaml` | Create | Minimal experiment config using `simple_18`, `d_model=48 or 64`, `n_layers=2`, `max_edges=1024`, `max_faces=1024`, binary CE. |
| `configs/attack_hodge_sheaf_lc0_static112.yaml` | Create only after simple_18 passes adapter tests | Same architecture with `lc0_static_112`; square stem reads all channels, builder uses current board slice only. |
| `tests/test_attack_complex_builder.py` | Create | Unit tests for pawn attacks, knight attacks, slider blockers, x-ray through one blocker, no castling edge, masks, deterministic truncation. |
| `tests/test_attack_hodge_sheaf_shapes.py` | Create | Forward-pass shape tests for `(B,C,8,8) -> (B,2)`, variable edge counts, all-empty masks if malformed sample is rejected safely. |
| `tests/test_encoding_adapter_current_board.py` | Create if adapter mappings are not already covered | Verify piece plane extraction for each supported encoding using simple handcrafted boards. |
| `ideas/chatgpt_pro_deep_math_research_prompt.md` | Edit after consuming this output | Preserve hard constraints; add reusable lessons, anti-duplicate rules, output clarity, and failure-mode guidance discovered here. |

Implementation cautions:

- Use vectorized tensor operations where practical, but correctness and leakage safety matter more than clever batching in the first pass.
- Do not call a chess engine. A lightweight rule library is acceptable only for attack generation if it does not compute engine evaluations, best moves, search, or mate status. Direct tensor/bitboard implementation is preferable for tests and speed.
- Debug outputs should be behind a flag and excluded from normal model outputs.
- The model must train through the existing benchmark scripts with the same API as other models.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: "chess_nn_research_2026-04-21_0256_tuesday_local_attack_hodge_sheaf.md"
  generated_at: "2026-04-21 02:56 PDT"
  weekday: "Tuesday"
  timezone: "local/america_los_angeles"
  idea_slug: "attack_hodge_sheaf"
  title: "Codex Handoff Packet: Attack-Hodge Sheaf Tension Network"
  created: true
  intended_next_consumer: "Codex"
```

```yaml
idea_yaml:
  idea_id: "2026_04_21_attack_hodge_sheaf"
  idea_name: "Attack-Hodge Sheaf Tension Network"
  short_name: "AHS-TensionNet"
  task: "binary chess puzzle-likeness classification"
  core_object:
    - "position-dependent directed pseudo-attack graph"
    - "bounded tactical 2-cells for fork/overload/pin motifs"
    - "learned side-aware cellular sheaf restrictions"
    - "Hodge tension pooling"
  forbidden_inputs:
    - "Stockfish or other engine scores"
    - "principal variations or best moves"
    - "engine node counts or depths"
    - "verification metadata"
    - "source/proposed labels as features"
    - "fabricated fine labels"
  primary_falsifier: "static-grid central ablation with matched parameter count"
  minimal_encoding: "simple_18"
  success_metric: "test macro-F1 plus fine-label-2 false-negative reduction"
```

```yaml
config_yaml:
  model:
    name: "attack_hodge_sheaf"
    num_classes: 2
    encoding: "simple_18"
    d_model: 64
    n_layers: 3
    transport_rank: 4
    max_edges: 1024
    max_faces: 1024
    face_types:
      - "fork_fan"
      - "overload_sink"
      - "ray_pin"
    use_xray_edges: true
    use_face_hodge: true
    use_energy_pool: true
    perspective_normalize: false
    dropout: 0.10
  data:
    train_split: "data/splits/crtk_sample_3class/split_train.parquet"
    val_split: "data/splits/crtk_sample_3class/split_val.parquet"
    test_split: "data/splits/crtk_sample_3class/split_test.parquet"
    binary_label_rule: "0 -> 0, 1|2 -> 1"
  training:
    loss: "cross_entropy"
    class_weighting: "train_binary_frequency_optional"
    optimizer: "adamw"
    learning_rate: 0.0003
    weight_decay: 0.0001
    batch_size: 128
    gradient_clip_norm: 1.0
    determinism: true
  regularization:
    energy_l2: 0.0001
    transport_norm: "weight_decay_or_clip"
    mask_dropout: 0.0
  evaluation:
    metrics:
      - "auroc"
      - "auprc"
      - "accuracy"
      - "balanced_accuracy"
      - "macro_f1"
      - "positive_f1"
      - "fine_label_confusion"
    compare_to_existing_single_model_baselines: true
```

```yaml
model_spec:
  class_name: "AttackHodgeSheafNet"
  module_path: "src/chess_nn_playground/models/attack_hodge_sheaf.py"
  input_shape: ["batch", "C", 8, 8]
  output_shape: ["batch", 2]
  components:
    EncodingAdapter:
      purpose: "extract current board occupancy for rule-only complex construction"
      fail_closed_if_unknown_encoding: true
    AttackComplexBuilder:
      node_count: 64
      max_edges: 1024
      max_faces: 1024
      edge_sources: "pseudo_legal_attacks_and_optional_one_blocker_xrays"
      face_sources: ["fork_fan", "overload_sink", "ray_pin"]
    SheafRestrictionMLP:
      map_parameterization: "diagonal_plus_low_rank"
      rank: 4
      side_aware: true
    HodgeTensionBlock:
      operators: ["D0", "D1", "D0_transpose", "D1_transpose"]
      masked: true
      layers: 3
    MaskedCochainPool:
      pools: ["node", "edge", "face", "energy"]
    AttackHodgeClassifierHead:
      logits: 2
  debug_outputs_optional:
    - "edge_count"
    - "face_count"
    - "energy_by_layer"
    - "mask_statistics"
```

```yaml
research_continuity:
  idea_fingerprint: "AHS-v1: side-aware pseudo-attack 0/1/2-cell complex plus learned sheaf-Hodge tension pooling; no engine data; binary CE only."
  closest_duplicate_risk: "Plain pseudo-legal attack graph GNN or LC0-style relational attention. The required distinction is learned sheaf restrictions plus 2-cell Hodge tension, tested by pairwise-only and static-grid ablations."
  do_not_repeat_if_this_fails:
    - "Do not propose another attack-sheaf, attack-Hodge, or pseudo-legal attack graph variant unless the failure analysis shows adapter bugs or face construction bugs."
    - "Do not replace it with a larger CNN, residual tower, ordinary ViT, or ensemble as the next core idea."
    - "Do not add engine-distillation targets to rescue the idea."
  suggested_next_search_directions:
    - "Causal invariance across legal board automorphisms and data-source shifts."
    - "Information bottleneck model that suppresses material/phase shortcuts while retaining tactical motifs."
    - "Differentiable bounded-width rule search surrogate using only legal move generation, not engine evaluation, if static attack topology fails."
    - "Optimal-transport matching between attacker and defender piece sets as a non-graph alternative."
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add “Do not repeat attack-Hodge/sheaf tactical cell complexes if the static-grid and pairwise-only ablations falsify them.” | Prevents the next research cycle from producing a cosmetic variant of this idea after a negative result. | Add to hard constraints or an “anti-duplicate memory” subsection. |
| Require every future idea to name its smallest central falsifier. | Forces the research packet to separate a real hypothesis from an implementation bundle. | Add to Required Markdown File Content under Ablation Plan or Benchmark And Falsification Criteria. |
| Clarify that rule-derived pseudo-legal attacks are allowed, but engine evaluations, mate solvers, legal-move-count shortcuts, and checkmate/stalemate oracles should be treated as leakage-prone unless explicitly justified. | The boundary between chess rules and engine leakage is subtle; this prevents accidental shortcuts. | Edit Project Context or Hard Constraints. |
| Require adapter fail-closed behavior for encoding-specific feature extraction. | Prevents models from silently using wrong LC0 channel slices or dataset metadata. | Add to Problem Restatement And Data Contract. |
| Preserve the instruction that unresolved candidates remain unresolved. | Keeps future cycles from pseudo-labeling unresolved data. | Keep unchanged in Hard Constraints. |
| Add a line requiring confusion by fine label `0/1/2` for all proposed models and ablations. | The benchmark already reports this; making it mandatory helps diagnose whether gains come from true puzzles or near-puzzles. | Edit Benchmark And Falsification Criteria. |

## 14. Final Sanity Check

- Downloadable Markdown file created: yes.
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0256_tuesday_local_attack_hodge_sheaf.md`.
- No forbidden engine features used as inputs: yes.
- Does not fabricate labels: yes.
- Not a routine CNN/ResNet/Transformer variant: yes.
- Minimal current-data experiment exists: yes, `simple_18` full model plus static-grid central falsifier.
- Falsification criterion is concrete: yes.
- Codex can implement without asking for missing architecture details: yes.
- Prompt maintenance notes included for Codex: yes.
