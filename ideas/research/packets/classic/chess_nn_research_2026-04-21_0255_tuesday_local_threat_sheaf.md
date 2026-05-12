# Codex Handoff Packet: Tactical Threat-Sheaf Network

## 1. File Metadata

- Filename: chess_nn_research_2026-04-21_0255_tuesday_local_threat_sheaf.md
- Generated at: 2026-04-21 02:55:30 America/Los_Angeles
- Weekday: Tuesday
- Timezone: local
- Idea slug: threat_sheaf
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Tactical Threat-Sheaf Network
- One-sentence thesis: Chess puzzle-likeness is often signaled by localized incompatibility among attack, defense, pin, and overload relations; a learned sheaf Laplacian on the position's pseudo-legal attack-defense complex should expose that tactical tension more directly than square-grid convolution.
- Idea fingerprint: deterministic pseudo-legal attack graph from the input planes only; typed directed attack/defense edges; learned nontrivial source/target restriction maps; sheaf-tension diffusion; contested-square energy readout; binary puzzle classifier.
- Why this is not a common CNN/ResNet/Transformer variant: the core computation is not translation convolution, residual image processing, token self-attention, or an LC0 tower; it is a position-dependent cellular-sheaf operator whose edges are chess attack relations and whose main latent signal is disagreement energy under learned tactical transports.
- Current-data minimal experiment: train `ThreatSheafNet` on `simple_18` with the existing `crtk_sample_3class` train/val/test split, run three seeds, compare to the strongest same-encoding non-ensemble baseline, and run the grid-edge and identity-restriction ablations in the same script.
- Expected information gain if it fails: a clean failure, especially if the grid-edge ablation matches it, would falsify the claim that static attack-defense incidence is the missing inductive bias; the next research cycle should move toward counterfactual one-ply move-delta operators or differentiable search surrogates rather than more attack-map topology.

## 3. Problem Restatement And Data Contract

Task: classify a chess board position as binary `0` non-puzzle or binary `1` puzzle-like.

Source fine labels available in reports:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

Training target: use the dataset's existing binary label only. Fine labels may be used for reporting the required `true fine label 0/1/2 -> predicted binary output 0/1` breakdown, but never as neural-network inputs, auxiliary targets, or handcrafted features unless the current benchmark already explicitly defines such a reporting-only field outside the model.

Allowed neural input: only board-position encodings already available to the project:

- `simple_18`
- `lc0_static_112`
- `lc0_bt4_112`

Forbidden neural inputs and forbidden feature sources:

- Stockfish scores
- engine principal variations
- engine node counts
- engine verification metadata
- source labels as model inputs
- proposed labels as model inputs
- unresolved-candidate flags as model inputs
- fabricated class `1` or class `2` labels
- any feature derived by solving the position with an engine or search oracle

Input/output tensor contract:

- model input: `(batch, C, 8, 8)`
- model output: logits `(batch, num_classes)`
- default `num_classes`: `2`
- model class must be a PyTorch `nn.Module`

Benchmark split contract:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

Leakage checklist for this idea:

- The attack-defense complex is computed deterministically from the current board tensor and ordinary chess movement rules only.
- It does not evaluate move quality.
- It does not call Stockfish, LC0 search, Syzygy, or any verifier.
- It does not use PVs, mate distances, centipawn scores, visits, search depths, or puzzle-generation metadata.
- It does not treat unresolved candidates as negative or positive examples.
- It does not pseudo-label near-puzzles or verified puzzles.
- It does not pass fine labels into the model.
- It may use side-to-move, castling, en-passant, and history planes only insofar as those planes are already present in the selected encoding.

## 4. Research Map

This idea borrows mathematical operators and inductive-bias language from the following papers or public technical notes. It does not copy any architecture wholesale.

1. Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing, Bodnar et al., 2022. URL: https://arxiv.org/abs/2202.04579  
   Borrowed: the cellular-sheaf view of graph diffusion, the use of nontrivial restriction maps, and the idea that sheaf geometry can encode incompatible local relations.  
   Not copied: their benchmark datasets, graph construction, node-classification setup, and exact layer implementation.

2. Sheaf Neural Networks, Hansen and Ghrist, 2020. URL: https://arxiv.org/abs/2012.06333  
   Borrowed: the sheaf Laplacian as a graph-convolution generalization.  
   Not copied: their handcrafted toy sheaves or experimental setting.

3. Group Equivariant Convolutional Networks, Cohen and Welling, 2016. URL: https://arxiv.org/abs/1602.07576  
   Borrowed: the discipline of making weight sharing match a real symmetry.  
   Not copied: full group convolution. Chess is not fully dihedral-invariant because pawn direction, castling, en-passant, and side-to-move break many board symmetries.

4. Learning Partial Equivariances from Data, Romero and Lohit, 2022. URL: https://arxiv.org/abs/2110.10211  
   Borrowed: the warning that exact equivariance can be harmful when a task has only partial or context-dependent symmetry.  
   Not copied: their partial group-convolution parameterization.

5. Relational Inductive Biases, Deep Learning, and Graph Networks, Battaglia et al., 2018. URL: https://arxiv.org/abs/1806.01261  
   Borrowed: the object-relation-global decomposition and the argument that structured relational computation can reduce sample complexity.  
   Not copied: generic graph-network blocks as the main model.

6. Cell Complex Neural Networks, Hajij et al. URL: https://openreview.net/pdf?id=6Tq18ySFpGU  
   Borrowed: the idea that higher-order cells can carry signals beyond pairwise adjacency.  
   Not copied: generic cell-complex message passing. This proposal uses a minimal contested-square cell summary rather than a full general cell-complex framework.

7. BScNets: Block Simplicial Complex Neural Networks, Chen et al., 2021. URL: https://arxiv.org/abs/2112.06826  
   Borrowed: the Hodge-Laplacian intuition that edge-level and higher-order tension can be informative.  
   Not copied: their link-prediction model and block Hodge implementation.

8. Leela Chess Zero neural-network topology note. URL: https://lczero.org/dev/backend/nn/  
   Borrowed: confirmation that LC0-style inputs and baselines are residual-tower image models over 112 planes.  
   Not copied: LC0 policy/value heads, search, residual tower depth, training targets, or engine evaluation.

9. Project-specific benchmark paths, encodings, labels, and hard constraints are taken from the user handoff prompt and are not independently verified here. Marked unverified because they are local repository facts rather than public citations.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN on the board tensor | simple CNN | Already present; it treats tactics as local image texture and does not encode long-range sliders, pins, or attack/defense incidence. |
| Wider or deeper plain CNN | small/medium/deep CNN variants | Routine capacity scaling; forbidden as the core idea and weakly diagnostic. |
| Standard residual CNN | residual CNN | Already present; residual square-grid processing remains a visual-board prior, not a chess-relation prior. |
| LC0-style CNN or LC0-style residual tower | LC0 BT4-style CNN/residual CNN variants | Too close to existing baselines and to engine architectures; also optimized for policy/value play, not puzzle-likeness. |
| Ordinary ViT over 64 square tokens | none or square Transformer variant | Generic all-to-all attention lacks the typed chess attack complex and is explicitly disallowed as a vanilla square Transformer. |
| Plain GNN on 64 squares with fixed grid or all-square adjacency | possible graph baseline | Too generic; it either recreates grid convolution or all-to-all attention and does not represent pseudo-legal tactical relations as sheaf restrictions. |
| Handcrafted attack-count features plus MLP | no direct baseline | Could be useful, but it collapses structure into counts and cannot test the sheaf-tension hypothesis. Use only as an ablation, not the core idea. |
| Hyperparameter tuning of existing models | all baselines | Forbidden as the core idea and unlikely to reveal a new inductive bias. |
| Ensembling several baselines | ensemble of CNN/residual/LC0 models | Forbidden as the core idea; improves leaderboard metrics without improving scientific understanding. |
| Pseudo-labeling unresolved positions | none | Violates the unresolved-candidate rule and can contaminate evaluation. |
| Engine-teacher distillation from Stockfish or LC0 search | none | Forbidden leakage: would use scores, PVs, or search-derived targets. |
| Contrastive pretraining on augmented board positions alone | no direct baseline | Interesting but not enough: it postpones the central question and risks becoming a generic representation-learning pass. |

## 6. Mathematical Thesis

### Input space

For an encoding `e`, the model observes

```text
x in R^{C_e x 8 x 8}
```

where `C_e` is determined by `simple_18`, `lc0_static_112`, or `lc0_bt4_112`. The model may derive deterministic board state fields from `x`: piece type, color, occupancy, side-to-move, castling rights, en-passant square, and history planes only when those fields are already encoded in `x`.

### Target definition

The target is binary:

```text
y = 0: non-puzzle
y = 1: puzzle-like
```

Fine labels `0/1/2` are reporting strata, not inputs.

### Distribution assumptions

The fixed train/validation/test split is treated as the empirical distribution. The thesis assumes puzzle-like positions are enriched for forcing tactical structures: pins, loose kings, overloaded defenders, double attacks, discovered attacks, trapped pieces, mating nets, and unstable defended targets. This is a hypothesis about the dataset, not a theorem about chess.

### Symmetry and equivariance assumptions

Do not impose full rotation/reflection invariance. Chess position semantics are not invariant under arbitrary rotations or reflections because:

- pawns have direction;
- side-to-move matters;
- castling distinguishes king-side and queen-side rights;
- en-passant state is directional and temporal;
- many encodings may be oriented from a fixed color or from side-to-move.

The only intended weight sharing is partial and typed:

- share restriction-map parameters across attack edges with the same chess relation type;
- distinguish relative color, side-to-move, piece type, target role, and pawn direction;
- optionally share across left-right mirror counterparts only if the encoding adapter can transform castling and coordinates correctly. This mirror sharing should be off by default in the minimal experiment.

### Core hypothesis

A chess tactic is often a small inconsistent local system: one defender is asked to protect two targets; a pinned piece locally appears to defend but globally cannot move; a king-line slider creates a long-range constraint; a sacrifice works because the defending transport along attack edges cannot be made mutually consistent. A nontrivial sheaf over the attack-defense complex can represent these inconsistencies as high or structured sheaf energy.

### Formal object

For each input `x`, construct a typed directed attack-defense complex `K(x)`.

- Vertices `V = {0, ..., 63}` are board squares.
- Directed 1-cells `E(x)` are pseudo-legal attack/defense rays or jumps from an occupied source square `u` to a target square `v`.
- Each edge has a type `r(e)` containing at least:
  - attacking piece type: pawn, knight, bishop, rook, queen, king;
  - attacker relative color: side-to-move or not-side-to-move;
  - target role: empty, own piece defended, enemy piece attacked, enemy king attacked, own king defended/protected;
  - geometry family: pawn diagonal, knight jump, king step, orthogonal slider, diagonal slider;
  - ray direction bucket for sliders and pawn attacks.
- Optional 2-cell summaries are attached to contested target squares. A target is contested when it receives incoming edges from both colors or multiple pieces with different relation types. These summaries are not separate labels; they are deterministic functions of `E(x)`.

Define a cellular sheaf `F_theta` on `K(x)`:

- vertex stalk: `F_v = R^d` for every square `v`;
- edge stalk: `F_e = R^d` for every attack edge `e`;
- source restriction: `rho_{e,src}: F_src(e) -> F_e`;
- target restriction: `rho_{e,dst}: F_dst(e) -> F_e`.

Restriction maps are learned and tied by edge type:

```text
rho_{e,src} = A_src[r(e)]
rho_{e,dst} = A_dst[r(e)]
```

Use low-rank-plus-diagonal matrices by default:

```text
A_role[r] = diag(a_role[r]) + U_role[r] V_role[r]^T
```

with rank `q = 4` in the minimal experiment.

For node features `z in R^{64 x d}`, define the sheaf coboundary on each edge:

```text
(delta_theta z)_e = A_src[r(e)] z_src(e) - A_dst[r(e)] z_dst(e)
```

and weighted sheaf energy:

```text
E_theta(z; x) = sum_{e in E(x)} g_e ||(delta_theta z)_e||_2^2
```

where `g_e in [0, 1]` is a learned gate depending only on node embeddings and edge type.

### Proposition or objective

Proposition, fixed graph and fixed gates: `L_theta(x) = delta_theta^T G delta_theta` is positive semidefinite, and `E_theta(z; x) = z^T L_theta(x) z >= 0`. A linear diffusion step

```text
z' = z - eta L_theta(x) z
```

is a gradient step on sheaf energy. For sufficiently small `eta`, it cannot increase `E_theta` in the purely linear, fixed-gate case.

Objective used by the classifier:

```text
min_theta CE(classifier(pool(z_T, edge_energy, contest_stats)), y)
```

where `z_T` is produced by a small number of gated sheaf-diffusion layers, and `edge_energy`/`contest_stats` are readout statistics over typed attack tensions.

### Proof sketch

The coboundary `delta_theta` is a linear map from vertex cochains to edge cochains once `K(x)`, restriction maps, and gates are fixed. Since `G` is diagonal with nonnegative entries, `L = delta^T G delta` is positive semidefinite because

```text
z^T L z = (delta z)^T G (delta z) = sum_e g_e ||(delta z)_e||^2 >= 0.
```

The gradient of `E(z) = z^T L z` is `2Lz` when `L` is symmetric and fixed, so the diffusion update is a scaled gradient step. Standard smooth quadratic descent gives non-increase for step size below `2 / lambda_max(L)`.

If a square permutation preserves piece semantics, side-to-move semantics, edge types, and attack incidence, then the corresponding node and edge permutation matrices satisfy

```text
delta(pi.x) = P_E delta(x) P_V^T
L(pi.x) = P_V L(x) P_V^T.
```

Therefore the sheaf diffusion is equivariant to those incidence-preserving relabelings. Because the model includes chess-specific edge types and may include absolute coordinate embeddings, this is deliberately partial equivariance, not full board rotation/reflection invariance.

### What is proven

- The sheaf energy is nonnegative under fixed gates.
- The linearized diffusion is a gradient step on a positive semidefinite quadratic energy.
- The operator is equivariant to attack-complex isomorphisms that preserve all typed chess semantics used by the restrictions.

### What is hypothesized

- Puzzle-like positions are more separable by learned attack-sheaf tension than by square-grid image features alone.
- Near-puzzles, fine label `1`, have intermediate or partially resolved tension patterns that the sheaf readout can separate better than CNN baselines.
- Learned restriction maps will discover tactical role transports resembling attack, defense, king-line, and overload constraints without engine supervision.

### Counterexamples

- Quiet endgame studies, zugzwang, fortress recognition, and opposition puzzles may have low attack-sheaf tension.
- Non-puzzle middlegame positions with many mutual attacks may create high raw tension and false positives.
- Underpromotion or stalemate motifs can depend on legal-move consequences not visible in pseudo-legal attack incidence.
- If the encoding adapter misreads orientation or side-to-move, the constructed complex will be wrong.
- If the dataset's positive label depends mainly on engine evaluation swings rather than human-like tactical structure, this architecture may fail without using forbidden engine signals.

## 7. Architecture Specification

### Model name

`ThreatSheafNet`

### Main modules

- `EncodingPieceAdapter`
  - Input: `x` with shape `(B, C, 8, 8)` and `encoding_name`.
  - Output:
    - `square_raw`: `(B, 64, C)`
    - `piece_onehot`: `(B, 64, 12)` for white/black piece type if recoverable from the encoding
    - `occupancy`: `(B, 64)`
    - `side_to_move`: `(B, 1)` or encoded default from input planes
    - optional castling/en-passant/history fields when already present
  - Must fail loudly if the current-piece plane mapping is unavailable.

- `PseudoLegalAttackBuilder`
  - Deterministically constructs attack/defense edges from decoded current pieces.
  - Output:
    - `edge_src`: `(B, E_max)`
    - `edge_dst`: `(B, E_max)`
    - `edge_type`: `(B, E_max)`
    - `edge_mask`: `(B, E_max)`
    - `target_square`: `(B, E_max)`
  - Default `E_max`: `768`; pad with masked dummy edges.
  - Sliding pieces stop at the first blocker. Include the blocker square as defended or attacked depending on color. Do not continue through blockers.
  - Pawns create diagonal attack edges, not forward quiet-move edges.
  - Kings create one-square attack edges. Castling is not an attack edge.
  - The builder does not test whether a move leaves its own king in check. That would move toward search; keep this pseudo-legal and local.

- `SquareStem`
  - A per-square MLP or `1x1` projection from raw planes plus trainable square embedding and side-to-move embedding.
  - Output `z0`: `(B, 64, d_model)`.

- `SheafRestrictionBank`
  - Stores `A_src[r]` and `A_dst[r]` for each relation type `r`.
  - Default relation count: `48` to `96`, depending on how Codex encodes direction buckets.
  - Default matrix form: diagonal plus rank-4 update.
  - Must support a config flag `restriction_form: diagonal_lowrank | full | identity_ablation`.

- `ThreatSheafLayer`
  - Gathers source and target node states.
  - Computes edge tension:

```text
eta_e = A_src[type_e] z_src - A_dst[type_e] z_dst
energy_e = ||eta_e||^2
```

  - Computes optional gate:

```text
g_e = sigmoid(MLP([z_src, z_dst, edge_type_embedding, target_role_embedding]))
```

  - Scatters sheaf-gradient messages back to source and target squares:

```text
m_src +=  A_src[type_e]^T (g_e eta_e)
m_dst += -A_dst[type_e]^T (g_e eta_e)
```

  - Updates nodes:

```text
z_next = LayerNorm(z - step_size * scatter_message + residual_mlp(z) + contest_message)
```

- `ContestCellPool`
  - For each target square, aggregate incoming edge energies by relative color and target role.
  - Suggested per-square stats:
    - attacker-energy sum from side-to-move
    - attacker-energy sum from not-side-to-move
    - max incoming edge energy
    - count of incoming side-to-move attackers
    - count of incoming not-side-to-move attackers
    - signed imbalance `(stm_energy - non_stm_energy)`
  - Feed stats through a small MLP to produce `contest_message`: `(B, 64, d_model)`.

- `SheafReadout`
  - Pools:
    - mean and max of final square states: `(B, 2d)`
    - sum/mean/max of edge energy by relation group: `(B, R_group * 3)`
    - contest stats pooled over targets: `(B, S)`
    - optional side-to-move global embedding: `(B, d_side)`
  - Classifier MLP produces logits `(B, num_classes)`.

### Forward-pass pseudocode

```text
forward(x):
    board = EncodingPieceAdapter(x, encoding_name)
    edges = PseudoLegalAttackBuilder(board)
    z = SquareStem(board.square_raw, board.side_to_move, square_id_embedding)

    all_edge_energy = []
    for layer in ThreatSheafLayer[0:T]:
        z, edge_energy = layer(z, edges)
        all_edge_energy.append(edge_energy)

    contest_stats = ContestCellPool(all_edge_energy, edges)
    pooled = SheafReadout.pool(z, all_edge_energy, contest_stats, board.side_to_move)
    logits = Classifier(pooled)
    return logits
```

### Tensor shapes

With default `d_model = 64`, `T = 3`, `E_max = 768`:

- input: `(B, C, 8, 8)`
- square raw: `(B, 64, C)`
- node embedding: `(B, 64, 64)`
- edge source/target/type/mask: `(B, 768)`
- edge tension per layer: `(B, 768, 64)`
- edge energy per layer: `(B, 768)`
- contest message: `(B, 64, 64)`
- logits: `(B, 2)` by default

### Parameter estimate

Default minimal model, approximate:

- square stem for `C = 18`: about `6k` to `12k` parameters;
- square stem for `C = 112`: about `12k` to `24k` parameters;
- restriction maps with `R = 64`, `d = 64`, rank `4`, source and target maps: about `64 * 2 * (64 + 2*64*4) = 73,728` parameters;
- edge gate MLP and edge-type embeddings: about `25k` to `45k` parameters;
- residual node MLPs for three layers: about `75k` to `100k` parameters if layer-specific, less if shared;
- readout/classifier: about `20k` to `40k` parameters.

Expected total: roughly `180k` to `280k` parameters, depending on relation count and whether layer weights are shared. This is intentionally smaller than an LC0-style residual tower.

### FLOP/complexity estimate

Let:

- `B`: batch size
- `E`: actual number of attack edges, padded to `E_max`
- `d`: hidden dimension
- `q`: low-rank restriction rank
- `T`: number of sheaf layers

Low-rank restriction application is approximately:

```text
O(B * T * E * d * q)
```

plus scatter/gather and MLP costs. With `E_max = 768`, `d = 64`, `q = 4`, `T = 3`, the sheaf-specific cost is modest compared with a deep CNN. If Codex chooses full `64x64` restriction matrices for an ablation, cost becomes `O(B * T * E * d^2)` and should be treated as a separate scaling experiment.

### Config fields

Required config fields:

```yaml
model_name: threat_sheaf
encoding_name: simple_18
num_classes: 2
d_model: 64
num_sheaf_layers: 3
max_edges: 768
restriction_form: diagonal_lowrank
restriction_rank: 4
relation_type_count: 64
use_edge_gates: true
use_contest_pool: true
use_square_embeddings: true
share_sheaf_layers: false
dropout: 0.10
```

### Encoding support

- `simple_18`: primary minimal experiment. Use current piece planes and side-to-move if present. If castling/en-passant are absent, do not invent them.
- `lc0_static_112`: use current-position piece planes for attack graph construction and all planes for `SquareStem` input. Do not use policy/value/search outputs.
- `lc0_bt4_112`: same as `lc0_static_112`; history planes can enter the square stem, but the attack graph should be built from the current board slice only unless the project already has a documented current-board extractor for BT4.

### Logits interface

The `forward` method returns logits only:

```text
logits: Tensor[B, num_classes]
```

Any diagnostic tensors such as edge energy should be returned only behind an explicit debug flag and must not change the training/evaluation API expected by existing benchmark scripts.

## 8. Loss, Training, And Regularization

Primary loss:

```text
CrossEntropyLoss(logits, binary_y)
```

Class weighting:

- Use inverse square-root binary class frequency computed on the training split only.
- Clip weights to `[0.5, 2.0]`.
- Do not use fine labels for class weights unless the existing benchmark already trains on fine labels, which this prompt says it does not.

Optional auxiliary losses, off by default in the minimal experiment:

- `restriction_conditioning_loss`: small penalty encouraging restriction maps not to collapse:

```text
lambda_cond * mean_r ||A_r^T A_r - I||_F^2 / d^2
```

  Suggested `lambda_cond = 1e-4` only if training is unstable.

- `gate_entropy_floor`: small penalty preventing all gates from saturating at zero in the first epochs. Use only as a stabilization ablation, not as the main result.

Regularization:

- dropout: `0.10` in residual node MLP and classifier;
- weight decay: `1e-4`;
- gradient clipping: global norm `2.0`;
- no label smoothing in the minimal experiment unless all baselines use it.

Optimizer and LR:

- Optimizer: AdamW.
- Learning rate: `3e-4` for minimal experiment.
- Scheduler: keep the repository's standard scheduler if baselines already use one; otherwise cosine decay with warmup `5%` of steps.
- Batch size: start with `256` for `simple_18`; use `128` for 112-plane encodings if memory requires it.

Determinism:

- Run seeds `0`, `1`, `2`.
- Use deterministic data split files exactly as given.
- Set PyTorch, NumPy, and Python seeds.
- Avoid nondeterministic scatter kernels if the repository has deterministic alternatives; if not, log that exact nondeterminism.

What must stay fixed for fair comparison:

- train/val/test split;
- encoding;
- binary labels;
- epoch budget or early-stopping policy;
- optimizer family and LR unless the repository has a fixed baseline protocol;
- no extra data;
- no engine-derived features;
- same metric computation and same per-fine-label reporting.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Fixed grid sheaf, same parameter budget | Replace pseudo-legal attack edges with 8-neighbor board-grid edges and relation types for compass direction | The dynamic chess attack complex, not just sheaf math, carries useful signal | If equal or better, the central attack-incidence claim is falsified. |
| Identity restrictions | Set `A_src = A_dst = I` and keep edge gates/messages | Nontrivial sheaf transports matter beyond typed message passing | If equal or better, learned restriction geometry is unnecessary. |
| No contest pooling | Remove contested-square summaries, keep sheaf diffusion | Multi-attacker/multi-defender local cells add value | If unchanged, edge tensions alone are sufficient or contest pooling is poorly designed. |
| Edge type shuffle | Randomly permute edge types within each batch while preserving edge endpoints and masks | Chess relation semantics matter, not just edge count | If unchanged, type system is not being used. |
| No gates | Force all `g_e = 1` | Learned edge relevance is needed to ignore noisy pseudo-legal attacks | If no gates improve performance, gates are overfitting or suppressing useful tension. |
| Attack-count MLP | Replace sheaf layers with deterministic per-square and global attack/defense counts fed to an MLP | Structured sheaf diffusion improves over simple handcrafted counts | If counts match performance, the idea should be simplified or abandoned. |
| CNN stem only | Use `SquareStem + pooled classifier`, no attack edges | Gains come from the relation operator rather than a new stem/readout | If equal, relation layers are not helping. |
| Relative-color ablation | Encode edge color as absolute white/black instead of side-to-move relative color | Side-to-move-relative tactical roles are important | If absolute color wins, the adapter's relative orientation may be wrong or the dataset has color bias. |
| Full matrices instead of low-rank restrictions | Use full `d x d` restrictions for a small run | Low-rank restrictions are sufficient | If full matrices greatly improve, low-rank bottleneck is too severe; scale only after central graph ablation passes. |

Smallest ablation that can falsify the central claim: fixed grid sheaf with matched parameter count and training protocol. If it matches or beats `ThreatSheafNet`, the position-dependent attack-defense complex is not the source of any gain.

## 10. Benchmark And Falsification Criteria

Baselines to compare against on the same encoding and split:

- simple CNN;
- residual CNN;
- small/medium/deep CNN variants;
- LC0 BT4-style CNN if using a 112-plane encoding;
- LC0 BT4-style residual CNN if using a 112-plane encoding;
- any current best non-ensemble model in the repository.

Primary metrics:

- binary accuracy;
- balanced accuracy;
- ROC-AUC;
- PR-AUC;
- macro F1;
- confusion matrix for binary labels;
- required fine-label report: `true fine label 0/1/2 -> predicted binary output 0/1`.

Secondary diagnostics:

- fine label `1` recall at matched fine label `0` false-positive rate;
- calibration error if repository already computes it;
- runtime per batch;
- parameter count;
- peak memory.

Artifacts Codex should produce:

- training logs for each seed;
- validation and test metrics;
- model parameter count;
- benchmark command lines;
- per-fine-label prediction table;
- ablation metrics table;
- saved config YAML;
- no extra labels or engine outputs.

Success threshold:

- On `simple_18`, mean over three seeds beats the strongest same-encoding non-ensemble baseline by at least one of:
  - `+2.0` percentage points balanced accuracy, or
  - `+0.020` ROC-AUC, or
  - `+0.030` macro F1;
- and improves fine label `1` recall by at least `+3.0` percentage points at no more than `+1.0` percentage point fine label `0` false-positive rate.

Failure threshold:

- Mean improvement over the strongest same-encoding baseline is within `±0.5` percentage points balanced accuracy and within `±0.005` ROC-AUC;
- and there is no consistent fine label `1` recall improvement across at least two of three seeds.

Abandon condition:

- The fixed grid sheaf ablation equals or beats the attack-sheaf model on validation and test; or
- the identity-restriction ablation equals or beats it while using fewer parameters; or
- the attack builder causes more than `2.5x` wall-clock training slowdown with no validation gain; or
- gains appear only on one seed and vanish under deterministic reruns.

Scaling condition:

Only after the minimal `simple_18` experiment beats the fixed-grid and identity-restriction ablations should Codex scale to:

1. `lc0_static_112`;
2. `lc0_bt4_112`;
3. `d_model = 96` or `num_sheaf_layers = 4`.

Scaling is not the core idea and should not be used to rescue a failed minimal experiment.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_threat_sheaf/codex_handoff.md` | Create | Copy this Markdown packet verbatim for repo-local provenance. |
| `ideas/20260421_threat_sheaf/results.md` | Create after experiments | Metrics, ablations, seed table, fine-label report, failure/success decision. |
| `src/chess_nn_playground/models/threat_sheaf.py` | Create | `ThreatSheafNet`, `EncodingPieceAdapter`, `PseudoLegalAttackBuilder`, `SheafRestrictionBank`, `ThreatSheafLayer`, `ContestCellPool`, and `SheafReadout`. Keep helper classes in this file unless repo style prefers a submodule. |
| `src/chess_nn_playground/models/registry.py` | Modify | Register `threat_sheaf` with the existing model factory. |
| `configs/threat_sheaf_simple18.yaml` | Create | Minimal experiment config using `simple_18`, `d_model: 64`, `num_sheaf_layers: 3`, `restriction_form: diagonal_lowrank`. |
| `configs/threat_sheaf_lc0_static_112.yaml` | Create only after minimal success | Same model over `lc0_static_112`; graph from current piece planes, stem from all planes. |
| `configs/threat_sheaf_lc0_bt4_112.yaml` | Create only after minimal success | Same model over `lc0_bt4_112`; graph from current board slice only. |
| `tests/test_threat_sheaf_attack_builder.py` | Create | Unit tests for knight attacks, rook blockers, bishop blockers, pawn attack direction, own-piece defense, enemy-piece attack, and edge padding masks. |
| `tests/test_threat_sheaf_forward.py` | Create | Forward-shape tests for `(B, C, 8, 8)` input and logits `(B, 2)` for each supported encoding when adapter metadata exists. |
| `tests/test_threat_sheaf_no_label_inputs.py` | Create if test framework allows | Assert model `forward` accepts only tensors/config and not labels, fine labels, engine fields, or metadata features. |
| `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Modify after consuming | Preserve hard constraints; add lessons, anti-duplicate rules, clearer output requirements, and failure-mode guidance from this pass. Codex must update this file after implementation and benchmarking. |

Implementation notes:

- Prefer a deterministic, vectorized attack builder, but correctness beats speed for the first run.
- Avoid depending on `python-chess` inside the model forward if that would break GPU batching. It is acceptable in tests or offline validation.
- If attack graph construction must run on CPU initially, log the overhead and keep the batch size fair. GPU vectorization can be a follow-up only after the central ablation passes.
- Do not silently infer encoding channel maps. Use existing repository encoding metadata or explicit config mappings.
- Do not add any engine calls, evaluator calls, or verifier-derived files.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0255_tuesday_local_threat_sheaf.md
  path: /mnt/data/chess_nn_research_2026-04-21_0255_tuesday_local_threat_sheaf.md
  generated_at: "2026-04-21T02:55:30-07:00"
  weekday: tuesday
  timezone_token: local
  content_type: text/markdown
  single_artifact: true
```

```yaml
idea_yaml:
  idea_id: "20260421_threat_sheaf"
  idea_name: "Tactical Threat-Sheaf Network"
  idea_slug: "threat_sheaf"
  task: "binary chess puzzle-likeness classification"
  core_object: "learned cellular sheaf Laplacian over deterministic pseudo-legal attack-defense complex"
  input_shape: "(batch, C, 8, 8)"
  output_shape: "(batch, num_classes)"
  default_num_classes: 2
  supported_encodings:
    - simple_18
    - lc0_static_112
    - lc0_bt4_112
  forbidden_inputs:
    - stockfish_scores
    - engine_principal_variations
    - node_counts
    - verification_metadata
    - source_labels_as_features
    - proposed_labels_as_features
    - unresolved_candidate_flags_as_features
    - fabricated_class_1_or_2_labels
  central_falsification_ablation: "replace attack-defense edges with matched fixed grid edges"
  minimal_experiment:
    encoding: simple_18
    split: data/splits/crtk_sample_3class
    seeds: [0, 1, 2]
    compare_to: "strongest same-encoding non-ensemble baseline"
```

```yaml
config_yaml:
  model:
    name: threat_sheaf
    encoding_name: simple_18
    num_classes: 2
    d_model: 64
    num_sheaf_layers: 3
    max_edges: 768
    relation_type_count: 64
    restriction_form: diagonal_lowrank
    restriction_rank: 4
    use_edge_gates: true
    use_contest_pool: true
    use_square_embeddings: true
    share_sheaf_layers: false
    dropout: 0.10
  attack_builder:
    graph_kind: pseudo_legal_attack_defense
    include_empty_attacked_squares: true
    include_own_piece_defense_edges: true
    include_enemy_piece_attack_edges: true
    include_king_attack_edges: true
    include_castling_edges: false
    test_self_check_legality: false
    sliding_stops_at_first_blocker: true
  training:
    loss: cross_entropy
    class_weighting: inverse_sqrt_train_binary_frequency_clipped
    class_weight_clip: [0.5, 2.0]
    optimizer: adamw
    learning_rate: 0.0003
    weight_decay: 0.0001
    batch_size: 256
    gradient_clip_norm: 2.0
    dropout: 0.10
    seeds: [0, 1, 2]
  data:
    train_split: data/splits/crtk_sample_3class/split_train.parquet
    val_split: data/splits/crtk_sample_3class/split_val.parquet
    test_split: data/splits/crtk_sample_3class/split_test.parquet
  evaluation:
    metrics:
      - accuracy
      - balanced_accuracy
      - roc_auc
      - pr_auc
      - macro_f1
      - binary_confusion_matrix
      - fine_label_prediction_breakdown
```

```yaml
model_spec:
  class_name: ThreatSheafNet
  module_path: src/chess_nn_playground/models/threat_sheaf.py
  registry_name: threat_sheaf
  forward_signature: "forward(x: Tensor) -> Tensor"
  input_tensor: "float tensor shaped (batch, C, 8, 8)"
  output_tensor: "logits shaped (batch, num_classes)"
  submodules:
    - EncodingPieceAdapter
    - PseudoLegalAttackBuilder
    - SquareStem
    - SheafRestrictionBank
    - ThreatSheafLayer
    - ContestCellPool
    - SheafReadout
  default_shapes:
    node_state: "(batch, 64, 64)"
    edge_index: "(batch, 768) for src and dst"
    edge_type: "(batch, 768)"
    edge_tension: "(batch, 768, 64)"
    logits: "(batch, 2)"
  parameter_budget: "about 180k-280k parameters for d_model=64, rank=4, three layers"
  required_tests:
    - attack_builder_knight_center
    - attack_builder_slider_blockers
    - attack_builder_pawn_direction
    - forward_shape_simple18
    - forward_shape_112_if_metadata_available
    - no_label_or_engine_inputs
```

```yaml
research_continuity:
  idea_fingerprint: "dynamic typed pseudo-legal attack graph + learned nontrivial sheaf restrictions + tension-energy readout; no engine features"
  closest_duplicate_risk: "plain chess GNN or attack-map feature MLP; distinguish by requiring sheaf restriction ablation and fixed-grid falsification"
  do_not_repeat_if_this_fails:
    - "Do not propose another static attack-defense graph model unless it adds genuine counterfactual move deltas or legal consequence modeling."
    - "Do not rebrand this as a larger GNN, graph Transformer, or LC0 tower."
    - "Do not tune relation counts, depth, or hidden size as the next core idea if fixed-grid or identity-restriction ablations match performance."
  suggested_next_search_directions:
    - "one-ply counterfactual board-delta operators without engine scores"
    - "differentiable legal-move consequence bottleneck trained only from puzzle labels"
    - "causal invariance across color/side-to-move transformations with explicit pawn/castling exceptions"
    - "optimal transport between attacker mass and defender mass over legal target squares"
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add an anti-duplicate note: “If `ThreatSheafNet` fails its fixed-grid or identity-restriction ablation, do not propose another static attack-defense graph/sheaf model without a new counterfactual mechanism.” | Prevents the next cycle from repeating the same idea under a new name. | `Common approaches / anti-duplicate guidance` or equivalent future section. |
| Clarify that deterministic pseudo-legal attack maps computed from board planes are allowed, while engine-evaluated move quality remains forbidden. | Avoids accidental over-refusal or accidental leakage. This does not weaken the engine-feature ban. | `Hard Constraints` leakage subsection. |
| Require every future idea to name its smallest central falsification ablation in the executive summary. | Forces testability and prevents vague deep-math proposals. | `Required Markdown File Content`, section `2. Executive Selection`. |
| Require encoding-adapter assumptions to be explicit for `simple_18`, `lc0_static_112`, and `lc0_bt4_112`. | Many architecture ideas silently assume channel mappings; Codex needs implementation-ready details. | `Required Markdown File Content`, section `7. Architecture Specification`. |
| Add prompt text saying that scaling hidden size, relation count, depth, or history channels may only happen after the minimal experiment passes its ablations. | Keeps future research from degenerating into hyperparameter tuning. | `Hard Constraints` and `Benchmark And Falsification Criteria`. |

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
