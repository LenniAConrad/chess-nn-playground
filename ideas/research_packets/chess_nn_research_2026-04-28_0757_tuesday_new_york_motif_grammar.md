# Codex Handoff Packet: Typed Hypergraph Motif Grammar

## 1. File Metadata

- **Filename:** `chess_nn_research_2026-04-28_0757_tuesday_new_york_motif_grammar.md`
- **Created:** 2026-04-28 07:57 new_york
- **Target repository:** `chess-nn-playground`
- **Target task:** `puzzle_binary`
- **Idea name:** Typed Hypergraph Motif Grammar, abbreviated `THMG`
- **Central mechanism:** a typed hyperedge replacement grammar over deterministic current-board relation facts, trained with a differentiable hypergraph chart composer.
- **Allowed inputs:** current-board tensor and deterministic relation facts computed only from the current board.
- **Forbidden inputs:** engine scores, principal variations, node counts, mate scores, solution moves, best moves, verification metadata, source labels, theme labels, or any field derived from solving the position.
- **Explicitly not used:** weighted finite automata, ray-language automata, subgoal automata, program induction, clause resolution, or proof-tree construction.

## 2. Executive Selection

Select **Typed Hypergraph Motif Grammar** as the research direction.

The model should convert each current chess position into a typed relational hypergraph: pieces, squares, kings, rays, attacks, defenses, pins, blockers, and king-zone relations become typed nodes and terminal hyperedges. A learnable hyperedge replacement grammar then composes small tactical facts into larger latent motifs such as pressure, fork-shape, pin-shape, overload-shape, battery-shape, and king-zone convergence. The final `puzzle_binary` prediction is made from the pooled scores and embeddings of all composed motif derivations, optionally fused with a normal current-board tensor encoder.

This is the cleanest fit for the constraints because it gives Codex a concrete grammar object and a concrete composition operator while staying entirely inside the current position. The parser never asks what the best move is, never checks an engine line, never evaluates mate distance, and never consumes puzzle-source metadata. It learns whether the current-board arrangement contains compositional tactical structure predictive of `puzzle_binary`.

The core bet is direct: many chess puzzles are not identifiable from isolated atomic facts alone. They are identifiable from **compositions** of facts: an attacked target plus a pinned defender plus an overloaded guard plus king-zone pressure. `THMG` forces the network to build these compositions explicitly instead of hoping a generic encoder discovers them as opaque correlations.

## 3. Data Contract

### Input batch

Each example contains exactly:

```text
board_tensor: FloatTensor[B, C_board, 8, 8]
relation_facts: deterministic facts derived from the same current board
y: BoolTensor[B] or FloatTensor[B] for puzzle_binary
```

`board_tensor` may include piece planes, side-to-move plane, castling-right planes if already present in the current-board representation, en-passant target plane if it is part of the legal current state, and other non-solving current-state encodings. It must not include anything from a solver, line verifier, puzzle generator, or source annotation.

### Deterministic relation facts

The feature extractor may compute relation facts by applying fixed chess geometry and current-board occupancy rules. Acceptable fact families:

```text
Piece(piece_id, color, piece_type, square)
SideToMove(color)
Square(file, rank, color_complex, edge_distance, center_distance)
SameFile(square_a, square_b)
SameRank(square_a, square_b)
SameDiagonal(square_a, square_b)
KnightReach(square_a, square_b)
KingReach(square_a, square_b)
Ray(square_a, square_b, direction, distance)
Between(square_a, square_mid, square_b, direction)
Occupied(square, piece_id)
AttacksSquare(piece_id, square)
AttacksPiece(piece_id, target_piece_id)
DefendsPiece(piece_id, friendly_piece_id)
DefendsSquare(piece_id, square)
AttackedByColor(color, square)
DefendedByColor(color, square)
Contested(square)
KingOf(color, king_piece_id, square)
KingZone(color, square)
NearKing(piece_id, king_piece_id, distance_bucket)
LineOfSight(piece_id, square_or_piece, direction)
SliderAligned(slider_piece_id, piece_or_square, direction)
OnlyBlockerBetween(blocker_piece_id, endpoint_a, endpoint_b, direction)
PinnedToKing(pinned_piece_id, king_piece_id, pinner_piece_id, direction)
DiscoveredLine(blocker_piece_id, friendly_slider_or_enemy_slider, target_piece_or_king, direction)
LoosePiece(piece_id)
UnderdefendedPiece(piece_id)
HighValueTarget(piece_id)
```

`LoosePiece`, `UnderdefendedPiece`, and `HighValueTarget` must be computed from deterministic attack/defense counts and static piece-type ranks only. They are not evaluations. They must not use engine scores, exchange search, move search, or solved-line metadata.

### Forbidden columns and leak checks

The dataset loader must fail closed if a requested feature column or metadata key matches any forbidden family. Suggested blocklist patterns:

```text
score, eval, cp, centipawn, mate, pv, line, variation, node, depth,
best, best_move, solution, move_uci, move_san, principal, engine,
verified, verification, source, origin, theme, tag, motif_label
```

This blocklist is intentionally conservative. If the repository has benign columns with these substrings, Codex should route them through an explicit allowlist only after confirming they are current-board-only. The default behavior should be to exclude them.

### Label

The only supervised label is:

```text
y = puzzle_binary
```

No auxiliary target may be built from engine lines, puzzle themes, source labels, mate markers, move annotations, or verification metadata.

### Splits

Use the repository's existing split protocol if it is already leak-safe. If a new split is needed, split by canonical current-board key, not by puzzle metadata. Do not stratify by source label or theme label. Deduplicate exact current boards before splitting when feasible.

## 4. Grammar/Compositionality Research Map

The research map has four layers.

| Layer | Object | Role | Why it fits the constraints |
|---|---|---|---|
| Current-board relations | Typed terminal hyperedges | Converts board geometry into symbolic-tensor facts | Deterministic from current board only |
| Typed motif grammar | Hyperedge replacement productions | Composes facts into latent tactical motifs | Concrete grammar, no automaton, no proof system |
| Differentiable chart composer | Soft inside-style dynamic program over graph bindings | Learns which compositions matter for `puzzle_binary` | Trained only from binary labels |
| Prediction head | Pooled motif chart plus optional board encoder | Produces binary logit | No solution move or engine target |

The grammar should be researched as a **hypergraph grammar**, not as a language recognizer over rays. Chess tactics are naturally relational: a motif can bind an attacker, target, defender, king, blocker, and square region in the same object. Linear strings are a poor fit. Hypergraphs let one production share several pieces or squares across child motifs without converting the position into a path or automaton state.

The useful compositional view is:

```text
atomic relation facts -> primitive pressure motifs -> coupled motifs -> board-level tactical-potential motif
```

Examples:

```text
AttacksPiece(a, t)
  -> Pressure(a, t)

Pressure(a, t) + LoosePiece(t)
  -> LooseTargetPressure(a, t)

PinnedToKing(d, k, pinner) + DefendsPiece(d, t)
  -> CompromisedDefender(d, t, k, pinner)

LooseTargetPressure(a, t) + CompromisedDefender(d, t, k, pinner)
  -> TacticalConvergence(a, t, d, k)

TacticalConvergence(...) + KingZonePressure(...)
  -> PuzzleLikeMotif(...)
```

This is deliberately not a proof that a tactic works. It is a compositional latent structure detector for current-board tactical density.

Three concrete research variants are worth keeping in scope:

1. **Typed HRG parser:** the primary path. Productions replace a nonterminal hyperedge with a typed sub-hypergraph. This gives the strongest grammar object.
2. **Typed motif algebra:** an implementation-friendly abstraction where motifs are typed tensors and composition is a masked join over shared ports. This is the recommended engineering surface even if the paper framing uses HRG language.
3. **Categorical pushout composition:** an optional mathematical interpretation. A motif is a morphism with typed boundary ports; composing motifs is gluing along matching ports. This helps keep the operator precise, but it should not become category-theory theater in the implementation.

Do not route this idea through finite-state paths, automata over ray symbols, induced programs, resolution proofs, or subgoal graphs. Those alternatives either violate the explicit constraints or move the project away from a current-board neural predictor.

## 5. Serious Candidate Rejections

### Rejection A: Ray-language recognizer

A ray-language recognizer encodes files, ranks, and diagonals as token sequences and detects tactical patterns with a finite-state or sequence model. Reject it. It overfits line pieces, handles non-line motifs awkwardly, and violates the prohibition on ray-language automata. It also encourages a misleading belief that tactics are mostly one-dimensional.

### Rejection B: Weighted finite automaton over attack paths

A weighted finite automaton could score paths such as attacker-blocker-target or attacker-defender-king. Reject it. Even if useful, it is explicitly out of bounds. The selected method must compose typed hypergraph fragments, not accept or score strings with automaton states.

### Rejection C: Subgoal automaton or tactical plan graph

A subgoal automaton would represent progress toward tactical objectives. Reject it. It smuggles in action-sequence semantics and invites hidden dependence on solved continuations. The task here is not to model a move plan; it is to classify the current board from current-board facts.

### Rejection D: Program induction over board predicates

Inducing programs such as `if pinned(x) and attacks(y,z) then ...` is tempting. Reject it. It risks becoming symbolic program search, encourages brittle hand-authored predicate logic, and violates the ban on program induction. The grammar can have typed productions, but learning must be differentiable and bounded, not an open-ended program synthesis process.

### Rejection E: Clause resolution or proof trees

A logic engine could derive tactical conclusions from predicates. Reject it. The requested output is a compositional grammar model, not a theorem prover. The parser may maintain latent derivation scores, but these derivations are not proof trees and must not be interpreted as verification of a tactic.

### Rejection F: Theme-label classifier

Training on puzzle theme labels such as fork, pin, skewer, mate, or discovered attack would be convenient. Reject it. Theme labels are source annotations and are outside the allowed contract. Motif names inside `THMG` are latent internal types, not supervised labels.

### Rejection G: Engine-distilled tactic detector

Distilling from engine score swings, principal variations, mate distance, best moves, or node-count features would likely improve metrics. Reject it. It directly violates the forbidden-input rules and would answer a different research question.

## 6. Common Approaches Rejected

- **Board tensor only, no grammar:** acceptable as a baseline, rejected as the main idea because it does not create a compositional motif grammar.
- **Generic GNN over pieces:** useful baseline, rejected as the main mechanism because message passing alone does not provide an explicit grammar object or production-level composition.
- **Move-conditioned classification:** rejected because it requires candidate moves or move labels and drifts toward best-move supervision.
- **Search-tree supervision:** rejected because it requires solved continuations, verification metadata, or engine traces.
- **Mate detector auxiliary head:** rejected because mate labels or mate distances are forbidden.
- **Puzzle-source balancing with source labels:** rejected because source labels are forbidden. Splits may use current-board identity, but not origin metadata.
- **Manual tactical rule list as final classifier:** rejected because it would become brittle expert logic. Hand-designed relation facts are acceptable; the composition parameters must be learned from `puzzle_binary`.
- **Automaton-based motif library:** rejected even if it looks compact. The central mechanism must remain hypergraph or typed-algebraic composition.

## 7. Mathematical Thesis

Let `B` be a current chess board. A deterministic relation extractor maps it to a typed hypergraph:

```text
H_B = (V_B, E_B, tau_V, tau_E)
```

`V_B` contains piece nodes, square nodes, color nodes, and optional region nodes. `E_B` contains terminal hyperedges such as `AttacksPiece(a,t)`, `PinnedToKing(d,k,p)`, and `KingZone(c,s)`. All terminal hyperedges are deterministic functions of `B`.

Define a typed hyperedge replacement grammar:

```text
G = (N, T, P, S, arity, type)
```

- `T` is the set of terminal relation types.
- `N` is the set of nonterminal motif types.
- `S` is the start motif type, e.g. `PuzzleLikeMotif`.
- Each nonterminal has typed boundary ports, such as `Pressure(attacker: Piece, target: Piece)`.
- Each production replaces a parent nonterminal with a small typed hypergraph whose boundary ports match the parent.

A production has the form:

```text
A[x_1:T_1, ..., x_m:T_m]
  => R(terminals, child_nonterminals, equalities, inequalities, type_masks)
```

The parser computes a soft chart value for every nonterminal and every compatible tuple of board entities:

```text
C_A(u) = logsumexp over productions p and bindings beta:
         theta_p
       + phi_p(terminals matched by beta)
       + sum_i C_{A_i}(u_i)
       + mask_p(beta)
```

where:

- `u` is the boundary tuple for parent nonterminal `A`.
- `beta` binds production variables to current-board pieces or squares.
- `mask_p(beta)` is `0` for type-consistent legal bindings and `-inf` otherwise.
- `theta_p` and `phi_p` are learnable production parameters and neural compatibility functions.
- The logsumexp aggregates all latent derivations without selecting a proof, move, or line.

The binary logit is:

```text
z(B) = h(pool({C_S(u)}_u), pool({C_A(u)}_{A,u}), BoardEncoder(B))
y_hat = sigmoid(z(B))
```

`BoardEncoder(B)` is optional but allowed because it consumes only the current-board tensor. The central object remains the grammar chart, not the board encoder.

Thesis: if `puzzle_binary` reflects compositional tactical structure, then a typed hypergraph grammar with bounded-depth differentiable composition should outperform terminal-only relation features and generic board encoders under leak-safe splits. If it does not, then either the dataset's binary label is not strongly compositional under current-board-only information, or the grammar object is too weak or too constrained.

## 8. Grammar Object

### Core types

```text
Entity types:
  Piece[color, piece_type]
  Square[file, rank]
  Color
  Direction
  Region

Terminal hyperedge types:
  PieceAt(piece, square)
  AttacksPiece(piece, piece)
  AttacksSquare(piece, square)
  DefendsPiece(piece, piece)
  DefendsSquare(piece, square)
  Ray(square, square, direction)
  Between(square, square, square, direction)
  OnlyBlockerBetween(piece, entity, entity, direction)
  PinnedToKing(piece, king, pinner, direction)
  KingZone(color, square)
  NearKing(piece, king, bucket)
  LoosePiece(piece)
  UnderdefendedPiece(piece)
  HighValueTarget(piece)
  SameColor(piece, piece)
  OppColor(piece, piece)
  SideToMove(color)
```

### Nonterminal motif types

```text
Pressure(attacker: Piece, target: Piece)
LooseTargetPressure(attacker: Piece, target: Piece)
KingZonePressure(attacker: Piece, king: Piece, square: Square)
LinePressure(slider: Piece, blocker: Piece, target: Piece)
PinShape(pinner: Piece, pinned: Piece, king: Piece)
ForkShape(attacker: Piece, target_1: Piece, target_2: Piece)
BatteryShape(front: Piece, rear: Piece, target: Piece)
CompromisedDefender(defender: Piece, defended: Piece, king_or_target: Piece)
OverloadShape(defender: Piece, obligation_1: Piece, obligation_2: Piece)
TacticalConvergence(anchor: Piece, target: Piece, support: Piece, king: Piece)
PuzzleLikeMotif(anchor: Piece, target: Piece, king: Piece)
```

The names are internal latent grammar symbols. They must not be supervised by theme labels.

### Production examples

Notation:

- `&` means typed conjunction inside one production body.
- `share(x)` means child motifs must bind the same board entity at that port.
- `distinct(x,y)` is a deterministic inequality mask.
- `forget(x)` means the variable is internal to the production and not exposed in the parent boundary.

```text
P1: Pressure(a, t)
    => AttacksPiece(a, t) & OppColor(a, t)

P2: LooseTargetPressure(a, t)
    => Pressure(a, t) & LoosePiece(t)

P3: LooseTargetPressure(a, t)
    => Pressure(a, t) & UnderdefendedPiece(t)

P4: KingZonePressure(a, k, s)
    => AttacksSquare(a, s) & KingZone(color_of(k), s) & NearKing(a, k, *)

P5: PinShape(pinner, pinned, king)
    => PinnedToKing(pinned, king, pinner, *)

P6: LinePressure(slider, blocker, target)
    => OnlyBlockerBetween(blocker, slider, target, *)
     & SliderAligned(slider, target, *)

P7: ForkShape(a, t1, t2)
    => Pressure(a, t1) & Pressure(a, t2) & distinct(t1, t2) & HighValueTarget(t1)

P8: BatteryShape(front, rear, target)
    => SameColor(front, rear)
     & OnlyBlockerBetween(front, rear, target, *)
     & LinePressure(rear, front, target)

P9: CompromisedDefender(d, t, k)
    => DefendsPiece(d, t) & PinShape(_, d, k)

P10: OverloadShape(d, t1, t2)
     => DefendsPiece(d, t1)
      & DefendsPiece(d, t2)
      & distinct(t1, t2)
      & LooseTargetPressure(_, t1)
      & LooseTargetPressure(_, t2)

P11: TacticalConvergence(a, t, d, k)
     => LooseTargetPressure(a, t)
      & CompromisedDefender(d, t, k)

P12: TacticalConvergence(a, t, d, k)
     => Pressure(a, t)
      & OverloadShape(d, t, _)
      & KingZonePressure(_, k, _)

P13: PuzzleLikeMotif(a, t, k)
     => TacticalConvergence(a, t, _, k)
      & KingZonePressure(_, k, _)

P14: PuzzleLikeMotif(a, t, k)
     => ForkShape(a, t, k)
      & HighValueTarget(t)
```

These examples are not intended as a complete chess theory. They define the implementation pattern: typed boundary ports, deterministic terminal matches, learned production compatibility, and bounded composition depth.

### Composition operators

The implementation should expose four operators:

```text
join_shared(M1, M2, shared_ports)
```

Composes motif charts by requiring selected ports to bind the same piece or square.

```text
join_relation(M, R, port_mapping)
```

Attaches terminal relation facts to a motif chart.

```text
forget(M, internal_ports)
```

Marginalizes internal variables with logsumexp or attention pooling, producing a lower-arity parent motif.

```text
rename(M, port_permutation)
```

Reorders ports to match a parent production signature.

Together these implement hyperedge replacement in tensor form. They are not automaton transitions and do not represent a line search.

## 9. Differentiable Parser Or Composer

Use a bounded-depth differentiable chart composer.

### Step 1: Build terminal tensors

For each board, create boolean or float masks for relation types:

```text
T_AttacksPiece[B, P, P]
T_DefendsPiece[B, P, P]
T_PinnedToKing[B, P, P, P]
T_KingZone[B, C, S]
T_AttacksSquare[B, P, S]
T_OnlyBlockerBetween[B, P, E, E, D]
...
```

`P` is the maximum piece-slot count, typically 32. `S = 64`. `E` can be a packed entity axis or separate piece/square variants. Sparse COO storage is acceptable and probably better.

### Step 2: Initialize primitive charts

Primitive productions create first-level nonterminal charts. Example:

```text
Pressure[a,t] = f_pressure(
    emb_piece[a], emb_piece[t], rel_AttacksPiece[a,t], rel_OppColor[a,t]
)
```

Invalid type or relation bindings receive `-inf` score and zero embedding.

### Step 3: Apply grammar productions by depth

For `depth = 1..K`, apply every production whose child motifs are available. A production computes:

```text
candidate_embedding = MLP_p(concat(child_embeddings, terminal_features, geometry_features))
candidate_score = theta_p + score_terms + terminal_bias + compatibility_mlp
parent_chart = logsumexp_or_attention(parent_chart, candidate)
```

`K = 3` or `K = 4` is the likely useful range. Deeper parsing risks memorizing dataset artifacts and increasing compute without clear tactical gain.

### Step 4: Pool chart items

For each nonterminal type, pool chart entries with multiple statistics:

```text
max_score
logsumexp_score
mean_top_k_score
count_above_threshold_soft
attention_pool_embedding
```

The final head receives these pooled motif summaries plus optional current-board CNN output.

### Step 5: Keep derivations latent

Do not train the chart to reproduce named motifs, puzzle themes, solution lines, or engine choices. The chart is an inductive bias and explanation aid, not a proof generator. Optional human-facing explanations may list high-scoring motif instances, but they must be presented as model activations, not as verified tactics.

### Differentiability choices

Recommended default:

```text
score aggregation: logsumexp
embedding aggregation: softmax attention weighted by candidate score
invalid bindings: additive -inf mask
regularization: entropy and production dropout
```

Alternative for ablation:

```text
product t-norm style composition in probability space
sparsemax or entmax candidate selection
hard top-k straight-through selection
```

The default should prioritize stability and clear gradients over interpretability theater.

## 10. Tensor Contract

### Batch-level tensors

```text
board_tensor:
  shape: [B, C_board, 8, 8]
  dtype: float32

piece_active:
  shape: [B, P]
  dtype: bool

piece_attr:
  shape: [B, P, F_piece]
  dtype: float32
  includes: color, type, square index, side-relative features, static type rank

square_attr:
  shape: [B, 64, F_square]
  dtype: float32
  includes: file, rank, edge distance, center distance, color complex
```

### Relation tensors

Use dense tensors for small arities and sparse tensors for high arities.

```text
rel_pp:
  shape: [B, R_pp, P, P]
  examples: attacks_piece, defends_piece, same_color, opp_color

rel_ps:
  shape: [B, R_ps, P, 64]
  examples: attacks_square, defends_square, piece_at_square, near_king_square

rel_ss:
  shape: [B, R_ss, 64, 64]
  examples: same_file, same_rank, same_diagonal, ray_direction_bucket

rel_ppp_sparse:
  fields: batch, relation_type, p1, p2, p3, optional_direction, value
  examples: pinned_to_king, only_blocker_piece_piece_piece
```

### Chart tensors

Each motif type has a port signature and a chart object:

```text
Chart[A]:
  ports: IntTensor[num_items, arity(A)]
  score: FloatTensor[B, num_items]
  embedding: FloatTensor[B, num_items, D_motif]
  mask: BoolTensor[B, num_items]
```

For low-arity motifs, dense charts are acceptable:

```text
Pressure:
  score: [B, P, P]
  embedding: [B, P, P, D]

ForkShape:
  score: [B, P, P, P]
  embedding: [B, P, P, P, D]
```

For high-arity motifs, use sparse candidate lists generated by typed joins. Keep a per-production cap such as `max_candidates_per_board_per_production`, but compute the cap from parser-internal scores only. Do not prune with engine, solution, or source metadata.

### Output tensors

```text
motif_summary:
  shape: [B, F_motif_summary]

logit:
  shape: [B]

prediction:
  shape: [B]
  value: sigmoid(logit)
```

### Device and precision

- Relation masks may be bool or float16.
- Scores should remain float32 for stable logsumexp.
- Embeddings can use mixed precision after mask application.
- Sparse joins should be benchmarked before optimizing dense einsum paths. The grammar is small enough that correctness matters more than clever tensor tricks at first.

## 11. Training Loss

### Primary loss

Use binary cross entropy with logits:

```text
L_bce = BCEWithLogitsLoss(logit, puzzle_binary)
```

If class imbalance exists, use a class-balanced BCE weight computed from the training split only. Do not use source labels, theme labels, or verification labels for balancing.

### Grammar regularizers

Use small regularization terms to prevent degenerate behavior:

```text
L = L_bce
  + lambda_entropy * L_entropy
  + lambda_sparse * L_sparse_activation
  + lambda_depth * L_depth_balance
  + lambda_sym * L_symmetry_consistency
  + lambda_dropout * L_composition_dropout_consistency
```

Recommended meanings:

- `L_entropy`: prevents one production from monopolizing all motif mass too early.
- `L_sparse_activation`: discourages every board from activating every motif.
- `L_depth_balance`: discourages the model from using only depth-1 terminal motifs when deeper compositions are available.
- `L_symmetry_consistency`: enforces consistent predictions under legal board symmetries that preserve the task label, such as color-swap plus side-to-move swap when implemented correctly.
- `L_composition_dropout_consistency`: randomly drops nonessential productions during training and encourages stable predictions.

### Forbidden auxiliary losses

Do not add losses for:

```text
best move
solution move
move ranking
engine score
centipawn swing
mate distance
PV token prediction
node count prediction
theme classification
source prediction
verification status
```

The only label-bearing objective is `puzzle_binary`.

## 12. Ablations

Run ablations that test whether the grammar is doing real work.

### Baselines

1. **Board-only CNN or transformer:** current-board tensor only.
2. **Relation-only MLP/GNN:** deterministic relation facts without grammar productions.
3. **Terminal motif model:** primitive motifs such as `Pressure` and `PinShape`, no depth greater than 1.
4. **Untyped composition model:** same production count, but type masks weakened or removed where safe.
5. **Random grammar structure:** same number of productions and parameters, random valid port signatures.

### Grammar-depth ablations

```text
K = 0: no grammar, pooled terminal facts only
K = 1: primitive motifs
K = 2: pairwise motif composition
K = 3: tactical convergence
K = 4: high-order convergence
```

Expected useful range is `K = 2..3`. If `K = 4` wins only on the training set, the model is probably overfitting.

### Production-family ablations

Remove one family at a time:

```text
no pin productions
no fork productions
no overload productions
no king-zone productions
no battery/line-pressure productions
no loose-target productions
no high-value-target static rank
```

The strongest result would show that several families matter and that higher-order compositions outperform their isolated parts.

### Relation-family ablations

Remove deterministic fact groups:

```text
attack/defense facts only
rays removed
pins removed
king-zone removed
piece-value rank removed
underdefended/loose facts removed
side-to-move removed
```

Side-to-move is especially important. If the model performs nearly the same without side-to-move, inspect for leakage or dataset bias.

### Composition-operator ablations

Compare:

```text
logsumexp inside composition
attention pooling over candidates
max-only pooling
product t-norm composition
sparsemax or entmax composition
hard top-k candidate selection
```

The default should be logsumexp because it is stable and keeps multiple latent derivations alive.

### Fusion ablations

Compare:

```text
grammar only
board encoder only
board encoder + terminal facts
board encoder + full grammar
late fusion vs cross-attention fusion
```

The research claim is strongest if full grammar improves over both board-only and relation-only alternatives.

## 13. Falsification

This idea should be considered falsified or downgraded if any of the following happen under clean data controls.

### Empirical falsifiers

- Full `THMG` does not beat a well-tuned board-only encoder by a meaningful margin on leak-safe validation.
- Full `THMG` does not beat relation-only GNN or terminal-only motif pooling.
- Grammar depth greater than 1 provides no improvement after hyperparameter tuning.
- Performance gains disappear when duplicate or near-duplicate current boards are removed.
- The model's predictions are insensitive to targeted corruption of key relation facts such as pins, attacks, or king-zone facts.
- The model relies almost entirely on material count or static piece-type rank rather than relational composition.

### Interpretability falsifiers

- High-scoring `PuzzleLikeMotif` items are mostly illegal type combinations or nonsensical bindings.
- Production usage collapses to one shallow production family across nearly all positives.
- Motif activations do not change when obvious current-board tactical relations are removed or flipped in controlled synthetic tests.
- Explanations require mentioning a solution move or engine line to make sense.

### Data-contract falsifiers

- Any feature path reads engine evaluation, principal variation, node count, mate marker, best move, solution move, source label, theme label, or verification metadata.
- Any preprocessing step filters examples using forbidden metadata.
- Any split or balancing strategy uses source labels or puzzle themes.

If a data-contract falsifier appears, the experiment is invalid, not merely weakened.

## 14. Codex Implementation Notes

### Repository navigation

Codex should first locate the existing `puzzle_binary` dataset path, current-board tensor construction, training loop, and model registry. Do not assume exact repository paths. Search for names such as:

```text
puzzle_binary
Dataset
DataModule
board_tensor
fen
train
model registry
binary classifier
```

Then add `THMG` as an optional model path without breaking existing baselines.

### Suggested modules

Use repository naming conventions, but the implementation likely wants components equivalent to:

```text
features/current_board_relations.py
models/typed_hypergraph_motif_grammar.py
models/differentiable_chart_composer.py
models/motif_chart.py
configs/thmg_puzzle_binary.yaml
tests/test_current_board_relation_contract.py
tests/test_forbidden_metadata_blocklist.py
tests/test_thmg_shapes.py
tests/test_thmg_symmetry.py
```

These are proposed locations, not mandates. Conform to the existing project layout.

### Implementation sequence

1. **Contract guard:** add a loader-level assertion that selected input columns exclude forbidden metadata.
2. **Relation extractor:** implement deterministic current-board facts from the board tensor or FEN-derived current board.
3. **Typed schema:** define entity types, relation types, motif types, and port signatures as dataclasses.
4. **Primitive charts:** implement `Pressure`, `PinShape`, `KingZonePressure`, and `LooseTargetPressure` first.
5. **Composition operators:** implement `join_shared`, `join_relation`, `forget`, and `rename` with masks.
6. **Full grammar config:** add production definitions as data, not hard-coded scattered logic.
7. **Prediction head:** pool motif charts and fuse with optional board encoder.
8. **Tests:** verify shapes, masks, forbidden-column exclusion, deterministic feature generation, and symmetry consistency.
9. **Ablations:** wire config flags for relation families, production families, depth, and fusion mode.

### Minimal pseudocode

```python
class THMGModel(nn.Module):
    def __init__(self, board_encoder, relation_schema, grammar, hidden_dim):
        super().__init__()
        self.board_encoder = board_encoder
        self.relation_encoder = RelationEncoder(relation_schema, hidden_dim)
        self.composer = DifferentiableChartComposer(grammar, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(grammar.summary_dim + board_encoder.out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch):
        assert_no_forbidden_metadata(batch)
        board_x = batch["board_tensor"]
        facts = build_current_board_relation_facts(board_x, batch.get("current_board_state"))
        terminal_charts = self.relation_encoder(facts)
        motif_charts = self.composer(terminal_charts, facts)
        motif_summary = pool_motif_charts(motif_charts)
        board_summary = self.board_encoder(board_x)
        return self.head(torch.cat([motif_summary, board_summary], dim=-1)).squeeze(-1)
```

### Guardrails for Codex

- The relation extractor may use chess rules and current occupancy. It may not call an engine or puzzle verifier.
- Do not create a candidate-move dataset.
- Do not add a move head.
- Do not train on tactic names.
- Do not treat latent derivations as proofs.
- Keep grammar depth bounded and configurable.
- Keep every production's port signature explicit.
- Prefer a small correct grammar over a large clever one.

### Acceptance tests

A first implementation is acceptable when:

```text
- It trains end-to-end on puzzle_binary.
- It passes forbidden-metadata blocklist tests.
- It produces valid chart shapes for a batch.
- It supports K=0, K=1, K=2, and K=3 depth configs.
- It can run board-only, relation-only, terminal-only, and full-grammar ablations.
- It logs production-family activation summaries without using theme labels.
```

## 15. Prompt Maintenance

Future prompts for this line of work should preserve the following invariant:

```text
The model classifies puzzle_binary from the current board only by composing deterministic current-board relation facts with a concrete typed grammar operator.
```

Keep these constraints explicit in every follow-up prompt:

```text
Allowed:
  current-board tensor
  deterministic current-board relation facts
  static chess geometry
  static piece-type attributes
  binary puzzle label

Forbidden:
  engine scores
  principal variations
  node counts
  mate scores or mate distances
  best moves
  solution moves
  verification metadata
  source labels
  theme labels
  weighted finite automata
  ray-language automata
  subgoal automata
  program induction
  clause resolution
  proof trees
```

Use the phrase **typed hypergraph motif grammar** or **typed motif algebra** when asking Codex to implement the idea. Avoid ambiguous phrases such as "learn tactical rules," "derive a proof," "recognize ray languages," or "find the tactic," because they invite prohibited mechanisms.

A good maintenance prompt is:

```text
Implement THMG for puzzle_binary as a bounded-depth typed hyperedge motif grammar over current-board relation facts. Use differentiable chart composition with typed joins and logsumexp pooling. Do not use engine data, solution moves, theme labels, source labels, automata, program induction, clause resolution, or proof trees. Provide board-only, relation-only, terminal-only, and full-grammar ablations.
```

The research packet should stay centered on grammar composition, not on chess-engine imitation. If later experiments add stronger encoders, keep the grammar path separately measurable so the project can answer the real question: whether compositional current-board motif structure improves `puzzle_binary` prediction under strict no-leak constraints.
