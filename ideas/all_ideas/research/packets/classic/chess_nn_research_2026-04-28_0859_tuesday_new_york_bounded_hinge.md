# Codex Handoff Packet: Bounded Board Hinge Logic

## 1. File Metadata

- **Filename:** `chess_nn_research_2026-04-28_0859_tuesday_new_york_bounded_hinge.md`
- **Generated:** Tuesday, 2026-04-28 08:59 new_york
- **Task:** `puzzle_binary` classification for chess positions.
- **Selected mechanism:** **Bounded Board Hinge Logic** (`BBHL`).
- **Precise logic object:** a closed-world, finite-domain **probabilistic soft logic / hinge-loss Markov random field energy-gap classifier** over board-local facts.
- **Core design constraint:** evaluate shallow, bounded, current-board formulas only. Do not build proof trees, recurse through clauses, search moves, ask an engine, pool one-ply move deltas, infer tactical subgoals, or use source/provenance features.
- **Implementation target:** PyTorch plus `python-chess` for deterministic current-board fact extraction only. `python-chess` must not be used to evaluate engine scores, best moves, mate distances, or future positions.

This packet is intentionally written as a build handoff. It is a falsifiable research design, not a claim that the mechanism will beat stronger non-symbolic baselines before testing.

## 2. Executive Selection

Select **Bounded Board Hinge Logic**: a differentiable logic classifier that compiles a fixed library of typed, shallow PSL-style formulas into tensor operations over the current chess board.

The model receives only current-board tensors and deterministic board facts. It computes fuzzy truth values for bounded existential formulas such as "there exists an attacking relation between a side-to-move piece and a high-value enemy target near the king zone," but those formulas are implemented as differentiable relation-algebra contractions, not as move search, clause resolution, proof-tree construction, or generated tactical programs.

The classifier has one latent query atom per example:

```text
Puzzle(B) in [0, 1]
```

For each formula truth value `a_k(B) in [0, 1]`, BBHL creates two possible soft rule channels:

```text
a_k(B) -> Puzzle(B)
a_k(B) -> NotPuzzle(B)
```

Under PSL/Lukasiewicz hinge semantics, the energy difference between `Puzzle=1` and `Puzzle=0` becomes a differentiable logit. Training is ordinary binary cross-entropy on `puzzle_binary`, but the features feeding the logit are fuzzy logical formula satisfactions, not neural embeddings produced by message passing.

Why this is the best fit for the prompt:

1. It uses a precise differentiable logic object: PSL/HL-MRF semantics over a finite chess board domain.
2. It avoids proof trees entirely: formulas are evaluated by bounded tensor contractions and soft existential aggregation.
3. It allows attack geometry from the current board only.
4. It is interpretable enough to inspect learned formula families and rule weights.
5. It cannot silently become a generic GNN if the implementation forbids hidden node-state updates and unrestricted learned adjacency message passing.

## 3. Problem Restatement And Data Contract

### Problem

Given a legal or pseudo-legal chess position represented by its current board state, predict a binary target:

```text
y = puzzle_binary in {0, 1}
```

The intended meaning is only that the example is positive or negative according to the dataset's target field. The model must not infer or reconstruct puzzle status from source names, verification labels, engine annotations, best-move metadata, or puzzle-provider identity.

### Allowed inputs

The model may use:

- Current-board tensor.
- Side to move.
- Piece locations.
- Castling/en-passant fields only if present in the raw current-position representation and treated as board-state facts, not as future-move labels.
- Deterministic facts derived from the current board.
- Pseudo-legal attack geometry computed from the current board only.
- Static square geometry such as same rank, same file, same diagonal, knight displacement, king adjacency, board color, distance to edge, and king-zone masks.

### Forbidden inputs

The model and all preprocessing must exclude:

- Stockfish scores.
- Principal variations.
- Mate scores or mate distances.
- Node counts.
- Best moves.
- Verification labels.
- Source labels.
- Source file identity.
- Any hash, path, row number, provider tag, filename, or import batch identifier used as a model feature.
- Any after-move board tensor.
- Any one-ply move-delta feature.
- Any generated candidate move list used for pooling or scoring.
- Any engine, tablebase, proof-number, minimax, response-minimax, or search-derived signal.

### Data item contract

Codex should enforce this contract at the dataloader boundary:

```python
@dataclass(frozen=True)
class PuzzleBinaryItem:
    fen: str                         # current board only
    y: int                           # 0 or 1; only supervised target
    split_key: str                   # sha256(canonical_fen), used only for splitting/dedup
```

The model-facing batch must be stricter:

```python
@dataclass(frozen=True)
class ModelBatch:
    board_tensor: FloatTensor        # [B, 12, 8, 8]
    unary_facts: FloatTensor         # [B, 64, F]
    binary_relations: FloatTensor    # [B, R, 64, 64]
    y: FloatTensor                   # [B]
```

No other fields are allowed into `ModelBatch`. In particular, `fen`, `split_key`, source metadata, file paths, and row IDs must not be passed into the model.

### Split and deduplication

Use `split_key = sha256(canonical_fen_without_clocks)` for splitting and deduplication. The key is allowed only for data hygiene. It must never be embedded, tokenized, bucketed, or joined with labels inside the model pipeline.

Recommended split:

```text
train: hash_mod in [0, 79]
valid: hash_mod in [80, 89]
test:  hash_mod in [90, 99]
```

If duplicate positions have conflicting labels, either drop all conflicting duplicates or collapse them with a documented rule before splitting. Do not use source identity to resolve conflicts.

## 4. Differentiable Logic Research Map

BBHL is closest to probabilistic soft logic. PSL uses soft truth values in `[0, 1]`, Lukasiewicz-style logical operators, and weighted formulas as Markov-network features; the original HL-MRF/PSL paper frames these models as scalable structured prediction with convex inference. See Bach et al., **Hinge-Loss Markov Random Fields and Probabilistic Soft Logic**, JMLR 2017, and the PSL project introduction for the specific soft-logic operators and rule-weight framing: [JMLR paper](https://jmlr.org/beta/papers/v18/15-631.html), [PSL introduction](https://psl.linqs.org/wiki/Introduction-to-Probabilistic-Soft-Logic.html).

Logic Tensor Networks are also relevant because they place first-order formulas in real-valued semantics with truth values in `[0, 1]`; however, BBHL deliberately avoids neural predicate networks over arbitrary learned embeddings because that route can collapse into a generic neural architecture. See Serafini and d'Avila Garcez, **Logic Tensor Networks: Deep Learning and Logical Reasoning from Data and Knowledge**: [overview](https://axi.lims.ac.uk/paper/1606.04422).

TensorLog is relevant as a differentiable deductive database, but it is rejected for this prompt because it compiles clauses and query proof structure into differentiable functions. That is too close to the forbidden proof-tree/neural-clause-resolution family. See Cohen, **TensorLog: A Differentiable Deductive Database**: [DBLP record](https://dblp.org/rec/journals/corr/Cohen16b.html).

Semiring/provenance relation algebra is useful background because it shows how relational operations and Datalog-style evaluations can be parameterized by algebraic semirings. BBHL borrows the finite tensor-relation mindset, but not recursive Datalog evaluation. See Green, Karvounarakis, and Tannen, **Provenance Semirings**: [publication summary](https://www.sciweavers.org/publications/provenance-semirings).

Research-map decision:

| Candidate family | Useful property | Rejection or selection reason |
|---|---:|---|
| Probabilistic soft logic / HL-MRF | Weighted fuzzy rules, hinge penalties, continuous truth values | **Selected.** Single-query conditional energy gap is simple, differentiable, bounded, and interpretable. |
| Logic Tensor Networks | Real-valued first-order semantics | Rejected as primary mechanism because free neural predicates over embeddings are too easy to turn into a generic neural net. |
| Differentiable Datalog / TensorLog | Relational differentiability | Rejected because proof depth, clauses, and query derivations are too close to forbidden proof-tree mechanisms. |
| Semiring relation algebra | Clean tensorized relation composition | Used as an implementation style inside BBHL, but not selected alone because pure matrix-polynomial features are less rule-readable. |
| Generic relational GNN | Strong relational inductive bias | Rejected. Message passing over squares or pieces is not the requested differentiable logic object. |

## 5. Serious Candidate Rejections

### Rejection A: TensorLog-style differentiable Datalog

This is tempting because chess facts are relational. It is rejected because Datalog-style rules naturally produce derivation paths. Even if implemented differentiably, query answers are usually explained by proof chains, proof depth, or clause recursion. The prompt explicitly asks for differentiable logic without proof trees and forbids neural clause resolution.

### Rejection B: Fuzzy description logic over square concepts only

A fuzzy DL concept hierarchy could define concepts such as `LoosePiece`, `KingZone`, or `OverloadedDefender`. The problem is that chess tactics are highly relational: attacks, ray-blocking, between-square relations, and same-line geometry matter. A pure concept lattice either becomes too weak or quietly reintroduces role chains that act like proof paths. BBHL keeps roles, but bounds them to fixed-depth tensor relations.

### Rejection C: Logic Tensor Network with learned square embeddings

An LTN with learned embeddings for squares and pieces would be easy to code, but the neural predicate modules could absorb the entire classification problem. That would satisfy the letter of differentiable logic while violating the spirit of the prompt. BBHL therefore uses observed board facts and low-capacity soft predicate mixtures, not unconstrained MLP predicates over learned node states.

### Rejection D: Pure tensorized relation algebra classifier

A semiring relation-algebra model can evaluate expressions such as `OwnPiece ; Attacks ; EnemyKingZone`. This is elegant, but a large bank of learned matrix products can become an opaque relational feature machine. BBHL uses relation algebra only to evaluate named PSL formula templates with explicit positive/negative rule channels.

### Rejection E: Differentiable tactical motif induction

Learning a catalog of named motifs like fork, pin, skewer, mate net, discovered attack, and overload would be attractive. It is rejected because it drifts into tactical program induction and tactical subgoal automata. BBHL may learn formula weights that correlate with such motifs, but it must not name, verify, or construct tactical programs.

## 6. Common Approach Rejections

Do not implement any of the following as the selected model or as hidden preprocessing:

1. **Neural clause resolution.** No learned Horn-clause resolver, no differentiable backward chaining, no unrolled proof search.
2. **Tactical program induction.** No induced fork/pin/mate program library, no learned tactic grammar, no symbolic tactic executor.
3. **Tactical subgoal automata.** No state machine over tactical goals, no latent sequence of tactical states.
4. **Proof-core verifier.** No verifier that tries to validate a tactic, mate, or proof core.
5. **Proof-number search.** No proof/disproof tree, even if shallow.
6. **Response-minimax.** No opponent-response search, no minimax rollout, no engine-like evaluator.
7. **One-ply move-delta pooling.** No legal move generation followed by before/after feature differences, even if no engine is called.
8. **Engine search.** No Stockfish, Leela, tablebase, Syzygy, cloud engine, node counts, principal variation, best move, centipawn, WDL, or mate score.
9. **Generic GNN.** No square-node or piece-node message-passing network with hidden states. Relation tensors may be used only for fixed-depth logic formula evaluation.
10. **Source leakage.** No source label, file identity, row order, path, import batch, provider, puzzle collection name, or verification status.

## 7. Mathematical Thesis

### Thesis

For chess `puzzle_binary` classification, a finite closed-world PSL theory over current-board facts can learn a useful decision boundary by measuring the soft satisfaction of bounded attack-and-geometry formulas. Because the theory has only one random query atom, `Puzzle(B)`, the classifier reduces to a differentiable energy gap between `Puzzle(B)=1` and `Puzzle(B)=0`, without proof trees or search.

### Formal setup

Let `B` be a current board. Let `D = {0, ..., 63}` be the finite square domain. Let `F(B)` be the set of observed unary and binary predicates derived deterministically from `B`.

For each formula template `phi_k`, BBHL computes a grounded fuzzy truth value:

```text
a_k(B; theta) = Truth_BBHL(phi_k, F(B); theta) in [0, 1]
```

where `theta` contains low-capacity parameters for soft predicate mixtures and temperature parameters for smooth existential aggregation.

For the target atom `P = Puzzle(B)`, define two PSL-style rule families:

```text
phi_k(B) -> P
phi_k(B) -> not P
```

Let `lambda_k_pos >= 0` and `lambda_k_neg >= 0` be learned nonnegative rule weights. For binary `y in {0,1}`, the rule distances are:

```text
D_pos_k(y) = max(a_k(B) - y, 0)^p
D_neg_k(y) = max(a_k(B) - (1 - y), 0)^p
```

with `p in {1, 2}`. For binary `y`, this simplifies:

```text
E(y=0 | B) = sum_k lambda_k_pos * a_k(B)^p
E(y=1 | B) = sum_k lambda_k_neg * a_k(B)^p
```

The conditional logit is the energy gap:

```text
logit(B) = beta + tau * (E(0 | B) - E(1 | B))
         = beta + tau * sum_k (lambda_k_pos - lambda_k_neg) * a_k(B)^p
```

and

```text
Pr(y=1 | B) = sigmoid(logit(B)).
```

This is not merely logistic regression on hand-coded features because the `a_k` values are differentiable fuzzy formula evaluations with learned predicate mixtures, bounded quantifiers, soft conjunctions, soft disjunctions, and typed relation compositions. It is also not a GNN because no node state is iteratively updated and no learned message passing occurs.

### Differentiability

Use smoothed versions of `max`, `min`, and existential aggregation where needed:

```text
softmax_tau(x_1, ..., x_n) = sum_i x_i * softmax(tau * x)_i
soft_or_tau(x_1, ..., x_n) = 1 - product_i (1 - clamp(x_i, eps, 1 - eps))
softplus_hinge(x) = softplus(alpha * x) / alpha
```

The hard Lukasiewicz definitions may also be used with subgradients, but the minimal implementation should start with smooth hinges for stable training.

## 8. Logic Object And Semantics

### Logic object

BBHL defines a finite, typed PSL theory:

```text
T_BBHL = (D, U, R, Phi, Lambda)
```

where:

- `D` is the finite square domain of size 64.
- `U` is a set of observed unary predicates over squares.
- `R` is a set of observed binary predicates over square pairs.
- `Phi` is a bounded set of shallow formula templates.
- `Lambda` is a set of nonnegative rule weights for positive and negative target implication rules.

The only unobserved atom is `Puzzle(B)`. All square and relation facts are observed from the current board.

### Primitive unary predicates

Each unary predicate is a tensor `[B, 64]`. Suggested primitives:

```text
stm_piece(s)              # side-to-move piece occupies square s
enemy_piece(s)            # opponent piece occupies s
empty(s)
stm_king(s)
enemy_king(s)
stm_pawn(s), stm_knight(s), ..., stm_queen(s)
enemy_pawn(s), enemy_knight(s), ..., enemy_queen(s)
stm_slider(s)             # bishop, rook, queen
enemy_slider(s)
center_square(s)
edge_square(s)
corner_square(s)
light_square(s)
dark_square(s)
enemy_king_zone(s)         # static king neighborhood around current enemy king
stm_king_zone(s)
attacked_by_stm_any(s)     # pseudo-legal current-board attack fact
attacked_by_enemy_any(s)
defended_by_stm_any(s)
defended_by_enemy_any(s)
occupied_and_attacked_by_stm(s)
occupied_and_attacked_by_enemy(s)
```

Do not compute these predicates by making moves. Attack maps are pseudo-legal attack geometry from the current board only.

### Primitive binary relations

Each binary relation is a tensor `[B, 64, 64]` or `[64, 64]` if board-independent:

```text
same_rank(s, t)
same_file(s, t)
same_diag(s, t)
knight_step(s, t)
king_step(s, t)
rays_align(s, t)              # same rank/file/diagonal
between_occupied_count_0(s,t) # unobstructed current-board ray, if aligned
between_occupied_count_1(s,t) # exactly one current-board blocker, if aligned
stm_attacks(s, t)             # piece on s attacks t pseudo-legally
enemy_attacks(s, t)
stm_ray_attacks(s, t)
enemy_ray_attacks(s, t)
stm_knight_attacks(s, t)
enemy_knight_attacks(s, t)
stm_pawn_attacks(s, t)
enemy_pawn_attacks(s, t)
near_enemy_king(s, t)         # t is enemy king or king-zone square
near_stm_king(s, t)
```

`between_occupied_count_0` and `between_occupied_count_1` are legal because they are deterministic facts from the current board. They must not encode a move, a response, a best line, or an after-move board.

### Learnable fuzzy predicates

BBHL can learn low-capacity mixtures of primitive predicates:

```text
C_m(s) = sum_j softmax(theta_C[m, :])_j * U_j(s)
Q_n(s,t) = sum_r softmax(theta_Q[n, :])_r * R_r(s,t)
```

These mixtures preserve interpretability: each learned concept or role is a convex combination of named board facts. Do not replace them with arbitrary MLPs over learned square embeddings.

### Fuzzy operators

Use Lukasiewicz-compatible semantics:

```text
not A       = 1 - A
A and B     = max(A + B - 1, 0)
A or B      = min(A + B, 1)
A -> B      = min(1, 1 - A + B)
dist(A -> B)= max(A - B, 0)
```

For `n`-ary conjunction:

```text
AND_L(x_1, ..., x_n) = max(sum_i x_i - (n - 1), 0)
```

For existential quantification over a finite domain, use a soft maximum or noisy-or:

```text
exists_s f(s) = softmax_pool_tau({f(s): s in D})
```

The first implementation should use temperature-controlled softmax pooling because it is stable and easy to audit.

### Formula families

Keep all formula templates bounded to depth at most two binary relations. Suggested families:

```text
F1: exists s. C_a(s)
F2: exists s,t. C_a(s) and Q_b(s,t) and C_c(t)
F3: exists s,t,u. C_a(s) and Q_b(s,t) and Q_c(t,u) and C_d(u)
F4: exists s,t. C_a(s) and Q_b(s,t) and C_c(t) and NearEnemyKing(t)
F5: exists s,t. C_a(s) and AttackRole(s,t) and EnemyValuable(t)
F6: exists s,t,u. C_a(s) and RayRole(s,t) and BlockerRole(t,u) and EnemyKingZone(u)
```

These are not tactical programs. They are shallow relation patterns over observed board facts. No formula may call another formula recursively. No formula may generate a move. No formula may verify a tactic.

## 9. Architecture Tensor Contract

### Input tensors

```text
board_tensor:      float32 [B, 12, 8, 8]
unary_facts:       float32 [B, 64, F]
binary_relations:  float32 [B, R, 64, 64]
y:                 float32 [B]
```

`board_tensor` is useful for auditing and optional baselines. The selected BBHL model should operate primarily on `unary_facts` and `binary_relations`.

### Canonicalization

Canonicalize positions to side-to-move perspective if desired:

- If black is to move, flip the board vertically and swap colors so the side to move is always treated as `stm`.
- Preserve enough orientation to compute pawn attacks correctly after canonicalization.
- Unit-test canonicalization with known white-to-move and black-to-move mirror positions.

This is allowed because it is a deterministic transform of the current board.

### Model modules

Recommended code-level module boundaries:

```text
FactExtractor
  FEN -> board_tensor, unary_facts, binary_relations

FuzzyPredicateBank
  unary_facts, binary_relations -> concept_bank, role_bank

BoundedFormulaEvaluator
  concept_bank, role_bank -> formula_truths [B, K]

PSLEnergyGapHead
  formula_truths -> logits [B]
```

### Tensor formulas

For binary formulas:

```python
# C_left:  [B, M, 64]
# Q:       [B, N, 64, 64]
# C_right: [B, M, 64]
# output:  [B, K]

truth_s_t = lukasiewicz_and(C_left[:, i, :, None], Q[:, j, :, :], C_right[:, k, None, :])
a_formula = soft_exists_2d(truth_s_t, tau=exists_tau)
```

For ternary formulas, avoid materializing `[B,64,64,64]` when possible. Use relation composition with bounded top-k or chunking:

```python
left_pair = lukasiewicz_and(C_a[:, :, None], Q_b)
right_pair = lukasiewicz_and(Q_c, C_d[:, None, :])
# combine through t by max/softmax over t, then over s,u
```

Maximum formula depth is two relations. If memory becomes an issue, disable ternary family `F3` in the minimal experiment.

### PSL head

```python
class PSLEnergyGapHead(nn.Module):
    def __init__(self, num_formulas: int):
        super().__init__()
        self.pos_raw = nn.Parameter(torch.zeros(num_formulas))
        self.neg_raw = nn.Parameter(torch.zeros(num_formulas))
        self.bias = nn.Parameter(torch.zeros(()))
        self.log_tau = nn.Parameter(torch.zeros(()))

    def forward(self, a):
        # a: [B, K], clamped to [0, 1]
        pos = F.softplus(self.pos_raw)
        neg = F.softplus(self.neg_raw)
        tau = F.softplus(self.log_tau) + 1e-4
        return self.bias + tau * (a.pow(2) @ (pos - neg))
```

This is the closed-form energy-gap version of the one-query PSL theory.

### Guardrail: not a GNN

The implementation must not contain:

```text
node_state = f(node_state, aggregate(neighbor_state))
message_passing_layers
GraphConv
GATConv
edge-conditioned hidden-state update
learned square embeddings updated over relation edges
```

Relation tensors are used only to evaluate bounded formulas.

## 10. Training Objective

### Primary loss

Use class-balanced binary cross-entropy:

```text
L_bce = BCEWithLogitsLoss(pos_weight = n_negative / n_positive)
```

or focal loss only if the class imbalance is extreme. Start with BCE first.

### Regularization

Use the following regularizers:

```text
L_weight_l1      = alpha_l1 * (sum_k pos_k + sum_k neg_k)
L_overlap        = alpha_overlap * sum_k min(pos_k, neg_k)
L_entropy_concept= alpha_ent * mean entropy of predicate-mixture softmaxes
L_temp           = alpha_temp * (exists_tau - tau_target)^2
```

Purpose:

- `L_weight_l1` encourages a small number of active formulas.
- `L_overlap` discourages the same formula from voting strongly both positive and negative.
- `L_entropy_concept` can be either positive or negative depending on experiment: positive entropy keeps mixtures broad early; negative entropy makes learned concepts more discrete late.
- `L_temp` prevents existential pooling from becoming a hard max too early.

### Calibration metrics

Track:

```text
AUROC
Average Precision
Brier score
Expected Calibration Error
Balanced accuracy at validation-selected threshold
Log loss
```

Do not select a threshold on the test set.

### Optimization

Recommended defaults:

```text
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 1e-4
batch_size: 128 or 256
max_epochs: 50
early_stopping: validation AUROC patience 8
mixed_precision: optional after tests pass
```

### Training invariants

Every training run must log:

```text
number of examples after dedup
class balance by split
fact tensor shape
relation tensor names
formula family counts
banned-field audit result
label-shuffle control result if run
random-relation control result if run
```

## 11. Ablations And Randomized Controls

### Required ablations

1. **No attack relations.** Remove `stm_attacks`, `enemy_attacks`, and piece-specific attack relations. Static geometry remains. If performance barely changes, the attack-logic thesis is weak.
2. **No ray/blocker facts.** Remove ray alignment and between-occupied-count relations. If performance improves, those facts may be noisy or leaking unintended information.
3. **Unary-only formulas.** Keep `F1` and disable binary/ternary formulas. This tests whether relational logic actually matters.
4. **Binary-only formulas.** Use `F2`, `F4`, and `F5`, disable ternary formulas. This is the likely best minimal model.
5. **Frozen predicate mixtures.** Replace learned mixtures with one-hot primitive facts. This tests whether learning concept/role mixtures is necessary.
6. **PSL head versus MLP head.** Feed the same `formula_truths` into both the PSL energy-gap head and a small MLP. The PSL head should be competitive; if the MLP dominates, formula evaluation may be useful but the PSL semantics may not be.
7. **Board tensor CNN baseline.** Small CNN on `[12,8,8]`, no engine and no source fields. This is a sanity baseline, not the selected mechanism.
8. **Logistic raw-fact baseline.** Mean/max pooled primitive facts plus logistic regression. BBHL should beat this to justify formula logic.

### Randomized controls

1. **Label shuffle.** Shuffle `y` within split and train the full pipeline. Test AUROC should be near 0.50. Anything materially higher implies leakage.
2. **Relation scramble.** Permute square indices in binary relations independently from unary facts. Performance should fall. If it does not, relation logic is not being used.
3. **Attack-role replacement.** Replace attack relations with degree-matched random binary matrices. This should underperform real attack relations.
4. **Color-perspective randomization.** Randomly swap side-to-move perspective in a controlled way and verify labels are not predictable from color artifacts.
5. **Formula-index permutation.** Randomly permute formula outputs before the head at evaluation time. A trained model should degrade unless the head collapsed to a constant.
6. **Symmetry consistency.** Apply legal board symmetries that preserve chess meaning under the chosen canonicalization policy. Predictions should be consistent within a small tolerance.

### Leakage controls

1. Verify `ModelBatch` has exactly four fields: `board_tensor`, `unary_facts`, `binary_relations`, `y`.
2. Search code for banned strings: `stockfish`, `engine`, `pv`, `mate`, `score`, `best_move`, `nodes`, `source`, `filename`, `path`, `verification`, `tablebase`, `syzygy`.
3. Fail CI if any banned string appears outside a documented denylist test.
4. Assert no dataloader returns source metadata.
5. Assert no model module accepts `fen`, `id`, `split_key`, or raw text.

## 12. Minimal Experiment

### Goal

Test whether BBHL can beat non-logic current-board baselines on `puzzle_binary` without using forbidden information.

### Minimal implementation

Use only formula families `F1`, `F2`, and `F4` at first. Disable ternary formulas until memory and correctness are stable.

Recommended formula counts:

```text
num_concepts: 24
num_roles: 16
num_unary_formulas: 24
num_binary_formulas: 96
num_kingzone_formulas: 48
total K: 168
```

Implementation layout:

```text
src/
  data/
    dataset.py              # reads fen,y only
    split.py                # hash split, dedup
  chesslogic/
    facts.py                # deterministic fact extraction
    geometry.py             # static square relations
    attacks.py              # pseudo-legal current-board attack tensors
  models/
    predicate_bank.py       # convex mixtures of primitive predicates
    fuzzy_ops.py            # Lukasiewicz ops and soft exists
    formula_eval.py         # bounded formula families
    psl_head.py             # energy-gap classifier
    bbhl.py                 # full model wrapper
  train.py
  evaluate.py
  audits.py
tests/
  test_no_banned_fields.py
  test_no_engine_calls.py
  test_fact_shapes.py
  test_canonicalization.py
  test_label_shuffle_control.py
```

### Minimal run matrix

Run these experiments with three seeds each:

```text
E0: logistic raw-fact baseline
E1: small CNN board baseline
E2: BBHL unary-only
E3: BBHL binary formulas only
E4: BBHL binary formulas + king-zone formulas
E5: E4 with relation scramble control
E6: E4 with label shuffle control
E7: E4 with attack relations removed
```

### Expected result pattern

BBHL is promising if:

```text
AUROC(E4) > AUROC(E0) by at least 0.02
AUROC(E4) >= AUROC(E1) or is close with better calibration/interpretablity
AUROC(E5) drops materially versus E4
AUROC(E6) is approximately 0.50
AUROC(E7) drops versus E4
```

Do not tune to the test set. Use validation AUROC for early stopping and model selection.

### Inspection output

After training, print the top formulas by absolute contribution:

```text
formula_id
family
positive_weight
negative_weight
net_weight
top unary primitive mixture terms
top role primitive mixture terms
mean truth on positives
mean truth on negatives
```

This inspection must not use source labels or verification labels. It is only a model-debugging view over learned formula parameters and dataset labels.

## 13. Falsification Criteria

BBHL should be considered falsified, or at least not supported, if any of these occur:

1. **Leakage failure:** label-shuffle AUROC is materially above 0.55 on test.
2. **Relation irrelevance:** relation-scramble control is within 0.01 AUROC of the real model.
3. **Attack irrelevance:** removing attack relations does not reduce performance and learned formula inspection shows no meaningful use of attack or king-zone relations.
4. **Baseline failure:** BBHL cannot beat logistic raw-fact baseline by at least 0.02 AUROC after reasonable tuning.
5. **Generic-neural dependence:** replacing the PSL head with an MLP is the only way to get competitive performance, and PSL rule weights remain unused or near zero.
6. **Formula collapse:** all high-weight formulas reduce to broad occupancy or material-count proxies, with no relational selectivity.
7. **Calibration failure:** BBHL improves AUROC but has worse Brier score and ECE than simpler baselines with no interpretability gain.
8. **Symmetry failure:** mirrored/canonical-equivalent boards produce inconsistent predictions beyond a documented tolerance.
9. **Banned-signal discovery:** any run uses engine output, best moves, mate labels, source labels, source file identity, verification labels, node counts, PVs, or after-move tensors. Such a run is invalid, not merely weak.
10. **Data-contract breach:** any model-facing tensor includes raw FEN text, row ID, file path, source key, or split key.

The project should stop or pivot if the falsification criteria hold after confirming implementation correctness.

## 14. Codex Implementation Notes

### First coding instruction

Start by implementing the audits, not the model. The easiest way to ruin this experiment is to allow a forbidden feature into the batch and get a fake win.

### `facts.py` requirements

Implement:

```python
def extract_model_facts(fen: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return board_tensor [12,8,8], unary_facts [64,F], binary_relations [R,64,64].

    Must use current board only.
    Must not call an engine.
    Must not generate after-move boards.
    Must not return source metadata.
    """
```

Use `python-chess` only for board parsing and current-board attack masks. Calling `board.attacks(square)` is allowed because it returns current-board attack geometry for the piece on `square`. Calling engine APIs, legal move evaluation, or push/pop to create future-board features is not allowed.

### Banned implementation patterns

Reject code like this:

```python
for move in board.legal_moves:
    board.push(move)
    features.append(extract_features(board))
    board.pop()
```

Reject code like this:

```python
engine.analyse(board, chess.engine.Limit(depth=...))
```

Reject code like this:

```python
x = torch.cat([logic_features, source_onehot], dim=-1)
```

Reject code like this:

```python
node_embeddings = graph_conv(node_embeddings, edge_index)
```

### `fuzzy_ops.py`

Implement numerically safe operators:

```python
def l_and(*xs):
    return torch.clamp(sum(xs) - (len(xs) - 1), min=0.0, max=1.0)

def l_or(a, b):
    return torch.clamp(a + b, min=0.0, max=1.0)

def l_not(a):
    return 1.0 - a

def soft_exists(x, dim, tau=12.0):
    w = torch.softmax(tau * x, dim=dim)
    return (w * x).sum(dim=dim)
```

Start with hard `clamp` Lukasiewicz ops. If gradients are too sparse, replace hinges with smoothed versions in one controlled ablation.

### `predicate_bank.py`

Use convex mixtures only:

```python
class FuzzyPredicateBank(nn.Module):
    def __init__(self, num_unary, num_binary, num_concepts=24, num_roles=16):
        super().__init__()
        self.concept_logits = nn.Parameter(torch.zeros(num_concepts, num_unary))
        self.role_logits = nn.Parameter(torch.zeros(num_roles, num_binary))

    def forward(self, U, R):
        # U: [B,64,F], R: [B,R,64,64]
        c_mix = torch.softmax(self.concept_logits, dim=-1)
        r_mix = torch.softmax(self.role_logits, dim=-1)
        C = torch.einsum('bsf,mf->bms', U, c_mix)
        Q = torch.einsum('brst,nr->bnst', R, r_mix)
        return C.clamp(0, 1), Q.clamp(0, 1)
```

No learned square embeddings. No MLP predicate networks in the selected model.

### `formula_eval.py`

Represent formulas as a small table of typed slots:

```python
@dataclass(frozen=True)
class BinaryFormulaSpec:
    left_concept: int
    role: int
    right_concept: int
    family: str
```

For formula selection, either instantiate a fixed deterministic grid or learn a small set of formula slots. If learning slots, use softmax mixtures over concept and role banks, not hard architecture search.

### Top formula reporting

Codex should add a function:

```python
def explain_top_formulas(model, fact_names, relation_names, top_n=20) -> list[dict]:
    ...
```

Each returned row should include human-readable primitive mixtures. This is not a proof explanation. It is parameter inspection.

### Test priorities

The first tests to write:

```text
test_model_batch_has_no_metadata
test_no_engine_imports_or_calls
test_no_push_pop_feature_extraction
test_attack_tensors_match_python_chess_current_attacks
test_canonicalization_round_trip
test_formula_truths_in_unit_interval
test_psl_head_nonnegative_weights
test_label_shuffle_auc_near_random
```

### Minimal acceptance target

A first successful implementation is one that:

1. Trains without banned inputs.
2. Produces formula truth tensors in `[0,1]`.
3. Passes all leakage tests.
4. Runs E0 through E7.
5. Produces a top-formula inspection table.
6. Gives an honest result even if BBHL underperforms.

## 15. Prompt-Maintenance Notes

Keep future prompts strict. The important phrase is:

```text
Use current-board deterministic facts only; no move generation for feature pooling, no engine outputs, no source/provenance labels, no proof trees, and no generic GNN hidden-state message passing.
```

When modifying this research direction, preserve these distinctions:

- **Allowed:** current-board pseudo-legal attack geometry.
- **Not allowed:** making a move and measuring what changes.
- **Allowed:** fixed-depth relation formula evaluation.
- **Not allowed:** recursive clause inference or proof-path enumeration.
- **Allowed:** PSL-style energy gap for `Puzzle(B)`.
- **Not allowed:** verifier labels, mate labels, or best-move supervision.
- **Allowed:** deterministic hash split key outside the model.
- **Not allowed:** source identity, file identity, path, row index, or verification status.

Suggested future prompt if this packet is reused:

```text
Implement Bounded Board Hinge Logic for chess puzzle_binary classification.
Use a finite PSL/HL-MRF energy-gap classifier over current-board board facts.
The only target atom is Puzzle(B). Use deterministic current-board unary facts,
binary square relations, and pseudo-legal attack geometry. Do not use engine
outputs, best moves, mate scores, PVs, node counts, source labels, source file
identity, verification labels, after-move tensors, one-ply move-delta pooling,
proof trees, clause resolution, tactical program induction, tactical subgoal
automata, proof-number search, response-minimax, or generic GNN message passing.
Start with audits, then fact extraction, then bounded formula evaluation, then
PSL energy-gap training and randomized controls.
```

The research bet is narrow and testable: if a shallow differentiable logic theory over current-board attack geometry contains enough signal, BBHL should show a real gain over raw current-board baselines while failing the randomized leakage controls in the right direction. If it does not, the result is still useful because it cleanly rejects this differentiable-logic route without contaminating the experiment with engine or source leakage.
