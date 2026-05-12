# Codex Handoff Packet: Traced Threat Motif Network

## 1. File Metadata

- **Filename:** `chess_nn_research_2026-04-28_0857_tuesday_new_york_trace_motif.md`
- **Created:** 2026-04-28 08:57, Tuesday, new_york
- **Artifact type:** Codex-ready Markdown handoff packet
- **Selected idea:** Traced Threat Motif Network, abbreviated **TTMN**
- **Primary output contract:** one scalar puzzle logit per current-board tensor
- **Label contract:** fine labels `0` and `1` map to binary target `0`; fine label `2` maps to binary target `1`
- **Forbidden-input guardrail:** no engine evaluations, principal variations, node counts, mate scores, best moves, source labels, verification metadata, or any future-position search products

## 2. Executive Selection

Select **Traced Threat Motif Network**.

The model is a concrete PyTorch architecture whose categorical content is not decorative. It maps typed, current-board chess relations into nonnegative square-to-square matrices and composes them with `torch.bmm`. Short composed paths become tactical motif tensors. Closed motifs are scored with a categorical trace surrogate, implemented as a diagonal sum. Parallel attacker/defender structure is represented by a block-diagonal monoidal product, implemented efficiently by separate channel stacks plus additive trace features. A pullback-like contest map is implemented by multiplying two target-square marginals that land on the same square.

This is the selected idea because it gives Codex exact tensors, masks, matrix products, traces, and diagnostics while staying inside the data contract. It does not ask for a chess engine, legal-move search, best-move labels, or verification metadata.

## 3. Problem And Data Contract

### Task

Binary chess puzzle classification from the **current board only**.

```python
target = (fine_label == 2).float()  # fine 0/1 -> 0, fine 2 -> 1
```

The training target is a single binary value. Fine label `1` is still a negative example, but it is retained during evaluation for near-puzzle diagnostics.

### Input

```python
x: FloatTensor[B, C, 8, 8]
```

Minimum required channels:

| Channel range | Meaning |
|---:|---|
| `0..5` | White `P,N,B,R,Q,K` occupancy planes |
| `6..11` | Black `P,N,B,R,Q,K` occupancy planes |
| `12` | Side-to-move plane, `1.0` for white to move and `0.0` for black to move |
| `13..C-1` | Optional current-state planes only, such as castling rights or en-passant square, if already present in the board tensor |

The architecture may accept more current-board channels through the convolutional stem, but only the first 13 channels are used by the relation-matrix constructor.

### Output

```python
logit: FloatTensor[B]
```

Optional diagnostics are returned in a separate dictionary and must not alter the one-logit contract:

```python
logit, diag = model(x, return_diag=True)
```

### Forbidden inputs

Do not pass any of the following into the dataset item, model forward pass, feature builder, sampler, loss, validation loop, or diagnostics:

- engine evaluations
- principal variations
- node counts
- mate scores
- best moves
- source labels
- verification metadata
- generated continuations from search
- puzzle tags that reveal construction or source process

Near-puzzle diagnostics may use `fine_label` **after** prediction to stratify reports, never as a model input.

## 4. Research Map With Citations

### Sources that directly justify the selected design

- **Categorical trace:** Joyal, Street, and Verity introduced traced monoidal categories and the trace operation; TTMN implements the trace concretely as `sum(diagonal(A @ B @ C))` over square-indexed matrices. [R1]
- **Compositional wiring / operads:** Applied category theory treats systems as boxes wired into larger boxes; TTMN uses this only as a coding discipline for matrix composition of motif boxes, not as abstract vocabulary. [R2], [R3]
- **Tensor-network precedent:** Supervised tensor-network models show that classification can be built from structured products of smaller multilinear maps; TTMN uses short matrix-product motif chains rather than an unrestricted CNN-only head. [R4]
- **Multiplicative relation scoring:** Neural tensor networks motivate learned multiplicative interactions between entities and relation types; TTMN uses learned source/target gates over square-to-square relation matrices. [R5]
- **Invariant aggregation:** Deep Sets motivates permutation-invariant aggregation over unordered motif banks; TTMN aggregates a fixed set of motif features with shared MLP layers and summary statistics. [R6]
- **Board-only neural precedent:** AlphaZero and NNUE demonstrate that board-state neural features are viable in chess, but TTMN deliberately avoids their search/evaluation targets and uses only current-board tensor features. [R7], [R8]
- **Binary-logit training:** PyTorch `BCEWithLogitsLoss` combines sigmoid and binary cross-entropy in a numerically stable form, matching the one-logit target contract. [R9]

### References

- **[R1]** Joyal, A.; Street, R.; Verity, D. “Traced monoidal categories.” *Mathematical Proceedings of the Cambridge Philosophical Society*, 119(3), 447–468, 1996. Cambridge Core: <https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/traced-monoidal-categories/2BE85628D269D9FABAB41B6364E117C8>
- **[R2]** Fong, B.; Spivak, D. *Seven Sketches in Compositionality: An Invitation to Applied Category Theory.* LibreTexts / MIT OCW mirror: <https://math.libretexts.org/Bookshelves/Applied_Mathematics/Seven_Sketches_in_Compositionality%3A_An_Invitation_to_Applied_Category_Theory_%28Fong_and_Spivak%29>
- **[R3]** Yau, D. *Operads of Wiring Diagrams.* Springer Lecture Notes in Mathematics 2192, 2018. <https://link.springer.com/book/10.1007/978-3-319-95001-3>
- **[R4]** Stoudenmire, E. M.; Schwab, D. J. “Supervised Learning with Tensor Networks.” NeurIPS 2016. <https://papers.nips.cc/paper/6211-supervised-learning-with-tensor-networks>
- **[R5]** Socher, R.; Chen, D.; Manning, C. D.; Ng, A. “Reasoning With Neural Tensor Networks for Knowledge Base Completion.” NeurIPS 2013. <https://papers.nips.cc/paper/5028-reasoning-with-neural-tensor-networks-for-knowledge-base-completion>
- **[R6]** Zaheer, M.; Kottur, S.; Ravanbakhsh, S.; Poczos, B.; Salakhutdinov, R.; Smola, A. “Deep Sets.” NeurIPS 2017. <https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html>
- **[R7]** Google DeepMind. “AlphaZero: Shedding new light on chess, shogi, and Go.” 2018. <https://deepmind.google/blog/alphazero-shedding-new-light-grand-games-chess-shogi-and-go/>
- **[R8]** Stockfish official NNUE PyTorch wiki. “NNUE.” <https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html>
- **[R9]** PyTorch documentation. `torch.nn.BCEWithLogitsLoss`. <https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html>

## 5. Candidate Search Trace

| Candidate | Concrete tensor object | Decision | Reason |
|---|---|---|---|
| Plain residual CNN with category-theory naming | Convolutions only | Reject | Strong baseline, but no concrete compositional operator beyond ordinary neural layers. |
| Functorial attack-matrix classifier | Typed square-to-square matrices | Keep as core | Directly implementable from board occupancy and learned gates. |
| Pullback-only contest heatmap | Product of attacker and defender target marginals | Reject as standalone | Useful diagnostic, but too shallow to model tactical continuation. |
| Operadic motif bank | List of typed matrix words with `bmm` composition | Keep | Gives exact composition semantics and easy ablations. |
| Categorical trace scorer | Diagonal sum of composed motif matrices | Keep | Concrete, differentiable, cheap, and aligned with recurrent tactical loops. |
| Search-based tactical verification | Move generation plus engine or PV labels | Reject | Violates forbidden-input rule and changes the task. |

Final choice: **attack-matrix functor + operadic motif bank + categorical trace + contest pullback surrogate**, all implemented as PyTorch tensors.

## 6. Rejected Common Approaches

- **Search-assisted classifiers:** reject any feature derived from engine output, forced lines, PVs, mate scores, node counts, or best moves.
- **Generic graph neural networks:** reject iterative neighbor-message passing over pieces or squares. TTMN uses fixed square-to-square matrices and short matrix products, not a generic message-passing stack.
- **Abstract-only category theory:** reject any proposal that cannot point to exact tensors and operations.
- **Tabular relational-composition layers:** reject database-like set operations over relations. TTMN uses dense or sparse tensor multiplication and Hadamard products only.
- **Program-induction puzzle solvers:** reject learned symbolic programs, move-sequence generators, or inferred proof scripts.
- **Automaton-style state machines:** reject hidden state transitions over move sequences. TTMN is a single current-board forward pass.
- **Sheaf-style consistency models:** reject because the requested architecture can be built more directly with matrix composition and trace.

## 7. Mathematical Thesis

A chess tactic is often visible before search as a compact pattern of composable square relations: control, capture pressure, quiet reach, and contested target squares. A board is puzzle-like when short compositions of these relations form high-mass open paths into valuable targets and high-mass closed loops around contested tactical regions.

TTMN turns that thesis into tensors:

1. Primitive current-board relations become matrices `A_r[b] in R^{64 x 64}`.
2. Composition of relations becomes matrix multiplication:

   ```text
   A_(r1 ; r2 ; r3) = A_r1 @ A_r2 @ A_r3
   ```

3. Closed tactical feedback becomes a trace:

   ```text
   trace(A_word) = sum_i A_word[i, i]
   ```

4. Parallel attacker/defender structure becomes a monoidal product via block diagonal combination:

   ```text
   A_us ⊕ A_them = [[A_us, 0], [0, A_them]]
   trace(A_us ⊕ A_them) = trace(A_us) + trace(A_them)
   ```

5. Shared target-square tension becomes a pullback surrogate:

   ```text
   contest[j] = incoming_us_control[j] * incoming_them_control[j]
   ```

This is not a proof that tactics are category theory. It is a concrete inductive bias: puzzle positions should activate composed short-path motifs more strongly than ordinary non-puzzle positions, while near-puzzle negatives should activate some motifs but fail target-specific or closure-specific tests.

## 8. Concrete Compositional Object

### Objects

The object set is the 64 board squares. Each square index is

```python
sq = rank * 8 + file  # 0..63
```

Optional boundary vectors are current-board square distributions, not new objects:

- `u_piece`: side-to-move piece source mass, shape `[B, 64]`
- `v_enemy_king`: opponent king one-hot, shape `[B, 64]`
- `v_enemy_value`: learned value-weighted opponent-piece mass, shape `[B, 64]`

### Primitive morphisms

Each primitive relation `r` maps source square to target square:

```python
A_raw[:, r, i, j] >= 0
```

Raw relation types:

```text
color in {white, black}
piece in {P, N, B, R, Q, K}
role  in {ctrl, hit, quiet}
```

So `K_raw = 2 * 6 * 3 = 36` relation matrices.

- `ctrl`: source piece controls target square by chess geometry.
- `hit`: source piece controls target square and target contains an enemy piece.
- `quiet`: source piece can pseudo-move to an empty target square by current-board geometry. For pawns, this uses forward pawn moves; for other pieces, it uses ordinary movement geometry to an empty square.

Sliding pieces use exact current-board line clearance from precomputed between-square masks.

### Functorial mapping into tensors

Define a parameterized map `F_theta`:

```text
F_theta(square i) = H[:, i, :]                in R^d
F_theta(raw relation r) = A_raw[:, r, :, :]   in R^{64 x 64}
F_theta(r ; s) = F_theta(r) @ F_theta(s)
F_theta(r ⊕ s) = block_diag(F_theta(r), F_theta(s))
Tr(F_theta(word)) = diagonal_sum(product matrices in word)
```

This is the concrete categorical content. If a future variant cannot write the above as PyTorch tensor code, reject it.

### Operad composition surrogate

A motif is a typed word over grouped matrices:

```python
word = ("u_quiet", "u_ctrl", "u_hit")
```

The operation of inserting one motif after another is ordinary matrix composition at matching square ports:

```python
C = G[word[0]]
for name in word[1:]:
    C = torch.bmm(C, G[name])
```

The motif bank is therefore an operad-like wiring discipline with explicit tensor semantics. No symbolic reasoning layer is required.

## 9. Tensor Contract

### Core tensors

| Name | Shape | Description |
|---|---:|---|
| `x` | `[B, C, 8, 8]` | Current-board tensor |
| `P` | `[B, 12, 64]` | Piece occupancy planes flattened |
| `occ` | `[B, 64]` | Any-piece occupancy |
| `H` | `[B, 64, d]` | Learned square embeddings from CNN stem |
| `mask_raw` | `[B, 36, 64, 64]` | Current-board geometric relation masks |
| `A_raw` | `[B, 36, 64, 64]` | Learned gated relation matrices |
| `G` | `[B, G, 64, 64]` | Grouped side-to-move relation matrices, recommended `G=10` |
| `motif_scores` | `[B, M]` | Trace and open-path scores per motif, recommended `M=32..64` |
| `contest_heatmap` | `[B, 8, 8]` | Near-puzzle diagnostic map |
| `logit` | `[B]` | Final puzzle logit |

### Recommended hyperparameters

```python
d = 96          # square embedding dimension
g = 32          # edge-gate inner dimension
K_raw = 36      # color x piece x role
G = 10          # grouped relation matrices
M = 48          # motif words
stem_blocks = 3
head_hidden = 256
dropout = 0.10
```

### Grouped relation matrices

After raw matrices are built, produce side-to-move groups:

```text
u_ctrl, u_hit, u_quiet,
t_ctrl, t_hit, t_quiet,
u_ray,  t_ray,
u_jump, t_jump
```

where `u` means side to move, `t` means opponent, `ray` mixes bishop/rook/queen matrices, and `jump` mixes knight/king/pawn-local control. Grouping is a learned nonnegative mixture over raw matrices, but side-to-move selection is deterministic from channel `12`.

### Primary model return

```python
return logit
```

### Diagnostic return

```python
return logit, {
    "prob": torch.sigmoid(logit),              # [B]
    "motif_scores": motif_scores,             # [B, M]
    "top_motif_idx": top_idx,                 # [B, k]
    "contest_heatmap": contest_heatmap,       # [B, 8, 8]
    "trace_closure": trace_closure,           # [B]
    "open_king_mass": open_king_mass,         # [B]
    "open_value_mass": open_value_mass,       # [B]
}
```

## 10. Architecture Specification

### 10.1 Module layout

```text
TracedThreatMotifNet
├── BoardStem
│   ├── Conv2d(C -> d, kernel=3, padding=1)
│   ├── 3 x residual Conv2d(d -> d)
│   └── LayerNorm over channel dimension after flattening
├── RelationMaskBuilder
│   ├── precomputed geometry masks
│   ├── precomputed between-square masks
│   └── current-board occupancy masks
├── RelationGate
│   ├── per-relation Q/K projections
│   ├── softplus edge scores
│   └── masked nonnegative A_raw matrices
├── GroupMixer
│   └── side-to-move grouped matrices G
├── MotifComposer
│   ├── matrix products for motif words
│   ├── categorical trace features
│   ├── open king/value path features
│   └── monoidal block-trace features
├── ContestPullback
│   └── target-square contest heatmap
└── Head
    ├── pooled CNN square features
    ├── motif features
    ├── contest features
    └── MLP -> one logit
```

### 10.2 Board stem

```python
class BoardStem(nn.Module):
    def __init__(self, in_channels: int, d: int = 96):
        super().__init__()
        self.inp = nn.Conv2d(in_channels, d, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(d) for _ in range(3)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        z = F.gelu(self.inp(x))
        for block in self.blocks:
            z = block(z)
        h = z.flatten(2).transpose(1, 2)  # [B, 64, d]
        return self.norm(h), z
```

### 10.3 Relation masks

Precompute once on CPU and register as buffers:

```python
geom_ctrl:  FloatTensor[6, 2, 64, 64]
geom_quiet: FloatTensor[6, 2, 64, 64]
between:    FloatTensor[64, 64, 64]
is_ray:     BoolTensor[6]
```

`between[i, j, k] = 1` if square `k` lies strictly between `i` and `j` on a bishop, rook, or queen ray; otherwise `0`.

Line clearance:

```python
blocked_count = torch.einsum("ijk,bk->bij", between, occ)  # [B, 64, 64]
clear = (blocked_count == 0).to(x.dtype)
```

Raw mask construction:

```python
mask_raw = []
for color in [WHITE, BLACK]:
    own_occ = occ_white if color == WHITE else occ_black
    enemy_occ = occ_black if color == WHITE else occ_white
    for piece in [P, N, B, R, Q, K]:
        src = P[:, idx(color, piece)]  # [B, 64]

        ctrl_geom = geom_ctrl[piece, color]  # [64, 64]
        quiet_geom = geom_quiet[piece, color]
        line = clear if piece in [B, R, Q] else 1.0

        ctrl = src[:, :, None] * ctrl_geom[None] * line
        hit = ctrl * enemy_occ[:, None, :]
        quiet = src[:, :, None] * quiet_geom[None] * line * (1.0 - occ[:, None, :])

        mask_raw.extend([ctrl, hit, quiet])
mask_raw = torch.stack(mask_raw, dim=1)  # [B, 36, 64, 64]
```

Pawn `quiet_geom` must include one-square forward moves and initial two-square moves only when the intermediate square is empty. Codex should handle the two-square case with an additional precomputed `pawn_mid_square[2, 64, 64]` mask and multiply by `1 - occ[mid]` where applicable.

### 10.4 Learned relation gates

```python
class RelationGate(nn.Module):
    def __init__(self, d=96, g=32, k_raw=36):
        super().__init__()
        self.wq = nn.Parameter(torch.randn(k_raw, d, g) * 0.02)
        self.wk = nn.Parameter(torch.randn(k_raw, d, g) * 0.02)
        self.bias = nn.Parameter(torch.zeros(k_raw))

    def forward(self, H, mask_raw):
        # H: [B, 64, d]
        Q = torch.einsum("bid,kdg->bkig", H, self.wq)  # [B, K, 64, g]
        K = torch.einsum("bid,kdg->bkig", H, self.wk)  # [B, K, 64, g]
        score = torch.einsum("bkig,bkjg->bkij", Q, K) / math.sqrt(Q.size(-1))
        score = score + self.bias[None, :, None, None]
        A = mask_raw * F.softplus(score)
        row_norm = A.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return A / row_norm
```

`softplus` keeps edge strengths nonnegative. Masking enforces current-board chess geometry.

### 10.5 Group mixer

Build color-role aggregates first, then select side-to-move.

```python
def stm_select(white_tensor, black_tensor, stm):
    # stm: [B], 1.0 white to move, 0.0 black to move
    s = stm[:, None, None]
    return s * white_tensor + (1.0 - s) * black_tensor
```

Recommended groups:

```python
groups = {
    "u_ctrl":  stm_select(white_ctrl,  black_ctrl,  stm),
    "u_hit":   stm_select(white_hit,   black_hit,   stm),
    "u_quiet": stm_select(white_quiet, black_quiet, stm),
    "t_ctrl":  stm_select(black_ctrl,  white_ctrl,  stm),
    "t_hit":   stm_select(black_hit,   white_hit,   stm),
    "t_quiet": stm_select(black_quiet, white_quiet, stm),
    "u_ray":   stm_select(white_ray,   black_ray,   stm),
    "t_ray":   stm_select(black_ray,   white_ray,   stm),
    "u_jump":  stm_select(white_jump,  black_jump,  stm),
    "t_jump":  stm_select(black_jump,  white_jump,  stm),
}
```

Raw-to-group mixture weights are learned but constrained with `softmax` within each group:

```python
mixed_group = torch.einsum("k,bkij->bij", alpha_group, A_raw)
```

### 10.6 Motif words

Use a fixed motif list. Start with these 24 and optionally extend to 48 by adding piece-family variants:

```python
MOTIF_WORDS = [
    ("u_ctrl", "u_hit"),
    ("u_quiet", "u_hit"),
    ("u_ray", "u_hit"),
    ("u_jump", "u_hit"),
    ("t_hit", "u_hit"),
    ("t_ctrl", "u_hit"),

    ("u_quiet", "u_ctrl", "u_hit"),
    ("u_ctrl", "u_ctrl", "u_hit"),
    ("u_ray", "u_ctrl", "u_hit"),
    ("u_jump", "u_ctrl", "u_hit"),
    ("t_ctrl", "u_ctrl", "u_hit"),
    ("t_hit", "u_ctrl", "u_hit"),

    ("u_ctrl", "t_ctrl", "u_hit"),
    ("u_quiet", "t_ctrl", "u_hit"),
    ("u_ray", "t_ctrl", "u_hit"),
    ("u_jump", "t_ctrl", "u_hit"),

    ("u_quiet", "u_ctrl", "t_ctrl", "u_hit"),
    ("u_ctrl", "u_ray", "u_ctrl", "u_hit"),
    ("u_ctrl", "u_jump", "u_ctrl", "u_hit"),
    ("t_hit", "u_quiet", "u_ctrl", "u_hit"),
    ("t_ctrl", "u_quiet", "u_ctrl", "u_hit"),
    ("u_ray", "t_ctrl", "u_ctrl", "u_hit"),
    ("u_jump", "t_ctrl", "u_ctrl", "u_hit"),
    ("u_ctrl", "t_hit", "u_ctrl", "u_hit"),
]
```

Composition code:

```python
def compose_word(groups, word):
    C = groups[word[0]]
    for name in word[1:]:
        C = torch.bmm(C, groups[name])
    return C
```

### 10.7 Trace and open-path features

For each composed motif matrix `C_m: [B, 64, 64]`:

```python
trace_m = C_m.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 64.0
mass_m = torch.log1p(C_m.sum(dim=(-1, -2)))
king_m = torch.einsum("bi,bij,bj->b", u_piece, C_m, v_enemy_king)
value_m = torch.einsum("bi,bij,bj->b", u_piece, C_m, v_enemy_value)
```

Concatenate:

```python
motif_features = torch.cat([trace_vec, mass_vec, king_vec, value_vec], dim=-1)
```

### 10.8 Monoidal product feature

Do not materialize the `128 x 128` block matrix unless debugging. Use the identity

```text
trace(block_diag(A, B)^n) = trace(A^n) + trace(B^n)
```

Recommended features:

```python
u_loop2 = trace(groups["u_ctrl"] @ groups["u_ctrl"])
t_loop2 = trace(groups["t_ctrl"] @ groups["t_ctrl"])
parallel_loop2 = u_loop2 + t_loop2
interaction_loop = trace(groups["u_ctrl"] @ groups["t_ctrl"])
```

Implementation:

```python
def tr(A):
    return A.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / A.size(-1)

parallel_loop2 = tr(torch.bmm(groups["u_ctrl"], groups["u_ctrl"])) + \
                 tr(torch.bmm(groups["t_ctrl"], groups["t_ctrl"]))
interaction_loop = tr(torch.bmm(groups["u_ctrl"], groups["t_ctrl"]))
```

### 10.9 Pullback surrogate contest map

Both control maps land in the same target-square coordinate. Multiply their target marginals:

```python
incoming_u = groups["u_ctrl"].sum(dim=1)  # [B, 64], target columns
incoming_t = groups["t_ctrl"].sum(dim=1)  # [B, 64]
contest = incoming_u * incoming_t          # [B, 64]
contest_heatmap = contest.view(B, 8, 8)
```

Aggregate contest diagnostics:

```python
contest_mean = contest.mean(dim=-1)
contest_top4 = contest.topk(4, dim=-1).values.mean(dim=-1)
contest_entropy = entropy(contest / contest.sum(dim=-1, keepdim=True).clamp_min(1e-6))
```

### 10.10 Final head

```python
cnn_pool = torch.cat([
    H.mean(dim=1),
    H.max(dim=1).values,
], dim=-1)  # [B, 2d]

head_in = torch.cat([
    cnn_pool,
    motif_features,
    monoidal_features,
    contest_features,
], dim=-1)

logit = mlp(head_in).squeeze(-1)
```

Recommended MLP:

```python
nn.Sequential(
    nn.LayerNorm(head_dim),
    nn.Linear(head_dim, 256),
    nn.GELU(),
    nn.Dropout(0.10),
    nn.Linear(256, 64),
    nn.GELU(),
    nn.Linear(64, 1),
)
```

## 11. Training Objective

### Main loss

```python
target = (fine_label == 2).float()
pos_weight = torch.tensor([num_negative / max(num_positive, 1)], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss = criterion(logit, target)
```

Use PyTorch `BCEWithLogitsLoss` rather than a separate sigmoid plus BCE because it is the numerically stable one-logit binary-classification loss. [R9]

### Regularization

Use ordinary regularization only:

```python
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
```

Optional non-label regularizer to prevent motif collapse:

```python
motif_prob = torch.softmax(motif_scores.detach(), dim=-1)
motif_entropy = -(motif_prob * motif_prob.clamp_min(1e-8).log()).sum(dim=-1).mean()
loss = loss - 0.001 * motif_entropy
```

Keep the entropy term small. It is not a substitute for validation ablations.

### Sampling

- Use balanced or class-aware mini-batches if positives are rare.
- Never use source labels or verification metadata for splits or sampling.
- Stratify validation reports by fine label after prediction:

```python
hard_negative = fine_label == 0
near_negative = fine_label == 1
positive = fine_label == 2
```

### Metrics

Report at minimum:

- binary ROC-AUC
- binary PR-AUC
- Brier score or expected calibration error
- recall at fixed false-positive rates
- mean predicted probability by fine label `0`, `1`, `2`
- false-positive rate separately for fine `0` and fine `1`
- top-motif distribution separately for fine `0`, `1`, `2`

## 12. Ablations

Run these as explicit falsification-oriented ablations.

| Ablation | Change | Expected outcome if TTMN is useful |
|---|---|---|
| CNN-only | Remove relation matrices, motif features, contest map | TTMN should beat this by a meaningful margin. |
| Geometry-only | Use `mask_raw` without learned gates | Should underperform learned gates but still expose basic signal. |
| No trace | Keep path mass and open paths, remove diagonal trace scores | Should lose performance on tactical-loop-heavy examples. |
| No open king/value paths | Keep traces only | Should lose target specificity. |
| No contest pullback | Remove `contest_heatmap` and contest aggregates | Should weaken near-puzzle diagnostics. |
| No quiet relations | Remove `quiet` primitive matrices | Should hurt quiet-move and latent-threat positions. |
| Random geometry masks | Shuffle target squares inside relation masks | Should collapse or sharply degrade. |
| Side-to-move removed | Force `u/t` grouping to fixed white/black | Should degrade because puzzle status is side-dependent. |
| Motif-word shuffle | Keep same matrices but randomize motif words | Should degrade if composition order matters. |
| Raw-matrix-only head | Pool `A_raw` statistics without products | Should underperform composed motif bank. |

## 13. Falsification Criteria

Reject or redesign TTMN if any of the following occur on a clean validation split:

1. **No baseline lift:** TTMN fails to outperform a comparable residual CNN by at least one of: higher PR-AUC, higher recall at the same false-positive rate, or better calibration.
2. **Composition is irrelevant:** removing trace and motif composition changes validation PR-AUC by less than noise across at least three seeds.
3. **Diagnostics are uninformative:** fine-label `1` near-negatives do not show a distinct distribution from fine-label `0` hard negatives in motif scores, contest heatmaps, or calibration curves.
4. **Mask randomization does not hurt:** random geometry masks perform within noise of real chess geometry. That would mean the architecture is not using the intended board relations.
5. **Forbidden leakage is discovered:** any feature path uses engine-derived values, best moves, PVs, mate labels, source tags, or verification metadata.
6. **Trace collapse:** one motif dominates all predictions across most samples and ablation shows no dependence on current-board relation masks.
7. **Side-to-move invariance failure:** flipping side-to-move while keeping pieces fixed does not change scores on positions where legal tactical responsibility clearly changes.
8. **Near-puzzle overpromotion:** fine-label `1` negatives are assigned positive-level probabilities with no threshold that preserves useful precision.

## 14. Codex Implementation Notes

### Files to implement

```text
models/traced_threat_motif_net.py
```

Optional tests:

```text
tests/test_traced_threat_motif_net.py
```

### Minimal class API

```python
class TracedThreatMotifNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d: int = 96,
        gate_dim: int = 32,
        motif_words: Optional[list[tuple[str, ...]]] = None,
        dropout: float = 0.10,
    ):
        ...

    def forward(self, x: torch.Tensor, return_diag: bool = False):
        ...
```

### Required assertions

```python
assert x.ndim == 4
assert x.shape[-2:] == (8, 8)
assert x.shape[1] >= 13
```

### Implementation order

1. Implement square-index helpers and geometry buffers.
2. Implement `RelationMaskBuilder` with tests for rook, bishop, queen, knight, king, and pawn masks.
3. Implement exact blocker handling with `between` masks.
4. Implement `BoardStem`.
5. Implement `RelationGate`.
6. Implement side-to-move group aggregation.
7. Implement motif composition and trace features.
8. Implement contest heatmap.
9. Implement MLP head.
10. Implement diagnostics output.

### Unit tests

Use synthetic boards only. Do not call an engine.

- Empty board plus one rook: rook control should extend along ranks/files.
- Rook with blocker: squares beyond blocker should be zero.
- Knight in center: exactly eight control targets when unobstructed by board edges.
- Pawn direction: white and black pawn control directions differ.
- Side-to-move grouping: `u_ctrl` swaps when `stm` flips.
- Trace shape: every motif trace returns `[B]`.
- Logit shape: forward returns `[B]`.
- Diagnostics shape: contest heatmap returns `[B, 8, 8]`.
- Forbidden inputs: dataset batch schema test should reject keys such as `engine_eval`, `pv`, `best_move`, `mate_score`, `node_count`, `source_label`, and `verification_metadata`.

### Skeleton forward pass

```python
def forward(self, x, return_diag=False):
    B = x.size(0)
    P = x[:, :12].flatten(2)                  # [B, 12, 64]
    stm = x[:, 12, 0, 0].float().clamp(0, 1)  # [B]

    H, Z = self.stem(x)                       # H [B, 64, d], Z [B, d, 8, 8]
    mask_raw = self.mask_builder(P, stm)      # [B, 36, 64, 64]
    A_raw = self.relation_gate(H, mask_raw)   # [B, 36, 64, 64]
    groups = self.group_mixer(A_raw, P, stm)  # dict[str, Tensor[B,64,64]]

    motif_features, motif_scores, extra = self.motif_composer(groups, P, stm)
    contest_heatmap, contest_features = self.contest_pullback(groups)

    cnn_pool = torch.cat([H.mean(1), H.max(1).values], dim=-1)
    head_in = torch.cat([cnn_pool, motif_features, contest_features], dim=-1)
    logit = self.head(head_in).squeeze(-1)

    if not return_diag:
        return logit

    diag = {
        "prob": torch.sigmoid(logit),
        "motif_scores": motif_scores,
        "top_motif_idx": motif_scores.topk(min(5, motif_scores.size(1)), dim=-1).indices,
        "contest_heatmap": contest_heatmap,
        **extra,
    }
    return logit, diag
```

### Near-puzzle diagnostic helper

```python
@torch.no_grad()
def near_puzzle_report(logit, fine_label, diag, threshold=0.5):
    prob = torch.sigmoid(logit)
    masks = {
        "fine0_hard_negative": fine_label == 0,
        "fine1_near_negative": fine_label == 1,
        "fine2_positive": fine_label == 2,
    }
    report = {}
    for name, m in masks.items():
        if m.any():
            report[name] = {
                "n": int(m.sum().item()),
                "prob_mean": float(prob[m].mean().item()),
                "prob_p90": float(prob[m].quantile(0.90).item()),
                "pred_positive_rate": float((prob[m] >= threshold).float().mean().item()),
                "trace_closure_mean": float(diag["trace_closure"][m].mean().item()),
                "open_king_mass_mean": float(diag["open_king_mass"][m].mean().item()),
            }
    return report
```

### Practical notes

- Use `torch.float32` for relation masks and matrix products initially. Mixed precision can be added after correctness tests.
- Matrix products are small: `64 x 64`, so the motif bank should be tractable even with dozens of motifs.
- If memory becomes tight, compute motif features in chunks and discard composed matrices immediately after extracting trace/open-path scores.
- Keep geometry masks as registered buffers so checkpoints are deterministic.
- Avoid in-place edits on tensors used in autograd paths.

## 15. Prompt-Maintenance Notes

Future revisions must preserve the following constraints:

- Keep the output as one puzzle logit.
- Keep fine-label mapping exactly `0/1 -> 0`, `2 -> 1`.
- Keep fine label `1` available only for near-puzzle diagnostics after prediction.
- Do not add any engine-derived inputs or labels.
- Do not add best-move, PV, node-count, mate-score, source-label, or verification-metadata features.
- Do not replace the architecture with generic graph message passing.
- Do not use symbolic program induction or move-sequence automata.
- Do not use abstract category-theory phrases unless each phrase is tied to a tensor operation.
- Required tensor-operation mapping:
  - composition -> `torch.bmm`
  - categorical trace -> diagonal sum
  - monoidal product -> block-diagonal parallel feature or its trace identity
  - pullback surrogate -> Hadamard product of target-square marginals
  - operad composition -> fixed motif-word matrix composition
- If a proposed enhancement cannot be implemented as shapes and PyTorch operations under the current-board-only contract, reject it.
