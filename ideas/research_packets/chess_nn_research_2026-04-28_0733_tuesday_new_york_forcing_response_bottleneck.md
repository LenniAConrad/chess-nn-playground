# Codex Handoff Packet: Forcing-Response Front-Door Bottleneck

## 1. File Metadata

- **Project:** `chess-nn-playground`
- **Task:** `puzzle_binary`
- **Packet date:** 2026-04-28
- **Packet time:** 07:33
- **Weekday:** Tuesday
- **Timezone:** new_york
- **Idea slug:** `forcing_response_bottleneck`
- **Selected causal object:** Front-door surrogate with a sparse causal bottleneck and rule-intervention consistency.
- **One-sentence summary:** Predict `puzzle_binary` through a bottleneck over deterministic legal-move response envelopes, so near-puzzle positions with tempting surface motifs but many rule-visible refutations are less likely to become false positives.

## 2. Executive Selection

Build **Forcing-Response Front-Door Bottleneck**: a model whose label head cannot directly consume raw board-surface style. The board is first converted into a set of deterministic legal interventions: apply each legal candidate move from the current position, summarize the opponent response envelope with rule-only features, encode those move-response nodes with a graph/DeepSets module, then pass only a sparse witness bottleneck to the binary classifier.

The selected causal object is a **front-door surrogate mediator**:

```text
current board X  ->  legal intervention response envelope M  ->  puzzle label Y
source/style U   ->  board surface shortcuts X_surface        ->  false positives
```

The intervention is `do(a)` where `a` is a legal move from the current board. It is label-safe because the after-move board is **not** treated as a newly labeled sample and is **not** assumed to keep the original label. It is only a deterministic mediator feature for the original board. No engine, PV, score, best move, mate score, source ID, verification metadata, or source-derived grouping is touched.

Why this should reduce near-puzzle false positives: near-puzzles often contain the same visible triggers as true puzzles, such as checks, captures, pins, exposed kings, or hanging material. The missing piece is usually the forced response structure. A shallow rule-only response envelope can expose “this tactic-looking move has too many legal escapes / recaptures / counterchecks / king exits” without needing an engine. The bottleneck forces the classifier to use that mechanism instead of broad composition-like texture.

## 3. Data Contract And Leakage Checklist

### Allowed inference inputs

- Existing current-board tensor for the side to move.
- Deterministic rule-derived features computed only from that tensor.
- Visible-board legal move set, using only state available in the tensor.
- Deterministic afterstate summaries from applying legal moves to the current board.
- Optional deterministic horizontal file reflection during training and evaluation checks.

### Allowed training-only targets

- Binary `puzzle_binary` label.
- Optional fine label `0/1/2` for auxiliary loss and reporting only.
- Fine label is never passed to `forward()` as an input feature.
- If the project convention is `0 = ordinary negative`, `1 = near-puzzle negative`, `2 = confirmed puzzle`, use that convention for near-puzzle FPR reporting. If the project already defines a different mapping, do not silently remap; bind the reporting code to the existing dataset metadata.

### Forbidden data and operations

- No engine scores.
- No principal variations.
- No node counts.
- No mate scores or mate-depth targets.
- No best moves.
- No verification metadata.
- No source IDs.
- No source-label inputs.
- No source-domain adversaries.
- No rule-partition V-REx over phase, material, or color.
- No nuisance-vector projection objective.
- No role-counterfactual swaps.
- No null-move contrast.
- No tempo odd/even intervention.
- No calibration-only solution.

### Legal-state restrictions

- Do not read FEN extras, PGN tags, study IDs, puzzle IDs, URL fragments, verification flags, engine annotations, or source folders.
- Castling and en-passant are disabled unless they are explicitly represented in the current board tensor. They are not inferable from piece placement alone.
- Check, pinned-piece, attack, legal-move, and afterstate summaries are allowed because they are deterministic consequences of the visible board state.
- A zero-reply afterstate may appear as a clipped legal-reply count if produced naturally by the legal generator, but do not create a dedicated mate-depth or engine-mate feature.

### Batch-key leakage asserts

Add a fail-fast guard that rejects feature dictionaries containing suspicious keys such as:

```text
source, site, study, lichess, chesscom, engine, eval, score, cp, mate,
pv, best, nodes, depth, verification, verified, generator, provenance
```

The guard should run in dataset construction and in the training loop before model input collation.

## 4. Causal Research Map

### Variables

- `X`: current board tensor and side-to-move state.
- `R(X)`: deterministic rule features from `X`.
- `A(X)`: legal candidate move set from the visible board.
- `T(X, a)`: deterministic transition after legal move `a`.
- `B(T(X, a))`: opponent legal-reply set after candidate move `a`.
- `M_a`: move-response mediator for candidate `a`.
- `M = {M_a : a in A(X)}`: front-door surrogate mediator set.
- `Z_c`: sparse causal bottleneck over `M`.
- `Y`: binary puzzle label.
- `F`: optional fine label used only for training/reporting.
- `U`: unobserved source/style/generator/curation artifacts.

### Causal claim

The model should estimate a stable mechanism closer to:

```text
P(Y | legal response envelope M)
```

instead of the brittle shortcut:

```text
P(Y | board surface X_surface)
```

This is not a proof of front-door identifiability. It is a practical front-door **surrogate**: `M` is a deterministic consequence of `X`, but the classifier is structurally forced to route prediction through legal consequences rather than through raw board texture.

### Why the mediator is useful

A true puzzle is not merely a board with tactical-looking atoms. It usually has an actionable forcing mechanism. A near-puzzle may share atoms but lacks the response envelope that makes the mechanism work. The mediator encodes:

- candidate move type;
- check/capture/promotion status;
- after-move king pressure;
- opponent reply availability;
- deterministic recapture and countercheck channels;
- attack-map and pin/ray changes;
- escape-square and line-blocking structure.

These are rule consequences, not source artifacts.

### Intervention safety

`do(a)` is applied uniformly to every board as a legal-rule operator. It does not inspect the label, source, solution, engine score, or puzzle verification. The afterstate is not relabeled. It only becomes part of the original board's mediator. Therefore the intervention does not smuggle in labels or source identity.

The optional horizontal file reflection `h(X)` is label-safe because chess rules are symmetric under left-right file reflection when color, side to move, and pawn direction are preserved. It is not a role swap, not a color swap, and not a tempo intervention. It removes coordinate-name nuisance without changing the tactical existence class.

## 5. Candidate Search Trace

### Candidate A: raw-board invariant classifier

Rejected as the main idea. A raw CNN/Transformer can learn useful chess features, but it has no structural reason to distinguish true tactical forcing mechanisms from puzzle-like board texture. It is the failure mode we are trying to fix.

### Candidate B: file-reflection consistency only

Kept only as a supporting loss. Reflection consistency is label-safe and source-free, but it mostly attacks coordinate shortcuts. It does not directly address near-puzzle false positives caused by superficial tactical motifs.

### Candidate C: deterministic rule-feature concatenation

Rejected as insufficient. Concatenating attack maps, pin maps, and mobility counts can help, but the classifier can still treat them as broad correlates. The final design instead builds a move-response mediator and restricts label access through a sparse causal bottleneck.

### Candidate D: legal afterstate response envelope without bottleneck

Rejected as incomplete. It gives the model better features, but a high-capacity head can still memorize dataset-specific patterns. The bottleneck and mirror consistency are needed to make the learned representation mechanism-focused.

### Candidate E: selected design

Selected: **Forcing-Response Front-Door Bottleneck**. It has the strongest fit to the constraints because it uses no source labels, no engine supervision, no role swaps, no null moves, no tempo parity, and no calibration-only postprocessing. It gives a concrete intervention object and a concrete implementation path.

## 6. Rejected Approaches

- **Source-domain adversary:** explicitly forbidden and would require source labels or source proxies.
- **Source-label input:** explicitly forbidden and likely to overfit curation artifacts.
- **Rule-partition V-REx over phase/material/color:** explicitly forbidden. The design does not form environments from those partitions.
- **Nuisance-vector projection:** explicitly forbidden. The design uses structural routing, not learned nuisance removal.
- **Role-counterfactual swap:** explicitly forbidden. The design never swaps side-to-move roles or colors.
- **Null-move contrast:** explicitly forbidden. The design never constructs a pass move.
- **Tempo odd/even intervention:** explicitly forbidden. The design never uses parity of tempo or move count.
- **Calibration-only fix:** rejected. The architecture changes representation and training, not just thresholds.
- **Engine-distillation teacher:** forbidden because it would require scores, PVs, best moves, or node/depth metadata.
- **Best-move existence classifier:** forbidden because best moves are not allowed and would leak solution semantics.
- **Mate-depth feature:** forbidden as a mate-score proxy. Single-ply legal response counts are allowed only as rule counts, not as mate-depth labels.

## 7. Mathematical Thesis

Let `x` be the current board tensor. Let `A(x)` be the set of visible-board legal candidate moves. Let `T(x, a)` be the deterministic afterstate created by applying legal move `a`. Let `rho(.)` be a deterministic rule-feature extractor. Define each mediator node:

```text
M_a = psi_rule(
    rho(x),
    a,
    rho(T(x, a)),
    summarize({ rho(T(T(x, a), b)) : b in A(T(x, a)) })
)
```

where `summarize` is permutation-invariant over opponent replies and contains no label, source, engine, PV, score, best move, or verification information.

The model computes:

```text
u_a       = phi_theta(M_a)
g_a       = sparse_gate_theta(u_a, {u_j})
Z_c(x)    = sum_a g_a * v_theta(u_a) / (epsilon + sum_a g_a)
y_hat(x)  = sigmoid(w^T Z_c(x) + b)
```

The key thesis is:

```text
Y ⟂ X_surface | Z_c(M)
```

as an intended inductive bias, not as a guaranteed property of the data. `Z_c` is forced to be a low-capacity witness set over legal intervention responses. If a position is a near-puzzle because it has tactical-looking surface features but many deterministic response channels, then `M` should make that difference available while raw surface shortcuts are unavailable to the final head.

The front-door analogy is:

```text
X_tactical_mechanism -> M -> Y
```

with unobserved curation/style artifacts `U` influencing board appearance. Since `M` is generated from legal rules applied to the same visible board, `U` cannot directly write metadata into `M`. It can only affect `M` through the actual board position. The model is therefore biased toward mechanisms that survive as legal consequences.

## 8. Causal Variables And Interventions

### Primary intervention: legal candidate move

```text
do(A = a): apply legal move a to the current board x and compute deterministic response features.
```

This intervention changes the simulated board, but it does not create a new labeled training example. The original label remains attached only to the original board. The afterstate is a mediator observation.

### Secondary intervention: file reflection consistency

```text
h(x): mirror files a<->h, b<->g, c<->f, d<->e while preserving color, side to move, and pawn direction.
```

Expected property:

```text
y_hat(x) ~= y_hat(h(x))
Z_c(x)  ~= unmirror(Z_c(h(x)))
```

This is label-safe because it is a chess-rule symmetry over coordinates, not a role swap. It is not a source artifact because it is defined independently of dataset provenance and applied to all examples.

### Self-supervised intervention: masked mediator-feature reconstruction

Mask a random subset of non-identity columns in `M_a` and ask the move-response encoder to reconstruct them from the surrounding legal graph. This is label-free and source-free. It encourages the representation to model deterministic rule mechanisms rather than treating rule features as flat correlates.

Do not mask source labels because none exist. Do not mask labels into inputs. Do not reconstruct forbidden metadata.

### Non-interventions

The following are intentionally absent:

- no source-domain intervention;
- no phase/material/color environment split;
- no null move;
- no side/role swap;
- no odd/even tempo split;
- no engine-guided intervention.

## 9. Architecture Specification

### 9.1 RuleFeatureBuilder

Input: current board tensor `X`.

Output:

- `rule_planes`: square-level deterministic planes.
- `move_features`: one row per visible-board legal candidate move.
- `response_features`: one row per candidate move summarizing opponent replies after the candidate.
- `move_mask`: valid candidate mask.
- Optional `mirror_permutation`: square and move-index mapping for file reflection.

Implementation may use a pure legal-move generator or a library such as `python-chess` in preprocessing. It must never call an engine or consume engine annotations.

### 9.2 Board stem

Use a small residual 8x8 encoder:

```text
H0 = ConvStem(concat(board_planes, rule_planes))
H  = 4 x ResidualBlock(64 or 96 channels, 3x3, layer norm/group norm)
```

The board stem is allowed to support move-node encoding, but the final classifier must not directly pool `H` into the label head.

### 9.3 Move-response node encoder

For each candidate `a`:

```text
from_emb = gather(H, from_square(a))
to_emb   = gather(H, to_square(a))
path_emb = deterministic ray/path pooling if sliding move, else zero
r_emb    = MLP_response(response_features[a])
m_emb    = MLP_move(move_features[a])
u_a      = MLP_node([from_emb, to_emb, path_emb, m_emb, r_emb])
```

The move encoder should include move identity only as board coordinates and piece/action type. It should not include best-move rank, puzzle solution order, source, or engine information.

### 9.4 Candidate interaction graph

Build edges using deterministic relations:

- same origin square;
- same target square;
- same captured piece square;
- both give check;
- both are captures;
- one candidate opens/closes the same ray;
- candidate afterstate allows reply that captures the moved piece;
- candidate affects king-ring attack planes.

Run 2 layers of graph attention or a set transformer with edge embeddings:

```text
U' = MoveGraphTransformer(U, edge_features, move_mask)
```

The graph is permutation-invariant over candidate ordering.

### 9.5 Sparse causal witness bottleneck

Use hard-concrete gates or entmax attention over candidate nodes:

```text
g_a = HardConcreteGate(U'_a)              during training
k_a = top_k(g_a, K=4 or 6)                during eval
Z_c = LayerNorm(sum_a k_a * Value(U'_a) / (epsilon + sum_a k_a))
```

Default bottleneck dimension: `D_z = 32` or `64`.

The bottleneck is the only path to the binary head:

```text
y_logit = MLP_binary(Z_c)
```

Optional training-only heads may branch from `Z_c`:

- fine-label ordinal head;
- masked mediator reconstruction head;
- witness diagnostics head for logging only.

No head may consume source labels because none are allowed.

### 9.6 Why this is not nuisance projection

The model does not learn a nuisance vector and subtract/project it away. It uses architectural separation: the binary decision must pass through deterministic legal-intervention mediators and a sparse causal bottleneck. The raw board stem cannot bypass that route.

## 10. Tensor Contract

### Batch dictionary

```python
batch = {
    "board": FloatTensor[B, Cb, 8, 8],
    "rule_planes": FloatTensor[B, Cr, 8, 8],
    "move_from": LongTensor[B, M_MAX],
    "move_to": LongTensor[B, M_MAX],
    "move_features": FloatTensor[B, M_MAX, Fm],
    "response_features": FloatTensor[B, M_MAX, Fr],
    "move_mask": BoolTensor[B, M_MAX],
    "label": FloatTensor[B],
    "fine_label": Optional[LongTensor[B]],
}
```

### Default sizes

- `M_MAX = 256` legal candidates.
- `Fm = 48 to 96`, depending on move feature vocabulary.
- `Fr = 64 to 128`, depending on response summary vocabulary.
- `D_node = 128`.
- `D_z = 32 or 64`.
- `K = 4 or 6` sparse witnesses.

### Move feature examples

Allowed:

- from square and to square;
- moved piece type;
- captured piece type if any;
- promotion type if any;
- check flag from legal rules;
- capture flag;
- promotion flag;
- visible-board legality flag;
- attack-map delta buckets;
- pin/ray delta buckets;
- king-ring pressure delta buckets.

Forbidden:

- engine evaluation;
- best-move rank;
- PV membership;
- node count;
- search depth;
- source or verification fields.

### Response feature examples

Allowed summaries after candidate `a`:

- clipped/log number of opponent legal replies;
- counts of opponent replies by type: check, capture, promotion, king move, block line, capture checking piece;
- whether the destination square can be attacked by opponent pieces after `a`;
- counts of replies that capture the moved piece;
- king escape-square count buckets;
- attack-map delta buckets;
- pin/ray delta buckets;
- piece-type bucket counts, not material evaluation scores.

Use `log1p` and clipping for counts. Do not encode a solved tactic, best reply, engine-optimal reply, mate depth, or PV.

### Forward signature

```python
logits, aux = model(
    board=batch["board"],
    rule_planes=batch["rule_planes"],
    move_from=batch["move_from"],
    move_to=batch["move_to"],
    move_features=batch["move_features"],
    response_features=batch["response_features"],
    move_mask=batch["move_mask"],
)
```

`label` and `fine_label` are consumed only by the loss function after `forward()` returns.

## 11. Losses

Total loss:

```text
L = L_binary
  + lambda_mirror * L_mirror
  + lambda_bottleneck * L_bottleneck
  + lambda_sparse * L_sparse
  + lambda_masked * L_masked_mediator
  + lambda_fine * L_fine_optional
  + lambda_near * L_near_optional
```

### Binary task loss

```text
L_binary = BCEWithLogitsLoss(y_logit, y)
```

Use class weighting only if already standard in `puzzle_binary`; this packet is not a calibration-only proposal.

### Mirror intervention consistency

For file reflection `h`:

```text
L_mirror = BCEConsistency(y_logit(x), y_logit(h(x)))
         + gamma_z * || Z_c(x) - unmirror_Z(Z_c(h(x))) ||_2^2
```

A simple implementation can use symmetric KL between Bernoulli probabilities for logits. If `Z_c` has no square-indexed components, `unmirror_Z` is identity.

### Bottleneck loss

If using a variational bottleneck:

```text
L_bottleneck = KL(N(mu_z, sigma_z^2) || N(0, I))
```

If using deterministic `Z_c`, replace with:

```text
L_bottleneck = ||Z_c||_2^2 / D_z
```

Default: start deterministic for implementation simplicity; add variational KL only after the pipeline is stable.

### Sparse witness loss

For hard-concrete gates:

```text
L_sparse = mean(sum_a E[g_a])
```

For entmax/top-k attention:

```text
L_sparse = max(0, active_count - K_target)^2
```

Do not force a single move. Some puzzles require a small set of interacting witnesses. Default `K_target = 4`.

### Masked mediator modeling

Randomly mask 15-30% of allowed mediator feature columns in `move_features` and `response_features`, excluding coordinates and validity masks. Reconstruct masked continuous buckets with Huber loss and categorical buckets with cross entropy:

```text
L_masked_mediator = Huber(M_hat_masked, M_masked)
```

This loss is label-free. It encourages the encoder to understand deterministic legal mechanisms.

### Optional fine-label loss

If fine labels are available and the project mapping supports it:

```text
L_fine_optional = CE(fine_logits(Z_c), fine_label)
```

This head is training-only. It must not change inference inputs.

### Optional near-puzzle margin

If `fine_label = 1` means near-puzzle negative and `fine_label = 2` means confirmed puzzle:

```text
L_near_optional = mean(max(0, margin + s_near - s_pos))
```

where `s` is the binary puzzle logit and positives are sampled from the same batch or a memory queue. Default `margin = 0.5`. This is not calibration-only because it shapes the bottleneck representation during training.

### Default loss weights

```text
lambda_mirror     = 0.20
lambda_bottleneck = 1e-4 deterministic, or 1e-3 variational
lambda_sparse     = 1e-3
lambda_masked     = 0.10
lambda_fine       = 0.20 if fine labels exist, else 0
lambda_near       = 0.20 if fine mapping is confirmed, else 0
```

Tune only on validation data without source metadata.

## 12. Ablations

Run these ablations in order, keeping the same train/validation/test split and reporting near-puzzle false positives separately.

1. **Baseline board model:** existing `puzzle_binary` architecture with current board tensor only.
2. **Rule planes only:** baseline plus deterministic square-level rule planes, no legal afterstate response envelope.
3. **Move-response features without bottleneck:** concatenate pooled move-response features into the classifier.
4. **Full model without mirror loss:** front-door mediator plus sparse bottleneck, no reflection consistency.
5. **Full model without masked mediator modeling:** tests whether self-supervised rule-mechanism modeling matters.
6. **Full model without optional fine loss:** confirms the design does not depend on fine labels.
7. **Full model without response summaries:** move features only, no opponent reply envelope.
8. **Different witness budgets:** `K = 1, 2, 4, 6, 8`.
9. **Different bottleneck widths:** `D_z = 16, 32, 64, 128`.
10. **No terminal-count sensitivity:** clip legal-reply counts into `0`, `1`, `2+` or redact zero-vs-one to ensure gains are not just mate-like leakage.

Primary comparison metric:

```text
near_puzzle_FPR_at_fixed_positive_TPR
```

Recommended fixed TPR values: 0.80, 0.90, and the operating point used by the current app.

Also report:

- AUROC;
- AUPRC;
- binary F1 at selected threshold;
- expected near-puzzle false positives per 1,000 candidates;
- mirror consistency gap;
- active witness count;
- failure examples with top witness moves and response summaries.

## 13. Falsification Criteria

Reject or redesign the idea if any of the following happen:

1. **No near-puzzle improvement:** near-puzzle FPR is not reduced by at least 15% relative at the same positive TPR after reasonable tuning.
2. **Only calibration improved:** ranking metrics and near-puzzle FPR do not improve before threshold tuning.
3. **Response envelope is irrelevant:** removing `response_features` gives statistically indistinguishable near-puzzle performance.
4. **Bottleneck is bypassed:** gradients or code inspection show the binary head can consume pooled raw board features outside `Z_c`.
5. **Mirror failure:** `abs(sigmoid(logit(x)) - sigmoid(logit(h(x))))` remains high on validation after mirror training.
6. **Fine-label dependence:** performance collapses when optional fine-label losses are disabled. The design may use fine labels, but it must not require them.
7. **Mate-like leakage:** gains disappear or reverse when terminal-count sensitivity is clipped/redacted, indicating the model mostly learned trivial mate-one detection.
8. **Sparse witnesses are nonsensical:** selected witness moves are dominated by quiet non-interacting moves with no response-envelope relevance across most positive predictions.
9. **Source proxy suspicion:** performance is high but fails on canonicalized board splits or hash-based deduplication, suggesting memorization or curation-style learning.
10. **Complexity not justified:** training/inference cost grows more than 2.5x while near-puzzle FPR gain is below the threshold in item 1.

## 14. Codex Implementation Notes

### Suggested files

```text
src/features/rule_intervention_features.py
src/models/forcing_response_bottleneck.py
src/training/losses_forcing_response.py
src/training/leakage_guards.py
configs/puzzle_binary_forcing_response_bottleneck.yaml
tests/test_rule_intervention_features.py
tests/test_no_forbidden_batch_keys.py
tests/test_mirror_equivariance.py
```

### Feature builder steps

1. Convert the current board tensor to an internal board object.
2. Generate visible-board legal candidate moves.
3. Pad/truncate to `M_MAX = 256` with `move_mask`.
4. For each candidate move:
   - apply the move deterministically;
   - compute afterstate rule planes/summaries;
   - enumerate opponent legal replies from the afterstate;
   - summarize reply types and attack/ray/pin deltas;
   - write `move_features[a]` and `response_features[a]`.
5. Cache features by a hash of the board tensor and rule-feature version string.
6. Ensure cache files contain no label, source, engine, or verification fields.

### Model implementation sketch

```python
class ForcingResponseBottleneck(nn.Module):
    def __init__(self, board_channels, rule_channels, move_dim, response_dim, z_dim=64, k=4):
        super().__init__()
        self.board_stem = BoardStem(board_channels + rule_channels)
        self.move_mlp = nn.Sequential(nn.Linear(move_dim, 128), nn.GELU(), nn.Linear(128, 128))
        self.response_mlp = nn.Sequential(nn.Linear(response_dim, 128), nn.GELU(), nn.Linear(128, 128))
        self.node_mlp = nn.Sequential(nn.Linear(128 * 4, 128), nn.GELU(), nn.Linear(128, 128))
        self.move_graph = MoveGraphTransformer(dim=128, layers=2)
        self.gate = SparseWitnessGate(dim=128, k=k)
        self.to_z = nn.Linear(128, z_dim)
        self.binary_head = nn.Sequential(nn.LayerNorm(z_dim), nn.Linear(z_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.masked_head = nn.Linear(128, move_dim + response_dim)
        self.fine_head = nn.Linear(z_dim, 3)

    def forward(self, board, rule_planes, move_from, move_to, move_features, response_features, move_mask):
        h = self.board_stem(torch.cat([board, rule_planes], dim=1))
        h_from = gather_square(h, move_from)
        h_to = gather_square(h, move_to)
        m = self.move_mlp(move_features)
        r = self.response_mlp(response_features)
        u = self.node_mlp(torch.cat([h_from, h_to, m, r], dim=-1))
        u = self.move_graph(u, move_mask=move_mask)
        gates = self.gate(u, move_mask)
        z = masked_weighted_mean(self.to_z(u), gates, move_mask)
        return self.binary_head(z).squeeze(-1), {
            "z_c": z,
            "gates": gates,
            "fine_logits": self.fine_head(z),
            "masked_pred": self.masked_head(u),
        }
```

The sketch is intentionally incomplete around helper functions, but the contract is strict: no binary head may receive raw pooled board features.

### Guardrails for Codex

- Add tests that monkeypatch forbidden keys into a batch and assert failure.
- Add tests that fine labels are absent from model `forward()`.
- Add tests that source IDs cannot be loaded from dataset rows.
- Add a test that horizontal reflection twice returns the original board tensor and candidate mapping.
- Add a test that candidate ordering permutation does not change logits beyond numerical tolerance.
- Log top witness moves only as coordinates and deterministic response summaries, never as best moves.
- Do not add a chess engine dependency.
- Do not call UCI, Stockfish, LC0, Syzygy, tablebases, or cloud analysis APIs.

### Training schedule

1. Precompute/cache rule-intervention features.
2. Warm up with `L_binary + L_masked_mediator` for 1-3 epochs.
3. Enable sparse gates and mirror consistency.
4. Enable optional fine/near losses only after confirming fine-label semantics.
5. Select checkpoint by validation near-puzzle FPR at fixed TPR, with AUROC/AUPRC as secondary metrics.

### Runtime notes

The expensive step is legal afterstate enumeration. Prefer cached preprocessing. For online inference, compute only summaries, not full reply tensors. If latency is high, keep all legal candidates but summarize replies in CPU preprocessing, then move fixed tensors to GPU.

## 15. Prompt Updates

Use this future implementation prompt for Codex:

```text
Implement Forcing-Response Front-Door Bottleneck for chess-nn-playground puzzle_binary.

Hard constraints:
- Inference inputs are current board tensor plus deterministic rule-derived features only.
- Fine labels 0/1/2 may be used only in training losses and reporting, never as forward inputs.
- Do not use engine scores, PVs, node counts, mate scores, best moves, verification metadata, source IDs, source labels, source adversaries, rule-partition V-REx, nuisance-vector projection, role swaps, null-move contrast, tempo parity, or calibration-only fixes.

Architecture:
- Build deterministic legal candidate move and opponent response-envelope features from the visible board.
- Encode candidate move-response nodes with board-square gathers, move MLP, response MLP, and a permutation-invariant move graph/set transformer.
- Route binary prediction only through a sparse causal witness bottleneck Z_c.
- Add label-safe horizontal file-reflection consistency and masked mediator reconstruction.
- Add optional fine-label and near-puzzle margin losses only after confirming fine-label semantics.

Validation:
- Report near-puzzle FPR at fixed positive TPR, AUROC, AUPRC, mirror gap, active witness count, and ablations.
- Include fail-fast leakage guards for forbidden batch keys and dataset fields.
- Prove by tests that the binary head cannot bypass Z_c and that model.forward has no label/fine/source arguments.
```
