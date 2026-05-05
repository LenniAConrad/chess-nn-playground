# Codex Handoff Packet: Tactical State Bottleneck Inference

## 1. File Metadata

- **Filename:** `chess_nn_research_2026-04-28_0901_tuesday_new_york_tactical_latent.md`
- **Generated at:** 2026-04-28 09:01 new_york
- **Task:** 11. Latent Variable Inference Puzzle Model
- **Dataset target:** `puzzle_binary`
- **Input:** current-board tensor only, shaped `[batch, C, 8, 8]`
- **Output:** one puzzle logit per board, shaped `[batch]`
- **Fine-to-binary mapping:** fine labels `0` and `1` map to binary `0`; fine label `2` maps to binary `1`
- **Selected idea:** Tactical State Bottleneck Inference, abbreviated `TSBI`
- **Primary requirement:** latent variables must represent a chess-specific hidden tactical state, not generic style, phase, board compression, source identity, or engine-derived strength.
- **Forbidden inputs:** engine scores, PVs, node counts, mate scores, best moves, solution moves, verification metadata, source labels, source provenance, and fields derived from any of them.

## 2. Executive Selection

Select **Tactical State Bottleneck Inference**.

TSBI predicts whether a board is a puzzle by first inferring a hidden tactical state:

```text
z = (motif, anchor_square, target_square, relation, vulnerability, tempo_bucket)
```

The model estimates:

```text
p(y = 1 | x) = sum_z p_theta(y = 1 | x, z) p_psi(z | x)
```

where `x` is the current board tensor and `y` is `puzzle_binary`. The latent state is not a generic embedding. It is a structured tactical hypothesis: what tactical mechanism may be present, where it is anchored, what it attacks or pressures, what geometric relation connects the objects, what tactical weakness exists, and how forcing the position appears.

This is the selected design because it satisfies all task constraints:

1. **Board-only inference.** The runtime model consumes only `[B, C, 8, 8]` current-board tensors.
2. **Chess-specific latent object.** The latent tuple represents tactical state, not style or phase.
3. **Single binary output.** The model returns one puzzle logit.
4. **No leakage.** It avoids engine data, solution data, verification data, and provenance data.
5. **Falsifiable contribution.** It must beat a no-latent matched baseline and show non-collapsed latent usage.
6. **Codex-ready.** The architecture, tensor shapes, losses, ablations, diagnostics, and implementation guardrails are explicit.

## 3. Data Contract

### Required batch schema

```python
batch = {
    "board": FloatTensor[B, C, 8, 8],
    "fine_label": LongTensor[B],  # values in {0, 1, 2}
}
```

### Binary target construction

```python
x = batch["board"]
fine = batch["fine_label"].long()
y = (fine == 2).float()
```

Rules:

```text
fine_label == 0 -> y = 0
fine_label == 1 -> y = 0
fine_label == 2 -> y = 1
```

Fine labels are used only for target construction and the required 3x2 diagnostic. Do not train a three-class classifier in this packet.

### Forbidden batch keys

The dataloader must reject or strip the following keys before the batch reaches the model or loss:

```python
FORBIDDEN_KEYS = {
    "engine_score",
    "engine_scores",
    "eval",
    "cp",
    "pv",
    "pvs",
    "principal_variation",
    "principal_variations",
    "node_count",
    "node_counts",
    "mate_score",
    "mate_scores",
    "best_move",
    "best_moves",
    "solution_move",
    "solution_moves",
    "verification_metadata",
    "verified_by",
    "source_label",
    "source_labels",
    "source_provenance",
    "site_origin",
    "database_origin",
    "puzzle_generator_id",
    "curation_status",
}
```

Training step guard:

```python
bad_keys = set(batch.keys()) & FORBIDDEN_KEYS
assert not bad_keys, f"Forbidden leakage keys in batch: {sorted(bad_keys)}"
```

### Permitted derived quantities

The model may compute deterministic board geometry from `x`, such as occupancy, piece-color masks, square coordinates, line masks, or attack-style geometry if those quantities are strictly functions of the current-board tensor. These are not additional inputs. They must not use best moves, search results, engine outputs, or puzzle metadata.

### Model output schema

Inference output:

```python
{
    "logit": FloatTensor[B]
}
```

Training/debug output:

```python
{
    "logit": FloatTensor[B],
    "logit_q": FloatTensor[B],          # posterior-path training logit
    "logit_p": FloatTensor[B],          # prior-path training logit
    "prior_logits": Dict[str, Tensor],
    "posterior_logits": Dict[str, Tensor],
    "latent_probs": Dict[str, Tensor],
    "losses": Dict[str, Tensor],
    "diag_3x2": LongTensor[3, 2],
}
```

### Required 3x2 diagnostic

Rows are original fine labels. Columns are binary prediction buckets using `logit > 0` as predicted positive.

```text
                 predicted_binary_0    predicted_binary_1
fine_label_0            count                 count
fine_label_1            count                 count
fine_label_2            count                 count
```

Implementation:

```python
def diagnostic_3x2(fine_label: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
    pred = (logit > 0).long()
    fine = fine_label.long().clamp(0, 2)
    table = torch.zeros(3, 2, dtype=torch.long, device=logit.device)
    table.view(-1).scatter_add_(0, fine * 2 + pred, torch.ones_like(fine, dtype=torch.long))
    return table
```

This diagnostic is mandatory because fine labels `0` and `1` are both negative after binary mapping. It catches failures where the model treats one negative subclass as puzzle-like.

## 4. Latent Inference Research Map

### Central research question

Can a board-only model improve puzzle detection by inferring a compact hidden tactical state instead of directly mapping board features to `puzzle_binary`?

### Model family

Use a supervised structured latent-variable classifier:

```text
current board x
  -> chess board trunk H(x)
  -> prior p_psi(z | x)
  -> tactical latent z
  -> puzzle logit g_theta(H(x), z)
```

During training, use a variational posterior:

```text
q_phi(z | x, y)
```

At inference, use only:

```text
p_psi(z | x)
```

### Research map

| Question | Mechanism | Required evidence |
|---|---|---|
| Does the latent object help? | Compare TSBI to no-latent matched baseline | TSBI improves AUROC/AUPRC/BCE without worse 3x2 behavior |
| Does the model use the latent? | Track KL, entropy, category usage, null-square rate | Multiple latent groups have nonzero KL and nontrivial usage |
| Does posterior training transfer to inference? | Train both posterior-path and prior-path logits | Prior-path validation is close to posterior-path validation |
| Are square latents meaningful? | Inspect anchor/target distributions and perturbation sensitivity | Anchor/target are not always null and are board-sensitive |
| Is the tactical state chess-specific? | Use motif, square, relation, vulnerability, tempo latents | No generic board reconstruction or style/phase latent |
| Is there leakage? | Batch-key audit and forbidden-field tests | No forbidden field reaches model, loss, metrics, or split logic |

### Interpretation standard

The latent state is successful only if it is both predictive and structurally used. A small metric gain with collapsed latents is not a success. Attractive latent visualizations without beating the matched baseline are also not a success.

## 5. Serious Candidate Rejections

### Rejected candidate A: latent best-move or solution-move state

Reject any design where `z` is a best move, candidate move, solution move, or hidden move sequence.

Reason: best moves and solution moves are forbidden. Even if the model tries to infer them indirectly, this would change the task from puzzle detection to move supervision and would risk encoding puzzle-solution metadata.

### Rejected candidate B: latent engine-evaluation state

Reject latent value buckets such as winning/equal/losing, mate-threat score, centipawn swing, or engine tactic strength.

Reason: engine scores, mate scores, PVs, and node counts are forbidden. The selected latent must be tactical geometry inferred from the board, not a proxy for search output.

### Rejected candidate C: generic board-compression VAE

Reject a VAE that reconstructs the board from a generic latent vector.

Reason: it would encode material, phase, piece placement, and style-like board identity. That is not the requested chess-specific hidden tactical object.

### Rejected candidate D: source/provenance latent

Reject any domain/source latent, even unsupervised.

Reason: source labels and provenance are forbidden. A source latent risks learning curation artifacts rather than tactics.

### Rejected candidate E: direct classifier as the main idea

A direct classifier is required as a baseline but rejected as the selected research idea.

Reason: the task asks for latent-variable inference. The direct model must be implemented only as the no-latent matched baseline.

### Rejected candidate F: motif-supervised auxiliary classifier

Reject supervised motif labels unless the dataset already contains board-only, non-engine, non-source motif annotations that are explicitly allowed. This packet assumes no such labels.

Reason: the latent state should be inferred from binary puzzle supervision and board geometry, not from extra tactical labels.

## 6. Common Approaches Rejected

Do not implement any of the following for this packet:

1. **Denoising score fields.** They push the model toward board repair or score shaping rather than tactical-state inference.
2. **Class-conditioned pseudo-likelihood.** The model must not estimate class-conditioned board likelihoods or pseudo-generative class scores.
3. **Masked codecs.** Masked board or piece reconstruction would reward generic board compression.
4. **Evidential heads.** The research contribution is a tactical latent state, not an uncertainty-head variant.
5. **Mixture-of-experts calibration.** This adds calibration machinery without solving the chess-specific latent-object requirement.
6. **Prototype margins.** Prototype separation is not the requested latent generative story.
7. **Generic VAEs without a chess-specific latent object.** The latent must be tactical state: motif, squares, relation, vulnerability, and tempo.

## 7. Mathematical Thesis

Let:

```text
x in R^{C x 8 x 8}
y in {0, 1}
z = (m, a, t, r, v, h)
```

where:

```text
m = motif category
a = anchor square
t = target square
r = tactical relation category
v = vulnerability category
h = tempo bucket
```

The predictive model is:

```text
p(y = 1 | x) = sum_z p_theta(y = 1 | x, z) p_psi(z | x)
```

with:

```text
p_theta(y = 1 | x, z) = sigmoid(g_theta(H_omega(x), e(z)))
```

`H_omega(x)` is the board trunk and `e(z)` is the concatenated embedding of the categorical latent variables.

Training uses a posterior network:

```text
q_phi(z | x, y)
```

The objective minimized over a batch is:

```text
L = L_pred
  + beta_kl * L_kl_freebits
  + lambda_prior * L_prior_pred
  + lambda_usage * L_batch_usage
  + lambda_entropy * L_entropy_floor
```

Prediction loss with posterior latents:

```text
L_pred = BCEWithLogits(g_theta(H(x), z_q), y),  z_q ~ q_phi(z | x, y)
```

Prediction loss with prior latents:

```text
L_prior_pred = BCEWithLogits(g_theta(H(x), z_p), y),  z_p ~ p_psi(z | x)
```

KL with free bits:

```text
L_kl_freebits = sum_j max(KL(q_phi(z_j | x, y) || p_psi(z_j | x)), tau_j)
```

Batch usage regularizer:

```text
L_batch_usage = sum_j KL(mean_batch q_phi(z_j | x, y) || Uniform_j)
```

Early entropy floor:

```text
L_entropy_floor = sum_j relu(H_min_j - H(q_phi(z_j | x, y)))
```

The thesis is that a puzzle position is more likely when the board supports a compact, localized, tactical explanation. The latent state is the bottleneck for that explanation. If the bottleneck is real, the model should improve generalization and produce non-collapsed, board-sensitive latent assignments.

## 8. Latent Variables And Generative Story

### Default latent cardinalities

```python
K_MOTIF = 10
K_SQUARE = 65          # 64 board squares plus null
K_RELATION = 8
K_VULNERABILITY = 8
K_TEMPO = 4
```

### Latent categories

```text
motif:
  0 null_or_quiet
  1 fork_or_double_attack
  2 pin_or_skewer
  3 discovered_attack
  4 deflection_or_decoy
  5 overload
  6 back_rank_or_mating_net
  7 loose_piece_or_hanging_piece
  8 trapped_piece_or_king_net
  9 other_forcing_tactic

anchor_square:
  0..63 board squares
  64 null

target_square:
  0..63 board squares
  64 null

relation:
  0 none
  1 line_relation
  2 knight_or_leaper_geometry
  3 king_ring_pressure
  4 defender_removal
  5 discovered_line_opening
  6 overloaded_defender_dependency
  7 loose_piece_capture_path

vulnerability:
  0 none
  1 loose_or_undefended_piece
  2 pinned_piece
  3 overloaded_defender
  4 exposed_king
  5 trapped_piece
  6 weak_back_rank
  7 fragile_king_ring

tempo_bucket:
  0 none_or_quiet
  1 immediate_tactic
  2 one_reply_forcing_sequence
  3 multi_reply_forcing_sequence
```

These names constrain the model design and diagnostics. They are not supervised labels.

### Generative story

For a board `x`:

1. The current pieces, occupancy, side-to-move context, and geometry imply a distribution over hidden tactical states.
2. A non-puzzle board usually has a null, diffuse, or incoherent tactical state.
3. A puzzle board usually has a more compact tactical state: a motif, anchor, target, relation, vulnerability, and forcing tempo.
4. The puzzle logit is produced from the board representation and this inferred tactical state.

The hidden state explicitly does not encode:

```text
best move
solution line
engine evaluation
mate score
search depth
node count
puzzle source
verification status
curation metadata
```

### Why this is not a generic VAE

TSBI does not reconstruct the board, does not optimize a generic board likelihood, and does not learn an unconstrained style vector. Its latent state is a typed tactical tuple with square-local and relation-local semantics.

## 9. Inference Network

### Module layout

```text
TacticalStateBottleneckModel
  ChessBoardTrunk
  PriorTacticalHead          # p_psi(z | x)
  PosteriorTacticalHead      # q_phi(z | x, y), training only
  LatentEmbeddingProjector
  PuzzleLogitHead
```

### Board trunk

Recommended default:

```text
input [B, C, 8, 8]
-> 3x3 convolution to D channels
-> 4 to 8 residual spatial blocks
-> optional internal coordinate planes
-> h [B, D, 8, 8]
-> pooled [B, D]
```

Keep the trunk strong enough to encode board geometry but not so large that it bypasses the latent state. The final head uses a capped direct path to prevent collapse.

### Prior head: `p_psi(z | x)`

The prior head sees only board features.

```python
prior_logits = {
    "motif": prior_motif(pooled),                   # [B, 10]
    "anchor": prior_anchor_square(h),              # [B, 65]
    "target": prior_target_square(h),              # [B, 65]
    "relation": prior_relation(pooled),             # [B, 8]
    "vulnerability": prior_vulnerability(pooled),   # [B, 8]
    "tempo": prior_tempo(pooled),                   # [B, 4]
}
```

Square logits use a 1x1 convolution plus a learned null logit:

```python
spatial = conv1x1(h).flatten(start_dim=1)  # [B, 64]
logits = torch.cat([spatial, null_logit.expand(B, 1)], dim=-1)  # [B, 65]
```

### Posterior head: `q_phi(z | x, y)`

The posterior is training-only. It sees board features and the binary target through a small learned label embedding.

```python
y_long = (fine_label == 2).long()
y_emb = label_embedding(y_long)  # [B, E_y]
pooled_post = torch.cat([pooled, y_emb], dim=-1)
y_map = y_emb[:, :, None, None].expand(-1, -1, 8, 8)
h_post = posterior_spatial_fuse(torch.cat([h, y_map], dim=1))
```

The posterior may sharpen latent assignment during training, but inference must never use `y`. The prior-path prediction loss is mandatory so `p_psi(z | x)` learns usable inference-time latents.

### Latent sampling and embedding

Training default:

```python
z = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
embedding = z @ embedding_table
```

Validation and inference default:

```python
probs = logits.softmax(dim=-1)
embedding = probs @ embedding_table
```

Recommended temperature schedule:

```text
early: 1.5, soft
middle: decay to 0.7
late: 0.5 to 0.7, optional straight-through hard samples
```

### Posterior-collapse controls

Use these controls in the selected model:

1. **KL warmup.** Start `beta_kl = 0`; ramp toward the target value.
2. **Free bits by latent group.** Avoid over-penalizing useful posterior information early.
3. **Prior-path prediction loss.** Always train a prior-only logit because inference uses the prior.
4. **Capped direct path.** Use:

   ```text
   logit = latent_logit + alpha_direct * direct_logit
   ```

   with `alpha_direct` warmed from `0.0` to at most `0.25`.

5. **Direct-feature dropout.** Drop pooled direct features during training, but do not drop latent embeddings.
6. **Batch usage regularizer.** Discourage global collapse to one motif/relation category.
7. **Early entropy floor.** Prevent premature hard collapse before specialization.
8. **Required no-latent matched baseline.** Latent usefulness is not accepted without this comparison.

## 10. Tensor Contract

### Inputs

```python
board: FloatTensor[B, C, 8, 8]
fine_label: Optional[LongTensor[B]]
```

### Derived target

```python
y_float = (fine_label == 2).float()  # [B]
y_long = (fine_label == 2).long()    # [B]
```

### Trunk outputs

```python
h = trunk(board)      # FloatTensor[B, D, 8, 8]
pooled = pool(h)      # FloatTensor[B, D]
```

### Prior and posterior logits

```python
logits["motif"]         # FloatTensor[B, 10]
logits["anchor"]        # FloatTensor[B, 65]
logits["target"]        # FloatTensor[B, 65]
logits["relation"]      # FloatTensor[B, 8]
logits["vulnerability"] # FloatTensor[B, 8]
logits["tempo"]         # FloatTensor[B, 4]
```

Posterior logits have the same shapes as prior logits.

### Latent probabilities or samples

```python
z["motif"]         # FloatTensor[B, 10]
z["anchor"]        # FloatTensor[B, 65]
z["target"]        # FloatTensor[B, 65]
z["relation"]      # FloatTensor[B, 8]
z["vulnerability"] # FloatTensor[B, 8]
z["tempo"]         # FloatTensor[B, 4]
```

### Latent embeddings

```python
e_motif = z["motif"] @ E_motif                  # [B, E]
e_anchor = z["anchor"] @ E_square              # [B, E]
e_target = z["target"] @ E_square              # [B, E]
e_relation = z["relation"] @ E_relation        # [B, E]
e_vulnerability = z["vulnerability"] @ E_vuln   # [B, E]
e_tempo = z["tempo"] @ E_tempo                 # [B, E]

z_emb = torch.cat([
    e_motif,
    e_anchor,
    e_target,
    e_relation,
    e_vulnerability,
    e_tempo,
], dim=-1)  # [B, 6 * E]
```

### Logit computation

```python
latent_input = torch.cat([pooled, z_emb], dim=-1)
latent_logit = latent_head(latent_input).squeeze(-1)  # [B]
direct_logit = direct_head(pooled).squeeze(-1)        # [B]
logit = latent_logit + alpha_direct * direct_logit    # [B]
```

### Forward signatures

Recommended clean split:

```python
def forward_train(self, board: torch.Tensor, fine_label: torch.Tensor) -> Dict[str, Tensor]:
    ...  # returns logit_q and logit_p

@torch.no_grad()
def forward_eval(self, board: torch.Tensor, return_latents: bool = False) -> Dict[str, Tensor]:
    ...  # returns prior-only logit
```

Generic wrapper:

```python
def forward(
    self,
    board: torch.Tensor,
    fine_label: Optional[torch.Tensor] = None,
    return_latents: bool = False,
) -> Dict[str, Tensor]:
    if self.training and fine_label is not None:
        return self.forward_train(board, fine_label)
    return self.forward_eval(board, return_latents=return_latents)
```

## 11. Objective Function

### Prediction losses

Posterior-path prediction:

```python
y = (fine_label == 2).float()
loss_pred = F.binary_cross_entropy_with_logits(outputs["logit_q"], y)
```

Prior-path prediction:

```python
loss_prior_pred = F.binary_cross_entropy_with_logits(outputs["logit_p"], y)
```

The prior-path loss is not optional. It closes the train/inference gap.

### KL with free bits

```python
def kl_freebits(posterior_logits, prior_logits, free_bits):
    total = 0.0
    per_group = {}
    for name in posterior_logits.keys():
        q = torch.distributions.Categorical(logits=posterior_logits[name])
        p = torch.distributions.Categorical(logits=prior_logits[name])
        kl = torch.distributions.kl_divergence(q, p).mean()
        kl_fb = torch.clamp(kl, min=free_bits[name])
        total = total + kl_fb
        per_group[name] = kl.detach()
    return total, per_group
```

Initial free-bit values:

```python
free_bits = {
    "motif": 0.05,
    "anchor": 0.10,
    "target": 0.10,
    "relation": 0.05,
    "vulnerability": 0.05,
    "tempo": 0.03,
}
```

### Batch usage loss

```python
def batch_usage_loss(logits_by_group):
    loss = 0.0
    for logits in logits_by_group.values():
        probs = logits.softmax(dim=-1)
        q_bar = probs.mean(dim=0).clamp_min(1e-8)
        uniform = torch.full_like(q_bar, 1.0 / q_bar.numel())
        loss = loss + F.kl_div(q_bar.log(), uniform, reduction="sum")
    return loss
```

This discourages every example from using the same category. Keep the coefficient small.

### Entropy floor loss

```python
def entropy_floor_loss(logits_by_group, entropy_floor):
    loss = 0.0
    for name, logits in logits_by_group.items():
        probs = logits.softmax(dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        loss = loss + F.relu(entropy_floor[name] - entropy)
    return loss
```

Use the entropy floor early, then decay it to zero.

### Full objective

```python
loss = (
    loss_pred
    + beta_kl * loss_kl
    + lambda_prior * loss_prior_pred
    + lambda_usage * loss_usage
    + lambda_entropy * loss_entropy
)
```

Recommended schedule:

```text
beta_kl: 0.0 -> 0.5 warmup
lambda_prior: 0.5 early -> 1.0
lambda_usage: 0.01
lambda_entropy: 0.01 early -> 0.0
alpha_direct: 0.0 -> max 0.25
```

### Required metrics

Report:

```text
BCE
AUROC
AUPRC
accuracy at threshold 0.5
balanced accuracy
Brier score
mean predicted probability by fine label
3x2 diagnostic table
latent KL by group
latent entropy by group
latent category usage by group
anchor null rate
target null rate
prior/posterior agreement by group
```

## 12. Ablations

### Required ablations

| Ablation | Description | Purpose |
|---|---|---|
| `no_latent_matched` | Same trunk, parameter-matched direct head, no latent variables | Required baseline |
| `latent_no_freebits` | Remove free bits | Tests posterior-collapse control |
| `latent_no_prior_pred` | Remove prior-path prediction loss | Tests train/inference gap |
| `latent_no_usage` | Remove batch usage loss | Tests category collapse |
| `latent_no_entropy_floor` | Remove early entropy floor | Tests premature hard assignment |
| `latent_no_square` | Remove anchor and target latents | Tests localization value |
| `latent_no_motif` | Remove motif latent | Tests motif abstraction |
| `latent_no_relation` | Remove relation latent | Tests geometric/dependency relation |
| `latent_no_vulnerability` | Remove vulnerability latent | Tests defender/king weakness modeling |
| `latent_no_tempo` | Remove tempo bucket | Tests forcing-depth abstraction |
| `latent_prior_only_train` | Train without posterior path | Tests value of variational inference |
| `latent_direct_alpha_1` | Let direct path have full strength | Tests latent bypass |
| `latent_soft_only` | Use expected embeddings only, no hard samples | Tests sampling need |

### No-latent matched baseline

The baseline must be serious, not intentionally weak.

Architecture:

```text
board [B, C, 8, 8]
-> same ChessBoardTrunk
-> same pooling
-> direct MLP head
-> logit [B]
```

Matching rules:

```text
same input tensor
same target mapping
same splits
same optimizer
same learning-rate schedule
same batch size
same maximum epochs
same early-stopping rule
same validation cadence
same metrics
same 3x2 diagnostic
inference-time parameter count within +/- 3 percent when practical
FLOPs within +/- 5 percent when practical
```

Parameter matching method:

1. Count TSBI inference-time parameters, excluding posterior-only modules.
2. Choose direct-head width for the baseline to match that count.
3. If exact matching is impossible, use the closest baseline that is not smaller.
4. Do not add unused dummy parameters.

### Acceptance standard

TSBI is promising only if:

```text
it beats no_latent_matched on validation AUROC or AUPRC across repeated seeds
it does not damage the 3x2 diagnostic, especially fine_label_1
it has nonzero KL in multiple latent groups
it has nontrivial motif/relation usage
anchor and target are not always null
prior-path performance is close to posterior-path performance
```

## 13. Falsification

Reject TSBI if any of the following hold after reasonable tuning.

### Predictive falsification

```text
no_latent_matched matches or beats TSBI across repeated seeds
TSBI improves training loss but not validation AUROC/AUPRC
TSBI worsens Brier score with no compensating ranking gain
TSBI only works on one split or one seed
```

### Latent-collapse falsification

```text
KL is approximately zero for all latent groups
motif is nearly always null_or_quiet
anchor_square is nearly always null
target_square is nearly always null
relation is nearly always none
posterior latents are useful but prior latents are not
latent assignments are unstable across seeds
```

### Chess-plausibility falsification

Use board-only diagnostics. Do not use these diagnostics as training labels unless a new experiment explicitly permits it.

Reject if:

```text
anchor squares rarely correspond to active pieces or tactical neighborhoods
target squares rarely correspond to kings, valuable pieces, defenders, or contested squares
line_relation latents ignore line geometry
knight_or_leaper latents ignore leaper geometry
piece perturbations at inferred anchor/target squares affect the logit less than unrelated perturbations
```

### Leakage falsification

Reject the run immediately if any forbidden input reaches:

```text
dataset item
collated batch
model forward
loss function
metric function
split logic
ablation selection
```

### Decision rule

Do not rescue a falsified TSBI with calibration tricks, larger direct heads, or extra metadata. If the latent object is not predictive and used, mark the approach as failed.

## 14. Codex Implementation Plan

### Repository files to add during implementation

This handoff creates only this Markdown artifact. The following are implementation targets for the repo:

```text
models/tactical_state_bottleneck.py
models/no_latent_matched_baseline.py
losses/tactical_state_losses.py
metrics/puzzle_binary_diagnostics.py
configs/model_tactical_state_bottleneck.yaml
configs/model_no_latent_matched.yaml
tests/test_tactical_state_bottleneck.py
tests/test_puzzle_binary_data_contract.py
```

### Step 1: Add data-contract guard

```python
def build_puzzle_binary_target(batch):
    bad_keys = set(batch.keys()) & FORBIDDEN_KEYS
    assert not bad_keys, f"Forbidden leakage keys in batch: {sorted(bad_keys)}"

    board = batch["board"]
    fine = batch["fine_label"].long()

    assert board.ndim == 4
    assert board.shape[-2:] == (8, 8)
    assert fine.ndim == 1
    assert fine.shape[0] == board.shape[0]
    assert int(fine.min()) >= 0
    assert int(fine.max()) <= 2

    y = (fine == 2).float()
    return board, fine, y
```

### Step 2: Implement categorical latent helper

```python
class CategoricalLatent(nn.Module):
    def __init__(self, num_categories: int, emb_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(num_categories, emb_dim) * 0.02)

    def expected_embedding(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = logits.softmax(dim=-1)
        return probs, probs @ self.embedding

    def sample_embedding(self, logits: torch.Tensor, temperature: float, hard: bool):
        z = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        return z, z @ self.embedding
```

### Step 3: Implement square-logit helper

```python
class SquareLogitHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.spatial = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.null_logit = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b = h.shape[0]
        spatial = self.spatial(h).flatten(start_dim=1)  # [B, 64]
        null = self.null_logit.expand(b, 1)
        return torch.cat([spatial, null], dim=-1)        # [B, 65]
```

### Step 4: Implement prior and posterior heads

```python
class PriorTacticalHead(nn.Module):
    def forward(self, h, pooled):
        return {
            "motif": self.motif(pooled),
            "anchor": self.anchor(h),
            "target": self.target(h),
            "relation": self.relation(pooled),
            "vulnerability": self.vulnerability(pooled),
            "tempo": self.tempo(pooled),
        }
```

```python
class PosteriorTacticalHead(nn.Module):
    def forward(self, h, pooled, y_long):
        y_emb = self.y_embedding(y_long)
        pooled_post = torch.cat([pooled, y_emb], dim=-1)
        y_map = y_emb[:, :, None, None].expand(-1, -1, 8, 8)
        h_post = self.spatial_fuse(torch.cat([h, y_map], dim=1))
        return {
            "motif": self.motif(pooled_post),
            "anchor": self.anchor(h_post),
            "target": self.target(h_post),
            "relation": self.relation(pooled_post),
            "vulnerability": self.vulnerability(pooled_post),
            "tempo": self.tempo(pooled_post),
        }
```

### Step 5: Implement TSBI forward paths

```python
class TacticalStateBottleneckModel(nn.Module):
    def forward_train(self, board, fine_label):
        y_long = (fine_label == 2).long()
        h = self.trunk(board)
        pooled = self.pool(h)

        prior_logits = self.prior_head(h, pooled)
        posterior_logits = self.posterior_head(h, pooled, y_long)

        zq_probs, zq_emb = self.project_latents(posterior_logits, sample=self.training)
        zp_probs, zp_emb = self.project_latents(prior_logits, sample=self.training)

        logit_q = self.compute_logit(pooled, zq_emb)
        logit_p = self.compute_logit(pooled, zp_emb)

        return {
            "logit": logit_p,
            "logit_q": logit_q,
            "logit_p": logit_p,
            "prior_logits": prior_logits,
            "posterior_logits": posterior_logits,
            "prior_probs": zp_probs,
            "posterior_probs": zq_probs,
        }

    def forward_eval(self, board, return_latents=False):
        h = self.trunk(board)
        pooled = self.pool(h)
        prior_logits = self.prior_head(h, pooled)
        prior_probs, z_emb = self.project_latents(prior_logits, sample=False)
        logit = self.compute_logit(pooled, z_emb)
        out = {"logit": logit}
        if return_latents:
            out["prior_logits"] = prior_logits
            out["prior_probs"] = prior_probs
        return out
```

### Step 6: Implement loss

```python
def tactical_state_loss(outputs, fine_label, schedule):
    y = (fine_label == 2).float()

    loss_pred = F.binary_cross_entropy_with_logits(outputs["logit_q"], y)
    loss_prior_pred = F.binary_cross_entropy_with_logits(outputs["logit_p"], y)
    loss_kl, kl_by_group = kl_freebits(
        outputs["posterior_logits"],
        outputs["prior_logits"],
        schedule.free_bits,
    )
    loss_usage = batch_usage_loss(outputs["posterior_logits"])
    loss_entropy = entropy_floor_loss(outputs["posterior_logits"], schedule.entropy_floor)

    loss = (
        loss_pred
        + schedule.beta_kl * loss_kl
        + schedule.lambda_prior * loss_prior_pred
        + schedule.lambda_usage * loss_usage
        + schedule.lambda_entropy * loss_entropy
    )

    return {
        "loss": loss,
        "loss_pred": loss_pred.detach(),
        "loss_prior_pred": loss_prior_pred.detach(),
        "loss_kl": loss_kl.detach(),
        "loss_usage": loss_usage.detach(),
        "loss_entropy": loss_entropy.detach(),
        "kl_by_group": kl_by_group,
    }
```

### Step 7: Implement required diagnostic

```python
def puzzle_binary_3x2(fine_label: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
    pred = (logit > 0).long()
    fine = fine_label.long().clamp(0, 2)
    table = torch.zeros(3, 2, dtype=torch.long, device=logit.device)
    table.view(-1).scatter_add_(0, fine * 2 + pred, torch.ones_like(fine, dtype=torch.long))
    return table
```

Also log:

```python
prob = logit.sigmoid()
mean_prob_by_fine = {k: prob[fine_label == k].mean() for k in [0, 1, 2]}
```

### Step 8: Implement no-latent matched baseline

```python
class NoLatentMatchedBaseline(nn.Module):
    def __init__(self, trunk_config, head_width):
        super().__init__()
        self.trunk = ChessBoardTrunk(**trunk_config)
        self.pool = GlobalPool()
        self.head = nn.Sequential(
            nn.Linear(trunk_config["dim"], head_width),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_width, head_width),
            nn.GELU(),
            nn.Linear(head_width, 1),
        )

    def forward(self, board):
        h = self.trunk(board)
        pooled = self.pool(h)
        return {"logit": self.head(pooled).squeeze(-1)}
```

### Step 9: Add tests

Minimum tests:

```text
test_binary_target_mapping
test_forbidden_keys_rejected
test_tsbi_logit_shape
test_prior_logit_shapes
test_posterior_logit_shapes
test_eval_does_not_require_fine_label
test_3x2_known_values
test_no_latent_baseline_logit_shape
test_training_output_contains_logit_q_and_logit_p
```

### Step 10: Run experiment matrix

Run at least:

```text
no_latent_matched
TSBI full
latent_no_freebits
latent_no_prior_pred
latent_no_usage
latent_no_square
latent_no_motif
latent_direct_alpha_1
```

For each run, report repeated-seed mean and standard deviation for predictive metrics, plus latent diagnostics for TSBI variants.

## 15. Prompt Updates

Use this prompt for the Codex implementation pass:

```text
Implement a board-only latent-variable classifier for puzzle_binary named TacticalStateBottleneckModel.

Input: current-board tensor [B, C, 8, 8].
Output: one binary puzzle logit [B].
Training target: fine labels 0 and 1 map to 0; fine label 2 maps to 1.

Do not use engine scores, PVs, node counts, mate scores, best moves, solution moves, verification metadata, source labels, source provenance, or any field derived from them. Add a batch guard that rejects forbidden keys.

The latent variable must be chess-specific tactical state:
z = (motif, anchor_square, target_square, relation, vulnerability, tempo_bucket).
Use categorical latents with prior p(z|x) and training-only posterior q(z|x,y). Inference must use only p(z|x). Do not reconstruct the board. Do not implement a generic VAE. Do not implement denoising score fields, class-conditioned pseudo-likelihood, masked codecs, evidential heads, mixture-of-experts calibration, or prototype margins.

Include posterior-collapse controls:
- KL warmup
- free bits by latent group
- prior-path prediction loss
- batch latent usage regularizer
- early entropy floor
- capped direct logit path

Implement a no-latent matched baseline with the same board trunk and a parameter-matched direct head. Use the same data, optimizer, schedule, metrics, and 3x2 diagnostic.

Required 3x2 diagnostic: rows are fine labels {0,1,2}; columns are predicted binary {0,1}; prediction threshold is logit > 0.

Required metrics: BCE, AUROC, AUPRC, balanced accuracy, Brier score, mean predicted probability by fine label, 3x2 diagnostic, latent KL by group, latent entropy by group, latent category usage, anchor null rate, target null rate, and prior/posterior agreement.

Falsify the approach if it fails to beat the no-latent matched baseline or if the latent variables collapse, remain unstable across seeds, or fail board-only chess-plausibility diagnostics.
```
