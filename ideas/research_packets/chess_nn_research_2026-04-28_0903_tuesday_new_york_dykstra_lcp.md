# Codex Handoff Packet: Soft-Dykstra Latent Constraint Projector

## 1. File Metadata

- Project: `chess-nn-playground`
- Task: `puzzle_binary`
- Idea slug: `dykstra_lcp`
- Filename: `chess_nn_research_2026-04-28_0903_tuesday_new_york_dykstra_lcp.md`
- Proposed module path: `models/puzzle_binary/dykstra_lcp.py`
- Model name: **Soft-Dykstra Latent Constraint Projector**
- Central mechanism: a finite unrolled **Dykstra projection solver** over board-conditioned latent convex constraints.
- Data contract: input tensor `x` has shape `(batch, C, 8, 8)`; output is exactly one logit per position.
- Inference contract: `forward(x)` receives no engine evaluations, PVs, node counts, mate scores, verification metadata, source labels, best moves, source identity, or fine labels.
- Fine-label contract: optional labels in `{0,1,2}` may be present in loaders only for diagnostic reporting. They must not be passed into the model, used for loss weights, used for sampling, used for thresholds, or used for inference-time branching.
- Duplicate guard: this is not an obligation-resource flow, not scheduling, not an opportunity-cost auction, not min-cut, not Hall defect, not proof-number search, not matrix-game equilibrium, not a legal move tree, and not boundary-edit energy.

## 2. Executive Selection

Build **Dykstra-LCP**, a classifier that treats a verified puzzle position as one whose board-only neural features can be projected into a compact, internally consistent latent tactical certificate. The core signal is not a supervised certificate; no such certificate is available. The core signal is the **projection trace**: how far the encoder’s proposed latent structure must move before satisfying a family of learned convex constraints.

This fits `puzzle_binary` because near-puzzles often contain convincing local tactical texture but fail global coherence. A plain CNN can see the local motif and overfire. Dykstra-LCP asks a stricter question: can the current board be organized into a small set of mutually compatible latent roles, relations, budgets, and closure constraints without excessive correction? Verified puzzles should require small correction and show fast residual contraction. Hard negatives should look locally plausible but force the solver to spend many corrections, activate slack, or converge to diffuse explanations.

The proposal is intentionally board-only. It does not predict the best move, inspect a continuation, or consume verification artifacts. It learns a differentiable latent feasibility prior trained end-to-end from the binary label.

## 3. Problem And Data Contract

The model solves binary classification.

- Input: `x ∈ R^(B,C,8,8)`, containing only current-position tensor planes already allowed by the project.
- Output: `logit ∈ R^(B,)` or `R^(B,1)`.
- Positive class: verified puzzle.
- Negative class: non-puzzle and near-puzzle positions.
- Hard-negative handling: use online hard-negative mining from binary negatives by current loss, logit, or residual score. Do not use fine labels to identify hard negatives.

The optional fine labels `{0,1,2}` are diagnostic labels only. Use them to print validation tables, measure false positives on near-puzzle slices, and inspect residual histograms. Never feed them to `forward`, never concatenate them to features, never use them as auxiliary targets, and never use them to form minibatches.

The model must not infer from source identity. This includes direct source fields, loader-order artifacts, puzzle IDs, verification flags, engine side channels, or file-origin metadata. The only semantic input is the board tensor.

## 4. Constraint-Solver Research Background

Differentiable optimization layers are useful when a model needs more than pattern recognition but cannot use explicit symbolic supervision. They let a network propose latent variables, then force those variables through a structured solver whose residuals become learnable evidence. For this task, the solver should encode consistency without becoming a move-search system.

Dykstra’s projection algorithm projects a point onto an intersection of convex sets by cycling through individual projection operators while maintaining correction buffers. Unlike a single naive alternating-projection pass, the correction buffers preserve information about earlier constraints and produce a meaningful trace of where the proposed latent certificate conflicts with the feasible family.

Dykstra-LCP uses a finite unroll, typically 6 to 12 cycles, so the entire solver remains a normal differentiable computation graph. The constraint sets are simple: boxes, simplexes, affine role-budget constraints, halfspace compatibility constraints, and sparse relation-closure constraints. These have cheap projection operators and can be implemented directly in PyTorch without calling an external optimizer.

The research idea is that the solver trace is class evidence. A verified puzzle is hypothesized to be close to a latent feasible certificate. A near-puzzle can have high local motif activation but still be far from the feasible intersection because its roles, supports, and closures cannot be made mutually consistent without large corrections.

## 5. Serious Candidates Rejected

| Candidate | Why it was rejected |
|---|---|
| Dense QP layer | Too expensive once relation variables are included, and too opaque for diagnosing which constraints separated near-puzzles. |
| Sinkhorn transport over motif slots | Attractive for assigning squares to slots, but too assignment-centric. It can reward tidy matching even when global certificate coherence is absent. |
| Differentiable dynamic program over templates | Too close to a hidden continuation recognizer. The task needs a board-only classifier, not a sequence parser. |
| Neural fixed-point consistency layer | Flexible, but harder to falsify. A gain could come from recurrent depth rather than constraint satisfaction. |
| Learned clause relaxation | Too binary and adjacent to a solver family explicitly listed in the prompt. Dykstra projections provide smoother real-valued residual traces. |
| Semidefinite relation relaxation | Theoretically appealing for global consistency, but memory-heavy and unnecessary for the first implementation. |

## 6. Common Approaches Rejected

- **Plain CNN or ViT classifier only:** useful baseline, but it does not directly test the latent-constraint hypothesis and is likely to overvalue local tactical texture in near-puzzles.
- **Engine-proxy classifier:** forbidden by the data contract and scientifically weak, because it would classify verification residue rather than board-only puzzle structure.
- **Best-move or continuation prediction:** rejected because inference must not receive best moves, and the model should not become a move-tree learner.
- **Source or metadata classifier:** rejected because it would leak dataset construction rather than learn the position property.
- **Handwritten tactical detector:** rejected because it would be brittle, hard to maintain, and likely to smuggle in assumptions unavailable as labels.
- **Auxiliary fine-label target:** rejected because the fine labels are diagnostics only and must not shape the learned representation.

## 7. Mathematical Thesis

Let `X` be the board tensor. The encoder produces a raw latent certificate proposal and board-conditioned constraint parameters:

`(e, z0, ψ) = Encoder_θ(X)`

Here `e` is a global board embedding, `z0` packs latent role, relation, motif, and slack variables, and `ψ` defines convex sets that may depend on the board tensor but not on forbidden inference inputs.

Define a board-conditioned feasible family:

`K_ψ(X) = C_box ∩ C_simplex ∩ C_budget(X) ∩ C_compat(X) ∩ C_closure(X) ∩ C_slack`

The solver computes an approximate projection:

`z* ≈ Π_K(z0) = argmin_z 0.5 ||z - z0||_2^2 subject to z ∈ K_ψ(X)`

The binary logit is:

`ℓ = Head_φ(e, z0, z*, z* - z0, Trace_Dykstra(z0, K_ψ))`

The thesis:

- verified puzzles are close to at least one compact latent feasible certificate;
- near-puzzles often satisfy local constraints but violate closure, budget, or compatibility constraints;
- projection distance, correction buffers, slack activations, and residual decay form a stronger hard-negative signal than raw convolutional features alone.

This is falsifiable. If the solver trace carries no separable information beyond an equally sized encoder, the idea should be abandoned.

## 8. Latent Constraint System

Use a compact certificate with four variable groups.

**Role mass `U`**

Shape: `(B, R, 64)`, with `R ≈ 10`.

Suggested roles:

- `focal_king`
- `target_piece`
- `trigger_square`
- `attacker_support`
- `defender`
- `blocker`
- `escape_square`
- `line_square`
- `tension_square`
- `noise`

These are latent roles, not supervised chess annotations. The names are implementation handles and should not require target labels.

**Relation mass `V`**

Shape: `(B, A, 64, 64)`, with `A ≈ 6`, optionally stored block-sparse using fixed geometric masks.

Suggested relation channels:

- `pressure`
- `shield`
- `alignment`
- `pin_like_support`
- `escape_cover`
- `local_contact`

The relation tensor is not a move tree. It is a soft board-geometry relation graph over the current position only. Use fixed ray, knight-neighborhood, king-neighborhood, and pawn-direction masks as board geometry gates; do not enumerate continuations.

**Motif mixture `M`**

Shape: `(B, K)`, with `K ≈ 12`, constrained to a simplex. Each motif component defines a learned budget vector for roles and relation types. The mixture lets the model represent multiple puzzle-like certificate families without hard-coding names such as fork, pin, or mate net.

**Slack and trace variables `S`**

Shape: `(B, G)` for grouped violations. Slack is allowed only as an explicit measured failure channel. It should be bounded, nonnegative, and exposed to the readout. Positive examples should learn to use little slack; negatives may require more.

Recommended constraint families:

1. **Box constraints:** `0 ≤ U,V,S ≤ 1` and nonnegative motif mass.
2. **Simplex constraints:** motif mixture sums to one; selected role groups may have normalized mass budgets.
3. **Board compatibility constraints:** roles that require occupancy, kings, empty squares, friendly pieces, or opposing pieces are upper-bounded by masks derived from `X`.
4. **Role-budget constraints:** `lower_r(M) ≤ Σ_s U[r,s] ≤ upper_r(M)`, where bounds are linear functions of motif mixture `M`.
5. **Relation-compatibility constraints:** relation mass `V[a,i,j]` is upper-bounded by geometry masks and compatible endpoint role mass.
6. **Closure constraints:** active targets, focal regions, and support roles must be explained by compatible relations. Example: `Σ_j V[pressure,j,i] + S_g ≥ U[target_piece,i]`.
7. **Mutual-exclusion constraints:** incompatible roles on the same square cannot both carry high mass.
8. **Compactness constraints:** active non-noise role mass is capped by a motif-dependent budget, preventing diffuse explanations.
9. **Turn/color covariance constraints:** role compatibility can depend on side-to-move and piece color planes from `X`, but not on future moves or engine facts.

All constraints should be boxes, simplexes, affine equalities, or affine inequalities after conditioning on `X`. That keeps each projection cheap and interpretable.

## 9. Differentiable Solver

Use an unrolled Dykstra projector as the central forward-pass mechanism.

Pack all variables into one vector per batch item:

`z = pack(U, V, M, S)`

Let the constraint groups be `C_1, ..., C_G`, each with a differentiable projection operator `Π_g(·; X)`. Initialize correction buffers `q_g = 0`. For `T` cycles:

```text
z = z0
for t in 1..T:
    for g in 1..G:
        y = z + q_g
        z_next = Project_g(y, X)
        q_g = y - z_next
        record pre_violation_g(y), correction_norm_g(||q_g||), step_norm_g(||z_next - z||)
        z = z_next
z_star = z
```

Projection operators:

- **Box projection:** clamp to `[0,1]` or `[0,S_max]` for slack.
- **Simplex projection:** exact Euclidean simplex projection for `M`; optionally use a temperature-smoothed approximation if early gradients are noisy.
- **Affine equality projection:** `z ← z - Aᵀ(AAᵀ + εI)^(-1)(Az - b)` for small block-local matrices.
- **Affine inequality projection:** for row `aᵀz ≤ b`, if violated, project by `z ← z - α a(aᵀz-b)/||a||²`; use a smooth active factor during training if hard branching causes instability.
- **Masked relation projection:** combine box projection with fixed sparse geometry masks so forbidden relation entries are projected to zero.

Readout features from the solver:

- final projected variables `z*`;
- correction vector `Δ = z* - z0`;
- per-group correction norms;
- per-cycle residual decay;
- final slack activations;
- ratio of closure residual to budget residual;
- number of groups still violated before the last projection pass.

The solver is not merely a regularizer. It must sit inside `forward`, and the logit must receive solver outputs.

## 10. Architecture Tensor Contract

Recommended first implementation:

```text
Input x:                 (B, C, 8, 8)
BoardStem:               Conv/Residual blocks -> F: (B, 128, 8, 8)
Global embedding:        e: (B, 256)
RoleInitHead:            U0 logits -> (B, R, 64)
RelationInitHead:        V0 logits -> (B, A, 64, 64), optionally sparse-masked
MotifInitHead:           M0 logits -> (B, K)
SlackInitHead:           S0 raw -> (B, G)
ConstraintParamHead:     board-only masks, role budgets, relation gates
DykstraProjector:        (U0,V0,M0,S0,ψ) -> (U*,V*,M*,S*,trace)
ReadoutHead:             concat(e, summaries(z0), summaries(z*), trace) -> (B,1)
Output:                  one raw logit per sample
```

Suggested constants:

- `R = 10` role channels.
- `A = 6` relation channels.
- `K = 12` motif components.
- `G = 8 to 14` grouped constraint families.
- `T = 8` Dykstra cycles for the main model; `T = 2,4,12` for ablation.
- relation storage can begin dense for simplicity, then move to sparse masks if memory becomes the bottleneck.

API requirement:

```python
class DykstraLCP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 8, 8)
        # returns logits: (B,) or (B, 1)
        ...
```

Do not add `fine_label`, `source`, `best_move`, `engine_eval`, or `metadata` arguments.

## 11. Training Objective

Use binary supervision only.

Base loss:

`L_bce = BCEWithLogits(logit, y_binary)`

Hard-negative emphasis without fine labels:

- include all positives;
- for binary negatives, upweight or select the top `q%` by current loss or predicted positive logit;
- do not use fine labels to identify near-puzzles.

Solver-aware residual terms, using only the binary label:

Let

`R_proj = ||z* - z0||_1 / sqrt(dim(z))`

and

`R_trace = mean_g,t pre_violation_g(t)`

A practical auxiliary term is:

`L_res = λ_pos y (R_proj + R_trace) + λ_neg (1-y) relu(μ - R_proj)^2`

This encourages positives to be close to feasible certificates and prevents all negatives from being silently projected into plausible certificates. Keep `λ_neg` modest so ordinary negatives are not overconstrained.

Trace-shape stabilizer:

`L_decay = λ_decay mean_t relu(R_after[t+1] - R_after[t])`

Total:

`L = L_bce_hard_negative + L_res + L_decay + λ_wd ||θ||²`

Training diagnostics using fine labels are allowed only after computing logits and losses. Recommended diagnostic tables:

- ROC-AUC by fine-label slice;
- false-positive rate on the near-puzzle diagnostic slice;
- mean projection distance by fine-label slice;
- residual histogram overlap between verified puzzles and near-puzzles;
- calibration error for binary labels, stratified by fine label only in reports.

Safe augmentations:

- file-mirror symmetry when tensor encoding supports it;
- color/side normalization if already used by the project;
- small input dropout on non-critical auxiliary planes, never on labels or metadata.

## 12. Ablation Matrix

| ID | Variant | What it tests | Expected useful result |
|---|---|---|---|
| A0 | Parameter-matched CNN/ViT baseline | Whether the solver adds anything beyond capacity | Dykstra-LCP wins on near-puzzle false positives and PR-AUC. |
| A1 | Encoder plus raw latent heads, no projection | Whether latent variables alone help | Should underperform full solver. |
| A2 | Dykstra with `T=1` | Whether one projection sweep is enough | Should be weaker than `T=6..12`. |
| A3 | Dykstra without correction buffers | Dykstra versus naive alternating projections | Should lose residual-trace quality. |
| A4 | Remove closure constraints | Whether relation-role closure matters | Near-puzzle false positives should increase. |
| A5 | Remove compactness budgets | Whether diffuse certificates are a failure mode | Busy non-puzzles may be classified as puzzles. |
| A6 | Remove relation tensor `V` | Whether role-only feasibility is sufficient | Should reduce global-coherence performance. |
| A7 | Hide solver trace from readout | Whether trace features are the signal | Should reduce hard-negative separation. |
| A8 | Disable residual auxiliary terms | Whether BCE alone learns the feasibility prior | Full model should calibrate better and use less slack. |
| A9 | Online hard-negative mining off | Hard-negative treatment without fine-label leakage | Mining should reduce diagnostic near-puzzle false positives. |
| A10 | Shuffle board-conditioned compatibility masks | Leakage and sanity test | Performance should collapse or clearly degrade. |
| A11 | Randomize constraints after training | Whether constraints are used rather than ignored | Logits and residual diagnostics should change materially. |

## 13. Minimal Benchmark Plan

Use the existing `puzzle_binary` split. Keep every model on the same allowed input tensor and the same binary labels.

Core baselines:

1. current project baseline for `puzzle_binary`;
2. parameter-matched CNN or small ViT;
3. encoder with latent heads but no Dykstra projection;
4. Dykstra-LCP full model.

Primary metrics:

- binary ROC-AUC;
- binary PR-AUC;
- BCE / log loss;
- Brier score;
- expected calibration error;
- false-positive rate at fixed recall targets;
- precision at the operating point the project actually uses.

Diagnostic metrics using fine labels only after inference:

- false-positive rate on near-puzzles;
- mean projection distance for each fine-label slice;
- slack activation distribution by slice;
- residual contraction curves by slice.

Minimal run protocol:

- train three random seeds for each core model;
- keep parameter counts comparable;
- log throughput and peak memory;
- freeze all forbidden fields out of the dataloader batch consumed by `forward`;
- run an audit proving that replacing fine labels with random values leaves training unchanged except for diagnostic table names.

Acceptance target:

- consistent PR-AUC gain over the parameter-matched baseline;
- materially lower false-positive rate on diagnostic near-puzzles at the same binary recall;
- visible separation in projection-distance histograms between verified puzzles and near-puzzles.

## 14. Falsification Criteria

Abandon or demote Dykstra-LCP if any of the following hold after a fair implementation:

- A parameter-matched non-solver baseline matches or beats the full model across three seeds on PR-AUC and near-puzzle false-positive rate.
- Projection distance `||z* - z0||` has binary-label AUC below `0.55`, indicating that latent feasibility is not producing signal.
- Removing Dykstra correction buffers has no measurable effect.
- Hiding the solver trace from the readout has no measurable effect.
- Fine-label diagnostic near-puzzles are not separated from positives better than by the plain baseline.
- Training is unstable: frequent NaNs, exploding correction buffers, or more than 20 percent of batches with saturated slack after reasonable gradient clipping.
- A leakage audit finds dependence on source identity, verification metadata, engine-derived fields, best-move fields, or fine labels.
- Constraint randomization after training barely changes logits, proving the constraint system is decorative.

A negative result is still useful if it shows that board-only latent feasibility is weaker than direct representation learning for this dataset.

## 15. Prompt-Maintenance Notes

Keep future iterations centered on **Dykstra projection over latent convex constraints**. Do not drift into a different solver family unless the packet is intentionally replaced.

Do not add any inference inputs beyond `(batch, C, 8, 8)`. In particular, do not add engine evaluations, PVs, node counts, mate scores, verification metadata, source labels, source identity, puzzle IDs, best moves, or fine labels.

Do not use fine labels for loss design, hard-negative selection, class weights, curricula, model routing, or thresholds. They exist only to diagnose whether binary training handles ordinary negatives and near-puzzles differently.

Preserve the hard-negative idea by online mining from binary negatives, not by using diagnostic class IDs.

Preserve the duplicate boundary: the model should remain a latent projection-feasibility classifier, not an obligation-resource flow, not scheduling, not an opportunity-cost auction, not min-cut, not Hall defect, not proof-number search, not matrix-game equilibrium, not a legal move tree, and not boundary-edit energy.

The implementation should be easy to delete if falsified: one model file, one projector module, one benchmark entry, and no changes to dataset semantics.
