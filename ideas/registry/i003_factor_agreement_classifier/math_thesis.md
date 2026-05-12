# Mathematical Thesis

## Actual Task And Labels

The first test is the corrected puzzle-binary benchmark:

```text
y = 0 for random/non-puzzle and verified near-puzzle
y = 1 for verified puzzle
```

The architecture is general for chess position classification and should support other binary or multi-class labels later.

Allowed inference inputs are deterministic views of the current position. Forbidden inputs include engine scores, PVs, nodes, source labels, verification metadata, and source file identity.

## Baseline And Weakness

Closest existing ideas:

- Piece-Token CNN Hybrid
- Cross-stitch CNN-token fusion
- Agreement-variance heads
- Critical-Square Budget Network
- simple CNN and LC0 BT4 tower

Overlap: this idea also combines multiple board views and logs disagreement.

Difference: this idea makes agreement mathematically part of the classification logit: the main evidence is penalized by factor disagreement, so a single branch cannot silently dominate.

## Definitions

Construct four deterministic views of the same position:

```text
g: grid/square texture view
p: occupied piece-token view
r: relation-geometry view
m: material/phase/global context view
```

Each branch emits:

```text
z_i = encoder_i(view_i)
e_i = evidence_head_i(z_i)
u_i = uncertainty_head_i(z_i)
```

Compute agreement:

```text
e_bar = mean_i e_i
D = mean_i (e_i - e_bar)^2
U = mean_i softplus(u_i)
f(x) = e_bar - alpha * D - beta * U + residual_joint_head([z_g,z_p,z_r,z_m])
```

The residual joint head is initialized small so agreement drives early learning.

## Assumptions

- Real chess class signal should appear in more than one representation factor.
- Source artifacts and superficial near-puzzle texture are more likely to be view-specific.
- Penalizing disagreement will improve calibration and hard-negative behavior.

## Claim

Hypothesis: a factor-agreement classifier should improve hard-negative classification because near-puzzles and artifacts often activate one factor strongly while failing to produce consistent evidence across piece, relation, and global context views.

## Mechanism

The architecture does not simply concatenate features. It requires the branches to agree on class evidence. Disagreement becomes a negative term or uncertainty term in the logit.

This is general for classification: any label whose signal is stable across multiple deterministic views should benefit.

## Proof Sketch

What can be reasoned about:

- If one factor is spuriously high and others are low, the disagreement penalty reduces the final logit.
- If all factors agree, the penalty is small and evidence passes through.
- Factor-drop ablations can identify which view carries the signal.

## Not Proven

- That puzzle labels are stable across the chosen factors.
- That the disagreement penalty will not suppress legitimate single-factor tactics.
- That the residual joint head will not learn to bypass the bottleneck unless constrained.

## Counterexamples

- A valid class may genuinely depend on one view only, such as a pure material-count label.
- Rare tactical motifs may appear in relation geometry but not global context.
- If all branches learn the same shortcut, agreement will not protect against it.

## Falsification Test

Train against size-matched fusion baselines. Revise or reject if:

```text
agreement penalty does not improve PR AUC or near-puzzle FP
and factor disagreement is not higher on near-puzzle mistakes than on correct puzzle predictions
```

Reject the bottleneck if a plain concatenation model matches it across two seeds.

