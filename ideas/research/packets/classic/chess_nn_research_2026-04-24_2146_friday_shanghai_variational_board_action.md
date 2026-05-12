# Codex Research Packet: Variational Board Action Network

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2146_friday_shanghai_variational_board_action.md`
- Generated at: 2026-04-24 21:46
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: full architecture packet, not implemented

## One-Sentence Thesis

Use calculus of variations on learned chess board fields: define a differentiable action functional over the current board, compute its discrete Euler-Lagrange residual, and classify puzzle-likeness from localized variational disequilibrium rather than from raw CNN features alone.

## Why This Is A Calculus Approach

The central object is an action:

```text
A[u; x] = sum_s L_theta(u(s), grad u(s), x(s))
```

where:

- `x` is the current board tensor
- `u` is a learned board field derived from `x`
- `grad u` is a finite-difference gradient over adjacent squares
- `L_theta` is a learned Lagrangian density

The model computes the Euler-Lagrange residual:

```text
R = dL/du - div(dL/d(grad u))
```

on the finite `8 x 8` board. The hypothesis is that puzzle-like positions produce localized high residuals because their learned tactical fields are not near equilibrium under the ordinary board action.

This is not just using gradients during training. The variational residual is an explicit model feature.

## Data Contract

Task:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels:

- `0`, `1`, and `2` remain diagnostics only
- train on binary labels
- report the fine-label `3 x 2` diagnostic matrix

First implementation:

- input: `simple_18`
- dataset: existing `crtk_sample_3class` splits
- trainer: shared experimental training pipeline

Forbidden model inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Engine search, forced-line search, legal mate/stalemate oracles, tablebase outcomes, or future game outcomes.

Allowed model inputs:

- Current board occupancy.
- Side-to-move.
- Castling/en-passant planes already in the input.
- Deterministic square coordinates.
- Side-relative coordinates.
- Current-board masks such as occupied squares, empty squares, king zones, and piece-type planes.

## Core Abstraction

Chess positions can be viewed as fields over a small square lattice:

```text
own force field
opponent force field
king danger field
blocker stiffness field
mobility proxy field
empty-space tension field
```

The model does not hand-code those fields. It learns them from `simple_18`:

```text
u = FieldEncoder(x)
```

Then it asks:

```text
If this board field were governed by a learned ordinary-board action, where is it out of equilibrium?
```

That out-of-equilibrium residual is the bottleneck.

## Why It May Fit Chess

Many tactical patterns are local violations of a smoother strategic picture:

- a pinned piece locally appears useful but globally cannot resolve line tension
- a king zone has sharp residual pressure
- a sacrifice creates a localized discontinuity in force balance
- a defender is overloaded because multiple field gradients meet there
- an ordinary-looking position becomes puzzle-like because one square has unusually high variational defect

This approach does not need a legal move tree. It searches for learned equilibrium defects in current-board fields.

## Why It Is Not A Duplicate

| Existing family | Why this differs |
|---|---|
| Harmonic Board Potential | Harmonic potential solves a fixed Poisson/Green system. This packet learns an action density and computes Euler-Lagrange residuals; no inverse Laplacian solve is central. |
| Score-field / curl-divergence variants | Those use denoising repair fields or vector-field summaries. This packet differentiates a learned action functional with respect to board fields. |
| Hodge/sheaf packets | No attack graph, cochains, sheaf restrictions, or graph Laplacian. |
| Cubical Euler topology | No Euler characteristic, Betti curves, or topological sweeps. |
| Schur-Ray Line Algebra | No line-incidence Woodbury solve. |
| Bitboard Shift-Algebra | No fixed movement-shift polynomial bank. |
| Differentiable Boolean / tropical circuits | No logical clause or bitboard predicate circuit. |
| Ordinary CNN | The Euler-Lagrange residual is an explicit calculus-derived feature, not just learned convolution. |

## Mathematical Derivation

### Continuous Motivation

For a scalar field `u(p)` on a domain, define:

```text
A[u] = integral L(u(p), grad u(p), p) dp
```

The stationary condition under small perturbations `u + eps v` is:

```text
d/deps A[u + eps v] at eps=0 = 0
```

After integration by parts, the Euler-Lagrange equation is:

```text
dL/du - div(dL/d(grad u)) = 0
```

The residual:

```text
R(u) = dL/du - div(dL/d(grad u))
```

measures how far `u` is from being stationary under the action.

### Discrete Board Version

The board is an `8 x 8` grid. For field channel `c`:

```text
u_c[i, j]
```

Define forward differences:

```text
Dx u[i, j] = u[i, j+1] - u[i, j]
Dy u[i, j] = u[i+1, j] - u[i, j]
```

with zero or reflected boundary handling.

For a simple quadratic-gradient Lagrangian:

```text
L[i,j] =
  V_theta(u[i,j], x[i,j])
  + 0.5 * gx[i,j] * (Dx u[i,j])^2
  + 0.5 * gy[i,j] * (Dy u[i,j])^2
```

the discrete action is:

```text
A[u; x] = sum_{i,j} L[i,j]
```

The discrete Euler-Lagrange residual is:

```text
R = dV/du - Dx^T(gx * Dx u) - Dy^T(gy * Dy u)
```

where `Dx^T` and `Dy^T` are the negative-divergence adjoints of the finite differences.

This formula is cheap and differentiable.

### Multi-Channel Version

For `C` learned fields:

```text
u in R^{C x 8 x 8}
```

use:

```text
L = V_theta(u, x)
  + 0.5 * sum_c gx_c * (Dx u_c)^2
  + 0.5 * sum_c gy_c * (Dy u_c)^2
  + cross_theta(u, Dx u, Dy u, x)
```

The first implementation should omit the cross term or make it low rank:

```text
cross = sum_{k=1}^r a_k(u,x) * b_k(Dx u, Dy u, x)
```

to avoid instability.

## Architecture Sketch

### Step 1: Field Encoder

Input:

```text
x: (B, 18, 8, 8)
```

Project to learned fields:

```text
u: (B, C, 8, 8)
```

Recommended start:

```text
C = 12 or 16
```

Use a small CNN stem:

```text
Conv1x1 -> Conv3x3 -> Conv1x1
```

### Step 2: Lagrangian Parameter Heads

Emit:

```text
potential_params: V_theta inputs
gx: positive x-gradient stiffness
gy: positive y-gradient stiffness
king_weight: optional king-zone weight
occupied_weight: optional occupied-square weight
```

Use:

```text
gx = softplus(raw_gx) + eps
gy = softplus(raw_gy) + eps
```

### Step 3: Finite Difference Layer

Compute:

```text
Dx u
Dy u
```

with fixed kernels:

```text
[-1, 1] horizontally
[-1, 1]^T vertically
```

Boundary handling should be explicit and tested.

### Step 4: Euler-Lagrange Residual

For the minimal quadratic-gradient version:

```text
R = dV/du - DxT(gx * Dx u) - DyT(gy * Dy u)
```

The simplest version can make:

```text
V_theta(u, x) = MLP_per_square([u, x_projected])
```

and compute `dV/du` with PyTorch autograd.

For speed, a first implementation may parameterize:

```text
dV/du = potential_force_head([u, x_projected])
```

and skip second-order autograd. That is no longer an exact action derivative, so it should be called the "force-head approximation" and ablated against the exact small version.

### Step 5: Residual Readout

Collect:

```text
action_value = sum L
residual_l1 = mean abs(R)
residual_l2 = mean R^2
residual_max = max abs(R)
king_zone_residual
occupied_square_residual
empty_square_residual
gradient_energy
potential_energy
boundary_flux
```

Also keep low-resolution residual maps:

```text
R_map: (B, C, 8, 8)
```

Feed both scalar summaries and a small residual-map CNN to the classifier.

## Tensor Contract

```text
input:             (B, 18, 8, 8)
fields_u:          (B, C, 8, 8)
dx_u:              (B, C, 8, 8)
dy_u:              (B, C, 8, 8)
gx:                (B, C, 8, 8)
gy:                (B, C, 8, 8)
potential_force:   (B, C, 8, 8)
residual_R:        (B, C, 8, 8)
summary:           (B, S)
logits:            (B, 2)
```

## Chess-Specific Weights

The action can include safe current-board weights:

```text
king_zone_mask
occupied_mask
own_piece_mask
opponent_piece_mask
empty_mask
center_mask
edge_mask
side_relative_rank
```

These do not encode legal moves or engine output.

Example weighted residual summaries:

```text
king_zone_residual = mean(abs(R) * king_zone_mask)
own_piece_residual = mean(abs(R) * own_piece_mask)
opp_piece_residual = mean(abs(R) * opponent_piece_mask)
empty_residual = mean(abs(R) * empty_mask)
```

## Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `cnn_only_matched` | Remove action/residual branch, keep same parameter count | Tests variational branch value | Full model should beat it. |
| `action_only` | Use action scalar and energies but remove Euler-Lagrange residual map | Tests residual value | Residual should add signal. |
| `no_gradient_terms` | Remove `Dx u`, `Dy u`; use only potential terms | Tests calculus gradient structure | Should degrade if field tension matters. |
| `random_difference_operators` | Replace finite differences with fixed random local linear operators | Tests board calculus geometry | Should degrade if spatial derivatives matter. |
| `residual_norm_only` | Use scalar residual norms, no residual maps | Tests localization | Full maps should help. |
| `force_head_only` | Predict residual directly from CNN, no action derivative | Tests actual variational derivation | Exact/structured residual should beat direct force head. |
| `harmonic_control` | Replace residual with fixed Laplacian/Poisson-style summaries | Tests difference from harmonic potential family | Variational action should add distinct signal. |

## Diagnostics

Required shared diagnostics:

- binary accuracy
- AUROC
- PR-AUC
- Brier
- ECE
- fine-label `3 x 2` matrix

Architecture diagnostics:

- action value by label
- potential energy by label
- gradient energy by label
- residual L1/L2/max by label
- king-zone residual by label
- occupied versus empty residual split
- residual heatmaps for correct positives and false positives
- ablation gaps for `no_gradient_terms` and `random_difference_operators`

## Expected Positive Result

The idea is promising if:

```text
full > cnn_only_matched
full > action_only
full > no_gradient_terms
full > random_difference_operators
king_zone_residual differs by fine label
residual maps are localized rather than globally saturated
```

The strongest evidence would be that residual maps correct CNN false negatives and remain useful inside material buckets.

## Expected Negative Result

Treat this as falsified if:

- `cnn_only_matched` matches the full model.
- `no_gradient_terms` matches the full model.
- `random_difference_operators` matches the full model.
- residual maps saturate everywhere.
- residual summaries mostly predict material count or phase.
- harmonic control performs the same with less complexity.

## Failure Modes

- Exact autograd through `V_theta` may be slow or numerically awkward.
- Residuals may become noisy if `gx` and `gy` are unconstrained.
- The model may learn to make `u` trivially stationary unless residual summaries are balanced with the supervised objective.
- A simple CNN may already learn equivalent local gradients.
- The calculus object may be too smooth for discrete chess tactics.

## Implementation Plan

### Files

```text
src/chess_nn_playground/models/trunk/variational_board_action.py
tests/test_variational_board_action.py
configs/model/variational_board_action.yaml
```

### Modules

```text
BoardFieldEncoder
FiniteDifferenceLayer
LagrangianHeads
EulerLagrangeResidualLayer
VariationalSummaryHead
VariationalBoardActionNet
```

### Forward Pseudocode

```text
def forward(x):
    u, context = field_encoder(x)

    gx = softplus(gx_head(context)) + eps
    gy = softplus(gy_head(context)) + eps

    dx = finite_diff_x(u)
    dy = finite_diff_y(u)

    potential_force = potential_force_head(concat(u, context))
    div_x = finite_diff_x_adjoint(gx * dx)
    div_y = finite_diff_y_adjoint(gy * dy)

    residual = potential_force - div_x - div_y

    summaries = summarize_action_and_residual(
        u, dx, dy, gx, gy, residual, masks_from_x
    )
    residual_map_features = residual_cnn(residual)
    board_features = small_board_cnn(x)

    return classifier(concat(summaries, residual_map_features, board_features))
```

### Minimal Config

```yaml
model:
  name: variational_board_action
  input_channels: 18
  field_channels: 12
  context_width: 48
  use_exact_potential_autograd: false
  use_force_head_approximation: true
  boundary_mode: reflect
  include_residual_map_cnn: true
  include_board_cnn_summary: true
  eps: 1.0e-4
training:
  loss: cross_entropy
  binary_target: true
diagnostics:
  fine_label_matrix: true
  log_action_terms: true
  log_residual_stats: true
ablations:
  - cnn_only_matched
  - action_only
  - no_gradient_terms
  - random_difference_operators
  - residual_norm_only
  - force_head_only
  - harmonic_control
```

## Unit Tests

Required tests:

- finite difference output shapes match input fields
- adjoint difference operator has expected shape and finite values
- constant field has zero gradient under chosen boundary mode
- residual layer returns finite tensor `(B, C, 8, 8)`
- ablation modes return logits `(B, 2)`
- gradients flow through field encoder and stiffness heads
- `random_difference_operators` preserves shape and parameter count

## Anti-Shortcut Controls

### Material Bucket Evaluation

Report metrics inside coarse material buckets. If the residual branch only helps across material phases, it is probably a shortcut.

### Residual Map Shuffle

Shuffle residual maps spatially before the classifier while preserving channel norms:

```text
R_shuffled = fixed_square_permutation(R)
```

If shuffled residual maps match full residual maps, localization is not being used.

### Difference-Operator Destruction

Replace `Dx` and `Dy` with random local operators of the same support and norm. This is the central semantics-destroying calculus ablation.

### Gradient-Energy Only

Use only:

```text
sum (Dx u)^2
sum (Dy u)^2
```

without Euler-Lagrange residual. If this matches full model, the model only needed roughness features.

## Relationship To Top-3 Derivations

This calculus approach is not one of the top-three implementation candidates yet. It should be treated as a high-math exploratory branch.

Most natural integration later:

```text
Piece-Token CNN Hybrid + variational residual summaries
```

or:

```text
Schur-Ray line fields as u, then Euler-Lagrange residual over line-corrected fields
```

Do not combine it with other complex parents until the standalone residual branch beats its controls.

## Best Immediate Experiment

Start with the force-head approximation, because exact second-order autograd is not needed for the first falsifier:

```text
field_channels = 12
boundary_mode = reflect
use_force_head_approximation = true
include_board_cnn_summary = true
```

Run:

```text
main
cnn_only_matched
no_gradient_terms
random_difference_operators
action_only
```

The central question is:

```text
Does an Euler-Lagrange-style residual over learned board fields add puzzle-likeness signal beyond ordinary CNN features and simple gradient energy?
```

If not, archive this as a clean negative result and do not repeat variational-board-action variants with only different field counts or boundary modes.

