# Codex Handoff Packet: Harmonic Board Potential Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-24_2045_friday_shanghai_harmonic_potential.md`
- Generated at: 2026-04-24 20:45
- Weekday: Friday
- Timezone: Asia/Shanghai
- Idea slug: `harmonic_potential`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Harmonic Board Potential Network
- One-sentence thesis: Puzzle-like positions may be identifiable by long-range board tension patterns that appear after solving fixed discrete Poisson equations over learned current-board charge maps.
- Idea fingerprint: current-board piece planes + learned safe charge maps + fixed discrete Green-function solvers + potential/flux/energy summaries + binary puzzle-likeness head.
- Why this is not a common CNN/ResNet/Transformer variant: the central operator is a fixed global inverse-Laplacian potential solver, not learned local convolution, residual stacking, self-attention, attack-graph propagation, or LC0 copying.
- Current-data minimal experiment: train a `simple_18` harmonic potential model for 3 epochs on the current source-balanced splits and compare against `simple_18` CNN/residual baselines with the same artifact pipeline.
- Smallest central falsification ablation: replace the Green-function solver with a random orthogonal transform that has the same output dimension and similar variance but destroys the harmonic distance law.
- Expected information gain if it fails: a clean failure rules out global electrostatic/harmonic smoothing as a useful chess-position bottleneck on this dataset, without conflating it with move generation or attack-graph semantics.

## 3. Problem Restatement And Data Contract

The task is binary chess puzzle-likeness classification from current board tensors. Fine labels `0`, `1`, and `2` remain evaluation diagnostics only. The model returns binary logits and must work with the shared trainer.

Allowed neural inputs:

- `simple_18` current-board piece planes.
- Side-to-move, castling, and en-passant planes already encoded.
- Deterministic board coordinates and fixed 8x8 finite-difference operators.
- Learned charge weights over current input channels.

Forbidden neural inputs:

- Engine evaluations, PVs, mate scores, node counts, verification metadata, source labels, proposed labels, dataset provenance, or unresolved candidate status.
- Legal move trees, move counts, checkmate/stalemate oracles, or any search-derived feature.

Tensor contract:

```text
input:        (B, 18, 8, 8)
charge maps:  (B, K, 8, 8)
potentials:   (B, K, L, 8, 8)
stats:        (B, K * L * S)
logits:       (B, 2)
```

Leakage checklist:

- The fixed Laplacian/Green kernels are board geometry only.
- Learned charges are functions of current tensor channels only.
- No legal move generator, attack map, engine score, or source metadata is used.

## 4. Research Map

External research anchors are conceptual only. No external citation was verified during generation.

| Source or concept | Borrowed | Not copied |
|---|---|---|
| Discrete potential theory / graph Laplacian Green functions | Solving `(L + lambda I) u = rho` to obtain global smooth potential fields from local charges. | No graph neural network over attack edges, no sheaf Laplacian, and no learned message passing. |
| Poisson image editing / inverse-Laplacian filtering | Fixed linear global solver as an inductive bias for long-range spatial coupling. | No image reconstruction objective. |
| Physics-inspired energy features | Dirichlet energy and charge-potential energy as compact summaries of tension. | No physical claim about chess pieces as true charges. |

Candidate search trace:

| Candidate mechanism | Why not selected |
|---|---|
| Heat-kernel diffusion stack with learned time steps | Too close to ordinary smoothing convolution unless the inverse solver is explicit. |
| Attack-map potential solver | Too close to imported sheaf/Hodge/attack graph packets. |
| King-path hazard potential | Too close to imported king-cage and escape path dynamic programs. |
| Full neural PDE solver | Too much capacity; a fixed Green function gives a cleaner falsifier. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Harmonic potential | Fixed inverse of board Laplacian applied to learned charge maps | `(B, K, 64) -> (B, K, L, 64)` | random same-shape orthogonal transform | Not attack graph, not sheaf Laplacian, not path DP. |
| Energy summary | `rho^T u`, `||grad u||^2`, boundary flux, king-neighborhood potential samples | `(B, K, L, 8, 8) -> (B, features)` | local Gaussian blur control | Tests inverse-Laplacian long-range law specifically. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN | Existing simple CNN | Local learned filters do not isolate global harmonic tension. |
| Residual CNN | Existing residual CNN | More depth is routine architecture scaling. |
| LC0-style residual tower | Existing LC0 configs | Copies engine-network conventions and does not test potential theory. |
| Vanilla ViT | Common Transformer | Attention capacity is too broad and less falsifiable. |
| Plain square GNN | Generic graph neural network | Too close to generic grid message passing. |
| Tactical attack graph Laplacian | Imported sheaf/Hodge packets | Already represented by attack-defense graph families. |
| King escape shortest-path potential | Imported king-path DP packets | The formal object is already researched. |
| Masked-board codec surprise | Imported masked-codec packet | A generative code-length field is a different family already imported. |
| Hyperparameter tuning | Existing training configs | Not a new architecture. |

## 6. Mathematical Thesis

Let `x in R^{C x 8 x 8}` be a current-board tensor. A safe charge encoder produces `rho_k(x) in R^{64}` for `k = 1..K` from current channels only. Let `L` be the fixed 8x8 grid Laplacian with chosen boundary condition and let `lambda_l > 0` be a small set of screening constants. Define:

```text
u_{k,l}(x) = (L + lambda_l I)^{-1} rho_k(x)
E_{k,l}(x) = rho_k(x)^T u_{k,l}(x)
D_{k,l}(x) = sum_edges (u_{k,l}(a) - u_{k,l}(b))^2
```

The hypothesis is that some tactical opportunities create long-range charge configurations whose smooth potential fields, fluxes, and energies separate puzzle-like positions from ordinary positions better than local texture alone.

Variational principle: `u = (L + lambda I)^{-1} rho` is the unique minimizer of

```text
J(u) = 0.5 * u^T (L + lambda I) u - rho^T u
```

for `lambda > 0`. Therefore the solver computes the lowest-energy global field matching the learned charges under the board geometry.

What is actually proven:

- The potential field is a deterministic, stable, global linear function of safe current-board charges.
- The screening constants define multiple spatial ranges.
- The solver cannot use engine or label metadata because its inputs are constrained to current tensor channels.

What remains hypothesized:

- That learned charges discover chess-relevant tension rather than material shortcuts.
- That inverse-Laplacian range coupling is better than CNN receptive fields at the current data scale.

Counterexamples:

- Labels mostly driven by local piece motifs that a small CNN already captures.
- Labels driven by source artifacts or material imbalances.
- Tactics requiring explicit move legality, pins, or forcing lines not visible through smooth board potentials.

Self-critique: harmonic smoothing may be too generic and may blur away chess-specific detail. The random-transform and local-blur controls isolate whether the Green-function distance law matters or whether any global projection is enough.

## 7. Architecture Specification

Module names:

- `Simple18ChargeEncoder`
- `FixedBoardPoissonSolver`
- `PotentialStatsPool`
- `HarmonicPotentialHead`

Forward pass:

1. A small 1x1 convolution maps `(B, 18, 8, 8)` to `K` signed charge maps, default `K=12`.
2. Optionally subtract each charge map mean to remove a trivial total-charge shortcut.
3. Flatten charges to `(B, K, 64)`.
4. Apply precomputed Green matrices for `L` screening constants: `G_l = (L + lambda_l I)^{-1}`.
5. Reshape potentials to `(B, K, L, 8, 8)`.
6. Compute potential statistics:
   - charge-potential energy `rho^T u`
   - Dirichlet energy over board-neighbor differences
   - mean/std/max/min by board quadrants and king rings if king extraction is safe
   - boundary flux summaries
7. Feed stats to an MLP head and return `(B, 2)` logits.

Shapes:

```text
input:       (B, 18, 8, 8)
charges:     (B, 12, 8, 8)
flat rho:    (B, 12, 64)
potentials:  (B, 12, 4, 8, 8)  # lambdas = [0.03, 0.1, 0.3, 1.0]
stats:       about (B, 12 * 4 * 12)
logits:      (B, 2)
```

Parameter estimate: 20k to 80k. The Green matrices are fixed buffers, not trainable parameters.

Complexity: applying all Green matrices costs `O(B * K * L * 64^2)` if implemented as dense matrix multiply. This is small for `K=12`, `L=4`.

Required config fields:

```yaml
model:
  name: harmonic_board_potential
  input_channels: 18
  num_classes: 2
  charge_channels: 12
  lambdas: [0.03, 0.1, 0.3, 1.0]
  boundary: neumann
  head_hidden: 128
  mean_center_charges: true
  ablation: none
```

Encoding adapters:

- `simple_18`: supported first.
- `lc0_static_112`: fail closed unless current-board piece channels and auxiliaries are explicitly mapped.
- `lc0_bt4_112`: optional later; deterministic charge maps may use only known current-slot channels. History planes should be passed only through a learned adapter in a separate experiment.

Pseudocode:

```python
rho = charge_conv(x)
if mean_center:
    rho = rho - rho.mean(dim=(-1, -2), keepdim=True)
rho_flat = rho.flatten(-2)
u = torch.einsum("lkj,bck->bclj", green_mats, rho_flat)
u = u.view(batch, charge_channels, num_lambdas, 8, 8)
stats = pool_potential_stats(rho, u, optional_king_masks_from_simple18(x))
return head(stats)
```

## 8. Loss, Training, And Regularization

- Primary loss: existing balanced cross entropy for coarse binary labels.
- Auxiliary loss: optional `L2` penalty on total charge magnitude; default small value `1e-5` or off for the first run.
- Batch size: 512.
- Optimizer: AdamW with learning rate `0.001`, weight decay `0.0001`.
- Determinism: fixed precomputed matrices, deterministic reductions, seed `42`.
- Fair comparison: same data paths, same epochs, same batch size, same artifact validation as CNN baselines.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| `random_orthogonal_solver` | Replace each Green matrix with a fixed random orthogonal matrix scaled to similar variance | Harmonic distance law matters | If it matches, the solver is just a global random projection. |
| `local_gaussian_solver` | Replace inverse Laplacian with fixed local blur kernels of similar scales | True global potential helps beyond smoothing | If it matches, local multiscale blur is enough. |
| `charge_only_stats` | Use charge map sums/moments without solving Poisson | Potential interactions carry signal | If it matches, learned charges alone explain results. |
| `material_centered_charges` | Freeze charge weights to material-like piece values | Learned charge semantics matter | If it matches, the network may only need material imbalance. |
| `square_permutation` | Apply a fixed random square permutation before solver and invert only for shape, preserving channel counts | Board geometry matters | If it matches, the harmonic board geometry is not being used. |

## 10. Benchmark And Falsification Criteria

Baselines:

- small/medium `simple_18` CNN
- small `simple_18` residual CNN
- local Gaussian ablation
- random orthogonal solver ablation

Metrics:

- AUROC, balanced accuracy, F1, calibration.
- Required fine-label `0/1/2 -> predicted 0/1` confusion matrix for main and central ablations.
- Class `1` recall at matched fine-label-`0` false-positive rate when available.

Success threshold:

- Main model beats the best same-budget `simple_18` CNN/residual baseline by at least `+1.0` AUROC point or improves class-`1` recall at matched fine-label-`0` FPR by `+2.0` points.
- Main beats `random_orthogonal_solver` and `charge_only_stats` by at least `+0.5` AUROC point.

Failure threshold:

- Main is no better than charge-only or random-transform controls and does not improve diagnostics over a small CNN.

Abandon if:

- Local blur, random transform, or material-centered charges match the main model.

Scale if:

- Harmonic solver beats all central ablations and shows a cleaner fine-label `1`/`2` diagnostic than CNN baselines.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260424_harmonic_potential/idea.yaml` | Create | Idea metadata. |
| `ideas/20260424_harmonic_potential/math_thesis.md` | Create | Variational Poisson thesis. |
| `ideas/20260424_harmonic_potential/architecture.md` | Create | Solver and pooling design. |
| `ideas/20260424_harmonic_potential/ablations.md` | Create | Random solver, local blur, charge-only controls. |
| `src/chess_nn_playground/models/harmonic_potential.py` | Create | Fixed Laplacian buffers, charge encoder, stats pool, builder. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `harmonic_board_potential`. |
| `configs/bench_harmonic_potential_simple18.yaml` | Create | Main config. |
| `configs/bench_harmonic_potential_random_solver.yaml` | Create | Central falsifier config. |
| `tests/test_harmonic_potential_forward.py` | Create | Forward shape, finite logits, fixed solver buffer checks. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-24_2045_friday_shanghai_harmonic_potential.md
  generated_at: 2026-04-24 20:45
  weekday: Friday
  timezone: Asia/Shanghai
  idea_slug: harmonic_potential
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260424_harmonic_potential
  name: Harmonic Board Potential Network
  slug: harmonic_potential
  status: draft
  created_at: 2026-04-24
  author: Codex
  short_thesis: Fixed inverse-Laplacian board potentials over learned safe charge maps may expose long-range tension patterns useful for puzzle-likeness classification.
  novelty_claim: Uses discrete Poisson/Green-function potential solvers over current-board charges, not CNN depth, attack graphs, sheaf Laplacians, path DP, OT, or masked-codec surprise.
  expected_advantage: Captures global board influence fields with a small fixed solver and clean random-transform falsifier.
  central_falsification_ablation: random_orthogonal_solver
  target_task: coarse_binary
  input_representation: simple_18
  output_heads: binary logits
  compute_notes: Dense Green multiplication costs batch * charge_channels * lambdas * 64^2.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/bench_harmonic_potential_simple18.yaml
  model_path: src/chess_nn_playground/models/harmonic_potential.py
  latest_result_path: null
  notes: Must include random-solver and charge-only ablations before scaling.
```

```yaml
config_yaml:
  run:
    name: bench_harmonic_potential_simple18
    output_dir: results
  seed: 42
  deterministic: true
  mode: coarse_binary
  device: nvidia
  data:
    train_path: data/splits/crtk_sample_3class/split_train.parquet
    val_path: data/splits/crtk_sample_3class/split_val.parquet
    test_path: data/splits/crtk_sample_3class/split_test.parquet
    encoding: simple_18
    cache_features: false
  model:
    name: harmonic_board_potential
    input_channels: 18
    num_classes: 2
    charge_channels: 12
    lambdas: [0.03, 0.1, 0.3, 1.0]
    boundary: neumann
    head_hidden: 128
    mean_center_charges: true
    ablation: none
  training:
    epochs: 3
    batch_size: 512
    num_workers: 0
    learning_rate: 0.001
    weight_decay: 0.0001
    class_weighting: balanced
    early_stopping_patience: 2
    mixed_precision: false
```

```yaml
model_spec:
  model_name: harmonic_board_potential
  file_path: src/chess_nn_playground/models/harmonic_potential.py
  builder_function: build_harmonic_board_potential_from_config
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18ChargeEncoder
    - FixedBoardPoissonSolver
    - PotentialStatsPool
    - HarmonicPotentialHead
  required_config_fields:
    - input_channels
    - num_classes
    - charge_channels
    - lambdas
    - boundary
    - ablation
  expected_parameter_count: 20000-80000
  expected_memory_notes: Potential tensor is batch * charge_channels * num_lambdas * 8 * 8 floats.
```

```yaml
research_continuity:
  idea_fingerprint: learned current-board charge maps + fixed inverse-Laplacian Green solvers + potential energy/flux summaries + binary puzzle-likeness
  already_researched_family_overlap: Adjacent only to generic potential theory; not attack graph/sheaf Laplacian, not king-path DP, not score-field denoising.
  closest_duplicate_risk: Could be confused with graph Laplacian/sheaf packets; distinguish by using only the fixed board grid Laplacian and no pseudo-legal attack relations.
  do_not_repeat_if_this_fails:
    - Poisson/Green-function board-potential classifiers with only different lambda lists or charge counts.
    - Harmonic potential models rescued by CNN fusion without first beating random-solver and charge-only controls.
  suggested_next_search_directions:
    - Only consider PDE-style operators with different boundary objects if random-solver ablation shows partial signal.
    - Investigate data-source artifacts if charge-only stats match the main model.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add this packet to imported memory with fingerprint `current-board charges + fixed Green-function potential solver`. | Blocks near-duplicate Poisson-field proposals. | `Imported Research Memory` |
| Add anti-duplicate text for harmonic/inverse-Laplacian board-potential models unless the PDE object or falsifier is genuinely different. | Prevents retreading the same potential-field idea with more charge channels. | Anti-duplicate region after graph/sheaf and topology/path warnings |
| Require random-solver, local-blur, and charge-only controls for future fixed-kernel global-field ideas. | These controls tell whether geometry, globality, or charges are doing the work. | `Ablation Plan` requirements |

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
- Repetition check against imported research packets completed: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Mobius-constellation, or pseudo-likelihood packets: yes
- Not an exact near-duplicate of imported orbit-symmetry, tempo-intervention, credal-evidence, rule-partition-invariance, kinematic-commutator, or masked-codec packets: yes
- Not an exact near-duplicate of imported cubical Euler/Betti topology, Hall-defect overload, or king-cage/king-escape path-DP packets: yes
- Not an exact near-duplicate of imported FCA/Galois-closure, denoising-score-field, or non-backtracking-edge-walk packets: yes
