# Codex Handoff Packet: Tropical Constraint Circuit Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-24_2046_friday_shanghai_tropical_circuit.md`
- Generated at: 2026-04-24 20:46
- Weekday: Friday
- Timezone: Asia/Shanghai
- Idea slug: `tropical_circuit`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Tropical Constraint Circuit Network
- One-sentence thesis: Puzzle-like positions may be better modeled as the near-satisfaction of a small number of latent tactical constraints, and a min-plus tropical circuit can test this OR-of-AND structure directly.
- Idea fingerprint: current-board literal-cost maps + soft min-plus monomial clauses + tropical margin/entropy summaries + binary puzzle-likeness head.
- Why this is not a common CNN/ResNet/Transformer variant: the central computation is a differentiable tropical semiring circuit, where conjunction is additive cost and disjunction is soft minimum, not convolution, residual stacking, square attention, or move enumeration.
- Current-data minimal experiment: train on `simple_18` for 3 epochs with the shared binary trainer and compare against same-budget CNN/residual baselines plus sum-product and mean-pooling ablations.
- Smallest central falsification ablation: replace every soft-min tropical clause with a matched soft-average/sum-product clause that preserves literal costs and parameter count but removes min-plus winner-take-most logic.
- Expected information gain if it fails: a clean failure says the current labels do not benefit from explicit near-satisfied constraint-circuit bottlenecks beyond ordinary differentiable pooling.

## 3. Problem Restatement And Data Contract

Task: classify current board positions into binary non-puzzle versus puzzle-like outputs. Fine labels remain diagnostics. The model must accept `(batch, C, 8, 8)` and return `(batch, 2)` logits.

Allowed neural inputs:

- Current `simple_18` board tensor.
- Side-to-move, castling, en-passant planes already present.
- Deterministic square coordinates, rank/file indicators, and side-relative coordinates.
- Learned literal costs computed only from current tensor channels.

Forbidden neural inputs:

- Engine/search fields, Stockfish scores, PVs, mate/node metadata, verification metadata, source labels, proposed labels, dataset provenance.
- Any legal move tree, checkmate/stalemate oracle, or forced-line result.

Leakage checklist:

- Clauses operate on current board literals only.
- No pseudo-legal move generation is required.
- Fine labels are not model inputs.

## 4. Research Map

External research anchors are conceptual only. No external citation was verified during generation.

| Source or concept | Borrowed | Not copied |
|---|---|---|
| Tropical algebra / min-plus semiring | The use of `min` as disjunction and addition as conjunction over nonnegative costs. | No theorem claiming chess tactics are tropical polynomials. |
| Differentiable logic / energy-based constraints | Near-satisfaction costs and soft-min relaxations of discrete clauses. | No SAT solver, no legal move oracle, and no symbolic tactical rule base. |
| Mixture-of-experts margins | Winner and runner-up clause gaps as uncertainty features. | No routing network with unconstrained expert capacity. |

Candidate search trace:

| Candidate mechanism | Why not selected |
|---|---|
| Differentiable SAT over explicit chess attack predicates | Too close to attack-graph/sheaf families and harder to keep small. |
| Neural cellular automaton over the board | Too likely to become another local iterative CNN. |
| Standard mixture of local experts | Lacks the min-plus constraint semantics and has a weaker falsifier. |
| Rough-path/signature sequence over board scans | Too close to high-order constellation or ray-language interactions. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Tropical polynomial | `p_k(x) = softmin_m (b_km + sum_l a_kml c_l(x))` over learned literal costs | `(B, L, 64) -> (B, K)` | sum-product/soft-average replacement | Not ray automata, not Mobius constellations, not FCA closure. |
| Constraint near-satisfaction | Nonnegative literal costs and additive monomial costs | `(B, M, literals) -> (B, M)` | literal permutation preserving material/counts | Tests constraint geometry rather than simple counts. |
| Clause margin | Best minus second-best tropical monomial cost | `(B, K, M) -> (B, K)` | remove margin stats | Tests whether sparse near-winner evidence helps. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN | Existing simple CNN | Tests local texture, not min-plus constraint near-satisfaction. |
| Residual CNN | Existing residual CNN | More layers are ordinary scaling. |
| LC0-style CNN/residual | Existing 112-plane configs | Copies a known engine-style family. |
| Vanilla ViT | Common square-token Transformer | Too broad and weakly falsifiable. |
| Plain GNN over squares | Generic graph neural network | Ordinary message passing rather than semiring logic. |
| Ray-language automaton | Imported ray-language packet | This idea is not string/ray automata; it uses board-wide literal-cost clauses. |
| Mobius/ANOVA constellation | Imported constellation packet | This idea uses min-plus OR-of-AND costs, not explicit degree-2/3 polynomial interactions. |
| Formal concept closure | Imported FCA packet | This idea does not compute closure operators over incidence attributes. |
| Hyperparameter tuning | Existing configs | Not a research architecture. |
| Ensembling | Any leaderboard ensemble | Would obscure the min-plus bottleneck test. |

## 6. Mathematical Thesis

Let `c_l(x) >= 0` be a learned literal cost for literal `l`, where a literal is a safe current-board condition such as "a side-relative square/channel pattern is present" as encoded by a 1x1 or shallow local literal encoder. A tropical clause has `M` monomials:

```text
m_{k,j}(x) = b_{k,j} + sum_l a_{k,j,l} c_l(x),  a_{k,j,l} >= 0
p_k(x) = softmin_tau_j m_{k,j}(x)
```

As `tau -> 0`, `p_k` approaches the minimum monomial cost. Low `p_k` means at least one learned conjunction of literals is nearly satisfied. The model classifies from clause costs, best-second margins, and softmin entropies.

Core hypothesis: puzzle-like positions often contain a compact latent constraint pattern that is "almost satisfied" by current board facts. An OR-of-AND cost circuit is a more direct bottleneck for this than averaging local features.

Variational view: `softmin_tau(m_1,...,m_M)` is the negative-temperature log partition function:

```text
-tau * log sum_j exp(-m_j / tau)
```

Its gradient concentrates on the lowest-cost monomials. This gives a differentiable relaxation of existential pattern matching while preserving a concrete central falsifier: replace softmin by averaging.

What is actually proven:

- With nonnegative coefficients, each monomial is monotone in literal costs.
- Softmin clauses implement a smooth relaxation of minimum conjunction cost.
- The sum-product ablation removes the min-plus existential bottleneck while preserving literal costs and parameters.

What remains hypothesized:

- That puzzle-likeness has learnable low-cost latent constraints in this representation.
- That the learned literals do not collapse to material/source shortcuts.

Counterexamples:

- Labels driven by broad positional distribution shifts rather than sparse constraints.
- Tactics that require exact legal move consequences or engine search.
- Datasets where material and phase dominate the label.

Self-critique: without sparsity pressure, clauses may use too many literals and become generic MLP features. The first implementation should enforce nonnegative low-rank clause weights and report literal-count/gate diagnostics, and the sum-product ablation should be run before scaling.

## 7. Architecture Specification

Module names:

- `Simple18LiteralCostEncoder`
- `TropicalClauseLayer`
- `TropicalMarginPool`
- `TropicalConstraintHead`

Forward pass:

1. Encode input into literal costs:
   - A 1x1 convolution from `18` channels to `literal_channels`, default `32`.
   - Add fixed coordinate/rank/file planes through a learned affine literal encoder.
   - Apply `softplus` so costs are nonnegative.
2. Flatten to `(B, L)` where `L = literal_channels * 64`.
3. Low-rank nonnegative clause weights:
   - `a_{k,j,l} = softplus(U_{k,j,r} V_{r,l})` with small rank `r=8`, or use top-k sparse masks after warmup.
4. Compute monomial costs `(B, K, M)`.
5. Apply softmin over `M` monomials for each clause.
6. Pool clause features: best cost, second-best margin, softmin entropy, mean monomial cost.
7. MLP head returns `(B, 2)`.

Shapes:

```text
input:           (B, 18, 8, 8)
literal_costs:   (B, 32, 8, 8)
flat_literals:   (B, 2048)
monomial_costs:  (B, 24, 12)
clause_stats:    (B, 24 * 4)
logits:          (B, 2)
```

Parameter estimate: 80k to 220k, mostly low-rank literal-to-clause weights and the final head.

Complexity: `O(B * K * M * rank + B * rank * L)` if using low-rank weights; default `K=24`, `M=12`, `rank=8`, `L=2048`.

Required config fields:

```yaml
model:
  name: tropical_constraint_circuit
  input_channels: 18
  num_classes: 2
  literal_channels: 32
  clause_count: 24
  monomials_per_clause: 12
  clause_rank: 8
  softmin_temperature: 0.25
  head_hidden: 128
  ablation: none
```

Encoding adapters:

- `simple_18`: supported first.
- `lc0_static_112`: fail closed unless current-channel semantics are known.
- `lc0_bt4_112`: optional later through learned literal encoder, but deterministic literal assumptions about history planes must be avoided.

Pseudocode:

```python
costs = F.softplus(literal_conv(x_with_coord_planes))
flat = costs.flatten(1)
weights = F.softplus(torch.einsum("kmr,rl->kml", U, V))
monomial = bias[None] + torch.einsum("bl,kml->bkm", flat, weights)
clause = -tau * torch.logsumexp(-monomial / tau, dim=-1)
margin = second_best(monomial) - best(monomial)
entropy = softmin_entropy(monomial, tau)
features = torch.cat([clause, margin, entropy, monomial.mean(-1)], dim=-1)
return head(features)
```

## 8. Loss, Training, And Regularization

- Primary loss: existing balanced coarse-binary cross entropy.
- Auxiliary regularizers:
  - Optional `L1` or entropy penalty on clause weights to keep clauses sparse.
  - Optional temperature schedule from `0.5` to `0.25`; keep fixed for first reproducible benchmark unless config support is clear.
- Batch size: 512.
- Optimizer: AdamW, learning rate `0.001`, weight decay `0.0001`.
- Determinism: fixed seed, no stochastic top-k masks in first run.
- Fair comparison: same splits, epoch count, batch size, reports, and artifact validation as `simple_18` baselines.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| `sum_product_clause` | Replace softmin with soft-average/sum-product pooling over monomials using the same literal costs and weights | Min-plus existential structure matters | If it matches, tropical logic is unnecessary. |
| `mean_literal_pool` | Pool literal costs directly into an MLP with matched parameter count | Constraint clauses add signal beyond literal summaries | If it matches, clauses are not useful. |
| `literal_square_shuffle` | Permute square locations with a fixed random permutation while preserving channels and counts | Board geometry matters | If it matches, the model likely uses material/count shortcuts. |
| `material_only_literals` | Keep only material/type aggregate literals | Sparse board constraints matter beyond material | If it matches, labels may be material dominated. |
| `high_temperature_softmin` | Increase temperature so softmin approaches averaging | Winner-take-most near-satisfaction matters | If it matches, exact min-plus behavior is not important. |

## 10. Benchmark And Falsification Criteria

Baselines:

- small/medium `simple_18` CNN
- small `simple_18` residual CNN
- `sum_product_clause` ablation
- `mean_literal_pool` ablation

Metrics:

- AUROC, balanced accuracy, F1, calibration.
- Fine-label `0/1/2 -> predicted 0/1` confusion matrices for main and central ablations.
- Class `1` recall at matched fine-label-`0` false-positive rate if available.
- Clause diagnostics: average effective monomial count, entropy, and mean active literal mass.

Success threshold:

- Main model beats best same-budget `simple_18` CNN/residual by `+1.0` AUROC point or improves class-`1` recall by `+2.0` points at matched fine-label-`0` FPR.
- Main beats `sum_product_clause` by at least `+0.5` AUROC point or clear class-`1` diagnostic gain.

Failure threshold:

- Sum-product, mean literal pool, or high-temperature softmin matches the main model.

Abandon if:

- The model only wins when clauses become dense and diagnostics show no sparse winner structure.

Scale if:

- The tropical model beats sum-product and material-only controls while maintaining interpretable low-entropy clause activations.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260424_tropical_circuit/idea.yaml` | Create | Idea metadata. |
| `ideas/20260424_tropical_circuit/math_thesis.md` | Create | Min-plus thesis. |
| `ideas/20260424_tropical_circuit/architecture.md` | Create | Literal and clause implementation. |
| `ideas/20260424_tropical_circuit/ablations.md` | Create | Sum-product, mean-pool, shuffle controls. |
| `src/chess_nn_playground/models/tropical_circuit.py` | Create | Literal encoder, tropical clause layer, builder. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `tropical_constraint_circuit`. |
| `configs/bench_tropical_circuit_simple18.yaml` | Create | Main config. |
| `configs/bench_tropical_circuit_sum_product.yaml` | Create | Central falsifier config. |
| `tests/test_tropical_circuit_forward.py` | Create | Forward shape, finite logits, ablation mode smoke tests. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-24_2046_friday_shanghai_tropical_circuit.md
  generated_at: 2026-04-24 20:46
  weekday: Friday
  timezone: Asia/Shanghai
  idea_slug: tropical_circuit
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260424_tropical_circuit
  name: Tropical Constraint Circuit Network
  slug: tropical_circuit
  status: draft
  created_at: 2026-04-24
  author: Codex
  short_thesis: A min-plus tropical circuit over learned current-board literal costs can test whether puzzle-like positions are near-satisfying sparse latent tactical constraints.
  novelty_claim: Uses differentiable tropical OR-of-AND clause costs, not CNNs, Transformers, ray automata, Mobius constellations, FCA closure, move deltas, OT, or sheaf graphs.
  expected_advantage: Directly models existential sparse constraint satisfaction with a clean sum-product falsifier.
  central_falsification_ablation: sum_product_clause
  target_task: coarse_binary
  input_representation: simple_18
  output_heads: binary logits
  compute_notes: Low-rank clause weights keep literal-to-clause cost manageable for 2048 literals.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/bench_tropical_circuit_simple18.yaml
  model_path: src/chess_nn_playground/models/tropical_circuit.py
  latest_result_path: null
  notes: Must report clause entropy/effective monomial count to verify the bottleneck is not dense averaging.
```

```yaml
config_yaml:
  run:
    name: bench_tropical_circuit_simple18
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
    name: tropical_constraint_circuit
    input_channels: 18
    num_classes: 2
    literal_channels: 32
    clause_count: 24
    monomials_per_clause: 12
    clause_rank: 8
    softmin_temperature: 0.25
    head_hidden: 128
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
  model_name: tropical_constraint_circuit
  file_path: src/chess_nn_playground/models/tropical_circuit.py
  builder_function: build_tropical_constraint_circuit_from_config
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18LiteralCostEncoder
    - TropicalClauseLayer
    - TropicalMarginPool
    - TropicalConstraintHead
  required_config_fields:
    - input_channels
    - num_classes
    - literal_channels
    - clause_count
    - monomials_per_clause
    - clause_rank
    - softmin_temperature
    - ablation
  expected_parameter_count: 80000-220000
  expected_memory_notes: Main monomial tensor is batch * clause_count * monomials_per_clause floats; low-rank weights avoid full K*M*L storage if needed.
```

```yaml
research_continuity:
  idea_fingerprint: current-board literal costs + low-rank nonnegative tropical clauses + softmin min-plus margins + binary puzzle-likeness
  already_researched_family_overlap: Adjacent to differentiable logic and constraint learning; not ray automata, FCA closure, Mobius polynomial constellations, move-delta sets, or sheaf graphs.
  closest_duplicate_risk: Could be mistaken for Mobius/ANOVA high-order interactions; distinguish by min-plus near-satisfaction and the sum-product falsifier.
  do_not_repeat_if_this_fails:
    - Tropical/min-plus OR-of-AND literal-cost circuits over simple_18 with only different clause counts or temperatures.
    - Dense differentiable-logic models rescued by larger literal encoders without sparse-clause diagnostics.
  suggested_next_search_directions:
    - Only revisit constraint circuits if ablations show min-plus signal but implementation capacity is too low.
    - Prefer data audits if material-only literals match the full model.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add this packet to imported memory with fingerprint `literal-cost tropical min-plus clause circuit`. | Prevents future repeats under differentiable logic, tropical polynomial, or constraint-circuit names. | `Imported Research Memory` |
| Add an anti-duplicate rule for min-plus/tropical OR-of-AND board-literal circuits unless the semiring object or falsifier changes. | Clause count, temperature, and literal vocabulary tweaks are not new architectures. | Anti-duplicate section after FCA/Mobius/ray warnings |
| Require sum-product, high-temperature, and literal-shuffle controls for future semiring/logic bottlenecks. | These controls isolate whether min-plus structure and board geometry matter. | `Ablation Plan` requirements |

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
