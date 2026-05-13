# Primitive Training TODO

This is the operating plan for turning the primitive research folder into code, scout runs, and eventually improved architectures.

The rule is simple: validate primitives one at a time on top of the current best trunk before building a hybrid. A primitive that cannot beat its own matched ablation on its declared slice should not be promoted into a larger model.

## Current Organization

| Area | Purpose | Status |
|---|---|---|
| `claude_*.md` | Claude Opus 4.7 primitives: DHPE, TDCD, PFCT, CAIO, TSDP | Documented with handoff and prototypes |
| `codex_*.md` | Codex GPT-5 candidate/reply primitives | Implemented in a worktree batch as `p001`-`p005`; pending integration review |
| `external_01`-`external_30` | Imported primitive reports from ChatGPT, Claude, and Google/Gemini | Implementation batches launched as `p006`-`p035`; some still pending validation |
| `external_31`-`external_41` | 2026-05-13 GPT-5.5 Pro primitive reports imported from Downloads | Raw research backlog; reserve `p036`-`p046` for the next implementation pass |
| `../architecture_bridges/` | Stacking strategy and hybrid architecture notes | Documented |
| `PRIMITIVE_TRAINING_TODO.md` | Execution plan for code, tests, training, and promotion | This file |
| `ideas/research/collections/` | Human index over primitive, classic, and registered ideas | Linked from the root idea docs |

Do not move registered `ideas/registry/i###_*` folders into this tree. They are the implementation registry. This primitive folder is research input until a candidate is promoted.

## Baseline Assumptions

- Baseline trunk: `i193_exchange_then_king_dual_stream`.
- Source implementation: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Current trainer passes only board tensor `batch["x"]` into the model, with FEN and CRTK tags retained for reports/predictions.
- CRTK metadata remains reporting-only. It must not be used as model input.
- TSDP needs exact legal-move state from FEN or equivalent rule state. If used as input, compute it from FEN/board rules, not from benchmark tags.
- Proposed primitive IDs from the Claude handoff are legacy implementation labels for that batch:
  - TDCD: `i244`
  - DHPE: `i245`
  - PFCT: `i246`
  - CAIO: `i247`
  - TSDP: `i248`
- New primitive-native work should use the `p###` registry prefix:
  - `p001`-`p005`: Codex candidate/reply primitives.
  - `p006`-`p035`: 2026-05-12 external import implementation batches.
  - `p036`-`p046`: reserved for the 2026-05-13 GPT import batch.

Implementation order follows expected value, not numeric order.

## High-Level TODO

- [x] Consolidate primitive research into one folder.
- [x] Preserve the complete Claude Code handoff.
- [x] Add a human idea library that separates primitive research, classic architecture packets, and registered ideas.
- [ ] Freeze the comparison baseline: rerun or identify a clean i193 result on the same split, seed, scale, and training protocol used for primitive scouts.
- [ ] Add primitive feature/code infrastructure.
- [ ] Promote and implement TSDP first.
- [ ] Run TSDP smoke tests, scout run, shuffled-feature ablation, and slice report.
- [ ] Promote and implement PFCT if TSDP does not solve the promotion/mate gaps alone.
- [ ] Promote and implement TDCD for the `equal` eval bucket.
- [ ] Promote DHPE only after cheaper counterfactual primitives are evaluated.
- [ ] Promote CAIO last, after a small autograd and `torch.compile` compatibility check.
- [ ] Review and integrate completed `p001`-`p035` worktree batches one at a time.
- [ ] Triage `external_31`-`external_41`, deduplicate repeated elementary-symmetric/poly-ledger variants, and launch only the highest-potential non-duplicates first.
- [ ] Build a hybrid only from primitives that pass individual falsifiers.

## Training Order

| Order | Primitive | Why here | First target | Main ablation | Decision |
|---:|---|---|---|---|---|
| 1 | TSDP | Cheapest, rule-exact, precomputable | `mate_in_1`, stalemate traps | Shuffle terminal indicators across samples | Keep only if indicator shuffle loses target-slice lift |
| 2 | PFCT | Sparse and directly targets largest reported gap | promotion and underpromotion | Replace fanout with copied baseline embedding | Keep only if true fanout beats copied fanout |
| 3 | TDCD | Targets stable hard slice but costs more | `crtk_eval_bucket = equal` | Main effects without mixed derivative | Keep only if mixed partial is load-bearing |
| 4 | DHPE | High-cost pair interaction test | hard / near-puzzle / mate-like ambiguity | Unsigned or first-order pair controls | Keep only if signed Hessian matters |
| 5 | CAIO | Speculative representation bet | broad structural coherence | Phase-randomized and relation-mask-shuffled controls | Keep only if phase/masks are load-bearing |

## Code Promotion Pattern

For each primitive that we choose to test:

1. Create or reserve the registered idea folder from `ideas/registry/template/`.
2. Copy the primitive thesis into the idea docs:
   - `math_thesis.md`
   - `architecture.md`
   - `ablations.md`
   - `implementation_notes.md`
   - `trainer_notes.md`
3. Add `idea.yaml` with:
   - `status: draft` or `implemented`
   - `implementation_kind: bespoke_model`
   - `input_representation: current-board simple_18 plus explicitly documented rule-derived primitive features when needed`
   - `baseline_comparison: i193_exchange_then_king_dual_stream`
4. Add reusable code under `src/chess_nn_playground/models/`.
5. Add a build function and registry key in `src/chess_nn_playground/models/registry.py`.
6. Add the idea-local wrapper in `ideas/registry/p###_<slug>/model.py`.
7. Add a config in `ideas/registry/p###_<slug>/config.yaml` using the same split and training protocol as i193.
8. Add focused tests under `tests/`.
9. Run static validation, smoke tests, then scout training.
10. Write run notes under `ideas/registry/i###_<slug>/runs/` and update `idea.yaml.latest_result_path`.

Suggested production modules:

| Primitive | Idea folder | Model module | Registry key |
|---|---|---|---|
| TSDP | `ideas/registry/i248_rule_aware_tactical_head/` | `src/chess_nn_playground/models/rule_aware_tactical_head.py` | `rule_aware_tactical_head` |
| PFCT | `ideas/registry/i246_promotion_aware_head/` | `src/chess_nn_playground/models/promotion_aware_head.py` | `promotion_aware_head` |
| TDCD | `ideas/registry/i244_tempo_defender_cross_derivative_network/` | `src/chess_nn_playground/models/tempo_defender_cross_derivative_network.py` | `tempo_defender_cross_derivative_network` |
| DHPE | `ideas/registry/i245_pair_resonance_hessian_network/` | `src/chess_nn_playground/models/pair_resonance_hessian_network.py` | `pair_resonance_hessian_network` |
| CAIO | `ideas/registry/i247_complex_amplitude_chess_network/` | `src/chess_nn_playground/models/complex_amplitude_chess_network.py` | `complex_amplitude_chess_network` |

If the reserved IDs conflict with a future registry state, use the next available `i###` and update this table.

## Shared Code Shape

Start conservative. The first implementation should not rewrite the trainer or replace i193.

Preferred first architecture:

```text
simple_18 board tensor
-> i193 exchange/king dual-stream trunk
-> base i193 logit and diagnostics
-> one primitive head
-> gated primitive logit delta
-> final logit = base_logit + primitive_delta
```

Each primitive head should return diagnostics in the standard model-output dict:

```text
logits
base_logit
primitive_delta
primitive_gate
<primitive-specific diagnostics>
```

This keeps the comparison clean: if the primitive fails, remove the head without disturbing the trunk.

If two or more primitive models need the same plumbing, add a small shared helper module instead of copy-pasting full model files:

```text
src/chess_nn_playground/models/primitive_heads.py
```

Do not prematurely create a large primitive framework. Add the shared abstraction only after TSDP and PFCT expose the repeated shape.

## Data And Feature Plan

### TSDP

TSDP should be precomputed from FEN/legal chess state. Do not call `python-chess` per sample inside every training forward pass.

Implementation options, in order:

1. Add a data script:

```text
scripts/data/precompute_primitive_features.py
```

It reads the canonical split Parquets, computes TSDP columns from `normalized_fen`, and writes a new split directory such as:

```text
data/splits/crtk_sample_3class_unique_crtk_tags_primitives/
```

2. Extend `ChessPositionDataset` with optional `data.primitive_feature_columns`.
3. Extend `collate_positions` to stack those columns into `batch["primitive_features"]`.
4. Extend the trainer only as much as needed to pass primitive features to models that declare they need them.

If trainer changes are too broad for the first pass, the fallback is to encode TSDP scalar features as extra constant planes and set `model.input_channels` accordingly. That is less elegant but keeps the current `model(x)` contract.

### PFCT, TDCD, DHPE

These are model-side counterfactual primitives. They should use the current `simple_18` board tensor and i193 trunk features first.

- PFCT: gate on near-promotion pawns; zero output when no candidate exists.
- TDCD: compute mixed tempo-defender responses using batched counterfactual board tensors.
- DHPE: limit pair sampling before trying dense all-pair Hessian features.

### CAIO

CAIO should begin as a small side head on i193 pooled features, not as a full trunk replacement. Run a CPU/GPU autograd check before scout training:

- complex backward works
- no NaNs under mixed precision disabled
- behavior is acceptable with `torch.compile` disabled first
- only test `torch.compile` after the eager model passes

## Tests To Add

| Test | Purpose |
|---|---|
| `tests/test_terminal_state_detection.py` | Known FENs for mate-in-1, stalemate, legal-move count, shuffled indicator contract |
| `tests/test_rule_aware_tactical_head.py` | Forward shape, diagnostics, ablation mode, registry build |
| `tests/test_promotion_aware_head.py` | PFCT zero-gate on no promotion candidate and nonzero fanout on near-promotion case |
| `tests/test_tempo_defender_cross_derivative.py` | Mixed partial differs from main-effect-only controls on toy defender cases |
| `tests/test_pair_resonance_hessian.py` | Signed Hessian detects constructive/destructive pair cases |
| `tests/test_complex_amplitude_chess_network.py` | Eager backward, phase randomization ablation, no complex dtype leakage in output dict |

Minimum pre-training commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/validate_training_config.py --static ideas/registry/i###_<slug>/config.yaml
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_<primitive>.py
PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py --config ideas/registry/i###_<slug>/config.yaml
```

After a completed run:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/reports/report_prediction_slices.py --run-dir results/<run_dir> --splits val test
PYTHONDONTWRITEBYTECODE=1 python scripts/compare_results.py
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/audit_implementation_kinds.py --check
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/audit_architecture_conformance.py --check
```

## Scout Evaluation Standard

Each primitive scout should compare against i193 under matched conditions:

- same train/val/test split
- same encoding unless the primitive explicitly requires extra rule-derived features
- same seed for the first scout
- same training budget and early-stopping policy
- same threshold-selection rule
- same slice reporting

Required outputs:

- aggregate validation and test PR AUC
- near-puzzle false-positive rate at matched recall
- slice PR AUC for the primitive's target slices
- cost: params, FLOPs/MACs, throughput, wall-clock per epoch
- ablation result on the same run protocol
- written keep/drop decision

Promotion threshold:

```text
Keep a primitive only if it improves its declared target slice and the matched ablation loses most of that lift.
```

Aggregate PR AUC alone is not enough. A primitive can be worth keeping if it fixes a known hard slice without hurting aggregate performance or cost too much.

## Hybrid Architecture Plan

Build a hybrid only after individual primitive decisions.

Recommended hybrid path:

1. `i193 + TSDP`: rule-exact terminal side head.
2. `i193 + TSDP + PFCT`: terminal plus promotion-specific fanout.
3. Add TDCD only if the `equal` bucket remains a clear failure.
4. Add DHPE only if pairwise signed interaction beats cheaper interaction controls.
5. Try CAIO as either:
   - side head on pooled i193 features, or
   - trunk replacement experiment after side-head evidence.

Hybrid fusion should remain additive and gated:

```text
final_logit =
  i193_logit
  + gate_tsdp * tsdp_delta
  + gate_pfct * pfct_delta
  + gate_tdcd * tdcd_delta
  + gate_dhpe * dhpe_delta
  + gate_caio * caio_delta
```

Drop any primitive whose gate collapses to noise or whose removal does not hurt its target slice.

## Open Engineering Questions

- Should primitive scalar features be passed as a separate `batch["primitive_features"]` tensor, or appended as constant planes to the board tensor? Separate tensors are cleaner; constant planes are faster to integrate.
- Should i193 expose pooled `exchange_pool`, `king_pool`, and `summary` directly for primitive heads? This would avoid recomputing trunk features.
- Should we reserve `i244` to `i248` now in the registry or wait until each primitive is actually implemented? Safer path: document reserved labels here, but only write registry rows when a folder is scaffolded.
- Should TDCD get a prototype script before promotion? The current Opus folder has a TDCD markdown spec but no dedicated `tdcd_prototype.py`.

## Immediate Next Step

Implement TSDP as the first primitive:

1. Add exact terminal feature extraction from FEN.
2. Add a small `rule_aware_tactical_head` model on top of i193.
3. Add tests for terminal feature extraction and model forward contract.
4. Run one matched i193 baseline and one TSDP scout.
5. Run the shuffled-indicator ablation.
6. Decide keep/drop before touching PFCT.
