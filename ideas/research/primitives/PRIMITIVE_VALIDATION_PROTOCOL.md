# Primitive Validation And Falsifier Protocol

Chess NN Playground already aims for comparable evidence through a shared trainer, a documented ladder from smoke to paper-grade runs, a primitive pipeline that currently runs prototypes, primitive-specific tests, static config validation, and optional shortened scout planning or training, plus a run-artifact validator for standardized outputs. This protocol adds a stricter native-primitive gate between “the code imports” and “the primitive is allowed to consume scout compute or publish metrics,” so AMP faults, dtype mismatches, contiguity bugs, artifact failures, and timeouts are classified as engineering defects instead of as evidence against a mathematical idea. fileciteturn12file0L3-L3 fileciteturn13file0L3-L3 fileciteturn14file0L3-L3 fileciteturn15file0L3-L3 fileciteturn16file0L3-L3

This protocol is mandatory for any primitive that adds new model code, new rule-derived features, new custom differentiable computations, new diagnostics, or new artifact expectations. It is especially important in this repository because the current primitive research plan already says primitives should be validated on top of the current best trunk, kept or dropped by matched ablations on their declared target slices, and never promoted into hybrids unless their individual falsifiers pass. The repo’s existing reliable-training rules also explicitly say smoke and triage runs are not final evidence, undertrained runs must not be called bad-architecture evidence, and repo-level promotion should rely on repeated seeds rather than single-seed luck. fileciteturn13file0L3-L3 fileciteturn17file0L3-L3 fileciteturn18file0L3-L3

A primitive is **not scout-eligible** until all validation stages below pass. A primitive is **not leaderboard-eligible** until it also produces a full trainer artifact set that validates cleanly. A primitive is **not promotion-eligible** until its declared falsifier passes under matched conditions and its performance evidence clears the seed policy described below. These gates align with the repo’s comparability goals and existing artifact contract. fileciteturn12file0L3-L3 fileciteturn13file0L3-L3 fileciteturn19file0L3-L3

## Purpose And Scope

The purpose of this file is to standardize pre-scout validation for future chess primitives so that expensive runs test mathematical claims, not broken plumbing. The scope is every future primitive scout, whether it is a side head on an existing trunk, a primitive-only head, or a primitive that requires explicit rule-derived feature inputs. The current primitive plan notes that the trainer presently passes only the board tensor into the model and that any extra primitive features must be explicitly documented and derived from chess state rather than from CRTK reporting metadata. This protocol therefore treats feature provenance as part of validation, not as an implementation detail. fileciteturn17file0L3-L3

Use the following status vocabulary throughout the repo:

| Term | Meaning | Consequence |
| --- | --- | --- |
| **validation_pass** | All required pre-scout stages passed on the declared hardware/config | Primitive may enter scout dry-run or scout training |
| **engineering_failure** | Import, shape, dtype, AMP, backward, linalg, serialization, artifact, or timeout failure before valid scout evidence exists | Primitive is blocked; no leaderboard entry |
| **validation_inconclusive** | Primitive passed engineering gates but evidence is underpowered, undertrained, or missing required falsifier or seed coverage | May be debugged further, but no scientific claim |
| **performance_failure** | Primitive passed engineering gates and falsifier gates, then lost under matched conditions | Mathematical claim failed on current benchmark |
| **performance_candidate** | Primitive passed engineering gates and falsifier gates, and has valid matched evidence that it may be worth promotion | Eligible for repeated-seed promotion testing |

Single-seed scouts do **not** justify repo-level positive or negative conclusions. They may only generate provisional evidence. That rule follows the repo’s existing standards for promotion-grade and paper-grade claims. fileciteturn13file0L3-L3

## Validation Stages

Every primitive must pass the stages below in order. The pipeline should fail closed: a later stage never overrides an earlier engineering failure.

| Stage | What runs | Minimum pass condition | Output artifact |
| --- | --- | --- | --- |
| **Static contract** | `compileall`, static config validation, metadata check, falsifier declaration check | Exit code `0`; required fields present | `stage_static.json` |
| **Import and build** | Import module, build model from config, move to CPU and CUDA if available | No exception; parameter count positive; device move succeeds | `stage_import_build.json` |
| **Forward and backward contract** | Synthetic batch and one real batch through forward, loss, backward, optimizer step | Finite logits/loss/grads; expected grads present; diagnostics serializable | `stage_fb_contract.json` |
| **Gradient correctness** | `gradcheck` on custom differentiable kernels or reduced toy versions | `gradcheck == True` or explicit “not applicable” justification | `stage_gradcheck.json` |
| **Real-data smoke** | Ten minibatches from the real dataloader on the declared split | Finite loss/gradients; optimizer steps complete; end-of-window loss lower than start-of-window loss | `stage_smoke10.json` |
| **Mixed precision and numerical safety** | FP32 run plus CUDA autocast run for declared AMP dtypes | No dtype/runtime faults; no NaN/Inf; acceptable FP32/AMP divergence | `stage_amp.json` |
| **Artifact dress rehearsal** | At least one short trainer-mediated run that exercises report/artifact writing | Validation bundle complete; full run artifacts validate cleanly if a trainer run was used | `stage_artifacts.json` |
| **Scout eligibility gate** | Consolidated status review | All prior stages passed; falsifier and keep/drop rule declared | `primitive_validation.json` |

The real-data smoke stage must use the actual dataloader contract, not a hand-built toy batch. In this repo, that means the canonical tagged benchmark split or a documented sample from it, with the same input encoding and label contract that the benchmark expects. If a primitive requires extra features, the validation bundle must enumerate them, state how they were derived, and prove they came from legal chess state or FEN rather than from CRTK reporting tags or benchmark metadata, which the repo reserves for reporting only. fileciteturn13file0L3-L3 fileciteturn17file0L3-L3 fileciteturn19file0L3-L3

The ten-minibatch smoke is intentionally stronger than a forward-only smoke. A forward-only test can catch import and tensor-shape failures, but it cannot detect broken backward graphs, dead gradients, optimizer incompatibilities, AMP failures, non-serializable diagnostics, or obvious “the model cannot even reduce loss on real data” defects. The repo’s own training standards already distinguish smoke and triage from reliable evidence, so this protocol adds a native validation layer before even a shortened primitive scout is considered meaningful. fileciteturn13file0L3-L3 fileciteturn16file0L3-L3

The recommended smoke criterion is simple and machine-checkable: let `loss_start` be the median of minibatches `1-3`, and `loss_end` the median of minibatches `8-10`. Pass if `loss_end < loss_start - max(1e-4, 0.01 * loss_start)`, all values are finite, and at least one intended parameter group receives nonzero gradient norm. If the parent baseline also fails this exact smoke criterion on the same minibatches, re-label the result `validation_inconclusive` and repair the smoke harness rather than blaming the primitive.

## Safety Checks And Timeout Budgets

### Required Dtype, Shape, Contiguity, And Amp Safety Checks

The repo expects model inputs of shape `(batch, input_channels, 8, 8)` and logits compatible with `(batch, num_classes)` style outputs. Primitive validation must therefore assert the exact input rank, spatial size, channel count, batch alignment of every declared diagnostic tensor, and loss compatibility before any training run is treated as valid. If a primitive changes the input contract by adding explicit primitive features, that contract must be declared and validated separately instead of being silently tunneled through ad hoc tensors. fileciteturn20file0L3-L3 fileciteturn17file0L3-L3

Contiguity must be treated as a first-class safety invariant. PyTorch documents that `Tensor.view()` only works when the requested shape is compatible with the original tensor’s size and stride, and explicitly recommends `reshape()` when it is unclear whether a view is safe, because `reshape()` will return a view when possible and copy when necessary. Validation must therefore reject any `.view()` call on a tensor whose contiguity or stride contract is not explicit; the safe patterns are either `assert x.is_contiguous(); x.view(...)` or `x.reshape(...)`. citeturn9view1

Mixed precision must be validated as an independent stage, not assumed from a successful FP32 forward pass. PyTorch’s AMP examples state that ordinary mixed-precision training uses `torch.autocast` with `torch.amp.GradScaler`, and that backward passes under autocast are not recommended because backward ops run in the same dtype chosen for the corresponding forward ops. Every primitive must therefore pass at least one FP32 forward/backward step and one AMP step on CUDA, with backward outside the autocast region, all finite checks enabled, and an explicit report of the dtypes seen for sensitive intermediate tensors. citeturn8view1turn8view0

Numerically sensitive operations must be upcast deliberately. PyTorch’s numerical-accuracy notes warn that `torch.linalg` backends provide no guarantees on non-finite inputs and may return non-finite outputs, raise, or even segfault; they also warn that ill-conditioned inputs can silently produce incorrect results and that running in `float64` often helps. The same notes also explain that some FP16 and BF16 reductions may truncate intermediate accumulations enough to yield unexpected `inf` values. Primitive validation must therefore enforce the following rules: run `torch.isfinite` checks before and after every linalg-heavy block, upcast linalg and quantile-like statistics to `float32` or `float64`, and document any block that intentionally opts out of reduced-precision accumulation. citeturn12view0turn12view1turn12view2turn12view3

Any primitive that introduces a custom differentiable kernel, a delicate reduction, or complex-valued autograd should also run `torch.autograd.gradcheck` on a reduced toy case. PyTorch documents that `gradcheck` compares finite-difference gradients to analytical gradients, is designed around double-precision inputs, and can fail spuriously on overlapping-memory tensors. For this protocol, that means reduced toy cases must use double precision, contiguous or otherwise non-overlapping toy tensors, and explicit justification when gradcheck is not applicable. This is especially important for custom or complex primitives, because the repo’s primitive plan already flags complex-amplitude primitives as requiring eager autograd checks and no complex-dtype leakage before scout training. citeturn16view3 fileciteturn17file0L3-L3

Determinism is part of validation, but its limits must be stated honestly. PyTorch notes that complete reproducibility is not guaranteed across releases, commits, platforms, or even CPU versus GPU, while `torch.use_deterministic_algorithms(True)` can force deterministic alternatives where available or raise on known nondeterministic operations. Validation runs should therefore fix seeds, enable deterministic algorithms in the validation harness, and treat deterministic-algorithm exceptions as engineering failures rather than as performance evidence. citeturn8view5turn8view6

A minimal required safety checklist for every primitive is shown below.

| Check family | Required rule |
| --- | --- |
| **Input shape** | Assert rank `4`, spatial `8x8`, declared channel count, and batch alignment |
| **Output shape** | Assert logits are loss-compatible and diagnostics are batch-aligned |
| **Dtypes** | Masks `bool`; indices integer; activations/loss in FP32 or declared AMP dtype; sensitive reductions in FP32/FP64 |
| **Contiguity** | No unsafe `.view()`; use `reshape()` or explicit `contiguous()` |
| **Finite values** | Assert finiteness for logits, loss, gradients, diagnostics, and linalg inputs/outputs |
| **Gradient flow** | Assert intended parameter groups receive gradients |
| **Grad correctness** | Run `gradcheck` for custom differentiable or complex-critical pieces |
| **Amp** | Run FP32 and AMP parity checks, with backward outside autocast |
| **Determinism** | Fix seeds; run deterministic validation harness; report any nondeterministic-op failures |

### Speed Timeout Checks

Timeouts are not mere infrastructure annoyances in this protocol. A primitive that cannot complete its validation stages within declared budgets has not yet demonstrated admissible engineering quality, even if its mathematical idea is attractive. GitHub Actions supports both step-level and job-level `timeout-minutes`, and the repo’s primitive pipeline already forwards a `--timeout-minutes` budget into the underlying runner when configured. That means timeout policy can be enforced both in CI and in the local primitive runner. citeturn15view0 fileciteturn16file0L3-L3

Every primitive markdown must declare named timeout budgets for the following checkpoints on a named reference machine: import and build, synthetic forward and backward, ten-minibatch real-data smoke, artifact dress rehearsal, and one full scout epoch. The default starting budgets for this repo should be conservative but strict enough to expose pathological primitives early:

| Budget name | Recommended default |
| --- | --- |
| **import_build_timeout_s** | `30` on CPU, `15` on CUDA |
| **fb_contract_timeout_s** | `60` on CPU, `20` on CUDA |
| **smoke10_timeout_min** | `10` |
| **artifact_dress_rehearsal_timeout_min** | `30` |
| **scout_epoch_timeout_min** | Primitive-specific; must be declared in advance |

Absolute timeouts alone are not enough, because hardware differs. The validation bundle must also record relative speed versus the parent baseline on the same hardware, same batch shape, and same split sample. If throughput or forward-backward latency degrades badly, that may still be acceptable, but only if the primitive markdown declared the expected cost and the scout report discusses the tradeoff. This repo already treats throughput and speed summaries as standard outputs, and example primitive notes already tell authors to inspect `speed_summary.json` during evaluation. fileciteturn12file0L3-L3 fileciteturn19file0L3-L3 fileciteturn23file0L3-L3

A validation timeout has only three legal meanings in this protocol: the stage budget was set unreasonably low and must be revised; the code is too slow and must be optimized; or the primitive is not yet admissible. It is never legal to call a timed-out validation run a mathematical failure.

## Artifact Validation And Failure Labels

### Artifact Validation Requirements

The repo’s shared trainer already defines a standardized artifact pipeline, and `validate_run_artifacts.py` checks for required metrics files, manifests, checkpoints, predictions, HTML reports, dashboards, calibration plots, confusion matrices, and slice artifacts when CRTK-tagged splits are used. The first hard rule of this protocol is therefore: no primitive metric may enter a leaderboard, aggregate scout file, or “worked” summary unless the corresponding run directory passes artifact validation with zero errors, and leaderboard publication should use strict warnings as the gate. fileciteturn14file0L3-L3 fileciteturn19file0L3-L3

Pre-scout validation needs its own artifact bundle even when the full trainer has not run yet. Every primitive validation attempt must write, at minimum, the following files:

```text
primitive_validation.json
config_resolved.yaml
environment.json
stage_timings.json
dtype_shape_report.json
loss_trace_smoke10.csv
stdout.log
stderr.log
falsifier_spec.json
```

If the validation harness runs a short trainer-mediated dress rehearsal, that run must also write the standard trainer artifacts and pass the repo validator. Missing or corrupt artifacts are not soft bookkeeping issues; they are validation failures because this repo’s comparability model depends on standardized outputs and artifact contracts. fileciteturn12file0L3-L3 fileciteturn14file0L3-L3

Primitive diagnostics are part of the artifact contract. The primitive research plan already expects heads to expose standardized outputs such as `logits`, `base_logit`, `primitive_delta`, `primitive_gate`, and primitive-specific diagnostics. If a primitive claims to use certain diagnostics for interpretation or falsification, validation must prove that those diagnostics are finite, batch-aligned, serializable, and actually written into the validation bundle or the trainer report path. fileciteturn17file0L3-L3

CI should upload the validation bundle as a workflow artifact. GitHub’s artifact documentation states that `upload-artifact` exposes a SHA-256 digest and that `download-artifact` automatically validates the downloaded artifact against that digest, warning on mismatch. In this protocol, a digest mismatch must be escalated from a warning to an `artifact_failure` label for the affected run. citeturn15view2turn15view3

### How To Label Engineering Failure Versus Performance Failure

The repo’s existing standards already prohibit calling undertrained runs “bad architecture evidence” and treat single-seed evidence as weaker than repeated-seed promotion evidence. This protocol makes that sharper by defining a decision ladder. fileciteturn13file0L3-L3

Use the following decision table:

| Label | Use when | Leaderboard eligible | Mathematical claim allowed |
| --- | --- | --- | --- |
| **engineering_failure** | Any exception, NaN/Inf, shape fault, dtype bug, AMP fault, gradcheck failure, nondeterministic-op failure under deterministic mode, feature-provenance violation, or broken serialization before scout validity | No | No |
| **artifact_failure** | Metrics or reports exist but artifact validator fails, digest mismatches, or required diagnostics are missing | No | No |
| **timeout_failure** | A required validation or dress-rehearsal stage exceeds declared time budget | No | No |
| **validation_inconclusive** | Engineering gates passed, but evidence is single-seed only, undertrained, missing falsifier results, or otherwise too weak to decide | No | Only “inconclusive” |
| **performance_failure** | Engineering gates passed, required falsifier(s) passed, repeated-seed matched scout completed, and the primitive loses its declared target under matched conditions | Yes, but only as a valid negative result | Yes |
| **performance_candidate** | Engineering gates passed, required falsifier(s) passed, and matched evidence shows target-slice or aggregate value worth promotion testing | Yes | Yes, provisionally until promotion policy is met |

The critical rule is this: **performance labels are illegal until engineering validity is already established**. That prevents the common mistake of misreading a bug as a failed primitive. A single-seed scout can populate debugging notes, but it must remain `validation_inconclusive` for repo-level conclusions. Repeated seeds are required for promotion-grade claims, and the existing repo standard names three seeds as the practical minimum for repo claims. fileciteturn13file0L3-L3

The following pseudocode is the canonical classification logic for automation:

```text
if any(pre_scout_stage in {static, import_build, fb_contract, gradcheck, smoke10, amp, artifacts} fails):
    label = engineering_failure
elif timeout:
    label = timeout_failure
elif artifact_validation_fails:
    label = artifact_failure
elif falsifier_missing or scout_undertrained or seeds < required_for_claim:
    label = validation_inconclusive
elif falsifier_passed and matched_repeated_scout_shows_loss:
    label = performance_failure
elif falsifier_passed and matched_repeated_scout_shows_gain:
    label = performance_candidate
else:
    label = validation_inconclusive
```

## Primitive Falsifiers And Standard Report Schema

### Required Falsifier Section For Every Primitive

Every primitive markdown file must contain a dedicated **Required Falsifier** section. This is not optional prose; it is a machine-checkable contract. The current primitive research plan already encodes this logic explicitly, and existing primitive docs such as `p004_tail_copula_concordance` already define primary falsifiers, additional ablations, and keep/drop thresholds tied to load-bearing mechanism tests rather than to aggregate metrics alone. fileciteturn17file0L3-L3 fileciteturn23file0L3-L3 fileciteturn24file0L3-L3 fileciteturn25file0L3-L3

The falsifier section must include the fields below.

| Field | What must be stated |
| --- | --- |
| **Mechanism claim** | One sentence saying what the primitive is supposed to do that the parent baseline does not |
| **Parent baseline** | Exact baseline model and config family |
| **Target slice** | The slice where the primitive is expected to matter most |
| **Primary falsifier** | The cheapest ablation or control that should erase the gain if the mechanism is not real |
| **Negative controls** | At least one structure-destroying control and one cheaper surrogate control |
| **Recovery control** | A `zero_delta`, `trunk_only`, or equivalent control that collapses to baseline behavior |
| **Expected diagnostics** | Which primitive-specific diagnostics should move if the mechanism is genuine |
| **Keep rule** | Quantitative keep threshold on target slice, aggregate metric, and cost |
| **Drop triggers** | Conditions under which the primitive is explicitly dropped |
| **Cost ceiling** | Maximum tolerable slowdown, memory increase, or per-epoch cost |
| **Seed policy** | Minimum seeds required for any positive or negative claim |
| **Next action** | Promote, retune, or drop |

The repo already provides good examples of primitive-specific falsifiers that this protocol should standardize. In the current primitive plan, TSDP is falsified by shuffling terminal indicators, PFCT by replacing true fanout with copied baseline embeddings, TDCD by removing mixed-derivative structure, DHPE by dropping signed pair structure, and CAIO by phase randomization or relation-mask shuffling. The existing `p004` docs similarly define `rank_quantile_only` as the primary falsifier, plus square and channel shuffle controls, recovery controls, and explicit keep/drop thresholds. That is the model to generalize. fileciteturn17file0L3-L3 fileciteturn24file0L3-L3

A scout is invalid if the primitive’s keep/drop decision depends only on aggregate PR AUC while ignoring its declared target slice. The current primitive plan already says aggregate PR AUC alone is not enough and that a primitive may be worth keeping if it fixes a known hard slice without introducing unacceptable cost. The falsifier section is the place where that promise becomes concrete and testable. fileciteturn17file0L3-L3

### Standard Report Table Schema

Each primitive validation or scout report must include a standard table with the following columns. This schema is intentionally narrow so it can be aggregated later into CSV or JSON without human re-interpretation.

| Column | Type | Description |
| --- | --- | --- |
| `primitive_id` | string | `p###_<slug>` or legacy registered id |
| `commit_sha` | string | Source commit tested |
| `config_path` | string | Resolved config path |
| `parent_baseline` | string | Matched baseline identifier |
| `feature_contract` | string | `board_only` or named primitive feature set |
| `device` | string | `cpu`, `cuda:0`, etc. |
| `amp_mode` | string | `off`, `fp16`, `bf16` |
| `seed` | integer | Run seed |
| `stage_static` | enum | `pass` / `fail` / `na` |
| `stage_fb_contract` | enum | `pass` / `fail` / `na` |
| `stage_gradcheck` | enum | `pass` / `fail` / `na` |
| `stage_smoke10` | enum | `pass` / `fail` / `inconclusive` |
| `stage_amp` | enum | `pass` / `fail` / `na` |
| `stage_artifacts` | enum | `pass` / `fail` |
| `timeout_status` | enum | `pass` / `fail` |
| `loss_start` | float | Median early-window loss |
| `loss_end` | float | Median late-window loss |
| `target_slice_metric` | string | Metric name for declared slice |
| `target_slice_delta` | float | Delta vs matched baseline or control |
| `aggregate_metric` | string | Primary overall metric |
| `aggregate_delta` | float | Delta vs matched baseline |
| `throughput_delta_pct` | float | Relative throughput change vs parent baseline |
| `falsifier_id` | string | Primary falsifier label |
| `falsifier_result` | enum | `pass` / `fail` / `missing` |
| `engineering_label` | enum | Final label from decision ladder |
| `leaderboard_eligible` | bool | May the run enter published summaries? |
| `notes` | string | Short free-text note |

For richer primitive reports, add a second table containing ablation and control evidence. Existing repo templates already use that style, and `p004` is a good example: it reports unablated and ablated variants together, ties them to target-slice hypotheses, and makes the keep/drop rule explicit. fileciteturn22file0L3-L3 fileciteturn25file0L3-L3

## Ci And Script Recommendations

This protocol should be integrated into the current primitive pipeline, not run as an optional side script. Right now the primitive runner performs inventory checks, prototype execution, primitive-specific pytest, static config validation, and then optional shortened dry-run or training. The recommended change is to insert a native validation stage between static config validation and any dry-run or scout plan. A primitive that fails native validation should never reach the shortened scout path. fileciteturn15file0L3-L3 fileciteturn16file0L3-L3

The repo should add three scripts:

```text
scripts/primitives/validate_primitive_native.py
scripts/primitives/run_primitive_falsifiers.py
scripts/primitives/check_primitive_report_contract.py
```

Recommended responsibilities:

- `validate_primitive_native.py`: static contract, import/build, forward/backward, gradcheck, ten-minibatch real-data smoke, AMP safety, timeout enforcement, and validation-bundle writing.
- `run_primitive_falsifiers.py`: execute declared primary falsifier, negative controls, and recovery control under matched settings.
- `check_primitive_report_contract.py`: verify that the primitive markdown contains the required falsifier fields, timeout budgets, feature provenance declaration, and standard schema columns.

CI should use a GitHub Actions matrix so the same validator runs across at least CPU and CUDA variants, and across AMP modes where CUDA support exists. GitHub’s workflow syntax supports `strategy.matrix`, step-level and job-level `timeout-minutes`, and artifact upload/download actions for retaining validation bundles. Validation artifacts should always be uploaded so failed native validations are inspectable without rerunning. citeturn15view1turn15view0turn15view2turn15view3

A minimal CI shape for this repo is:

```text
job: primitive-validation-cpu
  matrix: primitive config
  steps:
    - static contract
    - import/build
    - synthetic + real-batch fb contract
    - gradcheck where applicable
    - upload validation bundle

job: primitive-validation-cuda
  matrix: primitive config x amp_mode(fp16,bf16,off)
  steps:
    - same as above, plus AMP stage
    - artifact dress rehearsal
    - upload validation bundle

job: primitive-report-contract
  steps:
    - check markdown/template/falsifier fields
```

The final fail-closed rule is simple: the leaderboard builder and any `scout_all_runs.*` exporter must ignore runs unless `primitive_validation.json` says `engineering_label` is neither an engineering nor artifact nor timeout failure, `falsifier_result == pass`, and `leaderboard_eligible == true`. This prevents invalid metrics from entering public summaries even if a training subprocess happened to emit numbers.

## Example Template For A New Primitive Markdown File

The template below extends the repo’s existing idea report template and the more detailed `p004` report style by adding the native validation contract, timeout budgets, and a mandatory falsifier block. It is meant to be copied into each new primitive folder before any scout run is launched. fileciteturn22file0L3-L3 fileciteturn25file0L3-L3

```md
# p###_<slug>

## Primitive Summary

- Primitive name:
- Parent baseline:
- Registry key:
- Primary target slice:
- Expected failure slice:
- Mechanism claim:
- Why this should help the benchmark:
- Feature contract: `board_only` | `board_plus_primitive_features`
- Feature provenance:
- Forbidden inputs confirmed absent:
- Commit SHA:
- Config path:

## Validation Contract

### Static Contract

- [ ] Module imports on CPU
- [ ] Module builds from resolved config
- [ ] `compileall` passes
- [ ] `validate_training_config.py --static` passes
- [ ] Required falsifier fields are present
- [ ] Feature provenance is documented
- [ ] Extra primitive diagnostics are declared

### Forward And Backward Contract

- Input shape expectation:
- Output shape expectation:
- Declared diagnostics:
- Required parameter groups that must receive gradients:
- CPU forward/backward pass:
- CUDA forward/backward pass:
- Optimizer step pass:
- `gradcheck` required: yes/no
- `gradcheck` status:
- Any non-contiguous reshape points:
- Any linalg or quantile-like numerical hotspots:

### Real-Data Smoke

- Split:
- Encoding:
- Smoke seed:
- Batch size:
- Ten minibatches path:
- `loss_start`:
- `loss_end`:
- Gradient finite:
- Diagnostics finite:
- Smoke decision: `pass` | `fail` | `inconclusive`

### Mixed Precision And Numerical Safety

- AMP modes tested:
- FP32 pass:
- FP16 pass:
- BF16 pass:
- Backward outside autocast confirmed:
- GradScaler confirmed:
- Sensitive ops forced to FP32/FP64:
- Linalg finite checks:
- Any AMP exclusions:
- Numerical safety decision:

### Timeout Budget

| Budget | Value | Reference hardware |
| --- | --- | --- |
| import_build_timeout_s | | |
| fb_contract_timeout_s | | |
| smoke10_timeout_min | | |
| artifact_dress_rehearsal_timeout_min | | |
| scout_epoch_timeout_min | | |

### Artifact Dress Rehearsal

- Validation bundle path:
- Full trainer dress rehearsal path:
- `validate_run_artifacts.py --strict-warnings` status:
- Required diagnostics written:
- `primitive_validation.json` written:
- Artifact decision:

## Required Falsifier

### Mechanism Claim

State the smallest nontrivial claim this primitive is making.

### Primary Falsifier

- Falsifier ID:
- What it removes:
- Why failure here would refute the mechanism:

### Negative Controls

- Structure-destroying control:
- Cheap surrogate control:
- Recovery control (`zero_delta`, `trunk_only`, or equivalent):

### Expected Diagnostics

- Diagnostics that should increase when the primitive is active:
- Diagnostics that should collapse under the primary falsifier:

### Keep Rule

State quantitative thresholds for:
- target-slice gain,
- aggregate-metric floor,
- throughput or memory ceiling,
- falsifier loss of lift.

### Drop Triggers

State explicit drop conditions.

## Scout Plan

- Same baseline parent and same split confirmed:
- Same threshold-selection rule confirmed:
- Same reporting slices confirmed:
- Scout seed policy:
- Promotion seed policy:
- Matched ablations to run:
- Scout decision allowed only if validation contract passes: yes/no

## Validation Summary Table

| stage | status | device | amp_mode | duration_s | notes |
| --- | --- | --- | --- | ---: | --- |
| static_contract | | | | | |
| import_build | | | | | |
| fb_contract | | | | | |
| gradcheck | | | | | |
| smoke10 | | | | | |
| amp | | | | | |
| artifacts | | | | | |
| scout_gate | | | | | |

## Falsifier Summary Table

| variant | target_slice_metric | aggregate_metric | throughput_delta_pct | falsifier_result | notes |
| --- | ---: | ---: | ---: | --- | --- |
| none | | | | | |
| primary_falsifier | | | | | |
| structure_destroying_control | | | | | |
| cheap_surrogate_control | | | | | |
| recovery_control | | | | | |

## Final Label

- engineering label:
- leaderboard eligible:
- promotion eligible:
- keep/drop recommendation:
- exact next step:
```

Adopted as written, this protocol gives the repo a clear pre-scout firewall: primitives either demonstrate that they are valid engineering objects with explicit falsifiers, or they do not consume meaningful benchmark attention. That is the right default for a codebase whose stated goal is comparable evidence rather than numbers produced by guesswork. fileciteturn12file0L3-L3 fileciteturn13file0L3-L3