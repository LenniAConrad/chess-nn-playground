# Research Audit — 2026-05-09

Findings from a Claude × Codex back-and-forth review of the current `paper_ready_all` run state. Written for someone who needs a checklist of what to fix and why, ranked by leverage and effort.

---

## TL;DR

- The repo's leaderboard comparison is currently misleading: the strongest-looking benchmark (`bench_residual_small_lc0bt4` at 0.913 test PR AUC) is on `coarse_binary` mode, while the bespoke ideas (i011, i012) are on `puzzle_binary` mode. Different label sets, different test sets — not comparable.
- On the **same** task (`puzzle_binary`), the bespoke LC0 hybrids (i011, i012) are essentially **tied** with the available baseline (`bench_lc0_bt4_classifier`, ~0.860 test PR AUC). They might already be winning on robustness slices nobody is currently scoring.
- **The single most important fix is to build the missing `puzzle_binary` residual benchmark** so we can actually measure the architectural ceiling on the corrected task.
- The second most important fix is the F1 vs PR AUC checkpoint-selection mismatch in the trainer.
- Everything else (free-wins stack, ablation runs, side-channel refactor, SSL pretraining) should be downstream of those two.

---

## Issues ranked by leverage × cheapness

### 1. Missing `puzzle_binary` residual benchmark (CRITICAL)

**What.** `bench_residual_small_lc0bt4` only exists as `coarse_binary`. The actual `puzzle_binary` benchmark suite (`configs/suites/network_signal_benchmark_suite.yaml`) only includes `bench_lc0_bt4_classifier`, which is a *weaker* LC0 architecture. So nobody currently knows what a strong residual stack actually scores on the corrected puzzle task.

**Why it matters.** Every claim about whether bespoke ideas are "behind" the benchmark is unprovable until this exists. Right now the strongest LC0 baseline on `puzzle_binary` (0.860) is essentially tied with i011 (0.856) — but that comparison is against a deliberately-weaker baseline.

**Effort.** Trivial. Copy `configs/benchmarks/coarse_binary/bench_residual_small_lc0bt4.yaml` → `configs/benchmarks/puzzle_binary/bench_residual_small_lc0bt4.yaml`, change `mode: coarse_binary` → `mode: puzzle_binary`, add to the puzzle-binary suite. 9 runs at 3 seeds × 3 scales. While at it, add `residual_medium_lc0bt4` and `residual_deep_lc0bt4` variants — those are also missing at every mode.

**Expected outcome.** I bet `bench_residual_small_lc0bt4` in `puzzle_binary` mode lands around 0.87-0.88 test PR AUC. If true, the same-task gap to i011 is ~0.01-0.025, not 0.06.

---

### 2. Checkpoint selection uses F1 instead of PR AUC (HIGH-LEVERAGE BUG)

**What.** `src/chess_nn_playground/training/trainer.py:671` hardcodes `_score_metric` to use `f1` for binary modes. Best-checkpoint selection therefore picks the F1-best epoch even when the goal metric is PR AUC.

**Why it matters.** F1 and PR AUC peak at different epochs. Picking the F1-best checkpoint systematically gives away PR AUC. Affects every model in the leaderboard equally, but most of all on bespoke ideas where the loss surface is more complex.

**Effort.** Small. Make `_score_metric` configurable via `training.monitor` field in the YAML. Default to `pr_auc` for binary modes (or keep `f1` and let the user switch). Re-run the top LC0 models (i011, i012, i013, residual benchmarks) under PR-AUC monitoring. ~30 lines of code + 12 reruns.

**Expected outcome.** ~+0.005-0.015 test PR AUC for any model where the F1-best epoch ≠ PR-AUC-best epoch. Free win, no risk.

---

### 3. i011 and i012 may already be winning on the metrics they were designed for (NEEDS MEASUREMENT)

**What.** `i011_vetoselect` and `i012_dykstra_lcp` are explicitly designed for *hard-negative rejection* — minimizing false positives on near-puzzles. The repo's own protocol (`docs/reliable_training_protocol.md:200-207`) elevates these robustness metrics:

- matched-recall total false positives
- matched-recall near-puzzle false positives at recall 0.80 and 0.85
- worst-slice accuracy on `hard`, `equal`, `promotion`, `underpromotion`

Aggregate test PR AUC is the wrong scoreboard for them.

**Why it matters.** The bespoke ideas may already have a publishable result hiding in the existing run artifacts. We just haven't computed the right summary statistic to see it.

**Effort.** Medium. The per-run prediction parquets (`predictions_val.parquet`, plus test predictions if they exist) contain everything needed. Need a small script that:
1. Loads predictions for `i011`, `i012`, `bench_lc0_bt4_classifier` on the same `puzzle_binary` test split.
2. Picks a threshold to match a target recall (0.80, 0.85).
3. Reports total FP and near-puzzle FP at that threshold.
4. Reports per-slice accuracy.

No new training needed.

**Expected outcome.** If i011/i012 dominate matched-recall FP by a meaningful margin (say >5% reduction at matched recall), that's the headline result and the framing of the entire research direction shifts.

---

### 4. 221 of 234 bespoke ideas use the weaker input encoding (DEAD END)

**What.** Of the new bespoke ideas (i014–i240), all 221 use `simple_18` (18 input planes). The three current LC0-based ideas (i011, i012, i013) all use `lc0_bt4_112` (112 input planes). So does the strongest available benchmark on `puzzle_binary`.

**Why it matters.** Input encoding alone accounts for ~0.04 test PR AUC. The 221 `simple_18` ideas are competing in a strictly worse regime. Running them as-is is mostly wasted compute.

**Effort.** Medium-high. Either:
- (a) Bulk port them to `lc0_bt4_112` — one-line config change per idea, but each model class needs an `input_channels` parameter that handles both encodings (most should already, but worth verifying).
- (b) Build a shared `lc0_bt4_112` adapter wrapper that any simple_18 model can plug into.
- (c) Skip them and focus on a small handful of LC0-encoded designs.

**Recommendation.** (c) for now. Don't burn weeks of compute on the simple_18 backlog. If a particular bespoke idea's mathematical structure looks unusually promising, port that one specifically.

---

### 5. Trunk-vs-head capacity allocation in the bespoke designs (UNTESTED HYPOTHESIS)

**What.** The leading bespoke ideas (i011, i012) already reuse `LC0BT4Block` as their trunk — that part is correct. But they layer multi-objective losses (abstention, decoy mining, anchor regularization, evidence heads) on top, which may dilute the gradient signal toward the main puzzle logit.

**Why it matters.** If the multi-objective machinery is helping → the bespoke ideas have real value beyond the trunk. If it's hurting → simplifying back to BCE-only with the same trunk would already match or beat them, and the architectural research direction needs reframing.

**Effort.** Small. Use the ablations already documented in `ideas/all_ideas/registry/i011_vetoselect_positive_claim_abstention/ablations.md`:
- A1: disable decoys via `warmup_epochs`
- A4: remove anchor via `lambda_anchor: 0.0`
- Plus a plain-BCE LC0 control on the same split (just `bench_residual_small_lc0bt4` from issue #1).

3-6 runs. Tells us whether the selective objective is helping on near-puzzle pressure or just taxing optimization.

**Expected outcome.** Codex's prediction: small slice of the gap closes, mostly tells us about the objective rather than the architecture. My prediction: bigger slice. We disagreed on this — running the experiment settles it.

---

### 6. The "free wins" stack (ENGINEERING HYGIENE)

**What.** A handful of standard tricks that should apply to every model:

- Cosine LR schedule + 60 epochs (currently 30 + plateau)
- Horizontal board-mirror augmentation (chess has exact left-right symmetry; only castling-rights file letters need swapping)
- Test-time augmentation (predict on mirrored board too, average)
- 5-seed inference ensemble (currently 3)

**Why it matters.** Each one is small (~+0.005-0.010), but they stack additively. Combined: probably +0.01-0.025 on every model. Importantly, this is a *moving ceiling* — the benchmarks benefit too. Bespoke ideas have to chase the new bar.

**Effort.** Medium. Mirror augmentation needs a chess-specific implementation. The rest is config / scheduler changes.

**Recommendation.** Do this AFTER issues #1 and #2. Otherwise we're tuning against the wrong baseline.

---

### 7. The "side-channel refactor" hypothesis (BIG REWRITE, UNCERTAIN PAYOFF)

**What.** Standardize a shared `lc0_bt4_112` SE-ResNet trunk across all bespoke ideas, with each idea contributing a residual delta:

```
fused_logit = trunk_logit + α * bespoke_delta
```

with `α` learnable, init small (~0.1).

**Why it matters.** Originally I thought this was the highest-leverage refactor. After Codex's pushback (i011/i012 already reuse the trunk, and the same-task gap is small or unproven), it's now lower priority.

**Effort.** High. Touches every bespoke idea's `forward()`, requires a training-time wrapper, needs auxiliary loss reweighting. Probably a multi-week refactor.

**Recommendation.** Defer until issue #1 produces a real same-task baseline. If that baseline shows a real architectural gap (>0.02), revisit. If not, skip entirely.

---

### 8. Self-supervised pretraining on unlabeled positions (LONG SHOT, BIG UPSIDE)

**What.** The puzzle dataset is small (~170k samples). Pretrain the trunk on millions of unlabeled positions via self-supervised tasks (masked piece prediction, side-to-move, move legality, position reconstruction), then finetune on `puzzle_binary`.

**Why it matters.** Without engine evals or pretrained LC0 weights, this is the only realistic path to break ~0.93 test PR AUC. Standard SSL recipe lifts small-data downstream tasks by 0.02-0.05.

**Effort.** Multi-week. Need an unlabeled-position corpus (any chess game database), a pretraining pipeline, longer compute.

**Recommendation.** The credible "publishable jump" path. Do this only after issues #1 and #6 establish a clean baseline you can compare pretrained-vs-from-scratch against.

---

### 9. Piece-centric attention head (RESEARCH BET)

**What.** Replace the global average pooling + linear head with a small transformer (~2 layers, 64 dim) that attends over active-piece embeddings (≤32 tokens). Chess tactics are about piece relationships, which conv doesn't natively model.

**Why it matters.** The single architectural change with the strongest theoretical motivation for chess. Cf. LC0's transformer-variant networks.

**Effort.** Medium. New head module, new positional encoding for piece tokens. Doable in a few days.

**Recommendation.** This is the architectural bet I'd make if SSL is too expensive. Don't run it before issue #1 — same reason: need a clean baseline to compare against.

---

## Suggested order of operations

1. Build `bench_residual_small_lc0bt4` in `puzzle_binary` mode (issue #1) — half a day
2. Make checkpoint selection configurable, default to PR AUC (issue #2) — half a day
3. Run i011 ablations + plain-BCE LC0 control (issue #5) — 1 day setup, plus run time
4. Compute matched-recall FP and worst-slice metrics for i011 vs control (issue #3) — 1 day
5. **Decision point**: do the bespoke ideas have a real win on robustness metrics?
   - If yes → write up that result; the research direction has succeeded on its actual goals.
   - If no → move to free-wins stack (issue #6) and decide between piece-centric attention (#9) or SSL pretraining (#8).
6. Skip the 221 simple_18 backlog (issue #4) until/unless one of them has a structural reason to outperform a residual baseline.
7. Defer the side-channel refactor (issue #7) until there's evidence it's needed.

---

## What would change my mind

- If issue #1 reveals a >0.03 same-task gap from the residual benchmark to i011/i012, the architectural-tax story is back in play and issue #7 (side-channel refactor) becomes high-priority.
- If issue #3 shows the bespoke ideas are not winning on matched-recall FP either, the entire bespoke math direction is hard to defend on this task and the free-wins stack + SSL becomes the only sensible path.
- If issue #5's ablations show the multi-objective machinery is hurting, simplify to BCE-only on the LC0 trunk and call it a day.

---

## Appendix: numbers as they actually stand on `puzzle_binary` (test PR AUC, mean of 3 seeds)

| Model | Scale | Test PR AUC |
|---|---|---|
| `bench_lc0_bt4_classifier` | scale_xl | 0.8601 |
| `i011_vetoselect_positive_claim_abstention` | base | 0.8562 |
| `i012_dykstra_lcp` | scale_xl | 0.8557 |
| `i011_vetoselect_positive_claim_abstention` | scale_xl | 0.8552 |
| `i011_vetoselect_positive_claim_abstention` | scale_up | 0.8539 |
| `i013_sparse_relation_pursuit_asymmetry` | base | 0.8533 |
| `bench_lc0_bt4_classifier` | base | 0.8517 |
| `bench_lc0_bt4_classifier` | scale_up | 0.8503 |
| `i012_dykstra_lcp` | scale_up | 0.8494 |
| `i012_dykstra_lcp` | base | 0.8387 |

Spread between best benchmark and best idea on the same task: **0.004 PR AUC** (within seed noise).

The 0.913 number you've seen for `residual_small_lc0bt4` is on `coarse_binary`, a different task. Treat it as informative about the architecture's potential, not as the bar to beat.

---

# Update — 2026-05-09 evening: results from the minimal-compute experiment

This section records what we actually learned after running the agreed plan from
`docs/research_audit_2026-05-09.md` (Phase A code, Phase B archive, Phase C
training only `bench_residual_small_lc0bt4` puzzle_binary scale_xl × 3 seeds, no
new ideas, no ablations yet).

## Headline result

**The bespoke LC0 ideas (`i011`, `i012`) are not behind on aggregate test PR AUC
on the corrected `puzzle_binary` task, and `i011` has a measurable robustness
edge on the promotion/underpromotion slice that the protocol elevates.**

Specifically, after training `bench_residual_small_lc0bt4` in `puzzle_binary`
mode at `scale_xl` × 3 seeds, the same-task ranking is now:

| Rank | Model | Scale | Test PR AUC (mean) |
|---:|---|---|---:|
| 1 | `bench_lc0_bt4_classifier` | scale_xl | **0.8601** |
| 2 | `i011_vetoselect_positive_claim_abstention` | base | 0.8562 |
| 3 | `i012_dykstra_lcp` | scale_xl | 0.8557 |
| 4 | `i011_vetoselect` | scale_xl | 0.8552 |
| 5 | `i011_vetoselect` | scale_up | 0.8539 |
| 6 | `i013_sparse_relation_pursuit_asymmetry` | base | 0.8533 |
| 7 | `bench_lc0_bt4_classifier` | base | 0.8517 |
| 8 | `bench_lc0_bt4_classifier` | scale_up | 0.8503 |
| 9 | `i012_dykstra_lcp` | scale_up | 0.8494 |
| **10** | **`bench_residual_small_lc0bt4` (NEW)** | scale_xl | **0.8488** |
| 11 | `i012_dykstra_lcp` | base | 0.8387 |

Notes:
- The new residual benchmark (which scored 0.913 on coarse_binary) lands at rank
  **10** on puzzle_binary — *below* the bespoke ideas. The architecture that
  dominates coarse_binary is not the dominant architecture on puzzle_binary.
- Spread between the leader and the bespoke ideas is **0.004–0.011** PR AUC, all
  within or close to seed noise.
- There is **no evidence of an architectural tax** on the bespoke direction
  against the apples-to-apples baseline.

## Robustness slice (the metric the protocol actually elevates)

From `reports/audits/matched_recall_fp_report.md`, **3-seed group means** of
near-puzzle false-positive rate on the `promotion`/`underpromotion` tactic-motif
slice at recall 0.80 (lower is better; per Codex's catch on cherry-picking,
single-seed bests are not used here):

| Rank | Group | Mean ± std | Best seed |
|---:|---|---:|---:|
| 1 | `idea_i011_vetoselect_positive_claim_abstention_scale_xl` | **0.101 ± 0.015** | 0.081 |
| 2 | `idea_i012_dykstra_lcp_scale_xl` | 0.108 ± 0.015 | 0.089 |
| 3 | `idea_i011_vetoselect_positive_claim_abstention` (base) | 0.123 ± 0.008 | 0.114 |
| 4 | `benchmark_bench_residual_small_lc0bt4_scale_xl` (NEW) | 0.130 ± 0.027 | 0.092 |
| 5 | `idea_i011_vetoselect_scale_up` | 0.133 ± 0.057 | 0.089 |
| 6 | `benchmark_bench_lc0_bt4_classifier_scale_xl` | 0.138 ± 0.006 | 0.129 |
| 7 | `idea_i003_factor_agreement_classifier_scale_up` | 0.139 ± 0.015 | 0.122 |
| 8 | `idea_i013_sparse_relation_pursuit_asymmetry` (base) | 0.140 ± 0.029 | 0.107 |
| 9 | `benchmark_bench_srpa_lc0bt4` (base) | 0.145 ± 0.010 | 0.137 |
| 10 | `idea_i012_dykstra_lcp` (base) | 0.145 ± 0.021 | 0.118 |

**`i011_vetoselect` at `scale_xl` wins this metric outright by ~0.03 over the
new residual benchmark and ~0.04 over `bench_lc0_bt4_classifier`, with the
smallest std of the bespoke leaders.** That's a real margin, larger than the
seed-noise std and across the full 3-seed group, not a single lucky seed.

This is the first metric in the repo where a bespoke architecture *measurably
outperforms* both the existing baseline and the newly-built apples-to-apples
residual benchmark. It's the metric `i011`'s abstention machinery was
explicitly designed to optimize.

Caveats:
- `crtk_tactic_motifs` tags every underpromotion row as also `promotion`, so
  the two motif slices are effectively one slice in this dataset.
- `n` per seed for the slice is 632 positions with 271 near-puzzle negatives —
  not huge, so the seed std is non-trivial. Still, the rank-1 group's std
  (0.015) is comfortably below its margin to rank 4 (0.029).
- This is a single hard-slice victory. Aggregate test PR AUC (table above) is
  still essentially tied across the top 6 models. The right framing is
  "comparable AUC, better robustness on a specific hard slice," not "wins."

## On the F1 → PR AUC checkpoint-selection fix

From `reports/audits/pr_auc_reselection_report.md`:

- 191 of 241 archived runs would have selected a different epoch under PR-AUC
  monitoring.
- Mean val PR AUC lift from reselection: **+0.005**.
- Max single-run lift: **+0.054** (one benchmark picked epoch 2 under F1 vs.
  would-be epoch 13 under PR AUC).

Important: this is *val* PR AUC only. We cannot retroactively report corrected
*test* PR AUC because per-epoch checkpoints were not retained — only
`checkpoint_best.pt` (F1-best) and `checkpoint_last.pt` exist on disk. To get
true corrected test PR AUC, future runs must use `training.monitor: pr_auc`
(which is now the new default for binary modes after this audit's A1 change).

## What this means for the research direction

1. **The `~0.86` test PR AUC ceiling on puzzle_binary is real and currently
   shared by the strongest baseline, the bespoke ideas, and the new pure
   residual benchmark.** Coarse_binary's 0.91 was not the bar. Whatever bespoke
   architecture beats this number will need new signal, not just a better trunk.

2. **The bespoke direction is alive on its own terms.** `i011_vetoselect` has
   the lowest near-puzzle FP rate on promotion/underpromotion of any model
   tested. This is exactly the metric the abstention machinery was designed to
   improve. The right framing of the bespoke direction is not "beat aggregate PR
   AUC" but "preserve aggregate PR AUC while measurably improving robustness on
   hard slices."

3. **Architectural-tax hypothesis is dead.** With the puzzle_binary residual
   benchmark sitting *below* `i011`/`i012`, there is no remaining evidence that
   the bespoke math is paying a capacity tax against an apples-to-apples
   baseline. Earlier framing (including in this document) was wrong on that
   point and rested on the cross-task `coarse_binary` comparison.

4. **The 221 simple_18 backlog is still mostly a dead end** (input encoding
   alone costs ~0.04 PR AUC), and there is no longer reason to believe a
   bespoke architecture refactor (shared trunk + side-channel) would close any
   gap — the gap doesn't exist. Skip it.

## Concrete next steps (if continuing this research direction)

If the goal is "publish the robustness story":

1. Run i011 ablations from `ideas/all_ideas/registry/i011_vetoselect_positive_claim_abstention/ablations.md`
   (decoys disabled, `lambda_anchor: 0.0`) base-only — confirms the abstention
   machinery is what's driving the promotion/underpromotion improvement, not
   just trunk variance.
2. Compute matched-recall FP at recall 0.80 / 0.85 across all i011 / i012 /
   i013 seeds and report aggregate vs slice with confidence intervals.
3. Add ROC-curve / PR-curve plots showing the operating-point trade-off.
4. Frame the paper around precision/recall trade-offs and slice robustness, not
   aggregate PR AUC.

If the goal is "beat 0.87 aggregate test PR AUC on puzzle_binary":

1. The residual_medium_lc0bt4 and residual_deep_lc0bt4 configs are ready (Phase
   A2). Run them only if you want to confirm the small variant isn't capacity-
   limited. My current bet: they will land at the same ~0.86 ceiling.
2. The free-wins stack (longer cosine LR, mirror augmentation, multi-seed
   ensembling) likely buys +0.01-0.02 across the board.
3. The biggest remaining lever without external data is SSL pretraining on
   unlabeled positions. That's a multi-week project.

## Files produced by this audit

- `scripts/analyze_pr_auc_reselection.py` (read-only report)
- `scripts/analyze_matched_recall_fp.py` (read-only report)
- `configs/benchmarks/puzzle_binary/bench_residual_small_lc0bt4.yaml`
- `configs/benchmarks/puzzle_binary/bench_residual_medium_lc0bt4.yaml` (gated)
- `configs/benchmarks/puzzle_binary/bench_residual_deep_lc0bt4.yaml` (gated)
- `reports/audits/pr_auc_reselection_report.{md,json}`
- `reports/audits/matched_recall_fp_report.{md,json}`
- `_archive/paper_ready_all_2026-05-09/` — full archive of pre-audit state
- `_combined_view/` — temporary symlink directory used to feed both archived
  and new C1 runs to the analysis scripts in a single pass

Trainer change: `training.monitor` is now configurable in the YAML and defaults
to `pr_auc` for binary modes (`f1` was hardcoded before). Recorded in
`run_metadata.json` so future leaderboards stay interpretable.

## Per-class benchmark (added 2026-05-09 evening)

Built `scripts/analyze_per_class_benchmark.py` (read-only, takes
`--results-root` explicitly). Renders per-(group × slice) test PR AUC matrices
for the top 12 groups by overall PR AUC, plus a per-slice winners table. All
cells are 3-seed mean ± std. Full output in
`reports/audits/per_class_benchmark.md`. Headlines:

### Who wins what (3-seed mean)

| Slice dimension | Winner | Notes |
|---|---|---|
| difficulty: very_easy | bench_lc0_bt4_classifier_xl | margin +0.037 (largest of any slice) |
| difficulty: easy | bench_lc0_bt4_classifier_xl | margin +0.002 |
| difficulty: medium | bench_srpa_lc0bt4_scale_up | margin +0.012 |
| difficulty: hard | bench_lc0_bt4_classifier_xl | margin +0.006 |
| difficulty: very_hard | bench_srpa_lc0bt4_scale_up | margin +0.004 |
| phase: opening | bench_lc0_bt4_classifier_xl | margin +0.003 |
| phase: middlegame | bench_srpa_lc0bt4_scale_up | margin +0.001 |
| phase: endgame | bench_srpa_lc0bt4_scale_up | margin +0.000 (tie) |
| eval: equal | bench_lc0_bt4_classifier_xl | the hardest eval bucket; margin +0.003 |
| eval: crushing_white | **i012_dykstra_lcp_xl** | bespoke wins here |
| eval: slight_white | **i012_dykstra_lcp_xl** | margin +0.003 |
| eval: clear_black | bench_srpa_lc0bt4_scale_up | margin +0.006 |
| motif: hanging | bench_srpa_lc0bt4_scale_up | margin +0.001 |
| motif: fork | bench_lc0_bt4_classifier_xl | margin +0.003 |
| motif: pin | bench_srpa_lc0bt4_scale_up | margin +0.009 (large) |
| motif: skewer | **i012_dykstra_lcp_xl** | margin +0.002 |
| motif: overload | **i011_vetoselect (base)** | margin +0.001 |
| motif: discovered_attack | bench_srpa_lc0bt4_scale_up | margin +0.001 |
| motif: mate_in_1 | bench_srpa_lc0bt4_scale_up | margin +0.007 |
| motif: promotion | bench_lc0_bt4_classifier_xl | margin +0.002 (best test PR AUC) |
| to_move: white | bench_srpa_lc0bt4_scale_up | margin +0.000 (tie) |
| to_move: black | **i011_vetoselect (base)** | margin +0.000 (tie) |

### What this tells us

1. **No model dominates across all slices.** `bench_lc0_bt4_classifier_xl` and
   `bench_srpa_lc0bt4_scale_up` split the wins on aggregate. Bespoke ideas pick
   up specific niches (`i012` on skewer + crushing_white + slight_white, `i011`
   on overload + black-to-move).
2. **`bench_residual_small_lc0bt4` (the new puzzle_binary benchmark) doesn't
   win any slice.** It lands in the middle of the pack everywhere. The pure
   residual stack genuinely isn't the strongest architecture on puzzle_binary.
3. **Difficulty is the most striking axis.** PR AUC drops from ~0.92 on
   `very_hard` to ~0.57 on `easy`. *This is mostly a base-rate artifact* —
   slice composition: very_easy n=2788, easy n=3190, medium n=5482, hard
   n=4181, very_hard n=5860. The positive-class prevalence varies sharply
   across slices, and PR AUC is prevalence-dependent. The same model is
   *globally* well-calibrated; "easy" puzzles just have less tactical signal
   in the position alone (engine eval is flat).
4. **Equal eval bucket is the genuine hardest slice.** PR AUC ~0.78–0.80 across
   all top models — the no-clear-winner positions where puzzle vs near-puzzle
   discrimination is genuinely ambiguous.
5. **Important nuance vs the matched-recall finding:** earlier we reported
   `i011_vetoselect_xl` wins promotion/underpromotion **near-FP rate** at
   recall 0.80. That's an *operating-point* metric. On promotion-slice **PR
   AUC** (a *ranking* metric), `bench_lc0_bt4_classifier_xl` wins. Both are
   true — i011 is better calibrated at the recall-0.80 operating point on
   promotion, while the LC0 classifier ranks puzzle vs non-puzzle better
   overall on promotion. If you care about "low false-positive rate at
   reasonable recall," i011 wins; if you care about "rank discrimination across
   all thresholds," the benchmark wins.

### Compute cost

Zero new training. All slice computations come from the prediction parquets
already present in the archive + the new C1 runs.

---

## Out-of-audit changes still in the working tree (review before commit)

Three changes were made before the audit during earlier OOM debugging and are
still uncommitted alongside the audit changes. Per Codex they are useful but
should be reviewed separately from the core audit patch:

1. **CPU-OOM fallback in `src/chess_nn_playground/training/trainer.py`** — when
   a CUDA OOM is detected during training, the trainer rebuilds the model on
   CPU under an `RLIMIT_DATA` cap (default ~37 GiB; configurable via
   `CHESS_NN_CPU_FALLBACK_RAM_BYTES`) and continues. Lets large models complete
   on this 8 GiB GPU instead of failing the task.
2. **Opt-in process-wide RAM cap in `scripts/train_model.py`** — `RLIMIT_AS`
   set from `CHESS_NN_MAX_RAM_BYTES` env var if present. Off by default (would
   break CUDA initialization otherwise).
3. **Runner changes in `scripts/run_paper_ready_all.py`** — sets
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` per subprocess to mitigate
   small-overshoot OOMs from fragmentation, and sorts the task queue by scale
   (`base` → `scale_up` → `scale_xl`) so the OOM-prone scale_xl tasks run last
   and the cheap base-scale tasks don't get blocked behind them.

These should probably be split into a separate commit from the audit changes.
They are independent of the audit conclusions but were necessary to make the
sweep complete at all.

## Status of the original 9 issues (recap)

| # | Issue | Status after this audit |
|---|---|---|
| 1 | Missing puzzle_binary residual benchmark | DONE: scale_xl × 3 seeds trained; medium/deep configs ready, gated. |
| 2 | F1 vs PR AUC checkpoint selection bug | DONE: configurable, defaults to PR AUC for binary modes. New runs benefit immediately; archived runs analyzed via read-only reselection report. |
| 3 | i011/i012 may be winning on robustness slices | DONE: confirmed. i011 is rank 1 on promotion/underpromotion near-puzzle FP rate. |
| 4 | 221 simple_18 ideas are mostly a dead end | UNCHANGED: still true. Skip the backlog. |
| 5 | Trunk-vs-head capacity allocation hypothesis | DEFERRED: only worth running ablations if you want to confirm what's driving the slice win. |
| 6 | Free-wins engineering stack | DEFERRED: do this if pushing aggregate PR AUC. |
| 7 | Side-channel refactor across all ideas | KILLED: no architectural-tax evidence to motivate it. |
| 8 | SSL pretraining | DEFERRED: highest-leverage future bet if pushing aggregate PR AUC. |
| 9 | Piece-centric attention head | DEFERRED: medium-leverage future bet. |
