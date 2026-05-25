# Knowledge Base — Chess-NN-Playground

What we have learned from the completed and attempted `puzzle_binary`
experiments in this repo: the 234-task May 9 architecture scout, the May 13
primitive scout, paper-grade top-trunk reruns, i018 falsifier, i018 hybrids,
i249 speed attempt, BT4 primitive-mixer scout, CPU benchmark, and LC0 BT4
transformer run. Companion to `reports/EXPERIMENT_REPORT.md` and
`reports/aggregate_report.md`; this document **generalizes**.

Important denominator note: the May 9 scout attempted 234 tasks, but only 175
completed cleanly (plus 49 failed, 6 timeout/resume cases, 3 timeouts, and 1
artifact-validation failure). Across the whole repo there are more result
artifacts than that, but most are single-seed scouts, reruns, or partially
failed pipelines. Do not read this as "225 clean paper-grade models."

If you take one thing from this: **chess-aware inductive bias is the only
lever that has reliably moved the needle.** Param count, FLOPs, fancy math,
and clever optimization tricks have not. The falsifier in `ideas/registry/i018_*/`
proves this for the strongest architecture we have (i018 collapses by −0.042
PR-AUC when its chess-relation masks are scrambled). Everything else
(transformers, exotic-math primitives, generic CNNs) tops out near 0.86 on
this task while chess-aware sheaf architecture hits 0.89 at the same
parameter budget.

---

## TL;DR — the three findings that matter

1. **Chess-aware architectures dominate.** Of the top-15 entries in the
   234-task historical scout (`reports/audits/scout_all_runs.csv`),
   essentially every one encodes chess-specific structure (dual-stream
   king/exchange, sheaf-laplacian, rule-automorphism, legal-routing). The
   bottom of the scout is full of generic / exotic-math architectures
   (tropical constraint circuits, ray-language automata, multiplicative
   conjunctions) that score near majority-class baseline.
2. **i018 oriented_tactical_sheaf_laplacian is the current champion at
   accuracy-per-parameter.** At 91K params it beats a 501K BT4-conv tower by
   +0.016 PR-AUC, and at matched ~500K params it beats it by +0.031. It
   scales: 0.875 base → 0.880 scale_up → 0.890 scale_xl. The falsifier
   confirms its chess geometry is load-bearing.
3. **Speed is not what you think.** Param count and FLOPs are *not* good
   predictors of wall-clock speed. The brutal CPU benchmark: bt4_classifier
   (447K params in the random-input CPU harness) is **6.4× faster** than i018
   base (91K params) at batch=1. Dense conv maps to MKL/BLAS perfectly; i018's
   many small irregular ops in its TacticalIncidenceBuilder dominate
   per-forward overhead.

These three facts should anchor every future design decision.

---

## Architectures

### What works

**Chess-aware structural bias, baked into the whole architecture.** Concrete
patterns that have repeatedly landed in the top 10 across our experiments:

- **Sheaf-laplacian over typed chess relations** (i018, others in the
  research_packets/local_oriented_tactical_sheaf line). 12 typed relation
  masks (attacks, defenses, rays, pins, king-zones), learned restriction
  maps, bounded heat-flow diffusion. **Falsifier-verified.**
- **Dual-stream / multi-perspective CNNs** (i193 exchange_then_king_dual_stream
  — the prior top at base scale). Two streams that encode different chess
  perspectives (e.g., exchange-state vs king-zone) and combine.
- **Rule-automorphism / orbit-bottleneck networks** (i048, i042, i046). Use
  chess symmetries (color flip, board flip) as architectural invariances.
- **Compact specialist heads on chess-shaped features** (i147
  specialist_head_cnn, i145 piece_plane_gated_cnn). Mid-range scout scores,
  reliable.
- **Standard residual CNN tower with SqueezeExcite** (bench_lc0_bt4_classifier
  — the conv-tower we keep calling "BT4"). Plateaus around 0.86 but is
  ROBUST, FAST, and a useful baseline at any scale.

### What does not work

- **Generic transformers with off-the-shelf training.** Our authentic
  encoder-only transformer completed all 9 runs and averaged 0.757 base,
  0.640 scale_up, and **0.5875 scale_xl** — getting worse with more capacity.
  This is the classic over-parameterized-for-data-with-bad-LR-schedule
  failure mode, not a verdict on transformers. See *Training* below.
- **Chess-themed attention grafts at scout scale.** `i242_chess_decomposed_attention`
  combined NNUE/king-centric features, i193-style decomposition, and global
  self-attention, but scored 0.8677 PR-AUC against its conv-only parent i193 at
  0.8755 in the May 11 one-seed comparison. Its ablations were also worse:
  no-global 0.8621, no-chess-bias 0.8624, no-exchange 0.8529, i193
  hyperparameters 0.8661. The lesson is not "attention is useless"; it is
  "adding attention on 173K rows x 12 scout epochs did not beat the exchange
  + king conv decomposition, and the exchange stream was the most load-bearing
  part of i242."
- **Exotic math without chess grounding.** The bottom of the 234-task
  scout is full of these: tropical constraint circuits, ray-language
  automata, multiplicative conjunctions, finite field character sums — all
  near random. Fancy math is not a substitute for chess prior.
- **Primitives wedged into the wrong tower.** The bt4-primitive-mixer scout
  (40 variants) trained 37 and failed 3, **zero of which beat the conv baseline
  (~0.86)**.
  Best was a031 ray_cast_obstacle_pool_head at 0.8414 (still −0.018 below
  conv). Strong negative result: replacing a CNN's spatial mixer with a
  chess-primitive operator inside a fixed tower shell does NOT work.

### What might work but we never tested cleanly

- **i018 at higher channel/depth scales (scale_xxl).** Scale_xl was still
  gaining (0.890 vs 0.880 at scale_up). The curve has not plateaued.
- **Real BT4-style transformer with proper transformer training.** Not the
  CNN-trained baseline that failed; see the retry recipe in §Training.
- **Attention with enough data or a narrower operator.** i242 and the LC0 BT4
  transformer both say global attention is data/schedule sensitive here. A
  smaller chess-constrained attention operator, or attention trained on a much
  larger stream, is still untested.
- **Ensembles of i018 + bt4_classifier.** Two extremely different inductive
  biases (chess-aware structural vs spatial conv) with comparable params
  could ensemble well. Never tested.
- **Distillation from i018 → bt4_classifier shape.** Get i018's accuracy
  into bt4_classifier's CPU-friendly compute shape via teacher-student.
- **Mixture-of-experts where i018 is the routing key.** Use the relation
  energies as a gate over a CNN bank.

---

## Trunks, heads, and hybrids

### Width-scaling beats grafting

The single most actionable architectural lesson. Empirical: grafting a
primitive *head* onto i018 via gated-logit fusion (the hybrid experiments)
adds modest +0.006 PR-AUC over i018-base, **but only by adding ~3× the
parameters**. Width-scaling i018 alone from 91K → 234K params adds +0.005,
and to 474K adds another +0.010 — all reliable, no hybrid risk.

Trunk-plus-primitive grafts are a *worse* lever than just growing the trunk.

### Hybrids done so far

- **i245 pair_resonance_hessian_network** = i193 trunk + DHPE primitive
  head. Score 0.876 vs i193 alone 0.876 — **zero lift**.
- **i018 + p019** (kernel memory) = +0.006 vs i018 base, washes vs param-
  matched scale_up.
- **i018 + p034** (octilinear SSM scan) = +0.006, same wash.
- **i018 + p013** (sparse delta accumulator) = **−0.006 regression**.
- **i255 oriented_sheaf_dual_stream_fusion** = i018 sheaf trunk + i193
  dual-stream trunk in parallel, fused via learned gate on logits plus a
  12-diagnostic residual MLP. The most ambitious fusion attempt in the repo
  (two strongest architectures, complementary inductive biases, complementary
  slice strengths). Paper-grade, 3 seeds × 3 scales. Verdict: **barely
  competitive at base, doesn't scale, and is fragile at scale_xl.**

  | Scale (params) | fusion mean | nearest reference | Δ |
  |---|---:|---:|---:|
  | base (249K, 3 seeds)   | 0.8823 | i018 scale_up (234K) 0.8799 | **+0.0024** |
  | scale_up (670K, 3 seeds) | 0.8790 | i018 scale_xl (474K) 0.8901 | **−0.0111** |
  | scale_xl (1.47M, 1 seed + 1 fail + 1 in flight) | 0.8848 (seed43) | i018 scale_xl (474K) 0.8901 | **−0.0053** |

  Pure i018 climbs 0.8752 → 0.8799 → 0.8901 across scales; fusion goes
  0.8823 → 0.8790 → ~0.885 — **flat or declining**. Pure i018 captures
  whatever signal the fusion was supposed to add, and does it with fewer
  parameters. The fusion also failed entirely on scale_xl seed42 (acc
  collapsed to majority class, PR-AUC null) — a fragility signature absent
  from pure i018 at the same scale.

### When hybrids might actually help

The only case where adding a primitive head plausibly helps:
- The primitive's math is **orthogonal** to the trunk's (sequential vs
  static, delta vs absolute, etc.).
- The trunk has **headroom left** (i018 scale_xl was still gaining → some
  capacity for orthogonal signal).
- Comparison is param-matched, not just "beats base trunk".

We have not produced a single hybrid that wins this strict test yet, including
the strongest possible candidate — i018 fused with i193 (the two highest-PR-AUC
architectures in the repo). **Provisional generalization: feature/logit fusion
of two already-strong chess-aware architectures does not stack their wins.**
The most likely reason is that the two trunks' predictions are highly
correlated — they look different mathematically but encode largely the same
chess prior, so fusing them adds parameters without orthogonal signal. The
cheaper test of that hypothesis (late-ensemble of existing paper-grade
checkpoints' `predictions_test.parquet` files, no new training) was never run
and would settle whether the failure is "correlated predictions" or "fusion
mechanism is bad."

---

## Primitives

Across the primitive scout (40 attempted; 33 clean completions, 5 validation
failures, 2 artifact-validation failures), the 40-BT4-mixer scout, and the
4-hybrid run, the dominant lesson is: **most primitives don't survive being
wrenched out of their native architecture.**

### What worked (as standalone scouts)

Top primitives by single-seed scout PR-AUC (caveat: noise floor ~0.005, so
these are roughly indistinguishable):

```
0.8809  p034 octilinear_selective_scan       (8-direction Mamba-style SSM on rays)
0.8778  p035 sparse_legal_graph_transition   (typed legal-move graph conv)
0.8778  i248 rule_aware_tactical_head        (chess rule-aware head)
0.8777  p026 ray_cast_obstacle_pool_head     (8-direction prefix scan)
0.8771  p030 ray_parallel_ssm_head           (input-conditioned SSM per ray)
0.8764  p007 attack_ray_sparse_attention     (sparse attention over attack rays)
0.8764  p010 ray_occlusion_semiring_scan     (log-domain ray transmittance)
```

Common pattern: **ray-based geometric reasoning + chess-aware sparsity**. The
top primitives all encode the fact that chess pieces move along rays / fixed
geometric patterns.

### What didn't work (as drop-in mixers in bt4 tower)

```
worst-7 bt4 primitive-mixer scores:
0.7892  a018 bt4_sparse_delta_accumulator_mixer
0.7897  a019 bt4_delta_pair_accumulator_mixer
0.7932  a026 bt4_occlusion_semiring_ray_scan_mixer
0.7938  a038 bt4_move_kernel_operator_mixer
0.7948  a036 bt4_legal_move_laplacian_resolvent_mixer
0.7975  a004 bt4_complex_amplitude_chess_network_mixer
0.8018  a033 bt4_incremental_latent_accumulator_head_mixer
```

And **3 hard failures during training** (smoke-test passed, training did not):
- a011 move_graph_router
- a013 rule_conditioned_sparse_attention
- a014 legal_move_graph_delta

All three hit the same dtype bug during training:
`RuntimeError: quantile() input tensor must be either float or double dtype`.
The smoke gate did not cover the property we needed.

Separate standalone primitive failures:
- `i244_tempo_defender_cross_derivative` and
  `p031_legal_move_laplacian_resolvent` wrote artifacts but collapsed to
  degenerate metrics with blank PR-AUC, and artifact validation failed.
- `p001`–`p005` did not pass idea validation, so they are engineering failures,
  not evidence that their math is weak.

### Why most primitives fail as drop-in components

Three compounding reasons (in importance order):

1. **Inductive bias dies when you wrench out the operator.** A primitive
   designed within a specific pipeline depended on rule-derived inputs
   (knight masks, pawn directions, piece-type maps). Stripped of those
   (because the mixer slot has only opaque `(B, C, 8, 8)` channels), the
   primitive does math on the wrong tensor. The subagent adapters honestly
   documented these compromises but they're load-bearing.

2. **Most primitives are not natively token-mixer-shaped.** Of the 40
   primitives, only the ~15 in the legal_routing / legal_graph / ray_attention
   / ray_scan families have a natural `(B,C,8,8)→(B,C,8,8)` signature. The
   rest are standalone models — wrenching them into a mixer slot loses
   their architecture.

3. **The "head" primitives (`*_head`) pool to a logit by design.** When you
   force them to be shape-preserving, you've removed the part that gave them
   their predictive power. Several of the lowest scores in the bt4-mixer
   scout are `*_head` primitives.

### Untested primitive territory

- **Top standalone primitives have not been trained at promotion grade.**
  Top scout results are single-seed. p034 at 0.8809 may be 0.870 ± 0.010
  for all we know.
- **Primitives in their native bespoke form at multi-scale.** The scout
  was 1 seed × base scale. Did the top primitives scale like i018?
  Unknown.
- **Real attention-shaped primitives in isolation.** Several primitives
  (p007 attack_ray_sparse_attention, p008 rule_conditioned_sparse_attention)
  ARE attention variants — could they replace MHA inside a real transformer?
  Untested. The bt4-mixer scout was a different question.
- **Primitives with full access to rule-derived chess features in the mixer
  slot.** Our adapters had to use content-derived surrogates because the
  mixer signature didn't expose piece planes. A wider mixer interface
  (pass `simple_18` alongside `(B,C,8,8)`) might recover the chess prior.

---

## Encodings

### What we used

- **`simple_18`** — 18 planes (12 piece planes + side-to-move + castling +
  en-passant). What every primitive, every i018 / i193 variant, and our
  custom transformer used. This was deliberate: it kept everything
  comparable.

### What we didn't use

- **`lc0_bt4_112`** — 112 planes (LC0 BT4's actual input encoding). In the
  completed May 9 scout only 5 rows used it; most new primitives, i018/i193,
  hybrids, and the transformer used `simple_18`. Current configs also contain
  residual/SRPA LC0 variants, but we never ran a controlled encoding
  comparison on the same architecture.
- **Custom chess-aware encodings.** No experiments with mover-canonicalized
  attack/defense maps as input planes (instead of as an architectural
  feature like in i018).

### Open questions about encoding

- Does i018's score change if the input is `lc0_bt4_112` instead of
  `simple_18`? The richer encoding might let the sheaf builder skip some
  internal work.
- Does the transformer benefit disproportionately from `lc0_bt4_112`?
  Transformers tend to be more sensitive to input feature quality.
- Is there a learned encoding (a small CNN frontend) that improves
  downstream regardless of architecture?

---

## Training

### What we use (the repo's standard recipe)

The "scout" / "promotion" / "paper" tier scaffolding in `docs/reliable_training_protocol.md`:

```yaml
training:
  epochs: 20                  # paper-grade goes 30
  batch_size: 256             # adjusted per scale via batch-size-caps
  learning_rate: 0.0007
  weight_decay: 0.0001
  loss: bce_with_logits
  class_weighting: balanced
  early_stopping_patience: 5  # paper-grade 8
  min_epochs: 10              # paper-grade 15
  mixed_precision: true       # bf16 autocast
  matmul_precision: high
  lr_scheduler:
    name: reduce_on_plateau
    factor: 0.5
    patience: 2
```

**Tier definitions (from `docs/reliable_training_protocol.md`):**
- *Smoke* — 1 epoch, 1 seed, code-path proof only
- *Triage* — 3 epochs, 1 seed, filter obviously broken ideas only
- *Reliable* — 20 epochs, min 10, patience 5; one seed can support weak
  local comparisons but not champion claims
- *Promotion-grade* — 3 seeds at base, allows repo-level "this is the best"
  claims when the mean improvement is large enough
- *Paper-grade* — 3 seeds minimum, 5 preferred, matched baselines/ablations;
  recent paper-grade runs used 30 epochs, min 15, patience 8

This recipe works well for CNNs and i018-style sheaf architectures.

Actual recent run depth was uneven. The architecture scout was mostly
single-seed short training (12 epochs, min 6, patience 3), while the top-trunk,
hybrid, i249, falsifier, and transformer reruns were deeper repeated-seed
experiments. Treat the old scout as a filter, not as a final ranking.

One more training/evaluation trap: several reports optimize or checkpoint on
F1 while we rank by PR-AUC. `reports/audits/pr_auc_reselection_report.md`
reanalyzed 241 runs and found that PR-AUC checkpoint selection would have
chosen a different epoch in 190 of them, with mean validation PR-AUC lift
+0.0051 and max lift +0.0538. Because per-epoch checkpoints were not retained,
we cannot recover corrected test PR-AUC for those runs. Any future PR-AUC
leaderboard run should set the training monitor/checkpoint metric to PR-AUC.

### What does NOT work with this recipe

**Transformers.** Our authentic encoder-only transformer benchmark trained
with this recipe degraded with scale:

```
base (4.8M)      0.757
scale_up (12M)   0.640
scale_xl (25M)   0.5875
```

The monotone degradation is the smoking gun. The standard repo training
recipe is *missing* the things transformers need:
- **Linear warmup over first ~5% of steps.** Critical for attention
  initialization stability.
- **Cosine decay** instead of plateau-step. Transformers oscillate badly
  with sudden LR drops.
- **Lower peak LR at larger scales.** A fixed 7e-4 across 4.8M / 12M / 25M
  is too hot for the bigger ones.
- **Stronger weight decay** (0.05, not 0.0001) at transformer-scale params.
- **AdamW with `betas=(0.9, 0.95)`** instead of default Adam.
- **Label smoothing** (0.1) for cross-entropy-style losses.

Recommended transformer retry recipe (see EXPERIMENT_REPORT.md §8 for full
YAML):
```
warmup_cosine schedule, warmup_pct=0.05, min_lr_ratio=0.1
AdamW(betas=(0.9, 0.95), wd=0.05)
peak LR: base 3e-4, scale_up 2e-4, scale_xl 1e-4
label_smoothing=0.1
```

### What we have NEVER tried

- **Data augmentation.** No board rotations, color flips (other than i018's
  built-in mover canonicalization), or piece-type permutations. Chess has a
  rich symmetry group; we ignore it.
- **Mixup / cutmix.** Standard transformer-era regularizers, untried.
- **Curriculum learning.** Easy → hard puzzle difficulty progression,
  untried. Could help the bigger models converge.
- **Self-supervised pre-training.** Pre-train on masked-square prediction
  or move prediction, then fine-tune on `puzzle_binary`. Standard
  transformer recipe, untried.
- **Multi-task heads.** Train value + policy + puzzle-class jointly, as
  real Lc0 does. Untried.
- **Larger training data.** The current canonical split is underfilled:
  173,029 train rows, 21,305 validation rows, and 21,501 test rows
  (215,835 total). `data/reports/benchmark_data_readiness.md` marks this
  **blocked** against the intended 120K-per-class train and 15K-per-class
  val/test quotas. Real Lc0 / NNUE training uses millions. Untried — and
  might be the actual ceiling.
- **Split lineage matters.** Older April reports used the balanced
  `crtk_sample_3class` split (360K train / 45K val / 45K test). Newer paper
  runs use `crtk_sample_3class_unique_crtk_tags`, which is deduped and much
  smaller. Do not directly compare old April BT4/NNUE numbers to May scout
  numbers without noting the split change.
- **Different loss functions.** BCE-with-logits is fine; focal loss,
  contrastive losses for hard negatives, untried.
- **Knowledge distillation.** Use i018 as teacher to train a CNN-shaped
  student that retains accuracy at lower CPU latency. Untried.

### Training pitfalls observed

- **Seed variance is real.** Vetoselect dropped from 0.872 scout → ~0.86
  paper-grade across 3 seeds. Single-seed scout numbers within ~0.01 of each
  other are NOT a ranking.
- **Single-seed scout winners can be inflated.** i193 looked like 0.890 in the
  old scout, but paper-grade reruns landed around 0.876-0.877. That does not
  make the scout useless; it means it should be read as "promote this family"
  rather than "this exact row is the champion."
- **`scramble_relations`-style ablations are easy and high-information.**
  The falsifier ran in one afternoon and produced the single most defensible
  result in our entire repository. Apply this pattern to any new
  architecture that claims a chess-specific signal.
- **i249 lesson: training-time numerics ≠ inference-time numerics.** A
  model that matches another to 1e-10 in eager fp32 forward can diverge by
  0.02 PR-AUC across 30 epochs under bf16 autocast. Equivalence checks must
  use real training mode.

---

## Metrics and slices

### PR-AUC is not enough

The benchmark was built to separate verified puzzles from both known
non-puzzles and **near-puzzles**. Overall PR-AUC is useful, but an operational
puzzle filter also needs low near-puzzle false positives at a matched recall.
`reports/audits/matched_recall_fp_report.md` shows this clearly: at recall 0.8,
the best near-puzzle false-positive rates in the old scout were i193 at 0.128,
i024 at 0.140, i018 at 0.150, and i011 at 0.153. i018 is the repeated-seed
PR-AUC champion now, but i193 was the best fixed-recall near-puzzle rejector
in that audit.

Future leaderboards should report both:
- mean test PR-AUC across seeds
- near-puzzle FP rate at recall 0.8 and 0.85

### Threshold protocol compliance has been sloppy

`docs/reliable_training_protocol.md` §"Threshold And Comparison Rules"
mandates reporting at **four** thresholds for every promotion claim:

- default `0.5`
- validation-best F1
- validation-derived threshold targeting recall `0.80`
- validation-derived threshold targeting recall `0.85`

None of the recent paper-grade reports (paper-grade top-3 trunks + BT4-conv,
i018 hybrids, i018 falsifier, i249 fast, LC0 BT4 transformer) include all
four. The protocol also requires **validation-only threshold selection** and
**test set used only for final reporting** — easy to violate without
noticing. Any future promotion-grade claim should explicitly include the
four-threshold table.

### The weak slices are stable

The old slice reports and the April benchmark agree on the pressure points:

- `crtk_eval_bucket=equal` is usually the worst eval bucket.
- `hard` and `very_hard` positions produce many near-puzzle false positives.
- `promotion`, `underpromotion`, and often `mate_in_1` are weak motif slices.
  In `reports/audits/per_class_benchmark.md`, promotion/underpromotion PR-AUC
  for top scout models is only about 0.50-0.67 even when overall PR-AUC is
  around 0.85-0.88.

So a new model that only lifts easy negatives is not very interesting. The
high-value result is a lift on near-puzzle rejection and these weak slices
without losing verified-puzzle recall.

Caveat: the per-slice audit is mostly old scout evidence, not repeated-seed
paper-grade evidence. Its zero standard deviations are a reminder that this is
directional slice triage, not final statistical proof.

---

## Inference speed

### The hard truth

**Param count and FLOPs are not predictive of wall-clock speed.** Concrete
data from `reports/cpu_benchmark/results.md` (batch=1 latency, 4 CPU threads,
eager PyTorch, MKL enabled, no `torch.compile`):

| Model | Params in CPU harness | CPU ms |
|---|---:|---:|
| bt4_classifier base | 447K | **0.83** |
| bt4_classifier scale_xl | 2.55M | 2.26 |
| bt4_transformer base | 4.79M | 3.27 |
| i018 sheaf base | **91K** | **5.28** |
| i018 sheaf scale_xl | 419K | 11.17 |
| bt4_transformer scale_xl | 37.9M | 21.09 |

Note: the CPU benchmark constructs models on random `(B,18,8,8)` inputs, so
some parameter counts differ from trained BT4-112 runs. The latency lesson is
still brutal: **i018 base is 6.4× slower than bt4_classifier base** despite
being much smaller.

Paper-grade training throughput tells the same story. In
`results/paper_grade_top3/leaderboard_seed_summary.md`, i018 base trains at
about 2.3K samples/s and i018 scale_xl at about 1.1K samples/s, while i193
base is about 15K samples/s and bt4_classifier base is about 14K samples/s.
The relation builder is not just a CPU deployment problem; it is a training
throughput problem too.

### What's actually fast

- **Dense `Conv2d` ops over small spatial maps.** MKL/BLAS makes them flat-
  out efficient on CPU. The whole bt4_classifier story is "standard convs
  map cleanly to optimized kernels."
- **Standard `MultiheadAttention` at moderate sizes** (transformer base
  3.27ms — competitive). It degrades at scale because attention is O(N²)
  in tokens × O(d) in features.

### What's slow

- **i018's `TacticalIncidenceBuilder`.** Many small irregular ops
  (elementwise multiplies, gathers, conditional masks for visible rays and
  blockers, pin computation). Each is fast in isolation; the per-forward
  overhead doesn't amortize at batch=1.
- **Python-loop scans inside primitives.** Several bt4-mixers (a025
  blocker_reset_ray_scan, a026 occlusion_semiring_ray_scan) took ~22 min/task
  vs ~6 min for matrix-heavy mixers, on GPU.

### Why my optimization attempts failed (i249)

Documented in EXPERIMENT_REPORT.md §6. Short version:
1. Diagnosed bottleneck from code reading, not from `torch.profiler`. The
   real GPU bottleneck is likely in the relation builder (which I did not
   touch), not the diffusion loop (which I "fixed").
2. Vectorized chunked einsum traded launch overhead for memory bandwidth
   pressure on bf16 intermediates. On the 8GB GPU this is a worse trade.
3. `torch.compile` with the chunked vectorization changed numeric paths
   under bf16 autocast enough to drift the final loss by 0.022 PR-AUC at
   scale_xl. Forward equivalence in eager fp32 did not detect it.

### How to make models faster (the protocol)

1. **Profile first, before changing any code.** Use `torch.profiler` on a
   real training step with mixed precision on. Identify the actual top-3
   kernels by total time. Do NOT trust code-reading intuitions.
2. **Try `torch.compile` alone** before any model change. Static-shape
   architectures like i018 should benefit from CUDA graph capture if it
   fires. If it doesn't help, that's information.
3. **Fuse the right thing.** For i018 specifically, the next attempt
   should target the `TacticalIncidenceBuilder`'s ~20-op elementwise chain,
   not the diffusion loop. JIT-script or write a custom CUDA kernel.
4. **Verify equivalence under deployment conditions.** Training-mode +
   bf16 autocast + dropout + 100 mini-batches with same seed. Forward-only
   fp32-eager is necessary but NOT sufficient.

### CPU-specific deployment wins

- **bt4_classifier (the conv tower) is the deployment-friendly story.**
  Fast at all scales, BLAS-friendly, +0.86 PR-AUC is already useful for
  puzzle filtering.
- **Distillation from i018 → bt4_classifier** would be the obvious
  follow-up: get i018's +0.03 PR-AUC into bt4_classifier's CPU speed
  envelope.
- **Quantization** (int8 conv) on bt4_classifier — never tried, would
  probably give another 2–3× CPU speedup at minimal accuracy loss.

---

## Failed experiments — what we should have done differently

### May 9 architecture scout failures

**What went wrong:**
- The scout attempted 234 tasks but only 175 completed cleanly. The rest were
  49 hard failures, 6 timeout/resume cases, 3 timeouts, and 1 artifact
  validation failure.
- Recurring failure modes were mundane but important: AMP half overflow, dtype
  mismatches between Half/Float tensors, `linalg.solve` dtype mismatch,
  `.view()` on non-contiguous tensors, and operators slow enough to time out.

**What I should have done:**
1. Added a real-training smoke gate before launching the full scout, not just
   import/forward checks.
2. Run the first epoch without AMP for fragile exotic-math models, or at least
   add dtype assertions around linalg/quantile/indexing paths.
3. Split "mathematically bad" from "engineering failed" in the report. A
   timeout or dtype crash is not evidence about the architecture's ceiling.

### i249 fast (the optimization disaster)

**What went wrong:**
- Claimed 1.5–2.5× speedup; got 0.9× (slower).
- Claimed numerical equivalence; got −0.022 PR-AUC at scale_xl.

**What I should have done:**
1. Run `torch.profiler` on i018 base for 10 training steps and read the
   actual top-K kernel breakdown.
2. Try `torch.compile(model, mode="reduce-overhead")` alone with NO model
   changes, measure.
3. Only if compile alone is insufficient, vectorize whatever the profiler
   actually said was slow (probably the incidence builder).
4. Verify equivalence under `model.train()` + bf16 autocast + dropout +
   100 minibatches with same seed, NOT in eval-fp32-no-compile.

### bt4-primitive-mixer scout (37 trained, 0 beat baseline)

**What went wrong:**
- Built 40 spatial mixers from primitives that were never designed to be
  spatial mixers. Most lost their chess prior in the adaptation.
- The smoke gate (forward-on-random-input + backward sum) was too weak —
  3 mixers passed the gate but failed training.

**What I should have done:**
1. Restrict to the ~15 primitives that ARE natively `(B,C,8,8)→(B,C,8,8)`
   token-mixers (legal_routing, legal_graph, ray_attention, scan families).
   Forcing the other 25 was always going to fail.
2. Pass `simple_18` board planes alongside `(B,C,8,8)` features into the
   mixer interface, so adapters could still derive rule-based masks.
3. Strengthen the smoke gate: 10 mini-batches of real training, loss must
   decrease monotonically. This would have caught a011/a013/a014 before
   they wasted GPU time.

### Primitive scout validation/artifact failures

**What went wrong:**
- `p001`-`p005` failed validation before useful training evidence existed.
- `i244` and `p031` produced artifacts but failed artifact validation with
  degenerate blank PR-AUC behavior.

**What I should have done:**
- Keep validation failures out of performance leaderboards, and label them as
  engineering failures until rerun.
- Make artifact validation block aggregate/report publication for that row, or
  at least render it in a separate "invalid metrics" table.

### Real transformer benchmark

**What went wrong:**
- Trained an authentic 4.8M–25M-param transformer with the repo's
  CNN-style hyperparameters (LR 7e-4, weight_decay 1e-4, plateau LR,
  no warmup).
- Degraded with scale (0.757 -> 0.640 -> 0.5875).

**What I should have done:**
- Used standard transformer training recipe from day 1 (warmup + cosine,
  AdamW(0.9, 0.95), scale-dependent peak LR, weight_decay 0.05, label
  smoothing 0.1).

### Self-monitor silently died

**What went wrong:**
- Forgot to add `lc0_bt4_transformer` to the monitor's keep-alive pgrep
  check when I added the pipeline. Monitor exited when all *other*
  pipelines finished while transformer was still running. ~22 hours of
  monitoring lost.

**What I should have done:**
- Made pipeline registration the same commit as the monitor update.
- Or: made the monitor auto-discover pipelines via a directory glob
  instead of a hard-coded list.

---

## Cross-cutting lessons

1. **Param count, FLOPs, and wall-clock speed are three independent things.**
   Don't predict any from another.
2. **Forward-pass smoke tests are weak gates.** A model that produces finite
   output on one batch can still fail training. Gates should test the
   property you care about — for training, that's "loss decreases over a
   real shard of data."
3. **Speed predictions from code reading have a 0/3 track record in this
   conversation.** Always measure before claiming.
4. **Chess-aware bias only works when it shapes the WHOLE architecture.**
   Falsifier-confirmed for i018. Wrenching a chess primitive into a generic
   tower destroys the bias (bt4-mixer scout: 0 of 37 beat conv).
5. **Width-scaling a working architecture is more reliable than grafting
   a primitive head onto it.** Quantitative: i018 base → scale_xl gives
   +0.015 PR-AUC; best hybrid graft gives +0.006 at +3× params.
6. **Attention did not rescue weak integration.** i242's global attention and
   the LC0 BT4 transformer both underperformed simpler conv/sheaf approaches
   under the current data and schedule. Exchange/king decomposition mattered
   more than global self-attention in the i242 ablations.
7. **Single-seed scout differences under 0.01 PR-AUC are noise.** Treat the
   scout as a 3-tier filter (competitive / weak / broken), not a ranking.
8. **Equivalence checks under training conditions** are required for any
   optimization claim. Eager-fp32-forward equivalence ≠ training-mode
   equivalence under bf16 autocast.
9. **Gates that don't cover the property you care about** silently let bad
   states through. The monitor death and the 3 mixer training failures are
   the same class of bug.
10. **Transformers and CNNs need different training recipes.** Applying the
   repo's standard CNN-style protocol to a transformer guarantees a bad
   result.
11. **Checkpoint metric must match claim metric.** If the claim is PR-AUC,
    checkpoint on PR-AUC. The reselection audit says F1-selected checkpoints
    often miss better validation PR-AUC epochs.
12. **Reports can have different denominators.** Check `state.json`,
    aggregate reports, and raw metric files together before saying "trained N
    models." In this repo, attempted tasks, completed runs, leaderboard rows,
    and valid metric artifacts are not the same number.
13. **Implementation status matters.** Only ~6 of 329 registry idea folders
    are tagged `probe_scaffold_only` — they are rare, not common — but if
    your "promising idea" is one of them, it is running through a shared
    `ResearchPacketProbe` wrapper rather than a bespoke implementation of
    the named mathematical architecture. Do not draw strong conclusions
    about an exotic idea unless its implementation is bespoke, or its
    scaffold status is explicitly part of the claim.
14. **LC0 wording is overloaded.** Model names/configs such as
    `lc0_bt4_classifier` refer to BT4-style encodings or trunks, while the
    `session/*.log` files repeatedly say the live LC0 ablation backend was
    disabled and fell back to classical evaluation. Keep encoding/model claims
    separate from engine-backend claims.
15. **The repo's own ground rules are written for a reason.** Treat
    single-epoch runs as smoke tests, not evidence. Compare against matched
    baselines, not unmatched ones. Use repeated seeds for any promotion
    claim. We violated these rules in several places (most notably the i249
    speedup claim).

---

## Knowledge gaps — what we still don't know

In rough order of "would change our beliefs most if measured":

1. **What does a properly-trained transformer score?** Current 0.757 is
   strong evidence against my training config, not against transformers.
   3-hour redo with proper recipe would settle whether the conv-tower's
   ~0.86 plateau is the architecture ceiling or the data ceiling.
2. **Can i242-style attention work with deeper data or a narrower operator?**
   One-seed i242 ablations say global attention did not beat i193 at scout
   scale, but they do not settle attention under full data, PR-AUC monitoring,
   or a chess-constrained sparse attention operator.
3. **Can a primitive's NATIVE form beat its bt4-mixer adaptation?** We
   have scout-only numbers for primitives in native form (single seed,
   base scale). Promotion-grade (3 seeds × 3 scales) for the top 5 (p034,
   p035, i248, p026, p030) would tell us if any of them is genuinely
   comparable to i018.
4. **Does i018 keep scaling beyond scale_xl?** scale_up→scale_xl gained
   +0.010. No data on scale_xxl or larger.
5. **What's the late-ensemble (probability average) of any two strong
   architectures?** The i255 trained-end-to-end fusion of i018 + i193 did
   not stack their wins (see Hybrids done so far). The cheaper question
   that has *not* been answered: do their actual `predictions_test.parquet`
   probabilities, averaged, beat either alone? Concretely: i018 +
   bt4_classifier is the most interesting pair because their inductive
   biases are most different (chess-structural vs dense conv) and their
   strengths are most different (accuracy vs speed). If that ensemble
   lifts, the i255 result was about the fusion mechanism, not about
   correlated predictions. If it doesn't lift, all chess-aware models on
   this task make highly correlated mistakes and no fusion will save
   them. ~30 minutes of post-hoc compute, no new training.
6. **Does distilling i018 → bt4_classifier work?** Could give us
   "i018 accuracy at bt4_classifier CPU latency."
7. **Effect of training on the full available data.** The canonical converted
   dataset at `data/processed/crtk_training_20260419_180229_fast.parquet`
   has **45,002,737 rows**. We train on 215,835 — about **0.5 %** of what is
   available. The current benchmark split is also marked **blocked** in
   `data/reports/benchmark_data_readiness.md` because the per-class quotas
   were underfilled. This is likely the single largest "what we never
   tried" — many of our architecture conclusions (especially
   "transformers are bad on this task") may simply be data-limited and
   would flip on the full 45M-row regime.
8. **Effect of richer encoding (lc0_bt4_112).** Untested with i018 and
   most primitives.
9. **Effect of data augmentation** (board flip, color flip).
10. **What's the actual GPU bottleneck in i018?** We never ran
   `torch.profiler`. The whole i249 effort was speculative as a result.
11. **Quantization for CPU deployment.** bt4_classifier at int8 quant
    might cut its already-fast 0.83ms further by 2–3×.

---

## Concrete recommendations for the next iteration

### For training

1. **Always profile before optimizing.** Build `scripts/profile_model.py`
   that wraps `torch.profiler` for any registered model name. Make it the
   default first step for any speed work.
2. **Repair or expand the canonical split before big claims.** The data
   readiness report is blocked; rerun split construction until target quotas
   are met, or rewrite the benchmark expectations to match the actual sample
   budget.
3. **Checkpoint on the metric you will claim.** For PR-AUC leaderboards, set
   the monitor/checkpoint metric to PR-AUC and retain per-epoch checkpoints or
   at least per-epoch test-disabled validation metrics.
4. **Report matched-recall near-puzzle FP.** Every promotion report should
   include near-puzzle false-positive rates at recall 0.8 and 0.85, plus the
   weak motif slices. This directly tests the benchmark's intended hard
   negative contract.
5. **Use the right schedule per architecture family.** CNNs: the standard
   recipe. Transformers: warmup_cosine + AdamW(0.9, 0.95) + wd=0.05 +
   scale-dependent peak LR + label smoothing.
6. **Strengthen smoke gates.** For any new architecture: require 10
   minibatches of real training with monotone loss decrease, not just
   "forward on random data produces finite output."
7. **Try data augmentation** (board flip with side-to-move toggle, color
   permutation under standard chess equivariance) on the existing best
   architectures. Cheap experiment, possibly large win.
8. **Verify any equivalence claim under deployment conditions** —
   train mode, autocast, dropout, multiple mini-batches.

### For architectures

1. **The next architecture should bake chess bias into the WHOLE design**,
   not just one block. i018's falsifier is the proof template.
2. **Test the falsifier on every new chess-aware architecture.** If
   scrambling the chess-specific structure doesn't hurt, it wasn't doing
   anything anyway.
3. **Try i018 + `lc0_bt4_112` encoding** as a one-line config change. May
   give the sheaf builder cheaper inputs.
4. **Try a real transformer with proper transformer training** before
   concluding attention doesn't work on this task.
5. **Treat i242 as a failed scout, not a dead end.** Its ablations say
   exchange/king decomposition mattered more than global attention, so the next
   attention attempt should either be much better trained or much more
   chess-constrained.
6. **Try LATE-ensembles** (post-hoc probability averaging) before any new
   end-to-end fusion attempts. The i255 trained fusion of i018 + i193
   failed to scale, so the next fusion attempt should first confirm that
   the *cheaper* form (averaging existing `predictions_test.parquet`
   files) actually adds signal. If late ensemble does not lift, no
   trained fusion will either. Highest-priority pair: i018 + bt4_classifier
   (most-different biases, most-different strengths). ~30 min compute.
7. **Try distillation** from i018 → bt4_classifier shape. After i255's
   failure this is the most promising remaining "combine two
   architectures" play — teacher-student lets one architecture inherit
   another's signal without paying the parallel-trunk cost.

### For primitives

1. **Only test primitives in their NATIVE form** at promotion grade. The
   bt4-mixer scout already proved drop-in mixer use doesn't work.
2. **For the ~10 attention/scan-shaped primitives**, also try them as
   REPLACEMENTS for attention inside the (properly-trained) transformer,
   not as conv replacements. That's the experiment that wasn't run cleanly
   — bt4-primitive-mixer used the conv tower, not the transformer.
3. **For primitives that fail standalone** (the bottom of the scout), do
   NOT scale up or adapt — they were genuinely weak. The scout is the
   filter.
4. **For any new primitive**, the gate should be:
   - forward + backward on random input (existing)
   - 10 mini-batches of real training with monotone loss (new)
   - falsifier-style ablation that swaps its distinguishing math for
     a random surrogate; if accuracy doesn't drop, the primitive is
     decorative (new)

### For deployment / inference speed

1. **Use bt4_classifier for CPU inference.** It's the only architecture
   we've tested that's deployment-fast at batch=1.
2. **Distill i018 → bt4_classifier** if you want i018-quality predictions
   at deployment speed. Not yet done.
3. **Quantize bt4_classifier to int8.** Not yet tried; likely another
   2–3× speedup at minimal accuracy loss.
4. **Don't optimize i018 by code-reading.** Profile first, target the
   incidence builder, verify under training conditions. The i249 attempt
   shows what happens when you skip these steps.

---

## How to use this document

When approaching a new experiment in this repo:

- **Skim the TL;DR + relevant section** before designing the experiment.
- **Check the Knowledge Gaps section** — if your idea is on that list,
  it's probably high-value because we genuinely don't know.
- **Check the Failed Experiments section** — if your idea is on that list,
  read what went wrong before trying again.
- **When you finish an experiment, update this file** with what you
  learned. The point of this document is to compound knowledge over time.

Companion documents:
- `reports/EXPERIMENT_REPORT.md` — specific recent experiments, with full
  numbers and per-experiment postmortems.
- `reports/aggregate_report.md` — current aggregate means across primitive,
  paper-grade, hybrid, falsifier, speed, BT4-mixer, and transformer runs.
- `reports/cpu_benchmark/results.md` — CPU latency benchmark used for the
  speed section.
- `results/paper_grade_top3/leaderboard_seed_summary.md` — paper-grade
  repeated-seed training throughput and aggregate metric means.
- `data/reports/benchmark_data_readiness.md` — split integrity and quota
  status; currently the source for the "underfilled benchmark" caveat.
- `docs/reliable_training_protocol.md` — the formal training tier
  definitions (smoke / triage / promotion / paper-grade).
- `reports/audits/paper_report.pdf` — the prior published paper results
  (234-architecture scout).
- `reports/audits/scout_all_runs.csv` — the underlying 234-architecture
  scout data.
- `reports/audits/pr_auc_reselection_report.md` — validation-only audit
  showing how often PR-AUC checkpointing would differ from F1 checkpointing.
- `reports/audits/matched_recall_fp_report.md` and
  `reports/audits/per_class_benchmark.md` — near-puzzle false-positive and
  weak-slice diagnostics.
- `reports/audits/i242_ablation_results.json` — one-seed i242 attention
  ablation summary.
