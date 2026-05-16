# Experiment Report — Conversation Summary

Generated: 2026-05-16 (v2, updated with bt4-mixer final, CPU benchmark, transformer
results, and the monitor-bug postmortem). Covers all experiments and
infrastructure built in a single multi-day collaborative session. Written for
the future-you who needs to know what happened, what worked, what didn't,
*and why*.

The point of this document is to **learn from the mistakes**, not just catalog
the wins. The biggest mistakes — i249 fast, the bt4-primitive-mixer scout, my
transformer training config, and three wrong speed predictions in a row — are
the highest-value sections.

---

## TL;DR — Status of every experiment

| # | Experiment | Status | Verdict |
|---|---|---|---|
| 1 | Repo update + runner bug fix (`_kind_for_config`) | ✅ done | Fixed task-id collision that had silently broken the first primitive run |
| 2 | Primitive scout (40 primitives, 1 seed × base × 12 epochs) | ✅ done | **p034 octilinear_selective_scan top at 0.8809.** 5 pre-validation_failed (slug bug in p001–p005, upstream). |
| 3 | Paper-grade trunks + BT4-conv (4 × 3 seeds × 3 scales = 36 tasks) | ✅ done | **i018 scale_xl wins at 0.8901.** i018 keeps scaling; i193 and BT4-conv plateau. |
| 4 | i018 + primitive hybrids (4 × 3 seeds = 12 tasks) | ✅ done | Modest +0.006 vs i018-base for p019/p023/p034; p013 regresses. **Washes vs param-matched i018 scale_up.** |
| 5 | i018 falsifier (scrambled relation masks, 3 seeds) | ✅ done | **THESIS SUPPORTED, −0.0424 PR-AUC.** i018's chess geometry is doing real work. |
| 6 | i249 fast (i018 vectorized + torch.compile, 3 seeds × 3 scales) | ✅ done | **FAILURE on both axes: 0.90× speed (slower), up to −0.022 PR-AUC accuracy drift.** |
| 7 | BT4 primitive-mixer scout (40 mixers × 1 seed × base) | ✅ done (37 ok / 3 fail) | **0 of 37 beat the conv baseline.** Best: a031 ray_cast_obstacle_pool_head at 0.8414 (−0.018 vs conv). |
| 8 | LC0 BT4 real transformer benchmark (9 tasks, 3 seeds × 3 scales) | 🔄 8/9 done | **DEGRADES with scale: base 0.757, scale_up 0.640, scale_xl 0.59.** Probably my training-config fault (no warmup, CNN-style schedule). |
| 9 | CPU inference benchmark (3 models × 3 scales × 3 batch sizes) | ✅ done | **bt4_classifier is the FASTEST**, not i018. Another wrong speed prediction. |
| 10 | Self-monitor (auto-update + queue tracker) | ✅ alive (was buggy) | Forgot to include transformer in the keep-alive check → died yesterday 15:15. Fixed. |

---

## Per-experiment detail

### 1. Repo update + runner bug fix

Pulled 32 upstream commits introducing the p###-primitive infrastructure. Found
and fixed a real bug in `scripts/run_paper_ready_all.py:381` —
`_kind_for_config` used `path.as_posix().startswith("ideas/")`, which fails for
*absolute* idea paths, so every idea collapsed into the single benchmark
task_id `benchmark_config_seed42`. Tasks 2–40 of the first primitive run
attempted to resume from task 1's checkpoint with a different architecture and
crashed in ~12s each. One-line fix: check `"ideas" in path.parts` instead.

**Lesson:** trust runner output, not vibes. The very first primitive run showed
"40/40 attempted" and exited cleanly — but every single task had failed at
checkpoint load. The state.json + per-task log inspection caught it in
minutes; the high-level launcher log lied by omission.

### 2. Primitive scout (35 of 40 trainable)

After the bug fix, all 40 primitive configs ran (5 failed at idea-validation
because their `idea.yaml slug` had a `_network` suffix the folder didn't —
upstream metadata bug, not our run).

Top results (single seed, base scale, 12 epochs, simple_18):

```
0.8809  p034_octilinear_selective_scan
0.8778  p035_sparse_legal_graph_transition
0.8778  i248_rule_aware_tactical_head
0.8777  p026_ray_cast_obstacle_pool_head
0.8771  p030_ray_parallel_ssm_head
0.8764  p010_ray_occlusion_semiring_scan
0.8764  p007_attack_ray_sparse_attention
```

**Lesson:** scout-grade single-seed differences of 0.005 are noise. The top
~10 cluster within seed-noise of each other. Treat this as a 3-tier filter
(competitive ≥ 0.85 / weak 0.75–0.85 / broken < 0.75), not a fine ranking.

### 3. Paper-grade trunks + BT4-conv (3 seeds × 3 scales, 36 tasks)

Headline finding — i018 keeps scaling while the rest plateau:

```
                       base       scale_up   scale_xl
i018 oriented_sheaf    0.8752     0.8799     0.8901   ← keeps gaining
i193 dual_stream       0.8761     0.8765     0.8770   ← plateau
bt4_conv (lc0_bt4)     0.8589     0.8627     0.8619   ← plateau, slightly drops at xl
```

At matched ~500K params: **i018 (0.8901) beats bt4_conv (0.8589) by +0.031.**
At matched 91K, i018 still beats bt4_conv's 501K version by +0.016.

**Lesson:** "BT4 is a weak baseline" framing is *partially* misleading. What
the repo calls BT4 is a residual CNN tower with SE blocks — it only borrowed
the name from Lc0's actual BT4 transformer. The conv tower is genuinely not
the right inductive bias for puzzle_binary; whether a *real* transformer
(experiment 8) does better is the open question.

### 4. i018 + primitive hybrids (gated-logit fusion)

Grafted 4 primitives onto i018 via `final_logit = sheaf_logit + sigmoid(gate) * primitive_logit`:

```
i018 base baseline (3 seeds)          0.8752 ± 0.0045
i018 + p019 (kernel memory)           0.8817   (+0.0065)
i018 + p034 (SSM scan)                0.8812   (+0.0060)
i018 + p023 (bilinear hyperedge)      0.8808   (+0.0056)
i018 + p013 (sparse delta accum)      0.8693   (-0.0059)  REGRESSION
```

But the hybrids are 270–307K params and i018 *scale_up* (234K) already hits
0.8799 — so against param-matched i018, **the lift evaporates** (p019 only
+0.002 vs scale_up i018). Width-scaling i018 is a more reliable lever than
grafting a primitive head.

**Lesson:** "we beat the baseline" is not the right comparison. Beating
param-matched baseline is the only one that counts. A 270K-param model beating
a 91K-param model just means "more parameters help."

### 5. i018 falsifier — the cleanest positive result we have

Replaced i018's 12 typed chess-relation masks with degree-preserving random
column permutations (preserves per-source out-degree, randomizes which targets
each square reaches). 3 seeds at base:

```
Baseline (real chess geometry)    0.8752 ± 0.0045
Falsifier (scrambled relations)   0.8328 ± 0.0012
Delta                             -0.0424   THESIS SUPPORTED (≥ 0.02 threshold)
```

**This is the single most defensible finding from the whole conversation.**
i018's sheaf math is not decorative; it genuinely needs the real chess
relation graph. The −0.042 PR-AUC collapse is far past noise and rules out the
"the readout MLP does all the work" null hypothesis.

### 6. i249 fast — what went wrong, and how to actually do it

I built i249 as an "execution-optimized i018": same math, vectorized chunked
sheaf-diffusion loop + optional `torch.compile`. I claimed numerical equivalence
(forward logits to 1e-8 and gradients to 1e-10 on CPU eager) and predicted
1.5–2.5× speed.

**Actual results (3 seeds × 3 scales):**

```
              speed (vs i018)    accuracy delta (vs i018)
base          0.88x  (slower)    -0.0061
scale_up      0.89x              -0.0080
scale_xl      0.93x              -0.0218   way past seed noise
```

**Failure on both axes.** Slower AND less accurate, especially at scale_xl.

#### Why it failed

**Speed regression:**
1. I diagnosed the bottleneck *from code reading*, not from a profiler. I
   assumed the per-relation Python loop was launch-overhead-bound on GPU. The
   real GPU bottleneck may be elsewhere (likely the TacticalIncidenceBuilder
   with its 12 elementwise-multiply chain creating `(B, 12, 64, 64)` masks,
   which I never touched).
2. My "vectorization" traded 12 small `bmm` calls for one big `einsum` over a
   chunked `(B, chunk, 64, 64, stalk)` intermediate. On an 8 GB GPU with
   bf16 autocast, that intermediate is memory-bandwidth-bound. PyTorch's
   small-matmul fast paths exist for a reason — I removed them.
3. `torch.compile` with `mode="reduce-overhead"` adds CUDA graph capture
   overhead per scale change; I never measured whether it actually fired
   cleanly vs falling back to eager. May have done nothing.

**Accuracy regression:**
1. I verified equivalence in **eval mode on CPU with fp32, no dropout, no
   compile**. Training uses **bf16 autocast + dropout + compile**. Those
   numeric paths are different and small differences compound across thousands
   of optimizer steps.
2. The chunked einsum changes accumulation order. In fp32 forward that's
   ~1e-8; in bf16 training it's enough to converge to a different local
   minimum (−0.022 at scale_xl is unambiguous).

#### How to actually analyze i249 properly

**Step 0 — profile, don't speculate.** Run i018 base for 10 training steps under
`torch.profiler` with CUDA activity. Get the actual top-K kernel time
breakdown:

```python
with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            record_shapes=True, profile_memory=True) as prof:
    for _ in range(10):
        loss = model(x).sum(); loss.backward()
prof.export_chrome_trace("i018_trace.json")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
```

This tells you *which kernels actually dominate*, not which ones look slow in
the source. Likely culprits: `aten::bmm`, `aten::mul`, `aten::scatter_add` in
`TacticalIncidenceBuilder._visible_rays` / `_pin_relation`. Optimize the kernel
that actually shows up, not the one you assumed dominated.

**Step 1 — try `torch.compile` ALONE first.** No model changes. If
`torch.compile(model, mode="reduce-overhead")` on stock i018 gives a real
speedup, you don't need any vectorization. Measure: forward + backward
samples/sec, vs uncompiled. *Don't* trust forward-only or eval-only timings —
training has different kernels (gradient checkpointing, optimizer steps).

**Step 2 — if compile alone is insufficient, vectorize the right thing.**
The TacticalIncidenceBuilder builds the `(B, 12, 64, 64)` relation tensor with
~20 elementwise ops. Fuse those into one custom op (or a single jit-script).
That's where the 12-relation per-block loop's input *comes from* — and it
runs once per forward, vs the diffusion loop which runs per block.

**Step 3 — verify equivalence under the REAL training conditions.** Equivalence
checks must use: `model.train()`, bf16 autocast, dropout active, `torch.compile`
on. Run 100 mini-batches of both versions with the same seed and compare final
loss + a sample of parameter values. If they diverge under training mode,
you've already broken what you set out to preserve.

**Step 4 — only then commit and run the 9-task benchmark.** Not before.

The i249 idea folder and module are still in the repo; treat them as
a documented failure case. Don't delete — they're useful negative evidence.

### 7. BT4 primitive-mixer scout (40 mixers — FINAL, 37/40 trained, 3 failed)

Built a `bt4_primitive_mixer` model: BT4-style residual tower with swappable
per-block spatial mixer. Dispatched 8 parallel subagents to implement one
mixer adapter per primitive (40 files), each smoke-tested. All 40 idea folders
generated and validated.

**Final result: 0 of 37 successful runs beat the conv baseline (~0.86).** Top 5:

```
0.8414  a031 ray_cast_obstacle_pool_head        ← best, but -0.018 vs conv
0.8301  a034 occlusion_aware_ray_scan_head
0.8296  a005 rule_aware_tactical_head
0.8251  a016 legal_edge_compile_scatter
0.8250  a040 sparse_legal_graph_transition
```

The bottom of the leaderboard sits in the 0.78–0.80 range. **Best primitive
mixer is 0.045 below conv; median is closer to 0.06–0.07 below.**

**3 failed during training** despite passing forward smoke-tests:
- a011 move_graph_router
- a013 rule_conditioned_sparse_attention
- a014 legal_move_graph_delta

All three are from the **legal_routing family** — same subagent batch. The
forward smoke test ran random `(2, 64, 8, 8)` input and checked shape +
finiteness + backward sum. That's not sufficient: a smoke test passing a
trivial input does not mean the module trains stably over hundreds of
mini-batches with real data, mixed precision, and gradient accumulation. **The
gate was too cheap.** A real gate would do at least 10 mini-batches of actual
training on a tiny data shard and confirm loss decreases monotonically.

#### Why this is happening — and what to do about it

**Root cause: most primitives are not natively token-mixer-shaped.** Of the 40
primitives, only ~10–15 (the legal-graph, ray-attention, scan families) are
naturally `(B, C, 8, 8) → (B, C, 8, 8)` operators. The other ~25 are
standalone models — they have their own trunks, heads, accumulators, and
expect to consume the *full* simple_18 board with semantic piece planes.

When a subagent had to wrap one of those into a spatial mixer, two things
happened:

1. **The primitive lost its rule-derived inputs.** Many primitives depend on
   piece-type masks (knight squares, pawn directions). Inside a mixer the
   input is `(B, C, 8, 8)` with channel `C` semantically opaque — there's no
   way to know which channel is "white knight." The subagents replaced
   rule-derived features with content-derived surrogates (sigmoid of a learned
   linear). This is honest engineering, but it strips the primitive of the
   chess prior that was its whole point.

2. **The mixer slot is too narrow.** The BT4 block wraps `mixer(x)` with
   SqueezeExcite + residual + activation. If the mixer outputs something
   subtle (a small per-square message), the residual `x + se(mixer(x))` lets
   the block pass `x` through nearly unchanged, defaulting toward identity.
   Primitives designed for *whole-network* signal flow look like dropout when
   wedged into a 1-block mixer slot.

**The strong signal we DO have (the falsifier) reinforces this:** i018 works
because its chess prior is baked into the *whole architecture* — the typed
relation builder, the per-relation restriction maps, the readout. Wrenching
one piece out and dropping it into a different scaffold loses what made it
work.

**Recommended fixes for the bt4-mixer line:**

- Restrict the experiment to the ~10–15 natively-token-mixer primitives.
  Drop a001–a005, a017–a020, a022, a024, a025, a027–a030 from a follow-up.
- For the surviving ones, allow the mixer to read piece planes — pass the raw
  simple_18 tensor alongside the channel-opaque feature map so it can derive
  rule masks. This restores the chess prior.
- Use a *deeper* trunk than the i018 baseline so the mixer's signal has room
  to compound across blocks. One-block conv-vs-primitive isn't a fair fight.

### 8. LC0 BT4 real transformer benchmark — UNDERPERFORMS, my training config is to blame

Built an authentic encoder-only transformer (MHA + FFN pre-norm blocks over
the 64 square tokens), since the repo's `lc0_bt4_classifier` is conv, not a
transformer. **Final actual param counts** (runner-scaled): base 4.8M /
scale_up 12.5M / scale_xl 25.4M (not the 16M / 38M I designed — the runner's
scale keys multiplied width and depth jointly in a way I didn't predict).

**Results (8/9 done, scale_xl_seed44 still in flight):**

```
                 params       test PR-AUC (3 seeds)           mean
base              4.8M        0.7692 / 0.7546 / 0.7475       0.757
scale_up         12.5M        0.6318 / 0.6879 / 0.6010       0.640
scale_xl         25.4M        0.5665 / 0.6190 / (in flight)  ~0.59
```

Reference: bt4_classifier conv tower ~0.859, i018 ~0.890. **The transformer at
base is 0.10 PR-AUC below the conv tower, and degrades from there.**

#### Why this is bad — and probably my fault, not the architecture's

Three structural problems with my benchmark config:

1. **Wrong learning-rate schedule.** I used `lr=0.0007` with `reduce_on_plateau`
   — that's the schedule i018 / bt4_conv use. Transformers basically need
   **linear warmup + cosine decay**: ramp from 0 to peak over the first ~5% of
   steps, then cosine-decay to ~10% of peak. Without warmup, the first
   optimizer step on freshly-initialized attention layers can blow up
   gradients; without cosine decay, large transformers oscillate.
2. **Wrong peak LR for size.** A single fixed LR across all three scales is
   wrong. Larger transformers want lower peak LRs (e.g. base 3e-4, scale_up
   2e-4, scale_xl 1e-4). I gave them all 7e-4 — too hot for the bigger ones.
3. **No regularization for size.** `weight_decay=0.0001` is fine for a 100K
   conv but anemic for a 25M transformer on a 240K-sample dataset. Standard
   transformer recipe: weight_decay around 0.05, dropout active, label
   smoothing 0.1, AdamW with `betas=(0.9, 0.95)`. None of those are in my
   config.

The **monotone degradation across scales** (base 0.76 → scale_up 0.64 →
scale_xl 0.59) is the smoking gun: a properly-trained bigger transformer
should at minimum *not get worse*. Mine gets dramatically worse, which means
optimization is failing at scale.

#### Recommended retry config (small, focused)

A redo of just `bench_lc0_bt4_transformer` with:

```yaml
training:
  optimizer: adamw
  adamw_betas: [0.9, 0.95]
  weight_decay: 0.05
  lr_scheduler:
    name: warmup_cosine
    warmup_pct: 0.05
    min_lr_ratio: 0.1
  base_lr_per_scale:
    base:     3.0e-4
    scale_up: 2.0e-4
    scale_xl: 1.0e-4
  label_smoothing: 0.1
```

(The trainer would need a small extension to honor those keys — they may not
all exist yet.) That'd give a fair "transformer vs conv on this task" answer
in another 3-hour run. Without this, the current numbers are evidence about
*my training config*, not about transformers.

#### What this DOES tell us

The current bad numbers are still informative: **off-the-shelf CNN-style
training does not work for a transformer on this dataset.** So if someone in
the future grabs `lc0_bt4_transformer` and runs it with the same hyperparams
as the other models in this repo, they will get a bad result and it won't be
the architecture's fault.

### 9. CPU inference benchmark — wrong prediction (third time)

`scripts/benchmark_cpu_inference.py` + `run_cpu_benchmark.sh`. Pure inference
(eval + no_grad + forward only) for i018 / lc0_bt4_classifier /
lc0_bt4_transformer at base / scale_up / scale_xl, batch sizes 1, 8, 32. My
prediction going in: i018's many small ops would *help* on CPU because there's
no kernel-launch tax.

**Actual batch=1 per-position latency (the realistic chess-engine number):**

```
                  base       scale_up   scale_xl
bt4_classifier    0.83 ms    1.45 ms    2.26 ms    ← FASTEST at every scale
bt4_transformer   3.27 ms    8.65 ms    21.09 ms
i018 sheaf        5.28 ms    7.45 ms    11.17 ms   ← SLOWEST at base/scale_up
```

**bt4_classifier is 6.4× faster than i018 at batch=1, despite having 5.5×
MORE parameters** (501K vs 91K). Wrong prediction. Reason: bt4_classifier is
dense `Conv2d` over a small spatial map, which maps perfectly to MKL/BLAS.
i018's `TacticalIncidenceBuilder` has substantial Python-level overhead per
forward (irregular elementwise ops to build the 12-relation tensor) that
doesn't amortize at batch=1. At batch=32 i018 drops to 1.75 ms/sample (570
samples/sec) which IS competitive — but chess engines run at batch=1.

**Three wrong speed predictions in a row** (i249 1.5–2.5× faster → 0.9×;
transformer "GPU-efficient" → unknown but probably yes on GPU; i018 "CPU-
friendly" → slowest). The pattern: I predict speed from code reading, the
measurement disagrees. Antidote in the cross-cutting section.

i018's value is **accuracy per parameter**, not speed. At matched ~500K
params it still beats bt4_classifier by +0.031 PR-AUC. Different tradeoff
than the deployment-speed story.

Full table in `reports/cpu_benchmark/results.md`.

### 10. Self-monitor — silently died, my bug

When I built the monitor and later added the bt4-primitive-mixers pipeline, I
included its pgrep checks. But for the transformer pipeline I forgot to add
`lc0_bt4_transformer` to the "is any pipeline still alive" check. So once
hybrid / falsifier / i249 / bt4-mixers had all completed and only the
transformer was still running, the monitor saw "all done" and exited cleanly
— while one pipeline was very much not done. Last tick: 2026-05-15 15:15.

The bug was in `scripts/monitor_self_schedule.sh` — the keep-alive `if`
condition was missing the transformer pids. Fixed by adding `tfm_run_pid` /
`tfm_waiter_pid` to both the pgrep block and the exit condition.

**Lesson:** when I added new pipelines (transformer, CPU benchmark), I should
have updated the monitor in lockstep. I didn't. Same class of error as the
"3 mixers passed forward smoke test but failed training" — gates that don't
cover the relevant property silently let bad states through.

---

## Cross-cutting lessons

### Why my speed predictions have a 0/3 track record

Three confident speed predictions, all wrong:

| Prediction | What I said | Reality |
|---|---|---|
| i249 vs i018 GPU training | 1.5–2.5× faster | **0.9× (slower)** |
| Transformer scaling vs CNN | implied "competitive" | **monotone collapse with size** |
| i018 CPU inference | "shines on CPU, small ops help" | **slowest of the three** |

The recurring pattern:

1. **Diagnose from code reading, not measurement.** Theories that sound
   plausible but mis-identify the bottleneck (i249's bottleneck was probably
   in the relation builder, not the diffusion loop I "fixed").
2. **Optimize one thing, claim it'll help the whole.** Even if the change is
   locally a win, you don't know what fraction of wall time it owns.
3. **Verify equivalence in wrong conditions.** Eager-fp32-no-dropout matched
   for i249; bf16-amp-with-dropout-and-compile diverged by 0.022.
4. **Predict speedup with confidence and no data.** "1.5–2.5×" was nothing
   but vibes.

**Antidote — the protocol I should follow next time:**
1. Profile first (`torch.profiler` on a real training step). See what
   actually dominates wall time.
2. Change one variable at a time. Don't bundle "vectorize + compile" into
   one PR.
3. Verify equivalence under the actual deployment conditions (training mode
   if you're predicting training speed, bf16 autocast if that's enabled).
4. Predict only after at least one measurement of the modified version. No
   pre-measurement speedup claims.

### Why my gates were too weak

The 3 bt4-mixer training failures and the monitor's silent exit are the same
class of error: **a gate that doesn't cover the property you care about lets
bad states through silently.**

- Forward smoke test on random input ≠ "this will train without diverging".
- Monitor checking the pipelines I remembered ≠ "monitor stays alive while
  any pipeline runs".

**Antidote — a gate is only as strong as its weakest test.** When introducing
a new gate, write down what it claims to verify, and brutally interrogate
that claim. "Builds and returns finite outputs on one batch of random data"
is a *very weak* claim about training stability. "Will keep the monitor alive
through all pipelines" requires actually listing all pipelines, not just the
ones you remember.

### Why our primitives have been underperforming as drop-in components

Three structural reasons, in order of importance:

1. **Inductive bias dies when you wrench out the operator.** The falsifier
   proved i018's chess geometry is load-bearing for i018's score. The same is
   true of every primitive — it was designed within a particular pipeline
   that supplied rule-derived inputs. Stripped of those inputs (in a mixer
   slot, in a hybrid head), the primitive is doing math on the wrong tensor.

2. **Most primitives don't have a defensible inductive bias to begin with.**
   The scout shows a cluster at 0.85–0.88 — barely above the conv baseline at
   0.86. Many primitives are research scaffolds that *propose* a structure
   but never proved that structure helps over a CNN of matched capacity.
   i018 is the rare exception (validated by the falsifier).

3. **Hybrid grafts can't beat width-scaling.** The hybrid experiment showed
   that adding a primitive head adds ~0.006 PR-AUC at 3× the params, but
   simply growing i018 width adds +0.005 base→scale_up and +0.010 →scale_xl.
   Width is reliable; grafts are noisy and small.

### What the falsifier proved (the one ironclad result)

A learned cellular sheaf over a *real chess relation graph* genuinely
captures puzzle-likeness in a way that the same sheaf over a random graph
cannot. This is the strongest evidence we have that chess-aware inductive
bias matters at all on this task. It rules out the trivial null "the readout
MLP carries everything." Any followup work should build on this — start from
i018 and add bias, don't start from BT4 and add primitives.

---

## Recommended next steps

### Highest-value, lowest-cost

1. **Retry the transformer with proper transformer training** (warmup +
   cosine, AdamW betas (0.9, 0.95), weight_decay 0.05, scale-dependent peak
   LR, label smoothing) — see §8 for the exact recipe. The current numbers
   are evidence about my config, not about transformers. ~3-hour GPU run.
2. **Promotion-grade the bt4-mixers top entries** at 3 seeds × base just to
   confirm the single-seed numbers (a031 / a034 / a005). It would not change
   the verdict — they're 0.04+ below conv — but it nails it down.
3. **Drop the bt4-mixer approach for the underperformers.** 30+ of the 37
   trained mixers sit 0.05–0.07 below conv. They're done. The 3 failed runs
   (a011/a013/a014, legal_routing) aren't worth debugging — the family is
   already represented by the surviving members.

### Medium-cost, high-information

4. **Promotion-grade the existing top primitives in their NATIVE form.** Take
   the top 5 from the primitive scout (p034, p035, i248, p026, p030) and run
   3 seeds × 3 scales as standalone models, not as mixer adapters. That's
   the apples-to-apples comparison with the paper-grade i018/i193/i011/bt4
   numbers.
5. **Fix i249 properly** using the four-step protocol in §6: profile first
   on GPU, try `torch.compile` alone before changing any model code, fuse
   the right thing (probably the relation builder, not the diffusion loop),
   verify under training conditions.

### Defensive / methodology

6. **Always profile before optimizing.** Add `scripts/profile_model.py` that
   wraps `torch.profiler` for any model name. Make it the default first step
   for any speed work.
7. **Always equivalence-check under training conditions.** Forward-eager-fp32
   matching is necessary but not sufficient.
8. **Always pre-register the comparison.** "Beats baseline" is meaningless
   without specifying which baseline (param-matched? FLOP-matched? same
   training budget?). Default to param-matched.
9. **Strengthen the mixer smoke gate** before any future "subagent implements
   N variants" pattern. Random-input forward+backward is too weak. Minimum
   gate: 10 mini-batches of actual training on a tiny shard with monotone
   loss-decrease check.
10. **Add new pipelines to the monitor IN THE SAME COMMIT.** The transformer
    pipeline got built without updating the monitor's keep-alive list, and
    the monitor died the moment all *other* pipelines finished. Either make
    pipeline registration auto-discover pipelines, or treat the monitor
    update as part of the pipeline-PR's checklist.

---

## What's still pending

- **lc0_bt4_transformer scale_xl_seed44** — the final transformer task,
  in flight as of this writing (epoch 22/30). Won't change the verdict —
  the existing scale_xl seed42/seed43 are both around 0.6 and seed44 won't
  rescue that.
- Self-monitor is restarted and tracking all five GPU pipelines + transformer
  correctly. Writes snapshots into `reports/monitor/` every 60 min and
  rebuilds `reports/aggregate_report.md` on the same cadence.

Once the last transformer task finishes (~15–25 min), everything in this
report is final.
