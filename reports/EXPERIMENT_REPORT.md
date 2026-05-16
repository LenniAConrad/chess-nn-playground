# Experiment Report — Conversation Summary

Generated: 2026-05-16. Covers all experiments and infrastructure built in a single
multi-day collaborative session. Written for the future-you who needs to know
what happened, what worked, what didn't, *and why*.

The point of this document is to **learn from the mistakes**, not just catalog
the wins. The biggest mistakes (i249 fast and the bt4-primitive-mixer scout)
are the highest-value sections.

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
| 7 | BT4 primitive-mixer scout (40 mixers × 1 seed × base) | 🔄 ~30/40 done | Early read: most underperform conv baseline (~0.86) by 0.04–0.07 |
| 8 | LC0 BT4 real transformer benchmark (9 tasks, 3 seeds × 3 scales) | ⏳ queued | Not started; ETA late morning today |
| 9 | CPU inference benchmark (3 models × 3 scales × 3 batch sizes) | 🔄 running | Pure forward-pass inference timing; runs concurrent with GPU pipelines |

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

### 7. BT4 primitive-mixer scout (40 mixers; early results)

Built a `bt4_primitive_mixer` model: BT4-style residual tower with swappable
per-block spatial mixer. Dispatched 8 parallel subagents to implement one
mixer adapter per primitive (40 files), each smoke-tested. All 40 idea folders
generated and validated. 30/40 trained so far.

**Headline:** most underperform the conv baseline (~0.86) by 0.04–0.07.

Concrete numbers from the most recent 6 (a025–a030, base scale, seed 42):

```
0.8143  a025 blocker_reset_ray_scan
0.7932  a026 occlusion_semiring_ray_scan
0.8194  a027 event_delta_bilinear_accumulator
0.8116  a028 occlusion_semiring_delta_bilinear_hyperedge
0.8216  a029 event_symmetric_interaction_accumulator
0.8195  a030 incremental_delta_linear_head      (1.27M params, no payoff)
```

For comparison, the conv-mixer baseline (which reproduces lc0_bt4_classifier)
hits ~0.86 with 447K params.

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

### 8. LC0 BT4 real transformer benchmark (queued)

Built an authentic encoder-only transformer (MHA + FFN pre-norm blocks over
the 64 square tokens), since the repo's `lc0_bt4_classifier` is conv, not a
transformer. Scales: base 4.8M / scale_up 16M / scale_xl 38M params. 9 tasks
queued (3 seeds × 3 scales), runs last in the GPU queue. **Results not yet
available** — this is the genuine "is attention better than conv on this task"
data point. Will reveal whether the BT4-conv plateau at 0.86 was the
architecture or the task ceiling.

### 9. CPU inference benchmark (running now)

`scripts/benchmark_cpu_inference.py` + `run_cpu_benchmark.sh`. Pure inference
(eval + no_grad + forward only) for i018 / lc0_bt4_classifier / lc0_bt4_transformer
at base / scale_up / scale_xl, batch sizes 1, 8, 32. Writes
`reports/cpu_benchmark/results.md`. The hypothesis is that i018's many small
ops *help* on CPU (no kernel-launch tax) where they *hurt* on GPU.

---

## Cross-cutting lessons

### Why my optimization attempts have a bad track record

The i249 disaster is the clearest example. The recurring pattern:

1. **Diagnose from code reading, not measurement.** This is faster but
   produces theories that sound plausible but mis-identify the bottleneck.
2. **Optimize one thing, claim it'll help the whole.** The vectorized sheaf
   loop *might* be faster in isolation; whether that matters depends on what
   fraction of total wall time the sheaf loop owns, which I never measured.
3. **Verify equivalence in wrong conditions.** Eager-fp32-no-dropout is a
   weak check for training-mode numerics.
4. **Predict speedup with confidence and no data.** I said "1.5–2.5×" with
   nothing to back it.

**Antidote:** profile first, change one variable at a time, verify under
training conditions, predict only after measurement. The order matters.

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

1. **CPU benchmark** — already running. Numbers in ~5–15 min. If i018
   inference is competitive, that's a deployment story regardless of GPU
   speed.
2. **Run the bt4-mixer scout to completion** then *exclude* the underperforming
   adaptations from any follow-up. Promote only ones that beat conv baseline.
3. **When transformer benchmark lands**, contextualize: is conv-tower's ~0.86
   plateau the architecture or the task? If transformer also plateaus there,
   the ceiling is data, not architecture.

### Medium-cost, high-information

4. **Promotion-grade the existing top primitives.** Take the top 5 from the
   primitive scout (p034, p035, i248, p026, p030) and run 3 seeds × 3 scales
   in their *native* form, not as mixer adapters. That's the apples-to-apples
   comparison with the paper-grade i018/i193/i011/bt4_conv numbers.
5. **Fix i249 properly** using the four-step protocol in §6. Don't claim
   speedup before measuring.

### Defensive / methodology

6. **Always profile before optimizing.** Add `scripts/profile_model.py` that
   wraps `torch.profiler` for any model name. Make it the default first step
   for any speed work.
7. **Always equivalence-check under training conditions.** Forward-eager-fp32
   matching is a necessary but not sufficient check.
8. **Always pre-register the comparison.** "Beats baseline" is meaningless
   without specifying which baseline (param-matched? FLOP-matched? same
   training budget?). Default to param-matched.

---

## What's still pending

- **CPU benchmark** finishing (5–15 min ETA from launch)
- **bt4-primitive-mixers** finishing the last ~10 tasks (~2 h)
- **lc0_bt4_transformer benchmark** running (~4 h after bt4-mixers)
- Eventually: a final aggregate report covering all 9 experiments

The self-monitor (PID 1, detached) keeps writing snapshots into
`reports/monitor/` every 30–60 min and rebuilds `reports/aggregate_report.md`
on the same cadence.
