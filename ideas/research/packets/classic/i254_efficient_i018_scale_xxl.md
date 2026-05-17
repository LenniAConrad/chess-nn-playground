# i254_efficient_i018_scale_xxl.md

## Thesis

The right scale-XXL successor to i018 is **not** “more of everything.” The repo evidence points in a more specific direction: i018 improves from base to scale_up to scale_xl, the falsifier shows that the typed chess relation graph is load-bearing, hybrid grafts wash out against a param-matched larger i018, and i249 failed because it bundled speculative performance changes with an execution rewrite that was only checked in eval-mode fp32 rather than under the actual training path. The efficient XXL design should therefore **keep the relation-builder thesis fixed, scale the regular dense parts first, keep the sheaf core numerically conservative, and gate all speed work behind profiler evidence**. fileciteturn17file0L3-L3 fileciteturn24file0L3-L3

The strongest empirical facts in the repo are these: paper-grade i018 improves from **0.8752** at base to **0.8799** at scale_up and **0.8901** at scale_xl; the falsifier that scrambles typed relations drops mean test PR-AUC to **0.8328**, a **-0.0424** hit; and i249, which tried to optimize execution by vectorizing the sheaf block and optionally compiling it, ended up **slower** and **less accurate**, with the scale_xl accuracy drift reaching **-0.0218** versus i018. The experiment report explicitly attributes that failure to profiling the bottleneck from code-reading rather than measurement, and to verifying equivalence only in eval-mode fp32 rather than in the bf16/autocast training path. fileciteturn17file0L3-L3

That means the design goal for i254 is narrower than “make i018 fast.” The goal is: **preserve the current 12-relation tactical incidence thesis, preserve the current training numerics unless a train-mode parity ladder passes, and move scale budget toward capacity that does not multiply the number of irregular relation operations**. In practice, that means wider square-token channels and a much larger readout head are the first knobs; extra sheaf depth is a secondary knob; larger stalks are a late, profiler-gated knob; and fused or compiled incidence work is a separate execution branch, not part of the first architecture benchmark. fileciteturn4file0L3-L3 fileciteturn17file0L3-L3

## What should scale and what should stay fixed

The parts that should stay fixed in the first XXL run are the ones that encode the actual chess thesis: the side-to-move canonicalization in `BoardStateAdapter`, the **12 typed tactical relations** in `TacticalIncidenceBuilder`, the existing ray and pin logic, the triad-defect semantics, and the fact that the model operates on a learned sheaf over a typed tactical incidence complex rather than replacing that graph with generic convolutions. The architecture and code both make the current thesis explicit: the builder emits a dense `(B, R, 64, 64)` tensor with `R = 12`, and the current sheaf block uses learned source/target restriction maps per typed relation. Those are the invariants that the falsifier actually validated. fileciteturn4file0L3-L3 fileciteturn7file0L3-L3 fileciteturn9file0L3-L3

The parts that should scale first are the ones that buy capacity without multiplying the repo’s known irregular work too aggressively. The current base config is `channels=64`, `hidden_dim=96`, `depth=2`, `stalk_dim=8`, and the repo’s benchmarking script mirrors the current i018 scale ladder as **base = (64, 2, 8)**, **scale_up = (96, 3, 8)**, and **scale_xl = (128, 4, 8)** for `(channels, depth, stalk_dim)`. That existing ladder already tells us something important: the repo has scaled **channels and depth**, but it has **not** scaled stalk dimension. A scale-XXL design should continue that bias and add a much larger readout hidden dimension, because the readout is dense regular compute while the relation builder and per-block relation work are the irregular parts that already dominate the model’s speed reputation. fileciteturn19file0L3-L3 fileciteturn22file0L3-L3 fileciteturn17file0L3-L3

The practical scaling rule is therefore:

| Component | First XXL action | Why |
|---|---|---|
| `BoardStateAdapter` | keep fixed | preserves the mover-oriented thesis |
| `TacticalIncidenceBuilder` relation set and semantics | keep fixed | this is the chess relation thesis validated by the falsifier |
| Relation count `R=12` | keep fixed | adding relations changes the thesis instead of scaling it |
| Stalk dimension `s` | keep at `8` in the first run | stalk-only scaling increases the expensive `64×64×s` sheaf work fastest |
| Token width `channels` | scale up | buys regular dense capacity and historically helped i018 |
| Readout hidden size `hidden_dim` | scale aggressively | adds capacity with very small extra irregular work |
| Sheaf depth | scale cautiously, if at all | every extra block multiplies relation work |
| Compiled/fused incidence ops | defer until profiling | i249 is a direct warning against doing this speculatively |

That table is the synthesis of the repo’s architecture, the current scale definitions, the falsifier, and the i249 postmortem. fileciteturn4file0L3-L3 fileciteturn19file0L3-L3 fileciteturn22file0L3-L3 fileciteturn17file0L3-L3

A useful extra observation from the current code structure is that **readout scaling is especially attractive**. The current head is `LayerNorm -> Linear(readout_dim, hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim, 1)`, and `readout_dim` itself is driven by pooled node summaries plus 12-relation diagnostics and small board statistics. Increasing `hidden_dim` therefore increases parameters a lot more than it increases the irregular relation work, because it does not change the relation builder, relation count, or stalk dimension. That makes “larger readout with fixed builder” the cleanest way to test whether i018 still benefits from more dense capacity. fileciteturn9file0L3-L3 fileciteturn10file0L3-L3

## Low-rank and grouped sheaf math

The current i018 sheaf block is built around the standard relation-weighted coboundary
\[
(\delta_\rho h)_{(u,v,r)} \;=\; \sqrt{w_{uvr}}\Big(\rho^{\text{dst}}_r h_v \;-\; \sigma_r \rho^{\text{src}}_r h_u\Big),
\]
with learned relation-specific restriction maps \(\rho^{\text{src}}_r,\rho^{\text{dst}}_r \in \mathbb{R}^{s\times s}\), fixed signs \(\sigma_r \in \{-1,+1\}\), and a heat-step update driven by \(\delta_\rho^\top \delta_\rho\). Because the operator is built as \(\delta^\top\delta\), the resulting sheaf Laplacian is symmetric positive semidefinite for any choice of linear restrictions; changing how the restriction maps are parameterized does **not** change that fact. That is the main mathematical reason a grouped or low-rank restriction family is safe to consider without abandoning the sheaf thesis itself. fileciteturn24file0L3-L3 citeturn15academia2

A grouped low-rank parameterization can be written as follows. Partition the 12 relations into semantic groups \(g(r)\), for example:

- **attack group**: direct attacks on pieces, empty near-king attacks, knight attacks, pawn attacks  
- **defense group**: the two defense relations  
- **ray group**: bishop, rook, and queen visible rays  
- **pin group**: king-ray pin candidate

Then parameterize each restriction map as
\[
\rho^{\text{src}}_r \;=\; I_s + U^{\text{src}}_{g(r)} \operatorname{Diag}(a^{\text{src}}_r) V^{\text{src}\top}_{g(r)},
\qquad
\rho^{\text{dst}}_r \;=\; I_s + U^{\text{dst}}_{g(r)} \operatorname{Diag}(a^{\text{dst}}_r) V^{\text{dst}\top}_{g(r)},
\]
where \(U_g,V_g \in \mathbb{R}^{s\times k}\) are **group-shared** bases and \(a_r \in \mathbb{R}^k\) are **relation-specific** diagonal coefficients with \(k \ll s\). If more flexibility is needed later, \(\operatorname{Diag}(a_r)\) can be replaced by a small \(C_r \in \mathbb{R}^{k\times k}\), but the diagonal form is the safer first structured variant. This preserves typed relations while sharing statistical strength within semantically similar relation families. The resulting sheaf is still a learned cellular sheaf over the same 12-edge tactical complex; only the internal parameterization changes. fileciteturn24file0L3-L3 fileciteturn9file0L3-L3

For the current i018 defaults, the parameter arithmetic is revealing. With \(R=12\) relations and stalk size \(s=8\), the current full-map restriction parameters per block are
\[
2Rs^2 = 2 \cdot 12 \cdot 8^2 = 1536
\]
parameters. A grouped-diagonal low-rank family with \(G=4\) groups and rank \(k=4\) instead uses
\[
4Gsk + 2Rk = 4\cdot 4\cdot 8\cdot 4 + 2\cdot 12\cdot 4 = 608
\]
restriction parameters per block. That is a **60% map-parameter reduction**. However, the repo’s current stalk size is already small, and at \(s=8\) the **static compute** benefit of low-rank application is not compelling in eager PyTorch; the dominant `64×64×s` edge work remains, and factorization overhead can erase any gain. The grouped low-rank family becomes much more interesting only if a later experiment wants \(s=12\) or \(s=16\), because then it prevents the restriction-map cost from growing quadratically with \(s\). In other words: **grouped low-rank maps are a good way to make stalk scaling safer, but they are not the first efficiency lever at the current \(s=8\)**. That is also aligned with recent sheaf work reporting strong results even with highly structured—sometimes even diagonal—restriction families, rather than relying on ever-larger dense stalk maps. fileciteturn7file0L3-L3 fileciteturn9file0L3-L3 citeturn15academia1turn15academia2

That leads to a simple mathematical recommendation hierarchy:

- **first run**: keep the current full \(8\times 8\) restriction maps and keep \(s=8\); this is the most numerically conservative choice  
- **second sheaf-structure run**: if stalk scaling is desired, move to grouped low-rank at \(s=12\), \(k=4\), \(G=4\)  
- **avoid**: increasing both \(s\) and depth before profiling, because that multiplies the exact `64×64×s` work that already makes i018 slow on CPU and awkward on GPU fileciteturn17file0L3-L3 fileciteturn22file0L3-L3

## Architecture variants and static cost estimates

The table below uses the current repo formulas and scale definitions to give **static** parameter and rough forward-arithmetic growth estimates. These are derived from the current code structure and tensor shapes; they are **not wall-clock speed claims**. The experiment report is clear that code-reading-based speed predictions have already failed multiple times for this model family, and PyTorch’s profiler docs make the right remedy explicit: measure training and inference hotspots directly. fileciteturn7file0L3-L3 fileciteturn9file0L3-L3 fileciteturn10file0L3-L3 fileciteturn22file0L3-L3 fileciteturn17file0L3-L3 citeturn8view3

| Variant | Channels | Depth | Stalk | Head hidden | Restriction mode | Static params | Rough forward arithmetic vs current scale_xl | Read |
|---|---:|---:|---:|---:|---|---:|---:|---|
| Current scale_xl reference | 128 | 4 | 8 | 192 | full | 474,437 | 1.00x | current winner |
| Depth-only diagnostic | 128 | 6 | 8 | 192 | full | 614,767 | 1.43x | cheap param growth, but relation work grows linearly with depth |
| Width-only diagnostic | 160 | 4 | 8 | 320 | full | 785,217 | 1.26x | best first “efficient XXL” probe |
| Stalk-only diagnostic | 128 | 4 | 12 | 192 | full | 486,229 | 1.64x | bad efficiency; little param gain, big sheaf-cost gain |
| Grouped-relation research | 160 | 4 | 12 | 320 | grouped low-rank, \(G=4,k=4\) | 787,665 | 1.65x | only sensible if you explicitly want more stalk capacity |

The main lesson from that table is not the exact numbers; it is the **shape** of the tradeoff. Depth-only and stalk-only scaling disproportionately increase the part of the model that repeatedly touches the typed relation graph. Width-only plus a larger head instead puts more of the scale budget into regular dense operations—`Linear`, encoder fusion, node MLPs, and readout—which is exactly where modern kernels, compiler passes, and BLAS libraries are strongest. That makes width-plus-head the cleanest “efficient XXL” candidate. fileciteturn7file0L3-L3 fileciteturn9file0L3-L3 fileciteturn10file0L3-L3

The memory geometry tells the same story. The fixed relation tensor has shape `(B, 12, 64, 64)`, so at batch size **128** it is about **12 MiB in bf16** and **24 MiB in fp32**. The current eager sheaf block processes one relation at a time, so a single `(B, 64, 64, s)` residual at the current `s=8` is about **8 MiB in bf16** at batch 128; if someone raises `s` to 12, that becomes **12 MiB**; and an i249-style chunked vectorization over 3 relations would move that to roughly **24 MiB** at `s=8` and **36 MiB** at `s=12`. Width scaling does **not** change those `12×64×64` relation tensors at all. Stalk scaling does. That is another reason the first XXL run should leave `stalk_dim` alone. fileciteturn7file0L3-L3 fileciteturn9file0L3-L3 fileciteturn21file0L3-L3

The current code also exposes where later execution work is likely to live if profiling confirms a speed problem. `TacticalIncidenceBuilder` computes visible rays with an `einsum` against the precomputed `between` mask, computes pin candidates through a `matmul` against `pin_clear` followed by `scatter_add_`, and assembles the 12 typed relation planes with a long chain of elementwise compositions. That is structurally very different from the current head, encoder, and node MLP, which are dense and regular. Efficient scale should therefore mean **put the first extra parameters into the dense path**, not into a larger or deeper irregular sheaf path. fileciteturn7file0L3-L3 fileciteturn9file0L3-L3

## Profiling and equivalence protocol

The profiler-first rule should be implemented as a **two-pass protocol**. In the first pass, run a short training capture on the **actual train step**—forward, backward, and optimizer step—using `torch.profiler.profile` with CPU and CUDA activities, `record_shapes=True`, `profile_memory=True`, and `with_stack=True`. PyTorch’s profiler docs explicitly support collecting operator cost, input shapes, stack traces, and memory activity, and they also warn that shape and stack tracing add overhead, so this pass should be short and diagnostic rather than used for throughput claims. At minimum, instrument: adapter, incidence builder, `visible_rays`, `pin_relation`, encoder, every sheaf block, triad pool, head, optimizer step, and data transfer. Sort by `self_cuda_time_total`, `cuda_time_total`, `cpu_time_total`, and memory. Export a Chrome trace. citeturn8view3turn7view1turn7view2

In the second pass, repeat the same A/B comparison **without** diagnostic overhead and with the exact training settings that matter: current i018 scale_xl against the proposed candidate, same batch size cap, same mixed-precision path, same optimizer, same scheduler, same dataloader workers, same GPU. The experiment report’s most painful lesson is that forward-only timing and source-level intuition were not predictive for i249, and PyTorch’s profiler docs are explicit that the tool is meant for both training and inference rather than just microbenchmarks. The decision rule should be simple: if the incidence builder is not a top hotspot, do not touch it; if compile-only does not help the unchanged model, do not bundle it with architectural changes; and if a later fused incidence branch is explored, benchmark it separately from capacity changes. fileciteturn17file0L3-L3 citeturn8view3turn8view2

A concrete short diagnostic harness is:

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        with torch.autocast("cuda", dtype=resolved_amp_dtype):
            loss = training_step(batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if step == 19:
            break
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
prof.export_chrome_trace("i018_family_train_trace.json")
```

That pattern follows the repo’s own postmortem advice and PyTorch’s supported profiler workflow. fileciteturn17file0L3-L3 citeturn8view3

For **equivalence testing**, the repo should use a **parity ladder**, not a single forward check. The experiment report is explicit that i249 “matched” in eval-mode fp32 on CPU and still drifted badly under real training. PyTorch’s numerical-accuracy note explains why: floating-point addition and multiplication are not associative, batched computations are not guaranteed to be bitwise identical to equivalent sliced computations, and CPU and GPU results may differ even with the same inputs. Mixed precision adds another layer of sensitivity, and the AMP docs note both underflow issues and the fact that out-of-place versus in-place eligibility matters inside autocast regions. fileciteturn17file0L3-L3 citeturn8view0turn8view1turn7view5

The parity ladder should be:

1. **Structural smoke test**: same weights, eval mode, fp32, no compile.  
2. **Execution parity**: same weights, `model.train()`, dropout temporarily set to zero, actual autocast dtype enabled, no compile.  
3. **Real-path parity**: same weights, `model.train()`, configured dropout enabled, RNG reseeded before each paired step, actual autocast dtype enabled.  
4. **Optimizer parity**: one step and then 100 mini-batches with the same seed, comparing loss traces, logits, selected diagnostics, gradient cosine similarity, and parameter deltas.  
5. **Compile-only parity**: exact same ladder again with compile enabled on the unmodified baseline model before any custom fused work.  

The rule is strict: **execution-only** changes—compile, fusion, indexing rewrites, custom kernels—must pass this train-mode mixed-precision ladder before they are benchmarked for speed. If they fail, they are not “optimizations”; they are architecture changes. That is the central i249 lesson. fileciteturn17file0L3-L3 citeturn8view2turn8view0turn8view1

## Training plan and falsifiers

The training plan should deliberately separate **capacity scaling** from **execution scaling**. The first branch is architectural: identify whether i018 still improves when more capacity is placed in the regular dense path while the relation graph stays fixed. The second branch is systems work: only after profiling identifies the real hotspot should compile-only or fused-incidence work begin. That separation is necessary because the repo already has a negative case where both were changed together and the postmortem became ambiguous. fileciteturn17file0L3-L3

For the **capacity branch**, use the current paper-grade i018 training recipe as the control surface: same data splits, same BCE-with-logits task, same optimizer family, same `learning_rate: 0.0007`, same `weight_decay: 0.0001`, same `reduce_on_plateau` schedule, same mixed-precision flag, and the same seed discipline. The first scout should be **one seed, 12 epochs, minimum 6 epochs**, with the candidate architecture only; if it is stable and the validation curve is at least consistent with current scale_xl by the midpoint, promote it to the full **3-seed, 20-epoch paper-grade run**. This keeps the comparison aligned with how the repo already evaluated i018, its falsifier, and the trunk baselines. fileciteturn19file0L3-L3 fileciteturn13file0L3-L3 fileciteturn17file0L3-L3

For the **execution branch**, the order should be:

- baseline i018 scale_xl profile  
- recommended i254 candidate profile  
- compile-only A/B on baseline i018  
- compile-only A/B on the i254 candidate  
- only then, if the incidence builder is genuinely dominant, a fused-incidence prototype behind a feature flag  

That order matters because PyTorch’s `reduce-overhead` compile mode is intended to reduce Python overhead with CUDA graphs, may use more memory, and is not guaranteed to apply cleanly. The experiment report already notes that those conditions were not measured properly in i249. citeturn8view2 fileciteturn17file0L3-L3

The falsifiers for i254 should be explicit:

- **Scale falsifier**: if the 3-seed i254 capacity run does not beat i018 scale_xl’s **0.8901** by at least a small but meaningful margin—my recommendation is **+0.003 PR-AUC**—then do **not** assume the family still has a profitable scale runway. fileciteturn17file0L3-L3  
- **Efficiency falsifier**: if the first XXL candidate reduces measured train throughput by more than **25%** against current scale_xl while failing the scale falsifier, stop; that is not an efficient XXL. This is a policy recommendation grounded in the repo’s repeated speed-prediction failures. fileciteturn17file0L3-L3  
- **Stalk falsifier**: if any `s>8` variant fails to beat the best `s=8` width/head-scaled variant, then the repo should consider stalk scaling unsupported for this family. The current evidence base does not justify spending budget there first, and the static tensor growth is unfavorable. fileciteturn22file0L3-L3 citeturn15academia1  
- **Grouped-map falsifier**: if grouped low-rank restrictions underperform same-budget full maps by more than seed noise, drop them from the mainline and keep them only as a structured-stalk experiment.  
- **Systems falsifier**: if profiling shows the incidence builder is **not** a top hotspot, do not write a fused incidence kernel. If compile-only gives negligible benefit, do not attribute later speed changes to compile. fileciteturn17file0L3-L3 citeturn8view3turn8view2  
- **Parity falsifier**: if any execution-only rewrite fails the train-mode mixed-precision parity ladder, it is not allowed into the benchmark branch. fileciteturn17file0L3-L3 citeturn8view0turn8view1

## Implementation sketch and first-run recommendation

The implementation sketch should be intentionally conservative. The fastest way to learn something useful is **not** to write a new builder or a custom kernel first. It is to create a new i254 model that imports the current i018 thesis-carrying modules unchanged, adds one optional structured restriction parameterization behind a config flag, and otherwise scales only width and readout. Concretely:

- keep `BoardStateAdapter` unchanged  
- keep `TacticalIncidenceBuilder` unchanged  
- keep `SquareTokenEncoder` logic unchanged, but widen its target `channels`  
- keep `TriadDefectPool` unchanged  
- keep the current eager block semantics for the first run  
- add an optional `restriction_mode: full | grouped_lowrank` inside a new sheaf block class  
- add `record_function` scopes so profiling knows where time goes  
- keep `compile_model: false` and `fuse_incidence: false` by default for the first benchmark candidate fileciteturn18file0L3-L3 fileciteturn7file0L3-L3 fileciteturn9file0L3-L3 fileciteturn10file0L3-L3

A minimal file layout would be:

```text
ideas/registry/i254_efficient_i018_scale_xxl/
  architecture.md
  math_thesis.md
  config.yaml
  model.py

src/chess_nn_playground/models/trunk/
  oriented_tactical_sheaf_efficient_xxl.py
```

and the new trunk would expose:

```python
class GroupedLowRankRestriction(nn.Module):
    def __init__(self, stalk_dim, rank, group_count, relation_to_group):
        ...
    def src(self, relation_idx, z):
        ...
    def dst(self, relation_idx, z):
        ...

class EfficientSheafDiffusionBlock(nn.Module):
    def __init__(..., restriction_mode="full", restriction_rank=4, relation_groups=None):
        ...
```

The key design rule is that `restriction_mode="full"` should reproduce the current i018 block structure as closely as possible, so the new file can host both the conservative first run and the later grouped-low-rank research variant without forking the whole family. fileciteturn7file0L3-L3 fileciteturn9file0L3-L3

The **first scale-XXL run I recommend** is this:

```yaml
model:
  name: oriented_tactical_sheaf_efficient_xxl
  input_channels: 18
  num_classes: 1
  channels: 160
  hidden_dim: 320
  depth: 4
  stalk_dim: 8
  dropout: 0.1
  use_triads: true
  restriction_mode: full
  compile_model: false
  fuse_incidence: false

training:
  # identical to current i018 paper-grade recipe on the first pass
  learning_rate: 0.0007
  weight_decay: 0.0001
  lr_scheduler:
    name: reduce_on_plateau
    factor: 0.5
    patience: 2
    min_lr: 1.0e-05
  mixed_precision: true
  allow_tf32: true
  gradient_clip_norm: 1.0
  batch_size: 128
```

This candidate is the right first run because it does three things simultaneously: it preserves the chess relation thesis that the falsifier validated, it keeps the numerically risky execution path unchanged, and it directs most of the extra capacity into **regular dense width and a much larger head** instead of into more `64×64×s` relation work. In static terms it lands at about **785k parameters**, roughly **1.66×** current scale_xl, while only increasing rough forward arithmetic by about **1.26×** because it does **not** add sheaf depth or larger stalks. That is exactly the balance an “efficient XXL” should try first. fileciteturn17file0L3-L3 fileciteturn22file0L3-L3

If that first run is positive, the next move should **not** be a big systems rewrite. It should be a narrow follow-up: either `depth=5` at the same `channels=160, stalk_dim=8` setting if the profile shows relation work is still acceptable, or a grouped-low-rank `s=12` research variant if the goal is to test whether extra stalk capacity matters. The compile-only and fused-incidence branches should stay profiler-gated and parity-gated. That sequencing is the cleanest way to learn whether i018 still has headroom **and** avoid repeating i249’s core mistake. fileciteturn17file0L3-L3 citeturn8view2turn8view3