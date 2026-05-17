# p048_efficient_ray_occlusion_scan_primitive

## Thesis

The best version of p048 is a **compact, tensorized ray occlusion scan** that operates over legal rook/bishop/queen ray cells, not over dense source-target-between cubes and not through Python loops in `forward`. The repository already has the ingredients that point in this direction: `ray_geometry.py` precomputes fixed queen-direction tables with shape `(8, 64, 7)` and batched gather helpers; p020 `BlockerResetRayScan` advances a blocker-sensitive recurrence with a Python step loop; p021 `OcclusionSemiringRayScan` already uses scan-style transmittance with cumulative sums but reduces the ray to a weighted summary; p007 `AttackRaySparseAttention` only keeps first blockers plus a self-edge; p026 `RayCastObstaclePoolHead` again loops over directions and steps; and i018’s `TacticalIncidenceBuilder` still computes visible rook and bishop rays by contracting a dense `between` tensor with occupancy. That combination is exactly why a new primitive should be **native ray scan semantics with compact geometry and blocker identity**, not another attention or pooling head layered on top of the same bottlenecks. fileciteturn9file0L3-L3 fileciteturn10file0L3-L3 fileciteturn11file0L3-L3 fileciteturn21file0L3-L3 fileciteturn22file0L3-L3 fileciteturn26file0L3-L3

This design is also the right match for chess semantics. Sliding-piece attacks depend on **occupancy along a direction**; they are not local convolutions and they are not determined by ray density alone. X-rays are about a piece controlling through an intervening piece, and discovered attacks are about one piece moving away to reveal another attack. Those phenomena require the primitive to know the **first blocker**, the **second blocker**, their **side**, and the **value/identity of the target**, not just whether a line is “open enough.” citeturn7view5turn9view0turn9view1turn9view2

The implementation thesis is therefore simple: **gather all ray cells once, compute prefix blocker counts once, derive visibility and blocker identities by equality tests on those counts, then optionally scatter back to dense `(B,64,64)` maps for i018**. Scan is a standard parallel primitive on GPUs, and PyTorch already exposes the batched operators that make this layout realistic: `torch.cumsum`, `torch.gather`, `torch.compile`, `torch.utils.benchmark.Timer`, and scatter-style indexed writes. citeturn10academia0turn7view1turn7view2turn20view0turn7view4turn27view0

## Ray scan math

Let `c_{s,d,l}` be the `l`-th square on the ray from source square `s` in direction `d`, where `d` ranges over the eight queen directions and `l ∈ {1,…,7}`. The repo’s existing geometry already gives exactly this object as `step_index[d,s,l]` plus a validity mask. fileciteturn9file0L3-L3

For batch item `b`, define gathered occupancy and gathered target features

\[
o_{b,s,d,l} = \mathrm{Occ}_b(c_{s,d,l}), \qquad
z_{b,s,d,l} = \mathrm{Feat}_b(c_{s,d,l}).
\]

Define the **inclusive blocker count**

\[
k_{b,s,d,l} = \sum_{q \le l} o_{b,s,d,q}.
\]

Everything needed for line-of-sight, blockers, x-rays, and discovered-attack candidates follows from `k`:

\[
\mathrm{visible}_{b,s,d,l} = \mathbf{1}[k_{b,s,d,l} - o_{b,s,d,l}=0]\; m_{s,d,l},
\]

\[
\mathrm{first}_{b,s,d,l} = o_{b,s,d,l}\; \mathbf{1}[k_{b,s,d,l}=1]\; m_{s,d,l},
\]

\[
\mathrm{second}_{b,s,d,l} = o_{b,s,d,l}\; \mathbf{1}[k_{b,s,d,l}=2]\; m_{s,d,l},
\]

\[
\mathrm{xraylane}_{b,s,d,l} = \mathbf{1}[k_{b,s,d,l} - o_{b,s,d,l}=1]\; m_{s,d,l}.
\]

`visible` means there is no blocker before the current square. `first` and `second` are one-hot selectors for the first two occupied squares on the ray. `xraylane` marks squares that lie behind exactly one blocker and before a second blocker. Because `torch.cumsum` computes cumulative sums along one dimension, the full blocker structure is available without any Python loop over steps. citeturn7view1

The first-blocker and second-blocker summaries are then simple masked reductions:

\[
f^{(1)}_{b,s,d} = \sum_l \mathrm{first}_{b,s,d,l}\, z_{b,s,d,l}, \qquad
f^{(2)}_{b,s,d} = \sum_l \mathrm{second}_{b,s,d,l}\, z_{b,s,d,l}.
\]

If `z` includes mover-oriented piece one-hots, side flags, and a scalar value channel, then `f^{(1)}` and `f^{(2)}` preserve the exact blocker identity and target value. That is the key mathematical upgrade over p021: p021 already shows that scan-like transmittance is viable, but p048 should stop collapsing the ray too early and should preserve the first two blocker identities explicitly. fileciteturn11file0L3-L3

To connect the scan to rook/bishop/queen semantics, keep source-piece compatibility separate from geometry. Let `α_rook(s,d)` be `1` when the source square contains a rook or queen and `d` is orthogonal; let `α_bishop(s,d)` be `1` when the source contains a bishop or queen and `d` is diagonal. Then visible edge maps are

\[
E^{\mathrm{rook}}_{\mathrm{vis}}(b,s,t)
=
\sum_{d,l:\, c_{s,d,l}=t}
\alpha_{\mathrm{rook}}(s,d)\,\mathrm{visible}_{b,s,d,l},
\]

\[
E^{\mathrm{bishop}}_{\mathrm{vis}}(b,s,t)
=
\sum_{d,l:\, c_{s,d,l}=t}
\alpha_{\mathrm{bishop}}(s,d)\,\mathrm{visible}_{b,s,d,l}.
\]

The same compact scan also gives x-ray and discovered-attack features. A useful first discovered-attack candidate is

\[
D_{b,s,d}
=
\alpha(s,d)\,
\mathrm{OwnFirst}_{b,s,d}\,
\mathrm{EnemySecond}_{b,s,d}\,
\mathrm{ValueSecond}_{b,s,d},
\]

and a pin-style candidate is the dual pattern in which the first blocker is enemy-colored and the second blocker is that side’s king. Those are cheap tensor features, not a full rules engine inside the primitive. citeturn9view0turn9view1turn9view2

**Inputs.** For standalone use, p048 should accept either a `simple_18` board tensor `(B,18,8,8)` or an already-canonical pair `(piece_state, occupancy)`. For i018 integration, the preferred interface is the mover-oriented `piece_state` and `occupancy` that `BoardStateAdapter` already builds, so p048 does not duplicate board canonicalization. fileciteturn12file0L3-L3 fileciteturn26file0L3-L3

**Outputs.** The mandatory compact outputs should be `visible_steps`, `first_blocker_feat`, `second_blocker_feat`, `first_blocker_idx`, `second_blocker_idx`, `first_blocker_value`, `second_blocker_value`, `mobility_len`, `xray_pressure`, `discovered_attack_candidate`, and `pin_candidate`. An optional dense-output mode should scatter those compact ray results into `(B,64,64)` maps such as `rook_visible`, `bishop_visible`, `queen_visible`, `rook_xray`, `bishop_xray`, `queen_xray`, and discovered/pin target maps so the primitive can drop into i018-style graph builders directly. fileciteturn26file0L3-L3

## Tensor implementation plan

The implementation should reuse the repo’s current geometry module rather than inventing a new board-walk path. `ray_geometry.py` already gives the fixed `(8,64,7)` step indices, the step mask, and gather helpers. p048 should extend that module with direction-class masks and flattened target-slot indices, but the basic geometry contract is already correct. fileciteturn9file0L3-L3

The forward path should be completely tensorized:

First, derive or receive mover-oriented `piece_state` and `occupancy`.

Second, gather occupancy and feature channels along all eight directions and all seven padded steps in one batched pass.

Third, compute `k = cumsum(occ_ray, dim=-1)` and derive `visible`, `first`, `second`, and `xraylane` by equality tests on `k` and `k - occ_ray`.

Fourth, reduce masked blocker features to get exact first-blocker and second-blocker identities and values.

Fifth, only if the caller asks for dense graph output, scatter the compact step scores back into `(B,64,64)` edge maps. PyTorch’s `gather` and scatter-style indexed writes are a natural fit for that exact pattern. citeturn7view2turn27view0

The tensor layout should be **direction-major**, not piece-major. Chessprogramming’s guidance for multiple sliding-piece attacks is that one should traverse directions in parallel rather than serializing over sliders, and that matches the best GPU-shaped layout here: `(B, D, S, L, F)` or a flattened `(B, S, D·L, F)`. citeturn7view6

There should be two implementation tiers. The first tier is the golden-reference version in pure PyTorch, with no Python loops in `forward`, and wrapped in `torch.compile` because the shapes are static and the same compiled region can be cached and reused. The second tier is optional: if the benchmark still shows the reference path is memory-bound or launch-bound, lower exactly the same semantics into Triton or a custom C++/CUDA op without changing the public API. That keeps correctness and speed development separate. citeturn20view0turn20view1

## Integration targets

The first integration target is **i018 graph building**. Today, `TacticalIncidenceBuilder` computes `bishop_ray_visible`, `rook_ray_visible`, `queen_ray_visible`, and `king_ray_pin_candidate` through a dense `between` tensor plus a separate pin routine. p048 can replace the visibility path directly and can also expose richer relations that i018 does not currently build as first-class tensors, especially source-to-second-blocker x-ray maps and more explicit discovered-attack candidates. fileciteturn12file0L3-L3 fileciteturn26file0L3-L3

The second integration target is the repo’s **hybrid primitive scaffold**. `oriented_sheaf_plus_primitive.py` already fuses a primitive logit with the i018 sheaf logit through `final_logit = sheaf_logit + sigmoid(gate) * primitive_logit`, and the current hybrid setup already compares primitives against the i018 baseline. That makes p048 easy to evaluate both as a geometry kernel and as a predictive primitive. fileciteturn18file0L3-L3 fileciteturn14file0L3-L3 fileciteturn15file0L3-L3 fileciteturn31file0L3-L3

The third integration target is **standalone diagnostics**. p048 should be able to tell the model developer, for any square and direction, which piece blocks first, which target sits second, whether a line is a pin frame, whether a discovered attack exists after one friendly blocker moves, and how much x-ray pressure is latent behind one blocker. That diagnostic contract is not served cleanly by current primitives: p007 only looks at first blockers and a self-edge, p021 keeps transmittance but not explicit blocker identity, and p026 again prioritizes decayed pooling over blocker semantics. fileciteturn21file0L3-L3 fileciteturn11file0L3-L3 fileciteturn22file0L3-L3

**Complexity estimate.** In compact form, p048 is linear in ray cells: `O(B * D * S * L * F)` for gathered features and `O(B * D * S * L)` for blocker masks, with `D=8`, `S=64`, and `L=7`. The padded working set is `8*64*7 = 3584` slots per board, while the actual number of valid on-board queen-ray cells is `1456` by board geometry. By contrast, the current i018 visibility path contracts a dense `(64,64,64)` between-square structure against occupancy, which is `262,144` source-target-between positions per board before features. In other words, the compact scan works over the legal ray representation itself rather than the full cubic source-target-between expansion. That is the central efficiency argument for p048. The exact speedup must still be benchmarked, but the asymptotic and data-layout story is much stronger than the current dense contraction and stronger than the looped p020/p026 paths. fileciteturn26file0L3-L3 fileciteturn9file0L3-L3 fileciteturn10file0L3-L3 fileciteturn22file0L3-L3

## Expected strengths and falsifiers

**Expected strengths.** p048 should excel where the repo’s current ray-aware tools are either too compressed or too procedural: long rook/bishop/queen lines, blocker-side distinctions, x-rays through one blocker, discovered attacks through a friendly blocker, and pin structures that require seeing both the first and second occupied squares on a line. It should also be more reusable than the current family because the same compact tensor serves dense relation building, hybrid primitive experiments, and post-hoc diagnostics. fileciteturn21file0L3-L3 fileciteturn11file0L3-L3 citeturn9view0turn9view1turn9view2

The primitive should be treated as falsified or narrowed if any of the following are true.

If the compact p048 forward path does **not** beat the current i018 dense visibility builder or the looped p020/p026 implementations in steady-state GPU timing at the batch sizes that matter for repo experiments, then the “efficient” part of the primitive is false even if the algebra is attractive. fileciteturn10file0L3-L3 fileciteturn22file0L3-L3 fileciteturn26file0L3-L3

If an ablation that removes first-blocker and second-blocker identity/value while keeping the same compact visibility path performs essentially the same, then the “blocker identity and target value matter” claim is not supported. The repo’s own i018 falsifier script uses “within `0.01` PR-AUC of baseline” as a rejection-style threshold for a geometry claim; that is a sensible threshold family to reuse here. fileciteturn30file0L3-L3

If the p048 hybrid is a wash under the repo’s own hybrid interpretation—where `|delta| < 0.002` is effectively a wash and about `+0.005` PR-AUC is the threshold for a real lift—then p048 may still be a useful diagnostic kernel, but it is not a strong predictive primitive. fileciteturn31file0L3-L3

If removing second-blocker and discovered-attack channels leaves tactical subsets involving pins, x-rays, skewers, and discoveries unchanged, then the richer part of the primitive is unnecessary ornamentation rather than useful signal. citeturn9view0turn9view1turn9view2

## PyTorch pseudocode

The pseudocode below follows the repo’s mover-oriented `piece_state` convention and uses only batched tensor operations. The crucial point is that there is **no Python loop over ray steps or directions in `forward`**. The dense `(B,64,64)` maps are optional and come only from a final indexed scatter. The operator choices below are directly supported by PyTorch. citeturn7view1turn7view2turn27view0

```python
import torch

def ray_occlusion_scan(
    piece_state: torch.Tensor,   # (B, 64, 13): empty, us{P,N,B,R,Q,K}, them{P,N,B,R,Q,K}
    occupancy: torch.Tensor,     # (B, 64)
    step_index: torch.Tensor,    # (8, 64, 7) long
    step_mask: torch.Tensor,     # (8, 64, 7) float/bool
    *,
    return_dense_edges: bool = False,
) -> dict[str, torch.Tensor]:
    B, S, F = piece_state.shape
    D, S2, L = step_index.shape
    assert S == 64 and S2 == 64 and D == 8 and L == 7 and F == 13

    dtype = piece_state.dtype
    device = piece_state.device

    # Orthogonal directions: N, E, S, W in the repo's 8-direction order.
    ortho = piece_state.new_tensor([1, 0, 1, 0, 1, 0, 1, 0]).view(1, D, 1, 1)
    diag = 1.0 - ortho

    # Gather ray cells in one batched pass.
    flat_idx = step_index.reshape(-1)
    ray_mask = step_mask.to(device=device, dtype=dtype).view(1, D, S, L)

    occ_ray = occupancy[:, flat_idx].reshape(B, D, S, L) * ray_mask          # (B,D,S,L)
    feat_ray = piece_state[:, flat_idx, :].reshape(B, D, S, L, F)            # (B,D,S,L,F)
    feat_ray = feat_ray * ray_mask.unsqueeze(-1)

    # Inclusive blocker count and exclusive blockers-before count.
    k = occ_ray.cumsum(dim=-1)
    k_prev = k - occ_ray

    visible = (k_prev == 0).to(dtype) * ray_mask
    first = occ_ray * (k == 1).to(dtype) * ray_mask
    second = occ_ray * (k == 2).to(dtype) * ray_mask
    xray_lane = (k_prev == 1).to(dtype) * ray_mask

    # First/second blocker feature summaries.
    first_feat = (first.unsqueeze(-1) * feat_ray).sum(dim=-2)                 # (B,D,S,13)
    second_feat = (second.unsqueeze(-1) * feat_ray).sum(dim=-2)               # (B,D,S,13)

    piece_value = piece_state.new_tensor(
        [0.0, 1.0, 3.0, 3.0, 5.0, 9.0, 200.0, 1.0, 3.0, 3.0, 5.0, 9.0, 200.0]
    )
    value_ray = (feat_ray * piece_value.view(1, 1, 1, 1, F)).sum(dim=-1)

    first_value = (first * value_ray).sum(dim=-1)
    second_value = (second * value_ray).sum(dim=-1)
    first_exists = first.sum(dim=-1).clamp(0.0, 1.0)
    second_exists = second.sum(dim=-1).clamp(0.0, 1.0)

    # Side summaries on selected blockers.
    first_us = first_feat[..., 1:7].sum(dim=-1)
    first_them = first_feat[..., 7:13].sum(dim=-1)
    second_us = second_feat[..., 1:7].sum(dim=-1)
    second_them = second_feat[..., 7:13].sum(dim=-1)

    # Source piece compatibility by line type.
    src_us_bishoplike = piece_state[..., 3] + piece_state[..., 5]
    src_us_rooklike = piece_state[..., 4] + piece_state[..., 5]
    src_them_bishoplike = piece_state[..., 9] + piece_state[..., 11]
    src_them_rooklike = piece_state[..., 10] + piece_state[..., 11]

    us_slider = src_us_rooklike.unsqueeze(1).unsqueeze(-1) * ortho.squeeze(-1)
    us_slider = us_slider + src_us_bishoplike.unsqueeze(1).unsqueeze(-1) * diag.squeeze(-1)
    them_slider = src_them_rooklike.unsqueeze(1).unsqueeze(-1) * ortho.squeeze(-1)
    them_slider = them_slider + src_them_bishoplike.unsqueeze(1).unsqueeze(-1) * diag.squeeze(-1)

    mobility_len = (visible * (1.0 - occ_ray)).sum(dim=-1)
    xray_pressure = second_exists * second_value

    us_discovered = us_slider * first_us * second_them * second_value
    them_discovered = them_slider * first_them * second_us * second_value

    us_pin = us_slider * first_them * second_feat[..., 12]   # enemy king second
    them_pin = them_slider * first_us * second_feat[..., 6]  # our king second

    out = {
        "visible_steps": visible,
        "xray_lane_steps": xray_lane,
        "first_blocker_feat": first_feat,
        "second_blocker_feat": second_feat,
        "first_blocker_value": first_value,
        "second_blocker_value": second_value,
        "mobility_len": mobility_len,
        "xray_pressure": xray_pressure,
        "us_discovered_candidate": us_discovered,
        "them_discovered_candidate": them_discovered,
        "us_pin_candidate": us_pin,
        "them_pin_candidate": them_pin,
    }

    if not return_dense_edges:
        return out

    # Flatten (direction, step) into one per-source target-slot axis.
    target_slots = step_index.permute(1, 0, 2).reshape(1, S, D * L).expand(B, -1, -1)

    rook_visible_slots = (visible * ortho).permute(0, 2, 1, 3).reshape(B, S, D * L)
    bishop_visible_slots = (visible * diag).permute(0, 2, 1, 3).reshape(B, S, D * L)
    rook_xray_slots = (second * ortho).permute(0, 2, 1, 3).reshape(B, S, D * L)
    bishop_xray_slots = (second * diag).permute(0, 2, 1, 3).reshape(B, S, D * L)

    z = occupancy.new_zeros(B, S, S)
    rook_visible = z.clone().scatter_add_(2, target_slots, rook_visible_slots)
    bishop_visible = z.clone().scatter_add_(2, target_slots, bishop_visible_slots)
    rook_xray = z.clone().scatter_add_(2, target_slots, rook_xray_slots)
    bishop_xray = z.clone().scatter_add_(2, target_slots, bishop_xray_slots)

    out.update({
        "rook_visible": rook_visible.clamp_(0.0, 1.0),
        "bishop_visible": bishop_visible.clamp_(0.0, 1.0),
        "queen_visible": (rook_visible + bishop_visible).clamp_(0.0, 1.0),
        "rook_xray": rook_xray.clamp_(0.0, 1.0),
        "bishop_xray": bishop_xray.clamp_(0.0, 1.0),
        "queen_xray": (rook_xray + bishop_xray).clamp_(0.0, 1.0),
    })
    return out
```

## Benchmark and experiment plan

The repo already has a primitive pipeline that expects primitive docs, prototypes, tests, and config validation, but the current i018 hybrid and falsifier launchers explicitly pass `--no-benchmarks`. That means p048 needs an **explicit benchmark script** rather than relying on the existing i018 wrappers. In practical repo terms, the primitive should ship with a promoted idea folder for p048, a dedicated unit-test file, and a benchmark script such as `scripts/benchmarks/benchmark_ray_occlusion_scan.py`, and it should also be discoverable by the primitive pipeline. fileciteturn30file0L3-L3 fileciteturn31file0L3-L3 fileciteturn33file0L3-L3

The evaluation should have three layers.

The first layer is **correctness**. Compare p048 against an exact CPU ray walk on synthetic positions that isolate empty rays, immediate blockers, second blockers, x-rays, discovered attacks, and pin frames. Also compare `rook_visible` and `bishop_visible` directly against the current i018 visibility builder so that the “builder replacement” mode is guaranteed semantically correct before richer channels are added. fileciteturn26file0L3-L3

The second layer is **microbenchmarking**. Use `torch.utils.benchmark.Timer.blocked_autorange()` or `adaptive_autorange()` because PyTorch already handles warmups, synchronization of asynchronous accelerator work when needed, and replicate-friendly timing. Benchmark at batch sizes such as `{1, 16, 64, 256, 1024}`, in both eager and `torch.compile` modes, and separate **compact-output** timing from **dense-edge** timing. The comparison set should include the current i018 visibility path, p020, p021, p026, and p048. Report median latency, p95 latency, positions per second, and peak memory. citeturn7view4turn20view0

The third layer is **model experimentation**. There should be three ablations: a builder-replacement-only run where p048 only swaps the i018 visibility kernel; a relation-expansion run where x-ray and discovered-attack channels are turned on; and a hybrid-primitive run through the repo’s existing sheaf-plus-primitive scaffold. For consistency with the repo’s current i018 workflow, use the same three-seed, base-scale setup that the hybrid and falsifier scripts already use. fileciteturn31file0L3-L3 fileciteturn30file0L3-L3

A merge-worthy p048 result should satisfy two conditions simultaneously. It should produce a real steady-state speed win over the current dense or looped alternatives, and it should either improve i018 or hybrid tactical metrics or expose diagnostics that clearly separate line-of-sight tactical failures from generic occupancy failures. If it is only elegant but not faster, or only faster but tactically vacuous, it should remain a research artifact rather than a promoted primitive. fileciteturn31file0L3-L3

## Open questions

The main unresolved question is not the algebra; it is the **benchmarked speed crossover**. The compact scan is a much cleaner working set than the current dense builder, but the real win depends on how well `gather + cumsum + scatter` fuses under `torch.compile` on the target GPU. That must be measured, not assumed. citeturn20view0turn20view1turn7view4

A second open question is whether p048 is best used as a **graph-building primitive** or as a **hybrid predictive primitive**. The repo already supports both paths, and the hybrid thresholds in the existing launcher provide a concrete standard. If the hybrid lift is small but the builder replacement is significantly faster and much easier to diagnose, p048 could still be a successful primitive even without becoming a major standalone model head. fileciteturn18file0L3-L3 fileciteturn31file0L3-L3