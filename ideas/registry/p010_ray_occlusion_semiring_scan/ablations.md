# Ablations — p010 Ray-Occlusion Semiring Scan

## Switches (model.ablation)

| Mode | What it tests |
|---|---|
| `none` | Full architecture (default). |
| `uniform_transmittance` | Set `T = ray_valid` (1 wherever on board), removing occlusion. **Primary semiring-scan falsifier**: tests whether the blocker-aware prefix product carries the lift. |
| `constant_direction` | Collapse 8 direction matrices to one shared linear. Tests whether direction-specific weights are load-bearing. |
| `no_step_decay` | Disable learned per-step decay (force `λ_δ^k = 1`). Tests whether the decay is load-bearing beyond the transmittance itself. |
| `zero_delta` | Hold `primitive_delta = 0`. i193 baseline. |
| `disable_gate` | Hold `primitive_gate = 1`. |
| `trunk_only` | Strict no-op. |

## Falsification criteria

Promote p010 only if `model.ablation = none`:

- Aggregate PR AUC delta from i193 >= -0.005.
- CRTK class-1 matched-recall FP rate drops by >=5% relative.
- Wall-clock per epoch within 1.2x of i193.

Drop p010 if:

- `uniform_transmittance` matches `none` (occlusion was not the source
  of lift — operator is just a depthwise ray-conv).
- `constant_direction` matches `none` (per-direction tying gave no
  lift).
- `zero_delta` matches `none` (delta head was noise).

## Deferred internal proposals from external_12

The source primitive packet
(`ideas/research/primitives/external_12_ray_occlusion_legal_dispatch_delta_pair.md`)
ranks five proposals. Only the top-ranked **Ray-Occlusion Semiring
Scan** is implemented here. The others are deferred:

- **Legal-Move Sparse Dispatch**: covered in this batch by p008
  (MobScan) and p009 (LMGConv); the file's "in-op edge generation +
  fused segment softmax" structure overlaps p008/p009.
- **Delta-Factorized Pair Accumulator**: stateful second-order
  sufficient-statistic accumulator. Deferred — stateful operator with
  paired (apply, unmake) semantics whose falsifier needs a search-
  trajectory benchmark this batch does not have.
- **Chess-Group Orbit Contraction**: G-CNN reframe. Deferred — the
  file's own framing classifies it as "underexplored for chess, not
  fully new equivariance".
- **Soft Exchange Semiring Pool**: differentiable static-exchange
  evaluator. Deferred — chess-specific reducer with limited
  generality; better tested after the broader sparse-routing primitives
  in this batch ship.

If any prove relevant after p010's scout, promote under fresh `p###`.
