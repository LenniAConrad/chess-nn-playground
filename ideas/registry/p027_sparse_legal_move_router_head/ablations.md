# Ablations

`p027` exposes the shared primitive-head ablations plus four SLMR-
specific controls. Primary falsifier is `full_64x64_mask` — every
promotion run must include this matched control.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `full_64x64_mask` | Replace the legal-move mask with the all-ones mask. **Primary falsifier.** If the architecture matches A1 on the target slice, the topological constraint is not load-bearing and the primitive is dropped. |
| A2 | `self_loop_only` | Only allow each square to attend to itself. Tests whether the head is using the graph structure or just doing per-square feature mixing. |
| A3 | `shuffle_adjacency` | Random permutation of mask rows and columns. Decouples the legal-move pattern from the actual board squares. |
| A4 | `zero_router_features` | Replace the pooled output with zeros. Tests the trunk-diagnostics contribution. |
| A5 | `zero_delta` | Force `primitive_delta = 0`. Recovers i193 baseline. |
| A6 | `disable_gate` | Hold the gate at 1.0. Tests whether the gate is load-bearing. |
| A7 | `trunk_only` | Zero gate and delta together. Strongest control. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p027 >= i193 - 0.005, **and**
- the fork / knight-tactic / piece-routing slice PR AUC of unablated
  p027 >= i193 + 0.02, **and**
- A1 (`full_64x64_mask`) loses >= 70% of the target slice lift, **and**
- A3 (`shuffle_adjacency`) loses >= 50% of the target slice lift.

Drop if any condition fails. Drop especially if A1 matches the
unablated run — that means the head ignored the topology and just
behaved as a generic attention layer.

## Out-of-scope ablations (future)

- Per-edge piece-type embedding (use piece-type at source as the edge
  feature).
- Multi-round routing (current implementation is single-round).
