# Ablations

p053 supports eight ablation modes via `model.ablation`. The primary
falsifier is `no_pressure_delta` -- every LMGDP run must include
this matched control on the same split, seed, and training budget.
The remaining ablations each test a distinct structural claim.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `no_pressure_delta` | **Primary falsifier**. Zero out `pre_opp_attackers_at_target`, `pre_own_defenders_at_target`, `mover_post_attack_value_from_t`, and `mover_post_defender_value_from_t`. If A1 matches the unablated run, the pressure-delta story is false and the head collapses to a typed edge-count head. |
| A2 | `no_capture_value` | Zero out `enemy_value_at_target` and `gives_check_proxy`. If A2 matches `none`, the explicit captured-piece value / gives-check tagging is not load-bearing. |
| A3 | `random_typed_edges` | Replace the typed legal-move adjacency with a random mask of identical per-type density (mirrors p009 LMGConv's falsifier). If A3 matches `none`, the per-piece-type chess connectivity is not load-bearing. |
| A4 | `shared_target_pool` | Collapse the six per-piece-type per-target projection heads to a single shared linear. If A4 matches `none`, the per-type routing is not load-bearing and the primitive can be simplified to a single per-target projection. |
| A5 | `zero_delta` | Zero primitive delta. Recovers the i193 baseline. |
| A6 | `trunk_only` | Same as A5 (semantic alias). |
| A7 | `disable_gate` | Pin gate at 1.0. Tests gate load-bearing -- if A7 matches `none`, the gate is not actually filtering the delta into a no-op on edge-less positions. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p053 >= i193 - 0.005, AND
- the merged ``crtk_tactic_motifs in {capture, check, mate,
  promotion}`` slice lifts at least +0.01 PR AUC over i193, AND
- A1 (`no_pressure_delta`) loses >= 50% of that lift, AND
- (A2 or A3 or A4) loses >= 30% of that lift, AND
- the `crtk_eval_bucket = equal` slice does not regress more than
  the aggregate threshold, AND
- the near-puzzle FP-at-recall-0.80 on the target slice does not
  regress vs i193.

Drop if any condition fails.

## Out-of-scope ablations (future)

- **Special move classes** (en-passant, castling, promotion target
  selection). The current topology mirrors p009's pseudo-legal
  coverage. Adding these requires extending
  `_compute_typed_legal_edges` and the per-edge feature dictionary.
  Listed here so a future iteration knows the test slot exists.
- **Source-blocker correction** for slider post-move attack value.
  The current proxy uses the unoccluded geometric attack table so
  it does not subtract the source piece's own blocker contribution.
  The corrected variant would re-evaluate `compute_attack_relations`
  after stop-grad removal of the moving piece; cost is one extra
  einsum per edge. Listed as a future ablation.
- **Edge-square message-passing round**. The source markdown
  sketches a two-round edge-square message-passing stack. Adding
  even one round is a structural extension and is deferred until
  the static pool variant has proven a keep decision.

Run these only after the primary falsifier (`no_pressure_delta`)
passes.
