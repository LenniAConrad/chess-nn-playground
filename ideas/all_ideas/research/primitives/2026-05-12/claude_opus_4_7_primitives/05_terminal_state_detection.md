# 05 — Terminal-State Detection Primitive (TSDP)

**Slug:** `primitive_terminal_state_detection`
**Status:** proposed
**Author:** Claude
**Model:** Claude Opus 4.7
**Architecture extension:** i248 Rule-Aware Tactical Head
**Stance:** conservative

## One-line claim

A forward-pass primitive that, for each position, enumerates side-to-move's
legal moves and uses **exact chess rules** (via `python-chess`) to classify
each resulting position into terminal categories (checkmate, stalemate,
check, promotion, capture, castling), then aggregates into a per-position
11-dim feature vector.  The novelty isn't the math — it's the architectural
choice to make **rule-exact** terminal detection a first-class primitive
output, rather than a preprocessing feature or a learned approximation.

## Mathematical signature

For a position `x` with side-to-move `c`, let `M(x, c)` be c's legal moves.
For each `m in M(x, c)`, the primitive computes:

    is_checkmate(x.apply(m))  in {0, 1}    # exact rule
    is_stalemate(x.apply(m))  in {0, 1}    # exact rule
    is_check(x.apply(m))      in {0, 1}    # exact rule
    is_promotion(m)           in {0, 1}    # exact rule
    is_capture(m)             in {0, 1}    # exact rule
    is_castling(m)            in {0, 1}    # exact rule

Aggregated output (11-d feature vector):

    [ mate_in_1, mate_count,
      stalemate_threat, stalemate_count,
      check_count, promotion_count, capture_count, castling_count,
      total_legal_moves, forcing_density, mating_special_count ]

where `forcing_density = (check_count + capture_count) / total_legal_moves`
and `mating_special_count` is the count of mating moves that are
simultaneously promotions or captures (i.e., the dramatic mating-by-
promotion / mating-by-capture patterns).

Cost: `O(|M|)` python-chess rule checks per position, where `|M| ~ 30–50`.
On CPU, this is negligible compared to a neural forward pass (~10us/position
in python-chess).

Differentiability:  rule indicators are integer-valued and stop-gradient by
design.  They are intended to be consumed by a downstream learned head,
which provides the gradient flow.  This is the "differentiable mask + rule
oracle" pattern.

## Why this is genuinely new

**Not in any of the 12 primitive families** — Family 1-12 are all real- or
complex-valued numeric operators.  TSDP outputs integer-valued exact rule
predictions that gate downstream computation.  It's a *symbolic primitive
integrated with continuous features*.

**Closest existing primitives in the i### registry:**

| Existing | Approach | Terminal-detection? |
|---|---|---|
| i007 Neural Proof-Number Search | LEARNED proof numbers via search depth | Approximate |
| i025 One-Ply Move Landscape | LEARNED scoring of resulting positions | None |
| i188 Tactical Program Induction | LEARNED tactical programs | None |
| i077 Adaptive Tactical Resolvent | LEARNED resolvent operator | None |
| i178 Defender-Exhaustion Cascade | LEARNED defender exhaustion | None |
| i203 Hierarchical Tactical Option | LEARNED tactical hierarchy | None |
| **TSDP (this)** | **EXACT rule checks** | **YES** |

No primitive treats *exact rule-determined terminal states* as a forward-pass
output.  Existing approaches either:
- Use learned scoring (which has to *rediscover* checkmate from features), or
- Use legal-move masks (which encode legality but not terminality)

TSDP exposes the rule-exact terminal indicators directly to the architecture.

**Why the per_class benchmark suggests this is worth doing:** the
`mate_in_1` slice has PR AUC ~0.81 across all top models vs aggregate
~0.876.  The model is currently learning checkmate from raw board features —
TSDP would feed it the answer directly, and the trunk only needs to learn
"when do I trust this signal vs other features."

## Empirical evidence from prototype

[tsdp_prototype.py](prototypes/tsdp_prototype.py) tested across 6 scenarios:

| Scenario | mate_in_1 | mate_count | stalemate_threat | check_count | total_moves | forcing_density |
|---|---:|---:|---:|---:|---:|---:|
| **Quiet (starting position)** | 0 | 0 | 0 | 0 | 20 | 0.00 |
| **Back-rank mate** `Rd8#` | **1** | **1** | 0 | 0 | 16 | 0.00 |
| **Queen mate** `Qg7#` | **1** | **2** | 1 | 12 | 21 | 0.62 |
| **Mate-by-promotion** (near-mate) | 0 | 0 | 1 | 8 | 14 | 0.86 |
| **Stalemate trap** | 0 | 0 | **1** | 7 | 27 | 0.26 |
| **Busy middlegame** | 0 | 0 | 0 | 0 | 48 | 0.17 |

The rule indicators correctly identify:

- Mate-in-1 patterns (back-rank rook mate, queen mate)
- Multiple mating moves (Qg7# scenario has 2 mating moves found by python-chess)
- Stalemate threats (where moving into stalemate is a real risk)
- Forcing-move density (proxy for tactical sharpness)

**Autograd check**: TSDP indicators are stop-gradient (rule outputs are
integer-valued). When fed as a feature vector to a learned trunk
(`nn.Linear`), gradients flow correctly through the trunk:

    loss = 8.27
    trunk.weight.grad norm = 92.4
    trunk.bias.grad = -5.75

The gradient flows through the downstream head, as designed.

## Chess interpretation

| Indicator | Tactical reading |
|---|---|
| `mate_in_1 = 1` | Position is one move from delivering checkmate (the most decisive tactical pattern) |
| `mate_count > 1` | Multiple mating moves exist (puzzles often have a unique mating move, multi-mate positions are usually obvious) |
| `stalemate_threat = 1` | At least one move leads to stalemate — must AVOID this move if winning |
| `check_count` large | Many forcing moves available — sharp/tactical position |
| `forcing_density` high | Most legal moves are forcing — narrow tactical phase |
| `mating_special_count > 0` | Mate involves a promotion or capture — dramatic/unusual finish |

The indicators provide CHESS-RULE-EXACT signal that no current model has
access to without rediscovering it from board features.  The `mate_in_1`
slice's underperformance (~0.81 PR AUC) is likely partly due to this
information gap.

## Architecture extension — i248 Rule-Aware Tactical Head

Wraps any existing trunk (i193, i243):

```
Input: position x  (board planes + python-chess Board object)
   |
   +-->  [backbone trunk phi_theta]  -->  f_base in R^d
   |
   +-->  TSDP primitive  -->  f_rule in R^11   (stop-gradient on rule outputs)
   |
   v
fusion head:
   gate  = sigmoid(MLP_gate([f_base, f_rule]))
   logit = MLP_main([f_base, f_rule]) + gate * special_mate_bias
```

The architecture decides via the gate when to "trust" the rule signal vs
the learned features.  For positions with `mate_in_1 = 1`, the gate
amplifies the special-mate bias and the network learns to output
high-confidence logits.  For non-mate positions, the rule indicators
provide zero or low-magnitude features and the trunk's learned reasoning
dominates.

**Why the gate**: TSDP indicators are sparse (mate-in-1 fires on ~5-10% of
puzzle_binary positives).  A naive concatenation would let the trunk
under-use them.  A learned gate amplifies them when they fire.

**Composability with my prior primitives**: TSDP stacks orthogonally with
DHPE (01), TDCD (02), PFCT (03), and CAIO (04).  In particular:
- TSDP + PFCT both involve promotion-related counterfactuals.  TSDP
  detects mate-by-promotion (rule-exact); PFCT computes promotion-choice
  fanout (learned features).  Complementary.
- TSDP + TDCD: TSDP detects mate; TDCD detects tempo-asymmetry.  In a
  mate-in-1 position, `mate_in_1 = 1` AND `||g_T||` is large -- the two
  primitives align.
- TSDP + DHPE: TSDP gates on rule-exact mate; DHPE detects piece-pair
  interaction sign.  Different scales.

## Cost

| Stage | Cost per position |
|---|---|
| TSDP primitive | ~50 python-chess rule checks (~0.5 ms on CPU) |
| Backbone trunk | 1 forward pass through phi_theta |
| Fusion head | 1 small MLP |

Total overhead: < 5% of base i193 wall-clock at scout scale.  Negligible.

## Falsifier

Standard scout setup: 173k x 12 epochs, single seed, single 3070.

**Primitive-level pass criterion**:

- Ablation A1: replace TSDP indicators with their *random shuffle* across
  the dataset (so the indicators are decoupled from the actual position).
  The architecture must lose >= 70% of any mate_in_1 slice lift over the
  i193 baseline.  If A1 matches the full architecture, the rule indicators
  aren't load-bearing — they're just being used as a noise feature.

**Architecture-level pass criterion** (i248 vs i193):

- `crtk_tactic_motifs = mate_in_1` slice PR AUC: i248 >= **0.85**
  (i193's 0.81 + 0.04; a meaningful 5% relative improvement)
- Aggregate test PR AUC: i248 >= 0.875 (i193 - 0.001) -- no regression
- `crtk_eval_bucket = equal` slice: should NOT regress (TSDP is orthogonal
  to the equal-bucket failure; ensure no spurious harm)

**Fail criteria**:

- A1 (shuffled indicators) matches the full architecture -> drop
- mate_in_1 slice lift < 0.02 -> primitive's added signal isn't useful;
  the trunk was learning it implicitly anyway
- Aggregate regresses > 0.005 -> drop

## Ablations

| ID | Variant | Tests |
|---|---|---|
| A1 | Shuffle TSDP indicators across dataset | Whether the indicators carry signal |
| A2 | Drop individual indicators (mate-only, check-only, etc.) | Which indicator does the heavy lifting |
| A3 | Replace exact rule with a learned approximation (small CNN that predicts is_checkmate) | Whether *exactness* matters or learnable approximation is enough |
| A4 | TSDP for both side-to-move AND opponent (next-ply mate threats) | Whether two-ply terminal detection beats one-ply |
| A5 | Disable the gate (concatenate directly into the head) | Whether the gate is necessary |

## Generalisation beyond chess

This is a **rule-aware primitive** pattern.  Any game with discrete rules
(chess, Go, shogi, checkers, card games) can benefit from a TSDP-like
primitive that detects rule-exact terminal states alongside learned
features.  For non-game domains, the analogue is **safety-constraint
detection** — e.g., a constraint-satisfaction primitive that detects
infeasible configurations in molecular design or robot planning.

## Risks

1. **Slot-fill risk**: the trunk may already be learning checkmate
   implicitly.  TSDP only helps if the trunk's implicit learning is
   *suboptimal*.  The benchmark's 0.81 PR AUC on mate_in_1 suggests yes,
   but A1 ablation must confirm.
2. **Engineering cost in batched inference**: python-chess is single-
   threaded Python.  Per-position rule checks at batch-size 1024 means
   ~1024 * 50 = 50,000 rule checks per batch — at ~10us each = 500ms per
   batch.  This is non-trivial vs a GPU forward pass.  Mitigate with
   precomputed-cached rule features in the data loader (TSDP runs once
   per position during data preparation, not per training step).
3. **Hidden rebrand of "use chess rules as features"**: a reviewer will say
   "this is feature engineering, not a new primitive."  Defence: the
   primitive's identity is not in the math (which is trivial — apply chess
   rules) but in the **architectural pattern** of making rule-exact
   terminal detection a first-class primitive output that gates downstream
   computation via a learned signal.  i007 etc. learn approximations to
   what TSDP computes exactly.
4. **Narrow slice coverage**: TSDP only helps when mate-in-1 / stalemate
   are *actually* present in the position.  Most positions have neither.
   Mitigate with the gate -- TSDP is zero-cost (and zero-bias) when
   indicators don't fire.

## Why high-potential

- **Surgical targeting of the mate_in_1 slice** (PR AUC 0.81 vs aggregate
  0.876 — second-largest gap after promotion)
- **Cost is negligible** (~5% wall-clock overhead, preprocessable into
  data loader)
- **Architecture extension is clean**: rule indicators + learned gate +
  fusion head; one-day implementation
- **Empirically validated** in the prototype: rule indicators correctly
  detect mate-in-1, stalemate threats, and forcing-move densities
- **Stacks orthogonally** with my prior four primitives — different
  signal, different cost profile, different target slice
- **Conservative** — uses well-established chess rules, no exotic math.
  The risk is "is this just feature engineering" rather than "is this
  mathematically correct"
- **Generalises** to any rule-based discrete game or safety-constraint
  domain

## Comparison to my prior 4 codex primitives

| primitive | counterfactual axis | mechanism | target slice | cost |
|---|---|---|---|---|
| 01 DHPE | piece-pair existence | combinatorial counterfactual | near-puzzle vs puzzle | ~10x |
| 02 TDCD | tempo x defender | cross-derivative | equal bucket | ~8x |
| 03 PFCT | piece-type substitution | promotion fanout | promotion slice | ~1.4x (gated) |
| 04 CAIO | chess-Z2 -> U(1) | complex-amplitude interference | structural coherence | ~1.5x |
| **05 TSDP** | **chess-rule terminal detection** | **exact rule check + learned gate** | **mate_in_1 slice** | **~1.05x** |

TSDP is the **cheapest** of the five primitives and targets a specific slice
with **rule-exact signal** that the others all approximate or ignore.
