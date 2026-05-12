# Forcing Reply Envelope Veto Network

Author: Codex
Model: GPT-5 (Codex coding agent)
Date: 2026-05-12
Status: research packet

## One-Sentence Thesis

Near-puzzles often contain a tempting forcing move, but at least one safe reply survives; a puzzle detector should therefore score not only positive tactical evidence, but whether that evidence has an opponent-reply envelope that vetoes it.

## Why This Model Fits The Current Evidence

The current reports point to a narrow target, not a generic capacity problem:

- `i193_exchange_then_king_dual_stream` is the strongest recent parent: its conv-only dual exchange/king design is better than the larger i242 attention variant, and removing the exchange stream hurts badly.
- `i011_vetoselect` is not the aggregate PR-AUC champion, but it wins the promotion/underpromotion near-puzzle false-positive operating point at recall 0.80.
- `i013_sparse_relation_pursuit_asymmetry` has the best promotion-slice ranking PR AUC among the audited models.
- `i024_directed_attack_sheaf_tension_network` is strong on rule-geometry slices such as pins and low-difficulty tactical motifs.
- The hardest stable failure modes are equal-eval positions, hard/very-hard rows, mate-in-1 near-puzzles, and promotion/underpromotion where a position looks forcing but a defensive resource remains.

So the model should not be another broad transformer or bigger residual tower. The best bet is a small LC0/simple-current-board architecture that preserves i193's exchange/king split, then adds a reply-soundness certificate before accepting the positive puzzle claim.

## Architecture

Name: `forcing_reply_envelope_veto_network`

Recommended first idea id if promoted: next available `i###`.

Input:

- Primary target: `lc0_bt4_112`, using only current-position planes and auxiliary FEN planes.
- Implementation should also support `simple_18` by reusing existing current-board adapters, so it can compare directly against i193 and older idea code.
- No CRTK tags, source labels, engine scores, PVs, best moves, or verification metadata are model inputs.

Output:

- One puzzle logit for `puzzle_binary`.
- Diagnostics: `raw_claim_logit`, `reply_veto_logit`, `accepted_claim_logit`, `max_claim`, `min_reply_escape`, `selected_candidate_count`, `reply_escape_mass`, `promotion_claim`, `king_claim`, `exchange_claim`, `envelope_margin`, and `stream_gate`.

### 1. Current-Board Adapter

Decode the current board into side-relative piece planes:

```text
own P,N,B,R,Q,K
opp P,N,B,R,Q,K
side-relative castling/en-passant/rule50/all-ones if available
```

For `lc0_bt4_112`, use history slot 0 plus the auxiliary planes. For `simple_18`, use the existing side-to-move plane and piece planes.

### 2. I193-Style Exchange/King Parent

Use i193's proven split as the parent:

```text
exchange_stream = ExchangeConv(board + deterministic exchange planes)
king_stream     = KingConv(board + deterministic king-zone planes)
gate            = sigmoid(phase_router([exchange_pool, king_pool, summaries]))
parent_logit    = gate * king_logit + (1 - gate) * exchange_logit + residual_logit
```

Do not add full attention here. i242's result says the larger attention block is not the immediate path.

### 3. Forcing Candidate Compiler

Build a fixed maximum of `K` side-to-move candidate moves from the current board. Start with pseudo-legal moves; optionally add a cheap self-check filter later.

Keep only moves that can plausibly explain a puzzle:

- captures;
- promotions and near-promotion pushes;
- checks and king-zone attacks;
- moves that open or close a slider ray toward king/queen/high-value pieces;
- moves with high exchange-soundness from a small SEE-style field;
- moves touching a pinned, overloaded, or hanging piece.

Each candidate token contains:

```text
from_sq, to_sq, piece_type, capture_type, promotion_type,
is_check_like, is_promotion_like, is_capture,
ray_direction, path_clearance, target_value,
exchange_soundness, king_zone_overlap
```

The compiler is deterministic and board-only. It should be vectorized in torch where practical and may reuse the existing pseudo-legal move enumeration code from the counterfactual move-delta models.

### 4. Reply Envelope Compiler

For each forcing candidate `m`, construct a bounded set of opponent replies `R(m)`.

The key is not to enumerate every possible legal reply perfectly. The useful certificate is local: does the opponent have a plausible tactical escape around the candidate's affected squares?

Reply types:

- recapture on `to_sq`;
- king escape from the attacked king zone;
- interpose/block on a newly opened ray;
- capture or attack the moved piece;
- promote or stop promotion on the relevant promotion lane;
- countercheck/counter-threat against the side-to-move king.

Each reply token gets a learned safe-reply score:

```text
safe(m, r) = ReplyMLP([
  candidate_features(m),
  reply_features(r),
  post_move_delta_features(m),
  exchange_stream_features,
  king_stream_features,
  directed_attack/sheaf statistics around affected squares
])
```

Then define the reply envelope:

```text
reply_escape(m) = tau * logsumexp_r(safe(m, r) / tau)
```

This is intentionally a soft maximum: one good defensive resource should veto a tempting false positive.

### 5. Claim Minus Reply Envelope

Score each candidate's positive evidence:

```text
claim(m) = ClaimMLP([
  from_feature,
  to_feature,
  path_feature,
  exchange_soundness,
  king_zone_tension,
  promotion_lane_pressure,
  parent_context
])
```

Then compute:

```text
envelope_margin(m) = claim(m) - reply_escape(m)
```

Aggregate with sparse selection:

```text
alpha = sparsemax(envelope_margin)
certificate = sum_m alpha_m * envelope_margin(m)
claim_mass = logsumexp_m claim(m)
reply_mass = logsumexp_m reply_escape(m)
```

The intended semantics:

- true puzzle: at least one candidate has high claim and low reply escape;
- near-puzzle: claim is high, but reply escape is also high;
- random non-puzzle: claim is low.

### 6. VetoSelect Final Head

Use the successful i011 lesson directly:

```text
raw_claim_logit = ParentHead([parent_pool, claim_mass, certificate])
reply_veto_logit = VetoHead([parent_pool, reply_mass, min_reply_escape, envelope_margin_stats])
accepted_claim_logit = raw_claim_logit - softplus(reply_veto_logit)
final_logit = parent_logit + accepted_claim_logit
```

This keeps two separate ideas apart:

- "the board looks tactical";
- "the tactical claim survives replies."

Near-puzzle negatives should land in high raw claim / high veto. True puzzles should be high raw claim / low veto.

## Loss

Main loss:

```text
BCEWithLogits(final_logit, puzzle_binary_target)
```

Add only one auxiliary loss, after warmup:

```text
reply_veto_decoy_loss
```

Self-mine decoys inside each batch:

- non-puzzle rows with high `raw_claim_logit`;
- especially rows where `claim_mass` is high but `certificate` is low.

For those rows, push `reply_veto_logit` up. Do not use source class labels as a model input; fine/source labels are only training targets and diagnostics under the existing benchmark contract.

Avoid a large multi-objective stack. The audit suggests excessive auxiliary machinery can dilute the main puzzle signal.

## Why This Is Not Just Existing Work

It is not i193: i193 has exchange and king streams, but no explicit opponent-reply envelope.

It is not i011: i011 learns a veto head over trunk evidence, but it does not construct candidate moves and reply resources.

It is not i013: SRPA compares sparse tactical/background dictionaries, but it does not implement "one tempting candidate survives or fails reply search."

It is not i024: the sheaf network scores attack-graph tension, but it does not pair a positive forcing claim with the opponent's local defensive envelope.

It is not i242: there is no broad all-square attention block. The model uses sparse candidate/reply tokens with a chess-specific min-max bottleneck.

## Central Falsifier

The central claim is:

> Reply-envelope vetoing reduces near-puzzle false positives at matched recall without sacrificing the i193 aggregate signal.

Fail the model if, on the canonical tagged split:

- test PR AUC is below i193 by more than 0.003 and there is no compensating hard-negative gain;
- near-puzzle FP rate at recall 0.80 and 0.85 is not at least 5% lower than i193 and i011 on the same seed protocol;
- promotion/underpromotion near-puzzle FP does not beat i011 scale_xl's current operating-point result;
- equal-eval PR AUC does not improve over i193 or stays within noise while runtime increases materially.

## Required Ablations

1. `no_reply_envelope`: replace `reply_escape(m)` with zero. Should behave like an i193-plus-candidate-claim model.
2. `random_reply_edges`: keep reply-token count and shapes, but randomize reply squares. Tests whether reply geometry matters.
3. `claim_only`: use `logsumexp claim(m)` without sparse envelope margin.
4. `no_veto_head`: use the certificate directly as an additive logit; tests whether i011-style accepted/rejected evidence matters.
5. `king_only_candidates`: keep only checks/king-zone candidates.
6. `exchange_only_candidates`: keep only captures/SEE candidates.
7. `no_promotion_features`: remove promotion-specific descriptors; must hurt promotion/underpromotion if the mechanism is real.
8. `attention_over_candidates`: replace envelope bottleneck with ordinary candidate self-attention at matched params. Should not beat the sparse min-max bottleneck if the thesis is right.

## Expected First Training Protocol

Smoke:

```text
1 epoch, seed 42, small batch, validate artifact contract and diagnostics.
```

Triage:

```text
3 epochs, seed 42, canonical tagged split, monitor PR AUC.
```

Reliable:

```text
20 epoch convergence budget, seeds 42/43/44, monitor PR AUC,
compare against i193, i011 scale_xl, i013, i024, and bench_lc0_bt4_classifier.
```

Primary metrics:

- test PR AUC;
- matched-recall near-puzzle FP at recall 0.80 and 0.85;
- promotion/underpromotion near-puzzle FP;
- equal-eval PR AUC;
- hard and very-hard slice accuracy at matched recall;
- runtime and parameter count versus i193.

## Implementation Notes

Start with the smallest honest version:

- reuse i193's deterministic exchange/king feature builder;
- reuse or adapt the existing pseudo-legal move enumerator from the counterfactual move-delta code;
- cap candidates at `K=48`;
- cap reply tokens per candidate at `R=12`;
- use `channels=64`, `candidate_dim=96`, `reply_dim=64`;
- keep the final model under roughly 250k parameters for the first benchmark.

If full reply enumeration is too slow, first implement a local reply envelope around affected squares rather than all legal replies. The falsifier only requires that defensive resources around the tempting move are represented.

## Predicted Outcome

My expectation is not a huge aggregate jump. The realistic target is:

- aggregate PR AUC: within +0.000 to +0.006 of i193;
- near-puzzle FP at recall 0.80: 5-10% lower than i193;
- promotion/underpromotion near-FP: competitive with or better than i011 scale_xl;
- equal-eval PR AUC: modest improvement if reply envelopes really detect "looks forcing but safe defense exists."

If this fails, the result is still useful: it would say the current dataset does not reward explicit local reply certificates beyond what i193 already learns, and future effort should move to SSL/pretraining rather than more hand-built tactical bottlenecks.
