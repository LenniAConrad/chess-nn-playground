# Witness-Counterwitness Quantifier Primitive

Author: Codex
Model: GPT-5 (Codex coding agent)
Date: 2026-05-12
Status: research packet

## One-Line Claim

`primitive_witness_counterwitness_quantifier` is a differentiable ragged-set operator for chess-like reasoning with native "there exists a forcing witness and all local counterwitnesses fail" semantics.

## Why This Primitive Is Worth Adding

The imported primitive batch is already dense with:

- delta/edit accumulators;
- ray and blocker scans;
- legal graph scatter/attention;
- chess orbit and symmetry operators;
- exchange/SEE semiring reducers;
- hyperedge contraction;
- threat diffusion / DEQ layers.

The missing operator is a clean quantifier primitive. Puzzle-vs-near-puzzle is not just "does a tactical-looking feature exist?" Near-puzzles often have the same positive-looking witness as true puzzles, but they also have one surviving defensive counterwitness. Existing attention, pooling, and sparse graph reductions are structurally bad at this because they usually pool evidence additively. They can notice the forcing move and the reply, but the operator boundary does not say:

```text
exists own candidate m such that no reply r refutes m.
```

The repo evidence points directly at this gap:

- `i193_exchange_then_king_dual_stream` is strongest overall and already separates exchange and king evidence.
- `i191_safe_reply_certificate_verifier` is strong on promotion near-puzzle rejection, but it is an architecture-level safe-reply witness, not a reusable primitive.
- `i011_vetoselect` wins a key operating point by separating positive claim from rejected evidence.
- Equal-eval and hard/very-hard rows remain the stable failure modes, where material/eval shortcuts are least useful and proof-vs-refutation structure matters most.

This primitive turns that structure into the layer itself.

## Signature

For each board in a batch, the primitive receives a ragged set of attacker witnesses and nested ragged sets of defender counterwitnesses:

```text
W_b = {w_i}_{i=1..K_b}
C_b(i) = {c_{ij}}_{j=1..R_{b,i}}

claim_i        = f_theta(w_i, board_context)
counter_ij     = g_phi(w_i, c_ij, board_context)
compat_ij      = h_psi(w_i, c_ij)
```

Native output:

```text
counter_envelope_i = forall_softmax_j(counter_ij + compat_ij)
margin_i           = claim_i - counter_envelope_i
value              = exists_softmax_i(margin_i)
```

The primitive also returns diagnostics:

```text
best_witness_index
best_counterwitness_index
witness_weights
counterwitness_weights
min_margin
max_margin
counter_envelope
quantifier_entropy
```

The important part is the nested soft quantifier, not the MLPs used to score tokens.

## Forward Semantics

Use two independent temperatures:

```text
counter_envelope_i =
  tau_forall * logsumexp_j((counter_ij + compat_ij) / tau_forall)

value =
  tau_exists * logsumexp_i((claim_i - counter_envelope_i) / tau_exists)
```

High `counter_ij` means "reply `j` refutes witness `i`." One strong counterwitness should suppress the witness. One surviving witness should raise the board-level value.

Anneal both temperatures during training:

```text
tau_forall: 0.5 -> 0.15
tau_exists: 0.5 -> 0.15
```

This starts smooth and gradually becomes closer to:

```text
max_i [ claim_i - max_j counter_ij ].
```

## Why This Is Not Just Attention

Attention performs weighted averaging over values. This primitive performs nested extremal reasoning over a ragged set-of-sets with a sign-changing opponent relation.

It is not masked attention:

- there is no `QK^T softmax V` map;
- the opponent side enters with the opposite sign;
- the gradient intentionally flows to the best witness and the strongest counterwitness;
- the output is a scalar certificate and witness/counterwitness pair, not a contextualized token set.

It is not graph attention:

- graph attention aggregates neighbors;
- this operator asks whether a witness survives all local counterwitnesses.

It is not the existing exchange/SEE reducer:

- SEE reduces an alternating capture chain on one square;
- WCQ works over arbitrary candidate/countercandidate sets: checks, blocks, promotions, mate threats, overload obligations, or learned tactical certificates.

It is not the Hall-defect ideas:

- Hall-defect computes coverage deficiency over defender-obligation systems;
- WCQ computes nested adversarial witness survival, where a single counterwitness can refute one candidate but may be irrelevant to another.

## Tiny Sanity Test

I ran a toy torch check before writing this packet.

Two examples had the same strongest positive claim:

```text
claim = [4.0, 1.0, 0.5]
```

In the true-like case, no reply was strong. In the near-puzzle-like case, candidate 0 had one strong counter-reply. A simple max-pool over claims cannot distinguish them. WCQ produced:

```text
true-like value:        3.464
near-puzzle-like value: 0.413
```

The gradient on the near-puzzle row concentrated on the strong counter-reply for candidate 0, which is exactly the desired counterwitness behavior.

## Minimal Torch Reference

```python
def wcq(claim, reply_escape, cand_mask, reply_mask, tau_forall=0.2, tau_exists=0.2):
    neg = torch.finfo(claim.dtype).min / 4
    masked_reply = torch.where(reply_mask, reply_escape, reply_escape.new_full((), neg))
    counter = tau_forall * torch.logsumexp(masked_reply / tau_forall, dim=-1)
    margin = claim - counter
    margin = torch.where(cand_mask, margin, margin.new_full((), neg))
    value = tau_exists * torch.logsumexp(margin / tau_exists, dim=-1)
    witness_weights = torch.softmax(margin / tau_exists, dim=-1)
    return value, margin, witness_weights, counter
```

The production primitive should fuse ragged packing, nested reductions, witness/counterwitness index recovery, and backward into one operator. The reference above is only for correctness.

## Chess Instantiation

Recommended first use:

```text
Witness set W:
  side-to-move candidate forcing moves

Counterwitness set C(i):
  opponent replies local to candidate i
```

Witness candidates:

- checks;
- captures;
- promotion pushes and promotion captures;
- moves attacking queen/king/rook;
- moves opening a slider line;
- moves increasing exchange soundness on a high-value square.

Counterwitness candidates:

- capture the moved piece;
- recapture on destination square;
- move king out of attacked zone;
- interpose on the opened line;
- defend the target;
- countercheck;
- stop promotion or create a faster promotion.

The primitive should receive token embeddings from the model, not raw FEN strings. The deterministic move/reply compiler can be outside the primitive, but the nested ragged reduction and backward belong inside it.

## Why It Targets The Current Weak Slices

Equal-eval positions are hard because material and static pressure are ambiguous. A position is puzzle-like only if one line survives counterplay. WCQ expresses that directly.

Promotion/underpromotion near-puzzles are hard because the board screams "promotion tactic," but a defensive resource often exists. A max/mean pool over promotion evidence will overfire. WCQ should veto when the opponent has one strong stop-promotion reply.

Mate-in-1 near-puzzles are similar: the candidate mate-looking move exists, but one king escape, capture, or interposition survives.

## Implementation Sketch

Inputs:

```text
claim_scores:        Float[B, K]
counter_scores:      Float[B, K, R]
candidate_mask:      Bool[B, K]
counter_mask:        Bool[B, K, R]
compatibility_bias:  Float[B, K, R] optional
```

Outputs:

```text
value:                 Float[B]
margin:                Float[B, K]
counter_envelope:      Float[B, K]
witness_weights:       Float[B, K]
counter_weights:       Float[B, K, R]
best_witness_index:    Long[B]
best_counter_index:    Long[B]
```

CUDA/Triton path:

1. Pack valid `(b, i, j)` triples into contiguous ragged segments.
2. Segment-logsumexp over replies per witness.
3. Segment-logsumexp over witnesses per board.
4. Save per-segment softmax weights for backward.
5. Scatter gradients only to valid witness/counterwitness tokens.

CPU/PyTorch fallback can use padded tensors and masks.

## Falsification Tests

Primitive-level tests:

1. Padded and ragged implementations match to `1e-5`.
2. `gradcheck` passes in double precision on small random masks.
3. Adding a strong counterwitness to the current best witness must lower `value`.
4. Adding a weak irrelevant counterwitness must barely change `value`.
5. If two witnesses share the same claim, the one with lower counter-envelope must receive higher witness weight.
6. If a witness has no counterwitnesses, counter-envelope must fall back to a learned/no-counter baseline, not `-inf`.

Architecture-level tests:

1. Replace the candidate/reply head in the Forcing Reply Envelope Veto model with WCQ.
2. Compare against:
   - ordinary attention over candidate+reply tokens;
   - max claim only;
   - mean counter penalty;
   - random reply assignment;
   - no counterwitness branch.
3. Keep parameter count and candidate/reply compiler fixed.

Success criteria:

- matched-recall near-puzzle FP at recall 0.80 improves by at least 5% over i193 and i011 on the same seed protocol;
- promotion/underpromotion near-FP beats the current i011/i193 operating-point leaders;
- equal-eval PR AUC improves over i193 or the model is rejected as too narrow;
- runtime stays within 1.25x of the padded PyTorch baseline at `K <= 48`, `R <= 12`.

## Novelty Risk

The closest neighbors are:

- differentiable minimax backups;
- logsumexp pooling;
- sparse attention over ragged sets;
- differentiable sorting/top-k;
- Gumbel-softmax hard routing;
- game-tree neural backups.

The surviving novelty claim is narrow:

> WCQ is a reusable ragged-set neural primitive whose operator boundary is a nested adversarial quantifier, with saved witness and counterwitness soft assignments in the backward pass.

Do not claim it is the first differentiable max or the first game-tree backup. Claim it as a native neural layer for `exists witness / forall counterwitness` reasoning over input-generated ragged candidate systems.

## Generalization Beyond Chess

This primitive should apply anywhere positives require one surviving witness and negatives often have counterexamples:

- theorem proving: proposed proof step vs countermodel;
- program analysis: vulnerability witness vs sanitizing guard;
- retrieval QA: answer evidence vs contradiction evidence;
- recommender safety: purchase intent vs disqualifying constraint;
- robotics: feasible action vs collision/constraint witness;
- molecular design: desired substructure vs toxicity/reactivity counterwitness.

## Recommendation

Implement WCQ only after the Forcing Reply Envelope Veto model has a padded PyTorch prototype. If the padded WCQ improves the target slices, then a fused ragged primitive is worth building. If the prototype cannot beat ordinary attention or simple max-minus-max pooling, do not spend kernel time on it.
