# Mathematical Thesis

## Actual Task And Labels

For the first benchmark, train a single-logit binary classifier:

```text
source class 0 random/non-puzzle -> y = 0
source class 1 verified near-puzzle -> y = 0
source class 2 verified puzzle -> y = 1
```

The architecture is intended for general chess position classification, not only puzzles.

Allowed inputs:

- current board tensor
- deterministic pseudo-legal move lists
- deterministic move/reply descriptors from rules

Forbidden inputs:

- Stockfish evaluations, PVs, node counts, mate scores, verification status, source labels, source file identity, or generated labels not present in the data

## Baseline And Weakness

Closest existing ideas:

- One-Ply Counterfactual Move Landscape Network
- Counterfactual Move-Delta Spectrum Network
- Legal-Reaction Bottleneck Network
- LC0 BT4 tower

Overlap: these ideas also use current-board move or reply structure without engine search.

Difference: this idea makes the central representation a symmetric learned minimax bottleneck over side-to-move actions and opponent responses, rather than a one-sided move landscape or reply-only rejection feature.

## Definitions

Let `A(x)` be a deterministic capped set of pseudo-legal actions for the side to move. Let `R(x, a)` be a deterministic capped set of pseudo-legal opponent replies after applying action `a`.

Encode:

```text
h = board_encoder(x)
u_a = action_encoder(a, h)
v_{a,r} = reply_encoder(r, u_a, h)
```

Learn response scores:

```text
p_a = action_promise(u_a)
q_{a,r} = reply_safety(v_{a,r})
```

Define a soft minimax descriptor:

```text
m_a = p_a - tau_r * logsumexp_r(q_{a,r} / tau_r)
M(x) = tau_a * logsumexp_a(m_a / tau_a)
```

The classifier uses:

```text
f(x) = classifier([h_pool, M(x), top_k(m_a), reply_entropy_stats])
```

## Assumptions

- For classification labels like puzzle/non-puzzle, good actions and opponent replies contain information that a static board encoder may not expose cleanly.
- The deterministic pseudo-legal move set is sufficient as a structural probe, even though it is not engine search.
- Capped move/reply sets do not systematically drop the key tactical resources.

## Claim

Hypothesis: a learned one-ply minimax bottleneck should improve near-puzzle rejection because many near-puzzles contain apparent threats that are neutralized by plausible replies, while true puzzles have a stronger max-over-actions and min-over-replies signature.

## Mechanism

The model is forced to answer a classification-relevant question:

```text
is there a promising current-side action whose plausible replies are weak?
```

This is more general than puzzle detection; any chess classification target with action-response structure can use the same bottleneck.

## Proof Sketch

What can be reasoned about:

- The descriptor is permutation-invariant over actions and replies.
- The bottleneck represents a differentiable max-min over rule-generated alternatives.
- The model cannot directly use engine strength; it only learns from labels.

## Not Proven

- That pseudo-legal one-ply structure is enough for the current labels.
- That the learned reply scorer aligns with real chess safety.
- That capped candidate selection keeps the decisive action and reply.

## Counterexamples

- Quiet strategic classification labels may not be improved by action-response structure.
- Multi-move tactics may require deeper response modeling.
- Positions with many equivalent legal replies may make the soft minimax noisy.

## Falsification Test

Compare against a board-only trunk with the same parameter budget. Revise or reject if:

```text
response-minimax does not improve PR AUC by >= 0.015
and does not reduce near-puzzle false positives by >= 0.02 absolute
```

Reject the minimax mechanism if replacing legal replies with random legal-looking descriptors performs the same.

