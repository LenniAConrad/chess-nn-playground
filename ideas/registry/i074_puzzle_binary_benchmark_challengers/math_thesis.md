# Math Thesis

`Puzzle-Binary Benchmark Challengers` promotes Idea 1 of source packet
`ideas/research/packets/classic/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md`
into a bespoke architecture: the **Negative-Class Disentangled Puzzle Head**.

## Setting

The corrected `puzzle_binary` benchmark binarizes the fine source labels:

```text
fine 0 -> random non-puzzle      target 0
fine 1 -> near-puzzle hard neg   target 0
fine 2 -> verified puzzle        target 1
```

The current best baseline (`LC0 BT4`) achieves test F1 `0.7445` and PR-AUC
`0.8068`, but still calls roughly `24.8%` of near-puzzles puzzles. The
benchmark's failure mode is the *near-puzzle* false-positive rate, not
ordinary binary accuracy.

## Disentangled Evidence Decomposition

A single negative head averages two distinct sources of negativity. The
packet hypothesizes that the puzzle decision boundary is sharper if the
trunk is asked to expose three separate evidence channels and only
collapse them at inference. Let `h = trunk(board)` be a pooled board
descriptor and let `e_random, e_near, e_puzzle in R` be three scalar
evidence heads. Define

```text
puzzle_logit = e_puzzle - logsumexp([e_random, e_near])
```

This is the unique single-logit form that scores a board as a puzzle
*against* the soft-max of its two negative attractors. Equivalent
log-odds expression:

```text
sigmoid(puzzle_logit) = exp(e_puzzle) / (exp(e_puzzle) + exp(e_random) + exp(e_near))
```

i.e. inference is a 3-class softmax `[random, near, puzzle]` followed by
collapsing the two negatives.

## Aux 3-way Identifiability

The packet pairs the BCE-on-`puzzle_logit` objective with an auxiliary
3-way CE on the raw evidence stack `[e_random, e_near, e_puzzle]` keyed
to the fine source label. This is the structural disentanglement signal:
without it, the trunk is free to swap which negative each head encodes.
The model returns the raw 3-way logits in its diagnostic dict so a
trainer that adds the aux CE term has direct access; the in-tree trainer
currently runs only the BCE term, in which case the disentanglement
emerges only weakly through the logsumexp competition. This gap is
documented in `implementation_notes.md`.

## Promotion Criteria (from packet)

```text
test F1            >  0.755
test PR-AUC        >  0.820
near-puzzle FPR    <  0.20    with puzzle recall >= 0.78
```

The architecture must also pass at least one central ablation
(`no_aux_3way`, `random_near_merged`, or `aux_only_no_logsumexp`)
showing the disentanglement actually carries signal.
