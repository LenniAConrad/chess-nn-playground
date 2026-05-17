# Deep Research Prompt: Invent New Neural Primitives for Chess Evaluation

You are advising the `chess-nn-playground` research project. Your **single job** is to propose *new neural-network primitives* — not new architectures, not new input encodings, and not new training tricks. Each proposal must clear a high bar of novelty, and must be motivated by chess structure but plausibly generalise.

The user (a Tsinghua undergraduate, supervisor Prof. Han Jungong, Dept. of Automation) has already shipped a 234-architecture scout, an i242 chess-decomposed-attention follow-up with 4 ablations, and an i243 HalfKA+dual-stream proposal. All of those were *compositions of existing primitives*. The task now is one level deeper: **invent operators that do not exist in PyTorch yet.**

---

## What counts as a "primitive"

A primitive is a single mathematical operator that the deep-learning community would describe with a name (e.g., `Conv2d`, `Attention`, `LayerNorm`, `GELU`, `Mamba`, `MoE-gate`). It is "new" only if:

1. **It does not decompose into existing PyTorch ops at the computation-graph level.** "This is just `softmax(Wx + b)`" disqualifies a proposal. The new primitive must produce a different computation graph than any composition of existing ops would.
2. **It has a unique mathematical signature** — different input/output shapes, different gradient flow, different complexity class, or different connectivity pattern from anything in `torch.nn`.
3. **It is reusable beyond a single architecture.** A primitive that only fires once in one model is not a primitive — it's a layer. A real primitive should make sense as a future `torch.nn.<NewOp>`.

**Examples of real novel primitives** (use as calibration):

| primitive | inventor(s) | what was structurally new |
|---|---|---|
| GELU | Hendrycks, Gimpel (2016, grad students) | activation defined by Gaussian CDF, not piecewise |
| Adam | Kingma, Ba (PhDs) | per-parameter adaptive moments — new gradient flow |
| LayerNorm | Ba, Kiros, Hinton | normalisation over feature dimension (not batch), different gradient graph |
| Mixture-of-Experts gate | Jacobs et al. (1991) | routing-by-data is a non-decomposable conditional compute step |
| Mamba / S6 SSM | Gu, Dao (2023) | selective state-space recurrence with input-conditioned A, B matrices |
| RWKV | Peng et al. | a recurrent receptance variant that is not RNN, not attention |

**Anti-examples** (do NOT propose things like these):

- "A new attention mask shape" → still standard attention, just a different mask. Decomposes into `(Q@K^T + mask).softmax() @ V`.
- "A swish-but-with-different-polynomial activation" → just an activation function tweak; not a new primitive.
- "A new positional encoding" → encoding, not a primitive.
- "Two streams that share weights" → architectural composition, not a primitive.
- "A learned mixture of attention + conv" → composition of existing ops.

---

## Existing primitive memory: hard duplicate blocklist

The project already has a primitive research inventory under `ideas/research/primitives/`. Treat the following families as **already explored**. Do not return a proposal that is the same idea with renamed variables, a slightly different chess feature, a new acronym, or a thin combination of two listed families.

Already proposed local primitives:

- Signed Piece-Existence Hessian / Discrete Hessian-over-Piece-Existence / pair-resonance Hessian operators.
- Tempo-Defender Cross-Derivative operators.
- Promotion-Fanout Counterfactual Tensor operators.
- Complex-Amplitude Interference operators.
- Terminal-State Detection primitives.
- Pareto Antichain Frontier, Regret Saddlepoint, Reply Channel Capacity, Tail Copula Concordance, and Witness-Counterwitness Quantifier primitives.

Already imported external primitive families:

- Signed edit bilinear memory and ray-scan operators.
- Move-graph routers, legal-move graph accumulators, sparse legal graph transitions, legal-edge compilers, and legal-move kinematic state-space routers.
- Attack-ray sparse attention, rule-conditioned sparse attention, ray-occlusion dispatch, ray-blocked reducers, obstacle-pooling sparse emitters, ray-parallel SSMs, directional/octilinear scans, and ray-piece kernel updates.
- Sparse delta accumulators, segment scatters, delta-event routers, event-symmetric sparse scatters, incremental latent accumulators, reversible delta kernels, blocker-reset fastweights, and sparse differential move kernels.
- Delta pair selective bispectra, bilinear ray-blocked segment attention, occlusion semiring bilinear hyperedges, ray semiring exchange, and chi-head style reducers.
- CRELU/color-involution graph messages, color-involution adjacency updates, dynamic adjacency rank-order gates, and piece-relabelling/involution gates.
- High-risk legal graph delta-state, SLG diffusion, factor-graph/tensor-product style legal-state primitives.

Duplicate rejection standard:

- If the core state update is "maintain an accumulator under a bounded move delta", it is probably a duplicate unless the accumulator's algebra, gradient, and update complexity are genuinely different from the families above.
- If the connectivity is "legal move/ray graph decides where messages flow", it is probably a duplicate unless the primitive introduces a new non-message-passing computation graph.
- If the novelty is "content-conditioned sparse attention", it is probably a duplicate unless it is not expressible as sparse attention, graph attention, or routing over precomputed edges.
- If the primitive can be described by combining two blocklisted families, reject it. Do not propose hybrids such as "ray-occlusion delta accumulator" or "legal-move semiring scatter" unless the combination creates a new primitive-level operation with a distinct mathematical signature.
- Before finalising each proposal, explicitly compare it against the closest two blocklisted families. If the distinction is only domain vocabulary, feature choice, mask construction, or a different learned scoring function, drop the proposal and replace it.

---

## Non-negotiable rules

1. **No architecture proposals.** This prompt is not for new networks. If you find yourself drawing a block diagram with arrows between named modules, you are off-task.
2. **No input-encoding proposals.** HalfKA, simple_18, lc0_bt4_112, deterministic king/exchange planes are encodings. Encodings are the level *below* primitives in the stack.
3. **No training-trick proposals.** New losses, curricula, schedulers, data-augmentation tricks are out of scope.
4. **No hyperparameter rebrands.** "Conv with kernel size 9" or "attention with $h=32$ heads" is not a new primitive.
5. **Stockfish scores, PVs, node counts, verification metadata may NOT enter the primitive's compute graph as input features.** They are labels or audit fields, not inputs.
6. **No near-duplicates of the existing primitive memory.** A proposal is disqualified if its core operator is already covered by the blocklist above, even if the chess motivation, notation, or implementation harness changes.

---

## Principles from prior scout evidence

The 234-architecture scout, the i242 ablations, and the post-scout speed audit produced concrete empirical lessons. Encode these into your proposals:

- **Attention is data-hungry at small training scale.** i242's full chess-decomposed multi-stream transformer underperformed the conv-only parent (i193) at 173k positions × 12 epochs. Primitives that only justify themselves at LC0-scale (~10⁹) data should be flagged as *engine-scale-only*; primitives that win at scout-scale (~10⁵ positions) are more interesting because they are testable on the user's hardware.
- **Inference speed matters at least as much as accuracy.** A chess engine calls the network at every MCTS node — a 2× wall-clock speedup is typically worth more than 30 Elo of raw quality. Any new primitive must answer: (a) what is its asymptotic FLOPs profile vs the closest existing primitive? (b) does it support O(1) or O(log n) incremental update when the input changes by a bounded amount (the chess-move setting)?
- **HalfKA's O(1) accumulator update is the property to chase.** A primitive whose forward cost depends on *the change in input* rather than on *the input size* is structurally what makes Stockfish NNUE fast. Generalising that property as a primitive (e.g., for sparse-event sequences, dynamic graphs) is a promising direction.
- **Hard-negative discrimination is the real test.** The CRTK class 1 (verified-near-puzzle) is the discriminator that separates strong from weak architectures. A primitive that lifts overall PR AUC by exploiting easy negatives is uninteresting; a primitive that improves the *matched-recall near-puzzle FP rate* is genuinely architectural.
- **Group structure exists and is under-exploited.** Chess has more than dihedral-4 symmetry: there is also a color-swap involution, a piece-type relabelling structure, and a sparse legal-move graph that changes per board. Genuinely group-equivariant operators for the *chess group* (not the dihedral group) do not exist yet.
- **Sparse legal-move connectivity changes per token per forward pass.** This is the most chess-specific structural fact. A primitive whose connectivity is determined by the input (not by fixed positions) is structurally new — it does not decompose into masked attention because the mask is content-dependent in a way that standard attention masks are not.
- **Reproducibility constraints are real.** Single RTX 3070 (8 GiB), single seed, 12 epochs, 173k positions. A primitive that needs more than this to demonstrate value is fine — but say so explicitly and propose a smaller-scale falsification test that fits on the user's hardware.

---

## Required output schema (one block per proposal)

For each proposed primitive, produce a self-contained block with **exactly** the following fields. Do not skip fields; if a field is N/A, say so explicitly.

```
### primitive_<short_slug>

**Name:** <descriptive name, e.g. "Piece-Conditioned Legal-Move Attention">

**One-line claim:** <≤ 25 words describing the primitive in plain English>

**Mathematical signature:**
<exact formula, with shapes; e.g. f: R^{[B,n,d]} × R^{[B,n,n]} → R^{[B,n,d]}>
<give the forward equation; if recurrent, give the recurrence; if differentiable, the gradient must be well-defined>

**Why this does not decompose into existing PyTorch ops:**
<2–4 sentences identifying the structural property that prevents naive
decomposition. Be specific: name the existing op you'd compare to and
explain the computation-graph difference.>

**Duplicate audit against existing primitive memory:**
<Name the closest two blocklisted families and explain, in concrete
mathematical terms, why this proposal is not a duplicate. If you cannot
make a strong distinction, reject this proposal and do not output it.>

**Chess-specific motivation:**
<2–4 sentences pointing at a concrete chess structural fact that this
primitive exploits — legal-move sparsity, king-conditioning, piece-type
relabelling, color symmetry, incremental move updates, etc.>

**Generalisation beyond chess:**
<1–2 sentences naming at least one non-chess domain where this primitive
would plausibly find use (sparse-event sequences, scene graphs, dynamic
graphs, etc.). If the primitive is genuinely chess-only, say so —
that is allowed, but flag it.>

**Complexity (forward, backward, incremental-update):**
- Forward: O(...)  vs closest existing primitive O(...)
- Backward: O(...)
- Incremental update on a bounded-change input: O(...) — or "not applicable"

**Scout-scale falsification test:**
<a concrete experiment that fits on a single RTX 3070 in <2 GPU-hours.
Specify: which existing architecture to drop the primitive into,
what baseline to beat, what metric to measure, what result counts as
"primitive works" vs "primitive fails". The matched-recall near-puzzle
FP rate is preferred over aggregate PR AUC.>

**Failure mode catalogue:**
<3 bullet points: most likely ways this primitive could be (a) a
hidden rebrand of an existing op, (b) numerically unstable, or
(c) too slow to be useful even if it works. Anticipate the strongest
reviewer objection and name it.>

**Status:** proposed
```

---

## What I want you to do

When this prompt is pasted into GPT Deep Research:

1. **Survey the deep-learning literature** for actual recent novel primitives (last 5 years, prioritise 2024–2026). Use the calibration table above as the bar.
2. **Generate at least 10 rough candidate primitives internally**, then discard anything that duplicates the existing primitive memory before choosing the final 5.
3. **Generate 5 candidate primitives** for chess evaluation that clear the bar in the "What counts" section and pass the duplicate rejection standard.
4. **Rank them** on: (a) plausibility of novelty, (b) demonstrability on a single RTX 3070, (c) potential inference-speed advantage, (d) generalisation beyond chess.
5. **Self-audit all 5** by playing devil's advocate: for each, try to prove it is actually a hidden rebrand of an existing PyTorch primitive or one of the blocklisted project primitives. If you can prove it, drop the proposal and replace it.
6. **Cite real prior work** for any claimed novelty. If a proposal turns out to overlap with a 2024 NeurIPS / ICLR / ICML paper, say so and adjust the claim from "new primitive" to "underexplored primitive for chess."
7. **Do not propose architectures or encodings**, even if the user's earlier work suggests them. If you want to mention an architecture, do so only as the test harness in which the new primitive would be evaluated.
8. **Do not invent results.** No claims like "this primitive achieves +X PR AUC" without a citation or without explicitly marking the number as a prediction.
9. **Keep each proposal under 450 words.** The duplicate audit is mandatory, so the word budget is slightly larger than before.

Reply format: numbered list of 5 proposals using the schema above, followed by a short "what I cut" section listing the 5–10 candidate primitives you rejected during duplicate audit or self-audit and why.

Keep the answer concrete, evidence-bound, and structurally honest about novelty.
