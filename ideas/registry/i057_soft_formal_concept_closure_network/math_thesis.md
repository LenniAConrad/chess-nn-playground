# Math Thesis

Soft Formal-Concept Closure Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0922_tuesday_los_angeles_concept_closure.md`.

## Working thesis

Chess puzzle-likeness is partly expressed by small *closed* sets of
co-occurring, rule-derived board attributes; a differentiable Galois closure
bottleneck over a current-board formal context should detect tactical motif
coherence that marginal counts, CNN texture, and static attack graphs miss.

## Formal context

For each input `x` the model builds a per-board formal context
`K_x = (G, M, I_x)`:

- `G = {0, ..., 63}` are the 64 square objects.
- `M` is a fixed vocabulary of deterministic current-board predicates
  (coordinate, occupancy, king geometry, pseudo-legal pressure, ray geometry).
- `(g, m) ∈ I_x` iff square `g` has predicate `m` on board `x`.

The hard FCA derivation operators

    B'  = { g ∈ G : ∀ m ∈ B, (g, m) ∈ I }       for B ⊆ M
    A'  = { m ∈ M : ∀ g ∈ A, (g, m) ∈ I }       for A ⊆ G
    cl_M(B) = B'';   cl_G(A) = A''

form an antitone Galois connection. `cl_M` is extensive, monotone, and
idempotent (Proposition 1 below). The fixed points of `cl_M` are exactly
the intents of formal concepts.

## Soft (differentiable) Galois closure

Learned soft intent probes `q_k ∈ [0, 1]^M` parameterize `K` concepts. The
soft derivation operators with temperature-controlled relaxation are

    miss_x(g, k)        = (sum_m q_k[m] * (1 - A_x[g, m])) / (sum_m q_k[m] + eps)
    extent_x[g, k]      = exp(-miss_x(g, k) / tau_extent)
    w_x[g, k]           = extent_x[g, k] / (sum_h extent_x[h, k] + eps)
    miss_attr_x(k, m)   = sum_g w_x[g, k] * (1 - A_x[g, m])
    closed_intent_x[k, m] = exp(-miss_attr_x(k, m) / tau_closure)

The classifier consumes closure statistics (extent mass, extent entropy,
closure mass, closure expansion `||relu(closed - q)||_1`, closure violation
`||relu(q - closed)||_1`, closure cosine), learned attribute embeddings of
`closed_intent`, extent-weighted attribute embeddings, and per-probe
embeddings.

## Proposition 1 (hard closure facts)

For any finite formal context `K`, `cl_M(B) = B''` is extensive, monotone,
and idempotent:

    B ⊆ cl_M(B)
    B1 ⊆ B2  =>  cl_M(B1) ⊆ cl_M(B2)
    cl_M(cl_M(B)) = cl_M(B)

The fixed points are exactly the concept intents. The proof is the standard
Galois-connection argument.

## Optimization principle

Train `theta` by minimizing

    L(theta) =
        CE_balanced(y, f_theta(A_x, globals_x))
      + lambda_density   * R_density(q)
      + lambda_diversity * R_diversity(q)
      + lambda_idem      * R_idempotence(q)

where the regularizers are optional; the minimal experiment may set them all
to zero.

## What is proved vs hypothesized

- *Proved:* the hard FCA closure operator is extensive, monotone, idempotent;
  the row/column-preserving rewire ablation preserves first-order row and
  column degrees of `A_x` while breaking object-level co-instantiation.
- *Hypothesized:* puzzle-like positions show higher useful closure structure
  than non-puzzles under this attribute vocabulary; the soft relaxation is a
  faithful proxy for the hard closure.

The central row/column-preserving rewire control is implemented as
`semantic_rewire_ablation` in the bespoke model.
