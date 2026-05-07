# Architecture

`Soft Formal-Concept Closure Network` implements a differentiable Galois closure bottleneck
over a per-board formal context built from the simple_18 board tensor.

## Pipeline

1. **`Simple18BoardAdapter`** validates the 18-plane layout
   (`P,N,B,R,Q,K,p,n,b,r,q,k`, side-to-move, four castling planes, en-passant
   plane), exposes side-relative own/enemy piece tensors, and refuses any
   encoding whose channel semantics are not verified.
2. **`RuleAttributeBuilder`** assembles the binary incidence matrix
   `A_x ∈ {0,1}^{B,64,M}` from deterministic, label-independent, engine-free,
   current-board predicates: coordinate (file/rank/side-relative rank/square
   color/edge/center), occupancy (empty, own/enemy, own/enemy by piece type),
   king geometry (Chebyshev distance bins to own/enemy king, same-rank/file),
   pseudo-legal pressure (attacks-by piece type for each side, attack-count
   thresholds, defended/attacks-piece flags), and ray geometry (clear ray of
   own/enemy slider to opposing king, conservative between/pin-candidate
   markers). Pseudo-legal attacks are computed from current occupancy only.
   The builder also returns a fixed-length vector of broadcast globals
   (side-to-move, four castling rights, en-passant file one-hot, normalized
   occupancy and material counts).
3. **`SoftConceptClosureLayer`** applies the soft Galois derivation operators
   from `math_thesis.md` Section 6:
       miss(g,k) = sum_m q_k[m] * (1 - A_x[g,m]) / sum_m q_k[m]
       extent[g,k] = exp(-miss(g,k) / tau_extent)
       w[g,k] = extent[g,k] / sum_h extent[h,k]
       miss_attr(k,m) = sum_g w[g,k] * (1 - A_x[g,m])
       closed_intent[k,m] = exp(-miss_attr(k,m) / tau_closure)
   The probes `q_k = sigmoid(raw_intents / intent_temperature)` are learned.
4. **`ConceptClosureReadout`** produces a per-concept summary vector that
   bundles closure statistics (extent mass, extent entropy, closure mass,
   closure-expansion `||relu(C-q)||_1`, closure-violation `||relu(q-C)||_1`,
   closure cosine similarity to `q_k`), a learned attribute embedding of
   `closed_intent`, an extent-driven embedding of the per-square attribute
   vector, and a per-probe embedding.
5. The shared concept MLP is applied to each of the `K` concept summaries; the
   resulting `(B, K, H)` tensor is pooled via mean / max / log-mean-exp over
   `K` and concatenated with the global broadcast features. A two-layer
   classifier MLP emits a single puzzle logit (`num_classes=1`), with a
   matching `two_class_logits` diagnostic constructed by symmetric splitting.

## Falsifier and ablation hooks

- `semantic_rewire_ablation` triggers a deterministic, seed-controlled bipartite
  double-edge rewire of `A_x` that preserves row and column sums (per
  `math_thesis.md` Section 9). It is the central row/column-preserving
  randomization control demanded by the thesis.
- `marginal_only_ablation` replaces the closure path with attribute column
  marginals plus globals, so marginal-only nuisance can be measured.
- `use_attack_attributes` and `use_ray_attributes` toggle the pseudo-legal
  pressure and ray groups for the coordinate-only and attack/ray-only
  ablations.

## Output contract

`forward(x)` returns a dictionary with `logits` (shape `(B,)` for
`num_classes=1`), `two_class_logits`, and closure diagnostics
(`extent_mass_*`, `closure_mass_*`, `closure_expansion_l1_mean`,
`closure_violation_l1_mean`, `closure_energy`, `mechanism_energy`,
`intent_density_mean`, `global_features`).

## Implementation Binding

- Registered model name: `soft_formal_concept_closure_network`
- Source implementation: `src/chess_nn_playground/models/soft_formal_concept_closure.py`
- Idea-local wrapper: `ideas/i057_soft_formal_concept_closure_network/model.py`
  delegates to `build_soft_formal_concept_closure_network_from_config`.
- The idea-local wrapper does not import or call the shared
  `ResearchPacketProbe` / `build_research_packet_probe_from_config` scaffold.
