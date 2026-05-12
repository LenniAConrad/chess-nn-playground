# Mathematical Thesis

Soft-Dykstra LCP tests whether a board-only puzzle classifier benefits from forcing latent tactical evidence through a compact feasibility solver before reading out the binary logit.

The encoder proposes latent role mass `U`, relation mass `V`, motif mixture `M`, and bounded slack `S`. A finite unrolled Dykstra-style projector cycles through cheap constraints:

- box and simplex constraints;
- board-compatible role masks from the current tensor only;
- role mass budgets;
- fixed geometry masks for relation channels;
- endpoint compatibility between roles and relations;
- a simple target-closure constraint;
- compactness constraints that penalize diffuse certificates.

The classifier receives the board embedding, pre/post projection summaries, and solver trace features. The hypothesis is that verified puzzles should require smaller coherent corrections than hard negatives that look locally tactical but cannot satisfy the latent certificate constraints.

What is proven here is only the implementation of the board-only tensor contract and differentiable projection trace. The empirical claim remains falsifiable: if solver diagnostics do not separate positives from near-puzzle negatives, or if a parameter-matched encoder matches the result, this idea should be demoted.
