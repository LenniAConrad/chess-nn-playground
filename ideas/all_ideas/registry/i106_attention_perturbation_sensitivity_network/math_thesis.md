# Math Thesis

Attention Perturbation Sensitivity Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md`.

Batch candidate rank: `5`.

Working thesis: Attention maps are often decorative unless perturbing the
attended regions changes the evidence the model collects. APSN uses
deterministic attention-guided perturbation sensitivity as the bottleneck:
how much the latent moves when high-attention vs low-attention board regions
are safely masked is the central feature, not the attention map itself.

Concretely, a base attention reader yields a latent ``z(x)`` and a per-query
attention map ``A in R^{Q x 64}`` softmaxed over the 64 board squares. The
per-square score ``s(x) = mean_q A_{q,*}`` selects four deterministic mask
families:

- top-K attention squares,
- K lowest-attention occupied squares,
- K permutation-random occupied squares (fixed seeded permutation),
- the 3x3 neighbourhood of the top-attention square.

Each mask zeros the 12 piece planes at its selected squares while leaving the
six global planes (side-to-move, castling, en-passant) untouched. The shared
encoder is re-run on each masked board to obtain
``z_top, z_low, z_rand, z_nbhd`` and the puzzle classifier reads the base
latent together with the four sensitivity scalars
``delta_* = ||z(x) - z_*||_2`` and contrasts such as
``contrast_top_low = delta_top - delta_low``,
``contrast_top_rand = delta_top - delta_rand``, plus a ratio
``delta_top / (delta_low + eps)``. A small set of attention diagnostics
(query entropy, peak attention, top-K mass, occupied vs empty mass,
query-axis variance, per-square range) is concatenated as side information
but the bottleneck is the sensitivity contrast, so a model that exploits
attention only as decoration cannot recover the central signal once the
contrast collapses.
