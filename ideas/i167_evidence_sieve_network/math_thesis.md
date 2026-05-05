# Math Thesis

Evidence Sieve Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.

Batch candidate rank: `6`.

Working thesis: Instead of refining logits, the model can refine features by repeatedly filtering them through learned evidence sieves. Each sieve stage produces a soft mask over channels and squares, passes selected evidence onward, and leaves a diagnostic trail.

This registered implementation tests the thesis through the `information` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.
