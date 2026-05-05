# Math Thesis

Attention Perturbation Sensitivity Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md`.

Batch candidate rank: `5`.

Working thesis: Attention maps are often decorative unless perturbing attended regions changes evidence. This model uses deterministic attention-guided perturbation sensitivity as the bottleneck: how much the latent or logits move when high-attention versus low-attention b...

This registered implementation tests the thesis through the `graph` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.
