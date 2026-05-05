# Math Thesis

Bures-Wasserstein SPD Threat Manifold Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1510_tuesday_local_bures_wasserstein_threat.md`.

Working thesis: Embeds boards as SPD threat covariances and classifies via Bures-Wasserstein geodesic distances to learned class Frechet means; uses operator geometric mean rather than Fisher-Rao or log-Euclidean geometry.

This registered implementation routes the thesis through the `linear_algebra`
mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving
all source/CRTK metadata for reporting. See the source packet for the full
mathematical derivation, ablations, and falsification criteria.
