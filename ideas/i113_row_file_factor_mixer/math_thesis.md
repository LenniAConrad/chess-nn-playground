# Math Thesis

Row-File Factor Mixer

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`.

Batch candidate rank: `1`.

Working thesis: Chess boards have two privileged axes: ranks and files. A model can exploit this without a full Transformer by factorizing board processing into rank mixers, file mixers, and piece-channel mixers, then recombining them with bilinear interactions.

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.
