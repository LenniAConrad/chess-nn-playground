# Math Thesis

Board FPN CNN

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.

Batch candidate rank: `2`.

Working thesis: Chess positions often need both exact square detail and coarse whole-board phase. A plain feature-pyramid network can process the board at `8 x 8`, `4 x 4`, and `2 x 2` resolutions, then fuse the maps back into a single classifier.

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.
