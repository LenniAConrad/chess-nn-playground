# Math Thesis

## Scaffold-Only Implementation Notice

This folder is not a completed bespoke implementation of the architecture described below. `model.py` is a thin `ResearchPacketProbe` wrapper built with `build_research_packet_probe_from_config`, so this idea remains `implementation_kind: shared_probe_variant` and `implementation_status: probe_scaffold_only` until bespoke model code matching this markdown is added.


Commutative View-Consistency Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md`.

Batch candidate rank: `1`.

A chess position can be represented through several safe current-board views:
the square grid, occupied piece set, rank/file/diagonal line summaries,
king-centred regions, and material/phase summaries. Puzzle-like positions may
be recognizable not only by one view, but by how these views agree or disagree
after learned projections into a common latent space.

The model learns low-rank maps between latent views and classifies from
commutator-like residuals. The central bet is that near-tactical positions
create unusual cross-view consistency patterns, such as ordinary material
counts paired with high line or king-region defects.
