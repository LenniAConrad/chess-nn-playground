# Math Thesis

Patch Mixer BoardNet

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.

Batch candidate rank: `4`.

Working thesis: Use a plain MLP-Mixer-style model over `2 x 2` chess patches. This is a simple non-attention alternative to square-token models: mix information across board patches with MLPs, then mix channels with MLPs.

Implemented hypothesis: a compact patch Mixer over 16 board patches can test whether direct global patch mixing helps puzzle-binary classification without relying on convolutional locality or attention.
