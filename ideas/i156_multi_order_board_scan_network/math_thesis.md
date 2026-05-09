# Math Thesis

Multi-Order Board Scan Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md`.

Batch candidate rank: `2`.

Working thesis: A chess board can be read as several short sequences. Different scan orders expose different dependencies. Five fixed orderings are used: rank-major (horizontal sweep), file-major (vertical sweep), anti-diagonal (south-west to north-east sweep), spiral-from-king (Chebyshev rings around the side-to-move king), and center-out (Chebyshev rings around the board centre). A shared bidirectional GRU consumes each per-square token sequence in its prescribed order; the order-pooled summaries are concatenated and fed to a small classifier that produces the puzzle logit. Sharing the sequence model across orders forces it to extract dependencies that are useful from multiple scan directions, while a per-scan, per-position embedding lets it tell the orders apart.
