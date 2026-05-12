# Math Thesis

Tiny Chess MicroNet asks how much puzzle-binary signal survives under a strict
tiny-parameter constraint. The core claim is that a small model should spend its
capacity deciding which chess summaries matter, not relearning board geometry from
scratch.

The architecture is:

```text
simple_18 board
  -> low-rank 1x1 squeeze
  -> repeated depthwise/local plus fixed line-smoothing blocks
  -> deterministic chess sketch bank
  -> tiny ReLU6 descriptor head
  -> one puzzle logit
```

The deterministic sketch bank exposes low-dimensional chess structure:

- rank, file, diagonal, and anti-diagonal line summaries;
- side-relative line direction;
- occupancy-weighted line summaries;
- king-zone and king-ring pools;
- material and sparse state counts.

This makes the idea a tiny chess-aware architecture rather than a generic small
CNN, attention model, pruning recipe, or mobile baseline. The default implementation
targets the packet's `micro_25k` tier and reports parameter and INT8-size estimates
alongside the puzzle logit.

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2200_friday_shanghai_tiny_chess_micronet.md`.
