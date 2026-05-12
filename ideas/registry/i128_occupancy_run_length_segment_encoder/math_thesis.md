# Math Thesis

Occupancy Run-Length Segment Encoder

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md`.

Batch candidate rank: `4`.

Sliding tactics are often determined by compact line facts: how many empty squares separate pieces, whether a segment is open to the edge, which piece types terminate the segment, and whether a king zone lies on or near the interval. Pins, skewers, batteries, x-rays, and open-file pressure can therefore be represented with run-length summaries instead of full ray-token grammars.

For a board line `l`, let occupancy split the line into maximal empty runs and maximal occupied runs. The model computes segment rows

```text
s = [empty length, occupied count, endpoint types, king-slider gap, king-zone contact, edge openness, line type, side-relative direction]
```

for ranks, files, diagonals, and anti-diagonals. A shared MLP maps each row into a segment embedding, line pooling compresses segments within the same line, and line-type pooling keeps rank/file/diagonal/anti-diagonal contributions separate.

This keeps the architecture distinct from a ray-language automaton: the model does not learn ordered state transitions over square tokens. It also differs from line scans that retain every square state. The inductive bias is a compressed, deterministic line segmentation that exposes blocker gaps and endpoint identities to the classifier while a small CNN branch preserves local board context.
