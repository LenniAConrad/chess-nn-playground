# Implementation Notes

- Central code: `src/chess_nn_playground/models/tiny_chess_micronet.py`.
- Registry key: `tiny_chess_micronet`.
- Idea wrapper: `ideas/registry/i073_tiny_chess_micronet/model.py`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2200_friday_shanghai_tiny_chess_micronet.md`.
- Input is board-only `simple_18`; engine outputs, source metadata, verification
  fields, and CRTK provenance are not consumed as model inputs.
- The default config is the packet's `micro_25k` tier: `width: 16`,
  `squeeze_rank: 6`, `blocks: 3`, `mix_rank: 6`, and `head_hidden: 32`.
- Main-path operations are quantization-friendly: `Conv1x1`, depthwise `Conv2d`,
  fixed sums/averages, `ReLU6`, and small `Linear` layers.
- The main readout uses deterministic sketch descriptors rather than flattening
  all `8 x 8 x width` hidden activations into a large dense head.
