# Implementation Notes

- Central code: `src/chess_nn_playground/models/puzzle_binary_benchmark_challengers.py`
  (`NegativeClassDisentangledPuzzleHead`,
  `build_negative_class_disentangled_puzzle_head_from_config`).
- Idea-local wrapper: `ideas/i170_negative_class_disentangled_puzzle_head/model.py`.
- Registry key: `negative_class_disentangled_puzzle_head`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md`.
- Batch candidate: `Negative-Class Disentangled Puzzle Head` (rank 1).
- The bespoke implementation is shared with idea `i074`
  (`puzzle_binary_benchmark_challengers`); both registry entries point at
  the same `NegativeClassDisentangledPuzzleHead` class. There is no
  `ResearchPacketProbe` in the wiring.
- This is intentionally board-only and does not consume engine,
  verification, source, or CRTK metadata as input. Fine source labels
  feed the optional 3-way auxiliary CE through `aux_3way_logits` only at
  training time; inference uses `logits` (the disentangled
  `e_puzzle - logsumexp([e_random, e_near])`) exclusively.
- The bundled ablations (`no_aux_3way`, `random_near_merged`,
  `aux_only_no_logsumexp`, `shuffle_fine_negative_labels`) are accepted
  via `model.ablation` and exercised by the focused test in
  `tests/test_research_architectures.py`.
