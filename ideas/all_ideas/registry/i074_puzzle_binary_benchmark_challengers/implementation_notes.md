# Implementation Notes

- Bespoke source module: `src/chess_nn_playground/models/puzzle_binary_benchmark_challengers.py`.
- Bespoke class: `NegativeClassDisentangledPuzzleHead`
  (alias `PuzzleBinaryBenchmarkChallengersNetwork`).
- Registry key: `puzzle_binary_benchmark_challengers`.
- Idea-local wrapper: `ideas/all_ideas/registry/i074_puzzle_binary_benchmark_challengers/model.py`
  exposes `build_model_from_config(config)` and delegates to
  `build_negative_class_disentangled_puzzle_head_from_config(config["model"])`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md`.
- Input contract: current-board `simple_18` tensor only. CRTK / source
  / engine metadata is reporting-only and is never used as a model
  input. The constructor enforces `input_channels=18` via the standard
  `BoardTensorSpec`.
- Output contract: forward returns a dict with `logits` of shape
  `(B,)` plus the diagnostics listed in `architecture.md`.

## Aux 3-way Loss Gap

The packet describes pairing the BCE-on-`puzzle_logit` objective with
an auxiliary 3-way CE on `[e_random, e_near, e_puzzle]` keyed to the
fine source label, with default weight `aux_weight: 0.25`. The
in-tree `puzzle_binary` trainer (`bce_with_logits` on
`output["logits"]`) does not yet attach that auxiliary term; the model
exposes the raw 3-way logits in `output["aux_3way_logits"]` so a
trainer that wants the full packet loss can pick them up without
changing the model. Until the trainer wiring is added, the
disentanglement emerges only through the logsumexp negative
competition. This is the one element of the packet not yet wired
end-to-end and is recorded here rather than papered over.
