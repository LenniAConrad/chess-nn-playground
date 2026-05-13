# Implementation Notes

- Module location:
  `src/chess_nn_playground/models/primitives/complex_amplitude_chess_network.py`.
  Idea-local `model.py` calls the registry builder
  `build_complex_amplitude_chess_network_from_config`.

- Registry key: `complex_amplitude_chess_network`.

- Complex tensor handling: PyTorch's complex tensors and autograd
  through `torch.complex(real, imag)` are supported in eager mode. The
  module is intentionally kept *out* of `torch.compile` for the first
  scout run — the HANDOFF note warns that complex backward + compile
  combinations can be unstable. The eager forward is the production
  path; `torch.compile` can be revisited once a scout run validates the
  primitive empirically.

- Mixed precision: the CAIO outer product runs in `float32` by default.
  PyTorch's complex backward + AMP combination is well-supported when
  the magnitude / phase tensors are computed in `float32`. The trunk
  remains AMP-eligible.

- Complex dtype leakage: every value returned in the model's output dict
  is real-valued. Internal complex tensors live inside the head only.
  The test
  `test_caio_no_complex_dtype_leakage_in_output_dict` enforces this so
  the trainer's parquet writer never tries to serialise a complex
  tensor.

- Relation masks: closed-form `(64, 64)` indicator masks for four chess
  relations (king-zone adjacency, ray alignment, same-square-colour,
  file-rank adjacency). Built once with `build_relation_masks()` and
  registered as a non-persistent buffer. The masks have no learned
  parameters and zero self-edges (no `(u, u)` entries).

- Phase rule: a learned linear combination of three closed-form chess
  signals — piece colour `+1`/`-1`/`0`, side-to-move `±0.5`, and
  square colour `0 or 1`. The rule parameters `(alpha_piece,
  alpha_square, alpha_side)` are initialised to the chess-canonical
  values `(pi, pi/4, pi/2)` and are learnable. The `free_phase`
  ablation drops the rule contribution to test whether the rule tying
  is load-bearing.

- Conjugacy error: computed by running the encoder a second time on a
  colour-flipped copy of the board (the `color_flip_simple_18` helper
  swaps the white/black piece planes, side-to-move, and castling
  rights). The cost is one additional encoder forward — the trunk does
  not run on the colour-flipped board.

- Gate initialisation: the final linear layer of the gate MLP has its
  bias initialised to `gate_init = -2.0`, so the sigmoid starts near
  `0.12` and the primitive begins as a small additive correction. This
  matches the TSDP / TDCD / DHPE primitive head pattern.

- Trainer extension: none required. The model returns the standard
  output dict with `logits` and real per-sample scalar diagnostics. The
  shared trainer's `_primary_logits` and `_scalar_output_columns`
  helpers pick everything up automatically. No new dataset columns, no
  new losses.

- Cost: at the default `amplitude_dim=8`, every forward pass runs the
  encoder twice (once on the original board, once on the colour-flipped
  variant for conjugacy error) and computes a `(B, 8, 64, 64)` complex
  outer product per relation. Memory usage is dominated by the outer
  product — for `B=192, d=8, R=4` it is roughly
  `192 * 8 * 64 * 64 * 4 (real elements) * 8 bytes (complex64) = ~25 MB`
  per batch, well within scout GPU budgets.

- Future precomputation hook: if the rule-phase contribution becomes
  load-bearing, it can be moved into the data loader as a precomputed
  `(B, 8, 8)` real tensor at split-build time, reducing the encoder's
  per-batch CPU cost. The current path computes it from the simple_18
  tensor inside `_rule_phase`.
