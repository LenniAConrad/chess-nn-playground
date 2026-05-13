# Implementation Notes

- Source module:
  `src/chess_nn_playground/models/trunk/tempo_defender_cross_derivative_network.py`.
  The idea-local `model.py` is a thin wrapper that calls
  `build_tempo_defender_cross_derivative_network_from_config(config["model"])`.

- Registry key: `tempo_defender_cross_derivative_network` in
  `src/chess_nn_playground/models/registry.py`. `config.yaml model.name`
  matches the slug exactly.

- `implementation_kind: bespoke_model`. Detected as such by
  `audit_implementation_kinds` because `model.py` calls a non-shared
  builder and the trunk module defines local `nn.Module` classes
  (`SaliencyHead`, `TDCDEncoder`, `TempoDefenderCrossDerivativeNetwork`).

- The model uses `simple_18` only. Tempo flip `sigma_T` is `x[..., 12] :=
  1 - x[..., 12]` and is its own inverse. The defender-removal operator
  `delta_k` zeros the enemy-coloured piece planes at a single square; the
  enemy colour is selected per-sample from the stm plane. No
  `python-chess` calls in the forward path. No data loader changes needed.

- The cross-derivative grid is built explicitly in `_build_grid`. The
  `TDCDEncoder` is a compact two-layer GroupNorm+GELU CNN distinct from
  the i193 trunk; this keeps cost ~2-3x i193 (the lightweight encoder
  evaluates 2*(K+1)=8 board variants per sample but is significantly
  cheaper than the i193 dual-stream trunk). GroupNorm is used inside the
  TDCD encoder so the eight grid copies do not couple through running
  batch statistics.

- The discriminator gate is initialised with `gate_init = -2.0` so the
  TDCD head starts as a near no-op (sigmoid(-2) ~= 0.12). Training drives
  the gate up only if the cross-derivative spectrum improves the loss.

- Saliency masking uses an `-inf` fill on non-enemy squares before the
  softmax/topk. Empty boards are guarded by replacing `-inf` with a finite
  sentinel and by a `top_valid` mask that zeros out invalid slot
  contributions before the cross-derivative reduction.

- Ablations live in the `ablation` config field. Allowed values:
  `none`, `main_effects_only`, `no_mixed_partial`,
  `null_board_perturbation`, `attacker_perturbation`,
  `skip_cross_derivative`, `shared_saliency_uniform`, `fixed_zero_gate`.

- The model does not change the trainer or the dataset contract. It
  consumes only `batch["x"]` and returns the standard
  `{"logits": Tensor[B], ...}` dict that `Trainer._primary_logits`
  consumes. Diagnostics keys are flat scalar-per-sample tensors so
  `_scalar_output_columns` can write them into per-position reports.

- The model is gradient-safe under AMP: GroupNorm is used inside the
  cross-derivative encoder (BatchNorm would couple grid copies and break
  the perturbation semantics), and the softmax fill uses `-inf` only over
  masked-out squares which then get replaced with a finite sentinel
  before topk.

- Cost-control: `topk` is exposed via config (1, 3, 5) for the K-sweep
  ablation. `tdcd_channels` controls the encoder width independently of
  the i193 trunk width.

- Conformance: i244 reserved label is consistent with the
  `PRIMITIVE_TRAINING_TODO.md` table (`TDCD: i244`). Promotion order
  remains TSDP -> PFCT -> TDCD; this folder is the third primitive in the
  Claude Opus 4.7 batch and does *not* override the recommended order.
