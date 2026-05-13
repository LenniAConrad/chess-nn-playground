# Trainer Notes — DeltaCReLU + Involution Reynolds Head (p015)

Use the guarded idea ``train.py`` (``idea_train_cli``). The config is
paper-grade and CUDA-required, mirroring the i193 baseline so the
architecture-level comparison is matched on:

- same train/val/test split
- same encoding (``simple_18``)
- same seed
- same training budget and early-stopping policy
- same threshold-selection rule

Differences vs the i193 baseline:

- ``model.name = delta_crelu_involution_head`` (this idea's wrapper builder)
- ``model.accumulator_dim``, ``model.max_features``,
  ``model.head_hidden_dim``, ``model.head_dropout``, ``model.gate_init``,
  ``model.ablation`` for the delta-accumulator head plus the
  primitive-specific extras documented in ``config.yaml``
- All trunk hyper-parameters retain their i193 names with a ``trunk_``
  prefix in the config so the builder forwards them to the wrapped
  ``ExchangeThenKingDualStreamNetwork``.

## Loss

``bce_with_logits`` on the puzzle logit. No primitive-specific auxiliary
loss is required — the head learns through the main BCE signal.

## Cost expectation

Forward-pass overhead is dominated by the i193 trunk; the delta-
accumulator head is an embedding gather plus a small MLP and runs in
near-constant time relative to the trunk. At ``accumulator_dim = 64``
the head adds roughly ``12·64·64 ≈ 50k`` parameters (embedding) plus a
few hundred MLP parameters.

## Ablation runs

Promotion of p015 requires the falsifier ablations declared in
``ablations.md``. The primary falsifiers are listed there with the
expected behaviour ("matched ablation must lose the lift").

## Reports

Standard idea report. Required slices follow the project benchmark
spec (see ``ideas/docs/BENCHMARK_REPORTING.md``). The diagnostic
columns ``primitive_gate``, ``primitive_delta``,
``primitive_active_count``, and ``primitive_state_norm`` are surfaced
in ``predictions_<split>.parquet`` for slice analysis.
