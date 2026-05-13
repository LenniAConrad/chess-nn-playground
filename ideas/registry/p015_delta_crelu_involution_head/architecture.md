# Architecture — DeltaCReLU + Involution Reynolds Head (p015)

This idea is an additive, gated side head on top of the
``ExchangeThenKingDualStreamNetwork`` trunk (i193). It does not replace
the trunk; removing the primitive head returns the i193 base logit
unchanged.

The model consumes the repository ``simple_18`` ``(B, 18, 8, 8)`` current-
board tensor and returns one puzzle logit for the BCE-with-logits
``puzzle_binary`` trainer, plus a per-sample diagnostics dict.

## Mechanism

1. **i193 trunk forward**. The bespoke ``ExchangeThenKingDualStreamNetwork``
   runs unchanged and emits ``logits`` (the ``base_logit``) plus the
   standard i193 diagnostics.

2. **Active-feature extraction**. The simple_18 board is converted to
   a compact ``(B, K)`` long tensor of active ``(piece_type, square)``
   feature indices, padded to ``max_features`` with a validity mask.

3. **Primitive-specific accumulator state**. Implemented in
   ``compute_state`` of ``DeltaCReLUInvolutionHead``. See
   ``implementation_notes.md`` for the per-primitive algebra and the
   correspondence to the source primitive's mathematical signature.

4. **Fusion**. The primitive state is concatenated with four stop-
   gradient trunk diagnostics (``gate``, ``gate_entropy``,
   ``mechanism_energy``, ``stream_disagreement``) and fed to a small
   LayerNorm + GELU MLP that produces the ``primitive_delta_raw`` scalar.
   A sigmoid gate conditioned on the trunk diagnostics multiplies the
   delta into the base logit:

       final_logit = base_logit + primitive_gate * primitive_delta_raw

5. **Ablations**. The supported ablation modes are documented in
   ``ablations.md``. They include the standard ``zero_delta`` /
   ``trunk_only`` baselines plus the primitive-specific falsifiers.

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata are
**not** consumed by the model. The active-feature extraction is rule-
derived from the simple_18 piece planes and side-to-move plane only.

## Implementation Binding

- Registered model name: ``delta_crelu_involution_head``.
- Source implementation: ``src/chess_nn_playground/models/primitives/delta_crelu_involution.py``.
- Shared helper: ``src/chess_nn_playground/models/primitives/delta_accumulator.py``.
- Idea-local wrapper: ``ideas/registry/p015_delta_crelu_involution_head/model.py``.
- Training config: ``ideas/registry/p015_delta_crelu_involution_head/config.yaml``.
- Builder entry in ``src/chess_nn_playground/models/registry.py``:
  ``MODEL_BUILDERS['delta_crelu_involution_head'] = build_delta_crelu_involution_from_config``.
