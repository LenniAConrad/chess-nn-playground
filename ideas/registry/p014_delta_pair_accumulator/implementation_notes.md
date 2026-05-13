# Implementation Notes — Δ-Pair Accumulator (p014)

- Central model code: ``src/chess_nn_playground/models/primitives/delta_pair_accumulator.py``.
- Idea-local wrapper: ``ideas/registry/p014_delta_pair_accumulator/model.py``.
- Registry key: ``delta_pair_accumulator``.
- Source primitive packet: ``ideas/research/primitives/external_08_delta_pair_ray_selective_bispectrum.md``.

## Active-feature extraction

The simple_18 board is converted to a compact ``(B, K)`` long tensor of
active ``(piece_type, square)`` feature indices via
``extract_active_features`` in
``src/chess_nn_playground/models/primitives/delta_accumulator.py``. ``K``
defaults to 40 (a legal chess position has at most 32 pieces). The
extraction is deterministic and run inside the forward pass; it does
not need a python-chess fallback because piece placement is fully
captured by the 12 piece planes.

## ``O(|Δ|)`` inference contract

The static-position trainer evaluates the analytical fixed point of the
delta-accumulator recurrence:

    h = Σ_{i ∈ S(x)} W[i]

where S(x) is the active piece-square index set. At engine make/unmake
time the same model state can be advanced by

    h ← h + W[i_add] − W[i_remove]

for the bounded |Δ| ≤ 6 piece-square index changes per ply, matching the
HalfKA accumulator update path. The embedding table ``W`` is shared
between the forward and the delta paths so gradients are bitwise
consistent between the static-position trainer and any future
make/unmake inference wrapper (see ``DeltaAccumulator.embedding`` in
``delta_accumulator.py``).

## Stop-gradient contract

- Active-feature indices are integer-valued and not differentiable;
  they are computed inside the forward pass via boolean thresholding
  on the piece planes.
- Trunk diagnostics fed into the fusion MLPs are detached so the head
  cannot back-propagate into the trunk's gate / pooling logic.
- The gradient path is entirely through the trunk's normal autograd
  graph plus the head's learnable accumulator embedding, projections,
  and MLP weights.

## Output dict contract

The model output is a ``dict[str, Tensor]`` following the i193 contract
extended with:

- ``logits`` (rebound to ``base_logit + primitive_delta``)
- ``base_logit``
- ``primitive_delta``, ``primitive_delta_raw``, ``primitive_gate``,
  ``primitive_gate_logit``, ``primitive_active_count``,
  ``primitive_state_norm``
- primitive-specific per-sample diagnostics emitted from
  ``compute_state``.

All per-sample scalar tensors are emitted in the standard one-column-
per-key shape so the shared trainer copies them into
``predictions_<split>.parquet``.

## Why this is not a ``ResearchPacketProbe`` scaffold

The model is a bespoke ``nn.Module`` that wraps the bespoke i193
``ExchangeThenKingDualStreamNetwork`` and adds a delta-accumulator head.
It does not call ``build_research_packet_probe_from_config`` and does
not delegate to a shared CNN / MLP / NNUE baseline builder. The
``implementation_kind: bespoke_model`` declaration matches the
``audit_implementation_kinds.py`` heuristics.
