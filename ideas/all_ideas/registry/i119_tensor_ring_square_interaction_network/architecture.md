# Architecture

`Tensor-Ring Square Interaction Network` factorises high-order
square-by-square interactions through a learned tensor-ring
contraction. Tuples of squares are never enumerated explicitly --
instead, the model evaluates a cyclic trace of low-rank cores whose
contributions are gated by learned per-square role masks.

- Mechanism family: `linear_algebra`.
- Input: ``simple_18`` board tensor only; CRTK / source / engine
  metadata stays reporting-only.
- Square tokens: each of the 64 squares is projected from
  ``(input_channels + 5)`` features (board planes plus rank, file,
  side-relative rank, center distance, square parity) into a token of
  width ``token_dim`` and L2-normalised by a ``LayerNorm``.
- Role gate bank: a single ``Linear -> Sigmoid`` produces ``R = 5``
  per-square role gates ``g_r(s) in [0, 1]`` for the named roles
  ``own_piece``, ``opp_piece``, ``king_zone``, ``ray_relevant``,
  ``empty_square``. Gates are *learned*, not hard-coded legal-move
  labels.
- Tensor-ring cores: for each interaction order ``K in orders`` the
  model holds ``K`` independent linear maps
  ``G_k : R^{token_dim} -> R^{rank x rank}`` implemented as
  ``Linear`` layers from ``token_dim`` to ``rank * rank`` followed by
  a reshape.
- Pattern bank: for each order ``K`` a learned ``(num_patterns, K, R)``
  parameter is softmaxed along the role axis to produce per-pattern,
  per-slot role mixtures ``alpha_{p, k, r}``. This is the learned
  analogue of named pattern sequences such as
  ``own_attacker -> blocker -> king_zone``.
- Cyclic contraction: for every pattern ``p`` of order ``K`` the model
  computes
  ``M_{p, k} = (1 / 64) * sum_s alpha_{p, k, r} g_r(s) * G_k(x_s)`` and
  ``z_{p} = trace(M_{p, 1} M_{p, 2} ... M_{p, K})`` using
  ``O(64 * num_patterns * K * rank^2)`` work. The square-count
  normalisation keeps the cyclic trace bounded.
- Pooling: per-order summary statistics ``mean``, ``max``,
  ``variance`` and ``signed_abs_mean`` are computed over the
  ``num_patterns`` axis, concatenated with the raw pattern responses,
  and then concatenated with a small CNN-stem summary (a 2-layer 3x3
  convolutional encoder followed by global average pooling).
- Head: a ``LayerNorm -> Linear -> GELU -> Dropout -> Linear``
  classifier emits one puzzle logit. Diagnostic outputs include the
  per-order contractions, per-order summary statistics, per-order
  pattern softmax, role-gate activity / entropy, and per-order core
  Frobenius-norm magnitudes.

## Implementation Binding

- Registered model name: `tensor_ring_square_interaction_network`.
- Source implementation file: `src/chess_nn_playground/models/tensor_ring_square_interaction_network.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i119_tensor_ring_square_interaction_network/model.py`.

The wrapper imports
``TensorRingSquareInteractionNetwork`` and
``build_tensor_ring_square_interaction_network_from_config`` and is
free of any reference to ``ResearchPacketProbe`` or
``build_research_packet_probe_from_config``.
