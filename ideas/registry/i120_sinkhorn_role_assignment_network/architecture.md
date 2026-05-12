# Architecture

`Sinkhorn Role Assignment Network` assigns occupied piece tokens to a fixed
set of learned tactical role slots through a differentiable masked
optimal-transport layer. The transport matrix is the only mechanism by
which pieces reach the classifier: roles are pooled by the assignment, mixed
through a small pairwise interaction MLP, and fused with a light board CNN
summary before producing one puzzle logit.

- Mechanism family: `transport`.
- Input: ``simple_18`` board tensor only; CRTK / source / engine metadata
  stays reporting-only.
- Token extraction: the first ``Pmax = max_tokens`` occupied squares in
  rank-major order are extracted as piece tokens. Each token is a
  concatenation of the 12 piece planes, the 6 global ``simple_18`` planes,
  6 coordinate features (rank, file, centred rank/file, edge distance,
  square parity), and 4 local-occupancy features (own/opponent counts in a
  3x3 and 5x5 neighbourhood around the square). Padded slots receive a
  zero token mask.
- Token encoder: a 2-layer ``Linear -> LayerNorm -> GELU -> Linear ->
  LayerNorm`` MLP projects each piece token to ``D = token_dim``. Padded
  tokens are zeroed out.
- Role prototypes: a learned parameter matrix of shape ``(M_total, D)``
  with ``M_total = num_roles + 1`` (a dustbin role is appended unless
  ``use_dustbin = false``). The dustbin absorbs irrelevant pieces that do
  not fit any tactical slot.
- Cost matrix: tokens and prototypes are projected and L2-normalised so the
  per-pair cost is ``cost[b, i, j] = 1 - cosine_sim(W * token_i, proto_j)``.
- Sinkhorn-Knopp transport: the assignment ``A`` of shape
  ``(B, Pmax, M_total)`` is computed by ``T = sinkhorn_iters`` log-domain
  Sinkhorn iterations on the kernel ``exp(-cost / temperature)``. Row mass
  is the token mask (1 for occupied pieces, 0 for padding) and column mass
  is ``softmax(role_mass_logits) * sum(token_mask)``, i.e. a learned target
  role-mass distribution scaled to the number of active pieces. Padded rows
  and inactive columns are masked to ``-inf`` in the kernel so they cannot
  transport mass; iterations run in log-domain with ``logsumexp``. After the
  iterations ``sum_j A[b, i, j] = mask[b, i]`` and
  ``sum_i A[b, i, j] = role_target_mass[b, j]`` (up to numerical eps).
- Role pooling: ``role_vectors[b, j] = sum_i A[b, i, j] * tokens[b, i]``.
  Padded tokens contribute zero because their row of ``A`` is zero.
- Pairwise role-slot interactions: a 2-layer MLP processes
  ``concat(role_i, role_j, |role_i - role_j|)`` for every ordered pair of
  roles; the off-diagonal pair tensor is averaged into a single pair
  summary vector.
- CNN summary branch: a 2-layer 3x3 convolutional encoder over the simple_18
  board tensor with ``cnn_channels`` channels followed by mean+max global
  pooling.
- Head: the classifier receives the flattened role vectors, the pair
  summary, the CNN summary, and a diagnostic vector with role mass / role
  share, dustbin share, mean assignment entropy, role-vector norms, mean
  and variance of per-piece transported mass, and the active token count
  normalised by ``Pmax``. A ``LayerNorm -> Linear -> GELU -> Dropout ->
  Linear`` head emits a single puzzle logit.

Diagnostic outputs returned from ``forward`` include the cost matrix,
assignment matrix, role vectors, role mass, role share, role norms,
per-piece transported mass, dustbin share, mean assignment entropy,
pair-interaction summary, CNN summary, and the diagnostic feature block fed
to the classifier.

## Implementation Binding

- Registered model name: `sinkhorn_role_assignment_network`.
- Source implementation file: `src/chess_nn_playground/models/sinkhorn_role_assignment_network.py`.
- Idea-local wrapper: `ideas/registry/i120_sinkhorn_role_assignment_network/model.py`.

The wrapper imports ``SinkhornRoleAssignmentNetwork`` and
``build_sinkhorn_role_assignment_network_from_config`` and is free of any
reference to ``ResearchPacketProbe`` or
``build_research_packet_probe_from_config``.
