# Architecture

`Capsule Motif BoardNet` realises the source packet's capsule-style
motif-binding thesis as a bespoke architecture for the repo's
`puzzle_binary` task. Local board patterns are encoded as small
*primary capsule* vectors -- one per (square, primary-channel) cell --
and routed by iterative agreement into a small set of learned *motif
capsules*. Motif capsule lengths and the pooled trunk feature drive
the puzzle logit.

## Implementation Binding

- Registered model name: `capsule_motif_boardnet`
- Source implementation file: `src/chess_nn_playground/models/capsule_motif_boardnet.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i155_capsule_motif_boardnet/model.py`

## Modules

`CapsuleMotifBoardNet` accepts the project's `(B, 18, 8, 8)` board
tensor only. CRTK / source / engine / verification metadata is
reporting-only and is not consumed.

1. **Stem.** Two normalised rank/file coordinate planes are
   concatenated to the input. A `3x3` `Conv2d -> BatchNorm2d -> ReLU`
   stack of `depth` blocks lifts the `(input_channels + 2)` planes to
   the trunk channel dimension while preserving the `8x8` layout.
2. **Primary capsules.** A `3x3 Conv2d(channels -> num_primary_caps *
   primary_capsule_dim)` projects the trunk into `num_primary_caps`
   capsule channels per square. The output is reshaped to
   `(B, N_caps, primary_capsule_dim)` with `N_caps = 8 * 8 *
   num_primary_caps` and squashed along the capsule-vector axis to
   keep each primary capsule on a bounded manifold.
3. **Motif transforms.** A learned tensor
   `W` of shape `(num_motif_caps, motif_capsule_dim,
   primary_capsule_dim)` defines, for each motif `m`, a transformation
   matrix `W_m`. Motif predictions are computed via
   `u_hat[b, i, m] = W_m * primary[b, i]` (`einsum("mde,bie->bimd", W,
   primary)`), shared across primary capsule positions.
4. **Routing-by-agreement.** Routing logits `b[b, i, m]` start at
   zero. For `routing_iterations` iterations:
   - `c = softmax(b, dim=motif)`
   - `s[b, m] = sum_i c[b, i, m] * u_hat[b, i, m]`
   - `v[b, m] = squash(s[b, m])`
   - On all but the last iteration, update
     `b += <u_hat[b, i, m], v[b, m]>` where `v` is detached so the
     routing update does not differentiate through earlier rounds (the
     standard dynamic-routing recipe).
5. **Readout.** Motif capsule lengths `||v_m||` are concatenated with
   the global average-pooled trunk feature. A small MLP
   `Linear -> ReLU -> Dropout -> Linear` produces the single
   puzzle logit.

The squash function used is the standard
`squash(s) = ||s||^2 / (1 + ||s||^2) * s / ||s||`, applied along the
last (capsule-vector) axis.

## Loss

The default trainer wires standard BCE-with-logits on
`output["logits"]`. The capsule routing has no auxiliary loss term;
all gradient signal arrives through motif norms and the pooled trunk
feature feeding the head.

## Diagnostics

`forward` returns a dict containing:

- `logits`: shape `(B,)`. BCE-compatible log-odds for the one-logit
  puzzle_binary head.
- `logit`, `prob`: aliases of the log-odds and the sigmoid probability.
- `latent`: shape `(B, channels, 8, 8)`, the post-trunk feature map.
- `primary_capsules`: shape `(B, N_caps, primary_capsule_dim)`, the
  squashed primary capsule vectors.
- `motif_capsules`: shape `(B, num_motif_caps, motif_capsule_dim)`,
  the routed motif capsule vectors `v`.
- `motif_norms`: shape `(B, num_motif_caps)`, motif capsule lengths.
  This is the input the readout MLP sees alongside the pooled trunk.
- `routing_coupling`: shape `(B, N_caps, num_motif_caps)`, the final
  softmax coupling coefficients across motifs.
- `routing_logits`: shape `(B, N_caps, num_motif_caps)`, the final
  pre-softmax routing logits.
- `routing_entropy`: shape `(B,)`, the per-primary-capsule entropy of
  the coupling distribution averaged across primary capsules
  (detached).
- `max_motif_norm`, `mean_motif_norm`: shape `(B,)`, summary motif
  capsule lengths (detached).

The entropy and norm summaries are detached so they are reportable
without biasing the training loss toward minimising or maximising
agreement strength or capsule activation.

## Contract

- Input: `(B, C, 8, 8)` board tensor only. Engine, verification,
  source, CRTK, principal-variation, mate-score, and best-move
  metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit
  puzzle_binary BCE-with-logits trainer, plus the diagnostics listed
  above.
- Target mapping: fine labels `0` and `1` map to binary target `0`;
  fine label `2` maps to binary target `1`.
