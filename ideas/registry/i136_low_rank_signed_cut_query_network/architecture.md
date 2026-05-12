# Architecture

`Low-Rank Signed Cut Query Network` is the bespoke implementation of the
research-packet candidate "Low-Rank Signed Cut Query Network": it consumes the
simple_18 board tensor, projects it into ``C`` learned board fields, and reads
``K`` learned *low-rank signed cut queries* over those fields to produce one
``puzzle_binary`` logit.

## Input

- Repo `simple_18` board tensor with shape `(batch, 18, 8, 8)`.
- All 18 planes contribute to the field projection. Piece planes ``5`` (white
  king) and ``11`` (black king) are also read directly to anchor the
  king-relative mask variants.
- CRTK / source / engine metadata is ignored — only the board tensor is
  consumed by the model.

## Board Field Projection

A 1x1 convolution maps the 18 input planes to ``C = num_fields`` learned linear
fields ``F_c(s)`` over the 64 squares. No nonlinearity is applied so the
fields stay linear reductions of the input planes; all nonlinearity sits in
the cut summaries and the head.

## Low-Rank Signed Mask Pairs

For every query pair ``k ∈ {1, …, K}`` the model stores two pairs of
coordinate factors and forms the rank-1 mask

```
a_k(rank, file) = tanh(r^a_k(rank)) * tanh(f^a_k(file))
b_k(rank, file) = tanh(r^b_k(rank)) * tanh(f^b_k(file))
```

so each mask is bounded in ``[-1, 1]`` and respects the paper's
``a_k(rank, file) = r_k(rank) * f_k(file)`` constraint.

## Signed Cut Summaries

For every pair ``k`` and every field ``c`` the model computes

```
cut_{k,c}      = sum_s a_k(s) F_c(s) - sum_s b_k(s) F_c(s)
abs_cut_{k,c}  = |cut_{k,c}|
sq_cut_{k,c}   = cut_{k,c}^2
norm_cut_{k,c} = cut_{k,c} / (eps + sum_s |F_c(s)|).
```

The four scalar tensors of shape ``(B, K, C)`` are flattened to form the
*global* cut summary.

## King-Anchored Mask Variants

A separate set of ``K_king = num_king_pairs`` low-rank mask pairs is shifted so
that the canonical centre ``(3, 3)`` lies on each board's white king and on
each board's black king. The shift is implemented by indexing the table of
all 64 ``torch.roll`` translations of every mask using the king-square argmax
of the white-king and black-king piece planes; when no king is present, the
masks fall back to the centred anchor.

The same four scalar summaries (signed/abs/squared/normalised) are computed
under the white-king and black-king anchors and concatenated to the fusion
vector.

## CNN Trunk And Classifier

A two-layer ``3x3`` convolutional trunk (``BatchNorm + ReLU``) reads the same
projected fields and global-averages to a feature vector. The cut summaries,
the king-anchored summaries, per-field global statistics
(mean, abs-mean, squared-mean) and the trunk vector are concatenated and fed
to a ``GELU`` MLP head with optional ``BatchNorm1d``, dropout, and a final
``Linear`` projection that emits one ``puzzle_binary`` logit.

The forward pass returns a dict whose ``logits`` tensor has shape ``(B,)``
alongside diagnostics including ``cut_signed_mean``, ``cut_abs_mean``,
``cut_squared_mean``, ``normalised_cut_mean``, ``cut_abs_max``,
``field_total_energy``, ``field_max_abs``, ``rank_file_imbalance``,
``king_anchored_cut_mean``, ``king_anchored_abs_cut_mean``,
``white_king_cut_energy`` and ``black_king_cut_energy`` — all of shape
``(B,)`` — for ablation analysis.

## Implementation Binding

- Registered model name: `low_rank_signed_cut_query_network`.
- Source implementation: `src/chess_nn_playground/models/low_rank_signed_cut_query_network.py`.
- Idea-local wrapper: `ideas/registry/i136_low_rank_signed_cut_query_network/model.py`.
