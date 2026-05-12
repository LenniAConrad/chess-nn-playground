# Architecture

`Maxout Region Signature Network` is a board-only puzzle_binary classifier
that exposes the piecewise-linear activation regions induced by a stack of
maxout banks.  The classifier never sees raw activations; it only sees the
*signatures* of those regions: which expert wins each square, the margin by
which it wins, and how those winners change across the board.

## Pipeline

1. A small convolutional stem ``BoardConvStem`` (``input_channels`` ->
   ``channels``, ``depth`` blocks, optional ``BatchNorm`` + ``ReLU``) maps the
   `(B, 18, 8, 8)` `simple_18` tensor to a trunk feature map
   `(B, channels, 8, 8)`.
2. A stack of ``num_banks`` maxout banks is applied to that trunk.  Each bank
   is a 1x1 convolution to ``bank_units * bank_pieces`` channels followed by a
   reshape to `(B, bank_units, bank_pieces, 8, 8)` and a ``max`` reduction over
   ``bank_pieces``.  The deeper bank consumes the previous bank's maxout
   activation, so its region structure is conditioned on the earlier bank's
   region structure.
3. For each bank we build a *region signature* directly from its winners /
   margins, never from the raw activations:
   * **Winner histogram** -- mean one-hot frequency of the winning expert
     across the 64 squares, shape `(B, bank_units, bank_pieces)`.
   * **Region count** -- number of distinct experts that win at least one
     square per (B, unit), shape `(B, bank_units)`.
   * **Rank / file region count** -- mean number of distinct winners present
     per row and per column, shape `(B, bank_units)` each.  These pick up
     anisotropic region structure on the rank/file scaffolding of a chessboard.
   * **Transition counts** -- horizontal and vertical neighbour pairs whose
     winners differ.  These are exactly the number of region-boundary
     crossings under axis-aligned sweeps, shape `(B, bank_units)` each.
   * **Margin stats** -- per (B, unit) mean / std / max / min of `top1 - top2`
     over the 64 squares, shape `(B, bank_units, 4)`.  This is the
     piecewise-linear "confidence" with which the winning expert was chosen.
   * **Activation stats** -- per (B, unit) mean / std / max / min of the
     maxout output, shape `(B, bank_units, 4)`.
4. The classifier head receives the concatenation of every bank's flattened
   region signature plus a global-average pool of the trunk features, runs it
   through `LayerNorm -> Linear -> GELU -> Dropout -> Linear`, and emits one
   puzzle logit.  All region signatures are exposed alongside the logit so
   ablations and reports can read them without a second forward pass.

## Tensor Contract

```text
input:                     (B, 18, 8, 8)
trunk:                     (B, channels, 8, 8)
bank_activations[i]:       (B, bank_units, 8, 8)
winners[i]:                (B, bank_units, 8, 8)        argmax over bank_pieces
margins[i]:                (B, bank_units, 8, 8)        top1 - top2
winner_histogram[i]:       (B, bank_units, bank_pieces)
region_count[i]:           (B, bank_units)
rank_region_count[i]:      (B, bank_units)
file_region_count[i]:      (B, bank_units)
horizontal_transitions[i]: (B, bank_units)
vertical_transitions[i]:   (B, bank_units)
margin_stats[i]:           (B, bank_units, 4)           (mean, std, max, min)
activation_stats[i]:       (B, bank_units, 4)           (mean, std, max, min)
trunk_pool:                (B, channels)
logits:                    (B,)
```

The ``signature_dim_per_bank`` flattened length is
``units * pieces + units + 2*units + 2*units + 4*units + 4*units``.

## Why this is not a shared probe

There are no proposal-profile diagnostics, no mechanism-family embeddings, and
no shared `ResearchPacketProbe` code.  The signal that reaches the head is
exactly the maxout region structure prescribed by ``math_thesis.md`` --
winner identities, margins and region transitions -- supplemented only by a
single global-average trunk pool to anchor the head with a coarse board
context.  Ablations on ``num_banks``, ``bank_units`` and ``bank_pieces`` map
directly to the central design knobs in the source packet, and ablations that
hide individual signature components (e.g. drop the transition counts, or
drop the margin stats) are well-defined operations on this code path.

## Implementation Binding

- Registered model name: `maxout_region_signature_network`.
- Source implementation file:
  `src/chess_nn_playground/models/maxout_region_signature_network.py`.
- Idea-local wrapper:
  `ideas/registry/i109_maxout_region_signature_network/model.py` (a thin
  `build_model_from_config` over
  `build_maxout_region_signature_network_from_config`; no
  `ResearchPacketProbe` is involved).
