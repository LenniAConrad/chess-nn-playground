# Architecture

`Color-Flip Orbit Evidence Bottleneck` (idea i047, abbreviated `CFOEB`)
is a board-only `puzzle_binary` classifier whose central operator is
the deterministic two-element color-flip orbit `G = {e, tau}` over the
`simple_18` board planes followed by a harmonic evidence-intersection
head, exactly as specified by `math_thesis.md` and the source research
packet. The shared `ResearchPacketProbe` mechanism profile is no
longer used here; the implementation is materially distinct from the
i046 Reynolds-projection variant because the per-class evidence is
fused with a *harmonic* mean rather than a probability/logit average.

## Forward Pipeline

1. **`ColorFlipOrbitAdapter`.** A deterministic, parameter-free adapter
   constructs the orbit `{x, tau(x)}` from the `simple_18` board
   tensor. The exact color-flip transform `tau` performs (a) a vertical
   rank mirror, (b) a swap of the white and black piece occupancy
   planes (channels 0..5 and 6..11), (c) a complement of the
   side-to-move plane (channel 12), (d) the `KQkq -> kqKQ` swap of the
   four castling-rights planes (channels 13..16), and (e) the rank
   mirror of the en-passant plane (channel 17). With
   `orbit_transform="bad_rank_color"` the adapter applies a
   chess-invalid rank permutation plus color swap (the central
   semantics-destroying falsifier from the research packet) and with
   `orbit_transform="identity"` it duplicates the board view. The
   adapter fails closed if the channel schema is not `simple_18`/18
   channels unless explicitly opted out.
2. **Shared `SharedBoardEncoder`.** A compact convolutional encoder
   shared across orbit views: a two-layer `3x3` Conv stem with
   GroupNorm + GELU and a `Dropout2d` mid-norm regularizer, followed
   by `num_res_blocks` `ConvResidualBlock`s (each two `3x3`
   convolutions with GroupNorm + GELU). The shared encoder is applied
   to the flattened `[2B, C, 8, 8]` orbit, producing a per-view
   feature map of shape `[2B, H, 8, 8]`.
3. **Global pooling and shared projection.** Spatial mean over the
   `8x8` map yields `[2B, H]`, a `Linear -> LayerNorm -> GELU ->
   Dropout` projection produces `z` of shape `[2B, D]` and is
   reshaped to `[B, 2, D]`.
4. **`OrbitEvidenceIntersectionHead`.** A shared `Linear(D, 2)` head
   fed through `softplus(.) + epsilon` produces strictly positive
   per-view class evidence `e[B, 2 views, 2 classes]`. The harmonic
   evidence intersection
   `I_c = 2 * e0_c * e1_c / (e0_c + e1_c + epsilon)` fuses the two
   orbit views; the final two-class score is `s_c = log(1 + I_c)`.
   For the puzzle-binary trainer (`num_classes=1`) the model emits
   the single binary logit `s_1 - s_0`; for `num_classes=2` it emits
   the pair `[s_0, s_1]`. The harmonic intersection is symmetric in
   the two arguments, so swapping the views leaves it invariant —
   this is what makes the final logit exactly `tau`-invariant in
   eval mode (Proposition 1 of `math_thesis.md`).

## Output Contract

`forward(x)` returns a `dict` keyed by:

- `logits`: shape `(B,)` for `num_classes=1` (single-logit BCE used
  by the puzzle-binary contract) or `(B, 2)` for `num_classes=2`.
- `negative_evidence`, `puzzle_evidence`: harmonic-intersected
  evidence `I_0`, `I_1`.
- `evidence_balance`: `I_1 - I_0`, the orbit-stable evidence
  imbalance towards the puzzle class.
- `view_negative_evidence_gap`, `view_puzzle_evidence_gap`: per-class
  `|e_0 - e_1|`, the per-view evidence defect that the harmonic
  intersection bottlenecks.
- `orbit_evidence_residual`: mean-over-classes of `|e_0 - e_1|`, the
  central diagnostic from the math thesis.
- `symmetry_residual`: alias of `view_puzzle_evidence_gap` exposed
  for the falsification observable.
- `latent_orbit_variance`: per-sample variance of the orbit latent
  `z`, a diagnostic of how strongly the shared encoder breaks the
  orbit invariance before the head.
- `identity_puzzle_evidence`, `flipped_puzzle_evidence`,
  `identity_negative_evidence`, `flipped_negative_evidence`: the four
  raw per-view evidence tensors.
- `intersection_energy`: mean squared harmonic-intersection
  magnitude.

These auxiliary tensors are exactly the falsification observables
listed in `math_thesis.md`: the orbit evidence residual is the
symmetry defect that separates the rule-exact color flip from the
`bad_rank_color`/`identity` ablations.

## Falsifier Wiring

The packet's central falsifier is `tau -> bad_rank_color` (a
chess-invalid rank permutation that still swaps colors and side to
move). The same model class supports it directly through
`orbit_transform="bad_rank_color"`, which leaves the encoder, head,
and parameter count unchanged. `orbit_transform="identity"` recovers
the duplicated-view ablation. All three branches share the
`SharedBoardEncoder` and `OrbitEvidenceIntersectionHead` so that any
difference in puzzle-binary metrics isolates the orbit operator
rather than capacity or augmentation.

## Implementation Binding

- Registered model name: `color_flip_orbit_evidence_bottleneck`
- Source implementation file: `src/chess_nn_playground/models/color_flip_orbit_evidence.py`
- Idea-local wrapper: `ideas/registry/i047_color_flip_orbit_evidence_bottleneck/model.py`
