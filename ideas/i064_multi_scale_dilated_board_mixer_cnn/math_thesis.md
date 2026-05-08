# Math Thesis

Multi-Scale Dilated Board Mixer CNN

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2107_friday_shanghai_multiscale_cnn_mixer.md`.

Working thesis: a compact conventional chess CNN should mix local,
knight-distance, diagonal/ray-like, and board-wide context in every
block, so a parallel 3x3 dilation 1 / dilation 2 / dilation 3 / 1x1
mixer with appended coordinate planes and a global-context channel
gate can be a stronger regular benchmark than a purely local stacked
CNN without introducing exotic research machinery.

The block operator on a feature map ``F`` of shape ``(width, 8, 8)``
is

```text
B(F) = ReLU( BN( P( cat( D1(F), D2(F), D3(F), C(F) ) ) ) + F )
```

where ``Dk`` is a ``3x3`` convolution with dilation rate ``k`` and
matched ``branch_width`` channels, ``C`` is a ``1x1`` channel branch,
``P`` is a ``1x1`` projection back to ``width``, ``BN`` is BatchNorm
and the residual ``+ F`` is added before the activation. The trunk
stacks ``num_blocks`` such blocks. After the trunk a global context
vector is computed by concatenating mean and max global pooling and
passing the result through a small MLP that emits a sigmoid scale
``g in (0, 1)^width``; the gated trunk ``g . F`` is mean+max pooled
and projected to ``num_classes`` puzzle logits.

The central falsifier from the source packet is
``single_dilation_matched``: replacing every parallel branch with a
single ``3x3`` dilation-1 branch at matched parameter count must hurt
the puzzle-binary metrics if the multi-scale mixing is the source of
the win. The folder also exposes ``no_dilation_3``,
``no_coordinate_planes``, ``no_global_context_gate``,
``small_width_control`` and ``residual_cnn_matched_params`` from the
section 6 ablation table.
