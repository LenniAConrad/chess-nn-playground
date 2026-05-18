# Math Thesis

Board FPN CNN

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.

Batch candidate rank: `2`.

Working thesis: Chess positions often need both exact square detail and coarse whole-board phase. A plain feature-pyramid network can process the board at `8 x 8`, `4 x 4`, and `2 x 2` resolutions, then fuse the maps back into a single classifier.

## Operator description

Let `x in R^{C x 8 x 8}` be the simple_18 board tensor (`C = 18`). Optional coordinate planes `r(s), f(s), c(s), p(s)` are concatenated along the channel axis so the convolutions see absolute-board context that translation-equivariant 3x3 filters drop.

Three bottom-up convolutional stacks with width parameter `W` produce

```
x8 = level8(x)        in R^{B, W, 8, 8}
x4 = level4(avg_pool2(x8))  in R^{B, 2W, 4, 4}
x2 = level2(avg_pool2(x4))  in R^{B, 4W, 2, 2}
```

Top-down fusion uses 1x1 projections `P_{2->4}` and `P_{4->8}` followed by nearest-neighbor upsampling:

```
y4 = x4 + upsample(P_{2->4}(x2))
y8 = x8 + upsample(P_{4->8}(y4))
```

This is the standard feature-pyramid network identity restricted to the three resolutions the `8x8` board admits.

## Pooled features

For each level the head concatenates mean and max pools:

```
pool(z) = [mean_{h,w}(z); max_{h,w}(z)]   in R^{2 * channels(z)}
```

The classifier head sees `phi(x) = [pool(y8); pool(y4); pool(x2)] in R^{2 (W + 2W + 4W)} = R^{14 W}`.

## Decision rule

`phi(x)` is normalised by a `LayerNorm` and fed through a small MLP

```
phi(x) -> Linear(14W, H) -> GELU -> Dropout? -> Linear(H, H/2) -> GELU -> Dropout? -> Linear(H/2, num_classes)
```

producing one logit per board for the BCE-with-logits puzzle head.

## Falsification path

The central falsifier is `single_resolution_matched`, which zeros `y4` and the `2x2` head feature so only the `8x8` bottom-up pool reaches the head. If the matched single-scale model matches the full FPN, multi-resolution fusion is not what is helping. `bottom_up_only` and `late_pool_only` ablate the top-down fusion path while keeping all three resolutions in the head; if they match, the top-down identity is unnecessary. `no_2x2_level` zeros only the coarsest feature; if it matches, the coarsest scale is unnecessary. `no_coordinate_planes` drops the coordinate planes; if it matches, absolute-board context is unnecessary.
