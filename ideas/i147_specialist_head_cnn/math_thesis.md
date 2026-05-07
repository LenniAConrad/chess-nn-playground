# Math Thesis

`Specialist-Head CNN` tests whether a plain shared board CNN benefits from
small, semantically scoped readout heads instead of one undifferentiated global
pooling head.

The model starts with a shared convolutional square encoder:

```text
h = CNN(x),  h in R^(B x W x 8 x 8)
```

It then extracts fixed specialist summaries:

```text
global_feat   = mean/max_pool(h)
center_feat   = masked_pool(h, center_4x4)
edge_feat     = masked_pool(h, board_edge_ring)
king_feat     = masked_pool(h, own/opponent king zones)
material_feat = MLP(piece counts and phase features)
```

Each specialist produces a hidden feature and an auxiliary logit. The final
puzzle logit is produced by a learned MLP over the concatenated specialist
features and logits:

```text
z_i, l_i = head_i(feat_i)
logit = fusion_mlp(concat(z_i, l_i))
```

The falsifiable claim is that fixed chess-specific specialist summaries improve
the `puzzle_binary` signal over a single global pooled CNN head, while remaining
plain and diagnostic. Central falsifiers remove the king or material head,
replace learned fusion with a uniform logit average, or swap the semantic
center/edge masks for random masks of equal size.
