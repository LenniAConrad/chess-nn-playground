# Architecture

`TensorSketch Interaction Network` is a randomized polynomial-kernel feature
map that approximates high-order piece-square interactions without enumerating
tuples. The classifier reads only board features; CRTK / source metadata is
reporting-only.

## Pipeline

1. Build a compact board feature vector `x_vec` from the `(B, 18, 8, 8)`
   `simple_18` tensor:
   - flattened occupancy and global planes (`B, 18*64`),
   - 12 piece-plane material counts (`B, 12`),
   - per-global-plane scalar reductions for side-to-move, castling rights and
     en-passant (`B, 6`).
2. Project `x_vec` linearly into a `base_dim`-dimensional base feature vector,
   followed by a `LayerNorm`. This is the degree-1 representation.
3. Apply a frozen CountSketch `(h, s)` from `base_dim -> sketch_dim`, with
   bucket assignments `h_i ~ Uniform{0, ..., sketch_dim-1}` and signs
   `s_i ~ Uniform{-1, +1}` sampled once from a fixed `sketch_seed`.
4. For each polynomial degree `d in sketch_degrees` (default `(2, 3)`), compute
   the TensorSketch via the FFT trick:
   `sketch_d(x) = real(IFFT(FFT(CountSketch(x)) ** d))`. A learnable per-degree
   log-scale lets the head balance polynomial orders.
5. Concatenate `[base, sketch_2, sketch_3, diagnostics]` (with diagnostics
   covering base mean / energy and per-degree mean / energy) and feed it to a
   compact `LayerNorm -> Linear -> GELU -> Dropout -> Linear` MLP head that
   emits one puzzle logit.

## Tensor Contract

```text
input:        (B, 18, 8, 8)
x_vec:        (B, 18*64 + 12 + 6) = (B, 1170)
base:         (B, base_dim)            default base_dim = 512
count_sketch: (B, sketch_dim)          default sketch_dim = 512
sketch_d:     (B, sketch_dim) for each d in sketch_degrees
features:     (B, base_dim + sketch_dim * |sketch_degrees| + 2 + 2*|degrees|)
logits:       (B,)
```

## Why this is not a shared probe

There are no convolutional trunks, no proposal-profile diagnostics and no
mechanism-family embeddings. The signal that reaches the head is exactly the
randomized polynomial-kernel approximation prescribed by the source packet,
plus a small number of sketch-energy diagnostics. Ablations on
`sketch_degrees` directly correspond to the central ablations described in the
source packet (`degree1_only`, `degree2_only`, sign reshuffles).

## Implementation Binding

- Registered model name: `tensorsketch_interaction_network`.
- Source implementation file: `src/chess_nn_playground/models/tensorsketch_interaction_network.py`.
- Idea-local wrapper: `ideas/i108_tensorsketch_interaction_network/model.py` (a
  thin `build_model_from_config` over
  `build_tensorsketch_interaction_network_from_config`; no
  `ResearchPacketProbe` is involved).
