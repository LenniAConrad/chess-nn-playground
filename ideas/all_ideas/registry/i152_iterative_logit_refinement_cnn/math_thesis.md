# Math Thesis

Iterative Logit Refinement CNN

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.

Batch candidate rank: `4`.

## Working thesis

Instead of producing a single logit vector at the end, let a model make an
initial prediction and then apply several learned correction steps from
shared board features. The model tests whether puzzle evidence is better
accumulated as staged corrections rather than a single readout.

## Refinement recurrence

Let `h = trunk(x)` be the convolutional feature map and
`z = pool(h) ∈ R^C` its global-mean-pool latent. The initial logit is

```
l_0 = Head_0(z)
```

For each refinement step `t = 1 .. T`,

```
c_t = α · tanh( CorrectionMLP_t( [ z, l_{t-1}, ϕ(l_{t-1}) ] ) )
l_t = l_{t-1} + c_t
```

where `α` is a fixed per-step clamp (default `0.25`, matching the packet's
suggested `c_t = 0.25 · tanh(raw_c_t)` to keep correction magnitudes
stable), and `ϕ(l)` are deterministic confidence features derived from the
previous logit. For the puzzle-binary head (`num_classes = 1`)

```
ϕ(l) = ( l, σ(l), |l|, |2σ(l) − 1|, H_b(σ(l)) )
```

with `H_b` the binary entropy. By default the same `CorrectionMLP` is
shared across steps (weight tying); setting `untie_corrections = true`
instantiates a distinct head per step, which is the `untied_corrections`
ablation called out in the packet.

The forward output is `l_T` (training default) plus the entire trajectory
`(l_0, l_1, …, l_T)` and per-step correction vectors `c_t` so the trainer
can compute correction-norm and step-flip diagnostics.
