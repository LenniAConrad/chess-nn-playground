# Math Thesis

Evidence Sieve Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.

Batch candidate rank: `6`.

Working thesis: Instead of refining logits, the model can refine *features* by
repeatedly filtering them through learned evidence sieves. Each sieve stage
produces a soft mask over channels and squares, passes only the selected
evidence onward, and leaves a diagnostic trail.

## Setup

Let `H_0 \in R^{C \times 8 \times 8}` be the per-square channel features
produced by a compact convolutional trunk over the simple_18 board tensor.
We refine `H_0` through `T = num_sieves` sieve stages indexed by `t = 1..T`.

## Sieve stage

Each stage applies a *channel mask* `c_t \in (0, 1)^C` and a *spatial mask*
`s_t \in (0, 1)^{8 \times 8}`,

```text
c_t = sigmoid( MLP_c( [GAP(H_{t-1}); GMP(H_{t-1})] ) ),                   (1)
s_t = sigmoid( Head_s( H_{t-1} ) ),                                       (2)
```

where `GAP` and `GMP` are global average and max pooling over the 64 squares
and `Head_s` is a small `Conv3x3 -> GELU -> Conv1x1` head that emits one logit
per square. The masks act as soft selectors -- entries near `1` keep evidence
and entries near `0` filter it out.

The *selected evidence* at stage `t` is the elementwise product of the masks
with the input feature map,

```text
E_t = c_t \otimes s_t \otimes H_{t-1}
    = c_t[:, None, None] * s_t[None, :, :] * H_{t-1}.                     (3)
```

Only the sieved evidence then propagates to the next stage:

```text
H_t = GroupNorm( H_{t-1} + alpha * Conv3x3(E_t) ),                        (4)
```

with residual scale `alpha`. Crucially the residual conv consumes `E_t`, not
`H_{t-1}`, so the next stage is updated by *what survives the sieve*. This
gives the architecture its name: each stage is a learned filter, the next
stage operates on the residue, and a stack of stages refines the features
rather than the logits.

## Aggregation and head

After `T` stages the head pools the *aggregate of selected evidence across
stages* together with the final propagated trunk representation:

```text
\bar{E} = (1 / T) sum_{t=1..T} E_t,                                        (5)
z       = concat( pool(\bar{E}), pool(H_T) ),                              (6)
\hat{y} = Linear( GELU( Linear( LayerNorm(z) ) ) ),                        (7)
```

with global average pooling over the 64 squares. This guarantees the head
sees both the union of stage-wise selections (`pool(\bar{E})`) and the
information that survived the entire sieve cascade (`pool(H_T)`).

## Diagnostic trail

Every stage exposes its mask and selection statistics so downstream tooling
can inspect the sieve trail without re-running the model. For each stage we
record `c_t`, `s_t`, the per-stage *selection ratio*

```text
\rho_t = mean(c_t) * mean(s_t),                                            (8)
```

the *selected-evidence energy* `||E_t||_2^2 / (C * 64)`, and the
elementwise Bernoulli entropy of `c_t` and `s_t` (a calibrated sharpness
score: `0` for fully decisive masks, `log 2` for a uniform `1/2` mask).

## Why this is materially distinct from a probe scaffold

A `ResearchPacketProbe` wrapper would condition a generic CNN on a fixed
mechanism profile and read out classification logits. The Evidence Sieve
Network is structurally different: it builds an explicit cascade of learned
mask gates that filter the *feature map* itself, propagates only the selected
evidence between stages via a residual conv on `E_t` rather than `H_{t-1}`,
and exposes the per-stage masks/energies as part of the forward pass. The
head is a function of both the across-stage selection mean `\bar{E}` and the
final propagated trunk `H_T`, so the sieve trail is a load-bearing part of
the prediction rather than an auxiliary diagnostic.
