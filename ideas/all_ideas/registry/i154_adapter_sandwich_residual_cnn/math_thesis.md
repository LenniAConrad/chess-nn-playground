# Math Thesis

Adapter-Sandwich Residual CNN

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.

Batch candidate rank: `6`.

Working thesis: Instead of building a much larger new backbone, insert small bottleneck adapters before and after ordinary residual blocks. Concretely, a stage is

  `x_out = (I + A_post) ∘ R ∘ (I + A_pre)(x_in)`

where `R` is a conventional residual block and `A_pre`, `A_post` are
Houlsby-style 1×1-conv bottlenecks `A(x) = W_up · GELU(W_down · x)` with
`W_down ∈ R^{adapter_dim × channels}`, `W_up ∈ R^{channels × adapter_dim}`,
and `adapter_dim ≪ channels`.

`W_up` is zero-initialised, so at step 0 every adapter is the identity
map and the network is behaviourally a plain residual CNN. The adapters
introduce a low-rank, locally additive perturbation around each block:
their contribution lives in a `2 · depth · adapter_dim · channels`
parameter slack that is small relative to the surrounding `O(depth ·
channels²)` cost of the 3×3 residual blocks. Whether this slack
improves the puzzle_binary contract is the empirical question this idea
tests.

The diagnostics (`pre_adapter_energy`, `post_adapter_energy`,
`adapter_energy`, plus their per-stage decomposition) measure the L2
norm of the adapter deltas relative to the pre-adapter input on each
forward pass. They are reported but detached from the loss so the
trainer does not implicitly minimise or maximise adapter activity.
