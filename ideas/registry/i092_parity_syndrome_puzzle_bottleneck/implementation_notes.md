# Implementation Notes

- Central code: `src/chess_nn_playground/models/parity_syndrome.py`.
- Idea-local wrapper: `ideas/registry/i092_parity_syndrome_puzzle_bottleneck/model.py`.
- Registry key: `parity_syndrome_puzzle_bottleneck`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2048_friday_shanghai_architecture_batch_2.md`.
- Batch candidate: `Parity-Syndrome Puzzle Bottleneck`.
- Bespoke implementation: `LiteralEncoder` (compact CNN over `simple_18`) -> `ParityCheckBank` (low-rank sparse top-k gates over flattened literals, differentiable mod-2 syndrome surrogate) -> `SyndromeStats` (raw + top-k + soft-histogram + global-scalar pooling) -> MLP classifier head returning one puzzle logit.
- The classifier head only sees pooled syndrome statistics `phi(s, G)`; raw literals and board planes never reach it. This enforces the algebraic bottleneck described in `math_thesis.md`.
- `mode` selects the active variant: `parity` (default learned sparse mod-2 checks), `sum_checks` (count-only ablation), `random_parity_checks` (frozen random gates), `dense_parity_no_sparsity` (no top-k sparsity).
- Board-only by design. Engine, verification, source, and CRTK metadata are reporting-only and never consumed as input.
