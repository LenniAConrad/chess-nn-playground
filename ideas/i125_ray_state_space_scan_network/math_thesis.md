# Math Thesis

Ray State-Space Scan Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md`.

Batch candidate rank: `1`.

Working thesis: Chess line motifs (pins, skewers, batteries, x-rays, discovered checks, blocked files, mating lanes) often demand long-range context along chess rays. All-square attention or dynamic attack graphs are not the only way to obtain that context: a linear state-space recurrence applied to every rank, file, diagonal, and anti-diagonal can spread evidence along an entire line in O(L) operations while sharing parameters across rays of the same type.

Formally, for each line of length `L_r` along type `r in {rank, file, diagonal, anti-diagonal}` with token sequence `u_1, ..., u_{L_r}` (square-token features), the model maintains a hidden state `h_t in R^d` and an output `y_t` driven by line-type-conditioned matrices `(A_r, B_r, C_r, D_r)`:

- `h_t = A_r h_{t-1} + B_r u_t + b^h_r`
- `y_t = C_r h_t + D_r u_t + b^y_r`

`A_r` is constrained to a contractive regime (`A_r = 0.75 * tanh(W_r)` with diagonal prior 0.7) so iterated multiplication along an 8-square ray remains stable without explicit eigenvalue projection. The same parameters are applied in reverse to give every square both a forward and a backward line memory; padded entries preserve the previous hidden state so short corner rays do not corrupt aggregate pools. Mean / max / endpoint / per-line-type response statistics over the bidirectional outputs feed a compact classifier head.

The expectation is that this representation captures ray-aligned tactics (pins, batteries, blocked files, mating lines) more directly than generic CNN pooling while remaining cheap (O(38 * 8) sequential steps with shared weights) and fully differentiable. The architecture is board-only: CRTK and source metadata are reporting-only.
