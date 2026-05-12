# Architecture

Input is `(B, C, 8, 8)`, with the first run using `lc0_bt4_112`.

Forward pass:

1. LC0 BT4-style residual trunk maps the board tensor to `(B, channels, 8, 8)`.
2. Heads produce latent variables:
   - role mass `U0`: `(B, R, 64)`;
   - relation mass `V0`: `(B, A, 64, 64)`, generated from endpoint factors and geometry masks;
   - motif mixture `M0`: `(B, K)`;
   - slack `S0`: `(B, G)`.
3. `SoftDykstraProjector` runs a fixed number of projection cycles with correction buffers.
4. The readout receives the trunk embedding, summaries of `U0/V0/M0/S0`, summaries of projected `U/V/M/S`, and scalar trace diagnostics.
5. Output is one binary puzzle logit plus scalar diagnostics saved into prediction artifacts.

The first config uses `R=8`, `A=4`, `K=8`, `G=6`, and `T=4` cycles. This is a compact first test rather than the full 8-12 cycle research-packet target.

Projector v2 keeps the same tensor contract but tightens the linear-algebra constraints:

- motif projection is renormalized onto the simplex after numerical clamping;
- role budgets are learned motif-conditioned linear functions of `M`;
- compactness is motif-conditioned instead of tied only to the first motif;
- closure projection activates bounded slack when target-role mass is not explained by relation pressure.

## Implementation Binding

- Registered model name: `dykstra_lcp`.
- Source implementation: `src/chess_nn_playground/models/dykstra_lcp.py`.
- Idea-local wrapper: `ideas/registry/i012_dykstra_lcp/model.py`.
