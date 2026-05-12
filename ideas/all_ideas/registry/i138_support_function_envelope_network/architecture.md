# Architecture

`Support-Function Envelope Network` summarises a chess position via convex
support-function envelopes of differentiable nonnegative fields along fixed
chess-relevant directions.

- Mechanism family: `convex`.
- Input: board tensor only (`simple_18`); CRTK / source metadata is reporting-only.
- Board trunk: compact convolutional square encoder (`BoardConvStem`) over the
  configured input planes.
- Field head: a `1x1` projection followed by `softplus` produces `n_fields`
  nonnegative learned fields `rho_c(s)` over the 64 squares.
- Auxiliary fields: own/opponent piece-type planes (six per side) recovered
  from `simple_18` via the side-to-move flip, giving 12 deterministic piece
  fields concatenated with the learned ones.
- Soft support function: for each field `c` and fixed unit direction `u`,

  ```
  h_c(u) = tau * logsumexp_s ( (<u, coord_s> + log(eps + rho_c(s))) / tau )
  ```

  with antipodally paired directions, so width `w_c(u) = h_c(u) + h_c(-u)`
  and center `m_c(u) = h_c(u) - h_c(-u)` are obtained by an antipode lookup.
- Direction set: 16 chess-relevant unit vectors covering rank, file, both
  diagonals, and the four knight slopes (each plus its negative).
- Field statistics: per field we also extract total mass, normalised entropy,
  and maximum activation.
- Own / opponent contrast: for each learned own/opp pair and for each piece
  type we compute `|m_own - m_opp|` (overlap gap) and `w_own / (eps + w_opp)`
  (width ratio) along the eight primary directions.
- Head: a small two-layer MLP over the concatenated envelope and contrast
  descriptors emits the puzzle logit. Diagnostic outputs include the support
  values, widths, centers, masses, entropies, learned fields, contrast
  features, and the active directions.

## Implementation Binding

- Registered model name: `support_function_envelope_network`.
- Source implementation: `src/chess_nn_playground/models/support_function_envelope_network.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i138_support_function_envelope_network/model.py`.

The wrapper calls
`build_support_function_envelope_network_from_config` to instantiate the
bespoke `SupportFunctionEnvelopeNetwork` `nn.Module`. Registry build via
`build_model("support_function_envelope_network", ...)` returns the same
class.

## Ablations

The packet's central ablations are exposed via `model.ablation`:

| Name | Effect |
|------|--------|
| `none` | Full architecture above. |
| `mean_pool_fields` | Replace soft support descriptors with mean / max pools. |
| `random_directions` | Use a frozen random direction set with the same count. |
| `no_opponent_contrast` | Drop own/opp gap and width-ratio features. |
| `hard_max_support` | Replace `tau * logsumexp` with a per-square `max`. |
| `counts_plus_envelope_only` | Skip learned fields, use piece-count maps only. |
