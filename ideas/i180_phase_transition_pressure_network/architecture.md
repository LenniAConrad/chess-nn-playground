# Architecture

`Phase-Transition Pressure Network` is a board-only classifier for the
`puzzle_binary` task. It accepts the repository's `simple_18`
current-board tensor with shape `(B, 18, 8, 8)` and returns one
puzzle logit per position. Instead of measuring the magnitude of
tactical pressure, the model measures *transition curves* across a
sweep of learned thresholds: positions that sit at the edge of a
collapse should look very different across the threshold grid than
positions whose pressure is high but uniformly stable.

## Mechanism

A compact convolutional trunk turns the 18-plane board into a
per-square feature map. A `1x1` pressure head emits five learned
pressure fields per square, matching the packet's named fields:

- `attack_pressure`
- `defense_pressure`
- `escape_pressure`
- `line_block_pressure`
- `target_value_pressure`

The model then sweeps a learnable threshold grid `tau` of size
`thresholds` and a learnable scalar `temperature` (parameterised in
log-space to stay positive):

```
field_tau[i, t] = sigmoid((pressure_i - tau_t) / temperature)
```

For every `(field, threshold)` pair the model computes seven
differentiable summaries directly from `field_tau`:

- `mass` — total over the 64 squares,
- `king_zone_mass` — overlap with a 3×3-dilated king plane (own + opp),
- `largest_component` — soft-max-pool of a 3×3 local-sum approximation
  of the connected mass at each square (acts as a smooth
  "largest soft component" proxy),
- `boundary_length` — sum of absolute first differences across
  horizontal and vertical neighbours,
- `king_surplus`, `queen_surplus`, `rook_surplus` — overlap with
  3×3-dilated piece-zone masks for K, Q, R combined across both sides.

The packet's transition signal is the change of these summaries
across thresholds. The model exposes both the per-threshold summary
tensor and its first differences:

```
critical_curve_{f, s, t} = summary_{f, s, t+1} - summary_{f, s, t}
critical_pressure_score = sum |critical_curve|
```

The readout is an `MLP(LayerNorm → Linear → GELU → Dropout → Linear)`
that consumes the summary at the central threshold (the operating
point) concatenated with the full critical-curve tensor. It returns
one scalar puzzle logit. Inputs to the model are limited to the
`simple_18` board tensor; engine, verification, source, and CRTK
metadata are never used.

## Trunk and pressure head

The trunk is `depth` blocks of `Conv3x3 → BatchNorm → ReLU` from 18
input planes to `channels`. The pressure head is
`Conv1x1 → GELU → Conv1x1` mapping the trunk feature map to the five
pressure-field channels.

## Threshold sweep

The threshold grid is initialised as `linspace(-2, 2, thresholds)` and
made learnable so the model can pick the operating regime in which
transitions matter. The `temperature` parameter is also learnable and
clamped to a small positive lower bound at evaluation time. Setting
`single_threshold` collapses the grid to its central element so the
readout sees magnitude only, not transitions.

## Differentiable summaries

All summaries are computed by closed-form pooling on `field_tau`:

- `mass` and the four piece-zone surpluses are inner products of
  `field_tau` with the relevant zone mask.
- `largest_component` runs a `Conv2d(kernel=3x3, weights=ones)` to get
  a local-sum field, then takes a softmax-weighted maximum across
  squares with a fixed temperature, yielding a smooth proxy for
  "largest soft connected component" mass.
- `boundary_length` sums `|field_tau[i+1, j] - field_tau[i, j]|` and
  the analogous horizontal differences.

When `no_king_zone_features` is active the king-zone-mass and the
three piece-surplus summaries are dropped before the readout.

## Readout

```
anchor      = summaries[..., T // 2, :]        # operating-point summary
curves      = summaries[..., 1:, :] - summaries[..., :-1, :]
features    = concat(anchor, curves)
puzzle_logit = readout_mlp(features)
```

`pressure_mean_only` skips the threshold sweep entirely so the readout
sees only the per-field mean pressure (equivalent to the magnitude
baseline the packet calls out). When `num_classes > 1` the puzzle
logit is written into the last column of a zero-padded logits tensor
so the BCE-with-logits trainer contract still holds.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for
the repository `puzzle_binary` BCE-with-logits trainer. All tensors
are finite per batch:

- `logits`: `(B,)` puzzle logit (or `(B, num_classes)` when
  `num_classes > 1`).
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `pressure_fields`: `(B, 5, 8, 8)` raw learned pressure fields.
- `pressure_mean`: `(B, 5)` mean pressure per field.
- `thresholds`: `(T_eff,)` effective threshold grid.
- `temperature`: scalar effective temperature.
- `field_tau`: `(B, 5, T_eff, 8, 8)` sigmoid-thresholded fields.
- `summaries`: `(B, 5, T_eff, S_eff)` differentiable summaries.
- `critical_curves`: `(B, 5, max(T_eff - 1, 1), S_eff)` first
  differences across thresholds.
- `mass_curve`, `king_zone_mass_curve`, `largest_component_curve`,
  `boundary_length_curve`: `(B, 5, T_eff)` per-summary curves.
- `critical_pressure_score`: `(B,)` total |first-difference| energy.
- `readout_features`: `(B, R)` flattened readout input.
- `trunk_features`: `(B, channels, 8, 8)`.
- `ablation_active`, `uses_threshold_sweep`, `uses_pressure_curve`,
  `uses_king_zone_features`, `num_thresholds_effective`,
  `num_summaries_effective`: `(B,)` flags exposing the running
  ablation.

## Ablations

The packet's required ablations are exposed via `model.ablation`:

- `"none"` — main model (full threshold sweep, all summaries,
  learned grid).
- `"single_threshold"` — collapse the threshold sweep to one threshold
  so the readout cannot rely on transition curves.
- `"pressure_mean_only"` — replace the threshold sweep with the per-
  field mean pressure; tests criticality vs magnitude.
- `"no_king_zone_features"` — drop the king-zone-mass and
  king/queen/rook-surplus summaries.

## Implementation Binding

- Registered model name: `phase_transition_pressure_network`
- Source implementation file: `src/chess_nn_playground/models/phase_transition_pressure_network.py`
- Idea-local wrapper: `ideas/i180_phase_transition_pressure_network/model.py`
