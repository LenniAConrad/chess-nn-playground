# Architecture

## Overview

`Hall-Defect Dual-Residual Network` builds a board-derived obligation x defender
incidence system and classifies puzzle-likeness from the trajectory of an
unrolled differentiable projected dual ascent on the relaxed defender-covering
linear program.

## Components

- Board encoder: convolutional trunk producing a pooled board summary.
- Incidence head: learns a soft `(num_obligations x num_defenders)` incidence
  matrix via a sigmoid projection.
- Demand and cost heads: produce strictly positive `demand[i]` per obligation
  and `cost[j]` per defender via softplus.
- Unrolled dual ascent: starting from `z = 0`, `lambda = 0`, runs `T` projected
  dual-ascent steps on the relaxation
  `min c^T z s.t. A z >= demand, 0 <= z <= 1`.
- Trajectory readout: per-step primal violation, current objective,
  complementarity residual, and total assignment summed across iterations,
  plus the final dual vector and final assignment.
- Classifier: pooled board features + dual-residual trajectory + final
  primal/dual norms feed an MLP head.

## Diagnostics returned by the forward pass

- `primal_violation_final`, `dual_norm_final`, `objective_final`
- `primal_norm_final`, `primal_trajectory`
- `demand_total`, `cost_total`, `hall_defect_estimate`

## Implementation Binding

- Registered model name: `hall_defect_dual_residual_network`
- Source implementation file: `src/chess_nn_playground/models/hall_dual_residual.py`
- Idea-local wrapper: `ideas/registry/i219_hall_defect_dual_residual_network/model.py`
