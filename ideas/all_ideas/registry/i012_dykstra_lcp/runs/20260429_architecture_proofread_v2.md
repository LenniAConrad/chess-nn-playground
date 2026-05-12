# Dykstra-LCP Architecture Proofread V2

Date: 2026-04-29

Scope:

- `src/chess_nn_playground/models/dykstra_lcp.py`
- `src/chess_nn_playground/models/dykstra_vetoselect.py`
- `src/chess_nn_playground/models/vetoselect.py`
- `src/chess_nn_playground/training/losses.py`
- `ideas/all_ideas/registry/i012_dykstra_lcp/config*.yaml`

Reference idea:

- Soft-Dykstra Latent Constraint Projector research packet
- local `architecture.md`, `math_thesis.md`, and `implementation_notes.md`

Findings:

1. The VetoSelect probability factorization is consistent with the written idea. `pi_N`, `pi_R`, and `pi_P` sum to one, and `selective_puzzle_logit` is the log odds of accepted puzzle evidence against the two non-positive actions.
2. The Dykstra forward contract is correct: `forward(x)` consumes only the board tensor and returns one puzzle score plus solver diagnostics.
3. The Dykstra projector had three linear-algebra mismatches:
   - motif simplex projection clamped small positive values without renormalizing, so rows could drift above sum one;
   - role-budget projection used `m.mean(...)`, which is invariant for simplex `M`, so motif mixture did not actually define role budgets;
   - closure projection clipped target-role mass but did not use bounded slack as the measured violation channel.

Fixes:

- Renormalized the motif simplex projection after numerical clamping.
- Added learnable motif-to-role budget parameters initialized to the previous base budget, so role budgets are linear functions of `M`.
- Added motif-conditioned compactness budget parameters.
- Updated closure projection to activate slack for unexplained target-role mass before clipping residual target mass.
- Added `architecture_version: 2` and new Dykstra run names so future `--skip-existing` runs do not reuse pre-fix checkpoints.

Validation added:

- simplex projection preserves nonnegative probability mass;
- motif mixture changes role budgets when motif budget parameters differ;
- closure activates slack for unexplained target-role mass;
- existing Dykstra, Dykstra+VetoSelect, and VetoSelect output/loss tests still pass.

Status:

The code is now closer to the linear-algebra packet. Existing Dykstra result directories remain valid historical artifacts, but they are pre-v2 projector results and should not be treated as evidence for the fixed architecture. Retrain the v2/v3 configs before making new benchmark claims.
