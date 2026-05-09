# Ablations

The bespoke architecture exposes the source packet's required ablations
through `config.yaml` knobs and structurally identical baselines:

- `weight_2_only`: zero out the weight-3 / weight-4 portion of the
  Magnus feature vector by setting `bch_truncation_degree: 2`.
- `weight_3_only`: drop weight-4 contributions by setting
  `bch_truncation_degree: 3`.
- `norms_only`: hide the BCH log itself by zeroing the corresponding
  feature column at evaluation (`bch_log_F`).
- `swap_AB`: swap the attacker/defender heads to test asymmetry of the
  Hall basis.
- `commutator_random_replace`: replace `c_3a` with `[A, P c_2 P^T]` for
  random orthogonal `P` and confirm structure (not just norm) drives
  the lift.
- `random_geometry_M`: randomize the convolutional trunk weights to
  test that the Magnus structure (not learned chess geometry) is
  responsible for the lift.
- `cnn_same_params`: a matched CNN baseline with no Magnus head.
- `i040_commutator_baseline`: run the i040 single-commutator
  bottleneck on the same `(A, B)` pair.

Each ablation is reported on the canonical 3x2 puzzle_binary split with
extra slice attention to `crtk_difficulty`, since the central claim is
that the lift concentrates on harder / multi-step puzzles.
