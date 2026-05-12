# Ablations

Central falsifier:

- Replace the asymmetric dual dictionaries with a single shared sparse dictionary while keeping classifier capacity matched. If SRPA does not outperform this ablation on near-puzzle false-positive control and PR AUC, the asymmetry thesis is weak.

Recommended ablations:

- Remove `aux_logit` loss and keep the sparse descriptor classifier.
- Disable group thresholding while retaining atom-wise sparsity.
- Remove ray path summaries and use only source/destination/geometry.
- Halve and double `num_atom_groups` at fixed `atoms_per_group`.
- Compare `max_ray_distance: 3` against full-distance rays for speed/accuracy tradeoff.

Primary comparison models:

- LC0 BT4 single-logit classifier.
- VetoSelect positive-claim abstention.
- Dykstra VetoSelect hybrid.
