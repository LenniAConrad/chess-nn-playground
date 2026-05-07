# Ablations

The `model.ablation` config field selects between the section 9
falsifiers listed below. All preserve the head input dimensionality so
capacity is matched.

| Ablation | What changes | Hypothesis it falsifies |
|---|---|---|
| `none` | Main bespoke model. | Cross-subspace principal-angle geometry carries puzzle-likeness signal. |
| `no_cross_angles` | Replace per-pair cosine / angle / summary features with zeros (eigenvalues, gate masses, and globals are kept). | Removing cross-role spectra collapses the signal: matches the markdown's central falsifier. |
| `batch_shuffled_angles` | Shuffle the per-sample principal-angle feature block across the batch. | Angle spectra are sample-specific evidence rather than a constant prior. |
| `eigenvalues_only` | Same removal as `no_cross_angles`; named separately so report tooling that references this label finds a working config. | Within-role variance alone is sufficient. |
| `pooled_token_head` | Bypass the subspace machinery; use mean / max / std token pooling projected to the same head input dimensionality through a learned linear layer. | Subspace geometry beats ordinary set pooling at matched capacity. |
| `no_orthonormalization` | Replace `Q_a^T Q_b` with the rank-1 outer product of unit-normalized role means, so basis-rotation invariance is destroyed. | Orthonormalization (Grassmannian invariance) is what carries the signal. |
