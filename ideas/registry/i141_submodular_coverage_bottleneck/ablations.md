# Ablations

`SubmodularCoverageBottleneckNetwork.ABLATIONS` defines the testable variants. Each ablation is selected by setting `model.ablation` in `config.yaml`.

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `none` | Full implementation. | Submodular coverage + marginal gains are useful. | Baseline against which other ablations are compared. |
| `additive_pool` | Replace `c_k = 1 - prod_i (1 - a_i W_{i,k})` with the additive sum `sum_i a_i W_{i,k}`. | Diminishing returns matter. | If equal, coverage non-linearity is unnecessary. |
| `no_marginal_gains` | Zero the top-T marginal-gain features in the head input. | Marginal structure matters beyond `F(a)` and `c`. | If equal, only covered attributes carry signal. |
| `unconstrained_W` | Drop the `softplus` nonnegativity constraint on `W`. | The submodular monotonicity constraint matters. | If better, the constraint may be too restrictive. |
| `random_concepts` | Freeze the patch CNN and line/king/material MLPs at initialization. | Learned concepts matter. | If equal, the coverage head is doing all the work. |
| `material_concepts_only` | Mask non-material concept activations to zero. | The model does not shortcut to material balance. | If strong, spatial concepts are weak. |

Each ablation surfaces `submodular_coverage_ablation` in the forward output so prediction artifacts record which variant was used.
