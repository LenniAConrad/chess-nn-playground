# Math Thesis

Agreement-Variance Head Net.

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.

Batch candidate rank: `5`.

The puzzle decision is modelled by a shared convolutional trunk
`f_theta : R^{C x 8 x 8} -> R^{D}` followed by `K = num_heads`
independently initialised cheap heads
`g_{phi_k} : R^{D} -> R`, `k = 1, ..., K`. Let `h(x) = f_theta(x)` be
the pooled trunk embedding. The per-head logits are
`z_k(x) = g_{phi_k}(h(x))`, and the reported classification logit is
their mean:

\[
z(x) = \frac{1}{K} \sum_{k=1}^{K} z_k(x).
\]

The training objective is the standard one-logit BCE-with-logits loss
on the mean logit:

\[
L(\theta, \phi) = \mathrm{BCE}(z(x), y).
\]

Because each head receives the same target `y` and gradients flow only
through `mean(z_k)`, no explicit agreement penalty is added: the heads
are not encouraged to collapse to one another. The cross-head variance

\[
\mathrm{Var}_k z_k(x) = \frac{1}{K} \sum_{k=1}^{K} (z_k(x) - z(x))^2
\]

is computed under `torch.no_grad()` and exposed as
`head_variance`. Its square root, the disagreement, is exposed as
`head_disagreement`. Per-head probabilities and their variance are also
exposed as a calibration diagnostic.

The thesis is that mean-logit aggregation matches the marginal accuracy
of a heavier ensemble at a fraction of the cost (one trunk + `K` cheap
heads instead of `K` independent trunks), and that head variance
captures useful epistemic uncertainty for the puzzle target. The
falsification probes are:

- If `K = 1` (single-head ablation) matches the full model on
  puzzle accuracy, the multi-head averaging does not help.
- If `head_variance` does not correlate with classification error on
  held-out positions, the diagnostic is not informative.
- If the heads converge to identical functions despite independent
  init, the cheap-head structure is not exploiting trunk capacity.
