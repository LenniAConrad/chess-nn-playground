# Architecture

`Parity-Syndrome Puzzle Bottleneck` realises the source packet's parity / syndrome bottleneck as a bespoke PyTorch model for the repository's `puzzle_binary` task. The classifier is forced to read the position through a small bank of learned sparse XOR-like checks: it never sees the raw board features at the head, only how many of those linear-mod-2 constraints are satisfied or violated.

## Implementation Binding

- Registered model name: `parity_syndrome_puzzle_bottleneck`
- Source implementation file: `src/chess_nn_playground/models/trunk/parity_syndrome.py`
- Idea-local wrapper: `ideas/registry/i092_parity_syndrome_puzzle_bottleneck/model.py`

## Modules

`LiteralEncoder` is a compact `Conv2d` stack over the configured `simple_18` board planes. Stage one is a `3x3` lift to `hidden_dim` channels with `GroupNorm` + `GELU`; further stages repeat the same pattern with optional `Dropout2d`. A final `1x1` convolution maps to `literal_channels` planes that are squashed by `sigmoid` to bounded literal probabilities `l in [0, 1]^{literal_channels x 8 x 8}`. Flattening yields `num_literals = literal_channels x 64` literals per board, the symbols on which the parity-check bank operates.

`ParityCheckBank` is the syndrome layer. It owns:

- a low-rank score matrix `S = (left @ right^T) / sqrt(rank)` of shape `[num_checks, num_literals]`,
- a sparse top-k mask `M_topk` selecting the `topk` literals with the highest score per check (the sparse XOR-like constraint set),
- and a frozen random check bank `random_gates` of the same sparsity pattern, used by the `random_parity_checks` ablation mode.

The active gates are `G = sigmoid(S) * M_topk` (or `sigmoid(S)` when `mode == "dense_parity_no_sparsity"`, or the frozen random mask when `mode == "random_parity_checks"`). Each row `G_k` defines a soft sparse XOR constraint over literals.

The forward path is mode-dependent:

- `mode == "parity"` (default): compute the differentiable mod-2 parity surrogate. For literal probabilities `l_i in [0, 1]` and gates `g_{k,i} in [0, 1]`, define per-check signed factors `f_{k,i} = 1 - 2 g_{k,i} l_i` clamped into `(-0.999, 0.999)`. The signed product `prod_i f_{k,i}` is computed in log-space via `log_abs = sum_i log |f_{k,i}|` and `sign_product = prod_i sign(f_{k,i})`, recombined as `parity_product_k = sign_product * exp(log_abs)`. The syndrome is `s_k = 0.5 * (1 - parity_product_k)`, which is the standard Bernoulli relaxation of the XOR of the gated literals: `s_k -> 0` when the constraint is satisfied (even number of active literals) and `s_k -> 1` when violated.
- `mode == "sum_checks"`: the deterministic sum-checks ablation. Each syndrome is the gate-weighted mean of literals, `s_k = sum_i g_{k,i} l_i / sum_i g_{k,i}`, clipped to `[0, 1]`. This isolates count statistics from XOR structure.
- `mode == "random_parity_checks"`: same parity surrogate as the default mode but with the frozen random gates, isolating learned check selection from sparse mod-2 evidence.
- `mode == "dense_parity_no_sparsity"`: removes the top-k sparsity mask, isolating the role of sparsity in the bottleneck.

`SyndromeStats` is the diagnostics + readout layer that converts the per-check syndrome vector `s in [0, 1]^{num_checks}` into the head's input features. It computes:

- the raw syndromes `s` and the violation margins `m = |s - 0.5|`,
- the top-`top_values` syndromes and margins,
- soft histograms over `s` (`syndrome_histogram`, 16 bins) and `m` (`margin_histogram`, 16 bins) using a Gaussian kernel,
- ten global scalars: `mean(s)`, `std(s)`, `max(s)`, `min(s)`, `mean(m)`, `max(m)`, mean and min of the per-check entropy `H(s_k) = -s_k log s_k - (1-s_k) log(1-s_k)`, and two structural scalars (mean check degree and overall gate density).

The full feature vector `phi(s, G) = [s | m | top_s | top_m | hist_s | hist_m | globals]` is the only thing that reaches the classifier head, enforcing the algebraic bottleneck.

`ParitySyndromePuzzleBottleneck` glues the trunk together: `LiteralEncoder` -> flatten -> `ParityCheckBank` -> `SyndromeStats` -> a `LayerNorm + Linear(GELU) + Dropout + Linear(GELU) + Linear(1)` head. The head returns one logit per board; the rest of the diagnostics surface as named tensors on the output dict.

## Diagnostics

`forward(x, *, return_diag=False)` returns a dict containing:

- `logits`: shape `(B,)`, BCE-compatible for the one-logit `puzzle_binary` head.
- `prob`: sigmoid of the puzzle logit.
- `syndromes`: per-check syndrome activations `s in [0, 1]^{num_checks}`.
- `syndrome_features`: the full pooled feature vector `phi(s, G)`.
- `literal_mean`, `literal_entropy`: structural sanity diagnostics for the literal layer.
- `parity_check_mode`: integer code identifying the active syndrome mode (parity / sum_checks / random_parity_checks / dense_parity_no_sparsity).
- `mechanism_energy`: `mean_k s_k^2` — the syndrome energy that operationalises the packet's "linear_algebra" mechanism family.
- `proposal_profile_strength`: `max_k s_k` — the strongest single check violation.
- `proposal_keyword_count`: integer scalar preserved for compatibility with the project's research-packet diagnostic schema.
- `syndrome_mean`, `syndrome_std`, `syndrome_max`: scalars summarising the syndrome distribution.
- `syndrome_margin_mean`, `syndrome_margin_max`: scalars summarising violation magnitude `m_k = |s_k - 0.5|`.
- `syndrome_entropy`: mean per-check Bernoulli entropy of the syndromes.
- `check_degree_mean`, `check_gate_density`: structural sanity diagnostics for the parity-check bank.
- `top_syndrome_values`, `top_syndrome_margins`: the strongest `top_values` syndromes and margins.
- `syndrome_histogram`, `margin_histogram`: soft 16-bin histograms over `s` and `m`.

When `return_diag=True` the dict additionally contains `literal_probs` (shape `(B, literal_channels, 8, 8)`) and `check_gates` (shape `(num_checks, num_literals)`), used by ablation harnesses.

## Contract

- Input: `(B, C, 8, 8)` board tensor only. CRTK / verification / source / engine metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit `puzzle_binary` BCE-with-logits trainer, plus the diagnostics listed above.
- Target mapping: fine labels `0` and `1` map to binary target `0`; fine label `2` maps to binary target `1`.
- Model shapes: literal grid `[B, literal_channels, 8, 8]`, flattened literals `[B, num_literals]` with `num_literals = literal_channels * 64`, gates `[num_checks, num_literals]`, syndromes `[B, num_checks]`.
- The puzzle decision flows only through `phi(s, G)` — the head never sees raw literals or board planes — so the algebraic bottleneck is enforced architecturally.
- `mode` selects the active variant: `parity` (default, learned sparse XOR-like checks), `sum_checks` (count-only ablation), `random_parity_checks` (frozen random checks), `dense_parity_no_sparsity` (no top-k sparsity).
