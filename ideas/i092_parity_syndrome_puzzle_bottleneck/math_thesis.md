# Math Thesis

Parity-Syndrome Puzzle Bottleneck

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2048_friday_shanghai_architecture_batch_2.md`.

Batch candidate rank: `1`.

Working thesis: puzzle-like positions produce distinctive parity (mod-2) syndrome patterns over current-board facts. The classifier is forced to read the position through a small bank of learned sparse XOR-like constraints over Bernoulli literals and is allowed to look only at how many of those constraints are satisfied or violated, not at the raw literals or board planes. This tests a mod-2 algebraic bottleneck rather than a generic CNN aggregator.

Setup. Let `l in [0, 1]^L` be a vector of bounded literal probabilities produced by a compact convolutional encoder over the `simple_18` board planes (with `L = literal_channels x 64`). Let `G in [0, 1]^{K x L}` be a sparse soft gate matrix that, for each check `k`, selects a small subset of literals through a top-`topk` mask on a low-rank score `S = (left @ right^T) / sqrt(rank)`, then squashes through `sigmoid`.

Per-check syndrome. Define the per-check syndrome `s_k` via the standard differentiable mod-2 surrogate:

`s_k = 0.5 * (1 - prod_i (1 - 2 g_{k,i} l_i))`,

with the product computed in log-space (`log_abs = sum_i log |1 - 2 g_{k,i} l_i|`, `sign_product = prod_i sign(...)`, `parity_product_k = sign_product * exp(log_abs)`) for numerical stability. For `g_{k,i} in {0, 1}` and `l_i in {0, 1}`, this exactly recovers XOR of the gated literals; for soft inputs it is the Bernoulli relaxation. Intuitively, `s_k -> 0` when an even number of active literals participate (constraint satisfied), `s_k -> 1` otherwise.

Bottleneck. Let `m_k = |s_k - 0.5|` be the per-check violation margin and `H(s_k) = -s_k log s_k - (1 - s_k) log(1 - s_k)` the per-check Bernoulli entropy. The classifier head receives only

`phi(s, G) = [s | m | top_{T}(s) | top_{T}(m) | hist(s) | hist(m) | globals]`,

where `globals` collects 10 scalars (`mean(s)`, `std(s)`, `max(s)`, `min(s)`, `mean(m)`, `max(m)`, `mean H(s)`, `min H(s)`, mean check degree, gate density). The puzzle logit is `f_theta(phi(s, G))`. The head never sees `l` or the raw board planes, so the puzzle decision must be expressible as a function of which sparse mod-2 constraints fire — this is the algebraic bottleneck under test.

Ablations (selectable through `mode`):

- `mode = sum_checks`: replace the parity surrogate by the gate-weighted mean `s_k = sum_i g_{k,i} l_i / sum_i g_{k,i}`. Isolates count statistics from XOR structure.
- `mode = random_parity_checks`: keep the parity surrogate but freeze the gate selection at a random sparse mask. Isolates learned check selection from the value of sparse mod-2 evidence.
- `mode = dense_parity_no_sparsity`: drop the top-`topk` mask. Isolates the role of sparsity in the bottleneck.

Falsifier. If puzzles are not characterised by a small number of sparse mod-2 constraints over current-board facts, the parity bottleneck should not match a count-based or unconstrained baseline at parameter parity, and the `sum_checks`, `random_parity_checks`, `dense_parity_no_sparsity` ablations should not separate from the default `parity` mode in a way that tracks the packet's predictions.
