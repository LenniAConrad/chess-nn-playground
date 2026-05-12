# Math Thesis

Kernel Mean Prototype Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md`.

Batch candidate rank: `1`.

Working thesis: Puzzle-like positions may be separable by the distribution of
occupied piece tokens in a learned kernel feature space.  Instead of attending
to pieces or computing pairwise transport, KMPN embeds the occupied-piece set
as a kernel mean and compares it to a bank of learnable prototype embeddings.

Concretely, each occupied square ``i`` is tokenised with a piece-and-geometry
feature vector ``x_i`` and lifted into a kernel feature space through a learnable
random-Fourier-style map ``phi: R^{token_dim} -> R^{phi_dim}``,

    phi(t) = sqrt(2 / m) * cos(W t + b),

where ``W`` and ``b`` are trainable parameters with random initialisation.  The
empirical kernel mean is

    mu(x) = (1 / N(x)) * sum_{i in occupied(x)} phi(x_i),

i.e. the single ``R^{phi_dim}`` vector that the bottleneck exposes about the
piece set.  ``P`` learnable prototype embeddings ``mu_p`` live in the same
feature space and the puzzle classifier reads ``mu(x)`` together with the
squared MMD-style distances ``d_p = ||mu(x) - mu_p||^2`` and the corresponding
RBF similarities ``s_p = exp(-gamma_p * d_p)``, plus a small set of
set-cardinality diagnostics (log-N, side-canonical per-piece-type counts,
us-vs-them imbalance, ``||mu(x)||^2``).

Two boards that share the same empirical kernel mean over occupied pieces
produce the same forward pass: the model is permutation-invariant by
construction over the occupied set and pieces interact only through their
contribution to the kernel mean.  This is the precise sense in which KMPN
trades attention and transport for a kernel-mean readout against learned
prototypes.
