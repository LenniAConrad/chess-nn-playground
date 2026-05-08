# Math Thesis

Tensor-Ring Square Interaction Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md`.

Batch candidate rank: `1`.

## Working thesis

Many chess cues depend on interactions among several squares at once:
king square, attacking piece, blocker, defender, escape square,
promotion path. The full square-tuple interaction tensor over 64 squares
is too large to materialise. A *tensor-ring* factorisation can capture
high-order interactions with a controlled parameter budget by replacing
explicit tuple enumeration with a cyclic trace of low-rank cores.

## Core object

Each square ``s in {0, ..., 63}`` carries an embedding ``x_s in R^D``
built from the ``simple_18`` planes plus a small coordinate embedding.
The model also holds a learned per-square role-gate bank
``g_r(s) in [0, 1]`` with ``R = 5`` roles (own piece, opponent piece,
king zone, ray-relevant square, empty square) and, for each interaction
order ``K``, a stack of ``K`` low-rank cores

::

    G_k : R^D -> R^{rank x rank}

implemented as linear maps. A learned bank of ``num_patterns`` role
sequences provides, for every pattern ``p`` and every slot ``k``, a
softmax mixture ``alpha_{p, k, r}`` over the ``R`` role gates. The
cyclic contraction of pattern ``p`` is

::

    M_{p, k} = (1 / 64) * sum_s alpha_{p, k, r} g_r(s) * G_k(x_s)
    z_{p}    = trace(M_{p, 1} M_{p, 2} ... M_{p, K})

Tuples of squares are never enumerated; the per-pattern contraction
costs ``O(64 * num_patterns * K * rank^2)``.

## Why the trace of cores captures interactions

Expanding the trace recovers a sum over square tuples,

::

    z_{p} = sum_{s_1, ..., s_K} (prod_k g_{r_k}(s_k))
            * trace(G_1(x_{s_1}) ... G_K(x_{s_K})),

so the cyclic core product is exactly a learned, low-rank
approximation of the order-``K`` square interaction tensor. The role
gates carve the contraction into interpretable subsets (king zone,
own pieces, ...) without enumerating tuples.

## Outputs

Per-order summary statistics ``mean``, ``max``, ``variance`` and
``signed_abs_mean`` are taken across the ``num_patterns`` contractions,
concatenated with the raw pattern responses and a small CNN-stem
summary, and fed to a single-logit puzzle-binary classifier head.
Diagnostic outputs expose the per-order contractions, per-pattern role
softmax, per-order core Frobenius norms, and per-square role gate
activity and entropy.
