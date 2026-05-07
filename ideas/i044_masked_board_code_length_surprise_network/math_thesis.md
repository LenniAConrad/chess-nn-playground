# Math Thesis

Masked Board Code-Length Surprise Network (`MBCS-Net`).

Source packet:
`ideas/research_packets/chess_nn_research_2026-04-21_0739_tuesday_los_angeles_masked_surprise_codec.md`.

## Working Thesis

Train a label-free masked board codec to estimate how many nats it
takes to reconstruct each hidden square from the rest of the current
board, then classify puzzle-likeness from the resulting spatial
code-length / entropy / true-token-probability fields plus the original
board tensor.

## Input Space And Tokenizer

For `simple_18` boards `x \in R^{18 x 8 x 8}` define the deterministic
tokenizer

\[
T : R^{18 x 8 x 8} -> A^{S}, \quad
A = {empty, WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK}
\]

over the 64 squares `S = {0, ..., 63}`. `T(x)_s` is extracted only from
the 12 piece planes; side-to-move, castling, and en-passant remain
visible to the codec but are never reconstruction targets. LC0
encodings (`lc0_static_112`, `lc0_bt4_112`) require an explicit
current-board piece-channel schema; otherwise the codec must fail
closed.

## Mask Bank

The fixed `2x2_residue` mask bank has four masks
`M_k = {(r, f) : r % 2 = r_k, f % 2 = f_k}, k = 1..4`, each covering 16
squares so that every square is masked by exactly one mask. This keeps
the number of masked squares constant per mask and prevents
candidate-count shortcuts.

## Codec And Code-Length Field

Train a codec
`q_theta(a | x_{\M}, M, s), a \in A`
by the label-free objective

\[
L_codec(theta) = E_{x ~ D, M ~ mu} [ (1/|M|) sum_{s in M}
    -log q_theta(T(x)_s | x_{\M}, M, s) ].
\]

Standard cross-entropy decomposition gives
`L_codec(q) - L_codec(p) = E [ KL(p || q) ] >= 0`, so the optimum codec
recovers the true conditional `p(T_s | x_{\M}, M, s)` and produces
genuine conditional code lengths.

The mask-averaged spatial fields are

\[
S_theta(s, x) = ( sum_k 1[s in M_k] * (-log q_theta(T(x)_s | x_{\M_k}, M_k, s)) )
              / ( sum_k 1[s in M_k] ),
\]

`H_theta(s, x)` -- the analogous mask-averaged predictive entropy --,
and `P_theta(s, x) = q_theta(T(x)_s | x_{\M}, M, s)` averaged over
covering masks. `S` is clipped to `surprise_clip_nats=8.0` nats and
fed through `log1p` before the classifier.

## Classifier

The supervised classifier

\[
f_psi : Phi_theta(x) -> R^{num_classes}, \qquad
Phi_theta(x) = concat(x, S_scaled, H, P, coords),
\]

is a compact residual CNN trained with binary cross-entropy on the
puzzle target

\[
y = 1[ a \in {1, 2} ], \quad a \in {0, 1, 2}.
\]

Fine label `1` (verified near-puzzle) is *not* fabricated -- it remains
a positive coarse-binary label and is reported only in the diagnostic
3 x 2 confusion matrix.

## Proposition (Conditional-Code Optimality And Sparse Surprise)

Assume the codec approximates the conditional `p(T_s | x_{\M}, M, s)`
and that the binary log odds factor through a continuous board pooling
of the spatial fields `{S(s, x), H(s, x) : s \in S}`. Then a
sufficiently expressive classifier on `Phi_theta(x)` can approximate
the Bayes decision rule, while a unigram/material-prior codec
(removing dependence on `x_{\M}`) cannot represent any signal that
specifically comes from conditional inconsistency between a square and
the rest of the board. Sketch: the cross-entropy decomposition above
shows the optimum codec gives valid conditional code lengths; the
ablation removes conditional context, so only base-rate rarity
remains.

## Falsification

The central ablations are:

1. unigram/material codec replacing the conditional codec while
   keeping the surprise interface and classifier;
2. no-surprise classifier (drop `S`, `H`, `P_true`);
3. square-shuffled surprise that preserves the multiset but breaks
   spatial alignment;
4. random frozen codec.

Treat the idea as falsified if these ablations match the main model
within the published tolerances.

## Hypothesized But Not Proven

- Puzzle and near-puzzle positions in the current CRTK split contain
  localized conditional-surprise patterns absorbed efficiently by
  this fixed mask bank.
- Three epochs of label-free codec pretraining on the standard split
  are enough to learn chess-relevant context regularity.
- The sparse-surprise factorization holds well enough that the small
  residual classifier can read out useful signal without resorting to
  global histogram shortcuts.
