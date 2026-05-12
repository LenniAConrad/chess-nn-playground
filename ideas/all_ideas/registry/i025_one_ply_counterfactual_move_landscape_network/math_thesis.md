# Math Thesis

One-Ply Counterfactual Move Landscape Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0429_tuesday_local_move_landscape.md`.

The thesis is that puzzle-like positions often have a sharply structured set of immediate, rule-derived counterfactual consequences. Given a current board tensor \(x\), the model extracts a board state \(b\), enumerates the side-to-move pseudo-legal move set \(\widetilde{M}(b)\), and encodes each move as a sparse counterfactual delta using source-square features, destination-square features, their difference, and typed move metadata.

For each move \(m\), the shared move encoder computes

\[
z_m = \phi_\theta(b, m, \Delta_m(b)).
\]

A learned energy head then scores the move relative to the root board embedding \(r_\theta(b)\):

\[
e_m = q_\theta(z_m, r_\theta(b)).
\]

The landscape pool computes permutation-invariant statistics:

\[
\bar z = \operatorname{mean}_{m\in\widetilde{M}(b)}z_m,\quad
v_z = \operatorname{var}_{m\in\widetilde{M}(b)}z_m,
\]

\[
p_m = \frac{\exp(e_m/\tau)}{\sum_{m'}\exp(e_{m'}/\tau)},\quad
z_{\text{attn}} = \sum_m p_m z_m.
\]

It also computes entropic landscape scalars, including

\[
\tau\log\left(\frac{1}{|\widetilde{M}(b)|}\sum_m \exp(e_m/\tau)\right)
- \operatorname{mean}_m e_m,
\]

the top-2 energy gap, normalized attention entropy, and attention peak. These features let the classifier distinguish broad ordinary move landscapes from positions where a few one-ply consequences are structurally exceptional.

The folder's benchmark contract uses one puzzle logit: fine labels `0` and `1` map to non-puzzle, and fine label `2` maps to puzzle. Fine labels, source metadata, engine scores, PVs, mate/search information, and verification artifacts are never model inputs.
