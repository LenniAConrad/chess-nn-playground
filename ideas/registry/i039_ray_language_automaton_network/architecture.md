# Architecture

`Ray-Language Automaton Network` (`RLAN`) is a board-only `puzzle_binary`
classifier whose central operator is a family of differentiable
weighted finite automata (WFAs) over chess-ray piece-token strings. The
implementation replaces the shared research-packet probe with a
materially distinct bespoke model so the markdown thesis is exercised
by trainable code rather than a generic mechanism profile.

## Forward Pipeline

1. **Side-relative tokenizer.** A deterministic parser converts the
   `simple_18` board tensor `(B, 18, 8, 8)` into per-square tokens over
   the 14-symbol alphabet `{empty, friend_{P,N,B,R,Q,K},
   enemy_{P,N,B,R,Q,K}, pad}`. Side-to-move comes from plane 12 and is
   used to flip white/black piece planes into friend/enemy planes.
2. **Oriented ray gathering.** A registered buffer enumerates the
   oriented rank, file, diagonal, and anti-diagonal rays (each line in
   both directions, lengths >= 2) and produces `(ray_indices, ray_mask,
   axis_ids, ray_context)`. Padded slots receive the explicit `pad`
   token id; the ray mask gates the recurrence.
3. **Weighted finite automata.** `R` learned automata with `Q` states
   each are evaluated in the log semiring. Per ray and step, the
   recurrence is
   `h_t[j] = logsumexp_i(h_{t-1}[i] + T_{r,a_t}[i,j])`, with start
   weights `alpha`, symbol-conditioned transition tensors `T`, and
   final weights `omega`. Padded steps short-circuit by re-using the
   previous hidden state.
4. **Context bias.** Deterministic ray context (axis one-hot, length,
   forward orientation, edge flags) is projected to a per-(ray,
   automaton) bias and added to the accept score.
5. **Pooling.** Per-automaton scores are summarized with global max,
   global log-sum-exp, and per-axis max / log-sum-exp.
6. **Classifier.** Pooled features are concatenated with safe board
   metadata (side-to-move, four castling channels, en-passant
   indicator) and passed through a small MLP to produce one puzzle
   logit.

## Output Contract

Forward returns a `dict` whose `"logits"` entry has shape `(B,)` so the
shared `puzzle_binary` BCE-with-logits trainer can consume it
directly. Diagnostics include `ray_scores`, `ray_language_energy`,
`ray_score_logsumexp`, `ray_automaton_diversity`, and `ray_axis_max`.
All diagnostic tensors are finite by construction.

## Implementation Binding

- Registered model name: `ray_language_automaton_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/ray_language_automaton_network.py`
- Idea-local wrapper: `ideas/registry/i039_ray_language_automaton_network/model.py`
