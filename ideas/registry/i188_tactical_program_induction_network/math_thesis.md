# Math Thesis

Tactical Program Induction Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.

Batch candidate rank: `3`.

Working thesis: A puzzle is the existence of a tiny *latent program*
on the side-to-move's pieces and squares. Concretely, a sequence of
typed operations

```
(o_k, s_k, t_k)_{k=1..K}, with o_k in OPS, s_k, t_k in {1,...,64}
```

over the operation alphabet

```
OPS = {threaten, pin, deflect, overload, fork,
       clear_line, trap_king, win_target}
```

is a puzzle program when each step is supported by a typed relation
(i.e. the chosen operation `o_k` actually holds between the source
square `s_k` and the target square `t_k` on the current board) and
when the chosen sequence is coherent under preconditions and
postconditions:

```
puzzle(board) = exists (o_k, s_k, t_k)_{k=1..K} such that
                relation(o_k, s_k, t_k | board) is high,
                pre(o_k, s_k, t_k | state_{k-1}) is high,
                post(o_k, s_k, t_k | state_k) is high,
                state_k = exec(state_{k-1}, o_k, s_k, t_k).
```

## Differentiable Program Induction

We relax the discrete program into a soft program by replacing each
step's discrete choice with categorical distributions over its
arguments:

```
p_k(o)        in Delta^{|OPS|},
p_k(s | own)  in Delta^{64},
p_k(t | enemy or zone)  in Delta^{64}.
```

Source and target distributions are biased toward own pieces and
toward enemy pieces / king-zone squares respectively, so the
program can only act on real participants in the position. The
typed relation tensor

```
R[o, s, t] in [0, 1]
```

is the closed-form board-only evidence that operation `o` is
supported between squares `s` and `t`. The expected typed relation
under the soft step is

```
E[R | step k] = sum_{o, s, t} p_k(o) p_k(s) p_k(t) R[o, s, t]
              = sum_o p_k(o) * (sum_{s, t} p_k(s) p_k(t) R[o, s, t]).
```

The per-step **relation score**, **precondition score** and
**postcondition score**

```
r_k = E[R | step k],
pre_k = sigmoid(MLP_pre([state_{k-1}, source_ctx_k, target_ctx_k, op_ctx_k, ...])),
post_k = sigmoid(MLP_post([state_k, source_ctx_k, target_ctx_k, op_ctx_k, ...])),
```

combine into a **program log-coherence**

```
log L = mean_k [ log r_k + log pre_k + log post_k ].
```

`exp(log L) = L` is the program's coherence: how strongly the soft
program is jointly supported by typed evidence, by step
preconditions, and by step postconditions.

## What This Buys

- **Existential structure.** The puzzle question becomes *does any
  short program coherently fire* on this board, instead of *does a
  pooled feature crosses a threshold*.
- **Typed grounding.** Every step is forced through `R[o, s, t]`, a
  fixed board-only relation tensor. The model cannot label a step
  `pin` unless the typed fact tensor actually supports a pin on the
  chosen squares -- the `random_op_labels` ablation breaks exactly
  this binding to confirm the head depends on it.
- **Sequential structure.** Without the GRU state update and the
  ordered step embeddings (the `bag_of_ops_no_order` ablation) the
  program loses its ordered structure; without `pre_k` (the
  `no_precondition_scores` ablation) the program loses the ability
  to reject steps with missing preconditions; with only one step
  (`one_step_program`) the program loses depth. Each ablation
  isolates one structural ingredient.
- **Calibration.** The mean operation entropy, the per-step
  pre/post/relation scores, and the operation histogram all expose
  *how* the program voted for the puzzle, not just *whether* the
  logit was high.

The puzzle binary classifier is then a thin head over the program's
coherence summary, the operation histogram, the typed fact summary,
and pooled square tokens, exactly as described in `architecture.md`.
