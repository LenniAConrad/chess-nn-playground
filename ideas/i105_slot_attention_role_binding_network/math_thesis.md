# Math Thesis

Slot Attention Role Binding Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md`.

Batch candidate rank: `4`.

Working thesis: Puzzle-like positions may be characterized by how occupied
pieces bind to a small number of latent tactical roles. Slot attention can
softly assign pieces to roles and expose role competition without selecting a
hard witness subset.

## Formalisation

Let `P_b` be the set of occupied squares of board `b` from the current-board
`simple_18` tensor and let `phi(p) in R^F` be the per-square feature vector
formed from the 12 piece-plane one-hot, 6 global planes, and 6 deterministic
coordinates. We pad `P_b` to a length-`32` index list and write
`m_p in {0, 1}` for the occupancy mask. A learned encoder `E: R^F -> R^d`
produces piece tokens `x_p = m_p * E(phi(p))`.

A bank of `S = 8` learnable slot prototypes `(mu_s, log sigma_s) in R^D` is
instantiated at the start of each forward pass as
`slot_s^{(0)} = mu_s + sigma_s * eps_s`, with `eps_s` drawn from a standard
Gaussian during training and held to zero at evaluation. With masked
key/value projections `K = W_k x`, `V = W_v x` and slot queries
`Q^{(t)} = W_q LN(slot^{(t)}) / sqrt(D)`, slot attention iterates for
`t = 0, ..., T - 1` (`T = 3`):

```
attn^{(t)}_{s, p} = softmax_s(Q^{(t)}_s . K_p) * m_p
W^{(t)}_{s, p}    = attn^{(t)}_{s, p} / sum_p attn^{(t)}_{s, p}
update^{(t)}_s    = sum_p W^{(t)}_{s, p} V_p
slot^{(t + 1)}_s  = GRU(update^{(t)}_s, slot^{(t)}_s)
                  + MLP(LN(GRU(update^{(t)}_s, slot^{(t)}_s)))
```

The per-iteration update residual `r^{(t)}_b = || slot^{(t + 1)} - slot^{(t)} ||_F`
quantifies how much role binding is still moving at iteration `t`.

## Diagnostics

The classifier reads the final slot bank `slot^{(T)}` together with shape
features of the final assignment `attn^{(T-1)}`:

- `slot_mass_s = sum_p attn^{(T-1)}_{s, p}` and the share
  `slot_mass_s / sum_p m_p` — measures role competition without picking a hard
  witness subset.
- `slot_self_entropy_s = -sum_p attn^{(T-1)}_{s, p} log attn^{(T-1)}_{s, p}` —
  per-slot focus.
- `per_token_entropy_p = -sum_s a_{s, p} log a_{s, p}` after renormalising
  `a_{s, p} = attn^{(T-1)}_{s, p} / sum_s attn^{(T-1)}_{s, p}` — assignment
  entropy from the piece side.
- `slot_norms_s = || slot^{(T)}_s ||`, `slot_dispersion`, and the L2 norms of
  `update^{(t)}` summarised across iterations.

## Falsifiers

The packet's central ablations remain valid: replacing slots with a
mean/max occupied-piece pool, freezing random slot prototypes, classifying
only from the entropy/mass diagnostics, or fixing slots to material roles
should each weaken the model if the iterative role-binding signal is real.
