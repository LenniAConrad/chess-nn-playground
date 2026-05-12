# Math Thesis

`King-Zone Evidence Ledger` claims that puzzles are usually decided by
king safety or forcing geometry, so a useful inductive bias is a small
bank of *learned evidence ledger slots* anchored to each king.

Source packet:
`ideas/research/packets/classic/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md`,
batch candidate rank `5`.

## Setup

For a board state `s` we extract a per-square feature map
`F(s) ∈ R^{C × 8 × 8}` from a compact convolutional trunk over the
`simple_18` planes. From `s` we also recover the two king squares
`k_W, k_B` (planes 5 and 11) and the side-to-move flag (plane 12).
Re-keying to the player to move gives an *own* king `k_o` and an
*opponent* king `k_x`.

For a king square `k` and any board square `q ∈ {0..7}^2` we form
king-relative features
```
ϕ(q | k) = (
    (q.rank - k.rank) / 7,
    (q.file - k.file) / 7,
    chebyshev(q, k) / 7,
    manhattan(q, k) / 14,
    1[chebyshev(q, k) ≤ r]
)
```
where `r` is the king-ring radius (default `2`). The augmented per-
square feature for a king-anchored ledger is
`F̂(s, k) = concat(F(s), ϕ(· | k))`.

## Ledger banks

We instantiate three banks of slots, each of shape `K × D`:
```
own_king_slots,  opp_king_slots,  global_slots ∈ R^{K × D}
```
The slots are learned parameters initialised independently per bank.
Each bank applies the packet's update rule
```
slot_k ← slot_k + gated_pool_k(F̂(s, k_anchor)).
```

We realise `gated_pool_k` as slot-conditioned attention. Slot `k` of
bank `B` produces a query `q_k = W_q^B slot_k`, scores every square
`n` by `s_{k,n} = ⟨q_k, f_n⟩ / √F`, applies a softmax over `n` to get
attention `α_{k,n}`, pools the values `v_n = W_v^B f_n` into
`p_k = Σ_n α_{k,n} v_n`, and gates the residual update by a per-slot
sigmoid `g_k = σ(W_g^B slot_k)`:
```
slot_k ← LayerNorm(slot_k + g_k ⊙ p_k).
```
This is repeated for `L` ledger layers so each bank accumulates
evidence rather than overwriting it. The own/opp banks consume
`F̂(s, k_o)` and `F̂(s, k_x)` respectively; the global bank consumes
`F̂(s, ·)` with the king-relative features set to zero.

## Readout

Let `S_o, S_x, S_g ∈ R^{K × D}` be the final slot tensors. The puzzle
logit is
```
ŷ = MLP([
    flat(S_o),
    flat(S_x),
    flat(S_o − S_x),
    flat(S_o ⊙ S_x),
    flat(S_g),
])
```
which matches the packet's required readout
`[own_king_ledger, opp_king_ledger, ledger_difference, ledger_product,
global_board_pool]` after a `LayerNorm → Linear → GELU → Linear` head.

## Why the ledger should help

A pure CNN has no explicit handle on *which king is in danger* or on
the symmetry between the own-king and opponent-king ledgers. The
`(S_o − S_x)` and `(S_o ⊙ S_x)` channels in the readout make that
symmetry first-class, and the gated update makes the bank a soft
bottleneck of width `K · D` rather than an unconstrained pooled
descriptor — close to the `BT4`-vs-bottleneck tradeoff the packet
predicts.

## Required ablations

| Ablation | Effect |
| --- | --- |
| `no_king_relative` | `ϕ(· | k)` is dropped, so the banks see only `F(s)`. Tests king anchoring. |
| `random_king_anchor` | `k_o, k_x` are replaced by deterministic per-batch random squares. Tests real king semantics. |
| `global_slots_only` | Only `S_g` is fed to the head. Tests king-specific ledger value. |
| `slot_count_sweep` | A no-op flag for runs that sweep `num_slots`. Detects bottleneck vs capacity. |

The bespoke implementation lives in
`src/chess_nn_playground/models/trunk/king_zone_evidence_ledger.py`
(`KingZoneEvidenceLedger`); the idea-local `model.py` wraps the
registered builder.
