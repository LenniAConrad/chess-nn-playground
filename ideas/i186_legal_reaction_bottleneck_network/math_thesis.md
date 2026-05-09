# Math Thesis

Legal-Reaction Bottleneck Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.

Batch candidate rank: `1`.

Working thesis: A real puzzle is not merely a position with a threat. It is a position where normal-looking defensive reactions fail or are too few. Near-puzzles often contain pressure, but the opponent still has many valid ways to defuse it.

Operationalisation: model the defender's reaction set as a softmax `p` over opponent-piece squares (the legal-reply graph), read the *effective reaction count* `K_eff = exp(H(p))`, and route the puzzle logit through a bottleneck pool of trunk features weighted by `p`. The threat side is summarised as `own_piece_pressure`, the mean of a per-square threat sigmoid over own-piece squares. The puzzle logit is then a function of the bottleneck pool, the threat pool and explicit scalars `defense_gap = own_piece_pressure - log1p(K_eff)` and `reply_pressure = own_piece_pressure / (K_eff + 1)`. Both rise when threat is high and reactions are scarce, which is the puzzle / non-puzzle separation the packet calls for.
