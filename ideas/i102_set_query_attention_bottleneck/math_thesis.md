# Math Thesis

Set-Query Attention Bottleneck

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md`.

Batch candidate rank: `1`.

Working thesis: Puzzle-like positions may be recognized by a small number of latent tactical questions, each expressed as an attention distribution over board tokens. The model should classify not from unconstrained token mixing, but from query attention statistics, attend...

The implemented classifier follows the source packet fingerprint:

`current-board square tokens + learned query bank + query-to-token attention maps + attention statistics + binary puzzle-likeness head`.

The bottleneck is the point of the architecture. Square tokens are not processed
by a token-to-token self-attention stack. A fixed number of learned queries read
from the board once, producing attended values and diagnostic attention
statistics. The classifier can therefore be falsified by uniform attention,
frozen random queries, value-only, diagnostics-only, and mean-pooling ablations.
