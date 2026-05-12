# Math Thesis

Rank-File Memory Grid Net

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.

Batch candidate rank: `8`.

Working thesis: Maintain learned memory vectors for each rank and each
file. Squares write into their rank/file memories, then rank/file memories
write back to squares. This gives global rank/file communication without
axial convolutions, line solves, or attention.

The implemented bespoke architecture realises this thesis as a stack of
`depth` memory blocks. Each block

1. projects every square `x_{b, h, w} \in R^C` to a memory token
   `w_{b, h, w} = W_write x_{b, h, w} \in R^M`,
2. aggregates the tokens into 8 rank memories
   `m_rank_{b, h} = LayerNorm( mean_w w_{b, h, w} + p_rank_h )` and 8 file
   memories `m_file_{b, w} = LayerNorm( mean_h w_{b, h, w} + p_file_w )`,
   where the per-rank `p_rank \in R^{8 \times M}` and per-file
   `p_file \in R^{8 \times M}` parameters are the explicit "learned memory
   vectors" named by the thesis,
3. broadcasts the rank/file memories back to every square that lies on
   them and reads `r_{b, h, w} = W_read [ m_rank_{b, h} ; m_file_{b, w} ]`
   into a residual update.

There are no convolutions inside the memory block, no axial line solves,
and no attention. The only cross-square communication is the rank-file
write / read, exactly as the thesis prescribes.
