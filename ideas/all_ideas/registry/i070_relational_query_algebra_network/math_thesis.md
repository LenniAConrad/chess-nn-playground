# Math Thesis

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2139_friday_shanghai_relational_query_algebra.md`.

The thesis is that tactical puzzle positions are often recognizable as relational
join patterns over the current board: a piece attacks or blocks a square, two pieces
share a line or knight/king relation, or a middle square witnesses a pin, skewer, or
clearance pattern between endpoints. A generic CNN can learn some of these patterns,
but it has to encode the fact-table structure implicitly. This idea makes the
relational structure explicit and lets the classifier learn which joins matter.

The model builds a piece fact table \(P\), a square fact table \(S\), and a fixed bank
of square relations \(R_k(a,b)\). For each learned query \(q\), it computes:

- a piece-square join \(P_q \bowtie_R S_q\);
- a piece-piece join \(P^L_q \bowtie_R P^R_q\);
- a line-between semijoin where aligned piece pairs gather square evidence on the
  ray between their occupied squares.

The relation bank includes same-rank/file/diagonal relations, square-color relations,
king and knight offsets, distance bins, half-board, center, and edge bins. Learned
query predicates choose piece and square evidence, while learned relation mixtures
select the useful relation family. The downstream readout receives aggregate join
statistics rather than raw source metadata, preserving the puzzle-binary contract:
fine labels 0 and 1 map to non-puzzle, fine label 2 maps to puzzle.
