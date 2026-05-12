# Ablations

- `model.ablation: no_joins` keeps the CNN/material readout while zeroing all learned
  relational joins.
- `model.ablation: relation_shuffle` keeps relation density but permutes square
  identities inside each fixed relation.
- `model.ablation: piece_pair_only` keeps the piece-piece join and removes square joins.
- `model.ablation: no_semijoin` removes the line-between witness path.
- `model.ablation: static_relation_mix_only` freezes query-conditioned relation gates
  to the learned static mixtures.
- `model.ablation: fact_table_permutation` consistently permutes square fact indices
  before query execution.
- Compare against LC0 BT4, NNUE, and the strongest registered idea runs on the same
  split and seeds.
