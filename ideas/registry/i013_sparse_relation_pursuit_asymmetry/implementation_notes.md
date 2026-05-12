# Implementation Notes

Code-complete components:

- `SparseRelationPursuitClassifier`: board-to-square stem, fixed chess relation tokenization, sparse descriptor classifier.
- `GroupSparsePursuit`: unrolled group-sparse pursuit with normalized dictionary decoder.
- `SRPALoss`: BCE plus auxiliary residual-asymmetry and sparse/dictionary regularizers.
- Model registry aliases: `sparse_relation_pursuit_asymmetry` and `sparse_relation_pursuit`.

Correctness constraints implemented:

- Background and tactical dictionaries use identical atom count, group count, atom dimension, and pursuit depth.
- Relation edges are generated deterministically from board geometry.
- The classifier head input dimension is exactly the sparse descriptor dimension.
- Dense board embeddings only create relation tokens; they do not bypass the pursuit layer into the classifier.
- Fine labels are not consumed by the model or loss. In `puzzle_binary`, they remain available only for saved diagnostic slice reports.

The first production config keeps the model deliberately moderate because relation-token pursuit is heavier than the LC0 BT4 baseline.
