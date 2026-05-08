# Math Thesis

Negative-Class Disentangled Puzzle Head

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md`.

Batch candidate rank: `1`.

Working thesis: The `puzzle_binary` target is binary, but the negative
class is generated from two structurally different sources: random
non-puzzle positions and near-puzzle hard negatives. Forcing both into
one undifferentiated negative representation pushes the model toward an
average negative concept and lets near-puzzle false positives leak past
the threshold. The thesis is therefore to expose three explicit evidence
channels — random-negative, near-puzzle-negative, and puzzle-positive —
and collapse them into a single inference logit by a logsumexp negative
competition.

Concretely, let `h = trunk(board)` denote pooled board features from a
board-only `simple_18` trunk and `z = SharedProj(h)` the shared evidence
representation. Three independent two-layer MLP heads produce scalar
evidence

    e_random = head_random(z)
    e_near   = head_near(z)
    e_puzzle = head_puzzle(z)

The puzzle inference logit is the disentangled formula prescribed by the
packet,

    puzzle_logit = e_puzzle - logsumexp([e_random, e_near]),

so that

    sigmoid(puzzle_logit) = exp(e_puzzle) / (exp(e_random) + exp(e_near) + exp(e_puzzle)).

The puzzle channel must win an explicit competition against both
negative channels at once; it is not enough to look unlike "average
negative". The raw `[e_random, e_near, e_puzzle]` stack is also exposed
as `aux_3way_logits` so the trainer may attach the packet's auxiliary
3-way cross-entropy on the fine source label
(`fine 0 -> random`, `fine 1 -> near`, `fine 2 -> puzzle`) without
changing the inference contract.

The bespoke model implementing this thesis is registered under the model
name `negative_class_disentangled_puzzle_head` and shares its
implementation with idea `i074`'s
`puzzle_binary_benchmark_challengers`; see `architecture.md` and
`implementation_notes.md` for the binding.
