# Architecture

`Side-Canonical Rule-Partition Invariant Bottleneck` (SCRIB) is a board-only
`puzzle_binary` classifier whose central operator is the rule-partition
invariant bottleneck `B(x) = h(z), z ~ q(z | C(x))` from the markdown
math thesis. The implementation replaces the shared research-packet probe
with a materially distinct bespoke model so the SCRIB minimax objective
(CE + KL + V-REx + gradient-reversed environment adversaries) can be
exercised by trainable code rather than a generic mechanism profile.

## Forward Pipeline

1. **Side-to-move canonicalization (`Simple18SideCanonicalizer`).** The
   `simple_18` board tensor `(B, 18, 8, 8)` is rewritten into a 17-channel
   side-relative tensor `(B, 17, 8, 8)`: white-to-move samples are passed
   through unchanged on the piece, castling and en-passant planes, while
   black-to-move samples have ranks vertically flipped, white/black piece
   planes swapped, white/black castling planes swapped (king/queen-side
   preserved) and the en-passant plane rank-mirrored. The absolute
   side-to-move plane is removed. Unsupported encodings fail closed.
2. **Rule partitioner (`Simple18RulePartitioner`).** Three deterministic
   integer labels are computed from the `simple_18` material counts and
   the absolute side-to-move plane: `phase_labels in {0,1,2}` (coarse
   total non-king material bucket cut at <=20, 20-49, >=50 pawn units),
   `adv_labels in {0,1,2,3,4}` (side-relative material balance cut at
   <=-5, -4..-2, -1..1, 2..4, >=5 with standard P=1, N=3, B=3, R=5, Q=9
   weights), and `color_labels in {0,1}` (absolute side-to-move). The
   coarse group id `group_ids = phase + 3*adv + 15*color in {0,...,29}`
   is also returned. Partitions are *not* concatenated to the model
   input; they are exposed only as supervision targets and as the V-REx
   group key.
3. **Compact convolutional trunk (`ConvTinyBackbone`).** The canonical
   17-channel tensor flows through a two-stage trunk at default widths
   `(64, 96)`: `Conv(17 -> 64) + norm/GELU`, two residual micro-blocks
   at width 64, `Conv(64 -> 96) + norm/GELU`, two residual micro-blocks
   at width 96. The 8x8 spatial map is concatenated (mean and max pool)
   into a 192-d feature, then passed through `Linear(192 -> 256) + GELU
   + Dropout` to produce a 256-d trunk feature.
4. **Variational information bottleneck (`VariationalBottleneck`).** Two
   linear heads produce `mu` and `logvar` of size `latent_dim=128`.
   During training `z = mu + exp(0.5 * logvar) * eps` with reparameterized
   Gaussian noise; during evaluation `z = mu`. The per-sample KL
   divergence to `N(0, I)` is returned for the `beta` term in the SCRIB
   objective.
5. **Label head.** `LayerNorm(128) -> Linear(128 -> head_hidden) -> GELU
   -> Dropout -> Linear(head_hidden -> num_classes)` consumes only `z`
   and emits the puzzle logit. With `num_classes=1` the trainer reads a
   single BCE-with-logits scalar; with `num_classes=2` the markdown's
   `(B, 2)` cross-entropy contract is supported.
6. **Environment adversary heads (`GradientReversalLayer` +
   `EnvAdversaryHead`).** A gradient-reversal layer projects `z` to
   `z_rev` and three linear adversary heads predict the rule partitions
   from `z_rev`: `phase` (3-way), `adv` (5-way), `color` (2-way). Their
   cross-entropy contributes to the SCRIB minimax objective; gradient
   reversal makes the trunk and VIB push `z` toward partition
   non-identifiability.

## Output Contract

`forward(x)` returns a `dict` whose `"logits"` entry has shape `(B,)` for
the `puzzle_binary` BCE-with-logits trainer when `num_classes=1` (or
`(B, 2)` for cross-entropy when `num_classes=2`). Auxiliary tensors
exposed for the SCRIB trainer include `z`, `mu`, `logvar`, the per-sample
`kl`, the adversary `phase_logits` / `adv_logits` / `color_logits` and
their softmax probabilities, the deterministic `phase_labels`,
`adv_labels`, `color_labels`, and `group_ids` plus diagnostic
`total_material` and `side_relative_advantage` scalars. Adversary heads
read the gradient-reversed latent so the encoder maximizes adversary CE
while the heads minimize it.

## Implementation Binding

- Registered model name: `side_canonical_rule_partition_invariant_bottleneck`
- Source implementation file: `src/chess_nn_playground/models/rule_partition_invariant_bottleneck.py`
- Idea-local wrapper: `ideas/i043_side_canonical_rule_partition_invariant_bottleneck/model.py`
