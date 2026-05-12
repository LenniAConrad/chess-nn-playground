# Architecture

`Prototype-Margin Puzzle Network` is a board-only classifier for the
`puzzle_binary` task. It accepts the repository's `simple_18`
current-board tensor with shape `(B, 18, 8, 8)` and returns one
puzzle logit per position by comparing the encoded board to three
banks of learned prototypes — random non-puzzle, near-puzzle, and
real puzzle — and taking the margin between the puzzle similarity
and the largest non-puzzle similarity.

## Mechanism

The architecture follows the packet thesis verbatim. Three
prototype banks are learned:

```
P_random:  K x D
P_near:    K x D
P_puzzle:  K x D
```

For an encoded board latent `z`, per-class similarities are computed
as

```
sim_class(z) = logsumexp_k cosine(z, P_class[k]) / temperature
```

and the puzzle logit is the prototype margin

```
puzzle_logit = sim_puzzle - logsumexp([sim_random, sim_near])
```

so puzzles compete directly with separate attractors for random
non-puzzle positions and near-puzzle hard negatives.

Inputs to the model are limited to the `simple_18` board tensor.
Engine, verification, source, and CRTK metadata are never used.

## Trunk and encoder

A stack of `depth` `Conv3x3 → BatchNorm → ReLU` layers turns the
18-plane board into a per-square feature map of width `channels`.
The map is reduced with mean-and-max global pooling to a vector of
width `2 * channels`, then a `LayerNorm → Linear → GELU → Dropout →
Linear` encoder projects it to the `proto_dim`-wide board latent
`z` that all three prototype banks see.

## Prototype banks

Each bank holds `num_prototypes` learnable prototypes of width
`proto_dim` initialised with Kaiming-uniform. For an input `z`, the
bank computes

```
scores  = cosine(z, prototypes)               # (B, K)
sim     = logsumexp(scores, dim=-1) / temperature
```

`sim_puzzle`, `sim_random`, and `sim_near` are produced by the
`puzzle_bank`, `random_bank`, and `near_bank` respectively.

## Margin head

The default head is the packet's prototype margin

```
negative_logsumexp = logsumexp([sim_random, sim_near])
puzzle_logit       = sim_puzzle - negative_logsumexp
```

When `num_classes > 1` the puzzle margin is written into the last
column of a zero-padded logits tensor so the BCE-with-logits trainer
contract still holds.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for
the repository `puzzle_binary` BCE-with-logits trainer. All
tensors are finite per batch:

- `logits`: `(B,)` puzzle logit (or `(B, num_classes)` when
  `num_classes > 1`).
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `z`: `(B, proto_dim)` board latent fed to the prototype banks.
- `trunk_features`: `(B, channels, 8, 8)`.
- `trunk_energy`: `(B,)` mean-square trunk activation.
- `sim_random`, `sim_near`, `sim_puzzle`: `(B,)` per-class
  log-sum-exp similarities.
- `negative_logsumexp`: `(B,)` log-sum-exp aggregation of the two
  negative similarities (the right-hand side of the margin).
- `puzzle_margin_signal`: `(B,)` raw value the puzzle head consumes
  (== `logits` when `num_classes == 1` and the prototype-margin
  head is active).
- `random_scores`, `near_scores`, `puzzle_scores`:
  `(B, num_prototypes)` cosine similarities to each individual
  prototype, useful for diagnosing prototype collapse.
- `num_prototypes_levels`, `proto_dim_levels`,
  `temperature_levels`: `(B,)` scalar tags carrying the configured
  prototype geometry.
- `ablation_active`, `uses_separate_negatives`,
  `uses_margin_head`, `random_proto_frozen`: `(B,)` flags exposing
  the running ablation.

## Ablations

The packet's required ablations are exposed via `ablation`:

- `"none"` — main model.
- `"single_negative_proto"` — drop the near-puzzle bank and use
  only `P_random` as the negative attractor. Tests whether the
  separate near prototype is doing real work.
- `"no_margin_logsumexp"` — replace the prototype-margin head with
  a plain linear puzzle head over `z`. Tests prototype competition.
- `"random_proto_freeze"` — freeze the random-proto bank at its
  Kaiming initialisation so it cannot adapt during training. Tests
  whether learned random prototypes matter.
- `"prototype_count_sweep"` — no-op structural flag. The sweep
  itself is driven by the `num_prototypes` config value; tagging a
  run with this ablation marks it as a sweep entry without changing
  the main model.

## Implementation Binding

- Registered model name: `prototype_margin_puzzle_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/prototype_margin_puzzle_network.py`
- Idea-local wrapper: `ideas/registry/i175_prototype_margin_puzzle_network/model.py`
