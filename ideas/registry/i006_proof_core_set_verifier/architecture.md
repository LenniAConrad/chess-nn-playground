# Architecture

## Architecture Description

The architecture has three parts:

1. token generator for candidate pieces/squares
2. differentiable top-k proof-core selector
3. relation-aware set verifier

The verifier only sees selected tokens and their deterministic chess relations.

## Input Format

```text
board: batch x 18 x 8 x 8
tokens: batch x max_tokens x token_features
relations: batch x max_tokens x max_tokens x relation_features
```

Candidate tokens include:

- occupied pieces
- king-zone squares
- high-value target squares
- slider line intersections
- promotion squares if relevant

## Forward Pass

```text
H = board_stem(board)
T = token_encoder(tokens, H)
w = selector(T, pool(H))
S = differentiable_topk(T, w, k)
Rel_S = gather_relations(relations, S)
proof_logit = set_verifier(S, Rel_S)
residual = bounded_global_head(pool(H))
logit = proof_logit + residual
```

## Tensor Shapes

Suggested first version:

```text
max_tokens: 96
selected_k: 8 or 12
token_dim: 96
relation_dim: 16
output: batch x 1
```

## Output Heads

Primary:

```text
puzzle_logit
```

Diagnostics:

```text
witness_mask
proof_logit
global_residual
deletion_gap
selection_entropy
```

## Parameter Estimate

```text
1M to 2.5M parameters
```

## FLOP Estimate

Verifier cost is:

```text
O(batch * selected_k^2 * token_dim)
```

The selector over all candidates is cheaper than dense board attention.

## Implementation Binding

- Registered model name: `proof_core_set_verifier`.
- Source implementation: `src/chess_nn_playground/models/trunk/research_architectures.py`.
- Idea-local wrapper: `ideas/registry/i006_proof_core_set_verifier/model.py`.
