# Architecture

`Forcing-Response Front-Door Bottleneck` is a board-only `puzzle_binary`
classifier that follows the markdown thesis from
`ideas/research/packets/classic/chess_nn_research_2026-04-28_0733_tuesday_new_york_forcing_response_bottleneck.md`.
The decisive layer is a sparse causal witness bottleneck `Z_c` over
deterministic legal-move response envelopes, so the binary head can
never read pooled raw board features directly.

## Mechanism

1. **Rule-intervention feature builder.** `RuleInterventionFeatureBuilder`
   converts each `simple_18` board tensor into a `python-chess` board,
   enumerates visible-board legal candidates `A(x)` (padded/truncated to
   `M_MAX = 256`), and computes:
   - 12 deterministic rule planes (own/opponent attack maps, attack
     overlap, normalised legal from/to densities, own/opponent king
     rings, occupancy/colour planes, empty squares, current checkers).
   - 64-dim `move_features[a]` covering coordinates, motion deltas,
     moving/captured piece types, promotion bucket, capture/check/
     castle/en-passant flags, attacker/defender of from/to squares,
     king-ring membership, slider type, material delta, centre/edge
     bias, and after-move check indicators.
   - 64-dim `response_features[a]` summarising the opponent reply
     envelope after `do(a)`: clipped/log reply counts and
     {check, capture, promotion, king-move, recapture-of-moved-piece,
     captures-of-checking-piece, quiet} bucket counts, attack-map and
     slider-pressure deltas, king-ring pressure for both colours,
     destination-attacker counts, and remaining piece-type counts for
     the defender.
   - `move_mask`, `move_from`, `move_to`, and `path_weights` (a
     deterministic ray pooling weight per slider candidate).
2. **Board-rule stem.** The `simple_18` board tensor is concatenated
   with the rule planes and passed through `BoardRuleStem`: a
   `Conv2d -> 4 ResidualBlock(channels = 64) -> GroupNorm` encoder.
   The stem produces site embeddings `H` of shape `(B, 64, 8, 8)`. The
   binary head never reads pooled `H` directly.
3. **Move-response node encoder.** Per candidate `a` the model gathers
   `from_emb = H[from(a)]`, `to_emb = H[to(a)]`, a deterministic ray
   `path_emb = path_weights[a] * H` (zero for non-sliders), and
   embeds the move/response feature vectors with two MLPs:
   ```
   u_a = MLP_node([from_emb, to_emb, path_emb, MLP_move(move_features[a]),
                   MLP_response(response_features[a])])
   ```
   `u_a` lives in `R^{hidden_dim}` and is masked to legal candidates.
4. **Permutation-invariant move graph.** Two `MoveRelationGraphBlock`
   layers run set-style message passing over the candidate set. Each
   block aggregates eight masked group means (global, same-from-square,
   same-to-square, capture / check / recapture / king-pressure
   indicators) so the graph is permutation-invariant over candidate
   ordering and uses only deterministic relations from the packet's
   section 9.4.
5. **Sparse causal witness gate.** `SparseWitnessGate` produces masked
   gate scores from `u_a`. During training, hard-concrete relaxed
   gates `g_a ∈ [0, 1]` (Gumbel-sigmoid noise + linear stretch with
   clamp) approximate the discrete top-`K` selection. During eval, the
   model picks the top `K = witness_count` legal candidates.
6. **Bottleneck `Z_c`.** Values `v_a = Linear(u_a)` are weighted by the
   gates and length-normalised:
   ```
   Z_c = LayerNorm(sum_a g_a * v_a / (epsilon + sum_a g_a))
   ```
   `Z_c` is the only path to the binary head, so the front-door
   surrogate `X -> M -> Y` is enforced architecturally rather than by a
   regulariser.
7. **Heads.**
   - Binary head: `Linear(Z_c) -> GELU -> Linear -> 1` returns the
     puzzle logit `(B,)`.
   - Fine-label head: `Linear(Z_c, 3)` for optional training-only
     auxiliary loss (never an input).
   - Masked mediator head: `Linear(u_a, move_dim + response_dim)` for
     optional masked rule-feature reconstruction.

## Output Contract

Forward returns a dict whose `"logits"` entry is `(B,)` for the
repository `puzzle_binary` BCE-with-logits trainer. Diagnostic
tensors appended to prediction artefacts include:

- `z_c`: `(B, bottleneck_dim)` causal witness bottleneck.
- `witness_gates`: `(B, M_MAX)` per-candidate gate values.
- `witness_gate_logits`: `(B, M_MAX)` masked gate logits (for
  diagnostics / KL of the gate distribution).
- `fine_logits`: `(B, 3)` auxiliary fine-label logits.
- `masked_pred`: `(B, M_MAX, move_dim + response_dim)` masked-mediator
  reconstruction targets for self-supervised loss.
- `mechanism_energy`: `(B,)` mean-squared bottleneck activation.
- `proposal_profile_strength`: `(B,)` normalised gate mass.
- `proposal_keyword_count`: `(B,)` constant 6.0 marker for legacy
  reporting parity.
- `reply_pressure`: `(B,)` gate-weighted log reply count.
- `defense_gap`: `(B,)` `(1 - reply_pressure)(1 - recapture_pressure)`
  proxy from the packet's defence diagnostics.
- `sparse_witness_count`: `(B,)` active-witness count under the
  `> 0.05` threshold.
- `sparse_gate_mass`, `gate_entropy`, `front_door_bottleneck_l2`,
  `top_witness_gate`: extra diagnostics for gate sparsity, witness
  diversity, bottleneck capacity, and the dominant witness weight.

## Leakage Guards

The forward pass consumes only the board tensor and deterministic
rule features computed from it. The packet's forbidden inputs (engine
scores, PVs, best moves, mate scores, source IDs, verification flags,
fine labels) are never passed to the model. Fine labels and CRTK
metadata remain reporting-only.

## Implementation Binding

- Registered model name: `forcing_response_front_door_bottleneck`.
- Source implementation file: `src/chess_nn_playground/models/trunk/forcing_response_front_door_bottleneck.py`.
- Idea-local wrapper: `ideas/registry/i081_forcing_response_front_door_bottleneck/model.py`.
