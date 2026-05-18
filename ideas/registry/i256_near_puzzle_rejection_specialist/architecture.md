# Architecture

`Near Puzzle Rejection Specialist` (i256) targets the verified near-puzzle
false-positive failure mode highlighted by the matched-recall report: a board
emits a strong positive puzzle claim because tactical texture is present
(checks, hanging material, exposed king, promotion tension, one visually
dominant move) but at least one safe defensive reply survives. The architecture
separates a *raw_claim* from a *veto* and combines them as

```text
final_logit = raw_claim - softplus(veto)
```

so the veto can only suppress, never invent, a positive claim.

The source research packet is
`ideas/research/packets/classic/i256_near_puzzle_rejection_specialist.md`. This
implementation realises the packet's recommended `C1_student_full` deployment
target: a fast conv trunk with the chess-explained specialist heads layered
above it, all driven from the simple_18 board only.

## Implementation Binding

- Registered model name: `near_puzzle_rejection_specialist`
- Source implementation:
  `src/chess_nn_playground/models/trunk/near_puzzle_rejection_specialist.py`
  (`NearPuzzleRejectionSpecialist`,
  `build_near_puzzle_rejection_specialist_from_config`)
- Idea-local wrapper:
  `ideas/registry/i256_near_puzzle_rejection_specialist/model.py`
  (`build_model_from_config`)
- Registry manifest key:
  `near_puzzle_rejection_specialist` in
  `src/chess_nn_playground/models/_registry_manifest.py`

## Dataflow

```text
simple_18 board
   │
   ├── DualStreamFeatureBuilder (reused from i193, no learned weights)
   │      ├── exchange feature stack (8 planes)
   │      ├── king feature stack    (8 planes)
   │      └── board-level summary   (8 scalars)
   │
   ├── compact conv encoder (channels=32, depth=2 by default)
   │
   ├── per-square claim_head and reply_head
   │      └── forcedness_gap = claim - reply_escape  (B, 64)
   │
   ├── per-square overload_score_head over own-piece squares
   │
   ├── king_escape_head over enemy-king-zone trunk pool
   │
   ├── candidate_concentration_head over forcedness summary scalars
   │
   ├── raw_claim_head (trunk pool + summary + forcedness stats)
   ├── veto_head      (reply mass + overload + king + concentration)
   │
   └── final_logit = raw_claim_logit - softplus(veto_logit)
```

The forward pass is fully tensor-only: there is no python-chess fallback and no
engine search. The candidate mask is the side-to-move attacker mask derived
from the exchange feature stack (own attacker pressure > 0).

## Specialist Heads

### Forcedness gap head

For each of the 64 squares the model computes two per-square scalars from the
concatenation of the trunk features, the exchange feature planes, and the king
feature planes:

```text
claim(s)         = MLP_claim([trunk[s], exchange[s], king[s]])
reply_escape(s)  = MLP_reply([trunk[s], exchange[s], king[s]])
forcedness_gap(s) = claim(s) - reply_escape(s)
```

Aggregation uses a masked softmax over the side-to-move attacker mask so a
single dominant forcing line is preserved instead of being averaged away. The
exported scalars are `max_forcedness_gap` (soft-max gap), `top2_forcedness_gap`
(top-1 minus top-2 raw gap), `forcedness_gap_entropy` (entropy of the masked
softmax), and `effective_candidate_count` (`exp(entropy)`).

### Defender overload head

A per-square MLP scores each own-piece square from
`[trunk[s], exchange[s]]` and aggregates with a masked softmax over the
own-piece mask. The score is intended to learn an obligation-vs-safe-budget
margin from the deterministic attacker / defender / value planes already in
the exchange feature stack.

### King escape pressure head

Pools the trunk activations weighted by the enemy-king mask and the enemy
king-zone mask, then concatenates with the deterministic king feature mean and
the dual-stream board-level summary. A small MLP emits one scalar
(`king_escape_pressure`).

### Candidate concentration head

Consumes only the four scalars
`[max_forcedness_gap, top2_forcedness_gap, forcedness_gap_entropy,
candidate_count / 64]` and emits one scalar. This keeps concentration
explicitly downstream of the forcedness gap so concentration alone cannot
masquerade as forcing.

### Raw claim head and veto head

`raw_claim_head` combines mean / max / own-attacker-weighted trunk pools with
the dual-stream summary and the forcedness statistics. `veto_head` combines
`reply_escape_mass`, `forcedness_gap_entropy`, negated `top2_forcedness_gap`,
normalised candidate count, the overload score, the king-escape pressure, and
the concentration score. The veto is non-negative after softplus, so it can
only subtract from the raw claim.

## Inputs and Contract

- Input: simple_18 current-board tensor `(B, 18, 8, 8)`.
- Output: dict with `logits` of shape `(B,)` plus per-sample diagnostic
  scalars. Compatible with the repo's shared trainer artifact pipeline.
- The model never reads CRTK metadata, source labels, verification flags, PVs,
  or engine evaluations. Slice tags remain reporting-only.

## Ablations

The model exposes a single `ablation` enum (also surfaced as a config field):

| Ablation | Effect |
|---|---|
| `none` | Full specialist (default). |
| `no_forcedness_gap` | Replace `claim - reply_escape` with raw `claim` only. |
| `no_reply_envelope` | Zero out the reply head before computing the gap. |
| `no_overload_head` | Zero the overload contribution into the veto. |
| `no_king_escape_head` | Zero the king-escape contribution into the veto. |
| `no_concentration_head` | Zero the concentration contribution. |
| `trunk_only` | Disable every specialist head; only the trunk pool feeds the raw claim. |

These match the chess-semantic ablations called out in the research packet
(`P1_i018_gap_only`, `A0_no_reply_envelope`, `A1_no_king_escape`,
`A2_no_overload`, `parent_only`). Each ablation is a one-flag config change
inside the same registered builder.

## Scope Notes

The current implementation is the C1 student-full first deployment target. It
intentionally does *not* yet:

- attach the `L_gap_rank` margin loss or the `L_veto` auxiliary loss from the
  packet (the shared trainer uses BCE-with-logits on the puzzle logit); these
  require trainer extensions that should not be bundled with the
  architecture promotion;
- run any chess-explained near-puzzle curriculum; the shared sampler is used
  so unrelated runs are not perturbed;
- enumerate explicit reply families (recapture, escape, interposition, …);
  the reply envelope is represented by the per-square reply MLP plus the
  pooled overload / king-escape signals. Adding bounded reply-family
  enumeration is the natural next iteration.

These extensions are listed in `implementation_notes.md` as the planned
follow-up work and are deliberately not claimed here.
