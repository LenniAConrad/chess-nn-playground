# Architecture

`Near-Puzzle Hard-Negative Veto` (p055, NPHN) is an additive, gated
*rejection* head on top of the i193 `ExchangeThenKingDualStreamNetwork`
trunk. It is designed to lower the puzzle logit on near-puzzle false
positives by scoring the gap between **surface tactical temptation**
and **verified surviving force** for the side to move.

The model consumes `simple_18` `(B, 18, 8, 8)` and returns one puzzle
logit plus a per-sample diagnostics dict.

## Forward pass

1. **i193 trunk forward.** Emits `base_logit` and trunk diagnostics
   plus the joint pool feature (via `trunk_joint_features`).
2. **Spatial features.** Re-run the dual-stream encoders to get the
   `(B, 2*C, 8, 8)` spatial map (concat of `ex_h` and `kg_h`).
3. **Candidate / reply token pools.** Two `BoardTokenAttention`
   modules compile `num_candidates` candidate tokens and `num_replies`
   defender-reply tokens from the spatial map. No `python-chess` move
   enumeration is required in the forward path.
4. **Surface / verified scoring.** Two MLPs score each candidate
   token; their gap is the legality discount `Disc(m)`.
5. **Candidate concentration.** Softmax the verified scores; report
   the normalized concentration `Conc` and the top-two gap `Gap12`.
6. **Reply neutralization.** A bilinear head produces per-(candidate,
   reply) neutralization scores `n(m, r)`; the soft-existential
   `ReplyMass(m)` is `tau_r * logsumexp(n / tau_r)` and the soft
   `SafeCount(m)` uses a sigmoid threshold.
7. **Forcedness gap.** `FG(m) = u_ver(m) - ReplyMass(m)`,
   `FG* = max FG(m)`, and `FG(m*) = FG[u_ver_argmax]`.
8. **Reply-channel information (RCI).** Mutual information proxy
   computed from the joint softmax of `n(m, r)` (clipped to
   `[0, log num_candidates]` and normalized to `[0, 1]`).
9. **Board-only king-pressure reductions.** `KEP` (king-escape
   pressure) and `DOA` (defender-overload asymmetry) are bounded
   reductions of king-zone attack-defense differences derived from
   `simple_18`'s piece-presence planes and the side-to-move plane.
10. **Diagnostic vector.** `z(x) = [FG*, FG(m*), Disc(m*), Conc,
    Gap12, Avail, RCI, dBal, KEP, DOA, Counter]` (11 entries).
11. **Veto head.** `softplus(MLP(LayerNorm([z; joint])))` produces
    `veto_raw`; the primitive delta is `-veto_raw` so high veto
    pressure subtracts from the puzzle logit.
12. **Gate.** Sigmoid MLP over `[joint; veto_raw; FG*; Avail]` with
    initial bias `gate_init = -2.0`.
13. **Output.** `final_logit = base_logit + gate * (-veto_raw)`.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full NPHN architecture (default). |
| `no_replies` | **Primary falsifier.** Zero out `ReplyMass`, `Avail`, `RCI`. If A1 matches the unablated run, the safe-reply envelope is not load-bearing. |
| `no_legality_discount` | **Primary falsifier.** Collapse `Disc(m*)` to zero. |
| `concentration_only` | Zero every z entry except `Conc` and `Gap12`. |
| `shuffle_replies` | In-batch permutation of reply tokens. Decouples replies from position. |
| `no_overload` | Drop `DOA` from `z`. |
| `no_king_escape` | Drop `KEP` from `z`. |
| `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| `trunk_only` | Same as `zero_delta`; semantic alias. |
| `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Inputs not used

CRTK metadata, source labels, fine labels, verification flags, engine
evaluations, and principal variations are **not** consumed. Hard-
negative mining is a sampler-level concern; this primitive only
provides the rejection diagnostic, not the contrastive loss.

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Trunk joint refeat | Two encoder passes total |
| Candidate / reply pools | Two attention pools, each `O(num_tokens * 64 * token_dim)` |
| Bilinear neutralization | `O(num_candidates * num_replies * token_dim)` |
| Veto / gate MLPs | Small fixed-size MLPs |

At defaults (num_candidates=24, num_replies=24, token_dim=32,
head_hidden_dim=64, B=256) the per-step overhead is small compared
to the trunk. Head adds ~30k-60k parameters.

## Implementation Binding

- Registered model name: `near_puzzle_hard_negative`.
- Source implementation: `src/chess_nn_playground/models/primitives/near_puzzle_hard_negative.py`.
- Shared helper: `trunk_joint_features` from
  `src/chess_nn_playground/models/primitives/trunk_features.py`;
  `BoardTokenAttention` from
  `src/chess_nn_playground/models/primitives/codex_reply_primitives.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p055_near_puzzle_hard_negative/model.py`.
- Training config: `ideas/registry/p055_near_puzzle_hard_negative/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["near_puzzle_hard_negative"] = build_near_puzzle_hard_negative_from_config`.
