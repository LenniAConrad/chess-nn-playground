# Codex Research Synthesis: 2026-05-12 Created Idea Ledger

Author: Codex
Model: GPT-5 (Codex coding agent)
Date: 2026-05-12
Status: synthesis packet

## Purpose

This file is the explicit Markdown ledger for the ideas Codex created during the 2026-05-12 primitive/model exploration session. Each full idea remains in its own packet; this ledger makes the set discoverable from one place.

Imported Claude, GPT, and Google/Gemini primitive reports are not counted as Codex-created ideas here. They are documented separately in [external_imports/MANIFEST.md](external_imports/MANIFEST.md).

## Codex-Created Ideas

| Idea | Kind | Packet | Main role |
|---|---|---|---|
| Forcing Reply Envelope Veto Network | model architecture | [forcing reply envelope veto](architecture_bridges/forcing_reply_envelope_veto_network.md) | i193-style exchange/king parent plus reply-envelope veto head for near-puzzle false-positive rejection. |
| Witness-Counterwitness Quantifier Primitive | neural primitive | [witness counterwitness quantifier](codex_candidate_reply_primitives/05_witness_counterwitness_quantifier.md) | Differentiable `exists witness / no surviving counterwitness` reducer over ragged candidate-reply sets. |
| Pareto Antichain Frontier Primitive | neural primitive | [pareto antichain frontier](codex_candidate_reply_primitives/01_pareto_antichain_frontier.md) | Partial-order reducer that keeps nondominated tactical candidate frontiers instead of collapsing early to one scalar. |
| Regret Saddlepoint Primitive | neural primitive | [regret saddlepoint](codex_candidate_reply_primitives/02_regret_saddlepoint.md) | Entropy-regularized zero-sum reducer for attacker-candidate versus defender-reply payoff tables. |
| Reply Channel Capacity Primitive | neural primitive | [reply channel capacity](codex_candidate_reply_primitives/03_reply_channel_capacity.md) | Information-theoretic reducer measuring whether candidate choice collapses or controls defender reply distributions. |
| Tail Copula Concordance Primitive | neural primitive | [tail copula concordance](codex_candidate_reply_primitives/04_tail_copula_concordance.md) | Rank-copula reducer checking whether multiple evidence fields become extreme on the same squares or candidates. |

## Shared Architecture Direction

The five primitives are designed to share a future candidate/reply infrastructure:

```text
current board
-> i193-style exchange/king features
-> candidate compiler
-> reply compiler
-> candidate x reply utility table
-> WCQ / PAFR / RSP / RCC / TCC diagnostics
-> VetoSelect-style acceptance head
```

The first implementation path should be padded PyTorch prototypes, not fused kernels. Fused or ragged custom kernels should wait until a primitive wins against simple attention, max-minus-max, shuffled-reply, and scalar-pooling controls on the hard near-puzzle slices.

## Minimum Documentation Check

Every Codex-created idea in this session has a standalone Markdown packet and is linked above.

Additional Codex/Claude-authored primitive notes were also present in a subdirectory and have been made internally complete:

| Idea | Kind | Packet |
|---|---|---|
| Signed Piece-Existence Hessian Operator | neural primitive | [DHPE](claude_opus_4_7_primitives/01_signed_hessian_resonance.md) |
| Tempo-Defender Cross-Derivative Operator | neural primitive | [TDCD](claude_opus_4_7_primitives/02_tempo_defender_cross_derivative.md) |
| Promotion-Fanout Counterfactual Tensor | neural primitive | [PFCT](claude_opus_4_7_primitives/03_promotion_fanout_counterfactual.md) |
| Complex-Amplitude Interference Operator | neural primitive | [CAIO](claude_opus_4_7_primitives/04_complex_amplitude_interference.md) |
| Terminal-State Detection Primitive | neural primitive | [TSDP](claude_opus_4_7_primitives/05_terminal_state_detection.md) |
| Codex Primitive Stacking Strategy | synthesis | [stacking strategy](architecture_bridges/codex_primitive_stacking_strategy.md) |

The imported primitive research set also has a Markdown manifest:

- [external_imports/MANIFEST.md](external_imports/MANIFEST.md)
