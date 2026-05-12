# Codex Primitive Stacking Strategy

Author: Codex
Model: GPT-5 (Codex coding agent)
Date: 2026-05-12
Status: synthesis packet

## Purpose

This file documents how the five primitives in this directory could be tested
alone and then combined into a later architecture without blurring their
individual falsifiers.

## Primitive map

| Primitive | Proposed architecture | Main target | Expected cost | First decision |
|---|---|---|---|---|
| DHPE, signed piece-existence Hessian | i245 Pair-Resonance Hessian Network | hard, mate-in-1, near-puzzle sign ambiguity | high, many trunk passes | keep only if signed Hessian beats unsigned Hessian |
| TDCD, tempo-defender cross-derivative | i244 Tempo-Defender Cross-Derivative Network | equal eval bucket | medium-high, batched counterfactual passes | keep only if mixed partial beats main effects |
| PFCT, promotion-fanout counterfactual | i246 Promotion-Aware Head | promotion and underpromotion slices | sparse, zero on most positions | keep only if substitution fanout beats copied baseline |
| CAIO, complex-amplitude interference | i247 Complex-Amplitude Chess Network | broad color/tempo/relation coherence | medium | keep only if phase and relation masks are load-bearing |
| TSDP, terminal-state detection | i248 Rule-Aware Tactical Head | mate_in_1 slice + stalemate-trap avoidance | very low (~1.05x i193); precomputable in data loader | keep only if shuffled-indicator ablation hurts mate_in_1 slice |

## Recommended test order

1. **TSDP first.** It is the cheapest primitive (~1.05x i193 wall-clock,
   precomputable in the data loader) and uses chess-rule-exact signal that no
   existing model has direct access to. If it doesn't move the mate_in_1
   slice by >0.02 PR AUC, it's clear the trunk is already learning that
   signal implicitly and we save expensive primitives for the next test.
2. **PFCT second.** It targets the largest measured slice gap and has no
   overhead when a position has no near-promotion pawn.
3. **TDCD third.** It targets the `equal` bucket, one of the most stable hard
   failure modes in the audit reports.
4. **DHPE fourth.** It is expensive, but directly tests pairwise constructive
   versus substitutive tactical interactions.
5. **CAIO fifth.** It is the speculative structural bet and should run only
   after the easier counterfactual primitives have clear outcomes.

## Composite architecture sketch

The eventual hybrid should not concatenate every intermediate tensor. It
should export small diagnostics from each primitive:

```text
i193 exchange/king trunk
-> base logit and base feature

TSDP branch:
  rule-exact mate_in_1, stalemate_threat, forcing-move counts (stop-grad)

PFCT branch:
  near-promotion fanout summaries

TDCD branch:
  tempo gradient, defender cross-derivative spectrum

DHPE branch:
  signed pair-Hessian top-k summaries

CAIO branch:
  constructive/destructive interference summaries

gated fusion:
  z = [base_feature, tsdp_diag, pfct_diag, tdcd_diag, dhpe_diag, caio_diag]
  logit = base_logit + MLP(z)
```

The fusion MLP should be small and regularized. A large fusion head would make
it hard to tell whether the primitives or added capacity caused any gain.

## Compatibility with the candidate/reply primitives

The separate 2026-05-12 candidate/reply primitives can sit downstream of this
stack:

- DHPE can help score candidate-piece interaction features.
- TDCD can produce defender-criticality priors for reply selection.
- PFCT can add explicit promotion candidates to the candidate compiler.
- CAIO can add relation-coherence features to candidate and reply utilities.
- TSDP can short-circuit reply evaluation when a candidate move is detected
  as immediate mate or stalemate — saving compute on obviously-decisive
  candidates and biasing the head toward rule-exact signal where available.

The resulting path is:

```text
board
-> i193-style trunk
-> DHPE / TDCD / PFCT / CAIO / TSDP diagnostics
-> candidate compiler
-> reply compiler
-> WCQ / PAFR / RSP / RCC / TCC reducers
-> VetoSelect-style acceptance head
```

Do not build this full stack first. Build each primitive as a side head, prove
its ablation is load-bearing, then add only the winners to the candidate/reply
architecture.

## Drop rules

Drop a primitive if any of these occur:

- its main ablation matches the full version;
- aggregate PR AUC regresses more than the stated tolerance;
- the target slice does not move by a meaningful amount;
- the win disappears when relation masks or rule-specific counterfactuals are
  shuffled;
- the only benefit can be matched by a same-parameter ordinary MLP or attention
  control.

## Promotion rule

Promote a primitive into an `ideas/registry/i###_*` folder only after a scout run has:

- one full result note under `runs/`;
- aggregate PR AUC, near-puzzle false-positive rate, and slice reports;
- the primitive-specific ablation;
- a clear keep/drop decision in Markdown.
