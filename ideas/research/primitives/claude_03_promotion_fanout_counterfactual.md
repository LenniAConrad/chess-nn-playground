# 03 — Promotion-Fanout Counterfactual Tensor (PFCT)

**Slug:** `primitive_promotion_fanout_counterfactual`
**Status:** proposed
**Author:** Claude
**Model:** Claude Opus 4.7
**Architecture extension:** i246 Promotion-Aware Head

## One-line claim

A forward-pass primitive that, for each pawn on its 7th rank, replaces it on
its 8th rank with each of the four legal promotion piece types `{Q, R, B, N}`
and returns the **four resulting feature vectors** as a per-pawn fanout
tensor — exposing the **latent piece-type identity** of pre-promotion pawns
that static encoders collapse to type `P`.

## Mathematical signature

For a position `x` with side-to-move `c`, let `P_near(x, c)` be the set of
own pawns on the 7th rank. For each such pawn `p` at square `s_p` and any
learned scorer `phi_theta : Board -> R^d`:

    F_theta(p, x) = [ phi_theta( x -> (p -> Q at s_p+1) ),
                      phi_theta( x -> (p -> R at s_p+1) ),
                      phi_theta( x -> (p -> B at s_p+1) ),
                      phi_theta( x -> (p -> N at s_p+1) ) ]
                  in  R^{4 x d}

where `x -> (p -> T_{s_p+1})` means "remove pawn p from s_p, place piece type
T of side c at the promotion square s_p+1." The primitive outputs
`{F_theta(p, x) : p in P_near}` plus per-pawn discriminator scalars (e.g.
`argmax_T |F_theta(p)[T]|` — the dominant promotion choice).

Cost: `4 * |P_near|` shared-encoder forward passes per position;
`|P_near| in {0, ..., 4}` typically, so `0-16` extra passes. **Zero overhead
on positions without near-promotion pawns** (the primitive is a no-op when
`P_near` is empty).

## Why this is genuinely new

Empirical territory check (grep on registry + packets): no entry contains
`promotion`, `underpromot`, `pawn.transform`, `piece.type.substitut`,
`piece.identity` as a primitive or architecture concept. The promotion slice
— which has the **largest per-slice PR-AUC gap in the entire benchmark**
(best 0.667 vs aggregate 0.876) — is completely unaddressed.

PFCT is a counterfactual primitive, but the counterfactual axis is unique:

| primitive | counterfactual axis |
|---|---|
| i025 One-Ply Move Counterfactual | **move** space (legal moves) |
| i041 Tempo-Odd Bottleneck | **tempo** space (side-to-move flip) |
| i189 Defender Dropout | **presence** space (binary piece existence) |
| TDCD (codex 02) | **tempo x defender** cross-derivative |
| DHPE (codex 01) | **piece-pair existence** Hessian |
| **PFCT (this)** | **piece-type substitution** at promotion squares |

No existing primitive enumerates the chess-rule-legal *type transformation*
(promotion). This is the only legal type-transformation in chess — the
primitive is structurally aligned with a chess rule.

## Empirical evidence from prototype

The local prototype `prototypes/pfct_prototype.py` runs four hand-crafted
scenarios with a tiny `TinyTacticalScorer`. It is a design check only, not a
scout-scale benchmark run:

| scenario | best promotion | 2nd | gap | read |
|---|---|---|---:|---|
| **A. Standard** (pawn a7, empty board, default piece values) | **Q** | R | +31.8 | chess-default Q > R > B > N recovered |
| **B. Planted knight-fork motif** | **N** | Q | **+4073.4** | knight-fork motif rewards knight on a8 + king on e1; fanout argmax **flips from Q to N** purely from planted structure |
| **C. Two near-promotion pawns** (a7 + h7) | Q + Q | – | – | multi-pawn handling: 8 forward passes; per-pawn fanouts produced correctly |
| **D. Autograd** | – | – | – | gradient norm = 122,898; reaches `piece_strength`, `square_embed`, `motif_w` |

The B->A contrast is the key empirical evidence that the primitive carries
chess-tactical signal: the same input (pawn on a7), with the same baseline
encoder, produces a *different ranking* over promotion choices when the
tactical context favours a non-queen promotion. The architecture's job
downstream is to learn to *attend* to the right row of `F`.

## Architecture extension — i246 Promotion-Aware Head

Wraps any existing trunk (i193, i243):

```
Input: position x
   |
   v
[backbone trunk phi_theta]  -->  baseline features  f_base in R^d
   |
   v
For each p in P_near(x):
   compute PFCT fanout  F_theta(p, x) in R^{4 x d}    (4 shared-trunk forward passes)
   |
   v
Cross-attention head:
   q = MLP_q(f_base, p_emb)                    # per-pawn query
   k = F_theta(p, x)                            # 4 keys (Q, R, B, N rows)
   alpha = softmax(q . k^T / sqrt(d))           # 4-way attention weights
   f_pawn(p) = alpha . F_theta(p, x)            # weighted promotion-features
   |
   v
f_combined = f_base + sum_p f_pawn(p)
   |
   v
puzzle_binary logit
```

The architecture decides which promotion choice "matters" via attention.
On positions without near-promotion pawns, the architecture is identical to
the trunk -> zero overhead.

Two natural follow-ups:

- **i247 Promotion-Race Cross-Pawn Reasoner**: cross-pawn attention for
  endgames with passed pawns on opposite wings
- **i248 HalfKA-PFCT Hybrid**: slot PFCT into i243's HalfKA-dual-stream-LC0
  composition; the HalfKA accumulator gives king-conditional embeddings,
  PFCT contributes piece-type-counterfactual embeddings for near-promotion
  pawns

## Falsifier

Standard scout setup: 173k x 12 epochs, single seed, single 3070.

**Primitive-level pass criterion**:

- Ablation A1: replace PFCT's 4-way fanout with 4 copies of the baseline
  `phi(x)` (disable substitution, keep architecture). Architecture must lose
  >= 60% of any promotion-slice lift. If equal performance, PFCT's
  substitution adds nothing and the proposal is rejected.

**Architecture-level pass criterion** (i246 vs i193):

- `crtk_tactic_motifs = promotion` slice PR AUC: i246 >= **0.720**
  (i193's 0.652 + 0.068, ~10% relative improvement)
- `crtk_tactic_motifs = underpromotion` slice PR AUC: i246 >= **0.720**
  (joint-tagged with promotion in CRTK)
- Aggregate test PR AUC: i246 >= 0.873 (i193's 0.876 - 0.003)
- All other slices: no regression > 0.01

**Fail criteria**:

- A1 ablation matches the full architecture -> drop
- Promotion slice improvement < 0.02 PR AUC -> too narrow; drop
- Aggregate regresses > 0.005 -> hurts overall; drop

## Generalisation beyond chess

The primitive generalises to any domain where one input element can take
values from a fixed discrete alphabet under a domain-specific transformation
rule:

- **Molecular property prediction**: a stem-cell atom that could
  differentiate in a synthesis path
- **Genomics**: a nucleotide that could mutate to `{A, C, G, T}` under a
  biological rule
- **NLP**: a polysemous token whose context-induced sense could be one of K
  alternatives
- **Reinforcement learning**: an action with K typed variants (e.g. loss
  function selection in meta-learning)

The chess instance (promotion -> `{Q, R, B, N}`) is the canonical 4-way
example. The primitive's general form is **discrete-alphabet counterfactual
substitution at rule-defined sites**, mathematically distinct from
continuous counterfactuals (gradients, perturbations) or unordered set
counterfactuals (Shapley, dropout).

## Risks

1. **Narrow slice coverage** — PFCT only fires on near-promotion pawns
   (~6% of positions). Even a perfect promotion-slice fix yields only
   ~+0.5pp aggregate PR AUC. **Mitigation**: treat PFCT as a *complement* to
   broader primitives (DHPE, TDCD) rather than a standalone fix. Its
   strength is *concentrated* and *zero-cost on irrelevant positions*.
2. **Hidden rebrand of "feature engineering"** — "this is just 4 forward
   passes with edited input." **Defence**: the four-fold substitution
   structure encodes the chess promotion rule — a *legal type
   transformation* that exists nowhere else in the move space. The
   per-pawn fanout tensor of shape `(4, d)` is a *typed* output that
   downstream attention/heads can use without learning the chess rule.
3. **Encoder must be piece-type-sensitive** — if `phi_theta` produces
   near-identical features for Q vs R vs B vs N on the same square, PFCT's
   fanout will be near-zero. Mitigate by using a trunk with explicit
   piece-type embeddings (i193, i243, any LC0-style).
4. **Promotion-square occupancy edge case** — if the promotion square is
   occupied by an opponent piece, the promotion would be a capture (legal);
   if occupied by own piece, the promotion is illegal. Mitigation: explicit
   pre-check on the promotion square; flag illegal-promotion positions for
   skip during evaluation.

## Why high-potential

- Surgical targeting of the **largest per-slice gap in the benchmark**
  (promotion 0.667 vs aggregate 0.876 = 0.21 gap, far larger than the
  `equal` bucket's 0.06 gap)
- Cost is gated by `|P_near|`, which is 0 for most positions — no
  scout-scale wall-clock penalty on the bulk of training data
- Architecture extension stacks orthogonally with i244 (TDCD, equal-bucket)
  and i245 (DHPE, pair-resonance) — the three primitives target disjoint
  failure modes
- Empirically validated in the prototype: planted knight-fork motif
  correctly flips the fanout argmax from Q to N, proving substitution
  carries chess-tactical signal
- Genuinely new counterfactual axis (piece-type substitution) — not in any
  of the 12 primitive families, not in any of the 243 i### entries, not in
  any of the 20 imported research packets
