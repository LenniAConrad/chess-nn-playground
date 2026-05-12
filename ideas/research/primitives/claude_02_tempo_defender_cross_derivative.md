# 02 — Tempo-Defender Cross-Derivative Operator (TDCD)

**Slug:** `primitive_tempo_defender_cross_derivative`
**Status:** proposed
**Author:** Claude
**Model:** Claude Opus 4.7
**Architecture extension:** i244 Tempo-Defender Cross-Derivative Network

## One-line claim

A forward-pass primitive that computes the **mixed partial derivative** of a
learned scoring function with respect to two *different* perturbation axes —
**tempo flip** and **single-defender removal** — producing a per-defender
interaction signature that distinguishes true puzzles (insensitive to
specific defender removal) from near-puzzles (highly sensitive to the
*critical* defender's removal).

This was originally framed as an architecture (i244 TDCD) in an earlier
session, but the underlying *operator* — joint cross-derivative under two
chess-natural Z2 actions — is itself primitive-worthy and documented here
alongside the architecture.

## Mathematical signature

For a position `x` with side-to-move involution `sigma_T` (swap colors +
tempo bit) and per-piece-removal operators `delta_k` (zero the planes for
opponent piece at square `s_k`), and learned encoder `phi_theta : Board -> R^d`:

**Saliency stage** (selects K critical defenders):

    sal(x) = MLP_sal(phi_theta(x))         # per-opponent-piece criticality
    {k_1, ..., k_K} = top-K(sal(x))         # K=3 by default

**Cross-derivative grid** (2(K+1) shared-encoder forward passes):

    B0+ = phi_theta(x)                      # baseline, T+
    B0- = phi_theta(sigma_T x)              # baseline, T-
    Bk+ = phi_theta(delta_{k_i} x)          # defender removed, T+
    Bk- = phi_theta(sigma_T delta_{k_i} x)  # defender removed, T-

**Main effects**:

    g_T  = B0+ - B0-                        # baseline tempo gradient
    g_Dk = B0+ - Bk+                        # baseline defender-k gradient
    tau_k = Bk+ - Bk-                       # tempo gradient AFTER defender-k removed

**Mixed partial** (the central signal):

    DeltaDelta_k = ||tau_k|| - ||g_T||

`DeltaDelta_k` measures how much removing defender k changes the tempo
asymmetry of the position.

**Discriminator head**:

    z = [ ||B0+||, ||g_T||, max_k DeltaDelta_k, mean_k DeltaDelta_k,
          std_k DeltaDelta_k, topk(DeltaDelta) / ||g_T||,
          sal_entropy, sal_concentration ]
    puzzle_logit = MLP_head(z) + alpha * BCE_logit(B0+)

## Why this is genuinely new

Many ideas perturb the position to extract signal, but they all reduce to
**main effects of a single perturbation axis**:

| existing | axis | what it computes |
|---|---|---|
| i041 Centered Tempo-Odd | tempo flip tau + null board nu | scalars: tempo_odd_norm, null_odd_norm, centered_odd_norm |
| i049 Tempo-Odd Bottleneck | tempo flip only | tempo-odd scalar |
| i189 Counterfactual Defender Dropout | typed-mask dropout | 13-d sensitivity vector |
| i211 Role-Counterfactual Necessity | role-preserving counterfactual | necessity bottleneck |
| i025-i027 | one-ply move enumeration | move-landscape spectrum |
| i106 Attention Perturbation Sensitivity | attention map perturbation | sensitivity scalar |
| i222 Schur Defender Elimination | algebraic defender elimination | Schur-complement features |

**No registered idea computes the joint cross-derivative of tempo and
defender perturbations.** That's the signal needed: a true puzzle's
tempo-asymmetry should be *insensitive* to removing any single defender (the
tactic is distributed/forcing), while a near-puzzle's tempo-asymmetry should
*collapse* when the critical defender is removed.

The discriminating signature is the **mixed partial** `d^2 phi / dT dD_k`,
not `d phi/dT` or `d phi/dD_k` alone.

## Targeting the `equal` bucket

Decompose the discriminator's 2-D signature `(||g_T||, max_k DeltaDelta_k)`:

| region | reading | predicted class |
|---|---|---|
| `||g_T||` small | tempo flip changes little -> no forcing initiative for either side | non-puzzle |
| `||g_T||` large, `max DeltaDelta_k` small | tempo gives side-to-move a *distributed* tactic; no single piece supports it | **true puzzle** |
| `||g_T||` large, `max DeltaDelta_k` large at specific k* | tempo *appears* to give an initiative, but it rests on a single critical defender k* — remove that piece and the asymmetry collapses | **near-puzzle** |

Crucially, `||g_T||` is small for genuinely balanced equal-eval *non-puzzles*
even when `||B0+||` (baseline feature) is balanced too. The model can't use
static material to discriminate — it must use the tempo-response.

## Architecture extension — i244 TDCD Network

The encoder `phi_theta` is shared across all 2(K+1) passes — they batch as a
single forward of shape `(2(K+1)B, C, 8, 8)`. The discriminator consumes a
low-dimensional fingerprint of the cross-derivative spectrum.

Default trunk: i193's exchange/king dual-stream verbatim. This makes the
comparison "i193 baseline" vs "i193 trunk wrapped in TDCD" clean: if TDCD
wins on `equal` it's the cross-derivative head doing the work; if it
doesn't, the hypothesis is falsified at lowest cost.

## Cost

- 2(K+1) shared-encoder passes; at K=3 -> 8x forward cost vs single-pass
- Batch as 8B with reduced per-pass batch size; fits comfortably in 8 GiB
- Estimated wall-clock: ~6-8x i193 epoch time; 12 epochs x 173k positions ->
  roughly 3-5 GPU-hours on the 3070

## Falsifier

Train on canonical `puzzle_binary` split, 173k x 12 epochs, single seed (42),
single 3070.

**Primary pass criterion** (architecture is designed for this):

- `crtk_eval_bucket = equal` slice PR AUC >= **0.832** (i193 0.817 + 0.015)
- AND aggregate test PR AUC >= 0.871 (i193 - 0.005)
- AND no slice regresses > 0.01 vs i193

**Secondary checks**:

- The 2-D signature `(||g_T||, max_k DeltaDelta_k)` shows 3-cluster structure
  on held-out, with cluster assignments correlated with fine label (0/1/2)
- Saliency-head argmax on near-puzzle positives correlates with the actual
  defender resource (manually labelled on 50 sampled cases). >50%
  correlation expected

**Fail criteria**:

- Equal-slice PR AUC improvement < 0.005 (central hypothesis fails)
- Aggregate test PR AUC < 0.85 (the cross-derivative signal hurts overall)

## Ablations

| id | variant | tests |
|---|---|---|
| A1 | Disable mixed partial; use only `g_T` and `{g_Dk}` main effects | Whether the interaction term is the source of any win |
| A2 | K=1 vs K=3 vs K=5 | Whether multi-candidate matters |
| A3 | Replace learned saliency with i189's closed-form typed mask | Whether learned saliency beats geometric heuristic |
| A4 | Replace defender perturbation with attacker perturbation | Symmetry check |
| A5 | Replace per-defender perturbation with null-board nu (i041-style) | Whether *localised* perturbation matters or just global centring |
| A6 | Concatenate `B0+` directly into the head (skip cross-derivative pathway) | Sanity: shouldn't beat i193 alone if the head architecture isn't the source of the win |
| A7 | Tie phi_theta to i193 frozen weights vs train end-to-end | Whether trunk needs to be re-shaped by the loss |

## Risks

1. **Saliency collapse** — saliency head may collapse to "always pick the
   king" or "always pick the most valuable piece." Mitigate with entropy
   regulariser on `sal(x)` softmax; auxiliary loss on 5k positions where the
   critical defender is manually annotated.
2. **Hidden rebrand objection** — reviewer says "this is i041 + i189."
   Defence is the cross-derivative `DeltaDelta_k = ||tau_k|| - ||g_T||`,
   which neither computes. If A1 ablation matches the full network, that
   defence collapses.
3. **FP-rate vs PR-AUC mismatch** — TDCD is designed for PR AUC on a
   specific slice; check both metrics before claiming a win.
4. **Saliency leaks tactic-motif info** — saliency head should be trained
   only against puzzle_binary signal, not against `crtk_tactic_motifs` tags.
   Add data-loader audit assertion.

## Why high-potential

- Targets the **only universally-hard slice** (`equal` eval bucket) directly
- Mechanism aligns with what the audit said is needed ("new signal, not just
  a better trunk")
- Provably distinct from closest existing entries (i041, i189, i211): they
  compute first-order responses; TDCD's central feature is second-order
- Scout-scale falsifiable in ~5 GPU-hours; sharp empirical decision rule
- Reuses i193 as trunk so success is *additive*: TDCD over i193 either
  improves on i193 (keep both) or doesn't (keep i193 unchanged)
- Generates interpretable evidence (per-piece saliency + per-piece
  `DeltaDelta_k`) that maps to chess intuition about "the one defender that
  saves the position"
