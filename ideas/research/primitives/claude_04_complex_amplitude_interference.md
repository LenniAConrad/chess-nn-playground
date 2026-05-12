# 04 - Complex-Amplitude Interference Operator (CAIO)

**Slug:** `primitive_complex_amplitude_interference`
**Status:** proposed
**Author:** Claude
**Model:** Claude Opus 4.7
**Architecture extension:** i247 Complex-Amplitude Chess Network
**Documentation note:** this packet completes the missing Markdown entry referenced by this directory's README and manifest.

## One-line claim

A forward-pass primitive that lifts chess evidence fields into complex
amplitudes and measures constructive/destructive interference under
side-to-move, color-flip, square-color, and directed-relation phase shifts.

The goal is not to make a generic complex neural network. The goal is to give
the model a native operator for chess facts whose meaning changes under
involutions: side swap, color swap, tempo flip, and attack-defense direction.

## Mathematical signature

Let a trunk produce real board features:

```text
h in R^{B x C x 8 x 8}
```

CAIO splits the feature channels into magnitude and phase logits:

```text
rho = softplus(W_r h)
theta = W_t h + theta_rule(square, side, relation)
z = rho * exp(i theta)
```

The rule phase term is fixed or lightly learned from chess symmetries:

```text
theta_rule =
    a_side   * phase(side_to_move)
  + a_color  * phase(piece_color)
  + a_square * phase(square_color)
  + a_rel    * phase(relation_type)
```

For a relation mask `M_r[u, v]`, such as attack, defense, xray, king-zone, or
promotion-lane adjacency, CAIO computes interference:

```text
I_r(u, v) = Re( z_u * conj(z_v) * exp(i beta_r) )
D_r(u, v) = Im( z_u * conj(z_v) * exp(i beta_r) )
```

It then returns pooled diagnostics:

```text
constructive_r = sum_{u,v} M_r[u,v] * relu( I_r(u,v))
destructive_r  = sum_{u,v} M_r[u,v] * relu(-I_r(u,v))
phase_curl_r   = sum cycle residuals over short relation cycles
conj_error     = || z(color_flip(x)) - conj(z(x)) ||_2
```

The output is a compact vector of constructive mass, destructive mass, phase
curl, and conjugacy error by relation type.

## Why this is a primitive, not just a layer choice

Ordinary CNNs and attention layers treat sign, color, and side-to-move as
channels to be mixed. CAIO treats them as phase actions. That changes the
operator boundary:

| comparison | why CAIO is different |
|---|---|
| real-valued CNN | no complex phase, no interference term |
| standard complex neural network | no chess-rule phase tying or relation masks |
| Fourier features | phase is not spatial frequency; it is rule/symmetry state |
| equivariant group norm | checks invariance/equivariance, but does not score destructive interference |
| bilinear relation head | computes `u^T W v`, not `Re(z_u conj(z_v) e^{i beta})` with conjugacy constraints |

The primitive is most defensible if the phase action is load-bearing. The
central falsifier is therefore a phase-randomization or real-only ablation.

## Architecture extension - i247 Complex-Amplitude Chess Network

```text
simple_18 or lc0-style board planes
-> i193 exchange/king dual stream trunk
-> complex amplitude lift
-> fixed chess relation masks
-> CAIO interference pooling
-> small gated fusion with trunk logit
-> puzzle_binary logit
```

The first prototype should not replace the whole trunk. It should add CAIO as
a diagnostic side head on top of a known strong parent, preferably i193. That
makes the question clean:

```text
does phase interference add signal beyond the best current exchange/king trunk?
```

## Benchmark target

CAIO is the unorthodox member of this local primitive batch. It is not tied to a
single tactical motif the way PFCT targets promotion. Its intended target is
cross-slice coherence:

- color/side symmetry failures;
- tempo-dependent near-puzzle false positives;
- hard positions where real-valued evidence is present but incoherent across
  attack and defense relations;
- positions where constructive own-force and destructive defender interference
separate true puzzles from static tactical texture.

## Scout falsifier

Train i247 against i193 on the canonical `puzzle_binary` split.

Pass criteria:

- matched-recall near-puzzle false-positive rate improves by at least 0.01
  absolute at recall 0.80;
- aggregate PR AUC does not regress by more than 0.005;
- at least two hard slices improve, especially `equal`, `hard`, `mate_in_1`,
  promotion, or underpromotion;
- real-only and randomized-phase ablations lose at least half of the measured
  near-puzzle FP improvement.

Fail criteria:

- phase-randomized CAIO matches full CAIO;
- relation-mask-shuffled CAIO matches full CAIO;
- aggregate PR AUC drops by more than 0.01;
- gains appear only through extra parameters in a complex MLP control.

## Ablations

| id | variant | tests |
|---|---|---|
| A1 | real-only bilinear relation head | whether complex phase is load-bearing |
| A2 | random fixed phases | whether chess-rule phase assignments matter |
| A3 | learned free phases with no rule tying | whether the primitive is just extra parameters |
| A4 | shuffle relation masks | whether chess relations, not spatial coincidence, create the signal |
| A5 | remove conjugacy loss | whether color-flip structure is useful |
| A6 | use only constructive mass | whether destructive interference is necessary |
| A7 | use only phase curl | whether cycle inconsistency is the source of any gain |

## Risks

1. **Hidden rebrand as complex-valued neural network.** The defense is the
   rule-phase tying plus relation-masked interference readout. If free-phase
   controls match CAIO, downgrade the idea.
2. **Numerical instability.** Keep phases bounded with `atan2`-style real/imag
   normalization or represent complex values as two real channels.
3. **Overhead without targeted lift.** CAIO may be elegant but too broad. It
   should be dropped if it does not improve the hard near-puzzle slices.
4. **Symmetry overconstraint.** Some chess positions are intentionally
   asymmetric by side or color. Use a soft conjugacy penalty, not a hard
   equality constraint.

## Why it may be worth one scout run

The other three primitives in this directory are conservative counterfactual
operators. CAIO is the speculative option. It tests whether chess tactical
coherence is partly a phase-alignment problem: not just which facts are
present, but whether attack, defense, tempo, and color structure point in
compatible directions.

If CAIO wins only weakly, keep it as a diagnostic side head. If it wins with
the phase and relation ablations intact, it becomes a plausible future
architecture primitive for relation-heavy chess models.
