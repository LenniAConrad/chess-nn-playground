# Math Thesis

Side-Canonical Rule-Partition Invariant Bottleneck (`SCRIB`)

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0732_tuesday_pdt_rule_partition_bottleneck.md`.

Working thesis: puzzle-likeness should be predicted from a side-relative
representation that is approximately invariant to the coarse rule
partitions of phase (total non-king material), side-relative material
advantage, and absolute side-to-move color. Concretely, the central
object is the rule-partition invariant bottleneck

```
B(x) = h(z),   z ~ q(z | C(x)),
```

trained with the minimax objective

```
min over (encoder, head) max over (env adversaries)
    L_cls + beta * L_KL + lambda * L_VREx - gamma * L_adv,
```

where:

- `C` is a deterministic side-to-move canonicalizer mapping the
  `simple_18` tensor to a 17-channel side-relative representation;
- `q(z | C(x))` is a Gaussian variational information bottleneck;
- `L_KL = E_x[KL(q(z|C(x)) || N(0, I))]` compresses `z`;
- `L_VREx = Var_e R_e` equalizes the per-environment classification risk
  across the 30 phase x adv x color groups;
- `L_adv = CE(E_phase, a_1(z)) + CE(E_adv, a_2(z)) + CE(E_color, a_3(z))`
  is implemented via a gradient-reversal layer so the encoder pushes `z`
  toward non-identifiability of the rule partitions while the adversary
  heads predict them.

The architecture lives in
`src/chess_nn_playground/models/rule_partition_invariant_bottleneck.py`
and is registered as `side_canonical_rule_partition_invariant_bottleneck`.
The full deterministic forward pipeline (canonicalizer, partitioner,
compact convolutional trunk, VIB, label head, gradient-reversed
adversary heads) is documented in `architecture.md`. The idea is
intentionally board-only: CRTK / source / engine / verification metadata
remain reporting-only and never enter the model input.
