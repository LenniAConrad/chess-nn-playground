# Ablations

i254 is a capacity-scaled controlled-scaling study, not a new learning
hypothesis. The ablation matrix below is structured around the
research markdown's three axes: capacity allocation
(width / head / depth / stalk), restriction parameterization
(`full` vs `grouped_lowrank`), and an explicit *execution* axis that is
gated behind the parity ladder.

## Primary Comparison (5 cells, 3 seeds each)

| ID  | channels | hidden_dim | depth | stalk | restriction_mode | Hypothesis |
|-----|---------:|-----------:|------:|------:|------------------|------------|
| P1  | 128      | 192        | 4     | 8     | full             | Capacity falsifier (matched i018 scale_xl). Reference cell. |
| P2  | 160      | 320        | 4     | 8     | full             | Recommended first XXL: width+head-only, no parameterization confound. |
| P3  | 160      | 320        | 6     | 8     | full             | Depth diagnostic: does an extra two blocks help? |
| P4  | 160      | 320        | 4     | 12    | full             | Stalk diagnostic: does extra stalk capacity help with full maps? |
| P5  | 160      | 320        | 4     | 12    | grouped_lowrank  | Grouped-low-rank stalk: does parameter sharing make stalk scaling safer? |

## Falsifier Tranches (4 cells, 3 seeds each)

| ID  | Knob | Hypothesis |
|-----|------|------------|
| F1  | P2 with `model.scramble_relations=true` | i018 load-bearing-geometry falsifier under XXL scale: scrambled XXL must still drop PR-AUC by at least `0.02` vs intact P2. |
| F2  | P2 with i018 init seed (state_dict transfer) | Equivalence verification: i018 weights loaded into i254 (`strict=True`) must produce zero logit diff. |
| F3  | P2 with `model.restriction_mode=grouped_lowrank, model.restriction_rank=4` (same s=8) | Grouped-map falsifier at s=8: must beat seed noise to count. |
| F4  | P5 with `model.relation_groups=[0]*12` (single group) | Single-group degenerate: tests whether group structure matters or only the rank-k constraint matters. |

## Execution Branch (gated, 6 cells, 1 seed for prototyping)

These cells must pass the train-mode mixed-precision parity ladder
described in `math_thesis.md` before being benchmarked for speed.
Reported as wall-clock + memory only when parity holds.

| ID  | Switch | What it tests |
|-----|--------|---------------|
| E1  | i018 scale_xl + `compile_model=true` | Compile-only on the unmodified baseline. If this is the win, do not write custom code. |
| E2  | P2 + `compile_model=true` | Compile-only on the XXL candidate. |
| E3  | P2 + parity-checked algebraic-block (i249-style) | Re-runs the i249 execution path at i254 scale. Only kept if parity ladder passes in train mode. |
| E4  | P2 + bf16 autocast on inference | Inference-only speed lever. Must not regress logits beyond `1e-3`. |
| E5  | P2 + chunked relation loop | Process k=3 relations at a time. Only kept if parity holds. |
| E6  | P2 with `record_function` profile dump | Diagnostic only; produces the trace JSON for the math-thesis profile-first protocol. |

## Keep / Drop Rule

Keep i254 as the canonical i018 scale ladder if all of these hold:

- The 3-seed P2 capacity run beats the 3-seed P1 capacity falsifier
  by at least `+0.003` mean PR-AUC. This is the scale falsifier.
- F1 keeps i018's load-bearing-geometry result: scrambled XXL drops
  PR-AUC by at least `0.02` versus intact P2.
- F2 produces zero logit diff (this is also a hard correctness test
  in the test suite).
- No execution-branch cell is promoted unless it passes parity in
  train-mode mixed-precision with full dropout and configured AMP.

Drop the XXL conclusion (not the architecture) if:

- The scale falsifier fails. i018 has plateaued in this direction;
  the next move is a structural change, not a wider trunk.
- F3 / F4 both beat full maps. That suggests the family was
  over-parameterised at the original `s = 8`, and the right step is
  not "scale up" but "tighten the restriction family at the existing
  scale and re-evaluate".
- E1 produces a meaningful speedup. That means Python overhead was
  the bottleneck and the architectural conclusion about the relation
  loop should be revisited before any custom kernel work begins.
