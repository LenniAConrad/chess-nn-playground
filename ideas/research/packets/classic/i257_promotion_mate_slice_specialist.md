# i257 Promotion Mate Slice Specialist

*Intended filename: `i257_promotion_mate_slice_specialist.md`*

## Context and thesis

### Thesis

The highest-value design for this repo is **not** a new monolithic trunk. It is an **i018-family sheaf trunk with small, falsifiable specialist side heads** for three failure modes that the current audits already isolate: **promotion**, **underpromotion**, and **mate-like king-zone forcing**. The recommended implementation is an **i249-backed** variant of i018, because i249 is explicitly documented as having the **same math, parameters, and numerics** as i018 while changing only execution efficiency; that makes it the safest place to add specialists without changing the trunk’s inductive bias. The repo already uses this exact architectural move elsewhere: small additive gated heads on top of a strong trunk, with matched ablations and keep/drop rules, rather than replacing the trunk or training specialists in isolation. fileciteturn15file0L3-L3 fileciteturn20file0L3-L3 fileciteturn26file0L3-L3 fileciteturn35file0L3-L3

Concretely, I recommend an architecture I would name **i257 Promotion Mate Slice Specialist**: keep the i018/i249 sheaf geometry intact, expose its intermediate square and relation summaries, and add three bounded side branches that each produce a **small logit delta** under a **sparse specialist gate**. That is the best fit to the repo’s benchmark contract, which keeps CRTK metadata and tactic tags out of model inputs, maps near-puzzles to negative in the binary task, and evaluates models not only by aggregate PR-AUC but also by worst-slice behavior and near-puzzle false positives. fileciteturn6file0L3-L3 fileciteturn36file0L3-L3

If a smaller deployment model is needed, the correct compression path is **teacher-student distillation from the full i257 model**, not “specialist-only” training on slice labels. Distillation is a better fit because it preserves the global puzzle-vs-non-puzzle decision boundary while allowing the student to inherit the teacher’s slice-aware behavior; Hinton et al. explicitly frame distillation as a way to compress a larger teacher and also discuss the usefulness of specialist models for fine-grained confusions. citeturn5academia0

### Why current models struggle on these slices

The repo’s own scout audit already shows the problem clearly. In the per-class benchmark report, the best overall group reaches **0.876** PR-AUC overall, but only **0.812** on `mate_in_1` and **0.652** on `promotion` and `underpromotion`. For **i018** specifically, the report shows **0.861** overall, **0.764** on `mate_in_1`, and **0.555** on `promotion` and `underpromotion`. In other words, the target slices remain materially weaker even when the aggregate classifier looks strong. fileciteturn18file0L3-L3

The matched-recall false-positive report shows why this matters operationally. At recall **0.8**, `idea_i018_oriented_tactical_sheaf_laplacian_seed42` reports overall `near_puzzle_fp_rate` **0.150**, and its worst slices include `mate_in_1` accuracy **0.775** and `promotion` accuracy **0.801**. The same report also shows that, on the promotion/underpromotion motif slice, i018’s matched-recall `near_FP_rate` is **0.129**. So the specialist problem is not only about missed positives; it is also about **wrongly accepting puzzle-looking near-negatives** in exactly the regions where the trunk is under-specified. fileciteturn19file0L3-L3

Architecturally, the shortfall is understandable. i018 is strong because it canonicalizes side-to-move orientation, builds a typed tactical incidence complex, and diffuses square states through a learned sheaf Laplacian over 12 relation types, including attacks, defenses, king-ring empties, sliding rays, knight attacks, pawn attacks, and pin candidates. But it does **not** natively enumerate the **four legal promotion piece identities**, and it does **not** natively expose **exact legal reply structure** around checks. Its square encoder includes a fixed coordinate called `promotion_distance`, but that feature is just a board-coordinate scalar, not a pawn-conditioned representation of “which pawn is about to promote, on what square, to what type, with what tactical consequence.” Likewise, the model includes `king_ring_pressure`, but not explicit counts of legal checking moves, escape squares, interpositions, or captures of the checker. fileciteturn11file0L3-L3 fileciteturn12file0L3-L3 fileciteturn13file0L3-L3 fileciteturn14file0L3-L3

Promotion and underpromotion are especially hard because they are **counterfactual piece-type problems**. Before the move is played, the board only contains a pawn; the decisive information is latent in the legal fanout over `{Q, R, B, N}`. The repo’s i246 packet makes exactly this point: static encoders collapse a pre-promotion pawn to type `P`, hiding the future piece identity until the move is played, which is why PFCT-style promotion fanout can help. The same packet also notes that underpromotion currently shares the same CRTK slice in reporting, which explains why the audit tables list identical promotion and underpromotion PR-AUCs and why i257 should emit its own underpromotion diagnostics instead of trusting the benchmark tags alone. fileciteturn27file0L3-L3 fileciteturn18file0L3-L3

`mate_in_1` is hard for a different reason: it is not just “high king pressure.” It is “high king pressure **with no legal escape**.” The repo’s i248 packet and the forcing-response bottleneck research packet both converge on the same insight: near-puzzles often look tactical on the surface, but the missing variable is the **response envelope**—king exits, captures of the checker, interpositions, and counterchecks. A trunk that sees sheaf tension around the king but not explicit reply structure can still over-score puzzle-like negatives. fileciteturn20file0L3-L3 fileciteturn21file0L3-L3 fileciteturn30file0L3-L3

The good news is that the repo already shows the extension path is viable. In the i018 thesis file, the paper-grade 3-seed baseline for i018 is recorded at **0.8752** test PR-AUC, and several primitive hybrids improve that by about **+0.0056 to +0.0065** through **gated-logit fusion** rather than trunk replacement. Just as importantly, the i018 falsifier shows that scrambling the trunk’s typed relation masks drops mean PR-AUC by **0.0424**, which is strong evidence that the sheaf trunk’s chess geometry is load-bearing. The right architecture is therefore **keep the trunk, specialize the slices**. fileciteturn43file0L3-L3 fileciteturn41file0L3-L3

## Specialist architecture

### Architecture modules

The proposed i257 model should combine **one shared sheaf trunk** with **three bounded specialists** and **one tiny joint interaction branch**. This is also aligned with the repo’s primitive-stacking strategy, which explicitly recommends exporting small diagnostics from each primitive and using a **small regularized fusion MLP**, because a large fusion head would blur whether the gain came from the primitive or just extra capacity. Conditional computation via sparse gates is also a standard way to increase task-specific capacity without paying full compute on every example. fileciteturn35file0L3-L3 citeturn5academia2turn5academia1

| Module | Core inputs | Output | Purpose |
|---|---|---|---|
| **Shared sheaf trunk** | i249/i018 board-only trunk | base logit `z0`, square states `H`, relation energies `E` | Preserve side-to-move tactical geometry |
| **Promotion candidate field** | canonical pawns, occupancy, rays, king-zone overlap | per-pawn descriptors | Identify realistic promotion candidates and context |
| **Promotion fanout lite** | candidate descriptors + analytic Q/R/B/N attack deltas | `s_prom`, type attention, `δ_prom` | Model legal promotion identity directly |
| **Underpromotion divergence head** | same fanout + non-queen-vs-queen comparisons | `s_under`, `δ_under` | Rescue N/B/R wins and suppress queen-default errors |
| **King-zone forcing witness head** | checking moves, escape squares, checker-capture/interpose/countercheck summaries, king-ring sheaf features | `s_mate`, `δ_mate` | Distinguish real forcing patterns from superficial king pressure |
| **Promotion–mate joint branch** | promotion fanout + forcing witness overlap | `δ_joint` | Catch mating promotions and underpromotion mates |
| **Sparse specialist mixer** | base trunk summary + specialist summaries + uncertainty | final logit `z` | Add specialists only when confident and relevant |

The **shared trunk** should be the current `oriented_tactical_sheaf_fast` implementation rather than the older i018 file, because the repo explicitly says it is a pure execution optimization of i018 with the same numerics. That gives the specialist architecture more latency headroom while keeping the line of evidence clean. fileciteturn15file0L3-L3

The **promotion candidate field** should be i018-native rather than i193-native. Instead of re-running a full external trunk four times per promotion type, use the geometry that i018 already exposes: canonical pawns, occupancy, attack masks, rays, king-zone empties, and pin structure. For each own pawn within one or two pushes of promotion, compute a descriptor containing at least: normalized promotion distance, open-file/open-diagonal status, blocker on the promotion square, capture-promotion availability, support/attack balance on the destination square, whether the destination square enters the enemy king zone, and whether promoting there opens or closes a pin/ray. This extends i018’s existing generic square-level promotion coordinate into a pawn-conditioned feature family without using any metadata. fileciteturn11file0L3-L3 fileciteturn12file0L3-L3

The **promotion fanout lite** branch should then build four analytic type-conditioned descriptors per candidate, one each for `{Q, R, B, N}`. The key change from i246 is computational: i246 uses full counterfactual trunk passes over promoted boards on top of i193, which is correct for that trunk but too expensive as the first move for i018. In i257, the fanout can be approximated much more cheaply by recomputing **local typed attack deltas** on the promotion square using i018’s own rook-ray, bishop-ray, knight, king-zone, and visibility masks, plus sheaf square embeddings from the source and destination. That turns “promotion choice” into a learned decision over four compact descriptors rather than four full-board re-encodes. The i246 packet is still the right conceptual ancestor, because it correctly identifies the missing object as a promotion fanout over legal piece identities and insists on a **matched copy-baseline-fanout falsifier**. fileciteturn26file0L3-L3 fileciteturn27file0L3-L3 fileciteturn45file0L3-L3

The **underpromotion divergence head** should not merely reuse the promotion head. It needs to answer a different question: *when is queen-default wrong or fragile?* That means explicitly comparing each non-queen candidate against the queen candidate. The most useful learned features here are the ones the repo already hints at: knight-fork emergence, bishop diagonal mating cage, rook file/corridor pressure, and “queen overshoots into stalemate-ish or over-defended geometry.” That last pattern matters because i248’s terminal-state packet already treats “mating special” and stalemate-adjacent behavior as its own mechanism, and the promotion packet explicitly flags the need to audit capture-promotion and non-queen cases rather than assume queen is always best. fileciteturn21file0L3-L3 fileciteturn26file0L3-L3 fileciteturn27file0L3-L3

The **king-zone forcing witness head** should combine two kinds of information. The first is **exact one-ply rule summaries** derived only from the current board—features in the spirit of TSDP: legal checking move count, exact mate-in-1 flag, number of legal promotions that check or mate, total legal replies after a checking move, king escape-square count, checker-capture count, and interposition count. The second is **sheaf-native geometry**: local king-ring pressure, ray opening/closing, pin pressure near the king, and attack/defense imbalance. This hybrid is important because exact rule counts alone can become brittle, while geometry alone misses reply cardinality. The repo’s TSDP implementation and forcing-response packet give the safe contract: use current-board legal moves only, do not consume engine scores, best moves, PVs, source fields, or verification metadata, and prefer a precomputed cache once the feature version stabilizes. fileciteturn20file0L3-L3 fileciteturn21file0L3-L3 fileciteturn22file0L3-L3 fileciteturn34file0L3-L3 fileciteturn30file0L3-L3

The **promotion–mate joint branch** is worth keeping, but it must stay tiny. Some of the hardest long-tail examples are not “promotion” or “mate” separately; they are **mating promotions**, including underpromotion mates. The TSDP packet already exposes this conjunction as `mating_special_count`, so the repo has a precise precedent for treating this overlap as a distinct signal rather than hoping the two heads discover it independently. fileciteturn21file0L3-L3 fileciteturn22file0L3-L3

### Math for specialist gates

Let the shared trunk map a board `x` to square states `H(x) ∈ ℝ^{64×d}`, relation summaries `E(x)`, and a base puzzle logit `z0(x)`. Let the specialist set be `K = {prom, under, mate, joint}`. The architecture should use **bounded deltas** and **sparse gates**:

```text
δk(x) = Δk * tanh(vkᵀ sk(x))
gk(x) = mk(x) * HC(akᵀ [hbase(x), sk(x), ck(x)] + bk)
z(x)  = z0(x) + Σk∈K gk(x) δk(x)
```

Here, `mk(x) ∈ {0,1}` is a **structural mask** that prevents impossible specialists from firing, `HC` is a hard-concrete gate, `sk(x)` is the branch summary, `ck(x)` is a branch-confidence scalar, and `Δk` is a fixed delta bound that keeps any one specialist from overwhelming the trunk. Hard-concrete gates are a good fit because they allow sparse, differentiable conditional computation with an expected `L0` penalty. citeturn5academia1

For the **promotion** branch, define a candidate set `C(x)` over own near-promotion pawns. For candidate `j ∈ C(x)` and promotion type `t ∈ {Q,R,B,N}`, define a compact type-conditioned descriptor `u_{j,t}` from trunk embeddings and analytic attack deltas. Then use sparse type attention:

```text
αj = sparsemax_t(qjᵀ kj,t)
pj = Σt αj,t V u_{j,t}
sprom(x) = ρprom({pj : j ∈ C(x)})
```

Using a sparse normalizer over `{Q,R,B,N}` is attractive here because the true promotion choice is often genuinely selective: most candidates should get zero or near-zero mass on implausible types. Sparsemax and entmax were designed exactly for this kind of selective attention, where interpretability and exact zeros can be more useful than dense softmax weights. citeturn6academia1turn6academia0

For the **underpromotion** branch, the key object is not the absolute non-queen score; it is the **non-queen margin versus queen**:

```text
sj,t      = wtᵀ u_{j,t}
rj        = max(sj,N, sj,B, sj,R) - sj,Q
sunder(x) = Σj ψunder([u_{j,Q}, u_{j,R}, u_{j,B}, u_{j,N}, rj])
```

If `rj` is strongly negative, the branch learns to stay silent; if `rj` is positive or near zero in a position with high tactical payoff near the king or on overloaded pieces, the branch can add a nontrivial `δ_under`. This is a direct architectural translation of the promotion packet’s central insight: the supervision problem is not “is there a pawn near promotion?” but “which future piece identity changes the tactical geometry?” fileciteturn27file0L3-L3

For the **mate** branch, treat checking and forcing patterns as a permutation-invariant witness set `W(x)`. A clean formulation is a Deep Sets encoder:

```text
um       = φ(wm)           for wm ∈ W(x)
smate(x) = ρ(Σm um)
```

where each witness `wm` contains only board-derived legal or geometric features, such as `is_check`, `is_capture`, `is_promotion`, `mate_in_1_flag`, `reply_count`, `escape_square_count`, `checker_capture_count`, `interpose_count`, `countercheck_count`, and local king-ring deltas. Deep Sets gives the right theoretical form for permutation-invariant set functions, and a Set Transformer is a sensible drop-in if later experiments show that interactions among witness moves matter more than simple additive pooling. citeturn1academia1turn1academia3

The **confidence term** `ck(x)` should be branch-specific and explicitly discourage noisy specialist contributions. For promotion and underpromotion, the obvious confidence signal is **type-attention concentration**—low entropy over `{Q,R,B,N}` and high fanout dispersion. For the mate branch, the analog is concentration over a small number of checking/forcing witnesses and a low normalized escape-square count. This confidence-conditioned gate is one of the main mechanisms that protects near-puzzle rejection: the specialist is allowed to speak strongly only when its internal decision is sharp. That follows both the repo’s own concern with near-puzzle false positives and the general specialist-model logic from distillation literature. fileciteturn19file0L3-L3 citeturn5academia0turn5academia1

If a distilled student is needed, the training loss should stay centered on the **main task**:

```text
L = LBCE(y, z)
  + λgate Σk E[gk]
  + λkd T² KL(σ(zteacher / T) || σ(zstudent / T))
  + λnear Lnear
  + λslice (Lprom-rank + Lmate-rank)
```

`Lslice` should be a **restricted ranking or contrastive term on the main puzzle label**, applied only on slice-defined training pairs; it should not become a separate supervised slice-classification problem. That preserves the user’s constraint: slice labels may shape sampling and pair construction, but the model should not be trained as if “promotion” or “mate_in_1” were the actual task. fileciteturn36file0L3-L3 citeturn5academia0

## Dataflow and training

### Dataflow

The cleanest internal dataflow is:

```text
board x
  ├─> i249 sheaf trunk
  │      ├─> square states H
  │      ├─> relation summaries E
  │      └─> base puzzle logit z0
  │
  ├─> promotion candidate field
  │      └─> near-promotion pawn descriptors C
  │
  ├─> promotion fanout lite on C
  │      └─> sprom, δprom, promo confidence
  │
  ├─> underpromotion divergence head on C
  │      └─> sunder, δunder, under confidence
  │
  ├─> king-zone forcing witness builder
  │      └─> witness set W
  │              └─> smate, δmate, mate confidence
  │
  ├─> tiny promotion–mate overlap branch
  │      └─> δjoint
  │
  └─> sparse specialist mixer
         └─> final logit z = z0 + gated deltas
```

This requires one practical refactor to the current i018/i249 code: the trunk should expose a `forward_features` path returning square states, relation energies, and base readout features before the final head. Right now `OrientedTacticalSheafNet.forward()` returns logits and diagnostics, but the internal objects needed for specialist heads are all already computed in the current implementation, so the engineering change is local and low-risk. fileciteturn13file0L3-L3 fileciteturn14file0L3-L3

For the **mate branch**, exact one-ply legal summaries should follow the same contract as TSDP: either compute them in a `torch.no_grad()` fallback from the current `simple_18` tensor, or, preferably, precompute and cache them by board hash and feature-version string so training throughput does not collapse. The repo’s i248 notes are very explicit that the precomputed path is the production path, while forward-time legal enumeration is a temporary fallback. fileciteturn20file0L3-L3 fileciteturn22file0L3-L3 fileciteturn34file0L3-L3

### Training protocol and sampling

Training should follow the repo’s **paper-grade protocol**: canonical tagged split, same label contract, same threshold-selection discipline, same matched baselines, and at least the standard convergence budget with 3 seeds for repo-level promotion claims. That means the default should stay close to the existing paper-grade pattern: `epochs: 20`, `min_epochs: 10`, `early_stopping_patience: 5`, mixed precision, TF32 allowed, and seeds `{42, 43, 44}`. Since the repo’s rules explicitly require reporting matched-recall false positives and worst slices, i257 should not be judged by global PR-AUC alone. fileciteturn36file0L3-L3

I recommend a **three-stage schedule**. First, warm-start from a converged i249 or i018 checkpoint and train only the specialist heads and mixer for 1–2 epochs with the trunk frozen. Second, unfreeze the final sheaf block and readout for joint fine-tuning at a lower learning rate. Third, if deployment cost matters, distill the resulting teacher into a smaller student with the same specialist interface. This is safer than cold-starting the whole architecture, because the i018 trunk is already demonstrably load-bearing and falsified against scrambled relations. fileciteturn43file0L3-L3 citeturn5academia0

Sampling should be **slice-aware as a training strategy, not as an input feature**. The main batch still needs global BCE supervision on `puzzle_binary`, but within that batch the positive sampler should upweight `promotion`, `underpromotion`, and `mate_in_1` positions, while the negative sampler should overrepresent **near-puzzle negatives** that show the same kind of surface signals. For example, promotion positives should be contrasted disproportionately against negatives that also have near-promotion pawns; mate positives should be contrasted disproportionately against negatives that also have checking moves or severe king-zone pressure. This is the right way to stop the specialist from learning “near-promotion implies puzzle” or “king exposure implies puzzle,” which is exactly the false-positive failure mode the repo is already tracking. fileciteturn19file0L3-L3 fileciteturn36file0L3-L3

A practical batch policy would be: keep the repo’s balanced label weighting, but within positive examples use capped oversampling, such as an effective **2–4× weight** for promotion/underpromotion/mate slices, and within negative examples target a batch composition where **near-puzzle negatives are at least half of all negatives**. The cap matters. The point is to make the slices visible without letting them dominate the main task. If validation shows aggregate PR-AUC or near-puzzle FP degrading, reduce the slice weighting before touching the architecture. That is consistent with the user’s constraint to avoid training only on slice labels. fileciteturn36file0L3-L3

Losses should stay modest. The primary loss remains BCE on the final logit. Add only three structural terms: a gate sparsity penalty, a small near-puzzle margin or reweighting term, and limited slice-restricted ranking losses. Avoid auxiliary heads that directly learn CRTK tactic tags. Those tags are for **sampling and reporting**, not for teaching the model a second, potentially brittle task. The repo’s own docs repeatedly enforce that tactic tags, source fields, and verification metadata are reporting-only and must not enter the model as features. fileciteturn6file0L3-L3 fileciteturn36file0L3-L3

Finally, any precomputed exact-rule cache must ship with the same kind of leakage guards recommended in the forcing-response packet: reject forbidden keys such as source, site, engine, eval, PV, mate score, best move, and verification fields at feature-build time and again before collation. That keeps the specialist architecture rule-aware, not metadata-aware. fileciteturn30file0L3-L3

## Evaluation plan

### Metrics

The official metric set should remain exactly what the repo already cares about: **global PR-AUC**, **slice PR-AUC**, and **near-puzzle false positives at matched recall**. The reliable training protocol explicitly says that a model is only interesting if it improves one of these without breaking the others, especially on `hard`, `equal`, `endgame`, `mate_in_1`, `promotion`, and `underpromotion`. fileciteturn36file0L3-L3

For comparability, report the official slices first. The current scout report gives a useful baseline: roughly **0.861 / 0.764 / 0.555 / 0.555** for i018 on overall / mate_in_1 / promotion / underpromotion, and the best overall scout model is still only **0.812 / 0.652 / 0.652** on the target slices. These are the numbers i257 must improve. At matched recall **0.8**, i018’s seed42 overall `near_puzzle_fp_rate` is **0.150**, and the promotion/underpromotion motif slice shows **0.129**. A viable i257 should therefore be judged against both slice lift and FP non-regression. fileciteturn18file0L3-L3 fileciteturn19file0L3-L3

I would use the following **keep thresholds** for a scout-to-paper pipeline:

| Metric | Minimum keep rule |
|---|---|
| Aggregate test PR-AUC | no regression worse than **-0.005** versus the matched i018/i249 baseline |
| `promotion` slice PR-AUC | **+0.03** absolute over matched i018 baseline |
| `underpromotion` slice PR-AUC | **+0.03** absolute over matched i018 baseline |
| `mate_in_1` slice PR-AUC | **+0.04** absolute over matched i018 baseline |
| Overall near-puzzle FP rate @ recall 0.8 | no worse than baseline |
| Promotion-slice near-puzzle FP rate @ recall 0.8 | improve or stay within noise |
| `equal` eval bucket | no regression worse than aggregate threshold |

Those thresholds are strict enough to matter but still aligned with the repo’s existing keep/drop rules around `±0.005` aggregate non-regression and meaningful slice lifts for specialist primitives. fileciteturn45file0L3-L3 fileciteturn46file0L3-L3

Because the current reporting appears to collapse promotion and underpromotion into effectively the same CRTK slice, i257 should also emit **board-derived secondary diagnostics** for better analysis. At minimum, report: `promotion_has_candidate`, `promotion_best_type`, `promotion_attention_entropy`, `underpromotion_margin`, `mating_special_count`, `escape_square_count`, `checking_move_count`, and each branch’s gate and logit contribution. These do not replace the official benchmark; they make it possible to tell whether a promotion gain came from real non-queen reasoning or from a generic “pawn on the 7th” prior. fileciteturn18file0L3-L3 fileciteturn21file0L3-L3 fileciteturn22file0L3-L3 fileciteturn33file0L3-L3

### Falsifiers and ablations

The repo has already established the right falsifier culture for this family: matched controls, small mechanism-specific ablations, and branch-level keep/drop decisions. i246 requires a matched `copy_baseline_fanout` falsifier for promotion fanout; i248 requires `shuffle_tsdp` for rule-derived mate features; and the i018 thesis itself rejects geometry changes if scrambled relations do not hurt. i257 should inherit that discipline directly. fileciteturn45file0L3-L3 fileciteturn46file0L3-L3 fileciteturn40file0L3-L3 fileciteturn43file0L3-L3

| Ablation | What it tests | Drop rule |
|---|---|---|
| **`trunk_only`** | specialist-free i249/i018 baseline inside same wrapper | sanity baseline |
| **`copy_baseline_fanout`** | replace Q/R/B/N promotion fanout with repeated baseline feature | if within **0.005** of full on promotion/underpromotion, drop promotion branch |
| **`uniform_type_attention`** | remove selective promotion-type weighting | if it matches full, simplify the branch |
| **`zero_under_margin`** | remove nonqueen-vs-queen comparison features | if it matches full, underpromotion branch is decorative |
| **`shuffle_mate_witness`** | decouple forcing witness set from the position | if it matches full on `mate_in_1`, drop mate branch |
| **`redact_exact_mate_flag`** | remove direct `mate_in_1` bit but keep reply/escape counts | if gains vanish completely, branch may be too brittle |
| **`disable_gate`** | force all active specialists open | if near-puzzle FP rises, the gate is load-bearing |
| **`force_zero_gate`** | set all specialist gates to zero | wrapper should recover trunk baseline closely |
| **`no_slice_sampler`** | turn off slice-aware sampling | measures training-strategy contribution |
| **`no_near_neg_weight`** | remove near-puzzle negative emphasis | tests whether FP control is genuinely structural |

The most important philosophical rule is **branch-local drop, not whole-model drop**. If the promotion falsifier fails, remove the promotion branch and keep the mate branch if it still passes. The primitive-stacking strategy in the repo says exactly this: build each primitive as a side head, prove its own ablation is load-bearing, then add only the winners to the stack. fileciteturn35file0L3-L3

## Implementation sketch

### Implementation sketch

The most direct implementation is a new trunk file that imports the existing i249/i018 internals unchanged and adds specialist modules after a `forward_features` refactor:

```python
class PromotionMateSliceSpecialist(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.trunk = OrientedTacticalSheafFastWithFeatures(...)
        self.promo_candidates = PromotionCandidateField(...)
        self.promo_head = PromotionFanoutLite(...)
        self.under_head = UnderpromotionDivergenceHead(...)
        self.mate_builder = KingZoneWitnessBuilder(...)
        self.mate_head = MateWitnessSetHead(...)
        self.joint_head = PromotionMateJointHead(...)
        self.mixer = SparseSpecialistMixer(...)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        trunk = self.trunk.forward_features(x)
        # trunk: base_logit, h, energy, incidence, board_state, base_summary

        promo = self.promo_candidates(trunk.board_state, trunk.incidence, trunk.h)
        prom_stats = self.promo_head(promo, trunk)
        under_stats = self.under_head(promo, trunk)

        mate_witness = self.mate_builder(x, trunk)   # exact-rule cache if available
        mate_stats = self.mate_head(mate_witness, trunk)

        joint_stats = self.joint_head(prom_stats, mate_stats, trunk.base_summary)

        out = self.mixer(
            base_logit=trunk.base_logit,
            base_summary=trunk.base_summary,
            prom=prom_stats,
            under=under_stats,
            mate=mate_stats,
            joint=joint_stats,
        )

        return {
            "logits": out.logits,
            "base_logit": trunk.base_logit,
            "specialist_gate_promo": out.gate_promo,
            "specialist_gate_under": out.gate_under,
            "specialist_gate_mate": out.gate_mate,
            "specialist_delta_promo": out.delta_promo,
            "specialist_delta_under": out.delta_under,
            "specialist_delta_mate": out.delta_mate,
            "promotion_best_type": prom_stats.best_type,
            "promotion_attention_entropy": prom_stats.attn_entropy,
            "underpromotion_margin": under_stats.margin,
            "mate_witness_count": mate_stats.witness_count,
            "escape_square_count": mate_stats.escape_count,
            "checking_move_count": mate_stats.check_count,
            **trunk.diagnostics,
        }
```

On the file-layout side, I would keep it boring and repo-native:

```text
src/chess_nn_playground/models/trunk/promotion_mate_slice_specialist.py
src/chess_nn_playground/data/i257_rule_features.py
ideas/registry/i257_promotion_mate_slice_specialist/
tests/test_i257_specialist_gates.py
tests/test_i257_promotion_fanout.py
tests/test_i257_mate_witness_cache.py
```

That follows the repo’s documented model-registration and idea-workflow contract, while keeping the architecture comparable under the shared trainer and artifact pipeline. fileciteturn6file0L3-L3

A minimal config should stay close to existing paper-grade defaults and make the specialist surface explicit:

```yaml
model:
  name: promotion_mate_slice_specialist
  trunk_name: oriented_tactical_sheaf_fast
  gate_kind: hard_concrete
  promotion_type_attention: sparsemax
  use_exact_oneply_cache: true
  max_promotion_candidates: 4
  specialist_delta_bound: 1.5

training:
  reliability_tier: paper_grade
  epochs: 20
  min_epochs: 10
  early_stopping_patience: 5
  class_weighting: balanced
  mixed_precision: true
  allow_tf32: true

sampling:
  upweight_promotion: 3.0
  upweight_underpromotion: 3.0
  upweight_mate_in_1: 2.5
  near_negative_min_fraction: 0.5
```

The only dataset-side addition should be a **versioned board-derived feature cache** for the one-ply mate witness summaries. If that cache is unavailable, the model may fall back to exact rule enumeration from `simple_18`, but that path should remain a fallback exactly as in the TSDP implementation notes. fileciteturn20file0L3-L3 fileciteturn22file0L3-L3 fileciteturn34file0L3-L3

## Final recommendation

### Final recommendation

The best next architecture for this repo is:

**Implement i257 as an `oriented_tactical_sheaf_fast` trunk plus three small bounded specialist branches—promotion, underpromotion, and mate-like forcing—mixed through sparse confidence-conditioned gates, with one tiny promotion–mate overlap branch and branch-local falsifiers.** That recommendation is grounded in four repo-specific facts: the slices are truly weak in the audits, the i018 trunk’s typed geometry is strongly load-bearing, the repo already has successful additive gated specialist primitives for promotion and mate on i193, and i018-side gated primitive hybrids have already shown meaningful aggregate lifts without discarding the trunk. fileciteturn18file0L3-L3 fileciteturn19file0L3-L3 fileciteturn43file0L3-L3 fileciteturn20file0L3-L3 fileciteturn26file0L3-L3

The strongest practical version is a **two-step rollout**. First, build the full i257 on **simple_18 + i249** so the mate branch can use exact board-derived one-ply summaries and the promotion branches can use exact candidate geometry. Second, if deployment speed matters, distill that teacher into a smaller student with the same specialist interface. This preserves generalization and keeps the user’s constraints intact: no metadata leakage, no giant opaque specialist head, no slice-only training objective, and no neglect of near-puzzle FP. fileciteturn20file0L3-L3 fileciteturn27file0L3-L3 fileciteturn35file0L3-L3 citeturn5academia0turn5academia1

If I had to reduce this to one sentence: **keep i018’s chess geometry, add only the missing counterfactual and legal-reply objects, and force every new slice specialist to earn its place with a matched falsifier.** That is the most repo-consistent, highest-expected-value path to improving promotion, underpromotion, and mate slices without giving back global PR-AUC or near-puzzle rejection. fileciteturn43file0L3-L3 fileciteturn45file0L3-L3 fileciteturn46file0L3-L3