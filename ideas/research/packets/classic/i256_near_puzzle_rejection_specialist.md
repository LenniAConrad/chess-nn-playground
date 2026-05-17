# i256 Near Puzzle Rejection Specialist

## Thesis and failure mode

### Thesis

The architecture should be optimized for **rejecting verified near-puzzles at fixed verified-puzzle recall**, not for maximizing overall PR-AUC in isolation. In this repo, the `puzzle_binary` benchmark is explicitly defined so that source class `1` is a **verified near-puzzle** but still maps to binary target `0`, and the benchmark goal states that the real question is whether a model can separate true puzzles from positions that look similar, especially hard negatives that are close to puzzle positions but are not puzzles. The repo also keeps a rectangular `3x2` diagnostic matrix specifically so near-puzzle behavior remains visible instead of being hidden inside a generic binary confusion matrix. fileciteturn9file0L3-L3

The current repo evidence says this is primarily an **operating-point** problem. In the matched-recall report, `i193_exchange_then_king_dual_stream` has the best reported near-puzzle false-positive rate at recall `0.80` with `0.128`, while `i018_oriented_tactical_sheaf_laplacian` is at `0.150`; at recall `0.85`, those become `0.165` and `0.186`. The same report repeatedly identifies `equal` eval, `hard`, `very_hard`, `promotion`, `underpromotion`, and `mate_in_1` as weak slices. The audit also notes that one model can win a slice on fixed-recall near-puzzle rejection while another wins the same slice on PR-AUC, so the deployment-relevant headline cannot be a single global ranking metric. fileciteturn10file0L3-L3 fileciteturn14file0L3-L3

My recommendation is therefore a **board-only specialist rejection head** that can sit on top of either i018 or a fast conv student and answer one explicit chess question:

> Does the strongest tactical claim survive the opponent’s best local defensive reply?

That direction is strongly supported by the repo’s internal evidence. A repo-local bridge proposal already argues that near-puzzles often contain a tempting forcing move but still admit a safe reply, and the repo already contains component ideas for safe-reply certificates, reply-set entropy, exchange soundness, king-escape pressure, move-landscape concentration, and counterfactual defender removal. The strongest path is to assemble those ideas into one coherent near-puzzle rejection specialist instead of trying to squeeze the behavior out of a generic negative classifier. fileciteturn20file0L3-L3 fileciteturn22file0L3-L3 fileciteturn24file0L3-L3 fileciteturn31file0L3-L3 fileciteturn33file0L3-L3 fileciteturn35file0L3-L3 fileciteturn37file0L3-L3

### Definition of the near-puzzle failure mode

A **near-puzzle failure** should be defined operationally as follows:

A board is a near-puzzle failure when the model emits a high positive puzzle claim because it detects tactical **texture**—checks, hanging material, exposed kings, promotion tension, pins, x-rays, overloaded-looking defenders, or one visually dominant move—but the position still contains at least one **surviving defensive resource** that prevents the tactic from being genuinely forced. The model is therefore mistaking *tactical appearance* for *forced tactical necessity*. This is exactly the failure the benchmark was built to surface. fileciteturn9file0L3-L3 fileciteturn10file0L3-L3

That definition also matches standard chess-engine reasoning. Static exchange evaluation studies whether an apparently tactical capture remains favorable after likely recaptures on the same square. King-safety evaluation is built from king-zone square control and attack units rather than from raw “sharpness.” Pins matter because they restrict movement and create contingent obligations, x-rays matter because they encode hidden slider dependencies, and safe mobility matters because a defender with surviving safe squares is different from a defender that is actually trapped. citeturn6view0turn14view2turn14view3turn9view0turn9view2

A useful failure taxonomy is:

| Failure family | What the model sees | What is actually true | What the specialist must learn |
|---|---|---|---|
| Equal-eval mirage | Tactical tension with no obvious eval edge | Best attack and best defense are close | Small or negative forcedness gap |
| Hard or very-hard mirage | Many plausible tactical continuations | Only one line, or no line, is truly forcing | Reply pressure and concentration must disagree |
| Promotion mirage | Passed-pawn or promotion-square pressure | Stop resource or counterplay survives | Promotion-lane reply and overload reasoning |
| Mate-in-1 mirage | A check exists | Escape square or interposition survives | King-escape pressure veto |
| Overload mirage | Defender looks burdened | Defender still has enough safe coverage | Obligation budget, not just attack count |

That table is a synthesis of the repo’s current matched-recall and slice evidence. fileciteturn10file0L3-L3 fileciteturn14file0L3-L3

## Architecture dataflow

The cleanest implementation is **not** a new monolith. It should be a **specialist rejection head with trunk adapters**, so it can attach to either i018 or to a fast conv student while staying inside the repo’s shared trainer, artifact contract, and board-only input rule. The repo documentation is explicit that CRTK tags and similar metadata are for reporting and slice analysis only, not model input, and the existing idea docs do the same for source labels, verification fields, PVs, and best moves. fileciteturn25file0L3-L3 fileciteturn26file0L3-L3 fileciteturn15file0L3-L3 fileciteturn39file0L3-L3

```text
board tensor
   │
   ├── side-to-move canonicalization / board adapter
   │
   ├── trunk adapter
   │      ├── i018 path: sheaf node states + relation energies + gates
   │      └── fast conv path: exchange/king or BT4-style square/context features
   │
   ├── deterministic chess feature builders
   │      ├── candidate move compiler
   │      ├── bounded reply envelope compiler
   │      ├── SEE / exchange-soundness field
   │      ├── king-zone / escape-pressure field
   │      └── overload / pin / ray-blocker field
   │
   ├── specialist heads
   │      ├── raw positive claim head
   │      ├── forcedness gap head
   │      ├── reply-pressure head
   │      ├── defender-overload head
   │      ├── king-escape pressure head
   │      └── candidate concentration head
   │
   ├── accepted-claim / veto fusion
   │
   └── final puzzle logit + diagnostics
```

The **i018 path** should reuse what already works: side-to-move canonicalization, typed attack/defense/ray/king-zone relations, learned sheaf restriction maps, per-relation energies, and relation gates. That keeps the new model faithful to the repo’s strongest architectural finding so far: chess-aware relation geometry is load-bearing, and i018’s architecture doc already exposes exactly the kind of typed attacker/defender/ray/king-zone structure that a near-puzzle rejection specialist wants to exploit. The surrounding sheaf literature also supports the idea that nontrivial sheaf geometry can give diffusion models more expressive control than ordinary graph diffusion in settings where class structure depends on structured, nonuniform relations. fileciteturn15file0L3-L3 fileciteturn8file0L3-L3 citeturn14view0turn14view1

The **fast conv path** should reuse a compact exchange-plus-king or BT4-style student. `i193_exchange_then_king_dual_stream` is the best current matched-recall performer and already decomposes evidence into exchange and king-safety streams, which is exactly the right fast parent for a rejection specialist. If the goal is speed, this is the most practical first deployment target. If the goal is maximum chess bias, the i018 parent is the better second target. fileciteturn21file0L3-L3 fileciteturn10file0L3-L3

The **candidate move compiler** should remain deterministic and board-only. The repo already contains a one-ply counterfactual move-landscape architecture that enumerates pseudo-legal moves, encodes move deltas, and pools move-set entropy, top-2 gap, move count, capture fraction, and promotion fraction from the current board alone. That is the right substrate for “candidate move concentration” because it adds move structure without adding engine search or metadata leakage. fileciteturn33file0L3-L3

## Chess-specific modules

### Forcedness gap module

This should be the central mechanism.

For each side-to-move candidate \(m\), compute a **positive claim** and a **reply escape**:

```text
claim(m)          = how tactical and decisive the candidate looks
reply_escape(m)   = how good the best local defensive reply family looks
forcedness_gap(m) = claim(m) - reply_escape(m)
```

The claim side should combine trunk context with move-local descriptors: move type, check flag, promotion flag, line opening or closing, king-zone overlap, touched pinned pieces, touched overloaded defenders, and a lightweight SEE-style exchange-soundness feature. The reply side should enumerate only bounded, local reply families: recapture, king escape, interposition, defend-target, promotion stop, and counter-threat. This is directly supported by the repo’s bridge proposal and by the two best reply-oriented specialist families already implemented in the repo. fileciteturn20file0L3-L3 fileciteturn22file0L3-L3 fileciteturn24file0L3-L3

The SEE descriptor matters especially for `equal` eval and `hard` slices. Static exchange evaluation is specifically about whether a local tactical operation remains sound after the likely exchange sequence on one square, and the repo already has an `exchange_soundness_graph_network` built around attacker intensity, defender intensity, defender gap, reply pressure, and a differentiable SEE field. That makes SEE the right local material-soundness signal for the forcedness gap instead of a generic learned “threat score.” fileciteturn35file0L3-L3 citeturn6view0

At aggregation time, I would not use a plain mean over candidates. Use **entmax or sparsemax** over `forcedness_gap(m)` so the model can represent “one forcing line dominates” without averaging the signal away. The aggregate diagnostics should include `max_forcedness_gap`, `top2_forcedness_gap`, `forcedness_gap_entropy`, `claim_mass`, `reply_escape_mass`, and `selected_candidate_count`. That gives both the model and the evaluator an explicit notion of forcedness instead of a hidden latent. fileciteturn20file0L3-L3

### Defender overload and reply-pressure heads

The repo’s `counterfactual_defender_dropout_network` is already built around the thesis that real puzzles hinge on a small set of causally critical participants—overloaded defenders, ray blockers, and the one escape square—and that deleting those participants should change the logit much more on a true puzzle than on a near-puzzle. That is exactly the right bias for this benchmark; it simply needs to be fused with a stronger parent trunk and a stronger reply model. fileciteturn37file0L3-L3

I would formalize **defender overload** as an obligation budget. For each defending piece \(d\):

```text
obligation_count(d) =
    critical targets defended by d
  + king-zone squares only safely covered by d
  + recapture duties on top candidate squares
  + ray-blocker duties on slider lines
  + promotion-stop duties

safe_budget(d) =
    safe mobility of d
  + alternative defenders for the same obligations

overload_margin(d) = obligation_count(d) - safe_budget(d)
```

The point is not merely “many attacks are good.” Pinned pieces have restricted mobility, x-rays create hidden dependencies, and safe mobility matters more than gross mobility when deciding whether a defensive resource truly survives. That is consistent with the engine-side discussion of pins, x-rays, and safe mobility. citeturn9view0turn9view2turn14view3

The **reply-pressure head** should summarize the bounded reply envelope with variables already used by the repo’s reply-set models: valid reply count, effective reply count, reply-family masses, top-1 safe-reply score, top-2 gap, and normalized entropy. A true puzzle should look like **high claim, compressed safe reply set**. A near-puzzle should look like **high claim, but at least one reply family survives strongly enough to veto acceptance**. fileciteturn24file0L3-L3 fileciteturn22file0L3-L3

### King safety and escape-square pressure

This head exists mainly to attack the repo’s `mate_in_1` weakness. On that slice, the model often does not miss the idea of check; it over-credits the checking idea while missing that the “mate” still leaks one king escape or one interposition. The repo already exposes the correct ingredients. `i193_exchange_then_king_dual_stream` includes king-zone pressure and escape-square features, while `king_escape_percolation_network` turns king-cage structure into escape-reachability maps, edge energies, reachable mass, escape asymmetry, and defense gap. Those are exactly the kinds of diagnostics a near-puzzle rejector should consume after a candidate move is proposed. fileciteturn21file0L3-L3 fileciteturn31file0L3-L3

This is also consistent with classical engine king-safety practice. Engine-style king-safety scoring uses king-zone square control and attack units because king danger depends on constrained local geometry, not on a generic visual impression that “pieces are near the king.” citeturn14view2turn7view2

### Candidate move concentration

Candidate concentration should be a **supporting** module, not the decision by itself. The repo’s move-landscape architecture already provides a board-only way to compute candidate-set entropy, top-2 gap, move count, capture fraction, and promotion fraction, and broader engine heuristics treat mobility and safe mobility as meaningful because the number and quality of available choices matter. fileciteturn33file0L3-L3 citeturn14view3

But concentration alone is not enough. A `mate_in_1` near-puzzle can still have one dominant checking move. What matters is **concentration of the forcedness gap**, not concentration of raw tactical appearance. So this head should consume the candidate-wise `forcedness_gap(m)` landscape and emit `gap_entropy`, `gap_top1_minus_top2`, `effective_forcing_candidate_count`, and `pseudo_legal_forcing_fraction`. Then the fusion rule becomes clear:

- **high concentration + positive max gap** supports acceptance,
- **high concentration + nonpositive max gap** strengthens the veto.

That preserves the user’s requested “candidate move concentration” idea without letting it collapse back into a generic puzzle-likeness score.

## Loss and sampling strategy for near-puzzle negatives

### Loss design

The objective stack should stay intentionally small. The repo audit is right that overcomplicated auxiliary stacks can tax optimization. I would use:

```text
L = L_main + λ_gap L_gap_rank + λ_veto L_veto
```

with

```text
L_main     = BCEWithLogits(final_logit, y_binary)
L_gap_rank = max(0, margin - gap_puzzle + gap_near)
L_veto     = BCEWithLogits(veto_logit, 1) on near-puzzles with high raw_claim
```

`L_main` keeps the model aligned with the actual benchmark contract. `L_gap_rank` encodes the ranking statement the benchmark really cares about: matched puzzles should have larger forcedness gaps than matched near-puzzles. `L_veto` trains the rejector only where it matters: examples that already generate a strong raw tactical claim and are therefore at real risk of becoming false positives. This is much closer to the repo’s evidence than generic hard-negative mining because the extra supervision is tied to explicit chess mechanisms. fileciteturn9file0L3-L3 fileciteturn20file0L3-L3

The pairwise term should reuse the logic already explored by the repo’s near-puzzle twin architecture. That idea argued that the benchmark’s core failure mode is ranking—real puzzles must score above near-puzzle hard negatives—and exposed pair-friendly latents specifically so the trainer could attach margin losses while keeping the model itself board-only. fileciteturn29file0L3-L3

For calibration, keep it mostly **post hoc**. Modern neural networks are often poorly calibrated, and temperature scaling remains a strong low-complexity default. That matters here because the main deployment metrics are thresholds at recall `0.80` and `0.85`, not just threshold-free ranking summaries. citeturn13view1

### Sampling strategy for near-puzzle negatives

The repo’s canonical tagged split is source-balanced, so warmup should stay close to source balance. But once the parent trunk has stabilized, the specialist head should move to a **chess-explained curriculum**, not a generic mining loop. The matched-recall report already tells us where to spend that budget: `equal`, `hard`, `very_hard`, `promotion`, `underpromotion`, and `mate_in_1`. fileciteturn10file0L3-L3

A practical curriculum is:

| Phase | Suggested composition | Purpose |
|---|---|---|
| Warmup | 1/3 random non-puzzle, 1/3 verified near-puzzle, 1/3 verified puzzle | Learn the overall benchmark contract |
| Specialization | 20% random, 40% verified near-puzzle, 40% verified puzzle | Shift capacity toward the real decision boundary |
| Slice focus | Within the near-puzzle quota, 2× weight for `equal`, `hard`, `very_hard`, `promotion`, `underpromotion`, `mate_in_1` | Attack the audited weak slices directly |
| Mirage replay | Extra replay queue for near-puzzles with high `raw_claim` but nonpositive `max_forcedness_gap` | Train the veto on chess-explained false-positive patterns |

Two rules make this legitimate. First, slice tags may be used for **sampling and reporting**, but not as inference inputs; that stays inside the repo’s contract. Second, replay should be keyed to the model’s own chess mechanisms—surviving king escape, surviving reply family, unresolved overload, low exchange soundness—not to a generic “highest loss” criterion. That preserves the chess explanation the user explicitly requested. fileciteturn25file0L3-L3

## Evaluation at recall 0.8 and 0.85

Global PR-AUC should still appear in the report, but it should become **secondary**. The repo audit already shows that promotion-slice PR-AUC and promotion-slice near-puzzle false-positive control can disagree, and the ML literature warns that PR-space behavior depends on skew. For this architecture, the primary scoreboard should therefore be **fixed-recall operating points selected on validation only**. fileciteturn14file0L3-L3 citeturn12academia1turn13view1

The protocol should be:

1. Train on the canonical tagged split with the repo’s reliable protocol and repeated seeds.
2. Fit one global temperature on validation logits.
3. Choose thresholds \(t_{0.80}\) and \(t_{0.85}\) on the **validation** set to hit puzzle recall `0.80` and `0.85`.
4. Freeze those thresholds and evaluate once on test.
5. Report both overall and slice metrics at both thresholds. fileciteturn25file0L3-L3 fileciteturn26file0L3-L3

The required overall table at each recall point should include:

| Metric | Why it belongs |
|---|---|
| Puzzle recall | This is the operating-point constraint |
| Precision | Accepted-puzzle reliability |
| Total false positives | Operational screening cost |
| Near-puzzle false positives | The primary benchmark failure mode |
| Near-puzzle FP rate | Threshold-stable comparison across runs |
| Far/random FP rate | Guards against indiscriminate conservatism |
| Mean `raw_claim_logit` on accepted positives | Separates claim from acceptance |
| Mean `reply_veto_logit` on rejected near-puzzles | Shows the veto path is genuinely used |

For concrete success criteria, use the current parents as floors. On an i018-compatible path, the first goal should be to beat i018’s current near-puzzle FP rates of `0.150` at recall `0.80` and `0.186` at recall `0.85`, with preferred first targets of roughly `≤ 0.140` and `≤ 0.175` while staying within about `0.003` PR-AUC of the parent trunk. On a fast conv-student path, the target should be at least a **5–10% relative reduction** in near-puzzle FP versus the parent at both recall points. Those targets are anchored directly in the current matched-recall report. fileciteturn10file0L3-L3

### Required weak-slice reports

The minimum required weak-slice report should include these slices at both recall thresholds:

- `crtk_eval_bucket = equal`
- `crtk_difficulty = hard`
- `crtk_difficulty = very_hard`
- `crtk_tactic_motifs = promotion`
- `crtk_tactic_motifs = underpromotion`
- `crtk_tactic_motifs = mate_in_1`

Those are the recurring weak spots across the best runs and especially around i018 and its neighbors. fileciteturn10file0L3-L3

For each slice, report:

| Column | Meaning |
|---|---|
| `n` | Slice size |
| `puzzle_recall` | Whether recall preservation still holds on the weak slice |
| `near_FP_rate` | Core rejection metric |
| `far_FP_rate` | Whether the model is becoming broadly conservative |
| `precision` | Practical acceptance quality |
| `accuracy@recall` | Keeps continuity with the repo’s audit style |
| `mean_max_forcedness_gap` | Mechanism visibility |
| `median_effective_reply_count` | Whether the reply-pressure head is behaving sensibly |
| `dominant_veto_family_share` | What the model says it used to reject the slice |

That last column matters. A near-puzzle specialist should not only win the slice table; it should say **why** it won. If `mate_in_1` gains come mostly from king-escape vetoes, or promotion gains come mostly from promotion-stop replies and defender-overload margins, that is exactly the kind of evidence that would make the result believable. fileciteturn31file0L3-L3 fileciteturn37file0L3-L3

## Ablations, expected speed cost, implementation sketch, and recommended experiment matrix

### Ablations

The ablation suite should be chess-semantic rather than arbitrary. I would require at least these:

| Ablation | What changes | What failure would mean |
|---|---|---|
| `parent_only` | Trunk with no specialist head | Establishes the honest baseline |
| `no_forcedness_gap` | Keep candidates and replies, but remove `claim - reply_escape` and use raw claim only | If this ties the full model, the specialist is not really about forcedness |
| `no_reply_envelope` | Zero out reply family scores | If this ties the full model, reply modeling is not load-bearing |
| `no_overload_head` | Remove obligation budgeting | If promotion and equal-eval do not worsen, overload modeling is cosmetic |
| `no_king_escape_head` | Remove escape-pressure head | If `mate_in_1` does not worsen, the escape story is false |
| `no_concentration_head` | Remove candidate concentration features | If hard and very-hard do not worsen, concentration is not useful |
| `uniform_sampler` | Disable slice-aware curriculum | If performance is unchanged, the curriculum is not worth the complexity |
| `threshold_0.5_only` | Skip validation calibration and use 0.5 | Shows whether fixed-recall gains are mostly calibration or mostly representation |

Every one of these ablations corresponds to a falsifiable chess claim. That is the right shape for this research direction. fileciteturn20file0L3-L3

### Expected speed cost

The repo already warns that parameter count and FLOPs are poor proxies for latency. Its knowledge base reports that, in the CPU harness, the BT4 classifier is about **6.4× faster** than i018 base at batch size 1 despite having far more parameters, because i018 pays heavily for many small irregular operations. Any near-puzzle specialist that adds another pile of irregular per-move logic on top of i018 will therefore lose on practical speed unless it is tightly capped and vectorized. fileciteturn8file0L3-L3

My expected cost envelope is:

| Deployment path | Added params | Expected latency delta | Recommendation |
|---|---:|---:|---|
| i018 + specialist head, capped `K=24`, `R=8` | ~30k–70k | ~20–35% GPU, ~35–60% CPU | Good if best accuracy matters more than throughput |
| Fast conv student + specialist head, capped `K=24`, `R=8` | ~40k–80k | ~10–20% GPU, ~15–30% CPU | Best first deployment target |
| Full paper-grade setting `K=48`, `R=12` | ~60k–120k | ~35–60% GPU, potentially worse on CPU | Use only if the smaller envelope clearly works |

Those are engineering estimates, not measured repo benchmarks, but they are consistent with the repo’s observed speed shape and with the fact that the specialist head adds bounded local candidate and reply reasoning rather than full search. The first implementation should therefore be aggressively vectorized, GPU-first, and hard-capped in candidate and reply counts. fileciteturn20file0L3-L3 fileciteturn33file0L3-L3

### Implementation sketch

This proposal fits naturally into the repo’s existing structure. The pipeline docs show how to add a new `torch.nn.Module`, register a builder, add configs, and rely on the shared artifact pipeline for prediction parquets, calibration plots, confusion matrices, speed summaries, and run reports. fileciteturn25file0L3-L3 fileciteturn26file0L3-L3 fileciteturn27file0L3-L3

A concrete layout would be:

```text
src/chess_nn_playground/models/trunk/near_puzzle_rejection_specialist.py
ideas/registry/i256_near_puzzle_rejection_specialist/
  architecture.md
  idea.yaml
  config.yaml
  model.py
  ablations.md
  implementation_notes.md
configs/benchmarks/puzzle_binary/i256_near_puzzle_rejection_specialist_*.yaml
```

Inside the main trunk file, define these reusable components:

```python
class TrunkAdapter(nn.Module): ...
class CandidateMoveCompiler(nn.Module): ...
class ReplyEnvelopeCompiler(nn.Module): ...
class ForcednessGapHead(nn.Module): ...
class DefenderOverloadHead(nn.Module): ...
class KingEscapePressureHead(nn.Module): ...
class CandidateConcentrationHead(nn.Module): ...
class NearPuzzleVetoFusionHead(nn.Module): ...
class NearPuzzleRejectionSpecialist(nn.Module): ...
```

The forward path should look like this, conceptually:

```python
def forward(self, x):
    trunk = self.trunk_adapter(x)              # i018 or conv student
    cand  = self.candidate_compiler(x, trunk)
    rep   = self.reply_compiler(x, cand, trunk)

    claim = self.claim_head(cand, trunk)
    reply = self.reply_head(rep, cand, trunk)
    gap   = claim - reply

    overload = self.overload_head(x, cand, rep, trunk)
    king     = self.king_escape_head(x, cand, rep, trunk)
    conc     = self.concentration_head(gap, cand)

    raw_claim = self.parent_head(trunk, claim, gap, conc)
    veto      = self.veto_head(reply, overload, king, conc, trunk)
    final     = raw_claim - F.softplus(veto)

    return {
        "logits": final,
        "raw_claim_logit": raw_claim,
        "reply_veto_logit": veto,
        "max_forcedness_gap": gap.max(dim=-1).values,
        "effective_reply_count": rep.effective_count,
        "candidate_gap_entropy": conc.entropy,
        "defender_overload": overload.score,
        "king_escape_pressure": king.score,
    }
```

The shortest honest reuse path is:

- reuse i018’s canonicalization and typed relation outputs when `trunk_kind="i018"`; fileciteturn15file0L3-L3
- reuse i193-style exchange/king or BT4-style student features when `trunk_kind="conv_student"`; fileciteturn21file0L3-L3
- reuse i025-style pseudo-legal candidate enumeration and move-landscape pooling; fileciteturn33file0L3-L3
- reuse i191 and i192 reply-family structure; fileciteturn22file0L3-L3 fileciteturn24file0L3-L3
- reuse i051 escape-pressure ideas; fileciteturn31file0L3-L3
- reuse i187 exchange-soundness and reply-pressure cues; fileciteturn35file0L3-L3
- optionally keep an i011-style selective fusion head for interpretability. fileciteturn39file0L3-L3

### Recommended experiment matrix

The first matrix should be small enough to complete and strong enough to answer the real design questions.

| Run family | Trunk | Key change | Purpose | Pass condition |
|---|---|---|---|---|
| `P0_parent_i018` | i018 | no specialist | Honest parent baseline | Reproduce current parent metrics |
| `P1_i018_gap_only` | i018 | forcedness gap only | Test whether reply-aware gap helps on its own | Better near-FP at matched recall |
| `P2_i018_gap_escape` | i018 | gap + king-escape head | Target `mate_in_1` and equal-eval | Slice gains on `mate_in_1` or `equal` |
| `P3_i018_full` | i018 | full specialist | Best accuracy-first variant | Best i018-path operating point |
| `C0_parent_student` | fast conv student | no specialist | Fast baseline | Reproduce parent metrics |
| `C1_student_full` | fast conv student | full specialist | Best speed/quality candidate | ≥5% relative near-FP reduction |
| `C2_student_full_uniform` | fast conv student | full specialist, no slice curriculum | Test sampler value | Worse weak-slice performance than `C1` |
| `A0_no_reply_envelope` | best trunk | ablation | Is reply modeling load-bearing? | Near-FP worsens |
| `A1_no_king_escape` | best trunk | ablation | Is `mate_in_1` gain real? | `mate_in_1` worsens |
| `A2_no_overload` | best trunk | ablation | Is overload reasoning real? | Promotion or equal worsens |
| `A3_threshold_0.5_only` | best trunk | no calibration | Is fixed-recall calibration necessary? | Operating-point quality worsens |
| `R0_best_reliable` | best trunk | 3 seeds, reliable budget | Promotion-grade evidence | Confidence intervals at recall `0.80` and `0.85` |

The training budget should follow the repo’s reliable protocol rather than scout shortcuts: canonical tagged split, NVIDIA path, repeated seeds, enough epochs for convergence, validation-only threshold selection, and full artifact validation. The pipeline docs and audit both say paper-grade claims require matched baselines, repeated seeds, slice analysis, ablations, and confidence intervals. fileciteturn25file0L3-L3 fileciteturn14file0L3-L3

The single best first bet is **`C1_student_full`**: a fast conv student parent with the full specialist head, capped candidate and reply counts, validation temperature scaling, and the chess-explained near-puzzle curriculum. If it delivers a cheap fixed-recall near-puzzle FP reduction, then `P3_i018_full` becomes the accuracy-maximizing follow-up. If it does not, that is strong evidence that the next meaningful gain will need more pretraining or more data rather than more hand-built rejection logic. fileciteturn8file0L3-L3 fileciteturn20file0L3-L3

## Open questions and limitations

This proposal is high-confidence in **direction**, but some implementation details remain empirical rather than proven. The main unresolved question is whether the reply envelope should stay bounded and local or whether the benchmark slice failures require slightly richer legality handling. Another open question is whether the i018 path can keep the specialist head fast enough in practice, because the repo’s own speed evidence shows that i018 is latency-sensitive to irregular operations. Finally, the best calibration strategy may turn out to be simple global temperature scaling rather than any more expressive calibrator, because the validation set size on the canonical split is finite and post-hoc calibrators become less robust as flexibility increases. fileciteturn8file0L3-L3 citeturn13view1

Even with those limitations, the evidence is already strong enough to recommend a clear architecture: **a trunk-compatible, board-only, forcedness-gap rejector that treats near-puzzles as chess-structured counterexamples, not as generic negatives.**