# p046_promotion_underpromotion_primitive.md

## Thesis

The repo’s puzzle-binary benchmark is explicitly position-only: it maps a FEN-derived board tensor to one puzzle logit, and it treats verified near-puzzles as the central hard-negative class. The reporting standard also makes promotion and underpromotion pressure slices first-class evaluation targets, alongside aggregate PR AUC and matched-recall near-puzzle false positives. On the current audit tables, the strongest referenced i193-style group shows **0.876** overall test PR AUC but only **0.652** on both the `promotion` and `underpromotion` motif slices, while the best reported promotion-slice PR AUC among the top audited groups is **0.667**. That is a large slice-specific ranking gap, not a rounding error. fileciteturn54file0L1-L120 fileciteturn65file0L1-L200 fileciteturn53file0L1-L120

The same slice is also weak under the operating-point metric the repo cares about most. In the research audit, the best 3-seed promotion/underpromotion **near-puzzle false-positive rate at recall 0.80** is reported as **0.101 ± 0.015** for `i011_vetoselect_positive_claim_abstention_scale_xl`, versus **0.130 ± 0.027** for `bench_residual_small_lc0bt4_scale_xl` and **0.138 ± 0.006** for `bench_lc0_bt4_classifier_scale_xl`. The audit also warns that promotion-slice PR AUC and promotion-slice near-FP can disagree, so a useful primitive must improve both ranking quality and thresholded rejection behavior rather than optimizing one and ignoring the other. fileciteturn67file0L220-L430

A new primitive is justified because the repo’s existing promotion primitive, PFCT / `i246_promotion_aware_head`, is a stronger idea than a naive baseline but still leaves obvious geometric coverage gaps. Its own docs say it identifies **own** near-promotion pawns from `simple_18`, writes the promoted piece on the **same file**, and **does not enumerate diagonal capture promotions**. Its math note also says the substituted output is a valid one-hot board “regardless of whether the chess move would be legal,” and its trainer notes reduce batch size from **256 to 128** because the worst case expands work to `1 + K*4` trunk passes, with an estimated **1.4–1.5×** i193 wall-clock cost. That means the repo already has evidence that promotion matters, but not yet an exact, geometry-first, low-overhead promotion primitive. fileciteturn60file0L1-L220 fileciteturn58file0L1-L220 fileciteturn59file0L1-L260 fileciteturn61file0L1-L180

My thesis is therefore:

> **Build a Promotion and Underpromotion Geometry primitive that is exact about side-to-move pawn direction and candidate promotion squares, includes quiet and capture promotions, scores the promotion square with attack/defense and king-zone geometry, exposes underpromotion-as-deviation-from-queen features, and stays cheap enough to run at roughly i193 cost when features are precomputed.**

That thesis is rule-aligned. Under FIDE rules, a pawn moves forward one square if that square is empty, captures diagonally forward onto an occupied enemy square, and when it reaches the furthest rank it must be exchanged immediately for a queen, rook, bishop, or knight, with immediate effect. Those rules are exactly the geometry a board-only primitive should encode. citeturn9view0

The primitive should also be **board-only in the repo’s sense**. That means no CRTK motif tags, no source labels, no engine PVs, and no verification metadata as model inputs. The repo’s primitive plan, benchmark standard, and reliable-training protocol all make that restriction explicit. fileciteturn55file0L1-L180 fileciteturn65file0L1-L200 fileciteturn64file0L1-L260

## Promotion geometry definitions

I recommend calling the primitive **PUGP**: **Promotion and Underpromotion Geometry Primitive**.

The central design choice is **side-to-move canonicalization**. In `simple_18`, piece planes are fixed by color, the side-to-move indicator is plane 12, and board rows are indexed with rank 8 at row 0 and rank 1 at row 7. That makes white and black promotion directions different in raw tensor space, which is exactly where random pawn-direction bugs happen. Python-chess also documents a `mirror()` transform that flips the board vertically, swaps colors, and preserves position equivalence modulo color. PUGP should use the same conceptual move: canonicalize every sample so the side to move is treated as the “own” side moving toward a canonical promotion rank. fileciteturn63file0L1-L160 citeturn13view0

Define the canonical board as \(B^c = C(B)\), where \(C\) is the identity for white-to-move positions and a vertical mirror plus color swap for black-to-move positions. After canonicalization:

- own pawns always move “forward” toward canonical rank \(0\);
- opponent pawns always move away from canonical rank \(0\);
- the promotion rank is always canonical row \(0\);
- near-promotion own pawns are always on canonical row \(1\).

That one step should eliminate the most common failure mode in promotion feature code: getting black pawn direction wrong. It also makes every later equation color-stable. fileciteturn63file0L1-L160 citeturn13view0

A **candidate promotion move** is any side-to-move pawn move that lands on the last rank and therefore must promote under the chess rules. In canonical coordinates, for an own pawn at \((1,f)\), the legal target templates are:

- quiet promotion to \((0,f)\) if that square is empty;
- capture promotion to \((0,f-1)\) or \((0,f+1)\) if the square exists and is occupied by an enemy piece.

Those are the only geometrically possible promotion arrivals. Unlike PFCT, PUGP should not collapse all promotions to a same-file arrival square. citeturn9view0turn12view0turn13view0 fileciteturn60file0L1-L220

A **promotion square safety profile** should be defined on the *post-promotion board*, not the pre-move board, because the move changes occupancy and piece attack sets. Python-chess exposes exactly the primitives needed for this without engine search: `legal_moves`, `gives_check()`, `attackers()`, and `is_attacked_by()`. That means PUGP can remain board-only while still being rule-exact if desired. citeturn12view0turn13view0

An **underpromotion hint** should be defined as a **typed deviation from queening**, not as a motif label. In chess, queen is the default promotion because it is usually strongest; the primitive should therefore represent rook, bishop, and knight promotions primarily by how their local geometry differs from queen on the same arrival square. That preserves board-only modeling and avoids label leakage. The repo’s own i246 design already exports a `promotion_dominant_type` diagnostic for this reason, but its current construction undercovers legal move geometry. fileciteturn61file0L1-L180

One more evaluation caveat matters: the current audit says every `underpromotion` row in CRTK is also tagged `promotion`, so the two public motif slices are effectively one dataset slice today. PUGP should therefore export **internal diagnostics that separate quiet promotions, capture promotions, knight-favoring cases, and queen-dominant cases**, even if the benchmark’s external slice labels are still merged. fileciteturn67file0L220-L430

## Feature equations

Let \(P_{\text{own}}\) be the set of own pawns in canonical coordinates, and let \(r(p)\in\{0,\dots,7\}\) denote the canonical row of pawn \(p\).

The **side-to-move canonical promotion distance** for an own pawn is

\[
d_{\text{own}}(p) = r(p),
\]

so a pawn on canonical row 1 has \(d_{\text{own}}=1\), a pawn on row 2 has \(d_{\text{own}}=2\), and so on. For the opponent, define the analogous distance in the opponent’s own canonical frame, or equivalently compute a mirrored copy of the board and reuse the same function. PUGP should expose at least the following global distance summaries:

\[
D^{\min}_{\text{own}} = \min_{p\in P_{\text{own}}} d_{\text{own}}(p),
\qquad
D^{\min}_{\text{opp}} = \min_{q\in P_{\text{opp}}} d_{\text{opp}}(q),
\]

\[
N_{\text{own}}^{(k)} = \sum_{p\in P_{\text{own}}} \mathbf{1}[d_{\text{own}}(p)=k],
\qquad
N_{\text{opp}}^{(k)} = \sum_{q\in P_{\text{opp}}} \mathbf{1}[d_{\text{opp}}(q)=k],
\]

for \(k\in\{1,2,3\}\). These features keep the primitive informative on non-promotion positions too, which is important because the repo explicitly warns against building ideas that only win on a declared slice while harming global behavior. fileciteturn55file0L1-L180 fileciteturn65file0L1-L200

For each own near-promotion pawn \(p=(1,f)\), define its **candidate set**

\[
M(p)=M_{\text{push}}(p)\cup M_{\text{cap}}(p),
\]

with

\[
M_{\text{push}}(p)=\{(p\to (0,f)) : \text{occ}((0,f))=0\},
\]

\[
M_{\text{cap}}(p)=\{(p\to (0,f+\delta)) : \delta\in\{-1,+1\},\ 0\le f+\delta \le 7,\ \text{occ}_{\text{opp}}((0,f+\delta))=1\}.
\]

This is the exact board-rule template implied by FIDE Article 3.7. citeturn9view0

If exact legality is available from FEN or python-chess, define the **legal candidate subset**

\[
M^\star(p) = \{m\in M(p) : m \in \texttt{Board.legal\_moves}\}.
\]

If strict tensor-only operation is needed, expose both a **pseudo-legal** and a **rule-exact** version of the feature vector, and use the exact version for offline precompute. That lets the primitive distinguish “candidate exists geometrically” from “candidate is legal after self-check filtering.” Python-chess’s `legal_moves` interface is sufficient for this and does not require engine search. citeturn12view0

For each candidate \(m\in M^\star(p)\) and each promotion type \(t\in\{Q,R,B,N\}\), construct the post-move board \(B_{m,t}\). Then define the **promotion-square attack/defense features**

\[
a_t(m) = \left| \operatorname{attackers}_{\text{opp}}(u; B_{m,t}) \right|,
\qquad
d_t(m) = \left| \operatorname{attackers}_{\text{own}}(u; B_{m,t}) \right|,
\]

where \(u\) is the arrival square. A simple bounded safety score is

\[
s_t(m)=\frac{\operatorname{clip}(d_t(m)-a_t(m),-4,4)}{4}.
\]

These are the cheapest load-bearing features in the design because they directly represent whether a promoted piece arrives on a dominated or well-defended square. citeturn13view0

Let \(k_{\text{opp}}\) be the enemy king square on \(B_{m,t}\), and let the enemy king zone be

\[
Z(k_{\text{opp}})=\{z:\max(|r_z-r_k|,|f_z-f_k|)\le 1\}.
\]

Then define the **immediate promotion-check and king-zone features**

\[
c_t(m)=\mathbf{1}\!\left[k_{\text{opp}}\in A_t(u; B_{m,t})\right],
\]

\[
z_t(m)=\left|A_t(u; B_{m,t})\cap Z(k_{\text{opp}})\right|,
\]

where \(A_t(u; B_{m,t})\) is the attack set of a promoted piece of type \(t\) placed on \(u\). If legal-move generation is available, also define

\[
g_t(m)=\mathbf{1}[\texttt{Board.gives\_check}(m_t)],
\]

where \(m_t\) is the fully specified promotion move. This is the most direct board-only way to represent “king-zone promotion checks.” citeturn12view0turn13view0

The key underpromotion representation should be **delta-to-queen encoding**. For each non-queen type \(t\in\{R,B,N\}\), define

\[
\Delta_t(m)=
\Big[
c_t(m)-c_Q(m),\ 
z_t(m)-z_Q(m),\ 
s_t(m)-s_Q(m),\ 
\eta_t(m)-\eta_Q(m)
\Big],
\]

where \(\eta_t(m)\) is an optional move-quality proxy such as an enemy-king escape reduction or a low-cost legal-reply count on \(B_{m,t}\). The model then learns when rook, bishop, or knight differs *usefully* from queen on the same arrival square, without being handed any underpromotion labels. That is the cleanest way to encode “underpromotion tactical hints” while remaining board-only. citeturn9view0turn12view0turn13view0

Knight promotion deserves one extra scalar because queen does **not** subsume knight geometry. Let \(H_{\text{hi}}\) be enemy high-value targets, for example queens, rooks, bishops, knights, and optionally the enemy king zone. Then define

\[
\kappa_N(m)=\sum_{v\in H_{\text{hi}}} w(v)\,\mathbf{1}[v\in A_N(u; B_{m,N})].
\]

This is the **knight-fork hint**. It is the main typed feature that should fire when a non-queen promotion is tactically distinct rather than merely smaller. The repo’s own i246 notes already expect non-queen dominance on underpromotion and knight-fork puzzles; PUGP should encode that case explicitly instead of hoping trunk reruns discover it indirectly. fileciteturn61file0L1-L180

The final per-candidate token can then be

\[
\phi(m)=
\Big[
1,\ 
\mathbf{1}_{\text{capture}}(m),\ 
\mathbf{1}_{\text{edge-file}}(m),\ 
s_Q(m),\ c_Q(m),\ z_Q(m),\ 
\Delta_R(m),\ \Delta_B(m),\ \Delta_N(m),\ 
\kappa_N(m)
\Big].
\]

Aggregate over candidates with both max and sum pooling:

\[
F_{\text{cand,sum}} = \sum_{m\in M^\star} \phi(m), \qquad
F_{\text{cand,max}} = \max_{m\in M^\star} \phi(m),
\]

and expose the full primitive vector as

\[
F_{\text{PUGP}} =
\Big[
F_{\text{global-dist}},
\ F_{\text{cand,sum}},
\ F_{\text{cand,max}},
\ |M^\star|,
\ |M_{\text{push}}^\star|,
\ |M_{\text{cap}}^\star|
\Big].
\]

That gives the primitive a global branch, a candidate branch, and a typed underpromotion branch, all without motif labels. The choice to include both quiet and capture counts is especially important because the current PFCT implementation and tests are same-file oriented and therefore incomplete for true promotion move geometry. fileciteturn60file0L1-L220 fileciteturn62file0L1-L240

## Integration options

The cleanest first integration is the repo’s existing **additive gated primitive-head contract**:

\[
\text{final\_logit} = \text{base\_logit} + \text{primitive\_delta},
\qquad
\text{primitive\_delta} = \sigma(\text{gate\_mlp}) \cdot \delta_{\text{raw}}.
\]

That is already the shared primitive shape in `primitive_heads.py`, and the primitive TODO explicitly recommends “i193 trunk -> one primitive head -> gated primitive logit delta” for first-pass primitive implementations. PUGP fits that shape almost perfectly. fileciteturn56file0L1-L220 fileciteturn55file0L1-L180

The most practical option is therefore:

### Additive i193 side head

Use `i193_exchange_then_king_dual_stream` as the trunk, feed it the normal `simple_18` tensor, and add a small MLP/token-pooling head that consumes \(F_{\text{PUGP}}\). This preserves all current reporting, keeps ablations clean, and matches the repo’s primitive promotion path. It also directly supports matched comparisons against i193, i246, and any p046 ablations. fileciteturn55file0L1-L180 fileciteturn56file0L1-L220

### Precomputed primitive-feature columns

The primitive TODO already describes the preferred path for rule-derived primitive features: precompute them from FEN into Parquet columns, extend the dataset with `data.primitive_feature_columns`, and stack them into `batch["primitive_features"]`. PUGP should follow that pattern for the exact-legal variant, because it preserves the board-only modeling contract while removing per-sample CPU work from training. This is especially attractive here because promotion legality, `gives_check`, and attack/defense are all deterministic and sparse. fileciteturn55file0L1-L180

### Constant-plane fallback

If touching the trainer is undesirable in the first pass, the same TODO recommends encoding primitive scalars as constant extra planes and increasing `input_channels`. PUGP can support this fallback, but I would treat it as the second-best path because scalar duplication across planes is a poor fit for candidate-token features like `capture_promotion_count` or `knight_fork_hint_max`. fileciteturn55file0L1-L180

### BT4 mixer study

The repo already has a BT4 primitive-mixer architecture study for the existing promotion-aware mixer. That means p046 can also be evaluated as a **shape-preserving BT4 mixer** after the i193-side-head version is proven useful. This should be a *second* experiment, not the first one, because the additive side-head path makes causal interpretation much cleaner. fileciteturn68file0L1-L120

My recommendation is:

1. implement **p046 as an additive i193 head** first;
2. make the **rule-exact feature extractor precomputable**;
3. only then, if the signal is real, consider a **BT4 mixer** study.

## Slice-aware training plan

The training plan should follow the repo’s paper-grade rules: canonical tagged split, matched baselines, no metadata leakage, aggregate PR AUC plus matched-recall false positives, and explicit worst-slice reporting on `hard`, `equal`, `endgame`, `promotion`, and `underpromotion`. Reliable evidence is a 20-epoch convergence budget with patience 5, and repo-level promotion claims should be made on 3 seeds at minimum. fileciteturn64file0L1-L260 fileciteturn65file0L1-L200

The primitive should **not** use motif labels as input, but it **should** be trained and audited with slice-aware evaluation. I recommend a three-part plan.

### Scout phase

Run one-seed smoke and triage passes against i193 using the additive side-head design. The first scout target is not aggregate PR AUC; it is whether the primitive raises promotion/underpromotion slice metrics **without** obvious global regression. Because the benchmark standard says the strongest same-input baseline must be matched, the minimum scout comparison set should be: i193 trunk-only, i246 PFCT, and p046 full. fileciteturn55file0L1-L180 fileciteturn65file0L1-L200

### Reliable phase

Move to seeds 42, 43, and 44 with the canonical tagged split, paper-grade budget, and validation-only threshold selection. Report:

- aggregate PR AUC and F1;
- promotion and underpromotion slice PR AUC;
- total and near-puzzle false positives at recall 0.80 and 0.85, globally and on the promotion/underpromotion slice;
- worst slices on `hard`, `equal`, `endgame`, `promotion`, and `underpromotion`;
- p046 diagnostics such as `promo_candidate_count`, `promo_capture_count`, `promo_knight_hint_max`, and `promo_gives_check_count`.

That matches the repo protocol and also solves one practical problem in the current audit: external reporting still merges underpromotion with promotion, so p046 must provide its own internal diagnostic breakdown. fileciteturn64file0L1-L260 fileciteturn65file0L1-L200 fileciteturn67file0L220-L430

### Anti-overfitting guardrails

Do **not** train a primitive that fires only when a side-to-move pawn is already on row 1. Use the global distance branch on *all* positions, and define an extra board-derived audit slice such as `promo_pressure = 1[D^{\min}_{own}\le 2 \lor D^{\min}_{opp}\le 2]`. That gives the model a way to learn pawn-race geometry outside literal promotion positions and helps protect global PR AUC. This matters because the benchmark goal is not “memorize a slice”; it is “distinguish real puzzles from near-puzzles from the board alone.” fileciteturn54file0L1-L120

If class or slice reweighting is used, keep it mild. A good default is standard balanced BCE plus **sampling emphasis** on board-derived `promo_pressure` rows, not a large loss multiplier on `promotion` tags. That respects the no-motif-input rule and avoids turning the primitive into a dataset-composition exploit. fileciteturn64file0L1-L260 fileciteturn65file0L1-L200

## Falsifiers and expected metrics

### Falsifiers

PUGP should be rejected unless it survives matched ablations that test its claimed mechanism rather than its parameter count.

The first falsifier is **pseudo-legal versus rule-exact candidates**. Replace \(M^\star\) with raw geometric \(M\) and drop all legality-based features. If slice gains hold unchanged, the claim that exact promotion geometry matters is unsupported. If the exact version materially outperforms the pseudo-only version, that is strong evidence that fixing PFCT’s legality blind spots was worthwhile. This is the most important falsifier because the current i246 docs explicitly tolerate substitutions that need not correspond to legal moves. fileciteturn58file0L1-L220

The second falsifier is **quiet-only versus quiet-plus-capture**. Remove diagonal capture promotions from the candidate set and compare against the full model. If there is no measurable difference, then capture-promotion geometry is not load-bearing and the primitive can be simplified. If the full model wins, that directly validates one of the user’s core design goals and one of PFCT’s known omissions. fileciteturn60file0L1-L220

The third falsifier is **queen-only collapse**. Set all underpromotion deltas \(\Delta_R,\Delta_B,\Delta_N\) and the knight-fork score \(\kappa_N\) to zero. If the model keeps the same promotion/underpromotion slice behavior, then the underpromotion-hint story is false and p046 should be reframed as a promotion-only primitive. If the full model wins specifically on merged promotion/underpromotion rows and on candidate cases with `promo_knight_hint_max > 0`, keep the typed branch. fileciteturn61file0L1-L180

The fourth falsifier is **attack/defense shuffle**. Shuffle the promotion-square safety features across samples while keeping counts and distances intact. If the full model does not beat this ablation, then arrival-square attack/defense is not doing the work claimed for it. Python-chess attack primitives make this a clean test. citeturn13view0

The final falsifier is the repo-level non-regression rule: if p046 wins the promotion slice but loses aggregate PR AUC by more than the repo’s practical margin or increases matched-recall near-puzzle false positives globally, it should not be promoted. The reliable-training protocol explicitly says a model is only interesting if it improves one of the central metrics without breaking the others, and practical promotion margins are on the order of **+0.003** mean PR AUC or **1% fewer** near-puzzle FPs at matched recall. fileciteturn64file0L1-L260

### Expected metrics

Because the current audited numbers are **0.652** for i193 on promotion/underpromotion PR AUC and **0.667** for the best published promotion-slice PR AUC among the top groups, anything above about **0.69** would already be a meaningful step. A first reliable target for p046 should therefore be:

- **promotion / underpromotion slice PR AUC:** **0.69–0.71**
- **stretch target:** **0.72+**

That range is deliberately ambitious but still grounded in the repo’s existing slice gap and in the fact that i246’s own spec was targeting **0.720** slice PR AUC as a pass condition. fileciteturn53file0L1-L120 fileciteturn57file0L1-L180

On the operating-point metric, i193’s seed-42 matched-recall slice report is **0.103** near-FP rate at recall 0.80 on the promotion and underpromotion slice, while the best 3-seed audited robustness result reported in the research audit is **0.101 ± 0.015** for `i011_xl`. A realistic p046 target is therefore not a miracle result like 0.05; it is to move into the **0.090–0.098** band on that slice while keeping aggregate behavior stable. That would be enough to matter. fileciteturn66file0L1-L260 fileciteturn67file0L220-L430

For aggregate metrics, the expectation should be modest. PFCT’s own math note says promotion is only a small percentage of positions, so even a strong slice-only fix produces limited aggregate lift. I would therefore set the primary aggregate expectation at:

- **aggregate test PR AUC:** flat to **+0.003**
- **global near-puzzle FP at recall 0.80 / 0.85:** no regression, with a stretch goal of **~1% relative reduction**

That aligns with the repo’s practical promotion criteria. fileciteturn58file0L1-L220 fileciteturn64file0L1-L260

### Open questions and limitations

The biggest current limitation is evaluation, not design. The public CRTK motif audit currently merges underpromotion with promotion, so p046 can improve underpromotion-specific geometry without that being perfectly visible in the baseline slice table. That is why internal diagnostics and candidate-type breakdowns are mandatory. fileciteturn67file0L220-L430

The second limitation is representation scope. If p046 is implemented as a strict `simple_18` side head first, it will be maximally comparable to i193 and i246, but not necessarily maximally strong compared with BT4-family trunks. That trade-off is acceptable for the first pass because the repo’s primitive methodology values clean matched comparisons before hybridization. fileciteturn55file0L1-L180

## Runtime estimate and implementation sketch

### Runtime estimate

The runtime estimate should be treated as a projection, not a measurement. PFCT / i246 already documents the expensive end of the design space: worst-case `1 + K*4` trunk passes, batch size reduced from **256** to **128**, and an estimated **1.4–1.5×** i193 wall-clock. PUGP is designed to be much cheaper because it does **not** need repeated trunk reruns for every promotion piece type if its geometry is precomputed into a small side tensor. fileciteturn59file0L1-L260 fileciteturn60file0L1-L220 fileciteturn61file0L1-L180

With **offline precomputed features**, the train-time overhead should be close to “small MLP plus token pooling,” which is near-i193 cost. A practical expectation is **about 1.00–1.08× i193 wall-clock**, with the same batch-size regime i193 already supports. That is an inference from the repo contracts, not an observed measurement. fileciteturn56file0L1-L220 fileciteturn61file0L1-L180

With **online rule-exact python-chess computation inside `forward()`**, the primitive would likely become CPU-bound in the same way the repo already warns against for TSDP-style rule features. I would support that path only for smoke tests and feature validation, not for reliable training. The preferred production path is still feature precompute into Parquet columns. fileciteturn55file0L1-L180

### Implementation sketch

Create a new registered idea folder and keep it within the repo’s existing primitive-head pattern:

```text
ideas/registry/p046_promotion_underpromotion_geometry/
  idea.yaml
  config.yaml
  model.py
  math_thesis.md
  architecture.md
  ablations.md
  implementation_notes.md
  trainer_notes.md
```

Add a reusable feature module:

```text
src/chess_nn_playground/data/promotion_geometry.py
```

That module should expose three pure functions:

```python
canonicalize_to_stm(board_tensor_or_fen) -> CanonicalBoard
enumerate_promotion_candidates(canonical_board, exact_legal: bool) -> list[PromotionMove]
build_pugp_features(canonical_board, candidates) -> dict[str, np.ndarray | float]
```

The feature builder should output:

- global pawn-distance summaries;
- per-candidate tokens \(\phi(m)\);
- pooled scalar summaries;
- diagnostics such as `promo_candidate_count`, `promo_capture_count`, `promo_check_count_q`, `promo_knight_hint_max`, and `promo_exact_legal_count`.

That module can be used both by a precompute script and by unit tests. The repo already has a clear pattern for this kind of deterministic primitive code. fileciteturn55file0L1-L180 fileciteturn56file0L1-L220

Add a precompute script:

```text
scripts/data/precompute_promotion_geometry_features.py
```

It should read the canonical tagged split from:

```text
data/splits/crtk_sample_3class_unique_crtk_tags/
```

and emit a new split directory with primitive columns, following the same philosophy the primitive TODO gives for TSDP. This keeps training fast and repeatable. fileciteturn54file0L1-L120 fileciteturn55file0L1-L180

Add the model module:

```text
src/chess_nn_playground/models/primitives/promotion_underpromotion_geometry.py
```

Its forward path should be:

```text
simple_18 board tensor
-> i193 trunk
-> base_logit + trunk diagnostics
-> PUGP feature tensor or candidate-token tensor
-> small gate MLP + delta MLP / token pooler
-> final_logit = base_logit + primitive_delta
```

That is exactly the contract already documented in `primitive_heads.py`. fileciteturn56file0L1-L220

Add tests:

```text
tests/test_p046_promotion_underpromotion_geometry.py
```

The minimum test set should include:

- white quiet promotion;
- black quiet promotion;
- white capture promotion on both diagonals;
- black capture promotion on both diagonals;
- pinned near-promotion pawn with pseudo candidate but no legal candidate;
- queen-check promotion;
- knight-fork underpromotion case;
- no-promotion initial position;
- black/white canonicalization equivalence.

Those tests are especially important because the current PFCT tests and notes reveal same-file assumptions and a capture-promotion handling model that is not the full legal move geometry we want here. fileciteturn60file0L1-L220 fileciteturn62file0L1-L240

## Experiment matrix

| Variant | What changes | Why run it | Keep if |
|---|---|---|---|
| `i193_baseline` | No primitive | Matched reference | Baseline only |
| `p046_global_only` | Only distance buckets and pawn-race summaries | Tests whether broad pawn geometry alone helps | Promotion slice improves a little with zero global harm |
| `p046_global_plus_candidates` | Add quiet and capture candidate counts/tokens | Tests whether exact arrival geometry matters | Beats `global_only` on promotion slice |
| `p046_pseudo_only` | Use geometric candidates without legality filtering | Falsifier for rule-exact legality | Loses to exact-legal full model |
| `p046_no_capture_promotions` | Drop diagonal capture promotions | Falsifier for capture-promotion contribution | Loses to full model |
| `p046_queen_only` | Remove \(\Delta_R,\Delta_B,\Delta_N,\kappa_N\) | Falsifier for underpromotion hints | Loses on promotion/underpromotion slice or candidate-token diagnostics |
| `p046_no_attack_defense` | Remove \(a_t,d_t,s_t\) features | Falsifier for arrival-square safety | Loses on slice PR AUC or near-FP |
| `p046_full` | Global + exact candidates + square safety + king-zone + typed underpromotion deltas | Main candidate | Meets slice and non-regression targets |
| `i246_pfct` | Existing promotion-aware head | Direct comparison to current repo promotion primitive | Used as matched cost/benefit reference |
| `a003_bt4_mixer_p046` | Optional later BT4 mixer study | Tests whether p046 is useful as a spatial mixer, not just a side head | Only after `p046_full` wins as a side head |

The ordering should be: smoke on `p046_global_plus_candidates`, then one-seed scout on `p046_full`, `p046_pseudo_only`, and `p046_queen_only`, then a reliable 3-seed comparison on `i193_baseline`, `i246_pfct`, and `p046_full`. That sequence follows the repo’s primitive doctrine: validate one primitive at a time, require matched ablations, and do not promote a larger hybrid until the primitive proves it can beat its own controls on the declared slice. fileciteturn55file0L1-L180 fileciteturn64file0L1-L260