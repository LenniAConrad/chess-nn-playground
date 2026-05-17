# i259_i018_bt4_ensemble_compression.md

## Thesis

The strongest architecture strategy for this repo is a **two-stage pipeline**: first, build a **research-only teacher ensemble** that combines i018’s chess-structural signal with BT4 conv’s fast, robust spatial signal; then compress that ensemble into a **single deployment student** that keeps most of the ensemble’s near-puzzle rejection gains at roughly BT4-class latency. The point of the ensemble is not to become the shipped model. The point is to create a better teacher boundary—especially around fine-label `1` near-puzzles—then distill that boundary, along with a small subset of i018’s structural diagnostics, into a fast student. The repo’s own knowledge base explicitly calls out untested opportunities in **i018 + bt4_classifier ensembles** and **distillation from i018 into a BT4-shaped student**, while also warning that large hybrid trunks have usually been a worse lever than simply scaling a good trunk or compressing into a better compute shape. fileciteturn34file0L3-L3 fileciteturn41file0L3-L3

The success criterion should be stricter than “higher mean PR-AUC.” A variant should count as better only if it improves the repo’s **matched-recall near-puzzle false-positive metrics** and preserves or improves **probability calibration** after post-hoc scaling. This is aligned with the repo’s benchmark framing, where fine label `1` is intentionally trained as negative but behaves like the hardest negative class, and with the repo’s audit reports that rank models by near-puzzle false-positive rate at fixed recall levels. fileciteturn25file0L3-L3 fileciteturn35file0L3-L3

## Why i018 and BT4 should be complementary

i018 is complementary to BT4 conv because it is not “just another CNN.” i018 consumes `simple_18`, canonicalizes the board to side-to-move perspective, builds a dense **12-relation tactical incidence complex** over the 64 squares, performs learned **cellular sheaf diffusion**, optionally pools triad-defect structure, and emits not only a puzzle logit but a large set of **diagnostic outputs** such as `sheaf_tension`, `transport_imbalance`, `king_ring_pressure`, `reply_pressure`, `defense_gap`, `triad_defect_energy`, and `pin_pressure`. Those diagnostics are already produced in repo artifacts and are reporting-only, which makes them ideal meta-features for fusion and hint targets for distillation. fileciteturn15file0L3-L3 fileciteturn13file0L3-L3

BT4 conv, by contrast, is a compact residual conv tower with a 3×3 stem, four residual blocks, and Squeeze-Excite modulation in the default `lc0_bt4_classifier` configuration. Its default puzzle-binary benchmark uses the richer `lc0_bt4_112` encoding, `channels: 64`, `num_blocks: 4`, and a simpler value head over dense spatial features. In practice, the repo treats it as the **fast, robust baseline** compute shape. fileciteturn17file0L3-L3 fileciteturn19file0L3-L3

The repo’s summary evidence is exactly what makes the pair appealing. The knowledge base describes i018 as the current **accuracy-per-parameter champion**, stating that a 91K-parameter i018 beats a roughly 501K-parameter BT4 conv tower by about **+0.016 test PR-AUC** and that matched-parameter i018 widens that gap to about **+0.031**. The same document also says BT4 conv remains **robust, fast, and useful as a baseline**, and that on the repo’s CPU benchmark a BT4 classifier is about **6.4× faster** than i018 base at batch size 1 despite having more parameters, because dense conv maps far better to the underlying kernels than i018’s many small irregular tactical operations. fileciteturn34file0L3-L3

That is almost the textbook profile for a productive ensemble: two models that differ in **representation**, **inductive bias**, **input encoding**, and **runtime behavior**. i018 is strongest when chess structure is load-bearing; BT4 conv is strongest when a dense, highly optimized conv stack gives stable decisions cheaply. The repo’s knowledge base also notes that such an ensemble has **never been cleanly tested**, which means the complementarity is still a hypothesis but one with unusually strong prior evidence from the codebase itself. fileciteturn15file0L3-L3 fileciteturn17file0L3-L3 fileciteturn34file0L3-L3

## Ensemble variants

The first ensemble to test should be the **lowest-risk one**: a **temperature-scaled weighted logit average**. Deep ensembles often improve both predictive performance and uncertainty estimation relative to a single network, and simple averaging is the correct baseline to beat before introducing a learned meta-model. In this repo, that means calibrating i018 and BT4 separately, then sweeping a single weight parameter on a held-out fusion split. citeturn4academia3turn3academia2

The second variant should be a **low-capacity learned fusion** trained only on **out-of-fold or otherwise non-reused predictions**. A stacked linear or lightly regularized logistic meta-model is preferable to a large MLP because the repo has already seen that heavier integrated hybridization usually gives only modest lift or washes out against simply scaling the better trunk. The repo’s `i018 + primitive` hybrids used gated-logit fusion and reported only about **+0.006 PR-AUC** over i018 base while costing far more parameters, which is strong evidence to keep the late fuser **small, transparent, and easy to recalibrate**. Cross-validated stacking is the right way to avoid contaminating fusion training with reused teacher fit data. fileciteturn41file0L3-L3 fileciteturn34file0L3-L3 citeturn15academia0turn17academia1turn17academia3

The third variant should be an **uncertainty-gated fusion**. Here, the gate decides how much to trust i018 versus BT4 as a function of **disagreement**, **entropy**, and selected i018 diagnostics. This is attractive because the repo already values reject-option style robustness metrics, and selective prediction is a principled way to trade coverage against risk in high-uncertainty regions. During research, this variant can also be implemented as a **cascade**: run BT4 first, and invoke i018 only in a gray zone. That is not the final deployment answer, but it is a highly informative teacher experiment and a good source of distillation targets concentrated near the decision boundary. fileciteturn35file0L3-L3 citeturn9academia0turn9academia1turn9academia2

A practical repo-specific refinement is to use **i249** as the i018 branch implementation during ensemble R&D whenever runtime matters. The repo documents i249 as a pure execution optimization of i018 with the **same math, same parameters, and the same numerics**, achieved through chunked batched sheaf diffusion and optional `torch.compile`. That means it is a valid research-time surrogate for the i018 branch when generating teacher predictions or sweeping fusion variants, without changing the underlying teacher function. fileciteturn28file0L3-L3 fileciteturn29file0L3-L3

## Fusion equations

Let \(z_s(x)\) be the i018 logit and \(z_b(x)\) be the BT4 conv logit for board state \(x\). Let \(T_s>0\) and \(T_b>0\) be per-model temperature parameters fit on a calibration split, and define calibrated logits
\[
\tilde z_s(x)=\frac{z_s(x)}{T_s}, \qquad \tilde z_b(x)=\frac{z_b(x)}{T_b}.
\]
The corresponding probabilities are
\[
p_s(x)=\sigma(\tilde z_s(x)), \qquad p_b(x)=\sigma(\tilde z_b(x)).
\]
This makes temperature scaling the first calibration step before any threshold search or distillation target generation. citeturn3academia2

The **fixed weighted-average ensemble** should be
\[
z_{\mathrm{avg}}(x)=\alpha \tilde z_s(x)+(1-\alpha)\tilde z_b(x)+c,
\qquad
p_{\mathrm{avg}}(x)=\sigma\!\big(z_{\mathrm{avg}}(x)\big),
\]
with \(\alpha\in[0,1]\) chosen on a fusion split by the primary validation bundle: matched-recall near-puzzle FP first, PR-AUC second, calibration third. This is the strongest “boring” baseline and should be treated as the default to beat. citeturn4academia3turn15academia0

For **learned diagnostic fusion**, define a compact feature vector
\[
\phi(x)=\Big[
\tilde z_s,\ \tilde z_b,\ |p_s-p_b|,\ H\!\left(\tfrac{p_s+p_b}{2}\right),\ d_1,\ldots,d_k
\Big],
\]
where \(H(\cdot)\) is binary entropy and \(d_j\) are selected i018 diagnostics such as `sheaf_tension`, `triad_defect_energy`, `king_ring_pressure`, `reply_pressure`, `defense_gap`, and `pin_pressure`, all of which i018 already exports. Then fit
\[
z_{\mathrm{stack}}(x)=\beta^\top \phi(x)+c,
\qquad
p_{\mathrm{stack}}(x)=\sigma\!\big(z_{\mathrm{stack}}(x)\big).
\]
The key repo-specific choice is that the meta-model should be **linear or nearly linear**. A large nonlinear fuser is exactly the kind of fragile hybridization the repo’s own history warns against. fileciteturn15file0L3-L3 fileciteturn34file0L3-L3

For **uncertainty-gated fusion**, define a gate
\[
u(x)=|p_s-p_b|+\lambda\, H\!\left(\tfrac{p_s+p_b}{2}\right),
\]
\[
w(x)=\sigma\!\big(\gamma^\top [u(x), d_1,\ldots,d_k] + c\big),
\]
\[
z_{\mathrm{gate}}(x)=w(x)\tilde z_s(x)+(1-w(x))\tilde z_b(x),
\qquad
p_{\mathrm{gate}}(x)=\sigma\!\big(z_{\mathrm{gate}}(x)\big).
\]
This directly encodes the desired behavior: when i018’s structural evidence is coherent and disagreement is meaningful, shift weight toward i018; when i018’s diagnostics are weak or ambiguous, lean on BT4’s robustness. Because the repo already records these diagnostics, this adds no teacher-side architecture changes. fileciteturn15file0L3-L3

For **near-puzzle calibrated thresholding**, the operating threshold should be chosen after calibration by solving
\[
t_r=\arg\min_t \ \mathrm{FP}_{\mathrm{near}}(t)
\quad
\text{subject to}
\quad
\mathrm{Recall}_{\mathrm{puzzle}}(t)\ge r,
\]
for at least \(r\in\{0.80,0.85\}\), because those are already first-class audit thresholds in the repo. This is better aligned with the task than a fixed 0.5 threshold. fileciteturn35file0L3-L3 fileciteturn38file0L3-L3

## Compression and distillation plan

The default student should be a **BT4-shaped conv student**, not a smaller ensemble and not a heavier hybrid. The repo’s own summary argues that BT4 conv is the right deployment compute shape—fast, robust, and CPU-friendly—while i018 is the research teacher with the richer chess-structural signal. That strongly suggests distilling **into** BT4 form rather than trying to ship a conditional dual-model system. A secondary fallback for GPU-only deployment is i249, but that is still a sheaf-family model and therefore not the first choice when the deployment goal is fast single-model inference. fileciteturn34file0L3-L3 fileciteturn28file0L3-L3

The distillation target should not be only the ensemble probability. It should be a compact **teacher package** created offline for every training row: ensemble logit, ensemble calibrated probability, teacher disagreement, and a normalized subset of i018 diagnostics. Because i018 already emits diagnostics as prediction artifacts and those diagnostics are not used in its own loss, they can become **hint targets** for the student without changing the teacher. This is exactly the kind of auxiliary hinting that has worked well in distillation literature. fileciteturn15file0L3-L3 citeturn3academia0turn16academia0

A good student loss is
\[
\mathcal L
=
\lambda_{\mathrm{hard}}\,
\mathrm{BCE}\!\big(y,\sigma(z_{\mathrm{stu}})\big)
+
\lambda_{\mathrm{KD}}\,T_d^2
\mathrm{KL}\!\left(
\sigma\!\left(\frac{z_{\mathrm{teach}}}{T_d}\right)
\middle\|
\sigma\!\left(\frac{z_{\mathrm{stu}}}{T_d}\right)
\right)
+
\lambda_{\mathrm{diag}}
\sum_{j=1}^{k}
\|h_j(x)-\hat d_j(x)\|_2^2 ,
\]
where \(z_{\mathrm{teach}}\) is the chosen fused teacher score, \(\hat d_j\) are normalized teacher diagnostics, and \(h_j\) are small student auxiliary heads. The first term keeps the student anchored to true labels, the second transfers the ensemble boundary, and the third transfers i018’s structural signal into a conv-friendly representation. citeturn3academia0turn16academia0

The hard-label term should be **fine-label aware**. The benchmark contract maps fine label `1` to the negative class, but the repo explicitly describes it as the hardest negative and the main source of false positives in practice. That means the hard-label portion of the student loss should overweight fine-label `1` negatives, or use a fine-label-conditioned focal term, so that the student directly optimizes the deployment pain point instead of merely copying teacher smoothness. fileciteturn25file0L3-L3

The compression target should be framed operationally: the student is successful if it recovers **most of the ensemble lift on matched-recall near-puzzle rejection** while staying within roughly **BT4-class latency** and retaining acceptable calibration after a final temperature-scaling pass. In practice, that means preferring the smallest student that preserves the ensemble’s thresholded near-puzzle gains, not the student with the highest raw PR-AUC. fileciteturn35file0L3-L3 fileciteturn34file0L3-L3

## Evaluation metrics and calibration

The benchmark contract matters. The repo’s fresh benchmark report defines fine labels `0` and `1` as training target `0`, with fine label `1` specifically interpreted as **near-puzzle / unresolved candidate**, while fine label `2` is the positive puzzle class. The same report shows that the canonical train/val/test split sizes are 360K / 45K / 45K rows, each balanced across the three fine labels, which is enough to reserve a dedicated calibration subset or to use out-of-fold teacher generation cleanly. fileciteturn25file0L3-L3

The primary score bundle should therefore be: **overall test PR-AUC**, **matched-recall near-puzzle FP rate at recall 0.80 and 0.85**, **worst-slice accuracy** on the repo’s elevated hard slices (`hard`, `equal`, `promotion`, `underpromotion`), and **per-slice PR-AUC** by difficulty, phase, eval bucket, tactic motif, and side to move. The repo already computes these views, and its audits explicitly rank models by near-puzzle FP at fixed recall, which is exactly the deployment-relevant behavior the user called out. fileciteturn35file0L3-L3 fileciteturn32file0L3-L3 fileciteturn38file0L3-L3

Calibration must be treated as a first-class deliverable. For every teacher, ensemble, and student, report **Brier score**, **log loss / NLL**, **reliability diagrams**, and **adaptive or class-conditional ECE**, not just vanilla ECE. Modern neural nets are often poorly calibrated, temperature scaling is usually a strong first post-hoc choice, and adaptive/class-conditional calibration metrics are preferable because naïve ECE can be unstable or misleading. citeturn3academia2turn5academia0turn5academia4

The recommended validation protocol is: fit teachers on training folds only; generate **out-of-fold teacher predictions** for fusion learning and student-target generation; reserve the official validation split for **family selection, calibration, and threshold search**; touch the official test split exactly once. This avoids reusing the same rows for both base-teacher fitting and fusion fitting, which is a classic source of optimistic bias in stacking-style systems. citeturn17academia1turn17academia3

For statistical reporting, use **paired bootstrap confidence intervals** or paired permutation tests on the same test positions for the differences in PR-AUC, Brier, and matched-recall near-puzzle FP. Because the final deployment question is threshold-sensitive, also report **risk-coverage curves** or threshold sweeps around the chosen operating points. A fusion that looks slightly better on mean PR-AUC but is less stable at the recall targets should be rejected. fileciteturn35file0L3-L3 citeturn9academia0turn9academia1

## Speed analysis

A full teacher ensemble is acceptable for research and for generating distillation targets, but it is not the final answer. The repo’s CPU benchmark script explicitly times i018 base/scale-up/scale-xl and BT4 classifier base/scale-up/scale-xl at batch sizes 1, 8, and 32, and the repo’s knowledge base summarizes the result bluntly: **BT4 conv is about 6.4× faster than i018 base at batch 1 on CPU**. That means a naïve “run both forever” production ensemble would inherit the worst runtime characteristic of the pair. fileciteturn20file0L3-L3 fileciteturn34file0L3-L3

For stage-one experiments, the expected latency is approximately
\[
t_{\mathrm{ens}} \approx t_{\mathrm{i018}} + t_{\mathrm{bt4}} + t_{\mathrm{fuser}},
\]
and since \(t_{\mathrm{fuser}}\) is negligible, the only meaningful way to cut research-time cost is to reduce the i018 branch. That is exactly why i249 matters: the repo documents it as a numerically equivalent i018 implementation with vectorized chunked sheaf diffusion and optional `torch.compile`, created purely to lower wall-clock cost. Using i249 as the i018 branch during teacher generation is therefore a high-confidence optimization. fileciteturn28file0L3-L3 fileciteturn29file0L3-L3

If a cascade-style uncertainty gate is tested during stage one, its average cost is roughly
\[
t_{\mathrm{cascade}} \approx t_{\mathrm{bt4}} + q\, t_{\mathrm{i018}} + t_{\mathrm{gate}},
\]
where \(q\) is the fraction of positions sent to i018. This is worth measuring because it can expose whether most ensemble lift comes from a small ambiguous subset. But even if that turns out true, the final shipped model should still be the distilled student unless there is a hard requirement for online abstention or review-time fallback. citeturn9academia0turn9academia2

A sensible deployment target is therefore: **one encoding, one student, one calibration layer**. If the final student cannot stay close to BT4-class latency—roughly within a modest multiple of current BT4 batch-1 CPU inference—then the compression stage has not yet succeeded, even if it recovered most of the ensemble’s raw PR-AUC. fileciteturn34file0L3-L3

## Ablations, failure modes, and implementation plan

**Ablations.** The highest-value ablation grid is small and surgical. First, compare four teacher pairings: i018-base + BT4-base, i018-base + BT4-scale-xl, i018-scale-xl + BT4-scale-xl, and i249-exact + BT4-scale-xl. Second, compare fusion capacity: equal-weight average, tuned-\(\alpha\) average, linear stacked fusion, and uncertainty-gated fusion. Third, compare feature sets: logits only; logits plus disagreement and entropy; logits plus a six-diagnostic i018 subset. Fourth, compare calibration placement: no calibration, per-model temperature scaling only, and post-fusion temperature scaling. Fifth, compare students: BT4 base, BT4 scale-up, BT4 scale-xl, and puzzle-binary residual-small LC0 as a secondary control. Sixth, compare losses: hard labels only, hard + KD, hard + KD + diagnostic hints, and hard + KD + diagnostic hints + fine-label-1 reweighting. Every ablation should be scored on the full metric bundle, not just PR-AUC. fileciteturn17file0L3-L3 fileciteturn28file0L3-L3 fileciteturn35file0L3-L3

**Failure modes.** The biggest risk is **leakage in learned fusion**: using reused teacher-fit data to train the fuser will almost certainly overstate gains. The second risk is **calibration drift**: ensembles often change score distributions, and students often inherit the teacher ranking without the teacher calibration. The third risk is **label-noise amplification** on fine-label `1`; if teacher disagreement is mostly unresolved label ambiguity, the student can learn to imitate uncertainty rather than to reject near-puzzles better. The fourth risk is **slice regression**: equal-eval-bucket, hard, mate-in-1, promotion, and underpromotion slices are exactly where the repo has already found fragility, so they must gate advancement. The fifth risk is **shipping the wrong compute shape**: if the student begins to resemble a heavy hybrid instead of a clean conv model, the project has drifted away from the deployment goal. fileciteturn35file0L3-L3 fileciteturn34file0L3-L3

**Implementation plan.** Start with existing checkpoints and do a cheap **Phase A**: calibrate standalone i018 and BT4 teachers, evaluate weighted logit averaging on a held-out fusion subset, and report matched-recall near-puzzle FP plus calibration. If that shows real lift, move to **Phase B**: generate 5-fold out-of-fold teacher caches on the official training split, fit the linear stacked fuser on OOF predictions only, and reserve the repo’s official validation split for calibration and threshold search. If the best learned variant still wins after that leakage-safe protocol, move to **Phase C**: generate an offline distillation cache from the chosen teacher ensemble, train BT4-shaped students with KD and diagnostic-hint heads, then recalibrate every student on the validation split and rank them by matched-recall near-puzzle FP under latency constraints. Finally, in **Phase D**, run a single untouched test evaluation with paired confidence intervals and a concise report that includes overall PR-AUC, matched-recall near-puzzle FP, worst slices, Brier, NLL, and latency. fileciteturn25file0L3-L3 fileciteturn35file0L3-L3 citeturn17academia1turn17academia3turn3academia0turn16academia0

**Open questions and limitations.** The repo evidence is strong on the *reason* to try this plan, but weak on one important point: there is not yet a clean, apples-to-apples published result for an actual i018 + BT4 ensemble in this repository. Some summary documents also report different absolute PR-AUC views depending on which audit or run family they summarize, so the exact absolute lift should be treated as unconfirmed until the leakage-safe teacher comparison is run. The good news is that the repo already has the necessary pieces—teacher diagnostics, calibration-aware audits, matched-recall reports, and a faster numerically equivalent i018 implementation—to answer that question cleanly without inventing a new benchmark contract. fileciteturn34file0L3-L3 fileciteturn35file0L3-L3 fileciteturn28file0L3-L3