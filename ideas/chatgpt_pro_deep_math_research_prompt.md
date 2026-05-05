# ChatGPT Pro Deep Math Research Prompt

Copy only the prompt inside the fenced block below into ChatGPT Pro. Do not attach repo files or provide any extra context.

````markdown
You are an autonomous research mathematician and machine-learning architect helping with `chess-nn-playground`, a chess neural-network research lab.

This prompt is self-contained. Do not ask me for repository files, previous prompts, existing code, or extra context. Use only the context below plus your own research.

Your job is to produce exactly one original, testable, Codex-ready research idea for chess puzzle-likeness classification. Do not give a list of generic ideas. Do not implement code. I will give your final Markdown file to Codex, which will create the files, implement the model, train it, benchmark it, and update this prompt before the next research cycle.

## Required Delivery Format

Create one downloadable Markdown file as your final result. Do not create any other files.

Use ChatGPT's file/download feature if available. The file must be named with this exact pattern:

```text
chess_nn_research_<YYYY-MM-DD>_<HHMM>_<weekday>_<timezone>_<idea_slug>.md
```

Filename rules:

- Use your current date and time when generating the answer.
- Use 24-hour time.
- Use lowercase ASCII.
- Replace spaces and punctuation with underscores.
- Sanitize the timezone into a short token such as `utc`, `local`, `shanghai`, or `new_york`.
- Keep `idea_slug` short, descriptive, and lowercase.

Example filename:

```text
chess_nn_research_2026-04-21_1730_tuesday_shanghai_attack_sheaf.md
```

If your interface cannot create a downloadable file, output exactly one fenced Markdown block containing the complete file content and put the intended filename immediately before it. Do not add commentary outside that fallback.

## Project Context You Must Respect

The project is `chess-nn-playground`.

The current research task is chess puzzle classification from board positions:

- output `0`: non-puzzle
- output `1`: puzzle-like

The current source classes are:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

The default benchmark is binary, but reports include a rectangular `3x2` diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Current available encodings:

- `simple_18`: 12 piece planes + side-to-move + castling + en-passant
- `lc0_static_112`
- `lc0_bt4_112`: LC0-style 112-plane BT4 layout from a single FEN, with unavailable history planes zero-filled until exporter support exists

Current baselines already exist:

- simple CNN under `src/chess_nn_playground/models/cnn.py`
- residual CNN under `src/chess_nn_playground/models/residual_cnn.py`
- small/medium/deep variants
- LC0 BT4-style CNN and residual CNN variants

Current benchmark split:

```text
data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet
data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet
data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet
```

The full Parquet dataset has roughly 45M rows, but the current trainer should not be pointed directly at the full file until streaming support exists.

The implementation target is PyTorch. A new model should normally be a `torch.nn.Module` accepting:

```text
(batch, C, 8, 8)
```

and returning logits:

```text
(batch, num_classes)
```

The shared trainer, reports, confusion matrices, predictions, and leaderboards should keep working.

## Non-Negotiable Constraints

- Do not use Stockfish scores, PVs, node counts, verification metadata, source labels, or proposed labels as neural-network input features.
- Do not fabricate class `1` or class `2` labels.
- If an unresolved candidate pool is mentioned, treat it as unresolved, not as verified near-puzzles or verified puzzles.
- Do not propose ordinary depth/width/hyperparameter tuning as a research idea.
- Do not propose "use a bigger CNN", "use a standard ResNet", "use a vanilla Transformer over 64 squares", "copy LC0", "use an ensemble", "add more data", or "just tune the optimizer" as the core idea.
- Do not make the idea depend on future data unless you explicitly mark that dependency and also provide a minimal current-data experiment.
- Separate proven claims from hypotheses. If a proof is not possible, say so plainly.

## Research Mode

Think deeply and take your time. Do not answer with the first plausible idea. Spend a long research pass exploring multiple mathematically distinct directions, stress-testing them against the constraints, and only then select the strongest one.

Use current research knowledge and web/deep research if available. Look outside ordinary chess-engine model design for useful mathematics, such as:

- group representation theory and chess-specific partial equivariance
- graph, hypergraph, simplicial, sheaf, or cell-complex neural operators
- spectral methods on attack/defense relations
- energy-based models or constrained latent-variable models
- optimal transport, diffusion, or information bottleneck methods
- differentiable search surrogates that do not leak engine analysis
- causal or invariant-risk ideas for separating puzzle structure from superficial material patterns

These are suggestions, not requirements. Choose the mechanism only if it fits the task and can be tested.

You may use high-level mathematical concepts as search seeds, including but not limited to:

- weighted finite automata and formal languages
- information geometry
- causal invariance and invariant-risk objectives
- selective prediction, abstention, calibration, and evidential uncertainty
- ordinal regression
- minimum description length and pseudo-likelihood
- energy-based models
- polynomial feature interactions, ANOVA decompositions, and low-rank tensor factorizations
- sparse rationales and bottleneck models
- spectral compression
- differentiable dynamic programming

However, a named concept is not an idea. Do not simply name-drop a concept and wrap a CNN around it. If a concept survives to the final idea, translate it into:

- a concrete board operator or learning objective
- the exact tensors it consumes and emits
- the hypothesis it tests
- the smallest ablation that removes the concept while preserving nuisance statistics
- the reason it is not a near-duplicate of imported packets

Concept families already represented in imported packets are not automatically novel: sheaves, Hodge operators, graph Laplacians, one-ply move-delta sets, Sinkhorn/optimal transport, static attack-defense graphs, simple ordinal ladder heads, sparse witness-piece bottlenecks, Möbius/ANOVA piece constellations, ray-language automata, pseudo-likelihood board-ratio models, cubical Euler/Betti topology, Hall-defect overload profiles, king-cage/king-escape path dynamic programs, formal concept/Galois closure bottlenecks, class-conditional denoising score fields, non-backtracking tactical edge-walk operators, defender timing schedules, discovered-ray switchboards, counterplay ledgers, pinned mobility nullspaces, tactical effective-resistance graph solves, defender opportunity-cost auctions, role-counterfactual necessity probes, phase-specialist calibration mixtures, forced-target funnels, tactical subgoal automata, support-polar zonotopes, loop-frustration curvature, forcing-response front-door mediators, hypercut polynomials, robust tail DRO, material-locked mask DRO, Fisher-geodesic excess, hypergraph motif grammars, soft sorting residuals, sparse relation pursuit, Hall-defect zeta spectra, differentiable abstract interpretation, tactical radius filtrations, traced motif composition, conditional surprisal gates, bounded hinge logic, Tucker relation certificates, tactical latent-state inference, Dykstra constraint projection, and positive-claim abstention.

Depth requirements:

- Explore at least 12 candidate mechanisms internally before selecting one.
- Surface at least 8 rejected common approaches in the required rejection table.
- Surface at least 4 rejected serious-but-not-selected research candidates in the rejection table or research map, with precise reasons.
- For the selected idea, run a self-critique before finalizing: identify likely shortcuts, leakage risks, implementation risks, ablation weaknesses, and why the idea still survives.
- Prefer a slower, more rigorous answer over a fast one. The final file should be concise enough for Codex to act on, but the reasoning behind the selected mechanism must be deep and explicit.

## Research Continuity

This is an iterative research loop. Codex will use your file and then update this prompt before the next run, so include enough prompt-maintenance information to help the next ChatGPT Pro pass avoid repeating the same mechanism.

Your output must include:

- a compact idea fingerprint
- the closest baseline or common method it resembles
- mechanisms that should not be repeated next time if this fails
- prompt changes Codex should make after consuming your output

## Imported Research Memory

Codex has imported recent ChatGPT Pro research packets into `ideas/research_packets/`. Treat these as already researched families, not fresh ideas:

- `Attack-Hodge Sheaf Tension Network`
- `Tactical Threat-Sheaf Network`
- `Tactical Sheaf Tension Network`
- `Attack-Defense Sheaf Energy Network`
- `Tactical Sheaf Curvature Network`
- `Oriented Tactical Sheaf Laplacian`
- `Directed Attack-Sheaf Tension Network`
- `File-Mirror Tension Sheaf`
- `Rule-Only Counterfactual Move-Delta Bottleneck`
- `Counterfactual Move-Delta Spectrum Network`
- `One-Ply Counterfactual Move Landscape Network`
- `Nuisance-Orthogonal Puzzle Bottleneck`
- `Entropic Piece-Target Transport Bottleneck`
- `Tactical Transport Imbalance Network`
- `Piece-Target Entropic Transport Bottleneck`
- `Entropic Chess Geometry Transport Network`
- `King-Anchored Material-Null Transport Bottleneck`
- `Ordinal Evidence Ladder Network`
- `Sparse Witness-Piece Bottleneck Network`
- `Ray-Language Automaton Network`
- `Möbius Piece-Constellation Network`
- `Geometry-Conditioned Board Pseudo-Likelihood Ratio Network`
- `Color-Flip Orbit Evidence Bottleneck`
- `Credal Near-Puzzle Evidence Network`
- `Rule-Exact Orbit Bottleneck Network`
- `Tempo-Odd Bottleneck Network`
- `Rule-Automorphism Quotient Bottleneck Network`
- `Kinematic Commutator Bottleneck Network`
- `Masked Board Code-Length Surprise Network`
- `Side-Canonical Rule-Partition Invariant Bottleneck`
- `Legal Automorphism Quotient Network`
- `Centered Tempo-Odd Interventional Bottleneck`
- `King-Anchored Euler Interaction Network`
- `Threat-Topology Betti Bottleneck Network`
- `Hall-Defect Obligation Matroid Network`
- `King Escape Percolation Network`
- `Soft King-Cage Path Bottleneck Network`
- `Soft Formal-Concept Closure Network`
- `Non-Puzzle Score-Field Bottleneck Network`
- `Non-Backtracking Tactical Walk Network`
- `Defender Timing Schedule Network`
- `Discovered-Ray Switchboard Network`
- `Counterplay Insolvency Ledger`
- `Pinned Mobility Nullspace Network`
- `Tactical Effective Resistance Network`
- `Defender Opportunity-Cost Auction Network`
- `Role-Counterfactual Necessity Network`
- `Phase-Specialist Calibration Mixture`
- `Forced-Target Funnel Network`
- `Tactical Subgoal Automaton Network`
- `Support-Polar Zonotope Certificate Network`
- `Loop-Frustration Curvature Network`
- `Forcing-Response Front-Door Bottleneck`
- `Chess Hypercut Polynomial Network`
- `Contamination-DRO Huber Tail Rejection`
- `Fisher-Geodesic Tension Network`
- `Material-Locked Tactical Mask DRO`
- `Typed Hypergraph Motif Grammar`
- `Soft Sorting Order Residual Ranker`
- `Sparse Relation Pursuit Asymmetry`
- `Hall-Defect Zeta Operator`
- `Differentiable Chess Fact Lattice`
- `Tactical Radius Filtration`
- `Traced Threat Motif Network`
- `Conditional Surprisal Gate`
- `Bounded Board Hinge Logic`
- `Chess-Mode Tucker Relation Certificate`
- `Tactical State Bottleneck Inference`
- `Soft-Dykstra Latent Constraint Projector`
- `VetoSelect Positive-Claim Abstention`

Shared fingerprint of the tactical sheaf/Hodge packets:

```text
current-board pseudo-legal attack/defense/x-ray incidence or cell complex
+ learned sheaf restrictions, sheaf Laplacian, Hodge operator, curvature, or tension-energy pooling
+ binary puzzle-likeness target
+ no engine metadata
```

Shared fingerprint of the counterfactual move-delta packets:

```text
current-board pseudo-legal side-to-move one-ply move-delta set or multiset
+ DeepSets, attention, finite-difference spectrum, entropy/free-energy, or sparse bottleneck pooling
+ binary puzzle-likeness target
+ no engine metadata or search scores
```

Shared fingerprint of the optimal-transport packets:

```text
current-board piece/source/target measures or king/value target atoms
+ entropic Sinkhorn / optimal-transport coupling, transport bottleneck, pressure maps, transport imbalance, or material-null residual descriptors
+ binary puzzle-likeness target
+ no engine metadata, no legal move tree, no one-ply move-delta bag
```

Shared fingerprint of the nuisance-orthogonal packet:

```text
current-board deterministic material/phase/king/castling/en-passant nuisance vector
+ closed-form ridge residualization / orthogonal projection of CNN latents away from nuisance span
+ binary puzzle-likeness target
+ no engine metadata
```

Additional distinct imported packet fingerprints:

```text
ordinal cumulative fine-label evidence ladder with nested thresholds P(fine>=1), P(fine>=2)
top-k occupied-piece sparse witness bottleneck with masked-board classifier
weighted finite automata over side-relative oriented rank/file/diagonal ray token strings
degree-2/3 ANOVA or Möbius occupied-piece constellation interactions
class-conditioned static-geometry board pseudo-likelihood / description-length ratio
color-flip / legal automorphism orbit quotient, Reynolds pooling, and evidence-intersection views
side-to-move tempo odd/even intervention bottlenecks and null-board centering
binary Dirichlet / credal evidence treatment of fine-label-1 ambiguity
side-canonical rule-partition invariance with V-REx, VIB, and environment adversaries
degree-two Lie commutators of deterministic chess kinematic operators
label-free masked board code-length / surprise codec fields
cubical Euler characteristic / Euler-additivity interaction curves over current-board role bitboards
rank-top-k cubical Betti, perimeter, and topology bottlenecks over pseudo-legal pressure fields
Hall-defect / transversal-matroid overload profiles over defender-obligation set systems
king escape / king-cage soft shortest-path, Bellman-Ford, percolation, and path free-energy bottlenecks
formal context / Formal Concept Analysis / Galois closure bottlenecks over current-board square or piece attributes
class-0-only denoising score matching / non-puzzle repair vector fields over current-board tensors
Hashimoto / non-backtracking tactical edge-walk propagation over pseudo-legal attack/protection graphs
differentiable defensive timing schedules with learned deadlines, precedence, lateness, and slack
hidden discovered-ray switchboards over blocker vacation, line activation, and target exposure
counterplay insolvency ledgers and defender opportunity-cost auctions over forcing assets and liabilities
pinned defender mobility nullspace projections through pin, x-ray, king-safety, and target constraints
tactical effective-resistance graph solves between threats, defenders, and targets
role-counterfactual necessity probes from safe material/role-preserving synthetic board views
phase-specialist calibration mixtures over mate, material, promotion, endgame, and opening-trap phases
forced-target funnel entropy/consensus over candidate tactical actions and target identities
tactical subgoal automata over ordered typed predicates such as overload, blocker removal, and counterplay suppression
support-polar zonotope containment certificates over square-pair generators
spin-glass loop-frustration products and temperature-curvature observables
front-door forcing-response mediators from legal current-position interventions
high-order chess hypercut polynomials over deterministic rule hyperedges
robust near-puzzle upper-tail DRO and material-locked tactical-mask DRO
Fisher-Rao geodesic excess over learned square distributions
typed hypergraph motif grammars and differentiable motif composition
soft sorting-network order residual losses
sparse relation pursuit dictionaries over deterministic relation tokens
Hall-defect zeta incidence-algebra spectra
differentiable abstract-interpretation fact lattices with join, meet, transfer, and widening
tactical radius filtrations over rule-distance shells
categorical trace-style motif composition over typed relation matrices
conditional surprisal gates / conditional rate bottlenecks
bounded PSL-style hinge logic over current-board facts
Tucker relation certificates over fixed chess-relation moment tensors
structured tactical latent-state inference over motif, anchor, target, relation, vulnerability, and tempo tuples
Soft-Dykstra latent constraint projection traces
positive-claim abstention / veto-select factorization
```

Do not propose another static attack-defense graph, tactical sheaf, chess-incidence sheaf, Hodge/sheaf-Laplacian, curvature/tension-energy, file-mirror sheaf, or attack-sheaf variant unless it changes the falsifiable operator in a way that is not merely more edge types, more relation labels, larger hidden size, more layers, different pooling, partial symmetry gating, or renamed sheaf terminology.

Do not propose another one-ply pseudo-legal move-delta bag/set/multiset model using DeepSets, attention, MIL, covariance/eigen-spectrum, entropy/free-energy, sparse bottlenecks, move-energy, move-curvature, or destination-shuffle variants unless the operator is mathematically different from the imported move-delta family.

Do not propose another current-board piece-target/material-target entropic optimal-transport or Sinkhorn model using transport bottlenecks, pressure maps, forward/reverse imbalance, material-null shuffles, cost randomization, cost histograms, target buckets, entropy-temperature changes, extra heads, or different transport pooling unless the formal object is genuinely different from the imported OT family.

Do not propose another deterministic nuisance-vector residualization / orthogonal-projection bottleneck over material, phase, king, castling, or en-passant features unless the mechanism is genuinely different from closed-form latent projection and has a distinct falsifier.

Do not propose exact near-duplicates of the imported ordinal ladder, sparse occupied-piece witness bottleneck, ray-language automaton, Möbius/ANOVA piece-constellation model, or static-geometry board pseudo-likelihood ratio unless the formal observable and central falsifier are genuinely different.

Do not propose exact near-duplicates of imported color-flip/orbit quotient/Reynolds pooling, side-to-move tempo odd/even intervention, credal/Dirichlet near-puzzle evidence, rule-partition V-REx/VIB invariance, kinematic Lie-commutator, or masked board surprise-codec packets unless the formal observable and central falsifier are genuinely different.

Do not propose exact near-duplicates of imported cubical Euler/Euler-additivity topology, pressure-field Betti topology, Hall-defect/transversal-matroid overload, or king-cage/king-escape soft path/percolation dynamic-program packets unless the formal observable and central falsifier are genuinely different. Merely changing thresholds, anchors, pressure fields, temperatures, shell radii, graph neighborhoods, histogram summaries, or CNN fusion is not enough.

Do not propose exact near-duplicates of imported formal-context/FCA/Galois-closure, class-0-only denoising score-field ordinaryness, or Hashimoto/non-backtracking tactical edge-walk packets unless the formal state space and central falsifier are genuinely different. Merely changing the attribute vocabulary, number of probes, t-norm temperature, denoising noise schedule, score bottleneck size, walk depth, relation labels, pooling statistics, or hidden size is not enough.

Do not propose exact near-duplicates of local puzzle-batch timing, discovered-ray switchboard, counterplay ledger, pinned mobility nullspace, effective-resistance, opportunity-cost auction, role-counterfactual necessity, phase-specialist calibration, forced-target funnel, or tactical subgoal automaton packets unless the formal bottleneck and central falsifier are genuinely different. Merely changing bucket counts, hidden dimensions, target vocabularies, expert counts, number of counterfactual views, graph relation labels, predicate labels, or pooling statistics is not enough.

Do not propose exact near-duplicates of the newest Downloads imports: support-polar zonotopes, spin-glass loop-frustration curvature, forcing-response front-door bottlenecks, chess hypercut polynomials, robust tail DRO, material-locked tactical-mask DRO, Fisher-geodesic tension, hypergraph motif grammars, soft sorting residual objectives, sparse relation pursuit, Hall-defect zeta operators, differentiable abstract interpretation, tactical radius filtrations, traced threat motifs, conditional surprisal gates, bounded hinge logic, Tucker relation certificates, tactical latent-state inference, Soft-Dykstra constraint projection, or veto-select positive-claim abstention. Merely changing vocabulary, tensor rank, relation labels, number of shells, dictionary size, solver iteration count, hidden dimensions, or pooling statistics is not enough.

Prefer next-cycle ideas that are genuinely outside the imported sheaf, move-delta, OT, nuisance-projection, ordinal-head, sparse-witness, ray-language, high-order-constellation, pseudo-likelihood, orbit-symmetry, tempo-intervention, credal-evidence, rule-partition-invariance, kinematic-commutator, masked-codec, cubical-topology, Hall-defect-overload, king-path-DP, FCA-closure, denoising-score-field, non-backtracking-edge-walk, defender-timing, discovered-ray, counterplay-ledger, pinned-nullspace, effective-resistance, opportunity-auction, role-counterfactual, phase-specialist, target-funnel, subgoal-automaton, zonotope-certificate, loop-frustration, forcing-response-front-door, hypercut-polynomial, robust-DRO, Fisher-geodesic, hypergraph-grammar, soft-sort-ranker, sparse-relation-pursuit, Hall-zeta, abstract-interpretation, tactical-radius, trace-motif, conditional-surprisal, bounded-hinge-logic, Tucker-certificate, tactical-latent, Dykstra-projection, and veto-select families, such as:

- causal invariance across genuinely new environments or data-source shifts, not the imported phase/material/color rule partitions
- information bottlenecks that suppress source artifacts while preserving tactical signal without enumerating moves, using Sinkhorn transport, or using closed-form nuisance projection as the central operator
- label-safe uncertainty or abstention models that are not another ordinal ladder or binary Dirichlet/credal evidence head
- calibration or selective-prediction mechanisms for ambiguous near-puzzles that are not just another cumulative ordinal ladder or credal-evidence variant
- generative compression or minimum-description-length motif models that are not another class-conditioned pseudo-likelihood ratio or masked-board surprise codec

## What Counts As Creative Enough

The selected idea must have a distinct inductive bias about why a chess position is puzzle-like. It should not be a renamed baseline. It should be able to fail in an informative way.

Before selecting the final idea, explicitly reject at least eight common candidate approaches. For each rejection, write one sentence saying why it is too ordinary, too leaky, too untestable, or already covered by the existing baseline suite.

## Required Markdown File Content

The downloadable Markdown file must contain one document titled:

```markdown
# Codex Handoff Packet: <idea name>
```

Use exactly the sections below.

## 1. File Metadata

- Filename:
- Generated at:
- Weekday:
- Timezone:
- Idea slug:
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name:
- One-sentence thesis:
- Idea fingerprint:
- Why this is not a common CNN/ResNet/Transformer variant:
- Current-data minimal experiment:
- Smallest central falsification ablation:
- Expected information gain if it fails:

## 3. Problem Restatement And Data Contract

Restate the task, labels, allowed inputs, forbidden inputs, tensor shapes, and benchmark split. Include a short leakage checklist.

Clarify the boundary between safe rule-derived features and leakage:

- Deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack geometry derived only from the current board are allowed.
- Full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences are leakage-prone unless explicitly justified as rule-only, label-independent, engine-free, and ablated.
- Engine evaluation, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, and dataset provenance must never be neural-network inputs.
- For `lc0_static_112` and `lc0_bt4_112`, distinguish current-board channels used for deterministic geometry from history channels used only by learned neural adapters.

## 4. Research Map

Summarize the external ideas or papers you used. Include citations or URLs when available. For each source, say exactly what is borrowed and exactly what is not copied.

If you cannot verify a citation, mark it as unverified instead of inventing details.

Also include a short “candidate search trace” with at least 4 serious candidate mechanisms you considered but did not select, and why each lost to the final idea.

If you use a high-level concept, include a concept-to-operator mapping table:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|

## 5. Common Approaches Rejected

Reject at least eight approaches. Include the existing simple CNN, residual CNN, LC0-style CNN/residual CNN, ordinary ViT, plain GNN-on-squares, hyperparameter tuning, ensembling, and any other close duplicate you considered.

Use this table:

| Approach | Closest existing baseline | Why rejected |
|---|---|---|

## 6. Mathematical Thesis

Write this as real math, not marketing language.

Include:

- Input space definition.
- Label/target definition.
- Data distribution assumptions.
- Allowed symmetry or equivariance assumptions. Be careful: chess is not fully rotation/reflection invariant because pawns, castling, and side-to-move matter.
- The core hypothesis.
- A formal object or operator introduced by the idea.
- A proposition, theorem, variational principle, or optimization objective that explains why the mechanism should help.
- Proof sketch or derivation.
- What is actually proven.
- What remains only hypothesized.
- Counterexamples where the idea should fail.
- Self-critique: the strongest mathematical or empirical objection to the idea, and why the minimal experiment is still worth running.

## 7. Architecture Specification

Give a Codex-implementable design.

Include:

- Module names.
- Forward-pass steps.
- Tensor shapes after each major operation.
- Parameter-count estimate.
- FLOP or complexity estimate.
- For any generated candidate set, estimate memory as a function of batch size, max candidates, and candidate dimension; include a chunking plan if the candidate set could be large.
- Required config fields.
- How to support both `simple_18` and `lc0_bt4_112` if feasible, or why the first experiment should use only one encoding.
- Explicit encoding-adapter assumptions for `simple_18`, `lc0_static_112`, and `lc0_bt4_112`; adapters must fail closed when channel semantics are unknown.
- How the model returns logits compatible with the shared trainer.

Provide pseudocode, but do not write the full final implementation.

## 8. Loss, Training, And Regularization

Specify:

- Primary loss.
- Any auxiliary loss and whether it is optional.
- Class weighting.
- Batch size expectations.
- Learning-rate and optimizer defaults.
- Regularizers.
- Determinism requirements.
- What must stay unchanged from the existing benchmark configs for a fair comparison.

## 9. Ablation Plan

Provide a table:

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|

Include the smallest ablation that can falsify the central mathematical claim.

For any structured operator over graph, hypergraph, sheaf, transport, counterfactual, or search-surrogate objects, include a semantics-destroying or degree-preserving randomized ablation that tests whether the structure itself matters.

For any rule-generated move-set or candidate-set model, also include count-only and nuisance-preserving ablations that preserve obvious shortcuts such as candidate count, degree, material, side-to-move, moving-piece identity, source-square marginal, and capture histogram while destroying the proposed semantics.

For imported-family-adjacent ideas, include the relevant hard control: ordinal heads need unconstrained softmax or independent-binary-head controls; sparse witness models need matched random and top-material witness controls; ray-language models need token-shuffle or geometry-destroying ray controls; high-order constellation models need degree-1-only controls; pseudo-likelihood models need randomized-neighborhood and unary/material-only controls; symmetry/orbit quotient models need exact transform tests plus semantics-destroyed pseudo-orbit and augmentation-only controls; side-to-move intervention models need identity/random-side twins and an out-of-distribution critique; credal/evidential models need hard-positive BCE or ordinary-softmax controls plus exported uncertainty/evidence diagnostics; causal-invariance partition models need random-environment and no-invariance/material-only/color-only controls; kinematic operator-algebra models need symmetric-product, first-order-only, and degree-preserving randomized-operator ablations; masked-codec models need unigram/material-prior and square-shuffled surprise controls.

Cubical Euler/Betti topology ideas need rank- or histogram-preserving topology-destroying shuffles plus count-only or individual-curve controls. Hall/matching/overload ideas need degree-matched edge rewires plus obligation-count/material-only controls. King-cage or king-escape path-DP ideas need ring/bin-preserving hazard shuffles plus random-graph topology controls.

Formal-context/FCA closure ideas need row/column-sum-preserving incidence rewires plus marginal-only controls. Denoising-score or generative-prior ideas need all-class-prior, nuisance-only-prior, and random/permuted-field controls. Non-backtracking or edge-walk graph ideas need degree/type-preserving randomized transition controls plus count/degree/material-only controls.

## 10. Benchmark And Falsification Criteria

Define exactly how Codex should benchmark it:

- Baselines to compare against.
- Metrics to inspect.
- Required fine-label `0/1/2 -> predicted 0/1` confusion for the main model and every central ablation.
- Required slice reports for the main model and every central ablation. Do not stop at the aggregate matrix.
- Per-slice performance by `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`, `crtk_tactic_motifs`, and `crtk_tag_families`.
- A statement of which difficulty/tag slices the idea should improve, which slices it may worsen, and which ablation should erase the claimed slice-level gain.
- Highest-confidence wrong examples with FEN, difficulty, phase, motifs, true label, predicted label, and confidence.
- A near-puzzle diagnostic, preferably class `1` recall or precision at a matched fine-label-`0` false-positive rate.
- Required artifacts.
- Success threshold.
- Failure threshold.
- What result would make you abandon the idea.
- What result would justify scaling.

## 11. Implementation Plan For Codex

List exact repo changes Codex should make.

Use this table:

| Path | Action | Contents |
|---|---|---|

Include at least:

- `ideas/<idea_id>_<slug>/idea.yaml`
- `ideas/<idea_id>_<slug>/math_thesis.md`
- `ideas/<idea_id>_<slug>/architecture.md`
- `ideas/<idea_id>_<slug>/implementation_notes.md`
- `ideas/<idea_id>_<slug>/trainer_notes.md`
- `ideas/<idea_id>_<slug>/ablations.md`
- `ideas/<idea_id>_<slug>/train.py`
- `ideas/<idea_id>_<slug>/config.yaml`
- `ideas/<idea_id>_<slug>/report_template.md`
- `ideas/chatgpt_pro_deep_math_research_prompt.md`
- `src/chess_nn_playground/models/<model_name>.py`
- `src/chess_nn_playground/models/registry.py`
- `configs/benchmarks/<task>/<config_name>.yaml`
- focused tests if needed

For `ideas/chatgpt_pro_deep_math_research_prompt.md`, Codex must update the prompt after consuming your output. The update should preserve hard constraints while adding any reusable lessons, new anti-duplicate rules, clearer output requirements, or failure-mode guidance discovered from this research pass.

## 12. Machine-Readable Blocks

Provide these fenced blocks exactly.

```yaml
download_artifact:
  filename: null
  generated_at: null
  weekday: null
  timezone: null
  idea_slug: null
  format: markdown
```

```yaml
idea_yaml:
  idea_id: null
  name: null
  slug: null
  status: draft
  created_at: null
  author: ChatGPT Pro
  short_thesis: null
  novelty_claim: null
  expected_advantage: null
  central_falsification_ablation: null
  target_task: coarse_binary
  input_representation: null
  output_heads: null
  compute_notes: null
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: null
  model_path: null
  latest_result_path: null
  notes: null
```

```yaml
config_yaml:
  run:
    name: null
    output_dir: results
  seed: 42
  deterministic: true
  mode: coarse_binary
  device: nvidia
  data:
    train_path: data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet
    val_path: data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet
    test_path: data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet
    encoding: null
    cache_features: false
  model:
    name: null
    input_channels: null
    num_classes: 2
  training:
    epochs: 3
    batch_size: 512
    num_workers: 0
    learning_rate: 0.001
    weight_decay: 0.0001
    class_weighting: balanced
    early_stopping_patience: 2
    mixed_precision: false
```

```yaml
model_spec:
  model_name: null
  file_path: null
  builder_function: null
  input_shape: null
  output_shape: [batch, num_classes]
  key_modules: []
  required_config_fields: []
  expected_parameter_count: null
  expected_memory_notes: null
```

```yaml
research_continuity:
  idea_fingerprint: null
  already_researched_family_overlap: null
  closest_duplicate_risk: null
  do_not_repeat_if_this_fails: []
  suggested_next_search_directions: []
```

## 13. Prompt Maintenance Notes For Codex

Tell Codex how this prompt should be improved after it receives your packet. Keep this concrete and scoped.

Use this table:

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|

Do not propose weakening the leakage rules, label rules, falsification requirements, or anti-duplicate requirements.

## 14. Final Sanity Check

End with this checklist, filled in:

- Downloadable Markdown file created:
- Filename follows required date/time/day/timezone/slug pattern:
- No forbidden engine features used as inputs:
- Does not fabricate labels:
- Not a routine CNN/ResNet/Transformer variant:
- Minimal current-data experiment exists:
- Falsification criterion is concrete:
- Codex can implement without asking for missing architecture details:
- Prompt maintenance notes included for Codex:
- Repetition check against imported research packets completed:
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant:
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant:
- Not a deterministic nuisance-orthogonal projection bottleneck variant:
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Möbius-constellation, or pseudo-likelihood packets:
- Not an exact near-duplicate of imported orbit-symmetry, tempo-intervention, credal-evidence, rule-partition-invariance, kinematic-commutator, or masked-codec packets:
- Not an exact near-duplicate of imported cubical Euler/Betti topology, Hall-defect overload, or king-cage/king-escape path-DP packets:
- Not an exact near-duplicate of imported FCA/Galois-closure, denoising-score-field, or non-backtracking-edge-walk packets:
````
