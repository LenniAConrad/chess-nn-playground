# Codex Handoff Packet: Soft Formal-Concept Closure Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0922_tuesday_los_angeles_concept_closure.md`
- Generated at: 2026-04-21 09:22 America/Los_Angeles
- Weekday: Tuesday
- Timezone: America/Los_Angeles
- Idea slug: `concept_closure`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Soft Formal-Concept Closure Network
- One-sentence thesis: Chess puzzle-likeness is partly expressed by small closed sets of co-occurring rule-derived board attributes, so a differentiable Galois-closure bottleneck over a current-board formal context should detect tactical motif coherence that marginal counts, CNN texture, and static attack graphs miss.
- Idea fingerprint: `current-board square/object attribute incidence matrix + learned soft FCA intent probes + differentiable object→attribute→object closure summaries + binary puzzle-like logits; no move tree, no engine scores, no transport, no sheaf, no topology, no source metadata`
- Why this is not a common CNN/ResNet/Transformer variant: The central computation is not convolution, residual depth, square self-attention, or graph message passing; it is an explicit formal-context derivation operator `intent -> extent -> closed intent` whose hard limit is the Formal Concept Analysis closure operator.
- Current-data minimal experiment: Train the model on `simple_18` using the existing `crtk_sample_3class` train/val/test parquet split for the standard coarse binary target, report ordinary binary metrics plus the required fine-label `0/1/2 -> predicted 0/1` diagnostic matrix, and compare to the existing simple CNN and residual CNN with the same data, epochs, class weighting, and seed.
- Smallest central falsification ablation: Replace each board's object-attribute incidence matrix with a row/column-sum-preserving bipartite edge-swap randomization within attribute groups, then run the same soft-closure layer; this preserves candidate count, square/object attribute degree, attribute prevalence, material/coordinate marginals, side-to-move, and global nuisance features while destroying the co-instantiation relation that FCA closure needs.
- Expected information gain if it fails: A failure against the row/column-preserving rewire says the closure profile is not adding puzzle signal beyond marginal board descriptors, and future cycles should avoid FCA/closure bottlenecks unless they introduce a materially different formal context or target.

## 3. Problem Restatement And Data Contract

The project is `chess-nn-playground`. The active task is chess puzzle-likeness classification from board positions. The model receives a board tensor and emits two logits:

```text
input:  (batch, C, 8, 8)
output: (batch, 2)
```

The coarse target is:

```text
fine label 0 -> binary 0  # known non-puzzle
fine label 1 -> binary 1  # verified near-puzzle
fine label 2 -> binary 1  # verified puzzle
```

The benchmark must still report the rectangular diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Use the current split:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

The full roughly 45M-row parquet dataset must not be used directly until streaming support exists.

Allowed current encodings:

```text
simple_18
lc0_static_112
lc0_bt4_112
```

The first experiment should use `simple_18` because its current-board plane semantics are explicit enough to build deterministic rule attributes. `lc0_static_112` and `lc0_bt4_112` support should fail closed unless Codex can confirm an exact channel map for current board piece planes, side-to-move, castling, and en-passant. History planes in `lc0_bt4_112` must not be treated as deterministic geometry unless exporter support and channel semantics are known; if a learned neural adapter consumes history channels later, that adapter must be separate from the rule-derived closure builder.

Leakage checklist:

- Safe as neural-network input: deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack geometry derived only from the current board.
- Use caution: full legal-move generation, legal move counts, checkmate/stalemate predicates, forced-line search, or move-tree consequences. They are leakage-prone unless explicitly justified as rule-only, label-independent, engine-free, and ablated.
- Forbidden as neural-network input: Stockfish or other engine scores, principal variations, node counts, mate scores, verification metadata, source labels, proposed labels, unresolved-candidate status, dataset provenance, and any field derived from the labeling pipeline.
- The formal-context builder may compute pseudo-legal square attacks and line-of-sight blockers from the current board only. It must not enumerate legal moves, choose candidate moves, evaluate resulting boards, or call an engine.

## 4. Research Map

External sources used as design seeds:

| Source | What is borrowed | What is not copied |
|---|---|---|
| Bernhard Ganter and Rudolf Wille, *Formal Concept Analysis: Mathematical Foundations* (Springer, 1999). Public mirror found at `https://math.ubbcluj.ro/~csacarea/wordpress/wp-content/uploads/Prof.-Dr.-Bernhard-Ganter-Prof.-Dr.-Rudolf-Wille-auth.-Formal-Concept-Analysis_-Mathematical-Foundations-1999-Springer-Verlag-Berlin-Heidelberg.pdf` | The formal context `(G,M,I)`, derivation operators, Galois connection, concept lattice, and closure fixed-point view. | No concept-lattice enumeration, no offline mining of labels, no hand-coded chess concepts, and no full lattice visualization. |
| Radim Belohlavek, “Introduction to Formal Concept Analysis,” `https://phoenix.inf.upol.cz/esf/ucebni/formal.pdf` | The closure-operator framing and the idea that attributes shared by an extent define an intent. | No exact NextClosure lattice construction; this packet uses a differentiable, sampled-probe closure layer. |
| Jingyi Xu et al., “A Semantic Loss Function for Deep Learning with Symbolic Knowledge,” ICML 2018, `https://proceedings.mlr.press/v80/xu18h.html` | The general lesson that symbolic constraints can be represented differentiably and used with neural training. | No semantic-loss compilation, no output-constraint SAT, and no structured-output constraint target. |
| M. M. Grespan et al., “Evaluating Relaxations of Logic for Neural Networks,” IJCAI 2021, `https://www.ijcai.org/proceedings/2021/0387.pdf` | The caution that different t-norm relaxations behave differently; motivates making the soft closure temperature and relaxation explicit. | No reliance on a specific theorem from the paper and no claim that a particular fuzzy logic is optimal for chess. |
| Honghua Dong et al., “Neural Logic Machines,” ICLR 2019, `https://openreview.net/forum?id=B1xY-hRctX` | The broad idea that neural modules can operate over object/property tensors and learn lifted logical patterns. | No lifted arity expansion, no variable-binding rule induction, no planning tasks, and no NLM architecture. |
| Juho Lee et al., “Set Transformer,” ICML 2019, `https://proceedings.mlr.press/v97/lee19d.html` | The engineering precedent that finite learned inducing/probe vectors can summarize set-structured data efficiently. | No vanilla set self-attention over 64 squares and no Transformer as the central mechanism. |

Candidate search trace:

| Serious candidate mechanism considered | Why it lost to Soft Formal-Concept Closure |
|---|---|
| Causal IRM across unsupervised source-like clusters | Interesting but brittle: real source/provenance is forbidden as input, deterministic material/phase partitions were already imported, and unsupervised clusters risk rediscovering dataset artifacts without a new board operator. |
| Differentiable SAT-style consistency of checking, pinning, and escape clauses | Too close to legal-move or mate-oracle territory unless heavily weakened; the weakened version became a static attack rule list and lost a clean falsifier. |
| Quadtree or 2D grammar MDL motif codec | Could be fresh, but it is close to imported pseudo-likelihood and masked-code-length packets unless it becomes a full grammar project; the minimal experiment would mainly test compression, not tactical coherence. |
| Sparse Boolean DNF over rule predicates | Too near sparse witness-piece bottlenecks and Möbius/ANOVA constellation packets unless the closure operator is added; FCA gives the DNF-like motif detector a distinct mathematical object and falsifier. |
| Calibration/selective-prediction head for fine-label-1 ambiguity | Useful for reporting but not a representation idea; close to imported ordinal/credal uncertainty families. |
| Poset/order-ideal model of piece dominance relations | Elegant but collapses into a static attack/defense graph with different pooling, which is explicitly over-researched. |
| Differentiable parser over rank/file/diagonal strings | Too close to the imported ray-language automaton family. |
| Spectral compression of pressure fields | Too close to imported graph Laplacian, Hodge, pressure Betti, and topology packets. |
| Soft Formal-Concept Closure Network | Selected because its central object is the object-attribute closure operator, it does not enumerate moves or lattice concepts, and it has a sharp row/column-preserving randomized falsifier. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Formal Concept Analysis | A per-board formal context `K_x=(G,M,I_x)` with `G=64` square objects and `M≈96-160` deterministic current-board attributes | `A ∈ {0,1}^{batch,64,M}` | Row/column-sum-preserving swaps in `A` | Not a graph, sheaf, Hodge operator, move bag, OT coupling, topology curve, or pseudo-likelihood model |
| Galois closure | Learned soft intent probes `q_k`, soft extent membership `E(q_k)`, and soft closed-intent vector `C_x(q_k)` | `q ∈ [0,1]^{K,M}`, `extent ∈ [0,1]^{batch,K,64}`, `closed_intent ∈ [0,1]^{batch,K,M}` | Remove `C_x(q)` and classify only from intent-to-extent scores and marginals | Distinct from ordinary attention because the computation has a hard closure-limit with monotonicity/idempotence properties |
| Symbolic-neural relaxation | Temperature-controlled soft AND for derivation operators | Same as above plus scalar temperatures | Change relaxation to marginal pooling with same parameter count | Not a semantic-loss output constraint and not a neural logic machine |
| Tactical motif coherence | Closure expansion and stability statistics of attributes shared by the probe's support | `z ∈ R^{batch,K,D}` then pooled to logits | Attribute-semantics rewiring and groupwise column permutation | Tests co-instantiation of attributes, not piece count, attack count, material, or side-to-move alone |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Increase depth or width of the simple CNN | `src/chess_nn_playground/models/cnn.py` | Ordinary capacity scaling is already represented and does not create a new falsifiable chess operator. |
| Another residual CNN variant | `src/chess_nn_playground/models/residual_cnn.py` | A standard residual stack is covered by current baselines and would mostly test optimization/capacity. |
| LC0-style CNN or residual CNN with more filters | Existing LC0 BT4-style CNN/residual variants | Too close to imported LC0-style baselines and not a new research idea. |
| Vanilla ViT over 64 squares | Generic square Transformer | Explicitly disallowed as a core idea; it has weak chess-specific structure and no sharp falsifier. |
| Plain GNN on squares or pieces | Static attack-defense graph ideas | Static graph/message-passing is over-represented and risks becoming another attack-defense graph packet. |
| Hyperparameter tuning, optimizer changes, or longer training | Shared trainer configs | Tuning may improve leaderboard numbers but would not test a new hypothesis about puzzle-likeness. |
| Ensemble of existing CNNs and residual CNNs | Any baseline ensemble | Ensembling is disallowed and obscures mechanism-level interpretation. |
| One-ply legal or pseudo-legal move-delta pooling | Imported move-delta landscape/spectrum family | Already researched; also risks leaning on move-generation artifacts rather than current-board structure. |
| Sheaf, Hodge, curvature, or tactical Laplacian over attacks | Imported sheaf/Hodge packets | Explicitly over-researched; changing edge labels or pooling would be a near-duplicate. |
| Sinkhorn or piece-target transport pressure | Imported OT packets | Current-board piece-target/material-target transport bottlenecks are already imported. |
| Static cubical topology over pressure fields | Imported Euler/Betti topology packets | Threshold/topology summaries over pressure maps are already covered. |
| Hall-defect or matching overload | Imported Hall-defect obligation matroid packet | Defender obligation and matching defects are already represented. |
| King escape path or percolation dynamic program | Imported king-cage/path-DP packets | Path free-energy and escape percolation are already researched. |
| Ordinal or evidential uncertainty head for fine labels | Imported ordinal ladder and credal evidence packets | Good diagnostic direction, but not novel here and not the selected representation mechanism. |
| Class-conditioned pseudo-likelihood or masked code-length | Imported pseudo-likelihood and masked-codec packets | Compression is promising but this exact family is already covered. |
| Ray-string finite automata | Imported ray-language automaton | Ordered ray token languages are already researched. |
| Möbius/ANOVA piece constellations | Imported high-order constellation packet | Explicit high-order interactions over piece tuples are already represented; FCA closure is selected specifically to avoid enumerating piece tuples. |

## 6. Mathematical Thesis

### Input space definition

Let `X_C = {0,1}^{C×8×8}` be the encoded-board input space. For the first experiment, `C=18` and the adapter interprets the 12 piece planes, side-to-move, castling, and en-passant planes of `simple_18`.

For each input `x ∈ X_18`, define a deterministic formal context:

```text
K_x = (G, M, I_x)
```

where:

- `G = {0,1,...,63}` is the set of square objects.
- `M` is a fixed attribute vocabulary of deterministic current-board predicates.
- `(g,m) ∈ I_x` if square `g` has attribute `m` in board `x`.

Attributes may include coordinate predicates, occupancy predicates, side-relative piece predicates, king-distance bins, color complex, edge/center bins, side-to-move broadcast attributes, castling/en-passant broadcast attributes, and pseudo-legal current-board attack/line predicates. They must not include legal move counts, engine evaluations, mate status, source provenance, or label metadata.

Let `A_x ∈ {0,1}^{64×M}` be the incidence matrix with `A_x[g,m]=1` iff `(g,m)∈I_x`.

### Label/target definition

Let the fine label be `ell ∈ {0,1,2}`. The training target is:

```text
y = 1[ell >= 1]
```

Fine label `1` is not fabricated and is not treated as class `2`; it is only mapped to the binary positive target for the default benchmark. The fine label is used for diagnostics, not as an input feature.

### Data distribution assumptions

Assume train, validation, and test rows are sampled from the provided split distribution. The model assumes that, after excluding forbidden metadata, some puzzle-likeness signal remains in current-board structure. It does not assume that all puzzles are tactics of one type, nor that class `1` and class `2` have identical distributions.

The key data assumption is weaker than “FCA concepts equal chess tactics”:

```text
There exist a small number of attribute intents whose per-board closure profiles have nonzero conditional mutual information with y beyond material/coordinate/attack marginals.
```

### Allowed symmetry or equivariance assumptions

Chess is not fully invariant to rotations or reflections because pawns, castling, en-passant, board orientation, and side-to-move matter. This idea uses only side-relative attributes where they are semantically safe, such as “own piece,” “enemy piece,” and “rank toward promotion from side-to-move.” It does not perform full board orbit averaging, Reynolds pooling, D4 symmetry, color-flip quotienting, or side-intervention.

### Core hypothesis

Puzzle-like positions more often contain compact, closed, rule-coherent attribute extents: sets of squares or pieces whose common attributes imply additional shared structure. Non-puzzles may share the same row/column marginals—same material, similar attack counts, same number of occupied squares, similar king-distance bins—but lack the same object-level co-instantiation pattern. A differentiable closure bottleneck should therefore improve near-puzzle/puzzle discrimination over marginal and convolutional baselines.

### Formal object/operator introduced by the idea

For a hard formal context `K=(G,M,I)`, define FCA derivation operators:

```text
B'  = { g ∈ G : for all m ∈ B, (g,m) ∈ I }       for B ⊆ M
A'  = { m ∈ M : for all g ∈ A, (g,m) ∈ I }       for A ⊆ G
cl_M(B) = B''                                    intent closure
cl_G(A) = A''                                    extent closure
```

A formal concept is a pair `(A,B)` such that `A'=B` and `B'=A`.

The neural layer uses learned soft intent probes `q_k ∈ [0,1]^M`, `k=1..K`, and temperature-controlled relaxations:

```text
miss_x(g,k) = sum_m q_k[m] * (1 - A_x[g,m]) / (sum_m q_k[m] + eps)
E_x[g,k]    = exp(-miss_x(g,k) / tau_extent)

w_x[g,k]    = E_x[g,k] / (sum_h E_x[h,k] + eps)

miss_attr_x(k,m) = sum_g w_x[g,k] * (1 - A_x[g,m])
C_x[k,m]         = exp(-miss_attr_x(k,m) / tau_closure)
```

`E_x[:,k]` is the soft extent of probe `k`; `C_x[k,:]` is its soft closed intent. The classifier sees closure statistics such as extent mass, extent entropy, closure mass, closure expansion `||relu(C_x[k]-q_k)||_1`, closure violation `||relu(q_k-C_x[k])||_1`, and learned embeddings of `C_x[k,:]` and the soft extent.

### Proposition / optimization principle

**Proposition 1, hard closure facts.** For any finite formal context `K`, `cl_M(B)=B''` is extensive, monotone, and idempotent:

```text
B ⊆ cl_M(B)
B1 ⊆ B2 => cl_M(B1) ⊆ cl_M(B2)
cl_M(cl_M(B)) = cl_M(B)
```

The fixed points of `cl_M` are exactly the intents of formal concepts.

**Proof sketch.** The derivation operators form an antitone Galois connection between subsets of `G` and subsets of `M`. Standard Galois-connection closure arguments give extensivity, monotonicity of the double application, and idempotence. If `B=B''`, then with `A=B'`, we have `A'=B`, so `(A,B)` is a formal concept. Conversely every concept intent satisfies `B=A'=B''`.

**Neural optimization objective.** Train parameters `theta` by minimizing:

```text
L(theta) =
  CE_balanced(y, f_theta(A_x, globals_x))
  + lambda_sparse * R_intent(q)
  + lambda_div * R_diversity(q)
  + lambda_idem * || SoftClose_x(SoftClose_x(q)) - SoftClose_x(q) ||_1
```

The last three terms are optional regularizers. The minimal experiment may set `lambda_idem=0` and use only light intent sparsity/diversity. The objective is not claimed to prove puzzle structure; it operationalizes the hypothesis that a closure-stable bottleneck is useful.

### What is actually proven

The hard FCA closure operator has the closure properties above. The row/column-preserving randomization is guaranteed to preserve the first-order object and attribute degrees of the incidence matrix while changing which attributes co-occur on which squares. Therefore, if the full model beats the rewire ablation under matched training, the improvement cannot be explained solely by those preserved first-order marginals.

### What remains only hypothesized

It is only a hypothesis that puzzle-like positions have higher-useful closure structure than non-puzzles under this attribute vocabulary. It is also unproven that the proposed soft relaxation faithfully captures the best hard concepts for classification. The experiment tests both.

### Counterexamples where the idea should fail

- A puzzle whose decisive feature depends on a specific legal move consequence not visible in current-board static attributes.
- A quiet defensive puzzle where the relevant property is multi-ply zugzwang rather than closed tactical geometry.
- Dataset artifacts where positives differ mainly by material, rating, source, or construction pipeline, all of which the closure layer should not be allowed to use.
- Positions where the attribute vocabulary is too coarse, so the useful hard concepts do not exist in `M`.
- Positions where pseudo-legal attack attributes overstate illegal pinned attacks; the model may learn misleading closure motifs.

### Self-critique

The strongest objection is that the closure layer may simply repackage attack counts, king distance, and material into a more complicated MLP. The central rewire ablation is designed to attack that objection: it preserves object degrees, attribute prevalences, global nuisance features, and grouped marginal statistics while destroying object-level attribute co-occurrence. If the full model does not beat this control, abandon the mechanism. The experiment is still worth running because it is cheap, current-data compatible, and tests a mathematical object not already present in the imported packets.

## 7. Architecture Specification

### Module names

Implement the main model in:

```text
src/chess_nn_playground/models/formal_concept_closure.py
```

Suggested classes:

```text
Simple18BoardAdapter
RuleAttributeBuilder
SoftConceptClosureLayer
ConceptClosureReadout
SoftFormalConceptClosureNet
```

Suggested registry builder:

```text
build_formal_concept_closure_net(config) -> torch.nn.Module
```

### Forward-pass steps

Assume `simple_18` input for the first experiment.

1. **Input**

   ```text
   x: float tensor [B, C, 8, 8]
   ```

   For first experiment, require `C=18`.

2. **Board parsing**

   `Simple18BoardAdapter` extracts:

   ```text
   piece_planes: [B, 12, 8, 8]
   side_to_move: [B, 1] or scalar per board
   castling:     [B, 4]
   en_passant:   [B, 8] or [B,1,8,8] converted to safe global/file attributes
   occupancy:    [B, 64]
   piece_type:   [B, 64, 6]
   color:        [B, 64, 2]
   own/enemy:    [B, 64, 2]
   ```

   The adapter must validate that no square contains more than one piece plane above threshold. If semantics are unknown, raise a clear error instead of silently guessing.

3. **Rule attribute construction**

   `RuleAttributeBuilder` creates a binary incidence matrix:

   ```text
   A_bool: [B, 64, M]
   globals: [B, G0]
   ```

   Recommended first vocabulary, with `M≈96-160`:

   - coordinate: file one-hot, rank one-hot, side-relative rank one-hot, square color, edge/corner/center/ring bins;
   - occupancy: empty, occupied, own occupied, enemy occupied, piece type, own piece type, enemy piece type, slider/leaper/pawn/king/value-tier flags;
   - king geometry: Chebyshev distance bins to own king and enemy king, same rank/file/diagonal as own/enemy king;
   - pseudo-legal pressure: attacked-by-own and attacked-by-enemy flags by piece family, total attack count bins by side, defended occupied square flags;
   - ray geometry: square lies between a king and an opposing slider on a clear rank/file/diagonal, occupied piece is a static pin candidate, slider has clear pseudo-legal ray to enemy king;
   - global broadcast should be minimal; put side-to-move, castling, and en-passant file in `globals` rather than making every closure probe trivially depend on them.

   Pseudo-legal attacks are computed from current occupancy only. Do not enumerate legal moves or resulting positions.

4. **Soft concept closure**

   Parameters:

   ```text
   raw_intents: [K, M]
   q = sigmoid(raw_intents / intent_temperature): [K, M]
   ```

   Compute:

   ```text
   miss:         [B, K, 64]
   extent:       [B, K, 64]
   extent_norm:  [B, K, 64]
   closed_intent:[B, K, M]
   ```

   Suggested relaxation:

   ```python
   q_sum = q.sum(dim=-1).clamp_min(eps)                     # [K]
   miss = einsum("km,bom->bko", q, 1 - A) / q_sum[None,:,None]
   extent = exp(-miss / tau_extent)

   extent_norm = extent / extent.sum(dim=-1, keepdim=True).clamp_min(eps)
   miss_attr = einsum("bko,bom->bkm", extent_norm, 1 - A)
   closed = exp(-miss_attr / tau_closure)
   ```

5. **Concept summaries**

   For every board and concept probe:

   ```text
   extent_mass      = sum_o extent[b,k,o]
   extent_entropy   = entropy(extent_norm[b,k,:])
   closure_mass     = sum_m closed[b,k,m]
   expansion_l1     = sum_m relu(closed[b,k,m] - q[k,m])
   violation_l1     = sum_m relu(q[k,m] - closed[b,k,m])
   closure_cosine   = cosine(closed[b,k,:], q[k,:])
   closed_embed     = closed[b,k,:] @ W_attr              # [D_attr]
   extent_embed     = extent_norm[b,k,:] @ square_proj(A) # [D_attr]
   probe_embed      = learned_probe_embedding[k]          # [D_probe]
   z[b,k]           = concat(stats, closed_embed, extent_embed, probe_embed)
   ```

   Shapes:

   ```text
   z: [B, K, D_z]
   ```

6. **Readout**

   Apply a shared small MLP to each concept summary:

   ```text
   h = concept_mlp(z): [B, K, H]
   pooled = concat(mean_K(h), max_K(h), logsumexp_K(h), globals): [B, 3H + G0]
   logits = classifier_mlp(pooled): [B, 2]
   ```

   This keeps the trainer contract unchanged.

### Parameter-count estimate

Default first-run config:

```text
K = 64 concept probes
M = 128 attributes, auto-detected after builder construction
D_attr = 32
D_probe = 16
H = 64
classifier hidden = 128
```

Approximate trainable parameters:

```text
raw_intents:          64 * 128              = 8,192
attribute embedding:  128 * 32              = 4,096
square projection:    128 * 32 + bias       = 4,128
probe embeddings:     64 * 16               = 1,024
concept MLP:          ~9k-15k
classifier MLP:       ~35k-60k
batchnorm/layernorm:  <2k
total:                ~70k-120k
```

If Codex widens `H` to 96 or `K` to 96, the model should still stay below roughly 300k parameters. This is intentionally much smaller than many CNN baselines; the experiment tests structure, not capacity.

### FLOP / complexity estimate

Let:

```text
B = batch size
O = 64 square objects
M = number of attributes
K = concept probes
```

Soft closure cost is dominated by two dense contractions:

```text
O(B*K*O*M)
```

With `B=512`, `K=64`, `O=64`, `M=128`, this is about `268M` multiply/add-like scalar operations, but implemented as batched tensor operations. Attribute construction includes pseudo-legal attack masks and sliding rays; for 8x8 boards this is small compared with training convolutional baselines.

### Candidate-set memory and chunking

The generated “candidate set” is the fixed set of `K` learned concept probes, not legal moves.

Main activations:

```text
A_float:        [B, 64, M]
extent:         [B, K, 64]
closed_intent:  [B, K, M]
z:              [B, K, D_z]
```

Memory is:

```text
O(B*64*M + B*K*64 + B*K*M + B*K*D_z)
```

For `B=512`, `K=64`, `M=128`, float32 `closed_intent` uses about `512*64*128*4 ≈ 16 MB`, and `extent` uses about `8 MB`. If `K` or `M` grows, implement chunking over `K`:

```text
for concept_chunk in chunks(raw_intents, chunk_size=32):
    compute extent/closed/z for that chunk
    accumulate pooled mean/max/logsumexp
```

This avoids storing all concept activations at once.

### Required config fields

```yaml
model:
  name: formal_concept_closure
  input_channels: 18
  num_classes: 2
  num_concepts: 64
  attr_embedding_dim: 32
  probe_embedding_dim: 16
  concept_hidden_dim: 64
  classifier_hidden_dim: 128
  tau_extent: 0.15
  tau_closure: 0.15
  intent_temperature: 1.0
  closure_eps: 1.0e-6
  use_attack_attributes: true
  use_ray_attributes: true
  adapter: simple_18
  simple18_piece_order: standard
  semantic_rewire_ablation: false
```

### Encoding-adapter assumptions

- `simple_18`: supported in the first experiment. The adapter must have an explicit piece-plane order. If the repo already defines this order, import it. If not, require `simple18_piece_order` in config and fail if absent.
- `lc0_static_112`: not supported by default. Add support only if the repo has a documented current-board channel map. Current-board piece planes may feed the rule attribute builder; unknown/history-like planes must be ignored by the deterministic builder.
- `lc0_bt4_112`: not supported by default for deterministic geometry because BT4 history planes are zero-filled until exporter support exists. If a later learned adapter uses all 112 planes, it must be clearly separated from the rule closure path and ablated.
- Unknown channel semantics: fail closed with a `ValueError`, not a silent best guess.

### Return value

`SoftFormalConceptClosureNet.forward(x)` returns:

```text
logits: [B, 2]
```

No trainer changes should be needed beyond model registry/config registration.

## 8. Loss, Training, And Regularization

Primary loss:

```text
balanced cross-entropy over binary target y ∈ {0,1}
```

Class weighting:

```text
class_weighting: balanced
```

Optional auxiliary losses:

```text
R_intent(q)    = mean_k (mean_m q[k,m] - target_density)^2
R_diversity(q) = mean_{i<j} cosine(q_i, q_j)^2
R_idem         = mean || SoftClose(SoftClose(q)) - SoftClose(q) ||_1
```

Recommended first run:

```text
lambda_sparse = 0.001
target_density = 0.08 to 0.15
lambda_div = 0.001
lambda_idem = 0.0
```

`lambda_idem` is optional because computing `SoftClose(SoftClose(q))` costs extra and may over-regularize early experiments. Keep the minimal run simple.

Batch size expectations:

```text
batch_size: 512 for simple_18 on a typical GPU or CPU-friendly test run
```

Optimizer defaults:

```text
AdamW
learning_rate: 1.0e-3
weight_decay: 1.0e-4
epochs: 3
early_stopping_patience: 2
mixed_precision: false for first deterministic comparison
```

Regularizers:

- Light dropout `0.05-0.10` inside readout MLP only.
- Intent density/diversity regularization as above.
- No label smoothing in the first comparison unless all baselines use it.
- No data augmentation in the first comparison unless all baselines use it.

Determinism requirements:

- Set torch, numpy, and Python seeds.
- Use deterministic trainer mode already present in project configs.
- The semantic-rewire ablation must be seeded and reproducible.
- If edge swaps are stochastic per batch, log the seed and implement deterministic epoch-based generators.

What must stay unchanged for fair comparison:

- Same train/val/test split.
- Same binary label mapping.
- Same metrics and report pipeline.
- Same class weighting policy.
- Same epoch budget and early-stopping policy as the selected baseline comparison.
- No extra data, no engine features, no source metadata.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Row/column-preserving context rewire | Performs bipartite double-edge swaps in `A_bool` within attribute groups while preserving each square's group degree and each attribute's board prevalence; global material/side/castling/EP features stay unchanged | Tests whether object-attribute co-instantiation and closure, not marginals, carry signal | If this matches the full model, closure semantics are not useful; abandon this FCA version |
| No-closure extent-only model | Computes soft extents from `q` but replaces `closed_intent` with marginal attribute pooling and extent stats only | Tests whether the second FCA derivation `extent -> intent` matters | If equal, the model is just a soft rule-probe presence detector |
| Marginal-only nuisance control | Uses material vector, occupancy count, side-to-move, castling/EP globals, attack-count histograms, and attribute column sums; no per-square incidence | Tests whether first-order board statistics explain performance | If equal to full, there is no evidence for closure-level structure |
| Attribute group column permutation | Permutes attribute identities within semantic groups, preserving column sums but corrupting meaning while keeping `M`, `K`, and parameter count | Tests whether specific chess semantics matter or only attribute density | If equal, the vocabulary semantics may be irrelevant |
| Coordinate-only closure | Keeps coordinate, occupancy, and global features but removes attack/ray attributes | Tests dependence on pseudo-legal tactical predicates | If strong, the model may be learning dataset layout/artifacts rather than tactics |
| Attack/ray-only closure | Keeps pressure/ray attributes and minimal occupancy, removes absolute file/rank identities except side-relative distance bins | Tests whether closure signal depends on tactical geometry rather than square memorization | If this beats full, absolute coordinates may be a nuisance |
| Random intent probes frozen | Freezes `q_k` from a seeded random sparse distribution; trains only embeddings/readout | Tests whether learned intents are important | If equal, the readout may be using generic random features, weakening the concept-learning claim |
| Harder sparse intent probes | Adds top-k or entmax-like sparsification to `q_k` after warmup | Tests whether interpretable sparse intents improve or harm | If full soft model wins, fuzzy combinations are likely needed |
| No attack attributes, CNN adapter added | Replaces closure tactical attributes with a tiny CNN of matched parameter count | Tests whether the closure layer beats a cheap learned texture adapter | If CNN control wins decisively, closure vocabulary may be too brittle |
| Label-shuffle sanity | Train on shuffled labels with same pipeline | Verifies no leakage from metadata or split artifacts | Any above-chance performance indicates a serious bug or leakage |

Structured-object randomized control:

- The central row/column-preserving context rewire is the semantics-destroying control for this formal-context operator.
- It should preserve:
  - number of true attributes per square within groups;
  - prevalence of each attribute within the board;
  - total occupied squares and material vector;
  - side-to-move, castling, en-passant globals;
  - attack-count histogram by side/piece group if attack attributes are grouped correctly.
- It should destroy:
  - which coordinates, piece identities, pressure predicates, and ray predicates co-occur on the same square/object;
  - therefore the hard FCA closure relation.

This idea does not use a rule-generated move set or candidate legal-move set. Count-only controls should therefore target concept/marginal counts, not move counts.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

```text
simple_18 simple CNN, same split and epoch budget
simple_18 residual CNN, same split and epoch budget
best existing simple_18 baseline result in leaderboard, if already available
```

Do not compare the first `simple_18` run directly against LC0 BT4 models as the main success claim unless the encoding difference is explicitly called out.

Metrics to inspect:

- accuracy;
- balanced accuracy;
- precision/recall/F1 for binary positive class;
- AUROC if probability outputs are already exported;
- AUPRC if already exported;
- calibration/ECE optional, not central;
- rectangular fine-label `0/1/2 -> predicted 0/1` matrix for every main and central ablation run.

Required near-puzzle diagnostic:

```text
class-1 recall at a validation-chosen threshold that gives a matched fine-label-0 false-positive rate, preferably 5% and 10%.
```

Also report:

```text
class-1 precision among predicted positives
class-2 recall at the same matched fine-label-0 false-positive thresholds
```

Required artifacts:

```text
results/.../metrics.json
results/.../confusion_binary.png or .csv
results/.../confusion_fine3_by_binary.csv
results/.../predictions_test.parquet
results/.../ablation_metrics.json
results/.../near_puzzle_threshold_diagnostics.csv
ideas/<idea_id>_<slug>/report.md
```

Success threshold:

- Full model improves over the best same-encoding baseline by at least `+1.0` percentage point balanced accuracy or `+2.0` percentage points positive F1, **and**
- Full model improves class-1 recall at matched fine-label-0 FPR by at least `+3.0` percentage points, **and**
- Full model beats the row/column-preserving rewire ablation by at least `+1.0` percentage point balanced accuracy or `+2.0` percentage points class-1 recall at matched FPR.

Failure threshold:

- Full model is within `±0.5` percentage points of the row/column-preserving rewire ablation on balanced accuracy and class-1 matched-FPR recall, or
- Full model underperforms both simple CNN and residual CNN with no compensating near-puzzle diagnostic gain, or
- Attribute ablations show performance is almost entirely from coordinate-only or marginal-only controls.

What result would make us abandon the idea:

```text
The row/column-preserving context rewire matches or beats the full model, and marginal-only controls explain nearly all class-1/class-2 performance.
```

What result would justify scaling:

```text
The full closure model beats same-encoding baselines and central ablations, especially on fine-label-1 recall at matched fine-label-0 FPR, with stable results across at least three seeds.
```

Scaling path, only after success:

- Increase `K` from 64 to 128.
- Add a second closure pass with tied probes.
- Add safe support for `lc0_static_112` only after exact channel-map validation.
- Keep the row/column-preserving rewire ablation in every scaled report.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_0922_concept_closure/idea.yaml` | Create | Machine-readable idea metadata, central thesis, status, expected compute, and links to configs/results. |
| `ideas/20260421_0922_concept_closure/math_thesis.md` | Create | Expanded version of Section 6, including hard FCA closure facts, soft relaxation, and failure cases. |
| `ideas/20260421_0922_concept_closure/architecture.md` | Create | Module-level design, tensor shapes, attribute vocabulary, and adapter assumptions. |
| `ideas/20260421_0922_concept_closure/implementation_notes.md` | Create | Safe board parsing, pseudo-legal attack/ray attribute construction, deterministic rewire implementation, and fail-closed channel semantics. |
| `ideas/20260421_0922_concept_closure/trainer_notes.md` | Create | Loss, optimizer, class weighting, seed/determinism, and fair-comparison requirements. |
| `ideas/20260421_0922_concept_closure/ablations.md` | Create | Ablation plan table, required central controls, and diagnostics. |
| `ideas/20260421_0922_concept_closure/train.py` | Create | Thin entrypoint that loads the config and calls the existing shared trainer; do not fork the trainer unless necessary. |
| `ideas/20260421_0922_concept_closure/config.yaml` | Create | Copy of the first-run config from Section 12. |
| `ideas/20260421_0922_concept_closure/report_template.md` | Create | Report skeleton requiring baseline comparison, binary/fine confusion, near-puzzle diagnostics, and central ablation results. |
| `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this idea to imported memory after Codex consumes it; add anti-duplicate guidance for FCA/formal-context closure if it fails or succeeds. |
| `src/chess_nn_playground/models/formal_concept_closure.py` | Create | `Simple18BoardAdapter`, `RuleAttributeBuilder`, `SoftConceptClosureLayer`, `ConceptClosureReadout`, `SoftFormalConceptClosureNet`. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `formal_concept_closure` and builder function. |
| `configs/formal_concept_closure_simple18.yaml` | Create | Main benchmark config for `simple_18`. |
| `configs/formal_concept_closure_rewire_simple18.yaml` | Create | Central row/column-preserving rewire ablation config. |
| `configs/formal_concept_closure_marginal_simple18.yaml` | Create | Marginal-only nuisance control config. |
| `tests/test_formal_concept_closure.py` | Create | Unit tests for shape contract, fail-closed adapter behavior, deterministic rewire preserving row/column/group degrees, no multi-piece square parsing, and logits shape. |
| `tests/test_rule_attribute_builder.py` | Create if useful | Focused tests for pseudo-legal attack maps and ray predicates from simple toy boards without using legal move generation or engine calls. |

Implementation notes:

- Keep all rule-derived computations inside the model file or a small helper imported by it.
- Do not add dependencies beyond PyTorch/numpy unless the project already uses them.
- Make ablation modes config-driven, not separate model classes when possible.
- For deterministic row/column-preserving swaps, start with simple seeded swaps per attribute group and board. If exact preservation is too slow, provide a deterministic fallback that preserves row degrees and approximate column degrees, but mark the report accordingly. The preferred control preserves both.
- Log the resolved attribute vocabulary and channel map into the result directory for reproducibility.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0922_tuesday_los_angeles_concept_closure.md
  generated_at: "2026-04-21 09:22 America/Los_Angeles"
  weekday: Tuesday
  timezone: America/Los_Angeles
  idea_slug: concept_closure
  format: markdown
```

```yaml
idea_yaml:
  idea_id: "20260421_0922_concept_closure"
  name: "Soft Formal-Concept Closure Network"
  slug: concept_closure
  status: draft
  created_at: "2026-04-21 09:22 America/Los_Angeles"
  author: ChatGPT Pro
  short_thesis: "Use differentiable Formal Concept Analysis closure over current-board rule-derived square attributes to detect closed tactical motif coherence for puzzle-likeness."
  novelty_claim: "Central operator is soft Galois intent/extents/closed-intent closure over a board formal context, not convolution, ResNet scaling, square Transformer, sheaf/Hodge, move-delta pooling, Sinkhorn transport, topology, Hall overload, king path DP, pseudo-likelihood, or ordinal/credal evidence."
  expected_advantage: "Should improve fine-label-1 near-puzzle recall at matched fine-label-0 FPR if puzzle-like positions contain closed co-instantiated board attributes beyond material and attack-count marginals."
  central_falsification_ablation: "Row/column-sum-preserving bipartite rewire of the object-attribute incidence matrix within semantic groups before the same closure layer."
  target_task: coarse_binary
  input_representation: "simple_18 primary; lc0_static_112/lc0_bt4_112 fail closed unless exact current-board channel map is provided"
  output_heads: "binary logits [batch,2]"
  compute_notes: "Closure cost O(batch*num_concepts*64*num_attributes); default K=64, M≈128, under about 120k parameters."
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/formal_concept_closure_simple18.yaml
  model_path: src/chess_nn_playground/models/formal_concept_closure.py
  latest_result_path: null
  notes: "Keep central rewire and marginal-only controls; do not use legal move generation, engine data, source labels, or verification metadata."
```

```yaml
config_yaml:
  run:
    name: formal_concept_closure_simple18
    output_dir: results
  seed: 42
  deterministic: true
  mode: coarse_binary
  device: nvidia
  data:
    train_path: data/splits/crtk_sample_3class/split_train.parquet
    val_path: data/splits/crtk_sample_3class/split_val.parquet
    test_path: data/splits/crtk_sample_3class/split_test.parquet
    encoding: simple_18
    cache_features: false
  model:
    name: formal_concept_closure
    input_channels: 18
    num_classes: 2
    adapter: simple_18
    simple18_piece_order: standard
    num_concepts: 64
    attr_embedding_dim: 32
    probe_embedding_dim: 16
    concept_hidden_dim: 64
    classifier_hidden_dim: 128
    tau_extent: 0.15
    tau_closure: 0.15
    intent_temperature: 1.0
    closure_eps: 1.0e-6
    use_attack_attributes: true
    use_ray_attributes: true
    semantic_rewire_ablation: false
    marginal_only_ablation: false
    dropout: 0.05
    intent_density_target: 0.10
    lambda_intent_density: 0.001
    lambda_intent_diversity: 0.001
    lambda_idempotence: 0.0
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
  model_name: formal_concept_closure
  file_path: src/chess_nn_playground/models/formal_concept_closure.py
  builder_function: build_formal_concept_closure_net
  input_shape: [batch, input_channels, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18BoardAdapter
    - RuleAttributeBuilder
    - SoftConceptClosureLayer
    - ConceptClosureReadout
    - SoftFormalConceptClosureNet
  required_config_fields:
    - input_channels
    - num_classes
    - adapter
    - num_concepts
    - attr_embedding_dim
    - tau_extent
    - tau_closure
    - simple18_piece_order
  expected_parameter_count: "about 70k-120k for K=64, M≈128; under 300k for moderate widening"
  expected_memory_notes: "Stores A [B,64,M], extent [B,K,64], closed_intent [B,K,M]; chunk over K if K or M grows."
```

```yaml
research_continuity:
  idea_fingerprint: "current-board formal context over square attributes + learned soft intent probes + differentiable FCA closure summaries + binary puzzle-like classifier"
  already_researched_family_overlap: "Uses pseudo-legal attack/ray attributes as atoms, but not static attack graph message passing, sheaf/Hodge, transport, topology, move deltas, or pseudo-likelihood."
  closest_duplicate_risk: "Could be mistaken for a sparse Boolean motif/DNF or static attack-feature MLP; central distinction is the intent->extent->closed-intent Galois closure and row/column-preserving closure falsifier."
  do_not_repeat_if_this_fails:
    - "Do not propose another FCA/formal-context/Galois-closure bottleneck over current-board square or piece attributes unless the formal context or target is genuinely different and has a new falsifier."
    - "Do not merely change the attribute vocabulary, number of probes, t-norm temperature, or pooling readout."
    - "Do not repackage this as concept lattice enumeration, soft DNF motifs, or closure attention without a different mathematical test."
  suggested_next_search_directions:
    - "If closure fails, explore label-safe selective prediction/calibration for ambiguous near-puzzles without ordinal or credal heads."
    - "Explore causal invariance only if genuinely new environments are available without source-label leakage."
    - "Explore a true grammar or program-induction motif model only if it avoids masked-codec and ray-automaton duplicates."
    - "Explore source-artifact suppression through adversarial bottlenecks if environments can be defined safely and ablated."
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Soft Formal-Concept Closure Network` to imported research memory after implementation, with fingerprint `formal context object-attribute incidence + learned intent probes + differentiable Galois closure + row/column-preserving rewire control`. | Prevents future cycles from repeating FCA/closure with only changed attributes or temperatures. | `Imported Research Memory` |
| Add an anti-duplicate rule: “Do not propose another FCA, concept-lattice, Galois-closure, soft formal-context, or closure-attention bottleneck over current-board square/piece attributes unless the formal object and central falsifier differ materially.” | The current idea creates a new family that should be treated as researched after Codex consumes it. | Anti-duplicate paragraphs after imported packet fingerprints |
| Add a required control for formal-context ideas: row/column-sum-preserving incidence rewires and marginal-only controls. | Makes the key falsifier reusable and prevents closure models from hiding marginal shortcuts. | `Ablation Plan` guidance |
| Clarify that pseudo-legal attack/ray predicates may be safe atoms, but using them as the whole idea is not enough. | Avoids future proposals that rename static attack features as new mathematics. | `What Counts As Creative Enough` and leakage boundary |
| Add a note that exact channel maps are required for deterministic rule builders on LC0 encodings. | Prevents silent misuse of `lc0_static_112` or `lc0_bt4_112` history/current-board planes. | `Project Context You Must Respect` or `Problem Restatement` template |
| If the central rewire ablation matches the full model, mark FCA/closure as low-priority for at least several cycles. | Converts empirical failure into research memory rather than repeated near-duplicates. | `Research Continuity` |

Do not weaken leakage rules, label rules, falsification requirements, or anti-duplicate requirements.

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0922_tuesday_los_angeles_concept_closure.md`
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes
- Falsification criterion is concrete: yes
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported research packets completed: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Möbius-constellation, or pseudo-likelihood packets: yes
- Not an exact near-duplicate of imported orbit-symmetry, tempo-intervention, credal-evidence, rule-partition-invariance, kinematic-commutator, or masked-codec packets: yes
- Not an exact near-duplicate of imported cubical Euler/Betti topology, Hall-defect overload, or king-cage/king-escape path-DP packets: yes
