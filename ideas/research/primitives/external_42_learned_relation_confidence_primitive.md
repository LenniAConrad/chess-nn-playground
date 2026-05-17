# p041 Learned Relation Confidence Primitive

**File name:** `p041_learned_relation_confidence_primitive.md`

## Primitive thesis

**Primitive thesis.** The right primitive for this repo is not a learned topology module. It is a **learned confidence layer over exact chess topology**. The reason is already established by i018: the model builds a mover-oriented 64-square tactical incidence complex with 12 typed relations, and its central falsifier—degree-preserving random rewiring of those relation masks—drops mean test PR-AUC from **0.8752** to **0.8328**. That is strong evidence that exact chess relation structure matters, so the next step is to keep the topology fixed and learn which of the already-valid edges deserve more or less mass in context. fileciteturn15file0 fileciteturn14file0 fileciteturn30file0

Concretely, the primitive should sit **between** deterministic relation construction and downstream consumption. It should receive board features plus fixed masks for attacks, defenses, rays, pins, king-zone attacks, and candidate moves, and emit **weighted relation masks** whose support is still bounded by the original chess masks. In symbols: if an edge is absent in the deterministic mask, it must stay absent after learning. That preserves the falsifiable thesis of i018 while upgrading its current mostly binary relation weighting into something board-conditioned and reusable across architectures. fileciteturn14file0 fileciteturn19file0 fileciteturn20file0

This direction is aligned with the broader graph-learning literature: relational inductive bias is valuable when the problem already comes with meaningful entities and lawful relations, and message-passing operators become stronger when edge labels or gates are learned from features on top of a fixed graph rather than replacing the graph with unconstrained all-pairs mixing. Edge-conditioned filters, message-passing neural networks, and gated graph convolutions all support that pattern. citeturn5academia1turn0academia1turn5academia2turn0academia2

It is also worth building because the repo already shows that **small, reusable primitives can add real signal to i018** when fused cleanly. The existing hybrid path uses additive gated fusion, and several hybrids improved i018 by roughly **+0.005 to +0.0065 PR-AUC**, which is the repo’s own threshold for a “real” lift. A relation-confidence module is therefore a plausible *complement* to the sheaf, not just an alternative architecture. fileciteturn13file0 fileciteturn10file0 fileciteturn14file0

## Formal math definition

Let a board tensor \(x\) induce:

- square features \(H(x) \in \mathbb{R}^{B \times N \times d}\), with \(N=64\),
- deterministic relation masks \(M(x) \in \{0,1\}^{B \times R \times N \times N}\),
- deterministic edge attributes \(A(x) \in \mathbb{R}^{B \times R \times N \times N \times f}\),

where \(R\) spans the relation vocabulary used by the consumer. For i018, the current relation vocabulary is the 12 typed tactical relations already emitted by `TacticalIncidenceBuilder`; for move-aware consumers, the repo also already exposes deterministic legal-move graphs and typed legal-move graphs. fileciteturn15file0 fileciteturn19file0 fileciteturn20file0 fileciteturn47file0 fileciteturn51file0

Define a **Learned Relation Confidence Primitive** with three parts:

\[
q_i = W_q h_i,\qquad k_j = W_k h_j
\]

\[
e_{ijr} = \phi\!\left(a_{ijr}\right)
\]

\[
s_{ijr}
=
\alpha_r
\left[
u^\top e_{ijr}
+
\left((U q_i) \odot (V k_j)\right)^\top m_r
+
b_r
\right]
\]

Here:

- \(h_i, h_j\) are source and target square tokens,
- \(a_{ijr}\) is the deterministic chess edge-feature vector for edge \((i,j,r)\),
- \(\phi\) is a small edge MLP,
- \(m_r \in \mathbb{R}^k\) is a relation embedding or low-rank relation vector,
- \(\alpha_r\) is a learned relation-specific scalar gate,
- the bilinear term uses **low rank** \(k \ll d\).

Then produce masked confidence scores:

\[
\tilde c_{ijr} = \sigma\!\left(\frac{s_{ijr}}{\tau_r}\right)
\]

\[
c_{ijr} = M_{ijr} \cdot \operatorname{NormOrPrune}_{r,i}\!\left(\tilde c_{ijr}\right)
\]

\[
\widehat M_{ijr} = \lambda_r \, c_{ijr}
\]

where \(\lambda_r\) is an optional second relation-scale parameter, and `NormOrPrune` is one of:

- identity,
- masked row normalization,
- masked sparsemax,
- masked \(\alpha\)-entmax,
- masked differentiable top-\(k\) followed by renormalization.

Because normalization or pruning only acts **inside the active support of \(M\)**, this remains a topology-preserving operator rather than generic dense attention. Sparsemax and entmax are suitable when we want row-sparse mass over active edges; differentiable sparse top-\(k\) is suitable when we want an explicit hard budget per source square and relation. citeturn0academia0turn4academia1turn2academia1

A clean downstream contract is then:

\[
\text{weighted\_mask} = \widehat M
\]

and the consumer replaces its binary mask \(M\) or scalar edge weight \(w_e\) with \(\widehat M\). For i018, that means substituting \(\widehat M\) into the sheaf coboundary and energy computation in place of the binary/hand-weighted relation plane. For graph or compile-scatter consumers, it means the message passing operator aggregates through \(\widehat M\) instead of the original binary adjacency. This follows the same structural pattern as the repo’s existing move-graph and legal-edge compile-scatter primitives, which already use deterministic chess-derived adjacency plus vectorized gated message passing. fileciteturn15file0 fileciteturn44file0 fileciteturn45file0 fileciteturn47file0 fileciteturn51file0

The low-rank term is the right compromise between expressivity and speed. It borrows the useful part of multi-relational scoring from bilinear and Tucker-style factorization without turning the primitive into a full relation-learning stack. DistMult-style bilinear scoring gives a lightweight relation-conditioned interaction term; Tucker-style factorization and R-GCN-style multi-relation parameter sharing justify keeping relation-specific structure compact rather than allocating a fully separate dense scorer per relation. citeturn5academia0turn1academia0turn1academia1

![Learned relation confidence dataflow](sandbox:/mnt/data/p041_learned_relation_confidence_flow.png)

## Inputs, outputs, and chess features

**Inputs.** The primitive should accept the current-board tensor only, following the repo’s board-only contract. In the simplest version that means `(B, 18, 8, 8)` with `simple_18`, plus either square tokens from i018’s `SquareTokenEncoder` or the cheaper primitive-side `SquareTokenEmbedder`. It should also accept deterministic relation masks coming from either `TacticalIncidenceBuilder` or the rule-graph helpers that compute attacks, rays, ray transmittance, first blockers, legal move graphs, and typed legal move graphs. The important constraint is that nothing from engine scores, CRTK metadata, verification fields, or future boards enters the module. fileciteturn15file0 fileciteturn16file0 fileciteturn39file0 fileciteturn41file0 fileciteturn46file0 fileciteturn50file0 fileciteturn51file0

**Outputs.** The primitive should return a reusable dictionary, not just one fused logit. The core outputs should be `edge_logits`, `edge_confidence`, and `weighted_relation_masks`, all shaped `(B, R, 64, 64)`. Optional sparse outputs should include `topk_indices` and `topk_values` per `(batch, relation, source)` row. Diagnostics should include `relation_gate`, `mean_confidence_per_relation`, `kept_fraction_per_relation`, and `confidence_entropy_per_relation`, plus one exported `candidate_move_confidence` tensor when move relations are enabled. That contract makes the primitive reusable as a sheaf-weighting layer, a graph edge-strength layer, or a conv-style mixer.

**Chess features used.** The edge-feature vector should be deterministic and explicitly chess-aware. A good first-pass set is:

- source piece class and side,
- target piece class / empty / own / enemy category,
- relation type embedding,
- distance bucket or exact ray step,
- slider-direction family when relevant,
- pin-status bit from `king_ray_pin_candidate`,
- king-zone bit from the deterministic king-zone masks,
- target piece value,
- ray transmittance or first-blocker information on slider edges,
- candidate-move piece type and capture/no-capture metadata for move relations,
- source and target square tokens from the current board. fileciteturn15file0 fileciteturn19file0 fileciteturn20file0 fileciteturn41file0 fileciteturn46file0 fileciteturn50file0 fileciteturn51file0

The most important design choice here is that **piece identity and tactical context must be explicit**. i018 already distinguishes attacks, defenses, king-zone attacks, visible rays, knight/pawn attacks, and pins; the graph model already computes target-value and exchange-soundness summaries; and the move-graph primitives already compile deterministic pseudo-legal move connectivity. A confidence module that ignores those typed signals and only looks at generic square embeddings would waste the repo’s strongest inductive bias. fileciteturn15file0 fileciteturn41file0 fileciteturn47file0 fileciteturn51file0

## Integration targets

**i018.** The cleanest integration point is directly between `TacticalIncidenceBuilder` and the sheaf diffusion blocks. i018 already computes `relation_masks` in dense `(B, R, 64, 64)` form and passes them into `SheafDiffusionBlock`; the primitive simply changes that hand-weighted binary tensor into a learned weighted tensor with the same support. Nothing else in the sheaf contract needs to change. Better still, the repo already has i249, a numerically equivalent fast execution path for i018 that vectorizes relation handling and is intended precisely to avoid the original per-relation Python loop overhead. That makes **i249 the best engineering host** for the first implementation of p041. fileciteturn15file0 fileciteturn19file0 fileciteturn20file0 fileciteturn21file0 fileciteturn34file0 fileciteturn17file0

There is also a strong thesis reason to start inside i018. The sheaf already provides the exact falsifier the primitive needs, and it is the repo’s clearest proof that exact relation topology matters. If a learned confidence layer cannot improve or sharpen the edge weights **there**, it is unlikely to be worth porting elsewhere. fileciteturn14file0 fileciteturn30file0

**Graph model.** The repo’s `exchange_soundness_graph_network` is the best graph-side consumer. It already defines a learned attack/defense graph with attacker intensity, defender intensity, cheapest-attacker and cheapest-defender values, exact target-value fields, and a differentiable static-exchange-style score over candidate targets. A relation-confidence primitive can improve that model in two complementary ways: first, by replacing plain attack/defense intensity heuristics with confidence-weighted exact attack and defense edges; second, by exporting confidence-weighted attacker sets that feed a softer, more faithful cheapest-attacker / cheapest-defender estimator. In that model, target value is not a vague heuristic; it is already a first-class field, so it should be one of the strongest p041 edge features. fileciteturn41file0

**Conv auxiliary.** For a conv auxiliary target, the repo gives two good options. The first is `auxiliary_reconstruction_boardnet`: use p041 to create extra confidence planes or relation-summary maps, then either concatenate them as side channels or ask the auxiliary decoder to reconstruct them. That makes the primitive regularizable: a degenerate, nearly uniform confidence field becomes visible immediately through reconstruction diagnostics. The second, cleaner A/B harness is `BT4PrimitiveMixerNet`, which was explicitly built so that the spatial mixer can be swapped between conv, attention, or a chess-aware primitive with the same `(B, C, 8, 8) -> (B, C, 8, 8)` contract. p041 can therefore be exposed as a confidence-weighted scatter mixer and compared directly against the baseline conv mixer without changing the rest of the residual tower. fileciteturn39file0 fileciteturn48file0

If a lighter-weight integration path is desired before a full mixer rewrite, the repo’s additive hybrid convention is already established: `final_logit = sheaf_logit + sigmoid(gate) * primitive_logit`. That pattern is proven in the existing `oriented_sheaf_plus_primitive` wrapper and is appropriate for a first sheaf-side pilot if you want the primitive to ship as a separate head rather than an in-trunk mask layer. fileciteturn13file0

## Falsifiers, ablations, and validation protocol

**Falsifiers.** The core falsifier should be a direct extension of the existing i018 falsifier: train the learned-confidence module on **degree-preserving scrambled masks** instead of real chess masks. If performance remains within about **0.01 PR-AUC** of the real-mask version, reject the thesis that “exact topology plus learned confidence” is doing real work. That test matters more than any internal ablation because it attacks the primitive’s central claim directly. fileciteturn14file0 fileciteturn30file0

A second falsifier should test for **confidence collapse**. After training, replace the learned edge confidences by a single per-relation scalar equal to that relation’s mean active-edge confidence. If metrics barely move, then p041 did not learn edge-level structure; it only learned coarse relation reweighting, which is a much smaller primitive. A third falsifier should test **feature semantics** by scrambling source/target piece-type labels or target-value features while leaving the topology fixed. If that does not hurt, the module is not earning its chess-aware feature set.

**Ablations.** The minimum useful ablation grid is:

- binary masks only,
- relation gates only,
- edge MLP only,
- low-rank bilinear term only,
- full model,
- full model without pin/king-zone/target-value features,
- full model without candidate-move relations,
- full model with sparsemax/entmax,
- full model with differentiable top-\(k\),
- full model with hard top-\(k\) at inference.

The repo’s own primitive heads also make ablation style clear: p006, p009, and p011 all include topology randomization controls, gate disablement, type-sharing controls, and trunk-only baselines. p041 should follow that standard instead of inventing a looser evaluation culture. fileciteturn47file0 fileciteturn51file0 fileciteturn44file0

**Validation protocol.** Use the existing i018 paper-grade setup so comparisons stay interpretable: `simple_18`, the repo’s canonical CRTK-tagged split, three seeds, 20 epochs with minimum 10, patience 5, batch size 256, and the same optimizer/scheduler family. The base comparator is i018 at **0.8752 ± 0.0045** test PR-AUC; the mandatory falsifier comparator is the scrambled-mask configuration at **0.8328 ± 0.0012**. In addition to PR-AUC, report retained-edge fraction, confidence entropy, per-relation sparsity, and wall-clock metrics like `samples_per_second` and `fit_elapsed_seconds`. fileciteturn14file0 fileciteturn30file0 fileciteturn31file0

The repo already gives practical lift thresholds through its hybrid summary logic: a mean delta of **at least +0.005 PR-AUC** counts as a real lift, while **|delta| < 0.002** is effectively a wash. That is a good decision rule for p041 too. Promote only if p041 clears the lift threshold or matches baseline while producing materially sparser, more interpretable active-edge structure. Reject if it lands in wash territory and the confidence maps are nearly uniform. fileciteturn10file0

## Complexity, speed, and implementation sketch

**Complexity and speed.** Let \(B\) be batch size, \(N=64\), \(R\) the number of relation types, \(f\) the edge-feature width, \(m\) the edge-MLP hidden width, and \(k\) the low-rank dimension. A dense vectorized scorer costs roughly

\[
O\!\left(BRN^2(fm + k)\right)
\]

for the confidence pass, plus the downstream cost of whatever consumes \(\widehat M\). For a drop-in i018 replacement, asymptotics do not change much because i018 already stores dense `(B, R, 64, 64)` relation tensors and runs dense pairwise residual math in the sheaf block. The engineering goal is therefore not to beat i018’s asymptotic cost on day one; it is to add confidence scoring **without falling back to Python edge loops**. fileciteturn19file0 fileciteturn20file0 fileciteturn17file0 fileciteturn34file0

The repo already shows the correct implementation style for that. i249 replaces per-relation Python looping with vectorized `einsum` plus chunked batched computation; p011 uses dense typed masks and batched `bmm` because ragged edge lists are a poor fit for scout-scale PyTorch eager mode; p006 and p009 use stop-gradient deterministic rule graphs with dense vectorized gather/scatter. p041 should follow that same pattern: static-shape tensors, masked broadcast feature assembly, batched scoring, and optional rowwise `topk`. fileciteturn34file0 fileciteturn44file0 fileciteturn45file0 fileciteturn47file0 fileciteturn51file0

A useful practical distinction is this:

- **Dense drop-in p041** is the **first** version to build, because it preserves semantics and integrates cleanly with i249.
- **Sparse consumed p041** is the **second** version, where rowwise top-\(k\) pruning is followed by sparse gather/scatter so there is an actual wall-clock payoff.

Top-\(k\) without sparse consumption mostly improves interpretability and regularization. Top-\(k\) with sparse consumption is what changes downstream compute meaningfully. Differentiable top-\(k\) operators are mature enough to make that second step realistic. citeturn2academia1

**Implementation sketch.** A repo-native layout could look like this:

```python
# src/chess_nn_playground/models/primitives/learned_relation_confidence.py

class LearnedRelationConfidence(nn.Module):
    def __init__(self, token_dim=32, edge_hidden=48, low_rank_dim=8,
                 relation_count=13, sparse_mode="sigmoid", topk=None):
        super().__init__()
        self.token_embed = SquareTokenEmbedder(input_channels=18, embed_dim=token_dim)
        self.rel_embed = nn.Embedding(relation_count, low_rank_dim)
        self.q_proj = nn.Linear(token_dim, low_rank_dim)
        self.k_proj = nn.Linear(token_dim, low_rank_dim)
        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(edge_attr_dim),
            nn.Linear(edge_attr_dim, edge_hidden),
            nn.GELU(),
            nn.Linear(edge_hidden, 1),
        )
        self.rel_gate = nn.Parameter(torch.zeros(relation_count))
        self.topk = topk
        self.sparse_mode = sparse_mode

    def forward(self, board, relation_masks, edge_attrs):
        tokens = self.token_embed(board)                     # (B, 64, d)
        q = self.q_proj(tokens)                             # (B, 64, k)
        k = self.k_proj(tokens)                             # (B, 64, k)

        q_e = q[:, None, :, None, :]                        # (B, R, 64, 1, k)
        k_e = k[:, None, None, :, :]                        # (B, R, 1, 64, k)
        rel = self.rel_embed.weight[None, :, None, None, :]# (1, R, 1, 1, k)

        low_rank = ((q_e * k_e) * rel).sum(dim=-1)         # (B, R, 64, 64)
        edge_bias = self.edge_mlp(edge_attrs).squeeze(-1)  # (B, R, 64, 64)

        logits = (edge_bias + low_rank) * torch.sigmoid(self.rel_gate)[None, :, None, None]
        logits = logits.masked_fill(relation_masks <= 0, float("-inf"))

        conf = masked_confidence(logits, relation_masks, mode=self.sparse_mode)
        if self.topk is not None:
            conf = masked_row_topk(conf, relation_masks, k=self.topk)

        weighted_masks = relation_masks * conf
        return {
            "edge_logits": logits,
            "edge_confidence": conf,
            "weighted_relation_masks": weighted_masks,
        }
```

That module should then be wired into three adapters: a sheaf adapter that swaps `relation_masks` for `weighted_relation_masks`, a graph adapter that aggregates weighted attack/defense evidence, and an optional BT4 mixer that scatter-mixes channels through those weighted masks. This mirrors existing repo patterns much more closely than inventing a brand-new ragged-graph infrastructure. fileciteturn44file0 fileciteturn45file0 fileciteturn47file0 fileciteturn51file0 fileciteturn48file0 fileciteturn34file0

**Recommended first experiment.** Start with **i249-fast as the host**, not the slower original i018 module. Use only the **existing 12 i018 relations** in the first run. Feed the confidence scorer these edge features: source piece class, target piece class / empty, relation type, distance bucket, pin bit, king-zone bit, and target value. Use a **shared edge MLP plus a rank-8 low-rank bilinear term**, with one learned scalar gate per relation. Use **masked sigmoid confidence with no top-\(k\) in run one** so the experiment isolates the value of learned edge confidence before adding sparsity semantics. Initialize the confidence bias so active edges start near a mean confidence of roughly 0.9, train on the same three seeds as the paper-grade i018 runs, and compare against both the i018 baseline and the scrambled-mask counterpart. If that version does not beat the baseline by at least **+0.005 PR-AUC** or produce clearly non-uniform, motif-aligned confidence maps, stop. If it does, phase two should add rowwise top-\(k\) pruning and the candidate-move branch. fileciteturn14file0 fileciteturn10file0 fileciteturn34file0 fileciteturn41file0

The reason this is the right first experiment is simple: it is the **lowest-risk, highest-signal test**. It maximally reuses the repo’s strongest existing evidence, its fastest execution path, and its cleanest falsifier. If p041 cannot win there, it should not be generalized. If it does win there, it becomes a genuinely reusable primitive for sheaf, graph, and conv-style models in the repo.