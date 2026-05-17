# i250 Learned Relation Confidence Sheaf

*Thesis: keep i018’s exact side-to-move-oriented chess relation topology, but replace its relation-wide weighting with normalized edge-wise confidence learned from board-only chess features, so the model can answer which specific attacks, defenses, rays, king-zone edges, and pin edges actually drive puzzle-likeness.*

Filename: `i250_learned_relation_confidence_sheaf.md`

## Thesis and background

The repo’s current `puzzle_binary` task is a board-only binary classifier: source fine labels `0` and `1` map to non-puzzle, and fine label `2` maps to puzzle, while evaluation keeps the full `3x2` diagnostic matrix because near-puzzle false positives are the main pressure test. The benchmark goal document explicitly frames PR-AUC, near-puzzle false positives, puzzle recall, and matched-recall comparisons as more important than raw accuracy, and the README ties all benchmark claims to one shared data contract and artifact contract. fileciteturn18file0L3-L3 fileciteturn5file0L3-L3

The shipped i018 implementation is a bespoke sheaf model, not a generic CNN or transformer substitute. Its idea metadata, architecture notes, and model code all agree on the same core recipe: side-to-move canonicalization; a 12-relation tactical incidence builder over the 64 squares; a square token encoder; learned relation-specific sheaf restriction maps; bounded sheaf diffusion; an optional triad-defect pool; and a one-logit tactical readout for the current `puzzle_binary` contract. The current base config uses `simple_18`, `channels: 64`, `hidden_dim: 96`, `depth: 2`, `dropout: 0.1`, `batch_size: 256`, `learning_rate: 0.0007`, `epochs: 20`, and `bce_with_logits`. fileciteturn8file0L3-L3 fileciteturn10file0L3-L3 fileciteturn27file0L3-L3 fileciteturn28file0L3-L3 fileciteturn17file0L3-L3

Why i018 is the right parent is no longer speculative. Its own math thesis reports a 3-seed base result of `0.8752 ± 0.0045` test PR-AUC, and the central falsifier—degree-preserving randomization of the typed relation masks—drops that to `0.8328 ± 0.0012`, a mean delta of `-0.0424`. The verdict recorded in the repo is blunt: real chess geometry is doing real work. That is exactly the kind of result that argues for *preserving topology* and learning better weights on top of it, rather than replacing the whole mechanism with generic attention. Repo-generated hybrid experiments also show i018 still has headroom: several primitive grafts lift mean test PR-AUC into the `0.8808–0.8817` range. fileciteturn9file0L3-L3

There is also a very specific architectural gap in the current code. i018 already learns relation-specific restriction maps and a **single scalar gate per relation** inside each sheaf block, but it does **not** learn edge-wise confidence within a relation. In other words, it can learn that “rook rays” matter more than “knight attacks” on average, but it cannot directly learn that *this* rook ray into the king ring matters while *that* rook ray on the other side of the board is noise. That is the narrowest high-information next step. fileciteturn28file0L3-L3

## Mathematical design

The theory base for this move is solid. Sheaf neural networks generalize graph diffusion by attaching vector spaces and linear maps to nodes and edges, which is useful when relations are asymmetric, signed, and typed rather than uniform. Neural sheaf diffusion then shows that non-trivial sheaves give finer control over diffusion than ordinary graph Laplacians, while the cellular-sheaf Laplacian formulation keeps the core positive-semidefinite structure \(L = \delta^\top \delta\). citeturn4academia1turn4academia0turn4academia2

### Exact relation masks

Keep i018’s exact board-conditioned relation topology. Let \(x \in \mathbb{R}^{B\times C\times 8\times 8}\) be the input board tensor, and let the oriented board adapter produce mover-relative piece state \(P \in [0,1]^{B\times 64\times 13}\) and occupancy \(O \in [0,1]^{B\times 64}\). For each shipped relation type \(r \in \{1,\dots,12\}\), build the exact binary or bounded relation mask

\[
M_r(x) \in [0,1]^{B\times 64\times 64},
\]

using the same deterministic chess-rule geometry already in i018: knight masks, oriented pawn masks, king-zone masks, rook/bishop/queen visible rays, and pin-candidate construction from between-square blocker masks. The proposal does **not** add new topology in v1; it only adds learned confidence on top of the topology that is already implemented and has already passed a geometry falsifier. fileciteturn10file0L3-L3 fileciteturn27file0L3-L3 fileciteturn28file0L3-L3

Use the current implementation’s concrete 12 relations, not the older packet’s provisional list. The April packet still described a looser “10–12 relation” set and mentioned a separate `king_adjacency`, but the shipped model solidified the design as a 12-relation builder without a separate king-adjacency relation. i250 should extend the code path that exists, not resurrect packet-only relations and unintentionally change the benchmarked parent. fileciteturn25file0L3-L3 fileciteturn27file0L3-L3

### Learned edge confidence

For every active edge \((u,v,r)\) with \(M_r(u,v)=1\), compute a board-only edge feature vector

\[
\phi_{r}(u,v;x).
\]

I recommend the following exact feature families for \(\phi_r\):

\[
\phi_r = \operatorname{concat}(
e_r,\;
t_{\text{src}}(u),\;
t_{\text{dst}}(v),\;
\mathrm{val}_{\text{src}}(u),\;
\mathrm{val}_{\text{dst}}(v),\;
d(u,v),\;
\mathrm{pin}(u,v),\;
\mathrm{xray}(u,v),\;
\mathrm{kingzone}_{\text{dst}}(v),\;
\mathrm{att\_count}(v),\;
\mathrm{def\_count}(v),\;
\psi_u,\;
\psi_v,\;
\psi_u \odot \psi_v
).
\]

Here \(e_r\) is a learned relation embedding; \(t_{\text{src}}, t_{\text{dst}}\) are source/target piece-type encodings from the mover-oriented piece state; \(\mathrm{val}\) is a fixed heuristic piece-value scalar; \(d(u,v)\) is geometric distance; \(\mathrm{pin}\) and \(\mathrm{xray}\) are deterministic chess features; \(\mathrm{att\_count}\) and \(\mathrm{def\_count}\) come from exact relation in-degrees; and \(\psi_u = P_s h_u^{(0)}\), \(\psi_v = P_t h_v^{(0)}\) are **low-rank** projections of the initial square tokens. This keeps the module chess-specific and contextual without turning it into dense generic attention. The confidence net only scores already-active i018 edges. fileciteturn10file0L3-L3 fileciteturn25file0L3-L3

Map relation types to a small number of semantic confidence groups \(g(r)\), and use a tiny grouped MLP rather than a separate network per relation:

\[
\tilde{\alpha}_{b,r,u,v}
=
\varepsilon_r + (1-\varepsilon_r)\,
\sigma\!\left(f_{g(r)}(\phi_{b,r,u,v}) + b_r\right),
\qquad
\varepsilon_r \in (0, 0.1].
\]

This gives a positive edge confidence with a small learned floor, which prevents dead edges and keeps the downstream Laplacian well behaved. The group mapping is part of the architectural prior: direct attacks/defenses can share one small network, visible rays another, king-zone pressure another, leapers/pawns another, and pin candidates a final one. That is the right scope if the goal is “learn which chess edges matter” rather than “discover an unrestricted pairwise communication rule.” This is a design recommendation, not a claim already present in the repo.

The crucial trick is to make confidence **relative within relation**, not a second global gate. Let

\[
\bar{\alpha}_{b,r}
=
\frac{1}{|E_{b,r}|}
\sum_{(i,j): M_{b,r,i,j}=1}
\tilde{\alpha}_{b,r,i,j},
\]

and define a normalized confidence

\[
\hat{\alpha}_{b,r,u,v}
=
\frac{\tilde{\alpha}_{b,r,u,v}}
{\bar{\alpha}_{b,r} + \delta}.
\]

Then the final sheaf weight entering block \(\ell\) is

\[
w^{(\ell)}_{b,r,u,v}
=
M_{b,r,u,v}\;
g^{(\ell)}_r\;
\hat{\alpha}_{b,r,u,v},
\]

where \(g^{(\ell)}_r\) is the current i018 relation-level gate. This separation is important. It keeps \(g_r\) as the relation-level “how much does this relation family matter” parameter, while \(\hat{\alpha}\) answers the new question “which exact edges inside this relation family matter on this board.” It also makes zero-initialization safe: if the confidence net outputs a constant, then \(\hat{\alpha}\equiv 1\), so the model starts as i018. That makes i250 a low-risk extension rather than a brittle rewrite.

### Confidence-weighted sheaf diffusion

Keep i018’s signed typed coboundary and just replace \(w_e\) with the confidence-weighted \(w^{(\ell)}_{b,r,u,v}\). For stalk state \(z_u^{(\ell)} \in \mathbb{R}^s\),

\[
(\delta_{\rho,w}^{(\ell)} z)_{b,r,u,v}
=
\sqrt{w^{(\ell)}_{b,r,u,v}}\;
\Big(
\rho^{(\ell)}_{\mathrm{dst},r} z_v^{(\ell)}
-
\sigma_r\,
\rho^{(\ell)}_{\mathrm{src},r} z_u^{(\ell)}
\Big).
\]

Then

\[
L_{\rho,w}^{(\ell)}(x)
=
\left(\delta_{\rho,w}^{(\ell)}\right)^\top
\delta_{\rho,w}^{(\ell)}
\]

remains symmetric positive semidefinite because the weights stay nonnegative, exactly the same structural reason used in the i018 thesis and in the sheaf literature. The diffusion step can therefore stay in the same bounded form as i018:

\[
z^{(\ell+1)}
=
z^{(\ell)}
-
\eta_\ell\,
D_\ell^{-1}
\left(\delta_{\rho,w}^{(\ell)}\right)^\top
\delta_{\rho,w}^{(\ell)}
z^{(\ell)},
\]

followed by the existing residual node projection and MLP. This preserves the parent architecture’s mathematical identity while giving it the missing degree of freedom: *intra-relation edge weighting*. fileciteturn9file0L3-L3 fileciteturn28file0L3-L3 citeturn4academia1turn4academia0turn4academia2

## Architecture and modules

### Dataflow from input to logit

The cleanest implementation is to insert one new confidence stage between the existing incidence builder and the existing sheaf blocks:

```text
(B, C, 8, 8)
  -> BoardStateAdapter
       square_raw:   (B, 64, C)
       piece_state:  (B, 64, 13)
       occupancy:    (B, 64)

  -> TacticalIncidenceBuilder
       M:            (B, 12, 64, 64)
       exact attack/defense/ray/pin tensors

  -> SquareTokenEncoder
       h0:           (B, 64, d_model)

  -> RelationFeatureExtractor
       phi:          (B, 12, 64, 64, d_edge) on active edges

  -> GroupedRelationConfidence
       alpha_hat:    (B, 12, 64, 64)

  -> ConfidenceWeightedSheafBlocks x depth
       weighted M:   M * alpha_hat * g_r
       hL:           (B, 64, d_model)
       energies:     (B, depth, 12)

  -> TriadDefectPool
       triad_stats:  (B, 4)

  -> TacticalReadout
       puzzle_logit: (B,)
```

This preserves the current i018 module decomposition almost exactly: the new code path is “extract edge features, predict normalized confidences, multiply them into the existing relation masks, then let the current sheaf blocks operate.” That is why this proposal is much narrower and lower-risk than switching trunks entirely. fileciteturn10file0L3-L3 fileciteturn28file0L3-L3

### Exact chess-specific relations

These are the 12 exact relations in the shipped i018 implementation, and i250 should keep them unchanged. fileciteturn27file0L3-L3 fileciteturn28file0L3-L3

| Relation group | Exact relations kept from i018 |
| --- | --- |
| Direct combat | `us_attacks_them_piece`, `them_attacks_us_piece`, `us_defends_us_piece`, `them_defends_them_piece` |
| King-zone pressure | `us_attacks_empty_near_king`, `them_attacks_empty_near_king` |
| Visible rays | `bishop_ray_visible`, `rook_ray_visible`, `queen_ray_visible` |
| Local leapers | `knight_attack`, `pawn_attack_forward_oriented` |
| Pin geometry | `king_ray_pin_candidate` |

The right place to innovate is therefore **not** relation discovery. It is relation **confidence**. Do not add or remove relations in v1. Make the first experiment about whether the exact relation topology becomes stronger when the model can reweight the active edges inside each relation type.

### New modules and parameter budget

The new modules I would add are:

- `RelationFeatureExtractor`: deterministic board-only feature builder for each active edge, including source/target piece class, normalized piece value, relation embedding, geometric distance, king-zone flags, pin/x-ray flags, attacker count, defender count, and low-rank contextual node projections from `h0`.
- `GroupedRelationConfidence`: five tiny grouped confidence MLPs, one per semantic relation group, with mean-normalized outputs so the model starts as i018 when the confidence net is constant.
- `ConfidenceSummaryHead`: optional diagnostics only, not a separate trunk. Save per-relation confidence mean/max/std, top-k confident edges, pin-edge confidence, king-zone confidence, and high-target-value confidence for prediction artifacts.

Use the repo’s `base / scale_up / scale_xl` naming convention and keep the parent i018 width/depth pattern recognizable. The repo’s paper-ready runner treats those variants as its standard size sweep, with `scale_up` roughly `1.5x` and `scale_xl` roughly `2x` capacity relative to base. The current single-seed scout records base i018 at `91,363` parameters and about `9.0M` estimated FLOPs per position, so the proposal should stay in the same operating regime at base scale. fileciteturn5file0L3-L3 fileciteturn13file0L3-L3

| Variant | `channels` | `hidden_dim` | `depth` | `stalk_dim` | `confidence_ctx_dim` | `confidence_hidden_dim` | Approx. params | Rationale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `base` | 64 | 96 | 2 | 8 | 8 | 24 | **~101k** | Minimal overhead over i018; should train under the same budget. |
| `scale_up` | 96 | 144 | 3 | 12 | 12 | 32 | **~258k** | First serious capacity expansion while staying moderate. |
| `scale_xl` | 128 | 192 | 4 | 16 | 16 | 40 | **~524k** | Upper bound before the confidence head stops being a “cheap extension.” |

Those parameter counts are engineering estimates, not measured repo counts, but they are intentionally modest. The confidence head itself is only a small fraction of the total budget; most parameters still belong to the sheaf trunk and readout. The more important runtime cost comes from per-edge feature evaluation, which is why the grouped low-rank design matters.

A concrete proposed config for the first implementation would look like this:

```yaml
model:
  name: learned_relation_confidence_sheaf
  input_channels: 18
  num_classes: 1
  channels: 64
  hidden_dim: 96
  depth: 2
  stalk_dim: 8
  dropout: 0.1
  encoding: simple_18

  confidence_context_dim: 8
  confidence_hidden_dim: 24
  confidence_group_count: 5
  confidence_floor: 0.05
  normalize_confidence_within_relation: true
  use_triads: true
```

## Training and evaluation

### Training recipe

The safest training recipe is “same as i018 unless a change directly serves the confidence module.” That means the canonical tagged split, the current `puzzle_binary` label mapping, board-only inputs, the NVIDIA path, balanced BCE-with-logits, the existing convergence budget, and repeated-seed paper-grade evaluation. The repo’s reliable protocol is very explicit: paper-grade or promotion-grade claims should use the canonical tagged split, convergence-oriented training, repeated seeds, validation-only model selection, matched baselines, and matched-recall false-positive reporting. fileciteturn19file0L3-L3

A very important implementation note is that an older April 2026 research packet described a different binary mapping in which fine label `1` was treated as positive, but the **current** benchmark goal and the implemented i018 idea metadata both use the current contract `0/1 -> 0`, `2 -> 1`. i250 must follow the current benchmark contract or every comparison to the existing repo results will be invalid. fileciteturn25file0L3-L3 fileciteturn18file0L3-L3 fileciteturn8file0L3-L3

The first training recipe I would ship is:

```yaml
training:
  reliability_tier: paper_grade
  monitor: pr_auc
  epochs: 20
  min_epochs: 10
  min_active_epochs: 10
  early_stopping_patience: 5
  batch_size: 256
  learning_rate: 0.0007
  weight_decay: 0.0001
  class_weighting: balanced
  gradient_clip_norm: 1.0
  mixed_precision: true
  allow_tf32: true
  matmul_precision: high
```

There is one subtle but important reason to write `monitor: pr_auc` explicitly even though the trainer already defaults binary modes to PR-AUC: the trainer’s scoring path falls back to F1, then accuracy, then negative loss if the configured monitor metric is unavailable. For a paper-grade architecture comparison, that fallback is too dangerous to leave implicit. Pin the monitor metric in the YAML. fileciteturn24file0L3-L3 fileciteturn21file0L3-L3

I would also keep the confidence head **zero-initialized after mean-normalization**, so the first forward pass is effectively i018. That makes optimization safer than random edge weights. In practice, that means initialize the grouped confidence MLP output layers to zero, leave the relation-level i018 gates alone, and let training discover deviations from uniform edge confidence only if they improve validation PR-AUC. Because the benchmark forbids feature leakage, every confidence feature must be computed from the board tensor and fixed chess rules only; do not use tactic tags, source metadata, solution moves, engine values, or verification fields as model input. fileciteturn19file0L3-L3 fileciteturn25file0L3-L3

### Evaluation plan

The primary evaluation target should be **mean test PR-AUC across repeated seeds**, because that is both the repo’s primary binary checkpoint metric and the metric that best matches the hard-negative nature of the task. F1 should remain secondary and threshold-dependent. The reliable protocol also requires reporting default threshold `0.5`, validation-best F1 threshold, and validation-derived thresholds for recall `0.80` and `0.85`, with near-puzzle false positives emphasized at the matched-recall points. fileciteturn24file0L3-L3 fileciteturn19file0L3-L3

The paper-grade comparison set for `i250` should therefore be:

- mean and standard deviation of test PR-AUC over seeds `42`, `43`, and `44`;
- mean and standard deviation of test F1 at a validation-derived threshold;
- near-puzzle false positives at validation-derived recall `0.80`;
- near-puzzle false positives at validation-derived recall `0.85`;
- total false positives at the same matched-recall thresholds;
- default `0.5` threshold confusion matrices and the rectangular `3x2` fine-to-binary confusion;
- worst-slice behavior on `hard`, `equal`, `endgame`, `mate_in_1`, `promotion`, and `underpromotion`, because those are the repo’s named pressure slices. fileciteturn19file0L3-L3 fileciteturn18file0L3-L3

I would add one more mechanism-specific evaluation layer that the current repo does not yet emphasize enough: **confidence attribution slices**. Because i250 is explicitly claiming to learn which edges matter, each prediction artifact should save the top-k confident edges per board, grouped by relation, with source square, target square, relation name, confidence, source piece, target piece, and whether the edge was pinned or king-zone related. That makes it possible to compare true positives, near-puzzle false positives, and false negatives at the mechanism level, not just the logit level. This is a proposal, not a factual repo claim.

I would also compute a few board-derived analysis slices offline, without feeding them into the model: positions with at least one pin candidate, positions with high king-zone pressure, positions with high-value attacked targets, and positions with x-ray structures behind the immediate target. Those slices are especially relevant for a confidence-on-exact-edges architecture and can be derived from the same deterministic chess geometry used by i018.

## Falsifiers and failure modes

### Required falsifiers and ablations

The current i018 family already has the most important falsifier: degree-preserving relation scrambling. i250 must keep that falsifier, but it also needs **confidence-specific** falsifiers, because otherwise a positive result would still leave the central scientific question unanswered. The repo’s reliable protocol says major new mechanisms need ablations, and the i018 thesis already set the standard for what a real geometry falsifier looks like. fileciteturn19file0L3-L3 fileciteturn9file0L3-L3

| Experiment | What changes | What it tests | Interpretation |
| --- | --- | --- | --- |
| Real topology, learned confidence | Full i250 | Main result | Reference point |
| Degree-preserving relation scramble | Reuse i018’s per-relation column permutation falsifier | Whether exact chess topology still matters | If the drop is small, reject the family |
| Flat confidence | Force \(\hat{\alpha}\equiv 1\) | Whether edge-wise confidence adds anything beyond i018 | If equal to full i250, the new module is unnecessary |
| Confidence permutation | Keep real masks, but shuffle \(\hat{\alpha}\) across active edges within each relation | Whether **which edge** gets which weight matters | If equal to full i250, the learned weights are not edge-specific in a useful way |
| No chess features | Confidence net sees only low-rank node context, not piece/value/distance/pin/x-ray/count features | Whether the gain is actually chess-specific | If equal to full i250, the module is drifting toward generic pair scoring |
| No piece value | Remove target/source value scalars | Whether material salience matters | Useful for interpretability |
| No king-zone / no pin / no x-ray | Remove each feature family separately | Which tactical cues are load-bearing | Direct mechanism test |
| Mean-normalization off | Let confidence absorb relation-level mass | Whether confidence is merely duplicating global gates | If performance improves only when normalization is off, the design is less interpretable |
| Confidence in readout only | Use confidence summaries, but diffuse with raw i018 masks | Whether the gain comes from edge-weighted diffusion or only from extra features in the head | If equal to full i250, the sheaf weighting claim is weak |

For decision thresholds, I would use the repo’s own promotion logic. The full model should not be called a meaningful improvement unless it clears at least one of the repo’s practical bars: roughly `+0.003` absolute mean PR-AUC across seeds, or at least `1%` fewer near-puzzle false positives at matched recall `0.80` or `0.85` without a compensating regression in precision or puzzle recall. For the topology falsifier, I would keep i018’s standard: a geometry scramble drop of about `>= 0.02` PR-AUC is strong support; a result within about `0.01` of baseline is effectively rejection. fileciteturn19file0L3-L3 fileciteturn9file0L3-L3

### Expected failure modes

The most likely failure mode is **confidence collapse to relation constants**. Because i018 already has relation-level scalar gates, a naïve confidence head could simply relearn a second set of global multipliers and never meaningfully differentiate one edge from another. That is exactly why relation-wise mean-normalization belongs in the design. Without it, the model will have a serious identifiability problem.

The second likely failure mode is **high-tension overfitting**. The benchmark goal repeatedly warns that near-puzzles are the central challenge: many positions look sharp without being verified puzzles. A confidence head that learns to upweight every king-zone edge, every pin candidate, and every heavy-piece ray in sharp positions may improve raw recall while making near-puzzle false positives worse, which would be a benchmark failure even if aggregate accuracy looks decent. fileciteturn18file0L3-L3

The third likely failure mode is **quiet-puzzle blindness**. i018’s existing thesis already acknowledges that quiet zugzwang, study-like, or low-tension positions may not be well captured by static tactical incidence. An edge-confidence extension does not solve that family weakness by itself. It only sharpens the existing tactical graph. If the dataset slice includes genuine quiet puzzles, i250 could become *more* confident on noisy tactical edges and still miss the quiet wins. fileciteturn9file0L3-L3

The fourth likely failure mode is **feature-construction bugs**, especially around orientation and pin/x-ray logic. i018 already depends on side-to-move canonicalization, pinned-blocker geometry, and visible-ray computation through the between-square masks. Adding more derived edge features increases the chance of a subtle bug that looks like an architecture win or loss. That is another reason to keep v1 as a strict extension of the existing i018 builder rather than inventing new topology. fileciteturn10file0L3-L3 fileciteturn27file0L3-L3 fileciteturn28file0L3-L3

## Implementation sketch and recommendation

### PyTorch implementation sketch

The cleanest way to implement i250 is to leave the existing adapter, incidence builder, triad pool, and trainer contract alone, and add one confidence module that multiplies into the existing relation masks before the sheaf blocks run. The sketch below is intentionally narrow and compatible with the current repo style. The omitted `build_edge_features(...)` function is where the exact chess features live.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


RELATION_GROUPS = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4], dtype=torch.long)


@dataclass(frozen=True)
class ConfidenceOutput:
    raw_confidence: torch.Tensor          # (B, R, 64, 64)
    normalized_confidence: torch.Tensor   # (B, R, 64, 64)


class GroupedRelationConfidence(nn.Module):
    """
    Scores ONLY the active i018 edges.
    No softmax, no new edges, no topology discovery.
    """

    def __init__(
        self,
        d_model: int,
        relation_count: int = 12,
        context_dim: int = 8,
        hidden_dim: int = 24,
        relation_group_count: int = 5,
        confidence_floor: float = 0.05,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.relation_count = int(relation_count)
        self.context_dim = int(context_dim)
        self.confidence_floor = float(confidence_floor)
        self.eps = float(eps)

        self.src_ctx = nn.Linear(d_model, context_dim)
        self.dst_ctx = nn.Linear(d_model, context_dim)

        # Small learned embeddings for exact chess edge metadata.
        self.relation_emb = nn.Embedding(relation_count, 8)

        # Five small grouped heads:
        # direct combat, king-zone pressure, rays, leapers/pawns, pin.
        input_dim = 24 + 3 * context_dim
        self.group_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(relation_group_count)
            ]
        )

        self.register_buffer("relation_groups", RELATION_GROUPS, persistent=False)

        # Zero-init makes confidence constant; after mean-normalization,
        # the model starts effectively as i018.
        for head in self.group_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        h0: torch.Tensor,                 # (B, 64, d_model)
        active_mask: torch.Tensor,        # (B, R, 64, 64), exact i018 masks
        edge_feats: torch.Tensor,         # (B, R, 64, 64, 24), board-only chess features
    ) -> ConfidenceOutput:
        B, R, N, _ = active_mask.shape
        assert R == self.relation_count, f"expected {self.relation_count} relations, got {R}"
        assert N == 64, f"expected 64 squares, got {N}"

        src = self.src_ctx(h0).unsqueeze(1).unsqueeze(3).expand(B, R, N, N, self.context_dim)
        dst = self.dst_ctx(h0).unsqueeze(1).unsqueeze(2).expand(B, R, N, N, self.context_dim)
        pair = src * dst

        fused = torch.cat([edge_feats, src, dst, pair], dim=-1)
        logits = h0.new_zeros(B, R, N, N)

        for r in range(R):
            group_idx = int(self.relation_groups[r].item())
            logits[:, r] = self.group_heads[group_idx](fused[:, r]).squeeze(-1)

        raw = self.confidence_floor + (1.0 - self.confidence_floor) * torch.sigmoid(logits)
        raw = raw * active_mask

        # Relation-wise mean normalization:
        # confidence redistributes mass inside each relation instead of duplicating global gates.
        denom = active_mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        rel_mean = raw.sum(dim=(2, 3), keepdim=True) / denom
        normalized = raw / rel_mean.clamp_min(self.eps)
        normalized = normalized * active_mask

        return ConfidenceOutput(raw_confidence=raw, normalized_confidence=normalized)


class LearnedRelationConfidenceSheafNet(nn.Module):
    def __init__(self, parent_i018: nn.Module, confidence: GroupedRelationConfidence) -> None:
        super().__init__()
        self.parent = parent_i018
        self.confidence = confidence

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = self.parent.adapter(x)
        incidence = self.parent.incidence(board.piece_state, board.occupancy)
        h0 = self.parent.encoder(board.square_raw, board.piece_state)

        edge_feats = build_edge_features(
            piece_state=board.piece_state,
            occupancy=board.occupancy,
            incidence=incidence,
            h0=h0,
        )

        conf = self.confidence(
            h0=h0,
            active_mask=incidence.relation_masks,
            edge_feats=edge_feats,
        )

        h = h0
        block_energies = []
        block_gates = []

        for block in self.parent.blocks:
            weighted_masks = incidence.relation_masks * conf.normalized_confidence
            h, energy, gates = block(h, weighted_masks)
            block_energies.append(energy)
            block_gates.append(gates)

        triad_stats = self.parent.triad_pool(h, incidence) if self.parent.triad_pool is not None else h.new_zeros(h.size(0), 0)

        readout = build_readout(
            h=h,
            incidence=incidence,
            block_energies=block_energies,
            block_gates=block_gates,
            triad_stats=triad_stats,
            confidence=conf.normalized_confidence,
        )

        logits = self.parent.head(readout).squeeze(-1)
        return {
            "logits": logits,
            "relation_confidence_mean": conf.normalized_confidence.mean(dim=(2, 3)),
            "relation_confidence_max": conf.normalized_confidence.amax(dim=(2, 3)),
        }
```

This sketch deliberately does **not** change the trainer interface, the benchmark contract, or the parent i018 incidence topology. It is a targeted extension of the current repo structure, which is exactly what makes it a good research candidate. The current trainer already supports one-logit binary outputs, PR-AUC monitoring in binary modes, checkpoint artifacts, and reporting fields for prediction exports. fileciteturn10file0L3-L3 fileciteturn24file0L3-L3 fileciteturn21file0L3-L3

### Final recommendation

**Recommendation: implement.**

I recommend implementation because this is the highest-information, lowest-topology-risk extension of i018 that the repo currently supports. The parent architecture has already shown that exact chess relation geometry is load-bearing under a strong falsifier, and the current code still leaves a very specific degree of freedom unused: edge-wise importance within each relation family. i250 directly targets that gap while preserving the exact chess graph that made i018 interesting in the first place. fileciteturn9file0L3-L3 fileciteturn28file0L3-L3

I also recommend it because the proposal has a clean rollback path. With relation-wise mean-normalized confidence and zero initialization, the model starts effectively as i018. That means the first implementation is not a blind leap away from the parent; it is an “improve only if useful” extension. If the learned confidence fails, collapses to constants, or cannot beat flat confidence under repeated seeds, the ablations will say so clearly.

The main risk is not mathematical elegance. The main risk is that the confidence head becomes a fancy way to upweight noisy sharp-looking edges and worsens near-puzzle false positives. But that is exactly why the required evaluation should prioritize PR-AUC, matched-recall near-puzzle false positives, and the confidence-permutation falsifier rather than a single threshold F1 win. The repo’s current evaluation protocol is already designed for that kind of discipline. fileciteturn19file0L3-L3

So the practical verdict is: **implement i250 as a narrow, confidence-on-exact-edges extension of i018, benchmark it first on the current `simple_18` i018 path with PR-AUC checkpointing and 3-seed evaluation, and promote it only if it beats flat-confidence i018 on mean PR-AUC or matched-recall near-puzzle false positives without introducing slice regressions.**