# i253_i018_bt4_112_controlled_encoding.md

## Thesis

The controlled question is not whether `lc0_bt4_112` has more planes than `simple_18`; it is whether the **current repo’s** BT4-style encoding adds enough usable board-state signal to strengthen i018 **after exact mover-oriented chess relations are kept fixed**. That distinction matters because i018’s current strength is tied to explicit tactical geometry: the repo’s own knowledge base says i018 is the strongest accuracy-per-parameter trunk in the current line, and its falsifier reports a large drop when real chess relation masks are scrambled, which means the exact relation scaffold is load-bearing rather than decorative. The same knowledge base also says that most of the strongest historical runs came from architectures with strong chess-aware inductive bias, not from generic capacity alone. fileciteturn35file0L3-L3 fileciteturn28file0L3-L3 fileciteturn25file0L3-L3

The proposal below therefore keeps the current i018 sheaf trunk, keeps side-to-move canonicalization, keeps the exact 12 relation families, and tests BT4 in three increasingly permissive ways: **exact-relations-only**, **exact support plus learned confidence**, and **exact support plus bounded augmentation**. Every comparison is rerun on the same clean tagged split, with repeated seeds, PR-AUC checkpointing, and the same training budget, so that any lift can be attributed to encoding content instead of split drift, budget drift, or architecture drift. fileciteturn9file0L3-L3 fileciteturn16file0L3-L3 fileciteturn17file0L3-L3 fileciteturn36file0L3-L3

## What BT4-112 can add

### What BT4-112 currently contains in this repo

In `chess-nn-playground`, `lc0_bt4_112` is defined as an 8-step, 13-planes-per-step layout plus 8 auxiliary planes, but the implementation is explicit that the dataset currently contains **a single FEN per record**. That means history slot `0` is populated, while older history slots and repetition planes remain zero until the exporter provides true move history. The BT4 aux planes that currently carry signal are localized castling-rook markers, en-passant file, a scaled rule-50 halfmove clock, and an all-ones plane; several other aux slots are reserved zeros. In other words, the current repo is **not yet testing full historical BT4**. It is testing a mover-relative current-board encoding with a few richer auxiliary planes. fileciteturn8file0L3-L3

That sharply limits what “BT4 benefit” can mean here. Relative to `simple_18`, the current BT4 exporter is not adding seven real past positions, repetition context, or search-derived data. What it is adding is a more LC0-like mover-relative plane layout, localized castling-right representation, en-passant file encoding, the rule-50 clock, and an always-on bias plane. Any genuine gain should therefore be interpreted as “**richer current-position coding helps i018 a bit**,” not “**more planes are automatically better**.” fileciteturn8file0L3-L3

### What is genuinely new beyond `simple_18`

`simple_18` already carries the current position in 12 absolute piece planes, a side-to-move plane, four global castling-right planes, and an exact en-passant square plane. By contrast, the current BT4 exporter writes the first 12 planes in mover-relative `our_*` / `their_*` form and uses auxiliary channels that are more localized or scalar-like. That means the plausible incremental information for i018 is mostly:

- less work for the adapter to recover mover-relative piece ownership,
- a slightly richer castling representation,
- a distinct rule-50 signal,
- a more standardized 112-plane layout for raw-feature mixing. fileciteturn8file0L3-L3

The current i018 implementation already has hooks for `lc0_static_112` and `lc0_bt4_112`, and in the BT4 branch it extracts the first 12 planes directly as exact mover-relative pieces instead of falling back to a learned probe. That is encouraging, because it means a controlled BT4 test does **not** require abandoning exact relation masks. The repo already has the right conceptual seam; it just does not yet have a clean controlled experiment around that seam. fileciteturn5file0L3-L3 fileciteturn31file0L3-L3

## Controlled architecture

### Architecture variants

The controlled proposal should inherit the **current base i018 trunk shape**: `channels: 64`, `hidden_dim: 96`, `depth: 2`, `stalk_dim: 8`, triad pool on, and the same readout family. That is the registered i018 base configuration today, and it is the right starting point because the goal is to isolate encoding effects before changing scale. fileciteturn21file0L3-L3

The three variants are:

**Exact-relations-only.**  
This is the hard control. Relation masks are built exactly from canonical piece planes and occupancy, exactly as in current i018, and the richer encoding can only help through the node/raw feature path. Formally, the sheaf sees `W_r = M_r`, where `M_r` is the exact relation mask for relation `r`. If BT4 wins here, then the benefit is real and clean: the encoding helped without any learned edge invention. fileciteturn32file0L3-L3 fileciteturn33file0L3-L3

**Exact support plus learned confidence.**  
This variant keeps the exact support fixed and only lets raw planes learn a bounded confidence on that support:
`W_r = M_r ⊙ sigmoid(C_r)`.
No new edges are created. This is the safest learned-relation test because it cannot replace chess geometry with arbitrary dense pair scores. It only asks whether BT4 planes help calibrate the importance of already-correct edges. This middle ground is also consistent with the sheaf literature, which treats relation-specific maps as part of the model’s expressivity, and with graph-learning work showing that adaptive edge features or edge attention can help without discarding the underlying graph structure. fileciteturn33file0L3-L3 citeturn4academia0turn7academia0turn9academia0turn9academia1

**Exact support plus bounded augmentation.**  
This variant keeps exact support and confidence, then adds a small residual on a **fixed, relation-specific geometric superset**:
`W_r = clamp(M_r ⊙ sigmoid(C_r) + λ_aug · T_r ⊙ sigmoid(A_r), 0, 1)`,
with `λ_aug ≤ 0.25`. Here `T_r` is a hand-defined template superset for relation `r`, not a free all-pairs graph. This is the most permissive variant, but it still forbids the “just learn arbitrary edges from 112 planes” failure mode. In practice, it asks whether BT4 aux planes help i018 model **uncertain tactical shadow structure** around the exact relation graph. That is a credible test of encoding benefit; a dense unconstrained augmentation would not be. citeturn7academia1turn9academia3

### Dataflow and canonicalization details

The most important control is to make the **raw-input pathway literally identical across encodings**. The cleanest way to do that is to standardize the sheaf trunk on **112 raw channels for both encodings**:

- for `simple_18`, run the existing mover-canonical adapter first, then place its 18 canonical raw planes into a fixed 112-channel padded tensor, leaving the remaining channels zero;
- for `lc0_bt4_112`, use the native 112 channels as-is. fileciteturn31file0L3-L3 fileciteturn8file0L3-L3

That gives one architecture, one input projection size, and one parameter count per variant. The **piece-state path remains exact and encoding-aware**, so relation construction still comes from exact chess content, not from the padded/raw control trick. The raw branch is where the richer encoding is allowed to help.

A clean forward pass is:

```text
input board tensor
  -> BoardStateAdapterControlled
       -> square_raw_112
       -> piece_state_exact
       -> occupancy
  -> TacticalIncidenceBuilder
       -> exact masks M
       -> optional templates T
  -> SquareTokenEncoder(square_raw_112, piece_state_exact)
       -> h0
  -> optional RelationConfidenceHead(square_raw_112)
       -> C or (C, A)
  -> sheaf weights:
       exact:      M
       confidence: M ⊙ sigmoid(C)
       hybrid:     clamp(M ⊙ sigmoid(C) + λ_aug · T ⊙ sigmoid(A), 0, 1)
  -> same SheafDiffusionBlocks
  -> same TriadDefectPool
  -> same readout head
```

This keeps the current i018 architectural thesis intact: the board-only adapter feeds exact tactical incidence; the sheaf block still performs relation-specific diffusion over chess-shaped structure; and the richer encoding is tested as a controlled additive benefit rather than as a replacement mechanism. fileciteturn5file0L3-L3 fileciteturn31file0L3-L3 fileciteturn32file0L3-L3 fileciteturn33file0L3-L3

One additional control is important: the new learned-relation modules should **not** consume an original white-vs-black identity bit that exists in `simple_18` but is absent from BT4’s mover-relative layout. After canonicalization, learned relation heads should operate in the **mover frame only**. That preserves the intended premise of i018’s side-to-move canonicalization rather than letting one encoding cheat through leftover absolute-color identity. This is an implementation choice derived directly from how the current adapter treats `simple_18` versus BT4. fileciteturn31file0L3-L3

### Relation construction details

The exact relation inventory should stay exactly at the current i018 set of 12 typed masks: attacker-target, defender-target, king-zone pressure, rook/bishop/queen visible rays, knight attacks, oriented pawn attacks, and pin candidates. Those masks are already built from exact piece planes using precomputed knight, king, king-zone, pawn, rook-ray, bishop-ray, and between-square blocker masks, with occupancy gating for visibility and aligned king-blocker-slider logic for pin candidates. That is the part of i018 the repo has already shown to be load-bearing. fileciteturn5file0L3-L3 fileciteturn32file0L3-L3

For the **confidence** variant, learned scores must be support-only. A small low-rank head is enough:

- project each square’s raw 112-vector to a 16-d relation feature,
- pool a global board context from those features,
- produce relation-specific source and target low-rank codes,
- form a pair logit by low-rank source-target interaction plus global relation bias,
- apply `sigmoid` only where `M_r = 1`.

That is deliberately weaker than a free learned graph. The point is not to let BT4 rediscover chess; the point is to let BT4 decide which exact chess edges are more or less trustworthy in context. Theoretical support for this choice comes from both sheaf models and edge-feature-aware message passing: learned edge modulation can add expressive power, but the domain graph still matters. citeturn7academia0turn7academia1turn9academia0turn9academia3

For the **hybrid** variant, the augmentation template `T_r` should be narrow and relation-specific:

- for rook, bishop, and queen relations, use the full geometric ray template before blocker gating;
- for attacker/defender relations, allow augmentation only on slider geometries where occupancy ambiguity is the contested variable;
- for knight, king, and pawn relations, do **not** permit non-geometric augmentation beyond legal shape;
- for pin candidates, use aligned king-blocker-slider templates before clear-ray gating. fileciteturn32file0L3-L3

This is the key discipline that makes the study interpretable. If the hybrid beats the confidence-only variant, the likely conclusion is not “BT4 invented chess relations,” but “BT4 helped i018 add bounded residual structure around exact chess geometry.” If a dense unrestricted 64×64 augmentation were allowed instead, that conclusion would no longer be defensible.

### Parameter budgets

Current base i018 is reported at **91,363 parameters** in repo runs, using `input_channels: 18`, `channels: 64`, `hidden_dim: 96`, and `depth: 2`. Moving the controlled study to a shared 112-channel raw branch changes only the size of the raw input projection and adds **3,008** parameters, producing a shared **94,371-parameter** exact-only base for both encodings. fileciteturn21file0L3-L3 fileciteturn27file0L3-L3 fileciteturn31file0L3-L3 fileciteturn32file0L3-L3 fileciteturn33file0L3-L3

Under the proposed sidecar sizes `relation_hidden=16` and `relation_rank=8`, the controlled parameter budgets are:

| Variant | Total params | Delta vs exact-only | Interpretation |
|---|---:|---:|---|
| Exact-relations-only | 94,371 | — | identical trunk and identical raw width across encodings |
| Exact support plus learned confidence | 99,487 | +5,116 | one low-rank confidence head over raw 112 planes |
| Exact support plus bounded augmentation | 102,763 | +8,392 | shared pre-proj plus separate confidence and augmentation heads |

These are intentionally tight budgets. The largest variant is only about 8.9% larger than the exact-only control, so the comparison still centers on encoding content and relation design rather than on brute-force capacity.

## Experiment plan

### Exact experiment matrix

The study should be run as a **36-run base package** on the canonical tagged split, with the same loss family, optimizer family, scheduler policy, early-stopping policy, and artifact pipeline across every cell. The repo’s reliability protocol explicitly requires matched splits, matched conventions, repeated seeds for promotion-grade claims, and validation-only model selection; the trainer now supports PR-AUC as the binary monitor and should be set explicitly to avoid ambiguity. For alignment with the repo’s repeated-seed paper-grade trunk reruns, the safest budget is the same regime used in `run_paper_grade_top3.sh`: seeds `42,43,44`, paper-grade convergence budget, and explicit PR-AUC monitoring. fileciteturn9file0L3-L3 fileciteturn16file0L3-L3 fileciteturn17file0L3-L3 fileciteturn36file0L3-L3

The matrix is:

| Tranche | Cells | Seeds | Runs |
|---|---|---:|---:|
| Primary comparison | `exact`, `confidence`, `hybrid` × `simple18_padded112`, `lc0_bt4_112` | 3 | 18 |
| Exact-support falsifier | `exact_scramble` × both encodings | 3 | 6 |
| Hybrid falsifier | `hybrid_scramble_exact_support` × both encodings | 3 | 6 |
| Augmentation-only falsifier | `hybrid_augmentation_only` × both encodings | 3 | 6 |
| **Total** |  |  | **36** |

A concrete shared training header should look like this:

```yaml
seed: 42
deterministic: true
mode: puzzle_binary
device: nvidia

data:
  train_path: data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet
  val_path: data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet
  test_path: data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet
  cache_features: false

training:
  monitor: pr_auc
  epochs: 30
  min_epochs: 15
  min_active_epochs: 15
  early_stopping_patience: 8
  batch_size: 192
  class_weighting: balanced
  mixed_precision: true
  allow_tf32: true
  matmul_precision: high
  lr_scheduler:
    name: reduce_on_plateau
    factor: 0.5
    patience: 2
    min_lr: 1.0e-05
```

That budget is stricter than the minimum reliable default and keeps the study in the same research-quality regime as the repo’s repeated-seed top-trunk comparisons. Using a shared `batch_size: 192` is conservative, but it removes memory-driven optimization drift between exact and hybrid rows. fileciteturn9file0L3-L3 fileciteturn36file0L3-L3

The primary decision rule should use the repo’s own promotion logic: BT4 earns a real encoding win only if, against the **matched** simple18 row of the same variant, it improves mean PR-AUC by at least about `+0.003` or reduces near-puzzle false positives at matched recall by at least `1%`, without causing obvious regressions on hard, equal, endgame, mate-in-1, promotion, or underpromotion slices. That is much more defensible than comparing against old 12-epoch scout numbers. fileciteturn9file0L3-L3

### Ablations and falsifiers

The repo already has the right falsifier spirit for i018: scramble relation masks while preserving out-degree, and treat a tiny drop as a rejection of the thesis. The controlled BT4 study should preserve that exact logic in both encodings. In the existing falsifier launcher, a drop of at least `0.02` PR-AUC is treated as strong support for the thesis, while a gap below `0.01` is treated as rejection. Those thresholds should be reused here. fileciteturn28file0L3-L3 fileciteturn29file0L3-L3

Beyond the required relation scramble, four additional ablations are worth running because they directly improve interpretability:

- **Channel-count control.** Run `simple18_native18` versus `simple18_padded112` on the exact-only row. If they match, the padding trick is inert and the literal same-architecture comparison is valid.
- **Dead-plane control.** Run BT4 with all currently dead history/repetition/reserved planes hard-masked to zero at the adapter output. Because the exporter already makes those planes zero, this should be identical; if it is not, something else in the implementation is leaking.
- **File-sign control.** Remove signed file coordinates from the encoder for one seed per encoding. This checks whether convention drift between the current simple18 canonicalizer and the already-canonical BT4 exporter is affecting results through coordinate features rather than through chess content.
- **Aux-only control.** Compare full BT4 against a retained subset that keeps the current-board piece planes plus only the nontrivial aux planes, to learn whether any gain comes from BT4’s auxiliary symbols rather than from the nominal 112-channel format itself. fileciteturn8file0L3-L3 fileciteturn31file0L3-L3

The most important new falsifier is the **hybrid augmentation-only** row. If a learned BT4 augmentation graph, with exact masks turned off, comes close to the intact hybrid, that would meaningfully weaken the current belief that exact chess relations are essential. If it collapses, the repo’s present thesis survives: BT4 may help, but it helps around exact geometry rather than instead of it.

## Expected outcomes

### Expected outcomes and belief updates

The strongest prior from the repo is that exact chess structure matters much more than generic capacity, and that the current BT4 reference family is robust but not obviously better than the strongest chess-aware trunks. At the same time, the current exporter’s BT4 representation carries only limited extra signal because true history is not yet populated. Putting those together, the most likely outcome is a **small** BT4 effect, not a large one: exact-only BT4 may slightly beat or tie exact-only simple18, confidence may provide a modest extra lift if the auxiliary planes help calibrate edge strength, and hybrid augmentation is more likely to be neutral or slightly noisy than to deliver a dramatic jump. fileciteturn35file0L3-L3 fileciteturn8file0L3-L3 fileciteturn25file0L3-L3

A sensible belief table is:

| Observed result | Belief update |
|---|---|
| `BT4 exact > simple18 exact` by a clean repeated-seed margin | richer current-position coding helps i018 even when exact relations are fixed; prioritize exact BT4 for scaling |
| `BT4 confidence > BT4 exact`, while simple18 confidence does not help | BT4’s extra aux planes are useful mainly as relation-strength modulator, not as node-only features |
| `Hybrid` clearly beats `confidence` in BT4 but not in simple18 | the richer encoding carries enough context to justify bounded residual structure around exact masks |
| No BT4 row beats its matched simple18 row | the current single-FEN BT4 exporter is not richer enough in practice; do not generalize that to “BT4 never helps” |
| Relation scramble remains catastrophic in both encodings | exact chess geometry is still the load-bearing mechanism |
| Augmentation-only stays far below intact exact-support models | learned BT4 relations are additive, not substitutive |

The most important negative-result interpretation is this one: if BT4 fails here, the conclusion should be **about the current exporter**, not about the abstract idea of richer mover-relative encodings. Because history slots beyond step 0 are presently zero, a null result would mean “current-FEN BT4 did not beat simple18 in i018,” not “filling true BT4 history would never matter.” fileciteturn8file0L3-L3

## Implementation notes for chess-nn-playground

The cleanest repo path is to add a new idea folder, for example `ideas/registry/i253_i018_bt4_112_controlled_encoding/`, while reusing as much of the current i018 code as possible. The current implementation already has the right seams: `BoardStateAdapter`, `TacticalIncidenceBuilder`, `SquareTokenEncoder`, `SheafDiffusionBlock`, `TriadDefectPool`, and the build function all exist and already acknowledge `lc0_bt4_112`. The new work is mainly controlled wiring, not a full rewrite. fileciteturn22file0L3-L3 fileciteturn31file0L3-L3 fileciteturn32file0L3-L3 fileciteturn33file0L3-L3

The minimal code changes are:

```yaml
model:
  name: oriented_tactical_sheaf_controlled_encoding
  input_channels: 112
  channels: 64
  hidden_dim: 96
  depth: 2
  stalk_dim: 8
  use_triads: true

  relation_mode: exact          # exact | confidence | hybrid
  pad_simple18_to_112: true
  relation_hidden: 16
  relation_rank: 8
  augmentation_lambda: 0.25
  augmentation_support: slider_superset

  scramble_exact_relations: false
  augmentation_only: false
```

That implies three focused implementation tasks:

**Adapter task.**  
Extend `BoardStateAdapter` with a controlled mode that returns `square_raw_112` for both encodings. For `simple_18`, canonicalize exactly as today, then pad to 112. For BT4, keep the existing direct extraction of the first 12 mover-relative piece planes. This preserves exact piece-state construction while making the raw branch architecture-identical. fileciteturn31file0L3-L3

**Incidence task.**  
Extend `TacticalIncidenceBuilder` to optionally return fixed augmentation templates `T_r` alongside exact masks `M_r`. The exact masks should remain the current implementation. The templates should be relation-specific and narrow, especially for sliders and pin candidates. fileciteturn32file0L3-L3

**Relation-head task.**  
Add a small `RelationConfidenceHead` that consumes `square_raw_112` and emits either `C_r` only or `(C_r, A_r)` for the hybrid mode. Keep it low-rank and bounded. Do not let it touch metadata, labels, or any reporting-only fields; the repo’s reliability protocol is explicit that those must not become model inputs. fileciteturn9file0L3-L3 fileciteturn22file0L3-L3

On the training side, the repo’s trainer already supports the key requirement: for binary modes, checkpoint selection can now monitor `pr_auc`, and the metric is written into run metadata. Even so, the controlled configs should set `training.monitor: pr_auc` explicitly, because explicitness matters when the whole purpose of the study is to isolate one causal factor. fileciteturn16file0L3-L3 fileciteturn17file0L3-L3

For execution, there is one practical option worth calling out. If wall-clock becomes a bottleneck, the repo already contains `i249_oriented_tactical_sheaf_fast`, which is documented as a **pure execution optimization** of i018 with the same math, parameters, and numerics. It is acceptable to run the entire controlled study on the i249 execution path instead of vanilla i018, but only if **every** cell uses it. Mixing i018 and i249 within the same comparison would reintroduce an avoidable confound. fileciteturn23file0L3-L3 fileciteturn24file0L3-L3

The final recommendation is therefore straightforward: implement `i253` as a **112-channel controlled i018 family** with exact chess relations preserved, a padded simple18 control, an exact-support confidence head, a bounded hybrid augmentation head, PR-AUC checkpointing, and the 36-run base matrix above. That is the smallest design that can actually answer the question the repo cares about.