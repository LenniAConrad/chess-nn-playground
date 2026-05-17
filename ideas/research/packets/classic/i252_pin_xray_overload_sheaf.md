# i252_pin_xray_overload_sheaf.md

## Thesis and current gap

**Thesis.** Upgrade i018 by keeping its board-only, side-to-move-oriented sheaf architecture intact, but replacing its coarse tactical dependency graph with a compact bank of tensorized single-screen and defender-dependency relations so the model can distinguish direct attacks from latent x-rays, skewers, discovered attacks, pinned defenses, and overloaded defenses. fileciteturn8file0L3-L3 fileciteturn18file0L3-L3

i018 already has the right architectural spine for this job. The repo’s `Oriented Tactical Sheaf Laplacian` canonicalizes to the mover’s perspective, builds a dense tactical incidence tensor from the board alone, and feeds typed relations into a learned sheaf diffusion stack; its current `RELATION_NAMES` are the 12 planes for us/them attacks, us/them defenses, empty king-ring pressure, visible bishop/rook/queen rays, knight geometry, oriented pawn geometry, and a king-ray pin candidate. The readout then pools per-relation densities, per-relation energies, gates, triad features, and a small board-statistics vector, including `pin_pressure`. fileciteturn8file0L3-L3 fileciteturn7file0L3-L3 fileciteturn17file0L3-L3

The right execution substrate is not the original eager i018 block but the repo’s i249 fast variant. i249 keeps i018’s math and parameters unchanged, but replaces the per-relation Python diffusion loop with batched `einsum` projections and chunked coboundary evaluation, specifically to reduce launch overhead and control the peak `(B, chunk, 64, 64, stalk)` intermediate on 8 GB GPUs. That means the fastest way to land this upgrade is to change the graph construction and relation indexing, while inheriting i249’s vectorized diffusion path. fileciteturn14file0L3-L3 fileciteturn15file0L3-L3

What i018 misses is now visible in the repo’s own slice reports. In the current scout-style per-class benchmark, `idea_i018_oriented_tactical_sheaf_laplacian` posts an overall PR AUC of `0.861`, but on the explicitly logged tactic-motif slices it sits at `0.837` on `pin`, `0.830` on `skewer`, `0.824` on `overload`, and `0.805` on `discovered_attack`; the slice leaders are `0.859`, `0.865`, `0.853`, and `0.852`, respectively. So the graph is already good enough to be broadly competitive, but its weakest motif slices are exactly the ones named in this request. fileciteturn25file0L3-L3

That nuance matters because the goal should be targeted sharpening, not unrelated invention. In the repo’s matched-recall report at recall `0.80`, the seed-42 i018 run is already third by overall near-puzzle false-positive rate, which suggests the architecture has useful tactical signal already; the remaining headroom is more likely in representing *conditional* tactical dependencies than in discarding the sheaf design. This is also consistent with sheaf-learning literature: sheaf neural networks are precisely a way to extend graph diffusion to asymmetric, relation-specific local maps, and neural sheaf diffusion is especially motivated when relations are heterophilic rather than simple same-type neighborhoods. fileciteturn27file0L3-L3 citeturn3academia2turn3academia0turn3academia1

## Relation inventory and equations

Standard chess usage fits the proposed relation family well: an x-ray is line control through an intervening piece, a skewer is effectively an inverse pin along a line, a discovered attack reveals a line attack by moving a screening piece, and overloading gives one defender more than one duty it cannot comfortably satisfy. The upgrade below encodes those semantics as bounded pairwise planes, not as an open-ended hypergraph. citeturn5search3turn3search5turn5search0turn5search2

The proposal keeps the current 12 i018 planes and adds 10 new ones, for a total relation count of `R = 22`. That stays compact enough for the existing sheaf readout, whose dimensionality already scales with `len(RELATION_NAMES)`. The crucial design choice is that every higher-order tactical pattern is *collapsed* into either a pairwise square-to-square plane or a per-square soft mass before diffusion, so the rest of the network remains unchanged in spirit. fileciteturn17file0L3-L3 fileciteturn15file0L3-L3

| Added plane | Source → target semantics | Intended meaning |
|---|---|---|
| `us_xray_them_piece` | slider → rear enemy piece | exactly one occupied screen lies on the line; a latent line attack exists behind it |
| `them_xray_us_piece` | slider → rear enemy piece | mirror relation |
| `us_skewer_them_piece` | slider → rear enemy piece | front enemy screen is more valuable than the rear enemy piece |
| `them_skewer_us_piece` | slider → rear enemy piece | mirror relation |
| `us_discovered_attack_candidate` | own screening piece → rear enemy piece | moving the screen would reveal an own slider attack |
| `them_discovered_attack_candidate` | own screening piece → rear enemy piece | mirror relation |
| `us_attacks_them_piece_with_pinned_defender` | attacker → enemy target | target is defended, but some defender is absolutely pinned to its king |
| `them_attacks_us_piece_with_pinned_defender` | attacker → enemy target | mirror relation |
| `us_attacks_them_overloaded_piece` | attacker → enemy target | target depends on a defender whose second meaningful defensive duty is nontrivial |
| `them_attacks_us_overloaded_piece` | attacker → enemy target | mirror relation |

Let the board have `N = 64` squares. Let `o ∈ {0,1}^N` be occupancy, `e = 1 - o`, `u_i` and `t_i` be mover-oriented us/them piece indicators on square `i`, and let `μ_i` be a fixed piece-order score with values `{pawn: 1, knight: 3, bishop: 3, rook: 5, queen: 9, king: 100}` while `ν_i = min(μ_i, 9) / 9 ∈ [0,1]` is the normalized non-search tactical importance used for overload weighting. Let `B_{ijq}` be the existing between-square tensor already used by i018, and let `V^{rook}` and `V^{bish}` be the current visible-ray masks. Then the existing visible-ray equations remain unchanged: fileciteturn16file0L3-L3

```text
clear_{ij} = 1[∑_q B_{ijq} o_q = 0]

V^{rook}_{ij} = R^{rook}_{ij} · clear_{ij}
V^{bish}_{ij} = R^{bish}_{ij} · clear_{ij}
```

Absolute pins should also be made explicit rather than left as a coarse symmetric candidate plane. Reusing i018’s pin-template idea, define a precomputed pin bank `T_pin` whose item `m` stores `(slider s_m, blocker d_m, king k_m, line ℓ_m, clear-mask Q^pin_m)`. Then the *side-specific* absolute pin mask is:

```text
Γ^pin_m = 1[∑_q Q^pin_{mq} o_q = 0]

P^{us}_{sd} =
clip( ∑_{m∈T_pin}
      1[s=s_m] 1[d=d_m]
      Γ^pin_m
      S^{us}_{s,ℓ_m}
      t_d
      K^{them}_{k_m},
      0, 1 )
```

where `S^{us}_{s,rook}` means “our rook-or-queen on `s`” and `S^{us}_{s,bish}` means “our bishop-or-queen on `s`”. The mirror mask `P^{them}` is defined analogously. This keeps the operator class identical to i018’s sheaf construction: only the edge weights change, not the `δᵀδ` sheaf-Laplacian logic. fileciteturn16file0L3-L3 fileciteturn18file0L3-L3 citeturn3academia1turn8academia3

The key new bank is a **single-screen ray template bank** `T₁`. Each template `m ∈ T₁` stores ordered `(source s_m, screen c_m, rear r_m, line ℓ_m, clear-mask Q^1_m)`, where `Q^1_m` is the set of squares strictly between `s_m` and `r_m`, excluding the designated screen square `c_m`. Runtime evaluation is then fully batched:

```text
Γ_m = 1[∑_q Q^1_{mq} o_q = 0]
```

and the new relation planes are:

```text
X^{us}_{sr} =
clip( ∑_{m∈T₁}
      1[s=s_m] 1[r=r_m]
      Γ_m
      S^{us}_{s,ℓ_m}
      o_{c_m}
      t_r,
      0, 1 )

K^{us}_{sr} =
clip( ∑_{m∈T₁}
      1[s=s_m] 1[r=r_m]
      Γ_m
      S^{us}_{s,ℓ_m}
      t_{c_m}
      t_r
      1[μ_{c_m} > μ_r],
      0, 1 )

D^{us}_{cr} =
clip( ∑_{m∈T₁}
      1[c=c_m] 1[r=r_m]
      Γ_m
      S^{us}_{s_m,ℓ_m}
      U^{us,nonking}_c
      t_r
      (1 - pin^{us}_c),
      0, 1 )
```

`X` is the x-ray plane, `K` is the skewer plane, and `D` is the discovered-attack-candidate plane. `U^{us,nonking}` excludes the king as a “screen to move,” and `(1 - pin^{us}_c)` suppresses discovered-attack candidates where the screening piece is itself absolutely pinned and therefore not freely movable. The mirror relations `X^{them}`, `K^{them}`, and `D^{them}` are defined identically. Because discovered attack is encoded as `screen → rear target`, it contributes a new dependency edge instead of merely duplicating x-ray’s `slider → rear target` geometry. fileciteturn16file0L3-L3

Pinned defenders are best represented as a **target exposure mass**, then turned back into an attacker-to-target relation. Let `A^{us}` and `A^{them}` be the current mover-oriented attack tensors, and let `Def^{them}` be same-side defense from nonking defenders only. Then:

```text
pin^{them}_d = clip(∑_s P^{us}_{sd}, 0, 1)

ρ^{us,pdef}_r =
clip( ∑_d Def^{them}_{dr} · pin^{them}_d, 0, 1 )

M^{us,pdef}_{sr} = A^{us}_{sr} · ρ^{us,pdef}_r
```

The mirror plane `M^{them,pdef}` is identical with sides swapped. This is the minimal pairwise object that says, “yes, the target is defended geometrically, but some of that defense is tactically fake because the defender is pinned.” That is exactly the tactical dependency that the current `king_ray_pin_candidate` does *not* feed into target vulnerability. fileciteturn16file0L3-L3

Overload can be made similarly compact by using second-duty strength rather than enumerating defender-target-target hyperedges. Let target criticality under our pressure be

```text
crit^{us}_r = ν_r · 1[(∑_s A^{us}_{sr}) > 0]
g^{them}_{dr} = Def^{them}_{dr} · crit^{us}_r
ω^{them}_d = second_largest_r g^{them}_{dr}
```

Then the overload exposure of enemy target `r` is

```text
ρ^{us,ovl}_r =
clip( ∑_d Def^{them}_{dr} · ω^{them}_d, 0, 1 )

M^{us,ovl}_{sr} = A^{us}_{sr} · ρ^{us,ovl}_r
```

This definition is deliberate. It says a defender is overloaded to the extent that it has a *second* meaningful defensive assignment among currently pressured targets. The score is continuous, board-only, and tensor-friendly; it avoids hard search-like “is this tactic winning?” labels while still modeling the central overload motif. fileciteturn13file0L3-L3

All new planes are attack-like dependencies, so their sheaf sign should match the current attack-family convention. In practice, keep `σ_r = +1` only for same-side defense planes (`us_defends_us_piece`, `them_defends_them_piece`) and assign `σ_r = -1` to x-rays, skewers, discovered attacks, pinned-defender attacks, and overloaded-defender attacks, consistent with the current sign policy that distinguishes attack from defense semantics inside the sheaf block. fileciteturn17file0L3-L3

## Dataflow and PyTorch sketch

Architecturally, this should be a **graph upgrade, not a trunk rewrite**. Keep `BoardStateAdapter`, `SquareTokenEncoder`, the basic readout pattern, and the sheaf diffusion idea. Swap only the incidence builder and relation-index handling, then route the resulting 22-plane tensor through the fast i249 diffusion block. That stays aligned with both the repo’s original i018 thesis and the later i249 optimization work. It also respects the repo’s explicit board-only contract and its choice to use side-to-move canonicalization rather than full board `D4` equivariance, which is consistent with the broader warning from equivariance literature that over-enforcing symmetry can hurt when the real domain symmetry is only partial. fileciteturn8file0L3-L3 fileciteturn9file0L3-L3 fileciteturn14file0L3-L3 citeturn8academia0turn8academia2

```text
(Board tensor)
      │
      ▼
BoardStateAdapter
      │
      ├── square_raw
      ├── piece_state
      └── occupancy
      │
      ▼
TacticalIncidenceBuilderV2
      │
      ├── base attacks / defenses / visible rays
      ├── absolute pin bank
      ├── single-screen bank
      ├── x-ray / skewer / discovered planes
      ├── pinned-defender exposure
      └── overload exposure
      │
      ▼
relation_masks ∈ R^(B,22,64,64)
      │
      ▼
FastSheafDiffusionBlock × depth
      │
      ├── energy_mean / energy_max
      ├── gate_mean
      └── updated node states
      │
      ▼
existing pooled readout
  + dependency diagnostics
      │
      ▼
puzzle_binary logit
```

A repo-specific implementation detail matters here: the current code hard-codes relation index ranges in diagnostics, for example `energy_mean[:, 6:9]` for ray energy and `relation_density[:, 11]` for `pin_pressure`. Once relation order changes, those index literals become fragile. So the first mechanical refactor should be a `relation_index: dict[str, int]` and named family lookups for diagnostics; after that, the head dimension will already grow correctly because the readout is built from `len(RELATION_NAMES)`. fileciteturn17file0L3-L3

A minimal PyTorch sketch looks like this:

```python
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

NEW_RELATIONS = (
    "us_xray_them_piece",
    "them_xray_us_piece",
    "us_skewer_them_piece",
    "them_skewer_us_piece",
    "us_discovered_attack_candidate",
    "them_discovered_attack_candidate",
    "us_attacks_them_piece_with_pinned_defender",
    "them_attacks_us_piece_with_pinned_defender",
    "us_attacks_them_overloaded_piece",
    "them_attacks_us_piece_with_pinned_defender",
)

def _scatter_pair(
    batch_size: int,
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
    values: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    flat = torch.zeros(batch_size, 64 * 64, device=device, dtype=dtype)
    pair_index = (src_idx * 64 + dst_idx).view(1, -1).expand(batch_size, -1)
    flat.scatter_add_(1, pair_index, values)
    return flat.view(batch_size, 64, 64).clamp_(0.0, 1.0)

class TacticalIncidenceBuilderV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Reuse base i018 geometry buffers, then add two banks:
        #   pin_bank:     slider / blocker / king / line / clear_without_blocker
        #   screen_bank:  source / screen / rear / line / clear_without_screen
        base = _make_geometry_masks()
        for name, value in base.items():
            self.register_buffer(name, value, persistent=False)
        bank = _make_single_screen_template_bank(base["between"], base["rook_ray"], base["bishop_ray"])
        for name, value in bank.items():
            self.register_buffer(name, value, persistent=False)

    def _single_screen_clear(self, occupancy: torch.Tensor) -> torch.Tensor:
        # (B, M) where M is the bounded single-screen template count.
        return (1.0 - occupancy @ self.screen_clear.t()).clamp_(0.0, 1.0)

    def _overload_mass(
        self,
        attack: torch.Tensor,          # (B, 64, 64)
        defense: torch.Tensor,         # (B, 64, 64), same-side defense by non-king defenders
        target_value: torch.Tensor,    # (B, 64), normalized to [0,1]
    ) -> torch.Tensor:
        critical = target_value * (attack.sum(dim=1) > 0).to(target_value.dtype)
        contrib = defense * critical.unsqueeze(1)          # defender x target
        top2 = contrib.topk(k=2, dim=-1).values
        second = top2[..., 1]                              # second-biggest duty
        return second.clamp_(0.0, 1.0)                     # defender-level overload score

    def forward(self, piece_state: torch.Tensor, occupancy: torch.Tensor):
        # 1) Build base i018 attacks / visible rays / absolute pin masks.
        # 2) Evaluate single-screen bank in batch:
        #       xray:    source -> rear
        #       skewer:  source -> rear with value(front) > value(rear)
        #       disc:    screen -> rear, masked by own non-king and not self-pinned
        # 3) Compute pinned-defender and overload target masses.
        # 4) Turn masses back into attack-conditioned pairwise planes.
        # 5) Stack base 12 + new 10 = 22 relation planes.
        raise NotImplementedError
```

Two design choices are worth keeping explicit. First, this builder should live under a new idea/module boundary, not as a silent behavior change to i018, because it changes graph semantics and needs its own falsifier config. Second, the diffusion block itself does **not** need to become mathematically fancier: i249’s chunked batched sheaf block is already the compact typed-relation consumer this upgrade needs. The parameter increase from 10 extra relations is tiny compared with the graph-value increase; the runtime and memory story is about relation planes, not restriction-map size. fileciteturn14file0L3-L3 fileciteturn15file0L3-L3

## Expected effect and complexity

The strongest case for this upgrade is not “it will surely win overall,” but “it points directly at the repo’s weakest tactical slices for i018.” The benchmark already measures `pin`, `skewer`, `overload`, and `discovered_attack` as explicit motif slices, and i018 lags the slice leaders on all four—most sharply on `discovered_attack`, where it is `0.805` versus a leader at `0.852`. Because these gaps line up exactly with the missing relation family, this is the rare architecture proposal whose target is not generic capacity but a specific audited blind spot. fileciteturn25file0L3-L3

The expected benefit on `puzzle_binary` should therefore be framed in two layers. On the *main* metric, the right goal is non-regression or a modest lift, because i018 is already broadly competitive. On the *reliability* metrics that the repo explicitly elevates—matched-recall false positives, matched-recall near-puzzle false positives at recall `0.80` and `0.85`, and worst-slice behavior—the new relations should help by discounting nominal defenses that are structurally fake. A near-puzzle often looks close to a real tactic under raw attack counts; the missing signal is whether a defense is x-rayed, pinned, overloaded, or only temporarily screening a discovered line. fileciteturn13file0L3-L3 fileciteturn27file0L3-L3

The runtime story is favorable if the builder stays tensorized. For an 8×8 board, the bounded ordered one-screen template bank has only `2,576` `(source, screen, rear)` templates. So the new relation computation is a small constant-size batched gather/matmul/scatter problem: roughly `O(B · N · M₁)` for template-bank clearing plus `O(B · N²)` for attack/defense and overload masses, with `N = 64` and `M₁ = 2,576`. There is no runtime Python scan over squares or rays, no search, and no stored 3-cell hypergraph; each higher-order motif is reduced immediately to pairwise planes or square masses.

Memory also stays tame. The main dense relation tensor grows from `12` to `22` planes. At batch size `256`, that is about `88 MB` in FP32 or `44 MB` in FP16 for `B × R × 64 × 64`, which is quite manageable relative to the rest of the model. The fast i249 diffusion block’s peak memory is driven by chunk size, not total relation count, because the expensive intermediate is chunk-local; i249 already uses chunking specifically to keep that intermediate within the envelope of an 8 GB GPU. Parameter growth is trivial by comparison: 10 more relations add only `10 × (2·8·8 + 1) = 1,290` diffusion parameters per block before the head, plus a small readout-width increase. fileciteturn14file0L3-L3 fileciteturn15file0L3-L3

The proposer should also treat this as a *diagnostics upgrade*. Once these planes exist, the model can report `xray_pressure`, `skewer_pressure`, `discovered_pressure`, `pinned_defender_pressure`, and `overload_pressure` alongside the existing `pin_pressure`, `sheaf_tension`, `defense_gap`, and `reply_pressure`. That makes future failure analysis sharply easier than trying to infer everything from one coarse pin plane. fileciteturn17file0L3-L3

## Falsifiers and training protocol

The repo already established the right style of falsifier for i018: degree-preserving scrambling of tactical relation planes, with the thesis rejected if performance is unchanged. i252 should preserve that logic, but localize it to the new dependency family instead of scrambing the whole graph immediately. This keeps the test aligned with the architecture change being proposed. fileciteturn23file0L3-L3 fileciteturn18file0L3-L3

| Ablation or falsifier | What changes | What would falsify the upgrade |
|---|---|---|
| Dependency-only scramble | Randomly rewire only the 10 new planes with degree-preserving column permutations; keep the original 12 planes real | If target-slice gains and near-puzzle FP gains disappear only weakly or not at all, the new family is not doing real work |
| Family collapse | Replace the 10 new planes by one generic `tactical_dependency` plane | If performance matches the fully typed version, x-ray/skewer/discovered/overload/pinned-defender separation is unnecessary |
| No pinned-defender planes | Keep x-ray/skewer/discovered/overload, drop only `*_with_pinned_defender` | If pin-slice and near-puzzle behavior do not move, defender invalidation is not the missing signal |
| No overload planes | Keep everything else, drop overload | If overload-slice PR AUC and matched-recall near-FP do not move, second-duty modeling is not buying enough |
| Value-blind skewer/overload | Use screen occupancy only, with no `μ` or `ν` ordering | If the typed, value-aware version is not better, fixed piece-value ordering should be removed for simplicity |
| No self-pin legality filter on discovered planes | Let pinned screens still fire discovered candidates | If the legal filter does not help, candidate masking can be simplified |

Before any training run, add synthetic board fixtures for each motif family. Because these relations are definitional, exact unit tests are more revealing than aggregate learning curves: one-screen x-ray should fire with exactly one blocker and fail with two; skewer should fire only when the front enemy piece outranks the rear; discovered attack should vanish when the screening piece is self-pinned; pinned-defender exposure should increase target vulnerability only when the target is actually defended by the pinned piece; overload should rise when a defender’s second meaningful assignment becomes nonzero. Those are not optional niceties—they are the minimum defense against silently wrong geometry.

For model fitting, use the repo’s corrected reliability discipline. The research audit recorded that binary runs now default to `training.monitor: pr_auc`, after the earlier hard-coded F1 selection bug was fixed, and it explicitly elevated matched-recall false positives and worst-slice behavior as key evaluation outputs. So i252 should inherit that training-monitor choice and those report scripts from day one. fileciteturn13file0L3-L3

The practical training protocol should be staged. **Stage one** is a correctness scout: `simple_18`, one seed, mixed precision, batch `256`, BCE-with-logits, balanced class weighting, gradient clip `1.0`, and a short 20-epoch schedule matching the current i018 falsifier config. **Stage two** is the real comparison: three seeds, same scale sweep used for i249 (`base`, `scale_up`, `scale_xl`), same optimizer budget as the ancestor run, and direct comparison against the fast i249/i018 baseline plus the dependency-scrambled control. fileciteturn23file0L3-L3 fileciteturn14file0L3-L3

Acceptance should be explicit. I recommend adopting the upgrade only if, over three seeds, it satisfies all three conditions: overall PR AUC is not worse than i249-fast by more than `0.003`; matched-recall near-puzzle FP improves at recall `0.80` or `0.85`, or the mean PR AUC across the four target motif slices (`pin`, `skewer`, `overload`, `discovered_attack`) rises by at least `0.010`; and the dependency-scrambled control loses most of that gain. If those conditions are not met, the correct conclusion is not “train longer,” but “the richer graph did not buy enough signal.” This is exactly the discipline the user asked for.

## Risks and final recommendation

One real implementation risk is **relation redundancy**. A single geometry can fire x-ray, skewer, and pinned-defender exposure simultaneously. If left unchecked, the model may simply overcount the same line feature three times. The mitigation is straightforward: clamp every plane to `[0,1]`, normalize per-family densities before readout, and include a small relation-family dropout over the ten new planes so the model cannot depend on only one redundant coding.

A second risk is **false discovered-attack optimism**. Board-only geometry cannot prove that a screening piece has a tactically safe move, only that moving it would reveal a line. The mitigation is to keep the plane explicitly named `candidate`, exclude kings as screens, and suppress self-pinned screens from the candidate mask. If later ablations show this is still too loose, add an optional pseudo-legal mobility factor without ever consulting search. That keeps the feature board-only. fileciteturn18file0L3-L3

A third risk is **diagnostic index drift**. The current model hard-codes relation slices by numeric position, so adding ten planes can silently corrupt `pin_pressure` and ray-energy reporting if the code is not refactored first. The mitigation is to convert every relation-family diagnostic to name-based indexing before introducing the new planes. That refactor should happen before any first training run, not after unexplained discrepancies appear. fileciteturn17file0L3-L3

A fourth risk is **using too much architecture change at once**. If the builder, diffusion block, readout, and training schedule all move simultaneously, no ablation will be interpretable. The mitigation is to keep the sheaf operator class unchanged, inherit i249’s fast execution path as-is, and alter only the incidence builder, relation index map, and a small set of diagnostic/readout features. That preserves the scientific meaning of the result. fileciteturn14file0L3-L3 citeturn3academia2turn3academia0

**Final implementation recommendation.** Land this as a new idea, `i252_pin_xray_overload_sheaf`, implemented as a fast i249-style sheaf net with a `TacticalIncidenceBuilderV2` that preserves i018’s 12 existing planes and adds exactly the 10 bounded dependency planes above. Do **not** introduce a generic hypergraph, a second message-passing stack, or engine-derived supervision. Refactor diagnostics to be name-based, keep the diffusion block mathematically unchanged, expose five new pressure diagnostics, and ship the upgrade only if it beats the dependency-scrambled control on the repo’s own target outputs: matched-recall near-puzzle FP and the `pin` / `skewer` / `overload` / `discovered_attack` motif slices. That is the highest-information, lowest-churn way to make i018’s graph genuinely richer without departing from what the repository already built well. fileciteturn8file0L3-L3 fileciteturn14file0L3-L3 fileciteturn25file0L3-L3 fileciteturn27file0L3-L3