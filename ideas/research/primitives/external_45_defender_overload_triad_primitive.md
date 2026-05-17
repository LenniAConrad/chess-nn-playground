# Defender Overload Triad Primitive

Filename: `p044_defender_overload_triad_primitive.md`

## Context and thesis

### Thesis

The repository already has most of the raw chess structure needed for a strong overload primitive. The current i018 architecture canonicalizes the board to side-to-move perspective, constructs a 64-square tactical incidence tensor with 12 typed relations, includes attack, defense, king-zone, ray-visibility, and king-ray pin-candidate masks, and then optionally applies `TriadDefectPool`. That existing triad pool forms attacker and defender means per target, weights by `attack_in * defense_in * target_piece`, and emits only four pooled outputs, so it captures tension but does not explicitly measure whether the **same defender** is being reused across several critical targets. fileciteturn10file0L1-L3 fileciteturn12file0L1-L3 fileciteturn13file0L1-L3

In chess-programming terms, overloading is not generic “many-node tension.” It is a defender having more than one exclusive defensive obligation on critical squares, and those critical squares can be ordinary occupied targets or mate-threatening squares near the king. Classical attack/defend maps and attacks-to-a-square are already square-centric data structures used for evaluation, move generation, and static exchange evaluation, which makes a target-pooled overload primitive the natural fit here rather than an explicit cubic attacker × target × defender tensor. citeturn4view6turn4view4turn5view2turn5view1

The right replacement for the legacy triad pool is therefore a **Defender Overload Triad** operator with three moves: compute target criticality from attack/defense counts, pin information, and target value; pull that criticality back onto defender squares to estimate each defender’s total obligation load; then attribute the resulting overload burden back to targets and finally to a small, side-asymmetric feature vector. This stays board-only, works directly from relation masks already allowed by the project contract, and remains implementable in plain tensor algebra. fileciteturn20file0L1-L3 fileciteturn26file0L1-L3

## Formalization and equations

### Formal triad definition

Let `N = 64`. For one attacking side `σ ∈ {us, them}`, define:

```text
Aσ ∈ [0,1]^(B × N × N)    attacker-to-target mask
Dσ ∈ [0,1]^(B × N × N)    defender-to-target mask for the defending side
πσ ∈ [0,1]^(B × N)        pinned-defender indicator on defender squares
v_tarσ ∈ R_+^(B × N)      target-square piece value
v_defσ ∈ R_+^(B × N)      defender-square piece value
v_attσ ∈ R_+^(B × N)      attacker-square piece value
```

`Aσ[b, i, t]` is nonzero when attacker square `i` attacks target square `t`. `Dσ[b, d, t]` is nonzero when defender square `d` protects target `t`. In the repository’s current incidence layout, the natural piece-target choices are:

```text
σ = us:    Aσ = us_attacks_them_piece,   Dσ = them_defends_them_piece
σ = them:  Aσ = them_attacks_us_piece,   Dσ = us_defends_us_piece
```

and the existing `pin_mask` already encodes slider-to-blocker pin candidates, so summing it over slider source squares yields a blocker-level pin indicator that can be intersected with `our_piece` or `them_piece` to obtain side-specific pinned defenders. fileciteturn12file0L1-L3

A **Defender Overload Triad** is any weighted triple `(a, t, d)` such that attacker square `a` attacks target `t` and defender square `d` defends `t`. Its weight should rise when target `t` is critical and defender `d` is already committed elsewhere. The clean definition is:

```text
τσ(a, t, d) = Aσ(a, t) · Dσ(d, t) · cσ(t) · rσ(d, t)
```

where `cσ(t)` is target criticality and `rσ(d, t)` is defender residual load:

```text
Oσ(d, t) = Dσ(d, t) · cσ(t)
Lσ(d)    = Σ_u Oσ(d, u)
rσ(d, t) = (1 + μ · πσ(d)) · (Lσ(d) - Oσ(d, t))
```

So the triad says: a target matters when it is pressured, and a defender matters when it is defending that target **and also something else**. This follows the engine-style square-centric notion of overload and pin-aware restriction rather than a generic hypergraph motif language. citeturn4view6turn5view0turn5view2

### Efficient equations

The target-criticality features should be computed by target square first:

```text
a      = Aσ^T 1                    attack count per target
d      = Dσ^T 1                    defense count per target
p      = Dσ^T πσ                   pinned-defender count per target
a_val  = Aσ^T v_attσ               attacker-value sum per target
d_val  = Dσ^T ((1 - λ_pin πσ) ⊙ v_defσ)   effective defender-value sum
m_att  = masked_min(v_attσ, Aσ)    cheapest attacker per target
m_def  = masked_min(v_defσ, Dσ ⊙ (1 - πσ)) cheapest unpinned defender
```

Then define a small explicit target feature vector

```text
x_t = [a_t, d_t, p_t, a_val_t, d_val_t, m_att_t, m_def_t, v_tar_t]
```

and turn it into a scalar criticality

```text
c_t = softplus(gθ(x_t)) · 1[target exists]
```

where `gθ` is a tiny per-target MLP. If a zero-parameter sanity ablation is desired, replace it with

```text
c_t = v_tar_t · relu(a_t - (d_t - λ_pin p_t)).
```

Now pull criticality back to defenders:

```text
O = Dσ ⊙ c.unsqueeze(1)            obligation from defender to target
L = O.sum(dim=2)                   total obligation per defender
m = 1 + μ · πσ                     pin-amplified defender fragility
```

From this, the defender burden is

```text
Ω_def = m ⊙ (L ⊙ L - (O ⊙ O) 1)
```

and the target overload exposure is

```text
X_tar = c ⊙ [ Dσ^T (m ⊙ L) - c ⊙ (Dσ ⊙ Dσ)^T m ].
```

The key identity is:

```text
L_d^2 - Σ_t O_dt^2 = Σ_{t ≠ u} O_dt O_du.
```

So `Ω_def[d]` is exactly the weighted mass of **distinct critical targets simultaneously assigned to the same defender**. That is the overload signal, and it only uses `O(BN^2)` tensor work. The same square-centric viewpoint is standard in engine attack-to-square logic and SEE-style exchange reasoning, but here it is used as a differentiable pooling primitive rather than a search heuristic. citeturn4view4turn5view2turn5view1

## Feature design

### Chess features

Pins must be first-class, not an afterthought. The repository’s incidence builder already precomputes king-blocker-slider triples with between-square occupancy gating and emits a `pin_mask`; the pin literature also emphasizes that pinned pieces matter both for legal move generation and evaluation, including reduced mobility and “working the pin.” The overload primitive should therefore discount or amplify defenders through `πσ`, not treat every defender as equally active. In a first version, absolute king-ray pins from the current builder are sufficient. In a second version, relative pins can be added by extending the pin builder from “king behind blocker” to “higher-valued piece behind blocker.” fileciteturn12file0L1-L3 citeturn5view0

Target value must also be first-class. Classical point-value scales use roughly pawn `1`, knight `3.2`, bishop `3.3`, rook `5`, queen `9`, with the king often represented by a sentinel or large special value in engines. For this primitive, the king should **not** receive its full sentinel value, because i018 already represents king danger separately through `us_attacks_empty_near_king`, `them_attacks_empty_near_king`, and a `king_ring_pressure` diagnostic; a literal king value would swamp ordinary overload statistics. The practical choice is to clip `K` to queen-level for occupied-target overload and optionally run a second overload pass on empty king-ring targets with a surrogate `v_ring` constant. citeturn6view0turn6view1 fileciteturn10file0L1-L3 fileciteturn13file0L1-L3

A strong feature set is therefore: attack count, defense count, pinned-defender count, attacker-value sum, effective defender-value sum, cheapest attacker, cheapest unpinned defender, target value, and optional king-ring pressure. The cheapest-attacker and cheapest-defender terms are only SEE-like proxies, not a full recursive SEE, but SEE is explicitly a single-square exchange abstraction in engine practice, so these light-weight features are exactly the right compromise here. citeturn5view1

The final pooled side vector should remain small and interpretable. A good default is

```text
Sσ = [
  mean(X_tar),
  max(X_tar),
  mean(Ω_def on defending pieces),
  pinned overload share,
  mean(c)
]
```

and the final primitive output should be

```text
F = [S_us, S_them, S_us - S_them, |S_us - S_them|].
```

That gives an output dimension of `20`: five us-features, five them-features, five signed asymmetry features, and five absolute imbalance features. This directly satisfies the prompt’s requirement for us/them asymmetry and imbalance, and it is still tiny relative to the current i018 readout, which already concatenates four node pools, per-relation energy and density summaries, the legacy triad vector, and board statistics. fileciteturn13file0L1-L3

### Complexity

The primitive should be strictly `O(BN^2)` in its deterministic core and `O(BN^2 d)` only if an optional hidden-state context projection is added. With `N = 64`, that means the operator stays in the same asymptotic regime as the repository’s existing relation-mask construction and remains far cheaper than any explicit `O(BN^3)` triple materialization. Square-centric attack maps and attacks-to-a-square are already a standard engine abstraction precisely because they let many tactical statistics be computed by target aggregation rather than by enumerating all combinations. citeturn4view4turn5view2

In practice the steps are just a handful of reductions, masked minima, elementwise products, and batched matrix multiplies. If the target MLP uses only the explicit scalar features above, there is no large activation tensor beyond the already-existing `B × 64 × 64` relation masks. If a direction-aware refinement is later added, it can still remain `O(BN^2K)` with a tiny constant `K` for direction bins. The added readout width is also modest: replacing the current 4-dim triad vector with a 20-dim overload vector only adds 16 features to a readout that is already much larger. fileciteturn13file0L1-L3

This also fits the repository’s fast-path philosophy. i249 keeps the i018 adapter, incidence builder, triad pool, and head unchanged and only accelerates the diffusion block, so a shared overload primitive living above incidence and below readout will port cleanly to the fast model without changing its core numerical story. fileciteturn17file0L1-L3 fileciteturn18file0L1-L3

## Integration and implementation

### Integration plan

The primitive should be integrated **after** `TacticalIncidenceBuilder` and **before** the readout head. No new input channels are required, and no engine or metadata features are introduced. That fits both the repository’s implementation notes and the broader board-only contract, which explicitly allows deterministic features such as occupancy, pseudo-legal attacks, visible rays, defenses, and pin geometry derived from the board tensor alone. fileciteturn20file0L1-L3 fileciteturn26file0L1-L3

At the code level, the cleanest change is to add a new `DefenderOverloadTriadPool` class to `src/chess_nn_playground/models/trunk/oriented_tactical_sheaf.py` next to the current `TriadDefectPool`. The first shipping version can derive side-specific pinned defenders from the existing combined `pin_mask`:

```text
pinned_any  = incidence.pin_mask.sum(dim=1).clamp(0, 1)
pinned_us   = pinned_any * incidence.our_piece
pinned_them = pinned_any * incidence.them_piece
```

It should use relations `0..3` for piece-target overload, optionally relations `4..5` for king-ring overload, and piece values read directly from `piece_state`. Because the current piece ordering in the source is pawn, knight, bishop, rook, queen, king for each side, value extraction is a single fixed linear projection of the corresponding slice. fileciteturn12file0L1-L3

The migration plan should be staged. First, add a config flag such as `triad_mode ∈ {"legacy", "overload", "both"}`. Second, if `both`, concatenate the two primitive outputs and let the head learn fusion. Third, add overload-specific diagnostics to `forward`, such as `overload_us_mean`, `overload_them_mean`, `overload_peak`, `overload_pinned_share`, and `overload_signed_imbalance`. That makes the new mechanism auditable in the same way that i018 already reports `triad_defect_energy`, `pin_pressure`, `king_ring_pressure`, and `defense_gap`. fileciteturn13file0L1-L3

Once the primitive is stable in i018, i249 should inherit it automatically. That fast model explicitly imports the adapter, incidence builder, triad pool, and head logic from the original module and only swaps the diffusion block execution pattern, so keeping the overload primitive in the shared source file preserves exact architectural alignment between the slow and fast variants. fileciteturn17file0L1-L3 fileciteturn18file0L1-L3

### Implementation sketch

```python
import torch
from torch import nn

class DefenderOverloadTriadPool(nn.Module):
    """Square-centric overload primitive over existing relation masks."""

    def __init__(self, dropout: float, hidden: int = 16, output_dim: int = 20):
        super().__init__()
        self.pin_discount_logit = nn.Parameter(torch.tensor(1.1))   # sigmoid -> ~0.75
        self.pin_boost_logit = nn.Parameter(torch.tensor(0.0))      # sigmoid -> ~0.5
        self.target_gate = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.out_norm = nn.Sequential(nn.LayerNorm(output_dim), nn.Dropout(dropout))
        self.output_dim = output_dim
        # pawn, knight, bishop, rook, queen, king-clipped
        self.register_buffer("piece_values", torch.tensor([1.0, 3.2, 3.3, 5.0, 9.0, 9.0]), persistent=False)

    def _piece_values_from_state(self, piece_state: torch.Tensor, ours: bool) -> torch.Tensor:
        sl = slice(1, 7) if ours else slice(7, 13)
        return piece_state[..., sl] @ self.piece_values.to(piece_state)

    def _masked_min(self, adj: torch.Tensor, src_vals: torch.Tensor) -> torch.Tensor:
        # adj: [B, N, N], src_vals: [B, N]
        inf = torch.full((), float("inf"), dtype=src_vals.dtype, device=src_vals.device)
        vals = src_vals.unsqueeze(-1).expand_as(adj)
        mins = vals.masked_fill(adj <= 0, inf).amin(dim=1)
        return torch.where(torch.isfinite(mins), mins, mins.new_zeros(mins.shape))

    def _side_stats(
        self,
        attack: torch.Tensor,       # [B, N, N]
        defense: torch.Tensor,      # [B, N, N]
        pinned_def: torch.Tensor,   # [B, N]
        v_att: torch.Tensor,        # [B, N]
        v_def: torch.Tensor,        # [B, N]
        v_tar: torch.Tensor,        # [B, N]
        defender_occ: torch.Tensor, # [B, N]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pin_discount = torch.sigmoid(self.pin_discount_logit)
        pin_boost = torch.sigmoid(self.pin_boost_logit)

        a = attack.sum(dim=1)
        d = defense.sum(dim=1)
        p = (defense * pinned_def.unsqueeze(-1)).sum(dim=1)
        a_val = (attack * v_att.unsqueeze(-1)).sum(dim=1)
        d_val = (defense * ((1.0 - pin_discount * pinned_def) * v_def).unsqueeze(-1)).sum(dim=1)
        m_att = self._masked_min(attack, v_att)
        m_def = self._masked_min(defense * (1.0 - pinned_def).unsqueeze(-1), v_def)

        x = torch.stack([a, d, p, a_val, d_val, m_att, m_def, v_tar], dim=-1)
        c = torch.nn.functional.softplus(self.target_gate(x).squeeze(-1)) * (v_tar > 0).to(v_tar.dtype)

        O = defense * c.unsqueeze(1)
        L = O.sum(dim=2)
        m = 1.0 + pin_boost * pinned_def

        defender_burden = m * (L.square() - O.square().sum(dim=2))
        target_exposure = c * (
            torch.bmm(defense.transpose(1, 2), (m * L).unsqueeze(-1)).squeeze(-1)
            - c * torch.bmm((defense.square()).transpose(1, 2), m.unsqueeze(-1)).squeeze(-1)
        )

        target_mass = c.sum(dim=1).clamp_min(1e-6)
        exposure_mass = target_exposure.sum(dim=1).clamp_min(1e-6)
        pinned_share = (target_exposure * (p / d.clamp_min(1.0))).sum(dim=1) / exposure_mass
        mean_burden = (defender_burden * defender_occ).sum(dim=1) / defender_occ.sum(dim=1).clamp_min(1.0)

        vec = torch.stack(
            [
                target_exposure.sum(dim=1) / target_mass,   # mean overload exposure
                target_exposure.amax(dim=1),                # peak overload exposure
                mean_burden,                                # mean defender burden
                pinned_share,                               # pinned overload share
                target_mass / (v_tar > 0).sum(dim=1).clamp_min(1.0),  # mean criticality
            ],
            dim=1,
        )
        aux = {
            "criticality": c,
            "target_exposure": target_exposure,
            "defender_burden": defender_burden,
            "attack_count": a,
            "defense_count": d,
            "pinned_defense_count": p,
        }
        return vec, aux

    def forward(self, board, incidence):
        pinned_any = incidence.pin_mask.sum(dim=1).clamp(0.0, 1.0)
        our_val = self._piece_values_from_state(board.piece_state, ours=True)
        them_val = self._piece_values_from_state(board.piece_state, ours=False)

        us_vec, us_aux = self._side_stats(
            attack=incidence.relation_masks[:, 0],
            defense=incidence.relation_masks[:, 3],
            pinned_def=pinned_any * incidence.them_piece,
            v_att=our_val,
            v_def=them_val,
            v_tar=them_val,
            defender_occ=incidence.them_piece,
        )
        them_vec, them_aux = self._side_stats(
            attack=incidence.relation_masks[:, 1],
            defense=incidence.relation_masks[:, 2],
            pinned_def=pinned_any * incidence.our_piece,
            v_att=them_val,
            v_def=our_val,
            v_tar=our_val,
            defender_occ=incidence.our_piece,
        )

        diff = us_vec - them_vec
        out = torch.cat([us_vec, them_vec, diff, diff.abs()], dim=1)
        return self.out_norm(out), {"us": us_aux, "them": them_aux}
```

The only optional extension omitted above is the king-ring channel. That can be added by running a second `_side_stats` pass over relations `4` and `5` with a constant `v_ring`, then concatenating the piece-target and king-ring summaries.

## Evaluation design

### Falsifiers and ablations

The central falsifier must be a **defender-shuffle falsifier**, not a generic relation scramble. For each target square `t`, independently permute the source rows of the defender matrix `D[:, :, t]` within defender buckets defined by piece value and pin status. That preserves, per target, the number of defenders, the approximate defender-value histogram, and the pinned-defender count, while destroying the one thing the primitive cares about: whether the **same defender identity** carries several targets at once. If accuracy barely changes, the overload-by-reuse thesis is false even if ordinary target pressure remains useful.

The main ablations should be surgical. `no_pins` sets `π = 0` everywhere. `no_target_value` sets `v_tar = 1` on all occupied targets. `no_cross_target_load` replaces `L_d - O_dt` with `0`, leaving only single-target under-defense and removing overload proper. `counts_only` drops `a_val`, `d_val`, `m_att`, and `m_def` from the target gate. `piece_only` omits the optional king-ring overload pass. `legacy_only`, `overload_only`, and `both` test whether the new primitive replaces or merely complements the current `TriadDefectPool`. These ablations are also well aligned with repository precedent: i018 already uses a degree-preserving relation-scramble falsifier, and the counterfactual defender-dropout model already treats overloaded defenders, pinning sliders, and escape squares as causally critical and uses structure-destroying ablations such as `random_masks`, `defenders_only`, and `no_intervention_head`. fileciteturn19file0L1-L3 fileciteturn16file0L1-L3

### Experiment matrix

The repository’s current practice is to screen with three seeds and then scale promising variants; i018 reports three-seed base results, and i249’s benchmark plan uses seeds `42, 43, 44` across model scales. The matrix below follows that pattern and keeps PR-AUC as the primary score. fileciteturn19file0L1-L3 fileciteturn17file0L1-L3

| Variant | Primitive setting | Purpose | Scale and seeds | Pass signal |
|---|---|---|---|---|
| `baseline_i018` | legacy triad only | Anchor against current sheaf model | base, seeds 42/43/44 | Match current baseline behavior |
| `overload_replace` | overload only | Test sufficiency of new primitive | base, seeds 42/43/44 | At least parity with legacy triad |
| `overload_plus_legacy` | concatenate legacy + overload | Test complementarity | base, seeds 42/43/44 | Preferred main candidate if it beats both single modes |
| `no_pins` | overload with `π = 0` | Test whether pinned defenders matter | base, seeds 42/43/44 | Meaningful drop versus `overload_plus_legacy` |
| `no_target_value` | set `v_tar = 1` | Test whether value weighting matters | base, seeds 42/43/44 | Meaningful drop, especially on material-tactic puzzles |
| `counts_only` | remove value sums and minima from target gate | Test whether SEE-light/value features add signal | base, seeds 42/43/44 | Lower than full overload model |
| `no_cross_target_load` | zero `L - O` term | Test overload proper versus plain under-defense | base, seeds 42/43/44 | Clear drop if defender reuse is real mechanism |
| `defender_shuffle` | shuffle defender rows independently per target within value × pin buckets | Falsifier | base, seeds 42/43/44 | `≥ 0.01` PR-AUC drop is meaningful; `≥ 0.02` is strong support |
| `piece_plus_kingring` | add king-ring overload channel | Test mate-square extension | base, seeds 42/43/44 | Helps king-net and mating-grid cases without harming others |
| `best_fast_port` | port winner into i249-style fast net | Efficiency validation | base / 1.5 / xl, seeds 42/43/44 | Accuracy within noise of winner, throughput notably better |

## Expected impact

### Expected improvements

A realistic target is a **modest but real** lift, not a miracle jump. The repo’s i018 baseline over three seeds is reported at `0.8752 ± 0.0045` test PR-AUC, while the best recent single-primitive hybrids grafted onto i018 improved PR-AUC by about `+0.0056` to `+0.0065`. That makes a defender-overload primitive look more like a “good primitive graft” opportunity than a whole-family rewrite. The most plausible target range is therefore roughly `+0.004` to `+0.008` PR-AUC when used as a complement to the existing sheaf readout, with replacement-only performance possibly flatter. This is an inference from the repo’s current hybrid-graft behavior, not an observed result for this primitive yet. fileciteturn19file0L1-L3

The expected gains should concentrate in exactly the motifs named in the prompt: overloaded defenders, pinned defenders, unstable attack/defense triples, and king-net positions where multiple critical squares compete for the same few defenders. That expectation is reinforced by another architecture already in this repository, the counterfactual defender-dropout network, whose whole causal story is that overloaded defenders, pinning sliders, and the one escape square are often the critical participants in puzzle-like positions. fileciteturn16file0L1-L3

The primitive should not be oversold. The i018 math thesis already warns that quiet endgame studies, zugzwang-like positions, or other puzzle-like examples with weak static attack/defense incidence may evade this family, and that non-puzzle but tactically noisy positions can still show high tension. A defender-overload primitive shares those limitations. If `overload_plus_legacy` is only marginally better than baseline and the defender-shuffle falsifier is weak, the correct conclusion is that overload is mostly redundant with existing i018 relation geometry rather than a new mechanism worth scaling. fileciteturn19file0L1-L3

The more important upside is mechanistic clarity. Full relation scrambling in i018 already costs about `-0.0424` PR-AUC, which shows that real chess geometry matters as a whole. The overload primitive asks a sharper question: does **defender identity reuse** matter on top of geometry? If the defender-shuffle falsifier causes a material drop while `no_cross_target_load` also degrades, then the repository will have isolated a specific, interpretable tactical mechanism rather than just another anonymous feature block. fileciteturn19file0L1-L3