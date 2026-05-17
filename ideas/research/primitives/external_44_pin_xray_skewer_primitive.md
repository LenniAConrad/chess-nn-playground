# p043_pin_xray_skewer_primitive.md

## Thesis

The repository already has most of the raw geometry needed for a strong sliding-piece tactic primitive. The current i018 `Oriented Tactical Sheaf Laplacian` canonicalizes the board to side-to-move orientation, builds exact visible rook/bishop/queen rays from precomputed masks, and already includes a coarse `king_ray_pin_candidate` relation inside a 12-relation tactical incidence tensor. That means p043 does **not** need to invent new board geometry; it should refine existing geometry into a sharper, value-aware, reusable operator. fileciteturn11file0L3-L3 fileciteturn25file0L3-L3

The strongest design direction is to distill **ordered blocker facts** into a native tensor primitive. That is also consistent with the repo’s later i190 `Blocker-Pin Lattice Network`, whose core claim is that slider tactics depend on the **order** of blockers and on pin constraints, not just on line sharing. Its implementation explicitly computes first, second, and third occupied squares on every ray and uses “remove-first” and “remove-second” counterfactual states. p043 should borrow that exactness, but stop short of a separate lattice trunk: the reusable primitive should emit crisp event tensors for pins, x-rays, skewers, discovered attacks, and pinned-defender load. fileciteturn19file0L3-L3 fileciteturn20file0L3-L3 fileciteturn21file0L3-L3

The repo also shows what **not** to do. p020, p023, and p034 are all useful ray-adjacent ideas, but they are additive heads whose defining scan logic uses Python-side sequential loops, and both p020 and p034 explicitly point to a fused CUDA or Triton kernel as the deferred production speed path. The BT4 mixer variants are also described as **controlled architecture studies**, not the final resting place for a primitive. So p043 should be neither a generic “ray feature” head nor a BT4-only mixer. It should be a small, exact, tensorized geometry module that can run natively inside i018 and can also be wrapped as a standalone head for clean ablations. fileciteturn26file0L3-L3 fileciteturn28file0L3-L3 fileciteturn29file0L3-L3 fileciteturn38file0L3-L3 fileciteturn39file0L3-L3

That thesis is also compatible with the repo’s current evidence standard. The i018 thesis page records that i018 is already strong, that some primitive grafts helped while others regressed, and that the geometry falsifier meaningfully hurt performance when real relations were scrambled. So p043 should be justified as a **more specific geometry prior**, not as a generic add-on. Any claim that it works must be backed by ablations and slice movement, especially on `pin`, `skewer`, and `discovered_attack`. fileciteturn12file0L3-L3 fileciteturn30file0L3-L3 fileciteturn31file0L3-L3

## Ray geometry math

### Geometry basis

Chessprogramming’s taxonomy is a good fit for the primitive the repo needs: **pin, skewer, discovered attack, and x-ray** are all sliding-piece motifs that differ mainly by what the blocker is and what sits behind it on the same ray. Chessprogramming also notes that absolute pin detection in engines is commonly built from x-ray routines plus an **in-between lookup**. citeturn3view0turn3view1turn4view1

The repo already has both representations needed for that:

- an **ordered ray-step table** in `ray_geometry.py`, with `RAY_STEP_INDEX (8, 64, 7)` and `RAY_STEP_MASK (8, 64, 7)` for gather-based directional scans; and
- a **pairwise in-between tensor** in i018’s `_make_geometry_masks`, where `between[src, dst, q] = 1` when square `q` lies strictly between `src` and `dst` on a rook or bishop line. fileciteturn23file0L3-L3 fileciteturn25file0L3-L3

Use both, but for different jobs:

- **ordered ray tables** for fast training-time event extraction;  
- **between-square masks** for exact pairwise visibility checks, unit tests, and direct integration with i018’s relation builder. fileciteturn23file0L3-L3 fileciteturn25file0L3-L3

Let \(O \in \{0,1\}^{64}\) be occupancy, and let \(L_f(s,t)\) be the slider-family compatibility mask for family \(f \in \{\text{rook},\text{bishop},\text{queen}\}\). With the precomputed in-between tensor \(B(s,t,q)\), define the blocker count:

\[
N(s,t) = \sum_{q=1}^{64} B(s,t,q)\,O(q).
\]

Then the exact visibility masks are:

\[
C_0^{(f)}(s,t) = L_f(s,t)\,\mathbf{1}[N(s,t)=0]
\]

\[
C_1^{(f)}(s,t) = L_f(s,t)\,\mathbf{1}[N(s,t)=1].
\]

\(C_0\) is the clear-ray mask. \(C_1\) is the **one-blocker x-ray** mask. This is the fixed-mask tensor analogue of the standard x-ray bitboard trick on Chessprogramming, where one computes attacks once, removes blockers on the attacked ray, and compares the occupancy-modified attacks. citeturn3view1turn4view1

### Ordered blocker extraction

The ordered form is what makes the primitive useful for scoring. Using `RAY_STEP_INDEX`, gather occupancy and piece facts along each directed ray:

\[
o_{d,s,\ell} = O(R[d,s,\ell]) \cdot M[d,s,\ell]
\]

for directions \(d \in \{1,\dots,8\}\), sources \(s \in \{1,\dots,64\}\), and steps \(\ell \in \{1,\dots,7\}\). Then compute cumulative occupancy:

\[
c_{d,s,\ell} = \sum_{j \le \ell} o_{d,s,j}, \qquad
p_{d,s,\ell} = c_{d,s,\ell} - o_{d,s,\ell}.
\]

Now everything needed for ordered blocker logic appears without loops:

\[
\text{first}_{d,s,\ell} = o_{d,s,\ell}\,\mathbf{1}[c_{d,s,\ell}=1]
\]

\[
\text{second}_{d,s,\ell} = o_{d,s,\ell}\,\mathbf{1}[c_{d,s,\ell}=2]
\]

\[
\text{third}_{d,s,\ell} = o_{d,s,\ell}\,\mathbf{1}[c_{d,s,\ell}=3]
\]

\[
\text{clear\_step}_{d,s,\ell} = M[d,s,\ell]\,\mathbf{1}[p_{d,s,\ell}=0]
\]

\[
\text{xray1\_step}_{d,s,\ell} = M[d,s,\ell]\,\mathbf{1}[p_{d,s,\ell}=1].
\]

This is the right abstraction level for p043. It captures the exact blocker order that i190 cares about, but it stays inside pure tensor ops such as gather, `cumsum`, comparison, and `scatter_add_`, avoiding the Python scan loops that limit earlier ray heads. fileciteturn21file0L3-L3 fileciteturn26file0L3-L3 fileciteturn29file0L3-L3

## Pin x-ray skewer equations

### Core event equations

Chessprogramming’s tactical table is the cleanest way to organize the primitive: the same x-ray geometry underlies discovered attacks, pins, skewers, and latent x-ray pressure, and the differences come from blocker ownership and target value. citeturn3view1turn3view0turn4view1

Let the canonical mover-oriented piece planes be those already produced by i018’s `BoardStateAdapter`: `piece_state[:, :, 0]` is empty; `1:7` are “us”; `7:13` are “them”. Use a target-value field

\[
v(q) = w_P P(q) + w_N N(q) + w_B B(q) + w_R R(q) + w_Q Q(q) + w_K K(q),
\]

with initialization emphasizing the user’s requested targets, for example

\[
(w_P,w_N,w_B,w_R,w_Q,w_K)=(1,3,3,5,9,12)/12.
\]

That is close to the repo’s existing tactical weighting in i190, which already uses canonical piece-value-style scaling and explicitly singles out king or high-value targets behind blockers. fileciteturn21file0L3-L3

Define a source activation mask \(S_f(s)\) for slider family \(f\), using rook directions for rooks, bishop directions for bishops, and all 8 directions for queens. Then the proposed p043 event scores are:

**One-blocker x-ray pressure**

\[
X_{\text{xray}}(s,t) = S(s)\,C_1(s,t)\,v(t).
\]

This is not a generic ray feature. It is explicitly a **latent target-behind-one-blocker** score.

**Absolute pin**

Let \(b\) be the unique blocker on the line from slider \(s\) to king \(k\). Then

\[
X_{\text{abs-pin}}(s,b) = \sum_k S(s)\,M_1(s,k,b)\,K_{\text{opp}}(k),
\]

where

\[
M_1(s,t,b)=B(s,t,b)\,O(b)\,\mathbf{1}[N(s,t)=1].
\]

This is just the exact “one blocker between slider and king” condition that Chessprogramming recommends for absolute pin logic, expressed as a tensor instead of a per-piece loop. citeturn4view1turn3view0

**Relative pin**

\[
X_{\text{rel-pin}}(s,b) = \sum_t S(s)\,M_1(s,t,b)\,\big(w_Q\,Q_{\text{opp}}(t) + w_R\,R_{\text{opp}}(t)\big).
\]

The repo prompt explicitly calls out king/queen/rook weighting, so relative pin should emphasize queen and rook targets rather than all targets equally.

**Discovered attack potential**

\[
X_{\text{disc}}(s,b) = \sum_t S(s)\,M_1(s,t,b)\,\text{Own}(b)\,v_{\text{opp}}(t).
\]

This is the same one-blocker x-ray, but the blocker is **our own** piece, so the event is a discovered-attack candidate rather than a pin.

**Skewer**

Let \(t_1\) be the first enemy piece and \(t_2\) the second enemy piece on the same ray. Then

\[
X_{\text{skewer}}(s,t_1)
=
S(s)\,\text{Enemy}(t_1)\,\sum_{t_2} M_1(s,t_2,t_1)\,
\operatorname{ReLU}\big(v(t_1)-v(t_2)-\tau\big)\,v(t_2),
\]

with a small margin \(\tau \ge 0\). A skewer is exactly the “front target more valuable than back target” case; the front piece is the unique blocker on the line to the back piece. That matches standard skewer definitions and the x-ray taxonomy. citeturn3view1turn2search0

The key design decision is that p043 should emit **typed event tensors**, not a single undifferentiated “ray activation.” Earlier repo ray heads mostly pool or recurrently summarize lines. p043 should instead expose the exact tactical causes: “enemy blocker in front of king,” “own blocker in front of queen,” “valuable front piece with cheaper back piece,” and so on. That is how it stays reusable and how it avoids being a generic ray feature with no target or value context. fileciteturn26file0L3-L3 fileciteturn28file0L3-L3 fileciteturn29file0L3-L3

### Pinned defender scoring

Pinned pieces matter in evaluation not just because they are immobilized, but because they may be **important defenders**. Chessprogramming explicitly notes that engines often score restricted pinned mobility and even “working the pin” by checking what the pinned piece was guarding. citeturn3view0

The repo already has the right support signal for this inside i018: the tactical incidence builder computes per-source attack and defense matrices, including `us_defends_us_piece` and `them_defends_them_piece`, and the resulting attack matrices are available to downstream modules. fileciteturn11file0L3-L3 fileciteturn25file0L3-L3

So define same-side defender load for blocker square \(b\) as:

\[
D_{\text{def}}(b)
=
\sum_u A_{\text{same}}(b,u)\,\big(v(u)+\lambda_z Z_{\text{king-zone}}(u)\big),
\]

where \(A_{\text{same}}(b,u)\) is the nominal defense relation from \(b\) to friendly asset \(u\), and \(Z_{\text{king-zone}}\) upweights critical king-zone points if desired. Then the pinned-defender score is:

\[
X_{\text{pin-def}}(b)
=
\big(\alpha_{\text{abs}} X_{\text{abs-pin}}(\cdot,b)
+
\alpha_{\text{rel}} X_{\text{rel-pin}}(\cdot,b)\big)\,
D_{\text{def}}(b).
\]

This score matters because a pinned knight defending mate squares and a pinned rook defending a queen are not equally important, even if both satisfy the same geometric pin condition.

## Inputs outputs and integration

### Inputs and outputs

The natural input contract is the same one i018 already uses: side-to-move-canonical square tokens and piece-state planes. i018’s `BoardStateAdapter` already emits canonical raw planes, a mover-oriented 13-channel `piece_state`, occupancy, and side info, with `simple_18` as the primary exact encoding. That should be p043’s default input path. fileciteturn11file0L3-L3 fileciteturn45file0L3-L3

A practical output contract is:

| tensor | shape | meaning |
|---|---:|---|
| `edge_maps` | `(B, 6, 64, 64)` | typed pairwise tactical edges |
| `square_maps` | `(B, 6, 64)` | blocker/target pressure per square |
| `summary` | `(B, 12)` | pooled board-level pin/x-ray/skewer statistics |
| `diagnostics` | dict | scalar masses and densities |

The six edge types should be:

- `absolute_pin` source \(\rightarrow\) pinned blocker  
- `relative_pin` source \(\rightarrow\) pinned blocker  
- `xray_one_blocker` source \(\rightarrow\) target behind one blocker  
- `skewer_front` source \(\rightarrow\) front valuable target  
- `discovered_attack` source \(\rightarrow\) own removable blocker  
- `pinned_defender_gap` source \(\rightarrow\) pinned defender, weighted by defender load

This edge-centric layout matches i018’s existing relation convention, where relations are square-to-square masks over the 64-square incidence complex. fileciteturn11file0L3-L3 fileciteturn25file0L3-L3

### Integration into i018

The best native integration is to append the six new relations to i018’s current 12 relations, replacing the coarse `king_ray_pin_candidate` with richer line-tactic structure or, more conservatively, keeping the old relation for one ablation cycle and appending the new six. In the “append six” version, the relation count grows from 12 to 18. Because each i018 diffusion block stores only small per-relation \(8\times8\) restriction maps and one scalar gate per relation, that parameter increase is tiny relative to the rest of the model. fileciteturn11file0L3-L3 fileciteturn25file0L3-L3

The cleanest wiring is:

1. `BoardStateAdapter` unchanged.  
2. `TacticalIncidenceBuilder` gains a `PinXraySkewerBuilder`.  
3. `relation_masks = cat(old_relations, p043_relations, dim=1)`.  
4. `TacticalReadout` gains pooled summaries such as `absolute_pin_mass`, `relative_pin_mass`, `xray_mass`, `skewer_mass`, `discovered_attack_mass`, and `pinned_defender_mass`.  
5. Keep all existing reporting fields; add the new ones as diagnostics only.

This is preferable to a mixer-only design because i018 is already a relation-based architecture. p043 becomes another native relation family, not an unrelated spatial mixer. That is also more aligned with the repo’s current evidence than sending the idea straight to BT4 studies, which the repo documents as controlled mixer comparisons rather than finished primitive products. fileciteturn38file0L3-L3 fileciteturn39file0L3-L3

### Standalone head

A standalone `pin_xray_skewer_head` is still valuable for research staging. The repo already has a hybrid scaffold, `oriented_sheaf_plus_primitive`, that fuses a primitive logit with i018’s sheaf logit via

\[
\text{final\_logit}
=
\text{sheaf\_logit}
+
\sigma(g)\,\text{primitive\_logit}.
\]

That makes it easy to test whether p043 contributes complementary signal before fully rewriting i018’s relation builder. fileciteturn35file0L3-L3 fileciteturn32file0L3-L3 fileciteturn33file0L3-L3

So the recommended sequence is:

- **first**: standalone head + i018 hybrid, for easy complementarity testing;  
- **second**: native i018 relation integration, if the head is genuinely load-bearing.

That sequence matches the repo’s current habit of measuring primitive complementarity against i018 before promoting a deeper architectural merge. fileciteturn12file0L3-L3 fileciteturn30file0L3-L3

## Speed plan and implementation sketch

### Speed plan

p043 should be explicitly designed around **small fixed chess geometry** and **vectorized tensor kernels**, not around recurrent scans. The core ordered tensor has only \(8 \times 64 \times 7 = 3584\) ray cells per board, which is tiny. That means the whole primitive can be built from:

- one or a few gathers from `RAY_STEP_INDEX`,  
- boolean occupancy comparisons,  
- `cumsum` along the 7-step axis,  
- elementwise masking for first/second/third occupied squares, and  
- one `scatter_add_` back into pairwise `(64,64)` edge tensors. fileciteturn23file0L3-L3

That is materially different from p020 and p034, both of which explicitly document Python-side sequential scan loops as the dominant remaining speed problem. p043 should have **no Python loops at train time** over step length, direction, or source square. Even direction-family handling can be done with precomputed one-hot masks. fileciteturn26file0L3-L3 fileciteturn27file0L3-L3 fileciteturn29file0L3-L3

A concrete speed plan is:

- keep geometry buffers as non-persistent registered buffers;
- perform blocker counts in boolean or integer space, not float-thresholded soft counts;
- compute step facts in fp16 or bf16, but keep index tensors in `long`;
- emit pairwise edge maps in fp16, release step tensors immediately after scatter;
- reuse i018 attack/defense maps for pinned-defender load instead of recomputing defender graphs;
- wrap the builder in `torch.compile` once correctness is locked.

The central design point is that p043 should be fast **without** needing a deferred Triton rescue. That is realistic here because chess ray depth is capped at 7 and the tensor is minuscule.

### Implementation sketch

```python
import torch
from torch import nn

class PinXraySkewerBuilder(nn.Module):
    """
    Input:
        piece_state: (B, 64, 13)  # [empty, us P N B R Q K, them P N B R Q K]
        occupancy:  (B, 64)
        same_def:   (B, 64, 64)   # optional, from i018 attack builder
    Output:
        edge_maps:  (B, 6, 64, 64)
        square_maps:(B, 6, 64)
        summary:    (B, 12)
    """

    def __init__(self, ray_step_index, ray_step_mask, dir_is_orth, dir_is_diag):
        super().__init__()
        self.register_buffer("ray_step_index", ray_step_index, persistent=False)   # (8,64,7)
        self.register_buffer("ray_step_mask",  ray_step_mask,  persistent=False)   # (8,64,7)
        self.register_buffer("dir_is_orth",    dir_is_orth,    persistent=False)   # (8,)
        self.register_buffer("dir_is_diag",    dir_is_diag,    persistent=False)   # (8,)

        # Bounded learned scalars, initialized to canonical values.
        self.value_logits = nn.Parameter(torch.tensor([1., 3., 3., 5., 9., 12.]))
        self.rel_pin_scale = nn.Parameter(torch.tensor(0.0))
        self.pin_def_scale = nn.Parameter(torch.tensor(0.0))
        self.skewer_margin = nn.Parameter(torch.tensor(0.0))

    def forward(self, piece_state, occupancy, same_def=None):
        B = piece_state.size(0)
        flat_idx = self.ray_step_index.reshape(-1)

        # Gather step-wise facts: (B,8,64,7)
        occ_seq = occupancy[:, flat_idx].view(B, 8, 64, 7)
        occ_seq = occ_seq * self.ray_step_mask.view(1, 8, 64, 7)

        us = piece_state[:, :, 1:7]
        them = piece_state[:, :, 7:13]
        us_any = us.sum(-1).clamp(0, 1)
        them_any = them.sum(-1).clamp(0, 1)

        def gather_scalar(x):
            return x[:, flat_idx].view(B, 8, 64, 7) * self.ray_step_mask.view(1, 8, 64, 7)

        us_any_seq   = gather_scalar(us_any)
        them_any_seq = gather_scalar(them_any)
        them_king_seq  = gather_scalar(them[:, :, 5])
        them_queen_seq = gather_scalar(them[:, :, 4])
        them_rook_seq  = gather_scalar(them[:, :, 3])

        # Target value field
        vals = torch.softmax(self.value_logits, dim=0)
        target_value = (them * vals.view(1, 1, 6)).sum(-1)
        target_value += 0.5 * (us * vals.view(1, 1, 6)).sum(-1)  # optional for symmetric channels
        val_seq = gather_scalar(target_value)

        occ_bool = occ_seq > 0.5
        occ_ord = occ_bool.long().cumsum(dim=-1)
        prev_occ = occ_ord - occ_bool.long()

        first_occ  = occ_bool & (occ_ord == 1)
        second_occ = occ_bool & (occ_ord == 2)

        first_us   = first_occ & (us_any_seq > 0.5)
        first_them = first_occ & (them_any_seq > 0.5)

        second_them_king  = second_occ & (them_king_seq > 0.5)
        second_them_queen = second_occ & (them_queen_seq > 0.5)
        second_them_rook  = second_occ & (them_rook_seq > 0.5)
        second_them_any   = second_occ & (them_any_seq > 0.5)

        us_rook   = us[:, :, 3]
        us_bishop = us[:, :, 2]
        us_queen  = us[:, :, 4]

        src_slider = (
            us_queen.unsqueeze(1)
            + us_rook.unsqueeze(1)   * self.dir_is_orth.view(1, 8, 1)
            + us_bishop.unsqueeze(1) * self.dir_is_diag.view(1, 8, 1)
        ).clamp(max=1.0)

        # Event steps
        xray1_step = src_slider.unsqueeze(-1) * (prev_occ == 1).float() * val_seq

        abs_pin_step = (
            src_slider.unsqueeze(-1)
            * first_them.float()
            * second_them_king.any(dim=-1, keepdim=True).float()
        )

        rel_pin_step = (
            src_slider.unsqueeze(-1)
            * first_them.float()
            * (second_them_queen.any(dim=-1, keepdim=True).float()
               + 0.6 * second_them_rook.any(dim=-1, keepdim=True).float())
            * torch.sigmoid(self.rel_pin_scale)
        )

        disc_step = (
            src_slider.unsqueeze(-1)
            * first_us.float()
            * ((prev_occ == 1).float() * val_seq).sum(dim=-1, keepdim=True)
        )

        # Skewer uses first target > second target
        first_val = (first_occ.float() * val_seq).sum(dim=-1, keepdim=True)
        second_val = (second_occ.float() * val_seq).sum(dim=-1, keepdim=True)
        margin = torch.relu(first_val - second_val - torch.relu(self.skewer_margin))
        skewer_step = src_slider.unsqueeze(-1) * first_them.float() * second_them_any.any(dim=-1, keepdim=True).float() * margin

        # Scatter step tensors back to pairwise edge maps
        dst = self.ray_step_index.view(1, 8, 64, 7).expand(B, -1, -1, -1)
        src = torch.arange(64, device=piece_state.device).view(1, 1, 64, 1).expand(B, 8, 64, 7)
        pair = (src * 64 + dst).reshape(B, -1)

        def scatter_steps(step_score):
            out = torch.zeros(B, 64 * 64, device=piece_state.device, dtype=piece_state.dtype)
            out.scatter_add_(1, pair, step_score.reshape(B, -1))
            return out.view(B, 64, 64)

        edge_abs_pin = scatter_steps(abs_pin_step)
        edge_rel_pin = scatter_steps(rel_pin_step)
        edge_xray1   = scatter_steps(xray1_step)
        edge_skewer  = scatter_steps(skewer_step)
        edge_disc    = scatter_steps(disc_step)

        # Optional pinned-defender load
        if same_def is not None:
            pin_any = (edge_abs_pin + edge_rel_pin).sum(dim=1).clamp(min=0)  # (B,64)
            def_load = (same_def * target_value.unsqueeze(1)).sum(dim=-1)    # (B,64)
            pin_def_square = pin_any * def_load * torch.sigmoid(self.pin_def_scale)
        else:
            pin_def_square = torch.zeros(B, 64, device=piece_state.device, dtype=piece_state.dtype)

        edge_pin_def = pin_def_square.unsqueeze(1).expand(-1, 64, -1).transpose(1, 2)

        edge_maps = torch.stack(
            [edge_abs_pin, edge_rel_pin, edge_xray1, edge_skewer, edge_disc, edge_pin_def],
            dim=1
        )

        square_maps = torch.stack(
            [
                edge_abs_pin.sum(-1),
                edge_rel_pin.sum(-1),
                edge_xray1.sum(-1),
                edge_skewer.sum(-1),
                edge_disc.sum(-1),
                pin_def_square,
            ],
            dim=1
        )

        summary = torch.cat(
            [square_maps.mean(-1), square_maps.amax(-1)],
            dim=-1
        )

        return edge_maps, square_maps, summary
```

This sketch is intentionally simple, but it shows the essential p043 properties: exact ordered blocker logic, target-value conditioning, optional pinned-defender scoring, and no training-time Python loops over rays.

## Validation falsifiers and weak slices

### Falsifiers

The repo’s benchmark standard is explicit that ideas must report aggregate and slice-level behavior, and the p020 ablations file is explicit that a primitive should be dropped if its defining mechanism can be ablated away without losing the claimed slice lift. p043 should follow the same philosophy. fileciteturn40file0L3-L3 fileciteturn51file0L3-L3

The most important falsifiers are these:

- **No one-blocker x-ray ablation.** Replace \(C_1\) with either \(C_0\) only or a generic “same-line feature.” If `pin/skewer/discovered_attack` slices do not fall meaningfully, the primitive is not actually using x-ray-through-one-blocker logic.

- **Uniform target-value ablation.** Set all target weights equal. If performance is unchanged, the king/queen/rook value context is not load-bearing and the primitive has collapsed into generic ray geometry.

- **No pinned-defender ablation.** Zero the pinned-defender score while keeping pin geometry. If overload-style or king-pressure slices do not move, defender-load modeling is not buying anything.

- **Order-scramble ablation.** Preserve the set of ray squares but randomly permute step order inside each ray or degree-preservingly scramble the new relation channels. If the primitive survives that, it is not using blocker order in the way it claims.

A reasonable keep/drop rule is:

- no aggregate regression worse than **0.005 PR-AUC** versus the matched strong baseline;
- target-slice gain on declared sliding motifs of at least **0.01 PR-AUC** on either `pin`, `skewer`, or `discovered_attack`, or a clear win on `overload` / rook-file pressure with no meaningful aggregate loss;
- the primary falsifier should remove **most** of that slice lift;
- native i018 overhead should remain modest enough to justify keeping the feature.

Those thresholds are proposed acceptance criteria, not observed results.

### Weak-slice expectations

The repo’s benchmark metadata is favorable for this idea: the test split contains **10,814** `pin` examples, **8,984** `skewer` examples, **3,429** `discovered_attack` examples, and **3,722** `overload` examples, and the reporting rules explicitly require those motifs to be examined by slice. So p043 is targeting slices that are both semantically relevant and statistically large enough to evaluate. fileciteturn42file0L3-L3 fileciteturn40file0L3-L3

The expected strengths are:

- clear improvement on `pin`, `skewer`, and `discovered_attack`;
- likely improvement on `overload`, because pinned defender load should expose frozen defenders;
- some improvement on rook-file and king-line pressure positions, because one-blocker x-rays are common there;
- potentially stronger gains on middlegame sliding tactics than on purely local motifs. fileciteturn27file0L3-L3 fileciteturn42file0L3-L3

The expected weak slices are:

- `fork`, `mate_in_1`, `promotion`, and `underpromotion`, because they often do not depend on long slider geometry;
- quiet endgames and zugzwang-like studies, where tactical incidence is sparse and i018’s existing sheaf logic may already be the right tool;
- positions with **two or more** tactically relevant blockers on the same line, because p043 is intentionally centered on clear and one-blocker relations, not deep multi-blocker latent programs;
- legality-sensitive edge cases such as en passant unpins or partial-pin move legality, unless the primitive is explicitly fed legal-move constraints. Chessprogramming notes that absolute pins matter directly for legal move generation and that partial pins need direction information; p043 is primarily an evaluation primitive, not a full legal-move oracle. citeturn3view0turn4view1

So the right expectation is **sharp lift on a narrow family of slider motifs**, not universal improvement across all tactical categories.

### Validation and run plan

The run plan should have three phases.

**Phase one: exactness tests**

Before training, test p043 against brute-force geometry on curated FENs and random legal positions. The deterministic event builder should exactly match:
- clear-ray visibility;
- one-blocker x-ray destinations;
- absolute pin detection relative to king;
- discovered-attack detection through one own blocker;
- skewer ordering by front/back target values.

It should also be cross-checked against i018’s existing `king_ray_pin_candidate` relation on the subset of positions where only an absolute pin is present, and against i190’s ordered blocker extraction on first/second-occupied-square facts. i018 and i190 already contain the relevant reference geometry. fileciteturn25file0L3-L3 fileciteturn21file0L3-L3

**Phase two: standalone complementarity**

Train a standalone `pin_xray_skewer_head` and fuse it with i018 through the existing hybrid wrapper. Use the same protocol the repo used for hybrid i018 experiments: the canonical tagged split, seeds 42/43/44, base scale, 20 epochs, minimum 10 active epochs, balanced BCE, and validation-only threshold selection. This tells you whether p043 adds complementary signal before you perturb i018 internals. fileciteturn30file0L3-L3 fileciteturn33file0L3-L3 fileciteturn35file0L3-L3

**Phase three: native i018 integration**

If phase two is supportive, run:

- `i018_base`
- `i018_native_p043`
- `i018_native_p043_no_xray1`
- `i018_native_p043_uniform_values`
- `i018_native_p043_no_pin_def`
- `i018_native_p043_scrambled_relations`

Then produce the full benchmark package required by the repo:

- aggregate val/test PR-AUC and calibration;
- `slice_report_val.md` and `slice_report_test.md`;
- motif slices for `pin`, `skewer`, `discovered_attack`, `overload`, and `(if tagged) rook/open-file family`;
- cost reporting: params, MACs/FLOPs, throughput;
- a short “what this model can and cannot learn” summary. fileciteturn40file0L3-L3 fileciteturn27file0L3-L3

A compact first-pass success criterion would be:

- aggregate test PR-AUC within **0.005** of i018 or better;
- at least one of `pin`, `skewer`, or `discovered_attack` improves by **0.01 PR-AUC** or more;
- the `no_xray1` or `uniform_values` falsifier removes most of that motif lift;
- native overhead is small enough that the primitive is operationally credible.

### Open questions and limitations

This report proposes a design, not an observed win. I did **not** run p043 or its ablations, so any performance expectation here is a research hypothesis rather than a result. The main unresolved choice is whether the first production version should go directly into i018’s relation tensor or should first ship as a standalone head fused through the existing hybrid infrastructure. A second unresolved point is how much of pinned-defender load should reuse i018’s existing defense relations versus being recomputed inside the primitive. Those are implementation choices, not thesis blockers, but they should be decided explicitly before coding.