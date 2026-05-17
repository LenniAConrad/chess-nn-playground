# p045_king_zone_reply_pressure_primitive.md

Proposed primitive: **King-Zone Escape and Reply Pressure**, a board-local, side-to-move-asymmetric forcingness module for puzzle-binary classification.

## Thesis and motivation

The repository already has pieces of this idea, but not in the right shape. The `i018_oriented_tactical_sheaf_laplacian` model is board-only, side-to-move canonical, and already emits reporting-only diagnostics such as `king_ring_pressure`, `reply_pressure`, `defense_gap`, and `pin_pressure`. It also builds explicit king-zone relations like `us_attacks_empty_near_king` and `king_ray_pin_candidate`. Even so, the repo’s current reports show that i018 still sits at **0.861** overall test PR AUC, only **0.764** on the `mate_in_1` motif slice, and a **0.150** near-puzzle false-positive rate at matched recall 0.8. The more generic `i192_latent_reply_entropy_network` does not fix that slice either, landing at **0.759** on `mate_in_1` and **0.156** near-puzzle FP rate, while `i193_exchange_then_king_dual_stream` reaches **0.812** on `mate_in_1` and **0.128** near-puzzle FP rate at the same recall target. The benchmark standard in the repo also explicitly requires motif slices like `mate_in_1` and matched-recall near-puzzle behavior, so a specialized king-pressure primitive is directly aligned with the evaluation contract rather than being an auxiliary curiosity. fileciteturn13file0L3-L3 fileciteturn32file0L3-L3 fileciteturn24file0L3-L3 fileciteturn30file0L3-L3 fileciteturn12file0L3-L3

The repo also already contains two nearby but incomplete answers. `i248_rule_aware_tactical_head` promotes the TSDP concept, which enumerates legal moves and exact terminal outcomes, but its own notes say that legal-move evaluation is the dominant CPU cost in the fallback path and that the intended production path is precomputed parquet features. `p003_reply_channel_capacity` and `i192` both reason about reply structure in a more generic way, but neither is specialized to king-zone geometry, escape closure, checker lanes, or fake defense from pinned defenders. That makes room for a lighter primitive that is **denser than TSDP**, **more targeted than i192/p003**, and **more spatially useful for BT4**. fileciteturn21file0L3-L3 fileciteturn22file0L3-L3 fileciteturn31file0L3-L3 fileciteturn29file0L3-L3

The design principle comes straight from classical engine practice: king safety is not a flat count of “attacks near the king.” Strong schemes use **king-zone definition, square control, attack units, safe contact checks, nonlinear accumulation, and pin/x-ray structure** precisely because not all king-side pressure is equally forcing. That is the right conceptual template for this primitive. citeturn2view0turn4view0turn2view1turn3view1turn5view2

**Thesis.** The primitive should estimate how close the side to move is to collapsing the defender’s legal-reply families around the king, using only current-board structure: weighted king-zone control, escape-square closure, current and candidate checking lanes, fake defense from pinned defenders, and a cheap upper bound on reply capacity. It should avoid engine search, avoid full legal reply enumeration in the common path, and still be spatial enough to help both i018 hybrids and BT4-style students. This is a proposed design inference grounded in the repo’s benchmark gaps and in established king-safety heuristics. fileciteturn13file0L3-L3 fileciteturn24file0L3-L3 fileciteturn32file0L3-L3 citeturn2view0turn7view0

## King-zone definitions

The primitive should operate in a **mover-relative frame**. That is already how i018 thinks: its `BoardStateAdapter` rotates and color-swaps the board so the mover always sees their own side from the bottom, and its `TacticalIncidenceBuilder` already uses king-zone masks and oriented relations. Reusing that convention is important because this primitive is supposed to be **side-to-move asymmetric**, not color-invariant. fileciteturn13file0L3-L3

I would define three nested structures per defender king \(k_d\):

\[
Z_{\text{core}} = \{k_d\} \cup N_8(k_d)
\]

where \(N_8(k_d)\) is the king’s adjacent neighborhood. This is the “king field” in composition terms and the most important part of the zone, because it contains the king square itself and every immediate escape square. That emphasis is consistent with both king-safety practice and check-evasion logic. citeturn2view0turn7view0

\[
Z_{\text{front}} = \text{the three squares one extra rank beyond the front edge of } N_8(k_d)
\]

measured in the attacker-facing direction in the mover-relative frame. Chessprogramming explicitly notes that king zones are often taken as the squares the king can reach **plus two or three additional forward squares facing the enemy**. The point of \(Z_{\text{front}}\) is not that those squares are legal king moves right now, but that they measure whether the attack is **compressing future shelter and flight channels** rather than just touching the ring cosmetically. citeturn2view0

\[
E_{\text{adj}} = N_8(k_d)
\]

as the raw adjacent escape set, later partitioned into `live`, `sealed`, `blocked`, and optional `capturable` escape classes. This is where safe-mobility logic matters: what counts is not adjacency alone, but whether the square is actually safe under enemy control and physically available. Safe mobility and square control are both standard evaluation ideas for exactly this reason. citeturn5view3turn4view0

\[
L_{\text{ray}}(k_d) = \{\text{orthogonal and diagonal rays from } k_d\}
\]

used to detect current slider checks, one-blocker x-rays, interposition lanes, and pinned defenders. CPW’s `Checks and Pinned Pieces` and `X-ray Attacks` pages are especially relevant here: they show that, around a king, one-blocker ray structure is precisely where pins, discovered checks, and fake defenders live. citeturn3view1turn5view2

Finally, the primitive should distinguish **nominal defenders** from **free defenders**. A pinned piece may still appear in an attack map, but it is often a fake reply resource. Python-chess’s `pin()` and `is_pinned()` semantics are convenient for this exact distinction, because a pinned piece returns a directional mask and an unpinned piece returns the whole board, which is a clean way to gate whether a defender can actually support a relevant square. citeturn3view3turn3view5

## Feature equations

I would parameterize the primitive as a **small vector of interpretable intermediate terms**, not as one hand-written scalar from the start. A downstream learned head can combine them, but the primitive itself should surface the load-bearing geometry.

Use CPW-style attack-unit priors as initialization:

\[
u(P,N,B,R,Q) = (1,2,2,3,5)
\]

with extra bonuses for safe contact checks. That unit schedule matches the conventional king-safety idea that minor attacks are meaningful, rook attacks are stronger, queen attacks are strongest, and safe contact checks deserve extra emphasis. citeturn2view0

Define weighted attack and defense on a square \(q\) in the defender’s king zone as:

\[
A(q) = \sum_{p \in P_a} u_{t(p)} \mathbf{1}[p \rightsquigarrow q]
\]

\[
D_{\text{nom}}(q) = \sum_{p \in P_d} u_{t(p)} \mathbf{1}[p \rightsquigarrow q]
\]

\[
D_{\text{free}}(q) = \sum_{p \in P_d} u_{t(p)} \mathbf{1}[p \rightsquigarrow q] \mathbf{1}[q \in \text{pinMask}(p)]
\]

where \(a\) is the attacker, \(d\) is the defender, \(t(p)\) is piece type, and `pinMask(p)` is the legal support mask induced by an absolute pin. Because unpinned pieces return the whole board mask, the same equation handles both cases cleanly. This gives the primitive a principled way to discount “fake defense” instead of merely counting nominal defenders. citeturn3view5turn3view3turn2view1

The first core term is **zone pressure**:

\[
ZP =
\sum_{q \in Z_{\text{core}}} w_{\text{core}}(q)\,[A(q) - \lambda D_{\text{free}}(q)]_+
\;+\;
\eta \sum_{q \in Z_{\text{front}}} w_{\text{front}}(q)\,[A(q) - \lambda D_{\text{free}}(q)]_+
\]

with \([x]_+ = \max(0,x)\). A practical initialization is:

- \(w_{\text{core}}(k_d)=4\)
- \(w_{\text{core}}(\text{empty adjacent square})=3\)
- \(w_{\text{core}}(\text{occupied adjacent square})=2\)
- \(w_{\text{front}}=1\)

This respects the user’s requirement not to treat all king-zone attacks equally: pressure on the king square and open flight squares matters more than pressure on a merely nearby occupied square. The weights are only priors; the final fusion head should still learn how much each term matters. The underlying justification is standard king-safety weighting plus square-control logic. citeturn2view0turn4view0

The second core term is **fake-defense loss**:

\[
FD = \sum_{q \in Z_{\text{core}} \cup Z_{\text{front}}}
\big(D_{\text{nom}}(q) - D_{\text{free}}(q)\big)
\]

This is the value of counting pinned defenders separately. A zone that looks defended in a nominal attack map may actually be close to collapse if some of those defenders are pinned to the king or pinned off the relevant interposition line. CPW’s pin and x-ray discussions make that distinction explicit. citeturn2view1turn3view1turn5view2

The third core term is **escape closure**. Partition the adjacent king neighborhood into live, sealed, and blocked classes:

\[
E_{\text{live}} = \sum_{e \in E_{\text{adj}}}
\mathbf{1}[\text{empty}(e)]\,\mathbf{1}[A(e)=0]
\]

\[
E_{\text{sealed}} = \sum_{e \in E_{\text{adj}}}
\mathbf{1}[\text{empty}(e)]\,\mathbf{1}[A(e)>0]
\]

\[
E_{\text{blocked}} = \sum_{e \in E_{\text{adj}}}
\mathbf{1}[\text{defender occupies } e]
\]

and then

\[
EP = \alpha_1 E_{\text{sealed}} + \alpha_2 E_{\text{blocked}} - \alpha_3 E_{\text{live}}
\]

This is the primitive’s explicit answer to “escape-square availability.” It borrows the logic of safe mobility, but applies it only where it matters most: around the defender king. citeturn5view3turn4view0

The fourth core term is **current check severity plus candidate checking mass**. For current checks, let

\[
C_{\text{curr}} = \text{checkers}(d), \qquad n_{\check} = |C_{\text{curr}}|
\]

and define

\[
CP_{\text{curr}} = \beta_1 n_{\check} + \beta_2 \mathbf{1}[n_{\check} \ge 2]
\]

because double check deserves a separate jump. CPW’s check-evasion logic matters a lot here: under double check, only king moves remain; under single check, capture and interposition may also exist. citeturn7view0turn3view4

For **candidate checking moves**, do not enumerate all legal moves. Instead generate only **checking targets** for each compatible piece type:

\[
T_N(k_d), T_B(k_d), T_R(k_d), T_Q(k_d), T_P(k_d)
\]

and count only attacker moves that land on one of those targets:

\[
CC_t = \sum_{p \in P_a^t} |MoveMask(p) \cap T_t(k_d)|
\]

with a safe version

\[
CC_t^{\text{safe}} = \sum_{s \in T_t(k_d)}
\mathbf{1}[A(s) > D_{\text{free}}(s)]
\]

Then

\[
CP_{\text{cand}} = \sum_t \beta_t CC_t^{\text{safe}}
\]

This is targeted, not exhaustive. The checking-target sets are tiny compared with the full move list, so the primitive gets candidate checking pressure without a general legal-reply search. Python-chess’s `gives_check()`, `gives_checkmate()`, and `legal_moves` are useful for offline validation of this targeted generator, but not required in the main forward path. citeturn10view0turn10view1turn10view3

The fifth core term is **reply-capacity upper bound** for the defender. This is the key “reply pressure” idea. If the defender is currently in check, CPW tells us the legal reply families are limited: king move; capture of the checker; and, only for a slider check, interposition. Under double check, only king moves survive. That lets us estimate the size of the reply family cheaply:

\[
RC_{\text{king}} = E_{\text{live}}
\]

\[
RC_{\text{cap}} =
\begin{cases}
0, & n_{\check} \neq 1 \\
\min\{1,\ |\text{freeAttackers}_d(c)|\}, & n_{\check}=1,\ c\in C_{\text{curr}}
\end{cases}
\]

\[
RC_{\text{block}} =
\begin{cases}
0, & n_{\check}\neq 1 \text{ or checker is not a slider} \\
\sum_{s \in between(c,k_d)} \mathbf{1}[D_{\text{free}}(s)>0], & \text{otherwise}
\end{cases}
\]

\[
RC_{\text{active}} = RC_{\text{king}} + RC_{\text{cap}} + RC_{\text{block}}
\]

When the defender is **not** currently in check, use the same family logic on the **best safe candidate check** rather than on every legal move. That gives a projected \(RC_{\text{proj}}\) without moving into full reply enumeration. citeturn7view0turn3view2turn9view0

The final directional primitive score is then:

\[
KZRP_{a \to d}
=
ZP + EP + \gamma FD + CP_{\text{curr}} + CP_{\text{cand}}
-
\delta \log(1 + RC)
\]

where \(RC\) is \(RC_{\text{active}}\) if the defender is already in check, otherwise the projected reply-capacity term from the strongest safe candidate check. Finally expose the side-to-move asymmetry explicitly:

\[
KZRP_{\Delta}
=
KZRP_{\text{stm} \to \text{opp}}
-
KZRP_{\text{opp} \to \text{stm}}
\]

That difference is important. A position with mutual king danger is not the same as a position where **the side to move** has a forcing attack. The primitive should model that asymmetry directly rather than expecting the trunk to rediscover it. That recommendation is especially natural in i018 because the model already uses side-to-move canonicalization. fileciteturn13file0L3-L3

## Inputs outputs and integration

**Inputs.** The primitive should consume only the current board state. The preferred path is `simple_18`, because both i018 and several primitive heads in the repo already use that contract, and the repo is explicit that CRTK metadata, source labels, verification flags, and engine numbers are reporting-only rather than model inputs. For BT4 students on `lc0_bt4_112`, the primitive should decode only the current-board occupancy and side-to-move portion needed for attack geometry, ignoring history planes in the first version. fileciteturn8file0L3-L3 fileciteturn13file0L3-L3 fileciteturn34file0L3-L3

**Outputs.** I would expose both a compact vector and spatial maps:

- a global vector with terms such as `kzrp_stm`, `kzrp_opp`, `kzrp_asym`, `zone_pressure`, `fake_defense_loss`, `live_escapes`, `sealed_escapes`, `blocked_escapes`, `current_checkers`, `safe_contact_checks`, `safe_slider_checks`, `active_reply_capacity`, and `projected_reply_capacity`;
- four spatial maps over \(8 \times 8\): `zone_weight_map`, `net_control_map`, `escape_state_map`, and `pin_line_map`;
- an optional `primitive_delta` logit if the primitive is used as an additive head.

That output shape is deliberate: i018 benefits mostly from the vector; BT4 benefits from the maps.

**Integration with i018.** The lowest-risk repo-native path is an additive gated hybrid, because the repository already has a common pattern for “i018 plus primitive” where the final logit is `sheaf_logit + sigmoid(gate) * prim_logit`, and concrete configs already follow that scheme. So the first implementation should be a new primitive module plus `configs/hybrid_i018_plus_primitive/i018_plus_p045.yaml`, not a rewrite of the sheaf trunk. If the hybrid lifts the target slices, a second pass can append raw KZRP vector terms to `TacticalReadout` inside i018 itself. fileciteturn15file0L3-L3 fileciteturn33file0L3-L3

**Integration with the BT4 student.** For the plain `lc0_bt4_classifier`, the least confounded experiment is **plane augmentation**: concatenate the four primitive maps as extra stem channels and inject the global vector into the value head by broadcast or a tiny side MLP. That changes the input signal but keeps the BT4 tower intact. For a cleaner architecture study after that, implement `bt4_mixers/king_zone_reply_pressure.py` and test it under the repo’s `bt4_primitive_mixer`, which is specifically built so that the only changed component is the per-block spatial mixer with the fixed `(B, C, 8, 8) -> (B, C, 8, 8)` contract. The scaffold script and mixer registry already support that workflow. fileciteturn35file0L3-L3 fileciteturn16file0L3-L3 fileciteturn17file0L3-L3 fileciteturn18file0L3-L3

## Metrics falsifiers and speed

**Metrics affected.** The primary metric target should be the `mate_in_1` slice, because the repo’s strongest current models still leave visible headroom there and i018 is notably weak on it. The second primary target should be matched-recall near-puzzle rejection, because that is exactly where a focused “reply pressure” feature should help: it should separate sharp king attacks from noisy near-king activity that still leaves too many viable replies. Secondary metrics are the `pin` motif, `discovered_attack`, the `equal` eval bucket, and `crtk_to_move` asymmetry. Concretely, i018 currently reports `mate_in_1 = 0.764` PR AUC, `pin = 0.837`, `discovered_attack = 0.805`, `equal = 0.799`, and near-puzzle FP rate `0.150` at recall 0.8; those are the benchmark lines to beat without sacrificing overall PR AUC too much. fileciteturn32file0L3-L3 fileciteturn24file0L3-L3

A realistic promotion target for the i018 hybrid is:

- at least **+0.020** absolute PR AUC on `mate_in_1` over i018;
- at least **-0.010** absolute near-puzzle FP rate at recall 0.8 over i018;
- no more than **-0.003** overall PR AUC regression, with a preference for flat-to-up aggregate quality.

Those thresholds are proposal-level targets derived from the current repo gaps: they push i018 materially toward `i193` on the intended slices without pretending this primitive alone should solve the whole benchmark. fileciteturn24file0L3-L3 fileciteturn32file0L3-L3

**Falsifiers.** I would use structure-preserving falsifiers, not randomization.

A **sealed-escape monotonicity falsifier** should take matched board pairs where the attacker additionally seals one empty adjacent escape square while leaving defender resources no better than before. `KZRP_{a \to d}` should not go down. If it does, the escape term is not load-bearing.

A **fake-defense falsifier** should hold king-zone attack units roughly constant while removing an absolute pin on a key defender. The pinned version should score higher pressure and lower free defense. If not, the primitive is only counting nominal attacks and defenders.

A **reply-family discrimination falsifier** should compare same-zone-attack positions where one defender has many legal-family resources against checks—king escapes, capture of checker, interposition—and the other does not. The primitive must rank the wide-reply position lower. If not, it is not really a reply-pressure primitive.

A **side-to-move falsifier** should flip the side to move on the same board when legal to do so and compare \(KZRP_{\Delta}\). If the signal hardly changes, the primitive failed the asymmetry requirement.

A **cosmetic-attack falsifier** should use near-puzzle negatives with lots of king-zone touches but intact defender reply capacity. If those score almost as high as genuine mating-net positives, the primitive has collapsed back to naïve attack counting.

**Speed considerations.** The common path should be **bitboard attack geometry only**: king-neighborhood masks, pawn/knight masks, slider rays, in-between tables, x-ray pinners, and targeted check-target masks. CPW’s attacked-square, in-between, x-ray, and checks-and-pins pages all point toward that style of implementation, and python-chess can be kept for testing rather than for every forward pass. Unlike TSDP, the primitive does not need to walk the full legal move list in its main path. That makes it much more suitable as a dense training-time feature and especially as BT4 spatial maps. The preferred production path is still **offline precompute into parquet or cached tensors**, because the repo is already using that idea for rule-derived primitives and because it removes CPU variance from training entirely. citeturn9view0turn3view1turn5view2turn3view2turn10view0 fileciteturn21file0L3-L3 fileciteturn22file0L3-L3

## Implementation sketch and experiment plan

A minimal implementation should live at a new module such as `src/chess_nn_playground/models/primitives/king_zone_reply_pressure.py`, with a matching hybrid config `configs/hybrid_i018_plus_primitive/i018_plus_p045.yaml` and, if the BT4 mixer variant is attempted, `src/chess_nn_playground/models/architecture/bt4_mixers/king_zone_reply_pressure.py`. That file naming matches the repo’s existing primitive and hybrid patterns. fileciteturn23file0L3-L3 fileciteturn33file0L3-L3 fileciteturn18file0L3-L3

```python
def compute_kzrp(board_tensor):
    board = decode_current_board(board_tensor)          # simple_18 or lc0 current-board slice
    stm, opp = side_to_move(board), not side_to_move(board)

    features = {}
    maps = {}

    for atk, dfn, prefix in [(stm, opp, "stm"), (opp, stm, "opp")]:
        king_sq = king_square(board, dfn)
        z_core, z_front = king_zone_masks(board, atk, dfn, king_sq)
        attack_units = weighted_attacks(board, atk, z_core | z_front)
        defense_nom = weighted_attacks(board, dfn, z_core | z_front)
        pin_masks = defender_pin_masks(board, dfn)
        defense_free = discount_pinned_defense(defense_nom, pin_masks, z_core | z_front)

        escapes = classify_adjacent_escapes(board, atk, dfn, king_sq, attack_units)
        current_checkers = find_current_checkers(board, atk, king_sq)
        candidate_checks = targeted_check_candidates(board, atk, king_sq)
        reply_capacity = estimate_reply_capacity(
            board, atk, dfn, king_sq, current_checkers, candidate_checks, defense_free
        )

        score = combine_zone_escape_check_pin_terms(
            attack_units, defense_free, defense_nom, escapes, current_checkers, candidate_checks, reply_capacity
        )

        features[prefix] = summarize(score, attack_units, defense_nom, defense_free, escapes, current_checkers, reply_capacity)
        maps[prefix] = make_spatial_maps(z_core, z_front, attack_units, defense_free, escapes, pin_masks)

    features["kzrp_asym"] = features["stm"]["kzrp"] - features["opp"]["kzrp"]
    return {"primitive_features": flatten(features), "primitive_maps": maps}
```

The experiment plan should be staged so the primitive proves its signal before it is asked to carry architecture complexity.

- **Feature audit first.** Precompute the p045 vector and maps for the canonical tagged splits, and test whether they separate `mate_in_1` and near-puzzle negatives at all with a linear or shallow-MLP probe. If the raw signal is weak, do not proceed to full hybrids. This follows the repo’s emphasis on slice-level evidence rather than aggregate-only optimism. fileciteturn12file0L3-L3

- **i018 hybrid next.** Run `i018_plus_p045` with the existing gated-logit fusion contract, exactly because it minimizes confounds and is already the repo’s established hybrid pattern. Then ablate: no candidate-check term, no fake-defense term, no asymmetry term, no projected reply-capacity term. The first question is whether the primitive is useful at all on top of i018. fileciteturn15file0L3-L3 fileciteturn33file0L3-L3

- **BT4 plane augmentation after that.** Add only the spatial maps to the BT4 student first. If that works, then test a second variant that also injects the global vector into the value head. Only after those two should the repo’s `bt4_primitive_mixer` harness be used for a mixer-native p045 study. That keeps the progression from lower-risk to higher-risk changes. fileciteturn35file0L3-L3 fileciteturn16file0L3-L3

- **Compare against the exact rule baseline on the target slice.** Not as a replacement, but as calibration. On the `mate_in_1` slice, compare p045’s lift to `i248_rule_aware_tactical_head`, because i248 is the repo’s explicit “exact terminality” baseline for this kind of problem. If p045 closes part of that gap at much lower cost and with denser spatial output, it has earned its place. fileciteturn21file0L3-L3 fileciteturn22file0L3-L3

- **Use the repository’s paper-grade benchmark protocol.** The run contract should stay on the canonical tagged splits, with slice reports, matched-recall false-positive analysis, validation-only threshold selection, and three seeds where the result is promising enough to promote. The benchmark-reporting standard is already explicit about that. fileciteturn12file0L3-L3 fileciteturn15file0L3-L3

The key experimental question is not “does p045 help aggregate PR AUC a little?” It is narrower and more valuable: **does a specialized, board-local estimate of king-zone escape and reply pressure improve exactly the slices where the repo still leaks tactical false positives and under-detects immediate king collapses, without paying the runtime cost of a legal-move terminal oracle?** That is the right bar for this primitive.