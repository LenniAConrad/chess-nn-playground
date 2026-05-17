# p042 Candidate Move Forcedness Primitive

[Download the markdown file](sandbox:/mnt/data/p042_candidate_move_forcedness_primitive.md)

## Thesis

p042 should be a **board-only latent move-scoring primitive**, not a policy head trained on puzzle solution moves. In this repo, the closest current reference point, i018, is already board-only: it canonicalizes side to move, builds a typed tactical incidence tensor over the 64 squares, and reads out pooled relation energies, node statistics, and triad-defect summaries. The repo’s benchmark standard also explicitly says CRTK metadata is for benchmarking and error analysis only, and must not be used as neural-network input. That means the most promising missing signal is not more metadata or more static board descriptors, but a new **candidate-move consequence layer** that turns tactical tension into a forcing-line prior. fileciteturn8file0L3-L3 fileciteturn9file0L3-L3 fileciteturn19file0L3-L3

That framing is consistent with the broader literature. AlphaZero encodes chess state from rule-based spatial planes, outputs move probabilities for actions plus a scalar value, and treats chess actions as the legal destinations available from the board; recent work on chess “state affordances” likewise treats downstream legal-move distributions as a meaningful semantic summary of board state. p042 should borrow that move-level interface, but replace search-based targets with deterministic forcing descriptors derived from the board and the immediate post-move position. citeturn9view0turn11view0

The practical thesis is therefore:

```text
simple_18 board
  -> batched candidate enumerator
  -> batched post-move apply
  -> forcing descriptors per candidate
  -> move-level forcedness logits
  -> top-k / category pooling
  -> standalone p042 logit or i018 fusion
```

The goal is not “find the best move” in the engine sense. The goal is “estimate whether this board contains one or a few unusually coercive candidate moves,” then pool that signal into a board-level feature vector that helps puzzle detection, near-puzzle rejection, and weak tactical slices.

## Move Edge Construction

The repo already has the right substrate for a fast implementation. `compute_legal_move_graph(board)` builds a batched `(B, 64, 64)` move adjacency from `simple_18` using precomputed knight, king, pawn, and slider geometry plus a `(64, 64, 64)` `between` tensor for blocker resolution. Other legal-move primitives in the repo already consume rule-derived move tensors as **stop-gradient** structures and aggregate them with dense batched matrix operators, rather than Python loops. p042 should follow that same contract. fileciteturn23file0L3-L3 fileciteturn24file0L3-L3 fileciteturn20file0L3-L3 fileciteturn22file0L3-L3

I would split move construction into **two synchronized surfaces**.

The first surface is an **executable candidate surface** used for scoring actual moves. The second is a **threat surface** used only for after-move attack and pressure features. That split matters because the repo’s current move-graph helper intentionally allows pawn-diagonal “capture” edges as threat geometry even when the destination is empty; that is useful for threat maps, but too noisy for ranking actual candidate moves. p042 should therefore reuse the existing geometry tables while separating “executable move” semantics from “attacked-square” semantics. fileciteturn24file0L3-L3

For the executable surface, I recommend **legal moves** as the default representation, with pseudo-legal moves retained only as an ablation. The Laws of Chess make the required filters explicit: a move is legal only when the conditions of Articles 3.1–3.9 are satisfied; no move may expose or leave the mover’s king in check; pawn pushes require empty forward squares; diagonal pawn captures require an opponent piece or the one-ply en-passant condition; promotion expands into queen, rook, bishop, or knight choices; and castling requires rights, empty transit squares, and an unattacked king path. citeturn7view5turn7view6turn7view7turn12view0turn12view1turn12view2

A fast batched legalizer can still stay analytic. The four filters that matter are king-destination safety, pin-ray compliance, check-evasion filtering, and castling / en-passant special cases. This is conceptually the same logic exposed by `python-chess`: `is_attacked_by()` and `attackers()` support x-ray occupancy control, `pin()` identifies absolute pins and their direction, `checkers()` identifies current checking pieces, and `gives_check()` / `gives_checkmate()` treat move consequences as one-ply rule probes rather than as search. p042 should implement the same ideas with repo-native tensors, not with Python object boards. citeturn10view0turn10view2turn7view3turn10view3

The packed move tensor should contain one row per candidate move:
`(src, dst, mover_piece_type, captured_piece_type, promo_piece_type, move_type, ray_direction, flags)`.

Useful flags are `is_capture`, `is_promotion`, `is_underpromotion`, `is_castle`, `is_en_passant`, `is_check_evasion`, `is_pinned_source`, and `is_check_seed`. That gives the scorer exact move geometry without any engine search or external labels.

## Feature Definitions

The feature set should be **small, deterministic, and consequence-oriented**. Every feature below can be computed from the current board `x`, the post-move board `x^m`, and attack / defense summaries. The underlying notions are standard: FIDE defines check, legality, castling, en passant, pawn movement, and promotion; `python-chess` documents attacked squares, attackers, pins, and checking pieces in the same rule-based language. citeturn7view5turn7view6turn7view7turn12view0turn12view1turn12view2turn10view0turn10view2turn7view3turn10view3

I would define the per-move descriptor vector `f(m)` around the following families:

| Family | Proposed statistic | Why it matters |
|---|---|---|
| Check pressure | `check(m)`, `double_check(m)`, `checkers_after(m)` | Directly measures coercion on the opponent king |
| Capture pressure | victim value, target-square defender slack, SEE-lite gain | Separates forcing captures from loose or losing captures |
| Promotion pressure | promotion bit, underpromotion bit, promoted-piece value | Promotions often create immediate forcing states |
| Threat creation | value-weighted sum of newly attacked enemy pieces that are now loose or under-defended | Captures “threat” without search |
| Discovered attack | new slider attack opened on king, queen, rook, or loose minor after the source square vacates | Critical for line-opening tactics |
| Overload | weighted count of enemy defenders now covering more than one urgent asset | Measures whether one reply cannot solve all threats |
| King-space compression | reduction in legal king escapes, increase in attacked king-ring squares, interposition scarcity | Approximates forcing pressure even when the move is not an immediate check |
| Evasion scarcity | exact number of legal evasions only when `m` checks | Cheap one-ply reply compression, not search |
| Fork multiplicity | number and value of separate high-value targets attacked by the moved piece after `m` | Measures multi-threat creation |
| Mobility shock | local drop in opponent mobility near king sector or threatened assets | Another reply-space surrogate |

Two implementation details are especially important.

First, **SEE-lite** should remain local and rule-based. Use a fixed heuristic value scale such as pawn = 1, knight/bishop = 3, rook = 5, queen = 9, plus post-move attacker / defender counts on the destination square. Do not recurse into an exchange tree. The point is not exact exchange search; the point is to distinguish obviously forcing captures from obviously self-destructive ones.

Second, **overload** should be defined as a defender-incidence statistic, not as a vague motif label. Form a critical-asset set `C_m` after move `m`—at minimum the opponent king, the opponent king ring, the destination square, and the top few newly threatened valuables. Then define `load_m(d)` as the number of critical assets defended by enemy defender `d`. If one defender must cover multiple urgent assets created by the move, the defender is overloaded in the literal “one move cannot answer all demands” sense.

A compact notation that keeps the implementation disciplined is:

```text
Loose(q; x)   = 1[A_us(q; x) > A_them(q; x)]
NewThreat(m)  = Σ_q value(q) · 1[A_us(q; x^m) > A_us(q; x)] · Loose(q; x^m)
EscapeRed(m)  = max(0, |Ksafe_opp(x)| - |Ksafe_opp(x^m)|) / 8
Discover(m)   = value-weighted new slider attacks caused by vacating src(m)
Overload(m)   = Σ_d ReLU(load_m(d) - 1)
SEE_lite(m)   = cap_value(m)
                - λ1 · defended_dst_by_them(m) · mover_value(m)
                + λ2 · defended_dst_by_us(m)
```

The essential modeling choice is that **none of this requires solution-move labels**. The board label supervises the primitive only through the final pooled board representation.

## Forcedness Equations

A good default is a **latent per-move scorer** rather than a single hand-written scalar formula. AlphaZero shows that board-only models can map rule-based state encodings to action-level outputs, and the more recent affordance paper reinforces the idea that legal-move distributions themselves carry semantic content. p042 should exploit that interface, but with a narrower target: not “best move,” but “most forcing move candidate.” citeturn9view0turn11view0

I would therefore use:

```text
e_m = concat(
        embed(src),
        embed(dst),
        embed(piece_type),
        embed(move_type),
        direct_flags(m),
        f(m)
      )

s_m = MLP(LayerNorm(e_m))      # forcedness logit
p_m = sigmoid(s_m)             # optional interpretability view
```

where `direct_flags(m)` contains move-class indicators such as capture, promotion, underpromotion, castling, en passant, checking move, and pinned-source move.

For checking moves only, an exact one-ply reply feature is worth keeping:

```text
E_m = number of legal evasions in x^m, if check(m)=1
      0 otherwise

EvasionScarcity(m) = 1 - log(1 + E_m) / log(1 + E_cap)
```

This is still not engine search. It is a one-ply legality count on the post-move board, and it is only computed for the relatively small subset of moves that actually check the king.

The full move score can stay learned:

```text
ForcednessLogit(m) = s_m
ForcednessProb(m)  = sigmoid(s_m)
```

with the deterministic channels acting as structured inputs rather than as a frozen formula.

## Pooling Method

The board-level pool should be **top-k by default**, with category maxima alongside it. A plain mean over all legal moves will blur exactly the signal p042 is meant to isolate.

My recommended pool is:

```text
T_k(x) = TopK_k({s_m})
π_m    = masked_softmax(s_m / τ)

z_top  = concat(
           e_(1),                       # top-1 move descriptor
           mean_{m in T_k} e_m,
           max_{m in M(x)} e_m
         )

z_dist = concat(
           s_(1),                       # top-1 logit
           s_(1) - s_(2),               # winner margin
           Σ_{m in T_k} π_m,            # top-k mass
           H(π),                        # move entropy
           log(1 + |M(x)|)              # move-count context
         )

z_cat  = concat(
           max check(m),
           max capture(m),
           max promotion(m),
           max NewThreat(m),
           max Discover(m),
           max Overload(m),
           max EscapeRed(m),
           max EvasionScarcity(m)
         )

z_p042 = concat(z_top, z_dist, z_cat)
```

`k = 4` is the most sensible default, because many tactical positions have one truly forcing move plus a small halo of near-misses, but this should be ablated with `k ∈ {1, 2, 4, 8}`. The most important pooled diagnostics are the **top-1 / top-2 gap** and the **entropy**. Low entropy and a large gap mean the board contains a singular forcing candidate; high entropy means either several roughly equivalent tactical shots or no particularly forcing move at all.

The standalone primitive logit is then simply:

```text
logit_p042 = Head(z_p042)
```

and the primitive should export scalar diagnostics such as `forcedness_top1`, `forcedness_gap12`, `forcedness_entropy`, `forcedness_check_peak`, `forcedness_overload_peak`, and `forcedness_topk_mass` into the predictions parquet, matching the repo’s existing one-column-per-diagnostic pattern. fileciteturn22file0L3-L3

## Integration Plan

The repo’s primitive ledger reserves `p036` through `p046` for the current backlog, so `p042` is a coherent identifier for this proposal. The easiest first integration is a standalone primitive module plus a hybrid config, because the repo already has an `oriented_sheaf_plus_primitive` architecture that runs i018 and a primitive in parallel and fuses them as `sheaf_logit + sigmoid(gate) * primitive_logit`; the common hybrid config explicitly frames this as the way to test whether a primitive adds non-overlapping signal to i018. fileciteturn15file0L3-L3 fileciteturn26file0L3-L3 fileciteturn10file0L3-L3 fileciteturn11file0L3-L3

The first implementation pass should therefore add:

- `src/chess_nn_playground/models/primitives/candidate_move_forcedness.py`
- registry entry `candidate_move_forcedness`
- `configs/hybrid_i018_plus_primitive/i018_plus_p042.yaml`
- focused tests for move enumeration, legality masking, and feature correctness
- the usual idea-registry documents under `ideas/registry/p042_candidate_move_forcedness_primitive/`

That gives an immediate apples-to-apples comparison against existing i018 hybrids.

If p042 shows signal in hybrid form, the second pass should integrate `z_p042` more deeply into i018’s readout pathway. i018’s `TacticalReadout` already concatenates pooled node embeddings, relation-energy summaries, triad statistics, and board statistics. A projected `z_p042` vector fits naturally into that same readout stage. fileciteturn8file0L3-L3

## Evaluation Plan

**Efficient implementation sketch.** The implementation should stay batched and preserve the repo’s stop-gradient contract for rule-derived tensors. A straightforward first pass is:

```python
def forward(board):
    with torch.no_grad():
        geom = precomputed_geometry()
        exec_edges, threat_edges, move_meta = enumerate_candidates(board, geom, legal=True)
        moves = pack_candidates(exec_edges, move_meta)      # [B, M, fields]
        board_m = apply_moves_batched(board, moves)         # [B, M, C, 8, 8] or flat planes
        atk_before = attack_summaries(board, threat_only=True)
        atk_after  = attack_summaries(board_m, threat_only=True)
        reply_feats = check_reply_stats(board_m, only_when_check=True)

    feat_m = build_forcedness_features(board, board_m, moves, atk_before, atk_after, reply_feats)
    score_m = move_scorer(feat_m, moves)
    z_p042 = pool_topk(score_m, feat_m, moves.mask)
    logits = head(z_p042)
    return diagnostics_and_logits(logits, score_m, z_p042)
```

This is realistic in this repo because the existing legal-move primitives already materialize dense `(B, 64, 64)` masks, use batched `einsum` and `bmm`, and explicitly keep the rule graph inside `torch.no_grad()`. p031 also already identifies dense batched materialization as the current baseline and treats sparse CSR as a later optimization, not as a prerequisite. p042 therefore does **not** need a Python-loop move generator and does **not** need engine search. It needs a new batched post-move consequence layer. fileciteturn20file0L3-L3 fileciteturn22file0L3-L3 fileciteturn23file0L3-L3 fileciteturn24file0L3-L3

If memory becomes tight, the first optimization should be candidate-major compaction rather than exotic sparse kernels: pack only active candidates per board, keep replicated planes in half precision, and compute expensive reply features only for checking candidates or for the top provisional move tranche. On 8×8 chess boards, that is usually the highest-value optimization.

**Falsifiers.** The falsifiers should be structural rather than random, because the prompt explicitly rejects a random-move falsifier and because random corruption mostly proves trivial semantics loss. The important falsifiers are these:

- **Pseudo-legal instead of legal candidates.** If results match, exact legality is not load-bearing.
- **Move metadata only.** Keep `(src, dst, piece_type, move_type, flags)` but remove all post-move consequence features. If results match, the primitive is not actually measuring forcedness.
- **Mean-pool over all moves.** Replace top-k pooling with uniform averaging. If results match, candidate concentration is not load-bearing.
- **No reply-pressure features.** Remove `EscapeRed` and `EvasionScarcity`. If results match, the primitive is not really learning reply compression.
- **No overload / discovered features.** If results match, the primitive is “checks and captures only” and not adding deeper tactical structure.

**Expected slice benefits.** The repo’s benchmark standard says ideas are only interesting if they improve overall quality without hurting hard slices, or if they improve declared target slices such as `hard`, `very_hard`, `endgame`, `pin`, `overload`, `mate_in_1`, or `promotion` within a documented overall tolerance. It also requires reporting by difficulty, phase, eval bucket, tactic motifs, and tag families, with special attention to false-positive and false-negative behavior inside fine-label rows. That makes the expected p042 target slices fairly clear: `hard`, `very_hard`, `THREAT`, `fork`, `pin`, `skewer`, `hanging`, `overload`, `discovered_attack`, `mate_in_1`, `promotion`, and `underpromotion`, plus better near-puzzle rejection through lower tactical false-positive rates. fileciteturn19file0L3-L3 fileciteturn18file0L3-L3

Quiet opening positions and `(none)` motif slices may show little or no gain, because p042 is intentionally biased toward forcing-move geometry. That is acceptable if the declared tactical slices improve and overall PR-AUC remains stable, because the project’s own benchmark rules explicitly recognize target-slice lifts as a valid reason to keep an architecture. fileciteturn19file0L3-L3

**Experiment matrix.** The run protocol should follow the repo standard: canonical tagged splits, convergence-budget training, validation-only thresholding, slice reports on val and test, and promotion-grade comparisons over seeds `42`, `43`, and `44`. The current i018 hybrid launcher uses that same seed discipline and interprets `+0.005` test PR-AUC over baseline as a real lift while `|delta| < 0.002` is a wash. p042 should adopt that same interpretation rule. fileciteturn19file0L3-L3 fileciteturn12file0L3-L3 fileciteturn10file0L3-L3

| Run set | Variant | What changes | What it tests |
|---|---|---|---|
| Baselines | `i018` | Existing oriented tactical sheaf only | Reference point |
| Baselines | `p042` standalone | Candidate forcedness primitive only | Whether move-level forcing alone has puzzle signal |
| Mainline | `i018 + p042` | Existing gated-logit hybrid path | Whether p042 adds non-overlapping signal |
| Move surface | legal | Exact legal candidates | Preferred formulation |
| Move surface | pseudo-legal | Skip exact legality filter | Whether legality matters beyond geometry |
| Features | checks + captures only | Keep only obvious forcing features | Lower bound |
| Features | + promotions + discovered | Add line-opening and promotion signals | Mid-strength model |
| Features | + overload + escape + evasions | Full forcing descriptor stack | Full model |
| Pooling | mean-all | Uniform pool over moves | Anti-top-k falsifier |
| Pooling | top-1 | Only best candidate | Tests single-move concentration |
| Pooling | top-4 | Default | Expected sweet spot |
| Pooling | top-8 | Broader move mass | Whether several candidates matter |
| Consequence ablation | metadata-only moves | No post-move board apply | Syntax-only falsifier |
| Consequence ablation | no reply-pressure | Remove escape / evasion features | Tests reply-space compression |
| Consequence ablation | no overload / discovered | Remove deeper tactical channels | Tests whether p042 is just checks and captures |
| Efficiency | dense exact | Full batched apply | Initial implementation |
| Efficiency | partial delta | Recompute only affected rays or checking candidates | Profiling optimization |

**Open questions / limitations.** Two things are genuinely unresolved until implementation. First, exact legal-move filtering including castling and en passant is the principled default, but a pseudo-legal variant may end up delivering most of the gain at substantially lower engineering cost. Second, overload and discovered-attack features are conceptually attractive, but they must beat the “checks + captures + promotions + king-escape reduction” core before they deserve long-term complexity. If they do not survive the structured falsifiers above, they should be cut.