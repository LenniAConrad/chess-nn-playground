# Codex Handoff Packet: Hall-Defect Zeta Operator

## 1. File Metadata

- **Filename:** `chess_nn_research_2026-04-28_0802_tuesday_new_york_hall_defect_zeta.md`
- **Generated:** 2026-04-28 08:02 new_york
- **Weekday:** Tuesday
- **Packet type:** Deep Research handoff packet
- **Idea name:** Hall-Defect Zeta Operator, abbreviated **HDZ**
- **Domain:** finite algebra for chess puzzle classification
- **Model contract:** current-board tensor only; one scalar logit out
- **Fine-label convention in this packet:** `0 = ordinary negative`, `1 = near-puzzle hard negative`, `2 = true puzzle positive`
- **Binary target mapping:** `y = 1[fine_label == 2]`
- **Forbidden inference inputs:** engine evaluations, principal variations, node counts, mate scores, best moves, puzzle source labels, verification metadata, source identity, or any post-position search artifact

## 2. Executive Selection

Selected concept: **Hall-Defect Zeta Operator**, a deterministic finite-algebraic board operator that computes local overload spectra in a pin-filtered defense relation.

The core claim is simple: many true tactical puzzles are positions where the defending side has too many local obligations for the number of independently usable defenders. Near-puzzles often look similar but contain one extra legal defender, one unpinned defender, or one escape/recapture resource. HDZ tries to expose that difference before the neural network sees the board.

HDZ builds, for each side and anchor square, a finite bipartite relation:

\[
R_{c,t} \subseteq \Omega_{c,t} \times P_c,
\]

where:

- \(\Omega_{c,t}\) is a bounded set of local tactical obligations near anchor square \(t\), such as recapture squares, king-zone squares, and line-interposition squares.
- \(P_c\) is the set of pieces belonging to side \(c\).
- \(R_{c,t}(o,p)=1\) when piece \(p\) can legally and effectively answer obligation \(o\), after removing defenses that are invalidated by pins or king exposure.

For each small subset \(U \subseteq \Omega_{c,t}\), the operator computes the **Hall defect**:

\[
\delta(U)=\max(0, |U|-|N(U)|),
\]

where \(N(U)\) is the set of effective defenders touching at least one obligation in \(U\). Positive defect means that the local obligation set has fewer independent responders than tasks. The operator reports truncated spectra for \(|U|\in\{1,2,3,4\}\), producing a dense tensor that is fed to a small neural classifier.

The finite-algebraic object is the Boolean incidence algebra of the obligation lattice \(2^{\Omega_{c,t}}\). The zeta step aggregates defender neighborhoods over subsets of obligations. This is not a positional encoding, not an engine proxy, and not a search trace. It is a current-board algebraic measurement of overloaded legal defense.

## 3. Data Contract

### Input

Minimum board tensor:

\[
X \in \{0,1\}^{8\times 8\times 13}.
\]

Channels:

1. White pawn
2. White knight
3. White bishop
4. White rook
5. White queen
6. White king
7. Black pawn
8. Black knight
9. Black bishop
10. Black rook
11. Black queen
12. Black king
13. Side-to-move broadcast plane, with `1` for White to move and `0` for Black to move

If the data pipeline uses a richer current-board tensor, HDZ may read only current-position state. It must not read search products or source metadata. Castling and en-passant channels may be present but should be ignored in the first implementation unless the benchmark specifically contains positions where those rules are material to the label.

### Output

The network returns exactly one scalar logit:

\[
z \in \mathbb{R}.
\]

Training probability:

\[
\hat p = \sigma(z),
\]

interpreted as the probability that the current board is a true puzzle.

### Fine-label mapping

Use fine labels for target mapping and diagnostics only:

| Fine label | Meaning in this packet | Binary target | Diagnostic role |
|---:|---|---:|---|
| `0` | ordinary negative | `0` | easy-negative calibration |
| `1` | near-puzzle hard negative | `0` | main false-positive stress test |
| `2` | true puzzle | `1` | positive class |

Do not give the model the fine label. Do not train auxiliary heads that predict puzzle source, verification status, engine identity, or any field that is not part of the current board.

## 4. Algebraic Research Map

HDZ sits in the finite incidence-algebra family. The board is converted into finite sets and relations, then summarized through a zeta aggregation over a Boolean lattice.

| Layer | Finite object | Chess interpretation | Tensor result |
|---|---|---|---|
| Board parse | finite set of occupied squares | pieces and colors | piece list \(P_W,P_B\) |
| Contact relation | finite directed relation on squares | attacks, defenses, line contacts | raw legal contact table |
| Pin filter | finite predicate on piece-square pairs | defense removed if moving exposes own king | effective defense relation |
| Obligation universe | bounded finite set \(\Omega_{c,t}\), max size 12 | local tactical tasks near anchor \(t\) | ordered obligation atoms |
| Boolean lattice | \(2^{\Omega_{c,t}}\) | small groups of simultaneous local tasks | subset masks |
| Zeta neighborhood | \(N(U)=\bigcup_{o\in U}N(o)\) | defenders available to a task group | defender union counts |
| Hall defect | \([|U|-|N(U)|]_+\) | overloaded response structure | scalar spectra per anchor |
| Neural head | trainable map on board + HDZ tensor | classify true puzzle vs negative | one logit |

The finite-algebraic contribution is not the attack generator itself. The contribution is the **defect spectrum** over the Boolean lattice of local obligations, computed after legal-effectiveness filtering. Ordinary attack counts ask, "How many defenders touch this square?" HDZ asks, "For small collections of nearby tactical duties, how many independent legal responders are missing?"

## 5. Candidate Search Trace

The design search used the following constraints:

1. The operator must use only the current board.
2. It must produce a differentiable-friendly tensor but may itself be deterministic.
3. It must be able to distinguish true puzzles from near-puzzles where a single hidden resource invalidates the tactic.
4. It must be finite-algebraic in a concrete way, not merely inspired by algebraic vocabulary.
5. It must include a semantics-destroying algebraic ablation and a same-parameter neural control.

Candidates considered:

| Candidate | Summary | Failure mode | Decision |
|---|---|---|---|
| Local attack-count spectrum | Count attackers and defenders by piece type around targets | Too close to ordinary handcrafted features; misses overload | Rejected |
| King-zone domination number | Estimate minimum attackers required to cover king zone | Good for mate puzzles, narrow for general tactics | Rejected as sole idea |
| Piece-value residue code | Encode pieces by modular value classes and aggregate local residues | Risk of becoming a disguised positional/material encoding | Rejected |
| Pin-only algebra | Tensor of pinned pieces and pin directions | Useful but too sparse; not enough for near-puzzle separation | Subsumed inside HDZ |
| Matching-defect algebra | Use Hall defects over finite obligation-defender relations | Directly targets overload, pins, recapture scarcity, and near-puzzle resources | Selected |

The selected design keeps the useful parts of pin geometry and local tactical obligation modeling, but makes the final signal an algebraic deficiency rather than a raw count.

## 6. Rejected Approaches

These approaches are not used in HDZ:

- Zobrist-style hashing or kernelization.
- Finite-field character-sum features.
- Bitboard shift-polynomial feature algebra.
- Kinematic commutator constructions.
- Orbit-quotient symmetry pooling.
- Learned square IDs, rank/file one-hots, sinusoidal square embeddings, or other ordinary positional encodings.
- Engine-derived targets or side channels, including evaluations, PVs, mate depths, node counts, best moves, source labels, verification metadata, or source identity.

Other rejected but tempting approaches:

- **Pure material imbalance:** often separates obvious positions but fails hard negatives.
- **Legal-move count:** leaks mobility but does not identify tactical non-shareability.
- **Check indicator only:** too narrow; many puzzles begin from quiet or non-checking positions.
- **Full move-search supervision:** outside the data contract and forbidden at inference.

## 7. Mathematical Thesis

### Thesis

A true puzzle is more likely than a near-puzzle to contain a small local set of tactical obligations whose effective defender neighborhood has positive Hall defect after pins and king-exposure constraints are applied.

Equivalently, for some side \(c\), anchor square \(t\), and small obligation set \(U\):

\[
|U|>|N(U)|.
\]

This inequality means the side has fewer independent legal responders than local tactical tasks. In practice, such defects correspond to themes like:

- overloaded defender,
- pinned defender,
- loose high-value target,
- defender dragged away from king safety,
- recapture square not actually covered,
- one missing interposition square,
- apparent defense that is illegal because it exposes the king.

Near-puzzles often repair the inequality by adding exactly one defender, unpinning a defender, changing a ray blocker, or adding a king escape resource. HDZ is designed to make those small algebraic differences visible to the classifier.

### Proof sketch of what HDZ measures

Fix a side \(c\), anchor \(t\), obligation universe \(\Omega_{c,t}\), and effective defender set \(P_c\). Define the finite relation \(R_{c,t}\subseteq \Omega_{c,t}\times P_c\). For an obligation subset \(U\), let:

\[
N(U)=\{p\in P_c: \exists o\in U, R_{c,t}(o,p)=1\}.
\]

By Hall's theorem, a matching that assigns every obligation in \(U\) to a distinct effective defender exists only if every \(V\subseteq U\) satisfies \(|N(V)|\ge |V|\). Therefore, a positive value

\[
\delta(V)=|V|-|N(V)|>0
\]

certifies that no distinct-responder assignment can cover all obligations in \(V\). HDZ computes these positive deficiencies for all small subsets up to order four. Because \(R_{c,t}\) removes pinned or king-illegal defenses, the measured defect is not just a geometric count; it is a finite-algebraic summary of legal tactical overload under a one-response-per-piece abstraction.

HDZ does not prove that a position is a puzzle. It measures whether the current board contains local non-shareability patterns that are common in true tactical positions and often absent in near-puzzles.

## 8. Algebraic Operator

### Name

**Hall-Defect Zeta Operator**, abbreviated **HDZ**.

### Fixed sets

Let:

\[
S=\{0,1,\ldots,7\}\times\{0,1,\ldots,7\}
\]

be the 64 board squares. Let colors be \(C=\{W,B\}\). Let \(P_c\) be the pieces of color \(c\) on the current board.

For each color \(c\) and anchor square \(t\in S\), HDZ constructs a finite ordered obligation universe:

\[
\Omega_{c,t}=\operatorname{trim}_{12}(A_{c,t}).
\]

The candidate atom set \(A_{c,t}\) is ordered deterministically as follows:

1. The anchor square \(t\).
2. The square of the king of color \(c\), if it is within Chebyshev distance two of \(t\).
3. Empty or occupied squares in the king ring \(N_\infty(k_c)\), if \(t\) is within Chebyshev distance two of \(k_c\).
4. Squares occupied by color \(c\) pieces of value at least three within Chebyshev distance two of \(t\).
5. Squares occupied by color \(c\) pieces currently attacked by the opponent within Chebyshev distance three of \(t\).
6. Interposition or capture squares on any opponent rook, bishop, or queen line that reaches \(t\), \(k_c\), or a high-value nearby color-\(c\) piece through at most one blocker.
7. Remaining squares in the Chebyshev ring around \(t\), ordered by increasing distance, then rank, then file.

Duplicate squares are removed at first occurrence. If fewer than 12 atoms are present, pad with null atoms. Null atoms contribute no relation edges and are masked out of subset enumeration.

### Effective defense relation

For every color \(c\), piece \(p\in P_c\), and square \(s\in S\), compute a raw contact predicate:

\[
A_c(p,s)=1
\]

when \(p\) attacks or defends \(s\) under current occupancy using normal chess movement geometry. Sliding pieces require an unblocked segment. Pawns use attack squares, not push squares. Kings use adjacent contact squares but are not allowed to defend squares occupied or controlled in a way that would violate king safety.

Then apply a pin and exposure filter:

\[
E_c(p,s)=A_c(p,s)\land L_c(p,s),
\]

where \(L_c(p,s)=1\) only if moving or assigning piece \(p\) to respond on square \(s\) does not expose the king of color \(c\) to an opponent rook, bishop, or queen line. For pinned pieces, this usually means that only responses along the king-pin line remain effective. For knights and pawns pinned off-line, most apparent defenses are removed.

The local incidence relation is:

\[
R_{c,t}(o,p)=1 \quad \text{iff} \quad E_c(p,o)=1,
\]

for non-null obligation atom \(o\in\Omega_{c,t}\).

### Boolean-lattice zeta aggregation

For each non-null obligation atom \(o_i\), define its defender neighborhood:

\[
D_i=\{p\in P_c:R_{c,t}(o_i,p)=1\}.
\]

For each subset \(U\subseteq\Omega_{c,t}\), define the zeta union:

\[
D(U)=\bigcup_{o_i\in U}D_i.
\]

This is the Boolean-lattice zeta step: atom neighborhoods are aggregated upward from atoms to subsets by finite union. For implementation, \(D_i\) can be stored as a 16-bit set over the side's pieces. That is only a finite-set container; the feature is not built from bitboard shifts or polynomial board shifts.

For subset order \(r\in\{1,2,3,4\}\), compute:

\[
\operatorname{maxdef}_{c,t,r}=\max_{|U|=r}\left[ r-|D(U)| \right]_+,
\]

\[
\operatorname{meandef}_{c,t,r}=\operatorname{mean}_{|U|=r}\left[ r-|D(U)| \right]_+,
\]

\[
\operatorname{mindefenders}_{c,t,r}=\min_{|U|=r}|D(U)|,
\]

\[
\operatorname{pinshare}_{c,t,r}=\max_{|U|=r}\frac{|D(U)\cap \operatorname{Pinned}_c|}{\max(1,|D(U)|)}.
\]

These four values per subset order form the main HDZ spectrum.

### Additional scalar channels

For each \((c,t)\), also compute:

1. `raw_attackers_on_t`: number of opponent pieces attacking \(t\).
2. `effective_defenders_on_t`: number of color-\(c\) pieces effectively defending \(t\).
3. `pinned_defenders_on_t`: number of effective defenders of \(t\) that are pinned.
4. `loose_target_flag`: `1` if \(t\) is occupied by a color-\(c\) piece, attacked by the opponent, and has zero effective defenders.

Total per side per square:

\[
4 \text{ subset orders}\times 4 \text{ spectrum values}+4 \text{ scalar values}=20.
\]

Total HDZ channels for both sides:

\[
C_{HDZ}=40.
\]

## 9. Tensor Contract

### HDZ tensor

\[
H=\operatorname{HDZ}(X)\in\mathbb{R}^{8\times8\times40}.
\]

Channel layout:

For each side in order `[white, black]`, repeat the following 20 channels:

| Local channel | Description |
|---:|---|
| 0 | `maxdef_r1` |
| 1 | `meandef_r1` |
| 2 | `mindefenders_r1` |
| 3 | `pinshare_r1` |
| 4 | `maxdef_r2` |
| 5 | `meandef_r2` |
| 6 | `mindefenders_r2` |
| 7 | `pinshare_r2` |
| 8 | `maxdef_r3` |
| 9 | `meandef_r3` |
| 10 | `mindefenders_r3` |
| 11 | `pinshare_r3` |
| 12 | `maxdef_r4` |
| 13 | `meandef_r4` |
| 14 | `mindefenders_r4` |
| 15 | `pinshare_r4` |
| 16 | `raw_attackers_on_t` |
| 17 | `effective_defenders_on_t` |
| 18 | `pinned_defenders_on_t` |
| 19 | `loose_target_flag` |

Normalize count channels by dividing by 16. Defect channels may remain in `[0,4]` or be divided by the subset order. Use the same normalization in all ablations.

### Network input

The model receives:

\[
[X,H]\in\mathbb{R}^{8\times8\times53}.
\]

No square-id plane, rank/file plane, learned absolute square table, source field, engine field, puzzle ID, or verification field may be concatenated.

### Output

One logit:

\[
z=f_\theta(X,H).
\]

## 10. Architecture

Recommended first architecture: **HDZ-ConvLite**.

### Branches

1. **Raw board branch**
   - Input: \(X\in\mathbb{R}^{8\times8\times13}\)
   - `3x3 conv, 64 channels, padding 1`
   - `GELU`
   - `3x3 conv, 64 channels, padding 1`
   - `GELU`

2. **Algebraic branch**
   - Input: \(H\in\mathbb{R}^{8\times8\times40}\)
   - `1x1 conv, 64 channels`
   - `GELU`
   - `1x1 conv, 64 channels`
   - `GELU`

3. **Fusion branch**
   - Concatenate raw and algebraic features: 128 channels
   - `3x3 conv, 96 channels, padding 1`
   - `GELU`
   - `3x3 conv, 96 channels, padding 1`
   - `GELU`
   - Global mean pooling over the 64 squares
   - Side-to-move scalar gate from channel 13, applied after pooling
   - `linear 96 -> 64`
   - `GELU`
   - `linear 64 -> 1`

No absolute positional embeddings. No orbit-quotient pooling. Board symmetries may be used as data augmentation, provided the label is unchanged and the side-to-move plane is transformed consistently.

### Same-parameter neural control

Control name: **NeuralSynth-40 Control**.

Use the identical architecture and identical trainable parameter count. Replace only the deterministic HDZ tensor:

- Main model algebraic input: `H = HDZ(X)`, 40 channels.
- Control algebraic input: `H_control = FixedTile40(X)`, 40 channels.

`FixedTile40(X)` is a parameter-free tiling/padding map that repeats the 13 current-board channels into 40 channels using a fixed channel schedule. The same two `1x1 conv` layers are then applied. Therefore the algebraic branch has exactly the same trainable dimensions in the main model and the control model.

This control asks whether the gain comes from HDZ semantics rather than from merely giving the model a second 40-channel branch.

## 11. Training Objective

### Primary loss

Weighted binary cross entropy on the single logit:

\[
\mathcal{L}=w_y\cdot\operatorname{BCEWithLogits}(z,y),
\]

where:

\[
y=1[\text{fine_label}=2].
\]

Suggested initial weights:

| Fine label | Binary target | Suggested weight |
|---:|---:|---:|
| 0 | 0 | 1.0 |
| 1 | 0 | 2.0 |
| 2 | 1 | class-balanced positive weight |

The near-puzzle hard negative should be upweighted because it is the key test of whether HDZ captures tactical overload rather than superficial puzzle-like structure.

### Diagnostics, not extra inference heads

Report:

- AUROC: label `2` vs labels `0,1`.
- AUROC: label `2` vs label `1` only.
- False-positive rate on label `1` near-puzzles at fixed true-puzzle recall.
- Calibration curves separately for labels `0`, `1`, and `2`.
- Mean HDZ defect by fine label, stratified by side to move and material bucket.

Do not train an auxiliary label-0/1/2 classifier unless the deployment still outputs one logit and the auxiliary label is not used as an inference side channel. The clean first experiment should use one binary head only.

## 12. Ablations

### A. Semantics-destroying algebraic ablation: AtomScramble-HDZ

Keep the HDZ pipeline, tensor shape, normalization, subset enumeration, and downstream architecture. Destroy chess semantics by replacing effective defense queries with scrambled atom queries.

Define a fixed permutation of board-square indices:

\[
\pi(i)=(37i+11)\bmod 64.
\]

Because 37 is a unit modulo 64, \(\pi\) is a bijection. It is intentionally not a chess-board symmetry. For an obligation atom square \(o\), query defenders of \(\pi(o)\) instead of defenders of \(o\):

\[
R^{scr}_{c,t}(o,p)=E_c(p,\pi(o)).
\]

The local obligation subset \(U\) remains anchored to \(t\), but the support relation is evaluated on unrelated squares. This preserves:

- finite Boolean-lattice size,
- subset order distribution,
- channel count,
- downstream parameter count,
- count normalization,
- deterministic algebraic computation.

It destroys:

- the meaning of "defender of this local obligation,"
- pin-sensitive recapture semantics,
- local overload interpretation.

Expected result: AtomScramble-HDZ should underperform real HDZ, especially on label `2` vs label `1`. If it does not, the claimed semantics are probably not carrying the gain.

### B. Same-parameter neural control: NeuralSynth-40

Use `FixedTile40(X)` instead of `HDZ(X)` while keeping the algebraic branch trainable layers identical. This preserves parameter count and input branch width but removes the finite-algebraic measurements.

Expected result: Real HDZ should beat NeuralSynth-40 on near-puzzle rejection. If NeuralSynth-40 matches HDZ within confidence intervals, the algebraic operator is not justified.

### C. Pin filter ablation

Replace \(E_c(p,s)\) with raw attack/defense \(A_c(p,s)\), leaving the Hall-defect zeta computation unchanged.

Expected result: performance should drop on examples where a near-puzzle is repaired by an unpinned defender or a true puzzle relies on a pinned defender being unusable.

### D. Subset-order ablation

Train with max subset order:

- `r <= 1`: only single-obligation scarcity.
- `r <= 2`: pair overload.
- `r <= 3`: small tactical clusters.
- `r <= 4`: default.

Expected result: `r <= 1` should be too weak. The main gain should appear at `r = 2` and `r = 3`; `r = 4` may add marginal signal.

### E. Obligation-universe ablation

Compare:

1. Anchor-only obligations.
2. Anchor plus local ring.
3. Anchor plus local ring plus king-zone atoms.
4. Full default with line-interposition atoms.

Expected result: full HDZ should perform best on mixed puzzle types; king-zone atoms should matter most for mating puzzles; line-interposition atoms should matter for skewers, pins, and discovered attacks.

## 13. Falsification

HDZ should be considered falsified or at least demoted if any of the following occur on clean splits:

1. **AtomScramble parity:** AtomScramble-HDZ matches or beats real HDZ on label `2` vs label `1`.
2. **NeuralSynth parity:** the same-parameter NeuralSynth-40 control matches real HDZ within statistical uncertainty.
3. **No pin sensitivity:** removing the pin/exposure filter does not change near-puzzle false-positive rate.
4. **Only material correlation:** HDZ improvement disappears after stratifying by material bucket and side to move.
5. **Poor hard-negative behavior:** HDZ improves ordinary-negative discrimination but worsens label `1` near-puzzle rejection.
6. **Calibration damage:** HDZ raises AUROC but produces badly miscalibrated high-confidence false positives on near-puzzles.
7. **Symmetry brittleness:** legal board transformations used as augmentation produce inconsistent logits beyond normal stochastic variance.

Minimum credible win condition:

- HDZ-ConvLite beats both AtomScramble-HDZ and NeuralSynth-40 on label `2` vs label `1`, with the same data, same optimizer, same trainable parameter count for the branch, and no forbidden inputs.

## 14. Implementation Notes

### Determinism

HDZ should be deterministic given the current-board tensor. The ordered obligation list must use a stable ordering:

1. priority group,
2. Chebyshev distance to anchor,
3. rank,
4. file.

Do not use random order during feature generation, except in the fixed ablation permutation where the permutation is documented and seeded once.

### Pseudocode

```python
def hdz(board_tensor):
    pieces = parse_current_board(board_tensor)
    contacts = compute_raw_contacts(pieces)          # current board only
    effective = apply_pin_exposure_filter(contacts)  # no engine, no search

    H = zeros((8, 8, 40))
    channel_offset = {WHITE: 0, BLACK: 20}

    for c in [WHITE, BLACK]:
        pinned = pinned_piece_set(pieces, c)
        for t in all_64_squares():
            omega = ordered_obligation_atoms(board_tensor, c, t, max_atoms=12)
            omega = [o for o in omega if o is not NULL]

            supports = []
            for o in omega:
                defenders = set()
                for p in pieces[c]:
                    if effective[p, o]:
                        defenders.add(p)
                supports.append(defenders)

            local = []
            for r in [1, 2, 3, 4]:
                defects = []
                defender_counts = []
                pinshares = []
                for U in combinations(range(len(omega)), r):
                    D = union(supports[i] for i in U)
                    defect = max(0, r - len(D))
                    defects.append(defect)
                    defender_counts.append(len(D))
                    pinshares.append(len(D & pinned) / max(1, len(D)))

                local.extend([
                    max(defects, default=0),
                    mean(defects, default=0),
                    min(defender_counts, default=0),
                    max(pinshares, default=0),
                ])

            local.extend([
                count_raw_attackers_on_square(pieces, c, t),
                count_effective_defenders_on_square(effective, c, t),
                count_pinned_effective_defenders(effective, pinned, c, t),
                loose_target_flag(pieces, effective, c, t),
            ])

            H[t.rank, t.file, channel_offset[c]:channel_offset[c]+20] = normalize(local)

    return H
```

### Complexity

For each side and square, enumerate at most:

\[
\binom{12}{1}+\binom{12}{2}+\binom{12}{3}+\binom{12}{4}=793
\]

subsets. Across 2 sides and 64 anchors, this is about 101,504 small subset evaluations per board. Since each support is at most a 16-piece set, this is cheap relative to a neural forward pass if vectorized or cached by support masks.

Recommended implementation representation:

- Store each defender support as a 16-bit integer over the side's pieces.
- Precompute subset masks for each active atom count from 0 to 12.
- Use popcount for defender counts.
- Use ordinary tensor operations for the neural network.

The 16-bit support representation is only a compact finite-set representation. Do not construct board features through shift-polynomial operations.

### Current-board legality limits

HDZ is not a legal move search. It may approximate rare state-dependent details such as castling rights and en-passant unless those channels are explicitly included as current-board state. The operator should still enforce basic king-exposure legality for defensive effectiveness, because pinned and illegal defenders are central to the thesis.

### Leakage checklist

Before training, verify that the model input batch contains only:

- piece planes,
- side-to-move plane,
- deterministic HDZ tensor computed from those planes,
- binary target derived from fine label for training only.

It must not contain:

- engine score,
- best move,
- principal variation,
- mate score,
- node count,
- source label,
- puzzle ID as feature,
- verification metadata,
- source identity,
- move count from a search process.

## 15. Prompt Maintenance

When reusing or extending this packet, preserve these requirements:

1. Keep the model output to one logit.
2. Keep inference inputs limited to the current-board tensor and deterministic features computed from it.
3. Keep fine labels `0/1/2` out of the model input; use them only for binary target mapping and diagnostics.
4. Keep HDZ finite-algebraic: the central feature is Hall defect over the Boolean lattice of local obligations.
5. Keep the semantics-destroying algebraic ablation. It is not optional.
6. Keep the same-parameter neural control. It is not optional.
7. Do not add ordinary positional encodings or source-derived metadata.
8. Do not silently replace pin-filtered effective defense with raw attack counts.
9. Report hard-negative performance on label `1` separately from ordinary negatives.
10. Treat HDZ as a falsifiable hypothesis, not as a guaranteed puzzle detector.

The one-sentence maintenance summary:

**HDZ tests whether finite Hall defects in pin-filtered local obligation algebras expose the tactical overload patterns that distinguish true puzzles from near-puzzles using only the current board.**
