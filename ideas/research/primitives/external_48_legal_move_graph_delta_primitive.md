# p047_legal_move_graph_delta_primitive.md

## Why this primitive exists

### Thesis

The `puzzle_binary` benchmark is a one-logit puzzle-versus-non-puzzle task where the central failure mode is not random-position confusion, but near-puzzle false positives. The benchmark documentation makes that explicit, and it also pins the current reference line to the LC0 BT4-style tower, with historical reference metrics of test F1 `0.7445`, PR AUC `0.8068`, near-puzzle false-positive rate `0.2477`, and puzzle recall `0.7943`. fileciteturn32file0L3-L3

The repo already contains three relevant legal-routing branches: `p009` as an additive legal-move graph head over the `ExchangeThenKingDualStreamNetwork`, `p011` as a typed legal-edge compile-scatter head with per-edge gating, and `a014` as a **controlled BT4 mixer study** rather than a standalone board-aware primitive. The BT4 mixer adaptation is especially revealing: it explicitly says the mixer only sees opaque `(B, C, 8, 8)` channels and therefore cannot recover literal chess piece-type identity, so it replaces real chess-piece relations with content-derived relation thresholds. That is useful as a controlled mixer experiment, but it is the wrong place to anchor a native legal-move primitive. fileciteturn8file0L3-L3 fileciteturn23file0L3-L3 fileciteturn6file0L3-L3 fileciteturn17file0L3-L3

My thesis is therefore:

**`p047` should be a board-aware, reusable candidate-move graph primitive, not a BT4-only mixer.** It should compile a batched **pseudo-legal-plus** move graph directly from `simple_18` tensors, keep graph topology discrete and stop-gradient, attach rich move-edge features that measure **tactical pressure deltas** along each candidate move, run a small number of edge-square message-passing steps, and emit a gated puzzle-delta head that can sit on top of the repo’s existing puzzle_binary backbones before any later mixer adaptation is attempted. That keeps the math aligned with the original legal-move-graph proposal while removing the main engineering failure points. fileciteturn29file0L3-L3 fileciteturn11file0L3-L3

## Candidate move graph

### Move graph math

The codebase already has the right substrate for a native compiler. `simple_18` encodes white and black piece planes, side to move, castling rights, and en-passant; `rule_graph_features.py` already precomputes geometric attack masks, between-square masks, and ray-step tables. It also already contains a `compute_legal_move_graph()` helper, but that helper openly describes itself as an **approximation** that drops some subtleties including in-check filtering, en-passant, and castling. The current `p009` `_compute_typed_legal_edges()` is even narrower: it is “attack-style; pseudo-legal” and excludes own-piece targets, which means it is not yet a full candidate-move compiler for puzzle reasoning. fileciteturn24file0L3-L3 fileciteturn11file0L3-L3

So the right graph for `p047` is not the old attack-only adjacency. It is:

\[
G(B) = (V, E(B)), \qquad V = \{0,\dots,63\},
\]

where \(B\) is one board in `simple_18`, and \(E(B)\) is a **pseudo-legal-plus candidate edge set** for the side to move. “Pseudo-legal-plus” means:

\[
E(B)=E_{\text{pseudo}}(B)\cup E_{\text{special}}(B),
\]

where \(E_{\text{pseudo}}\) is compiled directly from board tensors by piece geometry and occupancy, and \(E_{\text{special}}\) adds correctly handled promotion, en-passant, and castling edges because the input tensor already contains the necessary state for them. fileciteturn24file0L3-L3

The compiler should be **edge-centric**, not adjacency-centric. In other words, do not start from a dense learned score matrix and threshold it, which is what the current BT4 legal-move mixer does with `torch.quantile`. Instead, emit explicit candidate move edges:

\[
e = (b, s, t, m),
\]

where \(b\) is batch id, \(s\) is source square, \(t\) is target square, and \(m\) is move metadata. This avoids both the BT4 compromise and the quantile-threshold path. The current BT4 mixer explicitly uses `torch.quantile` to threshold relation scores, while the PyTorch quantile operator is defined in terms of sorted-order interpolation. That is fine for floating summaries, but it is unnecessary fragility for discrete move-topology construction. fileciteturn17file0L3-L3 citeturn6view5

The message-passing form should borrow from standard graph-neural-network practice rather than inventing an exotic new optimizer surface. The relevant priors are clear: MPNN gives the node/edge/message/update template, GraphSAGE motivates degree-normalized neighborhood means, R-GCN motivates relation-conditioned transforms, and GAT motivates learned edge weighting over sparse neighborhoods. `p047` should use that family, but with chess-native candidate edges and chess-native delta features. citeturn10academia0turn10academia1turn11academia0turn11academia1

### Edge feature design

The edge feature vector should have four blocks:

**Move identity.** This block should encode mover piece type, move mode, source square, target square, source-target displacement, ray length, whether the move is quiet or a capture, captured piece type if any, promotion piece type if any, and castling/en-passant indicators. This is the minimum metadata needed to tell a rook x-ray from a knight fork and a pawn promotion from a quiet king step. Because `simple_18` contains castling-rights planes and an en-passant plane, there is no data-contract reason to omit those move classes. fileciteturn24file0L3-L3

**Legality and tactical tags.** Python-chess is useful here as the oracle for semantics, even though it should stay out of the training forward path. Its docs distinguish `legal_moves` from `pseudo_legal_moves`, note that pseudo-legal moves may leave or put the king in check, expose `is_pinned()`, and expose `gives_check()` for moves that are at least pseudo-legal. That implies the primitive should not try to compress everything into one binary “legal” flag. Instead, attach tags such as `source_is_absolutely_pinned`, `move_stays_on_pin_ray`, `side_to_move_is_in_check`, `move_resolves_check`, `king_move_into_attack`, `castle_path_safe`, and `gives_check`. citeturn6view0turn6view1turn8view1turn9view0

**Pressure-delta features.** This is the load-bearing idea. For each candidate edge \(e:s\rightarrow t\), define a small local counterfactual \(B^{(e)}\) that updates only the affected squares: source, target, captured square for en-passant, rook squares for castling, and promotion replacement if needed. Then compute tactical pressure deltas only on a **local stencil** of critical squares instead of recomputing the whole board. A good default stencil is

\[
Q_e=\{t, k_{\text{opp}}, k_{\text{self}}\}\cup \mathcal{N}(t)\cup \mathcal{N}(k_{\text{opp}}),
\]

where \(k_{\text{opp}}\) and \(k_{\text{self}}\) are the king squares and \(\mathcal{N}\) is the 8-neighborhood. For each \(q\in Q_e\), compute deltas like friendly attackers, enemy attackers, friendly defenders, enemy defenders, and occupancy-opened ray indicators. Those local deltas are the primitive’s main signal about whether a move increases forcing pressure, releases an x-ray, hangs material, or exposes a king. The repo’s existing geometry tables and ray helpers are already enough to support this localized recomputation. fileciteturn24file0L3-L3

**Pin-aware separation of attack and legality.** Python-chess explicitly notes that pinned pieces still count as attackers. That subtle point matters. It means the primitive should not zero out a pinned source square’s tactical influence. Instead, it should keep attack/defense pressure features and add separate pin-legality tags. In practice: a pinned bishop can still contribute to a target’s attack count, but the move edge that leaves the pin line should be tagged as tactically suspect or illegal. That separation is much more faithful than trying to bake pin consequences into one mask. citeturn8view1

## Delta propagation

### Update equations

Let \(h_i^{(0)}\in\mathbb{R}^{d_n}\) be the square token for square \(i\), produced by a cheap board-aware encoder over `simple_18`, for example the repo’s existing 1x1-conv token tower style. Let \(f_e\in\mathbb{R}^{d_f}\) be the static edge feature vector described above. Let \(s(e)\) and \(t(e)\) denote source and target squares. Then initialize edge states as:

\[
m_e^{(0)} = \phi_e\!\left([h_{s(e)}^{(0)} \,\|\, h_{t(e)}^{(0)} \,\|\, f_e]\right).
\]

A robust default then uses two edge-square rounds. The square update is:

\[
\bar m^{\text{out}}_i = \operatorname{mean}_{e:s(e)=i} m_e^{(\ell)}, \qquad
\bar m^{\text{in}}_i = \operatorname{mean}_{e:t(e)=i} m_e^{(\ell)},
\]

\[
h_i^{(\ell+1)} = \operatorname{LN}\!\Big(
h_i^{(\ell)} +
W_{\text{self}} h_i^{(\ell)} +
W_{\text{out}}\bar m^{\text{out}}_i +
W_{\text{in}}\bar m^{\text{in}}_i
\Big).
\]

The edge update is:

\[
g_e^{(\ell)}=\sigma\!\Big(
w_g^\top \operatorname{LN}\big([h_{s(e)}^{(\ell+1)} \,\|\, h_{t(e)}^{(\ell+1)} \,\|\, f_e]\big)
\Big),
\]

\[
m_e^{(\ell+1)}=
\operatorname{LN}\!\Big(
m_e^{(\ell)} + g_e^{(\ell)}\cdot
\phi_m\!\left([h_{s(e)}^{(\ell+1)} \,\|\, h_{t(e)}^{(\ell+1)} \,\|\, f_e]\right)
\Big).
\]

This combines the repo’s existing move-edge intuition with three stable ideas from the graph literature: mean-normalized neighborhood aggregation, relation-aware edge modeling, and learned edge gating inside a sparse neighborhood. fileciteturn23file0L3-L3 citeturn10academia0turn10academia1turn11academia0turn11academia1

The board readout should **not** be a plain mean over edges, because puzzle positions often hinge on one or a few forcing moves. A better readout is:

\[
z_{\text{board}}=
\Big[
\operatorname{mean}_i h_i^{(L)} \,\|\, 
\operatorname{amax}_i h_i^{(L)} \,\|\, 
\operatorname{mean}_e m_e^{(L)} \,\|\, 
\operatorname{amax}_e m_e^{(L)} \,\|\, 
\operatorname{logsumexp}_e (w_s^\top m_e^{(L)})
\Big].
\]

Then produce a primitive delta:

\[
\delta_{\text{prim}} = \phi_{\text{head}}(z_{\text{board}}),
\]

and combine it with a base trunk logit using a gated additive form that the repo already uses in `p009` and `p011`:

\[
\text{logit} = \text{base\_logit} + \sigma(g(\text{base\_ctx}))\cdot \delta_{\text{prim}}.
\]

That is the safest first integration because it lets the new primitive prove its value as a controlled add-on before it is allowed to reshape the main spatial backbone. fileciteturn11file0L3-L3 fileciteturn16file0L3-L3

### Mixed-precision safety plan

The mixed-precision plan should be conservative and explicit.

**Topology path.** Compile move edges under `torch.no_grad()` and keep topology tensors as `bool` and index tensors as integer types until the last possible moment. That matches the existing legal-routing implementation style, and it preserves the original “no gradient with respect to the discrete graph” intent from the legal-move-graph proposal. fileciteturn9file0L3-L3 fileciteturn11file0L3-L3 fileciteturn29file0L3-L3

**Layout discipline.** PyTorch’s docs are very clear that `Tensor.view()` requires compatible strides and that when that is unclear it is better to use `reshape()`, which returns a view when possible and a copy otherwise. So `p047` should treat raw `.view()` on post-transpose or post-pack tensors as forbidden. Use `flatten`, `permute(...).contiguous()`, and `reshape`, especially in edge packing/unpacking and batch-edge collapsing. This directly removes one of the fragility classes visible in the current legal-move implementations. citeturn6view6turn6view7 fileciteturn11file0L3-L3 fileciteturn17file0L3-L3

**Autocast boundaries.** PyTorch lists `bmm`, `linear`, and convolution ops among CUDA operators that autocast to `float16`, and it explicitly allows nested `autocast(enabled=False)` regions when a subregion must run in a particular dtype. The safe pattern here is: keep token projection MLPs and batched GEMMs under autocast, but force edge aggregation, degree computation, `index_add_`/scatter accumulation, and pressure-delta normalization into an autocast-disabled `float32` block. That prevents half-precision reductions from becoming the silent failure mode. citeturn6view2turn6view3

**No quantile-dependent topology.** `torch.quantile` exists for floating quantile computation with interpolation, but topology construction in this primitive should not depend on quantile thresholds at all. Candidate edges should come from chess rules, not from thresholding dense learned scores. If a “top forcing moves” readout is needed later, do it with `topk` or `amax` on **floating edge scores after** the move graph already exists. That keeps the discrete graph compiler away from dtype/interpolation corner cases and avoids repeating the current BT4 mixer’s thresholding strategy. fileciteturn17file0L3-L3 citeturn6view5

**Loss and head path.** The benchmark already uses `BCEWithLogitsLoss`, and PyTorch AMP explicitly says `binary_cross_entropy_with_logits` and `BCEWithLogitsLoss` are safe to autocast, whereas `binary_cross_entropy`/`BCELoss` can fail in autocast-enabled regions. So `p047` should keep the single-logit interface and never insert an explicit sigmoid before the criterion. fileciteturn32file0L3-L3 citeturn7view0

**Numerical debug mode.** PyTorch’s numerical notes also matter: floating-point addition is not associative, batched `bmm` is not guaranteed to be bitwise identical to slice-by-slice computation, and reduced-precision reductions in FP16/BF16 GEMMs can sometimes produce unexpected results. Therefore promotion-grade testing should include a debug flag that disables reduced-precision GEMM reductions when needed, and all fp32-vs-AMP comparisons should use tolerances, not exact equality. citeturn17view0turn17view1turn17view2

## Code path

### Integration plan

The cleanest integration path is a **reusable core plus thin wrappers**:

- `LegalMoveGraphDeltaCore`: compiles candidate edges, builds edge features, runs edge-square message passing, and returns square states, edge states, and diagnostics.
- `LegalMoveGraphDeltaHead`: wraps the core as a gated additive puzzle_binary head on top of an existing board-aware trunk.
- `LegalMoveGraphDeltaStandalone`: optional direct classifier if later ablations show the primitive can carry more of the burden itself.
- `BT4 adapter`: explicitly deferred until the core proves value outside the mixer-only setting.

That plan fits the repo’s existing model-registration style, preserves the `simple_18` and one-logit contract, and avoids repeating the “controlled BT4 mixer study” trap. fileciteturn6file0L3-L3 fileciteturn8file0L3-L3 fileciteturn23file0L3-L3 fileciteturn26file0L3-L3

A practical first wrapper should be an additive head over the repo’s board-aware trunk family, not a wholesale trunk replacement. That mirrors how `p009` and `p011` were introduced and keeps the experiment controlled: the primitive only has to prove that candidate-move deltas add useful tactical evidence beyond the existing trunk, which is exactly the right first research question for puzzle_binary. fileciteturn8file0L3-L3 fileciteturn23file0L3-L3

Recommended diagnostics in `forward()` are:

`edge_count`, `legalish_edge_count`, `check_edge_count`, `capture_edge_count`, `promotion_edge_count`, `pinned_source_edge_count`, `mean_edge_score`, `max_edge_score`, `king_pressure_delta_mean`, and per-piece-type move-pressure norms.

That follows the precedent set by the current legal-move branches, which already expose edge counts and per-type norms for auditability. fileciteturn9file0L3-L3 fileciteturn10file0L3-L3

### Speed and batching

The right engineering pattern is **dense reference, packed fast path**.

The dense reference path uses fixed-shape boolean masks such as `(B, M, 64, 64)` for debug and correctness, where `M` is the number of move-mode channels. This is easy to reason about, easy to compare against old code, and small enough at 8x8 scale to keep in CI. The packed fast path then turns active move masks into one batched edge table `(E_total, …)` and uses batched gathers plus `index_add_` to aggregate back to flattened square ids `(b*64 + sq)`. Because the board is tiny but batch size can be large, that hybrid pattern is better than either pure sparse Python loops or a permanently dense edge-feature tensor everywhere. The repo’s current legal head already batches per-type `bmm` through a collapsed `(B*6, 64, 64) @ (B*6, 64, m)` path, so the codebase is already oriented toward batched graph kernels rather than per-position move objects. fileciteturn9file0L3-L3 fileciteturn11file0L3-L3

The forward path should never call python-chess or create move objects. Python-chess belongs only in the oracle tests. In the model itself, every move class should be emitted by tensorized rules over the precomputed geometry tables and the input planes. That is the only way to avoid the “expensive unbatched legal move generation” failure mode and still preserve exact-or-nearly-exact chess semantics where they matter. fileciteturn24file0L3-L3

One more operational note: PyTorch warns that batched matrix products and slice-by-slice equivalents are not guaranteed to be bitwise identical, even when mathematically identical. So microbenchmarks should compare **throughput and tolerated numeric drift**, not exact bitwise equality. That matters if you carry both dense-reference and packed-fast-path implementations. citeturn17view0turn17view1

### Implementation sketch

A promotion-grade implementation can stay surprisingly compact if the compiler and message passing are kept separate:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

@dataclass
class MoveGraphBatch:
    edge_batch: torch.Tensor        # (E,)
    src: torch.Tensor               # (E,)
    dst: torch.Tensor               # (E,)
    move_mode: torch.Tensor         # (E,)
    mover_type: torch.Tensor        # (E,)
    static_feat: torch.Tensor       # (E, F_static) float32
    edge_ptr: torch.Tensor | None   # optional CSR-style segmentation

class LegalMoveGraphDeltaCore(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        node_dim: int = 48,
        edge_dim: int = 64,
        mp_steps: int = 2,
    ) -> None:
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Conv2d(input_channels, node_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(node_dim, node_dim, kernel_size=1),
        )
        self.edge_init = nn.Sequential(
            nn.LayerNorm(2 * node_dim + 64),  # example feature size
            nn.Linear(2 * node_dim + 64, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.node_update = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(node_dim + 2 * edge_dim),
                nn.Linear(node_dim + 2 * edge_dim, node_dim),
                nn.GELU(),
                nn.Linear(node_dim, node_dim),
            ) for _ in range(mp_steps)
        ])
        self.edge_update = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(2 * node_dim + edge_dim + 64),
                nn.Linear(2 * node_dim + edge_dim + 64, edge_dim),
                nn.GELU(),
                nn.Linear(edge_dim, edge_dim),
            ) for _ in range(mp_steps)
        ])
        self.node_norm = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(mp_steps)])
        self.edge_norm = nn.ModuleList([nn.LayerNorm(edge_dim) for _ in range(mp_steps)])
        self.edge_gate = nn.ModuleList([nn.Linear(2 * node_dim + 64, 1) for _ in range(mp_steps)])
        self.readout = nn.Sequential(
            nn.LayerNorm(2 * node_dim + 3 * edge_dim),
            nn.Linear(2 * node_dim + 3 * edge_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    @torch.no_grad()
    def compile_graph(self, board: torch.Tensor) -> MoveGraphBatch:
        # Tensorized pseudo-legal-plus compiler:
        # pawns, knights, kings, sliders, promotions, castling, en-passant.
        # Returns packed edges and float32 static features.
        raise NotImplementedError

    def forward(self, board: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz = board.shape[0]
        node = self.node_embed(board).flatten(2).transpose(1, 2)  # (B, 64, Dn)

        graph = self.compile_graph(board)
        flat_node = node.reshape(bsz * 64, -1)
        src_h = flat_node[graph.edge_batch * 64 + graph.src]
        dst_h = flat_node[graph.edge_batch * 64 + graph.dst]

        with torch.autocast(device_type=board.device.type, enabled=False):
            edge_static = graph.static_feat.float()
            edge = self.edge_init(torch.cat([src_h.float(), dst_h.float(), edge_static], dim=-1))

            for upd_node, upd_edge, norm_n, norm_e, gate in zip(
                self.node_update, self.edge_update, self.node_norm, self.edge_norm, self.edge_gate
            ):
                # edge -> node aggregation in fp32
                out_acc = torch.zeros(bsz * 64, edge.shape[-1], device=edge.device, dtype=torch.float32)
                in_acc = torch.zeros_like(out_acc)
                out_deg = torch.zeros(bsz * 64, 1, device=edge.device, dtype=torch.float32)
                in_deg = torch.zeros_like(out_deg)

                flat_src = graph.edge_batch * 64 + graph.src
                flat_dst = graph.edge_batch * 64 + graph.dst

                out_acc.index_add_(0, flat_src, edge)
                in_acc.index_add_(0, flat_dst, edge)
                out_deg.index_add_(0, flat_src, torch.ones_like(out_deg[:edge.shape[0]]))
                in_deg.index_add_(0, flat_dst, torch.ones_like(in_deg[:edge.shape[0]]))

                out_mean = out_acc / out_deg.clamp_min(1.0)
                in_mean = in_acc / in_deg.clamp_min(1.0)
                node32 = flat_node.float()

                node_next = upd_node(torch.cat([node32, out_mean, in_mean], dim=-1))
                node32 = norm_n(node32 + node_next)

                src_h = node32[flat_src]
                dst_h = node32[flat_dst]
                gate_val = torch.sigmoid(gate(torch.cat([src_h, dst_h, edge_static], dim=-1)))
                edge_next = upd_edge(torch.cat([src_h, dst_h, edge, edge_static], dim=-1))
                edge = norm_e(edge + gate_val * edge_next)

                flat_node = node32

            node_final = flat_node.reshape(bsz, 64, -1)
            edge_score = edge
            board_feat = torch.cat([
                node_final.mean(dim=1),
                node_final.amax(dim=1),
                edge_score.mean(dim=0, keepdim=False).view(1, -1).expand(bsz, -1),  # placeholder; use segmented mean in real code
                edge_score.amax(dim=0, keepdim=False).view(1, -1).expand(bsz, -1),  # placeholder; use segmented max in real code
                torch.zeros(bsz, edge.shape[-1], device=edge.device, dtype=edge.dtype),  # placeholder segmented logsumexp/topk
            ], dim=-1)
            delta = self.readout(board_feat).squeeze(-1)

        return {
            "primitive_delta_raw": delta,
            "edge_count": torch.bincount(graph.edge_batch, minlength=bsz),
        }
```

The important properties in that sketch are not the exact layer sizes. They are the architectural guardrails: topology compiled under `no_grad`, no `.view()` on uncertain layouts, fp32 aggregation islands, packed edges instead of per-position move objects, and a reusable core that can be wrapped by a gated head. Those are the parts most likely to determine whether this primitive survives contact with the repo’s training loop. citeturn6view2turn6view6turn6view7turn7view0

## Validation

### Falsifiers

`p047` should be considered **falsified**, not merely “underperforming,” if any of the following hold:

- **Topology falsifier.** Replace the compiled candidate graph with a random graph preserving per-position edge counts and move-mode histogram. If performance holds, the move graph was not load-bearing. This is the natural successor to `p009`’s `random_typed_edges` falsifier. fileciteturn22file0L3-L3
- **Delta falsifier.** Keep the candidate graph but zero the pressure-delta block. If performance barely moves, the primitive is not really a “move-graph delta” model; it is only a move-graph model.
- **Tactical-tag falsifier.** Remove `gives_check`, pin-line, capture, and promotion features while keeping the graph. If that ablation matches full `p047`, the tactical annotations are ornamental.
- **Attachment falsifier.** If the primitive only shows gains in a BT4 mixer adapter but not as a native board-aware head, then it has not solved the problem stated here; it has merely found a niche inside one tower family.
- **Benchmark falsifier.** On the canonical puzzle_binary split, promotion should require at minimum beating or matching the current reference thresholds documented in the benchmark goal, with a specific focus on reducing near-puzzle false positives rather than buying gains only by becoming too conservative. fileciteturn32file0L3-L3

A practical scout threshold is to inherit the spirit of `p009`’s original promotion rules and combine them with the benchmark’s harder goalposts: no degradation larger than a few basis points in PR AUC, a clear reduction in near-puzzle false positives, and only modest wall-clock overhead relative to the chosen base trunk. fileciteturn22file0L3-L3 fileciteturn32file0L3-L3

### Smoke-test and promotion-grade plan

The smoke-test suite should be wider than the current legal-move tests, but it should start from the same philosophy: shape checks, gradient-flow checks, registry/build checks, and edge-count sanity are already standard in the existing branch. `p047` should keep those and add correctness oracles for move compilation and tactical tags. fileciteturn9file0L3-L3 fileciteturn10file0L3-L3

The fastest useful smoke set is:

- a **start-position** case, which should emit the expected initial move family and serve as a promotion/castling sanity floor;
- a **pinned-piece** case using the same kind of semantics demonstrated in python-chess, to verify `source_is_pinned` and `move_stays_on_pin_ray`;  
- a **check-giving** case, to verify `gives_check` tagging on pseudo-legal candidates;
- a **promotion** case with at least the four standard promotion variants;
- an **en-passant x-ray** case, because that is the easiest place for a native compiler to lie;
- a **non-contiguous tensor** case, where input has been permuted or sliced before the model sees it, to verify the absence of hidden `.view()` assumptions;
- an **AMP forward/backward** case, using autocast plus `BCEWithLogitsLoss`, requiring finite logits, finite grads, and no dtype-mismatch errors. citeturn6view0turn8view1turn9view0turn6view2turn7view0

The promotion-grade plan should then run in four gates.

**Correctness gate.** Cross-check the candidate compiler against python-chess on a large random FEN sample. Use python-chess only as the oracle. Compare move sets, pin tags, and check tags separately. Its docs give the relevant semantics: legal vs pseudo-legal distinction, `is_pinned`, and `gives_check`. Pinned pieces still counting as attackers should be tested explicitly because that is exactly the kind of subtle bug that silently distorts tactical pressure features. citeturn6view0turn6view1turn8view1turn9view0

**Reference-parity gate.** Maintain both dense-reference and packed-fast-path compilers and require them to match on topology exactly and on floating features within tolerance. Because PyTorch explicitly warns that floating-point arithmetic is non-associative and that batched computation is not bitwise identical to slice-by-slice computation, toleranced parity is the right criterion. citeturn17view0turn17view1

**Benchmark gate.** Run the canonical puzzle_binary protocol with repeated seeds against the current reference baseline. Report not just accuracy and F1, but the rectangular diagnostic emphasized in the benchmark docs: random non-puzzle false positives, near-puzzle false positives, puzzle recall, F1, and PR AUC. The near-puzzle row is the whole point. fileciteturn32file0L3-L3

**Operational gate.** Profile compile time, aggregation time, and peak memory in three modes: fp32 dense reference, AMP packed fast path, and AMP packed fast path with reduced-precision GEMM reductions disabled for debugging. If the packed path cannot stay near the intended wall-clock envelope, the idea may still be scientifically valid, but it is not yet promotion-grade engineering. citeturn17view2

The bottom line is simple: **`p047` should be promoted only if it proves that true puzzle positions contain a stronger and more coherent edge-level tactical delta signature than near-puzzles, and if it proves that with a compiler and message-passing stack that is robust under AMP, robust under contiguous-layout changes, and fast enough to be a first-class puzzle_binary primitive rather than another fragile research side branch.** fileciteturn32file0L3-L3