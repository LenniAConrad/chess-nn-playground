# Codex Handoff Packet: Fisher-Geodesic Tension Network

## 1. File Metadata

- **Filename:** `chess_nn_research_2026-04-28_0755_tuesday_new_york_fisher_geodesic.md`
- **Created:** 2026-04-28 07:55 new_york
- **Weekday:** Tuesday
- **Short slug:** `fisher_geodesic`
- **Task family:** Chess neural-network research packet
- **Primary idea:** Detect tactical-puzzle positions by measuring Fisher-Rao geodesic excess among board-derived categorical square distributions.
- **Required input contract:** `x.shape == (batch, C, 8, 8)`
- **Required output contract:** one scalar logit per board, `logit.shape == (batch,)` or `(batch, 1)`
- **Binary target contract:** `fine in {0, 1} -> y = 0`; `fine == 2 -> y = 1`
- **Mandatory diagnostic:** report false positives separately on near-puzzle examples, meaning examples with `fine == 1` but binary target `0`.
- **Forbidden inputs:** Stockfish scores, principal variations, node counts, mate scores, best moves, verification metadata, source labels, and source identity.

## 2. Executive Selection

Build a **Fisher-Geodesic Tension Network**: a small convolutional board encoder that maps each board into several learned categorical distributions over the 64 squares. Each distribution is produced only from the board tensor. For every route, the network forms a three-point path on the square-probability simplex:

```text
source distribution -> hinge/tension distribution -> sink/criticality distribution
```

The information-geometric object is the **Fisher-Rao geodesic excess** of that path on the categorical simplex. The core scalar is:

```text
E(p, h, q) = d_FR(p, h) + d_FR(h, q) - d_FR(p, q)
```

where `p`, `h`, and `q` are board-derived categorical distributions over squares, and `d_FR` is Fisher-Rao distance. The quantity is nonnegative by the triangle inequality. It is near zero when the hinge distribution lies along a smooth Fisher geodesic between source and sink. It grows when board pressure is forced through a sharply bent, concentrated, or tactically awkward intermediate distribution.

The thesis is that real puzzle positions often contain a localized tactical hinge: a square, piece relation, or forcing region that does not look like a smooth continuation of ordinary material pressure toward ordinary king or target pressure. Near-puzzles may have pressure, checks, material imbalance, or tempting motifs, but they should produce more diffuse or less structurally aligned geodesic excess. The model is not asked to explain itself with an engine line; it is asked to learn whether the board-induced information geometry has the shape of a tactical singularity.

This is not an uncertainty head, not calibration, not ordinal evidence, not a pseudo-likelihood ratio, not a masked codec, and not generic contrastive learning. It is a supervised binary classifier whose inductive bias is an explicit Fisher-Rao geometry over square distributions.

## 3. Data Contract

### Input

```python
x: torch.Tensor  # shape [B, C, 8, 8], dtype float32 or float16
fine: torch.Tensor  # shape [B], integer labels in {0, 1, 2}
```

The model must treat `C` as configurable. Do not hard-code a particular channel count unless the training repository already defines one. The architecture may use all channels as board planes but must not add forbidden channels.

### Binary label mapping

```python
y = (fine == 2).float()
```

- `fine == 0`: ordinary negative
- `fine == 1`: near-puzzle negative
- `fine == 2`: positive puzzle

The model trains on the binary target only. The `fine == 1` label is retained for stratified reporting, not for an ordinal objective.

### Output

```python
logit: torch.Tensor  # shape [B] or [B, 1]
```

Use `BCEWithLogitsLoss` for the main objective.

### Required validation report

At minimum, every validation run must print:

```text
auc_roc_all
auc_pr_all
accuracy_at_threshold
precision_at_threshold
recall_at_threshold
fpr_fine0
fpr_fine1_near_puzzle
fpr_all_binary_negative
positive_rate_fine0
positive_rate_fine1
positive_rate_fine2
```

The near-puzzle false-positive rate is:

```python
pred = (torch.sigmoid(logit) >= threshold)
near_mask = (fine == 1)
fpr_fine1_near_puzzle = (pred[near_mask].float().mean()).item()
```

The threshold may be fixed at `0.5` for comparability, but the report should also include a validation-selected operating threshold when the project already uses one. Do not tune the threshold on test data.

### Forbidden-input guard

The dataloader or training script should assert that no batch contains engine or provenance artifacts. Suggested defensive check:

```python
FORBIDDEN_KEYS = {
    "stockfish", "sf", "score", "cp", "centipawn", "pv", "nodes", "mate",
    "best_move", "bestmove", "verification", "verified", "source",
    "source_label", "source_id", "site", "origin"
}

for key in batch.keys():
    k = key.lower()
    assert not any(bad in k for bad in FORBIDDEN_KEYS), f"Forbidden batch key: {key}"
```

This check is not a substitute for dataset review. It is only a guardrail against accidental leakage.

## 4. Information Geometry Background

The model works on the categorical probability simplex over board squares:

```text
Δ^63 = {p in R^64 : p_i > 0, sum_i p_i = 1}
```

A categorical distribution over squares is a natural representation for board-derived attention, pressure, target, or tension mass. The Fisher information metric on the categorical simplex is:

```text
g_p(u, v) = sum_i u_i v_i / p_i
```

for tangent vectors `u` and `v` satisfying `sum_i u_i = sum_i v_i = 0`.

The square-root embedding maps the simplex into the positive orthant of the unit sphere:

```text
φ(p) = sqrt(p)
```

Under this embedding, Fisher-Rao geometry becomes spherical geometry up to a constant factor. A convenient distance is:

```text
d_FR(p, q) = 2 arccos(sum_i sqrt(p_i q_i))
```

Some implementations omit the factor of `2`; this only rescales distances and margins. The packet uses the factor of `2` for the canonical Fisher-Rao distance.

For a three-point path `(p, h, q)`, define Fisher-Rao geodesic excess:

```text
E(p, h, q) = d_FR(p, h) + d_FR(h, q) - d_FR(p, q)
```

This is zero when `h` lies on a shortest geodesic from `p` to `q`, and positive when the path bends away from the direct geodesic. In this project:

- `p` is a learned source distribution over squares.
- `h` is a learned hinge/tension distribution over squares.
- `q` is a learned sink/criticality distribution over squares.

All three are derived from the board tensor through the network. No engine information is used.

The product manifold for `R` routes is:

```text
M = (Δ^63)^(3R)
```

with product Fisher metric:

```text
G = direct_sum_{r=1..R, j=1..3} g_{p_{r,j}}
```

The network reads geometric features from this manifold and combines them with normal convolutional board features to produce one logit.

## 5. Candidate Search Trace

### Candidate A: Fisher distance to a quiet-position prototype

Idea: learn a Riemannian barycenter of negative examples and classify by Fisher-Rao distance from that quiet prototype.

Why not selected: it is simple but too global. A puzzle is not merely far from a negative prototype. Many sharp-looking near-puzzles would also be far from quiet positions. It risks becoming a novelty detector rather than a tactical-structure detector.

### Candidate B: Natural-gradient norm of a learned square-energy field

Idea: construct an energy distribution over squares and measure the natural-gradient norm induced by Fisher geometry.

Why not selected: the natural-gradient norm is mathematically clean, but by itself it only says that the board-derived distribution changes steeply. It does not distinguish a direct, obvious pressure pattern from a bent tactical hinge.

### Candidate C: Bregman divergence between attack and defense square distributions

Idea: produce attack and defense distributions and classify from their KL or generalized Bregman divergence.

Why not selected: divergence between two distributions can detect mismatch, but chess tactics often involve a three-part relation: source pressure, forcing hinge, and final sink. A two-point divergence loses the intermediate hinge structure.

### Candidate D: Fisher-Rao geodesic curvature or excess over source-hinge-sink paths

Idea: produce three distributions per route and measure how much the path bends on the Fisher simplex.

Why selected: it directly encodes the idea that a puzzle contains a forcing hinge that makes ordinary board pressure geometrically non-smooth. It gives interpretable route-level scalars, remains fully board-derived, and gives a direct way to audit near-puzzle false positives.

Selected implementation: **Fisher-Rao geodesic excess**, optionally accompanied by the spherical angle at the hinge as a secondary feature.

## 6. Rejected Common Approaches

Do not implement any of these as the main idea or auxiliary objective:

1. **Plain uncertainty heads**: no variance head, confidence head, abstention head, or entropy-as-uncertainty objective.
2. **Calibration framing**: no temperature-scaling research packet, no ECE-centered objective, no calibration-only contribution.
3. **Credal evidence**: no Dirichlet evidence network, belief mass, or evidential uncertainty head.
4. **Ordinal evidence**: do not train `fine == 0`, `fine == 1`, and `fine == 2` as ordered targets. The binary mapping is mandatory.
5. **Pseudo-likelihood ratios**: do not frame the output as a likelihood ratio between puzzle and non-puzzle generators unless a true generative model is built and validated.
6. **Masked codecs**: no masked board reconstruction objective, no autoencoding-as-pretraining contribution.
7. **Generic contrastive learning**: no SimCLR-style or pairwise contrastive framing as the core idea.
8. **Engine leakage**: no Stockfish scores, PVs, nodes, mate scores, best moves, verification metadata, source labels, or source identity.
9. **Source shortcut detection**: no domain-adversarial or source-ID removal story, because source identity is forbidden as an input and should not be present.
10. **Move-solution supervision**: no best-move labels, no tactic-line labels, and no move-ranking loss.

## 7. Mathematical Thesis

Let `x` be a board tensor. The network maps `x` to `R` source-hinge-sink paths on the categorical square simplex:

```text
F_θ(x) = {(p_r(x), h_r(x), q_r(x))}_{r=1..R}
```

where each element is in `Δ^63`.

The thesis is:

```text
A position is puzzle-like when at least one board-derived route has high Fisher-Rao geodesic excess and the global board features support that excess as tactically meaningful rather than merely noisy or diffuse.
```

Formally, define route excess:

```text
E_r(x) = d_FR(p_r(x), h_r(x)) + d_FR(h_r(x), q_r(x)) - d_FR(p_r(x), q_r(x))
```

and route directness ratio:

```text
ρ_r(x) = E_r(x) / (d_FR(p_r(x), q_r(x)) + ε)
```

A high `E_r` says the hinge is not on the direct Fisher geodesic between source and sink. A high `ρ_r` says the bend is large relative to endpoint separation. The readout network learns when these geometric patterns imply `fine == 2` under the binary target.

Important nuance: the model should not classify every high-excess board as a puzzle. Some near-puzzles may contain sharp-looking pressure but lack actual forcing structure. This is why near-puzzle false positives are a required metric. The desired model should increase positive recall without letting `fine == 1` become the main false-positive sink.

## 8. Manifold/Metric Object

### Object name

**Fisher-Rao route manifold over board-square distributions**

### Base manifold

```text
Δ^63_ε = {p in R^64 : p_i >= ε, sum_i p_i = 1}
```

Use a small numerical floor through logits and softmax rather than manually clipping after the fact. A practical implementation is:

```python
p = torch.softmax(square_logits, dim=-1)
p = (1.0 - 64 * eps) * p + eps
```

with `eps = 1e-6` for float32. For mixed precision, use `eps = 1e-5` and compute geometry in float32.

### Product manifold

For `R` routes, each with source, hinge, and sink distributions:

```text
M_R = (Δ^63_ε)^(3R)
```

### Metric

The metric is the product Fisher metric. For each categorical factor:

```text
g_p(u, v) = sum_i u_i v_i / p_i
```

The implementation uses the equivalent square-root sphere embedding.

### Distance

```python
def fisher_rao_distance(p, q, eps=1e-6):
    # p, q: [..., 64], each sums to 1
    p = p.float()
    q = q.float()
    bc = torch.sum(torch.sqrt(p.clamp_min(eps) * q.clamp_min(eps)), dim=-1)
    bc = bc.clamp(min=-1.0 + eps, max=1.0 - eps)
    return 2.0 * torch.acos(bc)
```

Here `bc` is the Bhattacharyya coefficient.

### Geodesic excess

```python
def fisher_geodesic_excess(p, h, q, eps=1e-6):
    d_ph = fisher_rao_distance(p, h, eps)
    d_hq = fisher_rao_distance(h, q, eps)
    d_pq = fisher_rao_distance(p, q, eps)
    return d_ph + d_hq - d_pq
```

### Optional hinge angle

Use the square-root embedding onto the unit sphere. The tangent vector at `h` pointing toward `p` is the spherical log map:

```python
def sphere_log(base, target, eps=1e-6):
    # base, target: [..., 64], unit vectors in sqrt-simplex sphere
    dot = (base * target).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    theta = torch.acos(dot)
    direction = target - dot * base
    norm = direction.norm(dim=-1, keepdim=True).clamp_min(eps)
    return theta * direction / norm
```

Then:

```python
u_h = torch.sqrt(h.clamp_min(eps))
u_p = torch.sqrt(p.clamp_min(eps))
u_q = torch.sqrt(q.clamp_min(eps))

v_hp = sphere_log(u_h, u_p, eps)
v_hq = sphere_log(u_h, u_q, eps)

cos_angle = torch.sum(v_hp * v_hq, dim=-1) / (
    v_hp.norm(dim=-1).clamp_min(eps) * v_hq.norm(dim=-1).clamp_min(eps)
)
angle = torch.acos(cos_angle.clamp(-1 + eps, 1 - eps))
turn = torch.pi - angle
```

`turn` is near zero for a straight path and larger for a sharp bend. Geodesic excess should remain the primary object because it is simpler, stable, and directly distance-based.

## 9. Architecture Tensor Contract

### Recommended module name

```text
models/fisher_geodesic_tension_net.py
```

### Top-level class

```python
class FisherGeodesicTensionNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        width: int = 96,
        depth: int = 5,
        routes: int = 8,
        eps: float = 1e-6,
        use_angle: bool = True,
    ):
        ...

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        ...
```

### Input/output behavior

```python
x: [B, C, 8, 8]
logit: [B]
aux: dict  # only if return_aux=True
```

The auxiliary dictionary should contain detached or raw tensors for analysis:

```python
aux = {
    "route_probs": probs,              # [B, R, 3, 64]
    "excess": excess,                  # [B, R]
    "direct_distance": d_pq,           # [B, R]
    "route_ratio": ratio,              # [B, R]
    "hinge_turn": turn,                # [B, R], if use_angle=True
    "route_gate": route_gate,          # [B, R]
    "geometry_features": geom_feat,    # [B, G]
}
```

### Trunk

Use a compact residual convolutional trunk. The board is only `8x8`, so avoid architectures that downsample away square identity.

Suggested block:

```python
class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))
```

Top trunk:

```python
stem = nn.Sequential(
    nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
    nn.BatchNorm2d(width),
    nn.SiLU(inplace=True),
)
blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
```

### Distribution heads

Produce `R * 3` distributions over `64` squares:

```python
route_head = nn.Conv2d(width, routes * 3, kernel_size=1)
route_logits = route_head(h)                  # [B, R*3, 8, 8]
route_logits = route_logits.view(B, R, 3, 64)
probs = torch.softmax(route_logits, dim=-1)   # [B, R, 3, 64]
probs = (1.0 - 64 * eps) * probs + eps
```

Interpretation per route:

```text
probs[:, r, 0, :] = source distribution p_r
probs[:, r, 1, :] = hinge/tension distribution h_r
probs[:, r, 2, :] = sink/criticality distribution q_r
```

Do not supervise these distributions with engine moves or solution squares.

### Geometry feature construction

For each route:

```python
p = probs[:, :, 0, :]
hh = probs[:, :, 1, :]
q = probs[:, :, 2, :]

d_ph = fisher_rao_distance(p, hh, eps)  # [B, R]
d_hq = fisher_rao_distance(hh, q, eps)  # [B, R]
d_pq = fisher_rao_distance(p, q, eps)   # [B, R]
excess = d_ph + d_hq - d_pq             # [B, R]
ratio = excess / (d_pq + eps)           # [B, R]
```

Aggregate route features with a learned route gate derived from board features, not labels:

```python
pooled = h.mean(dim=(2, 3))              # [B, width]
route_gate = torch.softmax(route_gate_mlp(pooled), dim=-1)  # [B, R]

weighted_excess = (route_gate * excess).sum(dim=-1, keepdim=True)
max_excess = excess.max(dim=-1, keepdim=True).values
weighted_ratio = (route_gate * ratio).sum(dim=-1, keepdim=True)
max_ratio = ratio.max(dim=-1, keepdim=True).values
```

Concatenate:

```python
geom_feat = torch.cat([
    excess,
    ratio,
    weighted_excess,
    max_excess,
    weighted_ratio,
    max_ratio,
    d_ph,
    d_hq,
    d_pq,
], dim=-1)
```

If using hinge angle:

```python
geom_feat = torch.cat([geom_feat, turn, weighted_turn, max_turn], dim=-1)
```

### Readout

Use both board features and geometry features:

```python
readout_input = torch.cat([pooled, geom_feat], dim=-1)
logit = readout_mlp(readout_input).squeeze(-1)
```

Suggested MLP:

```python
readout_mlp = nn.Sequential(
    nn.LayerNorm(width + geom_dim),
    nn.Linear(width + geom_dim, width),
    nn.SiLU(inplace=True),
    nn.Dropout(0.10),
    nn.Linear(width, width // 2),
    nn.SiLU(inplace=True),
    nn.Linear(width // 2, 1),
)
```

### Geometry bottleneck variant

To verify that the information geometry matters, include a stricter variant:

```python
logit = geometry_only_mlp(geom_feat).squeeze(-1)
```

This should be an ablation, not necessarily the production model. If the full model improves but the geometry-only model is useless, the geometry heads may be decorative. If geometry-only performs competitively and full model improves further, the idea is more credible.

## 10. Objective Function

### Main binary objective

```python
y = (fine == 2).float()
loss_bce = F.binary_cross_entropy_with_logits(logit, y)
```

### Geodesic-margin auxiliary objective

Use only the binary target. Do not use `fine == 0` versus `fine == 1` as an ordinal target.

Let:

```python
route_score = weighted_excess.squeeze(-1)
```

Use a soft margin:

```python
m_pos = 0.20
m_neg = 0.08
loss_geo_pos = y * F.softplus(m_pos - route_score)
loss_geo_neg = (1.0 - y) * F.softplus(route_score - m_neg)
loss_geo = (loss_geo_pos + loss_geo_neg).mean()
```

This does not require route excess to be the whole classifier. It nudges positives to contain at least one meaningful bend and negatives to avoid excessive geometric bends.

### Route anti-collapse regularizer

Routes should not all learn the same distribution triple. Use a mild within-example decorrelation penalty in square-root coordinates. This is not a contrastive objective across examples; it is a same-board anti-collapse constraint over route heads.

```python
def route_decorrelation_loss(probs, eps=1e-6):
    # probs: [B, R, 3, 64]
    u = torch.sqrt(probs.clamp_min(eps))       # [B, R, 3, 64]
    u = u.flatten(2)                           # [B, R, 192]
    u = F.normalize(u, dim=-1)
    gram = torch.matmul(u, u.transpose(1, 2))  # [B, R, R]
    R = gram.size(-1)
    eye = torch.eye(R, device=gram.device, dtype=gram.dtype).unsqueeze(0)
    off_diag = gram * (1.0 - eye)
    return off_diag.pow(2).mean()
```

Keep this small. It is only to prevent all routes from becoming duplicates.

### Numerical barrier

The softmax floor already prevents exact zeros. No entropy or uncertainty objective is needed. If training still collapses distributions too sharply, prefer logit weight decay and a lower learning rate over entropy regularization.

### Total loss

```python
loss = loss_bce + lambda_geo * loss_geo + lambda_decorr * loss_decorr
```

Recommended initial weights:

```text
lambda_geo = 0.10
lambda_decorr = 0.01
```

Training schedule:

1. Epochs 1-3: `lambda_geo = 0.00`, `lambda_decorr = 0.01`
2. Epochs 4 onward: `lambda_geo = 0.10`, `lambda_decorr = 0.01`
3. If validation near-puzzle FPR rises sharply, reduce `lambda_geo` to `0.03` before changing architecture.

### Optimizer

```text
AdamW
learning_rate = 3e-4
weight_decay = 1e-4
batch_size = as large as memory permits
mixed_precision = allowed, but compute Fisher geometry in float32
```

### Class imbalance

If positives are rare, use `pos_weight` in `BCEWithLogitsLoss`, but report unweighted metrics. Do not oversample near-puzzles as positives. They are binary negatives.

## 11. Ablations

Run these ablations before claiming the idea works.

### A0: Plain CNN baseline

Same trunk and readout capacity, but no distribution heads and no Fisher geometry.

Purpose: test whether the proposed geometry adds value over a normal board CNN.

### A1: Full Fisher-Geodesic Tension Network

Trunk plus route distributions plus Fisher-Rao geodesic excess features plus BCE and small auxiliary geometry margin.

Purpose: main candidate.

### A2: Geometry-only readout

Remove pooled convolutional features from the final readout. Use only geometric features.

Purpose: test whether the geometric object itself carries signal.

### A3: Euclidean replacement

Replace Fisher-Rao distances with Euclidean distances between raw probability vectors:

```text
||p - h||_2 + ||h - q||_2 - ||p - q||_2
```

Purpose: test whether Fisher geometry matters or any distance would work.

### A4: Hellinger no-geodesic two-point model

Use only two-point Fisher/Hellinger distances such as `d_FR(p, q)`, removing the hinge and excess.

Purpose: test whether the three-point path is necessary.

### A5: No geometry margin

Use the full architecture but train with BCE only.

Purpose: test whether the auxiliary margin helps or hurts.

### A6: Route count sweep

Test:

```text
R in {2, 4, 8, 12, 16}
```

Expected: too few routes underfit tactical diversity; too many routes may overfit and increase near-puzzle false positives.

### A7: Near-puzzle stress report

Do not train differently. Only evaluate:

```text
fpr_fine0
fpr_fine1_near_puzzle
fpr_fine1 / fpr_fine0 ratio
```

Expected: near-puzzle FPR will be higher than ordinary-negative FPR. The model is useful only if this gap is controlled while positive recall improves.

### A8: Geometry randomization test

At validation time, permute route geometric features across examples while keeping pooled CNN features attached to the original board.

Purpose: if metrics do not change, the readout is ignoring geometry.

### A9: Square permutation sanity failure

Apply a fixed random permutation to the 64 square indices inside geometry calculations only, leaving CNN features unchanged.

Purpose: this should degrade the geometric contribution. If it does not, the geometry object is not learning board-square structure.

## 12. Falsification

The idea should be considered falsified or weakened under any of the following outcomes.

### F1: No gain over capacity-matched CNN

If A1 does not improve over A0 on AUC-PR, recall at fixed precision, or another predeclared metric, the geometry is not earning its complexity.

### F2: Near-puzzle false positives explode

If A1 improves positive recall only by predicting many `fine == 1` near-puzzles as positive, the model is not solving the intended problem. A useful result must report and control:

```text
fpr_fine1_near_puzzle
fpr_fine1 / fpr_fine0
```

A rough initial red flag:

```text
fpr_fine1_near_puzzle > 2.5 * fpr_fine0
```

This threshold is not universal, but it forces honest reporting.

### F3: Euclidean replacement matches Fisher-Rao

If A3 matches or beats A1 across repeated seeds, the information-geometric choice is not supported. The project may still have a useful architecture, but the Fisher-Rao claim should be dropped.

### F4: Geometry-only readout is random

If A2 is near random and A1 performs well, the final model may be using the CNN trunk while ignoring geometry. Check gradient norms into route heads and run A8.

### F5: Geometry features fail randomization tests

If permuting route geometry across validation examples does not affect A1, the route object is decorative.

### F6: Distribution heads collapse

If all routes produce identical or nearly uniform distributions, the manifold object is not being used. Look at route pair Gram matrices and per-route excess histograms.

### F7: Forbidden-input dependence is discovered

If any forbidden field entered training or validation, discard the run. Do not try to patch metrics after the fact.

## 13. Implementation Plan

### Step 1: Add metrics utility

Create or update:

```text
metrics/puzzle_binary_metrics.py
```

Implement:

```python
def compute_binary_puzzle_metrics(logits, fine, threshold=0.5):
    y = (fine == 2).float()
    prob = torch.sigmoid(logits.float())
    pred = prob >= threshold

    out = {}
    for value, name in [(0, "fine0"), (1, "fine1_near_puzzle"), (2, "fine2")]:
        mask = fine == value
        if mask.any():
            out[f"positive_rate_{name}"] = pred[mask].float().mean().item()
            out[f"mean_prob_{name}"] = prob[mask].mean().item()
        else:
            out[f"positive_rate_{name}"] = float("nan")
            out[f"mean_prob_{name}"] = float("nan")

    neg_mask = fine != 2
    fine0_mask = fine == 0
    fine1_mask = fine == 1

    out["fpr_all_binary_negative"] = pred[neg_mask].float().mean().item() if neg_mask.any() else float("nan")
    out["fpr_fine0"] = pred[fine0_mask].float().mean().item() if fine0_mask.any() else float("nan")
    out["fpr_fine1_near_puzzle"] = pred[fine1_mask].float().mean().item() if fine1_mask.any() else float("nan")
    return out
```

Add AUC-ROC and AUC-PR through the repository's existing metric stack or `sklearn.metrics` if available.

### Step 2: Implement Fisher geometry functions

Create:

```text
models/geometry/fisher_square_simplex.py
```

Include:

```python
import torch
import torch.nn.functional as F


def simplex_floor(p, eps=1e-6):
    n = p.size(-1)
    return (1.0 - n * eps) * p + eps


def fisher_rao_distance(p, q, eps=1e-6):
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    bc = torch.sqrt(p * q).sum(dim=-1)
    bc = bc.clamp(-1.0 + eps, 1.0 - eps)
    return 2.0 * torch.acos(bc)


def fisher_geodesic_excess(p, h, q, eps=1e-6):
    d_ph = fisher_rao_distance(p, h, eps)
    d_hq = fisher_rao_distance(h, q, eps)
    d_pq = fisher_rao_distance(p, q, eps)
    excess = d_ph + d_hq - d_pq
    return excess, d_ph, d_hq, d_pq


def sphere_log(base, target, eps=1e-6):
    base = F.normalize(base.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    dot = (base * target).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)
    direction = target - dot * base
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(eps)
    return theta * direction


def hinge_turn(p, h, q, eps=1e-6):
    u_p = torch.sqrt(p.float().clamp_min(eps))
    u_h = torch.sqrt(h.float().clamp_min(eps))
    u_q = torch.sqrt(q.float().clamp_min(eps))
    v_hp = sphere_log(u_h, u_p, eps)
    v_hq = sphere_log(u_h, u_q, eps)
    denom = v_hp.norm(dim=-1).clamp_min(eps) * v_hq.norm(dim=-1).clamp_min(eps)
    cos_angle = (v_hp * v_hq).sum(dim=-1) / denom
    angle = torch.acos(cos_angle.clamp(-1.0 + eps, 1.0 - eps))
    return torch.pi - angle
```

### Step 3: Implement model

Create:

```text
models/fisher_geodesic_tension_net.py
```

Skeleton:

```python
class FisherGeodesicTensionNet(nn.Module):
    def __init__(self, in_channels, width=96, depth=5, routes=8, eps=1e-6, use_angle=True):
        super().__init__()
        self.routes = routes
        self.eps = eps
        self.use_angle = use_angle

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
        self.route_head = nn.Conv2d(width, routes * 3, 1)
        self.route_gate = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, routes),
        )

        base_geom_dim = routes * 6 + 4
        # excess, ratio, d_ph, d_hq, d_pq, optional turn per route = handled below
        if use_angle:
            geom_dim = routes * 6 + 6
        else:
            geom_dim = routes * 5 + 4

        self.readout = nn.Sequential(
            nn.LayerNorm(width + geom_dim),
            nn.Linear(width + geom_dim, width),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(width, width // 2),
            nn.SiLU(inplace=True),
            nn.Linear(width // 2, 1),
        )
```

Be careful to compute `geom_dim` exactly from the concatenated feature list. Add a unit test that instantiates the model and performs a forward pass for several `routes` values.

Forward pass outline:

```python
def forward(self, x, return_aux=False):
    B = x.size(0)
    h = self.blocks(self.stem(x))
    pooled = h.mean(dim=(2, 3))

    logits = self.route_head(h).view(B, self.routes, 3, 64)
    probs = torch.softmax(logits.float(), dim=-1)
    probs = simplex_floor(probs, self.eps)

    p = probs[:, :, 0, :]
    mid = probs[:, :, 1, :]
    q = probs[:, :, 2, :]

    excess, d_ph, d_hq, d_pq = fisher_geodesic_excess(p, mid, q, self.eps)
    ratio = excess / (d_pq + self.eps)

    gate = torch.softmax(self.route_gate(pooled), dim=-1)
    weighted_excess = (gate * excess).sum(dim=-1, keepdim=True)
    max_excess = excess.max(dim=-1, keepdim=True).values
    weighted_ratio = (gate * ratio).sum(dim=-1, keepdim=True)
    max_ratio = ratio.max(dim=-1, keepdim=True).values

    geom_parts = [
        excess, ratio, d_ph, d_hq, d_pq,
        weighted_excess, max_excess, weighted_ratio, max_ratio,
    ]

    if self.use_angle:
        turn = hinge_turn(p, mid, q, self.eps)
        weighted_turn = (gate * turn).sum(dim=-1, keepdim=True)
        max_turn = turn.max(dim=-1, keepdim=True).values
        geom_parts.extend([turn, weighted_turn, max_turn])
    else:
        turn = None

    geom_feat = torch.cat(geom_parts, dim=-1)
    readout_input = torch.cat([pooled, geom_feat.to(pooled.dtype)], dim=-1)
    logit = self.readout(readout_input).squeeze(-1)

    if not return_aux:
        return logit

    aux = {
        "route_probs": probs,
        "excess": excess,
        "direct_distance": d_pq,
        "route_ratio": ratio,
        "route_gate": gate,
        "geometry_features": geom_feat,
    }
    if turn is not None:
        aux["hinge_turn"] = turn
    return logit, aux
```

### Step 4: Add training loss wrapper

Create:

```text
losses/fisher_geodesic_loss.py
```

Implement:

```python
def fisher_geodesic_training_loss(logit, fine, aux, pos_weight=None, lambda_geo=0.10, lambda_decorr=0.01):
    y = (fine == 2).float()
    bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)

    gate = aux["route_gate"]
    excess = aux["excess"]
    route_score = (gate * excess).sum(dim=-1)

    m_pos = 0.20
    m_neg = 0.08
    geo = (y * F.softplus(m_pos - route_score) + (1.0 - y) * F.softplus(route_score - m_neg)).mean()

    decorr = route_decorrelation_loss(aux["route_probs"])
    loss = bce + lambda_geo * geo + lambda_decorr * decorr

    logs = {
        "loss_bce": bce.detach(),
        "loss_geo": geo.detach(),
        "loss_decorr": decorr.detach(),
        "route_score_mean": route_score.detach().mean(),
    }
    return loss, logs
```

If the repository's training loop expects only logits, make `return_aux=True` during training for this model type.

### Step 5: Add leakage checks

In the dataset or collate function, reject forbidden fields. Also inspect configuration files for any references to engine outputs. This model must stand on board tensors and labels only.

### Step 6: Add validation tables

Validation output should include a compact table like:

```text
split      auc_pr  auc_roc  fpr_fine0  fpr_fine1_near  recall_fine2  threshold
valid      ...     ...      ...        ...             ...           0.500
valid@op   ...     ...      ...        ...             ...           selected
test       ...     ...      ...        ...             ...           frozen
```

Also log route summaries:

```text
mean_excess_fine0
mean_excess_fine1
mean_excess_fine2
max_excess_fine0
max_excess_fine1
max_excess_fine2
mean_weighted_ratio_by_fine
```

These summaries are not training targets. They are diagnostics.

### Step 7: Unit tests

Add tests:

```text
tests/test_fisher_square_simplex.py
tests/test_fisher_geodesic_tension_net.py
```

Required checks:

```python
# distance symmetry
d(p, q) == d(q, p)

# zero-ish distance
d(p, p) < tolerance

# geodesic excess nonnegative within numerical tolerance
E(p, h, q) >= -1e-5

# model forward shape
logit.shape == (B,)

# aux shape
aux["route_probs"].shape == (B, R, 3, 64)
aux["excess"].shape == (B, R)

# no NaNs under autocast-style input
```

### Step 8: Run seed set

Use at least three seeds:

```text
seeds = [1, 2, 3]
```

Report mean and standard deviation for:

```text
AUC-PR
AUC-ROC
recall at fixed precision
fpr_fine1_near_puzzle
fpr_fine1 / fpr_fine0
```

### Step 9: Interpret route maps

For debugging only, visualize the source, hinge, and sink distributions as 8x8 heatmaps for true positives, false positives on `fine == 0`, and false positives on `fine == 1`. Do not turn those visualizations into new labels. The purpose is to find collapse, leakage, or spurious board regions.

## 14. Prompt Maintenance

Preserve these invariants in future revisions:

1. **One-logit output:** the model returns one binary logit per board.
2. **Binary target mapping:** `fine == 0` and `fine == 1` are both negative; `fine == 2` is positive.
3. **Near-puzzle reporting:** false positives on `fine == 1` must be reported separately.
4. **No engine leakage:** never use Stockfish scores, PVs, node counts, mate scores, best moves, verification metadata, source labels, or source identity.
5. **Information-geometric core:** the research idea must remain anchored to Fisher-Rao geometry on board-derived square distributions.
6. **No forbidden framing drift:** do not rewrite this as uncertainty estimation, calibration, credal evidence, ordinal evidence, pseudo-likelihood ratios, masked codecs, or generic contrastive learning.
7. **Geometry must be falsifiable:** always include capacity-matched CNN, Euclidean replacement, geometry-only readout, and near-puzzle stress tests.
8. **Board-derived only:** every distribution on the simplex must be produced from `x` through the network or a deterministic transform of `x`; no move solution or engine annotation may supervise the route distributions.
9. **Codex implementation target:** keep tensor shapes explicit and add unit tests for distance, excess, model outputs, and metric reporting.
10. **Do not hide failures:** if Fisher-Rao geometry does not beat Euclidean or CNN baselines, report that directly and simplify the model.

The final deliverable is a supervised chess puzzle classifier with an explicit information-geometric bottleneck: source-hinge-sink paths on the Fisher simplex of board squares. Its success criterion is not just better aggregate classification. It must improve puzzle detection while making near-puzzle false positives visible and controlled.
