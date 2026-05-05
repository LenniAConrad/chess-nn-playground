# Codex Handoff Packet: Loop-Frustration Curvature Network

## 1. File Metadata

- **Filename:** `chess_nn_research_2026-04-28_0729_tuesday_new_york_frustration_curvature.md`
- **Timestamp:** 2026-04-28 07:29 new_york
- **Idea name:** Loop-Frustration Curvature Network, abbreviated `LFCN`
- **Task:** classify true chess puzzles versus near-puzzles from current-board tensor features only.
- **Input:** `x` with shape `(batch, C, 8, 8)`.
- **Output:** binary logit with shape `(batch, 1)`.
- **Fine-label mapping:** fine labels `0` and `1` map to target `0`; fine label `2` maps to target `1`.
- **Core observable:** a differentiable spin-glass loop-frustration free-energy curvature score computed on a static board graph.
- **Non-goals:** no engine features, no move search, no best-move supervision, no source/provenance features, and no generic energy-model wrapper around a CNN.

## 2. Executive Selection

Build a network whose decisive layer is not a standard convolutional classifier, but a chess-board spin-glass observable: **loop-frustration curvature**. The model learns site-conditioned pair couplings on a fixed 8x8 board graph, evaluates closed-loop frustration products, estimates a small free-energy curvature by finite differences in inverse temperature, and classifies using only summary statistics and spatial concentration of that observable.

The hypothesis is that true tactical puzzles contain sharper, more localized contradiction structure than near-puzzles. A near-puzzle may look materially or positionally similar, but its learned interaction graph should have weaker or more diffuse closed-loop frustration. The architecture therefore asks: “Does this board induce a concentrated frustrated spin-glass response under temperature perturbation?” rather than “Does a generic CNN recognize puzzle-looking patterns?”

Selected model: `LFCN`, a parameterized current-board spin-glass layer with these components:

1. a small board encoder that only parameterizes local site embeddings;
2. a static edge-and-loop graph over the 64 squares;
3. learned replica couplings `J[e, k]` for edge `e` and spin replica `k`;
4. loop products `P[loop, k] = product_e tanh(beta * J[e, k])`;
5. free-energy loop term `A = log(1 + eta * P)`;
6. finite-difference curvature in `beta`, weighted toward negative loop products;
7. a compact observable head that receives loop-curvature moments, not raw board features.

Central ablation: destroy the closed-loop product while preserving edge-coupling marginals and parameter count. If that control performs similarly, the statistical-physics thesis is false.

## 3. Leakage-Safe Data Contract

### Inputs

`x: FloatTensor[batch, C, 8, 8]`

Allowed channels are current-position board features only. Typical safe channels:

- piece occupancy planes by color and piece type;
- side-to-move broadcast plane, if already part of the current board representation;
- legal state planes that are intrinsic to the current position, such as castling-right or en-passant availability, only if they are consistently available for every example and are not generated from puzzle verification metadata.

Disallowed channels and side data:

- Stockfish scores;
- principal variations;
- node counts;
- mate scores;
- best moves;
- move-count-to-mate labels;
- verification status;
- puzzle source labels;
- source provenance;
- lichess/chess.com/study/import tags;
- any field whose value is available because the example was found or verified as a puzzle.

The model never receives fine labels directly. It receives only the binary target below.

### Labels

```python
# fine_label: LongTensor[batch], values in {0, 1, 2}
target = (fine_label == 2).float().view(batch, 1)
```

- fine `0` -> target `0`
- fine `1` -> target `0`
- fine `2` -> target `1`

### Output

```python
logit = model(x)          # FloatTensor[batch, 1]
prob = sigmoid(logit)     # FloatTensor[batch, 1]
```

### Mandatory 3x2 fine-label diagnostic

Every validation and test report must include this exact matrix:

```python
# rows: fine labels 0, 1, 2
# cols: predicted binary target 0, predicted binary target 1
pred = (torch.sigmoid(logit).view(-1) >= 0.5).long()
fine_diag = torch.zeros(3, 2, dtype=torch.long)
for r in (0, 1, 2):
    for c in (0, 1):
        fine_diag[r, c] = ((fine_label == r) & (pred == c)).sum()
```

Report it as:

| fine label | predicted 0 | predicted 1 |
|---:|---:|---:|
| 0 | `fine_diag[0,0]` | `fine_diag[0,1]` |
| 1 | `fine_diag[1,0]` | `fine_diag[1,1]` |
| 2 | `fine_diag[2,0]` | `fine_diag[2,1]` |

The fine-label diagnostic is required because the binary task merges fine labels `0` and `1`. The most important failure mode is that the model learns to separate only easy non-puzzles from everything else while confusing fine `1` near-puzzles with fine `2` true puzzles.

### Split hygiene

- Deduplicate exact board tensors before splitting.
- If FENs are available upstream, split by canonicalized current position, not by puzzle ID.
- Do not let augmented symmetries of the same position cross train/validation/test boundaries.
- Do not stratify by source or provenance if that information is forbidden to the model; use only target and fine-label counts for split balance.

## 4. Statistical Physics Background

In an Ising spin glass, each site has a spin, and each edge has a coupling. A coupling can prefer aligned or anti-aligned spins. A closed loop is **frustrated** when its couplings cannot all be simultaneously satisfied. For signed Ising couplings, this is captured by the sign of the product of couplings around the loop: a negative product indicates a frustrated loop.

`LFCN` uses this idea without sampling spin states and without running a Boltzmann-pooling classifier. The board encoder learns chess-conditioned edge couplings. A static set of closed board loops then supplies differentiable loop products. The model measures how the loop free-energy contribution changes under a small inverse-temperature perturbation. That temperature curvature behaves like a local susceptibility: weak, diffuse interactions give low curvature; concentrated contradictory interactions give high curvature.

The chess analogy is direct but not engine-dependent. A true puzzle is often a position where several local and line-based constraints collide: king safety, pinned pieces, overloaded defenders, trapped pieces, mating nets, or forced tactical motifs. A near-puzzle may contain similar material and visible threats but lacks the same sharply constrained contradiction. The proposed observable tries to detect the latter distinction through current-board geometry only.

## 5. Candidate Search Trace

Considered physics candidates:

1. **Simple magnetization/order parameter.** Too weak. It can describe global board imbalance but does not directly represent contradictory tactical structure.
2. **Plain susceptibility of a learned scalar field.** Better, but risks becoming a generic smoothness statistic over CNN activations.
3. **High-temperature partition approximation.** Useful, because loop products naturally appear in spin-system expansions and can be computed on a finite board graph.
4. **Spin-glass frustration.** Strong candidate because it is explicitly about mutually inconsistent pair constraints.
5. **Free-energy curvature of frustrated loops.** Selected. It combines spin-glass frustration with a concrete thermodynamic response statistic and gives a localized map that can be falsified against topology-destroying controls.

Final selection: **loop-frustration curvature**. It is concrete, differentiable, implementable in a single PyTorch module, and directly testable against parameter-matched controls.

## 6. Rejected Common Approaches

Rejected for this handoff:

- A generic CNN, ResNet, ConvNeXt, or ViT binary classifier on `(C, 8, 8)`. These may be useful baselines but do not satisfy the request for a statistical-physics idea.
- A generic learned energy head that calls its output “energy” without a physical observable.
- Ordinary Boltzmann pooling over board squares or channels.
- Move-generation features, attack maps computed from legal move generation, engine evaluations, search depths, principal variations, mate-distance labels, best moves, or puzzle verification fields.
- Source/provenance shortcuts, including dataset origin, puzzle collection name, popularity, rating, tags, or verification status.
- The explicitly disallowed families: phase-transition pressure packet, Markov absorption, random walks, percolation, effective resistance, and harmonic-potential constructions.

The accepted idea keeps the statistical-physics object narrow: a finite spin-glass loop bank with a curvature observable.

## 7. Mathematical Thesis

Let `x` be a current-board tensor. The network learns a finite spin-glass graph over the 64 squares with `K` replicas. For square/site `i`, the encoder produces an embedding `g_i(x)`. For each static graph edge `e = (i, j)` and replica `k`, the model produces a coupling:

```text
J_e,k(x) = EdgeMLP_k(g_i, g_j, |g_i - g_j|, g_i * g_j, type(e))
```

Define inverse temperature `beta > 0` and bounded edge response:

```text
U_e,k(beta, x) = tanh(beta * J_e,k(x))
```

For a closed loop `ell` with edge set `E(ell)`:

```text
P_ell,k(beta, x) = product over e in E(ell) of U_e,k(beta, x)
```

`P_ell,k < 0` is the differentiable analogue of a frustrated signed loop. To connect this to free energy, define a stable loop expansion term:

```text
A_ell,k(beta, x) = log(1 + eta * P_ell,k(beta, x))
```

where `0 < eta < 1`, default `eta = 0.90`, so the log argument remains positive because `P in (-1, 1)`.

Define loop curvature by a centered finite difference:

```text
D2A_ell,k(beta, x) = [A_ell,k(beta + delta, x)
                      - 2 * A_ell,k(beta, x)
                      + A_ell,k(beta - delta, x)] / delta^2
```

with default `delta = 0.125` and constrained `beta - delta > 0`.

The physical observable is:

```text
Omega_ell,k(x) = sigmoid(-nu * P_ell,k(beta, x)) * abs(D2A_ell,k(beta, x))
```

where default `nu = 4.0`. The sigmoid factor emphasizes negative loop products, while the curvature term measures thermodynamic sensitivity.

**Thesis:** true puzzles have higher concentrated `Omega` than near-puzzles after controlling for material, occupancy, and generic visual complexity. More precisely, for a trained leakage-safe model, the distribution of board-level summaries of `Omega` should separate fine label `2` from fine labels `0` and `1`, and especially from fine label `1`.

## 8. Physical Observable

Name: **Frustrated Loop Free-Energy Curvature**, abbreviated `FLFEC`.

### Static graph

Use the 64 board squares as sites. Build an undirected edge list using unit orthogonal and unit diagonal board adjacencies:

- horizontal nearest-neighbor edges: `8 * 7 = 56`;
- vertical nearest-neighbor edges: `7 * 8 = 56`;
- down-right diagonal edges: `7 * 7 = 49`;
- down-left diagonal edges: `7 * 7 = 49`.

Total edge count:

```text
M = 210
```

Each edge has a type ID in `{horizontal, vertical, diag_down_right, diag_down_left}`.

### Static loop bank

Use two families of closed loops:

1. **Orthogonal rectangles** with height and width in `{1, 2, 3}`. Each rectangle contributes the unit-edge boundary cycle. Count:

```text
L_rect = sum_{h=1..3} sum_{w=1..3} (8 - h) * (8 - w) = 324
```

2. **Unit-square diagonal triangles.** For each 1x1 square, include the four triangles formed by one side pair and one diagonal. Count:

```text
L_tri = 49 * 4 = 196
```

Total loop count:

```text
L = 520
```

Pad loop edge IDs to `Lmax = 12`, because the largest 3x3 rectangle boundary has length `12`. Use a boolean mask for shorter loops.

### Observable computation

Shapes:

```text
J                  : (B, K, M)
loop_edge_ids       : (L, Lmax)
loop_edge_mask      : (L, Lmax)
U(beta)             : (B, K, M)
U_on_loops          : (B, K, L, Lmax)
P                  : (B, K, L)
A                  : (B, K, L)
Omega              : (B, K, L)
Omega_site          : (B, K, 8, 8)
```

Compute loop products in log-stable form:

```python
u_edges = torch.tanh(beta * J)                         # (B, K, M)
u_loop = gather_edges(u_edges, loop_edge_ids)          # (B, K, L, Lmax)
mask = loop_edge_mask.view(1, 1, L, Lmax)

sign = torch.sign(u_loop).masked_fill(~mask, 1.0)
log_abs = torch.log(u_loop.abs().clamp_min(1e-6)).masked_fill(~mask, 0.0)
P = sign.prod(dim=-1) * torch.exp(log_abs.sum(dim=-1))  # (B, K, L)
```

Then compute:

```python
A = torch.log1p(eta * P).clamp_min(-20.0)
D2A = (A_plus - 2.0 * A_mid + A_minus) / (delta ** 2)
Omega = torch.sigmoid(-nu * P_mid) * D2A.abs()
```

Scatter loop values to their participating vertices to obtain `Omega_site`. This creates a physical saliency map without using search, engine evaluations, or move labels.

Board-level observable vector:

```text
mean_Omega_by_replica        : (B, K)
std_Omega_by_replica         : (B, K)
top8_Omega_by_replica        : (B, K)
max_Omega_by_replica         : (B, K)
frustration_rate_by_replica  : (B, K), mean sigmoid(-nu * P)
spatial_concentration        : (B, K), normalized top8 / mean
EA_order                     : (B, K), mean_i m_i,k^2 - mean_i(m_i,k)^2
```

Concatenate these into `obs: (B, 7K)` and classify with a small MLP.

## 9. Architecture And Tensor Shapes

Default hyperparameters:

```text
C         : provided by dataset
F         : 64 site-embedding channels
K         : 8 spin replicas
M         : 210 static edges
L         : 520 static loops
Lmax      : 12 padded loop length
eta       : 0.90
nu        : 4.00
delta     : 0.125
```

### Forward pass

#### Step 1: board encoder

The encoder is allowed to use convolutions, but it is not allowed to classify directly from its feature map. Its only role is to parameterize spin-glass fields and couplings.

```text
x                  : (B, C, 8, 8)
f0 = Conv1x1       : (B, 64, 8, 8)
f1 = Conv3x3       : (B, 64, 8, 8)
f2 = Conv3x3       : (B, 64, 8, 8)
g = LayerNormSites : (B, 64, 8, 8)
g_flat             : (B, 64, 64) after flattening squares
```

Recommended encoder:

```python
Conv2d(C, 64, kernel_size=1)
GELU()
Conv2d(64, 64, kernel_size=3, padding=1)
GELU()
Conv2d(64, 64, kernel_size=3, padding=1)
GELU()
```

No global pooling from `g` may be fed directly to the final classifier in the main model.

#### Step 2: site spin order

```text
m = tanh(Conv1x1(g))
m shape: (B, K, 8, 8)
```

`m` supplies an Edwards-Anderson-style order statistic and can also be concatenated into edge-pair features.

#### Step 3: edge coupling head

Flatten sites to `N = 64`. Precompute edge endpoint indices:

```text
edge_i: LongTensor[M]
edge_j: LongTensor[M]
edge_type: LongTensor[M]
```

Gather site features:

```text
gi = g_flat[:, :, edge_i].transpose(1, 2)  # (B, M, F)
gj = g_flat[:, :, edge_j].transpose(1, 2)  # (B, M, F)
```

Build pair feature:

```text
pair = concat(gi, gj, abs(gi - gj), gi * gj, edge_type_embedding)
pair shape: (B, M, 4F + T)
```

With `F = 64` and `T = 8`, pair shape is `(B, M, 264)`.

Edge MLP:

```text
Linear(264, 128)
GELU()
Linear(128, K)
tanh scale or spectral clamp
```

Output:

```text
J_raw : (B, M, K)
J     : (B, K, M)
```

Recommended bound:

```python
J = 2.5 * torch.tanh(J_raw.transpose(1, 2) / 2.5)
```

This prevents saturated products early in training.

#### Step 4: loop-frustration curvature layer

Use `J` and the static loop bank to compute `Omega: (B, K, L)` as defined above.

Constrain inverse temperature:

```python
beta = 0.20 + F.softplus(raw_beta)   # scalar or per-replica (K,)
beta = beta.clamp(max=3.0)
```

For finite differences, use:

```python
beta_minus = (beta - delta).clamp_min(0.05)
beta_mid = beta
beta_plus = beta + delta
```

#### Step 5: loop-to-site scatter

Precompute loop vertex IDs:

```text
loop_vertex_ids  : LongTensor[L, Vmax]
loop_vertex_mask : BoolTensor[L, Vmax]
Vmax             : 12
```

Scatter `Omega` equally to loop vertices:

```text
Omega_site_flat : (B, K, 64)
Omega_site      : (B, K, 8, 8)
```

#### Step 6: observable statistics and head

Do not feed raw `g` to the head. Feed only physics-derived statistics:

```text
obs_mean       = mean(Omega_site over squares)                  # (B, K)
obs_std        = std(Omega_site over squares)                   # (B, K)
obs_top8       = mean(topk(Omega_site flattened, k=8))          # (B, K)
obs_max        = max(Omega_site over squares)                   # (B, K)
obs_frust      = mean(sigmoid(-nu * P_mid) over loops)          # (B, K)
obs_conc       = obs_top8 / (obs_mean + 1e-6)                   # (B, K)
obs_ea         = mean(m^2) - mean(m)^2 over squares             # (B, K)
obs            = concat(all above)                              # (B, 7K)
```

Classifier:

```text
Linear(7K, 32)
GELU()
Dropout(0.10)
Linear(32, 1)
```

Output:

```text
logit: (B, 1)
```

## 10. Losses And Regularizers

### Primary loss

Use binary cross-entropy with logits:

```python
loss_bce = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight)
```

`pos_weight` should be computed from the training split only.

### Coupling magnitude regularizer

Prevent trivial saturation of `tanh(beta * J)`:

```python
loss_j = (J ** 2).mean()
```

Default weight: `lambda_j = 1e-4`.

### Replica diversity regularizer

Avoid all `K` replicas learning the same coupling field:

```python
# J_centered: (B, K, M), centered over M
Jc = J - J.mean(dim=-1, keepdim=True)
Jc = F.normalize(Jc, dim=-1)
replica_corr = torch.einsum("bkm,blm->bkl", Jc, Jc)
offdiag = replica_corr - torch.eye(K, device=J.device).view(1, K, K)
loss_replica = (offdiag ** 2).mean()
```

Default weight: `lambda_replica = 1e-3`.

### Curvature smoothness regularizer

Prevent a single numerically unstable loop from dominating:

```python
loss_curv_tail = torch.quantile(Omega.detach(), 0.95)  # threshold only, no gradient
loss_curv = F.relu(Omega - loss_curv_tail).pow(2).mean()
```

Default weight: `lambda_curv = 1e-5`.

### Full loss

```python
loss = loss_bce \
     + lambda_j * loss_j \
     + lambda_replica * loss_replica \
     + lambda_curv * loss_curv
```

Do not add auxiliary losses that predict fine labels, source labels, engine labels, best moves, or mate distances.

## 11. Ablations

### Central ablation: no closed-loop frustration

Name: `LFCN-NoLoopProduct`.

Keep the board encoder, edge MLP, replica count, beta parameters, observable MLP size, and optimizer unchanged. Replace each loop product with an open-chain magnitude statistic:

```text
P_ell,k(beta, x) := mean over e in E(ell) of abs(tanh(beta * J_e,k(x)))
```

Then compute the same finite-difference machinery on this non-frustrated surrogate. This ablation preserves edge strength, temperature response, and almost all compute, but removes signed closed-loop inconsistency.

Expected outcome if thesis is right: `LFCN` beats `LFCN-NoLoopProduct`, especially on separating fine `1` from fine `2`.

### Parameter-matched control: cycle-scrambled curvature

Name: `LFCN-CycleScramble`.

Use the exact same trainable modules and parameter count. During the observable computation only, apply a fixed random permutation to one gathered edge position inside every loop:

```python
scrambled_loop_edge_ids = loop_edge_ids.clone()
scrambled_loop_edge_ids[:, 0] = scrambled_loop_edge_ids[perm, 0]
```

The edge-coupling distribution, loop lengths, beta finite differences, and head dimensionality remain unchanged. What is destroyed is the real closed-loop topology of the board.

Expected outcome if thesis is right: the scrambled control should underperform the real loop bank. If it matches performance within confidence intervals, the claimed physical structure is not doing work.

### Additional ablations

1. **No curvature:** use only `sigmoid(-nu * P)` frustration density, removing `D2A`.
2. **No frustration weighting:** use only `abs(D2A)`, removing the negative-loop emphasis.
3. **Fixed beta:** set `beta = 1.0`, remove learned temperature.
4. **Single replica:** set `K = 1`, testing whether replica diversity matters.
5. **Rectangles only:** remove triangle loops.
6. **Triangles only:** remove rectangle loops.
7. **Head capacity control:** double the observable MLP hidden size in the ablation to make sure the main model is not merely winning from classifier capacity.
8. **Raw-CNN baseline:** train a parameter-matched CNN classifier as a baseline, but not as the proposed model.

## 12. Falsification

The idea should be considered falsified or at least seriously weakened if any of the following occurs:

1. `LFCN-CycleScramble` matches or exceeds `LFCN` on validation AUC, validation AP, and the 3x2 fine-label diagnostic.
2. `LFCN-NoLoopProduct` matches `LFCN`, implying edge magnitudes alone are enough and closed-loop frustration is irrelevant.
3. The model improves only on fine label `0` but not on fine label `1` versus fine label `2`.
4. The observable `Omega` is nearly constant across positions or collapses to a single replica.
5. High `Omega` correlates mainly with material count, number of occupied squares, side to move, or augmentation artifacts.
6. A shuffled-label run gives non-trivial validation performance, indicating leakage or split contamination.
7. Mirror/rotation consistency checks fail when the input representation is canonicalized to side-to-move perspective.
8. The model performance depends on forbidden metadata being present in `C` channels.

Minimum falsification report:

```text
main LFCN metrics
central ablation metrics
cycle-scrambled control metrics
3x2 fine-label diagnostic for each
mean Omega by fine label
Omega top8/mean concentration by fine label
calibration curve or ECE by target
```

## 13. Implementation Notes

### Static graph builder

Create graph buffers once in the module constructor and register them as non-trainable buffers:

```python
self.register_buffer("edge_i", edge_i.long())
self.register_buffer("edge_j", edge_j.long())
self.register_buffer("edge_type", edge_type.long())
self.register_buffer("loop_edge_ids", loop_edge_ids.long())
self.register_buffer("loop_edge_mask", loop_edge_mask.bool())
self.register_buffer("loop_vertex_ids", loop_vertex_ids.long())
self.register_buffer("loop_vertex_mask", loop_vertex_mask.bool())
```

Edge IDs should be canonicalized by sorted endpoint pair so every undirected unit edge has one ID.

### Loop construction recipe

1. Map square `(r, c)` to site ID `8*r + c`.
2. Add unit edges:
   - `(r, c) <-> (r, c+1)` for horizontal;
   - `(r, c) <-> (r+1, c)` for vertical;
   - `(r, c) <-> (r+1, c+1)` for down-right diagonal;
   - `(r, c) <-> (r+1, c-1)` for down-left diagonal.
3. For each rectangle height `h in {1,2,3}` and width `w in {1,2,3}`, add the ordered boundary vertices and convert consecutive vertex pairs to unit edge IDs.
4. For each unit square, add four triangles:
   - top-left, top-right, bottom-right;
   - top-left, bottom-left, bottom-right;
   - top-left, top-right, bottom-left;
   - top-right, bottom-left, bottom-right.
5. Pad edge IDs to `12`; pad vertex IDs to `12`; masks mark real entries.

### Minimal class skeleton

```python
class LoopFrustrationCurvatureNet(nn.Module):
    def __init__(self, in_channels: int, replicas: int = 8):
        super().__init__()
        self.K = replicas
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
        )
        self.site_spin = nn.Conv2d(64, replicas, 1)
        self.edge_type_emb = nn.Embedding(4, 8)
        self.edge_mlp = nn.Sequential(
            nn.Linear(4 * 64 + 8, 128), nn.GELU(),
            nn.Linear(128, replicas),
        )
        self.raw_beta = nn.Parameter(torch.tensor(0.8))
        self.head = nn.Sequential(
            nn.Linear(7 * replicas, 32), nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )
        # register graph buffers here

    def forward(self, x):
        B = x.shape[0]
        g = self.encoder(x)                         # (B, 64, 8, 8)
        m = torch.tanh(self.site_spin(g))           # (B, K, 8, 8)
        gf = g.flatten(2)                           # (B, 64, 64)

        gi = gf[:, :, self.edge_i].transpose(1, 2)  # (B, M, 64)
        gj = gf[:, :, self.edge_j].transpose(1, 2)  # (B, M, 64)
        te = self.edge_type_emb(self.edge_type)      # (M, 8)
        te = te.unsqueeze(0).expand(B, -1, -1)       # (B, M, 8)
        pair = torch.cat([gi, gj, (gi - gj).abs(), gi * gj, te], dim=-1)

        J_raw = self.edge_mlp(pair)                  # (B, M, K)
        J = 2.5 * torch.tanh(J_raw.transpose(1, 2) / 2.5)  # (B, K, M)

        beta = (0.20 + F.softplus(self.raw_beta)).clamp(max=3.0)
        omega, p_mid = self.loop_curvature(J, beta)  # (B, K, L), (B, K, L)
        omega_site = self.scatter_loops_to_sites(omega)  # (B, K, 8, 8)
        obs = self.make_observables(omega_site, omega, p_mid, m)
        return self.head(obs)
```

### Training defaults

```text
optimizer      : AdamW
learning rate  : 3e-4
weight decay   : 1e-4
batch size     : 128 or largest stable batch
max epochs     : 50
early stopping : validation AP or validation BCE, patience 8
mixed precision: allowed
augmentation   : board symmetries only if labels remain invariant and split hygiene prevents leakage
```

### Diagnostics to log

Per epoch:

- train BCE;
- validation BCE;
- validation AUC;
- validation AP;
- 3x2 fine-label diagnostic;
- mean `Omega` by fine label;
- top8/mean `Omega` concentration by fine label;
- learned `beta`;
- mean and max `abs(J)`;
- replica correlation mean off-diagonal.

## 14. Future Prompt Updates

Recommended future constraints to make the research program sharper:

1. Require a parameter-count table for `LFCN`, `LFCN-NoLoopProduct`, `LFCN-CycleScramble`, and the raw-CNN baseline.
2. Require an explicit `build_loop_bank()` implementation with deterministic unit tests for `M = 210`, `L = 520`, and `Lmax = 12`.
3. Require validation broken out as fine `1` versus fine `2`, because that is the near-puzzle boundary that matters most.
4. Require a leakage audit that prints the names and meanings of all `C` input channels before training starts.
5. Require a symmetry test: original board, horizontal mirror if legal under representation, vertical perspective transform if canonicalized, and color/side-to-move inversion if the data pipeline supports it.
6. Require observable visualizations on a fixed validation panel: overlay `Omega_site` on the board without showing engine lines or best moves.
7. Require a “material-matched” evaluation slice where fine labels have similar piece counts and side-to-move balance.
8. Require confidence intervals across at least five seeds for the main model and both central controls.
