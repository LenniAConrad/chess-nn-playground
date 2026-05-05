# Codex Handoff Packet: Chess-Mode Tucker Relation Certificate

## 1. File Metadata

- **Filename:** `chess_nn_research_2026-04-28_0900_tuesday_new_york_relation_tucker.md`
- **Generated:** 2026-04-28 09:00, Tuesday, `new_york`
- **Task:** Codex-ready architecture packet for `puzzle_binary` using tensor algebra beyond ANOVA-style piece constellations.
- **Selected idea name:** **Chess-Mode Tucker Relation Certificate** (`CMTRC`)
- **Central operator:** chess-constrained Tucker contraction over a fixed relation-moment tensor, with a multilinear-rank / tensor-nuclear-norm certificate.
- **Input / output:** model consumes `x` with shape `(batch, C, 8, 8)` and returns logits with shape `(batch, 1)`.
- **Target mapping:** fine labels `0` and `1` map to binary target `0`; fine label `2` maps to binary target `1`.
- **Required diagnostic:** report a `3 x 2` fine-label diagnostic matrix, with rows for fine labels `0, 1, 2` and columns for predicted binary classes `0, 1`.

## 2. Executive Selection

Build **CMTRC**, a compact neural architecture whose main learnable decision operator is not a CNN block, attention module, Transformer, pair-field model, or tensor ring. The model first embeds the board channels into latent piece-state channels, then constructs a **fixed chess-relation moment tensor**:

\[
T_b \in \mathbb{R}^{K \times R \times D \times G},
\]

where `K` is latent channel, `R` is chess relation family, `D` is relation depth / jump index, and `G` is board-region group. A Tucker head then applies mode factors and a core tensor:

\[
S_b = T_b \times_K U_K^\top \times_R U_R^\top \times_D U_D^\top \times_G U_G^\top,
\]

\[
z_{b,h} = \langle S_b, \Omega_{:,:,:,:,h} \rangle.
\]

The core thesis is that the binary puzzle label should be predictable from low-multilinear-rank interactions among **piece-state channel**, **lawful chess relation**, **distance / jump**, and **board-region** modes. The model is intentionally constrained: it can learn that certain relation modes cohere, but it cannot learn arbitrary square-pair attention maps, cannot ingest engine fields, and cannot memorize source metadata.

The same-parameter control is **FlatProjectedMLP**, which uses the same stem and the same fixed relation tensor, flattens that tensor through a fixed random projection, and uses an MLP with the exact same number of trainable head parameters as the Tucker head. This isolates whether the tensor-mode structure itself adds value.

## 3. Data Contract

### Inputs

```python
x: torch.FloatTensor  # shape: (B, C, 8, 8)
fine_label: torch.LongTensor  # shape: (B,), values in {0, 1, 2}
```

`C` is supplied by the existing `puzzle_binary` board-plane encoder. The architecture assumes only legal board-state planes and non-leaking position features already permitted by the dataset contract.

### Forbidden inference inputs

Do **not** pass any of the following into the model, collate function, inference batch, validation batch, or test batch:

- engine analysis
- principal variations / PVs
- node counts
- mate scores
- best moves
- verification metadata
- source labels
- source files

These fields should not be hidden in auxiliary tensors, side-channel dictionaries, sample weights, batch metadata used by `forward`, or post-hoc threshold rules.

### Target mapping

```python
y = (fine_label == 2).float()  # shape: (B,)
y = y[:, None]                # shape: (B, 1)
```

Fine labels `0` and `1` remain distinct only for diagnostics. They are both binary target `0`.

### Model output

```python
logits = model(x)  # shape: (B, 1)
```

The forward path must return only `(B, 1)` logits. A separate method such as `forward_with_aux(x)` may return the projected tensor and rank certificate during training, but production inference should call `forward(x)`.

### Required `3 x 2` fine-label diagnostic

For a validation or test split, report:

\[
D[f, p] = \#\{i : \text{fine\_label}_i = f, \; \mathbf{1}[\sigma(\ell_i) \ge \tau] = p\},
\]

where `f in {0,1,2}` and `p in {0,1}`. Row order is fixed as `[0, 1, 2]`; column order is fixed as `[predicted_binary_0, predicted_binary_1]`.

Recommended output block:

```text
fine_label_diagnostic_counts_3x2:
           pred_0  pred_1
fine_0       ...     ...
fine_1       ...     ...
fine_2       ...     ...

fine_label_diagnostic_row_rates_3x2:
           pred_0  pred_1
fine_0       ...     ...
fine_1       ...     ...
fine_2       ...     ...
```

## 4. Tensor-Algebra Research Background

This architecture should be read as a supervised, chess-constrained tensor model rather than a generic neural compression trick.

Kolda and Bader define tensors as multidimensional arrays and describe CP and Tucker as canonical higher-order decompositions; Tucker is especially relevant here because it separates mode factors from a compact interaction core. See Tamara G. Kolda and Brett W. Bader, “Tensor Decompositions and Applications,” *SIAM Review*, 2009, DOI: [10.1137/07070111X](https://doi.org/10.1137/07070111X).

Oseledets introduced tensor trains as a stable low-rank representation for high-dimensional tensors, with operations based on low-rank auxiliary unfoldings. This packet does **not** select a tensor train because square-wise tensor trains and tensor rings are too close to the forbidden square-interaction family, but the TT literature motivates explicit mode rank control. See I. V. Oseledets, “Tensor-Train Decomposition,” *SIAM Journal on Scientific Computing*, 2011, DOI: [10.1137/090752286](https://doi.org/10.1137/090752286).

Hillar and Lim show that many natural tensor problems, including rank and spectral-norm variants, are NP-hard in general. That hardness argues for small, fixed, chess-constrained tensor objects rather than unconstrained tensor optimization over all square tuples. See Christopher J. Hillar and Lek-Heng Lim, “Most Tensor Problems Are NP-Hard,” *Journal of the ACM*, 2013, DOI: [10.1145/2512329](https://doi.org/10.1145/2512329).

Novikov et al. demonstrated that dense neural layers can be tensorized with tensor-train formats to reduce parameters. CMTRC should **not** be reduced to that pattern: its tensor object is a chess relation tensor, not a compressed dense layer. See Alexander Novikov, Dmitrii Podoprikhin, Anton Osokin, and Dmitry Vetrov, “Tensorizing Neural Networks,” NeurIPS 2015, paper page: [papers.nips.cc/paper/5787-tensorizing-neural-networks](https://papers.nips.cc/paper/5787-tensorizing-neural-networks).

Cohen, Sharir, and Shashua relate deep network expressivity to tensor factorizations, including CP-like shallow structure and hierarchical Tucker-like deep structure. CMTRC borrows the idea that mode factorization can encode compositional interactions, but it uses explicit chess relation modes rather than image-patch pooling. See Nadav Cohen, Or Sharir, and Amnon Shashua, “On the Expressive Power of Deep Learning: A Tensor Analysis,” COLT 2016, PMLR page: [proceedings.mlr.press/v49/cohen16.html](https://proceedings.mlr.press/v49/cohen16.html).

Kossaifi et al. argued that flattening high-order activations discards multilinear structure, and proposed tensor contraction / tensor regression layers. CMTRC applies the same warning to chess boards: flattening `C x 8 x 8` into a vector hides the distinction among channel, relation, distance, and board-region modes. See Jean Kossaifi, Zachary C. Lipton, Aran Khanna, Tommaso Furlanello, and Anima Anandkumar, “Tensor Regression Networks,” arXiv: [1707.08308](https://arxiv.org/abs/1707.08308).

## 5. Candidate Search Trace

| Candidate | Central tensor object | Decision | Reason |
|---|---:|---|---|
| **Chess-Mode Tucker Relation Certificate** | `K x R x D x G` relation-moment tensor with Tucker core | **Selected** | Explicit tensor operator; chess-mode constraints; avoids learned square-pair attention; supports rank certificate and clean same-parameter control. |
| Tensor train over all 64 squares | `q_1 x ... x q_64` or folded square sequence | Rejected | Too close to square-sequence / tensor-ring interaction patterns; ordering artifacts dominate chess geometry. |
| CP decomposition over source-target piece pairs | `piece x source x target` | Rejected | Too close to pair-field attention and pairwise lookup; also collapses tactical structure into source-target dyads. |
| Hyperdeterminant-inspired invariant | Small `2 x 2 x 2 x 2` tactical tensor | Rejected as primary | Interesting but brittle: hyperdeterminants are hard to stabilize and likely underfit broad puzzle patterns. Can be a later diagnostic feature, not the first architecture. |
| Tensor nuclear norm only | Learned activation tensor with nuclear penalty | Rejected as primary | A regularizer alone is not a thesis; needs an explicit relation-mode tensor and core. |
| Tucker-compressed CNN | Low-rank convolutional kernel factorization | Rejected | Falls into ordinary low-rank factorized CNN territory, which the task explicitly excludes. |
| Generic Transformer over squares | 64 tokens with attention | Rejected | Generic Transformer and pair-field attention are forbidden; attention maps would also be hard to distinguish from leakage memorization. |

## 6. Rejected Common Approaches

The implementation must not drift into the following patterns:

1. **Möbius / ANOVA piece constellations.** No subset-lattice expansion over piece constellations, no ANOVA-style inclusion-exclusion features, and no hand-built constellation tables.
2. **Tensor-ring square interactions.** No ring or cyclic contraction over the 64 squares. The selected tensor modes are semantic chess modes, not square-index factors.
3. **Simple bilinear channel mixing.** The model may use a `1 x 1` channel lift, but the decision operator is not a two-factor bilinear channel mixer. The head must use at least the four modes `K, R, D, G` and a Tucker core.
4. **Pair-field attention.** Relation masks are fixed, sparse chess masks. There is no learned query-key attention and no learned per-square-pair attention matrix.
5. **Ordinary low-rank factorized CNNs.** Do not replace the operator with Tucker/CP-compressed convolutions. Convolutions, if any, are only a minimal stem; the main learnable interaction is the chess relation Tucker core.
6. **Generic Transformers.** No square-token Transformer encoder, no positional-attention stack, and no attention pooling.

## 7. Mathematical Thesis

The binary task is modeled as a decision over low-multilinear-rank chess relation interactions:

\[
\Pr(y=1 \mid x) = \sigma\left(g\left(\left\{\langle T(x) \times_K U_K^\top \times_R U_R^\top \times_D U_D^\top \times_G U_G^\top,\; \Omega_h\rangle\right\}_{h=1}^{H}\right)\right).
\]

The tensor object `T(x)` is not a generic learned attention field. It is built from fixed legal chess relation masks and board-region masks. The learnable part is the low-rank mode interaction, not the relation topology.

Let:

- `s, t` index board squares in `{0, ..., 63}`.
- `k` index latent board-state channels, `k in {1, ..., K}`.
- `rho` index relation families, `rho in {1, ..., R}`.
- `delta` index relation depths or jump variants, `delta in {1, ..., D}`.
- `gamma` index board-region groups, `gamma in {1, ..., G}`.
- `E_{b,k,s}` be the latent board embedding at square `s`.
- `M_{rho,delta,s,t}` be a fixed sparse relation mask.
- `A_{gamma,s}` be a fixed normalized board-region mask.

Relation scan:

\[
N_{b,k,\rho,\delta,s} = \sum_t M_{\rho,\delta,s,t} E_{b,k,t}.
\]

Relation moment tensor:

\[
T_{b,k,\rho,\delta,\gamma}
= \sum_s A_{\gamma,s}\, E_{b,k,s}\, \tanh\left(\frac{N_{b,k,\rho,\delta,s}}{\sqrt{\operatorname{deg}(\rho,\delta,s)+\epsilon}}\right).
\]

Tucker projection:

\[
S_b = T_b \times_K U_K^\top \times_R U_R^\top \times_D U_D^\top \times_G U_G^\top,
\]

with shape:

\[
S_b \in \mathbb{R}^{r_K \times r_R \times r_D \times r_G}.
\]

Core contraction:

\[
z_{b,h} = \sum_{a=1}^{r_K}\sum_{r=1}^{r_R}\sum_{d=1}^{r_D}\sum_{g=1}^{r_G}
S_{b,a,r,d,g}\,\Omega_{a,r,d,g,h}.
\]

Logit:

\[
\ell_b = w_2^\top \operatorname{SiLU}(W_1 z_b + b_1) + b_2.
\]

The tensor thesis is falsifiable: if a same-parameter non-tensor flat projection performs as well across seeds, then the chess-mode Tucker structure is not earning its complexity.

## 8. Tensor Object Definition

### Latent embedding

```python
K = 32
E = channel_lift(x)       # (B, K, 8, 8)
E = group_norm(E)
E = silu(E)
E = E.flatten(2)          # (B, K, 64)
```

The channel lift is a plain `1 x 1` convolution from `C` to `K`. It is only a board-state embedding, not the research contribution.

### Relation-family mode `R = 12`

Use fixed masks. The relation family mode is not learned.

| `rho` range | Family | Meaning |
|---:|---|---|
| `0..7` | Sliding ray directions | `N, S, E, W, NE, NW, SE, SW` |
| `8` | Knight jumps | eight signed knight offsets |
| `9` | King-adjacent jumps | eight adjacent offsets |
| `10` | White-pawn attack geometry | `(+rank, -file)` and `(+rank, +file)` under the board encoder's canonical orientation |
| `11` | Black-pawn attack geometry | `(-rank, -file)` and `(-rank, +file)` under the board encoder's canonical orientation |

### Depth / jump mode `D = 8`

- For sliding rays, `delta = 0..6` corresponds to distance `1..7`; `delta = 7` is all-zero padding.
- For knight and king jumps, `delta = 0..7` indexes the signed jump variant.
- For pawn attacks, `delta = 0..1` indexes left/right attack; `delta = 2..7` is all-zero padding.

### Board-region mode `G = 10`

Use overlapping fixed masks, each normalized to sum to one over its active squares:

| `gamma` | Region |
|---:|---|
| `0` | all squares |
| `1` | light squares |
| `2` | dark squares |
| `3` | center four squares |
| `4` | extended center sixteen squares |
| `5` | corners |
| `6` | board edges |
| `7` | back ranks `1` and `8` |
| `8` | side files `a` and `h` |
| `9` | promotion bands, ranks `2` and `7` |

These are not labels or source metadata. They are deterministic functions of square coordinates.

### Fixed relation mask tensor

```python
M: torch.FloatTensor  # shape: (R, D, 64, 64)
A: torch.FloatTensor  # shape: (G, 64)
```

`M[rho, delta, s, t] = 1` iff target square `t` is connected from anchor square `s` by the corresponding fixed chess relation and depth / jump variant. Otherwise it is `0`.

Normalize scans by degree:

```python
deg = M.sum(dim=-1).clamp_min(1.0)  # (R, D, 64)
```

### Relation scan and moment tensor

```python
N = torch.einsum("rdst,bkt->bkrds", M, E)  # (B, K, R, D, 64)
N = N / deg.sqrt()[None, None, :, :, :]
T = torch.einsum("gs,bks,bkrds->bkrdg", A, E, torch.tanh(N))  # (B, K, R, D, G)
```

This is a fixed-geometry multiplicative moment, not pair attention. The model never creates learned `64 x 64` attention weights.

## 9. Architecture And Tensor Shapes

### Recommended default hyperparameters

```text
K   = 32   # latent channels
R   = 12   # relation families
D   = 8    # depth / jump variants
G   = 10   # board groups
rK  = 8
rR  = 6
rD  = 4
rG  = 5
H   = 24
MLP = 32
```

### Forward pass

| Stage | Operation | Shape |
|---|---|---:|
| Input | `x` | `(B, C, 8, 8)` |
| Channel lift | `Conv2d(C, 32, kernel_size=1)` + `GroupNorm` + `SiLU` | `(B, 32, 8, 8)` |
| Flatten board | `E = E.flatten(2)` | `(B, 32, 64)` |
| Fixed relation scan | `N = einsum(M, E)` | `(B, 32, 12, 8, 64)` |
| Relation moment | `T = einsum(A, E, tanh(N))` | `(B, 32, 12, 8, 10)` |
| Tucker mode projection | `S = T x_K Uk^T x_R Ur^T x_D Ud^T x_G Ug^T` | `(B, 8, 6, 4, 5)` |
| Core contraction | `z = <S, Omega>` | `(B, 24)` |
| Small head | `Linear(24,32) -> SiLU -> Linear(32,1)` | `(B, 1)` |

### Trainable parameter count

The stem is shared by the main model and the control:

```text
channel_lift: C*32 + 32
GroupNorm:    64
shared stem:  32*C + 96
```

CMTRC head parameters:

```text
U_K:      32 * 8              =    256
U_R:      12 * 6              =     72
U_D:       8 * 4              =     32
U_G:      10 * 5              =     50
Omega:     8 * 6 * 4 * 5 * 24 = 23,040
MLP1:     24 * 32 + 32        =    800
MLP2:     32 * 1 + 1          =     33
-----------------------------------------
CMTRC head params                 24,283
Total params                      32*C + 24,379
```

### Same-parameter non-tensor control

Control name: **FlatProjectedMLP**.

The control uses the same `channel_lift`, same fixed `M`, same fixed `A`, and same relation tensor `T`, but destroys the explicit tensor-mode operator:

```python
flat = T.flatten(1)                 # (B, 32*12*8*10) = (B, 30720)
proj = fixed_signed_jl(flat, q=112) # (B, 112), no trainable params
logit = Linear(112, 213) -> SiLU -> Linear(213, 1)
```

Control head parameter count:

```text
Linear(112,213): 112 * 213 + 213 = 24,069
Linear(213,1):   213 * 1 + 1     =    214
------------------------------------------
FlatProjectedMLP head params        24,283
Total params                        32*C + 24,379
```

This is an exact trainable-parameter match to CMTRC. Any CMTRC advantage should therefore come from the chess-mode Tucker structure, not parameter count.

### Multilinear-rank / tensor-nuclear-norm certificate

Use the projected tensor `S` with shape `(B, rK, rR, rD, rG) = (B, 8, 6, 4, 5)`.

For each example and each tensor mode `m`, unfold `S_b` into a matrix `S_b^(m)` and compute singular values:

```python
sv_m = torch.linalg.svdvals(unfold(S, mode=m))
nuc_m = sv_m.sum(dim=-1)
fro_m = torch.linalg.vector_norm(sv_m, dim=-1)
eff_rank_m = (nuc_m / fro_m.clamp_min(1e-8)).pow(2)
```

Report mean effective rank by mode:

```text
rank_certificate:
  K_mode_eff_rank: ...  # max 8
  R_mode_eff_rank: ...  # max 6
  D_mode_eff_rank: ...  # max 4
  G_mode_eff_rank: ...  # max 5
```

The certificate is a diagnostic and optional regularizer. It is not an input feature.

## 10. Training Objective

### Main objective

```python
binary_target = (fine_label == 2).float().unsqueeze(1)
bce = torch.nn.functional.binary_cross_entropy_with_logits(
    logits,
    binary_target,
    pos_weight=pos_weight,  # computed on train split only, optional
)
```

### Tensor bottleneck regularization

Use a small coefficient and tune only on validation:

\[
\mathcal{L} = \mathcal{L}_{BCE} + \lambda_{nuc}\mathcal{R}_{nuc} + \lambda_{orth}\mathcal{R}_{orth}.
\]

Recommended starting values:

```text
lambda_nuc  = 1e-4
lambda_orth = 1e-5
```

Nuclear bottleneck:

\[
\mathcal{R}_{nuc} = \sum_{m \in \{K,R,D,G\}} \mathbb{E}_b
\frac{\lVert S_b^{(m)} \rVert_*}{\lVert S_b \rVert_F + \epsilon}.
\]

Orthogonality stabilizer for mode factors:

\[
\mathcal{R}_{orth} =
\lVert U_K^\top U_K - I \rVert_F^2 +
\lVert U_R^\top U_R - I \rVert_F^2 +
\lVert U_D^\top U_D - I \rVert_F^2 +
\lVert U_G^\top U_G - I \rVert_F^2.
\]

If SVD cost matters, compute `R_nuc` every `k=4` batches and reuse zero otherwise; the certificate tensor is small, so full computation should usually be acceptable.

### Optimization defaults

```text
optimizer: AdamW
learning_rate: 3e-4
weight_decay: 1e-4
batch_size: use existing puzzle_binary default; 256 if memory allows
scheduler: cosine decay with 3-5% warmup
max_epochs: 30-80 with early stopping on validation balanced accuracy or AUROC
threshold tau: choose on validation split, then freeze for test
seeds: at least 5 for main vs control comparison
```

### Required metrics

Report at minimum:

- validation/test BCE
- AUROC
- PR-AUC
- balanced accuracy at frozen threshold `tau`
- calibration error if already available
- `3 x 2` fine-label diagnostic counts
- `3 x 2` fine-label diagnostic row-normalized rates
- rank certificate means by mode

### Diagnostic implementation

```python
@torch.no_grad()
def fine_label_diagnostic_3x2(logits, fine_label, tau=0.5):
    pred = (logits.sigmoid().view(-1) >= tau).long()
    fine = fine_label.view(-1).long()
    counts = torch.zeros(3, 2, dtype=torch.long, device=fine.device)
    for f in range(3):
        mask_f = fine == f
        for p in range(2):
            counts[f, p] = (mask_f & (pred == p)).sum()
    row_rates = counts.float() / counts.sum(dim=1, keepdim=True).clamp_min(1)
    return counts.cpu(), row_rates.cpu()
```

## 11. Ablations

Run all ablations with identical train/validation/test splits, seeds, optimizer schedule, batch size, target mapping, and threshold-selection protocol.

| Ablation | Description | Purpose |
|---|---|---|
| **CMTRC full** | Fixed relation tensor + Tucker core + nuclear/rank certificate | Main proposal. |
| **FlatProjectedMLP same-param control** | Same stem and relation tensor; fixed flat projection; MLP with exactly 24,283 head params | Tests whether tensor-mode structure matters. |
| **No nuclear bottleneck** | Remove `R_nuc`; keep Tucker operator | Tests whether rank certificate regularization matters. |
| **No orthogonality stabilizer** | Remove `R_orth` | Tests whether mode-factor stability matters. |
| **Random relation masks** | Replace legal chess masks `M` with degree-matched random masks | Tests whether chess geometry matters. |
| **Region mode removed** | Collapse `G` to all-squares only | Tests whether region mode adds useful multilinear structure. |
| **Depth mode collapsed** | Sum over `D` before Tucker projection | Tests whether distance / jump indexing matters. |
| **CP head** | Replace Tucker core with CP rank chosen to match parameters as closely as possible | Tests whether full Tucker core is necessary. |
| **Flat board MLP budget control** | Flatten `E` through fixed projection, matched MLP | Tests whether relation tensor construction matters at all. |
| **Rank sweep** | Try `(rK,rR,rD,rG)` = `(4,4,3,4)`, `(8,6,4,5)`, `(12,8,5,6)` | Tests under/over-parameterization and rank-certificate behavior. |

Success should not be declared from one seed. Use mean and confidence intervals across seeds.

## 12. Falsification Rule

Reject CMTRC as the selected architecture if any of the following occur under the agreed split protocol:

1. **Same-parameter control parity:** FlatProjectedMLP matches or exceeds CMTRC on validation/test AUROC and balanced accuracy across seeds, with no meaningful confidence-interval separation.
2. **Chess geometry irrelevance:** Degree-matched random relation masks match legal chess masks across seeds.
3. **Rank certificate collapse:** Effective ranks collapse to near `1` in all modes while performance remains high, suggesting the head is acting like a trivial scalar feature rather than a multilinear operator.
4. **Rank certificate saturation:** Effective ranks saturate near their maxima in all modes and the nuclear penalty cannot reduce them without hurting performance, suggesting the low-multilinear-rank thesis is wrong.
5. **Fine-label pathology:** The `3 x 2` diagnostic shows fine label `2` is not separated from both fine labels `0` and `1`, or one negative fine label is consistently treated as positive in a way that dominates the aggregate metric.
6. **Leakage sensitivity:** Any performance advantage disappears after strict removal of forbidden fields from batches, logs, collators, cached tensors, or threshold rules.
7. **Board perturbation sanity failure:** Performance remains high under invalid perturbations that destroy chess relations, such as square permutation with masks not correspondingly permuted. That would indicate leakage or non-board artifacts.

A positive result means CMTRC beats the exact-parameter control and the random-mask control, while producing a plausible rank certificate and clean fine-label diagnostics.

## 13. Implementation Plan

### Files to add

```text
puzzle_binary/models/cmtrc.py
puzzle_binary/training/fine_label_diagnostics.py
puzzle_binary/tests/test_cmtrc_shapes.py
puzzle_binary/tests/test_cmtrc_param_match.py
```

Do not add side-channel metadata files. Do not add features derived from engines or source labels.

### `cmtrc.py` class layout

```python
class RelationMaskBuilder:
    def build_relation_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return M: (12,8,64,64), A: (10,64). Fixed, float32."""

class ChessModeTuckerRelationCertificate(nn.Module):
    def __init__(self, in_channels: int, ...):
        super().__init__()
        self.channel_lift = nn.Conv2d(in_channels, 32, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=32)
        self.register_buffer("M", M, persistent=False)
        self.register_buffer("A", A, persistent=False)
        self.register_buffer("deg_sqrt", deg.sqrt(), persistent=False)
        self.Uk = nn.Parameter(torch.empty(32, 8))
        self.Ur = nn.Parameter(torch.empty(12, 6))
        self.Ud = nn.Parameter(torch.empty(8, 4))
        self.Ug = nn.Parameter(torch.empty(10, 5))
        self.core = nn.Parameter(torch.empty(8, 6, 4, 5, 24))
        self.out1 = nn.Linear(24, 32)
        self.out2 = nn.Linear(32, 1)

    def relation_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Return T: (B,32,12,8,10)."""

    def tucker_project(self, T: torch.Tensor) -> torch.Tensor:
        """Return S: (B,8,6,4,5)."""

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return z: (B,24), S: (B,8,6,4,5)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits: (B,1)."""

    def forward_with_aux(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Training-only: logits, S, rank certificate values."""
```

### Core forward pseudocode

```python
def relation_tensor(self, x):
    E = self.channel_lift(x)            # (B,32,8,8)
    E = F.silu(self.norm(E))
    E = E.flatten(2)                    # (B,32,64)

    N = torch.einsum("rdst,bkt->bkrds", self.M, E)
    N = N / self.deg_sqrt[None, None, :, :, :].clamp_min(1.0)
    T = torch.einsum("gs,bks,bkrds->bkrdg", self.A, E, torch.tanh(N))
    return T

def tucker_project(self, T):
    S = torch.einsum(
        "bkrdg,ka,rl,dm,gn->balmn",
        T, self.Uk, self.Ur, self.Ud, self.Ug,
    )
    return S

def forward_features(self, x):
    T = self.relation_tensor(x)
    S = self.tucker_project(T)
    z = torch.einsum("balmn,almnh->bh", S, self.core)
    return z, S

def forward(self, x):
    z, _ = self.forward_features(x)
    return self.out2(F.silu(self.out1(z)))
```

### Initialization

```python
nn.init.kaiming_uniform_(self.channel_lift.weight, a=math.sqrt(5))
for U in [self.Uk, self.Ur, self.Ud, self.Ug]:
    nn.init.orthogonal_(U)
nn.init.normal_(self.core, mean=0.0, std=0.02)
```

For rectangular factors, `orthogonal_` gives orthonormal columns when rows exceed columns, which is the intended orientation.

### Same-parameter control class

```python
class FlatProjectedMLPControl(nn.Module):
    def __init__(self, in_channels: int, seed: int = 1729):
        # same channel_lift, norm, M, A as CMTRC
        # fixed projection from 30720 to 112; no trainable params
        self.fc1 = nn.Linear(112, 213)
        self.fc2 = nn.Linear(213, 1)
```

Use a deterministic fixed signed projection:

```python
# P is a buffer, not a parameter.
# Shape can be dense (30720,112) if memory is acceptable, or CountSketch-sparse.
proj = flat @ P / math.sqrt(112)
```

If dense projection memory is undesirable, use CountSketch:

```python
bucket: LongTensor  # shape (30720,), values 0..111
sign: FloatTensor   # shape (30720,), values {-1,+1}
proj = torch.zeros(B, 112, device=flat.device, dtype=flat.dtype)
proj.scatter_add_(1, bucket.expand(B, -1), flat * sign)
proj = proj / math.sqrt(112)
```

The control must have exactly the same trainable parameter count as CMTRC:

```text
CMTRC head: 24,283
Control head: 112*213 + 213 + 213 + 1 = 24,283
```

### Diagnostics module

Add `fine_label_diagnostic_3x2` exactly as specified in Section 10. Store both counts and row-normalized rates in the experiment result JSON or log.

### Tests

1. **Shape test**

```python
x = torch.randn(4, C, 8, 8)
model = ChessModeTuckerRelationCertificate(C)
assert model(x).shape == (4, 1)
```

2. **Relation tensor shape test**

```python
T = model.relation_tensor(x)
assert T.shape == (4, 32, 12, 8, 10)
```

3. **Projected tensor shape test**

```python
S = model.tucker_project(T)
assert S.shape == (4, 8, 6, 4, 5)
```

4. **Parameter match test**

```python
main = ChessModeTuckerRelationCertificate(C)
ctrl = FlatProjectedMLPControl(C)
assert count_trainable(main) == count_trainable(ctrl)
```

5. **Forbidden-batch test**

Make the trainer fail fast if a batch passed to `model.forward` contains keys matching:

```text
engine, pv, node, mate, best_move, verification, source_label, source_file
```

6. **Diagnostic shape test**

```python
counts, rates = fine_label_diagnostic_3x2(logits, fine_label, tau=0.5)
assert counts.shape == (3, 2)
assert rates.shape == (3, 2)
```

### Training integration

- Add CLI option `--model cmtrc`.
- Add CLI option `--model flat_projected_mlp_control`.
- Log parameter count before training.
- Log the exact target mapping.
- Log the `3 x 2` fine-label diagnostic on validation and test.
- Log rank certificate only for CMTRC.
- Use the same threshold-selection code for CMTRC and control.

## 14. Prompt-Maintenance Notes

- Preserve the central operator: **fixed chess relation tensor + Tucker mode projection + core contraction**.
- Do not rewrite the model into a square-token Transformer, pair attention model, tensor-ring square model, or low-rank CNN.
- Keep fine labels `0` and `1` mapped to target `0`; keep fine label `2` mapped to target `1`.
- Always report the `3 x 2` diagnostic. Aggregate binary accuracy alone is not enough.
- Keep forbidden inference inputs out of the dataset batch path, not just out of the model class.
- The same-parameter control is mandatory. Without it, the experiment cannot distinguish tensor-mode benefit from parameter-budget benefit.
- Treat source labels and source files as non-inference metadata at most. They should never influence features, logits, thresholds, or diagnostics except for explicit leakage audits outside the model path.
- Keep relation masks deterministic and auditable. Any learned relation topology turns this into pair-field attention and should be rejected.
- When changing ranks, update both the main head parameter count and the control parameter matcher.
- If implementation pressure tempts simplification, simplify the rank sizes first; do not remove the `K x R x D x G` tensor object.
