# Math Thesis

Fixed-Point Residual Defect Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md`.

Batch candidate rank: `1`.

## Setup

A board encoder $E_\theta$ maps the simple_18 board tensor $x \in \mathbb{R}^{C \times 8 \times 8}$ to an initial latent $h_0 = E_\theta(x) \in \mathbb{R}^d$ together with a board-conditioning embedding $c = C_\theta(x) \in \mathbb{R}^{d_c}$. A learned update operator $T_\phi : \mathbb{R}^d \times \mathbb{R}^{d_c} \to \mathbb{R}^d$ defines a damped fixed-point iteration

$$h_{k+1} = h_k + \alpha \, r_k, \qquad r_k = T_\phi(h_k, c) - h_k, \qquad k = 0, \ldots, K-1,$$

with damping $\alpha \in (0, 1]$. The trajectory of *defects* (residuals of the fixed-point map $h \mapsto T_\phi(h, c)$) is

$$\mathcal{R}(x) = (r_0, r_1, \ldots, r_{K-1}), \qquad r_K = T_\phi(h_K, c) - h_K.$$

Working thesis: Puzzle-like positions may be harder for a learned board-state operator to equilibrate. Instead of classifying only the final latent, classify from the residual defects of an unrolled update process.

## Defect functional

The classifier head consumes a permutation-invariant statistic of the defect path rather than the final latent alone. Concretely, with $\| \cdot \|$ the $\ell_2$ norm and $P \in \mathbb{R}^{p \times d}$ a learned projection (row-normalised at use time):

$$\psi(x) = \big(\, \|r_k\|_2,\ \|r_k\|_1,\ \cos(r_k, r_{k-1}),\ \tfrac{\|r_k\|}{\|r_{k-1}\|},\ \|r_k\| - \|r_{k-1}\|,\ P r_k \,\big)_{k=0}^{K-1} \,\oplus\, \mathcal{S}(x) \,\oplus\, h_K,$$

where $\mathcal{S}(x)$ is a global block of path-summary scalars: total path length $\sum_k \|r_k\|$, mean and max defect, terminal defect $\|r_K\|$, $\|r_K\|_1$, mean contraction ratio, and mean oscillation $1 - \cos(r_k, r_{k-1})$. The puzzle logit is $\hat{y} = w^\top \mathrm{MLP}(\psi(x))$, and the overall puzzle decision flows through $\psi$ — the head never sees the raw board features directly except through $h_K$ when that channel is enabled.

## Why fixed-point defects discriminate puzzles

Two intuitions motivate the architecture.

1. **Slow contraction on tactical positions.** If $T_\phi$ is approximately a contraction with operator norm $L < 1$ on quiet positions, $\|r_k\|$ decays geometrically with rate $\le L^k$ and $\sum_k \|r_k\| \le \|r_0\| / (1 - L)$. Tactical positions break this regime: the operator does not contract uniformly, $\|r_k\|$ stalls, oscillates, or grows, and $1 - \cos(r_k, r_{k-1})$ stays away from zero. The signed delta $\|r_k\| - \|r_{k-1}\|$ and the contraction ratio $\|r_k\| / \|r_{k-1}\|$ thus directly expose the failure of equilibration.

2. **Defect direction carries problem-specific structure.** Two positions with similar $\|r_K\|$ can equilibrate against very different sub-spaces. Projecting $r_k$ onto the learned axes $P$ (row-normalised) exposes which directions of the latent state space the operator cannot resolve, regardless of overall residual magnitude.

## Ablation modes

The architecture admits five named modes that disable specific claims:

- `none` / `fixed_point` — full defect-trajectory readout (the thesis above).
- `final_latent_only` — classify only from $h_K$. Tests whether the residual-defect channels add anything beyond the final latent.
- `defect_norm_only` — keep only the scalar norms $(\|r_k\|, \|r_k\|_1, \|r_k\|/\|r_{k-1}\|)$ and a small global block. Tests whether sign/direction matter.
- `single_step` — run one update step and pad the rest with zeros. Tests whether unrolling matters.
- `untied_residual_blocks` — replace the shared $T_\phi$ with $K$ untied blocks. Tests whether a *fixed-point* operator is required, or whether $K$ generic residual layers suffice.
- `random_update_operator` — freeze $T_\phi$ at initialisation. Tests whether the defects of any operator carry signal, or whether a *learned* operator is required.

The mode code is exposed as a diagnostic so the ablation harness can attach the active branch to each prediction.

## Inputs and contract

- Input: `(B, C, 8, 8)` simple_18 board tensor only. CRTK / source / verification metadata is reporting-only and never consumed by the model.
- Output: a single puzzle logit per board for the `puzzle_binary` BCE-with-logits trainer, with $0, 1 \mapsto 0$ and $2 \mapsto 1$ on the fine label, plus the full path of residual norms, cosines, contraction ratios, projections, and signed deltas as diagnostics.
