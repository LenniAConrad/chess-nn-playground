# Architecture

`Harmonic Board Potential Network` realises the markdown thesis as a bespoke
model: the central computation is a **fixed inverse-Laplacian board potential
solver** over learned current-board charge maps, not a CNN, attention,
sheaf, or move-delta mechanism. The Green matrices `G_l = (L + lambda_l I)^{-1}`
are precomputed at construction and stored as non-trainable buffers, so the
solver is a fixed global linear operator and no message passing is involved.

## Pipeline

1. **`Simple18ChargeEncoder`** is a 1x1 convolution that maps the simple_18
   board tensor `(B, 18, 8, 8)` to `K = charge_channels` signed charge maps
   `rho_k(x) in R^{8 x 8}`. Charges are mean-centred per board so the model
   cannot use a trivial total-charge shortcut that would already be captured
   by a constant potential (`mean_center_charges: true` by default).
2. **`FixedBoardPoissonSolver`** stores precomputed Green matrices for the
   `L = len(lambdas)` screening constants. The 8x8 grid Laplacian uses
   Neumann (zero-flux) boundary conditions by default, with Dirichlet
   available via `boundary: dirichlet`. Solving is a deterministic einsum:

   ```
   u_{k,l}(x) = G_l @ vec(rho_k(x))            # G_l = (L + lambda_l I)^{-1}
   ```

   Section 9 of the markdown packet identifies three falsifier ablations
   that are exposed via the `ablation` config field, all of which preserve
   the `(N, N)` matrix contract so the head input dimensionality is
   unchanged:

   - `random_orthogonal_solver`: each Green matrix is replaced by a fixed
     deterministic orthogonal matrix scaled to match the Frobenius norm of
     the corresponding harmonic Green matrix. This destroys the harmonic
     distance law while preserving the global linear-projection structure.
   - `local_gaussian_solver`: each Green matrix is replaced by a row-
     normalised isotropic Gaussian blur with `sigma = 1/sqrt(lambda_l)`
     clamped to `[0.5, 6.0]` squares. This keeps smoothing but removes the
     long-range inverse-Laplacian coupling.
   - `charge_only_stats`: the solver is bypassed (`u = 0`) so only charge
     moments reach the head. This isolates the contribution of the
     potential field versus the learned charges alone.

3. **`PotentialStatsPool`** computes 11 statistics per `(charge k, lambda l)`
   pair:

   ```
   energy = rho_k^T u_{k,l}
   dirichlet = sum_{(a,b) in edges} (u_a - u_b)^2
   u_mean, u_std, u_max, u_min, u_absmax
   boundary_flux = mean_{boundary squares} u
   king_us_potential = mean over the side-to-move's king ring
   king_them_potential = mean over the opponent's king ring
   charge_magnitude = mean(|rho_k|)
   ```

   The total feature dimensionality fed to the head is
   `K * L * 11 + global_feature_dim`.
4. **`HarmonicPotentialHead`** is a `LayerNorm -> Linear -> ReLU -> Linear`
   MLP (with optional dropout) consuming the flattened stats together with
   a small global broadcast vector (side-to-move, four castling flags, an
   eight-way en-passant file mask, and the normalised king-ring sizes).
   It returns one puzzle logit; a symmetric `two_class_logits` diagnostic
   is produced by splitting the binary logit so reporting can use the
   binary contract.

## Output Contract

`forward(x)` returns a dictionary including
`logits` of shape `(B,)` for `num_classes=1`, `two_class_logits`,
`charge_potential_energy`, `charge_potential_energy_mean`,
`dirichlet_energy`, `dirichlet_energy_mean`, `potential_mean`,
`potential_std`, `potential_absmax`, `boundary_flux`, `king_us_potential`,
`king_them_potential`, `charge_magnitude`, `mechanism_energy`, and
`ablation_active`. Engine, verification, source, and CRTK metadata are
never used as input.

## Why This Is Not a Generic CNN

The central operator is a **fixed dense linear inverse** of the
Laplacian-plus-screening operator on the 8x8 grid, not a learned local
convolution stack. Every spatial coupling is global by construction: each
output cell of `u_{k,l}` is a fixed linear combination of all 64 input
cells of `rho_k`. The only learned pieces are the 1x1 charge encoder and
the head MLP; the spatial reasoning operator itself is not learned.

## Implementation Binding

- Registered model name: `harmonic_board_potential_network`
- Source implementation: `src/chess_nn_playground/models/harmonic_board_potential_network.py`
- Idea-local wrapper: `ideas/registry/i059_harmonic_board_potential_network/model.py`
  delegates to `build_harmonic_board_potential_network_from_config`.
- The idea-local wrapper does not import or call the shared
  `ResearchPacketProbe` / `build_research_packet_probe_from_config` scaffold.
