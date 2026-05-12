# Architecture

`Determinantal Tactical Volume Bottleneck` realises the markdown thesis as a
bespoke model: the central computation is a **role-gated PSD log-volume** over
the set of occupied piece tokens, not a CNN, attention, sheaf, or move-delta
mechanism.

## Pipeline

1. **`Simple18OccupiedTokenExtractor`** decodes the simple_18 board tensor
   into up to `N_max = 32` occupied piece tokens. Each token carries a
   permutation-invariant feature vector built from current-board state only:
   12 piece-color one-hots, an own/enemy flag relative to side-to-move,
   absolute (row/7, col/7) and side-relative coordinates, four castling
   broadcast flags, and an en-passant target flag. Inactive token slots are
   masked to zero so they cannot leak into the determinant.
2. **`PieceSquareTokenEncoder`** maps token features through a small MLP
   `(B, N_max, F) -> (B, N_max, d)` with `d = token_dim` (default 48). The
   embedding is masked so unoccupied slots remain zero vectors.
3. **`RoleGatedPSDVolume`** computes, for each of `R = role_count` roles
   (default 8), the PSD Gram matrix from `math_thesis.md`:

   ```
   K_r(x) = D_r Phi A_r A_r^T Phi^T D_r + eps * I_N
   D_r    = diag(sqrt(g_{r,1} * mask_1), ..., sqrt(g_{r,N} * mask_N))
   ```

   The role projector `A_r in R^{d x q}` has rank `q = role_rank` (default
   16). The log-volume `V_r = log det K_r` is computed via the Sylvester /
   Weinstein identity in (q x q) space:

   ```
   log det(Z Z^T + eps I_N) = N log(eps) + log det(I_q + Z^T Z / eps)
   ```

   The constant `N log(eps)` is dropped, so the active log-volume only
   depends on the gated occupied tokens. Diagonal trace, top-eigenvalue
   ratio, gate mass, and active-count fractions are computed alongside the
   log-volume to give the head a small, interpretable per-role feature
   vector. The diagonal-trace ablation (`ablation: diagonal_trace_only`)
   replaces `V_r` with the gated diagonal trace, preserving gates / norms /
   role marginals while removing every off-diagonal determinant interaction;
   this is the central falsifier from section 9 of the markdown packet.
4. **`DeterminantalVolumeHead`** concatenates the `R x stats_per_role`
   per-role statistics with a small global broadcast vector
   (side-to-move, four castling flags, an eight-way en-passant file, and
   the normalised active-token count) and emits a single puzzle logit. A
   matching `two_class_logits` diagnostic is produced by symmetric splitting
   so reporting can use the binary contract.

## Permutation Invariance

For any token permutation matrix `P`,
`K_r(P x) = P D_r Phi A_r A_r^T Phi^T D_r P^T = P K_r(x) P^T`, so
`log det K_r` is unchanged. The bottleneck therefore cannot use the order in
which tokens appear in the extractor — it responds only to the spectrum of
the role-gated occupied-token covariance.

## Output Contract

`forward(x)` returns a dictionary including
`logits` of shape `(B,)` for `num_classes=1`, `two_class_logits`,
`log_volume`, `log_volume_mean`, `log_volume_max`, `log_volume_min`,
`trace`, `trace_mean`, `gate_mass`, `gate_mass_mean`, `top_eig_ratio`,
`top_eig_ratio_mean`, `active_count`, and `mechanism_energy`. Engine,
verification, source, and CRTK metadata are never used as input.

## Implementation Binding

- Registered model name: `determinantal_tactical_volume_bottleneck`
- Source implementation: `src/chess_nn_playground/models/determinantal_volume.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i058_determinantal_tactical_volume_bottleneck/model.py`
  delegates to `build_determinantal_tactical_volume_bottleneck_from_config`.
- The idea-local wrapper does not import or call the shared
  `ResearchPacketProbe` / `build_research_packet_probe_from_config` scaffold.
