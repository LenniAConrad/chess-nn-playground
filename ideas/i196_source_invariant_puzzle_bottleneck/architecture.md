# Architecture

`Source-Invariant Puzzle Bottleneck` is a bespoke board-only puzzle_binary
architecture that forces the main puzzle representation to live in the
symmetry-orbit invariant subspace, which acts as a proxy for "removed source
identity". The thesis is in `math_thesis.md`.

The model consumes the repository `simple_18` current-board tensor `(B, 18,
8, 8)` and returns one puzzle logit for the BCE-with-logits `puzzle_binary`
trainer. CRTK / source metadata is reporting-only and is never used as model
input.

## Mechanism

1. **Conv trunk.** A compact `BoardFeatureTrunk` (Conv → BN/GroupNorm → GELU
   → optional Dropout2d, repeated `depth` times) produces a per-square
   feature map `(B, channels, 8, 8)`.

2. **Symmetry orbit.** A fixed orbit of board-level transformations is
   applied: identity, file flip (mirror across the central file), rank flip
   (mirror across the central rank), and 180-rotation (file + rank flip).
   The trunk weights are shared across views.

3. **Per-view pooling.** Each view is pooled by mean+max along the spatial
   dimensions, yielding `K = num_views` pooled feature vectors of size
   `2·channels`.

4. **Invariant bottleneck.** A shared MLP `g_φ` projects each per-view vector
   into a code of size `code_dim`. The orbit-mean code `c̄` and per-view
   residuals `r_k = c_k − c̄` are computed.

5. **Residual-direction orthogonalisation.** The dominant residual axis is
   the L2-normalised residual sum `u = Σ_k r_k / ||Σ_k r_k||`. The main code
   subtracts the projection of `c̄` onto that axis, gated by
   `α = sigmoid(orthogonalize_logit)`:

   ```text
   c_main = c̄ − α · (c̄ · u) · u
   ```

6. **Puzzle head.** A LayerNorm → Linear → GELU → (Dropout) → Linear head
   reads only `c_main` and produces the puzzle logit:

   ```text
   logits = h_ψ(c_main)
   ```

7. **Auxiliary residual head.** A separate small MLP reads the per-view
   residuals (mean across views) and produces `aux_residual_logit`, exposed
   as a diagnostic only. It does not enter the main puzzle prediction path.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for the
`puzzle_binary` BCE-with-logits trainer (`num_classes == 1`), plus
diagnostics, all of shape `(B,)`:

- `logits`, `invariant_code_norm`, `main_code_norm`, `residual_energy`,
  `residual_direction_strength`, `orthogonalize_gate`, `aux_residual_logit`,
  `view_consistency`, `num_views`, `mechanism_energy`, `symmetry_residual`,
  `proposal_profile_strength`, `proposal_keyword_count`.

`mechanism_energy` and `symmetry_residual` are aliases of `residual_energy`
to keep the puzzle_binary diagnostic packet contract consistent with the
`mechanism_family: symmetry` family expected by the source packet.

## Ablations

The constructor accepts the following ablations:

- `none` — the full source-invariant bottleneck described above.
- `no_invariance` — drop the symmetry orbit; only the identity view feeds
  the bottleneck. Tests whether multi-view averaging matters at all.
- `no_orthogonalization` — keep the orbit but skip the explicit
  residual-direction subtraction; `c_main = c̄`. Tests whether mean-pooling
  alone is enough.
- `no_aux_residual_logit` — drop the auxiliary residual logit head. Tests
  whether the residual-channel diagnostic is load-bearing.

## Implementation Binding

- Registered model name: `source_invariant_puzzle_bottleneck`
- Source implementation file: `src/chess_nn_playground/models/source_invariant_puzzle_bottleneck.py`
- Idea-local wrapper: `ideas/i196_source_invariant_puzzle_bottleneck/model.py`
