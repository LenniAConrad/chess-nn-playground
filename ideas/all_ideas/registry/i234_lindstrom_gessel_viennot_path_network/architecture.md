# Architecture

`Lindstrom-Gessel-Viennot Path Determinant Network` is a bespoke
implementation of idea `i234`. It builds a learned acyclic chess DAG on the
64 squares, evaluates its path-generating function via a truncated Neumann
series, and reads tactical content off the determinant of a path matrix
between soft-selected source and target squares. By the LGV lemma,
`det(M)` is the signed enumerator of non-intersecting source-to-target
`num_paths`-tuples in the DAG.

## Pipeline

- Input: board tensor `(B, 18, 8, 8)`. CRTK / source metadata is
  reporting-only and never used as model input.
- Convolutional trunk lifts each square to `channels` features.
- Two `1 x 1` projections produce per-square "left" and "right" edge
  embeddings of size `edge_embed_dim`. Their scaled outer product gives
  asymmetric edge scores; sigmoid gives non-negative edge weights, and a
  strictly upper-triangular mask under the row-major topological order
  (a8, b8, ..., h1) makes the resulting weighted graph a DAG.
- Row-stochastic normalisation keeps the spectral radius of `W` at most 1,
  so for `alpha = sigmoid(alpha_logit) * 0.99 < 1` the path-generating
  function `G = sum_{k>=1} (alpha W)^k` converges and the truncated series
  with `neumann_steps` terms is a faithful proxy.
- Two more `1 x 1` projections plus learned `source_queries` and
  `target_queries` produce soft-selection matrices `A_src`, `A_tgt` of
  shape `(num_paths, 64)` whose rows are softmaxes over the 64 squares.
  These play the role of source / target square choices in the LGV setup.
- Path matrix: `M = A_src @ G @ A_tgt^T`, of shape
  `(num_paths, num_paths)`. `M[i, j]` is the alpha-weighted enumerator of
  paths from soft-source `i` to soft-target `j`.
- LGV readout: `slogdet(M + det_eps * I)` returns
  `(sign(det), log|det|)`. Together with `tr(M)`, `||M||_F`, the
  per-row diagonal and off-diagonal `L1` magnitudes, the alpha scalar
  and the source / target selection entropies, these are the
  diagnostics fed to the puzzle head.
- A LayerNorm + GELU MLP head consumes pooled trunk features (mean and
  max) plus the LGV diagnostics and returns one puzzle logit.

## Implementation Binding

- Registered model name: `lindstrom_gessel_viennot_path_network` (registered in
  `src/chess_nn_playground/models/registry.py`).
- Source implementation file:
  `src/chess_nn_playground/models/lindstrom_gessel_viennot_path_network.py`
  (`LindstromGesselViennotPathNetwork` and
  `build_lindstrom_gessel_viennot_path_network_from_config`).
- Idea-local wrapper:
  `ideas/all_ideas/registry/i234_lindstrom_gessel_viennot_path_network/model.py` calls
  `build_lindstrom_gessel_viennot_path_network_from_config`.
- The shared `ResearchPacketProbe` scaffold is no longer used by this
  idea.
