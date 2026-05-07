# Ablations

- Set `model.expected_mix: 1.0` to use only the direct piece-by-square product baseline.
- Set `model.expected_mix: 0.0` to use only the side-relative rank/file low-rank baseline.
- Reduce `model.depth` to 1 to test whether the signed residual signal survives a smaller residual-map head.
- Compare against matched raw-piece-plane and marginals-only controls before treating residualization as the source of any gain.
- Compare against LC0 BT4, NNUE, and the strongest registered idea runs on the same split and seeds.
