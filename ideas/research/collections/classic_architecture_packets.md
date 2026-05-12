# Classic Architecture Packets

This collection groups the pre-primitive architecture research packets. It does not move the packet files, because registered ideas and prompts already cite many stable paths under `ideas/research/packets/classic/`.

Use this as the "classic architectures" view of the corpus: older packet-era ideas, practical NN baselines, puzzle-binary challengers, and mathematical operator architectures that are not part of the May 12 primitive folder.

## Major Families

| Family | Typical files or registered regions | What to check before adding more |
|---|---|---|
| Sheaf, Hodge, and tactical graph tension | early 2026-04-21 sheaf packets; registered ideas around the first tactical sheaf imports | Avoid another static attack/defense/x-ray sheaf unless the operator and falsifier change materially. |
| One-ply move-delta and counterfactual landscapes | `move_delta_*`, `move_landscape`; related `i025-i027` neighborhood | Avoid simple pseudo-legal move multiset pooling variants. |
| Transport and target-flow bottlenecks | transport, OT, material-null, geometry transport packets | Avoid another Sinkhorn target measure variant unless the mathematical object is genuinely different. |
| Orbit, automorphism, color-flip, and tempo parity | orbit quotient, automorphism quotient, color-flip, tempo-odd packets; `i041` neighborhood | Check whether the new idea is only another symmetry pooling or tempo toggle. |
| Topology, Hall defect, king path, and cage dynamics | Euler, Betti, Hall-defect, king-cage, percolation packets | Avoid threshold or shell-radius variants with the same observable. |
| Score fields, closure, and non-backtracking walks | concept closure, non-puzzle score field, non-backtracking tactical walk | Check that the state space and falsifier are not just vocabulary changes. |
| Linear algebra and spectral operators | determinant, Grassmannian, matrix pencil, Procrustes, bispectral, Krylov, resolvent, Gramian packets | Compare against the high-math registered block around `i221-i240`. |
| Practical CNN, token, attention, and residual baselines | multiscale CNN, piece-token CNN, set-query attention, residual and attention batches | Treat these as baselines or implementation scaffolds unless the architecture has a distinct falsifier. |
| Puzzle-binary hard-negative challengers | 2026-04-25 and 2026-04-28 puzzle batches | Check slice-level claims against near-puzzle false-positive and CRTK tag reports. |
| Robust, selective, and objective-side methods | DRO, bounded hinge, soft sort, VetoSelect, calibration packets | Keep separate from architecture claims unless the model wiring changes. |

## Where The Files Live

- Raw packet markdowns: `ideas/research/packets/classic/*.md`
- Generated packet catalog: `ideas/research/packets/CATALOG.md`
- Registered implementations: `ideas/registry/i###_*/`
- Current primitive-era work: [Primitive Research](primitive_research.md)

## Practical Use

When inventing or promoting a new idea, search this collection first, then search the generated catalog and registry:

```bash
rg -n "<concept words>" ideas/research/collections ideas/research/packets ideas/registry/i*_*/{math_thesis.md,architecture.md,ablations.md,idea.yaml}
```

If the closest match is only a raw packet, promote the new idea only after writing a clearer falsifier than the old packet had. If the closest match is already a registered idea, update or compare against that registered idea instead of creating a near-duplicate.
