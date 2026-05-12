# Ablations

- Set `model.mechanism_family: generic` to test the linear-algebra profile against
  a generic diagnostic profile.
- Reduce `model.depth` to 1 to test whether the mechanism survives a smaller trunk.
- Compare against LC0 BT4, NNUE, and the strongest registered linear-algebra ideas
  (i061 Grassmannian, i062 Matrix-Pencil, i076 Krylov, i077 Resolvent, i078 Gramian)
  on the same split and seeds.
- Source packet `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-05-05_1505_tuesday_local_schur_complement_defender.md` enumerates the central
  falsification ablations; once a bespoke `model.py` lands, port them here.
