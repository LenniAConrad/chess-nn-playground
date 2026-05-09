# Math Thesis

Pawn Skeleton Barrier Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md`.

Batch candidate rank: `2`.

Working thesis: Pawn structure is a slow, chess-specific skeleton that shapes king safety, open lines, promotion lanes, and tactical vulnerability. From the current board we can compute deterministic pawn barrier and distance fields — front spans, attack fronts, isolated/doubled/passed-pawn masks, open-file masks, distance-to-pawn and distance-to-open-file fields, and king shelter zones — purely from the simple_18 pawn and king planes. The bespoke architecture lifts these fields into a 30-channel side-canonical skeleton tensor, projects it with a small CNN, and uses a sigmoid gate over the projected pawn features to multiplicatively condition a compact board trunk. The classifier then pools the conditioned trunk through global, open-file, and own/opponent king-zone masks and combines those pools with deterministic pawn-structure scalars (pawn counts, isolated/doubled/passed-pawn shares, shelter distances, pawn-frontier density, open-file density). The hypothesis is that explicitly handing the model these chess-specific pawn-skeleton priors makes king-safety, blockade, and promotion-lane signals separable for `puzzle_binary` discrimination at modest parameter and FLOP cost.
