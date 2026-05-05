# ChatGPT Pro Prompt: Tested Runs Context

You are advising the `chess-nn-playground` research project from a pasted run summary.

Your job is to reason from tested evidence only. Do not invent results, do not treat unresolved labels as verified classes, and do not propose a new neural architecture unless the user explicitly asks for idea generation.

## Non-Negotiable Label Rules

- `known_non_puzzle`: `coarse_label = 0`, `fine_label = 0`
- `candidate_1_or_2_unresolved`: `coarse_label = 1`, `fine_label = null`
- `verified_near_puzzle`: `coarse_label = 1`, `fine_label = 1`
- `verified_puzzle`: `coarse_label = 1`, `fine_label = 2`

Unresolved candidate positions are not verified near-puzzles and are not verified real puzzles.

Stockfish scores, PVs, node counts, verification metadata, source labels, and proposed labels must not be used as neural-network input features. They may only be targets, metadata, audit fields, or weak-label proposal evidence.

## Current Tested Run Context

This block was generated automatically from local `results/*/metrics_final.json` and `run_metadata.json` files.

```json
{
  "active_label_policy": {
    "candidate_1_or_2_unresolved": {
      "coarse_label": 1,
      "fine_label": null
    },
    "known_non_puzzle": {
      "coarse_label": 0,
      "fine_label": 0
    },
    "verified_near_puzzle": {
      "coarse_label": 1,
      "fine_label": 1
    },
    "verified_puzzle": {
      "coarse_label": 1,
      "fine_label": 2
    }
  },
  "generated_at": "2026-04-30T17:09:21Z",
  "idea_registry": [
    {
      "created_at": "2026-04-25T00:43:00+08:00",
      "folder": "ideas/i001_chess_operator_basis_classifier",
      "idea_id": "i001",
      "name": "Chess Operator Basis Classifier",
      "short_thesis": "Mix a small learned basis of rule-shaped square operators for general chess classification.",
      "slug": "chess_operator_basis_classifier",
      "status": "implemented",
      "target_task": "general_chess_position_classification_first_puzzle_binary"
    },
    {
      "created_at": "2026-04-25T00:43:00+08:00",
      "folder": "ideas/i002_response_minimax_classifier",
      "idea_id": "i002",
      "name": "Response-Minimax Chess Classifier",
      "short_thesis": "Classify through a learned one-ply max-min bottleneck over actions and replies.",
      "slug": "response_minimax_classifier",
      "status": "implemented",
      "target_task": "general_chess_position_classification_first_puzzle_binary"
    },
    {
      "created_at": "2026-04-25T00:43:00+08:00",
      "folder": "ideas/i003_factor_agreement_classifier",
      "idea_id": "i003",
      "name": "Factor-Agreement Chess Classifier",
      "short_thesis": "Require agreement between grid, piece, relation, and global factors before trusting class evidence.",
      "slug": "factor_agreement_classifier",
      "status": "implemented",
      "target_task": "general_chess_position_classification_first_puzzle_binary"
    },
    {
      "created_at": "2026-04-25T00:50:34+08:00",
      "folder": "ideas/i004_puzzle_obligation_flow_network",
      "idea_id": "i004",
      "name": "Puzzle Obligation Flow Network",
      "short_thesis": "Classify puzzlehood through a learned obligation-resource flow residual.",
      "slug": "puzzle_obligation_flow_network",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-25T00:53:36+08:00",
      "folder": "ideas/i005_null_move_contrast_puzzle_network",
      "idea_id": "i005",
      "name": "Null-Move Contrast Puzzle Network",
      "short_thesis": "Classify puzzlehood through current-vs-null-move tempo contrast.",
      "slug": "null_move_contrast_puzzle_network",
      "status": "tested",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-25T00:53:36+08:00",
      "folder": "ideas/i006_proof_core_set_verifier",
      "idea_id": "i006",
      "name": "Proof-Core Set Verifier",
      "short_thesis": "Classify puzzlehood through a sparse selected proof core and verifier.",
      "slug": "proof_core_set_verifier",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-25T00:57:35+08:00",
      "folder": "ideas/i007_neural_proof_number_search",
      "idea_id": "i007",
      "name": "Neural Proof-Number Search Network",
      "short_thesis": "Classify puzzlehood through a bounded learned proof/disproof search tree.",
      "slug": "neural_proof_number_search",
      "status": "tested",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-25T01:00:33+08:00",
      "folder": "ideas/i008_boundary_edit_lagrangian_network",
      "idea_id": "i008",
      "name": "Boundary-Edit Lagrangian Network",
      "short_thesis": "Classify puzzlehood through learned minimum chess-edit energy to and from the puzzle boundary.",
      "slug": "boundary_edit_lagrangian_network",
      "status": "tested",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-25T01:03:44+08:00",
      "folder": "ideas/i009_tactical_equilibrium_network",
      "idea_id": "i009",
      "name": "Tactical Equilibrium Network",
      "short_thesis": "Classify puzzlehood through a learned attacker/defender tactical equilibrium layer.",
      "slug": "tactical_equilibrium_network",
      "status": "tested",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-25T01:07:11+08:00",
      "folder": "ideas/i010_rule_consistent_latent_dynamics",
      "idea_id": "i010",
      "name": "Rule-Consistent Latent Dynamics Network",
      "short_thesis": "Classify from a representation trained to model legal chess dynamics and latent consequences.",
      "slug": "rule_consistent_latent_dynamics",
      "status": "implemented",
      "target_task": "general_chess_position_classification_first_puzzle_binary"
    },
    {
      "created_at": "2026-04-28T00:00:00+08:00",
      "folder": "ideas/i011_vetoselect_positive_claim_abstention",
      "idea_id": "i011",
      "name": "VetoSelect Positive-Claim Abstention",
      "short_thesis": "Classify near-puzzle negatives as rejected positive evidence instead of ordinary low-confidence negatives.",
      "slug": "vetoselect_positive_claim_abstention",
      "status": "tested",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-29T00:00:00+08:00",
      "folder": "ideas/i012_dykstra_lcp",
      "idea_id": "i012",
      "name": "Soft-Dykstra Latent Constraint Projector",
      "short_thesis": "Classify puzzlehood through projection distance and correction trace of a board-only latent feasibility solver.",
      "slug": "dykstra_lcp",
      "status": "tested",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-29T00:00:00+08:00",
      "folder": "ideas/i013_sparse_relation_pursuit_asymmetry",
      "idea_id": "i013",
      "name": "Sparse Relation Pursuit Asymmetry",
      "short_thesis": "Classify puzzlehood through asymmetric reconstruction by equal-capacity background and tactical sparse relation dictionaries.",
      "slug": "sparse_relation_pursuit_asymmetry",
      "status": "tested",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T00:00:00+08:00",
      "folder": "ideas/i014_contamination_dro_huber_tail_rejection",
      "idea_id": "i014",
      "name": "Contamination-DRO Huber Tail Rejection",
      "short_thesis": "Classify puzzlehood with a bounded-influence upper-tail robust loss on near-puzzle margin violations.",
      "slug": "contamination_dro_huber_tail_rejection",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T00:00:00+08:00",
      "folder": "ideas/i015_material_locked_tactical_dro",
      "idea_id": "i015",
      "name": "Material-Locked Tactical Mask DRO",
      "short_thesis": "Classify puzzlehood under bounded contamination of deterministic tactical masks while material stays fixed.",
      "slug": "material_locked_tactical_dro",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T00:00:00+08:00",
      "folder": "ideas/i016_soft_sorting_order_residual_ranker",
      "idea_id": "i016",
      "name": "Soft Sorting Order Residual Ranker",
      "short_thesis": "Classify puzzlehood with a differentiable batch-order residual aligned to ranking metrics.",
      "slug": "soft_sorting_order_residual_ranker",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T00:00:00+08:00",
      "folder": "ideas/i017_conditional_surprisal_gate",
      "idea_id": "i017",
      "name": "Conditional Surprisal Gate",
      "short_thesis": "Classify puzzlehood through a gate bottleneck penalized against an easy board-statistics prior.",
      "slug": "conditional_surprisal_gate",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:30+00:00",
      "folder": "ideas/i018_oriented_tactical_sheaf_laplacian",
      "idea_id": "i018",
      "mechanism_family": "sheaf",
      "name": "Oriented Tactical Sheaf Laplacian",
      "short_thesis": "- Idea name: Oriented Tactical Sheaf Laplacian - One-sentence thesis: Classify puzzle-likeness by learning whether the board-only attack/defense incidence structure ha...",
      "slug": "oriented_tactical_sheaf_laplacian",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0254_tuesday_local_oriented_tactical_sheaf.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:30+00:00",
      "folder": "ideas/i019_tactical_sheaf_curvature_network",
      "idea_id": "i019",
      "mechanism_family": "sheaf",
      "name": "Tactical Sheaf Curvature Network",
      "short_thesis": "- Idea name: Tactical Sheaf Curvature Network, abbreviated `TSCN`. - One-sentence thesis: Chess puzzle-likeness is better detected as localized inconsistency in a boar...",
      "slug": "tactical_sheaf_curvature_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0254_tuesday_local_tactical_sheaf_curvature.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:30+00:00",
      "folder": "ideas/i020_attack_defense_sheaf_energy_network",
      "idea_id": "i020",
      "mechanism_family": "sheaf",
      "name": "Attack-Defense Sheaf Energy Network",
      "short_thesis": "- Idea name: Attack-Defense Sheaf Energy Network - One-sentence thesis: Puzzle-likeness in a chess position is better modeled as localized inconsistency in attack-defe...",
      "slug": "attack_defense_sheaf_energy_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0255_tuesday_local_attack_defense_sheaf.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:30+00:00",
      "folder": "ideas/i021_tactical_sheaf_tension_network",
      "idea_id": "i021",
      "mechanism_family": "sheaf",
      "name": "Tactical Sheaf Tension Network",
      "short_thesis": "- Idea name: Tactical Sheaf Tension Network, abbreviated `TSTN`. - One-sentence thesis: classify chess puzzle-likeness by learning a side-aware cellular sheaf over pse...",
      "slug": "tactical_sheaf_tension_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0255_tuesday_local_tactical_sheaf.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:30+00:00",
      "folder": "ideas/i022_tactical_threat_sheaf_network",
      "idea_id": "i022",
      "mechanism_family": "sheaf",
      "name": "Tactical Threat-Sheaf Network",
      "short_thesis": "- Idea name: Tactical Threat-Sheaf Network - One-sentence thesis: Chess puzzle-likeness is often signaled by localized incompatibility among attack, defense, pin, and...",
      "slug": "tactical_threat_sheaf_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0255_tuesday_local_threat_sheaf.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:30+00:00",
      "folder": "ideas/i023_attack_hodge_sheaf_tension_network",
      "idea_id": "i023",
      "mechanism_family": "sheaf",
      "name": "Attack-Hodge Sheaf Tension Network",
      "short_thesis": "- Idea name: Attack-Hodge Sheaf Tension Network, abbreviated `AHS-TensionNet`. - One-sentence thesis: chess puzzle-likeness is often signaled by locally inconsistent a...",
      "slug": "attack_hodge_sheaf_tension_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0256_tuesday_local_attack_hodge_sheaf.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:30+00:00",
      "folder": "ideas/i024_directed_attack_sheaf_tension_network",
      "idea_id": "i024",
      "mechanism_family": "sheaf",
      "name": "Directed Attack-Sheaf Tension Network",
      "short_thesis": "- Idea name: Directed Attack-Sheaf Tension Network - One-sentence thesis: A chess position is puzzle-like when its static attack geometry contains localized, asymmetri...",
      "slug": "directed_attack_sheaf_tension_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0427_tuesday_local_attack_sheaf_tension.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:30+00:00",
      "folder": "ideas/i025_one_ply_counterfactual_move_landscape_network",
      "idea_id": "i025",
      "mechanism_family": "move_delta",
      "name": "One-Ply Counterfactual Move Landscape Network",
      "short_thesis": "- Idea name: One-Ply Counterfactual Move Landscape Network, abbreviated `CML-Net` - One-sentence thesis: A chess position is puzzle-like when the deterministic, engine...",
      "slug": "one_ply_counterfactual_move_landscape_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0429_tuesday_local_move_landscape.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:30+00:00",
      "folder": "ideas/i026_counterfactual_move_delta_spectrum_network",
      "idea_id": "i026",
      "mechanism_family": "move_delta",
      "name": "Counterfactual Move-Delta Spectrum Network",
      "short_thesis": "- Idea name: Counterfactual Move-Delta Spectrum Network - One-sentence thesis: A chess position is puzzle-like when the rule-only one-ply board-delta neighborhood of t...",
      "slug": "counterfactual_move_delta_spectrum_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0429_tuesday_los_angeles_move_delta_spectrum.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:30+00:00",
      "folder": "ideas/i027_rule_only_counterfactual_move_delta_bottleneck",
      "idea_id": "i027",
      "mechanism_family": "move_delta",
      "name": "Rule-Only Counterfactual Move-Delta Bottleneck",
      "short_thesis": "- Idea name: Rule-Only Counterfactual Move-Delta Bottleneck, abbreviated `CDBN`. - One-sentence thesis: A position is puzzle-like when the side-to-move's rule-only one...",
      "slug": "rule_only_counterfactual_move_delta_bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0436_tuesday_los_angeles_move_delta_bottleneck.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:30+00:00",
      "folder": "ideas/i028_file_mirror_tension_sheaf",
      "idea_id": "i028",
      "mechanism_family": "sheaf",
      "name": "File-Mirror Tension Sheaf",
      "short_thesis": "- Idea name: File-Mirror Tension Sheaf - One-sentence thesis: Puzzle-like chess positions can be detected from board-only inputs by learning a small signed directed sh...",
      "slug": "file_mirror_tension_sheaf",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0437_tuesday_los_angeles_mirror_tension_sheaf.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:30+00:00",
      "folder": "ideas/i029_entropic_piece_target_transport_bottleneck",
      "idea_id": "i029",
      "mechanism_family": "transport",
      "name": "Entropic Piece-Target Transport Bottleneck",
      "short_thesis": "- Idea name: **Entropic Piece-Target Transport Bottleneck** - One-sentence thesis: A chess position is puzzle-like when side-to-move force can be geometrically coupled...",
      "slug": "entropic_piece_target_transport_bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0507_tuesday_los_angeles_transport_bottleneck.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:30+00:00",
      "folder": "ideas/i030_nuisance_orthogonal_puzzle_bottleneck",
      "idea_id": "i030",
      "mechanism_family": "generic",
      "name": "Nuisance-Orthogonal Puzzle Bottleneck",
      "short_thesis": "- Idea name: Nuisance-Orthogonal Puzzle Bottleneck, abbreviated `NOPB`. - One-sentence thesis: A chess puzzle-like position should remain recognizable after the model'...",
      "slug": "nuisance_orthogonal_puzzle_bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0508_tuesday_local_nuisance_orthogonal.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:30+00:00",
      "folder": "ideas/i031_tactical_transport_imbalance_network",
      "idea_id": "i031",
      "mechanism_family": "transport",
      "name": "Tactical Transport Imbalance Network",
      "short_thesis": "- Idea name: Tactical Transport Imbalance Network - One-sentence thesis: A chess puzzle-like position should often exhibit an asymmetric low-cost, low-entropy transpor...",
      "slug": "tactical_transport_imbalance_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0512_tuesday_local_transport_imbalance.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:30+00:00",
      "folder": "ideas/i032_king_anchored_material_null_transport_bottleneck",
      "idea_id": "i032",
      "mechanism_family": "transport",
      "name": "King-Anchored Material-Null Transport Bottleneck",
      "short_thesis": "- Idea name: King-Anchored Material-Null Transport Bottleneck, abbreviated `KAMN-OTB`. - One-sentence thesis: A puzzle-like position should often exhibit unusually eff...",
      "slug": "king_anchored_material_null_transport_bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0657_tuesday_local_material_ot_bottleneck.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:30+00:00",
      "folder": "ideas/i033_piece_target_entropic_transport_bottleneck",
      "idea_id": "i033",
      "mechanism_family": "transport",
      "name": "Piece-Target Entropic Transport Bottleneck",
      "short_thesis": "- Idea name: Piece-Target Entropic Transport Bottleneck, abbreviated **PT-ETB**. - One-sentence thesis: A puzzle-like chess position should often expose a low-entropy,...",
      "slug": "piece_target_entropic_transport_bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0657_tuesday_los_angeles_piece_transport.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i034_entropic_chess_geometry_transport_network",
      "idea_id": "i034",
      "mechanism_family": "transport",
      "name": "Entropic Chess Geometry Transport Network",
      "short_thesis": "- Idea name: Entropic Chess Geometry Transport Network, abbreviated ECGT-Net. - One-sentence thesis: Puzzle-like positions often contain an unusually organized transpo...",
      "slug": "entropic_chess_geometry_transport_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0703_tuesday_los_angeles_geom_ot.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i035_ordinal_evidence_ladder_network",
      "idea_id": "i035",
      "mechanism_family": "generic",
      "name": "Ordinal Evidence Ladder Network",
      "short_thesis": "- Idea name: Ordinal Evidence Ladder Network - One-sentence thesis: Treat `known non-puzzle -> verified near-puzzle -> verified puzzle` as an ordered ladder and force...",
      "slug": "ordinal_evidence_ladder_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0711_tuesday_los_angeles_ordinal_evidence_ladder.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i036_geometry_conditioned_board_pseudo_likelihood_ratio_network",
      "idea_id": "i036",
      "mechanism_family": "generic",
      "name": "Geometry-Conditioned Board Pseudo-Likelihood Ratio Network",
      "short_thesis": "- Idea name: Geometry-Conditioned Board Pseudo-Likelihood Ratio Network, abbreviated `GeomPLR`. - One-sentence thesis: classify puzzle-likeness by the log description-...",
      "slug": "geometry_conditioned_board_pseudo_likelihood_ratio_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0713_tuesday_local_geom_plr.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i037_mobius_piece_constellation_network",
      "idea_id": "i037",
      "mechanism_family": "sparse",
      "name": "M\u00f6bius Piece-Constellation Network",
      "short_thesis": "- Idea name: M\u00f6bius Piece-Constellation Network, abbreviated `MPCN`. - One-sentence thesis: Chess puzzle-likeness is often carried by sparse, unordered, high-order con...",
      "slug": "mobius_piece_constellation_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0713_tuesday_los_angeles_mobius_constellation.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i038_sparse_witness_piece_bottleneck_network",
      "idea_id": "i038",
      "mechanism_family": "sparse",
      "name": "Sparse Witness-Piece Bottleneck Network",
      "short_thesis": "- Idea name: Sparse Witness-Piece Bottleneck Network, abbreviated `swpb` - One-sentence thesis: Chess puzzle-likeness should often be decidable from a small witness se...",
      "slug": "sparse_witness_piece_bottleneck_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0713_tuesday_los_angeles_sparse_witness_bottleneck.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i039_ray_language_automaton_network",
      "idea_id": "i039",
      "mechanism_family": "grammar",
      "name": "Ray-Language Automaton Network",
      "short_thesis": "- Idea name: Ray-Language Automaton Network, abbreviated `RLAN`. - One-sentence thesis: Many chess puzzle-like positions contain short, ordered, gapped piece strings a...",
      "slug": "ray_language_automaton_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0719_tuesday_local_ray_language_automaton.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i040_kinematic_commutator_bottleneck_network",
      "idea_id": "i040",
      "mechanism_family": "generic",
      "name": "Kinematic Commutator Bottleneck Network",
      "short_thesis": "- Idea name: Kinematic Commutator Bottleneck Network, abbreviated KCBN. - One-sentence thesis: Puzzle-like positions should be enriched for non-commuting interactions...",
      "slug": "kinematic_commutator_bottleneck_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0728_tuesday_local_kinematic_commutator.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i041_centered_tempo_odd_interventional_bottleneck",
      "idea_id": "i041",
      "mechanism_family": "tempo",
      "name": "Centered Tempo-Odd Interventional Bottleneck",
      "short_thesis": "- Idea name: **Centered Tempo-Odd Interventional Bottleneck**. - One-sentence thesis: Puzzle-likeness should be predicted from the board-dependent part of the position...",
      "slug": "centered_tempo_odd_interventional_bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0729_tuesday_pacific_tempo_odd_bottleneck.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i042_legal_automorphism_quotient_network",
      "idea_id": "i042",
      "mechanism_family": "symmetry",
      "name": "Legal Automorphism Quotient Network",
      "short_thesis": "- Idea name: Legal Automorphism Quotient Network - One-sentence thesis: A chess puzzle-likeness classifier should quotient out the exact current-board automorphisms of...",
      "slug": "legal_automorphism_quotient_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0731_tuesday_los_angeles_orbit_quotient.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i043_side_canonical_rule_partition_invariant_bottleneck",
      "idea_id": "i043",
      "mechanism_family": "generic",
      "name": "Side-Canonical Rule-Partition Invariant Bottleneck",
      "short_thesis": "- Idea name: Side-Canonical Rule-Partition Invariant Bottleneck, abbreviated `SCRIB`. - One-sentence thesis: Puzzle-likeness should be predicted from side-relative tac...",
      "slug": "side_canonical_rule_partition_invariant_bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0732_tuesday_pdt_rule_partition_bottleneck.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i044_masked_board_code_length_surprise_network",
      "idea_id": "i044",
      "mechanism_family": "information",
      "name": "Masked Board Code-Length Surprise Network",
      "short_thesis": "- Idea name: **Masked Board Code-Length Surprise Network** (`MBCS-Net`) - One-sentence thesis: Train a label-free masked board codec to estimate how many nats it takes...",
      "slug": "masked_board_code_length_surprise_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0739_tuesday_los_angeles_masked_surprise_codec.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i045_credal_near_puzzle_evidence_network",
      "idea_id": "i045",
      "mechanism_family": "generic",
      "name": "Credal Near-Puzzle Evidence Network",
      "short_thesis": "- Idea name: Credal Near-Puzzle Evidence Network - One-sentence thesis: Train a binary puzzle-likeness classifier whose output is a Dirichlet evidence distribution, tr...",
      "slug": "credal_near_puzzle_evidence_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0750_tuesday_los_angeles_credal_evidence.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i046_rule_exact_orbit_bottleneck_network",
      "idea_id": "i046",
      "mechanism_family": "symmetry",
      "name": "Rule-Exact Orbit Bottleneck Network",
      "short_thesis": "- Idea name: **Rule-Exact Orbit Bottleneck Network** (`REOBN`) - One-sentence thesis: A chess position should remain equally puzzle-like under the exact color-flip aut...",
      "slug": "rule_exact_orbit_bottleneck_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0750_tuesday_los_angeles_orbit_bottleneck.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i047_color_flip_orbit_evidence_bottleneck",
      "idea_id": "i047",
      "mechanism_family": "symmetry",
      "name": "Color-Flip Orbit Evidence Bottleneck",
      "short_thesis": "- Idea name: Color-Flip Orbit Evidence Bottleneck, abbreviated `CFOEB` - One-sentence thesis: A chess position is puzzle-like only if its evidence survives the exact c...",
      "slug": "color_flip_orbit_evidence_bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0751_tuesday_los_angeles_color_flip_orbit.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i048_rule_automorphism_quotient_bottleneck_network",
      "idea_id": "i048",
      "mechanism_family": "symmetry",
      "name": "Rule-Automorphism Quotient Bottleneck Network",
      "short_thesis": "- Idea name: Rule-Automorphism Quotient Bottleneck Network, abbreviated `RAQ-Net`. - One-sentence thesis: A chess puzzle-like position should remain puzzle-like under...",
      "slug": "rule_automorphism_quotient_bottleneck_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0751_tuesday_pdt_automorphism_quotient.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i049_tempo_odd_bottleneck_network",
      "idea_id": "i049",
      "mechanism_family": "tempo",
      "name": "Tempo-Odd Bottleneck Network",
      "short_thesis": "- Idea name: Tempo-Odd Bottleneck Network, abbreviated `TempoOddBottleneckNet`. - One-sentence thesis: Puzzle-likeness is often a side-to-move interaction property, so...",
      "slug": "tempo_odd_bottleneck_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0755_tuesday_los_angeles_tempo_odd_bottleneck.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i050_king_anchored_euler_interaction_network",
      "idea_id": "i050",
      "mechanism_family": "topology",
      "name": "King-Anchored Euler Interaction Network",
      "short_thesis": "- Idea name: **King-Anchored Euler Interaction Network** - One-sentence thesis: Puzzle-like positions should show sharply organized swept contact, enclosure, and separ...",
      "slug": "king_anchored_euler_interaction_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0809_tuesday_local_euler_interaction.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i051_king_escape_percolation_network",
      "idea_id": "i051",
      "mechanism_family": "king_path",
      "name": "King Escape Percolation Network",
      "short_thesis": "- Idea name: **King Escape Percolation Network** - One-sentence thesis: A puzzle-like chess position often contains a frozen-board tactical cage around one king, so ex...",
      "slug": "king_escape_percolation_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0811_tuesday_pacific_king_escape_percolation.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i052_soft_king_cage_path_bottleneck_network",
      "idea_id": "i052",
      "mechanism_family": "king_path",
      "name": "Soft King-Cage Path Bottleneck Network",
      "short_thesis": "- Idea name: Soft King-Cage Path Bottleneck Network - One-sentence thesis: A chess position is puzzle-like partly when one king is separated from the broader board by...",
      "slug": "soft_king_cage_path_bottleneck_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0812_tuesday_los_angeles_king_cage_dp.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i053_hall_defect_obligation_matroid_network",
      "idea_id": "i053",
      "mechanism_family": "generic",
      "name": "Hall-Defect Obligation Matroid Network",
      "short_thesis": "- Idea name: Hall-Defect Obligation Matroid Network - One-sentence thesis: Puzzle-like chess positions often contain a static overload certificate: a small set of defe...",
      "slug": "hall_defect_obligation_matroid_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0813_tuesday_los_angeles_hall_defect.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i054_threat_topology_betti_bottleneck_network",
      "idea_id": "i054",
      "mechanism_family": "topology",
      "name": "Threat-Topology Betti Bottleneck Network",
      "short_thesis": "- Idea name: Threat-Topology Betti Bottleneck Network - One-sentence thesis: Puzzle-like chess positions often contain unusually coherent high-pressure tactical region...",
      "slug": "threat_topology_betti_bottleneck_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0814_tuesday_los_angeles_threat_topology.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i055_non_backtracking_tactical_walk_network",
      "idea_id": "i055",
      "mechanism_family": "graph",
      "name": "Non-Backtracking Tactical Walk Network",
      "short_thesis": "- Idea name: Non-Backtracking Tactical Walk Network - One-sentence thesis: Puzzle-like positions are disproportionately marked by short, directed chains of current-boa...",
      "slug": "non_backtracking_tactical_walk_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0922_tuesday_local_nonbacktracking_walk.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i056_non_puzzle_score_field_bottleneck_network",
      "idea_id": "i056",
      "mechanism_family": "generic",
      "name": "Non-Puzzle Score-Field Bottleneck Network",
      "short_thesis": "- Idea name: **Non-Puzzle Score-Field Bottleneck Network** - One-sentence thesis: Train a rule-safe denoising score prior only on verified non-puzzle boards, then clas...",
      "slug": "non_puzzle_score_field_bottleneck_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0922_tuesday_local_nonpuzzle_score_field.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i057_soft_formal_concept_closure_network",
      "idea_id": "i057",
      "mechanism_family": "generic",
      "name": "Soft Formal-Concept Closure Network",
      "short_thesis": "- Idea name: Soft Formal-Concept Closure Network - One-sentence thesis: Chess puzzle-likeness is partly expressed by small closed sets of co-occurring rule-derived boa...",
      "slug": "soft_formal_concept_closure_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-21_0922_tuesday_los_angeles_concept_closure.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i058_determinantal_tactical_volume_bottleneck",
      "idea_id": "i058",
      "mechanism_family": "generic",
      "name": "Determinantal Tactical Volume Bottleneck",
      "short_thesis": "- Idea name: Determinantal Tactical Volume Bottleneck - One-sentence thesis: Puzzle-like positions may concentrate the occupied pieces into a low-dimensional learned t...",
      "slug": "determinantal_tactical_volume_bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2044_friday_shanghai_determinantal_volume.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i059_harmonic_board_potential_network",
      "idea_id": "i059",
      "mechanism_family": "generic",
      "name": "Harmonic Board Potential Network",
      "short_thesis": "- Idea name: Harmonic Board Potential Network - One-sentence thesis: Puzzle-like positions may be identifiable by long-range board tension patterns that appear after s...",
      "slug": "harmonic_board_potential_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2045_friday_shanghai_harmonic_potential.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i060_tropical_constraint_circuit_network",
      "idea_id": "i060",
      "mechanism_family": "generic",
      "name": "Tropical Constraint Circuit Network",
      "short_thesis": "- Idea name: Tropical Constraint Circuit Network - One-sentence thesis: Puzzle-like positions may be better modeled as the near-satisfaction of a small number of laten...",
      "slug": "tropical_constraint_circuit_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2046_friday_shanghai_tropical_circuit.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i061_grassmannian_principal_angle_bottleneck",
      "idea_id": "i061",
      "mechanism_family": "generic",
      "name": "Grassmannian Principal-Angle Bottleneck",
      "short_thesis": "- Idea name: Grassmannian Principal-Angle Bottleneck - High-level linear algebra concept: Grassmannians, principal angles between subspaces, and canonical-correlation...",
      "slug": "grassmannian_principal_angle_bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2058_friday_shanghai_grassmannian_angles.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i062_matrix_pencil_generalized_spectrum_bottleneck",
      "idea_id": "i062",
      "mechanism_family": "linear_algebra",
      "name": "Matrix-Pencil Generalized Spectrum Bottleneck",
      "short_thesis": "- Idea name: Matrix-Pencil Generalized Spectrum Bottleneck - High-level linear algebra concept: matrix pencils, generalized eigenvalue problems, and generalized Raylei...",
      "slug": "matrix_pencil_generalized_spectrum_bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2101_friday_shanghai_matrix_pencil.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i063_polar_procrustes_alignment_bottleneck",
      "idea_id": "i063",
      "mechanism_family": "linear_algebra",
      "name": "Polar-Procrustes Alignment Bottleneck",
      "short_thesis": "- Idea name: Polar-Procrustes Alignment Bottleneck - High-level linear algebra concept: polar decomposition, orthogonal Procrustes alignment, and matrix strain spectra...",
      "slug": "polar_procrustes_alignment_bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2104_friday_shanghai_polar_procrustes.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i064_multi_scale_dilated_board_mixer_cnn",
      "idea_id": "i064",
      "mechanism_family": "generic",
      "name": "Multi-Scale Dilated Board Mixer CNN",
      "short_thesis": "- Idea name: Multi-Scale Dilated Board Mixer CNN - One-sentence thesis: A practical chess-board CNN should mix local, knight-distance, diagonal/ray-like, and board-wid...",
      "slug": "multi_scale_dilated_board_mixer_cnn",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2107_friday_shanghai_multiscale_cnn_mixer.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i065_piece_token_cnn_hybrid",
      "idea_id": "i065",
      "mechanism_family": "generic",
      "name": "Piece-Token CNN Hybrid",
      "short_thesis": "- Idea name: Piece-Token CNN Hybrid - One-sentence thesis: A strong regular chess-board benchmark should combine dense 8x8 convolutional features with an explicit occu...",
      "slug": "piece_token_cnn_hybrid",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2109_friday_shanghai_piece_token_cnn_hybrid.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i066_bispectral_phase_coupling_board_network",
      "idea_id": "i066",
      "mechanism_family": "generic",
      "name": "Bispectral Phase-Coupling Board Network",
      "short_thesis": "- Idea name: Bispectral Phase-Coupling Board Network - One-sentence thesis: Puzzle-like positions may have distinctive spatial phase-coupling patterns between piece pl...",
      "slug": "bispectral_phase_coupling_board_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2110_friday_shanghai_bispectral_phase.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i067_finite_field_character_sum_board_network",
      "idea_id": "i067",
      "mechanism_family": "generic",
      "name": "Finite-Field Character-Sum Board Network",
      "short_thesis": "- Idea name: Finite-Field Character-Sum Board Network - Heavy math concept: finite-field harmonic analysis, additive and multiplicative characters, Gauss/Jacobi-style...",
      "slug": "finite_field_character_sum_board_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2115_friday_shanghai_character_sums.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i068_schur_ray_line_algebra_network",
      "idea_id": "i068",
      "mechanism_family": "generic",
      "name": "Schur-Ray Line Algebra Network",
      "short_thesis": "Schur-Ray Line Algebra Network research-packet promotion.",
      "slug": "schur_ray_line_algebra_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2127_friday_shanghai_schur_ray_line_algebra.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i069_bitboard_shift_algebra_network",
      "idea_id": "i069",
      "mechanism_family": "generic",
      "name": "Bitboard Shift-Algebra Network",
      "short_thesis": "Bitboard Shift-Algebra Network research-packet promotion.",
      "slug": "bitboard_shift_algebra_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2131_friday_shanghai_bitboard_shift_algebra.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i070_relational_query_algebra_network",
      "idea_id": "i070",
      "mechanism_family": "generic",
      "name": "Relational Query Algebra Network",
      "short_thesis": "Relational Query Algebra Network research-packet promotion.",
      "slug": "relational_query_algebra_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2139_friday_shanghai_relational_query_algebra.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i071_variational_board_action_network",
      "idea_id": "i071",
      "mechanism_family": "generic",
      "name": "Variational Board Action Network",
      "short_thesis": "Variational Board Action Network research-packet promotion.",
      "slug": "variational_board_action_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2146_friday_shanghai_variational_board_action.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i072_tensor_core_square_pair_field_network",
      "idea_id": "i072",
      "mechanism_family": "linear_algebra",
      "name": "Tensor-Core Square-Pair Field Network",
      "short_thesis": "Tensor-Core Square-Pair Field Network research-packet promotion.",
      "slug": "tensor_core_square_pair_field_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2148_friday_shanghai_tensorcore_pairfield.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i073_tiny_chess_micronet",
      "idea_id": "i073",
      "mechanism_family": "generic",
      "name": "Tiny Chess MicroNet",
      "short_thesis": "Tiny Chess MicroNet research-packet promotion.",
      "slug": "tiny_chess_micronet",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2200_friday_shanghai_tiny_chess_micronet.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i074_puzzle_binary_benchmark_challengers",
      "idea_id": "i074",
      "mechanism_family": "generic",
      "name": "Puzzle-Binary Benchmark Challengers",
      "short_thesis": "This packet adds new ideas after the corrected benchmark was established:",
      "slug": "puzzle_binary_benchmark_challengers",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i075_tactical_bisimulation_puzzle_network",
      "idea_id": "i075",
      "mechanism_family": "generic",
      "name": "Tactical Bisimulation Puzzle Network",
      "short_thesis": "Tactical Bisimulation Puzzle Network research-packet promotion.",
      "slug": "tactical_bisimulation_puzzle_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0113_saturday_shanghai_tactical_bisimulation.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i076_krylov_tactical_subspace_network",
      "idea_id": "i076",
      "mechanism_family": "linear_algebra",
      "name": "Krylov Tactical Subspace Network",
      "short_thesis": "Krylov Tactical Subspace Network research-packet promotion.",
      "slug": "krylov_tactical_subspace_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_2000_saturday_shanghai_krylov_tactical_subspace.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i077_adaptive_tactical_resolvent_network",
      "idea_id": "i077",
      "mechanism_family": "generic",
      "name": "Adaptive Tactical Resolvent Network",
      "short_thesis": "Adaptive Tactical Resolvent Network research-packet promotion.",
      "slug": "adaptive_tactical_resolvent_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_2002_saturday_shanghai_adaptive_tactical_resolvent.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i078_tactical_controllability_gramian_network",
      "idea_id": "i078",
      "mechanism_family": "linear_algebra",
      "name": "Tactical Controllability Gramian Network",
      "short_thesis": "Tactical Controllability Gramian Network research-packet promotion.",
      "slug": "tactical_controllability_gramian_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_2004_saturday_shanghai_tactical_controllability_gramian.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i079_support_polar_zonotope_certificate_network",
      "idea_id": "i079",
      "mechanism_family": "linear_algebra",
      "name": "Support-Polar Zonotope Certificate Network",
      "short_thesis": "Build a classifier whose central computation is not average pooling, attention pooling, or a plain convolutional head. SPZC-Net maps the current board into a latent co...",
      "slug": "support_polar_zonotope_certificate_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_0718_tuesday_new_york_support_polar_zonotope.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i080_loop_frustration_curvature_network",
      "idea_id": "i080",
      "mechanism_family": "generic",
      "name": "Loop-Frustration Curvature Network",
      "short_thesis": "Build a network whose decisive layer is not a standard convolutional classifier, but a chess-board spin-glass observable: **loop-frustration curvature**. The model lea...",
      "slug": "loop_frustration_curvature_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_0729_tuesday_new_york_frustration_curvature.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i081_forcing_response_front_door_bottleneck",
      "idea_id": "i081",
      "mechanism_family": "generic",
      "name": "Forcing-Response Front-Door Bottleneck",
      "short_thesis": "Build **Forcing-Response Front-Door Bottleneck**: a model whose label head cannot directly consume raw board-surface style. The board is first converted into a set of...",
      "slug": "forcing_response_front_door_bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_0733_tuesday_new_york_forcing_response_bottleneck.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i082_chess_hypercut_polynomial_network",
      "idea_id": "i082",
      "mechanism_family": "graph",
      "name": "Chess Hypercut Polynomial Network",
      "short_thesis": "Select `CHPNet`: a current-board hypergraph model that builds deterministic chess-rule hyperedges over the 64 board squares and applies a high-order cut polynomial ove...",
      "slug": "chess_hypercut_polynomial_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_0733_tuesday_new_york_hypercut_poly.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i083_fisher_geodesic_tension_network",
      "idea_id": "i083",
      "mechanism_family": "information",
      "name": "Fisher-Geodesic Tension Network",
      "short_thesis": "Build a **Fisher-Geodesic Tension Network**: a small convolutional board encoder that maps each board into several learned categorical distributions over the 64 square...",
      "slug": "fisher_geodesic_tension_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_0755_tuesday_new_york_fisher_geodesic.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i084_typed_hypergraph_motif_grammar",
      "idea_id": "i084",
      "mechanism_family": "grammar",
      "name": "Typed Hypergraph Motif Grammar",
      "short_thesis": "Select **Typed Hypergraph Motif Grammar** as the research direction.",
      "slug": "typed_hypergraph_motif_grammar",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_0757_tuesday_new_york_motif_grammar.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i085_hall_defect_zeta_operator",
      "idea_id": "i085",
      "mechanism_family": "generic",
      "name": "Hall-Defect Zeta Operator",
      "short_thesis": "Selected concept: **Hall-Defect Zeta Operator**, a deterministic finite-algebraic board operator that computes local overload spectra in a pin-filtered defense relation.",
      "slug": "hall_defect_zeta_operator",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_0802_tuesday_new_york_hall_defect_zeta.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i086_differentiable_chess_fact_lattice",
      "idea_id": "i086",
      "mechanism_family": "logic",
      "name": "Differentiable Chess Fact Lattice",
      "short_thesis": "Build **DCFL**, a neural classifier with an explicit differentiable abstract interpretation bottleneck. The model converts a board tensor into interval-valued chess fa...",
      "slug": "differentiable_chess_fact_lattice",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_0857_tuesday_new_york_diff_ai.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i087_tactical_radius_filtration",
      "idea_id": "i087",
      "mechanism_family": "generic",
      "name": "Tactical Radius Filtration",
      "short_thesis": "Select **Tactical Radius Filtration**.",
      "slug": "tactical_radius_filtration",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_0857_tuesday_new_york_tactical_radius.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i088_traced_threat_motif_network",
      "idea_id": "i088",
      "mechanism_family": "generic",
      "name": "Traced Threat Motif Network",
      "short_thesis": "Select **Traced Threat Motif Network**.",
      "slug": "traced_threat_motif_network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_0857_tuesday_new_york_trace_motif.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i089_bounded_board_hinge_logic",
      "idea_id": "i089",
      "mechanism_family": "logic",
      "name": "Bounded Board Hinge Logic",
      "short_thesis": "Select **Bounded Board Hinge Logic**: a differentiable logic classifier that compiles a fixed library of typed, shallow PSL-style formulas into tensor operations over...",
      "slug": "bounded_board_hinge_logic",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_0859_tuesday_new_york_bounded_hinge.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i090_chess_mode_tucker_relation_certificate",
      "idea_id": "i090",
      "mechanism_family": "linear_algebra",
      "name": "Chess-Mode Tucker Relation Certificate",
      "short_thesis": "Build **CMTRC**, a compact neural architecture whose main learnable decision operator is not a CNN block, attention module, Transformer, pair-field model, or tensor ri...",
      "slug": "chess_mode_tucker_relation_certificate",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_0900_tuesday_new_york_relation_tucker.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:43:31+00:00",
      "folder": "ideas/i091_tactical_state_bottleneck_inference",
      "idea_id": "i091",
      "mechanism_family": "generic",
      "name": "Tactical State Bottleneck Inference",
      "short_thesis": "Select **Tactical State Bottleneck Inference**.",
      "slug": "tactical_state_bottleneck_inference",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_0901_tuesday_new_york_tactical_latent.md",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i092_parity_syndrome_puzzle_bottleneck",
      "idea_id": "i092",
      "mechanism_family": "robustness",
      "name": "Parity-Syndrome Puzzle Bottleneck",
      "short_thesis": "Puzzle-like positions may produce distinctive parity or syndrome patterns over current-board facts: not just which local facts are present, but whether learned sparse XOR-like constraints are satisfied or violated. This tests a mod-2 algebraic bottleneck ra...",
      "slug": "parity_syndrome_puzzle_bottleneck",
      "source_packet_candidate": "Parity-Syndrome Puzzle Bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2048_friday_shanghai_architecture_batch_2.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i093_wavelet_scattering_board_network",
      "idea_id": "i093",
      "mechanism_family": "robustness",
      "name": "Wavelet Scattering Board Network",
      "short_thesis": "Puzzle-like structure may live in multiscale arrangements of piece planes. A fixed wavelet scattering front end can test whether stable multiscale modulus features help beyond learned CNN filters while avoiding engine-specific priors.",
      "slug": "wavelet_scattering_board_network",
      "source_packet_candidate": "Wavelet Scattering Board Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2048_friday_shanghai_architecture_batch_2.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i094_convex_feasibility_residual_network",
      "idea_id": "i094",
      "mechanism_family": "robustness",
      "name": "Convex Feasibility Residual Network",
      "short_thesis": "Puzzle-like positions may be those that lie near the boundary of several learned safe convex feasibility regions in board-feature space. An unrolled projection layer can test whether distance-to-feasibility is useful without using closed-form nuisance resid...",
      "slug": "convex_feasibility_residual_network",
      "source_packet_candidate": "Convex Feasibility Residual Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2048_friday_shanghai_architecture_batch_2.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i095_rank_quantile_evidence_field_network",
      "idea_id": "i095",
      "mechanism_family": "information",
      "name": "Rank-Quantile Evidence Field Network",
      "short_thesis": "Puzzle-likeness may be driven by extreme sparse evidence fields rather than average board evidence. Differentiable rank and quantile pooling can test this while still allowing the classifier to see the full board, unlike a sparse witness mask.",
      "slug": "rank_quantile_evidence_field_network",
      "source_packet_candidate": "Rank-Quantile Evidence Field Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2048_friday_shanghai_architecture_batch_2.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i096_oriented_matroid_covector_bottleneck",
      "idea_id": "i096",
      "mechanism_family": "robustness",
      "name": "Oriented Matroid Covector Bottleneck",
      "short_thesis": "Puzzle-like positions may be characterized by sign-pattern arrangements of occupied pieces in learned tactical coordinate systems. A covector bottleneck records which side of learned hyperplanes each occupied piece lies on, then pools sign-pattern histograms.",
      "slug": "oriented_matroid_covector_bottleneck",
      "source_packet_candidate": "Oriented Matroid Covector Bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2048_friday_shanghai_architecture_batch_2.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i097_fixed_point_residual_defect_network",
      "idea_id": "i097",
      "mechanism_family": "generic",
      "name": "Fixed-Point Residual Defect Network",
      "short_thesis": "Puzzle-like positions may be harder for a learned board-state operator to equilibrate. Instead of classifying only the final latent, classify from the residual defects of an unrolled update process:",
      "slug": "fixed_point_residual_defect_network",
      "source_packet_candidate": "Fixed-Point Residual Defect Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i098_baseline_logit_residual_adapter",
      "idea_id": "i098",
      "mechanism_family": "grammar",
      "name": "Baseline Logit Residual Adapter",
      "short_thesis": "The existing simple CNN likely has systematic errors. A small residual adapter can test what information remains after the baseline logit and latent representation are known:",
      "slug": "baseline_logit_residual_adapter",
      "source_packet_candidate": "Baseline Logit Residual Adapter",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i099_coarse_to_fine_board_residual_pyramid",
      "idea_id": "i099",
      "mechanism_family": "generic",
      "name": "Coarse-to-Fine Board Residual Pyramid",
      "short_thesis": "A puzzle-like position may be present in details not explained by coarse board summaries. Build a residual pyramid over the board: classify from what remains after each scale's coarse reconstruction explains the finer scale.",
      "slug": "coarse_to_fine_board_residual_pyramid",
      "source_packet_candidate": "Coarse-to-Fine Board Residual Pyramid",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i100_independence_residual_interaction_network",
      "idea_id": "i100",
      "mechanism_family": "generic",
      "name": "Independence Residual Interaction Network",
      "short_thesis": "Some puzzle-like signals may be interactions that remain after subtracting a simple independence explanation of board occupancy. Instead of modeling all piece-square interactions directly, compute signed residuals:",
      "slug": "independence_residual_interaction_network",
      "source_packet_candidate": "Independence Residual Interaction Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i101_residual_calibration_error_field",
      "idea_id": "i101",
      "mechanism_family": "information",
      "name": "Residual Calibration Error Field",
      "short_thesis": "If the existing CNN has good accuracy but poor reliability on near-puzzles, a residual calibration architecture can predict where the baseline is likely overconfident. The model learns a spatial \"calibration error field\" and uses it to adjust logits or prod...",
      "slug": "residual_calibration_error_field",
      "source_packet_candidate": "Residual Calibration Error Field",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i102_set_query_attention_bottleneck",
      "idea_id": "i102",
      "mechanism_family": "graph",
      "name": "Set-Query Attention Bottleneck",
      "short_thesis": "Puzzle-like positions may be recognized by a small number of latent tactical questions, each expressed as an attention distribution over board tokens. The model should classify not from unconstrained token mixing, but from query attention statistics, attend...",
      "slug": "set_query_attention_bottleneck",
      "source_packet_candidate": "Set-Query Attention Bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i103_attention_disagreement_residual_network",
      "idea_id": "i103",
      "mechanism_family": "graph",
      "name": "Attention Disagreement Residual Network",
      "short_thesis": "Near-puzzle and puzzle-like positions may contain competing interpretations. Independent attention query families should disagree more on ambiguous or tactically dense boards. The classifier uses the residual disagreement among attention maps as evidence.",
      "slug": "attention_disagreement_residual_network",
      "source_packet_candidate": "Attention Disagreement Residual Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i104_cross_scale_attention_residual_network",
      "idea_id": "i104",
      "mechanism_family": "graph",
      "name": "Cross-Scale Attention Residual Network",
      "short_thesis": "Puzzle-like evidence may appear when fine-square attention cannot be predicted from coarse board context. This model computes attention from fine tokens to coarse tokens, reconstructs expected fine attention, and classifies from the residual attention map.",
      "slug": "cross_scale_attention_residual_network",
      "source_packet_candidate": "Cross-Scale Attention Residual Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i105_slot_attention_role_binding_network",
      "idea_id": "i105",
      "mechanism_family": "graph",
      "name": "Slot Attention Role Binding Network",
      "short_thesis": "Puzzle-like positions may be characterized by how occupied pieces bind to a small number of latent tactical roles. Slot attention can softly assign pieces to roles and expose role competition without selecting a hard witness subset.",
      "slug": "slot_attention_role_binding_network",
      "source_packet_candidate": "Slot Attention Role Binding Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i106_attention_perturbation_sensitivity_network",
      "idea_id": "i106",
      "mechanism_family": "graph",
      "name": "Attention Perturbation Sensitivity Network",
      "short_thesis": "Attention maps are often decorative unless perturbing attended regions changes evidence. This model uses deterministic attention-guided perturbation sensitivity as the bottleneck: how much the latent or logits move when high-attention versus low-attention b...",
      "slug": "attention_perturbation_sensitivity_network",
      "source_packet_candidate": "Attention Perturbation Sensitivity Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i107_kernel_mean_prototype_network",
      "idea_id": "i107",
      "mechanism_family": "sparse",
      "name": "Kernel Mean Prototype Network",
      "short_thesis": "Puzzle-like positions may be separable by the distribution of occupied piece tokens in a learned kernel feature space. Instead of attending to pieces or computing pairwise transport, embed the occupied-piece set as a kernel mean and compare it to learned pr...",
      "slug": "kernel_mean_prototype_network",
      "source_packet_candidate": "Kernel Mean Prototype Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i108_tensorsketch_interaction_network",
      "idea_id": "i108",
      "mechanism_family": "generic",
      "name": "TensorSketch Interaction Network",
      "short_thesis": "Some puzzle-like signals may require high-order interactions among piece-square facts. Exact high-order tuple enumeration is expensive and overlaps with Mobius/ANOVA packets, but TensorSketch can approximate polynomial-kernel interactions with a compact ran...",
      "slug": "tensorsketch_interaction_network",
      "source_packet_candidate": "TensorSketch Interaction Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i109_maxout_region_signature_network",
      "idea_id": "i109",
      "mechanism_family": "generic",
      "name": "Maxout Region Signature Network",
      "short_thesis": "Puzzle-like boards may fall into distinctive piecewise-linear activation regions. A maxout bank can expose those regions directly by reporting winner identities, margins, and region-transition statistics.",
      "slug": "maxout_region_signature_network",
      "source_packet_candidate": "Maxout Region Signature Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i110_spline_board_surface_network",
      "idea_id": "i110",
      "mechanism_family": "grammar",
      "name": "Spline Board Surface Network",
      "short_thesis": "Chess boards may benefit from a smooth geometric baseline that is not convolutional. Fit learned tensor-product spline surfaces to piece planes and classify from low-degree surface coefficients plus residual maps.",
      "slug": "spline_board_surface_network",
      "source_packet_candidate": "Spline Board Surface Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i111_boundary_condition_disagreement_cnn",
      "idea_id": "i111",
      "mechanism_family": "generic",
      "name": "Boundary-Condition Disagreement CNN",
      "short_thesis": "Chess board edges matter: pawns, rooks, kings, and tactics behave differently near boundaries. A CNN's padding convention imposes a boundary assumption. Run a shared CNN under multiple boundary conditions and classify from disagreement.",
      "slug": "boundary_condition_disagreement_cnn",
      "source_packet_candidate": "Boundary-Condition Disagreement CNN",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i112_piece_drop_stability_network",
      "idea_id": "i112",
      "mechanism_family": "robustness",
      "name": "Piece-Drop Stability Network",
      "short_thesis": "Puzzle-like positions may be less stable under deterministic removal of specific safe current-board evidence groups. Instead of forcing a classifier to use sparse witnesses, measure how a small encoder's latent changes when piece groups are dropped.",
      "slug": "piece_drop_stability_network",
      "source_packet_candidate": "Piece-Drop Stability Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md",
      "source_packet_rank": 6,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i113_row_file_factor_mixer",
      "idea_id": "i113",
      "mechanism_family": "generic",
      "name": "Row-File Factor Mixer",
      "short_thesis": "Chess boards have two privileged axes: ranks and files. A model can exploit this without a full Transformer by factorizing board processing into rank mixers, file mixers, and piece-channel mixers, then recombining them with bilinear interactions.",
      "slug": "row_file_factor_mixer",
      "source_packet_candidate": "Row-File Factor Mixer",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i114_piece_conditioned_hypernetwork_cnn",
      "idea_id": "i114",
      "mechanism_family": "generic",
      "name": "Piece-Conditioned Hypernetwork CNN",
      "short_thesis": "The best local filters may depend on material and piece inventory. A lightweight hypernetwork can condition CNN channel gates or depthwise kernels on safe current-board summaries, adapting the feature extractor without using engine metadata.",
      "slug": "piece_conditioned_hypernetwork_cnn",
      "source_packet_candidate": "Piece-Conditioned Hypernetwork CNN",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i115_neural_board_cellular_automaton",
      "idea_id": "i115",
      "mechanism_family": "grammar",
      "name": "Neural Board Cellular Automaton",
      "short_thesis": "Some board patterns may be recognized by repeated local relaxation. A neural cellular automaton applies the same local update rule for several steps and classifies from the evolving board state and update energy.",
      "slug": "neural_board_cellular_automaton",
      "source_packet_candidate": "Neural Board Cellular Automaton",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i116_symmetric_difference_twin_encoder",
      "idea_id": "i116",
      "mechanism_family": "generic",
      "name": "Symmetric Difference Twin Encoder",
      "short_thesis": "Safe deterministic board transforms should preserve some evidence and change other evidence. Instead of enforcing invariance, compare the original and transformed board latents by symmetric difference features.",
      "slug": "symmetric_difference_twin_encoder",
      "source_packet_candidate": "Symmetric Difference Twin Encoder",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i117_prototype_patch_dictionary_network",
      "idea_id": "i117",
      "mechanism_family": "sparse",
      "name": "Prototype Patch Dictionary Network",
      "short_thesis": "Puzzle-like positions may contain local motifs, but a standard CNN may hide them in distributed filters. A learned patch dictionary can expose motif assignments, reconstruction residuals, and prototype activation histograms.",
      "slug": "prototype_patch_dictionary_network",
      "source_packet_candidate": "Prototype Patch Dictionary Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i118_channel_dropout_consensus_network",
      "idea_id": "i118",
      "mechanism_family": "robustness",
      "name": "Channel Dropout Consensus Network",
      "short_thesis": "The classifier should not depend too heavily on one piece channel or artifact. Train several shared encoders on deterministic channel-dropped views and classify from consensus and disagreement.",
      "slug": "channel_dropout_consensus_network",
      "source_packet_candidate": "Channel Dropout Consensus Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md",
      "source_packet_rank": 6,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i119_tensor_ring_square_interaction_network",
      "idea_id": "i119",
      "mechanism_family": "generic",
      "name": "Tensor-Ring Square Interaction Network",
      "short_thesis": "Many chess cues depend on interactions among several squares at once: king square, attacking piece, blocker, defender, escape square, and promotion path. A full square-pair or square-tuple interaction tensor is too large. A tensor-ring factorization can mod...",
      "slug": "tensor_ring_square_interaction_network",
      "source_packet_candidate": "Tensor-Ring Square Interaction Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i120_sinkhorn_role_assignment_network",
      "idea_id": "i120",
      "mechanism_family": "generic",
      "name": "Sinkhorn Role Assignment Network",
      "short_thesis": "Puzzle-like positions often contain latent tactical roles: target king, forcing piece, blocker, loose defender, escape square, overloaded piece, promotion candidate. Instead of asking attention to discover these roles implicitly, assign board objects to a f...",
      "slug": "sinkhorn_role_assignment_network",
      "source_packet_candidate": "Sinkhorn Role Assignment Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i121_morphological_threat_field_network",
      "idea_id": "i121",
      "mechanism_family": "logic",
      "name": "Morphological Threat Field Network",
      "short_thesis": "CNNs learn filters, but chess tactics often have shape operations: expand a king danger zone, close gaps in a pawn shield, erode escape squares, and detect thin corridors. Differentiable mathematical morphology gives an architecture that explicitly processe...",
      "slug": "morphological_threat_field_network",
      "source_packet_candidate": "Morphological Threat Field Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i122_invertible_board_coupling_network",
      "idea_id": "i122",
      "mechanism_family": "generic",
      "name": "Invertible Board Coupling Network",
      "short_thesis": "Standard encoders can discard information early, which makes it hard to know whether a model learned legitimate current-board structure or fragile shortcuts. A reversible board encoder preserves information by construction and classifies from latent distort...",
      "slug": "invertible_board_coupling_network",
      "source_packet_candidate": "Invertible Board Coupling Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i123_sparse_expert_board_router",
      "idea_id": "i123",
      "mechanism_family": "generic",
      "name": "Sparse Expert Board Router",
      "short_thesis": "Chess positions are heterogeneous. Endgames, king attacks, pawn races, blocked centers, and material imbalances may need different feature extractors. A sparse mixture of small board experts can route positions to specialized encoders without requiring a gi...",
      "slug": "sparse_expert_board_router",
      "source_packet_candidate": "Sparse Expert Board Router",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i124_local_neighborhood_geometry_network",
      "idea_id": "i124",
      "mechanism_family": "generic",
      "name": "Local Neighborhood Geometry Network",
      "short_thesis": "A puzzle-like position may be locally sharp: small current-board perturbations such as removing one piece plane, masking one square neighborhood, or reflecting a safe orientation can move its representation more than a quiet non-puzzle position. The classif...",
      "slug": "local_neighborhood_geometry_network",
      "source_packet_candidate": "Local Neighborhood Geometry Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md",
      "source_packet_rank": 6,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i125_ray_state_space_scan_network",
      "idea_id": "i125",
      "mechanism_family": "grammar",
      "name": "Ray State-Space Scan Network",
      "short_thesis": "Chess line motifs often require long-range context, but all-square attention and dynamic attack graphs are not the only way to get it. A state-space scan can process every rank, file, diagonal, and anti-diagonal as a short sequence with shared continuous re...",
      "slug": "ray_state_space_scan_network",
      "source_packet_candidate": "Ray State-Space Scan Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i126_pawn_skeleton_barrier_network",
      "idea_id": "i126",
      "mechanism_family": "generic",
      "name": "Pawn Skeleton Barrier Network",
      "short_thesis": "Pawn structure is a slow, chess-specific skeleton that shapes king safety, open lines, promotion lanes, and tactical vulnerability. A model can compute deterministic pawn barrier and distance fields from the current board, then learn how these fields condit...",
      "slug": "pawn_skeleton_barrier_network",
      "source_packet_candidate": "Pawn Skeleton Barrier Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i127_square_color_parity_mixer",
      "idea_id": "i127",
      "mechanism_family": "generic",
      "name": "Square-Color Parity Mixer",
      "short_thesis": "The chessboard is naturally bipartite by square color. Bishops stay on one color, knights alternate color, kings and queens mix colors locally, and pawn captures switch files and square color. A neural model can explicitly split dark/light square subspaces...",
      "slug": "square_color_parity_mixer",
      "source_packet_candidate": "Square-Color Parity Mixer",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i128_occupancy_run_length_segment_encoder",
      "idea_id": "i128",
      "mechanism_family": "generic",
      "name": "Occupancy Run-Length Segment Encoder",
      "short_thesis": "Sliding tactics depend on contiguous empty and occupied segments along ranks, files, and diagonals. Instead of parsing full piece-token ray strings, encode run-length segment summaries: empty run lengths, blocker positions, endpoint piece types, and segment...",
      "slug": "occupancy_run_length_segment_encoder",
      "source_packet_candidate": "Occupancy Run-Length Segment Encoder",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i129_king_shelter_microkernel_network",
      "idea_id": "i129",
      "mechanism_family": "generic",
      "name": "King-Shelter Microkernel Network",
      "short_thesis": "Many puzzles are decided near the king. A specialized side-relative microkernel branch can examine king neighborhoods, escape rings, near sliders, and local blockers with high resolution while the main CNN handles global context.",
      "slug": "king_shelter_microkernel_network",
      "source_packet_candidate": "King-Shelter Microkernel Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i130_material_phase_low_rank_adapter_network",
      "idea_id": "i130",
      "mechanism_family": "generic",
      "name": "Material-Phase Low-Rank Adapter Network",
      "short_thesis": "Chess positions vary greatly by material phase. Instead of one encoder for every position, condition low-rank adapter weights on material summaries while keeping a shared backbone. The architecture tests whether small material-conditioned rank updates impro...",
      "slug": "material_phase_low_rank_adapter_network",
      "source_packet_candidate": "Material-Phase Low-Rank Adapter Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md",
      "source_packet_rank": 6,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i131_replicator_payoff_piece_dynamics",
      "idea_id": "i131",
      "mechanism_family": "generic",
      "name": "Replicator Payoff Piece Dynamics",
      "short_thesis": "Puzzle-like positions often feel like unstable games among pieces: one attacker increases pressure, one defender is overloaded, one target becomes strategically dominant. A differentiable payoff game over occupied pieces can model this as a small dynamical...",
      "slug": "replicator_payoff_piece_dynamics",
      "source_packet_candidate": "Replicator Payoff Piece Dynamics",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i132_differentiable_bitboard_boolean_network",
      "idea_id": "i132",
      "mechanism_family": "generic",
      "name": "Differentiable Bitboard Boolean Network",
      "short_thesis": "Chess rules are often written as bitboard Boolean algebra: masks, shifts, intersections, unions, and complements. A neural model can learn soft bitboard predicates and combine them with differentiable Boolean operations, producing an efficient symbolic-neur...",
      "slug": "differentiable_bitboard_boolean_network",
      "source_packet_candidate": "Differentiable Bitboard Boolean Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i133_orthogonal_board_moment_network",
      "idea_id": "i133",
      "mechanism_family": "generic",
      "name": "Orthogonal Board Moment Network",
      "short_thesis": "Puzzle-like positions may differ in global spatial moments of piece fields: centralization, skew, diagonal concentration, king-side imbalance, and high-order shape. Orthogonal polynomial moments provide a compact linear-algebra descriptor that is not convol...",
      "slug": "orthogonal_board_moment_network",
      "source_packet_candidate": "Orthogonal Board Moment Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i134_legal_constraint_projection_residual_network",
      "idea_id": "i134",
      "mechanism_family": "generic",
      "name": "Legal-Constraint Projection Residual Network",
      "short_thesis": "Even when the input board is legal, a learned latent explanation of \"why this is puzzle-like\" may produce soft piece/square beliefs that violate basic legal-board constraints. Projecting those beliefs back onto a soft legal-board constraint set and reading...",
      "slug": "legal_constraint_projection_residual_network",
      "source_packet_candidate": "Legal-Constraint Projection Residual Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i135_zobrist_kernel_feature_network",
      "idea_id": "i135",
      "mechanism_family": "generic",
      "name": "Zobrist Kernel Feature Network",
      "short_thesis": "Zobrist hashing gives chess a compact random fingerprint of piece-square occupancy. A neural model can use many fixed Zobrist-style random feature maps as a cheap kernel approximation, then learn a small classifier over stable board fingerprints.",
      "slug": "zobrist_kernel_feature_network",
      "source_packet_candidate": "Zobrist Kernel Feature Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i136_low_rank_signed_cut_query_network",
      "idea_id": "i136",
      "mechanism_family": "generic",
      "name": "Low-Rank Signed Cut Query Network",
      "short_thesis": "Puzzle-like positions may separate the board into tense regions: attacking mass versus defending mass, king-side versus center, blocked wing versus open wing. A model can learn low-rank signed cut queries over board fields and classify from imbalance statis...",
      "slug": "low_rank_signed_cut_query_network",
      "source_packet_candidate": "Low-Rank Signed Cut Query Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md",
      "source_packet_rank": 6,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i137_commutative_view_consistency_network",
      "idea_id": "i137",
      "mechanism_family": "generic",
      "name": "Commutative View-Consistency Network",
      "short_thesis": "A chess position can be represented through several safe current-board views:",
      "slug": "commutative_view_consistency_network",
      "source_packet_candidate": "Commutative View-Consistency Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i138_support_function_envelope_network",
      "idea_id": "i138",
      "mechanism_family": "generic",
      "name": "Support-Function Envelope Network",
      "short_thesis": "A chess position has geometric envelopes: where the side-to-move has force, where the opponent has force, how far pieces extend toward the king, and how concentrated material is along important directions. A differentiable support-function readout can summa...",
      "slug": "support_function_envelope_network",
      "source_packet_candidate": "Support-Function Envelope Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i139_soft_majorization_line_sorter",
      "idea_id": "i139",
      "mechanism_family": "grammar",
      "name": "Soft Majorization Line Sorter",
      "short_thesis": "On a tactical line, the exact order and dominance of pieces often matters more than a bag of line pieces. Instead of a ray automaton or line language model, compute differentiable sorted salience profiles along ranks/files/diagonals and classify from majori...",
      "slug": "soft_majorization_line_sorter",
      "source_packet_candidate": "Soft Majorization Line Sorter",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i140_low_displacement_rank_board_operator",
      "idea_id": "i140",
      "mechanism_family": "generic",
      "name": "Low-Displacement-Rank Board Operator",
      "short_thesis": "Global square mixing can be parameterized by structured matrices instead of dense attention or convolutions. A low-displacement-rank operator over the flattened board can express long-range interactions with Toeplitz/Hankel-like structure and few parameters.",
      "slug": "low_displacement_rank_board_operator",
      "source_packet_candidate": "Low-Displacement-Rank Board Operator",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i141_submodular_coverage_bottleneck",
      "idea_id": "i141",
      "mechanism_family": "generic",
      "name": "Submodular Coverage Bottleneck",
      "short_thesis": "Puzzle evidence may behave like coverage: once a tactical theme is strongly present, another redundant cue adds less value, but a distinct cue adds more. A differentiable submodular coverage layer can force the model to aggregate learned concepts with dimin...",
      "slug": "submodular_coverage_bottleneck",
      "source_packet_candidate": "Submodular Coverage Bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i142_pivot_trace_elimination_network",
      "idea_id": "i142",
      "mechanism_family": "generic",
      "name": "Pivot Trace Elimination Network",
      "short_thesis": "Gaussian elimination exposes interaction structure through pivot sizes, residual norms, and Schur updates. A chess board can be encoded into a small square matrix, then passed through a fixed-order differentiable elimination procedure. The pivot trace becom...",
      "slug": "pivot_trace_elimination_network",
      "source_packet_candidate": "Pivot Trace Elimination Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md",
      "source_packet_rank": 6,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i143_convnext_boardnet",
      "idea_id": "i143",
      "mechanism_family": "generic",
      "name": "ConvNeXt BoardNet",
      "short_thesis": "Use a small ConvNeXt-style architecture adapted to `8 x 8` chess boards: depthwise spatial mixing, inverted channel MLPs, residual scaling, coordinate planes, and a strong global pooling head.",
      "slug": "convnext_boardnet",
      "source_packet_candidate": "ConvNeXt BoardNet",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i144_board_fpn_cnn",
      "idea_id": "i144",
      "mechanism_family": "generic",
      "name": "Board FPN CNN",
      "short_thesis": "Chess positions often need both exact square detail and coarse whole-board phase. A plain feature-pyramid network can process the board at `8 x 8`, `4 x 4`, and `2 x 2` resolutions, then fuse the maps back into a single classifier.",
      "slug": "board_fpn_cnn",
      "source_packet_candidate": "Board FPN CNN",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i145_piece_plane_gated_cnn",
      "idea_id": "i145",
      "mechanism_family": "generic",
      "name": "Piece-Plane Gated CNN",
      "short_thesis": "The `simple_18` channels are not arbitrary image channels. A plain CNN can respect this by first processing semantically related channel groups, then using learned gates to mix piece types and colors.",
      "slug": "piece_plane_gated_cnn",
      "source_packet_candidate": "Piece-Plane Gated CNN",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i146_patch_mixer_boardnet",
      "idea_id": "i146",
      "mechanism_family": "generic",
      "name": "Patch Mixer BoardNet",
      "short_thesis": "Use a plain MLP-Mixer-style model over `2 x 2` chess patches. This is a simple non-attention alternative to square-token models: mix information across board patches with MLPs, then mix channels with MLPs.",
      "slug": "patch_mixer_boardnet",
      "source_packet_candidate": "Patch Mixer BoardNet",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i147_specialist_head_cnn",
      "idea_id": "i147",
      "mechanism_family": "generic",
      "name": "Specialist-Head CNN",
      "short_thesis": "A plain shared CNN trunk can feed several small specialist heads: king-zone head, center-control head, material/phase head, and global board head. A learned fusion layer combines their logits/features. This tests specialization without a full mixture-of-exp...",
      "slug": "specialist_head_cnn",
      "source_packet_candidate": "Specialist-Head CNN",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i148_shallow_wide_residual_boardnet",
      "idea_id": "i148",
      "mechanism_family": "generic",
      "name": "Shallow Wide Residual BoardNet",
      "short_thesis": "On an `8 x 8` board, depth may be less useful than width and a good head. A shallow wide residual CNN can test whether the benchmark wants broad feature extraction rather than long convolutional stacks.",
      "slug": "shallow_wide_residual_boardnet",
      "source_packet_candidate": "Shallow Wide Residual BoardNet",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md",
      "source_packet_rank": 6,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i149_axial_rank_file_convnet",
      "idea_id": "i149",
      "mechanism_family": "generic",
      "name": "Axial Rank-File ConvNet",
      "short_thesis": "Use ordinary convolutions, but factor long-range board mixing into alternating `8`-length rank and file convolutions. This gives every square access to same-rank and same-file context cheaply while preserving an ordinary CNN training path.",
      "slug": "axial_rank_file_convnet",
      "source_packet_candidate": "Axial Rank-File ConvNet",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i150_early_exit_cascade_boardnet",
      "idea_id": "i150",
      "mechanism_family": "generic",
      "name": "Early-Exit Cascade BoardNet",
      "short_thesis": "Some positions may be easy and should not need a heavy model, while ambiguous near-puzzles need deeper computation. Build a cascade with several classifier exits and train it to produce useful early predictions plus a final refined prediction.",
      "slug": "early_exit_cascade_boardnet",
      "source_packet_candidate": "Early-Exit Cascade BoardNet",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i151_auxiliary_reconstruction_boardnet",
      "idea_id": "i151",
      "mechanism_family": "generic",
      "name": "Auxiliary Reconstruction BoardNet",
      "short_thesis": "A classifier trunk may discard board detail too early. Add a lightweight decoder that reconstructs safe current-board planes from the latent feature map, using reconstruction only as an auxiliary training loss. The classifier still sees no future or engine...",
      "slug": "auxiliary_reconstruction_boardnet",
      "source_packet_candidate": "Auxiliary Reconstruction BoardNet",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i152_iterative_logit_refinement_cnn",
      "idea_id": "i152",
      "mechanism_family": "generic",
      "name": "Iterative Logit Refinement CNN",
      "short_thesis": "Instead of producing a single logit vector at the end, let a model make an initial prediction and then apply several learned correction steps from shared board features. The model tests whether puzzle evidence is better accumulated as staged corrections.",
      "slug": "iterative_logit_refinement_cnn",
      "source_packet_candidate": "Iterative Logit Refinement CNN",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i153_agreement_variance_head_net",
      "idea_id": "i153",
      "mechanism_family": "generic",
      "name": "Agreement-Variance Head Net",
      "short_thesis": "Use one shared trunk and several cheap heads trained on the same label. Classify from the mean logits, and log head variance as an uncertainty diagnostic. This is a lightweight alternative to full ensembles.",
      "slug": "agreement_variance_head_net",
      "source_packet_candidate": "Agreement-Variance Head Net",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i154_adapter_sandwich_residual_cnn",
      "idea_id": "i154",
      "mechanism_family": "generic",
      "name": "Adapter-Sandwich Residual CNN",
      "short_thesis": "Instead of building a much larger new backbone, insert small bottleneck adapters before and after ordinary residual blocks. This tests whether parameter-efficient adapters can improve the existing CNN family while leaving most of the architecture conventional.",
      "slug": "adapter_sandwich_residual_cnn",
      "source_packet_candidate": "Adapter-Sandwich Residual CNN",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md",
      "source_packet_rank": 6,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i155_capsule_motif_boardnet",
      "idea_id": "i155",
      "mechanism_family": "generic",
      "name": "Capsule Motif BoardNet",
      "short_thesis": "Local chess motifs are not only scalar activations; they have type, pose, orientation, and part-whole relationships. A capsule-style model can encode local patterns as small vectors and route them into higher-level tactical motif capsules by agreement.",
      "slug": "capsule_motif_boardnet",
      "source_packet_candidate": "Capsule Motif BoardNet",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i156_multi_order_board_scan_network",
      "idea_id": "i156",
      "mechanism_family": "generic",
      "name": "Multi-Order Board Scan Network",
      "short_thesis": "A chess board can be read as several short sequences. Different scan orders expose different dependencies: rank-major order, file-major order, diagonal order, spiral-from-king order, and center-out order. A shared sequence model over fixed board orders can...",
      "slug": "multi_order_board_scan_network",
      "source_packet_candidate": "Multi-Order Board Scan Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i157_cross_stitch_cnn_token_fusion_net",
      "idea_id": "i157",
      "mechanism_family": "generic",
      "name": "Cross-Stitch CNN-Token Fusion Net",
      "short_thesis": "Late fusion between a CNN branch and a piece-token branch may be too weak. A cross-stitch network can let the branches exchange information at multiple depths through learned linear mixing, while still keeping the model practical.",
      "slug": "cross_stitch_cnn_token_fusion_net",
      "source_packet_candidate": "Cross-Stitch CNN-Token Fusion Net",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i158_neural_decision_forest_boardnet",
      "idea_id": "i158",
      "mechanism_family": "logic",
      "name": "Neural Decision Forest BoardNet",
      "short_thesis": "Chess puzzle-likeness may be piecewise: different board regimes require different cues. A differentiable decision forest on top of a CNN feature vector can model soft oblique splits and leaf predictors without a sparse expert router.",
      "slug": "neural_decision_forest_boardnet",
      "source_packet_candidate": "Neural Decision Forest BoardNet",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i159_vector_quantized_motif_codebook_net",
      "idea_id": "i159",
      "mechanism_family": "sparse",
      "name": "Vector-Quantized Motif Codebook Net",
      "short_thesis": "Force local board features to pass through a learned discrete codebook. The classifier reads code usage, spatial code maps, and quantized features. This tests whether a compact inventory of board motifs is useful for puzzle-likeness.",
      "slug": "vector_quantized_motif_codebook_net",
      "source_packet_candidate": "Vector-Quantized Motif Codebook Net",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i160_hypercolumn_square_readout_cnn",
      "idea_id": "i160",
      "mechanism_family": "generic",
      "name": "Hypercolumn Square Readout CNN",
      "short_thesis": "Intermediate CNN layers may detect different chess cues: early local piece contacts, middle motifs, and later global context. A hypercolumn readout gathers per-square features from every depth and classifies from square-level evidence maps plus global pooling.",
      "slug": "hypercolumn_square_readout_cnn",
      "source_packet_candidate": "Hypercolumn Square Readout CNN",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md",
      "source_packet_rank": 6,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i161_multiplicative_conjunction_convnet",
      "idea_id": "i161",
      "mechanism_family": "generic",
      "name": "Multiplicative Conjunction ConvNet",
      "short_thesis": "Many chess motifs are conjunctions: attacker plus target plus blocker absence, king exposure plus line pressure, or material cue plus square pattern. A conv net with explicit multiplicative gates can represent local AND-like interactions more directly than...",
      "slug": "multiplicative_conjunction_convnet",
      "source_packet_candidate": "Multiplicative Conjunction ConvNet",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md",
      "source_packet_rank": 7,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i162_empty_square_opportunity_network",
      "idea_id": "i162",
      "mechanism_family": "generic",
      "name": "Empty-Square Opportunity Network",
      "short_thesis": "Chess tactics often depend on empty squares: escape squares, mating squares, promotion paths, discovered-attack landing squares, fork squares, and blocking/interference squares. A classifier that separately models occupied-square evidence and empty-square o...",
      "slug": "empty_square_opportunity_network",
      "source_packet_candidate": "Empty-Square Opportunity Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i163_global_scratchpad_boardnet",
      "idea_id": "i163",
      "mechanism_family": "generic",
      "name": "Global Scratchpad BoardNet",
      "short_thesis": "A board CNN can be augmented with a small recurrent global scratchpad: a fixed number of memory vectors that summarize the board, are updated a few times, and broadcast context back to squares through affine modulation. This gives global communication witho...",
      "slug": "global_scratchpad_boardnet",
      "source_packet_candidate": "Global Scratchpad BoardNet",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i164_learnable_pooling_tree_boardnet",
      "idea_id": "i164",
      "mechanism_family": "generic",
      "name": "Learnable Pooling Tree BoardNet",
      "short_thesis": "Instead of pooling the whole board at once or using an FPN, build a fixed hierarchy over the `8 x 8` board: squares become `2 x 2` cells, cells become quadrants, quadrants become a board root. Each tree node has a small learned aggregator and passes feature...",
      "slug": "learnable_pooling_tree_boardnet",
      "source_packet_candidate": "Learnable Pooling Tree BoardNet",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i165_spatial_film_coordinate_net",
      "idea_id": "i165",
      "mechanism_family": "generic",
      "name": "Spatial FiLM Coordinate Net",
      "short_thesis": "Appending coordinate planes may be too weak. Instead, generate per-square affine modulation parameters from deterministic coordinate features and side-relative coordinates, then modulate CNN features at multiple depths.",
      "slug": "spatial_film_coordinate_net",
      "source_packet_candidate": "Spatial FiLM Coordinate Net",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i166_channel_bilinear_role_mixer",
      "idea_id": "i166",
      "mechanism_family": "grammar",
      "name": "Channel-Bilinear Role Mixer",
      "short_thesis": "Ordinary heads pool channels additively. A low-rank bilinear head can explicitly model pairwise interactions between role summaries, such as own-heavy-piece features with opponent-king-zone features, without building square-pair tensors or local product con...",
      "slug": "channel_bilinear_role_mixer",
      "source_packet_candidate": "Channel-Bilinear Role Mixer",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i167_evidence_sieve_network",
      "idea_id": "i167",
      "mechanism_family": "information",
      "name": "Evidence Sieve Network",
      "short_thesis": "Instead of refining logits, the model can refine features by repeatedly filtering them through learned evidence sieves. Each sieve stage produces a soft mask over channels and squares, passes selected evidence onward, and leaves a diagnostic trail.",
      "slug": "evidence_sieve_network",
      "source_packet_candidate": "Evidence Sieve Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md",
      "source_packet_rank": 6,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i168_ring_shell_recurrent_boardnet",
      "idea_id": "i168",
      "mechanism_family": "generic",
      "name": "Ring-Shell Recurrent BoardNet",
      "short_thesis": "Important chess context often radiates from anchors: kings, center squares, edges, and promotion zones. Summarize the board in fixed rings/shells around these anchors and process the shells with a small recurrent model.",
      "slug": "ring_shell_recurrent_boardnet",
      "source_packet_candidate": "Ring-Shell Recurrent BoardNet",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md",
      "source_packet_rank": 7,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i169_rank_file_memory_grid_net",
      "idea_id": "i169",
      "mechanism_family": "generic",
      "name": "Rank-File Memory Grid Net",
      "short_thesis": "Maintain learned memory vectors for each rank and each file. Squares write into their rank/file memories, then rank/file memories write back to squares. This gives global rank/file communication without axial convolutions, line solves, or attention.",
      "slug": "rank_file_memory_grid_net",
      "source_packet_candidate": "Rank-File Memory Grid Net",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md",
      "source_packet_rank": 8,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i170_negative_class_disentangled_puzzle_head",
      "idea_id": "i170",
      "mechanism_family": "generic",
      "name": "Negative-Class Disentangled Puzzle Head",
      "short_thesis": "The target is binary, but the negative class has two very different sources:",
      "slug": "negative_class_disentangled_puzzle_head",
      "source_packet_candidate": "Negative-Class Disentangled Puzzle Head",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i171_line_piece_crossbar_network",
      "idea_id": "i171",
      "mechanism_family": "grammar",
      "name": "Line-Piece Crossbar Network",
      "short_thesis": "Schur-Ray is mathematically powerful but more complex. A simpler line-aware architecture can create line tokens and piece tokens, then pass messages only through deterministic piece-line incidence.",
      "slug": "line_piece_crossbar_network",
      "source_packet_candidate": "Line-Piece Crossbar Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i172_near_puzzle_margin_twin_network",
      "idea_id": "i172",
      "mechanism_family": "robustness",
      "name": "Near-Puzzle Margin Twin Network",
      "short_thesis": "The benchmark is fundamentally about ranking:",
      "slug": "near_puzzle_margin_twin_network",
      "source_packet_candidate": "Near-Puzzle Margin Twin Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i173_stripe_selective_mixer_cnn",
      "idea_id": "i173",
      "mechanism_family": "grammar",
      "name": "Stripe-Selective Mixer CNN",
      "short_thesis": "A practical line-aware CNN may be enough to beat the current BT4 while staying simpler than Schur-Ray. Instead of ordinary `3x3` convolutions only, mix along chess stripes:",
      "slug": "stripe_selective_mixer_cnn",
      "source_packet_candidate": "Stripe-Selective Mixer CNN",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i174_king_zone_evidence_ledger",
      "idea_id": "i174",
      "mechanism_family": "information",
      "name": "King-Zone Evidence Ledger",
      "short_thesis": "Many real puzzles are ultimately about king safety or forcing geometry. A model can maintain a small set of learned evidence ledger slots around each king:",
      "slug": "king_zone_evidence_ledger",
      "source_packet_candidate": "King-Zone Evidence Ledger",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i175_prototype_margin_puzzle_network",
      "idea_id": "i175",
      "mechanism_family": "sparse",
      "name": "Prototype-Margin Puzzle Network",
      "short_thesis": "The model should not merely say \"puzzle-like.\" It should compare the board to separate learned prototypes:",
      "slug": "prototype_margin_puzzle_network",
      "source_packet_candidate": "Prototype-Margin Puzzle Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md",
      "source_packet_rank": 6,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i176_source_rate_calibrated_objective",
      "idea_id": "i176",
      "mechanism_family": "robustness",
      "name": "Source-Rate Calibrated Objective",
      "short_thesis": "The current benchmark's central question is not only \"is F1 high?\" It is:",
      "slug": "source_rate_calibrated_objective",
      "source_packet_candidate": "Source-Rate Calibrated Objective",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md",
      "source_packet_rank": 7,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i177_forcing_certificate_transformer",
      "idea_id": "i177",
      "mechanism_family": "graph",
      "name": "Forcing-Certificate Transformer",
      "short_thesis": "A real puzzle should admit a compact tactical certificate:",
      "slug": "forcing_certificate_transformer",
      "source_packet_candidate": "Forcing-Certificate Transformer",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i178_defender_exhaustion_cascade_network",
      "idea_id": "i178",
      "mechanism_family": "generic",
      "name": "Defender-Exhaustion Cascade Network",
      "short_thesis": "Many puzzles exist because one side cannot satisfy all defensive obligations at once:",
      "slug": "defender_exhaustion_cascade_network",
      "source_packet_candidate": "Defender-Exhaustion Cascade Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i179_causal_piece_derivative_network",
      "idea_id": "i179",
      "mechanism_family": "generic",
      "name": "Causal Piece-Derivative Network",
      "short_thesis": "In true puzzles, the puzzle signal often depends sharply on a few critical pieces or squares. In near-puzzles, the score may come from broad tactical texture without a decisive dependency.",
      "slug": "causal_piece_derivative_network",
      "source_packet_candidate": "Causal Piece-Derivative Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i180_phase_transition_pressure_network",
      "idea_id": "i180",
      "mechanism_family": "generic",
      "name": "Phase-Transition Pressure Network",
      "short_thesis": "The key difference between a true puzzle and a near-puzzle may be criticality. The board may sit near a threshold where small increases in pressure, line opening, or defender loss cause a tactical collapse.",
      "slug": "phase_transition_pressure_network",
      "source_packet_candidate": "Phase-Transition Pressure Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i181_disproof_ledger_puzzle_network",
      "idea_id": "i181",
      "mechanism_family": "generic",
      "name": "Disproof-Ledger Puzzle Network",
      "short_thesis": "The model should not only collect evidence for \"puzzle.\" It should collect explicit disproof evidence:",
      "slug": "disproof_ledger_puzzle_network",
      "source_packet_candidate": "Disproof-Ledger Puzzle Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i182_motif_tensor_factorization_network",
      "idea_id": "i182",
      "mechanism_family": "generic",
      "name": "Motif Tensor Factorization Network",
      "short_thesis": "Puzzle signal is often a multiplicative relation among typed roles:",
      "slug": "motif_tensor_factorization_network",
      "source_packet_candidate": "Motif Tensor Factorization Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md",
      "source_packet_rank": 6,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i183_tempo_alignment_gate_network",
      "idea_id": "i183",
      "mechanism_family": "generic",
      "name": "Tempo-Alignment Gate Network",
      "short_thesis": "Many near-puzzles are tactical-looking for the wrong side or require a tempo that the side to move does not have. The model should explicitly gate static tactical danger by side-to-move tempo.",
      "slug": "tempo_alignment_gate_network",
      "source_packet_candidate": "Tempo-Alignment Gate Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md",
      "source_packet_rank": 7,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i184_puzzle_boundary_twin_encoder",
      "idea_id": "i184",
      "mechanism_family": "generic",
      "name": "Puzzle Boundary Twin Encoder",
      "short_thesis": "The hardest part is the boundary between verified puzzles and verified near-puzzles. Learn that boundary directly with a twin encoder and margin objective.",
      "slug": "puzzle_boundary_twin_encoder",
      "source_packet_candidate": "Puzzle Boundary Twin Encoder",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md",
      "source_packet_rank": 8,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i185_critical_square_budget_network",
      "idea_id": "i185",
      "mechanism_family": "generic",
      "name": "Critical-Square Budget Network",
      "short_thesis": "Puzzles often hinge on a small number of critical squares: king escape squares, line intersections, pinned-piece squares, promotion squares, or overloaded defender squares.",
      "slug": "critical_square_budget_network",
      "source_packet_candidate": "Critical-Square Budget Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md",
      "source_packet_rank": 9,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i186_legal_reaction_bottleneck_network",
      "idea_id": "i186",
      "mechanism_family": "generic",
      "name": "Legal-Reaction Bottleneck Network",
      "short_thesis": "A real puzzle is not merely a position with a threat. It is a position where normal-looking defensive reactions fail or are too few. Near-puzzles often contain pressure, but the opponent still has many valid ways to defuse it.",
      "slug": "legal_reaction_bottleneck_network",
      "source_packet_candidate": "Legal-Reaction Bottleneck Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i187_exchange_soundness_graph_network",
      "idea_id": "i187",
      "mechanism_family": "generic",
      "name": "Exchange-Soundness Graph Network",
      "short_thesis": "Many false puzzle signals come from attacks that look strong but lose material or fail tactically after exchanges. A puzzle detector should know whether an apparent tactic is exchange-sound.",
      "slug": "exchange_soundness_graph_network",
      "source_packet_candidate": "Exchange-Soundness Graph Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i188_tactical_program_induction_network",
      "idea_id": "i188",
      "mechanism_family": "generic",
      "name": "Tactical Program Induction Network",
      "short_thesis": "A puzzle can be viewed as a tiny latent program:",
      "slug": "tactical_program_induction_network",
      "source_packet_candidate": "Tactical Program Induction Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i189_counterfactual_defender_dropout_network",
      "idea_id": "i189",
      "mechanism_family": "robustness",
      "name": "Counterfactual Defender Dropout Network",
      "short_thesis": "If a near-puzzle is only superficially tactical, randomly removing defenders or attackers may not reveal a sharp causal structure. If a true puzzle hinges on overloaded defenders, pinning, or one critical escape square, dropout interventions should produce...",
      "slug": "counterfactual_defender_dropout_network",
      "source_packet_candidate": "Counterfactual Defender Dropout Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i190_blocker_pin_lattice_network",
      "idea_id": "i190",
      "mechanism_family": "logic",
      "name": "Blocker-Pin Lattice Network",
      "short_thesis": "Line tactics are not only about pieces sharing ranks, files, or diagonals. They depend on ordered blockers and pin constraints. A line can be almost tactical, but one blocker order or one unpinned defender changes everything.",
      "slug": "blocker_pin_lattice_network",
      "source_packet_candidate": "Blocker-Pin Lattice Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i191_safe_reply_certificate_verifier",
      "idea_id": "i191",
      "mechanism_family": "generic",
      "name": "Safe-Reply Certificate Verifier",
      "short_thesis": "Instead of proving that a position is a puzzle, try to prove that it is not a puzzle. If the model can find a cheap safe-reply certificate, the puzzle logit should go down.",
      "slug": "safe_reply_certificate_verifier",
      "source_packet_candidate": "Safe-Reply Certificate Verifier",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md",
      "source_packet_rank": 6,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i192_latent_reply_entropy_network",
      "idea_id": "i192",
      "mechanism_family": "generic",
      "name": "Latent Reply Entropy Network",
      "short_thesis": "A forcing puzzle often reduces the opponent's viable reply distribution. A near-puzzle may have many replies that keep the position acceptable. The network can learn a reply entropy proxy without engine labels.",
      "slug": "latent_reply_entropy_network",
      "source_packet_candidate": "Latent Reply Entropy Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md",
      "source_packet_rank": 7,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i193_exchange_then_king_dual_stream",
      "idea_id": "i193",
      "mechanism_family": "generic",
      "name": "Exchange-Then-King Dual Stream",
      "short_thesis": "Puzzle data likely mixes at least two broad families:",
      "slug": "exchange_then_king_dual_stream",
      "source_packet_candidate": "Exchange-Then-King Dual Stream",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md",
      "source_packet_rank": 8,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i194_tactical_symptom_bayesian_network",
      "idea_id": "i194",
      "mechanism_family": "generic",
      "name": "Tactical Symptom Bayesian Network",
      "short_thesis": "Many tactical concepts behave like noisy logical symptoms:",
      "slug": "tactical_symptom_bayesian_network",
      "source_packet_candidate": "Tactical Symptom Bayesian Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md",
      "source_packet_rank": 9,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i195_minimal_edit_puzzle_distance_network",
      "idea_id": "i195",
      "mechanism_family": "generic",
      "name": "Minimal-Edit Puzzle Distance Network",
      "short_thesis": "A near-puzzle may be one small edit away from being a true puzzle:",
      "slug": "minimal_edit_puzzle_distance_network",
      "source_packet_candidate": "Minimal-Edit Puzzle Distance Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md",
      "source_packet_rank": 10,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i196_source_invariant_puzzle_bottleneck",
      "idea_id": "i196",
      "mechanism_family": "generic",
      "name": "Source-Invariant Puzzle Bottleneck",
      "short_thesis": "The dataset has three source groups. A model may accidentally learn source artifacts instead of puzzle structure. This architecture tries to preserve puzzle signal while removing source identity from the main representation.",
      "slug": "source_invariant_puzzle_bottleneck",
      "source_packet_candidate": "Source-Invariant Puzzle Bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md",
      "source_packet_rank": 11,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i197_reply_set_contrastive_transformer",
      "idea_id": "i197",
      "mechanism_family": "graph",
      "name": "Reply-Set Contrastive Transformer",
      "short_thesis": "A puzzle position should embed differently from its plausible reply positions. A near-puzzle may remain close to one or more safe replies. Use contrastive learning over current position and pseudo-reply positions.",
      "slug": "reply_set_contrastive_transformer",
      "source_packet_candidate": "Reply-Set Contrastive Transformer",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md",
      "source_packet_rank": 12,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i198_barrier_cut_puzzle_network",
      "idea_id": "i198",
      "mechanism_family": "generic",
      "name": "Barrier-Cut Puzzle Network",
      "short_thesis": "A true puzzle often exists because the defender cannot maintain a barrier between attacking force and a valuable target: king, queen, promotion square, pinned defender, or mating square. A near-puzzle may contain pressure, but there is still a strong defens...",
      "slug": "barrier_cut_puzzle_network",
      "source_packet_candidate": "Barrier-Cut Puzzle Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0109_saturday_shanghai_high_upside_puzzle_batch_4.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:01+00:00",
      "folder": "ideas/i199_tactical_hessian_spectrum_network",
      "idea_id": "i199",
      "mechanism_family": "linear_algebra",
      "name": "Tactical Hessian Spectrum Network",
      "short_thesis": "A real puzzle may be a sharp local maximum of tactical evidence under legal perturbations. Near-puzzles may have high raw evidence but flatter or less stable local geometry.",
      "slug": "tactical_hessian_spectrum_network",
      "source_packet_candidate": "Tactical Hessian Spectrum Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0109_saturday_shanghai_high_upside_puzzle_batch_4.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:02+00:00",
      "folder": "ideas/i200_absorbing_threat_markov_network",
      "idea_id": "i200",
      "mechanism_family": "generic",
      "name": "Absorbing Threat Markov Network",
      "short_thesis": "Puzzle detection can be treated as a probabilistic process over tactical states:",
      "slug": "absorbing_threat_markov_network",
      "source_packet_candidate": "Absorbing Threat Markov Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0109_saturday_shanghai_high_upside_puzzle_batch_4.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:02+00:00",
      "folder": "ideas/i201_neural_clause_resolution_puzzle_network",
      "idea_id": "i201",
      "mechanism_family": "logic",
      "name": "Neural Clause-Resolution Puzzle Network",
      "short_thesis": "A puzzle often follows from a small proof made of typed facts:",
      "slug": "neural_clause_resolution_puzzle_network",
      "source_packet_candidate": "Neural Clause-Resolution Puzzle Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0109_saturday_shanghai_high_upside_puzzle_batch_4.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:02+00:00",
      "folder": "ideas/i202_piece_liability_gradient_network",
      "idea_id": "i202",
      "mechanism_family": "generic",
      "name": "Piece Liability Gradient Network",
      "short_thesis": "In many puzzles, one piece is not merely attacked; it is liable. It cannot move, defend, capture, or stay without losing something. Near-puzzles may attack pieces, but the liability does not propagate.",
      "slug": "piece_liability_gradient_network",
      "source_packet_candidate": "Piece Liability Gradient Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0109_saturday_shanghai_high_upside_puzzle_batch_4.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:02+00:00",
      "folder": "ideas/i203_hierarchical_tactical_option_network",
      "idea_id": "i203",
      "mechanism_family": "generic",
      "name": "Hierarchical Tactical Option Network",
      "short_thesis": "Chess tactics are not just moves; they are options:",
      "slug": "hierarchical_tactical_option_network",
      "source_packet_candidate": "Hierarchical Tactical Option Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0109_saturday_shanghai_high_upside_puzzle_batch_4.md",
      "source_packet_rank": 6,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:02+00:00",
      "folder": "ideas/i204_cross_defense_consistency_network",
      "idea_id": "i204",
      "mechanism_family": "generic",
      "name": "Cross-Defense Consistency Network",
      "short_thesis": "A true puzzle should survive multiple independent defensive interpretations:",
      "slug": "cross_defense_consistency_network",
      "source_packet_candidate": "Cross-Defense Consistency Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-25_0109_saturday_shanghai_high_upside_puzzle_batch_4.md",
      "source_packet_rank": 7,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:02+00:00",
      "folder": "ideas/i205_defender_timing_schedule_network",
      "idea_id": "i205",
      "mechanism_family": "generic",
      "name": "Defender Timing Schedule Network",
      "short_thesis": "A true puzzle is not only a position where the defender has too few resources. It is often a position where the defender cannot schedule resources before tactical deadlines expire.",
      "slug": "defender_timing_schedule_network",
      "source_packet_candidate": "Defender Timing Schedule Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_1859_tuesday_shanghai_puzzle_architecture_batch_5.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:02+00:00",
      "folder": "ideas/i206_discovered_ray_switchboard_network",
      "idea_id": "i206",
      "mechanism_family": "grammar",
      "name": "Discovered-Ray Switchboard Network",
      "short_thesis": "Many tactics are not visible in the current attack map because the critical line appears only after a blocker moves. A discovered attack, skewer, deflection, or back-rank tactic can be modeled as a switchboard:",
      "slug": "discovered_ray_switchboard_network",
      "source_packet_candidate": "Discovered-Ray Switchboard Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_1859_tuesday_shanghai_puzzle_architecture_batch_5.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:02+00:00",
      "folder": "ideas/i207_counterplay_insolvency_ledger",
      "idea_id": "i207",
      "mechanism_family": "generic",
      "name": "Counterplay Insolvency Ledger",
      "short_thesis": "Near-puzzles often fail because the defender has counterplay. A model that only measures side-to-move pressure may overcall these. Puzzlehood should depend on whether the opponent's counterthreats remain solvent after the side-to-move begins forcing play.",
      "slug": "counterplay_insolvency_ledger",
      "source_packet_candidate": "Counterplay Insolvency Ledger",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_1859_tuesday_shanghai_puzzle_architecture_batch_5.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:02+00:00",
      "folder": "ideas/i208_pinned_mobility_nullspace_network",
      "idea_id": "i208",
      "mechanism_family": "generic",
      "name": "Pinned Mobility Nullspace Network",
      "short_thesis": "Many near-puzzles contain apparent defenders that are actually mobile. Many true puzzles contain apparent defenders whose legal or pseudo-legal mobility lies in a nullspace because moving them exposes a king, queen, mate square, or promotion stop.",
      "slug": "pinned_mobility_nullspace_network",
      "source_packet_candidate": "Pinned Mobility Nullspace Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_1859_tuesday_shanghai_puzzle_architecture_batch_5.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:02+00:00",
      "folder": "ideas/i209_tactical_effective_resistance_network",
      "idea_id": "i209",
      "mechanism_family": "generic",
      "name": "Tactical Effective Resistance Network",
      "short_thesis": "A cut measures the weakest separation between attacker and target. Effective resistance measures how many redundant routes exist and how well defenders can dissipate pressure across the whole tactical graph.",
      "slug": "tactical_effective_resistance_network",
      "source_packet_candidate": "Tactical Effective Resistance Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_1859_tuesday_shanghai_puzzle_architecture_batch_5.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:02+00:00",
      "folder": "ideas/i210_defender_opportunity_cost_auction_network",
      "idea_id": "i210",
      "mechanism_family": "generic",
      "name": "Defender Opportunity-Cost Auction Network",
      "short_thesis": "A defender can often answer one threat only by abandoning another duty. Static coverage says a resource exists. Opportunity-cost pricing asks what the resource gives up when assigned.",
      "slug": "defender_opportunity_cost_auction_network",
      "source_packet_candidate": "Defender Opportunity-Cost Auction Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_1859_tuesday_shanghai_puzzle_architecture_batch_5.md",
      "source_packet_rank": 6,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:02+00:00",
      "folder": "ideas/i211_role_counterfactual_necessity_network",
      "idea_id": "i211",
      "mechanism_family": "move_delta",
      "name": "Role-Counterfactual Necessity Network",
      "short_thesis": "Some false positives come from shortcut features: material imbalance, king exposure, or generic pressure. A real tactic should depend on exact role geometry. If safe role-preserving counterfactuals destroy the evidence, the puzzle signal is more credible.",
      "slug": "role_counterfactual_necessity_network",
      "source_packet_candidate": "Role-Counterfactual Necessity Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_1859_tuesday_shanghai_puzzle_architecture_batch_5.md",
      "source_packet_rank": 7,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:02+00:00",
      "folder": "ideas/i212_phase_specialist_calibration_mixture",
      "idea_id": "i212",
      "mechanism_family": "information",
      "name": "Phase-Specialist Calibration Mixture",
      "short_thesis": "The boundary between near-puzzle and true puzzle differs across opening tactics, mating attacks, material tactics, promotion races, and simplified endings. A single global head may overcall near-puzzles in one phase to preserve recall in another.",
      "slug": "phase_specialist_calibration_mixture",
      "source_packet_candidate": "Phase-Specialist Calibration Mixture",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_1859_tuesday_shanghai_puzzle_architecture_batch_5.md",
      "source_packet_rank": 8,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:02+00:00",
      "folder": "ideas/i213_forced_target_funnel_network",
      "idea_id": "i213",
      "mechanism_family": "generic",
      "name": "Forced-Target Funnel Network",
      "short_thesis": "A true puzzle often funnels different tactical symptoms toward the same target: king, queen, pinned defender, promotion square, or overloaded defender. A near-puzzle may have many threats, but they point in different directions.",
      "slug": "forced_target_funnel_network",
      "source_packet_candidate": "Forced-Target Funnel Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_1859_tuesday_shanghai_puzzle_architecture_batch_5.md",
      "source_packet_rank": 9,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:02+00:00",
      "folder": "ideas/i214_tactical_subgoal_automaton_network",
      "idea_id": "i214",
      "mechanism_family": "grammar",
      "name": "Tactical Subgoal Automaton Network",
      "short_thesis": "Many puzzles are short scripts:",
      "slug": "tactical_subgoal_automaton_network",
      "source_packet_candidate": "Tactical Subgoal Automaton Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-28_1859_tuesday_shanghai_puzzle_architecture_batch_5.md",
      "source_packet_rank": 10,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:29+00:00",
      "folder": "ideas/i215_masked_codec_interaction_curvature_network",
      "idea_id": "i215",
      "mechanism_family": "information",
      "name": "Masked Codec Interaction-Curvature Network",
      "short_thesis": "- Parent family: `Masked Board Code-Length Surprise Network`",
      "slug": "masked_codec_interaction_curvature_network",
      "source_packet_candidate": "Masked Codec Interaction-Curvature Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2051_friday_shanghai_researched_family_variants.md",
      "source_packet_rank": 1,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:29+00:00",
      "folder": "ideas/i216_non_puzzle_score_curl_divergence_bottleneck",
      "idea_id": "i216",
      "mechanism_family": "generic",
      "name": "Non-Puzzle Score Curl-Divergence Bottleneck",
      "short_thesis": "- Parent family: `Non-Puzzle Score-Field Bottleneck Network`",
      "slug": "non_puzzle_score_curl_divergence_bottleneck",
      "source_packet_candidate": "Non-Puzzle Score Curl-Divergence Bottleneck",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2051_friday_shanghai_researched_family_variants.md",
      "source_packet_rank": 2,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:29+00:00",
      "folder": "ideas/i217_ray_grammar_edit_distance_network",
      "idea_id": "i217",
      "mechanism_family": "grammar",
      "name": "Ray Grammar Edit-Distance Network",
      "short_thesis": "- Parent family: `Ray-Language Automaton Network`",
      "slug": "ray_grammar_edit_distance_network",
      "source_packet_candidate": "Ray Grammar Edit-Distance Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2051_friday_shanghai_researched_family_variants.md",
      "source_packet_rank": 3,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:29+00:00",
      "folder": "ideas/i218_orbit_disagreement_residual_network",
      "idea_id": "i218",
      "mechanism_family": "symmetry",
      "name": "Orbit Disagreement Residual Network",
      "short_thesis": "- Parent families: `Legal Automorphism Quotient Network`, `Rule-Exact Orbit Bottleneck Network`, and `Color-Flip Orbit Evidence Bottleneck`",
      "slug": "orbit_disagreement_residual_network",
      "source_packet_candidate": "Orbit Disagreement Residual Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2051_friday_shanghai_researched_family_variants.md",
      "source_packet_rank": 4,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:29+00:00",
      "folder": "ideas/i219_hall_defect_dual_residual_network",
      "idea_id": "i219",
      "mechanism_family": "generic",
      "name": "Hall-Defect Dual-Residual Network",
      "short_thesis": "- Parent family: `Hall-Defect Obligation Matroid Network`",
      "slug": "hall_defect_dual_residual_network",
      "source_packet_candidate": "Hall-Defect Dual-Residual Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2051_friday_shanghai_researched_family_variants.md",
      "source_packet_rank": 5,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    },
    {
      "created_at": "2026-04-30T15:50:29+00:00",
      "folder": "ideas/i220_credal_temperature_field_network",
      "idea_id": "i220",
      "mechanism_family": "generic",
      "name": "Credal Temperature Field Network",
      "short_thesis": "- Parent family: `Credal Near-Puzzle Evidence Network`",
      "slug": "credal_temperature_field_network",
      "source_packet_candidate": "Credal Temperature Field Network",
      "source_packet_path": "ideas/research_packets/chess_nn_research_2026-04-24_2051_friday_shanghai_researched_family_variants.md",
      "source_packet_rank": 6,
      "source_packet_status": "batch packet",
      "status": "implemented",
      "target_task": "puzzle_binary"
    }
  ],
  "max_runs_included": 25,
  "results_dir": "results",
  "runs": [
    {
      "artifacts": {
        "checkpoint_best": "results/20260429_033755_idea_i002_response_minimax_simple18/checkpoint_best.pt",
        "checkpoint_last": "results/20260429_033755_idea_i002_response_minimax_simple18/checkpoint_last.pt",
        "metrics_final": "results/20260429_033755_idea_i002_response_minimax_simple18/metrics_final.json",
        "report_html": "results/20260429_033755_idea_i002_response_minimax_simple18/report.html",
        "run_metadata": "results/20260429_033755_idea_i002_response_minimax_simple18/run_metadata.json",
        "run_summary": "results/20260429_033755_idea_i002_response_minimax_simple18/run_summary.md"
      },
      "best_epoch": 19,
      "class_counts": {
        "test": {
          "0": 30000,
          "1": 15000
        },
        "train": {
          "0": 240000,
          "1": 120000
        },
        "val": {
          "0": 30000,
          "1": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class_unique_crtk_tags",
      "device": "cuda",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "puzzle_binary",
      "model_name": "response_minimax_classifier",
      "notes": "Implemented GPU-required benchmark config for the response-minimax architecture.",
      "num_params": 231844,
      "run_dir": "results/20260429_033755_idea_i002_response_minimax_simple18",
      "run_name": "20260429_033755_idea_i002_response_minimax_simple18",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet",
        "train": "data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet",
        "val": "data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet"
      },
      "test": {
        "accuracy": 0.8403777777777778,
        "f1": 0.7823267371738537,
        "loss": 0.48376765589851306,
        "pr_auc": 0.8433727042923765,
        "precision": 0.7171509528307128,
        "recall": 0.8605333333333334,
        "roc_auc": 0.9211059
      },
      "timestamp": "2026-04-29T05:25:05Z",
      "validation": {
        "accuracy": 0.8405555555555555,
        "f1": 0.7825691687626898,
        "loss": 0.47793535786524,
        "pr_auc": 0.8481739733727617,
        "precision": 0.7173731873993,
        "recall": 0.8608,
        "roc_auc": 0.9226766344444445
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18/checkpoint_best.pt",
        "checkpoint_last": "results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18/checkpoint_last.pt",
        "metrics_final": "results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18/metrics_final.json",
        "report_html": "results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18/report.html",
        "run_metadata": "results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18/run_metadata.json",
        "run_summary": "results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18/run_summary.md"
      },
      "best_epoch": 20,
      "class_counts": {
        "test": {
          "0": 30000,
          "1": 15000
        },
        "train": {
          "0": 240000,
          "1": 120000
        },
        "val": {
          "0": 30000,
          "1": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class_unique_crtk_tags",
      "device": "cuda",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "puzzle_binary",
      "model_name": "boundary_edit_lagrangian_network",
      "notes": "Implemented GPU-required benchmark config for the boundary-edit Lagrangian architecture.",
      "num_params": 173764,
      "run_dir": "results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18",
      "run_name": "20260429_030542_idea_i008_boundary_edit_lagrangian_simple18",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet",
        "train": "data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet",
        "val": "data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet"
      },
      "test": {
        "accuracy": 0.8369333333333333,
        "f1": 0.7768519644812066,
        "loss": 0.4886580416932702,
        "pr_auc": 0.8375124689605993,
        "precision": 0.7142138224110938,
        "recall": 0.8515333333333334,
        "roc_auc": 0.9182358966666668
      },
      "timestamp": "2026-04-29T03:30:00Z",
      "validation": {
        "accuracy": 0.8378444444444444,
        "f1": 0.7788050562308648,
        "loss": 0.4812585483728485,
        "pr_auc": 0.8430021914769968,
        "precision": 0.714103062982934,
        "recall": 0.8564,
        "roc_auc": 0.9203616311111111
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260429_023704_idea_i005_null_move_contrast_simple18/checkpoint_best.pt",
        "checkpoint_last": "results/20260429_023704_idea_i005_null_move_contrast_simple18/checkpoint_last.pt",
        "metrics_final": "results/20260429_023704_idea_i005_null_move_contrast_simple18/metrics_final.json",
        "report_html": "results/20260429_023704_idea_i005_null_move_contrast_simple18/report.html",
        "run_metadata": "results/20260429_023704_idea_i005_null_move_contrast_simple18/run_metadata.json",
        "run_summary": "results/20260429_023704_idea_i005_null_move_contrast_simple18/run_summary.md"
      },
      "best_epoch": 18,
      "class_counts": {
        "test": {
          "0": 30000,
          "1": 15000
        },
        "train": {
          "0": 240000,
          "1": 120000
        },
        "val": {
          "0": 30000,
          "1": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class_unique_crtk_tags",
      "device": "cuda",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "puzzle_binary",
      "model_name": "null_move_contrast_puzzle_network",
      "notes": "Implemented GPU-required benchmark config for the null-move contrast architecture.",
      "num_params": 240578,
      "run_dir": "results/20260429_023704_idea_i005_null_move_contrast_simple18",
      "run_name": "20260429_023704_idea_i005_null_move_contrast_simple18",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet",
        "train": "data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet",
        "val": "data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet"
      },
      "test": {
        "accuracy": 0.8478888888888889,
        "f1": 0.7912536976609436,
        "loss": 0.47310742050070653,
        "pr_auc": 0.8538221663249441,
        "precision": 0.7291889157439154,
        "recall": 0.8648666666666667,
        "roc_auc": 0.9274141211111111
      },
      "timestamp": "2026-04-29T02:59:20Z",
      "validation": {
        "accuracy": 0.8492666666666666,
        "f1": 0.7932327389117513,
        "loss": 0.46844687685370445,
        "pr_auc": 0.8595689121188905,
        "precision": 0.7307497893850042,
        "recall": 0.8674,
        "roc_auc": 0.9292866655555556
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4/checkpoint_best.pt",
        "checkpoint_last": "results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4/checkpoint_last.pt",
        "metrics_final": "results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4/metrics_final.json",
        "report_html": "results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4/report.html",
        "run_metadata": "results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4/run_metadata.json",
        "run_summary": "results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4/run_summary.md"
      },
      "best_epoch": 20,
      "class_counts": {
        "test": {
          "0": 30000,
          "1": 15000
        },
        "train": {
          "0": 240000,
          "1": 120000
        },
        "val": {
          "0": 30000,
          "1": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class_unique_crtk_tags",
      "device": "cuda",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "puzzle_binary",
      "model_name": "sparse_relation_pursuit_asymmetry",
      "notes": "Sparse Relation Pursuit Asymmetry v1: deterministic chess relation tokens, equal-capacity background/tactical sparse dictionaries, LISTA-style group pursuit, no dense classifier bypass, and paper-grade CUDA training defaults.",
      "num_params": 121115,
      "run_dir": "results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4",
      "run_name": "20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet",
        "train": "data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet",
        "val": "data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet"
      },
      "test": {
        "accuracy": 0.8608222222222223,
        "f1": 0.8073456581254422,
        "loss": 0.5438584427730265,
        "pr_auc": 0.8687288947146012,
        "precision": 0.7495002570106802,
        "recall": 0.8748666666666667,
        "roc_auc": 0.936883571111111
      },
      "timestamp": "2026-04-29T02:28:14Z",
      "validation": {
        "accuracy": 0.8623111111111111,
        "f1": 0.8095764951748724,
        "loss": 0.5369015578815545,
        "pr_auc": 0.8748444507782331,
        "precision": 0.7509978332763143,
        "recall": 0.8780666666666667,
        "roc_auc": 0.938883978888889
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260428_183322_idea_i007_neural_proof_number_simple18/checkpoint_best.pt",
        "checkpoint_last": "results/20260428_183322_idea_i007_neural_proof_number_simple18/checkpoint_last.pt",
        "metrics_final": "results/20260428_183322_idea_i007_neural_proof_number_simple18/metrics_final.json",
        "report_html": "results/20260428_183322_idea_i007_neural_proof_number_simple18/report.html",
        "run_metadata": "results/20260428_183322_idea_i007_neural_proof_number_simple18/run_metadata.json",
        "run_summary": "results/20260428_183322_idea_i007_neural_proof_number_simple18/run_summary.md"
      },
      "best_epoch": 20,
      "class_counts": {
        "test": {
          "0": 30000,
          "1": 15000
        },
        "train": {
          "0": 240000,
          "1": 120000
        },
        "val": {
          "0": 30000,
          "1": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class_unique_crtk_tags",
      "device": "cuda",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "puzzle_binary",
      "model_name": "neural_proof_number_search",
      "notes": "Implemented GPU-required benchmark config for the neural proof-number architecture.",
      "num_params": 275909,
      "run_dir": "results/20260428_183322_idea_i007_neural_proof_number_simple18",
      "run_name": "20260428_183322_idea_i007_neural_proof_number_simple18",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet",
        "train": "data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet",
        "val": "data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet"
      },
      "test": {
        "accuracy": 0.8453555555555555,
        "f1": 0.7810602485449111,
        "loss": 0.49712595502450774,
        "pr_auc": 0.839876990912235,
        "precision": 0.7395293416741138,
        "recall": 0.8275333333333333,
        "roc_auc": 0.9199110277777779
      },
      "timestamp": "2026-04-28T19:16:12Z",
      "validation": {
        "accuracy": 0.8466444444444444,
        "f1": 0.7831306370007228,
        "loss": 0.48813378324767925,
        "pr_auc": 0.8457770631622175,
        "precision": 0.7407407407407407,
        "recall": 0.8306666666666667,
        "roc_auc": 0.9224409555555557
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260428_180243_idea_i009_tactical_equilibrium_simple18/checkpoint_best.pt",
        "checkpoint_last": "results/20260428_180243_idea_i009_tactical_equilibrium_simple18/checkpoint_last.pt",
        "metrics_final": "results/20260428_180243_idea_i009_tactical_equilibrium_simple18/metrics_final.json",
        "report_html": "results/20260428_180243_idea_i009_tactical_equilibrium_simple18/report.html",
        "run_metadata": "results/20260428_180243_idea_i009_tactical_equilibrium_simple18/run_metadata.json",
        "run_summary": "results/20260428_180243_idea_i009_tactical_equilibrium_simple18/run_summary.md"
      },
      "best_epoch": 18,
      "class_counts": {
        "test": {
          "0": 30000,
          "1": 15000
        },
        "train": {
          "0": 240000,
          "1": 120000
        },
        "val": {
          "0": 30000,
          "1": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class_unique_crtk_tags",
      "device": "cuda",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "puzzle_binary",
      "model_name": "tactical_equilibrium_network",
      "notes": "Implemented GPU-required benchmark config for the tactical-equilibrium architecture.",
      "num_params": 176676,
      "run_dir": "results/20260428_180243_idea_i009_tactical_equilibrium_simple18",
      "run_name": "20260428_180243_idea_i009_tactical_equilibrium_simple18",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet",
        "train": "data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet",
        "val": "data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet"
      },
      "test": {
        "accuracy": 0.8455555555555555,
        "f1": 0.7854010992404126,
        "loss": 0.4836489659818736,
        "pr_auc": 0.8468951273506644,
        "precision": 0.7315081099735419,
        "recall": 0.8478666666666667,
        "roc_auc": 0.9233379388888888
      },
      "timestamp": "2026-04-28T18:28:04Z",
      "validation": {
        "accuracy": 0.8485555555555555,
        "f1": 0.7893028288761788,
        "loss": 0.4749451650475914,
        "pr_auc": 0.8539998762095637,
        "precision": 0.7359469587777457,
        "recall": 0.851,
        "roc_auc": 0.9258365500000001
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260428_164249_idea_i012_dykstra_vetoselect_lc0bt4_v2/checkpoint_best.pt",
        "checkpoint_last": "results/20260428_164249_idea_i012_dykstra_vetoselect_lc0bt4_v2/checkpoint_last.pt",
        "metrics_final": "results/20260428_164249_idea_i012_dykstra_vetoselect_lc0bt4_v2/metrics_final.json",
        "report_html": "results/20260428_164249_idea_i012_dykstra_vetoselect_lc0bt4_v2/report.html",
        "run_metadata": "results/20260428_164249_idea_i012_dykstra_vetoselect_lc0bt4_v2/run_metadata.json",
        "run_summary": "results/20260428_164249_idea_i012_dykstra_vetoselect_lc0bt4_v2/run_summary.md"
      },
      "best_epoch": 3,
      "class_counts": {
        "test": {
          "0": 30000,
          "1": 15000
        },
        "train": {
          "0": 240000,
          "1": 120000
        },
        "val": {
          "0": 30000,
          "1": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class_unique_crtk_tags",
      "device": "cuda",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "puzzle_binary",
      "model_name": "dykstra_vetoselect",
      "notes": "Dykstra-LCP v2 hybrid: projection diagnostics feed a VetoSelect positive-claim accept/reject head with projection-weighted decoy mining.",
      "num_params": 531584,
      "run_dir": "results/20260428_164249_idea_i012_dykstra_vetoselect_lc0bt4_v2",
      "run_name": "20260428_164249_idea_i012_dykstra_vetoselect_lc0bt4_v2",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet",
        "train": "data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet",
        "val": "data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet"
      },
      "test": {
        "accuracy": 0.8334666666666667,
        "f1": 0.779562301447229,
        "loss": 0.649112344422239,
        "pr_auc": 0.8474298736067405,
        "precision": 0.6975679090334808,
        "recall": 0.8834,
        "roc_auc": 0.923047188888889
      },
      "timestamp": "2026-04-28T16:54:21Z",
      "validation": {
        "accuracy": 0.8344444444444444,
        "f1": 0.7810626542847067,
        "loss": 0.640758460252843,
        "pr_auc": 0.8540147525919779,
        "precision": 0.6983918435989068,
        "recall": 0.8859333333333334,
        "roc_auc": 0.9251379244444444
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260428_162513_idea_i012_dykstra_lcp_lc0bt4/checkpoint_best.pt",
        "checkpoint_last": "results/20260428_162513_idea_i012_dykstra_lcp_lc0bt4/checkpoint_last.pt",
        "metrics_final": "results/20260428_162513_idea_i012_dykstra_lcp_lc0bt4/metrics_final.json",
        "report_html": "results/20260428_162513_idea_i012_dykstra_lcp_lc0bt4/report.html",
        "run_metadata": "results/20260428_162513_idea_i012_dykstra_lcp_lc0bt4/run_metadata.json",
        "run_summary": "results/20260428_162513_idea_i012_dykstra_lcp_lc0bt4/run_summary.md"
      },
      "best_epoch": 2,
      "class_counts": {
        "test": {
          "0": 30000,
          "1": 15000
        },
        "train": {
          "0": 240000,
          "1": 120000
        },
        "val": {
          "0": 30000,
          "1": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class_unique_crtk_tags",
      "device": "cuda",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "puzzle_binary",
      "model_name": "dykstra_lcp",
      "notes": "Soft-Dykstra LCP A0: compact LC0 BT4 trunk with 4-cycle latent constraint projection and binary-only hard-negative emphasis.",
      "num_params": 338963,
      "run_dir": "results/20260428_162513_idea_i012_dykstra_lcp_lc0bt4",
      "run_name": "20260428_162513_idea_i012_dykstra_lcp_lc0bt4",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet",
        "train": "data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet",
        "val": "data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet"
      },
      "test": {
        "accuracy": 0.8189777777777778,
        "f1": 0.7632389699471023,
        "loss": 0.5947706817903302,
        "pr_auc": 0.8330579196230986,
        "precision": 0.6765948675667319,
        "recall": 0.8753333333333333,
        "roc_auc": 0.9128822055555555
      },
      "timestamp": "2026-04-28T16:33:54Z",
      "validation": {
        "accuracy": 0.8195111111111111,
        "f1": 0.7641010746442056,
        "loss": 0.5845980584960092,
        "pr_auc": 0.8420414302830956,
        "precision": 0.6769943386515698,
        "recall": 0.8769333333333333,
        "roc_auc": 0.916433951111111
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260428_160748_idea_i011_vetoselect_lc0bt4_v2_texture/checkpoint_best.pt",
        "checkpoint_last": "results/20260428_160748_idea_i011_vetoselect_lc0bt4_v2_texture/checkpoint_last.pt",
        "metrics_final": "results/20260428_160748_idea_i011_vetoselect_lc0bt4_v2_texture/metrics_final.json",
        "report_html": "results/20260428_160748_idea_i011_vetoselect_lc0bt4_v2_texture/report.html",
        "run_metadata": "results/20260428_160748_idea_i011_vetoselect_lc0bt4_v2_texture/run_metadata.json",
        "run_summary": "results/20260428_160748_idea_i011_vetoselect_lc0bt4_v2_texture/run_summary.md"
      },
      "best_epoch": 3,
      "class_counts": {
        "test": {
          "0": 30000,
          "1": 15000
        },
        "train": {
          "0": 240000,
          "1": 120000
        },
        "val": {
          "0": 30000,
          "1": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class_unique_crtk_tags",
      "device": "cuda",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "puzzle_binary",
      "model_name": "vetoselect_positive_claim_abstention",
      "notes": "VetoSelect v2/A3: board-only model with deterministic rule-texture-weighted self-mined decoy negatives after warmup.",
      "num_params": 501602,
      "run_dir": "results/20260428_160748_idea_i011_vetoselect_lc0bt4_v2_texture",
      "run_name": "20260428_160748_idea_i011_vetoselect_lc0bt4_v2_texture",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet",
        "train": "data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet",
        "val": "data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet"
      },
      "test": {
        "accuracy": 0.8468666666666667,
        "f1": 0.7784173124537767,
        "loss": 0.865400637610484,
        "pr_auc": 0.8395983364292976,
        "precision": 0.7518479408658922,
        "recall": 0.8069333333333333,
        "roc_auc": 0.9197980377777778
      },
      "timestamp": "2026-04-28T16:15:01Z",
      "validation": {
        "accuracy": 0.8504888888888888,
        "f1": 0.7838324122863385,
        "loss": 0.8500697920888157,
        "pr_auc": 0.8512632748111177,
        "precision": 0.7565120317539072,
        "recall": 0.8132,
        "roc_auc": 0.9241330433333332
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260428_155338_idea_i011_vetoselect_lc0bt4/checkpoint_best.pt",
        "checkpoint_last": "results/20260428_155338_idea_i011_vetoselect_lc0bt4/checkpoint_last.pt",
        "metrics_final": "results/20260428_155338_idea_i011_vetoselect_lc0bt4/metrics_final.json",
        "report_html": "results/20260428_155338_idea_i011_vetoselect_lc0bt4/report.html",
        "run_metadata": "results/20260428_155338_idea_i011_vetoselect_lc0bt4/run_metadata.json",
        "run_summary": "results/20260428_155338_idea_i011_vetoselect_lc0bt4/run_summary.md"
      },
      "best_epoch": 2,
      "class_counts": {
        "test": {
          "0": 30000,
          "1": 15000
        },
        "train": {
          "0": 240000,
          "1": 120000
        },
        "val": {
          "0": 30000,
          "1": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class_unique_crtk_tags",
      "device": "cuda",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "puzzle_binary",
      "model_name": "vetoselect_positive_claim_abstention",
      "notes": "VetoSelect A2: LC0 BT4-style board-only positive-claim abstention with self-mined decoy negatives after warmup.",
      "num_params": 501602,
      "run_dir": "results/20260428_155338_idea_i011_vetoselect_lc0bt4",
      "run_name": "20260428_155338_idea_i011_vetoselect_lc0bt4",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet",
        "train": "data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet",
        "val": "data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet"
      },
      "test": {
        "accuracy": 0.8275555555555556,
        "f1": 0.7639328303723534,
        "loss": 1.399917987443633,
        "pr_auc": 0.8324266559534413,
        "precision": 0.7025514771709938,
        "recall": 0.8370666666666666,
        "roc_auc": 0.9122847944444445
      },
      "timestamp": "2026-04-28T15:58:20Z",
      "validation": {
        "accuracy": 0.8320222222222222,
        "f1": 0.7699004596511522,
        "loss": 1.398143558178918,
        "pr_auc": 0.8401884133772167,
        "precision": 0.7084196963755532,
        "recall": 0.8430666666666666,
        "roc_auc": 0.9161693577777779
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260428_153027_bench_signal_lc0_bt4_classifier/checkpoint_best.pt",
        "checkpoint_last": "results/20260428_153027_bench_signal_lc0_bt4_classifier/checkpoint_last.pt",
        "metrics_final": "results/20260428_153027_bench_signal_lc0_bt4_classifier/metrics_final.json",
        "report_html": "results/20260428_153027_bench_signal_lc0_bt4_classifier/report.html",
        "run_metadata": "results/20260428_153027_bench_signal_lc0_bt4_classifier/run_metadata.json",
        "run_summary": "results/20260428_153027_bench_signal_lc0_bt4_classifier/run_summary.md"
      },
      "best_epoch": 3,
      "class_counts": {
        "test": {
          "0": 30000,
          "1": 15000
        },
        "train": {
          "0": 240000,
          "1": 120000
        },
        "val": {
          "0": 30000,
          "1": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class_unique_crtk_tags",
      "device": "cuda",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "puzzle_binary",
      "model_name": "lc0_bt4_classifier",
      "notes": "Single-logit LC0 BT4-style puzzle detector: fine labels 0/1 are non-puzzle, fine label 2 is puzzle; reports include 3x2 source-class diagnostics.",
      "num_params": 501473,
      "run_dir": "results/20260428_153027_bench_signal_lc0_bt4_classifier",
      "run_name": "20260428_153027_bench_signal_lc0_bt4_classifier",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet",
        "train": "data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet",
        "val": "data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet"
      },
      "test": {
        "accuracy": 0.8349555555555556,
        "f1": 0.7742347326503937,
        "loss": 0.48700825061838504,
        "pr_auc": 0.838294195943952,
        "precision": 0.7115717718053305,
        "recall": 0.849,
        "roc_auc": 0.916969701111111
      },
      "timestamp": "2026-04-28T15:35:01Z",
      "validation": {
        "accuracy": 0.8385111111111111,
        "f1": 0.7788160097397656,
        "loss": 0.4722135476136612,
        "pr_auc": 0.8483366109308116,
        "precision": 0.7165499859983198,
        "recall": 0.8529333333333333,
        "roc_auc": 0.9216562244444444
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260428_152623_bench_signal_cnn_simple18/checkpoint_best.pt",
        "checkpoint_last": "results/20260428_152623_bench_signal_cnn_simple18/checkpoint_last.pt",
        "metrics_final": "results/20260428_152623_bench_signal_cnn_simple18/metrics_final.json",
        "report_html": "results/20260428_152623_bench_signal_cnn_simple18/report.html",
        "run_metadata": "results/20260428_152623_bench_signal_cnn_simple18/run_metadata.json",
        "run_summary": "results/20260428_152623_bench_signal_cnn_simple18/run_summary.md"
      },
      "best_epoch": 3,
      "class_counts": {
        "test": {
          "0": 30000,
          "1": 15000
        },
        "train": {
          "0": 240000,
          "1": 120000
        },
        "val": {
          "0": 30000,
          "1": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class_unique_crtk_tags",
      "device": "cuda",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "puzzle_binary",
      "model_name": "simple_cnn",
      "notes": "Single-logit puzzle detector CNN: fine labels 0/1 are non-puzzle, fine label 2 is puzzle; reports include 3x2 source-class diagnostics.",
      "num_params": 70417,
      "run_dir": "results/20260428_152623_bench_signal_cnn_simple18",
      "run_name": "20260428_152623_bench_signal_cnn_simple18",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet",
        "train": "data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet",
        "val": "data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet"
      },
      "test": {
        "accuracy": 0.7264222222222222,
        "f1": 0.6803583019602752,
        "loss": 0.6893558813767,
        "pr_auc": 0.7173817299892247,
        "precision": 0.5571762704656602,
        "recall": 0.8734666666666666,
        "roc_auc": 0.8477523577777777
      },
      "timestamp": "2026-04-28T15:30:25Z",
      "validation": {
        "accuracy": 0.7287555555555556,
        "f1": 0.6819200500338771,
        "loss": 0.6870987530459057,
        "pr_auc": 0.7214235472808679,
        "precision": 0.5597672627705998,
        "recall": 0.8722666666666666,
        "roc_auc": 0.8483088133333334
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260428_152223_bench_signal_mlp_simple18/checkpoint_best.pt",
        "checkpoint_last": "results/20260428_152223_bench_signal_mlp_simple18/checkpoint_last.pt",
        "metrics_final": "results/20260428_152223_bench_signal_mlp_simple18/metrics_final.json",
        "report_html": "results/20260428_152223_bench_signal_mlp_simple18/report.html",
        "run_metadata": "results/20260428_152223_bench_signal_mlp_simple18/run_metadata.json",
        "run_summary": "results/20260428_152223_bench_signal_mlp_simple18/run_summary.md"
      },
      "best_epoch": 3,
      "class_counts": {
        "test": {
          "0": 30000,
          "1": 15000
        },
        "train": {
          "0": 240000,
          "1": 120000
        },
        "val": {
          "0": 30000,
          "1": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class_unique_crtk_tags",
      "device": "cuda",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "puzzle_binary",
      "model_name": "mlp",
      "notes": "Single-logit puzzle detector MLP: fine labels 0/1 are non-puzzle, fine label 2 is puzzle; reports include 3x2 source-class diagnostics.",
      "num_params": 756481,
      "run_dir": "results/20260428_152223_bench_signal_mlp_simple18",
      "run_name": "20260428_152223_bench_signal_mlp_simple18",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet",
        "train": "data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet",
        "val": "data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet"
      },
      "test": {
        "accuracy": 0.7512222222222222,
        "f1": 0.6519508782838489,
        "loss": 0.6678324219855395,
        "pr_auc": 0.7239916521060704,
        "precision": 0.610836003495485,
        "recall": 0.699,
        "roc_auc": 0.8316130777777778
      },
      "timestamp": "2026-04-28T15:26:22Z",
      "validation": {
        "accuracy": 0.7515111111111111,
        "f1": 0.6519330137583266,
        "loss": 0.6644854504953731,
        "pr_auc": 0.7247120428768786,
        "precision": 0.6114679434777531,
        "recall": 0.6981333333333334,
        "roc_auc": 0.833689648888889
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260428_151823_bench_signal_stockfish_style_nnue_simple18/checkpoint_best.pt",
        "checkpoint_last": "results/20260428_151823_bench_signal_stockfish_style_nnue_simple18/checkpoint_last.pt",
        "metrics_final": "results/20260428_151823_bench_signal_stockfish_style_nnue_simple18/metrics_final.json",
        "report_html": "results/20260428_151823_bench_signal_stockfish_style_nnue_simple18/report.html",
        "run_metadata": "results/20260428_151823_bench_signal_stockfish_style_nnue_simple18/run_metadata.json",
        "run_summary": "results/20260428_151823_bench_signal_stockfish_style_nnue_simple18/run_summary.md"
      },
      "best_epoch": 3,
      "class_counts": {
        "test": {
          "0": 30000,
          "1": 15000
        },
        "train": {
          "0": 240000,
          "1": 120000
        },
        "val": {
          "0": 30000,
          "1": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class_unique_crtk_tags",
      "device": "cuda",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "puzzle_binary",
      "model_name": "stockfish_nnue",
      "notes": "Single-logit puzzle detector: fine labels 0/1 are non-puzzle, fine label 2 is puzzle; reports include 3x2 source-class diagnostics.",
      "num_params": 271617,
      "run_dir": "results/20260428_151823_bench_signal_stockfish_style_nnue_simple18",
      "run_name": "20260428_151823_bench_signal_stockfish_style_nnue_simple18",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet",
        "train": "data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet",
        "val": "data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet"
      },
      "test": {
        "accuracy": 0.8005555555555556,
        "f1": 0.7354770255533615,
        "loss": 0.5516478070481257,
        "pr_auc": 0.7994308846593219,
        "precision": 0.659147340060225,
        "recall": 0.8318,
        "roc_auc": 0.8918766622222223
      },
      "timestamp": "2026-04-28T15:22:21Z",
      "validation": {
        "accuracy": 0.802,
        "f1": 0.7371216144450345,
        "loss": 0.5489889870990406,
        "pr_auc": 0.8017957106879039,
        "precision": 0.6611622737376945,
        "recall": 0.8328,
        "roc_auc": 0.8928327844444445
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260424_154030_bench_signal_lc0_bt4_classifier/checkpoint_best.pt",
        "checkpoint_last": "results/20260424_154030_bench_signal_lc0_bt4_classifier/checkpoint_last.pt",
        "metrics_final": "results/20260424_154030_bench_signal_lc0_bt4_classifier/metrics_final.json",
        "report_html": "results/20260424_154030_bench_signal_lc0_bt4_classifier/report.html",
        "run_metadata": "results/20260424_154030_bench_signal_lc0_bt4_classifier/run_metadata.json",
        "run_summary": "results/20260424_154030_bench_signal_lc0_bt4_classifier/run_summary.md"
      },
      "best_epoch": 1,
      "class_counts": {
        "test": {
          "0": 30000,
          "1": 15000
        },
        "train": {
          "0": 240000,
          "1": 120000
        },
        "val": {
          "0": 30000,
          "1": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class",
      "device": "cpu",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "puzzle_binary",
      "model_name": "lc0_bt4_classifier",
      "notes": "Single-logit LC0 BT4-style puzzle detector: fine labels 0/1 are non-puzzle, fine label 2 is puzzle; reports include 3x2 source-class diagnostics.",
      "num_params": 501473,
      "run_dir": "results/20260424_154030_bench_signal_lc0_bt4_classifier",
      "run_name": "20260424_154030_bench_signal_lc0_bt4_classifier",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class/split_test.parquet",
        "train": "data/splits/crtk_sample_3class/split_train.parquet",
        "val": "data/splits/crtk_sample_3class/split_val.parquet"
      },
      "test": {
        "accuracy": 0.8182666666666667,
        "f1": 0.7444854089858152,
        "loss": 0.5424360170707865,
        "pr_auc": 0.8068225106572791,
        "precision": 0.7005762671998118,
        "recall": 0.7942666666666667,
        "roc_auc": 0.8989450411111112
      },
      "timestamp": "2026-04-24T15:51:35Z",
      "validation": {
        "accuracy": 0.8214,
        "f1": 0.748474321659938,
        "loss": 0.5324629744735815,
        "pr_auc": 0.8163493339370635,
        "precision": 0.7053618828525925,
        "recall": 0.7972,
        "roc_auc": 0.903038211111111
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260424_153505_bench_signal_cnn_simple18/checkpoint_best.pt",
        "checkpoint_last": "results/20260424_153505_bench_signal_cnn_simple18/checkpoint_last.pt",
        "metrics_final": "results/20260424_153505_bench_signal_cnn_simple18/metrics_final.json",
        "report_html": "results/20260424_153505_bench_signal_cnn_simple18/report.html",
        "run_metadata": "results/20260424_153505_bench_signal_cnn_simple18/run_metadata.json",
        "run_summary": "results/20260424_153505_bench_signal_cnn_simple18/run_summary.md"
      },
      "best_epoch": 3,
      "class_counts": {
        "test": {
          "0": 30000,
          "1": 15000
        },
        "train": {
          "0": 240000,
          "1": 120000
        },
        "val": {
          "0": 30000,
          "1": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class",
      "device": "cpu",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "puzzle_binary",
      "model_name": "simple_cnn",
      "notes": "Single-logit puzzle detector CNN: fine labels 0/1 are non-puzzle, fine label 2 is puzzle; reports include 3x2 source-class diagnostics.",
      "num_params": 70417,
      "run_dir": "results/20260424_153505_bench_signal_cnn_simple18",
      "run_name": "20260424_153505_bench_signal_cnn_simple18",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class/split_test.parquet",
        "train": "data/splits/crtk_sample_3class/split_train.parquet",
        "val": "data/splits/crtk_sample_3class/split_val.parquet"
      },
      "test": {
        "accuracy": 0.7605777777777778,
        "f1": 0.6823328222667767,
        "loss": 0.6820345812223174,
        "pr_auc": 0.7026285871660931,
        "precision": 0.6117043772467752,
        "recall": 0.7714,
        "roc_auc": 0.8418866066666668
      },
      "timestamp": "2026-04-24T15:40:26Z",
      "validation": {
        "accuracy": 0.7645777777777778,
        "f1": 0.6866422148603881,
        "loss": 0.6786931116472591,
        "pr_auc": 0.7092840934355649,
        "precision": 0.6171310080816673,
        "recall": 0.7738,
        "roc_auc": 0.8444511688888887
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260424_153123_bench_signal_mlp_simple18/checkpoint_best.pt",
        "checkpoint_last": "results/20260424_153123_bench_signal_mlp_simple18/checkpoint_last.pt",
        "metrics_final": "results/20260424_153123_bench_signal_mlp_simple18/metrics_final.json",
        "report_html": "results/20260424_153123_bench_signal_mlp_simple18/report.html",
        "run_metadata": "results/20260424_153123_bench_signal_mlp_simple18/run_metadata.json",
        "run_summary": "results/20260424_153123_bench_signal_mlp_simple18/run_summary.md"
      },
      "best_epoch": 2,
      "class_counts": {
        "test": {
          "0": 30000,
          "1": 15000
        },
        "train": {
          "0": 240000,
          "1": 120000
        },
        "val": {
          "0": 30000,
          "1": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class",
      "device": "cpu",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "puzzle_binary",
      "model_name": "mlp",
      "notes": "Single-logit puzzle detector MLP: fine labels 0/1 are non-puzzle, fine label 2 is puzzle; reports include 3x2 source-class diagnostics.",
      "num_params": 756481,
      "run_dir": "results/20260424_153123_bench_signal_mlp_simple18",
      "run_name": "20260424_153123_bench_signal_mlp_simple18",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class/split_test.parquet",
        "train": "data/splits/crtk_sample_3class/split_train.parquet",
        "val": "data/splits/crtk_sample_3class/split_val.parquet"
      },
      "test": {
        "accuracy": 0.693,
        "f1": 0.6502620186830713,
        "loss": 0.6968209784139286,
        "pr_auc": 0.7089334656275097,
        "precision": 0.5241826864209624,
        "recall": 0.8562,
        "roc_auc": 0.8244560788888888
      },
      "timestamp": "2026-04-24T15:34:59Z",
      "validation": {
        "accuracy": 0.6944666666666667,
        "f1": 0.652284970031107,
        "loss": 0.6960642805153673,
        "pr_auc": 0.7110748964106426,
        "precision": 0.5254879589258792,
        "recall": 0.8597333333333333,
        "roc_auc": 0.8261967277777778
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260424_152740_bench_signal_stockfish_style_nnue_simple18/checkpoint_best.pt",
        "checkpoint_last": "results/20260424_152740_bench_signal_stockfish_style_nnue_simple18/checkpoint_last.pt",
        "metrics_final": "results/20260424_152740_bench_signal_stockfish_style_nnue_simple18/metrics_final.json",
        "report_html": "results/20260424_152740_bench_signal_stockfish_style_nnue_simple18/report.html",
        "run_metadata": "results/20260424_152740_bench_signal_stockfish_style_nnue_simple18/run_metadata.json",
        "run_summary": "results/20260424_152740_bench_signal_stockfish_style_nnue_simple18/run_summary.md"
      },
      "best_epoch": 3,
      "class_counts": {
        "test": {
          "0": 30000,
          "1": 15000
        },
        "train": {
          "0": 240000,
          "1": 120000
        },
        "val": {
          "0": 30000,
          "1": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class",
      "device": "cpu",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "puzzle_binary",
      "model_name": "stockfish_nnue",
      "notes": "Single-logit puzzle detector: fine labels 0/1 are non-puzzle, fine label 2 is puzzle; reports include 3x2 source-class diagnostics.",
      "num_params": 271617,
      "run_dir": "results/20260424_152740_bench_signal_stockfish_style_nnue_simple18",
      "run_name": "20260424_152740_bench_signal_stockfish_style_nnue_simple18",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class/split_test.parquet",
        "train": "data/splits/crtk_sample_3class/split_train.parquet",
        "val": "data/splits/crtk_sample_3class/split_val.parquet"
      },
      "test": {
        "accuracy": 0.7988222222222222,
        "f1": 0.7339934769194605,
        "loss": 0.5560357946563851,
        "pr_auc": 0.7981507988491225,
        "precision": 0.656228655493091,
        "recall": 0.8326666666666667,
        "roc_auc": 0.8905058133333332
      },
      "timestamp": "2026-04-24T15:31:19Z",
      "validation": {
        "accuracy": 0.7978444444444445,
        "f1": 0.732590611129075,
        "loss": 0.5555797581645575,
        "pr_auc": 0.8020116087752603,
        "precision": 0.6551869183448131,
        "recall": 0.8307333333333333,
        "roc_auc": 0.8909406466666667
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260424_150558_bench_fine3_lc0_bt4_classifier/checkpoint_best.pt",
        "checkpoint_last": "results/20260424_150558_bench_fine3_lc0_bt4_classifier/checkpoint_last.pt",
        "metrics_final": "results/20260424_150558_bench_fine3_lc0_bt4_classifier/metrics_final.json",
        "report_html": "results/20260424_150558_bench_fine3_lc0_bt4_classifier/report.html",
        "run_metadata": "results/20260424_150558_bench_fine3_lc0_bt4_classifier/run_metadata.json",
        "run_summary": "results/20260424_150558_bench_fine3_lc0_bt4_classifier/run_summary.md"
      },
      "best_epoch": 3,
      "class_counts": {
        "test": {
          "0": 15000,
          "1": 15000,
          "2": 15000
        },
        "train": {
          "0": 120000,
          "1": 120000,
          "2": 120000
        },
        "val": {
          "0": 15000,
          "1": 15000,
          "2": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class",
      "device": "cpu",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "fine_3class",
      "model_name": "lc0_bt4_classifier",
      "notes": "3-class LC0 BT4-style tower benchmark: random/non-puzzle vs near-puzzle hard negative vs puzzle.",
      "num_params": 501731,
      "run_dir": "results/20260424_150558_bench_fine3_lc0_bt4_classifier",
      "run_name": "20260424_150558_bench_fine3_lc0_bt4_classifier",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class/split_test.parquet",
        "train": "data/splits/crtk_sample_3class/split_train.parquet",
        "val": "data/splits/crtk_sample_3class/split_val.parquet"
      },
      "test": {
        "accuracy": 0.6855111111111111,
        "f1": 0.6823330460615388,
        "loss": 0.7008048482870651,
        "pr_auc": null,
        "precision": null,
        "recall": null,
        "roc_auc": null
      },
      "timestamp": "2026-04-24T15:17:28Z",
      "validation": {
        "accuracy": 0.6873111111111111,
        "f1": 0.6841112016769042,
        "loss": 0.6929107823614347,
        "pr_auc": null,
        "precision": null,
        "recall": null,
        "roc_auc": null
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260424_150002_bench_fine3_cnn_simple18/checkpoint_best.pt",
        "checkpoint_last": "results/20260424_150002_bench_fine3_cnn_simple18/checkpoint_last.pt",
        "metrics_final": "results/20260424_150002_bench_fine3_cnn_simple18/metrics_final.json",
        "report_html": "results/20260424_150002_bench_fine3_cnn_simple18/report.html",
        "run_metadata": "results/20260424_150002_bench_fine3_cnn_simple18/run_metadata.json",
        "run_summary": "results/20260424_150002_bench_fine3_cnn_simple18/run_summary.md"
      },
      "best_epoch": 3,
      "class_counts": {
        "test": {
          "0": 15000,
          "1": 15000,
          "2": 15000
        },
        "train": {
          "0": 120000,
          "1": 120000,
          "2": 120000
        },
        "val": {
          "0": 15000,
          "1": 15000,
          "2": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class",
      "device": "cpu",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "fine_3class",
      "model_name": "simple_cnn",
      "notes": "3-class plain CNN benchmark: random/non-puzzle vs near-puzzle hard negative vs puzzle.",
      "num_params": 70515,
      "run_dir": "results/20260424_150002_bench_fine3_cnn_simple18",
      "run_name": "20260424_150002_bench_fine3_cnn_simple18",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class/split_test.parquet",
        "train": "data/splits/crtk_sample_3class/split_train.parquet",
        "val": "data/splits/crtk_sample_3class/split_val.parquet"
      },
      "test": {
        "accuracy": 0.6047555555555556,
        "f1": 0.5949009309681154,
        "loss": 0.866994321346283,
        "pr_auc": null,
        "precision": null,
        "recall": null,
        "roc_auc": null
      },
      "timestamp": "2026-04-24T15:05:53Z",
      "validation": {
        "accuracy": 0.6039555555555556,
        "f1": 0.5942273168674933,
        "loss": 0.8648269616744735,
        "pr_auc": null,
        "precision": null,
        "recall": null,
        "roc_auc": null
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260424_145600_bench_fine3_mlp_simple18/checkpoint_best.pt",
        "checkpoint_last": "results/20260424_145600_bench_fine3_mlp_simple18/checkpoint_last.pt",
        "metrics_final": "results/20260424_145600_bench_fine3_mlp_simple18/metrics_final.json",
        "report_html": "results/20260424_145600_bench_fine3_mlp_simple18/report.html",
        "run_metadata": "results/20260424_145600_bench_fine3_mlp_simple18/run_metadata.json",
        "run_summary": "results/20260424_145600_bench_fine3_mlp_simple18/run_summary.md"
      },
      "best_epoch": 3,
      "class_counts": {
        "test": {
          "0": 15000,
          "1": 15000,
          "2": 15000
        },
        "train": {
          "0": 120000,
          "1": 120000,
          "2": 120000
        },
        "val": {
          "0": 15000,
          "1": 15000,
          "2": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class",
      "device": "cpu",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "fine_3class",
      "model_name": "mlp",
      "notes": "3-class flattened board-plane MLP benchmark: random/non-puzzle vs near-puzzle hard negative vs puzzle.",
      "num_params": 756739,
      "run_dir": "results/20260424_145600_bench_fine3_mlp_simple18",
      "run_name": "20260424_145600_bench_fine3_mlp_simple18",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class/split_test.parquet",
        "train": "data/splits/crtk_sample_3class/split_train.parquet",
        "val": "data/splits/crtk_sample_3class/split_val.parquet"
      },
      "test": {
        "accuracy": 0.5964444444444444,
        "f1": 0.5873264268040237,
        "loss": 0.8498655422167345,
        "pr_auc": null,
        "precision": null,
        "recall": null,
        "roc_auc": null
      },
      "timestamp": "2026-04-24T14:59:58Z",
      "validation": {
        "accuracy": 0.5933333333333334,
        "f1": 0.5838134966507252,
        "loss": 0.8491102030331438,
        "pr_auc": null,
        "precision": null,
        "recall": null,
        "roc_auc": null
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260424_145206_bench_fine3_stockfish_style_nnue_simple18/checkpoint_best.pt",
        "checkpoint_last": "results/20260424_145206_bench_fine3_stockfish_style_nnue_simple18/checkpoint_last.pt",
        "metrics_final": "results/20260424_145206_bench_fine3_stockfish_style_nnue_simple18/metrics_final.json",
        "report_html": "results/20260424_145206_bench_fine3_stockfish_style_nnue_simple18/report.html",
        "run_metadata": "results/20260424_145206_bench_fine3_stockfish_style_nnue_simple18/run_metadata.json",
        "run_summary": "results/20260424_145206_bench_fine3_stockfish_style_nnue_simple18/run_summary.md"
      },
      "best_epoch": 3,
      "class_counts": {
        "test": {
          "0": 15000,
          "1": 15000,
          "2": 15000
        },
        "train": {
          "0": 120000,
          "1": 120000,
          "2": 120000
        },
        "val": {
          "0": 15000,
          "1": 15000,
          "2": 15000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class",
      "device": "cpu",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "fine_3class",
      "model_name": "stockfish_nnue",
      "notes": "3-class Stockfish-style NNUE benchmark: random/non-puzzle vs near-puzzle hard negative vs puzzle.",
      "num_params": 271747,
      "run_dir": "results/20260424_145206_bench_fine3_stockfish_style_nnue_simple18",
      "run_name": "20260424_145206_bench_fine3_stockfish_style_nnue_simple18",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class/split_test.parquet",
        "train": "data/splits/crtk_sample_3class/split_train.parquet",
        "val": "data/splits/crtk_sample_3class/split_val.parquet"
      },
      "test": {
        "accuracy": 0.6548444444444445,
        "f1": 0.6553297146283983,
        "loss": 0.7617115053263578,
        "pr_auc": null,
        "precision": null,
        "recall": null,
        "roc_auc": null
      },
      "timestamp": "2026-04-24T14:55:54Z",
      "validation": {
        "accuracy": 0.6534,
        "f1": 0.6537095033532955,
        "loss": 0.7607737915082411,
        "pr_auc": null,
        "precision": null,
        "recall": null,
        "roc_auc": null
      }
    },
    {
      "artifacts": {
        "checkpoint_best": "results/20260420_070900_bench_cnn_small_simple18/checkpoint_best.pt",
        "checkpoint_last": "results/20260420_070900_bench_cnn_small_simple18/checkpoint_last.pt",
        "metrics_final": "results/20260420_070900_bench_cnn_small_simple18/metrics_final.json",
        "report_html": "results/20260420_070900_bench_cnn_small_simple18/report.html",
        "run_metadata": "results/20260420_070900_bench_cnn_small_simple18/run_metadata.json",
        "run_summary": "results/20260420_070900_bench_cnn_small_simple18/run_summary.md"
      },
      "best_epoch": 2,
      "class_counts": {
        "test": {
          "0": 15000,
          "1": 30000
        },
        "train": {
          "0": 120000,
          "1": 240000
        },
        "val": {
          "0": 15000,
          "1": 30000
        }
      },
      "dataset_path": "data/splits/crtk_sample_3class",
      "device": "cpu",
      "is_smoke_or_tiny_run": false,
      "metric_reasons": {
        "test": {},
        "validation": {}
      },
      "mode": "coarse_binary",
      "model_name": "simple_cnn",
      "notes": "Small simple CNN baseline, simple 18-plane FEN encoding.",
      "num_params": 23874,
      "run_dir": "results/20260420_070900_bench_cnn_small_simple18",
      "run_name": "20260420_070900_bench_cnn_small_simple18",
      "sample_counts": {
        "test": 45000,
        "total": 450000,
        "train": 360000,
        "val": 45000
      },
      "split_paths": {
        "test": "data/splits/crtk_sample_3class/split_test.parquet",
        "train": "data/splits/crtk_sample_3class/split_train.parquet",
        "val": "data/splits/crtk_sample_3class/split_val.parquet"
      },
      "test": {
        "accuracy": 0.7048666666666666,
        "f1": 0.7622065853789547,
        "loss": 0.577568688175895,
        "pr_auc": 0.84736347775601,
        "precision": 0.8233724033886504,
        "recall": 0.7095,
        "roc_auc": 0.77198059
      },
      "timestamp": "2026-04-20T07:26:41Z",
      "validation": {
        "accuracy": 0.7054444444444444,
        "f1": 0.7625360540317813,
        "loss": 0.577195135707205,
        "pr_auc": 0.8462964179653759,
        "precision": 0.8242766954568341,
        "recall": 0.7094,
        "roc_auc": 0.7729303433333333
      }
    }
  ],
  "runs_found": 23
}
```

## Current Leaderboard

```markdown
| run_name | created_at | seed | mode | model_name | num_params | train_samples | val_samples | test_samples | best_val_loss | best_val_accuracy | best_val_f1 | test_accuracy | test_precision | test_recall | test_f1 | test_roc_auc | test_pr_auc | roc_auc | pr_auc | worst_test_slice | worst_test_slice_accuracy | worst_test_slice_rows | worst_val_slice | worst_val_slice_accuracy | checkpoint_path | report_path | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4 | 2026-04-29T02:28:14Z | 42.0 | puzzle_binary | sparse_relation_pursuit_asymmetry | 121115 | 360000 | 45000 | 45000 | 0.5369015578815545 | 0.8623111111111111 | 0.8095764951748724 | 0.8608222222222223 | 0.7495002570106802 | 0.8748666666666667 | 0.8073456581254422 | 0.936883571111111 | 0.8687288947146012 | 0.938883978888889 | 0.8748444507782331 | crtk_eval_bucket=equal | 0.757 | 7376.0 | crtk_eval_bucket=equal | 0.76 | results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4/checkpoint_best.pt | results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4/run_summary.md | Sparse Relation Pursuit Asymmetry v1: deterministic chess relation tokens, equal-capacity background/tactical sparse dictionaries, LISTA-style group pursuit, no dense classifier bypass, and paper-grade CUDA training defaults. |
| 20260429_023704_idea_i005_null_move_contrast_simple18 | 2026-04-29T02:59:20Z | 42.0 | puzzle_binary | null_move_contrast_puzzle_network | 240578 | 360000 | 45000 | 45000 | 0.46844687685370445 | 0.8492666666666666 | 0.7932327389117513 | 0.8478888888888889 | 0.7291889157439154 | 0.8648666666666667 | 0.7912536976609436 | 0.9274141211111111 | 0.8538221663249441 | 0.9292866655555556 | 0.8595689121188905 | crtk_eval_bucket=equal | 0.7294 | 7376.0 | crtk_eval_bucket=equal | 0.7346 | results/20260429_023704_idea_i005_null_move_contrast_simple18/checkpoint_best.pt | results/20260429_023704_idea_i005_null_move_contrast_simple18/run_summary.md | Implemented GPU-required benchmark config for the null-move contrast architecture. |
| 20260428_164249_idea_i012_dykstra_vetoselect_lc0bt4_v2 | 2026-04-28T16:54:21Z | nan | puzzle_binary | dykstra_vetoselect | 531584 | 360000 | 45000 | 45000 | 0.640758460252843 | 0.8344444444444444 | 0.7810626542847067 | 0.8334666666666667 | 0.6975679090334808 | 0.8834 | 0.779562301447229 | 0.923047188888889 | 0.8474298736067405 | 0.9251379244444444 | 0.8540147525919779 | crtk_eval_bucket=equal | 0.7034 | 7376.0 | crtk_eval_bucket=equal | 0.7108 | results/20260428_164249_idea_i012_dykstra_vetoselect_lc0bt4_v2/checkpoint_best.pt | results/20260428_164249_idea_i012_dykstra_vetoselect_lc0bt4_v2/run_summary.md | Dykstra-LCP v2 hybrid: projection diagnostics feed a VetoSelect positive-claim accept/reject head with projection-weighted decoy mining. |
| 20260428_180243_idea_i009_tactical_equilibrium_simple18 | 2026-04-28T18:28:04Z | 42.0 | puzzle_binary | tactical_equilibrium_network | 176676 | 360000 | 45000 | 45000 | 0.4749451650475914 | 0.8485555555555555 | 0.7893028288761788 | 0.8455555555555555 | 0.7315081099735419 | 0.8478666666666667 | 0.7854010992404126 | 0.9233379388888888 | 0.8468951273506644 | 0.9258365500000001 | 0.8539998762095637 | crtk_eval_bucket=equal | 0.7355 | 7376.0 | crtk_eval_bucket=equal | 0.7318 | results/20260428_180243_idea_i009_tactical_equilibrium_simple18/checkpoint_best.pt | results/20260428_180243_idea_i009_tactical_equilibrium_simple18/run_summary.md | Implemented GPU-required benchmark config for the tactical-equilibrium architecture. |
| 20260428_160748_idea_i011_vetoselect_lc0bt4_v2_texture | 2026-04-28T16:15:01Z | nan | puzzle_binary | vetoselect_positive_claim_abstention | 501602 | 360000 | 45000 | 45000 | 0.8500697920888157 | 0.8504888888888888 | 0.7838324122863385 | 0.8468666666666667 | 0.7518479408658922 | 0.8069333333333333 | 0.7784173124537767 | 0.9197980377777778 | 0.8395983364292976 | 0.9241330433333332 | 0.8512632748111177 | crtk_eval_bucket=equal | 0.7202 | 7376.0 | crtk_eval_bucket=equal | 0.7283 | results/20260428_160748_idea_i011_vetoselect_lc0bt4_v2_texture/checkpoint_best.pt | results/20260428_160748_idea_i011_vetoselect_lc0bt4_v2_texture/run_summary.md | VetoSelect v2/A3: board-only model with deterministic rule-texture-weighted self-mined decoy negatives after warmup. |
| 20260428_153027_bench_signal_lc0_bt4_classifier | 2026-04-28T15:35:01Z | nan | puzzle_binary | lc0_bt4_classifier | 501473 | 360000 | 45000 | 45000 | 0.4722135476136612 | 0.8385111111111111 | 0.7788160097397656 | 0.8349555555555556 | 0.7115717718053305 | 0.849 | 0.7742347326503937 | 0.916969701111111 | 0.838294195943952 | 0.9216562244444444 | 0.8483366109308116 | crtk_eval_bucket=equal | 0.7038 | 7376.0 | crtk_eval_bucket=equal | 0.7093 | results/20260428_153027_bench_signal_lc0_bt4_classifier/checkpoint_best.pt | results/20260428_153027_b
...<truncated>
```

## How To Interpret These Runs

- If `is_smoke_or_tiny_run` is true, treat the run only as infrastructure validation.
- Do not make scientific claims from smoke runs.
- A benchmark is meaningful only if it has enough samples, balanced or documented class counts, leakage-safe splits, and a clear dataset source.
- For `coarse_binary`, the task is known non-puzzle versus unresolved candidate pool.
- For `fine_3class`, only use rows where `fine_label` is truly `0`, `1`, or `2`; do not infer class 1 or class 2 from the candidate pool.

## What I Want You To Do

When I paste this prompt into ChatGPT Pro, do the following:

1. Summarize what has actually been tested.
2. Identify which runs are smoke tests and which, if any, are real benchmarks.
3. State the best current result per mode, but only if the run is not tiny.
4. List metric gaps, missing labels, class imbalance, leakage risks, and data-quality blockers.
5. Recommend the next 1-3 practical experiments or data actions.
6. If suggesting a new run, give exact command-line steps and what result would count as success or failure.
7. If suggesting label work, keep verified labels separate from engine-proposed or weak labels.
8. Do not repeat old ideas or rebrand ordinary CNN hyperparameter changes as novel research ideas.

Keep the answer concrete and evidence-bound.
