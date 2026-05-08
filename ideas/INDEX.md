# Ideas Index

This is the navigation file for the `ideas/` workspace. It separates implementable registered ideas from raw research packets so future Codex sessions can move directly from research to code.

Architectural honesty note: `implementation_status: implemented` / `tested` is reserved for trainable bespoke architecture implementations. Shared-probe folders are marked scaffold-only until their markdown thesis has matching bespoke code.

## What Goes Where

| Path | Role | Edit policy |
|---|---|---|
| `ideas/i###_*` | Registered idea folders with documentation, metadata, and either bespoke implementation code or explicit scaffold-only status. | Update when promoting, implementing, benchmarking, or rejecting an idea. |
| `ideas/registry.jsonl` | Machine-readable list of registered ideas. | Append/update only for registered ideas, not raw packets. |
| `ideas/TODO.md` | Execution checklist with implementation state, performance state, and next action. | Regenerate after changing idea status or packet imports. |
| `ideas/research_packets/` | Raw imported or generated research handoff packets. | Keep packet files immutable except filename/metadata normalization; use catalogs for organization. |
| `ideas/idea_template/` | Scaffold for a future registered idea folder. | Keep as template only. |
| `ideas/BENCHMARK_REPORTING.md` | Required aggregate and slice-level reporting standard. | Update when benchmark metadata or report artifacts change. |
| `ideas/chatgpt_pro_deep_math_research_prompt.md` | Deep Research prompt with duplicate-memory rules. | Update after importing meaningful new packets. |
| `ideas/idea_generation_prompt.md` | Generated prompt from `scripts/ideas/build_idea_prompt.py`. | Do not hand-edit; regenerate. |

## Current Counts

- Registered idea folders: `240`
- Research packet files cataloged: `130`
- Registered implementation states: `{'implemented': 161, 'probe_scaffold_only': 72, 'tested': 7}`
- Registered implementation kinds: `{'bespoke_model': 168, 'shared_probe_variant': 72}`
- Research packet statuses: `{'batch packet': 19, 'duplicate import': 5, 'handoff packet': 70, 'link stub': 3, 'prompt snapshot': 1, 'research packet': 30, 'synthesis packet': 2}`

| Implementation kind | Count | Meaning |
|---|---:|---|
| `bespoke_model` | 168 | Materially distinct model implementation. |
| `shared_probe_variant` | 72 | Thin wrapper around `ResearchPacketProbe`; not a separate bespoke architecture. |
| `other_shared_scaffold` | 0 | Thin wrapper around another shared scaffold/baseline builder. |
| `unknown` | 0 | Not classifiable from current wiring; should remain rare. |

Full implementation-kind audit: [ideas/implementation_audit.md](implementation_audit.md) and [ideas/implementation_audit.json](implementation_audit.json).
Implemented-architecture conformance audit: [ideas/architecture_conformance_audit.md](architecture_conformance_audit.md) and [ideas/architecture_conformance_audit.json](architecture_conformance_audit.json).

## Registered Ideas

| ID | Idea | Status | Trainable state | Implementation kind | Target |
|---|---|---|---|---|---|
| `i001` | [Chess Operator Basis Classifier](i001_chess_operator_basis_classifier) | `implemented` | `implemented` | `bespoke_model` | General chess position classification; first benchmark is puzzle_binary with source cla... |
| `i002` | [Response-Minimax Chess Classifier](i002_response_minimax_classifier) | `implemented` | `implemented` | `bespoke_model` | General chess position classification; first benchmark is puzzle_binary. |
| `i003` | [Factor-Agreement Chess Classifier](i003_factor_agreement_classifier) | `implemented` | `implemented` | `bespoke_model` | General chess position classification; first benchmark is puzzle_binary. |
| `i004` | [Puzzle Obligation Flow Network](i004_puzzle_obligation_flow_network) | `implemented` | `implemented` | `bespoke_model` | Puzzle-specific architecture for puzzle_binary: source classes 0 and 1 map to target 0,... |
| `i005` | [Null-Move Contrast Puzzle Network](i005_null_move_contrast_puzzle_network) | `tested` | `tested` | `bespoke_model` | Puzzle-specific architecture for puzzle_binary: source classes 0 and 1 map to target 0,... |
| `i006` | [Proof-Core Set Verifier](i006_proof_core_set_verifier) | `implemented` | `implemented` | `bespoke_model` | Puzzle-specific architecture for puzzle_binary: source classes 0 and 1 map to target 0,... |
| `i007` | [Neural Proof-Number Search Network](i007_neural_proof_number_search) | `tested` | `tested` | `bespoke_model` | Puzzle-specific architecture for puzzle_binary: source classes 0 and 1 map to target 0,... |
| `i008` | [Boundary-Edit Lagrangian Network](i008_boundary_edit_lagrangian_network) | `tested` | `tested` | `bespoke_model` | Puzzle-specific architecture for puzzle_binary: source classes 0 and 1 map to target 0,... |
| `i009` | [Tactical Equilibrium Network](i009_tactical_equilibrium_network) | `tested` | `tested` | `bespoke_model` | Puzzle-specific architecture for puzzle_binary: source classes 0 and 1 map to target 0,... |
| `i010` | [Rule-Consistent Latent Dynamics Network](i010_rule_consistent_latent_dynamics) | `implemented` | `implemented` | `bespoke_model` | General chess position classification with puzzle_binary as the first benchmark. |
| `i011` | [VetoSelect Positive-Claim Abstention](i011_vetoselect_positive_claim_abstention) | `tested` | `tested` | `bespoke_model` | Puzzle-specific puzzle_binary classification: source classes 0 and 1 map to non-puzzle,... |
| `i012` | [Soft-Dykstra Latent Constraint Projector](i012_dykstra_lcp) | `tested` | `tested` | `bespoke_model` | Puzzle-specific puzzle_binary classification: source classes 0 and 1 map to non-puzzle,... |
| `i013` | [Sparse Relation Pursuit Asymmetry](i013_sparse_relation_pursuit_asymmetry) | `tested` | `tested` | `bespoke_model` | Puzzle-specific puzzle_binary classification: source classes 0 and 1 map to non-puzzle,... |
| `i014` | [Contamination-DRO Huber Tail Rejection](i014_contamination_dro_huber_tail_rejection) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i015` | [Material-Locked Tactical Mask DRO](i015_material_locked_tactical_dro) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification on the canonical CRTK split. |
| `i016` | [Soft Sorting Order Residual Ranker](i016_soft_sorting_order_residual_ranker) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification. |
| `i017` | [Conditional Surprisal Gate](i017_conditional_surprisal_gate) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification. |
| `i018` | [Oriented Tactical Sheaf Laplacian](i018_oriented_tactical_sheaf_laplacian) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i019` | [Tactical Sheaf Curvature Network](i019_tactical_sheaf_curvature_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i020` | [Attack-Defense Sheaf Energy Network](i020_attack_defense_sheaf_energy_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i021` | [Tactical Sheaf Tension Network](i021_tactical_sheaf_tension_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i022` | [Tactical Threat-Sheaf Network](i022_tactical_threat_sheaf_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i023` | [Attack-Hodge Sheaf Tension Network](i023_attack_hodge_sheaf_tension_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i024` | [Directed Attack-Sheaf Tension Network](i024_directed_attack_sheaf_tension_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i025` | [One-Ply Counterfactual Move Landscape Network](i025_one_ply_counterfactual_move_landscape_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i026` | [Counterfactual Move-Delta Spectrum Network](i026_counterfactual_move_delta_spectrum_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i027` | [Rule-Only Counterfactual Move-Delta Bottleneck](i027_rule_only_counterfactual_move_delta_bottleneck) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i028` | [File-Mirror Tension Sheaf](i028_file_mirror_tension_sheaf) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i029` | [Entropic Piece-Target Transport Bottleneck](i029_entropic_piece_target_transport_bottleneck) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i030` | [Nuisance-Orthogonal Puzzle Bottleneck](i030_nuisance_orthogonal_puzzle_bottleneck) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i031` | [Tactical Transport Imbalance Network](i031_tactical_transport_imbalance_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i032` | [King-Anchored Material-Null Transport Bottleneck](i032_king_anchored_material_null_transport_bottleneck) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i033` | [Piece-Target Entropic Transport Bottleneck](i033_piece_target_entropic_transport_bottleneck) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i034` | [Entropic Chess Geometry Transport Network](i034_entropic_chess_geometry_transport_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i035` | [Ordinal Evidence Ladder Network](i035_ordinal_evidence_ladder_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i036` | [Geometry-Conditioned Board Pseudo-Likelihood Ratio Network](i036_geometry_conditioned_board_pseudo_likelihood_ratio_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i037` | [Möbius Piece-Constellation Network](i037_mobius_piece_constellation_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i038` | [Sparse Witness-Piece Bottleneck Network](i038_sparse_witness_piece_bottleneck_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i039` | [Ray-Language Automaton Network](i039_ray_language_automaton_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i040` | [Kinematic Commutator Bottleneck Network](i040_kinematic_commutator_bottleneck_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i041` | [Centered Tempo-Odd Interventional Bottleneck](i041_centered_tempo_odd_interventional_bottleneck) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i042` | [Legal Automorphism Quotient Network](i042_legal_automorphism_quotient_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i043` | [Side-Canonical Rule-Partition Invariant Bottleneck](i043_side_canonical_rule_partition_invariant_bottleneck) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i044` | [Masked Board Code-Length Surprise Network](i044_masked_board_code_length_surprise_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i045` | [Credal Near-Puzzle Evidence Network](i045_credal_near_puzzle_evidence_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i046` | [Rule-Exact Orbit Bottleneck Network](i046_rule_exact_orbit_bottleneck_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i047` | [Color-Flip Orbit Evidence Bottleneck](i047_color_flip_orbit_evidence_bottleneck) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i048` | [Rule-Automorphism Quotient Bottleneck Network](i048_rule_automorphism_quotient_bottleneck_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i049` | [Tempo-Odd Bottleneck Network](i049_tempo_odd_bottleneck_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i050` | [King-Anchored Euler Interaction Network](i050_king_anchored_euler_interaction_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i051` | [King Escape Percolation Network](i051_king_escape_percolation_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i052` | [Soft King-Cage Path Bottleneck Network](i052_soft_king_cage_path_bottleneck_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i053` | [Hall-Defect Obligation Matroid Network](i053_hall_defect_obligation_matroid_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i054` | [Threat-Topology Betti Bottleneck Network](i054_threat_topology_betti_bottleneck_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i055` | [Non-Backtracking Tactical Walk Network](i055_non_backtracking_tactical_walk_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i056` | [Non-Puzzle Score-Field Bottleneck Network](i056_non_puzzle_score_field_bottleneck_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i057` | [Soft Formal-Concept Closure Network](i057_soft_formal_concept_closure_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i058` | [Determinantal Tactical Volume Bottleneck](i058_determinantal_tactical_volume_bottleneck) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i059` | [Harmonic Board Potential Network](i059_harmonic_board_potential_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i060` | [Tropical Constraint Circuit Network](i060_tropical_constraint_circuit_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i061` | [Grassmannian Principal-Angle Bottleneck](i061_grassmannian_principal_angle_bottleneck) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i062` | [Matrix-Pencil Generalized Spectrum Bottleneck](i062_matrix_pencil_generalized_spectrum_bottleneck) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i063` | [Polar-Procrustes Alignment Bottleneck](i063_polar_procrustes_alignment_bottleneck) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i064` | [Multi-Scale Dilated Board Mixer CNN](i064_multi_scale_dilated_board_mixer_cnn) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i065` | [Piece-Token CNN Hybrid](i065_piece_token_cnn_hybrid) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i066` | [Bispectral Phase-Coupling Board Network](i066_bispectral_phase_coupling_board_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i067` | [Finite-Field Character-Sum Board Network](i067_finite_field_character_sum_board_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i068` | [Schur-Ray Line Algebra Network](i068_schur_ray_line_algebra_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i069` | [Bitboard Shift-Algebra Network](i069_bitboard_shift_algebra_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i070` | [Relational Query Algebra Network](i070_relational_query_algebra_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i071` | [Variational Board Action Network](i071_variational_board_action_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i072` | [Tensor-Core Square-Pair Field Network](i072_tensor_core_square_pair_field_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i073` | [Tiny Chess MicroNet](i073_tiny_chess_micronet) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i074` | [Puzzle-Binary Benchmark Challengers](i074_puzzle_binary_benchmark_challengers) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i075` | [Tactical Bisimulation Puzzle Network](i075_tactical_bisimulation_puzzle_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i076` | [Krylov Tactical Subspace Network](i076_krylov_tactical_subspace_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i077` | [Adaptive Tactical Resolvent Network](i077_adaptive_tactical_resolvent_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i078` | [Tactical Controllability Gramian Network](i078_tactical_controllability_gramian_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i079` | [Support-Polar Zonotope Certificate Network](i079_support_polar_zonotope_certificate_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i080` | [Loop-Frustration Curvature Network](i080_loop_frustration_curvature_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i081` | [Forcing-Response Front-Door Bottleneck](i081_forcing_response_front_door_bottleneck) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i082` | [Chess Hypercut Polynomial Network](i082_chess_hypercut_polynomial_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i083` | [Fisher-Geodesic Tension Network](i083_fisher_geodesic_tension_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i084` | [Typed Hypergraph Motif Grammar](i084_typed_hypergraph_motif_grammar) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i085` | [Hall-Defect Zeta Operator](i085_hall_defect_zeta_operator) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i086` | [Differentiable Chess Fact Lattice](i086_differentiable_chess_fact_lattice) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i087` | [Tactical Radius Filtration](i087_tactical_radius_filtration) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i088` | [Traced Threat Motif Network](i088_traced_threat_motif_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i089` | [Bounded Board Hinge Logic](i089_bounded_board_hinge_logic) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i090` | [Chess-Mode Tucker Relation Certificate](i090_chess_mode_tucker_relation_certificate) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i091` | [Tactical State Bottleneck Inference](i091_tactical_state_bottleneck_inference) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i092` | [Parity-Syndrome Puzzle Bottleneck](i092_parity_syndrome_puzzle_bottleneck) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i093` | [Wavelet Scattering Board Network](i093_wavelet_scattering_board_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i094` | [Convex Feasibility Residual Network](i094_convex_feasibility_residual_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i095` | [Rank-Quantile Evidence Field Network](i095_rank_quantile_evidence_field_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i096` | [Oriented Matroid Covector Bottleneck](i096_oriented_matroid_covector_bottleneck) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i097` | [Fixed-Point Residual Defect Network](i097_fixed_point_residual_defect_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i098` | [Baseline Logit Residual Adapter](i098_baseline_logit_residual_adapter) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i099` | [Coarse-to-Fine Board Residual Pyramid](i099_coarse_to_fine_board_residual_pyramid) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i100` | [Independence Residual Interaction Network](i100_independence_residual_interaction_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i101` | [Residual Calibration Error Field](i101_residual_calibration_error_field) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i102` | [Set-Query Attention Bottleneck](i102_set_query_attention_bottleneck) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i103` | [Attention Disagreement Residual Network](i103_attention_disagreement_residual_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i104` | [Cross-Scale Attention Residual Network](i104_cross_scale_attention_residual_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i105` | [Slot Attention Role Binding Network](i105_slot_attention_role_binding_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i106` | [Attention Perturbation Sensitivity Network](i106_attention_perturbation_sensitivity_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i107` | [Kernel Mean Prototype Network](i107_kernel_mean_prototype_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i108` | [TensorSketch Interaction Network](i108_tensorsketch_interaction_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i109` | [Maxout Region Signature Network](i109_maxout_region_signature_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i110` | [Spline Board Surface Network](i110_spline_board_surface_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i111` | [Boundary-Condition Disagreement CNN](i111_boundary_condition_disagreement_cnn) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i112` | [Piece-Drop Stability Network](i112_piece_drop_stability_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i113` | [Row-File Factor Mixer](i113_row_file_factor_mixer) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i114` | [Piece-Conditioned Hypernetwork CNN](i114_piece_conditioned_hypernetwork_cnn) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i115` | [Neural Board Cellular Automaton](i115_neural_board_cellular_automaton) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i116` | [Symmetric Difference Twin Encoder](i116_symmetric_difference_twin_encoder) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i117` | [Prototype Patch Dictionary Network](i117_prototype_patch_dictionary_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i118` | [Channel Dropout Consensus Network](i118_channel_dropout_consensus_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i119` | [Tensor-Ring Square Interaction Network](i119_tensor_ring_square_interaction_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i120` | [Sinkhorn Role Assignment Network](i120_sinkhorn_role_assignment_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i121` | [Morphological Threat Field Network](i121_morphological_threat_field_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i122` | [Invertible Board Coupling Network](i122_invertible_board_coupling_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i123` | [Sparse Expert Board Router](i123_sparse_expert_board_router) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i124` | [Local Neighborhood Geometry Network](i124_local_neighborhood_geometry_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i125` | [Ray State-Space Scan Network](i125_ray_state_space_scan_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i126` | [Pawn Skeleton Barrier Network](i126_pawn_skeleton_barrier_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i127` | [Square-Color Parity Mixer](i127_square_color_parity_mixer) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i128` | [Occupancy Run-Length Segment Encoder](i128_occupancy_run_length_segment_encoder) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i129` | [King-Shelter Microkernel Network](i129_king_shelter_microkernel_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i130` | [Material-Phase Low-Rank Adapter Network](i130_material_phase_low_rank_adapter_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i131` | [Replicator Payoff Piece Dynamics](i131_replicator_payoff_piece_dynamics) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i132` | [Differentiable Bitboard Boolean Network](i132_differentiable_bitboard_boolean_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i133` | [Orthogonal Board Moment Network](i133_orthogonal_board_moment_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i134` | [Legal-Constraint Projection Residual Network](i134_legal_constraint_projection_residual_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i135` | [Zobrist Kernel Feature Network](i135_zobrist_kernel_feature_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i136` | [Low-Rank Signed Cut Query Network](i136_low_rank_signed_cut_query_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i137` | [Commutative View-Consistency Network](i137_commutative_view_consistency_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i138` | [Support-Function Envelope Network](i138_support_function_envelope_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i139` | [Soft Majorization Line Sorter](i139_soft_majorization_line_sorter) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i140` | [Low-Displacement-Rank Board Operator](i140_low_displacement_rank_board_operator) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i141` | [Submodular Coverage Bottleneck](i141_submodular_coverage_bottleneck) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i142` | [Pivot Trace Elimination Network](i142_pivot_trace_elimination_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i143` | [ConvNeXt BoardNet](i143_convnext_boardnet) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i144` | [Board FPN CNN](i144_board_fpn_cnn) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i145` | [Piece-Plane Gated CNN](i145_piece_plane_gated_cnn) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i146` | [Patch Mixer BoardNet](i146_patch_mixer_boardnet) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i147` | [Specialist-Head CNN](i147_specialist_head_cnn) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i148` | [Shallow Wide Residual BoardNet](i148_shallow_wide_residual_boardnet) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i149` | [Axial Rank-File ConvNet](i149_axial_rank_file_convnet) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i150` | [Early-Exit Cascade BoardNet](i150_early_exit_cascade_boardnet) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i151` | [Auxiliary Reconstruction BoardNet](i151_auxiliary_reconstruction_boardnet) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i152` | [Iterative Logit Refinement CNN](i152_iterative_logit_refinement_cnn) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i153` | [Agreement-Variance Head Net](i153_agreement_variance_head_net) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i154` | [Adapter-Sandwich Residual CNN](i154_adapter_sandwich_residual_cnn) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i155` | [Capsule Motif BoardNet](i155_capsule_motif_boardnet) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i156` | [Multi-Order Board Scan Network](i156_multi_order_board_scan_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i157` | [Cross-Stitch CNN-Token Fusion Net](i157_cross_stitch_cnn_token_fusion_net) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i158` | [Neural Decision Forest BoardNet](i158_neural_decision_forest_boardnet) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i159` | [Vector-Quantized Motif Codebook Net](i159_vector_quantized_motif_codebook_net) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i160` | [Hypercolumn Square Readout CNN](i160_hypercolumn_square_readout_cnn) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i161` | [Multiplicative Conjunction ConvNet](i161_multiplicative_conjunction_convnet) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i162` | [Empty-Square Opportunity Network](i162_empty_square_opportunity_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i163` | [Global Scratchpad BoardNet](i163_global_scratchpad_boardnet) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i164` | [Learnable Pooling Tree BoardNet](i164_learnable_pooling_tree_boardnet) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i165` | [Spatial FiLM Coordinate Net](i165_spatial_film_coordinate_net) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i166` | [Channel-Bilinear Role Mixer](i166_channel_bilinear_role_mixer) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i167` | [Evidence Sieve Network](i167_evidence_sieve_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i168` | [Ring-Shell Recurrent BoardNet](i168_ring_shell_recurrent_boardnet) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i169` | [Rank-File Memory Grid Net](i169_rank_file_memory_grid_net) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i170` | [Negative-Class Disentangled Puzzle Head](i170_negative_class_disentangled_puzzle_head) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i171` | [Line-Piece Crossbar Network](i171_line_piece_crossbar_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i172` | [Near-Puzzle Margin Twin Network](i172_near_puzzle_margin_twin_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i173` | [Stripe-Selective Mixer CNN](i173_stripe_selective_mixer_cnn) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i174` | [King-Zone Evidence Ledger](i174_king_zone_evidence_ledger) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i175` | [Prototype-Margin Puzzle Network](i175_prototype_margin_puzzle_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i176` | [Source-Rate Calibrated Objective](i176_source_rate_calibrated_objective) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i177` | [Forcing-Certificate Transformer](i177_forcing_certificate_transformer) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i178` | [Defender-Exhaustion Cascade Network](i178_defender_exhaustion_cascade_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i179` | [Causal Piece-Derivative Network](i179_causal_piece_derivative_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i180` | [Phase-Transition Pressure Network](i180_phase_transition_pressure_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i181` | [Disproof-Ledger Puzzle Network](i181_disproof_ledger_puzzle_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i182` | [Motif Tensor Factorization Network](i182_motif_tensor_factorization_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i183` | [Tempo-Alignment Gate Network](i183_tempo_alignment_gate_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i184` | [Puzzle Boundary Twin Encoder](i184_puzzle_boundary_twin_encoder) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i185` | [Critical-Square Budget Network](i185_critical_square_budget_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i186` | [Legal-Reaction Bottleneck Network](i186_legal_reaction_bottleneck_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i187` | [Exchange-Soundness Graph Network](i187_exchange_soundness_graph_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i188` | [Tactical Program Induction Network](i188_tactical_program_induction_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i189` | [Counterfactual Defender Dropout Network](i189_counterfactual_defender_dropout_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i190` | [Blocker-Pin Lattice Network](i190_blocker_pin_lattice_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i191` | [Safe-Reply Certificate Verifier](i191_safe_reply_certificate_verifier) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i192` | [Latent Reply Entropy Network](i192_latent_reply_entropy_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i193` | [Exchange-Then-King Dual Stream](i193_exchange_then_king_dual_stream) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i194` | [Tactical Symptom Bayesian Network](i194_tactical_symptom_bayesian_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i195` | [Minimal-Edit Puzzle Distance Network](i195_minimal_edit_puzzle_distance_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i196` | [Source-Invariant Puzzle Bottleneck](i196_source_invariant_puzzle_bottleneck) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i197` | [Reply-Set Contrastive Transformer](i197_reply_set_contrastive_transformer) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i198` | [Barrier-Cut Puzzle Network](i198_barrier_cut_puzzle_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i199` | [Tactical Hessian Spectrum Network](i199_tactical_hessian_spectrum_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i200` | [Absorbing Threat Markov Network](i200_absorbing_threat_markov_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i201` | [Neural Clause-Resolution Puzzle Network](i201_neural_clause_resolution_puzzle_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i202` | [Piece Liability Gradient Network](i202_piece_liability_gradient_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i203` | [Hierarchical Tactical Option Network](i203_hierarchical_tactical_option_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i204` | [Cross-Defense Consistency Network](i204_cross_defense_consistency_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i205` | [Defender Timing Schedule Network](i205_defender_timing_schedule_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i206` | [Discovered-Ray Switchboard Network](i206_discovered_ray_switchboard_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i207` | [Counterplay Insolvency Ledger](i207_counterplay_insolvency_ledger) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i208` | [Pinned Mobility Nullspace Network](i208_pinned_mobility_nullspace_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i209` | [Tactical Effective Resistance Network](i209_tactical_effective_resistance_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i210` | [Defender Opportunity-Cost Auction Network](i210_defender_opportunity_cost_auction_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i211` | [Role-Counterfactual Necessity Network](i211_role_counterfactual_necessity_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i212` | [Phase-Specialist Calibration Mixture](i212_phase_specialist_calibration_mixture) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i213` | [Forced-Target Funnel Network](i213_forced_target_funnel_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i214` | [Tactical Subgoal Automaton Network](i214_tactical_subgoal_automaton_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i215` | [Masked Codec Interaction-Curvature Network](i215_masked_codec_interaction_curvature_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i216` | [Non-Puzzle Score Curl-Divergence Bottleneck](i216_non_puzzle_score_curl_divergence_bottleneck) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i217` | [Ray Grammar Edit-Distance Network](i217_ray_grammar_edit_distance_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i218` | [Orbit Disagreement Residual Network](i218_orbit_disagreement_residual_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i219` | [Hall-Defect Dual-Residual Network](i219_hall_defect_dual_residual_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i220` | [Credal Temperature Field Network](i220_credal_temperature_field_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i221` | [Sylvester Tactical Coupling Network](i221_sylvester_tactical_coupling_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i222` | [Schur-Complement Defender Elimination Network](i222_schur_complement_defender_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i223` | [Bures-Wasserstein SPD Threat Manifold Network](i223_bures_wasserstein_threat_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i224` | [Numerical-Range Boundary Network](i224_numerical_range_boundary_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i225` | [Lyapunov Stability Threat Network](i225_lyapunov_threat_stability_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i226` | [Pfaffian Skew Threat Network](i226_pfaffian_skew_threat_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i227` | [p-adic Ultrametric Threat Embedding Network](i227_padic_ultrametric_threat_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i228` | [Free-Probability R-Transform Spectrum Network](i228_free_probability_r_transform_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i229` | [Williamson Symplectic-Eigenvalue Threat Network](i229_williamson_symplectic_threat_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i230` | [Magnus-BCH Operator-Coupling Series Network](i230_magnus_bch_coupling_series_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i231` | [Riccati Optimal-Defense Network](i231_riccati_optimal_defense_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i232` | [Clifford Rotor Threat Network](i232_clifford_rotor_threat_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i233` | [Tracy-Widom Level-Spacing Network](i233_tracy_widom_level_spacing_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i234` | [Lindstrom-Gessel-Viennot Path Determinant Network](i234_lindstrom_gessel_viennot_path_network) | `scaffolded` | `probe_scaffold_only` | `shared_probe_variant` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i235` | [Toda Isospectral Flow Network](i235_toda_isospectral_flow_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i236` | [Hadamard Walsh-Spectrum Network](i236_hadamard_spectrum_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i237` | [Cayley Orthogonal Map Network](i237_cayley_orthogonal_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i238` | [Stable-Rank Multiscale Network](i238_stable_rank_multiscale_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i239` | [Permanent Ryser Coupling Network](i239_permanent_ryser_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |
| `i240` | [Cayley-Hamilton Coefficient Network](i240_cayley_hamilton_coeffs_network) | `implemented` | `implemented` | `bespoke_model` | puzzle_binary classification: fine labels 0 and 1 map to non-puzzle, fine label 2 maps... |

## Research Packet Map

- Execution TODO: [TODO.md](TODO.md)
- Human catalog: [research_packets/CATALOG.md](research_packets/CATALOG.md)
- Machine catalog: [research_packets/CATALOG.jsonl](research_packets/CATALOG.jsonl)
- Import memory and family warnings: [research_packets/README.md](research_packets/README.md)

Most frequent packet tags:

| Tag | Count |
|---|---:|
| `linear-algebra` | 13 |
| `sheaf` | 8 |
| `puzzle-binary` | 6 |
| `symmetry` | 6 |
| `transport` | 5 |
| `information` | 5 |
| `topology` | 5 |
| `sparse` | 4 |
| `move-delta` | 3 |
| `graph` | 3 |
| `robustness` | 3 |
| `convex` | 3 |
| `grammar` | 2 |
| `tempo` | 2 |

## Recommended Work Loop

1. Pick one research packet or registered idea and read only its packet plus this index.
2. If the source is a packet, promote it into the next `ideas/i###_*` folder using `ideas/idea_template/`.
3. Update `ideas/registry.jsonl` only after the promoted folder has the complete scaffold.
4. Implement reusable model code in `src/chess_nn_playground/models/`; idea-local `model.py` should be a thin registered-builder wrapper only after the bespoke model exists.
5. Add a config under `configs/benchmarks/<task>/` or keep an idea-local `config.yaml`, then add a focused smoke test before training.
6. Run the benchmark suite, then update the idea folder with result links and status.

For detailed steps, see `ideas/WORKFLOW.md`.

Generated by `PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_catalog.py`.
