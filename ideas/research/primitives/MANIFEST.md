# Primitive Research — Manifest

Single flat directory of every primitive research report from the 2026-05-12 and 2026-05-13 sessions. Each row tells you who wrote it and what model produced it. Prototype scripts live in [prototypes/](prototypes/); architecture-level notes that combine primitives live one level up in [../architecture_bridges/](../architecture_bridges/).

## Claude Opus 4.7 primitive proposals (5)

Designed by Claude after reviewing the external research batch, the i### registry, and the audit reports. No scout-scale falsification has been performed yet.

| File | Primitive | Slug | Status | Prototype | Model |
|---|---|---|---|---|---|
| [claude_01_signed_hessian_resonance.md](claude_01_signed_hessian_resonance.md) | Signed Piece-Existence Hessian Operator | `signed_hessian_resonance` (DHPE) | proposed | [prototypes/dhpe_prototype.py](prototypes/dhpe_prototype.py), [prototypes/dhpe_v2.py](prototypes/dhpe_v2.py) | Claude Opus 4.7 |
| [claude_02_tempo_defender_cross_derivative.md](claude_02_tempo_defender_cross_derivative.md) | Tempo-Defender Cross-Derivative Operator | `tempo_defender_cross_derivative` (TDCD) | proposed | — | Claude Opus 4.7 |
| [claude_03_promotion_fanout_counterfactual.md](claude_03_promotion_fanout_counterfactual.md) | Promotion-Fanout Counterfactual Tensor | `promotion_fanout_counterfactual` (PFCT) | proposed | [prototypes/pfct_prototype.py](prototypes/pfct_prototype.py) | Claude Opus 4.7 |
| [claude_04_complex_amplitude_interference.md](claude_04_complex_amplitude_interference.md) | Complex-Amplitude Interference Operator | `complex_amplitude_interference` (CAIO) | proposed | [prototypes/caio_prototype.py](prototypes/caio_prototype.py) | Claude Opus 4.7 |
| [claude_05_terminal_state_detection.md](claude_05_terminal_state_detection.md) | Terminal-State Detection Primitive | `terminal_state_detection` (TSDP) | proposed | [prototypes/tsdp_prototype.py](prototypes/tsdp_prototype.py) | Claude Opus 4.7 |

Claude Code implementation handoff for this batch: [HANDOFF.md](HANDOFF.md).

## Codex candidate/reply primitives (5)

Produced locally in the 2026-05-12 session by the Codex GPT-5 coding agent. Research input only; promote to `ideas/registry/i###_*` before training.

| File | Primitive | Slug | Status | Model |
|---|---|---|---|---|
| [codex_01_pareto_antichain_frontier.md](codex_01_pareto_antichain_frontier.md) | Pareto Antichain Frontier Primitive | `primitive_pareto_antichain_frontier` | research packet | Codex GPT-5 |
| [codex_02_regret_saddlepoint.md](codex_02_regret_saddlepoint.md) | Regret Saddlepoint Primitive | `primitive_regret_saddlepoint` | research packet | Codex GPT-5 |
| [codex_03_reply_channel_capacity.md](codex_03_reply_channel_capacity.md) | Reply Channel Capacity Primitive | `primitive_reply_channel_capacity` | research packet | Codex GPT-5 |
| [codex_04_tail_copula_concordance.md](codex_04_tail_copula_concordance.md) | Tail Copula Concordance Primitive | `primitive_tail_copula_concordance` | research packet | Codex GPT-5 |
| [codex_05_witness_counterwitness_quantifier.md](codex_05_witness_counterwitness_quantifier.md) | Witness-Counterwitness Quantifier Primitive | `primitive_witness_counterwitness_quantifier` | research packet | Codex GPT-5 |

## External primitive imports (41)

Markdown reports downloaded from external chat services. Producer is from the download URI and filename context; the exact model column comes from the user's session note where available (Claude outputs from Opus 4.7, ChatGPT outputs from GPT-5.5 Pro; Gemini model unspecified).

| File | Original download | Producer | Exact model |
|---|---|---|---|
| [external_01_signed_edit_bilinear_memory_ray_scan.md](external_01_signed_edit_bilinear_memory_ray_scan.md) | `new_neural_primitives_for_chess_evaluation.md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_02_move_graph_router_delta_accumulator.md](external_02_move_graph_router_delta_accumulator.md) | `compass_artifact_wf-7e5d02f8-...md` | Claude | Claude Opus 4.7 |
| [external_03_attack_ray_sparse_attention_delta_accumulator.md](external_03_attack_ray_sparse_attention_delta_accumulator.md) | `chess_nn_primitives(3).md` | Claude | Claude Opus 4.7 |
| [external_04_rule_conditioned_sparse_attention_mobscan.md](external_04_rule_conditioned_sparse_attention_mobscan.md) | `compass_artifact_wf-afb1fce9-...md` | Claude | Claude Opus 4.7 |
| [external_05_legal_move_graph_delta_accumulator.md](external_05_legal_move_graph_delta_accumulator.md) | `compass_artifact_wf-1c6eb74b-...md` | Claude | Claude Opus 4.7 |
| [external_06_high_risk_legal_graph_delta_state_primitives.md](external_06_high_risk_legal_graph_delta_state_primitives.md) | `compass_artifact_wf-700f7fda-...md` | Claude | Claude Opus 4.7 |
| [external_07_sparse_delta_accumulator_segment_scatter.md](external_07_sparse_delta_accumulator_segment_scatter.md) | `compass_artifact_wf-2de56a10-...md` | Claude | Claude Opus 4.7 |
| [external_08_delta_pair_ray_selective_bispectrum.md](external_08_delta_pair_ray_selective_bispectrum.md) | `chess_nn_primitives(2).md` | Claude | Claude Opus 4.7 |
| [external_09_delta_crelu_involution_graph_message.md](external_09_delta_crelu_involution_graph_message.md) | `compass_artifact_wf-7bcc6702-...md` | Claude | Claude Opus 4.7 |
| [external_10_ray_semiring_exchange_and_chi_head.md](external_10_ray_semiring_exchange_and_chi_head.md) | `chess_nn_primitives(1).md` | Claude | Claude Opus 4.7 |
| [external_11_delta_event_legal_move_routing.md](external_11_delta_event_legal_move_routing.md) | `deep_research_primitive_results(8).md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_12_ray_occlusion_legal_dispatch_delta_pair.md](external_12_ray_occlusion_legal_dispatch_delta_pair.md) | `deep_research_primitive_results(7).md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_13_reversible_delta_kernel_occlusion_transport.md](external_13_reversible_delta_kernel_occlusion_transport.md) | `deep_research_primitive_results(6).md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_14_ray_occlusion_legal_edge_compile_scatter.md](external_14_ray_occlusion_legal_edge_compile_scatter.md) | `deep_research_primitive_results(5).md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_15_blocker_reset_edit_delta_fastweight.md](external_15_blocker_reset_edit_delta_fastweight.md) | `deep_research_primitive_results(4).md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_16_ray_blocked_delta_pair_legal_edge_reduce.md](external_16_ray_blocked_delta_pair_legal_edge_reduce.md) | `deep_research_primitive_results(3).md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_17_delta_state_slg_diffusion_fg_tp.md](external_17_delta_state_slg_diffusion_fg_tp.md) | `chess_nn_primitives.md` | Claude | Claude Opus 4.7 |
| [external_18_delta_bilinear_ray_blocked_segment_attention.md](external_18_delta_bilinear_ray_blocked_segment_attention.md) | `deep_research_primitive_results(2).md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_19_occlusion_semiring_delta_bilinear_hyperedge.md](external_19_occlusion_semiring_delta_bilinear_hyperedge.md) | `deep_research_primitive_results(1).md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_20_event_symmetric_sparse_scatter_ray_scan.md](external_20_event_symmetric_sparse_scatter_ray_scan.md) | `deep_research_primitive_results.md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_21_incremental_delta_linear_color_involution_adjacency.md](external_21_incremental_delta_linear_color_involution_adjacency.md) | `Chess_Neural_Primitives_Full_Report.md` | Google / Gemini | Unspecified Gemini model |
| [external_22_ray_cast_obstacle_pooling_sparse_emit.md](external_22_ray_cast_obstacle_pooling_sparse_emit.md) | `chess_neural_primitives(3).md` | Google / Gemini | Unspecified Gemini model |
| [external_23_sparse_legal_move_router_kinematic_state_space.md](external_23_sparse_legal_move_router_kinematic_state_space.md) | `chess_neural_primitives(2).md` | Google / Gemini | Unspecified Gemini model |
| [external_24_incremental_latent_accumulator_directional_scan.md](external_24_incremental_latent_accumulator_directional_scan.md) | `chess_neural_primitives(1).md` | Google / Gemini | Unspecified Gemini model |
| [external_25_dynamic_adjacency_rank_order_involution_gate.md](external_25_dynamic_adjacency_rank_order_involution_gate.md) | `chess_neural_primitives.md` | Google / Gemini | Unspecified Gemini model |
| [external_26_delta_update_occlusion_ray_piece_kernels.md](external_26_delta_update_occlusion_ray_piece_kernels.md) | `gemini-code-1778570861971.md` | Google / Gemini | Unspecified Gemini model |
| [external_27_ray_parallel_ssm_delta_accumulator_sparse_conv.md](external_27_ray_parallel_ssm_delta_accumulator_sparse_conv.md) | `Proposed_Neural_Primitives_Chess.md` | Google / Gemini | Unspecified Gemini model |
| [external_28_sparse_differential_accumulator_move_kernel.md](external_28_sparse_differential_accumulator_move_kernel.md) | `chess_primitives_proposal.md` | Google / Gemini | Unspecified Gemini model |
| [external_29_incremental_move_update_octilinear_scan.md](external_29_incremental_move_update_octilinear_scan.md) | `gemini-code-1778570804098.md` | Google / Gemini | Unspecified Gemini model |
| [external_30_sparse_legal_graph_transition_delta_accumulator.md](external_30_sparse_legal_graph_transition_delta_accumulator.md) | `chess_nn_primitives(4).md` | Google / Gemini | Unspecified Gemini model |
| [external_31_canonical_orbit_bdd_wmc_primitives.md](external_31_canonical_orbit_bdd_wmc_primitives.md) | `chess_nn_new_primitives.md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_32_elementary_symmetric_gibbs_hodge_primitives.md](external_32_elementary_symmetric_gibbs_hodge_primitives.md) | `chess_nn_new_primitives(1).md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_33_esp_permanent_woodbury_orbit_primitives.md](external_33_esp_permanent_woodbury_orbit_primitives.md) | `chess_nn_new_primitives(2).md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_34_active_esp_conflict_matching_eikonal_primitives.md](external_34_active_esp_conflict_matching_eikonal_primitives.md) | `chess_nn_new_primitives(3).md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_35_espa_conservation_isotypic_green_primitives.md](external_35_espa_conservation_isotypic_green_primitives.md) | `chess_nn_new_primitives(4).md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_36_exterior_product_rank1_resolvent_primitives.md](external_36_exterior_product_rank1_resolvent_primitives.md) | `chess_nn_new_primitives(5).md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_37_truncated_multiset_polynomial_rook_matching_primitives.md](external_37_truncated_multiset_polynomial_rook_matching_primitives.md) | `chess_nn_new_primitives(6).md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_38_polynomial_ledger_grassmann_rook_primitives.md](external_38_polynomial_ledger_grassmann_rook_primitives.md) | `chess_nn_new_primitives(7).md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_39_orbit_irrep_hodge_projection_primitives.md](external_39_orbit_irrep_hodge_projection_primitives.md) | `new_neural_primitives_chess_evaluation.md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_40_symmetric_coalition_resolvent_primitives.md](external_40_symmetric_coalition_resolvent_primitives.md) | `neural_primitives_chess_evaluation.md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |
| [external_41_orbit_stabilizer_subset_logpartition_primitives.md](external_41_orbit_stabilizer_subset_logpartition_primitives.md) | `chess_neural_primitives_proposals.md` | GPT / ChatGPT Deep Research | GPT-5.5 Pro |

The first batch of 21 downloads contained two `compass_artifact_wf-7e5d02f8-...` files with identical SHA-256, so this directory keeps the 20 unique files. The later Google/Gemini batch added 10 more unique reports as `external_21` through `external_30`. The 2026-05-13 GPT batch added 11 unique reports as `external_31` through `external_41`; these are raw research inputs and are not implementation-complete until promoted into `ideas/registry/p###_*`.

## Cross-references

- Session ledger: [SESSION_LEDGER.md](SESSION_LEDGER.md)
- Training and promotion plan: [PRIMITIVE_TRAINING_TODO.md](PRIMITIVE_TRAINING_TODO.md)
- Architecture-level bridges that compose these primitives: [../architecture_bridges/](../architecture_bridges/)
