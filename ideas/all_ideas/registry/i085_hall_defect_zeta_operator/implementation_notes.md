# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/hall_defect_zeta.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i085_hall_defect_zeta_operator/model.py` (`build_model_from_config`).
- Registry key: `hall_defect_zeta_operator` in `src/chess_nn_playground/models/registry.py::MODEL_BUILDERS`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-28_0802_tuesday_new_york_hall_defect_zeta.md`.
- Input: repo `simple_18` board tensor only. Channels `0..12` are interpreted as the packet's piece planes plus a side-to-move broadcast plane; the remaining repo metadata channels are ignored by the deterministic operator.
- HDZ tensor: `R^{8x8x40}` per board, computed deterministically with pin/king-exposure-filtered defense, subset enumeration through order four, defender supports stored as 16-bit integer sets, and normalised counts.
- Algebra modes: `hdz` (default), `atom_scramble_hdz` (semantics-destroying ablation), and `neural_synth_40` (same-parameter neural control). `use_pin_filter`, `max_subset_order`, and `max_atoms` expose the remaining packet ablations.
- Output: one BCE puzzle logit plus HDZ diagnostics (`hdz_only_logits`, `mean_hall_defect`, `max_hall_defect`, `hall_defect_energy`, `effective_defense_density`, `pinned_defender_density`, `loose_target_density`, `loose_target_count`, `pinned_piece_count`, `effective_defense_total`, `proposal_profile_strength`, `mechanism_energy`).
- The deterministic algebraic branch is detached and not differentiated through. CRTK and source metadata are never consumed as model input.
