# Implemented Architecture Conformance Audit

This report is generated for idea folders whose `implementation_status` is `implemented` or `tested`.
It checks that those rows are bespoke implementations, have an architecture document tied to the registered model/source file, resolve to registered model code, and contain no obvious shell markers such as `TODO`, `FIXME`, `placeholder`, `stub`, `NotImplemented`, or bare `pass` statements.

It does not certify the 218 `shared_probe_variant` folders as implemented architectures; those remain scaffolded until their markdown proposals receive bespoke model code.

## Summary

- Implemented architecture rows audited: `22`
- Validation issues: `0`

| ID | Folder | Model name | Implementation kind | Status | Markdown binding | Source files | Issues |
|---|---|---|---|---|---|---|---|
| `i001` | `ideas/i001_chess_operator_basis_classifier` | `chess_operator_basis_classifier` | `bespoke_model` | `implemented` | section+model+source+wrapper | `ideas/i001_chess_operator_basis_classifier/model.py`<br>`src/chess_nn_playground/models/research_architectures.py` | - |
| `i002` | `ideas/i002_response_minimax_classifier` | `response_minimax_classifier` | `bespoke_model` | `implemented` | section+model+source+wrapper | `ideas/i002_response_minimax_classifier/model.py`<br>`src/chess_nn_playground/models/research_architectures.py` | - |
| `i003` | `ideas/i003_factor_agreement_classifier` | `factor_agreement_classifier` | `bespoke_model` | `implemented` | section+model+source+wrapper | `ideas/i003_factor_agreement_classifier/model.py`<br>`src/chess_nn_playground/models/research_architectures.py` | - |
| `i004` | `ideas/i004_puzzle_obligation_flow_network` | `puzzle_obligation_flow_network` | `bespoke_model` | `implemented` | section+model+source+wrapper | `ideas/i004_puzzle_obligation_flow_network/model.py`<br>`src/chess_nn_playground/models/research_architectures.py` | - |
| `i005` | `ideas/i005_null_move_contrast_puzzle_network` | `null_move_contrast_puzzle_network` | `bespoke_model` | `tested` | section+model+source+wrapper | `ideas/i005_null_move_contrast_puzzle_network/model.py`<br>`src/chess_nn_playground/models/research_architectures.py` | - |
| `i006` | `ideas/i006_proof_core_set_verifier` | `proof_core_set_verifier` | `bespoke_model` | `implemented` | section+model+source+wrapper | `ideas/i006_proof_core_set_verifier/model.py`<br>`src/chess_nn_playground/models/research_architectures.py` | - |
| `i007` | `ideas/i007_neural_proof_number_search` | `neural_proof_number_search` | `bespoke_model` | `tested` | section+model+source+wrapper | `ideas/i007_neural_proof_number_search/model.py`<br>`src/chess_nn_playground/models/research_architectures.py` | - |
| `i008` | `ideas/i008_boundary_edit_lagrangian_network` | `boundary_edit_lagrangian_network` | `bespoke_model` | `tested` | section+model+source+wrapper | `ideas/i008_boundary_edit_lagrangian_network/model.py`<br>`src/chess_nn_playground/models/research_architectures.py` | - |
| `i009` | `ideas/i009_tactical_equilibrium_network` | `tactical_equilibrium_network` | `bespoke_model` | `tested` | section+model+source+wrapper | `ideas/i009_tactical_equilibrium_network/model.py`<br>`src/chess_nn_playground/models/research_architectures.py` | - |
| `i010` | `ideas/i010_rule_consistent_latent_dynamics` | `rule_consistent_latent_dynamics` | `bespoke_model` | `implemented` | section+model+source+wrapper | `ideas/i010_rule_consistent_latent_dynamics/model.py`<br>`src/chess_nn_playground/models/research_architectures.py` | - |
| `i011` | `ideas/i011_vetoselect_positive_claim_abstention` | `vetoselect_positive_claim_abstention` | `bespoke_model` | `tested` | section+model+source+wrapper | `ideas/i011_vetoselect_positive_claim_abstention/model.py`<br>`src/chess_nn_playground/models/vetoselect.py` | - |
| `i012` | `ideas/i012_dykstra_lcp` | `dykstra_lcp` | `bespoke_model` | `tested` | section+model+source+wrapper | `ideas/i012_dykstra_lcp/model.py`<br>`src/chess_nn_playground/models/dykstra_lcp.py` | - |
| `i013` | `ideas/i013_sparse_relation_pursuit_asymmetry` | `sparse_relation_pursuit_asymmetry` | `bespoke_model` | `tested` | section+model+source+wrapper | `ideas/i013_sparse_relation_pursuit_asymmetry/model.py`<br>`src/chess_nn_playground/models/sparse_relation_pursuit.py` | - |
| `i014` | `ideas/i014_contamination_dro_huber_tail_rejection` | `contamination_dro_huber_tail_rejection` | `bespoke_model` | `implemented` | section+model+source+wrapper | `ideas/i014_contamination_dro_huber_tail_rejection/model.py`<br>`src/chess_nn_playground/models/gpt_research_architectures.py` | - |
| `i015` | `ideas/i015_material_locked_tactical_dro` | `material_locked_tactical_dro` | `bespoke_model` | `implemented` | section+model+source+wrapper | `ideas/i015_material_locked_tactical_dro/model.py`<br>`src/chess_nn_playground/models/gpt_research_architectures.py` | - |
| `i016` | `ideas/i016_soft_sorting_order_residual_ranker` | `soft_sorting_order_residual_ranker` | `bespoke_model` | `implemented` | section+model+source+wrapper | `ideas/i016_soft_sorting_order_residual_ranker/model.py`<br>`src/chess_nn_playground/models/gpt_research_architectures.py` | - |
| `i017` | `ideas/i017_conditional_surprisal_gate` | `conditional_surprisal_gate` | `bespoke_model` | `implemented` | section+model+source+wrapper | `ideas/i017_conditional_surprisal_gate/model.py`<br>`src/chess_nn_playground/models/gpt_research_architectures.py` | - |
| `i236` | `ideas/i236_hadamard_spectrum_network` | `hadamard_spectrum_network` | `bespoke_model` | `implemented` | section+model+source+wrapper | `ideas/i236_hadamard_spectrum_network/model.py`<br>`src/chess_nn_playground/models/hadamard_spectrum.py` | - |
| `i237` | `ideas/i237_cayley_orthogonal_network` | `cayley_orthogonal_network` | `bespoke_model` | `implemented` | section+model+source+wrapper | `ideas/i237_cayley_orthogonal_network/model.py`<br>`src/chess_nn_playground/models/cayley_orthogonal.py` | - |
| `i238` | `ideas/i238_stable_rank_multiscale_network` | `stable_rank_multiscale_network` | `bespoke_model` | `implemented` | section+model+source+wrapper | `ideas/i238_stable_rank_multiscale_network/model.py`<br>`src/chess_nn_playground/models/stable_rank_multiscale.py` | - |
| `i239` | `ideas/i239_permanent_ryser_network` | `permanent_ryser_network` | `bespoke_model` | `implemented` | section+model+source+wrapper | `ideas/i239_permanent_ryser_network/model.py`<br>`src/chess_nn_playground/models/permanent_ryser.py` | - |
| `i240` | `ideas/i240_cayley_hamilton_coeffs_network` | `cayley_hamilton_coeffs_network` | `bespoke_model` | `implemented` | section+model+source+wrapper | `ideas/i240_cayley_hamilton_coeffs_network/model.py`<br>`src/chess_nn_playground/models/cayley_hamilton_coeffs.py` | - |

Validation command:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/audit_architecture_conformance.py --check
```
