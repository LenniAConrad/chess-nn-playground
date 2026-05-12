# Architecture

`Blocker-Pin Lattice Network` implements the packet thesis that line tactics
depend on ordered blockers and pin constraints, not only on whether two pieces
share a rank, file, or diagonal. It is board-only for the `puzzle_binary`
contract and returns one BCE puzzle logit plus ray-lattice diagnostics.

## Mechanism

The forward path follows the proposal:

1. **Board and ray encoding.** A compact convolutional trunk encodes the
   simple_18 current board. Fixed geometry enumerates all 512 directed slider
   rays and gathers ordered ray-square tokens from the trunk.
2. **Ordered blocker sequence.** For each side-to-move bishop, rook, or queen
   ray, deterministic facts identify the first, second, and third occupied
   squares, blocker ownership, blocker value, and high-value or king targets
   behind those blockers.
3. **Four lattice states.** Each ray receives the packet states
   `state_0`, `state_remove_first`, `state_remove_second`, and
   `state_swap_side`. The remove states delete the corresponding ordered
   blocker from the current-board ray facts; the swap state flips ownership
   signs as a role-swap diagnostic.
4. **State-space scan.** A gated GRU-style lattice scanner runs along each
   ordered blocker/target sequence:
   `lattice_state_{i+1} = gated_update(lattice_state_i, blocker_token_i,
   target_context)`.
5. **Ray pooling and readout.** Learned state scores are masked to active
   slider rays, pooled with board features and deterministic lattice summaries,
   and passed to the puzzle classifier.

The model does not use engine scores, legal continuation search, solution
moves, theme labels, CRTK/source labels, verification metadata, principal
variations, or node counts.

## Diagnostics

The output dictionary includes `pin_strength`, `discovered_attack_potential`,
`blocked_tactic_residual`, `lattice_energy`, `pin_lattice_entropy`,
`ray_count`, `ordered_blocker_mass`, and per-state strengths for current,
remove-first, remove-second, and swap-side lattice states.

## Ablations

The `ablation` config switch supports the packet falsifiers:
`unordered_blockers`, `no_remove_states`, and `only_rank_file`.

## Implementation Binding

- Registered model name: `blocker_pin_lattice_network`.
- Source implementation file: `src/chess_nn_playground/models/trunk/blocker_pin_lattice.py`.
- Idea-local wrapper: `ideas/registry/i190_blocker_pin_lattice_network/model.py`.
