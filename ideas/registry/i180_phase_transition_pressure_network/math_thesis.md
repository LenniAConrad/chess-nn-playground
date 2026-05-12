# Math Thesis

Phase-Transition Pressure Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.

Batch candidate rank: `4`.

Working thesis: The key difference between a true puzzle and a
near-puzzle may be criticality. The board may sit near a threshold
where small increases in pressure, line opening, or defender loss
cause a tactical collapse. Instead of measuring pressure magnitude
the model measures pressure *phase transitions* across a sweep of
learned thresholds.

Formalisation. For each square the trunk emits five learned pressure
fields (`attack`, `defense`, `escape`, `line_block`, `target_value`).
Given a learnable threshold grid `tau_1, ..., tau_T` and learnable
temperature `T`, the model computes

```
field_tau_{f, t}(s) = sigmoid((pressure_f(s) - tau_t) / T)
```

For each `(f, t)` it then computes seven differentiable summaries on
the 8x8 board: mass, king-zone mass, a soft largest-component proxy,
boundary length, and pressure surplus around the king, queen, and
rook (own + opp combined). The readout uses the *first differences*
of those summaries across thresholds — the packet's `critical_curve`
— concatenated with the operating-point summary, fed to a small MLP.
A position that lies near a tactical phase transition will have
sharp first differences across thresholds; a stably high or stably
low pressure profile will have a flat curve.

Implementation binding. The bespoke implementation lives at
`src/chess_nn_playground/models/trunk/phase_transition_pressure_network.py`
and is registered as the model named
`phase_transition_pressure_network`. The idea-local
`ideas/registry/i180_phase_transition_pressure_network/model.py` is a thin
`build_model_from_config(config)` wrapper around the registered
builder; it does not import or call the shared
`ResearchPacketProbe` scaffold.
