# Ablations

- Ablation switches:
  - `none`: full mixer (depthwise ray scans + forcing-feature MLP + gated
    additive fusion).
  - `mixer=conv` (baseline): swap to the BT4 conv-pair mixer; isolates the
    benefit of the rule-aware tactical mixer over the original BT4 block.
  - `mixer=attention` (baseline): swap to the attention mixer; isolates the
    benefit over a generic global token mixer.
  - `shuffle_tsdp`: randomly permute the forcing-feature channels across
    the batch before they enter the delta MLP -- removes the
    forcing-feature signal while keeping parameter count and base mix
    intact.
  - `disable_gate`: clamp `gate = 0`, leaving only `base_mix(x)` -- proves
    that any lift over the conv baseline must flow through the gate.
  - `zero_delta`: clamp `delta = 0`, leaving the gate intact -- proves the
    delta path (not the gate path) carries the signal.

- What each ablation tests:
  - `mixer=conv` / `mixer=attention`: the headline comparison -- does the
    rule-aware tactical mixer beat the per-block mixers used by the
    existing BT4 family on puzzle_binary, especially on tactical slices?
  - `shuffle_tsdp`: does the lift come from the forcing-feature *content*
    (the ray-geometry surrogate for check / capture / promotion) rather
    than from extra parameters?
  - `disable_gate`: confirms that the additive-gated fusion is the
    operative mechanism, not the base spatial mix.
  - `zero_delta`: dual of `disable_gate`; confirms the delta path is
    non-trivial.

- Falsification criteria:
  - Headline: aggregate puzzle_binary PR AUC must be within 0.5 percentage
    points of the conv baseline (no large regression) AND must improve the
    `crtk_tactic_motifs = mate_in_1` slice PR AUC by at least +0.02 over
    the conv baseline. If both fail, drop.
  - Mechanism: `shuffle_tsdp` must lose at least 50 percent of the
    tactical-slice lift; otherwise the lift is not driven by the
    forcing-feature content.
  - Stability: `disable_gate` must regress to the conv baseline within 0.5
    PR AUC points on the tactical slice; otherwise the gated fusion is not
    the operative mechanism.
  - Throughput: average step time must stay within 25 percent of the conv
    baseline; otherwise the mixer pays for itself only at extreme tower
    sizes and the comparison is not apples-to-apples.
