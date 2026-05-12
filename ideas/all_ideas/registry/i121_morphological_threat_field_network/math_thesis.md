# Math Thesis

Morphological Threat Field Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md`.

Batch candidate rank: `3`.

Working thesis: CNNs learn filters, but chess tactics often have shape
operations: expand a king danger zone, close gaps in a pawn shield, erode
escape squares, and detect thin corridors. Differentiable mathematical
morphology gives an architecture that explicitly processes those shape
operations. We treat per-square scalar fields as threat surfaces and apply
learned structuring elements through:

- soft dilation `y(p) = (1/τ) log Σ_s exp(τ (x(p+s) + w⁺(s)))` — a
  differentiable approximation of `max_s (x(p+s) + w⁺(s))`, the canonical
  morphological dilation rule.
- soft erosion `y(p) = -(1/τ) log Σ_s exp(-τ (x(p+s) - w⁻(s)))` — the dual
  approximation of `min_s (x(p+s) - w⁻(s))`.
- opening (`erode → dilate`) — preserves regions large enough to contain the
  structuring element while erasing thin spurs and isolated pixels.
- closing (`dilate → erode`) — fills small gaps in shape interiors.
- morphological gradient (`dilation − erosion`) — concentrates energy on
  shape boundaries, which is where thin corridors and breakthrough lanes live.
- top-hat (`field − opening`) and bottom-hat (`closing − field`) — extract
  bright peaks above the threat baseline and dark gaps below it, in line with
  the king-shield and escape-square motifs the thesis calls out.

The temperature `τ` controls how sharply the soft-min/soft-max approximate the
discrete chess shape operations: in the `τ → ∞` limit we recover classical
mathematical morphology over the chosen structuring element, while finite `τ`
keeps the operations differentiable so the structuring elements `w⁺`, `w⁻` can
be trained end-to-end.
