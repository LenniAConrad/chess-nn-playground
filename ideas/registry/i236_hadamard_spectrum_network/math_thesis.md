# Math Thesis

Hadamard Walsh-Spectrum Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-05-05_1700_tuesday_local_hadamard_walsh_spectrum.md`.

Working thesis: Applies the 64x64 Walsh-Hadamard transform (Sylvester construction) to per-channel pooled square signals; classifies puzzle-likeness from top-k Walsh coefficient energies. Uses the boolean Fourier basis on Z_2^6 -- distinct from DCT, wavelets, and spectral-Laplacian features.

This idea is **implemented as a bespoke torch module** at
`src/chess_nn_playground/models/trunk/hadamard_spectrum.py`
(class `HadamardSpectrumNetwork`, builder `build_hadamard_spectrum_from_config`); not routed
through the generic ResearchPacketProbe.
