# Chess-Decomposed Attention Network Math Thesis

The core thesis is that a small puzzle classifier can benefit from three explicit chess decomposition axes before fusion:

- exchange-local square interactions;
- king-zone and checking geometry;
- global square-token attention for long-range relations.

The model computes per-stream embeddings and logits, then learns a soft phase router over the stream logits plus a residual joint head. The falsifiable claim is that this decomposition should beat either a pure global-attention trunk or the exchange/king dual-stream trunk alone at matched training budget.
