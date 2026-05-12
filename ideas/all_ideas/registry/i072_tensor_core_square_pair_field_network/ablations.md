# Ablations

Set `model.ablation` to one of:

- `cnn_only_matched`: replace pair-field blocks with a matched-width board CNN control and zero pair summaries.
- `no_pair_update`: compute pair fields and pair summaries, but do not update square tokens through pair messages.
- `no_pair_readout`: keep pair-field messages, but remove pair-energy summaries from the classifier.
- `relation_bank_shuffle`: use deterministic shuffled relation masks preserving density, diagonal count, and symmetry class.
- `softmax_attention_control`: replace tanh-normalized pair weights with ordinary row softmax.
- `low_head_count`: restrict the pair update and summaries to the first two heads.
- `pair_energy_only`: classify from pair-energy summaries while zeroing square-token readout.

Central comparisons are `main` versus `cnn_only_matched`, `relation_bank_shuffle`,
`no_pair_update`, `no_pair_readout`, and `softmax_attention_control`.
