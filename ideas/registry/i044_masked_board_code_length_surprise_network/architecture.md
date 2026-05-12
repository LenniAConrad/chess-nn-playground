# Architecture

`Masked Board Code-Length Surprise Network` (`MBCS-Net`) is a board-only
`puzzle_binary` classifier whose central operator is the mask-averaged
conditional code-length field produced by a label-free masked board
codec, exactly as specified in the markdown math thesis. The shared
`ResearchPacketProbe` mechanism profile has been replaced by a
materially distinct bespoke implementation.

## Forward Pipeline

1. **Deterministic piece-token tokenizer (`Simple18PieceTokenizer`).**
   The 12 piece planes of the `simple_18` board are mapped to a
   `(B, 8, 8)` int64 token map with vocabulary
   `{empty=0, WP=1, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK=12}`. A
   strict mode raises if more than one piece plane is active on a
   square; a non-strict mode falls back to argmax. Side-to-move,
   castling, and en-passant planes are kept as visible context but are
   never reconstructed. LC0 encodings fail closed.
2. **Fixed mask bank (`MaskBank2x2Residues`).** Four 2x2 residue masks
   cover every square exactly once, giving a finite mask bank
   `M_1..M_4` with constant 16-square coverage per mask. The bank is
   stored as a non-persistent buffer so the experiment is deterministic.
3. **Masked board codec (`MaskedBoardCodec`).** A compact 3-block
   convolutional codec consumes `concat(x_masked, mask_plane)` of shape
   `(J*B, C+1, 8, 8)` -- the 12 piece planes are zeroed under the mask
   while the side-to-move/castling/en-passant context planes are kept.
   The codec emits 13-way piece-token logits at every square. Mask
   chunks of size `mask_chunk_size` keep peak memory bounded.
4. **Code-length field builder.** For every chunk, log-softmax of the
   token logits gives `q(token | x_{\setminus M}, M, square)`. The
   per-square cross-entropy `\ell(s,x,M) = -\log q(T(x)_s | ...)` and
   the predictive entropy and true-token probability are scattered into
   `(B, 1, 8, 8)` accumulators only for masked squares. After all four
   masks the mask-averaged fields are
   - `S(s, x)`: mean code length in nats,
   - `H(s, x)`: predictive entropy,
   - `P_true(s, x)`: probability of the actual token.
   `S` is clipped to `surprise_clip_nats=8.0` and rescaled with
   `log1p(.)` for numerical stability before the classifier.
5. **Coordinate planes.** Optional deterministic planes are appended:
   normalized rank, normalized file, center distance, and a
   side-to-move-relative promotion direction. Coordinates depend only
   on the board grid and the side-to-move plane.
6. **Surprise residual classifier (`SurpriseResidualClassifier`).**
   `concat(x, S_scaled, H, P_true, coords)` is fed to a
   `Conv3x3(C+3+coord -> 64) -> 4 x ResidualBlock(width=64)` trunk
   followed by global average + global max pooling and a `LayerNorm
   + Dropout + Linear` head. The classifier returns a single puzzle
   logit when `num_classes=1` (for the `puzzle_binary` BCE-with-logits
   trainer) or `(B, num_classes)` otherwise.

## Output Contract

`forward(x)` returns a `dict` with `"logits"` of shape `(B,)` for
`num_classes=1` plus diagnostic tensors used by the trainer and report
artifacts:

- `code_length_field`, `code_length_scaled_field`, `entropy_field`,
  `p_true_field`: spatial `(B, 8, 8)` maps of `S`, `log1p(clamp(S))`,
  `H`, and `P_true`.
- `code_length_mean`, `code_length_max`, `entropy_mean`, `entropy_max`,
  `p_true_mean`: scalar pooled summaries per sample.
- `codec_nll`: average codec cross-entropy across the mask bank, used
  as the label-free codec objective (Section 6 of the math thesis).
- `mask_coverage`: per-square coverage check (constant 1 for the 2x2
  residue bank).

## Codec Freezing And Surprise Detachment

The default config sets `freeze_codec: true` and `detach_surprise:
true` per the markdown two-stage protocol. When `freeze_codec=True`
the codec parameters are excluded from `nn.Module.parameters()`
(via `requires_grad_(False)`) and the codec runs under `no_grad`, so
the classifier sees a frozen masked codec. When `detach_surprise=True`
the `S`, `H`, `P_true` planes are detached before concatenation so the
classifier cannot push gradients into the codec. Setting either flag
to `False` enables joint fine-tuning, which the markdown explicitly
allows only as an ablation. A pretrained codec checkpoint can be
loaded from `codec_checkpoint` by external scripts using
`MaskedBoardCodec.load_state_dict`.

## Implementation Binding

- Registered model name: `masked_board_code_length_surprise_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/masked_surprise_codec.py`
- Idea-local wrapper: `ideas/registry/i044_masked_board_code_length_surprise_network/model.py`
