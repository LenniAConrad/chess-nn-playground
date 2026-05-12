"""Masked Board Code-Length Surprise Network for idea i044.

Implements the ``MBCS-Net`` architecture from the markdown packet. The
core observable is a mask-averaged conditional code-length field
produced by a label-free masked board codec. The supervised classifier
sees the original board tensor concatenated with the spatial code-length
``S``, predictive entropy ``H``, true-token probability ``Ptrue`` and
deterministic coordinate planes. The implementation is materially
distinct from the shared ``ResearchPacketProbe`` mechanism profile.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models._packet_bespoke_base import (
    BoardTensorSpec,
    format_logits,
    rank_file_grid,
    require_board_tensor,
)


PIECE_TOKEN_VOCAB = 13  # 0=empty, 1..12=piece planes


class Simple18PieceTokenizer(nn.Module):
    """Deterministic ``simple_18`` -> piece-token map."""

    def __init__(self, strict: bool = True) -> None:
        super().__init__()
        self.strict = bool(strict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < 12:
            raise ValueError("Simple18PieceTokenizer expects at least 12 piece planes")
        piece_planes = x[:, :12].clamp(0.0, 1.0) > 0.5
        active_count = piece_planes.sum(dim=1)
        if self.strict and torch.any(active_count > 1):
            raise ValueError(
                "simple_18 token extraction found more than one active piece plane on a square; "
                "set strict_tokenizer=False to fall back to argmax"
            )
        # piece index in {0..11}; argmax returns 0 when all planes are inactive
        argmax = piece_planes.float().argmax(dim=1)
        empty_mask = active_count == 0
        tokens = torch.where(empty_mask, torch.zeros_like(argmax), argmax + 1)
        return tokens.long()


class MaskBank2x2Residues(nn.Module):
    """Fixed mask bank with four 2x2 residue masks covering every square once."""

    def __init__(self) -> None:
        super().__init__()
        masks = torch.zeros(4, 1, 8, 8)
        for k, (r_k, f_k) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            for r in range(8):
                for f in range(8):
                    if r % 2 == r_k and f % 2 == f_k:
                        masks[k, 0, r, f] = 1.0
        self.register_buffer("masks", masks, persistent=False)

    @property
    def num_masks(self) -> int:
        return int(self.masks.shape[0])

    def get_masks(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.masks.to(device=device, dtype=dtype)


class _CodecResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(h + residual)


class MaskedBoardCodec(nn.Module):
    """Small convolutional codec ``q(piece_token | visible board, mask, square)``."""

    def __init__(self, input_channels: int, codec_width: int, codec_blocks: int) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.codec_width = int(codec_width)
        self.codec_blocks = int(codec_blocks)
        # +1 for the mask indicator plane appended to the codec input.
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels + 1, codec_width, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, codec_width), codec_width),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [_CodecResidualBlock(codec_width) for _ in range(self.codec_blocks)]
        )
        self.token_decoder = nn.Conv2d(codec_width, PIECE_TOKEN_VOCAB, kernel_size=1)

    def forward(self, x_masked: torch.Tensor, mask_plane: torch.Tensor) -> torch.Tensor:
        codec_input = torch.cat([x_masked, mask_plane], dim=1)
        h = self.stem(codec_input)
        for block in self.blocks:
            h = block(h)
        return self.token_decoder(h)


class _ClassifierResidualBlock(nn.Module):
    def __init__(self, channels: int, use_batchnorm: bool) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm1 = (
            nn.BatchNorm2d(channels) if use_batchnorm else nn.GroupNorm(min(8, channels), channels)
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm2 = (
            nn.BatchNorm2d(channels) if use_batchnorm else nn.GroupNorm(min(8, channels), channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = F.relu(self.norm1(self.conv1(x)), inplace=True)
        h = self.norm2(self.conv2(h))
        return F.relu(h + residual, inplace=True)


class SurpriseResidualClassifier(nn.Module):
    """Compact residual CNN consuming ``concat(x, S, H, Ptrue, coords)``."""

    def __init__(
        self,
        in_channels: int,
        classifier_width: int,
        classifier_blocks: int,
        num_classes: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.classifier_width = int(classifier_width)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, classifier_width, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(classifier_width) if use_batchnorm else nn.GroupNorm(min(8, classifier_width), classifier_width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [_ClassifierResidualBlock(classifier_width, use_batchnorm) for _ in range(classifier_blocks)]
        )
        head_in = 2 * classifier_width
        self.norm = nn.LayerNorm(head_in)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(head_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        gap = h.mean(dim=(2, 3))
        gmp = h.amax(dim=(2, 3))
        feats = torch.cat([gap, gmp], dim=-1)
        return self.head(self.dropout(self.norm(feats)))


class MaskedBoardCodeLengthSurpriseNet(nn.Module):
    """End-to-end MBCS-Net producing a single puzzle logit plus diagnostics."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding: str = "simple_18",
        codec_width: int = 32,
        codec_blocks: int = 3,
        classifier_width: int = 64,
        classifier_blocks: int = 4,
        mask_chunk_size: int = 2,
        surprise_clip_nats: float = 8.0,
        append_coord_planes: bool = True,
        freeze_codec: bool = True,
        detach_surprise: bool = True,
        strict_tokenizer: bool = True,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if encoding != "simple_18":
            raise ValueError(
                "MaskedBoardCodeLengthSurpriseNet currently only supports encoding='simple_18'; "
                "LC0 schemas must fail closed until an explicit current-board piece-channel map exists"
            )
        if input_channels < 13:
            raise ValueError("simple_18 requires at least 13 channels (12 piece planes + side-to-move)")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.encoding = encoding
        self.mask_chunk_size = max(1, int(mask_chunk_size))
        self.surprise_clip_nats = float(surprise_clip_nats)
        self.append_coord_planes = bool(append_coord_planes)
        self.freeze_codec = bool(freeze_codec)
        self.detach_surprise = bool(detach_surprise)

        self.tokenizer = Simple18PieceTokenizer(strict=strict_tokenizer)
        self.mask_bank = MaskBank2x2Residues()
        self.codec = MaskedBoardCodec(input_channels, codec_width, codec_blocks)

        self.num_coord_planes = 4 if self.append_coord_planes else 0
        # 3 surprise planes (S, H, Ptrue) + coord planes appended to original input.
        classifier_in_channels = input_channels + 3 + self.num_coord_planes
        self.classifier = SurpriseResidualClassifier(
            in_channels=classifier_in_channels,
            classifier_width=classifier_width,
            classifier_blocks=classifier_blocks,
            num_classes=num_classes,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

        if self.freeze_codec:
            for param in self.codec.parameters():
                param.requires_grad_(False)

    def _zero_piece_planes_under_mask(
        self, x_rep: torch.Tensor, mask_plane: torch.Tensor
    ) -> torch.Tensor:
        keep = 1.0 - mask_plane  # (J*B, 1, 8, 8)
        x_masked = x_rep.clone()
        # Zero only the 12 piece planes; keep side-to-move/castling/ep context intact.
        x_masked[:, :12] = x_masked[:, :12] * keep
        return x_masked

    def _coord_planes(self, batch: int, device: torch.device, dtype: torch.dtype, side_to_move: torch.Tensor) -> torch.Tensor:
        rank, file = rank_file_grid(batch, device, dtype)
        # Center distance: euclidean distance from board center, normalized.
        cx, cy = 3.5, 3.5
        rank_idx = torch.arange(8, device=device, dtype=dtype).view(1, 1, 8, 1).expand(batch, 1, 8, 8)
        file_idx = torch.arange(8, device=device, dtype=dtype).view(1, 1, 1, 8).expand(batch, 1, 8, 8)
        center_dist = torch.sqrt((rank_idx - cx) ** 2 + (file_idx - cy) ** 2) / (2.0 * (3.5 ** 2)) ** 0.5
        # Promotion direction relative to side-to-move (white=+1, black=-1).
        side_signed = (2.0 * side_to_move - 1.0).view(batch, 1, 1, 1).expand(batch, 1, 8, 8)
        promo_dir = side_signed * (1.0 - rank / 7.0 * 2.0)  # +1 at far rank for white, -1 for black
        return torch.cat([rank, file, center_dist, promo_dir], dim=1)

    def _run_codec(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the codec across the full mask bank and accumulate S, H, Ptrue."""

        device = x.device
        dtype = x.dtype
        batch = x.shape[0]
        masks = self.mask_bank.get_masks(device=device, dtype=dtype)  # (K, 1, 8, 8)
        K = masks.shape[0]

        tokens = self.tokenizer(x)  # (B, 8, 8) int64
        tokens_full = tokens.unsqueeze(1).float()  # for diagnostics only

        S = x.new_zeros(batch, 1, 8, 8)
        H = x.new_zeros(batch, 1, 8, 8)
        Ptrue = x.new_zeros(batch, 1, 8, 8)
        coverage = x.new_zeros(batch, 1, 8, 8)
        codec_nll_total = x.new_zeros(batch)
        codec_nll_count = x.new_zeros(batch)

        chunk = max(1, min(self.mask_chunk_size, K))
        for start in range(0, K, chunk):
            end = min(start + chunk, K)
            mask_chunk = masks[start:end]  # (J, 1, 8, 8)
            J = mask_chunk.shape[0]

            # Tile inputs across the mask chunk: (J*B, ...).
            x_rep = x.unsqueeze(0).expand(J, -1, -1, -1, -1).reshape(J * batch, *x.shape[1:])
            mask_plane = (
                mask_chunk.unsqueeze(1)
                .expand(J, batch, 1, 8, 8)
                .reshape(J * batch, 1, 8, 8)
            )
            x_masked = self._zero_piece_planes_under_mask(x_rep, mask_plane)
            token_logits = self.codec(x_masked, mask_plane)  # (J*B, 13, 8, 8)

            log_probs = F.log_softmax(token_logits, dim=1)
            probs = log_probs.exp()
            entropy_full = -(probs * log_probs).sum(dim=1, keepdim=True)  # (J*B, 1, 8, 8)

            tokens_rep = (
                tokens.unsqueeze(0).expand(J, -1, -1, -1).reshape(J * batch, 8, 8)
            )
            true_log_prob = log_probs.gather(1, tokens_rep.unsqueeze(1)).squeeze(1)  # (J*B, 8, 8)
            ce_full = (-true_log_prob).unsqueeze(1)  # (J*B, 1, 8, 8)
            ptrue_full = true_log_prob.exp().unsqueeze(1)

            mask_per_sample = mask_plane  # (J*B, 1, 8, 8)
            ce_full = ce_full * mask_per_sample
            entropy_full = entropy_full * mask_per_sample
            ptrue_full = ptrue_full * mask_per_sample

            # Reshape and sum across the J dimension.
            ce_chunked = ce_full.view(J, batch, 1, 8, 8).sum(dim=0)
            ent_chunked = entropy_full.view(J, batch, 1, 8, 8).sum(dim=0)
            ptrue_chunked = ptrue_full.view(J, batch, 1, 8, 8).sum(dim=0)
            cov_chunked = mask_per_sample.view(J, batch, 1, 8, 8).sum(dim=0)

            S = S + ce_chunked
            H = H + ent_chunked
            Ptrue = Ptrue + ptrue_chunked
            coverage = coverage + cov_chunked

            ce_per_square = (-true_log_prob) * mask_per_sample.squeeze(1)
            codec_nll_total = codec_nll_total + ce_per_square.view(J, batch, -1).sum(dim=(0, 2))
            codec_nll_count = codec_nll_count + mask_per_sample.squeeze(1).view(J, batch, -1).sum(dim=(0, 2))

        coverage = coverage.clamp_min(1.0)
        S = S / coverage
        H = H / coverage
        Ptrue = Ptrue / coverage

        S_clipped = S.clamp(0.0, self.surprise_clip_nats)
        S_scaled = torch.log1p(S_clipped)

        codec_nll = codec_nll_total / codec_nll_count.clamp_min(1.0)

        return {
            "S": S,
            "S_scaled": S_scaled,
            "H": H,
            "Ptrue": Ptrue,
            "tokens": tokens_full,
            "codec_nll": codec_nll,
            "coverage": coverage,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)

        # Codec forward (under no_grad if frozen, otherwise normal autograd).
        if self.freeze_codec and not self.training:
            with torch.no_grad():
                codec_out = self._run_codec(x)
        elif self.freeze_codec:
            with torch.no_grad():
                codec_out = self._run_codec(x)
        else:
            codec_out = self._run_codec(x)

        S_scaled = codec_out["S_scaled"]
        H = codec_out["H"]
        Ptrue = codec_out["Ptrue"]
        if self.detach_surprise:
            S_scaled = S_scaled.detach()
            H = H.detach()
            Ptrue = Ptrue.detach()

        side_to_move = x[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        coord = (
            self._coord_planes(x.shape[0], x.device, x.dtype, side_to_move)
            if self.append_coord_planes
            else x.new_zeros(x.shape[0], 0, 8, 8)
        )

        x_aug = torch.cat([x, S_scaled, H, Ptrue, coord], dim=1)
        logits = self.classifier(x_aug)

        # Spatial diagnostics aggregated to scalars for the trainer/leaderboard.
        S_mean = codec_out["S"].mean(dim=(1, 2, 3))
        H_mean = codec_out["H"].mean(dim=(1, 2, 3))
        Ptrue_mean = codec_out["Ptrue"].mean(dim=(1, 2, 3))
        S_max = codec_out["S"].amax(dim=(1, 2, 3))
        H_max = codec_out["H"].amax(dim=(1, 2, 3))

        return {
            "logits": format_logits(logits, self.num_classes),
            "code_length_field": codec_out["S"].squeeze(1),
            "code_length_scaled_field": S_scaled.squeeze(1),
            "entropy_field": codec_out["H"].squeeze(1),
            "p_true_field": codec_out["Ptrue"].squeeze(1),
            "code_length_mean": S_mean,
            "code_length_max": S_max,
            "entropy_mean": H_mean,
            "entropy_max": H_max,
            "p_true_mean": Ptrue_mean,
            "codec_nll": codec_out["codec_nll"],
            "mask_coverage": codec_out["coverage"].squeeze(1),
        }


def build_masked_board_code_length_surprise_network_from_config(
    config: dict[str, Any],
) -> MaskedBoardCodeLengthSurpriseNet:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return MaskedBoardCodeLengthSurpriseNet(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        encoding=str(cfg.get("encoding", "simple_18")),
        codec_width=int(cfg.get("codec_width", 32)),
        codec_blocks=int(cfg.get("codec_blocks", 3)),
        classifier_width=int(cfg.get("classifier_width", 64)),
        classifier_blocks=int(cfg.get("classifier_blocks", 4)),
        mask_chunk_size=int(cfg.get("mask_chunk_size", 2)),
        surprise_clip_nats=float(cfg.get("surprise_clip_nats", 8.0)),
        append_coord_planes=bool(cfg.get("append_coord_planes", True)),
        freeze_codec=bool(cfg.get("freeze_codec", False)),
        detach_surprise=bool(cfg.get("detach_surprise", False)),
        strict_tokenizer=bool(cfg.get("strict_tokenizer", False)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
