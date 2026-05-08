"""Wavelet Scattering Board Network for idea i093."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


_HAAR_FILTERS = {
    "LL": ((1.0, 1.0), (1.0, 1.0)),
    "H": ((1.0, 1.0), (-1.0, -1.0)),
    "V": ((1.0, -1.0), (1.0, -1.0)),
    "D": ((1.0, -1.0), (-1.0, 1.0)),
}
ORIENTATIONS = ("H", "V", "D")


def _haar_filters() -> torch.Tensor:
    rows = [_HAAR_FILTERS["LL"], _HAAR_FILTERS["H"], _HAAR_FILTERS["V"], _HAAR_FILTERS["D"]]
    tensor = torch.tensor(rows, dtype=torch.float32) / 2.0
    return tensor.view(4, 1, 2, 2)


def _random_orthogonal_filters(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    raw = torch.randn(4, 4, generator=generator)
    q, _ = torch.linalg.qr(raw)
    return q.view(4, 1, 2, 2) / math.sqrt(4.0)


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _pad_for_dilation(x: torch.Tensor, dilation: int, mode: str = "circular") -> torch.Tensor:
    if mode == "zeros":
        return F.pad(x, (0, dilation, 0, dilation), mode="constant", value=0.0)
    return F.pad(x, (0, dilation, 0, dilation), mode=mode)


class FixedWaveletBank(nn.Module):
    """Fixed 2x2 wavelet bank applied depthwise across channels at a fixed dilation."""

    def __init__(
        self,
        input_channels: int,
        dilation: int,
        kind: str = "haar",
        seed: int = 0,
        pad_mode: str = "circular",
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.dilation = int(dilation)
        self.pad_mode = str(pad_mode)
        if kind == "haar":
            base = _haar_filters()
        elif kind == "random":
            base = _random_orthogonal_filters(seed)
        else:
            raise ValueError(f"unknown wavelet kind: {kind}")
        # weight shape: (out_channels=in_channels*4, 1, 2, 2)
        weight = base.unsqueeze(0).expand(self.input_channels, -1, -1, -1, -1).contiguous()
        weight = weight.view(self.input_channels * 4, 1, 2, 2)
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        padded = _pad_for_dilation(x, self.dilation, mode=self.pad_mode)
        out = F.conv2d(padded, self.weight, dilation=self.dilation, groups=self.input_channels)
        # out shape: (B, in_channels*4, 8, 8) where the 4 fast axis is (LL, H, V, D).
        b, _, h, w = out.shape
        out = out.view(b, self.input_channels, 4, h, w)
        return {
            "ll": out[:, :, 0],
            "high": out[:, :, 1:],  # (B, C, 3, 8, 8) ordered (H, V, D)
        }


class WaveletScatteringFeatures(nn.Module):
    """Two-layer Haar scattering with channel/scale/orientation pooling stats."""

    def __init__(
        self,
        input_channels: int = 18,
        scales: tuple[int, ...] = (1, 2, 4),
        kind: str = "haar",
        pad_mode: str = "circular",
        second_order: bool = True,
        seed_base: int = 1093,
    ) -> None:
        super().__init__()
        if not scales:
            raise ValueError("scales must be non-empty")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.input_channels = int(input_channels)
        self.scales = tuple(int(s) for s in scales)
        self.kind = str(kind)
        self.second_order = bool(second_order)
        self.first_layer = nn.ModuleList(
            FixedWaveletBank(
                input_channels=self.input_channels,
                dilation=int(scale),
                kind=self.kind,
                seed=seed_base + idx,
                pad_mode=pad_mode,
            )
            for idx, scale in enumerate(self.scales)
        )
        if self.second_order:
            self.second_layer = nn.ModuleList(
                FixedWaveletBank(
                    input_channels=self.input_channels,
                    dilation=int(scale),
                    kind=self.kind,
                    seed=seed_base + 100 + idx,
                    pad_mode=pad_mode,
                )
                for idx, scale in enumerate(self.scales)
            )

    @property
    def num_scales(self) -> int:
        return len(self.scales)

    @property
    def num_orientations(self) -> int:
        return len(ORIENTATIONS)

    @property
    def first_order_count(self) -> int:
        return self.num_scales * self.num_orientations

    @property
    def second_order_pairs(self) -> list[tuple[int, int]]:
        if not self.second_order:
            return []
        return [(s1, s2) for s1 in range(self.num_scales) for s2 in range(self.num_scales) if s2 > s1]

    @property
    def second_order_count(self) -> int:
        return len(self.second_order_pairs) * self.num_orientations * self.num_orientations

    @property
    def feature_count_per_channel(self) -> int:
        # First-order: 3 stats (mean, std, max) per (scale, orientation) modulus field.
        first = 3 * self.first_order_count
        # Lowpass signed energy per scale.
        low = self.num_scales
        # Second-order: mean of |W_{s2, o2} * |W_{s1, o1} * x|| for s2 > s1.
        second = self.second_order_count if self.second_order else 0
        return first + low + second

    @property
    def feature_dim(self) -> int:
        return self.feature_count_per_channel * self.input_channels

    def forward(
        self,
        x: torch.Tensor,
        *,
        lowpass_only: bool = False,
        return_diag: bool = False,
    ) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        first_lows: list[torch.Tensor] = []
        first_highs: list[torch.Tensor] = []
        for bank in self.first_layer:
            decomposition = bank(board)
            first_lows.append(decomposition["ll"])  # (B, C, 8, 8)
            first_highs.append(decomposition["high"])  # (B, C, O, 8, 8)
        # Stack first-order tensors along scale dim.
        ll_stack = torch.stack(first_lows, dim=2)  # (B, C, S, 8, 8)
        high_stack = torch.stack(first_highs, dim=2)  # (B, C, S, O, 8, 8)
        if lowpass_only:
            modulus_stack = torch.zeros_like(high_stack)
        else:
            modulus_stack = high_stack.abs()

        b, c, scales_count, orient_count, h, w = modulus_stack.shape
        # First-order pooling: per-channel/scale/orientation mean, std, max.
        flat = modulus_stack.flatten(start_dim=4)  # (B, C, S, O, 64)
        mean_field = flat.mean(dim=-1)
        std_field = flat.std(dim=-1, unbiased=False)
        max_field = flat.max(dim=-1).values

        # Lowpass signed energy: per channel/scale, take mean of LL.
        ll_flat = ll_stack.flatten(start_dim=3)  # (B, C, S, 64)
        ll_energy = ll_flat.mean(dim=-1)

        feature_chunks: list[torch.Tensor] = [
            mean_field.flatten(start_dim=1),
            std_field.flatten(start_dim=1),
            max_field.flatten(start_dim=1),
            ll_energy.flatten(start_dim=1),
        ]

        second_features: torch.Tensor | None = None
        if self.second_order:
            # modulus_stack: (B, C, S, O, 8, 8). Treat (C * O) as channels and apply per-scale bank.
            # We need second-order means for s2 > s1.
            second_chunks: list[torch.Tensor] = []
            # Reshape modulus into (B, C * O, 8, 8) per s1 to feed the second layer banks.
            for s1_idx in range(scales_count):
                first_order = modulus_stack[:, :, s1_idx]  # (B, C, O, 8, 8)
                first_order_flat = first_order.reshape(b, c * orient_count, h, w)
                # Treat each (channel, orientation) as a separate signal.
                fake_bank = self._second_layer_for_signals(s1_idx)
                if fake_bank is None:
                    continue
                for s2_idx in range(scales_count):
                    if s2_idx <= s1_idx:
                        continue
                    bank = self.second_layer[s2_idx]
                    padded = _pad_for_dilation(first_order_flat, bank.dilation, mode=bank.pad_mode)
                    # Need a per-(C*O) depthwise conv with the same wavelet kernel as bank.
                    weight = self._depthwise_weight_for(bank.weight, c * orient_count)
                    conv = F.conv2d(padded, weight, dilation=bank.dilation, groups=c * orient_count)
                    # conv shape: (B, C*O*4, 8, 8). The fast 4 axis is (LL, H, V, D); take high-band modulus.
                    conv = conv.view(b, c * orient_count, 4, h, w)
                    second_modulus = conv[:, :, 1:].abs()  # (B, C*O, O2=3, 8, 8)
                    pooled = second_modulus.flatten(start_dim=3).mean(dim=-1)  # (B, C*O, 3)
                    pooled = pooled.view(b, c, orient_count, 3)
                    second_chunks.append(pooled.flatten(start_dim=1))
            if second_chunks:
                second_features = torch.cat(second_chunks, dim=1)
                feature_chunks.append(second_features)

        features = torch.cat(feature_chunks, dim=1)
        diagnostics = {
            "first_order_mean_field": mean_field,
            "first_order_std_field": std_field,
            "first_order_max_field": max_field,
            "lowpass_energy": ll_energy,
            "scattering_features": features,
        }
        if second_features is not None:
            diagnostics["second_order_mean_field"] = second_features
        if return_diag:
            diagnostics["first_order_modulus"] = modulus_stack
            diagnostics["lowpass_field"] = ll_stack
        return diagnostics

    def _second_layer_for_signals(self, s1_idx: int) -> nn.Module | None:
        if not self.second_order:
            return None
        return self.second_layer[s1_idx]

    @staticmethod
    def _depthwise_weight_for(weight: torch.Tensor, num_signals: int) -> torch.Tensor:
        # `weight` is (input_channels * 4, 1, 2, 2) where input_channels was the first-layer width.
        # We want a depthwise weight for `num_signals` groups, each replicating the 4 Haar filters.
        # Extract the 4 unique 2x2 filters: take the first 4 rows.
        base = weight[:4]  # (4, 1, 2, 2)
        repeated = base.unsqueeze(0).expand(num_signals, -1, -1, -1, -1).contiguous()
        repeated = repeated.view(num_signals * 4, 1, 2, 2)
        return repeated


class WaveletScatteringBoardNetwork(nn.Module):
    """Fixed Haar scattering front end + small MLP head for puzzle_binary."""

    MODES = ("haar", "random_fixed_filters", "lowpass_only", "channel_shuffle")

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        scales: tuple[int, ...] = (1, 2, 4),
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        mode: str = "haar",
        second_order: bool = True,
        pad_mode: str = "circular",
        random_seed: int = 1093,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("WaveletScatteringBoardNetwork supports the puzzle_binary one-logit contract")
        if mode not in self.MODES:
            raise ValueError(f"unknown mode: {mode!r}; expected one of {self.MODES}")
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.scales = tuple(int(s) for s in scales)
        self.mode = str(mode)
        kind = "random" if mode == "random_fixed_filters" else "haar"
        self.scattering = WaveletScatteringFeatures(
            input_channels=self.input_channels,
            scales=self.scales,
            kind=kind,
            pad_mode=pad_mode,
            second_order=bool(second_order),
            seed_base=int(random_seed),
        )
        if self.mode == "channel_shuffle":
            generator = torch.Generator().manual_seed(int(random_seed) + 7)
            permutation = torch.randperm(self.input_channels, generator=generator)
            self.register_buffer("channel_permutation", permutation, persistent=False)
        else:
            self.register_buffer("channel_permutation", torch.arange(self.input_channels), persistent=False)

        feature_dim = self.scattering.feature_dim
        head_hidden = int(hidden_dim)
        head_layers: list[nn.Module] = [nn.LayerNorm(feature_dim)]
        in_features = feature_dim
        for _ in range(max(1, int(depth))):
            head_layers.append(nn.Linear(in_features, head_hidden))
            head_layers.append(nn.GELU())
            if dropout > 0:
                head_layers.append(nn.Dropout(float(dropout)))
            in_features = head_hidden
        head_layers.append(nn.Linear(in_features, self.num_classes))
        self.head = nn.Sequential(*head_layers)
        self._feature_dim = feature_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor, *, return_diag: bool = False) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.scattering.spec)
        if self.mode == "channel_shuffle":
            board = board.index_select(1, self.channel_permutation)
        scattering = self.scattering(
            board,
            lowpass_only=(self.mode == "lowpass_only"),
            return_diag=return_diag,
        )
        features = scattering["scattering_features"]
        logits = _format_logits(self.head(features), self.num_classes)
        output = {
            "logits": logits,
            "prob": torch.sigmoid(logits),
            "scattering_features": features,
            "first_order_mean_field": scattering["first_order_mean_field"],
            "first_order_std_field": scattering["first_order_std_field"],
            "first_order_max_field": scattering["first_order_max_field"],
            "lowpass_energy": scattering["lowpass_energy"],
            "scattering_mode": logits.new_full((logits.shape[0],), self._mode_code()),
            "mechanism_energy": scattering["first_order_mean_field"].pow(2).flatten(1).mean(dim=1),
            "proposal_profile_strength": scattering["first_order_max_field"].flatten(1).max(dim=1).values,
            "proposal_keyword_count": logits.new_full((logits.shape[0],), 3.0),
            "scale_count": logits.new_full((logits.shape[0],), float(len(self.scales))),
        }
        if "second_order_mean_field" in scattering:
            output["second_order_mean_field"] = scattering["second_order_mean_field"]
        if return_diag:
            output["first_order_modulus"] = scattering["first_order_modulus"]
            output["lowpass_field"] = scattering["lowpass_field"]
        return output

    def _mode_code(self) -> float:
        return float(self.MODES.index(self.mode))


def build_wavelet_scattering_board_network_from_config(config: dict[str, Any]) -> WaveletScatteringBoardNetwork:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    raw_scales = cfg.get("scales", (1, 2, 4))
    if isinstance(raw_scales, (list, tuple)):
        scales = tuple(int(s) for s in raw_scales)
    else:
        scales = (int(raw_scales),)
    hidden_dim = int(cfg.get("hidden_dim", cfg.get("channels", 96)))
    return WaveletScatteringBoardNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        scales=scales,
        hidden_dim=hidden_dim,
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        mode=str(cfg.get("mode", "haar")),
        second_order=bool(cfg.get("second_order", True)),
        pad_mode=str(cfg.get("pad_mode", "circular")),
        random_seed=int(cfg.get("random_seed", 1093)),
    )
