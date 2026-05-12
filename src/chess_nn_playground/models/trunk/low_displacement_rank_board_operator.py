"""Low-Displacement-Rank Board Operator for idea i140.

The model parameterises a 64x64 global square-mixing operator

    A = T_rank + T_file + H_diag + H_anti + U V^T

so that the Sylvester displacement ``A - Z A Z^T`` (with ``Z`` a shift on the
flattened board) is low rank by construction.  Concretely:

* ``T_rank = T_r (x) I_8`` is rank-only Toeplitz (15 params, applied per file).
* ``T_file = I_8 (x) T_f`` is file-only Toeplitz (15 params, applied per rank).
* ``H_diag[s1, s2] = h_main[d_main(s1) + d_main(s2)]`` is a 64x64 Hankel-like
  mixer indexed by the main diagonal ``d_main(r, f) = r - f + 7`` (29 params).
* ``H_anti[s1, s2] = h_anti[d_anti(s1) + d_anti(s2)]`` is the anti-diagonal
  Hankel-like counterpart with ``d_anti = r + f`` (29 params).
* ``U V^T`` is a small learned low-rank residual (``2 * 64 * low_rank_dim``
  params).

Each layer is then ``h_{t+1} = sigma(A_t h_t + pointwise_mix(h_t))`` with a
1x1 channel mix, exactly as the packet sketch prescribes, and the head
consumes the operator response statistics: per-component energies, the
operator-residual norm ``||A h - h||``, the displacement residual norm
``||A - Z A Z^T||``, and pooled trunk features.

The mechanism is structurally distinct from a CNN (no 2D translation
invariance: ``T_rank`` is global along rank but separable across files) and
from attention (no data-dependent pair weights): all four structured
components are static low-displacement-rank operators learned as small
parameter vectors over the flattened board.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


class _StructuredBoardOperator(nn.Module):
    """Build the 64x64 low-displacement-rank operator and apply it.

    The five structured components are each materialised explicitly so that
    the per-component response statistics required by the packet diagnostics
    can be read off directly.
    """

    def __init__(self, low_rank_dim: int = 4) -> None:
        super().__init__()
        if low_rank_dim < 1:
            raise ValueError("low_rank_dim must be >= 1")
        self.low_rank_dim = int(low_rank_dim)

        # 15 Toeplitz offsets per axis, 29 Hankel offsets per diagonal family.
        self.t_rank = nn.Parameter(torch.zeros(15))
        self.t_file = nn.Parameter(torch.zeros(15))
        self.h_diag = nn.Parameter(torch.zeros(29))
        self.h_anti = nn.Parameter(torch.zeros(29))

        # Initialise so the operator is small but nonzero, breaking the
        # zero gradient on the structured offsets.
        nn.init.normal_(self.t_rank, std=0.05)
        nn.init.normal_(self.t_file, std=0.05)
        nn.init.normal_(self.h_diag, std=0.05)
        nn.init.normal_(self.h_anti, std=0.05)

        self.U = nn.Parameter(torch.randn(64, low_rank_dim) * 0.05)
        self.V = nn.Parameter(torch.randn(64, low_rank_dim) * 0.05)

        rank = torch.arange(8).view(8, 1).expand(8, 8).reshape(64)
        file = torch.arange(8).view(1, 8).expand(8, 8).reshape(64)
        rank_diff = rank.view(64, 1) - rank.view(1, 64) + 7
        file_diff = file.view(64, 1) - file.view(1, 64) + 7
        same_file = (file.view(64, 1) == file.view(1, 64)).to(torch.float32)
        same_rank = (rank.view(64, 1) == rank.view(1, 64)).to(torch.float32)
        d_main = rank - file + 7  # in [0, 14]
        d_anti = rank + file  # in [0, 14]
        diag_idx = d_main.view(64, 1) + d_main.view(1, 64)
        anti_idx = d_anti.view(64, 1) + d_anti.view(1, 64)

        # Cyclic shift Z on 64 squares (row-major), used for the
        # displacement residual diagnostic.
        shift = torch.zeros(64, 64)
        for i in range(63):
            shift[i + 1, i] = 1.0

        self.register_buffer("rank_diff_idx", rank_diff.to(torch.long), persistent=False)
        self.register_buffer("file_diff_idx", file_diff.to(torch.long), persistent=False)
        self.register_buffer("same_file_mask", same_file, persistent=False)
        self.register_buffer("same_rank_mask", same_rank, persistent=False)
        self.register_buffer("diag_idx", diag_idx.to(torch.long), persistent=False)
        self.register_buffer("anti_idx", anti_idx.to(torch.long), persistent=False)
        self.register_buffer("shift", shift, persistent=False)

    def components(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(T_rank, T_file, H_diag, H_anti, U V^T)`` 64x64 tensors."""
        t_rank = self.t_rank[self.rank_diff_idx] * self.same_file_mask
        t_file = self.t_file[self.file_diff_idx] * self.same_rank_mask
        h_diag = self.h_diag[self.diag_idx]
        h_anti = self.h_anti[self.anti_idx]
        uv = self.U @ self.V.transpose(-1, -2)
        return t_rank, t_file, h_diag, h_anti, uv

    def operator(self) -> torch.Tensor:
        return sum(self.components())

    def displacement_residual_norm(self) -> torch.Tensor:
        a = self.operator()
        d = a - self.shift @ a @ self.shift.transpose(-1, -2)
        return torch.linalg.norm(d.reshape(-1))

    def forward(
        self, flat: torch.Tensor, return_diagnostics: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Apply ``A`` to a flattened board ``flat`` of shape ``(B, C, 64)``."""
        t_rank, t_file, h_diag, h_anti, uv = self.components()
        a = t_rank + t_file + h_diag + h_anti + uv
        out = torch.matmul(flat, a.transpose(-1, -2))
        diagnostics: dict[str, torch.Tensor] | None = None
        if return_diagnostics:

            def energy(component: torch.Tensor) -> torch.Tensor:
                y = torch.matmul(flat, component.transpose(-1, -2))
                # Mean squared response per sample, averaged over channels and squares.
                return y.pow(2).mean(dim=(-1, -2))

            diagnostics = {
                "energy_t_rank": energy(t_rank),
                "energy_t_file": energy(t_file),
                "energy_h_diag": energy(h_diag),
                "energy_h_anti": energy(h_anti),
                "energy_low_rank": energy(uv),
            }
        return out, diagnostics


class _LDRBlock(nn.Module):
    """One structured-operator layer ``h -> sigma(A h + pointwise_mix(h))``."""

    def __init__(
        self,
        channels: int,
        low_rank_dim: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        self.operator = _StructuredBoardOperator(low_rank_dim=low_rank_dim)
        self.pointwise_mix = nn.Conv2d(channels, channels, kernel_size=1, bias=not use_batchnorm)
        self.norm = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self, x: torch.Tensor, return_diagnostics: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        bsz, c, h, w = x.shape
        flat = x.reshape(bsz, c, h * w)
        op_out, diagnostics = self.operator(flat, return_diagnostics=return_diagnostics)
        op_out = op_out.reshape(bsz, c, h, w)
        mix = self.pointwise_mix(x)
        y = self.activation(self.norm(op_out + mix))
        y = self.dropout(y)
        if return_diagnostics:
            assert diagnostics is not None
            # Operator response residual ||A h - h|| / sqrt(C * 64), per sample.
            residual = (op_out - x).reshape(bsz, -1)
            response_residual = residual.pow(2).mean(dim=-1).sqrt()
            diagnostics["operator_response_residual"] = response_residual
            diagnostics["displacement_residual_norm"] = (
                self.operator.displacement_residual_norm().detach().expand(bsz)
            )
        return y, diagnostics


class LowDisplacementRankBoardOperator(nn.Module):
    """Bespoke implementation of idea i140.

    The model consumes only the board tensor; CRTK / source metadata is
    reporting-only and never used as input.
    """

    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        num_layers: int = 3,
        low_rank_dim: int = 4,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "LowDisplacementRankBoardOperator follows the puzzle_binary one-logit contract"
            )
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.channels = int(channels)
        self.num_layers = int(num_layers)
        self.low_rank_dim = int(low_rank_dim)

        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=1, bias=not use_batchnorm),
            nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity(),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [
                _LDRBlock(
                    channels=channels,
                    low_rank_dim=low_rank_dim,
                    dropout=dropout,
                    use_batchnorm=use_batchnorm,
                )
                for _ in range(num_layers)
            ]
        )

        # Diagnostics fed into the head:
        # 5 per-component energies + operator response residual + displacement residual
        diagnostic_dim = 5 + 2
        pooled_trunk_dim = 2 * channels  # mean + max pool
        head_in = pooled_trunk_dim + diagnostic_dim

        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        h = self.input_proj(x)
        diagnostics: dict[str, torch.Tensor] | None = None
        for idx, layer in enumerate(self.layers):
            return_diag = idx == self.num_layers - 1
            h, layer_diag = layer(h, return_diagnostics=return_diag)
            if layer_diag is not None:
                diagnostics = layer_diag
        assert diagnostics is not None  # last layer always returns diagnostics

        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        diag_vec = torch.stack(
            [
                diagnostics["energy_t_rank"],
                diagnostics["energy_t_file"],
                diagnostics["energy_h_diag"],
                diagnostics["energy_h_anti"],
                diagnostics["energy_low_rank"],
                diagnostics["operator_response_residual"],
                diagnostics["displacement_residual_norm"],
            ],
            dim=-1,
        )
        feat = torch.cat([pooled, diag_vec], dim=-1)
        logits = self.head(feat).view(-1)

        # Aggregate per-layer displacement residual norms for reporting; here we
        # only expose the last layer's value because earlier layers are skipped
        # for cost.  Callers needing all layers can fold the diagnostics
        # outwards from this module.
        return {
            "logits": logits,
            "ldr_energy_t_rank": diagnostics["energy_t_rank"],
            "ldr_energy_t_file": diagnostics["energy_t_file"],
            "ldr_energy_h_diag": diagnostics["energy_h_diag"],
            "ldr_energy_h_anti": diagnostics["energy_h_anti"],
            "ldr_energy_low_rank": diagnostics["energy_low_rank"],
            "ldr_operator_response_residual": diagnostics["operator_response_residual"],
            "ldr_displacement_residual_norm": diagnostics["displacement_residual_norm"],
        }


def build_low_displacement_rank_board_operator_from_config(
    config: dict[str, Any],
) -> LowDisplacementRankBoardOperator:
    cfg = dict(config)
    return LowDisplacementRankBoardOperator(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        num_layers=int(cfg.get("num_layers", cfg.get("depth", 3))),
        low_rank_dim=int(cfg.get("low_rank_dim", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        num_classes=int(cfg.get("num_classes", 1)),
    )
