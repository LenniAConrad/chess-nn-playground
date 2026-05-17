"""Algebraic inference rewrite of i018 oriented_tactical_sheaf_laplacian.

Idea i249 is intentionally not a new architecture. It keeps i018's adapter,
incidence builder, encoder, readout, diagnostics, parameter names, and sheaf
math, but replaces the slow edge-materializing diffusion block with an
algebraically equivalent implementation.

The original block forms, for each relation, a dense edge residual tensor
``(B, 64, 64, stalk_dim)`` and then projects/reduces it. This rewrite expands
the same sums before materialization:

    sum_j W_ij (dst_j - sign * src_i)
    sum_i W_ij (dst_j - sign * src_i)

which can be computed from ``W @ dst``, ``W.T @ src``, in-degrees, and
out-degrees. That removes the largest intermediate and collapses the 12-relation
Python loop into a small set of batched matrix multiplications.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import require_board_tensor
from chess_nn_playground.models.trunk.oriented_tactical_sheaf import (
    RELATION_NAMES,
    OrientedTacticalSheafNet,
    _format_logits,
    _weighted_mean,
)


def _resolve_autocast_dtype(value: Any) -> torch.dtype | None:
    if value is None or value is False:
        return None
    text = str(value).strip().lower()
    if text in {"", "0", "false", "none", "off", "disabled"}:
        return None
    if text in {"float16", "fp16", "half"}:
        return torch.float16
    if text in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError("inference_autocast_dtype must be one of: none, float16, bfloat16")


class FastSheafDiffusionBlock(nn.Module):
    """Algebraically equivalent, dense-edge-free i018 diffusion block.

    Parameter names and shapes match ``SheafDiffusionBlock`` exactly so i018 and
    i249 checkpoints can be loaded in either direction. Only the execution order
    changes.
    """

    def __init__(
        self,
        d_model: int,
        relation_count: int,
        stalk_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.relation_count = int(relation_count)
        self.stalk_dim = int(stalk_dim)
        self.node_to_stalk = nn.Linear(d_model, stalk_dim)
        self.stalk_to_node = nn.Linear(stalk_dim, d_model)
        eye = torch.eye(stalk_dim).unsqueeze(0).repeat(relation_count, 1, 1)
        self.rho_src = nn.Parameter(eye + 0.02 * torch.randn(relation_count, stalk_dim, stalk_dim))
        self.rho_dst = nn.Parameter(eye + 0.02 * torch.randn(relation_count, stalk_dim, stalk_dim))
        self.relation_gate_logits = nn.Parameter(torch.zeros(relation_count))
        self.eta_logit = nn.Parameter(torch.tensor(0.0))
        signs = torch.tensor([-1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=torch.float32)
        self.register_buffer("relation_signs", signs, persistent=False)
        self.node_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, h: torch.Tensor, relation_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.node_to_stalk(h)
        gates = 2.0 * torch.sigmoid(self.relation_gate_logits)
        eta = 0.25 * torch.sigmoid(self.eta_logit)

        # Project all relation-specific source/target stalks once.
        src = torch.matmul(z.unsqueeze(1), self.rho_src.unsqueeze(0))
        dst = torch.matmul(z.unsqueeze(1), self.rho_dst.unsqueeze(0))

        out_degree = relation_masks.sum(dim=-1)
        in_degree = relation_masks.sum(dim=-2)
        w_dst = torch.matmul(relation_masks, dst)
        wt_src = torch.matmul(relation_masks.transpose(-1, -2), src)

        signs = self.relation_signs.to(dtype=z.dtype).view(1, self.relation_count, 1, 1)
        gates_view = gates.to(dtype=z.dtype).view(1, self.relation_count, 1, 1)

        source_pre = signs * w_dst - out_degree.unsqueeze(-1) * src
        target_pre = signs * wt_src - in_degree.unsqueeze(-1) * dst
        source_back = torch.matmul(source_pre, self.rho_src.transpose(-1, -2).unsqueeze(0))
        target_back = torch.matmul(target_pre, self.rho_dst.transpose(-1, -2).unsqueeze(0))
        z_update = (gates_view * (source_back + target_back)).sum(dim=1)

        degree = (gates.to(dtype=z.dtype).view(1, self.relation_count, 1) * (out_degree + in_degree)).sum(dim=1)
        z_update = eta.to(dtype=z.dtype) * z_update / degree.unsqueeze(-1).clamp_min(1.0)
        h = self.norm(h + self.stalk_to_node(z_update) + self.node_mlp(h))

        src_norm = src.square().sum(dim=-1)
        dst_norm = dst.square().sum(dim=-1)
        cross = (src * w_dst).sum(dim=-1)
        energy_numer = (
            (out_degree * src_norm).sum(dim=-1)
            + (in_degree * dst_norm).sum(dim=-1)
            - 2.0 * self.relation_signs.to(dtype=z.dtype).view(1, self.relation_count) * cross.sum(dim=-1)
        )
        denom = out_degree.sum(dim=-1).clamp_min(1.0)
        energies = gates.to(dtype=z.dtype).view(1, self.relation_count) * energy_numer / denom
        return h, energies, gates


class OrientedTacticalSheafFastNet(OrientedTacticalSheafNet):
    """i018 with algebraic sheaf diffusion and optional compiled forward."""

    def __init__(
        self,
        *args: Any,
        compile_model: bool = True,
        compile_mode: str = "reduce-overhead",
        return_diagnostics: bool = True,
        inference_autocast_dtype: str | torch.dtype | None = None,
        inference_autocast_min_batch: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.return_diagnostics = bool(return_diagnostics)
        self.inference_autocast_dtype = _resolve_autocast_dtype(inference_autocast_dtype)
        self.inference_autocast_min_batch = max(1, int(inference_autocast_min_batch))
        depth = len(self.blocks)
        relation_count = len(RELATION_NAMES)
        d_model = self.blocks[0].node_to_stalk.in_features
        stalk_dim = self.blocks[0].node_to_stalk.out_features
        dropout = self.blocks[0].node_mlp[3].p if len(self.blocks[0].node_mlp) > 3 else 0.1
        self.blocks = nn.ModuleList(
            [FastSheafDiffusionBlock(d_model, relation_count, stalk_dim, dropout) for _ in range(depth)]
        )
        self._compiled_forward = None
        if compile_model:
            try:
                self._compiled_forward = torch.compile(self._raw_forward, mode=compile_mode)
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"[oriented_tactical_sheaf_fast] torch.compile unavailable, running eager: {exc}")
                self._compiled_forward = None

    def _forward_impl(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.adapter(x)
        incidence = self.incidence(board.piece_state, board.occupancy)
        if self.scramble_relations:
            sheaf_masks = incidence.relation_masks
            batch, relations, squares, _ = sheaf_masks.shape
            perm = torch.argsort(torch.rand(batch, relations, squares, device=sheaf_masks.device), dim=-1)
            perm_expanded = perm.unsqueeze(-2).expand(-1, -1, squares, -1)
            sheaf_masks = torch.gather(sheaf_masks, dim=-1, index=perm_expanded)
        else:
            sheaf_masks = incidence.relation_masks

        h = self.encoder(board.square_raw, board.piece_state)
        block_energies: list[torch.Tensor] = []
        block_gates: list[torch.Tensor] = []
        for block in self.blocks:
            h, energy, gates = block(h, sheaf_masks)
            block_energies.append(energy)
            block_gates.append(gates.unsqueeze(0).expand(x.shape[0], -1))

        energy_stack = torch.stack(block_energies, dim=1)
        gate_stack = torch.stack(block_gates, dim=1)
        energy_mean = energy_stack.mean(dim=1)
        energy_max = energy_stack.amax(dim=1)
        gate_mean = gate_stack.mean(dim=1)
        triad_stats = (
            self.triad_pool(h, incidence)
            if self.triad_pool is not None
            else h.new_zeros(h.shape[0], 0)
        )
        readout = torch.cat(
            [
                h.mean(dim=1),
                h.amax(dim=1),
                _weighted_mean(h, incidence.our_piece),
                _weighted_mean(h, incidence.them_piece),
                energy_mean,
                energy_max,
                incidence.relation_density,
                gate_mean,
                triad_stats,
                self._board_stats(board, incidence),
            ],
            dim=1,
        )
        logits = _format_logits(self.head(readout), self.num_classes)
        if not self.return_diagnostics:
            return {"logits": logits}

        sheaf_tension = energy_stack.mean(dim=(1, 2))
        us_pressure = incidence.relation_masks[:, 0].sum(dim=(1, 2))
        them_pressure = incidence.relation_masks[:, 1].sum(dim=(1, 2))
        us_defense = incidence.relation_masks[:, 2].sum(dim=(1, 2))
        them_defense = incidence.relation_masks[:, 3].sum(dim=(1, 2))
        rank_counts = torch.matmul(board.occupancy, self.incidence.rank_one_hot)
        file_counts = torch.matmul(board.occupancy, self.incidence.file_one_hot)
        piece_entropy = -(board.piece_state * board.piece_state.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=1)
        return {
            "logits": logits,
            "mechanism_energy": torch.log1p(sheaf_tension),
            "proposal_profile_strength": gate_mean.mean(dim=1),
            "proposal_keyword_count": logits.new_full((x.shape[0],), 4.0),
            "sheaf_tension": sheaf_tension,
            "transport_imbalance": (us_pressure - them_pressure).abs() / (us_pressure + them_pressure).clamp_min(1.0),
            "symmetry_residual": (incidence.our_attack.mean(dim=(1, 2)) - incidence.them_attack.mean(dim=(1, 2))).abs(),
            "topology_pressure": incidence.relation_density.mean(dim=1),
            "ray_language_energy": energy_mean[:, 6:9].mean(dim=1),
            "information_surprisal": piece_entropy,
            "sparse_certificate_energy": energy_stack.amax(dim=(1, 2)),
            "rank_file_imbalance": (rank_counts.std(dim=1) - file_counts.std(dim=1)).abs(),
            "king_ring_pressure": incidence.relation_density[:, 4] + incidence.relation_density[:, 5],
            "reply_pressure": 0.5 * (us_pressure + them_pressure) / 64.0,
            "defense_gap": ((us_pressure + them_pressure) - (us_defense + them_defense)) / 64.0,
            "triad_defect_energy": triad_stats[:, 0] if triad_stats.numel() else logits.new_zeros(x.shape[0]),
            "pin_pressure": incidence.relation_density[:, 11],
        }

    def _raw_forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        autocast_dtype = self.inference_autocast_dtype
        if (
            autocast_dtype is not None
            and not self.training
            and not torch.is_grad_enabled()
            and x.is_cuda
            and int(x.shape[0]) >= self.inference_autocast_min_batch
        ):
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                return self._forward_impl(x)
        return self._forward_impl(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self._compiled_forward is not None:
            return self._compiled_forward(x)
        return self._raw_forward(x)


def build_oriented_tactical_sheaf_fast_from_config(
    config: dict[str, Any],
) -> OrientedTacticalSheafFastNet:
    return OrientedTacticalSheafFastNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("sheaf_layers", config.get("depth", 2))),
        stalk_dim=int(config.get("stalk_dim", 8)),
        dropout=float(config.get("dropout", 0.1)),
        encoding=str(config.get("encoding", "simple_18")),
        piece_adapter=str(config.get("piece_adapter", "exact")),
        use_triads=bool(config.get("use_triads", True)),
        scramble_relations=bool(config.get("scramble_relations", False)),
        compile_model=bool(config.get("compile_model", True)),
        compile_mode=str(config.get("compile_mode", "reduce-overhead")),
        return_diagnostics=bool(config.get("return_diagnostics", True)),
        inference_autocast_dtype=config.get("inference_autocast_dtype"),
        inference_autocast_min_batch=int(config.get("inference_autocast_min_batch", 1)),
    )
