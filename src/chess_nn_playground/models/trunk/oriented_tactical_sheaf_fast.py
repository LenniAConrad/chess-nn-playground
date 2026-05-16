"""Speed-optimized variant of i018 oriented_tactical_sheaf_laplacian (idea i249).

Same math, same parameters, same numerics as the original
``oriented_tactical_sheaf`` module -- only the GPU execution pattern changes:

1. ``FastSheafDiffusionBlock`` replaces the original per-relation Python ``for``
   loop (12 iterations x ~6 small ops = ~72 kernel launches per block) with a
   single vectorized projection over all relations plus a chunked batched
   coboundary. This collapses kernel-launch overhead and reuses the projected
   stalks instead of recomputing them per relation. The chunk size caps the
   peak ``(B, chunk, 64, 64, stalk)`` intermediate so it stays within an 8 GB
   GPU. The arithmetic is identical to the original block, so a model trained
   with either block is the same architecture.

2. Optional ``torch.compile`` wrapping of the forward pass. i018 has fully
   static shapes (board ``(B, 18, 8, 8)``, relations ``(B, 12, 64, 64)``),
   which is the ideal case for kernel fusion + CUDA graphs via
   ``mode="reduce-overhead"``. Compilation wraps a *bound method*, not the
   module, so ``state_dict`` keys stay clean (no ``_orig_mod.`` prefix).

Everything else -- BoardStateAdapter, TacticalIncidenceBuilder,
SquareTokenEncoder, TriadDefectPool, the readout head -- is imported unchanged
from the original module so this variant can never silently drift from i018.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.oriented_tactical_sheaf import (
    RELATION_NAMES,
    OrientedTacticalSheafNet,
)


class FastSheafDiffusionBlock(nn.Module):
    """Vectorized, chunked equivalent of the original SheafDiffusionBlock.

    Identical parameter set and identical math; only the execution pattern
    differs. Parameter names match the original so a state_dict transfers
    cleanly in either direction.
    """

    def __init__(
        self,
        d_model: int,
        relation_count: int,
        stalk_dim: int,
        dropout: float,
        chunk_size: int = 3,
    ) -> None:
        super().__init__()
        self.relation_count = int(relation_count)
        self.stalk_dim = int(stalk_dim)
        self.chunk_size = max(1, int(chunk_size))
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
        batch, squares, stalk_dim = z.shape
        gates = 2.0 * torch.sigmoid(self.relation_gate_logits)
        eta = 0.25 * torch.sigmoid(self.eta_logit)

        # Project every relation's source/target stalks in one shot, instead of
        # recomputing z @ rho_src / z @ rho_dst inside a 12-iteration loop.
        # src_all[b, r, n, t] = sum_s z[b, n, s] * rho_src[r, s, t]
        src_all = torch.einsum("bns,rst->brnt", z, self.rho_src)
        dst_all = torch.einsum("bns,rst->brnt", z, self.rho_dst)

        z_update = z.new_zeros(batch, squares, stalk_dim)
        degree = z.new_zeros(batch, squares)
        energy_chunks: list[torch.Tensor] = []

        for start in range(0, self.relation_count, self.chunk_size):
            stop = min(start + self.chunk_size, self.relation_count)
            w = relation_masks[:, start:stop]                       # (B, Rc, Nv, Nu)
            src = src_all[:, start:stop]                            # (B, Rc, N, s)
            dst = dst_all[:, start:stop]                            # (B, Rc, N, s)
            signs = self.relation_signs[start:stop]                 # (Rc,)
            g = gates[start:stop]                                   # (Rc,)
            rho_src_c = self.rho_src[start:stop]                    # (Rc, s, s)
            rho_dst_c = self.rho_dst[start:stop]                    # (Rc, s, s)

            # Match the original block exactly: residual[b, r, i, j, s] indexes
            # src on i (dim 2) and dst on j (dim 3), i.e.
            #   residual[b, r, i, j, :] = dst[b, r, j, :] - sign[r] * src[b, r, i, :]
            residual = dst[:, :, None, :, :] - signs[None, :, None, None, None] * src[:, :, :, None, :]
            weighted_residual = g[None, :, None, None, None] * w[..., None] * residual

            energy = weighted_residual.mul(residual).sum(dim=(2, 3, 4)) / w.sum(dim=(2, 3)).clamp_min(1.0)
            energy_chunks.append(energy)

            # src_back[b, r, v, u, t] = sum_s weighted_residual[...,s] * rho_src[r, t, s]
            src_back = torch.einsum("brvus,rts->brvut", weighted_residual, rho_src_c)
            dst_back = torch.einsum("brvus,rts->brvut", weighted_residual, rho_dst_c)

            z_update = z_update + (
                signs[None, :, None, None] * src_back.sum(dim=3)
            ).sum(dim=1) - dst_back.sum(dim=2).sum(dim=1)
            degree = degree + (g[None, :, None] * (w.sum(dim=3) + w.sum(dim=2))).sum(dim=1)

        z_update = eta * z_update / degree.unsqueeze(-1).clamp_min(1.0)
        h = self.norm(h + self.stalk_to_node(z_update) + self.node_mlp(h))
        energies = torch.cat(energy_chunks, dim=1)
        return h, energies, gates


class OrientedTacticalSheafFastNet(OrientedTacticalSheafNet):
    """i018 with FastSheafDiffusionBlock and optional torch.compile.

    Subclasses the original net, so adapter / incidence builder / encoder /
    triad pool / readout / forward logic are all inherited unchanged. Only the
    diffusion blocks are swapped and (optionally) the forward is compiled.
    """

    def __init__(
        self,
        *args: Any,
        chunk_size: int = 3,
        compile_model: bool = True,
        compile_mode: str = "reduce-overhead",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        depth = len(self.blocks)
        relation_count = len(RELATION_NAMES)
        # Rebuild diffusion blocks as the fast variant (same param structure).
        d_model = self.blocks[0].node_to_stalk.in_features
        stalk_dim = self.blocks[0].node_to_stalk.out_features
        dropout = self.blocks[0].node_mlp[3].p if len(self.blocks[0].node_mlp) > 3 else 0.1
        self.blocks = nn.ModuleList(
            [
                FastSheafDiffusionBlock(d_model, relation_count, stalk_dim, dropout, chunk_size=chunk_size)
                for _ in range(depth)
            ]
        )
        self._compiled_forward = None
        if compile_model:
            try:
                self._compiled_forward = torch.compile(self._raw_forward, mode=compile_mode)
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"[oriented_tactical_sheaf_fast] torch.compile unavailable, running eager: {exc}")
                self._compiled_forward = None

    def _raw_forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return OrientedTacticalSheafNet.forward(self, x)

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
        chunk_size=int(config.get("sheaf_chunk_size", 3)),
        compile_model=bool(config.get("compile_model", True)),
        compile_mode=str(config.get("compile_mode", "reduce-overhead")),
    )
