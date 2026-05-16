"""Legal-Edge Compile-Scatter spatial mixer (p011).

The p011 primitive compiles a *typed* legal-move adjacency ``A_r`` from the
board, computes a per-edge sigmoid gate ``g_{r,i,j} = A_r * sigma(a_r . [x_i,
x_j])``, projects per-type source messages ``m_{r,i,j} = W_r x_i`` and emits a
degree-normalised scatter ``y_j = sum_r (1/deg) sum_i g_{r,i,j} m_{r,i,j}``.

As a ``(B, C, 8, 8) -> (B, C, 8, 8)`` spatial mixer the 64 board squares are
the tokens. The core operator is reproduced faithfully:

- The typed adjacency ``A_r`` is the load-bearing rule structure. Here it is
  built once from pure 8x8 board GEOMETRY: one channel per move pattern
  (knight, king, rook-rays, bishop-rays, white/black pawn pushes). This is
  the occupancy-free skeleton of ``_compute_typed_legal_edges`` -- the mixer
  receives generic ``C``-channel features, not a simple_18 board, so the
  content-conditioned occupancy masking of the original cannot be recovered.
  The deterministic-rule-structure / stop-gradient property is preserved
  (``A_r`` is a registered buffer, no gradient).
- The per-edge sigmoid gate from the (src, dst) feature pair, the per-type
  message projection ``W_r``, the gate-weighted bmm scatter and the typed
  in-degree normalisation are all kept verbatim in spirit.

Compromise: "typed" = move-pattern type rather than the original's
piece-occupancy-conditioned legal edges, because channel ``C`` is arbitrary
and carries no piece-plane semantics. The sigma-gate is therefore the only
content-conditioned term, exactly as in the ``no_edge_gate`` boundary case
of the source -- except here it is always on.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer


def _geometric_typed_edges() -> torch.Tensor:
    """Return ``(R, 64, 64)`` 0/1 move-pattern adjacency, R move types."""
    ranks = torch.arange(64) // 8
    files = torch.arange(64) % 8
    dr = ranks.view(64, 1) - ranks.view(1, 64)
    df = files.view(64, 1) - files.view(1, 64)
    adr, adf = dr.abs(), df.abs()
    same = (dr == 0) & (df == 0)

    knight = ((adr == 1) & (adf == 2)) | ((adr == 2) & (adf == 1))
    king = (adr <= 1) & (adf <= 1) & ~same
    rook = ((dr == 0) | (df == 0)) & ~same
    bishop = (adr == adf) & ~same
    wpawn = (dr == 1) & (adf <= 1)  # forward-ish for one colour
    bpawn = (dr == -1) & (adf <= 1)

    edges = torch.stack(
        [knight, king, rook, bishop, wpawn, bpawn], dim=0
    ).to(dtype=torch.float32)
    return edges  # (6, 64, 64)


class LegalEdgeCompileScatterMixer(nn.Module):
    def __init__(self, channels: int, message_dim: int = 32, edge_gate_hidden: int = 16) -> None:
        super().__init__()
        edges = _geometric_typed_edges()
        self.register_buffer("edges", edges, persistent=False)
        self.num_types = edges.shape[0]

        self.in_norm = nn.LayerNorm(channels)
        self.edge_gate_mlps = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(2 * channels),
                nn.Linear(2 * channels, edge_gate_hidden),
                nn.GELU(),
                nn.Linear(edge_gate_hidden, 1),
            )
            for _ in range(self.num_types)
        )
        self.message_linears = nn.ModuleList(
            nn.Linear(channels, message_dim) for _ in range(self.num_types)
        )
        self.message_norm = nn.LayerNorm(message_dim)
        self.out_proj = nn.Linear(message_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        tokens = self.in_norm(x.flatten(2).transpose(1, 2))  # (B, 64, C)

        edges = self.edges  # (R, 64, 64)
        x_src = tokens.unsqueeze(2).expand(b, n, n, c)
        x_dst = tokens.unsqueeze(1).expand(b, n, n, c)
        edge_in = torch.cat([x_src, x_dst], dim=-1)  # (B, 64, 64, 2C)

        msg_sum = tokens.new_zeros(b, n, self.out_proj.in_features)
        for r in range(self.num_types):
            mask = edges[r].unsqueeze(0)  # (1, 64, 64)
            gate = torch.sigmoid(self.edge_gate_mlps[r](edge_in).squeeze(-1)) * mask
            proj = self.message_linears[r](tokens)  # (B, 64, m)
            # y_dst = sum_src gate[src,dst] * proj[src]
            scattered = torch.bmm(gate.transpose(1, 2), proj)  # (B, 64_dst, m)
            deg = gate.sum(dim=1).unsqueeze(-1).clamp_min(1.0e-3)  # (B, 64_dst, 1)
            msg_sum = msg_sum + scattered / deg

        msgs = self.message_norm(msg_sum)
        out = self.out_proj(msgs)  # (B, 64, C)
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("legal_edge_compile_scatter")
def build(channels: int, message_dim: int = 32, edge_gate_hidden: int = 16, **_: object) -> nn.Module:
    return LegalEdgeCompileScatterMixer(
        channels=channels, message_dim=message_dim, edge_gate_hidden=edge_gate_hidden
    )
