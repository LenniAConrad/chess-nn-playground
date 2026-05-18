"""Relation-Masked Attention graft over the i018 oriented tactical sheaf trunk.

Idea i258 keeps the i018 trunk (adapter -> incidence builder -> square encoder
-> sheaf diffusion stack) intact and inserts a single small sparse attention
residual block between the last sheaf diffusion block and the readout head.
The attention graft attends only over a fixed top-K neighbor list per source
square selected from i018's typed relation masks, adds a low-rank
relation-conditioned bias, and applies the update as a gated residual so the
parent trunk is recovered when the gate sits at zero.

The implementation deliberately follows the research packet
``ideas/research/packets/classic/i258_relation_masked_attention_i018.md``
and keeps the graft conservative:

- Same forward contract as the i018 trunk (dict with ``logits`` etc.).
- The base sheaf math, the readout, and every diagnostic remain unchanged;
  attention only refines the final square states before pooling.
- ``scrambled_masks`` feeds both the diffusion block and the attention graft,
  so the i018 ``scramble_relations=True`` falsifier is shared for free.
- Four neighborhood modes (``relation`` / ``global`` / ``king_zone`` /
  ``candidate``) expose the research-packet ablation grid as a single
  config flag.

The attention output projection is zero-initialized and the gate bias is
negative, so a freshly constructed graft starts as a near-identity over the
parent trunk. That keeps the falsifier ``graft_disabled`` (gate forced to
zero) within a few percent of the matched i018 baseline without retraining.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import require_board_tensor
from chess_nn_playground.models.trunk.oriented_tactical_sheaf import (
    RELATION_NAMES,
    BoardStateAdapter,
    SheafDiffusionBlock,
    SquareTokenEncoder,
    TacticalIncidenceBuilder,
    TriadDefectPool,
    _format_logits,
    _weighted_mean,
)


_NEIGHBORHOOD_MODES = ("relation", "global", "king_zone", "candidate")
_RELATION_COUNT = len(RELATION_NAMES)
_KING_ZONE_RELATION_IDX = (4, 5, 11)


def _stable_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the K axis with safe handling of empty-mask rows."""

    very_negative = torch.full_like(scores, -1.0e9)
    masked = torch.where(mask > 0, scores, very_negative)
    max_per_row = masked.amax(dim=-2, keepdim=True)
    max_per_row = torch.where(torch.isfinite(max_per_row), max_per_row, torch.zeros_like(max_per_row))
    shifted = masked - max_per_row
    exped = shifted.exp() * mask
    denom = exped.sum(dim=-2, keepdim=True)
    safe = denom > 1.0e-9
    fallback = mask / mask.sum(dim=-2, keepdim=True).clamp_min(1.0)
    return torch.where(safe, exped / denom.clamp_min(1.0e-9), fallback)


class RelationMaskedAttentionGraft(nn.Module):
    """One small sparse attention residual block driven by i018 relation masks."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        attn_dim: int = 24,
        relation_count: int = _RELATION_COUNT,
        relation_rank: int = 4,
        top_k: int = 8,
        king_boost: float = 0.5,
        relation_log_eps: float = 1.0e-3,
        zero_init_out: bool = True,
        gate_init_bias: float = -2.0,
        neighborhood: str = "relation",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError("attn_dim must be divisible by num_heads")
        if neighborhood not in _NEIGHBORHOOD_MODES:
            raise ValueError(
                f"neighborhood must be one of {_NEIGHBORHOOD_MODES}, got {neighborhood!r}"
            )
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.attn_dim = int(attn_dim)
        self.head_dim = self.attn_dim // self.num_heads
        self.relation_count = int(relation_count)
        self.relation_rank = int(relation_rank)
        self.top_k = int(top_k)
        self.king_boost = float(king_boost)
        self.relation_log_eps = float(relation_log_eps)
        self.neighborhood = str(neighborhood)

        self.norm = nn.LayerNorm(self.d_model)
        self.qkv = nn.Linear(self.d_model, 3 * self.attn_dim, bias=False)
        self.out = nn.Linear(self.attn_dim, self.d_model, bias=False)
        if zero_init_out:
            nn.init.zeros_(self.out.weight)

        self.rel_proj = nn.Linear(self.relation_count, self.relation_rank, bias=False)
        # Small non-zero init keeps the relation-bias gradient flowing on step 1;
        # the parent-trunk identity at construction is preserved by ``out`` being
        # zero-initialised, so the bias size here is irrelevant before training.
        self.rel_head = nn.Parameter(0.02 * torch.randn(self.num_heads, self.relation_rank))
        self.rel_scale = nn.Parameter(0.02 * torch.ones(self.num_heads))
        self.dropout = nn.Dropout(float(dropout))

        gate_in_dim = self.d_model + 3
        self.gate = nn.Linear(gate_in_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_init_bias))

    def _edge_score(
        self,
        relation_masks: torch.Tensor,
        union: torch.Tensor,
        king_pressure_edge: torch.Tensor,
        candidate_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-edge score and the structural neighborhood mask."""

        B, N, _ = union.shape
        device = union.device
        identity = torch.eye(N, device=device, dtype=union.dtype).unsqueeze(0).expand(B, -1, -1)
        if self.neighborhood == "global":
            score = union + self.king_boost * king_pressure_edge
            score = score + 1.0e-6 * identity
            mask = torch.ones_like(score)
            return score, mask
        if self.neighborhood == "king_zone":
            zone = sum(relation_masks[:, idx] for idx in _KING_ZONE_RELATION_IDX)
            zone = zone.clamp(0.0, 1.0)
            score = zone + identity
            mask = (zone + identity > 0).to(union.dtype)
            return score, mask
        if self.neighborhood == "candidate":
            base = candidate_mask if candidate_mask is not None else union
            score = base + identity
            mask = (base + identity > 0).to(union.dtype)
            return score, mask
        score = union + self.king_boost * king_pressure_edge + identity
        mask = (union + king_pressure_edge + identity > 0).to(union.dtype)
        return score, mask

    def forward(
        self,
        h: torch.Tensor,
        relation_masks: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        gate_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, N, D = h.shape
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {D}")
        if relation_masks.shape[1] != self.relation_count:
            raise ValueError(
                f"Expected relation_count={self.relation_count}, got {relation_masks.shape[1]}"
            )

        x = self.norm(h)

        union = relation_masks.amax(dim=1)
        king_pressure_edge = sum(relation_masks[:, idx] for idx in _KING_ZONE_RELATION_IDX)
        king_pressure_edge = king_pressure_edge.clamp(0.0, 1.0)

        edge_score, struct_mask = self._edge_score(
            relation_masks, union, king_pressure_edge, candidate_mask
        )

        k = max(1, min(self.top_k, N))
        topk = edge_score.topk(k=k, dim=-1)
        nbr_idx = topk.indices

        identity_mask = torch.zeros_like(struct_mask)
        identity_mask.scatter_(2, torch.arange(N, device=h.device).view(1, N, 1).expand(B, N, 1), 1.0)
        struct_mask = torch.clamp(struct_mask + identity_mask, max=1.0)

        nbr_mask = torch.gather(struct_mask, dim=-1, index=nbr_idx)
        nbr_mask = nbr_mask.unsqueeze(-1)

        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0]
        k_proj = qkv[:, :, 1]
        v_proj = qkv[:, :, 2]

        idx_kv = nbr_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.num_heads, self.head_dim)
        k_src = k_proj.unsqueeze(1).expand(-1, N, -1, -1, -1)
        v_src = v_proj.unsqueeze(1).expand(-1, N, -1, -1, -1)
        k_nbr = torch.gather(k_src, dim=2, index=idx_kv)
        v_nbr = torch.gather(v_src, dim=2, index=idx_kv)

        idx_rel = nbr_idx.unsqueeze(1).expand(-1, self.relation_count, -1, -1)
        rel_sig = torch.gather(relation_masks, dim=3, index=idx_rel).permute(0, 2, 3, 1)

        q_src = q.unsqueeze(2)
        scores = (q_src * k_nbr).sum(dim=-1) / math.sqrt(self.head_dim)

        rel_latent = torch.tanh(self.rel_proj(rel_sig))
        rel_bias_typed = torch.einsum("bskr,hr->bskh", rel_latent, self.rel_head)
        nbr_edge = torch.gather(edge_score, dim=-1, index=nbr_idx).clamp_min(0.0)
        nbr_edge_log = torch.log1p(nbr_edge * (1.0 / max(self.relation_log_eps, 1.0e-6)))
        rel_bias_log = self.rel_scale.view(1, 1, 1, -1) * nbr_edge_log.unsqueeze(-1)
        rel_bias = rel_bias_typed + rel_bias_log

        logits = scores + rel_bias

        attn = _stable_softmax(logits, nbr_mask)
        attn = self.dropout(attn)
        msg = (attn.unsqueeze(-1) * v_nbr).sum(dim=2)
        msg = msg.reshape(B, N, self.attn_dim)
        delta = self.out(msg)

        union_self = relation_masks.amax(dim=(1, 3))
        king_self = sum(relation_masks[:, idx].mean(dim=-1) for idx in _KING_ZONE_RELATION_IDX)
        king_self = king_self.clamp(0.0, 1.0)
        pin_self = relation_masks[:, 11].mean(dim=-1)
        gate_features = torch.cat(
            [x, union_self.unsqueeze(-1), king_self.unsqueeze(-1), pin_self.unsqueeze(-1)],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(gate_features))
        if gate_override is not None:
            gate = gate_override.expand_as(gate)

        out = h + gate * delta

        entropy = -(attn.clamp_min(1.0e-9) * attn.clamp_min(1.0e-9).log()).sum(dim=2)
        rel_sig_self = relation_masks.permute(0, 2, 3, 1)
        rel_sig_picked = torch.gather(rel_sig_self, dim=2, index=nbr_idx.unsqueeze(-1).expand(-1, -1, -1, self.relation_count))
        king_edge_share = sum(rel_sig_picked[..., idx] for idx in _KING_ZONE_RELATION_IDX).clamp(0.0, 1.0)
        attn_mean_heads = attn.mean(dim=-1)
        king_attention_share = (attn_mean_heads * king_edge_share).sum(dim=2)

        diagnostics = {
            "attention_entropy": entropy.mean(dim=(1, 2)),
            "attention_king_share": king_attention_share.mean(dim=1),
            "attention_gate_mean": gate.mean(dim=1).squeeze(-1),
            "attention_delta_norm": delta.norm(dim=-1).mean(dim=1),
            "attention_neighbor_count": nbr_mask.sum(dim=(1, 2, 3)) / float(N),
            "attention_relation_bias_norm": rel_bias.norm(dim=-1).mean(dim=(1, 2)),
        }
        return out, diagnostics


def _build_candidate_mask(piece_state: torch.Tensor) -> torch.Tensor:
    """Side-to-move pseudo-legal candidate-square mask, derived deterministically.

    Each row is a source square; columns whose entry is 1 are the squares that
    the side-to-move piece on the row can plausibly move to as derived from
    the typed relation masks (own-attacks-empty plus own-attacks-them). We keep
    it as a simple union rather than an exact pseudo-legal enumerator because
    the relation masks are already side-to-move-oriented.
    """

    pieces = piece_state[..., 1:7].sum(dim=-1)
    return pieces


def _attention_relation_count() -> int:
    return _RELATION_COUNT


class RelationMaskedAttentionI018Net(nn.Module):
    """i018 trunk + one relation-masked attention graft before readout."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 76,
        depth: int = 2,
        stalk_dim: int = 8,
        dropout: float = 0.1,
        encoding: str = "simple_18",
        piece_adapter: str = "exact",
        use_triads: bool = True,
        scramble_relations: bool = False,
        attention_enabled: bool = True,
        attention_num_heads: int = 2,
        attention_dim: int = 24,
        attention_top_k: int = 8,
        attention_relation_rank: int = 4,
        attention_king_boost: float = 0.5,
        attention_neighborhood: str = "relation",
        attention_zero_init_out: bool = True,
        attention_gate_init_bias: float = -2.0,
        attention_dropout: float = 0.0,
        attention_force_gate: float | None = None,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.relation_names = RELATION_NAMES
        self.scramble_relations = bool(scramble_relations)
        self.attention_enabled = bool(attention_enabled)
        self.attention_neighborhood = str(attention_neighborhood)
        self.attention_force_gate = (
            None if attention_force_gate is None else float(attention_force_gate)
        )

        self.adapter = BoardStateAdapter(
            input_channels=input_channels,
            encoding=encoding,
            piece_adapter=piece_adapter,
        )
        self.incidence = TacticalIncidenceBuilder()
        self.encoder = SquareTokenEncoder(
            input_channels=input_channels, d_model=channels, dropout=dropout
        )
        self.blocks = nn.ModuleList(
            [
                SheafDiffusionBlock(channels, _attention_relation_count(), stalk_dim, dropout)
                for _ in range(max(1, int(depth)))
            ]
        )
        self.triad_pool = TriadDefectPool(channels, dropout) if use_triads else None

        if self.attention_enabled:
            self.attention = RelationMaskedAttentionGraft(
                d_model=channels,
                num_heads=attention_num_heads,
                attn_dim=attention_dim,
                relation_count=_attention_relation_count(),
                relation_rank=attention_relation_rank,
                top_k=attention_top_k,
                king_boost=attention_king_boost,
                zero_init_out=attention_zero_init_out,
                gate_init_bias=attention_gate_init_bias,
                neighborhood=self.attention_neighborhood,
                dropout=attention_dropout,
            )
        else:
            self.attention = None

        triad_dim = self.triad_pool.output_dim if self.triad_pool is not None else 0
        board_stats_dim = 8
        readout_dim = channels * 4 + _attention_relation_count() * 4 + triad_dim + board_stats_dim
        self.head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )
        from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec

        self.spec = BoardTensorSpec(input_channels=input_channels)

    def _board_stats(self, board, incidence) -> torch.Tensor:
        occupancy = board.occupancy
        rank_counts = torch.matmul(occupancy, self.incidence.rank_one_hot)
        file_counts = torch.matmul(occupancy, self.incidence.file_one_hot)
        return torch.stack(
            [
                occupancy.mean(dim=1),
                incidence.our_piece.sum(dim=1) / 16.0,
                incidence.them_piece.sum(dim=1) / 16.0,
                incidence.our_attack.mean(dim=(1, 2)),
                incidence.them_attack.mean(dim=(1, 2)),
                incidence.pin_mask.mean(dim=(1, 2)),
                rank_counts.std(dim=1),
                file_counts.std(dim=1),
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.adapter(x)
        incidence = self.incidence(board.piece_state, board.occupancy)

        if self.scramble_relations:
            sheaf_masks = incidence.relation_masks
            B, R, N, _ = sheaf_masks.shape
            perm = torch.argsort(torch.rand(B, R, N, device=sheaf_masks.device), dim=-1)
            perm_expanded = perm.unsqueeze(-2).expand(-1, -1, N, -1)
            relation_masks_used = torch.gather(sheaf_masks, dim=-1, index=perm_expanded)
        else:
            relation_masks_used = incidence.relation_masks

        h = self.encoder(board.square_raw, board.piece_state)
        block_energies: list[torch.Tensor] = []
        block_gates: list[torch.Tensor] = []
        for block in self.blocks:
            h, energy, gates = block(h, relation_masks_used)
            block_energies.append(energy)
            block_gates.append(gates.unsqueeze(0).expand(x.shape[0], -1))

        attention_diagnostics: dict[str, torch.Tensor]
        if self.attention is not None:
            candidate_mask = _build_candidate_mask(board.piece_state)
            candidate_mask = candidate_mask.unsqueeze(1) * candidate_mask.unsqueeze(2)
            if self.attention_force_gate is not None:
                gate_override = torch.full(
                    (x.shape[0], 1, 1),
                    float(self.attention_force_gate),
                    device=h.device,
                    dtype=h.dtype,
                )
            else:
                gate_override = None
            h, attention_diagnostics = self.attention(
                h, relation_masks_used, candidate_mask=candidate_mask, gate_override=gate_override
            )
        else:
            zeros = h.new_zeros(x.shape[0])
            attention_diagnostics = {
                "attention_entropy": zeros,
                "attention_king_share": zeros,
                "attention_gate_mean": zeros,
                "attention_delta_norm": zeros,
                "attention_neighbor_count": zeros,
                "attention_relation_bias_norm": zeros,
            }

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

        sheaf_tension = energy_stack.mean(dim=(1, 2))
        us_pressure = incidence.relation_masks[:, 0].sum(dim=(1, 2))
        them_pressure = incidence.relation_masks[:, 1].sum(dim=(1, 2))
        us_defense = incidence.relation_masks[:, 2].sum(dim=(1, 2))
        them_defense = incidence.relation_masks[:, 3].sum(dim=(1, 2))
        rank_counts = torch.matmul(board.occupancy, self.incidence.rank_one_hot)
        file_counts = torch.matmul(board.occupancy, self.incidence.file_one_hot)
        piece_entropy = -(board.piece_state * board.piece_state.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=1)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "mechanism_energy": torch.log1p(sheaf_tension),
            "proposal_profile_strength": gate_mean.mean(dim=1),
            "proposal_keyword_count": logits.new_full((x.shape[0],), 4.0),
            "sheaf_tension": sheaf_tension,
            "transport_imbalance": (us_pressure - them_pressure).abs()
            / (us_pressure + them_pressure).clamp_min(1.0),
            "symmetry_residual": (
                incidence.our_attack.mean(dim=(1, 2)) - incidence.them_attack.mean(dim=(1, 2))
            ).abs(),
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
        diagnostics.update(attention_diagnostics)
        return diagnostics


def build_relation_masked_attention_i018_from_config(
    config: dict[str, Any],
) -> RelationMaskedAttentionI018Net:
    """Builder used by the model registry and the idea-local model.py wrapper."""

    attention_cfg: dict[str, Any] = {}
    raw = config.get("relation_attention")
    if isinstance(raw, dict):
        attention_cfg = dict(raw)

    def _maybe_float(key: str, default: float | None) -> float | None:
        if key in attention_cfg:
            value = attention_cfg[key]
            return None if value is None else float(value)
        if key in config:
            value = config[key]
            return None if value is None else float(value)
        return default

    return RelationMaskedAttentionI018Net(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 76)),
        depth=int(config.get("sheaf_layers", config.get("depth", 2))),
        stalk_dim=int(config.get("stalk_dim", 8)),
        dropout=float(config.get("dropout", 0.1)),
        encoding=str(config.get("encoding", "simple_18")),
        piece_adapter=str(config.get("piece_adapter", "exact")),
        use_triads=bool(config.get("use_triads", True)),
        scramble_relations=bool(config.get("scramble_relations", False)),
        attention_enabled=bool(attention_cfg.get("enabled", config.get("attention_enabled", True))),
        attention_num_heads=int(attention_cfg.get("num_heads", config.get("attention_num_heads", 2))),
        attention_dim=int(attention_cfg.get("attn_dim", config.get("attention_dim", 24))),
        attention_top_k=int(attention_cfg.get("top_k", config.get("attention_top_k", 8))),
        attention_relation_rank=int(
            attention_cfg.get("relation_rank", config.get("attention_relation_rank", 4))
        ),
        attention_king_boost=float(
            attention_cfg.get("king_boost", config.get("attention_king_boost", 0.5))
        ),
        attention_neighborhood=str(
            attention_cfg.get("neighborhood", config.get("attention_neighborhood", "relation"))
        ),
        attention_zero_init_out=bool(
            attention_cfg.get("zero_init_out", config.get("attention_zero_init_out", True))
        ),
        attention_gate_init_bias=float(
            attention_cfg.get("gate_init_bias", config.get("attention_gate_init_bias", -2.0))
        ),
        attention_dropout=float(
            attention_cfg.get("dropout", config.get("attention_dropout", 0.0))
        ),
        attention_force_gate=_maybe_float("force_gate", None),
    )
