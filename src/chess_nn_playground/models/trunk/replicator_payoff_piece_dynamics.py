"""Replicator Payoff Piece Dynamics (idea i131).

A small differentiable game over occupied piece tokens. A learned pairwise
payoff matrix split across role heads drives a few replicator-dynamics steps,
and the equilibrium / instability statistics of the resulting per-head
populations are fused with a compact CNN board summary to predict the
``puzzle_binary`` logit.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.trunk.idea_blocks import BoardConvStem, BoardTensorSpec, require_board_tensor


_MASK_NEG = 1.0e9


class OccupiedPieceTokenizer(nn.Module):
    """Build per-square raw tokens, an occupancy mask, and pairwise geometry.

    Tokens cover the full 64-square grid and are then top-k selected down to
    ``max_pieces`` slots ordered by occupancy. Padded slots carry mask 0.
    """

    raw_dim = 23

    def __init__(self, max_pieces: int = 32) -> None:
        super().__init__()
        if max_pieces < 2 or max_pieces > 64:
            raise ValueError("max_pieces must be in [2, 64]")
        self.max_pieces = max_pieces

        rank_idx = torch.arange(8).view(8, 1).expand(8, 8).contiguous().float().flatten()
        file_idx = torch.arange(8).view(1, 8).expand(8, 8).contiguous().float().flatten()
        self.register_buffer("rank_grid", rank_idx, persistent=False)
        self.register_buffer("file_grid", file_idx, persistent=False)

        df = file_idx.unsqueeze(0) - file_idx.unsqueeze(1)
        dr = rank_idx.unsqueeze(0) - rank_idx.unsqueeze(1)
        cheb = torch.maximum(df.abs(), dr.abs())
        l1 = df.abs() + dr.abs()
        same_file = (df == 0).float()
        same_rank = (dr == 0).float()
        diag_pos = ((df == dr) & (df != 0)).float()
        diag_neg = ((df == -dr) & (df != 0)).float()
        knight = (((df.abs() == 1) & (dr.abs() == 2)) | ((df.abs() == 2) & (dr.abs() == 1))).float()
        geometry = torch.stack(
            [
                df / 7.0,
                dr / 7.0,
                cheb / 7.0,
                l1 / 14.0,
                same_file,
                same_rank,
                diag_pos,
                diag_neg,
                knight,
            ],
            dim=-1,
        )
        self.register_buffer("geometry", geometry, persistent=False)
        self.geometry_dim = geometry.shape[-1]

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        flat = x.flatten(2)  # (B, 18, 64)
        own_planes = flat[:, 0:6]
        opp_planes = flat[:, 6:12]
        own_occ = own_planes.sum(dim=1)
        opp_occ = opp_planes.sum(dim=1)
        occupied = (own_occ + opp_occ).clamp(0.0, 1.0)

        piece_ohe = torch.cat([own_planes, opp_planes], dim=1).transpose(1, 2)  # (B, 64, 12)
        color_flag = (opp_occ - own_occ).unsqueeze(-1)
        rank_norm = ((self.rank_grid - 3.5) / 3.5).expand(batch, -1).unsqueeze(-1)
        file_norm = ((self.file_grid - 3.5) / 3.5).expand(batch, -1).unsqueeze(-1)

        own_king = own_planes[:, 5]
        opp_king = opp_planes[:, 5]
        own_king_count = own_king.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        opp_king_count = opp_king.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        own_king_rank = (own_king * self.rank_grid).sum(dim=1, keepdim=True) / own_king_count
        own_king_file = (own_king * self.file_grid).sum(dim=1, keepdim=True) / own_king_count
        opp_king_rank = (opp_king * self.rank_grid).sum(dim=1, keepdim=True) / opp_king_count
        opp_king_file = (opp_king * self.file_grid).sum(dim=1, keepdim=True) / opp_king_count

        rank_full = self.rank_grid.expand(batch, -1)
        file_full = self.file_grid.expand(batch, -1)
        dist_own_king = torch.maximum(
            (rank_full - own_king_rank).abs(), (file_full - own_king_file).abs()
        ).unsqueeze(-1) / 7.0
        dist_opp_king = torch.maximum(
            (rank_full - opp_king_rank).abs(), (file_full - opp_king_file).abs()
        ).unsqueeze(-1) / 7.0

        stm = (x[:, 12, 0, 0] - 0.5).view(batch, 1, 1).expand(-1, 64, 1)
        castling = x[:, 13:17].sum(dim=1).flatten(1).clamp(0.0, 4.0).unsqueeze(-1) / 4.0
        en_passant = x[:, 17].flatten(1).clamp(0.0, 1.0).unsqueeze(-1)

        center_l1 = rank_norm.abs() + file_norm.abs()
        center_cheb = torch.maximum(rank_norm.abs(), file_norm.abs())
        occ_feat = occupied.unsqueeze(-1)

        token_raw = torch.cat(
            [
                piece_ohe,
                color_flag,
                rank_norm,
                file_norm,
                dist_own_king,
                dist_opp_king,
                stm,
                castling,
                en_passant,
                center_l1,
                center_cheb,
                occ_feat,
            ],
            dim=-1,
        )

        sort_idx = torch.argsort(occupied, dim=-1, descending=True, stable=True)
        sel_idx = sort_idx[:, : self.max_pieces].contiguous()  # (B, Pmax)

        gather_idx = sel_idx.unsqueeze(-1).expand(-1, -1, token_raw.shape[-1])
        tokens = token_raw.gather(1, gather_idx)
        mask = occupied.gather(1, sel_idx)

        i_idx = sel_idx.unsqueeze(2).expand(batch, self.max_pieces, self.max_pieces)
        j_idx = sel_idx.unsqueeze(1).expand(batch, self.max_pieces, self.max_pieces)
        pair_geometry = self.geometry[i_idx, j_idx]

        return tokens, mask, pair_geometry, sel_idx


class ReplicatorPayoffPieceDynamicsNetwork(nn.Module):
    """Bespoke implementation of the replicator-payoff piece-dynamics architecture."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        max_pieces: int = 32,
        token_dim: int = 64,
        pair_hidden_dim: int = 64,
        num_heads: int = 4,
        num_steps: int = 5,
        eta: float = 0.5,
        encoding_adapter: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if input_channels != 18 or encoding_adapter != SIMPLE_18:
            raise ValueError(
                "ReplicatorPayoffPieceDynamicsNetwork currently supports simple_18 with 18 input channels"
            )
        if num_classes != 1:
            raise ValueError(
                "ReplicatorPayoffPieceDynamicsNetwork supports the puzzle_binary one-logit contract"
            )
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if token_dim < 4:
            raise ValueError("token_dim must be >= 4")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.max_pieces = max_pieces
        self.num_heads = num_heads
        self.num_steps = num_steps

        self.tokenizer = OccupiedPieceTokenizer(max_pieces=max_pieces)
        self.token_proj = nn.Sequential(
            nn.Linear(self.tokenizer.raw_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        pair_input_dim = token_dim * 2 + self.tokenizer.geometry_dim
        self.payoff_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, pair_hidden_dim),
            nn.GELU(),
            nn.Linear(pair_hidden_dim, num_heads),
        )

        self.init_logits = nn.Linear(token_dim, num_heads)
        self.head_eta = nn.Parameter(torch.full((num_heads,), float(eta)))
        self.head_pool_weight = nn.Parameter(torch.zeros(num_heads))

        self.backbone = BoardConvStem(
            input_channels=input_channels,
            channels=channels,
            depth=max(1, depth),
            use_batchnorm=use_batchnorm,
        )
        backbone_pool_dim = channels * 2

        self._stats_per_head = 11
        stats_dim = num_heads * self._stats_per_head + num_heads + 2

        layers: list[nn.Module] = [
            nn.Linear(backbone_pool_dim + stats_dim, hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def _replicator(
        self,
        payoff: torch.Tensor,
        mask_expand: torch.Tensor,
        init_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        masked_logits = init_scores + (mask_expand - 1.0) * _MASK_NEG
        log_p = torch.log_softmax(masked_logits, dim=-1)
        initial_p = log_p.exp()

        avg_history: list[torch.Tensor] = []
        eta = self.head_eta.view(1, self.num_heads, 1)
        for _ in range(self.num_steps):
            p = log_p.exp()
            fitness = torch.einsum("bhij,bhj->bhi", payoff, p)
            avg = (p * fitness).sum(dim=-1, keepdim=True)
            log_p = log_p + eta * (fitness - avg)
            log_p = log_p + (mask_expand - 1.0) * _MASK_NEG
            log_p = torch.log_softmax(log_p, dim=-1)
            avg_history.append(avg.squeeze(-1))

        return log_p.exp(), initial_p, log_p, avg_history

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]
        pmax = self.max_pieces

        token_raw, mask, pair_geometry, _ = self.tokenizer(x)
        tokens = self.token_proj(token_raw)

        ti = tokens.unsqueeze(2).expand(batch, pmax, pmax, tokens.shape[-1])
        tj = tokens.unsqueeze(1).expand(batch, pmax, pmax, tokens.shape[-1])
        pair_input = torch.cat([ti, tj, pair_geometry], dim=-1)
        payoff = self.payoff_mlp(pair_input).permute(0, 3, 1, 2).contiguous()

        init_scores = self.init_logits(tokens).permute(0, 2, 1)
        mask_expand = mask.unsqueeze(1).expand(-1, self.num_heads, -1)

        final_p, initial_p, _, avg_history = self._replicator(payoff, mask_expand, init_scores)

        eps = 1.0e-8
        log_final = final_p.clamp_min(eps).log()
        entropy = -(final_p * log_final).sum(dim=-1)
        top_mass = final_p.amax(dim=-1)
        log_initial = initial_p.clamp_min(eps).log()
        kl_from_initial = (final_p * (log_final - log_initial)).sum(dim=-1)

        final_fitness = torch.einsum("bhij,bhj->bhi", payoff, final_p)
        avg_payoff = (final_p * final_fitness).sum(dim=-1)
        fitness_variance = ((final_fitness - avg_payoff.unsqueeze(-1)) ** 2 * final_p).sum(dim=-1)

        piece_ohe = token_raw[..., :12]
        is_own = piece_ohe[..., :6].sum(dim=-1)
        is_opp = piece_ohe[..., 6:12].sum(dim=-1)
        is_king = piece_ohe[..., 5] + piece_ohe[..., 11]
        is_pawn = piece_ohe[..., 0] + piece_ohe[..., 6]
        is_minor = piece_ohe[..., 1] + piece_ohe[..., 2] + piece_ohe[..., 7] + piece_ohe[..., 8]
        is_major = piece_ohe[..., 3] + piece_ohe[..., 4] + piece_ohe[..., 9] + piece_ohe[..., 10]

        def _mass_on(indicator: torch.Tensor) -> torch.Tensor:
            return (final_p * indicator.unsqueeze(1)).sum(dim=-1)

        own_mass = _mass_on(is_own)
        opp_mass = _mass_on(is_opp)
        king_mass = _mass_on(is_king)
        pawn_mass = _mass_on(is_pawn)
        minor_mass = _mass_on(is_minor)
        major_mass = _mass_on(is_major)

        per_head_stats = torch.stack(
            [
                entropy,
                top_mass,
                kl_from_initial,
                avg_payoff,
                fitness_variance,
                own_mass,
                opp_mass,
                king_mass,
                pawn_mass,
                minor_mass,
                major_mass,
            ],
            dim=-1,
        )
        stats_flat = per_head_stats.flatten(1)

        head_pool = torch.softmax(self.head_pool_weight, dim=0).unsqueeze(0).expand(batch, -1)
        total_pieces = mask.sum(dim=-1, keepdim=True)
        payoff_asymmetry = (payoff - payoff.transpose(-1, -2)).flatten(1).norm(dim=-1, keepdim=True)
        stats_vec = torch.cat([stats_flat, head_pool, total_pieces / 32.0, payoff_asymmetry / 32.0], dim=-1)

        backbone_features = self.backbone(x)
        pooled = torch.cat(
            [backbone_features.mean(dim=(2, 3)), backbone_features.amax(dim=(2, 3))], dim=1
        )

        logits = self.classifier(torch.cat([pooled, stats_vec], dim=1)).squeeze(-1)

        diagnostics: dict[str, torch.Tensor] = {"logits": logits}
        for h in range(self.num_heads):
            diagnostics[f"head{h}_entropy"] = entropy[:, h]
            diagnostics[f"head{h}_top_mass"] = top_mass[:, h]
            diagnostics[f"head{h}_kl_from_initial"] = kl_from_initial[:, h]
            diagnostics[f"head{h}_avg_payoff"] = avg_payoff[:, h]
            diagnostics[f"head{h}_fitness_variance"] = fitness_variance[:, h]
            diagnostics[f"head{h}_own_mass"] = own_mass[:, h]
            diagnostics[f"head{h}_opp_mass"] = opp_mass[:, h]
            diagnostics[f"head{h}_king_mass"] = king_mass[:, h]
            diagnostics[f"head{h}_pawn_mass"] = pawn_mass[:, h]
            diagnostics[f"head{h}_minor_mass"] = minor_mass[:, h]
            diagnostics[f"head{h}_major_mass"] = major_mass[:, h]
            diagnostics[f"head{h}_avg_payoff_step{len(avg_history) - 1}"] = avg_history[-1][:, h]
        diagnostics["mean_entropy"] = entropy.mean(dim=-1)
        diagnostics["mean_top_mass"] = top_mass.mean(dim=-1)
        diagnostics["mean_kl_from_initial"] = kl_from_initial.mean(dim=-1)
        diagnostics["mean_avg_payoff"] = avg_payoff.mean(dim=-1)
        diagnostics["mean_fitness_variance"] = fitness_variance.mean(dim=-1)
        diagnostics["payoff_asymmetry_norm"] = payoff_asymmetry.squeeze(-1)
        diagnostics["total_piece_count"] = mask.sum(dim=-1)
        diagnostics["backbone_feature_norm"] = backbone_features.flatten(1).norm(dim=1)
        return diagnostics


def build_replicator_payoff_piece_dynamics_from_config(
    config: dict[str, Any],
) -> ReplicatorPayoffPieceDynamicsNetwork:
    return ReplicatorPayoffPieceDynamicsNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        max_pieces=int(config.get("max_pieces", 32)),
        token_dim=int(config.get("token_dim", 64)),
        pair_hidden_dim=int(config.get("pair_hidden_dim", 64)),
        num_heads=int(config.get("num_heads", 4)),
        num_steps=int(config.get("num_steps", 5)),
        eta=float(config.get("eta", 0.5)),
        encoding_adapter=str(config.get("encoding_adapter", config.get("encoding", SIMPLE_18))),
    )
