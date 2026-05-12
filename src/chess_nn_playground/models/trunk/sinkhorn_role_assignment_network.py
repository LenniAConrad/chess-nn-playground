"""Sinkhorn Role Assignment Network for idea i120.

Extracts up to ``Pmax`` occupied piece tokens from the simple_18 board tensor
and assigns each token to one of ``M`` learned tactical role slots (plus an
explicit dustbin role) by computing a doubly-stochastic transport matrix
through ``T`` log-domain Sinkhorn iterations. Role vectors are pooled by the
transport matrix, mixed through a small pairwise interaction MLP, and fused
with a light board CNN summary before classifying with a single puzzle logit.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


PIECE_PLANES = 12
GLOBAL_PLANES = 6
DEFAULT_MAX_TOKENS = 32
TOKEN_COORD_FEATURES = 6
LOCAL_OCCUPANCY_FEATURES = 4


def _board_coords() -> torch.Tensor:
    rank = torch.arange(8, dtype=torch.float32).view(8, 1).expand(8, 8)
    file = torch.arange(8, dtype=torch.float32).view(1, 8).expand(8, 8)
    centered_rank = (rank - 3.5) / 3.5
    centered_file = (file - 3.5) / 3.5
    edge_distance = torch.minimum(
        torch.minimum(rank, 7.0 - rank), torch.minimum(file, 7.0 - file)
    ) / 3.5
    square_color = ((rank + file).remainder(2.0) * 2.0) - 1.0
    return torch.stack(
        [
            rank / 7.0,
            file / 7.0,
            centered_rank,
            centered_file,
            edge_distance,
            square_color,
        ],
        dim=-1,
    ).view(64, TOKEN_COORD_FEATURES)


def _select_occupied_tokens(
    square_features: torch.Tensor, occupancy: torch.Tensor, max_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    b, num_squares, _ = square_features.shape
    occ_int = occupancy.to(torch.int64)
    rank_within = torch.arange(num_squares, device=square_features.device).expand(b, -1)
    sort_key = (1 - occ_int) * (num_squares + 1) + rank_within
    _, ordered = torch.sort(sort_key, dim=-1, stable=True)
    ordered = ordered[:, :max_tokens]
    gather_idx = ordered.unsqueeze(-1).expand(-1, -1, square_features.shape[-1])
    selected = torch.gather(square_features, 1, gather_idx)
    selected_mask = torch.gather(occupancy, 1, ordered)
    return selected, selected_mask


def _local_occupancy_planes(piece_planes: torch.Tensor) -> torch.Tensor:
    """Compute side-aware local-occupancy context for each square.

    Returns a (B, 4, 8, 8) tensor:
      [own_count_3x3, opp_count_3x3, own_count_5x5, opp_count_5x5]
    counts include the centre square; counts are normalised by the kernel size.
    """
    own = piece_planes[:, :6].sum(dim=1, keepdim=True)
    opp = piece_planes[:, 6:].sum(dim=1, keepdim=True)
    pad3 = (1, 1, 1, 1)
    pad5 = (2, 2, 2, 2)
    kernel3 = own.new_ones(1, 1, 3, 3) / 9.0
    kernel5 = own.new_ones(1, 1, 5, 5) / 25.0
    own_pad3 = F.pad(own, pad3)
    opp_pad3 = F.pad(opp, pad3)
    own_pad5 = F.pad(own, pad5)
    opp_pad5 = F.pad(opp, pad5)
    own3 = F.conv2d(own_pad3, kernel3)
    opp3 = F.conv2d(opp_pad3, kernel3)
    own5 = F.conv2d(own_pad5, kernel5)
    opp5 = F.conv2d(opp_pad5, kernel5)
    return torch.cat([own3, opp3, own5, opp5], dim=1)


def masked_log_sinkhorn(
    cost: torch.Tensor,
    row_mass: torch.Tensor,
    col_mass: torch.Tensor,
    *,
    iterations: int,
    temperature: float,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Differentiable masked Sinkhorn-Knopp in log domain.

    Returns the transport matrix ``A`` of shape ``(B, R, C)`` such that
    ``sum_j A[b, i, j] == row_mass[b, i]`` (up to eps) for active rows and
    ``sum_i A[b, i, j] == col_mass[b, j]`` for active columns. Rows with zero
    row mass and columns with zero column mass receive zero transport.

    ``cost`` is the raw cost matrix (B, R, C); the kernel is
    ``exp(-cost / temperature)`` evaluated in log space.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    minus_inf = torch.finfo(cost.dtype).min
    log_K = -cost / float(temperature)

    log_row = row_mass.clamp_min(eps).log()
    log_col = col_mass.clamp_min(eps).log()
    inactive_rows = row_mass <= 0.0
    inactive_cols = col_mass <= 0.0
    log_row = torch.where(inactive_rows, torch.full_like(log_row, minus_inf), log_row)
    log_col = torch.where(inactive_cols, torch.full_like(log_col, minus_inf), log_col)

    # Mask the kernel so inactive rows/cols cannot transport mass.
    row_mask = inactive_rows.unsqueeze(-1).expand_as(log_K)
    col_mask = inactive_cols.unsqueeze(-2).expand_as(log_K)
    log_K = log_K.masked_fill(row_mask | col_mask, minus_inf)

    log_u = torch.zeros_like(log_row)
    log_v = torch.zeros_like(log_col)
    log_u = torch.where(inactive_rows, torch.full_like(log_u, minus_inf), log_u)
    log_v = torch.where(inactive_cols, torch.full_like(log_v, minus_inf), log_v)

    for _ in range(int(iterations)):
        # log_v[b, j] = log_col[b, j] - logsumexp_i(log_K[b, i, j] + log_u[b, i])
        log_v = log_col - torch.logsumexp(log_K + log_u.unsqueeze(-1), dim=1)
        log_v = torch.where(inactive_cols, torch.full_like(log_v, minus_inf), log_v)
        # log_u[b, i] = log_row[b, i] - logsumexp_j(log_K[b, i, j] + log_v[b, j])
        log_u = log_row - torch.logsumexp(log_K + log_v.unsqueeze(-2), dim=2)
        log_u = torch.where(inactive_rows, torch.full_like(log_u, minus_inf), log_u)

    log_A = log_K + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    A = log_A.exp()
    inactive = (row_mask | col_mask)
    A = A.masked_fill(inactive, 0.0)
    return A


class SinkhornRoleAssignmentNetwork(nn.Module):
    """Optimal-transport role assignment over piece tokens for puzzle binary."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        token_dim: int = 64,
        num_roles: int = 10,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        sinkhorn_iters: int = 8,
        sinkhorn_temperature: float = 0.5,
        token_hidden: int = 96,
        pair_hidden: int = 64,
        head_hidden: int = 128,
        cnn_channels: int = 32,
        cnn_depth: int = 2,
        dropout: float = 0.1,
        use_dustbin: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "SinkhornRoleAssignmentNetwork supports the puzzle_binary one-logit contract"
            )
        if input_channels != 18:
            raise ValueError("SinkhornRoleAssignmentNetwork expects the simple_18 board tensor")
        if num_roles < 1:
            raise ValueError("num_roles must be positive")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if sinkhorn_iters < 1:
            raise ValueError("sinkhorn_iters must be positive")
        if sinkhorn_temperature <= 0:
            raise ValueError("sinkhorn_temperature must be positive")

        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.token_dim = int(token_dim)
        self.num_roles = int(num_roles)
        self.use_dustbin = bool(use_dustbin)
        self.total_roles = self.num_roles + (1 if self.use_dustbin else 0)
        self.max_tokens = int(max_tokens)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_temperature = float(sinkhorn_temperature)

        token_input_dim = (
            PIECE_PLANES + GLOBAL_PLANES + TOKEN_COORD_FEATURES + LOCAL_OCCUPANCY_FEATURES
        )
        self.token_encoder = nn.Sequential(
            nn.Linear(token_input_dim, int(token_hidden)),
            nn.LayerNorm(int(token_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(token_hidden), self.token_dim),
            nn.LayerNorm(self.token_dim),
        )
        self.register_buffer("coords", _board_coords(), persistent=False)

        self.role_prototypes = nn.Parameter(torch.empty(self.total_roles, self.token_dim))
        nn.init.xavier_uniform_(self.role_prototypes)
        # Learned target role-mass priors over the `total_roles` slots
        # (softmaxed to a probability vector during forward).
        self.role_mass_logits = nn.Parameter(torch.zeros(self.total_roles))

        self.cost_proj = nn.Linear(self.token_dim, self.token_dim, bias=False)

        # Pairwise role interaction MLP. Input: concat(role_i, role_j, |role_i - role_j|).
        pair_in = self.token_dim * 3
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in, int(pair_hidden)),
            nn.GELU(),
            nn.Linear(int(pair_hidden), int(pair_hidden)),
        )

        # Light convolutional board summary branch.
        cnn_layers: list[nn.Module] = []
        in_c = input_channels
        for _ in range(int(cnn_depth)):
            cnn_layers.extend(
                [
                    nn.Conv2d(in_c, int(cnn_channels), kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(int(cnn_channels)),
                    nn.GELU(),
                ]
            )
            in_c = int(cnn_channels)
        self.cnn_stem = nn.Sequential(*cnn_layers)
        self.cnn_summary_dim = int(cnn_channels) * 2  # mean + max pool concat

        # Diagnostics summary fed to the head:
        #   role_mass (total_roles)
        #   role_share (total_roles)
        #   piece_share_mean, piece_share_var, dustbin_share, mean_assignment_entropy,
        #   token_count_norm, role_norms (total_roles)
        diagnostic_dim = (
            self.total_roles  # role_mass
            + self.total_roles  # role_share
            + 5
            + self.total_roles  # role_norms
        )
        head_input = (
            self.total_roles * self.token_dim
            + int(pair_hidden)
            + self.cnn_summary_dim
            + diagnostic_dim
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Linear(head_input, int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(head_hidden), 1),
        )

    def _build_tokens(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        piece_planes = x[:, :PIECE_PLANES]
        global_planes = x[:, PIECE_PLANES:]
        if global_planes.shape[1] != GLOBAL_PLANES:
            raise ValueError(
                "Expected 6 global planes after the 12 piece planes in simple_18"
            )

        local_occ = _local_occupancy_planes(piece_planes)
        per_square_pieces = piece_planes.flatten(2).transpose(1, 2)
        per_square_globals = global_planes.flatten(2).transpose(1, 2)
        per_square_local = local_occ.flatten(2).transpose(1, 2)
        coords = self.coords.to(dtype=x.dtype, device=x.device).unsqueeze(0).expand(b, -1, -1)
        per_square_features = torch.cat(
            [per_square_pieces, per_square_globals, coords, per_square_local], dim=-1
        )

        occupancy = (per_square_pieces.sum(dim=-1) > 0).to(dtype=x.dtype)
        selected_features, selected_mask = _select_occupied_tokens(
            per_square_features, occupancy, self.max_tokens
        )
        tokens = self.token_encoder(selected_features)
        tokens = tokens * selected_mask.unsqueeze(-1)
        return tokens, selected_mask, occupancy

    def _compute_assignment(
        self, tokens: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Cost is the negative cosine similarity between projected tokens and
        # learned role prototypes. Tokens for padded slots are ignored via the
        # masked Sinkhorn iterations.
        projected = self.cost_proj(tokens)
        prototypes = self.role_prototypes
        token_norm = F.normalize(projected, dim=-1, eps=1.0e-8)
        proto_norm = F.normalize(prototypes, dim=-1, eps=1.0e-8)
        # cost[b, i, j] in [0, 2]; lower = more aligned.
        cost = 1.0 - torch.einsum("bid,jd->bij", token_norm, proto_norm)

        token_count = token_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        role_prior = F.softmax(self.role_mass_logits, dim=-1)
        col_mass = role_prior.unsqueeze(0).expand(tokens.shape[0], -1) * token_count

        assignment = masked_log_sinkhorn(
            cost,
            row_mass=token_mask,
            col_mass=col_mass,
            iterations=self.sinkhorn_iters,
            temperature=self.sinkhorn_temperature,
        )
        return assignment, cost

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        tokens, token_mask, occupancy = self._build_tokens(x)
        b = tokens.shape[0]

        assignment, cost = self._compute_assignment(tokens, token_mask)
        # role_vectors[b, j, d] = sum_i A[b, i, j] * tokens[b, i, d]
        role_vectors = torch.einsum("bij,bid->bjd", assignment, tokens)

        # Pairwise role-slot interactions.
        role_i = role_vectors.unsqueeze(2).expand(-1, -1, self.total_roles, -1)
        role_j = role_vectors.unsqueeze(1).expand(-1, self.total_roles, -1, -1)
        pair_input = torch.cat([role_i, role_j, (role_i - role_j).abs()], dim=-1)
        pair_features = self.pair_mlp(pair_input)
        # Keep only off-diagonal pairs to avoid trivially encoding role norms twice.
        eye = torch.eye(self.total_roles, device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
        pair_mask = (1.0 - eye).unsqueeze(-1)
        pair_features = pair_features * pair_mask
        denom = max(self.total_roles * (self.total_roles - 1), 1)
        pair_summary = pair_features.sum(dim=(1, 2)) / float(denom)

        cnn_features = self.cnn_stem(x)
        cnn_mean = cnn_features.mean(dim=(-1, -2))
        cnn_max = cnn_features.amax(dim=(-1, -2))
        cnn_summary = torch.cat([cnn_mean, cnn_max], dim=-1)

        token_count = token_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        role_mass = assignment.sum(dim=1)
        role_share = role_mass / token_count
        piece_share = assignment.sum(dim=-1)  # mass spent per piece (active <= 1)
        active_pieces = token_mask
        piece_share_active = piece_share * active_pieces
        piece_share_mean = piece_share_active.sum(dim=-1) / token_count.squeeze(-1)
        piece_share_var = (
            ((piece_share_active - piece_share_mean.unsqueeze(-1)) ** 2) * active_pieces
        ).sum(dim=-1) / token_count.squeeze(-1)
        if self.use_dustbin:
            dustbin_share = role_share[:, -1]
        else:
            dustbin_share = torch.zeros(b, device=tokens.device, dtype=tokens.dtype)

        # Per-piece assignment entropy over role axis (active pieces only).
        normalised_pieces = assignment / piece_share.unsqueeze(-1).clamp_min(1.0e-8)
        log_pieces = normalised_pieces.clamp_min(1.0e-8).log()
        per_piece_entropy = -(normalised_pieces * log_pieces).sum(dim=-1) * active_pieces
        mean_assignment_entropy = per_piece_entropy.sum(dim=-1) / token_count.squeeze(-1)

        role_norms = role_vectors.norm(dim=-1)

        diagnostics = torch.cat(
            [
                role_mass,
                role_share,
                piece_share_mean.unsqueeze(-1),
                piece_share_var.unsqueeze(-1),
                dustbin_share.unsqueeze(-1),
                mean_assignment_entropy.unsqueeze(-1),
                token_count.squeeze(-1).unsqueeze(-1) / float(self.max_tokens),
                role_norms,
            ],
            dim=-1,
        )

        flat_roles = role_vectors.reshape(b, self.total_roles * self.token_dim)
        features = torch.cat([flat_roles, pair_summary, cnn_summary, diagnostics], dim=-1)
        logits = self.classifier(features).view(-1)

        return {
            "logits": logits,
            "tokens": tokens,
            "token_mask": token_mask,
            "occupancy_mask": occupancy,
            "cost": cost,
            "assignment": assignment,
            "role_vectors": role_vectors,
            "role_mass": role_mass,
            "role_share": role_share,
            "role_norms": role_norms,
            "piece_share": piece_share,
            "piece_share_mean": piece_share_mean,
            "piece_share_var": piece_share_var,
            "dustbin_share": dustbin_share,
            "mean_assignment_entropy": mean_assignment_entropy,
            "pair_summary": pair_summary,
            "cnn_summary": cnn_summary,
            "diagnostic_features": diagnostics,
        }


def build_sinkhorn_role_assignment_network_from_config(
    config: dict[str, Any],
) -> SinkhornRoleAssignmentNetwork:
    cfg = dict(config)
    token_dim = int(cfg.get("token_dim", cfg.get("channels", 64)))
    return SinkhornRoleAssignmentNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        token_dim=token_dim,
        num_roles=int(cfg.get("num_roles", 10)),
        max_tokens=int(cfg.get("max_tokens", DEFAULT_MAX_TOKENS)),
        sinkhorn_iters=int(cfg.get("sinkhorn_iters", 8)),
        sinkhorn_temperature=float(cfg.get("sinkhorn_temperature", 0.5)),
        token_hidden=int(cfg.get("token_hidden", cfg.get("hidden_dim", 96))),
        pair_hidden=int(cfg.get("pair_hidden", cfg.get("hidden_dim", 64))),
        head_hidden=int(cfg.get("head_hidden", cfg.get("hidden_dim", 128))),
        cnn_channels=int(cfg.get("cnn_channels", cfg.get("channels", 32))),
        cnn_depth=int(cfg.get("cnn_depth", cfg.get("depth", 2))),
        dropout=float(cfg.get("dropout", 0.1)),
        use_dustbin=bool(cfg.get("use_dustbin", True)),
    )
