"""Tactical State Bottleneck Inference for idea i091."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


FORBIDDEN_KEYS = {
    "engine_score",
    "engine_scores",
    "eval",
    "cp",
    "pv",
    "pvs",
    "principal_variation",
    "principal_variations",
    "node_count",
    "node_counts",
    "mate_score",
    "mate_scores",
    "best_move",
    "best_moves",
    "solution_move",
    "solution_moves",
    "verification_metadata",
    "verified_by",
    "source_label",
    "source_labels",
    "source_provenance",
    "site_origin",
    "database_origin",
    "puzzle_generator_id",
    "curation_status",
}

LATENT_SIZES = {
    "motif": 10,
    "anchor": 65,
    "target": 65,
    "relation": 8,
    "vulnerability": 8,
    "tempo": 4,
}
FREE_BITS = {
    "motif": 0.05,
    "anchor": 0.10,
    "target": 0.10,
    "relation": 0.05,
    "vulnerability": 0.05,
    "tempo": 0.03,
}
ENTROPY_FLOORS = {
    "motif": 1.0,
    "anchor": 2.0,
    "target": 2.0,
    "relation": 0.8,
    "vulnerability": 0.8,
    "tempo": 0.5,
}


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -(probs * probs.clamp_min(1.0e-8).log()).sum(dim=-1)


def _group_usage(probs: torch.Tensor) -> torch.Tensor:
    return probs.mean(dim=0)


@dataclass(frozen=True)
class TacticalStateSchedule:
    beta_kl: float = 0.5
    lambda_prior: float = 1.0
    lambda_usage: float = 0.01
    lambda_entropy: float = 0.01


def build_puzzle_binary_target(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bad_keys = set(batch) & FORBIDDEN_KEYS
    if bad_keys:
        raise ValueError(f"Forbidden leakage keys in batch: {sorted(bad_keys)}")
    board = batch["board"]
    fine = batch["fine_label"].long()
    if board.ndim != 4 or tuple(board.shape[-2:]) != (8, 8):
        raise ValueError(f"Expected board tensor [B,C,8,8], got {tuple(board.shape)}")
    if fine.ndim != 1 or fine.shape[0] != board.shape[0]:
        raise ValueError("fine_label must be [B] and match board batch size")
    if fine.numel() and (int(fine.min()) < 0 or int(fine.max()) > 2):
        raise ValueError("fine_label values must be in {0,1,2}")
    return board, fine, (fine == 2).float()


@torch.no_grad()
def diagnostic_3x2(fine_label: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
    pred = (logit.view(-1) > 0).long()
    fine = fine_label.view(-1).long().clamp(0, 2)
    table = torch.zeros(3, 2, dtype=torch.long, device=logit.device)
    table.view(-1).scatter_add_(0, fine * 2 + pred, torch.ones_like(fine, dtype=torch.long))
    return table


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(max(1, min(8, channels)), channels)
        self.norm2 = nn.GroupNorm(max(1, min(8, channels)), channels)
        self.dropout = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.norm1(self.conv1(x)))
        z = self.dropout(z)
        z = self.norm2(self.conv2(z))
        return F.gelu(x + z)


class ChessBoardTrunk(nn.Module):
    def __init__(self, input_channels: int = 18, hidden_dim: int = 96, blocks: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.input = nn.Conv2d(int(input_channels) + 2, int(hidden_dim), kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(int(hidden_dim), dropout=dropout) for _ in range(max(1, int(blocks)))])
        self.norm = nn.GroupNorm(max(1, min(8, int(hidden_dim))), int(hidden_dim))
        coords = self._coords()
        self.register_buffer("coords", coords, persistent=False)
        self.output_dim = int(hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        coords = self.coords.to(device=board.device, dtype=board.dtype).expand(board.shape[0], -1, -1, -1)
        h = F.gelu(self.input(torch.cat([board, coords], dim=1)))
        for block in self.blocks:
            h = block(h)
        h = F.gelu(self.norm(h))
        pooled = h.mean(dim=(2, 3))
        return h, pooled

    @staticmethod
    def _coords() -> torch.Tensor:
        rows = torch.linspace(-1.0, 1.0, 8).view(1, 8, 1).expand(1, 8, 8)
        cols = torch.linspace(-1.0, 1.0, 8).view(1, 1, 8).expand(1, 8, 8)
        return torch.stack([rows[0], cols[0]], dim=0).unsqueeze(0)


class SquareLogitHead(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(int(in_channels), 1, kernel_size=1)
        self.null_logit = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial(h).flatten(start_dim=1)
        return torch.cat([spatial, self.null_logit.expand(h.shape[0], 1)], dim=-1)


class PriorTacticalHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.motif = nn.Linear(hidden_dim, LATENT_SIZES["motif"])
        self.anchor = SquareLogitHead(hidden_dim)
        self.target = SquareLogitHead(hidden_dim)
        self.relation = nn.Linear(hidden_dim, LATENT_SIZES["relation"])
        self.vulnerability = nn.Linear(hidden_dim, LATENT_SIZES["vulnerability"])
        self.tempo = nn.Linear(hidden_dim, LATENT_SIZES["tempo"])

    def forward(self, h: torch.Tensor, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "motif": self.motif(pooled),
            "anchor": self.anchor(h),
            "target": self.target(h),
            "relation": self.relation(pooled),
            "vulnerability": self.vulnerability(pooled),
            "tempo": self.tempo(pooled),
        }


class PosteriorTacticalHead(nn.Module):
    def __init__(self, hidden_dim: int, label_dim: int = 16) -> None:
        super().__init__()
        self.y_embedding = nn.Embedding(2, int(label_dim))
        self.spatial_fuse = nn.Sequential(
            nn.Conv2d(int(hidden_dim) + int(label_dim), int(hidden_dim), kernel_size=1),
            nn.GELU(),
        )
        self.motif = nn.Linear(int(hidden_dim) + int(label_dim), LATENT_SIZES["motif"])
        self.anchor = SquareLogitHead(hidden_dim)
        self.target = SquareLogitHead(hidden_dim)
        self.relation = nn.Linear(int(hidden_dim) + int(label_dim), LATENT_SIZES["relation"])
        self.vulnerability = nn.Linear(int(hidden_dim) + int(label_dim), LATENT_SIZES["vulnerability"])
        self.tempo = nn.Linear(int(hidden_dim) + int(label_dim), LATENT_SIZES["tempo"])

    def forward(self, h: torch.Tensor, pooled: torch.Tensor, y_long: torch.Tensor) -> dict[str, torch.Tensor]:
        y_emb = self.y_embedding(y_long)
        pooled_post = torch.cat([pooled, y_emb], dim=-1)
        y_map = y_emb[:, :, None, None].expand(-1, -1, h.shape[-2], h.shape[-1])
        h_post = self.spatial_fuse(torch.cat([h, y_map], dim=1))
        return {
            "motif": self.motif(pooled_post),
            "anchor": self.anchor(h_post),
            "target": self.target(h_post),
            "relation": self.relation(pooled_post),
            "vulnerability": self.vulnerability(pooled_post),
            "tempo": self.tempo(pooled_post),
        }


class CategoricalLatent(nn.Module):
    def __init__(self, num_categories: int, emb_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(int(num_categories), int(emb_dim)) * 0.02)

    def expected_embedding(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(logits, dim=-1)
        return probs, probs @ self.embedding

    def sample_embedding(self, logits: torch.Tensor, temperature: float, hard: bool) -> tuple[torch.Tensor, torch.Tensor]:
        probs = F.gumbel_softmax(logits, tau=float(temperature), hard=bool(hard), dim=-1)
        return probs, probs @ self.embedding


class TacticalStateBottleneckModel(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        hidden_dim: int = 96,
        trunk_blocks: int = 4,
        latent_dim: int = 24,
        head_hidden: int = 128,
        direct_alpha: float = 0.25,
        temperature: float = 0.7,
        hard_gumbel: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("TacticalStateBottleneckModel supports the puzzle_binary one-logit contract")
        self.num_classes = int(num_classes)
        self.direct_alpha = float(direct_alpha)
        self.temperature = float(temperature)
        self.hard_gumbel = bool(hard_gumbel)
        self.trunk = ChessBoardTrunk(input_channels=int(input_channels), hidden_dim=int(hidden_dim), blocks=int(trunk_blocks), dropout=dropout)
        self.prior_head = PriorTacticalHead(int(hidden_dim))
        self.posterior_head = PosteriorTacticalHead(int(hidden_dim), label_dim=max(8, int(latent_dim) // 2))
        self.latents = nn.ModuleDict({name: CategoricalLatent(size, int(latent_dim)) for name, size in LATENT_SIZES.items()})
        latent_total = int(latent_dim) * len(LATENT_SIZES)
        self.latent_head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim) + latent_total),
            nn.Linear(int(hidden_dim) + latent_total, int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(head_hidden), 1),
        )
        self.direct_head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), max(16, int(head_hidden) // 2)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(max(16, int(head_hidden) // 2), 1),
        )

    def forward(
        self,
        board: torch.Tensor,
        fine_label: torch.Tensor | None = None,
        *,
        return_latents: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if self.training and fine_label is not None:
            return self.forward_train(board, fine_label)
        return self.forward_eval(board, return_latents=return_latents)

    def forward_train(self, board: torch.Tensor, fine_label: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        y_long = (fine_label.long() == 2).long()
        h, pooled = self.trunk(board)
        prior_logits = self.prior_head(h, pooled)
        posterior_logits = self.posterior_head(h, pooled, y_long)
        prior_probs, prior_emb = self.project_latents(prior_logits, sample=True)
        posterior_probs, posterior_emb = self.project_latents(posterior_logits, sample=True)
        logit_p = self.compute_logit(pooled, prior_emb)
        logit_q = self.compute_logit(pooled, posterior_emb)
        losses = tactical_state_loss_components(
            logit_q=logit_q,
            logit_p=logit_p,
            posterior_logits=posterior_logits,
            prior_logits=prior_logits,
            fine_label=fine_label,
        )
        output = self._base_output(
            logit=logit_p,
            pooled=pooled,
            probs=prior_probs,
            logits_by_group=prior_logits,
            prefix="prior",
        )
        output.update(
            {
                "logit": logit_p,
                "logit_q": logit_q,
                "logit_p": logit_p,
                "prior_logits": prior_logits,
                "posterior_logits": posterior_logits,
                "prior_probs": prior_probs,
                "posterior_probs": posterior_probs,
                "losses": losses,
                "diag_3x2": diagnostic_3x2(fine_label, logit_p),
                "prior_posterior_agreement": self._agreement(prior_probs, posterior_probs),
            }
        )
        return output

    def forward_eval(self, board: torch.Tensor, return_latents: bool = False) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        h, pooled = self.trunk(board)
        prior_logits = self.prior_head(h, pooled)
        prior_probs, prior_emb = self.project_latents(prior_logits, sample=False)
        logit = self.compute_logit(pooled, prior_emb)
        output = self._base_output(logit=logit, pooled=pooled, probs=prior_probs, logits_by_group=prior_logits, prefix="prior")
        output["logit"] = logit
        if return_latents:
            output["prior_logits"] = prior_logits
            output["prior_probs"] = prior_probs
        return output

    def project_latents(
        self,
        logits_by_group: dict[str, torch.Tensor],
        *,
        sample: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        probs: dict[str, torch.Tensor] = {}
        embeddings = []
        for name in LATENT_SIZES:
            latent = self.latents[name]
            if sample:
                prob, emb = latent.sample_embedding(logits_by_group[name], self.temperature, self.hard_gumbel)
            else:
                prob, emb = latent.expected_embedding(logits_by_group[name])
            probs[name] = prob
            embeddings.append(emb)
        return probs, torch.cat(embeddings, dim=-1)

    def compute_logit(self, pooled: torch.Tensor, latent_emb: torch.Tensor) -> torch.Tensor:
        latent_logit = self.latent_head(torch.cat([pooled, latent_emb], dim=-1)).view(-1)
        direct_logit = self.direct_head(pooled).view(-1)
        return latent_logit + self.direct_alpha * direct_logit

    def _base_output(
        self,
        *,
        logit: torch.Tensor,
        pooled: torch.Tensor,
        probs: dict[str, torch.Tensor],
        logits_by_group: dict[str, torch.Tensor],
        prefix: str,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        logits = _format_logits(logit, self.num_classes)
        entropy_by_group = {name: _entropy_from_logits(group_logits) for name, group_logits in logits_by_group.items()}
        usage = {name: _group_usage(group_probs) for name, group_probs in probs.items()}
        return {
            "logits": logits,
            "logit": logits,
            "prob": torch.sigmoid(logits),
            "latent_probs": probs,
            f"{prefix}_entropy_by_group": entropy_by_group,
            f"{prefix}_usage_by_group": usage,
            "motif_usage": usage["motif"],
            "relation_usage": usage["relation"],
            "vulnerability_usage": usage["vulnerability"],
            "tempo_usage": usage["tempo"],
            "anchor_null_rate": probs["anchor"][:, -1],
            "target_null_rate": probs["target"][:, -1],
            "motif_entropy": entropy_by_group["motif"],
            "anchor_entropy": entropy_by_group["anchor"],
            "target_entropy": entropy_by_group["target"],
            "relation_entropy": entropy_by_group["relation"],
            "vulnerability_entropy": entropy_by_group["vulnerability"],
            "tempo_entropy": entropy_by_group["tempo"],
            "pooled_energy": pooled.pow(2).mean(dim=1),
            "direct_alpha": logits.new_full((logits.shape[0],), self.direct_alpha),
            "mechanism_energy": torch.cat([group.mean(dim=1, keepdim=True) for group in probs.values()], dim=1).pow(2).mean(dim=1),
            "proposal_profile_strength": torch.stack([group.max(dim=1).values for group in probs.values()], dim=1).mean(dim=1),
            "proposal_keyword_count": logits.new_full((logits.shape[0],), 6.0),
        }

    @staticmethod
    def _agreement(prior_probs: dict[str, torch.Tensor], posterior_probs: dict[str, torch.Tensor]) -> torch.Tensor:
        agreements = []
        for name in LATENT_SIZES:
            agreements.append((prior_probs[name].argmax(dim=-1) == posterior_probs[name].argmax(dim=-1)).float())
        return torch.stack(agreements, dim=1).mean(dim=1)


def categorical_kl(q_logits: torch.Tensor, p_logits: torch.Tensor) -> torch.Tensor:
    q = torch.softmax(q_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    log_p = torch.log_softmax(p_logits, dim=-1)
    return (q * (log_q - log_p)).sum(dim=-1)


def kl_freebits(
    posterior_logits: dict[str, torch.Tensor],
    prior_logits: dict[str, torch.Tensor],
    free_bits: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    free_bits = FREE_BITS if free_bits is None else free_bits
    total = None
    per_group: dict[str, torch.Tensor] = {}
    for name in LATENT_SIZES:
        kl = categorical_kl(posterior_logits[name], prior_logits[name]).mean()
        per_group[name] = kl.detach()
        term = torch.clamp(kl, min=float(free_bits.get(name, 0.0)))
        total = term if total is None else total + term
    if total is None:
        raise ValueError("No latent groups available for KL")
    return total, per_group


def batch_usage_loss(logits_by_group: dict[str, torch.Tensor]) -> torch.Tensor:
    total = None
    for logits in logits_by_group.values():
        probs = torch.softmax(logits, dim=-1)
        q_bar = probs.mean(dim=0).clamp_min(1.0e-8)
        uniform = torch.full_like(q_bar, 1.0 / q_bar.numel())
        term = (q_bar * (q_bar.log() - uniform.log())).sum()
        total = term if total is None else total + term
    if total is None:
        raise ValueError("No latent groups available for usage loss")
    return total


def entropy_floor_loss(logits_by_group: dict[str, torch.Tensor], entropy_floor: dict[str, float] | None = None) -> torch.Tensor:
    entropy_floor = ENTROPY_FLOORS if entropy_floor is None else entropy_floor
    total = None
    for name, logits in logits_by_group.items():
        entropy = _entropy_from_logits(logits).mean()
        term = F.relu(logits.new_tensor(float(entropy_floor.get(name, 0.0))) - entropy)
        total = term if total is None else total + term
    if total is None:
        raise ValueError("No latent groups available for entropy loss")
    return total


def tactical_state_loss_components(
    *,
    logit_q: torch.Tensor,
    logit_p: torch.Tensor,
    posterior_logits: dict[str, torch.Tensor],
    prior_logits: dict[str, torch.Tensor],
    fine_label: torch.Tensor,
    schedule: TacticalStateSchedule | None = None,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    schedule = TacticalStateSchedule() if schedule is None else schedule
    y = (fine_label.long() == 2).float()
    loss_pred = F.binary_cross_entropy_with_logits(logit_q.view(-1), y)
    loss_prior_pred = F.binary_cross_entropy_with_logits(logit_p.view(-1), y)
    loss_kl, kl_by_group = kl_freebits(posterior_logits, prior_logits)
    loss_usage = batch_usage_loss(posterior_logits)
    loss_entropy = entropy_floor_loss(posterior_logits)
    loss = (
        loss_pred
        + float(schedule.beta_kl) * loss_kl
        + float(schedule.lambda_prior) * loss_prior_pred
        + float(schedule.lambda_usage) * loss_usage
        + float(schedule.lambda_entropy) * loss_entropy
    )
    return {
        "loss": loss,
        "loss_pred": loss_pred.detach(),
        "loss_prior_pred": loss_prior_pred.detach(),
        "loss_kl": loss_kl.detach(),
        "loss_usage": loss_usage.detach(),
        "loss_entropy": loss_entropy.detach(),
        "kl_by_group": kl_by_group,
    }


class NoLatentMatchedBaseline(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        hidden_dim: int = 96,
        trunk_blocks: int = 4,
        head_width: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.trunk = ChessBoardTrunk(input_channels=input_channels, hidden_dim=hidden_dim, blocks=trunk_blocks, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, head_width),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(head_width, head_width),
            nn.GELU(),
            nn.Linear(head_width, 1),
        )

    def forward(self, board: torch.Tensor) -> dict[str, torch.Tensor]:
        _, pooled = self.trunk(board)
        logits = self.head(pooled).view(-1)
        return {
            "logits": logits,
            "logit": logits,
            "prob": torch.sigmoid(logits),
            "direct_feature_energy": pooled.pow(2).mean(dim=1),
        }


def build_tactical_state_bottleneck_from_config(config: dict[str, Any]) -> TacticalStateBottleneckModel:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return TacticalStateBottleneckModel(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        hidden_dim=int(cfg.get("hidden_dim", cfg.get("channels", 96))),
        trunk_blocks=int(cfg.get("trunk_blocks", cfg.get("depth", 4))),
        latent_dim=int(cfg.get("latent_dim", 24)),
        head_hidden=int(cfg.get("head_hidden", 128)),
        direct_alpha=float(cfg.get("direct_alpha", 0.25)),
        temperature=float(cfg.get("temperature", 0.7)),
        hard_gumbel=bool(cfg.get("hard_gumbel", False)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
