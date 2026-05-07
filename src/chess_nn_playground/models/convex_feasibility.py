"""Convex Feasibility Residual Network for idea i094."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


class BoardFeasibilityEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        latent_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.0,
        include_coordinates: bool = True,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.include_coordinates = bool(include_coordinates)
        in_channels = int(input_channels) + (2 if self.include_coordinates else 0)
        width = int(channels)
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, min(8, width)), width),
            nn.GELU(),
        ]
        for _ in range(max(0, int(depth) - 1)):
            layers.extend(
                [
                    nn.Conv2d(width, width, kernel_size=3, padding=1),
                    nn.GroupNorm(max(1, min(8, width)), width),
                    nn.GELU(),
                    nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
                ]
            )
        self.conv = nn.Sequential(*layers)
        self.project = nn.Sequential(
            nn.Linear(width * 2, int(latent_dim)),
            nn.LayerNorm(int(latent_dim)),
            nn.GELU(),
        )
        if self.include_coordinates:
            coords = torch.linspace(-1.0, 1.0, 8)
            rank = coords.view(1, 1, 8, 1).expand(1, 1, 8, 8)
            file = coords.view(1, 1, 1, 8).expand(1, 1, 8, 8)
            self.register_buffer("coord_planes", torch.cat([rank, file], dim=1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        board = require_board_tensor(x, self.spec)
        if self.include_coordinates:
            coords = self.coord_planes.to(device=board.device, dtype=board.dtype).expand(board.shape[0], -1, -1, -1)
            board = torch.cat([board, coords], dim=1)
        features = self.conv(board)
        pooled = torch.cat([features.mean(dim=(2, 3)), features.amax(dim=(2, 3))], dim=1)
        return self.project(pooled)


class MaterialOnlyEncoder(nn.Module):
    def __init__(self, input_channels: int = 18, latent_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.net = nn.Sequential(
            nn.Linear(int(input_channels) * 4, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(latent_dim)),
            nn.LayerNorm(int(latent_dim)),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        board = require_board_tensor(x, self.spec)
        counts = board.sum(dim=(2, 3))
        means = board.mean(dim=(2, 3))
        maxes = board.amax(dim=(2, 3))
        mins = board.amin(dim=(2, 3))
        return self.net(torch.cat([counts, means, maxes, mins], dim=1))


class LearnedConvexConstraints(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        halfspace_constraints: int = 32,
        ball_constraints: int = 16,
        ball_dim: int = 16,
        random_seed: int = 1940,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.halfspace_constraints = int(halfspace_constraints)
        self.ball_constraints = int(ball_constraints)
        self.ball_dim = int(ball_dim)
        self.halfspace_normals = nn.Parameter(torch.randn(self.halfspace_constraints, self.latent_dim) * 0.08)
        self.halfspace_bias = nn.Parameter(torch.full((self.halfspace_constraints,), 0.35))
        self.ball_projectors = nn.Parameter(torch.randn(self.ball_constraints, self.ball_dim, self.latent_dim) * 0.04)
        self.ball_centers = nn.Parameter(torch.randn(self.ball_constraints, self.ball_dim) * 0.04)
        self.ball_radius = nn.Parameter(torch.full((self.ball_constraints,), 0.65))

        generator = torch.Generator().manual_seed(int(random_seed))
        random_normals = torch.randn(self.halfspace_constraints, self.latent_dim, generator=generator)
        random_projectors = torch.randn(self.ball_constraints, self.ball_dim, self.latent_dim, generator=generator)
        random_centers = torch.randn(self.ball_constraints, self.ball_dim, generator=generator) * 0.04
        self.register_buffer("random_halfspace_normals", random_normals, persistent=False)
        self.register_buffer("random_halfspace_bias", torch.full((self.halfspace_constraints,), 0.35), persistent=False)
        self.register_buffer("random_ball_projectors", random_projectors, persistent=False)
        self.register_buffer("random_ball_centers", random_centers, persistent=False)
        self.register_buffer("random_ball_radius", torch.full((self.ball_constraints,), 0.65), persistent=False)

    @property
    def num_constraints(self) -> int:
        return self.halfspace_constraints + self.ball_constraints

    def tensors(self, *, random_constraints: bool, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
        if random_constraints:
            normals = self.random_halfspace_normals.to(device=device, dtype=dtype)
            halfspace_bias = self.random_halfspace_bias.to(device=device, dtype=dtype)
            projectors = self.random_ball_projectors.to(device=device, dtype=dtype)
            centers = self.random_ball_centers.to(device=device, dtype=dtype)
            radius = self.random_ball_radius.to(device=device, dtype=dtype)
        else:
            normals = self.halfspace_normals.to(device=device, dtype=dtype)
            halfspace_bias = F.softplus(self.halfspace_bias.to(device=device, dtype=dtype)) + 1.0e-4
            projectors = self.ball_projectors.to(device=device, dtype=dtype)
            centers = self.ball_centers.to(device=device, dtype=dtype)
            radius = F.softplus(self.ball_radius.to(device=device, dtype=dtype)) + 1.0e-4
        normals = F.normalize(normals, dim=1)
        projectors = F.normalize(projectors, dim=2)
        return normals, halfspace_bias, projectors, centers, radius


class SoftProjectionLayer(nn.Module):
    def __init__(
        self,
        constraints: LearnedConvexConstraints,
        projection_steps: int = 3,
        step_size: float = 0.25,
        hinge_temperature: float = 0.1,
        gate_temperature: float = 8.0,
    ) -> None:
        super().__init__()
        self.constraints = constraints
        self.projection_steps = max(1, int(projection_steps))
        self.step_size = float(step_size)
        self.hinge_temperature = float(hinge_temperature)
        self.gate_temperature = float(gate_temperature)

    def forward(self, z: torch.Tensor, *, random_constraints: bool = False) -> dict[str, torch.Tensor]:
        path: list[torch.Tensor] = []
        step_norms: list[torch.Tensor] = []
        current = z
        last: dict[str, torch.Tensor] | None = None
        for _ in range(self.projection_steps):
            last = self._violation_and_gradient(current, random_constraints=random_constraints)
            delta = last["gradient"]
            next_z = current - self.step_size * delta
            path.append(next_z)
            step_norms.append((next_z - current).norm(dim=1))
            current = next_z
        if last is None:
            last = self._violation_and_gradient(current, random_constraints=random_constraints)
            projection_path = current.unsqueeze(1)
            path_step_norms = current.new_zeros(current.shape[0], 1)
        else:
            projection_path = torch.stack(path, dim=1)
            path_step_norms = torch.stack(step_norms, dim=1)
        final = self._violation_and_gradient(current, random_constraints=random_constraints)
        return {
            "projected_z": current,
            "projection_path": projection_path,
            "path_step_norms": path_step_norms,
            "path_length": path_step_norms.sum(dim=1),
            "halfspace_violations": final["halfspace_violations"],
            "ball_violations": final["ball_violations"],
            "violations": final["violations"],
            "halfspace_raw": final["halfspace_raw"],
            "ball_raw": final["ball_raw"],
            "constraint_gates": final["constraint_gates"],
        }

    def _violation_and_gradient(self, z: torch.Tensor, *, random_constraints: bool) -> dict[str, torch.Tensor]:
        normals, halfspace_bias, projectors, centers, radius = self.constraints.tensors(
            random_constraints=random_constraints,
            device=z.device,
            dtype=z.dtype,
        )
        tau = max(self.hinge_temperature, 1.0e-4)

        halfspace_raw = z @ normals.T - halfspace_bias.view(1, -1)
        halfspace_violations = F.softplus(halfspace_raw / tau) * tau
        halfspace_gates = torch.sigmoid(self.gate_temperature * halfspace_raw)
        halfspace_grad = torch.einsum("bk,kd->bd", halfspace_gates * halfspace_violations, normals)

        projected = torch.einsum("bd,kmd->bkm", z, projectors)
        diff = projected - centers.view(1, centers.shape[0], centers.shape[1])
        dist = diff.square().sum(dim=-1).clamp_min(1.0e-8).sqrt()
        ball_raw = dist - radius.view(1, -1)
        ball_violations = F.softplus(ball_raw / tau) * tau
        ball_gates = torch.sigmoid(self.gate_temperature * ball_raw)
        ball_unit = diff / dist.clamp_min(1.0e-6).unsqueeze(-1)
        ball_grad_each = torch.einsum("bkm,kmd->bkd", ball_unit, projectors)
        ball_grad = ((ball_gates * ball_violations).unsqueeze(-1) * ball_grad_each).sum(dim=1)

        gradient = halfspace_grad + ball_grad
        violations = torch.cat([halfspace_violations, ball_violations], dim=1)
        gates = torch.cat([halfspace_gates, ball_gates], dim=1)
        return {
            "gradient": gradient,
            "halfspace_violations": halfspace_violations,
            "ball_violations": ball_violations,
            "violations": violations,
            "halfspace_raw": halfspace_raw,
            "ball_raw": ball_raw,
            "constraint_gates": gates,
        }


class ConvexFeasibilityResidualNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        latent_dim: int = 64,
        halfspace_constraints: int = 32,
        ball_constraints: int = 16,
        ball_dim: int = 16,
        projection_steps: int = 3,
        step_size: float = 0.25,
        hinge_temperature: float = 0.1,
        gate_temperature: float = 8.0,
        head_hidden: int = 192,
        dropout: float = 0.1,
        mode: str = "projection",
        include_coordinates: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("ConvexFeasibilityResidualNetwork supports the puzzle_binary one-logit contract")
        self.num_classes = int(num_classes)
        self.mode = str(mode)
        self.latent_dim = int(latent_dim)
        self.encoder = BoardFeasibilityEncoder(
            input_channels=int(input_channels),
            channels=int(channels),
            latent_dim=self.latent_dim,
            depth=int(depth),
            dropout=float(dropout),
            include_coordinates=bool(include_coordinates),
        )
        self.material_encoder = MaterialOnlyEncoder(
            input_channels=int(input_channels),
            latent_dim=self.latent_dim,
            hidden_dim=int(hidden_dim),
        )
        self.constraints = LearnedConvexConstraints(
            latent_dim=self.latent_dim,
            halfspace_constraints=int(halfspace_constraints),
            ball_constraints=int(ball_constraints),
            ball_dim=int(ball_dim),
        )
        projection_steps = max(1, int(projection_steps))
        self.projector = SoftProjectionLayer(
            self.constraints,
            projection_steps=projection_steps,
            step_size=float(step_size),
            hinge_temperature=float(hinge_temperature),
            gate_temperature=float(gate_temperature),
        )
        feature_dim = self.latent_dim * 3 + self.constraints.num_constraints + projection_steps + 8
        self.residual_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(head_hidden), max(32, int(head_hidden) // 4)),
            nn.GELU(),
            nn.Linear(max(32, int(head_hidden) // 4), 1),
        )
        self.no_projection_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, int(head_hidden)),
            nn.GELU(),
            nn.Linear(int(head_hidden), 1),
        )

    def forward(self, x: torch.Tensor, *, return_projection: bool = False) -> dict[str, torch.Tensor]:
        z = self.material_encoder(x) if self.mode == "material_only_encoder" else self.encoder(x)
        use_random = self.mode == "random_constraints"
        no_projection = self.mode in {"no_projection", "linear_head_same_params"}

        if no_projection:
            projection = self._zero_projection(z)
            logits = _format_logits(self.no_projection_head(z), self.num_classes)
            residual_features = z
        else:
            projection = self.projector(z, random_constraints=use_random)
            residual = z - projection["projected_z"]
            stats = self._projection_stats(z, projection)
            residual_features = torch.cat(
                [
                    z,
                    projection["projected_z"],
                    residual,
                    projection["violations"],
                    projection["path_step_norms"],
                    stats,
                ],
                dim=1,
            )
            logits = _format_logits(self.residual_head(residual_features), self.num_classes)

        residual = z - projection["projected_z"]
        violations = projection["violations"]
        halfspace_feasible = (projection["halfspace_raw"] <= 0.0).float().mean(dim=1)
        ball_feasible = (projection["ball_raw"] <= 0.0).float().mean(dim=1)
        feasible_fraction = torch.cat(
            [(projection["halfspace_raw"] <= 0.0).float(), (projection["ball_raw"] <= 0.0).float()],
            dim=1,
        ).mean(dim=1)
        output = {
            "logits": logits,
            "prob": torch.sigmoid(logits),
            "z": z,
            "projected_z": projection["projected_z"],
            "feasibility_residual": residual,
            "violations": violations,
            "halfspace_violations": projection["halfspace_violations"],
            "ball_violations": projection["ball_violations"],
            "path_step_norms": projection["path_step_norms"],
            "path_length": projection["path_length"],
            "residual_norm": residual.norm(dim=1),
            "max_violation": violations.max(dim=1).values,
            "mean_violation": violations.mean(dim=1),
            "feasibility_energy": violations.square().mean(dim=1),
            "feasible_fraction": feasible_fraction,
            "halfspace_feasible_fraction": halfspace_feasible,
            "ball_feasible_fraction": ball_feasible,
            "constraint_gate_mean": projection["constraint_gates"].mean(dim=1),
            "projection_mode": logits.new_full((logits.shape[0],), self._mode_code()),
            "mechanism_energy": violations.square().mean(dim=1),
            "proposal_profile_strength": residual.norm(dim=1),
            "proposal_keyword_count": logits.new_full((logits.shape[0],), 4.0),
        }
        if return_projection:
            output["projection_path"] = projection["projection_path"]
            output["residual_features"] = residual_features
            output["constraint_gates"] = projection["constraint_gates"]
        return output

    def _projection_stats(self, z: torch.Tensor, projection: dict[str, torch.Tensor]) -> torch.Tensor:
        residual = z - projection["projected_z"]
        violations = projection["violations"]
        stats = torch.stack(
            [
                residual.norm(dim=1),
                residual.square().mean(dim=1),
                projection["path_length"],
                projection["path_step_norms"].max(dim=1).values,
                violations.mean(dim=1),
                violations.max(dim=1).values,
                violations.square().mean(dim=1),
                projection["constraint_gates"].mean(dim=1),
            ],
            dim=1,
        )
        return stats

    def _zero_projection(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        zeros_constraints = z.new_zeros(z.shape[0], self.constraints.num_constraints)
        zeros_half = z.new_zeros(z.shape[0], self.constraints.halfspace_constraints)
        zeros_ball = z.new_zeros(z.shape[0], self.constraints.ball_constraints)
        step_count = max(1, self.projector.projection_steps)
        return {
            "projected_z": z,
            "projection_path": z.unsqueeze(1).expand(-1, step_count, -1),
            "path_step_norms": z.new_zeros(z.shape[0], step_count),
            "path_length": z.new_zeros(z.shape[0]),
            "halfspace_violations": zeros_half,
            "ball_violations": zeros_ball,
            "violations": zeros_constraints,
            "halfspace_raw": -torch.ones_like(zeros_half),
            "ball_raw": -torch.ones_like(zeros_ball),
            "constraint_gates": zeros_constraints,
        }

    def _mode_code(self) -> float:
        return {
            "projection": 0.0,
            "no_projection": 1.0,
            "random_constraints": 2.0,
            "linear_head_same_params": 3.0,
            "material_only_encoder": 4.0,
        }.get(self.mode, 0.0)


def build_convex_feasibility_residual_network_from_config(config: dict[str, Any]) -> ConvexFeasibilityResidualNetwork:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    channels = int(cfg.get("channels", cfg.get("hidden_dim", 64)))
    hidden_dim = int(cfg.get("hidden_dim", max(64, channels)))
    return ConvexFeasibilityResidualNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=channels,
        hidden_dim=hidden_dim,
        depth=int(cfg.get("depth", 2)),
        latent_dim=int(cfg.get("latent_dim", 64)),
        halfspace_constraints=int(cfg.get("halfspace_constraints", cfg.get("num_halfspaces", 32))),
        ball_constraints=int(cfg.get("ball_constraints", cfg.get("num_balls", 16))),
        ball_dim=int(cfg.get("ball_dim", 16)),
        projection_steps=int(cfg.get("projection_steps", 3)),
        step_size=float(cfg.get("step_size", 0.25)),
        hinge_temperature=float(cfg.get("hinge_temperature", 0.1)),
        gate_temperature=float(cfg.get("gate_temperature", 8.0)),
        head_hidden=int(cfg.get("head_hidden", max(128, hidden_dim * 2))),
        dropout=float(cfg.get("dropout", 0.1)),
        mode=str(cfg.get("mode", "projection")),
        include_coordinates=bool(cfg.get("include_coordinates", True)),
    )
