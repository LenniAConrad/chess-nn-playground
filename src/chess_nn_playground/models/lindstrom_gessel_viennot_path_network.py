"""Lindstrom-Gessel-Viennot Path Determinant Network for idea i234.

The model assembles a learned chess DAG, builds the path-generating-function
matrix ``G = sum_{k>=1} (alpha W)^k`` via a truncated Neumann series, then
extracts a path matrix ``M[i, j]`` between ``num_paths`` soft-selected source
and target squares.  By the Lindstrom-Gessel-Viennot lemma, ``det(M)`` is the
signed enumerator of non-intersecting attacker-to-target ``num_paths``-tuples,
so its log absolute value and sign are the central tactical invariants used by
the puzzle head.

The mechanism is materially distinct from any spectrum-based architecture
(Hadamard / Schur / Sylvester / Hessian / level-spacing), because the
classifier reads the *path determinant* and per-pair path counts of a
DAG-restricted Neumann resolvent, not eigenvalues of a single Hermitian
operator.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


class _Trunk(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(max(1, depth)):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    bias=not use_batchnorm,
                )
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = channels
        self.trunk = nn.Sequential(*layers)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class LindstromGesselViennotPathNetwork(nn.Module):
    """Bespoke implementation of idea i234.

    The model is intentionally board-only; CRTK / source metadata is
    reporting-only and never consumed as input.
    """

    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        edge_embed_dim: int = 24,
        num_paths: int = 4,
        neumann_steps: int = 4,
        source_target_temperature: float = 1.0,
        alpha_init: float = 0.6,
        det_eps: float = 1.0e-4,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "LindstromGesselViennotPathNetwork supports the puzzle_binary one-logit contract"
            )
        if num_paths < 2:
            raise ValueError("num_paths must be >= 2 to form a non-trivial path determinant")
        if neumann_steps < 1:
            raise ValueError("neumann_steps must be >= 1")
        if edge_embed_dim < 1:
            raise ValueError("edge_embed_dim must be >= 1")
        if not 0.0 < alpha_init < 1.0:
            raise ValueError("alpha_init must lie in (0, 1)")
        if source_target_temperature <= 0.0:
            raise ValueError("source_target_temperature must be > 0")
        if det_eps <= 0.0:
            raise ValueError("det_eps must be > 0")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.channels = int(channels)
        self.edge_embed_dim = int(edge_embed_dim)
        self.num_paths = int(num_paths)
        self.neumann_steps = int(neumann_steps)
        self.source_target_temperature = float(source_target_temperature)
        self.det_eps = float(det_eps)

        self.trunk = _Trunk(input_channels, channels, depth, dropout, use_batchnorm)

        # Asymmetric edge embeddings: u -> "left" (source side), v -> "right" (target side).
        self.proj_edge_left = nn.Conv2d(channels, edge_embed_dim, kernel_size=1)
        self.proj_edge_right = nn.Conv2d(channels, edge_embed_dim, kernel_size=1)

        # Soft source / target selection: queries pick k squares from the trunk.
        self.proj_src = nn.Conv2d(channels, edge_embed_dim, kernel_size=1)
        self.proj_tgt = nn.Conv2d(channels, edge_embed_dim, kernel_size=1)
        self.source_queries = nn.Parameter(torch.randn(num_paths, edge_embed_dim) * 0.1)
        self.target_queries = nn.Parameter(torch.randn(num_paths, edge_embed_dim) * 0.1)

        # alpha is a sigmoid of a learned logit, capped just below 1 to keep
        # ``alpha * W`` strictly contractive when paired with row-stochastic W.
        alpha_logit = math.log(alpha_init / (1.0 - alpha_init))
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))

        # Strict upper-triangular DAG mask under a row-major topological order
        # of the 64 squares (a8, b8, ..., h1).
        n = 64
        dag_mask = torch.triu(torch.ones(n, n, dtype=torch.float32), diagonal=1)
        self.register_buffer("dag_mask", dag_mask, persistent=False)
        identity = torch.eye(num_paths, dtype=torch.float32)
        self.register_buffer("identity_paths", identity, persistent=False)

        pooled_trunk_dim = 2 * channels
        # Diagnostics fed into head:
        # - log |det M|, sign(det M), trace(M), Frobenius norm, alpha
        # - per-pair diagonal of M (num_paths)
        # - per-pair off-diagonal magnitude (num_paths)
        # - mean entropy of source / target soft selections (2 scalars)
        diagnostic_dim = 5 + 2 * self.num_paths + 2
        head_in = pooled_trunk_dim + diagnostic_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def _edge_weights(self, feat: torch.Tensor) -> torch.Tensor:
        """Build the DAG-masked edge weight matrix W on the 64 squares.

        W[u, v] in [0, 1) by sigmoid; W[u, v] = 0 unless u < v under the
        row-major topological order, ensuring acyclicity.  The row-stochastic
        normalisation keeps the spectral radius < 1 so that
        ``G = sum_{k>=1} (alpha W)^k`` converges and the truncated Neumann
        series is a faithful proxy.
        """
        bsz = feat.shape[0]
        left = self.proj_edge_left(feat).reshape(bsz, self.edge_embed_dim, -1).transpose(1, 2)
        right = self.proj_edge_right(feat).reshape(bsz, self.edge_embed_dim, -1).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.edge_embed_dim)
        scores = torch.matmul(left, right.transpose(-1, -2)) * scale
        gated = torch.sigmoid(scores) * self.dag_mask
        # Row-stochastic normalisation guarantees spectral radius <= 1; combined
        # with alpha < 1 the Neumann series is an absolutely convergent
        # path-generating function.
        row_sum = gated.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return gated / row_sum

    def _path_generating_matrix(self, w: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Truncated Neumann series ``G = sum_{k=1..K} (alpha W)^k``.

        Each ``(alpha W)^k[u, v]`` is the alpha-weighted enumerator of length-k
        directed paths from ``u`` to ``v`` in the learned DAG; their sum is the
        full path-generating function evaluated at ``alpha``.
        """
        a_w = alpha * w
        g = a_w.clone()
        cur = a_w
        for _ in range(self.neumann_steps - 1):
            cur = torch.matmul(cur, a_w)
            g = g + cur
        return g

    def _soft_selection(self, feat: torch.Tensor, kind: str) -> torch.Tensor:
        """Return ``A`` of shape ``(B, num_paths, 64)`` whose rows are softmaxes
        over squares; row ``i`` is the soft choice of the ``i``-th source / target.
        """
        bsz = feat.shape[0]
        if kind == "source":
            proj = self.proj_src(feat)
            queries = self.source_queries
        else:
            proj = self.proj_tgt(feat)
            queries = self.target_queries
        proj_flat = proj.reshape(bsz, self.edge_embed_dim, -1)  # (B, d, 64)
        scores = torch.einsum("kd,bdn->bkn", queries, proj_flat) / self.source_target_temperature
        return F.softmax(scores, dim=-1)

    @staticmethod
    def _selection_entropy(a: torch.Tensor) -> torch.Tensor:
        log_a = (a.clamp_min(1.0e-12)).log()
        return -(a * log_a).sum(dim=-1).mean(dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feat = self.trunk(x)  # (B, C, 8, 8)
        bsz = feat.shape[0]

        w = self._edge_weights(feat)
        alpha = torch.sigmoid(self.alpha_logit) * 0.99
        g = self._path_generating_matrix(w, alpha)

        a_src = self._soft_selection(feat, kind="source")  # (B, K, 64)
        a_tgt = self._soft_selection(feat, kind="target")  # (B, K, 64)
        # Path matrix: M[i, j] = sum_{u, v} A_src[i, u] * G[u, v] * A_tgt[j, v].
        m = torch.matmul(torch.matmul(a_src, g), a_tgt.transpose(-1, -2))

        # det(M) is the signed enumerator of non-intersecting source-to-target
        # k-tuples by the LGV lemma; eps regularisation keeps slogdet finite
        # when the DAG admits no full non-intersecting matching.
        m_safe = m + self.det_eps * self.identity_paths.unsqueeze(0)
        sign, log_abs_det = torch.linalg.slogdet(m_safe)
        log_abs_det = log_abs_det.clamp(min=-30.0, max=30.0)
        sign = sign.to(dtype=feat.dtype)

        diag = m.diagonal(dim1=-2, dim2=-1)  # (B, K)
        trace = diag.sum(dim=-1)
        frobenius = torch.sqrt((m * m).sum(dim=(-2, -1)).clamp_min(1.0e-12))
        offdiag_mask = 1.0 - self.identity_paths
        offdiag_l1_per_row = (m.abs() * offdiag_mask.unsqueeze(0)).sum(dim=-1)  # (B, K)
        offdiag_l1 = offdiag_l1_per_row.sum(dim=-1)

        src_entropy = self._selection_entropy(a_src)
        tgt_entropy = self._selection_entropy(a_tgt)
        alpha_vec = alpha.view(1).expand(bsz)

        pooled_trunk = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)
        feat_vec = torch.cat(
            [
                pooled_trunk,
                log_abs_det.unsqueeze(-1),
                sign.unsqueeze(-1),
                trace.unsqueeze(-1),
                frobenius.unsqueeze(-1),
                alpha_vec.unsqueeze(-1),
                diag,
                offdiag_l1_per_row,
                src_entropy.unsqueeze(-1),
                tgt_entropy.unsqueeze(-1),
            ],
            dim=-1,
        )
        logits = self.head(feat_vec).view(-1)

        return {
            "logits": logits,
            "lgv_log_abs_det": log_abs_det,
            "lgv_det_sign": sign,
            "lgv_path_matrix_trace": trace,
            "lgv_path_matrix_frobenius": frobenius,
            "lgv_path_diagonal": diag,
            "lgv_offdiag_l1_per_row": offdiag_l1_per_row,
            "lgv_offdiag_l1": offdiag_l1,
            "lgv_neumann_alpha": alpha_vec,
            "lgv_source_entropy": src_entropy,
            "lgv_target_entropy": tgt_entropy,
        }


def build_lindstrom_gessel_viennot_path_network_from_config(
    config: dict[str, Any],
) -> LindstromGesselViennotPathNetwork:
    cfg = dict(config)
    return LindstromGesselViennotPathNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        edge_embed_dim=int(cfg.get("edge_embed_dim", 24)),
        num_paths=int(cfg.get("num_paths", 4)),
        neumann_steps=int(cfg.get("neumann_steps", 4)),
        source_target_temperature=float(cfg.get("source_target_temperature", 1.0)),
        alpha_init=float(cfg.get("alpha_init", 0.6)),
        det_eps=float(cfg.get("det_eps", 1.0e-4)),
        num_classes=int(cfg.get("num_classes", 1)),
    )
