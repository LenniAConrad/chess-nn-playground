from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


FAMILY_NAMES = [
    "sheaf",
    "move_delta",
    "transport",
    "symmetry",
    "topology",
    "king_path",
    "logic",
    "grammar",
    "linear_algebra",
    "information",
    "sparse",
    "graph",
    "convex",
    "tempo",
    "robustness",
    "generic",
]

PROFILE_NAMES = [
    "sheaf",
    "move_delta",
    "transport",
    "symmetry",
    "topology",
    "king_path",
    "logic",
    "grammar",
    "linear_algebra",
    "information",
    "sparse",
    "graph",
    "convex",
    "tempo",
    "robustness",
    "spatial_cnn",
    "token_attention",
    "sequence_memory",
    "proof_certificate",
    "defender_reply",
    "barrier_cut",
    "auction_cost",
    "probabilistic",
    "phase_calibration",
]

STATS_DIM = 28 + len(FAMILY_NAMES) + len(PROFILE_NAMES) * 2
MECHANISM_DIM = 32
BRANCH_KEYS = [
    "sheaf",
    "move_delta",
    "transport",
    "symmetry",
    "topology_path",
    "logic_grammar",
    "spectral",
    "information_sparse",
    "graph_attention",
    "sequence_memory",
    "proof_reply",
    "convex_calibration",
    "spatial_cnn",
]


@dataclass(frozen=True)
class PacketProbeConfig:
    input_channels: int = 18
    num_classes: int = 1
    channels: int = 64
    hidden_dim: int = 96
    depth: int = 2
    dropout: float = 0.1
    mechanism_family: str = "generic"
    packet_profile: str = "research_packet"
    use_batchnorm: bool = True


def _family_index(name: str) -> int:
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized not in FAMILY_NAMES:
        normalized = "generic"
    return FAMILY_NAMES.index(normalized)


def _profile_phase(profile: str) -> float:
    digest = hashlib.sha256(profile.encode("utf-8")).digest()
    return float(int.from_bytes(digest[:2], "big") % 1024) / 1024.0


def _normalized_terms(text: str) -> tuple[str, set[str]]:
    normalized = str(text).strip().lower().replace("-", "_").replace(" ", "_")
    return f"_{normalized}_", {token for token in normalized.split("_") if token}


def _contains_term(haystack: str, tokens: set[str], term: str) -> bool:
    term = term.lower().replace("-", "_").replace(" ", "_")
    if "_" in term:
        return f"_{term}_" in haystack
    return term in tokens


def _profile_flags(profile: str, family: str) -> torch.Tensor:
    haystack, tokens = _normalized_terms(f"{profile}_{family}")
    groups = {
        "sheaf": ["sheaf", "hodge", "tension"],
        "move_delta": ["move_delta", "counterfactual", "one_ply", "null_move", "edit"],
        "transport": ["transport", "sinkhorn", "assignment"],
        "symmetry": ["symmetry", "symmetric", "orbit", "automorphism", "mirror", "color_flip", "invariant"],
        "topology": [
            "topology",
            "euler",
            "betti",
            "percolation",
            "curvature",
            "frustration",
            "filtration",
            "radius",
            "geometry",
            "neighborhood",
            "harmonic",
            "potential",
        ],
        "king_path": ["king", "cage", "escape", "shelter", "target"],
        "logic": [
            "logic",
            "clause",
            "resolution",
            "lattice",
            "hinge",
            "boolean",
            "matroid",
            "hall",
            "zeta",
            "concept",
            "formal_concept",
            "bisimulation",
            "fixed_point",
            "verifier",
            "soundness",
            "forest",
            "decision_forest",
            "conjunction",
            "multiplicative",
            "tropical",
            "circuit",
            "scratchpad",
            "disproof",
        ],
        "grammar": ["grammar", "automaton", "ray", "line", "stripe", "walk", "program", "scan", "run_length"],
        "linear_algebra": [
            "spectrum",
            "spectral",
            "matrix",
            "pencil",
            "rank",
            "tucker",
            "tensor",
            "gramian",
            "hessian",
            "nullspace",
            "orthogonal",
            "displacement",
            "moment",
            "schur",
            "bispectral",
            "bitboard",
            "finite_field",
            "commutator",
            "determinantal",
            "grassmannian",
            "procrustes",
            "krylov",
            "resolvent",
            "parity",
            "syndrome",
            "wavelet",
            "tensorsketch",
            "spline",
            "invertible",
            "bilinear",
            "derivative",
            "curl",
            "divergence",
            "morphological",
            "replicator",
            "mobius",
            "constellation",
            "pivot",
            "trace",
            "elimination",
            "row_file",
            "factor",
            "maxout",
            "signature",
            "sylvester",
            "lyapunov",
            "complement",
            "bures",
            "wasserstein",
            "numerical",
            "range",
            "pfaffian",
            "skew",
            "padic",
            "ultrametric",
            "newton",
            "free_probability",
            "r_transform",
            "williamson",
            "symplectic",
            "magnus",
            "bch",
            "coupling",
            "series",
            "riccati",
            "hamiltonian",
            "clifford",
            "rotor",
            "multivector",
            "bivector",
            "tracy",
            "widom",
            "rmt",
            "spacing",
            "lindstrom",
            "gessel",
            "viennot",
            "determinant",
            "toda",
            "isospectral",
            "lax",
            "manakov",
        ],
        "information": [
            "information",
            "surprisal",
            "surprise",
            "codec",
            "entropy",
            "fisher",
            "zobrist",
            "rate",
            "evidence",
            "likelihood",
            "score_field",
            "sieve",
            "bayesian",
            "credal",
            "temperature",
            "variational",
            "inference",
            "variance",
            "agreement",
        ],
        "sparse": ["sparse", "witness", "prototype", "dictionary", "codebook", "expert", "capsule", "motif"],
        "graph": [
            "graph",
            "hypergraph",
            "relation",
            "effective_resistance",
            "markov",
            "attention",
            "slot",
            "transformer",
            "query",
            "token",
            "cross_stitch",
            "defense",
            "defender",
            "reply",
            "reaction",
            "counterplay",
            "safe_reply",
            "option",
            "front_door",
            "causal",
            "interaction",
            "tree",
            "hypernetwork",
        ],
        "convex": [
            "convex",
            "zonotope",
            "projection",
            "submodular",
            "support_function",
            "barrier",
            "cut",
            "boundary",
            "distance",
            "liability",
            "opportunity",
            "funnel",
            "budget",
            "empty_square",
            "hypercut",
        ],
        "tempo": ["tempo", "phase", "timing", "recurrent", "cellular", "iterative", "cascade", "early_exit"],
        "robustness": [
            "robust",
            "dro",
            "margin",
            "calibration",
            "credal",
            "temperature",
            "disentangled",
            "negative_class",
            "stability",
            "dropout",
            "consensus",
        ],
        "spatial_cnn": [
            "cnn",
            "convnet",
            "convnext",
            "fpn",
            "patch",
            "axial",
            "hypercolumn",
            "film",
            "microkernel",
            "dilated",
            "mixer",
        ],
        "token_attention": ["attention", "slot", "transformer", "query", "token", "cross_stitch", "capsule"],
        "sequence_memory": ["scan", "memory", "automaton", "recurrent", "cellular", "state_space", "run_length"],
        "proof_certificate": ["proof", "certificate", "verifier", "obligation", "ledger", "claim", "subgoal"],
        "defender_reply": ["defender", "reply", "reaction", "counterplay", "safe_reply", "exchange"],
        "barrier_cut": ["barrier", "cut", "blocker", "pin", "pinned", "funnel"],
        "auction_cost": ["auction", "cost", "opportunity", "liability", "budget"],
        "probabilistic": ["bayesian", "credal", "variational", "temperature", "uncertainty"],
        "phase_calibration": ["phase", "calibration", "source_rate", "rate_calibrated", "transition"],
    }
    values = []
    for name in PROFILE_NAMES:
        values.append(1.0 if any(_contains_term(haystack, tokens, term) for term in groups[name]) else 0.0)
    if not any(values):
        values[PROFILE_NAMES.index("spatial_cnn")] = 1.0
    return torch.tensor(values, dtype=torch.float32)


def _scalar_mean(value: torch.Tensor) -> torch.Tensor:
    return value.mean(dim=(1, 2, 3), keepdim=True).view(-1, 1)


def infer_mechanism_family(profile: str) -> str:
    """Infer the mechanism family for direct registry calls without a full idea config."""

    flags = _profile_flags(profile, "generic")
    for name in FAMILY_NAMES:
        if name in PROFILE_NAMES and float(flags[PROFILE_NAMES.index(name)]) > 0.0:
            return name
    for profile_name, family in [
        ("spatial_cnn", "generic"),
        ("token_attention", "graph"),
        ("sequence_memory", "grammar"),
        ("proof_certificate", "logic"),
        ("defender_reply", "graph"),
        ("barrier_cut", "convex"),
        ("auction_cost", "convex"),
        ("probabilistic", "information"),
        ("phase_calibration", "tempo"),
    ]:
        if float(flags[PROFILE_NAMES.index(profile_name)]) > 0.0:
            return family
    return "generic"


def _profile_signature(profile: str, family: str, dim: int) -> torch.Tensor:
    seed = f"{family}:{profile}".encode("utf-8")
    values: list[float] = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(seed + counter.to_bytes(2, "big")).digest()
        values.extend((float(byte) / 127.5) - 1.0 for byte in digest)
        counter += 1
    return torch.tensor(values[:dim], dtype=torch.float32)


def _piece_occupancy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    piece = x[:, : min(12, x.shape[1])].clamp(0.0, 1.0)
    if piece.shape[1] < 12:
        piece = F.pad(piece, (0, 0, 0, 0, 0, 12 - piece.shape[1]))
    first_side = piece[:, :6]
    second_side = piece[:, 6:12]
    if x.shape[1] <= 18 and x.shape[1] > 12:
        white_to_move = x[:, 12:13].mean(dim=(2, 3), keepdim=True).clamp(0.0, 1.0)
        own_planes = white_to_move * first_side + (1.0 - white_to_move) * second_side
        opp_planes = white_to_move * second_side + (1.0 - white_to_move) * first_side
    else:
        own_planes = first_side
        opp_planes = second_side
    own = own_planes.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    opp = opp_planes.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    occupancy = (own + opp).clamp(0.0, 1.0)
    empty = 1.0 - occupancy
    return piece, own, opp, occupancy, empty


def _diagnostic_matrix(diagnostics: dict[str, torch.Tensor], names: list[str], batch_size: int) -> torch.Tensor:
    values = []
    template = next(iter(diagnostics.values()))
    for name in names:
        value = diagnostics.get(name)
        if value is None:
            value = template.new_zeros(batch_size)
        values.append(value.view(batch_size, -1)[:, 0])
    return torch.stack(values, dim=1)


class SheafTensionMechanism(nn.Module):
    def __init__(self, channels: int, stats_dim: int, output_dim: int) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, output_dim, kernel_size=1),
        )
        self.stats = nn.Sequential(nn.Linear(stats_dim + 3, output_dim), nn.LayerNorm(output_dim), nn.GELU())

    def forward(
        self,
        x: torch.Tensor,
        board: torch.Tensor,
        stats: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        horizontal = (board[:, :, :, 1:] - board[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
        vertical = (board[:, :, 1:, :] - board[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
        laplacian = (
            board
            - 0.25
            * (
                torch.roll(board, shifts=1, dims=2)
                + torch.roll(board, shifts=-1, dims=2)
                + torch.roll(board, shifts=1, dims=3)
                + torch.roll(board, shifts=-1, dims=3)
            )
        ).abs()
        tension = laplacian.mean(dim=(1, 2, 3))
        local = self.local(laplacian).mean(dim=(2, 3))
        features = local + self.stats(torch.cat([stats, horizontal[:, None], vertical[:, None], tension[:, None]], dim=1))
        return features, {
            "branch_sheaf_laplacian": tension,
            "branch_sheaf_horizontal": horizontal,
            "branch_sheaf_vertical": vertical,
        }


class MoveDeltaMechanism(nn.Module):
    def __init__(self, channels: int, stats_dim: int, output_dim: int) -> None:
        super().__init__()
        self.delta_conv = nn.Sequential(nn.Conv2d(channels * 4, output_dim, kernel_size=1), nn.GELU())
        self.stats = nn.Sequential(nn.Linear(stats_dim + 4, output_dim), nn.LayerNorm(output_dim), nn.GELU())

    def forward(
        self,
        x: torch.Tensor,
        board: torch.Tensor,
        stats: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        deltas = [
            board - torch.roll(board, shifts=1, dims=2),
            board - torch.roll(board, shifts=-1, dims=2),
            board - torch.roll(board, shifts=1, dims=3),
            board - torch.roll(board, shifts=-1, dims=3),
        ]
        delta_board = torch.cat(deltas, dim=1)
        pooled_delta = self.delta_conv(delta_board).mean(dim=(2, 3))
        delta_stats = torch.stack([delta.abs().mean(dim=(1, 2, 3)) for delta in deltas], dim=1)
        features = pooled_delta + self.stats(torch.cat([stats, delta_stats], dim=1))
        return features, {
            "branch_move_delta_rank": delta_stats[:, :2].mean(dim=1),
            "branch_move_delta_file": delta_stats[:, 2:].mean(dim=1),
        }


class TransportMechanism(nn.Module):
    def __init__(self, stats_dim: int, output_dim: int) -> None:
        super().__init__()
        self.transport = nn.Sequential(nn.Linear(32, output_dim), nn.LayerNorm(output_dim), nn.GELU())
        self.stats = nn.Sequential(nn.Linear(stats_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU())

    def forward(
        self,
        x: torch.Tensor,
        board: torch.Tensor,
        stats: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _piece, own, opp, occupancy, _empty = _piece_occupancy(x)
        own_rank = own.sum(dim=3).squeeze(1) / 8.0
        own_file = own.sum(dim=2).squeeze(1) / 8.0
        opp_rank = opp.sum(dim=3).squeeze(1) / 8.0
        opp_file = opp.sum(dim=2).squeeze(1) / 8.0
        occ_rank = occupancy.sum(dim=3).squeeze(1) / 8.0
        occ_file = occupancy.sum(dim=2).squeeze(1) / 8.0
        margins = torch.cat([own_rank - opp_rank, own_file - opp_file, occ_rank, occ_file], dim=1)
        imbalance = margins[:, :16].abs().mean(dim=1)
        features = self.transport(margins) + self.stats(stats)
        return features, {"branch_transport_margin": imbalance}


class SymmetryOrbitMechanism(nn.Module):
    def __init__(self, channels: int, stats_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(stats_dim + channels * 3 + 4, output_dim), nn.LayerNorm(output_dim), nn.GELU())

    def forward(
        self,
        x: torch.Tensor,
        board: torch.Tensor,
        stats: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        file_flip = (board - torch.flip(board, dims=[3])).abs().mean(dim=(2, 3))
        rank_flip = (board - torch.flip(board, dims=[2])).abs().mean(dim=(2, 3))
        color_flip = (board - torch.flip(board, dims=[2, 3])).abs().mean(dim=(2, 3))
        scalars = torch.stack(
            [
                file_flip.mean(dim=1),
                rank_flip.mean(dim=1),
                color_flip.mean(dim=1),
                (file_flip - rank_flip).abs().mean(dim=1),
            ],
            dim=1,
        )
        features = self.proj(torch.cat([stats, file_flip, rank_flip, color_flip, scalars], dim=1))
        return features, {
            "branch_file_orbit_residual": scalars[:, 0],
            "branch_rank_orbit_residual": scalars[:, 1],
            "branch_color_orbit_residual": scalars[:, 2],
        }


class TopologyPathMechanism(nn.Module):
    def __init__(self, channels: int, stats_dim: int, output_dim: int) -> None:
        super().__init__()
        self.path_conv = nn.Sequential(
            nn.Conv2d(3, output_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.stats = nn.Sequential(nn.Linear(stats_dim + 3, output_dim), nn.LayerNorm(output_dim), nn.GELU())

    def forward(
        self,
        x: torch.Tensor,
        board: torch.Tensor,
        stats: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _piece, _own, _opp, occupancy, empty = _piece_occupancy(x)
        local_empty = F.avg_pool2d(empty, kernel_size=3, stride=1, padding=1)
        bottleneck = (local_empty * occupancy).mean(dim=(1, 2, 3))
        frontier = (local_empty - empty).abs().mean(dim=(1, 2, 3))
        board_pressure = board.abs().mean(dim=1, keepdim=True)
        maps = torch.cat([empty, local_empty, board_pressure], dim=1)
        pooled = self.path_conv(maps).mean(dim=(2, 3))
        scalars = torch.stack([bottleneck, frontier, board_pressure.mean(dim=(1, 2, 3))], dim=1)
        return pooled + self.stats(torch.cat([stats, scalars], dim=1)), {
            "branch_path_bottleneck": bottleneck,
            "branch_path_frontier": frontier,
        }


class LogicGrammarMechanism(nn.Module):
    def __init__(self, channels: int, stats_dim: int, output_dim: int) -> None:
        super().__init__()
        self.rank_scan = nn.Conv1d(channels, output_dim, kernel_size=3, padding=1)
        self.file_scan = nn.Conv1d(channels, output_dim, kernel_size=3, padding=1)
        self.stats = nn.Sequential(nn.Linear(stats_dim + output_dim * 2, output_dim), nn.LayerNorm(output_dim), nn.GELU())

    def forward(
        self,
        x: torch.Tensor,
        board: torch.Tensor,
        stats: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rank_tokens = board.mean(dim=3)
        file_tokens = board.mean(dim=2)
        rank_features = self.rank_scan(rank_tokens).amax(dim=2)
        file_features = self.file_scan(file_tokens).amax(dim=2)
        features = self.stats(torch.cat([stats, rank_features, file_features], dim=1))
        return features, {
            "branch_rank_grammar": rank_features.mean(dim=1),
            "branch_file_grammar": file_features.mean(dim=1),
        }


class SpectralMechanism(nn.Module):
    def __init__(self, channels: int, stats_dim: int, output_dim: int) -> None:
        super().__init__()
        rank_dim = min(12, output_dim)
        self.token_projection = nn.Linear(channels, rank_dim)
        self.spectrum = nn.Sequential(nn.Linear(stats_dim + rank_dim * 2 + 2, output_dim), nn.LayerNorm(output_dim), nn.GELU())

    def forward(
        self,
        x: torch.Tensor,
        board: torch.Tensor,
        stats: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tokens = self.token_projection(board.flatten(2).transpose(1, 2))
        gram = torch.matmul(tokens.transpose(1, 2), tokens) / float(tokens.shape[1])
        diagonal = gram.diagonal(dim1=1, dim2=2)
        offdiag = (gram - torch.diag_embed(diagonal)).abs().mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        trace = diagonal.sum(dim=1, keepdim=True)
        features = self.spectrum(torch.cat([stats, diagonal, diagonal.abs(), trace, offdiag], dim=1))
        return features, {
            "branch_spectral_trace": trace.view(-1),
            "branch_spectral_offdiag": offdiag.view(-1),
        }


class InformationSparseMechanism(nn.Module):
    def __init__(self, channels: int, stats_dim: int, output_dim: int) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(8, channels) * 0.02)
        self.proj = nn.Sequential(nn.Linear(stats_dim + 8 + 3, output_dim), nn.LayerNorm(output_dim), nn.GELU())

    def forward(
        self,
        x: torch.Tensor,
        board: torch.Tensor,
        stats: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tokens = board.flatten(2).transpose(1, 2)
        distances = torch.cdist(tokens, self.prototypes.to(dtype=board.dtype).unsqueeze(0).expand(x.shape[0], -1, -1))
        assignments = torch.softmax(-distances, dim=2).mean(dim=1)
        probabilities = torch.sigmoid(board).flatten(1).clamp(1e-5, 1.0 - 1e-5)
        entropy = -(probabilities * probabilities.log() + (1.0 - probabilities) * (1.0 - probabilities).log()).mean(dim=1)
        sparse_energy = board.abs().mean(dim=(1, 2, 3))
        prototype_peak = assignments.amax(dim=1)
        features = self.proj(torch.cat([stats, assignments, entropy[:, None], sparse_energy[:, None], prototype_peak[:, None]], dim=1))
        return features, {
            "branch_entropy": entropy,
            "branch_sparse_energy": sparse_energy,
            "branch_prototype_peak": prototype_peak,
        }


class GraphAttentionMechanism(nn.Module):
    def __init__(self, channels: int, stats_dim: int, output_dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(channels, output_dim)
        self.k = nn.Linear(channels, output_dim)
        self.v = nn.Linear(channels, output_dim)
        self.out = nn.Sequential(nn.Linear(stats_dim + output_dim * 2 + 1, output_dim), nn.LayerNorm(output_dim), nn.GELU())

    def forward(
        self,
        x: torch.Tensor,
        board: torch.Tensor,
        stats: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tokens = board.flatten(2).transpose(1, 2)
        query = self.q(tokens)
        key = self.k(tokens)
        value = self.v(tokens)
        attention = torch.softmax(torch.matmul(query, key.transpose(1, 2)) / math.sqrt(query.shape[-1]), dim=-1)
        message = torch.matmul(attention, value)
        pooled = torch.cat([message.mean(dim=1), message.amax(dim=1)], dim=1)
        concentration = attention.amax(dim=-1).mean(dim=1, keepdim=True)
        features = self.out(torch.cat([stats, pooled, concentration], dim=1))
        return features, {"branch_attention_concentration": concentration.view(-1)}


class SpatialCNNMechanism(nn.Module):
    def __init__(self, channels: int, output_dim: int) -> None:
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, output_dim, kernel_size=1),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        board: torch.Tensor,
        stats: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = self.depthwise(board)
        local_peak = features.amax(dim=(1, 2, 3))
        return features.mean(dim=(2, 3)), {"branch_spatial_peak": local_peak}


class SequenceMemoryMechanism(nn.Module):
    def __init__(self, channels: int, output_dim: int) -> None:
        super().__init__()
        self.rank_gru = nn.GRU(channels, output_dim, batch_first=True)
        self.file_gru = nn.GRU(channels, output_dim, batch_first=True)
        self.mix = nn.Sequential(nn.Linear(output_dim * 2, output_dim), nn.LayerNorm(output_dim), nn.GELU())

    def forward(
        self,
        x: torch.Tensor,
        board: torch.Tensor,
        stats: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rank_sequence = board.mean(dim=3).transpose(1, 2)
        file_sequence = board.mean(dim=2).transpose(1, 2)
        rank_out, _ = self.rank_gru(rank_sequence)
        file_out, _ = self.file_gru(file_sequence)
        features = self.mix(torch.cat([rank_out[:, -1], file_out[:, -1]], dim=1))
        return features, {
            "branch_rank_memory": rank_out[:, -1].abs().mean(dim=1),
            "branch_file_memory": file_out[:, -1].abs().mean(dim=1),
        }


class ProofReplyMechanism(nn.Module):
    def __init__(self, stats_dim: int, output_dim: int) -> None:
        super().__init__()
        names = [
            "reply_pressure",
            "defense_gap",
            "king_ring_pressure",
            "ray_language_energy",
            "topology_pressure",
            "material_delta",
            "material_total",
        ]
        self.names = names
        self.net = nn.Sequential(nn.Linear(stats_dim + len(names), output_dim), nn.LayerNorm(output_dim), nn.GELU())

    def forward(
        self,
        x: torch.Tensor,
        board: torch.Tensor,
        stats: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        diag = _diagnostic_matrix(diagnostics, self.names, x.shape[0])
        certificate_pressure = diag[:, :4].mean(dim=1)
        features = self.net(torch.cat([stats, diag], dim=1))
        return features, {"branch_certificate_pressure": certificate_pressure}


class ConvexCalibrationMechanism(nn.Module):
    def __init__(self, channels: int, stats_dim: int, output_dim: int) -> None:
        super().__init__()
        self.support = nn.Sequential(nn.Linear(channels * 2 + stats_dim + 4, output_dim), nn.LayerNorm(output_dim), nn.GELU())

    def forward(
        self,
        x: torch.Tensor,
        board: torch.Tensor,
        stats: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mean = board.mean(dim=(2, 3))
        peak = board.amax(dim=(2, 3))
        barrier = F.softplus(board).mean(dim=(1, 2, 3))
        margin = board.abs().amax(dim=(1, 2, 3))
        variance = board.flatten(1).var(dim=1, unbiased=False)
        phase = diagnostics.get("material_total", barrier).view(x.shape[0])
        scalars = torch.stack([barrier, margin, variance, phase], dim=1)
        features = self.support(torch.cat([mean, peak, stats, scalars], dim=1))
        return features, {
            "branch_convex_barrier": barrier,
            "branch_calibration_variance": variance,
        }


def _branch_key(profile: str, family: str) -> str:
    flags = _profile_flags(profile, family)

    def active(name: str) -> bool:
        return float(flags[PROFILE_NAMES.index(name)]) > 0.0

    if active("spatial_cnn"):
        return "spatial_cnn"
    if active("sequence_memory"):
        return "sequence_memory"
    if active("token_attention"):
        return "graph_attention"
    if active("proof_certificate") or active("defender_reply"):
        return "proof_reply"
    if active("barrier_cut") or active("auction_cost") or active("probabilistic") or active("phase_calibration"):
        return "convex_calibration"
    family = family if family in FAMILY_NAMES else infer_mechanism_family(profile)
    return {
        "sheaf": "sheaf",
        "move_delta": "move_delta",
        "transport": "transport",
        "symmetry": "symmetry",
        "topology": "topology_path",
        "king_path": "topology_path",
        "logic": "logic_grammar",
        "grammar": "logic_grammar",
        "linear_algebra": "spectral",
        "information": "information_sparse",
        "sparse": "information_sparse",
        "graph": "graph_attention",
        "convex": "convex_calibration",
        "tempo": "sequence_memory",
        "robustness": "convex_calibration",
    }.get(family, "spatial_cnn")


def _make_mechanism_branch(key: str, channels: int, stats_dim: int, output_dim: int) -> nn.Module:
    if key == "sheaf":
        return SheafTensionMechanism(channels, stats_dim, output_dim)
    if key == "move_delta":
        return MoveDeltaMechanism(channels, stats_dim, output_dim)
    if key == "transport":
        return TransportMechanism(stats_dim, output_dim)
    if key == "symmetry":
        return SymmetryOrbitMechanism(channels, stats_dim, output_dim)
    if key == "topology_path":
        return TopologyPathMechanism(channels, stats_dim, output_dim)
    if key == "logic_grammar":
        return LogicGrammarMechanism(channels, stats_dim, output_dim)
    if key == "spectral":
        return SpectralMechanism(channels, stats_dim, output_dim)
    if key == "information_sparse":
        return InformationSparseMechanism(channels, stats_dim, output_dim)
    if key == "graph_attention":
        return GraphAttentionMechanism(channels, stats_dim, output_dim)
    if key == "sequence_memory":
        return SequenceMemoryMechanism(channels, output_dim)
    if key == "proof_reply":
        return ProofReplyMechanism(stats_dim, output_dim)
    if key == "convex_calibration":
        return ConvexCalibrationMechanism(channels, stats_dim, output_dim)
    return SpatialCNNMechanism(channels, output_dim)


class ResearchPacketProbe(nn.Module):
    """Mechanism-profiled first-pass implementation for promoted research packets.

    The packet profile keeps each promoted packet distinct while sharing a
    board-only implementation scaffold. The family branch and profile flags
    choose deterministic diagnostics aligned with the packet thesis: sheaf
    tension, transport moments, symmetry residuals, topology/path pressure,
    logic/grammar ray evidence, spectral summaries, information bottleneck
    scores, sparse/convex energies, tempo deltas, robustness margins, spatial
    CNN cues, token/attention cues, certificate/reply cues, and cost/auction
    proxies.
    """

    def __init__(self, cfg: PacketProbeConfig) -> None:
        super().__init__()
        if cfg.num_classes != 1:
            raise ValueError("ResearchPacketProbe supports the puzzle_binary one-logit benchmark contract")
        if cfg.depth < 1:
            raise ValueError("depth must be >= 1")
        self.cfg = cfg
        self.spec = BoardTensorSpec(input_channels=cfg.input_channels)
        self.profile_count = len(PROFILE_NAMES)
        layers: list[nn.Module] = []
        in_channels = cfg.input_channels
        for _ in range(cfg.depth):
            layers.append(nn.Conv2d(in_channels, cfg.channels, kernel_size=3, padding=1, bias=not cfg.use_batchnorm))
            if cfg.use_batchnorm:
                layers.append(nn.BatchNorm2d(cfg.channels))
            layers.append(nn.GELU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout2d(cfg.dropout))
            in_channels = cfg.channels
        self.stem = nn.Sequential(*layers)
        family_count = len(FAMILY_NAMES)
        stats_dim = STATS_DIM
        self.mechanism_key = _branch_key(cfg.packet_profile, cfg.mechanism_family)
        self.mechanism_id = BRANCH_KEYS.index(self.mechanism_key) if self.mechanism_key in BRANCH_KEYS else len(BRANCH_KEYS)
        self.mechanism = _make_mechanism_branch(
            self.mechanism_key,
            channels=cfg.channels,
            stats_dim=stats_dim,
            output_dim=MECHANISM_DIM,
        )
        pooled_dim = cfg.channels * 2 + stats_dim + MECHANISM_DIM
        self.family_embedding = nn.Embedding(family_count, 8)
        self.profile_projection = nn.Linear(4, 8)
        self.profile_gate = nn.Sequential(
            nn.Linear(family_count + self.profile_count + 4, MECHANISM_DIM),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(pooled_dim + 16, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.aux_head = nn.Linear(pooled_dim + 16, 1)
        self.family_id = _family_index(cfg.mechanism_family)
        phase = _profile_phase(cfg.packet_profile)
        angle = phase * 2.0 * math.pi
        profile = torch.tensor([phase, phase * phase, math.sin(angle), math.cos(angle)])
        self.register_buffer("profile_vector", profile.float())
        self.register_buffer("family_one_hot", F.one_hot(torch.tensor(self.family_id), num_classes=family_count).float())
        self.register_buffer("profile_flags", _profile_flags(cfg.packet_profile, cfg.mechanism_family))
        self.register_buffer("profile_signature", _profile_signature(cfg.packet_profile, cfg.mechanism_family, MECHANISM_DIM))

        cross = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        diag = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
        full = torch.ones(3, 3)
        self.register_buffer("kernels", torch.stack([cross, diag, full], dim=0).unsqueeze(1))

    def _piece_planes(self, x: torch.Tensor) -> torch.Tensor:
        piece = x[:, : min(12, x.shape[1])].clamp(0.0, 1.0)
        if piece.shape[1] < 12:
            piece = F.pad(piece, (0, 0, 0, 0, 0, 12 - piece.shape[1]))
        return piece

    def _stats(self, x: torch.Tensor, board: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        piece, own, opp, occupancy, empty = _piece_occupancy(x)
        kernels = self.kernels.to(dtype=x.dtype)
        local = F.conv2d(occupancy, kernels, padding=1)
        own_pressure = F.conv2d(own, kernels[:1], padding=1)
        opp_pressure = F.conv2d(opp, kernels[1:2], padding=1)
        pressure_gap = own_pressure - opp_pressure
        if x.shape[1] <= 18 and x.shape[1] > 12:
            white_to_move = x[:, 12:13].mean(dim=(2, 3), keepdim=True).clamp(0.0, 1.0)
            own_planes = white_to_move * piece[:, :6] + (1.0 - white_to_move) * piece[:, 6:12]
            opp_planes = white_to_move * piece[:, 6:12] + (1.0 - white_to_move) * piece[:, :6]
        else:
            own_planes = piece[:, :6]
            opp_planes = piece[:, 6:12]
        piece_balance = own_planes - opp_planes
        rank_mass = occupancy.sum(dim=3).squeeze(1) / 8.0
        file_mass = occupancy.sum(dim=2).squeeze(1) / 8.0
        rank_file_imbalance = (rank_mass.std(dim=1, keepdim=True) + file_mass.std(dim=1, keepdim=True)) * 0.5
        diagonals = local[:, 1:2]
        adjacency = local[:, 0:1]
        king_like = (piece[:, 5:6] + piece[:, 11:12]).clamp(0.0, 1.0)
        king_ring = F.max_pool2d(king_like, kernel_size=3, stride=1, padding=1)
        reply_pressure = opp_pressure.abs().mean(dim=(1, 2, 3), keepdim=True).view(-1, 1)
        defense_gap = pressure_gap.abs().mean(dim=(1, 2, 3), keepdim=True).view(-1, 1)
        center = occupancy[:, :, 2:6, 2:6].mean(dim=(1, 2, 3))
        rim = (
            occupancy[:, :, 0].mean(dim=(1, 2))
            + occupancy[:, :, 7].mean(dim=(1, 2))
            + occupancy[:, :, :, 0].mean(dim=(1, 2))
            + occupancy[:, :, :, 7].mean(dim=(1, 2))
        ) / 4.0
        counts = piece.flatten(2).sum(dim=2) / 16.0
        side = x[:, 12:13].mean(dim=(2, 3)) if x.shape[1] > 12 else counts.new_zeros(counts.shape[0], 1)
        own_counts = own_planes.flatten(2).sum(dim=(1, 2), keepdim=False).unsqueeze(1) / 16.0
        opp_counts = opp_planes.flatten(2).sum(dim=(1, 2), keepdim=False).unsqueeze(1) / 16.0
        material_delta = own_counts - opp_counts
        material_total = counts.sum(dim=1, keepdim=True)
        board_mean = board.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        board_std = board.flatten(1).std(dim=1, keepdim=True)
        mirror_residual = (board - torch.flip(board, dims=[3])).abs().mean(dim=(1, 2, 3), keepdim=True).view(-1, 1)
        color_residual = (piece[:, :6] - torch.flip(piece[:, 6:12], dims=[2, 3])).abs().mean(dim=(1, 2, 3), keepdim=True).view(-1, 1)
        ray_energy = local.mean(dim=(1, 2, 3), keepdim=True).view(-1, 1)
        sheaf_tension = pressure_gap.abs().mean(dim=(1, 2, 3), keepdim=True).view(-1, 1)
        transport_imbalance = (own.flatten(1).sum(dim=1, keepdim=True) - opp.flatten(1).sum(dim=1, keepdim=True)).abs() / 16.0
        empty_connectivity = F.avg_pool2d(empty, kernel_size=3, stride=1, padding=1).mean(dim=(1, 2, 3), keepdim=True).view(-1, 1)
        entropy = -(occupancy.clamp_min(1e-4) * occupancy.clamp_min(1e-4).log()).mean(dim=(1, 2, 3), keepdim=True).view(-1, 1)
        sparse_energy = board.abs().mean(dim=(1, 2, 3), keepdim=True).view(-1, 1)
        topology_pressure = (local[:, 2:3] * empty).mean(dim=(1, 2, 3), keepdim=True).view(-1, 1)
        corner_pressure = (
            occupancy[:, :, :2, :2].mean(dim=(1, 2, 3), keepdim=True)
            + occupancy[:, :, :2, -2:].mean(dim=(1, 2, 3), keepdim=True)
            + occupancy[:, :, -2:, :2].mean(dim=(1, 2, 3), keepdim=True)
            + occupancy[:, :, -2:, -2:].mean(dim=(1, 2, 3), keepdim=True)
        ).view(-1, 1) / 4.0
        phase_pressure = material_total.clamp_min(0.0) / 8.0
        token_dispersion = board.flatten(2).std(dim=2).mean(dim=1, keepdim=True)
        pairfield_energy = _scalar_mean(piece_balance.abs())
        proposal_candidates = torch.cat(
            [
                sheaf_tension,
                ray_energy,
                transport_imbalance,
                mirror_residual + color_residual,
                topology_pressure,
                empty_connectivity + _scalar_mean(king_ring * empty),
                ray_energy + _scalar_mean(diagonals),
                ray_energy + adjacency.mean(dim=(1, 2, 3), keepdim=True).view(-1, 1),
                board_std + rank_file_imbalance,
                entropy,
                sparse_energy,
                topology_pressure + reply_pressure,
                sparse_energy + corner_pressure,
                material_delta.abs(),
                defense_gap,
                rank_file_imbalance + _scalar_mean(diagonals),
                token_dispersion,
                rank_file_imbalance + _scalar_mean(king_ring),
                reply_pressure + defense_gap,
                reply_pressure,
                corner_pressure + empty_connectivity,
                material_total.abs() + defense_gap,
                entropy + board_std,
                phase_pressure + entropy,
            ],
            dim=1,
        )
        flags = self.profile_flags.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1)
        proposal_features = proposal_candidates * flags
        family = self.family_one_hot.unsqueeze(0).expand(x.shape[0], -1)
        raw_stats = torch.cat(
            [
                counts,
                side,
                material_delta,
                material_total,
                center.unsqueeze(1),
                rim.unsqueeze(1),
                board_mean,
                board_std,
                mirror_residual,
                color_residual,
                ray_energy,
                sheaf_tension,
                transport_imbalance,
                empty_connectivity,
                entropy,
                sparse_energy,
                topology_pressure,
                family,
                flags,
                proposal_features,
            ],
            dim=1,
        )
        family_energy = {
            "sheaf": sheaf_tension,
            "move_delta": ray_energy,
            "transport": transport_imbalance,
            "symmetry": mirror_residual + color_residual,
            "topology": topology_pressure,
            "king_path": empty_connectivity,
            "logic": ray_energy,
            "grammar": ray_energy,
            "linear_algebra": board_std,
            "information": entropy,
            "sparse": sparse_energy,
            "graph": topology_pressure,
            "convex": sparse_energy,
            "tempo": material_delta.abs(),
            "robustness": pressure_gap.abs().amax(dim=(1, 2, 3), keepdim=True).view(-1, 1),
            "generic": board_mean.abs() + board_std,
        }.get(self.cfg.mechanism_family, board_mean.abs() + board_std)
        active_profiles = flags.sum(dim=1).clamp_min(1.0)
        diagnostics = {
            "mechanism_energy": family_energy.view(-1),
            "proposal_profile_strength": (proposal_features.sum(dim=1) / active_profiles).view(-1),
            "proposal_keyword_count": flags.sum(dim=1),
            "sheaf_tension": sheaf_tension.view(-1),
            "transport_imbalance": transport_imbalance.view(-1),
            "symmetry_residual": mirror_residual.view(-1),
            "color_orbit_residual": color_residual.view(-1),
            "topology_pressure": topology_pressure.view(-1),
            "ray_language_energy": ray_energy.view(-1),
            "information_surprisal": entropy.view(-1),
            "sparse_certificate_energy": sparse_energy.view(-1),
            "rank_file_imbalance": rank_file_imbalance.view(-1),
            "king_ring_pressure": _scalar_mean(king_ring).view(-1),
            "reply_pressure": reply_pressure.view(-1),
            "defense_gap": defense_gap.view(-1),
            "token_dispersion": token_dispersion.view(-1),
            "material_delta": material_delta.view(-1),
            "material_total": material_total.view(-1),
        }
        return raw_stats, diagnostics

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.stem(x)
        stats, diagnostics = self._stats(x, board)
        mechanism_features, mechanism_diagnostics = self.mechanism(x, board, stats, diagnostics)
        family = self.family_one_hot.unsqueeze(0).expand(x.shape[0], -1)
        flags = self.profile_flags.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1)
        profile_vector = self.profile_vector.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1)
        gate = self.profile_gate(torch.cat([family, flags, profile_vector], dim=1))
        signature = self.profile_signature.to(device=x.device, dtype=x.dtype).unsqueeze(0)
        mechanism_features = mechanism_features * (0.5 + gate) + 0.05 * signature
        pooled = torch.cat([board.mean(dim=(2, 3)), board.amax(dim=(2, 3)), stats, mechanism_features], dim=1)
        family_idx = torch.full((x.shape[0],), self.family_id, dtype=torch.long, device=x.device)
        family_embedding = self.family_embedding(family_idx)
        profile = self.profile_projection(profile_vector)
        features = torch.cat([pooled, family_embedding, profile], dim=1)
        logits = self.head(features).view(-1)
        aux = self.aux_head(features).view(-1)
        return {
            "logits": logits,
            "packet_aux_logit": aux,
            "packet_family_id": torch.full_like(logits, float(self.family_id)),
            "packet_mechanism_id": torch.full_like(logits, float(self.mechanism_id)),
            "packet_profile_phase": torch.full_like(logits, float(self.profile_vector[0].detach().cpu())),
            **diagnostics,
            **mechanism_diagnostics,
        }


def build_research_packet_probe_from_config(config: dict[str, Any]) -> ResearchPacketProbe:
    cfg = PacketProbeConfig(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        mechanism_family=str(config.get("mechanism_family", "generic")),
        packet_profile=str(config.get("packet_profile", config.get("name", "research_packet"))),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
    )
    return ResearchPacketProbe(cfg)
