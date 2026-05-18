"""i259 i018+BT4 Ensemble Compression — distilled BT4-shaped conv student.

This module promotes the i259 research packet
(``ideas/research/packets/classic/i259_i018_bt4_ensemble_compression.md``)
into a trainable bespoke architecture. The packet's deployment claim is
explicit: the production output is a single BT4-shaped conv student that
inherits the matched-recall near-puzzle gains of an i018 + BT4 teacher
ensemble while staying within BT4-class latency. Running both teachers
forever is research-only; the student is what ships.

Architecture (single network at inference):

    student_logit = ConvTower(simple_18)              # BT4-shaped trunk
    final_logit   = student_logit
    diagnostics   = {
        # always emitted, used by audits and the loss schedule
        student_logit, student_probability,
        # student auxiliary "diagnostic-hint" heads (KD compression targets)
        diagnostic_hint_<name>,
    }

In ``teacher_mode``, the wrapped i018 trunk + a sibling BT4 teacher trunk
also run on the same input. Their logits are exposed as
``teacher_i018_logit``, ``teacher_bt4_logit``, ``teacher_ensemble_logit``,
``teacher_disagreement``, and ``teacher_entropy`` so that distillation
caches and the gated-fusion ablations in
``ideas/registry/i259_i018_bt4_ensemble_compression/ablations.md`` can be
generated from the same network. Teacher parameters are always frozen
(``requires_grad_(False)``) and excluded from gradient flow with
``torch.no_grad()`` so the student loss is well-defined.

The default ``teacher_mode='off'`` keeps the deployment forward fast and
compatible with the shared puzzle_binary trainer (BCE-with-logits on
``logits``). Teacher mode is intended for research-time teacher-cache
generation or oracle evaluation — it can also be left on for full
end-to-end training when ``CLAUDE_ALLOW_TRAINING=1`` and the teachers'
checkpoints are loaded.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import (
    BoardTensorSpec,
    require_board_tensor,
)
from chess_nn_playground.models.trunk.lc0_bt4 import LC0BT4Classifier
from chess_nn_playground.models.trunk.oriented_tactical_sheaf import (
    OrientedTacticalSheafNet,
)


_DIAGNOSTIC_HINT_KEYS: tuple[str, ...] = (
    "sheaf_tension",
    "triad_defect_energy",
    "king_ring_pressure",
    "reply_pressure",
    "defense_gap",
    "pin_pressure",
)

_ALLOWED_TEACHER_MODES = ("off", "research", "frozen")
_ALLOWED_FUSION_MODES = ("equal_weight", "tuned_alpha", "uncertainty_gated")
_ALLOWED_ABLATIONS = (
    "none",
    "student_only",
    "zero_hint_heads",
    "teacher_logits_only",
    "shuffle_teacher_logits",
)


class _StudentConvTower(nn.Module):
    """Compact BT4-shaped student conv tower for the simple_18 encoding.

    Wraps ``LC0BT4Classifier`` so the student inherits the residual BT4
    block + Squeeze-Excite pattern from ``lc0_bt4_classifier``, but adds
    diagnostic-hint auxiliary heads on top of the spatial trunk so KD
    can transfer i018's structural signal as scalar regression targets.
    """

    def __init__(
        self,
        input_channels: int,
        channels: int,
        num_blocks: int,
        se_channels: int,
        value_channels: int,
        value_hidden: int,
        dropout: float,
        use_batchnorm: bool,
        hint_keys: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.classifier = LC0BT4Classifier(
            input_channels=int(input_channels),
            num_classes=1,
            channels=int(channels),
            num_blocks=int(num_blocks),
            value_channels=int(value_channels),
            value_hidden=int(value_hidden),
            se_channels=int(se_channels),
            dropout=float(dropout),
            use_batchnorm=bool(use_batchnorm),
        )
        self.hint_keys = tuple(str(name) for name in hint_keys)
        if self.hint_keys:
            self.hint_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.hint_heads = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.LayerNorm(int(channels)),
                        nn.Linear(int(channels), 1),
                    )
                    for name in self.hint_keys
                }
            )
        else:
            self.hint_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.hint_heads = nn.ModuleDict()
        self.output_channels = int(channels)

    def trunk_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.classifier.stem(x)
        return self.classifier.blocks(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = self.trunk_features(x)
        logits = self.classifier.value_head(features).view(-1)
        hints: dict[str, torch.Tensor] = {}
        if self.hint_heads:
            pooled = self.hint_pool(features).flatten(1)
            for name, head in self.hint_heads.items():
                hints[name] = head(pooled).view(-1)
        return logits, hints


def _build_bt4_teacher(
    *,
    input_channels: int,
    channels: int,
    num_blocks: int,
    se_channels: int,
    value_channels: int,
    value_hidden: int,
    dropout: float,
    use_batchnorm: bool,
) -> LC0BT4Classifier:
    return LC0BT4Classifier(
        input_channels=int(input_channels),
        num_classes=1,
        channels=int(channels),
        num_blocks=int(num_blocks),
        value_channels=int(value_channels),
        value_hidden=int(value_hidden),
        se_channels=int(se_channels),
        dropout=float(dropout),
        use_batchnorm=bool(use_batchnorm),
    )


def _build_i018_teacher(
    *,
    input_channels: int,
    channels: int,
    hidden_dim: int,
    depth: int,
    stalk_dim: int,
    dropout: float,
    use_triads: bool,
    encoding: str,
) -> OrientedTacticalSheafNet:
    return OrientedTacticalSheafNet(
        input_channels=int(input_channels),
        num_classes=1,
        channels=int(channels),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        stalk_dim=int(stalk_dim),
        dropout=float(dropout),
        encoding=str(encoding),
        use_triads=bool(use_triads),
    )


class I018Bt4EnsembleCompressionNet(nn.Module):
    """i259 ensemble-compression network.

    The deployment forward is a single BT4-shaped conv student. The
    research-mode teachers (frozen i018 + frozen BT4) co-evaluate the
    same input only when ``teacher_mode != 'off'`` and contribute to the
    output dict but never to the student loss; their parameters are not
    optimized. Distillation losses are computed in the trainer using
    the teacher logits surfaced here.
    """

    ALLOWED_TEACHER_MODES = _ALLOWED_TEACHER_MODES
    ALLOWED_FUSION_MODES = _ALLOWED_FUSION_MODES
    ALLOWED_ABLATIONS = _ALLOWED_ABLATIONS

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        student_channels: int = 64,
        student_num_blocks: int = 4,
        student_value_channels: int = 16,
        student_value_hidden: int = 128,
        student_se_channels: int = 16,
        student_dropout: float = 0.1,
        student_use_batchnorm: bool = True,
        teacher_mode: str = "off",
        teacher_alpha: float = 0.5,
        teacher_temperature: float = 1.0,
        fusion_mode: str = "tuned_alpha",
        i018_channels: int = 64,
        i018_hidden_dim: int = 96,
        i018_depth: int = 2,
        i018_stalk_dim: int = 8,
        i018_dropout: float = 0.1,
        i018_use_triads: bool = True,
        i018_encoding: str = "simple_18",
        bt4_channels: int = 64,
        bt4_num_blocks: int = 4,
        bt4_value_channels: int = 16,
        bt4_value_hidden: int = 128,
        bt4_se_channels: int = 16,
        bt4_dropout: float = 0.1,
        bt4_use_batchnorm: bool = True,
        diagnostic_hint_keys: tuple[str, ...] = _DIAGNOSTIC_HINT_KEYS,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "I018Bt4EnsembleCompressionNet supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "I018Bt4EnsembleCompressionNet expects the simple_18 board tensor "
                "(18 channels) so the i018 teacher and BT4 student share the same input"
            )
        teacher_mode_str = str(teacher_mode)
        if teacher_mode_str not in _ALLOWED_TEACHER_MODES:
            raise ValueError(
                f"teacher_mode={teacher_mode!r} must be one of {list(_ALLOWED_TEACHER_MODES)}"
            )
        fusion_mode_str = str(fusion_mode)
        if fusion_mode_str not in _ALLOWED_FUSION_MODES:
            raise ValueError(
                f"fusion_mode={fusion_mode!r} must be one of {list(_ALLOWED_FUSION_MODES)}"
            )
        ablation_str = str(ablation)
        if ablation_str not in _ALLOWED_ABLATIONS:
            raise ValueError(
                f"ablation={ablation!r} must be one of {list(_ALLOWED_ABLATIONS)}"
            )
        if float(teacher_temperature) <= 0.0:
            raise ValueError("teacher_temperature must be > 0")
        alpha = float(teacher_alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("teacher_alpha must lie in [0, 1]")

        self.num_classes = 1
        self.teacher_mode = teacher_mode_str
        self.fusion_mode = fusion_mode_str
        self.ablation = ablation_str
        self._teacher_alpha = alpha
        self._teacher_temperature = float(teacher_temperature)
        self.diagnostic_hint_keys = tuple(str(name) for name in diagnostic_hint_keys)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))

        self.student = _StudentConvTower(
            input_channels=int(input_channels),
            channels=int(student_channels),
            num_blocks=int(student_num_blocks),
            se_channels=int(student_se_channels),
            value_channels=int(student_value_channels),
            value_hidden=int(student_value_hidden),
            dropout=float(student_dropout),
            use_batchnorm=bool(student_use_batchnorm),
            hint_keys=self.diagnostic_hint_keys,
        )

        if self.teacher_mode != "off":
            self.teacher_i018 = _build_i018_teacher(
                input_channels=int(input_channels),
                channels=int(i018_channels),
                hidden_dim=int(i018_hidden_dim),
                depth=int(i018_depth),
                stalk_dim=int(i018_stalk_dim),
                dropout=float(i018_dropout),
                use_triads=bool(i018_use_triads),
                encoding=str(i018_encoding),
            )
            self.teacher_bt4 = _build_bt4_teacher(
                input_channels=int(input_channels),
                channels=int(bt4_channels),
                num_blocks=int(bt4_num_blocks),
                se_channels=int(bt4_se_channels),
                value_channels=int(bt4_value_channels),
                value_hidden=int(bt4_value_hidden),
                dropout=float(bt4_dropout),
                use_batchnorm=bool(bt4_use_batchnorm),
            )
            self.teacher_i018.requires_grad_(False)
            self.teacher_bt4.requires_grad_(False)
        else:
            self.teacher_i018 = None
            self.teacher_bt4 = None

        gate_input_dim = 2 + len(self.diagnostic_hint_keys)
        self.fusion_gate = nn.Linear(gate_input_dim, 1)

    @property
    def teacher_alpha(self) -> float:
        return float(self._teacher_alpha)

    @property
    def teacher_temperature(self) -> float:
        return float(self._teacher_temperature)

    def set_teacher_mode(self, mode: str) -> None:
        if mode not in _ALLOWED_TEACHER_MODES:
            raise ValueError(
                f"teacher_mode={mode!r} must be one of {list(_ALLOWED_TEACHER_MODES)}"
            )
        if mode != "off" and (self.teacher_i018 is None or self.teacher_bt4 is None):
            raise RuntimeError(
                "Cannot enable teacher_mode at runtime when teachers were not built; "
                "rebuild the model with teacher_mode != 'off' first"
            )
        self.teacher_mode = mode

    def _run_teachers(
        self, board: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        assert self.teacher_i018 is not None and self.teacher_bt4 is not None
        with torch.no_grad():
            i018_out = self.teacher_i018(board)
            bt4_out = self.teacher_bt4(board)
        i018_logits = i018_out["logits"].view(-1) if isinstance(i018_out, dict) else i018_out.view(-1)
        bt4_logits = bt4_out.view(-1) if not isinstance(bt4_out, dict) else bt4_out["logits"].view(-1)
        diagnostics: dict[str, torch.Tensor] = {}
        if isinstance(i018_out, dict):
            for key in self.diagnostic_hint_keys:
                if key in i018_out:
                    diagnostics[f"teacher_i018_{key}"] = i018_out[key].detach()
        return i018_logits, bt4_logits, diagnostics

    def _fuse_teacher_logits(
        self,
        i018_logits: torch.Tensor,
        bt4_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temperature = self._teacher_temperature
        i018_calibrated = i018_logits / temperature
        bt4_calibrated = bt4_logits / temperature
        if self.fusion_mode == "equal_weight":
            alpha = torch.full_like(i018_logits, 0.5)
        elif self.fusion_mode == "tuned_alpha":
            alpha = torch.full_like(i018_logits, self._teacher_alpha)
        else:
            disagreement = (torch.sigmoid(i018_calibrated) - torch.sigmoid(bt4_calibrated)).abs()
            mean_prob = 0.5 * (torch.sigmoid(i018_calibrated) + torch.sigmoid(bt4_calibrated))
            entropy = -(
                mean_prob.clamp_min(1e-6) * mean_prob.clamp_min(1e-6).log()
                + (1.0 - mean_prob).clamp_min(1e-6) * (1.0 - mean_prob).clamp_min(1e-6).log()
            )
            gate_input = torch.stack(
                [disagreement, entropy, *([torch.zeros_like(disagreement)] * len(self.diagnostic_hint_keys))],
                dim=-1,
            )
            alpha = torch.sigmoid(self.fusion_gate(gate_input).view(-1))
        fused = alpha * i018_calibrated + (1.0 - alpha) * bt4_calibrated
        return fused, alpha

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        student_logits, hint_outputs = self.student(board)

        if self.ablation == "zero_hint_heads":
            hint_outputs = {name: torch.zeros_like(t) for name, t in hint_outputs.items()}

        out: dict[str, torch.Tensor] = {
            "logits": student_logits,
            "student_logit": student_logits.detach(),
            "student_probability": torch.sigmoid(student_logits.detach()),
        }
        for name, value in hint_outputs.items():
            out[f"diagnostic_hint_{name}"] = value

        if self.teacher_mode == "off" or self.ablation == "student_only":
            zeros = torch.zeros_like(student_logits)
            out["teacher_i018_logit"] = zeros
            out["teacher_bt4_logit"] = zeros
            out["teacher_ensemble_logit"] = zeros
            out["teacher_disagreement"] = zeros
            out["teacher_entropy"] = zeros
            out["teacher_alpha"] = torch.full_like(student_logits, self._teacher_alpha)
            if self.ablation == "teacher_logits_only":
                out["logits"] = zeros
            return out

        i018_logits, bt4_logits, teacher_diagnostics = self._run_teachers(board)
        if self.ablation == "shuffle_teacher_logits" and i018_logits.shape[0] > 1:
            perm = torch.randperm(i018_logits.shape[0], device=i018_logits.device)
            i018_logits = i018_logits[perm]
            bt4_logits = bt4_logits[perm]
        fused, alpha = self._fuse_teacher_logits(i018_logits, bt4_logits)
        with torch.no_grad():
            mean_prob = 0.5 * (torch.sigmoid(i018_logits) + torch.sigmoid(bt4_logits))
            disagreement = (torch.sigmoid(i018_logits) - torch.sigmoid(bt4_logits)).abs()
            entropy = -(
                mean_prob.clamp_min(1e-6) * mean_prob.clamp_min(1e-6).log()
                + (1.0 - mean_prob).clamp_min(1e-6) * (1.0 - mean_prob).clamp_min(1e-6).log()
            )

        out["teacher_i018_logit"] = i018_logits
        out["teacher_bt4_logit"] = bt4_logits
        out["teacher_ensemble_logit"] = fused
        out["teacher_disagreement"] = disagreement
        out["teacher_entropy"] = entropy
        out["teacher_alpha"] = alpha
        for name, value in teacher_diagnostics.items():
            out[name] = value
        if self.ablation == "teacher_logits_only":
            out["logits"] = fused
        return out


def build_i018_bt4_ensemble_compression_from_config(
    config: dict[str, Any],
) -> I018Bt4EnsembleCompressionNet:
    cfg = dict(config)
    hint_keys_raw = cfg.get("diagnostic_hint_keys", _DIAGNOSTIC_HINT_KEYS)
    if isinstance(hint_keys_raw, (list, tuple)):
        hint_keys = tuple(str(name) for name in hint_keys_raw)
    else:
        hint_keys = tuple(str(name) for name in str(hint_keys_raw).split(","))
    return I018Bt4EnsembleCompressionNet(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        student_channels=int(cfg.get("student_channels", cfg.get("channels", 64))),
        student_num_blocks=int(cfg.get("student_num_blocks", cfg.get("num_blocks", 4))),
        student_value_channels=int(cfg.get("student_value_channels", 16)),
        student_value_hidden=int(cfg.get("student_value_hidden", 128)),
        student_se_channels=int(cfg.get("student_se_channels", 16)),
        student_dropout=float(cfg.get("student_dropout", cfg.get("dropout", 0.1))),
        student_use_batchnorm=bool(cfg.get("student_use_batchnorm", cfg.get("use_batchnorm", True))),
        teacher_mode=str(cfg.get("teacher_mode", "off")),
        teacher_alpha=float(cfg.get("teacher_alpha", 0.5)),
        teacher_temperature=float(cfg.get("teacher_temperature", 1.0)),
        fusion_mode=str(cfg.get("fusion_mode", "tuned_alpha")),
        i018_channels=int(cfg.get("i018_channels", 64)),
        i018_hidden_dim=int(cfg.get("i018_hidden_dim", 96)),
        i018_depth=int(cfg.get("i018_depth", 2)),
        i018_stalk_dim=int(cfg.get("i018_stalk_dim", 8)),
        i018_dropout=float(cfg.get("i018_dropout", 0.1)),
        i018_use_triads=bool(cfg.get("i018_use_triads", True)),
        i018_encoding=str(cfg.get("i018_encoding", "simple_18")),
        bt4_channels=int(cfg.get("bt4_channels", 64)),
        bt4_num_blocks=int(cfg.get("bt4_num_blocks", 4)),
        bt4_value_channels=int(cfg.get("bt4_value_channels", 16)),
        bt4_value_hidden=int(cfg.get("bt4_value_hidden", 128)),
        bt4_se_channels=int(cfg.get("bt4_se_channels", 16)),
        bt4_dropout=float(cfg.get("bt4_dropout", 0.1)),
        bt4_use_batchnorm=bool(cfg.get("bt4_use_batchnorm", True)),
        diagnostic_hint_keys=hint_keys,
        ablation=str(cfg.get("ablation", "none")),
    )
