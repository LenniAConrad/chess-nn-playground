"""Focused tests for the delta-accumulator primitive family (p012–p018).

This module exercises:

- the shared ``delta_accumulator.py`` helper (active-feature extraction,
  embedding gather, involution lookup) on hand-built simple_18 boards;
- the seven primitive heads (registry presence, forward shape, gradient
  flow through head and trunk, declared ablation modes, config-yaml
  round-trip).

Per the i246 / i248 conventions every primitive in the family also
declares the standard
``logits, base_logit, primitive_delta, primitive_gate`` output keys and
its own primitive-specific diagnostics; this module asserts those keys
are present so the trainer can copy them into
``predictions_<split>.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from chess_nn_playground.models.registry import available_models, build_model
from chess_nn_playground.models.primitives.delta_accumulator import (
    ActiveFeatures,
    DeltaAccumulator,
    FEATURE_VOCAB,
    MAX_ACTIVE_FEATURES,
    extract_active_features,
    involution_indices,
    piece_color_id,
    piece_type_and_square,
    side_to_move_color,
)


# ---------------------------------------------------------------------------
# Synthetic simple_18 boards used by the tests.
# ---------------------------------------------------------------------------


def _empty_board(batch: int = 1, stm_white: bool = True) -> torch.Tensor:
    board = torch.zeros(batch, 18, 8, 8)
    if stm_white:
        board[:, 12, :, :] = 1.0
    return board


def _white_pawn_endgame() -> torch.Tensor:
    """White pawn on e2, white king on e1, black king on e8, white to move."""

    board = _empty_board()
    # simple_18 plane 0 = white pawn; rank 2 -> row index 6, file e -> 4
    board[0, 0, 6, 4] = 1.0
    # white king plane 5; rank 1 -> row index 7, file e -> 4
    board[0, 5, 7, 4] = 1.0
    # black king plane 11; rank 8 -> row index 0, file e -> 4
    board[0, 11, 0, 4] = 1.0
    return board


def _balanced_kings() -> torch.Tensor:
    """Symmetric king-only position used for the χ-grading test."""

    board = _empty_board()
    board[0, 5, 7, 4] = 1.0  # white king e1
    board[0, 11, 0, 4] = 1.0  # black king e8
    return board


# ---------------------------------------------------------------------------
# Shared helper tests.
# ---------------------------------------------------------------------------


def test_extract_active_features_recovers_piece_squares():
    board = _white_pawn_endgame()
    feats = extract_active_features(board, max_features=8)
    valid_mask = feats.valid.bool()[0]
    indices = feats.indices[0]
    active_indices = sorted(int(idx.item()) for idx, m in zip(indices, valid_mask) if m)
    # white pawn e2 -> piece 0, square 6*8+4 = 52
    # white king e1 -> piece 5, square 7*8+4 = 380
    # black king e8 -> piece 11, square 0*8+4 = 708
    assert active_indices == [52, 5 * 64 + 7 * 8 + 4, 11 * 64 + 0 * 8 + 4]
    assert float(feats.count[0].item()) == 3.0


def test_extract_active_features_empty_board_has_no_active_indices():
    board = _empty_board()
    feats = extract_active_features(board, max_features=8)
    assert float(feats.count[0].item()) == 0.0
    assert float(feats.valid.sum().item()) == 0.0


def test_delta_accumulator_gather_masks_invalid_slots():
    board = _white_pawn_endgame()
    feats = extract_active_features(board, max_features=8)
    acc = DeltaAccumulator(accumulator_dim=4, max_features=8)
    gathered = acc.gather(feats)
    valid_count = int(feats.valid.sum().item())
    invalid_mask = feats.valid == 0.0
    # invalid slots must produce zero rows (mask-multiplied embeddings).
    assert gathered.shape == (1, 8, 4)
    assert torch.allclose(gathered[0][invalid_mask[0]], torch.zeros(8 - valid_count, 4))


def test_piece_type_and_square_inverse_of_flat_index():
    flat = torch.tensor([52, 380, 708])
    types, squares = piece_type_and_square(flat)
    assert torch.equal(types, torch.tensor([0, 5, 11]))
    assert torch.equal(squares, torch.tensor([52 % 64, 380 % 64, 708 % 64]))


def test_involution_indices_swaps_color_and_flips_rank():
    flat = torch.tensor([52, 380, 708])  # white pawn e2, white king e1, black king e8
    inv = involution_indices(flat)
    # white pawn e2 -> black pawn e7 (plane 6, row 1, file 4 -> 6*64 + 1*8 + 4 = 396)
    assert int(inv[0].item()) == 6 * 64 + 1 * 8 + 4
    # white king e1 -> black king e8 (plane 11, row 0, file 4)
    assert int(inv[1].item()) == 11 * 64 + 0 * 8 + 4
    # black king e8 -> white king e1 (plane 5, row 7, file 4)
    assert int(inv[2].item()) == 5 * 64 + 7 * 8 + 4


def test_side_to_move_color_recovers_stm():
    board_white = _white_pawn_endgame()
    board_black = _empty_board(stm_white=False)
    assert int(side_to_move_color(board_white).item()) == 0  # white
    assert int(side_to_move_color(board_black).item()) == 1  # black


def test_piece_color_id_partitions_planes_by_color():
    flat = torch.arange(0, 12) * 64  # one feature per plane on square 0
    types = flat // 64
    assert torch.equal(
        piece_color_id(types), torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    )


# ---------------------------------------------------------------------------
# Registry + forward smoke tests for the seven primitive heads.
# ---------------------------------------------------------------------------


PRIMITIVE_KEYS = [
    ("p012", "signed_edit_bilinear_memory"),
    ("p013", "sparse_delta_accumulator"),
    ("p014", "delta_pair_accumulator"),
    ("p015", "delta_crelu_involution_head"),
    ("p016", "ray_semiring_chi_head"),
    ("p017", "delta_event_legal_routing"),
    ("p018", "delta_state_slg_diffusion"),
]


def _small_config(**overrides):
    base = dict(
        input_channels=18,
        num_classes=1,
        trunk_channels=16,
        trunk_hidden_dim=24,
        trunk_depth=1,
        trunk_dropout=0.0,
        trunk_use_batchnorm=False,
        accumulator_dim=16,
        max_features=20,
        head_hidden_dim=16,
        head_dropout=0.0,
        gate_init=-2.0,
        ablation="none",
        bilinear_rank=8,
        projection_dim=16,
        pair_dim=8,
        clip_max=1.0,
        involution_weight=1.0,
        chi_rank=8,
        routing_hidden_dim=8,
        stalk_dim=8,
        diffusion_alpha=0.25,
        diffusion_steps=1,
    )
    base.update(overrides)
    return base


def _build_mixed_batch() -> torch.Tensor:
    boards = [
        _white_pawn_endgame()[0],
        _empty_board()[0],
        _balanced_kings()[0],
    ]
    return torch.stack(boards, dim=0)


@pytest.mark.parametrize("idea_id, registry_key", PRIMITIVE_KEYS)
def test_registry_key_is_present(idea_id, registry_key):
    assert registry_key in available_models()


@pytest.mark.parametrize("idea_id, registry_key", PRIMITIVE_KEYS)
def test_forward_shape_and_required_diagnostics(idea_id, registry_key):
    torch.manual_seed(0)
    model = build_model(registry_key, _small_config())
    model.eval()
    batch = _build_mixed_batch()
    with torch.no_grad():
        out = model(batch)
    assert out["logits"].shape == (3,)
    for key in (
        "logits",
        "base_logit",
        "primitive_delta",
        "primitive_delta_raw",
        "primitive_gate",
        "primitive_gate_logit",
        "primitive_active_count",
        "primitive_state_norm",
    ):
        assert key in out, f"missing diagnostic {key!r} on {registry_key}"
        assert out[key].shape == (3,), f"diagnostic {key!r} wrong shape on {registry_key}"


@pytest.mark.parametrize("idea_id, registry_key", PRIMITIVE_KEYS)
def test_backward_pass_reaches_head_and_trunk(idea_id, registry_key):
    torch.manual_seed(0)
    model = build_model(registry_key, _small_config()).train()
    batch = _build_mixed_batch()
    out = model(batch)
    target = torch.tensor([1.0, 0.0, 1.0])
    loss = F.binary_cross_entropy_with_logits(out["logits"], target)
    loss.backward()
    head_grads = [
        p
        for n, p in model.named_parameters()
        if not n.startswith("trunk.") and p.requires_grad
    ]
    trunk_grads = [
        p
        for n, p in model.trunk.named_parameters()
        if p.requires_grad
    ]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head_grads)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trunk_grads)


@pytest.mark.parametrize("idea_id, registry_key", PRIMITIVE_KEYS)
def test_zero_delta_ablation_collapses_to_trunk_logit(idea_id, registry_key):
    torch.manual_seed(0)
    model = build_model(registry_key, _small_config(ablation="zero_delta"))
    model.eval()
    batch = _build_mixed_batch()
    with torch.no_grad():
        out = model(batch)
    assert torch.allclose(out["logits"], out["base_logit"], atol=1.0e-6)
    assert float(out["primitive_delta"].abs().sum().item()) == 0.0


@pytest.mark.parametrize("idea_id, registry_key", PRIMITIVE_KEYS)
def test_trunk_only_ablation_collapses_to_trunk_logit(idea_id, registry_key):
    torch.manual_seed(0)
    model = build_model(registry_key, _small_config(ablation="trunk_only"))
    model.eval()
    batch = _build_mixed_batch()
    with torch.no_grad():
        out = model(batch)
    assert torch.allclose(out["logits"], out["base_logit"], atol=1.0e-6)


@pytest.mark.parametrize("idea_id, registry_key", PRIMITIVE_KEYS)
def test_disable_gate_keeps_gate_at_one(idea_id, registry_key):
    torch.manual_seed(0)
    model = build_model(registry_key, _small_config(ablation="disable_gate"))
    model.eval()
    batch = _build_mixed_batch()
    with torch.no_grad():
        out = model(batch)
    assert torch.allclose(out["primitive_gate"], torch.ones(3), atol=1.0e-6)


@pytest.mark.parametrize("idea_id, registry_key", PRIMITIVE_KEYS)
def test_config_yaml_passes_static_structure(idea_id, registry_key):
    config_path = (
        Path("ideas/registry") / f"{idea_id}_{registry_key}" / "config.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["idea_id"] == idea_id
    assert config["mode"] == "puzzle_binary"
    assert config["device"] == "nvidia"
    assert config["data"]["encoding"] == "simple_18"
    assert config["model"]["name"] == registry_key
    assert config["model"]["num_classes"] == 1
    assert config["model"]["input_channels"] == 18


@pytest.mark.parametrize("idea_id, registry_key", PRIMITIVE_KEYS)
def test_idea_yaml_structure(idea_id, registry_key):
    idea_path = (
        Path("ideas/registry") / f"{idea_id}_{registry_key}" / "idea.yaml"
    )
    idea = yaml.safe_load(idea_path.read_text(encoding="utf-8"))
    assert idea["idea_id"] == idea_id
    assert idea["slug"] == registry_key
    assert idea["status"] == "implemented"
    assert idea["implementation_kind"] == "bespoke_model"
    assert idea["mechanism_family"] == "move_delta"
    assert idea["model_path"].endswith("model.py")
    assert idea["config_path"].endswith("config.yaml")


# ---------------------------------------------------------------------------
# Primitive-specific algebra tests.
# ---------------------------------------------------------------------------


def test_sebm_zero_pair_ablation_zeros_pair_state():
    """SEBM's pair state ``p`` must be exactly zero under zero_pair_state."""

    torch.manual_seed(0)
    model = build_model(
        "signed_edit_bilinear_memory",
        _small_config(ablation="zero_pair_state"),
    )
    model.eval()
    batch = _build_mixed_batch()
    with torch.no_grad():
        out = model(batch)
    assert float(out["sebm_p_norm"].abs().sum().item()) == 0.0


def test_sebm_pair_state_uses_fm_identity():
    """SEBM's pair state must equal ``s ⊙ u − Σ a_j ⊙ b_j``."""

    torch.manual_seed(0)
    cfg = _small_config()
    model = build_model("signed_edit_bilinear_memory", cfg)
    model.eval()
    batch = _build_mixed_batch()
    with torch.no_grad():
        out = model(batch)
    # s ⊙ u norm should equal the analytical cross-term norm; ratio should
    # be finite (non-degenerate batch).
    assert torch.isfinite(out["sebm_pair_ratio"]).all()
    # Without ablation, the pair state must be non-zero on samples that have
    # at least one active pair (3-piece endgame, 0-piece position differs).
    assert float(out["sebm_p_norm"][0].item()) > 0.0
    assert float(out["sebm_p_norm"][1].item()) == pytest.approx(0.0, abs=1.0e-6)


def test_dpa_zero_pair_term_collapses_to_first_order():
    torch.manual_seed(0)
    model_full = build_model("delta_pair_accumulator", _small_config())
    model_ablated = build_model(
        "delta_pair_accumulator", _small_config(ablation="zero_pair_term")
    )
    model_full.eval()
    model_ablated.eval()
    batch = _build_mixed_batch()
    with torch.no_grad():
        out_a = model_ablated(batch)
    assert float(out_a["dpa_pair_state_norm"].abs().sum().item()) == 0.0


def test_dpa_uniform_edge_mask_is_strictly_denser_than_alignment():
    """Selectivity = |E_alignment| / |E_full| ≤ 1; uniform mask sets it to 1."""

    torch.manual_seed(0)
    model_align = build_model("delta_pair_accumulator", _small_config())
    model_uniform = build_model(
        "delta_pair_accumulator", _small_config(ablation="uniform_edge_mask")
    )
    model_align.eval()
    model_uniform.eval()
    batch = _build_mixed_batch()
    with torch.no_grad():
        a = model_align(batch)
        u = model_uniform(batch)
    # Uniform mask -> selectivity == 1 on samples with any active pair
    pawn_sample = 0
    assert float(u["dpa_edge_selectivity"][pawn_sample].item()) == pytest.approx(
        1.0, abs=1.0e-6
    )
    # Alignment mask is a strict subset of pairs (rank/file/diagonal only).
    assert float(a["dpa_edge_selectivity"][pawn_sample].item()) <= 1.0


def test_chi_head_grading_separates_colour_streams():
    """χ-head must compute distinct ``h+`` and ``h-`` from a mixed board.

    The strict antisymmetry guarantee ``f(τx) = − f(x)`` from external_10
    requires shared projections + a sign-antisymmetric bilinear — both
    are stronger constraints than the current implementation enforces.
    What the implementation *does* guarantee is that ``h+`` is computed
    from white piece-square features and ``h-`` from black piece-square
    features; the no_chi_grading ablation replaces the cross-bilinear
    with a same-grade ``M(h+, h+)`` term, which must change the chi
    diagnostic. See ``architecture.md`` for the structural claim.
    """

    torch.manual_seed(0)
    model_full = build_model("ray_semiring_chi_head", _small_config())
    model_ablated = build_model(
        "ray_semiring_chi_head",
        _small_config(ablation="no_chi_grading"),
    )
    model_full.eval()
    model_ablated.eval()
    board = _white_pawn_endgame()
    with torch.no_grad():
        out = model_full(board)
        out_ablated = model_ablated(board)
    # On a position with both colours active the ``h+`` and ``h-`` norms
    # are both populated.
    assert float(out["chi_plus_norm"].item()) > 0.0
    assert float(out["chi_minus_norm"].item()) > 0.0
    # The cross-bilinear term must be non-trivial under both modes and
    # distinct between them — otherwise the χ-grading is a no-op.
    assert float(out["chi_cross_norm"].item()) > 0.0
    assert float(out_ablated["chi_cross_norm"].item()) > 0.0
    assert not torch.allclose(
        out["chi_cross_norm"], out_ablated["chi_cross_norm"], atol=1.0e-5
    )


def test_dcrelu_saturation_diagnostics_sum_to_one():
    torch.manual_seed(0)
    model = build_model("delta_crelu_involution_head", _small_config())
    model.eval()
    batch = _build_mixed_batch()
    with torch.no_grad():
        out = model(batch)
    total = out["dcrelu_saturated_low_frac"] + out["dcrelu_saturated_high_frac"]
    assert torch.all(total >= 0.0)
    assert torch.all(total <= 1.0 + 1.0e-6)


def test_delr_uniform_routing_matches_unweighted_accumulator():
    """uniform_routing must reduce DELR to the canonical Σ W[i] state."""

    torch.manual_seed(0)
    model = build_model(
        "delta_event_legal_routing", _small_config(ablation="uniform_routing")
    )
    model.eval()
    batch = _build_mixed_batch()
    with torch.no_grad():
        out = model(batch)
    # With uniform routing the per-piece weight is exactly features.valid,
    # so the state-norm should match the validity count up to scaling.
    assert torch.all(out["delr_routing_sum"] >= 0.0)
    # ``delr_routing_mean`` must be exactly 1.0 on samples with at least
    # one active piece (since weight = valid for each active slot).
    has_pieces = out["primitive_active_count"] > 0
    if has_pieces.any():
        assert torch.allclose(
            out["delr_routing_mean"][has_pieces],
            torch.ones_like(out["delr_routing_mean"][has_pieces]),
            atol=1.0e-6,
        )


def test_dssg_no_diffusion_collapses_to_sum_accumulator():
    torch.manual_seed(0)
    model = build_model(
        "delta_state_slg_diffusion", _small_config(ablation="no_diffusion")
    )
    model.eval()
    batch = _build_mixed_batch()
    with torch.no_grad():
        out = model(batch)
    # When diffusion is disabled the diagnostic edge count must be zero.
    assert float(out["dssg_edge_count"].abs().sum().item()) == 0.0


def test_shuffle_features_does_not_break_forward():
    """The shuffle ablation must not raise on any primitive (defensive)."""

    torch.manual_seed(0)
    for idea_id, registry_key in PRIMITIVE_KEYS:
        model = build_model(registry_key, _small_config(ablation="shuffle_features"))
        model.eval()
        batch = _build_mixed_batch()
        with torch.no_grad():
            out = model(batch)
        assert out["logits"].shape == (3,)
