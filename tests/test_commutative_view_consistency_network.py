"""Focused tests for the bespoke i137 Commutative View-Consistency Network."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_tensor
from chess_nn_playground.ideas.architecture_conformance import audit_architecture_conformance
from chess_nn_playground.ideas.implementation import validate_idea_for_training
from chess_nn_playground.ideas.implementation_kind import detect_idea_implementation_kind
from chess_nn_playground.models.registry import available_models, build_model
from chess_nn_playground.models.research_packet_registry import RESEARCH_PACKET_MODEL_NAMES
from chess_nn_playground.models.research_packet_probe import ResearchPacketProbe
from chess_nn_playground.models.trunk.commutative_view_consistency import (
    CYCLE_DEFECTS,
    DEFECT_MAP_EDGES,
    DIRECT_DEFECTS,
    VIEW_NAMES,
    CommutativeViewConsistencyNetwork,
    _LowRankMap,
    _PieceDeepSets,
    _SquareEncoder,
    build_commutative_view_consistency_network_from_config,
)


REGISTRY_KEY = "commutative_view_consistency_network"
IDEA_DIR = Path("ideas/registry/i137_commutative_view_consistency_network")
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MATE_PUZZLE_FEN = "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"


def _toy_kwargs(**overrides) -> dict[str, object]:
    base: dict[str, object] = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 8,
        "hidden_dim": 16,
        "latent_dim": 8,
        "map_rank": 4,
        "depth": 2,
        "dropout": 0.0,
        "use_batchnorm": False,
        "random_map_seed": 7,
    }
    base.update(overrides)
    return base


def _board_batch(fens: list[str]) -> torch.Tensor:
    arrays = [fen_to_tensor(fen) for fen in fens]
    return torch.stack([torch.from_numpy(a).float() for a in arrays], dim=0)


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()


def test_registry_key_is_not_a_research_packet_probe_name() -> None:
    assert REGISTRY_KEY not in RESEARCH_PACKET_MODEL_NAMES


def test_builder_from_config_returns_bespoke_model() -> None:
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, CommutativeViewConsistencyNetwork)
    assert not isinstance(model, ResearchPacketProbe)


def test_builder_passes_through_relevant_config_keys() -> None:
    model = build_commutative_view_consistency_network_from_config(
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 12,
            "hidden_dim": 24,
            "latent_dim": 16,
            "map_rank": 6,
            "depth": 1,
            "dropout": 0.0,
            "use_batchnorm": False,
            "ablation": "none",
        }
    )
    assert model.channels == 12
    assert model.hidden_dim == 24
    assert model.latent_dim == 16
    assert model.map_rank == 6
    assert model.depth == 1
    expected_head_dim = len(VIEW_NAMES) * 16 + model.num_defects * 5
    assert model.head_input_dim == expected_head_dim
    assert model.classifier[1].in_features == expected_head_dim
    assert model.classifier[1].out_features == 24


def test_forward_shape_and_required_keys() -> None:
    model = CommutativeViewConsistencyNetwork(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["logits"].shape == (2,)
    assert torch.isfinite(out["logits"]).all()
    expected_keys = {
        "logits",
        "prob",
        "view_pooled",
        "view_norms",
        "defect_stats",
        "defect_l2",
        "defect_l1",
        "defect_cosine",
        "consistency_energy",
        "mean_defect_l1",
        "mean_defect_cosine",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
        "commutative_view_ablation",
        "commutative_view_count",
    }
    assert expected_keys.issubset(out)
    assert out["view_pooled"].shape == (2, model.num_views, model.latent_dim)
    assert out["view_norms"].shape == (2, model.num_views)
    assert out["defect_stats"].shape == (2, model.num_defects, 5)
    for key in ("defect_l2", "defect_l1", "defect_cosine"):
        assert out[key].shape == (2, model.num_defects), key
    for key in (
        "consistency_energy",
        "mean_defect_l1",
        "mean_defect_cosine",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
        "commutative_view_ablation",
        "commutative_view_count",
    ):
        assert out[key].shape == (2,), key
    for view in VIEW_NAMES:
        assert out[f"z_{view}"].shape == (2, model.latent_dim), view


def test_single_logit_prob_is_in_unit_interval() -> None:
    model = CommutativeViewConsistencyNetwork(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["prob"].shape == (2,)
    assert (out["prob"] >= 0).all() and (out["prob"] <= 1).all()


def test_backward_gradients_flow_through_encoders_maps_and_head() -> None:
    model = CommutativeViewConsistencyNetwork(**_toy_kwargs())
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()

    head_first = model.classifier[1].weight.grad
    assert head_first is not None and torch.isfinite(head_first).all()

    line_first = model.line_encoder.body[1].weight.grad
    assert line_first is not None and torch.isfinite(line_first).all()
    count_first = model.count_encoder.body[1].weight.grad
    assert count_first is not None and torch.isfinite(count_first).all()

    map_param = model.maps["square_to_line"].left.weight.grad
    assert map_param is not None and torch.isfinite(map_param).all()


def test_defect_map_edges_cover_all_used_paths() -> None:
    edges = set(DEFECT_MAP_EDGES)
    for target, source in DIRECT_DEFECTS:
        assert (source, target) in edges, (source, target)
    for loop, mid in CYCLE_DEFECTS:
        assert (loop, mid) in edges, (loop, mid)
        assert (mid, loop) in edges, (mid, loop)


def test_views_only_no_defects_zeros_defect_features_in_head() -> None:
    model = CommutativeViewConsistencyNetwork(**_toy_kwargs(), ablation="views_only_no_defects").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out_ablated = model(boards)
        view_features = out_ablated["view_pooled"].flatten(1)
        head_input_ablated = torch.cat(
            [view_features, torch.zeros(view_features.shape[0], model.head_defect_dim)], dim=-1
        )
        recomputed = model.classifier(head_input_ablated).squeeze(-1)
    assert torch.allclose(recomputed, out_ablated["logits"])


def test_single_square_view_zeros_other_view_latents() -> None:
    model = CommutativeViewConsistencyNetwork(**_toy_kwargs(), ablation="single_square_view").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    # square latent should be non-zero on a populated board; others must be zero.
    assert out["z_square"].abs().sum() > 0
    for view in VIEW_NAMES:
        if view == "square":
            continue
        assert torch.allclose(out[f"z_{view}"], torch.zeros_like(out[f"z_{view}"])), view


def test_count_to_all_only_zeros_non_count_view_latents() -> None:
    model = CommutativeViewConsistencyNetwork(**_toy_kwargs(), ablation="count_to_all_only").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["z_count"].abs().sum() > 0
    for view in VIEW_NAMES:
        if view == "count":
            continue
        assert torch.allclose(out[f"z_{view}"], torch.zeros_like(out[f"z_{view}"])), view


def test_random_view_maps_freezes_cross_view_maps() -> None:
    model = CommutativeViewConsistencyNetwork(**_toy_kwargs(), ablation="random_view_maps")
    for module in model.maps.values():
        for param in module.parameters():
            assert not param.requires_grad


def test_random_view_maps_is_deterministic_for_fixed_seed() -> None:
    kwargs = _toy_kwargs(random_map_seed=11)
    model_a = CommutativeViewConsistencyNetwork(**kwargs, ablation="random_view_maps")
    model_b = CommutativeViewConsistencyNetwork(**kwargs, ablation="random_view_maps")
    for key in model_a.maps:
        assert torch.allclose(model_a.maps[key].left.weight, model_b.maps[key].left.weight), key
        assert torch.allclose(model_a.maps[key].right.weight, model_b.maps[key].right.weight), key


def test_shuffled_piece_view_permutes_piece_latent_within_batch() -> None:
    encoder = _PieceDeepSets(hidden_dim=16, latent_dim=8, dropout=0.0).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        plain = encoder(boards, shuffle=False)
        # Repeat several seeds so we are guaranteed at least one non-identity permutation.
        any_permuted = False
        for seed in range(1, 25):
            torch.manual_seed(seed)
            shuffled = encoder(boards, shuffle=True)
            # Sorted token set is preserved by an intra-batch swap, so the union
            # of latents must match even when the per-sample assignment is swapped.
            assert torch.allclose(
                torch.sort(plain.flatten(), dim=0).values,
                torch.sort(shuffled.flatten(), dim=0).values,
            )
            if not torch.allclose(plain, shuffled):
                any_permuted = True
                break
        assert any_permuted, "intra-batch shuffle never produced a non-identity permutation"


def test_all_ablations_run_without_crash_and_emit_their_code() -> None:
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    for ablation in CommutativeViewConsistencyNetwork.ABLATIONS:
        torch.manual_seed(0)
        model = CommutativeViewConsistencyNetwork(**_toy_kwargs(), ablation=ablation).eval()
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation
        assert out["commutative_view_ablation"][0].item() == float(
            CommutativeViewConsistencyNetwork.ABLATIONS.index(ablation)
        ), ablation
        assert out["commutative_view_count"][0].item() == float(model.num_views), ablation


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        CommutativeViewConsistencyNetwork(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        CommutativeViewConsistencyNetwork(input_channels=12, num_classes=1)


def test_rejects_multiclass_head_contract() -> None:
    with pytest.raises(ValueError):
        CommutativeViewConsistencyNetwork(input_channels=18, num_classes=2)


def test_square_encoder_outputs_latent_width() -> None:
    encoder = _SquareEncoder(
        input_channels=18, channels=8, depth=2, latent_dim=8, dropout=0.0, use_batchnorm=False
    )
    out = encoder(torch.zeros(3, 18, 8, 8))
    assert out.shape == (3, 8)


def test_piece_deepsets_is_permutation_invariant_for_a_single_sample() -> None:
    encoder = _PieceDeepSets(hidden_dim=16, latent_dim=8, dropout=0.0).eval()
    boards = _board_batch([STARTING_FEN])
    with torch.no_grad():
        plain = encoder(boards, shuffle=False)
        # Shuffle is intra-batch; for batch size 1 there is no permutation, so
        # shuffle should produce the same latent. We use this as the basic
        # invariance check.
        shuffled = encoder(boards, shuffle=True)
    assert torch.allclose(plain, shuffled)


def test_low_rank_map_has_factorised_rank_and_correct_shape() -> None:
    layer = _LowRankMap(latent_dim=8, rank=3)
    out = layer(torch.zeros(2, 8))
    assert out.shape == (2, 8)
    assert layer.left.weight.shape == (3, 8)
    assert layer.right.weight.shape == (8, 3)


def test_coordinate_buffers_are_not_trainable_parameters() -> None:
    model = CommutativeViewConsistencyNetwork(**_toy_kwargs())
    parameter_ids = {id(p) for p in model.parameters()}
    assert id(model.piece_encoder.coords) not in parameter_ids


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "i137"
    assert data["slug"] == "commutative_view_consistency_network"
    assert data["implementation_kind"] == "bespoke_model"
    assert data["implementation_status"] == "implemented"
    assert data["status"] == "implemented"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "i137"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"


def _load_idea_model_module(folder: Path):
    module_path = folder / "model.py"
    spec = importlib.util.spec_from_file_location(f"idea_{folder.name}_model", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_idea_folder_is_bespoke_and_conformant() -> None:
    config = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model_module(IDEA_DIR)
    model = module.build_model_from_config(config).eval()
    assert isinstance(model, CommutativeViewConsistencyNetwork)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.square_encoder, _SquareEncoder)
    assert isinstance(model.piece_encoder, _PieceDeepSets)

    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["logits"].shape == (2,)
    assert torch.isfinite(out["logits"]).all()

    model_py = (IDEA_DIR / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(IDEA_DIR)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues, kind_row.issues

    training_report = validate_idea_for_training(IDEA_DIR)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in audit_architecture_conformance() if row.idea_id == "i137"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues, conformance_rows[0].issues
