from __future__ import annotations

import torch

from chess_nn_playground.models.registry import build_model
from chess_nn_playground.models.sparse_relation_pursuit import (
    GroupSparsePursuit,
    SparseRelationPursuitClassifier,
    _ordered_relation_edges,
)
from chess_nn_playground.training.losses import SRPALoss


def test_ordered_relation_edges_include_chess_relation_classes():
    edges = _ordered_relation_edges(max_ray_distance=2)

    assert edges["src"].shape == edges["dst"].shape
    assert edges["path_indices"].shape[0] == edges["src"].numel()
    assert set(edges["type"].tolist()) == {0, 1, 2, 3}
    assert (edges["path_mask"].sum(dim=1) > 0).any()


def test_group_sparse_pursuit_returns_residual_trace_and_group_stats():
    pursuit = GroupSparsePursuit(
        relation_dim=12,
        num_atom_groups=3,
        atoms_per_group=2,
        pursuit_steps=2,
    )
    relation_tokens = torch.randn(2, 5, 12)

    output = pursuit(relation_tokens)

    assert output["residual_by_step"].shape == (2, 2, 5)
    assert output["group_energy"].shape == (2, 3)
    assert output["active_atom_fraction"].shape == (2,)
    assert torch.isfinite(output["final_residual"]).all()


def test_srpa_outputs_logits_and_sparse_diagnostics_without_dense_bypass():
    model = SparseRelationPursuitClassifier(
        input_channels=18,
        square_dim=16,
        stem_depth=1,
        relation_dim=12,
        geom_dim=8,
        path_dim=8,
        num_atom_groups=3,
        atoms_per_group=2,
        pursuit_steps=2,
        classifier_hidden=16,
        dropout=0.0,
        max_ray_distance=2,
        edge_chunk_size=64,
    )

    output = model(torch.zeros(2, 18, 8, 8))

    assert output["logits"].shape == (2,)
    assert output["aux_logit"].shape == (2,)
    assert output["bg_group_energy"].shape == (2, 3)
    assert output["tac_group_energy"].shape == (2, 3)
    assert model.sparse_descriptor_dim == 3 * model.pursuit_steps + 3 * model.num_atom_groups + 17
    assert model.head[2].in_features == model.sparse_descriptor_dim
    assert torch.isfinite(output["logits"]).all()
    assert torch.isfinite(output["dictionary_coherence"])
    assert torch.isfinite(output["branch_separation"])


def test_srpa_registry_builder_uses_config():
    model = build_model(
        "sparse_relation_pursuit_asymmetry",
        {
            "input_channels": 18,
            "num_classes": 1,
            "square_dim": 12,
            "stem_depth": 1,
            "relation_dim": 10,
            "geom_dim": 6,
            "path_dim": 6,
            "num_atom_groups": 2,
            "atoms_per_group": 2,
            "pursuit_steps": 2,
            "classifier_hidden": 12,
            "dropout": 0.0,
            "max_ray_distance": 2,
            "edge_chunk_size": 128,
        },
    )

    output = model(torch.zeros(1, 18, 8, 8))

    assert output["logits"].shape == (1,)
    assert model.num_atom_groups == 2


def test_srpa_loss_backpropagates_through_logits_and_sparse_terms():
    output = {
        "logits": torch.tensor([0.5, -0.25, 1.0], requires_grad=True),
        "aux_logit": torch.tensor([0.4, -0.2, 0.8], requires_grad=True),
        "bg_final_residual": torch.tensor([0.4, 0.2, 0.5], requires_grad=True),
        "tac_final_residual": torch.tensor([0.2, 0.5, 0.1], requires_grad=True),
        "mean_abs_code": torch.tensor([0.01, 0.02, 0.03], requires_grad=True),
        "mean_group_norm": torch.tensor([0.04, 0.03, 0.02], requires_grad=True),
        "dictionary_coherence": torch.tensor(0.1, requires_grad=True),
        "branch_separation": torch.tensor(0.2, requires_grad=True),
        "dead_group_penalty": torch.tensor(0.05, requires_grad=True),
    }
    target = torch.tensor([1, 0, 1])
    loss_fn = SRPALoss(pos_weight=torch.tensor([2.0]))

    loss = loss_fn(output, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert output["logits"].grad is not None
    assert output["aux_logit"].grad is not None
    assert output["bg_final_residual"].grad is not None
    assert output["tac_final_residual"].grad is not None
