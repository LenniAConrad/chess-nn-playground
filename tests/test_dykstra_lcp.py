from __future__ import annotations

import torch

from chess_nn_playground.models.dykstra_lcp import DykstraLCP, SoftDykstraProjector, _simplex_projection
from chess_nn_playground.models.dykstra_vetoselect import DykstraVetoSelect
from chess_nn_playground.training.losses import DykstraLCPLoss, DykstraVetoSelectLoss


def test_simplex_projection_preserves_probability_mass():
    values = torch.tensor([[3.0, -1.0, 0.5], [0.2, 0.2, 0.2]])

    projected = _simplex_projection(values)

    assert torch.all(projected >= 0.0)
    assert torch.allclose(projected.sum(dim=1), torch.ones(2), atol=1e-6)


def test_dykstra_role_budgets_depend_on_motif_mixture():
    projector = SoftDykstraProjector(role_count=4, relation_channels=1, motif_count=2, slack_count=1)
    with torch.no_grad():
        projector.role_budget_delta[0, 0] = 6.0
        projector.role_budget_delta[1, 0] = -6.0
    motif_a = torch.tensor([[1.0, 0.0]])
    motif_b = torch.tensor([[0.0, 1.0]])

    budget_a = projector._role_budget_upper(motif_a, torch.float32)
    budget_b = projector._role_budget_upper(motif_b, torch.float32)

    assert budget_a[0, 0] > budget_b[0, 0]


def test_dykstra_closure_uses_slack_for_unexplained_target_mass():
    projector = SoftDykstraProjector(role_count=4, relation_channels=1, motif_count=2, slack_count=2)
    u = torch.zeros(1, 4, 64)
    u[:, 1] = 1.0
    v = torch.zeros(1, 1, 64, 64)
    m = torch.tensor([[0.5, 0.5]])
    s = torch.zeros(1, 2)
    role_masks = torch.ones_like(u)

    u_next, _v_next, _m_next, s_next = projector._project_closure(u, v, m, s, role_masks)

    assert s_next[0, 0] > 0.0
    assert u_next[:, 1].mean() < u[:, 1].mean()


def test_dykstra_lcp_outputs_logit_and_solver_diagnostics():
    model = DykstraLCP(
        input_channels=18,
        channels=8,
        num_blocks=1,
        value_channels=4,
        value_hidden=16,
        se_channels=4,
        role_count=4,
        relation_channels=2,
        motif_count=4,
        slack_count=3,
        solver_cycles=2,
    )

    output = model(torch.zeros(2, 18, 8, 8))

    assert output["logits"].shape == (2,)
    assert output["projection_distance"].shape == (2,)
    assert output["trace_residual"].shape == (2,)
    assert output["slack_mean"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()


def test_dykstra_lcp_loss_backpropagates():
    output = {
        "logits": torch.tensor([1.0, -0.5, 0.25], requires_grad=True),
        "projection_distance": torch.tensor([0.1, 0.4, 0.2], requires_grad=True),
        "trace_residual": torch.tensor([0.2, 0.1, 0.3], requires_grad=True),
        "decay_violation": torch.tensor([0.0, 0.1, 0.0], requires_grad=True),
    }
    target = torch.tensor([1, 0, 0])
    loss_fn = DykstraLCPLoss(pos_weight=torch.tensor([2.0]), hard_negative_fraction=0.5)

    loss = loss_fn(output, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert output["logits"].grad is not None
    assert output["projection_distance"].grad is not None


def test_dykstra_lcp_role_masks_follow_simple18_side_to_move():
    projector = SoftDykstraProjector(role_count=4, relation_channels=1, motif_count=2, slack_count=1)
    x = torch.zeros(1, 18, 8, 8)
    x[:, 12] = 0.0
    x[0, 5, 1, 1] = 1.0
    x[0, 9, 2, 2] = 1.0
    x[0, 11, 3, 3] = 1.0

    masks = projector._role_masks(x)

    assert masks[0, 0, 1 * 8 + 1] == 1.0
    assert masks[0, 1, 1 * 8 + 1] == 1.0
    assert masks[0, 1, 2 * 8 + 2] == 0.0
    assert masks[0, 3, 2 * 8 + 2] == 1.0


def test_dykstra_vetoselect_outputs_selector_and_solver_diagnostics():
    model = DykstraVetoSelect(
        input_channels=18,
        channels=8,
        num_blocks=1,
        value_channels=4,
        value_hidden=16,
        se_channels=4,
        role_count=4,
        relation_channels=2,
        motif_count=4,
        slack_count=3,
        solver_cycles=2,
    )

    output = model(torch.zeros(2, 18, 8, 8))

    assert output["selective_puzzle_logit"].shape == (2,)
    assert output["puzzle_logit"].shape == (2,)
    assert output["selector_logit"].shape == (2,)
    assert output["projection_distance"].shape == (2,)
    assert torch.isfinite(output["selective_puzzle_logit"]).all()


def test_dykstra_vetoselect_loss_backpropagates():
    output = {
        "puzzle_logit": torch.tensor([1.0, -0.5, 0.25], requires_grad=True),
        "selector_logit": torch.tensor([0.5, -0.25, 0.1], requires_grad=True),
        "projection_distance": torch.tensor([0.1, 0.4, 0.2], requires_grad=True),
        "trace_residual": torch.tensor([0.2, 0.1, 0.3], requires_grad=True),
        "decay_violation": torch.tensor([0.0, 0.1, 0.0], requires_grad=True),
    }
    target = torch.tensor([1, 0, 0])
    loss_fn = DykstraVetoSelectLoss(pos_weight=torch.tensor([2.0]))

    loss = loss_fn(output, target, texture=torch.ones(3))
    loss.backward()

    assert torch.isfinite(loss)
    assert output["puzzle_logit"].grad is not None
    assert output["selector_logit"].grad is not None
    assert output["projection_distance"].grad is not None
