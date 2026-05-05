from __future__ import annotations

import torch

from chess_nn_playground.data.tactical_texture import tactical_texture_score
from chess_nn_playground.models.vetoselect import VetoSelectPuzzleNet
from chess_nn_playground.training.losses import VetoSelectLoss


def test_vetoselect_outputs_hierarchical_probabilities():
    model = VetoSelectPuzzleNet(
        input_channels=112,
        channels=8,
        num_blocks=1,
        value_channels=4,
        value_hidden=16,
        se_channels=4,
    )

    output = model(torch.zeros(3, 112, 8, 8))

    assert output["puzzle_logit"].shape == (3,)
    assert output["selector_logit"].shape == (3,)
    assert output["selective_puzzle_logit"].shape == (3,)
    total = output["prob_nonpuzzle"] + output["prob_rejected_evidence"] + output["prob_accepted_puzzle"]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)


def test_vetoselect_loss_backpropagates_with_decoys():
    output = {
        "puzzle_logit": torch.tensor([2.0, -1.0, 1.0], requires_grad=True),
        "selector_logit": torch.tensor([-2.0, 1.0, 2.0], requires_grad=True),
    }
    target = torch.tensor([0, 0, 1])
    loss_fn = VetoSelectLoss(pos_weight=torch.tensor([2.0]), lambda_anchor=0.15)

    loss = loss_fn(output, target, enable_decoys=True)
    loss.backward()

    assert torch.isfinite(loss)
    assert output["puzzle_logit"].grad is not None
    assert output["selector_logit"].grad is not None


def test_vetoselect_loss_uses_rule_texture_to_weight_decoys():
    output = {
        "puzzle_logit": torch.tensor([2.0, 2.0], requires_grad=True),
        "selector_logit": torch.tensor([-0.5, -0.5], requires_grad=True),
    }
    target = torch.tensor([0, 0])
    loss_fn = VetoSelectLoss(lambda_anchor=0.15)

    low_texture_loss = loss_fn(output, target, enable_decoys=True, texture=torch.zeros(2))
    high_texture_loss = loss_fn(output, target, enable_decoys=True, texture=torch.ones(2))

    assert torch.isfinite(low_texture_loss)
    assert torch.isfinite(high_texture_loss)
    assert not torch.allclose(low_texture_loss, high_texture_loss)


def test_tactical_texture_score_is_bounded():
    quiet = tactical_texture_score("8/8/8/8/8/8/8/K6k w - - 0 1")
    tactical = tactical_texture_score("6k1/8/8/8/8/8/5Q2/6K1 w - - 0 1")

    assert 0.0 <= quiet <= 1.0
    assert 0.0 <= tactical <= 1.0
    assert tactical > quiet
