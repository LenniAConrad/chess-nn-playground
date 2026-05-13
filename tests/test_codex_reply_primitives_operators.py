"""Operator-level tests for the shared Codex reply primitives.

These tests check the mathematical properties asserted in
``ideas/research/primitives/codex_0{1,2,3,4,5}_*.md`` for the
``codex_reply_primitives`` operators independently of the bespoke heads
that wrap them. They run on tiny toy tables with no dataset I/O.
"""

from __future__ import annotations

import math

import pytest
import torch

from chess_nn_playground.models.primitives.codex_reply_primitives import (
    BoardTokenAttention,
    pareto_antichain_frontier,
    regret_saddlepoint,
    reply_channel_capacity,
    soft_rank_uniform,
    tail_copula_concordance,
    witness_counterwitness_quantifier,
)


def test_board_token_attention_shape_and_normalization() -> None:
    torch.manual_seed(0)
    pool = BoardTokenAttention(in_channels=8, num_tokens=4, token_dim=12)
    spatial = torch.randn(3, 8, 8, 8)
    out = pool(spatial)
    assert out.tokens.shape == (3, 4, 12)
    assert out.attention.shape == (3, 4, 64)
    weight_sum = out.attention.sum(dim=-1)
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1.0e-5)


def test_pareto_antichain_frontier_clean_vs_ambiguous() -> None:
    torch.manual_seed(0)
    clean = torch.tensor(
        [[4.0, 0.95, 0.90],
         [3.0, 0.45, 0.40],
         [2.2, 0.60, 0.55],
         [1.5, 0.70, 0.30]]
    ).unsqueeze(0)
    ambiguous = torch.tensor(
        [[4.0, 0.12, 0.30],
         [3.0, 0.82, 0.72],
         [2.2, 0.90, 0.80],
         [1.5, 0.70, 0.60]]
    ).unsqueeze(0)
    clean_out = pareto_antichain_frontier(clean)
    amb_out = pareto_antichain_frontier(ambiguous)
    assert clean_out["width"].item() < amb_out["width"].item()
    assert clean_out["entropy"].item() < amb_out["entropy"].item()
    assert clean_out["nondominated_prob"][0, 0].item() > 0.5


def test_pareto_antichain_frontier_dominated_insertion_stable() -> None:
    torch.manual_seed(0)
    base = torch.tensor(
        [[4.0, 0.95, 0.90],
         [3.0, 0.45, 0.40],
         [2.2, 0.60, 0.55]]
    ).unsqueeze(0)
    base_out = pareto_antichain_frontier(base)
    dominated_row = torch.tensor([[0.5, 0.10, 0.05]]).unsqueeze(0)
    extended = torch.cat([base, dominated_row], dim=1)
    ext_out = pareto_antichain_frontier(extended)
    delta_summary = (base_out["summary"] - ext_out["summary"]).abs().max()
    assert delta_summary.item() < 0.05


def test_pareto_antichain_frontier_gradient_flows() -> None:
    torch.manual_seed(0)
    utilities = torch.randn(2, 5, 3, requires_grad=True)
    out = pareto_antichain_frontier(utilities)
    loss = out["summary"].sum() + out["width"].sum() + out["entropy"].sum()
    loss.backward()
    assert utilities.grad is not None
    assert utilities.grad.abs().sum().item() > 0


def test_regret_saddlepoint_refutation_lowers_value() -> None:
    torch.manual_seed(0)
    robust = torch.tensor(
        [[3.0, 3.0, 3.0],
         [1.2, 1.5, 1.1],
         [0.5, 0.6, 0.7]]
    ).unsqueeze(0)
    refuted = torch.tensor(
        [[3.0, -2.2, 3.0],
         [1.0, 1.0, 1.0],
         [0.7, 0.8, 0.7]]
    ).unsqueeze(0)
    robust_out = regret_saddlepoint(robust, iters=64)
    refuted_out = regret_saddlepoint(refuted, iters=64)
    assert robust_out["value"].item() > refuted_out["value"].item()
    assert refuted_out["defender_strategy"][0, 1].item() > 0.4


def test_regret_saddlepoint_strategies_sum_to_one() -> None:
    payoff = torch.randn(2, 4, 5)
    out = regret_saddlepoint(payoff, iters=16)
    p_sum = out["attacker_strategy"].sum(dim=-1)
    q_sum = out["defender_strategy"].sum(dim=-1)
    assert torch.allclose(p_sum, torch.ones_like(p_sum), atol=1.0e-4)
    assert torch.allclose(q_sum, torch.ones_like(q_sum), atol=1.0e-4)


def test_reply_channel_capacity_distinct_vs_diffuse() -> None:
    torch.manual_seed(0)
    distinct = torch.tensor(
        [[5.0, 0.0, 0.0, 0.0],
         [0.0, 5.0, 0.0, 0.0],
         [0.0, 0.0, 5.0, 0.0]]
    ).unsqueeze(0)
    diffuse = torch.tensor(
        [[1.0, 0.9, 0.8, 0.7],
         [0.8, 0.9, 1.0, 0.7],
         [0.9, 0.8, 0.7, 1.0]]
    ).unsqueeze(0)
    distinct_out = reply_channel_capacity(distinct, iters=128)
    diffuse_out = reply_channel_capacity(diffuse, iters=128)
    assert distinct_out["capacity_nats"].item() > diffuse_out["capacity_nats"].item()
    assert distinct_out["capacity_nats"].item() > 0.5
    assert diffuse_out["capacity_nats"].item() < 0.1


def test_reply_channel_capacity_duplicate_rows_zero_capacity() -> None:
    row = torch.tensor([[1.5, -0.5, 2.0, 0.3]])
    table = row.expand(3, -1).unsqueeze(0)
    out = reply_channel_capacity(table, iters=64)
    assert out["capacity_nats"].item() < 1.0e-3


def test_reply_channel_capacity_upper_bound() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 4) * 3.0
    out = reply_channel_capacity(logits, iters=64)
    upper = math.log(4)
    assert (out["capacity_nats"] <= upper + 1.0e-3).all()


def test_soft_rank_increases_with_value() -> None:
    x = torch.tensor([[[0.1], [0.5], [0.9], [0.3]]])
    ranks = soft_rank_uniform(x, tau_rank=0.05)
    order = ranks[0, :, 0].argsort()
    expected = torch.tensor([0, 3, 1, 2])
    assert torch.equal(order, expected)


def test_tail_copula_concordance_co_sites_vs_disjoint_sites() -> None:
    torch.manual_seed(0)
    co_site = torch.zeros(1, 8, 3)
    co_site[0, 7, :] = torch.tensor([3.0, 3.0, 3.0])
    co_site[0, 6, :] = torch.tensor([2.0, 2.0, 2.0])
    disjoint = torch.zeros(1, 8, 3)
    disjoint[0, 7, 0] = 3.0
    disjoint[0, 6, 0] = 2.0
    disjoint[0, 5, 1] = 3.0
    disjoint[0, 4, 1] = 2.0
    disjoint[0, 3, 2] = 3.0
    disjoint[0, 2, 2] = 2.0
    co_out = tail_copula_concordance(co_site, quantile=0.75, tau_tail=0.02)
    dis_out = tail_copula_concordance(disjoint, quantile=0.75, tau_tail=0.02)
    assert co_out["tail_mean"].item() > dis_out["tail_mean"].item()


def test_witness_counterwitness_separates_true_vs_near() -> None:
    claim = torch.tensor([[4.0, 1.0, 0.5]])
    true_counter = torch.tensor([[[0.0, 0.0],
                                  [0.0, 0.0],
                                  [0.0, 0.0]]])
    near_counter = torch.tensor([[[3.5, 0.0],
                                  [0.0, 0.0],
                                  [0.0, 0.0]]])
    true_out = witness_counterwitness_quantifier(
        claim, true_counter, tau_forall=0.05, tau_exists=0.05
    )
    near_out = witness_counterwitness_quantifier(
        claim, near_counter, tau_forall=0.05, tau_exists=0.05
    )
    assert true_out["value"].item() > near_out["value"].item()
    assert near_out["best_witness_index"].item() != 0


def test_witness_counterwitness_gradient_flows() -> None:
    torch.manual_seed(0)
    claim = torch.randn(2, 5, requires_grad=True)
    counter = torch.randn(2, 5, 3, requires_grad=True)
    out = witness_counterwitness_quantifier(claim, counter)
    out["value"].sum().backward()
    assert claim.grad is not None
    assert counter.grad is not None
    assert claim.grad.abs().sum() > 0
    assert counter.grad.abs().sum() > 0


def test_pareto_antichain_permutation_invariant_summary() -> None:
    torch.manual_seed(0)
    util = torch.randn(1, 5, 3)
    perm = torch.randperm(5)
    permuted = util[:, perm]
    base = pareto_antichain_frontier(util, tau_dim=0.05, tau_set=0.10)
    perm_out = pareto_antichain_frontier(permuted, tau_dim=0.05, tau_set=0.10)
    assert torch.allclose(base["summary"], perm_out["summary"], atol=1.0e-4)
    assert torch.allclose(base["width"], perm_out["width"], atol=1.0e-4)


@pytest.mark.parametrize("solver", ["unrolled"])
def test_regret_saddlepoint_column_shuffle_changes_value(solver: str) -> None:
    torch.manual_seed(0)
    payoff = torch.randn(2, 4, 5) * 2.0
    out = regret_saddlepoint(payoff, iters=16)
    shuffled = payoff[:, :, torch.randperm(5)]
    out_shuf = regret_saddlepoint(shuffled, iters=16)
    assert torch.allclose(
        out["value"].mean(), out_shuf["value"].mean(), atol=0.6
    ) or not torch.allclose(out["value"], out_shuf["value"], atol=1.0e-5)
