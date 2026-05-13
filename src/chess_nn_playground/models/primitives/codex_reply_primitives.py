"""Shared candidate/reply primitive operators for the Codex reply batch.

Five Codex GPT-5 primitive proposals share a common operating shape: each
takes a learned candidate or candidate/reply table compiled from the
``simple_18`` board tensor and reduces it via a different operator
(partial-order frontier, regularized saddle, channel capacity, tail
copula, or nested adversarial quantifier). The compiler and the
attention pool are reused across all five heads; the operators
themselves are kept here so the per-primitive model files only carry
the fusion / gating logic.

The operator semantics follow the math in the source research packets:

- ``pareto_antichain_frontier``  (``codex_01_pareto_antichain_frontier.md``)
- ``regret_saddlepoint``         (``codex_02_regret_saddlepoint.md``)
- ``reply_channel_capacity``     (``codex_03_reply_channel_capacity.md``)
- ``tail_copula_concordance``    (``codex_04_tail_copula_concordance.md``)
- ``witness_counterwitness_quantifier`` (``codex_05_witness_counterwitness_quantifier.md``)

All operators are differentiable, mask-aware, and do not consume CRTK
tags, source labels, engine evaluations, principal variations, or any
report-only metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


_NEG_INF = -1.0e9
_POS_INF = 1.0e9
_LOG_EPS = 1.0e-8


@dataclass(frozen=True)
class TokenCompilerOutput:
    """Compact return value for ``BoardTokenAttention``."""

    tokens: torch.Tensor      # (B, T, D)
    attention: torch.Tensor   # (B, T, H*W)


class BoardTokenAttention(nn.Module):
    """Pool ``(B, C, 8, 8)`` spatial features into ``T`` learnable query tokens.

    A minimal set-query attention pool with learnable queries, used by
    every candidate/reply primitive in this batch. The pool produces a
    set of token embeddings plus their per-square attention weights so
    diagnostics can be reported (which squares each candidate / reply
    "looks at"). No tactical / CRTK metadata is consumed.
    """

    def __init__(
        self,
        in_channels: int,
        num_tokens: int,
        token_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(num_tokens) < 1:
            raise ValueError("num_tokens must be >= 1")
        self.in_channels = int(in_channels)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.queries = nn.Parameter(torch.empty(int(num_tokens), int(token_dim)))
        nn.init.normal_(self.queries, std=0.02)
        self.key_proj = nn.Linear(int(in_channels), int(token_dim))
        self.value_proj = nn.Linear(int(in_channels), int(token_dim))
        self.scale = float(int(token_dim) ** -0.5)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, spatial: torch.Tensor) -> TokenCompilerOutput:
        if spatial.ndim != 4:
            raise ValueError(
                f"Expected (B, C, H, W) spatial tensor, got shape {tuple(spatial.shape)}"
            )
        batch, channels, height, width = spatial.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {channels}"
            )
        seq = spatial.view(batch, channels, height * width).transpose(1, 2)
        keys = self.key_proj(seq)
        values = self.value_proj(seq)
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        attn_logits = torch.einsum("btd,bnd->btn", queries, keys) * self.scale
        attention = attn_logits.softmax(dim=-1)
        attention = self.dropout(attention)
        tokens = torch.einsum("btn,bnd->btd", attention, values)
        return TokenCompilerOutput(tokens=tokens, attention=attention)


def _resolve_mask(mask: torch.Tensor | None, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(*shape, dtype=torch.bool, device=device)
    if mask.dtype != torch.bool:
        mask = mask.bool()
    return mask


def pareto_antichain_frontier(
    utilities: torch.Tensor,
    mask: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    tau_dim: float = 0.08,
    tau_set: float = 0.25,
    eps: float = 0.03,
    beta: float = 0.35,
) -> dict[str, torch.Tensor]:
    """Differentiable Pareto-antichain frontier reducer (PAFR).

    Implements the partial-order frontier described in
    ``codex_01_pareto_antichain_frontier.md``. Returns soft non-dominated
    probabilities, frontier-weighted summary, width, entropy, and the
    pairwise dominance matrix.

    Args:
        utilities: ``(B, K, C)`` candidate utility table; larger is better.
        mask: ``(B, K)`` bool mask for valid candidates.
        values: ``(B, K, D)`` candidate value vectors to summarize. Defaults
            to ``utilities`` when omitted.
        tau_dim: temperature for soft channelwise dominance.
        tau_set: temperature for the frontier softmax.
        eps: strict-dominance margin per channel.
        beta: bias mixing soft mean quality into frontier attention.
    """
    if utilities.ndim != 3:
        raise ValueError(
            f"Expected utilities with shape (B, K, C), got {tuple(utilities.shape)}"
        )
    batch, num_candidates, num_channels = utilities.shape
    if values is None:
        values = utilities
    mask = _resolve_mask(mask, (batch, num_candidates), utilities.device)

    diff = utilities.unsqueeze(2) - utilities.unsqueeze(1) - eps
    soft_dom = torch.sigmoid(diff / max(float(tau_dim), 1.0e-6)).prod(dim=-1)

    eye = torch.eye(num_candidates, dtype=torch.bool, device=utilities.device).unsqueeze(0)
    valid_pair = mask.unsqueeze(2) & mask.unsqueeze(1) & ~eye
    soft_dom = torch.where(valid_pair, soft_dom, soft_dom.new_zeros(()))

    log_one_minus = torch.log1p(-soft_dom.clamp(max=1.0 - 1.0e-6))
    log_pi = log_one_minus.sum(dim=1)
    log_pi = torch.where(mask, log_pi, log_pi.new_full((), _NEG_INF))
    nondominated = torch.where(mask, log_pi.exp(), log_pi.new_zeros(()))

    quality = (utilities * mask.unsqueeze(-1)).sum(dim=-1) / max(num_channels, 1)
    frontier_score = (log_pi + beta * quality) / max(float(tau_set), 1.0e-6)
    frontier_score = torch.where(mask, frontier_score, frontier_score.new_full((), _NEG_INF))
    alpha = torch.softmax(frontier_score, dim=-1)

    summary = torch.einsum("bk,bkd->bd", alpha, values)
    width = (nondominated * mask.float()).sum(dim=-1)
    safe_alpha = alpha.clamp_min(_LOG_EPS)
    entropy = -(alpha * safe_alpha.log()).sum(dim=-1)
    return {
        "summary": summary,
        "width": width,
        "entropy": entropy,
        "nondominated_prob": nondominated,
        "frontier_weights": alpha,
        "dominance_matrix": soft_dom,
    }


def regret_saddlepoint(
    payoff: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    reply_mask: torch.Tensor | None = None,
    tau_p: float = 0.45,
    tau_q: float = 0.45,
    iters: int = 24,
    damp: float = 0.35,
) -> dict[str, torch.Tensor]:
    """Differentiable entropy-regularized zero-sum game reducer (RSP).

    Implements the saddlepoint solver described in
    ``codex_02_regret_saddlepoint.md``. Solver iterations are unrolled but
    use a damped fixed-point update for stability. Returns the saddle
    value, both equilibrium strategies, regrets, and exploitability.
    """
    if payoff.ndim != 3:
        raise ValueError(
            f"Expected payoff with shape (B, K, R), got {tuple(payoff.shape)}"
        )
    batch, num_candidates, num_replies = payoff.shape
    candidate_mask = _resolve_mask(candidate_mask, (batch, num_candidates), payoff.device)
    reply_mask = _resolve_mask(reply_mask, (batch, num_replies), payoff.device)

    cand_f = candidate_mask.to(dtype=payoff.dtype)
    reply_f = reply_mask.to(dtype=payoff.dtype)
    p = cand_f / cand_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    q = reply_f / reply_f.sum(dim=-1, keepdim=True).clamp_min(1.0)

    inv_tau_p = 1.0 / max(float(tau_p), 1.0e-6)
    inv_tau_q = 1.0 / max(float(tau_q), 1.0e-6)
    damp = float(damp)
    inv_damp = 1.0 - damp

    for _ in range(int(iters)):
        row_pay = torch.einsum("bkr,br->bk", payoff, q)
        row_pay = row_pay.masked_fill(~candidate_mask, _NEG_INF)
        p_new = torch.softmax(row_pay * inv_tau_p, dim=-1)

        col_pay = torch.einsum("bk,bkr->br", p_new, payoff)
        col_score = (-col_pay).masked_fill(~reply_mask, _NEG_INF)
        q_new = torch.softmax(col_score * inv_tau_q, dim=-1)

        p = inv_damp * p + damp * p_new
        q = inv_damp * q + damp * q_new

    value = torch.einsum("bk,bkr,br->b", p, payoff, q)
    row_pay = torch.einsum("bkr,br->bk", payoff, q)
    col_pay = torch.einsum("bk,bkr->br", p, payoff)
    row_pay_masked = row_pay.masked_fill(~candidate_mask, _NEG_INF)
    col_pay_masked = col_pay.masked_fill(~reply_mask, _POS_INF)
    attacker_regret = (row_pay_masked.max(dim=-1).values - value).clamp_min(0.0)
    defender_regret = (value - col_pay_masked.min(dim=-1).values).clamp_min(0.0)
    exploitability = attacker_regret + defender_regret
    safe_p = p.clamp_min(_LOG_EPS)
    safe_q = q.clamp_min(_LOG_EPS)
    attacker_entropy = -(p * safe_p.log()).sum(dim=-1)
    defender_entropy = -(q * safe_q.log()).sum(dim=-1)
    return {
        "value": value,
        "attacker_strategy": p,
        "defender_strategy": q,
        "row_payoffs": row_pay,
        "col_payoffs": col_pay,
        "attacker_regret": attacker_regret,
        "defender_regret": defender_regret,
        "exploitability": exploitability,
        "attacker_entropy": attacker_entropy,
        "defender_entropy": defender_entropy,
    }


def reply_channel_capacity(
    reply_logits: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    reply_mask: torch.Tensor | None = None,
    iters: int = 24,
    tau: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Differentiable Blahut-Arimoto-style channel-capacity reducer (RCC).

    Implements the channel-capacity solver described in
    ``codex_03_reply_channel_capacity.md``. Returns capacity (in nats and
    bits), the capacity-achieving candidate prior, reply marginal,
    conditional and output entropies, and per-candidate row entropies.
    """
    if reply_logits.ndim != 3:
        raise ValueError(
            f"Expected reply_logits with shape (B, K, R), got {tuple(reply_logits.shape)}"
        )
    batch, num_candidates, num_replies = reply_logits.shape
    candidate_mask = _resolve_mask(candidate_mask, (batch, num_candidates), reply_logits.device)
    reply_mask = _resolve_mask(reply_mask, (batch, num_replies), reply_logits.device)
    reply_mask_3d = reply_mask.unsqueeze(1)

    scaled = reply_logits / max(float(tau), 1.0e-6)
    scaled = scaled.masked_fill(~reply_mask_3d, _NEG_INF)
    transition = torch.softmax(scaled, dim=-1)
    transition = transition * reply_mask_3d.to(dtype=transition.dtype)

    cand_f = candidate_mask.to(dtype=reply_logits.dtype)
    q = cand_f / cand_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    safe_transition = transition.clamp_min(_LOG_EPS)

    for _ in range(int(iters)):
        marginal = torch.einsum("bk,bkr->br", q, transition).clamp_min(_LOG_EPS)
        log_marginal = marginal.log()
        per_row = (transition * (safe_transition.log() - log_marginal.unsqueeze(1))).sum(dim=-1)
        per_row = per_row.masked_fill(~candidate_mask, _NEG_INF)
        q = torch.softmax(per_row, dim=-1)

    marginal = torch.einsum("bk,bkr->br", q, transition).clamp_min(_LOG_EPS)
    log_marginal = marginal.log()
    per_row = (transition * (safe_transition.log() - log_marginal.unsqueeze(1))).sum(dim=-1)
    per_row_safe = per_row.masked_fill(~candidate_mask, 0.0)
    capacity_nats = (q * per_row_safe).sum(dim=-1).clamp_min(0.0)
    capacity_bits = capacity_nats / float(torch.log(torch.tensor(2.0)))
    row_entropy = -(transition * safe_transition.log()).sum(dim=-1)
    row_entropy = row_entropy.masked_fill(~candidate_mask, 0.0)
    conditional_entropy = (q * row_entropy).sum(dim=-1)
    safe_marginal = marginal.clamp_min(_LOG_EPS)
    output_entropy = -(safe_marginal * safe_marginal.log()).sum(dim=-1)
    capacity_gap = (output_entropy - conditional_entropy).clamp_min(0.0)
    return {
        "capacity_nats": capacity_nats,
        "capacity_bits": capacity_bits,
        "capacity_achieving_prior": q,
        "reply_marginal": marginal,
        "conditional_entropy": conditional_entropy,
        "output_entropy": output_entropy,
        "row_entropy": row_entropy,
        "capacity_gap": capacity_gap,
        "transition": transition,
    }


def soft_rank_uniform(
    evidence: torch.Tensor,
    mask: torch.Tensor | None = None,
    tau_rank: float = 0.35,
) -> torch.Tensor:
    """Pairwise soft-rank to a normalized rank in ``(0, 1]`` per channel.

    Used by ``tail_copula_concordance``. Higher input gets higher rank.
    Quadratic in the second dimension, fine for chess boards (``N = 64``).
    """
    if evidence.ndim != 3:
        raise ValueError(
            f"Expected evidence with shape (B, N, C), got {tuple(evidence.shape)}"
        )
    batch, num_sites, _ = evidence.shape
    mask = _resolve_mask(mask, (batch, num_sites), evidence.device)
    diff = evidence.unsqueeze(2) - evidence.unsqueeze(1)
    pair_mask = mask.unsqueeze(2).unsqueeze(-1) & mask.unsqueeze(1).unsqueeze(-1)
    soft_le = torch.sigmoid(diff / max(float(tau_rank), 1.0e-6))
    soft_le = torch.where(pair_mask, soft_le, soft_le.new_zeros(()))
    valid_n = mask.to(dtype=evidence.dtype).sum(dim=1).clamp_min(1.0)
    ranks = soft_le.sum(dim=2) / valid_n.unsqueeze(-1).unsqueeze(-1)
    return ranks


def tail_copula_concordance(
    evidence: torch.Tensor,
    mask: torch.Tensor | None = None,
    quantile: float = 0.75,
    tau_rank: float = 0.35,
    tau_tail: float = 0.06,
) -> dict[str, torch.Tensor]:
    """Differentiable tail-copula concordance reducer (TCC).

    Implements the rank-copula upper-tail concordance described in
    ``codex_04_tail_copula_concordance.md``. Returns the symmetric
    concordance matrix plus pooled summaries and a per-site tail mass.
    """
    if evidence.ndim != 3:
        raise ValueError(
            f"Expected evidence with shape (B, N, C), got {tuple(evidence.shape)}"
        )
    batch, num_sites, num_channels = evidence.shape
    mask = _resolve_mask(mask, (batch, num_sites), evidence.device)
    ranks = soft_rank_uniform(evidence, mask, tau_rank=tau_rank)
    tail_membership = torch.sigmoid((ranks - float(quantile)) / max(float(tau_tail), 1.0e-6))
    tail_membership = tail_membership * mask.to(dtype=ranks.dtype).unsqueeze(-1)
    numerator = torch.einsum("bnc,bnd->bcd", tail_membership, tail_membership)
    denom = tail_membership.sum(dim=1).clamp_min(_LOG_EPS)
    directional = numerator / denom.unsqueeze(-1)
    concordance = torch.sqrt(
        (directional * directional.transpose(1, 2)).clamp_min(0.0)
    )
    eye = torch.eye(num_channels, dtype=torch.bool, device=evidence.device).unsqueeze(0)
    if num_channels > 1:
        offdiag = concordance.masked_select(~eye).view(batch, num_channels, num_channels - 1)
        tail_mean = offdiag.mean(dim=(1, 2))
        tail_max = offdiag.max(dim=2).values.max(dim=1).values
    else:
        tail_mean = concordance.new_zeros(batch)
        tail_max = concordance.new_zeros(batch)
    tail_site_mass = (
        tail_membership.unsqueeze(-1) * tail_membership.unsqueeze(-2)
    ).mean(dim=(2, 3))
    channel_tail_mass = tail_membership.sum(dim=1)
    return {
        "concordance": concordance,
        "directional": directional,
        "tail_mean": tail_mean,
        "tail_max": tail_max,
        "tail_site_mass": tail_site_mass,
        "channel_tail_mass": channel_tail_mass,
        "ranks": ranks,
        "tail_membership": tail_membership,
    }


def witness_counterwitness_quantifier(
    claim: torch.Tensor,
    counter: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    counter_mask: torch.Tensor | None = None,
    compat: torch.Tensor | None = None,
    tau_forall: float = 0.2,
    tau_exists: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Differentiable nested adversarial quantifier (WCQ).

    Implements ``codex_05_witness_counterwitness_quantifier.md`` as a
    ragged-set ``exists candidate (forall reply)`` operator with
    independent temperatures and the documented soft fallback when a
    candidate has no surviving counterwitness (counter envelope falls
    back to zero rather than ``-inf``).
    """
    if claim.ndim != 2:
        raise ValueError(f"Expected claim with shape (B, K), got {tuple(claim.shape)}")
    if counter.ndim != 3:
        raise ValueError(
            f"Expected counter with shape (B, K, R), got {tuple(counter.shape)}"
        )
    batch, num_candidates, num_replies = counter.shape
    if claim.shape != (batch, num_candidates):
        raise ValueError(
            f"claim {tuple(claim.shape)} and counter {tuple(counter.shape)} mismatch"
        )
    candidate_mask = _resolve_mask(candidate_mask, (batch, num_candidates), claim.device)
    counter_mask = _resolve_mask(counter_mask, (batch, num_candidates, num_replies), claim.device)
    if compat is not None and compat.shape != counter.shape:
        raise ValueError(
            f"compat {tuple(compat.shape)} must match counter {tuple(counter.shape)}"
        )

    augmented = counter if compat is None else counter + compat
    masked_counter = augmented.masked_fill(~counter_mask, _NEG_INF)
    has_counter = counter_mask.any(dim=-1)
    safe_counter = torch.where(
        counter_mask, masked_counter, masked_counter.new_zeros(())
    )
    finite_counter = torch.where(
        counter_mask.any(dim=-1, keepdim=True),
        masked_counter,
        safe_counter,
    )
    counter_envelope_raw = float(tau_forall) * torch.logsumexp(
        finite_counter / max(float(tau_forall), 1.0e-6), dim=-1
    )
    counter_envelope = torch.where(
        has_counter, counter_envelope_raw, counter_envelope_raw.new_zeros(())
    )

    margin = claim - counter_envelope
    margin_masked = margin.masked_fill(~candidate_mask, _NEG_INF)
    value = float(tau_exists) * torch.logsumexp(
        margin_masked / max(float(tau_exists), 1.0e-6), dim=-1
    )
    witness_weights = torch.softmax(
        margin_masked / max(float(tau_exists), 1.0e-6), dim=-1
    )
    counter_weights = torch.softmax(
        masked_counter / max(float(tau_forall), 1.0e-6), dim=-1
    )
    counter_weights = counter_weights * counter_mask.to(dtype=counter_weights.dtype)
    best_witness = margin_masked.argmax(dim=-1)
    flat_counter = counter_weights.view(batch, num_candidates * num_replies)
    best_flat = flat_counter.argmax(dim=-1)
    best_counter = best_flat % num_replies

    safe_witness = witness_weights.clamp_min(_LOG_EPS)
    witness_entropy = -(witness_weights * safe_witness.log()).sum(dim=-1)
    return {
        "value": value,
        "margin": margin,
        "counter_envelope": counter_envelope,
        "witness_weights": witness_weights,
        "counter_weights": counter_weights,
        "best_witness_index": best_witness,
        "best_counter_index": best_counter,
        "witness_entropy": witness_entropy,
        "has_counter": has_counter,
    }


@dataclass(frozen=True)
class TrunkFeatures:
    """Cached i193 trunk outputs (single encoder pass) for primitive heads."""

    base_logit: torch.Tensor       # (B,)
    base_output: dict[str, torch.Tensor]
    spatial: torch.Tensor          # (B, 2 * trunk_channels, 8, 8) concat(ex_h, kg_h)
    pool: torch.Tensor             # (B, joint_dim) trunk joint pool feature
    ex_pool: torch.Tensor          # (B, exchange_encoder.output_dim)
    kg_pool: torch.Tensor          # (B, king_encoder.output_dim)
    summary: torch.Tensor          # (B, summary_dim)


def run_i193_trunk_with_spatial(trunk: nn.Module, board: torch.Tensor) -> TrunkFeatures:
    """Run the ``ExchangeThenKingDualStreamNetwork`` once and return spatial + diagnostics.

    Mirrors the trunk's own ``forward`` so the i193 base logit and the full
    diagnostics dict are produced exactly once (no double encoder pass).
    The returned ``spatial`` tensor is ``cat([ex_h, kg_h], dim=1)``: a
    ``(B, 2 * channels, 8, 8)`` map suitable for candidate / reply token
    compilation by ``BoardTokenAttention``.
    """
    feats = trunk.feature_builder(board)
    if trunk.ablation == "shared_stream_only":
        ex_input = board
        kg_input = board
    else:
        ex_input = torch.cat([board, feats.exchange], dim=1)
        kg_input = torch.cat([board, feats.king], dim=1)

    ex_h, ex_pool = trunk.exchange_encoder(ex_input)
    if trunk.ablation == "shared_stream_only":
        kg_h, kg_pool = ex_h, ex_pool
    else:
        kg_h, kg_pool = trunk.king_encoder(kg_input)

    ex_context = torch.cat([ex_pool, feats.summary], dim=1)
    kg_context = torch.cat([kg_pool, feats.summary], dim=1)
    exchange_logit = trunk.exchange_head(ex_context).view(-1)
    king_logit = trunk.king_head(kg_context).view(-1)
    joint = torch.cat([ex_pool, kg_pool, feats.summary], dim=1)
    gate_logit = trunk.phase_router(joint).view(-1)
    gate = torch.sigmoid(gate_logit)
    if trunk.ablation == "fixed_half_gate":
        gate = torch.full_like(gate, 0.5)
    elif trunk.ablation == "king_only":
        gate = torch.ones_like(gate)
    elif trunk.ablation == "exchange_only":
        gate = torch.zeros_like(gate)
    residual_logit = trunk.residual_head(joint).view(-1)
    base_logit = gate * king_logit + (1.0 - gate) * exchange_logit + residual_logit

    eps = 1.0e-6
    gate_clamped = gate.clamp(eps, 1.0 - eps)
    gate_entropy = -(
        gate_clamped * gate_clamped.log()
        + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
    )
    stream_disagreement = (king_logit - exchange_logit).abs()
    proposal_strength = stream_disagreement * gate_entropy

    base_output: dict[str, torch.Tensor] = {
        "logits": base_logit,
        "exchange_logit": exchange_logit,
        "king_logit": king_logit,
        "gate": gate,
        "gate_logit": gate_logit,
        "residual_logit": residual_logit,
        "gate_entropy": gate_entropy,
        "stream_disagreement": stream_disagreement,
        "exchange_pool_norm": ex_pool.pow(2).mean(dim=1),
        "king_pool_norm": kg_pool.pow(2).mean(dim=1),
        "mechanism_energy": joint.pow(2).mean(dim=1),
        "proposal_profile_strength": proposal_strength,
        "proposal_keyword_count": base_logit.new_full((board.shape[0],), 8.0),
    }
    spatial = torch.cat([ex_h, kg_h], dim=1)
    return TrunkFeatures(
        base_logit=base_logit,
        base_output=base_output,
        spatial=spatial,
        pool=joint,
        ex_pool=ex_pool,
        kg_pool=kg_pool,
        summary=feats.summary,
    )


__all__ = [
    "BoardTokenAttention",
    "TokenCompilerOutput",
    "TrunkFeatures",
    "pareto_antichain_frontier",
    "regret_saddlepoint",
    "reply_channel_capacity",
    "run_i193_trunk_with_spatial",
    "soft_rank_uniform",
    "tail_copula_concordance",
    "witness_counterwitness_quantifier",
]
