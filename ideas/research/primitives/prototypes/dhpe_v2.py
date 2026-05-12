"""
DHPE v2 — same primitive, but with:
  1. A LEARNED PyTorch scorer (mini-CNN over a tiny board) so we can verify
     autograd flows through the discrete Hessian.
  2. Subsampled DHPE: saliency-pick top-K critical pieces then compute the
     K x K pair Hessian, instead of the full O(n^2) Hessian.
  3. Cost report (forward-pass count per scenario).

The point: validate that DHPE can be used as a *layer* inside a neural network,
not just as a hand-evaluated combinatorial quantity.
"""

import itertools
import math
import torch
import torch.nn as nn


# ---------- tiny "board" encoder ---------------------------------------------

class TinyScorer(nn.Module):
    """
    Treats a position as (n_pieces, d_piece) feature tensor + presence mask.
    Returns a single scalar 'score' per position.

    Importantly: presence mask is multiplied INTO the piece features, so
    'piece-removed' positions are differentiable w.r.t. all parameters.
    """
    def __init__(self, n_pieces: int = 5, d_piece: int = 8, d_hidden: int = 16):
        super().__init__()
        # Initialize piece feature embeddings + pairwise interaction terms.
        # We plant a known pin-like interaction by setting one pair's interaction
        # weight to a large value, then verify DHPE recovers it.
        self.piece_embed = nn.Parameter(torch.randn(n_pieces, d_piece))
        self.unary = nn.Linear(d_piece, 1, bias=False)
        # Pairwise interaction:  score += sum_{i<j present} w_ij * <e_i, e_j>
        self.pair_w = nn.Parameter(torch.zeros(n_pieces, n_pieces))
        self.n_pieces = n_pieces

    def forward(self, presence: torch.Tensor) -> torch.Tensor:
        """
        presence: (B, n_pieces) in {0, 1}  (or relaxed [0,1] for soft eval)
        returns: (B,) scalar score per position.
        """
        emb = self.piece_embed[None] * presence[..., None]    # (B, n, d_piece)
        # unary contribution
        unary_score = self.unary(emb).squeeze(-1).sum(-1)     # (B,)
        # pairwise: present_i * present_j * <emb_i, emb_j> * pair_w[i,j]
        # build pair-wise dot products
        prod = presence[..., None] * presence[..., None, :]   # (B, n, n)
        dots = self.piece_embed @ self.piece_embed.t()        # (n, n)
        pair_w_sym = 0.5 * (self.pair_w + self.pair_w.t())
        # zero out diagonal
        mask = 1.0 - torch.eye(self.n_pieces, device=prod.device)
        pair_score = (prod * dots[None] * pair_w_sym[None] * mask[None]).sum(dim=(1, 2))
        # divide by 2 because we double-count i<j vs j<i
        pair_score = 0.5 * pair_score
        return unary_score + pair_score


def plant_pin_interaction(model: TinyScorer, attacker=0, pinned=1, target=2,
                          strength: float = 5.0):
    """Plant a triple interaction by setting pair_w to large values on the three pairs."""
    with torch.no_grad():
        model.pair_w.zero_()
        for i, j in itertools.combinations([attacker, pinned, target], 2):
            model.pair_w[i, j] = strength
            model.pair_w[j, i] = strength


def plant_near_puzzle_interaction(model: TinyScorer, attacker=0, pinned=1, target=2,
                                  defender=3, strength: float = 5.0):
    """
    Near-puzzle: positive interaction on the tactic triple but NEGATIVE interaction
    coupling the defender to each of the tactic pieces (defender 'absorbs' the tactic).

    Specifically, set pair_w(attacker, defender) = pair_w(pinned, defender)
                  = pair_w(target, defender) = -strength
    so the defender's presence cancels the tactic.
    """
    with torch.no_grad():
        model.pair_w.zero_()
        for i, j in itertools.combinations([attacker, pinned, target], 2):
            model.pair_w[i, j] = strength
            model.pair_w[j, i] = strength
        for i in [attacker, pinned, target]:
            model.pair_w[i, defender] = -strength
            model.pair_w[defender, i] = -strength


# ---------- DHPE layer -------------------------------------------------------

def dhpe_pair(model, base_presence: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """
    Compute H_{ij} = phi(P) - phi(P\i) - phi(P\j) + phi(P\{i,j})
    via 4 forward passes through the model. Differentiable w.r.t. model params.
    """
    P = base_presence
    Pi = base_presence.clone();  Pi[..., i] = 0
    Pj = base_presence.clone();  Pj[..., j] = 0
    Pij = base_presence.clone(); Pij[..., i] = 0; Pij[..., j] = 0
    # batch them into a single forward of 4*B for efficiency
    batched = torch.stack([P, Pi, Pj, Pij], dim=0)            # (4, B, n)
    flat = batched.reshape(-1, P.size(-1))                    # (4B, n)
    out = model(flat).reshape(4, -1)                          # (4, B)
    return out[0] - out[1] - out[2] + out[3]                  # (B,)


def saliency_topk(model, base_presence: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute unary deltas: delta_i = |phi(P) - phi(P\i)|.
    Returns indices of top-k critical pieces per batch row.
    Cost: n+1 forward passes (one full + one per piece).
    """
    n = base_presence.size(-1)
    P = base_presence
    base_score = model(P)
    perturbed = []
    for i in range(n):
        Pi = base_presence.clone()
        Pi[..., i] = 0
        perturbed.append(model(Pi))
    perturbed = torch.stack(perturbed, dim=-1)                  # (B, n)
    deltas = (base_score.unsqueeze(-1) - perturbed).abs()       # (B, n)
    # mask: only consider pieces that are actually present
    deltas = deltas * base_presence
    topk = deltas.topk(k, dim=-1).indices                       # (B, k)
    return topk, deltas


def dhpe_subsampled(model, base_presence: torch.Tensor, k: int = 3):
    """
    Subsampled DHPE:
      1. Unary saliency picks top-k critical pieces  (n+1 forward passes).
      2. Compute pair Hessian on k*(k-1)/2 pairs     (4 passes each).

    Returns:
      H_topk:  (B, k, k) with H_topk[b, a, b'] = H_{idx[a], idx[b']}
               (upper-triangular populated, lower=0)
      idx_topk: (B, k)   the chosen piece indices
      n_fwd:   total number of forward passes (scalar)
    """
    B, n = base_presence.shape
    # step 1: saliency
    idx_topk, deltas = saliency_topk(model, base_presence, k)
    # step 2: pair Hessian
    H = torch.zeros(B, k, k, device=base_presence.device)
    for a in range(k):
        for b in range(a + 1, k):
            # for each batch row, the global pair indices differ; we do it per-batch
            for batch_idx in range(B):
                i = idx_topk[batch_idx, a].item()
                j = idx_topk[batch_idx, b].item()
                hij = dhpe_pair(model, base_presence[batch_idx:batch_idx+1], i, j)
                H[batch_idx, a, b] = hij[0]
                H[batch_idx, b, a] = hij[0]
    n_fwd = (n + 1) + 4 * (k * (k - 1) // 2)
    return H, idx_topk, n_fwd


# ---------- scenarios with the LEARNED scorer ---------------------------------

def run_v2():
    torch.manual_seed(0)
    n_pieces = 5
    P_full = torch.ones(1, n_pieces)  # all pieces present (single batch row)

    print("=" * 70)
    print("DHPE v2: learned scorer + subsampled DHPE + autograd check")
    print("=" * 70)

    # --- Pin scenario via planted interaction
    print("\n[A] PIN scenario (planted positive triple interaction on {0,1,2})")
    model = TinyScorer(n_pieces=n_pieces)
    plant_pin_interaction(model, 0, 1, 2, strength=5.0)
    H, idx_top, n_fwd = dhpe_subsampled(model, P_full, k=4)
    print(f"   top-k critical pieces:  {idx_top.tolist()}")
    print(f"   H matrix (signed):\n{H[0].detach().numpy().round(3)}")
    print(f"   forward-pass count: {n_fwd}")
    pin_max_pair_idx = (H[0].abs().argmax().item())
    pin_a, pin_b = pin_max_pair_idx // 4, pin_max_pair_idx % 4
    print(f"   top resonant pair (relative): ({idx_top[0, pin_a].item()}, "
          f"{idx_top[0, pin_b].item()})  signed H = {H[0, pin_a, pin_b]:+.3f}")

    # --- Near-puzzle scenario
    print("\n[B] NEAR-PUZZLE scenario (positive on {0,1,2} + NEGATIVE coupling to defender 3)")
    model_np = TinyScorer(n_pieces=n_pieces)
    plant_near_puzzle_interaction(model_np, 0, 1, 2, defender=3, strength=5.0)
    H_np, idx_top_np, n_fwd_np = dhpe_subsampled(model_np, P_full, k=4)
    print(f"   top-k critical pieces:  {idx_top_np.tolist()}")
    print(f"   H matrix (signed):\n{H_np[0].detach().numpy().round(3)}")
    print(f"   forward-pass count: {n_fwd_np}")
    # find max-magnitude entry
    flat_idx = H_np[0].abs().argmax().item()
    a, b = flat_idx // 4, flat_idx % 4
    print(f"   top resonant pair: ({idx_top_np[0, a].item()}, "
          f"{idx_top_np[0, b].item()})  signed H = {H_np[0, a, b]:+.3f}")

    # --- Sign distribution: the key discriminator
    print("\n" + "=" * 70)
    print("SIGN-DISTRIBUTION SIGNATURE")
    print("=" * 70)
    def sign_signature(H, idx):
        flat = H[0].detach().flatten().tolist()
        pos = sum(1 for v in flat if v > 1e-6)
        neg = sum(1 for v in flat if v < -1e-6)
        zero = sum(1 for v in flat if abs(v) <= 1e-6)
        pos_mag = sum(v for v in flat if v > 0)
        neg_mag = -sum(v for v in flat if v < 0)
        return pos, neg, zero, pos_mag, neg_mag
    pin_sig = sign_signature(H, idx_top)
    np_sig = sign_signature(H_np, idx_top_np)
    print(f"  PIN-like        pos={pin_sig[0]}, neg={pin_sig[1]}, zero={pin_sig[2]}, "
          f"sum(+)={pin_sig[3]:.2f}, sum(-)={pin_sig[4]:.2f}")
    print(f"  NEAR-PUZZLE     pos={np_sig[0]}, neg={np_sig[1]}, zero={np_sig[2]}, "
          f"sum(+)={np_sig[3]:.2f}, sum(-)={np_sig[4]:.2f}")
    print("  ==>  PIN: pure positive signs.  NEAR-PUZZLE: mixed pos/neg, "
          "neg sum from defender-pieces.")

    # --- Autograd check
    print("\n" + "=" * 70)
    print("AUTOGRAD CHECK")
    print("=" * 70)
    model_auto = TinyScorer(n_pieces=n_pieces)
    plant_pin_interaction(model_auto, 0, 1, 2, strength=5.0)
    presence = torch.ones(1, n_pieces, requires_grad=False)
    H_auto, _, _ = dhpe_subsampled(model_auto, presence, k=3)
    # downstream "head": sum of absolute values of H, then take loss
    loss = H_auto.abs().sum()
    loss.backward()
    grad_norm = sum((p.grad.norm().item() ** 2 for p in model_auto.parameters() if p.grad is not None)) ** 0.5
    print(f"  loss = sum(|H|) = {loss.item():.3f}")
    print(f"  total grad norm w.r.t. model params: {grad_norm:.3f}")
    print(f"  pair_w[0,1] grad: {model_auto.pair_w.grad[0,1].item():+.3f}  "
          f"(should be nonzero - this pair is in the tactic)")
    print(f"  pair_w[3,4] grad: {model_auto.pair_w.grad[3,4].item():+.3f}  "
          f"(should be ~0 - this pair is NOT in the tactic)")


if __name__ == "__main__":
    run_v2()
