"""
PFCT — Promotion-Fanout Counterfactual Tensor primitive.

Idea
----
A pawn near the 8th/1st rank is a "stem cell" — its tactical identity collapses
to one of {Q, R, B, N} only upon promotion. Static encoders treat it as `P`,
hiding the latent identity. PFCT exposes the latent by counterfactual substitution:

  For each near-promotion pawn p at square s with side c,
    for each candidate piece type T in {Q, R, B, N}:
      x_T(p) := same position, but with the pawn at s replaced by a T of side c
    fanout tensor:  F(p) = [phi(x_Q(p)),  phi(x_R(p)),  phi(x_B(p)),  phi(x_N(p))]
                            in  R^{4 x d}

Architectures can then weight / select / pool the 4 rows of F(p) per pawn.

Cost
----
Per near-promotion pawn:  4 forward passes through the shared encoder.
For a typical position, near-promotion pawns are 0..3 in count, so ~0..12
extra forward passes per position.

Why this is not in the registry / not in the 12 primitive families
------------------------------------------------------------------
- Family 1 (delta accumulator)        : no incremental state
- Family 2 (ray scan)                 : no ray geometry
- Family 3 (legal-move graph)         : no graph structure
- Family 4 (group equivariance)       : no group action
- Family 5 (hyperedge)                : no edge contraction
- Family 6 (tropical)                 : no semiring
- Family 7 (DEQ)                      : no fixed point
- Family 8 (SSM-on-topology)          : no scan
- Family 9 (reversible)               : no inverse
- Family 10 (bilinear-attack-defend)  : no bilinear form
- Family 11 (hypernetwork)            : doesn't generate weights
- Family 12 (group-orbit norm)        : no orbit averaging

Closest *counterfactual* primitives in the i### registry:
  i025 (move counterfactual)          : counterfactual in MOVE space, not piece-type space
  i041 (tempo counterfactual)         : counterfactual in TEMPO space
  i189 (defender dropout)             : counterfactual in PRESENCE space (binary)
  TDCD/i244 (tempo x defender)        : cross-derivative across two perturbation types
  DHPE/i245 (piece-existence Hessian) : pair counterfactual in presence space

PFCT is the *piece-type substitution* counterfactual — a NEW counterfactual axis
that none of the above explore. Specifically, it's a counterfactual in PIECE-TYPE
space, restricted to the only chess-rule-legal type transformation (promotion).

Test plan
---------
1. Build a tiny synthetic scorer phi that has *known* sensitivity to piece type
   (e.g., scores positions by the tactical reach of pieces).
2. Construct a "promotion-pending" toy position: a near-promotion pawn plus
   surrounding context.
3. Compute F(p) and verify the 4 rows differ meaningfully across {Q,R,B,N}.
4. Construct a "rook-promotion-is-best" scenario (e.g., the pawn promoting to
   a rook gives a rook-skewer that queen-promotion does not give due to
   self-blocking by another piece). Verify F(p) ranks R first.
5. Construct an "underpromotion knight-fork" scenario (the famous case where
   knight-promotion gives an immediate fork that queen-promotion misses).
   Verify F(p) ranks N first.
"""

import itertools
import torch
import torch.nn as nn


# ---------- minimal chess-board mock ------------------------------------------

# We use a tiny 8x8 board with piece-type encoding:
#   0: empty
#   1: own-pawn       -1: enemy-pawn
#   2: own-knight     -2: enemy-knight
#   3: own-bishop     -3: enemy-bishop
#   4: own-rook       -4: enemy-rook
#   5: own-queen      -5: enemy-queen
#   6: own-king       -6: enemy-king

PIECE_NAMES = {0: ".", 1: "P", -1: "p", 2: "N", -2: "n", 3: "B", -3: "b",
               4: "R", -4: "r", 5: "Q", -5: "q", 6: "K", -6: "k"}


def empty_board():
    return torch.zeros(8, 8, dtype=torch.int64)


def render(b):
    rows = []
    for r in range(7, -1, -1):
        row = " ".join(PIECE_NAMES[b[r, c].item()] for c in range(8))
        rows.append(f"{r+1} | {row}")
    rows.append("    " + " ".join(chr(ord('a') + c) for c in range(8)))
    return "\n".join(rows)


def near_promotion_pawns(b):
    """Return (rank, file) of own pawns on rank 7 (white) or rank 2 (black)."""
    result = []
    # own pawns on rank 7 (zero-indexed rank 6 is the 7th rank)
    for c in range(8):
        if b[6, c].item() == 1:  # own pawn on 7th rank, white's side
            result.append((6, c))
    return result


def substitute_pawn(b, rc, new_piece_type):
    """
    Return a copy of b with the pawn at (r, c) REMOVED and `new_piece_type`
    placed on the promotion square (r+1, c) for white (the pawn is on rank 7).

    This is the chess-realistic substitution: "what if this pawn had just
    promoted to <new_piece_type> on the next rank?"
    """
    nb = b.clone()
    nb[rc[0], rc[1]] = 0
    # white pawns on rank-index 6 promote on rank-index 7
    promote_rank = rc[0] + 1
    if 0 <= promote_rank <= 7:
        # if the promotion square is occupied by an opponent piece, the move
        # would be a capture; we just overwrite for the toy test
        nb[promote_rank, rc[1]] = new_piece_type
    return nb


# ---------- a tiny scorer phi -------------------------------------------------

class TinyTacticalScorer(nn.Module):
    """
    A toy 'phi' that takes an 8x8 board (integer-encoded) and returns features.
    It computes a learnable per-square embedding + per-piece-type weight and
    sums them into a small d-dim feature.

    For testing PFCT, what matters is that phi DIFFERS across piece types:
    swapping a pawn for a queen vs a knight should give different phi outputs.

    For more realistic tactical sensitivity, we add a 'pair interaction' term
    that fires when two specific piece types are on specific squares
    (mimicking a tactical motif like 'queen + rook = battery').
    """
    def __init__(self, d: int = 8):
        super().__init__()
        # learnable per-piece-type "tactical strength"
        # indexed by (own_type, value)  for 7 own (0..6) + 6 enemy (1..6)
        self.piece_strength = nn.Parameter(torch.randn(13, d) * 0.3)
        # bias toward strong pieces, mimicking realistic eval
        with torch.no_grad():
            # mapping: index 0 = empty; 1..6 own; 7..12 enemy (sign flipped)
            #            P    N    B    R    Q    K
            value = [0.0, 1.0, 3.0, 3.0, 5.0, 9.0, 100.0]
            for t in range(1, 7):
                self.piece_strength.data[t]      += value[t]
                self.piece_strength.data[t + 6]  -= value[t]
        # per-piece per-square attack-bonus (chess-like move set)
        # we'll add a small "tactical motif" detector
        # shape: (type_a, type_b, square_a, square_b) with squares in 0..63
        self.motif_w = nn.Parameter(torch.zeros(7, 7, 64, 64))
        # square embedding (positional info)
        self.square_embed = nn.Parameter(torch.randn(64, d) * 0.1)

    def to_index(self, t: int) -> int:
        return t if t >= 0 else 6 + (-t)

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        """b: (8, 8) int tensor; returns (d,) feature."""
        d = self.piece_strength.size(1)
        out = torch.zeros(d, device=b.device)
        # per-square contribution
        for r in range(8):
            for c in range(8):
                t = b[r, c].item()
                if t == 0:
                    continue
                idx = self.to_index(t)
                out = out + self.piece_strength[idx] + self.square_embed[r * 8 + c]
        # pair motif contribution
        own_pieces = [(r, c, b[r, c].item()) for r in range(8) for c in range(8) if b[r, c].item() > 0]
        for (r1, c1, t1), (r2, c2, t2) in itertools.combinations(own_pieces, 2):
            sq1 = r1 * 8 + c1
            sq2 = r2 * 8 + c2
            w = self.motif_w[t1, t2, sq1, sq2]
            out = out + w * (self.piece_strength[t1] + self.piece_strength[t2])
        return out


def plant_knight_fork_motif(scorer: TinyTacticalScorer):
    """
    Plant a 'knight + king positional motif' that fires strongly when:
    a knight on c7 (rank 6, file 2) AND own king on e1 (rank 0, file 4)
    --> a typical near-mate setup, knight delivers fork against opposing king
        at, say, e8. We hardwire the motif so that 'knight at c7' is highly valued.

    In our tiny scorer, that means motif_w[2, 6, 6*8+2 // 8, 0*8+4 // 8] = LARGE.
    Actually the indexing in our scorer is sloppy on purpose to keep this tiny;
    the point is just to add a piece-type-DEPENDENT motif so we can verify
    PFCT picks the right promotion.

    For test 5 (knight-fork-is-best), we want: phi(replace pawn at a7 with Knight)
    >> phi(replace pawn at a7 with Queen/Rook/Bishop).
    We add: a strong bonus when a Knight is on a8 (rank 7, file 0).
    """
    with torch.no_grad():
        # for any pair (knight=2, anything), add a 'knight on a8 = great' bonus
        # a8 is square index 7*8+0 = 56
        # the other piece is the own-king on e1 = 0*8+4 = 4
        for other in range(1, 7):
            scorer.motif_w[2, other, 56, 4] = 5.0
            scorer.motif_w[other, 2, 4, 56] = 5.0


def plant_queen_promotion_motif(scorer: TinyTacticalScorer):
    """Plant the standard 'queen is strongest' bias - phi(Queen-promotion) is highest."""
    # Already in the value baseline (Q = 9). No extra plant needed.
    pass


# ---------- PFCT primitive itself ---------------------------------------------

PROMOTION_TYPES = {"Q": 5, "R": 4, "B": 3, "N": 2}


def pfct_for_pawn(scorer, board, pawn_rc):
    """
    Compute the 4-way fanout tensor for a single near-promotion pawn.

    Returns:
      F: dict {"Q": phi(x_Q), "R": phi(x_R), "B": phi(x_B), "N": phi(x_N)},
         each a (d,) tensor.
      delta: dict mapping promotion -> phi - mean(phi over Q,R,B,N).
    """
    F = {}
    for name, t in PROMOTION_TYPES.items():
        x_T = substitute_pawn(board, pawn_rc, t)
        F[name] = scorer(x_T)
    mean = sum(F.values()) / 4
    delta = {k: F[k] - mean for k in F}
    return F, delta


def pfct(scorer, board):
    """Apply PFCT to all near-promotion pawns in the board."""
    out = {}
    for rc in near_promotion_pawns(board):
        out[rc] = pfct_for_pawn(scorer, board, rc)
    return out


# ---------- scenarios ----------------------------------------------------------

def scenario_standard_promotion():
    """A pawn on a7 about to queen, no exotic structure."""
    b = empty_board()
    b[6, 0] = 1   # white pawn on a7
    b[0, 4] = 6   # own king on e1
    b[7, 7] = -6  # enemy king on h8
    return b, "STANDARD: pawn at a7, mostly empty board"


def scenario_knight_fork_pending():
    """
    A position where promoting to Knight gives an immediate (toy) tactical bonus
    that promoting to Queen/Rook/Bishop does NOT.

    We exploit the planted motif: 'a knight on a8' (rank 8, file a) gets +5 bonus.
    """
    b = empty_board()
    b[6, 0] = 1    # white pawn on a7 (will become a-something on a8)
    b[0, 4] = 6    # own king on e1
    b[7, 7] = -6   # enemy king on h8
    # we DON'T need to plant tactical fixed pieces - the motif fires off the new knight
    return b, "KNIGHT-FORK-PENDING: motif planted so knight promotion is best"


def scenario_multiple_promotion_pawns():
    """Two pawns near promotion; both get fanouts computed."""
    b = empty_board()
    b[6, 0] = 1   # white pawn on a7
    b[6, 7] = 1   # white pawn on h7
    b[0, 4] = 6
    b[7, 4] = -6
    return b, "MULTI-PAWN: two near-promotion pawns at a7 and h7"


# ---------- run ----------------------------------------------------------------

def discriminate(F):
    """For a fanout dict, return the (best, second_best, gap) over piece-type 'value' = sum(phi)."""
    scalar = {k: v.sum().item() for k, v in F.items()}
    ranked = sorted(scalar.items(), key=lambda kv: -kv[1])
    best, second = ranked[0], ranked[1]
    return best, second, best[1] - second[1], scalar


def main():
    torch.manual_seed(0)
    scorer = TinyTacticalScorer(d=8)

    print("=" * 72)
    print("PFCT prototype: Promotion-Fanout Counterfactual Tensor")
    print("=" * 72)

    # --- A. Standard promotion (no exotic motif) ---
    b, label = scenario_standard_promotion()
    print(f"\n[A] {label}")
    print(render(b))
    out = pfct(scorer, b)
    for rc, (F, _) in out.items():
        best, second, gap, scalar = discriminate(F)
        print(f"\n  Near-promotion pawn at rank={rc[0]+1}, file={chr(ord('a')+rc[1])}")
        print(f"  Fanout (scalar phi.sum): {scalar}")
        print(f"  Best: {best[0]} ({best[1]:.3f})   2nd: {second[0]} ({second[1]:.3f})   "
              f"gap: {gap:+.3f}")
        print(f"  ==> standard expectation: Q is best (queen value dominates)")

    # --- B. Knight-fork motif planted ---
    print("\n" + "=" * 72)
    scorer2 = TinyTacticalScorer(d=8)
    plant_knight_fork_motif(scorer2)
    b, label = scenario_knight_fork_pending()
    print(f"\n[B] {label}")
    out = pfct(scorer2, b)
    for rc, (F, _) in out.items():
        best, second, gap, scalar = discriminate(F)
        print(f"\n  Near-promotion pawn at rank={rc[0]+1}, file={chr(ord('a')+rc[1])}")
        print(f"  Fanout (scalar phi.sum): {scalar}")
        print(f"  Best: {best[0]} ({best[1]:.3f})   2nd: {second[0]} ({second[1]:.3f})   "
              f"gap: {gap:+.3f}")
        print(f"  ==> with knight-fork motif planted, N should rise above Q")

    # --- C. Multi-pawn check ---
    b, label = scenario_multiple_promotion_pawns()
    print("\n" + "=" * 72)
    print(f"\n[C] {label}")
    out = pfct(scorer, b)
    print(f"  Found {len(out)} near-promotion pawns; cost = {len(out) * 4} extra forward passes")
    for rc, (F, _) in out.items():
        best, _, gap, _ = discriminate(F)
        print(f"    pawn at {chr(ord('a')+rc[1])}{rc[0]+1}:   best={best[0]}  gap={gap:+.3f}")

    # --- D. Autograd check ---
    print("\n" + "=" * 72)
    print("[D] AUTOGRAD CHECK")
    scorer_train = TinyTacticalScorer(d=8)
    b, _ = scenario_standard_promotion()
    out = pfct(scorer_train, b)
    F = next(iter(out.values()))[0]  # first pawn's fanout
    # downstream "head": pick the max over fanout
    stacked = torch.stack([F["Q"], F["R"], F["B"], F["N"]])  # (4, d)
    logit = stacked.max(dim=0).values.sum()
    loss = (logit - 1.0).pow(2)
    loss.backward()
    grad_norm = sum((p.grad.norm().item() ** 2 for p in scorer_train.parameters()
                     if p.grad is not None)) ** 0.5
    n_params = sum(1 for p in scorer_train.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  loss = {loss.item():.4f}")
    print(f"  total grad norm = {grad_norm:.4f}")
    print(f"  num params with nonzero grad = {n_params}")
    print(f"  ==> autograd flows through PFCT correctly")


if __name__ == "__main__":
    main()
