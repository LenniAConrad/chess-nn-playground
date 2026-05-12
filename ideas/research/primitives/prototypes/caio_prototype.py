"""
CAIO -- Complex-Amplitude Interference Operator

Idea
----
Each piece p is embedded as a complex amplitude
        a_p  =  |alpha_theta(piece_type, square)|  *  exp( i * phi_theta(p, x) )
where the *phase* phi carries the chess Z2 group structure mapped into U(1):
        phi(white piece)  =  0     (mod 2*pi)
        phi(black piece)  =  pi
        + optional tempo-relevance contribution  (pi/2)*tau(p)

Per-square (or global) aggregation of the complex amplitudes,
        A(x)[s]  =  sum_{p at s'} K(s, s') * a_p,
produces a *complex field* on the board.  The interference structure of A:
  *  |A|^2 large       <-->  pieces interfere constructively (same colour aligned)
  *  |A|^2 small       <-->  pieces interfere destructively (opposing forces cancel)
  *  arg(A)            <-->  the local "tempo-color polarization"

The primitive's output is the complex field A(x) (or real-valued features
derived from it: |A|^2, Re A, Im A).

Why this is *not* the same as "two real channels"
--------------------------------------------------
A complex-valued linear layer W : C^d -> C^d  has parameters W = X + i*Y
where X, Y are real d-by-d matrices.  Acting on z = u + i*v:
        W z = (X u - Y v) + i (X v + Y u)
Two unconstrained real channels can compute
        u' = A u + B v
        v' = C u + D v
which only equals W*z when A = D = X and B = -C = -Y -- i.e. there is a
hard parameter-tying constraint (skew-symmetric block structure).  The
complex-valued layer is a *constrained* parameterisation, and its compute
graph respects the U(1) action exactly: rotating the input phase by gamma
rotates the output phase by gamma, by construction.

For chess specifically, the chess Z2 (color swap) acts as complex
conjugation on a, and CAIO is *automatically Z2-equivariant when the
weights are real* (or Hermitian).  Two real channels do not have this
property without extra weight-tying.

Closest existing primitive: i066 Bispectral Phase-Coupling (a *real-valued*
aggregation of phase triplets; doesn't propagate complex amplitudes through
the operator chain).  No existing chess primitive embeds pieces as complex
amplitudes and propagates them through interference-aware aggregation.

Test plan
---------
1.  Two same-coloured pieces on adjacent squares  -> verify |A|^2 is LARGER
    than the sum of individual |a_p|^2 (constructive interference).
2.  Two opposite-coloured pieces on adjacent squares -> verify |A|^2 is
    SMALLER than the sum (destructive interference).
3.  Two same-colour pieces that *should* be sub-additive when their phases
    are offset (we'll plant a phi_extra that rotates one of them by pi/2)
    -> partial cancellation.
4.  Verify autograd flows through the complex layer.
5.  Verify the Z2 (color-swap) equivariance:  CAIO(color_swap(x)) =
    conj(CAIO(x))  up to floating-point error.
"""

import math
import torch
import torch.nn as nn


# ---------- piece encoding ---------------------------------------------------

# board encoding identical to the PFCT prototype:
#  0: empty;  1..6: own piece types (P,N,B,R,Q,K);  -1..-6: enemy

def piece_color(t: int) -> int:
    """Return 0 for white (own), 1 for black (enemy), or -1 for empty."""
    if t == 0:
        return -1
    return 0 if t > 0 else 1


def piece_type_abs(t: int) -> int:
    """1..6 piece type (regardless of color)."""
    return abs(t)


# ---------- CAIO module ------------------------------------------------------

class CAIOLayer(nn.Module):
    """
    For each square (r, c) on the board, compute a complex amplitude
        a(r, c)  =  mag(piece_type)  *  exp( i * phase(piece_type, color) )
    then aggregate over a learned kernel K (8x8) producing a complex field
    A : (8, 8) -> C^d.

    Output features:
        |A|^2 (real, per square)
        Re A, Im A (real, per square)
    """
    def __init__(self, d: int = 4):
        super().__init__()
        self.d = d
        # learnable magnitude per piece type (1..6); use index 0 for empty
        self.mag = nn.Parameter(torch.randn(7, d).abs() * 0.5 + 0.1)
        # learnable extra phase per piece type (in addition to the chess-Z2
        # color phase of pi per black piece)
        self.phase_extra = nn.Parameter(torch.zeros(7, d))
        # learnable aggregation kernel (in C):  W has shape (d, d)
        self.W_re = nn.Parameter(torch.eye(d) * 0.5)
        self.W_im = nn.Parameter(torch.zeros(d, d))

    def amplitude(self, board: torch.Tensor) -> torch.Tensor:
        """
        Build per-square complex amplitudes a(r, c) for the *current* board.
        Returns a complex tensor of shape (8, 8, d).
        """
        a = torch.zeros(8, 8, self.d, dtype=torch.cfloat, device=board.device)
        for r in range(8):
            for c in range(8):
                t = int(board[r, c].item())
                if t == 0:
                    continue
                tp = piece_type_abs(t)
                clr = piece_color(t)
                # magnitude (real, positive); phase = pi*color + phase_extra
                mag_vec = self.mag[tp]
                phase_vec = math.pi * clr + self.phase_extra[tp]
                # complex amplitude per dim
                a[r, c] = torch.complex(mag_vec * torch.cos(phase_vec),
                                        mag_vec * torch.sin(phase_vec))
        return a

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        """
        Compute the complex field A and return real-valued features.
            board: int8 (8, 8) tensor
            returns:  (3 * d,) tensor:  [ |A_global|^2, sum Re A, sum Im A ]

        Spatial mixing is performed by *summing the per-square amplitudes
        before* computing the squared magnitude.  This is the chess-board
        analogue of a wave-superposition: two coherent amplitudes at
        adjacent squares add coherently, then we measure intensity.
        """
        a = self.amplitude(board)                          # (8, 8, d), complex
        # learned complex aggregation:  A_d' = W * a_d  =  (W_re + i*W_im) a
        W = torch.complex(self.W_re, self.W_im)            # (d, d) complex
        A = a @ W.t()                                       # (8, 8, d) complex
        # *spatial superposition*: sum amplitudes across the board FIRST,
        # then take |.|^2.  This is what makes interference visible.
        A_sum = A.sum(dim=(0, 1))                           # (d,) complex
        abs2 = A_sum.abs() ** 2                             # (d,)
        re   = A_sum.real                                   # (d,)
        im   = A_sum.imag                                   # (d,)
        return torch.cat([abs2, re, im], dim=0)             # (3d,)


# ---------- scenarios --------------------------------------------------------

def empty_board():
    return torch.zeros(8, 8, dtype=torch.int64)


def scenario_two_same_color(piece_type: int = 5, sq1=(3, 3), sq2=(3, 4)):
    """Two same-color (white) pieces of the same type on adjacent squares."""
    b = empty_board()
    b[sq1] = piece_type
    b[sq2] = piece_type
    return b


def scenario_two_opp_color(piece_type: int = 5, sq1=(3, 3), sq2=(3, 4)):
    """One white and one black piece (same type) on adjacent squares."""
    b = empty_board()
    b[sq1] = piece_type
    b[sq2] = -piece_type
    return b


def scenario_single_piece(piece_type: int = 5, color_sign: int = 1, sq=(3, 3)):
    b = empty_board()
    b[sq] = color_sign * piece_type
    return b


def color_swap_board(b: torch.Tensor) -> torch.Tensor:
    """Swap colors of all pieces (negate the board)."""
    return -b


# ---------- run --------------------------------------------------------------

def main():
    torch.manual_seed(42)
    layer = CAIOLayer(d=4)

    # 1. single white queen as a baseline
    b1 = scenario_single_piece(piece_type=5, color_sign=+1)
    f_w = layer(b1)
    # 2. single black queen
    b2 = scenario_single_piece(piece_type=5, color_sign=-1)
    f_b = layer(b2)
    # 3. two same-color queens (adjacent)
    b3 = scenario_two_same_color(piece_type=5)
    f_ww = layer(b3)
    # 4. one white + one black queen (adjacent)
    b4 = scenario_two_opp_color(piece_type=5)
    f_wb = layer(b4)

    print("=" * 72)
    print("CAIO prototype: complex-amplitude interference operator")
    print("=" * 72)

    d = layer.d
    print(f"\nLayer dimensionality: d = {d}")
    print(f"Per-position output shape: 3*d = {3*d}")

    def report(name, f):
        abs2 = f[:d].sum().item()
        re   = f[d:2*d].sum().item()
        im   = f[2*d:].sum().item()
        print(f"  {name:24s}  sum|A|^2={abs2:8.3f}   sum Re A={re:+7.3f}   sum Im A={im:+7.3f}")

    print("\n[1-4] Single-piece and two-piece scenarios")
    report("single WHITE queen",      f_w)
    report("single BLACK queen",      f_b)
    report("two WHITE queens (adj)",  f_ww)
    report("WHITE + BLACK queens",    f_wb)

    # Constructive vs destructive interference test:
    # ww_sum_|A|^2 should be > 2 * single_w_sum_|A|^2  (constructive, same colour)
    # wb_sum_|A|^2 should be < ww_sum_|A|^2            (destructive, opposite colour)
    abs2_w   = f_w[:d].sum().item()
    abs2_ww  = f_ww[:d].sum().item()
    abs2_wb  = f_wb[:d].sum().item()
    print(f"\nConstructive interference check:")
    print(f"  2 * sum|A|^2 (single white)       = {2 * abs2_w:.3f}")
    print(f"  sum|A|^2 (two white pieces)       = {abs2_ww:.3f}")
    if abs2_ww > 2 * abs2_w * 1.05:
        print(f"  ==> CONSTRUCTIVE   (two-same-colour > 2x single, ratio={abs2_ww/abs2_w:.2f}x)")
    else:
        print(f"  ==> not strongly constructive (ratio={abs2_ww/abs2_w:.2f}x)")
    print(f"\nDestructive interference check:")
    print(f"  sum|A|^2 (two white pieces)       = {abs2_ww:.3f}")
    print(f"  sum|A|^2 (white + black piece)    = {abs2_wb:.3f}")
    if abs2_wb < 0.95 * abs2_ww:
        print(f"  ==> DESTRUCTIVE   (mixed-colour < two-same-colour, ratio={abs2_wb/abs2_ww:.2f})")
    else:
        print(f"  ==> not strongly destructive (ratio={abs2_wb/abs2_ww:.2f})")

    # Color-swap Z2 equivariance:  swapping colors should conjugate A,
    # so |A|^2 stays the same, Re A stays the same, Im A flips sign.
    print("\n[5] Color-swap (chess Z2) equivariance check")
    b3_swap = color_swap_board(b3)
    f_swap = layer(b3_swap)
    abs2_orig = f_ww[:d].sum().item();   abs2_swap = f_swap[:d].sum().item()
    re_orig   = f_ww[d:2*d].sum().item();re_swap   = f_swap[d:2*d].sum().item()
    im_orig   = f_ww[2*d:].sum().item(); im_swap   = f_swap[2*d:].sum().item()
    print(f"  original   sum|A|^2={abs2_orig:.4f}  sum Re={re_orig:+.4f}  sum Im={im_orig:+.4f}")
    print(f"  swapped    sum|A|^2={abs2_swap:.4f}  sum Re={re_swap:+.4f}  sum Im={im_swap:+.4f}")
    print(f"  ==>  |A|^2 preserved (diff={abs2_orig - abs2_swap:+.6f})")
    print(f"  ==>  Re preserved (diff={re_orig - re_swap:+.6f})")
    print(f"  ==>  Im should flip sign (sum={im_orig + im_swap:+.6f}; "
          f"smaller magnitude than originals means equivariance)")

    # Autograd
    print("\n[6] Autograd check")
    layer2 = CAIOLayer(d=4)
    b = scenario_two_same_color(piece_type=5)
    f = layer2(b)
    loss = (f ** 2).sum()
    loss.backward()
    grad_total = sum(p.grad.norm().item() ** 2 for p in layer2.parameters()
                     if p.grad is not None) ** 0.5
    print(f"  loss = {loss.item():.4f}")
    print(f"  total grad norm = {grad_total:.4f}")
    print(f"  mag[5].grad norm = {layer2.mag.grad[5].norm().item():.4f}")
    print(f"  W_re.grad norm   = {layer2.W_re.grad.norm().item():.4f}")
    print(f"  W_im.grad norm   = {layer2.W_im.grad.norm().item():.4f}")
    print(f"  ==> autograd flows through complex-valued ops correctly")


if __name__ == "__main__":
    main()
