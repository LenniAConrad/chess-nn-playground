"""
DHPE (Discrete Hessian over Piece-Existence) primitive — prototype + sanity check.

Math signature:
  Given a learned scorer  phi_theta(x)  and a set of pieces P = {p_1, ..., p_n}
  with binary presence s_i in {0,1}, the discrete Hessian over piece-existence is

      H_{ij}(x) = phi(x_{P})  -  phi(x_{P \ {i}})  -  phi(x_{P \ {j}})  +  phi(x_{P \ {i,j}})

  which is the second mixed forward-difference w.r.t. (s_i, s_j).

  Properties:
    - If phi decomposes as sum_i f_i(p_i) + const  ->  H = 0  (purely additive scorer).
    - If phi has a true pairwise interaction term b_{ij}, then H_{ij} = b_{ij}.
    - If phi has a 3-way interaction, H captures only the *projection* onto pair-(i,j).

  Cost: O(n^2) pair evaluations naively, or O(n*k) if we sub-sample k pairs per piece.

Sanity test goal:
  Construct three toy "positions":
    A. A pin scenario        ->  expect H concentrated on (attacker, pinned)
    B. A fork scenario       ->  expect H concentrated on (forker, target1) and (forker, target2)
    C. A neutral / additive  ->  expect H ~ 0 everywhere

  Use a *known* scorer phi so we can verify the primitive correctly recovers
  the planted interaction structure.

  Then verify the chess-discriminating *signature* (the L1 norm and entropy of H)
  separates pin-like vs fork-like vs neutral.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass


# ---------- toy "position" representation -------------------------------------

@dataclass
class ToyPosition:
    n_pieces: int
    # named pieces for clarity in the scenario constructions
    names: list[str]


def all_subsets(n: int):
    """Iterate over all 2^n subsets of {0, ..., n-1}."""
    for r in range(n + 1):
        for sub in itertools.combinations(range(n), r):
            yield set(sub)


# ---------- known-interaction scorers -----------------------------------------

def make_pin_scorer(n_pieces: int, attacker: int, pinned: int, target: int,
                    base_value: list[float]):
    """
    A scorer where:
      - Each piece has a baseline value contribution
      - The pin is worth a LOT when {attacker, pinned, target} are all present
        AND the *removal of pinned* destroys the tactic.
      - Concretely, the pin contributes +V if attacker present AND pinned present
        AND target present, where V is the pin's tactical value.

    True interaction structure:  triple-interaction on {a, p, t}.
    Pair projections:  H_{a,p}, H_{a,t}, H_{p,t}  will all be nonzero by symmetry
    of the triple term, BUT the dominant (chess-meaningful) pair is (a, p).
    """
    def phi(present: set[int]) -> float:
        score = sum(base_value[i] for i in present)
        if attacker in present and pinned in present and target in present:
            score += 10.0  # pin tactical bonus
        return score
    return phi


def make_fork_scorer(n_pieces: int, forker: int, target1: int, target2: int,
                     base_value: list[float]):
    """
    A scorer with a 'fork' interaction: forker present AND target1 present
    AND target2 present gives a tactical bonus.

    True interaction structure:  triple {f, t1, t2}.
    Dominant pairs: (f, t1) and (f, t2)  -- the forker pairs with each target.
    """
    def phi(present: set[int]) -> float:
        score = sum(base_value[i] for i in present)
        if forker in present and target1 in present and target2 in present:
            score += 10.0
        return score
    return phi


def make_neutral_scorer(n_pieces: int, base_value: list[float]):
    """A purely additive scorer: phi(S) = sum_{i in S} base[i].  H == 0 everywhere."""
    def phi(present: set[int]) -> float:
        return sum(base_value[i] for i in present)
    return phi


def make_near_puzzle_scorer(n_pieces: int, attacker: int, pinned: int, target: int,
                            defender: int, base_value: list[float]):
    """
    A 'near-puzzle': same tactical bonus as the pin scenario, BUT the bonus
    only fires if the defender is ABSENT.

    Concretely:  pin_bonus * [attacker in S and pinned in S and target in S
                              and defender NOT in S]

    True interaction structure:  4-way {a, p, t, ~d}.
    Dominant pairs in the position with all 4 present:
       (a, p), (a, t), (p, t) all have small H (because the bonus isn't active)
       (a, d), (p, d), (t, d) each carry the 'defender absorbs the threat' signature
    """
    def phi(present: set[int]) -> float:
        score = sum(base_value[i] for i in present)
        if (attacker in present and pinned in present and target in present
                and defender not in present):
            score += 10.0
        return score
    return phi


# ---------- the DHPE primitive itself -----------------------------------------

def dhpe_full(phi, all_pieces: set[int]) -> dict[tuple[int, int], float]:
    """
    Compute the full discrete Hessian H_{ij} for i < j  over the given piece set.

    H_{ij}(x) = phi(P) - phi(P\{i}) - phi(P\{j}) + phi(P\{i,j})

    where  P = all_pieces  is the 'full position'.
    """
    P = all_pieces
    H = {}
    full = phi(P)
    for i, j in itertools.combinations(sorted(P), 2):
        Pi = P - {i}
        Pj = P - {j}
        Pij = P - {i, j}
        H[(i, j)] = full - phi(Pi) - phi(Pj) + phi(Pij)
    return H


def hessian_l1(H: dict[tuple[int, int], float]) -> float:
    return sum(abs(v) for v in H.values())


def hessian_entropy(H: dict[tuple[int, int], float]) -> float:
    """Entropy of the |H_{ij}| distribution (over pairs). 0 -> concentrated; high -> diffuse."""
    s = sum(abs(v) for v in H.values())
    if s == 0:
        return 0.0
    e = 0.0
    for v in H.values():
        p = abs(v) / s
        if p > 0:
            e -= p * math.log(p)
    return e


def top_pairs(H: dict[tuple[int, int], float], k: int = 3):
    return sorted(H.items(), key=lambda kv: -abs(kv[1]))[:k]


# ---------- scenarios ---------------------------------------------------------

def run_scenarios():
    # 5 pieces, indices 0..4
    # Roles per scenario described in each block.
    n = 5
    base = [1.0, 1.0, 1.0, 1.0, 1.0]
    P = set(range(n))

    print("=" * 70)
    print("DHPE prototype: discrete Hessian over piece-existence indicators")
    print("=" * 70)

    # --- A. Pin scenario: pieces {attacker=0, pinned=1, target=2}, plus 2 noise pieces
    print("\n[A] PIN scenario")
    print("    pieces:  0=attacker  1=pinned  2=target  3,4=noise")
    phi_pin = make_pin_scorer(n, attacker=0, pinned=1, target=2, base_value=base)
    H_pin = dhpe_full(phi_pin, P)
    print(f"    L1(H)   = {hessian_l1(H_pin):.4f}")
    print(f"    entropy = {hessian_entropy(H_pin):.4f}  (lower = more concentrated)")
    for pair, val in top_pairs(H_pin):
        print(f"    H{pair} = {val:+.4f}")

    # --- B. Fork scenario: pieces {forker=0, target1=1, target2=2}, plus 2 noise
    print("\n[B] FORK scenario")
    print("    pieces:  0=forker  1=target1  2=target2  3,4=noise")
    phi_fork = make_fork_scorer(n, forker=0, target1=1, target2=2, base_value=base)
    H_fork = dhpe_full(phi_fork, P)
    print(f"    L1(H)   = {hessian_l1(H_fork):.4f}")
    print(f"    entropy = {hessian_entropy(H_fork):.4f}")
    for pair, val in top_pairs(H_fork):
        print(f"    H{pair} = {val:+.4f}")

    # --- C. Neutral / additive scenario
    print("\n[C] NEUTRAL / additive scenario (no interactions)")
    phi_neut = make_neutral_scorer(n, base_value=base)
    H_neut = dhpe_full(phi_neut, P)
    print(f"    L1(H)   = {hessian_l1(H_neut):.4f}")
    print(f"    entropy = {hessian_entropy(H_neut):.4f}")
    for pair, val in top_pairs(H_neut):
        print(f"    H{pair} = {val:+.4f}")

    # --- D. Near-puzzle: tactic fires only if defender ABSENT.
    #     With all pieces present, the static eval is the same as 'no tactic',
    #     but the *cross-derivative against the defender* is nontrivial.
    print("\n[D] NEAR-PUZZLE scenario")
    print("    pieces:  0=attacker  1=pinned  2=target  3=defender  4=noise")
    print("    (tactic fires only when defender absent)")
    phi_np = make_near_puzzle_scorer(n, attacker=0, pinned=1, target=2,
                                     defender=3, base_value=base)
    H_np = dhpe_full(phi_np, P)
    print(f"    L1(H)   = {hessian_l1(H_np):.4f}")
    print(f"    entropy = {hessian_entropy(H_np):.4f}")
    for pair, val in top_pairs(H_np, k=6):
        print(f"    H{pair} = {val:+.4f}")

    # --- Discriminator summary
    print("\n" + "=" * 70)
    print("DISCRIMINATOR SIGNATURE  (L1(H), entropy(H), max(|H|)):")
    print("=" * 70)
    rows = [
        ("PIN",        H_pin),
        ("FORK",       H_fork),
        ("NEUTRAL",    H_neut),
        ("NEAR-PUZZLE", H_np),
    ]
    max_pair_val = lambda H: max((abs(v) for v in H.values()), default=0.0)
    for name, H in rows:
        print(f"  {name:12s}  L1={hessian_l1(H):6.3f}  "
              f"entropy={hessian_entropy(H):.3f}  "
              f"max|H|={max_pair_val(H):6.3f}")

    # --- Key chess hypothesis: in the NEAR-puzzle, the top-|H| pair should
    #     involve the DEFENDER (piece 3), whereas in the PIN, the top pair
    #     should involve the ATTACKER, PINNED, or TARGET (pieces 0,1,2).
    print("\nKey hypothesis: top resonant pair INCLUDES defender (piece 3) "
          "for NEAR-PUZZLE only.")
    for name, H in rows:
        top = top_pairs(H, k=1)
        if top:
            pair, val = top[0]
            has_def = 3 in pair
            print(f"  {name:12s}  top pair = {pair}  |H|={abs(val):.3f}  "
                  f"contains_defender={has_def}")


if __name__ == "__main__":
    run_scenarios()
