"""
Symmetric Prisoner's Dilemma payoff parameters (T, R, P, S) and validation.

Requires T > R > P > S and 2R > T + S for a strict PD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class PayoffParams:
    """
    Four scalars (T, R, P, S) for a symmetric two-player Prisoner's Dilemma.

    Mutual cooperation gives R to each player; mutual defection gives P.
    When strategies differ, the defector earns T and the cooperator earns S.
    """

    T: float
    R: float
    P: float
    S: float

    def __post_init__(self) -> None:
        validate_prisoners_dilemma(self.T, self.R, self.P, self.S)

    def to_pair_payoffs(self) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """Payoffs for (player1_action, player2_action) → (payoff1, payoff2)."""
        return {
            ("C", "C"): (self.R, self.R),
            ("C", "D"): (self.S, self.T),
            ("D", "C"): (self.T, self.S),
            ("D", "D"): (self.P, self.P),
        }

    def my_action_payoffs(self) -> Dict[Tuple[str, str], float]:
        """
        Map each (my move, opponent move) pair to my payoff only.
        """
        return {
            ("C", "C"): self.R,
            ("C", "D"): self.S,
            ("D", "C"): self.T,
            ("D", "D"): self.P,
        }


def validate_prisoners_dilemma(T: float, R: float, P: float, S: float) -> None:
    if not (T > R > P > S):
        raise ValueError(f"Require T > R > P > S; got T={T}, R={R}, P={P}, S={S}")
    if not (2 * R > T + S):
        raise ValueError(
            f"Require 2R > T + S; got 2R={2 * R}, T+S={T + S}"
        )


DEFAULT_PAYOFFS = PayoffParams(T=5.0, R=3.0, P=1.0, S=0.0)
