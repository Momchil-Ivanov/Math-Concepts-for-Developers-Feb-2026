"""Iterated Prisoner's Dilemma: strategies, tournaments, and simulations."""

from .payoffs import DEFAULT_PAYOFFS, PayoffParams, validate_prisoners_dilemma
from .strategies import Strategy, get_all_strategies
from .tournament import EvolutionarySimulation, Game, Tournament

__all__ = [
    "DEFAULT_PAYOFFS",
    "EvolutionarySimulation",
    "Game",
    "PayoffParams",
    "Strategy",
    "Tournament",
    "get_all_strategies",
    "validate_prisoners_dilemma",
]
