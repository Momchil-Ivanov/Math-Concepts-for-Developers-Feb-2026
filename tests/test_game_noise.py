"""Tests for trembling-hand noise in Game."""

from ipd.strategies import AlwaysCooperate, AlwaysDefect
from ipd.tournament import Game


def test_noise_zero_preserves_intended_actions():
    g = Game(
        AlwaysCooperate(),
        AlwaysDefect(),
        rounds=30,
        noise=0.0,
        rng=12345,
    )
    r = g.play_match()
    assert r["history1"] == ["C"] * 30
    assert r["history2"] == ["D"] * 30


def test_noise_one_always_flips_execution():
    g = Game(
        AlwaysCooperate(),
        AlwaysDefect(),
        rounds=30,
        noise=1.0,
        rng=0,
    )
    r = g.play_match()
    assert r["history1"] == ["D"] * 30
    assert r["history2"] == ["C"] * 30
