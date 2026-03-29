"""Tests for PD payoff parameters and validation."""

import pytest

from ipd.payoffs import DEFAULT_PAYOFFS, PayoffParams, validate_prisoners_dilemma


def test_default_payoffs_satisfy_pd_constraints():
    p = DEFAULT_PAYOFFS
    assert p.T > p.R > p.P > p.S
    assert 2 * p.R > p.T + p.S


def test_to_pair_payoffs_matches_standard_matrix():
    p = DEFAULT_PAYOFFS
    m = p.to_pair_payoffs()
    assert m[("C", "C")] == (p.R, p.R)
    assert m[("C", "D")] == (p.S, p.T)
    assert m[("D", "C")] == (p.T, p.S)
    assert m[("D", "D")] == (p.P, p.P)


def test_my_action_payoffs():
    p = DEFAULT_PAYOFFS
    m = p.my_action_payoffs()
    assert m[("C", "C")] == p.R
    assert m[("D", "C")] == p.T


def test_validate_accepts_valid_custom_params():
    validate_prisoners_dilemma(4.0, 3.0, 1.0, 0.0)


def test_constructor_rejects_invalid_ordering():
    with pytest.raises(ValueError, match="T > R > P > S"):
        PayoffParams(T=2.0, R=3.0, P=1.0, S=0.0)


def test_constructor_rejects_fails_2r_t_plus_s():
    with pytest.raises(ValueError, match="2R"):
        PayoffParams(T=10.0, R=3.0, P=1.0, S=0.0)
