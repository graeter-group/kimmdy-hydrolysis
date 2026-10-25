import numpy as np
import pytest
from kimmdy_hydrolysis.constants import K_b, T_EXPERIMENT
from kimmdy_hydrolysis.rates import (
    e_ts1,
    e_ts2,
    low_force_log_rate,
    high_force_log_rate,
    experimental_reaction_rate_per_s,
    theoretical_reaction_rate_per_s,
)


def test_ets1():
    assert e_ts1(0, 0, 0) == 0.0
    assert e_ts1(1, 1, 1) == 0.0
    assert e_ts1() == 80.0


def test_ets2():
    assert e_ts2(0, 0, 0) == 0.0
    assert e_ts2(1, 1, 1) == 0.0
    assert e_ts2() == 92.0


def test_low_force_rate():
    assert round(low_force_log_rate(0), 2) == -19.77
    assert round(low_force_log_rate(1), 2) == 6.49


def test_high_force_rate():
    assert high_force_log_rate(0, 0) == -20.342988
    assert high_force_log_rate(1, 1) == -18.667107
    assert round(high_force_log_rate(0), 7) == 0.4381212
    assert high_force_log_rate(1, 0) == -18.737755


def test_experimental_rate():
    assert experimental_reaction_rate_per_s(0, 0) == pytest.approx(
        2.594167969667349e-09
    )
    assert experimental_reaction_rate_per_s(1, 0) == pytest.approx(
        7.282766741703993e-09
    )
    assert round(experimental_reaction_rate_per_s(0.7), 6) == 1.090779


def test_theoretical_rate():
    assert theoretical_reaction_rate_per_s(0, 0, 0, 0, 0) == 5e9
    assert theoretical_reaction_rate_per_s(1, 1, 1, 1, 1) == 5e9
