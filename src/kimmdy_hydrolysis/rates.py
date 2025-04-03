import numpy as np
import logging
from kimmdy_hydrolysis.constants import K_b, T_EXPERIMENT

logger = logging.getLogger("kimmdy.hydrolysis")

def e_ts1(
    force: float = 0,
    ts1: float = 80,
    ts1_force_scaling: float = 1.67,
) -> float:
    return ts1 - ts1_force_scaling * force


def e_ts2(
    force: float = 0,
    ts2: float = 92,
    ts2_force_scaling: float = 25.83,
) -> float:
    return ts2 - ts2_force_scaling * force


def low_force_log_rate(force):
    log_slope = 26.26
    log_offset = -19.77
    return log_slope * force + log_offset


def high_force_log_rate(force, temperature: float = T_EXPERIMENT):
    log_slope_t = 0.070648
    log_slope_f = 1.605233
    log_offset = -20.342988
    log_k = log_offset + log_slope_t * temperature + log_slope_f * force

    return log_k


def experimental_reaction_rate_per_s(
    force: float, temperature: float = T_EXPERIMENT
) -> float:
    critical_force = 0.7
    interpolation_width = 0.05

    if force <= (critical_force - interpolation_width):
        log_k = low_force_log_rate(force)
    elif force > (critical_force + interpolation_width):
        log_k = high_force_log_rate(force, temperature)
    else:
        # linear interpolation between the two log-linear regimes
        low = low_force_log_rate(force)
        high = high_force_log_rate(force)
        high_percentage = (force - (critical_force - interpolation_width)) / (
            2 * interpolation_width
        )
        low_percentage = 1 - high_percentage
        log_k = low_percentage * low + high_percentage * high

    k = np.exp(log_k)
    return k


def theoretical_reaction_rate_per_s(
    force: float = 0,
    ts1: float = 80,
    ts2: float = 92,
    ts1_force_scaling: float = 1.67,
    ts2_force_scaling: float = 25.83,
    A: float = 1e11,  # 1/s
    temperature: float = 300,
    ph_value: float = 7.4,
) -> float:
    """Calculate reaction rate in 1/s

    see SI of pill et al. 2019
    <http://dx.doi.org/10.1002/anie.201902752>
    """
    # energy barriers in kJ/mol
    # high force regime, TS1 is rate-determining
    E_ts1 = e_ts1(force, ts1, ts1_force_scaling)
    # low force regime, TS2 is rate-determining
    E_ts2 = e_ts2(force, ts2, ts2_force_scaling)

    # concentration of OH-
    c_oh = 10 ** (-(14 - ph_value))
    c_oh_experiment = 10 ** (-(14 - 7.4))

    A1 = A / c_oh_experiment / 10
    k1 = A1 * np.exp(-E_ts1 / (K_b * temperature))
    k2 = A1 * np.exp(-E_ts2 / (K_b * temperature))  # k2' in the paper
    # reaction rate in 1/s (depending on how A is chosen)
    k_hyd = (k1 * k2 * c_oh) / (k1 + k2)

    logger.debug(f"TS1: {E_ts1} TS2: {E_ts2} Force: {force} k_hyd: {k_hyd}")

    return k_hyd

