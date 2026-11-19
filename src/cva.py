"""Credit valuation adjustments.

Default intensities come from a credit spread through the "credit triangle":
    lambda = spread / LGD     (flat hazard rate),   survival S(t) = exp(-lambda t).

Unilateral CVA (only the counterparty can default):
    CVA = LGD * sum_i  EE*(t_mid) [S(t_{i-1}) - S(t_i)],
with EE*(t_mid) the average of the discounted EE at both ends of the interval.
Bilateral (first-to-default) CVA and DVA multiply each term by the survival of the other party.
"""

import numpy as np
from scipy.optimize import brentq


def hazard_from_spread(spread, lgd):
    return spread / lgd


def survival(times, hazard):
    return np.exp(-hazard * np.asarray(times, dtype=float))


def _mid(profile):
    profile = np.asarray(profile)
    return 0.5 * (profile[1:] + profile[:-1])


def cva(times, ee_discounted, hazard, lgd, other_hazard=0.0):
    """Unilateral CVA (other_hazard = 0) or the CVA leg of first-to-default bilateral CVA."""
    s = survival(times, hazard)
    default_prob = s[:-1] - s[1:]
    other_survives = survival(0.5 * (times[1:] + times[:-1]), other_hazard)
    return float(lgd * np.sum(_mid(ee_discounted) * default_prob * other_survives))


def cva_per_path(times, discounted_positive_exposure, hazard, lgd):
    """CVA contribution of each simulated path (its mean is the CVA; used for standard errors)."""
    s = survival(times, hazard)
    weights = lgd * (s[:-1] - s[1:])
    exposure_mid = 0.5 * (discounted_positive_exposure[1:] + discounted_positive_exposure[:-1])
    return weights @ exposure_mid


def antithetic_standard_error(per_path):
    """Standard error of the mean when the second half of the paths mirrors the first half."""
    half = len(per_path) // 2
    pairs = 0.5 * (per_path[:half] + per_path[half:2 * half])
    return float(pairs.std(ddof=1) / np.sqrt(half))


def dva(times, ene_discounted, own_hazard, own_lgd, counterparty_hazard=0.0):
    return cva(times, ene_discounted, own_hazard, own_lgd, counterparty_hazard)


def cva_term_structure(times, ee_discounted, hazard, lgd):
    """Contribution of each interval to the CVA (for plots)."""
    s = survival(times, hazard)
    return lgd * _mid(ee_discounted) * (s[:-1] - s[1:])


# ---------------- wrong-way risk ----------------

def simulate_log_intensity(times, z_rate, rho, kappa, eta, seed=1):
    """Ornstein-Uhlenbeck factor Y driving the default intensity lambda = phi(t) exp(Y).

    The shocks of Y have correlation rho with the shocks z_rate that drive the short rate
    (exact OU transition on the grid). Returns Y with shape (n_times, n_paths).
    """
    rng = np.random.default_rng(seed)
    independent = rng.standard_normal(z_rate.shape)
    z = rho * z_rate + np.sqrt(1 - rho**2) * independent
    y = np.zeros((len(times), z_rate.shape[1]))
    for i in range(len(times) - 1):
        h = times[i + 1] - times[i]
        e = np.exp(-kappa * h)
        y[i + 1] = y[i] * e + eta * np.sqrt((1 - e**2) / (2 * kappa)) * z[i]
    return y


def calibrate_intensity(times, y, market_survival):
    """Piecewise-constant phi so that the average simulated survival matches the market curve.

    Integrated intensity over [t_i, t_{i+1}] = phi_i * h * (exp(Y_i) + exp(Y_{i+1})) / 2.
    Returns the path-wise survival matrix (n_times, n_paths) and phi.
    """
    growth = 0.5 * (np.exp(y[1:]) + np.exp(y[:-1])) * np.diff(times)[:, None]
    surv = np.ones((len(times), y.shape[1]))
    phi = np.zeros(len(times) - 1)
    for i in range(len(times) - 1):
        target = market_survival[i + 1]

        def gap(p):
            return np.mean(surv[i] * np.exp(-p * growth[i])) - target

        phi[i] = brentq(gap, 0.0, 10.0)
        surv[i + 1] = surv[i] * np.exp(-phi[i] * growth[i])
    return surv, phi


def cva_pathwise(times, discounted_positive_exposure, pathwise_survival, lgd):
    """CVA = LGD * E[ sum_i D V^+ (t_mid) * (S(t_{i-1}) - S(t_i)) ] with path-wise survival.

    With a deterministic survival curve this equals `cva` exactly.
    """
    exposure_mid = 0.5 * (discounted_positive_exposure[1:] + discounted_positive_exposure[:-1])
    default_prob = pathwise_survival[:-1] - pathwise_survival[1:]
    return float(lgd * np.mean(np.sum(exposure_mid * default_prob, axis=0)))
