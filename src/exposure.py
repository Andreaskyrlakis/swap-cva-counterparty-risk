"""Exposure simulation: value a netting set of swaps on every date of a time grid.

Exposure measures (all per date of the grid):
- EE(t)   = E[max(V(t), 0)]                 expected exposure (undiscounted)
- EE*(t)  = E[D(t) max(V(t), 0)]            discounted expected exposure (used in CVA)
- ENE*(t) = E[D(t) max(-V(t), 0)]           discounted expected negative exposure (used in DVA)
- PFE(t)  = 97.5% quantile of max(V(t), 0)  potential future exposure
- EPE     = time average of EE over a horizon
"""

import numpy as np


def time_grid(maturity, steps_per_year=26):
    """Uniform grid. 26 steps a year = two weeks = about 10 business days (one margin period)."""
    n = int(round(maturity * steps_per_year))
    return np.linspace(0.0, maturity, n + 1)


def simulate_values(model, trades, times, n_paths, seed=0):
    """Simulate the short rate and value every trade at every grid date.

    Returns the simulation dict plus `values`: array (n_trades, n_times, n_paths).
    Every reset date of every trade must be on the grid.
    """
    sim = model.simulate(times, n_paths, seed=seed)
    x = sim["x"]
    for trade in trades:
        for r in trade.reset_times():
            if not np.any(np.isclose(times, r)):
                raise ValueError(f"reset date {r} is not on the simulation grid")
    values = np.zeros((len(trades), len(times), x.shape[1]))
    fixings = [dict() for _ in trades]
    for i, t in enumerate(times):
        for k, trade in enumerate(trades):
            # store the floating-rate fixing of any period starting today
            for r in trade.reset_times():
                if abs(r - t) < 1e-8:
                    fixings[k][round(r, 8)] = model.bond_price(t, r + trade.accrual, x[i])
            values[k, i] = trade.value_on_paths(model, t, x[i], fixings[k])
    sim["values"] = values
    return sim


def collateralised_values(values, threshold, mpr_steps, two_way=True):
    """Netting-set value after collateral under a CSA with a threshold and a margin period of risk.

    Collateral held at t is the amount called at t - MPR: (V - H)^+ (and -(-V - H)^+ posted by
    us if the CSA is two-way). The exposure is then (V(t) - C(t))^+.
    """
    lagged = np.empty_like(values)
    if mpr_steps > 0:
        lagged[mpr_steps:] = values[:-mpr_steps]
        lagged[:mpr_steps] = values[0]
    else:
        lagged[:] = values
    collateral = np.maximum(lagged - threshold, 0.0)
    if two_way:
        collateral -= np.maximum(-lagged - threshold, 0.0)
    return values - collateral


def exposure_profiles(values, deflator, quantile=0.975):
    """Exposure profiles of a netting-set value matrix (n_times x n_paths)."""
    positive = np.maximum(values, 0.0)
    negative = np.maximum(-values, 0.0)
    return {
        "EE": positive.mean(axis=1),
        "EE_discounted": (deflator * positive).mean(axis=1),
        "ENE_discounted": (deflator * negative).mean(axis=1),
        "PFE": np.quantile(positive, quantile, axis=1),
        "mean_value_discounted": (deflator * values).mean(axis=1),
    }


def expected_positive_exposure(times, ee, horizon=None):
    """EPE = time-weighted average of EE over [0, horizon] (trapezoid rule)."""
    if horizon is not None:
        keep = times <= horizon + 1e-9
        times, ee = times[keep], ee[keep]
    return float(np.trapezoid(ee, times) / (times[-1] - times[0]))


def effective_epe(times, ee, horizon=1.0):
    """Basel effective EPE: average of the non-decreasing envelope of EE over the first year."""
    keep = times <= horizon + 1e-9
    return expected_positive_exposure(times[keep], np.maximum.accumulate(ee[keep]))
