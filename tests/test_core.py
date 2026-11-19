"""Consistency tests: curve, Hull-White, swap valuation, exposure, CVA, collateral, wrong-way risk.

Run:  python -m pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.curve import DiscountCurve, bootstrap_treasury_curve, par_bond_price  # noqa: E402
from src.cva import (antithetic_standard_error, calibrate_intensity, cva, cva_pathwise,  # noqa: E402
                     cva_per_path, simulate_log_intensity, survival)
from src.exposure import (collateralised_values, effective_epe, expected_positive_exposure,  # noqa: E402
                          exposure_profiles, simulate_values, time_grid)
from src.hull_white import HullWhite, calibrate_to_yield_vols, model_zero_yield_vol  # noqa: E402
from src.swap import Swap, par_swap_rate  # noqa: E402

MATS = [1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
PARS = [0.041, 0.043, 0.044, 0.046, 0.048, 0.0484, 0.049, 0.0499, 0.051, 0.055, 0.0546]


@pytest.fixture(scope="module")
def curve():
    return bootstrap_treasury_curve(MATS, PARS)


@pytest.fixture(scope="module")
def model(curve):
    return HullWhite(0.03, 0.009, curve)


@pytest.fixture(scope="module")
def simulation(model, curve):
    swap = Swap(1e8, par_swap_rate(curve, 0, 10), 0, 10)
    times = time_grid(10, 26)
    sim = simulate_values(model, [swap], times, 20_000, seed=7)
    return swap, times, sim


# ---------------- curve ----------------
def test_bootstrap_reprices_par_bonds(curve):
    for m, y in zip(MATS, PARS):
        if m >= 1:
            assert par_bond_price(curve, m, y) == pytest.approx(1.0, abs=1e-12)


def test_bills_are_semiannual_zero_rates(curve):
    assert curve.df(0.25) == pytest.approx((1 + 0.043 / 2) ** -0.5, rel=1e-12)


def test_flat_curve_forward_equals_zero():
    flat = DiscountCurve([1, 5, 30], np.exp(-0.04 * np.array([1, 5, 30])))
    assert flat.forward_rate(7.3) == pytest.approx(0.04, abs=1e-8)
    assert flat.zero_rate(40.0) == pytest.approx(0.04, abs=1e-10)  # flat extrapolation


# ---------------- Hull-White ----------------
def test_bond_price_at_time_zero_matches_curve(model, curve):
    T = np.array([0.5, 3.0, 10.0])
    assert np.allclose(model.bond_price(0.0, T, 0.0), curve.df(T), rtol=1e-12)


def test_deflator_reprices_discount_factors(simulation, curve):
    _, times, sim = simulation
    ratio = sim["deflator"].mean(axis=1) / curve.df(times)
    assert np.abs(ratio - 1).max() < 2e-3


def test_simulated_x_has_model_variance(model):
    times = np.array([0.0, 5.0])
    x = model.simulate(times, 40_000, seed=3)["x"][-1]
    theory = model.sigma**2 / (2 * model.a) * (1 - np.exp(-2 * model.a * 5.0))
    assert x.var() == pytest.approx(theory, rel=0.03)


def test_zcb_put_call_parity(model, curve):
    T, S, K = 2.0, 7.0, 0.85
    call = model.zcb_option(T, S, K, "call")
    put = model.zcb_option(T, S, K, "put")
    assert call - put == pytest.approx(curve.df(S) - K * curve.df(T), abs=1e-12)


def test_swaption_parity(model, curve):
    """Payer - receiver swaption = forward-starting payer swap (same strike)."""
    T, K = 3.0, 0.05
    pay = np.arange(3.5, 10.0 + 1e-9, 0.5)
    payer = model.swaption(T, pay, K, payer=True)
    receiver = model.swaption(T, pay, K, payer=False)
    forward_swap = Swap(1.0, K, 3.0, 10.0).value(curve)
    assert payer - receiver == pytest.approx(forward_swap, abs=1e-10)


def test_calibration_recovers_parameters():
    tenors = np.array([2, 3, 5, 7, 10, 20, 30.0])
    vols = model_zero_yield_vol(0.05, 0.01, tenors)
    a, sigma, rmse = calibrate_to_yield_vols(tenors, vols)
    assert a == pytest.approx(0.05, rel=1e-4) and sigma == pytest.approx(0.01, rel=1e-4)
    assert rmse < 1e-8


# ---------------- swap and exposure ----------------
def test_par_swap_is_worth_zero(curve, simulation):
    swap, _, sim = simulation
    assert swap.value(curve) == pytest.approx(0.0, abs=1e-6)
    assert np.abs(sim["values"][0][0]).max() < 1e-6  # same on every path at t = 0


def test_discounted_value_is_a_martingale(simulation, curve):
    """E[D(t) V(t)] = value today of the swap that remains after t (at reset dates)."""
    swap, times, sim = simulation
    for T in [2.0, 5.0, 8.0]:
        i = int(np.argmin(np.abs(times - T)))
        mc = (sim["deflator"][i] * sim["values"][0][i]).mean()
        exact = Swap(swap.notional, swap.fixed_rate, T, 10.0).value(curve)
        assert mc == pytest.approx(exact, abs=0.03 * swap.notional * 0.01)  # within 3 bp of notional


def test_expected_exposure_matches_swaption(simulation, model):
    """EE*(T) of a swap at a reset date = Jamshidian payer swaption price (within 3 s.e.)."""
    swap, times, sim = simulation
    for T in [1.0, 4.0, 7.0]:
        i = int(np.argmin(np.abs(times - T)))
        disc = sim["deflator"][i] * np.maximum(sim["values"][0][i], 0)
        se = antithetic_standard_error(disc)
        closed = model.swaption(T, swap.pay_times[swap.pay_times > T + 1e-9], swap.fixed_rate, swap.notional)
        assert abs(disc.mean() - closed) < 3 * se


def test_reset_dates_must_be_on_grid(model, curve):
    swap = Swap(1e6, 0.05, 0, 2)
    with pytest.raises(ValueError):
        simulate_values(model, [swap], np.linspace(0, 2, 7), 100)


def test_epe_definitions():
    t = np.linspace(0, 1, 11)
    ee = np.r_[np.linspace(0, 10, 6), np.linspace(8, 0, 5)]
    assert expected_positive_exposure(t, np.full(11, 3.0)) == pytest.approx(3.0)
    assert effective_epe(t, ee) >= expected_positive_exposure(t, ee)


# ---------------- CVA ----------------
def test_cva_zero_without_default_risk_and_linear_in_lgd(simulation):
    swap, times, sim = simulation
    ee = exposure_profiles(sim["values"][0], sim["deflator"])["EE_discounted"]
    assert cva(times, ee, 0.0, 0.6) == 0.0
    assert cva(times, ee, 0.02, 0.6) == pytest.approx(1.5 * cva(times, ee, 0.02, 0.4))


def test_cva_constant_exposure_closed_form():
    times = np.linspace(0, 5, 501)
    ee = np.full_like(times, 1e6)
    assert cva(times, ee, 0.02, 0.6) == pytest.approx(0.6 * 1e6 * (1 - np.exp(-0.1)), rel=1e-12)


def test_per_path_cva_averages_to_cva(simulation):
    swap, times, sim = simulation
    disc_pos = sim["deflator"] * np.maximum(sim["values"][0], 0)
    ee = disc_pos.mean(axis=1)
    assert cva_per_path(times, disc_pos, 0.02, 0.6).mean() == pytest.approx(cva(times, ee, 0.02, 0.6), rel=1e-10)


# ---------------- collateral and netting ----------------
def test_full_daily_collateral_removes_exposure(simulation):
    _, _, sim = simulation
    v = sim["values"][0]
    assert np.abs(collateralised_values(v, 0.0, 0)).max() < 1e-9
    assert np.array_equal(collateralised_values(v, 1e18, 1), v)  # huge threshold = no collateral


def test_collateral_reduces_exposure(simulation):
    _, _, sim = simulation
    v = sim["values"][0]
    ee_no = exposure_profiles(v, sim["deflator"])["EE"]
    ee_csa = exposure_profiles(collateralised_values(v, 5e6, 1), sim["deflator"])["EE"]
    assert ee_csa.sum() < ee_no.sum()


def test_netting_never_increases_exposure(simulation):
    _, _, sim = simulation
    v1 = sim["values"][0]
    v2 = -0.5 * v1 + 1e6 * np.sin(np.arange(v1.shape[1]))[None, :]
    netted = np.maximum(v1 + v2, 0).mean(axis=1)
    standalone = np.maximum(v1, 0).mean(axis=1) + np.maximum(v2, 0).mean(axis=1)
    assert np.all(netted <= standalone + 1e-6)


# ---------------- wrong-way risk ----------------
def test_intensity_calibration_matches_market_survival(simulation):
    _, times, sim = simulation
    y = simulate_log_intensity(times, sim["z_rate"], 0.5, 0.1, 0.5, seed=1)
    market = survival(times, 0.02)
    surv, phi = calibrate_intensity(times, y, market)
    assert np.abs(surv.mean(axis=1) - market).max() < 1e-8
    assert np.all(phi > 0)


def test_pathwise_cva_with_deterministic_survival_equals_cva(simulation):
    _, times, sim = simulation
    disc_pos = sim["deflator"] * np.maximum(sim["values"][0], 0)
    s = np.repeat(survival(times, 0.02)[:, None], disc_pos.shape[1], axis=1)
    assert cva_pathwise(times, disc_pos, s, 0.6) == pytest.approx(cva(times, disc_pos.mean(axis=1), 0.02, 0.6), rel=1e-10)


def test_wrong_way_risk_raises_payer_cva(simulation):
    _, times, sim = simulation
    disc_pos = sim["deflator"] * np.maximum(sim["values"][0], 0)
    market = survival(times, 0.02)
    results = {}
    for rho in (-0.6, 0.6):
        y = simulate_log_intensity(times, sim["z_rate"], rho, 0.1, 0.5, seed=1)
        surv, _ = calibrate_intensity(times, y, market)
        results[rho] = cva_pathwise(times, disc_pos, surv, 0.6)
    assert results[0.6] > 1.2 * results[-0.6]
