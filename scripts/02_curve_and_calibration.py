"""Bootstrap the zero curve and calibrate the Hull-White parameters (a, sigma).

The mean reversion a and volatility sigma are fitted to the HISTORICAL volatility of zero
yields (last 3 years of daily changes), because swaption implied volatilities, which a desk
would use, are not available for free. In Hull-White the normal volatility of the tau-year
zero yield is sigma * B(tau) / tau, so the shape across tenors identifies a.

Outputs: results/calibration.json, figures/fig1_curve.png, figures/fig2_yield_vol_fit.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.curve import bootstrap_treasury_curve, par_bond_price  # noqa: E402
from src.data import CURVE_SERIES, curve_history, load_meta, valuation_inputs  # noqa: E402
from src.hull_white import calibrate_to_yield_vols, model_zero_yield_vol  # noqa: E402
from src.market import MATURITY, LGD, OWN_LGD, save_json  # noqa: E402
from src.plotting import BLUE, ORANGE, GREEN, GREY, plt, save  # noqa: E402
from src.swap import par_swap_rate  # noqa: E402

HISTORY_YEARS = 3
VOL_TENORS = np.array([1, 2, 3, 5, 7, 10, 20, 30.0])
FIT_TENORS = VOL_TENORS >= 2  # the 1-year point sits below a hump that one factor cannot fit

if __name__ == "__main__":
    inputs = valuation_inputs()
    curve = bootstrap_treasury_curve(inputs["maturities"], inputs["par_yields"])
    print(f"valuation date {inputs['date']}")

    # --- check: the curve reprices every par bond (1y and above) at 100 ---
    repricing = {m: par_bond_price(curve, m, y) - 1 for m, y in
                 zip(inputs["maturities"], inputs["par_yields"]) if m >= 1}
    max_repricing_error = max(abs(e) for e in repricing.values())
    print(f"max par-bond repricing error: {max_repricing_error:.2e}")

    # --- historical zero-yield volatilities ---
    history = curve_history()
    start = history.index.max() - pd.DateOffset(years=HISTORY_YEARS)
    history = history[history.index > start]
    maturities = list(CURVE_SERIES.values())
    zeros = np.array([bootstrap_treasury_curve(maturities, row.values).zero_rate(VOL_TENORS)
                      for _, row in history.iterrows()])
    daily_changes = np.diff(zeros, axis=0)
    observed_vol = daily_changes.std(axis=0, ddof=1) * np.sqrt(252)
    a, sigma, rmse = calibrate_to_yield_vols(VOL_TENORS[FIT_TENORS], observed_vol[FIT_TENORS])
    fitted_vol = model_zero_yield_vol(a, sigma, VOL_TENORS)
    print(f"Hull-White a = {a:.4f}, sigma = {sigma * 1e4:.1f} bp, fit RMSE = {rmse * 1e4:.2f} bp")

    par10 = par_swap_rate(curve, 0, MATURITY)
    spreads = inputs["spreads"]
    payload = {
        "valuation_date": inputs["date"],
        "data": load_meta(),
        "par_yields": dict(zip([str(round(m, 4)) for m in maturities], inputs["par_yields"])),
        "sofr_overnight": inputs["sofr"],
        "zero_rates_cc": {str(t): float(curve.zero_rate(t)) for t in [0.25, 1, 2, 5, 10, 20, 30]},
        "max_par_bond_repricing_error": max_repricing_error,
        "par_swap_rate_10y": par10,
        "history": {"first_day": str(history.index.min().date()), "last_day": str(history.index.max().date()),
                    "n_days": len(history)},
        "zero_yield_vol_bp": dict(zip([str(t) for t in VOL_TENORS], observed_vol * 1e4)),
        "fitted_zero_yield_vol_bp": dict(zip([str(t) for t in VOL_TENORS], fitted_vol * 1e4)),
        "hull_white": {"a": a, "sigma": sigma, "fit_rmse_bp": rmse * 1e4,
                       "fit_tenors": VOL_TENORS[FIT_TENORS].tolist()},
        "credit": {
            "counterparty_spread_BBB_OAS": spreads["counterparty_BBB"],
            "bank_spread_A_OAS": spreads["bank_A"],
            "lgd": LGD, "own_lgd": OWN_LGD,
            "hazard_counterparty": spreads["counterparty_BBB"] / LGD,
            "hazard_bank": spreads["bank_A"] / OWN_LGD,
            "pd_10y_counterparty": 1 - np.exp(-spreads["counterparty_BBB"] / LGD * MATURITY),
            "pd_10y_bank": 1 - np.exp(-spreads["bank_A"] / OWN_LGD * MATURITY),
        },
    }
    save_json("calibration.json", payload)

    # --- figure 1: the curve ---
    t = np.linspace(0.05, 30, 600)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(inputs["maturities"], np.array(inputs["par_yields"]) * 100, "o", color=GREY, label="Treasury par yields (FRED)")
    ax.plot(t, curve.zero_rate(t) * 100, color=BLUE, label="bootstrapped zero rate (cont. comp.)")
    ax.plot(t, curve.forward_rate(t) * 100, color=ORANGE, lw=1, label="instantaneous forward rate")
    ax.axvline(MATURITY, color=GREY, ls=":", lw=1)
    ax.set_xlabel("maturity (years)")
    ax.set_ylabel("rate (%)")
    ax.set_title(f"US Treasury curve on {inputs['date']}")
    ax.legend()
    save(fig, "fig1_curve.png")

    # --- figure 2: volatility fit ---
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(VOL_TENORS, observed_vol * 1e4, "o-", color=BLUE,
            label=f"historical vol of zero yields ({history.index.min():%b %Y}–{history.index.max():%b %Y})")
    grid = np.linspace(0.5, 30, 200)
    ax.plot(grid, model_zero_yield_vol(a, sigma, grid) * 1e4, color=ORANGE,
            label=f"Hull-White fit: a = {a:.3f}, σ = {sigma * 1e4:.0f} bp")
    ax.set_xlabel("zero-coupon maturity (years)")
    ax.set_ylabel("normal volatility (bp / year)")
    ax.set_ylim(0, None)
    ax.legend(loc="lower right")
    save(fig, "fig2_yield_vol_fit.png")
