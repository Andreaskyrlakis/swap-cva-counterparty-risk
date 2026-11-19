"""Wrong-way risk: default intensity correlated with interest rates.

The counterparty's default intensity is lambda(t) = phi(t) exp(Y(t)), with Y an
Ornstein-Uhlenbeck factor whose shocks have correlation rho with the shocks of the short
rate. phi(t) is recalibrated for every rho so that the average survival curve stays equal
to the market curve from the BBB spread: only the dependence between exposure and default
changes, not the default probabilities. The same random numbers are used for every rho.

For a payer swap, exposure is high when rates are high: rho > 0 is wrong-way risk.
For a receiver swap the same rho > 0 is right-way risk.

Outputs: results/wrong_way_risk.json, figures/fig8_wrong_way_risk.png
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cva import calibrate_intensity, cva, cva_pathwise, simulate_log_intensity, survival  # noqa: E402
from src.exposure import exposure_profiles, simulate_values, time_grid  # noqa: E402
from src.market import (LGD, MATURITY, N_PATHS, SEED, STEPS_PER_YEAR, WWR_ETA, WWR_KAPPA,  # noqa: E402
                        base_swap, load_market, save_json)
from src.plotting import BLUE, GREY, ORANGE, RED, plt, save  # noqa: E402

RHOS = [-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9]
ETAS = [0.25, WWR_ETA, 1.0]

if __name__ == "__main__":
    market = load_market()
    model, lam = market["model"], market["hazard_cpty"]
    times = time_grid(MATURITY, STEPS_PER_YEAR)
    swap = base_swap(market)
    sim = simulate_values(model, [swap], times, N_PATHS, seed=SEED)
    deflator, values = sim["deflator"], sim["values"][0]
    payer_exposure = deflator * np.maximum(values, 0)
    receiver_exposure = deflator * np.maximum(-values, 0)
    market_survival = survival(times, lam)

    independent = {
        "payer": cva(times, exposure_profiles(values, deflator)["EE_discounted"], lam, LGD),
        "receiver": cva(times, exposure_profiles(-values, deflator)["EE_discounted"], lam, LGD),
    }
    rows = []
    for eta in ETAS:
        for rho in RHOS:
            y = simulate_log_intensity(times, sim["z_rate"], rho, WWR_KAPPA, eta, seed=SEED + 1)
            surv, _ = calibrate_intensity(times, y, market_survival)
            max_survival_gap = float(np.abs(surv.mean(axis=1) - market_survival).max())
            payer = cva_pathwise(times, payer_exposure, surv, LGD)
            receiver = cva_pathwise(times, receiver_exposure, surv, LGD)
            rows.append({"eta": eta, "rho": rho, "cva_payer": payer, "cva_receiver": receiver,
                         "payer_vs_independent": payer / independent["payer"],
                         "receiver_vs_independent": receiver / independent["receiver"],
                         "max_survival_calibration_gap": max_survival_gap})
            print(f"eta {eta:.2f} rho {rho:+.1f}: payer CVA {payer:>9,.0f} ({payer / independent['payer']:.2f}x), "
                  f"receiver {receiver:>9,.0f} ({receiver / independent['receiver']:.2f}x)")

    save_json("wrong_way_risk.json", {
        "model": "lambda = phi(t) exp(Y), dY = -kappa Y dt + eta dW_Y, corr(dW_Y, dW_r) = rho",
        "kappa": WWR_KAPPA, "eta_base": WWR_ETA, "independent_cva": independent, "rows": rows})

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, side, title in [(axes[0], "payer", "10y payer swap"), (axes[1], "receiver", "10y receiver swap")]:
        for eta, colour in zip(ETAS, [GREY, BLUE, RED]):
            sel = [r for r in rows if r["eta"] == eta]
            ax.plot([r["rho"] for r in sel], [r[f"{side}_vs_independent"] for r in sel], "o-", color=colour,
                    label=f"intensity vol η = {eta:.0%}")
        ax.axhline(1, color=GREY, ls=":", lw=1)
        ax.axvline(0, color=GREY, ls=":", lw=1)
        ax.set_xlabel("correlation ρ between default intensity and rates")
        ax.set_title(title)
    axes[0].set_ylabel("CVA / CVA with independence")
    axes[0].legend(fontsize=8)
    save(fig, "fig8_wrong_way_risk.png")
