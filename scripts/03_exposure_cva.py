"""Exposure profiles and CVA of a 10-year payer swap, with the Monte Carlo checks.

1. Simulate Hull-White paths (exact scheme, two-week grid) and value the swap on every date.
2. EE, PFE 97.5%, ENE, EPE, effective EPE.
3. Unilateral CVA, DVA, bilateral (first-to-default) CVA - DVA.
4. Checks: E[D(t)] = P(0, t); the swap is worth 0 at t = 0; E[D(t) V(t)] equals the
   forward value of the remaining swap; EE*(T_k) equals the Jamshidian swaption price at
   each reset date; convergence of the CVA with the number of paths.
5. Sensitivities: rate volatility, credit spread, payer vs receiver.

Outputs: results/exposure_cva.json, figures/fig3_exposure.png, fig4_swaption_check.png,
fig5_convergence.png
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cva import (antithetic_standard_error, cva, cva_per_path, cva_term_structure,  # noqa: E402
                     dva)
from src.exposure import (effective_epe, expected_positive_exposure, exposure_profiles,  # noqa: E402
                          simulate_values, time_grid)
from src.hull_white import HullWhite  # noqa: E402
from src.market import (LGD, MATURITY, N_PATHS, NOTIONAL, OWN_LGD, QUANTILE, SEED,  # noqa: E402
                        STEPS_PER_YEAR, base_swap, load_market, save_json)
from src.plotting import BLUE, GREY, ORANGE, RED, GREEN, millions, plt, save  # noqa: E402
from src.swap import Swap  # noqa: E402

CONVERGENCE_PATHS = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
BATCH = 20_000  # keep memory low: large runs are split into batches


def cva_estimate(model, trade, times, n_paths, hazard, seed):
    """CVA and its standard error from n_paths (in batches of at most BATCH paths)."""
    per_path, errors = [], []
    remaining, batch_seed = n_paths, seed
    while remaining > 0:
        n = min(BATCH, remaining)
        sim = simulate_values(model, [trade], times, n, seed=batch_seed)
        contrib = cva_per_path(times, sim["deflator"] * np.maximum(sim["values"][0], 0), hazard, LGD)
        per_path.append(contrib)
        errors.append(antithetic_standard_error(contrib) ** 2 * n**2)
        remaining -= n
        batch_seed += 1000
    estimate = float(np.concatenate(per_path).mean())
    return estimate, float(np.sqrt(sum(errors)) / n_paths)


if __name__ == "__main__":
    market = load_market()
    curve, model = market["curve"], market["model"]
    lam_c, lam_b = market["hazard_cpty"], market["hazard_own"]
    swap = base_swap(market)
    times = time_grid(MATURITY, STEPS_PER_YEAR)
    print(f"10y par swap rate {swap.fixed_rate:.4%}, value at t=0 on the curve {swap.value(curve):.2e}")

    sim = simulate_values(model, [swap], times, N_PATHS, seed=SEED)
    values, deflator = sim["values"][0], sim["deflator"]
    prof = exposure_profiles(values, deflator, QUANTILE)

    # ---------------- CVA / DVA ----------------
    per_path = cva_per_path(times, deflator * np.maximum(values, 0), lam_c, LGD)
    cva_uni = cva(times, prof["EE_discounted"], lam_c, LGD)
    cva_se = antithetic_standard_error(per_path)
    dva_uni = dva(times, prof["ENE_discounted"], lam_b, OWN_LGD)
    cva_bil = cva(times, prof["EE_discounted"], lam_c, LGD, other_hazard=lam_b)
    dva_bil = dva(times, prof["ENE_discounted"], lam_b, OWN_LGD, counterparty_hazard=lam_c)
    annuity = swap.annuity(curve) * NOTIONAL
    print(f"CVA {cva_uni:,.0f} USD (s.e. {cva_se:,.0f}), DVA {dva_uni:,.0f}, BCVA {cva_bil - dva_bil:,.0f}")

    epe_life = expected_positive_exposure(times, prof["EE"])
    epe_1y = expected_positive_exposure(times, prof["EE"], horizon=1.0)
    eepe_1y = effective_epe(times, prof["EE"], horizon=1.0)
    peak = int(np.argmax(prof["EE"]))
    peak_pfe = int(np.argmax(prof["PFE"]))

    # ---------------- checks ----------------
    deflator_ratio = deflator.mean(axis=1) / curve.df(times)
    swaption_rows = []
    for T in swap.reset_times()[1:]:
        i = int(np.argmin(np.abs(times - T)))
        pay = swap.pay_times[swap.pay_times > T + 1e-9]
        disc_pos = deflator[i] * np.maximum(values[i], 0)
        half = N_PATHS // 2
        se = np.std(0.5 * (disc_pos[:half] + disc_pos[half:]), ddof=1) / np.sqrt(half)
        closed = model.swaption(T, pay, swap.fixed_rate, NOTIONAL, payer=True)
        forward_value = Swap(NOTIONAL, swap.fixed_rate, T, MATURITY).value(curve)
        swaption_rows.append({
            "expiry": float(T), "mc_ee_discounted": float(prof["EE_discounted"][i]), "mc_se": float(se),
            "jamshidian": float(closed), "z_score": float((prof["EE_discounted"][i] - closed) / se),
            "mc_mean_value_discounted": float(prof["mean_value_discounted"][i]),
            "forward_swap_value": float(forward_value)})
    z_scores = np.array([r["z_score"] for r in swaption_rows])
    rel_err = np.array([abs(r["mc_ee_discounted"] / r["jamshidian"] - 1) for r in swaption_rows])
    print(f"swaption check: max |z| = {np.abs(z_scores).max():.2f}, max rel. error = {rel_err.max():.2%}")

    # CVA with EE* replaced by the closed form at reset dates (semi-annual grid) vs MC on that grid
    reset_grid = np.r_[0.0, swap.reset_times()[1:], MATURITY]
    # at t = 0 the swap is at par (exposure 0); after the last payment it is gone (exposure 0)
    closed_ee = np.r_[0.0, [r["jamshidian"] for r in swaption_rows], 0.0]
    mc_on_reset = np.interp(reset_grid, times, prof["EE_discounted"])
    cva_closed_semiannual = cva(reset_grid, closed_ee, lam_c, LGD)
    cva_mc_semiannual = cva(reset_grid, mc_on_reset, lam_c, LGD)

    # ---------------- convergence ----------------
    convergence = []
    for k, n in enumerate(CONVERGENCE_PATHS):
        est, se = cva_estimate(model, swap, times, n, lam_c, seed=10 + k)
        convergence.append({"n_paths": n, "cva": est, "se": se})
        print(f"  {n:>7} paths: CVA {est:,.0f} ± {1.96 * se:,.0f}")

    # ---------------- sensitivities ----------------
    sens = {}
    for label, factor in [("sigma_x0.5", 0.5), ("sigma_x1.5", 1.5)]:
        m2 = HullWhite(model.a, model.sigma * factor, curve)
        s2 = simulate_values(m2, [swap], times, N_PATHS, seed=SEED)
        ee2 = exposure_profiles(s2["values"][0], s2["deflator"])["EE_discounted"]
        sens[label] = cva(times, ee2, lam_c, LGD)
    for label, factor in [("spread_x0.5", 0.5), ("spread_x2", 2.0)]:
        sens[label] = cva(times, prof["EE_discounted"], lam_c * factor, LGD)
    rec_ee = exposure_profiles(-values, deflator)  # same paths, opposite side
    sens["receiver_swap"] = cva(times, rec_ee["EE_discounted"], lam_c, LGD)
    sens["receiver_swap_peak_EE"] = float(rec_ee["EE"].max())
    print("sensitivities:", {k: round(v) for k, v in sens.items()})

    payload = {
        "trade": {"notional": NOTIONAL, "maturity": MATURITY, "fixed_rate": swap.fixed_rate,
                  "side": "payer (pay fixed, receive floating)", "frequency": "semi-annual both legs"},
        "monte_carlo": {"n_paths": N_PATHS, "antithetic": True, "grid_steps": len(times) - 1,
                        "step_years": float(times[1]), "seed": SEED},
        "value_at_t0_curve": swap.value(curve),
        "value_at_t0_mc": float(values[0].mean()),
        "max_abs_deflator_error": float(np.abs(deflator_ratio - 1).max()),
        "exposure": {
            "peak_EE": float(prof["EE"][peak]), "peak_EE_time": float(times[peak]),
            "peak_PFE": float(prof["PFE"][peak_pfe]), "peak_PFE_time": float(times[peak_pfe]),
            "EPE_life": epe_life, "EPE_1y": epe_1y, "effective_EPE_1y": eepe_1y,
            "EE_at": {str(t): float(np.interp(t, times, prof["EE"])) for t in [1, 2, 3, 5, 7, 9]},
            "PFE_at": {str(t): float(np.interp(t, times, prof["PFE"])) for t in [1, 2, 3, 5, 7, 9]},
        },
        "cva": {
            "cva_unilateral": cva_uni, "cva_se": cva_se, "cva_bp_notional": cva_uni / NOTIONAL * 1e4,
            "cva_running_bp": cva_uni / annuity * 1e4,
            "dva_unilateral": dva_uni, "cva_bilateral": cva_bil, "dva_bilateral": dva_bil,
            "bcva": cva_bil - dva_bil, "hazard_counterparty": lam_c, "hazard_bank": lam_b, "lgd": LGD,
            "cva_semiannual_closed_form": cva_closed_semiannual,
            "cva_semiannual_mc": cva_mc_semiannual,
        },
        "swaption_check": swaption_rows,
        "swaption_check_max_abs_z": float(np.abs(z_scores).max()),
        "swaption_check_max_rel_error": float(rel_err.max()),
        "convergence": convergence,
        "sensitivities": sens,
    }
    save_json("exposure_cva.json", payload)

    # ---------------- figures ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for j in range(40):
        ax.plot(times, values[:, j], color=GREY, lw=0.4, alpha=0.5)
    ax.plot(times, prof["PFE"], color=RED, lw=2, label="PFE 97.5%")
    ax.plot(times, prof["EE"], color=BLUE, lw=2, label="EE")
    ax.plot(times, -np.quantile(np.maximum(-values, 0), QUANTILE, axis=1), color=GREEN, lw=1.5, ls="--",
            label="negative exposure, 97.5%")
    ax.axhline(epe_life, color=ORANGE, ls=":", label=f"EPE (life) = {epe_life / 1e6:.1f}m")
    ax.set_xlabel("years")
    ax.set_ylabel("USD million")
    millions(ax)
    ax.set_title("10y payer swap, USD 100m: simulated values and exposure")
    ax.legend(loc="lower left", fontsize=8)
    ax = axes[1]
    contrib = cva_term_structure(times, prof["EE_discounted"], lam_c, LGD)
    ax.bar(0.5 * (times[1:] + times[:-1]), contrib / np.diff(times) / 1e3, width=times[1], color=BLUE)
    ax.set_xlabel("years")
    ax.set_ylabel("USD thousand per year")
    ax.set_title(f"Where the CVA comes from (total {cva_uni / 1e3:,.0f}k)")
    save(fig, "fig3_exposure.png")

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    exp_ = [r["expiry"] for r in swaption_rows]
    ax.plot(exp_, [r["jamshidian"] for r in swaption_rows], "-", color=ORANGE, lw=2,
            label="Jamshidian closed form (payer swaption)")
    ax.errorbar(exp_, [r["mc_ee_discounted"] for r in swaption_rows],
                yerr=[1.96 * r["mc_se"] for r in swaption_rows], fmt="o", color=BLUE, ms=4,
                label="Monte Carlo EE*(t) ± 1.96 s.e.")
    ax.set_xlabel("exposure date = swaption expiry (years)")
    ax.set_ylabel("USD million")
    millions(ax)
    ax.legend()
    save(fig, "fig4_swaption_check.png")

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    n = np.array([c["n_paths"] for c in convergence])
    est = np.array([c["cva"] for c in convergence])
    se = np.array([c["se"] for c in convergence])
    ax.errorbar(n, est / 1e3, yerr=1.96 * se / 1e3, fmt="o-", color=BLUE, capsize=3, label="CVA ± 1.96 s.e.")
    ax.axhline(convergence[-1]["cva"] / 1e3, color=GREY, ls=":", label="100,000 paths")
    ax.set_xscale("log")
    ax.set_xlabel("number of paths")
    ax.set_ylabel("CVA (USD thousand)")
    ax.legend()
    save(fig, "fig5_convergence.png")
