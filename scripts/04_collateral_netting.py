"""Effect of a CSA (threshold, margin period of risk) and of netting on exposure and CVA.

Collateral: two-way CSA, collateral = amount called at t - MPR above the threshold H.
Netting: the 10y payer swap is put in one netting set with a second swap on the same
paths, either an offsetting 5y receiver or a same-direction 5y payer.

Outputs: results/collateral_netting.json, figures/fig6_collateral.png, figures/fig7_netting.png
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cva import cva  # noqa: E402
from src.exposure import (collateralised_values, expected_positive_exposure,  # noqa: E402
                          exposure_profiles, simulate_values, time_grid)
from src.market import (LGD, MATURITY, N_PATHS, QUANTILE, SEED, STEPS_PER_YEAR,  # noqa: E402
                        base_swap, load_market, save_json)
from src.plotting import BLUE, GREEN, GREY, ORANGE, RED, PURPLE, millions, plt, save  # noqa: E402

THRESHOLDS = [np.inf, 10e6, 5e6, 2e6, 1e6, 0.0]
MPR_STEPS = {0: 0, 10: 1, 20: 2}  # business days -> two-week grid steps

if __name__ == "__main__":
    market = load_market()
    model, lam = market["model"], market["hazard_cpty"]
    times = time_grid(MATURITY, STEPS_PER_YEAR)
    swap10 = base_swap(market)
    receiver5 = base_swap(market, payer=False, maturity=5.0)
    payer5 = base_swap(market, payer=True, maturity=5.0)
    sim = simulate_values(model, [swap10, receiver5, payer5], times, N_PATHS, seed=SEED)
    deflator = sim["deflator"]
    v10, vr5, vp5 = sim["values"]

    def cva_of(values):
        prof = exposure_profiles(values, deflator, QUANTILE)
        return cva(times, prof["EE_discounted"], lam, LGD), prof

    base_cva, base_prof = cva_of(v10)

    # ---------------- collateral ----------------
    table = []
    profiles = {}
    for mpr_days, lag in MPR_STEPS.items():
        for h in THRESHOLDS:
            if mpr_days == 0 and h not in (np.inf, 0.0, 5e6):
                continue
            collat = v10 if np.isinf(h) else collateralised_values(v10, h, lag)
            value, prof = cva_of(collat)
            table.append({"threshold": None if np.isinf(h) else h, "mpr_business_days": mpr_days,
                          "cva": value, "cva_vs_uncollateralised": value / base_cva,
                          "peak_EE": float(prof["EE"].max()), "peak_PFE": float(prof["PFE"].max()),
                          "EPE_life": expected_positive_exposure(times, prof["EE"])})
            profiles[(h, mpr_days)] = prof
            print(f"H = {'inf' if np.isinf(h) else f'{h / 1e6:.0f}m':>4}, MPR {mpr_days:>2} days: "
                  f"CVA {value:>10,.0f} ({value / base_cva:.1%}), peak PFE {prof['PFE'].max() / 1e6:.2f}m")

    # ---------------- netting ----------------
    netting = {}
    for name, other in [("offsetting_receiver_5y", vr5), ("same_direction_payer_5y", vp5)]:
        cva_other, prof_other = cva_of(other)
        cva_net, prof_net = cva_of(v10 + other)
        netting[name] = {"cva_10y_alone": base_cva, "cva_5y_alone": cva_other,
                         "sum_standalone": base_cva + cva_other, "netted": cva_net,
                         "netting_benefit": 1 - cva_net / (base_cva + cva_other),
                         "peak_PFE_sum": float((base_prof["PFE"] + prof_other["PFE"]).max()),
                         "peak_PFE_netted": float(prof_net["PFE"].max())}
        netting[name]["_profiles"] = (base_prof["EE"] + prof_other["EE"], prof_net["EE"])
        print(f"{name}: sum of standalone CVAs {base_cva + cva_other:,.0f}, netted {cva_net:,.0f} "
              f"(benefit {netting[name]['netting_benefit']:.1%})")

    save_json("collateral_netting.json", {
        "uncollateralised_cva": base_cva, "csa_two_way": True, "collateral": table,
        "netting": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in netting.items()},
        "fixed_rates": {"swap10": swap10.fixed_rate, "receiver5": receiver5.fixed_rate, "payer5": payer5.fixed_rate},
    })

    # ---------------- figures ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for (h, mpr), colour, label in [((np.inf, 10), BLUE, "no CSA"),
                                    ((5e6, 10), ORANGE, "threshold 5m, MPR 10 days"),
                                    ((0.0, 10), GREEN, "threshold 0, MPR 10 days"),
                                    ((0.0, 20), PURPLE, "threshold 0, MPR 20 days")]:
        ax.plot(times, profiles[(h, mpr if not np.isinf(h) else 0)]["EE"], color=colour, label=label)
    ax.set_xlabel("years")
    ax.set_ylabel("EE (USD million)")
    millions(ax)
    ax.set_title("Expected exposure under a CSA")
    ax.legend(fontsize=8)
    ax = axes[1]
    finite = [h for h in THRESHOLDS if not np.isinf(h)]
    for mpr, colour in [(10, ORANGE), (20, RED)]:
        vals = [next(r["cva"] for r in table if r["threshold"] == h and r["mpr_business_days"] == mpr)
                for h in finite]
        ax.plot(np.array(finite) / 1e6, np.array(vals) / 1e3, "o-", color=colour, label=f"MPR {mpr} business days")
    ax.axhline(base_cva / 1e3, color=GREY, ls=":", label="no CSA")
    ax.set_xlabel("threshold (USD million)")
    ax.set_ylabel("CVA (USD thousand)")
    ax.set_title("CVA against the CSA threshold")
    ax.legend(fontsize=8)
    save(fig, "fig6_collateral.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (name, title) in zip(axes, [("offsetting_receiver_5y", "+ 5y receiver (offsetting)"),
                                        ("same_direction_payer_5y", "+ 5y payer (same direction)")]):
        summed, netted = netting[name]["_profiles"]
        ax.plot(times, base_prof["EE"], color=GREY, ls="--", label="10y payer alone")
        ax.plot(times, summed, color=BLUE, label="sum of standalone EEs (no netting)")
        ax.plot(times, netted, color=ORANGE, label="netting set EE")
        ax.set_title(f"10y payer {title}: benefit {netting[name]['netting_benefit']:.0%}")
        ax.set_xlabel("years")
        millions(ax)
    axes[0].set_ylabel("EE (USD million)")
    axes[0].legend(fontsize=8)
    save(fig, "fig7_netting.png")
