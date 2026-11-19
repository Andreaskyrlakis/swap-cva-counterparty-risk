"""Shared set-up: the trade, the Monte Carlo settings and the calibrated market."""

import json

import numpy as np

from .curve import bootstrap_treasury_curve
from .cva import hazard_from_spread
from .data import RESULTS, valuation_inputs
from .hull_white import HullWhite
from .swap import Swap, par_swap_rate

NOTIONAL = 100e6          # USD 100 million
MATURITY = 10.0           # years
LGD = 0.60                # loss given default of the counterparty (recovery 40%)
OWN_LGD = 0.60            # our own loss given default (for DVA)
N_PATHS = 20_000          # base case Monte Carlo paths (antithetic pairs)
STEPS_PER_YEAR = 26       # two-week grid: one step = margin period of risk of 10 business days
SEED = 2026
QUANTILE = 0.975          # PFE level

# wrong-way risk: log-intensity Y with dY = -kappa Y dt + eta dW_Y, corr(dW_Y, dW_r) = rho
WWR_KAPPA = 0.10
WWR_ETA = 0.50


def load_market():
    """Curve, calibrated Hull-White model, par swap rate and hazard rates."""
    inputs = valuation_inputs()
    curve = bootstrap_treasury_curve(inputs["maturities"], inputs["par_yields"])
    calib = json.loads((RESULTS / "calibration.json").read_text())
    model = HullWhite(calib["hull_white"]["a"], calib["hull_white"]["sigma"], curve)
    spreads = inputs["spreads"]
    return {
        "inputs": inputs,
        "curve": curve,
        "model": model,
        "par_rate": par_swap_rate(curve, 0.0, MATURITY),
        "hazard_cpty": hazard_from_spread(spreads["counterparty_BBB"], LGD),
        "hazard_own": hazard_from_spread(spreads["bank_A"], OWN_LGD),
    }


def base_swap(market, payer=True, maturity=MATURITY, start=0.0):
    rate = par_swap_rate(market["curve"], start, maturity)
    return Swap(NOTIONAL, rate, start, maturity, payer=payer)


def save_json(name, payload):
    RESULTS.mkdir(exist_ok=True)

    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple, np.ndarray)):
            return [clean(x) for x in v]
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v

    (RESULTS / name).write_text(json.dumps(clean(payload), indent=2))
