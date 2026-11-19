"""Plain-vanilla interest rate swap, valued on the curve and on Hull-White paths.

Single-curve set-up: the floating leg pays the simple rate of the model curve over each
6-month period (fixed at the start of the period, paid at the end), and both legs pay
semi-annually with accrual 0.5. A payer swap pays fixed and receives floating.
"""

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9


@dataclass
class Swap:
    notional: float
    fixed_rate: float
    start: float
    end: float
    payer: bool = True  # True: pay fixed, receive floating
    frequency: int = 2
    pay_times: np.ndarray = field(init=False)

    def __post_init__(self):
        step = 1.0 / self.frequency
        self.pay_times = np.arange(self.start + step, self.end + EPS, step)
        self.accrual = step

    @property
    def sign(self):
        return 1.0 if self.payer else -1.0

    def reset_times(self):
        return self.pay_times - self.accrual

    # ---------- value today on the curve ----------
    def annuity(self, curve):
        return self.accrual * curve.df(self.pay_times).sum()

    def value(self, curve):
        floating = curve.df(self.start) - curve.df(self.end)
        fixed = self.fixed_rate * self.annuity(curve)
        return self.sign * self.notional * (floating - fixed)

    # ---------- value at a future date on simulated paths ----------
    def value_on_paths(self, model, t, x_t, fixings):
        """Swap value at time t on each path (just after any payment due at t).

        `fixings` maps a reset time to P(reset, reset + accrual) on each path; it must contain
        the reset of the current period once t has passed that reset.
        """
        remaining = self.pay_times[self.pay_times > t + EPS]
        if remaining.size == 0:
            return np.zeros_like(x_t)
        bonds = model.bond_price(t, remaining[:, None], x_t[None, :])  # (n_remaining, n_paths)
        first_reset = remaining[0] - self.accrual
        if first_reset < t - EPS:  # current coupon already fixed
            first_float = bonds[0] / fixings[round(first_reset, 8)]
        elif first_reset <= t + EPS:  # fixing today: the floating leg is worth par
            first_float = np.ones_like(x_t)
        else:  # forward-starting swap
            first_float = model.bond_price(t, first_reset, x_t)
        floating = first_float - bonds[-1]
        fixed = self.fixed_rate * self.accrual * bonds.sum(axis=0)
        return self.sign * self.notional * (floating - fixed)


def par_swap_rate(curve, start, end, frequency=2):
    pay = np.arange(start + 1 / frequency, end + EPS, 1 / frequency)
    return float((curve.df(start) - curve.df(end)) / (curve.df(pay).sum() / frequency))
