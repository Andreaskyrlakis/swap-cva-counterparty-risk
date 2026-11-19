"""Zero-coupon curve bootstrapped from US Treasury par yields.

Conventions (kept simple and written in the report):
- maturities up to 6 months are Treasury bills: the quoted bond-equivalent yield is read as a
  zero rate with semi-annual compounding, P(T) = (1 + y/2)^(-2T);
- from 1 year on, the quoted yield is the coupon of a par bond paying semi-annual coupons.
  Par yields are interpolated linearly on a 6-month grid and the discount factors are
  bootstrapped one coupon date at a time;
- between nodes, log discount factors are interpolated linearly (piecewise flat forwards).
"""

import numpy as np


class DiscountCurve:
    """Discount factors P(0, t) with log-linear interpolation (flat forwards between nodes)."""

    def __init__(self, times, discount_factors):
        times = np.asarray(times, dtype=float)
        dfs = np.asarray(discount_factors, dtype=float)
        if times[0] > 0:  # add P(0, 0) = 1
            times, dfs = np.r_[0.0, times], np.r_[1.0, dfs]
        self.times = times
        self.log_dfs = np.log(dfs)

    def df(self, t):
        t = np.asarray(t, dtype=float)
        inside = np.interp(t, self.times, self.log_dfs)
        # beyond the last node: keep the last forward rate flat
        last_forward = (self.log_dfs[-2] - self.log_dfs[-1]) / (self.times[-1] - self.times[-2])
        beyond = self.log_dfs[-1] - last_forward * (t - self.times[-1])
        return np.exp(np.where(t > self.times[-1], beyond, inside))

    def zero_rate(self, t):
        """Continuously compounded zero rate."""
        t = np.maximum(np.asarray(t, dtype=float), 1e-8)
        return -np.log(self.df(t)) / t

    def forward_rate(self, t, h=1e-4):
        """Instantaneous forward rate f(0, t) (numerical derivative)."""
        t = np.asarray(t, dtype=float)
        lo = np.maximum(t - h, 0.0)
        return -(np.log(self.df(t + h)) - np.log(self.df(lo))) / (t + h - lo)

    def shifted(self, bp):
        """Parallel shift of the continuously compounded zero curve, in basis points."""
        t = self.times[1:]
        return DiscountCurve(t, self.df(t) * np.exp(-bp * 1e-4 * t))


def bootstrap_treasury_curve(maturities, par_yields, last=30.0):
    """Bootstrap discount factors from Treasury constant-maturity par yields (in decimals)."""
    maturities = np.asarray(maturities, dtype=float)
    par_yields = np.asarray(par_yields, dtype=float)
    bills = maturities <= 0.5 + 1e-9

    times = list(maturities[bills])
    dfs = list((1 + par_yields[bills] / 2) ** (-2 * maturities[bills]))

    coupon_dates = np.arange(1.0, last + 1e-9, 0.5)
    coupons = np.interp(coupon_dates, maturities, par_yields)
    df_by_date = {0.5: dfs[times.index(0.5)]}
    for date, c in zip(coupon_dates, coupons):
        earlier = [df_by_date[d] for d in np.arange(0.5, date - 1e-9, 0.5)]
        p = (1 - c / 2 * sum(earlier)) / (1 + c / 2)
        df_by_date[date] = p
        times.append(date)
        dfs.append(p)
    return DiscountCurve(times, dfs)


def par_bond_price(curve, maturity, coupon):
    """Price of a bond paying coupon/2 every 6 months (used to check the bootstrap)."""
    dates = np.arange(0.5, maturity + 1e-9, 0.5)
    return coupon / 2 * curve.df(dates).sum() + curve.df(maturity)
