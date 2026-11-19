"""One-factor Hull-White model, fitted exactly to today's zero curve.

We write the short rate as r(t) = x(t) + phi(t), with
    dx = -a x dt + sigma dW,   x(0) = 0,
and phi(t) chosen so that the model reproduces the market discount factors P(0, t)
(Brigo and Mercurio, 2006, section 3.8, "G1++"). Bond prices are then
    P(t, T) = P(0, T) / P(0, t) * exp( 0.5 [V(t, T) - V(0, T) + V(0, t)] - B(t, T) x(t) ),
so the model needs only discount factors, not the derivative of the forward curve.

Paths are simulated EXACTLY (no time-discretisation error): x and its integral
I(t) = int_0^t x(s) ds are jointly Gaussian, and the bank-account deflator is
    D(t) = exp(-int_0^t r) = P(0, t) * exp(-0.5 V(0, t) - I(t)).
"""

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm


def B(a, t, T):
    return (1 - np.exp(-a * (T - t))) / a


def V(a, sigma, t, T):
    """Variance of int_t^T x(s) ds given x(t)."""
    tau = T - t
    return sigma**2 / a**2 * (tau + 2 / a * np.exp(-a * tau) - np.exp(-2 * a * tau) / (2 * a) - 1.5 / a)


class HullWhite:
    def __init__(self, a, sigma, curve):
        self.a, self.sigma, self.curve = float(a), float(sigma), curve

    # ---------- bond prices ----------
    def bond_price(self, t, T, x):
        """P(t, T) given x(t). T may be an array (broadcast against x)."""
        a, s, P = self.a, self.sigma, self.curve.df
        log_a = (np.log(P(T)) - np.log(P(t)) + 0.5 * (V(a, s, t, T) - V(a, s, 0, T) + V(a, s, 0, t)))
        return np.exp(log_a - B(a, t, T) * x)

    def short_rate_shift(self, t):
        """phi(t) = f(0, t) + sigma^2 / (2 a^2) (1 - e^{-a t})^2, so that r = x + phi."""
        a, s = self.a, self.sigma
        return self.curve.forward_rate(t) + s**2 / (2 * a**2) * (1 - np.exp(-a * t)) ** 2

    # ---------- simulation ----------
    def simulate(self, times, n_paths, seed=0, antithetic=True):
        """Exact joint simulation of x(t) and the deflator D(t) on the grid `times` (starts at 0).

        Returns a dict with x, deflator (n_times x n_paths) and the standard normals z_rate
        ((n_times-1) x n_paths) that drive x, which the wrong-way-risk model reuses.
        """
        times = np.asarray(times, dtype=float)
        a, s = self.a, self.sigma
        rng = np.random.default_rng(seed)
        half = n_paths // 2 if antithetic else n_paths
        n_steps = len(times) - 1
        z = rng.standard_normal((2, n_steps, half))
        if antithetic:
            z = np.concatenate([z, -z], axis=2)
        x = np.zeros((n_steps + 1, z.shape[2]))
        integral = np.zeros_like(x)
        for i in range(n_steps):
            h = times[i + 1] - times[i]
            e = np.exp(-a * h)
            var_x = s**2 / (2 * a) * (1 - e**2)
            var_i = s**2 / a**2 * (h - 2 * (1 - e) / a + (1 - e**2) / (2 * a))
            cov = s**2 / (2 * a**2) * (1 - e) ** 2
            # Cholesky of the 2x2 covariance of (x increment, integral increment)
            l11 = np.sqrt(var_x)
            l21 = cov / l11
            l22 = np.sqrt(max(var_i - l21**2, 0.0))
            x[i + 1] = x[i] * e + l11 * z[0, i]
            integral[i + 1] = integral[i] + x[i] * (1 - e) / a + l21 * z[0, i] + l22 * z[1, i]
        p0 = self.curve.df(times)[:, None]
        v0 = V(a, s, 0, times)[:, None]
        deflator = p0 * np.exp(-0.5 * v0 - integral)
        return {"times": times, "x": x, "deflator": deflator, "z_rate": z[0]}

    # ---------- closed forms ----------
    def zcb_option(self, T, S, strike, kind="put"):
        """European option expiring at T on the zero-coupon bond maturing at S (Jamshidian 1989)."""
        a, s, P = self.a, self.sigma, self.curve.df
        sigma_p = s * B(a, T, S) * np.sqrt((1 - np.exp(-2 * a * T)) / (2 * a))
        h = np.log(P(S) / (P(T) * strike)) / sigma_p + sigma_p / 2
        if kind == "call":
            return P(S) * norm.cdf(h) - strike * P(T) * norm.cdf(h - sigma_p)
        return strike * P(T) * norm.cdf(-h + sigma_p) - P(S) * norm.cdf(-h)

    def swaption(self, expiry, pay_times, strike, notional=1.0, payer=True):
        """European swaption on a swap starting at `expiry` (Jamshidian decomposition).

        A payer swaption is a put on a coupon bond (coupons strike * accrual, plus 1 at the end)
        struck at 1; Jamshidian writes it as a sum of zero-coupon bond puts.
        """
        pay_times = np.asarray(pay_times, dtype=float)
        accruals = np.diff(np.r_[expiry, pay_times])
        cash = strike * accruals
        cash[-1] += 1.0

        def bond_minus_one(x):
            return np.sum(cash * self.bond_price(expiry, pay_times, x)) - 1.0

        x_star = brentq(bond_minus_one, -2.0, 2.0)
        strikes = self.bond_price(expiry, pay_times, x_star)
        kind = "put" if payer else "call"
        price = sum(c * self.zcb_option(expiry, T, k, kind) for c, T, k in zip(cash, pay_times, strikes))
        return notional * price


def model_zero_yield_vol(a, sigma, tau):
    """Normal volatility of the tau-year zero yield in Hull-White: sigma * B(tau) / tau."""
    tau = np.asarray(tau, dtype=float)
    return sigma * (1 - np.exp(-a * tau)) / (a * tau)


def calibrate_to_yield_vols(tenors, observed_vols, a_bounds=(0.005, 1.0)):
    """Least-squares fit of (a, sigma) to the term structure of zero-yield volatilities."""
    tenors = np.asarray(tenors, dtype=float)
    observed_vols = np.asarray(observed_vols, dtype=float)

    def residuals(p):
        return model_zero_yield_vol(p[0], p[1], tenors) - observed_vols

    fit = least_squares(residuals, x0=[0.05, observed_vols.mean()],
                        bounds=([a_bounds[0], 1e-5], [a_bounds[1], 0.1]))
    a, sigma = fit.x
    rmse = float(np.sqrt(np.mean(fit.fun**2)))
    return float(a), float(sigma), rmse
