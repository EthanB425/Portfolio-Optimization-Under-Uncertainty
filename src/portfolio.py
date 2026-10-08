import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf

DELTA = 2.5  # risk aversion, standard value from the Black-Litterman literature


def mean_variance_weights(mu, sigma, delta=DELTA):
    """Long-only, fully invested mean-variance weights.

    Maximizes  mu'w - (delta/2) w'Sigma w  subject to  sum(w) = 1, w >= 0.
    mu and sigma must use the same time units (e.g. both monthly).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(mu)

    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - (delta / 2) * cp.quad_form(w, cp.psd_wrap(sigma)))
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if w.value is None:
        raise RuntimeError(f"Optimization failed with status: {problem.status}")

    weights = np.clip(w.value, 0, None)  # remove tiny negative values from solver rounding
    return weights / weights.sum()


def equal_weight(returns_window):
    """1/N benchmark: same weight in every asset."""
    n = returns_window.shape[1]
    return pd.Series(np.full(n, 1 / n), index=returns_window.columns)


def naive_mean_variance(returns_window, delta=DELTA):
    """Mean-variance using raw sample estimates of mu and Sigma."""
    mu = returns_window.mean().values
    sigma = returns_window.cov().values
    w = mean_variance_weights(mu, sigma, delta)
    return pd.Series(w, index=returns_window.columns)


def ledoit_wolf_mean_variance(returns_window, delta=DELTA):
    """Mean-variance with sample mu but Ledoit-Wolf shrunk Sigma."""
    mu = returns_window.mean().values
    sigma = LedoitWolf().fit(returns_window.values).covariance_
    w = mean_variance_weights(mu, sigma, delta)
    return pd.Series(w, index=returns_window.columns)


def _make_mv_solver(n, delta=DELTA):
    """Build a reusable mean-variance problem for n assets (much faster for repeated solves)."""
    w = cp.Variable(n)
    mu_p = cp.Parameter(n)
    chol_p = cp.Parameter((n, n))  # transpose of the Cholesky factor of Sigma
    objective = cp.Maximize(mu_p @ w - (delta / 2) * cp.sum_squares(chol_p @ w))
    problem = cp.Problem(objective, [cp.sum(w) == 1, w >= 0])

    def solve(mu, sigma):
        mu_p.value = np.asarray(mu, dtype=float)
        chol_p.value = np.linalg.cholesky(sigma).T  # w'Sigma w = ||L'w||^2
        problem.solve()
        if w.value is None:
            raise RuntimeError(f"Optimization failed with status: {problem.status}")
        weights = np.clip(w.value, 0, None)
        return weights / weights.sum()

    return solve


def resampled_mean_variance(returns_window, n_draws=500, delta=DELTA, seed=0):
    """Michaud-style resampling: average the optimal weights across simulated histories."""
    T, n = returns_window.shape
    mu_hat = returns_window.mean().values
    sigma_hat = returns_window.cov().values
    rng = np.random.default_rng(seed)
    solve = _make_mv_solver(n, delta)

    all_weights = np.zeros((n_draws, n))
    for b in range(n_draws):
        simulated = rng.multivariate_normal(mu_hat, sigma_hat, size=T)
        mu_b = simulated.mean(axis=0)
        sigma_b = np.cov(simulated, rowvar=False)
        all_weights[b] = solve(mu_b, sigma_b)

    return pd.Series(all_weights.mean(axis=0), index=returns_window.columns)


MARKET_WEIGHTS = pd.Series({
    "SPY": 0.25, "EFA": 0.15, "EEM": 0.07, "TLT": 0.08, "IEF": 0.12,
    "LQD": 0.12, "HYG": 0.05, "GLD": 0.06, "VNQ": 0.05, "DBC": 0.05,
})  # assumed approximate global multi-asset market portfolio

TAU = 0.05  # uncertainty scaling on the prior


def implied_returns(sigma, w_mkt, delta=DELTA):
    """Equilibrium returns implied by market weights: Pi = delta * Sigma * w_mkt."""
    return delta * np.asarray(sigma) @ np.asarray(w_mkt)


def black_litterman_returns(sigma, w_mkt, P=None, Q=None, delta=DELTA, tau=TAU):
    """Black-Litterman posterior expected returns.

    With no views (P is None), this is just the equilibrium prior.
    View uncertainty Omega follows He & Litterman: diag(P (tau Sigma) P').
    """
    sigma = np.asarray(sigma, dtype=float)
    pi = implied_returns(sigma, w_mkt, delta)
    if P is None:
        return pi

    P = np.atleast_2d(np.asarray(P, dtype=float))
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    tau_sigma = tau * sigma
    omega = np.diag(np.diag(P @ tau_sigma @ P.T))

    inv_tau_sigma = np.linalg.inv(tau_sigma)
    inv_omega = np.linalg.inv(omega)
    A = inv_tau_sigma + P.T @ inv_omega @ P
    b = inv_tau_sigma @ pi + P.T @ inv_omega @ Q
    return np.linalg.solve(A, b)


def black_litterman_no_views(returns_window, delta=DELTA):
    """Black-Litterman with no views: optimize on the equilibrium prior."""
    sigma = LedoitWolf().fit(returns_window.values).covariance_
    w_mkt = MARKET_WEIGHTS[returns_window.columns].values
    mu = black_litterman_returns(sigma, w_mkt, delta=delta)
    w = mean_variance_weights(mu, sigma, delta)
    return pd.Series(w, index=returns_window.columns)


MOMENTUM_LOOKBACK = 12   # months used to rank assets
MOMENTUM_N = 3           # assets in each of the top and bottom groups
VIEW_SPREAD = 0.03 / 12  # view: top group beats bottom group by 3% per year (monthly units)


def momentum_view(returns_window, lookback=MOMENTUM_LOOKBACK, n=MOMENTUM_N, spread=VIEW_SPREAD):
    """Relative view: the top-n trailing performers outperform the bottom-n by `spread`."""
    trailing = (1 + returns_window.iloc[-lookback:]).prod() - 1
    ranked = trailing.sort_values()
    bottom, top = list(ranked.index[:n]), list(ranked.index[-n:])
    P = pd.Series(0.0, index=returns_window.columns)
    P[top] = 1 / n
    P[bottom] = -1 / n
    return P.values, np.array([spread]), top, bottom


def black_litterman_momentum(returns_window, delta=DELTA, tau=TAU):
    """Black-Litterman with a systematic 12-month momentum view."""
    sigma = LedoitWolf().fit(returns_window.values).covariance_
    w_mkt = MARKET_WEIGHTS[returns_window.columns].values
    P, Q, _, _ = momentum_view(returns_window)
    mu = black_litterman_returns(sigma, w_mkt, P=P, Q=Q, delta=delta, tau=tau)
    w = mean_variance_weights(mu, sigma, delta)
    return pd.Series(w, index=returns_window.columns)