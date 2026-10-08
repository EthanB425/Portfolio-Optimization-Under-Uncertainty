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