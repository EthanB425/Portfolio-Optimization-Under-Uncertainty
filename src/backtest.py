import numpy as np
import pandas as pd


def run_backtest(strategy, returns, rf, window=60, cost_rate=0.001):
    """Walk-forward backtest: estimate on the past `window` months, hold one month, repeat.

    Returns monthly results (gross return, net return, turnover, risk-free rate)
    and the weights held each month.
    """
    n = returns.shape[1]
    w_drifted = np.zeros(n)  # start fully in cash

    records = []
    weights_history = {}

    for t in range(window, len(returns)):
        past = returns.iloc[t - window:t]  # only data available before month t
        w = strategy(past).reindex(returns.columns).values

        turnover = np.abs(w - w_drifted).sum()
        r_t = returns.iloc[t].values
        gross = float(w @ r_t)
        net = gross - cost_rate * turnover

        date = returns.index[t]
        records.append({"date": date, "gross": gross, "net": net, "turnover": turnover})
        weights_history[date] = w

        # weights drift with returns before the next rebalance
        w_drifted = w * (1 + r_t) / (1 + gross)

    results = pd.DataFrame(records).set_index("date")
    results["rf"] = rf.reindex(results.index).values
    weights = pd.DataFrame(weights_history, index=returns.columns).T
    return results, weights


def performance_summary(results, weights):
    """Headline metrics for one strategy's backtest."""
    net = results["net"]
    excess = net - results["rf"]
    wealth = (1 + net).cumprod()
    n_months = len(net)
    drawdown = wealth / np.maximum(wealth.cummax(), 1.0) - 1

    return pd.Series({
        "ann_return": wealth.iloc[-1] ** (12 / n_months) - 1,
        "ann_vol": net.std() * np.sqrt(12),
        "sharpe": excess.mean() / excess.std() * np.sqrt(12),
        "max_drawdown": drawdown.min(),
        "avg_monthly_turnover": results["turnover"].iloc[1:].mean(),  # skip initial buy-in
        "avg_effective_n": (1 / (weights ** 2).sum(axis=1)).mean(),
    })