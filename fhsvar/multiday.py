"""Multi-day (h-day) FHS VaR/ES by simulating GARCH(1,1) variance paths.

Why not sqrt(h) scaling? It assumes i.i.d. constant-variance returns, which we
rejected: volatility mean-reverts and (with leverage) the multi-day loss is
skewed. So we simulate the actual variance path forward, drawing empirical
shocks at each step, and read the quantile off the h-day return distribution.

Scope note: the 1-day engine uses the best model (EGARCH-skewt). Here we use
GARCH(1,1) because its variance recursion is clean to propagate step by step.
The goal is to show correct path simulation and why sqrt(h) is wrong; extending
the recursion to EGARCH is a documented next step.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fhsvar.config import Config
from fhsvar.data import load_prices
from fhsvar.residuals import extract_residuals
from fhsvar.returns import log_returns
from fhsvar.volatility import fit_and_forecast

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def simulate_hday_returns(res, resid, h, n=20000, seed=42):
    """Simulate n paths of h-day total returns under the fitted GARCH(1,1).

    Start from the fitted state at time T (last variance and shock). At each
    step draw a residual, form the return, update variance via the GARCH
    recursion using that simulated shock. Sum the h daily returns per path.
    """
    rng = np.random.default_rng(seed)
    p = res.params
    omega = float(p["omega"])
    alpha = float(p["alpha[1]"])
    beta = float(p["beta[1]"])
    mu = float(p["mu"])

    z = np.asarray(resid, dtype=float)
    # .iloc[-1] = last by POSITION (these are Series with a date index, so [-1] fails)
    last_sigma = float(res.conditional_volatility.iloc[-1])
    last_eps = float(res.resid.iloc[-1])

    # start each path from the last fitted variance and last shock
    sigma2 = np.full(n, last_sigma**2)
    eps_prev = np.full(n, last_eps)

    totals = np.zeros(n)
    for _ in range(h):
        # variance for this step from the previous shock and variance
        sigma2 = omega + alpha * eps_prev**2 + beta * sigma2
        draws = rng.choice(z, size=n)          # empirical shocks (fat tails, skew)
        eps = np.sqrt(sigma2) * draws          # this step's shock
        totals += mu + eps                     # daily return, accumulated
        eps_prev = eps                         # carry forward for next step's variance
    return totals


def hday_var_es(res, resid, h, alpha=0.01, n=20000, seed=42):
    """h-day VaR and ES from simulated GARCH paths. Positive loss magnitudes."""
    totals = simulate_hday_returns(res, resid, h, n=n, seed=seed)
    q = float(np.quantile(totals, alpha, method="lower"))
    tail = totals[totals <= q]
    return -q, -float(tail.mean())


def make_sqrt_h_plot(res, resid, var_1day, alpha, max_h=20, seed=42):
    """Plot simulated h-day VaR against the naive sqrt(h) * VaR_1 line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(exist_ok=True)
    hs = np.arange(1, max_h + 1)
    sim_var = np.array([hday_var_es(res, resid, int(h), alpha, seed=seed)[0] for h in hs])
    sqrt_line = var_1day * np.sqrt(hs)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hs, sim_var, marker="o", ms=4, label="simulated GARCH path VaR")
    ax.plot(hs, sqrt_line, ls="--", label="naive sqrt(h) scaling")
    ax.set_xlabel("horizon h (days)")
    ax.set_ylabel(f"{int((1 - alpha) * 100)}% VaR (%)")
    ax.set_title("Multi-day VaR: simulated paths vs sqrt(h) (the gap is the point)")
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / "multiday_sqrt_h.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    cfg = Config()
    r = log_returns(load_prices(cfg), in_percent=cfg.returns_in_percent)

    # use GARCH(1,1) here (clean recursion), skewt innovations kept
    fit = fit_and_forecast(r, model="GARCH", dist=cfg.dist)
    z = extract_residuals(fit.result, rescale=True)
    mu = float(fit.result.params["mu"])

    # 1-day VaR from this same GARCH fit, for an apples-to-apples sqrt(h) baseline
    from fhsvar.fhs import fhs_var_es

    var_1, es_1 = fhs_var_es(fit.sigma_next, z, cfg.alpha, mu)

    print(f"=== Multi-day FHS on {cfg.ticker} (GARCH-{cfg.dist} for paths) ===")
    print(f"  1-day  VaR / ES : {var_1:.4f} % / {es_1:.4f} %")

    for h in (1, 5, 10):
        v, e = hday_var_es(fit.result, z, h, cfg.alpha, seed=cfg.seed)
        sqrt_v = var_1 * np.sqrt(h)
        print(f"  {h:2d}-day VaR / ES : {v:.4f} % / {e:.4f} %   (sqrt(h) would give {sqrt_v:.4f} %)")

    fig_path = make_sqrt_h_plot(fit.result, z, var_1, cfg.alpha, seed=cfg.seed)
    print(f"\n  Saved sqrt(h) comparison figure: {fig_path}")
    print("  Note: sqrt(h) is wrong under GARCH. Mean-reversion pushes multi-day VaR")
    print("  below sqrt(h); fat-tailed skewed shocks compounding push it above. Here")
    print("  the tail effect wins by h=10, so sqrt(h) UNDERSTATES the 10-day loss.")