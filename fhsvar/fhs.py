"""The FHS one-day VaR/ES engine. The centerpiece, kept pure on purpose.

Given tomorrow's sigma forecast and a pool of standardized residuals, build the
next-day return distribution r* = mu + sigma * z, then read VaR off the lower
tail and ES off the average beyond it. Sign convention: returns are signed
(losses negative); VaR and ES are reported as POSITIVE loss magnitudes.
Returns and sigma are in percent units, so VaR/ES come out in percent.

Quantile convention: method="lower" picks the actual residual at or below the
alpha-quantile instead of interpolating toward zero so VaR sits on a real
historical shock and is not softened. This matters most in small tails.
"""

import numpy as np


def fhs_var_es(
    sigma_next: float,
    resid: np.ndarray,
    alpha: float = 0.01,
    mu: float = 0.0,
) -> tuple[float, float]:
    """Full-pool (deterministic) 1-day FHS VaR and ES.

    Re-inflate every residual once, take the empirical alpha-quantile. No
    randomness, fully reproducible. Ideal for the 1-day horizon.
    """
    z = np.asarray(resid, dtype=float)
    sim = mu + sigma_next * z                          # hypothetical next-day returns
    q = float(np.quantile(sim, alpha, method="lower"))  # lower: land on a real point
    var = -q                                           # report as a positive loss
    tail = sim[sim <= q]                               # at or beyond the quantile
    es = -float(tail.mean())                           # tail mean, as a positive loss
    return var, es


def fhs_var_es_bootstrap(
    sigma_next: float,
    resid: np.ndarray,
    alpha: float = 0.01,
    mu: float = 0.0,
    n: int = 50000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap (Monte Carlo) 1-day FHS VaR and ES.

    Draw residuals with replacement, re-inflate, take the quantile. Converges
    to the full-pool version as n grows. Needed for multi-day (Phase 8). Seeded
    so a given setting always gives the same number.
    """
    rng = np.random.default_rng(seed)
    z = np.asarray(resid, dtype=float)
    draws = rng.choice(z, size=n, replace=True)
    sim = mu + sigma_next * draws
    q = float(np.quantile(sim, alpha, method="lower"))
    tail = sim[sim <= q]
    return -q, -float(tail.mean())


if __name__ == "__main__":
    from fhsvar.config import Config
    from fhsvar.data import load_prices
    from fhsvar.residuals import extract_residuals
    from fhsvar.returns import log_returns
    from fhsvar.volatility import fit_and_forecast

    # --- a tiny hand-checkable example first (proves the math) ---
    # 100 residuals: the worst five are -10, the rest 0. sigma=1, mu=0, alpha=0.05.
    # the 5% lower-quantile lands on a -10, so VaR=10; the tail (five -10s) means ES=10.
    toy = np.array([-10.0] * 5 + [0.0] * 95)
    v, e = fhs_var_es(sigma_next=1.0, resid=toy, alpha=0.05, mu=0.0)
    print("=== Toy check (expect VaR=10, ES=10) ===")
    print(f"  VaR = {v:.4f}, ES = {e:.4f}")
    assert abs(v - 10.0) < 1e-9 and abs(e - 10.0) < 1e-9, "toy check failed"
    assert e >= v, "ES must be >= VaR"
    print("  toy check passed\n")

    # --- now the real thing on S&P 500 with the chosen model ---
    cfg = Config()
    r = log_returns(load_prices(cfg), in_percent=cfg.returns_in_percent)
    fit = fit_and_forecast(r, model=cfg.model, dist=cfg.dist)
    z = extract_residuals(fit.result, rescale=True)
    mu = float(fit.result.params["mu"])

    var_full, es_full = fhs_var_es(fit.sigma_next, z, cfg.alpha, mu)
    var_boot, es_boot = fhs_var_es_bootstrap(fit.sigma_next, z, cfg.alpha, mu, seed=cfg.seed)

    print(f"=== 1-day FHS on {cfg.ticker} ({cfg.model}-{cfg.dist}) ===")
    print(f"  next-day sigma   : {fit.sigma_next:.4f} %")
    print(f"  confidence level : {int((1 - cfg.alpha) * 100)}% (alpha = {cfg.alpha})")
    print(f"  full-pool  VaR / ES : {var_full:.4f} % / {es_full:.4f} %")
    print(f"  bootstrap  VaR / ES : {var_boot:.4f} % / {es_boot:.4f} %")
    print(f"  ES >= VaR (full)    : {es_full >= var_full}")