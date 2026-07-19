"""Uniform wrapper over the arch library for GARCH / GJR / EGARCH.

One interface for all three models: same fit call, same one-day-ahead sigma
forecast. Everything downstream (FHS engine, backtester) plugs in here and
never needs to know which model is underneath. Returns are in percent units.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

from fhsvar.config import Config
from fhsvar.data import load_prices
from fhsvar.returns import log_returns


def build_model(returns: pd.Series, model: str = "GJR", dist: str = "skewt"):
    """Map a config (model, dist) to the right arch_model spec. Constant mean."""
    model = model.upper()
    if model == "GARCH":
        # plain symmetric GARCH(1,1)
        return arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, dist=dist)  # type: ignore[arg-type]
    if model == "GJR":
        # o=1 adds the leverage (down-day) term
        return arch_model(returns, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist=dist)  # type: ignore[arg-type]
    if model == "EGARCH":
        # log-variance form, leverage built in, no positivity constraints needed
        return arch_model(returns, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist=dist)  # type: ignore[arg-type]
    raise ValueError(f"unknown model: {model}")


@dataclass
class VolFit:
    """What downstream code needs from a fitted model."""
    result: ARCHModelResult  # the full arch result (params, resid, etc.)
    sigma_next: float        # one-day-ahead conditional volatility forecast
    model: str
    dist: str


def fit_and_forecast(returns: pd.Series, model: str = "GJR", dist: str = "skewt") -> VolFit:
    """Fit the model and forecast next-day sigma. The main entry point."""
    res = build_model(returns, model, dist).fit(disp="off")
    fc = res.forecast(horizon=1, reindex=False)
    # forecast gives VARIANCE; sqrt it to get sigma. This is a common bug spot.
    var_next = float(fc.variance.iloc[-1, 0])  # type: ignore[arg-type]
    sigma_next = float(np.sqrt(var_next))
    return VolFit(result=res, sigma_next=sigma_next, model=model.upper(), dist=dist)


def leverage_param(res: ARCHModelResult) -> float | None:
    """Return the leverage term if the model has one (GJR/EGARCH), else None."""
    for key in ("gamma[1]",):
        if key in res.params.index:
            return float(res.params[key])
    return None


if __name__ == "__main__":
    cfg = Config()
    r = log_returns(load_prices(cfg), in_percent=cfg.returns_in_percent)

    for m in ("GARCH", "GJR", "EGARCH"):
        fit = fit_and_forecast(r, model=m, dist=cfg.dist)
        res = fit.result
        p = res.params

        print(f"\n=== {m} ({cfg.dist} innovations) ===")
        print(f"  log-likelihood : {res.loglikelihood:.2f}")
        print(f"  AIC / BIC      : {res.aic:.2f} / {res.bic:.2f}")
        print(f"  next-day sigma : {fit.sigma_next:.4f} %")

        # persistence and long-run vol only make sense for the non-log models
        if m in ("GARCH", "GJR"):
            alpha = float(p.get("alpha[1]", np.nan))
            beta = float(p.get("beta[1]", np.nan))
            gamma = leverage_param(res) or 0.0
            # GARCH persistence is a+b; GJR adds half the leverage term
            persistence = alpha + beta + (gamma / 2.0 if m == "GJR" else 0.0)
            omega = float(p.get("omega", np.nan))
            longrun_var = omega / (1.0 - persistence) if persistence < 1 else np.nan
            longrun_sigma = float(np.sqrt(longrun_var))
            print(f"  alpha (react)  : {alpha:.4f}")
            print(f"  beta (memory)  : {beta:.4f}")
            if m == "GJR":
                print(f"  gamma (leverage): {gamma:.4f}  (>0 means down-days spike vol)")
            print(f"  persistence    : {persistence:.4f}  (must be < 1 for stationarity)")
            print(f"  long-run sigma : {longrun_sigma:.4f} %")
        else:
            gamma = leverage_param(res) or 0.0
            print(f"  gamma (leverage): {gamma:.4f}  (<0 means down-days spike vol)")
import numpy as np

def forecast_vol(returns, horizon=10, model="GJR", dist="skewt"):
    """
    Standalone function to forecast a volatility path for a given horizon.
    Useful for option pricing, position sizing, or vol-targeting.
    """
    # Assuming build_model is already defined in this file from Phase 4
    res = build_model(returns, model, dist).fit(disp="off")
    fc = res.forecast(horizon=horizon, reindex=False)
    
    # Return the forecasted volatility path (square root of variance)
    return np.sqrt(fc.variance.values[-1, :])

def qlike_loss(realized_var, predicted_var):
    """
    Calculate the QLIKE loss function.
    QLIKE is robust to the noisy squared-return proxy for true, latent variance.
    """
    # Ensure strictly positive inputs to avoid log(0) or division by zero
    eps = 1e-8
    predicted_var = np.maximum(predicted_var, eps)
    realized_var = np.maximum(realized_var, eps)
    
    # QLIKE formula: (Realized / Predicted) - ln(Realized / Predicted) - 1
    loss = (realized_var / predicted_var) - np.log(realized_var / predicted_var) - 1
    return np.mean(loss)