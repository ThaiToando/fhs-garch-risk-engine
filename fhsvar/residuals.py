"""Extract and validate the standardized-residual pool that FHS resamples.

The pool is the empirical distribution of shocks with the time-varying volatility
divided out (z = eps/sigma). It must be clean: no warm-up NaNs, mean near 0,
std near 1. If std drifts from 1 we rescale, so the FHS re-inflation scale is
exactly sigma. Keeping this here leaves the FHS engine a pure quantile calc.
"""

import numpy as np
import pandas as pd

from fhsvar.config import Config
from fhsvar.data import load_prices
from fhsvar.returns import log_returns
from fhsvar.volatility import fit_and_forecast


def extract_residuals(res, rescale: bool = True) -> np.ndarray:
    """Standardized residuals as a clean 1-D array. Drop warm-up NaNs.

    With rescale=True, divide by the sample std so std is exactly 1. That
    keeps the re-inflation scale honest (r* = mu + sigma * z with std(z)=1).
    """
    z = np.asarray(res.std_resid, dtype=float)
    z = z[~np.isnan(z)]  # drop GARCH warm-up NaNs
    if rescale:
        s = float(z.std(ddof=0))
        if s > 0:
            z = z / s
    return z


def residual_summary(z: np.ndarray) -> dict:
    zs = pd.Series(z)
    return {
        "n": int(z.size),
        "mean": float(z.mean()),
        "std": float(z.std(ddof=0)),
        "skew": float(zs.skew()),                 # type: ignore[arg-type]
        "excess_kurtosis": float(zs.kurtosis()),  # type: ignore[arg-type]
        "min": float(z.min()),
        "max": float(z.max()),
    }


def validate_pool(z: np.ndarray, tol: float = 0.05) -> None:
    """Fail loudly if the pool is not clean enough to resample."""
    assert z.ndim == 1, "residual pool must be 1-D"
    assert not np.isnan(z).any(), "residual pool still has NaNs"
    mean = float(z.mean())
    std = float(z.std(ddof=0))
    assert abs(mean) < tol, f"pool mean {mean:.4f} not within +/-{tol}"
    assert abs(std - 1.0) < tol, f"pool std {std:.4f} not near 1"


if __name__ == "__main__":
    cfg = Config()
    r = log_returns(load_prices(cfg), in_percent=cfg.returns_in_percent)

    fit = fit_and_forecast(r, model=cfg.model, dist=cfg.dist)
    z = extract_residuals(fit.result, rescale=True)
    validate_pool(z)

    s = residual_summary(z)
    print(f"=== Standardized residual pool ({cfg.model}-{cfg.dist}) ===")
    for k, v in s.items():
        print(f"  {k:16s}: {v:.4f}" if isinstance(v, float) else f"  {k:16s}: {v}")

    print("\n  Pool is clean: no NaNs, mean ~ 0, std ~ 1.")
    print(f"  Ready for FHS: {s['n']} residuals, next-day sigma = {fit.sigma_next:.4f} %")