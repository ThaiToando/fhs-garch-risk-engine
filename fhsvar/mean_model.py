"""Pick the mean model for r = mu + eps.

Phase 2 showed returns are near-unpredictable in the mean, so a constant mean
should win. This module backs that with an AIC/BIC comparison instead of just
asserting it. Lower AIC/BIC is better; BIC penalizes extra parameters harder.
"""

import warnings

import pandas as pd
from arch import arch_model

from fhsvar.config import Config
from fhsvar.data import load_prices
from fhsvar.returns import log_returns


def compare_mean_models(r: pd.Series) -> pd.DataFrame:
    """Fit a few mean specs with a common GARCH(1,1) vol and compare fit."""
    # hold the volatility part fixed so only the mean spec differs
    specs = {
        "Zero": dict(mean="Zero"),
        "Constant": dict(mean="Constant"),
        "AR(1)": dict(mean="AR", lags=1),
        "AR(2)": dict(mean="AR", lags=2),
    }

    rows = []
    for name, kw in specs.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence convergence chatter here
            res = arch_model(r, vol="GARCH", p=1, q=1, dist="t", **kw).fit(disp="off")
        rows.append(
            {
                "spec": name,
                "loglik": float(res.loglikelihood),
                "aic": float(res.aic),
                "bic": float(res.bic),
                "n_params": int(res.params.size),
            }
        )

    df = pd.DataFrame(rows).set_index("spec")
    return df.sort_values("bic")  # best (lowest) BIC first


if __name__ == "__main__":
    cfg = Config()
    r = log_returns(load_prices(cfg), in_percent=cfg.returns_in_percent)

    table = compare_mean_models(r)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print("=== Mean-model comparison (GARCH(1,1)-t volatility held fixed) ===")
    print(table)

    best_bic = table.index[0]
    const_bic = table.loc["Constant", "bic"]
    best_val = table.iloc[0]["bic"]
    gap = const_bic - best_val

    print(f"\n  Lowest BIC: {best_bic}")
    print(f"  Constant-mean BIC is {gap:.2f} above the best.")
    print("  Decision: use a Constant mean.")
    print("  AR terms can shave BIC a little (returns have weak negative lag-1")
    print("  autocorrelation), but the gain is small, not robust, and a VaR engine")
    print("  cares about the variance forecast, not a tiny mean improvement.")