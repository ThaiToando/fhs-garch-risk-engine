"""Model selection and residual diagnostics. This is the phase FHS rests on.

FHS only works if the standardized residuals are approximately i.i.d. So we
rank (model, dist) pairs by AIC/BIC AND test that the chosen model removed the
volatility clustering (Ljung-Box on squared standardized residuals). The
before/after on that test is the proof the model did its job.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from fhsvar.config import Config
from fhsvar.data import load_prices
from fhsvar.returns import log_returns
from fhsvar.volatility import fit_and_forecast, leverage_param

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def clean_std_resid(res) -> np.ndarray:
    """Standardized residuals as a clean 1-D array, warm-up NaNs dropped."""
    z = np.asarray(res.std_resid, dtype=float)
    return z[~np.isnan(z)]


def selection_table(r: pd.Series, dists=("normal", "t", "skewt")) -> pd.DataFrame:
    """Fit every (model, dist) pair and tabulate fit + the key residual test."""
    rows = []
    for model in ("GARCH", "GJR", "EGARCH"):
        for dist in dists:
            fit = fit_and_forecast(r, model=model, dist=dist)
            res = fit.result
            z = clean_std_resid(res)

            # the key test: squared standardized residuals should be clean now.
            # a LARGE p-value means clustering was removed (what we want).
            lb = acorr_ljungbox(z**2, lags=[10], return_df=True)
            lb_p = float(lb["lb_pvalue"].iloc[0])

            gamma = leverage_param(res)
            rows.append(
                {
                    "model": model,
                    "dist": dist,
                    "loglik": float(res.loglikelihood),
                    "aic": float(res.aic),
                    "bic": float(res.bic),
                    "lb_sq_p": lb_p,       # want > 0.05 (clustering removed)
                    "gamma": gamma if gamma is not None else np.nan,
                }
            )

    df = pd.DataFrame(rows).set_index(["model", "dist"])
    return df.sort_values("bic")


def before_after_clustering(r: pd.Series, model: str, dist: str) -> dict:
    """Ljung-Box on squared returns before vs squared std resid after."""
    before = acorr_ljungbox(r**2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]

    fit = fit_and_forecast(r, model=model, dist=dist)
    z = clean_std_resid(fit.result)
    after = acorr_ljungbox(z**2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]

    return {"before_p": float(before), "after_p": float(after)}


def make_qq_plot(r: pd.Series, model: str, dist: str) -> Path:
    """QQ of standardized residuals vs Normal. Should be straighter than raw."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(exist_ok=True)
    z = clean_std_resid(fit_and_forecast(r, model=model, dist=dist).result)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{model}-{dist}: standardized residual checks", fontsize=13)

    stats.probplot(z, dist="norm", plot=ax[0])
    ax[0].set_title("QQ of std residuals vs Normal")

    ax[1].hist(z, bins=80, density=True, alpha=0.6)
    xs = np.linspace(z.min(), z.max(), 200)
    ax[1].plot(xs, stats.norm.pdf(xs), lw=1.5)
    ax[1].set_title("Std residual histogram vs Normal pdf")
    ax[1].set_xlabel("standardized residual")

    fig.tight_layout()
    out = RESULTS_DIR / "residual_diagnostics.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    cfg = Config()
    r = log_returns(load_prices(cfg), in_percent=cfg.returns_in_percent)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    table = selection_table(r)
    print("=== Model selection (sorted by BIC, lower is better) ===")
    print(table)
    print("\n  lb_sq_p is the Ljung-Box p on squared standardized residuals.")
    print("  Want it ABOVE 0.05: means volatility clustering was removed.")

    best_model, best_dist = table.index[0]
    print(f"\n  Lowest BIC: {best_model} with {best_dist} innovations")

    ba = before_after_clustering(r, best_model, best_dist)
    print("\n=== Clustering before vs after (squared, Ljung-Box p) ===")
    print(f"  before (raw returns)        : {ba['before_p']:.2e}  (tiny = clustering)")
    print(f"  after  ({best_model}-{best_dist} residuals) : {ba['after_p']:.4f}  (large = removed)")

    fig_path = make_qq_plot(r, best_model, best_dist)
    print(f"\n  Saved residual diagnostics figure: {fig_path}")