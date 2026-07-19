"""Run FHS, plain HS, and Normal-parametric through the SAME rolling harness,
then apply every backtest test to each. This produces the headline table.

Same window, same test period, same alpha. Only the VaR method differs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fhsvar.benchmarks import hs_var_es, normal_var_es
from fhsvar.config import Config
from fhsvar.data import load_prices
from fhsvar.fhs import fhs_var_es
from fhsvar.metrics import basel_traffic_light, christoffersen, kupiec_pof
from fhsvar.residuals import extract_residuals
from fhsvar.returns import log_returns
from fhsvar.volatility import fit_and_forecast

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_all_methods(returns: pd.Series, cfg: Config, test_days=None, verbose=True) -> pd.DataFrame:
    """One rolling loop that computes all three VaRs per day. Guarantees the
    same window and dates for every method (true apples-to-apples)."""
    window = cfg.window
    n = len(returns)
    start = window if test_days is None else max(window, n - 1 - test_days)

    rows = []
    for i, t in enumerate(range(start, n - 1)):
        train = returns.iloc[t - window : t]         # strictly up to day t
        train_arr = train.to_numpy()

        # FHS (fit the chosen model)
        fit = fit_and_forecast(train, model=cfg.model, dist=cfg.dist)
        z = extract_residuals(fit.result, rescale=True)
        mu = float(fit.result.params["mu"])
        fhs_v, fhs_e = fhs_var_es(fit.sigma_next, z, cfg.alpha, mu)

        # the two baselines on the same window
        hs_v, hs_e = hs_var_es(train_arr, cfg.alpha)
        nrm_v, nrm_e = normal_var_es(train_arr, cfg.alpha)

        realized = float(returns.iloc[t + 1])        # revealed after VaRs fixed
        rows.append(
            {
                "date": returns.index[t + 1],
                "realized": realized,
                "fhs_var": fhs_v, "fhs_exc": int(realized < -fhs_v),
                "hs_var": hs_v, "hs_exc": int(realized < -hs_v),
                "normal_var": nrm_v, "normal_exc": int(realized < -nrm_v),
            }
        )
        if verbose and (i + 1) % 200 == 0:
            print(f"  {i + 1} days done")

    return pd.DataFrame(rows).set_index("date")


def method_scorecard(exc: np.ndarray, alpha: float) -> dict:
    k = kupiec_pof(exc, alpha)
    c = christoffersen(exc, alpha)
    b = basel_traffic_light(exc)
    return {
        "exceptions": k["N"],
        "kupiec_p": k["pvalue"],
        "christ_ind_p": c["pvalue_ind"],
        "christ_cc_p": c["pvalue_cc"],
        "basel_zone": b["zone"],
    }


if __name__ == "__main__":
    cfg = Config()
    r = log_returns(load_prices(cfg), in_percent=cfg.returns_in_percent)

    print(f"=== Method comparison ({int((1-cfg.alpha)*100)}% VaR, window {cfg.window}) ===")
    print("  running all three methods through the same harness...")
    df = run_all_methods(r, cfg, test_days=None)

    T = len(df)
    expected = cfg.alpha * T
    print(f"\n  out-of-sample days: {T}, expected exceptions: {expected:.1f}\n")

    table = pd.DataFrame(
        {
            "FHS-GARCH": method_scorecard(df["fhs_exc"].to_numpy(), cfg.alpha),
            "Plain HS": method_scorecard(df["hs_exc"].to_numpy(), cfg.alpha),
            "Normal": method_scorecard(df["normal_exc"].to_numpy(), cfg.alpha),
        }
    ).T
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(table.to_string())

    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "backtest_comparison.csv")
    table.to_csv(RESULTS_DIR / "comparison_scorecard.csv")
    print(f"\n  Saved comparison table and per-day series to {RESULTS_DIR}")