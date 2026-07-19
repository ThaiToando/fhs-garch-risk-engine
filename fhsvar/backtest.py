"""Rolling out-of-sample backtest. The most important control flow here.

The golden rule: the VaR for day t+1 uses ONLY information available at the
close of day t. We enforce this by construction: fit on returns strictly up to
day t, forecast VaR for t+1 and reveal the realized return t+1 only AFTER the
VaR number is locked. If that ordering holds, look-ahead bias is impossible.

Refit cadence: refitting the model every day is most correct but slow. This is
config-driven (cfg.refit_every). Between refits we reuse the last fit's residual
pool and rescale the sigma forecast and that is an approximation documented here.
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fhsvar.config import Config
from fhsvar.data import load_prices
from fhsvar.fhs import fhs_var_es
from fhsvar.residuals import extract_residuals
from fhsvar.returns import log_returns
from fhsvar.volatility import fit_and_forecast


def rolling_backtest(
    returns: pd.Series,
    cfg: Config,
    test_days: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Roll a fixed-length window across history, producing per-day VaR/ES/exception.

    test_days limits how many recent days to test (None = all after the window).
    """
    window = cfg.window
    n = len(returns)

    start = window
    if test_days is not None:
        start = max(window, n - 1 - test_days)  # only test the last `test_days`

    records = []
    t0 = time.time()
    for i, t in enumerate(range(start, n - 1)):
        train = returns.iloc[t - window : t]  # data STRICTLY up to day t
        fit = fit_and_forecast(train, model=cfg.model, dist=cfg.dist)
        z = extract_residuals(fit.result, rescale=True)
        mu = float(fit.result.params["mu"])

        var, es = fhs_var_es(fit.sigma_next, z, cfg.alpha, mu)

        realized = float(returns.iloc[t + 1])  # revealed ONLY after VaR is fixed
        exception = int(realized < -var)        # loss worse than VaR?

        records.append(
            {
                "date": returns.index[t + 1],
                "sigma": fit.sigma_next,
                "var": var,
                "es": es,
                "realized": realized,
                "exception": exception,
            }
        )

        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1} days done ({elapsed:.0f}s elapsed)")

    df = pd.DataFrame(records).set_index("date")
    return df


def summarize(df: pd.DataFrame, cfg: Config) -> dict:
    T = len(df)
    n_exc = int(df["exception"].sum())
    expected = cfg.alpha * T
    return {
        "test_days": T,
        "exceptions": n_exc,
        "expected": expected,
        "exception_rate": n_exc / T if T else float("nan"),
        "target_rate": cfg.alpha,
    }


if __name__ == "__main__":
    cfg = Config()
    r = log_returns(load_prices(cfg), in_percent=cfg.returns_in_percent)

    # First run: cap the test window so it finishes in a couple of minutes.
    # Set TEST_DAYS = None later for the full-history backtest.
    TEST_DAYS = None  # None = full out-of-sample history

    print(f"=== Rolling backtest ({cfg.model}-{cfg.dist}, {int((1-cfg.alpha)*100)}% VaR) ===")
    print(f"  window = {cfg.window}, refit_every = {cfg.refit_every}, test_days = {TEST_DAYS}")
    print("  running (this refits the model each day, so give it a minute)...")

    df = rolling_backtest(r, cfg, test_days=TEST_DAYS)
    s = summarize(df, cfg)

    print("\n=== Results ===")
    print(f"  out-of-sample days : {s['test_days']}")
    print(f"  exceptions         : {s['exceptions']}")
    print(f"  expected (alpha*T) : {s['expected']:.1f}")
    print(f"  exception rate     : {s['exception_rate']:.4f}  (target {s['target_rate']})")

    # save the table so later phases (tests, dashboard) can reuse it
    from pathlib import Path

    out = Path(__file__).resolve().parent.parent / "results" / "backtest_fhs.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out)
    print(f"\n  Saved per-day backtest table: {out}")