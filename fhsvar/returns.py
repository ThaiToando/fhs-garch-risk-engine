"""Log returns plus the stylized-fact checks that justify using GARCH.

Each test here backs a later modeling choice. Returns are in percent (x100)
to match the config convention (keeps arch's optimizer stable).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from fhsvar.config import Config
from fhsvar.data import load_prices

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def log_returns(prices: pd.Series, in_percent: bool = True) -> pd.Series:
    """Daily log returns r = ln(P_t / P_t-1). Scaled x100 for percent units."""
    r = pd.Series(np.log(prices / prices.shift(1)), index=prices.index).dropna()
    if in_percent:
        r = r * 100.0  # arch prefers magnitudes near 1
    return r.rename("return")


def summary_stats(r: pd.Series) -> dict:
    # excess kurtosis is 0 for a Normal; daily equity runs +3 to +10 (fat tails)
    return {
        "n": int(r.size),
        "mean": float(r.mean()),
        "std": float(r.std()),
        "skew": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),  # Fisher, so Normal = 0
        "min": float(r.min()),
        "max": float(r.max()),
    }


def stylized_fact_tests(r: pd.Series) -> dict:
    """Numbers that show clustering and fat tails."""
    # tiny p on squared returns means volatility clustering is present
    lb_sq = acorr_ljungbox(r**2, lags=[10], return_df=True)
    lb_sq_p = float(lb_sq["lb_pvalue"].iloc[0])

    # raw returns should be far less autocorrelated (roughly unpredictable)
    lb_raw = acorr_ljungbox(r, lags=[10], return_df=True)
    lb_raw_p = float(lb_raw["lb_pvalue"].iloc[0])

    # lag-1 ACF: squared should clearly beat raw
    acf_raw = float(acf(r, nlags=1, fft=True)[1])
    acf_sq = float(acf(r**2, nlags=1, fft=True)[1])

    return {
        "ljungbox_p_raw": lb_raw_p,
        "ljungbox_p_squared": lb_sq_p,
        "acf1_raw": acf_raw,
        "acf1_squared": acf_sq,
    }


def make_plots(r: pd.Series, ticker: str) -> Path:
    """Four-panel diagnostic figure: returns, ACF raw, ACF squared, QQ."""
    import matplotlib

    matplotlib.use("Agg")  # write to file, no GUI window
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(exist_ok=True)
    acf_raw = acf(r, nlags=40, fft=True)
    acf_sq = acf(r**2, nlags=40, fft=True)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{ticker}: stylized facts (returns in %)", fontsize=13)

    ax[0, 0].plot(r.index, r.values, lw=0.5)
    ax[0, 0].set_title("Daily log returns (note the 2008 and 2020 clusters)")
    ax[0, 0].set_ylabel("return (%)")

    ax[0, 1].stem(range(len(acf_raw)), acf_raw)
    ax[0, 1].set_title("ACF of returns: near zero, so unpredictable")
    ax[0, 1].set_xlabel("lag")

    ax[1, 0].stem(range(len(acf_sq)), acf_sq)
    ax[1, 0].set_title("ACF of squared returns: persistent (clustering)")
    ax[1, 0].set_xlabel("lag")

    stats.probplot(r, dist="norm", plot=ax[1, 1])
    ax[1, 1].set_title("QQ vs Normal: heavy ends are the fat tails")

    fig.tight_layout()
    out = RESULTS_DIR / f"{ticker.lstrip('^')}_stylized_facts.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    cfg = Config()
    prices = load_prices(cfg)
    r = log_returns(prices, in_percent=cfg.returns_in_percent)

    s = summary_stats(r)
    t = stylized_fact_tests(r)

    print("=== Summary statistics (returns in %) ===")
    for k, v in s.items():
        print(f"  {k:16s}: {v:.4f}" if isinstance(v, float) else f"  {k:16s}: {v}")

    print("\n=== Stylized-fact tests ===")
    print(f"  Ljung-Box p (raw returns)     : {t['ljungbox_p_raw']:.2e}")
    print(f"  Ljung-Box p (squared returns) : {t['ljungbox_p_squared']:.2e}")
    print(f"  ACF lag-1 (raw)               : {t['acf1_raw']:.4f}")
    print(f"  ACF lag-1 (squared)           : {t['acf1_squared']:.4f}")

    verdict = (
        s["excess_kurtosis"] > 0
        and t["ljungbox_p_squared"] < 0.01
        and t["acf1_squared"] > t["acf1_raw"]
    )
    print(f"\n  Stylized facts confirmed: {verdict}")
    print("  Needs a conditional-volatility model with fat-tailed innovations.")

    fig_path = make_plots(r, cfg.ticker)
    print(f"\n  Saved diagnostic figure: {fig_path}")