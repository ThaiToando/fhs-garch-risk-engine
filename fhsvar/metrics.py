"""Backtest statistics: Kupiec, Christoffersen, Basel traffic-light.

These turn 'the exception count looks reasonable' into 'the VaR is validated'.
Kupiec checks the count; Christoffersen checks independence (no clustering);
Basel gives the regulator's green/yellow/red verdict on the last 250 days.
All take the 0/1 exception series produced by the rolling backtest.
"""

import numpy as np
from scipy import stats


def kupiec_pof(exceptions: np.ndarray, alpha: float) -> dict:
    """Kupiec proportion-of-failures (unconditional coverage) test.

    Likelihood ratio for whether the observed exception rate matches alpha.
    LR_uc ~ chi-square(1). Small p => reject (rate too high or too low).
    """
    x = np.asarray(exceptions, dtype=int)
    T = x.size
    N = int(x.sum())
    p = alpha

    # observed rate; guard the degenerate N=0 and N=T cases
    pi = N / T if T else 0.0
    if N == 0:
        lr = -2.0 * (T * np.log(1 - p))  # only the null term survives
    elif N == T:
        lr = -2.0 * (T * np.log(p))
    else:
        ll_null = (T - N) * np.log(1 - p) + N * np.log(p)
        ll_alt = (T - N) * np.log(1 - pi) + N * np.log(pi)
        lr = -2.0 * (ll_null - ll_alt)

    pval = float(stats.chi2.sf(lr, df=1))
    return {"test": "kupiec_pof", "N": N, "T": T, "rate": pi,
            "LR": float(lr), "pvalue": pval}


def christoffersen(exceptions: np.ndarray, alpha: float) -> dict:
    """Christoffersen independence + conditional coverage tests.

    Independence: does an exception today change tomorrow's exception odds?
    Conditional coverage combines correct frequency AND independence.
    LR_ind ~ chi2(1); LR_cc = LR_uc + LR_ind ~ chi2(2).
    """
    x = np.asarray(exceptions, dtype=int)

    # transition counts n_ij: from state i (yesterday) to j (today)
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(x[:-1], x[1:]):
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1

    # conditional and overall exception probabilities
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _safe_log(v):
        return np.log(v) if v > 0 else 0.0

    # likelihood under independence (single pi) vs Markov (pi01, pi11)
    ll_ind = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    ll_markov = (
        n00 * _safe_log(1 - pi01) + n01 * _safe_log(pi01)
        + n10 * _safe_log(1 - pi11) + n11 * _safe_log(pi11)
    )
    lr_ind = -2.0 * (ll_ind - ll_markov)
    p_ind = float(stats.chi2.sf(lr_ind, df=1))

    # conditional coverage = unconditional (Kupiec) + independence
    lr_uc = kupiec_pof(x, alpha)["LR"]
    lr_cc = lr_uc + lr_ind
    p_cc = float(stats.chi2.sf(lr_cc, df=2))

    return {
        "test": "christoffersen",
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "LR_ind": float(lr_ind), "pvalue_ind": p_ind,
        "LR_cc": float(lr_cc), "pvalue_cc": p_cc,
    }


def basel_traffic_light(exceptions: np.ndarray) -> dict:
    """Basel zone from exceptions in the most recent 250 trading days.

    Green 0-4, Yellow 5-9, Red 10+. Simple, robust, regulator-facing.
    """
    x = np.asarray(exceptions, dtype=int)
    last = x[-250:] if x.size >= 250 else x
    count = int(last.sum())
    if count <= 4:
        zone = "green"
    elif count <= 9:
        zone = "yellow"
    else:
        zone = "red"
    return {"test": "basel", "window": int(last.size), "exceptions_250": count, "zone": zone}


if __name__ == "__main__":
    from pathlib import Path

    import pandas as pd

    from fhsvar.config import Config

    cfg = Config()
    path = Path(__file__).resolve().parent.parent / "results" / "backtest_fhs.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    exc = df["exception"].to_numpy()

    print(f"=== Backtest tests on {len(df)} days ({cfg.model}-{cfg.dist}) ===\n")

    k = kupiec_pof(exc, cfg.alpha)
    print("Kupiec POF (unconditional coverage)")
    print(f"  exceptions {k['N']} / {k['T']}  (rate {k['rate']:.4f}, target {cfg.alpha})")
    print(f"  LR = {k['LR']:.3f}, p = {k['pvalue']:.3f}  "
          f"-> {'PASS' if k['pvalue'] > 0.05 else 'REJECT'} (want p > 0.05)\n")

    c = christoffersen(exc, cfg.alpha)
    print("Christoffersen independence + conditional coverage")
    print(f"  transitions: n00={c['n00']} n01={c['n01']} n10={c['n10']} n11={c['n11']}")
    print(f"  independence : LR = {c['LR_ind']:.3f}, p = {c['pvalue_ind']:.3f}  "
          f"-> {'PASS' if c['pvalue_ind'] > 0.05 else 'REJECT'}")
    print(f"  cond coverage: LR = {c['LR_cc']:.3f}, p = {c['pvalue_cc']:.3f}  "
          f"-> {'PASS' if c['pvalue_cc'] > 0.05 else 'REJECT'}\n")

    b = basel_traffic_light(exc)
    print("Basel traffic-light (last 250 days)")
    print(f"  {b['exceptions_250']} exceptions in {b['window']} days -> {b['zone'].upper()} zone")