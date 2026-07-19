"""Single source of truth for the whole engine.

UNITS CONVENTION (decided here, enforced everywhere): returns are stored in
PERCENT (log-return x 100). The arch optimizer is numerically stable when
returns are ~O(1); raw decimals trigger DataScaleWarning and fragile MLE.
"""

from dataclasses import dataclass


@dataclass
class Config:
    ticker: str = "^GSPC"          # S&P 500 index
    start: str = "2005-01-01"
    end: str = "2024-12-31"
    alpha: float = 0.01            # 99% VaR (Basel market-risk level)
    window: int = 1000             # rolling estimation window (days)
    model: str = "GJR"             # "GARCH" | "GJR" | "EGARCH"
    dist: str = "skewt"            # "normal" | "t" | "skewt"
    refit_every: int = 1           # refit cadence in the backtest
    returns_in_percent: bool = True
    seed: int = 42