"""Download, clean, and cache a daily adjusted-close price series.

Design: the loader reads a local CSV cache if present, otherwise downloads
and writes it. Caching pins the exact series behind a result as providers
silently revise history and a reproducible headline number must not drift.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

from fhsvar.config import Config

# cache lives in the project's data/ folder (repo-root relative, not CWD-relative)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _cache_path(ticker: str) -> Path:
    # sanitize "^GSPC" -> "GSPC" so the filename is filesystem-safe
    safe = ticker.lstrip("^").replace("/", "_")
    return DATA_DIR / f"{safe}.csv"


def _download(ticker: str, start: str, end: str) -> pd.Series:
    # auto_adjust=True -> Close is already split/dividend-adjusted (no spurious jumps)
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker} in {start}..{end}")
    # yfinance may return a single- or multi-index column frame; pull one Close series
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.rename("price")


def _clean(prices: pd.Series) -> pd.Series:
    prices = prices.sort_index()                      # ensure chronological
    prices = prices[~prices.index.duplicated(keep="first")]  # drop duplicate dates
    prices = prices.dropna()                          # drop missing observations
    prices = prices[prices > 0]                       # no non-positive prices
    return prices


def _validate(prices: pd.Series) -> None:
    # fail loudly on the classic data-integrity problems
    assert prices.index.is_monotonic_increasing, "dates not sorted"
    assert prices.index.is_unique, "duplicate dates remain"
    assert (prices > 0).all(), "non-positive prices remain"
    # a >40% one-day move is almost always a data error, not a real move — flag it
    moves = prices.pct_change().abs()
    if (moves > 0.40).any():
        bad = moves[moves > 0.40]
        print(f"WARNING: {len(bad)} one-day move(s) >40% — inspect for data errors:")
        print(bad)


def load_prices(cfg: Config, force_refresh: bool = False) -> pd.Series:
    """Return a clean daily price Series, using the CSV cache when available."""
    DATA_DIR.mkdir(exist_ok=True)
    path = _cache_path(cfg.ticker)

    if path.exists() and not force_refresh:
        prices = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        prices.name = "price"
    else:
        prices = _clean(_download(cfg.ticker, cfg.start, cfg.end))
        prices.to_csv(path)  # cache for reproducibility

    _validate(prices)
    return prices


if __name__ == "__main__":
    cfg = Config()
    px = load_prices(cfg)
    print(f"Ticker: {cfg.ticker}")
    print(f"Rows:   {len(px)}")
    print(f"Range:  {px.index.min().date()} -> {px.index.max().date()}")
    print(f"Price:  min={px.min():.2f}  max={px.max():.2f}")
    print(px.tail())