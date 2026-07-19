# FHS-GARCH VaR/ES Engine

A Value-at-Risk and Expected Shortfall engine using filtered historical
simulation on top of GARCH-family volatility models.

A GARCH model captures volatility clustering and forecasts tomorrow's
volatility; standardized residuals carry the market's
real fat tails; re-inflating those residuals by the forecast volatility builds
tomorrow's loss distribution. The result reacts to market regime like a
parametric model but keeps empirical tails like historical simulation.

## Status
Under construction — building phase by phase.

## Quickstart
```bash
python -m venv .venv
pip install -e .
```