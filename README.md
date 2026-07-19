# Quantitative Risk Analytics: Filtered Historical Simulation 

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

I built a Value-at-Risk (VaR) and Expected Shortfall (ES) engine using Filtered Historical Simulation that reacts instantly to market regimes like a parametric model, while keeping the real fat tails of historical simulation.

## Core Methodology
The engine sits on three conceptual layers:
1. Econometric Volatility Layer: An EGARCH model captures volatility clustering and asymmetry (the leverage effect).
2. Empirical Residuals Layer: History is divided by conditional volatility to produce roughly i.i.d. standardized residuals, capturing true market skew and fat tails without parametric assumptions.
2. Simulation Layer: Residuals are re-inflated by tomorrow's forecasted volatility to generate a robust, highly reactive risk threshold.

##  The 2020 Crash
Notice how the FHS VaR (Red) instantly drops to capture the extreme market volatility, while the traditional Historical Simulation VaR (Blue Dashed) lags dangerously behind.

![Crisis Overlay](results/crisis_overlay.png)

## Risk Lab
This project includes a custom-built dashboard built with Streamlit and Plotly. 

**To run the interactive lab locally:**
```bash
pip install -e .
streamlit run app/dashboard.py