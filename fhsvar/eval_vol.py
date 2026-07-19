import pandas as pd
import numpy as np
from fhsvar.config import Config
from fhsvar.volatility import build_model, qlike_loss

def run_evaluation():
    cfg = Config()
    
    # 1. Load the realized returns from our previous backtest data
    df = pd.read_csv('results/backtest_comparison.csv', index_col=0, parse_dates=True)
    returns = df['realized'].dropna()
    
    # 2. Fit the GARCH model to get our predicted in-sample variance
    print(f"Fitting {cfg.model}...")
    res = build_model(returns, cfg.model, cfg.dist).fit(disp="off")
    garch_var = res.conditional_volatility ** 2
    
    # 3. Create a naive baseline (just the constant overall sample variance)
    naive_var = np.full_like(garch_var, returns.var())
    
    # 4. Our "true" realized proxy is just the squared daily returns
    realized_var = returns ** 2
    
    # 5. Calculate QLIKE loss (lower is better!)
    garch_qlike = qlike_loss(realized_var, garch_var)
    naive_qlike = qlike_loss(realized_var, naive_var)
    
    print("\n--- Volatility Forecast Evaluation ---")
    print(f"GARCH ({cfg.model}) QLIKE Loss: {garch_qlike:.4f}")
    print(f"Naive Constant-Vol QLIKE Loss:  {naive_qlike:.4f}")
    
    if garch_qlike < naive_qlike:
        print("\nSuccess: GARCH beats the naive baseline!")
    else:
        print("\nWarning: Naive baseline won.")

if __name__ == "__main__":
    run_evaluation()