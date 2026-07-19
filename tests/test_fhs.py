import numpy as np
import pytest
from fhsvar.fhs import fhs_var_es

def test_es_greater_than_var():
    """
    Test the fundamental invariant: Expected Shortfall (tail mean) 
    must always be >= Value-at-Risk (tail threshold).
    """
    # 1. Create a dummy pool of 1000 standard normal residuals
    np.random.seed(42)
    dummy_resid = np.random.normal(0, 1, 1000)
    
    # 2. Set a hypothetical next-day volatility
    sigma_next = 1.5
    
    # 3. Calculate VaR and ES at 99% confidence
    var, es = fhs_var_es(sigma_next, dummy_resid, alpha=0.01, mu=0.0)
    
    # 4. Assert the mathematical invariant
    assert es >= var, f"ES ({es}) must be >= VaR ({var})"
    assert var > 0, "VaR should be reported as a positive loss magnitude"
    
    print("Test passed: ES is correctly calculated as greater than VaR.")