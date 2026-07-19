"""The two naive baselines FHS is designed to beat: plain HS and Normal-parametric.

Same signature idea as the FHS engine (window of returns in, VaR/ES out) so they
drop into the identical rolling harness. Apples-to-apples: only the method differs.
Returns are in percent; VaR/ES are positive loss magnitudes.
"""

import numpy as np
from scipy import stats


def hs_var_es(window_returns: np.ndarray, alpha: float = 0.01) -> tuple[float, float]:
    """Plain Historical Simulation: empirical quantile of the raw returns.

    Treats every day in the window as equally likely, so it reacts slowly to
    volatility changes and produces clustered exceptions. This is the weakness
    FHS fixes by re-scaling residuals with today's sigma.
    """
    r = np.asarray(window_returns, dtype=float)
    q = float(np.quantile(r, alpha, method="lower"))
    tail = r[r <= q]
    return -q, -float(tail.mean())


def normal_var_es(window_returns: np.ndarray, alpha: float = 0.01) -> tuple[float, float]:
    """Normal-parametric: assume returns ~ Normal(mu, sigma), read the quantile.

    Thin Normal tails underestimate extreme losses, so this typically takes far
    too many exceptions. VaR = -(mu + sigma * z_alpha), z_alpha = Phi^-1(alpha).
    """
    r = np.asarray(window_returns, dtype=float)
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    z_a = float(stats.norm.ppf(alpha))            # e.g. -2.326 at alpha=0.01
    var = -(mu + sd * z_a)
    # Normal ES has a closed form: mu - sd * phi(z_a)/alpha
    es = -(mu - sd * float(stats.norm.pdf(z_a)) / alpha)
    return var, es