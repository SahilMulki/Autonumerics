"""N09 geometric Asian at the textbook set: the price, E[A] and Var[A] of the continuous
average and of its three discrete surrogates at n = 25 steps -- all in closed form,
since each ln A is Gaussian with a variance that is a quadratic form in the Brownian
covariance. No simulation. Left-point sum: price -3.39%, Var[A] -6.14%
trapezoid:
-0.018%, -0.041%. (The plan's first draft had the sign of the left-point bias wrong.)
`uv run python docs/notable-sde-data/n09_geometric_asian.py`
"""
import numpy as np
from scipy.stats import norm

S0, K, r, sig, T = 100.0, 100.0, 0.05, 0.2, 1.0


def price(mean_ln, var_ln):
    EA = np.exp(mean_ln + var_ln / 2)
    sd = np.sqrt(var_ln)
    d1 = (mean_ln - np.log(K) + var_ln) / sd
    return np.exp(-r * T) * (EA * norm.cdf(d1) - K * norm.cdf(d1 - sd)), EA, (np.exp(var_ln) - 1) * EA ** 2


def gauss(ts, w=None):
    w = np.full(len(ts), 1.0 / len(ts)) if w is None else w
    return (np.log(S0) + (r - sig ** 2 / 2) * (w * ts).sum(),
            sig ** 2 * (w[:, None] * w[None, :] * np.minimum.outer(ts, ts)).sum())


n = 25
t = np.linspace(0, T, n + 1)
wt = np.full(n + 1, 1.0 / n)
wt[0] = wt[-1] = 0.5 / n
cases = {"exact (Kemna-Vorst)": (np.log(S0) + (r - sig ** 2 / 2) * T / 2, sig ** 2 * T / 3),
         "left-point": gauss(t[:-1]), "right-point": gauss(t[1:]), "trapezoid": gauss(t, wt)}
p0, EA0, VA0 = price(*cases["exact (Kemna-Vorst)"])
for k, (m, v) in cases.items():
    p, EA, VA = price(m, v)
    print(f"{k:22s} price {p:.6f} ({(p / p0 - 1):+.3%})  E[A] {EA:.5f} ({(EA / EA0 - 1):+.3%})  "
          f"Var[A] {VA:.4f} ({(VA / VA0 - 1):+.3%})")
