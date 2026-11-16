"""N08 the 3/2 model at Drimus's S&P 500 fit (kappa=22.84, theta=0.218, eps=8.56):
the moments of U = 1/V (CIR) at T = 0.1 for four starting variances, and five schemes at
dt = 0.004 -- an exact CIR step on U, full-truncation Euler on U, and explicit Euler on
V raw / clamped / reflected, read through 1/V. At the published V0 = 0.06 the mean of U
sits 6.2% from its initial value and 8.3% from the stationary law (1.2x and 1.7x the
gate); V0 = 0.15 is chosen, where it sits 74% and 40% away and the reflected Euler on V
fails the mean gate by 3.6x and the variance gate by 39x.
`uv run python docs/notable-sde-data/n08_three_halves.py`
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmark"))
import problems as P  # noqa: E402

kappa, theta, eps = 22.84, 0.218, 8.56
a, b, sig = kappa * theta, kappa, eps
k_u, th_u = a, (b + sig ** 2) / a
d = 4 * k_u * th_u / sig ** 2
T, dt, nt, n = 0.1, 0.004, 25, 100000
print(f"U = 1/V is CIR with rate {k_u:.4f}, level {th_u:.4f}, vol {sig}; Feller ratio {d / 2:.3f}")
for V0 in (0.06, 0.15, 0.2, 0.3):
    U0 = 1 / V0
    m = P.cir_moments(T, U0, k_u, th_u, sig)
    vst = th_u * sig ** 2 / (2 * k_u)
    print(f"V0={V0}: E[U] {m['mean']:.4f} (IC {abs(m['mean'] / U0 - 1):.1%}, stat {abs(m['mean'] / th_u - 1):.1%})"
          f"  Var[U] {m['variance']:.3f} (stat {abs(m['variance'] / vst - 1):.1%})")

    def em_V(mode, V0=V0):
        rng = np.random.default_rng(42)
        V = np.full(n, V0)
        for _ in range(nt):
            dW = rng.standard_normal(n) * np.sqrt(dt)
            V = V + V * (a - b * V) * dt + sig * np.abs(V) ** 1.5 * dW
            if mode == "clamp":
                V = np.maximum(V, 1e-10)
            if mode == "reflect":
                V = np.abs(V)
        return V

    def ft_U(U0=U0):
        rng = np.random.default_rng(42)
        U = np.full(n, U0)
        for _ in range(nt):
            dW = rng.standard_normal(n) * np.sqrt(dt)
            Up = np.maximum(U, 0)
            U = U + k_u * (th_u - Up) * dt - sig * np.sqrt(Up) * dW
        return np.maximum(U, 0)

    def ex_U(U0=U0):
        rng = np.random.default_rng(42)
        U = np.full(n, U0)
        cc = sig ** 2 * (1 - np.exp(-k_u * dt)) / (4 * k_u)
        for _ in range(nt):
            U = cc * rng.noncentral_chisquare(d, U * np.exp(-k_u * dt) / cc)
        return U

    with np.errstate(all="ignore"):
        rows = (("exact CIR step on U", ex_U()), ("full-trunc EM on U", ft_U()),
                ("EM on V raw -> 1/V", 1 / em_V("raw")), ("EM on V clamp -> 1/V", 1 / em_V("clamp")),
                ("EM on V reflect -> 1/V", 1 / em_V("reflect")))
    for name, U in rows:
        fin = np.isfinite(U)
        mm, vv = U[fin].mean(), U[fin].var(ddof=1)
        print(f"   {name:24s} finite {fin.mean():.4f} mean {mm:12.4f} ({(mm / m['mean'] - 1):+.2%}) "
              f"var {vv:14.3f} ({(vv / m['variance'] - 1):+.2%})")
