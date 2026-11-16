"""N04 Wright-Fisher with theta_1 = theta_2 = 1/2 (a = b = 1/4, sigma = 1): the truth
by the triangular moment hierarchy, distance from the initial condition and the
Beta(1/2, 1/2) stationary law, and the one-decision pair at dt = 0.04 -- Euler with a
clip to [0, 1] against the Lamperti chart Y = 2 arcsin(sqrt X) with reflection -- for
four starting points. x0 = 0.05 is chosen: the clipped scheme fails the mean gate by
1.7x there, the chart scheme clears it by 10x.
`uv run python docs/notable-sde-data/n04_wright_fisher.py`
"""
import numpy as np
from scipy.linalg import expm

a, b, sig, T = 0.25, 0.25, 1.0, 1.0
n, dt, nt = 100000, 0.04, 25


def truth(x0):
    M = np.zeros((3, 3))
    M[1, 0], M[1, 1] = a, -(a + b)
    M[2, 1], M[2, 2] = 2 * a + sig ** 2, -(2 * (a + b) + sig ** 2)
    y = expm(M * T) @ np.array([1.0, x0, x0 ** 2])
    return y[1], y[2] - y[1] ** 2


def em_clip(x0):
    rng = np.random.default_rng(42)
    X = np.full(n, x0)
    for _ in range(nt):
        dW = rng.standard_normal(n) * np.sqrt(dt)
        X = X + (a * (1 - X) - b * X) * dt + sig * np.sqrt(np.clip(X * (1 - X), 0, None)) * dW
        X = np.clip(X, 0, 1)
    return X


def lamperti(x0):
    rng = np.random.default_rng(42)
    Y = np.full(n, 2 / sig * np.arcsin(np.sqrt(x0)))
    L = np.pi / sig
    for _ in range(nt):
        X = np.sin(sig * Y / 2) ** 2
        q = np.sqrt(np.clip(X * (1 - X), 1e-300, None))
        f = (a * (1 - X) - b * X) / (sig * q) - 0.5 * sig * (1 - 2 * X) / (2 * q)
        Y = np.abs(Y + f * dt + rng.standard_normal(n) * np.sqrt(dt))
        Y = np.where(Y > L, 2 * L - Y, Y)
    return np.sin(sig * Y / 2) ** 2


al, be = 2 * a / sig ** 2, 2 * b / sig ** 2
vst = al * be / ((al + be) ** 2 * (al + be + 1))
for x0 in (0.02, 0.05, 0.1, 0.2):
    m1, v = truth(x0)
    print(f"x0={x0}: truth mean {m1:.6f} var {v:.6f}  from IC {abs(m1 / x0 - 1):.0%} "
          f"from stationary mean {abs(m1 / 0.5 - 1):.1%} / var {abs(v / vst - 1):.1%}")
    for name, X in (("EM+clip", em_clip(x0)), ("Lamperti+reflect", lamperti(x0))):
        m, vv = X.mean(), X.var(ddof=1)
        print(f"   {name:18s} mean {m:.5f} ({(m / m1 - 1):+.2%}, se {np.sqrt(vv / n) / m1:.2%}) "
              f"var {vv:.5f} ({(vv / v - 1):+.2%})")
