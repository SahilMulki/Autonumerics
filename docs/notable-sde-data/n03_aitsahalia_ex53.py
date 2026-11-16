"""N03 Ait-Sahalia, Example 5.3 of arXiv:2401.02596 ("borrowed from [10, SET II]"): the
critical case rho = 2, theta = 3/2 of the original 1996 model, a_{-1}=2, a_0=3, a_1=4,
a_2=7, sigma=sqrt(2), x0=1. The shipped entry.

Three measurements: (1) the Kolmogorov truth's dependence on the right end of the
interval -- the stationary tail is the power law x^-10, so hi must reach 12 before the
variance settles to 1e-6
(2) distance from x0 and from the stationary law at four
horizons, which fixes T = 0.1 (T = 0.25 is 3.9% from stationary on the mean)
(3) the
scheme calibration at dt = 0.004, 0.01, 0.02, 0.025: backward Euler against explicit
Euler raw and clamped. The decision has consequences only at dt >= 0.02.
`uv run python docs/notable-sde-data/n03_aitsahalia_ex53.py`
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n03_n06_kolmogorov as K  # noqa: E402

c = dict(a_m1=2.0, a_0=3.0, a_1=4.0, a_2=7.0, rho=2.0, sigma=np.sqrt(2.0), theta=1.5, x0=1.0)


def b(x):
    return c["a_m1"] / x - c["a_0"] + c["a_1"] * x - c["a_2"] * x ** c["rho"]


def s(x):
    return c["sigma"] * x ** c["theta"]


def logw(x):
    return (-c["a_m1"] / 3 * x ** -3 + c["a_0"] / 2 * x ** -2 - c["a_1"] / x
            - c["a_2"] * np.log(x)) - np.log(2 * x ** 3)


xs = np.linspace(0.05, 20.0, 400001)
lw = logw(xs)
w = np.exp(lw - lw.max())
Z = np.trapezoid(w, xs)
ms = np.trapezoid(xs * w, xs) / Z
vs = np.trapezoid(xs ** 2 * w, xs) / Z - ms ** 2
print(f"stationary mean {ms:.6f} var {vs:.6f}; rel density at x=0.1: "
      f"{np.exp(logw(0.1) - lw.max()):.1e}, at x=3: {np.exp(logw(3.0) - lw.max()):.1e}")

print("\n--- (1) right-end dependence at T = 0.1 (lo = 0.1) ---")
for n, hi in ((384, 4.0), (512, 5.0), (640, 6.0), (768, 8.0), (896, 10.0), (1024, 12.0)):
    m1, m2 = K.kolmogorov_general(b, s, 1.0, 0.1, 0.1, hi, N=n)
    print(f"  N={n} hi={hi}: mean {m1:.10f} var {m2 - m1 ** 2:.10f}")
print("  left-end independence (hi = 3, N = 256):",
      [round(K.kolmogorov_general(b, s, 1.0, 0.1, lo, 3.0, N=256)[0], 10) for lo in (0.1, 0.15, 0.2)])

print("\n--- (2) horizons ---")
for T in (0.1, 0.25, 0.5, 1.0):
    m1, m2 = K.kolmogorov_general(b, s, 1.0, T, 0.1, 12.0, N=1024)
    v = m2 - m1 ** 2
    print(f"  T={T}: mean {m1:.8f} var {v:.8f}  from x0 {abs(m1 - 1):.1%}  "
          f"from stationary {abs(m1 / ms - 1):.1%} / {abs(v / vs - 1):.1%}")

print("\n--- (3) schemes at T = 0.1, 100k paths ---")
T = 0.1
m1, m2 = K.kolmogorov_general(b, s, 1.0, T, 0.1, 12.0, N=1024)
v = m2 - m1 ** 2
n = 100000


def bem(dt, nt, rng):
    x = np.full(n, 1.0)
    for _ in range(nt):
        rhs = x + s(x) * np.sqrt(dt) * rng.standard_normal(n)
        xn = np.maximum(x, 0.1)
        for _ in range(40):
            F = xn - b(xn) * dt - rhs
            Fp = 1 - dt * (-c["a_m1"] / xn ** 2 + c["a_1"] - c["a_2"] * c["rho"] * xn ** (c["rho"] - 1))
            xn = np.maximum(xn - F / Fp, 1e-6)
        x = xn
    return x


def em(dt, nt, rng, clamp):
    x = np.full(n, 1.0)
    for _ in range(nt):
        xp = np.maximum(x, 1e-12) if clamp else x
        x = x + b(xp) * dt + c["sigma"] * np.abs(xp) ** c["theta"] * np.sqrt(dt) * rng.standard_normal(n)
        if clamp:
            x = np.maximum(x, 1e-3)
    return x


for dt in (0.004, 0.01, 0.02, 0.025):
    nt = round(T / dt)
    for name, fn in (("BEM", lambda r, dt=dt, nt=nt: bem(dt, nt, r)),
                     ("EM raw", lambda r, dt=dt, nt=nt: em(dt, nt, r, False)),
                     ("EM clamp", lambda r, dt=dt, nt=nt: em(dt, nt, r, True))):
        with np.errstate(all="ignore"):
            X = fn(np.random.default_rng(42))
        fin = np.isfinite(X)
        m = X[fin].mean()
        vv = X[fin].var(ddof=1)
        print(f"  dt={dt:<6} {name:9s} finite {fin.mean():.4f} min {X[fin].min():9.3f} "
              f"mean {m:.5f} ({(m / m1 - 1):+.2%}, se {np.sqrt(vv / n) / m1:.2%}) "
              f"var {vv:.5f} ({(vv / v - 1):+.2%})")
