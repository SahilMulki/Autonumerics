"""Step-2 check for plan-sciml-benchmark.md §2a: does each closed form *solve* its Ito SDE?

A correctly transcribed formula paired with the wrong equation still type-checks, so a textual
diff against the library source is not enough. This runs the check that would catch it: a
common-random-number strong-convergence ladder. One fine Brownian path per path-index is drawn
once and aggregated down to each coarser level, Milstein is run at every level, and the terminal
state is compared to the analytic formula evaluated at the *same* terminal W.

Reading: a correct (SDE, analytic) pair gives strong order ~1; a mismatched pair plateaus at O(1).
S06-S08 instead return a strong error of exactly 0.0 -- that is §2a finding 1 (the terminal state
is the absorbing constant -1 at SciML's horizon), not a pass. See stiffquad_degeneracy.py.

numpy only; no repo imports, no Julia. Seeded, so the printed numbers are the ones in §2a.
"""

import numpy as np


def milstein(f, g, gp, u0, T, dt, dW):
    """Scalar Milstein over supplied increments. dW is (paths, nsteps); returns terminal state."""
    u = np.full(dW.shape[0], float(u0))
    t = 0.0
    for i in range(dW.shape[1]):
        dw = dW[:, i]
        gu = g(u, t)
        u = u + f(u, t) * dt + gu * dw + 0.5 * gu * gp(u, t) * (dw * dw - dt)
        t += dt
    return u


def stiff_quad(alpha, beta):
    """SciML's quadratic-stiffness problem at one (alpha, beta), Ito form, from the library source:
    f = -(alpha + beta^2 u)(1 - u^2), g = beta(1 - u^2), u0 = 0.5, tspan (0, 3)."""

    def f(u, t):
        return -(alpha + beta * beta * u) * (1 - u**2)

    def g(u, t):
        return beta * (1 - u**2)

    def gp(u, t):
        return -2 * beta * u

    def exact(u0, T, W):
        e = np.exp(np.clip(-2 * alpha * T + 2 * beta * W, -745, 709))
        return ((1 + u0) * e + u0 - 1) / ((1 + u0) * e - u0 + 1)

    return dict(f=f, g=g, gp=gp, u0=0.5, T=3.0, exact=exact)


def cbrt_pow(u, n):
    """Real (u^(1/3))^n -- S05's drift and diffusion are powers of a real cube root."""
    return np.cbrt(u) ** n


PROBS = {
    "S01 linear": dict(
        f=lambda u, t: 1.01 * u,
        g=lambda u, t: 0.87 * u,
        gp=lambda u, t: 0.87 * np.ones_like(u),
        u0=0.5,
        T=1.0,
        exact=lambda u0, T, W: u0 * np.exp(0.63155 * T + 0.87 * W),
        levels=[2**k for k in (6, 7, 8, 9)],
        paths=4000,
    ),
    "S02 additive": dict(
        f=lambda u, t: 0.05 / np.sqrt(1 + t) - u / (2 * (1 + t)),
        g=lambda u, t: 0.1 * 0.05 / np.sqrt(1 + t) * np.ones_like(u),
        gp=lambda u, t: np.zeros_like(u),
        u0=1.0,
        T=1.0,
        exact=lambda u0, T, W: u0 / np.sqrt(1 + T) + 0.05 * (T + 0.1 * W) / np.sqrt(1 + T),
        levels=[2**k for k in (6, 7, 8, 9)],
        paths=4000,
    ),
    "S03 cubic": dict(
        f=lambda u, t: -0.25 * u * (1 - u**2),
        g=lambda u, t: 0.5 * (1 - u**2),
        gp=lambda u, t: -u,
        u0=0.5,
        T=1.0,
        exact=lambda u0, T, W: ((1 + u0) * np.exp(W) + u0 - 1) / ((1 + u0) * np.exp(W) + 1 - u0),
        levels=[2**k for k in (6, 7, 8, 9)],
        paths=4000,
    ),
    "S04 wave": dict(
        f=lambda u, t: -0.01 * np.sin(u) * np.cos(u) ** 3,
        g=lambda u, t: 0.1 * np.cos(u) ** 2,
        gp=lambda u, t: -0.2 * np.cos(u) * np.sin(u),
        u0=1.0,
        T=1.0,
        exact=lambda u0, T, W: np.arctan(0.1 * W + np.tan(u0)),
        levels=[2**k for k in (6, 7, 8, 9)],
        paths=4000,
    ),
    "S05 nltest": dict(
        f=lambda u, t: (1 / 3) * cbrt_pow(u, 1) + 6 * cbrt_pow(u, 2),
        g=lambda u, t: cbrt_pow(u, 2),
        gp=lambda u, t: (2 / 3) / np.cbrt(np.where(np.abs(u) < 1e-12, 1e-12, u)),
        u0=1.0,
        T=10.0,
        exact=lambda u0, T, W: (2 * T + 1 + W / 3) ** 3,
        levels=[2**k for k in (10, 11, 12, 13)],
        paths=2000,
    ),
    # SciML's three parameter sets come from StiffSDE/QuadraticStiffness.jmd, NOT the library,
    # whose own prob_sde_stiffquadito carries p = (1.0, 1.0). See §2a.3.
    "S06 stiffquad (50, 1)": dict(
        stiff_quad(50.0, 1.0), levels=[2**k for k in (9, 10, 11, 12)], paths=2000
    ),
    "S07 stiffquad (500, 1)": dict(
        stiff_quad(500.0, 1.0), levels=[2**k for k in (11, 12, 13, 14)], paths=1000
    ),
    "S08 stiffquad (5000, 70)": dict(
        stiff_quad(5000.0, 70.0), levels=[2**k for k in (18, 19, 20)], paths=200
    ),
}


def main():
    print(f"{'problem':26s} {'dt':>12s} {'rel strong err':>15s} {'order':>7s}")
    for name, P in PROBS.items():
        rng = np.random.default_rng(20261110)
        nfine = max(P["levels"])
        dWf = rng.standard_normal((P["paths"], nfine)) * np.sqrt(P["T"] / nfine)
        ex = P["exact"](P["u0"], P["T"], dWf.sum(axis=1))
        denom = np.sqrt(np.mean(ex**2))
        prev = None
        for n in P["levels"]:
            dW = dWf.reshape(P["paths"], n, nfine // n).sum(axis=2)
            num = milstein(P["f"], P["g"], P["gp"], P["u0"], P["T"], P["T"] / n, dW)
            err = np.sqrt(np.mean((num - ex) ** 2)) / denom
            if prev is None or prev == 0.0 or err == 0.0:
                order = "     --"
            else:
                order = f"{np.log2(prev / err):7.2f}"
            print(f"{name:26s} {P['T'] / n:12.3e} {err:15.3e} {order}")
            prev = err
        print()


if __name__ == "__main__":
    main()
