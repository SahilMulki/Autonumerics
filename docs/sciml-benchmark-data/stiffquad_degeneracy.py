"""§2a.1 of plan-sciml-benchmark.md: the stiff-quadratic terminal state is a constant.

SciML's quadratic-stiffness problem has the exact solution

    u(t) = ((1+u0)E + u0 - 1) / ((1+u0)E - u0 + 1),   E = exp(-2*alpha*t + 2*beta*W_t)

and u = -1 is both the attracting limit of E -> 0 and an *absorbing* point of the equation itself
(f and g both carry the factor 1 - u^2). At every one of the three parameter sets the notebook
uses, E has underflowed long before SciML's T = 3, so the terminal state is -1 for every path.

A terminal-graded entry on these problems is therefore vacuous: `-np.ones(num_paths)` scores a
perfect error at any dt without touching the increments. The fix is to grade in the transient --
roughly T = 0.05, 0.005, 5e-4 -- which this table locates.

numpy only; seeded, so the printed numbers are the ones in §2a.1.
"""

import numpy as np

U0 = 0.5
PARAMS = ((1.0, 1.0), (50.0, 1.0), (500.0, 1.0), (5000.0, 70.0))
TIMES = (1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.25, 1.0, 3.0)
PATHS = 200_000


def exact_terminal(alpha, beta, t, W):
    e = np.exp(np.clip(-2 * alpha * t + 2 * beta * W, -745, 709))
    return ((1 + U0) * e + U0 - 1) / ((1 + U0) * e - U0 + 1)


def main():
    rng = np.random.default_rng(7)
    for alpha, beta in PARAMS:
        tag = "the library's own p" if (alpha, beta) == (1.0, 1.0) else "a notebook parameter set"
        print(f"--- alpha={alpha:g} beta={beta:g}  ({tag}) ---")
        for t in TIMES:
            u = exact_terminal(alpha, beta, t, rng.standard_normal(PATHS) * np.sqrt(t))
            m, s = u.mean(), u.std()
            flag = "   <-- bit-identical constant" if s == 0.0 else ""
            print(f"   t={t:<8g} mean={m:+.6f}  std={s:.3e}  std/|mean|={s / abs(m):.2e}{flag}")
        print()


if __name__ == "__main__":
    main()
