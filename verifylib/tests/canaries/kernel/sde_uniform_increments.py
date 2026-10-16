"""Rademacher increments on additive-noise OU -- and **nothing catches it**. Finding 4.

A measured correction to the plan. ``docs/plan-sde-verification-gaps.md`` §4a lists
this canary with "Must be caught by: the distributional check (3b)" and "Today:
nothing sees it". After 3b, still nothing sees it on *this* problem, and the reason
is the central limit theorem rather than a gap in the check.

``dW = +-sqrt(dt)`` has exactly the right mean and variance, and on a linear
additive-noise SDE the terminal state is a weighted **sum** of ``Nt`` of them -- so
it converges to the Gaussian it is supposed to be. Measured, 50 000 paths,
standardized KS against the exact terminal law, with a Gaussian-increment control
run at the same ``dt``:

======  ========  =================  ===============  ==========  ==========
``Nt``  ``dt``    KS, Rademacher     KS, Gaussian     mean err    var err
======  ========  =================  ===============  ==========  ==========
1000    0.001     0.00233            0.00295          0.0011      0.0027
100     0.01      0.00296            0.00208          0.0106      0.0090
25      0.04      0.00572            0.00215          0.0439      0.0294
10      0.1       0.01324            0.00218          0.1179      0.0927
4       0.25      0.09551            0.00261          0.3177      0.2698
======  ========  =================  ===============  ==========  ==========

The bar is 0.00607. At the ``dt`` any live spec uses the Rademacher statistic is
**indistinguishable from the control** -- 0.00233 against 0.00295, both well under
the bar. It only separates from ``dt >= 0.04``, by which point the moment errors are
4.4% and 2.9% and the moment check is about to fail anyway.

So this file asserts the limitation rather than a catch, which is the honest thing
for a canary to do: the terminal law of this process genuinely *is* Gaussian to
within the test's resolution, and a check that rejected it would be wrong.

**Where the same defect is caught**: on a *Milstein* plan.
``dW**2 - dt`` is identically zero under Rademacher increments, so the correction
vanishes and the self-convergence order reads 1.895 against a declared 1.0 --
rejected by the two-sided band. See ``sde_milstein_rademacher.py``. The instrument
is the order study, not the distributional check.
"""
import numpy as np

PARAMS = {"X_0": 2.0, "theta": 1.5, "mu": 0.0, "sigma": 0.5}


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None, observables: dict | None = None) -> dict:
    p = PARAMS
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        # The defect: a two-point law with the right mean and variance.
        dW = np.sqrt(dt) * rng.choice([-1.0, 1.0], size=(num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt
    X = np.full(num_paths, p["X_0"], dtype=float)
    acc = {k: np.zeros(num_paths) for k in (observables or {})}
    prev = {k: phi(X) for k, phi in (observables or {}).items()}
    for n in range(Nt):
        X = X + p["theta"] * (p["mu"] - X) * dt + p["sigma"] * dW[:, n]
        for k, phi in (observables or {}).items():
            now = phi(X)
            acc[k] += 0.5 * (prev[k] + now) * dt
            prev[k] = now
    out = {"terminal_paths": X, "empirical_mean": float(np.mean(X)),
           "empirical_variance": float(np.var(X, ddof=1)), "dt": dt}
    if observables:
        out["path_integrals"] = acc
    return out
