"""The Milstein correction at ``X_{n+1}`` -- and the strong order does **not** move.

A **false-positive guard**, and a correction to the plan that asked for it.
``docs/plan-sde-verification-gaps.md`` §4a lists this defect as one the strong-order
gate must catch. Measured against the exact GBM solution at ``M = 40 000`` over four
``dt`` levels, it does not and cannot:

===================================  ==============  ================
scheme                               strong order    E|err| at dt=8e-3
===================================  ==============  ================
correct Milstein                     0.998           2.155e-04
**correction at the post-step X**    **0.998**       **4.482e-04**
correction omitted                   0.504           8.488e-03
correction with the wrong sign       0.505           1.700e-02
===================================  ==============  ================

The reason is arithmetic rather than empirical. The two schemes differ per step by
``0.5 sigma^2 (X_{n+1} - X_n)(dW^2 - dt)``, whose leading term is
``0.5 sigma^3 X dW (dW^2 - dt) = O(dt^{3/2})`` with mean zero; summing ``T/dt``
near-independent such terms gives ``O(sqrt(1/dt) * dt^{3/2}) = O(dt)``. An ``O(dt)``
perturbation of an ``O(dt)`` scheme leaves the **rate** alone and only doubles the
constant -- which is exactly what the table shows.

So this file asserts that the gate does *not* fire here. ``agents/solver-sde.md``
lists three Milstein defects together -- *"the correction term is missing, has the
wrong sign, or is being applied with the post-step X instead of the pre-step one"* --
and the order study separates the first two from a correct scheme and the third from
nothing. Nothing else the kernel measures sees it either: the weak error is ``O(dt)``
either way, so both moments pass. It belongs on the plan's "what this does not fix"
list, and the honest canary for the gate having teeth is
``sde_milstein_no_correction.py``.
"""
import numpy as np

PARAMS = {"X_0": 1.0, "mu": 0.05, "sigma": 0.4}


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None, observables: dict | None = None) -> dict:
    p = PARAMS
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        dW = np.sqrt(dt) * np.random.default_rng(seed).standard_normal((num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt
    X = np.full(num_paths, p["X_0"], dtype=float)
    acc = {k: np.zeros(num_paths) for k in (observables or {})}
    prev = {k: phi(X) for k, phi in (observables or {}).items()}
    for n in range(Nt):
        inc = dW[:, n]
        euler = X + p["mu"] * X * dt + p["sigma"] * X * inc
        # The defect: the Ito correction is 0.5*g(X_n)*g'(X_n)*(dW^2 - dt) and uses
        # the PRE-step state. Here it is evaluated at the post-step one.
        X = euler + 0.5 * p["sigma"] ** 2 * euler * (inc ** 2 - dt)
        for k, phi in (observables or {}).items():
            now = phi(X)
            acc[k] += 0.5 * (prev[k] + now) * dt
            prev[k] = now
    out = {"terminal_paths": X, "empirical_mean": float(np.mean(X)),
           "empirical_variance": float(np.var(X, ddof=1)), "dt": dt}
    if observables:
        out["path_integrals"] = acc
    return out
