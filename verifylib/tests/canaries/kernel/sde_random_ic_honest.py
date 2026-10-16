"""A correct solver with a **random** initial condition. Finding 5.

Not a defect -- the canary that asserts a correct solver is *not* failed, which is
the kind this repo has needed repeatedly (``plan-no-closed-form.md``'s I9, I10 and
I15 are all false failures on correct work).

``dX = -X**5 dt`` with ``X(0) = xi ~ Normal(0, sigma_bar**2)``. The spec declares no
``X_0``, because there is no single initial state. The Dynkin identity needs
``E[phi(X_0)]``; the kernel read ``parameters.get("X_0", 0.0)`` and formed it with
``phi(0)``, so for ``phi = X**2`` it used 0 where it needed ``sigma_bar**2``. The
residual then carried a fixed offset at every level, survived the Richardson
extrapolation, and landed the rubric's 3 row -- *"the wrong equation integrated
correctly"* -- on this file.
"""
import numpy as np

PARAMS = {"sigma_bar": 0.3333333333333333}


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
    # The initial condition is drawn from its own stream, so it is the same sample at
    # every level of a CRN ladder -- the increments are what the ladder coarsens.
    X = p["sigma_bar"] * np.random.default_rng(seed + 90000).standard_normal(num_paths)
    acc = {k: np.zeros(num_paths) for k in (observables or {})}
    prev = {k: phi(X) for k, phi in (observables or {}).items()}
    for n in range(Nt):
        X = X - X ** 5 * dt            # deterministic drift; the noise is zero here
        for k, phi in (observables or {}).items():
            now = phi(X)
            acc[k] += 0.5 * (prev[k] + now) * dt
            prev[k] = now
    out = {"terminal_paths": X, "empirical_mean": float(np.mean(X)),
           "empirical_variance": float(np.var(X, ddof=1)), "dt": dt}
    if observables:
        out["path_integrals"] = acc
    return out
