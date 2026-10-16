"""Correct Milstein on the GBM canary: the control for ``sde_milstein_post_step``.

The correction at the **pre-step** state, which is the only difference between the
two files. Strong order 1, measured, against the spec's
``expected_strong_order_milstein`` of 1.0 and the two-sided band a declared
``scheme_family`` selects.
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
        X = (X + p["mu"] * X * dt + p["sigma"] * X * inc
             + 0.5 * p["sigma"] ** 2 * X * (inc ** 2 - dt))
        for k, phi in (observables or {}).items():
            now = phi(X)
            acc[k] += 0.5 * (prev[k] + now) * dt
            prev[k] = now
    out = {"terminal_paths": X, "empirical_mean": float(np.mean(X)),
           "empirical_variance": float(np.var(X, ddof=1)), "dt": dt}
    if observables:
        out["path_integrals"] = acc
    return out
