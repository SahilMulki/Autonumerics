"""Correct Euler-Maruyama on the GBM canary. The control for the shape check.

Criterion 16's configuration: a *correct* weak-order-1 scheme at the ``dt`` and path
count the live specs use. An un-standardized KS test against the exact lognormal law
resolves this scheme's own ``O(dt)`` bias and rejects it; the standardized one does
not, because standardizing both sides removes exactly the location and scale error
that the ``O(dt)`` weak error mostly is.
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
        X = X + p["mu"] * X * dt + p["sigma"] * X * dW[:, n]
        for k, phi in (observables or {}).items():
            now = phi(X)
            acc[k] += 0.5 * (prev[k] + now) * dt
            prev[k] = now
    out = {"terminal_paths": X, "empirical_mean": float(np.mean(X)),
           "empirical_variance": float(np.var(X, ddof=1)), "dt": dt}
    if observables:
        out["path_integrals"] = acc
    return out
