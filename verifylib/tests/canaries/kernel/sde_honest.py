"""Euler-Maruyama for the OU canary. Correct, and honours ``dW`` and ``observables``.

The seeded defect beside it is not in this file: it is the *path count*. §14's
three-way outcome exists because a point estimate landing inside a threshold band
proves nothing when its own error bar is wider than the band, and the fix for that
is more paths, not a different scheme.
"""
import numpy as np

PARAMS = {"X_0": 2.0, "theta": 1.5, "mu": 0.0, "sigma": 0.5}


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
