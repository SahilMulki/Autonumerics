"""A bare ``np.maximum(X, 0)`` on the state after the update. Finding 10.

The pattern ``verification_manual.md`` §21 and ``agents/solver-sde.md`` both warn
about and neither checked for. Written on the OU canary, where the process is *not*
positive -- so the clip is both a scheme defect and a visible moment error, which is
what makes it a usable canary: the scan must see it whether or not the moments do.

The distinction the scan turns on is in this file twice. ``np.maximum`` inside the
diffusion coefficient would be full truncation and endorsed; the statement
``X = np.maximum(X, 0.0)`` after the update is the defect.
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
        X = np.maximum(X, 0.0)        # <- the defect: the state, after the update
        for k, phi in (observables or {}).items():
            now = phi(X)
            acc[k] += 0.5 * (prev[k] + now) * dt
            prev[k] = now
    out = {"terminal_paths": X, "empirical_mean": float(np.mean(X)),
           "empirical_variance": float(np.var(X, ddof=1)), "dt": dt}
    if observables:
        out["path_integrals"] = acc
    return out
