"""Seeded defect: the drift sign is flipped.

``dX = theta*(X - mu) dt + sigma dW`` instead of ``theta*(mu - X)``. Every path
is finite, the scheme converges strongly at the right rate under CRN (it is still
Euler-Maruyama), and nothing structural is violated -- the solver has simply
integrated the wrong equation. Dynkin's identity is built from the *spec's*
drift, so it is the check that sees it.
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
        X = X + p["theta"] * (X - p["mu"]) * dt + p["sigma"] * dW[:, n]   # sign flipped
        for k, phi in (observables or {}).items():
            now = phi(X)
            acc[k] += 0.5 * (prev[k] + now) * dt
            prev[k] = now
    out = {"terminal_paths": X, "empirical_mean": float(np.mean(X)),
           "empirical_variance": float(np.var(X, ddof=1)), "dt": dt}
    if observables:
        out["path_integrals"] = acc
    return out
