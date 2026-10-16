"""The same 2-D GBM with **independent** increments. Finding 3.

``sde_gbm_2d_high_corr``'s spec note states the consequence in as many words:
*"This is the ONLY place rho appears in any terminal statistic. A solver that drops
the correlation entirely (independent increments) still passes both per-component
moment checks, so this check is the only thing that detects it."*

One line differs from ``sde_two_component.py``: ``z2`` is the second channel
unmixed. Both marginals are exactly right -- ``rho`` does not appear in either
marginal law -- so every per-component mean and variance lands inside tolerance and
only the cross moment can see it.
"""
import numpy as np

PARAMS = {"X_0": 1.0, "Y_0": 1.0, "mu1": 0.1, "sigma1": 0.3,
          "mu2": 0.1, "sigma2": 0.3, "rho": 0.9}


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None, observables: dict | None = None) -> dict:
    p = PARAMS
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        dW = np.sqrt(dt) * np.random.default_rng(seed).standard_normal((num_paths, Nt, 2))
    else:
        Nt = dW.shape[1]
        dt = T / Nt
    X = np.full(num_paths, p["X_0"], dtype=float)
    Y = np.full(num_paths, p["Y_0"], dtype=float)
    acc = {k: np.zeros(num_paths) for k in (observables or {})}
    state = np.stack([X, Y], axis=1)
    prev = {k: phi(state) for k, phi in (observables or {}).items()}
    for n in range(Nt):
        z1, z2 = dW[:, n, 0], dW[:, n, 1]      # <- the defect: rho is never applied
        X = X + p["mu1"] * X * dt + p["sigma1"] * X * z1
        Y = Y + p["mu2"] * Y * dt + p["sigma2"] * Y * z2
        state = np.stack([X, Y], axis=1)
        for k, phi in (observables or {}).items():
            now = phi(state)
            acc[k] += 0.5 * (prev[k] + now) * dt
            prev[k] = now
    out = {"terminal_paths": state,
           "empirical_mean": [float(np.mean(X)), float(np.mean(Y))],
           "empirical_variance": [float(np.var(X, ddof=1)), float(np.var(Y, ddof=1))],
           "dt": dt}
    if observables:
        out["path_integrals"] = acc
    return out
