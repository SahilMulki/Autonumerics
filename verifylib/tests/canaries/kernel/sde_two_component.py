"""Honest 2-D correlated GBM: 2 state components, 2 noise channels. Finding 2.

Not a defect -- a **control**. It follows ``agents/solver-sde.md``'s contract
exactly, indexing ``dW[:, n, 0]`` and ``dW[:, n, 1]`` for the two channels, which is
what every multi-noise solver in ``workspace/`` does. Before the declared
``noise_dimension`` reached ``ladder_spec``, the CRN ladder handed it a
``(num_paths, Nt)`` array and it raised ``IndexError: too many indices for array``
-- reported by the kernel as *"the CRN level at dt=0.02 crashed"*, which reads in a
review as a solver defect.

So this canary asserts two things no other one can: that the ladder **runs** at
``m = 2``, and that the recovered terminal correlation is the spec's ``rho``.
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
        if dW.ndim == 2:                       # one channel supplied for a 2-channel SDE
            raise ValueError("this solver needs dW of shape (num_paths, Nt, 2); "
                             f"got {dW.shape}")
    # The correlation lives in the increments: Cholesky of [[1, rho], [rho, 1]].
    lower = np.sqrt(max(1.0 - p["rho"] ** 2, 0.0))
    X = np.full(num_paths, p["X_0"], dtype=float)
    Y = np.full(num_paths, p["Y_0"], dtype=float)
    acc = {k: np.zeros(num_paths) for k in (observables or {})}
    state = np.stack([X, Y], axis=1)
    prev = {k: phi(state) for k, phi in (observables or {}).items()}
    for n in range(Nt):
        z1 = dW[:, n, 0]
        z2 = p["rho"] * dW[:, n, 0] + lower * dW[:, n, 1]
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
