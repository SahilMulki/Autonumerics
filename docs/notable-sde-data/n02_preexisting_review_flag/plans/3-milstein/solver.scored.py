"""Kubo oscillator (Stratonovich) -- Milstein scheme on the equivalent Ito form.

Stratonovich:  dX = a J X dt + sigma J X o dW,   J = [[0,-1],[1,0]]
Ito:           dX = (a J - 0.5 sigma^2 I) X dt + sigma J X dW
Milstein:      + 0.5 (dg/dX) g (dW^2 - dt),  (dg/dX) g = -sigma^2 X   (single channel,
               so the noise is commutative and no Levy area appears).
All coefficients are evaluated at X_n.
"""
import numpy as np

A = 1.0
SIGMA = 1.0
X0 = 1.0
Y0 = 0.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))   # one block, path-major
    else:
        dW = np.asarray(dW, dtype=float)
        if dW.ndim == 3:            # (num_paths, Nt, 1) -> (num_paths, Nt)
            dW = dW[:, :, 0]
        num_paths = dW.shape[0]
        Nt = dW.shape[1]
        dt = T / Nt

    X = np.empty((num_paths, 2), dtype=float)
    X[:, 0] = X0
    X[:, 1] = Y0
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}
    s2 = SIGMA ** 2

    for n in range(Nt):
        X_prev = X
        x1 = X_prev[:, 0]
        x2 = X_prev[:, 1]
        dw = dW[:, n]
        corr = dw * dw - dt
        X = np.empty_like(X_prev)
        X[:, 0] = (x1 + (-A * x2 - 0.5 * s2 * x1) * dt - SIGMA * x2 * dw
                   - 0.5 * s2 * x1 * corr)
        X[:, 1] = (x2 + (A * x1 - 0.5 * s2 * x2) * dt + SIGMA * x1 * dw
                   - 0.5 * s2 * x2 * corr)
        for k, phi in obs.items():
            acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt

    out = {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "num_paths": int(num_paths),
        "dt": float(dt),
        "T": float(T),
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    M, dt0, T0 = 2000, 0.04, 1.0
    res = solve_sde(num_paths=M, dt=dt0, T=T0, seed=42)
    print("Empirical mean:    ", res["empirical_mean"])
    print("Empirical variance:", res["empirical_variance"])
    print("Exact mean:         [0.32770991402245986, 0.5103779515445728]")
    print("Exact variance:     [0.36444653725532805, 0.26767402157322967]")
    amp = np.mean(np.abs(np.hypot(res["terminal_paths"][:, 0], res["terminal_paths"][:, 1]) - 1.0))
    print("Mean |  |X(T)| - 1 |:", amp)

    # explicit-dW hook, with strong error against the exact pathwise rotation
    Nt = 25
    dW = np.sqrt(T0 / Nt) * np.random.default_rng(42).standard_normal((M, Nt))
    r = solve_sde(M, dt0, T0, dW=dW)
    th = A * T0 + SIGMA * dW.sum(axis=1)
    ex = np.stack([np.cos(th), np.sin(th)], axis=1)
    rel = np.sqrt(np.mean((r["terminal_paths"] - ex) ** 2)) / np.sqrt(np.mean(ex ** 2))
    print("dW hook: dt =", r["dt"], " rel strong error vs exact (check only):", rel)

    # observables hook
    o = solve_sde(M, dt0, T0, observables={"x": lambda a: a[:, 0]})
    print("obs hook:", o["path_integrals"]["x"].shape)
