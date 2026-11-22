"""Stochastic Heun (Stratonovich predictor-corrector) for the Kubo oscillator.

Stratonovich form:  dX = a J X dt + sigma J X o dW,  J = [[0, -1], [1, 0]],  X(0) = (X_0, Y_0).
The explicit trapezoid predictor-corrector converges to the Stratonovich solution (strong
order 1.0 for a single, commutative noise channel).  The Stratonovich drift a*J*X is used
directly: the trapezoid in the diffusion generates the Ito correction itself.
"""
import numpy as np

A = 1.0
SIGMA = 1.0
X_0 = 1.0
Y_0 = 0.0


def _f(X):
    """Stratonovich drift a J X, shape (num_paths, 2)."""
    out = np.empty_like(X)
    out[:, 0] = -A * X[:, 1]
    out[:, 1] = A * X[:, 0]
    return out


def _g(X):
    """Diffusion sigma J X (single channel), shape (num_paths, 2)."""
    out = np.empty_like(X)
    out[:, 0] = -SIGMA * X[:, 1]
    out[:, 1] = SIGMA * X[:, 0]
    return out


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))
    else:
        dW = np.asarray(dW, dtype=float)
        if dW.ndim == 3:            # (num_paths, Nt, 1) -> single channel
            dW = dW[:, :, 0]
        Nt = dW.shape[1]
        dt = T / Nt

    X = np.zeros((num_paths, 2))
    X[:, 0] = X_0
    X[:, 1] = Y_0
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    for n in range(Nt):
        X_prev = X
        dw = dW[:, n][:, None]
        fx = _f(X_prev)
        gx = _g(X_prev)
        Xp = X_prev + fx * dt + gx * dw                            # predictor (Euler)
        X = X_prev + 0.5 * (fx + _f(Xp)) * dt + 0.5 * (gx + _g(Xp)) * dw   # corrector
        for k, phi in obs.items():
            acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt

    out = {
        "terminal_paths": X,
        "empirical_mean": [float(m) for m in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    res = solve_sde(num_paths=2000, dt=0.04, T=1.0, seed=42)
    print("Empirical mean:    ", res["empirical_mean"])
    print("Empirical variance:", res["empirical_variance"])
    print("Exact mean:         [0.32770991402246, 0.51037795154457]")
    print("Exact variance:     [0.36444653725533, 0.26767402157323]")
    amp = np.mean(np.abs(np.hypot(res["terminal_paths"][:, 0], res["terminal_paths"][:, 1]) - 1.0))
    print("Mean |amp-1|:      ", amp)

    dW = np.sqrt(0.04) * np.random.default_rng(0).standard_normal((1000, 25))
    r = solve_sde(1000, 0.04, 1.0, dW=dW)
    # strong error against exact rotation on the same increments (check only)
    th = A * 1.0 + SIGMA * dW.sum(axis=1)
    ex = np.stack([np.cos(th), np.sin(th)], axis=1)
    err = np.sqrt(np.mean((r["terminal_paths"] - ex) ** 2)) / np.sqrt(np.mean(ex ** 2))
    print("dW hook: dt =", r["dt"], " rel strong error vs exact =", err)

    o = solve_sde(1000, 0.04, 1.0, observables={"x": lambda a: a[:, 0]})
    print("obs hook:", o["path_integrals"]["x"].shape)
