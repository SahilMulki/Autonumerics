"""Stochastic Heun (explicit trapezoidal predictor-corrector) for the underdamped
Langevin oscillator

    dq = p dt
    dp = (-omega^2 q - gamma p) dt + sqrt(2 gamma / beta) dW

State columns are (q, p).  Noise is additive and enters p only, so the same
dW_n is used in predictor and corrector (Heun is then weak order 2).

Corrector: p_{n+1} is the Heun update; q_{n+1} = q_n + (h/2)(p_n + p_{n+1}) is the
trapezoid of the reported momenta (sequential / Gauss-Seidel form of the corrector).
"""
import numpy as np

Q0, P0 = 1.0, 0.0
OMEGA = 4.0
GAMMA = 1.0
BETA = 1.0
SIGMA = np.sqrt(2.0 * GAMMA / BETA)


def _drift(X):
    F = np.empty_like(X)
    F[:, 0] = X[:, 1]
    F[:, 1] = -OMEGA**2 * X[:, 0] - GAMMA * X[:, 1]
    return F


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))  # one block, path-major
    else:
        dW = np.asarray(dW, dtype=float)
        if dW.ndim == 3:                 # trailing singleton noise axis
            dW = dW.reshape(dW.shape[0], dW.shape[1])
        Nt = dW.shape[1]
        dt = T / Nt
        num_paths = dW.shape[0]

    X = np.empty((num_paths, 2), dtype=float)
    X[:, 0] = Q0
    X[:, 1] = P0
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    for n in range(Nt):
        X_prev = X
        dw = SIGMA * dW[:, n]
        f0 = _drift(X_prev)
        # predictor
        Xs = X_prev + f0 * dt
        Xs[:, 1] += dw
        # corrector, same increment: Heun update of the momentum ...
        f1 = _drift(Xs)
        X = X_prev + 0.5 * (f0 + f1) * dt
        X[:, 1] += dw
        # ... then q takes the trapezoid of the *reported* momenta p_n, p_{n+1}
        # (rather than of the predictor p*), so that q_{n+1} - q_n is exactly the
        # trapezoid quadrature of the p values the scheme returns.  This is what
        # Dynkin's identity for phi = q (generator = p) integrates.
        X[:, 0] = X_prev[:, 0] + 0.5 * (X_prev[:, 1] + X[:, 1]) * dt
        for k, phi in obs.items():
            acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt

    out = {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    res = solve_sde(num_paths=100000, dt=0.04, T=1.0, seed=42)
    print("Empirical mean (q, p):    ", res["empirical_mean"])
    print("Empirical variance (q, p):", res["empirical_variance"])
    print("cov(q,p):", float(np.cov(res["terminal_paths"].T)[0, 1]))
    print("dt used:", res["dt"])
    dWt = np.sqrt(0.04) * np.random.default_rng(0).standard_normal((1000, 25))
    print("dW hook dt:", solve_sde(1000, 0.04, 1.0, dW=dWt)["dt"])
    r = solve_sde(1000, 0.04, 1.0, observables={"q": lambda a: a[:, 0]})
    print("obs hook shape:", r["path_integrals"]["q"].shape)
